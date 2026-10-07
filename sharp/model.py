"""CNN + BiGRU with two heads: boundary probability and section type, per frame.

    log-mel (B, T, 80)
      -> 3 conv blocks (3x3 convs, pooling over frequency only, so time resolution is kept;
         80 mel bands -> 20 -> 5 -> 1)
         learn local sound patterns: timbre, drums, vocals on/off
      -> linear projection per frame
      -> 2-layer bidirectional GRU over the whole song
         learns how sections follow and repeat each other
      -> boundary head: P(a section starts at this frame)      (B, T)
      -> section head:  P(section type | frame), 10 classes     (B, T, 10)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn

from . import config
from .features import FeatureStats
from .labels import CLASSES, N_CLASSES


def _freq_pool(f: int) -> int:
    """Pooling factor for a block with f frequency rows: 4 when it divides f, else collapse small f."""
    if f % 4 == 0 and f > 4:
        return 4
    if f <= 5:
        return max(1, f)
    return 2 if f % 2 == 0 else 1


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, freq_pool: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(freq_pool, 1)),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def reverse_padded(h: torch.Tensor, lengths: Optional[torch.Tensor]) -> torch.Tensor:
    """Reverse each sequence in time within its own length; padding stays at the end."""
    if lengths is None:
        return h.flip(1)
    b, t, d = h.shape
    steps = torch.arange(t, device=h.device).unsqueeze(0).expand(b, t)
    lens = lengths.to(h.device).unsqueeze(1)
    idx = torch.where(steps < lens, lens - 1 - steps, steps)
    return h.gather(1, idx.unsqueeze(-1).expand(b, t, d))


class BiGRU(nn.Module):
    """Bidirectional multi-layer GRU that handles padded batches exactly.

    Padding sits at the end, so the forward direction never sees it before the
    real frames; the backward direction runs on each song reversed *within its
    own length*. This gives the same result as packed sequences but keeps
    PyTorch's fast fused GRU kernels (packing is ~6x slower on CPU).
    """

    def __init__(self, input_size: int, hidden: int, num_layers: int, dropout: float):
        super().__init__()
        self.fwd = nn.ModuleList()
        self.bwd = nn.ModuleList()
        for i in range(num_layers):
            d_in = input_size if i == 0 else 2 * hidden
            self.fwd.append(nn.GRU(d_in, hidden, batch_first=True))
            self.bwd.append(nn.GRU(d_in, hidden, batch_first=True))
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        for i, (gf, gb) in enumerate(zip(self.fwd, self.bwd)):
            if i > 0:
                h = self.dropout(h)
            out_f, _ = gf(h)
            out_b, _ = gb(reverse_padded(h, lengths))
            h = torch.cat([out_f, reverse_padded(out_b, lengths)], dim=-1)
        return h


class StructureNet(nn.Module):
    def __init__(self, n_bands: int = config.N_BANDS, n_classes: int = N_CLASSES,
                 conv_channels: Sequence[int] = (16, 32, 64), gru_hidden: int = 128,
                 gru_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hparams = {"n_bands": n_bands, "n_classes": n_classes,
                        "conv_channels": list(conv_channels), "gru_hidden": gru_hidden,
                        "gru_layers": gru_layers, "dropout": dropout}
        blocks = []
        c_in, f = 1, n_bands
        for c_out in conv_channels:
            pool = _freq_pool(f)
            blocks.append(ConvBlock(c_in, c_out, pool, dropout=dropout / 3))
            c_in, f = c_out, f // pool
        self.cnn = nn.Sequential(*blocks)
        self.proj = nn.Sequential(nn.Linear(c_in * f, gru_hidden), nn.ReLU(inplace=True), nn.Dropout(dropout))
        if gru_layers > 0:
            self.gru = BiGRU(gru_hidden, gru_hidden, gru_layers, dropout)
            out_dim = 2 * gru_hidden
        else:                                   # ablation: CNN only, no sequence model
            self.gru = None
            out_dim = gru_hidden
        self.head_dropout = nn.Dropout(dropout)
        self.boundary_head = nn.Linear(out_dim, 1)
        self.section_head = nn.Linear(out_dim, n_classes)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        """x: (B, T, F) -> boundary logits (B, T), section logits (B, T, C)."""
        b, t, _ = x.shape
        h = self.cnn(x.transpose(1, 2).unsqueeze(1))           # (B, C, F', T)
        h = h.permute(0, 3, 1, 2).reshape(b, t, -1)            # (B, T, C*F')
        h = self.proj(h)
        if self.gru is not None:
            if lengths is not None and int(lengths.min()) == t:
                lengths = None
            h = self.gru(h, lengths)
        h = self.head_dropout(h)
        return self.boundary_head(h).squeeze(-1), self.section_head(h)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def run_model(model: StructureNet, x: np.ndarray, device: torch.device | str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Standardised features (T, F) -> boundary prob (T,), class probs (T, C)."""
    model.eval()
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0).to(device)
    b_logit, s_logit = model(xt)
    return (torch.sigmoid(b_logit)[0].cpu().numpy(),
            torch.softmax(s_logit, dim=-1)[0].cpu().numpy())


def save_checkpoint(path: Path | str, model: StructureNet, stats: FeatureStats, extra: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "model_hparams": model.hparams,
        "stats": stats.to_dict(),
        "classes": list(CLASSES),
        "feature_params": {"n_bands": config.N_BANDS, "time_pool": config.TIME_POOL,
                           "sr": config.HARMONIX_SR, "hop": config.HARMONIX_HOP,
                           "n_mels": config.HARMONIX_N_MELS},
        **extra,
    }, path)


def load_checkpoint(path: Path | str, device: torch.device | str = "cpu") -> tuple[StructureNet, FeatureStats, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if list(ckpt.get("classes", CLASSES)) != list(CLASSES):
        raise ValueError("checkpoint was trained with a different class list")
    model = StructureNet(**ckpt["model_hparams"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, FeatureStats.from_dict(ckpt["stats"]), ckpt
