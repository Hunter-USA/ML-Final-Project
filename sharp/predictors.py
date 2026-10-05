"""A common interface over the main model and every baseline.

Every predictor maps standardised features (T, F) to
    boundary probability (T,)  and  class probabilities (T, C)
and carries the peak-picking parameters tuned for it on the validation split,
so evaluation treats all methods identically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
from torch import nn

from . import config
from .features import FeatureStats
from .labels import CLASS_TO_INDEX, N_CLASSES
from .model import load_checkpoint, run_model
from .postprocess import DecodeParams

CONTEXT_FRAMES = 8          # +-8 frames (~1.5 s) of context for the frame-level baselines
NOVELTY_HALF_WIDTH = 32     # Foote kernel half-width in frames (~6 s)


class Predictor:
    name = "predictor"
    params = DecodeParams()
    stats: Optional[FeatureStats] = None     # feature standardisation the model was trained with

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Main model
# --------------------------------------------------------------------------- #
class StructureNetPredictor(Predictor):
    name = "cnn_bigru"

    def __init__(self, checkpoint: Path | str, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.model, self.stats, ckpt = load_checkpoint(checkpoint, self.device)
        self.params = DecodeParams.from_dict(ckpt.get("decode"))

    def predict(self, x: np.ndarray):
        return run_model(self.model, x, self.device)


# --------------------------------------------------------------------------- #
# Frame-level baselines: logistic regression and MLP
# --------------------------------------------------------------------------- #
def context_features(x: np.ndarray, context: int = CONTEXT_FRAMES) -> np.ndarray:
    """Per-frame features for models that see one frame at a time.

    [frame, mean over +-context, (mean of next context) - (mean of previous context), relative position]
    The difference term is a cheap "did the sound just change?" cue for boundaries;
    relative position lets the baselines learn that intros come first and outros last.
    """
    t, f = x.shape
    c = context
    pad = np.pad(x.astype(np.float64), ((c, c), (0, 0)), mode="edge")
    cs = np.concatenate([np.zeros((1, f)), np.cumsum(pad, axis=0)], axis=0)
    idx = np.arange(t)
    past = (cs[idx + c] - cs[idx]) / c
    future = (cs[idx + 2 * c] - cs[idx + c]) / c
    local = (cs[idx + 2 * c + 1] - cs[idx]) / (2 * c + 1)
    pos = ((idx + 0.5) / t)[:, None]
    return np.concatenate([x, local, future - past, pos], axis=1).astype(np.float32)


class FrameModel(nn.Module):
    """hidden=() is multinomial + binary logistic regression; hidden=(256, 128) is an MLP."""

    def __init__(self, in_dim: int, n_classes: int = N_CLASSES, hidden: tuple = (), dropout: float = 0.2):
        super().__init__()
        self.hparams = {"in_dim": in_dim, "n_classes": n_classes, "hidden": list(hidden), "dropout": dropout}
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.boundary_head = nn.Linear(d, 1)
        self.section_head = nn.Linear(d, n_classes)

    def forward(self, f: torch.Tensor):
        h = self.trunk(f)
        return self.boundary_head(h).squeeze(-1), self.section_head(h)


class FrameModelPredictor(Predictor):
    def __init__(self, name: str, model: FrameModel, params: DecodeParams, device: str | torch.device = "cpu",
                 stats: Optional[FeatureStats] = None):
        self.name = name
        self.model = model.to(device).eval()
        self.params = params
        self.device = torch.device(device)
        self.stats = stats

    @torch.no_grad()
    def predict(self, x: np.ndarray):
        f = torch.from_numpy(context_features(x)).to(self.device)
        b, s = self.model(f)
        return torch.sigmoid(b).cpu().numpy(), torch.softmax(s, dim=-1).cpu().numpy()

    def save(self, path: Path | str, stats_dict: dict) -> None:
        torch.save({"name": self.name, "model_state": self.model.state_dict(),
                    "model_hparams": self.model.hparams, "decode": self.params.to_dict(),
                    "stats": stats_dict}, path)

    @classmethod
    def load(cls, path: Path | str, device: str | torch.device = "cpu") -> "FrameModelPredictor":
        ckpt = torch.load(path, map_location=device, weights_only=True)
        hp = dict(ckpt["model_hparams"])
        hp["hidden"] = tuple(hp["hidden"])
        model = FrameModel(**hp)
        model.load_state_dict(ckpt["model_state"])
        stats = FeatureStats.from_dict(ckpt["stats"]) if ckpt.get("stats") else None
        return cls(ckpt["name"], model, DecodeParams.from_dict(ckpt["decode"]), device, stats)


# --------------------------------------------------------------------------- #
# Majority class and Foote novelty (unsupervised boundaries)
# --------------------------------------------------------------------------- #
def one_hot_probs(n_frames: int, class_name: str) -> np.ndarray:
    p = np.zeros((n_frames, N_CLASSES), dtype=np.float32)
    p[:, CLASS_TO_INDEX[class_name]] = 1.0
    return p


class MajorityPredictor(Predictor):
    """Always the most common training class; never predicts a boundary (one segment per song)."""
    name = "majority"

    def __init__(self, class_name: str):
        self.class_name = class_name
        self.params = DecodeParams(threshold=1.1)

    def predict(self, x: np.ndarray):
        return np.zeros(x.shape[0], dtype=np.float32), one_hot_probs(x.shape[0], self.class_name)


def foote_novelty(x: np.ndarray, half_width: int = NOVELTY_HALF_WIDTH, smooth: int = 2) -> np.ndarray:
    """Checkerboard-kernel novelty on a cosine self-similarity matrix (Foote, 2000), scaled to [0, 1]."""
    xs = uniform_filter1d(x.astype(np.float64), size=2 * smooth + 1, axis=0, mode="nearest")
    xs = xs - xs.mean(axis=0, keepdims=True)
    xn = xs / (np.linalg.norm(xs, axis=1, keepdims=True) + 1e-8)
    ssm = xn @ xn.T
    offsets = np.arange(-half_width, half_width) + 0.5
    taper = np.exp(-0.5 * (offsets / (0.5 * half_width)) ** 2) * np.sign(offsets)
    kernel = np.outer(taper, taper)
    padded = np.pad(ssm, half_width, mode="constant")
    w = 2 * half_width
    nov = np.array([np.sum(kernel * padded[i:i + w, i:i + w]) for i in range(ssm.shape[0])])
    nov = np.maximum(nov, 0.0)
    return (nov / nov.max() if nov.max() > 0 else nov).astype(np.float32)


class NoveltyPredictor(Predictor):
    """Classic signal-processing boundaries (no learning) + majority-class labels."""
    name = "novelty"

    def __init__(self, class_name: str, params: Optional[DecodeParams] = None,
                 half_width: int = NOVELTY_HALF_WIDTH):
        self.class_name = class_name
        self.params = params or DecodeParams(0.3, 6.0)
        self.half_width = half_width

    def predict(self, x: np.ndarray):
        return foote_novelty(x, self.half_width), one_hot_probs(x.shape[0], self.class_name)


# --------------------------------------------------------------------------- #
# Loading whatever has been trained
# --------------------------------------------------------------------------- #
ALL_METHODS = ("cnn_bigru", "mlp", "logreg", "novelty", "majority")


def load_predictors(runs_dir: Path | str = config.RUNS_DIR, checkpoint: Optional[Path | str] = None,
                    device: str | torch.device = "cpu", methods=ALL_METHODS) -> list[Predictor]:
    runs_dir = Path(runs_dir)
    base = runs_dir / "baselines"
    out: list[Predictor] = []
    for m in methods:
        if m == "cnn_bigru":
            ckpt = Path(checkpoint) if checkpoint else runs_dir / "cnn_bigru" / "best.pt"
            if ckpt.exists():
                out.append(StructureNetPredictor(ckpt, device))
            else:
                print(f"[skip] cnn_bigru: {ckpt} not found (run `python -m sharp.train`)")
        elif m in ("mlp", "logreg"):
            path = base / f"{m}.pt"
            if path.exists():
                out.append(FrameModelPredictor.load(path, device))
            else:
                print(f"[skip] {m}: {path} not found (run `python -m sharp.baselines`)")
        elif m in ("majority", "novelty"):
            path = base / f"{m}.json"
            if not path.exists():
                print(f"[skip] {m}: {path} not found (run `python -m sharp.baselines`)")
                continue
            info = json.loads(path.read_text())
            if m == "majority":
                out.append(MajorityPredictor(info["class"]))
            else:
                out.append(NoveltyPredictor(info["class"], DecodeParams.from_dict(info["decode"]),
                                            info.get("half_width", NOVELTY_HALF_WIDTH)))
        else:
            raise ValueError(f"unknown method {m!r}; choose from {ALL_METHODS}")
    return out
