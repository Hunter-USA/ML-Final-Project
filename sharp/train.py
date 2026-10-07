"""Train the CNN + BiGRU.

    python -m sharp.train                       # defaults from sharp/config.py:TrainConfig
    python -m sharp.train --epochs 60 --lr 5e-4 --run-name cnn_bigru_lr5e-4
    python -m sharp.train --device cuda

Loss = weighted cross-entropy on section type (ignoring unlabelled frames)
     + lambda * BCE on the smeared boundary targets (positives up-weighted).
Model selection: lowest validation loss (early stopping). Afterwards the
peak-picking threshold / minimum boundary spacing are tuned on the validation
split and stored in the checkpoint.

Writes runs/<run-name>/{best.pt, history.csv, config.json, decode_tuning.csv}.

The best checkpoint is saved after every improving epoch, so you can stop at any
time with Ctrl+C: training ends cleanly and the tuning step still runs. If the
process was killed some other way, finish with
    python -m sharp.train --tune-only
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import config
from .cli import add_dataclass_args, add_path_args, dataclass_from_args, set_seed
from .config import TrainConfig
from .data import SongDataset, class_weights_from_counts, collate_songs, load_split, load_stats
from .labels import CLASSES, IGNORE_INDEX, N_CLASSES
from .metrics import tune_decode_params
from .model import StructureNet, count_parameters, load_checkpoint, resolve_device, run_model, save_checkpoint
from .postprocess import DecodeParams


def compute_loss(b_logit: torch.Tensor, s_logit: torch.Tensor, batch: dict, class_w: torch.Tensor,
                 cfg: TrainConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = batch["mask"].float()
    bce = F.binary_cross_entropy_with_logits(
        b_logit, batch["boundary"], reduction="none",
        pos_weight=torch.tensor(cfg.boundary_pos_weight, device=b_logit.device))
    loss_b = (bce * mask).sum() / mask.sum().clamp(min=1.0)
    target = batch["section"].reshape(-1)
    if (target != IGNORE_INDEX).any():
        loss_s = F.cross_entropy(s_logit.reshape(-1, s_logit.shape[-1]), target, weight=class_w,
                                 ignore_index=IGNORE_INDEX, label_smoothing=cfg.label_smoothing)
    else:
        loss_s = s_logit.sum() * 0.0
    return loss_s + cfg.boundary_weight * loss_b, loss_s, loss_b


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) and k != "lengths" else v) for k, v in batch.items()}


@torch.no_grad()
def validate(model: StructureNet, loader: DataLoader, class_w: torch.Tensor, cfg: TrainConfig,
             device: torch.device) -> dict:
    model.eval()
    tot = {"loss": 0.0, "loss_section": 0.0, "loss_boundary": 0.0}
    n_batches, correct, counted = 0, 0, 0
    for batch in loader:
        batch = _to_device(batch, device)
        b_logit, s_logit = model(batch["x"], batch["lengths"])
        loss, ls, lb = compute_loss(b_logit, s_logit, batch, class_w, cfg)
        tot["loss"] += loss.item()
        tot["loss_section"] += ls.item()
        tot["loss_boundary"] += lb.item()
        n_batches += 1
        valid = batch["section"] != IGNORE_INDEX
        correct += (s_logit.argmax(-1)[valid] == batch["section"][valid]).sum().item()
        counted += valid.sum().item()
    out = {k: v / max(n_batches, 1) for k, v in tot.items()}
    out["frame_acc"] = correct / max(counted, 1)
    return out


def train(cfg: TrainConfig, annotation_dir: Path = config.ANNOTATION_DIR, cache_dir: Path = config.CACHE_DIR,
          splits_dir: Path = config.SPLITS_DIR, runs_dir: Path = config.RUNS_DIR,
          limit: Optional[int] = None, verbose: bool = True) -> Path:
    set_seed(cfg.seed)
    if cfg.num_threads:
        torch.set_num_threads(cfg.num_threads)
    device = resolve_device(cfg.device)
    log = print if verbose else (lambda *a, **k: None)

    stats = load_stats(cache_dir)
    train_ids, train_anns, _ = load_split("train", annotation_dir, cache_dir, splits_dir, limit)
    val_ids, val_anns, index = load_split("val", annotation_dir, cache_dir, splits_dir, limit)
    train_ds = SongDataset(train_ids, train_anns, cache_dir, stats, train=True, max_frames=cfg.max_frames,
                           spec_augment=cfg.spec_augment, boundary_sigma=cfg.boundary_sigma, seed=cfg.seed)
    val_ds = SongDataset(val_ids, val_anns, cache_dir, stats, boundary_sigma=cfg.boundary_sigma)

    counts = train_ds.class_counts()
    class_w = torch.tensor(class_weights_from_counts(counts), device=device)
    log(f"device={device}  train songs={len(train_ds)}  val songs={len(val_ds)}")
    log("train frames per class: " + ", ".join(f"{c}={n}" for c, n in zip(CLASSES, counts)))

    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_songs,
                        generator=torch.Generator().manual_seed(cfg.seed))
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_songs)

    model = StructureNet(n_bands=config.N_BANDS, n_classes=N_CLASSES, conv_channels=cfg.conv_channels,
                         gru_hidden=cfg.gru_hidden, gru_layers=cfg.gru_layers, dropout=cfg.dropout).to(device)
    log(f"StructureNet: {count_parameters(model):,} parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

    run_dir = Path(runs_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    best_path = run_dir / "best.pt"

    best_loss, bad_epochs, history = float("inf"), 0, []
    try:
        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            model.train()
            sums, n = np.zeros(3), 0
            for batch in loader:
                batch = _to_device(batch, device)
                b_logit, s_logit = model(batch["x"], batch["lengths"])
                loss, ls, lb = compute_loss(b_logit, s_logit, batch, class_w, cfg)
                opt.zero_grad()
                loss.backward()
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                sums += [loss.item(), ls.item(), lb.item()]
                n += 1
            tr = sums / max(n, 1)
            val = validate(model, val_loader, class_w, cfg, device)
            sched.step(val["loss"])
            row = {"epoch": epoch, "train_loss": tr[0], "train_loss_section": tr[1], "train_loss_boundary": tr[2],
                   "val_loss": val["loss"], "val_loss_section": val["loss_section"],
                   "val_loss_boundary": val["loss_boundary"], "val_frame_acc": val["frame_acc"],
                   "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0}
            history.append(row)
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
            improved = val["loss"] < best_loss - 1e-4
            log(f"epoch {epoch:3d} | train {tr[0]:.4f} | val {val['loss']:.4f} "
                f"(sec {val['loss_section']:.3f}, bnd {val['loss_boundary']:.3f}) | "
                f"val frame acc {val['frame_acc']:.3f} | {row['seconds']:.0f}s" + ("  *" if improved else ""))
            if improved:
                best_loss, bad_epochs = val["loss"], 0
                save_checkpoint(best_path, model, stats, {
                    "train_config": cfg.to_dict(), "epoch": epoch, "val_loss": val["loss"],
                    "val_frame_acc": val["frame_acc"], "decode": DecodeParams().to_dict()})
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    log(f"early stopping: no improvement for {cfg.patience} epochs")
                    break
    except KeyboardInterrupt:
        log("\nstopped by Ctrl+C - keeping the best checkpoint so far and tuning it")
    if not best_path.exists():
        raise SystemExit("no epoch finished, so there is no checkpoint to keep")

    tune_checkpoint(best_path, annotation_dir, cache_dir, splits_dir, cfg.device, limit, verbose)
    return best_path


def tune_checkpoint(best_path: Path, annotation_dir: Path = config.ANNOTATION_DIR,
                    cache_dir: Path = config.CACHE_DIR, splits_dir: Path = config.SPLITS_DIR,
                    device: str = "auto", limit: Optional[int] = None, verbose: bool = True) -> None:
    """Tune peak picking (threshold, min boundary spacing) on the validation split; store it in the checkpoint."""
    log = print if verbose else (lambda *a, **k: None)
    best_path = Path(best_path)
    if not best_path.exists():
        raise SystemExit(f"{best_path} not found - train a model first")
    dev = resolve_device(device)
    model, stats, ckpt = load_checkpoint(best_path, dev)
    val_ids, val_anns, index = load_split("val", annotation_dir, cache_dir, splits_dir, limit)
    val_ds = SongDataset(val_ids, val_anns, cache_dir, stats)
    log(f"tuning peak picking on {len(val_ds)} validation songs...")
    val_songs = []
    for item in val_ds.items:
        bprob, _ = run_model(model, item["x"], dev)
        val_songs.append((val_anns[item["file_id"]], bprob, float(index.loc[item["file_id"], "audio_sec"])))
    params, table = tune_decode_params(val_songs)
    table.to_csv(best_path.parent / "decode_tuning.csv", index=False)
    ckpt["decode"] = params.to_dict()
    ckpt["val_boundary_score"] = float(table["score"].max())
    torch.save(ckpt, best_path)
    log(f"best epoch {ckpt['epoch']} (val loss {ckpt['val_loss']:.4f}); tuned peak picking: "
        f"threshold={params.threshold:.2f}, min distance={params.min_distance_sec:.0f}s "
        f"(val mean HR0.5F/HR3F = {ckpt['val_boundary_score']:.3f})")
    log(f"saved {best_path}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_args(p, "annotation_dir", "cache_dir", "splits_dir", "runs_dir")
    add_dataclass_args(p, TrainConfig)
    p.add_argument("--limit-songs", type=int, default=None, help="debug: use only the first N songs per split")
    p.add_argument("--tune-only", action="store_true",
                   help="skip training; just tune peak picking for runs/<run-name>/best.pt "
                        "(use after a run was stopped before it finished)")
    args = p.parse_args(argv)
    cfg = dataclass_from_args(TrainConfig, args)
    if args.tune_only:
        tune_checkpoint(Path(args.runs_dir) / cfg.run_name / "best.pt", args.annotation_dir, args.cache_dir,
                        args.splits_dir, cfg.device, args.limit_songs)
        return
    train(cfg, args.annotation_dir, args.cache_dir, args.splits_dir, args.runs_dir, args.limit_songs)


if __name__ == "__main__":
    main()
