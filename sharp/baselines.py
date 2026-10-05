"""Train and tune the baselines from the pitch (plus one classic unsupervised method).

    python -m sharp.baselines                  # all of them
    python -m sharp.baselines --only logreg mlp

* majority  - every frame is the most common training class, no boundaries
* logreg    - logistic regression on per-frame context features
              (softmax regression for section type + binary logistic regression for boundaries)
* mlp       - 2-hidden-layer MLP on the same features, same two outputs
* novelty   - Foote checkerboard novelty on a self-similarity matrix (SciPy/NumPy, no learning)
              for boundaries, majority class for labels

The frame baselines see ~1.5 s of context around each frame; the CNN + BiGRU sees
the whole song. Each method's peak-picking parameters are tuned on the
validation split exactly like the main model's.

Writes runs/baselines/{majority.json, novelty.json, logreg.pt, mlp.pt}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from . import config
from .cli import add_path_args, set_seed
from .data import SongDataset, class_weights_from_counts, load_split, load_stats
from .labels import CLASSES, IGNORE_INDEX, N_CLASSES
from .metrics import tune_decode_params
from .model import resolve_device
from .postprocess import DecodeParams
from .predictors import (FrameModel, FrameModelPredictor, MajorityPredictor, NoveltyPredictor,
                         Predictor, context_features)

FRAME_METHODS = {"logreg": (), "mlp": (256, 128)}


def frame_arrays(ds: SongDataset, stride: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs, ys, bs = [], [], []
    for it in ds.items:
        f = context_features(it["x"])
        idx = np.arange(int(rng.integers(0, stride)), f.shape[0], stride)
        xs.append(f[idx])
        ys.append(it["section"][idx])
        bs.append(it["boundary"][idx])
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(bs)


def train_frame_model(hidden: tuple, train: tuple, val: tuple, class_w: np.ndarray, device: torch.device,
                      epochs: int = 20, lr: float = 1e-3, batch_size: int = 2048, pos_weight: float = 5.0,
                      patience: int = 4, seed: int = 0, verbose: bool = True) -> FrameModel:
    torch.manual_seed(seed)
    xtr, ytr, btr = (torch.from_numpy(a) for a in train)
    xva, yva, bva = (torch.from_numpy(a).to(device) for a in val)
    model = FrameModel(xtr.shape[1], N_CLASSES, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    cw = torch.tensor(class_w, device=device)
    pw = torch.tensor(pos_weight, device=device)

    def loss_fn(b_logit, s_logit, y, bnd):
        ls = F.cross_entropy(s_logit, y, weight=cw, ignore_index=IGNORE_INDEX) \
            if (y != IGNORE_INDEX).any() else s_logit.sum() * 0
        lb = F.binary_cross_entropy_with_logits(b_logit, bnd, pos_weight=pw)
        return ls + lb

    best, best_state, bad = float("inf"), None, 0
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr), generator=gen)
        for i in range(0, len(perm), batch_size):
            j = perm[i:i + batch_size]
            b_logit, s_logit = model(xtr[j].to(device))
            loss = loss_fn(b_logit, s_logit, ytr[j].to(device), btr[j].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = 0.0
            for i in range(0, len(xva), 8192):
                b_logit, s_logit = model(xva[i:i + 8192])
                vl += loss_fn(b_logit, s_logit, yva[i:i + 8192], bva[i:i + 8192]).item() * len(b_logit)
            vl /= len(xva)
        if verbose:
            print(f"    epoch {epoch:2d}  val loss {vl:.4f}")
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model.eval()


def tune(predictor: Predictor, val_ds: SongDataset, val_anns: dict, index) -> DecodeParams:
    songs = [(val_anns[it["file_id"]], predictor.predict(it["x"])[0], float(index.loc[it["file_id"], "audio_sec"]))
             for it in val_ds.items]
    params, table = tune_decode_params(songs)
    print(f"  {predictor.name}: threshold={params.threshold:.2f}, min distance={params.min_distance_sec:.0f}s "
          f"(val mean HR0.5F/HR3F = {table['score'].max():.3f})")
    return params


def run(only=("majority", "novelty", "logreg", "mlp"), annotation_dir: Path = config.ANNOTATION_DIR,
        cache_dir: Path = config.CACHE_DIR, splits_dir: Path = config.SPLITS_DIR,
        runs_dir: Path = config.RUNS_DIR, epochs: int = 20, stride: int = 3, device: str = "auto",
        seed: int = config.SPLIT_SEED, limit: Optional[int] = None, verbose: bool = True) -> None:
    set_seed(seed)
    dev = resolve_device(device)
    out_dir = Path(runs_dir) / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = load_stats(cache_dir)
    train_ids, train_anns, _ = load_split("train", annotation_dir, cache_dir, splits_dir, limit)
    val_ids, val_anns, index = load_split("val", annotation_dir, cache_dir, splits_dir, limit)
    train_ds = SongDataset(train_ids, train_anns, cache_dir, stats)
    val_ds = SongDataset(val_ids, val_anns, cache_dir, stats)
    counts = train_ds.class_counts()
    majority = CLASSES[int(np.argmax(counts))]
    print(f"majority class in train: {majority} ({counts.max() / counts.sum():.1%} of labelled frames)")

    if "majority" in only:
        (out_dir / "majority.json").write_text(json.dumps({"class": majority}, indent=2))
        print("  majority: saved")
    if "novelty" in only:
        nov = NoveltyPredictor(majority)
        params = tune(nov, val_ds, val_anns, index)
        (out_dir / "novelty.json").write_text(json.dumps(
            {"class": majority, "half_width": nov.half_width, "decode": params.to_dict()}, indent=2))

    frame_methods = [m for m in ("logreg", "mlp") if m in only]
    if frame_methods:
        print(f"building frame features (1 in every {stride} frames)...")
        tr = frame_arrays(train_ds, stride, seed)
        va = frame_arrays(val_ds, stride, seed + 1)
        class_w = class_weights_from_counts(np.bincount(tr[1][tr[1] != IGNORE_INDEX], minlength=N_CLASSES))
        print(f"  {len(tr[0]):,} train frames x {tr[0].shape[1]} features")
        for m in frame_methods:
            print(f"training {m}...")
            model = train_frame_model(FRAME_METHODS[m], tr, va, class_w, dev, epochs=epochs, seed=seed,
                                      verbose=verbose)
            pred = FrameModelPredictor(m, model, DecodeParams(), dev)
            pred.params = tune(pred, val_ds, val_anns, index)
            pred.save(out_dir / f"{m}.pt", stats.to_dict())
    print(f"saved baselines to {out_dir}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_args(p, "annotation_dir", "cache_dir", "splits_dir", "runs_dir")
    p.add_argument("--only", nargs="+", default=["majority", "novelty", "logreg", "mlp"],
                   choices=["majority", "novelty", "logreg", "mlp"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--stride", type=int, default=3, help="use every Nth frame for training the frame models")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    p.add_argument("--limit-songs", type=int, default=None)
    args = p.parse_args(argv)
    run(tuple(args.only), args.annotation_dir, args.cache_dir, args.splits_dir, args.runs_dir,
        args.epochs, args.stride, args.device, args.seed, args.limit_songs)


if __name__ == "__main__":
    main()
