"""Song-level train/val/test splits.

Harmonix contains several versions of the same song (radio edit, extended mix,
remixes, e.g. "Poker Face" x3, "Dynamite" x3). Putting one version in train
and another in test would leak, so songs are grouped by normalised title and
every group lands in exactly one split.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

SPLIT_NAMES = ("train", "val", "test")


def title_group_key(title: str, file_id: str = "") -> str:
    t = str(title).lower()
    base = re.split(r"[\(\[]| - | feat\.? | ft\.? ", t)[0]
    key = re.sub(r"[^a-z0-9]", "", base)
    if not key:
        key = re.sub(r"[^a-z0-9]", "", t)
    return key or file_id


def make_splits(metadata: pd.DataFrame, seed: int = config.SPLIT_SEED,
                fractions: tuple[float, float, float] = config.SPLIT_FRACTIONS) -> dict[str, list[str]]:
    files = metadata["File"].astype(str).tolist()
    titles = metadata["Title"].tolist() if "Title" in metadata else files
    groups: dict[str, list[str]] = {}
    for fid, title in zip(files, titles):
        groups.setdefault(title_group_key(title, fid), []).append(fid)

    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)

    n = len(files)
    cut_train = fractions[0] * n
    cut_val = (fractions[0] + fractions[1]) * n
    out: dict[str, list[str]] = {s: [] for s in SPLIT_NAMES}
    count = 0
    for k in keys:
        split = "train" if count < cut_train else ("val" if count < cut_val else "test")
        out[split].extend(groups[k])
        count += len(groups[k])
    return {s: sorted(v) for s, v in out.items()}


def save_splits(splits: dict[str, list[str]], splits_dir: Path | str = config.SPLITS_DIR) -> None:
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        (splits_dir / f"{name}.txt").write_text("\n".join(splits[name]) + "\n", encoding="utf-8")


def load_splits(splits_dir: Path | str = config.SPLITS_DIR) -> dict[str, list[str]]:
    splits_dir = Path(splits_dir)
    out = {}
    for name in SPLIT_NAMES:
        path = splits_dir / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run `python -m sharp.prepare` first")
        out[name] = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return out


def splits_exist(splits_dir: Path | str = config.SPLITS_DIR) -> bool:
    return all((Path(splits_dir) / f"{n}.txt").exists() for n in SPLIT_NAMES)
