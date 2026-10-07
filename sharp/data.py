"""Feature cache and PyTorch dataset.

``python -m sharp.prepare`` turns each ~1.5 MB Harmonix mel file into a ~0.2 MB
float16 array of pooled log-mel features under ``data/processed/features``.
Training then loads the whole cache into memory (a few hundred MB).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import config
from .annotations import SongAnnotation
from .features import FeatureStats, make_targets, pool_mel
from .labels import IGNORE_INDEX, N_CLASSES

FEATURE_SUBDIR = "features"
INDEX_FILE = "index.csv"
STATS_FILE = "stats.npz"
_MEL_SUFFIXES = ("-mel", "_mel", "_full")


# --------------------------------------------------------------------------- #
# Locating the raw Harmonix mel-spectrograms
# --------------------------------------------------------------------------- #
def find_mel_files(mel_dir: Path | str = config.MEL_DIR) -> dict[str, Path]:
    """Map file id (e.g. '0001_12step') -> path of its '<id>-mel.npy' file."""
    mel_dir = Path(mel_dir)
    out: dict[str, Path] = {}
    if not mel_dir.exists():
        return out
    for p in sorted(mel_dir.rglob("*.npy")):
        if p.name.startswith("._"):          # macOS resource-fork junk inside tarballs
            continue
        stem = p.stem
        for suf in _MEL_SUFFIXES:
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
                break
        out[stem] = p
    return out


def check_mel_info(mel_dir: Path | str = config.MEL_DIR, strict: bool = False) -> Optional[dict]:
    """Read the info.json shipped with the mel archive and check it matches sharp/config.py.

    With ``strict=True`` a mismatch raises (used by ``prepare``), otherwise it warns.
    """
    infos = sorted(p for p in Path(mel_dir).rglob("info.json") if not p.name.startswith("._")) \
        if Path(mel_dir).exists() else []
    if not infos:
        return None
    info = json.loads(infos[0].read_text())
    expected = {"SR": config.HARMONIX_SR, "HOP_LENGTH": config.HARMONIX_HOP,
                "N_MELS": config.HARMONIX_N_MELS, "N_FFT": config.HARMONIX_N_FFT,
                "MEL_FMIN": config.HARMONIX_FMIN, "MEL_FMAX": config.HARMONIX_FMAX}
    problems = []
    for key, val in expected.items():
        if key not in info:
            continue
        got = info[key]
        if key == "MEL_FMAX" and got is None:
            got = info.get("SR", config.HARMONIX_SR) / 2.0       # librosa default: Nyquist
        if not np.isclose(float(got), float(val)):
            problems.append(f"{key}={info[key]} (sharp/config.py has {val})")
    if problems:
        msg = (f"{infos[0]} does not match the spectrogram settings in sharp/config.py: "
               + ", ".join(problems) + ". Update the HARMONIX_* constants in sharp/config.py.")
        if strict:
            raise ValueError(msg)
        warnings.warn(msg)
    return info


# --------------------------------------------------------------------------- #
# Feature cache
# --------------------------------------------------------------------------- #
def feature_path(cache_dir: Path | str, file_id: str) -> Path:
    return Path(cache_dir) / FEATURE_SUBDIR / f"{file_id}.npy"


def build_cache(file_ids: Sequence[str], mel_files: dict[str, Path],
                cache_dir: Path | str = config.CACHE_DIR, force: bool = False,
                verbose: bool = True) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    (cache_dir / FEATURE_SUBDIR).mkdir(parents=True, exist_ok=True)
    old_index = None
    if (cache_dir / INDEX_FILE).exists() and not force:
        old_index = load_index(cache_dir)
    records = []
    for i, fid in enumerate(file_ids):
        out = feature_path(cache_dir, fid)
        if out.exists() and not force:
            n_frames = int(np.load(out, mmap_mode="r").shape[0])
            if old_index is not None and fid in old_index.index:
                audio_sec = float(old_index.loc[fid, "audio_sec"])
            else:
                audio_sec = n_frames * config.FRAME_HOP_SEC
        else:
            mel = np.load(mel_files[fid])
            n_mel = mel.shape[1] if mel.shape[0] == config.HARMONIX_N_MELS else mel.shape[0]
            feats = pool_mel(mel)
            np.save(out, feats.astype(np.float16))
            n_frames = feats.shape[0]
            audio_sec = n_mel * config.HARMONIX_HOP / config.HARMONIX_SR
        records.append({"file_id": fid, "n_frames": n_frames, "audio_sec": audio_sec})
        if verbose and ((i + 1) % 50 == 0 or i + 1 == len(file_ids)):
            print(f"  cached {i + 1}/{len(file_ids)} songs")
    index = pd.DataFrame.from_records(records)
    index.to_csv(cache_dir / INDEX_FILE, index=False)
    return index


def load_index(cache_dir: Path | str = config.CACHE_DIR) -> pd.DataFrame:
    path = Path(cache_dir) / INDEX_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python -m sharp.prepare` first")
    return pd.read_csv(path, dtype={"file_id": str}).set_index("file_id", drop=False)


def load_features(cache_dir: Path | str, file_id: str) -> np.ndarray:
    return np.load(feature_path(cache_dir, file_id)).astype(np.float32)


def load_stats(cache_dir: Path | str = config.CACHE_DIR) -> FeatureStats:
    path = Path(cache_dir) / STATS_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python -m sharp.prepare` first")
    return FeatureStats.load(path)


def load_split(split: str, annotation_dir: Path | str = config.ANNOTATION_DIR,
               cache_dir: Path | str = config.CACHE_DIR, splits_dir: Path | str = config.SPLITS_DIR,
               limit: Optional[int] = None) -> tuple[list[str], dict[str, SongAnnotation], pd.DataFrame]:
    """Song ids of one split that have both cached features and annotations."""
    from .annotations import load_annotations
    from .splits import load_splits

    index = load_index(cache_dir)
    ids = [f for f in load_splits(splits_dir)[split] if f in index.index]
    anns = load_annotations(annotation_dir, ids)
    ids = [f for f in ids if f in anns]
    if limit:
        ids = ids[:limit]
    if not ids:
        raise RuntimeError(f"no usable songs in the {split!r} split - did `python -m sharp.prepare` run?")
    return ids, anns, index


# --------------------------------------------------------------------------- #
# PyTorch dataset
# --------------------------------------------------------------------------- #
class SongDataset(Dataset):
    """One item = one song: standardised features (T, F) + section/boundary targets."""

    def __init__(self, file_ids: Sequence[str], annotations: dict[str, SongAnnotation],
                 cache_dir: Path | str, stats: FeatureStats, train: bool = False,
                 max_frames: Optional[int] = None, spec_augment: bool = False,
                 boundary_sigma: float = 1.0, seed: int = 0):
        self.train = train
        self.max_frames = max_frames
        self.spec_augment = spec_augment
        self.rng = np.random.default_rng(seed)
        self.items = []
        for fid in file_ids:
            x = stats.apply(load_features(cache_dir, fid))
            section, boundary = make_targets(annotations[fid], x.shape[0], sigma=boundary_sigma)
            self.items.append({"file_id": fid, "x": x, "section": section, "boundary": boundary})

    def __len__(self) -> int:
        return len(self.items)

    def class_counts(self) -> np.ndarray:
        counts = np.zeros(N_CLASSES, dtype=np.int64)
        for it in self.items:
            s = it["section"]
            counts += np.bincount(s[s != IGNORE_INDEX], minlength=N_CLASSES)
        return counts

    def __getitem__(self, i: int) -> dict:
        it = self.items[i]
        x, section, boundary = it["x"], it["section"], it["boundary"]
        if self.train:
            if self.max_frames and x.shape[0] > self.max_frames:
                s = int(self.rng.integers(0, x.shape[0] - self.max_frames + 1))
                x = x[s:s + self.max_frames]
                section = section[s:s + self.max_frames]
                boundary = boundary[s:s + self.max_frames]
            if self.spec_augment:
                x = x.copy()
                for _ in range(2):                                  # frequency masks
                    w = int(self.rng.integers(0, 9))
                    f0 = int(self.rng.integers(0, max(1, x.shape[1] - w)))
                    x[:, f0:f0 + w] = 0.0
                x += np.float32(self.rng.normal(0.0, 0.2))           # random gain (standardised units)
        return {"file_id": it["file_id"],
                "x": torch.from_numpy(np.ascontiguousarray(x)),
                "section": torch.from_numpy(np.ascontiguousarray(section)),
                "boundary": torch.from_numpy(np.ascontiguousarray(boundary))}


def collate_songs(batch: list[dict]) -> dict:
    lengths = torch.tensor([b["x"].shape[0] for b in batch], dtype=torch.long)
    t_max = int(lengths.max())
    n_feat = batch[0]["x"].shape[1]
    x = torch.zeros(len(batch), t_max, n_feat)
    section = torch.full((len(batch), t_max), IGNORE_INDEX, dtype=torch.long)
    boundary = torch.zeros(len(batch), t_max)
    mask = torch.zeros(len(batch), t_max, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b["x"].shape[0]
        x[i, :n] = b["x"]
        section[i, :n] = b["section"]
        boundary[i, :n] = b["boundary"]
        mask[i, :n] = True
    return {"x": x, "section": section, "boundary": boundary, "mask": mask,
            "lengths": lengths, "file_ids": [b["file_id"] for b in batch]}


def class_weights_from_counts(counts: np.ndarray) -> np.ndarray:
    """Inverse-square-root frequency weights, normalised to mean 1 over observed classes."""
    counts = np.asarray(counts, dtype=np.float64)
    w = np.zeros_like(counts)
    seen = counts > 0
    w[seen] = 1.0 / np.sqrt(counts[seen])
    w[seen] /= w[seen].mean()
    return w.astype(np.float32)
