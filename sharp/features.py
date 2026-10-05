"""Model input features and frame-level training targets.

Pipeline per song:  Harmonix mel (80 bands x ~21.5 fps, power)
    -> average 4-frame blocks (~5.4 fps, 0.186 s)           [power domain]
    -> dB with an 80 dB floor                                [pool_mel]
    -> per-band standardisation with training-set stats      [FeatureStats]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from . import config
from .annotations import SongAnnotation
from .labels import IGNORE_INDEX, to_index


def pool_mel(mel: np.ndarray, n_bands: int = config.N_BANDS, time_pool: int = config.TIME_POOL,
             n_mels: int = config.HARMONIX_N_MELS) -> np.ndarray:
    """Harmonix mel-spectrogram -> (frames, n_bands) float32 log-mel in dB."""
    mel = np.asarray(mel, dtype=np.float64)
    if mel.ndim != 2:
        raise ValueError(f"expected a 2-D mel-spectrogram, got shape {mel.shape}")
    if mel.shape[0] != n_mels and mel.shape[1] == n_mels:
        mel = mel.T                                   # stored as (time, mels)
    if mel.shape[0] != n_mels:
        raise ValueError(f"expected {n_mels} mel bands, got shape {mel.shape}")
    already_db = bool(np.nanmin(mel) < 0)            # tolerate pre-computed dB input
    mel = np.nan_to_num(mel, nan=0.0 if not already_db else -config.TOP_DB)

    edges = np.linspace(0, n_mels, n_bands + 1).round().astype(int)
    banded = np.add.reduceat(mel, edges[:-1], axis=0) / np.diff(edges)[:, None]

    n = banded.shape[1]
    starts = np.arange(0, n, time_pool)
    counts = np.diff(np.append(starts, n))
    pooled = np.add.reduceat(banded, starts, axis=1) / counts[None, :]

    db = pooled if already_db else 10.0 * np.log10(np.maximum(pooled, config.AMIN))
    db = np.maximum(db, db.max() - config.TOP_DB)
    return db.T.astype(np.float32)


def frame_times(n_frames: int, time_pool: int = config.TIME_POOL, hop: int = config.HARMONIX_HOP,
                sr: int = config.HARMONIX_SR) -> np.ndarray:
    """Centre time (s) of each pooled frame (mel frames are centred, librosa center=True)."""
    return (np.arange(n_frames) * time_pool + (time_pool - 1) / 2.0) * hop / sr


def times_to_frames(times: np.ndarray, time_pool: int = config.TIME_POOL,
                    hop: int = config.HARMONIX_HOP, sr: int = config.HARMONIX_SR) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64)
    return np.round((times * sr / hop - (time_pool - 1) / 2.0) / time_pool).astype(int)


def mel_frames_to_seconds(n_mel_frames: int, hop: int = config.HARMONIX_HOP,
                          sr: int = config.HARMONIX_SR) -> float:
    return n_mel_frames * hop / sr


@dataclass
class FeatureStats:
    mean: np.ndarray
    std: np.ndarray

    def apply(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": [float(v) for v in self.mean], "std": [float(v) for v in self.std]}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureStats":
        return cls(np.asarray(d["mean"], dtype=np.float32), np.asarray(d["std"], dtype=np.float32))

    def save(self, path: Path | str) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: Path | str) -> "FeatureStats":
        with np.load(path) as z:
            return cls(z["mean"].astype(np.float32), z["std"].astype(np.float32))


def compute_stats(arrays: Iterable[np.ndarray]) -> FeatureStats:
    total = None
    total_sq = None
    count = 0
    for x in arrays:
        x = np.asarray(x, dtype=np.float64)
        if total is None:
            total = np.zeros(x.shape[1])
            total_sq = np.zeros(x.shape[1])
        total += x.sum(axis=0)
        total_sq += (x ** 2).sum(axis=0)
        count += x.shape[0]
    if not count:
        raise ValueError("no frames to compute statistics from")
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean ** 2, 1e-8))
    return FeatureStats(mean.astype(np.float32), std.astype(np.float32))


def make_targets(ann: SongAnnotation, n_frames: int, sigma: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Frame-level targets.

    Returns
    -------
    section : (n_frames,) int64 class index, IGNORE_INDEX outside [0, end) or for unmapped labels
    boundary : (n_frames,) float32 in [0, 1], Gaussian bumps (std = sigma frames) at each boundary
    """
    t = frame_times(n_frames)
    section = np.full(n_frames, IGNORE_INDEX, dtype=np.int64)
    seg_idx = np.searchsorted(ann.starts, t, side="right") - 1
    valid = (seg_idx >= 0) & (t < ann.end)
    cls_idx = np.array([to_index(l) for l in ann.raw_labels], dtype=np.int64)
    section[valid] = cls_idx[seg_idx[valid]]

    boundary = np.zeros(n_frames, dtype=np.float32)
    frames = times_to_frames(ann.boundary_times)
    frames = frames[(frames >= 0) & (frames < n_frames)]
    if sigma <= 0:
        boundary[frames] = 1.0
        return section, boundary
    radius = int(np.ceil(3 * sigma))
    offsets = np.arange(-radius, radius + 1)
    bump = np.exp(-0.5 * (offsets / sigma) ** 2).astype(np.float32)
    for f in frames:
        idx = f + offsets
        keep = (idx >= 0) & (idx < n_frames)
        boundary[idx[keep]] = np.maximum(boundary[idx[keep]], bump[keep])
    return section, boundary
