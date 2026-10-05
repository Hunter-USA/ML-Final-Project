"""Turn frame-wise predictions into a labelled segmentation.

1. Boundaries: peaks of the boundary-probability curve above ``threshold`` and at
   least ``min_distance_sec`` apart (scipy.signal.find_peaks). Both values are
   tuned on the validation split.
2. Labels: each resulting segment gets the class with the highest *average*
   probability over its frames (a vote that smooths frame-level flicker).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from . import config
from .features import frame_times
from .labels import CLASSES


@dataclass
class DecodeParams:
    threshold: float = 0.5
    min_distance_sec: float = 3.0

    def to_dict(self) -> dict:
        return {"threshold": float(self.threshold), "min_distance_sec": float(self.min_distance_sec)}

    @classmethod
    def from_dict(cls, d: dict | None) -> "DecodeParams":
        return cls(**d) if d else cls()


def pick_peaks(prob: np.ndarray, threshold: float, min_distance_sec: float,
               frame_hop: float = config.FRAME_HOP_SEC) -> np.ndarray:
    distance = max(1, int(round(min_distance_sec / frame_hop)))
    peaks, _ = find_peaks(np.asarray(prob, dtype=np.float64), height=threshold, distance=distance)
    return peaks


def boundaries_from_prob(prob: np.ndarray, params: DecodeParams, duration: float) -> np.ndarray:
    """Boundary times (s) including 0 and ``duration``."""
    peaks = pick_peaks(prob, params.threshold, params.min_distance_sec)
    times = frame_times(len(prob))[peaks]
    margin = 0.5 * params.min_distance_sec
    times = times[(times > margin) & (times < duration - margin)]
    return np.concatenate([[0.0], times, [duration]])


def label_segments(bounds: np.ndarray, class_probs: np.ndarray) -> list[str]:
    centres = frame_times(class_probs.shape[0])
    out = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        m = (centres >= s) & (centres < e)
        if not m.any():
            m = np.zeros(len(centres), dtype=bool)
            m[min(int(np.searchsorted(centres, s)), len(centres) - 1)] = True
        out.append(CLASSES[int(np.argmax(class_probs[m].mean(axis=0)))])
    return out


def decode(boundary_prob: np.ndarray, class_probs: np.ndarray, params: DecodeParams,
           duration: float | None = None) -> tuple[np.ndarray, list[str]]:
    """-> intervals (n, 2) in seconds and n class names."""
    if duration is None:
        duration = len(boundary_prob) * config.FRAME_HOP_SEC
    bounds = boundaries_from_prob(boundary_prob, params, duration)
    intervals = np.stack([bounds[:-1], bounds[1:]], axis=1)
    return intervals, label_segments(bounds, class_probs)


def merge_repeated_labels(intervals: np.ndarray, seg_labels: list[str]) -> tuple[np.ndarray, list[str]]:
    """Optional: merge neighbouring segments with the same label (for display only;
    Harmonix itself often has back-to-back choruses, so evaluation does not merge)."""
    if len(seg_labels) == 0:
        return intervals, seg_labels
    out_int = [list(intervals[0])]
    out_lab = [seg_labels[0]]
    for (s, e), lab in zip(intervals[1:], seg_labels[1:]):
        if lab == out_lab[-1]:
            out_int[-1][1] = e
        else:
            out_int.append([s, e])
            out_lab.append(lab)
    return np.asarray(out_int), out_lab
