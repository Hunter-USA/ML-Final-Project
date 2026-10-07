"""Scoring against the expert annotations (mir_eval + label metrics).

Per song we report the standard MIREX structural-segmentation metrics, using the
same column names as the Harmonix paper's published results:

* HitRate_0.5{P,R,F}, HitRate_3{P,R,F}: boundary detection within +-0.5 s / +-3 s
* HitRate_t0.5F, HitRate_t3F: same, ignoring the trivial first/last boundary ("trimmed")
* PW{P,R,F}: pairwise frame clustering (do two frames share a label in both?)
* S{o,u,f}: normalised conditional entropy scores (over-/under-segmentation)

and, because our model also names sections, frame-level label metrics:

* LabelAcc: fraction of 0.1 s frames whose predicted class equals the expert's
* per-class precision/recall/F1 and macro-F1 over the whole split (from a confusion matrix)

Evaluation covers [0, end] of the annotation; predictions past the annotated
end (fade-outs) are clipped.
"""

from __future__ import annotations

import warnings
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from mir_eval import segment as mseg
from mir_eval import util as mutil

from . import config
from .annotations import SongAnnotation
from .labels import CLASS_TO_INDEX, CLASSES, N_CLASSES
from .postprocess import DecodeParams, boundaries_from_prob

HEADLINE_METRICS = ["HitRate_0.5F", "HitRate_3F", "HitRate_t0.5F", "HitRate_t3F", "PWF", "Sf", "LabelAcc"]


def clip_estimate(intervals: np.ndarray, seg_labels: Sequence[str], t_max: float) -> tuple[np.ndarray, list[str]]:
    """Restrict an estimated segmentation to [0, t_max] so it spans exactly the reference."""
    intervals = np.asarray(intervals, dtype=np.float64)
    keep = intervals[:, 0] < t_max - 1e-6
    if not keep.any():
        keep[0] = True
    iv = intervals[keep].copy()
    lab = [l for l, k in zip(seg_labels, keep) if k]
    iv[:, 1] = np.minimum(iv[:, 1], t_max)
    iv[-1, 1] = t_max
    iv[0, 0] = 0.0
    return iv, lab


def boundary_scores(ref_int: np.ndarray, est_int: np.ndarray) -> dict:
    out = {}
    for window in (0.5, 3.0):
        for trim in (False, True):
            with warnings.catch_warnings():       # a one-segment estimate has no inner boundaries
                warnings.simplefilter("ignore", UserWarning)
                p, r, f = mseg.detection(ref_int, est_int, window=window, trim=trim)
            tag = f"HitRate_{'t' if trim else ''}{window:g}"
            out[f"{tag}P"], out[f"{tag}R"], out[f"{tag}F"] = p, r, f
    return out


def sample_labels(intervals: np.ndarray, seg_labels: Sequence, end: float,
                  step: float = config.LABEL_SAMPLE_SEC) -> list:
    times = np.arange(0.0, end, step) + step / 2
    times = times[times < end]
    return mutil.interpolate_intervals(np.asarray(intervals), list(seg_labels), times, fill_value=None)


def evaluate_song(ann: SongAnnotation, est_intervals: np.ndarray,
                  est_labels: Sequence[str]) -> tuple[dict, np.ndarray, np.ndarray]:
    """-> (metrics dict, reference class indices, estimated class indices) on a 0.1 s grid."""
    ref_int = ann.intervals
    ref_lab = ann.eval_labels()
    est_int, est_lab = clip_estimate(est_intervals, est_labels, ann.end)

    m = boundary_scores(ref_int, est_int)
    m["PWP"], m["PWR"], m["PWF"] = mseg.pairwise(ref_int, ref_lab, est_int, est_lab,
                                                 frame_size=config.LABEL_SAMPLE_SEC)
    m["So"], m["Su"], m["Sf"] = mseg.nce(ref_int, ref_lab, est_int, est_lab,
                                         frame_size=config.LABEL_SAMPLE_SEC)

    ref_s = sample_labels(ref_int, ann.classes, ann.end)
    est_s = sample_labels(est_int, est_lab, ann.end)
    pairs = [(CLASS_TO_INDEX[r], CLASS_TO_INDEX[e]) for r, e in zip(ref_s, est_s)
             if r is not None and e in CLASS_TO_INDEX]
    ref_idx = np.array([p[0] for p in pairs], dtype=np.int64)
    est_idx = np.array([p[1] for p in pairs], dtype=np.int64)
    m["LabelAcc"] = float(np.mean(ref_idx == est_idx)) if len(pairs) else float("nan")
    m["n_ref_segments"] = len(ref_int)
    m["n_est_segments"] = len(est_int)
    return m, ref_idx, est_idx


def confusion_matrix(ref_idx: np.ndarray, est_idx: np.ndarray, n: int = N_CLASSES) -> np.ndarray:
    return np.bincount(np.asarray(ref_idx) * n + np.asarray(est_idx), minlength=n * n).reshape(n, n)


def per_class_scores(cm: np.ndarray) -> pd.DataFrame:
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return pd.DataFrame({"class": list(CLASSES), "precision": precision, "recall": recall,
                         "f1": f1, "support": support})


def macro_f1(cm: np.ndarray) -> float:
    df = per_class_scores(cm)
    df = df[df["support"] > 0]
    return float(df["f1"].mean()) if len(df) else float("nan")


def tune_decode_params(songs: Iterable[tuple[SongAnnotation, np.ndarray, float]],
                       thresholds: Sequence[float] = (0.02,) + tuple(np.round(np.arange(0.05, 0.96, 0.05), 2)),
                       min_distances: Sequence[float] = (2.0, 3.0, 4.0, 6.0)) -> tuple[DecodeParams, pd.DataFrame]:
    """Grid-search peak-picking parameters on validation songs.

    ``songs`` yields (annotation, boundary probability curve, audio duration).
    Objective: mean over songs of (HitRate_0.5F + HitRate_3F) / 2.
    """
    songs = list(songs)
    rows = []
    for dist in min_distances:
        for thr in thresholds:
            params = DecodeParams(float(thr), float(dist))
            scores = []
            for ann, prob, duration in songs:
                bounds = boundaries_from_prob(prob, params, duration)
                est = np.stack([bounds[:-1], bounds[1:]], axis=1)
                est, _ = clip_estimate(est, ["x"] * len(est), ann.end)
                _, _, f05 = mseg.detection(ann.intervals, est, window=0.5)
                _, _, f3 = mseg.detection(ann.intervals, est, window=3.0)
                scores.append(0.5 * (f05 + f3))
            rows.append({"threshold": thr, "min_distance_sec": dist, "score": float(np.mean(scores))})
    table = pd.DataFrame(rows)
    best = table.loc[table["score"].idxmax()]
    return DecodeParams(float(best["threshold"]), float(best["min_distance_sec"])), table
