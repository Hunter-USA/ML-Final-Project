"""Figures: structure strips (like the pitch slide) and confusion matrices."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .features import frame_times  # noqa: E402
from .labels import CLASSES, color_for, display_name  # noqa: E402


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


def plot_structure(rows: Sequence[tuple[str, np.ndarray, Sequence[str]]], out_path: Path | str,
                   title: str = "", boundary_curve: Optional[np.ndarray] = None,
                   threshold: Optional[float] = None) -> None:
    """One horizontal strip per row (name, intervals, labels); optional boundary-probability panel."""
    n_rows = len(rows) + (1 if boundary_curve is not None else 0)
    heights = [1.0] * len(rows) + ([1.6] if boundary_curve is not None else [])
    fig, axes = plt.subplots(n_rows, 1, figsize=(13, 0.75 * sum(heights) + 1.2), sharex=True,
                             gridspec_kw={"height_ratios": heights}, squeeze=False)
    axes = axes[:, 0]
    duration = max(float(np.max(iv)) for _, iv, _ in rows)
    seen: list[str] = []
    for ax, (name, intervals, seg_labels) in zip(axes, rows):
        for (s, e), lab in zip(intervals, seg_labels):
            ax.barh(0, e - s, left=s, height=0.85, color=color_for(lab), edgecolor="white", linewidth=1.5)
            if e - s > 0.04 * duration:
                ax.text((s + e) / 2, 0, display_name(lab), ha="center", va="center", fontsize=8)
            if lab in CLASSES and lab not in seen:
                seen.append(lab)
        ax.set_yticks([])
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
    if boundary_curve is not None:
        ax = axes[-1]
        t = frame_times(len(boundary_curve))
        ax.plot(t, boundary_curve, color="#333333", linewidth=1)
        if threshold is not None and threshold <= 1:
            ax.axhline(threshold, color="#d62728", linestyle="--", linewidth=1, label=f"threshold {threshold:.2f}")
            ax.legend(loc="upper right", fontsize=8, frameon=False)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("P(boundary)", rotation=0, ha="right", va="center", fontsize=10)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[-1].tick_params(axis="x", length=4)
    step = 30 if duration <= 360 else 60
    ticks = np.arange(0, duration + 1e-6, step)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([fmt_time(t) for t in ticks])
    axes[-1].set_xlim(0, duration)
    order = [c for c in CLASSES if c in seen]
    if order:
        fig.legend(handles=[Patch(color=color_for(c), label=display_name(c)) for c in order],
                   loc="lower center", ncol=min(len(order), 10), frameon=False, fontsize=8,
                   bbox_to_anchor=(0.5, -0.02))
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm: np.ndarray, out_path: Path | str, title: str = "") -> None:
    cm = np.asarray(cm, dtype=np.float64)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    names = [display_name(c) for c in CLASSES]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i].sum() > 0:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("expert label")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction of expert-labelled frames")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_history(history_csv: Path | str, out_path: Path | str) -> None:
    import pandas as pd
    h = pd.read_csv(history_csv)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    a1.plot(h["epoch"], h["train_loss"], label="train")
    a1.plot(h["epoch"], h["val_loss"], label="validation")
    a1.set_xlabel("epoch")
    a1.set_ylabel("loss")
    a1.legend(frameon=False)
    a2.plot(h["epoch"], h["val_frame_acc"], color="#2ca02c")
    a2.set_xlabel("epoch")
    a2.set_ylabel("validation frame accuracy")
    for a in (a1, a2):
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
