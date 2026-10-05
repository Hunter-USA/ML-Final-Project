"""Stand-in data for smoke tests - NOT for results.

Generates files in the exact Harmonix format (``<id>-mel.npy``, 80 mel bands,
power, 22.05 kHz / hop 1024, plus ``info.json``) whose "sound" is shaped noise that
changes timbre at each annotated boundary and repeats timbre for repeated
sections. It lets you exercise the whole pipeline before (or without)
downloading the real 1.2 GB archive:

    python -m sharp.synthetic --out data/raw/melspecs_standin --n-songs 150
    python -m sharp.prepare --mel-dir data/raw/melspecs_standin --cache-dir data/processed_standin

Scores obtained on stand-in data say nothing about real performance.
"""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from . import config
from .annotations import SongAnnotation, load_annotations, load_metadata
from .labels import CLASSES, normalize_label, to_class

FORM = ("intro", "verse", "prechorus", "chorus", "verse", "prechorus", "chorus", "bridge", "chorus", "outro")


def class_templates(seed: int = 1234, n_mels: int = config.HARMONIX_N_MELS) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    k = np.arange(n_mels)
    out = {}
    for c in CLASSES:
        if c == "silence":
            out[c] = np.full(n_mels, -75.0)
            continue
        env = -15.0 - 0.12 * k
        for _ in range(3):
            mu, width, height = rng.uniform(10, n_mels - 10), rng.uniform(8, 40), rng.uniform(-10, 12)
            env = env + height * np.exp(-0.5 * ((k - mu) / width) ** 2)
        out[c] = env + rng.uniform(-6, 6)
    return out


def synth_mel(ann: SongAnnotation, duration: float, bpm: float, rng: np.random.Generator,
              templates: dict[str, np.ndarray], n_mels: int = config.HARMONIX_N_MELS) -> np.ndarray:
    n = 1 + int(round(duration * config.HARMONIX_SR)) // config.HARMONIX_HOP
    t = np.arange(n) * config.HARMONIX_HOP / config.HARMONIX_SR
    k = np.arange(n_mels)
    tilt = rng.normal(0, 0.03) * (k - n_mels / 2)
    seg = np.clip(np.searchsorted(ann.starts, t, side="right") - 1, 0, None)
    db = np.empty((n_mels, n), dtype=np.float32)
    voiced = np.ones(n, dtype=np.float32)
    variants: dict[str, np.ndarray] = {}
    for i, raw in enumerate(ann.raw_labels):
        m = (seg == i) & (t < ann.end)
        if not m.any():
            continue
        cls = to_class(raw) or "break"
        key = normalize_label(raw)
        if key not in variants:
            variants[key] = gaussian_filter1d(rng.normal(0, 4, n_mels), 6)
        db[:, m] = (templates[cls] + tilt + variants[key])[:, None]
        if cls == "silence":
            voiced[m] = 0.0
    tail = t >= ann.end
    if tail.any():
        db[:, tail] = (templates["outro"] + tilt)[:, None] - 3.0 * (t[tail] - ann.end)[None, :]
    beat = (np.cos(2 * np.pi * t * bpm / 60.0) > 0.7).astype(np.float32) * 8.0
    db[: max(4, n_mels // 6)] += (beat * voiced)[None, :]
    db += rng.normal(0, 5, size=db.shape).astype(np.float32)
    return (10.0 ** (db / 10.0)).astype(np.float32)


def info_dict() -> dict:
    return {"SR": config.HARMONIX_SR, "N_MELS": config.HARMONIX_N_MELS, "N_FFT": config.HARMONIX_N_FFT,
            "HOP_LENGTH": config.HARMONIX_HOP, "MEL_FMIN": config.HARMONIX_FMIN, "MEL_FMAX": None,
            "note": "SYNTHETIC STAND-IN DATA - not real audio"}


def write_standins(annotation_dir: Path, out_dir: Path, n_songs: Optional[int] = None,
                   file_ids: Optional[Sequence[str]] = None, seed: int = 0,
                   archive: Optional[Path] = None, verbose: bool = True) -> list[str]:
    md = load_metadata(annotation_dir)
    anns = load_annotations(annotation_dir, file_ids)
    ids = sorted(anns)
    rng = np.random.default_rng(seed)
    if n_songs and n_songs < len(ids):
        ids = sorted(rng.choice(ids, size=n_songs, replace=False).tolist())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    templates = class_templates()
    for i, fid in enumerate(ids):
        duration = float(md.loc[fid, "Duration"])
        bpm = float(md.loc[fid, "BPM"]) if "BPM" in md.columns and pd.notna(md.loc[fid, "BPM"]) else 120.0
        np.save(out_dir / f"{fid}-mel.npy", synth_mel(anns[fid], duration, bpm, rng, templates))
        if verbose and ((i + 1) % 25 == 0 or i + 1 == len(ids)):
            print(f"  wrote {i + 1}/{len(ids)} stand-in spectrograms")
    (out_dir / "info.json").write_text(json.dumps(info_dict(), indent=2))
    if archive is not None:
        with tarfile.open(archive, "w:gz") as tar:
            for p in sorted(out_dir.iterdir()):
                tar.add(p, arcname=f"melspecs/{p.name}")
    return ids


def make_fake_dataset(root: Path, n_songs: int = 8, seed: int = 0,
                      min_duration: float = 30.0, max_duration: float = 50.0) -> tuple[Path, Path]:
    """A tiny fully synthetic Harmonix-like dataset (annotations + mels) for unit tests."""
    root = Path(root)
    ann_dir = root / "harmonixset" / "dataset"
    (ann_dir / "segments").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_songs):
        fid = f"{i + 1:04d}_song{i + 1}"
        duration = float(rng.uniform(min_duration, max_duration))
        lens = rng.uniform(0.6, 1.4, len(FORM))
        lens = lens / lens.sum() * (duration - 2.0)
        t = float(rng.uniform(0.0, 0.5))
        lines = []
        for lab, length in zip(FORM, lens):
            lines.append(f"{t:.6f} {lab}")
            t += length
        lines.append(f"{t:.6f} end")
        (ann_dir / "segments" / f"{fid}.txt").write_text("\n".join(lines) + "\n")
        rows.append({"File": fid, "Title": f"Song {i + 1}", "Artist": "Synthetic", "Release": "Test",
                     "Duration": round(duration, 3), "BPM": int(rng.integers(80, 140)),
                     "Ratio Bars in 4": 100.0, "Time Signature": "4|4", "Genre": "Pop",
                     "MusicBrainz Id": "", "Acoustid Id": ""})
    pd.DataFrame(rows).to_csv(ann_dir / "metadata.csv", index=False)
    mel_dir = root / "melspecs"
    write_standins(ann_dir, mel_dir, seed=seed, verbose=False)
    return ann_dir, mel_dir


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annotation-dir", type=Path, default=config.ANNOTATION_DIR)
    p.add_argument("--out", type=Path, default=config.RAW_DIR / "melspecs_standin")
    p.add_argument("--n-songs", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--archive", type=Path, default=None, help="also pack the files into a .tgz like the real download")
    args = p.parse_args(argv)
    ids = write_standins(args.annotation_dir, args.out, args.n_songs, seed=args.seed, archive=args.archive)
    print(f"{len(ids)} stand-in spectrograms in {args.out} (synthetic - for pipeline testing only)")


if __name__ == "__main__":
    main()
