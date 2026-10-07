"""Build everything training needs from the raw download.

    python -m sharp.prepare

1. Loads the 912 annotations and metadata.
2. Uses the committed splits/{train,val,test}.txt (or creates them with --resplit).
3. Pools every Harmonix mel-spectrogram (80 bands) to ~0.19 s frames, in dB,
   cached as float16 under data/processed/features/ (~200 MB total).
4. Computes per-band mean/std on the training split (data/processed/stats.npz).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import config
from .annotations import load_annotations, load_metadata
from .cli import add_path_args
from .data import STATS_FILE, build_cache, check_mel_info, find_mel_files, load_features
from .features import compute_stats, make_targets
from .labels import CLASSES, IGNORE_INDEX, N_CLASSES
from .splits import load_splits, make_splits, save_splits, splits_exist


def prepare(annotation_dir: Path = config.ANNOTATION_DIR, mel_dir: Path = config.MEL_DIR,
            cache_dir: Path = config.CACHE_DIR, splits_dir: Path = config.SPLITS_DIR,
            force: bool = False, resplit: bool = False) -> None:
    md = load_metadata(annotation_dir)
    anns = load_annotations(annotation_dir)
    print(f"annotations: {len(anns)} songs, {sum(a.n_segments for a in anns.values()):,} sections")

    if resplit or not splits_exist(splits_dir):
        splits = make_splits(md)
        save_splits(splits, splits_dir)
        print(f"wrote new splits to {splits_dir}")
    else:
        splits = load_splits(splits_dir)
    print("splits: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    mel_files = find_mel_files(mel_dir)
    check_mel_info(mel_dir, strict=True)
    ids = [f for f in sorted(anns) if f in mel_files]
    if not ids:
        raise SystemExit(f"no *-mel.npy files found under {mel_dir}; run `python -m sharp.download` first")
    if len(ids) < len(anns):
        print(f"warning: {len(anns) - len(ids)} annotated songs have no mel-spectrogram and are skipped")

    print(f"caching features for {len(ids)} songs -> {cache_dir}")
    index = build_cache(ids, mel_files, cache_dir, force=force)

    have = set(ids)
    train_ids = [f for f in splits["train"] if f in have]
    stats = compute_stats(load_features(cache_dir, f) for f in train_ids)
    stats.save(Path(cache_dir) / STATS_FILE)
    print(f"feature stats from {len(train_ids)} training songs -> {Path(cache_dir) / STATS_FILE}")

    index = index.set_index("file_id", drop=False)
    short = [f for f in ids if anns[f].end > index.loc[f, "audio_sec"] + 2.0]
    if short:
        print(f"warning: {len(short)} songs have annotations running >2 s past the audio, e.g. {short[:3]}")
    for name, split_ids in splits.items():
        use = [f for f in split_ids if f in have]
        hours = index.loc[use, "audio_sec"].sum() / 3600 if use else 0.0
        print(f"  {name:5s}: {len(use):4d} songs, {hours:5.1f} h")

    counts = np.zeros(N_CLASSES, dtype=np.int64)
    for f in train_ids:
        sec, _ = make_targets(anns[f], int(index.loc[f, "n_frames"]))
        counts += np.bincount(sec[sec != IGNORE_INDEX], minlength=N_CLASSES)
    total = counts.sum()
    print("train label distribution: " + ", ".join(f"{c} {n / total:.1%}" for c, n in zip(CLASSES, counts)))
    print("next: python -m sharp.train   and   python -m sharp.baselines")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_args(p, "annotation_dir", "mel_dir", "cache_dir", "splits_dir")
    p.add_argument("--force", action="store_true", help="recompute cached features")
    p.add_argument("--resplit", action="store_true", help="overwrite splits/*.txt with a fresh split")
    args = p.parse_args(argv)
    prepare(args.annotation_dir, args.mel_dir, args.cache_dir, args.splits_dir, args.force, args.resplit)


if __name__ == "__main__":
    main()
