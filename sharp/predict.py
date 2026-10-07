"""Any song in, a labelled map of its structure out.

    python -m sharp.predict my_song.wav
    python -m sharp.predict song.mp3 --out-dir predictions/          # mp3/m4a/flac need ffmpeg
    python -m sharp.predict data/raw/melspecs/0001_12step-mel.npy \\
        --reference data/raw/harmonixset/dataset/segments/0001_12step.txt

For each input this prints the sections and writes to --out-dir:
    <name>.segments.txt   Harmonix-style "start_time label" rows + "end"
    <name>.json           the same, plus the boundary-probability curve
    <name>.png            structure strip (+ the expert one if --reference is given)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from . import config
from .annotations import load_segments, write_segments
from .audio import audio_to_mel
from .features import pool_mel
from .labels import display_name
from .model import load_checkpoint, resolve_device, run_model
from .plotting import fmt_time, plot_structure
from .postprocess import DecodeParams, decode


def predict_file(path: Path, model, stats, params: DecodeParams, device) -> dict:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        mel = np.load(path)
    else:
        mel = audio_to_mel(path)
    n_mel = mel.shape[1] if mel.shape[0] == config.HARMONIX_N_MELS else mel.shape[0]
    duration = n_mel * config.HARMONIX_HOP / config.HARMONIX_SR
    x = stats.apply(pool_mel(mel))
    bprob, cprob = run_model(model, x, device)
    intervals, seg_labels = decode(bprob, cprob, params, duration=duration)
    return {"name": path.stem.removesuffix("-mel"), "duration": duration,
            "intervals": intervals, "labels": seg_labels, "boundary_prob": bprob, "class_probs": cprob}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("songs", nargs="+", type=Path, help="audio files (.wav natively; others via ffmpeg) or Harmonix *-mel.npy")
    p.add_argument("--checkpoint", type=Path, default=config.RUNS_DIR / "cnn_bigru" / "best.pt")
    p.add_argument("--out-dir", type=Path, default=config.REPO_ROOT / "predictions")
    p.add_argument("--reference", type=Path, nargs="*", default=None,
                   help="Harmonix segments .txt per song, to plot the expert structure underneath")
    p.add_argument("--threshold", type=float, default=None, help="override the tuned boundary threshold")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    device = resolve_device(args.device)
    model, stats, ckpt = load_checkpoint(args.checkpoint, device)
    params = DecodeParams.from_dict(ckpt.get("decode"))
    if args.threshold is not None:
        params.threshold = args.threshold
    refs: list[Optional[Path]] = list(args.reference or [])
    refs += [None] * (len(args.songs) - len(refs))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for song, ref in zip(args.songs, refs):
        out = predict_file(song, model, stats, params, device)
        name = out["name"]
        print(f"\n{song}  ({fmt_time(out['duration'])})")
        for (s, e), lab in zip(out["intervals"], out["labels"]):
            print(f"  {fmt_time(s):>6} - {fmt_time(e):>6}  {display_name(lab)}")

        write_segments(args.out_dir / f"{name}.segments.txt", out["intervals"], out["labels"])
        (args.out_dir / f"{name}.json").write_text(json.dumps({
            "song": str(song), "duration": out["duration"], "decode": params.to_dict(),
            "segments": [{"start": float(s), "end": float(e), "label": lab}
                         for (s, e), lab in zip(out["intervals"], out["labels"])],
            "frame_hop_sec": config.FRAME_HOP_SEC,
            "boundary_prob": [round(float(v), 4) for v in out["boundary_prob"]],
        }, indent=1))
        rows = [("predicted", out["intervals"], out["labels"])]
        if ref is not None:
            ann = load_segments(ref)
            rows.insert(0, ("expert", ann.intervals,
                            [c if c is not None else r for c, r in zip(ann.classes, ann.raw_labels)]))
        plot_structure(rows, args.out_dir / f"{name}.png", title=name,
                       boundary_curve=out["boundary_prob"], threshold=params.threshold)
        print(f"  -> {args.out_dir / name}.segments.txt / .json / .png")


if __name__ == "__main__":
    main()
