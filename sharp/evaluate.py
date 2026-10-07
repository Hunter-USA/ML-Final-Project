"""Score the CNN + BiGRU and every baseline against the expert labels.

    python -m sharp.evaluate                 # test split (use only once you're done tuning!)
    python -m sharp.evaluate --split val     # while developing

Writes to results/:
    summary_<split>.csv / .md      mean metric per method (+ the published MSAF numbers for context)
    per_song_<split>.csv           every metric for every song and method
    per_class_<split>.csv          precision / recall / F1 per section type and method
    significance_<split>.csv       paired Wilcoxon tests: CNN + BiGRU vs each baseline
    figures/confusion_<method>_<split>.png
    figures/examples_<split>/<song>.png   reference vs predictions, like the pitch slide
    figures/training_curves.png    (if runs/cnn_bigru/history.csv exists)
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from . import config
from .cli import add_path_args
from .data import load_features, load_split, load_stats
from .labels import N_CLASSES
from .metrics import HEADLINE_METRICS, confusion_matrix, evaluate_song, macro_f1, per_class_scores
from .model import resolve_device
from .plotting import plot_confusion, plot_history, plot_structure
from .postprocess import decode
from .predictors import ALL_METHODS, load_predictors

PRETTY = {
    "HitRate_0.5F": "HR.5F", "HitRate_3F": "HR3F", "HitRate_t0.5F": "HR.5F (trim)",
    "HitRate_t3F": "HR3F (trim)", "PWF": "PWF", "Sf": "Sf", "LabelAcc": "Label acc", "MacroF1": "Macro-F1",
    "n_est_segments": "Sections/song",
}
METHOD_NAMES = {
    "cnn_bigru": "CNN + BiGRU (ours)", "mlp": "MLP", "logreg": "Logistic regression",
    "novelty": "Foote novelty (unsupervised)", "majority": "Majority class",
    "msaf_published": "MSAF, published with Harmonix*",
}


def published_msaf(annotation_dir: Path, ids: Sequence[str]) -> Optional[dict]:
    """Mean scores of the unsupervised MSAF run released with the dataset, on the same songs."""
    path = Path(annotation_dir).parent / "results" / "segmentation" / "annot_beats.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"track_id": str})
    df = df[df["track_id"].isin(set(ids))]
    if df.empty:
        return None
    row = {"method": "msaf_published"}
    for k in HEADLINE_METRICS:
        row[k] = float(df[k].mean()) if k in df else float("nan")
    row["MacroF1"] = float("nan")
    return row


def to_markdown(summary: pd.DataFrame, split: str, n_songs: int, expert_sections: float = float("nan")) -> str:
    cols = [c for c in list(PRETTY) if c in summary.columns]
    lines = [f"# Results on the {split} split ({n_songs} songs)", "",
             "| Method | " + " | ".join(PRETTY[c] for c in cols) + " |",
             "|---|" + "---|" * len(cols)]
    for _, r in summary.iterrows():
        vals = ["-" if pd.isna(r[c]) else (f"{r[c]:.1f}" if c == "n_est_segments" else f"{r[c]:.3f}")
                for c in cols]
        lines.append(f"| {METHOD_NAMES.get(r['method'], r['method'])} | " + " | ".join(vals) + " |")
    lines += ["",
              "HR = boundary hit rate F-measure within ±0.5 s / ±3 s (trim: ignoring the first and last boundary); "
              "PWF = pairwise frame-clustering F; Sf = normalised conditional entropy F; "
              "Label acc = share of 0.1 s frames whose section type matches the expert; "
              "Macro-F1 = mean per-class F1 over the section types; "
              f"Sections/song = predicted sections per song (the experts average {expert_sections:.1f}).",
              "",
              "*MSAF row: per-song results published with the Harmonix Set (Nieto et al., ISMIR 2019; "
              "unsupervised, annotated-beat features), averaged over the same songs; it predicts no section names."]
    return "\n".join(lines) + "\n"


def significance(per_song: pd.DataFrame, reference: str = "cnn_bigru") -> pd.DataFrame:
    rows = []
    if reference not in set(per_song["method"]):
        return pd.DataFrame(rows)
    ref = per_song[per_song["method"] == reference].set_index("file_id")
    for other in per_song["method"].unique():
        if other == reference:
            continue
        oth = per_song[per_song["method"] == other].set_index("file_id")
        common = ref.index.intersection(oth.index)
        for metric in HEADLINE_METRICS:
            a = ref.loc[common, metric].astype(float)
            b = oth.loc[common, metric].astype(float)
            ok = a.notna() & b.notna()
            a, b = a[ok], b[ok]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    p = float(wilcoxon(a, b).pvalue)
            except ValueError:            # e.g. all differences are zero
                p = float("nan")
            rows.append({"method": other, "metric": metric, "ours_mean": a.mean(), "baseline_mean": b.mean(),
                         "mean_difference": (a - b).mean(), "wilcoxon_p": p, "n_songs": int(ok.sum())})
    return pd.DataFrame(rows)


def run(split: str = "test", annotation_dir: Path = config.ANNOTATION_DIR, cache_dir: Path = config.CACHE_DIR,
        splits_dir: Path = config.SPLITS_DIR, runs_dir: Path = config.RUNS_DIR,
        results_dir: Path = config.RESULTS_DIR, checkpoint: Optional[Path] = None,
        methods: Sequence[str] = ALL_METHODS, n_examples: int = 4, device: str = "auto",
        limit: Optional[int] = None) -> pd.DataFrame:
    results_dir = Path(results_dir)
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    dev = resolve_device(device)

    ids, anns, index = load_split(split, annotation_dir, cache_dir, splits_dir, limit)
    cache_stats = load_stats(cache_dir)
    predictors = load_predictors(runs_dir, checkpoint, dev, methods)
    if not predictors:
        raise SystemExit("nothing to evaluate - train the model and/or baselines first")
    raw = {fid: load_features(cache_dir, fid) for fid in ids}
    example_ids = [ids[i] for i in np.linspace(0, len(ids) - 1, min(n_examples, len(ids))).astype(int)] \
        if n_examples > 0 else []
    examples: dict[str, list] = {fid: [] for fid in example_ids}

    rows, cms = [], {}
    for pred in predictors:
        stats = pred.stats or cache_stats
        cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
        for fid in ids:
            bprob, cprob = pred.predict(stats.apply(raw[fid]))
            intervals, seg_labels = decode(bprob, cprob, pred.params, duration=float(index.loc[fid, "audio_sec"]))
            m, ref_idx, est_idx = evaluate_song(anns[fid], intervals, seg_labels)
            cm += confusion_matrix(ref_idx, est_idx)
            rows.append({"method": pred.name, "file_id": fid, **m})
            if fid in examples:
                examples[fid].append((pred.name, intervals, seg_labels))
        cms[pred.name] = cm
        print(f"evaluated {pred.name} on {len(ids)} {split} songs")

    per_song = pd.DataFrame(rows)
    per_song.to_csv(results_dir / f"per_song_{split}.csv", index=False)

    metric_cols = [c for c in per_song.columns if c not in ("method", "file_id")]
    summary = per_song.groupby("method", sort=False)[metric_cols].mean().reset_index()
    summary["MacroF1"] = summary["method"].map(lambda m: macro_f1(cms[m]))
    msaf = published_msaf(annotation_dir, ids)
    if msaf is not None:
        summary = pd.concat([summary, pd.DataFrame([msaf])], ignore_index=True)
    summary.to_csv(results_dir / f"summary_{split}.csv", index=False)
    md = to_markdown(summary, split, len(ids), float(per_song["n_ref_segments"].mean()))
    (results_dir / f"summary_{split}.md").write_text(md, encoding="utf-8")

    per_class = pd.concat([per_class_scores(cm).assign(method=m) for m, cm in cms.items()], ignore_index=True)
    per_class[["method", "class", "precision", "recall", "f1", "support"]].to_csv(
        results_dir / f"per_class_{split}.csv", index=False)
    sig = significance(per_song)
    if not sig.empty:
        sig.to_csv(results_dir / f"significance_{split}.csv", index=False)

    for name, cm in cms.items():
        if name in ("cnn_bigru", "mlp", "logreg"):
            plot_confusion(cm, fig_dir / f"confusion_{name}_{split}.png",
                           title=f"{METHOD_NAMES.get(name, name)} - {split} split")
    for fid, preds in examples.items():
        ann = anns[fid]
        ref_labels = [c if c is not None else r for c, r in zip(ann.classes, ann.raw_labels)]
        rows_ = [("expert", ann.intervals, ref_labels)] + \
                [(METHOD_NAMES.get(n, n).replace(" (ours)", ""), iv, lab) for n, iv, lab in preds]
        plot_structure(rows_, fig_dir / f"examples_{split}" / f"{fid}.png", title=fid)
    history = Path(runs_dir) / "cnn_bigru" / "history.csv"
    if history.exists():
        plot_history(history, fig_dir / "training_curves.png")

    print()
    print(md)
    return summary


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_args(p, "annotation_dir", "cache_dir", "splits_dir", "runs_dir", "results_dir")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--checkpoint", type=Path, default=None, help="default: runs/cnn_bigru/best.pt")
    p.add_argument("--methods", nargs="+", default=list(ALL_METHODS), choices=list(ALL_METHODS))
    p.add_argument("--n-examples", type=int, default=4, help="songs to draw as structure strips")
    p.add_argument("--device", default="auto")
    p.add_argument("--limit-songs", type=int, default=None)
    args = p.parse_args(argv)
    run(args.split, args.annotation_dir, args.cache_dir, args.splits_dir, args.runs_dir, args.results_dir,
        args.checkpoint, args.methods, args.n_examples, args.device, args.limit_songs)


if __name__ == "__main__":
    main()
