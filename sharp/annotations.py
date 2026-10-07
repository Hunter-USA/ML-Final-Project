"""Reading Harmonix metadata and segment annotations.

A segments file has one ``<time> <label>`` row per section start and a final
``<time> end`` row. Real-file quirks handled here:

* many songs start at the first beat (e.g. 0.85 s) instead of 0 s: a first
  boundary under ``PREROLL_SNAP_SEC`` is snapped to 0, a later one gets a
  leading "silence" segment;
* a few files have two ``end`` rows (we keep the first) or none (we use the
  track duration from metadata.csv);
* the audio often runs past ``end`` (fade-outs); that tail is unannotated and
  excluded from label training and from evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from . import config, labels


@dataclass
class SongAnnotation:
    file_id: str
    starts: np.ndarray            # (n,) section start times, starts[0] == 0
    end: float                    # end of the last annotated section
    raw_labels: list[str] = field(default_factory=list)

    @property
    def n_segments(self) -> int:
        return len(self.starts)

    @property
    def intervals(self) -> np.ndarray:
        """(n, 2) array of [start, end) times."""
        ends = np.append(self.starts[1:], self.end)
        return np.stack([self.starts, ends], axis=1)

    @property
    def classes(self) -> list[Optional[str]]:
        return [labels.to_class(l) for l in self.raw_labels]

    @property
    def boundary_times(self) -> np.ndarray:
        """Section changes the model should detect: every start after 0, plus the end."""
        return np.append(self.starts[1:], self.end)

    def eval_labels(self) -> list[str]:
        """Labels used as cluster ids by mir_eval: the class name, or the raw label if unmapped."""
        return [c if c is not None else f"raw:{labels.normalize_label(r)}"
                for c, r in zip(self.classes, self.raw_labels)]


def parse_segments(text: str, file_id: str = "", duration: Optional[float] = None) -> SongAnnotation:
    rows: list[tuple[float, str]] = []
    for line in text.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) < 2:
            continue
        try:
            t = float(parts[0])
        except ValueError:
            continue  # header or junk line
        rows.append((t, parts[1].strip()))
    if not rows:
        raise ValueError(f"{file_id}: no segment rows")
    rows.sort(key=lambda r: r[0])

    end: Optional[float] = None
    segs: list[tuple[float, str]] = []
    for t, lab in rows:
        if lab.lower() == "end":
            end = t
            break
        if segs and np.isclose(segs[-1][0], t):
            segs[-1] = (t, lab)          # zero-length duplicate -> keep the later label
        else:
            segs.append((t, lab))
    if end is None:
        if duration is None or duration <= segs[-1][0]:
            raise ValueError(f"{file_id}: no 'end' row and no usable duration")
        end = float(duration)

    segs = [(t, lab) for t, lab in segs if t < end]
    if not segs:
        raise ValueError(f"{file_id}: no segments before 'end'")

    if segs[0][0] < config.PREROLL_SNAP_SEC:
        segs[0] = (0.0, segs[0][1])
    else:
        segs.insert(0, (0.0, "silence"))

    starts = np.array([t for t, _ in segs], dtype=np.float64)
    return SongAnnotation(file_id=file_id, starts=starts, end=float(end),
                          raw_labels=[lab for _, lab in segs])


def load_segments(path: Path | str, duration: Optional[float] = None) -> SongAnnotation:
    path = Path(path)
    return parse_segments(path.read_text(encoding="utf-8", errors="replace"),
                          file_id=path.stem, duration=duration)


def load_metadata(annotation_dir: Path | str = config.ANNOTATION_DIR) -> pd.DataFrame:
    path = Path(annotation_dir) / "metadata.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `python -m sharp.download --what annotations` first")
    md = pd.read_csv(path, dtype={"File": str})
    return md.set_index("File", drop=False)


def load_annotations(annotation_dir: Path | str = config.ANNOTATION_DIR,
                     file_ids: Optional[Iterable[str]] = None) -> dict[str, SongAnnotation]:
    annotation_dir = Path(annotation_dir)
    md = load_metadata(annotation_dir)
    ids = list(file_ids) if file_ids is not None else list(md.index)
    out: dict[str, SongAnnotation] = {}
    for fid in ids:
        seg_path = annotation_dir / "segments" / f"{fid}.txt"
        if not seg_path.exists():
            continue
        duration = float(md.loc[fid, "Duration"]) if fid in md.index else None
        out[fid] = load_segments(seg_path, duration=duration)
    return out


def write_segments(path: Path | str, intervals: np.ndarray, seg_labels: list[str]) -> None:
    """Write segments in the Harmonix text format (``time label`` rows + ``end``)."""
    lines = [f"{s:.3f} {lab}" for (s, _), lab in zip(intervals, seg_labels)]
    lines.append(f"{intervals[-1, 1]:.3f} end")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
