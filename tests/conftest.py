"""Shared fixtures. Everything here is synthetic, so the suite runs without the dataset."""

from __future__ import annotations

from pathlib import Path

import pytest

from sharp import config
from sharp.synthetic import make_fake_dataset

REAL_ANNOTATIONS = config.ANNOTATION_DIR
needs_real_annotations = pytest.mark.skipif(
    not (REAL_ANNOTATIONS / "metadata.csv").exists(),
    reason="Harmonix annotations not downloaded (python -m sharp.download --what annotations)")


@pytest.fixture(scope="session")
def fake_dataset(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("fake_harmonix")
    ann_dir, mel_dir = make_fake_dataset(root, n_songs=10, seed=0)
    return {"root": root, "annotation_dir": ann_dir, "mel_dir": mel_dir,
            "cache_dir": root / "processed", "splits_dir": root / "splits",
            "runs_dir": root / "runs", "results_dir": root / "results"}
