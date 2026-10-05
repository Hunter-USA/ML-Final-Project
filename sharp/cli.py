"""Small helpers shared by the command-line entry points."""

from __future__ import annotations

import argparse
import random
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from . import config


def add_path_args(p: argparse.ArgumentParser, *names: str) -> None:
    defaults = {
        "annotation_dir": (config.ANNOTATION_DIR, "Harmonix dataset/ folder (metadata.csv, segments/)"),
        "mel_dir": (config.MEL_DIR, "folder with the extracted Harmonix *-mel.npy files"),
        "cache_dir": (config.CACHE_DIR, "where pooled features + stats are cached"),
        "splits_dir": (config.SPLITS_DIR, "train/val/test song lists"),
        "runs_dir": (config.RUNS_DIR, "checkpoints"),
        "results_dir": (config.RESULTS_DIR, "metrics tables and figures"),
    }
    for name in names:
        default, help_ = defaults[name]
        p.add_argument("--" + name.replace("_", "-"), type=Path, default=default,
                       help=f"{help_} (default: {default})")


def add_dataclass_args(p: argparse.ArgumentParser, cls) -> None:
    """Expose every field of a config dataclass as --field-name."""
    for f in fields(cls):
        flag = "--" + f.name.replace("_", "-")
        default = f.default
        help_ = f"(default: {' '.join(map(str, default)) if isinstance(default, tuple) else default})"
        if isinstance(default, bool):
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        elif isinstance(default, tuple):
            p.add_argument(flag, type=int, nargs="+", default=list(default), help=help_)
        else:
            p.add_argument(flag, type=type(default), default=default, help=help_)


def dataclass_from_args(cls, args: argparse.Namespace):
    kwargs = {f.name: getattr(args, f.name) for f in fields(cls)}
    for f in fields(cls):
        if isinstance(f.default, tuple):
            kwargs[f.name] = tuple(kwargs[f.name])
    return cls(**kwargs)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
