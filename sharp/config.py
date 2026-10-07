"""Paths, feature parameters and default hyperparameters shared across the project."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (all relative to the repository root; every CLI lets you override them)
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
HARMONIX_DIR = RAW_DIR / "harmonixset"          # clone / zip of github.com/urinieto/harmonixset
ANNOTATION_DIR = HARMONIX_DIR / "dataset"        # metadata.csv + segments/*.txt
MEL_DIR = RAW_DIR / "melspecs"                   # extracted Harmonix_melspecs.tgz
CACHE_DIR = DATA_DIR / "processed"               # pooled log-mel features + stats
SPLITS_DIR = REPO_ROOT / "splits"                # train/val/test song lists (committed)
RUNS_DIR = REPO_ROOT / "runs"                    # checkpoints
RESULTS_DIR = REPO_ROOT / "results"              # metrics tables + figures

# --------------------------------------------------------------------------- #
# Parameters of the published Harmonix mel-spectrograms (Harmonix_melspecs.tgz,
# Dec 2020 release), as recorded in the info.json shipped inside the archive:
#   librosa 0.7.0, SR 22050, N_MELS 80, N_FFT 2048, HOP_LENGTH 1024, MEL_FMIN 0, MEL_FMAX null
# Files are <id>-mel.npy, shape (80, frames), float32 power (not dB).
# librosa 0.7 defaults apply to the rest: Hann window, centred frames with
# reflect padding, power 2, Slaney mel scale/normalisation, fmax = SR / 2.
# --------------------------------------------------------------------------- #
HARMONIX_SR = 22050
HARMONIX_N_FFT = 2048
HARMONIX_HOP = 1024
HARMONIX_N_MELS = 80
HARMONIX_FMIN = 0.0
HARMONIX_FMAX = HARMONIX_SR / 2.0
HARMONIX_PAD_MODE = "reflect"

# --------------------------------------------------------------------------- #
# Model input representation
# --------------------------------------------------------------------------- #
TIME_POOL = 4                 # average 4 mel frames -> 1 model frame
N_BANDS = 80                  # keep all 80 mel bands
FRAME_HOP_SEC = TIME_POOL * HARMONIX_HOP / HARMONIX_SR   # ~0.186 s per model frame
AMIN = 1e-10                  # floor before taking the log
TOP_DB = 80.0                 # dynamic range kept per song (like librosa.power_to_db)

# Annotation handling
PREROLL_SNAP_SEC = 1.0        # a first boundary earlier than this is snapped to 0 s

# Evaluation
LABEL_SAMPLE_SEC = 0.1        # grid used for frame-level label accuracy

SPLIT_SEED = 4342
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)


@dataclass
class TrainConfig:
    """Hyperparameters for the CNN + BiGRU (override any of them from the CLI)."""

    run_name: str = "cnn_bigru"
    epochs: int = 40
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_frames: int = 1800                 # random training crop (~5 min); full songs at eval
    conv_channels: tuple = (16, 32, 64)
    gru_hidden: int = 128
    gru_layers: int = 2
    dropout: float = 0.3
    boundary_weight: float = 1.0           # lambda in  L = CE(section) + lambda * BCE(boundary)
    boundary_pos_weight: float = 5.0       # boundaries are rare -> up-weight positives
    boundary_sigma: float = 1.0            # target smearing around each boundary (frames)
    label_smoothing: float = 0.05
    spec_augment: bool = True
    patience: int = 8                      # early stopping on validation loss
    grad_clip: float = 1.0
    seed: int = 4342
    device: str = "auto"                   # auto | cpu | cuda | mps
    num_threads: int = 0                   # 0 = let torch decide

    def to_dict(self) -> dict:
        d = asdict(self)
        d["conv_channels"] = list(self.conv_channels)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        names = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in names}
        if "conv_channels" in kwargs:
            kwargs["conv_channels"] = tuple(kwargs["conv_channels"])
        return cls(**kwargs)
