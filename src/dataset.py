
"""
This File represents the dataset pipeline.

It requires:

1. The annotation repo: git clone https://github.com/urinieto/harmonixset
    -> dataset/segments/*.txt
    -> dataset/metadata.csv

2. The mel spectrograms Harmonix_melspecs.tgz linked from the repo and untarred anywhere.
"""

from dataclasses import dataclass
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset

@dataclass
class SpecConfig:
    """
    Configuration for spectrogram
    """
    sr: int = 22050 # Sampling rate
    hop_length: int = 1024 # hop length for the spectrogram
    n_mels: int = 80 # number of mel bands

    @property
    def native_fps(self) -> float:
        return self.sr / self.hop_length

@dataclass
class Track:
    """
    Represents a track in the dataset.
    """
    name: str
    artist: str
    spec_path: Path
    boundaries: np.ndarray
    labels: list[str]
    duration: float


def load_spec_config(melspec_dir: Path, override: SpecConfig | None = None) -> SpecConfig:
    """
    Load the spectrogram configuration from the melspec directory.

    Args:
        melspec_dir (Path): Path to the directory containing the melspecs.
        override (SpecConfig | None): Optional override for the default configuration.

    Returns:
        SpecConfig: The loaded spectrogram configuration.
    """

    if override is not None:
        return override

    config = SpecConfig()
    info_paths = list(melspec_dir.glob("info.json"))

    # If no info.json is found, return the default SpecConfig and print a warning.
    if not info_paths:
        print(f"[WARNING] No info.json found in {melspec_dir}. Using default SpecConfig.")
        return config

    # Load the info.json file and update the SpecConfig accordingly.
    info = json.loads(info_paths[0].read_text())

    # Update the SpecConfig with values from info.json if they exist.
    if "SR" in info:
        config.sr = info["SR"]
    if "HOP_LENGTH" in info:
        config.hop_length = info["HOP_LENGTH"]
    if "N_MELS" in info:
        config.n_mels = info["N_MELS"]

    # Print the loaded configuration for debugging purposes.
    print(f"[INFO] Loaded SpecConfig from {info_paths[0]}: {config}")
    print(f"[INFO] Native FPS: {config.native_fps}")
    
    return config

def parse_segments(path: Path) -> tuple[np.ndarray, list[str]]:
    """
    Parse the segments from a given path.

    Args:
        path (Path): Path to the segments file.
    
    Returns:
        tuple[np.ndarray, list[str]]: A tuple containing the segments as a numpy array representing time and the corresponding labels as a list of strings.
    """
    times, labels = [], []
    # Read the segments file and parse the time and label information.
    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            # Skip lines that do not have at least two parts (time and label).
            if len(parts) < 2:
                continue
            try:
                time = float(parts[0])
            except ValueError:
                print(f"[WARNING] Could not convert {parts[0]} to float in file {path}. Skipping line.")
                continue
            times.append(time)
            labels.append(parts[1].strip().lower())
        # Sort the times and labels based on the time values to ensure they are in chronological order.
        order = np.argsort(times)
        times = np.asarray(times, dtype=np.float64)[order]
        labels = [labels[i] for i in order]

    # Filter out segments at time 0 and "end" labels, keeping times and labels paired.
    keep = [i for i, (t, lab) in enumerate(zip(times, labels)) if t != 0 and lab != "end"]
    times = times[keep]
    labels = [labels[i] for i in keep]

    return np.array(times), labels

def build_index(dataset_dir: Path, melspec_dir: Path) -> list[Track]:
    """
    Build an index of tracks from the dataset directory and the melspec directory.
    
    Args:
        dataset_dir (Path): Path to the dataset directory containing segments and metadata.
        melspec_dir (Path): Path to the directory containing the mel spectrograms.
    
    Returns:
        list[Track]: A list of Track objects representing the tracks in the dataset.
    """
    meta = pd.read_csv(dataset_dir / "metadata.csv")
    meta.columns = [c.strip() for c in meta.columns]
    meta = meta.set_index("File")

    specs: dict[str, Path] = {}
    for p in melspec_dir.glob("*.npy"):
        specs.setdefault(p.stem, p)
    if not specs:
        raise ValueError(f"[ERROR] No mel spectrograms found in {melspec_dir}")

    # Build a mapping from track stems to their corresponding mel spectrogram paths.
    prefix_map: dict[str, Path] = {}
    for stem, p in specs.items():
        prefix_map.setdefault(stem.split("-mel")[0], p)

    # Build the list of tracks, keeping track of missing spectrograms and metadata.
    tracks, missing_spec, missing_meta = [], 0, 0
    for seg_path in sorted((dataset_dir / "segments").glob("*.txt")):
        name = seg_path.stem
        spec_path = specs.get(name) or prefix_map.get(name)
        if spec_path is None:
            missing_spec += 1
            print(f"[WARNING] Missing mel spectrogram for {name}. Skipping track.")
            continue
        if name not in meta.index:
            missing_meta += 1
            print(f"[WARNING] Missing metadata for {name}. Skipping track.")
            continue
        row = meta.loc[name]
        times, labels = parse_segments(seg_path)
        if len(times) < 2:
            print(f"[WARNING] Not enough segments for {name}. Skipping track.")
            continue
        tracks.append(
            Track(
                name=name,
                artist=str(row.get("Artist", "Unknown").strip().lower()),
                spec_path=spec_path,
                boundaries=times,
                labels=labels,
                duration=float(row.get("Duration", times[-1]))
            )
        )
    print(f"[INFO] Build index completed. {len(tracks)} tracks built. Missing spectrograms: {missing_spec}, Missing metadata: {missing_meta}")
    return tracks


def gaussian_targets(
        boundaries: np.ndarray,
        n_frames: int,
        fps: float,
        sigma_sec: float
) -> np.ndarray:
    """
    Generate Gaussian targets for the given boundaries.
    
    Args:
        boundaries (np.ndarray): Array of boundary times.
        n_frames (int): Number of frames in the spectrogram.
        fps (float): Frames per second of the spectrogram.
        sigma_sec (float): Standard deviation of the Gaussian in seconds.
        
    Returns:
        np.ndarray: The generated Gaussian targets.
    """
    t = np.arange(n_frames) / fps
    y = np.zeros(n_frames, dtype=np.float32)
    for b in boundaries:
        # exp(-t² / 2σ²) 
        # basically dimished returns for points further away from the boundary 
        # so if it misses by like 1 millisecond it doesn't matter, 
        # but if it misses by 1 second it matters a lot.
        y = np.maximum(y, np.exp(-((t - b) ** 2) / (2.0 * sigma_sec**2)).astype(np.float32))
    return y

def load_spec(path: Path, config: SpecConfig, pool: int) -> np.ndarray:
    """
    Load a melspec and make it (n_mels, time)
    
    Args:
        path (Path): Path to the mel spectrogram file.
        config (SpecConfig): Configuration for the spectrogram.
        pool (int): Pooling factor to reduce the time dimension.
        
    Returns:
        np.ndarray: The loaded and pooled mel spectrogram.
    """

    spec = np.load(path).astype(np.float32)
    if spec.ndim != 2 or spec.shape[0] != config.n_mels:
        raise ValueError(f"[ERROR] Invalid mel spectrogram shape {spec.shape} for {path}. Expected shape: ({config.n_mels}, time).")
    # Normalize the spectrogram if all values are non-negative.
    if spec.min() >= 0.0:
        spec = np.log1p(spec)

    # Pool the spectrogram along the time dimension if the pooling factor is greater than 1.
    if pool > 1:
        t = (spec.shape[1] // pool) * pool
        spec = spec[:, :t].reshape(spec.shape[0], t // pool, pool).mean(axis=2)
    
    # Normalize the spectrogram to have zero mean and unit variance.
    spec -= spec.mean()
    spec /= spec.std() + 1e-9

    return spec

class PatchDataset(Dataset):
    """
    A PyTorch Dataset for making patches from the mel spectrograms
    """
    def __init__(
            self,
            tracks: list[Track],
            config: SpecConfig,
            pool: int,
            context_sec: float = 16.0,
            sigma_sec: float = 1.0,
            negative_per_positive: int = 3,
            positive_threshold: float = 0.5,
            seed: int = 42
    ):
        """
        Initialize the PatchDataset.
        
        Args:
            tracks (list[Track]): List of Track objects representing the dataset.
            config (SpecConfig): Configuration for the spectrogram.
            pool (int): Pooling factor to reduce the time dimension.
            context_sec (float): Context window size in seconds.
            sigma_sec (float): Standard deviation of the Gaussian in seconds.
            negative_per_positive (int): Ratio of negative to positive samples.
            positive_threshold (float): Threshold for considering a sample as positive.
            seed (int): Random seed for reproducibility.
        """
        # Pooling in load_spec reduces the frame rate, so the targets and context must use the pooled rate.
        self.fps = config.native_fps / max(1, pool)
        self.width = int(round(context_sec * self.fps)) | 1
        self.half_width = self.width // 2
        self.negative_per_positive = negative_per_positive
        self.positive_threshold = positive_threshold
        self.rng = np.random.default_rng(seed)

        # Load the mel spectrograms, targets, labels, and names for each track.
        self.specs, self.targets, self.labels, self.names = [], [], [], []
        for track in tracks:
            # Load the mel spectrogram for the track and pad it to account for context.
            spec = load_spec(track.spec_path, config, pool)
            self.specs.append(np.pad(spec, ((0, 0), (self.half_width, self.half_width)), mode="edge"))

            # Generate Gaussian targets for the track's boundaries and append them to the targets list.
            y = gaussian_targets(track.boundaries, spec.shape[1], self.fps, sigma_sec)
            self.targets.append(y)

            # Append the track's name and labels to the respective lists.
            self.names.append(track.name)
            self.labels.append(track.labels)

        self.n_mels = self.specs[0].shape[0]
        self.resample()
    
    
    def resample(self) -> None:
        """
        Resample the dataset balancing positive and negative samples based on the specified thresholds.
        """
        idx = []
        # Iterate over each track's targets to identify positive and negative samples.
        for i, y in enumerate(self.targets):
            positive = np.where(y > self.positive_threshold)[0]
            n_negative = min(len(y), self.negative_per_positive * max(1, len(positive)))
            negative_pool = np.flatnonzero(y < 0.05)
            negative = (
                self.rng.choice(negative_pool, size=min(n_negative, len(negative_pool)), replace=False)
                if len(negative_pool) > 0 
                else np.array([], dtype=int)
            )
            # Combine positive and negative indices for the current track and append them to the index list.
            for f in np.concatenate([positive, negative]):
                idx.append((i, int(f))) # (Track index, Frame index)
        self.index = idx
        self.rng.shuffle(self.index)

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.
        
        Returns:
            int: Total number of samples.
        """
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset at the specified index.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the mel spectrogram patch and the corresponding target value.
        """
        track_idx, frame_idx = self.index[idx]
        patch = self.specs[track_idx][:, frame_idx:frame_idx + self.width] # (n_mels, width)
        return (
            torch.from_numpy(np.ascontiguousarray(patch)).unsqueeze(0),  # Add channel dimension
            torch.tensor(self.targets[track_idx][frame_idx], dtype=torch.float32)
        )