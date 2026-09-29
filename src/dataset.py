
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
        raise ValueError(f"No mel spectrograms found in {melspec_dir}")

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