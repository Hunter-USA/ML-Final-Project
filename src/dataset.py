
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
from pathlib import Path

@dataclass
class SpecConfig:
    """
    Configuration for spectrogram
    """
    sr: int = 22050
    hop_length: int = 1024
    n_mels: int = 80

    @property
    def native_fps(self) -> float:
        return self.sr / self.hop_length


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
        order = np.argsort(times)
        times = np.asarray(times, dtype=np.float64)[order]
        labels = [labels[i] for i in order]
        

    return np.array(times), labels