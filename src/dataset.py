
"""
This File represents the dataset pipeline.

It requires:

1. The annotation repo: git clone https://github.com/urinieto/harmonixset
    -> dataset/segments/*.txt
    -> dataset/metadata.csv

2. The mel spectrograms Harmonix_melspecs.tgz linked from the repo and untarred anywhere.
"""

from dataclasses import dataclass


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