import json

import numpy as np
import pytest

from sharp.data import check_mel_info, find_mel_files

# The info.json shipped inside the real Harmonix_melspecs.tgz
REAL_INFO = {"librosa_version": "0.7.0", "numpy_version": "1.17.2", "SR": 22050, "N_MELS": 80,
             "N_FFT": 2048, "HOP_LENGTH": 1024, "MEL_FMIN": 0, "MEL_FMAX": None}


def test_real_info_json_matches_config(tmp_path):
    (tmp_path / "melspecs").mkdir()
    (tmp_path / "melspecs" / "info.json").write_text(json.dumps(REAL_INFO))
    assert check_mel_info(tmp_path, strict=True)["N_MELS"] == 80


def test_mismatched_info_json_fails_clearly(tmp_path):
    (tmp_path / "info.json").write_text(json.dumps({**REAL_INFO, "N_MELS": 256, "SR": 24000}))
    with pytest.raises(ValueError, match="N_MELS=256"):
        check_mel_info(tmp_path, strict=True)


def test_find_mel_files_skips_macos_metadata(tmp_path):
    d = tmp_path / "melspecs"
    d.mkdir()
    np.save(d / "0001_12step-mel.npy", np.ones((80, 10), np.float32))
    (d / "._0001_12step-mel.npy").write_bytes(b"\x00" * 212)
    files = find_mel_files(tmp_path)
    assert list(files) == ["0001_12step"]
    assert files["0001_12step"].name == "0001_12step-mel.npy"
