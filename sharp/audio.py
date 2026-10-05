"""Audio -> mel-spectrogram with the same settings as the published Harmonix features.

This is what lets the trained model run on *any* song, not just Harmonix ones.
It reimplements ``librosa.feature.melspectrogram`` as of librosa 0.7 (the version
that made the Harmonix files: Slaney mel scale and normalisation, centred Hann
STFT with reflect padding, power 2) with NumPy/SciPy only, so librosa is not a
dependency. ``tests/test_audio.py`` checks it against librosa when
librosa happens to be installed.
"""

from __future__ import annotations

import io
import shutil
import subprocess
from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window, resample_poly

from . import config


def _to_float(y: np.ndarray) -> np.ndarray:
    if np.issubdtype(y.dtype, np.floating):
        return y.astype(np.float32)
    if y.dtype == np.uint8:
        return (y.astype(np.float32) - 128.0) / 128.0
    info = np.iinfo(y.dtype)
    return y.astype(np.float32) / float(max(abs(info.min), info.max))


def load_audio(path: Path | str, sr: int = config.HARMONIX_SR) -> np.ndarray:
    """Load an audio file as mono float32 at ``sr`` Hz.

    WAV files are read with SciPy. Anything else (mp3, m4a, flac, ...) is decoded
    with ffmpeg if it is installed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".wav":
        file_sr, y = wavfile.read(path)
    else:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                f"Cannot decode {path.suffix} without ffmpeg. Install ffmpeg or convert the song to WAV.")
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-f", "wav", "-acodec", "pcm_s16le",
             "-ac", "1", "-ar", str(sr), "pipe:1"],
            capture_output=True, check=True)
        file_sr, y = wavfile.read(io.BytesIO(proc.stdout))
    y = _to_float(np.asarray(y))
    if y.ndim == 2:
        y = y.mean(axis=1)
    if file_sr != sr:
        g = gcd(int(file_sr), int(sr))
        y = resample_poly(y, sr // g, int(file_sr) // g).astype(np.float32)
    return y


def hz_to_mel(f: np.ndarray) -> np.ndarray:
    """Slaney mel scale (librosa default, htk=False)."""
    f = np.asanyarray(f, dtype=np.float64)
    f_sp = 200.0 / 3
    mels = f / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, mels)


def mel_to_hz(m: np.ndarray) -> np.ndarray:
    m = np.asanyarray(m, dtype=np.float64)
    f_sp = 200.0 / 3
    freqs = f_sp * m
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), freqs)


def mel_filterbank(sr: int = config.HARMONIX_SR, n_fft: int = config.HARMONIX_N_FFT,
                   n_mels: int = config.HARMONIX_N_MELS, fmin: float = config.HARMONIX_FMIN,
                   fmax: float = config.HARMONIX_FMAX) -> np.ndarray:
    """(n_mels, 1 + n_fft // 2) Slaney-normalised triangular filters, as librosa.filters.mel."""
    fft_freqs = np.linspace(0, sr / 2.0, 1 + n_fft // 2)
    mel_pts = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(mel_pts)
    ramps = mel_pts[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_pts[2:n_mels + 2] - mel_pts[:n_mels])
    return weights * enorm[:, None]


def melspectrogram(y: np.ndarray, sr: int = config.HARMONIX_SR, n_fft: int = config.HARMONIX_N_FFT,
                   hop: int = config.HARMONIX_HOP, n_mels: int = config.HARMONIX_N_MELS,
                   fmin: float = config.HARMONIX_FMIN, fmax: float = config.HARMONIX_FMAX,
                   pad_mode: str = config.HARMONIX_PAD_MODE, chunk_frames: int = 2048) -> np.ndarray:
    """Power mel-spectrogram (n_mels, 1 + len(y) // hop) with centred frames."""
    y = np.asarray(y, dtype=np.float64)
    if pad_mode == "reflect" and len(y) <= n_fft // 2:
        pad_mode = "constant"                      # reflect needs a signal longer than the pad
    y = np.pad(y, n_fft // 2, mode=pad_mode)
    frames = np.lib.stride_tricks.sliding_window_view(y, n_fft)[::hop]
    window = get_window("hann", n_fft, fftbins=True)
    fb = mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
    out = np.empty((n_mels, frames.shape[0]), dtype=np.float32)
    for i in range(0, frames.shape[0], chunk_frames):
        spec = np.abs(np.fft.rfft(frames[i:i + chunk_frames] * window, axis=1)) ** 2
        out[:, i:i + chunk_frames] = (fb @ spec.T).astype(np.float32)
    return out


def audio_to_mel(path: Path | str) -> np.ndarray:
    return melspectrogram(load_audio(path))
