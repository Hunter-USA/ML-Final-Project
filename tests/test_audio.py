import numpy as np
import pytest
from scipy.io import wavfile

from sharp import audio, config


def _tone(freq, seconds=2.0, sr=config.HARMONIX_SR):
    t = np.arange(int(sr * seconds)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_filterbank_shape_and_peak():
    fb = audio.mel_filterbank()
    assert fb.shape == (config.HARMONIX_N_MELS, config.HARMONIX_N_FFT // 2 + 1)
    assert (fb >= 0).all()
    mel = audio.melspectrogram(_tone(1000.0))
    band = int(np.argmax(mel.mean(axis=1)))
    centres = audio.mel_to_hz(np.linspace(audio.hz_to_mel(config.HARMONIX_FMIN),
                                          audio.hz_to_mel(config.HARMONIX_FMAX), config.HARMONIX_N_MELS + 2))[1:-1]
    assert abs(centres[band] - 1000.0) < 60


def test_frame_count_matches_librosa_convention():
    y = np.zeros(24000 * 3 + 123, dtype=np.float32)
    assert audio.melspectrogram(y).shape[1] == 1 + len(y) // config.HARMONIX_HOP


def test_mel_scale_roundtrip():
    f = np.array([30.0, 500.0, 1000.0, 4000.0, 12000.0])
    np.testing.assert_allclose(audio.mel_to_hz(audio.hz_to_mel(f)), f, rtol=1e-9)


def test_matches_librosa_if_installed():
    """Same settings librosa 0.7 used for the Harmonix files (reflect padding, fmax = Nyquist)."""
    librosa = pytest.importorskip("librosa")
    rng = np.random.default_rng(0)
    tone = _tone(440.0, 3.0)
    y = (tone + 0.1 * rng.standard_normal(len(tone))).astype(np.float32)
    ours = audio.melspectrogram(y)
    ref = librosa.feature.melspectrogram(
        y=y, sr=config.HARMONIX_SR, n_fft=config.HARMONIX_N_FFT, hop_length=config.HARMONIX_HOP,
        window="hann", center=True, pad_mode="reflect", power=2.0, n_mels=config.HARMONIX_N_MELS,
        fmin=config.HARMONIX_FMIN, fmax=None)
    assert ours.shape == ref.shape
    assert np.max(np.abs(ours - ref)) / ref.max() < 1e-4


def test_load_wav_resamples_and_downmixes(tmp_path):
    sr = 44100
    t = np.arange(sr) / sr
    stereo = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 440 * t)], axis=1)
    path = tmp_path / "tone.wav"
    wavfile.write(path, sr, (stereo * 20000).astype(np.int16))
    y = audio.load_audio(path)
    assert y.ndim == 1
    assert abs(len(y) - config.HARMONIX_SR) <= 1
    assert 0.5 < np.abs(y).max() < 0.7
