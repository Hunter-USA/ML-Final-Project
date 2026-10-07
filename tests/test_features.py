import numpy as np

from sharp import config
from sharp.annotations import parse_segments
from sharp.features import (compute_stats, frame_times, make_targets, pool_mel, times_to_frames)
from sharp.labels import CLASS_TO_INDEX, IGNORE_INDEX


def test_pool_mel_shapes_and_orientation():
    rng = np.random.default_rng(0)
    mel = rng.random((config.HARMONIX_N_MELS, 101)) + 1e-3
    out = pool_mel(mel)
    assert out.shape == (26, config.N_BANDS)          # ceil(101 / 4) frames
    assert out.dtype == np.float32
    np.testing.assert_allclose(pool_mel(mel.T), out)  # (time, mels) input is accepted


def test_pool_mel_db_floor():
    mel = np.full((config.HARMONIX_N_MELS, 40), 1.0)
    mel[:, :20] = 1e-30
    out = pool_mel(mel)
    assert out.max() - out.min() <= config.TOP_DB + 1e-4
    np.testing.assert_allclose(out[-1], 0.0, atol=1e-5)    # power 1 -> 0 dB


def test_frame_time_roundtrip():
    t = frame_times(500)
    np.testing.assert_array_equal(times_to_frames(t), np.arange(500))
    assert abs((t[1] - t[0]) - config.FRAME_HOP_SEC) < 1e-9


def test_make_targets():
    ann = parse_segments("0 intro\n10 verse\n20 section\n30 chorus\n40 end\n", "x")
    n = int(50 / config.FRAME_HOP_SEC)
    section, boundary = make_targets(ann, n, sigma=1.0)
    t = frame_times(n)
    assert section[np.argmin(abs(t - 5))] == CLASS_TO_INDEX["intro"]
    assert section[np.argmin(abs(t - 15))] == CLASS_TO_INDEX["verse"]
    assert section[np.argmin(abs(t - 25))] == IGNORE_INDEX          # unmapped label
    assert section[np.argmin(abs(t - 35))] == CLASS_TO_INDEX["chorus"]
    assert section[np.argmin(abs(t - 45))] == IGNORE_INDEX          # after "end"
    for b in (10, 20, 30, 40):
        assert boundary[times_to_frames(np.array([b]))[0]] == 1.0
    assert boundary[np.argmin(abs(t - 5))] == 0.0
    assert boundary.max() <= 1.0


def test_hard_targets_with_zero_sigma():
    ann = parse_segments("0 intro\n10 verse\n20 end\n", "x")
    _, boundary = make_targets(ann, 200, sigma=0)
    assert set(np.unique(boundary)) <= {0.0, 1.0}
    assert boundary.sum() == 2


def test_stats_standardise():
    rng = np.random.default_rng(0)
    arrays = [rng.normal(5, 3, size=(100, 8)) for _ in range(5)]
    stats = compute_stats(arrays)
    z = np.concatenate([stats.apply(a) for a in arrays])
    np.testing.assert_allclose(z.mean(0), 0, atol=1e-4)
    np.testing.assert_allclose(z.std(0), 1, atol=1e-3)
