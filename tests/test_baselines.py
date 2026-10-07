import numpy as np
import torch

from sharp.labels import CLASS_TO_INDEX, N_CLASSES
from sharp.predictors import FrameModel, MajorityPredictor, context_features, foote_novelty


def _step_features(n=200, at=100, f=64, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.1, size=(n, f)).astype(np.float32)
    x[at:, : f // 2] += 2.0
    x[:at, f // 2:] += 2.0
    return x


def test_context_features_shape_and_change_cue():
    x = _step_features()
    feats = context_features(x)
    assert feats.shape == (200, 3 * 64 + 1)
    delta = np.abs(feats[:, 128:192]).mean(1)
    assert abs(int(np.argmax(delta)) - 100) <= 1          # biggest "change" exactly at the step
    np.testing.assert_allclose(feats[[0, -1], -1], [0.5 / 200, 199.5 / 200])


def test_foote_novelty_peaks_at_change():
    nov = foote_novelty(_step_features(), half_width=16)
    assert nov.max() == 1.0
    assert abs(int(np.argmax(nov)) - 100) <= 2


def test_majority_predictor():
    bp, cp = MajorityPredictor("chorus").predict(np.zeros((30, 64), np.float32))
    assert bp.max() == 0.0
    assert (cp.argmax(1) == CLASS_TO_INDEX["chorus"]).all()


def test_frame_models():
    lr = FrameModel(193, N_CLASSES, hidden=())
    mlp = FrameModel(193, N_CLASSES, hidden=(32, 16))
    x = torch.randn(7, 193)
    for m in (lr, mlp):
        b, s = m(x)
        assert b.shape == (7,) and s.shape == (7, N_CLASSES)
    assert len(list(lr.trunk)) == 0          # logistic regression = linear heads only
