import numpy as np

from sharp import config
from sharp.annotations import parse_segments
from sharp.features import frame_times, times_to_frames
from sharp.labels import CLASS_TO_INDEX, N_CLASSES
from sharp.metrics import (clip_estimate, confusion_matrix, evaluate_song, macro_f1, per_class_scores,
                           tune_decode_params)
from sharp.postprocess import DecodeParams, decode, merge_repeated_labels, pick_peaks

REF = "0 intro\n10 verse\n30 chorus\n50 verse\n70 chorus\n90 outro\n100 end\n"


def _fake_outputs(ann, duration, noise=0.0, seed=0):
    """Frame-wise outputs that a perfect model would produce for ``ann``."""
    n = int(np.ceil(duration / config.FRAME_HOP_SEC))
    t = frame_times(n)
    bprob = np.zeros(n)
    for b in ann.boundary_times[:-1]:
        bprob[times_to_frames(np.array([b]))[0]] = 0.9
    cprob = np.full((n, N_CLASSES), 0.01)
    seg = np.searchsorted(ann.starts, t, side="right") - 1
    for i, c in enumerate(ann.classes):
        cprob[seg == i, CLASS_TO_INDEX[c]] = 1.0
    cprob /= cprob.sum(1, keepdims=True)
    rng = np.random.default_rng(seed)
    return np.clip(bprob + noise * rng.random(n), 0, 1), cprob


def test_pick_peaks_respects_distance():
    prob = np.zeros(200)
    prob[[50, 55, 120]] = [0.8, 0.9, 0.7]
    peaks = pick_peaks(prob, threshold=0.5, min_distance_sec=3.0)
    assert list(peaks) == [55, 120]


def test_decode_perfect_outputs():
    ann = parse_segments(REF, "x")
    bprob, cprob = _fake_outputs(ann, 100.0)
    iv, lab = decode(bprob, cprob, DecodeParams(0.5, 3.0), duration=100.0)
    assert lab == ["intro", "verse", "chorus", "verse", "chorus", "outro"]
    np.testing.assert_allclose(iv[1:, 0], [10, 30, 50, 70, 90], atol=config.FRAME_HOP_SEC)


def test_perfect_prediction_scores_one():
    ann = parse_segments(REF, "x")
    m, r, e = evaluate_song(ann, ann.intervals, ann.classes)
    for k in ("HitRate_0.5F", "HitRate_3F", "HitRate_t0.5F", "PWF", "Sf", "LabelAcc"):
        assert abs(m[k] - 1.0) < 1e-9, k
    assert (r == e).all()


def test_single_segment_baseline():
    ann = parse_segments(REF, "x")
    m, _, _ = evaluate_song(ann, np.array([[0.0, 120.0]]), ["chorus"])   # longer than the annotation
    assert m["HitRate_t3F"] == 0.0
    assert abs(m["LabelAcc"] - 0.4) < 0.01              # chorus covers 40 of 100 s


def test_clip_estimate():
    iv, lab = clip_estimate(np.array([[0, 40], [40, 90], [90, 130]]), ["a", "b", "c"], 80.0)
    np.testing.assert_allclose(iv, [[0, 40], [40, 80]])
    assert lab == ["a", "b"]


def test_confusion_and_f1():
    cm = confusion_matrix(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), n=N_CLASSES)
    assert cm.sum() == 4 and cm[0, 1] == 1
    df = per_class_scores(cm)
    assert abs(df.loc[0, "precision"] - 1.0) < 1e-9 and abs(df.loc[0, "recall"] - 0.5) < 1e-9
    assert 0 < macro_f1(cm) < 1


def test_tuning_finds_working_threshold():
    ann = parse_segments(REF, "x")
    songs = [(ann, _fake_outputs(ann, 100.0, noise=0.3, seed=s)[0], 100.0) for s in range(3)]
    params, table = tune_decode_params(songs)
    assert 0.3 <= params.threshold < 0.9
    assert table["score"].max() > 0.9


def test_merge_repeated_labels():
    iv, lab = merge_repeated_labels(np.array([[0, 10], [10, 20], [20, 30]]), ["chorus", "chorus", "outro"])
    np.testing.assert_allclose(iv, [[0, 20], [20, 30]])
    assert lab == ["chorus", "outro"]
