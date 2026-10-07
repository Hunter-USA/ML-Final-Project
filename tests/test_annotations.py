import numpy as np
import pytest

from sharp import annotations
from sharp.annotations import parse_segments

from .conftest import needs_real_annotations

BASIC = """0.0 intro
8.5 verse
25.5 chorus
42.5 outro
50.0 end
"""


def test_parse_basic():
    a = parse_segments(BASIC, "x")
    assert a.raw_labels == ["intro", "verse", "chorus", "outro"]
    assert a.end == 50.0
    np.testing.assert_allclose(a.intervals, [[0, 8.5], [8.5, 25.5], [25.5, 42.5], [42.5, 50.0]])
    np.testing.assert_allclose(a.boundary_times, [8.5, 25.5, 42.5, 50.0])
    assert a.classes == ["intro", "verse", "chorus", "outro"]


def test_tabs_and_blank_lines():
    a = parse_segments("0.0\tintro\n\n4.0\tchorus2\n9.0\tend\n", "x")
    assert a.raw_labels == ["intro", "chorus2"]
    assert a.classes == ["intro", "chorus"]


def test_preroll_is_snapped_to_zero():
    a = parse_segments("0.85 intro\n10 verse\n20 end\n", "x")
    assert a.starts[0] == 0.0 and a.raw_labels[0] == "intro"


def test_long_preroll_becomes_silence():
    a = parse_segments("2.95 intro\n10 verse\n20 end\n", "x")
    assert a.raw_labels[0] == "silence"
    np.testing.assert_allclose(a.starts, [0.0, 2.95, 10.0])


def test_duplicate_end_keeps_first():
    a = parse_segments("0 verse\n10 chorus\n20 end\n25 end\n", "x")
    assert a.end == 20.0 and a.raw_labels == ["verse", "chorus"]


def test_missing_end_uses_duration():
    a = parse_segments("0 verse\n10 chorus\n", "x", duration=30.0)
    assert a.end == 30.0
    with pytest.raises(ValueError):
        parse_segments("0 verse\n10 chorus\n", "x")


def test_eval_labels_keep_unmapped_as_raw():
    a = parse_segments("0 section\n10 chorus\n20 end\n", "x")
    assert a.eval_labels() == ["raw:section", "chorus"]


def test_write_roundtrip(tmp_path):
    a = parse_segments(BASIC, "x")
    path = tmp_path / "x.txt"
    annotations.write_segments(path, a.intervals, a.raw_labels)
    b = annotations.load_segments(path)
    np.testing.assert_allclose(a.intervals, b.intervals)
    assert a.raw_labels == b.raw_labels


@needs_real_annotations
def test_all_real_annotations_parse():
    anns = annotations.load_annotations()
    assert len(anns) == 912
    for a in anns.values():
        assert a.starts[0] == 0.0
        assert np.all(np.diff(a.starts) > 0)
        assert a.end > a.starts[-1]
