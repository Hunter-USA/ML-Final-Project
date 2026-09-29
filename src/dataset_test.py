"""
Tests for the dataset pipeline in dataset.py.

Run from the project root with:
    python -m pytest src/dataset_test.py
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.dataset import (
    SpecConfig,
    Track,
    build_index,
    load_spec_config,
    parse_segments,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_segments(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n")
    return path


def write_metadata(dataset_dir: Path, rows: list[dict], header: list[str] | None = None) -> None:
    header = header or ["File", "Artist", "Duration"]
    out = [",".join(header)]
    for row in rows:
        out.append(",".join(str(row.get(h.strip(), "")) for h in header))
    (dataset_dir / "metadata.csv").write_text("\n".join(out) + "\n")


def write_spec(melspec_dir: Path, stem: str) -> Path:
    p = melspec_dir / f"{stem}.npy"
    np.save(p, np.zeros((80, 10), dtype=np.float32))
    return p


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "segments").mkdir(parents=True)
    melspec_dir = tmp_path / "melspecs"
    melspec_dir.mkdir()
    return dataset_dir, melspec_dir


# ---------------------------------------------------------------------------
# SpecConfig
# ---------------------------------------------------------------------------

class TestSpecConfig:
    def test_defaults(self):
        cfg = SpecConfig()
        assert cfg.sr == 22050
        assert cfg.hop_length == 1024
        assert cfg.n_mels == 80

    def test_native_fps_default(self):
        assert SpecConfig().native_fps == pytest.approx(22050 / 1024)

    def test_native_fps_custom(self):
        assert SpecConfig(sr=16000, hop_length=160).native_fps == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# load_spec_config
# ---------------------------------------------------------------------------

class TestLoadSpecConfig:
    def test_override_is_returned_as_is(self, tmp_path):
        # Even with an info.json present, the override wins.
        (tmp_path / "info.json").write_text(json.dumps({"SR": 44100}))
        override = SpecConfig(sr=8000, hop_length=256, n_mels=40)
        assert load_spec_config(tmp_path, override) is override

    def test_missing_info_json_returns_defaults(self, tmp_path, capsys):
        cfg = load_spec_config(tmp_path)
        assert cfg == SpecConfig()
        assert "[WARNING]" in capsys.readouterr().out

    def test_reads_all_fields(self, tmp_path):
        (tmp_path / "info.json").write_text(
            json.dumps({"SR": 44100, "HOP_LENGTH": 512, "N_MELS": 128})
        )
        cfg = load_spec_config(tmp_path)
        assert cfg == SpecConfig(sr=44100, hop_length=512, n_mels=128)
        assert cfg.native_fps == pytest.approx(44100 / 512)

    def test_partial_info_keeps_defaults(self, tmp_path):
        (tmp_path / "info.json").write_text(json.dumps({"HOP_LENGTH": 256}))
        cfg = load_spec_config(tmp_path)
        assert cfg.sr == 22050
        assert cfg.hop_length == 256
        assert cfg.n_mels == 80

    def test_unknown_keys_ignored(self, tmp_path):
        (tmp_path / "info.json").write_text(json.dumps({"FOO": 1, "sr": 1}))
        assert load_spec_config(tmp_path) == SpecConfig()

    def test_does_not_mutate_class_defaults(self, tmp_path):
        (tmp_path / "info.json").write_text(json.dumps({"SR": 1}))
        load_spec_config(tmp_path)
        assert SpecConfig().sr == 22050


# ---------------------------------------------------------------------------
# parse_segments
# ---------------------------------------------------------------------------

class TestParseSegments:
    def test_basic_parse_drops_zero_and_end(self, tmp_path):
        p = write_segments(tmp_path / "s.txt", [
            "0.0 intro",
            "10.5 verse",
            "30.25 chorus",
            "60.0 end",
        ])
        times, labels = parse_segments(p)
        np.testing.assert_array_equal(times, [10.5, 30.25])
        assert labels == ["verse", "chorus"]

    def test_returns_float64_array(self, tmp_path):
        p = write_segments(tmp_path / "s.txt", ["1 a", "2 b"])
        times, _ = parse_segments(p)
        assert isinstance(times, np.ndarray)
        assert times.dtype == np.float64

    def test_labels_lowercased(self, tmp_path):
        p = write_segments(tmp_path / "s.txt", ["1.0 VERSE", "2.0 Chorus"])
        _, labels = parse_segments(p)
        assert labels == ["verse", "chorus"]

    def test_end_label_case_insensitive(self, tmp_path):
        p = write_segments(tmp_path / "s.txt", ["1.0 verse", "2.0 END", "3.0 End"])
        times, labels = parse_segments(p)
        np.testing.assert_array_equal(times, [1.0])
        assert labels == ["verse"]

    def test_unsorted_input_is_sorted_with_labels_paired(self, tmp_path):
        p = write_segments(tmp_path / "s.txt", [
            "30.0 chorus",
            "10.0 verse",
            "20.0 bridge",
        ])
        times, labels = parse_segments(p)
        np.testing.assert_array_equal(times, [10.0, 20.0, 30.0])
        assert labels == ["verse", "bridge", "chorus"]

    def test_skips_blank_and_short_lines(self, tmp_path):
        p = write_segments(tmp_path / "s.txt", [
            "",
            "5.0",
            "   ",
            "10.0 verse",
        ])
        times, labels = parse_segments(p)
        np.testing.assert_array_equal(times, [10.0])
        assert labels == ["verse"]

    def test_skips_non_numeric_times_with_warning(self, tmp_path, capsys):
        p = write_segments(tmp_path / "s.txt", [
            "abc verse",
            "10.0 chorus",
        ])
        times, labels = parse_segments(p)
        np.testing.assert_array_equal(times, [10.0])
        assert labels == ["chorus"]
        assert "[WARNING]" in capsys.readouterr().out

    def test_tab_separated_and_extra_columns(self, tmp_path):
        p = write_segments(tmp_path / "s.txt", [
            "1.5\tverse\textra",
            "2.5\tchorus",
        ])
        times, labels = parse_segments(p)
        np.testing.assert_array_equal(times, [1.5, 2.5])
        assert labels == ["verse", "chorus"]

    def test_empty_file(self, tmp_path):
        p = tmp_path / "s.txt"
        p.write_text("")
        times, labels = parse_segments(p)
        assert times.shape == (0,)
        assert labels == []

    def test_only_zero_and_end(self, tmp_path):
        p = write_segments(tmp_path / "s.txt", ["0.0 intro", "100.0 end"])
        times, labels = parse_segments(p)
        assert times.shape == (0,)
        assert labels == []

    def test_times_and_labels_same_length(self, tmp_path):
        p = write_segments(tmp_path / "s.txt", [
            "0 silence", "3 intro", "bad x", "7 verse", "9", "12 end", "11 outro",
        ])
        times, labels = parse_segments(p)
        assert len(times) == len(labels)
        np.testing.assert_array_equal(times, [3.0, 7.0, 11.0])
        assert labels == ["intro", "verse", "outro"]


# ---------------------------------------------------------------------------
# build_index
# ---------------------------------------------------------------------------

class TestBuildIndex:
    def test_builds_track(self, dirs):
        dataset_dir, melspec_dir = dirs
        write_segments(dataset_dir / "segments" / "0001_song.txt",
                       ["0.0 intro", "5.0 verse", "20.0 chorus", "40.0 end"])
        write_metadata(dataset_dir, [{"File": "0001_song", "Artist": "The Band", "Duration": 42.0}])
        spec = write_spec(melspec_dir, "0001_song")

        tracks = build_index(dataset_dir, melspec_dir)

        assert len(tracks) == 1
        t = tracks[0]
        assert isinstance(t, Track)
        assert t.name == "0001_song"
        assert t.artist == "the band"
        assert t.spec_path == spec
        np.testing.assert_array_equal(t.boundaries, [5.0, 20.0])
        assert t.labels == ["verse", "chorus"]
        assert t.duration == pytest.approx(42.0)
        assert isinstance(t.duration, float)

    def test_matches_mel_suffixed_spec(self, dirs):
        dataset_dir, melspec_dir = dirs
        write_segments(dataset_dir / "segments" / "0002_song.txt", ["1.0 a", "2.0 b"])
        write_metadata(dataset_dir, [{"File": "0002_song", "Artist": "X", "Duration": 3}])
        spec = write_spec(melspec_dir, "0002_song-mel")

        tracks = build_index(dataset_dir, melspec_dir)

        assert len(tracks) == 1
        assert tracks[0].spec_path == spec

    def test_exact_stem_preferred_over_prefix(self, dirs):
        dataset_dir, melspec_dir = dirs
        write_segments(dataset_dir / "segments" / "0003_song.txt", ["1.0 a", "2.0 b"])
        write_metadata(dataset_dir, [{"File": "0003_song", "Artist": "X", "Duration": 3}])
        write_spec(melspec_dir, "0003_song-mel")
        exact = write_spec(melspec_dir, "0003_song")

        tracks = build_index(dataset_dir, melspec_dir)

        assert tracks[0].spec_path == exact

    def test_no_specs_raises(self, dirs):
        dataset_dir, melspec_dir = dirs
        write_segments(dataset_dir / "segments" / "a.txt", ["1.0 a", "2.0 b"])
        write_metadata(dataset_dir, [{"File": "a", "Artist": "X", "Duration": 3}])

        with pytest.raises(ValueError, match="No mel spectrograms"):
            build_index(dataset_dir, melspec_dir)

    def test_missing_spec_skipped(self, dirs, capsys):
        dataset_dir, melspec_dir = dirs
        write_segments(dataset_dir / "segments" / "a.txt", ["1.0 a", "2.0 b"])
        write_segments(dataset_dir / "segments" / "b.txt", ["1.0 a", "2.0 b"])
        write_metadata(dataset_dir, [
            {"File": "a", "Artist": "X", "Duration": 3},
            {"File": "b", "Artist": "Y", "Duration": 3},
        ])
        write_spec(melspec_dir, "a")

        tracks = build_index(dataset_dir, melspec_dir)

        assert [t.name for t in tracks] == ["a"]
        out = capsys.readouterr().out
        assert "Missing mel spectrogram for b" in out
        assert "Missing spectrograms: 1" in out

    def test_missing_metadata_skipped(self, dirs, capsys):
        dataset_dir, melspec_dir = dirs
        write_segments(dataset_dir / "segments" / "a.txt", ["1.0 a", "2.0 b"])
        write_segments(dataset_dir / "segments" / "b.txt", ["1.0 a", "2.0 b"])
        write_metadata(dataset_dir, [{"File": "a", "Artist": "X", "Duration": 3}])
        write_spec(melspec_dir, "a")
        write_spec(melspec_dir, "b")

        tracks = build_index(dataset_dir, melspec_dir)

        assert [t.name for t in tracks] == ["a"]
        out = capsys.readouterr().out
        assert "Missing metadata for b" in out
        assert "Missing metadata: 1" in out

    def test_too_few_segments_skipped(self, dirs, capsys):
        dataset_dir, melspec_dir = dirs
        # Only one boundary survives filtering (0.0 and "end" are dropped).
        write_segments(dataset_dir / "segments" / "a.txt", ["0.0 intro", "5.0 verse", "9.0 end"])
        write_metadata(dataset_dir, [{"File": "a", "Artist": "X", "Duration": 9}])
        write_spec(melspec_dir, "a")

        tracks = build_index(dataset_dir, melspec_dir)

        assert tracks == []
        assert "Not enough segments for a" in capsys.readouterr().out

    def test_tracks_sorted_by_name(self, dirs):
        dataset_dir, melspec_dir = dirs
        names = ["c_track", "a_track", "b_track"]
        for n in names:
            write_segments(dataset_dir / "segments" / f"{n}.txt", ["1.0 a", "2.0 b"])
            write_spec(melspec_dir, n)
        write_metadata(dataset_dir, [{"File": n, "Artist": "X", "Duration": 3} for n in names])

        tracks = build_index(dataset_dir, melspec_dir)

        assert [t.name for t in tracks] == sorted(names)

    def test_metadata_column_whitespace_stripped(self, dirs):
        dataset_dir, melspec_dir = dirs
        write_segments(dataset_dir / "segments" / "a.txt", ["1.0 a", "2.0 b"])
        write_metadata(
            dataset_dir,
            [{"File": "a", "Artist": "Someone", "Duration": 7.5}],
            header=[" File", " Artist ", "Duration "],
        )
        write_spec(melspec_dir, "a")

        tracks = build_index(dataset_dir, melspec_dir)

        assert len(tracks) == 1
        assert tracks[0].artist == "someone"
        assert tracks[0].duration == pytest.approx(7.5)

    def test_artist_whitespace_stripped_and_lowercased(self, dirs):
        dataset_dir, melspec_dir = dirs
        write_segments(dataset_dir / "segments" / "a.txt", ["1.0 a", "2.0 b"])
        (dataset_dir / "metadata.csv").write_text('File,Artist,Duration\na,"  MiXeD Case  ",3\n')
        write_spec(melspec_dir, "a")

        tracks = build_index(dataset_dir, melspec_dir)

        assert tracks[0].artist == "mixed case"

    def test_missing_artist_column_defaults_to_unknown(self, dirs):
        dataset_dir, melspec_dir = dirs
        write_segments(dataset_dir / "segments" / "a.txt", ["1.0 a", "2.0 b"])
        write_metadata(dataset_dir, [{"File": "a", "Duration": 3}], header=["File", "Duration"])
        write_spec(melspec_dir, "a")

        tracks = build_index(dataset_dir, melspec_dir)

        assert tracks[0].artist == "unknown"

    def test_missing_duration_column_falls_back_to_last_boundary(self, dirs):
        dataset_dir, melspec_dir = dirs
        write_segments(dataset_dir / "segments" / "a.txt", ["1.0 a", "2.0 b", "12.5 c"])
        write_metadata(dataset_dir, [{"File": "a", "Artist": "X"}], header=["File", "Artist"])
        write_spec(melspec_dir, "a")

        tracks = build_index(dataset_dir, melspec_dir)

        assert tracks[0].duration == pytest.approx(12.5)

    def test_ignores_non_txt_segment_files(self, dirs):
        dataset_dir, melspec_dir = dirs
        write_segments(dataset_dir / "segments" / "a.txt", ["1.0 a", "2.0 b"])
        (dataset_dir / "segments" / "README.md").write_text("not a segment file")
        write_metadata(dataset_dir, [{"File": "a", "Artist": "X", "Duration": 3}])
        write_spec(melspec_dir, "a")

        tracks = build_index(dataset_dir, melspec_dir)

        assert [t.name for t in tracks] == ["a"]

    def test_summary_line_printed(self, dirs, capsys):
        dataset_dir, melspec_dir = dirs
        write_segments(dataset_dir / "segments" / "a.txt", ["1.0 a", "2.0 b"])
        write_metadata(dataset_dir, [{"File": "a", "Artist": "X", "Duration": 3}])
        write_spec(melspec_dir, "a")

        build_index(dataset_dir, melspec_dir)

        assert "1 tracks built" in capsys.readouterr().out
