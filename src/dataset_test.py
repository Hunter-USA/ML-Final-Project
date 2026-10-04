"""
Tests for the dataset pipeline in dataset.py.

Run from the project root with:
    python -m pytest src/dataset_test.py
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.dataset import (
    PatchDataset,
    SpecConfig,
    Track,
    build_index,
    gaussian_targets,
    load_spec,
    load_spec_config,
    parse_segments,
)

# Small config so frame math is easy to check by hand: native fps = 100 / 10 = 10.
SMALL = SpecConfig(sr=100, hop_length=10, n_mels=8)


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


def make_track(
        directory: Path,
        name: str,
        spec: np.ndarray,
        boundaries: list[float],
        labels: list[str] | None = None,
) -> Track:
    spec_path = directory / f"{name}.npy"
    np.save(spec_path, spec)
    return Track(
        name=name,
        artist="x",
        spec_path=spec_path,
        boundaries=np.asarray(boundaries, dtype=np.float64),
        labels=labels or [f"seg{i}" for i in range(len(boundaries))],
        duration=spec.shape[1] / SMALL.native_fps,
    )


def normalize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)


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


# ---------------------------------------------------------------------------
# gaussian_targets
# ---------------------------------------------------------------------------

class TestGaussianTargets:
    def test_shape_and_dtype(self):
        y = gaussian_targets(np.array([2.0]), n_frames=50, fps=10.0, sigma_sec=0.5)
        assert y.shape == (50,)
        assert y.dtype == np.float32

    def test_no_boundaries_all_zero(self):
        y = gaussian_targets(np.array([]), n_frames=50, fps=10.0, sigma_sec=0.5)
        np.testing.assert_array_equal(y, np.zeros(50, dtype=np.float32))

    def test_peak_is_one_on_boundary_frame(self):
        y = gaussian_targets(np.array([2.0]), n_frames=50, fps=10.0, sigma_sec=0.5)
        assert y[20] == pytest.approx(1.0)
        assert int(np.argmax(y)) == 20

    def test_one_sigma_away(self):
        # sigma = 0.5 s at 10 fps is 5 frames.
        y = gaussian_targets(np.array([2.0]), n_frames=50, fps=10.0, sigma_sec=0.5)
        assert y[25] == pytest.approx(np.exp(-0.5), rel=1e-5)
        assert y[15] == pytest.approx(np.exp(-0.5), rel=1e-5)

    def test_symmetric_around_boundary(self):
        y = gaussian_targets(np.array([2.0]), n_frames=50, fps=10.0, sigma_sec=0.5)
        np.testing.assert_allclose(y[20 - 10:20], y[20 + 10:20:-1], rtol=1e-6)

    def test_overlapping_boundaries_take_max_not_sum(self):
        y = gaussian_targets(np.array([2.0, 2.2]), n_frames=50, fps=10.0, sigma_sec=0.5)
        assert y.max() <= 1.0 + 1e-6
        assert y[20] == pytest.approx(1.0)
        assert y[22] == pytest.approx(1.0)

    def test_boundary_past_end_is_near_zero(self):
        y = gaussian_targets(np.array([100.0]), n_frames=50, fps=10.0, sigma_sec=0.5)
        assert y.max() < 1e-6

    def test_values_in_unit_range(self):
        y = gaussian_targets(np.array([0.7, 1.3, 3.9]), n_frames=50, fps=10.0, sigma_sec=1.0)
        assert y.min() >= 0.0
        assert y.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# load_spec
# ---------------------------------------------------------------------------

class TestLoadSpec:
    @pytest.mark.parametrize("shape", [(80,), (1, 80, 10), (40, 10)])
    def test_invalid_shape_raises(self, tmp_path, shape):
        p = tmp_path / "s.npy"
        np.save(p, np.zeros(shape, dtype=np.float32))
        with pytest.raises(ValueError, match="Invalid mel spectrogram shape"):
            load_spec(p, SpecConfig(), pool=1)

    def test_output_float32_and_normalized(self, tmp_path):
        p = tmp_path / "s.npy"
        np.save(p, np.random.default_rng(0).random((8, 40)))  # float64 on disk
        spec = load_spec(p, SMALL, pool=1)
        assert spec.dtype == np.float32
        assert spec.mean() == pytest.approx(0.0, abs=1e-5)
        assert spec.std() == pytest.approx(1.0, abs=1e-4)

    def test_non_negative_input_gets_log1p(self, tmp_path):
        raw = np.random.default_rng(0).random((8, 40)).astype(np.float32) * 10
        p = tmp_path / "s.npy"
        np.save(p, raw)
        np.testing.assert_allclose(
            load_spec(p, SMALL, pool=1), normalize(np.log1p(raw)), rtol=1e-4, atol=1e-5
        )

    def test_negative_input_skips_log1p(self, tmp_path):
        raw = np.random.default_rng(0).normal(size=(8, 40)).astype(np.float32)
        assert raw.min() < 0
        p = tmp_path / "s.npy"
        np.save(p, raw)
        np.testing.assert_allclose(load_spec(p, SMALL, pool=1), normalize(raw), rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("pool", [0, 1])
    def test_pool_of_one_or_less_keeps_frames(self, tmp_path, pool):
        p = tmp_path / "s.npy"
        np.save(p, np.random.default_rng(0).random((8, 10)).astype(np.float32))
        assert load_spec(p, SMALL, pool=pool).shape == (8, 10)

    def test_pooling_drops_remainder_and_averages(self, tmp_path):
        raw = np.random.default_rng(0).random((8, 10)).astype(np.float32)
        p = tmp_path / "s.npy"
        np.save(p, raw)
        spec = load_spec(p, SMALL, pool=3)
        assert spec.shape == (8, 3)
        expected = np.log1p(raw)[:, :9].reshape(8, 3, 3).mean(axis=2)
        np.testing.assert_allclose(spec, normalize(expected), rtol=1e-4, atol=1e-5)

    def test_constant_spec_has_no_nan(self, tmp_path):
        p = tmp_path / "s.npy"
        np.save(p, np.ones((8, 10), dtype=np.float32))
        spec = load_spec(p, SMALL, pool=1)
        assert np.isfinite(spec).all()


# ---------------------------------------------------------------------------
# PatchDataset
# ---------------------------------------------------------------------------

@pytest.fixture
def tracks(tmp_path: Path) -> list[Track]:
    rng = np.random.default_rng(0)
    return [
        make_track(tmp_path, "a", rng.random((8, 100)).astype(np.float32), [3.0, 6.0]),
        make_track(tmp_path, "b", rng.random((8, 100)).astype(np.float32), [2.0, 5.0, 8.0]),
    ]


def make_dataset(tracks: list[Track], pool: int = 1, seed: int = 42) -> PatchDataset:
    return PatchDataset(tracks, SMALL, pool=pool, context_sec=1.0, sigma_sec=0.2, seed=seed)


class TestPatchDataset:
    def test_window_geometry(self, tracks):
        ds = make_dataset(tracks)
        # 1.0 s at 10 fps = 10 frames, forced odd -> 11.
        assert ds.width == 11
        assert ds.width % 2 == 1
        assert ds.half_width == ds.width // 2
        assert ds.n_mels == SMALL.n_mels

    @pytest.mark.parametrize("pool", [1, 2])
    def test_fps_accounts_for_pooling(self, tracks, pool):
        ds = make_dataset(tracks, pool=pool)
        assert ds.fps == pytest.approx(SMALL.native_fps / pool)

    @pytest.mark.parametrize("pool", [1, 2])
    def test_specs_padded_and_targets_match_frames(self, tracks, pool):
        ds = make_dataset(tracks, pool=pool)
        for track, spec, y in zip(tracks, ds.specs, ds.targets):
            n_frames = load_spec(track.spec_path, SMALL, pool).shape[1]
            assert spec.shape == (SMALL.n_mels, n_frames + 2 * ds.half_width)
            assert y.shape == (n_frames,)

    def test_names_and_labels_keep_order(self, tracks):
        ds = make_dataset(tracks)
        assert ds.names == ["a", "b"]
        assert ds.labels == [t.labels for t in tracks]

    def test_index_contains_every_positive(self, tracks):
        ds = make_dataset(tracks)
        indexed = set(ds.index)
        for i, y in enumerate(ds.targets):
            positives = np.flatnonzero(y > ds.positive_threshold)
            assert len(positives) > 0
            assert {(i, int(f)) for f in positives} <= indexed

    def test_negatives_are_far_from_boundaries_and_capped(self, tracks):
        ds = make_dataset(tracks)
        for i, y in enumerate(ds.targets):
            n_pos = int((y > ds.positive_threshold).sum())
            negatives = [f for (t, f) in ds.index if t == i and y[f] <= ds.positive_threshold]
            assert all(y[f] < 0.05 for f in negatives)
            assert 0 < len(negatives) <= ds.negative_per_positive * max(1, n_pos)

    def test_no_duplicate_samples(self, tracks):
        ds = make_dataset(tracks)
        assert len(ds.index) == len(set(ds.index))

    def test_len_matches_index(self, tracks):
        ds = make_dataset(tracks)
        assert len(ds) == len(ds.index) > 0

    def test_getitem_shapes_and_target(self, tracks):
        ds = make_dataset(tracks)
        for idx in range(5):
            patch, target = ds[idx]
            t, f = ds.index[idx]
            assert isinstance(patch, torch.Tensor)
            assert patch.dtype == torch.float32
            assert patch.shape == (1, SMALL.n_mels, ds.width)
            assert target.dtype == torch.float32
            assert target.shape == ()
            assert target.item() == pytest.approx(float(ds.targets[t][f]))

    def test_patch_is_centered_on_frame(self, tracks):
        ds = make_dataset(tracks)
        unpadded = [load_spec(t.spec_path, SMALL, 1) for t in tracks]
        for idx in range(10):
            patch, _ = ds[idx]
            t, f = ds.index[idx]
            np.testing.assert_allclose(patch[0, :, ds.half_width].numpy(), unpadded[t][:, f])

    def test_edge_frames_give_full_width_patches(self, tracks):
        ds = make_dataset(tracks)
        last = ds.targets[0].shape[0] - 1
        ds.index = [(0, 0), (0, last)]
        unpadded = load_spec(tracks[0].spec_path, SMALL, 1)
        for idx, f in enumerate([0, last]):
            patch, _ = ds[idx]
            assert patch.shape == (1, SMALL.n_mels, ds.width)
            np.testing.assert_allclose(patch[0, :, ds.half_width].numpy(), unpadded[:, f])

    def test_same_seed_is_reproducible(self, tracks):
        assert make_dataset(tracks, seed=7).index == make_dataset(tracks, seed=7).index

    def test_different_seed_changes_index(self, tracks):
        assert make_dataset(tracks, seed=1).index != make_dataset(tracks, seed=2).index

    def test_resample_keeps_positives_and_reshuffles(self, tracks):
        ds = make_dataset(tracks)
        before = list(ds.index)

        def positives(index):
            return {(t, f) for (t, f) in index if ds.targets[t][f] > ds.positive_threshold}

        ds.resample()
        assert positives(ds.index) == positives(before)
        assert ds.index != before

    def test_pooled_targets_peak_at_pooled_frame(self, tmp_path):
        # Boundary at 2.0 s with pool=2 -> 5 fps -> peak at frame 10 (not 20).
        track = make_track(tmp_path, "p", np.random.default_rng(0).random((8, 100)).astype(np.float32), [2.0])
        ds = make_dataset([track], pool=2)
        y = ds.targets[0]
        assert y.shape == (50,)
        assert int(np.argmax(y)) == 10
        assert y[10] == pytest.approx(1.0)

    def test_works_with_dataloader(self, tracks):
        ds = make_dataset(tracks)
        patches, targets = next(iter(DataLoader(ds, batch_size=4, shuffle=False)))
        assert patches.shape == (4, 1, SMALL.n_mels, ds.width)
        assert targets.shape == (4,)
