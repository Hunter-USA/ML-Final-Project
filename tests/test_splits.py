import pandas as pd
import pytest

from sharp import config, splits

from .conftest import needs_real_annotations


def _metadata(titles):
    return pd.DataFrame({"File": [f"{i:04d}_x" for i in range(len(titles))], "Title": titles})


def test_title_group_key_merges_versions():
    k = splits.title_group_key
    assert k("Poker Face") == k("Poker Face (Dave Aude Remix)") == k("Poker Face - Radio Edit")
    assert k("(When You Gonna) Give It Up to Me") != ""


def test_split_is_deterministic_disjoint_and_complete():
    md = _metadata([f"Song {i}" for i in range(100)] + ["Hit", "Hit (Remix)", "Hit (Extended Mix)"])
    a = splits.make_splits(md, seed=1)
    b = splits.make_splits(md, seed=1)
    assert a == b
    all_ids = sum(a.values(), [])
    assert sorted(all_ids) == sorted(md["File"])
    assert len(set(all_ids)) == len(all_ids)
    hit_ids = set(md["File"][-3:])
    assert sum(bool(hit_ids & set(v)) for v in a.values()) == 1     # versions stay together
    assert 60 <= len(a["train"]) <= 80


def test_save_and_load(tmp_path):
    s = {"train": ["a", "b"], "val": ["c"], "test": ["d"]}
    splits.save_splits(s, tmp_path)
    assert splits.load_splits(tmp_path) == s


@pytest.mark.skipif(not splits.splits_exist(config.SPLITS_DIR), reason="no committed splits")
def test_committed_splits_are_disjoint():
    s = splits.load_splits(config.SPLITS_DIR)
    ids = sum(s.values(), [])
    assert len(ids) == len(set(ids)) == 912


@needs_real_annotations
def test_committed_splits_have_no_title_leakage():
    from sharp.annotations import load_metadata
    md = load_metadata()
    key = {f: splits.title_group_key(t, f) for f, t in zip(md["File"], md["Title"])}
    s = splits.load_splits(config.SPLITS_DIR)
    groups = {name: {key[f] for f in ids} for name, ids in s.items()}
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]
