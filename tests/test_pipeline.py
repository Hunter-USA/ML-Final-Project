"""End-to-end run on a tiny synthetic dataset: prepare -> train -> baselines -> evaluate -> predict."""

import json

import numpy as np
import pandas as pd
from scipy.io import wavfile

from sharp import baselines, evaluate, predict, prepare, train
from sharp.config import TrainConfig
from sharp.data import find_mel_files


def test_full_pipeline(fake_dataset, tmp_path):
    d = fake_dataset
    prepare.prepare(d["annotation_dir"], d["mel_dir"], d["cache_dir"], d["splits_dir"])
    assert (d["cache_dir"] / "stats.npz").exists()
    assert (d["splits_dir"] / "train.txt").exists()

    cfg = TrainConfig(epochs=2, batch_size=4, conv_channels=(8, 8, 8), gru_hidden=16, gru_layers=1,
                      max_frames=200, num_threads=2)
    ckpt = train.train(cfg, d["annotation_dir"], d["cache_dir"], d["splits_dir"], d["runs_dir"], verbose=False)
    assert ckpt.exists()
    assert (d["runs_dir"] / "cnn_bigru" / "history.csv").exists()

    baselines.run(("majority", "novelty", "logreg", "mlp"), d["annotation_dir"], d["cache_dir"],
                  d["splits_dir"], d["runs_dir"], epochs=2, verbose=False)
    for f in ("majority.json", "novelty.json", "logreg.pt", "mlp.pt"):
        assert (d["runs_dir"] / "baselines" / f).exists()

    summary = evaluate.run("test", d["annotation_dir"], d["cache_dir"], d["splits_dir"], d["runs_dir"],
                           d["results_dir"], n_examples=1)
    assert set(summary["method"]) == {"cnn_bigru", "mlp", "logreg", "novelty", "majority"}
    for f in ("summary_test.csv", "summary_test.md", "per_song_test.csv", "per_class_test.csv",
              "significance_test.csv"):
        assert (d["results_dir"] / f).exists(), f
    per_song = pd.read_csv(d["results_dir"] / "per_song_test.csv")
    assert per_song[["HitRate_3F", "PWF", "LabelAcc"]].notna().all().all()
    assert list((d["results_dir"] / "figures").rglob("*.png"))

    # predict from a Harmonix-style mel file and from a WAV file
    mel_path = sorted(find_mel_files(d["mel_dir"]).values())[0]
    rng = np.random.default_rng(0)
    wav = tmp_path / "noise.wav"
    wavfile.write(wav, 22050, (rng.standard_normal(22050 * 20) * 3000).astype(np.int16))
    out_dir = tmp_path / "pred"
    predict.main([str(mel_path), str(wav), "--checkpoint", str(ckpt), "--out-dir", str(out_dir)])
    for stem in (mel_path.stem.removesuffix("-mel"), "noise"):
        assert (out_dir / f"{stem}.segments.txt").exists()
        assert (out_dir / f"{stem}.png").exists()
        info = json.loads((out_dir / f"{stem}.json").read_text())
        assert info["segments"][0]["start"] == 0.0
