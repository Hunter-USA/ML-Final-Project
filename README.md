# Learning the Shape of a Song

**Team Sharp · CS 4342 Machine Learning · Final Project**
Maddox Neddo · Hunter Boles · Arlo Dzik · Matthew Nickerson · Narongtat "Frankie" Wongwattanakij

Any song in, a labelled map of its structure out. A **CNN + BiGRU** reads a song's mel-spectrogram and predicts, for every ~0.19 s frame, **(1) whether a new section starts there** and **(2) what type of section it is**. Post-processing turns those frame predictions into a segmentation like `intro → verse → pre-chorus → chorus → …`, which is scored against professional annotations with **mir_eval** and compared with **majority-class, logistic-regression and MLP baselines** (plus a classic unsupervised method for context).

```
mel-spectrogram ──► CNN (local sound patterns) ──► BiGRU (whole-song context, repetition)
                                                     ├─► P(section starts here)   ─► peak picking ─► boundaries
                                                     └─► P(section type)          ─► vote per segment ─► labels
```

---

## Quick start

Python 3.12 or newer (the pinned NumPy/SciPy versions need it). From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest                       # ~10 s, needs no data

python -m sharp.download               # annotations (GitHub) + 1.2 GB mel-spectrograms (Dropbox)
python -m sharp.prepare                # cached features, normalisation stats
python -m sharp.train                  # CNN + BiGRU          -> runs/cnn_bigru/best.pt
python -m sharp.baselines              # 4 baselines          -> runs/baselines/
python -m sharp.evaluate --split val   # while developing     -> results/
python -m sharp.evaluate               # test split, once, for the final numbers
python -m sharp.predict path/to/song.wav
```

If the Dropbox download fails, open the `Harmonix_melspecs.tgz` link in the [Harmonix README](https://github.com/urinieto/harmonixset#audio-data-updated-december-2020) in a browser, then run
`python -m sharp.download --mel-archive ~/Downloads/Harmonix_melspecs.tgz`. Downloading the spectrograms means agreeing to the license inside the archive.

Every script has `--help`. All paths and hyperparameters can be overridden from the command line, e.g.
`python -m sharp.train --lr 5e-4 --gru-layers 3 --run-name bigger --device cuda`.

---

## Data: the Harmonix Set

[Harmonix Set v1.2](https://github.com/urinieto/harmonixset) (Nieto et al., ISMIR 2019): 912 Western pop songs, 56 hours, ~9,700 sections annotated by professional musicians. Audio is not distributed; the authors publish mel-spectrograms instead, which is what we train on. Their `info.json` gives the settings: librosa 0.7, 22.05 kHz, 2048-point FFT, hop 1024, 80 mel bands from 0 to 11.025 kHz, stored as power (not dB). The archive also holds a macOS `._*` metadata file next to every spectrogram; the code ignores those. `sharp/audio.py` reproduces exactly that transform with NumPy/SciPy (checked against librosa in the tests), so the trained model also runs on new audio files.

**Features** (`sharp/features.py`). All 80 mel bands are kept, frames are averaged in blocks of 4 (one model frame = 0.186 s), converted to dB with an 80 dB floor, and standardised per band with training-set statistics.

**Section types** (`sharp/labels.py`). The annotators used 126 different strings, so we fold them into 10 functional classes. Shares are of labelled frames in the training split:

| Class | Share | Raw Harmonix labels folded in |
|---|---:|---|
| silence | 0.5% | `silence`, plus any gap over 1 s before the first annotated section |
| intro | 7.2% | `intro*`, `fadein`, `opening`, `instintro`, `rhythmlessintro`, `introverse`, `introchorus` |
| verse | 27.1% | `verse*`, `miniverse`, `versepart`, `slowverse`, `preverse`, `postverse`, `raps` |
| pre-chorus | 5.3% | `prechorus*`, `build` |
| chorus | 35.9% | `chorus*`, `altchorus`, `quietchorus`, `refrain`, instrumental choruses (`instchorus`, `chorusinst`, …) |
| post-chorus | 3.4% | `postchorus*` |
| bridge | 8.6% | `bridge*`, `instbridge` |
| solo / instrumental | 6.2% | `inst*`, `instrumental*`, `solo*`, `gtr*`, `guitarsolo`, `mainriff`, `synth`, `saxobeat`, … |
| break | 1.7% | `break*`, `breakdown*`, `transition*`, `stutter`, `quiet`, `slow` |
| outro | 4.1% | `outro*`, `bigoutro`, `vocaloutro` |
| *(ignored)* | – | `section*`, `worstthingever`, `fast`: no functional meaning. Their boundaries still count. |

Chorus is the majority class (35.9%), which is what the majority baseline predicts everywhere.

**Annotation quirks handled** (`sharp/annotations.py`):
* Many songs start at the first beat (e.g. 0.85 s). A first boundary under 1 s is snapped to 0; a later one gets a leading `silence` segment.
* 4 files have two `end` rows (we keep the first) and 2 have none (we use the track duration).
* In 716 songs the audio runs more than 2 s past `end` (fade-outs; the median gap over all songs is 4.4 s). That tail is unannotated, so it is excluded from the label loss and from evaluation.

**Splits** (`splits/*.txt`, committed so the whole team uses the same ones): 639 train / 137 validation / 136 test songs. Harmonix contains several versions of the same song (radio edits, extended mixes, remixes: 75 title groups, e.g. three versions of *Poker Face*). Songs are grouped by normalised title, so no song appears in two splits. **Only evaluate on `test` once, for the final report.** Tune everything on `val`.

---

## Model (`sharp/model.py`)

| Stage | Details | Output |
|---|---|---|
| Input | standardised log-mel, 80 bands × T frames | (B, T, 80) |
| 3 conv blocks | each: 2 × [3×3 conv → BatchNorm → ReLU], max-pool over frequency only (80 → 20 → 5 → 1 bands), dropout; 16/32/64 channels | (B, 64, 1, T) |
| Projection | linear 64 → 128, ReLU, dropout | (B, T, 128) |
| BiGRU | 2 layers, 128 units per direction, over the whole song | (B, T, 256) |
| Boundary head | linear → sigmoid: P(section starts at this frame) | (B, T) |
| Section head | linear → softmax over 10 section types | (B, T, 10) |

About 578k parameters. The convolutions see about ±1 s of audio. The BiGRU gives every frame the context of the whole song, which is how the model can tell a chorus from a verse that sounds alike: the chorus is the part that keeps coming back. The BiGRU handles padded batches exactly (each song is reversed within its own length for the backward direction). The tests check this against PyTorch's packed GRU, and it is about 4× faster on CPU.

**Training** (`sharp/train.py`, defaults in `sharp/config.py:TrainConfig`):
* Loss = class-weighted cross-entropy on section type (weights ∝ 1/√frequency, label smoothing 0.05) + BCE on boundaries. Boundary targets are Gaussian bumps (σ = 1 frame) around each annotated boundary, with positives up-weighted ×5 because boundaries are rare.
* AdamW (lr 1e-3, weight decay 1e-4), batch 8 songs, random 5-minute crops, light SpecAugment (frequency masks + gain), gradient clipping.
* The learning rate halves when validation loss plateaus. Early stopping (patience 8) keeps the epoch with the lowest validation loss.

**From frames to sections** (`sharp/postprocess.py`). Boundaries are peaks of the boundary curve (`scipy.signal.find_peaks`) above a threshold and at least *d* seconds apart. Each segment takes the section type with the highest average probability over its frames. The threshold and *d* are grid-searched on the validation split (objective: mean of HR.5F and HR3F) and stored in the checkpoint.

---

## Baselines (`sharp/baselines.py`, `sharp/predictors.py`)

| Method | Boundaries | Section types |
|---|---|---|
| **Majority class** | none (one segment per song) | always `chorus` |
| **Logistic regression** | binary logistic regression per frame | multinomial logistic regression per frame |
| **MLP** | 2 hidden layers (256, 128), two output heads | same network |
| Foote novelty *(extra, unsupervised)* | checkerboard kernel on a self-similarity matrix | majority class |

Logistic regression and the MLP get per-frame features: the frame, the mean of ±8 frames (±1.5 s), the difference between the next and previous 8 frames (a cheap "did the sound just change?" cue), and the frame's relative position in the song. That lets them learn that intros come first and outros last. Every baseline goes through the same peak picking, with its own threshold tuned on validation. So the comparison isolates what the CNN + BiGRU adds: learned local patterns plus whole-song context.

---

## Evaluation (`sharp/evaluate.py`, `sharp/metrics.py`)

Per song, over the annotated span `[0, end]`:

| Metric | Meaning |
|---|---|
| **HR.5F / HR3F** | boundary hit-rate F-measure: a predicted boundary counts if within ±0.5 s / ±3 s of an expert one (`mir_eval.segment.detection`) |
| HR.5F / HR3F (trim) | same, ignoring the trivial first and last boundary |
| **PWF** | pairwise frame-clustering F: do two frames share a label in both segmentations? (`mir_eval.segment.pairwise`) |
| **Sf** | normalised-conditional-entropy F (over-/under-segmentation balance, `mir_eval.segment.nce`) |
| **Label acc** | share of 0.1 s frames whose predicted section type equals the expert's |
| **Macro-F1** | per-class F1 averaged over the 10 section types (so rare types like pre-chorus count as much as chorus) |

`python -m sharp.evaluate` writes to `results/`:
* `summary_<split>.md/.csv`: one row per method, ready for the report. It also includes the unsupervised MSAF results published with Harmonix on the same songs, for context.
* `per_song_<split>.csv`, `per_class_<split>.csv`
* `significance_<split>.csv`: paired Wilcoxon signed-rank tests, CNN + BiGRU vs each baseline, per metric
* `figures/`: confusion matrices, training curves, and expert-vs-predicted structure strips for a few songs

`python -m sharp.predict song.wav` prints the sections and writes `predictions/<song>.segments.txt` (Harmonix format), `.json` (with the boundary curve) and a `.png` strip like the one on our pitch slide. It accepts `.wav` directly, other formats if `ffmpeg` is installed, and Harmonix `*-mel.npy` files. Add `--reference <segments.txt>` to plot the expert labels underneath.

---

## Runtime

Measured at full scale (912 songs) on a 2-core cloud CPU with no GPU:

| Step | Time |
|---|---|
| `prepare` | under a minute once the archive is extracted (feature cache is ~150 MB) |
| `train`, one epoch (639 songs) | ~3.5 min, so a full run with early stopping is 1–2 h |
| `baselines` (all four, including tuning) | ~1 min |
| `evaluate` (all methods, 136 songs) | ~40 s |
| `predict`, one song | a few seconds |

A laptop with more cores is proportionally faster, and a GPU (`--device cuda`, e.g. Colab) cuts training to a few minutes. `--num-threads N` controls CPU threads.

Training runs at most 40 epochs and stops early once validation loss hasn't improved for 8. The best model is saved after every improving epoch, so **you can stop whenever you like with Ctrl+C**: training ends cleanly and still runs the final tuning step. If a run was killed some other way (closed terminal, crash), finish it with `python -m sharp.train --tune-only`. To cap the length up front, pass e.g. `--epochs 15`.

---

## Experiments worth reporting

All are one flag away. Use `--run-name` to keep checkpoints apart and `--results-dir` to keep results apart. Compare on `--split val`.

```bash
# Does whole-song context matter? CNN only, no BiGRU
python -m sharp.train --gru-layers 0 --run-name cnn_only
python -m sharp.evaluate --split val --methods cnn_bigru --checkpoint runs/cnn_only/best.pt --results-dir results/cnn_only

# Hard (one-frame) vs smeared boundary targets
python -m sharp.train --boundary-sigma 0 --run-name hard_targets

# No data augmentation / heavier boundary loss / bigger recurrent layer
python -m sharp.train --no-spec-augment --run-name no_aug
python -m sharp.train --boundary-weight 2 --run-name bnd_w2
python -m sharp.train --gru-hidden 256 --run-name gru256
```

---

## Testing without the real data

`python -m pytest` runs the test suite (about 70 tests, ~10 s), including a full prepare → train → baselines → evaluate → predict run on a tiny synthetic dataset.

To exercise the pipeline at full scale before downloading, generate stand-in spectrograms. They use the real annotations, but the "audio" is shaped noise that changes timbre at each boundary. **The scores they produce are meaningless.**

```bash
python -m sharp.download --what annotations
python -m sharp.synthetic --out data/raw/melspecs_standin
python -m sharp.prepare --mel-dir data/raw/melspecs_standin --cache-dir data/processed_standin
python -m sharp.train --cache-dir data/processed_standin --runs-dir runs/standin --epochs 3
python -m sharp.baselines --cache-dir data/processed_standin --runs-dir runs/standin
python -m sharp.evaluate --split val --cache-dir data/processed_standin --runs-dir runs/standin --results-dir results_standin
```

---

## Repository layout

```
sharp/
  config.py        paths, Harmonix spectrogram parameters, TrainConfig defaults
  labels.py        126 raw labels -> 10 section types
  annotations.py   reading segments/*.txt and metadata.csv (quirks handled)
  splits.py        leakage-safe song-level splits
  audio.py         audio file -> Harmonix-compatible mel-spectrogram (NumPy/SciPy)
  features.py      pooling, dB, standardisation, frame-level targets
  data.py          feature cache, PyTorch Dataset, batching
  model.py         CNN + BiGRU, checkpoint I/O
  postprocess.py   peak picking + segment labelling
  metrics.py       mir_eval metrics, label metrics, threshold tuning
  predictors.py    common interface for the model and all baselines
  download.py  prepare.py  train.py  baselines.py  evaluate.py  predict.py   (entry points)
  plotting.py      structure strips, confusion matrices, training curves
  synthetic.py     stand-in data for tests
splits/            train.txt, val.txt, test.txt (committed)
tests/             pytest suite (no data needed)
data/ runs/ predictions/   created by the scripts (git-ignored)
results/           metrics tables and figures (small; commit the final ones)
```

---

## References

* O. Nieto, M. McCallum, M. Davies, A. Robertson, A. Stark, E. Egozy. *The Harmonix Set: Beats, Downbeats, and Functional Segment Annotations of Western Popular Music.* ISMIR 2019.
* C. Raffel et al. *mir_eval: A Transparent Implementation of Common MIR Metrics.* ISMIR 2014.
* K. Ullrich, J. Schlüter, T. Grill. *Boundary Detection in Music Structure Analysis Using Convolutional Neural Networks.* ISMIR 2014. (source of the smeared boundary targets)
* J. Foote. *Automatic Audio Segmentation Using a Measure of Audio Novelty.* ICME 2000.
