"""Team Sharp - learning the shape of a song.

Song-structure analysis on the Harmonix Set: a CNN + BiGRU reads a
mel-spectrogram and predicts (1) where each section starts and (2) what
type of section it is (intro, verse, chorus, ...).

Entry points (run from the repository root):

    python -m sharp.download    # fetch annotations + mel-spectrograms
    python -m sharp.prepare     # splits, cached features, normalisation stats
    python -m sharp.train       # train the CNN + BiGRU
    python -m sharp.baselines   # train majority / logistic regression / MLP / novelty baselines
    python -m sharp.evaluate    # score everything on the test split with mir_eval
    python -m sharp.predict     # any song in, labelled structure map out
"""

__version__ = "1.0.0"
