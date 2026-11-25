"""The two-stage model (doc "Goal and the two-stage design").

Stage 1: rest vs gesture, gradient-boosted trees on feature sets A + B, trained
         with a random gain per window (up to 10x either way) so it leans less
         on absolute amplitude (25 Nov comparison: best on test-like windows).
Stage 2: which of the five gestures, gradient-boosted trees on the stage-2 (S_)
         features, trained on gesture windows only.
Combining: "gate" = rest if p(gesture) < 0.5, else the stage-2 argmax (fixed,
         not tuned); "soft" = argmax over P(rest) = 1 - p, P(g) = p * p(g | gesture).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .io import GESTURES, LABELS

STAGE1_PREFIXES = ("A_", "B_")
STAGE2_PREFIXES = ("S_",)
GATE = 0.5
GAIN_DECADES = 1.0     # stage-1 augmentation: gain 10**U(-1, 1) per window


def feature_cols(df: pd.DataFrame, prefixes) -> list[str]:
    return [c for c in df.columns if c.startswith(tuple(prefixes))]


def _trees(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, l2_regularization=1.0, random_state=seed)


def gain_augment(X: np.ndarray, cols: list[str], rng: np.random.Generator) -> np.ndarray:
    """Scaling a window's EMG by g adds 2 ln g to every absolute log-power (A_)
    feature and leaves the others unchanged, so augment in feature space."""
    X = X.copy()
    ia = [i for i, c in enumerate(cols) if c.startswith("A_")]
    g = rng.uniform(-GAIN_DECADES, GAIN_DECADES, size=len(X)) * np.log(10)
    X[:, ia] += 2 * g[:, None]
    return X


class TwoStage:
    def __init__(self, seed: int = 0, gain_aug: bool = True):
        self.seed, self.gain_aug = seed, gain_aug

    def fit(self, train: pd.DataFrame) -> "TwoStage":
        self.c1 = feature_cols(train, STAGE1_PREFIXES)
        self.c2 = feature_cols(train, STAGE2_PREFIXES)
        X1 = train[self.c1].to_numpy(float)
        if self.gain_aug:
            X1 = gain_augment(X1, self.c1, np.random.default_rng(self.seed))
        self.m1 = _trees(self.seed).fit(X1, (train["label"] != "rest").to_numpy().astype(int))
        g = train[train["label"] != "rest"]
        self.m2 = _trees(self.seed).fit(g[self.c2].to_numpy(float), g["label"].to_numpy())
        self.order2 = [list(self.m2.classes_).index(c) for c in GESTURES]
        return self

    def predict_parts(self, df: pd.DataFrame):
        """(p_gesture (n,), p_which (n, 5) in GESTURES order)."""
        p1 = self.m1.predict_proba(df[self.c1].to_numpy(float))[:, 1]
        p2 = self.m2.predict_proba(df[self.c2].to_numpy(float))[:, self.order2]
        return p1, p2


def combine(p1: np.ndarray, p2: np.ndarray, rule: str = "gate") -> np.ndarray:
    """Six-class labels from the two stages."""
    if rule == "gate":
        out = np.array(GESTURES, dtype=object)[p2.argmax(1)]
        out[p1 < GATE] = "rest"
        return out
    if rule == "soft":
        p6 = np.column_stack([1 - p1, p1[:, None] * p2])
        return np.array(LABELS, dtype=object)[p6.argmax(1)]
    raise ValueError(rule)
