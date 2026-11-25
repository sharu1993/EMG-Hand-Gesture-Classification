"""The two-stage model, quiet/active design (doc "Goal and the two-stage design").

Stage 1  quiet (rest or pinch) vs active (fist, open, point, thumb-up).
         Pinch and rest look alike in activity level, so stage 1 does not
         have to tell them apart.
Quiet    rest vs pinch, a model that only ever sees those two, trained on
         equal numbers of rest and pinch windows (a random subset of rest),
         the same training-mix rule as stage 1.
Active   which of the four active gestures.

All three are gradient-boosted trees on feature sets A + B + S. The two models
that involve rest are trained with a random gain per window (up to 10x either
way) so they lean less on absolute amplitude. Every decision is a fixed 0.5
gate or an argmax: no tuned thresholds, no class weights.

Cross-validation, 25 Nov 2026 (3 fold repeats, gestures at 0.7 s): six-class
macro F1 0.652 vs 0.634 for rest/gesture + five gestures with the same features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .io import LABELS

PREFIXES = ("A_", "B_", "S_")
QUIET = ["rest", "pinch"]
ACTIVE = ["fist", "open", "point", "thumb-up"]
GATE = 0.5
GAIN_DECADES = 1.0     # gain augmentation: 10**U(-1, 1) per window


def feature_cols(df: pd.DataFrame, prefixes=PREFIXES) -> list[str]:
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

    def _x(self, df: pd.DataFrame, aug_seed: int | None = None) -> np.ndarray:
        X = df[self.cols].to_numpy(float)
        if aug_seed is not None and self.gain_aug:
            X = gain_augment(X, self.cols, np.random.default_rng(aug_seed))
        return X

    def fit(self, train: pd.DataFrame) -> "TwoStage":
        self.cols = feature_cols(train)
        lab = train["label"].to_numpy()
        s = self.seed
        # stage 1: quiet vs active, all windows
        self.m_quiet = _trees(s).fit(self._x(train, s), np.isin(lab, QUIET).astype(int))
        # quiet branch: rest vs pinch on equal numbers of each
        ip = np.flatnonzero(lab == "pinch")
        ir = np.random.default_rng(s + 1).choice(np.flatnonzero(lab == "rest"), size=len(ip), replace=False)
        q = train.iloc[np.r_[ip, ir]]
        self.m_pinch = _trees(s).fit(self._x(q, s + 2), (q["label"] == "pinch").to_numpy().astype(int))
        # active branch: four gestures
        a = train[np.isin(lab, ACTIVE)]
        self.m_active = _trees(s).fit(self._x(a), a["label"].to_numpy())
        self.order_active = [list(self.m_active.classes_).index(c) for c in ACTIVE]
        return self

    def predict_parts(self, df: pd.DataFrame) -> dict:
        """p_quiet (n,), p_pinch = p(pinch | quiet) (n,), p_active (n, 4) in ACTIVE order."""
        X = self._x(df)
        return {"p_quiet": self.m_quiet.predict_proba(X)[:, 1],
                "p_pinch": self.m_pinch.predict_proba(X)[:, 1],
                "p_active": self.m_active.predict_proba(X)[:, self.order_active]}


def average_parts(parts: list[dict]) -> dict:
    return {k: np.mean([p[k] for p in parts], axis=0) for k in parts[0]}


def combine(parts: dict) -> np.ndarray:
    """Six-class labels: quiet if p_quiet >= 0.5, then pinch if p_pinch >= 0.5
    else rest; otherwise the most likely active gesture."""
    quiet = np.where(parts["p_pinch"] >= GATE, "pinch", "rest").astype(object)
    active = np.array(ACTIVE, dtype=object)[parts["p_active"].argmax(1)]
    return np.where(parts["p_quiet"] >= GATE, quiet, active)


def proba6(parts: dict) -> np.ndarray:
    """(n, 6) probabilities in LABELS order, for records and later ensembles."""
    q, pp, pa = parts["p_quiet"], parts["p_pinch"], parts["p_active"]
    cols = {"rest": q * (1 - pp), "pinch": q * pp}
    cols.update({c: (1 - q) * pa[:, j] for j, c in enumerate(ACTIVE)})
    return np.column_stack([cols[c] for c in LABELS])
