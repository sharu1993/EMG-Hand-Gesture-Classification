"""NumPy-only metrics and session folds.

The PyTorch training script must not import scikit-learn: on macOS both
libraries bring their own OpenMP runtime (libomp.dylib), and loading two copies
in one process aborts with "OMP: Error #15". These helpers reproduce the
scikit-learn behaviour the project relies on.
"""
from __future__ import annotations

import numpy as np


def confusion(y_true, y_pred, n_classes: int) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def f1_per_class(y_true, y_pred, labels) -> np.ndarray:
    """F1 for each label in ``labels``; 0 where precision + recall is 0."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = []
    for c in labels:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = 2 * tp + fp + fn
        out.append(2 * tp / denom if denom else 0.0)
    return np.array(out)


def macro_f1(y_true, y_pred) -> float:
    """Unweighted mean F1 over the labels present in y_true or y_pred
    (same as sklearn.metrics.f1_score(average="macro"))."""
    labels = np.union1d(np.asarray(y_true), np.asarray(y_pred))
    return float(f1_per_class(y_true, y_pred, labels).mean()) if len(labels) else 0.0


def session_folds(sessions, n_folds: int) -> list[np.ndarray]:
    """Held-out session sets, one per fold.

    Sessions are dealt round-robin from the highest number down, which gives the
    same folds as sklearn's GroupKFold when every session is its own group, so
    results line up with scripts/baseline.py (fold 0 = sessions 7, 14, 21, 28).
    """
    s = np.sort(np.asarray(sessions))[::-1]
    return [np.sort(s[k::n_folds]) for k in range(n_folds)]
