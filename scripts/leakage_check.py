"""Can the label be predicted from length/timing alone? It should not be.

Trains a classifier on non-signal features only (length, raw row count, row
rate, offset) with session-grouped cross-validation, and reports macro F1 next
to the chance level. Run for whole segments (the leaky setup) and for windows
from WindowSampler (the setup we train on).

    python scripts/leakage_check.py data/train_segments.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emg_gestures.segments import load_segments  # noqa: E402
from emg_gestures.windows import WindowSampler  # noqa: E402


def grouped_macro_f1(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=0)
    pred = cross_val_predict(clf, X, y, groups=groups, cv=GroupKFold(n_splits=7))
    return f1_score(y, pred, average="macro")


def chance_f1(y: np.ndarray, seed: int = 0, reps: int = 20) -> float:
    """Macro F1 of label-shuffled predictions (respects class balance)."""
    rng = np.random.default_rng(seed)
    return float(np.mean([f1_score(y, rng.permutation(y), average="macro") for _ in range(reps)]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("segments", type=Path)
    ap.add_argument("--n-windows", type=int, default=12000)
    ap.add_argument("--min-len", type=float, default=0.75)
    ap.add_argument("--max-len", type=float, default=1.4)
    args = ap.parse_args()

    segs = load_segments(args.segments)
    rows = []

    # 1. Whole segments: duration and row count go straight to the model (leaky).
    X = np.column_stack([segs.duration, segs.n_rows, segs.n_rows / segs.duration])
    y, g = segs.label.to_numpy(), segs.session.to_numpy()
    rows.append(("whole segments", len(y), grouped_macro_f1(X, y, g), chance_f1(y)))

    # 2. Sampled windows: same features plus where the window starts.
    # 2. Sampled windows. The model sees only the resampled arrays, so their
    #    length is the one timing feature it can use; it must sit at chance.
    #    Raw row count/rate and offset are not visible to the model; they are
    #    reported as diagnostics.
    sampler = WindowSampler(segs, args.min_len, args.max_len, seed=0)
    ws = sampler.sample_many(args.n_windows)
    y = np.array([w.label for w in ws])
    g = np.array([w.session for w in ws])
    feats = {
        "windows: resampled length (model-visible)": [[len(w.emg)] for w in ws],
        "windows: raw rows + row rate (hidden)": [[w.n_rows_raw, w.n_rows_raw / w.length_s] for w in ws],
        "windows: offset in segment (hidden)": [[w.offset_s] for w in ws],
    }
    for name, X in feats.items():
        rows.append((name, len(y), grouped_macro_f1(np.array(X), y, g), chance_f1(y)))

    out = pd.DataFrame(rows, columns=["setup", "n", "macro_F1_from_timing", "chance"]).round(3)
    print(f"segments skipped (shorter than {args.min_len} s): {sampler.dropped}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
