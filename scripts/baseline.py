"""Feature baseline: gradient-boosted trees on hand-crafted window features,
scored with session-held-out cross-validation.

Training windows: WindowSampler (random length 0.75-1.2 s, random offset,
class-balanced) from the steady part of each segment, so transitions between
gestures are never seen; drawn per session so every fold reuses the same pool.
Validation windows: fixed 1.2 s windows (stride 0.5 s) from the steady part.
Also reported: the same models on untrimmed 1.4 s windows (test-like, they may
include transitions), and scores per recording setup (50 Hz vs 60 Hz mains).

    python scripts/baseline.py data/train_segments.npz
    python scripts/baseline.py data/train_segments.npz --per-session 400 --folds 7
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emg_gestures.features import window_features  # noqa: E402
from emg_gestures.segments import LABELS, load_segments  # noqa: E402
from emg_gestures.windows import NO_TRIM, WindowSampler, fixed_windows  # noqa: E402

# Sessions recorded with 60 Hz mains and (mostly) a ~45 Hz IMU: a second setup.
SESSIONS_60HZ = {10, 11, 15, 16, 22, 23, 24}

CONFIGS = {
    # name: (feature-name filter, use common-average reference)
    "EMG": (lambda c: c.startswith("emg_") and c != "emg_gain", False),
    "EMG + gain": (lambda c: c.startswith("emg_"), False),
    "IMU": (lambda c: c.startswith("imu_"), False),
    "EMG + IMU": (lambda c: c != "emg_gain", False),
}


def _feats(args):
    w, car = args
    f = window_features(w.emg, w.imu, car=car)  # IMU already low-passed in make_window
    f.update(label=w.label, session=w.session, length_s=w.length_s, offset_s=w.offset_s)
    return f


def feature_table(windows, car: bool, workers: int) -> pd.DataFrame:
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_feats, [(w, car) for w in windows], chunksize=64))
    return pd.DataFrame(rows)


def run_cv(train: pd.DataFrame, val: pd.DataFrame, cols, folds: int, seed: int,
           extra: pd.DataFrame | None = None):
    """Session-held-out CV. ``extra`` is a second validation table scored with
    the same fold models (untrimmed windows)."""
    sessions = np.array(sorted(train.session.unique()))
    pred = pd.Series(index=val.index, dtype=object)
    pred_extra = None if extra is None else pd.Series(index=extra.index, dtype=object)
    fold_f1 = []
    for k, (_, te) in enumerate(GroupKFold(n_splits=folds).split(sessions, groups=sessions)):
        held = set(sessions[te])
        tr = train[~train.session.isin(held)]
        va = val[val.session.isin(held)]
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                             l2_regularization=1.0, random_state=seed)
        clf.fit(tr[cols], tr.label)
        p = clf.predict(va[cols])
        pred.loc[va.index] = p
        fold_f1.append(f1_score(va.label, p, average="macro"))
        if extra is not None:
            ex = extra[extra.session.isin(held)]
            pred_extra.loc[ex.index] = clf.predict(ex[cols])
    return pred, np.array(fold_f1), pred_extra


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("segments", type=Path)
    ap.add_argument("--per-session", type=int, default=300, help="training windows per session")
    ap.add_argument("--folds", type=int, default=7)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--car", action="store_true", help="also run with common-average reference")
    args = ap.parse_args()
    if args.car:
        CONFIGS["EMG + IMU, CAR"] = (lambda c: c != "emg_gain", True)

    t0 = time.time()
    segs = load_segments(args.segments)
    train_w, val_w, raw_w = [], [], []
    for s, g in segs.groupby("session"):
        train_w += WindowSampler(g, seed=args.seed * 1000 + s).sample_many(args.per_session)
        val_w += fixed_windows(g, length=1.2, stride=0.5)
        raw_w += fixed_windows(g, length=1.4, stride=0.5, trims=NO_TRIM)
    print(f"{len(train_w)} training windows, {len(val_w)} steady validation windows, "
          f"{len(raw_w)} untrimmed 1.4 s windows, {segs.session.nunique()} sessions")

    tables = {}
    for car in sorted({c for _, c in CONFIGS.values()}):
        tables[car] = (feature_table(train_w, car, args.workers),
                       feature_table(val_w, car, args.workers),
                       feature_table(raw_w, car, args.workers))
    print(f"features: {time.time() - t0:.0f} s")

    results, best = [], None
    def group_f1(df, p, sixty: bool):
        m = df.session.isin(SESSIONS_60HZ) == sixty
        return round(f1_score(df.label[m], p[m], average="macro"), 3)

    for name, (keep, car) in CONFIGS.items():
        train, val, raw = tables[car]
        cols = [c for c in train.columns
                if c not in ("label", "session", "length_s", "offset_s") and keep(c)]
        pred, fold_f1, pred_raw = run_cv(train, val, cols, args.folds, args.seed, extra=raw)
        f1 = f1_score(val.label, pred, average="macro")
        results.append((name, len(cols), round(f1, 3), round(fold_f1.std(), 3),
                        group_f1(val, pred, False), group_f1(val, pred, True),
                        round(f1_score(raw.label, pred_raw, average="macro"), 3)))
        if best is None or f1 > best[1]:
            best = (name, f1, pred, val, train, cols)

    chance = 1 / len(LABELS)
    print(f"\nMacro F1, sessions held out ({args.folds} folds). Chance ~{chance:.2f}")
    print(pd.DataFrame(results, columns=["features", "n_feat", "steady_F1", "fold_std",
                                         "50Hz_sessions", "60Hz_sessions",
                                         "untrimmed_1.4s_F1"]).to_string(index=False))

    name, f1, pred, val, train, cols = best
    print(f"\nBest: {name}")
    print("Per-class F1:", dict(zip(LABELS, np.round(
        f1_score(val.label, pred, average=None, labels=LABELS), 3))))
    cm = pd.DataFrame(confusion_matrix(val.label, pred, labels=LABELS), index=LABELS,
                      columns=LABELS)
    print("Confusion matrix (rows = true):\n" + cm.to_string())
    per_session = val.assign(pred=pred).groupby("session").apply(
        lambda d: f1_score(d.label, d.pred, average="macro"))
    print("Per-session macro F1:", per_session.round(2).to_dict())
    by_offset = val.assign(ok=(pred == val.label)).groupby(
        pd.cut(val.offset_s, [-0.01, 0.1, 0.3, 2.0]), observed=True).ok.mean()
    print("Accuracy by window offset in segment (s):", by_offset.round(3).to_dict())

    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                         l2_regularization=1.0, random_state=args.seed)
    clf.fit(train[cols], train.label)
    from sklearn.inspection import permutation_importance
    sub = val.sample(min(2000, len(val)), random_state=0)
    imp = permutation_importance(clf, sub[cols], sub.label, scoring="f1_macro",
                                 n_repeats=3, random_state=0, n_jobs=args.workers)
    top = pd.Series(imp.importances_mean, index=cols).sort_values(ascending=False).head(15)
    print("Top features (permutation importance, in-sample fit; indicative only):")
    print(top.round(4).to_string())
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
