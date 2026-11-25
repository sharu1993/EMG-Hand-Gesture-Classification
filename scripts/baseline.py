"""Feature baseline: gradient-boosted trees on hand-crafted window features,
scored with session-held-out cross-validation.

Training windows: WindowSampler (random length 0.75-1.2 s, random offset,
class-balanced) from the steady part of each segment, so transitions between
gestures are never seen; drawn per session so every fold reuses the same pool.
With --align the steady part comes from each segment's own gyro-based span
(emg_gestures/align.py) instead of fixed trims.
Validation windows: fixed 1.2 s windows (stride 0.5 s) from the steady part
(always fixed trims, so all runs are scored on the same windows).
Also reported: the same models on untrimmed 1.4 s windows (test-like, they may
include transitions), and scores per recording setup (50 Hz vs 60 Hz mains).

Two learners per feature set: gradient-boosted trees, and a small neural
network (MLP, one hidden layer of 128 units, on standardised features). The MLP
is weaker alone but makes different mistakes, which helps the ensemble
(scripts/ensemble.py), mainly on the 60 Hz sessions. --no-mlp skips it.

--shift-aug small|wide adds one copy of every training window with the electrode
patch shifted, as if the band were worn rotated (small: up to one row; wide: up to
two rows or the band reversed; emg_gestures/augment.py). Validation windows are
never shifted. Output files then end in _shift<level>.

--train-windows steady|testlike|both|both_natural picks the training windows:
steady (default, above), test-like windows that include the end of the onset
movement as the test actions do (emg_gestures/windows.py, testlike_windows), or
both. Window lengths are drawn independently of the class. both_natural keeps
every rest window instead of balancing classes. Output files then carry
_win<set>. Every run also scores test-like validation windows (1.4 s from 0.5 s
after the cue, one per segment), the closest match to the test actions.

Every configuration saves its cross-validated class probabilities to
outputs/baseline_<name>.probs.npz (trees) and outputs/mlp_<name>.probs.npz
(MLP), in the same format and window order as scripts/train_tcn.py, for
scripts/ensemble.py.

    python scripts/baseline.py                       # data/train_segments.npz
    python scripts/baseline.py --align
    python scripts/baseline.py --ablations           # also EMG-only / IMU-only / +gain
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emg_gestures.align import align_segments  # noqa: E402
from emg_gestures.augment import SHIFT_LEVELS, patch_shift, random_patch_shift  # noqa: E402
from emg_gestures.features import window_features  # noqa: E402
from emg_gestures.segments import LABELS, load_segments  # noqa: E402
from emg_gestures.windows import (ALIGNED, NO_TRIM, STEADY_STATE, TRAIN_WINDOW_SETS,  # noqa: E402
                                  fixed_windows, testlike_val_windows, training_windows)

# Sessions recorded with 60 Hz mains and (mostly) a ~45 Hz IMU: a second setup.
SESSIONS_60HZ = {10, 11, 15, 16, 22, 23, 24}
META = ("label", "session", "length_s", "offset_s", "sample_id")

# Feature-name prefixes: emg_riem = covariance tangent vectors, emg_rcor = the
# same covariances as correlation matrices (pattern only), emg_ = other EMG
# features, imu_ = IMU features (emg_gestures/features.py). emg_rshp / emg_rscl =
# the covariances with each band's power divided out, and those band powers
# (emg_gestures.riemann.shape_scale); only "Riemann shape + Corr + IMU" uses them.
_SHAPE = ("emg_rshp", "emg_rscl")
_RIEM = ("emg_riem", "emg_rcor") + _SHAPE
CONFIGS = {
    # name: feature-name filter
    "EMG + IMU": lambda c: c != "emg_gain" and not c.startswith(_RIEM),
    "Riemann + IMU": lambda c: c.startswith("emg_riem") or c.startswith("imu_"),
    "Riemann + Corr + IMU": lambda c: c.startswith(("emg_riem", "emg_rcor")) or c.startswith("imu_"),
    "EMG + Riemann + IMU": lambda c: c != "emg_gain" and not c.startswith(("emg_rcor",) + _SHAPE),
    "Riemann shape + Corr + IMU": lambda c: c.startswith(_SHAPE + ("emg_rcor",)) or c.startswith("imu_"),
}
ABLATIONS = {
    "EMG": lambda c: c.startswith("emg_") and c != "emg_gain" and not c.startswith(_RIEM),
    "EMG + gain": lambda c: c.startswith("emg_") and not c.startswith(_RIEM),
    "IMU": lambda c: c.startswith("imu_"),
}


def _feats(job):
    """job: a window, or (window, patch shift, seed) for a shifted training copy."""
    w, shift, seed = job if isinstance(job, tuple) else (job, None, 0)
    emg = w.emg if shift is None else patch_shift(w.emg, shift, np.random.default_rng(seed))
    f = window_features(emg, w.imu, riemann=True, corr=True)  # IMU already low-passed in make_window
    f.update(label=w.label, session=w.session, length_s=w.length_s, offset_s=w.offset_s,
             sample_id=w.sample_id)
    return f


def feature_table(windows, workers: int, shift_level: str = "none", seed: int = 0) -> pd.DataFrame:
    """Features of ``windows``; with a shift level, plus one patch-shifted copy of each
    window (shift drawn per window from SHIFT_LEVELS[shift_level])."""
    jobs = list(windows)
    if shift_level != "none":
        rng = np.random.default_rng(seed + 7919)
        jobs += [(w, random_patch_shift(shift_level, rng), seed * 1_000_003 + k)
                 for k, w in enumerate(windows)]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_feats, jobs, chunksize=64))
    return pd.DataFrame(rows)


def new_model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=seed)


class LabelCodedMLP:
    """One hidden layer of 128 units on standardised features; strong L2
    (alpha 0.1) and early stopping on a 10% split of the training windows.

    Trained on integer class codes: recent scikit-learn releases fail inside
    MLPClassifier's early stopping when the labels are strings (an isnan check
    on the predicted labels). ``classes_`` and ``predict_proba`` columns are in
    sorted label order, as for the other scikit-learn classifiers."""

    def __init__(self, seed: int):
        self.model = make_pipeline(StandardScaler(),
                                   MLPClassifier((128,), alpha=0.1, early_stopping=True,
                                                 max_iter=300, random_state=seed))

    def fit(self, X, y):
        self.classes_, codes = np.unique(np.asarray(y), return_inverse=True)
        self.model.fit(X, codes)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


def new_mlp(seed: int) -> LabelCodedMLP:
    return LabelCodedMLP(seed)


LEARNERS = {"hgb": new_model, "mlp": new_mlp}


def run_cv(train: pd.DataFrame, tables: dict, cols, folds: int, seed: int, learner: str = "hgb"):
    """Session-held-out CV. Returns class probabilities (columns in LABELS
    order) for every table in ``tables`` and the per-fold macro F1 on the first."""
    sessions = np.array(sorted(train.session.unique()))
    probs = {k: np.full((len(t), len(LABELS)), np.nan) for k, t in tables.items()}
    fold_f1 = []
    first = next(iter(tables))
    for _, te in GroupKFold(n_splits=folds).split(sessions, groups=sessions):
        held = set(sessions[te])
        tr = train[~train.session.isin(held)]
        clf = LEARNERS[learner](seed).fit(tr[cols], tr.label)
        order = [list(clf.classes_).index(lab) for lab in LABELS]   # sklearn sorts class names
        for k, t in tables.items():
            m = t.session.isin(held).to_numpy()
            probs[k][m] = clf.predict_proba(t.loc[m, cols])[:, order]
        m = tables[first].session.isin(held).to_numpy()
        pred = np.array(LABELS)[probs[first][m].argmax(1)]
        fold_f1.append(f1_score(tables[first].label[m], pred, average="macro"))
    return probs, np.array(fold_f1)


def save_probs(path: Path, val: pd.DataFrame, raw: pd.DataFrame, p_val, p_raw,
               tl: pd.DataFrame | None = None, p_tl=None):
    lab_id = {lab: i for i, lab in enumerate(LABELS)}
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {} if tl is None else dict(
        prob_tl=p_tl.astype(np.float32), y_tl=tl.label.map(lab_id).to_numpy(),
        session_tl=tl.session.to_numpy(np.int64), sample_id_tl=tl.sample_id.to_numpy(str),
        offset_tl=tl.offset_s.to_numpy(np.float32))
    np.savez_compressed(
        path, **extra, prob_val=p_val.astype(np.float32), y_val=val.label.map(lab_id).to_numpy(),
        session_val=val.session.to_numpy(np.int64), sample_id_val=val.sample_id.to_numpy(str),
        offset_val=val.offset_s.to_numpy(np.float32),
        prob_raw=p_raw.astype(np.float32), y_raw=raw.label.map(lab_id).to_numpy(),
        session_raw=raw.session.to_numpy(np.int64), sample_id_raw=raw.sample_id.to_numpy(str),
        offset_raw=raw.offset_s.to_numpy(np.float32), labels=np.array(LABELS))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("segments", type=Path, nargs="?", default=Path("data/train_segments.npz"))
    ap.add_argument("--per-session", type=int, default=300, help="training windows per session")
    ap.add_argument("--folds", type=int, default=7)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--align", action="store_true",
                    help="training windows from per-segment aligned spans (emg_gestures/align.py)")
    ap.add_argument("--ablations", action="store_true", help="also run single-group feature sets")
    ap.add_argument("--mlp", action=argparse.BooleanOptionalAction, default=True,
                    help="also cross-validate the MLP on each main feature set (default on)")
    ap.add_argument("--train-windows", choices=list(TRAIN_WINDOW_SETS), default="steady",
                    help="training windows: steady (default), testlike, both, both_natural")
    ap.add_argument("--shift-aug", choices=list(SHIFT_LEVELS), default="none",
                    help="add patch-shifted copies of the training windows (default none)")
    ap.add_argument("--outdir", type=Path, default=Path("outputs"))
    args = ap.parse_args()
    configs = {**CONFIGS, **(ABLATIONS if args.ablations else {})}

    t0 = time.time()
    segs = load_segments(args.segments)
    train_segs = align_segments(segs) if args.align else segs
    train_w = training_windows(train_segs, args.train_windows, args.per_session, args.seed,
                               ALIGNED if args.align else STEADY_STATE)
    val_w, raw_w = [], []
    tl_w = testlike_val_windows(segs)
    for s, g in segs.groupby("session"):
        val_w += fixed_windows(g, length=1.2, stride=0.5)
        raw_w += fixed_windows(g, length=1.4, stride=0.5, trims=NO_TRIM)
    print(f"{len(train_w)} training windows ({args.train_windows}, align {args.align}), {len(val_w)} "
          f"steady validation windows, {len(raw_w)} untrimmed 1.4 s windows, {len(tl_w)} test-like "
          f"windows, {segs.session.nunique()} sessions")

    train = feature_table(train_w, args.workers, args.shift_aug, args.seed)
    if args.shift_aug != "none":
        print(f"shift augmentation '{args.shift_aug}': {len(train)} training rows "
              f"({len(train_w)} as worn + {len(train) - len(train_w)} shifted)")
    val = feature_table(val_w, args.workers)
    raw = feature_table(raw_w, args.workers)
    tl = feature_table(tl_w, args.workers)
    print(f"features: {time.time() - t0:.0f} s")

    def group_f1(df, pred, sixty: bool):
        m = (df.session.isin(SESSIONS_60HZ) == sixty).to_numpy()
        return round(f1_score(df.label[m], pred[m], average="macro"), 3)

    results, best = [], None
    jobs = [(name, keep, "hgb") for name, keep in configs.items()]
    if args.mlp:
        jobs += [(name, keep, "mlp") for name, keep in CONFIGS.items()]
    for name, keep, learner in jobs:
        cols = [c for c in train.columns if c not in META and keep(c)]
        probs, fold_f1 = run_cv(train, {"val": val, "raw": raw, "tl": tl}, cols, args.folds, args.seed,
                                learner)
        pred = np.array(LABELS)[probs["val"].argmax(1)]
        pred_raw = np.array(LABELS)[probs["raw"].argmax(1)]
        pred_tl = np.array(LABELS)[probs["tl"].argmax(1)]
        f1 = f1_score(val.label, pred, average="macro")
        results.append((name, learner, len(cols), round(f1, 3), round(fold_f1.std(), 3),
                        group_f1(val, pred, False), group_f1(val, pred, True),
                        round(f1_score(raw.label, pred_raw, average="macro"), 3),
                        round(f1_score(tl.label, pred_tl, average="macro"), 3)))
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        prefix = "baseline" if learner == "hgb" else "mlp"
        suffix = (("_align" if args.align else "")
                  + ("" if args.train_windows == "steady" else f"_win{args.train_windows}")
                  + ("" if args.shift_aug == "none" else f"_shift{args.shift_aug}"))
        save_probs(args.outdir / f"{prefix}_{slug}{suffix}.probs.npz",
                   val, raw, probs["val"], probs["raw"], tl, probs["tl"])
        if best is None or f1 > best[1]:
            best = (f"{name} ({learner})", f1, pred, cols)

    chance = 1 / len(LABELS)
    print(f"\nMacro F1, sessions held out ({args.folds} folds). Chance ~{chance:.2f}")
    print(pd.DataFrame(results, columns=["features", "learner", "n_feat", "steady_F1", "fold_std",
                                         "50Hz_sessions", "60Hz_sessions",
                                         "untrimmed_1.4s_F1", "testlike_F1"]).to_string(index=False))
    print(f"probabilities saved to {args.outdir}/baseline_*.probs.npz (trees) and mlp_*.probs.npz (MLP)")

    name, f1, pred, cols = best
    print(f"\nBest: {name}")
    print("Per-class F1:", dict(zip(LABELS, np.round(
        f1_score(val.label, pred, average=None, labels=LABELS), 3))))
    cm = pd.DataFrame(confusion_matrix(val.label, pred, labels=LABELS), index=LABELS,
                      columns=LABELS)
    print("Confusion matrix (rows = true):\n" + cm.to_string())
    per_session = pd.Series(pred, index=val.index).groupby(val.session).apply(
        lambda p: f1_score(val.label[p.index], p, average="macro"))
    print("Per-session macro F1:", per_session.round(2).to_dict())
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
