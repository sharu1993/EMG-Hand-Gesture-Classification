"""Windows with one misbehaving electrode, for training and for a stress test
(analysis of 26 Nov 2026: in session 5, channel 8 alone rose about 1,000x in
65-95 Hz power during fist, an electrode artefact; no clipping or stuck channels
were found in any session).

    python3 scripts/build_aug_channel.py      # after scripts/build_features.py

For every training window and every validation window (the same windows as
data/features/{train,val}.pkl), one copy in which one random channel of the raw,
mean-centred EMG is either
  "loud": multiplied by 10**U(0.5, 1.5) in amplitude (power 10-1,000x), or
  "dead": multiplied by 10**U(-2, -1) (an electrode that lost contact),
each with probability 0.5, then all features are recomputed: sets A, B, S and,
for use with --car, set C. The IMU is unchanged. Labels are kept.

Writes data/features/augch_train.pkl (extra training windows, scripts/cv_two_stage.py
--aug augch) and data/features/augch_val.pkl (the corrupted validation windows,
--val-aug augch), git-ignored: window metadata, aug_channel, aug_mode,
aug_factor, then the feature columns.
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aura.features import car_features, stage1_features, stage2_features  # noqa: E402
from aura.io import guided_segments, load_train  # noqa: E402
from aura.windows import training_windows, validation_windows, window_arrays  # noqa: E402

DATA = ROOT / "vr-calibration-free-emg-challenge-pieeg"
OUT = ROOT / "data/features"
SEED = 0                                   # as scripts/build_features.py: the same windows
AUG_SEED = {"train": 81, "val": 82}
LOUD, DEAD = (0.5, 1.5), (-2.0, -1.0)      # decades of amplitude
META = ["session", "seg", "label", "kind", "offset", "n_rows"]
_TRAIN = {}


def _init(train):
    _TRAIN.update(train)


def _session_rows(args):
    s, jobs = args
    d = _TRAIN[s]
    rows = []
    for w, ch, mode, dec in jobs:
        e, i, fr = window_arrays(d, w)
        e = e.copy()
        factor = 10 ** dec
        e[:, ch] *= factor
        rows.append({"session": w.session, "seg": w.seg, "label": w.label, "kind": w.kind, "offset": w.offset,
                     "n_rows": w.n_rows, "aug_channel": ch + 1, "aug_mode": mode, "aug_factor": factor,
                     **stage1_features(e, i, fr), **stage2_features(e, i, fr), **car_features(e, i, fr)})
    return rows


def plan(windows, rng):
    jobs = []
    for s, ws in windows:
        j = []
        for w in ws:
            ch = int(rng.integers(0, 8))
            mode = "loud" if rng.random() < 0.5 else "dead"
            dec = float(rng.uniform(*(LOUD if mode == "loud" else DEAD)))
            j.append((w, ch, mode, dec))
        jobs.append((s, j))
    return jobs


def main() -> None:
    t0 = time.time()
    train = load_train(DATA / "train", ROOT / "data/cache/train.pkl")
    rng = np.random.default_rng(SEED)
    win = {"train": [], "val": []}
    for s, d in train.items():
        g = guided_segments(d, s)
        win["train"].append((s, training_windows(d, g, rng)))     # same draws, same order as build_features.py
        win["val"].append((s, validation_windows(d, g)))
    workers = min(8, os.cpu_count() or 2)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(train,)) as ex:
        for name in ("train", "val"):
            jobs = plan(win[name], np.random.default_rng(AUG_SEED[name]))
            df = pd.DataFrame([r for rows in ex.map(_session_rows, jobs) for r in rows])
            ref = pd.read_pickle(OUT / f"{name}.pkl")
            if len(df) != len(ref) or not (df[META].to_numpy() == ref[META].to_numpy()).all():
                sys.exit(f"windows differ from {name}.pkl; rerun scripts/build_features.py first")
            want = [c for c in ref.columns if c[:2] in ("A_", "B_", "S_")]
            if [c for c in df.columns if c[:2] in ("A_", "B_", "S_")] != want:
                sys.exit(f"feature columns differ from {name}.pkl; rerun scripts/build_features.py first")
            feats = [c for c in df.columns if c[:2] in ("A_", "B_", "S_", "C_")]
            bad = (~np.isfinite(df[feats].to_numpy())).sum()
            df.to_pickle(OUT / f"augch_{name}.pkl")
            print(f"augch_{name}: {len(df)} windows ({(df.aug_mode == 'loud').mean():.0%} loud), "
                  f"{len(feats)} features, non-finite values: {bad} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
