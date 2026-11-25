"""Compute both stages' features for every training, validation and test window.

    python3 scripts/build_features.py           # after scripts/prepare_data.py

Writes data/features/{train,val,test}.pkl (git-ignored): one row per window,
window metadata columns first, then the stage-1 (A_*, B_*) and stage-2 (S_*)
features. Training windows are drawn with seed 0 (aura.windows).
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
from aura.features import stage1_features, stage2_features  # noqa: E402
from aura.io import guided_segments, load_test, load_train  # noqa: E402
from aura.windows import test_arrays, training_windows, validation_windows, window_arrays  # noqa: E402

DATA = ROOT / "vr-calibration-free-emg-challenge-pieeg"
OUT = ROOT / "data/features"
SEED = 0
_TRAIN = {}


def _init(train):
    _TRAIN.update(train)


def _session_rows(args):
    """Features for one session's windows (runs in a worker)."""
    s, ws = args
    d = _TRAIN[s]
    rows = []
    for w in ws:
        e, i, fr = window_arrays(d, w)
        rows.append({"session": w.session, "seg": w.seg, "label": w.label, "kind": w.kind, "offset": w.offset,
                     "n_rows": w.n_rows, **stage1_features(e, i, fr), **stage2_features(e, i, fr)})
    return rows


def _test_row(item):
    f, a, e, i, fr = item
    return {"file": f, "action_id": a, "n_rows": len(e), **stage1_features(e, i, fr), **stage2_features(e, i, fr)}


def main() -> None:
    t0 = time.time()
    train = load_train(DATA / "train", ROOT / "data/cache/train.pkl")
    test = load_test(DATA / "test", ROOT / "data/cache/test.pkl")
    rng = np.random.default_rng(SEED)
    jobs_tr, jobs_va = [], []
    for s, d in train.items():
        g = guided_segments(d, s)
        jobs_tr.append((s, training_windows(d, g, rng)))
        jobs_va.append((s, validation_windows(d, g)))
    workers = min(8, os.cpu_count() or 2)
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(train,)) as ex:
        for name, jobs in (("train", jobs_tr), ("val", jobs_va)):
            df = pd.DataFrame([r for rows in ex.map(_session_rows, jobs) for r in rows])
            df.to_pickle(OUT / f"{name}.pkl")
            print(f"{name}: {len(df)} windows, {df.shape[1]} columns ({time.time() - t0:.0f} s)", flush=True)
        te = pd.DataFrame(list(ex.map(_test_row, test_arrays(test), chunksize=16)))
    te.to_pickle(OUT / "test.pkl")
    print(f"test: {len(te)} actions ({time.time() - t0:.0f} s)")
    feats = [c for c in te.columns if c[:2] in ("A_", "B_", "S_")]
    bad = te[feats].isna().sum().sum() + (~np.isfinite(te[feats].to_numpy())).sum()
    print(f"features: {sum(c.startswith('A_') for c in feats)} A, {sum(c.startswith('B_') for c in feats)} B, "
          f"{sum(c.startswith('S_') for c in feats)} S; non-finite values in test: {bad}")


if __name__ == "__main__":
    main()
