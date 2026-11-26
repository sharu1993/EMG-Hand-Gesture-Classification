"""Compute the common-average-reference features (set C, aura.features.car_features)
for the same windows as scripts/build_features.py.

    python3 scripts/build_car_features.py      # after scripts/build_features.py

Why (analysis of 25 Nov 2026): at rest, a median 61% of the cleaned EMG power in
the 60 Hz sessions is one signal shared by all 8 channels (16% in the 50 Hz
sessions; 12.5% if the channels were independent), and it stays that high during
gestures, so it buries the gesture in the 60 Hz setup. The reference removes it.

Writes data/features/car_{train,val,test}.pkl (git-ignored): the window metadata
and the C_* columns, one row per window in the same order as
data/features/{train,val,test}.pkl; refuses to write if the windows differ.
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
from aura.features import car_features  # noqa: E402
from aura.io import guided_segments, load_test, load_train  # noqa: E402
from aura.windows import test_arrays, training_windows, validation_windows, window_arrays  # noqa: E402

DATA = ROOT / "vr-calibration-free-emg-challenge-pieeg"
OUT = ROOT / "data/features"
SEED = 0                      # as scripts/build_features.py, so the training windows are the same
META = ["session", "seg", "label", "kind", "offset", "n_rows"]
_TRAIN = {}


def _init(train):
    _TRAIN.update(train)


def _session_rows(args):
    s, ws = args
    d = _TRAIN[s]
    rows = []
    for w in ws:
        e, i, fr = window_arrays(d, w)
        rows.append({"session": w.session, "seg": w.seg, "label": w.label, "kind": w.kind,
                     "offset": w.offset, "n_rows": w.n_rows, **car_features(e, i, fr)})
    return rows


def _test_row(item):
    f, a, e, i, fr = item
    return {"file": f, "action_id": a, "n_rows": len(e), **car_features(e, i, fr)}


def check(new: pd.DataFrame, ref_path: Path, keys: list[str]) -> None:
    ref = pd.read_pickle(ref_path)
    if len(ref) != len(new) or not (ref[keys].to_numpy() == new[keys].to_numpy()).all():
        sys.exit(f"windows differ from {ref_path.name}; rerun scripts/build_features.py first")


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
    out = {}
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(train,)) as ex:
        for name, jobs in (("train", jobs_tr), ("val", jobs_va)):
            out[name] = pd.DataFrame([r for rows in ex.map(_session_rows, jobs) for r in rows])
            print(f"{name}: {len(out[name])} windows ({time.time() - t0:.0f} s)", flush=True)
        out["test"] = pd.DataFrame(list(ex.map(_test_row, test_arrays(test), chunksize=16)))
    for name, df in out.items():
        check(df, OUT / f"{name}.pkl", ["action_id", "n_rows"] if name == "test" else META)
    for name, df in out.items():
        cols = [c for c in df.columns if c.startswith("C_")]
        bad = (~np.isfinite(df[cols].to_numpy())).sum()
        df.to_pickle(OUT / f"car_{name}.pkl")
        print(f"car_{name}: {len(df)} windows, {len(cols)} C_ features, non-finite values: {bad}")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
