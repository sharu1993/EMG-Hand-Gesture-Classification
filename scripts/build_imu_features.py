"""Compute the per-axis IMU features (set I, aura.features.imu_axis_features) for
the same windows as scripts/build_features.py.

    python3 scripts/build_imu_features.py      # after scripts/build_features.py

Writes data/features/imu_{train,val,test}.pkl (git-ignored): the window metadata
and the I_* columns, one row per window in the same order as
data/features/{train,val,test}.pkl; refuses to write if the windows differ.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aura.features import imu_axis_features  # noqa: E402
from aura.io import guided_segments, load_test, load_train  # noqa: E402
from aura.windows import test_arrays, training_windows, validation_windows, window_arrays  # noqa: E402

DATA = ROOT / "vr-calibration-free-emg-challenge-pieeg"
OUT = ROOT / "data/features"
SEED = 0                      # as scripts/build_features.py, so the training windows are the same
META = ["session", "seg", "label", "kind", "offset", "n_rows"]


def rows(train, windows_by_session):
    out = []
    for s, ws in windows_by_session:
        d = train[s]
        for w in ws:
            e, i, fr = window_arrays(d, w)
            out.append({"session": w.session, "seg": w.seg, "label": w.label, "kind": w.kind,
                        "offset": w.offset, "n_rows": w.n_rows, **imu_axis_features(i, fr)})
    return pd.DataFrame(out)


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
    out = {"train": rows(train, jobs_tr), "val": rows(train, jobs_va)}
    out["test"] = pd.DataFrame([{"file": f, "action_id": a, "n_rows": len(e), **imu_axis_features(i, fr)}
                                for f, a, e, i, fr in test_arrays(test)])
    for name, df in out.items():
        check(df, OUT / f"{name}.pkl", ["action_id", "n_rows"] if name == "test" else META)
    for name, df in out.items():
        cols = [c for c in df.columns if c.startswith("I_")]
        bad = (~np.isfinite(df[cols].to_numpy())).sum()
        df.to_pickle(OUT / f"imu_{name}.pkl")
        print(f"imu_{name}: {len(df)} windows, {len(cols)} I_ features, non-finite values: {bad}")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
