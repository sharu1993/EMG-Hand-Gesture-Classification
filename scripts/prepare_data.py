"""Read the competition files once and prepare everything the models share.

    python3 scripts/prepare_data.py

Writes
  data/cache/train.pkl, data/cache/test.pkl   parsed sessions / test actions (git-ignored)
  data/segments.csv                           guided segments of every session (git-ignored)
  config/folds.json                           setup per session + fold assignments (committed)
and prints checks against the facts the design rests on.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aura.folds import write_folds  # noqa: E402
from aura.io import EMG, guided_segments, load_test, load_train, mains_hz  # noqa: E402
from aura.windows import test_arrays, training_windows, validation_windows, windows_table  # noqa: E402

DATA = ROOT / "vr-calibration-free-emg-challenge-pieeg"


def main() -> None:
    t0 = time.time()
    train = load_train(DATA / "train", ROOT / "data/cache/train.pkl")
    test = load_test(DATA / "test", ROOT / "data/cache/test.pkl")
    print(f"{len(train)} training sessions, {test.action_id.nunique()} test actions ({time.time() - t0:.0f} s)")

    segs = pd.concat([guided_segments(d, s) for s, d in train.items()], ignore_index=True)
    segs.to_csv(ROOT / "data/segments.csv", index=False)
    per = segs.groupby("session").size()
    print(f"guided segments per session: {per.min()}-{per.max()}; by label: {segs.label.value_counts().to_dict()}")

    setup = {s: mains_hz(d[EMG].to_numpy()) for s, d in train.items()}
    cfg = write_folds(ROOT / "config/folds.json", setup)
    print("60 Hz sessions:", sorted(s for s, v in setup.items() if v == 60))
    for seed, folds in cfg["repeats"].items():
        print(f"folds, seed {seed}:", folds)

    rng = np.random.default_rng(0)
    tw, vw = [], []
    for s, d in train.items():
        g = segs[segs.session == s]
        tw += training_windows(d, g, rng)
        vw += validation_windows(d, g)
    T, V = windows_table(tw), windows_table(vw)
    print("\ntraining windows by set and label:")
    print(T.pivot_table(index="kind", columns="label", values="row0", aggfunc="size").to_string())
    print("\nvalidation windows by set and label:")
    print(V.pivot_table(index="kind", columns="label", values="row0", aggfunc="size").to_string())
    print("\nrest window start after the cue (s), test-like training:",
          T[(T.kind == "testlike") & (T.label == "rest")].offset.describe()[["min", "50%", "max"]].round(2).to_dict())
    print("rows per window, test-like training:", T[T.kind == "testlike"].n_rows.describe()[["min", "50%", "max"]].to_dict())
    rows = np.array([len(e) for _, _, e, _, _ in test_arrays(test)])
    print("rows per test action:", dict(min=int(rows.min()), median=float(np.median(rows)), max=int(rows.max())))
    print(f"\ndone in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
