"""Simulated 60 Hz-like training windows (doc "Cross-validation"; analysis of 25 Nov 2026).

    python3 scripts/build_aug60.py      # after scripts/build_features.py

Why: in the 60 Hz sessions the EMG is about 5-8x weaker in power across
30-110 Hz and the contrast between a gesture and rest is smaller (a fist raises
power 2.0-3.3x over rest, against 2.8-4.1x at 50 Hz), so stage 1 sends 24% of
60 Hz gestures to the quiet branch (8% at 50 Hz). Gain augmentation changes the
level but not that contrast.

How: for every 50 Hz training window (the same windows as data/features/train.pkl),
  1. take a random stretch of a rest segment of a random 60 Hz training session,
     as long as the window (the "noise");
  2. scale the window so that its session's typical rest power is the noise's
     power divided by --noise-ratio (default 3: the ratio whose gesture-minus-rest
     contrast in the band powers comes closest to the real 60 Hz training
     sessions, of 1, 3, 10, 30), times a random factor 10**U(-0.25, 0.25) in amplitude;
  3. add the two raw signals, mean-centre, and compute the same features
     (A, B, S) with the window's own IMU.
The scale uses the 50 Hz session's median rest power, a statistic of the
training data only. Each simulated window records the 60 Hz session its noise
came from, so cross-validation can drop the windows whose noise comes from a
held-out session.

Writes data/features/aug60_train.pkl (git-ignored): window metadata, noise_session,
scale, then the feature columns.
"""
from __future__ import annotations

import argparse
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
from aura.folds import read_folds  # noqa: E402
from aura.io import EMG, IMU, guided_segments, load_train  # noqa: E402
from aura.preprocess import band, clean_emg  # noqa: E402
from aura.windows import training_windows  # noqa: E402

DATA = ROOT / "vr-calibration-free-emg-challenge-pieeg"
OUT = ROOT / "data/features"
SEED = 0              # as scripts/build_features.py: the same training windows
AUG_SEED = 60         # noise choice and scale jitter
JITTER = 0.25         # decades of amplitude, either way
BROAD = (20.0, 117.0)
_TRAIN, _NOISE, _REST50, _OPT = {}, [], {}, {}


def emg_power(x: np.ndarray) -> float:
    """Mean power of the cleaned EMG in 20-117 Hz, over channels."""
    return float((band(clean_emg(x - x.mean(0)), *BROAD) ** 2).mean())


def _init(train, noise, rest50, noise_ratio):
    _OPT["noise_ratio"] = noise_ratio
    _TRAIN.update(train)
    _NOISE.extend(noise)
    _REST50.update(rest50)


def _session_rows(args):
    s, jobs = args
    d = _TRAIN[s]
    rows = []
    for w, k, start, jit in jobs:
        ns, seg_emg = _NOISE[k]
        n = w.row1 - w.row0
        noise = seg_emg[start:start + n]
        part = d.iloc[w.row0:w.row1]
        e = part[EMG].to_numpy(float)
        e = e - e.mean(0)
        pn = emg_power(noise)
        a = np.sqrt(pn / (_OPT["noise_ratio"] * _REST50[s])) * 10 ** jit
        x = a * e + (noise - noise.mean(0))
        x = x - x.mean(0)
        imu = part[IMU].to_numpy(float)
        imu = imu - imu.mean(0)
        fr = part["imu_new"].to_numpy() == 1
        rows.append({"session": w.session, "seg": w.seg, "label": w.label, "kind": w.kind, "offset": w.offset,
                     "n_rows": n, "noise_session": ns, "scale": a,
                     **stage1_features(x, imu, fr), **stage2_features(x, imu, fr)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--noise-ratio", type=float, default=3.0,
                    help="power of the added 60 Hz rest over the scaled window's typical rest power")
    ap.add_argument("--sessions", type=int, nargs="*", help="only these 50 Hz sessions (for quick checks)")
    ap.add_argument("--out", default="aug60_train.pkl")
    args = ap.parse_args()
    t0 = time.time()
    train = load_train(DATA / "train", ROOT / "data/cache/train.pkl")
    setup = {int(s): v for s, v in read_folds(ROOT / "config/folds.json")["setup_hz"].items()}
    rng = np.random.default_rng(SEED)
    windows, segs = {}, {}
    for s, d in train.items():
        g = guided_segments(d, s)
        segs[s] = g
        windows[s] = training_windows(d, g, rng)      # same draws, in the same order, as build_features.py

    # noise: rest segments of the 60 Hz sessions (raw EMG, mean-centred)
    noise = []
    for s, d in train.items():
        if setup[s] != 60:
            continue
        for seg in segs[s][segs[s].label == "rest"].itertuples():
            x = d.iloc[seg.row0:seg.row1][EMG].to_numpy(float)
            noise.append((s, x - x.mean(0)))
    # typical rest power of each 50 Hz session
    rest50 = {}
    for s, d in train.items():
        if setup[s] == 50:
            rest50[s] = float(np.median([emg_power(d.iloc[r.row0:r.row1][EMG].to_numpy(float))
                                         for r in segs[s][segs[s].label == "rest"].itertuples()
                                         if r.row1 - r.row0 >= 120]))

    arng = np.random.default_rng(AUG_SEED)
    lens = np.array([len(x) for _, x in noise])
    jobs = []
    for s, ws in windows.items():
        if setup[s] != 50 or (args.sessions and s not in args.sessions):
            continue
        j = []
        for w in ws:
            n = w.row1 - w.row0
            ok = np.flatnonzero(lens >= n)
            if len(ok) == 0:
                continue
            k = int(arng.choice(ok))
            start = int(arng.integers(0, lens[k] - n + 1))
            j.append((w, k, start, float(arng.uniform(-JITTER, JITTER))))
        jobs.append((s, j))

    workers = min(8, os.cpu_count() or 2)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(train, noise, rest50, args.noise_ratio)) as ex:
        df = pd.DataFrame([r for rows in ex.map(_session_rows, jobs) for r in rows])

    ref = pd.read_pickle(OUT / "train.pkl")
    if list(df.columns[8:]) != [c for c in ref.columns if c[:2] in ("A_", "B_", "S_")]:
        sys.exit("feature columns differ from train.pkl; rerun scripts/build_features.py first")
    key = ["session", "seg", "label", "kind", "offset", "n_rows"]
    m = ref[key].merge(df[key], on=key, how="right", indicator=True)
    if (m["_merge"] != "both").any():
        sys.exit("simulated windows do not match train.pkl's windows; rerun scripts/build_features.py first")
    feats = [c for c in df.columns if c[:2] in ("A_", "B_", "S_")]
    bad = (~np.isfinite(df[feats].to_numpy())).sum()
    df.attrs["noise_ratio"] = args.noise_ratio
    df.to_pickle(OUT / args.out)
    print(f"{args.out}: noise ratio {args.noise_ratio}; {len(df)} simulated windows from {df.session.nunique()} 50 Hz sessions, "
          f"noise from {df.noise_session.nunique()} 60 Hz sessions; amplitude scale median {df.scale.median():.2f}; "
          f"{len(noise)} noise segments; non-finite values: {bad} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
