"""Per-frame features for the sequence model (aura/sequence.py), for every
training, validation and test window.

    python3 scripts/build_frames.py                      # power frames
    python3 scripts/build_frames.py --kind relations     # frames with channel relationships

Windows are drawn exactly as in scripts/build_features.py (same seed and
order), and the script checks that the rows line up with data/features/*.pkl.
Writes data/features/frames_{train,val,test}.npz for power frames and
data/features/frames_relations_{train,val,test}.npz for relation frames (git-ignored).
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
from aura.io import guided_segments, load_test, load_train  # noqa: E402
from aura.sequence import FRAME_KINDS, pad  # noqa: E402
from aura.windows import test_arrays, training_windows, validation_windows, window_arrays  # noqa: E402

DATA = ROOT / "vr-calibration-free-emg-challenge-pieeg"
OUT = ROOT / "data/features"
SEED = 0                      # must match scripts/build_features.py
_TRAIN = {}
_KIND = {"f": None}


def _init(train, kind):
    _TRAIN.update(train)
    _KIND["f"] = FRAME_KINDS[kind]


def _session(args):
    s, ws = args
    d = _TRAIN[s]
    return [(w.session, w.seg, w.kind, _KIND["f"](*window_arrays(d, w))) for w in ws]


def _test(item):
    f, a, e, i, fr = item
    return a, _KIND["f"](e, i, fr)


def _save(name, rows, meta, prefix):
    X, M = pad([r[-1] for r in rows])
    if name != "test":
        ok = (np.array([r[0] for r in rows]) == meta.session.to_numpy()).all() and \
             (np.array([r[1] for r in rows]) == meta.seg.to_numpy()).all() and \
             (np.array([r[2] for r in rows]) == meta.kind.to_numpy()).all()
        if not ok:
            sys.exit(f"{name}: windows do not line up with data/features/{name}.pkl; rebuild both")
        np.savez_compressed(OUT / f"{prefix}{name}.npz", X=X, M=M)
    else:
        if not (np.array([r[0] for r in rows]) == meta.action_id.to_numpy()).all():
            sys.exit("test: actions do not line up with data/features/test.pkl")
        np.savez_compressed(OUT / f"{prefix}{name}.npz", X=X, M=M)
    print(f"{name}: {X.shape[0]} windows, up to {int(M.sum(1).max())} frames of {X.shape[2]} features")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=list(FRAME_KINDS), default="power")
    args = ap.parse_args()
    prefix = "frames_" if args.kind == "power" else f"frames_{args.kind}_"
    t0 = time.time()
    train = load_train(DATA / "train", ROOT / "data/cache/train.pkl")
    test = load_test(DATA / "test", ROOT / "data/cache/test.pkl")
    rng = np.random.default_rng(SEED)
    jobs_tr, jobs_va = [], []
    for s, d in train.items():
        g = guided_segments(d, s)
        jobs_tr.append((s, training_windows(d, g, rng)))
        jobs_va.append((s, validation_windows(d, g)))
    with ProcessPoolExecutor(min(8, os.cpu_count() or 2), initializer=_init, initargs=(train, args.kind)) as ex:
        for name, jobs in (("train", jobs_tr), ("val", jobs_va)):
            rows = [r for part in ex.map(_session, jobs) for r in part]
            _save(name, rows, pd.read_pickle(OUT / f"{name}.pkl"), prefix)
        rows = list(ex.map(_test, test_arrays(test), chunksize=16))
    _save("test", [(a, f) for a, f in rows], pd.read_pickle(OUT / "test.pkl"), prefix)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
