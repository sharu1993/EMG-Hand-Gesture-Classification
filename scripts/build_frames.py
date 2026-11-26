"""Per-frame features for the sequence model (aura/sequence.py), for every
training, validation and test window.

    python3 scripts/build_frames.py                      # power frames
    python3 scripts/build_frames.py --kind relations     # frames with channel relationships
    python3 scripts/build_frames.py --kind relations_car # the same on the common-average-referenced EMG
    python3 scripts/build_frames.py --kind relations --corrupt augch   # one-bad-electrode copies

--corrupt augch applies, window by window, the same bad electrode as
data/features/augch_{train,val}.pkl (scripts/build_aug_channel.py: channel
aug_channel multiplied by aug_factor) and writes
frames_<kind>_augch_{train,val}.npz (no test file).

Windows are drawn exactly as in scripts/build_features.py (same seed and
order), and the script checks that the rows line up with data/features/*.pkl.
Writes data/features/frames_{train,val,test}.npz for power frames and
data/features/frames_<kind>_{train,val,test}.npz for the other kinds (git-ignored).
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
_CORRUPT = {"on": False}   # --corrupt: each job carries (channel index, factor) per window, in order


def _init(train, kind, corrupt=False):
    _TRAIN.update(train)
    _KIND["f"] = FRAME_KINDS[kind]
    _CORRUPT["on"] = corrupt


def _frames(d, w, bad=None):
    e, i, fr = window_arrays(d, w)
    if bad is not None:
        ch, factor = bad
        e = e.copy()
        e[:, ch] *= factor
    return _KIND["f"](e, i, fr)


def _session(args):
    if _CORRUPT["on"]:
        s, ws, bad = args
        d = _TRAIN[s]
        return [(w.session, w.seg, w.kind, _frames(d, w, b)) for w, b in zip(ws, bad)]
    s, ws = args
    d = _TRAIN[s]
    return [(w.session, w.seg, w.kind, _frames(d, w)) for w in ws]


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
    ap.add_argument("--corrupt", choices=["augch"], help="build the frames of the one-bad-electrode copies instead")
    args = ap.parse_args()
    prefix = "frames_" if args.kind == "power" else f"frames_{args.kind}_"
    if args.corrupt:
        prefix += f"{args.corrupt}_"
    t0 = time.time()
    train = load_train(DATA / "train", ROOT / "data/cache/train.pkl")
    test = load_test(DATA / "test", ROOT / "data/cache/test.pkl")
    rng = np.random.default_rng(SEED)
    jobs_tr, jobs_va = [], []
    for s, d in train.items():
        g = guided_segments(d, s)
        jobs_tr.append((s, training_windows(d, g, rng)))
        jobs_va.append((s, validation_windows(d, g)))
    if args.corrupt:
        for name, jobs in (("train", jobs_tr), ("val", jobs_va)):
            aug = pd.read_pickle(OUT / f"{args.corrupt}_{name}.pkl")      # rows in the same order as the windows
            flat = [w for _, ws in jobs for w in ws]
            meta = np.array([(w.session, w.seg, w.kind, w.offset, w.n_rows) for w in flat], dtype=object)
            if len(aug) != len(flat) or not (aug[["session", "seg", "kind", "offset", "n_rows"]].to_numpy(object) == meta).all():
                sys.exit(f"{args.corrupt}_{name}.pkl does not match the windows; rerun scripts/build_aug_channel.py")
            bad = list(zip(aug.aug_channel.to_numpy(int) - 1, aug.aug_factor.to_numpy(float)))
            cjobs, k = [], 0
            for s, ws in jobs:
                cjobs.append((s, ws, bad[k:k + len(ws)]))
                k += len(ws)
            with ProcessPoolExecutor(min(8, os.cpu_count() or 2), initializer=_init,
                                     initargs=(train, args.kind, True)) as ex:
                rows = [r for part in ex.map(_session, cjobs) for r in part]
            _save(name, rows, pd.read_pickle(OUT / f"{name}.pkl"), prefix)
        print(f"done in {time.time() - t0:.0f} s")
        return
    with ProcessPoolExecutor(min(8, os.cpu_count() or 2), initializer=_init, initargs=(train, args.kind)) as ex:
        for name, jobs in (("train", jobs_tr), ("val", jobs_va)):
            rows = [r for part in ex.map(_session, jobs) for r in part]
            _save(name, rows, pd.read_pickle(OUT / f"{name}.pkl"), prefix)
        rows = list(ex.map(_test, test_arrays(test), chunksize=16))
    _save("test", [(a, f) for a, f in rows], pd.read_pickle(OUT / "test.pkl"), prefix)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
