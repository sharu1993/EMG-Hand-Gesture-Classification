"""Reading the competition files, guided segments, recording setup.

Facts this relies on (analysis of 25 Nov 2026):
  * rows are contiguous 250 Hz samples; `t` is packet arrival time (rows repeat
    timestamps, jumps up to 0.1 s, no samples lost at the jumps);
  * only rows with label_source == "guided" carry labels that match the test
    actions; other rows are transitions;
  * two recording setups: 50 Hz mains and 60 Hz mains (with a 120 Hz harmonic).
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch

FS = 250.0
EMG = [f"ch{i}" for i in range(1, 9)]
IMU = ["ax", "ay", "az", "gx", "gy", "gz"]
TRAIN_COLS = ["t", "emg_new", "imu_new", *EMG, *IMU, "label", "label_source"]
TEST_COLS = ["action_id", "t", "emg_new", "imu_new", *EMG, *IMU]
LABELS = ["rest", "fist", "open", "pinch", "point", "thumb-up"]
GESTURES = LABELS[1:]


def read_xlsx(path: Path) -> pd.DataFrame:
    """First sheet of an .xlsx file. Uses the fast calamine reader when installed."""
    try:
        return pd.read_excel(path, engine="calamine")
    except (ImportError, ValueError):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Workbook contains no default style")
            return pd.read_excel(path, engine="openpyxl")


def _cached(cache: Path | None, build):
    if cache is not None and cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    obj = build()
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(obj, f)
    return obj


def load_train(train_dir: Path, cache: Path | None = None) -> dict[int, pd.DataFrame]:
    """{session number: rows in file order}, the columns in TRAIN_COLS."""
    def build():
        out = {}
        for p in sorted(Path(train_dir).glob("train_*.xlsx"), key=lambda p: int(p.stem.split("_")[1])):
            out[int(p.stem.split("_")[1])] = read_xlsx(p)[TRAIN_COLS].reset_index(drop=True)
        return out
    return _cached(cache, build)


def load_test(test_dir: Path, cache: Path | None = None) -> pd.DataFrame:
    """All test actions, the columns in TEST_COLS plus `file` (1-5), rows in file order."""
    def build():
        parts = []
        for p in sorted(Path(test_dir).glob("test_*.xlsx"), key=lambda p: int(p.stem.split("_")[1])):
            d = read_xlsx(p)[TEST_COLS]
            d.insert(0, "file", int(p.stem.split("_")[1]))
            parts.append(d)
        return pd.concat(parts, ignore_index=True)
    return _cached(cache, build)


def guided_segments(df: pd.DataFrame, session: int) -> pd.DataFrame:
    """One row per guided segment (a run of rows with the same guided label).

    row0, row1  first row and one past the last row of the segment
    t0          cue time = timestamp of the first row
    t_end       timestamp of the first row after the segment (end of the segment)
    """
    guided = (df["label_source"] == "guided").to_numpy(bool)
    labels = df["label"].to_numpy(object)
    # a row's key is its label if it is a guided row with one of the six labels, else None
    key = np.array([lab if (g and lab in LABELS) else None for lab, g in zip(labels, guided)], dtype=object)
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    ends = np.r_[starts[1:], len(df)]
    t = df["t"].to_numpy(float)
    rows = []
    for r0, r1 in zip(starts, ends):
        if key[r0] is None:
            continue
        rows.append(dict(session=session, label=key[r0], row0=int(r0), row1=int(r1), t0=float(t[r0]),
                         t_end=float(t[r1]) if r1 < len(df) else float(t[-1])))
    seg = pd.DataFrame(rows)
    seg.insert(1, "seg", np.arange(len(seg)))
    seg["dur"] = seg["t_end"] - seg["t0"]
    return seg


def mains_hz(emg: np.ndarray) -> int:
    """50 or 60: whichever mains frequency carries more power (rows as 250 Hz samples)."""
    x = sosfiltfilt(butter(4, 15, btype="highpass", fs=FS, output="sos"), np.asarray(emg, float), axis=0)
    f, p = welch(x, fs=FS, nperseg=1024, axis=0)
    p = np.median(p, axis=1)
    near = lambda f0: p[(f > f0 - 1.5) & (f < f0 + 1.5)].max()  # noqa: E731
    return 50 if near(50) >= near(60) else 60
