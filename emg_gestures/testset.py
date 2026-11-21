"""Load the competition test actions and turn them into model-ready windows.

Each test action is one 1.24-1.40 s window (rows in time order, t restarting at
0, channels mean-centred per action by the organisers). They go through the
same resampling, IMU low-pass and mean-centring as the training windows
(``windows.make_window`` via ``windows.test_windows``), so every model sees
test data exactly as it saw validation data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .segments import EMG_COLS, IMU_COLS
from .windows import test_windows

REQUIRED = ["action_id", "t", "imu_new", *EMG_COLS, *IMU_COLS]


def load_test_frame(test_dir: Path) -> pd.DataFrame:
    files = sorted(p for p in Path(test_dir).glob("test_*.xlsx") if not p.name.startswith("~$"))
    if not files:
        raise FileNotFoundError(f"no test_*.xlsx files in {test_dir}")
    parts = []
    for p in files:
        df = pd.read_excel(p, sheet_name=0, engine="openpyxl")
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"{p.name} is missing columns {missing}")
        parts.append(df[REQUIRED])
    df = pd.concat(parts, ignore_index=True)
    if df[REQUIRED].isna().any().any():
        raise ValueError("test data contains missing values")
    return df


def load_test_actions(test_dir: Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """[(action_id, emg (n, 8), imu (n, 6))] resampled to 250 Hz, mean-centred."""
    df = load_test_frame(test_dir)
    actions = test_windows(df)
    ids = [a for a, _, _ in actions]
    if len(set(ids)) != len(ids):
        raise ValueError("an action_id appears in more than one place in the test files")
    return actions
