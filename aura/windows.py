"""Training and validation windows that copy the test actions.

A test action is a 1.4 s window by timestamp, starting about 0.5-0.6 s after a
guided segment's cue, mean-centred per channel. So a window here is:
  * cut by timestamp: rows with t0 + offset <= t < t0 + offset + 1.4;
  * gesture windows stay inside their segment;
  * rest windows stay inside their segment plus at most SPILL s past its end
    (a rest runs straight into the next gesture cue, and EMG rises only about
    0.3 s after a cue); a rest window that would run further is moved earlier,
    never before the rest cue, and skipped if it still does not fit;
  * at least MIN_ROWS rows (the smallest test action has 120).

Sets (doc "Windows"):
  testlike   offset random in TESTLIKE (training), N_TESTLIKE per segment
  steady     gestures: offset random in STEADY_GESTURE; rest: offset 0-0.15 s;
             N_STEADY per segment (the hold without the onset, for the live check)
  val        one per segment at 0.5 s (primary score)
  val_late   one per segment at 0.6 s
  val_steady one per segment: gestures at 1.0 s, rest at 0.0 s
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .io import EMG, IMU

LENGTH = 1.4
MIN_ROWS = 120
SPILL = 0.25
TESTLIKE = (0.3, 0.7)
STEADY_GESTURE = (0.8, 1.5)
STEADY_REST = (0.0, 0.15)
N_TESTLIKE = 8
N_STEADY = 4
VAL_OFFSETS = {"val": (0.5, 0.5), "val_late": (0.6, 0.6), "val_steady": (1.0, 0.0)}  # (gesture, rest)


@dataclass(frozen=True)
class Window:
    session: int
    seg: int
    label: str
    kind: str
    offset: float     # start, seconds after the segment's cue (after any move for rest)
    row0: int
    row1: int

    @property
    def n_rows(self) -> int:
        return self.row1 - self.row0


def _place(t: np.ndarray, seg, offset: float, kind: str) -> Window | None:
    """The window starting `offset` s after the cue, moved/skipped by the rules above."""
    limit = seg.t_end + (SPILL if seg.label == "rest" else 0.0)   # latest allowed end
    if seg.t0 + offset + LENGTH > limit:
        if seg.label != "rest":
            return None
        offset = limit - LENGTH - seg.t0
        if offset < 0:
            return None
    a = seg.t0 + offset
    r0 = int(np.searchsorted(t, a, side="left"))
    r1 = int(np.searchsorted(t, a + LENGTH, side="left"))
    if r1 - r0 < MIN_ROWS:
        return None
    return Window(int(seg.session), int(seg.seg), str(seg.label), kind, float(offset), r0, r1)


def training_windows(df: pd.DataFrame, segs: pd.DataFrame, rng: np.random.Generator) -> list[Window]:
    t = df["t"].to_numpy(float)
    out = []
    for seg in segs.itertuples():
        for kind, n, span in (("testlike", N_TESTLIKE, TESTLIKE),
                              ("steady", N_STEADY, STEADY_REST if seg.label == "rest" else STEADY_GESTURE)):
            for o in rng.uniform(*span, size=n):
                w = _place(t, seg, float(o), kind)
                if w is not None:
                    out.append(w)
    return out


def validation_windows(df: pd.DataFrame, segs: pd.DataFrame) -> list[Window]:
    t = df["t"].to_numpy(float)
    out = []
    for kind, (og, orest) in VAL_OFFSETS.items():
        for seg in segs.itertuples():
            w = _place(t, seg, orest if seg.label == "rest" else og, kind)
            if w is not None:
                out.append(w)
    return out


def window_arrays(df: pd.DataFrame, w: Window):
    """(emg (n, 8), imu (n, 6), imu_fresh (n,) bool) for a window, each channel
    mean-centred over the window's rows as in the test files."""
    part = df.iloc[w.row0:w.row1]
    emg = part[EMG].to_numpy(float)
    imu = part[IMU].to_numpy(float)
    return emg - emg.mean(0), imu - imu.mean(0), part["imu_new"].to_numpy() == 1


def test_arrays(actions: pd.DataFrame):
    """[(file, action_id, emg, imu, imu_fresh)] for the test actions, in file order.
    The test files are already mean-centred; centring again changes nothing."""
    out = []
    for (f, a), g in actions.groupby(["file", "action_id"], sort=False):
        emg = g[EMG].to_numpy(float)
        imu = g[IMU].to_numpy(float)
        out.append((int(f), a, emg - emg.mean(0), imu - imu.mean(0), g["imu_new"].to_numpy() == 1))
    return out


def windows_table(ws: list[Window]) -> pd.DataFrame:
    return pd.DataFrame([dict(session=w.session, seg=w.seg, label=w.label, kind=w.kind, offset=w.offset,
                              row0=w.row0, row1=w.row1, n_rows=w.n_rows) for w in ws])
