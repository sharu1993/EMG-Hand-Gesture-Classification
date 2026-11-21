"""Per-segment label alignment: where is the hand actually holding the pose?

The guided labels follow the prompt, not the hand. People react late, take a
moment to form the gesture, and start releasing before the prompt ends. Fixed
trims (windows.STEADY_STATE: 1.2 s / 0.4 s) handle the average case; this
module finds the held part of each segment from the gyroscope instead, as in
the post-hoc alignment of prompts to actual gesture times used for Meta's
wristband data (Kaifosh et al., Nature 2025).

For each segment, gyro magnitude (fresh IMU readings, lightly smoothed) is
compared with a threshold set from the quiet middle of that segment:

    gesture:  held part starts after the last above-threshold movement in the
              first 2.0 s (the movement into the pose) and ends before the
              first above-threshold movement in the last 1.0 s (the release)
    rest:     starts after the release movement carried in from the previous
              gesture (first 0.8 s) and ends before the next movement starts
              (last 0.6 s)

A margin is kept on both sides. Where the movement is not visible (no
crossing) the segment keeps the fixed trims, and where the result is too
short for a window the segment falls back to the fixed trims as well, so no
segment is lost. Gesture starts are never earlier than 0.3 s after the label
(EMG follows the cue with a delay even when the wrist barely moves).

Only the training windows use alignment; validation keeps the fixed trims so
that scores stay comparable across runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .windows import STEADY_STATE

MARGIN = 0.15            # s kept clear of a detected movement
GESTURE_MIN_START = 0.3  # s after the label start
SEARCH = {"gesture": (2.0, 1.0), "rest": (0.8, 0.6)}   # (onset window, release window) in s
THRESH_FLOOR = 4.0       # deg/s above the quiet level
THRESH_MADS = 5.0


def _gyro_activity(t, imu, imu_new, smooth_s: float = 0.08):
    f = imu_new == 1
    tt, g = t[f], np.linalg.norm(imu[f, 3:6], axis=1)
    if len(g) < 5:
        return tt, g
    # mean spacing: timestamps arrive in bursts, so per-step differences are often 0
    dt = (tt[-1] - tt[0]) / (len(tt) - 1) if tt[-1] > tt[0] else 0.01
    k = min(len(g), max(1, int(round(smooth_s / max(dt, 1e-3)))))
    return tt, np.convolve(g, np.ones(k) / k, mode="same")


def align_segment(label, t, imu, imu_new, t_start, t_end, min_len: float = 0.75):
    """Return (steady_start, steady_end, method) for one segment."""
    kind = "rest" if label == "rest" else "gesture"
    fixed = (t_start + (STEADY_STATE["rest_trim_start"] if kind == "rest" else STEADY_STATE["trim_start"]),
             t_end - (STEADY_STATE["rest_trim_end"] if kind == "rest" else STEADY_STATE["trim_end"]))
    tt, g = _gyro_activity(t, imu, imu_new)
    if len(g) < 10:
        return (*fixed, "fixed:no_imu")
    on_win, off_win = SEARCH[kind]
    quiet = (tt > t_start + on_win) & (tt < t_end - off_win)
    if quiet.sum() < 5:                       # segment too short for a quiet reference
        quiet = (tt > t_start + 0.25 * (t_end - t_start)) & (tt < t_end - 0.25 * (t_end - t_start))
    if quiet.sum() < 3:
        return (*fixed, "fixed:short")
    base = np.median(g[quiet])
    thr = base + max(THRESH_FLOOR, THRESH_MADS * np.median(np.abs(g[quiet] - base)))

    head = (tt <= t_start + on_win) & (g > thr)
    tail = (tt >= t_end - off_win) & (g > thr)
    start = (tt[head].max() + MARGIN) if head.any() else (t_start if kind == "rest" else None)
    end = (tt[tail].min() - MARGIN) if tail.any() else t_end
    if kind == "gesture":
        start = fixed[0] if start is None else max(start, t_start + GESTURE_MIN_START)
    if end - start < min_len:
        return (*fixed, "fixed:too_short")
    return start, end, "aligned" if (head.any() or tail.any()) else "aligned:no_movement"


def align_segments(segs: pd.DataFrame, min_len: float = 0.75) -> pd.DataFrame:
    """Copy of ``segs`` with steady_start, steady_end and align_method columns."""
    out = [align_segment(r.label, r.t, r.imu, r.imu_new, r.t_start, r.t_end, min_len)
           for r in segs.itertuples(index=False)]
    segs = segs.copy()
    segs["steady_start"] = [o[0] for o in out]
    segs["steady_end"] = [o[1] for o in out]
    segs["align_method"] = [o[2] for o in out]
    return segs
