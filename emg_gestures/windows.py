"""Duration-agnostic training windows for the Aura gesture data.

Whole segments leak their length (gestures ~3 s, rest ~1.3 s) and the device's
bursty row rate, so the model never sees them directly. Instead every sample is
a window of random length cut at a random position inside a labelled segment,
resampled onto an even time grid and mean-centred per channel, matching the
test-set format:

    1. draw a length L ~ U[min_len, max_len] seconds, the same way for every class
    2. pick a class uniformly (class-balanced), then a segment of that class at
       least L long, then a random start so the window fits inside it (after
       optional onset/offset trimming)
    3. regularise the row times (packet-arrival jitter) and resample EMG and the
       fresh IMU readings onto a uniform grid at ``fs`` Hz
    4. subtract each channel's mean over the window

Every class therefore produces windows from the same length distribution, so
length carries no information about the label. Batches mix lengths; ``pad_batch``
returns a mask so a model can pool over real samples only.

    from emg_gestures.segments import load_segments
    from emg_gestures.windows import WindowSampler, pad_batch

    segs = load_segments("data/train_segments.npz")
    sampler = WindowSampler(segs[segs.session.isin(train_sessions)], seed=0)
    batch = [sampler.sample() for _ in range(64)]
    emg, imu, mask, y = pad_batch(batch)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .segments import LABELS

LABEL_TO_ID = {lab: i for i, lab in enumerate(LABELS)}


@dataclass
class Window:
    emg: np.ndarray        # (n, 8) resampled, mean-centred
    imu: np.ndarray        # (n, 6) resampled, mean-centred
    label: str
    sample_id: str         # source segment
    session: int
    length_s: float        # window length in seconds
    offset_s: float        # start relative to the segment start (after trimming)
    n_rows_raw: int        # rows in the raw window, kept only for leakage checks


def regularize_time(t: np.ndarray) -> np.ndarray:
    """Row times on a straight line fitted to (row index, timestamp).

    Timestamps are packet-arrival times: rows come in bursts that share a
    timestamp. Within a short window the device samples at a near-constant
    rate, so a least-squares line through the timestamps gives each row a
    jitter-free time while keeping the window's real duration.
    """
    idx = np.arange(len(t), dtype=float)
    if len(t) < 2 or np.ptp(t) == 0:
        return t.astype(float)
    slope, intercept = np.polyfit(idx, t, 1)
    return intercept + slope * idx


def resample(t: np.ndarray, x: np.ndarray, t0: float, n: int, fs: float) -> np.ndarray:
    """Linear interpolation of x(t) onto t0 + k/fs, k = 0..n-1 (per column)."""
    grid = t0 + np.arange(n) / fs
    return np.column_stack([np.interp(grid, t, x[:, j]) for j in range(x.shape[1])])


def make_window(t, emg, imu, imu_new, t0, length_s, fs=250.0):
    """Cut [t0, t0 + length_s) from one segment's raw arrays and normalise it."""
    rows = np.flatnonzero((t >= t0) & (t < t0 + length_s))
    if len(rows) < 2:
        raise ValueError("window contains fewer than two rows")
    tr = regularize_time(t[rows])
    n = max(2, int(round(length_s * fs)))
    emg_w = resample(tr, emg[rows], tr[0], n, fs)
    fresh = rows[imu_new[rows] == 1]
    if len(fresh) >= 2:
        # IMU values are held between updates; interpolate fresh readings only,
        # at the regularised times of those same rows.
        imu_w = resample(tr[np.isin(rows, fresh)], imu[fresh], tr[0], n, fs)
    else:
        imu_w = np.repeat(imu[rows[:1]], n, axis=0)
    emg_w -= emg_w.mean(axis=0)
    imu_w -= imu_w.mean(axis=0)
    return emg_w, imu_w, len(rows)


class WindowSampler:
    """Random-length, random-offset, class-balanced windows from segments.

    Parameters
    ----------
    segments : DataFrame from ``load_segments`` (filter to the training sessions
        first; never sample validation sessions here).
    min_len, max_len : window length range in seconds.
    fs : output sample rate (Hz).
    trim_start, trim_end : seconds dropped from each end of gesture segments
        before sampling (the labels can lead the EMG by ~0.5 s). Rest segments
        are not trimmed. Defaults to 0 = use the label boundaries as given.
    balance : pick classes uniformly (True) or segments uniformly (False).
    """

    def __init__(self, segments: pd.DataFrame, min_len: float = 0.75, max_len: float = 1.4,
                 fs: float = 250.0, trim_start: float = 0.0, trim_end: float = 0.0,
                 balance: bool = True, seed: int | None = None):
        if not 0 < min_len <= max_len:
            raise ValueError("need 0 < min_len <= max_len")
        self.min_len, self.max_len, self.fs = min_len, max_len, fs
        self.trim_start, self.trim_end = trim_start, trim_end
        self.balance = balance
        self.rng = np.random.default_rng(seed)

        usable = []
        for r in segments.itertuples(index=False):
            lo, hi = r.t_start, r.t_end
            if r.label != "rest":
                lo, hi = lo + trim_start, hi - trim_end
            if hi - lo >= min_len:
                usable.append((r, lo, hi))
        if not usable:
            raise ValueError("no segment is long enough for min_len")
        self.segments = usable
        self.dropped = len(segments) - len(usable)
        self.by_label = {}
        for i, (r, _, _) in enumerate(usable):
            self.by_label.setdefault(r.label, []).append(i)
        self.labels = sorted(self.by_label, key=LABEL_TO_ID.get)
        self.span = [hi - lo for _, lo, hi in usable]

    def sample(self) -> Window:
        # Draw the length first, identically for every class, then choose only
        # among segments long enough to hold it. Capping the length at each
        # segment's own duration would make rest windows shorter on average.
        length = self.rng.uniform(self.min_len, self.max_len)
        if self.balance:
            lab = self.labels[self.rng.integers(len(self.labels))]
            pool = self.by_label[lab]
        else:
            pool = range(len(self.segments))
        fits = [i for i in pool if self.span[i] >= length]
        if not fits:  # no segment of this class is that long: fall back to the longest
            fits = [max(pool, key=self.span.__getitem__)]
            length = self.span[fits[0]]
        i = fits[self.rng.integers(len(fits))]
        r, lo, hi = self.segments[i]
        t0 = self.rng.uniform(lo, hi - length)
        emg, imu, n_raw = make_window(r.t, r.emg, r.imu, r.imu_new, t0, length, self.fs)
        return Window(emg, imu, r.label, r.sample_id, r.session, length, t0 - lo, n_raw)

    def sample_many(self, n: int) -> list[Window]:
        return [self.sample() for _ in range(n)]


def fixed_windows(segments: pd.DataFrame, length: float = 1.4, stride: float = 0.5,
                  fs: float = 250.0) -> list[Window]:
    """Deterministic, test-like windows (fixed length, regular stride) for
    validation. Segments shorter than ``length`` yield no window."""
    out = []
    for r in segments.itertuples(index=False):
        t0 = r.t_start
        while t0 + length <= r.t_end + 1e-9:
            emg, imu, n_raw = make_window(r.t, r.emg, r.imu, r.imu_new, t0, length, fs)
            out.append(Window(emg, imu, r.label, r.sample_id, r.session, length,
                              t0 - r.t_start, n_raw))
            t0 += stride
    return out


def test_windows(test_df: pd.DataFrame, fs: float = 250.0) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Apply the same resampling and centring to the test actions.

    Returns (action_id, emg, imu) per action, in file order.
    """
    from .segments import EMG_COLS, IMU_COLS
    out = []
    for aid, a in test_df.groupby("action_id", sort=False):
        t = a["t"].to_numpy(dtype=float)
        length = t[-1] - t[0] + 1e-6
        emg, imu, _ = make_window(t, a[EMG_COLS].to_numpy(float), a[IMU_COLS].to_numpy(float),
                                  a["imu_new"].to_numpy(), t[0], length, fs)
        out.append((aid, emg, imu))
    return out


def pad_batch(windows: list[Window]):
    """Stack variable-length windows: zero-padded arrays plus a validity mask.

    Returns emg (B, T, 8), imu (B, T, 6), mask (B, T) bool, y (B,) int.
    Pool with the mask so padding never reaches the model's decision.
    """
    T = max(len(w.emg) for w in windows)
    B = len(windows)
    emg = np.zeros((B, T, 8), np.float32)
    imu = np.zeros((B, T, 6), np.float32)
    mask = np.zeros((B, T), bool)
    for b, w in enumerate(windows):
        n = len(w.emg)
        emg[b, :n], imu[b, :n], mask[b, :n] = w.emg, w.imu, True
    y = np.array([LABEL_TO_ID[w.label] for w in windows])
    return emg, imu, mask, y
