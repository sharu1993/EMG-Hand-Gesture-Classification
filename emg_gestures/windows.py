"""Duration-agnostic training windows for the Aura gesture data.

Whole segments leak their length (gestures ~3 s, rest ~1.3 s) and the device's
bursty row rate, so the model never sees them directly. Instead every sample is
a window of random length cut at a random position inside the steady part of a labelled segment,
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
length carries no information about the label.

Transitions are excluded by default. Movement into a gesture starts ~0.3 s after
the label begins and settles by ~1.2 s; the release starts ~0.3 s before the
label ends and carries ~0.3 s into the following rest (gyro/EMG profiles across
all 28 sessions). ``STEADY_STATE`` trims those parts, so the model learns the
held posture rather than the movement between gestures. The steady part of a
gesture is ~1.4 s and of a rest ~1.25 s, hence the 1.2 s maximum length. Batches mix lengths; ``pad_batch``
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
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from .segments import LABELS

LABEL_TO_ID = {lab: i for i, lab in enumerate(LABELS)}

# Seconds removed from each segment end before windows are cut (see module doc).
STEADY_STATE = {"trim_start": 1.2, "trim_end": 0.4, "rest_trim_start": 0.3, "rest_trim_end": 0.0}
NO_TRIM = {"trim_start": 0.0, "trim_end": 0.0, "rest_trim_start": 0.0, "rest_trim_end": 0.0}
IMU_LOWPASS_HZ = 10.0  # below the slower IMU setup's Nyquist (~22 Hz)


# Use per-segment steady spans from emg_gestures.align.align_segments instead of
# fixed trims (the segments need steady_start / steady_end columns).
ALIGNED = {"aligned": True}


def usable_span(label: str, t_start: float, t_end: float, trims: dict,
                row=None) -> tuple[float, float]:
    """The part of a segment that windows may be cut from, after trimming."""
    if trims.get("aligned"):
        if row is None or not hasattr(row, "steady_start"):
            raise ValueError("ALIGNED trims need segments from emg_gestures.align.align_segments")
        return row.steady_start, row.steady_end
    if label == "rest":
        return t_start + trims["rest_trim_start"], t_end - trims["rest_trim_end"]
    return t_start + trims["trim_start"], t_end - trims["trim_end"]


@lru_cache(maxsize=4)
def _imu_lowpass(fs: float, cutoff: float):
    return butter(4, cutoff, btype="low", fs=fs, output="sos")


@dataclass
class Window:
    emg: np.ndarray        # (n, 8) resampled, mean-centred
    imu: np.ndarray        # (n, 6) resampled, mean-centred
    label: str
    sample_id: str         # source segment
    session: int
    length_s: float        # window length in seconds
    offset_s: float        # start relative to the start of the trimmed span
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


def make_window(t, emg, imu, imu_new, t0, length_s, fs=250.0, imu_lowpass=IMU_LOWPASS_HZ):
    """Cut [t0, t0 + length_s) from one segment's raw arrays and normalise it.

    The IMU is low-passed at ``imu_lowpass`` Hz after resampling. The two
    recording setups update the IMU at ~95 Hz and ~45 Hz; without a common
    bandwidth, IMU roughness identifies the setup instead of the gesture.
    """
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
    if imu_lowpass:
        imu_w = sosfiltfilt(_imu_lowpass(fs, imu_lowpass), imu_w, axis=0,
                            padtype="even", padlen=min(n - 1, 27))
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
    trims : seconds removed from segment ends before sampling; ``STEADY_STATE``
        (default) excludes the movements between gestures, ``NO_TRIM`` uses the
        label boundaries as given, ``ALIGNED`` uses each segment's own steady
        span found by emg_gestures.align.align_segments.
    balance : pick classes uniformly (True) or segments uniformly (False).

    ``n_fallback`` counts draws where no segment of the chosen class could hold
    the drawn length (the length was then shortened, which leaks length); it
    should stay at 0. Keep ``max_len`` at or below the shortest class's span.
    """

    def __init__(self, segments: pd.DataFrame, min_len: float = 0.75, max_len: float = 1.2,
                 fs: float = 250.0, trims: dict = STEADY_STATE,
                 balance: bool = True, seed: int | None = None):
        if not 0 < min_len <= max_len:
            raise ValueError("need 0 < min_len <= max_len")
        self.min_len, self.max_len, self.fs = min_len, max_len, fs
        self.trims = trims
        self.balance = balance
        self.rng = np.random.default_rng(seed)
        self.n_fallback = 0

        usable = []
        for r in segments.itertuples(index=False):
            lo, hi = usable_span(r.label, r.t_start, r.t_end, trims, r)
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
            self.n_fallback += 1
            fits = [max(pool, key=self.span.__getitem__)]
            length = self.span[fits[0]]
        i = fits[self.rng.integers(len(fits))]
        r, lo, hi = self.segments[i]
        t0 = self.rng.uniform(lo, hi - length)
        emg, imu, n_raw = make_window(r.t, r.emg, r.imu, r.imu_new, t0, length, self.fs)
        return Window(emg, imu, r.label, r.sample_id, r.session, length, t0 - lo, n_raw)

    def sample_many(self, n: int) -> list[Window]:
        return [self.sample() for _ in range(n)]


def fixed_windows(segments: pd.DataFrame, length: float = 1.2, stride: float = 0.5,
                  fs: float = 250.0, trims: dict = STEADY_STATE) -> list[Window]:
    """Deterministic windows (fixed length, regular stride) for validation,
    cut from the trimmed span of each segment. Segments whose span is shorter
    than ``length`` yield no window."""
    out = []
    for r in segments.itertuples(index=False):
        lo, hi = usable_span(r.label, r.t_start, r.t_end, trims, r)
        t0 = lo
        while t0 + length <= hi + 1e-9:
            emg, imu, n_raw = make_window(r.t, r.emg, r.imu, r.imu_new, t0, length, fs)
            out.append(Window(emg, imu, r.label, r.sample_id, r.session, length, t0 - lo, n_raw))
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
