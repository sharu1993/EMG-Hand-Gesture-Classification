"""Duration-agnostic training windows for the Aura gesture data.

Whole segments leak their length (gestures ~3 s, rest ~1.3 s), so the model
never sees them directly. Instead every sample is a window of random length cut
at a random position inside the steady part of a labelled segment, cut by
timestamp like the test actions and mean-centred per channel:

    1. draw a length L ~ U[min_len, max_len] seconds, the same way for every class
    2. pick a class uniformly (class-balanced), then a segment of that class at
       least L long, then a random start so the window fits inside it (after
       optional onset/offset trimming)
    3. use the selected rows as consecutive 250 Hz samples (the timestamps are
       packet arrival times, see ``make_window``); interpolate the IMU's fresh
       readings over row positions
    4. subtract each channel's mean over the window

Every class therefore produces windows from the same length distribution, so
length carries no information about the label (the number of rows in a window
also depends on the session's packet rate; scripts/leakage_check.py verifies
that neither predicts the label).

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
# A timestamp-cut window can hold very few rows where packets were dropped
# (as few as 12 rows in 0.75 s). Below this many samples (0.4 s at 250 Hz) a
# window is redrawn (training) or skipped (validation). Every test action has
# at least 120 rows.
MIN_ROWS = 100


class TooFewRows(ValueError):
    pass


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
    emg: np.ndarray        # (n, 8) rows as 250 Hz samples, mean-centred
    imu: np.ndarray        # (n, 6) fresh readings interpolated over rows, mean-centred
    label: str
    sample_id: str         # source segment
    session: int
    length_s: float        # window length in timestamp seconds (as the test actions)
    offset_s: float        # start relative to the start of the trimmed span
    n_rows_raw: int        # rows in the window (= samples the model sees)


def make_window(t, emg, imu, imu_new, t0, length_s, fs=250.0, imu_lowpass=IMU_LOWPASS_HZ,
                min_rows: int = MIN_ROWS):
    """Cut [t0, t0 + length_s) from one segment's raw arrays and normalise it.

    Rows are selected by timestamp, the same way the organisers cut the test
    actions, but the selected rows are then used as consecutive samples at the
    device rate (250 Hz), without resampling. The timestamps are packet arrival
    times, not sample times: counting rows at 250 Hz puts every session's mains
    peak within 0.2 Hz of exactly 50 or 60 Hz, while the timestamps put it ~5 Hz
    off (dropped packets lower the apparent rate to 163-240 rows/s). Resampling
    to the timestamps would stretch each window's frequency axis by a
    session-dependent factor and move mains off the filters.

    The IMU is updated on only some rows; its fresh readings are interpolated
    over row positions, then low-passed at ``imu_lowpass`` Hz. The two recording
    setups update the IMU at ~100 Hz and ~47 Hz; without a common bandwidth, IMU
    roughness identifies the setup instead of the gesture.
    """
    rows = np.flatnonzero((t >= t0) & (t < t0 + length_s))
    if len(rows) < max(min_rows, 2):
        raise TooFewRows(f"window holds {len(rows)} rows (< {min_rows})")
    n = len(rows)
    emg_w = emg[rows].astype(np.float64)
    pos = np.flatnonzero(imu_new[rows] == 1)
    if len(pos) >= 2:
        idx = np.arange(n)
        imu_w = np.column_stack([np.interp(idx, pos, imu[rows[pos], j]) for j in range(imu.shape[1])])
    else:
        imu_w = np.repeat(imu[rows[:1]].astype(np.float64), n, axis=0)
    if imu_lowpass and n > 3:
        imu_w = sosfiltfilt(_imu_lowpass(fs, imu_lowpass), imu_w, axis=0,
                            padtype="even", padlen=min(n - 1, 27))
    emg_w -= emg_w.mean(axis=0)
    imu_w -= imu_w.mean(axis=0)
    return emg_w, imu_w, n


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
        self.n_redrawn = 0      # windows redrawn because they held < MIN_ROWS rows

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
        for _ in range(100):
            try:
                return self._draw()
            except TooFewRows:
                self.n_redrawn += 1
        raise RuntimeError("could not draw a window with enough rows in 100 attempts")

    def _draw(self) -> Window:
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
            try:
                emg, imu, n_raw = make_window(r.t, r.emg, r.imu, r.imu_new, t0, length, fs)
                out.append(Window(emg, imu, r.label, r.sample_id, r.session, length, t0 - lo, n_raw))
            except TooFewRows:
                pass                                  # too few rows: skipped (see MIN_ROWS)
            t0 += stride
    return out


# Test-like windows. The test actions (v2) are 1.4 s windows that start about
# 0.5 s after the cue: matching the time profile of movement (gyro) and muscle
# activity inside the test windows to training windows cut at known offsets gives
# the best match at 0.5 s (scripts/analysis notes). So a test window holds the end
# of the movement into the gesture, then the hold, unlike the steady windows,
# which skip the first 1.2 s of each gesture.
TESTLIKE_START = (0.3, 0.7)      # training windows start this long after the cue
TESTLIKE_VAL_START = 0.5         # validation windows, as the test actions
TEST_LENGTH = 1.4
TRAIN_WINDOW_SETS = ("steady", "testlike", "both", "both_natural")


def testlike_windows(segments: pd.DataFrame, per_segment: int = 3, min_len: float = 0.75,
                     max_len: float = 1.2, fs: float = 250.0, seed: int = 0,
                     balance: str = "balanced") -> list[Window]:
    """Training windows that include the end of the onset movement.

    Each window's length is drawn first (min_len-max_len), independent of the
    class, as in WindowSampler, so length carries no label information. The
    start is drawn 0.3-0.7 s after the cue such that the window ends inside the
    segment (never crossing into the next one); segments too short for the drawn
    length are skipped. ``balance="balanced"`` keeps, per session, as many rest
    windows as the average gesture class has; ``"natural"`` keeps all of them
    (rest is then about half of the windows, so models lean towards rest)."""
    rng = np.random.default_rng(seed)
    out = []
    for r in segments.itertuples(index=False):
        for _ in range(per_segment):
            length = rng.uniform(min_len, max_len)
            hi = min(TESTLIKE_START[1], r.t_end - r.t_start - length)
            if hi < TESTLIKE_START[0]:
                continue
            t0 = r.t_start + rng.uniform(TESTLIKE_START[0], hi)
            try:
                emg, imu, n_raw = make_window(r.t, r.emg, r.imu, r.imu_new, t0, length, fs)
            except TooFewRows:
                continue
            out.append(Window(emg, imu, r.label, r.sample_id, r.session, length, t0 - r.t_start, n_raw))
    if balance == "natural":
        return out
    kept = []
    for s in sorted({w.session for w in out}):
        ws = [w for w in out if w.session == s]
        gest = [w for w in ws if w.label != "rest"]
        rest = [w for w in ws if w.label == "rest"]
        n_cls = len({w.label for w in gest})
        n = int(round(len(gest) / n_cls)) if n_cls else len(rest)
        pick = rng.choice(len(rest), size=min(n, len(rest)), replace=False) if rest else []
        kept += gest + [rest[i] for i in sorted(pick)]
    return kept


def testlike_val_windows(segments: pd.DataFrame, length: float = TEST_LENGTH,
                         start: float = TESTLIKE_VAL_START, min_len: float = 1.24,
                         fs: float = 250.0) -> list[Window]:
    """One validation window per segment, shaped like a test action: 1.4 s from
    0.5 s after the cue. Rest segments are often shorter than 1.9 s and the
    segment files hold no rows beyond a segment's end, so the window is moved
    earlier to fit (it then starts with the release of the previous gesture);
    segments shorter than ``min_len`` (the shortest test action) are skipped."""
    out = []
    for r in segments.itertuples(index=False):
        dur = r.t_end - r.t_start
        s0 = max(0.0, min(start, dur - length))
        L = min(length, dur - s0)
        if L < min_len:
            continue
        try:
            emg, imu, n_raw = make_window(r.t, r.emg, r.imu, r.imu_new, r.t_start + s0, L, fs)
        except TooFewRows:
            continue
        out.append(Window(emg, imu, r.label, r.sample_id, r.session, L, s0, n_raw))
    return out


def training_windows(segments: pd.DataFrame, kind: str, per_session: int, seed: int,
                     steady_trims: dict, fs: float = 250.0) -> list[Window]:
    """Training windows of one of TRAIN_WINDOW_SETS: steady (WindowSampler on the
    steady span, per_session per session, class-balanced), testlike, or both."""
    if kind not in TRAIN_WINDOW_SETS:
        raise ValueError(f"unknown training window set {kind!r}")
    out = []
    if kind != "testlike":
        for s, g in segments.groupby("session"):
            out += WindowSampler(g, trims=steady_trims, seed=seed * 1000 + s).sample_many(per_session)
    if kind != "steady":
        out += testlike_windows(segments, fs=fs, seed=seed + 4241,
                                balance="natural" if kind == "both_natural" else "balanced")
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
                                  a["imu_new"].to_numpy(), t[0], length, fs, min_rows=2)
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
