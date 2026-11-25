"""Per-window preprocessing (doc "Preprocessing", shared steps).

Every function works on one window alone, so training, the Kaggle test and the
live check see the same processing. Inputs are the window's rows as 250 Hz
samples, already mean-centred per channel (aura.windows).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import butter, sosfiltfilt

FS = 250.0
HIGHPASS = 20.0
NOTCHES = ((48.0, 52.0), (58.0, 62.0), (118.0, 122.0))   # 50 Hz, 60 Hz, 120 Hz (60 Hz harmonic)
BANDS = ((20.0, 45.0), (65.0, 95.0), (100.0, 117.0))     # clean in both setups
MAINS_SINES = (50.0, 60.0, 100.0, 120.0)                 # 100 Hz = 50 Hz 3rd harmonic folded at 250 Hz


@lru_cache(maxsize=None)
def _sos(kind: str, lo: float, hi: float | None = None):
    if kind == "highpass":
        return butter(4, lo, btype="highpass", fs=FS, output="sos")
    return butter(4, [lo, hi], btype=kind, fs=FS, output="sos")


def _filt(sos, x: np.ndarray) -> np.ndarray:
    """Zero-phase filter along time, padded by reflection (short windows: pad at most n - 1)."""
    pad = min(3 * (2 * len(sos) + 1), len(x) - 1)
    return sosfiltfilt(sos, x, axis=0, padtype="even", padlen=pad)


def subtract_mains(x: np.ndarray, freqs=MAINS_SINES) -> np.ndarray:
    """Least-squares fit and subtraction of sines at the mains frequencies.

    Within one window the hum is close to a pure sine, so fitting it removes it
    without the edge leakage of a notch on a short window (55-76 dB down in the
    feature bands vs 21-35 dB for notches alone, synthetic test). A t*sin and
    t*cos term per frequency absorb a small frequency offset (peaks sit at
    50.05 Hz) and a change of amplitude across the window. Must run after the
    high-pass: on raw EMG the sines partly fit slow drift and inject power."""
    n = len(x)
    t = (np.arange(n) - (n - 1) / 2) / FS
    cols = []
    for f in freqs:
        s, c = np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)
        cols += [s, c, t * s, t * c]
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, x, rcond=None)
    return x - X @ beta


def clean_emg(emg: np.ndarray) -> np.ndarray:
    """High-pass 20 Hz, subtract the mains sines, then band-stops at 50, 60 and
    120 Hz for what is left, on every window (no setup detection)."""
    y = _filt(_sos("highpass", HIGHPASS), np.asarray(emg, float))
    y = subtract_mains(y)
    for lo, hi in NOTCHES:
        y = _filt(_sos("bandstop", lo, hi), y)
    return y


def band(emg_clean: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return _filt(_sos("bandpass", lo, hi), emg_clean)


def fresh_imu(imu: np.ndarray, fresh: np.ndarray) -> np.ndarray:
    """IMU rows that carry a new sample (the rest repeat the last one), so features
    do not depend on the 43 vs 90 samples/s IMU rate. Falls back to all rows."""
    return imu[fresh] if fresh.sum() >= 3 else imu
