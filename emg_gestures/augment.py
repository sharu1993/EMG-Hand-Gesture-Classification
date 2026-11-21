"""Training-time augmentation in the signal domain (NumPy), and its settings.

Two changes from the first version, both aimed at differences between people
and band placements:

* Magnitude warping: each channel's gain varies smoothly over the window
  (random log-gains at a few knots, interpolated), on top of a constant
  per-channel gain. Signal magnitude warping is among the most effective
  sEMG augmentations in Tsinganos et al., Sensors 2020 (doi 10.3390/s20174892).
* Channel subsets: 0-3 channels are dropped at random (simulated lost
  contacts), instead of exactly one in 15% of windows. Training on channel
  subsets increases robustness to electrode shift (Pereira et al., arXiv
  2401.02773). The 8 Aura contacts have no known ring order, so channels are
  dropped rather than rotated.

The raw-signal and frame inputs apply the same ideas on the GPU in
scripts/train_tcn.py (``augment``); covariance (Riemannian) features are not
linear in the signal, so for that input the signal is augmented here, before
the features are computed.
"""
from __future__ import annotations

import numpy as np

GAIN_SD = 0.15            # constant log-gain per channel
WARP_SD = 0.15            # log-gain at each warp knot
WARP_KNOTS = 4
NOISE_SD = 0.05           # in units of the window gain (signals are gain-normalised)
DROP_PROBS = (0.60, 0.20, 0.12, 0.08)   # P(0, 1, 2, 3 channels dropped)


def n_dropped(rng: np.random.Generator, size=None):
    return rng.choice(len(DROP_PROBS), size=size, p=DROP_PROBS)


def warp_curves(rng: np.random.Generator, n: int, n_ch: int) -> np.ndarray:
    """(n, n_ch) multiplicative gains: constant per channel times a smooth warp."""
    knots = rng.normal(0, WARP_SD, (WARP_KNOTS, n_ch))
    grid = np.linspace(0, WARP_KNOTS - 1, n)
    warp = np.stack([np.interp(grid, np.arange(WARP_KNOTS), knots[:, c]) for c in range(n_ch)], 1)
    return np.exp(rng.normal(0, GAIN_SD, n_ch)[None, :] + warp)


def augment_emg_signal(emg: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(T, 8) gain-normalised EMG -> augmented copy."""
    out = emg * warp_curves(rng, len(emg), emg.shape[1])
    out = out + rng.normal(0, NOISE_SD, out.shape)
    k = n_dropped(rng)
    if k:
        dead = rng.choice(emg.shape[1], size=k, replace=False)
        out[:, dead] = rng.normal(0, 0.01, (len(out), k))     # lost contact: near-silent channel
    return out


def augment_imu_signal(imu: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return imu * np.exp(rng.normal(0, 0.1, imu.shape[1])) + rng.normal(0, 0.05, imu.shape)
