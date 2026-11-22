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

Person-variation augmentation (``augment_person_signal``, optional), aimed at
the sessions that fail when held out. In those, gestures change the EMG much
less relative to rest (some 8% vs 21% in the others), or the whole signal is
weaker and closer to the noise floor. Two changes per window, both applied to
every training window alike and never at test time:

* Background mixing: band-limited (20-120 Hz) noise with a random level per
  channel is added, so the gesture's own activity is a smaller part of the
  window (lower contrast and signal-to-noise ratio, as in those sessions).
* Spectral tilt: the spectrum is multiplied by (f / 60 Hz)^a with a random
  slope a (one per window plus a small per-channel part), mimicking how skin,
  fat and electrode contact filter the signal differently from person to person.
The window's overall level is restored afterwards, so only its shape changes.

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

# person-variation augmentation
BG_PROB = 0.5             # share of windows that get background mixed in
BG_LEVEL = (0.15, 1.0)    # background RMS relative to the window's, log-uniform
BG_CH_SD = 0.3            # per-channel log-spread of the background level
TILT_PROB = 0.5           # share of windows that get a spectral tilt
TILT_SD = 0.5             # slope per window (power law exponent on amplitude)
TILT_CH_SD = 0.15         # extra slope per channel
BG_BAND = (20.0, 120.0)


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


def augment_person_signal(emg: np.ndarray, rng: np.random.Generator,
                          fs: float = 250.0) -> np.ndarray:
    """(T, 8) filtered, gain-normalised EMG -> copy that looks like a person
    with weaker gesture contrast and/or different tissue filtering."""
    T, n_ch = emg.shape
    level = np.sqrt((emg ** 2).mean()) + 1e-9
    f = np.fft.rfftfreq(T, 1 / fs)
    out = emg
    if rng.random() < TILT_PROB:
        slope = rng.normal(0, TILT_SD) + rng.normal(0, TILT_CH_SD, n_ch)       # (8,)
        gain = (np.maximum(f, 1.0)[:, None] / 60.0) ** slope[None, :]         # (F, 8)
        out = np.fft.irfft(np.fft.rfft(out, axis=0) * gain, n=T, axis=0)
    if rng.random() < BG_PROB:
        spec = rng.normal(size=(len(f), n_ch)) + 1j * rng.normal(size=(len(f), n_ch))
        spec[(f < BG_BAND[0]) | (f > BG_BAND[1])] = 0
        bg = np.fft.irfft(spec, n=T, axis=0)
        bg /= np.sqrt((bg ** 2).mean(axis=0, keepdims=True)) + 1e-12             # unit RMS per channel
        rel = np.exp(rng.uniform(*np.log(BG_LEVEL)) + rng.normal(0, BG_CH_SD, n_ch))
        out = out + bg * rel[None, :] * level
    return (out * (level / (np.sqrt((out ** 2).mean()) + 1e-9))).astype(emg.dtype)


def augment_imu_signal(imu: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return imu * np.exp(rng.normal(0, 0.1, imu.shape[1])) + rng.normal(0, 0.05, imu.shape)
