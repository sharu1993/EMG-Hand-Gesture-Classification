"""Short-time feature frames: the input representation for the frame TCN.

The feature baseline beat the raw-signal TCN, mostly through cross-channel
correlations and band powers, which a small stack of convolutions does not
learn easily from 28 sessions. Here those quantities are computed on short
frames and the TCN models how they evolve within the window.

Per frame (200 ms, hop 48 ms; causal: a frame only uses samples up to its end):
    EMG  log band power per channel, bands 20-45 / 55-95 / 105-125 Hz (24)
         correlation between every channel pair (28)
    IMU  mean and std of each axis (12)

Input is the output of ``emg_gestures.model.preprocess`` (EMG filtered and
divided by the window gain, IMU low-passed and scaled), so frames inherit the
same per-window normalisation as the baseline.

A 0.75 s window gives 12 frames, 1.2 s gives 21, 1.4 s gives 26.
"""
from __future__ import annotations

import numpy as np

FRAME = 50          # samples (200 ms at 250 Hz)
HOP = 12            # samples (48 ms)
BANDS = [(20, 45), (55, 95), (105, 125)]
N_CH = 8
PAIRS = [(i, j) for i in range(N_CH) for j in range(i + 1, N_CH)]

# Feature layout, used by augmentation: column indices per channel / per block.
N_BAND_FEATS = len(BANDS) * N_CH
EMG_DIM = N_BAND_FEATS + len(PAIRS)                     # 52
IMU_DIM = 12
BAND_COLS_OF_CHANNEL = [[b * N_CH + c for b in range(len(BANDS))] for c in range(N_CH)]
CORR_COLS_OF_CHANNEL = [[N_BAND_FEATS + k for k, (i, j) in enumerate(PAIRS) if c in (i, j)]
                        for c in range(N_CH)]


def _framed(x: np.ndarray) -> np.ndarray:
    """(T, C) -> (F, FRAME, C) views, frames ending at FRAME-1, FRAME-1+HOP, ..."""
    n = 1 + (len(x) - FRAME) // HOP
    if n < 1:
        raise ValueError(f"window shorter than one frame ({len(x)} < {FRAME} samples)")
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx]


def n_frames(n_samples: int) -> int:
    return 1 + (n_samples - FRAME) // HOP


def emg_frames(emg: np.ndarray, fs: float = 250.0) -> np.ndarray:
    """(T, 8) preprocessed EMG -> (F, 52) frame features."""
    fr = _framed(emg)                                           # (F, L, 8)
    fr = fr - fr.mean(axis=1, keepdims=True)
    spec = np.abs(np.fft.rfft(fr * np.hanning(FRAME)[None, :, None], axis=1)) ** 2
    f = np.fft.rfftfreq(FRAME, 1 / fs)
    bp = np.stack([spec[:, (f >= lo) & (f < hi)].sum(axis=1) for lo, hi in BANDS], axis=1)
    logbp = np.log(bp + 1e-6).reshape(len(fr), -1)              # (F, 3*8) band-major
    cov = np.einsum("flc,fld->fcd", fr, fr)
    sd = np.sqrt(np.einsum("fcc->fc", cov)) + 1e-9
    corr = cov / (sd[:, :, None] * sd[:, None, :])
    iu = np.triu_indices(N_CH, 1)
    return np.concatenate([logbp, corr[:, iu[0], iu[1]]], axis=1).astype(np.float32)


def imu_frames(imu: np.ndarray) -> np.ndarray:
    """(T, 6) preprocessed IMU -> (F, 12): per-axis mean and std per frame."""
    fr = _framed(imu)
    return np.concatenate([fr.mean(axis=1), fr.std(axis=1)], axis=1).astype(np.float32)


def window_frames(emg: np.ndarray, imu: np.ndarray, fs: float = 250.0):
    return emg_frames(emg, fs), imu_frames(imu)
