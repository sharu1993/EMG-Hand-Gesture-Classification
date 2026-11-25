"""Window features for both stages (doc "Preprocessing").

Stage 1, rest vs gesture:
  A  absolute: log band power per channel and averaged over channels (27)
  B  scale-free: time course of activity within the window, spectral shape,
     IMU onset movement (no absolute amplitude)
Stage 2, which gesture (overall amplitude removed first):
  per band, the 8x8 covariance in log-Euclidean tangent space split into shape
  (overall power divided out, 36) and scale (1); the correlation matrix's
  log (28 off-diagonal); plus the IMU features.

All features are means, ratios or positions, never sums or counts, so the
window's row count cannot carry the class.
"""
from __future__ import annotations

import numpy as np

from .preprocess import BANDS, band, clean_emg, fresh_imu

N_SUB = 7          # sub-windows for the time course (0.2 s each in a 1.4 s window)
EPS = 1e-12
_IU = np.triu_indices(8)
_IU_OFF = np.triu_indices(8, k=1)
_DIAG = np.flatnonzero(_IU[0] == _IU[1])


def _logm_spd(c: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(c)
    return (v * np.log(np.maximum(w, EPS))) @ v.T


def _bname(lo, hi):
    return f"{int(lo)}_{int(hi)}"


# ------------------------------------------------------------------ IMU (both stages)

def imu_features(imu: np.ndarray, fresh: np.ndarray) -> dict:
    x = fresh_imu(imu, fresh)
    gm = np.linalg.norm(x[:, 3:6], axis=1)               # gyro magnitude (deg/s), centred axes
    am = np.linalg.norm(x[:, 0:3], axis=1)               # accel deviation from the window mean
    n = len(gm)
    k = max(1, int(round(0.3 * n)))                       # first 30% of the window, about 0.4 s
    return {"imu_gyro_mean": gm.mean(), "imu_gyro_max": gm.max(), "imu_gyro_std": gm.std(),
            "imu_gyro_argmax": np.argmax(gm) / max(1, n - 1),
            "imu_gyro_early_minus_late": gm[:k].mean() - gm[k:].mean() if n > k else 0.0,
            "imu_acc_mean": am.mean(), "imu_acc_std": am.std()}


# ------------------------------------------------------------------ stage 1

def stage1_features(emg: np.ndarray, imu: np.ndarray, fresh: np.ndarray) -> dict:
    y = clean_emg(emg)
    f = {}
    bands = {b: band(y, *b) for b in BANDS}
    for b, z in bands.items():
        p = np.log((z ** 2).mean(0) + EPS)
        for c in range(8):
            f[f"A_logp_{_bname(*b)}_ch{c + 1}"] = p[c]
        f[f"A_logp_{_bname(*b)}_mean"] = np.log((z ** 2).mean() + EPS)
    # B: time course of activity (all bands together, mean over channels), relative to its own mean
    act = sum((z ** 2).mean(1) for z in bands.values())
    sub = np.array([s.mean() for s in np.array_split(act, N_SUB)])
    ls = np.log(sub + EPS) - np.log(sub + EPS).mean()
    for i, v in enumerate(ls):
        f[f"B_time_{i}"] = v
    f["B_time_range"] = ls.max() - ls.min()
    f["B_time_argmax"] = float(np.argmax(ls)) / (N_SUB - 1)
    f["B_time_slope"] = np.polyfit(np.arange(N_SUB), ls, 1)[0]
    # B: spectral shape (ratios between bands, mean over channels)
    pw = {b: (z ** 2).mean() for b, z in bands.items()}
    lo, mid, hi = BANDS
    f["B_spec_low_over_high"] = np.log((pw[lo] + EPS) / (pw[hi] + EPS))
    f["B_spec_mid_over_high"] = np.log((pw[mid] + EPS) / (pw[hi] + EPS))
    f["B_spec_low_over_mid"] = np.log((pw[lo] + EPS) / (pw[mid] + EPS))
    # B: spread of activity across channels (scale-free)
    cp = np.log((y ** 2).mean(0) + EPS)
    f["B_chan_spread"] = cp.std()
    for k, v in imu_features(imu, fresh).items():
        f["B_" + k] = v
    return f


# ------------------------------------------------------------------ stage 2

def stage2_features(emg: np.ndarray, imu: np.ndarray, fresh: np.ndarray) -> dict:
    y = clean_emg(emg)
    y = y / (np.sqrt((y ** 2).mean()) + EPS)              # gain normalisation: one number per window
    f = {}
    for b in BANDS:
        z = band(y, *b)
        c = z.T @ z / len(z)
        c = c + 1e-6 * np.trace(c) / 8 * np.eye(8)
        t = _logm_spd(c)
        scale = np.trace(t) / 8                           # log of the geometric-mean eigenvalue
        shape = t - scale * np.eye(8)
        v = shape[_IU]
        v[np.setdiff1d(np.arange(len(v)), _DIAG)] *= np.sqrt(2)   # tangent-space norm
        bn = _bname(*b)
        for j, (i0, i1) in enumerate(zip(*_IU)):
            f[f"S_shape_{bn}_{i0 + 1}_{i1 + 1}"] = v[j]
        f[f"S_scale_{bn}"] = scale
        d = np.sqrt(np.diag(c))
        r = c / np.outer(d, d)
        lr = _logm_spd(r + 1e-6 * np.eye(8))
        for i0, i1 in zip(*_IU_OFF):
            f[f"S_corr_{bn}_{i0 + 1}_{i1 + 1}"] = lr[i0, i1] * np.sqrt(2)
    for k, v in imu_features(imu, fresh).items():
        f["S_" + k] = v
    return f
