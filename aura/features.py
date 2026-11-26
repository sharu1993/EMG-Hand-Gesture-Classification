"""Window features for both stages (doc "Preprocessing").

Stage 1, rest vs gesture:
  A  absolute: log band power per channel and averaged over channels (27)
  B  scale-free: time course of activity within the window, spectral shape,
     IMU onset movement (no absolute amplitude)
Stage 2, which gesture (overall amplitude removed first):
  per band, the 8x8 covariance in log-Euclidean tangent space split into shape
  (overall power divided out, 36) and scale (1); the correlation matrix's
  log (28 off-diagonal); plus the IMU features.
Optional set I (imu_axis_features): direction of the movement per IMU axis.
Optional set C (car_features): the EMG after common-average reference.
Optional set T (td_features): time-domain shape of the EMG per channel.

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


# ------------------------------------------------------------------ IMU per axis (optional set I)

_AX = ("x", "y", "z")


def imu_axis_features(imu: np.ndarray, fresh: np.ndarray) -> dict:
    """Direction of the movement, per axis (prefix I_; 21 features).

    The window is mean-centred as in the test files, so absolute orientation is
    gone, but the change in orientation and the axis of rotation within the
    window are kept. The first 30% of the window (about 0.4 s) is compared with
    the rest, as in imu_features."""
    x = fresh_imu(imu, fresh)
    x = x - x.mean(0)
    acc, gyr = x[:, 0:3], x[:, 3:6]
    n = len(x)
    k = max(1, int(round(0.3 * n)))
    f = {}
    gs, as_ = gyr.std(0), acc.std(0)
    gv = gs ** 2
    for j, a in enumerate(_AX):
        f[f"I_gyro_rms_{a}"] = gs[j]
        f[f"I_gyro_share_{a}"] = gv[j] / (gv.sum() + EPS)                       # scale-free
        f[f"I_acc_rms_{a}"] = as_[j]
        late = slice(k, None) if n > k else slice(None)
        f[f"I_acc_shift_{a}"] = acc[late, j].mean() - acc[:k, j].mean()         # orientation change, signed
        f[f"I_gyro_early_{a}"] = gyr[:k, j].mean() - gyr[late, j].mean()        # early rotation, signed
    c = np.cov(gyr.T) if n > 2 else np.zeros((3, 3))
    d = np.sqrt(np.maximum(np.diag(c), EPS))
    for i0, i1 in ((0, 1), (0, 2), (1, 2)):
        f[f"I_gyro_corr_{_AX[i0]}{_AX[i1]}"] = c[i0, i1] / (d[i0] * d[i1]) if gv[i0] > EPS and gv[i1] > EPS else 0.0
    w, v = np.linalg.eigh(c + EPS * np.eye(3))
    for j, a in enumerate(_AX):
        f[f"I_gyro_dir_{a}"] = abs(v[j, -1])                                     # main rotation axis, sign-free
    return f


# ------------------------------------------------------------------ common-average reference (optional set C)

def _helmert(n: int = 8) -> np.ndarray:
    """(n-1, n) orthonormal rows, each orthogonal to the all-ones vector."""
    h = np.zeros((n - 1, n))
    for k in range(1, n):
        h[k - 1, :k] = 1.0
        h[k - 1, k] = -k
        h[k - 1] /= np.sqrt(k * (k + 1))
    return h


_H = _helmert()
_IU7 = np.triu_indices(7)
_DIAG7 = np.flatnonzero(_IU7[0] == _IU7[1])


def car_features(emg: np.ndarray, imu: np.ndarray, fresh: np.ndarray) -> dict:
    """The EMG after common-average reference (each channel minus the mean of
    all 8, sample by sample), prefix C_; 117 features.

    Per band: log power of each referenced channel and their mean (9); the share
    of the band's power that is common to all channels (1; 1/8 if the channels
    are independent, 1 if identical); and the covariance of the referenced
    signal, gain-normalised, in log-Euclidean tangent space split into shape (28)
    and scale (1). The referenced channels sum to zero, so the covariance is
    taken over the 7 independent directions (Helmert basis), which keeps it
    full rank. imu and fresh are unused (same signature as the other sets)."""
    y = clean_emg(emg)
    yc = y - y.mean(1, keepdims=True)
    g = np.sqrt((yc ** 2).mean()) + EPS
    f = {}
    for b in BANDS:
        bn = _bname(*b)
        z = band(y, *b)
        zc = z - z.mean(1, keepdims=True)
        p = np.log((zc ** 2).mean(0) + EPS)
        for c in range(8):
            f[f"C_logp_{bn}_ch{c + 1}"] = p[c]
        f[f"C_logp_{bn}_mean"] = np.log((zc ** 2).mean() + EPS)
        f[f"C_common_{bn}"] = (z.mean(1) ** 2).mean() / ((z ** 2).mean() + EPS)
        w = (zc / g) @ _H.T                                   # (n, 7)
        c7 = w.T @ w / len(w)
        c7 = c7 + 1e-6 * np.trace(c7) / 7 * np.eye(7)
        t = _logm_spd(c7)
        scale = np.trace(t) / 7
        v = (t - scale * np.eye(7))[_IU7]
        v[np.setdiff1d(np.arange(len(v)), _DIAG7)] *= np.sqrt(2)
        for j, (i0, i1) in enumerate(zip(*_IU7)):
            f[f"C_shape_{bn}_{i0 + 1}_{i1 + 1}"] = v[j]
        f[f"C_scale_{bn}"] = scale
    return f


# ------------------------------------------------------------------ time domain (optional set T)

TD_THRESH = 0.02      # zero-crossing / slope-change threshold, as a fraction of the channel's RMS


def td_features(emg: np.ndarray, imu: np.ndarray, fresh: np.ndarray) -> dict:
    """Classic time-domain EMG features of the cleaned signal, prefix T_; 45.

    Per channel, all scale-free (the absolute level is in set A): waveform length
    over RMS, zero-crossing rate, slope-sign-change rate (both with a threshold of
    TD_THRESH x RMS, so noise around zero does not count), Hjorth mobility (a mean
    frequency) and complexity (its spread); and their means over channels. Rates
    are per sample, so the window's length does not enter. imu and fresh are unused."""
    y = clean_emg(emg)
    rms = np.sqrt((y ** 2).mean(0)) + EPS
    d = np.diff(y, axis=0)
    dd = np.diff(d, axis=0)
    thr = TD_THRESH * rms
    wl = np.abs(d).mean(0) / rms
    zc = ((y[:-1] * y[1:] < 0) & (np.abs(d) >= thr)).mean(0)
    d1, d2 = y[1:-1] - y[:-2], y[1:-1] - y[2:]
    ssc = ((d1 * d2 > 0) & ((np.abs(d1) >= thr) | (np.abs(d2) >= thr))).mean(0)
    sx, sd, sdd = y.std(0) + EPS, d.std(0) + EPS, dd.std(0) + EPS
    mob = sd / sx
    comp = (sdd / sd) / mob
    f = {}
    for name, v in (("wl", wl), ("zc", zc), ("ssc", ssc), ("mob", mob), ("comp", comp)):
        for c in range(8):
            f[f"T_{name}_ch{c + 1}"] = float(v[c])
        f[f"T_{name}_mean"] = float(v.mean())
    return f


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
