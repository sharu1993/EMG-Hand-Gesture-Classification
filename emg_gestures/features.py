"""Hand-crafted EMG/IMU features for one window (baseline models).

Input is a window from ``emg_gestures.windows`` (rows taken as consecutive
250 Hz samples, mean-centred). Every feature is a rate, a ratio or a log-power, so none of them
scales with the window length.

EMG preprocessing (per window, identical for train, test and live use):
    PiEEG's filters: band-pass 20-120 Hz, band-stop 45-55 Hz (50 Hz mains),
    notch at 100 Hz; plus band-stop 55-65 Hz and notch at 120 Hz (60 Hz mains);
    optional common-average reference, then one gain for all channels
    (window RMS over all channels) so that the cross-channel pattern survives
    but the per-person/per-session amplitude does not.

Feature groups (names are prefixed so they can be selected or ablated):
    emg_rel_*   per-channel log RMS relative to the window's mean log RMS
    emg_wl_*    waveform length per sample (gain-normalised)
    emg_zc_*    zero-crossing rate, emg_ssc_* slope-sign-change rate
    emg_bp*_*   relative log band power: 20-45, 55-95, 105-125 Hz
    emg_hj*_*   Hjorth mobility and complexity
    emg_corr_*  cross-channel correlations (28 pairs)
    emg_gain    window log RMS over all channels (absolute amplitude:
                subject-dependent, excluded unless asked for)
    imu_*       per-axis std, peak-to-peak and mean |diff| of accel and gyro,
                gyro magnitude stats
    emg_riem*   (riemann=True) multi-band covariance in the tangent space,
                see emg_gestures/riemann.py
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos

from .riemann import feature_names as riemann_feature_names
from .riemann import riemann_window

N_CH = 8
BANDS = [(20, 45), (55, 95), (105, 125)]
IMU_AXES = ["ax", "ay", "az", "gx", "gy", "gz"]


# PiEEG's AuraNet preprocessing (huggingface.co/pieeg/Aura_Net): band-pass
# 20-120 Hz (order 4), band-stop 45-55 Hz (order 4) for 50 Hz mains, notch at
# its 100 Hz harmonic (Q = 30).
PIEEG_BANDPASS = (20.0, 120.0)
PIEEG_BANDSTOP = (45.0, 55.0)
PIEEG_NOTCH_HZ, PIEEG_NOTCH_Q = 100.0, 30.0
# Added to PiEEG's set, for 60 Hz mains: a 55-65 Hz band-stop (order 4, the
# same design as PiEEG's 50 Hz band-stop) and a notch at the 120 Hz harmonic.
# 7 of 28 training sessions and about half the test actions (test_3, test_5)
# were recorded with 60 Hz mains; with PiEEG's filters alone that hum stayed at
# 135-912x the background and cost those sessions accuracy. A narrow 60 Hz
# notch (Q = 30) only cut it to 7-49x: it needs ~0.5 s to settle, too long
# inside a 1.2 s window, while the band-stop removes it fully.
EXTRA_BANDSTOP = (55.0, 65.0)
EXTRA_NOTCHES_HZ = (120.0,)


@lru_cache(maxsize=8)
def _filters(fs: float):
    hi = min(PIEEG_BANDPASS[1], fs / 2 - 1)
    sos = [butter(4, [PIEEG_BANDPASS[0], hi], btype="bandpass", fs=fs, output="sos"),
           butter(4, list(PIEEG_BANDSTOP), btype="bandstop", fs=fs, output="sos"),
           butter(4, list(EXTRA_BANDSTOP), btype="bandstop", fs=fs, output="sos")]
    for f0 in (PIEEG_NOTCH_HZ, *EXTRA_NOTCHES_HZ):
        if f0 < fs / 2 - 1:
            sos.append(tf2sos(*iirnotch(f0, Q=PIEEG_NOTCH_Q, fs=fs)))
    return np.vstack(sos)


def filter_emg(emg: np.ndarray, fs: float = 250.0, car: bool = False) -> np.ndarray:
    """PiEEG's filters (band-pass 20-120 Hz, band-stop 45-55 Hz, 100 Hz notch)
    plus a 55-65 Hz band-stop and a 120 Hz notch for the 60 Hz-mains recordings.

    PiEEG runs them causally over a whole session and drops a 0.5 s warm-up.
    Our windows are only 0.75-1.4 s long, so the same filters are applied
    zero-phase (forward-backward) within each window instead: no warm-up is
    lost and no phase shift is introduced. For live streaming, the causal
    version over the continuous signal is the equivalent."""
    sos = _filters(fs)
    y = sosfiltfilt(sos, emg, axis=0, padtype="even", padlen=min(len(emg) - 1, 3 * 2 * len(sos)))
    if car:
        y = y - y.mean(axis=1, keepdims=True)
    return y


def _band_power(y: np.ndarray, fs: float) -> np.ndarray:
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))[:, None], axis=0)) ** 2
    f = np.fft.rfftfreq(len(y), 1 / fs)
    return np.stack([spec[(f >= lo) & (f < hi)].sum(axis=0) for lo, hi in BANDS])  # (3, ch)


def emg_features(emg: np.ndarray, fs: float = 250.0, car: bool = False) -> dict[str, float]:
    y = filter_emg(emg, fs, car)
    eps = 1e-9
    rms = np.sqrt((y ** 2).mean(axis=0)) + eps
    gain = np.sqrt((rms ** 2).mean())
    z = y / gain                                   # one gain for all channels
    log_rms = np.log(rms)
    out = {"emg_gain": float(np.log(gain))}

    dz = np.diff(z, axis=0)
    ddz = np.diff(dz, axis=0)
    thr = 0.05                                      # in window-gain units
    zc = ((z[:-1] * z[1:] < 0) & (np.abs(z[:-1] - z[1:]) > thr)).mean(axis=0)
    ssc = ((dz[:-1] * dz[1:] < 0) & (np.maximum(np.abs(dz[:-1]), np.abs(dz[1:])) > thr)).mean(axis=0)
    wl = np.abs(dz).mean(axis=0)
    var0, var1, var2 = z.var(axis=0) + eps, dz.var(axis=0) + eps, ddz.var(axis=0) + eps
    mob = np.sqrt(var1 / var0)
    comp = np.sqrt(var2 / var1) / mob
    bp = np.log(_band_power(y, fs) + eps)
    bp = bp - np.log((rms ** 2) * len(y) + eps)[None, :]   # relative to channel power

    for c in range(N_CH):
        k = c + 1
        out[f"emg_rel_{k}"] = float(log_rms[c] - log_rms.mean())
        out[f"emg_wl_{k}"] = float(wl[c])
        out[f"emg_zc_{k}"] = float(zc[c])
        out[f"emg_ssc_{k}"] = float(ssc[c])
        out[f"emg_hjm_{k}"] = float(mob[c])
        out[f"emg_hjc_{k}"] = float(comp[c])
        for b, (lo, hi) in enumerate(BANDS):
            out[f"emg_bp{lo}_{hi}_{k}"] = float(bp[b, c])
    corr = np.corrcoef(y.T)
    iu = np.triu_indices(N_CH, 1)
    for i, j, v in zip(iu[0], iu[1], corr[iu]):
        out[f"emg_corr_{i + 1}_{j + 1}"] = float(v)
    return out


def imu_features(imu: np.ndarray) -> dict[str, float]:
    out = {}
    d = np.abs(np.diff(imu, axis=0)).mean(axis=0)
    for a, name in enumerate(IMU_AXES):
        out[f"imu_std_{name}"] = float(imu[:, a].std())
        out[f"imu_ptp_{name}"] = float(np.ptp(imu[:, a]))
        out[f"imu_dmean_{name}"] = float(d[a])
    gmag = np.linalg.norm(imu[:, 3:6], axis=1)
    out["imu_gyro_mag_mean"] = float(gmag.mean())
    out["imu_gyro_mag_max"] = float(gmag.max())
    amag = np.linalg.norm(imu[:, 0:3], axis=1)
    out["imu_acc_mag_std"] = float(amag.std())
    return out


def riemann_features(emg: np.ndarray, fs: float = 250.0, corr: bool = False) -> dict[str, float]:
    """Multi-band covariance tangent vectors (emg_gestures/riemann.py) of the
    filtered, gain-normalised window: emg_riem<band>_<i>_<j>, 108 values; with
    ``corr`` also the correlation-matrix vectors emg_rcor<band>_<i>_<j>, 84 values."""
    y = filter_emg(emg, fs)
    y = y / (np.sqrt((y ** 2).mean()) + 1e-9)
    return dict(zip(riemann_feature_names(corr), riemann_window(y, fs, corr).tolist()))


def window_features(emg: np.ndarray, imu: np.ndarray, fs: float = 250.0,
                    car: bool = False, riemann: bool = False,
                    corr: bool = False) -> dict[str, float]:
    out = {**emg_features(emg, fs, car), **imu_features(imu)}
    if riemann:
        out.update(riemann_features(emg, fs, corr))
    return out
