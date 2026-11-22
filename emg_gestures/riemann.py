"""Multi-band covariance features in the Riemannian tangent space.

Covariance matrices between EMG channels separate hand gestures well and are
the basis of recent cross-subject EMG decoders (Gowda & Miller, arXiv
2311.08548; Colot et al., arXiv 2604.22499, "Temporal Riemannian Regressor":
a GRU over sequences of multi-band Riemannian covariance features). Our own
baseline's strongest features were channel-pair correlations.

For each frequency band the window (or frame) gives an 8x8 covariance matrix
C, a symmetric positive definite (SPD) matrix. SPD matrices live on a curved
manifold, so they are mapped to a flat tangent space before ordinary layers
use them. We use the log-Euclidean map at the identity: logm(C), keeping the
upper triangle with off-diagonal entries scaled by sqrt(2) so Euclidean
distances between vectors equal log-Euclidean distances between matrices.
The diagonal of logm(C) carries per-channel band power, the off-diagonal the
co-activation pattern. No reference point is fitted, so the features need no
data from the new user (calibration-free); the model's per-feature
standardisation, fitted on training data only, centres them.

Input is preprocessed EMG (emg_gestures.model.preprocess: high-pass, mains
notches, one gain for all channels).

Per band: 36 values. Three bands: 108 values per frame or window.

Correlation variant (``corr=True``): each band's covariance is also scaled to a
correlation matrix R = D^-1/2 C D^-1/2 (D = its diagonal) and mapped the same
way. R carries no per-channel power, only the co-activation pattern, which
differs between gestures even for people whose gestures change channel power
very little. logm(R) is kept off the diagonal only: 28 values per band, 84 for
three bands, appended after the 108 covariance values (192 in total).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .frames import _framed

BANDS = [(20, 45), (55, 95), (105, 124)]
N_CH = 8
_IU = np.triu_indices(N_CH)
_SCALE = np.where(_IU[0] == _IU[1], 1.0, np.sqrt(2.0))
DIM_PER_BAND = len(_IU[0])                      # 36
DIM = DIM_PER_BAND * len(BANDS)                 # 108
_IU1 = np.triu_indices(N_CH, 1)
CORR_DIM_PER_BAND = len(_IU1[0])                # 28
CORR_DIM = CORR_DIM_PER_BAND * len(BANDS)       # 84
SHRINK = 1e-3                                   # ridge, as a fraction of mean variance


@lru_cache(maxsize=8)
def _band_sos(fs: float):
    return [butter(4, [lo, min(hi, fs / 2 - 1)], btype="band", fs=fs, output="sos")
            for lo, hi in BANDS]


def band_signals(emg: np.ndarray, fs: float = 250.0) -> list[np.ndarray]:
    """(T, 8) -> list of (T, 8), one zero-phase band-passed copy per band."""
    return [sosfiltfilt(sos, emg, axis=0, padtype="even", padlen=min(len(emg) - 1, 3 * 2 * len(sos)))
            for sos in _band_sos(fs)]


def _cov(x: np.ndarray) -> np.ndarray:
    """(..., L, 8) -> (..., 8, 8) covariance with a small ridge (keeps it SPD
    even when a channel is flat)."""
    x = x - x.mean(axis=-2, keepdims=True)
    c = np.einsum("...lc,...ld->...cd", x, x) / x.shape[-2]
    ridge = SHRINK * np.trace(c, axis1=-2, axis2=-1) / N_CH + 1e-8
    return c + ridge[..., None, None] * np.eye(N_CH)


def tangent(c: np.ndarray) -> np.ndarray:
    """(..., 8, 8) SPD -> (..., 36) log-Euclidean tangent vectors."""
    w, v = np.linalg.eigh(c)
    logc = np.einsum("...ij,...j,...kj->...ik", v, np.log(np.maximum(w, 1e-12)), v)
    return logc[..., _IU[0], _IU[1]] * _SCALE


def corr_tangent(c: np.ndarray) -> np.ndarray:
    """(..., 8, 8) SPD -> (..., 28): log-Euclidean tangent vector of the
    correlation matrix, off-diagonal entries only (scaled by sqrt(2))."""
    d = 1.0 / np.sqrt(np.diagonal(c, axis1=-2, axis2=-1))
    r = c * d[..., :, None] * d[..., None, :]
    w, v = np.linalg.eigh(r)
    logr = np.einsum("...ij,...j,...kj->...ik", v, np.log(np.maximum(w, 1e-12)), v)
    return logr[..., _IU1[0], _IU1[1]] * np.sqrt(2.0)


def riemann_frames(emg: np.ndarray, fs: float = 250.0, corr: bool = False) -> np.ndarray:
    """(T, 8) preprocessed EMG -> (F, 108), or (F, 192) with ``corr``; frames as
    in emg_gestures.frames."""
    covs = [_cov(_framed(xb)) for xb in band_signals(emg, fs)]
    feats = [tangent(c) for c in covs] + ([corr_tangent(c) for c in covs] if corr else [])
    return np.concatenate(feats, axis=1).astype(np.float32)


def riemann_window(emg: np.ndarray, fs: float = 250.0, corr: bool = False) -> np.ndarray:
    """(T, 8) preprocessed EMG -> (108,), or (192,) with ``corr``: one covariance
    per band over the window."""
    covs = [_cov(xb) for xb in band_signals(emg, fs)]
    feats = [tangent(c) for c in covs] + ([corr_tangent(c) for c in covs] if corr else [])
    return np.concatenate(feats).astype(np.float32)


def feature_names(corr: bool = False) -> list[str]:
    """Covariance names emg_riem<band>_<i>_<j>; with ``corr`` also the
    correlation names emg_rcor<band>_<i>_<j> (same order as riemann_window)."""
    names = []
    for lo, hi in BANDS:
        for i, j in zip(*_IU):
            names.append(f"emg_riem{lo}_{hi}_{i + 1}_{j + 1}")
    if corr:
        for lo, hi in BANDS:
            for i, j in zip(*_IU1):
                names.append(f"emg_rcor{lo}_{hi}_{i + 1}_{j + 1}")
    return names
