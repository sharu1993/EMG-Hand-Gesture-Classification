"""Sequence model for the active branch: a small temporal conv net (TCN) over
per-frame features of the window (doc, "Later": a sequence model next to the trees).

Frames: 0.2 s (50 samples) every 0.048 s (12 samples) over the cleaned,
gain-normalised EMG, so a 1.4 s window gives about 25 frames. Per frame:
  log power per channel in the three feature bands (24), relative to the
  window's overall power (gain normalisation, as stage 2);
  mean of each IMU axis over the frame's fresh rows (6).
The TCN reads the frames in order, then pools over time (mean and max).

Three frame kinds: "power" (above), "relations" (below: longer frames that
add each frame's channel covariance shape and correlation) and "relations_car"
(the same frames on the common-average-referenced signal).
"""
from __future__ import annotations

import numpy as np

from .preprocess import BANDS, band, clean_emg

FRAME = 50
HOP = 12
MAX_FRAMES = 36            # 416 rows (the longest window) give 31 frames
N_EMG = 8 * len(BANDS)
N_FEAT = N_EMG + 6
EPS = 1e-12


def frame_features(emg: np.ndarray, imu: np.ndarray, fresh: np.ndarray) -> np.ndarray:
    """(n_frames, N_FEAT) float32 for one window (mean-centred rows, 250 Hz)."""
    y = clean_emg(emg)
    y = y / (np.sqrt((y ** 2).mean()) + EPS)
    pw = np.stack([band(y, *b) ** 2 for b in BANDS], axis=1)      # (n, 3, 8)
    n = len(y)
    starts = list(range(0, max(1, n - FRAME + 1), HOP))[:MAX_FRAMES]
    out = np.empty((len(starts), N_FEAT), np.float32)
    for i, a in enumerate(starts):
        seg = slice(a, min(n, a + FRAME))
        out[i, :N_EMG] = np.log(pw[seg].mean(0).reshape(-1) + EPS)
        f = fresh[seg]
        out[i, N_EMG:] = (imu[seg][f] if f.sum() >= 1 else imu[seg]).mean(0)
    return out


# ------------------------------------------------ frames with channel relationships
# Longer frames (0.4 s every 0.1 s, about 11 per 1.4 s window) so an 8x8 covariance
# per frame is not too noisy. Per frame, in this order:
#   log power per channel in the three bands (24), as the power frames (first, so
#   the gain augmentation in _augment applies to them);
#   covariance shape (36) and scale (1) over the whole cleaned band (20-117 Hz);
#   covariance shape in 20-45 Hz (36);
#   log of the correlation matrix, off-diagonal (28), whole band;
#   IMU mean over the frame's fresh rows (6).
# Shape and correlation are in log-Euclidean tangent space, as the window-level
# stage-2 features (aura.features), where the channel relationships carried most
# of the gesture information.
REL_FRAME = 100
REL_HOP = 25
_IU = np.triu_indices(8)
_IU_OFF = np.triu_indices(8, k=1)
_OFFW = np.where(_IU[0] == _IU[1], 1.0, np.sqrt(2.0))
N_REL = N_EMG + 37 + 36 + 28 + 6


def _logm(c):
    w, v = np.linalg.eigh(c)
    return (v * np.log(np.maximum(w, EPS))) @ v.T


def _shape_scale(z):
    c = z.T @ z / len(z)
    c = c + 1e-6 * np.trace(c) / 8 * np.eye(8)
    t = _logm(c)
    sc = np.trace(t) / 8
    return (t - sc * np.eye(8))[_IU] * _OFFW, sc, c


def relation_frame_features(emg: np.ndarray, imu: np.ndarray, fresh: np.ndarray) -> np.ndarray:
    """(n_frames, N_REL) float32 for one window (mean-centred rows, 250 Hz)."""
    y = clean_emg(emg)
    y = y / (np.sqrt((y ** 2).mean()) + EPS)
    bands = [band(y, *b) for b in BANDS]
    n = len(y)
    starts = list(range(0, max(1, n - REL_FRAME + 1), REL_HOP))[:MAX_FRAMES]
    out = np.empty((len(starts), N_REL), np.float32)
    for i, a in enumerate(starts):
        seg = slice(a, min(n, a + REL_FRAME))
        out[i, :N_EMG] = np.log(np.stack([(z[seg] ** 2).mean(0) for z in bands]).reshape(-1) + EPS)
        sh, sc, c = _shape_scale(y[seg])
        sh_low, _, _ = _shape_scale(bands[0][seg])
        d = np.sqrt(np.diag(c))
        lr = _logm(c / np.outer(d, d) + 1e-6 * np.eye(8))
        j = N_EMG
        out[i, j:j + 36] = sh; j += 36
        out[i, j] = sc; j += 1
        out[i, j:j + 36] = sh_low; j += 36
        out[i, j:j + 28] = lr[_IU_OFF] * np.sqrt(2.0); j += 28
        f = fresh[seg]
        out[i, j:] = (imu[seg][f] if f.sum() >= 1 else imu[seg]).mean(0)
    return out


# ------------------------------------------------ relation frames, common-average reference
# As the relation frames, on the EMG after common-average reference (each channel
# minus the mean of all 8, sample by sample; aura.features.car_features). The
# referenced channels sum to zero, so covariance is taken over the 7 independent
# directions (Helmert basis), and the correlation matrix (singular) is replaced by
# the share of each band's power that is common to all channels. Per frame:
#   log power per referenced channel in the three bands (24, first, for _augment);
#   covariance shape (28) and scale (1), whole band, 7 directions;
#   covariance shape in 20-45 Hz (28), 7 directions;
#   common share per band, of the signal before referencing (3);
#   IMU mean over the frame's fresh rows (6).
_IU7 = np.triu_indices(7)
_OFFW7 = np.where(_IU7[0] == _IU7[1], 1.0, np.sqrt(2.0))
N_REL_CAR = N_EMG + 29 + 28 + len(BANDS) + 6


def _helmert(n: int = 8) -> np.ndarray:
    h = np.zeros((n - 1, n))
    for k in range(1, n):
        h[k - 1, :k] = 1.0
        h[k - 1, k] = -k
        h[k - 1] /= np.sqrt(k * (k + 1))
    return h


_H = _helmert()


def _shape_scale7(z):
    w = z @ _H.T
    c = w.T @ w / len(w)
    c = c + 1e-6 * np.trace(c) / 7 * np.eye(7)
    t = _logm(c)
    sc = np.trace(t) / 7
    return (t - sc * np.eye(7))[_IU7] * _OFFW7, sc


def relation_car_frame_features(emg: np.ndarray, imu: np.ndarray, fresh: np.ndarray) -> np.ndarray:
    """(n_frames, N_REL_CAR) float32 for one window (mean-centred rows, 250 Hz)."""
    y = clean_emg(emg)
    yc = y - y.mean(1, keepdims=True)
    g = np.sqrt((yc ** 2).mean()) + EPS
    y, yc = y / g, yc / g
    bands = [band(y, *b) for b in BANDS]
    bands_c = [z - z.mean(1, keepdims=True) for z in bands]
    n = len(y)
    starts = list(range(0, max(1, n - REL_FRAME + 1), REL_HOP))[:MAX_FRAMES]
    out = np.empty((len(starts), N_REL_CAR), np.float32)
    for i, a in enumerate(starts):
        seg = slice(a, min(n, a + REL_FRAME))
        out[i, :N_EMG] = np.log(np.stack([(z[seg] ** 2).mean(0) for z in bands_c]).reshape(-1) + EPS)
        sh, sc = _shape_scale7(yc[seg])
        sh_low, _ = _shape_scale7(bands_c[0][seg])
        j = N_EMG
        out[i, j:j + 28] = sh; j += 28
        out[i, j] = sc; j += 1
        out[i, j:j + 28] = sh_low; j += 28
        for z in bands:
            zs = z[seg]
            out[i, j] = (zs.mean(1) ** 2).mean() / ((zs ** 2).mean() + EPS); j += 1
        f = fresh[seg]
        out[i, j:] = (imu[seg][f] if f.sum() >= 1 else imu[seg]).mean(0)
    return out


FRAME_KINDS = {"power": frame_features, "relations": relation_frame_features,
               "relations_car": relation_car_frame_features}


def pad(frames: list[np.ndarray]):
    """Stack variable-length frame arrays: (X (N, MAX_FRAMES, F), mask (N, MAX_FRAMES))."""
    X = np.zeros((len(frames), MAX_FRAMES, frames[0].shape[1]), np.float32)
    M = np.zeros((len(frames), MAX_FRAMES), bool)
    for i, f in enumerate(frames):
        X[i, :len(f)], M[i, :len(f)] = f, True
    return X, M


# ---------------------------------------------------------------- model (torch)

def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def pick_device(name: str = "auto"):
    torch, _ = _torch()
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_tcn(n_feat: int = N_FEAT, n_classes: int = 4, hidden: int = 64, dropout: float = 0.2):
    torch, nn = _torch()

    class Block(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(nn.Conv1d(hidden, hidden, 3, padding=d, dilation=d), nn.GroupNorm(8, hidden),
                                     nn.GELU(), nn.Dropout(dropout),
                                     nn.Conv1d(hidden, hidden, 3, padding=d, dilation=d), nn.GroupNorm(8, hidden))
            self.act = nn.GELU()

        def forward(self, h, m):
            return self.act(h + self.net(h)) * m

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Conv1d(n_feat, hidden, 1)
            self.blocks = nn.ModuleList([Block(d) for d in (1, 2, 4)])
            self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Dropout(dropout),
                                      nn.Linear(hidden, n_classes))

        def forward(self, x, mask):                       # x (B, T, F), mask (B, T) bool
            m = mask.unsqueeze(1).float()                 # (B, 1, T)
            h = self.inp(x.transpose(1, 2)) * m
            for b in self.blocks:
                h = b(h, m)
            mean = h.sum(2) / m.sum(2).clamp(min=1.0)
            mx = h.masked_fill(m == 0, -1e4).amax(2)
            return self.head(torch.cat([mean, mx], 1))

    return TCN()


class Standardizer:
    def __init__(self, X: np.ndarray, M: np.ndarray):
        v = X[M]
        self.mu, self.sd = v.mean(0), v.std(0) + 1e-6
        self.lo = ((v - self.mu) / self.sd).min(0)          # stands in for a dropped channel

    def __call__(self, X: np.ndarray, M: np.ndarray) -> np.ndarray:
        return (((X - self.mu) / self.sd) * M[..., None]).astype(np.float32)


def _augment(torch, x, m, lo, gen, gain_sd=0.3, p_drop=0.15, noise=0.05):
    """Per-channel gain jitter (adds 2 ln g to the channel's three band powers,
    in standardised units approximately), one dropped channel with prob p_drop,
    small feature noise. Applied on the device every batch."""
    B, T, _ = x.shape
    dev = x.device
    # random numbers are drawn on the CPU (gen is a CPU generator) and moved to the device
    g = (torch.randn(B, 1, 8, generator=gen) * gain_sd).to(dev)
    x = x.clone()
    x[:, :, :N_EMG] += g.repeat(1, 1, len(BANDS))
    drop = (torch.rand(B, generator=gen) < p_drop).to(dev)
    ch = torch.randint(0, 8, (B,), generator=gen).to(dev)
    for b in range(len(BANDS)):
        idx = b * 8 + ch
        rows = torch.arange(B, device=dev)[drop]
        x[rows, :, idx[drop]] = lo[idx[drop]].unsqueeze(1)
    x = x + noise * torch.randn(x.shape, generator=gen).to(dev)
    return x * m.unsqueeze(-1)


def train_tcn(X, M, y, device, seed=0, epochs=40, batch=256, lr=2e-3, verbose=False):
    """Fit a TCN on standardised frames. y: int class ids. Returns (model, standardizer)."""
    torch, nn = _torch()
    torch.manual_seed(seed)
    std = Standardizer(X, M)
    xt = torch.from_numpy(std(X, M)).to(device)
    mt = torch.from_numpy(M).to(device)
    yt = torch.from_numpy(np.asarray(y, np.int64)).to(device)
    lo = torch.from_numpy(std.lo.astype(np.float32)).to(device)
    gen = torch.Generator().manual_seed(seed)              # CPU generator, works with every device
    model = build_tcn(n_feat=X.shape[2], n_classes=int(yt.max()) + 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    steps = epochs * int(np.ceil(len(yt) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(yt), generator=gen).to(device)
        tot = 0.0
        for i in range(0, len(yt), batch):
            b = perm[i:i + batch]
            out = model(_augment(torch, xt[b], mt[b], lo, gen), mt[b])
            loss = lossf(out, yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(b)
        if verbose and (ep % 10 == 9 or ep == epochs - 1):
            print(f"    epoch {ep + 1}: loss {tot / len(yt):.3f}", flush=True)
    return model, std


def predict_tcn(model, std, X, M, device, batch=1024) -> np.ndarray:
    torch, _ = _torch()
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            x = torch.from_numpy(std(X[i:i + batch], M[i:i + batch])).to(device)
            m = torch.from_numpy(M[i:i + batch]).to(device)
            out.append(torch.softmax(model(x, m), 1).cpu().numpy())
    return np.concatenate(out)
