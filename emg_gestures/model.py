"""Dual-branch causal TCN for Aura gesture windows (PyTorch).

Three input representations share this network: the preprocessed signal
(``RAW_CONFIG``, below), short-time feature frames (``FRAME_CONFIG``, see
emg_gestures/frames.py) and multi-band Riemannian covariance frames
(``RIEMANN_CONFIG``, see emg_gestures/riemann.py).

    EMG (B, T, 8) -> stem conv -> 5 causal TCN blocks, dilations 1..16 --+
                                                                          +-> concat
    IMU (B, T, 6) -> stem conv -> 3 causal TCN blocks, dilations 1..4  --+
          -> masked attention pooling over time -> MLP -> 6 logits

Design choices
--------------
* Causal convolutions (left padding only): every output depends on the present
  and past, so the network can run as a streaming detector, and right-padding
  in a batch cannot change any valid time step.
* Depthwise-separable convolutions keep it small (~45k parameters).
* Normalisation is LayerNorm over channels at each time step: it uses no
  statistics over time, so padded steps never influence real ones (BatchNorm
  or GroupNorm over time would).
* Masked attention pooling: padded steps get zero weight, so any window length
  maps to the same computation. No layer depends on the input length.

Input preprocessing (``preprocess``) matches the feature baseline: EMG high-pass
+ mains notches, divided by one gain shared by all channels; IMU (already
low-passed and mean-centred by the window code) divided by fixed unit scales.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import filter_emg

IMU_SCALE = np.array([0.1, 0.1, 0.1, 20.0, 20.0, 20.0], dtype=np.float32)  # g, g, g, deg/s x3


def preprocess(emg: np.ndarray, imu: np.ndarray, fs: float = 250.0) -> tuple[np.ndarray, np.ndarray]:
    """One window -> model inputs (float32). Same function for train, test, live."""
    y = filter_emg(emg, fs)
    gain = np.sqrt((y ** 2).mean()) + 1e-9
    return (y / gain).astype(np.float32), (imu / IMU_SCALE).astype(np.float32)


class ChannelNorm(nn.Module):
    """LayerNorm over channels for (B, C, T) tensors."""

    def __init__(self, c: int):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class CausalSepConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int, dilation: int):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.dw = nn.Conv1d(c_in, c_in, k, dilation=dilation, groups=c_in)
        self.pw = nn.Conv1d(c_in, c_out, 1)

    def forward(self, x):
        return self.pw(self.dw(F.pad(x, (self.pad, 0))))


class TCNBlock(nn.Module):
    def __init__(self, c: int, k: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalSepConv(c, c, k, dilation)
        self.norm1 = ChannelNorm(c)
        self.conv2 = CausalSepConv(c, c, k, dilation)
        self.norm2 = ChannelNorm(c)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.drop(F.gelu(self.norm1(self.conv1(x))))
        h = self.drop(F.gelu(self.norm2(self.conv2(h))))
        return x + h


class Branch(nn.Module):
    def __init__(self, c_in: int, width: int, dilations, k: int, dropout: float):
        super().__init__()
        self.stem = nn.Conv1d(c_in, width, k)
        self.k = k
        self.dilations = tuple(dilations)
        self.blocks = nn.Sequential(*[TCNBlock(width, k, d, dropout) for d in dilations])

    def forward(self, x):  # x: (B, C, T)
        return self.blocks(self.stem(F.pad(x, (self.k - 1, 0))))

    @property
    def receptive_field(self) -> int:
        """In input steps (samples for raw input, frames for frame input)."""
        return 1 + (self.k - 1) + 2 * (self.k - 1) * sum(self.dilations)


class GradReverse(torch.autograd.Function):
    """Identity forward; gradient multiplied by -lambda backward (DANN)."""

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


# Two input representations, same network family.
RAW_CONFIG = dict(emg_in=8, imu_in=6, emg_width=32, imu_width=16, k=5,
                  emg_dilations=(1, 2, 4, 8, 16), imu_dilations=(1, 2, 4))
FRAME_CONFIG = dict(emg_in=52, imu_in=12, emg_width=48, imu_width=16, k=3,
                    emg_dilations=(1, 2, 4), imu_dilations=(1, 2))  # RF 31 frames ~1.5 s
# Riemannian frames (emg_gestures/riemann.py): 3 bands x 36 tangent values per frame.
RIEMANN_CONFIG = dict(FRAME_CONFIG, emg_in=108)
# Riemannian frames plus correlation-matrix vectors: 108 + 3 bands x 28 = 192.
RIEMANN_CORR_CONFIG = dict(FRAME_CONFIG, emg_in=192)


class AuraTCN(nn.Module):
    """Dual-branch causal TCN with masked attention pooling.

    ``n_domains`` > 0 adds a domain classifier behind a gradient-reversal layer
    (domain-adversarial training): the shared embedding is pushed to carry no
    information about the recording setup or session, only about the gesture.
    """

    def __init__(self, n_classes: int = 6, emg_in: int = 8, imu_in: int = 6,
                 emg_width: int = 32, imu_width: int = 16, k: int = 5,
                 emg_dilations=(1, 2, 4, 8, 16), imu_dilations=(1, 2, 4),
                 dropout: float = 0.15, use_imu: bool = True, n_domains: int = 0):
        super().__init__()
        self.use_imu = use_imu
        self.emg = Branch(emg_in, emg_width, emg_dilations, k, dropout)
        width = emg_width
        if use_imu:
            self.imu = Branch(imu_in, imu_width, imu_dilations, k, dropout)
            width += imu_width
        self.attn = nn.Conv1d(width, 1, 1)
        self.head = nn.Sequential(nn.Linear(2 * width, 32), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(32, n_classes))
        self.domain_head = None
        if n_domains > 0:
            self.domain_head = nn.Sequential(nn.Linear(2 * width, 32), nn.GELU(),
                                             nn.Linear(32, n_domains))

    def embed(self, emg, imu, mask):
        """emg (B, T, Ce), imu (B, T, Ci), mask (B, T) bool -> pooled (B, 2*width)."""
        h = self.emg(emg.transpose(1, 2))
        if self.use_imu:
            h = torch.cat([h, self.imu(imu.transpose(1, 2))], dim=1)
        score = self.attn(h).squeeze(1).masked_fill(~mask, float("-inf"))
        w = torch.softmax(score, dim=1).unsqueeze(1)                     # (B, 1, T)
        m = mask.unsqueeze(1).float()
        mean = (h * m).sum(2) / m.sum(2).clamp(min=1)
        return torch.cat([(h * w).sum(2), mean], dim=1)                  # attention + mean

    def forward(self, emg, imu, mask):
        """-> class logits (B, n_classes)."""
        return self.head(self.embed(emg, imu, mask))

    def forward_with_domain(self, emg, imu, mask, lam: float):
        z = self.embed(emg, imu, mask)
        return self.head(z), self.domain_head(GradReverse.apply(z, lam))

    @property
    def receptive_field(self) -> int:
        return self.emg.receptive_field


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
