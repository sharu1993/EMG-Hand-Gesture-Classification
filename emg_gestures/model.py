"""Dual-branch causal TCN for Aura gesture windows (PyTorch).

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
        self.blocks = nn.Sequential(*[TCNBlock(width, k, d, dropout) for d in dilations])

    def forward(self, x):  # x: (B, C, T)
        return self.blocks(self.stem(F.pad(x, (self.k - 1, 0))))


class AuraTCN(nn.Module):
    def __init__(self, n_classes: int = 6, emg_width: int = 32, imu_width: int = 16,
                 k: int = 5, dropout: float = 0.15, use_imu: bool = True):
        super().__init__()
        self.use_imu = use_imu
        self.emg = Branch(8, emg_width, (1, 2, 4, 8, 16), k, dropout)
        width = emg_width
        if use_imu:
            self.imu = Branch(6, imu_width, (1, 2, 4), k, dropout)
            width += imu_width
        self.attn = nn.Conv1d(width, 1, 1)
        self.head = nn.Sequential(nn.Linear(2 * width, 32), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(32, n_classes))

    def forward(self, emg, imu, mask):
        """emg (B, T, 8), imu (B, T, 6), mask (B, T) bool -> logits (B, n_classes)."""
        h = self.emg(emg.transpose(1, 2))
        if self.use_imu:
            h = torch.cat([h, self.imu(imu.transpose(1, 2))], dim=1)
        score = self.attn(h).squeeze(1).masked_fill(~mask, float("-inf"))
        w = torch.softmax(score, dim=1).unsqueeze(1)                     # (B, 1, T)
        m = mask.unsqueeze(1).float()
        mean = (h * m).sum(2) / m.sum(2).clamp(min=1)
        pooled = torch.cat([(h * w).sum(2), mean], dim=1)                # attention + mean
        return self.head(pooled)

    @property
    def receptive_field(self) -> int:
        k = self.emg.k
        return 1 + (k - 1) + 2 * (k - 1) * sum((1, 2, 4, 8, 16))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
