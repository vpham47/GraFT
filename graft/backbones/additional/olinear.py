"""OLinear as a GraFT per-sensor predictor.

Re-implementation from the paper (the official code has no license, so none of it is used):
    W. Yue et al., "OLinear: A Linear Model for Time Series Forecasting in Orthogonally Transformed
    Domain", 2025. https://github.com/jackyue1994/OLinear

Two ideas from the paper:
  OrthoTrans  the input window is projected onto an orthogonal basis Q (eigenvectors of the temporal
              Pearson correlation matrix of the training series), which decorrelates time steps;
              the forecast is built in that domain and mapped back with the output basis.
  NormLin     a linear token mixer whose weights are positive and row-normalized (softplus, then L1),
              used here across the predictor's input channels.

``temporal_basis`` computes Q from training data; run.py passes it in. Without one (for example in
unit tests) an orthonormal DCT-II basis is used.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def dct_basis(length):
    """Orthonormal DCT-II basis as columns, shape [length, length]."""
    n = np.arange(length)
    basis = np.cos(np.pi / length * (n[:, None] + 0.5) * n[None, :])
    basis[:, 0] *= 1 / np.sqrt(2)
    return (basis * np.sqrt(2 / length)).astype(np.float32)


def temporal_basis(series, length, max_windows=20000, max_sensors=1000, seed=0):
    """Eigenvectors (columns, by decreasing eigenvalue) of the channel-averaged Pearson correlation
    matrix of length-``length`` windows of ``series`` [T, N] (use the training part only)."""
    series = np.asarray(series, dtype=np.float64)
    T, N = series.shape
    rng = np.random.default_rng(seed)
    starts = np.arange(T - length + 1)
    if len(starts) > max_windows:
        starts = rng.choice(starts, max_windows, replace=False)
    sensors = np.arange(N) if N <= max_sensors else rng.choice(N, max_sensors, replace=False)
    # Linear algebra in torch: NumPy's MKL and torch can load clashing OpenMP runtimes on Windows.
    windows = torch.as_tensor(series[starts[:, None] + np.arange(length)[None, :]][..., sensors])
    windows = windows.permute(2, 0, 1)                               # [sensors, windows, length]
    centered = windows - windows.mean(1, keepdim=True)
    cov = centered.transpose(1, 2) @ centered                        # [sensors, length, length]
    scale = torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2))
    valid = (scale > 1e-12).all(1)
    if not bool(valid.any()):
        return dct_basis(length)
    corr = cov[valid] / (scale[valid, :, None] * scale[valid, None, :])
    values, vectors = torch.linalg.eigh(corr.mean(0))
    return vectors[:, torch.argsort(values, descending=True)].numpy().astype(np.float32)


class NormLin(nn.Module):
    """Token mixing with positive, row-normalized weights, followed by a feed-forward layer."""

    def __init__(self, num_tokens, d_model, d_ff, dropout):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(num_tokens) + torch.randn(num_tokens, num_tokens))
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                      # x: [B, tokens, d_model]
        mix = F.normalize(F.softplus(self.weight), p=1, dim=-1)
        x = self.norm1(x + self.dropout(self.out(mix @ self.value(x))))
        return self.norm2(x + self.dropout(self.ffn(x)))


class OLinearPredictor(nn.Module):
    def __init__(self, seq_len, pred_len, in_channels, out_channels, hidden_dim, dropout,
                 basis_in=None, basis_out=None, embed_size=8, num_layers=1, eps=1e-5):
        super().__init__()
        self.seq_len, self.pred_len = seq_len, pred_len
        self.in_channels, self.out_channels = in_channels, out_channels
        self.embed_size, self.eps = embed_size, eps
        q_in = basis_in if basis_in is not None else dct_basis(seq_len)
        q_out = basis_out if basis_out is not None else dct_basis(pred_len)
        self.register_buffer('q_in', torch.as_tensor(q_in, dtype=torch.float32))     # [T, T]
        self.register_buffer('q_out', torch.as_tensor(q_out, dtype=torch.float32))   # [H, H]

        self.affine_weight = nn.Parameter(torch.ones(in_channels))
        self.affine_bias = nn.Parameter(torch.zeros(in_channels))
        self.embedding = nn.Parameter(torch.randn(1, embed_size))
        self.delta_in = nn.Parameter(torch.zeros(1, in_channels, 1, seq_len))
        self.delta_out = nn.Parameter(torch.zeros(1, in_channels, 1, pred_len))

        self.encode = nn.Linear(seq_len * embed_size, hidden_dim)
        self.mixers = nn.ModuleList([NormLin(in_channels, hidden_dim, 2 * hidden_dim, dropout)
                                     for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.decode = nn.Linear(hidden_dim, pred_len * embed_size)
        self.fc = nn.Sequential(nn.Linear(pred_len * embed_size, 2 * hidden_dim), nn.GELU(),
                                nn.Linear(2 * hidden_dim, pred_len))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        leading = x.shape[:-2]
        T, C = x.shape[-2], x.shape[-1]
        if T != self.seq_len or C != self.in_channels:
            raise ValueError(f'expected (T={self.seq_len}, C={self.in_channels}), got (T={T}, C={C})')
        x = x.reshape(-1, T, C)
        B = x.shape[0]

        # instance normalization with a learnable affine map (reversed at the end)
        mean = x.mean(1, keepdim=True)
        std = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + self.eps)
        x = (x - mean) / std * self.affine_weight + self.affine_bias

        tokens = x.transpose(1, 2).unsqueeze(-1) * self.embedding           # [B, C, T, D]
        z = tokens.transpose(-1, -2) @ self.q_in + self.delta_in            # [B, C, D, T] in the Q domain
        z = self.encode(z.flatten(-2))                                      # [B, C, hidden]
        for mixer in self.mixers:
            z = mixer(z)
        z = self.decode(self.norm(z)).reshape(B, C, self.embed_size, self.pred_len)
        z = z @ self.q_out.T + self.delta_out                               # back to the time domain
        y = self.dropout(self.fc(z.transpose(-1, -2).flatten(-2)))          # [B, C, H]
        y = y.transpose(1, 2)                                               # [B, H, C]

        y = (y - self.affine_bias) / (self.affine_weight + self.eps * self.eps) * std + mean
        return self.head(y).reshape(*leading, self.pred_len, self.out_channels)
