"""Use a Time-Series-Library forecaster as GraFT's per-sensor predictor.

GraFT feeds the predictor one sensor at a time: x is [B*N, T, C], where the C channels are the
sensor's own inputs plus its time-of-day, day-of-week and node embeddings. Each library model maps
[B*N, T, C] -> [B*N, horizon, C] unchanged; a linear head then maps the C output channels to the
predictor latent d that the refinement stack works on.

Settings follow the library's long-term-forecasting defaults, scaled to short inputs (T = 12 or 24).
"""
from types import SimpleNamespace

import torch
import torch.nn as nn

from graft.backbones.additional.tslib.models import (Autoformer, DLinear, SegRNN, TimeMixer, TimesNet,
                                          iTransformer)

TSLIB_MODELS = {
    'dlinear': DLinear.Model,
    'timemixer': TimeMixer.Model,
    'timesnet': TimesNet.Model,
    'itransformer': iTransformer.Model,
    'autoformer': Autoformer.Model,
    'segrnn': SegRNN.Model,
}


def _heads(d_model):
    return next(h for h in (8, 4, 2, 1) if d_model % h == 0)


def _seg_len(seq_len, pred_len):
    for s in range(seq_len // 2, 0, -1):
        if seq_len % s == 0 and pred_len % s == 0:
            return s
    return 1


def tslib_config(seq_len, pred_len, channels, d_model, dropout):
    return SimpleNamespace(
        task_name='long_term_forecast', seq_len=seq_len, label_len=seq_len // 2, pred_len=pred_len,
        enc_in=channels, dec_in=channels, c_out=channels, num_class=1,
        d_model=d_model, d_ff=2 * d_model, n_heads=_heads(d_model), e_layers=2, d_layers=1,
        dropout=dropout, activation='gelu', factor=3, distil=True, output_attention=False,
        embed='timeF', freq='h',
        moving_avg=25 if seq_len >= 96 else 5,          # decomposition kernel, shortened for short inputs
        top_k=min(5, seq_len // 2), num_kernels=6,        # TimesNet
        channel_independence=0, decomp_method='moving_avg', use_norm=1,       # TimeMixer
        down_sampling_layers=1 if seq_len < 48 else 3, down_sampling_window=2, down_sampling_method='avg',
        seg_len=_seg_len(seq_len, pred_len),              # SegRNN
    )


class TSLibPredictor(nn.Module):
    def __init__(self, name, seq_len, pred_len, in_channels, out_channels, hidden_dim, dropout):
        super().__init__()
        self.seq_len, self.pred_len = seq_len, pred_len
        self.in_channels, self.out_channels = in_channels, out_channels
        self.config = tslib_config(seq_len, pred_len, in_channels, hidden_dim, dropout)
        self.model = TSLIB_MODELS[name](self.config)
        self.head = nn.Linear(in_channels, out_channels)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        leading = x.shape[:-2]
        T, C = x.shape[-2], x.shape[-1]
        if T != self.seq_len or C != self.in_channels:
            raise ValueError(f'expected (T={self.seq_len}, C={self.in_channels}), got (T={T}, C={C})')
        x = x.reshape(-1, T, C)
        label = self.config.label_len
        x_dec = torch.cat([x[:, -label:], x.new_zeros(x.shape[0], self.pred_len, C)], dim=1)
        y = self.model(x, None, x_dec, None)[:, -self.pred_len:, :]
        return self.head(y).reshape(*leading, self.pred_len, self.out_channels)
