import torch.nn as nn

from graft.models.core import GraFTBase, init_linear


class TSMixerPredictor(nn.Module):
    N_BLOCKS = 2

    def __init__(self, seq_len, pred_len, in_channels, out_channels, hidden_dim, dropout):
        super().__init__()
        if min(seq_len, pred_len, in_channels, out_channels, hidden_dim) <= 0:
            raise ValueError("All dims must be > 0.")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.time_norms = nn.ModuleList()
        self.time_mlps = nn.ModuleList()
        self.feat_norms = nn.ModuleList()
        self.feat_mlps = nn.ModuleList()

        for _ in range(self.N_BLOCKS):
            self.time_norms.append(nn.LayerNorm(in_channels))
            self.time_mlps.append(nn.Sequential(
                nn.Linear(seq_len, seq_len),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            ))
            self.feat_norms.append(nn.LayerNorm(in_channels))
            self.feat_mlps.append(nn.Sequential(
                nn.Linear(in_channels, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, in_channels),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            ))

        self.temporal_proj = nn.Linear(seq_len, pred_len)
        self.channel_proj = nn.Linear(in_channels, out_channels)

        self.apply(init_linear)

    def forward(self, x):
        leading = x.shape[:-2]
        T, C = x.shape[-2], x.shape[-1]
        if T != self.seq_len or C != self.in_channels:
            raise ValueError(
                f"TSMixerPredictor: expected (T={self.seq_len}, C={self.in_channels}), "
                f"got (T={T}, C={C})."
            )

        h = x.reshape(-1, T, C)

        for i in range(self.N_BLOCKS):
            z = self.time_norms[i](h)
            z = z.transpose(1, 2)
            z = self.time_mlps[i](z)
            z = z.transpose(1, 2)
            h = h + z

            z = self.feat_norms[i](h)
            z = self.feat_mlps[i](z)
            h = h + z

        h = self.temporal_proj(h.transpose(1, 2)).transpose(1, 2)
        y = self.channel_proj(h)

        return y.reshape(*leading, self.pred_len, self.out_channels)


class GraFT_TSMixer(GraFTBase):
    def build_predictor(self, in_channels, hidden_dim, dropout):
        return TSMixerPredictor(
            seq_len=self.seq_len,
            pred_len=self.horizon,
            in_channels=in_channels,
            out_channels=self.pred_latent,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
