import torch.nn as nn

from graft.models.core import GraFTBase, init_linear


class MLPPredictor(nn.Module):
    def __init__(self, seq_len, pred_len, in_channels, out_channels, hidden_dim, dropout):
        super().__init__()
        if min(seq_len, pred_len, in_channels, out_channels, hidden_dim) <= 0:
            raise ValueError("All MLP dims must be > 0.")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.in_channels = in_channels
        self.out_channels = out_channels

        in_dim = seq_len * in_channels
        out_dim = pred_len * out_channels

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.apply(init_linear)

    def forward(self, x):
        leading = x.shape[:-2]
        T, C = x.shape[-2], x.shape[-1]
        if T != self.seq_len or C != self.in_channels:
            raise ValueError(
                f"MLPPredictor: expected (T={self.seq_len}, C={self.in_channels}), "
                f"got (T={T}, C={C})."
            )
        x_flat = x.reshape(*leading, T * C)
        y_flat = self.net(x_flat)
        return y_flat.reshape(*leading, self.pred_len, self.out_channels)


class GraFT_MLP(GraFTBase):
    def build_predictor(self, in_channels, hidden_dim, dropout):
        return MLPPredictor(
            seq_len=self.seq_len,
            pred_len=self.horizon,
            in_channels=in_channels,
            out_channels=self.pred_latent,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
