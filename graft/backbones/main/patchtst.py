import torch
import torch.nn as nn

from graft.models.core import GraFTBase


class PatchTSTPredictor(nn.Module):
    N_LAYERS = 2

    def __init__(self, seq_len, pred_len, in_channels, out_channels, hidden_dim, dropout,
                 patch_len=2, stride=1):
        super().__init__()
        if min(seq_len, pred_len, in_channels, out_channels, hidden_dim) <= 0:
            raise ValueError("All dims must be > 0.")
        self.PATCH_LEN = patch_len
        self.STRIDE = stride
        if seq_len < self.PATCH_LEN:
            raise ValueError(
                f"seq_len ({seq_len}) must be >= PATCH_LEN ({self.PATCH_LEN})."
            )

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d_model = hidden_dim
        self.num_patches = (seq_len - self.PATCH_LEN) // self.STRIDE + 1

        for h in (8, 4, 2, 1):
            if hidden_dim % h == 0:
                n_heads = h
                break

        self.patch_embed = nn.Linear(self.PATCH_LEN * in_channels, self.d_model)
        nn.init.normal_(self.patch_embed.weight, std=0.02)
        nn.init.zeros_(self.patch_embed.bias)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=2 * self.d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.N_LAYERS, enable_nested_tensor=False
        )

        self.head_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(self.num_patches * self.d_model, pred_len * out_channels)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        leading = x.shape[:-2]
        T, C = x.shape[-2], x.shape[-1]
        if T != self.seq_len or C != self.in_channels:
            raise ValueError(
                f"PatchTSTPredictor: expected (T={self.seq_len}, C={self.in_channels}), "
                f"got (T={T}, C={C})."
            )

        x = x.reshape(-1, T, C)

        patches = x.unfold(dimension=1, size=self.PATCH_LEN, step=self.STRIDE)
        patches = patches.permute(0, 1, 3, 2).contiguous() \
                         .reshape(-1, self.num_patches, self.PATCH_LEN * C)

        z = self.patch_embed(patches) + self.pos_embed
        chunk = 32768
        if z.shape[0] > chunk:
            z = torch.cat([self.transformer(c) for c in z.split(chunk, dim=0)], dim=0)
        else:
            z = self.transformer(z)

        z = z.reshape(z.shape[0], -1)
        z = self.head_dropout(z)
        y = self.head(z)

        return y.reshape(*leading, self.pred_len, self.out_channels)


class GraFT_PatchTST(GraFTBase):
    def build_predictor(self, in_channels, hidden_dim, dropout):
        return PatchTSTPredictor(
            seq_len=self.seq_len,
            pred_len=self.horizon,
            in_channels=in_channels,
            out_channels=self.pred_latent,
            hidden_dim=hidden_dim,
            dropout=dropout,
            patch_len=int(self.args.graft_patch_len),
            stride=int(self.args.graft_patch_stride),
        )
