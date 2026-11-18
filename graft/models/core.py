import torch
import torch.nn as nn
import torch.nn.functional as F

from graft.models.base import BaseModel


def init_linear(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


class LinearAdaptiveGraphBlock(nn.Module):
    def __init__(self, node_num, channel_dim, dG=32, dropout=0.0, fixed_gate=False):
        super().__init__()
        self.node_num = node_num
        self.channel_dim = channel_dim
        self.dG = dG
        self.E1 = nn.Parameter(torch.empty(node_num, dG))
        self.E2 = nn.Parameter(torch.empty(node_num, dG))
        nn.init.xavier_uniform_(self.E1)
        nn.init.xavier_uniform_(self.E2)
        if fixed_gate:
            self.register_buffer("alpha", torch.ones(1))
        else:
            self.alpha = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(channel_dim)
        self.dropout = dropout

    def forward(self, y):
        E1 = self.E1
        E2 = self.E2
        if self.training and self.dropout > 0:
            p = 0.1 + (self.dropout - 0.1) * float(torch.rand(1).item()) \
                if self.dropout > 0.1 else self.dropout
            E1 = F.dropout(E1, p=p, training=True)
            E2 = F.dropout(E2, p=p, training=True)
        E1_sm = F.softmax(E1, dim=-1)
        E2_sm = F.softmax(E2, dim=-1)
        z = torch.einsum("nd,bhnf->bhdf", E2_sm, y)
        z = torch.einsum("nd,bhdf->bhnf", E1_sm, z)
        return y + self.alpha * self.norm(z)


class GraFTBase(BaseModel):
    def __init__(self, model_args, **kwargs):
        super().__init__(**kwargs)
        self.args = model_args

        self.output_start = int(model_args.graft_value_channel)
        if self.output_start < 0 or self.output_start >= self.input_dim:
            raise ValueError(
                f"graft_value_channel={self.output_start} out of range "
                f"for input_dim={self.input_dim}."
            )
        if self.output_start + self.output_dim > self.input_dim:
            raise ValueError(
                "Output channel slice exceeds input_dim: "
                f"start={self.output_start}, output_dim={self.output_dim}, "
                f"input_dim={self.input_dim}."
            )

        self.tod_dim = int(model_args.graft_tod_dim)
        self.dow_dim = int(model_args.graft_dow_dim)
        self.node_dim = int(model_args.graft_node_dim)

        self.steps_per_day = int(model_args.graft_steps_per_day)
        self.tod_channel = int(model_args.graft_tod_channel)
        self.dow_channel = int(model_args.graft_dow_channel)
        self.tod_is_fraction = bool(int(model_args.graft_tod_is_fraction))

        if self.tod_dim > 0:
            self.tod_embedding = nn.Embedding(self.steps_per_day, self.tod_dim)
            nn.init.normal_(self.tod_embedding.weight, std=0.02)
        if self.dow_dim > 0:
            self.dow_embedding = nn.Embedding(7, self.dow_dim)
            nn.init.normal_(self.dow_embedding.weight, std=0.02)
        if self.node_dim > 0:
            self.node_embedding = nn.Parameter(
                0.02 * torch.randn(self.node_num, self.node_dim)
            )

        self.embed_dim = self.tod_dim + self.dow_dim + self.node_dim
        if self.embed_dim > 0:
            self.embed_norm = nn.LayerNorm(self.embed_dim)

        self.pred_latent = int(model_args.graft_pred_latent)
        hidden_dim = int(model_args.graft_hidden)
        dropout = float(model_args.graft_predictor_dropout)
        if dropout < 0:
            dropout = float(model_args.dropout)

        self.predictor = self.build_predictor(
            in_channels=self.input_dim + self.embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.num_spatial_layers = int(model_args.graft_num_layers)

        dG = int(model_args.graft_graph_dG)
        graph_dropout = float(model_args.graft_graph_dropout)
        alpha_init = float(model_args.graft_graph_alpha_init)
        fixed_gate = bool(int(model_args.graft_fixed_gate))

        self.use_post_graph_ffn = bool(int(model_args.graft_post_graph_ffn))
        ratio = int(model_args.graft_post_graph_ratio)
        pg_dropout = float(model_args.graft_post_graph_dropout)

        self.graph_blocks = nn.ModuleList()
        self.post_graph_norms = nn.ModuleList()
        self.post_graph_ffns = nn.ModuleList()

        for _ in range(self.num_spatial_layers):
            block = LinearAdaptiveGraphBlock(
                node_num=self.node_num,
                channel_dim=self.pred_latent,
                dG=dG,
                dropout=graph_dropout,
                fixed_gate=fixed_gate,
            )
            if not fixed_gate:
                with torch.no_grad():
                    block.alpha.fill_(alpha_init)
            self.graph_blocks.append(block)

            if self.use_post_graph_ffn:
                self.post_graph_norms.append(nn.LayerNorm(self.pred_latent))
                ffn = nn.Sequential(
                    nn.Linear(self.pred_latent, ratio * self.pred_latent),
                    nn.GELU(),
                    nn.Dropout(pg_dropout) if pg_dropout > 0 else nn.Identity(),
                    nn.Linear(ratio * self.pred_latent, self.pred_latent),
                )
                ffn.apply(init_linear)
                self.post_graph_ffns.append(ffn)

        self.out_proj = nn.Linear(self.pred_latent, self.output_dim)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

    def build_predictor(self, in_channels, hidden_dim, dropout):
        raise NotImplementedError

    def _build_embeddings(self, x):
        if self.embed_dim == 0:
            return None
        B, T, N, _ = x.shape
        parts = []
        if self.tod_dim > 0:
            tod_raw = x[..., self.tod_channel]
            if self.tod_is_fraction:
                tod_idx = (tod_raw * self.steps_per_day).long()
            else:
                tod_idx = tod_raw.long()
            tod_idx = tod_idx.clamp(0, self.steps_per_day - 1)
            parts.append(self.tod_embedding(tod_idx))
        if self.dow_dim > 0:
            dow_idx = x[..., self.dow_channel].long().clamp(0, 6)
            parts.append(self.dow_embedding(dow_idx))
        if self.node_dim > 0:
            node = self.node_embedding.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
            parts.append(node)
        emb = torch.cat(parts, dim=-1)
        return self.embed_norm(emb)

    @torch.no_grad()
    def get_gate_magnitudes(self):
        if len(self.graph_blocks) == 0:
            return {"per_layer": [], "mean": 0.0}
        alphas = [float(block.alpha.abs().item()) for block in self.graph_blocks]
        return {
            "per_layer": alphas,
            "mean": sum(alphas) / len(alphas),
        }

    def forward(self, x, label=None):
        if x.dim() != 4:
            raise ValueError(f"Expected x of shape [B,T,N,C], got {tuple(x.shape)}")
        B, T, N, C = x.shape

        emb = self._build_embeddings(x)
        if emb is not None:
            h = torch.cat([x, emb], dim=-1)
        else:
            h = x

        Ce = h.shape[-1]
        h_perm = h.permute(0, 2, 1, 3).reshape(B * N, T, Ce)
        y_lat = self.predictor(h_perm)
        y_lat = y_lat.reshape(B, N, self.horizon, self.pred_latent).permute(0, 2, 1, 3)

        for i in range(self.num_spatial_layers):
            y_lat = self.graph_blocks[i](y_lat)
            if self.use_post_graph_ffn:
                y_lat = y_lat + self.post_graph_ffns[i](self.post_graph_norms[i](y_lat))

        y = self.out_proj(y_lat)

        start = self.output_start
        end = start + self.output_dim
        last_value = x[:, -1:, :, start:end]
        return y + last_value
