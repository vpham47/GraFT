"""Backbone name -> GraFT model class."""
from graft.backbones.additional import ADDITIONAL_BACKBONES, build_predictor
from graft.backbones.main.mlp import GraFT_MLP
from graft.backbones.main.patchtst import GraFT_PatchTST
from graft.backbones.main.tsmixer import GraFT_TSMixer
from graft.models.core import GraFTBase

MAIN_BACKBONES = ['mlp', 'patchtst', 'tsmixer']      # evaluated in the paper


def _graft_class(name):
    class GraFTWithBackbone(GraFTBase):
        backbone_name = name

        def build_predictor(self, in_channels, hidden_dim, dropout):
            return build_predictor(name, self.seq_len, self.horizon, in_channels, self.pred_latent,
                                   hidden_dim, dropout, self.args)

    GraFTWithBackbone.__name__ = f'GraFT_{name}'
    return GraFTWithBackbone


GRAFT_MODELS = {'mlp': GraFT_MLP, 'patchtst': GraFT_PatchTST, 'tsmixer': GraFT_TSMixer}
GRAFT_MODELS.update({name: _graft_class(name) for name in ADDITIONAL_BACKBONES})
