"""Additional backbones: forecasters anyone can plug into GraFT.

They were not evaluated in the paper and use untuned defaults.
  dlinear, timemixer, timesnet, itransformer, autoformer, segrnn   from Time-Series-Library (MIT)
  olinear                                                          our re-implementation from its paper
"""
from graft.backbones.additional.olinear import OLinearPredictor
from graft.backbones.additional.tslib_predictor import TSLIB_MODELS, TSLibPredictor

ADDITIONAL_BACKBONES = sorted(list(TSLIB_MODELS) + ['olinear'])


def build_predictor(name, seq_len, pred_len, in_channels, out_channels, hidden_dim, dropout, args):
    if name == 'olinear':
        basis = getattr(args, '_olinear_basis', None) or (None, None)
        return OLinearPredictor(seq_len, pred_len, in_channels, out_channels, hidden_dim, dropout,
                                basis_in=basis[0], basis_out=basis[1])
    return TSLibPredictor(name, seq_len, pred_len, in_channels, out_channels, hidden_dim, dropout)
