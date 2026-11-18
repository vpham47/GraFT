"""Hyperparameter presets for GraFT.

``paper``  the exact settings of the paper runs (configs/paper_configs.json, 3 backbones x 14 datasets).
``auto``   picks small / medium / large from the number of sensors N.
``small`` / ``medium`` / ``large``  force a size class.

The size presets are rules of thumb distilled from the paper settings, not tuned optima.
See the "Tuning by dataset size" section of the README for the reasoning behind each value.
"""
import json
import os

from graft.datasets import size_class

_PAPER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'configs', 'paper_configs.json')

# Settings shared by every paper run.
COMMON = {
    'wdecay': 1e-4, 'dropout': 0.1, 'clip_grad_value': 5, 'patience': 7, 'ema_decay': 0.999,
    'warmup_epochs': 5, 'warmup_init_lr': 1e-4, 'final_lr': 5e-5,
    'graft_graph_alpha_init': 0.0, 'graft_graph_dropout': 0.0,
    'graft_post_graph_ffn': 1, 'graft_post_graph_ratio': 2, 'graft_post_graph_dropout': 0.0,
    'graft_tod_dim': 24, 'graft_dow_dim': 24, 'graft_node_dim': 24, 'graft_tod_is_fraction': 1,
    'graft_value_channel': 0, 'graft_predictor_dropout': 0.0,
}

# Per-sensor predictor width (graft_hidden, graft_pred_latent): the paper's most common choice for
# the paper backbones, untuned defaults for the others (TimesNet's 2-D inception blocks are costly).
WIDTH = {'mlp': (512, 128), 'patchtst': (128, 64), 'tsmixer': (128, 64),
         'timesnet': (32, 64)}
DEFAULT_WIDTH = (64, 64)

# Size-dependent settings. L = refinement depth, dG = number of graph modes.
SIZE = {
    'small':  {'graft_num_layers': 3, 'graft_graph_dG': 64,  'bs': 16, 'lrate': 2e-3},
    'medium': {'graft_num_layers': 3, 'graft_graph_dG': 128, 'bs': 16, 'lrate': 1e-3},
    'large':  {'graft_num_layers': 1, 'graft_graph_dG': 128, 'bs': 16, 'lrate': 2e-3},
}

# Epochs: TSMixer converges much faster than the MLP and PatchTST predictors.
EPOCHS = {
    'mlp':      {'small': 60, 'medium': 60, 'large': 60},
    'patchtst': {'small': 60, 'medium': 60, 'large': 60},
    'tsmixer':  {'small': 30, 'medium': 30, 'large': 60},
}


def paper_configs():
    with open(_PAPER_PATH) as handle:
        return json.load(handle)


def paper_preset(backbone, dataset):
    configs = paper_configs()
    if backbone not in configs:
        raise ValueError(f'The paper did not use the {backbone} backbone; use --preset auto.')
    if dataset not in configs[backbone]:
        raise ValueError(f'No paper configuration for {backbone} on {dataset!r}. '
                         'Use --preset auto for datasets outside the paper.')
    cfg = dict(configs[backbone][dataset])
    cfg.pop('backbone', None)
    return {k: _number(v) for k, v in cfg.items()}


def size_preset(backbone, spec, size=None):
    """Preset for a dataset spec; ``size`` overrides the class inferred from N."""
    size = size or size_class(spec.num_nodes)
    cfg = dict(COMMON)
    cfg.update(SIZE[size])
    hidden, latent = WIDTH.get(backbone, DEFAULT_WIDTH)
    cfg.update({'graft_hidden': hidden, 'graft_pred_latent': latent,
                'max_epochs': EPOCHS.get(backbone, EPOCHS['patchtst'])[size]})
    low_frequency = spec.steps_per_day <= 24
    if low_frequency and backbone != 'tsmixer':
        cfg['max_epochs'] = 100      # hourly and coarser data: the paper used 100 epochs
    if low_frequency and backbone == 'mlp':
        cfg['graft_predictor_dropout'] = 0.1
    cfg.update({'graft_steps_per_day': spec.steps_per_day,
                'graft_tod_channel': spec.tod_channel, 'graft_dow_channel': spec.dow_channel})
    return cfg, size


def _number(value):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value
