"""Default run configuration. Every key can be overridden on the command line (``--key value``)."""

DEFAULTS = {
    # run
    'device': 'cuda:0', 'seed': 3028, 'mode': 'train', 'experiment_name': 'run', 'years': '2019',
    # training
    'bs': 64, 'output_dim': 1, 'max_epochs': 300, 'patience': 7,
    'lrate': 2e-2, 'wdecay': 1e-4, 'dropout': 0.1, 'clip_grad_value': 5,
    'warmup_epochs': 5, 'warmup_init_lr': 1e-6, 'final_lr': 1e-5, 'ema_decay': 0.0,
    # GraFT predictor and refinement stack
    'graft_num_layers': 3, 'graft_hidden': 64, 'graft_predictor_dropout': -1.0,
    'graft_pred_latent': 64, 'graft_value_channel': 0,
    'graft_graph_dG': 32, 'graft_graph_alpha_init': 0.0, 'graft_graph_dropout': 0.0,
    'graft_fixed_gate': 0, 'graft_post_graph_ffn': 0, 'graft_post_graph_ratio': 2,
    'graft_post_graph_dropout': 0.0, 'graft_patch_len': 2, 'graft_patch_stride': 1,
    # time and node embeddings
    'graft_tod_dim': 24, 'graft_dow_dim': 24, 'graft_node_dim': 0, 'graft_steps_per_day': 96,
    'graft_tod_channel': 1, 'graft_dow_channel': 2, 'graft_tod_is_fraction': 1,
}


def cast_like(key, value):
    """Cast a command-line string to the type of the default value for ``key``."""
    default = DEFAULTS[key]
    if isinstance(default, bool):
        return value.lower() in {'1', 'true', 'yes'}
    if isinstance(default, int):
        return int(float(value))
    if isinstance(default, float):
        return float(value)
    return value
