"""Train or evaluate GraFT on a paper benchmark or on your own data.

Examples
    python run.py --backbone patchtst --dataset sd                  # paper benchmark, paper settings
    python run.py --backbone mlp --data data/my_city                # your data, preset chosen from N
    python run.py --backbone tsmixer --data data/my_city --preset large
    python run.py --backbone timemixer --data data/my_city          # an additional backbone

Any setting in graft/config.py can be overridden, e.g. ``--graft_num_layers 2 --lrate 0.001``.
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from graft.backbones.additional.olinear import temporal_basis  # noqa: E402
from graft.config import DEFAULTS, cast_like  # noqa: E402
from graft.data import load_dataset  # noqa: E402
from graft.datasets import resolve  # noqa: E402
from graft.engine import BaseEngine  # noqa: E402
from graft.log import get_logger  # noqa: E402
from graft.metrics import masked_mae  # noqa: E402
from graft.presets import paper_preset, size_preset  # noqa: E402
from graft.registry import GRAFT_MODELS, MAIN_BACKBONES  # noqa: E402


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--backbone', default='patchtst', choices=sorted(GRAFT_MODELS),
                   help=f'per-sensor predictor; paper backbones: {", ".join(MAIN_BACKBONES)}')
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--dataset', help='paper benchmark name, e.g. sd, pems08, bay')
    src.add_argument('--data', help='folder in the GraFT format (see tools/csv_to_graft.py)')
    p.add_argument('--data_root', default='.', help='root folder of the paper datasets')
    p.add_argument('--preset', default=None, choices=['paper', 'auto', 'small', 'medium', 'large'],
                   help='hyperparameters (default: paper for a paper backbone on --dataset, else auto)')
    p.add_argument('--output_dir', default='experiments')
    args, rest = p.parse_known_args(argv)
    return args, _parse_overrides(rest, p)


def _parse_overrides(rest, parser):
    overrides = {}
    i = 0
    while i < len(rest):
        token = rest[i]
        if not token.startswith('--') or i + 1 >= len(rest):
            parser.error(f'unexpected argument {token!r}')
        key = token[2:].replace('-', '_')
        if key not in DEFAULTS:
            parser.error(f'unknown setting --{key}. Valid settings are listed in graft/config.py')
        overrides[key] = cast_like(key, rest[i + 1])
        i += 2
    return overrides


def build_config(cli, overrides, spec):
    cfg = dict(DEFAULTS)
    preset = cli.preset or ('paper' if cli.dataset and cli.backbone in MAIN_BACKBONES else 'auto')
    if preset == 'paper':
        if not cli.dataset:
            raise SystemExit('--preset paper is only defined for the paper datasets; use auto.')
        if cli.backbone not in MAIN_BACKBONES:
            raise SystemExit(f'The paper did not use the {cli.backbone} backbone; use --preset auto.')
        cfg.update(paper_preset(cli.backbone, cli.dataset))
        note = f'paper preset ({cli.backbone}, {cli.dataset})'
    else:
        values, size = size_preset(cli.backbone, spec, None if preset == 'auto' else preset)
        cfg.update(values)
        note = f'{size} preset (N={spec.num_nodes})'
    cfg.update(overrides)
    cfg.update({'dataset': spec.name, 'seq_len': spec.seq_len, 'horizon': spec.horizon,
                'input_dim': spec.input_dim, 'node_num': spec.num_nodes, 'backbone': cli.backbone})
    return SimpleNamespace(**cfg), note


def olinear_basis(spec):
    """OLinear's input and output bases from the training split of the target channel."""
    data = np.load(os.path.join(spec.data_dir, 'his.npz'), mmap_mode='r')['data']
    last = int(np.load(os.path.join(spec.data_dir, 'idx_train.npy')).max())
    inputs = np.asarray(data[:last + 1, :, 0])
    targets = np.asarray(data[:last + spec.horizon + 1, :, 0])
    return temporal_basis(inputs, spec.seq_len), temporal_basis(targets, spec.horizon)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_scheduler(optimizer, args):
    """Linear warmup from warmup_init_lr to lrate, then linear decay to final_lr (paper schedule)."""
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            start = args.warmup_init_lr / args.lrate
            return start + (1.0 - start) * (epoch / args.warmup_epochs)
        progress = (epoch - args.warmup_epochs) / max(1, args.max_epochs - args.warmup_epochs)
        return 1.0 - (1.0 - args.final_lr / args.lrate) * progress
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def main(argv=None):
    cli, overrides = parse(argv)
    spec = resolve(dataset=cli.dataset, data=cli.data, data_root=cli.data_root,
                   years=overrides.get('years', DEFAULTS['years']))
    args, note = build_config(cli, overrides, spec)

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    log_dir = os.path.join(cli.output_dir, spec.name, f'graft_{cli.backbone}', args.experiment_name)
    logger = get_logger(log_dir, __name__, f'record_s{args.seed}.log')
    logger.info('GraFT (%s) | data: %s (N=%d, %s) | %s', cli.backbone, spec.data_dir, spec.num_nodes,
                spec.size, note)
    logger.info(vars(args))

    dataloader, scaler = load_dataset(spec.data_dir, args, logger)
    scaler.mean = scaler.mean[..., :1]
    scaler.std = scaler.std[..., :1]

    if cli.backbone == 'olinear':
        args._olinear_basis = olinear_basis(spec)
        logger.info('OLinear: orthogonal bases computed from the training split')
    model = GRAFT_MODELS[cli.backbone](node_num=spec.num_nodes, input_dim=spec.input_dim,
                                       output_dim=args.output_dim, seq_len=spec.seq_len,
                                       horizon=spec.horizon, model_args=args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    engine = BaseEngine(device=device, model=model, dataloader=dataloader, scaler=scaler,
                        loss_fn=masked_mae, lrate=args.lrate, optimizer=optimizer,
                        scheduler=build_scheduler(optimizer, args),
                        clip_grad_value=args.clip_grad_value, max_epochs=args.max_epochs,
                        patience=args.patience, log_dir=log_dir, logger=logger, seed=args.seed,
                        args=args)

    if args.mode == 'train':
        mae, mape, rmse = engine.train()
    else:
        mae, mape, rmse = engine.evaluate('test', load_best=True, verbose=True)

    gates = model.get_gate_magnitudes()
    logger.info('Learned gate |alpha| per layer: %s', [f'{a:.4f}' for a in gates['per_layer']])
    logger.info('Mean gate |alpha|: %.4f', gates['mean'])
    result = {'model': f'graft_{cli.backbone}', 'dataset': spec.name, 'mae': mae, 'rmse': rmse,
              'mape': mape, 'seed': args.seed, 'params': model.param_num(), 'gates': gates['per_layer']}
    with open(os.path.join(log_dir, f'result_s{args.seed}.json'), 'w') as handle:
        json.dump(result, handle, indent=1)
    logger.info('RESULT %s', json.dumps(result))
    return result


if __name__ == '__main__':
    main()
