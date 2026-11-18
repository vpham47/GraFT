"""Dataset registry and on-disk format.

A dataset is a folder containing

    his.npz          data [T, N, C] float32, mean, std. Channel 0 = target (standardized);
                     time of day and day of week are fractions in [0, 1) (step/steps_per_day, dow/7)
    idx_train.npy    window end indices for training   (int array)
    idx_val.npy      window end indices for validation
    idx_test.npy     window end indices for testing
    meta.json        num_nodes, steps_per_day, seq_len, horizon, tod_channel, dow_channel

GraFT learns its own graph, so no adjacency matrix is needed.

``tools/csv_to_graft.py`` builds this folder from a CSV file. The 14 benchmark datasets of the
paper keep their original folder layout and are listed in ``PAPER_DATASETS``.
"""
import json
import os
from dataclasses import dataclass, field

# Size classes used by the hyperparameter presets (number of sensors N).
SIZE_BOUNDS = {'small': (0, 500), 'medium': (500, 2500), 'large': (2500, float('inf'))}


def size_class(num_nodes):
    for name, (lo, hi) in SIZE_BOUNDS.items():
        if lo <= num_nodes < hi:
            return name
    raise ValueError(num_nodes)


@dataclass
class DatasetSpec:
    name: str
    data_dir: str
    num_nodes: int
    input_dim: int
    steps_per_day: int
    seq_len: int
    horizon: int
    tod_channel: int = 1
    dow_channel: int = 2
    extra: dict = field(default_factory=dict)

    @property
    def size(self):
        return size_class(self.num_nodes)


# name: (base folder, sub folder, N, channels, steps/day, seq_len, horizon, tod channel, dow channel)
# Paths are relative to --data_root (default: current directory), matching the original release.
PAPER_DATASETS = {
    'pems03':  ('data_PeMS0X', 'pems03',  358,  3, 288, 12, 12, 1, 2),
    'pems04':  ('data_PeMS0X', 'pems04',  307,  5, 288, 12, 12, 1, 2),
    'pems07':  ('data_PeMS0X', 'pems07',  883,  3, 288, 12, 12, 1, 2),
    'pems08':  ('data_PeMS0X', 'pems08',  170,  5, 288, 12, 12, 1, 2),
    'bay':     ('data_METR-LA', 'bay',    325,  3, 288, 12, 12, 1, 2),
    'sd':      ('data', 'sd',             716,  3,  96, 12, 12, 1, 2),
    'gba':     ('data', 'gba',           2352,  3,  96, 12, 12, 1, 2),
    'gla':     ('data', 'gla',           3834,  3,  96, 12, 12, 1, 2),
    'ca':      ('data', 'ca',            8600,  3,  96, 12, 12, 1, 2),
    'urbanev': ('data', 'urbanev',        275,  3, 288, 24, 24, 1, 2),
    'ele':     ('data_ele', 'ele',        321,  3,  24, 24, 24, 1, 2),
    'sh':      ('data_sh', 'sh',         3042,  3,  24, 24, 24, 1, 2),
    'knowair': ('data_knowair', '24_24',  184, 13,   8, 24, 24, 11, 12),
    'ccaq':    ('datagagnn', '24_24',     209, 10,  24, 24, 24, 8, 9),
}

DATASET_LABELS = {
    'pems03': 'PEMS03', 'pems04': 'PEMS04', 'pems07': 'PEMS07', 'pems08': 'PEMS08',
    'bay': 'PEMS-Bay', 'sd': 'SD (LargeST)', 'gba': 'GBA (LargeST)', 'gla': 'GLA (LargeST)',
    'ca': 'CA (LargeST)', 'urbanev': 'UrbanEV', 'ele': 'Electricity', 'sh': 'Shanghai Mobile',
    'knowair': 'KnowAir', 'ccaq': 'China City Air Quality',
}


def paper_dataset(name, data_root='.', years='2019'):
    base, sub, n, c, spd, seq, hor, tod, dow = PAPER_DATASETS[name]
    data_dir = os.path.join(data_root, base, sub, years) if years else os.path.join(data_root, base, sub)
    return DatasetSpec(name=name, data_dir=data_dir, num_nodes=n, input_dim=c, steps_per_day=spd,
                       seq_len=seq, horizon=hor, tod_channel=tod, dow_channel=dow)


def custom_dataset(data_dir):
    meta_path = os.path.join(data_dir, 'meta.json')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f'{meta_path} not found. Build the folder with tools/csv_to_graft.py, '
            'or see the "Bring your own data" section of the README.')
    with open(meta_path) as handle:
        meta = json.load(handle)
    import numpy as np
    shape = np.load(os.path.join(data_dir, 'his.npz'), mmap_mode='r')['data'].shape
    if shape[1] != meta['num_nodes']:
        raise ValueError(f'meta.json says num_nodes={meta["num_nodes"]} but his.npz has N={shape[1]}')
    return DatasetSpec(
        name=meta.get('name', os.path.basename(os.path.normpath(data_dir))),
        data_dir=data_dir, num_nodes=int(meta['num_nodes']), input_dim=int(shape[2]),
        steps_per_day=int(meta['steps_per_day']), seq_len=int(meta['seq_len']),
        horizon=int(meta['horizon']), tod_channel=int(meta.get('tod_channel', 1)),
        dow_channel=int(meta.get('dow_channel', 2)),
        extra={k: v for k, v in meta.items() if k not in {
            'num_nodes', 'steps_per_day', 'seq_len', 'horizon', 'tod_channel', 'dow_channel'}},
    )


def resolve(dataset=None, data=None, data_root='.', years='2019'):
    if (dataset is None) == (data is None):
        raise ValueError('Pass exactly one of --dataset (a paper benchmark) or --data (a folder).')
    if dataset is not None:
        if dataset not in PAPER_DATASETS:
            raise ValueError(f'Unknown dataset {dataset!r}. Paper datasets: {sorted(PAPER_DATASETS)}. '
                             'For your own data use --data <folder>.')
        return paper_dataset(dataset, data_root, years)
    return custom_dataset(data)
