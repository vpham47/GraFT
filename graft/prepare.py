"""Write a dataset folder in the GraFT format (see graft/datasets.py) from a [T, N] array."""
import json
import os

import numpy as np


def write_dataset(out_dir, values, steps_per_day, tod_index=None, dow_index=None,
                  seq_len=12, horizon=12, split=(0.6, 0.2, 0.2), name=None, extra_meta=None):
    """Save ``values`` [T, N] as his.npz + idx_*.npy + meta.json.

    tod_index: [T] integer step of the day (0..steps_per_day-1); defaults to t % steps_per_day.
    dow_index: [T] integer day of the week (0..6); defaults to (t // steps_per_day) % 7.
    Both are stored as fractions in [0, 1): time of day = step / steps_per_day, day of week = dow / 7.
    Missing values (NaN) are stored as 0, which the metrics treat as missing.
    Mean and std are computed on the training split only.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f'values must be [T, N], got shape {values.shape}')
    T, N = values.shape
    if abs(sum(split) - 1.0) > 1e-6:
        raise ValueError(f'split must sum to 1, got {split}')
    if T < seq_len + horizon + 10:
        raise ValueError(f'only {T} time steps; need well over seq_len + horizon = {seq_len + horizon}')

    t = np.arange(T)
    tod_index = t % steps_per_day if tod_index is None else np.asarray(tod_index)
    dow_index = (t // steps_per_day) % 7 if dow_index is None else np.asarray(dow_index)

    idx = np.arange(seq_len - 1, T - horizon)
    n_train = int(len(idx) * split[0])
    n_val = int(len(idx) * split[1])
    train_end = idx[n_train - 1] + 1           # last time step visible to training inputs

    values = np.nan_to_num(values, nan=0.0)
    observed = values[:train_end][values[:train_end] != 0]
    mean = float(observed.mean()) if observed.size else 0.0
    std = float(observed.std()) if observed.size else 1.0
    std = std if std > 0 else 1.0

    data = np.zeros((T, N, 3), dtype=np.float32)
    data[..., 0] = (values - mean) / std
    data[..., 1] = (tod_index / steps_per_day)[:, None]
    data[..., 2] = (dow_index / 7.0)[:, None]      # day of week as a fraction, like the paper data

    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, 'his.npz'), data=data, mean=np.float32(mean), std=np.float32(std))
    np.save(os.path.join(out_dir, 'idx_train.npy'), idx[:n_train])
    np.save(os.path.join(out_dir, 'idx_val.npy'), idx[n_train:n_train + n_val])
    np.save(os.path.join(out_dir, 'idx_test.npy'), idx[n_train + n_val:])

    meta = {'name': name or os.path.basename(os.path.normpath(out_dir)), 'num_nodes': N,
            'num_steps': T, 'steps_per_day': int(steps_per_day), 'seq_len': int(seq_len),
            'horizon': int(horizon), 'tod_channel': 1, 'dow_channel': 2,
            'value_mean': mean, 'value_std': std, 'split': list(split)}
    meta.update(extra_meta or {})
    with open(os.path.join(out_dir, 'meta.json'), 'w') as handle:
        json.dump(meta, handle, indent=1)
    return meta
