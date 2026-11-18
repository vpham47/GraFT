import os
import threading
import multiprocessing as mp

import numpy as np
import torch


class DataLoader(object):
    def __init__(self, data, idx, seq_len, horizon, bs, logger, input_dim, pad_last_sample=True):
        if pad_last_sample:
            num_padding = (bs - (len(idx) % bs)) % bs
            idx_padding = np.repeat(idx[-1:], num_padding, axis=0)
            idx = np.concatenate([idx, idx_padding], axis=0)
        self.data = data
        self.idx = idx
        self.size = len(idx)
        self.bs = bs
        self.num_batch = int(self.size // self.bs)
        self.current_ind = 0
        logger.info('Sample num: ' + str(self.idx.shape[0]) + ', Batch num: ' + str(self.num_batch))

        self.x_offsets = np.arange(-(seq_len - 1), 1, 1)
        self.y_offsets = np.arange(1, (horizon + 1), 1)
        self.seq_len = seq_len
        self.horizon = horizon
        self.input_dim = input_dim

    def shuffle(self):
        perm = np.random.permutation(self.size)
        self.idx = self.idx[perm]

    def write_to_shared_array(self, x, y, idx_ind, start_idx, end_idx):
        for i in range(start_idx, end_idx):
            x[i] = self.data[idx_ind[i] + self.x_offsets, :, :self.input_dim]
            y[i] = self.data[idx_ind[i] + self.y_offsets, :, :1]

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.bs * self.current_ind
                end_ind = min(self.size, self.bs * (self.current_ind + 1))
                idx_ind = self.idx[start_ind:end_ind, ...]

                x_shape = (len(idx_ind), self.seq_len, self.data.shape[1],
                           min(self.data.shape[-1], self.input_dim))
                x_shared = mp.RawArray('f', int(np.prod(x_shape)))
                x = np.frombuffer(x_shared, dtype='f').reshape(x_shape)

                y_shape = (len(idx_ind), self.horizon, self.data.shape[1], 1)
                y_shared = mp.RawArray('f', int(np.prod(y_shape)))
                y = np.frombuffer(y_shared, dtype='f').reshape(y_shape)

                array_size = len(idx_ind)
                num_threads = len(idx_ind) if array_size == 1 else len(idx_ind) // 2
                chunk_size = array_size // num_threads
                threads = []
                for i in range(num_threads):
                    start_index = i * chunk_size
                    end_index = start_index + chunk_size if i < num_threads - 1 else array_size
                    thread = threading.Thread(
                        target=self.write_to_shared_array,
                        args=(x, y, idx_ind, start_index, end_index),
                    )
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

                yield (x, y)
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    def __init__(self, mean, std, input_dim, eps=1e-8):
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)

        if mean.ndim == 0:
            mean = np.full((input_dim,), float(mean), dtype=np.float32)
        if std.ndim == 0:
            std = np.full((input_dim,), float(std), dtype=np.float32)

        mean = mean.reshape(-1)
        std = np.maximum(std.reshape(-1), eps)

        self.mean = torch.tensor(mean.tolist(), dtype=torch.float32).view(1, 1, 1, -1)
        self.std = torch.tensor(std.tolist(), dtype=torch.float32).view(1, 1, 1, -1)

    def _to_device(self, ref_tensor):
        if self.mean.device != ref_tensor.device or self.mean.dtype != ref_tensor.dtype:
            self.mean = self.mean.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
            self.std = self.std.to(device=ref_tensor.device, dtype=ref_tensor.dtype)

    def transform(self, data):
        self._to_device(data)
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        self._to_device(data)
        return (data * self.std) + self.mean


def load_dataset(data_dir, args, logger):
    """Load ``his.npz`` and ``idx_{train,val,test}.npy`` from ``data_dir``.

    ``his.npz`` holds ``data`` [T, N, C] (channel 0 already standardized), ``mean`` and ``std``.
    """
    ptr = np.load(os.path.join(data_dir, 'his.npz'), allow_pickle=True)
    logger.info('Data shape: ' + str(ptr['data'].shape))

    dataloader = {}
    for cat in ['train', 'val', 'test']:
        idx = np.load(os.path.join(data_dir, 'idx_' + cat + '.npy'))
        dataloader[cat + '_loader'] = DataLoader(
            ptr['data'][..., :args.input_dim], idx,
            args.seq_len, args.horizon, args.bs,
            logger, args.input_dim,
        )

    scaler = StandardScaler(mean=ptr['mean'], std=ptr['std'], input_dim=args.input_dim)
    return dataloader, scaler
