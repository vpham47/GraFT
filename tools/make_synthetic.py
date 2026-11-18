"""Generate a small synthetic spatio-temporal dataset with known community structure.

Sensors fall into K communities. Each community shares a daily pattern plus a slowly drifting
latent signal; each sensor sees its community signal with its own scale and noise. Neighbors in
the same community are therefore informative, which is the setting where a graph should help.

    python tools/make_synthetic.py --out data/synthetic --nodes 24 --days 28
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graft.prepare import write_dataset  # noqa: E402


def make(nodes=24, communities=4, days=28, steps_per_day=24, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    T = days * steps_per_day
    t = np.arange(T)
    member = np.sort(rng.integers(0, communities, size=nodes))

    latent = np.zeros((T, communities))
    for k in range(communities):
        daily = np.sin(2 * np.pi * t / steps_per_day + rng.uniform(0, 2 * np.pi))
        drift = np.zeros(T)
        for i in range(1, T):
            drift[i] = 0.97 * drift[i - 1] + rng.normal(0, 0.3)
        latent[:, k] = daily + drift

    scale = rng.uniform(5, 15, size=nodes)
    values = 50 + latent[:, member] * scale[None, :] + rng.normal(0, noise * scale, size=(T, nodes))
    return values, member


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='data/synthetic')
    p.add_argument('--nodes', type=int, default=24)
    p.add_argument('--communities', type=int, default=4)
    p.add_argument('--days', type=int, default=28)
    p.add_argument('--steps_per_day', type=int, default=24)
    p.add_argument('--seq_len', type=int, default=12)
    p.add_argument('--horizon', type=int, default=12)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    values, member = make(a.nodes, a.communities, a.days, a.steps_per_day, seed=a.seed)
    meta = write_dataset(a.out, values, a.steps_per_day, seq_len=a.seq_len, horizon=a.horizon,
                         name='synthetic',
                         extra_meta={'communities': member.tolist()})
    print(f'Wrote {a.out}: T={meta["num_steps"]}, N={meta["num_nodes"]}')


if __name__ == '__main__':
    main()
