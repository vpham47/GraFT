"""Convert a CSV of sensor readings into a GraFT dataset folder.

CSV layout: one row per time step, one column per sensor. An optional first column with
timestamps is used to derive time-of-day and day-of-week features.

    timestamp,sensor_a,sensor_b,...
    2024-01-01 00:00,12.0,7.5,...

Examples
    python tools/csv_to_graft.py --csv my_city.csv --out data/my_city
    python tools/csv_to_graft.py --csv readings.csv --out data/x --no_timestamp --steps_per_day 288

No adjacency matrix is needed: GraFT learns which sensors to share information with.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graft.prepare import write_dataset  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--no_timestamp', action='store_true', help='the CSV has no timestamp column')
    p.add_argument('--steps_per_day', type=int, default=None,
                   help='required with --no_timestamp; otherwise inferred from the timestamps')
    p.add_argument('--seq_len', type=int, default=12, help='input window length')
    p.add_argument('--horizon', type=int, default=12, help='forecast length')
    p.add_argument('--split', type=float, nargs=3, default=[0.6, 0.2, 0.2], metavar=('TRAIN', 'VAL', 'TEST'))
    a = p.parse_args()

    import pandas as pd
    frame = pd.read_csv(a.csv)
    tod_index = dow_index = None
    if a.no_timestamp:
        if not a.steps_per_day:
            p.error('--steps_per_day is required with --no_timestamp')
        steps_per_day = a.steps_per_day
    else:
        stamps = pd.to_datetime(frame.iloc[:, 0])
        frame = frame.iloc[:, 1:]
        step = stamps.diff().median()
        if pd.isna(step) or step.total_seconds() <= 0:
            p.error('could not infer the sampling interval from the timestamp column')
        steps_per_day = a.steps_per_day or int(round(86400 / step.total_seconds()))
        seconds = stamps.dt.hour * 3600 + stamps.dt.minute * 60 + stamps.dt.second
        tod_index = np.floor(seconds.to_numpy() / 86400 * steps_per_day).astype(int) % steps_per_day
        dow_index = stamps.dt.dayofweek.to_numpy()

    values = frame.apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)
    meta = write_dataset(a.out, values, steps_per_day, tod_index, dow_index, a.seq_len, a.horizon,
                         tuple(a.split), extra_meta={'sensors': list(map(str, frame.columns))})
    missing = float(np.isnan(values).mean() * 100)
    print(f'Wrote {a.out}: T={meta["num_steps"]}, N={meta["num_nodes"]}, steps/day={steps_per_day}, '
          f'missing={missing:.1f}%')


if __name__ == '__main__':
    main()
