"""Convert the author's HDF5 and road distances without loading pickle objects.

Sources: https://github.com/liyaguang/DCRNN (data preparation and gen_adj_mx.py).
The output preserves HDF5 sensor ordering and explicitly inserts missing timestamps.
"""
import argparse
import csv
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from traffic_forecasting.multihorizon_data import atomic_json, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    with h5py.File(root / 'pems-bay.h5') as stream:
        ids = stream['speed/axis0'][:]
        times = stream['speed/axis1'][:]
        data = stream['speed/block0_values'][:].astype(np.float32)
        if not np.array_equal(ids, stream['speed/block0_items'][:]):
            raise ValueError('HDF5 column ordering is ambiguous')
    if len(set(ids)) != len(ids) or (np.diff(times) <= 0).any():
        raise ValueError('Duplicate sensors or unordered timestamps')
    # The source has naive local wall-clock timestamps and skips the spring DST hour.
    # This declared geographic timezone interpretation makes physical time regular.
    times = pd.DatetimeIndex(times).tz_localize(
        'America/Los_Angeles', ambiguous='raise', nonexistent='raise').tz_convert('UTC').asi8
    interval = 300_000_000_000
    if ((times - times[0]) % interval).any():
        raise ValueError('Timestamps are not on a five-minute grid')
    grid = np.arange(times[0], times[-1] + interval, interval, dtype=np.int64)
    values = np.full((len(grid), len(ids)), np.nan, dtype=np.float32)
    values[(times - times[0]) // interval] = data
    distances = np.full((len(ids), len(ids)), np.inf, dtype=np.float32)
    index = {str(sensor): i for i, sensor in enumerate(ids)}
    with (root / 'distances_bay_2017.csv').open() as stream:
        for origin, destination, distance in csv.reader(stream):
            if origin in index and destination in index:
                distances[index[origin], index[destination]] = float(distance)
    scale = distances[np.isfinite(distances)].std()
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Road distances could not be joined to sensor IDs')
    adj = np.exp(-np.square(distances / scale))
    adj[adj < .1] = 0
    for name, array in [('values', values), ('adj', adj), ('timestamps', grid), ('sensor_ids', ids)]:
        np.save(root / f'{name}.npy', array)
    atomic_json(root / 'source_manifest.json', {
        'source': 'https://github.com/liyaguang/DCRNN',
        'raw_sha256': {n: sha256(root / n) for n in ['pems-bay.h5', 'distances_bay_2017.csv']},
        'original_shape': list(data.shape), 'reindexed_shape': list(values.shape),
        'inserted_missing_timestamps': len(grid) - len(times),
        'graph': 'author distance Gaussian kernel, threshold 0.1; HDF5 column order',
        'timezone_interpretation': 'naive source is America/Los_Angeles; convert to UTC; spring DST gap is not an outage',
        'test_statistics_inspected': False})
    print('Prepared PEMS-BAY:', values.shape, 'inserted intervals:', len(grid) - len(times))


if __name__ == '__main__':
    main()
