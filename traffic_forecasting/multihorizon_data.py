"""Explicit masks and chronological partitions for 12-step speed forecasting."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class MaskedWindows(Dataset):
    def __init__(self, values, begin, end, mean, std, history=12, horizon=12, stride=1):
        if not 0 <= begin < end <= len(values) or std <= 0 or stride < 1:
            raise ValueError('Invalid partition, normalization, or stride')
        self.values, self.mean, self.std = values, mean, std
        self.history, self.horizon = history, horizon
        self.starts = np.arange(begin, end - history - horizon + 1, stride)
        if not len(self.starts):
            raise ValueError('Partition cannot contain a full window')

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = int(self.starts[index])
        raw = self.values[start:start + self.history + self.horizon]
        mask = np.isfinite(raw) & (raw > 0)
        normalized = np.where(mask, (raw - self.mean) / self.std, 0).astype(np.float32)
        return {'X': torch.from_numpy(normalized[:self.history, :, None].copy()),
                'X_mask': torch.from_numpy(mask[:self.history, :, None].copy()),
                'y': torch.from_numpy(normalized[self.history:, :, None].copy()),
                'y_mask': torch.from_numpy(mask[self.history:, :, None].copy()),
                'start': start}


def load_development(data_dir, dataset):
    """Load arrays; calculate summary statistics on training observations only."""
    root = Path(data_dir) / dataset
    if dataset == 'METR_LA':
        values = np.load(root / 'METR_LA.npz')['data'].astype(np.float32)
        graph = np.load(root / 'METR_LA_rn_adj.npy').astype(np.float32)
        paths = [root / 'METR_LA.npz', root / 'METR_LA_rn_adj.npy']
        timestamps = None
    else:
        values = np.load(root / 'values.npy', mmap_mode='r')
        graph = np.load(root / 'adj.npy')
        timestamps = np.load(root / 'timestamps.npy', mmap_mode='r')
        paths = [root / n for n in ['values.npy', 'adj.npy', 'timestamps.npy', 'sensor_ids.npy']]
    if values.ndim != 2 or graph.shape != (values.shape[1], values.shape[1]):
        raise ValueError('Speed series and graph dimensions disagree')
    if not np.isfinite(graph).all() or (graph < 0).any():
        raise ValueError('Graph must have finite nonnegative weights')
    # Gaps may not be silently interpreted as consecutive five-minute observations.
    if timestamps is not None and not np.all(np.diff(timestamps) == 300_000_000_000):
        raise ValueError('Timestamp gaps: explicit reindexing/missing policy required')
    train_end, val_end = int(.7 * len(values)), int(.8 * len(values))
    train = values[:train_end]
    observed = train[np.isfinite(train) & (train > 0)]
    mean, std = float(observed.mean(dtype=np.float64)), float(observed.std(dtype=np.float64))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError('Invalid training statistics')
    manifest = {'dataset': dataset, 'shape': list(values.shape),
                'hashes': {p.name: sha256(p) for p in paths},
                'boundaries': [train_end, val_end], 'mean': mean, 'std': std,
                'train_missing_fraction': float(1 - len(observed) / train.size),
                'unit': 'mph', 'history': 12, 'horizon': 12,
                'graph_symmetric': bool(np.allclose(graph, graph.T)),
                'graph_policy': 'STGCN uses max(A,A.T); DCRNN/STTN use supplied directed road graph',
                'test_status': 'previously_explored' if dataset == 'METR_LA' else 'unopened_for_model_selection'}
    return values, graph, timestamps, manifest
