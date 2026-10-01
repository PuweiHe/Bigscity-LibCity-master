"""MPS execution adapter for the frozen multi-horizon study protocol.

This module leaves the CPU study untouched. It changes only device placement,
checkpoint portability, and a mathematically equivalent 2-D Linear call needed
for STTN backward on the tested PyTorch MPS runtime.
"""
import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader as TorchDataLoader

from traffic_forecasting import multihorizon_study as study
from traffic_forecasting.metr_la_study import model_class


DEVICE = torch.device('mps')


class MPSDataLoader:
    """Keep each epoch re-iterable while moving only its current batch to MPS."""

    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.kwargs = kwargs

    def __iter__(self):
        for batch in TorchDataLoader(self.dataset, **self.kwargs):
            yield {key: value.to(DEVICE) for key, value in batch.items()}


def linear_2d(self, inputs):
    # MPS LinearBackward rejects this STTN model's rank-5 attention tensors.
    # Linear acts only on the final dimension, so flattening leading axes is exact.
    if inputs.ndim > 2:
        output = F.linear(inputs.reshape(-1, inputs.shape[-1]), self.weight, self.bias)
        return output.reshape(*inputs.shape[:-1], self.out_features)
    return F.linear(inputs, self.weight, self.bias)


def build_model(trial, graph):
    config = copy.deepcopy(trial['config'])
    config.update(device=DEVICE, input_window=12, output_window=12)
    if trial['model'] == 'STGCN':
        graph = np.maximum(graph, graph.T)
    model = model_class(trial['model'])(config, {
        'num_nodes': len(graph), 'feature_dim': 1, 'output_dim': 1, 'adj_mx': graph}).to(DEVICE)
    if trial['model'] == 'STTN':
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.forward = linear_2d.__get__(module, torch.nn.Linear)
    return model


@torch.inference_mode()
def score(model, loader, std, mean, diagnostics=False, baseline=None):
    if model is not None:
        model.eval()
    sums = {}
    for batch in loader:
        if baseline == 'persistence':
            pred = study.last_observed_value(batch['X'], batch['X_mask']).expand_as(batch['y'])
        elif baseline == 'moving_mean':
            observed = batch['X_mask']
            pred = ((batch['X'] * observed).sum(1, keepdim=True) /
                    observed.sum(1, keepdim=True).clamp_min(1)).expand_as(batch['y'])
        else:
            pred = study.predict(model, batch)
        target, valid = batch['y'], batch['y_mask']
        if pred.shape != target.shape or not torch.isfinite(pred).all():
            raise ValueError('Invalid evaluation output')
        # MPS does not support float64. Use the same CPU double-precision sums as
        # the CPU study after copying predictions; this also keeps group masks exact.
        error = (pred.float().cpu().double() - target.float().cpu().double()) * std
        speeds = target.float().cpu().double() * std + mean
        valid = valid.cpu()
        groups = {'all': valid}
        if diagnostics:
            groups.update(slow=valid & (speeds < 30),
                          moderate=valid & (speeds >= 30) & (speeds < 55),
                          free_flow=valid & (speeds >= 55),
                          missing_history=valid & ((~batch['X_mask']).float().cpu().mean(1, keepdim=True) > .25))
        for name, mask in groups.items():
            if name not in sums:
                sums[name] = np.zeros((3, 12), dtype=np.float64)
            sums[name][0] += (error.abs() * mask).sum((0, 2, 3)).numpy()
            sums[name][1] += (error.square() * mask).sum((0, 2, 3)).numpy()
            sums[name][2] += mask.sum((0, 2, 3)).numpy()

    def metrics(array):
        absolute, squared, count = array
        return {'mae_mph': float(absolute / count) if count else None,
                'rmse_mph': float(np.sqrt(squared / count)) if count else None,
                'observations': int(count)}

    result = {name: {'overall': metrics(array.sum(1)),
                     'horizons': {str((h + 1) * 5): metrics(array[:, h]) for h in range(12)}}
              for name, array in sums.items()}
    if not result['all']['overall']['observations']:
        raise ValueError('No observed validation/evaluation targets')
    return result


@torch.inference_mode()
def benchmark(model, dataset, batch_size):
    model.eval()
    output = {}
    for size in [1, batch_size]:
        example = next(iter(MPSDataLoader(dataset, batch_size=size)))
        for _ in range(3):
            study.predict(model, example)
        torch.mps.synchronize()
        times = []
        for _ in range(15):
            start = time.perf_counter()
            study.predict(model, example)
            torch.mps.synchronize()
            times.append(time.perf_counter() - start)
        output[str(size)] = {'median_ms': float(np.median(times) * 1000),
                             'p95_ms': float(np.percentile(times, 95) * 1000),
                             'windows_per_second': float(size / np.median(times))}
    return output


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def save_checkpoint(path, payload):
    temporary = path.with_suffix('.tmp')
    torch.save(cpu_tree(payload), temporary)
    os.replace(temporary, path)


def rng_state():
    return {**_cpu_rng_state(), 'mps': torch.mps.get_rng_state()}


def restore_rng(state):
    _cpu_restore_rng(state)
    torch.mps.set_rng_state(state['mps'])


_cpu_rng_state = study.rng_state
_cpu_restore_rng = study.restore_rng


def install():
    """Apply the adapter only inside the dedicated MPS runner process."""
    study.CODE_FILES.append('traffic_forecasting/multihorizon_mps.py')
    study.DataLoader = MPSDataLoader
    study.build_model = build_model
    study.score = score
    study.benchmark = benchmark
    study.save_checkpoint = save_checkpoint
    study.rng_state = rng_state
    study.restore_rng = restore_rng


if __name__ == '__main__':
    if not torch.backends.mps.is_available():
        raise SystemExit('MPS unavailable: run in a full macOS session; no CPU fallback')
    install()
    study.main()
