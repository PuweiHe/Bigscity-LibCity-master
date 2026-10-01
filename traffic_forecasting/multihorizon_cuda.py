"""CUDA execution adapter for the frozen multi-horizon experiment.

Run this as a module. Importing it does not alter the CPU or MPS runners.
The existing model definitions, data windows, loss, and validation selection
remain unchanged; this adapter places batches/models on CUDA and makes the
epoch checkpoints portable.
"""
import os
import time

import numpy as np
import torch

from traffic_forecasting import multihorizon_mps as device_adapter
from traffic_forecasting import multihorizon_study as study


def rng_state():
    return {**device_adapter._cpu_rng_state(),
            'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    device_adapter._cpu_restore_rng(state)
    torch.cuda.set_rng_state_all(state['cuda'])


@torch.inference_mode()
def benchmark(model, dataset, batch_size):
    model.eval()
    output = {}
    for size in [1, batch_size]:
        example = next(iter(device_adapter.MPSDataLoader(dataset, batch_size=size)))
        for _ in range(3):
            study.predict(model, example)
        torch.cuda.synchronize(device_adapter.DEVICE)
        times = []
        for _ in range(15):
            start = time.perf_counter()
            study.predict(model, example)
            torch.cuda.synchronize(device_adapter.DEVICE)
            times.append(time.perf_counter() - start)
        output[str(size)] = {'median_ms': float(np.median(times) * 1000),
                             'p95_ms': float(np.percentile(times, 95) * 1000),
                             'windows_per_second': float(size / np.median(times))}
    return output


def install(device_index=0):
    """Install CUDA placement only inside this dedicated runner process."""
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; no silent CPU fallback')
    if not 0 <= device_index < torch.cuda.device_count():
        raise ValueError('CUDA device index is out of range')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':16:8', ':4096:8'):
        raise RuntimeError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python')
    torch.cuda.set_device(device_index)
    device_adapter.DEVICE = torch.device('cuda', device_index)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device_adapter.install()
    study.CODE_FILES.append('traffic_forecasting/multihorizon_cuda.py')
    study.rng_state = rng_state
    study.restore_rng = restore_rng
    study.benchmark = benchmark


if __name__ == '__main__':
    install(int(os.environ.get('TRAFFIC_CUDA_DEVICE', '0')))
    study.main()
