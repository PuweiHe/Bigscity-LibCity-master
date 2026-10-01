"""One training and inference step per CUDA trial on real training windows.

This does not read or score validation/test targets and writes no experiment state.
"""
import argparse
import json
import os
from pathlib import Path

import torch

from traffic_forecasting import multihorizon_cuda as cuda
from traffic_forecasting import multihorizon_study as study
from traffic_forecasting.multihorizon_data import MaskedWindows, load_development


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    args = parser.parse_args()
    cuda.install(int(os.environ.get('TRAFFIC_CUDA_DEVICE', '0')))
    torch.use_deterministic_algorithms(True)
    protocol = json.loads(args.protocol.read_text())
    torch.set_num_threads(protocol['threads'])
    if protocol.get('device') != 'cuda':
        raise ValueError('The smoke check requires the separate CUDA protocol')
    for dataset in protocol['datasets']:
        values, graph, _, manifest = load_development(args.data_dir, dataset)
        windows = MaskedWindows(values, 0, 48, manifest['mean'], manifest['std'])
        batch = next(iter(study.DataLoader(windows, batch_size=protocol['batch_size'])))
        for trial in protocol['trials']:
            torch.manual_seed(17)
            model = study.build_model(trial, graph)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=trial['learning_rate'])
            output = study.predict(model, batch, training=True, step=0)
            loss = study.masked_loss(output, batch['y'], batch['y_mask'], manifest['std'])
            if output.shape != batch['y'].shape or output.device.type != 'cuda' or loss is None:
                raise RuntimeError(f'Invalid CUDA training output: {dataset}/{trial["id"]}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            if not any(p.grad is not None for p in model.parameters()):
                raise RuntimeError(f'No gradients: {dataset}/{trial["id"]}')
            optimizer.step()
            model.eval()
            with torch.inference_mode():
                predicted = study.predict(model, batch)
            if predicted.shape != output.shape or not torch.isfinite(predicted).all():
                raise RuntimeError(f'Invalid CUDA inference output: {dataset}/{trial["id"]}')
            torch.cuda.synchronize()
            print(json.dumps({'dataset': dataset, 'trial': trial['id'],
                              'device': str(output.device), 'shape': list(output.shape),
                              'train_loss_mph': float(loss)}), flush=True)
            del model, optimizer, output, loss, predicted
            torch.cuda.empty_cache()
    print('CUDA smoke check passed for every configured dataset and trial', flush=True)


if __name__ == '__main__':
    main()
