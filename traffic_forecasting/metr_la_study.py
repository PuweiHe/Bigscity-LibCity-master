"""Reproducible, resource-bounded METR-LA graph forecasting study.

Run after extracting METR_LA.zip into raw_data/METR_LA. The archive is not
redistributed. Model selection uses validation MAE; test is evaluated once.
"""

import argparse
import importlib.util
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

def model_class(name):
    # The package initializer imports unrelated optional DGL models.
    path = Path(__file__).resolve().parents[1] / f'libcity/model/traffic_speed_prediction/{name}.py'
    spec = importlib.util.spec_from_file_location(f'metr_la_{name}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


class Windows(Dataset):
    def __init__(self, series, starts, mean, std, history=12):
        self.series = series
        self.starts = starts
        self.mean = mean
        self.std = std
        self.history = history

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = int(self.starts[index])
        x = self.series[start:start + self.history].copy()
        y = self.series[start + self.history:start + self.history + 1].copy()
        x = np.where(x > 0, (x - self.mean) / self.std, 0)
        y = np.where(y > 0, (y - self.mean) / self.std, 0)
        return torch.from_numpy(x[..., None].astype(np.float32)), torch.from_numpy(y[..., None].astype(np.float32))


def starts_for_split(begin, end, stride, history=12):
    return np.arange(begin, end - history, stride, dtype=np.int64)


def masked_mae(pred, target, mean, std):
    valid = target != 0
    if not torch.any(valid):
        return None
    return torch.abs(pred[valid] - target[valid]).mean() * std


def last_available_speed(x):
    """Use the most recent observed speed, or the train mean if all 12 are missing."""
    observed = x != 0
    reverse_index = observed.flip(1).int().argmax(dim=1, keepdim=True)
    index = x.shape[1] - 1 - reverse_index
    anchor = torch.gather(x, 1, index)
    return torch.where(observed.any(dim=1, keepdim=True), anchor, 0)


def persistence_mae(loader, std):
    error = 0.0
    count = 0
    for x, y in loader:
        valid = y != 0
        error += ((last_available_speed(x) - y)[valid] * std).abs().sum().item()
        count += valid.sum().item()
    return error / count


def evaluate(model, loader, mean, std):
    model.eval()
    error = 0.0
    squared = 0.0
    count = 0
    baseline_error = 0.0
    with torch.no_grad():
        for x, y in loader:
            pred = model({'X': x, 'y': y})
            valid = y != 0
            delta = (pred - y)[valid] * std
            error += delta.abs().sum().item()
            squared += delta.square().sum().item()
            count += valid.sum().item()
            anchor = last_available_speed(x)
            baseline_error += ((anchor - y)[valid] * std).abs().sum().item()
    return {'mae_mph': error / count, 'rmse_mph': (squared / count) ** 0.5,
            'persistence_mae_mph': baseline_error / count, 'observations': count}


def run_model(name, config, adj, loaders, mean, std, epochs, seed,
              progress_path=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device('cpu')
    features = {'num_nodes': adj.shape[0], 'feature_dim': 1, 'output_dim': 1,
                'adj_mx': adj}
    common = {'device': device, 'input_window': 12, 'output_window': 1}
    common.update(config)
    model = model_class(name)(common, features)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    best = None
    first_epoch = 0
    if progress_path is not None and progress_path.exists():
        progress = torch.load(progress_path, map_location='cpu', weights_only=True)
        if (progress['model'] != name or progress['config'] != config
                or progress['seed'] != seed or progress['epochs'] != epochs
                or not np.isclose(progress['mean'], mean)
                or not np.isclose(progress['std'], std)):
            raise ValueError('Training progress does not match this experiment')
        model.load_state_dict(progress['current_model'])
        optimizer.load_state_dict(progress['optimizer'])
        torch.set_rng_state(progress['torch_rng'])
        best = progress['best']
        first_epoch = progress['completed_epoch']
    started = time.monotonic()
    for epoch in range(first_epoch, epochs):
        model.train()
        losses = []
        for x, y in loaders['train']:
            optimizer.zero_grad()
            pred = model({'X': x, 'y': y}, batches_seen=epoch) if name == 'DCRNN' else model({'X': x})
            loss = masked_mae(pred, y, mean, std)
            if loss is None:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.item())
        validation = evaluate(model, loaders['validation'], mean, std)
        print(json.dumps({'model': name, 'config': config, 'epoch': epoch + 1,
                          'train_mae_mph': float(np.mean(losses)),
                          'validation': validation, 'elapsed_s': round(time.monotonic() - started, 1)}), flush=True)
        if best is None or validation['mae_mph'] < best['validation']['mae_mph']:
            best = {'epoch': epoch + 1, 'validation': validation,
                    'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        if progress_path is not None:
            progress = {'model': name, 'config': config, 'seed': seed,
                        'epochs': epochs, 'mean': mean, 'std': std,
                        'completed_epoch': epoch + 1,
                        'current_model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'torch_rng': torch.get_rng_state(), 'best': best}
            temporary = progress_path.with_suffix('.tmp')
            torch.save(progress, temporary)
            os.replace(temporary, progress_path)
    model.load_state_dict(best['state'])
    return model, {'epoch': best['epoch'], 'validation': best['validation']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('raw_data/METR_LA'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/metr_la'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--train-stride', type=int, default=24)
    parser.add_argument('--eval-stride', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--models', nargs='+', choices=['STGCN', 'DCRNN', 'STTN'],
                        default=['STGCN', 'DCRNN'])
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--sttn-candidate-index', type=int,
                        help='Train only this STTN candidate; useful after a validation search')
    parser.add_argument('--sttn-candidate-order', nargs='+', type=int,
                        help='Train every STTN candidate in the specified index order')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--validation-only', action='store_true',
                      help='Select and save checkpoints without reading the test partition')
    mode.add_argument('--evaluation-only', action='store_true',
                      help='Evaluate previously selected checkpoints on the test partition')
    args = parser.parse_args()
    torch.set_num_threads(4)
    series = np.load(args.data_dir / 'METR_LA.npz')['data'].astype(np.float32)
    adj = np.load(args.data_dir / 'METR_LA_rn_adj.npy').astype(np.float32)
    if series.ndim != 2 or adj.shape != (series.shape[1], series.shape[1]):
        raise ValueError('METR-LA speed and adjacency dimensions do not match')
    if not np.isfinite(series).all() or np.any(series < 0):
        raise ValueError('METR-LA speeds must be finite and nonnegative')
    n = len(series)
    train_end, validation_end = int(n * .7), int(n * .8)
    valid_train = series[:train_end][series[:train_end] > 0]
    mean, std = float(valid_train.mean()), float(valid_train.std())
    ranges = {'train': (0, train_end, args.train_stride),
              'validation': (train_end, validation_end, args.eval_stride),
              'test': (validation_end, n, args.eval_stride)}
    starts = {k: starts_for_split(*v) for k, v in ranges.items()}
    loaders = {k: DataLoader(Windows(series, s, mean, std), batch_size=args.batch_size,
                             shuffle=(k == 'train'), num_workers=0)
               for k, s in starts.items()}
    # Candidate hyperparameters are declared before looking at the test partition.
    candidates = {
        'STGCN': [
            {'Ks': 3, 'Kt': 3, 'blocks': [[1, 8, 16], [16, 8, 16]],
             'dropout': 0.0, 'graph_conv_type': 'chebconv', 'learning_rate': .003},
            {'Ks': 3, 'Kt': 3, 'blocks': [[1, 16, 32], [32, 16, 32]],
             'dropout': .1, 'graph_conv_type': 'chebconv', 'learning_rate': .001,
             'residual_last_speed': True},
        ],
        'DCRNN': [
            {'max_diffusion_step': 1, 'num_rnn_layers': 1, 'rnn_units': 16,
             'filter_type': 'dual_random_walk', 'use_curriculum_learning': False,
             'learning_rate': .003},
            {'max_diffusion_step': 2, 'num_rnn_layers': 1, 'rnn_units': 32,
             'filter_type': 'dual_random_walk', 'use_curriculum_learning': False,
             'learning_rate': .001, 'residual_last_speed': True},
        ],
        'STTN': [
            {'embed_dim': 16, 'num_layers': 1, 'num_heads': 2,
             'forward_expansion': 2, 'dropout_rate': .1, 'learning_rate': .001},
            {'embed_dim': 16, 'num_layers': 1, 'num_heads': 2,
             'forward_expansion': 2, 'dropout_rate': .1, 'learning_rate': .001,
             'residual_last_speed': True},
            {'embed_dim': 32, 'num_layers': 2, 'num_heads': 4,
             'forward_expansion': 2, 'dropout_rate': .1, 'learning_rate': .0005,
             'residual_last_speed': True, 'residual_scale': .5},
        ],
    }
    if args.sttn_candidate_index is not None and args.sttn_candidate_order is not None:
        raise ValueError('Select either one STTN candidate or an ordered full search')
    if args.sttn_candidate_order is not None:
        order = args.sttn_candidate_order
        if args.models != ['STTN'] or sorted(order) != list(range(len(candidates['STTN']))):
            raise ValueError('--sttn-candidate-order must list every STTN index once')
        candidates['STTN'] = [candidates['STTN'][index] for index in order]
    if args.sttn_candidate_index is not None:
        if args.models != ['STTN'] or not 0 <= args.sttn_candidate_index < len(candidates['STTN']):
            raise ValueError('--sttn-candidate-index requires one valid STTN model index')
        candidates['STTN'] = [candidates['STTN'][args.sttn_candidate_index]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {'dataset': 'METR-LA', 'source_shape': list(series.shape), 'nodes': int(adj.shape[0]),
              'unit': 'mph', 'target': 'next 5-minute traffic speed',
              'split_boundaries': [train_end, validation_end],
              'window_counts': {k: len(v) for k, v in starts.items()},
              'train_only_mean': mean, 'train_only_std': std, 'seed': args.seed,
              'epochs': args.epochs, 'batch_size': args.batch_size, 'models': {}}
    if args.sttn_candidate_index is not None:
        report['sttn_candidate_original_index'] = args.sttn_candidate_index
    if args.sttn_candidate_order is not None:
        report['sttn_candidate_original_indices'] = args.sttn_candidate_order
    report_path = args.output_dir / 'report.json'
    if args.evaluation_only:
        report = json.loads(report_path.read_text())
        if (report['split_boundaries'] != [train_end, validation_end]
                or report['source_shape'] != list(series.shape)
                or report['window_counts']['test'] != len(starts['test'])
                or not np.isclose(report['train_only_mean'], mean)
                or not np.isclose(report['train_only_std'], std)):
            raise ValueError('Checkpoint report does not match dataset or evaluation split')
        validation_baseline = persistence_mae(loaders['validation'], std)
        for name in args.models:
            checkpoint = torch.load(args.output_dir / f'{name.lower()}.pt',
                                    map_location='cpu', weights_only=True)
            if not np.isclose(checkpoint['mean'], mean) or not np.isclose(checkpoint['std'], std):
                raise ValueError(f'{name} checkpoint normalization does not match dataset')
            config = {'device': torch.device('cpu'), 'input_window': 12, 'output_window': 1,
                      **checkpoint['config']}
            model = model_class(name)(config, {'num_nodes': adj.shape[0], 'feature_dim': 1,
                                               'output_dim': 1, 'adj_mx': adj})
            model.load_state_dict(checkpoint['state_dict'])
            for trial in report['models'][name]['trials']:
                trial['validation']['persistence_mae_mph'] = validation_baseline
            report['models'][name]['test'] = evaluate(model, loaders['test'], mean, std)
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
        return
    for name in args.models:
        trials = []
        best_model = None
        for index, config in enumerate(candidates[name]):
            progress_path = (args.output_dir / f'{name.lower()}_candidate_{index}.progress.pt'
                             if name == 'STTN' else None)
            model, result = run_model(name, config, adj, loaders, mean, std,
                                      args.epochs, args.seed, progress_path)
            trials.append({'config': config, **result})
            if best_model is None or result['validation']['mae_mph'] < trials[best_model[0]]['validation']['mae_mph']:
                best_model = (len(trials) - 1, model)
        selected_index, selected_model = best_model
        test = None if args.validation_only else evaluate(selected_model, loaders['test'], mean, std)
        torch.save({'model': name, 'config': candidates[name][selected_index],
                    'mean': mean, 'std': std, 'state_dict': selected_model.state_dict()},
                   args.output_dir / f'{name.lower()}.pt')
        report['models'][name] = {'trials': trials, 'selected_index': selected_index,
                                  'parameter_count': sum(p.numel() for p in selected_model.parameters())}
        if test is not None:
            report['models'][name]['test'] = test
        report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
