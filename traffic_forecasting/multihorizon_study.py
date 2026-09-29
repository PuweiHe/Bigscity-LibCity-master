"""Resumable, preregistered multi-horizon study; test scoring follows all training.

Run `--mode train`, then `--mode evaluate`. The public-data confirmation cohort
is PEMS-BAY; METR-LA remains an exploratory development benchmark.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from libcity.model.forecasting_utils import last_observed_value
from traffic_forecasting.metr_la_study import model_class
from traffic_forecasting.multihorizon_data import MaskedWindows, atomic_json, load_development, sha256

ROOT = Path(__file__).resolve().parents[1]
CODE_FILES = ['traffic_forecasting/prepare_pems_bay.py', 'traffic_forecasting/multihorizon_data.py',
              'traffic_forecasting/multihorizon_study.py',
              'traffic_forecasting/metr_la_study.py', 'libcity/model/forecasting_utils.py'] + [
                  f'libcity/model/traffic_speed_prediction/{n}.py' for n in ['STGCN', 'DCRNN', 'STTN']]


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def masked_loss(pred, target, mask, std):
    if pred.shape != target.shape or mask.shape != target.shape:
        raise ValueError('Prediction, target, and mask shapes must match exactly')
    if mask.dtype != torch.bool or not torch.isfinite(pred).all():
        raise ValueError('Boolean mask and finite predictions required')
    return (pred - target).abs()[mask].mean() * std if mask.any() else None


def predict(model, batch, training=False, step=0):
    # Future labels never enter evaluation/inference. Curriculum is disabled here.
    inputs = {'X': batch['X'], 'X_mask': batch['X_mask']}
    if type(model).__name__ == 'DCRNN':
        inputs['y'] = batch['y'] if training else None
        return model(inputs, batches_seen=step if training else None)
    if type(model).__name__ == 'STGCN' and not model.direct_multi_step:
        return model.predict(inputs)
    return model(inputs)


def build_model(trial, graph):
    config = copy.deepcopy(trial['config'])
    config.update(device=torch.device('cpu'), input_window=12, output_window=12)
    # Spectral Chebyshev convolution assumes an undirected graph.
    if trial['model'] == 'STGCN':
        graph = np.maximum(graph, graph.T)
    return model_class(trial['model'])(config, {
        'num_nodes': len(graph), 'feature_dim': 1, 'output_dim': 1, 'adj_mx': graph})


@torch.inference_mode()
def score(model, loader, std, mean, diagnostics=False, baseline=None):
    if model is not None:
        model.eval()
    sums = {}
    for batch in loader:
        if baseline == 'persistence':
            pred = last_observed_value(batch['X'], batch['X_mask']).expand_as(batch['y'])
        elif baseline == 'moving_mean':
            observed = batch['X_mask']
            pred = ((batch['X'] * observed).sum(1, keepdim=True) /
                    observed.sum(1, keepdim=True).clamp_min(1)).expand_as(batch['y'])
        else:
            pred = predict(model, batch)
        target, valid = batch['y'], batch['y_mask']
        if pred.shape != target.shape or not torch.isfinite(pred).all():
            raise ValueError('Invalid evaluation output')
        error = (pred - target).to(torch.float64) * std
        speeds = target * std + mean
        groups = {'all': valid}
        if diagnostics:
            groups.update(slow=valid & (speeds < 30),
                          moderate=valid & (speeds >= 30) & (speeds < 55),
                          free_flow=valid & (speeds >= 55),
                          missing_history=valid & ((~batch['X_mask']).float().mean(1, keepdim=True) > .25))
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


def save_checkpoint(path, payload):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def rng_state():
    state = np.random.get_state()
    return {'torch': torch.get_rng_state(), 'python': random.getstate(),
            'numpy': [state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])]}


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))


def train_trial(trial, seed, graph, train_loader, val_loader, manifest, protocol, out, identity):
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists():
        result = json.loads((out / 'result.json').read_text())
        if result['identity'] != identity:
            raise ValueError('Completed result belongs to another protocol')
        return result
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = build_model(trial, graph)
    optimizer = torch.optim.Adam(model.parameters(), lr=trial['learning_rate'],
                                 weight_decay=trial.get('weight_decay', 0))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=5)
    history, best, stale, step, elapsed = [], None, 0, 0, 0.
    progress = out / 'progress.pt'
    if progress.exists():
        state = torch.load(progress, map_location='cpu', weights_only=True)
        if state['identity'] != identity:
            raise ValueError('Checkpoint identity mismatch; refusing unsafe resume')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        history, best, stale, step, elapsed = (state[k] for k in ['history', 'best', 'stale', 'step', 'elapsed'])
        restore_rng(state['rng'])
    for epoch in range(len(history), protocol['epochs']):
        if epoch >= protocol['min_epochs'] and stale >= protocol['patience']:
            break
        began = time.monotonic(); model.train()
        absolute, count = 0., 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = predict(model, batch, training=True, step=step)
            loss = masked_loss(pred, batch['y'], batch['y_mask'], manifest['std'])
            if loss is None:
                continue
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step(); step += 1
            n = batch['y_mask'].sum().item()
            absolute += loss.item() * n; count += n
        validation = score(model, val_loader, manifest['std'], manifest['mean'])
        value = validation['all']['overall']['mae_mph']
        scheduler.step(value)
        duration = time.monotonic() - began; elapsed += duration
        row = {'epoch': epoch + 1, 'train_mae_mph': absolute / count,
               'validation': validation, 'learning_rate': optimizer.param_groups[0]['lr'],
               'duration_s': duration, 'last_gradient_norm': float(norm)}
        history.append(row)
        if best is None or value < best['validation']['all']['overall']['mae_mph']:
            best = {'epoch': epoch + 1, 'validation': validation,
                    'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
        save_checkpoint(progress, {'identity': identity, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                        'history': history, 'best': best, 'stale': stale, 'step': step,
                        'elapsed': elapsed, 'rng': rng_state()})
        atomic_json(out / 'epochs.json', history)
        atomic_json(out / 'status.json', {'identity': identity, 'completed_epoch': epoch + 1,
                     'best_epoch': best['epoch'], 'best_validation_mae_mph': best['validation']['all']['overall']['mae_mph'],
                     'elapsed_s': elapsed, 'state': 'training'})
        print(json.dumps({'dataset': manifest['dataset'], 'trial': trial['id'], 'seed': seed,
                         'epoch': epoch + 1, 'validation_mae': value, 'duration_s': duration}), flush=True)
    if best is None:
        raise ValueError('No completed epochs')
    save_checkpoint(out / 'selected.pt', {'identity': identity, 'trial': trial, 'seed': seed,
                     'normalization': {'mean': manifest['mean'], 'std': manifest['std']},
                     'data_hashes': manifest['hashes'], **best})
    result = {'identity': identity, 'trial': trial['id'], 'model': trial['model'], 'seed': seed,
              'completed_epochs': len(history), 'selected_epoch': best['epoch'],
              'validation': best['validation'], 'training_seconds': elapsed,
              'parameters': sum(p.numel() for p in model.parameters())}
    atomic_json(out / 'result.json', result)
    atomic_json(out / 'status.json', {**result, 'state': 'complete'})
    return result


@torch.inference_mode()
def benchmark(model, dataset, batch_size):
    model.eval()
    output = {}
    for batch in [1, batch_size]:
        example = next(iter(DataLoader(dataset, batch_size=batch)))
        for _ in range(3): predict(model, example)
        times = []
        for _ in range(15):
            begin = time.perf_counter(); predict(model, example)
            times.append(time.perf_counter() - begin)
        output[str(batch)] = {'median_ms': float(np.median(times) * 1000),
                             'p95_ms': float(np.percentile(times, 95) * 1000),
                             'windows_per_second': float(batch / np.median(times))}
    return output


def freeze(root, protocol):
    declaration = {'protocol': protocol,
                   'code_sha256': {name: sha256(ROOT / name) for name in CODE_FILES},
                   'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                                   'numpy': np.__version__, 'platform': platform.platform()}}
    path = root / 'frozen_protocol.json'
    if path.exists():
        old = json.loads(path.read_text())
        if old != declaration:
            raise ValueError('Frozen code/config/environment changed; use a new study directory')
    else:
        atomic_json(path, declaration)
    return declaration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=['train', 'evaluate'], required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    torch.set_num_threads(protocol['threads'])
    torch.use_deterministic_algorithms(True)
    root = args.output_dir; root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.runner.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another study process is already active')
    declaration = freeze(root, protocol)
    all_results = []
    for dataset in protocol['datasets']:
        values, graph, timestamps, manifest = load_development(args.data_dir, dataset)
        target = root / dataset; target.mkdir(exist_ok=True)
        manifest_path = target / 'data_manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Data changed since study was frozen')
        atomic_json(manifest_path, manifest)
        train_end, val_end = manifest['boundaries']
        splits = {'train': (0, train_end), 'validation': (train_end, val_end)}
        windows = {name: MaskedWindows(values, a, b, manifest['mean'], manifest['std'])
                   for name, (a, b) in splits.items()}
        if args.mode == 'evaluate':
            # Every preregistered run must finish before the first test target is scored.
            expected = [root / d / t['id'] / str(s) / 'result.json'
                        for d in protocol['datasets'] for t in protocol['trials'] for s in protocol['seeds']]
            if not all(p.exists() for p in expected):
                raise ValueError('Training incomplete: final evaluation is locked')
            if (target / 'test_report.json').exists():
                all_results.append(json.loads((target / 'test_report.json').read_text()))
                continue
            windows['test'] = MaskedWindows(values, val_end, len(values), manifest['mean'], manifest['std'])
        loaders = {name: DataLoader(data, batch_size=protocol['batch_size'], shuffle=(name == 'train'), num_workers=0)
                   for name, data in windows.items()}
        results = []
        for trial in protocol['trials']:
            for seed in protocol['seeds']:
                out = target / trial['id'] / str(seed)
                identity = fingerprint({'declaration': declaration, 'data': manifest, 'trial': trial, 'seed': seed})
                if args.mode == 'train':
                    result = train_trial(trial, seed, graph, loaders['train'], loaders['validation'],
                                         manifest, protocol, out, identity)
                else:
                    result = json.loads((out / 'result.json').read_text())
                    checkpoint = torch.load(out / 'selected.pt', map_location='cpu', weights_only=True)
                    if checkpoint['identity'] != identity or result['identity'] != identity:
                        raise ValueError('Evaluation artifact identity mismatch')
                    cached = out / 'test.json'
                    if cached.exists():
                        measured = json.loads(cached.read_text())
                        if measured['identity'] != identity: raise ValueError('Cached evaluation mismatch')
                    else:
                        model = build_model(trial, graph); model.load_state_dict(checkpoint['state_dict'])
                        measured = {'identity': identity,
                                    'test': score(model, loaders['test'], manifest['std'], manifest['mean'], True),
                                    'inference': benchmark(model, windows['validation'], protocol['batch_size'])}
                        atomic_json(cached, measured)
                    result = {**result, **measured}
                results.append(result)
        means = {t['id']: float(np.mean([r['validation']['all']['overall']['mae_mph'] for r in results if r['trial'] == t['id']]))
                 for t in protocol['trials']}
        report = {'dataset': dataset, 'protocol_sha256': fingerprint(declaration), 'manifest': manifest,
                  'window_counts': {k: len(v) for k, v in windows.items()}, 'runs': results,
                  'selected_by_mean_validation': min(means, key=means.get), 'mean_validation_mae': means}
        if args.mode == 'evaluate':
            report['baselines'] = {name: score(None, loaders['test'], manifest['std'], manifest['mean'], True, name)
                                   for name in ['persistence', 'moving_mean']}
        atomic_json(target / ('selection.json' if args.mode == 'train' else 'test_report.json'), report)
        all_results.append(report)
    atomic_json(root / ('training_complete.json' if args.mode == 'train' else 'evaluation_complete.json'),
                {'protocol_sha256': fingerprint(declaration), 'datasets': [r['dataset'] for r in all_results]})
    print('Completed', args.mode, flush=True)


if __name__ == '__main__':
    main()
