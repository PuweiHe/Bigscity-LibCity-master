"""Validation-only, resumable staged CUDA search. Never scores a test partition."""
import argparse
import copy
import fcntl
import json
import os
import platform
import subprocess
import traceback
from pathlib import Path

import numpy as np
import torch

from traffic_forecasting import multihorizon_cuda as cuda
from traffic_forecasting import multihorizon_study as study
from traffic_forecasting.multihorizon_data import MaskedWindows, atomic_json, load_development, sha256
from traffic_forecasting.search_plan import candidates, interventions, rank, validate_trial


def road_topk(graph, k):
    """Retain strongest road affinities per node, then symmetrize for Chebyshev filters."""
    if k is None:
        return graph
    if k < 1:
        raise ValueError('Positive neighbor count required')
    a = np.maximum(graph, graph.T).copy()
    np.fill_diagonal(a, 0)
    indices = np.argsort(-a, axis=1, kind='stable')[:, :min(k, len(a) - 1)]
    result = np.zeros_like(a)
    np.put_along_axis(result, indices, np.take_along_axis(a, indices, axis=1), axis=1)
    return np.maximum(result, result.T)


def horizon_objective(metrics):
    """Keep pooled metrics for diagnostics; select by equally weighted horizons."""
    metrics = copy.deepcopy(metrics)
    overall = metrics['all']['overall']
    overall['pooled_mae_mph'] = overall['mae_mph']
    values = [metrics['all']['horizons'][str(h * 5)]['mae_mph'] for h in range(1, 13)]
    if any(v is None or not np.isfinite(v) for v in values):
        raise ValueError('Every horizon must have a finite validation MAE')
    overall['mae_mph'] = float(np.mean(values))
    return metrics


def immutable_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Frozen artifact changed: {path}')
    else:
        atomic_json(path, value)


def install():
    cuda.install(int(os.environ.get('TRAFFIC_CUDA_DEVICE', '0')))
    build, score = study.build_model, study.score
    study.build_model = lambda trial, graph: build(trial, road_topk(graph, trial.get('graph_topk')))
    study.score = lambda *a, **kw: horizon_objective(score(*a, **kw))
    study.CODE_FILES.extend(['traffic_forecasting/search_plan.py', 'traffic_forecasting/cuda_search.py'])


def smoke(trial, graph, windows, manifest):
    """Check one full configured batch before each candidate's real training."""
    loader = study.DataLoader(windows, batch_size=trial['batch_size'], shuffle=False)
    batch = next(iter(loader))
    if len(batch['X']) != trial['batch_size']:
        raise ValueError('Insufficient training windows for the configured full batch')
    model = study.build_model(trial, graph)
    optimizer = torch.optim.Adam(model.parameters(), lr=trial['learning_rate'])
    model.train(); output = study.predict(model, batch, training=True, step=0)
    loss = study.masked_loss(output, batch['y'], batch['y_mask'], manifest['std'])
    if loss is None or output.device.type != 'cuda':
        raise ValueError('CUDA loss or placement check failed')
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise ValueError('Missing or nonfinite gradients')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
    optimizer.step(); model.eval()
    with torch.inference_mode():
        predicted = study.predict(model, batch)
        # Changes to future labels cannot change inference.
        altered = dict(batch, y=torch.full_like(batch['y'], 1234))
        repeated = study.predict(model, altered)
    if not torch.isfinite(predicted).all() or not torch.equal(predicted, repeated):
        raise ValueError('Inference is nonfinite or depends on future labels')
    torch.cuda.synchronize()
    return {'shape': list(predicted.shape), 'loss_mph': float(loss.detach()),
            'gradient_tensors': len(gradients), 'peak_memory_bytes': torch.cuda.max_memory_allocated()}


def run_trial(root, trial, seed, graph, windows, manifest, plan, declaration):
    validate_trial(trial)
    out = root / trial['id'] / str(seed); out.mkdir(parents=True, exist_ok=True)
    identity = study.fingerprint({'declaration': declaration, 'manifest': manifest, 'trial': trial, 'seed': seed})
    immutable_json(out / 'effective_config.json', {'trial': trial, 'seed': seed, 'identity': identity,
                   'epochs': plan['epochs'], 'min_epochs': plan['min_epochs'], 'patience': plan['patience'],
                   'optimizer': 'Adam', 'scheduler': {'name': 'ReduceLROnPlateau', 'factor': .5, 'patience': 5},
                   'gradient_clip': 5., 'precision': 'float32', 'input_window': 12, 'output_window': 12})
    loaders = {k: study.DataLoader(v, batch_size=trial['batch_size'], shuffle=k == 'train', num_workers=0)
               for k, v in windows.items()}
    try:
        if not (out / 'result.json').exists():
            torch.cuda.reset_peak_memory_stats()
            if not (out / 'smoke.json').exists():
                atomic_json(out / 'smoke.json', smoke(trial, graph, windows['train'], manifest))
            torch.cuda.empty_cache()
        result = study.train_trial(trial, seed, graph, loaders['train'], loaders['validation'],
                                   manifest, plan, out, identity)
    except Exception:
        (out / 'error.log').write_text(traceback.format_exc())
        raise  # Do not silently drop difficult/OOM candidates or alter batch size.
    if not (out / 'resources.json').exists():
        atomic_json(out / 'resources.json', {'peak_allocated_bytes_this_process': torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes_this_process': torch.cuda.max_memory_reserved(),
                    'note': 'Peak covers this resumed process segment, not necessarily prior segments.'})
    torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, default=Path('configs/forecasting/cuda_search_v1.json'))
    p.add_argument('--reference', type=Path, default=Path('configs/forecasting/multihorizon.json'))
    p.add_argument('--data-dir', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--dataset', choices=['METR_LA', 'PEMS_BAY'], required=True)
    p.add_argument('--stage', choices=['screen', 'replicate', 'ablate', 'all'], default='all')
    a = p.parse_args()
    plan, reference = json.loads(a.plan.read_text()), json.loads(a.reference.read_text())
    root = a.output_dir / a.dataset; root.mkdir(parents=True, exist_ok=True)
    with (root / '.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        install(); torch.set_num_threads(plan['threads']); torch.use_deterministic_algorithms(True)
        # Device/runtime are frozen too: a different architecture requires a new run.
        environment = {'gpu': torch.cuda.get_device_name(), 'capability': list(torch.cuda.get_device_capability()),
                       'total_memory': torch.cuda.get_device_properties(0).total_memory,
                       'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                       'cudnn': torch.backends.cudnn.version(), 'python': platform.python_version(),
                       'driver': subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip()}
        declaration = study.freeze(root, {'search': plan, 'reference': reference, 'hardware': environment})
        values, graph, _, manifest = load_development(a.data_dir, a.dataset)
        # File presence elsewhere cannot prove a holdout is untouched.
        manifest['test_status'] = 'exploratory' if a.dataset == 'METR_LA' else 'provenance_review_required'
        immutable_json(root / 'data_manifest.json', manifest)
        train_end, val_end = manifest['boundaries']
        windows = {k: MaskedWindows(values, lo, hi, manifest['mean'], manifest['std'])
                   for k, (lo, hi) in {'train': (0, train_end), 'validation': (train_end, val_end)}.items()}
        trials = candidates(reference, plan)
        immutable_json(root / 'candidate_manifest.json', trials)
        screen = []
        for trial in trials:
            path = root / 'screen' / trial['id'] / str(plan['screen_seed']) / 'result.json'
            if a.stage in ['all', 'screen']:
                result = run_trial(root / 'screen', trial, plan['screen_seed'], graph, windows, manifest, plan, declaration)
            elif path.exists():
                result = json.loads(path.read_text())
            else:
                raise ValueError('Complete screening before replication')
            screen.append(result)
        shortlist = []
        for model in ['STGCN', 'DCRNN', 'STTN']:
            ids = rank([r for r in screen if r['model'] == model])[:plan['shortlist_per_model']]
            shortlist.extend(t for t in trials if t['id'] in ids)
        immutable_json(root / 'shortlist.json', shortlist)
        if a.stage == 'screen': return
        replicated = []
        for trial in shortlist:
            for seed in plan['seeds']:
                replicated.append(run_trial(root / 'replicate', trial, seed, graph, windows, manifest, plan, declaration))
        winners = {model: rank([r for r in replicated if r['model'] == model])[0]
                   for model in ['STGCN', 'DCRNN', 'STTN']}
        immutable_json(root / 'replicated_selection.json', winners)
        if a.stage == 'replicate': return
        tuned = next(t for t in shortlist if t['id'] == winners['STGCN'])
        compact = next(t for t in trials if t['id'] == 'stgcn_reference')
        factorial = interventions(compact, 'compact') + interventions(tuned, 'tuned')
        recursive = copy.deepcopy(compact)
        recursive['id'] = 'compact_recursive'; recursive['config']['direct_multi_step'] = False
        controls = [copy.deepcopy(t) for t in trials if t['id'] in ['dcrnn_reference', 'sttn_reference']]
        controls += [copy.deepcopy(t) for t in shortlist if t['id'] in [winners['DCRNN'], winners['STTN']]
                     and t['id'] not in [v['id'] for v in controls]]
        final_trials = factorial + [recursive] + controls
        immutable_json(root / 'final_trial_manifest.json', final_trials)
        final = [run_trial(root / 'final', t, s, graph, windows, manifest, plan, declaration)
                 for t in final_trials for s in plan['seeds']]
        order = rank(final)
        immutable_json(root / 'locked_selection.json', {'ranking': order, 'selected_trial': order[0],
                       'publish_seed': 42, 'results': final,
                       'test_evaluation': 'NOT RUN; requires separate provenance review and locked evaluator'})
        print('Validation search complete. Test data have not been scored.', flush=True)


if __name__ == '__main__':
    main()
