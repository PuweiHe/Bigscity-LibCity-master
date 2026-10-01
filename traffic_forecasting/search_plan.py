"""Deterministic validation-search design, independent of running protocols."""
import copy
import itertools
import random


def validate_trial(trial):
    c = trial['config']
    if trial['model'] == 'STGCN':
        b = c['blocks']
        if len(b) != 2 or 12 - 4 * (c['Kt'] - 1) <= 0:
            raise ValueError('STGCN requires two blocks and a positive output time axis')
        if b[0][0] != 1 or b[0][2] != b[1][0] or any(v <= 0 for row in b for v in row):
            raise ValueError('Inconsistent STGCN channels')
    elif trial['model'] == 'STTN':
        if c['embed_dim'] % c['num_heads']:
            raise ValueError('Attention width must be divisible by head count')
    elif trial['model'] != 'DCRNN':
        raise ValueError('Unsupported model')
    if trial['batch_size'] < 1 or trial['learning_rate'] <= 0 or trial.get('weight_decay', 0) < 0:
        raise ValueError('Invalid optimizer configuration')
    if trial.get('graph_topk') is not None and trial['model'] != 'STGCN':
        raise ValueError('Graph ablation is defined only for STGCN')


def candidates(reference, plan):
    """Sample without replacement from declared finite spaces; always include references."""
    rng = random.Random(plan['search_seed'])
    result = []
    for model, name in [('STGCN', 'stgcn_reference'), ('DCRNN', 'dcrnn_reference'), ('STTN', 'sttn_reference')]:
        base = copy.deepcopy(next(t for t in reference['trials'] if t['id'] == name))
        base['batch_size'] = reference['batch_size']
        base['weight_decay'] = 0.0
        result.append(base)
        space = plan['spaces'][model]
        keys = sorted(space)
        grid = list(itertools.product(*(space[k] for k in keys)))
        rng.shuffle(grid)
        seen = set()
        count = 0
        for row in grid:
            v = dict(zip(keys, row)); t = copy.deepcopy(base)
            for k in ['learning_rate', 'weight_decay', 'batch_size']:
                t[k] = v.pop(k)
            if model == 'STGCN':
                width = v.pop('width')
                v['blocks'] = [[1, width, 2 * width], [2 * width, width, 2 * width]]
            t['config'].update(v)
            # Exclude the fixed reference from the sampled candidates.
            signature = repr((t['config'], t['batch_size'], t['learning_rate'], t['weight_decay']))
            if signature in seen or all(t[k] == base[k] for k in ['config', 'batch_size', 'learning_rate', 'weight_decay']):
                continue
            seen.add(signature); count += 1
            t['id'] = f'{model.lower()}_search_{count:03d}'
            validate_trial(t); result.append(t)
            if count >= plan['search_trials'][model]:
                break
        if count != plan['search_trials'][model]:
            raise ValueError('Requested more unique candidates than available')
    return result


def interventions(base, prefix):
    """Hold optimization settings fixed when changing residual or road graph."""
    result = []
    for residual, topk in itertools.product([False, True], [None, 8, 16]):
        t = copy.deepcopy(base)
        t['id'] = f'{prefix}_res{int(residual)}_graph{topk or "full"}'
        t['config'].update(residual_last_speed=residual, missing_aware_residual=residual)
        t['graph_topk'] = topk
        validate_trial(t); result.append(t)
    return result


def rank(results):
    """Stable tie-breaking and equal seed weighting; no test fields consumed."""
    grouped = {}
    for r in results:
        grouped.setdefault(r['trial'], []).append(r['validation']['all']['overall']['mae_mph'])
    return sorted(grouped, key=lambda k: (sum(grouped[k]) / len(grouped[k]), k))
