import copy
import json
from pathlib import Path

import numpy as np
import unittest
import tempfile

from traffic_forecasting.search_plan import candidates, interventions, rank, validate_trial
from traffic_forecasting.cuda_search import horizon_objective, immutable_json, road_topk

ROOT = Path(__file__).resolve().parents[2]


def design():
    return candidates(json.loads((ROOT / 'configs/forecasting/multihorizon.json').read_text()),
                      json.loads((ROOT / 'configs/forecasting/cuda_search_v1.json').read_text()))


class SearchTests(unittest.TestCase):
    def test_design_reproducible_and_legal(self):
        first = design()
        assert first == design()
        assert len(first) == 51
        assert len({t['id'] for t in first}) == 51
        for t in first: validate_trial(t)
        assert {t['config']['Kt'] for t in first if t['model'] == 'STGCN'} == {2, 3}


    def test_invalid_temporal_shape_rejected_before_gpu(self):
        t = design()[0]; t['config']['Kt'] = 4
        with self.assertRaisesRegex(ValueError, 'time axis'): validate_trial(t)


    def test_factorial_holds_optimizer_channels_and_batch_fixed(self):
        t = design()[3]; before = copy.deepcopy(t)
        cells = interventions(t, 'tuned')
        assert t == before and len(cells) == 6
        for c in cells:
            for key in ['learning_rate', 'weight_decay', 'batch_size']: assert c[key] == t[key]
            assert c['config']['blocks'] == t['config']['blocks']
        assert {(c['config']['residual_last_speed'], c['graph_topk']) for c in cells} == {
            (r, k) for r in [False, True] for k in [None, 8, 16]}


    def test_graph_uses_only_existing_roads_and_is_symmetric(self):
        g = np.array([[0., 3, 2], [1, 0, 4], [0, 0, 0]])
        old = g.copy(); out = road_topk(g, 1)
        np.testing.assert_array_equal(g, old)
        np.testing.assert_array_equal(out, out.T)
        assert (out <= np.maximum(g, g.T)).all()
        assert out[0, 2] == 0 and out[0, 1] == 3 and out[1, 2] == 4


    def test_selection_cannot_use_test_scores(self):
        rows = [{'trial': t, 'validation': {'all': {'overall': {'mae_mph': v}}}, 'test_mae': test}
                for t, v, test in [('a', 2, 100), ('a', 4, 100), ('b', 4, 0)]]
        assert rank(rows) == ['a', 'b']


    def test_horizon_objective_not_mask_count_weighted(self):
        data = {'all': {'overall': {'mae_mph': 99},
                       'horizons': {str(5 * h): {'mae_mph': float(h)} for h in range(1, 13)}}}
        out = horizon_objective(data)
        assert out['all']['overall']['mae_mph'] == 6.5
        assert out['all']['overall']['pooled_mae_mph'] == 99
        assert data['all']['overall']['mae_mph'] == 99


    def test_frozen_selection_cannot_be_overwritten(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        tmp_path = Path(temporary.name)
        p = tmp_path / 'selection.json'
        immutable_json(p, {'selected': 'a'})
        immutable_json(p, {'selected': 'a'})
        with self.assertRaisesRegex(ValueError, 'Frozen'): immutable_json(p, {'selected': 'b'})
        assert json.loads(p.read_text()) == {'selected': 'a'}
