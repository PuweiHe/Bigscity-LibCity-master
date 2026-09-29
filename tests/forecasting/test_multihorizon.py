import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from libcity.model.forecasting_utils import last_observed_value
from traffic_forecasting.multihorizon_data import MaskedWindows
from traffic_forecasting import multihorizon_study as study


class MultiHorizonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
        cls.trials = json.loads((study.ROOT / 'configs/forecasting/multihorizon.json').read_text())['trials']

    def test_observed_mean_is_not_missing_and_targets_stay_inside_split(self):
        values = np.full((60, 3), 10., dtype=np.float32)
        values[12, 0] = 0
        data = MaskedWindows(values, 0, 30, 10., 2.)
        batch = data[0]
        self.assertTrue(batch['X_mask'].all())
        self.assertFalse(batch['y_mask'][0, 0, 0])
        self.assertTrue(batch['y_mask'][0, 1, 0])
        self.assertEqual(int(data.starts[-1]) + 24, 30)
        self.assertEqual(tuple(batch['y'].shape), (12, 3, 1))

    def test_explicit_anchor_keeps_observed_zero_and_ignores_missing_tail(self):
        x = torch.tensor([1., 0., 5.]).reshape(1, 3, 1, 1)
        mask = torch.tensor([True, True, False]).reshape_as(x)
        self.assertEqual(last_observed_value(x, mask).item(), 0)
        self.assertEqual(last_observed_value(x, torch.zeros_like(mask)).item(), 0)

    def test_recursive_residual_propagates_explicit_mask(self):
        trial = copy.deepcopy(self.trials[0])
        trial['config'].update(direct_multi_step=False, residual_last_speed=True,
                               missing_aware_residual=True)
        model = study.build_model(trial, self.adj).eval()
        with torch.no_grad():
            model.output.fc.conv.weight.zero_(); model.output.fc.conv.bias.zero_()
        x = torch.ones(1, 12, 3, 1)
        x[:, -2:] = 0
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[:, -1:] = False
        # The second-last observation is a valid normalized zero, not missing.
        output = study.predict(model, {'X': x, 'X_mask': mask})
        torch.testing.assert_close(output, torch.zeros(1, 12, 3, 1))

    def test_loss_does_not_broadcast_and_counts_observed_zero(self):
        y = torch.zeros(1, 12, 3, 1)
        mask = torch.ones_like(y, dtype=torch.bool)
        self.assertEqual(study.masked_loss(y + 2, y, mask, 3).item(), 6)
        with self.assertRaises(ValueError): study.masked_loss(y[:, :1], y, mask, 1)
        self.assertIsNone(study.masked_loss(y, y, ~mask, 1))

    def test_all_models_predict_twelve_steps_with_finite_gradients(self):
        for trial in self.trials:
            model = study.build_model(trial, self.adj)
            x = torch.randn(2, 12, 3, 1, requires_grad=True)
            batch = {'X': x, 'X_mask': torch.ones_like(x, dtype=torch.bool),
                     'y': torch.randn(2, 12, 3, 1)}
            output = study.predict(model, batch)
            self.assertEqual(tuple(output.shape), (2, 12, 3, 1), trial['id'])
            output.square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_inference_ignores_future_labels_and_is_repeatable(self):
        for trial in self.trials:
            model = study.build_model(trial, self.adj).eval()
            x = torch.randn(2, 12, 3, 1)
            batch = {'X': x, 'X_mask': torch.ones_like(x, dtype=torch.bool), 'y': x}
            a = study.predict(model, batch)
            batch['y'] = torch.full_like(x, float('nan'))
            b = study.predict(model, batch)
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_stgcn_graph_buffer_follows_dtype_without_changing_legacy_keys(self):
        trial = self.trials[0]
        config = copy.deepcopy(trial)
        model = study.build_model(trial, self.adj).double()
        self.assertEqual(model.st_conv1.sconv.Lk.dtype, torch.float64)
        self.assertNotIn('st_conv1.sconv.Lk', model.state_dict())
        self.assertEqual(trial, config)
        x = torch.randn(2, 12, 3, 1, dtype=torch.float64)
        self.assertTrue(torch.isfinite(model({'X': x})).all())

    def test_metric_denominator_and_horizon_groups(self):
        values = np.full((30, 3), 10., dtype=np.float32)
        data = MaskedWindows(values, 0, 30, 10., 2.)
        report = study.score(None, DataLoader(data, batch_size=2), 2., 10., True, 'persistence')
        self.assertEqual(report['all']['overall']['mae_mph'], 0)
        self.assertEqual(report['all']['overall']['observations'], 7 * 12 * 3)
        self.assertEqual(report['free_flow']['overall']['mae_mph'], None)

    def test_resume_matches_uninterrupted_parameters(self):
        trial = copy.deepcopy(self.trials[0])
        trial['config']['blocks'] = [[1, 2, 2], [2, 2, 2]]
        values = np.random.default_rng(4).uniform(1, 20, (30, 3)).astype(np.float32)
        data = MaskedWindows(values, 0, 30, 10., 2.)
        train = DataLoader(data, batch_size=3, shuffle=True)
        val = DataLoader(data, batch_size=3)
        manifest = {'std': 2., 'mean': 10., 'hashes': {}, 'dataset': 'synthetic'}
        protocol = {'epochs': 2, 'min_epochs': 2, 'patience': 3}
        with tempfile.TemporaryDirectory() as d:
            first, resumed = Path(d) / 'first', Path(d) / 'resumed'
            study.train_trial(trial, 42, self.adj, train, val, manifest, protocol, first, 'unit')
            real_score = study.score
            calls = [0]
            def interrupt(*args, **kwargs):
                calls[0] += 1
                if calls[0] == 2: raise RuntimeError('simulated interruption')
                return real_score(*args, **kwargs)
            with patch.object(study, 'score', side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, 'simulated'):
                    study.train_trial(trial, 42, self.adj, train, val, manifest, protocol, resumed, 'unit')
            study.train_trial(trial, 42, self.adj, train, val, manifest, protocol, resumed, 'unit')
            a = torch.load(first / 'progress.pt', weights_only=True)
            b = torch.load(resumed / 'progress.pt', weights_only=True)
            for key in a['model']: torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
            self.assertEqual(a['best']['epoch'], b['best']['epoch'])


if __name__ == '__main__': unittest.main()
