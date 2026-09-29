import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, TensorDataset

from traffic_forecasting import metr_la_study
from traffic_forecasting.metr_la_study import Windows, last_available_speed, starts_for_split


def load_graph_model(name):
    path = Path(__file__).resolve().parents[2] / f'libcity/model/traffic_speed_prediction/{name}.py'
    spec = importlib.util.spec_from_file_location(f'test_{name}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GraphModelTests(unittest.TestCase):
    def test_sparse_support_keeps_values_attached_to_coordinates(self):
        dcrnn = load_graph_model('DCRNN')
        lap = sp.coo_matrix((np.array([7., 2., 5.], dtype=np.float32),
                             (np.array([2, 0, 1]), np.array([0, 2, 1]))), shape=(3, 3))
        actual = dcrnn.DCGRUCell._build_sparse_matrix(lap, torch.device('cpu')).to_dense().numpy()
        np.testing.assert_array_equal(actual, lap.toarray())

    def test_each_diffusion_support_starts_from_original_signal(self):
        dcrnn = load_graph_model('DCRNN')
        first = torch.tensor([[0., 1.], [1., 0.]]).to_sparse()
        second = torch.tensor([[2., 0.], [0., 3.]]).to_sparse()
        layer = dcrnn.GCONV(2, 2, [first, second], torch.device('cpu'),
                            input_dim=1, hid_dim=1, output_dim=10)
        with torch.no_grad():
            layer.weight.copy_(torch.eye(10))
            layer.biases.zero_()
        output = layer(torch.tensor([[1., 2.]]), torch.zeros(1, 2)).reshape(2, 10)
        # Channel 0, support 2, first diffusion step is B @ [1, 2].
        torch.testing.assert_close(output[:, 3], torch.tensor([2., 6.]))

    def test_split_windows_do_not_cross_boundaries(self):
        series = np.arange(100 * 3, dtype=np.float32).reshape(100, 3) + 1
        starts = starts_for_split(70, 80, 2)
        self.assertEqual(len(starts), 0)
        starts = starts_for_split(70, 95, 2)
        data = Windows(series, starts, 10, 2)
        x, y = data[-1]
        self.assertEqual(tuple(x.shape), (12, 3, 1))
        self.assertEqual(tuple(y.shape), (1, 3, 1))
        self.assertLessEqual(int(starts[-1] + 12), 94)

    def test_persistence_uses_last_available_reading(self):
        x = torch.tensor([[[[2.]], [[0.]], [[0.]]],
                          [[[0.]], [[0.]], [[0.]]]])
        torch.testing.assert_close(last_available_speed(x),
                                   torch.tensor([[[[2.]]], [[[0.]]]]))

    def test_stgcn_residual_prediction_shape_and_gradient(self):
        stgcn = load_graph_model('STGCN')
        adj = np.array([[0, 1], [1, 0]], dtype=np.float32)
        model = stgcn.STGCN({'input_window': 12, 'output_window': 1, 'device': torch.device('cpu'),
                             'Ks': 2, 'Kt': 3, 'blocks': [[1, 2, 2], [2, 2, 2]],
                             'residual_last_speed': True},
                            {'num_nodes': 2, 'feature_dim': 1, 'output_dim': 1, 'adj_mx': adj})
        x = torch.randn(3, 12, 2, 1, requires_grad=True)
        result = model({'X': x})
        self.assertEqual(tuple(result.shape), (3, 1, 2, 1))
        result.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_dcrnn_dual_walk_prediction_shape_and_gradient(self):
        dcrnn = load_graph_model('DCRNN')
        adj = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
        model = dcrnn.DCRNN({'input_window': 3, 'output_window': 1,
                             'device': torch.device('cpu'), 'num_rnn_layers': 1,
                             'rnn_units': 4, 'max_diffusion_step': 2,
                             'filter_type': 'dual_random_walk', 'residual_last_speed': True},
                            {'num_nodes': 3, 'feature_dim': 1, 'output_dim': 1, 'adj_mx': adj})
        x = torch.randn(2, 3, 3, 1, requires_grad=True)
        result = model({'X': x, 'y': None})
        self.assertEqual(tuple(result.shape), (2, 1, 3, 1))
        result.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_sttn_graph_is_stable_across_batches_and_gradients_flow(self):
        sttn = load_graph_model('STTN')
        adj = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
        model = sttn.STTN({'input_window': 12, 'output_window': 1,
                           'device': torch.device('cpu'), 'embed_dim': 8,
                           'num_layers': 1, 'num_heads': 2,
                           'forward_expansion': 2},
                          {'num_nodes': 3, 'feature_dim': 1,
                           'output_dim': 1, 'adj_mx': adj})
        graph = model.transformer.encoder.layers[0].STransformer
        before = graph.adj_mx.clone()
        x = torch.randn(2, 12, 3, 1, requires_grad=True)
        result = model({'X': x})
        self.assertEqual(tuple(result.shape), (2, 1, 3, 1))
        result.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(result).all())
        torch.testing.assert_close(graph.adj_mx, before)
        model({'X': x.detach()})
        torch.testing.assert_close(graph.adj_mx, before)

    def test_sttn_residual_uses_last_observed_speed(self):
        sttn = load_graph_model('STTN')
        adj = np.array([[0, 1], [1, 0]], dtype=np.float32)
        model = sttn.STTN({'input_window': 12, 'output_window': 1,
                           'device': torch.device('cpu'), 'embed_dim': 8,
                           'num_layers': 1, 'num_heads': 2,
                           'forward_expansion': 2, 'residual_last_speed': True},
                          {'num_nodes': 2, 'feature_dim': 1,
                           'output_dim': 1, 'adj_mx': adj})
        with torch.no_grad():
            model.conv3.weight.zero_()
            model.conv3.bias.zero_()
        x = torch.ones(2, 12, 2, 1)
        x[:, -1] = 0
        result = model({'X': x})
        torch.testing.assert_close(result, torch.ones(2, 1, 2, 1))

    def test_sttn_training_resumes_after_interrupted_epoch(self):
        adj = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
        x = torch.rand(4, 12, 3, 1)
        y = torch.rand(4, 1, 3, 1)
        loader = DataLoader(TensorDataset(x, y), batch_size=2)
        loaders = {'train': loader, 'validation': loader}
        config = {'embed_dim': 8, 'num_layers': 1, 'num_heads': 2,
                  'forward_expansion': 2, 'dropout_rate': .1,
                  'learning_rate': .001, 'residual_last_speed': True}
        evaluate = metr_la_study.evaluate
        calls = 0

        def interrupt_after_first_epoch(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('simulated interruption')
            return evaluate(*args)

        with tempfile.TemporaryDirectory() as directory:
            progress = Path(directory) / 'candidate.progress.pt'
            with patch.object(metr_la_study, 'evaluate', side_effect=interrupt_after_first_epoch):
                with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                    metr_la_study.run_model('STTN', config, adj, loaders, .5, 1., 2, 42, progress)
            self.assertEqual(torch.load(progress, weights_only=True)['completed_epoch'], 1)
            model, best = metr_la_study.run_model('STTN', config, adj, loaders,
                                                   .5, 1., 2, 42, progress)
            self.assertEqual(torch.load(progress, weights_only=True)['completed_epoch'], 2)
            self.assertIn(best['epoch'], (1, 2))
            self.assertEqual(tuple(model({'X': x}).shape), (4, 1, 3, 1))


if __name__ == '__main__':
    unittest.main()
