import importlib.util
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from traffic_forecasting.metr_la_study import Windows, starts_for_split


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


if __name__ == '__main__':
    unittest.main()
