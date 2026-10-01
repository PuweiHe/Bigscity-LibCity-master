import unittest

import torch

from traffic_forecasting import multihorizon_study as study
from traffic_forecasting import multihorizon_mps as mps


class MPSAdapterTests(unittest.TestCase):
    def test_import_does_not_change_cpu_study(self):
        self.assertIsNot(study.build_model, mps.build_model)
        self.assertNotIn('traffic_forecasting/multihorizon_mps.py', study.CODE_FILES)

    def test_flattened_linear_matches_high_rank_linear_and_gradient(self):
        torch.manual_seed(7)
        reference = torch.nn.Linear(4, 3)
        candidate = torch.nn.Linear(4, 3)
        candidate.load_state_dict(reference.state_dict())
        x_ref = torch.randn(2, 5, 3, 4, 4, requires_grad=True)
        x_new = x_ref.detach().clone().requires_grad_()
        expected = reference(x_ref)
        actual = mps.linear_2d(candidate, x_new)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected.square().sum().backward()
        actual.square().sum().backward()
        torch.testing.assert_close(x_new.grad, x_ref.grad)
        torch.testing.assert_close(candidate.weight.grad, reference.weight.grad)
        torch.testing.assert_close(candidate.bias.grad, reference.bias.grad)


if __name__ == '__main__':
    unittest.main()
