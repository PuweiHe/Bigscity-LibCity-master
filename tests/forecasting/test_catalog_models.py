import unittest

import torch

from traffic_forecasting.models import ForecastFNN


class CatalogModelTests(unittest.TestCase):
    def test_libcity_fnn_baseline_shape_and_gradients(self):
        model = ForecastFNN({"hidden_size": 128})
        x = torch.randn(5, 4, 4, requires_grad=True)
        prediction = model(x)
        self.assertEqual(prediction.shape, (5,))
        prediction.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_two_layer_residual_fnn_zero_correction_is_moving_mean(self):
        model = ForecastFNN({
            "hidden_size": 16, "hidden_size_2": 8,
            "residual_anchor": "mean", "residual_scale": 0.25,
        })
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        x = torch.randn(5, 4, 4)
        torch.testing.assert_close(model(x), x[:, :, 0].mean(dim=1))

    def test_invalid_residual_anchor(self):
        with self.assertRaises(ValueError):
            ForecastFNN({"hidden_size": 8, "residual_anchor": "future"})


if __name__ == "__main__":
    unittest.main()
