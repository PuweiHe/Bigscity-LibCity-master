import unittest

import numpy as np
import torch

from traffic_forecasting.data import feature_matrix
from traffic_forecasting.experiment import metrics
from traffic_forecasting.inference import validate_request
from traffic_forecasting.models import ForecastGRU


def request():
    return {
        "observations": [
            {
                "timestamp": f"2024-01-01T00:0{i}:00Z",
                "speed": 80 + i,
                "flow": 10,
                "speed_std": 4,
                "high_code_share": 0.4,
            }
            for i in range(4)
        ]
    }


class PipelineTests(unittest.TestCase):
    def test_schema_and_horizon(self):
        x, time = validate_request(request())
        self.assertEqual(x.shape, (1, 4, 4))
        self.assertEqual(time.minute, 4)
        self.assertEqual(x[0, -1, 0], 83)

    def test_invalid_values(self):
        for field, value in [
            ("speed", float("nan")),
            ("speed", -1),
            ("flow", 0),
            ("flow", 2.5),
            ("high_code_share", 1.1),
            ("speed_std", -1),
            ("flow", True),
        ]:
            payload = request()
            payload["observations"][1][field] = value
            with self.assertRaises(ValueError):
                validate_request(payload)

    def test_invalid_timestamps_and_fields(self):
        for value in ["2024-01-01T00:00:00Z", "2024-01-01T00:01:01Z", None]:
            payload = request()
            payload["observations"][1]["timestamp"] = value
            with self.assertRaises(ValueError):
                validate_request(payload)
        for payload in [{}, {"observations": []}, {**request(), "extra": 1}]:
            with self.assertRaises(ValueError):
                validate_request(payload)

    def test_features_and_validation(self):
        x, _ = validate_request(request())
        self.assertEqual(feature_matrix(x).shape, (1, 16))
        self.assertEqual(feature_matrix(x, True).shape, (1, 21))
        with self.assertRaises(ValueError):
            feature_matrix(np.zeros((1, 3, 4)))

    def test_residual_identity_when_correction_zero(self):
        model = ForecastGRU(
            {
                "hidden_size": 8,
                "decoder_start": "last",
                "residual": True,
                "residual_scale": 0.25,
            }
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        x = torch.randn(5, 4, 4)
        torch.testing.assert_close(model(x), x[:, -1, 0])

    def test_metrics_units(self):
        result = metrics(np.array([10, 20]), np.array([12, 18]))
        self.assertEqual(result, {"mae": 2.0, "rmse": 2.0})
