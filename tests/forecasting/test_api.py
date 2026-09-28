import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from fastapi.testclient import TestClient

from traffic_forecasting.api import create_app
from traffic_forecasting.inference import Predictor
from traffic_forecasting.models import ForecastGRU

ROOT = Path(__file__).resolve().parents[2]


class APITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "model.joblib"
        config = {
            "hidden_size": 8,
            "decoder_start": "last",
            "residual": True,
            "anchor": "mean",
            "residual_scale": 0.25,
        }
        model = ForecastGRU(config)
        joblib.dump(
            [
                {
                    "seed": 42,
                    "config": config,
                    "state": model.state_dict(),
                    "mean": np.array([80, 10, 5, 0.3], dtype=np.float32),
                    "std": np.array([10, 5, 2, 0.1], dtype=np.float32),
                }
            ],
            self.path,
        )
        self.payload = json.loads(
            (ROOT / "docs/forecasting/example_request.json").read_text()
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_loaded_once_and_repeatable_prediction(self):
        with patch("traffic_forecasting.api.Predictor", wraps=Predictor) as constructor:
            with TestClient(create_app(self.path, "gru")) as client:
                self.assertEqual(client.get("/health").status_code, 200)
                first = client.post("/predict", json=self.payload)
                second = client.post("/predict", json=self.payload)
                self.assertEqual(first.status_code, 200)
                self.assertEqual(first.json(), second.json())
                self.assertEqual(first.json()["horizon_minutes"], 1)
            self.assertEqual(constructor.call_count, 1)

    def test_rejects_invalid_requests(self):
        with TestClient(create_app(self.path, "gru")) as client:
            self.assertEqual(client.post("/predict", json={}).status_code, 422)
            self.payload["observations"][1]["timestamp"] = self.payload["observations"][
                0
            ]["timestamp"]
            self.assertEqual(
                client.post("/predict", json=self.payload).status_code, 422
            )
            self.payload["observations"][1]["flow"] = -1
            self.assertEqual(
                client.post("/predict", json=self.payload).status_code, 422
            )

    def test_array_cli_api_consistency(self):
        p = Predictor(self.path, "gru")
        with TestClient(create_app(self.path, "gru")) as client:
            result = client.post("/predict", json=self.payload).json()
        self.assertEqual(result, p.predict(self.payload))
