"""Validated, reusable model loading and batch-one JSON prediction."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from .data import FEATURES, WINDOW, feature_matrix
from .models import ForecastFNN, ForecastGRU, predict_rf


def validate_request(payload):
    if not isinstance(payload, dict) or set(payload) != {"observations"}:
        raise ValueError("Request must contain only observations")
    rows = payload["observations"]
    if not isinstance(rows, list) or len(rows) != WINDOW:
        raise ValueError("Exactly four one-minute observations are required")
    if any(not isinstance(r, dict) or set(r) != {"timestamp", *FEATURES} for r in rows):
        raise ValueError(
            "Each observation requires timestamp and all four feature fields"
        )
    for row in rows:
        for key in FEATURES:
            if isinstance(row[key], bool) or not isinstance(row[key], (int, float)):
                raise ValueError("Features must be JSON numbers")
    x = np.array([[r[k] for k in FEATURES] for r in rows], dtype=np.float32)
    if not np.isfinite(x).all() or (x[:, 0] < 0).any() or (x[:, 0] > 200).any():
        raise ValueError("Finite speed between 0 and 200 km/h required")
    if (
        (x[:, 1] < 1).any()
        or (x[:, 1] != np.floor(x[:, 1])).any()
        or (x[:, 2] < 0).any()
        or ((x[:, 3] < 0) | (x[:, 3] > 1)).any()
    ):
        raise ValueError("Invalid vehicle count, dispersion or category share")
    if any(not isinstance(r["timestamp"], str) for r in rows):
        raise ValueError("Timestamps must be ISO strings")
    timestamps = pd.to_datetime(
        [r["timestamp"] for r in rows], errors="raise", utc=True
    )
    if timestamps.isna().any() or not np.all(
        np.diff(timestamps.asi8) == 60_000_000_000
    ):
        raise ValueError("Timestamps must be increasing consecutive minutes")
    if not (timestamps == timestamps.floor("min")).all():
        raise ValueError("Timestamps must identify minute boundaries")
    return x[None], timestamps[-1] + pd.Timedelta(minutes=1)


class Predictor:
    """Load one trusted local checkpoint once; reuse for multiple requests."""

    def __init__(self, artifact, family, seed=42):
        if family not in {"rf", "gru", "fnn"}:
            raise ValueError("family must be rf, gru or fnn")
        bundles = joblib.load(artifact)  # Never load untrusted uploaded pickle files.
        matches = [b for b in bundles if b["seed"] == seed]
        if len(matches) != 1:
            raise ValueError("Requested seed is absent or ambiguous")
        self.bundle, self.family = matches[0], family
        self.model = None
        if family in {"gru", "fnn"}:
            if self.bundle["config"].get("decoder_start") == "random":
                raise ValueError("Serving requires deterministic decoding")
            model_class = ForecastGRU if family == "gru" else ForecastFNN
            self.model = model_class(self.bundle["config"])
            self.model.load_state_dict(self.bundle["state"])
            self.model.eval()

    def predict_array(self, x):
        feature_matrix(x)
        if self.family == "rf":
            return predict_rf(self.bundle, x)
        b = self.bundle
        normalized = torch.tensor((x - b["mean"]) / b["std"], dtype=torch.float32)
        with torch.inference_mode():
            return self.model(normalized).numpy() * b["std"][0] + b["mean"][0]

    def predict(self, payload):
        x, time = validate_request(payload)
        prediction = float(self.predict_array(x)[0])
        if not np.isfinite(prediction):
            raise ValueError("Model produced a nonfinite forecast")
        return {
            "prediction_time": time.isoformat(),
            "mean_speed_kmh": prediction,
            "horizon_minutes": 1,
            "model_family": self.family,
            "seed": self.bundle["seed"],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--family", choices=["rf", "gru", "fnn"], required=True)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    predictor = Predictor(args.artifact, args.family)
    print(json.dumps(predictor.predict(json.loads(args.request.read_text())), indent=2))


if __name__ == "__main__":
    main()
