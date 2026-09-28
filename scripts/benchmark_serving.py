"""Warm CPU inference benchmark; excludes model loading and HTTP transport."""

import json
import platform
from pathlib import Path
from time import perf_counter_ns

import joblib
import numpy as np
import torch

from traffic_forecasting.inference import Predictor, validate_request
from traffic_forecasting.models import ForecastGRU

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    payload = json.loads((ROOT / "docs/forecasting/example_request.json").read_text())
    x, _ = validate_request(payload)
    report = {
        "device": "CPU",
        "threads": 1,
        "batch_size": 1,
        "warmup": 50,
        "measured_calls": 1000,
        "scope": "Warm predictor call including normalization; excludes loading, HTTP and validation",
        "platform": platform.platform(),
        "models": {},
    }
    for name in ["gru_0", "gru_mean_1", "rf_mean_4"]:
        artifact = ROOT / "artifacts/forecasting" / f"{name}.joblib"
        if name == "gru_0":
            bundle = next(b for b in joblib.load(artifact) if b["seed"] == 42)
            model = ForecastGRU(bundle["config"])
            model.load_state_dict(bundle["state"])
            model.eval()

            def predict(bundle=bundle, model=model):
                tx = torch.tensor(
                    (x - bundle["mean"]) / bundle["std"], dtype=torch.float32
                )
                with torch.inference_mode():
                    return model(tx).numpy() * bundle["std"][0] + bundle["mean"][0]

            params = sum(p.numel() for p in model.parameters())
        else:
            engine = Predictor(artifact, "rf" if name.startswith("rf") else "gru")

            def predict(engine=engine):
                return engine.predict_array(x)

            params = (
                sum(p.numel() for p in engine.model.parameters())
                if engine.model
                else None
            )
        for _ in range(50):
            predict()
        samples = []
        for _ in range(1000):
            start = perf_counter_ns()
            predict()
            samples.append((perf_counter_ns() - start) / 1e6)
        report["models"][name] = {
            "median_ms": float(np.median(samples)),
            "p95_ms": float(np.quantile(samples, 0.95)),
            "parameters": params,
            "artifact_bytes_three_seeds": artifact.stat().st_size,
        }
    base = report["models"]["gru_0"]
    tuned = report["models"]["gru_mean_1"]
    report["gru_parameter_reduction_percent"] = 100 * (
        1 - tuned["parameters"] / base["parameters"]
    )
    report["gru_median_latency_reduction_percent"] = 100 * (
        1 - tuned["median_ms"] / base["median_ms"]
    )
    (ROOT / "docs/forecasting/serving_benchmark.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
