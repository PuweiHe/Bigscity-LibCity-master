"""Exercise the actual LibCity Seq2Seq module without importing unrelated models."""

import importlib.util
import json
from pathlib import Path

import torch


def load_model():
    path = (
        Path(__file__).resolve().parents[1]
        / "libcity/model/traffic_speed_prediction/Seq2Seq.py"
    )
    spec = importlib.util.spec_from_file_location("portfolio_seq2seq", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Seq2Seq


def inspect(decoder_start="random"):
    torch.manual_seed(42)
    cls = load_model()
    model = cls(
        {
            "input_window": 4,
            "output_window": 2,
            "hidden_size": 8,
            "decoder_start": decoder_start,
        },
        {"num_nodes": 2, "feature_dim": 1, "output_dim": 1},
    )
    model.eval()
    batch = {"X": torch.randn(3, 4, 2, 1), "y": torch.randn(3, 2, 2, 1)}
    with torch.no_grad():
        first, second = model.predict(batch), model.predict(batch)
        try:
            model.predict({"X": batch["X"]})
            unlabeled = True
        except KeyError:
            unlabeled = False
    return {
        "torch": torch.__version__,
        "seed": 42,
        "decoder_start": decoder_start,
        "output_shape": list(first.shape),
        "repeat_max_abs_difference": float((first - second).abs().max()),
        "inference_without_y": unlabeled,
        "scope": "Synthetic forward-pass reproduction, not training accuracy",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decoder-start", choices=["random", "last", "zero"], default="last"
    )
    args = parser.parse_args()
    print(json.dumps(inspect(args.decoder_start), indent=2))
