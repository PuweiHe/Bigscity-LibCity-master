"""Reusable tabular and recurrent forecasting estimators."""

import importlib.util
from pathlib import Path

import torch
from torch import nn

from .data import feature_matrix


def seq2seq_class():
    path = (
        Path(__file__).resolve().parents[1]
        / "libcity/model/traffic_speed_prediction/Seq2Seq.py"
    )
    spec = importlib.util.spec_from_file_location("forecast_seq2seq", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Seq2Seq


class ForecastGRU(nn.Module):
    """Opt-in residual correction around the last observed standardized speed."""

    def __init__(self, config):
        super().__init__()
        self.residual = config.get("residual", False)
        self.scale = config.get("residual_scale", 1.0)
        self.anchor = config.get("anchor", "last")
        self.architecture = config.get("architecture", "seq2seq")
        if self.architecture == "direct":
            self.model = DirectGRU(config["hidden_size"])
        elif self.architecture == "seq2seq":
            self.model = seq2seq_class()(
                {
                    "input_window": 4,
                    "output_window": 1,
                    "hidden_size": config["hidden_size"],
                    "rnn_type": "GRU",
                    "decoder_start": config.get("decoder_start", "random"),
                },
                {"num_nodes": 1, "feature_dim": 4, "output_dim": 1},
            )
        else:
            raise ValueError(f"Unknown GRU architecture: {self.architecture}")

    def forward(self, x):
        correction = (
            self.model(x)
            if self.architecture == "direct"
            else self.model({"X": x.unsqueeze(2)})[:, 0, 0, 0]
        )
        anchor = x[:, :, 0].mean(1) if self.anchor == "mean" else x[:, -1, 0]
        return anchor + self.scale * correction if self.residual else correction


class DirectGRU(nn.Module):
    """A one-step head without the redundant autoregressive decoder."""

    def __init__(self, hidden_size):
        super().__init__()
        self.encoder = nn.GRU(input_size=4, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, hidden = self.encoder(x)
        return self.head(hidden[-1]).squeeze(-1)


def predict_rf(bundle, x):
    result = bundle["model"].predict(feature_matrix(x, bundle["config"]["engineered"]))
    anchor = (
        x[:, :, 0].mean(1) if bundle["config"].get("anchor") == "mean" else x[:, -1, 0]
    )
    return (
        anchor + bundle["config"].get("residual_scale", 1.0) * result
        if bundle["config"]["residual"]
        else result
    )


def predict_gru(bundle, x):
    feature_matrix(x)  # Shared public input contract.
    model = ForecastGRU(bundle["config"])
    model.load_state_dict(bundle["state"])
    model.eval()
    x = torch.tensor((x - bundle["mean"]) / bundle["std"], dtype=torch.float32)
    torch.manual_seed(bundle["seed"] + 1000)
    with torch.inference_mode():
        return model(x).numpy() * bundle["std"][0] + bundle["mean"][0]
