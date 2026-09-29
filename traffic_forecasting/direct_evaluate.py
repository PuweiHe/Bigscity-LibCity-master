"""Evaluate the validation-selected direct GRU once on the previously seen holdout."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from .data import prepare
from .experiment import SEEDS, metrics, train_gru
from .models import predict_gru


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.artifact.exists() or args.report.exists():
        raise FileExistsError("This retrospective evaluation is write-once")
    torch.set_num_threads(1)
    study = json.loads(args.study.read_text())
    chosen = min(study["candidates"], key=lambda row: row["validation_mae_mean"])
    data, audit = prepare(args.data_root)
    if audit != study["data_audit"]:
        raise ValueError("Source data changed since validation study")
    train = data["split"] == "train"
    valid = data["split"] == "validation"
    test = data["split"] == "test"
    bundles, scores = [], []
    for seed in SEEDS:
        bundle, score = train_gru(
            data["X"][train], data["y"][train],
            data["X"][valid], data["y"][valid], chosen["config"], seed,
        )
        bundles.append(bundle)
        predicted = predict_gru(bundle, data["X"][test])
        scores.append({
            "seed": seed, "validation_mae": score["validation_mae"],
            **metrics(data["y"][test], predicted),
            "by_recording": {
                group: metrics(data["y"][test][data["groups"][test] == group],
                               predicted[data["groups"][test] == group])
                for group in np.unique(data["groups"][test])
            },
        })
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundles, args.artifact)
    report = {
        "scope": "Retrospective comparison on previously reported two-session holdout; exploratory",
        "selection_rule": "Minimum three-seed mean validation MAE among six predefined direct-GRU configurations",
        "study_sha256": hashlib.sha256(args.study.read_bytes()).hexdigest(),
        "configuration": chosen["config"],
        "parameters": chosen["parameters"],
        "validation_mae": chosen["validation_mae_mean"],
        "test_windows": int(test.sum()),
        "test_mae_mean": float(np.mean([row["mae"] for row in scores])),
        "test_rmse_mean": float(np.mean([row["rmse"] for row in scores])),
        "test_mae_seed_std": float(np.std([row["mae"] for row in scores], ddof=1)),
        "seeds": scores,
        "artifact_sha256": hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("configuration", "parameters", "validation_mae", "test_mae_mean", "test_rmse_mean")}, indent=2))


if __name__ == "__main__":
    main()
