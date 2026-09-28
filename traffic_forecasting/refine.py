"""Second validation-only search; preserves the first search record."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor

from .data import feature_matrix, prepare
from .experiment import SEEDS, metrics, train_gru
from .models import predict_rf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if (args.previous / "test_report.json").exists():
        raise ValueError("Refinement must precede test evaluation")
    if (args.output / "selection.json").exists():
        raise FileExistsError("Refined selection already exists")
    selection = json.loads((args.previous / "selection.json").read_text())
    data, audit = prepare(args.data_root)
    if (
        hashlib.sha256(json.dumps(audit, sort_keys=True).encode()).hexdigest()
        != selection["data_digest"]
    ):
        raise ValueError("Data changed")
    t, v = data["split"] == "train", data["split"] == "validation"
    x, y, vx, vy = data["X"][t], data["y"][t], data["X"][v], data["y"][v]
    args.output.mkdir(parents=True, exist_ok=True)
    for name in selection["checkpoints"]:
        shutil.copy2(args.previous / f"{name}.joblib", args.output / f"{name}.joblib")
    rows = selection["candidates"]
    new = {}
    for index, (leaf, depth, scale) in enumerate(
        (leaf_size, depth_limit, factor)
        for leaf_size in [3, 10]
        for depth_limit in [3, 6]
        for factor in [0.25, 0.5]
    ):
        config = {
            "n_estimators": 200,
            "max_depth": depth,
            "min_samples_leaf": leaf,
            "engineered": True,
            "residual": True,
            "anchor": "mean",
            "residual_scale": scale,
        }
        scores, models = [], []
        for seed in SEEDS:
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=depth,
                min_samples_leaf=leaf,
                random_state=seed,
                n_jobs=1,
            )
            model.fit(feature_matrix(x, True), y - x[:, :, 0].mean(1))
            bundle = {"config": config, "model": model, "seed": seed}
            scores.append({"seed": seed, **metrics(vy, predict_rf(bundle, vx))})
            models.append(bundle)
        name = f"rf_mean_{index}"
        new[name] = models
        rows.append(
            {
                "id": name,
                "family": "rf",
                "baseline": False,
                "config": config,
                "seeds": scores,
                "validation_mae": float(np.mean([s["mae"] for s in scores])),
            }
        )
        print(name, rows[-1]["validation_mae"], flush=True)
    for index, (hidden, scale) in enumerate(
        (h, s) for h in [8, 16] for s in [0.1, 0.25]
    ):
        config = {
            "hidden_size": hidden,
            "decoder_start": "last",
            "residual": True,
            "anchor": "mean",
            "residual_scale": scale,
            "loss": "huber",
            "weight_decay": 0.01,
        }
        scores, models = [], []
        for seed in SEEDS:
            model, score = train_gru(x, y, vx, vy, config, seed)
            models.append(model)
            scores.append(score)
        name = f"gru_mean_{index}"
        new[name] = models
        rows.append(
            {
                "id": name,
                "family": "gru",
                "baseline": False,
                "config": config,
                "seeds": scores,
                "validation_mae": float(np.mean([s["validation_mae"] for s in scores])),
            }
        )
        print(name, rows[-1]["validation_mae"], flush=True)
    # Include the baseline so unsuccessful tuning cannot force a worse deployment choice.
    selected = {
        family: min(
            [r for r in rows if r["family"] == family],
            key=lambda r: r["validation_mae"],
        )["id"]
        for family in ["rf", "gru"]
    }
    for name in selected.values():
        if name in new:
            joblib.dump(new[name], args.output / f"{name}.joblib")
    selection.update(
        selected=selected,
        candidates=rows,
        refinement_reason="Validation favored local mean over persistence; smooth residual anchor with stronger regularization",
        previous_selection_sha256=hashlib.sha256(
            (args.previous / "selection.json").read_bytes()
        ).hexdigest(),
        checkpoints={
            name: hashlib.sha256(
                (args.output / f"{name}.joblib").read_bytes()
            ).hexdigest()
            for name in set(["rf_0", "gru_0", *selected.values()])
        },
    )
    (args.output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    shutil.copy2(args.previous / "data_audit.json", args.output / "data_audit.json")
    print("Final validation selection frozen", selected, flush=True)


if __name__ == "__main__":
    main()
