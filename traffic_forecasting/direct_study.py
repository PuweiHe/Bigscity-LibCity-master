"""Retrospective one-step architecture ablation on the original validation split.

The original test set has already been reported, so this command never evaluates it.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import prepare
from .experiment import SEEDS, train_gru
from .models import ForecastGRU


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    data, audit = prepare(args.data_root)
    train = data["split"] == "train"
    valid = data["split"] == "validation"
    x, y = data["X"][train], data["y"][train]
    vx, vy = data["X"][valid], data["y"][valid]
    rows = []
    # Same features, train/validation split, optimizer and seeds as the original study.
    for hidden in (8, 16, 32):
        for scale in (0.1, 0.25):
            config = {
                "architecture": "direct",
                "hidden_size": hidden,
                "residual": True,
                "anchor": "mean",
                "residual_scale": scale,
                "loss": "huber",
                "weight_decay": 0.01,
            }
            scores = []
            for seed in SEEDS:
                _, score = train_gru(x, y, vx, vy, config, seed)
                scores.append({k: score[k] for k in ("seed", "validation_mae", "best_epoch", "epochs")})
            row = {
                "config": config,
                "parameters": sum(p.numel() for p in ForecastGRU(config).parameters()),
                "validation_mae_mean": float(np.mean([s["validation_mae"] for s in scores])),
                "validation_mae_std": float(np.std([s["validation_mae"] for s in scores], ddof=1)),
                "seeds": scores,
            }
            rows.append(row)
            print(hidden, scale, row["validation_mae_mean"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "scope": "Validation-only retrospective architecture ablation; original test was previously inspected",
        "data_audit": audit,
        "candidates": rows,
    }, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
