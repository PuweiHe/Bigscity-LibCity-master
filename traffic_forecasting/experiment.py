"""Separate validation tuning and locked-test evaluation commands."""

import argparse
import copy
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor

from .data import feature_matrix, prepare
from .models import ForecastGRU, predict_gru, predict_rf

SEEDS = [17, 42, 73]


def metrics(y, predicted):
    error = np.asarray(predicted) - y
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
    }


def train_gru(x, y, vx, vy, config, seed):
    torch.manual_seed(seed)
    mean, std = x.reshape(-1, 4).mean(0), x.reshape(-1, 4).std(0)
    std = np.maximum(std, 1e-6)
    tx = torch.tensor((x - mean) / std)
    ty = torch.tensor((y - mean[0]) / std[0])
    tvx = torch.tensor((vx - mean) / std)
    model = ForecastGRU(config)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.003, weight_decay=config.get("weight_decay", 0)
    )
    criterion = (
        torch.nn.SmoothL1Loss() if config.get("loss") == "huber" else torch.nn.MSELoss()
    )
    best, best_state, best_epoch, stale, history = float("inf"), None, 0, 0, []
    for epoch in range(120):
        model.train()
        order = torch.randperm(len(x))
        for ids in order.split(32):
            optimizer.zero_grad()
            loss = criterion(model(tx[ids]), ty[ids])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        # Isolate evaluation randomness so validation does not alter training RNG.
        with torch.random.fork_rng(), torch.inference_mode():
            torch.manual_seed(seed + 1000)
            p = model(tvx).numpy() * std[0] + mean[0]
        score = metrics(vy, p)["mae"]
        history.append(score)
        if score < best - 1e-6:
            best, best_epoch, stale = score, epoch + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= 20:
            break
    return {
        "config": config,
        "state": best_state,
        "mean": mean,
        "std": std,
        "seed": seed,
    }, {
        "seed": seed,
        "validation_mae": best,
        "best_epoch": best_epoch,
        "epochs": len(history),
        "validation_history": history,
    }


def tune(data_root, output):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "selection.json").exists():
        raise FileExistsError(
            "Selection already frozen; use a new experiment directory"
        )
    data, audit = prepare(data_root)
    train, valid = data["split"] == "train", data["split"] == "validation"
    x, y, vx, vy = (
        data["X"][train],
        data["y"][train],
        data["X"][valid],
        data["y"][valid],
    )
    rows, bundles = [], {}
    baseline_rf = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 1,
        "engineered": False,
        "residual": False,
    }
    rf_configs = [baseline_rf] + [
        {
            "n_estimators": 200,
            "max_depth": depth,
            "min_samples_leaf": leaf,
            "engineered": True,
            "residual": residual,
        }
        for leaf in [2, 5]
        for depth in [4, 8]
        for residual in [False, True]
    ]
    for index, config in enumerate(rf_configs):
        scores, models = [], []
        for seed in SEEDS:
            params = {
                k: config[k] for k in ["n_estimators", "max_depth", "min_samples_leaf"]
            }
            model = RandomForestRegressor(**params, random_state=seed, n_jobs=1)
            target = y - x[:, -1, 0] if config["residual"] else y
            model.fit(feature_matrix(x, config["engineered"]), target)
            bundle = {"model": model, "config": config, "seed": seed}
            scores.append({"seed": seed, **metrics(vy, predict_rf(bundle, vx))})
            models.append(bundle)
        name = f"rf_{index}"
        bundles[name] = models
        row = {
            "id": name,
            "family": "rf",
            "baseline": index == 0,
            "config": config,
            "seeds": scores,
            "validation_mae": float(np.mean([r["mae"] for r in scores])),
        }
        rows.append(row)
        print(name, row["validation_mae"], flush=True)
    baseline_gru = {
        "hidden_size": 64,
        "decoder_start": "random",
        "residual": False,
        "loss": "mse",
        "weight_decay": 0,
    }
    gru_configs = [baseline_gru] + [
        {
            "hidden_size": h,
            "decoder_start": "last",
            "residual": True,
            "residual_scale": s,
            "loss": "huber",
            "weight_decay": 0.001,
        }
        for h in [16, 32]
        for s in [0.25, 1.0]
    ]
    for index, config in enumerate(gru_configs):
        scores, models = [], []
        for seed in SEEDS:
            bundle, score = train_gru(x, y, vx, vy, config, seed)
            scores.append(score)
            models.append(bundle)
        name = f"gru_{index}"
        bundles[name] = models
        row = {
            "id": name,
            "family": "gru",
            "baseline": index == 0,
            "config": config,
            "seeds": scores,
            "validation_mae": float(np.mean([r["validation_mae"] for r in scores])),
        }
        rows.append(row)
        print(name, row["validation_mae"], flush=True)
    selection = {
        family: min(
            [r for r in rows if r["family"] == family and not r["baseline"]],
            key=lambda r: r["validation_mae"],
        )["id"]
        for family in ["rf", "gru"]
    }
    for name in ["rf_0", "gru_0", *selection.values()]:
        joblib.dump(bundles[name], output / f"{name}.joblib")
    (output / "data_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n"
    )
    payload = {
        "selected": selection,
        "candidates": rows,
        "seeds": SEEDS,
        "data_digest": hashlib.sha256(
            json.dumps(audit, sort_keys=True).encode()
        ).hexdigest(),
        "checkpoints": {
            name: hashlib.sha256((output / f"{name}.joblib").read_bytes()).hexdigest()
            for name in ["rf_0", "gru_0", *selection.values()]
        },
    }
    (output / "selection.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("Selection frozen:", selection, flush=True)


def evaluate(data_root, output):
    if (output / "test_report.json").exists():
        raise FileExistsError(
            "Test already evaluated; report is immutable by this command"
        )
    selection = json.loads((output / "selection.json").read_text())
    data, audit = prepare(data_root)
    digest = hashlib.sha256(json.dumps(audit, sort_keys=True).encode()).hexdigest()
    if digest != selection["data_digest"]:
        raise ValueError("Dataset changed after selection")
    mask = data["split"] == "test"
    x, y, groups = data["X"][mask], data["y"][mask], data["groups"][mask]
    report = {
        "scope": "Chronological holdout on two later recording sessions; offline small-data benchmark",
        "test_windows": len(y),
        "seeds": SEEDS,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "models": {},
    }
    predictions = {"persistence": x[:, -1, 0], "moving_average": x[:, :, 0].mean(1)}
    for name, p in predictions.items():
        report["models"][name] = {
            **metrics(y, p),
            "by_recording": {
                g: metrics(y[groups == g], p[groups == g]) for g in np.unique(groups)
            },
        }
    for name in ["rf_0", "gru_0", *selection["selected"].values()]:
        path = output / f"{name}.joblib"
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != selection["checkpoints"][name]
        ):
            raise ValueError("Checkpoint changed after selection")
        models = joblib.load(path)  # Only locally generated, trusted artifacts.
        predictor = predict_rf if name.startswith("rf") else predict_gru
        scores = []
        for bundle in models:
            p = predictor(bundle, x)
            predictions[f"{name}_seed{bundle['seed']}"] = p
            scores.append(
                {
                    "seed": bundle["seed"],
                    **metrics(y, p),
                    "by_recording": {
                        g: metrics(y[groups == g], p[groups == g])
                        for g in np.unique(groups)
                    },
                }
            )
        report["models"][name] = {
            "mae": float(np.mean([r["mae"] for r in scores])),
            "rmse": float(np.mean([r["rmse"] for r in scores])),
            "mae_seed_std": float(np.std([r["mae"] for r in scores], ddof=1)),
            "seeds": scores,
        }
    for family, name in selection["selected"].items():
        tuned = report["models"][name]["mae"]
        base = report["models"][family + "_0"]["mae"]
        persistence = report["models"]["persistence"]["mae"]
        report["models"][name]["mae_reduction_vs_untuned_percent"] = (
            100 * (base - tuned) / base
        )
        report["models"][name]["mae_reduction_vs_persistence_percent"] = (
            100 * (persistence - tuned) / persistence
        )
    np.savez(output / "test_predictions.npz", y=y, groups=groups, **predictions)
    (output / "test_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["tune", "evaluate"])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    (tune if args.stage == "tune" else evaluate)(args.data_root, args.output)


if __name__ == "__main__":
    main()
