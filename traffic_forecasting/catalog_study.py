"""Compare LibCity catalog SVR and FNN on the established speed task.

This is a retrospective extension: earlier work already opened the test sessions.
Selection uses only the original train and validation partitions.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .data import feature_matrix, prepare
from .experiment import SEEDS, metrics
from .models import ForecastFNN


def audit_digest(audit):
    return hashlib.sha256(json.dumps(audit, sort_keys=True).encode()).hexdigest()


def train_fnn(x, y, vx, vy, config, seed):
    torch.manual_seed(seed)
    mean = x.reshape(-1, 4).mean(axis=0)
    std = np.maximum(x.reshape(-1, 4).std(axis=0), 1e-6)
    tx = torch.tensor((x - mean) / std, dtype=torch.float32)
    ty = torch.tensor((y - mean[0]) / std[0], dtype=torch.float32)
    tvx = torch.tensor((vx - mean) / std, dtype=torch.float32)
    model = ForecastFNN(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003,
                                 weight_decay=config.get("weight_decay", 0.0))
    criterion = (torch.nn.SmoothL1Loss() if config.get("loss") == "huber"
                 else torch.nn.MSELoss())
    best = float("inf")
    best_state = None
    stale = 0
    best_epoch = 0
    for epoch in range(120):
        model.train()
        for ids in torch.randperm(len(tx)).split(32):
            optimizer.zero_grad()
            loss = criterion(model(tx[ids]), ty[ids])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite FNN training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            prediction = model(tvx).numpy() * std[0] + mean[0]
        score = metrics(vy, prediction)["mae"]
        if score < best - 1e-6:
            best = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
        if stale >= 20:
            break
    return {
        "config": config, "state": best_state, "mean": mean, "std": std,
        "seed": seed,
    }, {"seed": seed, "validation_mae": best, "best_epoch": best_epoch}


def predict_fnn(bundle, x):
    model = ForecastFNN(bundle["config"])
    model.load_state_dict(bundle["state"])
    model.eval()
    with torch.inference_mode():
        normalized = torch.tensor((x - bundle["mean"]) / bundle["std"],
                                  dtype=torch.float32)
        return model(normalized).numpy() * bundle["std"][0] + bundle["mean"][0]


def svrs():
    # The RBF default mirrors test/test_SVR.py; each candidate predicts speed only.
    yield {"kernel": "rbf", "C": 1.0, "epsilon": 0.1, "baseline": True}
    for c in (0.1, 1.0, 10.0, 100.0):
        for epsilon in (1.0, 3.0):
            yield {"kernel": "linear", "C": c, "epsilon": epsilon,
                   "baseline": False}
    for c in (0.1, 1.0):
        yield {"kernel": "rbf", "C": c, "epsilon": 3.0,
               "baseline": False}


def fnns():
    yield {"hidden_size": 128, "loss": "mse", "weight_decay": 0.0,
           "baseline": True}
    for hidden in (8, 16, 32):
        for scale in (0.1, 0.25):
            yield {"hidden_size": hidden, "hidden_size_2": 8,
                   "residual_anchor": "mean", "residual_scale": scale,
                   "loss": "huber", "weight_decay": 0.01, "baseline": False}


def select(data, audit, output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    train = data["split"] == "train"
    valid = data["split"] == "validation"
    x, y = data["X"][train], data["y"][train]
    vx, vy = data["X"][valid], data["y"][valid]
    rows = []
    saved = {}
    for index, config in enumerate(svrs()):
        model = make_pipeline(
            StandardScaler(),
            SVR(kernel=config["kernel"], C=config["C"], epsilon=config["epsilon"]),
        )
        model.fit(feature_matrix(x), y)
        score = metrics(vy, model.predict(feature_matrix(vx)))
        name = f"svr_{index}"
        rows.append({"id": name, "family": "svr", "config": config,
                     "validation": score})
        saved[name] = model
        print(name, score["mae"], flush=True)
    for index, config in enumerate(fnns()):
        bundles = []
        scores = []
        for seed in SEEDS:
            bundle, score = train_fnn(x, y, vx, vy, config, seed)
            bundles.append(bundle)
            scores.append(score)
        name = f"fnn_{index}"
        rows.append({
            "id": name, "family": "fnn", "config": config,
            "parameters": sum(p.numel() for p in ForecastFNN(config).parameters()),
            "validation": {"mae": float(np.mean([s["validation_mae"] for s in scores]))},
            "seeds": scores,
        })
        saved[name] = bundles
        print(name, rows[-1]["validation"]["mae"], flush=True)
    selected = {
        family: min((row for row in rows if row["family"] == family),
                    key=lambda row: row["validation"]["mae"])["id"]
        for family in ("svr", "fnn")
    }
    checkpoints = {}
    for name in ("svr_0", "fnn_0", *selected.values()):
        path = output / f"{name}.joblib"
        joblib.dump(saved[name], path)
        checkpoints[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "selection.json").write_text(json.dumps({
        "scope": "Retrospective LibCity model-list extension; original holdout already opened",
        "task": "Predict next-minute mean speed from four prior complete minutes",
        "data_digest": audit_digest(audit),
        "train_windows": int(train.sum()), "validation_windows": int(valid.sum()),
        "seeds": SEEDS, "candidates": rows, "selected": selected,
        "checkpoints": checkpoints,
    }, indent=2) + "\n")


def evaluate(data, audit, output):
    report_path = output / "test_report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    selection = json.loads((output / "selection.json").read_text())
    if audit_digest(audit) != selection["data_digest"]:
        raise ValueError("Data changed since model selection")
    test = data["split"] == "test"
    x, y, groups = data["X"][test], data["y"][test], data["groups"][test]
    report = {
        "scope": "Exploratory evaluation on two previously reported recording sessions",
        "test_windows": len(y), "models": {},
    }
    names = list(dict.fromkeys(("svr_0", "fnn_0", *selection["selected"].values())))
    for name in names:
        path = output / f"{name}.joblib"
        if hashlib.sha256(path.read_bytes()).hexdigest() != selection["checkpoints"][name]:
            raise ValueError("Checkpoint changed after validation selection")
        artifact = joblib.load(path)  # Locally generated trusted artifact only.
        bundles = [artifact] if name.startswith("svr") else artifact
        scores = []
        for bundle in bundles:
            prediction = (bundle.predict(feature_matrix(x)) if name.startswith("svr")
                          else predict_fnn(bundle, x))
            scores.append({
                **({"seed": bundle["seed"]} if name.startswith("fnn") else {}),
                **metrics(y, prediction),
                "by_recording": {
                    group: metrics(y[groups == group], prediction[groups == group])
                    for group in np.unique(groups)
                },
            })
        report["models"][name] = {
            "mae": float(np.mean([score["mae"] for score in scores])),
            "rmse": float(np.mean([score["rmse"] for score in scores])),
            "runs": scores,
        }
    report["selected"] = selection["selected"]
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({n: v["mae"] for n, v in report["models"].items()}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("select", "evaluate"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    data, audit = prepare(args.data_root)
    if args.stage == "select":
        select(data, audit, args.output)
    else:
        evaluate(data, audit, args.output)


if __name__ == "__main__":
    main()
