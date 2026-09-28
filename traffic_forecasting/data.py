"""Prepare minute features directly from source Excel recordings."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["speed", "flow", "speed_std", "high_code_share"]
WINDOW = 4


def prepare(root):
    root = Path(root)
    paths = sorted((root / "施工前/data_11").glob("*.xlsx")) + sorted(
        (root / "5.28施工期/data_5_28").glob("*.xlsx")
    )
    if len(paths) != 13:
        raise ValueError(f"Expected 13 top-level recording files, got {len(paths)}")
    recordings, audit = [], []
    seen = set()
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            raise ValueError("Duplicate source file content")
        seen.add(digest)
        raw = pd.read_excel(path)
        if raw.shape[1] != 11 or raw.columns[0] != "时间":
            raise ValueError(f"Unexpected schema in {path.name}")
        aliases = {"速度": "速度 (km/h)", "车头时距（s）": "车头时距(s)"}
        frame = raw.rename(columns=aliases).drop_duplicates().copy()
        time = pd.to_datetime(frame["时间"], errors="raise")
        speed = pd.to_numeric(frame["速度 (km/h)"], errors="raise").abs()
        # Signed speeds encode direction. Magnitude is the target of this task.
        if time.isna().any() or not np.isfinite(speed).all():
            raise ValueError(f"Invalid sensor record in {path.name}")
        outliers = speed > 200
        speed = speed.mask(outliers)
        features = pd.DataFrame(
            {"time": time, "speed": speed, "heavy": pd.to_numeric(frame["车型"]) >= 7}
        )
        features["minute"] = features["time"].dt.floor("min")
        grouped = features.groupby("minute").agg(
            speed=("speed", "mean"),
            flow=("speed", "size"),
            speed_std=("speed", "std"),
            high_code_share=("heavy", "mean"),
        )
        # Boundary bins cover partial minutes; drop them before modeling.
        grouped = grouped.iloc[1:-1].copy()
        grouped["speed_std"] = grouped["speed_std"].fillna(0)
        if (
            len(grouped) > 1
            and not grouped.index.to_series()
            .diff()
            .iloc[1:]
            .eq(pd.Timedelta(minutes=1))
            .all()
        ):
            raise ValueError(
                f"Missing observation interval in {path.name}; explicit outage policy required"
            )
        if not np.isfinite(grouped[FEATURES].to_numpy()).all():
            raise ValueError("Minute with no valid speed requires explicit handling")
        recordings.append((time.min(), grouped))
        audit.append(
            {
                "source": str(path.relative_to(root)),
                "sha256": digest,
                "raw_rows": len(raw),
                "duplicates_removed": len(raw) - len(frame),
                "speed_outliers_excluded": int(outliers.sum()),
                "complete_minutes": len(grouped),
                "start": str(time.min()),
                "end": str(time.max()),
            }
        )
    order = sorted(range(len(recordings)), key=lambda i: recordings[i][0])
    X, y, groups, times, splits, manifest = [], [], [], [], [], []
    for rank, original in enumerate(order):
        group = f"recording_{rank:02d}"
        part = "train" if rank < 9 else ("validation" if rank < 11 else "test")
        _, frame = recordings[original]
        values = frame[FEATURES].to_numpy(dtype=np.float32)
        count = max(0, len(frame) - WINDOW)
        manifest.append(
            {**audit[original], "recording": group, "split": part, "windows": count}
        )
        for i in range(WINDOW, len(frame)):
            X.append(values[i - WINDOW : i])
            y.append(values[i, 0])
            groups.append(group)
            times.append(str(frame.index[i]))
            splits.append(part)
    result = {
        "X": np.asarray(X, dtype=np.float32),
        "y": np.asarray(y, dtype=np.float32),
        "groups": np.asarray(groups),
        "times": np.asarray(times),
        "split": np.asarray(splits),
    }
    for earlier, later in [("train", "validation"), ("validation", "test")]:
        if max(result["times"][result["split"] == earlier]) >= min(
            result["times"][result["split"] == later]
        ):
            raise ValueError("Chronological split ordering violated")
    return result, {
        "files": manifest,
        "features": FEATURES,
        "window_minutes": WINDOW,
        "horizon_minutes": 1,
        "raw_rows": sum(row["raw_rows"] for row in audit),
        "split_counts": {
            s: int(np.sum(result["split"] == s))
            for s in ["train", "validation", "test"]
        },
    }


def feature_matrix(x, engineered=False):
    x = np.asarray(x)
    if (
        x.ndim != 3
        or x.shape[1:] != (WINDOW, len(FEATURES))
        or not np.isfinite(x).all()
    ):
        raise ValueError("Expected finite input shaped [batch, 4 minutes, 4 features]")
    flattened = x.reshape(len(x), -1)
    if not engineered:
        return flattened
    speed = x[:, :, 0]
    return np.column_stack(
        [
            flattened,
            speed.mean(1),
            speed.std(1),
            speed[:, -1] - speed[:, 0],
            x[:, :, 1].mean(1),
            speed[:, -1] - speed[:, -2],
        ]
    )
