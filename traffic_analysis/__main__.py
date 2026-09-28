"""Run with python -m traffic_analysis --help."""

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .dtw import exact_dtw, pairwise_dtw


def load_series(path, time_column, station_column, value_column):
    frame = pd.read_csv(path)
    required = [time_column, station_column, value_column]
    if len(set(required)) != 3 or not set(required).issubset(frame.columns):
        raise ValueError("Three distinct existing columns are required")
    frame = frame[required].copy()
    if frame.empty or frame.isna().any().any():
        raise ValueError("Input must contain nonmissing observations")
    frame[time_column] = pd.to_datetime(frame[time_column], errors="raise")
    frame[value_column] = pd.to_numeric(frame[value_column], errors="raise")
    if not np.isfinite(frame[value_column].to_numpy()).all():
        raise ValueError("Values must be finite")
    if frame.duplicated([station_column, time_column]).any():
        raise ValueError("Duplicate station/timestamp keys must be resolved explicitly")
    names, series, coverage = [], [], []
    for name, group in frame.groupby(station_column, sort=True):
        group = group.sort_values(time_column)
        if (
            len(group) > 1
            and not group[time_column].diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all()
        ):
            raise ValueError(
                "Each station must have a continuous one-minute grid; do not fill outages implicitly"
            )
        names.append(str(name))
        series.append(group[value_column].to_numpy(dtype=float))
        coverage.append(
            {
                "station": str(name),
                "samples": len(group),
                "start": str(group[time_column].iloc[0]),
                "end": str(group[time_column].iloc[-1]),
            }
        )
    return names, series, coverage


def benchmark(series, repeats):
    """Compare loop schedules using the SAME exact distance implementation."""
    baseline_times, optimized_times = [], []
    error = 0.0
    for repeat in range(repeats):
        results = {}
        # Alternate order to reduce systematic warmup bias.
        for method in (
            ["baseline", "optimized"] if repeat % 2 == 0 else ["optimized", "baseline"]
        ):
            start = perf_counter()
            if method == "baseline":
                matrix = np.zeros((len(series), len(series)))
                for i in range(len(series)):
                    for j in range(len(series)):
                        if i != j:
                            matrix[i, j] = exact_dtw(series[i], series[j])
            else:
                matrix = pairwise_dtw(series)
            elapsed = perf_counter() - start
            (baseline_times if method == "baseline" else optimized_times).append(
                elapsed
            )
            results[method] = matrix
        np.testing.assert_allclose(
            results["baseline"], results["optimized"], rtol=1e-12, atol=1e-12
        )
        error = max(
            error, float(np.max(np.abs(results["baseline"] - results["optimized"])))
        )
    n = len(series)
    report = {
        "stations": n,
        "baseline_calls": n * (n - 1),
        "optimized_calls": n * (n - 1) // 2,
        "max_absolute_error": error,
        "repeats": repeats,
        "baseline_seconds": baseline_times,
        "optimized_seconds": optimized_times,
        "median_speedup": float(np.median(baseline_times) / np.median(optimized_times)),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "distance": "exact DTW, absolute scalar cost",
        "scope": "Pair scheduling only; not a FastDTW comparison or forecasting accuracy result",
    }
    return results["optimized"], report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--time-column", default="分钟")
    parser.add_argument("--station-column", default="站点")
    parser.add_argument("--value-column", default="交通流量")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.input:
        names, series, coverage = load_series(
            args.input, args.time_column, args.station_column, args.value_column
        )
        source = "user-provided CSV"
    else:
        rng = np.random.default_rng(42)
        names = [f"station_{i}" for i in range(6)]
        series = [
            np.maximum(
                0, 20 + 8 * np.sin(np.arange(60) / 10 + i / 3) + rng.normal(size=60)
            )
            for i in range(6)
        ]
        coverage = [{"station": name, "samples": 60} for name in names]
        source = "synthetic, seed=42; not internship data"
    matrix, report = benchmark(series, args.repeats)
    report.update(source=source, coverage=coverage)
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, index=names, columns=names).to_csv(
        args.output / "dtw_matrix.csv"
    )
    (args.output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
