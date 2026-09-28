"""Reproduce original FastDTW loops and verify unordered-pair optimization."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from traffic_analysis.__main__ import load_series


def pairwise(series, optimized):
    matrix = np.zeros((len(series), len(series)))
    calls = 0
    for i in range(len(series)):
        for j in range(i + 1, len(series)) if optimized else range(len(series)):
            if i == j:
                continue
            matrix[i, j], _ = fastdtw(
                series[i][:, None], series[j][:, None], dist=euclidean
            )
            calls += 1
            if optimized:
                matrix[j, i] = matrix[i, j]
    return matrix, calls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, series, _ = load_series(args.input, "分钟", "站点", "交通流量")
    timings = {False: [], True: []}
    for repeat in range(5):
        results = {}
        for optimized in [False, True] if repeat % 2 == 0 else [True, False]:
            start = perf_counter()
            matrix, calls = pairwise(series, optimized)
            timings[optimized].append(perf_counter() - start)
            results[optimized] = (matrix, calls)
    report = {
        "source": "local preconstruction minute-flow CSV; raw data excluded",
        "stations": len(series),
        "station_minute_rows": sum(map(len, series)),
        "baseline_calls": results[False][1],
        "optimized_calls": results[True][1],
        "max_absolute_error": float(
            np.max(np.abs(results[False][0] - results[True][0]))
        ),
        "baseline_seconds": timings[False],
        "optimized_seconds": timings[True],
        "median_speedup": float(np.median(timings[False]) / np.median(timings[True])),
        "backend": "fastdtw 0.3.4; default radius=1; scipy Euclidean local cost",
        "equivalent": bool(
            np.allclose(results[False][0], results[True][0], rtol=1e-12, atol=1e-12)
        ),
        "scope": "Diagnostic only: approximate FastDTW is asymmetric on this input; mirroring changes output and is NOT accepted as an equivalent optimization",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
