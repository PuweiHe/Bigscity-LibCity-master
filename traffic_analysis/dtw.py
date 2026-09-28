"""Exact scalar DTW and duplicate-free pairwise evaluation."""

import numpy as np


def _series(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Each series must be nonempty, one-dimensional and finite")
    return values


def exact_dtw(left, right):
    """Return unnormalized DTW with absolute local cost and O(min(T,U)) space."""
    left, right = _series(left), _series(right)
    if len(left) < len(right):
        left, right = right, left
    previous = np.full(len(right) + 1, np.inf)
    previous[0] = 0.0
    for value in left:
        current = np.full(len(right) + 1, np.inf)
        for j, other in enumerate(right, start=1):
            current[j] = abs(value - other) + min(
                previous[j], current[j - 1], previous[j - 1]
            )
        previous = current
    return float(previous[-1])


def pairwise_dtw(series):
    """Evaluate each unordered pair once; diagonal is zero by DTW identity."""
    values = [_series(value) for value in series]
    if not values:
        raise ValueError("At least one series is required")
    matrix = np.zeros((len(values), len(values)))
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            matrix[i, j] = matrix[j, i] = exact_dtw(values[i], values[j])
    return matrix
