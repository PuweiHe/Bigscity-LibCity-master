"""Shared residual anchor; explicit masks preserve observed normalized zeros."""

import torch


def last_observed_value(history, observed=None):
    """Return [B, 1, N, C]; all-missing histories fall back to normalized zero.

    New pipelines must pass a boolean mask derived before normalization.
    The value-based fallback retains compatibility with historical checkpoints.
    """
    if observed is None:
        observed = history.ne(0)
    if observed.shape != history.shape or observed.dtype != torch.bool:
        raise ValueError('Observation mask must be boolean and match history')
    index = history.shape[1] - 1 - observed.flip(1).int().argmax(1, keepdim=True)
    anchor = history.gather(1, index)
    return torch.where(observed.any(1, keepdim=True), anchor, 0)
