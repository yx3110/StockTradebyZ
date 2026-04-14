"""ng1.2.3 soft-downside label transform.

Per spec §5 (docs/superpowers/specs/2026-04-14-ng123-design.md):
    label_kd = industry_excess_kd - lambda * max(0, -path_min_kd)

with default lambda=0.3 (1/5 of ng1.0.4's failed 1.5).
"""
from typing import Union

import numpy as np

ArrayLike = Union[float, np.ndarray]

DEFAULT_LAMBDA = 0.3


def compute_path_min_kd(today_close: float, future_closes: np.ndarray) -> float:
    """Return min(future_closes) / today_close - 1.

    Args:
        today_close: scalar close at date t (anchor).
        future_closes: 1-D array of closes from t+1 to t+k.

    Returns:
        path_min in [-1, +inf), typically negative. NaN if input invalid.
    """
    if today_close is None or today_close <= 0 or np.isnan(today_close):
        return np.nan
    if future_closes is None or len(future_closes) == 0:
        return np.nan
    arr = np.asarray(future_closes, dtype=np.float64)
    if not np.any(np.isfinite(arr)):
        return np.nan
    return float(np.nanmin(arr) / today_close - 1.0)


def compute_downside_kd(path_min: ArrayLike) -> ArrayLike:
    """downside = max(0, -path_min). NaN propagates."""
    if isinstance(path_min, np.ndarray):
        return np.where(np.isnan(path_min), np.nan, np.maximum(0.0, -path_min))
    if path_min is None or (isinstance(path_min, float) and np.isnan(path_min)):
        return np.nan
    return max(0.0, -float(path_min))


def apply_downside_penalty(
    excess: ArrayLike, downside: ArrayLike, lam: float = DEFAULT_LAMBDA
) -> ArrayLike:
    """Return excess - lam * downside (NaN-safe; broadcast on arrays)."""
    if lam < 0:
        raise ValueError(f"lambda must be non-negative, got {lam}")
    excess_arr = np.asarray(excess, dtype=np.float64)
    downside_arr = np.asarray(downside, dtype=np.float64)
    return excess_arr - lam * downside_arr
