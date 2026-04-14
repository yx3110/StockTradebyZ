"""ng1.2.2: Return-weighted CE quintile classification (Wei et al. 2025 CIKM, arXiv 2510.14156).

Output at inference is P(class=4) — user explicitly rejected blending across
classes, so the top-quintile probability is the sole ranking signal.
"""
from typing import Sequence, Tuple

import numpy as np


N_CLASSES = 5
RETURN_CAP = 0.25
QUINTILE_MODEL_KEY = 'lgb_quintile'


def build_quintile_labels(y: np.ndarray, dates: Sequence) -> np.ndarray:
    """Bin each date's `y` into 5 equal-frequency classes {0..4}.

    Args:
        y: shape (n,). Continuous label (typically label_10d industry-excess).
            NaN samples keep NaN (caller must filter them out before fitting).
        dates: shape (n,). Trading date per row, already aligned with y.

    Returns:
        classes: shape (n,). Integer {0,1,2,3,4}. NaN input → -1 sentinel
                 (caller filters before passing to LightGBM).

    Groups with < N_CLASSES valid rows are all marked -1: fitting on a date
    where quintiles are empty kills calibration.
    """
    y = np.asarray(y, dtype=np.float64)
    dates = np.asarray(dates)
    if len(y) != len(dates):
        raise ValueError(f"y and dates length mismatch: {len(y)} vs {len(dates)}")

    classes = np.full(len(y), -1, dtype=np.int8)

    # Iterate once — sort by date to get contiguous groups in a single pass.
    order = np.argsort(dates, kind='stable')
    sorted_dates = dates[order]
    sorted_y = y[order]

    # Group boundaries via run-length encoding on sorted dates
    change_points = np.concatenate(
        ([0], np.where(sorted_dates[1:] != sorted_dates[:-1])[0] + 1, [len(sorted_dates)])
    )

    for i in range(len(change_points) - 1):
        s, e = change_points[i], change_points[i + 1]
        grp_y = sorted_y[s:e]
        valid = ~np.isnan(grp_y)
        n_valid = int(valid.sum())
        if n_valid < N_CLASSES:
            continue  # leave as -1

        valid_y = grp_y[valid]
        # Equal-frequency bins: cut points at [0.2, 0.4, 0.6, 0.8] quantiles.
        # np.searchsorted + sorted edges reproduces pd.qcut behavior without pandas.
        edges = np.quantile(valid_y, [0.2, 0.4, 0.6, 0.8])
        # Monotonic tie-break: if edges collapse (many ties), demote to ranks.
        if not np.all(np.diff(edges) > 1e-12):
            ranks = np.argsort(np.argsort(valid_y, kind='stable'))
            grp_classes = np.clip(ranks * N_CLASSES // n_valid, 0, N_CLASSES - 1).astype(np.int8)
        else:
            grp_classes = np.searchsorted(edges, valid_y, side='right').astype(np.int8)
            grp_classes = np.clip(grp_classes, 0, N_CLASSES - 1)

        classes[order[s:e][valid]] = grp_classes

    return classes


def build_return_weights(y: np.ndarray, cap: float = RETURN_CAP) -> np.ndarray:
    """Weight = |clip(y, -cap, cap)|. NaN → 0 (unused anyway since class=-1)."""
    if cap <= 0:
        raise ValueError(f"cap must be positive, got {cap}")
    y = np.asarray(y, dtype=np.float64)
    # np.clip passes NaN through; abs(NaN) = NaN; nan_to_num handles it.
    w = np.abs(np.clip(y, -cap, cap))
    return np.nan_to_num(w, nan=0.0, posinf=cap, neginf=cap)


def strong_buy_prob(proba: np.ndarray) -> np.ndarray:
    """Extract P(class=4) = Strong Buy from a multiclass prediction matrix.

    LightGBM multiclass predict() returns shape (n, n_classes). Convention:
    class 4 is the best quintile — the one we want to select on.
    """
    proba = np.asarray(proba)
    if proba.ndim != 2 or proba.shape[1] != N_CLASSES:
        raise ValueError(
            f"proba must be (n, {N_CLASSES}), got {proba.shape}"
        )
    return proba[:, N_CLASSES - 1]


def make_quintile_dataset(
    y: np.ndarray, dates: Sequence
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-shot helper: returns (classes, weights, valid_mask) aligned to input.

    Callers should use `valid_mask` to subset X/classes/weights before fit.
    Rows with class=-1 are dropped (too few stocks on that date, or NaN label).
    """
    classes = build_quintile_labels(y, dates)
    weights = build_return_weights(y)
    valid_mask = classes >= 0
    return classes, weights, valid_mask


class QuintileStrongBuyModel:
    """Picklable wrapper: multiclass LGB `.predict(X)` → 1-D P(class=4).

    The ensemble pipeline expects 1-D scores from every member; raw multiclass
    predict returns (n, 5).
    """

    def __init__(self, booster):
        self.booster = booster

    def predict(self, X, **kwargs) -> np.ndarray:
        proba = self.booster.predict(X, **kwargs)
        return strong_buy_prob(proba)
