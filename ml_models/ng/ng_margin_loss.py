"""ng1.2.0: Pairwise Margin Ranking Loss for LightGBM.

Reference: Wei et al. 2025 CIKM "On Evaluating Loss Functions for Stock Ranking:
An Empirical Analysis With Transformer Model" (arXiv 2510.14156). Margin ranking
won against ListNet/RankNet/BPR/MSE on S&P 500 (AR=16.23%, Sharpe=0.7529).

Loss formula:
    L = sum_{i,j in same group, y_i > y_j} max(0, m - (ŷ_i - ŷ_j))

Groups = trading dates (cross-sectional ranking per day).
"""
from functools import partial
from typing import Tuple

import numpy as np


def _margin_objective_impl(
    y_pred: np.ndarray, train_data, margin: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Pairwise hinge gradient/hessian for LightGBM custom objective.

    Returns:
        grad[i] = in_deg[i] - out_deg[i]
            out_deg[i] = #pairs where i beats j (y_i>y_j, p_i-p_j<margin)
                         → positive loss slope, pushes grad DOWN (train higher)
            in_deg[i]  = #pairs where j beats i (y_j>y_i, p_j-p_i<margin)
                         → pushes grad UP (train lower)
        hess[i] = max(out_deg[i] + in_deg[i], 1.0)
            hinge loss has zero second-derivative almost everywhere; using the
            active-pair count keeps LightGBM's Newton step well-conditioned.
    """
    y_true = train_data.get_label()
    group = train_data.get_group()
    n = len(y_true)
    grad = np.zeros(n, dtype=np.float64)
    hess = np.ones(n, dtype=np.float64)

    if group is None or len(group) == 0:
        return grad, hess

    start = 0
    for g_size in group:
        g_size = int(g_size)
        end = start + g_size
        if g_size < 2:
            start = end
            continue

        y_g = y_true[start:end].astype(np.float64, copy=False)
        p_g = y_pred[start:end].astype(np.float64, copy=False)

        # Skip groups where all labels tie — zero active pairs, no work to do.
        # Common on days where most labels are NaN-filled to a constant.
        if y_g.max() == y_g.min():
            start = end
            continue

        # Fused expression — NumPy frees the (g,g) float64 matrices sooner than
        # if we bound them to named refs. At g=7000 each intermediate is ~400MB.
        active = (y_g[:, None] > y_g[None, :]) & \
                 ((p_g[:, None] - p_g[None, :]) < margin)

        out_deg = active.sum(axis=1)
        in_deg = active.sum(axis=0)
        grad[start:end] = in_deg - out_deg
        hess[start:end] = np.maximum(out_deg + in_deg, 1.0)

        start = end

    return grad, hess


def _margin_eval_impl(
    y_pred: np.ndarray, eval_data, margin: float
) -> Tuple[str, float, bool]:
    """Mean pairwise hinge loss across all (y_i > y_j) pairs within each group.

    Returns (metric_name, value, is_higher_better=False) per LightGBM's
    custom eval callback contract.
    """
    y_true = eval_data.get_label()
    group = eval_data.get_group()

    if group is None or len(group) == 0:
        return ('margin_loss', 0.0, False)

    total_loss = 0.0
    total_pairs = 0
    start = 0
    for g_size in group:
        g_size = int(g_size)
        end = start + g_size
        if g_size < 2:
            start = end
            continue
        y_g = y_true[start:end].astype(np.float64, copy=False)
        p_g = y_pred[start:end].astype(np.float64, copy=False)
        if y_g.max() == y_g.min():
            start = end
            continue
        pair_mask = y_g[:, None] > y_g[None, :]
        # np.where avoids a fancy-index copy of the (g²/2) active entries.
        hinge = np.where(
            pair_mask,
            np.maximum(0.0, margin - (p_g[:, None] - p_g[None, :])),
            0.0,
        )
        total_loss += float(hinge.sum())
        total_pairs += int(pair_mask.sum())
        start = end

    mean_loss = total_loss / max(total_pairs, 1)
    return ('margin_loss', mean_loss, False)


def make_margin_objective(margin: float = 0.05):
    """Return a picklable LightGBM objective callable with bound margin.

    Uses functools.partial so joblib.dump of the trained model can serialize
    the bound parameter; closures would silently fail or drop the reference
    on reload.
    """
    if margin <= 0:
        raise ValueError(f"margin must be positive, got {margin}")
    return partial(_margin_objective_impl, margin=margin)


def make_margin_eval_metric(margin: float = 0.05):
    """Return a picklable LightGBM eval-metric callable with bound margin."""
    if margin <= 0:
        raise ValueError(f"margin must be positive, got {margin}")
    return partial(_margin_eval_impl, margin=margin)
