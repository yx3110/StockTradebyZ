"""ng1.3.0 downside label — min-cumret over N days.

Spec: docs/superpowers/specs/2026-04-18-ng130-multitask-design.md §6.2

  downside_Nd(t) = min(close[t+1:t+N+1]) / close[t] - 1

Horizons: 3, 5, 10, 15 days.
Negative values = drop; closer to 0 or positive = no drop.

Used by:
  - ng_cache_updater.py (ng1.3.x branch): computes downside_Nd per (code, trade_date)
  - ng_trainer.py: Head B (downside) target labels for multi-task training
"""
from typing import Dict
import numpy as np

NG130_DOWNSIDE_LABELS = ('downside_3d', 'downside_5d', 'downside_10d', 'downside_15d')
NG130_DOWNSIDE_HORIZONS = (3, 5, 10, 15)


def compute_downside_label(t_close: float, future_closes: np.ndarray) -> float:
    """Single-horizon downside label.

    Args:
        t_close: Close price at date t.
        future_closes: Closes from t+1 onwards (length = horizon).

    Returns:
        min(future_closes) / t_close - 1; NaN if insufficient data or t_close<=0.
    """
    if len(future_closes) == 0 or t_close <= 0:
        return np.nan
    future = np.asarray(future_closes, dtype=np.float64)
    return float(future.min() / t_close - 1.0)


def compute_all_downside_horizons(t_close: float, future_closes: np.ndarray) -> Dict[str, float]:
    """Compute all 4 horizon downside labels.

    Args:
        t_close: Close price at t.
        future_closes: Length ≥ 15 ideally; shorter horizons handled via slice.

    Returns:
        Dict with keys downside_{3,5,10,15}d; NaN for horizons exceeding available data.
    """
    result: Dict[str, float] = {}
    for horizon in NG130_DOWNSIDE_HORIZONS:
        if len(future_closes) >= horizon:
            result[f'downside_{horizon}d'] = compute_downside_label(
                t_close, future_closes[:horizon],
            )
        else:
            result[f'downside_{horizon}d'] = np.nan
    return result
