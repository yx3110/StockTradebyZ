"""统一标签计算 — 所有 feature cache updater 共用。

label_Nd = close[T+1+N] / open[T+1] - 1
  T = signal date, T+1 = buy date (next open), T+1+N = sell date (N-day close)
  Suspension check: volume[T+1] == 0 → skip
"""
import numpy as np
from typing import Dict, Tuple


def compute_aligned_labels(
    opens: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    current_idx: int,
    horizons: Tuple[int, ...] = (3, 5, 10, 15),
    log_return: bool = False,
) -> Dict[str, float]:
    """Compute forward-looking labels from price arrays.

    current_idx is the signal day position. Buy at open[current_idx+1],
    sell at close[current_idx+1+h].
    """
    n = len(opens)
    buy_idx = current_idx + 1

    if buy_idx >= n:
        return {f'label_{h}d': np.nan for h in horizons}

    buy_open = opens[buy_idx]
    buy_vol = volumes[buy_idx] if volumes is not None else 1.0

    if buy_vol == 0 or buy_open <= 0 or np.isnan(buy_open):
        return {f'label_{h}d': np.nan for h in horizons}

    labels = {}
    for h in horizons:
        sell_idx = buy_idx + h
        if sell_idx >= n:
            labels[f'label_{h}d'] = np.nan
            continue
        sell_close = closes[sell_idx]
        if sell_close <= 0 or np.isnan(sell_close):
            labels[f'label_{h}d'] = np.nan
            continue
        if log_return:
            labels[f'label_{h}d'] = float(np.log(sell_close / buy_open))
        else:
            labels[f'label_{h}d'] = float(sell_close / buy_open - 1.0)
    return labels


def compute_labels_from_future_prices(
    base_open: float,
    future_closes: Dict[int, float],
    horizons: Tuple[int, ...] = (3, 5, 10, 15),
) -> Dict[str, float]:
    """Compute labels from pre-loaded future prices (NG cache updater style).

    base_open: T+1 open price
    future_closes: {horizon_days: close_price_at_T+1+h}
    """
    if base_open <= 0 or np.isnan(base_open):
        return {f'label_{h}d': np.nan for h in horizons}

    labels = {}
    for h in horizons:
        close = future_closes.get(h, np.nan)
        if close is None or np.isnan(close) or close <= 0:
            labels[f'label_{h}d'] = np.nan
        else:
            labels[f'label_{h}d'] = float(close / base_open - 1.0)
    return labels
