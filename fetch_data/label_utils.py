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


def compute_vn_labels_from_future_prices(
    base_open: float,
    future_closes: Dict[int, float],
    horizons: Tuple[int, ...] = (3, 5, 10, 15),
    path_horizon: int = 10,
    min_sigma: float = 1e-4,
) -> Dict[str, float]:
    """ng1.2.1 Sharpe-style path labels: cumret_Nd / path_std_Nd + path stats.

    Emits:
      vn_label_Nd for each N in horizons (cumret / per-horizon std of daily rets)
      path_mean_{path_horizon}d, path_std_{path_horizon}d,
      downside_std_{path_horizon}d (std of negative daily rets only).

    future_closes must include every day from 1..path_horizon so we can compute
    daily returns. base_open = T+1 open (entry). close@day i = close at T+1+i.
    """
    keys = [f'vn_label_{h}d' for h in horizons] + [
        f'path_mean_{path_horizon}d',
        f'path_std_{path_horizon}d',
        f'downside_std_{path_horizon}d',
    ]
    if base_open <= 0 or np.isnan(base_open):
        return {k: np.nan for k in keys}

    # Walk the path up to max(horizons) so every vn_label_Nd can be computed.
    walk_to = max(max(horizons), path_horizon)
    prev = base_open
    daily_rets = []
    for d in range(1, walk_to + 1):
        close_d = future_closes.get(d, np.nan)
        if close_d is None or np.isnan(close_d) or close_d <= 0:
            break
        daily_rets.append(close_d / prev - 1.0)
        prev = close_d
    daily_rets = np.asarray(daily_rets, dtype=np.float64)

    out = {k: np.nan for k in keys}
    if len(daily_rets) >= path_horizon:
        window = daily_rets[:path_horizon]
        out[f'path_mean_{path_horizon}d'] = float(window.mean())
        out[f'path_std_{path_horizon}d'] = float(window.std(ddof=1)) if len(window) > 1 else np.nan
        neg = window[window < 0]
        out[f'downside_std_{path_horizon}d'] = float(neg.std(ddof=1)) if len(neg) > 1 else 0.0

    # Prefix sums so per-horizon std is O(1): var(x[:h]) = (S2_h - S_h²/h) / (h-1)
    # where S_h = sum(x[:h]) and S2_h = sum(x[:h]**2).
    if len(daily_rets) > 0:
        cs = np.cumsum(daily_rets)
        cs2 = np.cumsum(daily_rets * daily_rets)
    for h in horizons:
        close_h = future_closes.get(h, np.nan)
        if close_h is None or np.isnan(close_h) or close_h <= 0:
            continue
        if len(daily_rets) < h:
            continue
        cumret = close_h / base_open - 1.0
        if h > 1:
            s_h = cs[h - 1]
            s2_h = cs2[h - 1]
            var = (s2_h - s_h * s_h / h) / (h - 1)
            sigma = float(np.sqrt(max(var, 0.0)))
        else:
            sigma = 0.0
        # sqrt(h) floor: for IID daily noise, std scales with sqrt(h) — a flat
        # floor would over-reward short horizons with tiny denominators.
        floor = min_sigma * np.sqrt(h)
        if sigma < floor:
            sigma = floor
        out[f'vn_label_{h}d'] = float(cumret / sigma)

    return out
