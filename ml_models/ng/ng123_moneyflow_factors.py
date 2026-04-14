"""ng1.2.3 moneyflow factor functions.

Per spec §4.1 (docs/superpowers/specs/2026-04-14-ng123-design.md):
  - 12 factors split into 4 groups (A: net flow, B: persistence, C: divergence, D: cs_rank)
  - All factors NaN-safe; division-by-zero guarded with +1e-8

Input shape: List[Dict] returned by ng_cache_updater._load_moneyflow_data
  Each dict has keys: buy_{sm,md,lg,elg}_amount, sell_{sm,md,lg,elg}_amount, net_mf_amount
  Sorted oldest → newest, length up to 20.
"""
from typing import Dict, List

import numpy as np

__all__ = [
    "EMPTY_MF_RESULT",
    "aggregate_moneyflow_window",
    "compute_group_a_factors",
]

# Sentinel: returned when no moneyflow data available
EMPTY_MF_RESULT = {
    'sum_net_sm': np.nan, 'sum_net_md': np.nan,
    'sum_net_lg': np.nan, 'sum_net_elg': np.nan,
    'sum_total_amount': np.nan,
    'daily_net_sm': np.array([], dtype=np.float64),
    'daily_net_md': np.array([], dtype=np.float64),
    'daily_net_lg': np.array([], dtype=np.float64),
    'daily_net_elg': np.array([], dtype=np.float64),
    'n_days_actual': 0,
}


def aggregate_moneyflow_window(
    rows: List[Dict], n_days: int
) -> Dict:
    """Aggregate last n_days of moneyflow rows into summary stats.

    Returns dict with sums per order-size class + per-day sign arrays +
    per-day raw net arrays (daily_net_*).

    Uses the LAST n_days rows (most recent) so callers can pass the full
    20-day history and request any sub-window without slicing themselves.

    Note: If raw row values are explicit NaN (rather than None), the `or 0`
    pattern propagates NaN through the sums. Tushare data is clean (None or
    real numbers), so this is currently a non-issue. _load_moneyflow_data
    converts None → 0.0 before reaching this function.
    """
    if not rows or n_days <= 0:
        return EMPTY_MF_RESULT.copy()

    window = rows[-n_days:]  # take last n_days (oldest → newest preserved)
    n_actual = len(window)

    # Per-day net flows (buy - sell) for each class
    net_sm = np.array(
        [(r.get('buy_sm_amount') or 0) - (r.get('sell_sm_amount') or 0) for r in window],
        dtype=np.float64)
    net_md = np.array(
        [(r.get('buy_md_amount') or 0) - (r.get('sell_md_amount') or 0) for r in window],
        dtype=np.float64)
    net_lg = np.array(
        [(r.get('buy_lg_amount') or 0) - (r.get('sell_lg_amount') or 0) for r in window],
        dtype=np.float64)
    net_elg = np.array(
        [(r.get('buy_elg_amount') or 0) - (r.get('sell_elg_amount') or 0) for r in window],
        dtype=np.float64)

    # Total amount = sum of all (buy + sell) across all 4 classes
    sum_total_amount = 0.0
    for r in window:
        for k in ('buy_sm_amount', 'sell_sm_amount', 'buy_md_amount', 'sell_md_amount',
                  'buy_lg_amount', 'sell_lg_amount', 'buy_elg_amount', 'sell_elg_amount'):
            sum_total_amount += (r.get(k) or 0)

    return {
        'sum_net_sm': float(net_sm.sum()),
        'sum_net_md': float(net_md.sum()),
        'sum_net_lg': float(net_lg.sum()),
        'sum_net_elg': float(net_elg.sum()),
        'sum_total_amount': sum_total_amount,
        'daily_net_sm': net_sm,
        'daily_net_md': net_md,
        'daily_net_lg': net_lg,
        'daily_net_elg': net_elg,
        'n_days_actual': n_actual,
    }


# ---------------------------------------------------------------------------
# Group A: Smart Money Net Flow Magnitude (4 factors)
# ---------------------------------------------------------------------------

def compute_group_a_factors(rows: List[Dict]) -> Dict[str, float]:
    """Compute factors 1-4 from spec §4.1 Group A.

    Returns dict with keys:
      mf_net_elg_5d_ratio   — net ELG flow / total flow, 5d window
      mf_net_elg_20d_ratio  — net ELG flow / total flow, 20d window
      mf_net_lg_5d_ratio    — net LG flow / total flow, 5d window
      mf_smart_net_share_20d — (net_elg+net_lg) / sum(|daily_net_X|), 20d

    All values NaN-safe; division-by-zero returns NaN.
    """
    result: Dict[str, float] = {
        'mf_net_elg_5d_ratio': np.nan,
        'mf_net_elg_20d_ratio': np.nan,
        'mf_net_lg_5d_ratio': np.nan,
        'mf_smart_net_share_20d': np.nan,
    }
    if not rows:
        return result

    agg5 = aggregate_moneyflow_window(rows, n_days=5)
    agg20 = aggregate_moneyflow_window(rows, n_days=20)

    # Factor 1: mf_net_elg_5d_ratio = sum_net_elg_5d / sum_total_5d
    if agg5['n_days_actual'] > 0 and agg5['sum_total_amount'] > 1e-8:
        result['mf_net_elg_5d_ratio'] = agg5['sum_net_elg'] / agg5['sum_total_amount']

    # Factor 2: mf_net_elg_20d_ratio = sum_net_elg_20d / sum_total_20d
    if agg20['n_days_actual'] > 0 and agg20['sum_total_amount'] > 1e-8:
        result['mf_net_elg_20d_ratio'] = agg20['sum_net_elg'] / agg20['sum_total_amount']

    # Factor 3: mf_net_lg_5d_ratio = sum_net_lg_5d / sum_total_5d
    if agg5['n_days_actual'] > 0 and agg5['sum_total_amount'] > 1e-8:
        result['mf_net_lg_5d_ratio'] = agg5['sum_net_lg'] / agg5['sum_total_amount']

    # Factor 4: mf_smart_net_share_20d = sum(net_elg+net_lg, 20d) / sum(|daily_net_X|, 20d)
    # CRITICAL: denominator uses sum of per-day absolute values (not abs of sum)
    # to avoid sign-cancellation bias when net flow alternates direction.
    # Spec §4.1 row 4: Σ(|net_elg|+|net_lg|+|net_md|+|net_sm|, 20d) is per-day |·|.
    if agg20['n_days_actual'] > 0:
        smart_num = agg20['sum_net_elg'] + agg20['sum_net_lg']
        abs_daily_total = float(
            np.abs(agg20['daily_net_elg']).sum()
            + np.abs(agg20['daily_net_lg']).sum()
            + np.abs(agg20['daily_net_md']).sum()
            + np.abs(agg20['daily_net_sm']).sum()
        )
        if abs_daily_total > 1e-8:
            result['mf_smart_net_share_20d'] = smart_num / abs_daily_total

    return result
