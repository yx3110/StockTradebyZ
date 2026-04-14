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

# Window sizes used across all moneyflow factor groups (per spec §4.1)
_SHORT_WINDOW = 5    # short-term window (factors 1, 3, 6, 7)
_LONG_WINDOW = 20    # long-term window (factors 2, 4, 5)

__all__ = [
    "EMPTY_MF_RESULT",
    "aggregate_moneyflow_window",
    "compute_group_a_factors",
    "compute_group_b_factors",
    "compute_group_c_factors",
    "compute_stock_mf_scalars",
    "compute_group_d_factors",
    "compute_all_moneyflow_factors",
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
    """Compute Group A factors 1-4 from spec §4.1.

    Returns dict with 4 float keys (NaN-safe).

    Partial windows: When `len(rows) < 5` (or `< 20`), the corresponding factors
    are computed using whatever data is available rather than emitting NaN.
    Group A divisor is `sum_total_amount` (computed from actual rows), so partial
    and full windows produce the same ratio when data is sparse — no spec divergence.
    (Contrast Group B/C factors 5+6, which use spec-literal `/20` and `/5`.)
    Rationale: Group D cs_rank factors handle IPO-edge noise via cross-sectional
    ranking; emitting NaN here would lose the IPO-window signal entirely. Stocks
    with very short histories should be filtered upstream by ng_cache_updater.

    Keys:
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

    agg5 = aggregate_moneyflow_window(rows, n_days=_SHORT_WINDOW)
    agg20 = aggregate_moneyflow_window(rows, n_days=_LONG_WINDOW)

    # Factor 1: mf_net_elg_5d_ratio = sum_net_elg_5d / sum_total_5d
    if agg5['sum_total_amount'] > 1e-8:
        result['mf_net_elg_5d_ratio'] = agg5['sum_net_elg'] / agg5['sum_total_amount']

    # Factor 2: mf_net_elg_20d_ratio = sum_net_elg_20d / sum_total_20d
    if agg20['sum_total_amount'] > 1e-8:
        result['mf_net_elg_20d_ratio'] = agg20['sum_net_elg'] / agg20['sum_total_amount']

    # Factor 3: mf_net_lg_5d_ratio
    if agg5['sum_total_amount'] > 1e-8:
        result['mf_net_lg_5d_ratio'] = agg5['sum_net_lg'] / agg5['sum_total_amount']

    # Factor 4: mf_smart_net_share_20d
    # CRITICAL: denominator uses sum of per-day absolute values (not abs of sum)
    # to avoid sign-cancellation bias when net flow alternates direction.
    # Spec §4.1 row 4: Σ(|net_elg|+|net_lg|+|net_md|+|net_sm|, 20d) is per-day |·|.
    abs_daily_total = float(
        np.abs(agg20['daily_net_sm']).sum()
        + np.abs(agg20['daily_net_md']).sum()
        + np.abs(agg20['daily_net_lg']).sum()
        + np.abs(agg20['daily_net_elg']).sum()
    )
    if abs_daily_total > 1e-8:
        smart_num = agg20['sum_net_elg'] + agg20['sum_net_lg']
        result['mf_smart_net_share_20d'] = smart_num / abs_daily_total

    return result


# ---------------------------------------------------------------------------
# Group B: Smart Money Persistence (2 factors)
# ---------------------------------------------------------------------------

def compute_group_b_factors(rows: List[Dict]) -> Dict[str, float]:
    """Compute Group B factors 5-6 from spec §4.1.

    Returns dict with 2 float keys (NaN-safe).

    Partial windows: factors 5 and 6 use spec-literal divisors (/20, /5) so
    stocks with short histories get proportionally smaller signals. This
    matches the spec's range guarantees and avoids IPO-edge inflation.

    Keys:
      mf_elg_persistence_20d — mean(sign(daily_net_elg)) over 20d, range [-1, 1]
      mf_smart_consistency_5d — fraction of 5d where sign(net_elg)==sign(net_lg),
                                 range [0, 1]
    """
    result: Dict[str, float] = {
        'mf_elg_persistence_20d': np.nan,
        'mf_smart_consistency_5d': np.nan,
    }
    if not rows:
        return result

    agg20 = aggregate_moneyflow_window(rows, n_days=_LONG_WINDOW)
    agg5 = aggregate_moneyflow_window(rows, n_days=_SHORT_WINDOW)

    # Factor 5: persistence = mean(sign(daily_net_elg)) over 20d (spec literal /20)
    # Partial windows: empty days count as zeros via the literal /20 divisor —
    # stocks with short histories get proportionally smaller signals (intended,
    # avoids IPO-edge inflation that would distort cs_rank in Group D).
    if agg20['n_days_actual'] > 0:
        signs_elg = np.sign(agg20['daily_net_elg'])
        result['mf_elg_persistence_20d'] = float(signs_elg.sum()) / _LONG_WINDOW

    # Factor 6: consistency = fraction of 5d where sign(net_elg)==sign(net_lg)
    # (spec literal /5; partial windows: empty days count as 0)
    if agg5['n_days_actual'] > 0:
        s_elg = np.sign(agg5['daily_net_elg'])
        s_lg = np.sign(agg5['daily_net_lg'])
        result['mf_smart_consistency_5d'] = float((s_elg == s_lg).sum()) / _SHORT_WINDOW

    return result


# ---------------------------------------------------------------------------
# Group C: Smart/Retail Divergence + ELG Acceleration (2 factors)
# ---------------------------------------------------------------------------

def compute_group_c_factors(rows: List[Dict]) -> Dict[str, float]:
    """Compute Group C factors 7-8 from spec §4.1.

    Returns dict with 2 float keys (NaN-safe).

    Keys:
      mf_smart_retail_divergence_5d — sign(sum_net_elg_5d) - sign(sum_net_sm_5d),
                                       range {-2, 0, +2}
      mf_elg_acceleration_5_20 — mf_net_elg_5d_ratio - mf_net_elg_20d_ratio,
                                  range ≈ [-1, 1]

    Note: Factor 8 (`mf_elg_acceleration_5_20`) recomputes the same ratios as
    Group A factors 1 and 2. This duplication is by design — each group function
    is independent and can be called in any order. The orchestrator
    `compute_all_moneyflow_factors` runs all 3 groups; the per-stock cost is
    measured (~46ms / 1000 calls) and acceptable for batch backfill.
    """
    result: Dict[str, float] = {
        'mf_smart_retail_divergence_5d': np.nan,
        'mf_elg_acceleration_5_20': np.nan,
    }
    if not rows:
        return result

    agg5 = aggregate_moneyflow_window(rows, n_days=_SHORT_WINDOW)
    agg20 = aggregate_moneyflow_window(rows, n_days=_LONG_WINDOW)

    # Factor 7: divergence = sign(sum_net_elg_5d) - sign(sum_net_sm_5d)
    if agg5['n_days_actual'] > 0:
        sign_elg = float(np.sign(agg5['sum_net_elg']))
        sign_sm = float(np.sign(agg5['sum_net_sm']))
        result['mf_smart_retail_divergence_5d'] = sign_elg - sign_sm

    # Factor 8: acceleration = ratio_5d - ratio_20d (both require nonzero total)
    if agg5['sum_total_amount'] > 1e-8 and agg20['sum_total_amount'] > 1e-8:
        ratio_5 = agg5['sum_net_elg'] / agg5['sum_total_amount']
        ratio_20 = agg20['sum_net_elg'] / agg20['sum_total_amount']
        result['mf_elg_acceleration_5_20'] = ratio_5 - ratio_20

    return result


# ---------------------------------------------------------------------------
# Helper: compute the 4 scalar values needed for cs_rank wrapper
# ---------------------------------------------------------------------------

def compute_stock_mf_scalars(rows: List[Dict]) -> Dict[str, float]:
    """Compute the 4 raw scalars that feed Group D cs_rank factors.

    Returns NaN-filled dict for empty input. Used by ng_cache_updater to
    pre-compute peer arrays per industry per date.

    Keys (4):
      net_elg_5d_ratio   — same as factor 1 (mf_net_elg_5d_ratio)
      net_elg_20d_ratio  — same as factor 2 (mf_net_elg_20d_ratio)
      smart_net_share_20d — same as factor 4 (mf_smart_net_share_20d)
      persistence_20d    — same as factor 5 (mf_elg_persistence_20d)
    """
    result = {
        'net_elg_5d_ratio': np.nan,
        'net_elg_20d_ratio': np.nan,
        'smart_net_share_20d': np.nan,
        'persistence_20d': np.nan,
    }
    if not rows:
        return result

    a = compute_group_a_factors(rows)
    b = compute_group_b_factors(rows)
    result['net_elg_5d_ratio'] = a['mf_net_elg_5d_ratio']
    result['net_elg_20d_ratio'] = a['mf_net_elg_20d_ratio']
    result['smart_net_share_20d'] = a['mf_smart_net_share_20d']
    result['persistence_20d'] = b['mf_elg_persistence_20d']
    return result


# ---------------------------------------------------------------------------
# Group D: Cross-Sectional Industry Ranks (4 factors)
# ---------------------------------------------------------------------------

def _industry_percentile_rank_safe(value: float, peer_values: np.ndarray) -> float:
    """Mirror of ng_feature_calculator._industry_percentile_rank but NaN-safe.

    Returns 0.5 if peer array empty or all NaN; otherwise fraction strictly < value.
    """
    if peer_values is None or len(peer_values) == 0:
        return 0.5
    valid = peer_values[~np.isnan(peer_values)]
    if len(valid) < 2:
        return 0.5
    if np.isnan(value):
        return 0.5
    return float(np.mean(valid < value))


def compute_group_d_factors(
    stock_scalars: Dict[str, float],
    peer_scalars: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Compute Group D factors 9-12 from spec §4.1.

    Args:
        stock_scalars: dict from compute_stock_mf_scalars(self_rows).
            Required keys: 'net_elg_5d_ratio', 'net_elg_20d_ratio',
            'smart_net_share_20d', 'persistence_20d'.
        peer_scalars: dict {factor_name → 1D array of peer values incl. self}.
            ⚠️ Use the SAME 4 scalar-namespace keys as stock_scalars (NOT the
            'mf_*' factor names from Group A/B output). Missing keys silently
            fall back to empty array → cs_rank = 0.5 (no error signal).

    Returns dict with 4 float keys, each in [0, 1] (or 0.5 if no peers).
    """
    return {
        'cs_rank_mf_net_elg_5d': _industry_percentile_rank_safe(
            stock_scalars.get('net_elg_5d_ratio', np.nan),
            peer_scalars.get('net_elg_5d_ratio', np.array([]))),
        'cs_rank_mf_net_elg_20d': _industry_percentile_rank_safe(
            stock_scalars.get('net_elg_20d_ratio', np.nan),
            peer_scalars.get('net_elg_20d_ratio', np.array([]))),
        'cs_rank_mf_smart_net_share_20d': _industry_percentile_rank_safe(
            stock_scalars.get('smart_net_share_20d', np.nan),
            peer_scalars.get('smart_net_share_20d', np.array([]))),
        'cs_rank_mf_elg_persistence_20d': _industry_percentile_rank_safe(
            stock_scalars.get('persistence_20d', np.nan),
            peer_scalars.get('persistence_20d', np.array([]))),
    }


# ---------------------------------------------------------------------------
# Top-level orchestrator: compute all 12 factors for a single stock
# ---------------------------------------------------------------------------

def compute_all_moneyflow_factors(
    rows: List[Dict],
    stock_scalars: Dict[str, float] = None,
    peer_scalars: Dict[str, np.ndarray] = None,
) -> Dict[str, float]:
    """Compute all 12 ng1.2.3 moneyflow factors for one stock on one date.

    Args:
        rows: List of moneyflow dicts (last 20 days), oldest → newest.
        stock_scalars: Pre-computed via compute_stock_mf_scalars(rows). If None,
            extracted from already-computed Group A/B results (no double-call).
            Pass pre-computed when called in batch context — saves one extraction.
        peer_scalars: Industry peer arrays (pre-computed once per industry-date).
            Expected keys: 'net_elg_5d_ratio', 'net_elg_20d_ratio',
            'smart_net_share_20d', 'persistence_20d' (NOTE: scalar-namespace
            names, NOT Group A/B factor names with 'mf_' prefix). Pass empty
            dict {} or None if peer info unavailable (cs_rank → 0.5).
    """
    a = compute_group_a_factors(rows)
    b = compute_group_b_factors(rows)
    c = compute_group_c_factors(rows)

    result: Dict[str, float] = {}
    result.update(a)
    result.update(b)
    result.update(c)

    if stock_scalars is None:
        # Extract from already-computed Group A/B results (avoid double-call).
        stock_scalars = {
            'net_elg_5d_ratio':    a['mf_net_elg_5d_ratio'],
            'net_elg_20d_ratio':   a['mf_net_elg_20d_ratio'],
            'smart_net_share_20d': a['mf_smart_net_share_20d'],
            'persistence_20d':     b['mf_elg_persistence_20d'],
        }
    if peer_scalars is None:
        peer_scalars = {}

    result.update(compute_group_d_factors(stock_scalars, peer_scalars))
    return result
