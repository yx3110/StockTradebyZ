"""ng1.3.0 Tier B moneyflow factor functions (3 factors).

Spec: docs/superpowers/specs/2026-04-18-ng130-multitask-design.md §5.3

  1. elg_net_inflow_20d_z:  log-sign transform of 20d sum of (buy_elg - sell_elg).
                             Optional CS z-score if history provided.
  2. mf_main_ratio_20d:      ∑(net_lg + net_elg) / (∑(total_amount) × window), 20d.
  3. mf_concentration_20d:   std(daily_net_mf, ddof=1) / mean(|daily_net_mf|), 20d.

Input: List[Dict] sorted oldest→newest, each dict has:
  buy_{sm,md,lg,elg}_amount, sell_{sm,md,lg,elg}_amount, net_mf_amount
"""
from typing import Dict, List, Optional, Sequence
import numpy as np

NG130_MF_FACTORS = (
    'elg_net_inflow_20d_z',
    'mf_main_ratio_20d',
    'mf_concentration_20d',
)

_EMPTY_RESULT = {name: np.nan for name in NG130_MF_FACTORS}
_WINDOW = 20


def compute_ng130_mf_factors(
    records: List[Dict],
    cs_z_history_elg: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Compute 3 Tier B moneyflow factors for ng1.3.0.

    Args:
        records: List of moneyflow dicts, oldest→newest. Uses last _WINDOW entries.
        cs_z_history_elg: Optional CS 20d-sum-net-elg history for z-scoring.
            If None or <2 entries, falls back to log-sign self-transform (stable, dimensionless).

    Returns:
        Dict with 3 keys; NaN if no records.
    """
    if not records:
        return dict(_EMPTY_RESULT)

    records_window = records[-_WINDOW:]
    n_days = len(records_window)

    result: Dict[str, float] = {}

    # --- Factor 1: elg_net_inflow_20d_z ---
    net_elg_daily = np.array([
        (r.get('buy_elg_amount', 0) or 0) - (r.get('sell_elg_amount', 0) or 0)
        for r in records_window
    ], dtype=np.float64)
    sum_net_elg = float(net_elg_daily.sum())

    if cs_z_history_elg is not None and len(cs_z_history_elg) >= 2:
        history = np.asarray(cs_z_history_elg, dtype=np.float64)
        mu = float(history.mean())
        sigma = float(history.std()) + 1e-8
        result['elg_net_inflow_20d_z'] = (sum_net_elg - mu) / sigma
    else:
        # Fallback: log-sign self-transform. Stable, dimensionless, monotonic in sign/magnitude.
        # Divisor 1e6 keeps typical stocks in ~[-5, 5]; extreme 龙头股 (20d|sum|≈2e10) ≈ ±10.
        # GBDT handles bounded outliers fine; downstream rank-normalization at composite time.
        result['elg_net_inflow_20d_z'] = float(
            np.sign(sum_net_elg) * np.log1p(abs(sum_net_elg) / 1e6)
        )

    # --- Factor 2: mf_main_ratio_20d ---
    # ∑(net_lg + net_elg) / (∑(total_amount) × window_size)
    # normalises by total traded amount per day across the window period
    net_lg_daily = np.array([
        (r.get('buy_lg_amount', 0) or 0) - (r.get('sell_lg_amount', 0) or 0)
        for r in records_window
    ], dtype=np.float64)
    total_amount_daily = np.array([
        (r.get('buy_sm_amount', 0) or 0) + (r.get('sell_sm_amount', 0) or 0)
        + (r.get('buy_md_amount', 0) or 0) + (r.get('sell_md_amount', 0) or 0)
        + (r.get('buy_lg_amount', 0) or 0) + (r.get('sell_lg_amount', 0) or 0)
        + (r.get('buy_elg_amount', 0) or 0) + (r.get('sell_elg_amount', 0) or 0)
        for r in records_window
    ], dtype=np.float64)

    sum_main_net = float(net_lg_daily.sum() + net_elg_daily.sum())
    sum_total = float(total_amount_daily.sum())
    denominator = sum_total * n_days
    result['mf_main_ratio_20d'] = sum_main_net / (denominator + 1e-8) if denominator > 0 else np.nan

    # --- Factor 3: mf_concentration_20d ---
    # std(daily_net_mf, ddof=1) / mean(|daily_net_mf|)
    # ddof=1 (sample std) ensures ratio > 1.0 for perfectly alternating ±signal
    net_mf_daily = np.array([
        r.get('net_mf_amount', 0) or 0
        for r in records_window
    ], dtype=np.float64)

    mean_abs = float(np.mean(np.abs(net_mf_daily)))
    if n_days < 2 or mean_abs <= 0:
        result['mf_concentration_20d'] = np.nan
    else:
        std_val = float(np.std(net_mf_daily, ddof=1))
        result['mf_concentration_20d'] = std_val / (mean_abs + 1e-8)

    return result
