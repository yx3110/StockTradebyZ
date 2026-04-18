"""ng1.3.0 Tier C amount-based candidate factors (4 candidates → 2 after EMT gate).

Spec: docs/superpowers/specs/2026-04-18-ng130-multitask-design.md §5.4

  1. amihud_illiq_20d:    mean(|ret| / amount × 1e8) — 非流动性溢价
  2. vwap_close_ratio_20d: mean((close - vwap) / vwap) — 偏离成交均价
  3. amount_acceleration_5d: amount_ma5 / amount_ma20 - 1 — 量能加速
  4. tail_beta_60d:       OLS β on bottom-30% market days — 尾部系统性风险

All factors NaN-safe; requires sufficient history (marked per function).
"""
from typing import Dict
import numpy as np

NG130_TIERC_CANDIDATES = (
    'amihud_illiq_20d',
    'vwap_close_ratio_20d',
    'amount_acceleration_5d',
    'tail_beta_60d',
)


def compute_amihud_illiq_20d(closes: np.ndarray, amounts: np.ndarray) -> float:
    """Amihud illiquidity = mean(|daily_ret| / amount × 1e8) over 20d.

    Args:
        closes: Length ≥ 21 (need 20 returns). Oldest→newest.
        amounts: Length ≥ 21. In CNY (Tushare scale).

    Returns:
        Amihud value; NaN if insufficient data or all amounts zero.
    """
    if len(closes) < 21 or len(amounts) < 21:
        return np.nan

    closes = np.asarray(closes[-21:], dtype=np.float64)
    amounts = np.asarray(amounts[-21:], dtype=np.float64)

    daily_rets = np.diff(closes) / (closes[:-1] + 1e-8)
    daily_amounts = amounts[1:]

    valid = daily_amounts > 0
    if valid.sum() < 3:
        return np.nan

    illiq = np.abs(daily_rets[valid]) / (daily_amounts[valid] / 1e8)
    return float(illiq.mean())


def compute_vwap_close_ratio_20d(
    closes: np.ndarray, amounts: np.ndarray, volumes: np.ndarray,
) -> float:
    """Mean (close - vwap) / vwap over 20 days."""
    if len(closes) < 20 or len(amounts) < 20 or len(volumes) < 20:
        return np.nan

    closes = np.asarray(closes[-20:], dtype=np.float64)
    amounts = np.asarray(amounts[-20:], dtype=np.float64)
    volumes = np.asarray(volumes[-20:], dtype=np.float64)

    valid = volumes > 0
    if valid.sum() < 5:
        return np.nan

    vwap = amounts[valid] / volumes[valid]
    ratio = (closes[valid] - vwap) / (vwap + 1e-8)
    return float(ratio.mean())


def compute_amount_acceleration_5d(amounts: np.ndarray) -> float:
    """Amount acceleration = ma5 / ma20 - 1."""
    if len(amounts) < 20:
        return np.nan

    amounts = np.asarray(amounts[-20:], dtype=np.float64)
    ma5 = float(amounts[-5:].mean())
    ma20 = float(amounts.mean())

    if ma20 <= 0:
        return np.nan

    return ma5 / ma20 - 1.0


def compute_tail_beta_60d(stock_rets: np.ndarray, market_rets: np.ndarray) -> float:
    """Tail beta = OLS β on bottom 30% of market return days over 60d."""
    if len(stock_rets) < 60 or len(market_rets) < 60:
        return np.nan

    s = np.asarray(stock_rets[-60:], dtype=np.float64)
    m = np.asarray(market_rets[-60:], dtype=np.float64)

    threshold = np.percentile(m, 30)
    mask = m <= threshold
    if mask.sum() < 10:
        return np.nan

    s_tail = s[mask]
    m_tail = m[mask]

    var_m = float(m_tail.var())
    if var_m < 1e-12:
        return np.nan

    cov = float(np.cov(s_tail, m_tail, ddof=1)[0, 1])
    return cov / var_m


def compute_all_tierC(
    closes: np.ndarray, amounts: np.ndarray, volumes: np.ndarray,
    stock_rets: np.ndarray, market_rets: np.ndarray,
) -> Dict[str, float]:
    """Compute all 4 Tier C candidate factors."""
    return {
        'amihud_illiq_20d': compute_amihud_illiq_20d(closes, amounts),
        'vwap_close_ratio_20d': compute_vwap_close_ratio_20d(closes, amounts, volumes),
        'amount_acceleration_5d': compute_amount_acceleration_5d(amounts),
        'tail_beta_60d': compute_tail_beta_60d(stock_rets, market_rets),
    }
