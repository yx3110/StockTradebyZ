"""ng1.3.0 β composite scoring.

Spec: docs/superpowers/specs/2026-04-18-ng130-multitask-design.md §4.1, §8

  Z_ex  = rank_pct(pred_excess_avg_10d)   (cross-sectional, per date)
  Z_dn  = rank_pct(pred_downside_avg_10d) (cross-sectional, per date)
  score = Z_ex - β · Z_dn

β ∈ [0, 1]. β=0 退化到 ng1.0.1 基线 (纯 excess). β=0.3 是初始保守默认.
β* 由 scripts/ng130_beta_search.py WF 网格搜索确定 (Phase 5).
"""
import pandas as pd

DEFAULT_BETA = 0.3


def rank_pct(series: pd.Series) -> pd.Series:
    """Cross-sectional rank percentile [1/N, 1.0]."""
    return series.rank(pct=True)


def compute_composite(
    pred_excess: pd.Series, pred_downside: pd.Series, beta: float,
) -> pd.Series:
    """Compute β-weighted composite score.

    Args:
        pred_excess: predicted excess return (higher = better)
        pred_downside: predicted downside (more negative = worse → lower rank_pct)
        beta: weight for downside penalty, ∈ [0, 1]

    Returns:
        Composite score Series, same index as inputs.
    """
    if not (0.0 <= beta <= 1.0):
        raise ValueError(f'beta must be in [0, 1], got {beta}')

    z_ex = rank_pct(pred_excess)
    z_dn = rank_pct(pred_downside)
    return z_ex - beta * z_dn
