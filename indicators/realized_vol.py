"""B2 已实现波动率信号 (60d RV + 252d percentile + hysteresis + 3d streak).

输入: 沪深 300 收盘价 Series (index=trade_date)
输出: DataFrame with columns rv_60d, rv_percentile_252, b2_bull, b2_streak

设计:
    - rv_60d = std(log_returns) over 60d (relative percentile — annualization not needed)
    - rv_percentile_252 = rolling rank of rv_60d in past 252d window (0..1)
    - hysteresis: percentile < 0.30 → bull (low vol = calm market);
                  percentile > 0.70 → bear (high vol = stressed market);
                  0.30-0.70 → hold previous state
    - 3d streak required before regime flip is confirmed

    Implementation note: apply_threshold_with_hysteresis_and_streak treats
    score > hi as bull. We feed score = 1 - percentile (inverted) so that
    low percentile (low vol) maps to a high score (bull-friendly):
        score = 1 - pct:  pct < 0.30 → score > 0.70 → bull
                          pct > 0.70 → score < 0.30 → bear
        lo_for_helper = 1 - hysteresis_hi = 1 - 0.70 = 0.30
        hi_for_helper = 1 - hysteresis_lo = 1 - 0.30 = 0.70

    Column name rv_percentile_252 hard-codes 252 to match Task 4 DB schema
    (b2_rv_percentile_252). If percentile_window differs from 252, the column
    name still reads rv_percentile_252 — callers should use the default.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from indicators.breadth import apply_threshold_with_hysteresis_and_streak


def compute_realized_vol_signal(
    close: pd.Series,
    rv_window: int = 60,
    percentile_window: int = 252,
    streak_days: int = 3,
    hysteresis_lo: float = 0.30,
    hysteresis_hi: float = 0.70,
) -> pd.DataFrame:
    """B2 realized vol signal.

    Parameters
    ----------
    close:             沪深 300 (or any index) closing prices, indexed by trade_date.
    rv_window:         window for rolling std of log-returns (default 60).
    percentile_window: window over which rv is ranked (default 252).
    streak_days:       consecutive days required to confirm regime flip (default 3).
    hysteresis_lo:     lower percentile band; below this → bull signal (default 0.30).
    hysteresis_hi:     upper percentile band; above this → bear signal (default 0.70).

    Returns
    -------
    DataFrame indexed like close with columns:
        rv_60d:            rolling std of log-returns (rv_window days)
        rv_percentile_252: rolling percentile rank in past percentile_window days
        b2_bull:           1=bull, 0=bear (Int8, NaN-safe)
        b2_streak:         consecutive days in current regime (Int32)
    """
    if close.empty:
        raise ValueError('close is empty')
    if (close <= 0).any():
        raise ValueError('close contains non-positive values')

    log_ret = np.log(close).diff()
    rv = log_ret.rolling(rv_window, min_periods=rv_window).std()

    # Rolling percentile rank: fraction of past window values that rv exceeds.
    pct = rv.rolling(percentile_window, min_periods=percentile_window).rank(pct=True)

    # Invert so that low-vol (low pct) → high score → bull.
    # See module docstring for the hysteresis threshold mapping.
    score = 1.0 - pct
    bull, streak = apply_threshold_with_hysteresis_and_streak(
        score,
        lo=1.0 - hysteresis_hi,   # 0.30: pct > 0.70 → score < 0.30 → bear
        hi=1.0 - hysteresis_lo,   # 0.70: pct < 0.30 → score > 0.70 → bull
        streak_days=streak_days,
        initial=1,  # start neutral-bull (low-vol baseline assumption)
    )

    return pd.DataFrame({
        'rv_60d': rv,
        'rv_percentile_252': pct,
        'b2_bull': bull.astype('Int8'),
        'b2_streak': streak.astype('Int32'),
    })
