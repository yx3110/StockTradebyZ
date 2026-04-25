"""B1 市场广度信号 (% stocks above MA + hysteresis + streak).

输入: 全 A 股每日收盘价面板 (DataFrame: index=trade_date, columns=stock_code)
输出: DataFrame with columns:
    pct_above_ma20: % stocks 收盘 > 自身 MA_{ma_short} (列名固定为 ma20，供 Task 4 DB schema 用)
    pct_above_ma60: % stocks 收盘 > 自身 MA_{ma_long}  (列名固定为 ma60)
    b1_score: weight_short * pct_above_ma20 + (1-weight_short) * pct_above_ma60 ∈ [0, 1]
    b1_bull:  1=bull, 0=bear (含 hysteresis 0.45/0.55 + streak_days 连续确认)
    b1_streak: 当前 regime 已连续天数

设计:
    - 滚动 MA 用 pandas rolling, min_periods=ma_window 保证前 N-1 日 NaN
    - hysteresis: score > hi → raw_bull=1; score < lo → raw_bull=0; 否则沿用前值
    - streak: raw_bull 连续 streak_days 日才确认翻转
    - 数据不足前 max(ma_long, streak_days) 日 pct/score 为 NaN, b1_bull=initial=0
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_threshold_with_hysteresis_and_streak(
    score: pd.Series,
    lo: float = 0.45,
    hi: float = 0.55,
    streak_days: int = 3,
    initial: int = 0,
) -> tuple[pd.Series, pd.Series]:
    """对 score 应用 hysteresis + streak, 返回 (bull, streak).

    raw_bull[t]:
        score[t] > hi  → 1
        score[t] < lo  → 0
        otherwise      → raw_bull[t-1]  (hysteresis band)

    confirmed_bull flips to candidate only after streak_days consecutive raw_bull
    equal to the candidate value.

    Parameters
    ----------
    score:       pd.Series of float (NaN allowed — treated as "hold previous")
    lo, hi:      hysteresis thresholds (lo < hi)
    streak_days: minimum consecutive days to confirm a regime change
    initial:     starting regime (0=bear, 1=bull)

    Returns
    -------
    (confirmed_bull, streak) both indexed like score
    """
    n = len(score)
    # --- Pass 1: apply hysteresis to get raw signal ---
    raw = np.full(n, initial, dtype=np.int8)
    prev = initial
    for i, s in enumerate(score.values):
        if np.isnan(s):
            raw[i] = prev
            continue
        if s > hi:
            raw[i] = 1
        elif s < lo:
            raw[i] = 0
        else:
            raw[i] = prev
        prev = raw[i]

    # --- Pass 2: accumulate raw streak counts, then confirm flip ---
    streak_arr = np.ones(n, dtype=np.int32)
    for i in range(1, n):
        if raw[i] == raw[i - 1]:
            streak_arr[i] = streak_arr[i - 1] + 1
        # else: already 1 from initialization

    confirmed = np.full(n, initial, dtype=np.int8)
    cur = initial
    for i in range(n):
        if raw[i] != cur and streak_arr[i] >= streak_days:
            cur = raw[i]
        confirmed[i] = cur

    return (
        pd.Series(confirmed, index=score.index, name='b1_bull'),
        pd.Series(streak_arr, index=score.index, name='b1_streak'),
    )


def compute_breadth_signal(
    close_panel: pd.DataFrame,
    ma_short: int = 20,
    ma_long: int = 60,
    weight_short: float = 0.5,
    streak_days: int = 3,
    hysteresis_lo: float = 0.45,
    hysteresis_hi: float = 0.55,
) -> pd.DataFrame:
    """B1 breadth signal.

    Parameters
    ----------
    close_panel:   DataFrame (index=trade_date, columns=stock_code). NaN allowed.
    ma_short:      short MA window (default 20). Column name 'pct_above_ma20' is fixed.
    ma_long:       long MA window (default 60). Column name 'pct_above_ma60' is fixed.
    weight_short:  weight of ma_short in b1_score (default 0.5).
    streak_days:   consecutive days required to confirm regime flip (default 3).
    hysteresis_lo: lower band of hysteresis zone (default 0.45).
    hysteresis_hi: upper band of hysteresis zone (default 0.55).

    Returns
    -------
    DataFrame indexed by trade_date with columns:
        pct_above_ma20, pct_above_ma60, b1_score, b1_bull, b1_streak
    """
    if close_panel.empty:
        raise ValueError('close_panel is empty')
    if not (0.0 <= weight_short <= 1.0):
        raise ValueError(f'weight_short={weight_short} must be in [0, 1]')

    ma_s = close_panel.rolling(ma_short, min_periods=ma_short).mean()
    ma_l = close_panel.rolling(ma_long, min_periods=ma_long).mean()

    valid_s = ma_s.notna() & close_panel.notna()
    valid_l = ma_l.notna() & close_panel.notna()

    above_s = (close_panel > ma_s).where(valid_s)
    above_l = (close_panel > ma_l).where(valid_l)

    n_valid_s = valid_s.sum(axis=1).replace(0, np.nan)
    n_valid_l = valid_l.sum(axis=1).replace(0, np.nan)

    pct_s = above_s.sum(axis=1) / n_valid_s
    pct_l = above_l.sum(axis=1) / n_valid_l

    score = weight_short * pct_s + (1.0 - weight_short) * pct_l

    bull, streak = apply_threshold_with_hysteresis_and_streak(
        score,
        lo=hysteresis_lo,
        hi=hysteresis_hi,
        streak_days=streak_days,
        initial=0,
    )

    return pd.DataFrame({
        'pct_above_ma20': pct_s,
        'pct_above_ma60': pct_l,
        'b1_score': score,
        'b1_bull': bull.astype('Int8'),
        'b1_streak': streak.astype('Int32'),
    })
