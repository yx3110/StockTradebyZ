"""0AMV 牛熊分类器 — 可组合框架.

用法:
    from indicators.regime_classifier import RegimeClassifier

    clf = RegimeClassifier('v5_smooth3')   # 推荐预设
    regime = clf.fit_predict(amv_df)       # returns np.ndarray of +1/-1

预设:
    v5_smooth3   = (var1>ma60) AND (macd>0), 3 日平滑           ← 2024-2026 SOTA
    v9_loose     = (var1>ma60) AND (macd>0 OR macd上升)         ← pre-2020 最稳
    v11_loose_smooth3 = v9 + 3 日平滑                            ← 兼具 v5+v9 优点
    v3_strict    = 当前生产 (急涨/急跌+缓跌)
    v1_simple    = 仅位置
    v2_base      = (var1>ma60) AND (macd>0), 无平滑

每个预设是 (raw_signal_fn, smoothing_fn) 的组合.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ============================================================
# 原子信号
# ============================================================

def sig_position(var1, ma60, **_):
    """长期位置: var1 > ma60"""
    return var1 > ma60


def sig_macd_water(macd, **_):
    """动量水上: macd 柱 > 0  (等价 dif > dea)"""
    return macd > 0


def sig_macd_rising(macd, **_):
    """动量加速: macd 柱本日 > 昨日"""
    arr = np.zeros(len(macd), dtype=bool)
    arr[1:] = macd[1:] > macd[:-1]
    return arr


def sig_compass_align(c5, c13, c34, **_):
    """指南针对齐: c5 > c13 > c34"""
    return (c5 > c13) & (c13 > c34)


# ============================================================
# 平滑器
# ============================================================

def persist_n(raw_bull: np.ndarray, n_days: int = 1) -> np.ndarray:
    """N 日 streak 才切换. n=1 即不平滑."""
    n = len(raw_bull)
    if n_days <= 1:
        return np.where(raw_bull, 1, -1).astype(int)
    regime = np.zeros(n, dtype=int)
    regime[0] = 1 if raw_bull[0] else -1
    bull_streak = 0
    bear_streak = 0
    for i in range(n):
        if raw_bull[i]:
            bull_streak += 1
            bear_streak = 0
        else:
            bear_streak += 1
            bull_streak = 0
        prev = regime[i - 1] if i > 0 else regime[0]
        if prev == 1:
            regime[i] = -1 if bear_streak >= n_days else 1
        else:
            regime[i] = 1 if bull_streak >= n_days else -1
    return regime


def asymmetric(raw_bull: np.ndarray, bull_n: int = 1, bear_n: int = 2) -> np.ndarray:
    """非对称切换: 切牛 bull_n 天, 切熊 bear_n 天."""
    n = len(raw_bull)
    regime = np.zeros(n, dtype=int)
    regime[0] = 1 if raw_bull[0] else -1
    bull_streak = 0
    bear_streak = 0
    for i in range(n):
        if raw_bull[i]:
            bull_streak += 1
            bear_streak = 0
        else:
            bear_streak += 1
            bull_streak = 0
        prev = regime[i - 1] if i > 0 else regime[0]
        if prev == 1:
            regime[i] = -1 if bear_streak >= bear_n else 1
        else:
            regime[i] = 1 if bull_streak >= bull_n else -1
    return regime


# ============================================================
# 预设
# ============================================================

def _v1_simple(amv: dict) -> np.ndarray:
    return persist_n(sig_position(**amv), 1)


def _v2_base(amv: dict) -> np.ndarray:
    return persist_n(sig_position(**amv) & sig_macd_water(**amv), 1)


def _v3_strict(amv: dict, slow_bear_days: int = 10) -> np.ndarray:
    """生产规则: 急涨切牛 + 急跌/缓跌切熊. 与 indicators/market_amv.compute_regime 等价."""
    var1, ma60, macd = amv['var1'], amv['ma60'], amv['macd']
    n = len(var1)
    regime = np.zeros(n, dtype=int)
    pct = np.zeros(n)
    pct[1:] = (var1[1:] - var1[:-1]) / (var1[:-1] + 1e-15)
    regime[0] = 1 if var1[0] > ma60[0] else -1
    bear_streak = 0
    for i in range(1, n):
        prev = regime[i - 1]
        if prev == -1:
            bear_streak = 0
            bull_signal = (pct[i] >= 0.043 and var1[i] > ma60[i] and macd[i] > 0)
            regime[i] = 1 if bull_signal else -1
        else:
            bear_signal = (pct[i] <= -0.023 and var1[i] < ma60[i] and macd[i] < 0)
            if var1[i] < ma60[i] and macd[i] < 0:
                bear_streak += 1
            else:
                bear_streak = 0
            if bear_signal or bear_streak >= slow_bear_days:
                regime[i] = -1
                bear_streak = 0
            else:
                regime[i] = 1
    return regime


def _v5_smooth3(amv: dict) -> np.ndarray:
    raw = sig_position(**amv) & sig_macd_water(**amv)
    return persist_n(raw, 3)


def _v9_loose(amv: dict) -> np.ndarray:
    raw = sig_position(**amv) & (sig_macd_water(**amv) | sig_macd_rising(**amv))
    return persist_n(raw, 1)


def _v11_loose_smooth3(amv: dict) -> np.ndarray:
    """V5 + V9 融合: 宽松条件 + 3 日平滑."""
    raw = sig_position(**amv) & (sig_macd_water(**amv) | sig_macd_rising(**amv))
    return persist_n(raw, 3)


PRESETS = {
    'v1_simple':           _v1_simple,
    'v2_base':             _v2_base,
    'v3_strict':           _v3_strict,
    'v5_smooth3':          _v5_smooth3,
    'v9_loose':            _v9_loose,
    'v11_loose_smooth3':   _v11_loose_smooth3,
}

DEFAULT_PRESET = 'v11_loose_smooth3'   # 改这一行可以切生产


# ============================================================
# 主入口
# ============================================================

class RegimeClassifier:
    """0AMV regime 分类器 — 输入 amv DataFrame, 输出 +1/-1 regime."""

    def __init__(self, preset: str = DEFAULT_PRESET):
        if preset not in PRESETS:
            raise ValueError(f'unknown preset {preset!r}, available: {list(PRESETS)}')
        self.preset = preset
        self._fn = PRESETS[preset]

    def fit_predict(self, amv_df: pd.DataFrame) -> np.ndarray:
        """amv_df 需含 var1, amv_ma60, amv_macd, amv_dif, amv_dea, amv_c5/c13/c34."""
        amv = {
            'var1': amv_df['var1'].values,
            'ma60': amv_df['amv_ma60'].values,
            'macd': amv_df['amv_macd'].values,
            'dif':  amv_df['amv_dif'].values,
            'dea':  amv_df['amv_dea'].values,
            'c5':   amv_df['amv_c5'].values,
            'c13':  amv_df['amv_c13'].values,
            'c34':  amv_df['amv_c34'].values,
        }
        return self._fn(amv)

    def __repr__(self):
        return f'RegimeClassifier(preset={self.preset!r})'


def classify(amv_df: pd.DataFrame, preset: str = DEFAULT_PRESET) -> np.ndarray:
    """便捷函数. 等价 RegimeClassifier(preset).fit_predict(amv_df)."""
    return RegimeClassifier(preset).fit_predict(amv_df)


# ============================================================
# ng2.0a: Multi-beta vote regime
# ============================================================

def compute_regime_v2(
    v11_bull: pd.Series,
    b1_bull: pd.Series,
    b2_bull: pd.Series,
    system_streak: int = 3,
    vote_threshold: int = 2,
) -> pd.DataFrame:
    """ng2.0a: hard vote across 3 binary signals + system-level streak.

    Args:
        v11_bull, b1_bull, b2_bull: each 1=bull, 0=bear, NaN allowed (treated as 0)
            All three Series must share the same DatetimeIndex.
        system_streak: streak_days at vote-output level (default 3)
        vote_threshold: minimum bull count to call bull (default 2 = majority;
            3 = unanimous).

    Returns:
        DataFrame indexed same as inputs with columns:
            vote_count: int 0..3 (count of bulls)
            regime_v2_raw: +1 if vote >= vote_threshold else -1
            regime_v2_streak: consecutive days where raw majority side is the same
            regime_v2: +1/-1 (after system_streak applied to raw)
    """
    if not (v11_bull.index.equals(b1_bull.index) and v11_bull.index.equals(b2_bull.index)):
        raise ValueError('v11_bull / b1_bull / b2_bull indices must match')
    if vote_threshold not in (1, 2, 3):
        raise ValueError(f'vote_threshold={vote_threshold} must be in {{1, 2, 3}}')

    v = v11_bull.fillna(0).astype(int)
    b1 = b1_bull.fillna(0).astype(int)
    b2 = b2_bull.fillna(0).astype(int)

    vote = v + b1 + b2  # 0..3
    raw_bull_int = (vote >= vote_threshold).astype(int)  # 1=bull, 0=bear

    # Apply system-level streak using existing persist_n
    raw_arr = raw_bull_int.to_numpy()
    confirmed = persist_n(raw_arr, n_days=system_streak)  # +1/-1

    # Track streak length of consecutive raw majority side (vectorized)
    grp = (raw_bull_int != raw_bull_int.shift()).cumsum()
    streak = (raw_bull_int.groupby(grp).cumcount() + 1).astype(np.int32)

    out = pd.DataFrame({
        'vote_count': vote.astype('Int8'),
        'regime_v2_raw': np.where(raw_bull_int == 1, 1, -1).astype(np.int8),
        'regime_v2_streak': streak,
        'regime_v2': confirmed.astype(np.int8),
    }, index=v11_bull.index)
    return out
