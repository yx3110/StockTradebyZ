#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BRAIN 验证过的 Alpha 因子

这些因子已在 WorldQuant BRAIN USA TOP3000 回测验证,
按 Sharpe/Fitness 排序, 选择与我们现有特征互补的因子导入.

验证日期: 2026-03-21
区域: USA TOP3000, decay=5, neutralization=SUBINDUSTRY

注意: 这些因子的信号方向在 US/A股可能不同, 但作为特征输入
LightGBM/XGBoost 会自动学习最优方向和非线性关系.
"""

import numpy as np
import pandas as pd
from typing import Dict


# ============================================================
# BRAIN 验证结果 (2026-03-21)
# ============================================================
BRAIN_VALIDATION = {
    '3d_reversal':        {'sharpe': 1.45, 'fitness': 0.86, 'turnover': 0.483, 'returns': 0.169},
    'combo_reversal_vol': {'sharpe': 1.43, 'fitness': 0.89, 'turnover': 0.400, 'returns': 0.154},
    '5d_reversal':        {'sharpe': 1.16, 'fitness': 0.69, 'turnover': 0.374, 'returns': 0.133},
    'volume_surprise':    {'sharpe': 0.75, 'fitness': 0.21, 'turnover': 0.444, 'returns': 0.036},
    'ma20_reversion':     {'sharpe': 0.71, 'fitness': 0.46, 'turnover': 0.199, 'returns': 0.085},
}


def compute_brain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算 BRAIN 验证过的新特征 (不与现有特征重复)

    输入 df 需要有: close, high, low, volume, price_change_pct 列
    按单只股票的时序数据传入

    Returns:
        新增列的 DataFrame (与 df 同 index)
    """
    result = pd.DataFrame(index=df.index)

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    returns = close.pct_change()

    # ------ 1. 日内强度 (Intraday Intensity) ------
    # (2*close - high - low) / (high - low)
    # Sharpe 在 US 为负 = 收盘偏低反而后续涨, 但树模型自动学方向
    hl_range = high - low
    result['brain_intraday_intensity'] = np.where(
        hl_range > 0,
        (2 * close - high - low) / hl_range,
        0
    )

    # ------ 2. 日内振幅均值 (High-Low Ratio) ------
    # mean(high/low - 1, 10): 衡量 10 日平均日内波动
    result['brain_high_low_ratio'] = (high / low - 1).rolling(10).mean()

    # ------ 3. 收盘偏离最高价 (Close-to-High) ------
    # mean((high - close) / (high - low), 10): 越大=收盘越偏离当日高点
    result['brain_close_to_high'] = np.where(
        hl_range > 0,
        (high - close) / hl_range,
        0.5
    )
    result['brain_close_to_high'] = result['brain_close_to_high'].rolling(10).mean()

    # ------ 4. 短/长波动率比 (Realized Vol Ratio) ------
    # std(returns, 5) / std(returns, 20): <1=短期平静, >1=短期激烈
    vol_5 = returns.rolling(5).std()
    vol_20 = returns.rolling(20).std()
    result['brain_vol_ratio'] = vol_5 / (vol_20 + 1e-8)

    # ------ 5. 波动率的波动率 (Vol-of-Vol) ------
    # std(std(returns, 5), 20): 波动率本身的不稳定性
    result['brain_vol_of_vol'] = vol_5.rolling(20).std()

    # ------ 6. 线性衰减动量 (Decay Linear Momentum) ------
    # 近期权重更大的加权平均收益, 比简单 mean 更灵敏
    def decay_linear(series, d):
        weights = np.arange(1, d + 1, dtype=float)
        weights /= weights.sum()
        return series.rolling(d).apply(lambda x: np.dot(x, weights), raw=True)

    result['brain_momentum_decay5'] = decay_linear(returns, 5)
    result['brain_momentum_decay10'] = decay_linear(returns, 10)

    # ------ 7. 量价背离 (Volume-Price Divergence) ------
    # corr(close, volume, 20) - corr(close, volume, 5)
    # 正=长期正相关但短期脱钩, 可能是趋势即将反转的信号
    corr_20 = close.rolling(20).corr(volume)
    corr_5 = close.rolling(5).corr(volume)
    result['brain_vol_price_divergence'] = corr_20 - corr_5

    # ------ 8. 换手率动量 (Turnover Momentum) ------
    # delta(volume/mean(volume,20), 5): 量比的 5 日变化
    vol_ratio = volume / volume.rolling(20).mean()
    result['brain_turnover_momentum'] = vol_ratio.diff(5)

    # ============================================================
    # Phase 2: 学术 + A股特色 + 微观结构因子 (20个新因子)
    # ============================================================

    # ------ 9. 52周低点反弹 (52-Week Low Bounce) ------
    # close / min(close, 252d): 离52周低点的距离, >1越远
    # 低位=价值/抄底, 高位=趋势/风险. Fama-French + Anchoring.
    min_252 = close.rolling(252, min_periods=60).min()
    result['brain_52w_low_bounce'] = close / (min_252 + 1e-8)

    # ------ 10. MA60 均值回归 (Mean Reversion to MA60) ------
    # (ma60 - close) / ma60: 正=低于均线(超卖), 负=高于均线(超买)
    ma60 = close.rolling(60).mean()
    result['brain_ma60_reversion'] = (ma60 - close) / (ma60 + 1e-8)

    # ------ 11. 波动率不对称 (Volatility Asymmetry) ------
    # downside_vol / upside_vol: >1=下行波动更大=风险信号
    # Campbell-Hentschel (1992): 波动率不对称预测未来收益
    up_ret = returns.clip(lower=0)
    dn_ret = returns.clip(upper=0)
    up_vol = up_ret.rolling(20).std()
    dn_vol = dn_ret.rolling(20).std()
    result['brain_vol_asymmetry'] = dn_vol / (up_vol + 1e-8)

    # ------ 12. Roll隐含价差 (Roll Spread Proxy) ------
    # sqrt(max(0, -cov(delta_close, lag_delta_close))): 隐含买卖价差
    # Roll (1984): 从价格序列反推流动性成本
    delta_close = close.diff()
    lag_delta = delta_close.shift(1)
    roll_cov = delta_close.rolling(20).cov(lag_delta)
    result['brain_roll_spread'] = np.sqrt(np.maximum(0, -roll_cov))

    # ------ 13. 极端日频率 (Extreme Day Frequency) ------
    # count(|ret| > 2*std, 20d) / 20: 近期出现极端波动的频率
    ret_std = returns.rolling(60, min_periods=20).std()
    is_extreme = (returns.abs() > 2 * ret_std).astype(float)
    result['brain_extreme_day_freq'] = is_extreme.rolling(20).mean()

    # ------ 14. 动量崩溃保护 (Momentum Crash Hedge) ------
    # momentum * (1 - |drawdown|): 回撤大时打折动量信号
    # AQR: 动量策略在崩溃期失效, 需要对冲
    mom_10d = close / close.shift(10) - 1
    dd_20d = (close / close.rolling(20).max() - 1).abs()
    result['brain_momentum_crash_hedge'] = mom_10d * (1 - dd_20d)

    # ------ 15. 损失厌恶指标 (Loss Aversion Proxy) ------
    # 近20日跌幅 * (跌幅为负): 大跌后的强制卖盘压力
    # Odean (1998): 投资者持有亏损太久
    ret_20d = close / close.shift(20) - 1
    result['brain_loss_aversion'] = np.where(ret_20d < 0, -ret_20d, 0)

    # ------ 16. 历史高点阻力 (Historical High Resistance) ------
    # (max_60d - close) / close: 近60日高点形成的压力位
    max_60d = close.rolling(60).max()
    result['brain_high_resistance'] = (max_60d - close) / (close + 1e-8)

    # ------ 17. 隐含买卖价差 (Bid-Ask Spread Proxy via HL) ------
    # mean(high/low - 1, 20d): 用日内振幅近似买卖价差
    # Corwin-Schultz (2012): high-low 价差估计器
    result['brain_hl_spread'] = (high / low - 1).rolling(20).mean()

    # ------ 18. 收益自相关 (Return Autocorrelation) ------
    # corr(ret_t, ret_{t-1}, 20d): 正=趋势延续, 负=均值回归
    ret_lag = returns.shift(1)
    result['brain_ret_autocorr'] = returns.rolling(20).corr(ret_lag)

    # ------ 19. 尾部风险 VaR (Tail Risk) ------
    # 5th percentile of returns over 60d: 越负=尾部风险越大
    result['brain_tail_risk'] = returns.rolling(60, min_periods=20).quantile(0.05)

    # ------ 20. 成交量加权动量 (VWAP Momentum) ------
    # sum(ret * volume, 10d) / sum(volume, 10d): 放量时的涨跌方向
    rv = returns * volume
    result['brain_vwap_momentum'] = rv.rolling(10).sum() / (volume.rolling(10).sum() + 1e-8)

    # ------ 21. 连涨/连跌天数 (Consecutive Up/Down Days) ------
    # 近期连续上涨/下跌天数, 衡量短期趋势强度
    up_streak = (returns > 0).astype(float)
    result['brain_up_streak_ratio'] = up_streak.rolling(10).mean()

    # ------ 22. Hurst 指数代理 (Hurst Exponent Proxy) ------
    # R/S 分析简化版: >0.5=趋势性, <0.5=均值回归, =0.5随机游走
    # 简化 Hurst: 用 R/S 分析的滚动窗口版
    def _rs_stat(x):
        x = x[~np.isnan(x)]
        if len(x) < 10:
            return np.nan
        mean_c = x.mean()
        devs = np.cumsum(x - mean_c)
        R = devs.max() - devs.min()
        S = x.std()
        return R / (S + 1e-8) if S > 1e-8 else np.nan

    rs = returns.rolling(40, min_periods=20).apply(_rs_stat, raw=True)
    result['brain_hurst_proxy'] = np.log(rs + 1e-8) / np.log(40)

    # ------ 23. 涨停后冷却 (Post-Limit-Up Cooldown) ------
    # A股特色: 涨停后1-5日的平均收益, 追涨的代价
    is_limit_up = (returns > 0.09).astype(float)
    result['brain_post_limitup_ret'] = is_limit_up.shift(1).rolling(5).mean()

    # ------ 24. 量价协同度 (Volume-Price Coordination) ------
    # 涨时放量+跌时缩量 = 健康趋势, 反之=不健康
    up_vol_signal = np.where(returns > 0, volume, 0)
    dn_vol_signal = np.where(returns < 0, volume, 0)
    up_vol_avg = pd.Series(up_vol_signal, index=df.index).rolling(10).mean()
    dn_vol_avg = pd.Series(dn_vol_signal, index=df.index).rolling(10).mean()
    result['brain_vol_price_coord'] = (up_vol_avg - dn_vol_avg) / (up_vol_avg + dn_vol_avg + 1e-8)

    # ------ 25. 价格加速度二阶导 (Price Jerk) ------
    # 动量的变化率: 加速上涨 vs 减速上涨
    mom_5d = close / close.shift(5) - 1
    mom_5d_prev = mom_5d.shift(5)
    result['brain_price_jerk'] = mom_5d - mom_5d_prev

    # ------ 26. 跳空缺口强度 (Gap Strength) ------
    # A股特色: 开盘跳空的方向和幅度
    open_price = df['open']
    prev_close = close.shift(1)
    gap = (open_price - prev_close) / (prev_close + 1e-8)
    result['brain_gap_strength'] = gap.rolling(10).mean()

    # ------ 27. 资金流向代理 (Money Flow Proxy) ------
    # 类似 MFI 但简化: (典型价格 * 成交量) 的方向变化
    typical_price = (high + low + close) / 3
    mf = typical_price * volume
    mf_ratio = mf / mf.rolling(20).mean()
    result['brain_money_flow'] = mf_ratio.rolling(5).mean() - 1

    # ------ 28. 波动率聚集度 (Volatility Clustering) ------
    # GARCH 效应简化: 当前波动率 vs 长期均值的比值持续性
    vol_5d = returns.rolling(5).std()
    vol_60d = returns.rolling(60, min_periods=20).std()
    vol_cluster = (vol_5d / (vol_60d + 1e-8)).rolling(5).mean()
    result['brain_vol_clustering'] = vol_cluster

    return result


def compute_brain_features_batch(stock_data: Dict[str, pd.DataFrame],
                                  target_date: str = None) -> pd.DataFrame:
    """
    批量计算所有股票的 BRAIN 特征

    Args:
        stock_data: {code: DataFrame with OHLCV}
        target_date: 如果指定, 只返回该日期的值

    Returns:
        DataFrame [code, trade_date, brain_xxx, ...]
    """
    rows = []
    for code, df in stock_data.items():
        if df.empty or len(df) < 25:
            continue

        brain_features = compute_brain_features(df)

        if target_date:
            mask = df['trade_date'] == target_date
            if mask.any():
                idx = mask.idxmax()
                row = {'code': code, 'trade_date': target_date}
                for col in brain_features.columns:
                    val = brain_features.loc[idx, col]
                    row[col] = float(val) if not pd.isna(val) else 0.0
                rows.append(row)
        else:
            for i, (idx, row_data) in enumerate(df.iterrows()):
                if i < 20:  # 前 20 天数据不够计算 rolling
                    continue
                row = {'code': code, 'trade_date': row_data['trade_date']}
                for col in brain_features.columns:
                    val = brain_features.loc[idx, col]
                    row[col] = float(val) if not pd.isna(val) else 0.0
                rows.append(row)

    return pd.DataFrame(rows)


# BRAIN 特征列名 (用于训练时识别)
BRAIN_FEATURE_COLS = [
    # Phase 1: BRAIN 验证因子 (9个)
    'brain_intraday_intensity',
    'brain_high_low_ratio',
    'brain_close_to_high',
    'brain_vol_ratio',
    'brain_vol_of_vol',
    'brain_momentum_decay5',
    'brain_momentum_decay10',
    'brain_vol_price_divergence',
    'brain_turnover_momentum',
    # Phase 2: 学术 + A股 + 微观结构因子 (20个)
    'brain_52w_low_bounce',
    'brain_ma60_reversion',
    'brain_vol_asymmetry',
    'brain_roll_spread',
    'brain_extreme_day_freq',
    'brain_momentum_crash_hedge',
    'brain_loss_aversion',
    'brain_high_resistance',
    'brain_hl_spread',
    'brain_ret_autocorr',
    'brain_tail_risk',
    'brain_vwap_momentum',
    'brain_up_streak_ratio',
    'brain_hurst_proxy',
    'brain_post_limitup_ret',
    'brain_vol_price_coord',
    'brain_price_jerk',
    'brain_gap_strength',
    'brain_money_flow',
    'brain_vol_clustering',
]

# 前一轮验证的最优3因子
BRAIN_TOP3 = [
    'brain_high_low_ratio',
    'brain_close_to_high',
    'brain_momentum_decay10',
]
