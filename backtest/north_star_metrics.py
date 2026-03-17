#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
北极星指标计算模块 (North Star Metrics)

统一的量化交易系统评估指标计算，覆盖4个层级：
- Layer 1: 信号质量 (IC, ICIR, Rank IC)
- Layer 2: 组合构建 (换手率, 交易成本, 选股精准度)
- Layer 3: 风险控制 (MaxDD, Sharpe, Sortino, Calmar, VaR, CVaR)
- Layer 4: 最终盈利 (年化收益, 月度收益, 超额收益)

用法:
    from backtest.north_star_metrics import NorthStarEvaluator
    evaluator = NorthStarEvaluator()
    report = evaluator.full_evaluation(daily_returns, daily_ic_series, benchmark_returns)
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Dict, List, Optional, Tuple
import sqlite3
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, 'data_adapter', 'stock_data.db')

# ═══════════════════════════════════════════════════════
# 北极星目标定义
# ═══════════════════════════════════════════════════════

NORTH_STAR_TARGETS = {
    # Layer 1: 信号质量
    'daily_ic':         {'target': 0.06, 'pass': 0.03, 'good': 0.05, 'direction': 'higher'},
    'icir':             {'target': 0.50, 'pass': 0.30, 'good': 0.40, 'direction': 'higher'},
    'ic_positive_pct':  {'target': 60.0, 'pass': 55.0, 'good': 58.0, 'direction': 'higher'},
    'rank_ic':          {'target': 0.07, 'pass': 0.04, 'good': 0.06, 'direction': 'higher'},

    # Layer 3: 风险控制
    'max_drawdown':     {'target': -0.12, 'pass': -0.25, 'good': -0.15, 'direction': 'higher'},  # less negative = better
    'sharpe_ratio':     {'target': 2.0,  'pass': 1.0,  'good': 1.5,  'direction': 'higher'},
    'sortino_ratio':    {'target': 2.5,  'pass': 1.5,  'good': 2.0,  'direction': 'higher'},
    'calmar_ratio':     {'target': 2.5,  'pass': 1.0,  'good': 2.0,  'direction': 'higher'},

    # Layer 4: 最终盈利
    'annual_return':    {'target': 0.30, 'pass': 0.15, 'good': 0.20, 'direction': 'higher'},
    'monthly_win_rate': {'target': 75.0, 'pass': 58.0, 'good': 67.0, 'direction': 'higher'},
    'annual_turnover':  {'target': 30.0, 'pass': 50.0, 'good': 35.0, 'direction': 'lower'},
    'annual_cost_drag': {'target': 0.08, 'pass': 0.15, 'good': 0.10, 'direction': 'lower'},
}

# ═══════════════════════════════════════════════════════
# V2 北极星目标 (21项指标, 6档评分 0-5, 满分105)
# ═══════════════════════════════════════════════════════

NORTH_STAR_TARGETS_V2 = {
    # Layer 1: 信号质量 (6项)
    'daily_ic': {
        'pass': 0.03, 'ok': 0.04, 'good': 0.05, 'great': 0.06, 'target': 0.08,
        'direction': 'higher', 'layer': 1, 'display': 'Daily IC',
    },
    'icir': {
        'pass': 0.25, 'ok': 0.35, 'good': 0.45, 'great': 0.55, 'target': 0.70,
        'direction': 'higher', 'layer': 1, 'display': 'ICIR',
    },
    'ic_positive_pct': {
        'pass': 55, 'ok': 57, 'good': 60, 'great': 63, 'target': 68,
        'direction': 'higher', 'layer': 1, 'display': 'IC>0%',
    },
    'ic_monotonicity': {
        'pass': 2.5, 'ok': 3.0, 'good': 3.5, 'great': 4.0, 'target': 4.5,
        'direction': 'higher', 'layer': 1, 'display': 'IC单调性',
    },
    'ic_time_stability': {
        'pass': 2.0, 'ok': 1.5, 'good': 1.0, 'great': 0.8, 'target': 0.6,
        'direction': 'lower', 'layer': 1, 'display': 'IC稳定性(CV)',
        'min_days': 120,
    },
    'signal_half_life': {
        'pass': 3, 'ok': 6, 'good': 8, 'great': 12, 'target': 20,
        'direction': 'higher', 'layer': 1, 'display': '信号半衰期(天)',
    },

    # Layer 2: 组合效率 (5项)
    'annual_turnover': {
        'pass': 45, 'ok': 35, 'good': 30, 'great': 25, 'target': 20,
        'direction': 'lower', 'layer': 2, 'display': '年化换手',
    },
    'annual_cost_drag': {
        'pass': 0.13, 'ok': 0.10, 'good': 0.08, 'great': 0.07, 'target': 0.05,
        'direction': 'lower', 'layer': 2, 'display': '年化成本',
    },
    'net_gross_ratio': {
        'pass': 0.60, 'ok': 0.70, 'good': 0.75, 'great': 0.80, 'target': 0.85,
        'direction': 'higher', 'layer': 2, 'display': '净/毛收益比',
    },
    'limit_up_fail_rate': {
        'pass': 0.15, 'ok': 0.10, 'good': 0.08, 'great': 0.05, 'target': 0.02,
        'direction': 'lower', 'layer': 2, 'display': '涨停失败率',
    },
    'liquidity_coverage': {
        'pass': 0.70, 'ok': 0.80, 'good': 0.85, 'great': 0.90, 'target': 0.95,
        'direction': 'higher', 'layer': 2, 'display': '流动性覆盖',
    },

    # Layer 3: 风险控制 (5项)
    'max_drawdown': {
        'pass': -0.25, 'ok': -0.18, 'good': -0.12, 'great': -0.10, 'target': -0.08,
        'direction': 'higher', 'layer': 3, 'display': '最大回撤',
    },
    'sharpe_ratio': {
        'pass': 1.0, 'ok': 1.5, 'good': 2.0, 'great': 2.5, 'target': 3.0,
        'direction': 'higher', 'layer': 3, 'display': 'Sharpe',
    },
    'sortino_ratio': {
        'pass': 1.5, 'ok': 2.0, 'good': 2.5, 'great': 3.0, 'target': 4.0,
        'direction': 'higher', 'layer': 3, 'display': 'Sortino',
    },
    'calmar_ratio': {
        'pass': 1.0, 'ok': 2.0, 'good': 2.5, 'great': 3.0, 'target': 4.0,
        'direction': 'higher', 'layer': 3, 'display': 'Calmar',
    },
    'worst_rolling_60d_icir': {
        'pass': -0.10, 'ok': 0.0, 'good': 0.10, 'great': 0.20, 'target': 0.30,
        'direction': 'higher', 'layer': 3, 'display': '最差60日ICIR',
        'min_days': 120,
    },

    # Layer 4: 盈利与鲁棒性 (5项)
    'annual_return': {
        'pass': 0.15, 'ok': 0.20, 'good': 0.30, 'great': 0.40, 'target': 0.50,
        'direction': 'higher', 'layer': 4, 'display': '年化收益',
        'min_days': 200,
    },
    'monthly_win_rate': {
        'pass': 55, 'ok': 60, 'good': 67, 'great': 75, 'target': 83,
        'direction': 'higher', 'layer': 4, 'display': '月度胜率%',
    },
    'half_period_consistency': {
        'pass': 0.35, 'ok': 0.50, 'good': 0.60, 'great': 0.70, 'target': 0.80,
        'direction': 'higher', 'layer': 4, 'display': '前后半段一致性',
        'min_days': 120,
    },
    'cap_balance_ratio': {
        'pass': 0.3, 'ok': 0.5, 'good': 0.6, 'great': 0.7, 'target': 0.8,
        'direction': 'higher', 'layer': 4, 'display': '市值均衡度',
    },
    'median_market_cap_bn': {
        'pass': 2.0, 'ok': 3.0, 'good': 5.0, 'great': 8.0, 'target': 10.0,
        'direction': 'higher', 'layer': 4, 'display': '中位市值(亿)',
    },
}

V2_LAYER_NAMES = {
    1: '信号质量', 2: '组合效率', 3: '风险控制', 4: '盈利与鲁棒性',
}

V2_GRADE_THRESHOLDS = [
    (80, 'S'), (70, 'A+'), (60, 'A'), (45, 'B'), (30, 'C'),
]  # (pct_threshold, grade) — below 30% is D


def score_metric_v2(current: float, target_info: dict) -> Tuple[int, str]:
    """
    V2 6档评分: 0-5

    Args:
        current: 当前值
        target_info: dict with pass/ok/good/great/target/direction

    Returns:
        (score: 0-5, grade_str)
    """
    higher = target_info['direction'] == 'higher'
    thresholds = [
        (target_info['target'], 5, '★★★★★'),
        (target_info['great'],  4, '★★★★☆'),
        (target_info['good'],   3, '★★★☆☆'),
        (target_info['ok'],     2, '★★☆☆☆'),
        (target_info['pass'],   1, '★☆☆☆☆'),
    ]

    for threshold, score, grade_str in thresholds:
        if higher:
            if current >= threshold:
                return score, grade_str
        else:
            if current <= threshold:
                return score, grade_str

    return 0, '☆☆☆☆☆'


def compute_v2_grade(total_score: int, max_score: int = 105) -> str:
    """计算V2等级: S/A+/A/B/C/D"""
    if max_score <= 0:
        return 'D'
    pct = total_score / max_score * 100
    for threshold, grade in V2_GRADE_THRESHOLDS:
        if pct >= threshold:
            return grade
    return 'D'

# A股交易成本参数（散户级别）
TRANSACTION_COST = {
    'commission_rate': 0.00025,    # 佣金 0.025% 单边
    'stamp_tax_rate': 0.0005,      # 印花税 0.05% (仅卖出)
    'transfer_fee_rate': 0.00001,  # 过户费 0.001%
    'slippage_rate': 0.001,        # 滑点 0.1% 单边
}


# ═══════════════════════════════════════════════════════
# Layer 1: 信号质量指标
# ═══════════════════════════════════════════════════════

def compute_daily_ic(scores: pd.Series, returns: pd.Series, dates: pd.Series,
                     min_stocks: int = 20) -> pd.DataFrame:
    """
    计算每日截面IC（Spearman Rank Correlation）

    Args:
        scores: 预测分数
        returns: 实际收益
        dates: 交易日期
        min_stocks: 每天最少股票数

    Returns:
        DataFrame with columns: [date, ic, n_stocks]
    """
    records = []
    for date in sorted(dates.unique()):
        mask = (dates == date) & scores.notna() & returns.notna()
        day_scores = scores[mask]
        day_returns = returns[mask]

        if len(day_scores) < min_stocks:
            continue

        ic, p_val = spearmanr(day_scores, day_returns)
        if not np.isnan(ic):
            records.append({
                'date': date,
                'ic': ic,
                'p_val': p_val,
                'n_stocks': len(day_scores),
            })

    return pd.DataFrame(records)


def compute_ic_summary(ic_df: pd.DataFrame) -> Dict:
    """
    从daily IC序列计算ICIR等汇总指标

    Args:
        ic_df: compute_daily_ic的输出

    Returns:
        dict with ic_mean, ic_std, icir, ic_positive_pct, etc.
    """
    if ic_df.empty or len(ic_df) < 3:
        return {
            'ic_mean': 0, 'ic_std': 0, 'icir': 0,
            'ic_positive_pct': 0, 'ic_median': 0, 'n_days': 0,
        }

    ic_vals = ic_df['ic']
    ic_mean = ic_vals.mean()
    ic_std = ic_vals.std()
    icir = ic_mean / ic_std if ic_std > 1e-8 else 0
    ic_positive_pct = (ic_vals > 0).mean() * 100
    ic_median = ic_vals.median()

    return {
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'icir': icir,
        'ic_positive_pct': ic_positive_pct,
        'ic_median': ic_median,
        'n_days': len(ic_df),
    }


def compute_monthly_ic(ic_df: pd.DataFrame) -> pd.DataFrame:
    """按月分解IC"""
    if ic_df.empty:
        return pd.DataFrame()

    ic_copy = ic_df.copy()
    ic_copy['month'] = pd.to_datetime(ic_copy['date']).dt.to_period('M')

    monthly = []
    for month, group in ic_copy.groupby('month'):
        m_ic = group['ic'].mean()
        m_std = group['ic'].std()
        m_icir = m_ic / m_std if m_std > 1e-8 else 0
        m_pos = (group['ic'] > 0).mean() * 100
        monthly.append({
            'month': str(month),
            'ic_mean': m_ic,
            'ic_std': m_std,
            'icir': m_icir,
            'ic_positive_pct': m_pos,
            'n_days': len(group),
        })

    return pd.DataFrame(monthly)


def compute_ic_monotonicity(scores: pd.Series, returns: pd.Series,
                             dates: pd.Series, n_quantiles: int = 5,
                             min_stocks: int = 20) -> float:
    """
    计算IC单调性: 按分数分N档，检查收益是否单调递增

    Args:
        scores: 预测分数
        returns: 实际收益
        dates: 日期
        n_quantiles: 分位数量 (默认5)
        min_stocks: 最少股票数

    Returns:
        float (0-5): 单调性分数，5.0=完美单调
    """
    monotonicity_scores = []

    for date in sorted(dates.unique()):
        mask = (dates == date) & scores.notna() & returns.notna()
        day_scores = scores[mask]
        day_returns = returns[mask]

        if len(day_scores) < min_stocks:
            continue

        # 按分数分n_quantiles档
        try:
            quantile_labels = pd.qcut(day_scores, n_quantiles, labels=False, duplicates='drop')
        except ValueError:
            continue

        # 各档平均收益
        group_means = day_returns.groupby(quantile_labels).mean().sort_index()
        if len(group_means) < 3:
            continue

        # 计算相邻档位的正确排序数 (higher quantile should have higher return)
        n_pairs = len(group_means) - 1
        correct = sum(1 for i in range(n_pairs) if group_means.iloc[i+1] > group_means.iloc[i])
        # 缩放到0-5
        mono_score = correct / n_pairs * 5.0
        monotonicity_scores.append(mono_score)

    return np.mean(monotonicity_scores) if monotonicity_scores else 0.0


def compute_ic_time_stability(ic_df: pd.DataFrame, window: int = 60) -> Dict:
    """
    计算IC时间稳定性: 滚动ICIR的变异系数(CV)

    Args:
        ic_df: compute_daily_ic的输出 (columns: date, ic)
        window: 滚动窗口大小

    Returns:
        dict{cv, rolling_icir_series, mean, std, warning?}
        cv越低越好 (更稳定)
    """
    n_days = len(ic_df) if not ic_df.empty else 0

    if n_days < 120:
        print(f"  ⚠️ IC稳定性: 仅{n_days}天数据(建议≥120天), 结果可能不可靠")

    if ic_df.empty or len(ic_df) < window:
        # 短窗口自适应
        window = max(10, len(ic_df) // 2) if len(ic_df) >= 20 else len(ic_df)
        if window < 10:
            result = {'cv': 999.0, 'mean': 0, 'std': 0, 'rolling_icir_series': pd.Series(dtype=float)}
            if n_days < 120:
                result['warning'] = f'仅{n_days}天数据'
            return result

    ic_series = ic_df.set_index('date')['ic'].sort_index()
    rolling_mean = ic_series.rolling(window, min_periods=max(10, window // 2)).mean()
    rolling_std = ic_series.rolling(window, min_periods=max(10, window // 2)).std()
    rolling_icir = (rolling_mean / rolling_std).dropna()
    rolling_icir = rolling_icir.replace([np.inf, -np.inf], np.nan).dropna()

    if len(rolling_icir) < 3:
        result = {'cv': 999.0, 'mean': 0, 'std': 0, 'rolling_icir_series': pd.Series(dtype=float)}
        if n_days < 120:
            result['warning'] = f'仅{n_days}天数据'
        return result

    mean_icir = rolling_icir.mean()
    std_icir = rolling_icir.std()
    if abs(mean_icir) > 1e-8:
        cv = abs(std_icir / mean_icir)
    elif std_icir < 0.1:
        cv = 3.0  # 弱信号但波动小，给中等CV
    else:
        cv = 5.0  # 弱信号且波动大，给较差CV（但非999）

    result = {
        'cv': cv,
        'mean': mean_icir,
        'std': std_icir,
        'rolling_icir_series': rolling_icir,
    }
    if n_days < 120:
        result['warning'] = f'自适应窗口{window}天'
    return result


def compute_signal_half_life(icir_by_days: Dict[int, float]) -> float:
    """
    计算信号半衰期: ICIR从峰值降到50%的天数

    Args:
        icir_by_days: {holding_days: icir} e.g. {1: 0.42, 3: 0.40, 5: 0.45, 10: 0.61, 15: 0.65}

    Returns:
        float: 半衰期(天)。越大越好(信号持久)。
        如果ICIR单调递增或从未降到50%，返回最大天数。
    """
    if not icir_by_days or len(icir_by_days) < 2:
        return 0.0

    sorted_days = sorted(icir_by_days.keys())
    values = [icir_by_days[d] for d in sorted_days]

    peak_val = max(values)
    if peak_val <= 0:
        return 0.0

    peak_idx = values.index(peak_val)
    half_val = peak_val * 0.5

    # 从peak之后寻找降到50%的点
    for i in range(peak_idx + 1, len(values)):
        if values[i] < half_val:
            # 线性插值
            prev_day = sorted_days[i - 1]
            curr_day = sorted_days[i]
            prev_val = values[i - 1]
            curr_val = values[i]
            # 插值找精确点
            if abs(prev_val - curr_val) > 1e-8:
                frac = (prev_val - half_val) / (prev_val - curr_val)
                half_life = prev_day + frac * (curr_day - prev_day)
            else:
                half_life = (prev_day + curr_day) / 2
            return half_life

    # 从未降到50%: 信号非常持久
    return float(sorted_days[-1])


def compute_half_period_consistency(period_returns: pd.Series,
                                     holding_days: int) -> Dict:
    """
    前后半段Sharpe一致性 — 多偏移平均法

    当n较小时，单一中点分割非常脆弱（1个异常值就影响结果）。
    改为在40%-60%范围内尝试多个分割点，取ratio的平均值。

    Args:
        period_returns: 非重叠调仓期收益序列
        holding_days: 持仓天数

    Returns:
        dict{ratio, sharpe_h1, sharpe_h2}
        ratio = 多偏移平均的 min(sharpe)/max(sharpe)
    """
    n = len(period_returns)
    if n < 20:
        print(f"  ⚠️ 前后半段一致性: 仅{n}个调仓期(建议≥20)")
    if n < 6:
        return {'ratio': 0, 'sharpe_h1': 0, 'sharpe_h2': 0}

    periods_per_year = 252 / holding_days

    def _sharpe(rets):
        if len(rets) < 3:
            return 0
        annual_ret = (1 + rets).prod() ** (periods_per_year / len(rets)) - 1
        annual_vol = rets.std() * np.sqrt(periods_per_year)
        return (annual_ret - 0.02) / annual_vol if annual_vol > 1e-8 else 0

    # 多偏移分割: 40%~60% 范围内的所有有效分割点
    lo = max(3, int(n * 0.40))
    hi = min(n - 3, int(n * 0.60))
    if lo > hi:
        lo = hi = n // 2

    ratios = []
    best_s1, best_s2 = 0, 0
    for mid in range(lo, hi + 1):
        s1 = _sharpe(period_returns.iloc[:mid])
        s2 = _sharpe(period_returns.iloc[mid:])
        if s1 > 0 and s2 > 0:
            ratios.append(min(s1, s2) / max(s1, s2))
        elif s1 < 0 and s2 < 0:
            # 两半段都亏损：比较亏损程度的一致性，×0.3折扣
            ratios.append(min(abs(s1), abs(s2)) / max(abs(s1), abs(s2)) * 0.3)
        else:
            # 一正一负：不一致
            ratios.append(0)
        if mid == n // 2:
            best_s1, best_s2 = s1, s2

    avg_ratio = np.mean(ratios) if ratios else 0

    return {'ratio': avg_ratio, 'sharpe_h1': best_s1, 'sharpe_h2': best_s2}


def compute_worst_rolling_icir(ic_df: pd.DataFrame, window: int = 60) -> Dict:
    """
    最差滚动ICIR: 在daily IC上滑动窗口找最差ICIR

    Args:
        ic_df: compute_daily_ic的输出
        window: 滚动窗口

    Returns:
        dict{worst_icir, start_date, end_date, warning?}
    """
    n_days = len(ic_df) if not ic_df.empty else 0

    if n_days < 120:
        print(f"  ⚠️ 最差滚动ICIR: 仅{n_days}天数据(建议≥120天)")

    if ic_df.empty or len(ic_df) < 10:
        result = {'worst_icir': None, 'start_date': '', 'end_date': ''}
        if n_days < 120:
            result['warning'] = f'仅{n_days}天数据'
        return result

    # 自适应窗口
    window = min(window, len(ic_df) // 2)
    if window < 10:
        window = min(10, len(ic_df))

    ic_series = ic_df.set_index('date')['ic'].sort_index()
    rolling_mean = ic_series.rolling(window, min_periods=max(5, window // 2)).mean()
    rolling_std = ic_series.rolling(window, min_periods=max(5, window // 2)).std()
    rolling_icir = (rolling_mean / rolling_std).replace([np.inf, -np.inf], np.nan).dropna()

    if rolling_icir.empty:
        result = {'worst_icir': None, 'start_date': '', 'end_date': ''}
        if n_days < 120:
            result['warning'] = f'仅{n_days}天数据'
        return result

    worst_idx = rolling_icir.idxmin()
    worst_val = rolling_icir.min()

    # 找窗口起始日
    pos = ic_series.index.get_loc(worst_idx) if worst_idx in ic_series.index else 0
    if isinstance(pos, slice):
        pos = pos.start
    start_pos = max(0, pos - window + 1)
    start_date = str(ic_series.index[start_pos])

    result = {
        'worst_icir': worst_val,
        'start_date': start_date,
        'end_date': str(worst_idx),
    }
    if n_days < 120:
        result['warning'] = f'仅{n_days}天数据'
    return result


def compute_net_gross_ratio(gross_annual: float, net_annual: float) -> float:
    """净/毛收益比"""
    if abs(gross_annual) < 0.01:  # 年化 < 1% 时不可靠
        return 0.0
    ratio = net_annual / gross_annual
    return max(0.0, min(ratio, 1.0))  # clamp to [0, 1]


def compute_top_bottom_spread(scores: pd.Series, returns: pd.Series,
                              dates: pd.Series, quantile: float = 0.1,
                              min_stocks: int = 20) -> Dict:
    """
    计算Top/Bottom分位组合超额收益

    Args:
        quantile: Top/Bottom分位数 (0.1 = 10%)
    """
    top_returns_list = []
    bottom_returns_list = []

    for date in sorted(dates.unique()):
        mask = (dates == date) & scores.notna() & returns.notna()
        day_scores = scores[mask]
        day_returns = returns[mask]

        if len(day_scores) < min_stocks:
            continue

        n = max(1, int(len(day_scores) * quantile))
        sorted_idx = day_scores.argsort()
        top_idx = sorted_idx.iloc[-n:]
        bottom_idx = sorted_idx.iloc[:n]

        top_ret = day_returns.iloc[top_idx.values].mean()
        bottom_ret = day_returns.iloc[bottom_idx.values].mean()

        top_returns_list.append(top_ret)
        bottom_returns_list.append(bottom_ret)

    if not top_returns_list:
        return {'top_return': 0, 'bottom_return': 0, 'spread': 0}

    return {
        'top_return': np.mean(top_returns_list),
        'bottom_return': np.mean(bottom_returns_list),
        'spread': np.mean(top_returns_list) - np.mean(bottom_returns_list),
        'n_days': len(top_returns_list),
    }


# ═══════════════════════════════════════════════════════
# Layer 2: 组合构建指标
# ═══════════════════════════════════════════════════════

def compute_turnover(holdings_by_date: Dict[str, List[str]]) -> Dict:
    """
    计算换手率

    Args:
        holdings_by_date: {date: [stock_codes]} 每天持仓列表

    Returns:
        dict with avg_turnover, total_rebalances, annual_turnover_estimate
    """
    dates = sorted(holdings_by_date.keys())
    if len(dates) < 2:
        return {'avg_turnover': 0, 'total_rebalances': 0, 'annual_turnover_estimate': 0}

    turnovers = []
    for i in range(1, len(dates)):
        prev = set(holdings_by_date[dates[i - 1]])
        curr = set(holdings_by_date[dates[i]])

        if not prev and not curr:
            continue

        changed = len(prev.symmetric_difference(curr))
        avg_size = (len(prev) + len(curr)) / 2
        turnover = changed / (2 * max(avg_size, 1))  # 单边换手
        turnovers.append(turnover)

    avg_turnover = np.mean(turnovers) if turnovers else 0
    n_rebalances = len(turnovers)

    # 估算年化换手率：每次换手率 × 年调仓次数
    # 假设持仓N天，则年调仓 252/N 次
    trading_days_span = len(dates)
    if trading_days_span > 1:
        rebalance_freq = n_rebalances / trading_days_span * 252
        annual_turnover = avg_turnover * 2 * rebalance_freq  # 双边
    else:
        annual_turnover = 0

    return {
        'avg_turnover': avg_turnover,
        'total_rebalances': n_rebalances,
        'rebalance_freq_annual': rebalance_freq if trading_days_span > 1 else 0,
        'annual_turnover_estimate': annual_turnover,
    }


def compute_transaction_costs(turnover_per_rebal: float,
                              holding_days: int,
                              gross_annual_return: float = 0,
                              cost_params: Dict = None) -> Dict:
    """
    计算交易成本拖累

    Args:
        turnover_per_rebal: 每次调仓单边换手率
        holding_days: 持仓天数
        gross_annual_return: 已计算的毛年化收益率（由调用方传入）
        cost_params: 交易成本参数

    Returns:
        dict with annual_cost_drag, net_annual_return, etc.
    """
    if cost_params is None:
        cost_params = TRANSACTION_COST

    # 每次调仓的总交易成本（买入+卖出）
    buy_cost = cost_params['commission_rate'] + cost_params['transfer_fee_rate'] + cost_params['slippage_rate']
    sell_cost = buy_cost + cost_params['stamp_tax_rate']
    round_trip_cost = buy_cost + sell_cost

    # 每次调仓的成本 = 换手率 × 双边成本
    cost_per_rebalance = turnover_per_rebal * round_trip_cost

    # 年化调仓次数
    annual_rebalances = 252 / holding_days

    # 年化成本拖累
    annual_cost_drag = cost_per_rebalance * annual_rebalances

    net_annual = gross_annual_return - annual_cost_drag

    return {
        'round_trip_cost': round_trip_cost,
        'cost_per_rebalance': cost_per_rebalance,
        'annual_rebalances': annual_rebalances,
        'annual_cost_drag': annual_cost_drag,
        'gross_annual_return': gross_annual_return,
        'net_annual_return': net_annual,
    }


# ═══════════════════════════════════════════════════════
# Layer 3: 风险控制指标
# ═══════════════════════════════════════════════════════

def compute_drawdown_series(returns: pd.Series) -> pd.Series:
    """计算回撤序列"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown


def compute_risk_metrics(daily_returns: pd.Series, risk_free_rate: float = 0.02) -> Dict:
    """
    计算全套风险指标

    Args:
        daily_returns: 日收益率序列
        risk_free_rate: 年化无风险利率 (默认2%, 中国国债)

    Returns:
        dict with sharpe, sortino, calmar, max_drawdown, etc.
    """
    if len(daily_returns) < 10:
        return {k: 0 for k in [
            'annual_return', 'annual_volatility', 'downside_volatility',
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
            'max_drawdown', 'max_dd_duration_days', 'max_dd_recovery_days',
            'var_95', 'cvar_95', 'omega_ratio',
            'monthly_win_rate', 'worst_month', 'best_month',
            'n_trading_days', 'positive_day_pct',
        ]}

    returns = daily_returns.dropna()
    n_days = len(returns)

    # 基础收益
    cumulative_return = (1 + returns).prod() - 1
    annual_return = (1 + cumulative_return) ** (252 / n_days) - 1

    # 波动率
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    excess_returns = returns - daily_rf
    annual_volatility = excess_returns.std() * np.sqrt(252)

    # 下行波动率 (仅负超额收益)
    negative_excess = excess_returns[excess_returns < 0]
    downside_volatility = negative_excess.std() * np.sqrt(252) if len(negative_excess) > 0 else 1e-8

    # Sharpe Ratio
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 1e-8 else 0

    # Sortino Ratio
    sortino_ratio = (annual_return - risk_free_rate) / downside_volatility if downside_volatility > 1e-8 else 0

    # 回撤分析
    drawdown = compute_drawdown_series(returns)
    max_drawdown = drawdown.min()  # 最大回撤（负值）

    # 回撤持续期和恢复时间
    dd_duration, dd_recovery = _compute_dd_duration(returns)

    # Calmar Ratio
    calmar_ratio = annual_return / abs(max_drawdown) if abs(max_drawdown) > 1e-8 else 0

    # VaR and CVaR (95%)
    var_95 = np.percentile(returns, 5)  # 5th percentile = 95% VaR
    cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95

    # Omega Ratio
    threshold = daily_rf
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns <= threshold]
    omega_ratio = gains.sum() / losses.sum() if losses.sum() > 1e-8 else float('inf')

    # 月度统计
    returns_with_index = returns.copy()
    if not isinstance(returns_with_index.index, pd.DatetimeIndex):
        try:
            returns_with_index.index = pd.to_datetime(returns_with_index.index)
        except Exception:
            # 用简单序列索引
            pass

    monthly_returns = _compute_monthly_returns(returns_with_index)
    if len(monthly_returns) > 0:
        monthly_win_rate = (monthly_returns > 0).mean() * 100
        worst_month = monthly_returns.min()
        best_month = monthly_returns.max()
        max_consecutive_loss_months = _max_consecutive(monthly_returns < 0)
    else:
        monthly_win_rate = 0
        worst_month = 0
        best_month = 0
        max_consecutive_loss_months = 0

    # 日度胜率
    positive_day_pct = (returns > 0).mean() * 100

    return {
        'annual_return': annual_return,
        'cumulative_return': cumulative_return,
        'annual_volatility': annual_volatility,
        'downside_volatility': downside_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown,
        'max_dd_duration_days': dd_duration,
        'max_dd_recovery_days': dd_recovery,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'omega_ratio': omega_ratio,
        'monthly_win_rate': monthly_win_rate,
        'worst_month': worst_month,
        'best_month': best_month,
        'max_consecutive_loss_months': max_consecutive_loss_months,
        'n_trading_days': n_days,
        'positive_day_pct': positive_day_pct,
    }


def _compute_dd_duration(returns: pd.Series) -> Tuple[int, int]:
    """计算最大回撤持续天数和恢复天数"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    is_underwater = cumulative < running_max

    max_duration = 0
    current_duration = 0

    for underwater in is_underwater:
        if underwater:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0

    # 恢复时间：从最大回撤点到回到前高的天数
    drawdown = (cumulative - running_max) / running_max
    if drawdown.empty:
        return max_duration, 0

    # 使用位置索引避免非唯一DatetimeIndex的KeyError
    min_pos = drawdown.values.argmin()

    after_dd = cumulative.iloc[min_pos:]
    pre_dd_max = running_max.iloc[min_pos]
    recovered = after_dd >= pre_dd_max
    if recovered.any():
        # 找到恢复点的位置索引
        recovery_pos = recovered.values.argmax()
        recovery_days = recovery_pos + 1  # 包含起始点
    else:
        recovery_days = len(after_dd)  # 还未恢复

    return max_duration, recovery_days


def _compute_monthly_returns(returns: pd.Series) -> pd.Series:
    """计算月度收益率"""
    try:
        if isinstance(returns.index, pd.DatetimeIndex):
            monthly = (1 + returns).resample('ME').prod() - 1
            return monthly.dropna()
    except Exception:
        pass

    # fallback: 按20天分组
    n = len(returns)
    if n < 20:
        return pd.Series(dtype=float)

    monthly = []
    for i in range(0, n, 20):
        chunk = returns.iloc[i:i+20]
        monthly.append((1 + chunk).prod() - 1)

    return pd.Series(monthly)


def _max_consecutive(bool_series: pd.Series) -> int:
    """计算最长连续True的长度"""
    max_count = 0
    current = 0
    for val in bool_series:
        if val:
            current += 1
            max_count = max(max_count, current)
        else:
            current = 0
    return max_count


# ═══════════════════════════════════════════════════════
# Layer 4: 基准对比
# ═══════════════════════════════════════════════════════

def load_benchmark_returns(index_code: str = '000905.SH',
                           start_date: str = None, end_date: str = None,
                           db_path: str = None) -> pd.Series:
    """
    从数据库加载基准指数收益率

    Args:
        index_code: 指数代码 (默认中证500: 000905.SH)
            常用: 000300.SH(沪深300), 000905.SH(中证500),
                  000852.SH(中证1000), 932000.CSI(中证2000)
        start_date: 开始日期 YYYY-MM-DD
        end_date: 结束日期 YYYY-MM-DD
    """
    if db_path is None:
        db_path = DB_PATH

    conn = sqlite3.connect(db_path)

    query = """
        SELECT dq.trade_date, dq.close, dq.price_change_pct
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.code = ?
    """
    params = [index_code]

    if start_date:
        query += " AND dq.trade_date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND dq.trade_date <= ?"
        params.append(end_date)

    query += " ORDER BY dq.trade_date"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if df.empty:
        return pd.Series(dtype=float)

    # price_change_pct 已经是小数形式的日涨跌幅 (e.g., 0.0117 = 1.17%)
    returns = df.set_index('trade_date')['price_change_pct']
    returns.index = pd.to_datetime(returns.index)
    returns.name = index_code

    return returns


def compute_benchmark_comparison(portfolio_returns: pd.Series,
                                 benchmark_returns: pd.Series,
                                 periods_per_year: float = 252) -> Dict:
    """
    计算相对基准的表现

    Args:
        portfolio_returns: 策略收益率 (DatetimeIndex)
        benchmark_returns: 基准收益率 (DatetimeIndex), 频率需与portfolio一致
        periods_per_year: 年化因子 (日频=252, 10日频=25.2)
    """
    # 对齐日期
    common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
    min_required = min(10, max(3, int(periods_per_year * 0.1)))
    if len(common_dates) < min_required:
        return {
            'alpha': 0, 'beta': 0, 'tracking_error': 0,
            'information_ratio': 0, 'excess_annual_return': 0,
        }

    p_ret = portfolio_returns.loc[common_dates]
    b_ret = benchmark_returns.loc[common_dates]

    # Alpha / Beta (CAPM回归)
    cov = np.cov(p_ret, b_ret)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 1e-10 else 0

    excess = p_ret - b_ret
    alpha_per_period = excess.mean()
    alpha_annual = alpha_per_period * periods_per_year

    # Tracking Error
    tracking_error = excess.std() * np.sqrt(periods_per_year)

    # Information Ratio
    information_ratio = alpha_annual / tracking_error if tracking_error > 1e-8 else 0

    # 超额年化收益
    n_periods = len(common_dates)
    total_equivalent_days = n_periods * (252 / periods_per_year)
    p_annual = (1 + p_ret).prod() ** (252 / total_equivalent_days) - 1
    b_annual = (1 + b_ret).prod() ** (252 / total_equivalent_days) - 1
    excess_annual = p_annual - b_annual

    # 超额收益胜率
    excess_win_rate = (excess > 0).mean() * 100

    return {
        'alpha': alpha_annual,
        'beta': beta,
        'tracking_error': tracking_error,
        'information_ratio': information_ratio,
        'excess_annual_return': excess_annual,
        'portfolio_annual': p_annual,
        'benchmark_annual': b_annual,
        'excess_win_rate': excess_win_rate,
        'n_common_periods': n_periods,
    }


# ═══════════════════════════════════════════════════════
# V2 批量数据加载 + 可执行性指标
# ═══════════════════════════════════════════════════════

def batch_load_market_cap_data(buy_dates: List[str], db_path: str = None) -> Dict[str, pd.DataFrame]:
    """
    批量加载市值和换手率数据

    Args:
        buy_dates: 买入日期列表 (YYYY-MM-DD)
        db_path: 数据库路径

    Returns:
        {date_str: DataFrame[code, total_mv, turnover_rate]}
        total_mv单位: 万元 (数据库原始单位)
    """
    if db_path is None:
        db_path = DB_PATH

    if not buy_dates:
        return {}

    conn = sqlite3.connect(db_path)
    placeholders = ','.join(['?' for _ in buy_dates])
    query = f"""
        SELECT db.trade_date, s.code, db.total_mv, db.turnover_rate
        FROM daily_basic db
        JOIN securities s ON db.security_id = s.id
        WHERE db.trade_date IN ({placeholders})
          AND db.total_mv IS NOT NULL AND db.total_mv > 0
    """
    df = pd.read_sql_query(query, conn, params=list(buy_dates))
    conn.close()

    result = {}
    for date, group in df.groupby('trade_date'):
        result[date] = group[['code', 'total_mv', 'turnover_rate']].reset_index(drop=True)
    return result


def batch_load_limit_up_data(buy_dates: List[str], db_path: str = None) -> Dict[str, set]:
    """
    批量加载涨停股票数据

    涨停判定 (小数格式): 主板≥0.095, 创业板(30x)/科创板(688x)≥0.195, 北交所(8x)≥0.295

    Args:
        buy_dates: 买入日期列表
        db_path: 数据库路径

    Returns:
        {date_str: set(涨停codes)}
    """
    if db_path is None:
        db_path = DB_PATH

    if not buy_dates:
        return {}

    conn = sqlite3.connect(db_path)
    placeholders = ','.join(['?' for _ in buy_dates])
    query = f"""
        SELECT dq.trade_date, s.code, dq.price_change_pct
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE dq.trade_date IN ({placeholders})
          AND dq.price_change_pct IS NOT NULL
          AND s.type = 'A股'
    """
    df = pd.read_sql_query(query, conn, params=list(buy_dates))
    conn.close()

    result = {}
    if df.empty:
        return result

    # 向量化涨停检测
    conditions = [
        df['code'].str.startswith('30') | df['code'].str.startswith('688'),
        df['code'].str.startswith('8'),
    ]
    thresholds = [0.195, 0.295]
    df['threshold'] = np.select(conditions, thresholds, default=0.095)
    limit_up_df = df[df['price_change_pct'] >= df['threshold']]

    for date, group in limit_up_df.groupby('trade_date'):
        result[date] = set(group['code'].tolist())

    # 确保所有日期都有条目（即使无涨停）
    for date in df['trade_date'].unique():
        if date not in result:
            result[date] = set()

    return result


def batch_load_universe_median_cap(buy_dates: List[str], db_path: str = None) -> Dict[str, float]:
    """
    批量加载全A股当日市值中位数

    Args:
        buy_dates: 买入日期列表

    Returns:
        {date_str: median_total_mv_万元}
    """
    if db_path is None:
        db_path = DB_PATH

    if not buy_dates:
        return {}

    conn = sqlite3.connect(db_path)
    placeholders = ','.join(['?' for _ in buy_dates])
    query = f"""
        SELECT db.trade_date, db.total_mv
        FROM daily_basic db
        JOIN securities s ON db.security_id = s.id
        WHERE db.trade_date IN ({placeholders})
          AND db.total_mv IS NOT NULL AND db.total_mv > 0
          AND s.type = 'A股'
    """
    df = pd.read_sql_query(query, conn, params=list(buy_dates))
    conn.close()

    result = {}
    for date, group in df.groupby('trade_date'):
        result[date] = group['total_mv'].median()
    return result


def compute_executability_metrics(holdings_by_date: Dict[str, List[str]],
                                   market_cap_data: Dict[str, pd.DataFrame],
                                   limit_up_data: Dict[str, set],
                                   universe_median_cap: Dict[str, float]) -> Dict:
    """
    计算实盘可执行性指标

    Args:
        holdings_by_date: {date: [stock_codes]}
        market_cap_data: batch_load_market_cap_data的输出
        limit_up_data: batch_load_limit_up_data的输出
        universe_median_cap: batch_load_universe_median_cap的输出

    Returns:
        dict{limit_up_fail_rate, liquidity_coverage, cap_balance_ratio, median_market_cap_bn}
    """
    total_picks = 0
    limit_up_picks = 0
    liquid_picks = 0  # 换手率>0.5%的股票
    small_cap_picks = 0  # 市值小于全市场中位数的股票
    all_market_caps = []

    for date, codes in holdings_by_date.items():
        if not codes:
            continue

        cap_df = market_cap_data.get(date, pd.DataFrame())
        limit_up_set = limit_up_data.get(date, set())
        median_cap = universe_median_cap.get(date, 0)

        for code in codes:
            total_picks += 1

            # 涨停检测
            if code in limit_up_set:
                limit_up_picks += 1

            # 市值和流动性
            if not cap_df.empty:
                stock_row = cap_df[cap_df['code'] == code]
                if len(stock_row) > 0:
                    mv = stock_row.iloc[0]['total_mv']
                    tr = stock_row.iloc[0].get('turnover_rate', 0) or 0
                    all_market_caps.append(mv)

                    # 流动性: 换手率 > 0.5%
                    if tr > 0.5:
                        liquid_picks += 1

                    # 小盘偏好: 市值 < 全市场中位数
                    if median_cap > 0 and mv < median_cap:
                        small_cap_picks += 1

    if total_picks == 0:
        return {
            'limit_up_fail_rate': None,
            'liquidity_coverage': None,
            'cap_balance_ratio': None,
            'median_market_cap_bn': None,
        }

    limit_up_rate = limit_up_picks / total_picks
    liquidity_coverage = liquid_picks / total_picks if total_picks > 0 else 0
    small_cap_ratio = small_cap_picks / total_picks if total_picks > 0 else 0

    # 市值均衡度: 1 - abs(small_cap_ratio - 0.5) * 2, 越接近50%越好
    cap_balance = 1.0 - abs(small_cap_ratio - 0.5) * 2

    # 中位市值 (万元→亿元)
    median_cap_bn = np.median(all_market_caps) / 10000 if all_market_caps else 0

    return {
        'limit_up_fail_rate': limit_up_rate,
        'liquidity_coverage': liquidity_coverage,
        'cap_balance_ratio': cap_balance,
        'median_market_cap_bn': median_cap_bn,
    }


# ═══════════════════════════════════════════════════════
# 市况分类 + 条件IC (Phase 2)
# ═══════════════════════════════════════════════════════

def classify_market_regime(benchmark_returns: pd.Series,
                            lookback: int = 60) -> pd.Series:
    """
    基于滚动收益分类市况

    Args:
        benchmark_returns: 日度基准收益率 (DatetimeIndex)
        lookback: 回看天数

    Returns:
        Series of regime labels: 'bull', 'bear', 'neutral'
    """
    if benchmark_returns.empty or len(benchmark_returns) < lookback:
        return pd.Series(dtype=str)

    rolling_ret = benchmark_returns.rolling(lookback, min_periods=lookback // 2).apply(
        lambda x: (1 + x).prod() - 1, raw=False
    )

    def _classify(r):
        if pd.isna(r):
            return 'neutral'
        if r > 0.05:
            return 'bull'
        elif r < -0.05:
            return 'bear'
        return 'neutral'

    return rolling_ret.map(_classify)


def compute_regime_conditional_metrics(ic_df: pd.DataFrame,
                                        regime_series: pd.Series) -> Dict:
    """
    分市况计算IC/ICIR

    Args:
        ic_df: compute_daily_ic的输出 (columns: date, ic)
        regime_series: classify_market_regime的输出

    Returns:
        dict{bull: {ic, icir, n}, bear: {...}, neutral: {...}}
    """
    if ic_df.empty or regime_series.empty:
        return {}

    ic_with_date = ic_df.copy()
    ic_with_date['date_dt'] = pd.to_datetime(ic_with_date['date'])
    ic_with_date = ic_with_date.set_index('date_dt')

    # 对齐regime到ic_df的日期
    common_idx = ic_with_date.index.intersection(regime_series.index)
    if len(common_idx) < 10:
        return {}

    result = {}
    for regime in ['bull', 'bear', 'neutral']:
        regime_dates = regime_series[regime_series == regime].index
        regime_ic = ic_with_date.loc[ic_with_date.index.isin(regime_dates), 'ic']

        if len(regime_ic) < 5:
            result[regime] = {'ic': 0, 'icir': 0, 'n_days': 0}
            continue

        ic_mean = regime_ic.mean()
        ic_std = regime_ic.std()
        icir = ic_mean / ic_std if ic_std > 1e-8 else 0

        result[regime] = {
            'ic': ic_mean,
            'icir': icir,
            'n_days': len(regime_ic),
            'ic_positive_pct': (regime_ic > 0).mean() * 100,
        }

    return result


# ═══════════════════════════════════════════════════════
# V3 北极星指标 (25项, 加权层级, 统计鲁棒性)
# ═══════════════════════════════════════════════════════

V3_LAYER_WEIGHTS = {1: 0.40, 2: 0.20, 3: 0.25, 4: 0.15}
V3_LAYER_NAMES = {1: '信号质量', 2: '组合效率', 3: '风险控制', 4: '统计鲁棒性'}

NORTH_STAR_TARGETS_V3 = {
    # ── Layer 1: 信号质量 (8 metrics × 5pts = 40pts) ──
    'daily_ic': {
        'pass': 0.03, 'ok': 0.04, 'good': 0.05, 'great': 0.06, 'target': 0.08,
        'direction': 'higher', 'layer': 1, 'display': 'Daily IC',
    },
    'icir': {
        'pass': 0.25, 'ok': 0.35, 'good': 0.45, 'great': 0.55, 'target': 0.70,
        'direction': 'higher', 'layer': 1, 'display': 'ICIR',
    },
    'ic_positive_pct': {
        'pass': 55, 'ok': 57, 'good': 60, 'great': 63, 'target': 68,
        'direction': 'higher', 'layer': 1, 'display': 'IC>0%',
    },
    'ic_monotonicity': {
        'pass': 2.5, 'ok': 3.0, 'good': 3.5, 'great': 4.0, 'target': 4.5,
        'direction': 'higher', 'layer': 1, 'display': 'IC单调性',
    },
    'ic_time_stability': {
        'pass': 2.0, 'ok': 1.5, 'good': 1.0, 'great': 0.8, 'target': 0.6,
        'direction': 'lower', 'layer': 1, 'display': 'IC稳定性(CV)',
        'min_days': 120,
    },
    'signal_half_life': {
        'pass': 3, 'ok': 6, 'good': 8, 'great': 12, 'target': 20,
        'direction': 'higher', 'layer': 1, 'display': '信号半衰期(天)',
    },
    'bear_icir': {
        'pass': 0.05, 'ok': 0.10, 'good': 0.15, 'great': 0.25, 'target': 0.35,
        'direction': 'higher', 'layer': 1, 'display': '熊市ICIR',
        'min_days': 30,
    },
    'ic_decay_ratio': {
        'pass': 0.50, 'ok': 0.65, 'good': 0.80, 'great': 0.90, 'target': 0.95,
        'direction': 'higher', 'layer': 1, 'display': 'IC衰减比(H2/H1)',
        'min_days': 120,
    },

    # ── Layer 2: 组合效率 (5 metrics × 5pts = 25pts) ──
    'annual_turnover': {
        'pass': 45, 'ok': 35, 'good': 30, 'great': 25, 'target': 20,
        'direction': 'lower', 'layer': 2, 'display': '年化换手',
    },
    'annual_cost_drag': {
        'pass': 0.13, 'ok': 0.10, 'good': 0.08, 'great': 0.07, 'target': 0.05,
        'direction': 'lower', 'layer': 2, 'display': '年化成本',
    },
    'net_gross_ratio': {
        'pass': 0.60, 'ok': 0.70, 'good': 0.75, 'great': 0.80, 'target': 0.85,
        'direction': 'higher', 'layer': 2, 'display': '净/毛收益比',
    },
    'limit_up_fail_rate': {
        'pass': 0.15, 'ok': 0.10, 'good': 0.08, 'great': 0.05, 'target': 0.02,
        'direction': 'lower', 'layer': 2, 'display': '涨停失败率',
    },
    'liquidity_coverage': {
        'pass': 0.70, 'ok': 0.80, 'good': 0.85, 'great': 0.90, 'target': 0.95,
        'direction': 'higher', 'layer': 2, 'display': '流动性覆盖',
    },

    # ── Layer 3: 风险控制 (7 metrics × 5pts = 35pts) ──
    'max_drawdown': {
        'pass': -0.25, 'ok': -0.18, 'good': -0.12, 'great': -0.10, 'target': -0.08,
        'direction': 'higher', 'layer': 3, 'display': '最大回撤',
    },
    'sharpe_ratio': {
        'pass': 1.0, 'ok': 1.5, 'good': 2.0, 'great': 2.5, 'target': 3.0,
        'direction': 'higher', 'layer': 3, 'display': 'Sharpe',
    },
    'sortino_ratio': {
        'pass': 1.5, 'ok': 2.0, 'good': 2.5, 'great': 3.0, 'target': 4.0,
        'direction': 'higher', 'layer': 3, 'display': 'Sortino',
    },
    'calmar_ratio': {
        'pass': 1.0, 'ok': 2.0, 'good': 2.5, 'great': 3.0, 'target': 4.0,
        'direction': 'higher', 'layer': 3, 'display': 'Calmar',
    },
    'worst_rolling_60d_icir': {
        'pass': -0.10, 'ok': 0.0, 'good': 0.10, 'great': 0.20, 'target': 0.30,
        'direction': 'higher', 'layer': 3, 'display': '最差60日ICIR',
        'min_days': 120,
    },
    'tail_ratio': {
        'pass': 0.8, 'ok': 1.0, 'good': 1.2, 'great': 1.5, 'target': 2.0,
        'direction': 'higher', 'layer': 3, 'display': '尾部比率(P95/|P5|)',
    },
    'max_consecutive_loss_periods': {
        'pass': 15, 'ok': 10, 'good': 8, 'great': 5, 'target': 3,
        'direction': 'lower', 'layer': 3, 'display': '最大连续亏损期数',
    },

    # ── Layer 4: 统计鲁棒性 (5 metrics × 5pts = 25pts) ──
    'annual_return': {
        'pass': 0.15, 'ok': 0.20, 'good': 0.30, 'great': 0.40, 'target': 0.50,
        'direction': 'higher', 'layer': 4, 'display': '年化收益',
        'min_days': 200,
    },
    'monthly_win_rate': {
        'pass': 55, 'ok': 60, 'good': 67, 'great': 75, 'target': 83,
        'direction': 'higher', 'layer': 4, 'display': '月度胜率%',
    },
    'half_period_consistency': {
        'pass': 0.35, 'ok': 0.50, 'good': 0.60, 'great': 0.70, 'target': 0.80,
        'direction': 'higher', 'layer': 4, 'display': '前后半段一致性',
        'min_days': 120,
    },
    'probabilistic_sharpe': {
        'pass': 0.80, 'ok': 0.85, 'good': 0.90, 'great': 0.95, 'target': 0.99,
        'direction': 'higher', 'layer': 4, 'display': 'PSR概率Sharpe',
    },
    'deflated_sharpe': {
        'pass': 0.70, 'ok': 0.80, 'good': 0.85, 'great': 0.90, 'target': 0.95,
        'direction': 'higher', 'layer': 4, 'display': 'DSR(多重测试校正)',
    },
}


# ── V3 新计算函数 ──

def compute_probabilistic_sharpe(returns: pd.Series,
                                  benchmark_sharpe: float = 0.0,
                                  periods_per_year: float = 25.2) -> float:
    """
    Probabilistic Sharpe Ratio (Bailey & López de Prado, 2012).

    PSR = Φ((SR̂ - SR*) · √(n-1) / √(1 - γ₃·SR̂ + (γ₄-1)/4 · SR̂²))

    Returns probability that true Sharpe > benchmark_sharpe.
    """
    from scipy.stats import norm, skew, kurtosis

    returns = returns.dropna()
    n = len(returns)
    if n < 10:
        return 0.0

    # Observed Sharpe (annualized)
    mean_ret = returns.mean()
    std_ret = returns.std(ddof=1)
    if std_ret < 1e-10:
        return 0.0

    sr_hat = (mean_ret / std_ret) * np.sqrt(periods_per_year)

    # De-annualize benchmark to per-period scale for comparison
    # SR̂ and SR* must be on same scale; keep annualized
    sr_star = benchmark_sharpe

    # Skewness and excess kurtosis of returns
    gamma3 = skew(returns, bias=False)
    gamma4 = kurtosis(returns, bias=False, fisher=True)  # excess kurtosis

    # PSR denominator: adjust for non-normality
    denom_sq = 1.0 - gamma3 * sr_hat / np.sqrt(periods_per_year) + \
               (gamma4 - 1) / 4.0 * (sr_hat / np.sqrt(periods_per_year)) ** 2
    if denom_sq <= 0:
        denom_sq = 1.0  # fallback to normal assumption

    z = (sr_hat - sr_star) * np.sqrt(n - 1) / np.sqrt(denom_sq)
    psr = float(norm.cdf(z))

    return max(0.0, min(1.0, psr))


def compute_deflated_sharpe(returns: pd.Series,
                             n_trials: int = 10,
                             benchmark_sharpe: float = 0.0,
                             periods_per_year: float = 25.2) -> float:
    """
    Deflated Sharpe Ratio (Harvey & Liu, 2015).

    Corrects for multiple testing: the more strategies you try,
    the higher the expected maximum Sharpe under the null.
    Uses expected max of N iid standard normals as deflated benchmark.

    Args:
        returns: per-period return series
        n_trials: number of strategy variants tested (model iterations)
        benchmark_sharpe: base benchmark Sharpe (before deflation)
        periods_per_year: annualization factor
    """
    from scipy.stats import norm

    returns = returns.dropna()
    n = len(returns)
    if n < 10 or n_trials < 1:
        return 0.0

    std_ret = returns.std(ddof=1)
    if std_ret < 1e-10:
        return 0.0

    # Expected maximum of N iid standard normals (Euler-Mascheroni approximation)
    # E[max] ≈ (1-γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-1/(N·e))
    # where γ ≈ 0.5772 (Euler-Mascheroni constant)
    gamma_em = 0.5772156649
    if n_trials <= 1:
        e_max_sr = 0.0
    else:
        p1 = max(1e-10, 1.0 - 1.0 / n_trials)
        p2 = max(1e-10, 1.0 - 1.0 / (n_trials * np.e))
        e_max_sr = (1 - gamma_em) * norm.ppf(p1) + gamma_em * norm.ppf(p2)

    # Variance of SR estimator: Var(SR̂) ≈ (1 + SR²/2) / (n-1)
    sr_per_period = returns.mean() / std_ret
    var_sr = (1.0 + 0.5 * sr_per_period ** 2) / max(1, n - 1)

    # Deflated benchmark: inflate null by expected max × std(SR̂)
    sr_star_deflated = benchmark_sharpe + e_max_sr * np.sqrt(var_sr) * np.sqrt(periods_per_year)

    # Now compute PSR against deflated benchmark
    return compute_probabilistic_sharpe(returns, sr_star_deflated, periods_per_year)


def compute_tail_ratio(returns: pd.Series) -> float:
    """
    Tail Ratio = P95 / |P5|.

    >1.0 means upside tail is fatter than downside (positive asymmetry).
    <1.0 means downside risk dominates.
    """
    returns = returns.dropna()
    if len(returns) < 20:
        return 0.0

    p95 = np.percentile(returns, 95)
    p5 = np.percentile(returns, 5)

    if abs(p5) < 1e-10:
        return 5.0 if p95 > 0 else 0.0  # cap at 5.0

    ratio = p95 / abs(p5)
    return max(0.0, min(ratio, 5.0))  # clamp


def compute_max_consecutive_loss_periods(returns: pd.Series) -> int:
    """最大连续亏损期数 (returns < 0)."""
    returns = returns.dropna()
    if returns.empty:
        return 0
    return _max_consecutive(returns < 0)


def compute_bear_icir(ic_df: pd.DataFrame,
                       benchmark_returns: pd.Series,
                       lookback: int = 60) -> Optional[float]:
    """
    熊市ICIR: 仅在熊市期间的IC信息比率.

    Returns None if insufficient bear market data (<10 days).
    """
    if ic_df.empty or benchmark_returns.empty:
        return None

    regime = classify_market_regime(benchmark_returns, lookback)
    if regime.empty:
        return None

    regime_metrics = compute_regime_conditional_metrics(ic_df, regime)
    bear = regime_metrics.get('bear', {})
    n_bear = bear.get('n_days', 0)

    if n_bear < 10:
        return None

    return bear.get('icir', 0.0)


def compute_ic_decay_ratio(ic_df: pd.DataFrame) -> float:
    """
    IC衰减比: mean(IC_后半段) / mean(IC_前半段).

    ~1.0 = 信号稳定, <0.5 = 可能过拟合.
    Returns 0 if 前半段IC ≤ 0.
    """
    if ic_df.empty or len(ic_df) < 20:
        return 0.0

    ic_sorted = ic_df.sort_values('date')
    mid = len(ic_sorted) // 2

    ic_h1 = ic_sorted.iloc[:mid]['ic'].mean()
    ic_h2 = ic_sorted.iloc[mid:]['ic'].mean()

    if ic_h1 <= 1e-6:
        # 前半段IC≤0, 无法计算有意义的衰减比
        # 但如果后半段IC>0说明信号在改善，给1.0
        if ic_h2 > 1e-6:
            return 1.0
        return 0.0

    ratio = ic_h2 / ic_h1
    return max(0.0, min(ratio, 2.0))  # clamp to [0, 2]


def compute_cumulative_quantile_monotonicity(scores: pd.Series,
                                              returns: pd.Series,
                                              dates: pd.Series,
                                              n_quantiles: int = 5,
                                              min_stocks: int = 20) -> float:
    """
    改进版IC单调性: 基于累积分位组合收益.

    方法:
    1. 每日将股票按分数分为n_quantiles档
    2. 计算每档的日均收益
    3. 累积收益后检查最终排序是否单调
    4. 同时检查中间时段的单调保持度

    Returns: 0-5 score, 5.0 = 完美单调.
    """
    # 每个分位组收集日均收益
    quantile_daily_returns = {q: [] for q in range(n_quantiles)}

    for date in sorted(dates.unique()):
        mask = (dates == date) & scores.notna() & returns.notna()
        day_scores = scores[mask]
        day_returns = returns[mask]

        if len(day_scores) < min_stocks:
            continue

        try:
            labels = pd.qcut(day_scores, n_quantiles, labels=False, duplicates='drop')
        except ValueError:
            continue

        if labels.nunique() < n_quantiles:
            continue

        for q in range(n_quantiles):
            q_mask = labels == q
            if q_mask.any():
                quantile_daily_returns[q].append(day_returns[q_mask].mean())

    # 需要足够多的天数
    min_len = min(len(v) for v in quantile_daily_returns.values())
    if min_len < 20:
        return 0.0

    # 截断到相同长度, 计算累积收益
    cum_returns = {}
    for q in range(n_quantiles):
        rets = np.array(quantile_daily_returns[q][:min_len])
        cum_returns[q] = np.cumprod(1 + rets)

    # 1) 最终累积收益的单调性 (权重60%)
    final_values = [cum_returns[q][-1] for q in range(n_quantiles)]
    n_pairs = n_quantiles - 1
    correct_final = sum(1 for i in range(n_pairs) if final_values[i+1] > final_values[i])
    final_mono = correct_final / n_pairs

    # 2) 中间检查点的单调保持度 (权重40%)
    checkpoints = [min_len // 4, min_len // 2, 3 * min_len // 4]
    checkpoint_scores = []
    for cp in checkpoints:
        if cp < 1:
            continue
        cp_values = [cum_returns[q][cp] for q in range(n_quantiles)]
        correct_cp = sum(1 for i in range(n_pairs) if cp_values[i+1] > cp_values[i])
        checkpoint_scores.append(correct_cp / n_pairs)

    avg_checkpoint = np.mean(checkpoint_scores) if checkpoint_scores else 0

    # 综合分数
    combined = final_mono * 0.6 + avg_checkpoint * 0.4
    return combined * 5.0


def compute_backtest_length_factor(n_days: int, min_days: int = 500) -> float:
    """
    回测长度折扣因子: min(1.0, sqrt(n_days / min_days)).

    500天以上无惩罚, 125天(半年)打5折, 250天(1年)打7折.
    """
    if n_days <= 0:
        return 0.0
    if n_days >= min_days:
        return 1.0
    return np.sqrt(n_days / min_days)


# ── V3 评分函数 ──

def score_metric_v3(current: float, target_info: dict) -> Tuple[int, str]:
    """V3评分 (与V2相同的0-5档逻辑)."""
    return score_metric_v2(current, target_info)


def compute_v3_score(metric_values: Dict[str, float],
                      n_trading_days: int = 500,
                      n_trials: int = 10) -> Dict:
    """
    V3加权评分.

    V3 vs V2区别:
    1. 层级加权: L1=40%, L2=20%, L3=25%, L4=15%
    2. 加权百分比 = Σ(layer_score/layer_max × layer_weight) × 100
    3. 回测长度惩罚: final_pct *= sqrt(n_days/500)

    Returns:
        dict{total_score, max_score, raw_pct, length_factor, final_pct,
             grade, layer_details: {layer: {score,max,pct,weight}},
             metric_scores: {metric: (score, grade_str, value)}}
    """
    layer_scores = {1: 0, 2: 0, 3: 0, 4: 0}
    layer_maxes = {1: 0, 2: 0, 3: 0, 4: 0}
    metric_scores = {}

    for metric_name, target_info in NORTH_STAR_TARGETS_V3.items():
        layer = target_info['layer']
        layer_maxes[layer] += 5

        value = metric_values.get(metric_name)
        if value is None:
            metric_scores[metric_name] = (0, '☆☆☆☆☆', None)
            continue

        score, grade_str = score_metric_v3(value, target_info)
        layer_scores[layer] += score
        metric_scores[metric_name] = (score, grade_str, value)

    # 加权百分比
    weighted_pct = 0.0
    layer_details = {}
    for layer in [1, 2, 3, 4]:
        lmax = layer_maxes[layer]
        lscore = layer_scores[layer]
        lpct = lscore / lmax * 100 if lmax > 0 else 0
        weight = V3_LAYER_WEIGHTS[layer]
        weighted_pct += lpct * weight
        layer_details[layer] = {
            'score': lscore, 'max': lmax,
            'pct': lpct, 'weight': weight,
        }

    # 回测长度惩罚
    length_factor = compute_backtest_length_factor(n_trading_days)
    final_pct = weighted_pct * length_factor

    total_score = sum(layer_scores.values())
    max_score = sum(layer_maxes.values())

    grade = compute_v3_grade(final_pct)

    return {
        'total_score': total_score,
        'max_score': max_score,
        'raw_pct': weighted_pct,
        'length_factor': length_factor,
        'final_pct': final_pct,
        'grade': grade,
        'layer_details': layer_details,
        'metric_scores': metric_scores,
    }


def compute_v3_grade(pct: float) -> str:
    """V3等级 (与V2相同阈值)."""
    for threshold, grade in V2_GRADE_THRESHOLDS:
        if pct >= threshold:
            return grade
    return 'D'


# ═══════════════════════════════════════════════════════
# 综合评估器
# ═══════════════════════════════════════════════════════

class NorthStarEvaluator:
    """
    北极星指标综合评估器

    用法:
        evaluator = NorthStarEvaluator()

        # 信号质量评估
        ic_df = evaluator.evaluate_signal_quality(scores, returns, dates)

        # 组合风险收益评估
        risk_report = evaluator.evaluate_portfolio(daily_returns)

        # 完整评估（含基准对比）
        full_report = evaluator.full_evaluation(
            daily_returns, scores, actual_returns, dates,
            benchmark_code='000905.SH'
        )

        # 打印评分卡
        evaluator.print_scorecard(full_report)
    """

    def __init__(self, db_path: str = None, risk_free_rate: float = 0.02):
        self.db_path = db_path or DB_PATH
        self.risk_free_rate = risk_free_rate

    def evaluate_signal_quality(self, scores: pd.Series, returns: pd.Series,
                                dates: pd.Series, holding_days: int = 5) -> Dict:
        """Layer 1: 评估信号质量"""
        ic_df = compute_daily_ic(scores, returns, dates)
        ic_summary = compute_ic_summary(ic_df)
        monthly_ic = compute_monthly_ic(ic_df)
        spread = compute_top_bottom_spread(scores, returns, dates, quantile=0.1)

        return {
            'holding_days': holding_days,
            'ic_df': ic_df,
            'ic_summary': ic_summary,
            'monthly_ic': monthly_ic,
            'top_bottom_spread': spread,
        }

    def evaluate_portfolio(self, daily_returns: pd.Series,
                           dates: pd.Series = None) -> Dict:
        """Layer 3+4: 评估组合风险收益"""
        risk = compute_risk_metrics(daily_returns, self.risk_free_rate)
        return risk

    def evaluate_with_benchmark(self, daily_returns: pd.Series,
                                benchmark_code: str = '000905.SH',
                                start_date: str = None,
                                end_date: str = None) -> Dict:
        """Layer 4: 基准对比"""
        benchmark = load_benchmark_returns(
            benchmark_code, start_date, end_date, self.db_path
        )
        if benchmark.empty:
            return {}

        return compute_benchmark_comparison(daily_returns, benchmark)

    def full_evaluation(self, daily_returns: pd.Series,
                        scores: pd.Series = None,
                        actual_returns: pd.Series = None,
                        dates: pd.Series = None,
                        holdings_by_date: Dict = None,
                        holding_days: int = 10,
                        benchmark_code: str = '000905.SH') -> Dict:
        """
        完整的北极星评估

        Returns:
            dict with all metrics organized by layer
        """
        report = {
            'layer1_signal': {},
            'layer2_portfolio': {},
            'layer3_risk': {},
            'layer4_pnl': {},
            'benchmark': {},
        }

        # Layer 1: 信号质量
        if scores is not None and actual_returns is not None and dates is not None:
            report['layer1_signal'] = self.evaluate_signal_quality(
                scores, actual_returns, dates, holding_days
            )

        # Layer 2: 换手率
        if holdings_by_date:
            report['layer2_portfolio'] = compute_turnover(holdings_by_date)

        # Layer 3: 风险
        if daily_returns is not None and len(daily_returns) > 0:
            report['layer3_risk'] = compute_risk_metrics(daily_returns, self.risk_free_rate)

        # Layer 4: 基准对比
        if daily_returns is not None and len(daily_returns) > 0:
            # 确保有DatetimeIndex
            if not isinstance(daily_returns.index, pd.DatetimeIndex):
                try:
                    daily_returns_dt = daily_returns.copy()
                    daily_returns_dt.index = pd.to_datetime(daily_returns_dt.index)
                except Exception:
                    daily_returns_dt = daily_returns
            else:
                daily_returns_dt = daily_returns

            benchmark = load_benchmark_returns(
                benchmark_code, db_path=self.db_path
            )
            if not benchmark.empty:
                report['benchmark'] = compute_benchmark_comparison(
                    daily_returns_dt, benchmark
                )

        # 交易成本
        if daily_returns is not None and len(daily_returns) > 0:
            turnover = report['layer2_portfolio'].get('avg_turnover', 0.5)
            report['layer4_pnl'] = compute_transaction_costs(
                daily_returns, turnover, holding_days
            )

        return report

    def print_scorecard(self, report: Dict, label: str = "模型评估"):
        """
        打印北极星评分卡

        对比当前指标与北极星目标，标记达标/未达标
        """
        print(f"\n{'═'*70}")
        print(f"  北极星评分卡: {label}")
        print(f"{'═'*70}")

        # 收集所有可用指标
        metrics_map = {}

        # Layer 1
        signal = report.get('layer1_signal', {})
        ic_summary = signal.get('ic_summary', {})
        if ic_summary:
            metrics_map['daily_ic'] = ic_summary.get('ic_mean', 0)
            metrics_map['icir'] = ic_summary.get('icir', 0)
            metrics_map['ic_positive_pct'] = ic_summary.get('ic_positive_pct', 0)

        # Layer 3
        risk = report.get('layer3_risk', {})
        if risk:
            metrics_map['max_drawdown'] = risk.get('max_drawdown', 0)
            metrics_map['sharpe_ratio'] = risk.get('sharpe_ratio', 0)
            metrics_map['sortino_ratio'] = risk.get('sortino_ratio', 0)
            metrics_map['calmar_ratio'] = risk.get('calmar_ratio', 0)
            metrics_map['annual_return'] = risk.get('annual_return', 0)
            metrics_map['monthly_win_rate'] = risk.get('monthly_win_rate', 0)

        # Layer 4
        pnl = report.get('layer4_pnl', {})
        if pnl:
            metrics_map['annual_cost_drag'] = pnl.get('annual_cost_drag', 0)

        portfolio = report.get('layer2_portfolio', {})
        if portfolio:
            metrics_map['annual_turnover'] = portfolio.get('annual_turnover_estimate', 0)

        # 打印评分表
        print(f"\n  {'指标':<20s} {'当前值':>10s} {'及格线':>10s} {'目标':>10s} {'评级':>8s}")
        print(f"  {'─'*62}")

        total_score = 0
        max_score = 0

        for metric_name, target_info in NORTH_STAR_TARGETS.items():
            if metric_name not in metrics_map:
                continue

            current = metrics_map[metric_name]
            target = target_info['target']
            pass_val = target_info['pass']
            higher = target_info['direction'] == 'higher'

            # 评级
            if higher:
                if current >= target:
                    grade = "★★★"
                    score = 3
                elif current >= target_info['good']:
                    grade = "★★☆"
                    score = 2
                elif current >= pass_val:
                    grade = "★☆☆"
                    score = 1
                else:
                    grade = "☆☆☆"
                    score = 0
            else:
                if current <= target:
                    grade = "★★★"
                    score = 3
                elif current <= target_info['good']:
                    grade = "★★☆"
                    score = 2
                elif current <= pass_val:
                    grade = "★☆☆"
                    score = 1
                else:
                    grade = "☆☆☆"
                    score = 0

            total_score += score
            max_score += 3

            # 格式化
            if metric_name in ('max_drawdown', 'annual_return', 'annual_cost_drag',
                               'gross_annual_return', 'net_annual_return'):
                current_str = f"{current:.1%}"
                target_str = f"{target:.1%}"
                pass_str = f"{pass_val:.1%}"
            elif metric_name in ('ic_positive_pct', 'monthly_win_rate', 'annual_turnover'):
                current_str = f"{current:.1f}"
                target_str = f"{target:.1f}"
                pass_str = f"{pass_val:.1f}"
            else:
                current_str = f"{current:.3f}"
                target_str = f"{target:.3f}"
                pass_str = f"{pass_val:.3f}"

            display_name = {
                'daily_ic': 'Daily IC',
                'icir': 'ICIR',
                'ic_positive_pct': 'IC>0%',
                'rank_ic': 'Rank IC',
                'max_drawdown': '最大回撤',
                'sharpe_ratio': 'Sharpe',
                'sortino_ratio': 'Sortino',
                'calmar_ratio': 'Calmar',
                'annual_return': '年化收益',
                'monthly_win_rate': '月度胜率%',
                'annual_turnover': '年化换手',
                'annual_cost_drag': '年化成本',
            }.get(metric_name, metric_name)

            print(f"  {display_name:<20s} {current_str:>10s} {pass_str:>10s} {target_str:>10s} {grade:>8s}")

        # 总分
        if max_score > 0:
            pct = total_score / max_score * 100
            print(f"\n  综合评分: {total_score}/{max_score} ({pct:.0f}%)")

            if pct >= 80:
                print(f"  等级: A+ (优秀，接近北极星目标)")
            elif pct >= 60:
                print(f"  等级: A  (良好，大部分指标达标)")
            elif pct >= 40:
                print(f"  等级: B  (及格，仍有较大提升空间)")
            else:
                print(f"  等级: C  (不及格，需重点改进)")

        # 基准对比
        bm = report.get('benchmark', {})
        if bm:
            print(f"\n  基准对比:")
            print(f"    策略年化:     {bm.get('portfolio_annual', 0):.1%}")
            print(f"    基准年化:     {bm.get('benchmark_annual', 0):.1%}")
            print(f"    超额收益:     {bm.get('excess_annual_return', 0):.1%}")
            print(f"    Alpha:        {bm.get('alpha', 0):.1%}")
            print(f"    Beta:         {bm.get('beta', 0):.3f}")
            print(f"    信息比率(IR):  {bm.get('information_ratio', 0):.3f}")
            print(f"    跟踪误差:     {bm.get('tracking_error', 0):.1%}")

        print(f"{'═'*70}\n")


def format_signal_report(signal_results: Dict, holding_days: int = None) -> str:
    """格式化信号质量报告（训练脚本可调用）"""
    lines = []
    ic_s = signal_results.get('ic_summary', {})
    spread = signal_results.get('top_bottom_spread', {})
    hd = holding_days or signal_results.get('holding_days', '?')

    lines.append(f"\n  信号质量评估 ({hd}d持仓):")
    lines.append(f"    Daily IC均值:   {ic_s.get('ic_mean', 0):.4f} ± {ic_s.get('ic_std', 0):.4f}")
    lines.append(f"    ICIR:           {ic_s.get('icir', 0):.4f}")
    lines.append(f"    IC>0占比:       {ic_s.get('ic_positive_pct', 0):.1f}% ({ic_s.get('n_days', 0)}天)")
    lines.append(f"    Top10%-Bottom10% Spread: {spread.get('spread', 0):.4f}")

    # 月度IC
    monthly = signal_results.get('monthly_ic')
    if monthly is not None and len(monthly) > 0:
        lines.append(f"    月度IC分解:")
        for _, row in monthly.iterrows():
            lines.append(
                f"      {row['month']}: IC={row['ic_mean']:+.4f}, "
                f"ICIR={row['icir']:+.3f}, IC>0={row['ic_positive_pct']:.0f}%"
            )

    return '\n'.join(lines)
