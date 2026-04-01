#!/usr/bin/env python3
"""
V4.8.7 阈值优化评估脚本 — autoresearch 用

评估指标: 综合得分 = 加权(各周期 强烈买入 利润因子)
参数文件: scripts/autoresearch_v487_threshold_params.json

输出单个数字到 stdout (越大越好)
"""
import json, sys, os
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARAMS_FILE = PROJECT_ROOT / 'scripts' / 'autoresearch_v487_threshold_params.json'
DATA_FILE = '/tmp/v487_full_backtest.pkl'


def load_params():
    with open(PARAMS_FILE) as f:
        return json.load(f)


def evaluate(params):
    """
    评估阈值组合的综合表现

    优化目标: 在控制每日强烈买入数量(≤max_strong_buy_per_day)的前提下,
    最大化多周期加权利润因子
    """
    strong_buy = params['strong_buy']
    buy = params['buy']
    max_per_day = params.get('max_strong_buy_per_day', 999)  # 可选: 每日上限
    holding_weights = params.get('holding_weights', {'3d': 0.15, '5d': 0.20, '10d': 0.40, '15d': 0.25})

    df = pd.read_pickle(DATA_FILE)

    total_score = 0.0
    total_weight = 0.0
    details = {}

    for horizon, weight in holding_weights.items():
        fwd_col = f'fwd_{horizon}'
        if fwd_col not in df.columns:
            continue

        valid = df[df[fwd_col].notna()].copy()
        if len(valid) == 0:
            continue

        # Apply strong_buy threshold
        strong = valid[valid['composite'] >= strong_buy].copy()

        # Apply per-day cap (keep top-ranked per day)
        if max_per_day < 999:
            strong = strong.sort_values(['date', 'rank'])
            strong = strong.groupby('date').head(max_per_day)

        n_days_with_signal = strong['date'].nunique()
        n_total_days = valid['date'].nunique()
        signal_coverage = n_days_with_signal / max(n_total_days, 1)

        if len(strong) < 30:
            # Too few samples, penalize
            details[horizon] = {'n': len(strong), 'score': 0}
            continue

        returns = strong[fwd_col].values
        avg_ret = np.mean(returns)
        win_rate = np.mean(returns > 0)

        # Profit factor = gross_profit / gross_loss
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        gross_profit = np.sum(gains) if len(gains) > 0 else 0
        gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 1e-8
        profit_factor = gross_profit / max(gross_loss, 1e-8)

        # Per-day average count
        avg_per_day = len(strong) / max(n_total_days, 1)

        # Penalty for too many signals per day (we want ≤10)
        if avg_per_day > 10:
            count_penalty = 0.8 ** ((avg_per_day - 10) / 10)
        else:
            count_penalty = 1.0

        # Penalty for too few signal days (want ≥30% coverage)
        if signal_coverage < 0.30:
            coverage_penalty = signal_coverage / 0.30
        else:
            coverage_penalty = 1.0

        # Cap profit_factor to prevent outlier domination (15d PF=219 is unrealistic)
        capped_pf = min(profit_factor, 10.0)

        # Horizon score = capped_PF * win_rate * penalties
        # Also bonus for coverage (useful signal > rare signal)
        coverage_bonus = 1.0 + 0.5 * min(signal_coverage, 1.0)  # up to 1.5x for 100% coverage
        h_score = capped_pf * win_rate * count_penalty * coverage_penalty * coverage_bonus

        details[horizon] = {
            'n': len(strong),
            'avg_per_day': round(avg_per_day, 1),
            'avg_ret': round(avg_ret * 100, 3),
            'win_rate': round(win_rate * 100, 1),
            'profit_factor': round(profit_factor, 3),
            'signal_days_pct': round(signal_coverage * 100, 1),
            'score': round(h_score, 4),
        }

        total_score += h_score * weight
        total_weight += weight

    if total_weight > 0:
        final_score = total_score / total_weight
    else:
        final_score = 0.0

    # Also evaluate "buy" threshold for secondary metric
    buy_scores = []
    for horizon in ['5d', '10d']:
        fwd_col = f'fwd_{horizon}'
        valid = df[df[fwd_col].notna()]
        buy_stocks = valid[(valid['composite'] >= buy) & (valid['composite'] < strong_buy)]
        if len(buy_stocks) >= 30:
            wr = (buy_stocks[fwd_col] > 0).mean()
            buy_scores.append(wr)

    # Log details to stderr (not stdout)
    print(json.dumps({
        'strong_buy': strong_buy,
        'buy': buy,
        'max_per_day': max_per_day,
        'details': details,
        'final_score': round(final_score, 6),
    }, indent=2), file=sys.stderr)

    # Output single number to stdout
    print(f"{final_score:.6f}")


if __name__ == '__main__':
    params = load_params()
    evaluate(params)
