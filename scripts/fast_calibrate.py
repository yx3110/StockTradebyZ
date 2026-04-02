#!/usr/bin/env python3
"""
快速超参数校准 — 预计算技术特征 + 向量化参数扫描

核心优化: ATR/支撑/阻力只计算一次, 参数扫描只做算术运算
速度: ~1秒/组合 (vs 旧版~15秒/组合, 15x加速)

用法:
    python3 scripts/fast_calibrate.py \
        --report-dir reports/daily_selection_v4.9.0.1_extended \
        --phase all
"""

import sys
import os
import json
import copy
import itertools
import time
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

DB_PATH = os.path.join(PROJECT_ROOT, 'data_adapter', 'stock_data.db')

from portfolio_optimizer import (
    load_params, compute_atr, compute_support, compute_resistance,
    compute_entry_price, compute_stop_price, compute_target_price,
    filter_by_signal_strength, allocate_positions, PortfolioOptimizer,
)
from backtest.backtest_report_based import load_reports
from backtest.backtest_stop_target_direct import preload_all_quotes
from backtest.dynamic_sl_tp_backtest import simulate_trade_with_trailing


# ============================================================
# Phase 1: 预计算技术特征 (一次性, ~2分钟)
# ============================================================

def precompute_features(report_dir: str, start_date: str, end_date: str) -> dict:
    """
    预计算所有(股票, 日期)的技术特征: ATR, support, resistance, close, pred_10d

    Returns: {
        'reports': dict,
        'dates': list,
        'features': dict[(code, date)] = {atr, support, resistance, close, pred_10d, is_wide, composite},
        'all_quotes': DataFrame,
        'env_scores': dict[date] = float,
    }
    """
    print("Phase 1: 预计算技术特征...")
    t0 = time.time()

    # 加载报告
    reports = load_reports(report_dir, rank_field='composite')
    dates = sorted(reports.keys())
    if start_date:
        dates = [d for d in dates if d >= start_date]
    if end_date:
        dates = [d for d in dates if d <= end_date]

    if not dates:
        print("  无可用报告")
        return {}

    print(f"  报告: {len(dates)}天, {dates[0]} -> {dates[-1]}")

    # 预加载日线
    all_quotes = preload_all_quotes(dates[0], '2026-12-31')

    # 找lookback起点
    conn = sqlite3.connect(DB_PATH, timeout=30)
    lookback_dates = [r[0] for r in conn.execute("""
        SELECT DISTINCT trade_date FROM daily_quotes
        WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT 80
    """, (dates[0],)).fetchall()]
    lookback_start = lookback_dates[-1] if lookback_dates else dates[0]
    conn.close()

    kline_quotes = preload_all_quotes(lookback_start, '2026-12-31')

    # 预加载env_scores
    env_scores = {}
    for date in dates:
        json_path = Path(report_dir) / f"analysis_data_{date.replace('-','')}.json"
        if json_path.exists():
            try:
                with open(json_path) as f:
                    data = json.load(f)
                te = data.get('trading_environment', {})
                env_scores[date] = te.get('total_score', 50.0)
            except Exception:
                env_scores[date] = 50.0
        else:
            env_scores[date] = 50.0

    # 核心: 预计算每个(code, date)的技术特征
    features = {}
    n_computed = 0

    for date in dates:
        stocks = reports[date]
        if not stocks:
            continue

        for s in stocks:
            code = s.get('code', '')
            if not code:
                continue

            try:
                kline = kline_quotes.loc[code]
                mask = kline.index <= date
                kline_before = kline[mask].tail(80)
                if len(kline_before) < 20:
                    continue

                highs = kline_before['high'].values
                lows = kline_before['low'].values
                closes = kline_before['close'].values
                close = closes[-1]

                atr = compute_atr(highs, lows, closes, period=20)
                support = compute_support(highs, lows, closes)
                resistance = compute_resistance(highs, closes)

                is_wide = code.startswith('30') or code.startswith('688')

                features[(code, date)] = {
                    'atr': atr,
                    'atr_pct': atr / close if close > 0 else 0.03,
                    'support': support,
                    'resistance': resistance,
                    'close': close,
                    'pred_10d': s.get('pred_10d', 0) or 0,
                    'is_wide': is_wide,
                    'composite': s.get('rank_score', s.get('score', 0)),
                }
                n_computed += 1
            except (KeyError, IndexError):
                continue

        if len(features) % 10000 == 0 and len(features) > 0:
            print(f"  已计算: {len(features)} 条特征...")

    elapsed = time.time() - t0
    print(f"  预计算完成: {n_computed} 条特征, {elapsed:.1f}秒")

    return {
        'reports': reports,
        'dates': dates,
        'features': features,
        'all_quotes': all_quotes,
        'env_scores': env_scores,
    }


# ============================================================
# Phase 2: 快速回测 (仅用预计算特征, ~1秒/组合)
# ============================================================

def fast_backtest(precomputed: dict, params: dict) -> dict:
    """
    用预计算特征快速回测一组参数

    不碰kline数据, 只做:
    1. 从precomputed features算买入价/止损/目标 (算术)
    2. 信号筛选+仓位分配
    3. 交易模拟 (用preloaded all_quotes)
    """
    reports = precomputed['reports']
    dates = precomputed['dates']
    features = precomputed['features']
    all_quotes = precomputed['all_quotes']
    env_scores = precomputed['env_scores']

    all_trades = []
    trade_cost = 0.00302
    max_hold = params['hold']['max_hold_days']

    for date in dates:
        stocks = reports[date]
        if not stocks:
            continue

        env_score = env_scores.get(date, 50.0)

        # 用预计算特征计算价格 (纯算术, 极快)
        priced_stocks = []
        for s in stocks:
            code = s.get('code', '')
            feat = features.get((code, date))
            if feat is None:
                continue

            close = feat['close']
            atr = feat['atr']
            support = feat['support']
            resistance = feat['resistance']
            pred_10d = feat['pred_10d']
            is_wide = feat['is_wide']

            buy = compute_entry_price(close, atr, support, pred_10d, params)
            stop = compute_stop_price(buy, close, atr, support, env_score, is_wide, params)
            target = compute_target_price(buy, stop, close, resistance, pred_10d, params)

            priced_stocks.append({
                'code': code,
                'buy_price': buy,
                'stop_loss': stop,
                'target': target,
                'composite': feat['composite'],
                'atr_pct': feat['atr_pct'],
                'pred_10d': pred_10d,
            })

        # 信号筛选+仓位
        selected = filter_by_signal_strength(priced_stocks, env_score, params)
        selected = allocate_positions(selected, env_score)

        # 交易模拟
        for s in selected:
            code = s.get('code', '')
            result = simulate_trade_with_trailing(
                code, date,
                s.get('buy_price', 0),
                s.get('stop_loss', 0),
                s.get('target', 0),
                all_quotes, params,
            )
            if result and result.get('outcome') != 'no_fill':
                result['code'] = code
                result['date'] = date
                result['position_pct'] = s.get('position_pct', 10.0)
                result['trade_return_net'] = result['trade_return'] - trade_cost
                all_trades.append(result)

    if not all_trades:
        return {'metrics': {'sharpe': 0, 'max_drawdown': -1, 'annual_return': 0}}

    trades_df = pd.DataFrame(all_trades)

    # 快速计算核心指标
    trades_df['weighted_return'] = trades_df['trade_return_net'] * trades_df['position_pct'] / 100
    daily_returns = trades_df.groupby('date')['weighted_return'].sum()

    n_periods = len(dates)
    periods_per_year = 252 / max(max_hold, 1)
    total_return = (1 + daily_returns).prod() - 1
    annual_return = (1 + total_return) ** (periods_per_year / max(n_periods, 1)) - 1

    if len(daily_returns) > 1:
        mean_ret = daily_returns.mean()
        std_ret = daily_returns.std()
        sharpe = (mean_ret * periods_per_year - 0.02) / (std_ret * np.sqrt(periods_per_year)) if std_ret > 0 else 0
    else:
        sharpe = 0

    cumulative = (1 + daily_returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_dd = float(drawdown.min())

    exit_counts = trades_df['outcome'].value_counts(normalize=True).to_dict()

    return {
        'metrics': {
            'annual_return': annual_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'n_trades': len(all_trades),
            'win_rate': float((trades_df['trade_return_net'] > 0).mean()),
            'avg_hold_days': float(trades_df['hold_days'].mean()),
            'stop_loss_rate': exit_counts.get('stop_loss', 0),
            'take_profit_rate': exit_counts.get('take_profit', 0),
        }
    }


# ============================================================
# Phase 3: 网格搜索
# ============================================================

SEARCH_GRIDS = {
    'stop': {
        'stop.atr_multiplier': [1.5, 2.0, 2.5, 3.0, 4.0],
        'stop.env_mult_bullish': [0.85, 1.0],
        'stop.env_mult_bearish': [1.0, 1.3],
        'stop.min_stop_pct': [0.02, 0.03, 0.05],
        'stop.max_stop_main': [0.08, 0.10, 0.15],
    },  # 180 combos
    'target': {
        'target.min_rr_ratio': [1.5, 2.0, 2.5, 3.0],
        'target.target_clip_min': [0.02, 0.03, 0.05],
        'target.target_clip_max': [0.10, 0.15, 0.20],
    },  # 36
    'entry': {
        'entry.atr_discount_mult': [0.0, 0.3, 0.5, 1.0],
        'entry.support_discount_mult': [0.0, 0.2, 0.5],
        'entry.ml_bullish_mult': [0.3, 0.5, 1.0],
        'entry.ml_bearish_mult': [1.0, 1.5, 2.0],
    },  # 108
    'filter': {
        'filter.composite_cutoff': [0, 0.0005, 0.001],
        'filter.min_n': [3, 5, 10],
        'filter.max_n_bull': [10, 15, 20],
        'filter.max_n_bear': [3, 5, 8],
    },  # 81
    'trailing': {
        'trailing.trailing_trigger_pct': [0.3, 0.5, 0.6, 0.8],
        'trailing.trailing_fallback_pct': [0.2, 0.3, 0.4, 0.5],
    },  # 16
    'hold': {
        'hold.max_hold_days': [5, 10, 15, 20, 30],
    },  # 5
}


def set_nested(d: dict, dotted_key: str, value):
    keys = dotted_key.split('.')
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def objective(metrics: dict) -> float:
    sharpe = metrics.get('sharpe', 0)
    max_dd = abs(metrics.get('max_drawdown', 0))
    return sharpe * (1 - max_dd)


def calibrate_phase(phase: str, grid: dict, base_params: dict,
                    is_precomp: dict, oos_precomp: dict) -> Tuple[dict, list]:
    """一个阶段的网格搜索"""
    param_names = list(grid.keys())
    param_values = list(grid.values())
    combos = list(itertools.product(*param_values))

    print(f"\n{'='*60}")
    print(f"  校准: {phase} ({len(combos)} 组合)")

    results = []
    t_start = time.time()

    for i, combo in enumerate(combos):
        params = copy.deepcopy(base_params)
        for name, val in zip(param_names, combo):
            set_nested(params, name, val)

        t0 = time.time()
        result = fast_backtest(is_precomp, params)
        elapsed = time.time() - t0

        obj = objective(result['metrics'])
        results.append({
            'combo': dict(zip(param_names, combo)),
            'metrics': result['metrics'],
            'objective': obj,
            'time': elapsed,
        })

        if (i + 1) % 20 == 0 or i == len(combos) - 1:
            best = max(r['objective'] for r in results)
            avg_t = (time.time() - t_start) / (i + 1)
            eta = avg_t * (len(combos) - i - 1)
            print(f"  [{i+1}/{len(combos)}] best_obj={best:.4f} "
                  f"({elapsed:.1f}s/combo, ETA {eta:.0f}s)")

    if not results:
        return base_params, []

    # Top 3 → OOS验证
    results.sort(key=lambda r: r['objective'], reverse=True)
    top3 = results[:3]

    print(f"\n  IS Top 3:")
    for j, r in enumerate(top3):
        m = r['metrics']
        print(f"    #{j+1}: obj={r['objective']:.4f}, "
              f"Sharpe={m.get('sharpe',0):.3f}, "
              f"MaxDD={m.get('max_drawdown',0):.1%}, "
              f"年化={m.get('annual_return',0):.1%}, "
              f"止损率={m.get('stop_loss_rate',0):.0%}")

    best_oos = None
    best_oos_obj = -999

    if oos_precomp:
        for j, r in enumerate(top3):
            params = copy.deepcopy(base_params)
            for name, val in r['combo'].items():
                set_nested(params, name, val)

            oos_result = fast_backtest(oos_precomp, params)
            oos_obj = objective(oos_result['metrics'])
            is_obj = r['objective']
            ratio = oos_obj / is_obj if is_obj > 0 else 0

            print(f"    OOS #{j+1}: obj={oos_obj:.4f} (IS={is_obj:.4f}, ratio={ratio:.0%})")

            if is_obj > 0 and oos_obj < is_obj * 0.6:
                print(f"    ⚠️ 过拟合!")
                continue

            if oos_obj > best_oos_obj:
                best_oos_obj = oos_obj
                best_oos = r['combo']

    # fallback
    if best_oos is None:
        mid_idx = len(results) // 2
        best_oos = results[mid_idx]['combo']
        print(f"  ⚠️ 使用IS中位数参数")

    # 应用最优
    best_params = copy.deepcopy(base_params)
    for name, val in best_oos.items():
        set_nested(best_params, name, val)

    print(f"  ✅ {phase} 最优: {best_oos}")
    return best_params, results


def get_benchmark_annual(index_code: str, start: str, end: str) -> float:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    df = pd.read_sql_query("""
        SELECT dq.trade_date, dq.close FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.code = ? AND dq.trade_date >= ? AND dq.trade_date <= ?
        ORDER BY dq.trade_date
    """, conn, params=[index_code, start, end])
    conn.close()
    if len(df) < 2:
        return 0
    total = df['close'].iloc[-1] / df['close'].iloc[0] - 1
    return (1 + total) ** (252 / max(len(df), 1)) - 1


def main():
    import argparse
    parser = argparse.ArgumentParser(description='快速超参数校准 (预计算+向量化)')
    parser.add_argument('--report-dir', required=True)
    parser.add_argument('--is-start', default='2024-01-01')
    parser.add_argument('--is-end', default='2025-06-30')
    parser.add_argument('--oos-start', default='2025-07-01')
    parser.add_argument('--oos-end', default='2026-03-31')
    parser.add_argument('--output', default='optimizer_params_calibrated.json')
    parser.add_argument('--phase', default='all',
                       choices=['all', 'stop', 'target', 'entry', 'filter', 'trailing', 'hold'])
    args = parser.parse_args()

    base_params = load_params()

    # 预计算 (一次性)
    print("=" * 60)
    print("  预计算IS期技术特征...")
    is_precomp = precompute_features(args.report_dir, args.is_start, args.is_end)
    print("  预计算OOS期技术特征...")
    oos_precomp = precompute_features(args.report_dir, args.oos_start, args.oos_end)

    phases = ['stop', 'target', 'entry', 'filter', 'trailing', 'hold']
    if args.phase != 'all':
        phases = [args.phase]

    all_logs = {}
    for phase in phases:
        if phase not in SEARCH_GRIDS:
            continue
        base_params, logs = calibrate_phase(
            phase, SEARCH_GRIDS[phase], base_params,
            is_precomp, oos_precomp,
        )
        all_logs[phase] = [{'combo': r['combo'], 'objective': r['objective'],
                            'metrics': {k: round(v, 6) if isinstance(v, float) else v
                                       for k, v in r['metrics'].items()}}
                           for r in logs[:20]]  # 只保存top 20

    # 保存
    with open(args.output, 'w') as f:
        json.dump(base_params, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 保存到: {args.output}")

    log_path = Path(args.output).with_suffix('.log.json')
    with open(log_path, 'w') as f:
        json.dump(all_logs, f, indent=2, default=str)

    # 用校准后参数跑完整回测对比
    print(f"\n{'='*60}")
    print("  校准后完整回测...")

    full_precomp = precompute_features(args.report_dir, args.is_start, args.oos_end)
    final = fast_backtest(full_precomp, base_params)
    fm = final['metrics']

    # 基准
    bench_300 = get_benchmark_annual('000300.SH', args.is_start, args.oos_end)
    bench_2000 = get_benchmark_annual('932000.CSI', args.is_start, args.oos_end)

    print(f"\n  校准后全周期结果 ({args.is_start} ~ {args.oos_end}):")
    print(f"  年化收益: {fm.get('annual_return',0):.1%}")
    print(f"  Sharpe: {fm.get('sharpe',0):.3f}")
    print(f"  MaxDD: {fm.get('max_drawdown',0):.1%}")
    print(f"  胜率: {fm.get('win_rate',0):.1%}")
    print(f"  平均持仓: {fm.get('avg_hold_days',0):.1f}天")
    print(f"  止损率: {fm.get('stop_loss_rate',0):.0%} / 止盈率: {fm.get('take_profit_rate',0):.0%}")
    print(f"  对沪深300超额: {fm.get('annual_return',0) - bench_300:.1%}")
    print(f"  对中证2000超额: {fm.get('annual_return',0) - bench_2000:.1%}")


if __name__ == '__main__':
    main()
