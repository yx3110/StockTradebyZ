#!/usr/bin/env python3
"""
超参数网格搜索校准

校准顺序: Stop → Target → Entry → Filter → Trailing → Hold
IS: 2022-01-01~2025-06-30, OOS: 2025-07-01~2026-03-31
综合目标: Sharpe × (1 - |MaxDD|)
"""

import sys
import os
import json
import itertools
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from portfolio_optimizer import load_params
from backtest.dynamic_sl_tp_backtest import run_portfolio_backtest


# ============================================================
# 搜索网格定义
# ============================================================
SEARCH_GRIDS = {
    'stop': {
        'stop.atr_multiplier': [1.0, 1.5, 2.0, 2.5, 3.0],
        'stop.env_mult_bullish': [0.7, 0.85, 1.0],
        'stop.env_mult_bearish': [1.0, 1.2, 1.5],
        'stop.min_stop_pct': [0.02, 0.03, 0.04],
        'stop.max_stop_main': [0.08, 0.10, 0.12],
        'stop.max_stop_wide': [0.12, 0.15, 0.18],
    },
    'target': {
        'target.min_rr_ratio': [1.5, 2.0, 2.5, 3.0],
        'target.target_clip_min': [0.02, 0.03, 0.05],
        'target.target_clip_max': [0.10, 0.15, 0.20],
    },
    'entry': {
        'entry.atr_discount_mult': [0.1, 0.3, 0.5, 0.7, 1.0],
        'entry.support_discount_mult': [0.1, 0.2, 0.3, 0.5],
        'entry.ml_bullish_threshold': [0.002, 0.005, 0.01],
        'entry.ml_bullish_mult': [0.3, 0.5, 0.7],
        'entry.ml_bearish_mult': [1.2, 1.5, 2.0],
        'entry.max_discount': [0.02, 0.03, 0.05],
    },
    'filter': {
        'filter.composite_cutoff': [0, 0.0001, 0.0005, 0.001],
        'filter.min_n': [3, 5, 8],
        'filter.max_n_bull': [10, 15, 20, 30],
        'filter.max_n_bear': [3, 5, 8],
    },
    'trailing': {
        'trailing.trailing_trigger_pct': [0.3, 0.5, 0.6, 0.8],
        'trailing.trailing_fallback_pct': [0.2, 0.3, 0.4, 0.5],
    },
    'hold': {
        'hold.max_hold_days': [5, 10, 15, 20, 30],
    },
}


def set_nested(d: dict, dotted_key: str, value):
    """设置嵌套dict值: 'stop.atr_multiplier' -> d['stop']['atr_multiplier']"""
    keys = dotted_key.split('.')
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def objective(metrics: dict) -> float:
    """综合优化目标: Sharpe * (1 - |MaxDD|)"""
    sharpe = metrics.get('sharpe', 0)
    max_dd = abs(metrics.get('max_drawdown', 0))
    return sharpe * (1 - max_dd)


def calibrate_group(group_name: str, grid: dict, base_params: dict,
                    report_dir: str, is_dates: Tuple[str, str],
                    oos_dates: Tuple[str, str]) -> Tuple[dict, list]:
    """
    对一组参数做网格搜索

    Returns: (best_params, search_log)
    """
    param_names = list(grid.keys())
    param_values = list(grid.values())
    combos = list(itertools.product(*param_values))

    print(f"\n{'='*60}")
    print(f"  校准: {group_name} ({len(combos)} 组合)")
    print(f"  IS: {is_dates[0]} -> {is_dates[1]}")

    results = []
    for i, combo in enumerate(combos):
        params = json.loads(json.dumps(base_params))  # deep copy
        for name, val in zip(param_names, combo):
            set_nested(params, name, val)

        t0 = time.time()
        result = run_portfolio_backtest(
            report_dir, params,
            start_date=is_dates[0], end_date=is_dates[1],
            label=f'{group_name}_{i}',
        )
        elapsed = time.time() - t0

        if not result:
            continue

        obj = objective(result['metrics'])
        entry = {
            'combo': dict(zip(param_names, combo)),
            'metrics': result['metrics'],
            'objective': obj,
            'time': elapsed,
        }
        results.append(entry)

        if (i + 1) % 10 == 0 or i == len(combos) - 1:
            print(f"  [{i+1}/{len(combos)}] best_obj={max(r['objective'] for r in results):.4f}")

    if not results:
        print("  无有效结果!")
        return base_params, []

    # 排序, 取Top 3在OOS验证
    results.sort(key=lambda r: r['objective'], reverse=True)
    top3 = results[:3]

    print(f"\n  IS Top 3:")
    for j, r in enumerate(top3):
        print(f"    #{j+1}: obj={r['objective']:.4f}, "
              f"Sharpe={r['metrics'].get('sharpe',0):.3f}, "
              f"MaxDD={r['metrics'].get('max_drawdown',0):.1%}")

    # OOS验证
    best_oos = None
    best_oos_obj = -999

    for j, r in enumerate(top3):
        params = json.loads(json.dumps(base_params))
        for name, val in r['combo'].items():
            set_nested(params, name, val)

        oos_result = run_portfolio_backtest(
            report_dir, params,
            start_date=oos_dates[0], end_date=oos_dates[1],
            label=f'{group_name}_OOS_{j}',
        )
        if not oos_result:
            continue

        oos_obj = objective(oos_result['metrics'])
        is_obj = r['objective']

        ratio_str = f"{oos_obj/is_obj:.1%}" if is_obj != 0 else "N/A"
        print(f"    OOS #{j+1}: obj={oos_obj:.4f} (IS={is_obj:.4f}, ratio={ratio_str})")

        # 过拟合检测: OOS < IS * 60%
        if is_obj > 0 and oos_obj < is_obj * 0.6:
            print(f"    !! 过拟合! OOS/IS={ratio_str} < 60%")
            continue

        if oos_obj > best_oos_obj:
            best_oos_obj = oos_obj
            best_oos = r['combo']

    # 如果所有Top3都过拟合, 用IS中位数参数
    if best_oos is None:
        print("  !! 全部过拟合, 使用IS中位数参数")
        mid = results[len(results) // 2]
        best_oos = mid['combo']

    # 应用最优参数
    best_params = json.loads(json.dumps(base_params))
    for name, val in best_oos.items():
        set_nested(best_params, name, val)

    print(f"\n  [OK] {group_name} 最优: {best_oos}")
    return best_params, results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='超参数网格搜索校准 — 6阶段IS/OOS验证')
    parser.add_argument('--report-dir', required=True, help='报告目录 (含 analysis_data_*.json)')
    parser.add_argument('--is-start', default='2022-01-01', help='IS开始日期 (默认 2022-01-01)')
    parser.add_argument('--is-end', default='2025-06-30', help='IS结束日期 (默认 2025-06-30)')
    parser.add_argument('--oos-start', default='2025-07-01', help='OOS开始日期 (默认 2025-07-01)')
    parser.add_argument('--oos-end', default='2026-03-31', help='OOS结束日期 (默认 2026-03-31)')
    parser.add_argument('--output', default='optimizer_params_calibrated.json', help='输出文件 (默认 optimizer_params_calibrated.json)')
    parser.add_argument('--phase', choices=['all', 'stop', 'target', 'entry', 'filter', 'trailing', 'hold'],
                       default='all', help='校准阶段 (默认全部)')
    args = parser.parse_args()

    base_params = load_params()
    is_dates = (args.is_start, args.is_end)
    oos_dates = (args.oos_start, args.oos_end)

    phases = ['stop', 'target', 'entry', 'filter', 'trailing', 'hold']
    if args.phase != 'all':
        phases = [args.phase]

    all_logs = {}
    for phase in phases:
        if phase not in SEARCH_GRIDS:
            continue
        base_params, logs = calibrate_group(
            phase, SEARCH_GRIDS[phase], base_params,
            args.report_dir, is_dates, oos_dates,
        )
        all_logs[phase] = logs

    # 保存校准结果
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(base_params, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 校准完成, 保存到: {output_path}")

    # 保存搜索日志
    log_path = output_path.with_suffix('.log.json')
    with open(log_path, 'w') as f:
        json.dump(all_logs, f, indent=2, default=str)
    print(f"  搜索日志: {log_path}")


if __name__ == '__main__':
    main()
