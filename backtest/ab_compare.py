#!/usr/bin/env python3
"""
A/B对比回测: 现有逻辑 vs 新Portfolio Optimizer

用法:
    python3 backtest/ab_compare.py \
        --report-dir reports/daily_selection_v4.9.0.2_fullmarket \
        --params optimizer_params.json
"""

import sys
import os
import json
import argparse
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from portfolio_optimizer import load_params
from backtest.dynamic_sl_tp_backtest import run_portfolio_backtest


def run_ab(report_dir: str, params_path: str = None,
           start_date: str = None, end_date: str = None):
    """运行A/B对比"""
    params = load_params(params_path) if params_path else load_params()
    benchmarks = ['000300.SH', '932000.CSI']

    # A: 旧逻辑 (固定百分比, Top 10等权, 固定持仓)
    print("\n" + "="*80)
    print("  A方案: 现有逻辑 (固定SL/TP, Top 10等权)")
    result_a = run_portfolio_backtest(
        report_dir, params, benchmarks,
        start_date=start_date, end_date=end_date,
        use_optimizer=False, label='A: Baseline',
    )

    # B: 新逻辑 (自适应价格, 动态SLTP, 风险预算仓位)
    print("\n" + "="*80)
    print("  B方案: Portfolio Optimizer (自适应价格+动态SLTP+风险预算)")
    result_b = run_portfolio_backtest(
        report_dir, params, benchmarks,
        start_date=start_date, end_date=end_date,
        use_optimizer=True, label='B: Optimized',
    )

    # 对比报告
    print("\n" + "="*80)
    print("  A/B 对比结果")
    print("="*80)

    if not result_a or not result_b:
        print("  某方案无结果, 无法对比")
        return

    ma = result_a['metrics']
    mb = result_b['metrics']

    rows = [
        ('年化收益', 'annual_return', '.1%'),
        ('Sharpe', 'sharpe', '.3f'),
        ('MaxDD', 'max_drawdown', '.1%'),
        ('月度胜率', 'monthly_win_rate', '.1%'),
        ('胜率', 'win_rate', '.1%'),
        ('平均收益', 'avg_return', '.2%'),
        ('交易笔数', 'n_trades', 'd'),
        ('平均持仓天数', 'avg_hold_days', '.1f'),
    ]

    for bench in benchmarks:
        rows.append((f'超额({bench})', f'excess_{bench}', '.1%'))

    print(f"\n{'指标':<20} {'A (Baseline)':>15} {'B (Optimized)':>15} {'差异':>12}")
    print("-" * 65)
    for label, key, fmt in rows:
        va = ma.get(key, 0)
        vb = mb.get(key, 0)
        diff = vb - va
        fmt_str = f'{{{fmt}}}'
        print(f"{label:<20} {format(va, fmt):>15} {format(vb, fmt):>15} {format(diff, fmt):>12}")

    # 退出方式对比
    print(f"\n{'退出方式':<20} {'A':>15} {'B':>15}")
    print("-" * 55)
    for outcome in ['stop_loss', 'take_profit', 'max_hold']:
        va = result_a['exit_stats'].get(outcome, 0)
        vb = result_b['exit_stats'].get(outcome, 0)
        print(f"{outcome:<20} {va:>15.1%} {vb:>15.1%}")

    # 判定
    print(f"\n{'='*60}")
    excess_300_a = ma.get('excess_000300.SH', 0)
    excess_300_b = mb.get('excess_000300.SH', 0)
    excess_2000_a = ma.get('excess_932000.CSI', 0)
    excess_2000_b = mb.get('excess_932000.CSI', 0)
    dd_a = abs(ma.get('max_drawdown', 0))
    dd_b = abs(mb.get('max_drawdown', 0))
    sharpe_a = ma.get('sharpe', 0)
    sharpe_b = mb.get('sharpe', 0)

    pass_300 = excess_300_b > excess_300_a
    pass_2000 = excess_2000_b > excess_2000_a
    pass_dd = dd_b <= dd_a + 0.02  # MaxDD不恶化超过2pp
    pass_sharpe = sharpe_b >= sharpe_a * 0.9  # Sharpe >= 90%

    all_pass = pass_300 and pass_2000 and pass_dd and pass_sharpe

    print(f"  对300超额: {'PASS' if pass_300 else 'FAIL'} B={excess_300_b:.1%} vs A={excess_300_a:.1%}")
    print(f"  对2000超额: {'PASS' if pass_2000 else 'FAIL'} B={excess_2000_b:.1%} vs A={excess_2000_a:.1%}")
    print(f"  MaxDD: {'PASS' if pass_dd else 'FAIL'} B={dd_b:.1%} vs A={dd_a:.1%} (容忍+2pp)")
    print(f"  Sharpe: {'PASS' if pass_sharpe else 'FAIL'} B={sharpe_b:.3f} vs A={sharpe_a:.3f} (>=90%)")
    print(f"\n  {'B方案通过! 建议切换到新逻辑' if all_pass else 'B方案未全部通过, 保留A方案'}")


def main():
    parser = argparse.ArgumentParser(description='A/B对比回测')
    parser.add_argument('--report-dir', required=True, help='报告目录')
    parser.add_argument('--params', default=None, help='参数文件 (默认optimizer_params.json)')
    parser.add_argument('--start-date', default=None, help='开始日期')
    parser.add_argument('--end-date', default=None, help='结束日期')
    args = parser.parse_args()

    run_ab(args.report_dir, args.params, args.start_date, args.end_date)


if __name__ == '__main__':
    main()
