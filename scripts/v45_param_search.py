#!/usr/bin/env python3
"""V4.5 CPPI参数搜索: 找到修复后的最优cppi_floor和cppi_multiplier"""

import sys, os, io, contextlib, json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')

from backtest import backtest_report_based as brb
from backtest import north_star_metrics as nsm
from backtest.run_north_star_eval import merge_report_dirs

nsm.DB_PATH = DB_PATH
brb.DB_PATH = DB_PATH

# 参数网格
CPPI_FLOORS = [0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
CPPI_MULTIPLIERS = [3, 5, 8, 10, 15, 20]

TOP_N = 10
FOCUS_DAYS = 10
BENCHMARK = '000905.SH'

REPORT_DIR = str(PROJECT_ROOT / 'reports' / 'daily_selection_v4.4_v2')
EXTENDED_DIR = str(PROJECT_ROOT / 'reports' / 'daily_selection_v4.4_v2_extended')

def main():
    # 合并报告（只做一次）
    merged_dir = str(Path(REPORT_DIR).parent / (Path(REPORT_DIR).name + '_merged_extended'))
    n_files = merge_report_dirs([REPORT_DIR, EXTENDED_DIR], merged_dir)
    if n_files == 0:
        print("无报告文件")
        return

    reports = brb.load_reports(merged_dir)
    print(f"已加载 {len(reports)} 天报告\n")

    # 基线：无overlay
    results = []

    print(f"{'='*90}")
    print(f"{'cppi_floor':>10s} {'multiplier':>10s} {'m*floor':>8s} │ "
          f"{'年化(毛)':>8s} {'年化(净)':>8s} {'MaxDD':>8s} {'Sharpe':>8s} {'ICIR':>8s} "
          f"{'V2分':>6s} {'等级':>4s}")
    print(f"{'─'*90}")

    # 基线
    print("  运行基线 (无overlay)...", end='', flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        r = brb.run_single_backtest(
            reports, "baseline", top_n=TOP_N, benchmark_code=BENCHMARK,
            focus_days=FOCUS_DAYS, vol_target=0, cppi_floor=0, cppi_multiplier=0)
    s = r['summary'].get(FOCUS_DAYS, {})
    v2_score, v2_max, v2_grade = _compute_v2(s)
    results.append((0, 0, v2_score, v2_max, v2_grade, s))
    print(f"\r{'无':>10s} {'无':>10s} {'—':>8s} │ "
          f"{s.get('gross_annual_return', s.get('annual_return', 0))*100:>7.1f}% "
          f"{s.get('net_annual_return', 0)*100:>7.1f}% "
          f"{s.get('max_drawdown', 0)*100:>7.1f}% "
          f"{s.get('sharpe_ratio', 0):>8.3f} "
          f"{s.get('icir', 0):>8.3f} "
          f"{v2_score:>3d}/{v2_max:<3d} "
          f"{v2_grade:>4s}")

    # CPPI参数网格
    total = len(CPPI_FLOORS) * len(CPPI_MULTIPLIERS)
    done = 0
    for floor in CPPI_FLOORS:
        for mult in CPPI_MULTIPLIERS:
            done += 1
            init_exp = floor * mult
            print(f"  [{done}/{total}] floor={floor}, m={mult} (初始exp={init_exp:.0%})...",
                  end='', flush=True)
            with contextlib.redirect_stdout(io.StringIO()):
                r = brb.run_single_backtest(
                    reports, f"F{floor}_M{mult}", top_n=TOP_N,
                    benchmark_code=BENCHMARK, focus_days=FOCUS_DAYS,
                    vol_target=0, cppi_floor=floor, cppi_multiplier=mult)
            s = r['summary'].get(FOCUS_DAYS, {})
            v2_score, v2_max, v2_grade = _compute_v2(s)
            results.append((floor, mult, v2_score, v2_max, v2_grade, s))

            print(f"\r{floor:>10.2f} {mult:>10d} {init_exp:>7.0%} │ "
                  f"{s.get('gross_annual_return', s.get('annual_return', 0))*100:>7.1f}% "
                  f"{s.get('net_annual_return', 0)*100:>7.1f}% "
                  f"{s.get('max_drawdown', 0)*100:>7.1f}% "
                  f"{s.get('sharpe_ratio', 0):>8.3f} "
                  f"{s.get('icir', 0):>8.3f} "
                  f"{v2_score:>3d}/{v2_max:<3d} "
                  f"{v2_grade:>4s}")

    print(f"\n{'='*90}")

    # 排序输出Top 10
    results.sort(key=lambda x: x[2], reverse=True)
    print(f"\n🏆 Top 10 参数组合 (按V2总分排序):")
    print(f"{'─'*90}")
    print(f"{'Rank':>4s} {'cppi_floor':>10s} {'multiplier':>10s} {'m*floor':>8s} │ "
          f"{'年化(毛)':>8s} {'年化(净)':>8s} {'MaxDD':>8s} {'Sharpe':>8s} {'ICIR':>8s} "
          f"{'V2分':>6s} {'等级':>4s}")
    print(f"{'─'*90}")
    for i, (floor, mult, score, mx, grade, s) in enumerate(results[:10]):
        init_exp = floor * mult if floor > 0 else 0
        f_str = f"{floor:.2f}" if floor > 0 else "无"
        m_str = f"{mult}" if mult > 0 else "无"
        e_str = f"{init_exp:.0%}" if floor > 0 else "—"
        print(f"{i+1:>4d} {f_str:>10s} {m_str:>10s} {e_str:>8s} │ "
              f"{s.get('gross_annual_return', s.get('annual_return', 0))*100:>7.1f}% "
              f"{s.get('net_annual_return', 0)*100:>7.1f}% "
              f"{s.get('max_drawdown', 0)*100:>7.1f}% "
              f"{s.get('sharpe_ratio', 0):>8.3f} "
              f"{s.get('icir', 0):>8.3f} "
              f"{score:>3d}/{mx:<3d} "
              f"{grade:>4s}")

    # 保存结果到文件
    out_file = PROJECT_ROOT / 'reports' / 'v45_param_search_results.txt'
    with open(out_file, 'w') as f:
        f.write(f"V4.5 CPPI参数搜索结果 ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n")
        f.write(f"{'='*100}\n")
        f.write(f"{'Rank':>4s} {'cppi_floor':>10s} {'multiplier':>10s} {'m*floor':>8s} │ "
                f"{'年化(毛)':>8s} {'年化(净)':>8s} {'MaxDD':>8s} {'Sharpe':>8s} {'Sortino':>8s} "
                f"{'ICIR':>8s} {'V2分':>6s} {'等级':>4s}\n")
        f.write(f"{'─'*100}\n")
        for i, (floor, mult, score, mx, grade, s) in enumerate(results):
            init_exp = floor * mult if floor > 0 else 0
            f_str = f"{floor:.2f}" if floor > 0 else "无"
            m_str = f"{mult}" if mult > 0 else "无"
            e_str = f"{init_exp:.0%}" if floor > 0 else "—"
            f.write(f"{i+1:>4d} {f_str:>10s} {m_str:>10s} {e_str:>8s} │ "
                    f"{s.get('gross_annual_return', s.get('annual_return', 0))*100:>7.1f}% "
                    f"{s.get('net_annual_return', 0)*100:>7.1f}% "
                    f"{s.get('max_drawdown', 0)*100:>7.1f}% "
                    f"{s.get('sharpe_ratio', 0):>8.3f} "
                    f"{s.get('sortino_ratio', 0):>8.3f} "
                    f"{s.get('icir', 0):>8.3f} "
                    f"{score:>3d}/{mx:<3d} "
                    f"{grade:>4s}\n")
    print(f"\n结果已保存到: {out_file}")


def _compute_v2(s):
    """计算V2总分"""
    from backtest.north_star_metrics import (
        NORTH_STAR_TARGETS_V2, score_metric_v2, compute_v2_grade
    )
    metric_key_map = {
        'daily_ic': s.get('ic_mean', 0),
        'icir': s.get('icir', 0),
        'ic_positive_pct': s.get('ic_positive_pct', 0),
        'ic_monotonicity': s.get('ic_monotonicity', 0),
        'ic_time_stability': s.get('ic_time_stability', 0),
        'signal_half_life': s.get('signal_half_life', 0),
        'annual_turnover': s.get('annual_turnover', 0),
        'annual_cost_drag': s.get('annual_cost_drag', 0),
        'net_gross_ratio': s.get('net_gross_ratio', 0),
        'limit_up_fail_rate': s.get('limit_up_fail_rate', 0),
        'liquidity_coverage': s.get('liquidity_coverage', 0),
        'max_drawdown': s.get('max_drawdown', 0),
        'sharpe_ratio': s.get('sharpe_ratio', 0),
        'sortino_ratio': s.get('sortino_ratio', 0),
        'calmar_ratio': s.get('calmar_ratio', 0),
        'worst_rolling_60d_icir': s.get('worst_rolling_60d_icir', 0),
        'annual_return': s.get('gross_annual_return', s.get('annual_return', 0)),
        'monthly_win_rate': s.get('monthly_win_rate', 0),
        'half_period_consistency': s.get('half_period_consistency', 0),
        'small_cap_bias_ratio': s.get('small_cap_bias_ratio', 0),
        'median_market_cap_bn': s.get('median_market_cap_bn', 0),
    }
    total = 0
    max_score = 0
    for key, cfg in NORTH_STAR_TARGETS_V2.items():
        val = metric_key_map.get(key, 0)
        sc, _ = score_metric_v2(val, cfg)
        total += sc
        max_score += 5
    grade = compute_v2_grade(total, max_score)
    return total, max_score, grade


if __name__ == '__main__':
    main()
