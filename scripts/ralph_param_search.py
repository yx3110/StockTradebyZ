#!/usr/bin/env python3
"""Ralph Loop Iteration 1: Comprehensive parameter search for North Star V2 optimization.

Tests combinations of:
- retention_bonus (reduce turnover)
- CPPI (control drawdown)
- score_floor (filter low-quality picks)
- sector_diversify (industry diversification)
"""

import sys, os, io, contextlib, json
from pathlib import Path
from datetime import datetime
from itertools import product

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')

from backtest import backtest_report_based as brb
from backtest import north_star_metrics as nsm

nsm.DB_PATH = DB_PATH
brb.DB_PATH = DB_PATH

# Load reports - use merged extended for full evaluation
REPORT_DIR = str(PROJECT_ROOT / 'reports' / 'daily_selection_v4.4_v2_merged_extended')
TOP_N = 10
FOCUS_DAYS = 10
BENCHMARK = '000905.SH'


def compute_v2(s):
    """Compute V2 score with per-metric breakdown."""
    from backtest.north_star_metrics import NORTH_STAR_TARGETS_V2, score_metric_v2, compute_v2_grade
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
        'cap_balance_ratio': s.get('cap_balance_ratio', 0),
        'median_market_cap_bn': s.get('median_market_cap_bn', 0),
    }
    total = 0
    max_score = 0
    details = {}
    for key, cfg in NORTH_STAR_TARGETS_V2.items():
        val = metric_key_map.get(key, 0)
        sc, _ = score_metric_v2(val, cfg)
        total += sc
        max_score += 5
        details[key] = {'score': sc, 'value': val, 'target': cfg.get('target', 0)}
    grade = compute_v2_grade(total, max_score)
    return total, max_score, grade, details


def run_combo(reports, label, retention_bonus=0.0, cppi_floor=0.0, cppi_multiplier=3.0,
              score_floor=0.0, sector_diversify=0):
    """Run backtest with given parameters and return V2 score."""
    with contextlib.redirect_stdout(io.StringIO()):
        r = brb.run_single_backtest(
            reports, label, top_n=TOP_N, benchmark_code=BENCHMARK,
            focus_days=FOCUS_DAYS,
            retention_bonus=retention_bonus,
            score_floor=score_floor,
            cppi_floor=cppi_floor,
            cppi_multiplier=cppi_multiplier,
            sector_diversify=sector_diversify,
        )
    if r is None:
        return None
    s = r['summary'].get(FOCUS_DAYS, {})
    v2_score, v2_max, v2_grade, details = compute_v2(s)
    return {
        'score': v2_score,
        'max': v2_max,
        'grade': v2_grade,
        'details': details,
        'summary': s,
        'params': {
            'retention_bonus': retention_bonus,
            'cppi_floor': cppi_floor,
            'cppi_multiplier': cppi_multiplier,
            'score_floor': score_floor,
            'sector_diversify': sector_diversify,
        }
    }


def main():
    print(f"Loading reports from: {REPORT_DIR}")
    reports = brb.load_reports(REPORT_DIR)
    print(f"Loaded {len(reports)} trading days\n")

    # Focused parameter grid - full 517-day evaluation
    # Key finding: retention_bonus=2.0 locks portfolio, bad for long windows
    # Focus on CPPI + moderate retention + sector diversification
    combos = [
        # (retention_bonus, (cppi_floor, cppi_mult), score_floor, sector_diversify)
        # Baseline (no optimization)
        (0.0, (0, 0), 0.0, 0),
        # Pure CPPI (proven best on old short window)
        (0.0, (0.05, 20), 0.0, 0),
        (0.0, (0.08, 10), 0.0, 0),
        (0.0, (0.10, 8), 0.0, 0),
        # Moderate retention (0.3-0.5, still allows rotation)
        (0.3, (0, 0), 0.0, 0),
        (0.3, (0.05, 20), 0.0, 0),
        (0.5, (0.05, 20), 0.0, 0),
        # Sector diversify without retention lock
        (0.0, (0, 0), 0.0, 2),
        (0.0, (0.05, 20), 0.0, 2),
        (0.3, (0.05, 20), 0.0, 2),
        (0.5, (0.05, 20), 0.0, 2),
        # Score floor combos
        (0.3, (0.05, 20), 40.0, 2),
    ]
    print(f"Testing {len(combos)} parameter combinations...\n")

    results = []
    header = (f"{'#':>3s} {'Ret.Bonus':>9s} {'CPPI':>12s} {'Floor':>5s} {'SecDiv':>6s} │ "
              f"{'V2':>5s} {'Grade':>5s} │ "
              f"{'AnnRet':>7s} {'MaxDD':>7s} {'Sharpe':>7s} {'Turn':>5s} {'NetGr':>6s} {'Consist':>7s}")
    print(header)
    print('─' * len(header))

    for i, (rb, (cf, cm), sf, sd) in enumerate(combos):
        label = f"RB{rb}_F{cf}_M{cm}_SF{sf}_SD{sd}"
        print(f"  [{i+1}/{len(combos)}] {label}...", end='', flush=True)

        result = run_combo(reports, label,
                          retention_bonus=rb,
                          cppi_floor=cf, cppi_multiplier=cm,
                          score_floor=sf,
                          sector_diversify=sd)

        if result is None:
            print(" FAILED")
            continue

        s = result['summary']
        results.append(result)

        cppi_str = f"F{cf}/M{cm}" if cf > 0 else "off"
        print(f"\r{i+1:>3d} {rb:>9.1f} {cppi_str:>12s} {sf:>5.0f} {sd:>6d} │ "
              f"{result['score']:>3d}/{result['max']:<3d} {result['grade']:>5s} │ "
              f"{s.get('gross_annual_return', s.get('annual_return', 0))*100:>6.1f}% "
              f"{s.get('max_drawdown', 0)*100:>6.1f}% "
              f"{s.get('sharpe_ratio', 0):>7.3f} "
              f"{s.get('annual_turnover', 0):>5.1f} "
              f"{s.get('net_gross_ratio', 0)*100:>5.1f}% "
              f"{s.get('half_period_consistency', 0)*100:>6.1f}%")

    # Sort by V2 score
    results.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n{'='*100}")
    print(f"🏆 Top 15 parameter combinations (by V2 score):")
    print('─' * 100)
    print(header)
    print('─' * len(header))

    for i, r in enumerate(results[:15]):
        p = r['params']
        s = r['summary']
        cppi_str = f"F{p['cppi_floor']}/M{p['cppi_multiplier']}" if p['cppi_floor'] > 0 else "off"
        print(f"{i+1:>3d} {p['retention_bonus']:>9.1f} {cppi_str:>12s} {p['score_floor']:>5.0f} "
              f"{p['sector_diversify']:>6d} │ "
              f"{r['score']:>3d}/{r['max']:<3d} {r['grade']:>5s} │ "
              f"{s.get('gross_annual_return', s.get('annual_return', 0))*100:>6.1f}% "
              f"{s.get('max_drawdown', 0)*100:>6.1f}% "
              f"{s.get('sharpe_ratio', 0):>7.3f} "
              f"{s.get('annual_turnover', 0):>5.1f} "
              f"{s.get('net_gross_ratio', 0)*100:>5.1f}% "
              f"{s.get('half_period_consistency', 0)*100:>6.1f}%")

    # Print best result's per-metric breakdown
    if results:
        best = results[0]
        print(f"\n{'='*100}")
        print(f"Best result: {best['score']}/{best['max']} ({best['grade']}) with params:")
        print(f"  retention_bonus={best['params']['retention_bonus']}")
        print(f"  cppi_floor={best['params']['cppi_floor']}, cppi_multiplier={best['params']['cppi_multiplier']}")
        print(f"  score_floor={best['params']['score_floor']}")
        print(f"  sector_diversify={best['params']['sector_diversify']}")

        print(f"\nPer-metric breakdown:")
        for key, info in sorted(best['details'].items(), key=lambda x: x[1]['score']):
            stars = '★' * info['score'] + '☆' * (5 - info['score'])
            print(f"  {key:30s} {info['value']:>10.4f}  target={info['target']:>8.3f}  {info['score']}/5 {stars}")

    # Save results
    out_file = PROJECT_ROOT / 'reports' / 'ralph_param_search_results.json'
    save_data = []
    for r in results[:30]:
        save_data.append({
            'score': r['score'],
            'max': r['max'],
            'grade': r['grade'],
            'params': r['params'],
            'key_metrics': {
                'annual_return': r['summary'].get('gross_annual_return', r['summary'].get('annual_return', 0)),
                'max_drawdown': r['summary'].get('max_drawdown', 0),
                'sharpe_ratio': r['summary'].get('sharpe_ratio', 0),
                'annual_turnover': r['summary'].get('annual_turnover', 0),
                'net_gross_ratio': r['summary'].get('net_gross_ratio', 0),
                'half_period_consistency': r['summary'].get('half_period_consistency', 0),
                'monthly_win_rate': r['summary'].get('monthly_win_rate', 0),
                'icir': r['summary'].get('icir', 0),
            },
            'details': {k: {'score': v['score'], 'value': v['value']} for k, v in r['details'].items()},
        })

    with open(out_file, 'w') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_file}")


if __name__ == '__main__':
    main()
