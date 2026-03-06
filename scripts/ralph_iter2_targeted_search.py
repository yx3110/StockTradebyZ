#!/usr/bin/env python3
"""Ralph Loop Iteration 2: Targeted search for remaining 8 points (97→105).

Focus areas:
1. Liquidity coverage: 91.6% → 95% (add min_turnover_rate)
2. IC monotonicity: 3.23 → 4.5 (hard, model-level)
3. Signal half-life: 15.0 → 20.0 (depends on ICIR decay)
4. Half-period consistency: 52.7% → 80% (hardest)
"""

import sys, os, io, contextlib, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')

from backtest import backtest_report_based as brb
from backtest import north_star_metrics as nsm

nsm.DB_PATH = DB_PATH
brb.DB_PATH = DB_PATH


def compute_v2_detailed(s):
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
        if val is None:
            val = 0
        sc, _ = score_metric_v2(val, cfg)
        total += sc
        max_score += 5
        details[key] = {'score': sc, 'value': val, 'target': cfg.get('target', 0)}
    grade = compute_v2_grade(total, max_score)
    return total, max_score, grade, details


def run_test(reports, label, **kwargs):
    """Run backtest and return score + details."""
    with contextlib.redirect_stdout(io.StringIO()):
        r = brb.run_single_backtest(
            reports, label, top_n=10, benchmark_code='000905.SH',
            focus_days=10, **kwargs)
    if r is None:
        return None
    s = r['summary'].get(10, {})
    v2_score, v2_max, v2_grade, details = compute_v2_detailed(s)
    return {
        'score': v2_score, 'max': v2_max, 'grade': v2_grade,
        'details': details, 'summary': s, 'params': kwargs,
        'all_summaries': r['summary'],
    }


def print_result(label, result, focus_metrics=None):
    """Print result with focus on specific metrics."""
    if result is None:
        print(f"  {label}: FAILED")
        return
    print(f"  {label}: {result['score']}/{result['max']} ({result['grade']})")
    if focus_metrics:
        for key in focus_metrics:
            d = result['details'].get(key, {})
            val = d.get('value', 0)
            sc = d.get('score', 0)
            stars = '★' * sc + '☆' * (5 - sc)
            if isinstance(val, float) and abs(val) < 10:
                print(f"    {key:30s} = {val:.4f}  ({sc}/5 {stars})")
            else:
                print(f"    {key:30s} = {val}  ({sc}/5 {stars})")


def main():
    # V4.4 CPPI = 79/105 (cap_balance 81.6%), V4.6 CPPI = 75/105 (cap_balance 11%)
    # V4.4 is BETTER overall. Focus on V4.4 + replace_threshold to reduce turnover

    # Phase 1: V4.4 reports with replace_threshold
    report_dir_v44 = str(PROJECT_ROOT / 'reports' / 'daily_selection_v4.4_v2_merged_extended')
    print(f"Loading V4.4 reports from: {report_dir_v44}")
    reports = brb.load_reports(report_dir_v44)
    print(f"Loaded {len(reports)} trading days\n")

    combos = [
        # V4.4 CPPI baseline (79/105 reference)
        ("V44_CPPI_base", dict(cppi_floor=0.05, cppi_multiplier=20)),
        # Replace threshold to reduce turnover (47.7x→<20x)
        ("V44_CPPI+RT10%", dict(cppi_floor=0.05, cppi_multiplier=20, replace_threshold=0.10)),
        ("V44_CPPI+RT20%", dict(cppi_floor=0.05, cppi_multiplier=20, replace_threshold=0.20)),
        ("V44_CPPI+RT30%", dict(cppi_floor=0.05, cppi_multiplier=20, replace_threshold=0.30)),
        ("V44_CPPI+RT50%", dict(cppi_floor=0.05, cppi_multiplier=20, replace_threshold=0.50)),
        # Liq filter to test
        ("V44_CPPI+Liq0.5+RT20%", dict(cppi_floor=0.05, cppi_multiplier=20,
                                         min_turnover_rate=0.5, replace_threshold=0.20)),
    ]

    print(f"Testing {len(combos)} combos (est. {len(combos)*20} min)...\n")

    results = []
    for i, (label, params) in enumerate(combos):
        print(f"  [{i+1}/{len(combos)}] {label}...", end='', flush=True)
        result = run_test(reports, label, **params)
        if result is None:
            print(" FAILED")
            continue
        s = result['summary']
        results.append((label, result))

        # Show key changing metrics
        not5 = [(k, v) for k, v in result['details'].items() if v['score'] < 5]
        not5_str = ", ".join(f"{k}={v['score']}" for k, v in sorted(not5, key=lambda x: x[1]['score']))
        print(f"\r  {label:>25s}: {result['score']}/105 ({result['grade']}) | "
              f"AnnRet={s.get('gross_annual_return', s.get('annual_return', 0))*100:.1f}% "
              f"MaxDD={s.get('max_drawdown', 0)*100:.1f}% "
              f"Turn={s.get('annual_turnover', 0):.1f}x "
              f"Liq={s.get('liquidity_coverage', 0)*100:.1f}%")
        print(f"{'':>29s}NOT-5: {not5_str}")

    # Sort + summary
    results.sort(key=lambda x: x[1]['score'], reverse=True)
    print(f"\n{'='*90}")
    print(f"🏆 Sorted results:")
    for i, (label, r) in enumerate(results):
        s = r['summary']
        print(f"  {i+1}. {label:>25s}: {r['score']}/105 ({r['grade']}) "
              f"AnnRet={s.get('gross_annual_return', s.get('annual_return', 0))*100:.1f}% "
              f"Sharpe={s.get('sharpe_ratio', 0):.3f}")

    # Best metric breakdown
    if results:
        label, best = results[0]
        print(f"\nBest: {label} → {best['score']}/{best['max']} ({best['grade']})")
        print(f"Params: {best['params']}")
        print(f"\nNon-perfect metrics:")
        for key, info in sorted(best['details'].items(), key=lambda x: x[1]['score']):
            if info['score'] < 5:
                stars = '★' * info['score'] + '☆' * (5 - info['score'])
                print(f"  {key:30s} {info['value']:>10.4f}  target={info['target']:>8.3f}  {info['score']}/5 {stars}")

    # Save
    out_file = PROJECT_ROOT / 'reports' / 'ralph_iter2_results.json'
    save_data = []
    for label, r in results:
        save_data.append({
            'label': label, 'score': r['score'], 'grade': r['grade'],
            'params': r['params'],
            'details': {k: {'score': v['score'], 'value': v['value']}
                       for k, v in r['details'].items()},
        })
    with open(out_file, 'w') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {out_file}")


if __name__ == '__main__':
    main()
