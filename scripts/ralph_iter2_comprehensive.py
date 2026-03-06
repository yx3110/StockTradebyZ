#!/usr/bin/env python3
"""Ralph Loop Iteration 2: Comprehensive optimization with HOLDING_DAYS=20 fix.

Key changes:
1. HOLDING_DAYS now includes 20 → signal_half_life can reach 20+
2. Test replace_threshold for turnover reduction without retention_bonus
3. Test min_turnover_rate for liquidity coverage
4. Focus on 517-day extended window for reliable evaluation
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

# Verify HOLDING_DAYS includes 20
assert 20 in brb.HOLDING_DAYS, f"HOLDING_DAYS must include 20, got {brb.HOLDING_DAYS}"
print(f"HOLDING_DAYS = {brb.HOLDING_DAYS}")


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
        val = metric_key_map.get(key)
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
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
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
    }


def print_gaps(result):
    """Print only metrics that are not 5/5."""
    if result is None:
        return
    gaps = [(k, v) for k, v in result['details'].items() if v['score'] < 5]
    gaps.sort(key=lambda x: x[1]['score'])
    for k, v in gaps:
        val = v['value']
        if isinstance(val, float):
            if abs(val) < 1:
                val_str = f"{val:.4f}"
            elif abs(val) < 100:
                val_str = f"{val:.2f}"
            else:
                val_str = f"{val:.1f}"
        else:
            val_str = str(val)
        stars = '★' * v['score'] + '☆' * (5 - v['score'])
        print(f"    {k:30s} {val_str:>10s} (target={v['target']}) {v['score']}/5 {stars}")


def main():
    # === Phase 1: 112-day short window (fast iteration) ===
    print("=" * 90)
    print("PHASE 1: 112-day short window (V4.4v2)")
    print("=" * 90)

    report_dir_short = str(PROJECT_ROOT / 'reports' / 'daily_selection_v4.4_v2')
    reports_short = brb.load_reports(report_dir_short)
    print(f"Loaded {len(reports_short)} trading days\n")

    # Test 1a: Previous best (retention_bonus=2.0, sector_diversify=2)
    print("--- Test 1a: Previous best (RB=2.0, SD=2) ---")
    r1a = run_test(reports_short, "Short-Prev-Best",
                   retention_bonus=2.0, sector_diversify=2)
    if r1a:
        print(f"  Score: {r1a['score']}/{r1a['max']} ({r1a['grade']})")
        print(f"  Signal half-life: {r1a['details']['signal_half_life']['value']:.1f} "
              f"(was 15.0, now with 20d period)")
        print(f"  Gaps:")
        print_gaps(r1a)

    # Test 1b: Best CPPI without retention (better for long window)
    print("\n--- Test 1b: Pure CPPI (F0.05/M20, no retention) ---")
    r1b = run_test(reports_short, "Short-Pure-CPPI",
                   cppi_floor=0.05, cppi_multiplier=20)
    if r1b:
        print(f"  Score: {r1b['score']}/{r1b['max']} ({r1b['grade']})")
        print(f"  Gaps:")
        print_gaps(r1b)

    # Test 1c: CPPI + replace_threshold (reduce turnover without locking)
    print("\n--- Test 1c: CPPI + replace_threshold combos ---")
    for rt in [0.05, 0.10, 0.15, 0.20, 0.30]:
        r = run_test(reports_short, f"Short-CPPI-RT{rt}",
                     cppi_floor=0.05, cppi_multiplier=20,
                     replace_threshold=rt)
        if r:
            s = r['summary']
            turn = r['details']['annual_turnover']
            sharpe = r['details']['sharpe_ratio']
            consist = r['details']['half_period_consistency']
            print(f"    RT={rt:.2f}: {r['score']}/105 ({r['grade']}) | "
                  f"Turn={turn['value']:.1f}({turn['score']}/5) "
                  f"Sharpe={sharpe['value']:.3f}({sharpe['score']}/5) "
                  f"Consist={consist['value']:.1%}({consist['score']}/5)")

    # Test 1d: CPPI + moderate retention + replace_threshold
    print("\n--- Test 1d: CPPI + moderate retention + replace_threshold ---")
    for rb in [0.3, 0.5, 1.0]:
        for rt in [0.10, 0.20]:
            r = run_test(reports_short, f"Short-RB{rb}-RT{rt}",
                         cppi_floor=0.05, cppi_multiplier=20,
                         retention_bonus=rb, replace_threshold=rt,
                         sector_diversify=2)
            if r:
                s = r['summary']
                turn = r['details']['annual_turnover']
                sharpe = r['details']['sharpe_ratio']
                consist = r['details']['half_period_consistency']
                mono = r['details']['ic_monotonicity']
                shl = r['details']['signal_half_life']
                not5_count = sum(1 for v in r['details'].values() if v['score'] < 5)
                print(f"    RB={rb:.1f} RT={rt:.2f}: {r['score']}/105 ({r['grade']}) | "
                      f"{not5_count} gaps | Turn={turn['value']:.1f} "
                      f"Sharpe={sharpe['value']:.3f} Consist={consist['value']:.1%} "
                      f"Mono={mono['value']:.3f} HalfLife={shl['value']:.1f}")

    # Test 1e: CPPI + min_turnover_rate (liquidity)
    print("\n--- Test 1e: CPPI + min_turnover_rate ---")
    for mtr in [0.5, 1.0, 2.0]:
        r = run_test(reports_short, f"Short-CPPI-MTR{mtr}",
                     cppi_floor=0.05, cppi_multiplier=20,
                     min_turnover_rate=mtr, sector_diversify=2)
        if r:
            liq = r['details']['liquidity_coverage']
            mono = r['details']['ic_monotonicity']
            print(f"    MTR={mtr:.1f}: {r['score']}/105 ({r['grade']}) | "
                  f"Liq={liq['value']:.1%}({liq['score']}/5) "
                  f"Mono={mono['value']:.3f}({mono['score']}/5)")

    # === Phase 2: 517-day extended window (reliable evaluation) ===
    print("\n" + "=" * 90)
    print("PHASE 2: 517-day extended window (V4.4v2 merged)")
    print("=" * 90)

    report_dir_long = str(PROJECT_ROOT / 'reports' / 'daily_selection_v4.4_v2_merged_extended')
    reports_long = brb.load_reports(report_dir_long)
    print(f"Loaded {len(reports_long)} trading days\n")

    # Test 2a: Pure CPPI baseline (previous best on 517d)
    print("--- Test 2a: Pure CPPI baseline (F0.05/M20) ---")
    r2a = run_test(reports_long, "Long-Pure-CPPI",
                   cppi_floor=0.05, cppi_multiplier=20)
    if r2a:
        print(f"  Score: {r2a['score']}/{r2a['max']} ({r2a['grade']})")
        print(f"  Signal half-life: {r2a['details']['signal_half_life']['value']:.1f}")
        print(f"  Gaps:")
        print_gaps(r2a)

    # Test 2b: Best short-window combo on long window
    # Pick top 3 combos from Phase 1 and test on 517d
    best_short_combos = [
        # Pure CPPI + replace_threshold (best RT from 1c)
        {'cppi_floor': 0.05, 'cppi_multiplier': 20, 'replace_threshold': 0.10},
        {'cppi_floor': 0.05, 'cppi_multiplier': 20, 'replace_threshold': 0.15},
        {'cppi_floor': 0.05, 'cppi_multiplier': 20, 'replace_threshold': 0.20},
        # CPPI + moderate retention + RT
        {'cppi_floor': 0.05, 'cppi_multiplier': 20,
         'retention_bonus': 0.3, 'replace_threshold': 0.10, 'sector_diversify': 2},
        {'cppi_floor': 0.05, 'cppi_multiplier': 20,
         'retention_bonus': 0.5, 'replace_threshold': 0.10, 'sector_diversify': 2},
        # CPPI + min_turnover_rate
        {'cppi_floor': 0.05, 'cppi_multiplier': 20, 'min_turnover_rate': 1.0},
        {'cppi_floor': 0.05, 'cppi_multiplier': 20,
         'min_turnover_rate': 1.0, 'sector_diversify': 2},
    ]

    print("\n--- Test 2b: Cross-validated combos on 517-day window ---")
    results_517 = []
    for i, kwargs in enumerate(best_short_combos):
        label_parts = []
        if kwargs.get('retention_bonus', 0) > 0:
            label_parts.append(f"RB={kwargs['retention_bonus']:.1f}")
        if kwargs.get('replace_threshold', 0) > 0:
            label_parts.append(f"RT={kwargs['replace_threshold']:.2f}")
        if kwargs.get('min_turnover_rate', 0) > 0:
            label_parts.append(f"MTR={kwargs['min_turnover_rate']:.1f}")
        if kwargs.get('sector_diversify', 0) > 0:
            label_parts.append(f"SD={kwargs['sector_diversify']}")
        label = "+".join(label_parts) if label_parts else "CPPI-only"

        r = run_test(reports_long, f"Long-{label}", **kwargs)
        if r:
            results_517.append((label, r, kwargs))
            s = r['summary']
            turn = r['details']['annual_turnover']
            sharpe = r['details']['sharpe_ratio']
            consist = r['details']['half_period_consistency']
            shl = r['details']['signal_half_life']
            not5_count = sum(1 for v in r['details'].values() if v['score'] < 5)
            print(f"    [{i+1}/{len(best_short_combos)}] {label:30s}: "
                  f"{r['score']}/105 ({r['grade']}) | {not5_count} gaps | "
                  f"Turn={turn['value']:.1f} Sharpe={sharpe['value']:.3f} "
                  f"Consist={consist['value']:.1%} HL={shl['value']:.1f}")

    # Print best result details
    if results_517:
        results_517.sort(key=lambda x: x[1]['score'], reverse=True)
        best_label, best_r, best_kwargs = results_517[0]
        print(f"\n{'='*90}")
        print(f"BEST 517-day result: {best_r['score']}/{best_r['max']} ({best_r['grade']})")
        print(f"Params: {best_kwargs}")
        print(f"Gaps:")
        print_gaps(best_r)

    # Save results
    out_file = PROJECT_ROOT / 'reports' / 'ralph_iter2_results.json'
    save_data = {
        'holding_days': brb.HOLDING_DAYS,
        'short_window_days': len(reports_short),
        'long_window_days': len(reports_long),
        'best_517d': {
            'score': results_517[0][1]['score'] if results_517 else 0,
            'grade': results_517[0][1]['grade'] if results_517 else 'N/A',
            'params': results_517[0][2] if results_517 else {},
            'details': {k: {'score': v['score'], 'value': v['value']}
                       for k, v in results_517[0][1]['details'].items()} if results_517 else {},
        } if results_517 else None,
    }
    with open(out_file, 'w') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_file}")

    print(f"\n{'='*90}")
    print("DONE")


if __name__ == '__main__':
    main()
