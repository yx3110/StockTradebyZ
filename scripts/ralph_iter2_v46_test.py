#!/usr/bin/env python3
"""Ralph Iter2: Test V4.6_step2 (86/105 baseline) with HOLDING_DAYS=20 fix.

Key question: Does adding 20d holding period improve signal_half_life?
Also test turnover reduction and liquidity improvements.
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

print(f"HOLDING_DAYS = {brb.HOLDING_DAYS}")
assert 20 in brb.HOLDING_DAYS, "Need 20 in HOLDING_DAYS"


def compute_v2(s):
    from backtest.north_star_metrics import NORTH_STAR_TARGETS_V2, score_metric_v2, compute_v2_grade
    mapping = {
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
        val = mapping.get(key)
        if val is None:
            val = 0
        sc, _ = score_metric_v2(val, cfg)
        total += sc
        max_score += 5
        details[key] = {'score': sc, 'value': val}
    grade = compute_v2_grade(total, max_score)
    return total, max_score, grade, details


def run(reports, label, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = brb.run_single_backtest(
            reports, label, top_n=10, benchmark_code='000905.SH',
            focus_days=10, **kw)
    if r is None:
        return None
    s = r['summary'].get(10, {})
    sc, mx, gr, det = compute_v2(s)
    # Also get ICIR profile for all holding periods
    icir_profile = {}
    for days in brb.HOLDING_DAYS:
        if days in r['summary']:
            icir_profile[days] = r['summary'][days].get('icir', 0)
    return {'score': sc, 'max': mx, 'grade': gr, 'details': det,
            'summary': s, 'icir_profile': icir_profile}


def show(label, r):
    if r is None:
        print(f"  {label}: FAILED")
        return
    print(f"  {label}: {r['score']}/{r['max']} ({r['grade']})")
    # ICIR profile
    print(f"    ICIR profile: {', '.join(f'{d}d={v:.3f}' for d, v in sorted(r['icir_profile'].items()))}")
    # Show signal half-life
    shl = r['details']['signal_half_life']
    print(f"    Signal half-life: {shl['value']:.1f} days ({shl['score']}/5)")
    # Show all non-5 metrics
    gaps = [(k, v) for k, v in r['details'].items() if v['score'] < 5]
    if gaps:
        gaps.sort(key=lambda x: x[1]['score'])
        print(f"    Non-perfect metrics ({len(gaps)}):")
        for k, v in gaps:
            val = v['value']
            stars = '★' * v['score'] + '☆' * (5 - v['score'])
            if isinstance(val, float):
                print(f"      {k:30s} {val:>10.4f}  {v['score']}/5 {stars}")
            else:
                print(f"      {k:30s} {val!s:>10s}  {v['score']}/5 {stars}")
    else:
        print(f"    ALL METRICS PERFECT! 🎯")


def main():
    report_dir = str(PROJECT_ROOT / 'reports' / 'daily_selection_v4.6_step2')
    print(f"Loading: {report_dir}")
    reports = brb.load_reports(report_dir)
    print(f"Loaded {len(reports)} days\n")

    # Test 1: Pure CPPI baseline (was 86/105 before HOLDING_DAYS=20 change)
    print("=" * 80)
    print("Test 1: V4.6+CPPI baseline (F0.05/M20)")
    r1 = run(reports, "V4.6-CPPI", cppi_floor=0.05, cppi_multiplier=20)
    show("V4.6+CPPI", r1)

    # Test 2: CPPI + replace_threshold (reduce turnover)
    print("\n" + "=" * 80)
    print("Test 2: + replace_threshold (reduce turnover)")
    for rt in [0.05, 0.10, 0.15, 0.20, 0.30]:
        r = run(reports, f"V4.6-RT{rt}",
                cppi_floor=0.05, cppi_multiplier=20, replace_threshold=rt)
        if r:
            turn = r['details']['annual_turnover']
            sharpe = r['details']['sharpe_ratio']
            shl = r['details']['signal_half_life']
            consist = r['details']['half_period_consistency']
            print(f"    RT={rt:.2f}: {r['score']}/105 ({r['grade']}) | "
                  f"Turn={turn['value']:.1f}({turn['score']}) "
                  f"Sharpe={sharpe['value']:.3f}({sharpe['score']}) "
                  f"HL={shl['value']:.1f}({shl['score']}) "
                  f"Consist={consist['value']:.1%}({consist['score']})")

    # Test 3: CPPI + min_turnover_rate (liquidity)
    print("\n" + "=" * 80)
    print("Test 3: + min_turnover_rate (improve liquidity)")
    for mtr in [0.5, 1.0, 2.0]:
        r = run(reports, f"V4.6-MTR{mtr}",
                cppi_floor=0.05, cppi_multiplier=20, min_turnover_rate=mtr)
        if r:
            liq = r['details']['liquidity_coverage']
            mono = r['details']['ic_monotonicity']
            print(f"    MTR={mtr:.1f}: {r['score']}/105 ({r['grade']}) | "
                  f"Liq={liq['value']:.1%}({liq['score']}) "
                  f"Mono={mono['value']:.3f}({mono['score']})")

    # Test 4: Combined best
    print("\n" + "=" * 80)
    print("Test 4: Combined (CPPI + RT + MTR + SD)")
    combos = [
        ("CPPI+RT10", dict(cppi_floor=0.05, cppi_multiplier=20, replace_threshold=0.10)),
        ("CPPI+RT15", dict(cppi_floor=0.05, cppi_multiplier=20, replace_threshold=0.15)),
        ("CPPI+RT10+MTR1", dict(cppi_floor=0.05, cppi_multiplier=20,
                                replace_threshold=0.10, min_turnover_rate=1.0)),
        ("CPPI+RT10+SD2", dict(cppi_floor=0.05, cppi_multiplier=20,
                               replace_threshold=0.10, sector_diversify=2)),
        ("CPPI+RT15+MTR1+SD2", dict(cppi_floor=0.05, cppi_multiplier=20,
                                     replace_threshold=0.15, min_turnover_rate=1.0,
                                     sector_diversify=2)),
        ("CPPI+RT20+MTR1", dict(cppi_floor=0.05, cppi_multiplier=20,
                                replace_threshold=0.20, min_turnover_rate=1.0)),
    ]
    best_score = 0
    best_result = None
    for label, params in combos:
        r = run(reports, label, **params)
        if r:
            not5 = sum(1 for v in r['details'].values() if v['score'] < 5)
            s = r['summary']
            print(f"    {label:25s}: {r['score']}/105 ({r['grade']}) | {not5} gaps | "
                  f"AnnRet={s.get('gross_annual_return', s.get('annual_return', 0))*100:.1f}% "
                  f"Turn={s.get('annual_turnover',0):.1f}x")
            if r['score'] > best_score:
                best_score = r['score']
                best_result = (label, r, params)

    if best_result:
        print(f"\n{'='*80}")
        print(f"BEST OVERALL: {best_result[0]} → {best_result[1]['score']}/105 ({best_result[1]['grade']})")
        show(best_result[0], best_result[1])

    print("\nDONE")


if __name__ == '__main__':
    main()
