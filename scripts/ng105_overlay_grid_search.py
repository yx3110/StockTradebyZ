#!/usr/bin/env python3
"""NG v1.0.5 Overlay网格搜索: 止损 × Regime门控 × VolTarget × CPPI 共96种组合

目标: MaxDD < -10%, Sharpe > 1.5, 年化 > 30%, 换手 < 20x
基线: ng1.0.1 + CPPI(F0.08, M20) — MaxDD=-12.6%, Sharpe=2.339
"""

import sys, os, io, contextlib, json, itertools
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')

from backtest import backtest_report_based as brb
from backtest import north_star_metrics as nsm

nsm.DB_PATH = DB_PATH
brb.DB_PATH = DB_PATH

# ═══════════════════════════════════════════════
# 参数网格 (spec: 4 × 3 × 4 × 2 = 96)
# ═══════════════════════════════════════════════
STOP_LOSS = [0, 0.06, 0.08, 0.10]          # 0=off, 6%, 8%, 10%
REGIME_GATE = ['off', 'standard', 'aggressive']
VOL_TARGET = [0, 0.15, 0.20, 0.25]         # 0=off, 15%, 20%, 25%
CPPI_CONFIGS = [(0.07, 20), (0.08, 20)]    # (floor, multiplier)

# 固定参数 (ng1.0.1生产配置)
REPORT_DIR = str(PROJECT_ROOT / 'reports' / 'daily_selection_ng101')
TOP_N = 10
FOCUS_DAYS = 10
BENCHMARK = '000905.SH'
RANK_FIELD = 'score'

# 目标阈值
TARGET_MAXDD = -0.10    # MaxDD < -10%
TARGET_SHARPE = 1.5
TARGET_ANNUAL = 0.30    # 年化 > 30%
TARGET_TURNOVER = 20    # 换手 < 20x


def run_combo(reports, stop_loss, regime, vol_target, cppi_floor, cppi_mult, cache=None):
    """运行单个参数组合，返回摘要字典"""
    # regime_gate_aggressive: only when 'aggressive'
    # For 'standard': existing regime_damping_map handles it (active when cppi_floor > 0)
    # For 'off': no extra gate, but regime_damping still active if cppi_floor > 0
    #   → We keep it as-is since regime_damping is integral to CPPI behavior
    regime_aggressive = (regime == 'aggressive')

    with contextlib.redirect_stdout(io.StringIO()):
        r = brb.run_single_backtest(
            reports, "grid",
            top_n=TOP_N,
            benchmark_code=BENCHMARK,
            focus_days=FOCUS_DAYS,
            cppi_floor=cppi_floor,
            cppi_multiplier=cppi_mult,
            vol_target=vol_target,
            stop_loss_pct=stop_loss,
            regime_gate_aggressive=regime_aggressive,
            cache=cache,
        )
    if r is None:
        return None
    return r['summary'].get(FOCUS_DAYS, {})


def fmt_pct(v, width=7):
    return f"{v*100:>{width}.1f}%"


def main():
    from backtest.eval_cache import EvalCache
    cache = EvalCache()

    # 预加载报告 (只做一次)
    print(f"加载报告: {REPORT_DIR}")
    reports = brb.load_reports(REPORT_DIR, rank_field=RANK_FIELD, cache=cache)
    if not reports:
        print("无报告!")
        return
    print(f"已加载 {len(reports)} 天报告\n")

    combos = list(itertools.product(STOP_LOSS, REGIME_GATE, VOL_TARGET, CPPI_CONFIGS))
    total = len(combos)
    print(f"网格规模: {len(STOP_LOSS)} × {len(REGIME_GATE)} × {len(VOL_TARGET)} × {len(CPPI_CONFIGS)} = {total} 组合")
    print(f"目标: MaxDD > {TARGET_MAXDD:.0%}, Sharpe > {TARGET_SHARPE}, 年化 > {TARGET_ANNUAL:.0%}, 换手 < {TARGET_TURNOVER}x")
    print(f"{'='*110}")

    header = (f"{'#':>3s} {'SL':>5s} {'Regime':>10s} {'VolTgt':>7s} {'CPPI':>10s} │ "
              f"{'年化(毛)':>8s} {'年化(净)':>8s} {'MaxDD':>8s} {'Sharpe':>8s} "
              f"{'换手':>6s} {'ICIR':>6s} {'Pass':>4s}")
    print(header)
    print(f"{'─'*110}")

    results = []
    for idx, (sl, rg, vt, (cf, cm)) in enumerate(combos, 1):
        print(f"  [{idx:>3d}/{total}] SL={sl:.0%} RG={rg:>10s} VT={vt:.0%} CPPI=F{cf}/M{cm}...",
              end='', flush=True)

        s = run_combo(reports, sl, rg, vt, cf, cm, cache=cache)
        if s is None:
            print(" FAILED")
            continue

        annual_gross = s.get('gross_annual_return', s.get('annual_return', 0))
        annual_net = s.get('net_annual_return', 0)
        maxdd = s.get('max_drawdown', 0)
        sharpe = s.get('sharpe_ratio', 0)
        turnover = s.get('annual_turnover', 0)
        icir = s.get('icir', 0)

        # 检查是否达标
        passed = (maxdd > TARGET_MAXDD and sharpe > TARGET_SHARPE
                  and annual_gross > TARGET_ANNUAL and turnover < TARGET_TURNOVER)
        pass_str = "YES" if passed else ""

        results.append({
            'stop_loss': sl, 'regime': rg, 'vol_target': vt,
            'cppi_floor': cf, 'cppi_mult': cm,
            'annual_gross': annual_gross, 'annual_net': annual_net,
            'maxdd': maxdd, 'sharpe': sharpe, 'turnover': turnover,
            'icir': icir, 'passed': passed,
        })

        sl_str = f"{sl:.0%}" if sl > 0 else "off"
        vt_str = f"{vt:.0%}" if vt > 0 else "off"
        print(f"\r{idx:>3d} {sl_str:>5s} {rg:>10s} {vt_str:>7s} F{cf:.2f}/M{cm:<3d} │ "
              f"{fmt_pct(annual_gross)} {fmt_pct(annual_net)} {fmt_pct(maxdd)} "
              f"{sharpe:>8.3f} {turnover:>5.1f}x {icir:>6.3f} {pass_str:>4s}")

    print(f"\n{'='*110}")

    # ═══════════════════════════════════════════════
    # 排序并输出Top 20 (按MaxDD优先，然后Sharpe)
    # ═══════════════════════════════════════════════
    # 排序: MaxDD接近0为好(降序), 然后Sharpe(降序)
    results.sort(key=lambda x: (x['maxdd'], x['sharpe']), reverse=True)

    passed_results = [r for r in results if r['passed']]
    print(f"\n达标组合: {len(passed_results)}/{len(results)}")
    print(f"  条件: MaxDD > {TARGET_MAXDD:.0%} AND Sharpe > {TARGET_SHARPE} "
          f"AND 年化 > {TARGET_ANNUAL:.0%} AND 换手 < {TARGET_TURNOVER}x")

    print(f"\n{'='*110}")
    print(f"Top 20 (按MaxDD→Sharpe排序):")
    print(f"{'─'*110}")
    print(header)
    print(f"{'─'*110}")

    for i, r in enumerate(results[:20], 1):
        sl_str = f"{r['stop_loss']:.0%}" if r['stop_loss'] > 0 else "off"
        vt_str = f"{r['vol_target']:.0%}" if r['vol_target'] > 0 else "off"
        pass_str = "YES" if r['passed'] else ""
        print(f"{i:>3d} {sl_str:>5s} {r['regime']:>10s} {vt_str:>7s} "
              f"F{r['cppi_floor']:.2f}/M{r['cppi_mult']:<3d} │ "
              f"{fmt_pct(r['annual_gross'])} {fmt_pct(r['annual_net'])} "
              f"{fmt_pct(r['maxdd'])} {r['sharpe']:>8.3f} "
              f"{r['turnover']:>5.1f}x {r['icir']:>6.3f} {pass_str:>4s}")

    # 保存完整结果到JSON
    out_json = PROJECT_ROOT / 'reports' / 'ng105_overlay_grid_results.json'
    with open(out_json, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'grid_spec': {
                'stop_loss': STOP_LOSS,
                'regime_gate': REGIME_GATE,
                'vol_target': VOL_TARGET,
                'cppi': [list(c) for c in CPPI_CONFIGS],
            },
            'fixed': {
                'report_dir': REPORT_DIR,
                'top_n': TOP_N, 'focus_days': FOCUS_DAYS,
                'benchmark': BENCHMARK, 'rank_field': RANK_FIELD,
            },
            'targets': {
                'maxdd': TARGET_MAXDD, 'sharpe': TARGET_SHARPE,
                'annual': TARGET_ANNUAL, 'turnover': TARGET_TURNOVER,
            },
            'n_passed': len(passed_results),
            'n_total': len(results),
            'results': results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_json}")

    # 如果有达标组合，推荐最优
    if passed_results:
        # 达标组合中Sharpe最高的
        best = max(passed_results, key=lambda x: x['sharpe'])
        sl_str = f"{best['stop_loss']:.0%}" if best['stop_loss'] > 0 else "off"
        vt_str = f"{best['vol_target']:.0%}" if best['vol_target'] > 0 else "off"
        print(f"\n{'='*110}")
        print(f"推荐配置 (达标+最高Sharpe):")
        print(f"  止损: {sl_str}")
        print(f"  Regime门控: {best['regime']}")
        print(f"  VolTarget: {vt_str}")
        print(f"  CPPI: floor={best['cppi_floor']}, multiplier={best['cppi_mult']}")
        print(f"  → 年化={best['annual_gross']*100:.1f}%, MaxDD={best['maxdd']*100:.1f}%, "
              f"Sharpe={best['sharpe']:.3f}, 换手={best['turnover']:.1f}x")
    else:
        print(f"\n⚠️ 无达标组合! 考虑放宽目标或增加overlay强度")


if __name__ == '__main__':
    main()
