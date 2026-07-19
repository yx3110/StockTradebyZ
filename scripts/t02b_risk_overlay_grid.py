#!/usr/bin/env python3
"""T02-b: NAV 级风控网格扫描 (SL × VT × CPPI × RG-agg, 10d/15d 双口径).

在生产列表级 overlay 重放目录 (ng_production_overlay_replay.py 产物) 上叠加
NAV 级风控旋钮, 复现完整生产栈并选型。对照基线 = 全旋钮关闭 (纯列表级重放)。

选型纪律 (2026-07-19 研究横切条款, 代码写死不商量):
  - 排名只看 MaxDD / Calmar / 累积收益; 年化 Sharpe 仅记录不参与排名
  - Sharpe > 4 标记 suspect_annualization 并从排名剔除 (稀疏 trading 膨胀陷阱)
  - 平均 exposure < 0.5 的配置标记 cash_heavy (年化数字不可信)
  - 网格严格限于 NAV 后视旋钮, 不含任何 regime-conditional 阈值 (panic grid 前科)

前置: T02-pre 已修卖出跌停/停牌顺延 + vix_p90 expanding (2026-07-19), 网格数字
建立在保守卖出假设上。

--profile t01a (T01-a 换手链扫参): 在裸信号目录上扫 retention/EMA/买卖阈值,
关注 annual_turnover / annual_cost_drag / 净 Sharpe / MaxDD。同一执行器两个
档位共用指标提取与纪律断言 (文件名保留 t02b 前缀是历史原因)。

Usage:
  caffeinate -i python3 scripts/t02b_risk_overlay_grid.py \
      [--profile t02b|t01a] [--report-dir ...] [--focus-days 10 15] [--out ...]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

ALIGNED_START, ALIGNED_END = '2018-11-02', '2026-04-08'  # 口径 tag 2026-07-11-p0fix
SHARPE_RED_FLAG = 4.0
CASH_HEAVY_EXPOSURE = 0.5

# (name, kwargs) — 单轴扫描 + ng1.0.5 canonical + canonical 邻域
GRID: list[tuple[str, dict]] = [
    ('base', {}),
    ('sl4', {'stop_loss_pct': 0.04}),
    ('sl6', {'stop_loss_pct': 0.06}),
    ('sl8', {'stop_loss_pct': 0.08}),
    ('vt15', {'vol_target': 0.15}),
    ('vt20', {'vol_target': 0.20}),
    ('vt25', {'vol_target': 0.25}),
    ('cppi_08_20', {'cppi_floor': 0.08, 'cppi_multiplier': 20}),
    ('cppi_10_10', {'cppi_floor': 0.10, 'cppi_multiplier': 10}),
    ('cppi_08_15', {'cppi_floor': 0.08, 'cppi_multiplier': 15}),
    ('rg_agg', {'regime_gate_aggressive': True}),
    ('ng105', {'stop_loss_pct': 0.06, 'vol_target': 0.20,
               'cppi_floor': 0.08, 'cppi_multiplier': 20,
               'regime_gate_aggressive': True}),
    ('ng105_sl4', {'stop_loss_pct': 0.04, 'vol_target': 0.20,
                   'cppi_floor': 0.08, 'cppi_multiplier': 20,
                   'regime_gate_aggressive': True}),
    ('ng105_sl8', {'stop_loss_pct': 0.08, 'vol_target': 0.20,
                   'cppi_floor': 0.08, 'cppi_multiplier': 20,
                   'regime_gate_aggressive': True}),
    ('ng105_vt15', {'stop_loss_pct': 0.06, 'vol_target': 0.15,
                    'cppi_floor': 0.08, 'cppi_multiplier': 20,
                    'regime_gate_aggressive': True}),
    ('ng105_vt25', {'stop_loss_pct': 0.06, 'vol_target': 0.25,
                    'cppi_floor': 0.08, 'cppi_multiplier': 20,
                    'regime_gate_aggressive': True}),
    ('ng105_cppi1010', {'stop_loss_pct': 0.06, 'vol_target': 0.20,
                        'cppi_floor': 0.10, 'cppi_multiplier': 10,
                        'regime_gate_aggressive': True}),
    ('ng105_norg', {'stop_loss_pct': 0.06, 'vol_target': 0.20,
                    'cppi_floor': 0.08, 'cppi_multiplier': 20}),
]

# T01-a 换手链: retention / EMA / ng1.0.8 买卖阈值 (裸信号目录上扫)
GRID_T01A: list[tuple[str, dict]] = [
    ('base', {}),
    ('ret10', {'retention_bonus': 0.10}),
    ('ret20', {'retention_bonus': 0.20}),
    ('ret30', {'retention_bonus': 0.30}),
    ('ema50', {'ema_alpha': 0.5}),
    ('ema70', {'ema_alpha': 0.7}),
    ('ema85', {'ema_alpha': 0.85}),
    ('sell30', {'sell_threshold': 30}),
    ('sell50', {'sell_threshold': 50}),
    ('bt15_sell50', {'buy_threshold': 15, 'sell_threshold': 50}),
    ('ret20_ema70', {'retention_bonus': 0.20, 'ema_alpha': 0.7}),
    ('ret20_ema70_sell50', {'retention_bonus': 0.20, 'ema_alpha': 0.7,
                            'sell_threshold': 50}),
    ('ret20_sell50', {'retention_bonus': 0.20, 'sell_threshold': 50}),
    ('ema70_sell50', {'ema_alpha': 0.7, 'sell_threshold': 50}),
]

PROFILES = {
    't02b': {'grid': GRID,
             'report_dir': 'reports/daily_selection_ng101_3seed_prodoverlay'},
    't01a': {'grid': GRID_T01A,
             'report_dir': 'reports/daily_selection_ng101_3seed'},
}

# summary dict 里 NAV 链指标的键名候选 (已对照评分卡读取处核实, 取第一个存在的)
METRIC_KEYS = {
    'max_drawdown': ('max_drawdown',),
    'sharpe': ('sharpe_ratio', 'sharpe'),
    'annual_return': ('annual_return',),
    'cumulative_return': ('cumulative_return', 'total_return'),
    'cvar_5': ('cvar_5pct',),
    'annual_turnover': ('annual_turnover',),
    'excess_annual': ('excess_annual_return',),
    'excess_max_drawdown': ('excess_max_drawdown',),
    'annual_cost_drag': ('annual_cost_drag',),
    'calmar_native': ('calmar_ratio',),
}


def _pick(s: dict, names: tuple) -> float | None:
    for n in names:
        if n in s and s[n] is not None:
            return float(s[n])
    return None


def run_one(report_dir: str, name: str, kwargs: dict, focus_days: int,
            cache, quiet: bool) -> dict:
    from backtest.run_north_star_eval import run_backtest
    t0 = time.time()
    stdout_sink = io.StringIO() if quiet else sys.stdout
    with contextlib.redirect_stdout(stdout_sink):
        result = run_backtest(
            report_dir, f't02b:{name}:{focus_days}d',
            top_n=10, focus_days=focus_days, rank_field='composite',
            start_date=ALIGNED_START, end_date=ALIGNED_END,
            cache=cache, **kwargs,
        )
    if not result:
        return {'name': name, 'focus_days': focus_days, 'error': 'no result'}

    s = result['summary'].get(focus_days, {})
    row = {'name': name, 'focus_days': focus_days, 'config': kwargs,
           'elapsed_sec': round(time.time() - t0, 1)}
    for out_key, cands in METRIC_KEYS.items():
        row[out_key] = _pick(s, cands)

    # exposure 统计 (cash 纪律) + 窗口交易日数
    df = result.get('daily_results')
    n_days = 0
    try:
        sub = df[df['days'] == focus_days]
        n_days = int(sub['date'].nunique())
        row['avg_exposure'] = round(float(sub['exposure'].mean()), 3)
        row['pct_days_exposure_lt50'] = round(
            float((sub['exposure'] < CASH_HEAVY_EXPOSURE).mean()), 3)
    except Exception:
        row['avg_exposure'] = None

    # V5.2 总分 (程序化, 与 _print_scorecard_v52 同参: 日数 + 风格适配)
    try:
        from backtest.north_star_metrics import compute_v52_score
        v52 = compute_v52_score(
            s, n_trading_days=n_days,
            median_market_cap_bn=s.get('median_market_cap_bn', 0) or 0)
        row['v52_pct'] = round(float(v52['final_pct']), 2)
        row['v52_grade'] = v52.get('grade')
    except Exception as e:
        row['v52_error'] = str(e)

    # Calmar: 优先 summary 原生 calmar_ratio, 缺失才自算
    ann, mdd = row.get('annual_return'), row.get('max_drawdown')
    row['calmar'] = (round(row['calmar_native'], 2)
                     if row.get('calmar_native')
                     else round(ann / abs(mdd), 2) if ann and mdd else None)

    # 纪律标记
    row['suspect_annualization'] = bool(
        row.get('sharpe') is not None and row['sharpe'] > SHARPE_RED_FLAG)
    row['cash_heavy'] = bool(
        row.get('avg_exposure') is not None
        and row['avg_exposure'] < CASH_HEAVY_EXPOSURE)
    row['rankable'] = not (row['suspect_annualization'] or row['cash_heavy'])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', choices=list(PROFILES), default='t02b')
    ap.add_argument('--report-dir', default=None,
                    help='默认取 profile 对应目录')
    ap.add_argument('--focus-days', type=int, nargs='+', default=[10, 15])
    ap.add_argument('--out', default=None,
                    help='默认 reports/system_evaluation/{profile}_grid_20260719.json')
    ap.add_argument('--only', nargs='+', default=None,
                    help='仅跑指定配置名 (调试用)')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    prof = PROFILES[args.profile]
    if args.report_dir is None:
        args.report_dir = prof['report_dir']
    if args.out is None:
        args.out = f'reports/system_evaluation/{args.profile}_grid_20260719.json'

    from backtest.eval_cache import EvalCache
    cache = EvalCache()

    grid = [(n, k) for n, k in prof['grid'] if not args.only or n in args.only]
    rows = []
    total = len(grid) * len(args.focus_days)
    i = 0
    for fd in args.focus_days:
        for name, kwargs in grid:
            i += 1
            row = run_one(args.report_dir, name, kwargs, fd, cache,
                          quiet=not args.verbose)
            rows.append(row)
            flag = ('⚠️sharpe>4' if row.get('suspect_annualization') else
                    '⚠️cash' if row.get('cash_heavy') else '')
            print(f'[{i}/{total}] {name} {fd}d: '
                  f'MaxDD={row.get("max_drawdown")}, '
                  f'Calmar={row.get("calmar")}, '
                  f'Sharpe={row.get("sharpe")}, '
                  f'V5.2={row.get("v52_pct")}% '
                  f'({row.get("elapsed_sec")}s) {flag}', flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as fh:
        json.dump({'report_dir': args.report_dir,
                   'window': [ALIGNED_START, ALIGNED_END],
                   'engine_note': 'T02-pre 卖出跌停顺延+vix_p90 expanding 已生效',
                   'rows': rows}, fh, ensure_ascii=False, indent=1)

    # 排名表 (只对 rankable 行)
    # t02b 风控旋钮: MaxDD 优先, Calmar 次之 (研究横切纪律)
    # t01a 换手旋钮: 净 Sharpe 优先 (已含成本), 换手低者优先
    for fd in args.focus_days:
        cand = [r for r in rows if r['focus_days'] == fd and r.get('rankable')
                and r.get('max_drawdown') is not None]
        if args.profile == 't01a':
            cand.sort(key=lambda r: (r.get('sharpe') or 0,
                                     -(r.get('annual_turnover') or 999)),
                      reverse=True)
            title = '净Sharpe 优先, 换手次之'
        else:
            # reverse=True 下两键同向: MaxDD 负值越大越好, Calmar 越大越好
            cand.sort(key=lambda r: (r['max_drawdown'], r['calmar'] or 0),
                      reverse=True)
            title = 'MaxDD 优先, Calmar 次之'
        print(f'\n=== {fd}d 口径排名 ({title}; 已剔除可疑行) ===')
        for r in cand[:10]:
            print(f'  {r["name"]:>18}: MaxDD={r["max_drawdown"]:.3f} '
                  f'Calmar={r["calmar"]} Sharpe={round(r["sharpe"], 3) if r["sharpe"] else None} '
                  f'换手={r.get("annual_turnover")} '
                  f'成本={r.get("annual_cost_drag")} '
                  f'V5.2={r.get("v52_pct")}%')
    print(f'\n结果已存 {out}')


if __name__ == '__main__':
    main()
