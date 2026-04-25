#!/usr/bin/env python3
"""ng1.0.62 regime 切换条件 A/B 对比 (扩展版 — 支持多窗口 + 多变体).

用法:
    # 2024-2026
    python3 scripts/regime_variant_compare.py --window 2024
    # 2020-2026 (full cycle)
    python3 scripts/regime_variant_compare.py --window 2020
    # 2018-2019 (pre-2020 OOS)
    python3 scripts/regime_variant_compare.py --window pre2020
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from backtest.backtest_report_based import load_reports, run_single_backtest

DB_PATH = PROJECT_ROOT / 'data_adapter' / 'stock_data.db'

# 各 window 对应的 bull/bear 报告目录
WINDOWS = {
    '2024': {
        'start': '2024-01-01', 'end': '2026-04-24',
        'bull_dir': 'reports/daily_selection_ng1.0.7_fast',
        'bear_dir': 'reports/daily_selection_ng104_ensemble_3seed',
    },
    '2020': {
        'start': '2020-01-01', 'end': '2026-04-24',
        'bull_dir': 'reports/daily_selection_ng1.0.7_fast',
        'bear_dir': 'reports/daily_selection_ng104_ensemble_3seed',
    },
    'pre2020': {
        'start': '2018-04-01', 'end': '2019-12-31',
        'bull_dir': 'reports/daily_selection_ng107_pre2020',
        'bear_dir': 'reports/daily_selection_ng104_pre2020',
    },
}


# ============================================================
# Regime 原子信号 (每个返回 bool 数组)
# ============================================================

def sig_position(var1, ma60, **_):
    """位置信号: var1 > ma60"""
    return var1 > ma60


def sig_macd_water(macd, **_):
    """MACD 水上: macd > 0"""
    return macd > 0


def sig_macd_cross(dif, dea, **_):
    """MACD 金叉/水上叉位: dif > dea"""
    return dif > dea


def sig_macd_rising(macd, **_):
    """MACD 柱上升: macd > macd_prev"""
    arr = np.zeros(len(macd), dtype=bool)
    arr[1:] = macd[1:] > macd[:-1]
    return arr


def sig_compass_align(c5, c13, c34, **_):
    """指南针对齐: c5 > c13 > c34 (三周期看多)"""
    return (c5 > c13) & (c13 > c34)


def sig_var1_above_c34(var1, c34, **_):
    """var1 > c34 (中期均线)"""
    return var1 > c34


def sig_panic_drop(var1, drop_thresh: float = -0.023, **_):
    """急跌信号: var1 单日跌幅 ≤ drop_thresh (默认 -2.3%)"""
    arr = np.zeros(len(var1), dtype=bool)
    arr[1:] = (var1[1:] - var1[:-1]) / (var1[:-1] + 1e-15) <= drop_thresh
    return arr


def sig_panic_streak2(var1, streak_thresh: float = -0.035, **_):
    """2 日累计跌信号: (var1[t]-var1[t-2])/var1[t-2] ≤ streak_thresh"""
    arr = np.zeros(len(var1), dtype=bool)
    arr[2:] = (var1[2:] - var1[:-2]) / (var1[:-2] + 1e-15) <= streak_thresh
    return arr


# ============================================================
# Regime 组合器
# ============================================================

def smooth_persistence(raw_bull: np.ndarray, n_days: int) -> np.ndarray:
    """需要 N 天连续才切换 (防震荡).

    Args:
        raw_bull: bool 数组, True=raw bull signal
        n_days: 持续 N 天才确认切换 (1=不平滑)
    """
    n = len(raw_bull)
    if n_days <= 1:
        return np.where(raw_bull, 1, -1)

    regime = np.zeros(n, dtype=int)
    regime[0] = 1 if raw_bull[0] else -1
    bull_streak = 0
    bear_streak = 0
    for i in range(n):
        if raw_bull[i]:
            bull_streak += 1
            bear_streak = 0
        else:
            bear_streak += 1
            bull_streak = 0
        prev = regime[i - 1] if i > 0 else regime[0]
        if prev == 1:
            # 已是牛, 需 bear_streak >= N 才切熊
            regime[i] = -1 if bear_streak >= n_days else 1
        else:
            # 已是熊, 需 bull_streak >= N 才切牛
            regime[i] = 1 if bull_streak >= n_days else -1
    return regime


# ============================================================
# 预定义 Regime 变体
# ============================================================

def regime_v1_simple(amv: dict) -> np.ndarray:
    raw = sig_position(**amv)
    return smooth_persistence(raw, 1)


def regime_v2_macd(amv: dict) -> np.ndarray:
    raw = sig_position(**amv) & sig_macd_water(**amv)
    return smooth_persistence(raw, 1)


def regime_v3_strict(amv: dict, slow_bear_days: int = 10) -> np.ndarray:
    """生产: 急涨切牛 + 急跌/缓跌切熊"""
    var1, ma60, macd = amv['var1'], amv['ma60'], amv['macd']
    n = len(var1)
    regime = np.zeros(n, dtype=int)
    pct = np.zeros(n)
    pct[1:] = (var1[1:] - var1[:-1]) / (var1[:-1] + 1e-15)
    regime[0] = 1 if var1[0] > ma60[0] else -1
    bear_streak = 0
    for i in range(1, n):
        prev = regime[i - 1]
        if prev == -1:
            bear_streak = 0
            bull_signal = (pct[i] >= 0.043 and var1[i] > ma60[i] and macd[i] > 0)
            regime[i] = 1 if bull_signal else -1
        else:
            bear_signal = (pct[i] <= -0.023 and var1[i] < ma60[i] and macd[i] < 0)
            if var1[i] < ma60[i] and macd[i] < 0:
                bear_streak += 1
            else:
                bear_streak = 0
            if bear_signal or bear_streak >= slow_bear_days:
                regime[i] = -1
                bear_streak = 0
            else:
                regime[i] = 1
    return regime


def regime_v4_macd_smooth2(amv: dict) -> np.ndarray:
    """V2 + 2 日平滑"""
    raw = sig_position(**amv) & sig_macd_water(**amv)
    return smooth_persistence(raw, 2)


def regime_v5_macd_smooth3(amv: dict) -> np.ndarray:
    """V2 + 3 日平滑"""
    raw = sig_position(**amv) & sig_macd_water(**amv)
    return smooth_persistence(raw, 3)


def regime_v6_macd_cross(amv: dict) -> np.ndarray:
    """位置 + MACD 金叉 (dif > dea)"""
    raw = sig_position(**amv) & sig_macd_cross(**amv)
    return smooth_persistence(raw, 1)


def regime_v7_full_alignment(amv: dict) -> np.ndarray:
    """位置 + MACD 水上 + 指南针 c5>c13>c34 三重对齐"""
    raw = sig_position(**amv) & sig_macd_water(**amv) & sig_compass_align(**amv)
    return smooth_persistence(raw, 1)


def regime_v8_majority(amv: dict) -> np.ndarray:
    """3 信号多数派: 位置 + MACD 水上 + 指南针对齐, 至少 2/3 才牛"""
    s1 = sig_position(**amv).astype(int)
    s2 = sig_macd_water(**amv).astype(int)
    s3 = sig_compass_align(**amv).astype(int)
    raw = (s1 + s2 + s3) >= 2
    return smooth_persistence(raw, 1)


def regime_v9_macd_rising(amv: dict) -> np.ndarray:
    """位置 + (MACD 水上 OR MACD 上升) — 容错版"""
    raw = sig_position(**amv) & (sig_macd_water(**amv) | sig_macd_rising(**amv))
    return smooth_persistence(raw, 1)


def regime_v11_loose_smooth3(amv: dict) -> np.ndarray:
    """V5 + V9 融合: 位置 + (水上 OR 上升) + 3 日平滑"""
    raw = sig_position(**amv) & (sig_macd_water(**amv) | sig_macd_rising(**amv))
    return smooth_persistence(raw, 3)


def regime_v12_v11_crisis_cash(amv: dict, crisis_thresh: float = 0.95) -> np.ndarray:
    """V11 + 危机现金: var1 < ma60 × thresh 时 → 0 (cash)"""
    base = regime_v11_loose_smooth3(amv)
    crisis = amv['var1'] < amv['ma60'] * crisis_thresh
    base[crisis] = 0
    return base


def regime_v13_v11_crisis_dd(amv: dict, dd_thresh: float = 0.85, lookback: int = 60) -> np.ndarray:
    """V11 + 60日滚动最大回撤 cash (深度下跌)"""
    base = regime_v11_loose_smooth3(amv)
    var1 = amv['var1']
    n = len(var1)
    rolling_max = np.zeros(n)
    for i in range(n):
        rolling_max[i] = var1[max(0, i - lookback + 1):i + 1].max()
    crisis = var1 < rolling_max * dd_thresh
    base[crisis] = 0
    return base


def regime_v14_v11_crisis_macd(amv: dict, macd_floor: float = -3.0) -> np.ndarray:
    """V11 + 极度负 MACD cash"""
    base = regime_v11_loose_smooth3(amv)
    crisis = amv['macd'] < macd_floor
    base[crisis] = 0
    return base


def regime_v15_v11_crisis_combo(amv: dict) -> np.ndarray:
    """V11 + 组合危机信号: var1<ma60×0.95 OR var1 60日drawdown>15% OR macd<-3"""
    base = regime_v11_loose_smooth3(amv)
    var1, ma60, macd = amv['var1'], amv['ma60'], amv['macd']
    n = len(var1)
    rolling_max = np.zeros(n)
    for i in range(n):
        rolling_max[i] = var1[max(0, i - 59):i + 1].max()
    crisis = (var1 < ma60 * 0.95) | (var1 < rolling_max * 0.85) | (macd < -3.0)
    base[crisis] = 0
    return base


def regime_v10_v2_asymmetric(amv: dict) -> np.ndarray:
    """V2 非对称: 切牛要 1 天, 切熊要 2 天 (牛市更敏感, 熊市更耐心)"""
    raw = sig_position(**amv) & sig_macd_water(**amv)
    n = len(raw)
    regime = np.zeros(n, dtype=int)
    regime[0] = 1 if raw[0] else -1
    bear_streak = 0
    for i in range(1, n):
        prev = regime[i - 1]
        if not raw[i]:
            bear_streak += 1
        else:
            bear_streak = 0
        if prev == -1:
            regime[i] = 1 if raw[i] else -1
        else:
            regime[i] = -1 if bear_streak >= 2 else 1
    return regime


def regime_v16_panic_immediate(amv: dict) -> np.ndarray:
    """V11 + 单日急跌 OR-trigger 强制熊"""
    base = regime_v11_loose_smooth3(amv)
    panic = sig_panic_drop(**amv)
    base[panic] = -1
    return base


def regime_v17_panic_cooldown_3d(amv: dict) -> np.ndarray:
    """V11 + 急跌触发未来 3 日强制熊 (含触发日)"""
    base = regime_v11_loose_smooth3(amv)
    panic = sig_panic_drop(**amv)
    n = len(base)
    for i in np.where(panic)[0]:
        for j in range(i, min(i + 3, n)):
            base[j] = -1
    return base


def regime_v18_panic_cash_3d(amv: dict) -> np.ndarray:
    """V11 + 急跌触发未来 3 日强制 cash (3-state)"""
    base = regime_v11_loose_smooth3(amv)
    panic = sig_panic_drop(**amv)
    n = len(base)
    for i in np.where(panic)[0]:
        for j in range(i, min(i + 3, n)):
            base[j] = 0
    return base


def regime_v19_panic_AND_position(amv: dict) -> np.ndarray:
    """V11 + 急跌 AND var1<ma60 强制熊"""
    base = regime_v11_loose_smooth3(amv)
    panic = sig_panic_drop(**amv)
    below = amv['var1'] < amv['ma60']
    base[panic & below] = -1
    return base


def regime_v20_panic_streak_2d(amv: dict) -> np.ndarray:
    """V11 + 2 日累计跌 ≤ -3.5% 强制熊"""
    base = regime_v11_loose_smooth3(amv)
    panic2 = sig_panic_streak2(**amv)
    base[panic2] = -1
    return base


VARIANTS = [
    ('V1 simple        位置', regime_v1_simple),
    ('V2 macd          位置+MACD水上', regime_v2_macd),
    ('V3 strict        生产 (急涨/急跌+缓跌)', regime_v3_strict),
    ('V4 v2+smooth2    V2+2日平滑', regime_v4_macd_smooth2),
    ('V5 v2+smooth3    V2+3日平滑', regime_v5_macd_smooth3),
    ('V6 macd_cross    位置+MACD金叉(dif>dea)', regime_v6_macd_cross),
    ('V7 full_align    位置+MACD水上+指南针对齐', regime_v7_full_alignment),
    ('V8 majority      3信号多数派(2/3)', regime_v8_majority),
    ('V9 macd_loose    位置+(MACD水上 OR 上升)', regime_v9_macd_rising),
    ('V10 v2_asym      V2非对称(切牛1日,切熊2日)', regime_v10_v2_asymmetric),
    ('V11 loose_sm3    位置+(水上 OR 上升)+3日平滑', regime_v11_loose_smooth3),
    ('V12 crisis_thresh V11+var1<ma60×0.95→cash', regime_v12_v11_crisis_cash),
    ('V13 crisis_dd     V11+60dDD>15%→cash', regime_v13_v11_crisis_dd),
    ('V14 crisis_macd   V11+macd<-3→cash', regime_v14_v11_crisis_macd),
    ('V15 crisis_combo  V11+三重危机OR→cash', regime_v15_v11_crisis_combo),
    ('V16 panic_imm    V11+单日-2.3%→bear', regime_v16_panic_immediate),
    ('V17 panic_cd3d   V11+急跌→未来3日bear', regime_v17_panic_cooldown_3d),
    ('V18 panic_cash3d V11+急跌→未来3日cash', regime_v18_panic_cash_3d),
    ('V19 panic_ANDpos V11+急跌AND var1<ma60→bear', regime_v19_panic_AND_position),
    ('V20 panic_str2d  V11+2日累计-3.5%→bear', regime_v20_panic_streak_2d),
]


# ============================================================
# 数据加载
# ============================================================

def load_amv_history() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(
            'SELECT trade_date, var1, amv_ma60, amv_macd, amv_dif, amv_dea, '
            'amv_c5, amv_c13, amv_c34 '
            'FROM market_amv ORDER BY trade_date',
            conn,
        )
    return df


def amv_to_dict(amv: pd.DataFrame) -> dict:
    return {
        'var1': amv['var1'].values,
        'ma60': amv['amv_ma60'].values,
        'macd': amv['amv_macd'].values,
        'dif': amv['amv_dif'].values,
        'dea': amv['amv_dea'].values,
        'c5': amv['amv_c5'].values,
        'c13': amv['amv_c13'].values,
        'c34': amv['amv_c34'].values,
    }


def merge_reports(bull_reports, bear_reports, regime_by_date, start, end):
    """合并报告. regime: 1=bull(取ng107), -1=bear(取ng104), 0=cash(跳过)."""
    merged = {}
    bull_n = bear_n = cash_n = miss_n = 0
    for date, r in regime_by_date.items():
        if date < start or date > end:
            continue
        if r == 0:
            cash_n += 1
            continue
        src = bull_reports if r == 1 else bear_reports
        if date in src:
            merged[date] = src[date]
            if r == 1:
                bull_n += 1
            else:
                bear_n += 1
        else:
            miss_n += 1
    return merged, bull_n, bear_n, miss_n, cash_n


# ============================================================
# 主流程
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', choices=list(WINDOWS.keys()), default='2024')
    ap.add_argument('--top-n', type=int, default=10)
    ap.add_argument('--focus-days', type=int, default=10)
    ap.add_argument('--rank-field', default='composite')
    ap.add_argument('--variants', default='all',
                    help='comma-separated subset, eg V1,V2,V3 (default: all)')
    ap.add_argument('--bull-dir', default=None,
                    help='override bull model report dir')
    ap.add_argument('--bear-dir', default=None,
                    help='override bear model report dir')
    ap.add_argument('--label-suffix', default='',
                    help='extra label suffix for output md filename')
    args = ap.parse_args()

    cfg = dict(WINDOWS[args.window])
    if args.bull_dir:
        cfg['bull_dir'] = args.bull_dir
    if args.bear_dir:
        cfg['bear_dir'] = args.bear_dir
    print(f'\n{"=" * 70}')
    print(f'  ng1.0.62 Regime 变体对比')
    print(f'  Window: {args.window} ({cfg["start"]} ~ {cfg["end"]})')
    print(f'  Bull  : {cfg["bull_dir"]}')
    print(f'  Bear  : {cfg["bear_dir"]}')
    print(f'  Top-{args.top_n}, focus={args.focus_days}d, rank={args.rank_field}')
    print(f'{"=" * 70}')

    print('\n[1] 加载 0AMV 历史...')
    amv = load_amv_history()
    print(f'  {len(amv)} 行  ({amv["trade_date"].iloc[0]} ~ {amv["trade_date"].iloc[-1]})')

    print(f'\n[2] 加载 bull reports: {cfg["bull_dir"]}')
    bull = load_reports(cfg['bull_dir'], rank_field=args.rank_field)
    print(f'  {len(bull)} 天')
    print(f'\n[3] 加载 bear reports: {cfg["bear_dir"]}')
    bear = load_reports(cfg['bear_dir'], rank_field=args.rank_field)
    print(f'  {len(bear)} 天')

    amv_dict = amv_to_dict(amv)
    dates = amv['trade_date'].values

    if args.variants == 'all':
        variants = VARIANTS
    else:
        wanted = {v.strip().upper() for v in args.variants.split(',')}
        variants = [(lbl, fn) for lbl, fn in VARIANTS
                    if lbl.split()[0].upper() in wanted]

    summary_rows = []
    for label, fn in variants:
        print(f'\n{"=" * 70}')
        print(f'  REGIME: {label}')
        print(f'{"=" * 70}')

        regime_arr = fn(amv_dict)
        regime_by_date = {str(d): int(r) for d, r in zip(dates, regime_arr)}
        win_dates = [d for d in regime_by_date if cfg['start'] <= d <= cfg['end']]
        win_regimes = [regime_by_date[d] for d in win_dates]
        bull_d = sum(1 for r in win_regimes if r == 1)
        bear_d = sum(1 for r in win_regimes if r == -1)
        cash_d = sum(1 for r in win_regimes if r == 0)
        switches = sum(
            1 for i in range(1, len(win_regimes))
            if win_regimes[i] != win_regimes[i - 1]
        )
        print(f'\n  窗口 regime: {len(win_regimes)}天, '
              f'牛{bull_d} ({100*bull_d/len(win_regimes):.0f}%) / '
              f'熊{bear_d} ({100*bear_d/len(win_regimes):.0f}%) / '
              f'cash{cash_d} ({100*cash_d/len(win_regimes):.0f}%) · '
              f'切换{switches}次')

        merged, bn, brn, missn, casn = merge_reports(bull, bear, regime_by_date,
                                                cfg['start'], cfg['end'])
        print(f'  合并: {len(merged)} 天 (bull={bn}, bear={brn}, cash={casn}, miss={missn})')

        if not merged or len(merged) < 30:
            print(f'  ⚠️ 报告太少, 跳过')
            continue

        result = run_single_backtest(
            merged, label, top_n=args.top_n, focus_days=args.focus_days,
        )
        s = result.get('summary', {}).get(args.focus_days, {})
        summary_rows.append({
            'variant': label,
            'bull_days': bull_d, 'bear_days': bear_d, 'cash_days': cash_d,
            'switches': switches,
            'annual_return': s.get('annual_return', 0) or 0,
            'net_annual': s.get('net_annual_return', 0) or 0,
            'sharpe': s.get('sharpe_ratio', 0) or 0,
            'sortino': s.get('sortino_ratio', 0) or 0,
            'calmar': s.get('calmar_ratio', 0) or 0,
            'max_drawdown': s.get('max_drawdown', 0) or 0,
            'monthly_win_rate': s.get('monthly_win_rate', 0) or 0,
            'excess_annual': s.get('excess_annual_return', 0) or 0,
            'turnover': s.get('annual_turnover', 0) or 0,
        })

    # 汇总
    print(f'\n\n{"=" * 130}')
    print(f'  汇总 ({args.window}: {cfg["start"]} ~ {cfg["end"]}, '
          f'Top-{args.top_n}, {args.focus_days}日, 无 CPPI)')
    print(f'{"=" * 130}')
    hdr = (f'{"Variant":<46} {"牛/熊":<10} {"切":>3} '
           f'{"年化%":>7} {"净%":>7} {"Sharpe":>7} {"Sortino":>8} '
           f'{"Calmar":>7} {"MaxDD%":>8} {"月胜%":>5} {"超额%":>7} {"换手":>7}')
    print(hdr)
    print('-' * 130)
    for r in summary_rows:
        print(
            f'{r["variant"]:<46} '
            f'{r["bull_days"]:>3}/{r["bear_days"]:<5} '
            f'{r["switches"]:>3} '
            f'{r["annual_return"]*100:>6.1f}% '
            f'{r["net_annual"]*100:>6.1f}% '
            f'{r["sharpe"]:>7.3f} '
            f'{r["sortino"]:>8.3f} '
            f'{r["calmar"]:>7.3f} '
            f'{r["max_drawdown"]*100:>7.1f}% '
            f'{r["monthly_win_rate"]:>4.0f}% '
            f'{r["excess_annual"]*100:>6.1f}% '
            f'{r["turnover"]:>6.1f}x'
        )

    # 写 markdown
    sfx = f'_{args.label_suffix}' if args.label_suffix else ''
    out_md = PROJECT_ROOT / 'reports' / f'regime_variant_compare_{args.window}{sfx}.md'
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, 'w') as f:
        f.write(f'# ng1.0.62 Regime 变体对比 — Window {args.window}\n\n')
        f.write(f'`{cfg["start"]} ~ {cfg["end"]}` · bull={cfg["bull_dir"].split("/")[-1]} · '
                f'bear={cfg["bear_dir"].split("/")[-1]} · Top-{args.top_n} · '
                f'{args.focus_days}d hold\n\n')
        f.write('| Variant | 牛/熊 | 切换 | 年化(毛) | 年化(净) | Sharpe | Sortino | '
                'Calmar | MaxDD | 月胜率 | 超额年化 | 年换手 |\n')
        f.write('|---|---|---|---|---|---|---|---|---|---|---|---|\n')
        for r in summary_rows:
            f.write(
                f'| {r["variant"]} | {r["bull_days"]}/{r["bear_days"]} | '
                f'{r["switches"]} | {r["annual_return"]*100:.1f}% | '
                f'{r["net_annual"]*100:.1f}% | '
                f'{r["sharpe"]:.3f} | {r["sortino"]:.3f} | {r["calmar"]:.3f} | '
                f'{r["max_drawdown"]*100:.1f}% | {r["monthly_win_rate"]:.0f}% | '
                f'{r["excess_annual"]*100:.1f}% | {r["turnover"]:.1f}x |\n'
            )
    print(f'\n✅ 报告: {out_md}')


if __name__ == '__main__':
    main()
