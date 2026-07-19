#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production overlay replay 回测 (2026-07-11, 风控内化 Phase 1)

背景: 生产 L1-L5 overlay + P0.1 sizing (stock_selctor/ng21_risk_overlay.py)
从未被任何回测引擎执行过 — 所有 headline 风控数字出自 backtest_report_based
的另一套平行实现 (VT 指数 EWMA 空间 / CPPI / checkpoint 止损), 参数互相矛盾。
本脚本直接 import 生产函数逐日重放, 首次把"生产实际执行的风控"与数字挂钩,
并在成分股 vol 空间 sweep VT 阈值 (含前瞻 vol 头 vs 60d 后视两种 est_vol 源)。

方法:
  每个报告日 D: 报告 JSON 排名列表 → build_risk_decision(regime@D)
  → apply_overlay_to_picks (L1/L2/L4) → est_vol (backward | forward-head)
  → compute_position_size (L3 VT + cash budget) → 组合 10d 前向收益
  = Σ w_i·ret_i (现金部分 0)。非重叠 10d 链式复利评估。
简化 (相对生产全链): 跳过 trust post_filters (历史 trust 标签不可复现) 与
booster (默认 off); regime 用 market_amv 现值 (含 4-25 preset repaint, 已知 caveat)。

用法:
  python3 scripts/production_overlay_replay.py --report-dir reports/daily_selection_ng106 \
      --start 2018-11-02 --end 2026-04-08
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.config import get_db_path
from stock_selctor.ng21_risk_overlay import (
    apply_overlay_to_picks, build_risk_decision, compute_position_size,
    estimate_portfolio_vol, estimate_portfolio_vol_forward,
)

HOLD_DAYS = 10


def load_report_picks(report_dir: Path, start: str, end: str) -> dict:
    """{date: ranked picks list} — 按报告内 score 降序。"""
    out = {}
    for f in sorted(report_dir.glob('analysis_data_*.json')):
        d = f.stem.replace('analysis_data_', '')
        d_iso = f"{d[:4]}-{d[4:6]}-{d[6:]}"
        if not (start <= d_iso <= end):
            continue
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        stocks = data.get('all_stocks_with_scores') or []
        stocks = [s for s in stocks if s.get('stock_code')]
        stocks.sort(key=lambda s: float(s.get('score', 0) or 0), reverse=True)
        # 只保留 overlay 需要的前 60 名 (L2 industry cap 后取 top_n=10 绰绰有余)
        out[d_iso] = [
            {'stock_code': s['stock_code'], 'score': s.get('score', 0),
             'industry': s.get('industry', '未知'),
             'exec_warning': s.get('exec_warning')}
            for s in stocks[:60]
        ]
    return out


def load_regime_map(db_path: str, source: str = 'v1') -> dict:
    """source='v1': market_amv.amv_regime; 'official': V11 on 官方活跃市值"""
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute('PRAGMA busy_timeout=30000')
    try:
        if source == 'official':
            from indicators.market_amv import compute_official_regime
            df = compute_official_regime(conn)
            return dict(zip(df['trade_date'], df['regime']))
        return dict(conn.execute('SELECT trade_date, amv_regime FROM market_amv'))
    finally:
        conn.close()


def perf(period_returns: list[float]) -> dict:
    """非重叠 10d 收益序列 → 年化/Sharpe/MaxDD (calendar-time 近似: 252/10 期/年)。"""
    r = np.asarray(period_returns, dtype=float)
    if len(r) < 8:
        return dict(ann=np.nan, sharpe=np.nan, maxdd=np.nan, n=len(r))
    ppy = 252 / HOLD_DAYS
    ann = float((1 + r.mean()) ** ppy - 1)
    sharpe = float(r.mean() / (r.std(ddof=1) + 1e-12) * np.sqrt(ppy))
    nav = np.cumprod(1 + r)
    maxdd = float((nav / np.maximum.accumulate(nav) - 1).min())
    return dict(ann=ann, sharpe=sharpe, maxdd=maxdd, n=len(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report-dir', default='reports/daily_selection_ng106')
    ap.add_argument('--start', default='2018-11-02')
    ap.add_argument('--end', default='2026-04-08')
    ap.add_argument('--vt-grid', default='0.15,0.20,0.25,0.30,0.35,0.45')
    ap.add_argument('--regime-source', choices=['v1', 'official'], default='v1',
                    help='v1=market_amv.amv_regime; official=官方活跃市值上跑V11')
    args = ap.parse_args()

    db = str(get_db_path())
    report_dir = ROOT / args.report_dir
    print(f"[1/4] 加载报告 {args.report_dir} {args.start}~{args.end} ...")
    picks_by_date = load_report_picks(report_dir, args.start, args.end)
    dates = sorted(picks_by_date)
    print(f"  {len(dates)} 个报告日")
    regime_map = load_regime_map(db, args.regime_source)
    print(f'  regime source: {args.regime_source}')

    print("[2/4] 未来收益 (复用北极星引擎批量查询, 新口径含复权)...")
    from backtest.backtest_report_based import batch_get_all_future_returns
    fut = batch_get_all_future_returns(dates, [HOLD_DAYS])
    # fut 键是买入日 (报告日次一交易日); 建 报告日→买入日 映射
    from backtest.backtest_report_based import _batch_get_next_trading_dates
    next_map = _batch_get_next_trading_dates(dates)

    print("[3/4] 逐日重放生产 overlay + sizing (含 fwd/bwd est_vol) ...")
    # 非重叠采样: 每 HOLD_DAYS 个报告日取一个
    rebal_dates = dates[::HOLD_DAYS]
    vt_grid = [float(x) for x in args.vt_grid.split(',')]
    # 序列容器: baseline / production replay / sweep {(vt, src): [...]}
    series = {'baseline_top10': [], 'prod_replay': []}
    sweep = {(vt, src): [] for vt in vt_grid for src in ('bwd', 'fwd')}
    fwd_missing = 0
    for d in rebal_dates:
        buy_d = next_map.get(d)
        rets = fut.get(buy_d, {}) if buy_d else {}
        if not rets:
            continue
        ranked = picks_by_date[d]
        regime = 'bull' if regime_map.get(d) == 1 else 'bear'

        def port_ret(weighted: list[tuple[str, float]]) -> float:
            """Σ w·ret_10d; 无数据的票该槽按现金 (0)。"""
            total = 0.0
            for code, w in weighted:
                r = rets.get(code.split('.')[0], {}).get(f'return_{HOLD_DAYS}d')
                if r is not None:
                    total += w * r
            return total

        # baseline: top10 等权满仓, 无 overlay
        base10 = [(s['stock_code'], 0.10) for s in ranked[:10]]
        series['baseline_top10'].append(port_ret(base10))

        # production replay: 真生产函数链
        decision = build_risk_decision(regime, d, db)
        kept, _dropped = apply_overlay_to_picks([dict(s) for s in ranked], decision)
        ev_bwd = estimate_portfolio_vol(kept, db, d)
        kept_sized = compute_position_size([dict(s) for s in kept], decision,
                                           est_portfolio_vol=ev_bwd)
        series['prod_replay'].append(port_ret(
            [(s['stock_code'], s.get('position_size', 0.0)) for s in kept_sized]))

        # VT sweep × est_vol 源
        ev_fwd = estimate_portfolio_vol_forward(kept, db, d)
        if ev_fwd is None:
            fwd_missing += 1
        for vt in vt_grid:
            for src, ev in (('bwd', ev_bwd), ('fwd', ev_fwd if ev_fwd else ev_bwd)):
                gross = min(1.0 - decision.cash_floor, vt / max(ev, 1e-6))
                n = max(len(kept), 1)
                w = min(gross / n, decision.pos_cap_per_stock)
                sweep[(vt, src)].append(port_ret(
                    [(s['stock_code'], w) for s in kept]))

    print(f"  重放 {len(series['prod_replay'])} 个非重叠期; fwd est_vol 缺失 {fwd_missing} 期(回退 bwd)")

    print("[4/4] 结果:\n")
    rows = [("baseline top10 等权满仓(无风控)", perf(series['baseline_top10'])),
            ("生产 replay (真 L1-L5+P0.1, bwd vol)", perf(series['prod_replay']))]
    for vt in vt_grid:
        for src in ('bwd', 'fwd'):
            rows.append((f"VT={vt:.2f} × {src}", perf(sweep[(vt, src)])))
    print(f"{'配置':<40}{'年化':>9}{'Sharpe':>9}{'MaxDD':>9}{'期数':>6}")
    for name, p in rows:
        print(f"{name:<40}{p['ann']:>9.1%}{p['sharpe']:>9.2f}{p['maxdd']:>9.1%}{p['n']:>6}")


if __name__ == '__main__':
    main()
