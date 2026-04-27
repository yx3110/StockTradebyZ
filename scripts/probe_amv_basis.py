"""
0AMV 基准探针 — 比较不同 amount 基准 / SMA 窗口下 V11 regime 切换的回测效果。

候选基准:
  A. 现行: SH(000001) + SZ成指(399001), N=10
  B. 全A加总 (daily_quotes WHERE type='A股'), N=10
  C. 中证全指(000985), N=10
  D. 现行 SH+SZ成指, N=15
  E. 全A加总, N=5
  F. 中证全指, N=15
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import numpy as np
import pandas as pd

from backtest.backtest_report_based import load_reports, run_single_backtest, compute_ns_scores
from backtest.regime_switch_backtest import merge_reports_by_regime
from indicators.market_amv import tdx_sma, ema, ma
from indicators.regime_classifier import PRESETS

V11 = PRESETS['v11_loose_smooth3']

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  'data_adapter', 'stock_data.db')

TOP_N = 10
FOCUS_DAYS = 10
BULL_DIR = 'reports/daily_selection_ng101'
BEAR_DIR = 'reports/daily_selection_ng104_ensemble_3seed'

QUERIES = {
    'sh_sz_idx': """
        SELECT dq.trade_date, SUM(dq.amount) AS amt
        FROM daily_quotes dq JOIN securities s ON dq.security_id=s.id
        WHERE s.code IN ('000001.SH','399001.SZ') AND dq.amount>0
        GROUP BY dq.trade_date HAVING COUNT(*)=2 ORDER BY dq.trade_date
    """,
    'all_a': """
        SELECT dq.trade_date, SUM(dq.amount) AS amt
        FROM daily_quotes dq JOIN securities s ON dq.security_id=s.id
        WHERE s.type='A股' AND dq.amount>0
        GROUP BY dq.trade_date ORDER BY dq.trade_date
    """,
    'csi_all': """
        SELECT dq.trade_date, dq.amount AS amt
        FROM daily_quotes dq JOIN securities s ON dq.security_id=s.id
        WHERE s.code='000985.SH' AND dq.amount>0 ORDER BY dq.trade_date
    """,
}

CANDIDATES = [
    ('A.SH+SZidx, N=10 (baseline)', 'sh_sz_idx', 10),
    ('B.AllA, N=10',                'all_a',     10),
    ('C.000985(中证全指), N=10',     'csi_all',   10),
    ('D.SH+SZidx, N=15',            'sh_sz_idx', 15),
    ('E.AllA, N=5',                 'all_a',      5),
    ('F.000985, N=15',              'csi_all',   15),
]


def compute_amv_fields(amounts: np.ndarray, n_var1: int) -> dict:
    """V11 只需要 var1/ma60/macd; 其余 amv 字段省略。"""
    var1 = tdx_sma(amounts, n_var1, 1) / 1e7
    ma60 = ma(var1, 60)
    dif = ema(var1, 12) - ema(var1, 26)
    dea = ema(dif, 9)
    macd = (dif - dea) * 2
    return {'var1': var1, 'ma60': ma60, 'macd': macd}


def run_one(label, regime_dict, bull_reports, bear_reports):
    merged = merge_reports_by_regime(bull_reports, bear_reports, regime_dict)
    if not merged:
        return None
    res = run_single_backtest(merged, label, top_n=TOP_N, focus_days=FOCUS_DAYS)
    s = (res.get('summary') or {}).get(FOCUS_DAYS, {})
    out = {
        'annual_return': s.get('annual_return') or 0,
        'sharpe': s.get('sharpe_ratio') or 0,
        'max_drawdown': s.get('max_drawdown') or 0,
        'cumulative': s.get('cumulative') or 0,
        'win_rate': s.get('win_rate') or 0,
        'icir': s.get('icir') or 0,
        'top_excess': s.get('top_excess') or 0,
        'v52_pct': None,
        'v52_grade': '-',
    }
    try:
        ns = compute_ns_scores(res['summary'], focus_days=FOCUS_DAYS,
                               n_trading_days=len(res.get('daily_results', [])))
        out['v52_pct'] = ns.get('v52_pct') or ns.get('v5_pct')
        out['v52_grade'] = ns.get('v52_grade') or ns.get('v5_grade') or '-'
    except Exception as e:
        print(f'  V5.2 calc failed: {type(e).__name__}: {e}')
    return out


def main():
    print('=' * 90)
    print('  0AMV 基准探针 — V11 regime 切换回测对比')
    print('=' * 90)

    print(f'\n加载 bull: {BULL_DIR}')
    bull_reports = load_reports(BULL_DIR, rank_field='score')
    print(f'  {len(bull_reports)} 天')
    print(f'加载 bear: {BEAR_DIR}')
    bear_reports = load_reports(BEAR_DIR, rank_field='score')
    print(f'  {len(bear_reports)} 天')

    conn = sqlite3.connect(DB, timeout=30)
    amount_cache = {}
    for qkey, sql in QUERIES.items():
        df = pd.read_sql(sql, conn)
        df.columns = ['trade_date', 'amt']
        amount_cache[qkey] = df.sort_values('trade_date').reset_index(drop=True)
    conn.close()

    rows = []
    for label, qkey, n_var1 in CANDIDATES:
        print(f'\n--- 候选 {label} ---')
        df = amount_cache[qkey]
        amv = compute_amv_fields(df['amt'].values, n_var1=n_var1)
        regime = V11(amv)

        last_idx = len(df) - 1
        bull_n = int((regime == 1).sum())
        bear_n = int((regime == -1).sum())
        regime_dict = dict(zip(df['trade_date'], regime))
        print(f'  历史 regime: 牛市 {bull_n} 天 / 熊市 {bear_n} 天 / 总 {len(regime_dict)}')

        bt = run_one(label, regime_dict, bull_reports, bear_reports) or {}
        rows.append({
            'label': label,
            'date': df.at[last_idx, 'trade_date'],
            'var1': float(amv['var1'][last_idx]),
            'ma60': float(amv['ma60'][last_idx]),
            'macd': float(amv['macd'][last_idx]),
            'regime_today': '牛' if regime[last_idx] == 1 else '熊',
            'bull_days': bull_n,
            'bear_days': bear_n,
            **bt,
        })

    print('\n' + '=' * 90)
    print('  今日各候选 regime 状态')
    print('=' * 90)
    print(f"{'候选':<35} {'date':<12} {'var1':>8} {'ma60':>8} {'gap%':>7} {'macd':>7} regime")
    for r in rows:
        gap_pct = (r['var1'] - r['ma60']) / r['ma60'] * 100
        print(f"  {r['label']:<33} {r['date']} {r['var1']:>8.2f} {r['ma60']:>8.2f} "
              f"{gap_pct:>+6.2f}% {r['macd']:>+7.3f} {r['regime_today']}")

    print('\n' + '=' * 110)
    print('  回测对比 (Top-10, 10日持仓)  — 北极星 V5.2 + 关键风险/收益指标')
    print('=' * 110)
    print(f"{'候选':<35} {'牛天':>5} {'熊天':>5} {'年化':>7} {'Sharpe':>7} {'MaxDD':>7} {'累计%':>8} {'胜率':>6} {'ICIR':>6} {'V5.2':>10}")
    print('-' * 110)
    for r in rows:
        if 'annual_return' not in r:
            continue
        v52p = r.get('v52_pct')
        v52_str = f"{v52p:.1f}% {r['v52_grade']}" if v52p is not None else '-'
        print(f"  {r['label']:<33} {r['bull_days']:>5} {r['bear_days']:>5} "
              f"{r['annual_return']*100:>6.1f}% {r['sharpe']:>7.3f} "
              f"{r['max_drawdown']*100:>+6.1f}% {r['cumulative']:>+7.1f}% "
              f"{r['win_rate']:>5.1f}% {r['icir']:>6.3f} {v52_str:>10}")

    return rows


if __name__ == '__main__':
    main()
