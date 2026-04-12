#!/usr/bin/env python3
"""
8策略长期回测后处理: 按年度/AMV regime切分 + 对比市场基线.

输入: backtest_strategy_metrics.py生成的strategy_signals_YYYYMMDD.csv
输出: reports/backtest/strategy_longhorizon_breakdown_YYYYMMDD.md
"""
import sqlite3
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_NAMES = ['少负战法','SuperB1战法','补票战法','TePu战法','填坑战法',
                  '知行战法','上穿60放量战法','暴力K战法']
REGIME_LABELS = {1: '牛', -1: '熊'}
YEAR_REGIME_HP = 10  # holding period used for year/regime breakdown tables


def load_market_baseline(conn, start_date, end_date, holding_periods):
    """A股 universe equal-weight forward returns per trade_date.

    End buffer uses max_hp*2+10 calendar days so `groupby.shift(-(hp+1))` can
    reach max_hp trading days forward even across weekends/holidays.
    """
    max_hp = max(holding_periods)
    end_buffered = (pd.Timestamp(end_date) + pd.Timedelta(days=max_hp * 2 + 10)).strftime('%Y-%m-%d')
    q = """
    SELECT dq.security_id as sid, dq.trade_date as date, dq.open, dq.close
    FROM daily_quotes dq
    JOIN securities s ON s.id = dq.security_id
    WHERE s.type = 'A股' AND dq.trade_date >= ? AND dq.trade_date <= ?
      AND dq.open > 0 AND dq.close > 0
    """
    df = pd.read_sql_query(q, conn, params=(start_date, end_buffered))
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['sid','date']).reset_index(drop=True)
    g = df.groupby('sid', sort=False)
    df['open_tp1'] = g['open'].shift(-1)
    for hp in holding_periods:
        close_tp = g['close'].shift(-(hp + 1))
        df[f'mkt_ret_{hp}d'] = (close_tp - df['open_tp1']) / df['open_tp1']

    agg_cols = {f'mkt_ret_{hp}d': 'mean' for hp in holding_periods}
    return df.groupby('date').agg(agg_cols).reset_index()


def load_regime(conn, start_date, end_date):
    q = """SELECT trade_date as date, amv_regime as regime FROM market_amv
           WHERE trade_date >= ? AND trade_date <= ?"""
    df = pd.read_sql_query(q, conn, params=(start_date, end_date))
    df['date'] = pd.to_datetime(df['date'])
    return df


def sharpe(returns, hp):
    """Per-signal Sharpe. Assumes independent periods — overlapping signals will inflate."""
    if len(returns) < 2:
        return 0.0
    std = np.std(returns, ddof=1)
    if std == 0:
        return 0.0
    return np.mean(returns) / std * np.sqrt(252 / hp)


def max_dd(returns):
    """Signal-level cumulative drawdown — assumes signals are sequential (they overlap in reality).
    Use for relative ranking across strategies in the same backtest, NOT as portfolio MaxDD.
    """
    if len(returns) == 0:
        return 0.0
    cum = np.cumprod(1 + returns)
    run_max = np.maximum.accumulate(cum)
    dd = (run_max - cum) / run_max
    return np.max(dd)


def aggregate_group(signals_with_ret, hp, group_col):
    ret_col = f'ret_{hp}d'
    mkt_col = f'mkt_ret_{hp}d'
    rows = []
    for g, sub in signals_with_ret.groupby(group_col):
        sub = sub.dropna(subset=[ret_col])
        if sub.empty:
            continue
        rets = sub[ret_col].values
        mkts = sub[mkt_col].dropna().values if mkt_col in sub else np.array([])
        baseline = np.mean(mkts) if len(mkts) else np.nan
        avg_ret = np.mean(rets)
        rows.append({
            group_col: g,
            'n_signals': len(rets),
            'avg_ret': avg_ret,
            'mkt_baseline': baseline,
            'alpha': avg_ret - baseline if not np.isnan(baseline) else np.nan,
            'hit_rate': np.mean(rets > 0),
            'sharpe': sharpe(rets, hp),
            'max_dd': max_dd(rets),
        })
    return pd.DataFrame(rows)


def fmt_pct(x, digits=2):
    if pd.isna(x):
        return 'N/A'
    return f'{x*100:+.{digits}f}%'


def render_strategy_table(df, lines, include_maxdd):
    """Render one row per strategy with ret/baseline/alpha/hit/sharpe (+maxdd optional)."""
    header = '| 策略 | 信号数 | 平均收益 | 市场基线 | **Alpha** | 胜率 | Sharpe'
    sep = '|:---|---:|---:|---:|---:|---:|---:'
    if include_maxdd:
        header += ' | MaxDD |'
        sep += '|---:|'
    else:
        header += ' |'
        sep += '|'
    lines.append(header)
    lines.append(sep)
    for _, r in df.iterrows():
        row = (f'| {r["strategy"]} | {int(r["n_signals"]):,} | {fmt_pct(r["avg_ret"])} '
               f'| {fmt_pct(r["mkt_baseline"])} | **{fmt_pct(r["alpha"])}** | '
               f'{r["hit_rate"]*100:.1f}% | {r["sharpe"]:.3f}')
        if include_maxdd:
            row += f' | {r["max_dd"]*100:.1f}%'
        row += ' |'
        lines.append(row)


def build_year_breakdown(sig, hp, years):
    """Vectorized year×strategy alpha/hit-rate table via single groupby."""
    ret_col, mkt_col = f'ret_{hp}d', f'mkt_ret_{hp}d'
    sub = sig.dropna(subset=[ret_col]).copy()
    agg = sub.groupby(['strategy', 'year']).agg(
        n=(ret_col, 'size'),
        ret_mean=(ret_col, 'mean'),
        mkt_mean=(mkt_col, 'mean'),
        hit=(ret_col, lambda s: (s > 0).mean()),
    )
    agg['alpha'] = agg['ret_mean'] - agg['mkt_mean']
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='strategy_signals CSV from backtest_strategy_metrics.py')
    ap.add_argument('--holding-periods', nargs='+', type=int, default=[5, 10, 15])
    ap.add_argument('--output-dir', default='reports/backtest')
    args = ap.parse_args()

    sig = pd.read_csv(args.csv)
    sig['date'] = pd.to_datetime(sig['date'])
    print(f'Loaded {len(sig)} signals, {sig["date"].min()} ~ {sig["date"].max()}')
    print(f'Strategies: {sig["strategy"].value_counts().to_dict()}')

    db = PROJECT_ROOT / 'data_adapter/stock_data.db'
    conn = sqlite3.connect(str(db))
    start = sig['date'].min().strftime('%Y-%m-%d')
    end = sig['date'].max().strftime('%Y-%m-%d')
    print(f'Loading market baseline {start} ~ {end}...')
    baseline = load_market_baseline(conn, start, end, args.holding_periods)
    print(f'Baseline {len(baseline)} days')
    regime = load_regime(conn, start, end)
    regime['regime_label'] = regime['regime'].map(REGIME_LABELS)
    print(f'Regime {len(regime)} days, bull={sum(regime["regime"]==1)}, bear={sum(regime["regime"]==-1)}')
    conn.close()

    sig = sig.merge(baseline, on='date', how='left')
    sig = sig.merge(regime[['date','regime_label']], on='date', how='left')
    sig['year'] = sig['date'].dt.year
    sig['regime_label'] = sig['regime_label'].fillna('?')

    lines = []
    lines.append(f'# 8策略长期回测分析 (2018-2026)')
    lines.append(f'')
    lines.append(f'- 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'- 信号文件: `{args.csv}`')
    lines.append(f'- 数据区间: {start} ~ {end}')
    lines.append(f'- 信号总数: {len(sig):,}')
    lines.append(f'- 收益定义: 买入价=T+1 open, 卖出价=T+1+N close')
    lines.append(f'')

    for hp in args.holding_periods:
        lines.append(f'## {hp}日持仓 — 整体表现')
        lines.append('')
        df = aggregate_group(sig, hp, 'strategy')
        df = df.set_index('strategy').reindex(STRATEGY_NAMES).reset_index()
        df = df.sort_values('sharpe', ascending=False)
        render_strategy_table(df, lines, include_maxdd=True)
        lines.append('')

    hp = YEAR_REGIME_HP
    years = sorted(sig['year'].unique())
    lines.append(f'## {hp}日持仓 — 按年度')
    lines.append('')
    lines.append(f'每单元格: Alpha(对比市场) / 胜率 / 信号数')
    lines.append('')
    lines.append('| 策略 | ' + ' | '.join(str(y) for y in years) + ' |')
    lines.append('|:---|' + '---:|' * len(years))
    year_agg = build_year_breakdown(sig, hp, years)
    for strat in STRATEGY_NAMES:
        row = f'| {strat} |'
        for y in years:
            if (strat, y) not in year_agg.index:
                row += ' - |'
                continue
            r = year_agg.loc[(strat, y)]
            if r['n'] < 5:
                row += ' - |'
                continue
            row += f' {fmt_pct(r["alpha"],1)}/{r["hit"]*100:.0f}%/{int(r["n"])} |'
        lines.append(row)
    lines.append('')

    lines.append(f'## {hp}日持仓 — 按市场regime (AMV牛熊)')
    lines.append('')
    for regime_lbl in ['牛', '熊']:
        sub_reg = sig[sig['regime_label'] == regime_lbl]
        if sub_reg.empty:
            continue
        lines.append(f'### {regime_lbl}市')
        lines.append('')
        df = aggregate_group(sub_reg, hp, 'strategy')
        df = df.set_index('strategy').reindex(STRATEGY_NAMES).reset_index().dropna(subset=['n_signals'])
        df = df.sort_values('alpha', ascending=False)
        render_strategy_table(df, lines, include_maxdd=False)
        lines.append('')

    lines.append('## 对照: 市场自身平均表现')
    lines.append('')
    lines.append('| 期间 | 交易日 | 10d平均 | 15d平均 |')
    lines.append('|:---|---:|---:|---:|')
    for y in years:
        sub = baseline[baseline['date'].dt.year == y]
        if sub.empty:
            continue
        r10 = sub['mkt_ret_10d'].mean()
        r15 = sub['mkt_ret_15d'].mean() if 'mkt_ret_15d' in sub else np.nan
        lines.append(f'| {y} | {len(sub)} | {fmt_pct(r10)} | {fmt_pct(r15)} |')
    lines.append('')
    for lbl in ['牛', '熊']:
        dates_in_regime = regime[regime['regime_label'] == lbl]['date']
        sub = baseline[baseline['date'].isin(dates_in_regime)]
        if sub.empty:
            continue
        r10 = sub['mkt_ret_10d'].mean()
        r15 = sub['mkt_ret_15d'].mean() if 'mkt_ret_15d' in sub else np.nan
        lines.append(f'| {lbl}市 | {len(sub)} | {fmt_pct(r10)} | {fmt_pct(r15)} |')
    lines.append('')

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f'strategy_longhorizon_breakdown_{datetime.now().strftime("%Y%m%d")}.md'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\nReport: {out_path}')


if __name__ == '__main__':
    main()
