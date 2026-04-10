#!/usr/bin/env python3
"""
Gap × ML Score 交叉收益分析

分析: 不同ML评分 + 不同开盘gap组合下的实际收益
目标: 构建 (score_bucket, gap_bucket) → expected_return 查找表
      用于指导实盘的执行决策

用法:
    cd /Users/yangxu/StockTradebyZ
    python3 backtest/gap_signal_analysis.py
"""

import sys
import json
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')
REPORT_DIR = PROJECT_ROOT / 'reports' / 'daily_selection_ng101'


def load_all_data(start_date='2024-01-01', end_date='2026-12-31'):
    """加载报告评分 + OHLC价格，合并为分析数据集"""

    # 1. 加载所有报告
    print("加载报告...")
    records = []
    for f in sorted(REPORT_DIR.glob('analysis_data_*.json')):
        date_str = f.stem.replace('analysis_data_', '')
        date_cmp = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        if date_cmp < start_date or date_cmp > end_date:
            continue
        with open(f, 'r') as fh:
            data = json.load(fh)
        stocks = data.get('all_stocks_with_scores', [])
        for s in stocks:
            records.append({
                'report_date': date_cmp,
                'code': s['stock_code'],
                'score': s.get('score', 0),
                'pred_5d': s.get('pred_5d', 0),
                'pred_10d': s.get('pred_10d', 0),
            })

    df_reports = pd.DataFrame(records)
    print(f"  报告: {len(df_reports)} 条 ({df_reports['report_date'].nunique()} 天)")

    # 2. 加载价格
    print("加载价格...")
    conn = sqlite3.connect(DB_PATH)
    df_prices = pd.read_sql_query("""
        SELECT s.code, dq.trade_date, dq.open, dq.close, dq.high, dq.low,
               dq.price_change_pct
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE dq.trade_date >= ? AND dq.trade_date <= ?
          AND dq.open > 0 AND dq.close > 0
          AND s.type = 'A股'
    """, conn, params=[start_date, end_date])

    # 获取交易日列表
    trading_dates = sorted(df_prices['trade_date'].unique())
    date_to_idx = {d: i for i, d in enumerate(trading_dates)}

    # 3. 构建: 对于每个 (report_date, code)，找 T+1 的 open/close 和 T+N 的 close
    print("构建分析数据集...")
    # 先建价格查找表
    price_dict = {}
    for _, row in df_prices.iterrows():
        price_dict[(row['trade_date'], row['code'])] = {
            'open': row['open'], 'close': row['close'],
        }

    analysis_rows = []
    for _, row in df_reports.iterrows():
        report_date = row['report_date']
        code = row['code']
        score = row['score']

        # T日收盘价 (报告基于T日数据)
        t_price = price_dict.get((report_date, code))
        if not t_price:
            continue
        t_close = t_price['close']

        # T+1 开盘/收盘
        idx = date_to_idx.get(report_date, -1)
        if idx < 0 or idx + 1 >= len(trading_dates):
            continue
        t1_date = trading_dates[idx + 1]
        t1_price = price_dict.get((t1_date, code))
        if not t1_price:
            continue
        t1_open = t1_price['open']
        t1_close = t1_price['close']

        # 开盘gap
        gap = (t1_open - t_close) / t_close

        # T+1 日内收益 (open→close)
        intraday_ret = (t1_close - t1_open) / t1_open

        # T+1→T+5 收益 (open→5日后close)
        ret_5d = None
        if idx + 5 < len(trading_dates):
            t5_date = trading_dates[idx + 5]
            t5_price = price_dict.get((t5_date, code))
            if t5_price:
                ret_5d = (t5_price['close'] - t1_open) / t1_open

        # T+1→T+10 收益
        ret_10d = None
        if idx + 10 < len(trading_dates):
            t10_date = trading_dates[idx + 10]
            t10_price = price_dict.get((t10_date, code))
            if t10_price:
                ret_10d = (t10_price['close'] - t1_open) / t1_open

        analysis_rows.append({
            'report_date': report_date,
            'code': code,
            'score': score,
            'gap': gap,
            'intraday_ret': intraday_ret,
            'ret_5d': ret_5d,
            'ret_10d': ret_10d,
        })

    conn.close()
    df = pd.DataFrame(analysis_rows)
    print(f"  分析数据: {len(df)} 条")
    return df


def analyze_gap_score_matrix(df: pd.DataFrame):
    """分析 Score × Gap 的收益矩阵"""

    # Score 分桶
    score_bins = [0, 30, 50, 70, 85, 100]
    score_labels = ['0-30', '30-50', '50-70', '70-85', '85-100']
    df['score_bucket'] = pd.cut(df['score'], bins=score_bins, labels=score_labels, right=True)

    # Gap 分桶
    gap_bins = [-1, -0.03, -0.01, 0.01, 0.03, 0.05, 1]
    gap_labels = ['<-3%', '-3~-1%', '-1~1%', '1~3%', '3~5%', '>5%']
    df['gap_bucket'] = pd.cut(df['gap'], bins=gap_bins, labels=gap_labels)

    print(f"\n{'='*80}")
    print(f"  Score × Gap 交叉收益矩阵 (10日收益, 买入开盘价)")
    print(f"{'='*80}")

    # 10日收益矩阵
    pivot_ret = df.groupby(['score_bucket', 'gap_bucket'])['ret_10d'].agg(['mean', 'count'])
    pivot_mean = pivot_ret['mean'].unstack(fill_value=0) * 100
    pivot_count = pivot_ret['count'].unstack(fill_value=0)

    print(f"\n  平均10日收益 (%):")
    print(f"  {'Score':<10}", end='')
    for g in gap_labels:
        print(f"  {g:>8}", end='')
    print()
    print(f"  {'─'*70}")
    for s in score_labels:
        print(f"  {s:<10}", end='')
        for g in gap_labels:
            val = pivot_mean.loc[s, g] if s in pivot_mean.index and g in pivot_mean.columns else 0
            cnt = pivot_count.loc[s, g] if s in pivot_count.index and g in pivot_count.columns else 0
            if cnt >= 10:
                marker = '++' if val > 1.5 else '+' if val > 0.5 else '--' if val < -1.5 else '-' if val < -0.5 else ''
                print(f"  {val:>6.1f}{marker:>2}", end='')
            else:
                print(f"  {'n/a':>8}", end='')
        print()

    print(f"\n  样本数:")
    print(f"  {'Score':<10}", end='')
    for g in gap_labels:
        print(f"  {g:>8}", end='')
    print()
    print(f"  {'─'*70}")
    for s in score_labels:
        print(f"  {s:<10}", end='')
        for g in gap_labels:
            cnt = pivot_count.loc[s, g] if s in pivot_count.index and g in pivot_count.columns else 0
            print(f"  {int(cnt):>8}", end='')
        print()

    # 同样看5日收益
    print(f"\n  平均5日收益 (%):")
    pivot_5d = df.groupby(['score_bucket', 'gap_bucket'])['ret_5d'].mean().unstack(fill_value=0) * 100
    print(f"  {'Score':<10}", end='')
    for g in gap_labels:
        print(f"  {g:>8}", end='')
    print()
    print(f"  {'─'*70}")
    for s in score_labels:
        print(f"  {s:<10}", end='')
        for g in gap_labels:
            val = pivot_5d.loc[s, g] if s in pivot_5d.index and g in pivot_5d.columns else 0
            print(f"  {val:>7.1f}%", end='')
        print()

    # Top-10 (高分组) 的 gap 分析
    print(f"\n{'='*80}")
    print(f"  Top 评分组 (85-100) Gap 分析")
    print(f"{'='*80}")

    top = df[df['score'] >= 85].copy()
    if len(top) > 0:
        gap_analysis = top.groupby('gap_bucket').agg(
            count=('ret_10d', 'count'),
            avg_ret_10d=('ret_10d', 'mean'),
            avg_ret_5d=('ret_5d', 'mean'),
            win_rate_10d=('ret_10d', lambda x: (x > 0).mean()),
            avg_gap=('gap', 'mean'),
        ).round(4)

        print(f"\n  {'Gap区间':<12} {'样本':>6} {'10日收益':>10} {'5日收益':>10} {'胜率':>8} {'均Gap':>8}")
        print(f"  {'─'*60}")
        for idx, row in gap_analysis.iterrows():
            print(f"  {str(idx):<12} {int(row['count']):>6} "
                  f"{row['avg_ret_10d']*100:>9.2f}% {row['avg_ret_5d']*100:>9.2f}% "
                  f"{row['win_rate_10d']*100:>7.1f}% {row['avg_gap']*100:>7.2f}%")

    # 低分组（要卖出的）gap 分析
    print(f"\n{'='*80}")
    print(f"  低评分组 (0-30) Gap 分析 — 卖出时机参考")
    print(f"{'='*80}")

    low = df[df['score'] <= 30].copy()
    if len(low) > 0:
        # 低开时卖出 vs 等反弹
        gap_sell = low.groupby('gap_bucket').agg(
            count=('ret_10d', 'count'),
            avg_ret_10d=('ret_10d', 'mean'),
            intraday=('intraday_ret', 'mean'),
        ).round(4)

        print(f"\n  {'Gap区间':<12} {'样本':>6} {'10日收益':>10} {'日内收益':>10} {'结论':>12}")
        print(f"  {'─'*60}")
        for idx, row in gap_sell.iterrows():
            # 日内收益>0说明低开反弹，应该等
            conclusion = "等反弹" if row['intraday'] > 0.005 else "立即卖"
            print(f"  {str(idx):<12} {int(row['count']):>6} "
                  f"{row['avg_ret_10d']*100:>9.2f}% {row['intraday']*100:>9.2f}% "
                  f"{conclusion:>12}")

    # 生成执行决策表
    print(f"\n{'='*80}")
    print(f"  执行决策建议")
    print(f"{'='*80}\n")

    # 对每个 (score_bucket, gap_bucket) 给出建议
    for s in ['85-100', '70-85', '50-70']:
        subset = df[(df['score_bucket'] == s)]
        for g in gap_labels:
            g_subset = subset[subset['gap_bucket'] == g]
            if len(g_subset) < 10:
                continue
            avg_ret = g_subset['ret_10d'].mean()
            win_rate = (g_subset['ret_10d'] > 0).mean()
            if avg_ret > 0.015 and win_rate > 0.55:
                action = "买入"
            elif avg_ret > 0.005:
                action = "轻仓买"
            elif avg_ret < -0.01:
                action = "不买"
            else:
                action = "观望"
            print(f"  Score {s:>8} + Gap {g:>8} → {action:>6} "
                  f"(10d收益={avg_ret*100:.1f}%, 胜率={win_rate*100:.0f}%, n={len(g_subset)})")

    return df


def main():
    df = load_all_data('2024-01-01', '2026-12-31')
    analyze_gap_score_matrix(df)


if __name__ == '__main__':
    main()
