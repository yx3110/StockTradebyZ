#!/usr/bin/env python3
"""
V3.95 89分以上股票回测分析 - 完整版
- 日期范围: 2025-09-01 到 2026-01-20
- 分析周期: 1d, 3d, 5d, 10d, 15d
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')
REPORT_DIR = 'reports/daily_selection_v3.95'
SCORE_THRESHOLD = 89
START_DATE = '2025-09-01'
END_DATE = '2026-01-20'
PERIODS = [1, 3, 5, 10, 15]

def collect_high_score_stocks():
    """收集指定日期范围内所有89分以上的股票"""
    report_path = Path(REPORT_DIR)
    all_stocks = []

    start_dt = datetime.strptime(START_DATE, '%Y-%m-%d')
    end_dt = datetime.strptime(END_DATE, '%Y-%m-%d')

    for f in sorted(report_path.glob('analysis_data_*.json')):
        date_str = f.stem.replace('analysis_data_', '')
        date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

        try:
            file_date = datetime.strptime(date_formatted, '%Y-%m-%d')
            if file_date < start_dt or file_date > end_dt:
                continue
        except:
            continue

        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)

            stocks = data.get('all_stocks_with_scores', data.get('stocks', []))
            for stock in stocks:
                score = stock.get('score', 0)
                if score >= SCORE_THRESHOLD:
                    strategies = stock.get('strategies', [])
                    if isinstance(strategies, list):
                        strategies_str = ', '.join(strategies)
                    else:
                        strategies_str = str(strategies)

                    code = stock.get('code', stock.get('stock_code', ''))
                    name = stock.get('name', stock.get('stock_name', ''))

                    all_stocks.append({
                        'select_date': date_formatted,
                        'code': code,
                        'name': name,
                        'score': score,
                        'strategies': strategies_str,
                    })
        except Exception as e:
            print(f"读取 {f} 失败: {e}")

    return all_stocks

def get_actual_returns(code, select_date, periods=PERIODS):
    """计算股票的实际收益 - 以选股日次日开盘价买入"""
    conn = sqlite3.connect(DB_PATH)

    query = '''
        SELECT dq.trade_date, dq.close, dq.open
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.code = ? AND dq.trade_date > ?
        ORDER BY dq.trade_date
        LIMIT 20
    '''

    df = pd.read_sql_query(query, conn, params=(code, select_date))
    conn.close()

    if len(df) < 1:
        return {f'return_{p}d': None for p in periods}

    buy_price = df.iloc[0]['open']

    returns = {}
    for period in periods:
        target_idx = period - 1
        if target_idx < len(df):
            sell_price = df.iloc[target_idx]['close']
            returns[f'return_{period}d'] = (sell_price - buy_price) / buy_price
        else:
            returns[f'return_{period}d'] = None

    return returns

def run_backtest():
    """运行回测分析"""
    print("="*70)
    print(f"V3.95 高分股票(>={SCORE_THRESHOLD}分) 回测分析 - 完整版")
    print(f"日期范围: {START_DATE} ~ {END_DATE}")
    print(f"分析周期: {PERIODS}")
    print("="*70)

    print(f"\n收集 {SCORE_THRESHOLD} 分以上股票...")
    high_score_stocks = collect_high_score_stocks()
    print(f"共找到 {len(high_score_stocks)} 只高分股票")

    if not high_score_stocks:
        print("没有找到高分股票，退出")
        return

    print("\n计算实际收益...")
    results = []
    total = len(high_score_stocks)

    for i, stock in enumerate(high_score_stocks):
        if (i + 1) % 50 == 0:
            print(f"进度: {i+1}/{total}")

        returns = get_actual_returns(stock['code'], stock['select_date'])

        result = {**stock}
        for p in PERIODS:
            result[f'actual_return_{p}d'] = returns.get(f'return_{p}d')
        results.append(result)

    df = pd.DataFrame(results)

    print(f"\n{'='*70}")
    print("回测结果分析")
    print(f"{'='*70}")

    print(f"\n总样本数: {len(df)}")
    print(f"评分范围: {df['score'].min():.1f} - {df['score'].max():.1f}")
    print(f"日期范围: {df['select_date'].min()} ~ {df['select_date'].max()}")
    print(f"涉及交易日: {df['select_date'].nunique()} 天")

    # 总体分析
    print(f"\n{'='*70}")
    print(f"总体分析 (所有{SCORE_THRESHOLD}分以上) - 各周期对比")
    print(f"{'='*70}")

    print(f"\n{'周期':<6} {'样本数':<8} {'平均收益':<12} {'中位数':<12} {'标准差':<12} {'胜率':<10} {'最大':<12} {'最小':<12}")
    print("-" * 90)

    overall_stats = []
    for period in PERIODS:
        col = f'actual_return_{period}d'
        valid = df[col].dropna()
        if len(valid) > 0:
            avg_ret = valid.mean() * 100
            median_ret = valid.median() * 100
            std_ret = valid.std() * 100
            win_rate = (valid > 0).mean() * 100
            max_ret = valid.max() * 100
            min_ret = valid.min() * 100

            print(f"{period}日{'':<4} {len(valid):<8} {avg_ret:+.2f}%{'':<6} {median_ret:+.2f}%{'':<6} {std_ret:.2f}%{'':<6} {win_rate:.1f}%{'':<5} {max_ret:+.2f}%{'':<5} {min_ret:+.2f}%")

            overall_stats.append({
                'period': f'{period}d',
                'count': len(valid),
                'avg': avg_ret,
                'median': median_ret,
                'std': std_ret,
                'win_rate': win_rate,
                'max': max_ret,
                'min': min_ret
            })

    # 按评分区间分析
    score_bins = [(89, 90), (90, 91)]

    print(f"\n{'='*70}")
    print("按评分区间分析")
    print(f"{'='*70}")

    for low, high in score_bins:
        mask = (df['score'] >= low) & (df['score'] < high)
        subset = df[mask]

        if len(subset) == 0:
            continue

        print(f"\n【评分 {low}-{high}】: {len(subset)} 只股票")
        print(f"{'周期':<6} {'样本':<6} {'平均':<10} {'中位数':<10} {'胜率':<8}")
        print("-" * 50)

        for period in PERIODS:
            col = f'actual_return_{period}d'
            valid = subset[col].dropna()
            if len(valid) > 0:
                avg_ret = valid.mean() * 100
                median_ret = valid.median() * 100
                win_rate = (valid > 0).mean() * 100
                print(f"{period}日{'':<4} {len(valid):<6} {avg_ret:+.2f}%{'':<4} {median_ret:+.2f}%{'':<4} {win_rate:.1f}%")

    # 按策略分析
    print(f"\n{'='*70}")
    print("按策略分析 (各周期)")
    print(f"{'='*70}")

    strategy_returns = {}
    for _, row in df.iterrows():
        strategies = row.get('strategies', '')
        if isinstance(strategies, str):
            strategy_list = [s.strip() for s in strategies.split(',')]
        else:
            strategy_list = []

        for strategy in strategy_list:
            if not strategy:
                continue
            if strategy not in strategy_returns:
                strategy_returns[strategy] = {f'{p}d': [] for p in PERIODS}

            for period in PERIODS:
                ret = row.get(f'actual_return_{period}d')
                if ret is not None and not pd.isna(ret):
                    strategy_returns[strategy][f'{period}d'].append(ret)

    # 按样本数排序
    sorted_strategies = sorted(strategy_returns.items(), key=lambda x: -len(x[1]['5d']))

    for strategy, returns in sorted_strategies:
        count = len(returns['5d'])
        if count < 3:
            continue

        print(f"\n【{strategy}】: {count} 只")
        print(f"{'周期':<6} {'平均':<10} {'胜率':<8}")
        print("-" * 30)

        for period in PERIODS:
            rets = np.array(returns[f'{period}d'])
            if len(rets) > 0:
                avg = rets.mean() * 100
                wr = (rets > 0).mean() * 100
                print(f"{period}日{'':<4} {avg:+.2f}%{'':<4} {wr:.1f}%")

    # 按月度分析
    print(f"\n{'='*70}")
    print("按月度分析 (各周期)")
    print(f"{'='*70}")

    df['month'] = pd.to_datetime(df['select_date']).dt.to_period('M')

    for month, group in df.groupby('month'):
        print(f"\n【{month}】: {len(group)} 只")
        print(f"{'周期':<6} {'平均':<10} {'胜率':<8}")
        print("-" * 30)

        for period in PERIODS:
            col = f'actual_return_{period}d'
            valid = group[col].dropna()
            if len(valid) > 0:
                avg = valid.mean() * 100
                wr = (valid > 0).mean() * 100
                print(f"{period}日{'':<4} {avg:+.2f}%{'':<4} {wr:.1f}%")

    # 保存结果
    output_dir = (PROJECT_ROOT / 'reports/backtest')
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    df.drop(columns=['month'], inplace=True, errors='ignore')
    detail_file = output_dir / f'v395_score89_full_backtest_{timestamp}.csv'
    df.to_csv(detail_file, index=False, encoding='utf-8-sig')
    print(f"\n详细数据已保存: {detail_file}")

    return df, overall_stats

if __name__ == '__main__':
    run_backtest()
