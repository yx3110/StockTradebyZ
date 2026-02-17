#!/usr/bin/env python3
"""
V3.95 89分以上股票回测分析
- 日期范围: 2025-09-01 到 2026-01-20
- 收集所有89分以上股票
- 计算5日/10日/15日实际收益
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

DB_PATH = 'data_adapter/stock_data.db'
REPORT_DIR = 'reports/daily_selection_v3.95'
SCORE_THRESHOLD = 89
START_DATE = '2025-09-01'
END_DATE = '2026-01-20'

def collect_high_score_stocks():
    """收集指定日期范围内所有89分以上的股票"""
    report_path = Path(REPORT_DIR)
    all_stocks = []

    start_dt = datetime.strptime(START_DATE, '%Y-%m-%d')
    end_dt = datetime.strptime(END_DATE, '%Y-%m-%d')

    for f in sorted(report_path.glob('analysis_data_*.json')):
        date_str = f.stem.replace('analysis_data_', '')
        date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

        # 日期范围过滤
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
                        'predicted_return_5d': stock.get('predicted_return_5d', stock.get('pred_5d', None)),
                        'recommendation': stock.get('recommendation', ''),
                        'confidence': stock.get('confidence_score', stock.get('confidence', None))
                    })
        except Exception as e:
            print(f"读取 {f} 失败: {e}")

    return all_stocks

def get_actual_returns(code, select_date, periods=[5, 10, 15]):
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
    print("="*60)
    print(f"V3.95 高分股票(>={SCORE_THRESHOLD}分) 回测分析")
    print(f"日期范围: {START_DATE} ~ {END_DATE}")
    print("="*60)

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

        result = {
            **stock,
            'actual_return_5d': returns.get('return_5d'),
            'actual_return_10d': returns.get('return_10d'),
            'actual_return_15d': returns.get('return_15d'),
        }
        results.append(result)

    df = pd.DataFrame(results)

    print(f"\n{'='*60}")
    print("回测结果分析")
    print(f"{'='*60}")

    print(f"\n总样本数: {len(df)}")
    print(f"评分范围: {df['score'].min():.1f} - {df['score'].max():.1f}")
    print(f"日期范围: {df['select_date'].min()} ~ {df['select_date'].max()}")
    print(f"涉及交易日: {df['select_date'].nunique()} 天")

    # 按评分区间分析
    score_bins = [(89, 90), (90, 91), (91, 92), (92, 93), (93, 94), (94, 95), (95, 100)]

    print(f"\n{'='*60}")
    print("按评分区间分析实际收益")
    print(f"{'='*60}")

    analysis_results = []

    for low, high in score_bins:
        mask = (df['score'] >= low) & (df['score'] < high)
        subset = df[mask]

        if len(subset) == 0:
            continue

        row = {
            'score_range': f'{low}-{high}',
            'count': len(subset),
        }

        print(f"\n评分 {low}-{high}: {len(subset)} 只股票")
        for period in [5, 10, 15]:
            col = f'actual_return_{period}d'
            valid = subset[col].dropna()
            if len(valid) > 0:
                avg_ret = valid.mean() * 100
                median_ret = valid.median() * 100
                win_rate = (valid > 0).mean() * 100
                row[f'avg_{period}d'] = avg_ret
                row[f'median_{period}d'] = median_ret
                row[f'win_rate_{period}d'] = win_rate
                row[f'count_{period}d'] = len(valid)
                print(f"  {period}日: 平均{avg_ret:+.2f}%, 中位数{median_ret:+.2f}%, 胜率{win_rate:.1f}% (n={len(valid)})")
            else:
                row[f'avg_{period}d'] = None
                row[f'median_{period}d'] = None
                row[f'win_rate_{period}d'] = None
                row[f'count_{period}d'] = 0

        analysis_results.append(row)

    # 总体分析
    print(f"\n{'='*60}")
    print(f"总体分析 (所有{SCORE_THRESHOLD}分以上)")
    print(f"{'='*60}")

    overall = {'score_range': f'ALL(>={SCORE_THRESHOLD})', 'count': len(df)}
    for period in [5, 10, 15]:
        col = f'actual_return_{period}d'
        valid = df[col].dropna()
        if len(valid) > 0:
            avg_ret = valid.mean() * 100
            median_ret = valid.median() * 100
            win_rate = (valid > 0).mean() * 100
            std_ret = valid.std() * 100
            overall[f'avg_{period}d'] = avg_ret
            overall[f'median_{period}d'] = median_ret
            overall[f'win_rate_{period}d'] = win_rate
            overall[f'count_{period}d'] = len(valid)
            print(f"{period}日收益: 平均{avg_ret:+.2f}%, 中位数{median_ret:+.2f}%, 标准差{std_ret:.2f}%, 胜率{win_rate:.1f}% (n={len(valid)})")

    analysis_results.append(overall)

    # 预测vs实际对比
    print(f"\n{'='*60}")
    print("预测收益 vs 实际收益对比 (5日)")
    print(f"{'='*60}")

    pred_col = 'predicted_return_5d'
    actual_col = 'actual_return_5d'

    valid_mask = df[pred_col].notna() & df[actual_col].notna()
    valid_df = df[valid_mask].copy()

    if len(valid_df) > 0:
        pred_values = valid_df[pred_col].values.copy()
        if np.nanmean(np.abs(pred_values)) < 1:
            pred_values = pred_values * 100

        actual_values = valid_df[actual_col].values * 100

        pred_mean = np.nanmean(pred_values)
        actual_mean = np.nanmean(actual_values)

        print(f"样本数: {len(valid_df)}")
        print(f"预测5日平均收益: {pred_mean:+.2f}%")
        print(f"实际5日平均收益: {actual_mean:+.2f}%")
        print(f"预测偏差: {pred_mean - actual_mean:+.2f}%")

        corr = np.corrcoef(pred_values, actual_values)[0, 1]
        print(f"预测与实际相关系数: {corr:.3f}")

    # 按月度分析
    print(f"\n{'='*60}")
    print("按月度分析")
    print(f"{'='*60}")

    df['month'] = pd.to_datetime(df['select_date']).dt.to_period('M')
    monthly_stats = []

    for month, group in df.groupby('month'):
        valid_5d = group['actual_return_5d'].dropna()
        if len(valid_5d) > 0:
            avg_5d = valid_5d.mean() * 100
            win_rate_5d = (valid_5d > 0).mean() * 100
            print(f"{month}: {len(group)}只, 5日平均{avg_5d:+.2f}%, 胜率{win_rate_5d:.1f}%")
            monthly_stats.append({
                'month': str(month),
                'count': len(group),
                'avg_5d': avg_5d,
                'win_rate_5d': win_rate_5d
            })

    # 保存结果
    output_dir = Path('reports/backtest')
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    detail_file = output_dir / f'v395_score89_backtest_detail_{timestamp}.csv'
    df.drop(columns=['month'], inplace=True, errors='ignore')
    df.to_csv(detail_file, index=False, encoding='utf-8-sig')
    print(f"\n详细数据已保存: {detail_file}")

    analysis_file = output_dir / f'v395_score89_backtest_analysis_{timestamp}.json'
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump({
            'score_analysis': analysis_results,
            'monthly_stats': monthly_stats
        }, f, ensure_ascii=False, indent=2)
    print(f"分析结果已保存: {analysis_file}")

    report_file = output_dir / f'v395_score89_backtest_report_{timestamp}.md'
    generate_markdown_report(df, analysis_results, monthly_stats, report_file)
    print(f"报告已保存: {report_file}")

    return df, analysis_results

def generate_markdown_report(df, analysis_results, monthly_stats, output_file):
    """生成Markdown报告"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# V3.95 高分股票(>={SCORE_THRESHOLD}分)回测报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**数据范围**: {START_DATE} ~ {END_DATE}\n\n")
        f.write(f"**评分阈值**: >= {SCORE_THRESHOLD}\n\n")

        f.write("## 总体统计\n\n")
        f.write(f"- 总样本数: {len(df)}\n")
        f.write(f"- 评分范围: {df['score'].min():.1f} - {df['score'].max():.1f}\n")
        f.write(f"- 涉及交易日: {df['select_date'].nunique()} 天\n\n")

        f.write("## 按评分区间分析\n\n")
        f.write("| 评分区间 | 样本数 | 5日平均 | 5日胜率 | 10日平均 | 10日胜率 | 15日平均 | 15日胜率 |\n")
        f.write("|---------|--------|---------|---------|----------|----------|----------|----------|\n")

        for row in analysis_results:
            f.write(f"| {row['score_range']} | {row['count']} |")
            for period in [5, 10, 15]:
                avg = row.get(f'avg_{period}d')
                wr = row.get(f'win_rate_{period}d')
                if avg is not None:
                    f.write(f" {avg:+.2f}% | {wr:.1f}% |")
                else:
                    f.write(" N/A | N/A |")
            f.write("\n")

        f.write("\n## 月度分析\n\n")
        f.write("| 月份 | 样本数 | 5日平均收益 | 5日胜率 |\n")
        f.write("|------|--------|-------------|--------|\n")
        for ms in monthly_stats:
            f.write(f"| {ms['month']} | {ms['count']} | {ms['avg_5d']:+.2f}% | {ms['win_rate_5d']:.1f}% |\n")

        f.write("\n## 收益分布\n\n")
        for period in [5, 10, 15]:
            col = f'actual_return_{period}d'
            valid = df[col].dropna() * 100
            if len(valid) > 0:
                f.write(f"### {period}日收益分布\n\n")
                f.write(f"- 样本数: {len(valid)}\n")
                f.write(f"- 平均值: {valid.mean():+.2f}%\n")
                f.write(f"- 中位数: {valid.median():+.2f}%\n")
                f.write(f"- 标准差: {valid.std():.2f}%\n")
                f.write(f"- 最大值: {valid.max():+.2f}%\n")
                f.write(f"- 最小值: {valid.min():+.2f}%\n")
                f.write(f"- 胜率: {(valid > 0).mean() * 100:.1f}%\n\n")

        # 按策略分析
        f.write("## 按策略分析\n\n")

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
                    strategy_returns[strategy] = {'5d': [], '10d': [], '15d': []}

                for period in [5, 10, 15]:
                    ret = row.get(f'actual_return_{period}d')
                    if ret is not None and not pd.isna(ret):
                        strategy_returns[strategy][f'{period}d'].append(ret)

        f.write("| 策略 | 样本数 | 5日平均 | 5日胜率 | 10日平均 | 10日胜率 |\n")
        f.write("|------|--------|---------|---------|----------|----------|\n")

        for strategy, returns in sorted(strategy_returns.items(), key=lambda x: -len(x[1]['5d'])):
            count = len(returns['5d'])
            if count == 0:
                continue

            f.write(f"| {strategy} | {count} |")
            for period in ['5d', '10d']:
                rets = np.array(returns[period])
                if len(rets) > 0:
                    avg = rets.mean() * 100
                    wr = (rets > 0).mean() * 100
                    f.write(f" {avg:+.2f}% | {wr:.1f}% |")
                else:
                    f.write(" N/A | N/A |")
            f.write("\n")

        f.write("\n## 高分股票明细 (Top 50 by Score)\n\n")
        f.write("| 日期 | 代码 | 名称 | 评分 | 策略 | 5日收益 | 10日收益 | 15日收益 |\n")
        f.write("|------|------|------|------|------|---------|----------|----------|\n")

        top_stocks = df.nlargest(50, 'score')
        for _, row in top_stocks.iterrows():
            strategies = row.get('strategies', '')
            if len(str(strategies)) > 20:
                strategies = str(strategies)[:20] + '...'

            r5 = row.get('actual_return_5d')
            r10 = row.get('actual_return_10d')
            r15 = row.get('actual_return_15d')

            r5_str = f"{r5*100:+.2f}%" if r5 is not None and not pd.isna(r5) else "N/A"
            r10_str = f"{r10*100:+.2f}%" if r10 is not None and not pd.isna(r10) else "N/A"
            r15_str = f"{r15*100:+.2f}%" if r15 is not None and not pd.isna(r15) else "N/A"

            f.write(f"| {row['select_date']} | {row['code']} | {row['name']} | {row['score']:.1f} | {strategies} | {r5_str} | {r10_str} | {r15_str} |\n")

if __name__ == '__main__':
    run_backtest()
