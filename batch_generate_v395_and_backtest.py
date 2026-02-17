#!/usr/bin/env python3
"""
批量生成V3.95报告并回测87分以上股票的5日/10日/15日收益
"""

import os
import sys
import json
import sqlite3
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# 配置
DB_PATH = 'data_adapter/stock_data.db'
REPORT_DIR = 'reports/daily_selection_v3.95'
START_DATE = '2025-09-01'
END_DATE = '2025-12-31'
SCORE_THRESHOLD = 90

def get_trading_days(start_date, end_date):
    """获取交易日列表"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT DISTINCT trade_date FROM daily_quotes
        WHERE trade_date >= ? AND trade_date <= ?
        ORDER BY trade_date
    ''', (start_date, end_date))
    dates = [row[0] for row in cursor.fetchall()]
    conn.close()
    return dates

def get_existing_reports():
    """获取已有报告的日期"""
    report_path = Path(REPORT_DIR)
    existing = set()
    if report_path.exists():
        for f in report_path.glob('analysis_data_*.json'):
            date_str = f.stem.replace('analysis_data_', '')
            # 转换为 YYYY-MM-DD 格式
            if len(date_str) == 8:
                existing.add(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")
    return existing

def generate_report(date_str):
    """生成单个日期的V3.95报告"""
    date_formatted = date_str.replace('-', '')
    print(f"\n{'='*60}")
    print(f"生成 {date_str} 的V3.95报告...")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            ['python3', 'tomorrow_stock_selector.py', date_str, '--scoring-version', 'v3.95'],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            print(f"✅ {date_str} 报告生成成功")
            return True
        else:
            print(f"❌ {date_str} 报告生成失败: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"❌ {date_str} 报告生成超时")
        return False
    except Exception as e:
        print(f"❌ {date_str} 报告生成异常: {e}")
        return False

def collect_high_score_stocks():
    """收集所有87分以上的股票"""
    report_path = Path(REPORT_DIR)
    all_stocks = []

    for f in sorted(report_path.glob('analysis_data_*.json')):
        date_str = f.stem.replace('analysis_data_', '')
        date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)

            stocks = data.get('all_stocks_with_scores', data.get('stocks', []))
            for stock in stocks:
                score = stock.get('score', 0)
                if score >= SCORE_THRESHOLD:
                    all_stocks.append({
                        'date': date_formatted,
                        'code': stock.get('stock_code', stock.get('code', '')),
                        'name': stock.get('stock_name', stock.get('name', '')),
                        'score': score,
                        'strategies': stock.get('strategies', []),
                        'predicted_return_5d': stock.get('predicted_return_5d', stock.get('pred_5d', None)),
                        'recommendation': stock.get('recommendation', ''),
                        'confidence': stock.get('confidence_score', stock.get('confidence', None))
                    })
        except Exception as e:
            print(f"读取 {f} 失败: {e}")

    return all_stocks

def get_actual_returns(code, buy_date, periods=[5, 10, 15]):
    """计算股票的实际收益"""
    conn = sqlite3.connect(DB_PATH)

    # 获取买入日期后的价格数据
    query = '''
        SELECT dq.trade_date, dq.close, dq.open
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.code = ? AND dq.trade_date >= ?
        ORDER BY dq.trade_date
        LIMIT 20
    '''

    df = pd.read_sql_query(query, conn, params=(code, buy_date))
    conn.close()

    if len(df) < 2:
        return {f'return_{p}d': None for p in periods}

    # 假设以第二天开盘价买入（因为选股是前一天收盘后做出的）
    # 但如果只有当天数据，就用当天收盘价作为基准
    if len(df) >= 2:
        buy_price = df.iloc[1]['open']  # 第二天开盘买入
        start_idx = 1
    else:
        buy_price = df.iloc[0]['close']
        start_idx = 0

    returns = {}
    for period in periods:
        target_idx = start_idx + period
        if target_idx < len(df):
            sell_price = df.iloc[target_idx]['close']
            returns[f'return_{period}d'] = (sell_price - buy_price) / buy_price
        else:
            returns[f'return_{period}d'] = None

    return returns

def run_backtest(stocks):
    """运行回测分析"""
    print(f"\n{'='*60}")
    print(f"开始回测 {len(stocks)} 只87分以上股票")
    print(f"{'='*60}")

    results = []
    total = len(stocks)

    for i, stock in enumerate(stocks):
        if (i + 1) % 50 == 0:
            print(f"进度: {i+1}/{total}")

        returns = get_actual_returns(stock['code'], stock['date'])

        result = {
            **stock,
            'actual_return_5d': returns.get('return_5d'),
            'actual_return_10d': returns.get('return_10d'),
            'actual_return_15d': returns.get('return_15d'),
        }
        results.append(result)

    return results

def analyze_results(results):
    """分析回测结果"""
    df = pd.DataFrame(results)

    print(f"\n{'='*60}")
    print("回测结果分析")
    print(f"{'='*60}")

    print(f"\n总样本数: {len(df)}")
    print(f"评分范围: {df['score'].min():.1f} - {df['score'].max():.1f}")
    print(f"日期范围: {df['date'].min()} ~ {df['date'].max()}")

    # 按评分区间分析
    score_bins = [(87, 88), (88, 89), (89, 90), (90, 100)]

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

        for period in [5, 10, 15]:
            col = f'actual_return_{period}d'
            valid = subset[col].dropna()
            if len(valid) > 0:
                row[f'avg_{period}d'] = valid.mean() * 100
                row[f'median_{period}d'] = valid.median() * 100
                row[f'win_rate_{period}d'] = (valid > 0).mean() * 100
                row[f'count_{period}d'] = len(valid)
            else:
                row[f'avg_{period}d'] = None
                row[f'median_{period}d'] = None
                row[f'win_rate_{period}d'] = None
                row[f'count_{period}d'] = 0

        analysis_results.append(row)

        print(f"\n评分 {low}-{high}: {len(subset)} 只股票")
        for period in [5, 10, 15]:
            col = f'actual_return_{period}d'
            valid = subset[col].dropna()
            if len(valid) > 0:
                avg_ret = valid.mean() * 100
                median_ret = valid.median() * 100
                win_rate = (valid > 0).mean() * 100
                print(f"  {period}日: 平均{avg_ret:+.2f}%, 中位数{median_ret:+.2f}%, 胜率{win_rate:.1f}% (n={len(valid)})")

    # 总体分析
    print(f"\n{'='*60}")
    print("总体分析 (所有87分以上)")
    print(f"{'='*60}")

    overall = {'score_range': 'ALL(>=87)', 'count': len(df)}
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
    print("预测收益 vs 实际收益对比")
    print(f"{'='*60}")

    pred_col = 'predicted_return_5d'
    actual_col = 'actual_return_5d'

    valid_mask = df[pred_col].notna() & df[actual_col].notna()
    valid_df = df[valid_mask].copy()

    if len(valid_df) > 0:
        # 转换预测收益（可能是百分比形式）
        if valid_df[pred_col].mean() > 1:
            valid_df[pred_col] = valid_df[pred_col] / 100

        pred_mean = valid_df[pred_col].mean() * 100
        actual_mean = valid_df[actual_col].mean() * 100

        print(f"样本数: {len(valid_df)}")
        print(f"预测5日平均收益: {pred_mean:+.2f}%")
        print(f"实际5日平均收益: {actual_mean:+.2f}%")
        print(f"预测偏差: {pred_mean - actual_mean:+.2f}%")

        # 计算相关性
        corr = valid_df[pred_col].corr(valid_df[actual_col])
        print(f"预测与实际相关系数: {corr:.3f}")

    return df, analysis_results

def save_results(df, analysis_results):
    """保存结果"""
    output_dir = Path('reports/backtest')
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 保存详细数据
    detail_file = output_dir / f'v395_score87_backtest_detail_{timestamp}.csv'
    df.to_csv(detail_file, index=False, encoding='utf-8-sig')
    print(f"\n详细数据已保存: {detail_file}")

    # 保存分析结果
    analysis_file = output_dir / f'v395_score87_backtest_analysis_{timestamp}.json'
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)
    print(f"分析结果已保存: {analysis_file}")

    # 生成Markdown报告
    report_file = output_dir / f'v395_score87_backtest_report_{timestamp}.md'
    generate_markdown_report(df, analysis_results, report_file)
    print(f"报告已保存: {report_file}")

    return detail_file, analysis_file, report_file

def generate_markdown_report(df, analysis_results, output_file):
    """生成Markdown报告"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# V3.95 高分股票(>=87分)回测报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**数据范围**: {START_DATE} ~ {END_DATE}\n\n")
        f.write(f"**评分阈值**: >= {SCORE_THRESHOLD}\n\n")

        f.write("## 总体统计\n\n")
        f.write(f"- 总样本数: {len(df)}\n")
        f.write(f"- 评分范围: {df['score'].min():.1f} - {df['score'].max():.1f}\n")
        f.write(f"- 涉及交易日: {df['date'].nunique()} 天\n\n")

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

        # 展开策略列表
        strategy_returns = {}
        for _, row in df.iterrows():
            strategies = row.get('strategies', [])
            if isinstance(strategies, str):
                strategies = [s.strip() for s in strategies.split(',')]

            for strategy in strategies:
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

        f.write("\n## 高分股票明细 (Top 50)\n\n")
        f.write("| 日期 | 代码 | 名称 | 评分 | 策略 | 5日收益 | 10日收益 | 15日收益 |\n")
        f.write("|------|------|------|------|------|---------|----------|----------|\n")

        top_stocks = df.nlargest(50, 'score')
        for _, row in top_stocks.iterrows():
            strategies = row.get('strategies', [])
            if isinstance(strategies, list):
                strategies = ', '.join(strategies[:2])

            r5 = row.get('actual_return_5d')
            r10 = row.get('actual_return_10d')
            r15 = row.get('actual_return_15d')

            r5_str = f"{r5*100:+.2f}%" if r5 is not None and not pd.isna(r5) else "N/A"
            r10_str = f"{r10*100:+.2f}%" if r10 is not None and not pd.isna(r10) else "N/A"
            r15_str = f"{r15*100:+.2f}%" if r15 is not None and not pd.isna(r15) else "N/A"

            f.write(f"| {row['date']} | {row['code']} | {row['name']} | {row['score']:.1f} | {strategies} | {r5_str} | {r10_str} | {r15_str} |\n")

def main():
    print("="*60)
    print("V3.95 高分股票(>=87分) 批量生成与回测")
    print("="*60)

    # 1. 获取交易日和已有报告
    trading_days = get_trading_days(START_DATE, END_DATE)
    existing_reports = get_existing_reports()

    print(f"\n交易日总数: {len(trading_days)}")
    print(f"已有报告数: {len(existing_reports)}")

    # 2. 找出缺失的报告
    missing_dates = [d for d in trading_days if d not in existing_reports]
    print(f"缺失报告数: {len(missing_dates)}")

    # 3. 批量生成缺失报告
    if missing_dates:
        print(f"\n开始批量生成 {len(missing_dates)} 个缺失报告...")
        success_count = 0
        for i, date in enumerate(missing_dates):
            print(f"\n[{i+1}/{len(missing_dates)}] ", end="")
            if generate_report(date):
                success_count += 1
        print(f"\n报告生成完成: 成功 {success_count}/{len(missing_dates)}")

    # 4. 收集高分股票
    print(f"\n收集 {SCORE_THRESHOLD} 分以上股票...")
    high_score_stocks = collect_high_score_stocks()
    print(f"共找到 {len(high_score_stocks)} 只高分股票")

    if not high_score_stocks:
        print("没有找到高分股票，退出")
        return

    # 5. 运行回测
    results = run_backtest(high_score_stocks)

    # 6. 分析结果
    df, analysis_results = analyze_results(results)

    # 7. 保存结果
    save_results(df, analysis_results)

    print(f"\n{'='*60}")
    print("回测完成!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
