#!/usr/bin/env python3
"""
批量生成V3.95报告并回测90分（满分）股票的5日/10日/15日收益
日期范围: 2025-09-01 ~ 2025-12-31
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
DB_PATH = 'stock_data.db'
REPORT_DIR = 'reports/daily_selection_v3.95'
START_DATE = '2025-09-01'
END_DATE = '2025-12-31'
SCORE_THRESHOLD = 90  # 90分（满分）

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
    """收集所有90分（满分）的股票"""
    report_path = Path(REPORT_DIR)
    all_stocks = []

    # 只处理指定日期范围内的报告
    start_dt = datetime.strptime(START_DATE, '%Y-%m-%d')
    end_dt = datetime.strptime(END_DATE, '%Y-%m-%d')

    for f in sorted(report_path.glob('analysis_data_*.json')):
        date_str = f.stem.replace('analysis_data_', '')
        date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

        # 检查是否在日期范围内
        try:
            report_dt = datetime.strptime(date_formatted, '%Y-%m-%d')
            if report_dt < start_dt or report_dt > end_dt:
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
        LIMIT 25
    '''

    df = pd.read_sql_query(query, conn, params=(code, buy_date))
    conn.close()

    if len(df) < 2:
        return {f'return_{p}d': None for p in periods}

    # 假设以第二天开盘价买入（因为选股是前一天收盘后做出的）
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
    print(f"开始回测 {len(stocks)} 只90分（满分）股票")
    print(f"{'='*60}")

    results = []
    total = len(stocks)

    for i, stock in enumerate(stocks):
        if (i + 1) % 20 == 0 or i == 0:
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
    print("V3.95 满分(90分)股票回测结果分析")
    print(f"{'='*60}")

    print(f"\n总样本数: {len(df)}")
    if len(df) > 0:
        print(f"评分范围: {df['score'].min():.1f} - {df['score'].max():.1f}")
        print(f"日期范围: {df['date'].min()} ~ {df['date'].max()}")
        print(f"涉及交易日: {df['date'].nunique()} 天")

    # 总体分析
    print(f"\n{'='*60}")
    print("总体分析 (所有90分股票)")
    print(f"{'='*60}")

    analysis_results = []
    overall = {'score_range': f'ALL(>={SCORE_THRESHOLD})', 'count': len(df)}

    for period in [5, 10, 15]:
        col = f'actual_return_{period}d'
        valid = df[col].dropna()
        if len(valid) > 0:
            avg_ret = valid.mean() * 100
            median_ret = valid.median() * 100
            win_rate = (valid > 0).mean() * 100
            std_ret = valid.std() * 100
            max_ret = valid.max() * 100
            min_ret = valid.min() * 100
            overall[f'avg_{period}d'] = avg_ret
            overall[f'median_{period}d'] = median_ret
            overall[f'win_rate_{period}d'] = win_rate
            overall[f'std_{period}d'] = std_ret
            overall[f'max_{period}d'] = max_ret
            overall[f'min_{period}d'] = min_ret
            overall[f'count_{period}d'] = len(valid)
            print(f"\n{period}日收益:")
            print(f"  平均: {avg_ret:+.2f}%")
            print(f"  中位数: {median_ret:+.2f}%")
            print(f"  标准差: {std_ret:.2f}%")
            print(f"  胜率: {win_rate:.1f}%")
            print(f"  最大: {max_ret:+.2f}%")
            print(f"  最小: {min_ret:+.2f}%")
            print(f"  有效样本: {len(valid)}")

    analysis_results.append(overall)

    # 按月份分析
    if len(df) > 0:
        print(f"\n{'='*60}")
        print("按月份分析")
        print(f"{'='*60}")

        df['month'] = df['date'].apply(lambda x: x[:7])
        for month in sorted(df['month'].unique()):
            month_df = df[df['month'] == month]
            print(f"\n{month}: {len(month_df)} 只股票")

            month_result = {'month': month, 'count': len(month_df)}
            for period in [5, 10, 15]:
                col = f'actual_return_{period}d'
                valid = month_df[col].dropna()
                if len(valid) > 0:
                    avg_ret = valid.mean() * 100
                    win_rate = (valid > 0).mean() * 100
                    month_result[f'avg_{period}d'] = avg_ret
                    month_result[f'win_rate_{period}d'] = win_rate
                    print(f"  {period}日: 平均{avg_ret:+.2f}%, 胜率{win_rate:.1f}% (n={len(valid)})")
            analysis_results.append(month_result)

    return df, analysis_results

def save_results(df, analysis_results):
    """保存结果"""
    output_dir = Path('reports/backtest')
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 保存详细数据
    detail_file = output_dir / f'v395_score90_backtest_detail_{timestamp}.csv'
    df.to_csv(detail_file, index=False, encoding='utf-8-sig')
    print(f"\n详细数据已保存: {detail_file}")

    # 保存分析结果
    analysis_file = output_dir / f'v395_score90_backtest_analysis_{timestamp}.json'
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)
    print(f"分析结果已保存: {analysis_file}")

    # 生成Markdown报告
    report_file = output_dir / f'v395_score90_backtest_report_{timestamp}.md'
    generate_markdown_report(df, analysis_results, report_file)
    print(f"报告已保存: {report_file}")

    return detail_file, analysis_file, report_file

def generate_markdown_report(df, analysis_results, output_file):
    """生成Markdown报告"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# V3.95 满分股票(90分)回测报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**数据范围**: {START_DATE} ~ {END_DATE}\n\n")
        f.write(f"**评分阈值**: >= {SCORE_THRESHOLD} 分\n\n")

        f.write("## 核心结论\n\n")

        if len(df) == 0:
            f.write("**未找到满足条件的90分股票。**\n\n")
            return

        f.write(f"- 总样本数: **{len(df)}**\n")
        f.write(f"- 评分范围: {df['score'].min():.1f} - {df['score'].max():.1f}\n")
        f.write(f"- 涉及交易日: {df['date'].nunique()} 天\n\n")

        # 核心指标表格
        f.write("## 收益表现汇总\n\n")
        f.write("| 持有周期 | 平均收益 | 中位数 | 胜率 | 标准差 | 最大 | 最小 | 样本数 |\n")
        f.write("|---------|---------|--------|------|--------|------|------|--------|\n")

        for period in [5, 10, 15]:
            col = f'actual_return_{period}d'
            valid = df[col].dropna() * 100
            if len(valid) > 0:
                f.write(f"| **{period}日** | {valid.mean():+.2f}% | {valid.median():+.2f}% | {(valid > 0).mean() * 100:.1f}% | {valid.std():.2f}% | {valid.max():+.2f}% | {valid.min():+.2f}% | {len(valid)} |\n")
            else:
                f.write(f"| **{period}日** | N/A | N/A | N/A | N/A | N/A | N/A | 0 |\n")

        # 按月份分析
        if 'month' in df.columns or len(df) > 0:
            f.write("\n## 按月份分析\n\n")
            f.write("| 月份 | 样本数 | 5日平均 | 5日胜率 | 10日平均 | 10日胜率 | 15日平均 | 15日胜率 |\n")
            f.write("|------|--------|---------|---------|----------|----------|----------|----------|\n")

            df_temp = df.copy()
            df_temp['month'] = df_temp['date'].apply(lambda x: x[:7])
            for month in sorted(df_temp['month'].unique()):
                month_df = df_temp[df_temp['month'] == month]
                f.write(f"| {month} | {len(month_df)} |")
                for period in [5, 10, 15]:
                    col = f'actual_return_{period}d'
                    valid = month_df[col].dropna()
                    if len(valid) > 0:
                        avg = valid.mean() * 100
                        wr = (valid > 0).mean() * 100
                        f.write(f" {avg:+.2f}% | {wr:.1f}% |")
                    else:
                        f.write(" N/A | N/A |")
                f.write("\n")

        # 股票明细
        f.write("\n## 90分股票明细\n\n")
        f.write("| 日期 | 代码 | 名称 | 评分 | 策略 | 5日收益 | 10日收益 | 15日收益 |\n")
        f.write("|------|------|------|------|------|---------|----------|----------|\n")

        for _, row in df.sort_values(['date', 'score'], ascending=[True, False]).iterrows():
            strategies = row.get('strategies', [])
            if isinstance(strategies, list):
                strategies = ', '.join(strategies[:2]) if strategies else ''
            elif pd.isna(strategies):
                strategies = ''

            r5 = row.get('actual_return_5d')
            r10 = row.get('actual_return_10d')
            r15 = row.get('actual_return_15d')

            r5_str = f"{r5*100:+.2f}%" if r5 is not None and not pd.isna(r5) else "N/A"
            r10_str = f"{r10*100:+.2f}%" if r10 is not None and not pd.isna(r10) else "N/A"
            r15_str = f"{r15*100:+.2f}%" if r15 is not None and not pd.isna(r15) else "N/A"

            f.write(f"| {row['date']} | {row['code']} | {row['name']} | {row['score']:.1f} | {strategies} | {r5_str} | {r10_str} | {r15_str} |\n")

        f.write("\n---\n\n")
        f.write("*报告由V3.95机器学习评分系统生成*\n")

def main():
    print("="*60)
    print("V3.95 满分股票(90分) 批量生成与回测")
    print(f"日期范围: {START_DATE} ~ {END_DATE}")
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

    # 4. 收集90分股票
    print(f"\n收集 {SCORE_THRESHOLD} 分以上股票...")
    high_score_stocks = collect_high_score_stocks()
    print(f"共找到 {len(high_score_stocks)} 只90分股票")

    if not high_score_stocks:
        print("没有找到90分股票")
        # 仍然生成空报告
        df = pd.DataFrame()
        analysis_results = [{'score_range': f'ALL(>={SCORE_THRESHOLD})', 'count': 0}]
        save_results(df, analysis_results)
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
