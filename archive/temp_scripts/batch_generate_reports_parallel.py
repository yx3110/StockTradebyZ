#!/usr/bin/env python3
"""
高效并行批量生成V3.90和V3.95报告

优化策略:
1. 多进程并行生成多个日期的报告
2. 同时处理v3.90和v3.95版本
3. 进度实时显示
"""

import os
import sys
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

DB_PATH = 'data_adapter/stock_data.db'

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

def generate_single_report(args):
    """生成单个报告"""
    date_str, version = args
    try:
        result = subprocess.run(
            ['python3', 'tomorrow_stock_selector.py', date_str, '--scoring-version', version],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=os.path.dirname(os.path.abspath(__file__)) or '.'
        )
        if result.returncode == 0:
            return (date_str, version, True, None)
        else:
            return (date_str, version, False, result.stderr[:200])
    except subprocess.TimeoutExpired:
        return (date_str, version, False, "Timeout")
    except Exception as e:
        return (date_str, version, False, str(e))

def get_existing_reports(report_dir):
    """获取已有报告的日期"""
    report_path = Path(report_dir)
    existing = set()
    if report_path.exists():
        for f in report_path.glob('analysis_data_*.json'):
            date_str = f.stem.replace('analysis_data_', '')
            if len(date_str) == 8:
                existing.add(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")
    return existing

def main():
    parser = argparse.ArgumentParser(description='并行批量生成V3.90/V3.95报告')
    parser.add_argument('--start-date', default='2025-09-01', help='开始日期')
    parser.add_argument('--end-date', default=datetime.now().strftime('%Y-%m-%d'), help='结束日期')
    parser.add_argument('--versions', default='v3.90,v3.95', help='版本列表，逗号分隔')
    parser.add_argument('--workers', type=int, default=min(4, cpu_count()), help='并行进程数')
    parser.add_argument('--force', action='store_true', help='强制重新生成已有报告')
    args = parser.parse_args()

    versions = [v.strip() for v in args.versions.split(',')]

    print("=" * 70)
    print("并行批量生成报告")
    print("=" * 70)
    print(f"日期范围: {args.start_date} ~ {args.end_date}")
    print(f"版本: {versions}")
    print(f"并行进程数: {args.workers}")
    print(f"强制重新生成: {args.force}")
    print("=" * 70)

    # 获取交易日
    trading_days = get_trading_days(args.start_date, args.end_date)
    print(f"交易日总数: {len(trading_days)}")

    # 准备任务列表
    tasks = []
    for version in versions:
        report_dir = f'reports/daily_selection_{version.replace(".", "")}'
        if args.force:
            existing = set()
        else:
            existing = get_existing_reports(report_dir)

        missing = [d for d in trading_days if d not in existing]
        print(f"\n{version}: 已有 {len(existing)} 个, 缺失 {len(missing)} 个")

        for date in missing:
            tasks.append((date, version))

    if not tasks:
        print("\n没有需要生成的报告!")
        return

    print(f"\n总任务数: {len(tasks)}")
    print("-" * 70)

    # 并行执行
    completed = 0
    failed = 0
    start_time = datetime.now()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(generate_single_report, task): task for task in tasks}

        for future in as_completed(futures):
            date_str, version, success, error = future.result()
            completed += 1

            if success:
                status = "✅"
            else:
                status = "❌"
                failed += 1

            elapsed = (datetime.now() - start_time).total_seconds()
            rate = completed / elapsed * 60 if elapsed > 0 else 0
            eta = (len(tasks) - completed) / rate if rate > 0 else 0

            print(f"[{completed}/{len(tasks)}] {status} {version} {date_str} | "
                  f"速率: {rate:.1f}/分钟 | ETA: {eta:.1f}分钟" +
                  (f" | {error}" if error else ""))

    elapsed_total = (datetime.now() - start_time).total_seconds() / 60
    print("\n" + "=" * 70)
    print(f"✅ 完成! 成功: {completed - failed}, 失败: {failed}, 耗时: {elapsed_total:.1f}分钟")
    print("=" * 70)

if __name__ == '__main__':
    main()
