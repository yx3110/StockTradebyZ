#!/usr/bin/env python3
"""批量生成选股报告 (支持 v4.0 / v4.3 等版本)"""

import sys
import sqlite3
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / 'data_adapter' / 'stock_data.db'

# 各版本使用的 feature_cache 表
VERSION_CACHE_TABLE = {
    'v4.0': 'v40_feature_cache',
    'v4.2': 'v40_feature_cache',
    'v4.3': 'v39_feature_cache',  # V4.3 基于 v39 特征 + 技术指标
}

VERSION_REPORT_DIR = {
    'v4.0': 'daily_selection_v4.0',
    'v4.2': 'daily_selection_v4.2',
    'v4.3': 'daily_selection_v4.3',
}


def get_trading_dates(start_date, end_date, cache_table='v39_feature_cache'):
    """从 feature_cache 获取交易日列表"""
    conn = sqlite3.connect(str(DB_PATH))
    dates = [r[0] for r in conn.execute(f"""
        SELECT DISTINCT trade_date FROM {cache_table}
        WHERE trade_date >= ? AND trade_date <= ?
        ORDER BY trade_date
    """, (start_date, end_date)).fetchall()]
    conn.close()
    return dates

def main():
    import argparse
    parser = argparse.ArgumentParser(description='批量生成选股报告')
    parser.add_argument('--force', action='store_true', help='强制重新生成所有报告')
    parser.add_argument('--version', default='v4.0', choices=['v4.0', 'v4.2', 'v4.3'],
                        help='评分版本 (默认v4.0)')
    parser.add_argument('--start-date', default='2025-09-01', help='开始日期')
    parser.add_argument('--end-date', default='2026-02-13', help='结束日期')
    args = parser.parse_args()

    version = args.version
    cache_table = VERSION_CACHE_TABLE.get(version, 'v39_feature_cache')
    report_subdir = VERSION_REPORT_DIR.get(version, f'daily_selection_{version}')

    dates = get_trading_dates(args.start_date, args.end_date, cache_table)
    if not dates:
        print(f"未找到 {args.start_date} ~ {args.end_date} 的交易日数据")
        return

    print(f"📋 {version} 待生成报告: {len(dates)} 天 ({dates[0]} ~ {dates[-1]})")

    report_dir = PROJECT_ROOT / 'reports' / report_subdir
    report_dir.mkdir(parents=True, exist_ok=True)

    if args.force:
        remaining = dates
        print(f"⚠️ 强制重新生成所有 {len(remaining)} 天报告")
    else:
        # 检查已有报告
        existing = set()
        for f in report_dir.glob('选股分析报告_*.md'):
            date_str = f.stem.split('_')[-1]
            existing.add(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")

        remaining = [d for d in dates if d not in existing]
        print(f"✅ 已有报告: {len(existing)}, 待生成: {len(remaining)}")

    if not remaining:
        print("所有报告已生成!")
        return

    start_time = datetime.now()
    success = 0
    failed = 0

    for i, date in enumerate(remaining):
        try:
            result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / 'tomorrow_stock_selector.py'),
                 date, '--scoring-version', version],
                capture_output=True, text=True, timeout=300,
                cwd=str(PROJECT_ROOT)
            )

            # 检查是否生成了报告
            date_str = date.replace('-', '')
            report_file = report_dir / f'选股分析报告_{date_str}.md'
            if report_file.exists():
                success += 1
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = elapsed / (i + 1)
                eta = rate * (len(remaining) - i - 1)
                print(f"  [{i+1}/{len(remaining)}] {date}: ✅ ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")
            else:
                failed += 1
                print(f"  [{i+1}/{len(remaining)}] {date}: ❌ 报告未生成")
                if result.stderr:
                    # Print last 3 lines of stderr
                    lines = result.stderr.strip().split('\n')
                    for line in lines[-3:]:
                        print(f"    {line}")
        except subprocess.TimeoutExpired:
            failed += 1
            print(f"  [{i+1}/{len(remaining)}] {date}: ❌ 超时")
        except Exception as e:
            failed += 1
            print(f"  [{i+1}/{len(remaining)}] {date}: ❌ {e}")

    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\n{'='*60}")
    print(f"📊 {version} 批量生成完成!")
    print(f"  成功: {success}, 失败: {failed}")
    print(f"  总耗时: {total_time:.0f}秒 ({total_time/60:.1f}分钟)")
    print(f"  平均: {total_time/max(success+failed,1):.1f}秒/天")

if __name__ == '__main__':
    main()
