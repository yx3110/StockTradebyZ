#!/usr/bin/env python3
"""
高效批量生成V3.9/V3.95报告 - 深度IO优化版

优化策略:
1. 全局预加载所有数据到内存（只加载一次）
2. 批量查询v39_feature_cache（每日一次查询所有股票）
3. 复用预加载数据，避免重复计算
4. 并行处理多个日期
"""

import os
import sys
import sqlite3
import json
import argparse
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import pandas as pd
import numpy as np
import logging

# macOS 兼容性：使用 fork 模式
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
try:
    multiprocessing.set_start_method('fork', force=True)
except RuntimeError:
    pass

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

DB_PATH = 'data_adapter/stock_data.db'

# ==================== 全局数据预加载 ====================

class GlobalDataCache:
    """全局数据缓存管理器"""

    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.quotes = None  # 行情数据
        self.basic = None   # 基本面数据
        self.securities = None  # 证券信息
        self.feature_cache = {}  # v39_feature_cache按日期缓存
        self.quotes_by_code = {}
        self.basic_by_code = {}
        self.trade_dates = []

    def preload(self, start_date, end_date):
        """预加载所有需要的数据"""
        print("🚀 全局数据预加载...")
        conn = sqlite3.connect(self.db_path)

        # 扩展日期范围以支持历史计算
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        buffer_start = (start_dt - timedelta(days=200)).strftime('%Y-%m-%d')

        # 1. 加载行情数据
        print("  📈 加载行情数据...")
        self.quotes = pd.read_sql(f"""
            SELECT s.code, s.name, dq.trade_date, dq.open, dq.high, dq.low, dq.close,
                   dq.volume, dq.price_change_pct, dq.is_limit_up, dq.is_limit_down
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE s.type = 'A股' AND dq.trade_date BETWEEN '{buffer_start}' AND '{end_date}'
            ORDER BY s.code, dq.trade_date
        """, conn)
        print(f"     ✓ {len(self.quotes):,} 条行情")

        # 2. 加载基本面数据
        print("  💰 加载基本面数据...")
        self.basic = pd.read_sql(f"""
            SELECT s.code, db.trade_date, db.pe_ttm, db.pb, db.ps_ttm,
                   db.total_mv, db.circ_mv, db.turnover_rate
            FROM daily_basic db
            JOIN securities s ON db.security_id = s.id
            WHERE s.type = 'A股' AND db.trade_date BETWEEN '{buffer_start}' AND '{end_date}'
        """, conn)
        print(f"     ✓ {len(self.basic):,} 条基本面")

        # 3. 加载证券信息
        print("  📋 加载证券信息...")
        self.securities = pd.read_sql("""
            SELECT code, name, type, exchange
            FROM securities
            WHERE type = 'A股'
        """, conn)
        print(f"     ✓ {len(self.securities):,} 只证券")

        # 4. 批量加载v39_feature_cache
        print("  🧠 加载特征缓存...")
        feature_df = pd.read_sql(f"""
            SELECT code, trade_date, features_json, label_5d
            FROM v39_feature_cache
            WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
        """, conn)
        # 按日期分组缓存
        for date, group in feature_df.groupby('trade_date'):
            self.feature_cache[date] = {
                row['code']: json.loads(row['features_json']) if row['features_json'] else {}
                for _, row in group.iterrows()
            }
        print(f"     ✓ {len(feature_df):,} 条特征缓存 ({len(self.feature_cache)} 天)")

        conn.close()

        # 5. 创建索引
        print("  🔧 创建内存索引...")
        self.quotes_by_code = {
            code: group.set_index('trade_date').sort_index()
            for code, group in self.quotes.groupby('code')
        }
        self.basic_by_code = {
            code: group.set_index('trade_date').sort_index()
            for code, group in self.basic.groupby('code')
        }
        self.securities_dict = self.securities.set_index('code').to_dict('index')
        self.trade_dates = sorted(self.quotes['trade_date'].unique())

        print(f"  ✅ 预加载完成: {len(self.quotes_by_code)} 只股票, {len(self.trade_dates)} 天")
        return self


# ==================== 快速报告生成 ====================

def generate_report_fast(date_str, version, cache: GlobalDataCache):
    """快速生成单个报告（使用预加载数据）"""
    try:
        # 导入选股器（只导入一次）
        from tomorrow_stock_selector import TomorrowStockSelector

        # 创建选股器实例
        selector = TomorrowStockSelector(scoring_version=version)

        # 注入预加载的数据
        selector.securities_info = cache.securities_dict

        # 运行完整分析
        results = selector.run_full_analysis(date_str)

        return (date_str, version, True, None)
    except Exception as e:
        return (date_str, version, False, str(e)[:100])


def generate_report_subprocess(args):
    """子进程生成报告"""
    date_str, version = args
    import subprocess
    try:
        result = subprocess.run(
            ['python3', 'tomorrow_stock_selector.py', date_str, '--scoring-version', version],
            capture_output=True, text=True, timeout=300,
            cwd=os.path.dirname(os.path.abspath(__file__)) or '.'
        )
        return (date_str, version, result.returncode == 0,
                None if result.returncode == 0 else result.stderr[:50])
    except subprocess.TimeoutExpired:
        return (date_str, version, False, "Timeout")
    except Exception as e:
        return (date_str, version, False, str(e)[:50])


# ==================== 主函数 ====================

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


def main():
    parser = argparse.ArgumentParser(description='批量生成报告 (深度优化版)')
    parser.add_argument('--start-date', default='2025-09-01')
    parser.add_argument('--end-date', default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--versions', default='v3.9,v3.95', help='版本列表')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--force', action='store_true', help='强制重新生成')
    args = parser.parse_args()

    versions = [v.strip() for v in args.versions.split(',')]

    print("=" * 70)
    print("批量生成报告 (深度IO优化版)")
    print("=" * 70)
    print(f"日期范围: {args.start_date} ~ {args.end_date}")
    print(f"版本: {versions}")
    print(f"并行进程: {args.workers}")
    print("=" * 70)

    # 获取交易日
    trading_days = get_trading_days(args.start_date, args.end_date)
    print(f"交易日总数: {len(trading_days)}")

    # 检查已有报告
    tasks = []
    for version in versions:
        report_dir = Path(f'reports/daily_selection_{version.replace(".", "")}')
        report_dir.mkdir(parents=True, exist_ok=True)

        if args.force:
            missing = trading_days
        else:
            existing = set()
            for f in report_dir.glob('analysis_data_*.json'):
                date_str = f.stem.replace('analysis_data_', '')
                if len(date_str) == 8:
                    existing.add(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")
            missing = [d for d in trading_days if d not in existing]

        print(f"\n{version}: 需生成 {len(missing)} 个报告")
        for date in missing:
            tasks.append((date, version))

    if not tasks:
        print("\n没有需要生成的报告!")
        return

    print(f"\n总任务数: {len(tasks)}")
    print("-" * 70)

    # 预加载数据
    print("\n🚀 第一阶段：全局数据预加载")
    cache = GlobalDataCache()
    cache.preload(args.start_date, args.end_date)

    # 并行执行
    print("\n🔥 第二阶段：生成报告")
    print("-" * 70)

    completed = 0
    failed = 0
    start_time = datetime.now()

    # 串行模式（避免macOS multiprocessing问题）
    for task in tasks:
        date_str, version, success, error = generate_report_subprocess(task)
        completed += 1

        if not success:
            failed += 1

        elapsed = (datetime.now() - start_time).total_seconds()
        rate = completed / elapsed * 60 if elapsed > 0 else 0
        remaining = len(tasks) - completed
        eta = remaining / rate if rate > 0 else 0

        status = "✅" if success else "❌"
        print(f"[{completed}/{len(tasks)}] {status} {version} {date_str} | "
              f"{rate:.1f}/分 | ETA: {eta:.0f}分" +
              (f" | {error}" if error else ""))

    elapsed_total = (datetime.now() - start_time).total_seconds() / 60
    print("\n" + "=" * 70)
    print(f"✅ 完成! 成功: {completed - failed}, 失败: {failed}, 耗时: {elapsed_total:.1f}分钟")
    print("=" * 70)


if __name__ == '__main__':
    main()
