#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.9特征多线程并行补全脚本

同时处理多个时间段，充分利用CPU资源
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count, Manager
import json
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')

# 全局变量
_worker_system = None

def init_worker(lookback_days, lookahead_days):
    """Worker进程初始化"""
    global _worker_system
    from ml_models.v39 import V390EnhancedFeatureMLSystem
    _worker_system = V390EnhancedFeatureMLSystem(
        lookback_days=lookback_days,
        lookahead_days=lookahead_days
    )

def compute_single_sample(args):
    """计算单个样本的特征"""
    code, date = args
    global _worker_system

    try:
        features = _worker_system.extract_features(code, date)
        if features is None or features.empty:
            return None

        label = _worker_system.calculate_label(code, date)
        if label is None:
            return None

        feature_dict = features.iloc[0].to_dict()
        feature_dict['code'] = code
        feature_dict['trade_date'] = date
        feature_dict['label_5d'] = label

        return feature_dict
    except Exception:
        return None

def get_missing_samples(start_date, end_date, sample_stocks=None):
    """获取缺失的样本列表"""
    conn = sqlite3.connect(DB_PATH)

    # 获取所有交易日
    trade_dates = pd.read_sql("""
        SELECT DISTINCT trade_date
        FROM daily_quotes
        WHERE trade_date BETWEEN ? AND ?
        ORDER BY trade_date
    """, conn, params=[start_date, end_date])['trade_date'].tolist()

    # 跳过最后5天（无法计算标签）
    trade_dates = trade_dates[:-5] if len(trade_dates) > 5 else []

    # 获取股票列表
    if sample_stocks:
        stocks = pd.read_sql(f"""
            SELECT code FROM securities
            WHERE type='A股'
            ORDER BY RANDOM()
            LIMIT {sample_stocks}
        """, conn)['code'].tolist()
    else:
        stocks = pd.read_sql("""
            SELECT code FROM securities WHERE type='A股'
        """, conn)['code'].tolist()

    # 获取已有数据
    existing = set()
    try:
        existing_df = pd.read_sql("""
            SELECT code, trade_date
            FROM v39_feature_cache
            WHERE trade_date BETWEEN ? AND ?
        """, conn, params=[start_date, end_date])
        existing = set(zip(existing_df['code'], existing_df['trade_date']))
    except Exception:
        pass

    conn.close()

    # 生成缺失样本列表
    all_samples = [(code, date) for date in trade_dates for code in stocks]
    missing = [s for s in all_samples if s not in existing]

    return missing, len(trade_dates), len(stocks)

def batch_insert(results):
    """批量插入结果到数据库"""
    if not results:
        return 0

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    sql = """
        INSERT OR REPLACE INTO v39_feature_cache
        (code, trade_date, features_json, label_5d)
        VALUES (?, ?, ?, ?)
    """

    values = []
    for d in results:
        if d is None:
            continue
        code = d.pop('code')
        trade_date = d.pop('trade_date')
        label_5d = d.pop('label_5d', None)
        features_json = json.dumps(d, ensure_ascii=False)
        values.append((code, trade_date, features_json, label_5d))

    if values:
        cursor.executemany(sql, values)
        conn.commit()

    conn.close()
    return len(values)

def run_backfill(start_date, end_date, sample_stocks=None, num_workers=None,
                 lookback_days=10, lookahead_days=5, batch_size=100):
    """运行数据补全"""

    if num_workers is None:
        num_workers = min(cpu_count(), 12)

    logger.info("=" * 70)
    logger.info(f"V3.9 数据补全: {start_date} ~ {end_date}")
    logger.info(f"并行进程数: {num_workers}")
    logger.info("=" * 70)

    # 获取缺失样本
    missing, n_dates, n_stocks = get_missing_samples(start_date, end_date, sample_stocks)

    logger.info(f"股票数: {n_stocks}, 交易日: {n_dates}")
    logger.info(f"缺失样本: {len(missing):,}")

    if not missing:
        logger.info("无缺失数据，跳过")
        return 0

    # 分批处理
    total_inserted = 0
    start_time = time.time()

    with Pool(processes=num_workers,
              initializer=init_worker,
              initargs=(lookback_days, lookahead_days)) as pool:

        # 使用imap处理
        results_buffer = []

        for i, result in enumerate(pool.imap_unordered(compute_single_sample, missing, chunksize=10), 1):
            if result is not None:
                results_buffer.append(result)

            # 每batch_size个结果写入一次
            if len(results_buffer) >= batch_size:
                inserted = batch_insert(results_buffer)
                total_inserted += inserted
                results_buffer = []

            # 进度报告
            if i % 500 == 0 or i == len(missing):
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(missing) - i) / rate / 60 if rate > 0 else 0
                pct = i / len(missing) * 100
                logger.info(f"进度: {pct:.1f}% ({i:,}/{len(missing):,}) | "
                           f"已插入: {total_inserted:,} | "
                           f"速率: {rate:.0f}/秒 | "
                           f"ETA: {eta:.1f}分钟")

        # 处理剩余
        if results_buffer:
            inserted = batch_insert(results_buffer)
            total_inserted += inserted

    elapsed_total = (time.time() - start_time) / 60
    logger.info("=" * 70)
    logger.info(f"✅ 完成! 插入: {total_inserted:,}, 耗时: {elapsed_total:.1f}分钟")
    logger.info("=" * 70)

    return total_inserted

def main():
    parser = argparse.ArgumentParser(description='V3.9特征多线程并行补全')
    parser.add_argument('--periods', type=str, default='2024,2025H1',
                        help='时间段，逗号分隔 (2024, 2025H1, 2024H1, 2024H2等)')
    parser.add_argument('--sample-stocks', type=int, default=None,
                        help='采样股票数量（None=全部）')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='并行进程数')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='批量写入大小')

    args = parser.parse_args()

    # 解析时间段
    period_map = {
        '2024': ('2024-01-01', '2024-12-31'),
        '2024H1': ('2024-01-01', '2024-06-30'),
        '2024H2': ('2024-07-01', '2024-12-31'),
        '2025H1': ('2025-01-01', '2025-05-31'),
        '2025': ('2025-01-01', '2025-12-31'),
    }

    periods = args.periods.split(',')

    logger.info(f"将处理 {len(periods)} 个时间段: {periods}")

    total = 0
    for period in periods:
        period = period.strip()
        if period in period_map:
            start_date, end_date = period_map[period]
        else:
            logger.warning(f"未知时间段: {period}, 跳过")
            continue

        inserted = run_backfill(
            start_date=start_date,
            end_date=end_date,
            sample_stocks=args.sample_stocks,
            num_workers=args.num_workers,
            batch_size=args.batch_size
        )
        total += inserted

    logger.info(f"\n🎉 全部完成! 总插入: {total:,}")

if __name__ == '__main__':
    main()
