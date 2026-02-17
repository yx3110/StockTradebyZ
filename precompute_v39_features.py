#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.9特征批量预计算脚本
目的：一次性计算所有特征并保存到数据库，加速后续训练
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import argparse
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from multiprocessing import Pool, cpu_count
from ml_models.v39 import V390EnhancedFeatureMLSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局变量：每个worker的系统实例
_worker_system = None

def init_worker(lookback_days, lookahead_days):
    """Worker进程初始化"""
    global _worker_system
    _worker_system = V390EnhancedFeatureMLSystem(
        lookback_days=lookback_days,
        lookahead_days=lookahead_days
    )
    logger.info(f"Worker initialized with lookback={lookback_days}, lookahead={lookahead_days}")


def compute_features_batch(args):
    """
    计算一批样本的特征

    Args:
        args: (股票代码列表, 交易日期)

    Returns:
        list of dicts: 特征字典列表
    """
    codes, date = args
    global _worker_system

    results = []
    for code in codes:
        try:
            # 提取特征
            features = _worker_system.extract_features(code, date)
            if features is None or features.empty:
                continue

            # 计算标签
            label = _worker_system.calculate_label(code, date)
            if label is None:
                continue

            # 构造特征字典
            feature_dict = features.iloc[0].to_dict()
            feature_dict['code'] = code
            feature_dict['trade_date'] = date
            feature_dict['label_5d'] = label

            results.append(feature_dict)

        except Exception as e:
            continue

    return results


class V39FeaturePrecomputer:
    """V3.9特征预计算器"""

    def __init__(self, db_path='data_adapter/stock_data.db'):
        self.db_path = db_path
        self.num_workers = min(cpu_count(), 8)

    def get_stocks_and_dates(self, start_date, end_date, sample_stocks=None):
        """
        获取股票列表和交易日列表

        Returns:
            (stock_list, trade_dates): 股票列表和交易日列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 获取股票列表
        if sample_stocks:
            cursor.execute(f"""
                SELECT code FROM securities
                WHERE type='A股'
                ORDER BY RANDOM()
                LIMIT {sample_stocks}
            """)
        else:
            cursor.execute("SELECT code FROM securities WHERE type='A股'")

        stock_list = [row[0] for row in cursor.fetchall()]

        # 获取交易日列表
        cursor.execute("""
            SELECT DISTINCT trade_date
            FROM daily_quotes
            WHERE trade_date BETWEEN ? AND ?
            ORDER BY trade_date
        """, (start_date, end_date))

        trade_dates = [row[0] for row in cursor.fetchall()]
        conn.close()

        return stock_list, trade_dates

    def batch_insert_features(self, feature_dicts):
        """
        批量插入特征到数据库（使用JSON存储）

        Args:
            feature_dicts: 特征字典列表
        """
        if not feature_dicts:
            return 0

        import json

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 构建INSERT语句（JSON存储方案）
        sql = """
            INSERT OR REPLACE INTO v39_feature_cache
            (code, trade_date, features_json, label_5d)
            VALUES (?, ?, ?, ?)
        """

        # 准备数据
        values = []
        for d in feature_dicts:
            code = d.pop('code')
            trade_date = d.pop('trade_date')
            label_5d = d.pop('label_5d', None)

            # 剩余的所有字段作为JSON存储
            features_json = json.dumps(d, ensure_ascii=False)

            values.append((code, trade_date, features_json, label_5d))

        # 批量插入
        cursor.executemany(sql, values)

        conn.commit()
        inserted = cursor.rowcount
        conn.close()

        return inserted

    def precompute(self, start_date, end_date, sample_stocks=None,
                   lookback_days=10, lookahead_days=5, batch_size=20):
        """
        批量预计算特征

        Args:
            start_date: 开始日期
            end_date: 结束日期
            sample_stocks: 采样股票数量（None=全部）
            lookback_days: 回望天数
            lookahead_days: 前瞻天数
            batch_size: 每批处理的股票数量
        """
        logger.info("="*80)
        logger.info("🚀 V3.9特征批量预计算")
        logger.info("="*80)
        logger.info(f"时间范围: {start_date} ~ {end_date}")
        logger.info(f"回望天数: {lookback_days}, 前瞻天数: {lookahead_days}")
        logger.info(f"批处理大小: {batch_size}")
        logger.info(f"并行进程数: {self.num_workers}")

        # 获取股票和日期列表
        stock_list, trade_dates = self.get_stocks_and_dates(
            start_date, end_date, sample_stocks
        )

        # 过滤掉最后lookahead_days天（无法计算标签）
        valid_dates = trade_dates[:-lookahead_days]

        logger.info(f"股票数量: {len(stock_list)}")
        logger.info(f"交易日数量: {len(trade_dates)}")
        logger.info(f"有效日期数: {len(valid_dates)}")

        total_tasks = len(valid_dates)
        logger.info(f"预计样本数: {len(stock_list)} × {len(valid_dates)} = {len(stock_list) * len(valid_dates):,}")

        # 将股票列表分批
        stock_batches = [stock_list[i:i+batch_size]
                         for i in range(0, len(stock_list), batch_size)]

        # 构建任务列表：[(股票批次, 日期), ...]
        tasks = [(batch, date) for date in valid_dates for batch in stock_batches]

        logger.info(f"总任务数: {len(tasks):,}")
        logger.info("\n🚀 启动并行预计算...")

        # 并行处理
        total_inserted = 0
        start_time = datetime.now()

        with Pool(processes=self.num_workers,
                  initializer=init_worker,
                  initargs=(lookback_days, lookahead_days)) as pool:

            # 使用imap_unordered处理任务
            for i, result_batch in enumerate(pool.imap_unordered(
                compute_features_batch, tasks, chunksize=1
            ), 1):
                # 批量插入到数据库
                if result_batch:
                    inserted = self.batch_insert_features(result_batch)
                    total_inserted += inserted

                # 进度报告
                if i % 100 == 0 or i == len(tasks):
                    progress = i / len(tasks) * 100
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = i / elapsed if elapsed > 0 else 0
                    eta_seconds = (len(tasks) - i) / rate if rate > 0 else 0
                    eta_minutes = eta_seconds / 60

                    logger.info(
                        f"进度: {progress:.1f}% ({i}/{len(tasks)}) | "
                        f"已插入: {total_inserted:,} | "
                        f"速率: {rate:.1f}任务/秒 | "
                        f"剩余: {eta_minutes:.1f}分钟"
                    )

        elapsed_total = (datetime.now() - start_time).total_seconds() / 60

        logger.info("\n" + "="*80)
        logger.info(f"✅ 预计算完成!")
        logger.info(f"总插入数: {total_inserted:,}")
        logger.info(f"总耗时: {elapsed_total:.1f}分钟")
        logger.info(f"平均速率: {total_inserted/elapsed_total:.0f}样本/分钟")
        logger.info("="*80)

        # 更新统计信息
        self._update_stats(total_inserted, start_date, end_date, len(stock_list))

    def _update_stats(self, total_records, start_date, end_date, stock_count):
        """更新统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO v39_feature_cache_stats
            (total_records, date_range_start, date_range_end, stock_count)
            VALUES (?, ?, ?, ?)
        """, (total_records, start_date, end_date, stock_count))

        conn.commit()
        conn.close()


def main():
    parser = argparse.ArgumentParser(description='V3.9特征批量预计算')
    parser.add_argument('--start-date', type=str, default='2025-06-12', help='开始日期')
    parser.add_argument('--end-date', type=str, default='2025-11-04', help='结束日期')
    parser.add_argument('--lookback-days', type=int, default=10, help='回望天数')
    parser.add_argument('--lookahead-days', type=int, default=5, help='前瞻天数')
    parser.add_argument('--sample-stocks', type=int, default=None, help='采样股票数量')
    parser.add_argument('--batch-size', type=int, default=20, help='每批处理的股票数')
    parser.add_argument('--num-workers', type=int, default=None, help='并行进程数')

    args = parser.parse_args()

    precomputer = V39FeaturePrecomputer()

    if args.num_workers:
        precomputer.num_workers = args.num_workers

    precomputer.precompute(
        start_date=args.start_date,
        end_date=args.end_date,
        sample_stocks=args.sample_stocks,
        lookback_days=args.lookback_days,
        lookahead_days=args.lookahead_days,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
