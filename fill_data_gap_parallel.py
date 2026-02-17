#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行数据缺口填补脚本 - 修复版
每个worker独立处理和保存数据,避免锁竞争
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import sqlite3
import json
from datetime import datetime
from multiprocessing import Pool, Manager, cpu_count
from tqdm import tqdm

# 全局变量供worker使用
_worker_system = None
_db_path = None


def worker_init(db_path):
    """初始化worker"""
    global _worker_system, _db_path
    _db_path = db_path

    # 抑制WARNING日志
    import logging
    logging.getLogger('ml_models.v39.features.technical_features').setLevel(logging.ERROR)
    logging.getLogger('ml_models.v39.features.fundamental_features').setLevel(logging.ERROR)
    logging.getLogger('ml_models.v39.features.market_features').setLevel(logging.ERROR)

    from ml_models.v39.v390_enhanced_feature_ml_system import V390EnhancedFeatureMLSystem
    _worker_system = V390EnhancedFeatureMLSystem()


def process_single_task(args):
    """
    处理单个任务(一个股票一个日期)
    每个worker独立保存数据

    Returns:
        (status, code, date): status in ['saved', 'skipped', 'failed']
    """
    code, trade_date = args
    global _worker_system, _db_path

    # 每个任务独立的数据库连接
    conn = sqlite3.connect(_db_path, timeout=30.0)

    try:
        # 检查是否已存在
        cursor = conn.execute(
            "SELECT 1 FROM v39_feature_cache WHERE code=? AND trade_date=?",
            (code, trade_date)
        )
        if cursor.fetchone():
            conn.close()
            return ('skipped', code, trade_date)

        # 提取特征
        features = _worker_system.extract_features(code, trade_date)
        if features is None or len(features) == 0:
            conn.close()
            return ('failed', code, trade_date)

        # 计算标签
        label = _worker_system.calculate_label(code, trade_date, lookahead_days=5)
        if label is None:
            conn.close()
            return ('failed', code, trade_date)

        # 准备特征JSON
        feature_dict = features.iloc[0].to_dict()
        features_json = json.dumps({
            k: v for k, v in feature_dict.items()
        })

        # 保存到数据库
        conn.execute("""
            INSERT OR REPLACE INTO v39_feature_cache
            (code, trade_date, features_json, label_5d)
            VALUES (?, ?, ?, ?)
        """, (code, trade_date, features_json, label))

        conn.commit()
        conn.close()

        return ('saved', code, trade_date)

    except Exception as e:
        try:
            conn.close()
        except:
            pass
        return ('failed', code, trade_date)


def get_trading_dates(db_path, start_date, end_date):
    """获取交易日列表"""
    conn = sqlite3.connect(db_path)
    query = """
        SELECT DISTINCT trade_date
        FROM daily_quotes
        WHERE trade_date >= ? AND trade_date <= ?
        ORDER BY trade_date
    """
    cursor = conn.execute(query, (start_date, end_date))
    dates = [row[0] for row in cursor.fetchall()]
    conn.close()
    return dates


def get_stock_list(db_path, limit=1000):
    """获取股票列表"""
    conn = sqlite3.connect(db_path)
    query = """
        SELECT DISTINCT code
        FROM securities
        WHERE type = 'A股'
          AND name NOT LIKE '%ST%'
          AND name NOT LIKE '%退%'
        ORDER BY code
        LIMIT ?
    """
    cursor = conn.execute(query, (limit,))
    stocks = [row[0] for row in cursor.fetchall()]
    conn.close()
    return stocks


def fill_gap_parallel(start_date='2023-01-01', end_date='2024-12-31',
                     stock_limit=1000, max_workers=None):
    """并行填补数据缺口"""
    db_path = 'data_adapter/stock_data.db'

    print("=" * 80)
    print("V3.9数据缺口填补 - 并行版")
    print("=" * 80)

    # 获取交易日
    print(f"\n[1/4] 获取交易日 ({start_date} ~ {end_date})...")
    dates = get_trading_dates(db_path, start_date, end_date)
    print(f"  找到 {len(dates)} 个交易日")

    # 获取股票列表
    print(f"\n[2/4] 获取股票列表 (限制{stock_limit}只)...")
    stocks = get_stock_list(db_path, stock_limit)
    print(f"  找到 {len(stocks)} 只股票")

    # 生成任务列表
    print(f"\n[3/4] 生成任务列表...")
    tasks = []
    for date in dates:
        for code in stocks:
            tasks.append((code, date))

    total_tasks = len(tasks)
    print(f"  总任务: {total_tasks:,}")

    # 并行处理
    if max_workers is None:
        max_workers = max(1, cpu_count() - 1)

    print(f"\n[4/4] 并行处理 (使用{max_workers}个进程)...")
    start_time = datetime.now()

    saved = 0
    skipped = 0
    failed = 0

    # 创建进程池
    with Pool(processes=max_workers,
              initializer=worker_init,
              initargs=(db_path,)) as pool:

        # 使用imap_unordered处理任务
        with tqdm(total=total_tasks, desc="填补进度") as pbar:
            for status, code, date in pool.imap_unordered(process_single_task, tasks, chunksize=50):
                if status == 'saved':
                    saved += 1
                elif status == 'skipped':
                    skipped += 1
                else:
                    failed += 1

                pbar.update(1)

                # 每1000个更新一次显示
                if (saved + skipped + failed) % 1000 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    speed = saved / elapsed if elapsed > 0 else 0
                    pbar.set_postfix({
                        '新增': f'{saved:,}',
                        '跳过': f'{skipped:,}',
                        '失败': f'{failed:,}',
                        '速度': f'{speed:.1f}/s'
                    })

    # 统计结果
    elapsed = (datetime.now() - start_time).total_seconds()

    print("\n" + "=" * 80)
    print("填补完成!")
    print("=" * 80)
    print(f"  总耗时: {elapsed/60:.1f} 分钟")
    print(f"  新增样本: {saved:,}")
    print(f"  已存在跳过: {skipped:,}")
    print(f"  提取失败: {failed:,}")
    print(f"  处理速度: {saved/elapsed:.1f} samples/s")
    print("=" * 80)

    # 验证结果
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("""
        SELECT COUNT(*) FROM v39_feature_cache
        WHERE trade_date >= ? AND trade_date <= ?
    """, (start_date, end_date))
    total_cached = cursor.fetchone()[0]
    conn.close()

    print(f"\n✅ {start_date}~{end_date}期间共缓存 {total_cached:,} 个样本")

    return saved


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='并行填补V3.9数据缺口')
    parser.add_argument('--start', type=str, default='2023-01-01',
                       help='开始日期')
    parser.add_argument('--end', type=str, default='2024-12-31',
                       help='结束日期')
    parser.add_argument('--stocks', type=int, default=1000,
                       help='股票数量')
    parser.add_argument('--workers', type=int, default=None,
                       help='并行进程数 (默认CPU核心数-1)')

    args = parser.parse_args()

    fill_gap_parallel(
        start_date=args.start,
        end_date=args.end,
        stock_limit=args.stocks,
        max_workers=args.workers
    )
