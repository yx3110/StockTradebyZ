#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无锁并行数据填补 - 最终解决方案
每个worker使用独立数据库,完成后合并
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import sqlite3
import json
import shutil
from datetime import datetime
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from pathlib import Path


# Worker全局变量
_worker_id = None
_worker_db_path = None
_worker_system = None
_worker_counter = None
_counter_lock = None


def init_with_counter(counter, lock, temp_dir):
    """使用共享计数器分配worker ID"""
    import os
    # 使用进程PID作为worker ID更简单
    wid = os.getpid() % 1000  # 取模避免ID过大
    worker_init(wid, temp_dir)


def worker_init(worker_id, temp_dir):
    """初始化worker - 每个worker有独立数据库"""
    global _worker_id, _worker_db_path, _worker_system

    _worker_id = worker_id
    _worker_db_path = f"{temp_dir}/worker_{worker_id}.db"

    # 抑制日志
    import logging
    logging.getLogger('ml_models.v39.features.technical_features').setLevel(logging.ERROR)
    logging.getLogger('ml_models.v39.features.fundamental_features').setLevel(logging.ERROR)
    logging.getLogger('ml_models.v39.features.market_features').setLevel(logging.ERROR)

    # 创建worker专属数据库
    conn = sqlite3.connect(_worker_db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS v39_cache (
            code TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            features_json TEXT NOT NULL,
            label_5d REAL,
            PRIMARY KEY (code, trade_date)
        )
    """)
    conn.close()

    # 初始化特征提取器
    from ml_models.v39.v390_enhanced_feature_ml_system import V390EnhancedFeatureMLSystem
    _worker_system = V390EnhancedFeatureMLSystem()

    print(f"Worker {worker_id} 初始化完成 → {_worker_db_path}")


def process_task(args):
    """
    处理单个任务
    写入worker自己的数据库(无锁竞争)
    """
    code, trade_date = args
    global _worker_db_path, _worker_system

    try:
        # 提取特征
        features = _worker_system.extract_features(code, trade_date)
        if features is None or len(features) == 0:
            return ('failed', f'{code} {trade_date}: 无法提取特征')

        # 计算标签
        label = _worker_system.calculate_label(code, trade_date)
        if label is None:
            return ('failed', f'{code} {trade_date}: 无法计算标签')

        # 保存到worker数据库(独立文件,无锁)
        feature_dict = features.iloc[0].to_dict()
        features_json = json.dumps(feature_dict)

        conn = sqlite3.connect(_worker_db_path)
        conn.execute("""
            INSERT OR REPLACE INTO v39_cache (code, trade_date, features_json, label_5d)
            VALUES (?, ?, ?, ?)
        """, (code, trade_date, features_json, label))
        conn.commit()
        conn.close()

        return ('saved', '')

    except Exception as e:
        return ('failed', f'{code} {trade_date}: {str(e)}')


def get_trading_dates(db_path, start_date, end_date):
    """获取交易日"""
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


def merge_worker_databases(main_db_path, temp_dir, num_workers):
    """
    合并所有worker数据库到主数据库
    单线程操作,但很快(批量INSERT)
    """
    print("\n" + "="*80)
    print("合并worker数据库到主数据库...")
    print("="*80)

    main_conn = sqlite3.connect(main_db_path)
    total_merged = 0

    # 扫描temp目录找到实际的worker数据库文件
    import glob
    worker_files = glob.glob(f"{temp_dir}/worker_*.db")

    if not worker_files:
        print(f"⚠️  未找到任何worker数据库文件")
        main_conn.close()
        return 0

    print(f"发现 {len(worker_files)} 个worker数据库文件")

    for worker_db in worker_files:
        # 提取worker ID (从文件名)
        worker_id = os.path.basename(worker_db).replace('worker_', '').replace('.db', '')

        # ATTACH worker数据库
        main_conn.execute(f"ATTACH DATABASE '{worker_db}' AS worker_{worker_id}")

        # 批量复制数据
        cursor = main_conn.execute(f"""
            SELECT COUNT(*) FROM worker_{worker_id}.v39_cache
        """)
        count = cursor.fetchone()[0]

        if count > 0:
            main_conn.execute(f"""
                INSERT OR REPLACE INTO v39_feature_cache (code, trade_date, features_json, label_5d)
                SELECT code, trade_date, features_json, label_5d
                FROM worker_{worker_id}.v39_cache
            """)
            main_conn.commit()
            total_merged += count
            print(f"✅ Worker {worker_id}: {count:,} 条记录已合并")
        else:
            print(f"⚠️  Worker {worker_id}: 数据库为空")

        # DETACH
        main_conn.execute(f"DETACH DATABASE worker_{worker_id}")

    main_conn.close()

    print(f"\n总计合并: {total_merged:,} 条记录")
    return total_merged


def cleanup_temp_files(temp_dir):
    """清理临时文件"""
    print("\n清理临时文件...")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print("✅ 临时文件已清理")


def fill_gap_lockfree(start_date, end_date, stock_limit=1000, max_workers=None):
    """无锁并行填补"""
    main_db_path = 'data_adapter/stock_data.db'
    temp_dir = './temp_v39_cache'

    print("="*80)
    print("V3.9无锁并行数据填补")
    print("="*80)
    print(f"策略: 每个worker使用独立数据库,完成后合并")
    print(f"优势: 彻底避免SQLite写入锁竞争")
    print("="*80)

    # 创建临时目录
    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    # 获取任务
    print(f"\n[1/5] 准备任务...")
    dates = get_trading_dates(main_db_path, start_date, end_date)
    stocks = get_stock_list(main_db_path, stock_limit)

    tasks = [(code, date) for date in dates for code in stocks]
    total_tasks = len(tasks)

    print(f"  交易日: {len(dates)}")
    print(f"  股票数: {len(stocks)}")
    print(f"  总任务: {total_tasks:,}")

    # 确定worker数量
    if max_workers is None:
        max_workers = max(1, cpu_count() - 1)

    print(f"\n[2/5] 启动{max_workers}个worker(每个使用独立数据库)...")

    # 并行处理
    print(f"\n[3/5] 并行处理任务...")
    start_time = datetime.now()
    saved = 0
    failed = 0
    error_samples = []  # 收集错误样本

    # 创建进程池 - 使用进程PID作为worker ID
    from multiprocessing import Manager
    manager = Manager()
    worker_counter = manager.Value('i', 0)
    counter_lock = manager.Lock()

    with Pool(processes=max_workers,
              initializer=init_with_counter,
              initargs=(worker_counter, counter_lock, temp_dir)) as pool:

        with tqdm(total=total_tasks, desc="处理进度") as pbar:
            for status, error_msg in pool.imap_unordered(process_task, tasks, chunksize=100):
                if status == 'saved':
                    saved += 1
                else:
                    failed += 1
                    # 收集前20个错误样本用于调试
                    if len(error_samples) < 20:
                        error_samples.append(error_msg)

                pbar.update(1)

                if (saved + failed) % 1000 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    speed = saved / elapsed if elapsed > 0 else 0
                    pbar.set_postfix({
                        '成功': f'{saved:,}',
                        '失败': f'{failed:,}',
                        '速度': f'{speed:.1f}/s'
                    })

    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"\n[4/5] 处理完成:")
    print(f"  耗时: {elapsed/60:.1f} 分钟")
    print(f"  成功: {saved:,}")
    print(f"  失败: {failed:,}")
    print(f"  速度: {saved/elapsed:.1f} samples/s" if elapsed > 0 else "  速度: N/A")

    # 如果有失败,显示错误样本
    if error_samples:
        print(f"\n⚠️  失败样本示例 (前{len(error_samples)}个):")
        for i, err in enumerate(error_samples[:10], 1):
            print(f"  {i}. {err}")
        if len(error_samples) > 10:
            print(f"  ... 还有 {len(error_samples) - 10} 个错误样本")

    # 合并数据库
    print(f"\n[5/5] 合并结果...")
    total_merged = merge_worker_databases(main_db_path, temp_dir, max_workers)

    # 清理
    cleanup_temp_files(temp_dir)

    # 验证
    conn = sqlite3.connect(main_db_path)
    cursor = conn.execute("""
        SELECT COUNT(*) FROM v39_feature_cache
        WHERE trade_date >= ? AND trade_date <= ?
    """, (start_date, end_date))
    final_count = cursor.fetchone()[0]
    conn.close()

    print("\n" + "="*80)
    print("完成!")
    print("="*80)
    print(f"  期间缓存: {final_count:,} 个样本")
    print("="*80)

    return final_count


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='V3.9无锁并行数据填补')
    parser.add_argument('--start', type=str, required=True, help='开始日期')
    parser.add_argument('--end', type=str, required=True, help='结束日期')
    parser.add_argument('--stocks', type=int, default=1000, help='股票数量')
    parser.add_argument('--workers', type=int, default=None, help='Worker数量')

    args = parser.parse_args()

    fill_gap_lockfree(
        start_date=args.start,
        end_date=args.end,
        stock_limit=args.stocks,
        max_workers=args.workers
    )
