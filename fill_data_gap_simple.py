#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版数据缺口填补脚本
单线程,稳定可靠,专门填补2023-2024年数据
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import sqlite3
import json
from datetime import datetime
from tqdm import tqdm

from ml_models.v39.v390_enhanced_feature_ml_system import V390EnhancedFeatureMLSystem


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


def check_existing(db_path, code, trade_date):
    """检查是否已缓存"""
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        "SELECT 1 FROM v39_feature_cache WHERE code=? AND trade_date=?",
        (code, trade_date)
    )
    exists = cursor.fetchone() is not None
    conn.close()
    return exists


def save_features(db_path, code, trade_date, features_dict, label):
    """保存特征到数据库"""
    features_json = json.dumps({
        k: v for k, v in features_dict.items()
        if k not in ['code', 'trade_date', 'label_5d']
    })

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("""
            INSERT OR REPLACE INTO v39_feature_cache
            (code, trade_date, features_json, label_5d)
            VALUES (?, ?, ?, ?)
        """, (code, trade_date, features_json, label))
        conn.commit()
        return True
    except Exception as e:
        print(f"保存失败 {code} {trade_date}: {e}")
        return False
    finally:
        conn.close()


def fill_gap(start_date='2023-01-01', end_date='2024-12-31',
             stock_limit=1000, batch_commit=100):
    """填补数据缺口"""
    db_path = 'data_adapter/stock_data.db'

    print("=" * 80)
    print("V3.9数据缺口填补 - 简化版")
    print("=" * 80)

    # 初始化
    print("\n[1/5] 初始化特征提取器...")
    system = V390EnhancedFeatureMLSystem()

    # 获取交易日
    print(f"\n[2/5] 获取交易日 ({start_date} ~ {end_date})...")
    dates = get_trading_dates(db_path, start_date, end_date)
    print(f"  找到 {len(dates)} 个交易日")

    # 获取股票列表
    print(f"\n[3/5] 获取股票列表 (限制{stock_limit}只)...")
    stocks = get_stock_list(db_path, stock_limit)
    print(f"  找到 {len(stocks)} 只股票")

    # 统计
    total_tasks = len(dates) * len(stocks)
    print(f"\n[4/5] 预计任务: {total_tasks:,} (已缓存的将跳过)")

    # 开始处理
    print("\n[5/5] 开始填补数据...")
    start_time = datetime.now()

    saved = 0
    skipped = 0
    failed = 0

    with tqdm(total=total_tasks, desc="处理进度") as pbar:
        for date in dates:
            for code in stocks:
                # 检查是否已存在
                if check_existing(db_path, code, date):
                    skipped += 1
                    pbar.update(1)
                    continue

                try:
                    # 提取特征
                    features = system.extract_features(code, date)
                    if features is None or len(features) == 0:
                        failed += 1
                        pbar.update(1)
                        continue

                    # 计算标签
                    label = system.calculate_label(code, date, lookahead_days=5)
                    if label is None:
                        failed += 1
                        pbar.update(1)
                        continue

                    # 保存
                    feature_dict = features.iloc[0].to_dict()
                    if save_features(db_path, code, date, feature_dict, label):
                        saved += 1
                    else:
                        failed += 1

                except Exception as e:
                    failed += 1

                # 更新进度
                pbar.update(1)
                pbar.set_postfix({
                    '新增': saved,
                    '跳过': skipped,
                    '失败': failed,
                    '速度': f'{saved/(datetime.now()-start_time).total_seconds():.1f}/s'
                })

    # 统计
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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='填补V3.9数据缺口')
    parser.add_argument('--start', type=str, default='2023-01-01',
                       help='开始日期')
    parser.add_argument('--end', type=str, default='2024-12-31',
                       help='结束日期')
    parser.add_argument('--stocks', type=int, default=1000,
                       help='股票数量')

    args = parser.parse_args()

    fill_gap(
        start_date=args.start,
        end_date=args.end,
        stock_limit=args.stocks
    )
