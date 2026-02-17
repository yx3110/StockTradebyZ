#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算多周期标签 (10日、15日收益率)
为V3.91多周期模型准备训练数据
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_future_returns(args):
    """
    计算单只股票所有日期的10日和15日收益率

    Args:
        args: (code, dates_list)

    Returns:
        list of (code, date, label_10d, label_15d)
    """
    code, dates_list = args

    try:
        conn = sqlite3.connect('data_adapter/stock_data.db')

        # 获取该股票的所有价格数据
        query = """
        SELECT trade_date, close
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.code = ?
        ORDER BY trade_date
        """
        price_df = pd.read_sql_query(query, conn, params=(code,))
        conn.close()

        if price_df.empty:
            return []

        # 创建日期到价格的映射
        price_df['trade_date'] = pd.to_datetime(price_df['trade_date'])
        price_df = price_df.set_index('trade_date')

        results = []
        for date_str in dates_list:
            date = pd.to_datetime(date_str)

            # 获取当日及之后的价格
            future_prices = price_df[price_df.index >= date].head(16)  # 需要16天数据（当天+15天）

            if len(future_prices) < 16:
                continue

            start_price = future_prices.iloc[0]['close']
            if start_price <= 0:
                continue

            # 计算10日和15日收益率
            price_10d = future_prices.iloc[10]['close'] if len(future_prices) > 10 else None
            price_15d = future_prices.iloc[15]['close'] if len(future_prices) > 15 else None

            label_10d = (price_10d - start_price) / start_price if price_10d else None
            label_15d = (price_15d - start_price) / start_price if price_15d else None

            results.append((code, date_str, label_10d, label_15d))

        return results

    except Exception as e:
        logger.error(f"处理{code}失败: {e}")
        return []


def main():
    logger.info("=" * 70)
    logger.info("🚀 V3.91 多周期标签计算")
    logger.info("=" * 70)

    conn = sqlite3.connect('data_adapter/stock_data.db')

    # 获取需要计算的样本
    query = """
    SELECT code, trade_date
    FROM v39_feature_cache
    WHERE label_5d IS NOT NULL
    AND (label_10d IS NULL OR label_15d IS NULL)
    ORDER BY code, trade_date
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        logger.info("✅ 所有样本已有多周期标签，无需计算")
        return

    logger.info(f"📊 需要计算的样本: {len(df):,} 条")

    # 按股票分组
    grouped = df.groupby('code')['trade_date'].apply(list).to_dict()
    logger.info(f"📊 涉及股票数: {len(grouped):,} 只")

    # 并行计算
    start_time = datetime.now()
    all_results = []

    num_workers = min(mp.cpu_count(), 8)
    logger.info(f"🔧 使用 {num_workers} 个进程并行计算...")

    tasks = list(grouped.items())
    completed = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(calculate_future_returns, task): task[0] for task in tasks}

        for future in as_completed(futures):
            code = futures[future]
            completed += 1

            try:
                results = future.result()
                all_results.extend(results)

                if completed % 500 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    speed = len(all_results) / elapsed if elapsed > 0 else 0
                    logger.info(f"  进度: {completed}/{len(tasks)} 股票 | {len(all_results):,} 条标签 | {speed:.0f}条/秒")

            except Exception as e:
                logger.error(f"处理{code}时出错: {e}")

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"\n✅ 计算完成: {len(all_results):,} 条标签, 耗时 {elapsed:.1f} 秒")

    # 批量更新数据库
    if all_results:
        logger.info("💾 更新数据库...")

        conn = sqlite3.connect('data_adapter/stock_data.db')
        cursor = conn.cursor()

        # 分批更新
        batch_size = 10000
        for i in range(0, len(all_results), batch_size):
            batch = all_results[i:i+batch_size]

            for code, date, label_10d, label_15d in batch:
                cursor.execute("""
                    UPDATE v39_feature_cache
                    SET label_10d = ?, label_15d = ?
                    WHERE code = ? AND trade_date = ?
                """, (label_10d, label_15d, code, date))

            conn.commit()
            logger.info(f"  已更新: {min(i+batch_size, len(all_results)):,}/{len(all_results):,}")

        conn.close()

    # 统计结果
    conn = sqlite3.connect('data_adapter/stock_data.db')
    stats = pd.read_sql_query("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN label_5d IS NOT NULL THEN 1 ELSE 0 END) as has_5d,
            SUM(CASE WHEN label_10d IS NOT NULL THEN 1 ELSE 0 END) as has_10d,
            SUM(CASE WHEN label_15d IS NOT NULL THEN 1 ELSE 0 END) as has_15d,
            SUM(CASE WHEN label_5d IS NOT NULL AND label_10d IS NOT NULL AND label_15d IS NOT NULL THEN 1 ELSE 0 END) as has_all
        FROM v39_feature_cache
    """, conn)
    conn.close()

    logger.info("\n" + "=" * 70)
    logger.info("📊 标签统计")
    logger.info("=" * 70)
    logger.info(f"  总样本数: {stats['total'].iloc[0]:,}")
    logger.info(f"  有5日标签: {stats['has_5d'].iloc[0]:,}")
    logger.info(f"  有10日标签: {stats['has_10d'].iloc[0]:,}")
    logger.info(f"  有15日标签: {stats['has_15d'].iloc[0]:,}")
    logger.info(f"  三周期完整: {stats['has_all'].iloc[0]:,}")


if __name__ == "__main__":
    main()
