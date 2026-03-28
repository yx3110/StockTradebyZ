#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量计算 BRAIN 验证因子并写入 brain_alpha_cache 表

用法:
    # 缓存最近1年 (推荐日常)
    python3 wqbrain_integration/cache_brain_features.py --start-date 2025-01-01

    # 缓存全量 (训练用, 约30分钟)
    python3 wqbrain_integration/cache_brain_features.py --start-date 2020-01-01

    # 指定日期范围
    python3 wqbrain_integration/cache_brain_features.py --start-date 2024-01-01 --end-date 2026-03-20
"""

import sys
import json
import sqlite3
import logging
import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from core.config import get_db_path
    DB_PATH = str(get_db_path())
except ImportError:
    DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')

from wqbrain_integration.validated_alphas import compute_brain_features, BRAIN_FEATURE_COLS

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def ensure_table(db_path: str):
    """创建 brain_alpha_cache 表"""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS brain_alpha_cache (
            code TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            features_json TEXT,
            PRIMARY KEY (code, trade_date)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_brain_alpha_cache_date
        ON brain_alpha_cache(trade_date)
    """)
    conn.commit()
    conn.close()


def get_trade_dates(db_path: str, start_date: str, end_date: str):
    """获取交易日列表"""
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("""
        SELECT DISTINCT trade_date FROM daily_quotes
        WHERE trade_date >= ? AND trade_date <= ?
        ORDER BY trade_date
    """, (start_date, end_date))
    dates = [row[0] for row in cursor.fetchall()]
    conn.close()
    return dates


def get_cached_dates(db_path: str):
    """获取已缓存的日期"""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute("SELECT DISTINCT trade_date FROM brain_alpha_cache")
        dates = set(row[0] for row in cursor.fetchall())
    except Exception:
        dates = set()
    conn.close()
    return dates


def _normalize_date(d: str) -> str:
    """归一化日期格式: YYYYMMDD → YYYY-MM-DD, 已有横杠则不变"""
    d = d.strip()
    if len(d) == 8 and '-' not in d:
        return f'{d[:4]}-{d[4:6]}-{d[6:8]}'
    return d


def batch_compute(db_path: str, start_date: str, end_date: str,
                  force: bool = False, batch_days: int = 5):
    """
    批量计算 BRAIN 因子

    策略: 一次性加载所有数据到内存, 按股票计算特征, 按日期写入缓存
    比逐日加载快 10x+
    """
    start_date = _normalize_date(start_date)
    end_date = _normalize_date(end_date)
    ensure_table(db_path)

    # 获取待计算的日期
    all_dates = get_trade_dates(db_path, start_date, end_date)
    if not force:
        cached = get_cached_dates(db_path)
        dates_to_compute = [d for d in all_dates if d not in cached]
        logger.info(f"交易日: {len(all_dates)}, 已缓存: {len(cached)}, 待计算: {len(dates_to_compute)}")
    else:
        dates_to_compute = all_dates
        logger.info(f"强制重算: {len(dates_to_compute)} 个交易日")

    if not dates_to_compute:
        logger.info("无需计算, 缓存已完整")
        return 0

    # 加载数据时需要多 30 天的 lookback
    lookback_start = pd.Timestamp(dates_to_compute[0]) - pd.Timedelta(days=45)
    data_start = lookback_start.strftime('%Y-%m-%d')

    logger.info(f"加载股票数据 ({data_start} ~ {end_date})...")
    t0 = time.time()

    conn = sqlite3.connect(db_path)
    query = """
        SELECT s.code, q.trade_date, q.open, q.high, q.low, q.close, q.volume, q.price_change_pct
        FROM daily_quotes q
        JOIN securities s ON q.security_id = s.id
        WHERE s.type = 'A股'
        AND q.trade_date >= ? AND q.trade_date <= ?
        AND q.volume > 0
        ORDER BY s.code, q.trade_date
    """
    df_all = pd.read_sql_query(query, conn, params=(data_start, end_date))
    conn.close()

    logger.info(f"加载完成: {len(df_all):,} 条, {df_all['code'].nunique()} 只股票, "
                f"耗时 {time.time()-t0:.1f}s")

    # 按股票分组计算特征
    logger.info("计算 BRAIN 特征...")
    t0 = time.time()

    dates_set = set(dates_to_compute)
    all_rows = []  # [{code, trade_date, features_json}, ...]

    stock_groups = df_all.groupby('code')
    total_stocks = len(stock_groups)

    for i, (code, stock_df) in enumerate(stock_groups):
        if len(stock_df) < 25:
            continue

        stock_df = stock_df.reset_index(drop=True)

        # 计算特征
        features = compute_brain_features(stock_df)

        # 收集需要缓存的日期
        for idx, row in stock_df.iterrows():
            if row['trade_date'] not in dates_set:
                continue

            feat_dict = {}
            for col in BRAIN_FEATURE_COLS:
                val = features.loc[idx, col] if idx in features.index else np.nan
                feat_dict[col] = round(float(val), 8) if not pd.isna(val) else 0.0

            all_rows.append({
                'code': code,
                'trade_date': row['trade_date'],
                'features_json': json.dumps(feat_dict),
            })

        if (i + 1) % 1000 == 0:
            logger.info(f"  进度: {i+1}/{total_stocks} 只股票, 累计 {len(all_rows):,} 条")

    calc_time = time.time() - t0
    logger.info(f"计算完成: {len(all_rows):,} 条, 耗时 {calc_time:.1f}s "
                f"({len(all_rows)/calc_time:.0f} 条/秒)")

    # 批量写入
    logger.info("写入 brain_alpha_cache...")
    t0 = time.time()

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")

    for i in range(0, len(all_rows), 10000):
        batch = all_rows[i:i+10000]
        conn.executemany(
            "INSERT OR REPLACE INTO brain_alpha_cache (code, trade_date, features_json) VALUES (?, ?, ?)",
            [(r['code'], r['trade_date'], r['features_json']) for r in batch]
        )
        conn.commit()

    conn.close()
    logger.info(f"写入完成: {len(all_rows):,} 条, 耗时 {time.time()-t0:.1f}s")

    return len(all_rows)


def main():
    parser = argparse.ArgumentParser(description='缓存 BRAIN 验证因子')
    parser.add_argument('--start-date', default='2020-01-01', help='开始日期')
    parser.add_argument('--end-date', default='2026-12-31', help='结束日期')
    parser.add_argument('--force', action='store_true', help='强制重算所有日期')
    parser.add_argument('--db-path', default=DB_PATH, help='数据库路径')
    args = parser.parse_args()

    total = batch_compute(args.db_path, args.start_date, args.end_date, args.force)
    print(f"\n完成: {total:,} 条记录已缓存")


if __name__ == '__main__':
    main()
