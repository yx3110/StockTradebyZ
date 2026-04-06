#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alpha158 特征缓存更新器

批量计算 Alpha158 特征并存入 SQLite 缓存表。
复用 v40_feature_cache_updater.py 的批量 SQL 加载模式。

用法:
    python3 fetch_data/alpha158_feature_cache_updater.py \
        --start-date 2020-01-02 --end-date 2026-02-13

    # 增量更新 (跳过已有日期)
    python3 fetch_data/alpha158_feature_cache_updater.py \
        --start-date 2025-01-01 --end-date 2026-02-13

    # 强制覆盖
    python3 fetch_data/alpha158_feature_cache_updater.py \
        --start-date 2025-01-01 --end-date 2026-02-13 --force
"""

import os
import sys
import json
import sqlite3
import logging
import argparse
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml_models.alpha158.alpha158_features import Alpha158FeatureCalculator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class Alpha158FeatureCacheUpdater:
    """Alpha158 特征缓存更新器"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')
        self.db_path = db_path

    def ensure_table(self):
        """创建 alpha158_feature_cache 表"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alpha158_feature_cache (
                code TEXT NOT NULL,
                trade_date DATE NOT NULL,
                features_json TEXT,
                label_3d REAL,
                label_5d REAL,
                label_10d REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(code, trade_date)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_alpha158_date
            ON alpha158_feature_cache(trade_date)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_alpha158_code_date
            ON alpha158_feature_cache(code, trade_date)
        """)
        conn.commit()
        conn.close()
        logger.info("alpha158_feature_cache 表已就绪")

    def get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取交易日列表 (从 daily_quotes 获取)"""
        conn = sqlite3.connect(self.db_path)
        dates = [r[0] for r in conn.execute("""
            SELECT DISTINCT trade_date FROM daily_quotes
            WHERE trade_date >= ? AND trade_date <= ?
            ORDER BY trade_date
        """, (start_date, end_date)).fetchall()]
        conn.close()
        return dates

    def get_existing_dates(self) -> set:
        """获取已缓存的日期集合"""
        conn = sqlite3.connect(self.db_path)
        try:
            dates = {r[0] for r in conn.execute(
                "SELECT DISTINCT trade_date FROM alpha158_feature_cache"
            ).fetchall()}
        except sqlite3.OperationalError:
            dates = set()
        conn.close()
        return dates

    def batch_load_stock_ohlcv(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """批量加载所有A股 OHLCV 数据

        Args:
            start_date: 开始日期 (含 lookback)
            end_date: 结束日期

        Returns:
            {code: DataFrame[trade_date, open, high, low, close, volume]}
        """
        logger.info(f"批量加载 OHLCV 数据: {start_date} ~ {end_date} ...")
        t0 = time.time()

        conn = sqlite3.connect(self.db_path)
        query = """
        SELECT s.code, q.trade_date, q.open, q.high, q.low, q.close, q.volume
        FROM daily_quotes q
        JOIN securities s ON q.security_id = s.id
        WHERE s.type = 'A股'
        AND q.trade_date >= ? AND q.trade_date <= ?
        AND q.volume > 0
        ORDER BY s.code, q.trade_date
        """
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()

        stock_data = {}
        for code, group in df.groupby('code'):
            stock_data[code] = group.reset_index(drop=True)

        elapsed = time.time() - t0
        logger.info(f"OHLCV 加载完成: {len(stock_data)} 只股票, "
                     f"{len(df)} 行, {elapsed:.1f}秒")
        return stock_data

    def compute_and_save(self, start_date: str, end_date: str, force: bool = False):
        """计算并保存 Alpha158 特征到缓存

        Args:
            start_date: 目标开始日期
            end_date: 目标结束日期
            force: 是否强制覆盖
        """
        self.ensure_table()

        # 获取交易日
        all_dates = self.get_trading_dates(start_date, end_date)
        if not all_dates:
            logger.warning(f"未找到 {start_date} ~ {end_date} 范围内的交易日")
            return

        # 过滤已有日期
        if not force:
            existing = self.get_existing_dates()
            target_dates = [d for d in all_dates if d not in existing]
            if existing:
                logger.info(f"已有 {len(existing)} 天缓存, 待计算 {len(target_dates)} 天")
        else:
            target_dates = all_dates

        if not target_dates:
            logger.info("所有日期已缓存, 无需计算")
            return

        logger.info(f"目标: {len(target_dates)} 天 ({target_dates[0]} ~ {target_dates[-1]})")

        # 加载 OHLCV (含 60 天 lookback + 10 天 forward for labels)
        lookback_start = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=100)
        forward_end = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=20)
        stock_data = self.batch_load_stock_ohlcv(
            lookback_start.strftime('%Y-%m-%d'),
            forward_end.strftime('%Y-%m-%d')
        )

        # 目标日期集合
        target_set = set(target_dates)

        # 按股票计算特征 (向量化批量版)
        total_records = 0
        total_stocks = len(stock_data)
        t_compute = time.time()
        batch_buffer = []
        BATCH_SIZE = 50000

        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        cursor = conn.cursor()

        insert_sql = """
        INSERT OR REPLACE INTO alpha158_feature_cache
        (code, trade_date, features_json, label_3d, label_5d, label_10d)
        VALUES (?, ?, ?, ?, ?, ?)
        """

        for i, (code, df) in enumerate(stock_data.items()):
            if len(df) < 61:
                continue

            # 批量计算所有日期的特征
            features_df = Alpha158FeatureCalculator.compute_features_batch(df)
            if features_df.empty:
                continue

            # 准备 labels (log return, 对齐回测: open[T+1] → close[T+1+N])
            closes = df['close'].values
            opens = df['open'].values
            volumes = df['volume'].values
            dates_arr = df['trade_date'].values
            date_to_idx = {d: idx for idx, d in enumerate(dates_arr)}

            feature_names = Alpha158FeatureCalculator.get_feature_names()

            for _, row in features_df.iterrows():
                td = str(row['trade_date'])[:10]
                if td not in target_set:
                    continue

                # 计算 labels (对齐回测: base=open[T+1], target=close[T+1+N])
                idx = date_to_idx.get(td)
                if idx is None:
                    continue

                label_3d = label_5d = label_10d = None
                buy_idx = idx + 1
                if buy_idx < len(opens) and opens[buy_idx] > 0 and volumes[buy_idx] > 0:
                    buy_open = opens[buy_idx]
                    if buy_idx + 3 < len(closes):
                        label_3d = float(np.log(closes[buy_idx + 3] / buy_open))
                    if buy_idx + 5 < len(closes):
                        label_5d = float(np.log(closes[buy_idx + 5] / buy_open))
                    if buy_idx + 10 < len(closes):
                        label_10d = float(np.log(closes[buy_idx + 10] / buy_open))

                # JSON 序列化特征
                feat_dict = {name: float(row[name]) for name in feature_names}
                features_json = json.dumps(feat_dict)

                batch_buffer.append((code, td, features_json, label_3d, label_5d, label_10d))

            # 批量写入
            if len(batch_buffer) >= BATCH_SIZE:
                cursor.executemany(insert_sql, batch_buffer)
                conn.commit()
                total_records += len(batch_buffer)
                batch_buffer = []

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t_compute
                rate = (i + 1) / elapsed
                eta = (total_stocks - i - 1) / rate
                buffered = total_records + len(batch_buffer)
                logger.info(f"  进度: {i+1}/{total_stocks} 只股票, "
                            f"{buffered} 条记录, "
                            f"速度: {rate:.1f}只/秒, ETA: {eta:.0f}秒")

        # 写入剩余
        if batch_buffer:
            cursor.executemany(insert_sql, batch_buffer)
            conn.commit()
            total_records += len(batch_buffer)

        conn.close()
        elapsed = time.time() - t_compute
        logger.info(f"Alpha158 特征计算完成: {total_records} 条记录, "
                     f"{total_stocks} 只股票, 耗时 {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")

    def backfill_labels(self):
        """回填缺失的标签 (用于最近 10 天的数据)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 找到缺失标签的记录
        cursor.execute("""
            SELECT DISTINCT c.code, c.trade_date
            FROM alpha158_feature_cache c
            WHERE c.label_10d IS NULL
            ORDER BY c.trade_date, c.code
        """)
        missing = cursor.fetchall()

        if not missing:
            logger.info("无需回填标签")
            conn.close()
            return

        logger.info(f"需要回填标签: {len(missing)} 条记录")

        updated = 0
        for code, trade_date in missing:
            # 查询该股票的后续行情 (open+close, 标签对齐回测执行)
            cursor.execute("""
                SELECT q.trade_date, q.open, q.close, q.volume
                FROM daily_quotes q
                JOIN securities s ON q.security_id = s.id
                WHERE s.code = ? AND q.trade_date >= ?
                ORDER BY q.trade_date
                LIMIT 13
            """, (code, trade_date))
            rows = cursor.fetchall()

            # 至少需要报告日 + 买入日 + 1天
            if len(rows) < 3:
                continue

            # 买入日 (T+1) 的开盘价作为 base
            buy_open = rows[1][1]  # open
            buy_volume = rows[1][3]  # volume
            if buy_open is None or buy_open <= 0:
                continue
            if buy_volume is not None and buy_volume == 0:
                continue

            label_3d = label_5d = label_10d = None
            if len(rows) > 1 + 3:
                label_3d = float(np.log(rows[1 + 3][2] / buy_open))  # [2]=close
            if len(rows) > 1 + 5:
                label_5d = float(np.log(rows[1 + 5][2] / buy_open))
            if len(rows) > 1 + 10:
                label_10d = float(np.log(rows[1 + 10][2] / buy_open))

            if label_3d is not None:
                cursor.execute("""
                    UPDATE alpha158_feature_cache
                    SET label_3d = ?, label_5d = ?, label_10d = ?
                    WHERE code = ? AND trade_date = ?
                """, (label_3d, label_5d, label_10d, code, trade_date))
                updated += 1

        conn.commit()
        conn.close()
        logger.info(f"回填完成: {updated}/{len(missing)} 条更新")

    def get_stats(self):
        """打印缓存统计信息"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM alpha158_feature_cache")
            total = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(DISTINCT trade_date) FROM alpha158_feature_cache")
            n_dates = cursor.fetchone()[0]
            cursor.execute("SELECT MIN(trade_date), MAX(trade_date) FROM alpha158_feature_cache")
            min_d, max_d = cursor.fetchone()
            cursor.execute("SELECT COUNT(*) FROM alpha158_feature_cache WHERE label_10d IS NOT NULL")
            n_labeled = cursor.fetchone()[0]
            print(f"\nalpha158_feature_cache 统计:")
            print(f"  总记录: {total:,}")
            print(f"  交易日: {n_dates}")
            print(f"  日期范围: {min_d} ~ {max_d}")
            print(f"  有标签: {n_labeled:,} ({n_labeled/total*100:.1f}%)" if total > 0 else "  有标签: 0")
        except sqlite3.OperationalError:
            print("alpha158_feature_cache 表不存在")
        finally:
            conn.close()


def main():
    parser = argparse.ArgumentParser(description='Alpha158 特征缓存更新器')
    parser.add_argument('--start-date', default='2020-01-02',
                        help='开始日期 (default: 2020-01-02)')
    parser.add_argument('--end-date', default=None,
                        help='结束日期 (default: today)')
    parser.add_argument('--force', action='store_true',
                        help='强制覆盖已有缓存')
    parser.add_argument('--backfill-labels', action='store_true',
                        help='仅回填缺失标签')
    parser.add_argument('--stats', action='store_true',
                        help='仅显示缓存统计')
    args = parser.parse_args()

    if args.end_date is None:
        from datetime import datetime
        args.end_date = datetime.now().strftime('%Y-%m-%d')

    updater = Alpha158FeatureCacheUpdater()

    if args.stats:
        updater.get_stats()
        return

    if args.backfill_labels:
        updater.backfill_labels()
        updater.get_stats()
        return

    print(f"Alpha158 特征缓存更新器")
    print(f"  日期范围: {args.start_date} ~ {args.end_date}")
    print(f"  强制覆盖: {args.force}")
    print()

    t0 = time.time()
    updater.compute_and_save(args.start_date, args.end_date, force=args.force)
    updater.backfill_labels()
    updater.get_stats()

    total = time.time() - t0
    print(f"\n总耗时: {total:.1f}秒 ({total/60:.1f}分钟)")


if __name__ == '__main__':
    main()
