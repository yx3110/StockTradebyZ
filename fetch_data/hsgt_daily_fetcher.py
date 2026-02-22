#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
北向/南向资金日度数据抓取器
- 从Tushare获取沪深港通资金流向数据
- 存储到 hsgt_daily 表
- 每日运行，支持历史回填
"""

import os
import sys
import json
import sqlite3
import logging
import argparse
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import tushare as ts
import pandas as pd

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class HSGTDailyFetcher:
    """沪深港通资金流向数据抓取器"""

    def __init__(self, db_path: str = None, config_path: str = None):
        if db_path is None:
            db_path = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')
        self.db_path = db_path

        if config_path is None:
            config_path = str(PROJECT_ROOT / 'config.json')

        with open(config_path, 'r') as f:
            config = json.load(f)
        token = config['tushare']['token']
        ts.set_token(token)
        self.pro = ts.pro_api()

        self._ensure_table()

    def _ensure_table(self):
        """确保 hsgt_daily 表存在"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS hsgt_daily (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_date DATE NOT NULL UNIQUE,
                north_money DECIMAL(20,4),
                south_money DECIMAL(20,4),
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_hsgt_daily_date
            ON hsgt_daily(trade_date)
        """)
        conn.commit()
        conn.close()

    def fetch_single_date(self, date_str: str) -> int:
        """
        抓取单日沪深港通资金数据

        Args:
            date_str: 日期 YYYYMMDD

        Returns:
            插入的记录数 (0 or 1)
        """
        logger.info(f"抓取 {date_str} 的沪深港通资金数据...")

        try:
            df = self.pro.moneyflow_hsgt(trade_date=date_str)

            if df is None or df.empty:
                logger.info(f"{date_str}: 无数据 (可能非交易日)")
                return 0

            row = df.iloc[0]
            trade_date = pd.to_datetime(str(row['trade_date']),
                                        format='%Y%m%d').strftime('%Y-%m-%d')

            # north_money = 沪股通 + 深股通 净买入 (万元)
            hgt = float(row.get('hgt', 0) or 0)  # 沪股通净买入
            sgt = float(row.get('sgt', 0) or 0)  # 深股通净买入
            north_money = hgt + sgt

            # south_money = 港股通(沪) + 港股通(深) 净买入 (万元)
            north_hgt = float(row.get('north_money', 0) or 0)
            south_hgt = float(row.get('south_money', 0) or 0)

            # Use the more reliable combined fields if available
            if north_money == 0 and north_hgt != 0:
                north_money = north_hgt
            south_money = south_hgt

            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT OR REPLACE INTO hsgt_daily
                (trade_date, north_money, south_money, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (trade_date, north_money, south_money))
            conn.commit()
            conn.close()

            logger.info(f"{date_str}: 北向 {north_money/10000:.2f}亿, "
                        f"南向 {south_money/10000:.2f}亿")
            return 1

        except Exception as e:
            logger.warning(f"{date_str} 抓取失败: {e}")
            return 0

    def fetch_date_range(self, start_date: str, end_date: str) -> int:
        """
        批量抓取日期范围的资金数据

        Args:
            start_date: YYYYMMDD
            end_date: YYYYMMDD

        Returns:
            总插入记录数
        """
        # 使用日期范围一次性获取 (moneyflow_hsgt 支持 start_date/end_date)
        logger.info(f"批量抓取 {start_date} ~ {end_date} 的北向资金数据...")

        try:
            df = self.pro.moneyflow_hsgt(start_date=start_date, end_date=end_date)

            if df is None or df.empty:
                logger.info("无数据")
                return 0

            logger.info(f"获取到 {len(df)} 条记录")

            records = []
            for _, row in df.iterrows():
                trade_date = pd.to_datetime(str(row['trade_date']),
                                            format='%Y%m%d').strftime('%Y-%m-%d')
                hgt = float(row.get('hgt', 0) or 0)
                sgt = float(row.get('sgt', 0) or 0)
                north_money = hgt + sgt

                south_hgt = float(row.get('south_money', 0) or 0)

                north_hgt_field = float(row.get('north_money', 0) or 0)
                if north_money == 0 and north_hgt_field != 0:
                    north_money = north_hgt_field

                records.append((trade_date, north_money, south_hgt))

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT OR REPLACE INTO hsgt_daily
                (trade_date, north_money, south_money, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, records)
            conn.commit()
            conn.close()

            logger.info(f"写入 {len(records)} 条记录")
            return len(records)

        except Exception as e:
            logger.error(f"批量抓取失败: {e}")
            # Fallback to single-date fetching
            logger.info("回退到逐日抓取模式...")
            conn = sqlite3.connect(self.db_path)
            start_dash = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
            end_dash = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
            dates_df = pd.read_sql_query("""
                SELECT DISTINCT trade_date FROM daily_quotes
                WHERE trade_date >= ? AND trade_date <= ?
                ORDER BY trade_date
            """, conn, params=(start_dash, end_dash))
            conn.close()

            total = 0
            for d in dates_df['trade_date'].tolist():
                d_compact = d.replace('-', '')
                time.sleep(0.5)
                total += self.fetch_single_date(d_compact)

            return total

    def get_latest_date(self) -> Optional[str]:
        """获取表中最新的交易日期"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(trade_date) FROM hsgt_daily")
        row = cursor.fetchone()
        conn.close()
        return row[0] if row and row[0] else None


def main():
    parser = argparse.ArgumentParser(description='沪深港通资金流向数据抓取')
    parser.add_argument('--date', type=str, help='单日抓取 (YYYYMMDD)')
    parser.add_argument('--start-date', type=str, help='批量回填开始日期 (YYYYMMDD)')
    parser.add_argument('--end-date', type=str, help='批量回填结束日期 (YYYYMMDD)')
    parser.add_argument('--db-path', type=str, help='数据库路径')
    args = parser.parse_args()

    fetcher = HSGTDailyFetcher(db_path=args.db_path)

    if args.date:
        count = fetcher.fetch_single_date(args.date)
        print(f"写入 {count} 条记录")
    elif args.start_date and args.end_date:
        count = fetcher.fetch_date_range(args.start_date, args.end_date)
        print(f"总计写入 {count} 条记录")
    else:
        today = datetime.now().strftime('%Y%m%d')
        count = fetcher.fetch_single_date(today)
        print(f"写入 {count} 条记录")


if __name__ == '__main__':
    main()
