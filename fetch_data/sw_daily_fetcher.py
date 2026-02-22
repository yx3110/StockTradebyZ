#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
申万行业指数日线数据抓取器
- 从Tushare获取31个申万一级行业指数的日线行情
- 存储到 sw_index_daily 表
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
from typing import Dict, List, Optional

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


class SWDailyFetcher:
    """申万行业指数日线数据抓取器"""

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
        """确保 sw_index_daily 表存在"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sw_index_daily (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                l1_code TEXT NOT NULL,
                trade_date DATE NOT NULL,
                open DECIMAL(10,4),
                high DECIMAL(10,4),
                low DECIMAL(10,4),
                close DECIMAL(10,4),
                volume DECIMAL(20,4),
                amount DECIMAL(20,4),
                pct_change DECIMAL(10,4),
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(l1_code, trade_date)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sw_index_daily_date
            ON sw_index_daily(trade_date)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sw_index_daily_code_date
            ON sw_index_daily(l1_code, trade_date)
        """)
        conn.commit()
        conn.close()

    def get_l1_codes(self) -> Dict[str, str]:
        """
        获取申万一级行业的指数代码列表

        Returns:
            {l1_code: l1_name} 映射, e.g. {'801010': '农林牧渔'}
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT l1_code, l1_name FROM sw_industry
            WHERE is_new = 'Y'
            ORDER BY l1_code
        """)
        result = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()

        if not result:
            logger.warning("sw_industry 表无数据，尝试从API获取行业列表...")
            try:
                df = self.pro.index_classify(level='L1', src='SW2021')
                if df is not None and not df.empty:
                    for _, row in df.iterrows():
                        code = row.get('index_code', '')
                        name = row.get('industry_name', '')
                        if code and name:
                            # l1_code format: 801010.SI -> 801010
                            l1_code = code.split('.')[0] if '.' in code else code
                            result[l1_code] = name
            except Exception as e:
                logger.error(f"从API获取行业列表失败: {e}")

        return result

    def fetch_single_date(self, date_str: str) -> int:
        """
        抓取单日所有行业指数数据

        Args:
            date_str: 日期 YYYYMMDD

        Returns:
            插入的记录数
        """
        l1_codes = self.get_l1_codes()
        if not l1_codes:
            logger.error("未获取到行业代码列表")
            return 0

        logger.info(f"抓取 {date_str} 的 {len(l1_codes)} 个行业指数数据...")

        records = []
        for l1_code, l1_name in l1_codes.items():
            try:
                # l1_code may already contain .SI suffix
                ts_code = l1_code if '.SI' in l1_code else f"{l1_code}.SI"
                time.sleep(0.3)  # API频率控制

                df = self.pro.sw_daily(
                    ts_code=ts_code,
                    trade_date=date_str,
                    fields='ts_code,trade_date,open,high,low,close,vol,amount,pct_change'
                )

                if df is not None and not df.empty:
                    row = df.iloc[0]
                    trade_date = pd.to_datetime(str(row['trade_date']),
                                                format='%Y%m%d').strftime('%Y-%m-%d')
                    # Store clean l1_code (with .SI for consistency)
                    store_code = l1_code if '.SI' in l1_code else f"{l1_code}.SI"
                    records.append((
                        store_code,
                        trade_date,
                        row.get('open'),
                        row.get('high'),
                        row.get('low'),
                        row.get('close'),
                        row.get('vol'),
                        row.get('amount'),
                        row.get('pct_change'),
                    ))
                else:
                    logger.debug(f"  {l1_name}({l1_code}): 无数据")

            except Exception as e:
                logger.warning(f"  {l1_name}({l1_code}) 抓取失败: {e}")
                continue

        if not records:
            logger.info(f"{date_str}: 无数据")
            return 0

        # 批量写入
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT OR REPLACE INTO sw_index_daily
            (l1_code, trade_date, open, high, low, close, volume, amount, pct_change, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, records)
        conn.commit()
        conn.close()

        logger.info(f"{date_str}: 写入 {len(records)} 条行业指数数据")
        return len(records)

    def fetch_date_range(self, start_date: str, end_date: str) -> int:
        """
        批量抓取日期范围内的行业指数数据

        Args:
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD

        Returns:
            总插入记录数
        """
        # 获取交易日列表
        conn = sqlite3.connect(self.db_path)
        start_dash = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
        end_dash = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
        df = pd.read_sql_query("""
            SELECT DISTINCT trade_date FROM daily_quotes
            WHERE trade_date >= ? AND trade_date <= ?
            ORDER BY trade_date
        """, conn, params=(start_dash, end_dash))
        conn.close()

        dates = df['trade_date'].tolist()
        logger.info(f"回填 {start_date} ~ {end_date}: {len(dates)} 个交易日")

        total = 0
        for i, d in enumerate(dates, 1):
            # Convert YYYY-MM-DD to YYYYMMDD for API
            d_compact = d.replace('-', '')
            logger.info(f"[{i}/{len(dates)}] {d}")
            count = self.fetch_single_date(d_compact)
            total += count

        logger.info(f"回填完成: 共 {total} 条记录")
        return total

    def get_latest_date(self) -> Optional[str]:
        """获取表中最新的交易日期"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(trade_date) FROM sw_index_daily")
        row = cursor.fetchone()
        conn.close()
        return row[0] if row and row[0] else None


def main():
    parser = argparse.ArgumentParser(description='申万行业指数日线数据抓取')
    parser.add_argument('--date', type=str, help='单日抓取 (YYYYMMDD)')
    parser.add_argument('--start-date', type=str, help='批量回填开始日期 (YYYYMMDD)')
    parser.add_argument('--end-date', type=str, help='批量回填结束日期 (YYYYMMDD)')
    parser.add_argument('--db-path', type=str, help='数据库路径')
    args = parser.parse_args()

    fetcher = SWDailyFetcher(db_path=args.db_path)

    if args.date:
        count = fetcher.fetch_single_date(args.date)
        print(f"写入 {count} 条记录")
    elif args.start_date and args.end_date:
        count = fetcher.fetch_date_range(args.start_date, args.end_date)
        print(f"总计写入 {count} 条记录")
    else:
        # 默认抓取今天
        today = datetime.now().strftime('%Y%m%d')
        count = fetcher.fetch_single_date(today)
        print(f"写入 {count} 条记录")


if __name__ == '__main__':
    main()
