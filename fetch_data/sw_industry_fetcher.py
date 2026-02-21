#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
申万2021行业分类数据抓取器
- 从Tushare获取申万一级行业分类 (SW2021)
- 存储到sw_industry表
- 月度运行，支持--force强制更新
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


class SWIndustryFetcher:
    """申万2021行业分类数据抓取器"""

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
        """确保sw_industry表存在"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sw_industry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL,
                ts_code TEXT NOT NULL,
                stock_name TEXT,
                l1_code TEXT NOT NULL,
                l1_name TEXT NOT NULL,
                in_date TEXT,
                out_date TEXT,
                is_new TEXT DEFAULT 'Y',
                src TEXT DEFAULT 'SW2021',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(code, l1_code)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sw_industry_code
            ON sw_industry(code)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sw_industry_l1
            ON sw_industry(l1_code, l1_name)
        """)
        conn.commit()
        conn.close()

    def fetch_l1_industries(self) -> pd.DataFrame:
        """获取申万一级行业列表"""
        logger.info("获取申万2021一级行业分类...")
        df = self.pro.index_classify(level='L1', src='SW2021')
        if df is not None and not df.empty:
            logger.info(f"获取到 {len(df)} 个一级行业")
        else:
            logger.warning("未获取到行业分类数据")
            df = pd.DataFrame()
        return df

    def fetch_all_members_paginated(self) -> pd.DataFrame:
        """
        分页获取所有申万行业成分股

        index_member_all API 每次最多返回3000条，需要分页获取全部数据。
        返回的列: l1_code, l1_name, ts_code, name, in_date, out_date, is_new
        """
        all_dfs = []
        offset = 0
        page_size = 3000

        while True:
            logger.info(f"获取行业成分股数据 (offset={offset})...")
            time.sleep(0.5)

            df = self.pro.index_member_all(offset=offset, limit=page_size)
            if df is None or df.empty:
                break

            all_dfs.append(df)
            logger.info(f"  获取 {len(df)} 条记录")

            if len(df) < page_size:
                break
            offset += page_size

        if not all_dfs:
            return pd.DataFrame()

        result = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"共获取 {len(result)} 条行业成分记录")
        return result

    def update_all(self, force: bool = False) -> int:
        """
        更新所有申万行业分类数据

        Args:
            force: 强制更新，忽略staleness检查

        Returns:
            更新的记录数
        """
        if not force and not self.is_stale():
            logger.info("申万行业数据未过期 (<30天)，跳过更新")
            return 0

        start_time = time.time()

        # 1. 分页获取所有行业成分股数据
        members = self.fetch_all_members_paginated()
        if members.empty:
            logger.error("无法获取行业成分数据，放弃更新")
            return 0

        # 2. 解析为记录列表
        all_records = []
        for _, m in members.iterrows():
            ts_code = m.get('ts_code', '')
            if not ts_code:
                continue
            code = ts_code.split('.')[0]
            is_new = m.get('is_new', 'Y')
            if pd.isna(is_new):
                is_new = 'Y' if pd.isna(m.get('out_date')) else 'N'

            all_records.append({
                'code': code,
                'ts_code': ts_code,
                'stock_name': m.get('name', ''),
                'l1_code': m.get('l1_code', ''),
                'l1_name': m.get('l1_name', ''),
                'in_date': m.get('in_date', ''),
                'out_date': m.get('out_date', '') if not pd.isna(m.get('out_date')) else '',
                'is_new': is_new,
            })

        # 统计行业分布
        from collections import Counter
        active = [r for r in all_records if r['is_new'] == 'Y']
        industry_counts = Counter(r['l1_name'] for r in active)
        logger.info(f"活跃成分股: {len(active)} 只, {len(industry_counts)} 个行业")

        # 3. 批量写入数据库
        if not all_records:
            logger.warning("未获取到任何行业成分数据")
            return 0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        sql = """
            INSERT OR REPLACE INTO sw_industry
            (code, ts_code, stock_name, l1_code, l1_name, in_date, out_date, is_new, src, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'SW2021', CURRENT_TIMESTAMP)
        """

        values = [
            (r['code'], r['ts_code'], r['stock_name'], r['l1_code'],
             r['l1_name'], r['in_date'], r['out_date'], r['is_new'])
            for r in all_records
        ]

        cursor.executemany(sql, values)
        conn.commit()
        conn.close()

        elapsed = time.time() - start_time
        active_count = sum(1 for r in all_records if r['is_new'] == 'Y')
        logger.info(f"申万行业数据更新完成: {len(all_records)} 条记录 "
                     f"(活跃 {active_count}), 耗时 {elapsed:.1f}秒")
        return len(all_records)

    def is_stale(self, max_age_days: int = 30) -> bool:
        """检查数据是否过期"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM sw_industry")
        count = cursor.fetchone()[0]
        if count == 0:
            conn.close()
            return True

        cursor.execute("SELECT MAX(updated_at) FROM sw_industry")
        row = cursor.fetchone()
        conn.close()

        if row is None or row[0] is None:
            return True

        try:
            last_update = datetime.strptime(row[0][:19], '%Y-%m-%d %H:%M:%S')
            age = datetime.now() - last_update
            is_old = age.days >= max_age_days
            if is_old:
                logger.info(f"申万行业数据已过期: 上次更新 {last_update.strftime('%Y-%m-%d')}, "
                            f"已 {age.days} 天")
            else:
                logger.info(f"申万行业数据有效: 上次更新 {last_update.strftime('%Y-%m-%d')}, "
                            f"{age.days} 天前")
            return is_old
        except (ValueError, TypeError):
            return True

    def get_industry_mapping(self) -> Dict[str, str]:
        """
        获取 code -> l1_name 映射 (仅活跃成分股)

        Returns:
            {code: l1_name} 字典
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT code, l1_name FROM sw_industry
            WHERE is_new = 'Y'
        """)
        mapping = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        return mapping

    def get_industry_code_mapping(self) -> Dict[str, str]:
        """
        获取 code -> l1_code 映射 (仅活跃成分股)

        Returns:
            {code: l1_code} 字典
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT code, l1_code FROM sw_industry
            WHERE is_new = 'Y'
        """)
        mapping = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        return mapping

    def get_l1_label_encoding(self) -> Dict[str, int]:
        """
        获取 l1_name -> integer label 编码

        Returns:
            {l1_name: label_int} 字典, 按名称排序 (0-30)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT l1_name FROM sw_industry
            WHERE is_new = 'Y'
            ORDER BY l1_name
        """)
        names = [row[0] for row in cursor.fetchall()]
        conn.close()
        return {name: i for i, name in enumerate(names)}


def main():
    parser = argparse.ArgumentParser(description='申万2021行业分类数据抓取')
    parser.add_argument('--force', action='store_true', help='强制更新（忽略30天过期检查）')
    parser.add_argument('--db-path', type=str, help='数据库路径')
    args = parser.parse_args()

    fetcher = SWIndustryFetcher(db_path=args.db_path)
    count = fetcher.update_all(force=args.force)

    if count > 0:
        # 显示统计
        mapping = fetcher.get_industry_mapping()
        encoding = fetcher.get_l1_label_encoding()
        print(f"\n活跃成分股: {len(mapping)} 只")
        print(f"一级行业数: {len(encoding)} 个")
        print("行业列表:")
        for name, label in sorted(encoding.items(), key=lambda x: x[1]):
            industry_count = sum(1 for v in mapping.values() if v == name)
            print(f"  [{label:2d}] {name}: {industry_count} 只")


if __name__ == '__main__':
    main()
