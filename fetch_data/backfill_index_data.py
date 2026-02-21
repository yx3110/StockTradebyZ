#!/usr/bin/env python3
"""
补全大盘指数历史数据
从 Tushare index_daily 接口获取 daily_quotes 中缺失的指数数据
"""

import os
import sys
import json
import time
import sqlite3
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

import tushare as ts
ts.set_token(config['tushare']['token'])
pro = ts.pro_api()

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_adapter', 'stock_data.db')

# 10个重要指数
INDICES = {
    '000001.SH': '上证指数',
    '399001.SZ': '深证成指',
    '399006.SZ': '创业板指',
    '000688.SH': '科创50',
    '000016.SH': '上证50',
    '000300.SH': '沪深300',
    '000905.SH': '中证500',
    '000852.SH': '中证1000',
    '932000.CSI': '中证2000',
    '000985.SH': '中证全指',
}


def backfill_indices():
    db = sqlite3.connect(DB_PATH)
    cur = db.cursor()

    # 获取 security_id 映射
    sid_map = {}
    for ts_code in INDICES:
        cur.execute("SELECT id FROM securities WHERE code = ?", (ts_code,))
        r = cur.fetchone()
        if r:
            sid_map[ts_code] = r[0]
        else:
            logger.warning(f"{ts_code} 不在 securities 表中，跳过")

    total_inserted = 0

    for ts_code, name in INDICES.items():
        if ts_code not in sid_map:
            continue

        sid = sid_map[ts_code]

        # 查看当前覆盖情况
        cur.execute("""
            SELECT MIN(trade_date), MAX(trade_date), COUNT(*)
            FROM daily_quotes WHERE security_id = ?
        """, (sid,))
        r = cur.fetchone()
        logger.info(f"\n{ts_code} {name}: 当前 {r[2]} 条 ({r[0]} ~ {r[1]})")

        # 获取已有日期
        cur.execute("SELECT trade_date FROM daily_quotes WHERE security_id = ?", (sid,))
        existing_dates = set(r[0] for r in cur.fetchall())

        # 从 Tushare 获取完整历史 (2018-01-01 ~ 2025-12-31)
        # 分段获取避免单次返回过多
        segments = [
            ('20180101', '20191231'),
            ('20200101', '20211231'),
            ('20220101', '20231231'),
            ('20240101', '20251231'),
        ]

        inserted_for_index = 0

        for start, end in segments:
            try:
                df = pro.index_daily(
                    ts_code=ts_code,
                    start_date=start,
                    end_date=end,
                    fields='ts_code,trade_date,open,close,high,low,vol,amount,pct_chg,change'
                )
                time.sleep(0.3)

                if df is None or df.empty:
                    logger.info(f"  {start}~{end}: 无数据")
                    continue

                rows_to_insert = []
                for _, row in df.iterrows():
                    trade_date_raw = row['trade_date']
                    # 转为 YYYY-MM-DD
                    trade_date = f"{trade_date_raw[:4]}-{trade_date_raw[4:6]}-{trade_date_raw[6:8]}"

                    if trade_date in existing_dates:
                        continue

                    rows_to_insert.append((
                        sid, trade_date,
                        row.get('open'), row.get('high'), row.get('low'), row.get('close'),
                        row.get('vol'), row.get('amount'),
                        row.get('change'), row.get('pct_chg'),
                        0, 0, 0, 0  # is_limit_up, is_limit_down, is_st, is_suspend
                    ))

                if rows_to_insert:
                    cur.executemany("""
                        INSERT OR IGNORE INTO daily_quotes
                        (security_id, trade_date, open, high, low, close, volume, amount,
                         price_change, price_change_pct,
                         is_limit_up, is_limit_down, is_st, is_suspend)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, rows_to_insert)
                    db.commit()
                    inserted_for_index += len(rows_to_insert)
                    logger.info(f"  {start}~{end}: 插入 {len(rows_to_insert)} 条")

            except Exception as e:
                logger.error(f"  {start}~{end}: 错误 {e}")
                time.sleep(5)

        total_inserted += inserted_for_index
        logger.info(f"  {ts_code} 共插入 {inserted_for_index} 条")

        # 验证
        cur.execute("SELECT COUNT(*) FROM daily_quotes WHERE security_id = ?", (sid,))
        new_count = cur.fetchone()[0]
        logger.info(f"  修复后: {new_count} 条")

    logger.info(f"\n总计插入: {total_inserted} 条")
    db.close()


if __name__ == '__main__':
    backfill_indices()
