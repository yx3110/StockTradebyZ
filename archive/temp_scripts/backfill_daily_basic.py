#!/usr/bin/env python3
"""
补全 daily_basic 表中 turnover_rate 等 NULL 字段
从 Tushare API 获取 2020-01 ~ 2024-12 缺失的基本面数据
"""

import os
import sys
import json
import time
import sqlite3
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# 加载配置
config_path = os.path.join(PROJECT_ROOT, 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

import tushare as ts
ts.set_token(config['tushare']['token'])
pro = ts.pro_api()

DB_PATH = os.path.join(PROJECT_ROOT, 'data_adapter', 'stock_data.db')


def get_null_dates(db):
    """获取 turnover_rate 为 NULL 的所有交易日"""
    cur = db.cursor()
    cur.execute("""
        SELECT DISTINCT trade_date FROM daily_basic
        WHERE turnover_rate IS NULL
        ORDER BY trade_date
    """)
    return [r[0] for r in cur.fetchall()]


def get_security_map(db):
    """获取 ts_code -> security_id 映射 (Tushare返回000001.SZ格式, securities表存000001)"""
    cur = db.cursor()
    cur.execute("SELECT code, id, exchange FROM securities WHERE code NOT LIKE '%.%'")
    result = {}
    for code, sid, exchange in cur.fetchall():
        if exchange:
            result[f"{code}.{exchange}"] = sid
        else:
            # 从代码前缀推断交易所
            prefix = code[:1] if not code.endswith('_ETF') else ''
            if prefix in ('6',):
                result[f"{code}.SH"] = sid
            elif prefix in ('0', '3'):
                result[f"{code}.SZ"] = sid
            elif prefix in ('4', '8'):
                result[f"{code}.BJ"] = sid
    return result


def backfill_daily_basic():
    db = sqlite3.connect(DB_PATH)
    null_dates = get_null_dates(db)
    security_map = get_security_map(db)

    total = len(null_dates)
    logger.info(f"需要补全 {total} 个交易日的 daily_basic 数据")

    if total == 0:
        logger.info("无需补全")
        db.close()
        return

    updated_total = 0
    errors = 0

    for i, date_str in enumerate(null_dates):
        # 转为 YYYYMMDD
        api_date = date_str.replace('-', '')

        try:
            df = pro.daily_basic(trade_date=api_date)
            time.sleep(0.22)  # API 频率限制

            if df is None or df.empty:
                logger.warning(f"[{i+1}/{total}] {date_str}: 无数据")
                continue

            cur = db.cursor()
            updated = 0

            for _, row in df.iterrows():
                code = row.get('ts_code')
                if code not in security_map:
                    continue

                sid = security_map[code]

                cur.execute("""
                    UPDATE daily_basic SET
                        turnover_rate = COALESCE(?, turnover_rate),
                        turnover_rate_f = COALESCE(?, turnover_rate_f),
                        volume_ratio = COALESCE(?, volume_ratio),
                        pe = COALESCE(?, pe),
                        pe_ttm = COALESCE(?, pe_ttm),
                        pb = COALESCE(?, pb),
                        ps = COALESCE(?, ps),
                        ps_ttm = COALESCE(?, ps_ttm),
                        dv_ratio = COALESCE(?, dv_ratio),
                        dv_ttm = COALESCE(?, dv_ttm),
                        total_share = COALESCE(?, total_share),
                        float_share = COALESCE(?, float_share),
                        free_share = COALESCE(?, free_share),
                        total_mv = COALESCE(?, total_mv),
                        circ_mv = COALESCE(?, circ_mv)
                    WHERE security_id = ? AND trade_date = ?
                """, (
                    row.get('turnover_rate'), row.get('turnover_rate_f'),
                    row.get('volume_ratio'), row.get('pe'), row.get('pe_ttm'),
                    row.get('pb'), row.get('ps'), row.get('ps_ttm'),
                    row.get('dv_ratio'), row.get('dv_ttm'),
                    row.get('total_share'), row.get('float_share'),
                    row.get('free_share'), row.get('total_mv'), row.get('circ_mv'),
                    sid, date_str
                ))
                if cur.rowcount > 0:
                    updated += 1

            db.commit()
            updated_total += updated

            if (i + 1) % 20 == 0 or i == 0 or updated > 0:
                logger.info(f"[{i+1}/{total}] {date_str}: 更新 {updated} 条, 累计 {updated_total:,}")

        except Exception as e:
            errors += 1
            logger.error(f"[{i+1}/{total}] {date_str}: 错误 {e}")
            if errors > 10:
                logger.error("错误次数过多，暂停30秒")
                time.sleep(30)
                errors = 0
            continue

    # 最终统计
    remaining = len(get_null_dates(db))
    logger.info(f"\n完成! 累计更新 {updated_total:,} 条, 剩余 NULL 日期: {remaining}")
    db.close()


if __name__ == '__main__':
    backfill_daily_basic()
