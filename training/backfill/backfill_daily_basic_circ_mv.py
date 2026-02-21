#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高效补充 daily_basic 表中 2020-2024 缺失的 circ_mv 和 turnover_rate 数据

策略：按日期批量获取，每天一次API调用获取所有股票数据
比逐股票查询快100倍以上

作者: Claude Code
创建时间: 2025-11-28
"""

import pandas as pd
import tushare as ts
import sqlite3
import json
import time
import logging
from datetime import datetime, timedelta
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# 配置
DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')
with open('config.json', 'r') as f:
    config = json.load(f)
ts.set_token(config['tushare']['token'])
pro = ts.pro_api()


def get_dates_needing_backfill():
    """获取需要补充数据的日期列表"""
    conn = sqlite3.connect(DB_PATH)

    # 查找2020-2024年circ_mv为空的日期
    query = """
    SELECT DISTINCT trade_date
    FROM daily_basic
    WHERE (circ_mv IS NULL OR circ_mv = '')
        AND trade_date >= '2020-01-01'
        AND trade_date < '2025-01-01'
    ORDER BY trade_date
    """

    df = pd.read_sql(query, conn)
    conn.close()

    return df['trade_date'].tolist()


def get_security_mapping():
    """获取股票代码到security_id的映射"""
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql("""
        SELECT id, code, exchange FROM securities WHERE type = 'A股'
    """, conn)

    conn.close()

    # 构建 ts_code -> security_id 映射
    mapping = {}
    for _, row in df.iterrows():
        ts_code = f"{row['code']}.{row['exchange']}"
        mapping[ts_code] = row['id']

    return mapping


def fetch_daily_basic_for_date(trade_date):
    """获取某一天的所有股票daily_basic数据"""
    # 转换日期格式 YYYY-MM-DD -> YYYYMMDD
    date_str = trade_date.replace('-', '')

    try:
        df = pro.daily_basic(
            trade_date=date_str,
            fields='ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,'
                   'pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,'
                   'total_share,float_share,free_share,total_mv,circ_mv'
        )
        return df
    except Exception as e:
        logger.error(f"获取 {trade_date} 数据失败: {e}")
        return None


def update_daily_basic(conn, security_id, trade_date, row):
    """更新单条daily_basic记录"""
    cursor = conn.cursor()

    try:
        cursor.execute("""
            UPDATE daily_basic SET
                close = COALESCE(?, close),
                turnover_rate = ?,
                turnover_rate_f = ?,
                volume_ratio = ?,
                pe = COALESCE(?, pe),
                pe_ttm = COALESCE(?, pe_ttm),
                pb = COALESCE(?, pb),
                ps = ?,
                ps_ttm = ?,
                dv_ratio = ?,
                dv_ttm = ?,
                total_share = ?,
                float_share = ?,
                free_share = ?,
                total_mv = ?,
                circ_mv = ?
            WHERE security_id = ? AND trade_date = ?
        """, (
            row.get('close'),
            row.get('turnover_rate'),
            row.get('turnover_rate_f'),
            row.get('volume_ratio'),
            row.get('pe'),
            row.get('pe_ttm'),
            row.get('pb'),
            row.get('ps'),
            row.get('ps_ttm'),
            row.get('dv_ratio'),
            row.get('dv_ttm'),
            row.get('total_share'),
            row.get('float_share'),
            row.get('free_share'),
            row.get('total_mv'),
            row.get('circ_mv'),
            security_id,
            trade_date
        ))
        return cursor.rowcount > 0
    except Exception as e:
        logger.debug(f"更新失败 ({security_id}, {trade_date}): {e}")
        return False


def backfill_circ_mv():
    """执行回填"""
    logger.info("=" * 70)
    logger.info("补充 2020-2024 年 daily_basic 的 circ_mv 和 turnover_rate 数据")
    logger.info("=" * 70)

    # 获取需要补充的日期
    dates = get_dates_needing_backfill()
    logger.info(f"需要补充的日期: {len(dates)} 天")

    if not dates:
        logger.info("没有需要补充的数据")
        return

    # 获取股票代码映射
    security_mapping = get_security_mapping()
    logger.info(f"股票代码映射: {len(security_mapping)} 只A股")

    conn = sqlite3.connect(DB_PATH)

    total_updated = 0
    total_dates_processed = 0

    for date in tqdm(dates, desc="补充数据"):
        try:
            # API速率限制 (约60次/分钟安全)
            time.sleep(1.0)

            # 获取当天所有股票数据
            df = fetch_daily_basic_for_date(date)

            if df is None or df.empty:
                continue

            # 更新数据库
            updated_count = 0
            for _, row in df.iterrows():
                ts_code = row['ts_code']
                security_id = security_mapping.get(ts_code)

                if security_id:
                    if update_daily_basic(conn, security_id, date, row):
                        updated_count += 1

            if updated_count > 0:
                conn.commit()
                total_updated += updated_count
                total_dates_processed += 1

            # 每50天报告一次进度
            if total_dates_processed % 50 == 0:
                logger.info(f"已处理 {total_dates_processed} 天, 更新 {total_updated} 条记录")

        except Exception as e:
            logger.error(f"处理 {date} 失败: {e}")
            continue

    conn.close()

    logger.info(f"\n" + "=" * 70)
    logger.info(f"补充完成!")
    logger.info(f"  处理日期: {total_dates_processed} 天")
    logger.info(f"  更新记录: {total_updated} 条")
    logger.info("=" * 70)


def verify_backfill():
    """验证补充结果"""
    conn = sqlite3.connect(DB_PATH)

    query = """
    SELECT
        substr(trade_date, 1, 4) as year,
        COUNT(*) as total,
        SUM(CASE WHEN circ_mv IS NOT NULL AND circ_mv != '' THEN 1 ELSE 0 END) as has_circ_mv,
        ROUND(100.0 * SUM(CASE WHEN circ_mv IS NOT NULL AND circ_mv != '' THEN 1 ELSE 0 END) / COUNT(*), 1) as pct
    FROM daily_basic
    GROUP BY substr(trade_date, 1, 4)
    ORDER BY year
    """

    df = pd.read_sql(query, conn)
    conn.close()

    logger.info("\n=== 数据完整度验证 ===")
    print(df.to_string(index=False))


if __name__ == '__main__':
    backfill_circ_mv()
    verify_backfill()
