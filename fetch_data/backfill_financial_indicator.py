#!/usr/bin/env python3
"""
回填 financial_indicator 表 — 从Tushare批量拉取季报数据

目标: 2018-2025年所有A股的财务指标, 约5600股 × 32季度 ≈ 150K+条记录
当前: 仅37,776条(2024-2025为主), 2018-2022几乎为空

用法:
  python3 fetch_data/backfill_financial_indicator.py                    # 全量回填2018-2025
  python3 fetch_data/backfill_financial_indicator.py --start 20200101   # 指定起始
  python3 fetch_data/backfill_financial_indicator.py --by-period        # 按季度批量(更快)
"""

import os
import sys
import json
import time
import sqlite3
import logging
import argparse
import numpy as np
import pandas as pd
import tushare as ts
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / 'data_adapter' / 'stock_data.db'

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Tushare fina_indicator 核心字段 (V4.7.1需要的 + 其他重要字段)
FIELDS = (
    'ts_code,ann_date,end_date,'
    # V4.7.1 必需6字段 (gross_margin在Tushare叫grossprofit_margin, DB列叫gross_margin)
    'roe,grossprofit_margin,current_ratio,assets_turn,netprofit_yoy,or_yoy,'
    # 其他重要字段 (与现有INSERT兼容)
    'eps,dt_eps,roe_waa,roe_dt,roa,'
    'netprofit_margin,profit_to_gr,ocf_to_profit,'
    'debt_to_assets,quick_ratio,ar_turn,ca_turn,fa_turn,'
    # 额外有用字段
    'op_yoy,ebt_yoy,basic_eps_yoy,bps,cfps,ocfps'
)

# Tushare字段名 → DB列名映射 (大部分相同, 少数需要转换)
TUSHARE_TO_DB = {
    'grossprofit_margin': 'gross_margin',
}

# DB列名列表 (用于INSERT)
DB_COLUMNS = [
    'security_id', 'ann_date', 'end_date',
    'roe', 'gross_margin', 'current_ratio', 'assets_turn', 'netprofit_yoy', 'or_yoy',
    'eps', 'dt_eps', 'roe_waa', 'roe_dt', 'roa',
    'netprofit_margin', 'profit_to_gr', 'ocf_to_profit',
    'debt_to_assets', 'quick_ratio', 'ar_turn', 'ca_turn', 'fa_turn',
    'op_yoy', 'ebt_yoy', 'basic_eps_yoy', 'bps', 'cfps', 'ocfps'
]

# Tushare API字段名列表 (与DB_COLUMNS一一对应, 跳过security_id)
TUSHARE_API_COLUMNS = [
    'ann_date', 'end_date',
    'roe', 'grossprofit_margin', 'current_ratio', 'assets_turn', 'netprofit_yoy', 'or_yoy',
    'eps', 'dt_eps', 'roe_waa', 'roe_dt', 'roa',
    'netprofit_margin', 'profit_to_gr', 'ocf_to_profit',
    'debt_to_assets', 'quick_ratio', 'ar_turn', 'ca_turn', 'fa_turn',
    'op_yoy', 'ebt_yoy', 'basic_eps_yoy', 'bps', 'cfps', 'ocfps'
]


def load_config():
    config_path = PROJECT_ROOT / 'config.json'
    with open(config_path) as f:
        return json.load(f)


def get_pro():
    config = load_config()
    token = config.get('tushare', {}).get('token', '')
    ts.set_token(token)
    return ts.pro_api()


def get_security_id_map(db_path):
    """获取 ts_code → security_id 映射 (DB存'000001', Tushare用'000001.SZ')"""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT id, code, exchange FROM securities WHERE type = 'A股'").fetchall()
    conn.close()
    result = {}
    for sec_id, code, exchange in rows:
        ts_code = f"{code}.{exchange}" if exchange else code
        result[ts_code] = sec_id
    return result


def get_stock_list(db_path):
    """获取所有A股ts_code列表 (格式: 000001.SZ)"""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT code, exchange FROM securities WHERE type = 'A股' ORDER BY code").fetchall()
    conn.close()
    return [f"{code}.{exchange}" if exchange else code for code, exchange in rows]


def get_existing_keys(db_path):
    """获取已存在的 (security_id, end_date) 集合, 用于跳过"""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT security_id, end_date FROM financial_indicator").fetchall()
    conn.close()
    return set(rows)


def fmt_date(d):
    """YYYYMMDD → YYYY-MM-DD"""
    if d and len(str(d)) == 8:
        s = str(d)
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return d


def backfill_by_stock(pro, db_path, start_date='20180101', end_date='20251231', max_workers=4):
    """按股票逐个拉取 (可靠但较慢, ~20分钟)"""
    stock_list = get_stock_list(db_path)
    sec_map = get_security_id_map(db_path)
    existing = get_existing_keys(db_path)

    logger.info(f"回填 financial_indicator: {len(stock_list)} 只A股, {start_date}~{end_date}")
    logger.info(f"  已有记录: {len(existing)}, 并发: {max_workers}")

    stats = {'fetched': 0, 'inserted': 0, 'skipped': 0, 'errors': 0}
    batch_buffer = []
    BATCH_SIZE = 500

    def flush_buffer(buffer):
        if not buffer:
            return 0
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        inserted = 0
        for row in buffer:
            try:
                placeholders = ','.join(['?'] * len(DB_COLUMNS))
                cols = ','.join(DB_COLUMNS)
                cursor.execute(f"INSERT OR IGNORE INTO financial_indicator ({cols}) VALUES ({placeholders})", row)
                if cursor.rowcount > 0:
                    inserted += 1
            except Exception as e:
                logger.debug(f"INSERT失败: {e}")
        conn.commit()
        conn.close()
        return inserted

    def fetch_one(ts_code):
        local_rows = []
        try:
            time.sleep(0.35)  # Rate limit: ~2.8/sec (avoid timeout)
            df = pro.fina_indicator(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields=FIELDS
            )
            if df is None or df.empty:
                return local_rows

            sec_id = sec_map.get(ts_code)
            if not sec_id:
                return local_rows

            for _, row in df.iterrows():
                ann_date = fmt_date(row.get('ann_date'))
                end_dt = fmt_date(row.get('end_date'))

                if (sec_id, end_dt) in existing:
                    continue

                values = [sec_id, ann_date, end_dt]
                for api_col in TUSHARE_API_COLUMNS:
                    if api_col in ('ann_date', 'end_date'):
                        continue
                    val = row.get(api_col)
                    if pd.isna(val):
                        values.append(None)
                    else:
                        values.append(float(val))
                local_rows.append(tuple(values))

        except Exception as e:
            err_str = str(e).lower()
            if 'timeout' in err_str or 'timed out' in err_str:
                # 超时: 等5秒后重试一次
                time.sleep(5)
                try:
                    df = pro.fina_indicator(
                        ts_code=ts_code,
                        start_date=start_date,
                        end_date=end_date,
                        fields=FIELDS
                    )
                    if df is not None and not df.empty:
                        sec_id = sec_map.get(ts_code)
                        if sec_id:
                            for _, row in df.iterrows():
                                ann_date = fmt_date(row.get('ann_date'))
                                end_dt = fmt_date(row.get('end_date'))
                                if (sec_id, end_dt) in existing:
                                    continue
                                values = [sec_id, ann_date, end_dt]
                                for api_col in TUSHARE_API_COLUMNS:
                                    if api_col in ('ann_date', 'end_date'):
                                        continue
                                    val = row.get(api_col)
                                    values.append(None if pd.isna(val) else float(val))
                                local_rows.append(tuple(values))
                        return local_rows
                except:
                    pass
                stats['errors'] += 1
            elif 'freq' not in err_str:
                logger.warning(f"  {ts_code}: {e}")
                stats['errors'] += 1
        return local_rows

    # 单线程顺序执行(Tushare有频率限制, 多线程意义不大)
    for i, ts_code in enumerate(stock_list):
        rows = fetch_one(ts_code)
        stats['fetched'] += 1
        batch_buffer.extend(rows)

        if len(batch_buffer) >= BATCH_SIZE:
            n = flush_buffer(batch_buffer)
            stats['inserted'] += n
            stats['skipped'] += len(batch_buffer) - n
            batch_buffer = []

        if (i + 1) % 200 == 0:
            logger.info(f"  进度: {i+1}/{len(stock_list)} ({(i+1)/len(stock_list)*100:.1f}%), "
                        f"新增: {stats['inserted']}, 跳过: {stats['skipped']}, 错误: {stats['errors']}")

    # 最后一批
    n = flush_buffer(batch_buffer)
    stats['inserted'] += n
    stats['skipped'] += len(batch_buffer) - n

    logger.info(f"\n回填完成! 新增: {stats['inserted']}, 跳过: {stats['skipped']}, 错误: {stats['errors']}")
    return stats


def backfill_by_period(pro, db_path, start_date='20180101', end_date='20251231'):
    """按季度拉取 (更快, 每季度1次API调用)

    Tushare fina_indicator 支持 period 参数, 一次获取该季度所有股票的数据
    """
    sec_map = get_security_id_map(db_path)
    existing = get_existing_keys(db_path)

    # 生成季度末日期列表
    periods = []
    for year in range(int(start_date[:4]), int(end_date[:4]) + 1):
        for q_end in ['0331', '0630', '0930', '1231']:
            period = f"{year}{q_end}"
            if start_date <= period <= end_date:
                periods.append(period)

    logger.info(f"按季度回填: {len(periods)} 个季度, {periods[0]}~{periods[-1]}")
    logger.info(f"  股票映射: {len(sec_map)}, 已有记录: {len(existing)}")

    stats = {'inserted': 0, 'skipped': 0, 'errors': 0, 'api_calls': 0}

    for i, period in enumerate(periods):
        try:
            time.sleep(0.5)  # Rate limit
            stats['api_calls'] += 1

            df = pro.fina_indicator(
                period=period,
                fields=FIELDS
            )

            if df is None or df.empty:
                logger.info(f"  {period}: 无数据")
                continue

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            period_inserted = 0

            for _, row in df.iterrows():
                ts_code = row.get('ts_code')
                sec_id = sec_map.get(ts_code)
                if not sec_id:
                    continue

                ann_date = fmt_date(row.get('ann_date'))
                end_dt = fmt_date(row.get('end_date'))

                if (sec_id, end_dt) in existing:
                    stats['skipped'] += 1
                    continue

                values = [sec_id, ann_date, end_dt]
                for api_col in TUSHARE_API_COLUMNS:
                    if api_col in ('ann_date', 'end_date'):
                        continue
                    val = row.get(api_col)
                    if pd.isna(val):
                        values.append(None)
                    else:
                        values.append(float(val))

                try:
                    placeholders = ','.join(['?'] * len(DB_COLUMNS))
                    cols = ','.join(DB_COLUMNS)
                    cursor.execute(f"INSERT OR IGNORE INTO financial_indicator ({cols}) VALUES ({placeholders})", tuple(values))
                    if cursor.rowcount > 0:
                        period_inserted += 1
                        existing.add((sec_id, end_dt))
                except Exception as e:
                    logger.debug(f"INSERT失败 {ts_code}: {e}")

            conn.commit()
            conn.close()
            stats['inserted'] += period_inserted

            logger.info(f"  [{i+1}/{len(periods)}] {period}: {len(df)} 条API返回, {period_inserted} 新增")

        except Exception as e:
            logger.warning(f"  {period}: API错误 - {e}")
            stats['errors'] += 1
            time.sleep(2)  # 出错后多等一会

    logger.info(f"\n回填完成! 新增: {stats['inserted']}, 跳过: {stats['skipped']}, "
                f"错误: {stats['errors']}, API调用: {stats['api_calls']}")
    return stats


def update_missing_yoy(pro, db_path):
    """补充已有记录中缺失的 netprofit_yoy 和 or_yoy"""
    conn = sqlite3.connect(db_path)

    # 找出有记录但YoY为NULL的
    missing = conn.execute("""
        SELECT fi.id, s.code, fi.end_date
        FROM financial_indicator fi
        JOIN securities s ON fi.security_id = s.id
        WHERE fi.netprofit_yoy IS NULL OR fi.or_yoy IS NULL
        LIMIT 10000
    """).fetchall()

    if not missing:
        logger.info("所有记录已有 netprofit_yoy 和 or_yoy, 无需补充")
        conn.close()
        return

    logger.info(f"需要补充YoY的记录: {len(missing)}")

    # 按股票分组
    from collections import defaultdict
    stock_records = defaultdict(list)
    for fi_id, code, end_date in missing:
        stock_records[code].append((fi_id, end_date))

    updated = 0
    for code, records in stock_records.items():
        try:
            time.sleep(0.22)
            df = pro.fina_indicator(
                ts_code=code,
                fields='ts_code,end_date,netprofit_yoy,or_yoy'
            )
            if df is None or df.empty:
                continue

            for fi_id, end_date in records:
                end_date_raw = end_date.replace('-', '')
                match = df[df['end_date'] == end_date_raw]
                if len(match) > 0:
                    row = match.iloc[0]
                    npyoy = None if pd.isna(row.get('netprofit_yoy')) else float(row['netprofit_yoy'])
                    oryoy = None if pd.isna(row.get('or_yoy')) else float(row['or_yoy'])
                    conn.execute("""
                        UPDATE financial_indicator SET netprofit_yoy=?, or_yoy=?
                        WHERE id=?
                    """, (npyoy, oryoy, fi_id))
                    updated += 1
        except Exception as e:
            logger.debug(f"UPDATE失败 {code}: {e}")

    conn.commit()
    conn.close()
    logger.info(f"YoY字段补充完成: {updated} 条更新")


def verify_result(db_path):
    """验证回填结果"""
    conn = sqlite3.connect(db_path)

    total = conn.execute("SELECT COUNT(*) FROM financial_indicator").fetchone()[0]
    logger.info(f"\n{'='*50}")
    logger.info(f"financial_indicator 回填验证")
    logger.info(f"{'='*50}")
    logger.info(f"  总记录数: {total:,}")

    # 按年统计
    yearly = conn.execute("""
        SELECT substr(end_date,1,4) as yr, COUNT(*)
        FROM financial_indicator GROUP BY yr ORDER BY yr
    """).fetchall()
    for yr, cnt in yearly:
        logger.info(f"  {yr}: {cnt:,} 条")

    # V4.7.1字段完整性
    cols = ['roe', 'gross_margin', 'current_ratio', 'assets_turn', 'netprofit_yoy', 'or_yoy']
    for col in cols:
        non_null = conn.execute(f"SELECT COUNT({col}) FROM financial_indicator WHERE {col} IS NOT NULL").fetchone()[0]
        pct = non_null / total * 100 if total > 0 else 0
        logger.info(f"  {col}: {non_null:,}/{total:,} ({pct:.1f}%)")

    # 股票覆盖
    stock_count = conn.execute("""
        SELECT COUNT(DISTINCT security_id) FROM financial_indicator
    """).fetchone()[0]
    logger.info(f"  股票覆盖: {stock_count}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description='回填 financial_indicator 表')
    parser.add_argument('--start', default='20180101', help='开始日期 YYYYMMDD (default: 20180101)')
    parser.add_argument('--end', default='20251231', help='结束日期 YYYYMMDD (default: 20251231)')
    parser.add_argument('--by-period', action='store_true', help='按季度批量拉取(更快)')
    parser.add_argument('--by-stock', action='store_true', help='按股票逐个拉取(更可靠)')
    parser.add_argument('--update-yoy', action='store_true', help='仅补充已有记录的YoY字段')
    parser.add_argument('--verify', action='store_true', help='仅验证当前数据')
    parser.add_argument('--workers', type=int, default=1, help='并发数 (default: 1)')
    args = parser.parse_args()

    db_path = str(DB_PATH)

    if args.verify:
        verify_result(db_path)
        return

    pro = get_pro()

    if args.update_yoy:
        update_missing_yoy(pro, db_path)
        verify_result(db_path)
        return

    # 默认: 先尝试按季度, 再按股票补漏
    if args.by_period or (not args.by_stock):
        logger.info("=== 阶段1: 按季度批量拉取 ===")
        backfill_by_period(pro, db_path, args.start, args.end)

    if args.by_stock or (not args.by_period):
        logger.info("\n=== 阶段2: 按股票逐个补漏 ===")
        backfill_by_stock(pro, db_path, args.start, args.end, args.workers)

    verify_result(db_path)


if __name__ == '__main__':
    main()
