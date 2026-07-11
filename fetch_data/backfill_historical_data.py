#!/usr/bin/env python3
"""
历史数据修补工具 (合并版)

合并自:
- backfill_daily_basic.py   (UPDATE NULL字段，COALESCE保护已有数据)
- backfill_index_data.py    (INSERT缺失指数日线数据)

用法:
  python3 fetch_data/backfill_historical_data.py --mode daily-basic
  python3 fetch_data/backfill_historical_data.py --mode index-data
  python3 fetch_data/backfill_historical_data.py --mode all
"""

import os
import sys
import json
import time
import sqlite3
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
DEFAULT_DB = os.path.join(PROJECT_ROOT, 'data_adapter', 'stock_data.db')
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, 'config.json')

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


def _init_tushare(config_path: str = None):
    """初始化Tushare API (token 优先从 core.config/.env 获取)"""
    import tushare as ts
    try:
        from core.config import get_tushare_token
        token = get_tushare_token()
    except ImportError:
        config_path = config_path or DEFAULT_CONFIG
        with open(config_path, 'r') as f:
            config = json.load(f)
        token = config['tushare']['token']
    ts.set_token(token)
    return ts.pro_api()


def _get_security_map(db):
    """获取 ts_code -> security_id 映射"""
    cur = db.cursor()
    # 优先匹配带交易所后缀的code (如 000001.SZ)
    cur.execute("SELECT code, id FROM securities WHERE code LIKE '%.%'")
    result = {row[0]: row[1] for row in cur.fetchall()}

    # 补充不带后缀的code (旧格式)
    cur.execute("SELECT code, id, exchange FROM securities WHERE code NOT LIKE '%.%'")
    for code, sid, exchange in cur.fetchall():
        if exchange:
            result[f"{code}.{exchange}"] = sid
        else:
            prefix = code[:1]
            if prefix in ('6',):
                result[f"{code}.SH"] = sid
            elif prefix in ('0', '3'):
                result[f"{code}.SZ"] = sid
            elif prefix in ('4', '8'):
                result[f"{code}.BJ"] = sid

    return result


# ========== daily-basic: 补全NULL字段 ==========

def backfill_daily_basic(db_path: str = None, config_path: str = None):
    """补全 daily_basic 表中 turnover_rate 等 NULL 字段"""
    db_path = db_path or DEFAULT_DB
    pro = _init_tushare(config_path)
    db = sqlite3.connect(db_path)

    # 获取 turnover_rate 为 NULL 的所有交易日
    cur = db.cursor()
    cur.execute("""
        SELECT DISTINCT trade_date FROM daily_basic
        WHERE turnover_rate IS NULL
        ORDER BY trade_date
    """)
    null_dates = [r[0] for r in cur.fetchall()]

    security_map = _get_security_map(db)

    total = len(null_dates)
    logger.info(f"需要补全 {total} 个交易日的 daily_basic 数据")

    if total == 0:
        logger.info("无需补全")
        db.close()
        return

    updated_total = 0
    errors = 0

    for i, date_str in enumerate(null_dates):
        api_date = date_str.replace('-', '')

        try:
            df = pro.daily_basic(trade_date=api_date)
            time.sleep(0.22)

            if df is None or df.empty:
                logger.warning(f"[{i + 1}/{total}] {date_str}: 无数据")
                continue

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
                logger.info(f"[{i + 1}/{total}] {date_str}: 更新 {updated} 条, 累计 {updated_total:,}")

        except Exception as e:
            errors += 1
            logger.error(f"[{i + 1}/{total}] {date_str}: 错误 {e}")
            if errors > 10:
                logger.error("错误次数过多，暂停30秒")
                time.sleep(30)
                errors = 0

    # 最终统计
    cur.execute("""
        SELECT COUNT(DISTINCT trade_date) FROM daily_basic
        WHERE turnover_rate IS NULL
    """)
    remaining = cur.fetchone()[0]
    logger.info(f"\n完成! 累计更新 {updated_total:,} 条, 剩余 NULL 日期: {remaining}")
    db.close()


# ========== index-data: 补全大盘指数历史数据 ==========

def backfill_index_data(db_path: str = None, config_path: str = None):
    """补全大盘指数历史数据"""
    db_path = db_path or DEFAULT_DB
    pro = _init_tushare(config_path)
    db = sqlite3.connect(db_path)
    cur = db.cursor()

    # 获取指数 security_id 映射
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

        cur.execute("""
            SELECT MIN(trade_date), MAX(trade_date), COUNT(*)
            FROM daily_quotes WHERE security_id = ?
        """, (sid,))
        r = cur.fetchone()
        logger.info(f"\n{ts_code} {name}: 当前 {r[2]} 条 ({r[0]} ~ {r[1]})")

        # 获取已有日期
        cur.execute("SELECT trade_date FROM daily_quotes WHERE security_id = ?", (sid,))
        existing_dates = set(r[0] for r in cur.fetchall())

        # 分段获取
        segments = [
            ('20180101', '20191231'),
            ('20200101', '20211231'),
            ('20220101', '20231231'),
            ('20240101', '20261231'),
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
                    continue

                rows_to_insert = []
                for _, row in df.iterrows():
                    trade_date_raw = row['trade_date']
                    trade_date = f"{trade_date_raw[:4]}-{trade_date_raw[4:6]}-{trade_date_raw[6:8]}"

                    if trade_date in existing_dates:
                        continue

                    # pct_chg from Tushare is percentage (9.5=9.5%), convert to decimal (0.095)
                    raw_pct = row.get('pct_chg')
                    pct_decimal = raw_pct / 100.0 if pd.notna(raw_pct) else None

                    rows_to_insert.append((
                        sid, trade_date,
                        row.get('open'), row.get('high'), row.get('low'), row.get('close'),
                        row.get('vol'), row.get('amount'),
                        row.get('change'), pct_decimal,
                        0, 0, 0, 0
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

        cur.execute("SELECT COUNT(*) FROM daily_quotes WHERE security_id = ?", (sid,))
        new_count = cur.fetchone()[0]
        logger.info(f"  修复后: {new_count} 条")

    logger.info(f"\n总计插入: {total_inserted} 条")
    db.close()


def main():
    parser = argparse.ArgumentParser(
        description='历史数据修补工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 补全 daily_basic NULL字段
  python3 fetch_data/backfill_historical_data.py --mode daily-basic

  # 补全大盘指数历史数据
  python3 fetch_data/backfill_historical_data.py --mode index-data

  # 全部补全
  python3 fetch_data/backfill_historical_data.py --mode all
        """
    )

    parser.add_argument('--mode', type=str, choices=['daily-basic', 'index-data', 'all'],
                        required=True, help='修补模式')
    parser.add_argument('--db', type=str, default=None, help='数据库路径')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')

    args = parser.parse_args()

    if args.mode in ('daily-basic', 'all'):
        logger.info("=" * 60)
        logger.info("补全 daily_basic NULL字段")
        logger.info("=" * 60)
        backfill_daily_basic(db_path=args.db, config_path=args.config)

    if args.mode in ('index-data', 'all'):
        logger.info("\n" + "=" * 60)
        logger.info("补全大盘指数历史数据")
        logger.info("=" * 60)
        backfill_index_data(db_path=args.db, config_path=args.config)


if __name__ == '__main__':
    main()
