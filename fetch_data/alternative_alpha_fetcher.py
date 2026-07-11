#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Alternative-alpha data fetchers: top_list / margin_detail / hsgt_top10.

Three Tushare APIs that survived the permission audit (2026-04-23):
- top_list       (龙虎榜)        ~50-90 rows/day     2018-至今
- margin_detail  (融资融券个股)   ~1000-4400 rows/day 2018-至今
- hsgt_top10     (北向十大活跃股) 20 rows/day         2018-至今

hk_hold (北向个股持股) was excluded — the upstream source has multi-year
data gaps (2023 all year blank, 2024-10 to 2026-03 blank), not a credit issue.
"""

import argparse
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import tushare as ts

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def _load_pro_api(config_path: Optional[str] = None):
    # token 优先从 core.config/.env 获取 (显式传入 config_path 时仍尊重旧路径)
    token = None
    if config_path is None:
        try:
            from core.config import get_tushare_token
            token = get_tushare_token()
        except ImportError:
            config_path = str(PROJECT_ROOT / "config.json")
    if token is None:
        with open(config_path) as f:
            token = json.load(f)["tushare"]["token"]
    ts.set_token(token)
    return ts.pro_api()


def _code6(ts_code: str) -> str:
    return ts_code.split(".")[0] if ts_code else ""


def _ymd_dash(ymd_compact: str) -> str:
    return f"{ymd_compact[:4]}-{ymd_compact[4:6]}-{ymd_compact[6:8]}"


class _BaseFetcher:
    """Shared batch / rate-limit / date-iteration logic."""

    TABLE_NAME: str = ""
    RATE_LIMIT_SEC: float = 0.35  # ~170 calls/min; comfortably under 500/min

    def __init__(self, db_path: Optional[str] = None, pro=None):
        self.db_path = db_path or str(PROJECT_ROOT / "data_adapter" / "stock_data.db")
        self.pro = pro or _load_pro_api()
        self._ensure_table()

    def _ensure_table(self):  # override
        raise NotImplementedError

    def fetch_single_date(self, date_compact: str) -> int:  # override
        raise NotImplementedError

    def get_latest_date(self) -> Optional[str]:
        conn = _connect(self.db_path)
        try:
            row = conn.execute(
                f"SELECT MAX(trade_date) FROM {self.TABLE_NAME}"
            ).fetchone()
        finally:
            conn.close()
        return row[0] if row and row[0] else None

    def _trading_dates_between(self, start_compact: str, end_compact: str) -> list[str]:
        """Use daily_quotes as canonical trading calendar."""
        conn = _connect(self.db_path)
        try:
            start_dash = _ymd_dash(start_compact)
            end_dash = _ymd_dash(end_compact)
            df = pd.read_sql_query(
                "SELECT DISTINCT trade_date FROM daily_quotes "
                "WHERE trade_date >= ? AND trade_date <= ? ORDER BY trade_date",
                conn,
                params=(start_dash, end_dash),
            )
        finally:
            conn.close()
        return [d.replace("-", "") for d in df["trade_date"].tolist()]

    def fetch_date_range(
        self,
        start_compact: str,
        end_compact: str,
        skip_existing: bool = True,
    ) -> int:
        """Iterate trading dates sequentially (APIs don't support date-range params)."""
        dates = self._trading_dates_between(start_compact, end_compact)
        if skip_existing:
            existing = self._existing_dates(start_compact, end_compact)
            dates = [d for d in dates if d not in existing]
        logger.info(
            f"[{self.TABLE_NAME}] {start_compact}~{end_compact}: "
            f"{len(dates)} trading days to fetch"
        )
        total = 0
        for i, d in enumerate(dates, 1):
            try:
                n = self.fetch_single_date(d)
                total += n
            except Exception as e:
                logger.warning(f"[{self.TABLE_NAME}] {d} failed: {e}")
            if i % 50 == 0:
                logger.info(
                    f"[{self.TABLE_NAME}] progress: {i}/{len(dates)}, "
                    f"total rows {total}"
                )
            time.sleep(self.RATE_LIMIT_SEC)
        logger.info(f"[{self.TABLE_NAME}] done: {total} rows written")
        return total

    def _existing_dates(self, start_compact: str, end_compact: str) -> set[str]:
        conn = _connect(self.db_path)
        try:
            df = pd.read_sql_query(
                f"SELECT DISTINCT trade_date FROM {self.TABLE_NAME} "
                f"WHERE trade_date >= ? AND trade_date <= ?",
                conn,
                params=(_ymd_dash(start_compact), _ymd_dash(end_compact)),
            )
        finally:
            conn.close()
        return {d.replace("-", "") for d in df["trade_date"].tolist()}


class TopListFetcher(_BaseFetcher):
    """龙虎榜每日数据. Same stock/date can appear multiple times with different
    上榜原因 (reason), so PK includes reason."""

    TABLE_NAME = "top_list_daily"

    def _ensure_table(self):
        conn = _connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS top_list_daily (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_date DATE NOT NULL,
                    ts_code VARCHAR(12) NOT NULL,
                    code VARCHAR(6) NOT NULL,
                    name VARCHAR(32),
                    close REAL,
                    pct_change REAL,
                    turnover_rate REAL,
                    amount REAL,
                    l_sell REAL,
                    l_buy REAL,
                    l_amount REAL,
                    net_amount REAL,
                    net_rate REAL,
                    amount_rate REAL,
                    float_values REAL,
                    reason TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(trade_date, ts_code, reason)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_top_list_date ON top_list_daily(trade_date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_top_list_code_date "
                "ON top_list_daily(code, trade_date)"
            )
            conn.commit()
        finally:
            conn.close()

    def fetch_single_date(self, date_compact: str) -> int:
        df = self.pro.top_list(trade_date=date_compact)
        if df is None or df.empty:
            return 0
        trade_date_dash = _ymd_dash(date_compact)
        rows = [
            (
                trade_date_dash,
                r["ts_code"],
                _code6(r["ts_code"]),
                r.get("name"),
                _f(r.get("close")),
                _f(r.get("pct_change")),
                _f(r.get("turnover_rate")),
                _f(r.get("amount")),
                _f(r.get("l_sell")),
                _f(r.get("l_buy")),
                _f(r.get("l_amount")),
                _f(r.get("net_amount")),
                _f(r.get("net_rate")),
                _f(r.get("amount_rate")),
                _f(r.get("float_values")),
                r.get("reason"),
            )
            for _, r in df.iterrows()
        ]
        conn = _connect(self.db_path)
        try:
            conn.executemany("""
                INSERT OR REPLACE INTO top_list_daily
                (trade_date, ts_code, code, name, close, pct_change, turnover_rate,
                 amount, l_sell, l_buy, l_amount, net_amount, net_rate, amount_rate,
                 float_values, reason, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
            """, rows)
            conn.commit()
        finally:
            conn.close()
        return len(rows)


class MarginDetailFetcher(_BaseFetcher):
    """融资融券个股明细. One row per (trade_date, ts_code)."""

    TABLE_NAME = "margin_detail_daily"

    def _ensure_table(self):
        conn = _connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS margin_detail_daily (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_date DATE NOT NULL,
                    ts_code VARCHAR(12) NOT NULL,
                    code VARCHAR(6) NOT NULL,
                    rzye REAL,      -- 融资余额
                    rqye REAL,      -- 融券余额
                    rzmre REAL,     -- 融资买入额
                    rqyl REAL,      -- 融券余量
                    rzche REAL,     -- 融资偿还额
                    rqchl REAL,     -- 融券偿还量
                    rqmcl REAL,     -- 融券卖出量
                    rzrqye REAL,    -- 融资融券余额
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(trade_date, ts_code)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_margin_date ON margin_detail_daily(trade_date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_margin_code_date "
                "ON margin_detail_daily(code, trade_date)"
            )
            conn.commit()
        finally:
            conn.close()

    def fetch_single_date(self, date_compact: str) -> int:
        df = self.pro.margin_detail(trade_date=date_compact)
        if df is None or df.empty:
            return 0
        trade_date_dash = _ymd_dash(date_compact)
        rows = [
            (
                trade_date_dash,
                r["ts_code"],
                _code6(r["ts_code"]),
                _f(r.get("rzye")),
                _f(r.get("rqye")),
                _f(r.get("rzmre")),
                _f(r.get("rqyl")),
                _f(r.get("rzche")),
                _f(r.get("rqchl")),
                _f(r.get("rqmcl")),
                _f(r.get("rzrqye")),
            )
            for _, r in df.iterrows()
        ]
        conn = _connect(self.db_path)
        try:
            conn.executemany("""
                INSERT OR REPLACE INTO margin_detail_daily
                (trade_date, ts_code, code, rzye, rqye, rzmre, rqyl, rzche, rqchl,
                 rqmcl, rzrqye, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
            """, rows)
            conn.commit()
        finally:
            conn.close()
        return len(rows)


class HsgtTop10Fetcher(_BaseFetcher):
    """北向十大活跃成交股. 20 rows/day (沪/深 各 10). market_type: 1=沪股通, 3=深股通."""

    TABLE_NAME = "hsgt_top10_daily"

    def _ensure_table(self):
        conn = _connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hsgt_top10_daily (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_date DATE NOT NULL,
                    ts_code VARCHAR(12) NOT NULL,
                    code VARCHAR(6) NOT NULL,
                    name VARCHAR(32),
                    close REAL,
                    change REAL,
                    rank INTEGER,
                    market_type INTEGER,  -- 1=沪股通, 3=深股通
                    amount REAL,
                    net_amount REAL,
                    buy REAL,
                    sell REAL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(trade_date, ts_code, market_type)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_hsgt_top10_date ON hsgt_top10_daily(trade_date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_hsgt_top10_code_date "
                "ON hsgt_top10_daily(code, trade_date)"
            )
            conn.commit()
        finally:
            conn.close()

    def fetch_single_date(self, date_compact: str) -> int:
        df = self.pro.hsgt_top10(trade_date=date_compact)
        if df is None or df.empty:
            return 0
        trade_date_dash = _ymd_dash(date_compact)
        rows = [
            (
                trade_date_dash,
                r["ts_code"],
                _code6(r["ts_code"]),
                r.get("name"),
                _f(r.get("close")),
                _f(r.get("change")),
                _i(r.get("rank")),
                _i(r.get("market_type")),
                _f(r.get("amount")),
                _f(r.get("net_amount")),
                _f(r.get("buy")),
                _f(r.get("sell")),
            )
            for _, r in df.iterrows()
        ]
        conn = _connect(self.db_path)
        try:
            conn.executemany("""
                INSERT OR REPLACE INTO hsgt_top10_daily
                (trade_date, ts_code, code, name, close, change, rank, market_type,
                 amount, net_amount, buy, sell, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
            """, rows)
            conn.commit()
        finally:
            conn.close()
        return len(rows)


def _f(v):
    if v is None or pd.isna(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _i(v):
    if v is None or pd.isna(v):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


FETCHER_REGISTRY = {
    "top_list": TopListFetcher,
    "margin": MarginDetailFetcher,
    "hsgt_top10": HsgtTop10Fetcher,
}


def main():
    parser = argparse.ArgumentParser(
        description="Alternative-alpha data fetchers (top_list / margin / hsgt_top10)"
    )
    parser.add_argument(
        "--api",
        choices=list(FETCHER_REGISTRY.keys()) + ["all"],
        default="all",
        help="Which API to fetch (default: all)",
    )
    parser.add_argument("--date", help="Single-day YYYYMMDD")
    parser.add_argument("--start-date", help="Batch start YYYYMMDD")
    parser.add_argument("--end-date", help="Batch end YYYYMMDD (default: today)")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="Re-fetch dates that already have rows (default: skip)")
    parser.add_argument("--db-path", help="Override DB path")
    args = parser.parse_args()

    apis = list(FETCHER_REGISTRY.keys()) if args.api == "all" else [args.api]
    pro = _load_pro_api()

    for api_name in apis:
        cls = FETCHER_REGISTRY[api_name]
        fetcher = cls(db_path=args.db_path, pro=pro)
        if args.date:
            n = fetcher.fetch_single_date(args.date)
            logger.info(f"[{api_name}] {args.date}: {n} rows")
        elif args.start_date:
            end = args.end_date or datetime.now().strftime("%Y%m%d")
            fetcher.fetch_date_range(
                args.start_date, end,
                skip_existing=not args.no_skip_existing,
            )
        else:
            today = datetime.now().strftime("%Y%m%d")
            n = fetcher.fetch_single_date(today)
            logger.info(f"[{api_name}] {today}: {n} rows")


if __name__ == "__main__":
    main()
