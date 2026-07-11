#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adj_factor 复权因子回填 (2026-07-11 北极星 P0 修复配套)

背景: daily_quotes.adj_factor 全库无一行真实值 — 历史行 NULL, 其余行是
schema DEFAULT 1.0 占位符 (日常管道从不抓取 adj_factor, INSERT OR REPLACE
列清单也不含该列)。回测因此无法复权, 除权日出现 -30%~-50% 假暴跌
(2023 主板实测 238 次)。
⚠️ 本脚本只负责«落地数据»; 收益路径的复权消费在 P0.1b (用复权价重算
个股 price_change_pct) 完成之前, 除权失真尚未修复。
详见 reports/system_evaluation/选股系统与北极星系统评估与风控内化可行性研究_20260711.md

策略: 按 trade_date 逐日调 pro.adj_factor(trade_date=...) (一次返回全市场),
回填 NULL 与占位符 1.0 的行。真实累计复权因子绝大多数 ≠ 1.0, 因此用
"当日 ≠1.0 的行占比 < 50%" 判定缺口日 (占位日实测 0%, 真实日 ~70-90%)。

配套修复 (同 commit): quick_daily_update.batch_update_stocks 每日抓取
adj_factor, insert_daily_quotes 列清单加 adj_factor — 否则缺口会在
增量日重新出现, 且重拉历史日会把回填值冲回 1.0。

用法:
  python3 fetch_data/backfill_adj_factor.py                         # 回填全部缺口日
  python3 fetch_data/backfill_adj_factor.py --start 2020-01-01 --end 2024-12-31
  python3 fetch_data/backfill_adj_factor.py --dry-run               # 只列缺口日不动库
"""

import argparse
import logging
import sqlite3
import sys
import time
from pathlib import Path

import tushare as ts

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import get_tushare_token, get_db_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RATE_LIMIT_SLEEP = 0.15  # 500次/分钟限制, 0.15s ≈ 400次/分钟留余量
REAL_COVERAGE_THRESHOLD = 0.5  # 当日 adj_factor≠1.0 行占比低于此值 → 缺口日

GAP_DAY_SQL = """
    SELECT dq.trade_date,
           COUNT(*) AS n,
           SUM(dq.adj_factor IS NOT NULL AND dq.adj_factor <> 1.0) AS n_real
    FROM daily_quotes dq JOIN securities s ON s.id = dq.security_id
    WHERE s.type = 'A股' AND dq.trade_date BETWEEN ? AND ?
    GROUP BY dq.trade_date
"""


def find_gap_dates(conn: sqlite3.Connection, start: str, end: str) -> list[str]:
    rows = conn.execute(GAP_DAY_SQL, (start, end)).fetchall()
    return [d for d, n, n_real in rows if n_real < REAL_COVERAGE_THRESHOLD * n]


def load_security_map(conn: sqlite3.Connection) -> dict[str, int]:
    """bare code -> security_id (仅 A股)"""
    return {
        code: sid
        for sid, code in conn.execute("SELECT id, code FROM securities WHERE type='A股'")
    }


def backfill_date(pro, conn: sqlite3.Connection, sec_map: dict[str, int], trade_date: str) -> int:
    """回填单日, 返回更新行数。只覆盖 NULL 与占位符 1.0, 不动其他已有值。"""
    df = pro.adj_factor(trade_date=trade_date.replace("-", ""))
    if df is None or df.empty:
        logger.warning(f"{trade_date}: API 返回空")
        return 0
    df = df.dropna(subset=["adj_factor"])
    sids = df["ts_code"].str.split(".").str[0].map(sec_map)
    valid = sids.notna()
    params = list(zip(df.loc[valid, "adj_factor"].astype(float),
                      sids[valid].astype(int),
                      [trade_date] * int(valid.sum())))
    cur = conn.executemany(
        "UPDATE daily_quotes SET adj_factor=? "
        "WHERE security_id=? AND trade_date=? "
        "AND (adj_factor IS NULL OR adj_factor = 1.0)",
        params,
    )
    conn.commit()
    return cur.rowcount


def main():
    parser = argparse.ArgumentParser(description="adj_factor 回填")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-12-31")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    conn = sqlite3.connect(str(get_db_path()))
    conn.execute("PRAGMA busy_timeout=30000")

    gap_dates = find_gap_dates(conn, args.start, args.end)
    logger.info(f"缺口交易日 (真实因子占比<{REAL_COVERAGE_THRESHOLD:.0%}): {len(gap_dates)} 天, "
                f"范围 {gap_dates[0] if gap_dates else '-'} ~ {gap_dates[-1] if gap_dates else '-'}")
    if args.dry_run or not gap_dates:
        return

    ts.set_token(get_tushare_token())
    pro = ts.pro_api()
    sec_map = load_security_map(conn)

    total = 0
    failed_dates = []
    for i, d in enumerate(gap_dates, 1):
        try:
            total += backfill_date(pro, conn, sec_map, d)
            if i % 100 == 0 or i == len(gap_dates):
                logger.info(f"进度 {i}/{len(gap_dates)} ({d}): 累计更新 {total} 行")
        except Exception as e:
            logger.error(f"{d}: {e} — 跳过该日继续")
            failed_dates.append(d)
        time.sleep(RATE_LIMIT_SLEEP)

    # 收尾复查: 只查本次处理过的日子, 避免再做全表聚合
    placeholders = ",".join("?" * len(gap_dates))
    rows = conn.execute(
        GAP_DAY_SQL.replace("dq.trade_date BETWEEN ? AND ?",
                            f"dq.trade_date IN ({placeholders})"),
        gap_dates,
    ).fetchall()
    remaining = [d for d, n, n_real in rows if n_real < REAL_COVERAGE_THRESHOLD * n]
    logger.info(f"完成: 共更新 {total} 行; API 失败 {len(failed_dates)} 天{failed_dates[:5] if failed_dates else ''}; "
                f"仍未达标 {len(remaining)} 天{remaining[:5] if remaining else ''}")


if __name__ == "__main__":
    main()
