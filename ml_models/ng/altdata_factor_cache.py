"""Alt-alpha factor cache — precomputed 4 factors shared across ng1.7+ versions.

Table: altdata_factor_cache
  - Row per (code, trade_date)
  - 4 factor columns: altdata_rzmre_5d_ratio, altdata_rzye_chg_10d,
                       altdata_lhb_inst_net_5d, altdata_lhb_count_20d
  - Source tables: margin_detail_daily, top_list_daily, daily_basic

Factor definitions (all passed 2026-04-23 IC fast-check):
  - rzmre_5d_ratio:    Σ rzmre / Σ rzye (近5日融资买入 / 近5日融资余额均值).
                        IC=-0.1024, ICIR=-0.72 (散户追高反向).
  - rzye_chg_10d:      rzye_t / rzye_{t-10} - 1 (10日融资余额变化率).
                        IC=-0.0462, ICIR=-0.92 (融资加速反向).
  - lhb_inst_net_5d:   Σ (l_buy - l_sell) / circ_mv  近5日机构席位净买占流通市值.
                        IC=+0.0391, ICIR=+0.26 (smart money).
  - lhb_count_20d:     近20日上龙虎榜天数 (同日多 reason 只记1).
                        IC=-0.0724, ICIR=-0.83 (过度炒作反向).

Usage:
  from ml_models.ng.altdata_factor_cache import load_factors_for_dates
  df = load_factors_for_dates(conn, ['2026-04-22', '2026-04-23'])
  # df has columns: code, trade_date, altdata_rzmre_5d_ratio, ...
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = str(PROJECT_ROOT / "data_adapter" / "stock_data.db")

FACTOR_COLS = [
    "altdata_rzmre_5d_ratio",
    "altdata_rzye_chg_10d",
    "altdata_lhb_inst_net_5d",
    "altdata_lhb_count_20d",
]


# ---------------- schema ---------------- #

def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS altdata_factor_cache (
            code VARCHAR(10) NOT NULL,
            trade_date DATE NOT NULL,
            altdata_rzmre_5d_ratio REAL,
            altdata_rzye_chg_10d REAL,
            altdata_lhb_inst_net_5d REAL,
            altdata_lhb_count_20d REAL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (code, trade_date)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_altdata_factor_date ON altdata_factor_cache(trade_date)"
    )
    conn.commit()


# ---------------- vectorized factor computation over a date window ---------------- #

def _compute_factors_window(
    conn: sqlite3.Connection,
    window_start: str,
    window_end: str,
) -> pd.DataFrame:
    """Compute all 4 factors for every (code, trade_date) in [window_start, window_end].

    Reads enough history before window_start to fill rolling windows (max 20-day lookback).
    Returns df with columns: code, trade_date, + 4 factor cols.
    """
    lookback_start = pd.to_datetime(window_start) - pd.Timedelta(days=45)
    lookback_start_str = lookback_start.strftime("%Y-%m-%d")

    logger.info(
        f"  computing factors for {window_start}~{window_end} "
        f"(lookback from {lookback_start_str})"
    )

    # --- Margin factors (F1, F2) --- #
    t0 = time.time()
    margin_df = pd.read_sql_query(
        """
        SELECT code, trade_date, rzmre, rzye
        FROM margin_detail_daily
        WHERE trade_date >= ? AND trade_date <= ?
        """,
        conn,
        params=(lookback_start_str, window_end),
    )
    logger.info(f"    margin rows: {len(margin_df):,} ({time.time()-t0:.1f}s)")

    margin_df = margin_df.sort_values(["code", "trade_date"])
    grp = margin_df.groupby("code", group_keys=False)
    # F1: Σ rzmre(5) / mean rzye(5)
    margin_df["rzmre_5d_sum"] = grp["rzmre"].transform(
        lambda s: s.rolling(5, min_periods=3).sum()
    )
    margin_df["rzye_5d_mean"] = grp["rzye"].transform(
        lambda s: s.rolling(5, min_periods=3).mean()
    )
    margin_df["altdata_rzmre_5d_ratio"] = (
        margin_df["rzmre_5d_sum"] / margin_df["rzye_5d_mean"].replace(0, np.nan)
    )
    # F2: rzye / rzye[-10] - 1
    margin_df["rzye_10d_ago"] = grp["rzye"].shift(10)
    margin_df["altdata_rzye_chg_10d"] = (
        margin_df["rzye"] / margin_df["rzye_10d_ago"].replace(0, np.nan) - 1
    )
    margin_feats = margin_df[
        ["code", "trade_date", "altdata_rzmre_5d_ratio", "altdata_rzye_chg_10d"]
    ]

    # --- LHB factors (F3, F4) --- #
    t1 = time.time()
    lhb_raw = pd.read_sql_query(
        """
        SELECT code, trade_date, l_buy, l_sell
        FROM top_list_daily
        WHERE trade_date >= ? AND trade_date <= ?
        """,
        conn,
        params=(lookback_start_str, window_end),
    )
    logger.info(f"    lhb rows: {len(lhb_raw):,} ({time.time()-t1:.1f}s)")

    # Collapse multiple-reason rows per (code, date)
    lhb_agg = (
        lhb_raw.groupby(["code", "trade_date"], as_index=False)
        .agg(l_buy=("l_buy", "sum"), l_sell=("l_sell", "sum"))
    )
    lhb_agg["inst_net"] = lhb_agg["l_buy"].fillna(0) - lhb_agg["l_sell"].fillna(0)

    # Get circ_mv for all (code, date) rows we need
    mv_df = pd.read_sql_query(
        """
        SELECT s.code, db.trade_date, db.circ_mv
        FROM daily_basic db
        JOIN securities s ON s.id = db.security_id
        WHERE db.trade_date >= ? AND db.trade_date <= ?
          AND db.circ_mv IS NOT NULL AND db.circ_mv > 0
        """,
        conn,
        params=(lookback_start_str, window_end),
    )

    # --- F4: 20d count of appearances (only for codes that ever appear) --- #
    lhb_hits = lhb_agg[["code", "trade_date"]].copy()
    lhb_hits["hit"] = 1

    # Grid over all trading dates × hit_codes
    trading_dates = pd.read_sql_query(
        """
        SELECT DISTINCT trade_date FROM daily_quotes
        WHERE trade_date >= ? AND trade_date <= ? ORDER BY trade_date
        """,
        conn,
        params=(lookback_start_str, window_end),
    )["trade_date"]

    hit_codes = lhb_hits["code"].unique()
    logger.info(f"    lhb grid: {len(hit_codes)} codes × {len(trading_dates)} dates")
    grid = pd.MultiIndex.from_product(
        [hit_codes, trading_dates], names=["code", "trade_date"]
    ).to_frame(index=False)
    grid = grid.merge(lhb_hits, on=["code", "trade_date"], how="left")
    grid["hit"] = grid["hit"].fillna(0)
    grid = grid.sort_values(["code", "trade_date"])
    grid["altdata_lhb_count_20d"] = grid.groupby("code", group_keys=False)["hit"].transform(
        lambda s: s.rolling(20, min_periods=5).sum()
    )

    # F3: need grid with inst_net carry-forward + 5d rolling sum / circ_mv
    grid = grid.merge(
        lhb_agg[["code", "trade_date", "inst_net"]],
        on=["code", "trade_date"],
        how="left",
    )
    grid["inst_net"] = grid["inst_net"].fillna(0)
    grid["inst_net_5d_sum"] = grid.groupby("code", group_keys=False)["inst_net"].transform(
        lambda s: s.rolling(5, min_periods=1).sum()
    )
    grid = grid.merge(mv_df, on=["code", "trade_date"], how="left")
    # inst_net_5d_sum in 元; circ_mv in 万元 → ratio = inst_net_5d_sum / (circ_mv * 1e4)
    grid["altdata_lhb_inst_net_5d"] = grid["inst_net_5d_sum"] / (grid["circ_mv"] * 1e4)
    # Keep factor only on dates where circ_mv is available (avoid spurious NaNs on ETFs)
    lhb_feats = grid[
        ["code", "trade_date", "altdata_lhb_inst_net_5d", "altdata_lhb_count_20d"]
    ]

    # --- Outer merge, keep only rows in the target window --- #
    merged = margin_feats.merge(
        lhb_feats, on=["code", "trade_date"], how="outer"
    )
    merged = merged[
        (merged["trade_date"] >= window_start) & (merged["trade_date"] <= window_end)
    ]

    # A row is useful if at least one factor is non-null
    merged = merged.dropna(
        subset=FACTOR_COLS, how="all"
    )

    logger.info(f"  output rows: {len(merged):,}")
    return merged


# ---------------- write-to-DB in date chunks ---------------- #

def _write_chunk(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    cols = ["code", "trade_date"] + FACTOR_COLS
    placeholders = ",".join("?" * len(cols))
    rows = [
        tuple(None if pd.isna(v) else v for v in row)
        for row in df[cols].itertuples(index=False, name=None)
    ]
    conn.executemany(
        f"""
        INSERT OR REPLACE INTO altdata_factor_cache
        ({','.join(cols)}, updated_at)
        VALUES ({placeholders}, CURRENT_TIMESTAMP)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def build_cache(
    start_date: str = "2018-01-01",
    end_date: Optional[str] = None,
    chunk_months: int = 6,
    db_path: Optional[str] = None,
) -> int:
    """Full build 2018-today, chunked by N months to bound memory."""
    end_date = end_date or pd.Timestamp.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect(db_path or DB_PATH, timeout=60)
    conn.execute("PRAGMA busy_timeout = 60000")
    _ensure_table(conn)

    chunks: list[tuple[str, str]] = []
    cur = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    while cur <= end_ts:
        chunk_end = min(cur + pd.DateOffset(months=chunk_months) - pd.Timedelta(days=1), end_ts)
        chunks.append((cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cur = chunk_end + pd.Timedelta(days=1)

    total = 0
    for i, (s, e) in enumerate(chunks, 1):
        logger.info(f"\n[{i}/{len(chunks)}] chunk {s} ~ {e}")
        t0 = time.time()
        df = _compute_factors_window(conn, s, e)
        n = _write_chunk(conn, df)
        total += n
        logger.info(f"  wrote {n:,} rows ({time.time()-t0:.1f}s)")

    conn.close()
    logger.info(f"\nBuild complete: {total:,} rows total")
    return total


def update_daily(date_str: str, db_path: Optional[str] = None) -> int:
    """Single-day update — used by quick_daily_update pipeline."""
    conn = sqlite3.connect(db_path or DB_PATH, timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")
    _ensure_table(conn)
    df = _compute_factors_window(conn, date_str, date_str)
    n = _write_chunk(conn, df)
    conn.close()
    logger.info(f"altdata_factor_cache {date_str}: {n} rows")
    return n


# ---------------- read helper for trainer / scorer ---------------- #

def load_factors_for_dates(
    conn: sqlite3.Connection,
    dates: Iterable[str],
    codes: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Return df(code, trade_date, + 4 factor cols) for the given dates.

    Missing (code, date) rows get NaN — caller decides fill policy.
    """
    dates = list(dates)
    if not dates:
        return pd.DataFrame(columns=["code", "trade_date"] + FACTOR_COLS)

    in_dates = ",".join("?" * len(dates))
    cols_select = ", ".join(FACTOR_COLS)

    if codes is not None:
        codes = list(codes)
        # SQLite parameter limit 999; chunk if needed
        out = []
        for i in range(0, len(codes), 900):
            chunk = codes[i : i + 900]
            in_codes = ",".join("?" * len(chunk))
            df = pd.read_sql_query(
                f"""
                SELECT code, trade_date, {cols_select}
                FROM altdata_factor_cache
                WHERE trade_date IN ({in_dates})
                  AND code IN ({in_codes})
                """,
                conn,
                params=list(dates) + chunk,
            )
            out.append(df)
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
            columns=["code", "trade_date"] + FACTOR_COLS
        )

    return pd.read_sql_query(
        f"""
        SELECT code, trade_date, {cols_select}
        FROM altdata_factor_cache
        WHERE trade_date IN ({in_dates})
        """,
        conn,
        params=list(dates),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2018-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--chunk-months", type=int, default=6)
    parser.add_argument("--daily", help="single-day update YYYY-MM-DD (overrides start/end)")
    args = parser.parse_args()

    if args.daily:
        update_daily(args.daily)
    else:
        build_cache(args.start_date, args.end_date, args.chunk_months)


if __name__ == "__main__":
    main()
