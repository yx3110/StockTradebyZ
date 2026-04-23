#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast-check IC / ICIR for candidate alt-alpha factors.

Design — no ML training, purely cross-sectional rank IC vs 10d forward return:
1. Load A-share universe (exclude *ST, ETFs, suspended).
2. Compute each registered factor over the eval window.
3. Compute 10d forward returns aligned to factor dates.
4. Per-day cross-sectional Spearman rank IC; summarize mean / std / ICIR / coverage.

Gate: |IC mean| > 0.015 AND |ICIR| > 0.25 → factor is candidate for ng1.7.
Lower than the usual 0.02/0.3 because alt-data is sparse by design (lots of
legitimate missing values) and the IC is diluted by zero-impute.
"""
import argparse
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = str(PROJECT_ROOT / "data_adapter" / "stock_data.db")
DEFAULT_START = "2024-07-01"
DEFAULT_END = "2026-04-22"
DEFAULT_HORIZON = 10


# ---------------- universe + returns ---------------- #

def load_universe_and_returns(conn, start_date: str, end_date: str, horizon: int) -> pd.DataFrame:
    """Return long-format df: code, trade_date, fwd_ret.

    Universe: A-shares only (exclude *ST from securities.name, exclude ETFs by type).
    Forward return: close_{t+horizon} / close_t - 1, shifted within each code.
    """
    logger.info("Loading universe + price series...")
    # Query everything back to `start_date - 30d` to avoid edge issues and forward to
    # `end_date + horizon+5d` so we can compute forward returns on the final factor date.
    # Database trade_date column is a string in 'YYYY-MM-DD'.
    q = """
        SELECT s.code, dq.trade_date, dq.close
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.type = 'A股'
          AND s.name NOT LIKE '%*ST%'
          AND dq.trade_date >= ?
          AND dq.trade_date <= date(?, '+30 days')
          AND dq.close IS NOT NULL
          AND dq.is_suspend = 0
        ORDER BY s.code, dq.trade_date
    """
    df = pd.read_sql_query(q, conn, params=(start_date, end_date))
    logger.info(f"  raw rows: {len(df):,}, codes: {df['code'].nunique()}")

    # Forward return per code
    df = df.sort_values(["code", "trade_date"])
    df["close_fwd"] = df.groupby("code")["close"].shift(-horizon)
    df["fwd_ret"] = df["close_fwd"] / df["close"] - 1

    # Trim to eval window
    df = df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)]
    df = df.dropna(subset=["fwd_ret"])
    logger.info(f"  post-horizon trimmed: {len(df):,} rows, {df['trade_date'].nunique()} dates")
    return df[["code", "trade_date", "fwd_ret"]].reset_index(drop=True)


# ---------------- factor builders ---------------- #

def _rolling_sum(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=max(1, window // 2)).sum()


def _rolling_mean(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=max(1, window // 2)).mean()


def factor_rzmre_5d_ratio(conn, start_date: str, end_date: str) -> pd.DataFrame:
    """F_margin_1: 近5日融资买入总和 / 融资余额均值. 主动加杠杆强度."""
    q = """
        SELECT code, trade_date, rzmre, rzye FROM margin_detail_daily
        WHERE trade_date >= date(?, '-20 days') AND trade_date <= ?
    """
    df = pd.read_sql_query(q, conn, params=(start_date, end_date))
    df = df.sort_values(["code", "trade_date"])
    df["rzmre_5d"] = df.groupby("code")["rzmre"].transform(lambda s: _rolling_sum(s, 5))
    df["rzye_5d"] = df.groupby("code")["rzye"].transform(lambda s: _rolling_mean(s, 5))
    df["factor"] = df["rzmre_5d"] / df["rzye_5d"].replace(0, np.nan)
    df = df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)]
    return df[["code", "trade_date", "factor"]].dropna()


def factor_rzye_chg_10d(conn, start_date: str, end_date: str) -> pd.DataFrame:
    """F_margin_2: 10日融资余额变化率. 资金持续流入趋势."""
    q = """
        SELECT code, trade_date, rzye FROM margin_detail_daily
        WHERE trade_date >= date(?, '-30 days') AND trade_date <= ?
    """
    df = pd.read_sql_query(q, conn, params=(start_date, end_date))
    df = df.sort_values(["code", "trade_date"])
    df["rzye_10d_ago"] = df.groupby("code")["rzye"].shift(10)
    df["factor"] = df["rzye"] / df["rzye_10d_ago"].replace(0, np.nan) - 1
    df = df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)]
    return df[["code", "trade_date", "factor"]].dropna()


def factor_rzrqye_to_mcap(conn, start_date: str, end_date: str) -> pd.DataFrame:
    """F_margin_3: 融资融券余额 / 流通市值. 拥挤度/杠杆水平."""
    q = """
        SELECT md.code, md.trade_date, md.rzrqye, db.circ_mv
        FROM margin_detail_daily md
        JOIN securities s ON s.code = md.code
        JOIN daily_basic db ON db.security_id = s.id AND db.trade_date = md.trade_date
        WHERE md.trade_date >= ? AND md.trade_date <= ?
          AND db.circ_mv IS NOT NULL AND db.circ_mv > 0
    """
    df = pd.read_sql_query(q, conn, params=(start_date, end_date))
    # circ_mv is in 万元; rzrqye in 元 → convert rzrqye to 万元
    df["factor"] = (df["rzrqye"] / 10000.0) / df["circ_mv"]
    return df[["code", "trade_date", "factor"]].dropna()


def factor_lhb_inst_net_5d(conn, start_date: str, end_date: str) -> pd.DataFrame:
    """F_lhb_1: 近5日机构席位净买 / 流通市值. 机构看法."""
    q = """
        SELECT code, trade_date, l_buy, l_sell FROM top_list_daily
        WHERE trade_date >= date(?, '-20 days') AND trade_date <= ?
    """
    df = pd.read_sql_query(q, conn, params=(start_date, end_date))
    # Aggregate by (code, trade_date) — same stock may have multiple reasons
    df = df.groupby(["code", "trade_date"], as_index=False).agg(
        {"l_buy": "sum", "l_sell": "sum"}
    )
    df["inst_net"] = df["l_buy"].fillna(0) - df["l_sell"].fillna(0)
    df = df.sort_values(["code", "trade_date"])
    df["factor_raw"] = df.groupby("code")["inst_net"].transform(lambda s: _rolling_sum(s, 5))
    # Join flowing mv
    mv = pd.read_sql_query(
        """SELECT s.code, db.trade_date, db.circ_mv
           FROM daily_basic db JOIN securities s ON s.id = db.security_id
           WHERE db.trade_date >= ? AND db.trade_date <= ? AND db.circ_mv > 0""",
        conn, params=(start_date, end_date),
    )
    df = df.merge(mv, on=["code", "trade_date"], how="left")
    df["factor"] = df["factor_raw"] / (df["circ_mv"] * 10000)  # 两者统一元
    df = df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)]
    return df[["code", "trade_date", "factor"]].dropna()


def factor_lhb_count_20d(conn, start_date: str, end_date: str) -> pd.DataFrame:
    """F_lhb_2: 近20日上龙虎榜次数 (按 code,date 去重后再累加). 关注度."""
    q = """
        SELECT DISTINCT code, trade_date FROM top_list_daily
        WHERE trade_date >= date(?, '-60 days') AND trade_date <= ?
    """
    hits = pd.read_sql_query(q, conn, params=(start_date, end_date))
    hits["hit"] = 1

    # Build a date grid of all trading dates in the window for each stock that ever hit
    dates_q = """
        SELECT DISTINCT trade_date FROM daily_quotes
        WHERE trade_date >= date(?, '-60 days') AND trade_date <= ?
        ORDER BY trade_date
    """
    all_dates = pd.read_sql_query(dates_q, conn, params=(start_date, end_date))["trade_date"]

    codes = hits["code"].unique()
    logger.info(f"    lhb_count_20d building grid for {len(codes)} unique hit codes × {len(all_dates)} dates")
    grid = pd.MultiIndex.from_product([codes, all_dates], names=["code", "trade_date"]).to_frame(index=False)
    grid = grid.merge(hits, on=["code", "trade_date"], how="left").fillna({"hit": 0})
    grid = grid.sort_values(["code", "trade_date"])
    grid["factor"] = grid.groupby("code")["hit"].transform(lambda s: _rolling_sum(s, 20))
    grid = grid[(grid["trade_date"] >= start_date) & (grid["trade_date"] <= end_date)]
    return grid[grid["factor"] > 0][["code", "trade_date", "factor"]]


def factor_hsgt_top10_flag(conn, start_date: str, end_date: str) -> pd.DataFrame:
    """F_hsgt_1: 近 20 日上北向 top10 次数. 高于 binary, 提供更多 cross-sectional 区分度.

    Expanded to all A-share universe (non-hit stocks get factor=0) so cross-sectional
    coverage is high enough for IC computation.
    """
    # Fetch hit history
    q = """
        SELECT DISTINCT code, trade_date FROM hsgt_top10_daily
        WHERE trade_date >= date(?, '-60 days') AND trade_date <= ?
    """
    hits = pd.read_sql_query(q, conn, params=(start_date, end_date))
    hits["hit"] = 1

    # A-share universe × all eval dates
    univ_q = """
        SELECT DISTINCT s.code, dq.trade_date
        FROM daily_quotes dq JOIN securities s ON dq.security_id = s.id
        WHERE s.type = 'A股' AND s.name NOT LIKE '%*ST%'
          AND dq.trade_date >= date(?, '-60 days') AND dq.trade_date <= ?
          AND dq.is_suspend = 0
    """
    grid = pd.read_sql_query(univ_q, conn, params=(start_date, end_date))
    grid = grid.merge(hits, on=["code", "trade_date"], how="left").fillna({"hit": 0})
    grid = grid.sort_values(["code", "trade_date"])
    grid["factor"] = grid.groupby("code")["hit"].transform(lambda s: _rolling_sum(s, 20))
    grid = grid[(grid["trade_date"] >= start_date) & (grid["trade_date"] <= end_date)]
    return grid[["code", "trade_date", "factor"]].dropna()


def factor_hsgt_streak(conn, start_date: str, end_date: str) -> pd.DataFrame:
    """F_hsgt_2: 连续上北向 top10 天数. 持续性."""
    # Fetch deeper history for streak calc
    q = """
        SELECT DISTINCT code, trade_date FROM hsgt_top10_daily
        WHERE trade_date >= date(?, '-60 days') AND trade_date <= ?
    """
    hits = pd.read_sql_query(q, conn, params=(start_date, end_date))
    hits["hit"] = 1

    # For each code that ever hit, expand dates
    dates_q = """
        SELECT DISTINCT trade_date FROM daily_quotes
        WHERE trade_date >= date(?, '-60 days') AND trade_date <= ?
        ORDER BY trade_date
    """
    all_dates = pd.read_sql_query(dates_q, conn, params=(start_date, end_date))["trade_date"]
    codes = hits["code"].unique()
    grid = pd.MultiIndex.from_product([codes, all_dates], names=["code", "trade_date"]).to_frame(index=False)
    grid = grid.merge(hits, on=["code", "trade_date"], how="left").fillna({"hit": 0})
    grid = grid.sort_values(["code", "trade_date"])
    # Streak: reset to 0 on miss, +1 on hit
    def _streak(s):
        out = np.zeros(len(s), dtype=float)
        cnt = 0
        for i, v in enumerate(s.values):
            cnt = cnt + 1 if v == 1 else 0
            out[i] = cnt
        return pd.Series(out, index=s.index)
    grid["factor"] = grid.groupby("code")["hit"].transform(_streak)
    grid = grid[(grid["trade_date"] >= start_date) & (grid["trade_date"] <= end_date)]
    # Keep all rows (including 0-streak) for cross-sectional coverage
    return grid[["code", "trade_date", "factor"]]


FACTOR_REGISTRY = {
    "F_margin_1_rzmre_5d_ratio": factor_rzmre_5d_ratio,
    "F_margin_2_rzye_chg_10d": factor_rzye_chg_10d,
    "F_margin_3_rzrqye_to_mcap": factor_rzrqye_to_mcap,
    "F_lhb_1_inst_net_5d": factor_lhb_inst_net_5d,
    "F_lhb_2_count_20d": factor_lhb_count_20d,
    "F_hsgt_1_top10_flag": factor_hsgt_top10_flag,
    "F_hsgt_2_streak": factor_hsgt_streak,
}


# ---------------- IC evaluation ---------------- #

def compute_ic(factor_df: pd.DataFrame, ret_df: pd.DataFrame) -> dict:
    """Cross-sectional Spearman rank IC per day, then summarize."""
    merged = factor_df.merge(ret_df, on=["code", "trade_date"], how="inner")
    if len(merged) == 0:
        return {"n_days": 0, "avg_coverage": 0, "ic_mean": np.nan,
                "ic_std": np.nan, "icir": np.nan, "ic_pos_rate": np.nan,
                "merged_rows": 0}

    daily_ics = []
    coverages = []
    for d, grp in merged.groupby("trade_date"):
        if len(grp) < 30:
            continue
        f = rankdata(grp["factor"].values)
        r = rankdata(grp["fwd_ret"].values)
        # Pearson on ranks == Spearman
        ic = np.corrcoef(f, r)[0, 1]
        if np.isnan(ic):
            continue
        daily_ics.append(ic)
        coverages.append(len(grp))

    if not daily_ics:
        return {"n_days": 0, "avg_coverage": 0, "ic_mean": np.nan,
                "ic_std": np.nan, "icir": np.nan, "ic_pos_rate": np.nan,
                "merged_rows": len(merged)}

    ics = np.array(daily_ics)
    return {
        "n_days": len(ics),
        "avg_coverage": int(np.mean(coverages)),
        "ic_mean": float(np.mean(ics)),
        "ic_std": float(np.std(ics)),
        "icir": float(np.mean(ics) / (np.std(ics) + 1e-9)),
        "ic_pos_rate": float(np.mean(ics > 0)),
        "merged_rows": len(merged),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default=DEFAULT_START)
    parser.add_argument("--end-date", default=DEFAULT_END)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--factors", nargs="*", default=list(FACTOR_REGISTRY.keys()),
                        help="subset of factor names, or all")
    parser.add_argument("--ic-threshold", type=float, default=0.015)
    parser.add_argument("--icir-threshold", type=float, default=0.25)
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")

    t0 = time.time()
    ret_df = load_universe_and_returns(conn, args.start_date, args.end_date, args.horizon)
    logger.info(f"Forward returns loaded in {time.time()-t0:.1f}s")

    results = []
    for fname in args.factors:
        if fname not in FACTOR_REGISTRY:
            logger.warning(f"Unknown factor: {fname}; skipping")
            continue
        t1 = time.time()
        logger.info(f"\n--- Computing {fname} ---")
        factor_df = FACTOR_REGISTRY[fname](conn, args.start_date, args.end_date)
        logger.info(f"  factor rows: {len(factor_df):,}, in {time.time()-t1:.1f}s")
        t2 = time.time()
        stats = compute_ic(factor_df, ret_df)
        stats["name"] = fname
        logger.info(
            f"  IC={stats['ic_mean']:+.4f} std={stats['ic_std']:.4f} "
            f"ICIR={stats['icir']:+.2f} pos_rate={stats['ic_pos_rate']:.2f} "
            f"coverage={stats['avg_coverage']}/day days={stats['n_days']} "
            f"({time.time()-t2:.1f}s)"
        )
        results.append(stats)

    # Summary table
    print("\n" + "=" * 90)
    print(f"{'factor':<32} {'IC':>8} {'ICIR':>7} {'pos%':>6} {'cov/day':>8} {'days':>6} {'gate':>6}")
    print("=" * 90)
    for r in results:
        passed = (abs(r["ic_mean"]) >= args.ic_threshold and
                  abs(r["icir"]) >= args.icir_threshold)
        gate = "✅" if passed else "❌"
        print(f"{r['name']:<32} {r['ic_mean']:+8.4f} {r['icir']:+7.2f} "
              f"{r['ic_pos_rate']*100:5.1f}% {r['avg_coverage']:>8} "
              f"{r['n_days']:>6} {gate:>6}")
    print("=" * 90)
    passed = [r for r in results if abs(r["ic_mean"]) >= args.ic_threshold
              and abs(r["icir"]) >= args.icir_threshold]
    print(f"\nGate: |IC|≥{args.ic_threshold} AND |ICIR|≥{args.icir_threshold}")
    print(f"Pass: {len(passed)}/{len(results)}")
    if passed:
        print("Candidates for ng1.7:")
        for r in passed:
            print(f"  {r['name']}  IC={r['ic_mean']:+.4f}  ICIR={r['icir']:+.2f}")


if __name__ == "__main__":
    main()
