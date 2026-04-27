"""P1.3 Step A: risk-adjusted label generators (Calmar / Sortino).

Outputs a parquet/csv keyed by (trade_date, code) with three labels:
  - label_industry_excess_10d   ← baseline (matches ng1.0.1)
  - label_calmar_10d            ← industry_excess / max(|future_maxdd_10d|, floor)
  - label_sortino_10d           ← industry_excess / future_downside_vol_10d

This is intentionally a *file artefact* (not a trainer change). The next step
of P1.3 imports this file as label override in the trainer; running it
independently lets us inspect distribution / correlation with the original
label before any trainer surgery.

Usage:
  python3 ml_models/ng/risk_adjusted_labels.py \\
      --start 2020-01-01 --end 2024-12-31 --out reports/labels/ng162_labels.csv
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DB = ROOT / "data_adapter" / "stock_data.db"


def load_close_panel(start: str, end: str, buffer_days: int = 30) -> pd.DataFrame:
    end_buf = (pd.Timestamp(end) + pd.Timedelta(days=buffer_days)).strftime("%Y-%m-%d")
    conn = sqlite3.connect(str(DB), timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        df = pd.read_sql_query(
            "SELECT s.code, dq.trade_date, dq.close FROM daily_quotes dq "
            "JOIN securities s ON s.id = dq.security_id "
            "WHERE dq.trade_date BETWEEN ? AND ? AND s.type = 'A股'",
            conn, params=[start, end_buf],
        )
    finally:
        conn.close()
    return df.pivot(index="trade_date", columns="code", values="close").sort_index()


def load_industry_lookup() -> pd.DataFrame:
    conn = sqlite3.connect(str(DB), timeout=30)
    try:
        df = pd.read_sql_query(
            "SELECT code, industry FROM securities WHERE type = 'A股'", conn,
        )
    finally:
        conn.close()
    return df


def compute_future_ret_10d(close: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Standard forward N-day return."""
    return (close.shift(-n) / close - 1.0).stack().rename("ret").reset_index()


def compute_future_maxdd_10d(close: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Worst path drawdown over the next N days, ∈ [-1, 0].

    Uses min over [t+1, t+n] / close[t] - 1.
    """
    fwd_min = close.shift(-1).rolling(n, min_periods=3).min().shift(-(n - 1))
    dd = (fwd_min / close - 1.0).clip(lower=-1.0, upper=0.0)
    return dd.stack().rename("maxdd").reset_index()


def compute_future_dnvol_10d(close: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Downside vol = std of negative log-rets only, annualized."""
    log_ret = np.log(close).diff()
    neg = log_ret.where(log_ret < 0)
    fwd_dn = (
        neg.shift(-1).rolling(n, min_periods=3).std().shift(-(n - 1)) * np.sqrt(252)
    )
    return fwd_dn.stack().rename("dnvol").reset_index()


def industry_excess(returns: pd.DataFrame, industry_lookup: pd.DataFrame,
                    ret_col: str = "ret") -> pd.DataFrame:
    """Per-(date, industry) mean subtracted from individual returns."""
    j = returns.merge(industry_lookup, on="code", how="left")
    j["industry"] = j["industry"].fillna("UNKNOWN")
    industry_avg = j.groupby(["trade_date", "industry"])[ret_col].transform("mean")
    j["label_industry_excess"] = j[ret_col] - industry_avg
    return j[["trade_date", "code", ret_col, "label_industry_excess"]]


def winsorize(s: pd.Series, lo: float = 0.005, hi: float = 0.995) -> pd.Series:
    if s.empty:
        return s
    lo_v, hi_v = s.quantile([lo, hi])
    return s.clip(lower=lo_v, upper=hi_v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--calmar-floor", type=float, default=0.05,
                    help="floor for |maxdd| denominator")
    ap.add_argument("--sortino-floor", type=float, default=0.10,
                    help="floor for downside vol (annual) denominator")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[1/5] loading close panel {args.start}~{args.end}...")
    close = load_close_panel(args.start, args.end)
    print(f"  → {close.shape[0]} rows × {close.shape[1]} codes")

    print(f"[2/5] computing forward returns / maxdd / dnvol (n={args.horizon})...")
    rets = compute_future_ret_10d(close, args.horizon)
    rets.columns = ["trade_date", "code", "ret"]
    dds = compute_future_maxdd_10d(close, args.horizon)
    dds.columns = ["trade_date", "code", "maxdd"]
    dnv = compute_future_dnvol_10d(close, args.horizon)
    dnv.columns = ["trade_date", "code", "dnvol"]

    print("[3/5] merging + computing industry-excess baseline label...")
    industry = load_industry_lookup()
    excess = industry_excess(rets, industry)
    df = excess.merge(dds, on=["trade_date", "code"], how="left")
    df = df.merge(dnv, on=["trade_date", "code"], how="left")
    df = df.dropna(subset=["label_industry_excess", "maxdd"])

    print("[4/5] computing Calmar / Sortino labels (winsorized 0.5%-99.5%)...")
    df["label_calmar"] = df["label_industry_excess"] / np.maximum(
        np.abs(df["maxdd"]), args.calmar_floor)
    df["label_sortino"] = df["label_industry_excess"] / np.maximum(
        df["dnvol"].fillna(args.sortino_floor), args.sortino_floor)
    df["label_calmar"] = winsorize(df["label_calmar"])
    df["label_sortino"] = winsorize(df["label_sortino"])

    print("[5/5] sanity checks + writing artefact...")
    print("\nlabel distribution (post-winsorize):")
    print(df[["label_industry_excess", "label_calmar", "label_sortino"]].describe(percentiles=[0.05, 0.5, 0.95]))

    corr = df[["label_industry_excess", "label_calmar", "label_sortino"]].corr()
    print("\nlabel correlation:")
    print(corr.round(3))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["trade_date", "code", "label_industry_excess",
            "label_calmar", "label_sortino", "maxdd", "dnvol"]
    df[cols].to_csv(out_path, index=False)
    print(f"\nwrote {out_path} ({len(df):,} rows)")

    if corr.loc["label_industry_excess", "label_calmar"] < 0.3:
        print("WARN: calmar label has weak corr with baseline — check distribution",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
