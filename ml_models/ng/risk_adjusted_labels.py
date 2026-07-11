"""P1.3 Step A/B: risk-adjusted label generators (Calmar / Sortino) — 4 horizons.

Outputs a CSV keyed by (trade_date, code) with three labels per horizon:
  - label_industry_excess_{H}d   ← baseline (matches ng1.0.1 / ng1.6.x)
  - label_calmar_{H}d            ← industry_excess / max(|future_maxdd_{H}d|, floor)
  - label_sortino_{H}d           ← industry_excess / future_downside_vol_{H}d
for H ∈ {3, 5, 10, 15} (configurable via --horizons).

Step A artefact mode (default): just emit the CSV for inspection.
Step B integration: trainer reads this CSV via --label-mode {calmar,sortino}.

Usage:
  python3 ml_models/ng/risk_adjusted_labels.py \\
      --start 2018-01-01 --end 2026-04-25 \\
      --horizons 3,5,10,15 \\
      --out reports/labels/ng162_labels.csv
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


def load_close_panel(start: str, end: str, buffer_days: int = 60) -> pd.DataFrame:
    """后复权 close 面板 (2026-07-12 修复: 原始价跨除权日会产生假 maxdd/假负收益标签).

    adj_factor 是阶跃函数 (仅除权日变化), 按股票 ffill/bfill 补缺行是正确语义;
    全程无 adj 的股票回退原始价 (整列一致, 不混用)。
    """
    end_buf = (pd.Timestamp(end) + pd.Timedelta(days=buffer_days)).strftime("%Y-%m-%d")
    conn = sqlite3.connect(str(DB), timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        df = pd.read_sql_query(
            "SELECT s.code, dq.trade_date, dq.close, dq.adj_factor FROM daily_quotes dq "
            "JOIN securities s ON s.id = dq.security_id "
            "WHERE dq.trade_date BETWEEN ? AND ? AND s.type = 'A股'",
            conn, params=[start, end_buf],
        )
    finally:
        conn.close()
    close = df.pivot(index="trade_date", columns="code", values="close").sort_index()
    adj = df.pivot(index="trade_date", columns="code", values="adj_factor").sort_index()
    adj = adj.where(adj != 1.0)          # 1.0 是历史占位符, 视同缺失
    adj = adj.ffill().bfill().fillna(1.0)  # 阶跃函数补缺; 全缺列 → 1.0 (原始价)
    return close * adj


def load_industry_lookup() -> pd.DataFrame:
    conn = sqlite3.connect(str(DB), timeout=30)
    try:
        df = pd.read_sql_query(
            "SELECT code, industry FROM securities WHERE type = 'A股'", conn,
        )
    finally:
        conn.close()
    return df


def compute_future_ret(close: pd.DataFrame, n: int) -> pd.DataFrame:
    """Forward N-day simple return (close[t+n]/close[t] - 1)."""
    return (close.shift(-n) / close - 1.0).stack().rename("ret").reset_index()


def compute_future_maxdd(close: pd.DataFrame, n: int) -> pd.DataFrame:
    """Worst path drawdown over [t+1, t+n], ∈ [-1, 0].

    fwd_min = rolling min of close over the next n days starting from t+1.
    dd = fwd_min / close[t] - 1.
    """
    fwd_min = close.shift(-1).rolling(n, min_periods=max(3, n // 2)).min().shift(-(n - 1))
    dd = (fwd_min / close - 1.0).clip(lower=-1.0, upper=0.0)
    return dd.stack().rename("maxdd").reset_index()


def compute_future_dnvol(close: pd.DataFrame, n: int) -> pd.DataFrame:
    """Downside vol over [t+1, t+n] = std of negative log-rets only, annualized."""
    log_ret = np.log(close).diff()
    neg = log_ret.where(log_ret < 0)
    fwd_dn = (
        neg.shift(-1).rolling(n, min_periods=max(3, n // 2)).std().shift(-(n - 1))
        * np.sqrt(252)
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


def build_labels_for_horizon(close: pd.DataFrame, industry: pd.DataFrame,
                             horizon: int, calmar_floor: float,
                             sortino_floor: float) -> pd.DataFrame:
    """Build (industry_excess, calmar, sortino) labels for one horizon."""
    rets = compute_future_ret(close, horizon)
    rets.columns = ["trade_date", "code", "ret"]
    dds = compute_future_maxdd(close, horizon)
    dds.columns = ["trade_date", "code", "maxdd"]
    dnv = compute_future_dnvol(close, horizon)
    dnv.columns = ["trade_date", "code", "dnvol"]

    excess = industry_excess(rets, industry)
    df = excess.merge(dds, on=["trade_date", "code"], how="left")
    df = df.merge(dnv, on=["trade_date", "code"], how="left")
    df = df.dropna(subset=["label_industry_excess", "maxdd"])

    df["label_calmar"] = df["label_industry_excess"] / np.maximum(
        np.abs(df["maxdd"]), calmar_floor)
    df["label_sortino"] = df["label_industry_excess"] / np.maximum(
        df["dnvol"].fillna(sortino_floor), sortino_floor)
    df["label_calmar"] = winsorize(df["label_calmar"])
    df["label_sortino"] = winsorize(df["label_sortino"])

    suffix = f"{horizon}d"
    return df.rename(columns={
        "label_industry_excess": f"label_industry_excess_{suffix}",
        "label_calmar": f"label_calmar_{suffix}",
        "label_sortino": f"label_sortino_{suffix}",
        "maxdd": f"maxdd_{suffix}",
        "dnvol": f"dnvol_{suffix}",
        "ret": f"ret_{suffix}",
    })[[
        "trade_date", "code",
        f"label_industry_excess_{suffix}",
        f"label_calmar_{suffix}",
        f"label_sortino_{suffix}",
        f"maxdd_{suffix}",
        f"dnvol_{suffix}",
    ]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--horizons", default="3,5,10,15",
                    help="comma-separated horizons in days")
    ap.add_argument("--calmar-floor", type=float, default=0.05,
                    help="floor for |maxdd| denominator")
    ap.add_argument("--sortino-floor", type=float, default=0.10,
                    help="floor for downside vol (annual) denominator")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    horizons = [int(h.strip()) for h in args.horizons.split(",")]

    print(f"[1/{2 + len(horizons)}] loading close panel {args.start}~{args.end}...")
    close = load_close_panel(args.start, args.end)
    print(f"  → {close.shape[0]} rows × {close.shape[1]} codes")

    print(f"[2/{2 + len(horizons)}] loading industry lookup...")
    industry = load_industry_lookup()
    print(f"  → {len(industry)} codes with industry")

    merged = None
    for i, h in enumerate(horizons, start=3):
        print(f"[{i}/{2 + len(horizons)}] horizon={h}d: forward ret/maxdd/dnvol + labels...")
        df_h = build_labels_for_horizon(close, industry, h,
                                        args.calmar_floor, args.sortino_floor)
        if merged is None:
            merged = df_h
        else:
            merged = merged.merge(df_h, on=["trade_date", "code"], how="outer")

    print(f"\n[summary] {len(merged):,} (trade_date, code) rows × {merged.shape[1]} cols")

    print("\nlabel distribution per horizon (post-winsorize):")
    for h in horizons:
        cols = [f"label_industry_excess_{h}d", f"label_calmar_{h}d", f"label_sortino_{h}d"]
        avail = [c for c in cols if c in merged.columns]
        print(f"\n  horizon={h}d:")
        print(merged[avail].describe(percentiles=[0.05, 0.5, 0.95]).round(4))

    print("\nlabel correlation per horizon (industry_excess vs calmar / sortino):")
    warn_count = 0
    for h in horizons:
        ie = f"label_industry_excess_{h}d"
        ca = f"label_calmar_{h}d"
        so = f"label_sortino_{h}d"
        if ie in merged.columns and ca in merged.columns:
            corr_ca = merged[[ie, ca]].corr().iloc[0, 1]
            corr_so = merged[[ie, so]].corr().iloc[0, 1]
            flag_ca = "OK" if corr_ca >= 0.3 else "WARN"
            flag_so = "OK" if corr_so >= 0.3 else "WARN"
            print(f"  {h:>2}d: corr(IE, calmar)={corr_ca:+.3f} [{flag_ca}], "
                  f"corr(IE, sortino)={corr_so:+.3f} [{flag_so}]")
            if corr_ca < 0.3:
                warn_count += 1
            if corr_so < 0.3:
                warn_count += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"\nwrote {out_path} ({len(merged):,} rows × {merged.shape[1]} cols)")

    if warn_count:
        print(f"\nWARN: {warn_count} (horizon × label) combos have corr < 0.3 vs baseline",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
