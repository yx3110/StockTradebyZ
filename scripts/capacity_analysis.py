"""P1.5: capacity analysis — what fraction of theoretical alpha survives at scale?

Lightweight (skips the 4000-line backtest engine) capacity decay model:

  1. Read forward_samples.csv produced by forward_test_tracker (already has
     daily Top-N selections + realized forward returns).
  2. For each (date, stock) pick, read ADV (avg daily traded value) over the
     trailing 20 trading days from daily_quotes.
  3. Constraint: per-stock fill = min(target_size, ADV_20d × adv_cap).
     If desired buy weight exceeds this, we either downsize (truncate) or
     skip (cash).
  4. Recompute portfolio return at each capital level using filled fractions.
  5. Compare net annualized return / Sharpe vs the no-cap baseline.

This answers production-relevant questions:
  - At what AUM does ng1.0.6 alpha collapse?
  - Which Top-N is the right operating point for 1亿 / 3亿 / 10亿?
  - Are the worst capacity offenders concentrated in micro-caps?

Output: reports/capacity/ng106_capacity_curve.md (table + interpretation).
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data_adapter" / "stock_data.db"
SAMPLES = ROOT / "reports" / "forward_test" / "forward_samples.csv"
OUT = ROOT / "reports" / "capacity" / "ng106_capacity_curve.md"

CAPITAL_LEVELS_YI = [1.0, 3.0, 10.0]   # 亿 RMB
ADV_CAP_DEFAULT = 0.05                  # 5% of 20d avg daily value


def load_picks(scoring_version: str, top_n: int) -> pd.DataFrame:
    if not SAMPLES.exists():
        raise SystemExit(f"missing {SAMPLES}; run forward_test_tracker scan first")
    df = pd.read_csv(SAMPLES, dtype={"stock_code": str})
    df = df[df["scoring_version"] == scoring_version].copy()
    df = df[df["top_n_rank"] <= top_n]
    df["report_date"] = pd.to_datetime(df["report_date"])
    return df


def load_adv_lookup(codes: list[str], dates: list[str]) -> pd.DataFrame:
    """Per-stock 20d avg daily traded value (close × volume × 100 shares/lot)."""
    if not codes or not dates:
        return pd.DataFrame()
    placeholders_codes = ",".join("?" * len(codes))
    min_date = min(dates)
    max_date = max(dates)
    conn = sqlite3.connect(str(DB), timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        rows = conn.execute(
            f"""
            SELECT s.code, dq.trade_date, dq.close * dq.volume * 100 AS dv
              FROM daily_quotes dq JOIN securities s ON s.id = dq.security_id
             WHERE s.code IN ({placeholders_codes})
               AND dq.trade_date BETWEEN date(?, '-40 days') AND ?
            """,
            list(codes) + [min_date, max_date],
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["code", "trade_date", "dv"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["code", "trade_date"])
    df["adv_20d"] = df.groupby("code")["dv"].rolling(20, min_periods=10).mean().reset_index(0, drop=True)
    return df[["code", "trade_date", "adv_20d"]].dropna()


def fill_at_capital(picks: pd.DataFrame, adv: pd.DataFrame, capital_yuan: float,
                    adv_cap: float, equal_weight: bool = True) -> pd.DataFrame:
    """For each (date, code), compute filled fraction of intended weight.

    Equal-weight assumption: each pick's target = capital / N where N = picks
    that day. Per-stock max fill = ADV × adv_cap (yuan). Filled weight =
    min(target, max_fill) / capital. Excess flows to cash.
    """
    p = picks.merge(
        adv.rename(columns={"trade_date": "report_date"}),
        left_on=["stock_code", "report_date"], right_on=["code", "report_date"],
        how="left",
    )
    n_per_day = p.groupby("report_date")["stock_code"].transform("size")
    p["target_value_yuan"] = capital_yuan / n_per_day
    p["max_fill_yuan"] = p["adv_20d"].fillna(0.0) * adv_cap
    p["filled_value_yuan"] = np.minimum(p["target_value_yuan"], p["max_fill_yuan"])
    p["fill_rate"] = p["filled_value_yuan"] / p["target_value_yuan"].replace(0, np.nan)
    p["filled_weight"] = p["filled_value_yuan"] / capital_yuan
    return p


def portfolio_return_curve(filled: pd.DataFrame, horizon: str = "10d") -> pd.DataFrame:
    """Sum of filled_weight × forward_ret per day → daily portfolio return."""
    ret_col = f"forward_ret_{horizon}"
    f = filled.dropna(subset=[ret_col, "filled_weight"]).copy()
    f["contribution"] = f["filled_weight"] * f[ret_col]
    daily = f.groupby("report_date").agg(
        port_ret=("contribution", "sum"),
        gross_invested=("filled_weight", "sum"),
        n_picks=("stock_code", "size"),
        avg_fill_rate=("fill_rate", "mean"),
    ).reset_index()
    return daily


def annualize(daily: pd.DataFrame, horizon: str = "10d") -> dict:
    """Convert N-day forward returns to ann return / Sharpe.
    Trades are non-overlapping at horizon stride; with overlapping (every
    day's report) we naively sum daily-mean returns × annual factor.
    """
    if daily.empty:
        return {"ann_ret": np.nan, "sharpe": np.nan, "n_days": 0,
                "avg_invested": np.nan, "avg_fill": np.nan}
    daily = daily.copy()
    # daily reports are overlapping; treat each day as one rebalance with
    # average fwd ret then divide by horizon for daily-equivalent rate.
    h = int(horizon.replace("d", ""))
    daily["daily_eq_ret"] = daily["port_ret"] / h
    mean = daily["daily_eq_ret"].mean()
    std = daily["daily_eq_ret"].std(ddof=1)
    return {
        "ann_ret": float((1 + mean) ** 252 - 1),
        "sharpe": float(mean / (std + 1e-12) * np.sqrt(252)),
        "n_days": int(len(daily)),
        "avg_invested": float(daily["gross_invested"].mean()),
        "avg_fill": float(daily["avg_fill_rate"].mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scoring-version", default="ng1.0.6")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--horizon", choices=["5d", "10d", "15d"], default="10d")
    ap.add_argument("--adv-cap", type=float, default=ADV_CAP_DEFAULT,
                    help="max fraction of 20d ADV per stock (default 0.05 = 5%%)")
    ap.add_argument("--capitals-yi", nargs="+", type=float, default=CAPITAL_LEVELS_YI,
                    help="capital levels in 亿 (yi=100M)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    print(f"[1/4] loading picks for {args.scoring_version} top {args.top_n}...")
    picks = load_picks(args.scoring_version, args.top_n)
    print(f"  → {len(picks):,} picks across {picks['report_date'].nunique()} days")
    if picks.empty:
        raise SystemExit("no picks for given scoring_version / top_n")

    codes = picks["stock_code"].unique().tolist()
    dates = picks["report_date"].dt.strftime("%Y-%m-%d").unique().tolist()
    print(f"[2/4] loading 20d ADV for {len(codes)} unique codes ({len(dates)} dates)...")
    adv = load_adv_lookup(codes, dates)
    print(f"  → {len(adv):,} (code, date) ADV rows")

    # No-cap baseline (∞ capital)
    print(f"[3/4] computing no-cap baseline + capacity curve...")
    base_filled = picks.copy()
    base_filled["filled_weight"] = 1.0 / base_filled.groupby("report_date")["stock_code"].transform("size")
    base_filled["fill_rate"] = 1.0
    base_daily = portfolio_return_curve(base_filled, args.horizon)
    base_metrics = annualize(base_daily, args.horizon)

    rows = []
    rows.append({"capital_yi": np.inf, **base_metrics})
    for cap_yi in sorted(args.capitals_yi):
        cap_yuan = cap_yi * 1e8
        f = fill_at_capital(picks, adv, cap_yuan, args.adv_cap)
        d = portfolio_return_curve(f, args.horizon)
        m = annualize(d, args.horizon)
        rows.append({"capital_yi": cap_yi, **m})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    print(f"[4/4] writing {args.out}...")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# 容量曲线 (P1.5) — {args.scoring_version} Top-{args.top_n} ({args.horizon})",
        "",
        f"**ADV cap**: {args.adv_cap:.0%} of 20-day avg daily traded value  ",
        f"**Capital levels**: ∞ (baseline), " + ", ".join(f"{c:.1f}亿" for c in args.capitals_yi),
        f"**Sample**: {len(picks):,} picks / {picks['report_date'].nunique()} days  ",
        "",
        "| 资金规模 | 平均仓位 | 平均 fill rate | 年化收益 | Sharpe | n_days |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        cap = "∞" if not np.isfinite(r["capital_yi"]) else f"{r['capital_yi']:.1f}亿"
        inv = "—" if pd.isna(r["avg_invested"]) else f"{r['avg_invested']:.1%}"
        fil = "—" if pd.isna(r["avg_fill"]) else f"{r['avg_fill']:.1%}"
        ar = "—" if pd.isna(r["ann_ret"]) else f"{r['ann_ret']:+.1%}"
        sh = "—" if pd.isna(r["sharpe"]) else f"{r['sharpe']:+.2f}"
        lines.append(f"| {cap} | {inv} | {fil} | {ar} | {sh} | {int(r['n_days'])} |")

    # decay relative to baseline
    base_ar = base_metrics["ann_ret"]
    if base_ar and not np.isnan(base_ar):
        lines += ["", "## 衰减率 (相对 ∞ 基线)", ""]
        lines.append("| 资金规模 | 年化收益保留率 |")
        lines.append("|---|---:|")
        for _, r in df.iterrows():
            if not np.isfinite(r["capital_yi"]):
                continue
            ratio = r["ann_ret"] / base_ar if base_ar != 0 else np.nan
            ratio_s = "—" if pd.isna(ratio) else f"{ratio:.1%}"
            lines.append(f"| {r['capital_yi']:.1f}亿 | {ratio_s} |")

    lines += [
        "",
        "## 解读",
        "",
        "- **fill rate < 100%** = 至少有一只票当日 ADV 不够买满 5%, 资金规模碰到容量天花板.",
        "- **avg_invested < 100%** = 部分仓位流失到现金, 总收益被稀释.",
        "- 拐点: 找年化收益保留率显著掉档的资金规模 — 那是 ng1.0.6 的容量天花板.",
        "- 缓解方法: 提高 top_n (摊薄单票) / 降低 adv_cap / 主动剔除微盘股.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
