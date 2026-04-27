"""P1.6b: ng2.2 Layer 2 — utility-aware meta-learner.

Combines alpha (existing ng1.0.6 ML rank_score) with risk auxiliary heads
(P1.6 Layer 1: pred_maxdd_60d, pred_vol_10d) via cross-sectionally normalized
penalties:

    final_score = z(alpha) - λ_dd × z(pred_maxdd) - λ_vol × z(pred_vol)

where z() is per-day cross-sectional rank-z normalization (mean 0, std 1).
λ_dd and λ_vol are learned via a coarse 2-D grid search on training-period
realized portfolio Sharpe of Top-N picks. Validation on held-out period
reports OOS Sharpe / annualized return / MaxDD vs the baseline (λ=0).

Why grid-search not gradient: only ~1500 days × 10 picks = 15K observations,
and 2 hyperparameters. Linear-utility ranking + grid is robust to overfitting
and trivially interpretable.

Inputs:
  - reports/forward_test/forward_samples.csv  (alpha + realized forward rets)
  - ml_models/trained_models/ng/risk_head_{maxdd_60d,vol_10d}_seed42.pkl
  - ng101_feature_cache  (features for inference of risk heads)
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DB = ROOT / "data_adapter" / "stock_data.db"
SAMPLES = ROOT / "reports" / "forward_test" / "forward_samples.csv"
HEAD_DIR = ROOT / "ml_models" / "trained_models" / "ng"


def load_alpha_panel(scoring_version: str = "ng1.0.6") -> pd.DataFrame:
    df = pd.read_csv(SAMPLES, dtype={"stock_code": str})
    df = df[df["scoring_version"] == scoring_version].copy()
    df["report_date"] = pd.to_datetime(df["report_date"])
    return df.rename(columns={"stock_code": "code", "report_date": "trade_date"})


def load_risk_head(target: str, seed: int = 42) -> dict:
    path = HEAD_DIR / f"risk_head_{target}_seed{seed}.pkl"
    if not path.exists():
        raise SystemExit(f"missing {path} — run ng_risk_head first")
    return joblib.load(path)


def load_feature_panel(start: str, end: str) -> pd.DataFrame:
    """Bulk load ng101 feature_cache for risk-head inference."""
    conn = sqlite3.connect(str(DB), timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        df = pd.read_sql_query(
            "SELECT trade_date, code, features_json FROM ng101_feature_cache "
            "WHERE trade_date BETWEEN ? AND ?",
            conn, params=[start, end],
        )
    finally:
        conn.close()
    if df.empty:
        return df
    feats = pd.DataFrame.from_records(df["features_json"].apply(json.loads).tolist())
    out = pd.concat([df[["trade_date", "code"]].reset_index(drop=True), feats], axis=1)
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    return out


def predict_risk(feat_panel: pd.DataFrame, head: dict) -> pd.Series:
    cols = head["feat_cols"]
    X = feat_panel[cols].astype(float).fillna(0.0).values
    return pd.Series(head["model"].predict(X), index=feat_panel.index, name=head["target"])


def cross_section_z(s: pd.Series) -> pd.Series:
    """z-score within each .name group (assumes index is grouped externally)."""
    mu, sd = s.mean(), s.std(ddof=0)
    return (s - mu) / (sd + 1e-9)


def daily_cross_section_z(df: pd.DataFrame, col: str, by: str = "trade_date") -> pd.Series:
    return df.groupby(by)[col].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))


def evaluate_config(panel: pd.DataFrame, lam_dd: float, lam_vol: float,
                    top_n: int = 10, horizon: str = "10d") -> dict:
    """Apply (λ_dd, λ_vol), pick top-N per day, return portfolio metrics."""
    z_alpha = daily_cross_section_z(panel, "alpha_z")
    z_dd = daily_cross_section_z(panel, "risk_dd_z")
    z_vol = daily_cross_section_z(panel, "risk_vol_z")
    final = z_alpha - lam_dd * z_dd - lam_vol * z_vol

    panel = panel.assign(_final=final)
    ret_col = f"forward_ret_{horizon}"

    # Pick top-N per day, equal-weight
    picked = (panel.sort_values(["trade_date", "_final"], ascending=[True, False])
                   .groupby("trade_date").head(top_n))
    daily = picked.groupby("trade_date").agg(port_ret=(ret_col, "mean"),
                                              n=(ret_col, "size")).reset_index()
    daily = daily.dropna(subset=["port_ret"])
    if daily.empty:
        return {"sharpe": np.nan, "ann_ret": np.nan, "maxdd": np.nan,
                "n_days": 0, "lam_dd": lam_dd, "lam_vol": lam_vol}

    h = int(horizon.replace("d", ""))
    daily_eq = daily["port_ret"] / h
    mean = daily_eq.mean()
    std = daily_eq.std(ddof=1)
    cum = (1 + daily_eq).cumprod()
    rolling_max = cum.cummax()
    maxdd = float((cum / rolling_max - 1.0).min())

    return {
        "sharpe": float(mean / (std + 1e-12) * np.sqrt(252)),
        "ann_ret": float((1 + mean) ** 252 - 1),
        "maxdd": maxdd,
        "n_days": int(len(daily)),
        "lam_dd": lam_dd,
        "lam_vol": lam_vol,
    }


def grid_search(panel: pd.DataFrame, top_n: int, horizon: str,
                lam_grid: list[float]) -> pd.DataFrame:
    rows = []
    for lam_dd in lam_grid:
        for lam_vol in lam_grid:
            rows.append(evaluate_config(panel, lam_dd, lam_vol, top_n, horizon))
    return pd.DataFrame(rows)


def build_panel(scoring_version: str, start: str, end: str) -> pd.DataFrame:
    print(f"[1/4] loading alpha panel ({scoring_version}) {start}..{end}...")
    alpha = load_alpha_panel(scoring_version)
    alpha = alpha[(alpha["trade_date"] >= pd.Timestamp(start)) &
                  (alpha["trade_date"] <= pd.Timestamp(end))]
    print(f"  alpha rows: {len(alpha):,}")

    print("[2/4] loading risk-head pickles + feature panel...")
    h_dd = load_risk_head("maxdd_60d")
    h_vol = load_risk_head("vol_10d")
    feat = load_feature_panel(start, end)
    print(f"  feature rows: {len(feat):,}")

    print("[3/4] inferring pred_maxdd_60d / pred_vol_10d on full panel...")
    feat = feat.assign(
        risk_dd=predict_risk(feat, h_dd).values,
        risk_vol=predict_risk(feat, h_vol).values,
    )
    feat["risk_dd_z"] = daily_cross_section_z(feat, "risk_dd")
    feat["risk_vol_z"] = daily_cross_section_z(feat, "risk_vol")

    print("[4/4] joining alpha + risk on (trade_date, code)...")
    j = alpha.merge(feat[["trade_date", "code", "risk_dd", "risk_vol",
                          "risk_dd_z", "risk_vol_z"]],
                    on=["trade_date", "code"], how="left")
    # Use rank_score as alpha; fallback to composite
    j["alpha_z"] = j.get("rank_score", j.get("composite"))
    j["alpha_z"] = j.groupby("trade_date")["alpha_z"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9)
    )
    j = j.dropna(subset=["alpha_z", "risk_dd_z", "risk_vol_z"])
    print(f"  joined panel: {len(j):,} rows / {j['trade_date'].nunique()} days")
    return j


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scoring-version", default="ng1.0.6")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--train-end", default="2023-12-31",
                    help="train period for grid search; > becomes OOS")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--horizon", choices=["5d", "10d", "15d"], default="10d")
    ap.add_argument("--lam-grid", nargs="+", type=float,
                    default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    panel = build_panel(args.scoring_version, args.start, args.end)
    if panel.empty:
        print("ERROR: empty panel after join", file=sys.stderr)
        return 2

    train = panel[panel["trade_date"] <= pd.Timestamp(args.train_end)]
    oos = panel[panel["trade_date"] > pd.Timestamp(args.train_end)]
    print(f"\nTrain: {len(train):,} rows / {train['trade_date'].nunique()} days "
          f"(≤ {args.train_end})")
    print(f"OOS:   {len(oos):,} rows / {oos['trade_date'].nunique()} days "
          f"(> {args.train_end})")

    print(f"\n[grid] lam grid: {args.lam_grid} ({len(args.lam_grid)**2} configs)")
    train_results = grid_search(train, args.top_n, args.horizon, args.lam_grid)
    train_results = train_results.sort_values("sharpe", ascending=False).reset_index(drop=True)
    print("\nTrain grid (top 8 by Sharpe):")
    print(train_results.head(8).to_string(index=False))

    best = train_results.iloc[0]
    print(f"\nbest train config: λ_dd={best['lam_dd']}, λ_vol={best['lam_vol']}")
    print(f"  train sharpe={best['sharpe']:.3f}, ann={best['ann_ret']:+.2%}, "
          f"maxdd={best['maxdd']:+.2%}")

    print("\n[OOS] applying best config to OOS period...")
    base = evaluate_config(oos, 0.0, 0.0, args.top_n, args.horizon)
    tuned = evaluate_config(oos, best["lam_dd"], best["lam_vol"], args.top_n, args.horizon)

    print("\n========= OOS COMPARISON =========")
    print(f"baseline (λ=0):     sharpe={base['sharpe']:.3f}, ann={base['ann_ret']:+.2%}, maxdd={base['maxdd']:+.2%}")
    print(f"ng2.2 PoC (best λ): sharpe={tuned['sharpe']:.3f}, ann={tuned['ann_ret']:+.2%}, maxdd={tuned['maxdd']:+.2%}")
    print(f"Δ Sharpe: {tuned['sharpe'] - base['sharpe']:+.3f}")
    print(f"Δ MaxDD:  {tuned['maxdd'] - base['maxdd']:+.2%} (positive = improvement)")
    print(f"Δ Ann:    {tuned['ann_ret'] - base['ann_ret']:+.2%}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report = pd.DataFrame([
            {"period": "train_best", **{k: best[k] for k in ("lam_dd", "lam_vol", "sharpe", "ann_ret", "maxdd", "n_days")}},
            {"period": "oos_baseline", **base},
            {"period": "oos_tuned", **tuned},
        ])
        report.to_csv(out_path, index=False)
        print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
