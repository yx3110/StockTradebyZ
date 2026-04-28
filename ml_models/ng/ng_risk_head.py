"""P1.6 (ng2.2 Layer 1): risk-aware auxiliary GBDT heads.

Trains independent LightGBM regressors on the same ng101 feature cache as the
main alpha model, but with risk targets:

  - **maxdd_60d**: most-negative cumulative return over the next 60 trading
    days, normalized to [-1, 0]. Captures forward downside.
  - **vol_10d**: realized log-return volatility (annualized) over the next
    10 trading days. Captures forward dispersion.

These two predictions are NOT used for ranking on their own — they are
inputs to a Layer 2 utility-aware meta-learner (next milestone). Path B in
the feasibility study: keeps the main signal model untouched (avoids the
ng1.5.0 / ng1.0.7 regime-feature β explosion) while giving the meta-learner
real signal about per-stock risk profile.

Usage:
    python3 ml_models/ng/ng_risk_head.py \\
        --target maxdd_60d --start 2020-01-01 --end 2024-12-31 --seed 42
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent.parent
DB = ROOT / "data_adapter" / "stock_data.db"


def load_feature_cache(start: str, end: str, table: str = "ng101_feature_cache") -> pd.DataFrame:
    """Expand features_json into one column per factor."""
    conn = sqlite3.connect(str(DB), timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        df = pd.read_sql_query(
            f"SELECT trade_date, code, features_json FROM {table} "
            "WHERE trade_date BETWEEN ? AND ?",
            conn,
            params=[start, end],
        )
    finally:
        conn.close()
    if df.empty:
        return df
    feats = pd.DataFrame.from_records(df["features_json"].apply(json.loads).tolist())
    return pd.concat([df[["trade_date", "code"]].reset_index(drop=True), feats], axis=1)


def load_close_panel(start: str, end: str, buffer_days: int = 80) -> pd.DataFrame:
    """Pivot daily_quotes close → date × code, padded by buffer_days for forward labels."""
    end_buf = (pd.Timestamp(end) + pd.Timedelta(days=buffer_days)).strftime("%Y-%m-%d")
    conn = sqlite3.connect(str(DB), timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        df = pd.read_sql_query(
            "SELECT s.code, dq.trade_date, dq.close FROM daily_quotes dq "
            "JOIN securities s ON s.id = dq.security_id "
            "WHERE dq.trade_date BETWEEN ? AND ? AND s.type = 'A股'",
            conn,
            params=[start, end_buf],
        )
    finally:
        conn.close()
    return df.pivot(index="trade_date", columns="code", values="close").sort_index()


def compute_label_maxdd_60d(close_panel: pd.DataFrame, n: int = 60) -> pd.DataFrame:
    """Future N-day worst drawdown relative to entry close.
    label = min(close[t+1..t+n]) / close[t] - 1, clipped to [-1, 0].
    """
    fwd_min = close_panel.shift(-1).rolling(n, min_periods=10).min().shift(-(n - 1))
    label = (fwd_min / close_panel - 1.0).clip(lower=-1.0, upper=0.0)
    return label.stack().rename("y").reset_index()


def compute_label_vol_10d(close_panel: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Future N-day realized log-return vol (annualized)."""
    log_ret = np.log(close_panel).diff()
    fwd_vol = (
        log_ret.shift(-1).rolling(n, min_periods=5).std().shift(-(n - 1)) * np.sqrt(252)
    )
    return fwd_vol.stack().rename("y").reset_index()


def train_lgb(X: np.ndarray, y: np.ndarray, valid_mask: np.ndarray, seed: int = 42,
              params_override: Optional[dict] = None,
              min_iter: int = 0, patience: int = 100) -> lgb.Booster:
    """Train LightGBM regressor with optional custom params + min-iter guard.

    For noisy labels (e.g. forward 60d maxDD) lgb early-stopping at iter=1 if
    the first round overshoots: bump min_iter so the booster gets a chance to
    find a useful split, and use a lower learning rate.
    """
    params = dict(
        objective="regression", metric="rmse",
        learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        verbose=-1, seed=seed,
    )
    if params_override:
        params.update(params_override)
    train_set = lgb.Dataset(X[~valid_mask], y[~valid_mask])
    valid_set = lgb.Dataset(X[valid_mask], y[valid_mask], reference=train_set)
    callbacks = [
        lgb.early_stopping(stopping_rounds=patience, verbose=False, min_delta=0,
                           first_metric_only=True),
        lgb.log_evaluation(0),
    ]
    booster = lgb.train(
        params, train_set, num_boost_round=2000,
        valid_sets=[valid_set],
        callbacks=callbacks,
    )
    # Guard: if early-stop triggered before min_iter, force more rounds.
    if booster.best_iteration < min_iter:
        booster = lgb.train(
            params, train_set, num_boost_round=min_iter,
            valid_sets=[valid_set],
            callbacks=[lgb.log_evaluation(0)],
        )
    return booster


def evaluate_oos_ic(model: lgb.Booster, X_oos: np.ndarray, y_oos: np.ndarray,
                    dates_oos: np.ndarray) -> tuple[float, float]:
    """Per-day Spearman IC mean + ICIR on OOS."""
    preds = model.predict(X_oos)
    df = pd.DataFrame({"date": dates_oos, "pred": preds, "y": y_oos}).dropna()
    ics = []
    for _, g in df.groupby("date"):
        if len(g) >= 30:
            ic, _ = spearmanr(g["pred"], g["y"])
            if not np.isnan(ic):
                ics.append(float(ic))
    if not ics:
        return float("nan"), float("nan")
    arr = np.asarray(ics)
    return float(arr.mean()), float(arr.mean() / (arr.std(ddof=1) + 1e-9))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", choices=["maxdd_60d", "vol_10d"], required=True)
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--purge-days", type=int, default=15,
                    help="purge gap between train cutoff and OOS start")
    ap.add_argument("--oos-window-days", type=int, default=180,
                    help="number of trailing days reserved for OOS evaluation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="fast smoke run: subsample to 10%% rows for quick verification")
    args = ap.parse_args()

    print(f"[1/5] loading ng101 feature cache {args.start}~{args.end}...")
    feats = load_feature_cache(args.start, args.end)
    if feats.empty:
        print("ERROR: empty feature cache for window", file=sys.stderr)
        return 2
    print(f"  → {len(feats):,} rows, {len(feats.columns) - 2} factors")

    print(f"[2/5] loading close panel + computing label '{args.target}'...")
    close_panel = load_close_panel(args.start, args.end)
    if args.target == "maxdd_60d":
        labels = compute_label_maxdd_60d(close_panel, n=60)
    else:
        labels = compute_label_vol_10d(close_panel, n=10)
    labels.columns = ["trade_date", "code", "y"]

    j = feats.merge(labels, on=["trade_date", "code"]).dropna(subset=["y"])
    print(f"  joined: {len(j):,} rows; label range [{j['y'].min():.4f}, {j['y'].max():.4f}]")

    if args.smoke:
        j = j.sample(frac=0.10, random_state=args.seed).reset_index(drop=True)
        print(f"  SMOKE sub-sample: {len(j):,} rows")

    feat_cols = [c for c in j.columns if c not in ("trade_date", "code", "y")]
    X = j[feat_cols].astype(float).fillna(0.0).values
    y = j["y"].astype(float).values
    dates = pd.to_datetime(j["trade_date"]).values

    cutoff = pd.Timestamp(args.end) - pd.Timedelta(days=args.oos_window_days)
    purge_cutoff = cutoff - pd.Timedelta(days=args.purge_days)
    valid_mask = dates >= np.datetime64(cutoff)
    train_mask = dates < np.datetime64(purge_cutoff)
    n_train = int(train_mask.sum())
    n_valid = int(valid_mask.sum())
    if n_train < 1000 or n_valid < 100:
        print(f"ERROR: insufficient samples (train={n_train}, valid={n_valid})", file=sys.stderr)
        return 3
    print(f"[3/5] split: train={n_train:,}, oos_valid={n_valid:,} "
          f"(purge {args.purge_days}d)")

    print("[4/5] training LightGBM...")
    np.random.seed(args.seed)
    # Re-form valid_mask aligned with split sub-arrays for early-stop only
    fit_mask = train_mask | valid_mask
    X_fit = X[fit_mask]
    y_fit = y[fit_mask]
    valid_mask_in_fit = valid_mask[fit_mask]
    # maxdd_60d label is noisy (mean-reverting cumulative drawdown);
    # use lower lr + minimum iter guard to avoid degenerate 1-tree models.
    if args.target == "maxdd_60d":
        params_override = dict(learning_rate=0.02, num_leaves=31,
                               min_data_in_leaf=500)
        min_iter, patience = 50, 200
    else:
        params_override, min_iter, patience = None, 0, 100
    model = train_lgb(X_fit, y_fit, valid_mask_in_fit, seed=args.seed,
                      params_override=params_override,
                      min_iter=min_iter, patience=patience)
    print(f"  best_iter={model.best_iteration}, num_trees={model.num_trees()}")

    print("[5/5] OOS IC evaluation...")
    ic_mean, icir = evaluate_oos_ic(model, X[valid_mask], y[valid_mask], dates[valid_mask])
    print(f"  OOS daily IC mean: {ic_mean:+.4f}")
    print(f"  OOS daily ICIR:     {icir:+.4f}")

    out_path = args.out or str(
        ROOT / "ml_models" / "trained_models" / "ng" / f"risk_head_{args.target}_seed{args.seed}.pkl"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "feat_cols": feat_cols,
        "target": args.target,
        "seed": args.seed,
        "train_window": (args.start, args.end),
        "purge_days": args.purge_days,
        "oos_ic_mean": ic_mean,
        "oos_icir": icir,
    }, out_path)
    print(f"saved {out_path}")

    accept = abs(ic_mean) >= 0.10
    print(f"\nGate (|OOS IC| >= 0.10): {'PASS ✓' if accept else 'FAIL ✗'}")
    return 0 if accept else 1


if __name__ == "__main__":
    sys.exit(main())
