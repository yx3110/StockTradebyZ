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
from typing import Optional

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


LABEL_HORIZON_TD = {"maxdd_60d": 60, "vol_10d": 10}  # 标签前瞻交易日数


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", choices=["maxdd_60d", "vol_10d"], required=True)
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--oos-window-td", type=int, default=120,
                    help="每折 OOS 窗口的交易日数")
    ap.add_argument("--folds", type=int, default=3,
                    help="滚动 WF 折数 (从数据尾部向前排)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="fast smoke run: subsample to 10%% rows for quick verification")
    args = ap.parse_args()

    # ── 2026-07-11 协议修复 (原版两处泄漏使 gate 数字偏乐观) ──
    # 1) 早停曾直接用 OOS 集选树数 (vol 头 177 树是在评估集上挑的) →
    #    改为训练段尾部切内部验证片 (与内部训练之间同样留 purge)
    # 2) purge 曾用 15 个«日历日», 而 maxdd_60d 标签前瞻 60 个«交易日»≈84 日历日,
    #    训练尾部标签路径伸进 OOS 达 68 日历日 → 改为按交易日 purge = horizon + 5
    # 3) 单一 trailing 窗口 → K 折滚动 WF, gate 看均值
    horizon_td = LABEL_HORIZON_TD[args.target]
    purge_td = horizon_td + 5
    inner_val_td = 40  # 早停用内部验证片 (训练段尾部 40 个交易日)

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

    # 交易日历: 用样本中实际出现的日期 (与 feature cache 对齐)
    cal = np.sort(np.unique(dates))
    W, P = args.oos_window_td, purge_td
    if args.target == "maxdd_60d":
        params_override = dict(learning_rate=0.02, num_leaves=31,
                               min_data_in_leaf=500)
        min_iter, patience = 50, 200
    else:
        params_override, min_iter, patience = None, 0, 100

    print(f"[3/5] {args.folds}-fold WF: oos={W}td/fold, purge={P}td "
          f"(=horizon {horizon_td}+5), inner-val={inner_val_td}td")
    np.random.seed(args.seed)
    fold_results = []
    last_model = None
    for k in range(args.folds):
        oos_end_i = len(cal) - k * W          # exclusive
        oos_start_i = oos_end_i - W
        train_end_i = oos_start_i - P          # exclusive; train dates < cal[train_end_i]
        inner_val_start_i = train_end_i - inner_val_td
        inner_train_end_i = inner_val_start_i - P
        if inner_train_end_i < 200:
            print(f"  fold {k}: 训练段不足 ({inner_train_end_i} 个交易日), 停止排折")
            break
        oos_mask = (dates >= cal[oos_start_i]) & (dates < cal[oos_end_i - 1] + np.timedelta64(1, 'D'))
        inner_train_mask = dates < cal[inner_train_end_i]
        inner_val_mask = (dates >= cal[inner_val_start_i]) & (dates < cal[train_end_i])
        fit_mask = inner_train_mask | inner_val_mask
        model = train_lgb(X[fit_mask], y[fit_mask], inner_val_mask[fit_mask],
                          seed=args.seed, params_override=params_override,
                          min_iter=min_iter, patience=patience)
        ic_mean, icir = evaluate_oos_ic(model, X[oos_mask], y[oos_mask], dates[oos_mask])
        oos_lo = pd.Timestamp(cal[oos_start_i]).date()
        oos_hi = pd.Timestamp(cal[oos_end_i - 1]).date()
        print(f"  fold {k}: OOS {oos_lo}~{oos_hi} | train={int(inner_train_mask.sum()):,} "
              f"trees={model.num_trees()}(best {model.best_iteration}) | "
              f"IC={ic_mean:+.4f} ICIR={icir:+.4f}")
        fold_results.append(dict(fold=k, oos_start=str(oos_lo), oos_end=str(oos_hi),
                                 ic=ic_mean, icir=icir,
                                 trees=model.num_trees(), best_iter=model.best_iteration))
        if k == 0:
            last_model = model  # 最新折的模型 = 候选生产模型

    if not fold_results:
        print("ERROR: no valid folds", file=sys.stderr)
        return 3

    ics = np.array([f["ic"] for f in fold_results])
    print(f"[4/5] 汇总: mean IC={ics.mean():+.4f} (±{ics.std(ddof=0):.4f}, "
          f"{len(ics)} folds), min={ics.min():+.4f}")

    print("[5/5] saving...")
    out_path = args.out or str(
        ROOT / "ml_models" / "trained_models" / "ng" / f"risk_head_{args.target}_seed{args.seed}.pkl"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": last_model,
        "feat_cols": feat_cols,
        "target": args.target,
        "seed": args.seed,
        "train_window": (args.start, args.end),
        "protocol": "2026-07-11-fixed",  # 内部验证早停 + 交易日 purge + K 折
        "purge_td": P,
        "oos_window_td": W,
        "fold_results": fold_results,
        "oos_ic_mean": float(ics.mean()),
        "oos_ic_min": float(ics.min()),
        "oos_icir": float(np.mean([f["icir"] for f in fold_results])),
    }, out_path)
    print(f"saved {out_path}")

    accept = abs(ics.mean()) >= 0.10
    print(f"\nGate (|mean OOS IC| >= 0.10): {'PASS ✓' if accept else 'FAIL ✗'}")
    return 0 if accept else 1


if __name__ == "__main__":
    sys.exit(main())
