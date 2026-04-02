#!/usr/bin/env python3
"""
Head-Focused 训练：只在头尾20%样本上训练 + truncation=5

核心改进:
  1. 每天只保留Top-20%和Bottom-20%样本 (去掉中间60%噪声)
  2. LambdaRank truncation_level=5 (只关注Top-5排序准确性)
  3. 10档relevance (比现有5档更细的头部区分)
  4. 同时训练MSE回归模型 (在filtered数据上)

用法:
    python3 ml_models/training/train_head_focused.py --version v4.8.5
"""

import sys, os, json, time, argparse, sqlite3, joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from datetime import datetime
from scipy.stats import rankdata, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / 'data_adapter' / 'stock_data.db'

VERSION_MAP = {
    'v4.7.5': 'v475', 'v4.8.1': 'v481', 'v4.8.2': 'v482', 'v4.8.3': 'v483',
    'v4.8.4': 'v484', 'v4.8.5': 'v485', 'v4.8.6': 'v486', 'v4.8.7': 'v487',
}


def load_scorer(version):
    vkey = VERSION_MAP[version]
    vkey_cls = {
        'v475': ('V475ProductionScorer', 'v475_production_scorer'),
        'v481': ('V481ProductionScorer', 'v481_production_scorer'),
        'v482': ('V482ProductionScorer', 'v482_production_scorer'),
        'v483': ('V483ProductionScorer', 'v483_production_scorer'),
        'v484': ('V484ProductionScorer', 'v484_production_scorer'),
        'v485': ('V485ProductionScorer', 'v485_production_scorer'),
        'v486': ('V486ProductionScorer', 'v486_production_scorer'),
        'v487': ('V487ProductionScorer', 'v487_production_scorer'),
    }
    cls_name, mod_name = vkey_cls[vkey]
    mod = __import__(f'ml_models.v39.{mod_name}', fromlist=[cls_name])
    return getattr(mod, cls_name)(model_type='small_data'), vkey


def load_data(start_date, end_date):
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute('PRAGMA busy_timeout=30000')
    t0 = time.time()
    dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT trade_date FROM v39_feature_cache "
        "WHERE trade_date >= ? AND trade_date <= ? ORDER BY trade_date",
        (start_date, end_date)
    ).fetchall()]
    CHUNK = 100
    frames = []
    for i in range(0, len(dates), CHUNK):
        chunk = dates[i:i + CHUNK]
        ph = ','.join(['?'] * len(chunk))
        df = pd.read_sql(f"""
            SELECT code, trade_date, features_json, label_10d, label_15d
            FROM v39_feature_cache
            WHERE trade_date IN ({ph}) AND label_10d IS NOT NULL
        """, conn, params=chunk)
        if not df.empty:
            frames.append(df)
    conn.close()
    data = pd.concat(frames, ignore_index=True)
    print(f"  {len(data):,} rows, {data['trade_date'].nunique()} dates ({time.time()-t0:.1f}s)")
    return data


def filter_head_tail(data, top_pct=0.20, bottom_pct=0.20):
    """Keep only top-20% and bottom-20% per day, remove middle 60%."""
    t0 = time.time()
    trade_dates = data['trade_date'].values
    label_10d = data['label_10d'].values
    keep_mask = np.zeros(len(data), dtype=bool)

    for date in np.unique(trade_dates):
        dmask = trade_dates == date
        idx = np.where(dmask)[0]
        n = len(idx)
        if n < 50:
            continue
        day_labels = label_10d[idx]
        ranks = rankdata(day_labels)
        pct = (ranks - 1) / (n - 1)
        # Keep top-20% and bottom-20%
        keep_mask[idx] = (pct >= (1 - top_pct)) | (pct <= bottom_pct)

    filtered = data[keep_mask].copy()
    print(f"  Head-tail filter: {len(data):,} → {len(filtered):,} ({len(filtered)/len(data)*100:.0f}%) [{time.time()-t0:.1f}s]")
    return filtered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default='v4.8.5', choices=list(VERSION_MAP.keys()))
    parser.add_argument('--start-date', default='2022-01-01')
    parser.add_argument('--end-date', default='2026-03-20')
    parser.add_argument('--top-pct', type=float, default=0.20)
    parser.add_argument('--purge-days', type=int, default=15)
    args = parser.parse_args()

    vkey = VERSION_MAP[args.version]
    model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / vkey

    print("=" * 70)
    print(f"  Head-Focused 训练 [{args.version}] — 头尾{int(args.top_pct*100)}% + truncation=5")
    print("=" * 70)

    scorer, vkey = load_scorer(args.version)
    feature_cols = scorer.feature_cols or []

    # Load full data
    print(f"\n[1/5] Loading data...")
    data = load_data(args.start_date, args.end_date)

    # Parse features
    print(f"\n[2/5] Preparing features...")
    t0 = time.time()
    parsed = data['features_json'].apply(json.loads)
    features_df = pd.DataFrame(parsed.tolist())
    for col in feature_cols:
        if col not in features_df.columns:
            features_df[col] = 0
    X_all = features_df[feature_cols].fillna(0).values
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  {X_all.shape} ({time.time()-t0:.1f}s)")

    trade_dates = data['trade_date'].values
    label_10d = data['label_10d'].values
    label_15d = data['label_15d'].fillna(0).values

    # Walk-forward split
    unique_dates = np.sort(np.unique(trade_dates))
    n_dates = len(unique_dates)
    train_end_idx = int(n_dates * 0.6)
    val_end_idx = int(n_dates * 0.8)
    train_end = unique_dates[train_end_idx]
    val_start = unique_dates[min(train_end_idx + args.purge_days, n_dates - 1)]
    val_end = unique_dates[val_end_idx]
    test_start = unique_dates[min(val_end_idx + args.purge_days, n_dates - 1)]

    train_mask = trade_dates <= train_end
    val_mask = (trade_dates >= val_start) & (trade_dates <= val_end)
    test_mask = trade_dates >= test_start

    # Filter head-tail for TRAINING only (val/test keep full for evaluation)
    print(f"\n[3/5] Filtering head-tail for training...")
    train_indices = np.where(train_mask)[0]
    train_dates_arr = trade_dates[train_mask]
    train_labels = label_10d[train_mask]

    # Per-day head-tail filter
    keep_train = np.zeros(len(train_indices), dtype=bool)
    for date in np.unique(train_dates_arr):
        dmask = train_dates_arr == date
        idx = np.where(dmask)[0]
        n = len(idx)
        if n < 50:
            continue
        day_labels = train_labels[idx]
        ranks = rankdata(day_labels)
        pct = (ranks - 1) / (n - 1)
        keep_train[idx] = (pct >= (1 - args.top_pct)) | (pct <= args.top_pct)

    X_train_full = X_all[train_mask]
    X_train_hf = X_train_full[keep_train]
    y_train_hf = train_labels[keep_train]
    dates_train_hf = train_dates_arr[keep_train]
    print(f"  Training: {len(X_train_full):,} → {len(X_train_hf):,} (head-tail {args.top_pct*100:.0f}%)")

    X_val = X_all[val_mask]
    y_val = label_10d[val_mask]
    dates_val = trade_dates[val_mask]

    X_test = X_all[test_mask]
    y_test = label_10d[test_mask]
    dates_test = trade_dates[test_mask]

    print(f"\n[4/5] Training models...")

    models = {}

    # Model A: LambdaRank with truncation=5 on head-tail data
    print(f"\n  --- LGB-LambdaRank (truncation=5, head-tail) ---")
    # Build relevance labels (10-grade for finer head distinction)
    def build_relevance_and_groups(y, dates, n_grades=10):
        relevance = np.zeros(len(y), dtype=np.int32)
        groups = []
        for d in np.unique(dates):
            dmask = dates == d
            n = dmask.sum()
            groups.append(n)
            if n >= 10:
                ranks = rankdata(y[dmask])
                pct = (ranks - 1) / (n - 1)
                relevance[dmask] = np.clip((pct * n_grades).astype(int), 0, n_grades - 1)
            else:
                relevance[dmask] = n_grades // 2
        return relevance, groups

    rel_train, grp_train = build_relevance_and_groups(y_train_hf, dates_train_hf, n_grades=10)
    rel_val, grp_val = build_relevance_and_groups(y_val, dates_val, n_grades=10)

    rank_params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'eval_at': [5, 10],
        'lambdarank_truncation_level': 5,   # ← KEY: focus on top-5
        'lambdarank_norm': True,
        'num_leaves': 31,
        'learning_rate': 0.03,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_data_in_leaf': 100,
        'reg_alpha': 0.5,
        'reg_lambda': 2.0,
        'verbose': -1,
    }

    ds_train = lgb.Dataset(X_train_hf, label=rel_train, group=grp_train)
    ds_val = lgb.Dataset(X_val, label=rel_val, group=grp_val, reference=ds_train)

    rank_model = lgb.train(
        rank_params, ds_train,
        num_boost_round=1000,
        valid_sets=[ds_val],
        callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(100)],
    )
    models['lgb_rank_hf'] = rank_model
    print(f"  LambdaRank(trunc=5) trained: {rank_model.num_trees()} trees")

    # Model B: MSE regression on head-tail data (for comparison)
    print(f"\n  --- LGB-MSE (head-tail) ---")
    mse_params = {
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 50,
        'verbose': -1,
    }
    ds_mse_train = lgb.Dataset(X_train_hf, label=y_train_hf)
    ds_mse_val = lgb.Dataset(X_val, label=y_val, reference=ds_mse_train)

    mse_model = lgb.train(
        mse_params, ds_mse_train,
        num_boost_round=1000,
        valid_sets=[ds_mse_val],
        callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(100)],
    )
    models['lgb_mse_hf'] = mse_model

    # Model C: LambdaRank with truncation=5 on FULL data (ablation: is head-tail filtering necessary?)
    print(f"\n  --- LGB-LambdaRank (truncation=5, FULL data, ablation) ---")
    rel_train_full, grp_train_full = build_relevance_and_groups(
        label_10d[train_mask], trade_dates[train_mask], n_grades=10)

    ds_train_full = lgb.Dataset(X_train_full, label=rel_train_full, group=grp_train_full)
    ds_val_full = lgb.Dataset(X_val, label=rel_val, group=grp_val, reference=ds_train_full)

    rank_model_full = lgb.train(
        rank_params, ds_train_full,
        num_boost_round=1000,
        valid_sets=[ds_val_full],
        callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(100)],
    )
    models['lgb_rank_full'] = rank_model_full

    # Evaluate all models
    print(f"\n[5/5] Evaluating on test set...")

    # Also get MSE ensemble predictions for comparison
    mse_ensemble_pred = np.zeros(len(X_test))
    tw = scorer.weights.get('label_10d', {})
    total_w = 0
    for name, m in scorer.models['10d'].items():
        try:
            if name == 'xgb':
                import xgboost as xgb_lib
                p = m.predict(xgb_lib.DMatrix(X_test))
            else:
                p = m.predict(X_test)
            w = tw.get(name, 0.2)
            mse_ensemble_pred += w * p
            total_w += w
        except Exception: pass
    if total_w > 0:
        mse_ensemble_pred /= total_w

    # Per-model predictions
    pred_rank_hf = rank_model.predict(X_test)
    pred_mse_hf = mse_model.predict(X_test)
    pred_rank_full = rank_model_full.predict(X_test)

    # Daily top-10 evaluation
    unique_test_dates = np.unique(dates_test)
    results = {
        'mse_ensemble': [],
        'rank_hf_top10': [],
        'mse_hf_top10': [],
        'rank_full_top10': [],
        'widen30_rank_hf': [],
    }

    for d in unique_test_dates:
        dmask = dates_test == d
        if dmask.sum() < 50:
            continue
        d_y = y_test[dmask]
        d_mse = mse_ensemble_pred[dmask]
        d_rank_hf = pred_rank_hf[dmask]
        d_mse_hf = pred_mse_hf[dmask]
        d_rank_full = pred_rank_full[dmask]

        # MSE ensemble Top-10
        idx = np.argsort(d_mse)[-10:]
        results['mse_ensemble'].extend(d_y[idx])

        # LambdaRank head-focused Top-10
        idx = np.argsort(d_rank_hf)[-10:]
        results['rank_hf_top10'].extend(d_y[idx])

        # MSE head-focused Top-10
        idx = np.argsort(d_mse_hf)[-10:]
        results['mse_hf_top10'].extend(d_y[idx])

        # LambdaRank full-data Top-10
        idx = np.argsort(d_rank_full)[-10:]
        results['rank_full_top10'].extend(d_y[idx])

        # Widen-then-Concentrate: MSE ensemble Top-30 → rank_hf Top-10
        pool = np.argsort(d_mse)[-30:]
        pool_scores = d_rank_hf[pool]
        top10 = pool[np.argsort(pool_scores)[-10:]]
        results['widen30_rank_hf'].extend(d_y[top10])

    print(f"\n{'='*90}")
    print(f"  测试集对比 ({unique_test_dates[0]} ~ {unique_test_dates[-1]})")
    print(f"{'='*90}")
    print(f"{'策略':>40} | {'n':>5} | {'10d均收':>8} | {'胜率':>6} | {'PF':>6}")
    print("-" * 90)

    for name, label in [
        ('mse_ensemble', 'MSE Ensemble Top-10 (基线)'),
        ('rank_full_top10', 'LambdaRank trunc=5 全量 Top-10'),
        ('rank_hf_top10', 'LambdaRank trunc=5 头尾20% Top-10'),
        ('mse_hf_top10', 'MSE 头尾20% Top-10'),
        ('widen30_rank_hf', 'MSE30 → LambdaRank-HF Top-10'),
    ]:
        rets = np.array(results[name])
        if len(rets) < 10:
            continue
        pf = rets[rets > 0].sum() / max(abs(rets[rets < 0].sum()), 1e-8)
        print(f"  {label:>38} | {len(rets):>5} | {rets.mean():>+8.2%} | {(rets>0).mean()*100:>5.1f}% | {pf:>6.3f}")

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = model_dir / f'head_focused_{vkey}_{timestamp}.pkl'
    save_data = {
        'models': models,
        'version': args.version,
        'vkey': vkey,
        'feature_cols': feature_cols,
        'head_tail_pct': args.top_pct,
        'truncation_level': 5,
        'n_grades': 10,
        'timestamp': timestamp,
    }
    joblib.dump(save_data, save_path, compress=3)
    print(f"\n  Saved: {save_path} ({save_path.stat().st_size / 1024:.0f} KB)")
    print("=" * 70)


if __name__ == '__main__':
    main()
