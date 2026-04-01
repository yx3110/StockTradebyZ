#!/usr/bin/env python3
"""
二阶段头部精筛模型 (Head Refiner) 训练脚本

目标: 在Stage 1的ensemble输出中精确识别真正的Top-10股票
方法: 用Stage 1各子模型的预测值作为meta-features, 训练LightGBM分类器

输入特征:
  - 6个子模型(lgb/xgb/cb/rf/hgb/lgb_rank)对10d的原始预测值
  - Meta-features: consensus_count, pred_mean, pred_std, pred_min, pred_max, pred_range
  - 原始stock features中最重要的20个 (降维避免过拟合)

标签: is_true_top10 = 当天label_10d排名前2% (约100只)

用法:
    python3 ml_models/training/train_head_refiner.py
    python3 ml_models/training/train_head_refiner.py --top-pct 0.02 --start-date 2022-01-01
"""

import sys, os, json, time, argparse, sqlite3, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import rankdata, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / 'data_adapter' / 'stock_data.db'

VERSION_MAP = {
    'v4.7.5': ('v475', 'V475ProductionScorer', 'v475_production_scorer'),
    'v4.8.1': ('v481', 'V481ProductionScorer', 'v481_production_scorer'),
    'v4.8.2': ('v482', 'V482ProductionScorer', 'v482_production_scorer'),
    'v4.8.3': ('v483', 'V483ProductionScorer', 'v483_production_scorer'),
    'v4.8.4': ('v484', 'V484ProductionScorer', 'v484_production_scorer'),
    'v4.8.5': ('v485', 'V485ProductionScorer', 'v485_production_scorer'),
    'v4.8.6': ('v486', 'V486ProductionScorer', 'v486_production_scorer'),
    'v4.8.7': ('v487', 'V487ProductionScorer', 'v487_production_scorer'),
}


def load_scorer(version):
    vkey, cls_name, mod_name = VERSION_MAP[version]
    mod = __import__(f'ml_models.v39.{mod_name}', fromlist=[cls_name])
    cls = getattr(mod, cls_name)
    scorer = cls(model_type='small_data')
    model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / vkey
    model_dir.mkdir(parents=True, exist_ok=True)
    return scorer, vkey, model_dir


def load_feature_cache(start_date='2022-01-01', end_date='2026-03-20'):
    """Load v39_feature_cache with labels."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute('PRAGMA busy_timeout=30000')

    print(f"Loading feature cache {start_date} ~ {end_date}...")
    t0 = time.time()

    # Load in chunks
    CHUNK = 100
    dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT trade_date FROM v39_feature_cache "
        "WHERE trade_date >= ? AND trade_date <= ? ORDER BY trade_date",
        (start_date, end_date)
    ).fetchall()]

    frames = []
    for i in range(0, len(dates), CHUNK):
        chunk = dates[i:i + CHUNK]
        ph = ','.join(['?'] * len(chunk))
        df = pd.read_sql(f"""
            SELECT code, trade_date, features_json, label_10d,
                   market_return_20d, market_volatility_20d, market_up_ratio_20d
            FROM v39_feature_cache
            WHERE trade_date IN ({ph})
              AND label_10d IS NOT NULL
        """, conn, params=chunk)
        if not df.empty:
            frames.append(df)

    conn.close()

    if not frames:
        print("ERROR: No data loaded")
        sys.exit(1)

    data = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(data)} rows, {data['trade_date'].nunique()} dates ({time.time()-t0:.1f}s)")
    return data


def parse_features(data):
    """Parse JSON features and extract top stock features."""
    print("Parsing features...")
    t0 = time.time()

    parsed = data['features_json'].apply(json.loads)
    features_df = pd.DataFrame(parsed.tolist())
    features_df['code'] = data['code'].values
    features_df['trade_date'] = data['trade_date'].values
    features_df['label_10d'] = data['label_10d'].values

    # Market features
    for col in ['market_return_20d', 'market_volatility_20d', 'market_up_ratio_20d']:
        if col in data.columns:
            features_df[col] = data[col].values

    print(f"  {len(features_df.columns)} feature columns ({time.time()-t0:.1f}s)")
    return features_df


def compute_stage1_meta_features(features_df, scorer):
    """Run Stage 1 model to generate per-model predictions as meta-features.

    For each date, run all base models on the feature matrix and collect:
    - Per-model raw predictions (6 values per stock)
    - Consensus count (how many models put stock in Top-20)
    - pred_mean, pred_std, pred_min, pred_max, pred_range
    - Per-model ranks (relative position)
    """
    print("Computing Stage 1 meta-features...")
    t0 = time.time()

    dates = sorted(features_df['trade_date'].unique())
    exclude_cols = {'code', 'trade_date', 'label_10d',
                    'market_return_20d', 'market_volatility_20d', 'market_up_ratio_20d'}

    # Get feature columns in model order
    feature_cols = scorer.feature_cols
    if not feature_cols:
        feature_cols = [c for c in features_df.columns if c not in exclude_cols]

    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in features_df.columns:
            features_df[col] = 0

    # Get model names for 10d target
    model_names = list(scorer.models.get('10d', {}).keys())
    if not model_names:
        print("ERROR: No 10d models found")
        sys.exit(1)
    print(f"  Base models: {model_names}")

    # Process all data at once (features are already loaded)
    X = features_df[feature_cols].fillna(0).values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    n = len(X)

    # Get predictions from each base model
    preds = {}
    for name, model in scorer.models['10d'].items():
        try:
            if name == 'xgb':
                import xgboost as xgb_lib
                preds[name] = model.predict(xgb_lib.DMatrix(X))
            else:
                preds[name] = model.predict(X)
            print(f"    {name}: predicted {len(preds[name])} stocks")
        except Exception as e:
            print(f"    {name}: FAILED ({e})")

    if not preds:
        print("ERROR: No model predictions")
        sys.exit(1)

    # Rescale rank models (same as V481)
    regression_names = [n for n in preds if n not in ('lgb_rank', 'lgb_listnet')]
    rank_names = [n for n in preds if n in ('lgb_rank', 'lgb_listnet')]
    if regression_names and rank_names:
        reg_means = [np.mean(preds[n]) for n in regression_names]
        reg_stds = [max(np.std(preds[n]), 1e-8) for n in regression_names]
        t_mean, t_std = np.mean(reg_means), np.mean(reg_stds)
        for rn in rank_names:
            rp = preds[rn]
            rp_std = max(np.std(rp), 1e-8)
            preds[rn] = (rp - np.mean(rp)) / rp_std * t_std + t_mean

    # Build meta-features per stock
    pred_matrix = np.column_stack([preds[name] for name in model_names if name in preds])
    available_names = [name for name in model_names if name in preds]
    n_models = pred_matrix.shape[1]

    # Global meta-features
    meta = pd.DataFrame(index=range(n))
    for j, name in enumerate(available_names):
        meta[f'pred_{name}'] = pred_matrix[:, j]

    meta['pred_mean'] = np.mean(pred_matrix, axis=1)
    meta['pred_std'] = np.std(pred_matrix, axis=1)
    meta['pred_min'] = np.min(pred_matrix, axis=1)
    meta['pred_max'] = np.max(pred_matrix, axis=1)
    meta['pred_range'] = meta['pred_max'] - meta['pred_min']

    # Prediction Sharpe (confidence)
    meta['pred_sharpe'] = meta['pred_mean'] / np.maximum(meta['pred_std'], 1e-8)

    # Per-date consensus count and rank features
    trade_dates = features_df['trade_date'].values
    consensus_counts = np.zeros(n)
    rank_in_day = np.zeros(n)  # percentile rank within daily cross-section

    TOP_K = 20
    unique_dates = np.unique(trade_dates)
    for date in unique_dates:
        mask = trade_dates == date
        idx = np.where(mask)[0]
        n_day = len(idx)
        if n_day < 20:
            continue

        day_preds = pred_matrix[idx]

        # Consensus: count models with stock in their Top-K
        for j in range(n_models):
            ranks = rankdata(-day_preds[:, j], method='ordinal')
            consensus_counts[idx] += (ranks <= TOP_K).astype(float)

        # Rank within day (by pred_mean)
        day_mean = meta['pred_mean'].values[idx]
        rank_in_day[idx] = rankdata(-day_mean, method='average') / n_day

    meta['consensus_count'] = consensus_counts
    meta['rank_in_day'] = rank_in_day

    # Cross-model agreement features
    # Kendall's W approximation: variance of ranks across models
    rank_matrix = np.zeros_like(pred_matrix)
    for date in unique_dates:
        mask = trade_dates == date
        idx = np.where(mask)[0]
        if len(idx) < 20:
            continue
        for j in range(n_models):
            rank_matrix[idx, j] = rankdata(-pred_matrix[idx, j], method='average')
    meta['rank_std'] = np.std(rank_matrix, axis=1)
    meta['rank_mean'] = np.mean(rank_matrix, axis=1)

    elapsed = time.time() - t0
    print(f"  Meta-features computed: {meta.shape[1]} columns ({elapsed:.1f}s)")
    return meta


def compute_labels(features_df, top_pct=0.02):
    """Compute is_true_top10 binary label.

    top_pct=0.02 means top 2% per day (about 100 stocks out of 5000).
    For the head refiner we want to identify true top performers.
    """
    print(f"Computing is_true_top labels (top {top_pct*100:.1f}% per day)...")
    labels = np.zeros(len(features_df), dtype=np.int32)
    trade_dates = features_df['trade_date'].values
    label_10d = features_df['label_10d'].values

    pos_count = 0
    for date in np.unique(trade_dates):
        mask = trade_dates == date
        idx = np.where(mask)[0]
        n_day = len(idx)
        if n_day < 50:
            continue

        day_labels = label_10d[idx]
        top_n = max(1, int(n_day * top_pct))
        threshold = np.partition(day_labels, -top_n)[-top_n]
        labels[idx] = (day_labels >= threshold).astype(np.int32)
        pos_count += np.sum(labels[idx])

    total = np.sum(labels >= 0)
    print(f"  Positive: {pos_count} ({pos_count/total*100:.1f}%), Negative: {total-pos_count}")
    return labels


def select_top_stock_features(features_df, labels, n_features=20):
    """Select most important stock features using correlation with label."""
    exclude_cols = {'code', 'trade_date', 'label_10d',
                    'market_return_20d', 'market_volatility_20d', 'market_up_ratio_20d'}
    stock_cols = [c for c in features_df.columns if c not in exclude_cols and features_df[c].dtype in ['float64', 'float32', 'int64']]

    # Quick mutual information via abs correlation
    corrs = {}
    for col in stock_cols:
        vals = features_df[col].fillna(0).values
        if np.std(vals) < 1e-10:
            continue
        try:
            ic, _ = spearmanr(vals, labels)
            if not np.isnan(ic):
                corrs[col] = abs(ic)
        except Exception:
            pass

    top_features = sorted(corrs, key=corrs.get, reverse=True)[:n_features]
    print(f"  Top {n_features} stock features: {top_features[:5]}... (IC range: {corrs.get(top_features[0],0):.4f} ~ {corrs.get(top_features[-1],0):.4f})")
    return top_features


def train_head_refiner(X_train, y_train, X_val, y_val):
    """Train LightGBM classifier for head refinement."""
    import lightgbm as lgb

    # Highly imbalanced: ~2% positive
    pos_ratio = np.sum(y_train == 1) / len(y_train)
    scale_pos = (1 - pos_ratio) / max(pos_ratio, 1e-8)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 50,
        'scale_pos_weight': min(scale_pos, 20),  # Cap to prevent instability
        'verbose': -1,
        'n_jobs': -1,
    }

    train_data = lgb.Dataset(X_train, y_train)
    val_data = lgb.Dataset(X_val, y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(50)],
    )

    return model


def evaluate_head_refiner(model, X_test, y_test, dates_test, meta_test):
    """Evaluate head refiner: daily precision/recall/win rate."""
    from sklearn.metrics import roc_auc_score, precision_recall_curve

    proba = model.predict(X_test)

    # Overall AUC
    auc = roc_auc_score(y_test, proba)
    print(f"\n  Overall AUC: {auc:.4f}")

    # Find threshold for daily ~10 stocks
    unique_dates = np.unique(dates_test)
    daily_results = []

    for date in unique_dates:
        mask = dates_test == date
        if mask.sum() < 50:
            continue
        day_proba = proba[mask]
        day_label = y_test[mask]
        day_meta = meta_test[mask] if meta_test is not None else None

        # Select top-10 by refiner probability
        top10_idx = np.argsort(day_proba)[-10:]
        top10_labels = day_label[top10_idx]

        # Precision: what fraction of refiner's top-10 are true top?
        precision = np.mean(top10_labels)

        # Also check actual forward return (from label_10d if available)
        daily_results.append({
            'date': date,
            'n_stocks': mask.sum(),
            'precision_at_10': precision,
            'avg_proba_top10': np.mean(day_proba[top10_idx]),
            'avg_proba_rest': np.mean(np.delete(day_proba, top10_idx)),
        })

    if daily_results:
        df_eval = pd.DataFrame(daily_results)
        avg_precision = df_eval['precision_at_10'].mean()
        print(f"  Daily Precision@10: {avg_precision:.1%} (refiner's top-10 中有多少是真top)")
        print(f"  vs random baseline: {np.mean(y_test)*100:.1f}%")
        print(f"  Lift: {avg_precision / max(np.mean(y_test), 1e-8):.1f}x")

    return auc, daily_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default='v4.8.7',
                        choices=list(VERSION_MAP.keys()))
    parser.add_argument('--start-date', default='2022-01-01')
    parser.add_argument('--end-date', default='2026-03-20')
    parser.add_argument('--top-pct', type=float, default=0.02, help='Top percentile for positive label')
    parser.add_argument('--n-stock-features', type=int, default=20, help='Number of stock features to include')
    parser.add_argument('--purge-days', type=int, default=15)
    args = parser.parse_args()

    vkey_short = VERSION_MAP[args.version][0]
    print("=" * 70)
    print(f"  Head Refiner 二阶段头部精筛模型训练 [{args.version}]")
    print("=" * 70)

    # Step 1: Load Stage 1 model
    print(f"\n[1/6] Loading Stage 1 scorer ({args.version})...")
    scorer, vkey, MODEL_DIR = load_scorer(args.version)

    # Step 2: Load data
    print("\n[2/6] Loading feature cache...")
    data = load_feature_cache(args.start_date, args.end_date)
    features_df = parse_features(data)
    del data  # free memory

    # Step 3: Compute labels
    print("\n[3/6] Computing labels...")
    labels = compute_labels(features_df, top_pct=args.top_pct)

    # Step 4: Compute meta-features from Stage 1
    print("\n[4/6] Computing Stage 1 meta-features...")
    meta_df = compute_stage1_meta_features(features_df, scorer)

    # Select top stock features to include
    top_stock_features = select_top_stock_features(features_df, labels, n_features=args.n_stock_features)

    # Build final feature matrix
    meta_cols = list(meta_df.columns)
    X_meta = meta_df.values
    X_stock = features_df[top_stock_features].fillna(0).values
    X_all = np.column_stack([X_meta, X_stock])
    all_feature_names = meta_cols + top_stock_features

    print(f"  Final feature matrix: {X_all.shape} ({len(meta_cols)} meta + {len(top_stock_features)} stock)")

    # Step 5: Walk-forward train/validate/test
    print("\n[5/6] Walk-forward training...")
    trade_dates = features_df['trade_date'].values
    unique_dates = sorted(np.unique(trade_dates))
    n_dates = len(unique_dates)

    # Simple 3-way split: train (first 60%), val (next 20%), test (last 20%)
    train_end_idx = int(n_dates * 0.6)
    val_end_idx = int(n_dates * 0.8)

    train_end_date = unique_dates[train_end_idx]
    # Purge gap
    purge_end_date = unique_dates[min(train_end_idx + args.purge_days, n_dates - 1)]
    val_start_date = purge_end_date
    val_end_date = unique_dates[val_end_idx]
    purge2_end_date = unique_dates[min(val_end_idx + args.purge_days, n_dates - 1)]
    test_start_date = purge2_end_date
    test_end_date = unique_dates[-1]

    print(f"  Train: <= {train_end_date} ({train_end_idx} dates)")
    print(f"  Val:   {val_start_date} ~ {val_end_date}")
    print(f"  Test:  {test_start_date} ~ {test_end_date}")

    train_mask = trade_dates <= train_end_date
    val_mask = (trade_dates >= val_start_date) & (trade_dates <= val_end_date)
    test_mask = trade_dates >= test_start_date

    X_train, y_train = X_all[train_mask], labels[train_mask]
    X_val, y_val = X_all[val_mask], labels[val_mask]
    X_test, y_test = X_all[test_mask], labels[test_mask]
    dates_test = trade_dates[test_mask]

    print(f"  Train: {len(X_train)} samples ({np.mean(y_train)*100:.1f}% positive)")
    print(f"  Val:   {len(X_val)} samples ({np.mean(y_val)*100:.1f}% positive)")
    print(f"  Test:  {len(X_test)} samples ({np.mean(y_test)*100:.1f}% positive)")

    # Train
    print("\n  Training LightGBM classifier...")
    model = train_head_refiner(X_train, y_train, X_val, y_val)

    # Evaluate
    print("\n  === Validation Set ===")
    dates_val = trade_dates[val_mask]
    val_auc, _ = evaluate_head_refiner(model, X_val, y_val, dates_val, None)

    print("\n  === Test Set ===")
    test_auc, test_results = evaluate_head_refiner(model, X_test, y_test, dates_test, None)

    # Feature importance
    importance = model.feature_importance(importance_type='gain')
    sorted_idx = np.argsort(importance)[::-1]
    print("\n  Top 15 features by gain:")
    for i in sorted_idx[:15]:
        print(f"    {all_feature_names[i]:30s}: {importance[i]:.0f}")

    # Step 6: Save model
    print("\n[6/6] Saving model...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = MODEL_DIR / f'head_refiner_{vkey}_{timestamp}.pkl'

    model_data = {
        'model': model,
        'version': args.version,
        'vkey': vkey,
        'feature_names': all_feature_names,
        'meta_feature_names': meta_cols,
        'stock_feature_names': top_stock_features,
        'top_pct': args.top_pct,
        'consensus_top_k': 20,
        'train_date_range': (args.start_date, train_end_date),
        'val_auc': val_auc,
        'test_auc': test_auc,
        'timestamp': timestamp,
    }
    joblib.dump(model_data, save_path, compress=3)
    print(f"  Saved: {save_path} ({save_path.stat().st_size / 1024:.0f} KB)")

    print("\n" + "=" * 70)
    print(f"  Training complete!")
    print(f"  Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
