#!/usr/bin/env python3
"""
Head Refiner V2 — 只在头部候选池训练 + 纯meta-features + LambdaRank

V1问题: stock features占87%重要性, 全市场训练学了选股而非精筛
V2改进:
  1. 只在Stage 1 Top-200候选上训练 (头部区分度)
  2. 纯meta-features, 不含stock features (避免跟Stage 1重复)
  3. LambdaRank目标 (直接优化排序, 替代binary classification)
  4. 更丰富的模型分歧特征 (两两排名差异, 头部稳定性)

用法:
    python3 ml_models/training/train_head_refiner_v2.py --version v4.8.7
    python3 ml_models/training/train_head_refiner_v2.py --version v4.8.4 --top-n 200
"""

import sys, os, json, time, argparse, sqlite3, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import rankdata, spearmanr
from itertools import combinations

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


def load_feature_cache(start_date, end_date):
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute('PRAGMA busy_timeout=30000')
    print(f"Loading feature cache {start_date} ~ {end_date}...")
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
            SELECT code, trade_date, features_json, label_10d
            FROM v39_feature_cache
            WHERE trade_date IN ({ph}) AND label_10d IS NOT NULL
        """, conn, params=chunk)
        if not df.empty:
            frames.append(df)
    conn.close()

    data = pd.concat(frames, ignore_index=True)
    print(f"  {len(data)} rows, {data['trade_date'].nunique()} dates ({time.time()-t0:.1f}s)")
    return data


def compute_meta_features_head_only(data, scorer, top_n=200):
    """Compute meta-features ONLY for top-N candidates per day.

    Returns DataFrame with columns:
      - date, code, label_10d (actual forward return)
      - Per-model predictions: pred_lgb, pred_xgb, ...
      - Per-model ranks within top-N: rank_lgb, rank_xgb, ...
      - Consensus features: consensus_count, consensus_ratio
      - Agreement features: rank_std, rank_range, pred_std, pred_cv
      - Pairwise disagreement: max_rank_diff between any two models
      - Head stability: rank_in_full (rank in full universe, normalized)
      - Gap features: gap_to_1st, gap_to_next
      - label_rank: actual rank within top-N by label_10d (for LambdaRank)
    """
    print(f"Computing meta-features for top-{top_n} candidates...")
    t0 = time.time()

    # Parse features
    parsed = data['features_json'].apply(json.loads)
    features_df = pd.DataFrame(parsed.tolist())
    features_df['code'] = data['code'].values
    features_df['trade_date'] = data['trade_date'].values
    features_df['label_10d'] = data['label_10d'].values

    # Get feature columns for scorer
    feature_cols = scorer.feature_cols or []
    for col in feature_cols:
        if col not in features_df.columns:
            features_df[col] = 0

    model_names = list(scorer.models.get('10d', {}).keys())
    print(f"  Models: {model_names}")

    dates = sorted(features_df['trade_date'].unique())
    all_rows = []

    for di, date in enumerate(dates):
        day = features_df[features_df['trade_date'] == date]
        n_full = len(day)
        if n_full < top_n:
            continue

        # Get feature matrix
        if feature_cols:
            X = day[feature_cols].fillna(0).values
        else:
            exclude = {'code', 'trade_date', 'label_10d'}
            cols = [c for c in day.columns if c not in exclude]
            X = day[cols].fillna(0).values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        codes = day['code'].values
        labels = day['label_10d'].values

        # Run all sub-models
        preds = {}
        for name, model in scorer.models['10d'].items():
            try:
                if name == 'xgb':
                    import xgboost as xgb_lib
                    preds[name] = model.predict(xgb_lib.DMatrix(X))
                else:
                    preds[name] = model.predict(X)
            except Exception:
                continue

        if not preds:
            continue

        # Rescale rank models
        regression_names = [nm for nm in preds if nm not in ('lgb_rank', 'lgb_listnet')]
        rank_model_names = [nm for nm in preds if nm in ('lgb_rank', 'lgb_listnet')]
        if regression_names and rank_model_names:
            reg_means = [np.mean(preds[nm]) for nm in regression_names]
            reg_stds = [max(np.std(preds[nm]), 1e-8) for nm in regression_names]
            t_mean, t_std = np.mean(reg_means), np.mean(reg_stds)
            for rn in rank_model_names:
                rp = preds[rn]
                rp_std = max(np.std(rp), 1e-8)
                preds[rn] = (rp - np.mean(rp)) / rp_std * t_std + t_mean

        avail = [nm for nm in model_names if nm in preds and len(preds[nm]) == n_full]
        if len(avail) < 2:
            continue

        # Weighted composite for Stage 1 ranking
        target_w = scorer.weights.get('label_10d', {})
        composite = np.zeros(n_full)
        total_w = 0
        for nm in avail:
            w = target_w.get(nm, 0.2)
            composite += w * preds[nm]
            total_w += w
        if total_w > 0:
            composite /= total_w

        # Select top-N by composite (Stage 1 filter)
        top_idx = np.argsort(composite)[-top_n:]

        # Per-model full-universe ranks (for rank_in_full)
        full_ranks = {}
        for nm in avail:
            full_ranks[nm] = rankdata(-preds[nm], method='average')

        # Now build features only for top-N
        pred_matrix_top = np.column_stack([preds[nm][top_idx] for nm in avail])
        n_models = len(avail)
        n_top = len(top_idx)

        # Per-model ranks WITHIN top-N
        rank_matrix_top = np.zeros((n_top, n_models))
        for j in range(n_models):
            rank_matrix_top[:, j] = rankdata(-pred_matrix_top[:, j], method='average')

        # Build meta-features for each top-N stock
        for ii, idx in enumerate(top_idx):
            row = {'date': date, 'code': codes[idx], 'label_10d': labels[idx]}

            # Per-model predictions
            for j, nm in enumerate(avail):
                row[f'pred_{nm}'] = float(pred_matrix_top[ii, j])
                row[f'rank_{nm}'] = float(rank_matrix_top[ii, j])
                row[f'fullrank_{nm}'] = float(full_ranks[nm][idx]) / n_full  # normalized

            # Aggregate prediction features
            preds_i = pred_matrix_top[ii]
            row['pred_mean'] = float(np.mean(preds_i))
            row['pred_std'] = float(np.std(preds_i))
            row['pred_min'] = float(np.min(preds_i))
            row['pred_max'] = float(np.max(preds_i))
            row['pred_range'] = row['pred_max'] - row['pred_min']
            row['pred_cv'] = row['pred_std'] / max(abs(row['pred_mean']), 1e-8)

            # Aggregate rank features (within top-N)
            ranks_i = rank_matrix_top[ii]
            row['rank_mean'] = float(np.mean(ranks_i))
            row['rank_std'] = float(np.std(ranks_i))
            row['rank_min'] = float(np.min(ranks_i))
            row['rank_max'] = float(np.max(ranks_i))
            row['rank_range'] = row['rank_max'] - row['rank_min']

            # Consensus: how many models put this in their Top-20 (within top-N)
            row['consensus_count'] = int(np.sum(ranks_i <= 20))
            row['consensus_ratio'] = row['consensus_count'] / n_models

            # Full-universe rank (average across models)
            fullranks_i = [full_ranks[nm][idx] / n_full for nm in avail]
            row['fullrank_mean'] = float(np.mean(fullranks_i))

            # Pairwise max disagreement
            max_diff = 0
            for a, b in combinations(range(n_models), 2):
                diff = abs(rank_matrix_top[ii, a] - rank_matrix_top[ii, b])
                max_diff = max(max_diff, diff)
            row['max_rank_diff'] = float(max_diff)

            # Gap features: composite rank within top-N
            composite_top = composite[top_idx]
            composite_rank = rankdata(-composite_top, method='ordinal')
            my_rank = composite_rank[ii]
            row['composite_rank'] = float(my_rank)

            # Gap to #1
            best_val = np.max(composite_top)
            row['gap_to_1st'] = float(best_val - composite_top[ii])

            # Label rank within top-N (for LambdaRank)
            row['label_rank'] = 0  # computed below

            all_rows.append(row)

        if (di + 1) % 100 == 0:
            print(f"    [{di+1}/{len(dates)}] {date}, {len(all_rows)} rows")

    df = pd.DataFrame(all_rows)

    # Compute label_rank per date (rank by actual label_10d within top-N, descending)
    df['label_rank'] = df.groupby('date')['label_10d'].rank(ascending=False, method='first').astype(int)

    elapsed = time.time() - t0
    n_features = len([c for c in df.columns if c not in ('date', 'code', 'label_10d', 'label_rank')])
    print(f"  Done: {len(df)} rows, {df['date'].nunique()} dates, {n_features} features ({elapsed:.1f}s)")
    return df


def train_refiner_v2(X_train, y_train, group_train, X_val, y_val, group_val):
    """Train LightGBM LambdaRank model for head refinement."""
    import lightgbm as lgb

    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'eval_at': [5, 10, 20],
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'verbose': -1,
        'n_jobs': -1,
        'lambdarank_truncation_level': 10,  # Focus on top-10 ranking
    }

    # Convert rank to relevance grades (LambdaRank needs small integer labels)
    # Rank 1-5 → grade 4, 6-10 → 3, 11-20 → 2, 21-50 → 1, 51+ → 0
    def rank_to_grade(ranks):
        grades = np.zeros_like(ranks)
        grades[ranks <= 5] = 4
        grades[(ranks > 5) & (ranks <= 10)] = 3
        grades[(ranks > 10) & (ranks <= 20)] = 2
        grades[(ranks > 20) & (ranks <= 50)] = 1
        return grades

    train_labels = rank_to_grade(y_train)
    val_labels = rank_to_grade(y_val)

    train_data = lgb.Dataset(X_train, train_labels, group=group_train)
    val_data = lgb.Dataset(X_val, val_labels, group=group_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(50)],
    )

    return model


def evaluate_v2(model, X_test, labels_10d, dates, feature_names):
    """Evaluate: refiner's top-10 actual returns vs composite top-10."""
    scores = model.predict(X_test)
    unique_dates = np.unique(dates)

    refiner_returns = []
    composite_returns = []

    for date in unique_dates:
        mask = dates == date
        if mask.sum() < 20:
            continue
        day_scores = scores[mask]
        day_labels = labels_10d[mask]

        # Refiner's top-10
        refiner_top10 = np.argsort(day_scores)[-10:]
        refiner_returns.extend(day_labels[refiner_top10])

        # Composite top-10 (already sorted by composite rank = first 10 in the group)
        # Since we selected top-N by composite, the first entries should be highest composite
        # But ordering may be jumbled. Use composite_rank feature.
        # Actually, top-10 by composite is simply the ones with lowest composite_rank
        # For simplicity, just take the first 10 (they were originally top by composite)
        composite_top10 = np.argsort(day_labels)[-10:]  # just use as reference: true top 10
        # Better: compare refiner's top-10 actual returns
        pass

    refiner_rets = np.array(refiner_returns)
    if len(refiner_rets) > 0:
        wr = (refiner_rets > 0).mean()
        pf = refiner_rets[refiner_rets > 0].sum() / max(abs(refiner_rets[refiner_rets < 0].sum()), 1e-8)
        print(f"  Refiner Top-10: avg={refiner_rets.mean():+.2%}, wr={wr:.1%}, PF={pf:.3f}, n={len(refiner_rets)}")

    # Feature importance
    importance = model.feature_importance(importance_type='gain')
    sorted_idx = np.argsort(importance)[::-1]
    print(f"\n  Top 15 features:")
    for i in sorted_idx[:15]:
        print(f"    {feature_names[i]:30s}: {importance[i]:.0f}")

    return refiner_rets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default='v4.8.7', choices=list(VERSION_MAP.keys()))
    parser.add_argument('--start-date', default='2022-01-01')
    parser.add_argument('--end-date', default='2026-03-20')
    parser.add_argument('--top-n', type=int, default=200, help='Top-N candidates for head training')
    parser.add_argument('--purge-days', type=int, default=15)
    args = parser.parse_args()

    print("=" * 70)
    print(f"  Head Refiner V2 [{args.version}] — Top-{args.top_n} + LambdaRank")
    print("=" * 70)

    # Load scorer
    scorer, vkey, model_dir = load_scorer(args.version)

    # Load data
    data = load_feature_cache(args.start_date, args.end_date)

    # Compute meta-features for head candidates only
    df = compute_meta_features_head_only(data, scorer, top_n=args.top_n)
    del data

    # Feature columns (everything except metadata)
    meta_cols = [c for c in df.columns if c not in ('date', 'code', 'label_10d', 'label_rank')]
    feature_names = meta_cols
    print(f"  Features: {len(feature_names)}")

    # Walk-forward split
    unique_dates = sorted(df['date'].unique())
    n_dates = len(unique_dates)
    train_end_idx = int(n_dates * 0.6)
    val_end_idx = int(n_dates * 0.8)

    train_end = unique_dates[train_end_idx]
    val_start = unique_dates[min(train_end_idx + args.purge_days, n_dates - 1)]
    val_end = unique_dates[val_end_idx]
    test_start = unique_dates[min(val_end_idx + args.purge_days, n_dates - 1)]

    print(f"  Train: <= {train_end}, Val: {val_start}~{val_end}, Test: >= {test_start}")

    train_df = df[df['date'] <= train_end]
    val_df = df[(df['date'] >= val_start) & (df['date'] <= val_end)]
    test_df = df[df['date'] >= test_start]

    X_train = train_df[feature_names].fillna(0).values
    X_val = val_df[feature_names].fillna(0).values
    X_test = test_df[feature_names].fillna(0).values

    y_train = train_df['label_rank'].values
    y_val = val_df['label_rank'].values

    # Group sizes for LambdaRank (number of candidates per date)
    group_train = train_df.groupby('date').size().values
    group_val = val_df.groupby('date').size().values

    print(f"  Train: {len(X_train)} ({len(group_train)} groups)")
    print(f"  Val:   {len(X_val)} ({len(group_val)} groups)")
    print(f"  Test:  {len(X_test)}")

    # Train
    print("\n  Training LambdaRank...")
    model = train_refiner_v2(X_train, y_train, group_train, X_val, y_val, group_val)

    # Evaluate on test set
    print("\n  === Test Set ===")
    test_labels = test_df['label_10d'].values
    test_dates = test_df['date'].values
    refiner_rets = evaluate_v2(model, X_test, test_labels, test_dates, feature_names)

    # Compare with composite top-10 on test dates
    print("\n  === Baseline: Composite Top-10 on test dates ===")
    composite_rets = []
    for date in sorted(test_df['date'].unique()):
        day = test_df[test_df['date'] == date]
        top10 = day.nsmallest(10, 'composite_rank')
        composite_rets.extend(top10['label_10d'].values)
    cr = np.array(composite_rets)
    if len(cr) > 0:
        pf = cr[cr > 0].sum() / max(abs(cr[cr < 0].sum()), 1e-8)
        print(f"  Composite Top-10: avg={cr.mean():+.2%}, wr={(cr>0).mean():.1%}, PF={pf:.3f}, n={len(cr)}")

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = model_dir / f'head_refiner_v2_{vkey}_{timestamp}.pkl'
    model_data = {
        'model': model,
        'version': args.version,
        'vkey': vkey,
        'feature_names': feature_names,
        'top_n': args.top_n,
        'objective': 'lambdarank',
        'truncation_level': 10,
        'train_date_range': (args.start_date, train_end),
        'timestamp': timestamp,
    }
    joblib.dump(model_data, save_path, compress=3)
    print(f"\n  Saved: {save_path} ({save_path.stat().st_size / 1024:.0f} KB)")
    print("=" * 70)


if __name__ == '__main__':
    main()
