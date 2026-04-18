"""Benchmark LightGBM LambdaRank wall time vs num_threads on Apple Silicon.

Why this exists: training launches 3 WF workers each with n_jobs=-1, risking
oversubscription on M-series chips. This measures actual per-config wall time
on a single fit so we can pick a non-pessimal num_threads for production.
"""

import json
import sqlite3
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

try:
    import orjson
    _json_loads = orjson.loads
except ImportError:
    _json_loads = json.loads

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml_models.common.lgb_rank_utils import RANK_BASE_PARAMS, build_groups_per_date

DB = PROJECT_ROOT / 'data_adapter' / 'stock_data.db'

TRAIN_END = '2023-12-29'
TRAIN_START = '2023-01-03'
VAL_START = '2024-01-15'
VAL_END = '2024-04-15'

NUM_BOOST_ROUND = 200
N_JOBS_GRID = [4, 8, 12, 18]

MARKET_COLS = [
    'market_return_5d', 'market_return_20d', 'market_volatility_20d',
    'market_breadth', 'market_new_high_ratio', 'northbound_flow_5d',
    'market_volume_ratio', 'market_drawdown', 'vix_proxy',
    'market_momentum_diff',
]


def load_window():
    print(f"[load] {TRAIN_START} -> {TRAIN_END}  val {VAL_START}->{VAL_END}")
    t0 = time.perf_counter()
    con = sqlite3.connect(str(DB))
    cols = ', '.join(['trade_date', 'features_json', 'label_10d'] + MARKET_COLS)
    q = f"""
        SELECT {cols}
        FROM ng101_feature_cache
        WHERE trade_date BETWEEN ? AND ?
          AND label_10d IS NOT NULL
        ORDER BY trade_date
    """
    df = pd.read_sql_query(q, con, params=(TRAIN_START, VAL_END))
    con.close()
    print(f"[load] rows={len(df):,} in {time.perf_counter()-t0:.1f}s")
    return df


def expand_features(df):
    t0 = time.perf_counter()
    feats = pd.DataFrame([_json_loads(s) for s in df['features_json']])
    X = pd.concat([feats.reset_index(drop=True),
                   df[MARKET_COLS].reset_index(drop=True)], axis=1)
    X = X.astype(np.float32).fillna(0.0)
    print(f"[features] shape={X.shape} in {time.perf_counter()-t0:.1f}s")
    return X


def build_relevance(df, label_col='label_10d'):
    ranks_pct = df.groupby('trade_date')[label_col].rank(pct=True)
    return np.clip((ranks_pct.values * 5).astype(np.int32), 0, 4)


def fit_one(train_set, val_set, num_threads,
            num_boost_round=NUM_BOOST_ROUND, seed=42):
    params = {
        **RANK_BASE_PARAMS,
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'eval_at': [10, 50],
        'lambdarank_truncation_level': 50,
        'num_threads': num_threads,
        'seed': seed,
        'force_col_wise': True,
    }
    # Early stopping disabled: we want fixed work to compare per-config throughput,
    # not training-quality dynamics that may plateau at iter 1 for this split.
    t0 = time.perf_counter()
    model = lgb.train(
        params, train_set, num_boost_round=num_boost_round,
        valid_sets=[train_set, val_set],
        callbacks=[lgb.log_evaluation(0)],
    )
    elapsed = time.perf_counter() - t0
    ndcg10 = model.best_score.get('valid_1', {}).get('ndcg@10')
    return elapsed, num_boost_round, ndcg10


def main():
    df = load_window()
    X = expand_features(df)
    rel = build_relevance(df)

    train_mask = df['trade_date'].between(TRAIN_START, TRAIN_END).values
    val_mask = df['trade_date'].between(VAL_START, VAL_END).values

    X_tr = X.values[train_mask]
    X_va = X.values[val_mask]
    rel_tr = rel[train_mask]
    rel_va = rel[val_mask]
    g_tr = build_groups_per_date(df.loc[train_mask, 'trade_date'].values)
    g_va = build_groups_per_date(df.loc[val_mask, 'trade_date'].values)
    n_train, n_val, n_feat = len(X_tr), len(X_va), X_tr.shape[1]
    del df, X, rel

    # Reused across all fits — bin construction happens once during warmup,
    # so the first timed run isn't penalized vs the others.
    train_set = lgb.Dataset(X_tr, label=rel_tr, group=g_tr, free_raw_data=False)
    val_set = lgb.Dataset(X_va, label=rel_va, group=g_va,
                          reference=train_set, free_raw_data=False)

    print(f"[split] train rows={n_train:,} groups={len(g_tr)}  "
          f"val rows={n_val:,} groups={len(g_va)}")
    print(f"[split] features={n_feat}")
    print()

    print("[warmup] n_jobs=8, 20 rounds (discarded)")
    fit_one(train_set, val_set, num_threads=8, num_boost_round=20)
    print()

    results = []
    for nj in N_JOBS_GRID:
        elapsed, best_iter, ndcg10 = fit_one(
            train_set, val_set, num_threads=nj
        )
        per_iter = elapsed / max(best_iter, 1) * 1000
        print(f"[bench] num_threads={nj:>2}  "
              f"wall={elapsed:6.1f}s  best_iter={best_iter:>3}  "
              f"per_iter={per_iter:6.1f}ms  ndcg@10={ndcg10:.4f}")
        results.append({
            'num_threads': nj, 'wall_s': round(elapsed, 2),
            'best_iter': best_iter, 'per_iter_ms': round(per_iter, 1),
            'ndcg10': round(ndcg10, 4) if ndcg10 else None,
        })

    print()
    print("=" * 64)
    fastest = min(results, key=lambda r: r['wall_s'])
    baseline = max(results, key=lambda r: r['num_threads'])
    speedup = baseline['wall_s'] / fastest['wall_s']
    print(f"Fastest: num_threads={fastest['num_threads']} "
          f"({fastest['wall_s']:.1f}s) -> {speedup:.2f}x vs n_jobs={baseline['num_threads']}")
    print("=" * 64)

    out = PROJECT_ROOT / 'reports' / 'benchmarks' / 'bench_lgb_n_jobs.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'window': {'train': [TRAIN_START, TRAIN_END],
                   'val': [VAL_START, VAL_END]},
        'rows': {'train': n_train, 'val': n_val, 'features': n_feat},
        'num_boost_round': NUM_BOOST_ROUND,
        'results': results,
    }, indent=2))
    print(f"[saved] {out}")


if __name__ == '__main__':
    main()
