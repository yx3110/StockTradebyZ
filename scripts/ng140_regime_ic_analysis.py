import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""ng1.4.0 per-regime per-algo OOS IC analysis.

Goal: compute IC per (seed, algo, horizon, regime) on 2024-2026 data.
Output: ng141_regime_weights.json with per-regime per-algo weights
(ICIR-normalized, negative algos clipped to 0).

This does NOT retrain. Reuses ng140_seed*.pkl + ng130_feature_cache +
market_amv regime labels.
"""
import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
DB = REPO / 'data_adapter' / 'stock_data.db'
PKL_DIR = REPO / 'ml_models' / 'trained_models' / 'ng'
OUT = REPO / 'ml_models' / 'ng' / 'ng141_regime_weights.json'

SEEDS = (42, 123, 456)
START = '2024-01-01'
END = '2026-04-17'
HORIZON = '10d'
ALGOS = ('lgb', 'xgb', 'cb', 'rf', 'hgb', 'lgb_rank')


def load_features_labels():
    """Load X, y_10d, dates, codes for 2024-2026."""
    import glob
    # Use ng_trainer's load_data via a shim
    from ml_models.ng.ng_trainer import NGTrainer
    t = NGTrainer(version='ng1.4.0')
    df = t.load_data(start_date=START, end_date=END)
    X, y3, y5, y10, y15, df_full = t.prepare_features(df)
    dates = df_full['trade_date'].values
    codes = df_full['code'].values
    return X, y10, dates, codes, t.feature_names


def load_regime_map():
    """Load trade_date → amv_regime (-1 bear, 1 bull)."""
    with sqlite3.connect(str(DB), timeout=30) as conn:
        rows = conn.execute(
            "SELECT trade_date, amv_regime FROM market_amv "
            "WHERE trade_date BETWEEN ? AND ?",
            (START, END)
        ).fetchall()
    return dict(rows)


def predict_per_algo(model_dict, X, horizon):
    """Return dict algo → predictions array (len N)."""
    target = model_dict['models'][horizon]
    boosters = target['models']
    preds = {}
    for algo, b in boosters.items():
        try:
            if algo == 'xgb':
                import xgboost as xgb
                dm = xgb.DMatrix(X)
                preds[algo] = b.predict(dm)
            else:
                preds[algo] = b.predict(X)
        except Exception as e:
            print(f'  [skip {algo}] {e}')
    return preds


def per_date_ic(pred, y, dates):
    """Compute spearman IC per trade_date. Returns DataFrame(date, ic, n)."""
    df = pd.DataFrame({'date': dates, 'pred': pred, 'y': y})
    rows = []
    for d, g in df.groupby('date'):
        if len(g) < 20 or g['y'].isna().all():
            continue
        ic, _ = spearmanr(g['pred'], g['y'])
        if np.isfinite(ic):
            rows.append({'date': d, 'ic': ic, 'n': len(g)})
    return pd.DataFrame(rows)


def main():
    print('Loading features + labels (2024-2026)...')
    X, y10, dates, codes, feature_names = load_features_labels()
    print(f'  {len(dates):,} rows, {len(feature_names)} features')

    print('Loading AMV regime map...')
    regime_map = load_regime_map()
    regime_arr = pd.Series(dates).map(regime_map).values
    bull_mask = regime_arr == 1
    bear_mask = regime_arr == -1
    print(f'  bull rows: {bull_mask.sum():,} / bear rows: {bear_mask.sum():,}')

    # Per-algo per-regime IC, averaged across seeds
    import glob
    results = {'bull': {}, 'bear': {}, 'all': {}}
    algo_preds = {algo: [] for algo in ALGOS}

    for seed in SEEDS:
        pkls = sorted(glob.glob(str(PKL_DIR / f'ng140_seed{seed}_multi_target_*.pkl')))
        if not pkls:
            print(f'  [skip seed{seed}] no pkl')
            continue
        print(f'Loading seed{seed}...')
        m = joblib.load(pkls[-1])
        preds_this_seed = predict_per_algo(m, X, HORIZON)
        for algo, p in preds_this_seed.items():
            algo_preds[algo].append(p)

    # Average across seeds per algo
    algo_avg = {}
    for algo, ps in algo_preds.items():
        if ps:
            algo_avg[algo] = np.mean(ps, axis=0)

    print('\nPer-regime per-algo IC analysis:')
    summary = []
    for algo, pred in algo_avg.items():
        for regime_name, mask in (('all', np.ones_like(bull_mask, bool)),
                                    ('bull', bull_mask), ('bear', bear_mask)):
            ic_df = per_date_ic(pred[mask], y10[mask], dates[mask])
            if len(ic_df) < 10:
                continue
            mean_ic = float(ic_df['ic'].mean())
            std_ic = float(ic_df['ic'].std())
            icir = mean_ic / std_ic if std_ic > 0 else 0.0
            summary.append({
                'algo': algo, 'regime': regime_name,
                'mean_ic': mean_ic, 'std_ic': std_ic, 'icir': icir,
                'n_days': len(ic_df),
            })
            print(f'  {algo:>10s} / {regime_name:>4s}: IC={mean_ic:+.4f} '
                  f'ICIR={icir:+.3f} n={len(ic_df):3d}')

    sdf = pd.DataFrame(summary)

    # Compute per-regime weights: softmax(ICIR_clipped)
    print('\nPer-regime weights (softmax of ICIR, negative clipped):')
    weights = {'bull': {}, 'bear': {}, 'all': {}}
    for regime in ('bull', 'bear', 'all'):
        sub = sdf[sdf['regime'] == regime]
        if sub.empty:
            continue
        icirs = np.array([max(x, 0) for x in sub['icir'].values])
        if icirs.sum() > 0:
            w = icirs / icirs.sum()
        else:
            w = np.ones(len(icirs)) / len(icirs)
        for algo, wi in zip(sub['algo'].values, w):
            weights[regime][algo] = float(wi)
        print(f'  {regime}: {weights[regime]}')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump({
            'summary': summary,
            'weights': weights,
            'window': f'{START} → {END}',
            'horizon': HORIZON,
            'seeds': list(SEEDS),
        }, f, indent=2, ensure_ascii=False)
    print(f'\nSaved: {OUT}')


if __name__ == '__main__':
    main()
