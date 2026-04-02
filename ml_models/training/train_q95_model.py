#!/usr/bin/env python3
"""
Q95 极端分位数模型训练

训练一个 LightGBM quantile(alpha=0.95) 模型，专门预测收益右尾。
与现有MSE回归模型互补：MSE预测均值，Q95预测"大赢家"概率。

用法:
    python3 ml_models/training/train_q95_model.py --version v4.8.5
    python3 ml_models/training/train_q95_model.py --version v4.8.5 --alpha 0.90
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


def load_data(start_date, end_date):
    """Load feature cache with labels."""
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
            SELECT code, trade_date, features_json, label_3d, label_5d, label_10d, label_15d
            FROM v39_feature_cache
            WHERE trade_date IN ({ph})
              AND label_10d IS NOT NULL
        """, conn, params=chunk)
        if not df.empty:
            frames.append(df)
    conn.close()

    data = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(data):,} rows, {data['trade_date'].nunique()} dates ({time.time()-t0:.1f}s)")
    return data


def prepare_features(data, scorer):
    """Parse features and align with scorer's feature columns."""
    t0 = time.time()
    parsed = data['features_json'].apply(json.loads)
    features_df = pd.DataFrame(parsed.tolist())

    feature_cols = scorer.feature_cols
    if not feature_cols:
        feature_cols = [c for c in features_df.columns]

    for col in feature_cols:
        if col not in features_df.columns:
            features_df[col] = 0

    X = features_df[feature_cols].fillna(0).values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Features: {X.shape[1]} cols ({time.time()-t0:.1f}s)")
    return X, feature_cols, data['trade_date'].values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default='v4.8.5', choices=list(VERSION_MAP.keys()))
    parser.add_argument('--start-date', default='2022-01-01')
    parser.add_argument('--end-date', default='2026-03-20')
    parser.add_argument('--alpha', type=float, default=0.95, help='Quantile alpha (0.95=right tail)')
    parser.add_argument('--purge-days', type=int, default=15)
    args = parser.parse_args()

    vkey = VERSION_MAP[args.version]
    model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / vkey

    print("=" * 70)
    print(f"  Q{int(args.alpha*100)} 极端分位数模型训练 [{args.version}]")
    print("=" * 70)

    # Load scorer (for feature_cols alignment)
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
    scorer = getattr(mod, cls_name)(model_type='small_data')

    # Load data
    print(f"\n[1/4] Loading data...")
    data = load_data(args.start_date, args.end_date)

    print(f"\n[2/4] Preparing features...")
    X, feature_cols, trade_dates = prepare_features(data, scorer)

    # Labels for each horizon
    labels = {}
    for target in ['10d', '15d']:
        col = f'label_{target}'
        labels[target] = data[col].fillna(0).values

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

    print(f"  Train: {train_mask.sum():,} (<= {train_end})")
    print(f"  Val: {val_mask.sum():,} ({val_start} ~ {val_end})")
    print(f"  Test: {test_mask.sum():,} (>= {test_start})")

    # Train Q95 for each horizon
    print(f"\n[3/4] Training Q{int(args.alpha*100)} models...")
    q95_models = {}

    for target in ['10d', '15d']:
        print(f"\n  --- {target} ---")
        y = labels[target]

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_va, y_va = X[val_mask], y[val_mask]
        X_te, y_te = X[test_mask], y[test_mask]
        dates_te = trade_dates[test_mask]

        params = {
            'objective': 'quantile',
            'alpha': args.alpha,
            'metric': 'quantile',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 100,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'verbose': -1,
            'n_jobs': -1,
        }

        train_data = lgb.Dataset(X_tr, y_tr)
        val_data = lgb.Dataset(X_va, y_va, reference=train_data)

        model = lgb.train(
            params, train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(100)],
        )

        q95_models[target] = model

        # Evaluate: daily IC of Q95 predictions vs actual returns
        pred_te = model.predict(X_te)

        # Overall correlation
        ic, _ = spearmanr(pred_te, y_te)
        print(f"  Q{int(args.alpha*100)} {target} IC: {ic:.4f}")

        # Head discrimination: top-10 by Q95 vs top-10 by MSE ensemble
        # Compare actual returns of Q95's top-10 vs composite top-10
        unique_test_dates = np.unique(dates_te)
        q95_top10_rets = []
        mse_top10_rets = []

        # MSE ensemble prediction (use scorer's models)
        mse_pred = np.zeros(len(X_te))
        avail_models = scorer.models.get(target, {})
        tw = scorer.weights.get(f'label_{target}', {})
        total_w = 0
        for name, m in avail_models.items():
            try:
                if name == 'xgb':
                    import xgboost as xgb_lib
                    p = m.predict(xgb_lib.DMatrix(X_te))
                else:
                    p = m.predict(X_te)
                w = tw.get(name, 0.2)
                mse_pred += w * p
                total_w += w
            except Exception:
                pass
        if total_w > 0:
            mse_pred /= total_w

        for d in unique_test_dates:
            dmask = dates_te == d
            if dmask.sum() < 50:
                continue
            d_pred_q95 = pred_te[dmask]
            d_pred_mse = mse_pred[dmask]
            d_actual = y_te[dmask]

            q95_top10 = np.argsort(d_pred_q95)[-10:]
            mse_top10 = np.argsort(d_pred_mse)[-10:]

            q95_top10_rets.extend(d_actual[q95_top10])
            mse_top10_rets.extend(d_actual[mse_top10])

        q95_r = np.array(q95_top10_rets)
        mse_r = np.array(mse_top10_rets)
        if len(q95_r) > 0 and len(mse_r) > 0:
            q95_pf = q95_r[q95_r > 0].sum() / max(abs(q95_r[q95_r < 0].sum()), 1e-8)
            mse_pf = mse_r[mse_r > 0].sum() / max(abs(mse_r[mse_r < 0].sum()), 1e-8)
            print(f"  Top-10 比较 ({target}):")
            print(f"    MSE ensemble: avg={mse_r.mean():+.2%}, wr={((mse_r>0).mean())*100:.1f}%, PF={mse_pf:.3f}")
            print(f"    Q{int(args.alpha*100)} model:    avg={q95_r.mean():+.2%}, wr={((q95_r>0).mean())*100:.1f}%, PF={q95_pf:.3f}")

            # Combined: 50% MSE rank + 50% Q95 rank → top-10
            combined_rets = []
            for d in unique_test_dates:
                dmask = dates_te == d
                if dmask.sum() < 50:
                    continue
                d_q95 = pred_te[dmask]
                d_mse = mse_pred[dmask]
                d_actual = y_te[dmask]
                n_d = dmask.sum()
                rank_q95 = rankdata(-d_q95, method='average') / n_d
                rank_mse = rankdata(-d_mse, method='average') / n_d
                combined = 0.5 * rank_mse + 0.5 * rank_q95
                top10 = np.argsort(combined)[:10]  # lower combined rank = better
                combined_rets.extend(d_actual[top10])

            cr = np.array(combined_rets)
            if len(cr) > 0:
                cpf = cr[cr > 0].sum() / max(abs(cr[cr < 0].sum()), 1e-8)
                print(f"    Combined 50/50: avg={cr.mean():+.2%}, wr={((cr>0).mean())*100:.1f}%, PF={cpf:.3f}")

    # Save
    print(f"\n[4/4] Saving...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = model_dir / f'q95_model_{vkey}_{timestamp}.pkl'

    save_data = {
        'models': q95_models,
        'version': args.version,
        'vkey': vkey,
        'alpha': args.alpha,
        'feature_cols': feature_cols,
        'train_date_range': (args.start_date, str(train_end)),
        'timestamp': timestamp,
    }
    joblib.dump(save_data, save_path, compress=3)
    print(f"  Saved: {save_path} ({save_path.stat().st_size / 1024:.0f} KB)")
    print("=" * 70)


if __name__ == '__main__':
    main()
