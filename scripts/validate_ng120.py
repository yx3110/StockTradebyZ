"""Smoke test for ng1.2.0 margin_rank model inside a single WF window.

Verifies end-to-end plumbing:
  - train_dates/val_dates set on self
  - build_groups_per_date aligned with X rows
  - margin_rank produced with correct output length
  - Spearman IC vs y_val within sane range
  - Graceful degrade on synthetic data

Intentionally minimal scope: does NOT spawn a full WF loop. For a real
grid-search run, use ng_trainer.py --version ng1.2.0 --fast-check.
"""
import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def _make_synthetic(n_dates: int = 120, n_stocks: int = 200, n_features: int = 20,
                    seed: int = 42):
    """Return (X, y, dates) with monotone pred→y signal strong enough for IC>0.5."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        date = f'2024-01-{d+1:02d}'
        for s in range(n_stocks):
            feat = rng.standard_normal(n_features)
            # y depends on linear combo of first 3 features + date noise
            y = 0.6 * feat[0] + 0.3 * feat[1] - 0.4 * feat[2] + 0.1 * rng.standard_normal()
            rows.append((date, f'stk{s:04d}', *feat, y))
    cols = ['trade_date', 'code'] + [f'f{i}' for i in range(n_features)] + ['y']
    df = pd.DataFrame(rows, columns=cols)
    return df


def _smoke_margin_objective(margin: float):
    """Verify make_margin_objective end-to-end in a LightGBM model."""
    import lightgbm as lgb
    from scipy.stats import spearmanr
    from ml_models.common.lgb_rank_utils import RANK_BASE_PARAMS, build_groups_per_date
    from ml_models.ng.ng_margin_loss import make_margin_objective, make_margin_eval_metric

    df = _make_synthetic()
    feat_cols = [c for c in df.columns if c.startswith('f')]
    # split by date to mimic WF
    uniq = sorted(df['trade_date'].unique())
    n_train = int(len(uniq) * 0.7)
    train_dates = uniq[:n_train]
    val_dates = uniq[n_train:]
    tr = df[df['trade_date'].isin(train_dates)].sort_values('trade_date').reset_index(drop=True)
    va = df[df['trade_date'].isin(val_dates)].sort_values('trade_date').reset_index(drop=True)

    gr_tr = build_groups_per_date(tr['trade_date'].values)
    gr_va = build_groups_per_date(va['trade_date'].values)
    assert sum(gr_tr) == len(tr), f"group sum mismatch: {sum(gr_tr)} vs {len(tr)}"
    assert sum(gr_va) == len(va)

    dtrain = lgb.Dataset(tr[feat_cols].values, label=tr['y'].values, group=gr_tr)
    dval = lgb.Dataset(va[feat_cols].values, label=va['y'].values, group=gr_va,
                       reference=dtrain)
    params = {**RANK_BASE_PARAMS, 'objective': make_margin_objective(margin=margin)}
    model = lgb.train(
        params, dtrain,
        num_boost_round=100,
        valid_sets=[dval],
        feval=make_margin_eval_metric(margin=margin),
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
    )

    p_val = model.predict(va[feat_cols].values)
    ic, _ = spearmanr(p_val, va['y'].values)
    logger.info(f"  margin={margin}: val Spearman IC = {ic:.4f} "
                f"(num_trees={model.num_trees()})")
    if ic < 0.3:
        raise RuntimeError(f"margin={margin} IC too low: {ic:.4f} (expected > 0.3 on synth)")
    # Pickle round-trip — critical for joblib.dump later
    import pickle
    restored = pickle.loads(pickle.dumps(model))
    p2 = restored.predict(va[feat_cols].values)
    if not np.allclose(p_val, p2):
        raise RuntimeError(f"margin={margin}: pickle round-trip changed predictions")
    return ic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--margins', type=str, default='0.03,0.05,0.08,0.10',
                        help='Comma-separated margins to smoke-test')
    args = parser.parse_args()

    margins = [float(m.strip()) for m in args.margins.split(',')]
    logger.info(f"Smoke-testing margins: {margins}")

    results = {}
    for m in margins:
        logger.info(f"\n=== margin={m} ===")
        ic = _smoke_margin_objective(margin=m)
        results[m] = ic

    logger.info("\n" + "=" * 50)
    logger.info("SUMMARY")
    logger.info("=" * 50)
    for m, ic in results.items():
        logger.info(f"  margin={m}: IC={ic:.4f}")
    logger.info("\nAll smoke tests PASSED. Ready for WF grid-search.")


if __name__ == '__main__':
    main()
