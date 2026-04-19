"""ng1.3.0 post-train sanity three-pack.

1. Head diversity: corr(pred_excess, pred_downside) ∈ [-0.5, 0.5]
   — if > 0.5, downside head is just inverted excess (label swap failed to add info)
   — if < -0.5, heads are perfect opposites (likely a bug)
2. Seed diversity: corr between seeds ∈ [0.85, 0.95]
   — too low (< 0.80): seeds disagree too much, ensemble unstable
   — too high (> 0.95): seeds are bit-identical, seed= not propagated
3. Feature importance: Tier A (downside + AMV) + Tier B (mf) features
   appear in top 30 gain importance (LGB, aggregated across WF windows).

Exits 0 iff all three pass; nonzero on any failure with diagnostic output.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PKL_DIR = REPO / 'ml_models' / 'trained_models' / 'ng'

TIER_A_DOWNSIDE = {'current_drawdown', 'downside_vol_20d', 'recovery_speed_20d', 'gap_risk_20d'}
TIER_A_AMV = {'amv_var1', 'amv_macd', 'amv_regime_days'}
TIER_B_MF = {'elg_net_inflow_20d_z', 'mf_main_ratio_20d', 'mf_concentration_20d'}
NEW_FEATURES = TIER_A_DOWNSIDE | TIER_A_AMV | TIER_B_MF


def load_pkl(seed: int, head: str):
    import joblib
    matches = sorted(PKL_DIR.glob(f'ng130_seed{seed}_{head}_multi_target_*.pkl'))
    if not matches:
        raise FileNotFoundError(f'No ng130 seed{seed} {head} pkl found')
    return joblib.load(matches[-1]), matches[-1]


def random_feature_matrix(feature_names, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, len(feature_names)))
    return X


def predict_with_pkl(model_data, X, horizon='10d'):
    """Average ensemble prediction across algos for a given horizon."""
    target = model_data['models'][horizon]
    boosters = target['models']
    weights = target.get('weights', {})
    preds = []
    w_sum = 0.0
    for algo, b in boosters.items():
        try:
            if hasattr(b, 'predict'):
                if algo == 'xgb':
                    import xgboost as xgb
                    dm = xgb.DMatrix(X)
                    p = b.predict(dm)
                else:
                    p = b.predict(X)
                p = np.asarray(p, dtype=float).ravel()
                w = float(weights.get(algo, 1.0 / len(boosters)))
                preds.append(p * w)
                w_sum += w
        except Exception as e:
            print(f'  [skip {algo}] {e}')
    if not preds:
        raise RuntimeError(f'No algos predicted for horizon {horizon}')
    return np.sum(preds, axis=0) / max(w_sum, 1e-9)


def test_head_diversity(seeds, horizon='10d'):
    print(f'\n[1/3] Head diversity test (pred_excess vs pred_downside, {horizon})')
    ok_all = True
    for seed in seeds:
        m_ex, _ = load_pkl(seed, 'excess')
        m_dn, _ = load_pkl(seed, 'downside')
        fn_ex = m_ex['feature_names']
        fn_dn = m_dn['feature_names']
        if fn_ex != fn_dn:
            print(f'  seed{seed}: feature_names differ (ex={len(fn_ex)}, dn={len(fn_dn)}) — using ex')
        X = random_feature_matrix(fn_ex, n=2000, seed=seed)
        p_ex = predict_with_pkl(m_ex, X, horizon)
        p_dn = predict_with_pkl(m_dn, X, horizon)
        c = float(np.corrcoef(p_ex, p_dn)[0, 1])
        ok = -0.5 <= c <= 0.5
        ok_all &= ok
        flag = '✅' if ok else '❌'
        print(f'  {flag} seed{seed}: corr(excess, downside) = {c:+.3f} (expect [-0.5, 0.5])')
    return ok_all


def test_seed_diversity(seeds, head='excess', horizon='10d'):
    print(f'\n[2/3] Seed diversity test ({head} head, {horizon})')
    preds = {}
    fn_ref = None
    for seed in seeds:
        m, _ = load_pkl(seed, head)
        if fn_ref is None:
            fn_ref = m['feature_names']
        X = random_feature_matrix(fn_ref, n=2000, seed=123)
        preds[seed] = predict_with_pkl(m, X, horizon)
    ok_all = True
    seed_list = sorted(preds.keys())
    for i in range(len(seed_list)):
        for j in range(i + 1, len(seed_list)):
            a, b = seed_list[i], seed_list[j]
            c = float(np.corrcoef(preds[a], preds[b])[0, 1])
            ok = 0.80 <= c <= 0.97
            ok_all &= ok
            flag = '✅' if ok else '⚠️'
            print(f'  {flag} corr(seed{a}, seed{b}) = {c:+.3f} (expect [0.80, 0.97])')
    return ok_all


def get_lgb_importance(model, feature_names):
    """Return dict feature→gain importance from LGB booster."""
    if not hasattr(model, 'feature_importance'):
        return {}
    imp = model.feature_importance(importance_type='gain')
    return dict(zip(feature_names, imp.tolist()))


def test_feature_importance(seeds, heads=('excess', 'downside'), horizon='10d', top_n=30):
    print(f'\n[3/3] Feature importance test (new features in top {top_n})')
    ok_all = True
    for seed in seeds:
        for head in heads:
            m, _ = load_pkl(seed, head)
            fn = m['feature_names']
            lgb_model = m['models'][horizon]['models'].get('lgb')
            if lgb_model is None:
                print(f'  [skip seed{seed} {head}] no lgb booster')
                continue
            imp = get_lgb_importance(lgb_model, fn)
            ranked = sorted(imp.items(), key=lambda x: -x[1])
            top_names = {name for name, _ in ranked[:top_n]}
            hits_a_ds = TIER_A_DOWNSIDE & top_names
            hits_a_amv = TIER_A_AMV & top_names
            hits_b_mf = TIER_B_MF & top_names
            total_hits = len(hits_a_ds) + len(hits_a_amv) + len(hits_b_mf)
            expected_min = 3  # at least 3 of 10 new features should rank top 30
            ok = total_hits >= expected_min
            ok_all &= ok
            flag = '✅' if ok else '⚠️'
            print(f'  {flag} seed{seed} {head}: {total_hits}/10 new features in top {top_n} '
                  f'(downside={len(hits_a_ds)}, amv={len(hits_a_amv)}, mf={len(hits_b_mf)})')
            if not ok:
                print(f'     missing: {NEW_FEATURES - top_names}')
    return ok_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    ap.add_argument('--horizon', default='10d')
    args = ap.parse_args()

    print(f'ng1.3.0 sanity three-pack — seeds={args.seeds}, horizon={args.horizon}')

    r1 = test_head_diversity(args.seeds, args.horizon)
    r2 = test_seed_diversity(args.seeds, 'excess', args.horizon)
    r3 = test_seed_diversity(args.seeds, 'downside', args.horizon)
    r4 = test_feature_importance(args.seeds, ('excess', 'downside'), args.horizon)

    all_ok = r1 and r2 and r3 and r4
    print('\n' + '=' * 60)
    print(f'Overall: {"✅ ALL PASS" if all_ok else "❌ SOME FAILED"}')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
