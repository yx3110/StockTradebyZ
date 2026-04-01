#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top-K 因子筛选器

解决 "IC陷阱": 全局IC提升但Top-10收益下降的问题。

核心思路:
  - 不用全局 IC/ICIR 选因子
  - 用 "Top-K Sharpe" 选因子: 只看因子对头部10只股票的区分度
  - 逐个加因子, 只保留能提升 Top-K 收益的因子

评估指标:
  1. Top-K Mean Pred: Top-10 的平均预测值 (越高越好)
  2. Head Sharpness: Top-10 pred vs P90 pred 的差距 (越大越好)
  3. Top-K Hit Rate: Top-10预测股在实际Top-50中的命中率
  4. Top-K Sharpe: Top-10组合的模拟Sharpe

用法:
    python3 wqbrain_integration/topk_feature_selector.py
"""

import sys
import json
import sqlite3
import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from core.config import get_db_path
    DB_PATH = str(get_db_path())
except ImportError:
    DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')

try:
    import orjson
    _json_loads = orjson.loads
except ImportError:
    _json_loads = json.loads

import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


ALL_BRAIN_FEATURES = [
    'brain_intraday_intensity', 'brain_high_low_ratio', 'brain_close_to_high',
    'brain_vol_ratio', 'brain_vol_of_vol', 'brain_momentum_decay5',
    'brain_momentum_decay10', 'brain_vol_price_divergence', 'brain_turnover_momentum',
    'brain_52w_low_bounce', 'brain_ma60_reversion', 'brain_vol_asymmetry',
    'brain_roll_spread', 'brain_extreme_day_freq', 'brain_momentum_crash_hedge',
    'brain_loss_aversion', 'brain_high_resistance', 'brain_hl_spread',
    'brain_ret_autocorr', 'brain_tail_risk', 'brain_vwap_momentum',
    'brain_up_streak_ratio', 'brain_hurst_proxy', 'brain_post_limitup_ret',
    'brain_vol_price_coord', 'brain_price_jerk', 'brain_gap_strength',
    'brain_money_flow', 'brain_vol_clustering',
]


def load_data(db_path, start_date, end_date, brain_features=None):
    """加载 v39 特征 + BRAIN 特征"""
    conn = sqlite3.connect(db_path)
    query = """
    SELECT v.code, v.trade_date, v.features_json, v.label_10d
    FROM v39_feature_cache v
    JOIN securities s ON v.code = s.code
    JOIN daily_quotes q ON q.security_id = s.id AND q.trade_date = v.trade_date
    WHERE v.label_10d IS NOT NULL AND q.volume > 0
      AND v.trade_date >= ? AND v.trade_date <= ?
    ORDER BY v.trade_date, v.code
    """
    df = pd.read_sql(query, conn, params=(start_date, end_date))

    parsed = df['features_json'].apply(_json_loads).tolist()
    df_features = pd.DataFrame(parsed)
    df_features['code'] = df['code'].values
    df_features['trade_date'] = df['trade_date'].values
    df_features['label_10d'] = df['label_10d'].values

    # daily_basic
    date_min, date_max = df_features['trade_date'].min(), df_features['trade_date'].max()
    basic = pd.read_sql("""
        SELECT s.code, db.trade_date, db.pe_ttm, db.pb, db.ps_ttm, db.turnover_rate, db.circ_mv
        FROM daily_basic db JOIN securities s ON db.security_id = s.id
        WHERE db.trade_date >= ? AND db.trade_date <= ?
    """, conn, params=[date_min, date_max])
    df_features = df_features.merge(basic, on=['code', 'trade_date'], how='left')
    df_features['log_market_cap'] = np.log1p(df_features['circ_mv'].fillna(0))
    df_features.drop(columns=['circ_mv'], inplace=True, errors='ignore')

    # BRAIN 特征
    if brain_features:
        brain_df = pd.read_sql("""
            SELECT code, trade_date, features_json FROM brain_alpha_cache
            WHERE trade_date >= ? AND trade_date <= ?
        """, conn, params=(start_date, end_date))
        if not brain_df.empty:
            bp = pd.json_normalize(brain_df['features_json'].apply(_json_loads))
            keep = [c for c in brain_features if c in bp.columns]
            bp = bp[keep]
            bp['code'] = brain_df['code'].values
            bp['trade_date'] = brain_df['trade_date'].values
            df_features = df_features.merge(bp, on=['code', 'trade_date'], how='left')
            df_features[keep] = df_features[keep].fillna(0.0)

    conn.close()
    return df_features.fillna(0)


def robust_zscore(data, dates):
    """截面 Robust Z-Score"""
    result = data.copy()
    for date in np.unique(dates):
        mask = dates == date
        chunk = result[mask]
        median = np.nanmedian(chunk, axis=0)
        mad = np.nanmedian(np.abs(chunk - median), axis=0) * 1.4826
        mad[mad < 1e-8] = 1e-8
        result[mask] = np.clip((chunk - median) / mad, -3, 3)
    return result


def evaluate_topk(df, brain_features=None, train_end='2025-06-30',
                  val_end='2025-10-31', K=10):
    """
    Top-K 评估: 不只看全局IC, 更看 Top-K 的选股质量

    Returns:
        {
            'global_ic': float,
            'global_icir': float,
            'topk_mean_pred': float,       # Top-K 平均预测值
            'topk_head_sharpness': float,  # Top-K vs P90 的跳跃
            'topk_hit_rate': float,        # Top-K预测在实际Top-50的命中率
            'topk_return': float,          # Top-K 平均实际收益
            'topk_sharpe_proxy': float,    # Top-K 模拟Sharpe
        }
    """
    exclude = ['code', 'trade_date', 'label_3d', 'label_5d', 'label_10d']
    feature_cols = [c for c in df.columns if c not in exclude]
    macro_cols = [c for c in feature_cols if c.startswith('market_')]
    stock_cols = [c for c in feature_cols if c not in macro_cols]

    train_mask = df['trade_date'] <= train_end
    val_mask = (df['trade_date'] > train_end) & (df['trade_date'] <= val_end)
    test_mask = df['trade_date'] > val_end

    X_all = df[feature_cols].values.copy()
    dates_all = df['trade_date'].values
    y_all = df['label_10d'].values

    stock_idx = [feature_cols.index(c) for c in stock_cols if c in feature_cols]
    X_all[:, stock_idx] = robust_zscore(X_all[:, stock_idx], dates_all)

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]
    test_dates = dates_all[test_mask]
    test_codes = df.loc[test_mask, 'code'].values

    # 训练 LightGBM
    params = {
        'objective': 'regression', 'metric': 'rmse',
        'num_leaves': 31, 'learning_rate': 0.05,
        'feature_fraction': 0.8, 'bagging_fraction': 0.8,
        'bagging_freq': 5, 'min_data_in_leaf': 200,
        'verbose': -1, 'n_jobs': -1,
    }
    dtrain = lgb.Dataset(X_train, y_train)
    dval = lgb.Dataset(X_val, y_val, reference=dtrain)
    model = lgb.train(params, dtrain, num_boost_round=500,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

    pred = model.predict(X_test)

    # === 全局 IC/ICIR ===
    unique_dates = np.unique(test_dates)
    daily_ics = []
    daily_topk_returns = []
    daily_topk_preds = []
    daily_head_sharpness = []
    daily_hit_rates = []

    for date in unique_dates:
        mask = test_dates == date
        p = pred[mask]
        y = y_test[mask]

        if len(p) < 100:
            continue

        ic, _ = spearmanr(p, y)
        if not np.isnan(ic):
            daily_ics.append(ic)

        # Top-K 分析
        sorted_idx = np.argsort(-p)  # 从高到低
        topk_idx = sorted_idx[:K]
        topk_preds_val = p[topk_idx]
        topk_actual = y[topk_idx]
        p90 = np.percentile(p, 90)

        daily_topk_preds.append(np.mean(topk_preds_val))
        daily_topk_returns.append(np.mean(topk_actual))
        daily_head_sharpness.append(np.mean(topk_preds_val) - p90)

        # Hit rate: Top-K 预测 中有多少在实际 Top-50 里
        actual_top50_idx = set(np.argsort(-y)[:50])
        hits = sum(1 for i in topk_idx if i in actual_top50_idx)
        daily_hit_rates.append(hits / K)

    mean_ic = np.mean(daily_ics) if daily_ics else 0
    std_ic = np.std(daily_ics) if daily_ics else 1
    icir = mean_ic / std_ic if std_ic > 1e-8 else 0

    topk_returns = np.array(daily_topk_returns) if daily_topk_returns else np.array([0])
    topk_sharpe = np.mean(topk_returns) / (np.std(topk_returns) + 1e-8) * np.sqrt(252 / 10)

    # 特征重要度
    importance = dict(zip(feature_cols, model.feature_importance(importance_type='gain')))
    brain_imp = {f: int(importance.get(f, 0)) for f in (brain_features or []) if f in importance}

    return {
        'global_ic': round(mean_ic, 6),
        'global_icir': round(icir, 4),
        'topk_mean_pred': round(np.mean(daily_topk_preds), 6) if daily_topk_preds else 0,
        'topk_head_sharpness': round(np.mean(daily_head_sharpness), 6) if daily_head_sharpness else 0,
        'topk_hit_rate': round(np.mean(daily_hit_rates), 4) if daily_hit_rates else 0,
        'topk_mean_return': round(np.mean(daily_topk_returns), 6) if daily_topk_returns else 0,
        'topk_sharpe_proxy': round(topk_sharpe, 4),
        'n_features': len(feature_cols),
        'brain_importance': brain_imp,
    }


def greedy_topk_selection(db_path=DB_PATH, start_date='2023-01-01', end_date='2026-03-20',
                          train_end='2025-06-30', val_end='2025-10-31'):
    """
    贪心 Top-K 因子筛选:
    1. 从基线开始 (无 BRAIN)
    2. 逐个加入 BRAIN 因子
    3. 只保留能提升 topk_sharpe_proxy 的因子
    """
    logger.info("=== Top-K 贪心因子筛选 ===")

    # 基线
    logger.info("加载基线数据...")
    df_base = load_data(db_path, start_date, end_date, brain_features=None)
    base_result = evaluate_topk(df_base, brain_features=None,
                                train_end=train_end, val_end=val_end)
    logger.info(f"基线: ICIR={base_result['global_icir']:.4f}, "
                f"TopK_Sharpe={base_result['topk_sharpe_proxy']:.4f}, "
                f"TopK_Return={base_result['topk_mean_return']:.6f}, "
                f"HitRate={base_result['topk_hit_rate']:.1%}")

    best_sharpe = base_result['topk_sharpe_proxy']
    best_return = base_result['topk_mean_return']
    selected = []

    # 逐个测试每个 BRAIN 因子
    logger.info(f"\n逐个测试 {len(ALL_BRAIN_FEATURES)} 个 BRAIN 因子...")

    single_results = []
    for factor in ALL_BRAIN_FEATURES:
        df = load_data(db_path, start_date, end_date, brain_features=[factor])
        result = evaluate_topk(df, brain_features=[factor],
                               train_end=train_end, val_end=val_end)
        delta_sharpe = result['topk_sharpe_proxy'] - base_result['topk_sharpe_proxy']
        delta_return = result['topk_mean_return'] - base_result['topk_mean_return']
        single_results.append((factor, result, delta_sharpe, delta_return))
        marker = '✅' if delta_sharpe > 0 and delta_return > 0 else '❌'
        logger.info(f"  {factor:35s} TopK_Sharpe={result['topk_sharpe_proxy']:+.4f} "
                    f"(Δ{delta_sharpe:+.4f}) Return={result['topk_mean_return']:+.6f} "
                    f"(Δ{delta_return:+.6f}) {marker}")

    # 按 delta_return 排序 (最终关心的是收益)
    single_results.sort(key=lambda x: x[3], reverse=True)

    print(f"\n{'='*70}")
    print(f"单因子 Top-K 贡献排名 (按 Top-10 收益增量)")
    print(f"{'='*70}")
    print(f"{'因子':35s} {'TopK_Sharpe':>12s} {'Δ_Sharpe':>10s} {'TopK_Return':>12s} {'Δ_Return':>10s}")
    for factor, result, ds, dr in single_results:
        marker = ' ★' if ds > 0 and dr > 0 else ''
        print(f"{factor:35s} {result['topk_sharpe_proxy']:>12.4f} {ds:>+10.4f} "
              f"{result['topk_mean_return']:>12.6f} {dr:>+10.6f}{marker}")

    # 贪心组合
    print(f"\n{'='*70}")
    print(f"贪心组合 (逐个加入, 只保留提升 Top-K 收益的)")
    print(f"{'='*70}")

    selected = []
    current_best_return = base_result['topk_mean_return']
    current_best_sharpe = base_result['topk_sharpe_proxy']

    # 按单因子贡献排序, 从最好的开始加
    candidates = [(f, dr) for f, r, ds, dr in single_results if dr > 0]

    for factor, _ in candidates:
        test_set = selected + [factor]
        df = load_data(db_path, start_date, end_date, brain_features=test_set)
        result = evaluate_topk(df, brain_features=test_set,
                               train_end=train_end, val_end=val_end)

        if result['topk_mean_return'] > current_best_return:
            selected.append(factor)
            current_best_return = result['topk_mean_return']
            current_best_sharpe = result['topk_sharpe_proxy']
            print(f"  ✅ +{factor:30s} → TopK_Return={current_best_return:+.6f}, "
                  f"TopK_Sharpe={current_best_sharpe:.4f} ({len(selected)}个)")
        else:
            print(f"  ❌ +{factor:30s} → 无提升, 跳过")

    print(f"\n{'='*70}")
    print(f"最终选择: {len(selected)} 个 BRAIN 因子")
    print(f"{'='*70}")
    for f in selected:
        print(f"  {f}")
    print(f"\n基线 TopK_Return: {base_result['topk_mean_return']:+.6f}")
    print(f"最优 TopK_Return: {current_best_return:+.6f}")
    print(f"提升: {current_best_return - base_result['topk_mean_return']:+.6f}")

    return selected


if __name__ == '__main__':
    selected = greedy_topk_selection()
