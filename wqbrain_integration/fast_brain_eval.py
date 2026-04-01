#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BRAIN 因子快速评估脚本

用单窗口 LightGBM 快速衡量 BRAIN 因子的边际贡献:
- 加载 v39_feature_cache + brain_alpha_cache
- 时序分割 train/val/test
- 训练 LightGBM (单模型, 非 ensemble)
- 输出: IC, ICIR, 特征重要度

一次评估 ~3-5 分钟 (vs 全量 WF 7+ 小时)
"""

import sys
import json
import sqlite3
import logging
import argparse
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


def load_data(db_path: str, start_date: str, end_date: str,
              brain_features: list = None) -> pd.DataFrame:
    """加载 v39 特征 + 可选 BRAIN 特征"""
    conn = sqlite3.connect(db_path)

    # v39 feature cache
    query = f"""
    SELECT v.code, v.trade_date, v.features_json,
           v.label_3d, v.label_5d, v.label_10d
    FROM v39_feature_cache v
    JOIN securities s ON v.code = s.code
    JOIN daily_quotes q ON q.security_id = s.id AND q.trade_date = v.trade_date
    WHERE v.label_3d IS NOT NULL AND v.label_5d IS NOT NULL AND v.label_10d IS NOT NULL
      AND q.volume > 0
      AND v.trade_date >= ? AND v.trade_date <= ?
    ORDER BY v.trade_date, v.code
    """
    df = pd.read_sql(query, conn, params=(start_date, end_date))
    logger.info(f"v39 缓存: {len(df):,} 条")

    # 解析 features_json
    parsed = df['features_json'].apply(_json_loads).tolist()
    df_features = pd.DataFrame(parsed)
    df_features['code'] = df['code'].values
    df_features['trade_date'] = df['trade_date'].values
    df_features['label_10d'] = df['label_10d'].values

    # 加载 daily_basic
    date_min, date_max = df_features['trade_date'].min(), df_features['trade_date'].max()
    basic = pd.read_sql("""
        SELECT s.code, db.trade_date, db.pe_ttm, db.pb, db.ps_ttm, db.turnover_rate, db.circ_mv
        FROM daily_basic db JOIN securities s ON db.security_id = s.id
        WHERE db.trade_date >= ? AND db.trade_date <= ?
    """, conn, params=[date_min, date_max])
    df_features = df_features.merge(basic, on=['code', 'trade_date'], how='left')
    df_features['log_market_cap'] = np.log1p(df_features['circ_mv'].fillna(0))
    df_features.drop(columns=['circ_mv'], inplace=True, errors='ignore')

    # 加载 BRAIN 特征
    if brain_features:
        try:
            brain_df = pd.read_sql("""
                SELECT code, trade_date, features_json
                FROM brain_alpha_cache
                WHERE trade_date >= ? AND trade_date <= ?
            """, conn, params=(start_date, end_date))

            if not brain_df.empty:
                brain_parsed = pd.json_normalize(brain_df['features_json'].apply(_json_loads))
                # 只保留请求的特征
                keep_cols = [c for c in brain_features if c in brain_parsed.columns]
                brain_parsed = brain_parsed[keep_cols]
                brain_parsed['code'] = brain_df['code'].values
                brain_parsed['trade_date'] = brain_df['trade_date'].values
                df_features = df_features.merge(brain_parsed, on=['code', 'trade_date'], how='left')
                df_features[keep_cols] = df_features[keep_cols].fillna(0.0)
                logger.info(f"BRAIN 特征: +{len(keep_cols)} 列, 覆盖 {len(brain_parsed):,} 条")
        except Exception as e:
            logger.warning(f"BRAIN 特征加载失败: {e}")

    conn.close()
    df_features = df_features.fillna(0)
    return df_features


def robust_zscore(data: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """截面 Robust Z-Score 归一化"""
    result = data.copy()
    unique_dates = np.unique(dates)
    for date in unique_dates:
        mask = dates == date
        chunk = result[mask]
        median = np.nanmedian(chunk, axis=0)
        mad = np.nanmedian(np.abs(chunk - median), axis=0) * 1.4826
        mad[mad < 1e-8] = 1e-8
        result[mask] = np.clip((chunk - median) / mad, -3, 3)
    return result


def evaluate(df: pd.DataFrame, brain_features: list = None,
             train_end: str = '2025-06-30',
             val_end: str = '2025-10-31',
             target: str = 'label_10d') -> dict:
    """
    单窗口 LightGBM 快速评估

    Returns:
        {
            'ic': float, 'icir': float, 'ic_positive_ratio': float,
            'n_features': int, 'brain_importance': dict,
            'train_samples': int, 'test_samples': int,
        }
    """
    # 分割
    exclude = ['code', 'trade_date', 'label_3d', 'label_5d', 'label_10d']
    feature_cols = [c for c in df.columns if c not in exclude]

    # 宏观特征不做截面归一化
    macro_cols = [c for c in feature_cols if c.startswith('market_')]
    stock_cols = [c for c in feature_cols if c not in macro_cols]

    # 时序分割
    train_mask = df['trade_date'] <= train_end
    val_mask = (df['trade_date'] > train_end) & (df['trade_date'] <= val_end)
    test_mask = df['trade_date'] > val_end

    X_all = df[feature_cols].values.copy()
    dates_all = df['trade_date'].values
    y_all = df[target].values

    # 截面 Z-Score (仅个股特征)
    stock_idx = [feature_cols.index(c) for c in stock_cols if c in feature_cols]
    X_all[:, stock_idx] = robust_zscore(X_all[:, stock_idx], dates_all)

    X_train = X_all[train_mask]
    y_train = y_all[train_mask]
    X_val = X_all[val_mask]
    y_val = y_all[val_mask]
    X_test = X_all[test_mask]
    y_test = y_all[test_mask]
    test_dates = dates_all[test_mask]

    logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    logger.info(f"特征数: {len(feature_cols)}")

    # 行业超额标签
    if 'sw_l1_code' in df.columns:
        for mask, y in [(train_mask, y_train), (val_mask, y_val), (test_mask, y_test)]:
            subset = df[mask]
            median = subset.groupby(['trade_date', 'sw_l1_code'])[target].transform('median')
            y[:] = df.loc[mask, target].values - median.values

    # LightGBM
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 200,
        'verbose': -1,
        'n_jobs': -1,
    }

    dtrain = lgb.Dataset(X_train, y_train)
    dval = lgb.Dataset(X_val, y_val, reference=dtrain)

    t0 = time.time()
    model = lgb.train(
        params, dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    train_time = time.time() - t0
    logger.info(f"训练耗时: {train_time:.1f}s, 最佳轮次: {model.best_iteration}")

    # 预测 test
    pred = model.predict(X_test)

    # 计算 daily IC
    unique_dates = np.unique(test_dates)
    daily_ics = []
    for date in unique_dates:
        mask = test_dates == date
        p = pred[mask]
        y = y_test[mask]
        if len(p) >= 30:
            ic, _ = spearmanr(p, y)
            if not np.isnan(ic):
                daily_ics.append(ic)

    mean_ic = np.mean(daily_ics) if daily_ics else 0
    std_ic = np.std(daily_ics) if daily_ics else 1
    icir = mean_ic / std_ic if std_ic > 1e-8 else 0
    ic_pos = np.mean([1 for ic in daily_ics if ic > 0]) if daily_ics else 0

    # 特征重要度
    importance = dict(zip(feature_cols, model.feature_importance(importance_type='gain')))

    # BRAIN 特征重要度
    brain_imp = {}
    if brain_features:
        for f in brain_features:
            if f in importance:
                brain_imp[f] = int(importance[f])

    # Top 10 特征
    top10 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

    result = {
        'ic': round(mean_ic, 6),
        'icir': round(icir, 4),
        'ic_positive_ratio': round(ic_pos, 4),
        'n_features': len(feature_cols),
        'n_test_days': len(daily_ics),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_time_s': round(train_time, 1),
        'brain_importance': brain_imp,
        'top10_features': [(name, int(imp)) for name, imp in top10],
    }

    return result


def print_result(result: dict, label: str = ''):
    """打印评估结果"""
    print(f"\n{'='*60}")
    if label:
        print(f"  {label}")
        print(f"{'='*60}")
    print(f"  IC:          {result['ic']:+.6f}")
    print(f"  ICIR:        {result['icir']:+.4f}")
    print(f"  IC>0比例:    {result['ic_positive_ratio']:.1%}")
    print(f"  特征数:      {result['n_features']}")
    print(f"  测试天数:    {result['n_test_days']}")
    print(f"  训练耗时:    {result['train_time_s']}s")

    if result.get('brain_importance'):
        print(f"\n  BRAIN 特征重要度:")
        for name, imp in sorted(result['brain_importance'].items(), key=lambda x: x[1], reverse=True):
            print(f"    {name:35s} {imp:>8,}")

    print(f"\n  Top 10 特征:")
    for name, imp in result['top10_features']:
        marker = ' ★' if name.startswith('brain_') else ''
        print(f"    {name:35s} {imp:>8,}{marker}")
    print()


# ============================================================
# CLI
# ============================================================

ALL_BRAIN = [
    # Phase 1 (9)
    'brain_intraday_intensity', 'brain_high_low_ratio', 'brain_close_to_high',
    'brain_vol_ratio', 'brain_vol_of_vol', 'brain_momentum_decay5',
    'brain_momentum_decay10', 'brain_vol_price_divergence', 'brain_turnover_momentum',
    # Phase 2 (20)
    'brain_52w_low_bounce', 'brain_ma60_reversion', 'brain_vol_asymmetry',
    'brain_roll_spread', 'brain_extreme_day_freq', 'brain_momentum_crash_hedge',
    'brain_loss_aversion', 'brain_high_resistance', 'brain_hl_spread',
    'brain_ret_autocorr', 'brain_tail_risk', 'brain_vwap_momentum',
    'brain_up_streak_ratio', 'brain_hurst_proxy', 'brain_post_limitup_ret',
    'brain_vol_price_coord', 'brain_price_jerk', 'brain_gap_strength',
    'brain_money_flow', 'brain_vol_clustering',
]


def main():
    parser = argparse.ArgumentParser(description='BRAIN 因子快速评估')
    parser.add_argument('--start-date', default='2023-01-01')
    parser.add_argument('--end-date', default='2026-03-20')
    parser.add_argument('--train-end', default='2025-06-30')
    parser.add_argument('--val-end', default='2025-10-31')
    parser.add_argument('--target', default='label_10d')
    parser.add_argument('--no-brain', action='store_true', help='不加载 BRAIN 特征 (基线)')
    parser.add_argument('--brain-subset', type=str, nargs='+', help='指定 BRAIN 特征子集')
    parser.add_argument('--compare', action='store_true', help='对比有/无 BRAIN')
    args = parser.parse_args()

    if args.compare:
        # 基线
        logger.info("=== 基线: 无 BRAIN 特征 ===")
        df_base = load_data(DB_PATH, args.start_date, args.end_date, brain_features=None)
        base_result = evaluate(df_base, brain_features=None,
                               train_end=args.train_end, val_end=args.val_end, target=args.target)
        print_result(base_result, '基线 (无 BRAIN)')

        # 全部 BRAIN
        logger.info("=== 全部 BRAIN 特征 ===")
        df_brain = load_data(DB_PATH, args.start_date, args.end_date, brain_features=ALL_BRAIN)
        brain_result = evaluate(df_brain, brain_features=ALL_BRAIN,
                                train_end=args.train_end, val_end=args.val_end, target=args.target)
        print_result(brain_result, '全部 BRAIN (+9 特征)')

        # 差异
        delta_ic = brain_result['ic'] - base_result['ic']
        delta_icir = brain_result['icir'] - base_result['icir']
        print(f"  Delta IC:   {delta_ic:+.6f}")
        print(f"  Delta ICIR: {delta_icir:+.4f}")
        print(f"  {'✅ BRAIN 有效' if delta_icir > 0 else '❌ BRAIN 无效'}")

    else:
        brain_feats = None if args.no_brain else (args.brain_subset or ALL_BRAIN)
        df = load_data(DB_PATH, args.start_date, args.end_date, brain_features=brain_feats)
        result = evaluate(df, brain_features=brain_feats,
                          train_end=args.train_end, val_end=args.val_end, target=args.target)
        label = '无 BRAIN' if args.no_brain else f'BRAIN ({len(brain_feats)} 特征)'
        print_result(result, label)


if __name__ == '__main__':
    main()
