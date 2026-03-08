#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局分位数校准脚本 — 为已有模型计算 global_quantiles

用法:
    # 校准 V4.4 模型 (推荐)
    python3 ml_models/calibrate_global_quantiles.py --version v4.4

    # 校准 V4.3 模型
    python3 ml_models/calibrate_global_quantiles.py --version v4.3

    # 校准 V3.96 模型
    python3 ml_models/calibrate_global_quantiles.py --version v3.96

    # 校准所有活跃版本
    python3 ml_models/calibrate_global_quantiles.py --all

原理:
    1. 加载已训练模型
    2. 从 v39_feature_cache 加载所有历史特征 (1000+天)
    3. 对每天的所有股票运行模型预测, 得到 combined_pred
    4. 收集所有 combined_pred, 计算 1001 个分位点
    5. 保存 global_quantiles.npy 到模型目录

校准后, scorer 加载时自动检测 global_quantiles.npy,
评分从 "每日截面百分位 [30,90]" 切换为 "全局百分位 [0,100]"
"""

import sys
import json
import sqlite3
import argparse
import numpy as np
import pandas as pd
import joblib
import pickle
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / 'data_adapter' / 'stock_data.db'


def load_model_and_config(version: str):
    """加载模型文件, 返回 (models, weights, target_weights, feature_names, scaler, model_dir, extra_config)"""
    version_map = {
        'v3.96': ('v396', ['v396_*.pkl', 'v395_multi_target_*.pkl']),
        'v4.3': ('v43', ['v43_*.pkl']),
        'v4.4': ('v44', ['v44_*.pkl']),
        'v4.6': ('v46', ['v46_*.pkl']),
        'v4.7.3': ('v473', ['v473_*.pkl']),
        'v4.7.5': ('v475', ['v475_*.pkl']),
        'v4.8': ('v48', ['v48_*.pkl']),
    }
    if version not in version_map:
        raise ValueError(f"Unknown version: {version}. Supported: {list(version_map.keys())}")

    dir_name, patterns = version_map[version]
    model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / dir_name
    model_files = []
    for pat in patterns:
        model_files.extend(model_dir.glob(pat))

    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir}")

    latest = max(model_files, key=lambda f: f.stat().st_mtime)
    print(f"加载模型: {latest.name} ({latest.stat().st_size / 1024 / 1024:.1f} MB)")

    try:
        model_data = joblib.load(latest)
    except Exception:
        with open(latest, 'rb') as f:
            model_data = pickle.load(f)

    # 解析模型结构
    raw_models = model_data.get('models', {})
    models = {}
    weights = model_data.get('ensemble_weights', {})
    for target, target_data in raw_models.items():
        if isinstance(target_data, dict) and 'models' in target_data:
            models[target] = target_data['models']
            if f'label_{target}' not in weights:
                weights[f'label_{target}'] = target_data.get('weights', {})
        else:
            models[target] = target_data

    feature_names = model_data.get('feature_names', model_data.get('feature_cols', []))
    scaler = model_data.get('scaler')

    if version == 'v3.96':
        target_weights = model_data.get('dynamic_weights',
                                         model_data.get('target_weights',
                                                        {'label_3d': 0.4, 'label_5d': 0.35, 'label_10d': 0.25}))
    elif version in ('v4.7.5', 'v4.6', 'v4.7.3'):
        # 10d+15d optimal weights (ablation confirmed across all models)
        target_weights = {'label_3d': 0.0, 'label_5d': 0.0, 'label_10d': 0.6, 'label_15d': 0.4}
    else:
        target_weights = model_data.get('target_weights', {
            'label_3d': 0.25, 'label_5d': 0.30, 'label_10d': 0.25, 'label_15d': 0.20
        })

    extra_config = {
        'robust_zscore': model_data.get('robust_zscore', False),
        'stock_feature_cols': model_data.get('stock_feature_cols', None),
        'extra_features_from_daily_basic': model_data.get('extra_features_from_daily_basic', None),
        'extra_features_from_tech_indicators': model_data.get('extra_features_from_tech_indicators', None),
        'winsorize_bounds': model_data.get('winsorize_bounds', None),
        'market_feature_cols': model_data.get('market_features', model_data.get('market_feature_cols', [])),
    }

    print(f"  模型目标: {list(models.keys())}")
    print(f"  特征数: {len(feature_names)}")
    print(f"  目标权重: {target_weights}")

    return models, weights, target_weights, feature_names, scaler, model_dir, extra_config


def load_all_feature_cache_dates():
    """从 v39_feature_cache 加载所有可用日期"""
    conn = sqlite3.connect(DB_PATH)
    dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT trade_date FROM v39_feature_cache ORDER BY trade_date"
    ).fetchall()]
    conn.close()
    print(f"v39_feature_cache: {len(dates)} 个交易日 ({dates[0]} ~ {dates[-1]})")
    return dates


def load_features_for_date(date: str, feature_names: list, market_feature_cols: list,
                            extra_config: dict) -> pd.DataFrame:
    """加载单日特征数据"""
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT code, features_json, " +
            ", ".join([f"market_{c}" if not c.startswith('market_') else c
                       for c in market_feature_cols]) +
            " FROM v39_feature_cache WHERE trade_date = ?",
            (date,)
        ).fetchall()

        if not rows:
            return pd.DataFrame()

        records = []
        for row in rows:
            code = row[0]
            features_json = row[1]
            market_vals = row[2:]

            try:
                features = json.loads(features_json)
            except (json.JSONDecodeError, TypeError):
                continue

            record = {'code': code}
            record.update(features)

            # 添加市场特征
            for i, col in enumerate(market_feature_cols):
                mkey = f"market_{col}" if not col.startswith('market_') else col
                record[mkey] = market_vals[i] if i < len(market_vals) else 0

            records.append(record)

        df = pd.DataFrame(records)
        return df

    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


def robust_zscore_normalize(features_df: pd.DataFrame, stock_cols: list) -> pd.DataFrame:
    """截面 Robust Z-Score 归一化"""
    if not stock_cols:
        return features_df
    rank_cols = [c for c in stock_cols if c in features_df.columns]
    for col in rank_cols:
        vals = features_df[col].dropna()
        if len(vals) < 10:
            continue
        median = vals.median()
        mad = np.median(np.abs(vals - median))
        if mad < 1e-8:
            features_df[col] = 0
        else:
            features_df[col] = np.clip((features_df[col] - median) / (mad * 1.4826), -3, 3)
    return features_df


def apply_winsorization(X: np.ndarray, feature_cols: list, winsorize_bounds: dict) -> np.ndarray:
    """Winsorization"""
    if not winsorize_bounds:
        return X
    for col_idx, col in enumerate(feature_cols):
        if col in winsorize_bounds:
            bounds = winsorize_bounds[col]
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                lo, hi = bounds
                X[:, col_idx] = np.clip(X[:, col_idx], lo, hi)
    return X


def compute_predictions_for_date(features_df: pd.DataFrame, models: dict, weights: dict,
                                   target_weights: dict, feature_names: list,
                                   extra_config: dict) -> tuple:
    """对单日数据计算 per-target predictions 和 combined_pred

    Returns:
        (combined_pred, per_target_preds): combined_pred 是加权融合, per_target_preds 是 {target_key: np.ndarray}
        如果数据无效返回 (np.array([]), {})
    """
    if features_df.empty or len(features_df) < 5:
        return np.array([]), {}

    # Robust Z-Score
    if extra_config.get('robust_zscore'):
        features_df = robust_zscore_normalize(features_df.copy(), extra_config.get('stock_feature_cols'))

    # 准备特征矩阵
    available_cols = [c for c in feature_names if c in features_df.columns]
    if len(available_cols) < len(feature_names) * 0.5:
        return np.array([]), {}

    X = features_df[available_cols].fillna(0).values

    # Winsorization
    winsorize_bounds = extra_config.get('winsorize_bounds')
    if winsorize_bounds:
        if isinstance(winsorize_bounds, dict):
            X = apply_winsorization(X, available_cols, winsorize_bounds)

    # 如果特征数不够, 补0
    if len(available_cols) < len(feature_names):
        full_X = np.zeros((X.shape[0], len(feature_names)))
        col_indices = [feature_names.index(c) for c in available_cols if c in feature_names]
        for i, idx in enumerate(col_indices):
            full_X[:, idx] = X[:, i]
        X = full_X

    # 集成预测
    predictions = {}
    for target_key, target_models in models.items():
        # 先收集所有预测
        preds = {}
        for name, model in target_models.items():
            try:
                if name == 'xgb':
                    import xgboost as xgb
                    preds[name] = model.predict(xgb.DMatrix(X))
                else:
                    preds[name] = model.predict(X)
            except Exception:
                continue

        # Rescale rank模型到回归模型尺度
        regression_names = [n for n in preds if n not in ('lgb_rank', 'lgb_listnet')]
        rank_names = [n for n in preds if n in ('lgb_rank', 'lgb_listnet')]
        if regression_names and rank_names:
            reg_means = [np.mean(preds[n]) for n in regression_names]
            reg_stds = [max(np.std(preds[n]), 1e-8) for n in regression_names]
            t_mean = np.mean(reg_means)
            t_std = np.mean(reg_stds)
            for rn in rank_names:
                rp = preds[rn]
                rp_std = max(np.std(rp), 1e-8)
                preds[rn] = (rp - np.mean(rp)) / rp_std * t_std + t_mean

        target_pred = np.zeros(X.shape[0])
        total_weight = 0
        target_w = weights.get(f'label_{target_key}', {})
        for name, pred in preds.items():
            w = target_w.get(name, 0.2)
            target_pred += w * pred
            total_weight += w

        if total_weight > 0:
            target_pred /= total_weight
        predictions[target_key] = target_pred

    # 加权融合
    combined = np.zeros(X.shape[0])
    for target_key, pred in predictions.items():
        w = target_weights.get(f'label_{target_key}', 0)
        combined += w * pred

    return combined, predictions


def compute_combined_pred_for_date(features_df, models, weights, target_weights, feature_names, extra_config):
    """兼容旧接口"""
    combined, _ = compute_predictions_for_date(features_df, models, weights, target_weights, feature_names, extra_config)
    return combined


def calibrate_version(version: str, n_quantiles: int = 1001, with_recommendation: bool = False):
    """对指定版本模型进行全局分位数校准 + 可选composite推荐阈值"""
    print(f"\n{'=' * 60}")
    print(f"校准 {version} 模型的全局分位数" + (" + 推荐阈值" if with_recommendation else ""))
    print(f"{'=' * 60}")

    models, weights, target_weights, feature_names, scaler, model_dir, extra_config = \
        load_model_and_config(version)

    dates = load_all_feature_cache_dates()
    market_feature_cols = extra_config.get('market_feature_cols', [])

    all_combined_preds = []
    all_composite_scores = [] if with_recommendation else None
    n_stocks_total = 0
    n_dates_success = 0

    # Composite 权重
    if version in ('v4.7.5', 'v4.6', 'v4.7.3'):
        composite_weights = {'3d': 0.0, '5d': 0.0, '10d': 0.6, '15d': 0.4}
    else:
        composite_weights = {'3d': 0.1, '5d': 0.2, '10d': 0.4, '15d': 0.3}

    for date in tqdm(dates, desc=f"校准 {version}"):
        features_df = load_features_for_date(date, feature_names, market_feature_cols, extra_config)
        if features_df.empty:
            continue

        combined, per_target = compute_predictions_for_date(
            features_df, models, weights, target_weights, feature_names, extra_config)

        if len(combined) > 0:
            all_combined_preds.append(combined)
            n_stocks_total += len(combined)
            n_dates_success += 1

            if with_recommendation and per_target:
                composite = np.zeros(len(combined))
                for target_key, w in composite_weights.items():
                    if target_key in per_target:
                        composite += w * per_target[target_key]
                all_composite_scores.append(composite)

    if not all_combined_preds:
        print(f"  ❌ 无有效预测数据, 跳过")
        return

    all_preds = np.concatenate(all_combined_preds)
    print(f"\n校准统计:")
    print(f"  有效交易日: {n_dates_success}/{len(dates)}")
    print(f"  总股票-日数: {n_stocks_total:,}")
    print(f"  combined_pred 分布:")
    print(f"    min={all_preds.min():.6f}, max={all_preds.max():.6f}")
    print(f"    P1={np.percentile(all_preds, 1):.6f}, P25={np.percentile(all_preds, 25):.6f}")
    print(f"    P50={np.percentile(all_preds, 50):.6f}, P75={np.percentile(all_preds, 75):.6f}")
    print(f"    P99={np.percentile(all_preds, 99):.6f}")

    # 计算分位数
    quantile_points = np.linspace(0, 100, n_quantiles)
    global_quantiles = np.percentile(all_preds, quantile_points)

    # 保存
    output_path = model_dir / 'global_quantiles.npy'
    np.save(output_path, global_quantiles)
    print(f"\n✅ 全局分位数已保存: {output_path}")
    print(f"   分位点数: {n_quantiles}")
    print(f"   文件大小: {output_path.stat().st_size / 1024:.1f} KB")

    # 演示评分映射
    print(f"\n评分映射示例:")
    test_percentiles = [1, 10, 25, 50, 75, 90, 99]
    for p in test_percentiles:
        threshold = global_quantiles[int(p / 100 * (n_quantiles - 1))]
        print(f"  全局 P{p:2d} (combined_pred={threshold:+.6f}) → 评分 {p}")

    # 计算推荐阈值 (基于composite score百分位)
    if with_recommendation and all_composite_scores:
        all_composites = np.concatenate(all_composite_scores)
        rec_thresholds = {
            'strong_buy': float(np.percentile(all_composites, 95)),   # Top 5%
            'buy': float(np.percentile(all_composites, 80)),          # Top 20%
            'cautious': float(np.percentile(all_composites, 60)),     # Top 40%
            'hold': float(np.percentile(all_composites, 40)),         # Top 60%
        }

        print(f"\n📊 Composite推荐阈值 (基于{n_stocks_total:,}样本):")
        cw_str = " + ".join(f"pred_{k}×{v}" for k, v in composite_weights.items() if v > 0)
        print(f"  composite = {cw_str}")
        print(f"  composite 分布: min={all_composites.min():.6f}, P50={np.percentile(all_composites, 50):.6f}, max={all_composites.max():.6f}")
        print(f"  强烈买入 (Top  5%): composite ≥ {rec_thresholds['strong_buy']:.6f}")
        print(f"  买入     (Top 20%): composite ≥ {rec_thresholds['buy']:.6f}")
        print(f"  谨慎买入 (Top 40%): composite ≥ {rec_thresholds['cautious']:.6f}")
        print(f"  观望     (Top 60%): composite ≥ {rec_thresholds['hold']:.6f}")
        print(f"  回避     (Bottom 40%)")

        # 保存为 JSON sidecar
        rec_path = model_dir / 'recommendation_thresholds.json'
        with open(rec_path, 'w') as f:
            json.dump(rec_thresholds, f, indent=2)
        print(f"\n✅ 推荐阈值已保存: {rec_path}")

        # 嵌入模型 pkl
        _embed_thresholds_in_model(model_dir, rec_thresholds, version)


def _embed_thresholds_in_model(model_dir: Path, rec_thresholds: dict, version: str):
    """将 recommendation_thresholds 嵌入模型 pkl 文件"""
    version_map = {
        'v3.96': ['v396_*.pkl', 'v395_multi_target_*.pkl'],
        'v4.3': ['v43_*.pkl'],
        'v4.4': ['v44_*.pkl'],
        'v4.6': ['v46_*.pkl'],
        'v4.7.3': ['v473_*.pkl'],
        'v4.7.5': ['v475_*.pkl'],
        'v4.8': ['v48_*.pkl'],
    }
    patterns = version_map.get(version, [])
    model_files = []
    for pat in patterns:
        model_files.extend(model_dir.glob(pat))

    if not model_files:
        print(f"  ⚠️ 无模型文件可嵌入阈值")
        return

    latest = max(model_files, key=lambda f: f.stat().st_mtime)
    print(f"  嵌入阈值到: {latest.name}")

    try:
        model_data = joblib.load(latest)
    except Exception:
        with open(latest, 'rb') as f:
            model_data = pickle.load(f)

    model_data['recommendation_thresholds'] = rec_thresholds
    joblib.dump(model_data, latest)
    print(f"  ✅ 阈值已嵌入模型文件 ({latest.stat().st_size / 1024 / 1024:.1f} MB)")


def main():
    all_versions = ['v3.96', 'v4.3', 'v4.4', 'v4.6', 'v4.7.3', 'v4.7.5', 'v4.8']

    parser = argparse.ArgumentParser(description='为已有模型计算全局分位数校准')
    parser.add_argument('--version', choices=all_versions,
                        help='模型版本')
    parser.add_argument('--all', action='store_true',
                        help='校准所有活跃版本')
    parser.add_argument('--n-quantiles', type=int, default=1001,
                        help='分位点数量 (默认1001)')
    parser.add_argument('--with-recommendation', action='store_true',
                        help='同时计算composite推荐阈值 (强烈买入/买入/谨慎/观望/回避)')

    args = parser.parse_args()

    if not args.version and not args.all:
        parser.print_help()
        print("\n请指定 --version 或 --all")
        return

    start = datetime.now()

    if args.all:
        for v in all_versions:
            try:
                calibrate_version(v, args.n_quantiles, args.with_recommendation)
            except Exception as e:
                print(f"  ⚠️ {v} 校准失败: {e}")
    else:
        calibrate_version(args.version, args.n_quantiles, args.with_recommendation)

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n校准完成, 总耗时: {elapsed:.0f}秒 ({elapsed / 60:.1f}分钟)")


if __name__ == '__main__':
    main()
