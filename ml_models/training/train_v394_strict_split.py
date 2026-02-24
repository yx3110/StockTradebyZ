#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.9.4 严格时间分割训练脚本

关键设计：
1. 严格时间分割：训练集 → 验证集 → 测试集，按时间顺序排列
2. Gap buffer: 在各个数据集之间添加缓冲期，防止label泄露
3. 5日收益label意味着需要5天以上的gap
4. 确保训练时不使用任何未来信息

数据分割示例 (假设label是5日收益):
- 训练集: 2024-01-01 ~ 2024-12-15
- Gap1: 2024-12-16 ~ 2024-12-22 (7天缓冲)
- 验证集: 2024-12-23 ~ 2025-06-30
- Gap2: 2025-07-01 ~ 2025-07-07 (7天缓冲)
- 测试集: 2025-07-08 ~ 2025-11-28

作者: Claude Code
创建时间: 2025-11-28
"""

import sys
import numpy as np
import pandas as pd
import sqlite3
import json
from datetime import datetime, timedelta
import logging
from pathlib import Path
import joblib
from tqdm import tqdm
import argparse

import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')


def load_merged_features(min_date=None, max_date=None):
    """
    加载并合并 v3.9.0 特征和活跃市值特征

    Args:
        min_date: 最小日期筛选
        max_date: 最大日期筛选
    """
    conn = sqlite3.connect(DB_PATH)

    logger.info("加载并合并特征数据...")

    # 构建日期筛选条件
    date_filter = ""
    date_params = []
    if min_date:
        date_filter += " AND v.trade_date >= ?"
        date_params.append(min_date)
    if max_date:
        date_filter += " AND v.trade_date <= ?"
        date_params.append(max_date)

    # 使用JOIN直接合并两个缓存表
    query = f"""
    SELECT
        v.code,
        v.trade_date,
        v.features_json,
        v.label_5d,
        a.market_active_mv_ratio,
        a.market_active_mv_zscore,
        a.market_active_mv_trend,
        a.stock_active_mv_rank,
        a.stock_relative_liquidity,
        a.market_cap_quality_score
    FROM v39_feature_cache v
    INNER JOIN active_mv_feature_cache a
        ON v.code = a.code AND v.trade_date = a.trade_date
    WHERE v.label_5d IS NOT NULL
        {date_filter}
    ORDER BY v.trade_date, v.code
    """

    logger.info("  执行SQL JOIN查询...")
    df = pd.read_sql(query, conn, params=date_params if date_params else None)
    logger.info(f"  合并数据: {len(df)} 条")

    if df.empty:
        conn.close()
        return None

    # 解析JSON特征
    logger.info("  解析 JSON 特征...")
    features_list = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="解析特征"):
        try:
            features = json.loads(row['features_json'])
            features['code'] = row['code']
            features['trade_date'] = row['trade_date']
            features['label'] = row['label_5d']
            # 添加活跃市值特征
            features['market_active_mv_ratio'] = row['market_active_mv_ratio']
            features['market_active_mv_zscore'] = row['market_active_mv_zscore']
            features['market_active_mv_trend'] = row['market_active_mv_trend']
            features['stock_active_mv_rank'] = row['stock_active_mv_rank']
            features['stock_relative_liquidity'] = row['stock_relative_liquidity']
            features['market_cap_quality_score'] = row['market_cap_quality_score']
            features_list.append(features)
        except:
            continue

    result = pd.DataFrame(features_list)
    logger.info(f"  解析完成: {len(result)} 条, {len(result.columns) - 3} 个特征")

    conn.close()
    return result


def strict_time_split(df, train_end, val_start, val_end, test_start, gap_days=7):
    """
    严格时间分割，确保无数据泄露

    Args:
        df: 完整数据集
        train_end: 训练集结束日期
        val_start: 验证集开始日期 (应该 > train_end + gap_days)
        val_end: 验证集结束日期
        test_start: 测试集开始日期 (应该 > val_end + gap_days)
        gap_days: 缓冲期天数

    Returns:
        train_df, val_df, test_df
    """
    df['trade_date'] = pd.to_datetime(df['trade_date'])

    train_end_dt = pd.to_datetime(train_end)
    val_start_dt = pd.to_datetime(val_start)
    val_end_dt = pd.to_datetime(val_end)
    test_start_dt = pd.to_datetime(test_start)

    # 检查gap是否足够
    gap1 = (val_start_dt - train_end_dt).days
    gap2 = (test_start_dt - val_end_dt).days

    logger.info(f"\n=== 严格时间分割检查 ===")
    logger.info(f"训练集 → 验证集 Gap: {gap1} 天 (要求 >= {gap_days})")
    logger.info(f"验证集 → 测试集 Gap: {gap2} 天 (要求 >= {gap_days})")

    if gap1 < gap_days:
        logger.warning(f"⚠️ Gap1 不足! 可能存在数据泄露风险!")
    if gap2 < gap_days:
        logger.warning(f"⚠️ Gap2 不足! 可能存在数据泄露风险!")

    # 分割数据
    train_df = df[df['trade_date'] <= train_end]
    val_df = df[(df['trade_date'] >= val_start) & (df['trade_date'] <= val_end)]
    test_df = df[df['trade_date'] >= test_start]

    logger.info(f"\n数据集划分:")
    logger.info(f"  训练集: {len(train_df):,} 条 ({train_df['trade_date'].min().date()} ~ {train_df['trade_date'].max().date()})")
    logger.info(f"  验证集: {len(val_df):,} 条 ({val_df['trade_date'].min().date()} ~ {val_df['trade_date'].max().date()})")
    logger.info(f"  测试集: {len(test_df):,} 条 ({test_df['trade_date'].min().date()} ~ {test_df['trade_date'].max().date()})")

    # 检查日期数量
    train_dates = train_df['trade_date'].nunique()
    val_dates = val_df['trade_date'].nunique()
    test_dates = test_df['trade_date'].nunique()
    logger.info(f"\n交易日数:")
    logger.info(f"  训练集: {train_dates} 天")
    logger.info(f"  验证集: {val_dates} 天")
    logger.info(f"  测试集: {test_dates} 天")

    return train_df, val_df, test_df


def train_model(X_train, y_train, X_val=None, y_val=None, params=None):
    """
    训练 LightGBM 模型
    """
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'learning_rate': 0.03,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 50,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbose': -1,
            'n_jobs': -1,
            'seed': 42
        }

    train_data = lgb.Dataset(X_train, label=y_train)

    callbacks = [
        lgb.log_evaluation(period=100)
    ]

    if X_val is not None and y_val is not None:
        val_data = lgb.Dataset(X_val, label=y_val)
        callbacks.append(lgb.early_stopping(stopping_rounds=50))
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=callbacks
        )
    else:
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data],
            callbacks=callbacks
        )

    return model


def evaluate_model(model, X_test, y_test, test_info, model_name=""):
    """
    评估模型性能
    """
    y_pred = model.predict(X_test)

    # 基础指标
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # IC (Information Coefficient)
    ic = np.corrcoef(y_test, y_pred)[0, 1] if len(y_test) > 1 else 0

    # 方向准确率
    direction_acc = ((y_test > 0) == (y_pred > 0)).mean()

    # Top N 分析
    test_result = test_info.copy()
    test_result['pred'] = y_pred
    test_result['actual'] = y_test.values

    top_results = {}
    for n in [10, 20, 50]:
        top_returns = []
        for date, group in test_result.groupby('trade_date'):
            if len(group) >= n:
                top_n = group.nlargest(n, 'pred')
                top_returns.extend(top_n['actual'].tolist())

        if top_returns:
            top_results[f'top_{n}_return'] = np.mean(top_returns) * 100
            top_results[f'top_{n}_win_rate'] = (np.array(top_returns) > 0).mean() * 100
        else:
            top_results[f'top_{n}_return'] = 0
            top_results[f'top_{n}_win_rate'] = 0

    # 按日期计算IC
    daily_ics = []
    for date, group in test_result.groupby('trade_date'):
        if len(group) > 10:
            ic_day = np.corrcoef(group['actual'], group['pred'])[0, 1]
            if not np.isnan(ic_day):
                daily_ics.append(ic_day)

    ic_mean = np.mean(daily_ics) if daily_ics else 0
    ic_ir = ic_mean / np.std(daily_ics) if daily_ics and np.std(daily_ics) > 0 else 0

    # 特征重要性
    importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    results = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'ic': ic,
        'ic_mean': ic_mean,
        'ic_ir': ic_ir,
        'direction_accuracy': direction_acc,
        **top_results,
        'feature_importance': importance,
        'n_test_samples': len(y_test),
        'n_test_days': test_info['trade_date'].nunique()
    }

    return results


def print_comparison(results_v394, results_v390):
    """打印A/B对比结果"""

    print("\n" + "=" * 80)
    print("                    A/B 对比: V3.9.4 (48特征) vs V3.9.0 (42特征)")
    print("=" * 80)

    metrics = [
        ('RMSE', 'rmse', '{:.4f}', False),
        ('MAE', 'mae', '{:.4f}', False),
        ('R²', 'r2', '{:.4f}', True),
        ('IC (整体)', 'ic', '{:.4f}', True),
        ('IC (日均)', 'ic_mean', '{:.4f}', True),
        ('IC IR', 'ic_ir', '{:.2f}', True),
        ('方向准确率', 'direction_accuracy', '{:.2%}', True),
        ('Top 10 收益', 'top_10_return', '{:.2f}%', True),
        ('Top 10 胜率', 'top_10_win_rate', '{:.2f}%', True),
        ('Top 20 收益', 'top_20_return', '{:.2f}%', True),
        ('Top 20 胜率', 'top_20_win_rate', '{:.2f}%', True),
        ('Top 50 收益', 'top_50_return', '{:.2f}%', True),
        ('Top 50 胜率', 'top_50_win_rate', '{:.2f}%', True),
    ]

    print(f"\n{'指标':<15} {'V3.9.0':<12} {'V3.9.4':<12} {'变化':<12} {'胜负':<6}")
    print("-" * 60)

    wins_v394 = 0
    wins_v390 = 0

    for name, key, fmt, higher_better in metrics:
        v390_val = results_v390[key]
        v394_val = results_v394[key]

        if '%' in fmt:
            diff = v394_val - v390_val
            diff_str = f"{diff:+.2f}%"
        elif 'rmse' in key or 'mae' in key:
            diff = v394_val - v390_val
            diff_str = f"{diff:+.4f}"
        else:
            diff = v394_val - v390_val
            diff_str = fmt.format(diff).replace('{', '').replace('}', '')
            if not diff_str.startswith('-'):
                diff_str = '+' + diff_str

        # 判断胜负
        if higher_better:
            if v394_val > v390_val + 0.001:
                winner = "V3.9.4 ✓"
                wins_v394 += 1
            elif v390_val > v394_val + 0.001:
                winner = "V3.9.0"
                wins_v390 += 1
            else:
                winner = "平局"
        else:
            if v394_val < v390_val - 0.001:
                winner = "V3.9.4 ✓"
                wins_v394 += 1
            elif v390_val < v394_val - 0.001:
                winner = "V3.9.0"
                wins_v390 += 1
            else:
                winner = "平局"

        v390_str = fmt.format(v390_val)
        v394_str = fmt.format(v394_val)

        print(f"{name:<15} {v390_str:<12} {v394_str:<12} {diff_str:<12} {winner:<6}")

    print("-" * 60)
    print(f"\n总结: V3.9.4 胜 {wins_v394} 项, V3.9.0 胜 {wins_v390} 项")

    if wins_v394 > wins_v390:
        print("结论: V3.9.4 (含活跃市值特征) 整体表现更优")
    elif wins_v390 > wins_v394:
        print("结论: V3.9.0 (原始特征) 整体表现更优")
    else:
        print("结论: 两个版本表现相当")


def main():
    parser = argparse.ArgumentParser(description='V3.9.4 严格时间分割训练')
    parser.add_argument('--train-end', default='2025-06-30', help='训练集结束日期')
    parser.add_argument('--val-start', default='2025-07-08', help='验证集开始日期')
    parser.add_argument('--val-end', default='2025-09-20', help='验证集结束日期')
    parser.add_argument('--test-start', default='2025-09-28', help='测试集开始日期')
    parser.add_argument('--gap-days', type=int, default=7, help='数据集间缓冲天数')

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("V3.9.4 严格时间分割训练 (防数据泄露)")
    logger.info("=" * 70)

    # 1. 加载数据
    df = load_merged_features()

    if df is None or df.empty:
        logger.error("无法加载数据，请先运行 backfill_active_mv_for_v39.py")
        return

    # 显示数据覆盖范围
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    logger.info(f"\n数据覆盖范围: {df['trade_date'].min().date()} ~ {df['trade_date'].max().date()}")
    logger.info(f"总样本数: {len(df):,}")
    logger.info(f"总交易日: {df['trade_date'].nunique()}")

    # 2. 严格时间分割
    train_df, val_df, test_df = strict_time_split(
        df,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        test_start=args.test_start,
        gap_days=args.gap_days
    )

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        logger.error("数据分割后某个集合为空，请调整日期参数")
        return

    # 3. 准备特征
    exclude_cols = ['code', 'trade_date', 'label']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # 定义活跃市值特征
    active_mv_features = [
        'market_active_mv_ratio', 'market_active_mv_zscore', 'market_active_mv_trend',
        'stock_active_mv_rank', 'stock_relative_liquidity', 'market_cap_quality_score'
    ]

    # v3.9.0 原有特征
    v39_features = [f for f in feature_cols if f not in active_mv_features]

    logger.info(f"\nv3.9.0 原有特征: {len(v39_features)} 个")
    logger.info(f"活跃市值特征: {len([f for f in active_mv_features if f in feature_cols])} 个")
    logger.info(f"总特征: {len(feature_cols)} 个")

    # 4. 准备训练数据
    X_train = train_df[feature_cols].fillna(0).astype(float)
    y_train = train_df['label'].astype(float)

    X_val = val_df[feature_cols].fillna(0).astype(float)
    y_val = val_df['label'].astype(float)

    X_test = test_df[feature_cols].fillna(0).astype(float)
    y_test = test_df['label'].astype(float)
    test_info = test_df[['code', 'trade_date']].copy()

    # ============================================================
    # 5. 训练 v3.9.4 完整模型 (48 特征)
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("训练 V3.9.4 完整模型 (48 特征)")
    logger.info("=" * 70)

    model_v394 = train_model(X_train, y_train, X_val, y_val)
    results_v394 = evaluate_model(model_v394, X_test, y_test, test_info, "V3.9.4")

    logger.info(f"\nV3.9.4 测试集评估:")
    logger.info(f"  样本数: {results_v394['n_test_samples']:,}")
    logger.info(f"  交易日: {results_v394['n_test_days']}")
    logger.info(f"  RMSE: {results_v394['rmse']:.4f}")
    logger.info(f"  IC: {results_v394['ic']:.4f}")
    logger.info(f"  IC (日均): {results_v394['ic_mean']:.4f}")
    logger.info(f"  Top 20 收益: {results_v394['top_20_return']:.2f}%")
    logger.info(f"  Top 20 胜率: {results_v394['top_20_win_rate']:.2f}%")

    # ============================================================
    # 6. 训练 v3.9.0 基准模型 (42 特征)
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("训练 V3.9.0 基准模型 (42 特征)")
    logger.info("=" * 70)

    X_train_v39 = X_train[v39_features]
    X_val_v39 = X_val[v39_features]
    X_test_v39 = X_test[v39_features]

    model_v390 = train_model(X_train_v39, y_train, X_val_v39, y_val)
    results_v390 = evaluate_model(model_v390, X_test_v39, y_test, test_info, "V3.9.0")

    logger.info(f"\nV3.9.0 测试集评估:")
    logger.info(f"  样本数: {results_v390['n_test_samples']:,}")
    logger.info(f"  交易日: {results_v390['n_test_days']}")
    logger.info(f"  RMSE: {results_v390['rmse']:.4f}")
    logger.info(f"  IC: {results_v390['ic']:.4f}")
    logger.info(f"  IC (日均): {results_v390['ic_mean']:.4f}")
    logger.info(f"  Top 20 收益: {results_v390['top_20_return']:.2f}%")
    logger.info(f"  Top 20 胜率: {results_v390['top_20_win_rate']:.2f}%")

    # ============================================================
    # 7. A/B 对比
    # ============================================================
    print_comparison(results_v394, results_v390)

    # 活跃市值特征重要性
    logger.info("\n活跃市值特征在 V3.9.4 中的重要性排名:")
    importance = results_v394['feature_importance']
    for feat in active_mv_features:
        if feat in importance['feature'].values:
            rank = importance[importance['feature'] == feat].index[0] + 1
            imp = importance[importance['feature'] == feat]['importance'].values[0]
            logger.info(f"  #{rank:2d} {feat}: {imp:.1f}")

    # Top 20 特征
    logger.info("\nV3.9.4 Top 20 特征:")
    for i, (_, row) in enumerate(importance.head(20).iterrows()):
        marker = "⭐" if row['feature'] in active_mv_features else "  "
        logger.info(f"  {marker} {i+1:2d}. {row['feature']}: {row['importance']:.1f}")

    # ============================================================
    # 8. 保存模型
    # ============================================================
    model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v394'
    model_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    model_path = model_dir / f'v394_strict_split_{timestamp}.pkl'
    joblib.dump({
        'model': model_v394,
        'feature_cols': feature_cols,
        'v39_features': v39_features,
        'active_mv_features': active_mv_features,
        'results': {k: v for k, v in results_v394.items() if k != 'feature_importance'},
        'split_config': {
            'train_end': args.train_end,
            'val_start': args.val_start,
            'val_end': args.val_end,
            'test_start': args.test_start,
            'gap_days': args.gap_days
        },
        'data_info': {
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_test': len(X_test),
            'train_dates': train_df['trade_date'].nunique(),
            'val_dates': val_df['trade_date'].nunique(),
            'test_dates': test_df['trade_date'].nunique()
        }
    }, model_path)

    logger.info(f"\n✅ V3.9.4 模型已保存: {model_path}")

    # 保存对比报告
    report_dir = (PROJECT_ROOT / str(PROJECT_ROOT / 'reports' / 'v394_evaluation')
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / f'strict_split_comparison_{timestamp}.md'
    with open(report_path, 'w') as f:
        f.write("# V3.9.4 vs V3.9.0 A/B对比报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 数据分割配置 (严格时间分割)\n\n")
        f.write(f"- 训练集: ~ {args.train_end}\n")
        f.write(f"- 验证集: {args.val_start} ~ {args.val_end}\n")
        f.write(f"- 测试集: {args.test_start} ~\n")
        f.write(f"- Gap缓冲: {args.gap_days} 天\n\n")

        f.write("## 数据量\n\n")
        f.write(f"| 数据集 | 样本数 | 交易日 |\n")
        f.write(f"|--------|--------|--------|\n")
        f.write(f"| 训练集 | {len(X_train):,} | {train_df['trade_date'].nunique()} |\n")
        f.write(f"| 验证集 | {len(X_val):,} | {val_df['trade_date'].nunique()} |\n")
        f.write(f"| 测试集 | {len(X_test):,} | {test_df['trade_date'].nunique()} |\n\n")

        f.write("## A/B对比结果\n\n")
        f.write(f"| 指标 | V3.9.0 (42特征) | V3.9.4 (48特征) | 变化 |\n")
        f.write(f"|------|-----------------|-----------------|------|\n")
        f.write(f"| RMSE | {results_v390['rmse']:.4f} | {results_v394['rmse']:.4f} | {results_v394['rmse']-results_v390['rmse']:+.4f} |\n")
        f.write(f"| IC | {results_v390['ic']:.4f} | {results_v394['ic']:.4f} | {results_v394['ic']-results_v390['ic']:+.4f} |\n")
        f.write(f"| IC (日均) | {results_v390['ic_mean']:.4f} | {results_v394['ic_mean']:.4f} | {results_v394['ic_mean']-results_v390['ic_mean']:+.4f} |\n")
        f.write(f"| 方向准确率 | {results_v390['direction_accuracy']:.2%} | {results_v394['direction_accuracy']:.2%} | {(results_v394['direction_accuracy']-results_v390['direction_accuracy'])*100:+.2f}% |\n")
        f.write(f"| Top 10 收益 | {results_v390['top_10_return']:.2f}% | {results_v394['top_10_return']:.2f}% | {results_v394['top_10_return']-results_v390['top_10_return']:+.2f}% |\n")
        f.write(f"| Top 10 胜率 | {results_v390['top_10_win_rate']:.2f}% | {results_v394['top_10_win_rate']:.2f}% | {results_v394['top_10_win_rate']-results_v390['top_10_win_rate']:+.2f}% |\n")
        f.write(f"| Top 20 收益 | {results_v390['top_20_return']:.2f}% | {results_v394['top_20_return']:.2f}% | {results_v394['top_20_return']-results_v390['top_20_return']:+.2f}% |\n")
        f.write(f"| Top 20 胜率 | {results_v390['top_20_win_rate']:.2f}% | {results_v394['top_20_win_rate']:.2f}% | {results_v394['top_20_win_rate']-results_v390['top_20_win_rate']:+.2f}% |\n\n")

        f.write("## 活跃市值特征重要性\n\n")
        for feat in active_mv_features:
            if feat in importance['feature'].values:
                rank = importance[importance['feature'] == feat].index[0] + 1
                imp = importance[importance['feature'] == feat]['importance'].values[0]
                f.write(f"- #{rank}: {feat} ({imp:.1f})\n")

    logger.info(f"✅ 对比报告已保存: {report_path}")

    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
