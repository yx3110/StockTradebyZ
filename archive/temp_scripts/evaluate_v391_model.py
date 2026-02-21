#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.91多周期模型完整评估脚本
使用ModelEvaluator评估5d、10d、15d各周期性能及综合性能
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import sqlite3
import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime
from model_evaluator import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_data(db_path='data_adapter/stock_data.db', test_ratio=0.2, random_state=42):
    """
    加载测试数据

    Returns:
        X: 特征矩阵
        y_5d, y_10d, y_15d: 各周期实际收益率
    """
    logger.info("=" * 80)
    logger.info("📥 加载V3.91评估数据...")
    logger.info("=" * 80)

    conn = sqlite3.connect(db_path)

    # 查询有完整多周期标签的样本
    query = """
        SELECT code, trade_date, features_json, label_5d, label_10d, label_15d
        FROM v39_feature_cache
        WHERE label_5d IS NOT NULL
        AND label_10d IS NOT NULL
        AND label_15d IS NOT NULL
        ORDER BY trade_date, code
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    logger.info(f"✅ 加载 {len(df):,} 个完整样本")

    # 解析特征JSON
    import json
    features_list = []
    valid_indices = []

    for idx, row in df.iterrows():
        try:
            features_dict = json.loads(row['features_json'])
            features_list.append(features_dict)
            valid_indices.append(idx)
        except:
            continue

    X = pd.DataFrame(features_list)
    df_valid = df.iloc[valid_indices].reset_index(drop=True)

    # 处理缺失值
    X = X.fillna(0)

    # 划分训练集和测试集（使用相同的随机种子确保一致性）
    np.random.seed(random_state)
    n_samples = len(X)
    test_size = int(n_samples * test_ratio)
    indices = np.random.permutation(n_samples)
    test_indices = indices[:test_size]

    X_test = X.iloc[test_indices].values
    y_5d_test = df_valid.iloc[test_indices]['label_5d'].values
    y_10d_test = df_valid.iloc[test_indices]['label_10d'].values
    y_15d_test = df_valid.iloc[test_indices]['label_15d'].values

    logger.info(f"✅ 测试集: {len(X_test):,} 个样本")

    return X_test, y_5d_test, y_10d_test, y_15d_test, X.columns.tolist()


def load_v391_model(model_path='ml_models/trained_models/v391/v391_multiperiod_latest.pkl'):
    """加载V3.91模型"""
    logger.info(f"📦 加载V3.91模型: {model_path}")

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    logger.info(f"✅ 模型版本: {model_data.get('version', 'unknown')}")
    logger.info(f"✅ 特征数量: {len(model_data.get('feature_columns', []))}")
    logger.info(f"✅ 周期权重: 5d={model_data['period_weights']['5d']}, "
               f"10d={model_data['period_weights']['10d']}, "
               f"15d={model_data['period_weights']['15d']}")

    return model_data


def predict_with_model(X, feature_columns, model_data, period):
    """使用模型预测指定周期的收益率"""
    # 对齐特征列
    X_df = pd.DataFrame(X, columns=feature_columns)
    model_feature_cols = model_data['feature_columns']

    # 重新索引以匹配模型特征
    X_aligned = X_df.reindex(columns=model_feature_cols, fill_value=0).values

    # 基础模型预测
    base_models = model_data['base_models'][period]
    base_preds = np.array([
        model.predict(X_aligned) for model in base_models.values()
    ]).T  # Shape: (n_samples, n_models)

    # 元模型预测
    meta_model = model_data['meta_models'][period]
    predictions = meta_model.predict(base_preds)

    return predictions


def evaluate_period(y_true, y_pred, period_name):
    """评估单个周期的性能"""
    print(f"\n{'=' * 80}")
    print(f"📊 {period_name} 周期模型评估")
    print(f"{'=' * 80}")

    evaluator = ModelEvaluator(y_true, y_pred)
    results = evaluator.full_evaluation()

    print(f"\n【基础准确性指标】")
    print(f"  方向准确率:    {results['direction_accuracy']*100:.2f}%  " +
          ("✅" if results['direction_accuracy'] > 0.55 else
           "🟡" if results['direction_accuracy'] > 0.50 else "❌"))
    print(f"  IC (信息系数):  {results['ic']:.4f}         " +
          ("✅" if results['ic'] > 0.05 else
           "🟡" if results['ic'] > 0.03 else "❌"))
    print(f"  R²:            {results['r2']:.4f}         " +
          ("✅" if results['r2'] > 0.15 else
           "🟡" if results['r2'] > 0.10 else "❌"))
    print(f"  MAE:           {results['mae']:.4f} ({results['mae']*100:.2f}%)")
    print(f"  RMSE:          {results['rmse']:.4f} ({results['rmse']*100:.2f}%)")

    print(f"\n【金融实战指标】")
    print(f"  Top 20平均收益: {results['top_20_mean_return']*100:.2f}%    " +
          ("✅" if results['top_20_mean_return'] > 0.02 else
           "🟡" if results['top_20_mean_return'] > 0 else "❌"))
    print(f"  Top 20胜率:     {results['top_20_positive_rate']*100:.2f}%")
    print(f"  Top 50平均收益: {results['top_50_mean_return']*100:.2f}%")
    print(f"  Top 50胜率:     {results['top_50_positive_rate']*100:.2f}%")
    print(f"  分位数单调性:   {'是' if results['quantile_analysis']['is_monotonic'] else '否'}           " +
          ("✅" if results['quantile_analysis']['is_monotonic'] else "❌"))

    print(f"\n【分位数详情】")
    for q in results['quantile_analysis']['quantile_returns']:
        print(f"  Q{q['quantile']} (n={q['count']:5d}): " +
              f"收益={q['mean_return']*100:+.2f}%, 胜率={q['win_rate']*100:.1f}%")

    print(f"\n【胜率与盈亏比 (Top 20%)】")
    print(f"  胜率:    {results['win_rate']*100:.2f}%")
    print(f"  盈亏比:  {results['profit_factor']:.2f}")
    print(f"  平均盈利: {results['avg_win']*100:.2f}%")
    print(f"  平均亏损: {results['avg_loss']*100:.2f}%")

    print(f"\n{'=' * 80}")
    print(f"📈 {period_name} 综合评分: {results['comprehensive']['score']:.1f}/100  " +
          f"等级: {results['comprehensive']['grade']}")
    print(f"{'=' * 80}")

    return results


def evaluate_composite(y_true_5d, y_pred_5d, y_true_10d, y_pred_10d,
                       y_true_15d, y_pred_15d, weights):
    """评估综合性能"""
    print(f"\n{'=' * 80}")
    print(f"📊 综合多周期模型评估")
    print(f"   权重: 5d={weights['5d']}, 10d={weights['10d']}, 15d={weights['15d']}")
    print(f"{'=' * 80}")

    # 加权组合
    y_true_composite = (
        weights['5d'] * y_true_5d +
        weights['10d'] * y_true_10d +
        weights['15d'] * y_true_15d
    )

    y_pred_composite = (
        weights['5d'] * y_pred_5d +
        weights['10d'] * y_pred_10d +
        weights['15d'] * y_pred_15d
    )

    evaluator = ModelEvaluator(y_true_composite, y_pred_composite)
    results = evaluator.full_evaluation()

    print(f"\n【基础准确性指标】")
    print(f"  方向准确率:    {results['direction_accuracy']*100:.2f}%  " +
          ("✅" if results['direction_accuracy'] > 0.55 else
           "🟡" if results['direction_accuracy'] > 0.50 else "❌"))
    print(f"  IC (信息系数):  {results['ic']:.4f}         " +
          ("✅" if results['ic'] > 0.05 else
           "🟡" if results['ic'] > 0.03 else "❌"))
    print(f"  R²:            {results['r2']:.4f}         " +
          ("✅" if results['r2'] > 0.15 else
           "🟡" if results['r2'] > 0.10 else "❌"))
    print(f"  MAE:           {results['mae']:.4f} ({results['mae']*100:.2f}%)")
    print(f"  RMSE:          {results['rmse']:.4f} ({results['rmse']*100:.2f}%)")

    print(f"\n【金融实战指标】")
    print(f"  Top 20平均收益: {results['top_20_mean_return']*100:.2f}%    " +
          ("✅" if results['top_20_mean_return'] > 0.02 else
           "🟡" if results['top_20_mean_return'] > 0 else "❌"))
    print(f"  Top 20胜率:     {results['top_20_positive_rate']*100:.2f}%")
    print(f"  Top 50平均收益: {results['top_50_mean_return']*100:.2f}%")
    print(f"  Top 50胜率:     {results['top_50_positive_rate']*100:.2f}%")
    print(f"  分位数单调性:   {'是' if results['quantile_analysis']['is_monotonic'] else '否'}           " +
          ("✅" if results['quantile_analysis']['is_monotonic'] else "❌"))

    print(f"\n【分位数详情】")
    for q in results['quantile_analysis']['quantile_returns']:
        print(f"  Q{q['quantile']} (n={q['count']:5d}): " +
              f"收益={q['mean_return']*100:+.2f}%, 胜率={q['win_rate']*100:.1f}%")

    print(f"\n【胜率与盈亏比 (Top 20%)】")
    print(f"  胜率:    {results['win_rate']*100:.2f}%")
    print(f"  盈亏比:  {results['profit_factor']:.2f}")
    print(f"  平均盈利: {results['avg_win']*100:.2f}%")
    print(f"  平均亏损: {results['avg_loss']*100:.2f}%")

    print(f"\n{'=' * 80}")
    print(f"📈 综合评分: {results['comprehensive']['score']:.1f}/100  " +
          f"等级: {results['comprehensive']['grade']}")
    print(f"{'=' * 80}")

    return results


def main():
    """主评估流程"""
    print("\n" + "=" * 80)
    print("🔍 V3.91 多周期模型完整评估")
    print("=" * 80)
    print(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 1. 加载测试数据
    X_test, y_5d, y_10d, y_15d, feature_columns = load_test_data()

    # 2. 加载模型
    model_data = load_v391_model()

    # 3. 生成预测
    logger.info("\n🔮 生成多周期预测...")

    pred_5d = predict_with_model(X_test, feature_columns, model_data, '5d')
    pred_10d = predict_with_model(X_test, feature_columns, model_data, '10d')
    pred_15d = predict_with_model(X_test, feature_columns, model_data, '15d')

    logger.info(f"✅ 5d预测: min={pred_5d.min():.4f}, max={pred_5d.max():.4f}, mean={pred_5d.mean():.4f}")
    logger.info(f"✅ 10d预测: min={pred_10d.min():.4f}, max={pred_10d.max():.4f}, mean={pred_10d.mean():.4f}")
    logger.info(f"✅ 15d预测: min={pred_15d.min():.4f}, max={pred_15d.max():.4f}, mean={pred_15d.mean():.4f}")

    # 4. 分周期评估
    results = {}

    results['5d'] = evaluate_period(y_5d, pred_5d, "5天")
    results['10d'] = evaluate_period(y_10d, pred_10d, "10天")
    results['15d'] = evaluate_period(y_15d, pred_15d, "15天")

    # 5. 综合评估
    weights = model_data['period_weights']
    results['composite'] = evaluate_composite(
        y_5d, pred_5d, y_10d, pred_10d, y_15d, pred_15d, weights
    )

    # 6. 总结报告
    print("\n" + "=" * 80)
    print("📊 V3.91 模型评估总结")
    print("=" * 80)

    print("\n【各周期综合评分】")
    print(f"  5天周期:   {results['5d']['comprehensive']['score']:.1f}/100  等级: {results['5d']['comprehensive']['grade']}")
    print(f"  10天周期:  {results['10d']['comprehensive']['score']:.1f}/100  等级: {results['10d']['comprehensive']['grade']}")
    print(f"  15天周期:  {results['15d']['comprehensive']['score']:.1f}/100  等级: {results['15d']['comprehensive']['grade']}")
    print(f"  综合评分:  {results['composite']['comprehensive']['score']:.1f}/100  等级: {results['composite']['comprehensive']['grade']}")

    print("\n【方向准确率对比】")
    print(f"  5天:  {results['5d']['direction_accuracy']*100:.2f}%")
    print(f"  10天: {results['10d']['direction_accuracy']*100:.2f}%")
    print(f"  15天: {results['15d']['direction_accuracy']*100:.2f}%")
    print(f"  综合: {results['composite']['direction_accuracy']*100:.2f}%")

    print("\n【IC (信息系数) 对比】")
    print(f"  5天:  {results['5d']['ic']:.4f}")
    print(f"  10天: {results['10d']['ic']:.4f}")
    print(f"  15天: {results['15d']['ic']:.4f}")
    print(f"  综合: {results['composite']['ic']:.4f}")

    print("\n【R² 对比】")
    print(f"  5天:  {results['5d']['r2']:.4f}")
    print(f"  10天: {results['10d']['r2']:.4f}")
    print(f"  15天: {results['15d']['r2']:.4f}")
    print(f"  综合: {results['composite']['r2']:.4f}")

    print("\n【Top 20 表现对比】")
    print(f"  5天:  收益={results['5d']['top_20_mean_return']*100:.2f}%, 胜率={results['5d']['top_20_positive_rate']*100:.1f}%")
    print(f"  10天: 收益={results['10d']['top_20_mean_return']*100:.2f}%, 胜率={results['10d']['top_20_positive_rate']*100:.1f}%")
    print(f"  15天: 收益={results['15d']['top_20_mean_return']*100:.2f}%, 胜率={results['15d']['top_20_positive_rate']*100:.1f}%")
    print(f"  综合: 收益={results['composite']['top_20_mean_return']*100:.2f}%, 胜率={results['composite']['top_20_positive_rate']*100:.1f}%")

    # 7. 最终建议
    avg_score = (
        results['5d']['comprehensive']['score'] +
        results['10d']['comprehensive']['score'] +
        results['15d']['comprehensive']['score']
    ) / 3

    print("\n" + "=" * 80)
    print(f"🎯 V3.91 模型最终评价")
    print("=" * 80)
    print(f"  平均周期评分: {avg_score:.1f}/100")
    print(f"  综合模型评分: {results['composite']['comprehensive']['score']:.1f}/100")

    if avg_score >= 70:
        print("\n✅ 建议：V3.91多周期模型表现优秀，可以考虑实盘使用")
    elif avg_score >= 60:
        print("\n🟡 建议：V3.91多周期模型表现良好，建议先模拟盘测试")
    elif avg_score >= 50:
        print("\n⚠️  建议：V3.91多周期模型表现一般，需要继续优化")
    else:
        print("\n❌ 建议：V3.91多周期模型表现不佳，需要重新训练或调整结构")

    print("=" * 80)

    # 保存评估结果
    import json
    eval_report = {
        'eval_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_version': 'V3.91',
        'test_samples': len(X_test),
        'period_results': {
            '5d': {
                'score': results['5d']['comprehensive']['score'],
                'grade': results['5d']['comprehensive']['grade'],
                'direction_accuracy': results['5d']['direction_accuracy'],
                'ic': results['5d']['ic'],
                'r2': results['5d']['r2'],
                'top_20_return': results['5d']['top_20_mean_return'],
                'top_20_win_rate': results['5d']['top_20_positive_rate']
            },
            '10d': {
                'score': results['10d']['comprehensive']['score'],
                'grade': results['10d']['comprehensive']['grade'],
                'direction_accuracy': results['10d']['direction_accuracy'],
                'ic': results['10d']['ic'],
                'r2': results['10d']['r2'],
                'top_20_return': results['10d']['top_20_mean_return'],
                'top_20_win_rate': results['10d']['top_20_positive_rate']
            },
            '15d': {
                'score': results['15d']['comprehensive']['score'],
                'grade': results['15d']['comprehensive']['grade'],
                'direction_accuracy': results['15d']['direction_accuracy'],
                'ic': results['15d']['ic'],
                'r2': results['15d']['r2'],
                'top_20_return': results['15d']['top_20_mean_return'],
                'top_20_win_rate': results['15d']['top_20_positive_rate']
            }
        },
        'composite_results': {
            'score': results['composite']['comprehensive']['score'],
            'grade': results['composite']['comprehensive']['grade'],
            'direction_accuracy': results['composite']['direction_accuracy'],
            'ic': results['composite']['ic'],
            'r2': results['composite']['r2'],
            'top_20_return': results['composite']['top_20_mean_return'],
            'top_20_win_rate': results['composite']['top_20_positive_rate']
        },
        'weights': weights,
        'avg_period_score': avg_score
    }

    report_path = 'reports/v391_model_evaluation.json'
    Path('reports').mkdir(exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(eval_report, f, indent=2, ensure_ascii=False)

    print(f"\n📄 评估报告已保存: {report_path}")

    return results


if __name__ == "__main__":
    main()
