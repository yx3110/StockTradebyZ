#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.80 根本原因调试 - 为什么所有评分都相同
"""

import sys
import numpy as np
import pandas as pd
sys.path.append('/Users/yangxu/StockTradebyZ')

from v380_advanced_incremental_ml_system import V380AdvancedIncrementalMLSystem
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_root_cause():
    """找出所有股票评分相同的根本原因"""
    print("🔍 V3.80根本原因调试 - 为什么所有评分都相同")

    try:
        # 初始化系统
        system = V380AdvancedIncrementalMLSystem()

        # 选择差异明显的测试股票
        test_codes = ['000001.SZ', '000002.SZ', '600000.SH', '000858.SZ', '300750.SZ']
        test_date = '2025-09-20'

        print(f"📊 测试股票: {test_codes}")
        print(f"📅 测试日期: {test_date}")

        # ========================================
        # 步骤1: 检查特征提取是否有差异
        # ========================================
        print(f"\n🔍 步骤1: 特征提取差异检查")

        features = system.extract_advanced_features(
            codes=test_codes,
            start_date=test_date,
            end_date=test_date,
            target_only=True
        )

        if features is None or len(features) == 0:
            print("❌ 特征提取失败")
            return

        feature_cols = [col for col in features.columns if col not in ['code', 'trade_date']]

        print(f"特征数量: {len(feature_cols)}")

        # 检查特征是否完全相同
        features_matrix = features[feature_cols].fillna(0).values
        print(f"特征矩阵形状: {features_matrix.shape}")

        # 比较第一行和其他行
        first_row = features_matrix[0]
        identical_features = True

        for i, code in enumerate(test_codes):
            current_row = features_matrix[i]

            # 检查与第一行的差异
            differences = np.abs(current_row - first_row)
            max_diff = np.max(differences)
            mean_diff = np.mean(differences)

            print(f"  {code}:")
            print(f"    与{test_codes[0]}最大差异: {max_diff:.6f}")
            print(f"    与{test_codes[0]}平均差异: {mean_diff:.6f}")
            print(f"    特征值范围: [{np.min(current_row):.6f}, {np.max(current_row):.6f}]")

            if max_diff > 1e-6:  # 如果差异大于阈值
                identical_features = False

        if identical_features:
            print("❌ 发现问题：所有股票的特征完全相同！")
            print("   这可能是特征提取逻辑的问题")
        else:
            print("✅ 特征提取正常：不同股票有不同特征")

        # ========================================
        # 步骤2: 检查模型预测是否有差异
        # ========================================
        print(f"\n🔍 步骤2: 模型预测差异检查")

        predictions_by_stock = {}

        for _, row in features.iterrows():
            code = row.get('code', 'UNKNOWN')
            full_feature_vector = row[feature_cols].fillna(0)

            print(f"\n📈 分析股票 {code}:")

            stock_predictions = []

            # 检查所有目标期间的预测
            for target_period in ['target_1d', 'target_3d', 'target_5d', 'target_10d']:
                if (target_period in system.base_models and
                    target_period in system.expert_models and
                    target_period in system.meta_learner):

                    print(f"  🎯 {target_period}:")

                    # Level 1: 基础模型预测
                    base_predictions = []
                    for model_name, model in system.base_models[target_period].items():
                        try:
                            if hasattr(model, 'feature_names_in_'):
                                model_features = model.feature_names_in_
                            else:
                                model_features = feature_cols

                            available_features = [f for f in model_features if f in full_feature_vector.index]
                            if len(available_features) > 0:
                                model_input = full_feature_vector[available_features].values.reshape(1, -1)
                                pred = model.predict(model_input)[0]
                                base_predictions.append(pred)
                                print(f"    {model_name}: {pred:.8f}")
                        except Exception as e:
                            base_predictions.append(0.0)
                            print(f"    {model_name}: 失败")

                    # Level 2: 专家模型预测
                    expert_predictions = []
                    feature_groups = system._group_features_for_experts()
                    for expert_name, expert_model in system.expert_models[target_period].items():
                        if expert_name in feature_groups:
                            try:
                                if hasattr(expert_model, 'feature_names_in_'):
                                    expected_features = expert_model.feature_names_in_
                                    expert_input_dict = {f: full_feature_vector.get(f, 0.0) for f in expected_features}
                                    expert_input = np.array([expert_input_dict[f] for f in expected_features]).reshape(1, -1)
                                else:
                                    expert_features = feature_groups[expert_name]
                                    available_expert_features = [f for f in expert_features if f in full_feature_vector.index]
                                    expert_input = full_feature_vector[available_expert_features].values.reshape(1, -1)

                                pred = expert_model.predict(expert_input)[0]
                                expert_predictions.append(pred)
                                print(f"    专家{expert_name}: {pred:.8f}")
                            except Exception as e:
                                expert_predictions.append(0.0)
                                print(f"    专家{expert_name}: 失败")

                    # Level 3: Meta学习器
                    if len(base_predictions) > 0 and len(expert_predictions) > 0:
                        meta_input_raw = np.array(base_predictions + expert_predictions)

                        # 调整meta输入
                        if hasattr(system.meta_learner[target_period], 'n_features_in_'):
                            expected_features = system.meta_learner[target_period].n_features_in_
                            if len(meta_input_raw) != expected_features:
                                if len(meta_input_raw) > expected_features:
                                    meta_input_raw = meta_input_raw[:expected_features]
                                else:
                                    avg_pred = np.mean(meta_input_raw) if len(meta_input_raw) > 0 else 0.0
                                    meta_input_raw = np.pad(meta_input_raw, (0, expected_features - len(meta_input_raw)), 'constant', constant_values=avg_pred)

                        meta_input = meta_input_raw.reshape(1, -1)
                        meta_pred = system.meta_learner[target_period].predict(meta_input)[0]
                        stock_predictions.append(meta_pred)
                        print(f"    Meta: {meta_pred:.8f}")

            if stock_predictions:
                avg_prediction = np.mean(stock_predictions)
                predictions_by_stock[code] = {
                    'raw_predictions': stock_predictions,
                    'avg_prediction': avg_prediction
                }
                print(f"  📊 平均预测: {avg_prediction:.8f}")

        # ========================================
        # 步骤3: 检查预测值是否相同
        # ========================================
        print(f"\n🔍 步骤3: 预测值差异分析")

        avg_predictions = [predictions_by_stock[code]['avg_prediction'] for code in test_codes if code in predictions_by_stock]

        if len(avg_predictions) > 1:
            prediction_std = np.std(avg_predictions)
            prediction_range = np.max(avg_predictions) - np.min(avg_predictions)

            print(f"预测值标准差: {prediction_std:.8f}")
            print(f"预测值范围: {prediction_range:.8f}")
            print(f"预测值列表: {avg_predictions}")

            if prediction_std < 1e-6:
                print("❌ 发现问题：所有股票的原始预测值完全相同！")
                print("   问题可能在模型训练或特征处理")
            else:
                print("✅ 预测值有差异")

        # ========================================
        # 步骤4: 检查标准化后的评分
        # ========================================
        print(f"\n🔍 步骤4: 标准化过程检查")

        if avg_predictions:
            print("标准化前的预测值:")
            for i, (code, pred) in enumerate(zip(test_codes, avg_predictions)):
                print(f"  {code}: {pred:.8f}")

            # 调用标准化函数
            normalized_scores = system._normalize_scores_to_100(avg_predictions)
            print("标准化后的评分:")
            for i, (code, score) in enumerate(zip(test_codes, normalized_scores)):
                print(f"  {code}: {score:.8f}")

            # 调用自适应标准化
            adaptive_scores = system.adaptive_score_normalization(
                normalized_scores, 0.25, 0.80
            )
            print("自适应调整后的评分:")
            for i, (code, score) in enumerate(zip(test_codes, adaptive_scores)):
                print(f"  {code}: {score:.8f}")

    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_root_cause()