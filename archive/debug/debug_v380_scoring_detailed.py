#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度调试V3.80评分系统 - 追踪每一步的评分计算
"""

import sys
import numpy as np
sys.path.append('/Users/yangxu/StockTradebyZ')

from v380_advanced_incremental_ml_system import V380AdvancedIncrementalMLSystem
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_detailed_scoring():
    """深度调试V3.80评分系统的每一步"""
    print("🔍 深度调试V3.80评分系统")

    try:
        # 初始化系统
        system = V380AdvancedIncrementalMLSystem()

        # 测试股票
        test_codes = ['000001.SZ', '000002.SZ', '600000.SH']
        test_date = '2025-09-20'

        print(f"📊 测试股票: {test_codes}")
        print(f"📅 测试日期: {test_date}")

        # 1. 检查特征提取
        features = system.extract_advanced_features(
            codes=test_codes,
            start_date=test_date,
            end_date=test_date,
            target_only=True
        )

        if features is None or len(features) == 0:
            print("❌ 特征提取失败")
            return

        print(f"\n🔍 步骤1: 特征提取结果")
        print(f"特征数量: {len(features.columns)-2}")
        feature_cols = [col for col in features.columns if col not in ['code', 'trade_date']]

        # 显示每个股票的特征统计
        for _, row in features.iterrows():
            code = row['code']
            feature_values = row[feature_cols].fillna(0)
            print(f"  {code}:")
            print(f"    特征均值: {np.mean(feature_values):.6f}")
            print(f"    特征标准差: {np.std(feature_values):.6f}")
            print(f"    特征范围: [{np.min(feature_values):.6f}, {np.max(feature_values):.6f}]")

        # 2. 手动模拟预测过程，追踪每一步
        print(f"\n🔍 步骤2: 手动模拟预测过程")

        for _, row in features.iterrows():
            code = row.get('code', 'UNKNOWN')
            full_feature_vector = row[feature_cols].fillna(0)

            print(f"\n📈 处理股票 {code}:")

            # 收集所有期间的ensemble预测
            all_ensemble_predictions = []

            for target_period in ['target_1d', 'target_3d', 'target_5d', 'target_10d']:
                if (target_period in system.base_models and
                    target_period in system.expert_models and
                    target_period in system.meta_learner):

                    print(f"  🎯 目标期间: {target_period}")

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
                                print(f"    Base {model_name}: {pred:.8f}")
                        except Exception as e:
                            print(f"    Base {model_name}: 失败 - {e}")
                            base_predictions.append(0.0)

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
                                print(f"    Expert {expert_name}: {pred:.8f}")
                            except Exception as e:
                                print(f"    Expert {expert_name}: 失败 - {e}")
                                expert_predictions.append(0.0)

                    # Level 3: Meta学习器
                    if len(base_predictions) > 0 and len(expert_predictions) > 0:
                        meta_input_raw = np.array(base_predictions + expert_predictions)
                        print(f"    Meta输入: {meta_input_raw}")

                        # 检查Meta学习器期望的特征数量
                        if hasattr(system.meta_learner[target_period], 'n_features_in_'):
                            expected_features = system.meta_learner[target_period].n_features_in_
                            if len(meta_input_raw) != expected_features:
                                print(f"    调整Meta输入: {len(meta_input_raw)} -> {expected_features}")
                                if len(meta_input_raw) > expected_features:
                                    meta_input_raw = meta_input_raw[:expected_features]
                                else:
                                    avg_pred = np.mean(meta_input_raw) if len(meta_input_raw) > 0 else 0.0
                                    meta_input_raw = np.pad(meta_input_raw, (0, expected_features - len(meta_input_raw)), 'constant', constant_values=avg_pred)

                        meta_input = meta_input_raw.reshape(1, -1)
                        meta_pred = system.meta_learner[target_period].predict(meta_input)[0]
                        all_ensemble_predictions.append(meta_pred)
                        print(f"    Meta预测: {meta_pred:.8f}")

            if all_ensemble_predictions:
                print(f"  📊 所有期间预测: {all_ensemble_predictions}")
                avg_prediction = np.mean(all_ensemble_predictions)
                print(f"  📊 平均预测: {avg_prediction:.8f}")

                # 步骤3: 标准化到0-100评分
                print(f"\n🔍 步骤3: 标准化评分过程")
                score_before_normalize = avg_prediction
                print(f"  标准化前: {score_before_normalize:.8f}")

                # 调用normalize函数
                score_after_normalize = system._normalize_scores_to_100([avg_prediction])[0]
                print(f"  标准化后: {score_after_normalize:.8f}")

                # 步骤4: 自适应评分调整
                print(f"\n🔍 步骤4: 自适应评分调整")
                market_volatility = 0.25
                confidence_level = 0.80
                print(f"  市场波动率: {market_volatility}")
                print(f"  置信水平: {confidence_level}")

                adjusted_score = system.adaptive_score_normalization(
                    [score_after_normalize], market_volatility, confidence_level
                )[0]
                print(f"  自适应调整后: {adjusted_score:.8f}")

                final_score = round(adjusted_score, 2)
                print(f"  最终评分: {final_score}")

                # 步骤5: 检查各项因子评分
                print(f"\n🔍 步骤5: 检查各项因子评分是否也相同")
                # 这里需要调用evaluate_stock_with_v38函数来获取因子评分
                evaluation_result = system.evaluate_stock_with_v38([code], test_date)
                if evaluation_result.get('success', False):
                    stock_data = evaluation_result['stocks'][0] if evaluation_result.get('stocks') else {}
                    factor_scores = stock_data.get('factor_scores', {})
                    print(f"  因子评分: {factor_scores}")

    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_detailed_scoring()