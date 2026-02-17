#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug V3.80评分系统 - 找出得分相同的根本原因
"""

import sys
import numpy as np
sys.path.append('/Users/yangxu/StockTradebyZ')

from v380_advanced_incremental_ml_system import V380AdvancedIncrementalMLSystem
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_v380_scoring():
    """调试V3.80评分系统"""
    print("🔍 V3.80评分系统调试")

    try:
        # 初始化系统
        system = V380AdvancedIncrementalMLSystem()

        # 测试股票
        test_codes = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '002415.SZ']
        test_date = '2025-09-20'

        print(f"📊 测试股票: {test_codes}")
        print(f"📅 测试日期: {test_date}")

        # 检查模型是否已训练
        print(f"🤖 模型状态:")
        print(f"  - base_models: {hasattr(system, 'base_models') and bool(system.base_models)}")
        print(f"  - expert_models: {hasattr(system, 'expert_models') and bool(system.expert_models)}")
        print(f"  - meta_learner: {hasattr(system, 'meta_learner') and bool(system.meta_learner)}")

        if not (hasattr(system, 'base_models') and system.base_models):
            print("❌ 模型未训练，无法继续调试")
            return

        # 1. 检查特征提取
        print("\n🔍 步骤1: 检查特征提取...")
        features = system.extract_advanced_features(
            codes=test_codes,
            start_date=test_date,
            end_date=test_date,
            target_only=True
        )

        if features is None or len(features) == 0:
            print("❌ 特征提取失败")
            return

        print(f"✅ 特征提取成功: {len(features)} 条记录, {len(features.columns)-2} 个特征")

        # 显示各股票的前几个特征值
        feature_cols = [col for col in features.columns if col not in ['code', 'trade_date']][:5]
        print(f"📊 前5个特征值对比: {feature_cols}")
        for _, row in features.iterrows():
            code = row['code']
            values = [f"{row[col]:.4f}" for col in feature_cols]
            print(f"  {code}: {values}")

        # 2. 逐步调试预测过程
        print("\n🔍 步骤2: 逐步调试预测过程...")

        # 模拟predict_scores函数的核心逻辑
        predictions = {}
        feature_cols_all = [col for col in features.columns if col not in ['code', 'trade_date']]

        for _, row in features.iterrows():
            code = row.get('code', 'UNKNOWN')
            full_feature_vector = row[feature_cols_all].fillna(0)

            print(f"\n📈 处理股票 {code}:")

            # 使用所有可用的目标期间模型进行ensemble预测
            ensemble_predictions = []

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
                                model_features = feature_cols_all

                            available_features = [f for f in model_features if f in full_feature_vector.index]
                            if len(available_features) > 0:
                                model_input = full_feature_vector[available_features].values.reshape(1, -1)
                                pred = model.predict(model_input)[0]
                                base_predictions.append(pred)
                                print(f"    Base {model_name}: {pred:.6f}")

                        except Exception as e:
                            print(f"    Base {model_name}: 失败 {e}")
                            base_predictions.append(0.0)

                    print(f"    Base predictions: {base_predictions}")

                    # Level 2: 专家模型预测 (简化检查)
                    expert_predictions = []
                    for expert_name, expert_model in system.expert_models[target_period].items():
                        try:
                            # 使用全部特征作为fallback
                            expert_input = full_feature_vector.values.reshape(1, -1)
                            pred = expert_model.predict(expert_input)[0]
                            expert_predictions.append(pred)
                            print(f"    Expert {expert_name}: {pred:.6f}")
                        except Exception as e:
                            print(f"    Expert {expert_name}: 失败 {e}")
                            expert_predictions.append(0.0)

                    print(f"    Expert predictions: {expert_predictions}")

                    # Level 3: Meta学习器
                    if len(base_predictions) > 0 and len(expert_predictions) > 0:
                        meta_input = np.array(base_predictions + expert_predictions).reshape(1, -1)
                        meta_pred = system.meta_learner[target_period].predict(meta_input)[0]
                        ensemble_predictions.append(meta_pred)
                        print(f"    Meta prediction: {meta_pred:.6f}")

            if ensemble_predictions:
                avg_prediction = np.mean(ensemble_predictions)
                print(f"  🎯 Ensemble平均: {avg_prediction:.6f}")

                # 标准化到0-100评分
                score = system._normalize_scores_to_100([avg_prediction])[0]
                print(f"  📊 标准化后评分: {score:.2f}")

                # 应用自适应评分调整
                adjusted_score = system.adaptive_score_normalization(
                    [score], 0.25, 0.80
                )[0]
                print(f"  🔧 自适应调整后: {adjusted_score:.2f}")

                predictions[code] = round(adjusted_score, 2)
            else:
                print(f"  ❌ 无有效预测")
                predictions[code] = 50.0

        # 3. 显示最终结果
        print(f"\n🎯 最终预测结果:")
        for code, score in predictions.items():
            print(f"  {code}: {score}")

        # 检查是否所有得分相同
        unique_scores = set(predictions.values())
        if len(unique_scores) == 1:
            print(f"❌ 所有股票得分相同: {list(unique_scores)[0]}")
        else:
            print(f"✅ 得分有差异: {sorted(unique_scores)}")

    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_v380_scoring()