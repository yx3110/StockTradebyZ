#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.80 + Level 4 Quality Meta-learner 集成系统
在V380基础上集成Level 4质量评分，解决质量评分聚集问题

🎯 核心改进:
- 保持V380三层架构完整性
- 新增Level 4 Quality Meta-learner
- 利用Level 1-3中间预测结果计算质量评分
- 实现端到端的股票质量差异化评估

作者: Claude Code
版本: V3.80 + Level 4 Integration
创建时间: 2025-09-23
"""

import sys
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# 导入V380原系统
sys.path.append('/Users/yangxu/StockTradebyZ')
from v380_advanced_incremental_ml_system import V380AdvancedIncrementalMLSystem

# 导入Level 4组件
from level4_quality_meta_learner import Level4QualityMetaLearner
from level4_quality_feature_extractor_v2 import Level4QualityFeatureExtractorV2

class V380Level4IntegratedSystem(V380AdvancedIncrementalMLSystem):
    """V380 + Level 4 Quality Meta-learner 集成系统"""

    def __init__(self):
        # 初始化V380基类
        super().__init__()

        # 初始化Level 4组件
        self.level4_learner = None
        self.level4_postprocessor = None
        self.level4_feature_extractor = None
        self.level4_best_method = None

        # 加载Level 4模型
        self._load_level4_models()

        self.logger.info("🎯 V380 + Level 4集成系统初始化完成")

    def _load_level4_models(self):
        """加载Level 4质量评分模型"""
        try:
            # 1. 加载Level 4 Meta-learner (使用重新训练的标准化模型)
            self.level4_learner = Level4QualityMetaLearner()
            normalized_model_path = "models/level4_quality_meta_learner_normalized.pkl"
            if Path(normalized_model_path).exists():
                self.level4_learner.load_model(normalized_model_path)
                self.logger.info("✅ Level 4 Meta-learner加载成功 (标准化版本)")
            else:
                # 回退到原模型
                self.level4_learner.load_model("models/level4_quality_meta_learner.pkl")
                self.logger.info("✅ Level 4 Meta-learner加载成功 (原版本)")

            # 2. 加载后处理器
            with open("models/level4_quality_postprocessor.pkl", 'rb') as f:
                postprocessor_data = pickle.load(f)
            self.level4_postprocessor = postprocessor_data['postprocessor']
            self.level4_best_method = postprocessor_data['best_method']
            self.logger.info(f"✅ Level 4后处理器加载成功 (方法: {self.level4_best_method})")

            # 3. 初始化特征提取器
            self.level4_feature_extractor = Level4QualityFeatureExtractorV2()
            self.logger.info("✅ Level 4特征提取器初始化完成")

        except Exception as e:
            self.logger.error(f"❌ Level 4模型加载失败: {e}")
            self.level4_learner = None
            self.level4_postprocessor = None

    def _extract_level4_features(self, prediction_data):
        """
        从V380预测结果中提取Level 4特征

        Args:
            prediction_data: V380预测结果字典，包含level1_predictions和level2_predictions

        Returns:
            25维Level 4特征向量
        """
        try:
            # 获取Level 1和Level 2预测结果
            level1_preds = prediction_data.get('level1_predictions', {})
            level2_preds = prediction_data.get('level2_predictions', {})
            level3_pred = prediction_data.get('overall_score', 50.0)

            if not level1_preds and not level2_preds:
                self.logger.warning("⚠️ 缺少Level 1/2预测结果，使用默认特征")
                return np.full(25, 0.5)  # 默认中等特征值

            # 构建完整的预测数据结构（包含Level 1/2预测）
            complete_prediction_data = {
                'final_score': level3_pred,
                'confidence_score': prediction_data.get('confidence_score', 0.5),
                'short_term_score': prediction_data.get('short_term_score', level3_pred),
                'medium_term_score': prediction_data.get('medium_term_score', level3_pred),
                'long_term_score': prediction_data.get('long_term_score', level3_pred),
                'level1_predictions': level1_preds,  # 直接使用原始Level 1预测
                'level2_predictions': level2_preds   # 直接使用原始Level 2预测
            }

            # 使用特征提取器计算25维特征
            features = self.level4_feature_extractor.extract_quality_features(
                prediction_data=complete_prediction_data
            )

            return features

        except Exception as e:
            self.logger.error(f"❌ Level 4特征提取失败: {e}")
            return np.full(25, 0.5)  # 返回默认中等特征值

    def predict_scores_with_quality(self, codes, date_str):
        """
        扩展V380预测，新增Level 4质量评分

        Returns:
            dict: 包含V380原有评分 + quality_score的完整预测结果
        """
        # 1. 调用V380原有预测方法 (使用父类方法避免递归)
        v380_predictions = super().predict_scores(codes, date_str)

        if not self.level4_learner or not self.level4_postprocessor:
            self.logger.warning("⚠️ Level 4模型未加载，返回V380原预测结果")
            # 添加默认质量评分
            for code in v380_predictions:
                if isinstance(v380_predictions[code], dict):
                    v380_predictions[code]['quality_score'] = 50.0
                else:
                    # 如果是简单评分，转换为dict格式
                    v380_predictions[code] = {
                        'overall_score': v380_predictions[code],
                        'quality_score': 50.0
                    }
            return v380_predictions

        # 2. 为每只股票计算Level 4质量评分
        enhanced_predictions = {}
        all_raw_quality_scores = []

        for code, prediction_data in v380_predictions.items():
            if isinstance(prediction_data, dict):
                # 提取Level 4特征
                level4_features = self._extract_level4_features(prediction_data)

                # Level 4质量评分预测
                raw_quality_score = self.level4_learner.predict_quality_score(
                    level4_features.reshape(1, -1)
                )[0]
                all_raw_quality_scores.append(raw_quality_score)

                # 保存增强的预测结果
                enhanced_predictions[code] = prediction_data.copy()
                enhanced_predictions[code]['raw_quality_score'] = raw_quality_score
            else:
                # 处理简单评分格式
                enhanced_predictions[code] = {
                    'overall_score': prediction_data,
                    'raw_quality_score': 0.4  # 默认中等原始质量评分
                }
                all_raw_quality_scores.append(0.4)

        # 3. 批量应用Level 4后处理
        if all_raw_quality_scores:
            processed_quality_scores = self.level4_postprocessor.transform(
                np.array(all_raw_quality_scores),
                method=self.level4_best_method
            )

            # 4. 更新最终质量评分并生成投资建议
            for i, code in enumerate(enhanced_predictions.keys()):
                enhanced_predictions[code]['quality_score'] = round(processed_quality_scores[i], 2)

                # 🎯 V3.81专用投资建议生成逻辑
                final_score = enhanced_predictions[code].get('overall_score', 50.0)
                quality_score = enhanced_predictions[code]['quality_score']
                confidence_score = enhanced_predictions[code].get('confidence_score', 0.5)

                # V3.81投资建议阈值（相比V380更宽松）
                if final_score >= 85:
                    recommendation = "强烈买入"
                elif final_score >= 80 or (final_score >= 75 and quality_score >= 0.7):
                    recommendation = "买入"
                elif final_score >= 70 or (final_score >= 65 and quality_score >= 0.6):
                    recommendation = "轻仓买入"
                elif final_score <= 40 or (final_score <= 50 and quality_score <= 0.3):
                    recommendation = "卖出"
                elif final_score <= 50 or (final_score <= 60 and quality_score <= 0.4):
                    recommendation = "减仓"
                else:
                    recommendation = "观望"

                enhanced_predictions[code]['recommendation'] = recommendation

        self.logger.info(f"✅ Level 4质量评分和投资建议生成完成: {len(enhanced_predictions)}只股票")

        # 输出质量评分统计
        quality_scores = [pred.get('quality_score', 50) for pred in enhanced_predictions.values()]
        if quality_scores:
            self.logger.info(f"📊 质量评分分布: 均值={np.mean(quality_scores):.2f}, "
                           f"std={np.std(quality_scores):.2f}, "
                           f"范围=[{np.min(quality_scores):.2f}, {np.max(quality_scores):.2f}]")

        return enhanced_predictions

    def predict_scores(self, codes, date_str):
        """
        重写预测方法，默认包含Level 4质量评分
        保持向后兼容性
        """
        return self.predict_scores_with_quality(codes, date_str)


def main():
    """测试V380 + Level 4集成系统"""
    print("🚀 V380 + Level 4集成系统测试")

    # 初始化集成系统
    system = V380Level4IntegratedSystem()
    print(f"✅ {system.version} + Level 4 系统初始化成功")

    # 测试预测
    test_codes = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '002215.SZ']
    test_date = '2025-09-23'

    print(f"\n🧪 测试预测: {len(test_codes)}只股票，日期: {test_date}")

    # 预测评分
    predictions = system.predict_scores_with_quality(test_codes, test_date)

    # 显示结果
    print("\n📊 预测结果:")
    for code, result in predictions.items():
        if isinstance(result, dict):
            overall = result.get('overall_score', 'N/A')
            quality = result.get('quality_score', 'N/A')
            confidence = result.get('confidence_score', 'N/A')
            print(f"  {code}: 综合评分={overall}, 质量评分={quality}, 置信度={confidence}")
        else:
            print(f"  {code}: 评分={result}")

    # 质量评分差异化验证
    quality_scores = [pred.get('quality_score', 50) for pred in predictions.values() if isinstance(pred, dict)]
    if quality_scores:
        std_quality = np.std(quality_scores)
        print(f"\n🎯 质量评分差异化验证:")
        print(f"  标准差: {std_quality:.4f}")
        print(f"  差异化状态: {'✅ 达标' if std_quality >= 0.15 else '❌ 不达标'} (目标>=0.15)")
        print(f"  改进效果: {'成功解决质量评分聚集问题' if std_quality >= 0.15 else '需要进一步优化'}")


if __name__ == "__main__":
    main()