#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 Level 4 训练数据准备器
将质量标签数据与25维特征组合，准备Level 4 Quality Meta-learner的训练数据

数据流程:
1. 读取质量标签数据集 (quality_training_dataset.csv)
2. 模拟V380预测结果格式 (由于历史数据没有中间预测)
3. 提取25维质量特征
4. 组合特征和标签形成训练数据集
5. 数据清洗和验证
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from level4_quality_feature_extractor import Level4QualityFeatureExtractor
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Level4TrainingDataPreparer:
    """Level 4训练数据准备器"""

    def __init__(self):
        self.feature_extractor = Level4QualityFeatureExtractor()

    def simulate_v380_prediction_format(self, row: pd.Series) -> Dict[str, Any]:
        """
        根据质量标签数据模拟V380预测结果格式

        由于历史数据没有真实的Level 1-2中间预测，我们基于已有信息进行合理模拟
        """
        try:
            final_score = row.get('final_score', 50.0)
            confidence_score = row.get('confidence_score', 0.5)
            short_term = row.get('short_term_score', 50.0)
            medium_term = row.get('medium_term_score', 50.0)
            long_term = row.get('long_term_score', 50.0)

            # 基于最终评分模拟Level 1基础模型预测
            # 添加合理的噪声来模拟不同模型的差异
            base_pred = (final_score - 50) / 10.0  # 转换为[-5, 5]范围
            level1_predictions = {
                'lgb_target_1d': base_pred + np.random.normal(0, 0.5),
                'xgb_target_1d': base_pred + np.random.normal(0, 0.3),
                'catboost_target_1d': base_pred + np.random.normal(0, 0.4),
                'rf_target_1d': base_pred + np.random.normal(0, 0.6),
                'nn_target_1d': base_pred + np.random.normal(0, 0.4),
                'lgb_target_3d': base_pred * 0.8 + np.random.normal(0, 0.3),
                'xgb_target_3d': base_pred * 0.8 + np.random.normal(0, 0.2),
                'catboost_target_3d': base_pred * 0.8 + np.random.normal(0, 0.3),
                'rf_target_3d': base_pred * 0.8 + np.random.normal(0, 0.4),
            }

            # 基于短中长期评分模拟Level 2专家预测
            tech_pred = (short_term - 50) / 10.0
            fund_pred = (medium_term - 50) / 10.0
            macro_pred = (long_term - 50) / 10.0
            sentiment_pred = base_pred + np.random.normal(0, 0.3)

            level2_predictions = {
                'technical_expert_target_1d': tech_pred,
                'fundamental_expert_target_1d': fund_pred,
                'macro_expert_target_1d': macro_pred,
                'sentiment_expert_target_1d': sentiment_pred,
                'technical_expert_target_3d': tech_pred * 0.9,
                'fundamental_expert_target_3d': fund_pred * 0.9,
                'macro_expert_target_3d': macro_pred * 0.9,
                'sentiment_expert_target_3d': sentiment_pred * 0.9,
            }

            # 构造raw_predictions
            raw_predictions = {
                'target_1d': base_pred,
                'target_3d': base_pred * 0.8,
                'target_5d': base_pred * 0.7,
                'target_10d': base_pred * 0.6
            }

            # 构造V380格式的预测结果
            prediction_data = {
                'overall_score': final_score,
                'short_term_score': short_term,
                'medium_term_score': medium_term,
                'long_term_score': long_term,
                'confidence_score': confidence_score,
                'level1_predictions': level1_predictions,
                'level2_predictions': level2_predictions,
                'raw_predictions': raw_predictions
            }

            return prediction_data

        except Exception as e:
            logger.error(f"模拟V380格式失败: {e}")
            return {}

    def prepare_training_dataset(self, quality_data_path: str = "quality_training_dataset.csv") -> pd.DataFrame:
        """准备完整的Level 4训练数据集"""
        try:
            logger.info("🚀 开始准备Level 4训练数据集")

            # 1. 读取质量标签数据
            if not Path(quality_data_path).exists():
                logger.error(f"质量数据文件不存在: {quality_data_path}")
                return pd.DataFrame()

            quality_df = pd.read_csv(quality_data_path)
            logger.info(f"📊 质量标签数据: {len(quality_df)} 条记录")

            # 2. 过滤有效数据（有质量标签的记录）
            valid_df = quality_df.dropna(subset=['quality_overall'])
            logger.info(f"✅ 有效记录数: {len(valid_df)}")

            if len(valid_df) == 0:
                logger.error("没有有效的质量标签数据")
                return pd.DataFrame()

            # 3. 提取特征和准备训练数据
            training_data = []
            feature_names = self.feature_extractor.get_feature_names()

            logger.info("🔧 提取25维质量特征...")
            for idx, row in valid_df.iterrows():
                if idx % 100 == 0:
                    logger.info(f"处理进度: {idx}/{len(valid_df)}")

                try:
                    # 模拟V380预测格式
                    prediction_data = self.simulate_v380_prediction_format(row)
                    if not prediction_data:
                        continue

                    # 提取25维特征
                    features = self.feature_extractor.extract_quality_features(prediction_data)

                    # 准备训练样本
                    sample = {
                        # 基础信息
                        'prediction_date': row.get('prediction_date'),
                        'stock_code': row.get('stock_code'),
                        'stock_name': row.get('stock_name', ''),

                        # 原始V380输出
                        'final_score': row.get('final_score', 50.0),
                        'confidence_score': row.get('confidence_score', 0.5),
                        'short_term_score': row.get('short_term_score', 50.0),
                        'medium_term_score': row.get('medium_term_score', 50.0),
                        'long_term_score': row.get('long_term_score', 50.0),

                        # 实际收益率
                        'return_1d': row.get('return_1d'),
                        'return_3d': row.get('return_3d'),
                        'return_5d': row.get('return_5d'),

                        # 质量标签 (目标变量)
                        'quality_1d': row.get('quality_1d', 0.5),
                        'quality_3d': row.get('quality_3d', 0.5),
                        'quality_5d': row.get('quality_5d', 0.5),
                        'quality_overall': row.get('quality_overall', 0.5),
                    }

                    # 添加25维质量特征
                    for i, feature_name in enumerate(feature_names):
                        sample[f'feature_{feature_name}'] = features[i]

                    training_data.append(sample)

                except Exception as e:
                    logger.debug(f"处理样本失败 {idx}: {e}")
                    continue

            # 4. 转换为DataFrame
            if not training_data:
                logger.error("没有成功提取任何训练样本")
                return pd.DataFrame()

            training_df = pd.DataFrame(training_data)
            logger.info(f"✅ 训练数据集构建完成: {len(training_df)} 样本")

            # 5. 数据质量检查
            self._validate_training_data(training_df, feature_names)

            return training_df

        except Exception as e:
            logger.error(f"训练数据准备失败: {e}")
            return pd.DataFrame()

    def _validate_training_data(self, df: pd.DataFrame, feature_names: List[str]):
        """验证训练数据质量"""
        try:
            logger.info("🔍 训练数据质量检查:")

            # 基础统计
            print(f"样本总数: {len(df)}")
            print(f"特征维度: {len(feature_names)}")
            print(f"股票数量: {df['stock_code'].nunique()}")
            print(f"日期范围: {df['prediction_date'].min()} - {df['prediction_date'].max()}")

            # 目标变量分布
            for target in ['quality_1d', 'quality_3d', 'quality_5d', 'quality_overall']:
                if target in df.columns:
                    values = df[target].dropna()
                    print(f"{target}: 均值={values.mean():.3f}, 标准差={values.std():.3f}, 范围=[{values.min():.3f}, {values.max():.3f}]")

            # 特征统计
            feature_cols = [f'feature_{name}' for name in feature_names]
            feature_data = df[feature_cols]

            print(f"\n特征统计:")
            print(f"特征范围: [{feature_data.min().min():.3f}, {feature_data.max().max():.3f}]")
            print(f"缺失值数量: {feature_data.isnull().sum().sum()}")

            # 检查数据分布
            quality_overall = df['quality_overall'].dropna()
            if len(quality_overall) > 0:
                # 质量评分分布
                print(f"\n质量评分分布:")
                print(f"低质量 (<0.4): {(quality_overall < 0.4).sum()} ({(quality_overall < 0.4).mean()*100:.1f}%)")
                print(f"中质量 (0.4-0.6): {((quality_overall >= 0.4) & (quality_overall <= 0.6)).sum()} ({((quality_overall >= 0.4) & (quality_overall <= 0.6)).mean()*100:.1f}%)")
                print(f"高质量 (>0.6): {(quality_overall > 0.6).sum()} ({(quality_overall > 0.6).mean()*100:.1f}%)")

        except Exception as e:
            logger.error(f"数据验证失败: {e}")

    def save_training_dataset(self, df: pd.DataFrame, output_path: str = "level4_training_dataset.csv"):
        """保存训练数据集"""
        try:
            if df.empty:
                logger.error("数据集为空，无法保存")
                return False

            df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"✅ 训练数据集已保存: {output_path}")

            # 保存特征名称映射
            feature_names = self.feature_extractor.get_feature_names()
            feature_mapping = {
                'feature_names': feature_names,
                'feature_groups': self.feature_extractor.get_feature_importance_groups(),
                'dataset_info': {
                    'total_samples': len(df),
                    'feature_count': len(feature_names),
                    'target_variables': ['quality_1d', 'quality_3d', 'quality_5d', 'quality_overall'],
                    'created_at': pd.Timestamp.now().isoformat()
                }
            }

            mapping_path = output_path.replace('.csv', '_feature_mapping.json')
            with open(mapping_path, 'w', encoding='utf-8') as f:
                json.dump(feature_mapping, f, indent=2, ensure_ascii=False)

            logger.info(f"✅ 特征映射已保存: {mapping_path}")
            return True

        except Exception as e:
            logger.error(f"保存数据集失败: {e}")
            return False

    def split_dataset(self, df: pd.DataFrame, test_size: float = 0.2,
                     validation_size: float = 0.1) -> tuple:
        """按时间顺序分割数据集"""
        try:
            # 按日期排序
            df_sorted = df.sort_values('prediction_date').reset_index(drop=True)

            total_size = len(df_sorted)
            train_size = int(total_size * (1 - test_size - validation_size))
            val_size = int(total_size * validation_size)

            train_df = df_sorted[:train_size]
            val_df = df_sorted[train_size:train_size + val_size]
            test_df = df_sorted[train_size + val_size:]

            logger.info(f"📊 数据集分割:")
            logger.info(f"训练集: {len(train_df)} 样本 ({len(train_df)/total_size*100:.1f}%)")
            logger.info(f"验证集: {len(val_df)} 样本 ({len(val_df)/total_size*100:.1f}%)")
            logger.info(f"测试集: {len(test_df)} 样本 ({len(test_df)/total_size*100:.1f}%)")

            return train_df, val_df, test_df

        except Exception as e:
            logger.error(f"数据集分割失败: {e}")
            return df, pd.DataFrame(), pd.DataFrame()

# 使用示例
if __name__ == "__main__":
    preparer = Level4TrainingDataPreparer()

    # 准备训练数据集
    training_df = preparer.prepare_training_dataset()

    if not training_df.empty:
        # 保存完整数据集
        preparer.save_training_dataset(training_df)

        # 分割数据集
        train_df, val_df, test_df = preparer.split_dataset(training_df)

        # 保存分割后的数据集
        if not train_df.empty:
            preparer.save_training_dataset(train_df, "level4_training_dataset_train.csv")
        if not val_df.empty:
            preparer.save_training_dataset(val_df, "level4_training_dataset_val.csv")
        if not test_df.empty:
            preparer.save_training_dataset(test_df, "level4_training_dataset_test.csv")

        print(f"\n🎯 Level 4训练数据准备完成!")
        print(f"总样本数: {len(training_df)}")
        print(f"特征维度: 25")
        print(f"目标变量: quality_1d, quality_3d, quality_5d, quality_overall")
    else:
        print("❌ 训练数据准备失败")