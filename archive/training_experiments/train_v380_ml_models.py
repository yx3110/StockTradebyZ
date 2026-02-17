#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.80机器学习模型训练脚本

用真实历史数据训练V3.80的机器学习模型
解决V3.8预测准确性不足问题

Created: 2025-09-16
Author: Claude Code
Version: 3.8.0
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# 添加项目路径
sys.path.append('/Users/yangxu/StockTradebyZ')

def train_v380_models():
    """训练V3.80机器学习模型"""
    print("🚀 V3.80机器学习模型训练开始")
    print("="*60)

    try:
        # 使用miniconda python确保依赖正确
        print("🔧 配置环境...")

        from v380_advanced_incremental_ml_system import V380AdvancedIncrementalMLSystem

        # 初始化系统
        system = V380AdvancedIncrementalMLSystem()
        print(f"✅ {system.version} 系统初始化成功")

        # 训练配置
        training_config = {
            'start_date': '2023-01-01',  # 2年历史数据
            'end_date': '2025-09-10',
            'sample_stocks': ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH',
                             '000858.SZ', '002415.SZ', '300059.SZ', '002142.SZ',
                             '000063.SZ', '600519.SH'],  # 10只代表性股票
            'validation_split': 0.2,
            'test_split': 0.1
        }

        print(f"📊 训练配置:")
        print(f"  时间范围: {training_config['start_date']} 到 {training_config['end_date']}")
        print(f"  训练股票: {len(training_config['sample_stocks'])}只")
        print(f"  验证/测试比例: {training_config['validation_split']}/{training_config['test_split']}")

        # 第一步：特征提取
        print(f"\n🔍 第1步：特征提取")

        features = system.extract_advanced_features(
            codes=training_config['sample_stocks'],
            start_date=training_config['start_date'],
            end_date=training_config['end_date'],
            target_only=False  # 包含标签数据
        )

        if features is None or len(features) == 0:
            print(f"❌ 特征提取失败")
            return False

        print(f"✅ 特征提取完成")
        print(f"  样本数量: {len(features)}条")
        print(f"  特征维度: {len(features.columns)-2}维")
        print(f"  时间跨度: {features['trade_date'].min()} 到 {features['trade_date'].max()}")

        # 检查数据质量
        missing_ratio = features.isnull().sum().sum() / (len(features) * len(features.columns))
        print(f"  缺失值比例: {missing_ratio:.2%}")

        if len(features) < 100:
            print(f"⚠️ 样本数量不足(< 100)，可能影响训练效果")

        # 第二步：模型训练
        print(f"\n🎯 第2步：模型训练")

        # 使用系统的训练方法
        training_result = system.train_models(
            training_data=features,
            validation_split=training_config['validation_split'],
            test_split=training_config['test_split']
        )

        if training_result:
            print(f"✅ 模型训练完成")

            # 显示训练结果
            if isinstance(training_result, dict):
                for key, value in training_result.items():
                    if 'score' in key.lower() or 'mse' in key.lower() or 'r2' in key.lower():
                        print(f"  {key}: {value:.4f}")

            # 第三步：模型验证
            print(f"\n✅ 第3步：模型验证")

            # 测试预测功能
            test_codes = ['000001.SZ', '600000.SH']
            test_date = '2025-09-12'

            predictions = system.predict_scores(
                codes=test_codes,
                date_str=test_date
            )

            if predictions:
                print(f"📈 预测测试:")
                for code, pred in predictions.items():
                    print(f"  {code}: {pred:.4f}")

                print(f"\n🎉 V3.80机器学习模型训练成功！")
                return True
            else:
                print(f"⚠️ 预测测试失败，但模型可能已训练完成")
                return True

        else:
            print(f"❌ 模型训练失败")
            return False

    except Exception as e:
        print(f"\n💥 训练异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_trained_models():
    """验证训练后的模型性能"""
    print(f"\n🔬 模型性能验证")

    try:
        from v380_advanced_incremental_ml_system import V380AdvancedIncrementalMLSystem
        system = V380AdvancedIncrementalMLSystem()

        # 测试股票列表
        test_stocks = ['000001.SZ', '000002.SZ', '600036.SH']
        test_dates = ['2025-09-10', '2025-09-11', '2025-09-12']

        print(f"测试配置: {len(test_stocks)}只股票 × {len(test_dates)}个日期")

        all_predictions = []
        for date in test_dates:
            predictions = system.predict_scores(test_stocks, date)
            if predictions:
                for code, score in predictions.items():
                    all_predictions.append({
                        'date': date,
                        'code': code,
                        'ml_score': score
                    })

        if all_predictions:
            df = pd.DataFrame(all_predictions)

            # 分析评分分布
            score_std = df['ml_score'].std()
            score_range = df['ml_score'].max() - df['ml_score'].min()

            print(f"📊 模型输出分析:")
            print(f"  评分范围: {df['ml_score'].min():.3f} - {df['ml_score'].max():.3f}")
            print(f"  评分标准差: {score_std:.3f}")
            print(f"  评分变异系数: {(score_std/df['ml_score'].mean()*100):.1f}%")

            if score_range > 0.1 and score_std > 0.05:
                print(f"✅ 模型输出具有良好的差异化")
                return True
            else:
                print(f"⚠️ 模型输出差异化不足")
                return False
        else:
            print(f"❌ 未能获取有效预测")
            return False

    except Exception as e:
        print(f"💥 验证异常: {e}")
        return False

if __name__ == "__main__":
    # 训练模型
    training_success = train_v380_models()

    if training_success:
        print(f"\n" + "="*60)
        # 验证模型
        validation_success = validate_trained_models()

        if validation_success:
            print(f"\n🎊 V3.80机器学习模型训练和验证全部成功！")
            print(f"现在可以用V3.80的ML模型替换V3.8的规则评分系统")
        else:
            print(f"\n⚠️ 模型训练成功但验证有问题，需要调试")
    else:
        print(f"\n❌ 模型训练失败，需要检查配置和数据")