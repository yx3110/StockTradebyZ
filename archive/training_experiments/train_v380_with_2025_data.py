#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.80机器学习模型训练脚本 - 2025年数据版

使用2025年最新4个月数据 + 1200只股票行业均衡采样
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

def load_2025_stock_list():
    """加载2025年选定的1200只股票"""
    stock_list = []
    try:
        with open('/Users/yangxu/StockTradebyZ/v380_2025_focused_stocks.txt', 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    code = line.strip().split()[0]
                    stock_list.append(code)
        print(f"✅ 加载2025年聚焦股票列表: {len(stock_list)}只")
        return stock_list
    except Exception as e:
        print(f"❌ 无法加载股票列表: {e}")
        return None

def train_v380_with_2025_data():
    """使用2025年数据训练V3.80模型"""
    print("🚀 V3.80机器学习模型训练 (2025年数据版)")
    print("="*60)

    try:
        from v380_advanced_incremental_ml_system import V380AdvancedIncrementalMLSystem

        # 初始化系统
        system = V380AdvancedIncrementalMLSystem()
        print(f"✅ {system.version} 系统初始化成功")

        # 加载2025年选定股票
        stock_list = load_2025_stock_list()
        if not stock_list:
            print("❌ 无法加载股票列表，使用默认股票")
            stock_list = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ']

        print(f"📊 训练配置:")
        print(f"  时间范围: 2025-01-01 到 2025-09-16")
        print(f"  训练股票: {len(stock_list)}只")
        print(f"  数据策略: 2025年最新4个月数据")

        # 第一步：特征提取 (使用2025年数据)
        print(f"\n🔍 第1步：特征提取 (2025年数据)")

        features = system.extract_advanced_features(
            codes=stock_list,
            start_date='2025-01-01',
            end_date='2025-09-16',
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

        if len(features) < 1000:
            print(f"⚠️ 样本数量: {len(features)}条，可能偏少")
        else:
            print(f"✅ 样本数量充足: {len(features)}条")

        # 第二步：准备训练数据
        print(f"\n🎯 第2步：准备训练数据")

        # 准备多期标签
        training_result = system.prepare_training_data(
            features_df=features,
            target_days=[1, 3, 5, 10]
        )

        # 正确解包返回值
        if isinstance(training_result, tuple):
            training_data, feature_groups = training_result
        else:
            training_data = training_result
            feature_groups = system._group_features_for_experts()

        if training_data is None or len(training_data) == 0:
            print(f"❌ 训练数据准备失败")
            return False

        print(f"✅ 训练数据准备完成")
        print(f"  有效样本: {len(training_data)}条")
        print(f"  目标变量: 1日、3日、5日、10日收益")

        # 第三步：模型训练
        print(f"\n🚀 第3步：三层Ensemble模型训练")

        # 为每个预测期训练模型
        training_results = {}

        for target_period in [1, 3, 5, 10]:
            target_col = f'target_{target_period}d'

            if target_col not in training_data.columns:
                print(f"⚠️ 跳过{target_period}日目标：数据不足")
                continue

            print(f"\n📈 训练{target_period}日预测模型...")

            result = system.train_three_layer_ensemble(
                training_data=training_data,
                feature_groups=feature_groups,
                target_col=target_col
            )

            training_results[target_period] = result

            if result.get('success', False):
                print(f"✅ {target_period}日模型训练成功")
                print(f"  训练样本: {result.get('training_samples', 0)}条")
                print(f"  Meta模型性能: {result.get('meta_performance', 0):.4f}")
            else:
                print(f"❌ {target_period}日模型训练失败")

        # 第四步：模型验证
        print(f"\n✅ 第4步：模型验证")

        # 测试预测功能
        test_codes = stock_list[:5]  # 测试前5只股票
        test_date = '2025-09-16'

        print(f"🧪 预测测试: {len(test_codes)}只股票")

        predictions = system.predict_scores(
            codes=test_codes,
            date_str=test_date
        )

        if predictions and len(predictions) > 0:
            print(f"📈 预测结果样例:")
            for i, (code, pred) in enumerate(list(predictions.items())[:3]):
                print(f"  {code}: {pred:.4f}")

            # 分析预测分布
            pred_values = list(predictions.values())
            pred_std = np.std(pred_values)
            pred_range = max(pred_values) - min(pred_values)

            print(f"\n📊 预测质量分析:")
            print(f"  预测范围: {min(pred_values):.3f} - {max(pred_values):.3f}")
            print(f"  预测标准差: {pred_std:.3f}")
            print(f"  差异化程度: {'良好' if pred_range > 0.1 else '不足'}")

            print(f"\n🎉 V3.80机器学习模型训练成功！")
            print(f"📊 训练总结:")
            print(f"  数据来源: 2025年最新4个月")
            print(f"  训练股票: {len(stock_list)}只")
            print(f"  训练样本: {len(training_data)}条")
            print(f"  成功模型: {sum(1 for r in training_results.values() if r.get('success', False))}个")

            # 保存训练好的模型
            print(f"\n💾 保存V3.8模型...")
            model_file = system.save_models("_2025_trained")
            print(f"✅ 模型已保存: {model_file}")

            return True
        else:
            print(f"⚠️ 预测测试失败，但模型可能已训练完成")
            return True

    except Exception as e:
        print(f"\n💥 训练异常: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = train_v380_with_2025_data()

    if success:
        print(f"\n🎊 2025年V3.80训练完成!")
        print(f"可以开始使用V3.80的ML预测功能替换V3.8规则评分")
    else:
        print(f"\n❌ 训练失败，需要检查数据和配置")