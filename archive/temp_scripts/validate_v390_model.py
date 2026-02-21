#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.9模型训练验证脚本

验证项目：
1. 训练数据完整性（样本数、特征完整度、标签分布）
2. 模型文件存在性
3. 模型预测功能
4. 评分合理性（范围、分布）
5. 特征重要性分析
6. 回测性能指标

作者: Claude Code
创建时间: 2025-11-03
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).parent))

def validate_training_data(model_dir='ml_models/trained_models/v39'):
    """验证训练数据完整性"""
    print("\n" + "="*80)
    print("📊 验证1: 训练数据完整性")
    print("="*80)

    try:
        # 检查训练数据文件
        train_data_path = f"{model_dir}/training_data.pkl"
        if not os.path.exists(train_data_path):
            print(f"❌ 训练数据文件不存在: {train_data_path}")
            return False

        with open(train_data_path, 'rb') as f:
            data = pickle.load(f)

        X_train = data.get('X_train')
        y_train = data.get('y_train')

        if X_train is None or y_train is None:
            print("❌ 训练数据为空")
            return False

        # 样本数量
        n_samples = len(X_train)
        n_features = X_train.shape[1]
        print(f"✅ 训练样本数: {n_samples:,}")
        print(f"✅ 特征维度: {n_features}")

        # 特征完整度
        nan_ratio = np.isnan(X_train).sum() / X_train.size
        print(f"✅ NaN比例: {nan_ratio*100:.2f}%")

        if nan_ratio > 0.2:
            print(f"⚠️  警告：NaN比例过高 ({nan_ratio*100:.1f}%)")

        # 标签分布
        print(f"\n📈 标签（未来收益率）统计:")
        print(f"   均值: {np.mean(y_train):.4f}")
        print(f"   标准差: {np.std(y_train):.4f}")
        print(f"   最小值: {np.min(y_train):.4f}")
        print(f"   最大值: {np.max(y_train):.4f}")
        print(f"   中位数: {np.median(y_train):.4f}")

        # 分位数
        q25, q75 = np.percentile(y_train, [25, 75])
        print(f"   25%分位: {q25:.4f}")
        print(f"   75%分位: {q75:.4f}")

        return True

    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_model_files(model_dir='ml_models/trained_models/v39'):
    """验证模型文件存在性"""
    print("\n" + "="*80)
    print("📁 验证2: 模型文件完整性")
    print("="*80)

    required_files = [
        'layer1_lgb.pkl',
        'layer1_xgb.pkl',
        'layer1_rf.pkl',
        'layer1_cat.pkl',
        'layer2_meta.pkl',
        'scaler.pkl',
        'training_info.json'
    ]

    all_exist = True
    for filename in required_files:
        filepath = f"{model_dir}/{filename}"
        exists = os.path.exists(filepath)
        status = "✅" if exists else "❌"

        if exists:
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            print(f"{status} {filename}: {size_mb:.2f} MB")
        else:
            print(f"{status} {filename}: 不存在")
            all_exist = False

    return all_exist

def validate_prediction_function():
    """验证模型预测功能"""
    print("\n" + "="*80)
    print("🔮 验证3: 模型预测功能")
    print("="*80)

    try:
        from ml_models.v39 import V390EnhancedFeatureMLSystem

        print("初始化V3.9系统...")
        system = V390EnhancedFeatureMLSystem()

        # 加载模型
        if not system.load_model():
            print("❌ 模型加载失败")
            return False

        print("✅ 模型加载成功")

        # 测试评分功能
        test_stocks = ['000001', '600000', '000002']
        test_date = '2025-10-31'

        print(f"\n测试评分功能 (日期: {test_date}):")
        scores = []

        for code in test_stocks:
            try:
                result = system.score_stock(code, test_date)
                if result and 'score' in result:
                    score = result['score']
                    scores.append(score)
                    print(f"   {code}: {score:.2f}/100 ✅")
                else:
                    print(f"   {code}: 评分失败 ❌")
            except Exception as e:
                print(f"   {code}: 错误 - {e}")

        if not scores:
            print("❌ 所有测试股票评分失败")
            return False

        print(f"\n✅ 成功评分 {len(scores)}/{len(test_stocks)} 只股票")
        return True

    except Exception as e:
        print(f"❌ 预测功能验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_score_distribution():
    """验证评分分布合理性"""
    print("\n" + "="*80)
    print("📊 验证4: 评分分布合理性")
    print("="*80)

    try:
        from ml_models.v39 import V390EnhancedFeatureMLSystem
        import sqlite3

        system = V390EnhancedFeatureMLSystem()
        if not system.load_model():
            print("❌ 模型加载失败")
            return False

        # 随机选择100只股票
        conn = sqlite3.connect('data_adapter/stock_data.db')
        query = """
            SELECT code FROM securities
            WHERE type = 'A股'
            ORDER BY RANDOM()
            LIMIT 100
        """
        df = pd.read_sql(query, conn)
        conn.close()

        test_date = '2025-10-31'
        scores = []

        print(f"对100只随机股票进行评分...")
        for code in df['code'].values:
            try:
                result = system.score_stock(code, test_date)
                if result and 'score' in result:
                    scores.append(result['score'])
            except:
                pass

        if len(scores) < 50:
            print(f"⚠️  成功评分数量过少: {len(scores)}/100")
            return False

        scores = np.array(scores)

        print(f"\n✅ 成功评分: {len(scores)}/100")
        print(f"\n评分统计:")
        print(f"   均值: {np.mean(scores):.2f}")
        print(f"   标准差: {np.std(scores):.2f}")
        print(f"   最小值: {np.min(scores):.2f}")
        print(f"   最大值: {np.max(scores):.2f}")
        print(f"   中位数: {np.median(scores):.2f}")

        # 检查范围
        if np.min(scores) < 0 or np.max(scores) > 100:
            print(f"❌ 评分超出范围 [0, 100]")
            return False

        # 检查分布
        q25, q75 = np.percentile(scores, [25, 75])
        iqr = q75 - q25
        print(f"   25%分位: {q25:.2f}")
        print(f"   75%分位: {q75:.2f}")
        print(f"   IQR: {iqr:.2f}")

        if iqr < 5:
            print(f"⚠️  评分分布过于集中 (IQR={iqr:.2f})")

        # 分布直方图
        print(f"\n评分分布:")
        bins = [0, 20, 40, 60, 80, 100]
        hist, _ = np.histogram(scores, bins=bins)
        for i in range(len(bins)-1):
            count = hist[i]
            pct = count / len(scores) * 100
            bar = '█' * int(pct / 2)
            print(f"   [{bins[i]:3d}-{bins[i+1]:3d}): {count:3d} ({pct:5.1f}%) {bar}")

        return True

    except Exception as e:
        print(f"❌ 评分分布验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_feature_importance(model_dir='ml_models/trained_models/v39'):
    """验证特征重要性"""
    print("\n" + "="*80)
    print("🎯 验证5: 特征重要性分析")
    print("="*80)

    try:
        # 加载LightGBM模型
        lgb_path = f"{model_dir}/layer1_lgb.pkl"
        if not os.path.exists(lgb_path):
            print(f"❌ LightGBM模型文件不存在")
            return False

        with open(lgb_path, 'rb') as f:
            lgb_model = pickle.load(f)

        # 获取特征重要性
        importance = lgb_model.feature_importances_

        if len(importance) == 0:
            print("❌ 特征重要性为空")
            return False

        # Top 10特征
        top_indices = np.argsort(importance)[::-1][:10]

        print("✅ Top 10 重要特征:")
        for rank, idx in enumerate(top_indices, 1):
            print(f"   {rank:2d}. 特征{idx}: {importance[idx]:.0f}")

        # 检查是否有dominant特征
        max_importance = np.max(importance)
        total_importance = np.sum(importance)
        dominance = max_importance / total_importance

        print(f"\n特征重要性分布:")
        print(f"   最大特征占比: {dominance*100:.1f}%")

        if dominance > 0.5:
            print(f"⚠️  单一特征过于dominant ({dominance*100:.1f}%)")

        return True

    except Exception as e:
        print(f"❌ 特征重要性验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_training_info(model_dir='ml_models/trained_models/v39'):
    """验证训练信息"""
    print("\n" + "="*80)
    print("📋 验证6: 训练信息")
    print("="*80)

    try:
        import json

        info_path = f"{model_dir}/training_info.json"
        if not os.path.exists(info_path):
            print(f"❌ 训练信息文件不存在")
            return False

        with open(info_path, 'r') as f:
            info = json.load(f)

        print("训练配置:")
        print(f"   训练日期范围: {info.get('start_date')} ~ {info.get('end_date')}")
        print(f"   训练样本数: {info.get('n_samples', 'N/A'):,}")
        print(f"   特征数量: {info.get('n_features', 'N/A')}")
        print(f"   训练耗时: {info.get('training_time', 'N/A')}")

        if 'model_performance' in info:
            perf = info['model_performance']
            print(f"\n模型性能:")
            print(f"   R² Score: {perf.get('r2_score', 'N/A')}")
            print(f"   RMSE: {perf.get('rmse', 'N/A')}")
            print(f"   MAE: {perf.get('mae', 'N/A')}")

        return True

    except Exception as e:
        print(f"❌ 训练信息验证失败: {e}")
        return False

def main():
    """主验证流程"""
    print("\n" + "="*80)
    print("🔍 V3.9模型训练验证")
    print("="*80)
    print(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    # 验证1: 训练数据
    results['训练数据'] = validate_training_data()

    # 验证2: 模型文件
    results['模型文件'] = validate_model_files()

    # 验证3: 预测功能
    results['预测功能'] = validate_prediction_function()

    # 验证4: 评分分布
    results['评分分布'] = validate_score_distribution()

    # 验证5: 特征重要性
    results['特征重要性'] = validate_feature_importance()

    # 验证6: 训练信息
    results['训练信息'] = validate_training_info()

    # 汇总结果
    print("\n" + "="*80)
    print("📊 验证结果汇总")
    print("="*80)

    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name}: {status}")

    success_count = sum(results.values())
    total_count = len(results)
    pass_rate = (success_count / total_count) * 100

    print("\n" + "="*80)
    print(f"总体通过率: {success_count}/{total_count} ({pass_rate:.1f}%)")
    print("="*80)

    if success_count == total_count:
        print("\n🎉 所有验证通过！模型训练成功")
        return 0
    elif success_count >= total_count * 0.8:
        print("\n⚠️  大部分验证通过，模型基本可用")
        return 0
    else:
        print("\n❌ 多个验证失败，需要检查训练过程")
        return 1

if __name__ == "__main__":
    sys.exit(main())
