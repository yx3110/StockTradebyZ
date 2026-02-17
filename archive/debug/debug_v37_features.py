#!/usr/bin/env python3
"""
V3.7特征诊断脚本
分析模型期望特征 vs 实际提取特征的差异
"""
import sys
sys.path.append('.')

from v370_advanced_ml_system import V370AdvancedMLSystem
import pandas as pd
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def diagnose_v37_features():
    """诊断V3.7特征不匹配问题"""

    print("🔍 V3.7特征诊断开始...")

    # 1. 初始化V3.7系统
    v37_system = V370AdvancedMLSystem()

    # 2. 提取一只测试股票的特征
    test_code = '000001'  # 测试股票
    test_date = '2025-09-20'

    print(f"\n📊 提取测试股票 {test_code} 在 {test_date} 的特征...")

    try:
        # 提取特征
        features_df = v37_system.extract_advanced_features([test_code], test_date, test_date)

        print(f"✅ 实际提取到的特征数量: {len(features_df.columns)}")
        print(f"📋 实际特征列表:")
        for i, col in enumerate(features_df.columns, 1):
            print(f"  {i:2d}. {col}")

        # 3. 检查模型期望的特征
        print(f"\n🤖 检查V3.7模型期望的特征...")

        # 检查scaler期望的特征
        target = 'target_1d'
        if target in v37_system.scalers:
            scaler = v37_system.scalers[target]
            if hasattr(scaler, 'feature_names_in_'):
                expected_features = list(scaler.feature_names_in_)
                print(f"✅ 模型期望的特征数量: {len(expected_features)}")
                print(f"📋 模型期望的特征列表:")
                for i, col in enumerate(expected_features, 1):
                    print(f"  {i:2d}. {col}")

                # 4. 分析差异
                print(f"\n🔍 特征差异分析:")
                actual_features = set(features_df.columns)
                expected_features_set = set(expected_features)

                missing_features = expected_features_set - actual_features
                extra_features = actual_features - expected_features_set

                print(f"❌ 缺失的特征 ({len(missing_features)}个):")
                for i, feature in enumerate(sorted(missing_features), 1):
                    print(f"  {i:2d}. {feature}")

                print(f"➕ 多余的特征 ({len(extra_features)}个):")
                for i, feature in enumerate(sorted(extra_features), 1):
                    print(f"  {i:2d}. {feature}")

                return expected_features, list(features_df.columns), missing_features, extra_features
            else:
                print("❌ Scaler没有feature_names_in_属性")
        else:
            print(f"❌ 找不到target {target}的scaler")

    except Exception as e:
        print(f"❌ 特征提取失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

if __name__ == "__main__":
    expected, actual, missing, extra = diagnose_v37_features()