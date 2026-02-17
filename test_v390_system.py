#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.9系统快速测试脚本

测试v3.9系统的各个组件是否正常工作

作者: Claude Code
创建时间: 2025-11-03
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def test_import():
    """测试导入"""
    print("=" * 60)
    print("测试 1: 导入v3.9模块")
    print("=" * 60)

    try:
        from ml_models.v39 import V390EnhancedFeatureMLSystem
        print("✅ V390EnhancedFeatureMLSystem 导入成功")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_initialization():
    """测试初始化"""
    print("\n" + "=" * 60)
    print("测试 2: 初始化v3.9系统")
    print("=" * 60)

    try:
        from ml_models.v39 import V390EnhancedFeatureMLSystem
        system = V390EnhancedFeatureMLSystem()
        print("✅ 系统初始化成功")
        print(f"   - 回望天数: {system.lookback_days}")
        print(f"   - 前瞻天数: {system.lookahead_days}")
        print(f"   - 特征提取器: 技术, 基本面, 市场")
        print(f"   - 量化选择器: {list(system.selectors.keys())}")
        return True
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_extraction():
    """测试特征提取"""
    print("\n" + "=" * 60)
    print("测试 3: 特征提取（使用测试数据）")
    print("=" * 60)

    try:
        from ml_models.v39 import V390EnhancedFeatureMLSystem
        system = V390EnhancedFeatureMLSystem()

        # 测试股票（平安银行）
        test_code = "000001.SZ"
        test_date = "2025-10-31"

        print(f"测试股票: {test_code}")
        print(f"测试日期: {test_date}")

        features = system.extract_features(test_code, test_date)

        if features is not None:
            print(f"✅ 特征提取成功")
            print(f"   - 特征数量: {features.shape[1]}")
            print(f"   - 特征列表（前10个）: {list(features.columns[:10])}")
            return True
        else:
            print(f"⚠️ 特征提取返回None（可能数据库为空）")
            return False

    except Exception as e:
        print(f"❌ 特征提取失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database_connection():
    """测试数据库连接"""
    print("\n" + "=" * 60)
    print("测试 4: 数据库连接")
    print("=" * 60)

    try:
        import sqlite3
        import os

        db_path = "data_adapter/stock_data.db"

        if not os.path.exists(db_path):
            print(f"⚠️ 数据库文件不存在: {db_path}")
            return False

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 检查表
        tables = ['securities', 'daily_quotes', 'daily_basic', 'financial_indicator']
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            status = "✅" if count > 0 else "❌"
            print(f"{status} {table}: {count:,} 条记录")

        conn.close()
        return True

    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return False

def main():
    """主测试函数"""
    print("\n🚀 V3.9系统测试")
    print("=" * 60)

    results = {
        '导入': test_import(),
        '初始化': test_initialization(),
        '数据库连接': test_database_connection(),
        '特征提取': test_feature_extraction(),
    }

    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)

    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")

    success_count = sum(results.values())
    total_count = len(results)
    pass_rate = (success_count / total_count) * 100

    print("\n" + "=" * 60)
    print(f"总体通过率: {success_count}/{total_count} ({pass_rate:.1f}%)")
    print("=" * 60)

    if success_count == total_count:
        print("\n🎉 所有测试通过！v3.9系统就绪")
    elif success_count >= total_count - 1:
        print("\n⚠️ 大部分测试通过，可能需要检查数据")
    else:
        print("\n❌ 多个测试失败，需要检查系统配置")

    return success_count == total_count


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
