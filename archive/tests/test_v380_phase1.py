#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.8 Phase 1测试运行器
验证Phase 1基础架构搭建的完成情况
"""

import sys
import os
from datetime import datetime

def main():
    print("🚀 V3.8 Phase 1 基础架构测试")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 导入并运行测试
    try:
        sys.path.append('/Users/yangxu/StockTradebyZ')
        from incremental_learning.tests.test_v380_basic import run_all_tests

        success = run_all_tests()

        print("\n" + "=" * 60)
        if success:
            print("🎉 Phase 1基础架构搭建完成！")
            print("✅ 所有基础测试通过")
            print("\n📋 Phase 1完成项目：")
            print("  ✓ V3.8主系统框架创建")
            print("  ✓ 增量学习基础接口设计")
            print("  ✓ 增强监控系统配置")
            print("  ✓ incremental_learning目录结构")
            print("  ✓ 实时数据获取接口")
            print("  ✓ 特征存储结构优化")
            print("  ✓ 基础测试用例")
            print("\n🎯 准备进入Phase 2: 实时特征增强")
        else:
            print("⚠️ Phase 1部分功能需要调整")
            print("请检查失败的测试用例")

    except ImportError as e:
        print(f"❌ 导入模块失败: {e}")
        print("请确保所有必要的文件都已创建")
    except Exception as e:
        print(f"❌ 测试运行失败: {e}")

if __name__ == '__main__':
    main()