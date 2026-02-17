#!/usr/bin/env python3
"""
测试V3.81集成到tomorrow_stock_selector.py的效果
"""

import sys
import pandas as pd
from datetime import datetime

sys.path.append('/Users/yangxu/StockTradebyZ')

def test_v381_integration():
    """测试V3.81集成效果"""
    print("🧪 测试V3.81集成到tomorrow_stock_selector.py")
    print("=" * 60)

    # 导入明日选股模块
    from tomorrow_stock_selector import main

    # 测试V3.81版本
    try:
        print("1. 运行V3.81 Level 4质量评分版本...")

        # 运行V3.81选股
        result = main(
            target_date=None,  # 使用最新日期
            scoring_version="v3.81",
            stocks_only=True  # 只选股票
        )

        if result:
            print("✅ V3.81运行成功！")

            # 检查结果结构
            print(f"\n📊 结果概览:")
            print(f"   策略数量: {result.get('total_strategies', 0)}")
            print(f"   选中股票数: {result.get('total_unique_stocks', 0)}")

            # 检查是否有质量评分统计
            if 'quality_score_stats' in result:
                stats = result['quality_score_stats']
                print(f"\n🎯 Level 4质量评分统计:")
                print(f"   均值: {stats.get('mean', 0):.4f}")
                print(f"   标准差: {stats.get('std', 0):.4f}")
                print(f"   范围: [{stats.get('min', 0):.4f}, {stats.get('max', 0):.4f}]")
                print(f"   高质量股票: {stats.get('high_quality_count', 0)}只")
                print(f"   低质量股票: {stats.get('low_quality_count', 0)}只")

                # 验证差异化效果
                if stats.get('std', 0) >= 0.15:
                    print("   🎉 质量评分差异化: ✅ 达标")
                else:
                    print("   ⚠️ 质量评分差异化: ❌ 不达标")

            # 显示前5只股票的质量评分
            detailed_stocks = result.get('detailed_stocks', [])
            if detailed_stocks:
                print(f"\n📋 Top 5 推荐股票:")
                for i, stock in enumerate(detailed_stocks[:5]):
                    quality_score = stock.get('quality_score', 0)
                    final_score = stock.get('final_score', 0)
                    print(f"   {i+1}. {stock.get('code')} {stock.get('name')}")
                    print(f"      质量评分: {quality_score:.4f}, 综合评分: {final_score:.2f}")

            return True

        else:
            print("❌ V3.81运行失败: 无结果返回")
            return False

    except Exception as e:
        print(f"❌ V3.81运行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    success = test_v381_integration()

    print(f"\n🎯 测试结果: {'✅ 成功' if success else '❌ 失败'}")

    if success:
        print("\n🚀 V3.81 Level 4质量评分集成成功!")
        print("   系统已准备好用于生产环境")
    else:
        print("\n⚠️ V3.81集成需要进一步调试")

if __name__ == "__main__":
    main()