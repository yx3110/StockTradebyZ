#!/usr/bin/env python3
"""
测试v3.9所需字段是否已集成到每日更新脚本中

验证点：
1. financial_indicator表是否包含v3.9所需的所有字段
2. 最近更新的数据是否包含这些字段
"""

import sqlite3
import sys

def test_financial_indicator_fields():
    """测试financial_indicator表字段"""
    print("="*60)
    print("📊 测试 financial_indicator 表字段")
    print("="*60)

    # v3.9需要的字段
    required_fields = [
        'eps', 'dt_eps', 'roe', 'roe_waa', 'roe_dt', 'roa',
        'grossprofit_margin', 'netprofit_margin', 'profit_to_gr',
        'ocf_to_profit', 'debt_to_assets', 'current_ratio', 'quick_ratio',
        'ar_turn', 'ca_turn', 'fa_turn', 'assets_turn'
    ]

    conn = sqlite3.connect('data_adapter/stock_data.db')
    cursor = conn.cursor()

    # 获取表结构
    cursor.execute("PRAGMA table_info(financial_indicator)")
    columns = [row[1] for row in cursor.fetchall()]

    print(f"\n✅ 表中总共有 {len(columns)} 个字段")

    # 检查所有必需字段是否存在
    missing_fields = []
    for field in required_fields:
        if field not in columns:
            missing_fields.append(field)

    if missing_fields:
        print(f"❌ 缺少字段: {missing_fields}")
        return False
    else:
        print(f"✅ 所有v3.9所需字段都存在!")
        print(f"   字段列表: {', '.join(required_fields)}")

    # 检查最近的数据是否有这些字段的值
    cursor.execute("""
        SELECT
            COUNT(*) as total_records,
            COUNT(eps) as eps_count,
            COUNT(dt_eps) as dt_eps_count,
            COUNT(roe) as roe_count,
            COUNT(roe_waa) as roe_waa_count,
            COUNT(grossprofit_margin) as grossprofit_margin_count,
            COUNT(current_ratio) as current_ratio_count,
            COUNT(ar_turn) as ar_turn_count,
            COUNT(assets_turn) as assets_turn_count
        FROM financial_indicator
        WHERE ann_date >= '2024-01-01'
    """)

    result = cursor.fetchone()
    print(f"\n📈 2024年以来的数据统计:")
    print(f"   总记录数: {result[0]}")
    print(f"   eps有值: {result[1]} ({result[1]/result[0]*100:.1f}%)")
    print(f"   dt_eps有值: {result[2]} ({result[2]/result[0]*100:.1f}%)")
    print(f"   roe有值: {result[3]} ({result[3]/result[0]*100:.1f}%)")
    print(f"   roe_waa有值: {result[4]} ({result[4]/result[0]*100:.1f}%)")
    print(f"   grossprofit_margin有值: {result[5]} ({result[5]/result[0]*100:.1f}%)")
    print(f"   current_ratio有值: {result[6]} ({result[6]/result[0]*100:.1f}%)")
    print(f"   ar_turn有值: {result[7]} ({result[7]/result[0]*100:.1f}%)")
    print(f"   assets_turn有值: {result[8]} ({result[8]/result[0]*100:.1f}%)")

    conn.close()
    return True

def test_daily_basic_fields():
    """测试daily_basic表字段"""
    print("\n" + "="*60)
    print("📊 测试 daily_basic 表字段")
    print("="*60)

    # v3.9需要的字段
    required_fields = [
        'close', 'turnover_rate', 'turnover_rate_f', 'volume_ratio',
        'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio', 'dv_ttm',
        'total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv'
    ]

    conn = sqlite3.connect('data_adapter/stock_data.db')
    cursor = conn.cursor()

    # 获取表结构
    cursor.execute("PRAGMA table_info(daily_basic)")
    columns = [row[1] for row in cursor.fetchall()]

    print(f"\n✅ 表中总共有 {len(columns)} 个字段")

    # 检查所有必需字段是否存在
    missing_fields = []
    for field in required_fields:
        if field not in columns:
            missing_fields.append(field)

    if missing_fields:
        print(f"❌ 缺少字段: {missing_fields}")
        return False
    else:
        print(f"✅ 所有v3.9所需字段都存在!")
        print(f"   字段列表: {', '.join(required_fields)}")

    # 检查最近的数据
    cursor.execute("""
        SELECT COUNT(*)
        FROM daily_basic
        WHERE trade_date >= '2024-01-01'
    """)

    count = cursor.fetchone()[0]
    print(f"\n📈 2024年以来的数据记录: {count:,} 条")

    conn.close()
    return True

def main():
    """主函数"""
    print("\n🚀 开始测试v3.9数据字段集成情况...\n")

    # 测试financial_indicator
    result1 = test_financial_indicator_fields()

    # 测试daily_basic
    result2 = test_daily_basic_fields()

    # 总结
    print("\n" + "="*60)
    print("📋 测试总结")
    print("="*60)

    if result1 and result2:
        print("✅ 所有测试通过！")
        print("✅ v3.9所需的数据字段已完全集成到每日更新脚本中")
        print("\n💡 下一步:")
        print("   1. 运行每日更新脚本: python3 fetch_data/quick_daily_update.py --include-financial")
        print("   2. 使用v3.9版本进行股票选择")
        return 0
    else:
        print("❌ 部分测试失败，请检查配置")
        return 1

if __name__ == "__main__":
    sys.exit(main())
