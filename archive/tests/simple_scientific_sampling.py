#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版科学采样策略

基于市值和数据质量进行分层抽样
确保训练样本的代表性和充分性

Created: 2025-09-16
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append('/Users/yangxu/StockTradebyZ')
from data_adapter.database_manager import DatabaseManager

def scientific_stock_sampling():
    """科学的股票采样"""
    print("🎯 执行科学股票采样策略")
    print("="*50)

    db_manager = DatabaseManager("data_adapter/stock_data.db")

    with db_manager.get_connection() as conn:
        # 获取有充足数据且有市值信息的股票
        query = """
        SELECT
            s.code,
            s.name,
            COUNT(DISTINCT dq.trade_date) as data_days,
            AVG(CAST(db.total_mv as REAL)) as avg_market_cap,
            MIN(dq.trade_date) as start_date,
            MAX(dq.trade_date) as end_date,
            COUNT(DISTINCT db.trade_date) as basic_data_days
        FROM securities s
        LEFT JOIN daily_quotes dq ON s.id = dq.security_id
        LEFT JOIN daily_basic db ON s.id = db.security_id AND dq.trade_date = db.trade_date
        WHERE s.type = 'A股'
            AND dq.trade_date >= '2020-01-01'
            AND db.total_mv IS NOT NULL
        GROUP BY s.code, s.name
        HAVING data_days >= 800  -- 至少3年多数据
            AND basic_data_days >= 400  -- 基本面数据也要充足
        ORDER BY data_days DESC, avg_market_cap DESC
        """

        candidates = pd.read_sql(query, conn)

    if len(candidates) == 0:
        print("❌ 未找到符合条件的股票")
        return None

    print(f"📊 候选股票数量: {len(candidates)}只")
    print(f"📅 数据时间跨度: {candidates['start_date'].min()} 到 {candidates['end_date'].max()}")
    print(f"📈 平均数据天数: {candidates['data_days'].mean():.0f}天")

    # 按市值分层抽样
    candidates = candidates.dropna(subset=['avg_market_cap'])
    candidates['market_cap_log'] = np.log(candidates['avg_market_cap'])

    # 分成5个市值层级
    candidates['cap_quintile'] = pd.qcut(
        candidates['market_cap_log'],
        q=5,
        labels=['微盘', '小盘', '中小盘', '中盘', '大盘']
    )

    print(f"\n📊 市值分布:")
    cap_dist = candidates['cap_quintile'].value_counts().sort_index()
    for quintile, count in cap_dist.items():
        print(f"  {quintile}: {count}只股票")

    # 目标：每个层级选30只股票，总共150只
    target_per_quintile = 30
    selected_stocks = []

    for quintile in ['微盘', '小盘', '中小盘', '中盘', '大盘']:
        quintile_stocks = candidates[candidates['cap_quintile'] == quintile]

        # 在该层级内按数据质量排序选择
        if len(quintile_stocks) >= target_per_quintile:
            # 选择数据最完整的股票
            selected = quintile_stocks.nlargest(target_per_quintile, 'data_days')
        else:
            # 如果该层级股票不够，全选
            selected = quintile_stocks

        selected_stocks.extend(selected['code'].tolist())
        print(f"  {quintile}: 选择{len(selected)}只")

    # 获取最终选择的股票详情
    final_selection = candidates[candidates['code'].isin(selected_stocks)].copy()

    print(f"\n✅ 最终采样结果:")
    print(f"  选择股票数: {len(final_selection)}只")
    print(f"  平均数据天数: {final_selection['data_days'].mean():.0f}天")
    print(f"  数据天数范围: {final_selection['data_days'].min()}-{final_selection['data_days'].max()}天")

    # 估算训练样本量
    total_samples = final_selection['data_days'].sum()
    print(f"  预计训练样本: {total_samples:,}条")

    # 验证样本充分性
    features_count = 49
    min_samples_needed = features_count * 10
    sample_adequacy = total_samples / min_samples_needed

    print(f"\n📏 样本充分性验证:")
    print(f"  特征维度: {features_count}")
    print(f"  最小样本需求: {min_samples_needed:,}条 (特征数×10)")
    print(f"  实际样本: {total_samples:,}条")
    print(f"  充分性比例: {sample_adequacy:.1f}x")

    if sample_adequacy >= 1.0:
        print(f"  ✅ 样本量充分")
        adequacy_status = "充分"
    else:
        print(f"  ⚠️ 样本量可能不足")
        adequacy_status = "不足"

    # 市值分布验证
    final_cap_dist = final_selection['cap_quintile'].value_counts().sort_index()
    print(f"\n📊 最终市值分布:")
    for quintile, count in final_cap_dist.items():
        print(f"  {quintile}: {count}只 ({count/len(final_selection)*100:.1f}%)")

    # 保存结果
    selected_stocks_info = {
        'codes': final_selection['code'].tolist(),
        'total_samples': int(total_samples),
        'sample_adequacy': adequacy_status,
        'time_span': f"{final_selection['start_date'].min()} to {final_selection['end_date'].max()}",
        'avg_data_days': int(final_selection['data_days'].mean())
    }

    # 保存到文件
    with open('/Users/yangxu/StockTradebyZ/v380_scientific_training_stocks.txt', 'w') as f:
        f.write("# V3.80科学采样股票列表\n")
        f.write(f"# 总计: {len(final_selection)}只股票\n")
        f.write(f"# 预计样本: {total_samples:,}条\n")
        f.write(f"# 样本充分性: {adequacy_status}\n\n")

        for _, row in final_selection.iterrows():
            f.write(f"{row['code']} # {row['name']} - {row['cap_quintile']} - {row['data_days']}天\n")

    print(f"\n📝 股票列表已保存到: v380_scientific_training_stocks.txt")

    return selected_stocks_info

def compare_with_original_strategy():
    """与原始策略对比"""
    print(f"\n📋 训练策略对比:")
    print("="*30)

    print("❌ 原始策略:")
    print("  - 股票数量: 10只")
    print("  - 样本数量: 6,457条")
    print("  - 代表性: 差")
    print("  - 过拟合风险: 高")

    print("\n✅ 科学策略:")
    result = scientific_stock_sampling()
    if result:
        print("  - 股票数量: {}只".format(len(result['codes'])))
        print("  - 样本数量: {:,}条".format(result['total_samples']))
        print("  - 代表性: 好 (市值分层)")
        print("  - 过拟合风险: 低")
        print("  - 样本充分性: {}".format(result['sample_adequacy']))

        improvement_factor = result['total_samples'] / 6457
        print(f"\n🚀 改进效果:")
        print(f"  - 样本量提升: {improvement_factor:.1f}倍")
        print(f"  - 股票数量提升: {len(result['codes'])/10:.1f}倍")

        return result
    return None

if __name__ == "__main__":
    result = compare_with_original_strategy()

    if result:
        print(f"\n🎯 推荐下一步:")
        print(f"  1. 使用新的股票列表重新训练V3.80")
        print(f"  2. 预计训练时间: 显著增加 (样本量大幅提升)")
        print(f"  3. 预期效果: 更好的泛化能力和预测准确性")
    else:
        print(f"\n❌ 采样失败，需要检查数据库")