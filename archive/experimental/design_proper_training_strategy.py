#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
科学的V3.80训练策略设计

解决样本选取、数据代表性、训练规模等问题
确保机器学习模型能学到A股市场的普遍规律

Created: 2025-09-16
Author: Claude Code
Version: 3.8.0
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

sys.path.append('/Users/yangxu/StockTradebyZ')
from data_adapter.database_manager import DatabaseManager

def analyze_available_data():
    """分析数据库中可用的训练数据"""
    print("📊 分析数据库中可用的训练数据")
    print("="*50)

    db_manager = DatabaseManager("data_adapter/stock_data.db")

    with db_manager.get_connection() as conn:
        # 1. 统计股票总数
        stock_count = pd.read_sql("SELECT COUNT(*) as count FROM securities WHERE type='A股'", conn)
        print(f"📈 A股总数: {stock_count.iloc[0]['count']}只")

        # 2. 统计数据时间跨度
        date_range = pd.read_sql("""
            SELECT
                MIN(trade_date) as start_date,
                MAX(trade_date) as end_date,
                COUNT(DISTINCT trade_date) as trading_days
            FROM daily_quotes
        """, conn)

        start = date_range.iloc[0]['start_date']
        end = date_range.iloc[0]['end_date']
        days = date_range.iloc[0]['trading_days']

        print(f"📅 数据时间跨度: {start} 到 {end}")
        print(f"📅 总交易日数: {days}天")

        # 3. 分析数据完整性
        completeness = pd.read_sql("""
            SELECT
                s.code,
                s.name,
                COUNT(dq.trade_date) as data_days,
                MIN(dq.trade_date) as first_date,
                MAX(dq.trade_date) as last_date
            FROM securities s
            LEFT JOIN daily_quotes dq ON s.id = dq.security_id
            WHERE s.type = 'A股'
            GROUP BY s.id, s.code, s.name
            HAVING COUNT(dq.trade_date) > 500  -- 至少2年数据
            ORDER BY data_days DESC
            LIMIT 20
        """, conn)

        print(f"\n📋 数据最完整的20只股票:")
        for _, row in completeness.iterrows():
            print(f"  {row['code']} {row['name']}: {row['data_days']}天 ({row['first_date']} 到 {row['last_date']})")

        # 4. 按行业分析
        industry_analysis = pd.read_sql("""
            SELECT
                sbi.industry,
                COUNT(DISTINCT s.code) as stock_count,
                AVG(data_counts.data_days) as avg_data_days
            FROM securities s
            LEFT JOIN stock_basic_info sbi ON s.code = sbi.code
            LEFT JOIN (
                SELECT
                    s.code,
                    COUNT(dq.trade_date) as data_days
                FROM securities s
                LEFT JOIN daily_quotes dq ON s.id = dq.security_id
                WHERE s.type = 'A股'
                GROUP BY s.code
            ) data_counts ON s.code = data_counts.code
            WHERE s.type = 'A股' AND data_counts.data_days > 500
            GROUP BY sbi.industry
            HAVING stock_count >= 5  -- 至少5只股票的行业
            ORDER BY stock_count DESC
            LIMIT 15
        """, conn)

        print(f"\n🏭 主要行业分布:")
        for _, row in industry_analysis.iterrows():
            industry = row['industry'] if row['industry'] else '未分类'
            print(f"  {industry}: {row['stock_count']}只股票, 平均{row['avg_data_days']:.0f}天数据")

        return {
            'total_stocks': stock_count.iloc[0]['count'],
            'date_range': (start, end, days),
            'top_stocks': completeness,
            'industries': industry_analysis
        }

def design_scientific_sampling_strategy(data_analysis):
    """设计科学的采样策略"""
    print(f"\n🎯 设计科学采样策略")
    print("="*40)

    # 目标：选择150-200只代表性股票
    target_stocks = 150

    db_manager = DatabaseManager("data_adapter/stock_data.db")

    with db_manager.get_connection() as conn:
        # 策略1: 按市值分层抽样
        market_cap_samples = pd.read_sql("""
            SELECT DISTINCT
                s.code,
                s.name,
                sbi.industry,
                sbi.area,
                AVG(db.total_mv) as avg_market_cap,
                COUNT(dq.trade_date) as data_days
            FROM securities s
            LEFT JOIN stock_basic_info sbi ON s.code = sbi.code
            LEFT JOIN daily_quotes dq ON s.id = dq.security_id
            LEFT JOIN daily_basic db ON s.id = db.security_id
            WHERE s.type = 'A股'
                AND dq.trade_date >= '2022-01-01'
                AND db.total_mv IS NOT NULL
            GROUP BY s.code, s.name, sbi.industry, sbi.area
            HAVING data_days >= 600  -- 至少2.5年数据
            ORDER BY avg_market_cap DESC
        """, conn)

        if len(market_cap_samples) == 0:
            print("❌ 未找到符合条件的股票数据")
            return None

        print(f"📊 符合条件的股票: {len(market_cap_samples)}只")

        # 分层抽样
        market_cap_samples['market_cap_log'] = np.log(market_cap_samples['avg_market_cap'])

        # 按市值分成4个层级
        market_cap_samples['cap_tier'] = pd.qcut(
            market_cap_samples['market_cap_log'],
            q=4,
            labels=['小盘', '中小盘', '中盘', '大盘']
        )

        # 按行业分组
        industry_groups = market_cap_samples.groupby('industry').size().sort_values(ascending=False)

        print(f"\n📈 市值分层:")
        cap_distribution = market_cap_samples['cap_tier'].value_counts().sort_index()
        for tier, count in cap_distribution.items():
            print(f"  {tier}: {count}只")

        print(f"\n🏭 主要行业 (前10):")
        for industry, count in industry_groups.head(10).items():
            industry_name = industry if pd.notna(industry) else '未分类'
            print(f"  {industry_name}: {count}只")

        # 执行分层抽样
        selected_stocks = []

        # 每个市值层级选择股票
        stocks_per_tier = target_stocks // 4

        for tier in ['小盘', '中小盘', '中盘', '大盘']:
            tier_stocks = market_cap_samples[market_cap_samples['cap_tier'] == tier]

            # 在该层级内按行业分布抽样
            tier_industries = tier_stocks['industry'].value_counts()

            tier_selected = []
            remaining_slots = stocks_per_tier

            # 先从每个主要行业选1-2只
            for industry in tier_industries.head(10).index:
                if remaining_slots <= 0:
                    break

                industry_stocks = tier_stocks[tier_stocks['industry'] == industry]

                # 每个行业最多选2只，按数据质量排序
                n_select = min(2, len(industry_stocks), remaining_slots)
                selected = industry_stocks.nlargest(n_select, 'data_days')

                tier_selected.extend(selected['code'].tolist())
                remaining_slots -= n_select

            # 如果还有剩余额度，随机选择
            if remaining_slots > 0:
                remaining_stocks = tier_stocks[~tier_stocks['code'].isin(tier_selected)]
                if len(remaining_stocks) > 0:
                    additional = remaining_stocks.sample(
                        min(remaining_slots, len(remaining_stocks)),
                        random_state=42
                    )
                    tier_selected.extend(additional['code'].tolist())

            selected_stocks.extend(tier_selected)
            print(f"  {tier}股票选择: {len(tier_selected)}只")

        # 获取选中股票的详细信息
        selected_details = market_cap_samples[
            market_cap_samples['code'].isin(selected_stocks)
        ].copy()

        print(f"\n✅ 最终选择股票数: {len(selected_details)}只")
        print(f"📊 按市值分布:")
        final_distribution = selected_details['cap_tier'].value_counts().sort_index()
        for tier, count in final_distribution.items():
            print(f"  {tier}: {count}只")

        print(f"📊 按行业分布 (前8):")
        final_industries = selected_details['industry'].value_counts()
        for industry, count in final_industries.head(8).items():
            industry_name = industry if pd.notna(industry) else '未分类'
            print(f"  {industry_name}: {count}只")

        # 估算训练样本量
        avg_days = selected_details['data_days'].mean()
        total_samples = len(selected_details) * avg_days

        print(f"\n📈 训练数据估算:")
        print(f"  股票数量: {len(selected_details)}只")
        print(f"  平均数据天数: {avg_days:.0f}天")
        print(f"  预计总样本数: {total_samples:.0f}条")
        print(f"  特征维度: 49维")

        return {
            'selected_stocks': selected_details['code'].tolist(),
            'stock_details': selected_details,
            'estimated_samples': int(total_samples)
        }

def validate_training_strategy(sampling_result):
    """验证训练策略的合理性"""
    print(f"\n🔍 验证训练策略合理性")
    print("="*35)

    if not sampling_result:
        print("❌ 采样策略失败")
        return False

    stocks = sampling_result['selected_stocks']
    estimated_samples = sampling_result['estimated_samples']

    print(f"✅ 样本量评估:")
    print(f"  股票数量: {len(stocks)}只")
    print(f"  预计样本数: {estimated_samples:,}条")
    print(f"  特征维度: 49维")

    # ML样本量经验法则检验
    min_samples_rule1 = 49 * 10  # 特征数量 × 10
    min_samples_rule2 = 1000    # 经验最小值

    print(f"\n📏 样本量充分性检验:")
    print(f"  规则1 (特征数×10): {min_samples_rule1}")
    print(f"  规则2 (经验最小值): {min_samples_rule2}")
    print(f"  实际样本数: {estimated_samples:,}")

    if estimated_samples >= min_samples_rule1 and estimated_samples >= min_samples_rule2:
        print(f"  ✅ 样本量充分")
        adequacy_score = min(estimated_samples / min_samples_rule1 / 5, 1.0)
    else:
        print(f"  ⚠️ 样本量可能不足")
        adequacy_score = estimated_samples / max(min_samples_rule1, min_samples_rule2)

    # 多样性评估
    diversity_score = min(len(stocks) / 100, 1.0)  # 目标100只股票

    print(f"\n📊 训练策略评分:")
    print(f"  样本充分性: {adequacy_score:.1%}")
    print(f"  样本多样性: {diversity_score:.1%}")

    overall_score = (adequacy_score + diversity_score) / 2
    print(f"  综合评分: {overall_score:.1%}")

    if overall_score >= 0.8:
        print(f"  ✅ 训练策略优秀")
        return True
    elif overall_score >= 0.6:
        print(f"  ⚡ 训练策略良好")
        return True
    else:
        print(f"  ❌ 训练策略需要改进")
        return False

if __name__ == "__main__":
    print("🚀 V3.80科学训练策略设计")
    print("="*60)

    # 步骤1: 分析数据
    data_analysis = analyze_available_data()

    # 步骤2: 设计采样策略
    sampling_result = design_scientific_sampling_strategy(data_analysis)

    # 步骤3: 验证策略
    is_valid = validate_training_strategy(sampling_result)

    if is_valid and sampling_result:
        print(f"\n🎯 推荐训练配置:")
        print(f"  训练股票数: {len(sampling_result['selected_stocks'])}只")
        print(f"  预计样本数: {sampling_result['estimated_samples']:,}条")
        print(f"  时间跨度: 2022-01-01 到 2025-09-10")
        print(f"  特征维度: 49维")

        # 保存选中的股票列表
        selected_stocks = sampling_result['selected_stocks']
        with open('/Users/yangxu/StockTradebyZ/v380_training_stocks.txt', 'w') as f:
            for stock in selected_stocks:
                f.write(stock + '\n')

        print(f"\n✅ 股票列表已保存到 v380_training_stocks.txt")
        print(f"📝 建议下一步：使用这些股票重新训练V3.80模型")
    else:
        print(f"\n❌ 训练策略需要重新设计")