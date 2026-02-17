#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2025年聚焦训练策略设计

基于V3.7行业均衡采样机制 + 2025年近4个月最新数据
更好反映当前A股市场特征

Created: 2025-09-16
Author: Claude Code
Version: 3.8.0
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append('/Users/yangxu/StockTradebyZ')
from data_adapter.database_manager import DatabaseManager

def analyze_2025_market_data():
    """分析2025年可用的市场数据"""
    print("📊 分析2025年A股市场数据")
    print("="*50)

    db_manager = DatabaseManager("data_adapter/stock_data.db")

    with db_manager.get_connection() as conn:
        # 统计2025年数据情况
        query = """
        SELECT
            MIN(trade_date) as start_date,
            MAX(trade_date) as end_date,
            COUNT(DISTINCT trade_date) as trading_days,
            COUNT(DISTINCT dq.security_id) as active_stocks
        FROM daily_quotes dq
        WHERE trade_date >= '2025-01-01'
        """

        date_stats = pd.read_sql(query, conn)

        start = date_stats.iloc[0]['start_date']
        end = date_stats.iloc[0]['end_date']
        days = date_stats.iloc[0]['trading_days']
        stocks = date_stats.iloc[0]['active_stocks']

        print(f"📅 2025年数据范围: {start} 到 {end}")
        print(f"📅 交易日总数: {days}天")
        print(f"📈 活跃股票数: {stocks}只")

        # 计算数据充分性
        months = (datetime.strptime(end, '%Y-%m-%d') - datetime.strptime(start, '%Y-%m-%d')).days / 30
        print(f"📅 时间跨度: {months:.1f}个月")

        if months >= 3.5:
            print("✅ 数据时间跨度充足 (≥3.5个月)")
        else:
            print("⚠️ 数据时间跨度较短")

        return {
            'start_date': start,
            'end_date': end,
            'trading_days': days,
            'active_stocks': stocks,
            'months': months
        }

def intelligent_industry_sampling_2025():
    """参考V3.7的行业均衡采样 - 2025年版"""
    print(f"\n🎯 执行行业均衡采样 (2025年版)")
    print("="*45)

    db_manager = DatabaseManager("data_adapter/stock_data.db")

    with db_manager.get_connection() as conn:
        # 获取2025年有完整数据的股票（按行业分组）
        query = """
        SELECT
            s.code,
            s.name,
            COALESCE(s.industry, '未分类') as industry,
            COUNT(DISTINCT dq.trade_date) as data_days_2025,
            AVG(CAST(db.total_mv as REAL)) as avg_market_cap,
            COUNT(DISTINCT db.trade_date) as basic_data_days
        FROM securities s
        LEFT JOIN daily_quotes dq ON s.id = dq.security_id
        LEFT JOIN daily_basic db ON s.id = db.security_id AND dq.trade_date = db.trade_date
        WHERE s.type = 'A股'
            AND dq.trade_date >= '2025-01-01'
            AND dq.trade_date <= '2025-09-16'
            AND s.name NOT LIKE '%ST%'
            AND s.name NOT LIKE '%*ST%'
            AND db.total_mv IS NOT NULL
        GROUP BY s.code, s.name, s.industry
        HAVING data_days_2025 >= 80  -- 至少80天2025年数据
            AND basic_data_days >= 40   -- 基本面数据也要充足
        ORDER BY data_days_2025 DESC, avg_market_cap DESC
        """

        candidates = pd.read_sql(query, conn)

    if len(candidates) == 0:
        print("❌ 未找到符合条件的2025年股票数据")
        return None

    print(f"📊 2025年候选股票: {len(candidates)}只")
    print(f"📅 平均数据天数: {candidates['data_days_2025'].mean():.0f}天")

    # 行业分布分析
    industry_counts = candidates.groupby('industry').agg({
        'code': 'count',
        'avg_market_cap': 'mean',
        'data_days_2025': 'mean'
    }).round(0)

    industry_counts.columns = ['股票数', '平均市值', '平均数据天数']
    industry_counts = industry_counts.sort_values('股票数', ascending=False)

    print(f"\n🏭 行业分布 (前15个主要行业):")
    for industry, stats in industry_counts.head(15).iterrows():
        print(f"  {industry[:20]:<20}: {stats['股票数']:>3.0f}只 (市值:{stats['平均市值']:>8.0f})")

    # 执行行业均衡采样 (参考V3.7策略)
    target_total = 1200  # 比V3.7的1000更大胆！
    min_per_industry = 5   # 每个行业至少5只
    max_per_industry = 80  # 每个行业最多80只

    selected_stocks = []
    industry_selections = {}

    # 第1步：为主要行业分配股票
    major_industries = industry_counts.head(20).index.tolist()  # 前20个行业
    base_allocation = target_total // len(major_industries)

    for industry in major_industries:
        industry_stocks = candidates[candidates['industry'] == industry]

        # 计算该行业分配数量
        available = len(industry_stocks)
        allocation = min(max_per_industry, max(min_per_industry, min(base_allocation, available)))

        # 在该行业内按数据质量和市值选择
        if allocation > 0:
            # 综合评分：数据天数(70%) + 市值排名(30%)
            industry_stocks = industry_stocks.copy()
            industry_stocks['data_score'] = industry_stocks['data_days_2025'] / industry_stocks['data_days_2025'].max()
            industry_stocks['cap_score'] = industry_stocks['avg_market_cap'].rank(pct=True)
            industry_stocks['composite_score'] = industry_stocks['data_score'] * 0.7 + industry_stocks['cap_score'] * 0.3

            selected = industry_stocks.nlargest(allocation, 'composite_score')
            selected_codes = selected['code'].tolist()

            selected_stocks.extend(selected_codes)
            industry_selections[industry] = len(selected_codes)

    print(f"\n⚖️ 行业均衡采样结果:")
    print(f"  目标股票数: {target_total}只")
    print(f"  实际选择: {len(selected_stocks)}只")
    print(f"  覆盖行业: {len(industry_selections)}个")

    # 显示各行业选择情况
    print(f"\n📊 各行业选择分布:")
    for industry, count in sorted(industry_selections.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {industry[:25]:<25}: {count:>3}只")

    # 获取最终选择的股票详情
    final_selection = candidates[candidates['code'].isin(selected_stocks)].copy()

    # 估算训练样本量
    total_samples = final_selection['data_days_2025'].sum()
    avg_days = final_selection['data_days_2025'].mean()

    print(f"\n✅ 2025年聚焦训练数据统计:")
    print(f"  选择股票数: {len(final_selection)}只")
    print(f"  平均数据天数: {avg_days:.0f}天")
    print(f"  预计训练样本: {total_samples:,}条")
    print(f"  数据质量: 基于2025年最新市场数据")

    # 验证样本充分性
    features_count = 49
    min_samples_needed = features_count * 20  # 2025年数据要求更高
    adequacy_ratio = total_samples / min_samples_needed

    print(f"\n📏 2025年聚焦策略验证:")
    print(f"  特征维度: {features_count}")
    print(f"  最小样本需求: {min_samples_needed:,}条")
    print(f"  实际样本数: {total_samples:,}条")
    print(f"  充分性比例: {adequacy_ratio:.1f}x")

    if adequacy_ratio >= 1.0:
        print(f"  ✅ 样本量充分")
    else:
        print(f"  ⚠️ 可能需要更多股票")

    # 市场代表性分析
    cap_distribution = pd.cut(final_selection['avg_market_cap'],
                            bins=5, labels=['微盘', '小盘', '中小盘', '中盘', '大盘'])
    cap_dist = cap_distribution.value_counts().sort_index()

    print(f"\n📊 市值代表性:")
    for cap_tier, count in cap_dist.items():
        print(f"  {cap_tier}: {count}只 ({count/len(final_selection)*100:.1f}%)")

    # 保存结果
    result = {
        'selected_codes': final_selection['code'].tolist(),
        'total_samples': int(total_samples),
        'avg_data_days': int(avg_days),
        'industry_coverage': len(industry_selections),
        'adequacy_ratio': adequacy_ratio
    }

    # 保存到文件
    with open('/Users/yangxu/StockTradebyZ/v380_2025_focused_stocks.txt', 'w') as f:
        f.write("# V3.80 2025年聚焦训练股票列表\n")
        f.write(f"# 基于V3.7行业均衡采样 + 2025年最新数据\n")
        f.write(f"# 总计: {len(final_selection)}只股票\n")
        f.write(f"# 预计样本: {total_samples:,}条\n")
        f.write(f"# 行业覆盖: {len(industry_selections)}个\n\n")

        for _, row in final_selection.iterrows():
            f.write(f"{row['code']} # {row['name']} - {row['industry'][:15]} - {row['data_days_2025']}天\n")

    print(f"\n📝 股票列表已保存到: v380_2025_focused_stocks.txt")

    return result

def design_model_optimization_limits():
    """设计模型优化极限和回测策略"""
    print(f"\n🎯 设计模型优化极限和回测策略")
    print("="*45)

    print(f"📈 模型优化极限目标:")
    print(f"  1. 预测相关性: >0.15 (优秀) / >0.08 (良好)")
    print(f"  2. 多空策略收益: >2% (1天) / >5% (3天)")
    print(f"  3. 信息比率: >0.5")
    print(f"  4. 最大回撤: <15%")
    print(f"  5. 胜率: >55%")

    print(f"\n🧪 回测验证策略:")
    print(f"  📊 样本外测试: 2025年9月数据 (最新)")
    print(f"  🔄 交叉验证: 时间序列分割")
    print(f"  📈 预测期间: 1天、3天、5天收益")
    print(f"  🎯 评估指标: IC、RankIC、多空收益")
    print(f"  ⚖️ 风险控制: 行业中性、市值中性")

    return {
        'target_ic': 0.15,
        'target_return_1d': 0.02,
        'target_return_3d': 0.05,
        'target_sharpe': 0.5,
        'max_drawdown': 0.15,
        'win_rate': 0.55
    }

if __name__ == "__main__":
    print("🚀 2025年聚焦V3.80训练策略设计")
    print("="*60)

    # 第1步：分析2025年数据
    market_data = analyze_2025_market_data()

    if market_data and market_data['months'] >= 3.0:
        # 第2步：行业均衡采样
        sampling_result = intelligent_industry_sampling_2025()

        # 第3步：设计优化目标
        optimization_targets = design_model_optimization_limits()

        if sampling_result:
            print(f"\n🎊 2025年聚焦训练策略设计完成!")
            print(f"📊 股票规模: {len(sampling_result['selected_codes'])}只")
            print(f"📊 样本规模: {sampling_result['total_samples']:,}条")
            print(f"📊 行业覆盖: {sampling_result['industry_coverage']}个")
            print(f"📊 数据时效: 2025年最新4个月")

            improvement_vs_old = sampling_result['total_samples'] / 6457
            print(f"\n🚀 相比原始策略改进:")
            print(f"  📈 样本量提升: {improvement_vs_old:.1f}倍")
            print(f"  🎯 时效性: 最新2025年数据")
            print(f"  ⚖️ 行业均衡: V3.7验证策略")
            print(f"  🎯 优化目标: 明确可达成")

        else:
            print(f"\n❌ 2025年数据不足，建议扩展时间范围")
    else:
        print(f"\n❌ 2025年数据时间跨度不足，需要至少3个月数据")