#!/usr/bin/env python3
"""
验证历史数据完整性和质量
检查下载的历史数据是否符合回测需求
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
import json

# 可选的matplotlib导入
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# 确保logs目录存在
import os
os.makedirs("../logs", exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("../logs/data_validation.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

def analyze_data_coverage(data_dir="full_securities_data", start_date="2018-01-01"):
    """分析数据覆盖情况"""
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*.csv"))
    
    # 排除非股票文件
    stock_files = [f for f in csv_files if not f.name.startswith("securities_list")]
    
    logger.info(f"发现 {len(stock_files)} 个数据文件")
    
    results = []
    target_start = pd.to_datetime(start_date)
    current_date = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
    
    for i, csv_file in enumerate(stock_files):
        if i % 1000 == 0:
            logger.info(f"已分析 {i}/{len(stock_files)} 个文件...")
        
        try:
            df = pd.read_csv(csv_file, parse_dates=['date'])
            if df.empty:
                continue
            
            stock_code = csv_file.stem.split('_')[0]
            earliest = df['date'].min()
            latest = df['date'].max()
            record_count = len(df)
            
            # 计算数据质量指标
            missing_days = (latest - earliest).days - record_count + 1
            coverage_ratio = record_count / max(1, (latest - earliest).days + 1)
            
            # 检查数据是否符合回测需求
            has_early_data = earliest <= target_start + timedelta(days=30)  # 允许30天误差
            is_recent = latest >= current_date - timedelta(days=7)  # 一周内的数据
            
            results.append({
                'stock_code': stock_code,
                'earliest_date': earliest,
                'latest_date': latest,
                'record_count': record_count,
                'missing_days': missing_days,
                'coverage_ratio': coverage_ratio,
                'has_early_data': has_early_data,
                'is_recent': is_recent,
                'file_path': str(csv_file)
            })
            
        except Exception as e:
            logger.warning(f"分析 {csv_file} 失败: {e}")
            continue
    
    return pd.DataFrame(results)

def generate_data_report(analysis_df):
    """生成数据质量报告"""
    if analysis_df.empty:
        return "无数据可分析"
    
    total_stocks = len(analysis_df)
    early_data_count = analysis_df['has_early_data'].sum()
    recent_data_count = analysis_df['is_recent'].sum()
    
    # 数据完整性统计
    avg_coverage = analysis_df['coverage_ratio'].mean()
    min_date = analysis_df['earliest_date'].min()
    max_date = analysis_df['latest_date'].max()
    total_records = analysis_df['record_count'].sum()
    
    # 按年份统计数据分布
    analysis_df['start_year'] = analysis_df['earliest_date'].dt.year
    year_distribution = analysis_df['start_year'].value_counts().sort_index()
    
    report = f"""
# 📊 历史数据质量报告

## 数据概览
- **总股票数量**: {total_stocks:,} 只
- **数据时间范围**: {min_date.strftime('%Y-%m-%d')} 至 {max_date.strftime('%Y-%m-%d')}
- **总数据记录**: {total_records:,} 条
- **平均数据覆盖率**: {avg_coverage:.2%}

## 回测数据准备情况
- **有早期数据(2018年左右)**: {early_data_count} 只 ({early_data_count/total_stocks:.1%})
- **有最新数据(一周内)**: {recent_data_count} 只 ({recent_data_count/total_stocks:.1%})
- **可用于回测的股票**: {min(early_data_count, recent_data_count)} 只

## 数据起始年份分布
"""
    
    for year, count in year_distribution.head(10).items():
        report += f"- **{year}年**: {count} 只股票\n"
    
    # 数据质量分级
    high_quality = analysis_df[
        (analysis_df['has_early_data']) & 
        (analysis_df['is_recent']) & 
        (analysis_df['coverage_ratio'] > 0.9)
    ]
    
    medium_quality = analysis_df[
        (analysis_df['has_early_data']) & 
        (analysis_df['is_recent']) & 
        (analysis_df['coverage_ratio'] > 0.7)
    ]
    
    report += f"""
## 数据质量分级
- **高质量数据** (覆盖率>90%, 有早期+最新数据): {len(high_quality)} 只
- **中等质量数据** (覆盖率>70%, 有早期+最新数据): {len(medium_quality)} 只

## 🎯 回测建议
- **推荐用于回测**: {len(high_quality)} 只高质量股票
- **数据时间跨度**: {(max_date - min_date).days} 天 ({(max_date - min_date).days / 365:.1f} 年)
- **适合策略类型**: 中长期策略, 多因子策略

## ⚠️ 数据缺失情况
"""
    
    # 找出数据缺失较多的股票
    poor_coverage = analysis_df[analysis_df['coverage_ratio'] < 0.5]
    if not poor_coverage.empty:
        report += f"- **数据缺失严重** (覆盖率<50%): {len(poor_coverage)} 只\n"
        for _, row in poor_coverage.head(5).iterrows():
            report += f"  - {row['stock_code']}: 覆盖率 {row['coverage_ratio']:.1%}\n"
    
    report += f"""
---
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    return report

def save_validation_results(analysis_df, report, output_dir="backtest/data_validation"):
    """保存验证结果"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存详细分析数据
    analysis_file = output_path / f"data_analysis_{timestamp}.csv"
    analysis_df.to_csv(analysis_file, index=False, encoding='utf-8')
    
    # 保存报告
    report_file = output_path / f"data_quality_report_{timestamp}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 保存高质量股票列表（用于回测）
    high_quality_stocks = analysis_df[
        (analysis_df['has_early_data']) & 
        (analysis_df['is_recent']) & 
        (analysis_df['coverage_ratio'] > 0.9)
    ]['stock_code'].tolist()
    
    stocks_file = output_path / f"high_quality_stocks_{timestamp}.json"
    with open(stocks_file, 'w', encoding='utf-8') as f:
        json.dump({
            'stocks': high_quality_stocks,
            'count': len(high_quality_stocks),
            'generated_at': datetime.now().isoformat(),
            'criteria': 'coverage_ratio > 0.9, has_early_data, is_recent'
        }, f, ensure_ascii=False, indent=2)
    
    logger.info(f"验证结果已保存到: {output_path}")
    return output_path

def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("开始验证历史数据质量")
    logger.info("=" * 60)
    
    # 分析数据覆盖情况
    logger.info("1. 分析数据覆盖情况...")
    analysis_df = analyze_data_coverage()
    
    if analysis_df.empty:
        logger.error("未找到有效的数据文件")
        return
    
    # 生成质量报告
    logger.info("2. 生成数据质量报告...")
    report = generate_data_report(analysis_df)
    
    # 保存结果
    logger.info("3. 保存验证结果...")
    output_path = save_validation_results(analysis_df, report)
    
    # 打印报告摘要
    print(report)
    
    logger.info("=" * 60)
    logger.info("数据验证完成")
    logger.info(f"详细结果保存在: {output_path}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()