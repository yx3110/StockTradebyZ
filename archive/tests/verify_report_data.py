#!/usr/bin/env python3
"""
验证报告数据脚本
检查所有报告的股票数量和独特股票总数
"""

import os
import re
from pathlib import Path
from collections import defaultdict, Counter

def extract_stocks_from_report(file_path):
    """从报告中提取股票代码"""
    stocks = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找股票评分表格
        lines = content.split('\n')
        in_table = False
        
        for line in lines:
            # 检测表格开始
            if '| 排名 |' in line and '股票代码' in line:
                in_table = True
                continue
            
            # 跳过分隔符行
            if in_table and line.startswith('|---'):
                continue
                
            # 解析股票行
            if in_table and line.startswith('| '):
                parts = [p.strip() for p in line.split('|')[1:-1]]
                if len(parts) >= 2:
                    try:
                        rank = int(parts[0])  # 验证是否为数字排名
                        stock_code = parts[1].strip()
                        if stock_code and len(stock_code) == 6 and stock_code.isdigit():
                            stocks.append(stock_code)
                    except (ValueError, IndexError):
                        continue
            
            # 检测表格结束
            if in_table and not line.startswith('|') and line.strip():
                break
                
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
    
    return stocks

def analyze_reports(report_dir):
    """分析报告目录"""
    report_dir = Path(report_dir)
    
    if not report_dir.exists():
        print(f"目录不存在: {report_dir}")
        return
    
    print(f"🔍 分析目录: {report_dir}")
    print("="*60)
    
    # 收集数据
    daily_stats = []
    all_stocks = set()
    total_selections = 0
    
    # 获取所有报告文件
    report_files = list(report_dir.glob("选股分析报告_*.md"))
    report_files.sort()
    
    print(f"📁 找到 {len(report_files)} 个报告文件\n")
    
    # 逐个分析报告
    for report_file in report_files:
        # 提取日期
        date_match = re.search(r'(\d{8})\.md$', report_file.name)
        if not date_match:
            continue
            
        report_date = date_match.group(1)
        
        # 提取股票
        stocks = extract_stocks_from_report(report_file)
        unique_stocks_in_report = len(set(stocks))
        
        # 统计
        daily_stats.append({
            'date': report_date,
            'total_selections': len(stocks),
            'unique_stocks': unique_stocks_in_report,
            'stocks': set(stocks)
        })
        
        all_stocks.update(stocks)
        total_selections += len(stocks)
        
        print(f"📊 {report_date}: {len(stocks)}只选股, {unique_stocks_in_report}只独特股票")
    
    # 汇总统计
    print("\n" + "="*60)
    print("📈 汇总统计:")
    print(f"  总报告数: {len(daily_stats)}")
    print(f"  总选股数: {total_selections:,}")
    print(f"  独特股票数: {len(all_stocks):,}")
    
    if daily_stats:
        avg_daily = total_selections / len(daily_stats)
        print(f"  平均每日选股: {avg_daily:.1f}只")
        
        # 选股数量分布
        daily_counts = [stat['total_selections'] for stat in daily_stats]
        print(f"  每日选股范围: {min(daily_counts)} - {max(daily_counts)}只")
        
        # 找出异常日期
        high_days = [stat for stat in daily_stats if stat['total_selections'] > 500]
        if high_days:
            print(f"\n⚠️  选股数量异常的日期 (>500只):")
            for day in high_days:
                print(f"    {day['date']}: {day['total_selections']}只")
    
    # 股票频次分析
    stock_frequency = Counter()
    for stat in daily_stats:
        for stock in stat['stocks']:
            stock_frequency[stock] += 1
    
    print(f"\n📋 股票选择频次:")
    print(f"  被选择1次的股票: {len([s for s, c in stock_frequency.items() if c == 1])}只")
    print(f"  被选择2-5次的股票: {len([s for s, c in stock_frequency.items() if 2 <= c <= 5])}只")
    print(f"  被选择6-10次的股票: {len([s for s, c in stock_frequency.items() if 6 <= c <= 10])}只")
    print(f"  被选择>10次的股票: {len([s for s, c in stock_frequency.items() if c > 10])}只")
    
    # 最频繁的股票
    most_frequent = stock_frequency.most_common(10)
    if most_frequent:
        print(f"\n🔥 最频繁被选择的股票 (Top 10):")
        for stock, count in most_frequent:
            print(f"    {stock}: {count}次")
    
    return {
        'total_reports': len(daily_stats),
        'total_selections': total_selections,
        'unique_stocks': len(all_stocks),
        'daily_stats': daily_stats,
        'stock_frequency': stock_frequency
    }

def main():
    """主函数"""
    print("🚀 股票选择报告数据验证")
    print("="*60)
    
    # 分析v2.0报告
    v2_results = analyze_reports('reports/daily_selection_v2')
    
    print("\n" + "="*60)
    print("✅ 验证完成!")
    
    # 判断数据是否合理
    if v2_results:
        unique_count = v2_results['unique_stocks']
        total_count = v2_results['total_selections']
        
        print(f"\n🎯 结论:")
        if unique_count > 2000:
            print(f"  ✅ 独特股票数 {unique_count:,} 确实超过2000只")
        else:
            print(f"  ❌ 独特股票数 {unique_count:,} 未超过2000只")
            
        if total_count > 8000:
            print(f"  ⚠️  总选股数 {total_count:,} 异常偏高")
        else:
            print(f"  ✅ 总选股数 {total_count:,} 在合理范围内")

if __name__ == "__main__":
    main()