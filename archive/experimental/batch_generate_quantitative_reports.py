#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量生成量化选股报告 - 支持v2/v3/v3.4/v3.41/v3.5/v3.6/v3.7/v4评分系统
重新生成从2025-01-01到2025-09-11的量化选股报告
支持多种评分系统：
- v2: 基于3949只股票实际表现优化
- v3: 智能动态权重版
- v3.4: 基于v3.0优化的增强版（新增ROE和营收增长）
- v3.41: 反向工程重构版（基于负相关发现的革命性改进）
- v3.5: 知行指标集成版（集成知行短期趋势线和多空线，权重20%）
- v3.6: 机器学习版（LightGBM+XGBoost双模型ensemble，非线性建模）
- v3.7: 高级机器学习版（49特征三层ensemble，5基础模型+4专家模型+Meta学习器）
- v4: 集成挤压动量指标增强版
支持并行处理提高效率
"""

import os
import subprocess
import time
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def generate_trading_dates(start_date: str, end_date: str) -> list:
    """生成交易日列表（排除周末）"""
    dates = []
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    current = start
    while current <= end:
        # 排除周末
        if current.weekday() < 5:  # 周一到周五
            dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    
    return dates

def generate_quantitative_report(date: str, output_dir: str = "reports/daily_selection", version: str = "v2") -> bool:
    """生成单个量化选股报告"""
    print(f"\n📊 生成量化选股报告 {version}: {date}")
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成报告文件路径
    date_str = date.replace('-', '')
    report_file = f"{output_dir}/选股分析报告_{date_str}.md"
    
    try:
        # 根据版本选择不同的命令
        if version == "v4":
            cmd = f"python3 tomorrow_stock_selector.py {date} --scoring-version v4"
            expected_report_file = f"reports/daily_selection_v4/选股分析报告_{date.replace('-', '')}.md"
        elif version == "v3.7":
            cmd = f"~/miniconda3/bin/python3 tomorrow_stock_selector.py {date} --scoring-version v3.7"
            expected_report_file = f"reports/daily_selection_v3.7/选股分析报告_{date.replace('-', '')}.md"
        elif version == "v3.6":
            cmd = f"~/miniconda3/bin/python3 tomorrow_stock_selector.py {date} --scoring-version v3.6"
            expected_report_file = f"reports/daily_selection_v3.6/选股分析报告_{date.replace('-', '')}.md"
        elif version == "v3.5":
            cmd = f"python3 tomorrow_stock_selector.py {date} --scoring-version v3.5"
            expected_report_file = f"reports/daily_selection_v3.5/选股分析报告_{date.replace('-', '')}.md"
        elif version == "v3.41":
            cmd = f"python3 tomorrow_stock_selector.py {date} --scoring-version v3.41"
            expected_report_file = f"reports/daily_selection_v3.41/选股分析报告_{date.replace('-', '')}.md"
        elif version == "v3.4":
            cmd = f"python3 tomorrow_stock_selector.py {date} --scoring-version v3.4"
            expected_report_file = f"reports/daily_selection_v3.4/选股分析报告_{date.replace('-', '')}.md"
        elif version == "v3.3":
            cmd = f"python3 tomorrow_stock_selector.py {date} --scoring-version v3.3"
            expected_report_file = f"reports/daily_selection_v3.3/选股分析报告_{date.replace('-', '')}.md"
        elif version == "v3.2":
            cmd = f"python3 tomorrow_stock_selector.py {date} --scoring-version v3.2"
            expected_report_file = f"reports/daily_selection_v3.2/选股分析报告_{date.replace('-', '')}.md"
        elif version == "v3.1":
            cmd = f"python3 tomorrow_stock_selector.py {date} --scoring-version v3.1"
            expected_report_file = f"reports/daily_selection_v3.1/选股分析报告_{date.replace('-', '')}.md"
        elif version == "v3":
            cmd = f"python3 tomorrow_stock_selector.py {date} --scoring-version v3"
            expected_report_file = f"reports/daily_selection_v3/选股分析报告_{date.replace('-', '')}.md"
        else:
            cmd = f"python3 tomorrow_stock_selector.py {date}"
            expected_report_file = f"reports/daily_selection/选股分析报告_{date.replace('-', '')}.md"
        
        print(f"   🔄 正在运行选股程序...")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180)
        
        # 如果输出目录与预期不同，需要移动文件
        if output_dir != os.path.dirname(expected_report_file):
            if os.path.exists(expected_report_file):
                os.makedirs(output_dir, exist_ok=True)
                target_file = f"{output_dir}/选股分析报告_{date.replace('-', '')}.md"
                os.rename(expected_report_file, target_file)
                report_file = target_file
            else:
                report_file = f"{output_dir}/选股分析报告_{date.replace('-', '')}.md"
        else:
            report_file = expected_report_file
        
        if result.returncode == 0 and os.path.exists(report_file):
            print(f"   ✅ 量化选股报告生成成功")
            return True
        else:
            # 检查stderr中的具体错误
            stderr_msg = result.stderr[:300] if result.stderr else "Unknown error"
            if "不是交易日" in stderr_msg:
                print(f"   ⏭️  跳过非交易日")
                return True  # 非交易日也算成功
            else:
                print(f"   ❌ 量化选股报告生成失败: {stderr_msg}")
                return False
                
    except subprocess.TimeoutExpired:
        print(f"   ⏰ 量化选股报告生成超时")
        return False
    except Exception as e:
        print(f"   ❌ 量化选股报告生成异常: {e}")
        return False

def main(output_dir: str = "reports/daily_selection_v2", version: str = "v2",
         start_date: str = '2025-01-01', end_date: str = '2025-09-02'):
    """主函数 - 支持v2/v3/v3.4/v3.41/v3.5/v3.6/v3.7/v4评分系统版本

    Args:
        output_dir: 报告输出目录，默认为 reports/daily_selection_v2
        version: 评分系统版本，支持v2/v3/v3.1/v3.2/v3.3/v3.4/v3.41/v3.5/v3.6/v3.7/v4
        start_date: 开始日期，格式YYYY-MM-DD
        end_date: 结束日期，格式YYYY-MM-DD
    """
    
    # 获取CPU核心数，但限制最大并行数避免过度负载
    max_workers = min(multiprocessing.cpu_count(), 6)  # 提升到6个进程加速生成
    
    # 版本说明
    version_desc = {
        "v2": "v2.0 优化评分框架",
        "v3": "v3.0 智能动态权重版",
        "v3.1": "v3.1 相关性分析优化版",
        "v3.2": "v3.2 挤压动量集成版",
        "v3.3": "v3.3 相关性深度优化版",
        "v3.4": "v3.4 基于v3.0优化增强版",
        "v3.41": "v3.41 反向工程重构版（革命性改进）",
        "v3.5": "v3.5 知行指标集成版（知行趋势线+多空线，权重20%）",
        "v3.6": "v3.6 机器学习版（LightGBM+XGBoost双模型ensemble，非线性建模）",
        "v3.7": "v3.7 高级机器学习版（49特征三层ensemble，5基础模型+4专家模型+Meta学习器）",
        "v4": "v4.0 挤压动量增强版"
    }
    
    print(f"🚀 开始批量生成量化选股报告（{version}评分系统）")
    print(f"📅 日期范围: {start_date} 到 {end_date}")
    print(f"📊 使用评分系统: {version_desc.get(version, version)}") 
    print(f"📁 输出目录: {output_dir}")
    print(f"🔧 并行进程数: {max_workers}")
    print("="*70)
    
    # 生成交易日列表
    trading_dates = generate_trading_dates(start_date, end_date)
    print(f"📊 共需生成 {len(trading_dates)} 个交易日的报告")
    
    success_count = 0
    failed_dates = []
    completed_count = 0
    
    print(f"\n🔄 开始并行处理...")
    
    # 使用进程池并行处理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_date = {executor.submit(generate_quantitative_report, date, output_dir, version): date 
                         for date in trading_dates}
        
        # 收集结果
        for future in as_completed(future_to_date):
            date = future_to_date[future]
            completed_count += 1
            
            try:
                success = future.result()
                if success:
                    success_count += 1
                    print(f"✅ [{completed_count}/{len(trading_dates)}] {date} - 成功")
                else:
                    failed_dates.append(date)
                    print(f"❌ [{completed_count}/{len(trading_dates)}] {date} - 失败")
                    
            except Exception as e:
                failed_dates.append(date)
                print(f"❌ [{completed_count}/{len(trading_dates)}] {date} - 异常: {e}")
    
    print("\n" + "="*70)
    print(f"📊 并行批量生成完成统计:")
    print(f"   ✅ 成功生成: {success_count}个")
    print(f"   ❌ 生成失败: {len(failed_dates)}个")
    if len(trading_dates) > 0:
        print(f"   📈 成功率: {success_count/len(trading_dates)*100:.1f}%")
    else:
        print(f"   ⚠️  没有交易日需要生成报告")
    
    if failed_dates:
        print(f"   失败日期: {', '.join(sorted(failed_dates))}")
    
    print(f"\n🎯 量化选股报告批量生成完成!")
    print(f"📁 报告保存位置: {output_dir}")
    print(f"📊 使用{version_desc.get(version, version)}")
    print(f"⚡ 并行处理显著提升了生成效率!")

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='批量生成量化选股报告')
    parser.add_argument('--output-dir', default="reports/daily_selection_v2", 
                       help='报告输出目录')
    parser.add_argument('--version', choices=['v2', 'v3', 'v3.1', 'v3.2', 'v3.3', 'v3.4', 'v3.41', 'v3.5', 'v3.6', 'v3.7', 'v4'], default='v2',
                       help='评分系统版本 (v2、v3、v3.1、v3.2、v3.3、v3.4、v3.41、v3.5、v3.6、v3.7 或 v4)')
    parser.add_argument('--start-date', default='2025-07-01',
                       help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2025-08-13',
                       help='结束日期 (YYYY-MM-DD)')
    
    args = parser.parse_args()
    main(args.output_dir, args.version, args.start_date, args.end_date)