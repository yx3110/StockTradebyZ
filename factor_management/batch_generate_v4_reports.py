#!/usr/bin/env python3
"""
并行批量生成V4选股报告
时间范围：2025-05-14 到 2025-08-19
然后调用相关性分析工具分析评分质量
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
# 移除多进程导入以避免子进程使用未修复的代码
# from concurrent.futures import ProcessPoolExecutor, as_completed
# import multiprocessing
import logging
from pathlib import Path
import time

# 添加项目根目录到路径
sys.path.append('..')
sys.path.append('../factor_management')

from v4_daily_report_v3_style import V4DailyReportV3Style

class BatchV4ReportGenerator:
    """批量V4报告生成器"""
    
    def __init__(self, max_workers=None):
        # 🔧 不再使用多进程，改为顺序生成以确保使用修复后的代码
        self.logger = self._setup_logger()
        
        # 确保输出目录存在
        self.output_dir = Path("../reports/daily_selection_v4")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"🚀 批量V4报告生成器初始化完成，顺序生成模式（避免多进程代码问题）")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger("BatchV4ReportGenerator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def generate_date_range(self, start_date: str, end_date: str) -> list:
        """生成日期范围"""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
        
        return dates
    
    def generate_single_report(self, trade_date: str) -> dict:
        """生成单个日期的V4报告"""
        try:
            # 检查是否已存在报告
            analysis_date = datetime.strptime(trade_date, '%Y-%m-%d').strftime('%Y%m%d')
            report_filename = f"V4选股分析报告_{analysis_date}.md"
            report_path = self.output_dir / report_filename
            
            if report_path.exists():
                return {
                    'trade_date': trade_date,
                    'status': 'skipped',
                    'message': f'报告已存在: {report_filename}',
                    'report_path': str(report_path)
                }
            
            # 生成新报告
            generator = V4DailyReportV3Style()
            
            # 修改生成器使其使用指定日期
            result_path = generator.generate_v3_style_report(trade_date)
            
            return {
                'trade_date': trade_date,
                'status': 'success',
                'message': f'报告生成成功: {report_filename}',
                'report_path': result_path
            }
            
        except Exception as e:
            return {
                'trade_date': trade_date,
                'status': 'error',
                'message': f'生成失败: {str(e)}',
                'report_path': None
            }
    
    def batch_generate_reports(self, start_date: str, end_date: str) -> dict:
        """批量生成报告"""
        self.logger.info(f"📅 开始批量生成V4报告: {start_date} 到 {end_date}")
        
        # 生成日期列表
        dates = self.generate_date_range(start_date, end_date)
        total_dates = len(dates)
        
        self.logger.info(f"📊 总共需要处理 {total_dates} 个日期")
        
        # 🔧 改为顺序生成避免多进程问题（使用修复后的代码）
        results = {
            'success': [],
            'skipped': [],
            'error': [],
            'total': total_dates
        }
        
        start_time = time.time()
        
        # 顺序处理每个日期
        for i, date in enumerate(dates):
            result = self.generate_single_report(date)
            
            if result['status'] == 'success':
                results['success'].append(result)
                self.logger.info(f"✅ {result['trade_date']}: {result['message']}")
            elif result['status'] == 'skipped':
                results['skipped'].append(result)
                self.logger.info(f"⏭️  {result['trade_date']}: {result['message']}")
            else:  # error
                results['error'].append(result)
                self.logger.error(f"❌ {result['trade_date']}: {result['message']}")
            
            completed = i + 1
            progress = (completed / total_dates) * 100
            self.logger.info(f"📈 进度: {completed}/{total_dates} ({progress:.1f}%)")
        
        elapsed_time = time.time() - start_time
        
        # 生成总结
        summary = {
            'total_dates': total_dates,
            'success_count': len(results['success']),
            'skipped_count': len(results['skipped']),
            'error_count': len(results['error']),
            'elapsed_time': elapsed_time,
            'average_time': elapsed_time / total_dates if total_dates > 0 else 0
        }
        
        self.logger.info(f"""
🎉 批量生成完成！
📊 总结统计:
   - 总日期数: {summary['total_dates']}
   - 成功生成: {summary['success_count']}
   - 跳过已存在: {summary['skipped_count']} 
   - 生成失败: {summary['error_count']}
   - 总耗时: {summary['elapsed_time']:.1f}秒
   - 平均耗时: {summary['average_time']:.1f}秒/报告
        """)
        
        return {
            'results': results,
            'summary': summary
        }
    


def main():
    """主函数"""
    
    # 设置参数 - 202508月份到最新交易日
    start_date = "2025-08-01"
    end_date = "2025-08-22"
    
    print(f"""
🚀 V4报告批量生成工具
📅 时间范围: {start_date} 到 {end_date}
🔧 生成模式: 顺序生成（已修复硬编码因子问题）
📁 输出目录: reports/daily_selection_v4/
    """)
    
    # 初始化生成器
    generator = BatchV4ReportGenerator()
    
    try:
        # 批量生成报告
        print("=" * 60)
        print("🎯 开始批量生成V4报告")
        print("=" * 60)
        
        batch_result = generator.batch_generate_reports(start_date, end_date)
        
        if batch_result['summary']['error_count'] > 0:
            print(f"⚠️  有 {batch_result['summary']['error_count']} 个报告生成失败")
            for error_result in batch_result['results']['error']:
                print(f"   ❌ {error_result['trade_date']}: {error_result['message']}")
        
        print("=" * 60)
        print("🎉 批量生成完成！")
        print("=" * 60)
        
        print(f"""
📊 生成统计:
   - 总日期数: {batch_result['summary']['total_dates']}
   - 成功生成: {batch_result['summary']['success_count']}
   - 跳过已存在: {batch_result['summary']['skipped_count']}
   - 生成失败: {batch_result['summary']['error_count']}
   - 总耗时: {batch_result['summary']['elapsed_time']:.1f}秒
   - 平均耗时: {batch_result['summary']['average_time']:.1f}秒/报告

📁 报告输出目录: reports/daily_selection_v4/

💡 下一步: 运行以下命令进行评分质量分析
   cd ..
   python3 analyze_quantitative_scoring_correlation.py --report-dir reports/daily_selection_v4 --version v4
        """)
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断操作")
        return 1
    except Exception as e:
        print(f"❌ 批量生成过程出错: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())