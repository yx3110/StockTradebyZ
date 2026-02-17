#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
去偏案例选择器 - 避免时间段重叠的股票案例选择
解决同一只股票在重叠时间段被重复选择造成的bias问题
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Set
import logging

class DebiasedCaseSelector:
    """去偏案例选择器"""
    
    def __init__(self, overlap_window: int = 20):
        """
        初始化去偏案例选择器
        
        Args:
            overlap_window: 时间窗口大小，默认20天（因为我们关心20天收益）
        """
        self.overlap_window = overlap_window
        self.logger = logging.getLogger(__name__)
        
    def detect_overlapping_cases(self, df: pd.DataFrame, 
                                 date_col: str = 'report_date',
                                 stock_col: str = 'stock_code',
                                 return_col: str = 'return_20d') -> Dict:
        """
        检测重叠案例
        
        Args:
            df: 包含股票数据的DataFrame
            date_col: 日期列名
            stock_col: 股票代码列名  
            return_col: 收益率列名
            
        Returns:
            dict: 包含重叠案例信息的字典
        """
        # 转换日期格式
        df[date_col] = pd.to_datetime(df[date_col], format='%Y%m%d', errors='coerce')
        
        # 按股票分组检测重叠
        overlapping_cases = {}
        overlap_stats = {
            'total_cases': len(df),
            'overlapping_cases': 0,
            'overlap_groups': 0,
            'affected_stocks': set()
        }
        
        for stock_code, stock_data in df.groupby(stock_col):
            stock_data = stock_data.sort_values(date_col)
            overlapping_periods = []
            
            for i, row1 in stock_data.iterrows():
                overlaps_with = []
                date1 = row1[date_col]
                
                for j, row2 in stock_data.iterrows():
                    if i >= j:  # 避免重复检查
                        continue
                        
                    date2 = row2[date_col]
                    
                    # 检查两个日期是否在overlap_window内重叠
                    days_diff = abs((date2 - date1).days)
                    if days_diff < self.overlap_window:
                        overlaps_with.append({
                            'index': j,
                            'date': date2,
                            'days_diff': days_diff,
                            'return': row2[return_col] if return_col in row2 else None
                        })
                
                if overlaps_with:
                    overlapping_periods.append({
                        'main_case': {
                            'index': i,
                            'date': date1,
                            'return': row1[return_col] if return_col in row1 else None
                        },
                        'overlaps': overlaps_with
                    })
            
            if overlapping_periods:
                overlapping_cases[stock_code] = overlapping_periods
                overlap_stats['affected_stocks'].add(stock_code)
                overlap_stats['overlapping_cases'] += sum(len(period['overlaps']) + 1 
                                                         for period in overlapping_periods)
        
        overlap_stats['overlap_groups'] = len(overlapping_cases)
        
        return {
            'overlapping_cases': overlapping_cases,
            'statistics': overlap_stats
        }
    
    def select_best_cases(self, df: pd.DataFrame,
                         n_best: int = 10,
                         n_worst: int = 10,
                         date_col: str = 'report_date',
                         stock_col: str = 'stock_code',
                         return_col: str = 'return_20d',
                         score_col: str = 'quantitative_score') -> Dict:
        """
        选择最佳和最差案例，避免时间段重叠
        
        Args:
            df: 包含股票数据的DataFrame
            n_best: 选择的最佳案例数量
            n_worst: 选择的最差案例数量
            date_col: 日期列名
            stock_col: 股票代码列名
            return_col: 收益率列名
            score_col: 评分列名
            
        Returns:
            dict: 包含去重后的最佳和最差案例
        """
        # 转换日期格式并去除无效收益数据
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], format='%Y%m%d', errors='coerce')
        df = df.dropna(subset=[return_col])
        
        # 按收益率排序
        df_sorted_best = df.nlargest(n_best * 3, return_col)  # 取3倍数量以备选择
        df_sorted_worst = df.nsmallest(n_worst * 3, return_col)
        
        # 去重选择最佳案例
        best_cases = self._deduped_selection(df_sorted_best, n_best, 
                                           date_col, stock_col, return_col)
        
        # 去重选择最差案例  
        worst_cases = self._deduped_selection(df_sorted_worst, n_worst,
                                            date_col, stock_col, return_col)
        
        return {
            'best_cases': best_cases,
            'worst_cases': worst_cases,
            'selection_stats': {
                'best_selected': len(best_cases),
                'best_target': n_best,
                'worst_selected': len(worst_cases), 
                'worst_target': n_worst,
                'best_unique_stocks': len(set(case[stock_col] for case in best_cases)),
                'worst_unique_stocks': len(set(case[stock_col] for case in worst_cases))
            }
        }
    
    def _deduped_selection(self, df_sorted: pd.DataFrame, n_target: int,
                          date_col: str, stock_col: str, return_col: str) -> List[Dict]:
        """
        去重选择案例
        
        Args:
            df_sorted: 已排序的DataFrame
            n_target: 目标案例数量
            date_col: 日期列名
            stock_col: 股票代码列名
            return_col: 收益率列名
            
        Returns:
            list: 去重后的案例列表
        """
        selected_cases = []
        used_periods = {}  # {stock_code: [date_ranges]}
        
        for _, row in df_sorted.iterrows():
            stock_code = row[stock_col]
            case_date = row[date_col]
            
            # 检查是否与已选案例重叠
            if self._is_overlapping(stock_code, case_date, used_periods):
                continue
                
            # 添加到选中案例
            selected_cases.append(row.to_dict())
            
            # 记录已使用的时间段
            if stock_code not in used_periods:
                used_periods[stock_code] = []
            
            period_start = case_date
            period_end = case_date + timedelta(days=self.overlap_window)
            used_periods[stock_code].append((period_start, period_end))
            
            # 达到目标数量即停止
            if len(selected_cases) >= n_target:
                break
                
        return selected_cases
    
    def _is_overlapping(self, stock_code: str, case_date: datetime, 
                       used_periods: Dict[str, List[Tuple]]) -> bool:
        """
        检查案例是否与已选案例重叠
        
        Args:
            stock_code: 股票代码
            case_date: 案例日期
            used_periods: 已使用的时间段
            
        Returns:
            bool: 是否重叠
        """
        if stock_code not in used_periods:
            return False
            
        for period_start, period_end in used_periods[stock_code]:
            # 检查新案例的20天窗口是否与已有窗口重叠
            new_period_end = case_date + timedelta(days=self.overlap_window)
            
            # 时间段重叠判断
            if not (new_period_end <= period_start or case_date >= period_end):
                return True
                
        return False
    
    def generate_debiased_report(self, df: pd.DataFrame, 
                                output_path: str,
                                title: str = "去偏案例分析报告") -> None:
        """
        生成去偏分析报告
        
        Args:
            df: 原始数据
            output_path: 输出路径
            title: 报告标题
        """
        # 检测重叠案例
        overlap_analysis = self.detect_overlapping_cases(df)
        
        # 选择去重后的最佳/最差案例
        debiased_cases = self.select_best_cases(df)
        
        # 生成报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# {title}\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"去重窗口: {self.overlap_window}天\n\n")
            
            # 重叠统计
            stats = overlap_analysis['statistics']
            f.write("## 🔍 重叠案例统计\n\n")
            f.write(f"- 总案例数: {stats['total_cases']}\n")
            f.write(f"- 重叠案例数: {stats['overlapping_cases']}\n") 
            f.write(f"- 重叠比例: {stats['overlapping_cases']/stats['total_cases']*100:.1f}%\n")
            f.write(f"- 涉及股票数: {len(stats['affected_stocks'])}\n")
            f.write(f"- 重叠组数: {stats['overlap_groups']}\n\n")
            
            # 详细重叠案例
            f.write("## 📊 重叠案例详情\n\n")
            for stock_code, periods in overlap_analysis['overlapping_cases'].items():
                f.write(f"### {stock_code}\n")
                for i, period in enumerate(periods):
                    main_case = period['main_case']
                    f.write(f"**重叠组 {i+1}:**\n")
                    f.write(f"- 主案例: {main_case['date'].strftime('%Y-%m-%d')} "
                           f"(收益: {main_case.get('return', 'N/A')}%)\n")
                    for overlap in period['overlaps']:
                        f.write(f"- 重叠案例: {overlap['date'].strftime('%Y-%m-%d')} "
                               f"(相隔{overlap['days_diff']}天, 收益: {overlap.get('return', 'N/A')}%)\n")
                    f.write("\n")
            
            # 去重后的案例
            selection_stats = debiased_cases['selection_stats']
            f.write("## 🎯 去重选择结果\n\n")
            f.write(f"### 最佳案例选择\n")
            f.write(f"- 目标数量: {selection_stats['best_target']}\n")
            f.write(f"- 实际选中: {selection_stats['best_selected']}\n")
            f.write(f"- 涉及股票: {selection_stats['best_unique_stocks']}只\n\n")
            
            f.write(f"### 最差案例选择\n")
            f.write(f"- 目标数量: {selection_stats['worst_target']}\n")
            f.write(f"- 实际选中: {selection_stats['worst_selected']}\n")
            f.write(f"- 涉及股票: {selection_stats['worst_unique_stocks']}只\n\n")
            
            # 去重后的最佳案例表
            f.write("## 🏆 去重后最佳案例 (TOP 10)\n\n")
            f.write("| 日期 | 股票代码 | 股票名称 | 量化评分 | 1天收益 | 5天收益 | 10天收益 | 20天收益 |\n")
            f.write("|------|----------|----------|----------|---------|---------|----------|----------|\n")
            
            for case in debiased_cases['best_cases']:
                f.write(f"| {case.get('report_date', 'N/A')} | {case.get('stock_code', 'N/A')} | ")
                f.write(f"{case.get('stock_name', 'N/A')} | {case.get('quantitative_score', 0):.1f} | ")
                f.write(f"{case.get('return_1d', 0):.2f}% | {case.get('return_5d', 0):.2f}% | ")
                f.write(f"{case.get('return_10d', 0):.2f}% | {case.get('return_20d', 0):.2f}% |\n")
            
            # 去重后的最差案例表
            f.write("\n## 📉 去重后最差案例 (BOTTOM 10)\n\n")
            f.write("| 日期 | 股票代码 | 股票名称 | 量化评分 | 1天收益 | 5天收益 | 10天收益 | 20天收益 |\n")
            f.write("|------|----------|----------|----------|---------|---------|----------|----------|\n")
            
            for case in debiased_cases['worst_cases']:
                f.write(f"| {case.get('report_date', 'N/A')} | {case.get('stock_code', 'N/A')} | ")
                f.write(f"{case.get('stock_name', 'N/A')} | {case.get('quantitative_score', 0):.1f} | ")
                f.write(f"{case.get('return_1d', 0):.2f}% | {case.get('return_5d', 0):.2f}% | ")
                f.write(f"{case.get('return_10d', 0):.2f}% | {case.get('return_20d', 0):.2f}% |\n")
            
            f.write("\n## 💡 改进建议\n\n")
            f.write("✅ **已解决的bias问题:**\n")
            f.write("- 消除了同一股票重叠时间段的重复案例\n")
            f.write("- 确保每个案例的收益计算窗口不重叠\n")
            f.write("- 提高了案例分析的客观性和代表性\n\n")
            
            f.write("🔧 **建议的后续优化:**\n")
            f.write("1. 在报告生成时自动应用去重逻辑\n")
            f.write("2. 根据不同分析周期动态调整去重窗口\n")
            f.write("3. 考虑股票行业分布的平衡性\n")
            f.write("4. 增加时间序列分析避免时期偏差\n")
            
            f.write("\n---\n🤖 Generated with Claude Code\n")
            
        self.logger.info(f"去偏分析报告已保存至: {output_path}")


def main():
    """主函数 - 演示去偏案例选择"""
    
    # 示例：从现有的相关性分析数据中测试去偏选择
    print("🚀 启动去偏案例选择器测试...")
    
    # 创建选择器实例  
    selector = DebiasedCaseSelector(overlap_window=20)
    
    print("📊 去偏案例选择器测试完成")
    print("💡 请将此模块集成到现有的报告生成流程中")


if __name__ == "__main__":
    main()