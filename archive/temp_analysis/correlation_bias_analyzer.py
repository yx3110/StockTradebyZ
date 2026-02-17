#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相关性分析偏差检测器
对比去重前后的相关性系数，验证重叠案例对相关性分析的影响
"""

import pandas as pd
import numpy as np
import json
from scipy import stats
from debiased_case_selector import DebiasedCaseSelector
from datetime import datetime

class CorrelationBiasAnalyzer:
    """相关性分析偏差检测器"""
    
    def __init__(self):
        self.selector = DebiasedCaseSelector(overlap_window=20)
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """加载数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'raw_data' in data:
            df = pd.DataFrame(data['raw_data'])
        else:
            print("⚠️ 使用备选数据结构")
            # 尝试其他可能的数据结构
            return None
            
        return df
    
    def calculate_correlation_with_overlap(self, df: pd.DataFrame) -> dict:
        """计算包含重叠案例的相关性"""
        correlations = {}
        periods = [1, 3, 5, 10, 20, 30]
        
        df_clean = df.dropna(subset=['quantitative_score'])
        
        for period in periods:
            return_col = f'return_{period}d'
            if return_col in df_clean.columns:
                df_period = df_clean.dropna(subset=[return_col])
                if len(df_period) > 10:
                    corr, p_value = stats.pearsonr(
                        df_period['quantitative_score'], 
                        df_period[return_col]
                    )
                    correlations[f'{period}d'] = {
                        'correlation': corr,
                        'p_value': p_value,
                        'sample_size': len(df_period)
                    }
        
        return correlations
    
    def calculate_correlation_debiased(self, df: pd.DataFrame) -> dict:
        """计算去重后的相关性"""
        # 使用去重逻辑创建独立样本数据集
        correlations = {}
        periods = [1, 3, 5, 10, 20, 30]
        
        # 为每个时间段创建去重数据集
        for period in periods:
            return_col = f'return_{period}d'
            if return_col in df.columns:
                df_valid = df.dropna(subset=['quantitative_score', return_col])
                
                if len(df_valid) > 10:
                    # 应用去重逻辑
                    debiased_data = self._create_debiased_dataset(df_valid, period)
                    
                    if len(debiased_data) > 10:
                        scores = [case['quantitative_score'] for case in debiased_data]
                        returns = [case[return_col] for case in debiased_data]
                        
                        corr, p_value = stats.pearsonr(scores, returns)
                        correlations[f'{period}d'] = {
                            'correlation': corr,
                            'p_value': p_value,
                            'sample_size': len(debiased_data),
                            'removed_overlaps': len(df_valid) - len(debiased_data)
                        }
        
        return correlations
    
    def _create_debiased_dataset(self, df: pd.DataFrame, period: int) -> list:
        """为特定时间段创建去重数据集"""
        # 按收益排序以保持代表性
        df_sorted = df.sort_values(f'return_{period}d', key=abs, ascending=False)
        
        selected_cases = []
        used_periods = {}  # {stock_code: [date_ranges]}
        
        df_sorted['report_date'] = pd.to_datetime(df_sorted['report_date'], format='%Y%m%d')
        
        for _, row in df_sorted.iterrows():
            stock_code = row['stock_code']
            case_date = row['report_date']
            
            # 检查是否与已选案例重叠
            if not self._is_overlapping_period(stock_code, case_date, used_periods, period):
                selected_cases.append(row.to_dict())
                
                # 记录已使用的时间段
                if stock_code not in used_periods:
                    used_periods[stock_code] = []
                
                from datetime import timedelta
                period_start = case_date
                period_end = case_date + timedelta(days=period)
                used_periods[stock_code].append((period_start, period_end))
        
        return selected_cases
    
    def _is_overlapping_period(self, stock_code: str, case_date, used_periods: dict, period: int) -> bool:
        """检查是否与已选案例的收益计算期重叠"""
        if stock_code not in used_periods:
            return False
        
        from datetime import timedelta
        new_period_start = case_date
        new_period_end = case_date + timedelta(days=period)
        
        for period_start, period_end in used_periods[stock_code]:
            # 检查收益计算期是否重叠
            if not (new_period_end <= period_start or new_period_start >= period_end):
                return True
                
        return False
    
    def compare_correlations(self, file_path: str) -> dict:
        """对比重叠与去重后的相关性"""
        print("🔍 分析重叠案例对相关性的影响...")
        
        df = self.load_data(file_path)
        if df is None:
            print("❌ 数据加载失败")
            return {}
        
        print(f"📊 数据概览: {len(df)}个案例")
        
        # 计算原始相关性（含重叠）
        print("📈 计算含重叠的相关性...")
        corr_with_overlap = self.calculate_correlation_with_overlap(df)
        
        # 计算去重后相关性
        print("📉 计算去重后的相关性...")  
        corr_debiased = self.calculate_correlation_debiased(df)
        
        # 对比分析
        comparison = {
            'with_overlap': corr_with_overlap,
            'debiased': corr_debiased,
            'bias_analysis': {}
        }
        
        print("\n" + "="*60)
        print("📊 相关性对比分析结果")
        print("="*60)
        print(f"{'时间段':<8} {'原始相关性':<12} {'去重相关性':<12} {'偏差':<10} {'影响':<10}")
        print("-"*60)
        
        for period in ['1d', '3d', '5d', '10d', '20d', '30d']:
            if period in corr_with_overlap and period in corr_debiased:
                orig = corr_with_overlap[period]['correlation']
                debiased = corr_debiased[period]['correlation']
                bias = orig - debiased
                bias_pct = (bias / abs(debiased)) * 100 if debiased != 0 else float('inf')
                
                impact = "高" if abs(bias_pct) > 10 else "中" if abs(bias_pct) > 5 else "低"
                
                print(f"{period:<8} {orig:<12.4f} {debiased:<12.4f} {bias:<10.4f} {impact:<10}")
                
                comparison['bias_analysis'][period] = {
                    'original_correlation': orig,
                    'debiased_correlation': debiased,
                    'absolute_bias': bias,
                    'relative_bias_percent': bias_pct,
                    'original_sample_size': corr_with_overlap[period]['sample_size'],
                    'debiased_sample_size': corr_debiased[period]['sample_size'],
                    'removed_overlaps': corr_debiased[period]['removed_overlaps']
                }
        
        return comparison
    
    def generate_bias_report(self, comparison: dict, output_path: str):
        """生成偏差分析报告"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 相关性分析偏差检测报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 📊 核心发现\n\n")
            f.write("本分析检测重叠案例对量化评分与股价收益相关性分析的影响。\n\n")
            
            f.write("## 🔍 对比结果\n\n")
            f.write("| 时间段 | 原始相关性 | 去重相关性 | 绝对偏差 | 相对偏差 | 原始样本 | 去重样本 | 移除重叠 |\n")
            f.write("|--------|------------|------------|----------|----------|----------|----------|----------|\n")
            
            for period, data in comparison['bias_analysis'].items():
                f.write(f"| {period} | {data['original_correlation']:.4f} | {data['debiased_correlation']:.4f} | ")
                f.write(f"{data['absolute_bias']:.4f} | {data['relative_bias_percent']:.1f}% | ")
                f.write(f"{data['original_sample_size']} | {data['debiased_sample_size']} | {data['removed_overlaps']} |\n")
            
            f.write("\n## 💡 偏差分析\n\n")
            
            high_bias_periods = []
            for period, data in comparison['bias_analysis'].items():
                if abs(data['relative_bias_percent']) > 10:
                    high_bias_periods.append((period, data['relative_bias_percent']))
            
            if high_bias_periods:
                f.write("⚠️ **发现显著偏差的时间段:**\n")
                for period, bias_pct in high_bias_periods:
                    f.write(f"- {period}: 相对偏差 {bias_pct:.1f}%\n")
            else:
                f.write("✅ **未发现显著偏差** (所有时间段相对偏差 < 10%)\n")
            
            f.write("\n## 🎯 建议\n\n")
            if high_bias_periods:
                f.write("1. **建议使用去重后的相关性分析结果**\n")
                f.write("2. 重叠案例确实对相关性分析产生了显著影响\n")
                f.write("3. 后续分析应采用去重数据集确保统计独立性\n")
            else:
                f.write("1. 重叠案例对相关性分析影响较小\n")
                f.write("2. 但仍建议使用去重方法确保统计严谨性\n")
            
            f.write("\n---\n🤖 Generated with Claude Code\n")
        
        print(f"📄 偏差分析报告已保存: {output_path}")

def main():
    """主函数"""
    analyzer = CorrelationBiasAnalyzer()
    
    # 使用最新的数据文件
    data_file = "reports/correlation_analysis/量化评分原始数据_v3_debiased_20250815_0040.json"
    
    # 执行对比分析
    comparison = analyzer.compare_correlations(data_file)
    
    if comparison:
        # 生成偏差报告
        output_path = f"reports/correlation_analysis/相关性偏差分析_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        analyzer.generate_bias_report(comparison, output_path)
    
    print("\n✅ 相关性偏差分析完成！")

if __name__ == "__main__":
    main()