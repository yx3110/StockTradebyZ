#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预测性能验证器 - 真正验证权重优化与未来收益的关联性
这才是判断权重优化是否有效的金标准
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import glob
from scipy import stats
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class PredictivePerformanceValidator:
    """预测性能验证器"""
    
    def __init__(self):
        self.db_path = "stock_data.db"
        
    def load_stock_returns(self, codes, start_date, end_date, periods=[1, 3, 5, 10]):
        """加载股票未来收益率数据"""
        print(f"📈 加载股票收益率数据 ({start_date} 到 {end_date})...")
        
        conn = sqlite3.connect(self.db_path)
        
        # 构建股票代码列表
        code_list = "','".join(codes)
        
        returns_data = {}
        
        for period in periods:
            query = f"""
            WITH future_prices AS (
                SELECT 
                    s.code,
                    dq1.trade_date as base_date,
                    dq1.close as base_price,
                    dq2.close as future_price,
                    (dq2.close - dq1.close) / dq1.close * 100 as return_{period}d
                FROM securities s
                LEFT JOIN daily_quotes dq1 ON s.id = dq1.security_id
                LEFT JOIN daily_quotes dq2 ON s.id = dq2.security_id 
                WHERE s.code IN ('{code_list}')
                AND dq1.trade_date >= '{start_date}'
                AND dq1.trade_date <= '{end_date}'
                AND dq2.trade_date = (
                    SELECT trade_date 
                    FROM daily_quotes dq_sub 
                    WHERE dq_sub.security_id = s.id 
                    AND dq_sub.trade_date > dq1.trade_date 
                    ORDER BY dq_sub.trade_date 
                    LIMIT 1 OFFSET {period-1}
                )
            )
            SELECT * FROM future_prices WHERE return_{period}d IS NOT NULL
            """
            
            df_returns = pd.read_sql_query(query, conn)
            returns_data[f'{period}d'] = df_returns
            print(f"  {period}日收益率: {len(df_returns)} 条记录")
        
        conn.close()
        return returns_data
    
    def validate_predictive_power(self, original_weights, optimized_weights):
        """验证权重优化的预测能力"""
        print("🔍 开始预测能力验证...")
        print("=" * 70)
        
        # 加载所有v3.1评分数据
        v31_data = self.load_v31_scoring_data()
        
        if len(v31_data) < 1000:
            print("⚠️ 数据量不足，需要至少1000条记录进行可靠验证")
            return None
        
        print(f"📊 评分数据: {len(v31_data)} 条记录")
        print(f"📅 时间范围: {v31_data['date'].min()} 到 {v31_data['date'].max()}")
        
        # 获取唯一股票代码和日期范围
        unique_codes = v31_data['code'].unique()[:500]  # 限制股票数量加快速度
        start_date = v31_data['date'].min()
        end_date = v31_data['date'].max()
        
        # 加载收益率数据
        returns_data = self.load_stock_returns(unique_codes, start_date, end_date)
        
        # 计算原始和优化权重的评分
        original_scores = self.calculate_weighted_scores(v31_data, original_weights)
        optimized_scores = self.calculate_weighted_scores(v31_data, optimized_weights)
        
        # 验证预测能力
        results = {}
        
        for period in ['1d', '3d', '5d', '10d']:
            if period in returns_data:
                period_results = self.validate_period_performance(
                    v31_data, original_scores, optimized_scores, returns_data[period], period
                )
                results[period] = period_results
        
        return results
    
    def load_v31_scoring_data(self):
        """加载v3.1评分数据"""
        print("📊 加载v3.1评分数据...")
        
        v31_dir = "reports/daily_selection_v3.1"
        json_files = glob.glob(f"{v31_dir}/analysis_data_*.json")
        
        all_data = []
        
        for json_file in sorted(json_files)[:50]:  # 限制文件数量加快速度
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                date = os.path.basename(json_file).replace('analysis_data_', '').replace('.json', '')
                date_formatted = f"{date[:4]}-{date[4:6]}-{date[6:]}"
                
                if 'all_stocks_with_scores' in data:
                    for stock in data['all_stocks_with_scores'][:100]:  # 限制每日股票数量
                        factor_scores = stock.get('factor_scores', {})
                        
                        stock_record = {
                            'date': date_formatted,
                            'code': stock['stock_code'],
                            'total_score': stock.get('score', 0)
                        }
                        
                        # 添加各维度分数
                        for dim in ['technical', 'fundamental', 'performance', 'sentiment', 'risk_control', 'market_regime']:
                            stock_record[dim] = factor_scores.get(dim, 0)
                        
                        all_data.append(stock_record)
            
            except Exception as e:
                print(f"⚠️ 读取文件失败 {json_file}: {e}")
                continue
        
        return pd.DataFrame(all_data)
    
    def calculate_weighted_scores(self, data, weights):
        """计算加权评分"""
        # 根据v3.1的实际逻辑（直接相加）计算
        weighted_scores = np.zeros(len(data))
        
        for dim, weight in weights.items():
            if dim in data.columns:
                weighted_scores += data[dim].values * weight * 100
        
        return weighted_scores
    
    def validate_period_performance(self, score_data, original_scores, optimized_scores, returns_data, period):
        """验证特定时期的预测性能"""
        print(f"🎯 验证 {period} 预测性能...")
        
        # 合并评分和收益率数据
        score_df = score_data.copy()
        score_df['original_score'] = original_scores
        score_df['optimized_score'] = optimized_scores
        
        # 转换日期格式用于匹配
        score_df['date'] = pd.to_datetime(score_df['date'])
        returns_data['base_date'] = pd.to_datetime(returns_data['base_date'])
        
        # 合并数据
        merged_data = pd.merge(
            score_df, 
            returns_data[['code', 'base_date', f'return_{period}']],
            left_on=['code', 'date'], 
            right_on=['code', 'base_date'],
            how='inner'
        )
        
        if len(merged_data) < 50:
            print(f"⚠️ {period} 数据匹配不足: {len(merged_data)} 条")
            return None
        
        print(f"📈 {period} 有效数据: {len(merged_data)} 条")
        
        # 计算相关性
        return_col = f'return_{period}'
        
        # 原始权重相关性
        original_corr, original_p = stats.pearsonr(
            merged_data['original_score'], 
            merged_data[return_col]
        )
        
        # 优化权重相关性  
        optimized_corr, optimized_p = stats.pearsonr(
            merged_data['optimized_score'],
            merged_data[return_col]
        )
        
        # 分层验证 - 按评分分组看收益差异
        original_deciles = pd.qcut(merged_data['original_score'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        optimized_deciles = pd.qcut(merged_data['optimized_score'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        
        original_group_returns = merged_data.groupby(original_deciles)[return_col].mean()
        optimized_group_returns = merged_data.groupby(optimized_deciles)[return_col].mean()
        
        # 计算信息比率 (高分组与低分组的收益差/标准差)
        original_info_ratio = (original_group_returns['Q5'] - original_group_returns['Q1']) / merged_data[return_col].std()
        optimized_info_ratio = (optimized_group_returns['Q5'] - optimized_group_returns['Q1']) / merged_data[return_col].std()
        
        results = {
            'sample_size': len(merged_data),
            'correlations': {
                'original': {'corr': original_corr, 'p_value': original_p},
                'optimized': {'corr': optimized_corr, 'p_value': optimized_p}
            },
            'group_returns': {
                'original': original_group_returns.to_dict(),
                'optimized': optimized_group_returns.to_dict()
            },
            'information_ratio': {
                'original': original_info_ratio,
                'optimized': optimized_info_ratio
            },
            'improvement': {
                'correlation_improvement': optimized_corr - original_corr,
                'info_ratio_improvement': optimized_info_ratio - original_info_ratio,
                'is_significant': abs(optimized_corr) > abs(original_corr) and optimized_p < 0.05
            }
        }
        
        return results
    
    def print_validation_results(self, results):
        """打印验证结果"""
        print("\n" + "=" * 70)
        print("🎯 权重优化预测性能验证结果")
        print("=" * 70)
        
        for period, result in results.items():
            if result is None:
                continue
                
            print(f"\n📊 {period.upper()} 预测性能:")
            print("-" * 50)
            
            orig_corr = result['correlations']['original']['corr']
            opt_corr = result['correlations']['optimized']['corr'] 
            corr_improvement = result['improvement']['correlation_improvement']
            
            print(f"相关系数:")
            print(f"  原始权重: {orig_corr:+.4f} (p={result['correlations']['original']['p_value']:.4f})")
            print(f"  优化权重: {opt_corr:+.4f} (p={result['correlations']['optimized']['p_value']:.4f})")
            print(f"  改善程度: {corr_improvement:+.4f}")
            
            print(f"\n分组收益率 (%):")
            print(f"  {'组别':>4s} {'原始权重':>10s} {'优化权重':>10s} {'差异':>8s}")
            print("  " + "-"*35)
            
            orig_returns = result['group_returns']['original']
            opt_returns = result['group_returns']['optimized']
            
            for group in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
                orig_ret = orig_returns.get(group, 0)
                opt_ret = opt_returns.get(group, 0)
                diff = opt_ret - orig_ret
                print(f"  {group:>4s} {orig_ret:>9.2f}% {opt_ret:>9.2f}% {diff:>+7.2f}%")
            
            orig_info = result['information_ratio']['original']
            opt_info = result['information_ratio']['optimized']
            info_improvement = result['improvement']['info_ratio_improvement']
            
            print(f"\n信息比率:")
            print(f"  原始权重: {orig_info:.4f}")
            print(f"  优化权重: {opt_info:.4f}")
            print(f"  改善程度: {info_improvement:+.4f}")
            
            significance = "✅ 显著改善" if result['improvement']['is_significant'] else "❌ 改善不显著"
            print(f"\n结论: {significance}")

def main():
    """主函数"""
    validator = PredictivePerformanceValidator()
    
    # 定义权重配置
    original_weights = {
        'technical': 0.60,
        'fundamental': 0.18,
        'performance': 0.15,
        'sentiment': 0.05,
        'risk_control': 0.05,
        'market_regime': 0.05
    }
    
    optimized_weights = {
        'technical': 0.450,
        'fundamental': 0.100,
        'performance': 0.200,
        'sentiment': 0.070,
        'risk_control': 0.070,
        'market_regime': 0.070
    }
    
    print("🔬 开始权重优化的预测性能验证...")
    print(f"这将验证优化权重是否真正提升了未来收益的预测能力")
    
    # 执行验证
    results = validator.validate_predictive_power(original_weights, optimized_weights)
    
    if results:
        validator.print_validation_results(results)
    else:
        print("❌ 验证失败，数据不足或其他问题")

if __name__ == "__main__":
    main()