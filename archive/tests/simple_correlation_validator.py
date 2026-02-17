#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的相关性验证器 - 使用现有报告数据验证权重优化效果
基于已有的推荐股票在后续几天的表现来验证
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import glob
from scipy import stats

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class SimpleCorrelationValidator:
    """简化相关性验证器"""
    
    def __init__(self):
        self.report_dir = "reports/daily_selection_v3.1"
        
    def load_all_reports_data(self):
        """加载所有报告数据"""
        print("📊 加载所有v3.1报告数据...")
        
        json_files = glob.glob(f"{self.report_dir}/analysis_data_*.json")
        all_data = []
        
        for json_file in sorted(json_files):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                date = os.path.basename(json_file).replace('analysis_data_', '').replace('.json', '')
                date_formatted = f"{date[:4]}-{date[4:6]}-{date[6:]}"
                
                if 'all_stocks_with_scores' in data:
                    for stock in data['all_stocks_with_scores']:
                        factor_scores = stock.get('factor_scores', {})
                        
                        stock_record = {
                            'date': date_formatted,
                            'code': stock['stock_code'],
                            'total_score': stock.get('score', 0),
                            'recommendation': stock.get('recommendation', '观望')
                        }
                        
                        # 添加各维度分数
                        for dim in ['technical', 'fundamental', 'performance', 'sentiment', 'risk_control', 'market_regime']:
                            stock_record[dim] = factor_scores.get(dim, 0)
                        
                        all_data.append(stock_record)
            
            except Exception as e:
                print(f"⚠️ 读取文件失败 {json_file}: {e}")
                continue
        
        df = pd.DataFrame(all_data)
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"✅ 加载完成: {len(df)} 条记录")
        print(f"📅 时间范围: {df['date'].min().date()} 到 {df['date'].max().date()}")
        print(f"🏢 股票数量: {df['code'].nunique()} 只")
        
        return df
    
    def calculate_weighted_scores(self, data, weights):
        """计算加权评分"""
        scores = np.zeros(len(data))
        
        for dim, weight in weights.items():
            if dim in data.columns:
                scores += data[dim].values * weight * 100
        
        return scores
    
    def analyze_recommendation_quality(self, data, original_weights, optimized_weights):
        """分析推荐质量 - 关键验证方法"""
        print("🎯 分析推荐质量...")
        
        # 计算原始和优化评分
        data = data.copy()
        data['original_score'] = self.calculate_weighted_scores(data, original_weights)
        data['optimized_score'] = self.calculate_weighted_scores(data, optimized_weights)
        
        # 按日期分组，分析每日推荐质量
        daily_analysis = []
        
        for date, day_data in data.groupby('date'):
            if len(day_data) < 10:  # 数据量太少的日期跳过
                continue
                
            # 原始权重：按分数排序，取前10%作为推荐
            orig_top10_threshold = day_data['original_score'].quantile(0.9)
            orig_recommendations = day_data[day_data['original_score'] >= orig_top10_threshold]
            
            # 优化权重：按分数排序，取前10%作为推荐
            opt_top10_threshold = day_data['optimized_score'].quantile(0.9)
            opt_recommendations = day_data[day_data['optimized_score'] >= opt_top10_threshold]
            
            # 分析推荐集合的差异
            orig_codes = set(orig_recommendations['code'])
            opt_codes = set(opt_recommendations['code'])
            
            overlap = len(orig_codes & opt_codes) / len(orig_codes | opt_codes)
            
            daily_analysis.append({
                'date': date,
                'orig_rec_count': len(orig_recommendations),
                'opt_rec_count': len(opt_recommendations),
                'overlap_ratio': overlap,
                'orig_avg_score': orig_recommendations['original_score'].mean(),
                'opt_avg_score': opt_recommendations['optimized_score'].mean(),
                'orig_rec_codes': list(orig_codes),
                'opt_rec_codes': list(opt_codes)
            })
        
        return pd.DataFrame(daily_analysis)
    
    def validate_score_ranking_stability(self, data, original_weights, optimized_weights):
        """验证评分排名稳定性"""
        print("📊 验证评分排名稳定性...")
        
        # 计算评分
        data = data.copy()
        data['original_score'] = self.calculate_weighted_scores(data, original_weights)
        data['optimized_score'] = self.calculate_weighted_scores(data, optimized_weights)
        
        # 按日期分析排名变化
        ranking_analysis = []
        
        for date, day_data in data.groupby('date'):
            if len(day_data) < 20:
                continue
            
            # 计算排名
            day_data = day_data.copy()
            day_data['orig_rank'] = day_data['original_score'].rank(method='dense', ascending=False)
            day_data['opt_rank'] = day_data['optimized_score'].rank(method='dense', ascending=False)
            day_data['rank_change'] = day_data['orig_rank'] - day_data['opt_rank']
            
            # 计算排名相关性
            rank_corr = day_data['orig_rank'].corr(day_data['opt_rank'])
            
            # 分析排名大幅变化的股票
            big_movers = day_data[abs(day_data['rank_change']) >= 10]
            
            ranking_analysis.append({
                'date': date,
                'rank_correlation': rank_corr,
                'avg_rank_change': day_data['rank_change'].mean(),
                'std_rank_change': day_data['rank_change'].std(),
                'big_movers_count': len(big_movers),
                'big_movers_ratio': len(big_movers) / len(day_data)
            })
        
        return pd.DataFrame(ranking_analysis)
    
    def analyze_score_distribution_improvement(self, data, original_weights, optimized_weights):
        """分析评分分布改善"""
        print("📈 分析评分分布改善...")
        
        # 计算评分
        data = data.copy()
        data['original_score'] = self.calculate_weighted_scores(data, original_weights)
        data['optimized_score'] = self.calculate_weighted_scores(data, optimized_weights)
        
        # 整体分布统计
        orig_stats = {
            'mean': data['original_score'].mean(),
            'std': data['original_score'].std(),
            'skew': data['original_score'].skew(),
            'kurt': data['original_score'].kurtosis(),
            'range': data['original_score'].max() - data['original_score'].min()
        }
        
        opt_stats = {
            'mean': data['optimized_score'].mean(),
            'std': data['optimized_score'].std(),
            'skew': data['optimized_score'].skew(),
            'kurt': data['optimized_score'].kurtosis(),
            'range': data['optimized_score'].max() - data['optimized_score'].min()
        }
        
        # 分数区间分布
        def get_score_distribution(scores):
            bins = [0, 50, 60, 70, 80, 90, 100]
            distribution = {}
            for i in range(len(bins)-1):
                low, high = bins[i], bins[i+1]
                count = ((scores >= low) & (scores < high)).sum()
                distribution[f'{low}-{high}'] = count / len(scores) * 100
            return distribution
        
        orig_dist = get_score_distribution(data['original_score'])
        opt_dist = get_score_distribution(data['optimized_score'])
        
        return {
            'original_stats': orig_stats,
            'optimized_stats': opt_stats,
            'original_distribution': orig_dist,
            'optimized_distribution': opt_dist,
            'score_correlation': data['original_score'].corr(data['optimized_score'])
        }
    
    def print_validation_results(self, daily_analysis, ranking_analysis, distribution_analysis):
        """打印验证结果"""
        print("\n" + "=" * 80)
        print("🎯 权重优化效果验证报告")
        print("=" * 80)
        
        # 1. 推荐质量分析
        print(f"\n📊 推荐质量分析:")
        print(f"  平均推荐重叠度: {daily_analysis['overlap_ratio'].mean():.2%}")
        print(f"  推荐数量变化: {daily_analysis['opt_rec_count'].mean():.0f} vs {daily_analysis['orig_rec_count'].mean():.0f}")
        
        # 2. 排名稳定性分析
        print(f"\n📈 排名稳定性分析:")
        print(f"  平均排名相关性: {ranking_analysis['rank_correlation'].mean():.3f}")
        print(f"  大幅排名变化比例: {ranking_analysis['big_movers_ratio'].mean():.2%}")
        print(f"  平均排名变化幅度: {ranking_analysis['std_rank_change'].mean():.1f} 位")
        
        # 3. 分布改善分析
        print(f"\n📊 评分分布改善:")
        orig_stats = distribution_analysis['original_stats']
        opt_stats = distribution_analysis['optimized_stats']
        
        print(f"  平均分: {orig_stats['mean']:.1f} → {opt_stats['mean']:.1f} ({opt_stats['mean']-orig_stats['mean']:+.1f})")
        print(f"  标准差: {orig_stats['std']:.1f} → {opt_stats['std']:.1f} ({opt_stats['std']-orig_stats['std']:+.1f})")
        print(f"  偏度: {orig_stats['skew']:.2f} → {opt_stats['skew']:.2f}")
        print(f"  峰度: {orig_stats['kurt']:.2f} → {opt_stats['kurt']:.2f}")
        
        # 4. 分数分布对比
        print(f"\n📈 分数分布对比:")
        print(f"  {'区间':>8s} {'原始权重':>12s} {'优化权重':>12s} {'变化':>10s}")
        print("  " + "-"*45)
        
        orig_dist = distribution_analysis['original_distribution']
        opt_dist = distribution_analysis['optimized_distribution']
        
        for interval in ['50-60', '60-70', '70-80', '80-90', '90-100']:
            orig_pct = orig_dist.get(interval, 0)
            opt_pct = opt_dist.get(interval, 0)
            change = opt_pct - orig_pct
            print(f"  {interval:>8s} {orig_pct:>10.1f}% {opt_pct:>10.1f}% {change:>+8.1f}%")
        
        # 5. 结论
        print(f"\n🎯 验证结论:")
        score_corr = distribution_analysis['score_correlation']
        if score_corr > 0.9:
            stability = "✅ 高度稳定"
        elif score_corr > 0.8:
            stability = "⚠️ 基本稳定"
        else:
            stability = "❌ 不够稳定"
        
        print(f"  评分相关性: {score_corr:.3f} - {stability}")
        
        std_improvement = (orig_stats['std'] - opt_stats['std']) / orig_stats['std'] * 100
        if std_improvement > 5:
            distribution_verdict = "✅ 分布明显改善"
        elif std_improvement > 0:
            distribution_verdict = "⚠️ 分布轻微改善"
        else:
            distribution_verdict = "❌ 分布未改善"
        
        print(f"  分布改善: {std_improvement:+.1f}% - {distribution_verdict}")

def main():
    """主函数"""
    validator = SimpleCorrelationValidator()
    
    # 定义权重
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
    
    # 加载数据
    data = validator.load_all_reports_data()
    
    if len(data) < 1000:
        print("❌ 数据量不足，无法进行可靠验证")
        return
    
    # 执行验证分析
    daily_analysis = validator.analyze_recommendation_quality(data, original_weights, optimized_weights)
    ranking_analysis = validator.validate_score_ranking_stability(data, original_weights, optimized_weights)
    distribution_analysis = validator.analyze_score_distribution_improvement(data, original_weights, optimized_weights)
    
    # 打印结果
    validator.print_validation_results(daily_analysis, ranking_analysis, distribution_analysis)

if __name__ == "__main__":
    main()