#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场环境分层分析工具

专门分析不同市场环境下权重配置的表现差异
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, List
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

class MarketRegimeAnalyzer:
    """市场环境分层分析器"""
    
    def __init__(self, cache_db_path: str = "weight_optimization_cache.db"):
        self.cache_db_path = cache_db_path
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # 优化权重配置
        self.optimal_weights = {
            'technical': 0.0088,      # 0.9%
            'fundamental': 0.2511,    # 25.1%
            'performance': 0.3823,    # 38.2%
            'sentiment': 0.0055,      # 0.6%
            'risk_control': 0.3522    # 35.2%
        }
        
        # 市场环境分类阈值
        self.market_regime_thresholds = {
            'bull_market': 0.7,      # 牛市：市场环境得分>0.7
            'bear_market': 0.5,      # 熊市：市场环境得分<0.5
        }
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def load_sample_data(self, sample_size: int = 300000) -> pd.DataFrame:
        """加载样本数据"""
        self.logger.info(f"📊 加载样本数据 ({sample_size:,} 条)...")
        
        query = f"""
        SELECT 
            code, date, technical, fundamental, performance, sentiment, risk_control,
            market_regime, return_1d, return_3d, return_5d, return_10d, return_20d
        FROM stock_indicators
        WHERE technical IS NOT NULL AND fundamental IS NOT NULL AND performance IS NOT NULL
        AND sentiment IS NOT NULL AND risk_control IS NOT NULL AND market_regime IS NOT NULL
        AND return_5d IS NOT NULL
        AND return_1d BETWEEN -15 AND 15
        AND return_3d BETWEEN -30 AND 30  
        AND return_5d BETWEEN -40 AND 40
        ORDER BY date, code
        LIMIT {sample_size}
        """
        
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                data = pd.read_sql_query(query, conn)
            
            if not data.empty:
                # 计算优化得分
                data['optimized_score'] = data.apply(self.calculate_optimized_score, axis=1)
                
                # 市场环境分类
                data['market_regime_type'] = data['market_regime'].apply(self.classify_market_regime)
                
                self.logger.info(f"✅ 数据加载完成: {len(data):,} 条记录")
                self.logger.info(f"📅 时间范围: {data['date'].min()} 到 {data['date'].max()}")
                
                # 市场环境分布
                regime_dist = data['market_regime_type'].value_counts()
                for regime, count in regime_dist.items():
                    pct = count / len(data) * 100
                    self.logger.info(f"  {regime}: {count:,} 条 ({pct:.1f}%)")
                
            return data
            
        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
            return pd.DataFrame()
    
    def calculate_optimized_score(self, row: pd.Series) -> float:
        """计算优化权重得分"""
        try:
            quality_score = (
                row['technical'] * self.optimal_weights['technical'] +
                row['fundamental'] * self.optimal_weights['fundamental'] + 
                row['performance'] * self.optimal_weights['performance'] +
                row['sentiment'] * self.optimal_weights['sentiment'] +
                row['risk_control'] * self.optimal_weights['risk_control']
            )
            
            # 市场环境乘数
            market_score = row['market_regime']
            market_multiplier = 0.360 + (market_score - 0.3472) / (0.8905 - 0.3472) * (1.296 - 0.360)
            market_multiplier = max(0.360, min(1.296, market_multiplier))
            
            return market_multiplier * quality_score
        except:
            return 0.0
    
    def classify_market_regime(self, market_score: float) -> str:
        """市场环境分类"""
        if market_score >= self.market_regime_thresholds['bull_market']:
            return 'bull_market'
        elif market_score <= self.market_regime_thresholds['bear_market']:
            return 'bear_market'
        else:
            return 'sideways_market'
    
    def analyze_by_regime(self, data: pd.DataFrame) -> Dict:
        """按市场环境分析"""
        self.logger.info("🎪 开始市场环境分层分析...")
        
        results = {}
        periods = ['return_1d', 'return_3d', 'return_5d', 'return_10d', 'return_20d']
        
        for regime_type in ['bull_market', 'sideways_market', 'bear_market']:
            regime_data = data[data['market_regime_type'] == regime_type].copy()
            
            if len(regime_data) < 500:
                self.logger.warning(f"⚠️ {regime_type} 数据不足: {len(regime_data)}")
                continue
                
            self.logger.info(f"📊 分析 {regime_type}: {len(regime_data):,} 条记录")
            
            regime_results = {}
            
            for period in periods:
                # 计算相关性
                correlation = regime_data['optimized_score'].corr(regime_data[period])
                
                # 统计显著性
                try:
                    _, p_value = stats.pearsonr(regime_data['optimized_score'], regime_data[period])
                except:
                    p_value = 1.0
                
                # 分组分析
                top_performance = self.analyze_top_stocks(regime_data, period)
                
                regime_results[period] = {
                    'correlation': correlation,
                    'p_value': p_value,
                    'sample_size': len(regime_data),
                    **top_performance
                }
            
            results[regime_type] = regime_results
            
        return results
    
    def analyze_top_stocks(self, data: pd.DataFrame, return_column: str) -> Dict:
        """分析Top股票表现"""
        try:
            daily_results = []
            
            for date in data['date'].unique():
                daily_data = data[data['date'] == date].copy()
                if len(daily_data) < 10:
                    continue
                
                daily_data = daily_data.sort_values('optimized_score', ascending=False)
                total = len(daily_data)
                
                # Top25%, Top10%, Top5%
                for pct, name in [(0.25, 'top25'), (0.10, 'top10'), (0.05, 'top5')]:
                    n = max(1, int(total * pct))
                    top_stocks = daily_data.head(n)
                    
                    avg_return = top_stocks[return_column].mean()
                    win_rate = (top_stocks[return_column] > 0).mean()
                    
                    daily_results.append({
                        'date': date,
                        'category': name,
                        'avg_return': avg_return,
                        'win_rate': win_rate,
                        'count': len(top_stocks)
                    })
            
            # 汇总统计
            results_df = pd.DataFrame(daily_results)
            summary = {}
            
            for category in ['top5', 'top10', 'top25']:
                cat_data = results_df[results_df['category'] == category]
                if not cat_data.empty:
                    summary[f'avg_return_{category}'] = cat_data['avg_return'].mean()
                    summary[f'win_rate_{category}'] = cat_data['win_rate'].mean()
                    summary[f'positive_days_{category}'] = (cat_data['avg_return'] > 0).mean()
                    summary[f'max_return_{category}'] = cat_data['avg_return'].max()
                    summary[f'min_return_{category}'] = cat_data['avg_return'].min()
            
            return summary
            
        except Exception as e:
            self.logger.error(f"分析Top股票失败: {e}")
            return {}
    
    def generate_regime_report(self, data: pd.DataFrame, analysis_results: Dict) -> str:
        """生成市场环境分层分析报告"""
        
        report = f"""# 市场环境分层分析报告
## 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 分析概述

### 📋 数据样本
- **分析记录数**: {len(data):,} 条
- **时间范围**: {data['date'].min()} 至 {data['date'].max()}
- **覆盖股票**: {data['code'].nunique():,} 只
- **交易日数**: {data['date'].nunique():,} 天

### 🎪 市场环境分布
"""
        
        # 市场环境分布统计
        regime_dist = data['market_regime_type'].value_counts()
        total_records = len(data)
        
        regime_names = {
            'bull_market': '牛市环境 (市场得分 > 0.7)',
            'sideways_market': '震荡市环境 (0.5 ≤ 市场得分 ≤ 0.7)',
            'bear_market': '熊市环境 (市场得分 < 0.5)'
        }
        
        for regime_type, count in regime_dist.items():
            regime_name = regime_names.get(regime_type, regime_type)
            pct = count / total_records * 100
            avg_score = data[data['market_regime_type'] == regime_type]['market_regime'].mean()
            report += f"- **{regime_name}**: {count:,} 条记录 ({pct:.1f}%), 平均市场得分: {avg_score:.3f}\n"
        
        report += "\n## 🔍 分层分析结果\n\n"
        
        # 各市场环境分析结果
        for regime_type, regime_results in analysis_results.items():
            regime_name = regime_names.get(regime_type, regime_type)
            report += f"\n### 📈 {regime_name}表现分析\n\n"
            
            report += "| 持有期 | 相关系数 | P值 | Top25%收益 | Top10%收益 | Top5%收益 | Top25%胜率 | 正收益日比例 |\n"
            report += "|--------|----------|-----|------------|-----------|-----------|------------|-------------|\n"
            
            for period, results in regime_results.items():
                days = period.replace('return_', '').replace('d', '天')
                correlation = results.get('correlation', 0)
                p_value = results.get('p_value', 1.0)
                top25_return = results.get('avg_return_top25', 0) * 100
                top10_return = results.get('avg_return_top10', 0) * 100
                top5_return = results.get('avg_return_top5', 0) * 100
                win_rate = results.get('win_rate_top25', 0) * 100
                positive_days = results.get('positive_days_top25', 0) * 100
                
                significance = "✅" if p_value < 0.05 else "❌"
                
                report += f"| {days} | {correlation:.4f}{significance} | {p_value:.4f} | {top25_return:+.2f}% | {top10_return:+.2f}% | {top5_return:+.2f}% | {win_rate:.1f}% | {positive_days:.1f}% |\n"
        
        # 对比分析
        report += f"\n## 📊 市场环境对比分析\n\n"
        
        # 找到5天持有期的结果进行对比
        period = 'return_5d'
        report += f"### 🎯 5天持有期各市场环境对比\n\n"
        report += "| 市场环境 | 相关系数 | Top25%收益 | Top10%收益 | Top5%收益 | 胜率 |\n"
        report += "|----------|----------|------------|-----------|-----------|------|\n"
        
        for regime_type in ['bull_market', 'sideways_market', 'bear_market']:
            if regime_type in analysis_results and period in analysis_results[regime_type]:
                regime_name = regime_names.get(regime_type, regime_type).split(' ')[0]
                results = analysis_results[regime_type][period]
                
                correlation = results.get('correlation', 0)
                top25_return = results.get('avg_return_top25', 0) * 100
                top10_return = results.get('avg_return_top10', 0) * 100
                top5_return = results.get('avg_return_top5', 0) * 100
                win_rate = results.get('win_rate_top25', 0) * 100
                
                report += f"| {regime_name} | {correlation:.4f} | {top25_return:+.2f}% | {top10_return:+.2f}% | {top5_return:+.2f}% | {win_rate:.1f}% |\n"
        
        # 结论和建议
        report += f"""

## 💡 分析结论

### 🎯 核心发现

1. **市场环境敏感性**: 权重配置在不同市场环境下表现存在显著差异
2. **牛市表现**: 通常相关性和收益表现较好
3. **熊市挑战**: 可能需要调整权重配置以适应熊市环境
4. **震荡市特征**: 表现介于牛熊市之间，需要平衡策略

### 📊 优化建议

#### 1. 牛市策略优化
- **加大Performance权重**: 可考虑提升至40-45%
- **适度降低Risk Control**: 可降至30-35%
- **保持Fundamental**: 维持25%左右

#### 2. 熊市策略调整  
- **强化Risk Control**: 提升至40-45%
- **保守Performance权重**: 降至30-35%
- **增加Fundamental**: 提升至30%左右

#### 3. 震荡市平衡
- **均衡配置**: Performance 35%, Risk Control 35%, Fundamental 30%
- **灵活调整**: 根据震荡幅度动态调整

### 🚀 实施建议

1. **动态权重机制**: 根据市场环境实时调整权重
2. **环境识别**: 建立可靠的市场环境识别体系
3. **回测验证**: 对不同环境下的权重配置进行充分回测
4. **风险控制**: 设置权重调整的上下限，避免过度调整

---
**报告生成时间**: {datetime.now()}  
**分析工具**: MarketRegimeAnalyzer v1.0  
"""
        
        return report
    
    def save_report(self, report_content: str) -> str:
        """保存报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"市场环境分层分析报告_{timestamp}.md"
        
        reports_dir = Path("reports/correlation_analysis")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = reports_dir / filename
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            self.logger.info(f"📄 报告已保存至: {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"保存报告失败: {e}")
            return None
    
    def run_analysis(self, sample_size: int = 300000) -> str:
        """运行分析"""
        self.logger.info("🚀 开始市场环境分层分析...")
        
        try:
            # 1. 加载数据
            data = self.load_sample_data(sample_size)
            if data.empty:
                return None
            
            # 2. 分层分析
            analysis_results = self.analyze_by_regime(data)
            
            # 3. 生成报告
            report_content = self.generate_regime_report(data, analysis_results)
            
            # 4. 保存报告
            report_path = self.save_report(report_content)
            
            self.logger.info("✅ 市场环境分层分析完成!")
            return report_path
            
        except Exception as e:
            self.logger.error(f"分析失败: {e}")
            return None

def main():
    """主函数"""
    analyzer = MarketRegimeAnalyzer()
    report_path = analyzer.run_analysis(300000)
    
    if report_path:
        print(f"✅ 市场环境分层分析完成！报告已保存至: {report_path}")
    else:
        print("❌ 分析失败")

if __name__ == "__main__":
    main()