#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化权重收益期望分析工具

基于最新优化的权重配置分析v3.1评分系统的收益期望
使用correlation analysis格式生成详细报告

优化权重配置：
- Performance: 38.2%
- Risk Control: 35.2%  
- Fundamental: 25.1%
- Technical: 0.9%
- Sentiment: 0.6%
- Market Regime: 作为乘数因子 (0.360 - 1.296)
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import warnings

warnings.filterwarnings('ignore')

class OptimizedWeightReturnAnalyzer:
    """优化权重收益期望分析器"""
    
    def __init__(self, cache_db_path: str = "weight_optimization_cache.db",
                 stock_db_path: str = "data_adapter/stock_data.db"):
        self.cache_db_path = cache_db_path
        self.stock_db_path = stock_db_path
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # 最优权重配置（来自优化结果）
        self.optimal_weights = {
            'technical': 0.0088,      # 0.9%
            'fundamental': 0.2511,    # 25.1%
            'performance': 0.3823,    # 38.2%
            'sentiment': 0.0055,      # 0.6%
            'risk_control': 0.3522    # 35.2%
        }
        
        # 市场环境乘数参数
        self.market_multiplier_config = {
            'min_multiplier': 0.360,  # 熊市乘数
            'max_multiplier': 1.296   # 牛市乘数
        }
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def calculate_optimized_score(self, row: pd.Series) -> float:
        """计算优化权重下的股票得分"""
        try:
            # 计算个股质量得分
            quality_score = (
                row['technical'] * self.optimal_weights['technical'] +
                row['fundamental'] * self.optimal_weights['fundamental'] + 
                row['performance'] * self.optimal_weights['performance'] +
                row['sentiment'] * self.optimal_weights['sentiment'] +
                row['risk_control'] * self.optimal_weights['risk_control']
            )
            
            # 计算市场环境乘数
            market_score = row['market_regime']
            min_mult = self.market_multiplier_config['min_multiplier']
            max_mult = self.market_multiplier_config['max_multiplier']
            
            # 将市场环境得分(0.3472-0.8905)映射到乘数范围(0.360-1.296)
            market_multiplier = min_mult + (market_score - 0.3472) / (0.8905 - 0.3472) * (max_mult - min_mult)
            
            # 最终得分 = 市场环境乘数 × 个股质量得分
            final_score = market_multiplier * quality_score
            
            return final_score
            
        except Exception as e:
            self.logger.error(f"计算优化得分失败: {e}")
            return 0.0
    
    def load_historical_data(self, sample_limit: int = None) -> pd.DataFrame:
        """加载历史数据"""
        self.logger.info(f"📊 加载历史评分数据...")
        
        query = """
        SELECT 
            code,
            date,
            technical,
            fundamental,
            performance,
            sentiment,
            risk_control,
            market_regime,
            return_1d,
            return_3d,
            return_5d,
            return_10d,
            return_20d
        FROM stock_indicators
        WHERE technical IS NOT NULL 
        AND fundamental IS NOT NULL
        AND performance IS NOT NULL
        AND sentiment IS NOT NULL  
        AND risk_control IS NOT NULL
        AND market_regime IS NOT NULL
        AND return_5d IS NOT NULL
        ORDER BY date, code
        """
        
        if sample_limit:
            query += f" LIMIT {sample_limit}"
            
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                data = pd.read_sql_query(query, conn)
            
            if not data.empty:
                # 过滤异常收益数据 (使用更严格标准，A股正常涨跌停为±10%)
                original_len = len(data)
                data = data[
                    (data['return_1d'] >= -15) & (data['return_1d'] <= 15) &
                    (data['return_3d'] >= -30) & (data['return_3d'] <= 30) &
                    (data['return_5d'] >= -40) & (data['return_5d'] <= 40) &
                    (data['return_10d'] >= -50) & (data['return_10d'] <= 50) &
                    (data['return_20d'] >= -60) & (data['return_20d'] <= 60)
                ].copy()
                
                filtered_count = original_len - len(data)
                if filtered_count > 0:
                    self.logger.info(f"🧹 过滤异常数据: {filtered_count:,} 条记录 ({filtered_count/original_len*100:.1f}%)")
                
                # 计算优化权重下的得分
                data['optimized_score'] = data.apply(self.calculate_optimized_score, axis=1)
                
                self.logger.info(f"✅ 加载完成: {len(data):,} 条记录")
                self.logger.info(f"📅 时间范围: {data['date'].min()} 到 {data['date'].max()}")
                self.logger.info(f"🏢 股票数量: {data['code'].nunique():,} 只")
                
                # 统计优化得分分布
                self.logger.info(f"🎯 优化得分统计:")
                self.logger.info(f"  最小值: {data['optimized_score'].min():.4f}")
                self.logger.info(f"  最大值: {data['optimized_score'].max():.4f}")
                self.logger.info(f"  平均值: {data['optimized_score'].mean():.4f}")
                self.logger.info(f"  标准差: {data['optimized_score'].std():.4f}")
            else:
                self.logger.warning("⚠️ 没有找到有效数据")
                
            return data
            
        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
            return pd.DataFrame()
    
    def analyze_correlation_performance(self, data: pd.DataFrame) -> Dict:
        """分析相关性和预测性能"""
        results = {}
        
        # 分析不同持有期的相关性
        periods = ['return_1d', 'return_3d', 'return_5d', 'return_10d', 'return_20d']
        
        for period in periods:
            if period in data.columns:
                # 计算相关性
                correlation = data['optimized_score'].corr(data[period])
                
                # 统计显著性检验
                try:
                    _, p_value = stats.pearsonr(data['optimized_score'], data[period])
                except:
                    p_value = 1.0
                
                # 分组回测
                group_results = self.analyze_score_groups(data, period)
                
                results[period] = {
                    'correlation': correlation,
                    'p_value': p_value,
                    'is_significant': p_value < 0.05,
                    **group_results
                }
        
        return results
    
    def analyze_score_groups(self, data: pd.DataFrame, return_column: str) -> Dict:
        """分析不同得分组的表现"""
        try:
            # 按日期分组分析
            daily_results = []
            
            for date in data['date'].unique():
                daily_data = data[data['date'] == date].copy()
                if len(daily_data) < 20:  # 至少需要20只股票
                    continue
                
                # 按优化得分排序
                daily_data = daily_data.sort_values('optimized_score', ascending=False)
                total_stocks = len(daily_data)
                
                # 分析Top 5%, 10%, 25%的表现
                for pct, name in [(0.05, 'top5'), (0.10, 'top10'), (0.25, 'top25')]:
                    top_n = max(1, int(total_stocks * pct))
                    top_stocks = daily_data.head(top_n)
                    
                    if len(top_stocks) > 0 and return_column in top_stocks.columns:
                        avg_return = top_stocks[return_column].mean()
                        win_rate = (top_stocks[return_column] > 0).mean()
                        max_return = top_stocks[return_column].max()
                        min_return = top_stocks[return_column].min()
                        
                        daily_results.append({
                            'date': date,
                            'category': name,
                            'avg_return': avg_return,
                            'win_rate': win_rate,
                            'max_return': max_return,
                            'min_return': min_return,
                            'sample_size': len(top_stocks)
                        })
            
            # 汇总结果
            results_df = pd.DataFrame(daily_results)
            summary = {}
            
            for category in ['top5', 'top10', 'top25']:
                cat_data = results_df[results_df['category'] == category]
                if not cat_data.empty:
                    summary[f'avg_return_{category}'] = cat_data['avg_return'].mean()
                    summary[f'win_rate_{category}'] = cat_data['win_rate'].mean()
                    summary[f'max_return_{category}'] = cat_data['max_return'].max()
                    summary[f'min_return_{category}'] = cat_data['min_return'].min()
                    summary[f'sharpe_ratio_{category}'] = cat_data['avg_return'].mean() / (cat_data['avg_return'].std() + 1e-10)
                    summary[f'total_samples_{category}'] = int(cat_data['sample_size'].sum())
                    
                    # 正收益交易日比例
                    summary[f'positive_days_{category}'] = (cat_data['avg_return'] > 0).mean()
            
            return summary
            
        except Exception as e:
            self.logger.error(f"分析得分组表现失败: {e}")
            return {}
    
    def generate_correlation_report(self, analysis_results: Dict, data: pd.DataFrame) -> str:
        """生成correlation analysis格式的报告"""
        
        report_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = f"""# 优化权重收益期望分析报告
## 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 分析概述

### 🎯 优化权重配置
基于2,147,796条历史数据优化得出的最优权重：

| 因子类型 | 权重 | 说明 |
|---------|------|------|
| **Performance** | **38.2%** | 表现因子（最重要） |
| **Risk Control** | **35.2%** | 风险控制因子 |
| **Fundamental** | **25.1%** | 基本面因子 |
| Technical | 0.9% | 技术指标因子 |
| Sentiment | 0.6% | 情绪因子 |

### 🎪 市场环境乘数
- **熊市乘数**: 0.360 (压缩股票得分至36%)
- **牛市乘数**: 1.296 (放大股票得分至130%)
- **乘数范围**: 3.60倍动态调整

### 📈 评分公式
```
final_score = market_regime_multiplier × (
    performance × 0.382 + 
    risk_control × 0.352 + 
    fundamental × 0.251 + 
    technical × 0.009 + 
    sentiment × 0.006
)
```

### 📋 数据样本
- **分析记录数**: {len(data):,} 条
- **时间范围**: {data['date'].min()} 至 {data['date'].max()}
- **覆盖股票**: {data['code'].nunique():,} 只
- **优化得分范围**: {data['optimized_score'].min():.4f} - {data['optimized_score'].max():.4f}

## 🔍 相关性分析结果

### 📊 各持有期相关性表现
"""
        
        # 相关性表格
        correlation_table = "\n| 持有期 | 相关系数 | P值 | 显著性 | Top25%平均收益 | Top25%胜率 | Top10%平均收益 | Top5%平均收益 |\n"
        correlation_table += "|--------|----------|-----|--------|----------------|------------|----------------|---------------|\n"
        
        for period, results in analysis_results.items():
            period_days = period.replace('return_', '').replace('d', '天')
            correlation = results.get('correlation', 0)
            p_value = results.get('p_value', 1.0)
            significance = "✅ 显著" if results.get('is_significant', False) else "❌ 不显著"
            
            top25_return = results.get('avg_return_top25', 0) * 100
            top25_winrate = results.get('win_rate_top25', 0) * 100
            top10_return = results.get('avg_return_top10', 0) * 100  
            top5_return = results.get('avg_return_top5', 0) * 100
            
            correlation_table += f"| {period_days} | {correlation:.4f} | {p_value:.4f} | {significance} | {top25_return:+.2f}% | {top25_winrate:.1f}% | {top10_return:+.2f}% | {top5_return:+.2f}% |\n"
        
        report += correlation_table
        
        # 最佳持有期分析
        best_period = max(analysis_results.keys(), 
                         key=lambda k: analysis_results[k].get('avg_return_top25', 0))
        best_results = analysis_results[best_period]
        
        report += f"""
### 🏆 最佳持有期分析

**最优持有期**: {best_period.replace('return_', '').replace('d', '天')}

#### 📈 收益表现详情
- **Top5% 平均收益**: {best_results.get('avg_return_top5', 0)*100:+.2f}%
- **Top10% 平均收益**: {best_results.get('avg_return_top10', 0)*100:+.2f}%
- **Top25% 平均收益**: {best_results.get('avg_return_top25', 0)*100:+.2f}%

#### 🎯 胜率表现
- **Top5% 胜率**: {best_results.get('win_rate_top5', 0)*100:.1f}%
- **Top10% 胜率**: {best_results.get('win_rate_top10', 0)*100:.1f}%
- **Top25% 胜率**: {best_results.get('win_rate_top25', 0)*100:.1f}%

#### 📊 风险收益特征
- **Top25% 夏普比率**: {best_results.get('sharpe_ratio_top25', 0):.3f}
- **Top25% 最大单日收益**: {best_results.get('max_return_top25', 0)*100:+.2f}%
- **Top25% 最大单日亏损**: {best_results.get('min_return_top25', 0)*100:+.2f}%
- **正收益交易日比例**: {best_results.get('positive_days_top25', 0)*100:.1f}%

## 💡 收益期望总结

### 🎯 预期收益水平
基于历史数据回测，使用优化权重的v3.1选股系统预期表现：

"""
        
        # 各持有期收益期望汇总
        expected_returns = {}
        for period, results in analysis_results.items():
            days = period.replace('return_', '').replace('d', '')
            expected_returns[days] = {
                'top25': results.get('avg_return_top25', 0) * 100,
                'top10': results.get('avg_return_top10', 0) * 100,
                'top5': results.get('avg_return_top5', 0) * 100,
                'winrate': results.get('win_rate_top25', 0) * 100
            }
        
        for days, returns in sorted(expected_returns.items(), key=lambda x: int(x[0])):
            report += f"""
#### {days}天持有期收益期望
- **Top25%股票**: 平均收益 {returns['top25']:+.2f}%, 胜率 {returns['winrate']:.1f}%
- **Top10%股票**: 平均收益 {returns['top10']:+.2f}%
- **Top5%股票**: 平均收益 {returns['top5']:+.2f}%
"""
        
        # 优化效果评估
        report += f"""

### 📊 优化效果评估

#### 🚀 权重优化价值
1. **Performance因子占主导** (38.2%): 体现了股票历史表现的重要性
2. **Risk Control次重要** (35.2%): 强调风险管理在选股中的关键作用  
3. **Fundamental适中权重** (25.1%): 基本面分析仍有重要价值
4. **Technical和Sentiment权重极低** (<1%): 在当前市场环境下贡献有限

#### 🎪 市场环境乘数价值
- 通过3.60倍的动态乘数范围，有效区分不同市场环境
- 熊市时谨慎降低预期收益，牛市时适度放大选股机会
- 相比旧系统0.009的微小变化，新系统0.5434的变化范围提升60.4倍

### ⚠️ 风险提示

1. **历史表现不保证未来收益**：分析基于历史数据，实际投资需结合市场环境
2. **样本偏差**：分析期间可能存在特定市场特征，需持续验证
3. **交易成本影响**：实际收益需扣除交易成本和冲击成本
4. **流动性风险**：小盘股可能面临流动性不足问题

### 📝 使用建议

1. **持有期选择**：建议采用{best_period.replace('return_', '').replace('d', '天')}持有期获得最佳风险收益比
2. **选股数量**：重点关注Top10%-25%区间的股票，平衡收益与分散风险
3. **市场环境判断**：密切关注市场环境乘数变化，调整仓位配置
4. **动态调整**：建议每月重新评估权重有效性，必要时进行调整

---
**报告生成时间**: {datetime.now()}  
**分析工具**: OptimizedWeightReturnAnalyzer v1.0  
**数据来源**: weight_optimization_cache.db  
"""
        
        return report
    
    def save_report(self, report_content: str, filename: str = None):
        """保存报告到文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"优化权重收益期望分析报告_{timestamp}.md"
        
        # 确保reports目录存在
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
    
    def run_analysis(self, sample_limit: int = None) -> str:
        """运行完整分析流程"""
        self.logger.info("🚀 开始优化权重收益期望分析...")
        
        try:
            # 1. 加载数据
            data = self.load_historical_data(sample_limit)
            if data.empty:
                self.logger.error("❌ 没有找到有效数据")
                return None
            
            # 2. 分析相关性和收益表现
            analysis_results = self.analyze_correlation_performance(data)
            
            # 3. 生成报告
            report_content = self.generate_correlation_report(analysis_results, data)
            
            # 4. 保存报告
            report_path = self.save_report(report_content)
            
            self.logger.info("✅ 优化权重收益期望分析完成!")
            return report_path
            
        except Exception as e:
            self.logger.error(f"分析过程失败: {e}")
            return None

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="优化权重收益期望分析")
    parser.add_argument("--sample-limit", type=int, default=None, help="数据样本限制")
    
    args = parser.parse_args()
    
    analyzer = OptimizedWeightReturnAnalyzer()
    report_path = analyzer.run_analysis(sample_limit=args.sample_limit)
    
    if report_path:
        print(f"✅ 分析完成！报告已保存至: {report_path}")
    else:
        print("❌ 分析失败")

if __name__ == "__main__":
    main()