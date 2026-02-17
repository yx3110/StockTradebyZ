#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态权重优化分析工具

实现权重配置的动态调整和时间序列稳定性分析
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats
from scipy.optimize import differential_evolution
import warnings

warnings.filterwarnings('ignore')

class DynamicWeightOptimizer:
    """动态权重优化器"""
    
    def __init__(self, cache_db_path: str = "weight_optimization_cache.db"):
        self.cache_db_path = cache_db_path
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # 当前最优权重（作为基准）
        self.base_weights = {
            'technical': 0.0088,      # 0.9%
            'fundamental': 0.2511,    # 25.1%
            'performance': 0.3823,    # 38.2%
            'sentiment': 0.0055,      # 0.6%
            'risk_control': 0.3522    # 35.2%
        }
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def load_time_series_data(self, sample_size: int = 500000) -> pd.DataFrame:
        """加载时间序列数据"""
        self.logger.info(f"📊 加载时间序列数据...")
        
        query = f"""
        SELECT 
            code, date, technical, fundamental, performance, sentiment, risk_control,
            market_regime, return_1d, return_3d, return_5d, return_10d, return_20d
        FROM stock_indicators
        WHERE technical IS NOT NULL AND fundamental IS NOT NULL AND performance IS NOT NULL
        AND sentiment IS NOT NULL AND risk_control IS NOT NULL AND market_regime IS NOT NULL
        AND return_5d IS NOT NULL
        AND return_1d BETWEEN -15 AND 15
        AND return_5d BETWEEN -40 AND 40
        ORDER BY date, code
        LIMIT {sample_size}
        """
        
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                data = pd.read_sql_query(query, conn)
            
            if not data.empty:
                # 转换日期格式
                data['date'] = pd.to_datetime(data['date'], format='ISO8601')
                data = data.sort_values(['date', 'code'])
                
                self.logger.info(f"✅ 数据加载完成: {len(data):,} 条记录")
                self.logger.info(f"📅 时间范围: {data['date'].min().date()} 到 {data['date'].max().date()}")
                self.logger.info(f"🏢 股票数量: {data['code'].nunique():,} 只")
                
            return data
            
        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
            return pd.DataFrame()
    
    def calculate_score_with_weights(self, data: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
        """使用给定权重计算得分"""
        # 个股质量得分
        quality_scores = (
            data['technical'] * weights['technical'] +
            data['fundamental'] * weights['fundamental'] + 
            data['performance'] * weights['performance'] +
            data['sentiment'] * weights['sentiment'] +
            data['risk_control'] * weights['risk_control']
        )
        
        # 市场环境乘数
        market_multipliers = 0.360 + (data['market_regime'] - 0.3472) / (0.8905 - 0.3472) * (1.296 - 0.360)
        market_multipliers = market_multipliers.clip(0.360, 1.296)
        
        return market_multipliers * quality_scores
    
    def evaluate_weights_performance(self, data: pd.DataFrame, weights: Dict[str, float], 
                                   target_period: str = 'return_5d') -> Dict:
        """评估权重配置的表现"""
        try:
            # 计算得分
            data['score'] = self.calculate_score_with_weights(data, weights)
            
            # 相关性分析
            correlation = data['score'].corr(data[target_period])
            
            # 分组回测分析
            daily_results = []
            
            for date in data['date'].dt.date.unique():
                daily_data = data[data['date'].dt.date == date].copy()
                if len(daily_data) < 20:
                    continue
                
                daily_data = daily_data.sort_values('score', ascending=False)
                total = len(daily_data)
                
                # Top25%表现
                top25_n = max(1, int(total * 0.25))
                top25_stocks = daily_data.head(top25_n)
                
                avg_return = top25_stocks[target_period].mean()
                win_rate = (top25_stocks[target_period] > 0).mean()
                
                daily_results.append({
                    'date': date,
                    'avg_return': avg_return,
                    'win_rate': win_rate
                })
            
            results_df = pd.DataFrame(daily_results)
            
            if not results_df.empty:
                avg_return = results_df['avg_return'].mean()
                win_rate = results_df['win_rate'].mean()
                sharpe_ratio = avg_return / (results_df['avg_return'].std() + 1e-10)
                positive_days = (results_df['avg_return'] > 0).mean()
            else:
                avg_return = win_rate = sharpe_ratio = positive_days = 0
            
            return {
                'correlation': correlation if not pd.isna(correlation) else 0,
                'avg_return': avg_return,
                'win_rate': win_rate,
                'sharpe_ratio': sharpe_ratio,
                'positive_days': positive_days,
                'sample_size': len(data)
            }
            
        except Exception as e:
            self.logger.error(f"评估权重表现失败: {e}")
            return {'correlation': 0, 'avg_return': 0, 'win_rate': 0, 'sharpe_ratio': 0, 'positive_days': 0}
    
    def rolling_window_optimization(self, data: pd.DataFrame, 
                                  train_months: int = 3, test_months: int = 1) -> List[Dict]:
        """滚动窗口优化"""
        self.logger.info(f"📈 开始滚动窗口优化 (训练:{train_months}月, 测试:{test_months}月)...")
        
        # 按月分组
        data['year_month'] = data['date'].dt.to_period('M')
        months = sorted(data['year_month'].unique())
        
        if len(months) < train_months + test_months + 2:
            self.logger.warning("⚠️ 数据月份不足")
            return []
        
        rolling_results = []
        
        for i in range(train_months, len(months) - test_months + 1):
            # 训练期和测试期
            train_periods = months[i-train_months:i]
            test_periods = months[i:i+test_months]
            
            train_data = data[data['year_month'].isin(train_periods)].copy()
            test_data = data[data['year_month'].isin(test_periods)].copy()
            
            if len(train_data) < 1000 or len(test_data) < 100:
                continue
            
            self.logger.info(f"📊 处理期间: {train_periods[0]} - {train_periods[-1]} -> {test_periods[0]}")
            
            # 在训练数据上优化权重
            optimal_weights = self.optimize_weights_on_data(train_data)
            
            # 在训练数据上评估基准权重和优化权重
            base_train_perf = self.evaluate_weights_performance(train_data, self.base_weights)
            opt_train_perf = self.evaluate_weights_performance(train_data, optimal_weights)
            
            # 在测试数据上评估
            base_test_perf = self.evaluate_weights_performance(test_data, self.base_weights)
            opt_test_perf = self.evaluate_weights_performance(test_data, optimal_weights)
            
            rolling_results.append({
                'train_period': f"{train_periods[0]} - {train_periods[-1]}",
                'test_period': f"{test_periods[0]}",
                'optimal_weights': optimal_weights,
                'base_train_performance': base_train_perf,
                'optimal_train_performance': opt_train_perf,
                'base_test_performance': base_test_perf,
                'optimal_test_performance': opt_test_perf,
                'improvement_train': opt_train_perf['correlation'] - base_train_perf['correlation'],
                'improvement_test': opt_test_perf['correlation'] - base_test_perf['correlation']
            })
        
        self.logger.info(f"✅ 滚动窗口优化完成: {len(rolling_results)} 个周期")
        return rolling_results
    
    def optimize_weights_on_data(self, data: pd.DataFrame, target_period: str = 'return_5d') -> Dict[str, float]:
        """在给定数据上优化权重"""
        
        def objective_function(x):
            """优化目标函数"""
            weights_array = np.array(x)
            weights_array = weights_array / weights_array.sum()  # 归一化
            
            weights = {
                'technical': weights_array[0],
                'fundamental': weights_array[1],
                'performance': weights_array[2],
                'sentiment': weights_array[3],
                'risk_control': weights_array[4]
            }
            
            performance = self.evaluate_weights_performance(data, weights, target_period)
            # 最大化相关性和收益的组合
            composite_score = performance['correlation'] * 0.6 + performance['avg_return'] * 0.4
            return -composite_score  # 最小化负数
        
        # 搜索边界
        bounds = [(0.001, 0.8) for _ in range(5)]
        
        try:
            # 差分进化优化
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=20,  # 减少迭代次数以提高速度
                popsize=10,
                seed=42
            )
            
            # 提取最优权重
            optimal_weights_array = result.x
            optimal_weights_array = optimal_weights_array / optimal_weights_array.sum()
            
            optimal_weights = {
                'technical': optimal_weights_array[0],
                'fundamental': optimal_weights_array[1],
                'performance': optimal_weights_array[2],
                'sentiment': optimal_weights_array[3],
                'risk_control': optimal_weights_array[4]
            }
            
            return optimal_weights
            
        except Exception as e:
            self.logger.error(f"权重优化失败: {e}")
            return self.base_weights.copy()
    
    def analyze_weight_stability(self, rolling_results: List[Dict]) -> Dict:
        """分析权重稳定性"""
        if not rolling_results:
            return {}
        
        # 收集所有优化权重
        weights_history = []
        for result in rolling_results:
            weights_history.append(result['optimal_weights'])
        
        # 计算每个因子的权重变化统计
        factor_stats = {}
        for factor in self.base_weights.keys():
            factor_weights = [w[factor] for w in weights_history]
            factor_stats[factor] = {
                'mean': np.mean(factor_weights),
                'std': np.std(factor_weights),
                'min': np.min(factor_weights),
                'max': np.max(factor_weights),
                'coefficient_of_variation': np.std(factor_weights) / (np.mean(factor_weights) + 1e-10)
            }
        
        # 性能稳定性分析
        train_improvements = [r['improvement_train'] for r in rolling_results]
        test_improvements = [r['improvement_test'] for r in rolling_results]
        
        performance_stats = {
            'avg_train_improvement': np.mean(train_improvements),
            'avg_test_improvement': np.mean(test_improvements),
            'train_improvement_std': np.std(train_improvements),
            'test_improvement_std': np.std(test_improvements),
            'positive_test_periods': sum(1 for x in test_improvements if x > 0) / len(test_improvements)
        }
        
        return {
            'factor_stability': factor_stats,
            'performance_stability': performance_stats,
            'total_periods': len(rolling_results)
        }
    
    def generate_dynamic_analysis_report(self, data: pd.DataFrame, rolling_results: List[Dict], 
                                       stability_analysis: Dict) -> str:
        """生成动态分析报告"""
        
        report = f"""# 动态权重优化分析报告
## 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 分析概述

### 📋 数据样本
- **分析记录数**: {len(data):,} 条
- **时间范围**: {data['date'].min().date()} 至 {data['date'].max().date()}
- **滚动分析周期**: {len(rolling_results)} 个测试期
- **训练窗口**: 3个月
- **测试窗口**: 1个月

### 🎯 基准权重配置
| 因子类型 | 基准权重 | 说明 |
|---------|----------|------|
| Performance | 38.2% | 表现因子 |
| Risk Control | 35.2% | 风险控制因子 |
| Fundamental | 25.1% | 基本面因子 |
| Technical | 0.9% | 技术指标因子 |
| Sentiment | 0.6% | 情绪因子 |

## 📈 滚动窗口优化结果

### 🎪 权重稳定性分析
"""
        
        if stability_analysis.get('factor_stability'):
            factor_stats = stability_analysis['factor_stability']
            
            report += "\n| 因子类型 | 平均权重 | 标准差 | 变异系数 | 最小值 | 最大值 |\n"
            report += "|---------|----------|--------|-----------|--------|--------|\n"
            
            for factor, stats in factor_stats.items():
                factor_name = {
                    'technical': 'Technical',
                    'fundamental': 'Fundamental', 
                    'performance': 'Performance',
                    'sentiment': 'Sentiment',
                    'risk_control': 'Risk Control'
                }.get(factor, factor)
                
                mean_weight = stats['mean'] * 100
                std_weight = stats['std'] * 100
                cv = stats['coefficient_of_variation']
                min_weight = stats['min'] * 100
                max_weight = stats['max'] * 100
                
                report += f"| {factor_name} | {mean_weight:.1f}% | {std_weight:.1f}% | {cv:.2f} | {min_weight:.1f}% | {max_weight:.1f}% |\n"
        
        # 性能稳定性分析
        if stability_analysis.get('performance_stability'):
            perf_stats = stability_analysis['performance_stability']
            
            report += f"""

### 🚀 性能优化效果

- **训练期平均改进**: {perf_stats['avg_train_improvement']:.4f}
- **测试期平均改进**: {perf_stats['avg_test_improvement']:.4f}
- **测试期改进标准差**: {perf_stats['test_improvement_std']:.4f}
- **测试期正向改进比例**: {perf_stats['positive_test_periods']:.1%}

"""
        
        # 详细周期结果（展示前5个）
        if rolling_results:
            report += "### 📊 详细周期分析 (前5个周期)\n\n"
            report += "| 训练期 | 测试期 | 训练改进 | 测试改进 | Performance权重 | Risk Control权重 |\n"
            report += "|--------|---------|----------|----------|-----------------|------------------|\n"
            
            for i, result in enumerate(rolling_results[:5]):
                train_period = result['train_period']
                test_period = result['test_period']
                train_imp = result['improvement_train']
                test_imp = result['improvement_test']
                perf_weight = result['optimal_weights']['performance'] * 100
                risk_weight = result['optimal_weights']['risk_control'] * 100
                
                report += f"| {train_period} | {test_period} | {train_imp:+.4f} | {test_imp:+.4f} | {perf_weight:.1f}% | {risk_weight:.1f}% |\n"
            
            if len(rolling_results) > 5:
                report += f"\n*... 还有 {len(rolling_results) - 5} 个周期的结果*\n"
        
        # 结论和建议
        report += f"""

## 💡 分析结论

### 🎯 核心发现

1. **权重稳定性**: 通过{len(rolling_results)}个滚动周期验证权重配置的时间稳定性
2. **优化效果**: 动态优化在测试期的改进效果
3. **因子重要性**: 各因子权重的变化范围和稳定性
4. **时间适应性**: 权重配置随时间和市场环境的适应能力

### 📊 优化建议

#### 1. 权重动态调整策略
- **季度调整**: 建议每季度基于最新3个月数据重新优化权重
- **渐进调整**: 权重变化幅度控制在±10%以内，避免剧烈调整
- **稳定性监控**: 重点监控变异系数较高的因子

#### 2. 因子权重管理
- **Performance因子**: 保持35-45%的权重范围
- **Risk Control因子**: 保持30-40%的权重范围  
- **Fundamental因子**: 保持20-30%的权重范围
- **Technical/Sentiment**: 保持较低权重，总计不超过5%

#### 3. 性能监控机制
- **实时追踪**: 监控当前权重配置的表现
- **预警机制**: 当连续2个周期表现下滑时触发重新优化
- **回撤控制**: 设置权重调整的回撤上限

### 🔄 动态调整流程

1. **数据准备**: 收集最近3个月的完整数据
2. **权重优化**: 使用差分进化算法优化权重
3. **稳定性检验**: 验证新权重与历史权重的差异
4. **渐进实施**: 分步骤实施权重调整
5. **效果监控**: 持续监控调整后的表现

---
**报告生成时间**: {datetime.now()}  
**分析工具**: DynamicWeightOptimizer v1.0  
"""
        
        return report
    
    def save_report(self, report_content: str) -> str:
        """保存报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"动态权重优化分析报告_{timestamp}.md"
        
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
    
    def run_analysis(self, sample_size: int = 500000) -> str:
        """运行动态优化分析"""
        self.logger.info("🚀 开始动态权重优化分析...")
        
        try:
            # 1. 加载数据
            data = self.load_time_series_data(sample_size)
            if data.empty:
                return None
            
            # 2. 滚动窗口优化
            rolling_results = self.rolling_window_optimization(data)
            
            if not rolling_results:
                self.logger.warning("⚠️ 滚动优化结果为空")
                return None
            
            # 3. 稳定性分析
            stability_analysis = self.analyze_weight_stability(rolling_results)
            
            # 4. 生成报告
            report_content = self.generate_dynamic_analysis_report(data, rolling_results, stability_analysis)
            
            # 5. 保存报告
            report_path = self.save_report(report_content)
            
            self.logger.info("✅ 动态权重优化分析完成!")
            return report_path
            
        except Exception as e:
            self.logger.error(f"分析失败: {e}")
            return None

def main():
    """主函数"""
    optimizer = DynamicWeightOptimizer()
    report_path = optimizer.run_analysis(500000)
    
    if report_path:
        print(f"✅ 动态权重优化分析完成！报告已保存至: {report_path}")
    else:
        print("❌ 分析失败")

if __name__ == "__main__":
    main()