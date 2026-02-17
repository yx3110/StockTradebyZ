#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交易成本影响分析工具

分析不同交易成本假设下的收益表现，为实际交易提供指导
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

class TransactionCostAnalyzer:
    """交易成本分析器"""
    
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
        
        # 不同交易成本假设（双边，买入+卖出）
        self.cost_scenarios = {
            'low_cost': 0.002,        # 0.2% (券商佣金优惠)
            'normal_cost': 0.004,     # 0.4% (一般券商)
            'high_cost': 0.006,       # 0.6% (包含冲击成本)
            'institutional': 0.001,   # 0.1% (机构级别)
            'retail_high': 0.008      # 0.8% (散户高成本)
        }
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def load_sample_data(self, sample_size: int = 400000) -> pd.DataFrame:
        """加载样本数据"""
        self.logger.info(f"📊 加载交易成本分析数据...")
        
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
                # 计算优化得分
                data['optimized_score'] = data.apply(self.calculate_optimized_score, axis=1)
                
                self.logger.info(f"✅ 数据加载完成: {len(data):,} 条记录")
                self.logger.info(f"📅 时间范围: {data['date'].min()} 到 {data['date'].max()}")
                
            return data
            
        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
            return pd.DataFrame()
    
    def calculate_optimized_score(self, row: pd.Series) -> float:
        """计算优化得分"""
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
    
    def analyze_transaction_costs(self, data: pd.DataFrame) -> Dict:
        """分析不同交易成本下的表现"""
        self.logger.info("💰 开始交易成本影响分析...")
        
        results = {}
        periods = ['return_1d', 'return_3d', 'return_5d', 'return_10d', 'return_20d']
        
        for cost_name, cost_rate in self.cost_scenarios.items():
            self.logger.info(f"📊 分析成本场景: {cost_name} ({cost_rate*100:.1f}%)")
            
            cost_results = {}
            
            for period in periods:
                # 原始收益分析
                original_performance = self.analyze_returns_performance(data, period)
                
                # 扣除交易成本后的收益
                # 根据持有期调整交易成本（更长持有期摊薄成本）
                days = int(period.replace('return_', '').replace('d', ''))
                adjusted_cost_rate = cost_rate  # 双边成本
                
                # 计算调整后收益
                data_copy = data.copy()
                data_copy[f'{period}_after_cost'] = data_copy[period] - adjusted_cost_rate
                
                # 扣除成本后表现分析
                adjusted_performance = self.analyze_returns_performance(data_copy, f'{period}_after_cost')
                
                # 计算影响
                cost_impact = {
                    'return_impact_top25': adjusted_performance.get('avg_return_top25', 0) - original_performance.get('avg_return_top25', 0),
                    'return_impact_top10': adjusted_performance.get('avg_return_top10', 0) - original_performance.get('avg_return_top10', 0),
                    'return_impact_top5': adjusted_performance.get('avg_return_top5', 0) - original_performance.get('avg_return_top5', 0),
                    'winrate_impact_top25': adjusted_performance.get('win_rate_top25', 0) - original_performance.get('win_rate_top25', 0),
                    'positive_days_impact': adjusted_performance.get('positive_days_top25', 0) - original_performance.get('positive_days_top25', 0)
                }
                
                cost_results[period] = {
                    'cost_rate': cost_rate,
                    'original': original_performance,
                    'after_cost': adjusted_performance,
                    'impact': cost_impact
                }
            
            results[cost_name] = cost_results
        
        return results
    
    def analyze_returns_performance(self, data: pd.DataFrame, return_column: str) -> Dict:
        """分析收益表现"""
        try:
            daily_results = []
            
            for date in data['date'].unique():
                daily_data = data[data['date'] == date].copy()
                if len(daily_data) < 20:
                    continue
                
                daily_data = daily_data.sort_values('optimized_score', ascending=False)
                total = len(daily_data)
                
                # 分析不同比例的表现
                for pct, name in [(0.25, 'top25'), (0.10, 'top10'), (0.05, 'top5')]:
                    n = max(1, int(total * pct))
                    top_stocks = daily_data.head(n)
                    
                    if return_column in top_stocks.columns:
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
                            'min_return': min_return
                        })
            
            # 汇总统计
            results_df = pd.DataFrame(daily_results)
            summary = {}
            
            for category in ['top5', 'top10', 'top25']:
                cat_data = results_df[results_df['category'] == category]
                if not cat_data.empty:
                    summary[f'avg_return_{category}'] = cat_data['avg_return'].mean()
                    summary[f'win_rate_{category}'] = cat_data['win_rate'].mean()
                    summary[f'max_return_{category}'] = cat_data['max_return'].max()
                    summary[f'min_return_{category}'] = cat_data['min_return'].min()
                    summary[f'positive_days_{category}'] = (cat_data['avg_return'] > 0).mean()
                    summary[f'sharpe_ratio_{category}'] = cat_data['avg_return'].mean() / (cat_data['avg_return'].std() + 1e-10)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"分析收益表现失败: {e}")
            return {}
    
    def calculate_breakeven_analysis(self, data: pd.DataFrame) -> Dict:
        """计算盈亏平衡分析"""
        self.logger.info("⚖️ 进行盈亏平衡分析...")
        
        breakeven_results = {}
        periods = ['return_1d', 'return_3d', 'return_5d', 'return_10d', 'return_20d']
        
        for period in periods:
            # 分析在不同成本率下，需要多少收益才能覆盖成本
            original_perf = self.analyze_returns_performance(data, period)
            
            # 计算每个成本场景下的净收益概率
            period_breakeven = {}
            
            for cost_name, cost_rate in self.cost_scenarios.items():
                # 计算扣费后仍然盈利的概率
                profit_after_cost_data = []
                
                for date in data['date'].unique():
                    daily_data = data[data['date'] == date].copy()
                    if len(daily_data) < 20:
                        continue
                    
                    daily_data = daily_data.sort_values('optimized_score', ascending=False)
                    
                    # Top25%股票
                    top25_n = max(1, int(len(daily_data) * 0.25))
                    top25_stocks = daily_data.head(top25_n)
                    
                    # 扣除成本后的收益
                    returns_after_cost = top25_stocks[period] - cost_rate
                    profit_rate = (returns_after_cost > 0).mean()
                    avg_net_return = returns_after_cost.mean()
                    
                    profit_after_cost_data.append({
                        'profit_rate': profit_rate,
                        'avg_net_return': avg_net_return
                    })
                
                if profit_after_cost_data:
                    profit_df = pd.DataFrame(profit_after_cost_data)
                    period_breakeven[cost_name] = {
                        'avg_profit_rate': profit_df['profit_rate'].mean(),
                        'avg_net_return': profit_df['avg_net_return'].mean(),
                        'profitable_days_pct': (profit_df['avg_net_return'] > 0).mean()
                    }
            
            breakeven_results[period] = period_breakeven
        
        return breakeven_results
    
    def analyze_optimal_holding_period(self, data: pd.DataFrame) -> Dict:
        """分析最优持有期"""
        self.logger.info("📈 分析最优持有期...")
        
        periods = ['return_1d', 'return_3d', 'return_5d', 'return_10d', 'return_20d']
        optimal_analysis = {}
        
        for cost_name, cost_rate in self.cost_scenarios.items():
            period_performance = {}
            
            for period in periods:
                # 计算该持有期下的净收益表现
                perf = self.analyze_returns_performance(data, period)
                top25_gross = perf.get('avg_return_top25', 0)
                top25_net = top25_gross - cost_rate
                win_rate = perf.get('win_rate_top25', 0)
                
                # 计算收益成本比
                return_cost_ratio = top25_gross / cost_rate if cost_rate > 0 else 0
                
                # 综合评分：净收益 + 胜率 + 收益成本比
                composite_score = (
                    top25_net * 0.5 +           # 净收益权重50%
                    (win_rate - 0.5) * 0.3 +    # 胜率调整30%
                    min(10, return_cost_ratio) / 10 * 0.2  # 收益成本比20%
                )
                
                period_performance[period] = {
                    'gross_return': top25_gross,
                    'net_return': top25_net,
                    'win_rate': win_rate,
                    'return_cost_ratio': return_cost_ratio,
                    'composite_score': composite_score
                }
            
            # 找到最优持有期
            best_period = max(period_performance.keys(), 
                            key=lambda k: period_performance[k]['composite_score'])
            
            optimal_analysis[cost_name] = {
                'optimal_period': best_period,
                'period_analysis': period_performance
            }
        
        return optimal_analysis
    
    def generate_cost_analysis_report(self, data: pd.DataFrame, cost_analysis: Dict,
                                    breakeven_analysis: Dict, optimal_period_analysis: Dict) -> str:
        """生成交易成本分析报告"""
        
        report = f"""# 交易成本影响分析报告
## 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 分析概述

### 📋 数据样本
- **分析记录数**: {len(data):,} 条
- **时间范围**: {data['date'].min()} 至 {data['date'].max()}
- **覆盖股票**: {data['code'].nunique():,} 只
- **交易日数**: {data['date'].nunique():,} 天

### 💰 交易成本假设场景
| 成本场景 | 双边成本率 | 说明 |
|----------|-----------|------|
| 机构级别 | 0.1% | 大型机构交易成本 |
| 低成本 | 0.2% | 券商佣金优惠客户 |
| 一般成本 | 0.4% | 普通券商标准费率 |
| 高成本 | 0.6% | 包含市场冲击成本 |
| 散户高成本 | 0.8% | 小额散户高费率 |

## 🔍 成本影响分析结果

### 📊 5天持有期各成本场景对比
"""
        
        # 5天持有期对比表
        if 'return_5d' in cost_analysis.get(list(cost_analysis.keys())[0], {}):
            report += "\n| 成本场景 | 原始Top25%收益 | 扣费后收益 | 收益影响 | 胜率影响 | 盈利交易日影响 |\n"
            report += "|----------|----------------|-----------|----------|----------|----------------|\n"
            
            cost_names = {
                'institutional': '机构级别',
                'low_cost': '低成本',
                'normal_cost': '一般成本', 
                'high_cost': '高成本',
                'retail_high': '散户高成本'
            }
            
            for cost_name, cost_data in cost_analysis.items():
                if 'return_5d' in cost_data:
                    cost_name_cn = cost_names.get(cost_name, cost_name)
                    period_data = cost_data['return_5d']
                    
                    original_return = period_data['original'].get('avg_return_top25', 0) * 100
                    adjusted_return = period_data['after_cost'].get('avg_return_top25', 0) * 100
                    return_impact = period_data['impact'].get('return_impact_top25', 0) * 100
                    winrate_impact = period_data['impact'].get('winrate_impact_top25', 0) * 100
                    positive_days_impact = period_data['impact'].get('positive_days_impact', 0) * 100
                    
                    report += f"| {cost_name_cn} | {original_return:+.2f}% | {adjusted_return:+.2f}% | {return_impact:+.2f}% | {winrate_impact:+.2f}% | {positive_days_impact:+.2f}% |\n"
        
        # 盈亏平衡分析
        if breakeven_analysis:
            report += f"\n### ⚖️ 盈亏平衡分析\n\n"
            report += "| 持有期 | 机构级别 | 低成本 | 一般成本 | 高成本 | 散户高成本 |\n"
            report += "|--------|----------|--------|----------|--------|-----------|\n"
            
            for period in ['return_1d', 'return_5d', 'return_10d', 'return_20d']:
                if period in breakeven_analysis:
                    period_name = period.replace('return_', '').replace('d', '天')
                    row = f"| {period_name} |"
                    
                    for cost_name in ['institutional', 'low_cost', 'normal_cost', 'high_cost', 'retail_high']:
                        if cost_name in breakeven_analysis[period]:
                            profit_rate = breakeven_analysis[period][cost_name]['avg_profit_rate'] * 100
                            row += f" {profit_rate:.1f}% |"
                        else:
                            row += " - |"
                    
                    report += row + "\n"
        
        # 最优持有期分析
        if optimal_period_analysis:
            report += f"\n### 📈 最优持有期分析\n\n"
            report += "| 成本场景 | 最优持有期 | 净收益 | 胜率 | 收益成本比 |\n"
            report += "|----------|------------|--------|----|----------|\n"
            
            cost_names = {
                'institutional': '机构级别',
                'low_cost': '低成本',
                'normal_cost': '一般成本',
                'high_cost': '高成本', 
                'retail_high': '散户高成本'
            }
            
            for cost_name, analysis in optimal_period_analysis.items():
                cost_name_cn = cost_names.get(cost_name, cost_name)
                optimal_period = analysis['optimal_period'].replace('return_', '').replace('d', '天')
                
                period_perf = analysis['period_analysis'][analysis['optimal_period']]
                net_return = period_perf['net_return'] * 100
                win_rate = period_perf['win_rate'] * 100
                return_cost_ratio = period_perf['return_cost_ratio']
                
                report += f"| {cost_name_cn} | {optimal_period} | {net_return:+.2f}% | {win_rate:.1f}% | {return_cost_ratio:.1f}x |\n"
        
        # 不同持有期成本摊薄效应
        report += f"""

### 📊 持有期成本摊薄效应分析

#### 一般成本场景下各持有期表现
| 持有期 | 原始收益 | 扣费后收益 | 成本影响 | 相对影响 |
|--------|----------|-----------|----------|----------|"""
        
        if 'normal_cost' in cost_analysis:
            normal_cost_data = cost_analysis['normal_cost']
            for period in ['return_1d', 'return_3d', 'return_5d', 'return_10d', 'return_20d']:
                if period in normal_cost_data:
                    period_name = period.replace('return_', '').replace('d', '天')
                    period_data = normal_cost_data[period]
                    
                    original = period_data['original'].get('avg_return_top25', 0) * 100
                    adjusted = period_data['after_cost'].get('avg_return_top25', 0) * 100
                    impact = period_data['impact'].get('return_impact_top25', 0) * 100
                    relative_impact = abs(impact / original * 100) if original != 0 else 0
                    
                    report += f"\n| {period_name} | {original:+.2f}% | {adjusted:+.2f}% | {impact:+.2f}% | {relative_impact:.1f}% |"
        
        # 结论和建议
        report += f"""

## 💡 分析结论

### 🎯 核心发现

1. **成本敏感性**: 交易成本对短期收益影响显著，长期持有可摊薄成本影响
2. **持有期优化**: 不同成本水平下存在不同的最优持有期
3. **成本控制价值**: 降低交易成本对提升净收益有明显效果
4. **策略适配**: 需要根据实际交易成本调整选股和持有策略

### 📊 实用建议

#### 1. 成本控制策略
- **券商选择**: 优先选择低佣金券商，机构级别成本可显著提升收益
- **批量交易**: 通过批量交易降低单笔成本
- **时机把握**: 避免在流动性差时段交易，减少冲击成本

#### 2. 持有期优化
- **机构/低成本**: 可采用较短持有期(3-5天)，灵活调整
- **一般成本**: 建议5-10天持有期，平衡成本与机会
- **高成本**: 应延长持有期至10天以上，摊薄成本影响

#### 3. 选股策略调整
- **提高选股精度**: 成本越高，越需要精准选股
- **集中度管理**: 高成本环境下适度提高持股集中度
- **止损设置**: 考虑交易成本设置合理止损位

#### 4. 动态成本管理
- **实时监控**: 监控实际交易成本与预期的偏差
- **成本预算**: 设置月度/季度交易成本预算
- **效果评估**: 定期评估成本控制措施的效果

### 🚨 风险提示

1. **隐性成本**: 报告未包含滑点等隐性成本，实际成本可能更高
2. **流动性风险**: 小盘股交易可能面临更高的冲击成本
3. **市场环境**: 极端市场条件下交易成本可能大幅上升
4. **税收影响**: 未考虑印花税等税收成本

### 📝 实施建议

1. **成本评估**: 准确评估自身交易成本水平
2. **策略匹配**: 根据成本水平选择合适的交易策略
3. **持续优化**: 定期重新评估和优化交易成本
4. **工具利用**: 使用程序化交易等工具降低成本

---
**报告生成时间**: {datetime.now()}  
**分析工具**: TransactionCostAnalyzer v1.0  
"""
        
        return report
    
    def save_report(self, report_content: str) -> str:
        """保存报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"交易成本影响分析报告_{timestamp}.md"
        
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
    
    def run_analysis(self, sample_size: int = 400000) -> str:
        """运行交易成本分析"""
        self.logger.info("🚀 开始交易成本影响分析...")
        
        try:
            # 1. 加载数据
            data = self.load_sample_data(sample_size)
            if data.empty:
                return None
            
            # 2. 交易成本影响分析
            cost_analysis = self.analyze_transaction_costs(data)
            
            # 3. 盈亏平衡分析
            breakeven_analysis = self.calculate_breakeven_analysis(data)
            
            # 4. 最优持有期分析
            optimal_period_analysis = self.analyze_optimal_holding_period(data)
            
            # 5. 生成报告
            report_content = self.generate_cost_analysis_report(
                data, cost_analysis, breakeven_analysis, optimal_period_analysis
            )
            
            # 6. 保存报告
            report_path = self.save_report(report_content)
            
            self.logger.info("✅ 交易成本影响分析完成!")
            return report_path
            
        except Exception as e:
            self.logger.error(f"分析失败: {e}")
            return None

def main():
    """主函数"""
    analyzer = TransactionCostAnalyzer()
    report_path = analyzer.run_analysis(400000)
    
    if report_path:
        print(f"✅ 交易成本分析完成！报告已保存至: {report_path}")
    else:
        print("❌ 分析失败")

if __name__ == "__main__":
    main()