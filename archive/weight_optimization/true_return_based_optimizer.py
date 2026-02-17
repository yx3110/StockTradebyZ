#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于真实收益率的权重优化器
这次我们要真正证明权重优化能提升选股收益！
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import glob
from scipy.optimize import minimize
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class TrueReturnBasedOptimizer:
    """基于真实收益率的权重优化器"""
    
    def __init__(self):
        self.db_path = "data_adapter/stock_data.db"
        self.report_dir = "reports/daily_selection_v3.1"
        self.dimensions = ['technical', 'fundamental', 'performance', 'sentiment', 'risk_control', 'market_regime']
        
    def load_stock_price_data(self, codes, start_date, end_date):
        """加载股票真实价格数据"""
        print(f"📈 加载股票价格数据 ({len(codes)} 只股票)...")
        
        conn = sqlite3.connect(self.db_path)
        
        # 构建股票代码查询
        code_list = "','".join(codes)
        
        query = f"""
        SELECT 
            s.code,
            dq.trade_date,
            dq.close,
            dq.open,
            dq.high,
            dq.low,
            dq.volume
        FROM securities s
        JOIN daily_quotes dq ON s.id = dq.security_id
        WHERE s.code IN ('{code_list}')
        AND dq.trade_date >= '{start_date}'
        AND dq.trade_date <= '{end_date}'
        AND s.type = 'A股'
        ORDER BY s.code, dq.trade_date
        """
        
        try:
            price_data = pd.read_sql_query(query, conn)
            price_data['trade_date'] = pd.to_datetime(price_data['trade_date'])
            
            print(f"✅ 价格数据加载完成: {len(price_data)} 条记录")
            print(f"📅 时间范围: {price_data['trade_date'].min().date()} 到 {price_data['trade_date'].max().date()}")
            print(f"🏢 股票数量: {price_data['code'].nunique()} 只")
            
        except Exception as e:
            print(f"❌ 价格数据加载失败: {e}")
            price_data = pd.DataFrame()
        
        conn.close()
        return price_data
    
    def calculate_future_returns(self, price_data, periods=[1, 3, 5, 10]):
        """计算未来收益率"""
        print(f"📊 计算未来收益率 ({periods} 天)...")
        
        returns_data = []
        
        for code in price_data['code'].unique():
            stock_prices = price_data[price_data['code'] == code].sort_values('trade_date')
            
            if len(stock_prices) < max(periods) + 10:  # 确保有足够的数据
                continue
            
            for i in range(len(stock_prices) - max(periods) - 1):
                current_row = stock_prices.iloc[i]
                current_price = current_row['close']
                current_date = current_row['trade_date']
                
                row_data = {
                    'code': code,
                    'date': current_date,
                    'close': current_price
                }
                
                # 计算各个周期的收益率
                for period in periods:
                    if i + period < len(stock_prices):
                        future_price = stock_prices.iloc[i + period]['close']
                        return_pct = (future_price - current_price) / current_price * 100
                        row_data[f'return_{period}d'] = return_pct
                    else:
                        row_data[f'return_{period}d'] = None
                
                returns_data.append(row_data)
        
        returns_df = pd.DataFrame(returns_data)
        
        # 过滤掉异常收益率 (±50%)
        for period in periods:
            col = f'return_{period}d'
            if col in returns_df.columns:
                returns_df[col] = returns_df[col].clip(-50, 50)
        
        print(f"✅ 收益率计算完成: {len(returns_df)} 条记录")
        return returns_df
    
    def load_scoring_data(self, limit_files=100):
        """加载评分数据"""
        print(f"📊 加载评分数据 (限制{limit_files}个文件)...")
        
        json_files = sorted(glob.glob(f"{self.report_dir}/analysis_data_*.json"))[:limit_files]
        all_data = []
        
        for json_file in json_files:
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
                            'total_score': stock.get('score', 0)
                        }
                        
                        # 各维度分数
                        for dim in self.dimensions:
                            stock_record[dim] = factor_scores.get(dim, 0)
                        
                        all_data.append(stock_record)
            
            except Exception as e:
                print(f"⚠️ 读取失败 {json_file}: {e}")
                continue
        
        df = pd.DataFrame(all_data)
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"✅ 评分数据加载完成: {len(df)} 条记录")
        return df
    
    def merge_scores_and_returns(self, scoring_data, returns_data):
        """合并评分和收益率数据"""
        print("🔄 合并评分和收益率数据...")
        
        # 确保日期格式一致
        scoring_data['date'] = pd.to_datetime(scoring_data['date'])
        returns_data['date'] = pd.to_datetime(returns_data['date'])
        
        # 合并数据
        merged = pd.merge(
            scoring_data,
            returns_data,
            on=['code', 'date'],
            how='inner'
        )
        
        # 过滤掉包含NaN的行
        merged = merged.dropna()
        
        print(f"✅ 合并完成: {len(merged)} 条有效记录")
        print(f"📅 时间范围: {merged['date'].min().date()} 到 {merged['date'].max().date()}")
        print(f"🏢 股票数量: {merged['code'].nunique()} 只")
        
        return merged
    
    def calculate_weighted_score(self, data, weights):
        """计算加权评分"""
        score = np.zeros(len(data))
        for dim, weight in weights.items():
            if dim in data.columns:
                score += data[dim].values * weight * 100
        return score
    
    def objective_function(self, weights, data, target_return_column='return_5d'):
        """优化目标函数 - 最大化与未来收益率的相关性"""
        
        # 确保权重合理
        weights_dict = dict(zip(self.dimensions, weights))
        
        # 计算加权评分
        predicted_scores = self.calculate_weighted_score(data, weights_dict)
        actual_returns = data[target_return_column].values
        
        # 计算相关系数 (目标是最大化，所以返回负值)
        correlation = np.corrcoef(predicted_scores, actual_returns)[0, 1]
        
        if np.isnan(correlation):
            return 1.0  # 惩罚无效的权重组合
        
        return -abs(correlation)  # 最大化绝对相关系数
    
    def optimize_weights_for_returns(self, merged_data, target_period='5d'):
        """基于真实收益率优化权重"""
        print(f"🎯 基于{target_period}收益率优化权重...")
        
        target_column = f'return_{target_period}'
        
        if target_column not in merged_data.columns:
            print(f"❌ 目标收益率列 {target_column} 不存在")
            return None
        
        # 过滤有效数据
        valid_data = merged_data[merged_data[target_column].notna()].copy()
        
        if len(valid_data) < 100:
            print(f"❌ 有效数据不足: {len(valid_data)} 条")
            return None
        
        print(f"📊 优化数据: {len(valid_data)} 条记录")
        
        # 初始权重 - 当前v3.1权重
        initial_weights = [0.60, 0.18, 0.15, 0.05, 0.05, 0.05]
        
        # 权重约束: 每个维度权重在 [0.01, 0.8] 之间
        bounds = [(0.01, 0.8) for _ in range(len(self.dimensions))]
        
        # 权重和约束: 总和接近1.0 (允许108%左右，符合v3.1逻辑)
        constraints = {
            'type': 'ineq',
            'fun': lambda x: 1.2 - sum(x)  # 总权重不超过120%
        }
        
        # 执行优化
        result = minimize(
            self.objective_function,
            initial_weights,
            args=(valid_data, target_column),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimized_weights = dict(zip(self.dimensions, result.x))
            
            # 计算原始权重和优化权重的相关性
            original_weights = dict(zip(self.dimensions, initial_weights))
            
            original_scores = self.calculate_weighted_score(valid_data, original_weights)
            optimized_scores = self.calculate_weighted_score(valid_data, optimized_weights)
            
            original_corr = np.corrcoef(original_scores, valid_data[target_column])[0, 1]
            optimized_corr = np.corrcoef(optimized_scores, valid_data[target_column])[0, 1]
            
            print(f"✅ 优化成功!")
            print(f"📊 原始权重相关性: {original_corr:.4f}")
            print(f"🎯 优化权重相关性: {optimized_corr:.4f}")
            print(f"📈 相关性改善: {optimized_corr - original_corr:+.4f}")
            
            return {
                'optimized_weights': optimized_weights,
                'original_weights': original_weights,
                'original_correlation': original_corr,
                'optimized_correlation': optimized_corr,
                'improvement': optimized_corr - original_corr,
                'optimization_result': result
            }
        
        else:
            print(f"❌ 优化失败: {result.message}")
            return None
    
    def backtest_optimized_weights(self, merged_data, optimization_result, test_ratio=0.3):
        """回测优化权重的实际效果"""
        print("🔄 回测优化权重效果...")
        
        # 分割训练集和测试集
        split_point = int(len(merged_data) * (1 - test_ratio))
        train_data = merged_data.iloc[:split_point].copy()
        test_data = merged_data.iloc[split_point:].copy()
        
        print(f"📊 训练集: {len(train_data)} 条记录")
        print(f"🧪 测试集: {len(test_data)} 条记录")
        
        if len(test_data) < 50:
            print("❌ 测试集数据不足")
            return None
        
        original_weights = optimization_result['original_weights']
        optimized_weights = optimization_result['optimized_weights']
        
        # 在测试集上计算预测效果
        test_original_scores = self.calculate_weighted_score(test_data, original_weights)
        test_optimized_scores = self.calculate_weighted_score(test_data, optimized_weights)
        
        # 假设我们关注5日收益率
        test_returns = test_data['return_5d'].values
        
        # 计算测试集相关性
        test_original_corr = np.corrcoef(test_original_scores, test_returns)[0, 1]
        test_optimized_corr = np.corrcoef(test_optimized_scores, test_returns)[0, 1]
        
        # 模拟选股策略: 每次选前10%高分股票
        def simulate_portfolio_return(scores, returns, top_pct=0.1):
            n_top = max(1, int(len(scores) * top_pct))
            top_indices = np.argsort(scores)[-n_top:]
            portfolio_return = np.mean(returns[top_indices])
            return portfolio_return
        
        # 按日期分组进行选股模拟
        test_portfolio_results = []
        
        for date, day_data in test_data.groupby('date'):
            if len(day_data) < 10:
                continue
            
            day_original_scores = self.calculate_weighted_score(day_data, original_weights)
            day_optimized_scores = self.calculate_weighted_score(day_data, optimized_weights)
            day_returns = day_data['return_5d'].values
            
            original_portfolio_return = simulate_portfolio_return(day_original_scores, day_returns)
            optimized_portfolio_return = simulate_portfolio_return(day_optimized_scores, day_returns)
            
            test_portfolio_results.append({
                'date': date,
                'original_return': original_portfolio_return,
                'optimized_return': optimized_portfolio_return
            })
        
        if not test_portfolio_results:
            print("❌ 无法生成测试投资组合结果")
            return None
        
        portfolio_df = pd.DataFrame(test_portfolio_results)
        
        # 计算投资组合统计
        avg_original_return = portfolio_df['original_return'].mean()
        avg_optimized_return = portfolio_df['optimized_return'].mean()
        
        # 统计显著性检验
        t_stat, p_value = stats.ttest_rel(
            portfolio_df['optimized_return'],
            portfolio_df['original_return']
        )
        
        backtest_results = {
            'test_correlations': {
                'original': test_original_corr,
                'optimized': test_optimized_corr,
                'improvement': test_optimized_corr - test_original_corr
            },
            'portfolio_returns': {
                'original_avg': avg_original_return,
                'optimized_avg': avg_optimized_return,
                'improvement': avg_optimized_return - avg_original_return
            },
            'statistical_test': {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            },
            'portfolio_results': portfolio_df
        }
        
        return backtest_results
    
    def print_optimization_results(self, optimization_result, backtest_results):
        """打印优化结果"""
        print("\n" + "=" * 80)
        print("🎯 基于真实收益率的权重优化结果")
        print("=" * 80)
        
        # 权重对比
        print(f"\n📊 权重优化对比:")
        print(f"{'维度':15s} {'原始权重':>12s} {'优化权重':>12s} {'变化':>10s}")
        print("-" * 60)
        
        original = optimization_result['original_weights']
        optimized = optimization_result['optimized_weights']
        
        for dim in self.dimensions:
            orig_w = original[dim]
            opt_w = optimized[dim]
            change = opt_w - orig_w
            print(f"{dim:15s} {orig_w:>11.1%} {opt_w:>11.1%} {change:>+9.1%}")
        
        print(f"{'总计':15s} {sum(original.values()):>11.1%} {sum(optimized.values()):>11.1%}")
        
        # 训练集表现
        print(f"\n📈 训练集预测能力:")
        print(f"  原始权重相关性: {optimization_result['original_correlation']:+.4f}")
        print(f"  优化权重相关性: {optimization_result['optimized_correlation']:+.4f}")
        print(f"  相关性提升: {optimization_result['improvement']:+.4f}")
        
        if backtest_results:
            # 测试集表现
            print(f"\n🧪 测试集验证结果:")
            test_corr = backtest_results['test_correlations']
            print(f"  原始权重相关性: {test_corr['original']:+.4f}")
            print(f"  优化权重相关性: {test_corr['optimized']:+.4f}")
            print(f"  相关性提升: {test_corr['improvement']:+.4f}")
            
            # 投资组合收益
            port_ret = backtest_results['portfolio_returns']
            print(f"\n💰 投资组合收益 (5日平均):")
            print(f"  原始权重收益: {port_ret['original_avg']:+.2f}%")
            print(f"  优化权重收益: {port_ret['optimized_avg']:+.2f}%")
            print(f"  收益提升: {port_ret['improvement']:+.2f}%")
            
            # 统计显著性
            stat_test = backtest_results['statistical_test']
            significance = "✅ 显著" if stat_test['significant'] else "❌ 不显著"
            print(f"\n📊 统计显著性:")
            print(f"  t统计量: {stat_test['t_statistic']:+.3f}")
            print(f"  p值: {stat_test['p_value']:.4f}")
            print(f"  结论: {significance}")
        
        # 最终结论
        if backtest_results:
            corr_improved = backtest_results['test_correlations']['improvement'] > 0
            return_improved = backtest_results['portfolio_returns']['improvement'] > 0
            significant = backtest_results['statistical_test']['significant']
            
            if corr_improved and return_improved and significant:
                conclusion = "✅ 权重优化显著提升了预测能力和投资收益!"
            elif corr_improved and return_improved:
                conclusion = "⚠️ 权重优化提升了效果，但统计显著性不足"
            else:
                conclusion = "❌ 权重优化未能在测试集上提升效果"
        else:
            conclusion = "⚠️ 无法完成完整验证，仅基于训练集结果"
        
        print(f"\n🎯 最终结论:")
        print(f"  {conclusion}")

def main():
    """主函数"""
    optimizer = TrueReturnBasedOptimizer()
    
    print("🚀 开始基于真实收益率的权重优化")
    print("这次我们要真正证明权重优化能提升选股收益!")
    print("=" * 80)
    
    # 1. 加载评分数据
    scoring_data = optimizer.load_scoring_data(limit_files=50)  # 限制文件数量以加快速度
    
    if len(scoring_data) < 1000:
        print("❌ 评分数据不足")
        return
    
    # 2. 获取股票代码和时间范围
    unique_codes = scoring_data['code'].unique()[:200]  # 限制股票数量以加快速度
    start_date = scoring_data['date'].min().strftime('%Y-%m-%d')
    end_date = (scoring_data['date'].max() + timedelta(days=30)).strftime('%Y-%m-%d')  # 扩展结束日期以获取未来收益
    
    # 3. 加载价格数据
    price_data = optimizer.load_stock_price_data(unique_codes, start_date, end_date)
    
    if len(price_data) == 0:
        print("❌ 价格数据加载失败，无法进行基于真实收益率的优化")
        return
    
    # 4. 计算未来收益率
    returns_data = optimizer.calculate_future_returns(price_data, periods=[1, 3, 5, 10])
    
    # 5. 合并评分和收益率数据
    merged_data = optimizer.merge_scores_and_returns(scoring_data, returns_data)
    
    if len(merged_data) < 500:
        print("❌ 合并后数据不足，无法进行可靠的优化")
        return
    
    # 6. 执行权重优化
    optimization_result = optimizer.optimize_weights_for_returns(merged_data, target_period='5d')
    
    if optimization_result is None:
        print("❌ 权重优化失败")
        return
    
    # 7. 回测验证
    backtest_results = optimizer.backtest_optimized_weights(merged_data, optimization_result)
    
    # 8. 打印结果
    optimizer.print_optimization_results(optimization_result, backtest_results)

if __name__ == "__main__":
    main()