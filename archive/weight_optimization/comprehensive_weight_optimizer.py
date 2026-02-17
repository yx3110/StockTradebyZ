#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全面权重优化器 - 按照用户建议的系统性方案
1. 计算所有股票在2023-2025全时间段的指标
2. 保存到本地数据库 
3. 用不同权重参数进行全时段回归验证
4. 找到胜率最高的权重组合
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
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class ComprehensiveWeightOptimizer:
    """全面权重优化器"""
    
    def __init__(self):
        self.db_path = "data_adapter/stock_data.db"
        self.report_dir = "reports/daily_selection_v3.1"
        self.cache_db_path = "weight_optimization_cache.db"
        self.dimensions = ['technical', 'fundamental', 'performance', 'sentiment', 'risk_control', 'market_regime']
        
        # 初始化缓存数据库
        self.init_cache_database()
        
    def init_cache_database(self):
        """初始化缓存数据库"""
        print("🗄️ 初始化权重优化缓存数据库...")
        
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        # 创建股票指标表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_indicators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT NOT NULL,
            date TEXT NOT NULL,
            technical REAL,
            fundamental REAL,
            performance REAL,
            sentiment REAL,
            risk_control REAL,
            market_regime REAL,
            total_score REAL,
            return_1d REAL,
            return_3d REAL,
            return_5d REAL,
            return_10d REAL,
            return_20d REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(code, date)
        )
        ''')
        
        # 创建权重测试结果表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS weight_test_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            weight_config TEXT,
            test_period TEXT,
            correlation_1d REAL,
            correlation_3d REAL,
            correlation_5d REAL,
            correlation_10d REAL,
            correlation_20d REAL,
            win_rate_1d REAL,
            win_rate_3d REAL,
            win_rate_5d REAL,
            win_rate_10d REAL,
            win_rate_20d REAL,
            avg_return_1d REAL,
            avg_return_3d REAL,
            avg_return_5d REAL,
            avg_return_10d REAL,
            avg_return_20d REAL,
            sharpe_ratio_5d REAL,
            max_drawdown_5d REAL,
            total_samples INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ 缓存数据库初始化完成")
    
    def load_all_historical_data(self, start_year=2023, end_year=2025):
        """加载全时间段历史数据"""
        print(f"📊 加载 {start_year}-{end_year} 全时间段历史数据...")
        
        # 1. 加载所有评分数据
        scoring_data = self.load_comprehensive_scoring_data()
        
        # 2. 加载所有价格数据并计算收益率
        returns_data = self.load_comprehensive_returns_data(scoring_data, start_year, end_year)
        
        # 3. 合并数据
        merged_data = self.merge_scoring_and_returns(scoring_data, returns_data)
        
        # 4. 保存到缓存数据库
        self.save_to_cache_database(merged_data)
        
        return merged_data
    
    def load_comprehensive_scoring_data(self):
        """加载全面的评分数据"""
        print("📈 加载全面评分数据...")
        
        json_files = sorted(glob.glob(f"{self.report_dir}/analysis_data_*.json"))
        print(f"找到 {len(json_files)} 个评分文件")
        
        all_data = []
        processed_count = 0
        
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
                
                processed_count += 1
                if processed_count % 20 == 0:
                    print(f"  已处理 {processed_count}/{len(json_files)} 个文件...")
            
            except Exception as e:
                print(f"⚠️ 读取失败 {json_file}: {e}")
                continue
        
        df = pd.DataFrame(all_data)
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"✅ 评分数据加载完成: {len(df)} 条记录")
        print(f"📅 时间范围: {df['date'].min().date()} 到 {df['date'].max().date()}")
        print(f"🏢 股票数量: {df['code'].nunique()} 只")
        
        return df
    
    def load_comprehensive_returns_data(self, scoring_data, start_year, end_year):
        """加载全面的收益率数据"""
        print("💰 计算全面收益率数据...")
        
        # 获取所有股票代码
        all_codes = scoring_data['code'].unique()
        print(f"需要计算 {len(all_codes)} 只股票的收益率")
        
        # 扩展时间范围以获取完整的未来收益率
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"
        
        # 加载价格数据
        conn = sqlite3.connect(self.db_path)
        
        # 分批查询以避免内存问题
        batch_size = 200
        all_returns = []
        
        for i in range(0, len(all_codes), batch_size):
            batch_codes = all_codes[i:i+batch_size]
            print(f"  处理第 {i//batch_size + 1} 批股票 ({len(batch_codes)} 只)...")
            
            code_list = "','".join(batch_codes)
            
            query = f"""
            SELECT 
                s.code,
                dq.trade_date,
                dq.close
            FROM securities s
            JOIN daily_quotes dq ON s.id = dq.security_id
            WHERE s.code IN ('{code_list}')
            AND dq.trade_date >= '{start_date}'
            AND dq.trade_date <= '{end_date}'
            AND s.type = 'A股'
            ORDER BY s.code, dq.trade_date
            """
            
            try:
                batch_data = pd.read_sql_query(query, conn)
                batch_data['trade_date'] = pd.to_datetime(batch_data['trade_date'])
                
                # 计算收益率
                batch_returns = self.calculate_returns_for_batch(batch_data)
                all_returns.append(batch_returns)
                
            except Exception as e:
                print(f"⚠️ 批次 {i//batch_size + 1} 处理失败: {e}")
                continue
        
        conn.close()
        
        # 合并所有批次
        if all_returns:
            returns_df = pd.concat(all_returns, ignore_index=True)
            print(f"✅ 收益率计算完成: {len(returns_df)} 条记录")
        else:
            print("❌ 没有成功计算任何收益率数据")
            returns_df = pd.DataFrame()
        
        return returns_df
    
    def calculate_returns_for_batch(self, price_data, periods=[1, 3, 5, 10, 20]):
        """为一批股票计算收益率"""
        returns_data = []
        
        for code in price_data['code'].unique():
            stock_prices = price_data[price_data['code'] == code].sort_values('trade_date')
            
            if len(stock_prices) < max(periods) + 5:
                continue
            
            for i in range(len(stock_prices) - max(periods)):
                current_row = stock_prices.iloc[i]
                current_price = current_row['close']
                current_date = current_row['trade_date']
                
                row_data = {
                    'code': code,
                    'date': current_date
                }
                
                # 计算各个周期的收益率
                for period in periods:
                    if i + period < len(stock_prices):
                        future_price = stock_prices.iloc[i + period]['close']
                        return_pct = (future_price - current_price) / current_price * 100
                        # 限制异常值
                        return_pct = max(-50, min(50, return_pct))
                        row_data[f'return_{period}d'] = return_pct
                
                returns_data.append(row_data)
        
        return pd.DataFrame(returns_data)
    
    def merge_scoring_and_returns(self, scoring_data, returns_data):
        """合并评分和收益率数据"""
        print("🔄 合并评分和收益率数据...")
        
        if returns_data.empty:
            print("❌ 收益率数据为空，无法合并")
            return pd.DataFrame()
        
        # 确保日期格式一致
        scoring_data['date'] = pd.to_datetime(scoring_data['date'])
        returns_data['date'] = pd.to_datetime(returns_data['date'])
        
        # 合并
        merged = pd.merge(
            scoring_data,
            returns_data,
            on=['code', 'date'],
            how='inner'
        )
        
        # 过滤掉包含NaN的行
        merged = merged.dropna()
        
        print(f"✅ 合并完成: {len(merged)} 条有效记录")
        return merged
    
    def save_to_cache_database(self, merged_data):
        """保存到缓存数据库"""
        print("💾 保存数据到缓存数据库...")
        
        if merged_data.empty:
            print("❌ 没有数据可保存")
            return
        
        conn = sqlite3.connect(self.cache_db_path)
        
        # 清空现有数据
        conn.execute("DELETE FROM stock_indicators")
        
        # 保存数据
        merged_data.to_sql('stock_indicators', conn, if_exists='append', index=False, method='multi')
        
        conn.commit()
        conn.close()
        
        print(f"✅ 数据保存完成: {len(merged_data)} 条记录")
    
    def load_from_cache_database(self):
        """从缓存数据库加载数据"""
        print("📂 从缓存数据库加载数据...")
        
        conn = sqlite3.connect(self.cache_db_path)
        
        try:
            data = pd.read_sql_query("SELECT * FROM stock_indicators", conn)
            data['date'] = pd.to_datetime(data['date'])
            print(f"✅ 缓存数据加载完成: {len(data)} 条记录")
        except Exception as e:
            print(f"❌ 缓存数据加载失败: {e}")
            data = pd.DataFrame()
        
        conn.close()
        return data
    
    def generate_weight_combinations(self, step=0.1):
        """生成权重组合进行测试"""
        print(f"🔧 生成权重组合 (步长={step})...")
        
        # 定义每个维度的权重范围
        weight_ranges = {
            'technical': np.arange(0.3, 0.8, step),       # 技术分析 30%-80%
            'fundamental': np.arange(0.05, 0.3, step),    # 基本面 5%-30%
            'performance': np.arange(0.05, 0.3, step),    # 市场表现 5%-30%
            'sentiment': np.arange(0.02, 0.15, step),     # 情绪 2%-15%
            'risk_control': np.arange(0.02, 0.15, step),  # 风控 2%-15%
            'market_regime': np.arange(0.02, 0.15, step)  # 市场环境 2%-15%
        }
        
        # 生成所有组合
        combinations = []
        ranges_values = list(weight_ranges.values())
        
        for combo in itertools.product(*ranges_values):
            weights = dict(zip(self.dimensions, combo))
            
            # 过滤权重和在合理范围内的组合
            total_weight = sum(weights.values())
            if 0.9 <= total_weight <= 1.2:  # 允许90%-120%的权重总和
                combinations.append(weights)
        
        print(f"✅ 生成 {len(combinations)} 个权重组合")
        return combinations
    
    def test_weight_combination(self, data, weights, periods=[1, 3, 5, 10, 20]):
        """测试单个权重组合的效果"""
        
        # 计算加权评分
        weighted_score = np.zeros(len(data))
        for dim, weight in weights.items():
            if dim in data.columns:
                weighted_score += data[dim].values * weight * 100
        
        results = {}
        
        # 测试各个周期的表现
        for period in periods:
            return_col = f'return_{period}d'
            if return_col in data.columns:
                actual_returns = data[return_col].values
                
                # 过滤有效数据
                valid_mask = ~np.isnan(actual_returns)
                if valid_mask.sum() < 50:
                    continue
                
                valid_scores = weighted_score[valid_mask]
                valid_returns = actual_returns[valid_mask]
                
                # 计算相关性
                correlation = np.corrcoef(valid_scores, valid_returns)[0, 1]
                if np.isnan(correlation):
                    correlation = 0
                
                # 计算胜率 - 按评分分组看高分组的胜率
                try:
                    # 取前20%高分股票
                    top_20_threshold = np.percentile(valid_scores, 80)
                    top_20_mask = valid_scores >= top_20_threshold
                    
                    if top_20_mask.sum() > 0:
                        top_20_returns = valid_returns[top_20_mask]
                        win_rate = (top_20_returns > 0).mean()
                        avg_return = top_20_returns.mean()
                    else:
                        win_rate = 0
                        avg_return = 0
                        
                except:
                    win_rate = 0
                    avg_return = 0
                
                results[f'correlation_{period}d'] = correlation
                results[f'win_rate_{period}d'] = win_rate
                results[f'avg_return_{period}d'] = avg_return
        
        # 计算综合指标
        if 'return_5d' in data.columns:
            # 5日夏普比率和最大回撤
            return_5d = data['return_5d'].values
            valid_mask = ~np.isnan(return_5d)
            
            if valid_mask.sum() > 50:
                valid_scores = weighted_score[valid_mask]
                valid_returns = return_5d[valid_mask]
                
                # 按评分排序，模拟投资组合
                sorted_indices = np.argsort(valid_scores)
                top_10_pct = sorted_indices[-len(sorted_indices)//10:]  # 前10%
                portfolio_returns = valid_returns[top_10_pct]
                
                if len(portfolio_returns) > 10:
                    sharpe_ratio = portfolio_returns.mean() / max(portfolio_returns.std(), 0.1)
                    
                    # 简化的最大回撤计算
                    cumulative = np.cumprod(1 + portfolio_returns/100)
                    running_max = np.maximum.accumulate(cumulative)
                    drawdown = (cumulative - running_max) / running_max
                    max_drawdown = drawdown.min()
                else:
                    sharpe_ratio = 0
                    max_drawdown = 0
            else:
                sharpe_ratio = 0
                max_drawdown = 0
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        results['sharpe_ratio_5d'] = sharpe_ratio
        results['max_drawdown_5d'] = max_drawdown
        results['total_samples'] = len(data)
        
        return results
    
    def batch_test_weights(self, data, weight_combinations, batch_size=100, max_workers=4):
        """批量测试权重组合"""
        print(f"🧪 批量测试权重组合 ({len(weight_combinations)} 个组合)...")
        
        results = []
        
        # 分批处理
        for i in range(0, len(weight_combinations), batch_size):
            batch = weight_combinations[i:i+batch_size]
            print(f"  测试第 {i//batch_size + 1} 批 ({len(batch)} 个组合)...")
            
            # 并行测试
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.test_weight_combination, data, weights): weights 
                    for weights in batch
                }
                
                for future in as_completed(futures):
                    weights = futures[future]
                    try:
                        result = future.result()
                        result['weights'] = weights
                        results.append(result)
                    except Exception as e:
                        print(f"⚠️ 权重测试失败: {e}")
        
        print(f"✅ 批量测试完成: {len(results)} 个有效结果")
        return results
    
    def analyze_and_rank_results(self, test_results):
        """分析和排序测试结果"""
        print("📊 分析和排序测试结果...")
        
        df_results = pd.DataFrame(test_results)
        
        if df_results.empty:
            print("❌ 没有测试结果可分析")
            return None
        
        # 计算综合评分 - 多个指标的加权平均
        # 重点关注5日收益率的相关性、胜率和夏普比率
        df_results['composite_score'] = (
            df_results.get('correlation_5d', 0).abs() * 0.3 +  # 预测能力 30%
            df_results.get('win_rate_5d', 0) * 0.3 +           # 胜率 30%
            np.clip(df_results.get('sharpe_ratio_5d', 0), -5, 5) * 0.1 * 0.2 +  # 夏普比率 20%
            (1 + df_results.get('max_drawdown_5d', 0)) * 0.1 +  # 回撤控制 10%
            df_results.get('avg_return_5d', 0) * 0.01 * 0.1     # 平均收益 10%
        )
        
        # 按综合评分排序
        df_results = df_results.sort_values('composite_score', ascending=False)
        
        print(f"✅ 结果分析完成，找到 {len(df_results)} 个有效权重方案")
        
        return df_results
    
    def print_optimization_results(self, ranked_results, top_n=5):
        """打印优化结果"""
        print("\n" + "=" * 100)
        print("🎯 全面权重优化结果 - 基于全时段回归验证")
        print("=" * 100)
        
        if ranked_results is None or len(ranked_results) == 0:
            print("❌ 没有可显示的结果")
            return
        
        print(f"\n📊 Top {top_n} 权重方案:")
        
        for i in range(min(top_n, len(ranked_results))):
            result = ranked_results.iloc[i]
            weights = result['weights']
            
            print(f"\n🏆 第 {i+1} 名 (综合评分: {result['composite_score']:.4f})")
            print("-" * 80)
            
            # 权重分布
            print("权重分布:")
            total_weight = 0
            for dim in self.dimensions:
                weight = weights[dim]
                total_weight += weight
                print(f"  {dim:15s}: {weight:6.1%}")
            print(f"  {'总计':15s}: {total_weight:6.1%}")
            
            # 关键性能指标
            print("\n关键性能指标:")
            metrics = [
                ('5日相关性', 'correlation_5d', '.4f'),
                ('5日胜率', 'win_rate_5d', '.1%'),
                ('5日平均收益', 'avg_return_5d', '.2f%'),
                ('5日夏普比率', 'sharpe_ratio_5d', '.3f'),
                ('5日最大回撤', 'max_drawdown_5d', '.1%'),
                ('样本数量', 'total_samples', 'd')
            ]
            
            for name, key, fmt in metrics:
                value = result.get(key, 0)
                if key == 'max_drawdown_5d':
                    value = abs(value)
                print(f"  {name:15s}: {value:{fmt}}")
        
        # 保存最佳权重
        best_weights = ranked_results.iloc[0]['weights']
        self.save_best_weights(best_weights, ranked_results.iloc[0])
    
    def save_best_weights(self, best_weights, best_result):
        """保存最佳权重配置"""
        print("\n💾 保存最佳权重配置...")
        
        config = {
            "version": "v3.1-ComprehensiveOptimized",
            "description": "基于2023-2025全时段数据的综合权重优化结果",
            "optimization_info": {
                "method": "comprehensive_regression",
                "optimization_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "composite_score": best_result['composite_score'],
                "total_samples": int(best_result.get('total_samples', 0))
            },
            "optimized_weights": best_weights,
            "performance_metrics": {
                "correlation_5d": best_result.get('correlation_5d', 0),
                "win_rate_5d": best_result.get('win_rate_5d', 0),
                "avg_return_5d": best_result.get('avg_return_5d', 0),
                "sharpe_ratio_5d": best_result.get('sharpe_ratio_5d', 0),
                "max_drawdown_5d": abs(best_result.get('max_drawdown_5d', 0))
            },
            "expected_improvements": [
                "基于全时段数据的系统性优化",
                f"5日胜率提升至 {best_result.get('win_rate_5d', 0):.1%}",
                f"预测相关性达到 {best_result.get('correlation_5d', 0):.4f}",
                "经过大规模回归验证的稳健权重"
            ]
        }
        
        # 保存配置
        os.makedirs("scoring/v3/comprehensive_optimization", exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_file = f"scoring/v3/comprehensive_optimization/comprehensive_optimized_config_{timestamp}.json"
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 最佳权重配置已保存: {config_file}")

def main():
    """主函数"""
    optimizer = ComprehensiveWeightOptimizer()
    
    print("🚀 全面权重优化器 - 基于用户建议的系统性方案")
    print("=" * 80)
    print("方案:")
    print("1. 计算所有股票2023-2025全时段指标")
    print("2. 保存到本地缓存数据库") 
    print("3. 用不同权重参数进行全时段回归")
    print("4. 找到胜率最高的权重组合")
    print("=" * 80)
    
    # 检查是否有缓存数据
    cached_data = optimizer.load_from_cache_database()
    
    if len(cached_data) < 1000:
        print("📊 缓存数据不足，开始加载和计算全时段历史数据...")
        merged_data = optimizer.load_all_historical_data()
    else:
        print("📂 使用缓存数据进行优化...")
        merged_data = cached_data
    
    if merged_data.empty or len(merged_data) < 1000:
        print("❌ 数据不足，无法进行优化")
        return
    
    print(f"\n📈 开始权重优化，数据量: {len(merged_data)} 条记录")
    
    # 生成权重组合
    weight_combinations = optimizer.generate_weight_combinations(step=0.15)  # 使用较大步长加快速度
    
    # 批量测试权重
    test_results = optimizer.batch_test_weights(merged_data, weight_combinations)
    
    # 分析和排序结果
    ranked_results = optimizer.analyze_and_rank_results(test_results)
    
    # 打印结果
    optimizer.print_optimization_results(ranked_results)

if __name__ == "__main__":
    main()