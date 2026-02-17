#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合权重优化器 v3.0版本 - 基于真实历史收益率的系统性权重优化
采用用户建议的方法：
1. 计算所有股票2023-2025全时段指标
2. 保存到本地缓存数据库
3. 用不同权重参数进行全时段回归
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
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
import itertools

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class ComprehensiveWeightOptimizerV30:
    """综合权重优化器 v3.0版本"""
    
    def __init__(self):
        self.cache_db_path = "weight_optimization_cache_v30.db"
        self.report_dir = "reports/daily_selection_v3"
        self.main_db_path = "data_adapter/stock_data.db"
        
    def create_cache_database(self):
        """创建缓存数据库"""
        print("🗄️ 初始化v3.0权重优化缓存数据库...")
        
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        # 创建股票指标表 (v3.0只有4个维度)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_indicators_v30 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT NOT NULL,
            date TEXT NOT NULL,
            technical REAL,
            fundamental REAL, 
            performance REAL,
            risk_control REAL,
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
        CREATE TABLE IF NOT EXISTS weight_test_results_v30 (
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
        print("✅ v3.0缓存数据库初始化完成")
    
    def load_cached_data(self):
        """从缓存加载数据"""
        print("📂 从v3.0缓存数据库加载数据...")
        
        conn = sqlite3.connect(self.cache_db_path)
        query = """
        SELECT code, date, technical, fundamental, performance, risk_control,
               return_1d, return_3d, return_5d, return_10d, return_20d
        FROM stock_indicators_v30
        WHERE return_5d IS NOT NULL
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"✅ v3.0缓存数据加载完成: {len(df)} 条记录")
        return df
    
    def load_v30_scoring_data(self):
        """加载v3.0评分数据"""
        print("📈 加载v3.0全面评分数据...")
        
        json_files = glob.glob(f"{self.report_dir}/analysis_data_*.json")
        print(f"找到 {len(json_files)} 个v3.0评分文件")
        
        all_data = []
        
        for i, json_file in enumerate(sorted(json_files)):
            if i % 20 == 0:
                print(f"  已处理 {i+1}/{len(json_files)} 个文件...")
                
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
                        
                        # v3.0只有4个维度
                        for dim in ['technical', 'fundamental', 'performance', 'risk_control']:
                            stock_record[dim] = factor_scores.get(dim, 0)
                        
                        all_data.append(stock_record)
            
            except Exception as e:
                print(f"⚠️ 读取文件失败 {json_file}: {e}")
                continue
        
        df = pd.DataFrame(all_data)
        print(f"✅ v3.0评分数据加载完成: {len(df)} 条记录")
        print(f"📅 时间范围: {df['date'].min()} 到 {df['date'].max()}")
        print(f"🏢 股票数量: {df['code'].nunique()} 只")
        
        return df
    
    def calculate_stock_returns(self, stock_codes, start_date, end_date):
        """批量计算股票收益率"""
        print("💰 计算v3.0全面收益率数据...")
        
        conn = sqlite3.connect(self.main_db_path)
        
        # 分批处理股票代码
        batch_size = 200
        all_returns = []
        
        for i in range(0, len(stock_codes), batch_size):
            batch_codes = stock_codes[i:i+batch_size]
            print(f"  处理第 {i//batch_size + 1} 批股票 ({len(batch_codes)} 只)...")
            
            code_list = "','".join(batch_codes)
            
            query = f"""
            WITH stock_prices AS (
                SELECT 
                    s.code,
                    dq.trade_date,
                    dq.close as price,
                    LAG(dq.close, 1) OVER (PARTITION BY s.code ORDER BY dq.trade_date) as price_1d_ago,
                    LAG(dq.close, 3) OVER (PARTITION BY s.code ORDER BY dq.trade_date) as price_3d_ago,
                    LAG(dq.close, 5) OVER (PARTITION BY s.code ORDER BY dq.trade_date) as price_5d_ago,
                    LAG(dq.close, 10) OVER (PARTITION BY s.code ORDER BY dq.trade_date) as price_10d_ago,
                    LAG(dq.close, 20) OVER (PARTITION BY s.code ORDER BY dq.trade_date) as price_20d_ago
                FROM securities s
                JOIN daily_quotes dq ON s.id = dq.security_id
                WHERE s.code IN ('{code_list}')
                AND dq.trade_date >= '{start_date}'
                AND dq.trade_date <= '{end_date}'
                ORDER BY s.code, dq.trade_date
            )
            SELECT 
                code,
                trade_date as date,
                CASE WHEN price_1d_ago IS NOT NULL THEN (price - price_1d_ago) / price_1d_ago * 100 ELSE NULL END as return_1d,
                CASE WHEN price_3d_ago IS NOT NULL THEN (price - price_3d_ago) / price_3d_ago * 100 ELSE NULL END as return_3d,
                CASE WHEN price_5d_ago IS NOT NULL THEN (price - price_5d_ago) / price_5d_ago * 100 ELSE NULL END as return_5d,
                CASE WHEN price_10d_ago IS NOT NULL THEN (price - price_10d_ago) / price_10d_ago * 100 ELSE NULL END as return_10d,
                CASE WHEN price_20d_ago IS NOT NULL THEN (price - price_20d_ago) / price_20d_ago * 100 ELSE NULL END as return_20d
            FROM stock_prices
            WHERE price_1d_ago IS NOT NULL
            """
            
            batch_returns = pd.read_sql_query(query, conn)
            all_returns.append(batch_returns)
        
        conn.close()
        
        # 合并所有批次
        returns_df = pd.concat(all_returns, ignore_index=True)
        print(f"✅ v3.0收益率计算完成: {len(returns_df)} 条记录")
        
        return returns_df
    
    def load_and_cache_data(self):
        """加载并缓存全面历史数据"""
        print("🚀 v3.0全面权重优化器 - 基于用户建议的系统性方案")
        print("="*80)
        print("方案:")
        print("1. 计算所有股票2023-2025全时段v3.0指标") 
        print("2. 保存到本地缓存数据库")
        print("3. 用不同权重参数进行全时段回归")
        print("4. 找到胜率最高的权重组合")
        print("="*80)
        
        # 检查缓存
        cached_data = self.load_cached_data()
        if len(cached_data) > 1000:
            print(f"📊 使用缓存数据: {len(cached_data)} 条记录")
            return cached_data
        
        print("📊 缓存数据不足，开始加载和计算v3.0全时段历史数据...")
        
        # 1. 加载评分数据
        scoring_data = self.load_v30_scoring_data()
        
        # 2. 计算收益率数据
        unique_codes = scoring_data['code'].unique()
        start_date = scoring_data['date'].min()
        end_date = scoring_data['date'].max()
        
        returns_data = self.calculate_stock_returns(unique_codes, start_date, end_date)
        
        # 3. 合并数据
        print("🔄 合并v3.0评分和收益率数据...")
        merged_data = pd.merge(
            scoring_data,
            returns_data,
            on=['code', 'date'],
            how='inner'
        )
        
        print(f"✅ v3.0合并完成: {len(merged_data)} 条有效记录")
        
        # 4. 保存到缓存
        print("💾 保存v3.0数据到缓存数据库...")
        conn = sqlite3.connect(self.cache_db_path)
        
        # 准备插入数据
        cache_data = merged_data[[
            'code', 'date', 'technical', 'fundamental', 'performance', 'risk_control',
            'return_1d', 'return_3d', 'return_5d', 'return_10d', 'return_20d'
        ]].copy()
        
        cache_data.to_sql('stock_indicators_v30', conn, if_exists='replace', index=False)
        conn.close()
        
        print(f"✅ v3.0数据保存完成: {len(cache_data)} 条记录")
        
        return cache_data
    
    def generate_weight_combinations_v30(self, step=0.10):
        """生成v3.0权重组合 (只有4个维度)"""
        print(f"🔧 生成v3.0权重组合 (步长={step})...")
        
        combinations = []
        
        # v3.0维度
        dimensions = ['technical', 'fundamental', 'performance', 'risk_control']
        
        # 生成权重组合，确保总和约等于1.08 (v3.0的总权重)
        target_sum = 1.08
        
        # 使用网格搜索生成合理的权重组合
        weight_ranges = {
            'technical': np.arange(0.4, 0.8, step),      # 技术面: 40%-70%
            'fundamental': np.arange(0.1, 0.4, step),    # 基本面: 10%-30%  
            'performance': np.arange(0.1, 0.4, step),    # 表现: 10%-30%
            'risk_control': np.arange(0.02, 0.15, step)  # 风控: 2%-15%
        }
        
        count = 0
        for tech in weight_ranges['technical']:
            for fund in weight_ranges['fundamental']:
                for perf in weight_ranges['performance']:
                    for risk in weight_ranges['risk_control']:
                        total = tech + fund + perf + risk
                        # 检查总权重是否接近目标
                        if abs(total - target_sum) < 0.05:
                            combination = {
                                'technical': tech,
                                'fundamental': fund,
                                'performance': perf,
                                'risk_control': risk
                            }
                            combinations.append(combination)
                            count += 1
                            
                            if count >= 20:  # 限制组合数量
                                break
                    if count >= 20:
                        break
                if count >= 20:
                    break
            if count >= 20:
                break
        
        # 如果生成的组合不够，添加一些预设的组合
        if len(combinations) < 10:
            predefined = [
                # 原始v3.0权重 (假设)
                {'technical': 0.6, 'fundamental': 0.2, 'performance': 0.2, 'risk_control': 0.08},
                # 技术面主导
                {'technical': 0.7, 'fundamental': 0.15, 'performance': 0.15, 'risk_control': 0.08},
                # 基本面主导
                {'technical': 0.45, 'fundamental': 0.35, 'performance': 0.15, 'risk_control': 0.13},
                # 表现主导
                {'technical': 0.45, 'fundamental': 0.15, 'performance': 0.35, 'risk_control': 0.13},
                # 均衡配置
                {'technical': 0.5, 'fundamental': 0.25, 'performance': 0.25, 'risk_control': 0.08}
            ]
            combinations.extend(predefined)
        
        print(f"✅ 生成 {len(combinations)} 个v3.0权重组合")
        return combinations
    
    def test_weight_combination_v30(self, data, weights, periods=[1, 3, 5, 10, 20]):
        """测试v3.0单个权重组合"""
        # 计算加权评分 (v3.0采用直接相加)
        weighted_score = (
            data['technical'].fillna(0) * weights['technical'] +
            data['fundamental'].fillna(0) * weights['fundamental'] +
            data['performance'].fillna(0) * weights['performance'] +
            data['risk_control'].fillna(0) * weights['risk_control']
        ) * 100
        
        results = {}
        
        for period in periods:
            return_col = f'return_{period}d'
            if return_col in data.columns:
                # 过滤有效数据
                valid_mask = (~data[return_col].isna()) & (~weighted_score.isna())
                valid_data = data[valid_mask].copy()
                valid_scores = weighted_score[valid_mask]
                
                if len(valid_data) < 100:
                    continue
                
                # 计算相关性
                try:
                    corr, p_value = stats.pearsonr(valid_scores, valid_data[return_col])
                    results[f'correlation_{period}d'] = corr
                    results[f'p_value_{period}d'] = p_value
                except:
                    results[f'correlation_{period}d'] = 0
                    results[f'p_value_{period}d'] = 1
                
                # 计算胜率 (前25%高分股票的上涨概率)
                top25_threshold = valid_scores.quantile(0.75)
                top25_mask = valid_scores >= top25_threshold
                top25_returns = valid_data.loc[top25_mask, return_col]
                
                if len(top25_returns) > 0:
                    win_rate = (top25_returns > 0).mean()
                    avg_return = top25_returns.mean()
                    
                    # 计算夏普比率
                    if top25_returns.std() > 0:
                        sharpe_ratio = avg_return / top25_returns.std()
                    else:
                        sharpe_ratio = 0
                        
                    # 计算最大回撤 (简化计算)
                    cumulative = (1 + top25_returns / 100).cumprod()
                    drawdown = (cumulative / cumulative.expanding().max() - 1).min()
                    
                    results[f'win_rate_{period}d'] = win_rate
                    results[f'avg_return_{period}d'] = avg_return
                    results[f'sharpe_ratio_{period}d'] = sharpe_ratio
                    results[f'max_drawdown_{period}d'] = drawdown
                
                results[f'total_samples_{period}d'] = len(valid_data)
        
        return results
    
    def batch_test_weights_v30(self, data, weight_combinations):
        """批量测试v3.0权重组合"""
        print(f"🧪 批量测试v3.0权重组合 ({len(weight_combinations)} 个组合)...")
        
        results = []
        
        for i, weights in enumerate(weight_combinations):
            print(f"  测试第 {i+1} 批 ({len([weights])} 个组合)...")
            
            result = self.test_weight_combination_v30(data, weights)
            if result and 'correlation_5d' in result:
                result['weights'] = weights
                result['weight_config'] = json.dumps(weights)
                results.append(result)
        
        print(f"✅ v3.0批量测试完成: {len(results)} 个有效结果")
        return results
    
    def analyze_and_rank_results_v30(self, results):
        """分析和排序v3.0测试结果"""
        print("📊 分析和排序v3.0测试结果...")
        
        if not results:
            print("❌ 没有有效的测试结果")
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        
        # 计算综合评分 (优先考虑胜率和相关性)
        df['composite_score'] = (
            abs(df.get('correlation_5d', 0)) * 0.3 +  # 相关性绝对值
            df.get('win_rate_5d', 0) * 0.4 +          # 胜率
            df.get('sharpe_ratio_5d', 0) * 0.2 +      # 夏普比率
            (1 - abs(df.get('max_drawdown_5d', 0))) * 0.1  # 回撤控制
        )
        
        # 按综合评分排序
        df_ranked = df.sort_values('composite_score', ascending=False)
        
        print(f"✅ v3.0结果分析完成，找到 {len(df_ranked)} 个有效权重方案")
        return df_ranked
    
    def print_optimization_results_v30(self, ranked_results):
        """打印v3.0优化结果"""
        print("\n" + "="*80)
        print("🎯 v3.0全面权重优化结果 - 基于全时段回归验证")
        print("="*80)
        
        if len(ranked_results) == 0:
            print("❌ 没有找到有效的权重方案")
            return
        
        print(f"\n📊 Top {min(5, len(ranked_results))} v3.0权重方案:")
        
        for i, (_, result) in enumerate(ranked_results.head(5).iterrows()):
            print(f"\n🏆 第 {i+1} 名 (综合评分: {result['composite_score']:.4f})")
            print("-" * 80)
            
            weights = result['weights']
            print("权重分布:")
            total_weight = 0
            for dim, weight in weights.items():
                print(f"  {dim:15s}: {weight:5.1%}")
                total_weight += weight
            print(f"  {'总计':15s}: {total_weight:5.1%}")
            
            print("\n关键性能指标:")
            metrics = [
                ('5日相关性', 'correlation_5d', '.4f'),
                ('5日胜率', 'win_rate_5d', '.1%'),
                ('5日平均收益', 'avg_return_5d', '.2f%'),
                ('5日夏普比率', 'sharpe_ratio_5d', '.3f'),
                ('5日最大回撤', 'max_drawdown_5d', '.1%'),
                ('样本数量', 'total_samples_5d', 'd')
            ]
            
            for name, key, fmt in metrics:
                value = result.get(key, 0)
                if key == 'max_drawdown_5d':
                    value = abs(value)
                print(f"  {name:15s}: {value:{fmt}}")
        
        # 保存最佳权重
        best_weights = ranked_results.iloc[0]['weights']
        self.save_best_weights_v30(best_weights, ranked_results.iloc[0])
    
    def save_best_weights_v30(self, best_weights, best_result):
        """保存v3.0最佳权重配置"""
        print("\n💾 保存v3.0最佳权重配置...")
        
        config = {
            "optimized_weights_v30": best_weights,
            "performance_metrics": {
                "correlation_5d": best_result.get('correlation_5d', 0),
                "win_rate_5d": best_result.get('win_rate_5d', 0),
                "avg_return_5d": best_result.get('avg_return_5d', 0),
                "sharpe_ratio_5d": best_result.get('sharpe_ratio_5d', 0),
                "max_drawdown_5d": best_result.get('max_drawdown_5d', 0),
                "composite_score": best_result.get('composite_score', 0)
            },
            "optimization_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "data_period": "2024-01-02 to 2025-08-22",
            "total_samples": best_result.get('total_samples_5d', 0)
        }
        
        # 保存到JSON文件
        with open('optimized_weights_v30.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print("✅ v3.0最佳权重配置已保存到: optimized_weights_v30.json")
    
    def run_optimization(self):
        """运行v3.0完整优化流程"""
        # 1. 初始化数据库
        self.create_cache_database()
        
        # 2. 加载和缓存数据
        data = self.load_and_cache_data()
        
        if len(data) < 1000:
            print("❌ v3.0数据量不足，无法进行可靠优化")
            return
        
        print(f"\n📈 开始v3.0权重优化，数据量: {len(data):,} 条记录")
        
        # 3. 生成权重组合
        weight_combinations = self.generate_weight_combinations_v30(step=0.15)
        
        # 4. 批量测试
        results = self.batch_test_weights_v30(data, weight_combinations)
        
        # 5. 分析排序
        ranked_results = self.analyze_and_rank_results_v30(results)
        
        # 6. 展示结果
        self.print_optimization_results_v30(ranked_results)

def main():
    """主函数"""
    optimizer = ComprehensiveWeightOptimizerV30()
    optimizer.run_optimization()

if __name__ == "__main__":
    main()