#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统性权重优化器 - 基于用户科学方法论的完整实现
方案：
1. 计算全市场所有股票每日6个v3.1指标
2. 用feature_vector和weight_vector计算评分
3. 迭代优化权重直到收敛，实现高分涨低分跌
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
from scipy.optimize import minimize
import itertools
from concurrent.futures import ThreadPoolExecutor
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入v3.1评分器
from scoring.v3.quantitative_scorer_v3_1 import QuantitativeScorerV31

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystematicWeightOptimizer:
    """系统性权重优化器"""
    
    def __init__(self):
        self.db_path = "systematic_weight_optimization.db"
        self.scorer = QuantitativeScorerV31()
        self.feature_names = ['technical', 'fundamental', 'performance', 'sentiment', 'risk_control', 'market_regime']
        
    def create_database(self):
        """创建系统性权重优化数据库"""
        print("🗄️ 创建系统性权重优化数据库...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建全市场特征指标表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT NOT NULL,
            date TEXT NOT NULL,
            technical REAL,
            fundamental REAL,
            performance REAL,
            sentiment REAL,
            risk_control REAL,
            market_regime REAL,
            return_1d REAL,
            return_3d REAL,
            return_5d REAL,
            return_10d REAL,
            return_20d REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(code, date)
        )
        ''')
        
        # 创建权重测试记录表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS weight_tests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            iteration INTEGER,
            weight_technical REAL,
            weight_fundamental REAL,
            weight_performance REAL,
            weight_sentiment REAL,
            weight_risk_control REAL,
            weight_market_regime REAL,
            correlation_1d REAL,
            correlation_3d REAL,
            correlation_5d REAL,
            win_rate_1d REAL,
            win_rate_3d REAL,
            win_rate_5d REAL,
            score_range_min REAL,
            score_range_max REAL,
            sample_size INTEGER,
            objective_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 创建优化历史表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS optimization_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phase TEXT,
            description TEXT,
            best_weights TEXT,
            best_score REAL,
            convergence_status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ 数据库创建完成")
    
    def get_all_stock_codes(self):
        """获取全市场股票代码"""
        print("📊 获取全市场股票代码...")
        
        conn = sqlite3.connect("data_adapter/stock_data.db")
        query = "SELECT DISTINCT code FROM securities WHERE type = 'A股' ORDER BY code"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        codes = df['code'].tolist()
        print(f"✅ 获取到 {len(codes)} 只A股代码")
        return codes
    
    def calculate_v31_features_for_date(self, date_str, stock_codes):
        """计算指定日期所有股票的v3.1特征"""
        print(f"📈 计算 {date_str} 的v3.1特征 ({len(stock_codes)} 只股票)...")
        
        features = []
        
        for i, code in enumerate(stock_codes):
            if i % 500 == 0:
                print(f"  处理进度: {i+1}/{len(stock_codes)}")
            
            try:
                # 使用v3.1评分器计算特征
                scoring_result = self.scorer.calculate_stock_score(code, date_str)
                if scoring_result is None or not scoring_result.get('scores'):
                    continue
                
                # 从结果中提取各维度评分
                scores = scoring_result.get('scores', {})
                technical_score = scores.get('technical', 0)
                fundamental_score = scores.get('fundamental', 0)
                performance_score = scores.get('performance', 0)
                sentiment_score = scores.get('sentiment', 0)
                risk_control_score = scores.get('risk_control', 0)
                market_regime_score = scores.get('market_regime', 0)
                
                features.append({
                    'code': code,
                    'date': date_str,
                    'technical': technical_score,
                    'fundamental': fundamental_score,
                    'performance': performance_score,
                    'sentiment': sentiment_score,
                    'risk_control': risk_control_score,
                    'market_regime': market_regime_score
                })
                
            except Exception as e:
                logger.warning(f"计算{code}特征失败: {e}")
                continue
        
        print(f"✅ {date_str} 完成特征计算: {len(features)} 只股票")
        return features
    
    def calculate_stock_returns(self, stock_codes, start_date, end_date):
        """计算股票收益率"""
        print(f"💰 计算收益率数据 {start_date} 到 {end_date}...")
        
        conn = sqlite3.connect("data_adapter/stock_data.db")
        
        # 分批处理
        batch_size = 100
        all_returns = []
        
        for i in range(0, len(stock_codes), batch_size):
            batch_codes = stock_codes[i:i+batch_size]
            code_list = "','".join(batch_codes)
            
            query = f"""
            WITH future_returns AS (
                SELECT 
                    s.code,
                    dq1.trade_date as date,
                    dq1.close as current_price,
                    dq2.close as price_1d,
                    dq3.close as price_3d,
                    dq4.close as price_5d,
                    dq5.close as price_10d,
                    dq6.close as price_20d,
                    CASE WHEN dq2.close IS NOT NULL THEN (dq2.close - dq1.close) / dq1.close * 100 ELSE NULL END as return_1d,
                    CASE WHEN dq3.close IS NOT NULL THEN (dq3.close - dq1.close) / dq1.close * 100 ELSE NULL END as return_3d,
                    CASE WHEN dq4.close IS NOT NULL THEN (dq4.close - dq1.close) / dq1.close * 100 ELSE NULL END as return_5d,
                    CASE WHEN dq5.close IS NOT NULL THEN (dq5.close - dq1.close) / dq1.close * 100 ELSE NULL END as return_10d,
                    CASE WHEN dq6.close IS NOT NULL THEN (dq6.close - dq1.close) / dq1.close * 100 ELSE NULL END as return_20d
                FROM securities s
                JOIN daily_quotes dq1 ON s.id = dq1.security_id
                LEFT JOIN daily_quotes dq2 ON s.id = dq2.security_id AND dq2.trade_date = date(dq1.trade_date, '+1 day')
                LEFT JOIN daily_quotes dq3 ON s.id = dq3.security_id AND dq3.trade_date = date(dq1.trade_date, '+3 days')
                LEFT JOIN daily_quotes dq4 ON s.id = dq4.security_id AND dq4.trade_date = date(dq1.trade_date, '+5 days')
                LEFT JOIN daily_quotes dq5 ON s.id = dq5.security_id AND dq5.trade_date = date(dq1.trade_date, '+10 days')
                LEFT JOIN daily_quotes dq6 ON s.id = dq6.security_id AND dq6.trade_date = date(dq1.trade_date, '+20 days')
                WHERE s.code IN ('{code_list}')
                AND dq1.trade_date >= '{start_date}'
                AND dq1.trade_date <= '{end_date}'
            )
            SELECT * FROM future_returns
            """
            
            batch_returns = pd.read_sql_query(query, conn)
            all_returns.append(batch_returns)
        
        conn.close()
        
        returns_df = pd.concat(all_returns, ignore_index=True) if all_returns else pd.DataFrame()
        print(f"✅ 收益率计算完成: {len(returns_df)} 条记录")
        return returns_df
    
    def build_comprehensive_dataset(self, start_date="2024-01-01", end_date="2025-08-22"):
        """构建全面数据集 - 第一步实现"""
        print("🚀 开始构建全面v3.1特征数据集...")
        print(f"📅 时间范围: {start_date} 到 {end_date}")
        
        # 获取所有股票代码
        stock_codes = self.get_all_stock_codes()
        
        # 获取交易日列表
        conn = sqlite3.connect("data_adapter/stock_data.db")
        trade_dates_query = f"""
        SELECT DISTINCT trade_date 
        FROM daily_quotes 
        WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
        ORDER BY trade_date
        """
        trade_dates_df = pd.read_sql_query(trade_dates_query, conn)
        conn.close()
        
        trade_dates = trade_dates_df['trade_date'].tolist()
        print(f"📊 需要处理 {len(trade_dates)} 个交易日")
        
        # 分批计算特征（避免内存溢出）
        batch_size = 10  # 每批处理10个交易日
        all_features = []
        
        for i in range(0, len(trade_dates), batch_size):
            batch_dates = trade_dates[i:i+batch_size]
            print(f"\n📈 处理交易日批次 {i//batch_size + 1}/{(len(trade_dates)-1)//batch_size + 1}")
            
            batch_features = []
            for date_str in batch_dates:
                date_features = self.calculate_v31_features_for_date(date_str, stock_codes)
                batch_features.extend(date_features)
            
            # 计算这批数据的收益率
            if batch_features:
                batch_codes = list(set([f['code'] for f in batch_features]))
                batch_start = batch_dates[0]
                batch_end = min(batch_dates[-1], end_date)
                
                returns_data = self.calculate_stock_returns(batch_codes, batch_start, batch_end)
                
                # 合并特征和收益率
                features_df = pd.DataFrame(batch_features)
                merged_data = pd.merge(features_df, returns_data, on=['code', 'date'], how='left')
                
                # 保存到数据库
                conn = sqlite3.connect(self.db_path)
                merged_data.to_sql('market_features', conn, if_exists='append', index=False)
                conn.close()
                
                print(f"✅ 批次完成，保存 {len(merged_data)} 条记录")
        
        # 验证数据完整性
        conn = sqlite3.connect(self.db_path)
        summary = pd.read_sql_query("""
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT code) as unique_stocks,
            COUNT(DISTINCT date) as trading_days,
            MIN(date) as start_date,
            MAX(date) as end_date,
            COUNT(CASE WHEN return_5d IS NOT NULL THEN 1 END) as records_with_returns
        FROM market_features
        """, conn)
        conn.close()
        
        print("\n" + "="*80)
        print("🎯 全面数据集构建完成")
        print("="*80)
        print(f"📊 总记录数: {summary.iloc[0]['total_records']:,}")
        print(f"🏢 股票数量: {summary.iloc[0]['unique_stocks']:,}")
        print(f"📅 交易日数: {summary.iloc[0]['trading_days']:,}")
        print(f"📈 有收益率记录: {summary.iloc[0]['records_with_returns']:,}")
        print(f"📅 时间跨度: {summary.iloc[0]['start_date']} 到 {summary.iloc[0]['end_date']}")
        
        return True
    
    def calculate_weighted_score(self, feature_vector, weight_vector):
        """计算加权评分 - 第二步核心函数"""
        return np.dot(feature_vector, weight_vector) * 100
    
    def evaluate_weight_vector(self, weights, data_sample=None):
        """评估权重向量的预测能力 - 第二步实现"""
        
        if data_sample is None:
            # 加载数据
            conn = sqlite3.connect(self.db_path)
            data_sample = pd.read_sql_query("""
            SELECT code, date, technical, fundamental, performance, sentiment, risk_control, market_regime,
                   return_1d, return_3d, return_5d, return_10d, return_20d
            FROM market_features 
            WHERE return_5d IS NOT NULL
            LIMIT 10000
            """, conn)
            conn.close()
        
        if len(data_sample) < 100:
            return -999  # 数据不足
        
        # 构建特征矩阵
        feature_matrix = data_sample[self.feature_names].values
        
        # 计算每个样本的评分
        scores = np.array([self.calculate_weighted_score(features, weights) for features in feature_matrix])
        
        # 评估预测能力
        returns_5d = data_sample['return_5d'].values
        
        # 计算相关性
        try:
            correlation, p_value = stats.pearsonr(scores, returns_5d)
        except:
            correlation = 0
            
        # 计算胜率（高分股票上涨的概率）
        high_score_threshold = np.percentile(scores, 75)  # 前25%高分
        high_score_mask = scores >= high_score_threshold
        high_score_returns = returns_5d[high_score_mask]
        
        if len(high_score_returns) > 0:
            win_rate = (high_score_returns > 0).mean()
        else:
            win_rate = 0.5
        
        # 计算目标函数值（我们要最大化这个值）
        # 目标：高相关性 + 高胜率 + 评分区分度
        score_std = np.std(scores)  # 评分的区分度
        objective = abs(correlation) * 0.4 + win_rate * 0.4 + (score_std / 100) * 0.2
        
        return -objective  # 因为minimize函数是最小化，所以返回负值
    
    def optimize_weights_iteratively(self, max_iterations=50, convergence_threshold=1e-6):
        """迭代优化权重向量 - 第三步实现"""
        print("🔧 开始迭代权重优化...")
        
        # 加载全部数据用于优化
        conn = sqlite3.connect(self.db_path)
        full_data = pd.read_sql_query("""
        SELECT code, date, technical, fundamental, performance, sentiment, risk_control, market_regime,
               return_1d, return_3d, return_5d, return_10d, return_20d
        FROM market_features 
        WHERE return_5d IS NOT NULL
        AND technical IS NOT NULL AND fundamental IS NOT NULL AND performance IS NOT NULL
        AND sentiment IS NOT NULL AND risk_control IS NOT NULL AND market_regime IS NOT NULL
        """, conn)
        conn.close()
        
        print(f"📊 优化数据集大小: {len(full_data):,} 条记录")
        
        if len(full_data) < 1000:
            print("❌ 数据量不足，无法进行可靠优化")
            return None
        
        # 初始权重（基于v3.1原始权重）
        initial_weights = np.array([0.60, 0.18, 0.15, 0.05, 0.05, 0.05])  # 总和1.08
        
        # 权重约束：每个权重在[0.01, 0.80]之间，总权重在[0.9, 1.2]之间
        bounds = [(0.01, 0.80) for _ in range(6)]
        
        def constraint_sum(weights):
            return 1.2 - np.sum(weights)  # 总权重不超过1.2
        
        def constraint_min_sum(weights):
            return np.sum(weights) - 0.9  # 总权重不少于0.9
        
        constraints = [
            {'type': 'ineq', 'fun': constraint_sum},
            {'type': 'ineq', 'fun': constraint_min_sum}
        ]
        
        # 优化过程记录
        iteration_results = []
        
        def callback_function(weights):
            iteration = len(iteration_results)
            objective_score = -self.evaluate_weight_vector(weights, full_data)
            
            print(f"  迭代 {iteration+1}: 目标函数值 = {objective_score:.6f}")
            print(f"    权重: {weights}")
            
            iteration_results.append({
                'iteration': iteration,
                'weights': weights.copy(),
                'objective_score': objective_score
            })
            
            # 保存到数据库
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO weight_tests (iteration, weight_technical, weight_fundamental, weight_performance,
                                    weight_sentiment, weight_risk_control, weight_market_regime, objective_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (iteration, weights[0], weights[1], weights[2], weights[3], weights[4], weights[5], objective_score))
            conn.commit()
            conn.close()
        
        # 执行优化
        print("🎯 开始权重优化...")
        result = minimize(
            fun=lambda w: self.evaluate_weight_vector(w, full_data),
            x0=initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            callback=callback_function,
            options={'maxiter': max_iterations, 'ftol': convergence_threshold}
        )
        
        print("\n" + "="*80)
        print("🎯 权重优化完成")
        print("="*80)
        print(f"✅ 优化成功: {result.success}")
        print(f"🔄 迭代次数: {result.nit}")
        print(f"📊 最终目标函数值: {-result.fun:.6f}")
        
        optimal_weights = result.x
        print(f"\n🏆 最优权重向量:")
        for i, (name, weight) in enumerate(zip(self.feature_names, optimal_weights)):
            print(f"  {name:15s}: {weight:.4f} ({weight*100:.1f}%)")
        print(f"  {'总权重':15s}: {np.sum(optimal_weights):.4f}")
        
        # 详细验证最优权重
        self.validate_optimal_weights(optimal_weights, full_data)
        
        return optimal_weights, result
    
    def validate_optimal_weights(self, optimal_weights, data):
        """验证最优权重的效果"""
        print("\n🔍 验证最优权重效果...")
        
        # 计算评分
        feature_matrix = data[self.feature_names].values
        scores = np.array([self.calculate_weighted_score(features, optimal_weights) for features in feature_matrix])
        
        # 多时间窗口验证
        for period in ['1d', '3d', '5d', '10d', '20d']:
            return_col = f'return_{period}'
            if return_col in data.columns:
                returns = data[return_col].dropna()
                valid_scores = scores[:len(returns)]
                
                if len(returns) > 100:
                    # 相关性
                    corr, p_val = stats.pearsonr(valid_scores, returns)
                    
                    # 胜率分析
                    top_25_threshold = np.percentile(valid_scores, 75)
                    top_stocks_mask = valid_scores >= top_25_threshold
                    top_returns = returns[top_stocks_mask]
                    win_rate = (top_returns > 0).mean() if len(top_returns) > 0 else 0
                    
                    # 平均收益
                    avg_return_top = top_returns.mean() if len(top_returns) > 0 else 0
                    avg_return_all = returns.mean()
                    
                    print(f"  {period:3s}: 相关性={corr:+.4f} (p={p_val:.4f}), 胜率={win_rate:.1%}, 高分收益={avg_return_top:+.2f}%, 全市场={avg_return_all:+.2f}%")
        
        # 保存最优权重
        self.save_optimal_weights(optimal_weights)
    
    def save_optimal_weights(self, optimal_weights):
        """保存最优权重配置"""
        config = {
            "optimal_weights_v31_systematic": {
                name: float(weight) for name, weight in zip(self.feature_names, optimal_weights)
            },
            "total_weight": float(np.sum(optimal_weights)),
            "optimization_method": "systematic_iterative_optimization",
            "optimization_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "data_period": "2024-01-01 to 2025-08-22",
            "convergence_status": "completed"
        }
        
        with open('optimal_weights_v31_systematic.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print("✅ 最优权重已保存到: optimal_weights_v31_systematic.json")
    
    def run_complete_optimization(self):
        """运行完整的系统性权重优化流程"""
        print("🚀 启动系统性权重优化器")
        print("="*80)
        print("基于用户科学方法论:")
        print("1. 计算全市场所有股票每日6个v3.1指标")
        print("2. 用feature_vector和weight_vector计算评分") 
        print("3. 迭代优化权重直到收敛，实现高分涨低分跌")
        print("="*80)
        
        # 第一步：构建数据集
        self.create_database()
        
        # 检查是否已有数据
        conn = sqlite3.connect(self.db_path)
        existing_data = pd.read_sql_query("SELECT COUNT(*) as count FROM market_features", conn)
        conn.close()
        
        if existing_data.iloc[0]['count'] < 10000:
            print("📊 数据不足，开始构建全面数据集...")
            success = self.build_comprehensive_dataset()
            if not success:
                print("❌ 数据集构建失败")
                return
        else:
            print(f"📊 发现已有数据 {existing_data.iloc[0]['count']} 条，跳过数据构建")
        
        # 第二步和第三步：权重优化
        optimal_weights, optimization_result = self.optimize_weights_iteratively()
        
        if optimal_weights is not None:
            print("\n🎉 系统性权重优化成功完成！")
            return optimal_weights
        else:
            print("\n❌ 权重优化失败")
            return None

def main():
    """主函数"""
    optimizer = SystematicWeightOptimizer()
    result = optimizer.run_complete_optimization()
    return result

if __name__ == "__main__":
    main()