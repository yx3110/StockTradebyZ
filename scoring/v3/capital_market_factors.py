#!/usr/bin/env python3
"""
资金面和市场面因子计算模块
Capital Flow and Market Factors Module

实现资金流向和市场结构因子：
1. 资金面因子：成交量分析、资金流向、筹码分布
2. 市场面因子：行业比较、大盘相对强度、市场情绪
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import sqlite3
from datetime import datetime, timedelta

from .core_framework import FactorCalculator, StockData

class VolumeAnalysisFactor(FactorCalculator):
    """成交量分析因子计算器"""
    
    def __init__(self, db_path: str):
        """初始化成交量分析因子计算器"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
    
    def get_volume_data(self, stock_code: str, trade_date: str, days: int = 30) -> Dict:
        """获取成交量数据"""
        try:
            query = """
                SELECT dq.volume, dq.amount, dq.close, dq.price_change_pct
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                WHERE s.code = ? AND dq.trade_date <= ?
                ORDER BY dq.trade_date DESC
                LIMIT ?
            """
            
            df = pd.read_sql_query(query, self.conn, params=[stock_code, trade_date, days])
            
            if df.empty:
                return {}
            
            return {
                'volumes': df['volume'].values[::-1],
                'amounts': df['amount'].values[::-1] if 'amount' in df.columns else None,
                'closes': df['close'].values[::-1],
                'price_changes': df['price_change_pct'].values[::-1]
            }
            
        except Exception as e:
            print(f"获取股票 {stock_code} 成交量数据失败: {e}")
            return {}
    
    def calculate_volume_ratio(self, volumes: np.array) -> float:
        """计算量比"""
        if len(volumes) < 10:
            return 1.0
        
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-10:-1])  # 前9日平均量
        
        return current_volume / avg_volume if avg_volume > 0 else 1.0
    
    def calculate_volume_price_correlation(self, volumes: np.array, price_changes: np.array) -> float:
        """计算量价相关性"""
        if len(volumes) < 10 or len(price_changes) < 10:
            return 0.0
        
        # 取最近10天的数据
        recent_volumes = volumes[-10:]
        recent_changes = price_changes[-10:]
        
        # 计算相关系数
        correlation = np.corrcoef(recent_volumes, recent_changes)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def calculate_volume_consistency(self, volumes: np.array, price_changes: np.array) -> float:
        """计算量价一致性得分"""
        if len(volumes) < 5:
            return 50.0
        
        # 统计上涨日和下跌日的成交量对比
        up_days = price_changes[-10:] > 0
        down_days = price_changes[-10:] < 0
        
        if np.sum(up_days) == 0 or np.sum(down_days) == 0:
            return 50.0
        
        avg_up_volume = np.mean(volumes[-10:][up_days])
        avg_down_volume = np.mean(volumes[-10:][down_days])
        
        # 上涨日放量，下跌日缩量为好
        volume_ratio = avg_up_volume / avg_down_volume if avg_down_volume > 0 else 1.0
        
        if volume_ratio >= 1.5:
            return 85.0  # 上涨放量，下跌缩量
        elif volume_ratio >= 1.2:
            return 70.0
        elif volume_ratio >= 1.0:
            return 55.0
        elif volume_ratio >= 0.8:
            return 40.0
        else:
            return 25.0  # 上涨缩量，下跌放量
    
    def calculate_volume_breakthrough(self, volumes: np.array, current_price_change: float) -> float:
        """计算放量突破得分"""
        if len(volumes) < 20:
            return 50.0
        
        current_volume = volumes[-1]
        avg_volume_20 = np.mean(volumes[-20:-1])
        volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1.0
        
        # 放量上涨为正面信号
        if current_price_change > 3 and volume_ratio > 2.0:
            return 90.0  # 大幅放量上涨
        elif current_price_change > 1 and volume_ratio > 1.5:
            return 75.0  # 放量上涨
        elif current_price_change < -3 and volume_ratio > 2.0:
            return 20.0  # 大幅放量下跌
        elif current_price_change < -1 and volume_ratio > 1.5:
            return 35.0  # 放量下跌
        elif volume_ratio < 0.5:
            return 40.0  # 缩量
        else:
            return 55.0  # 正常量能
    
    def calculate(self, stock_data: StockData, market_data: Dict) -> float:
        """计算成交量分析因子综合得分"""
        try:
            # 获取成交量数据
            volume_data = self.get_volume_data(stock_data.code, stock_data.trade_date)
            
            if not volume_data or len(volume_data.get('volumes', [])) < 10:
                return 50.0
            
            volumes = volume_data['volumes']
            price_changes = volume_data['price_changes']
            current_price_change = price_changes[-1] if len(price_changes) > 0 else 0.0
            
            # 计算各项成交量指标得分
            volume_ratio = self.calculate_volume_ratio(volumes)
            consistency_score = self.calculate_volume_consistency(volumes, price_changes)
            breakthrough_score = self.calculate_volume_breakthrough(volumes, current_price_change)
            
            # 量比得分
            if volume_ratio >= 3.0:
                ratio_score = 85.0
            elif volume_ratio >= 2.0:
                ratio_score = 75.0
            elif volume_ratio >= 1.5:
                ratio_score = 65.0
            elif volume_ratio >= 0.8:
                ratio_score = 55.0
            else:
                ratio_score = 35.0
            
            # 综合得分
            volume_score = (
                ratio_score * 0.3 +
                consistency_score * 0.4 +
                breakthrough_score * 0.3
            )
            
            return min(100.0, max(0.0, volume_score))
            
        except Exception as e:
            print(f"计算成交量因子失败: {e}")
            return 50.0
    
    def get_factor_name(self) -> str:
        return "VolumeAnalysisFactor"

class LiquidityFactor(FactorCalculator):
    """流动性因子计算器"""
    
    def __init__(self, db_path: str):
        """初始化流动性因子计算器"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
    
    def get_liquidity_data(self, stock_code: str, trade_date: str, days: int = 20) -> Dict:
        """获取流动性数据"""
        try:
            query = """
                SELECT dq.volume, dq.amount, dq.close, dq.turnover_rate
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                WHERE s.code = ? AND dq.trade_date <= ?
                ORDER BY dq.trade_date DESC
                LIMIT ?
            """
            
            df = pd.read_sql_query(query, self.conn, params=[stock_code, trade_date, days])
            
            if df.empty:
                return {}
            
            # 计算换手率（如果数据库中没有）
            if 'turnover_rate' not in df.columns or df['turnover_rate'].isna().all():
                # 简化计算：假设流通市值 = 成交额 / 换手率
                df['turnover_rate'] = df['volume'] / 1000000  # 简化估算
            
            return {
                'volumes': df['volume'].values[::-1],
                'amounts': df['amount'].values[::-1] if 'amount' in df.columns else None,
                'closes': df['close'].values[::-1],
                'turnover_rates': df['turnover_rate'].values[::-1]
            }
            
        except Exception as e:
            print(f"获取股票 {stock_code} 流动性数据失败: {e}")
            return {}
    
    def calculate_turnover_score(self, turnover_rates: np.array) -> float:
        """计算换手率得分"""
        if len(turnover_rates) < 5:
            return 50.0
        
        avg_turnover = np.mean(turnover_rates[-5:])
        
        if avg_turnover >= 10:
            return 40.0  # 换手率过高，投机性强
        elif avg_turnover >= 5:
            return 75.0  # 活跃度适中
        elif avg_turnover >= 2:
            return 85.0  # 健康的活跃度
        elif avg_turnover >= 1:
            return 65.0  # 一般活跃
        else:
            return 35.0  # 流动性不足
    
    def calculate_bid_ask_impact(self, volumes: np.array, amounts: np.array) -> float:
        """计算冲击成本得分（模拟）"""
        if volumes is None or amounts is None or len(volumes) < 10:
            return 50.0
        
        # 计算平均每手成交金额作为流动性深度的代理指标
        avg_amount_per_lot = []
        for i in range(len(volumes)):
            if volumes[i] > 0:
                avg_amount_per_lot.append(amounts[i] / volumes[i])
        
        if len(avg_amount_per_lot) < 5:
            return 50.0
        
        # 成交金额越稳定，冲击成本越小
        stability = np.std(avg_amount_per_lot[-10:]) / np.mean(avg_amount_per_lot[-10:])
        
        if stability < 0.05:
            return 85.0  # 非常稳定
        elif stability < 0.1:
            return 75.0  # 稳定
        elif stability < 0.2:
            return 60.0  # 一般
        else:
            return 40.0  # 不稳定
    
    def calculate(self, stock_data: StockData, market_data: Dict) -> float:
        """计算流动性因子综合得分"""
        try:
            # 获取流动性数据
            liquidity_data = self.get_liquidity_data(stock_data.code, stock_data.trade_date)
            
            if not liquidity_data:
                return 50.0
            
            # 计算各项流动性指标得分
            turnover_score = self.calculate_turnover_score(liquidity_data['turnover_rates'])
            
            # 计算冲击成本得分
            impact_score = self.calculate_bid_ask_impact(
                liquidity_data['volumes'],
                liquidity_data['amounts']
            )
            
            # 综合得分
            liquidity_score = turnover_score * 0.6 + impact_score * 0.4
            
            return min(100.0, max(0.0, liquidity_score))
            
        except Exception as e:
            print(f"计算流动性因子失败: {e}")
            return 50.0
    
    def get_factor_name(self) -> str:
        return "LiquidityFactor"

class CompositeCapitalFactor(FactorCalculator):
    """资金面综合因子计算器"""
    
    def __init__(self, db_path: str):
        """初始化资金面综合因子计算器"""
        self.volume_factor = VolumeAnalysisFactor(db_path)
        self.liquidity_factor = LiquidityFactor(db_path)
    
    def calculate(self, stock_data: StockData, market_data: Dict) -> float:
        """计算资金面综合得分"""
        try:
            # 计算各项资金面得分
            volume_score = self.volume_factor.calculate(stock_data, market_data)
            liquidity_score = self.liquidity_factor.calculate(stock_data, market_data)
            
            # 权重配置：成交量70%，流动性30%
            capital_score = volume_score * 0.7 + liquidity_score * 0.3
            
            return min(100.0, max(0.0, capital_score))
            
        except Exception as e:
            print(f"计算资金面综合因子失败: {e}")
            return 50.0
    
    def get_factor_name(self) -> str:
        return "CompositeCapitalFactor"

class IndustryComparisonFactor(FactorCalculator):
    """行业比较因子计算器"""
    
    def __init__(self, db_path: str):
        """初始化行业比较因子计算器"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        
        # 缓存行业数据
        self.industry_cache = {}
    
    def get_industry_performance(self, industry: str, trade_date: str) -> Dict:
        """获取行业表现数据"""
        cache_key = f"{industry}_{trade_date}"
        
        if cache_key in self.industry_cache:
            return self.industry_cache[cache_key]
        
        try:
            query = """
                SELECT s.code, dq.price_change_pct, dq.volume, dq.market_cap
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                WHERE s.industry = ? AND dq.trade_date = ?
                AND dq.price_change_pct IS NOT NULL
            """
            
            df = pd.read_sql_query(query, self.conn, params=[industry, trade_date])
            
            if df.empty:
                return {}
            
            industry_data = {
                'avg_change': df['price_change_pct'].mean(),
                'median_change': df['price_change_pct'].median(),
                'stock_count': len(df),
                'up_ratio': (df['price_change_pct'] > 0).sum() / len(df),
                'avg_volume': df['volume'].mean() if 'volume' in df.columns else 0
            }
            
            self.industry_cache[cache_key] = industry_data
            return industry_data
            
        except Exception as e:
            print(f"获取行业 {industry} 表现数据失败: {e}")
            return {}
    
    def get_stock_industry(self, stock_code: str) -> str:
        """获取股票所属行业"""
        try:
            query = "SELECT industry FROM securities WHERE code = ?"
            df = pd.read_sql_query(query, self.conn, params=[stock_code])
            
            if df.empty:
                return "其他"
            
            return df.iloc[0]['industry'] or "其他"
            
        except Exception as e:
            print(f"获取股票 {stock_code} 行业信息失败: {e}")
            return "其他"
    
    def calculate_industry_relative_strength(self, stock_change: float, industry_data: Dict) -> float:
        """计算行业相对强度得分"""
        if not industry_data:
            return 50.0
        
        industry_avg = industry_data.get('avg_change', 0.0)
        
        # 计算相对强度
        if industry_avg != 0:
            relative_strength = (stock_change - industry_avg) / abs(industry_avg)
        else:
            relative_strength = 0.0
        
        # 转换为得分
        if relative_strength >= 0.5:
            return 90.0  # 显著强于行业
        elif relative_strength >= 0.2:
            return 80.0  # 强于行业
        elif relative_strength >= 0:
            return 65.0  # 略强于行业
        elif relative_strength >= -0.2:
            return 45.0  # 略弱于行业
        elif relative_strength >= -0.5:
            return 30.0  # 弱于行业
        else:
            return 15.0  # 显著弱于行业
    
    def calculate_industry_momentum(self, industry_data: Dict) -> float:
        """计算行业动量得分"""
        if not industry_data:
            return 50.0
        
        avg_change = industry_data.get('avg_change', 0.0)
        up_ratio = industry_data.get('up_ratio', 0.5)
        
        # 行业整体表现得分
        if avg_change >= 3:
            performance_score = 90.0
        elif avg_change >= 1:
            performance_score = 75.0
        elif avg_change >= 0:
            performance_score = 60.0
        elif avg_change >= -2:
            performance_score = 40.0
        else:
            performance_score = 25.0
        
        # 行业个股表现一致性得分
        if up_ratio >= 0.8:
            consistency_score = 85.0
        elif up_ratio >= 0.6:
            consistency_score = 70.0
        elif up_ratio >= 0.4:
            consistency_score = 55.0
        else:
            consistency_score = 35.0
        
        return performance_score * 0.6 + consistency_score * 0.4
    
    def calculate(self, stock_data: StockData, market_data: Dict) -> float:
        """计算行业比较因子综合得分"""
        try:
            # 获取股票所属行业
            industry = self.get_stock_industry(stock_data.code)
            
            # 获取行业表现数据
            industry_data = self.get_industry_performance(industry, stock_data.trade_date)
            
            if not industry_data:
                return 50.0
            
            # 获取股票当日涨跌幅
            stock_change = self._get_stock_price_change(stock_data.code, stock_data.trade_date)
            
            # 计算行业相对强度得分
            relative_score = self.calculate_industry_relative_strength(stock_change, industry_data)
            
            # 计算行业动量得分
            momentum_score = self.calculate_industry_momentum(industry_data)
            
            # 综合得分：相对强度70%，行业动量30%
            industry_score = relative_score * 0.7 + momentum_score * 0.3
            
            return min(100.0, max(0.0, industry_score))
            
        except Exception as e:
            print(f"计算行业比较因子失败: {e}")
            return 50.0
    
    def _get_stock_price_change(self, stock_code: str, trade_date: str) -> float:
        """获取股票当日涨跌幅"""
        try:
            query = """
                SELECT dq.price_change_pct
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                WHERE s.code = ? AND dq.trade_date = ?
            """
            
            df = pd.read_sql_query(query, self.conn, params=[stock_code, trade_date])
            
            if df.empty:
                return 0.0
            
            return df.iloc[0]['price_change_pct'] or 0.0
            
        except Exception as e:
            print(f"获取股票 {stock_code} 涨跌幅失败: {e}")
            return 0.0
    
    def get_factor_name(self) -> str:
        return "IndustryComparisonFactor"

class MarketSentimentFactor(FactorCalculator):
    """市场情绪因子计算器"""
    
    def __init__(self, db_path: str):
        """初始化市场情绪因子计算器"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
    
    def get_market_sentiment_data(self, trade_date: str) -> Dict:
        """获取市场情绪数据"""
        try:
            # 查询大盘指数表现
            index_query = """
                SELECT dq.price_change_pct
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                WHERE s.code = '000001' AND s.type = '指数' AND dq.trade_date = ?
            """
            
            index_df = pd.read_sql_query(index_query, self.conn, params=[trade_date])
            
            # 查询市场整体表现
            market_query = """
                SELECT 
                    AVG(dq.price_change_pct) as avg_change,
                    COUNT(*) as total_stocks,
                    SUM(CASE WHEN dq.price_change_pct > 0 THEN 1 ELSE 0 END) as up_stocks,
                    SUM(CASE WHEN dq.price_change_pct > 5 THEN 1 ELSE 0 END) as limit_up,
                    SUM(CASE WHEN dq.price_change_pct < -5 THEN 1 ELSE 0 END) as limit_down
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                WHERE s.type = 'A股' AND dq.trade_date = ?
            """
            
            market_df = pd.read_sql_query(market_query, self.conn, params=[trade_date])
            
            sentiment_data = {}
            
            if not index_df.empty:
                sentiment_data['index_change'] = index_df.iloc[0]['price_change_pct']
            
            if not market_df.empty:
                row = market_df.iloc[0]
                sentiment_data.update({
                    'market_avg_change': row['avg_change'],
                    'total_stocks': row['total_stocks'],
                    'up_stocks': row['up_stocks'],
                    'limit_up': row['limit_up'],
                    'limit_down': row['limit_down']
                })
                
                # 计算涨跌比例
                if row['total_stocks'] > 0:
                    sentiment_data['up_ratio'] = row['up_stocks'] / row['total_stocks']
                else:
                    sentiment_data['up_ratio'] = 0.5
            
            return sentiment_data
            
        except Exception as e:
            print(f"获取市场情绪数据失败: {e}")
            return {}
    
    def calculate_market_breadth_score(self, sentiment_data: Dict) -> float:
        """计算市场宽度得分"""
        if not sentiment_data:
            return 50.0
        
        up_ratio = sentiment_data.get('up_ratio', 0.5)
        
        if up_ratio >= 0.8:
            return 90.0  # 极度乐观
        elif up_ratio >= 0.65:
            return 75.0  # 乐观
        elif up_ratio >= 0.55:
            return 60.0  # 略乐观
        elif up_ratio >= 0.45:
            return 50.0  # 中性
        elif up_ratio >= 0.35:
            return 40.0  # 略悲观
        elif up_ratio >= 0.2:
            return 25.0  # 悲观
        else:
            return 15.0  # 极度悲观
    
    def calculate_market_strength_score(self, sentiment_data: Dict) -> float:
        """计算市场强度得分"""
        if not sentiment_data:
            return 50.0
        
        market_change = sentiment_data.get('market_avg_change', 0.0)
        index_change = sentiment_data.get('index_change', 0.0)
        
        # 大盘指数得分
        if index_change >= 2:
            index_score = 85.0
        elif index_change >= 1:
            index_score = 70.0
        elif index_change >= 0:
            index_score = 60.0
        elif index_change >= -1:
            index_score = 45.0
        elif index_change >= -2:
            index_score = 30.0
        else:
            index_score = 20.0
        
        # 市场平均得分
        if market_change >= 1.5:
            market_score = 85.0
        elif market_change >= 0.5:
            market_score = 70.0
        elif market_change >= 0:
            market_score = 60.0
        elif market_change >= -0.5:
            market_score = 45.0
        elif market_change >= -1.5:
            market_score = 30.0
        else:
            market_score = 20.0
        
        return index_score * 0.4 + market_score * 0.6
    
    def calculate_extreme_sentiment_score(self, sentiment_data: Dict) -> float:
        """计算极端情绪得分"""
        if not sentiment_data:
            return 50.0
        
        total_stocks = sentiment_data.get('total_stocks', 1)
        limit_up = sentiment_data.get('limit_up', 0)
        limit_down = sentiment_data.get('limit_down', 0)
        
        if total_stocks == 0:
            return 50.0
        
        limit_up_ratio = limit_up / total_stocks
        limit_down_ratio = limit_down / total_stocks
        
        # 涨停股票比例得分
        if limit_up_ratio >= 0.05:
            up_score = 90.0  # 市场极度乐观
        elif limit_up_ratio >= 0.02:
            up_score = 75.0
        elif limit_up_ratio >= 0.01:
            up_score = 65.0
        else:
            up_score = 50.0
        
        # 跌停股票惩罚
        if limit_down_ratio >= 0.05:
            down_penalty = 40.0  # 市场恐慌
        elif limit_down_ratio >= 0.02:
            down_penalty = 25.0
        elif limit_down_ratio >= 0.01:
            down_penalty = 15.0
        else:
            down_penalty = 0.0
        
        return max(10.0, up_score - down_penalty)
    
    def calculate(self, stock_data: StockData, market_data: Dict) -> float:
        """计算市场情绪因子综合得分"""
        try:
            # 获取市场情绪数据
            sentiment_data = self.get_market_sentiment_data(stock_data.trade_date)
            
            if not sentiment_data:
                return 50.0
            
            # 计算各项市场情绪得分
            breadth_score = self.calculate_market_breadth_score(sentiment_data)
            strength_score = self.calculate_market_strength_score(sentiment_data)
            extreme_score = self.calculate_extreme_sentiment_score(sentiment_data)
            
            # 综合得分：宽度40%，强度40%，极端情绪20%
            sentiment_score = (
                breadth_score * 0.4 +
                strength_score * 0.4 +
                extreme_score * 0.2
            )
            
            return min(100.0, max(0.0, sentiment_score))
            
        except Exception as e:
            print(f"计算市场情绪因子失败: {e}")
            return 50.0
    
    def get_factor_name(self) -> str:
        return "MarketSentimentFactor"

class CompositeMarketFactor(FactorCalculator):
    """市场面综合因子计算器"""
    
    def __init__(self, db_path: str):
        """初始化市场面综合因子计算器"""
        self.industry_factor = IndustryComparisonFactor(db_path)
        self.sentiment_factor = MarketSentimentFactor(db_path)
    
    def calculate(self, stock_data: StockData, market_data: Dict) -> float:
        """计算市场面综合得分"""
        try:
            # 计算各项市场面得分
            industry_score = self.industry_factor.calculate(stock_data, market_data)
            sentiment_score = self.sentiment_factor.calculate(stock_data, market_data)
            
            # 根据市场状态调整权重
            market_state = market_data.get('market_state', 'sideways')
            
            if market_state == "bull":
                # 牛市：重市场情绪
                market_score = industry_score * 0.4 + sentiment_score * 0.6
            elif market_state == "bear":
                # 熊市：重行业比较
                market_score = industry_score * 0.6 + sentiment_score * 0.4
            else:
                # 震荡市：平衡权重
                market_score = industry_score * 0.5 + sentiment_score * 0.5
            
            return min(100.0, max(0.0, market_score))
            
        except Exception as e:
            print(f"计算市场面综合因子失败: {e}")
            return 50.0
    
    def get_factor_name(self) -> str:
        return "CompositeMarketFactor"

if __name__ == "__main__":
    # 测试代码
    test_stock = StockData(
        code="000001",
        name="平安银行",
        trade_date="2025-08-01",
        open=10.0,
        high=10.5,
        low=9.8,
        close=10.2,
        volume=100000
    )
    
    # 测试资金面和市场面因子
    capital_factor = CompositeCapitalFactor("data_adapter/stock_data.db")
    market_factor = CompositeMarketFactor("data_adapter/stock_data.db")
    
    capital_score = capital_factor.calculate(test_stock, {"trade_date": "2025-08-01"})
    market_score = market_factor.calculate(test_stock, {"market_state": "bull", "trade_date": "2025-08-01"})
    
    print(f"✅ 资金面和市场面因子计算完成")
    print(f"📊 测试股票 {test_stock.code} 资金面得分: {capital_score:.2f}")
    print(f"📊 测试股票 {test_stock.code} 市场面得分: {market_score:.2f}")