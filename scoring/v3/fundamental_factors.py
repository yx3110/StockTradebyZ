#!/usr/bin/env python3
"""
基本面因子计算模块
Fundamental Factors Calculation Module

实现基于财务指标的基本面评分因子：
1. 估值因子：PE、PB、PS相对位置
2. 盈利因子：ROE、ROA、净利润增长率
3. 成长因子：营收增长、盈利增长稳定性
4. 财务健康度：资产负债率、现金流
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import sqlite3
from datetime import datetime, timedelta

from .core_framework import FactorCalculator, StockData

class ValuationFactor(FactorCalculator):
    """估值因子计算器"""
    
    def __init__(self, db_path: str):
        """初始化估值因子计算器"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        
        # 缓存行业估值分位数
        self.industry_percentiles = {}
        
    def get_industry_percentiles(self, industry: str, trade_date: str) -> Dict[str, float]:
        """获取行业估值分位数"""
        cache_key = f"{industry}_{trade_date}"
        
        if cache_key in self.industry_percentiles:
            return self.industry_percentiles[cache_key]
        
        try:
            # 查询同行业股票的估值数据
            query = """
                SELECT db.pe_ttm, db.pb, db.ps_ttm
                FROM daily_basic db
                JOIN securities s ON db.security_id = s.id
                WHERE s.industry = ? AND db.trade_date = ?
                AND db.pe_ttm > 0 AND db.pe_ttm < 100  -- 过滤异常值
                AND db.pb > 0 AND db.pb < 10
                AND db.ps_ttm > 0 AND db.ps_ttm < 20
            """
            
            df = pd.read_sql_query(query, self.conn, params=[industry, trade_date])
            
            if df.empty:
                return {'pe_25': 15, 'pe_75': 30, 'pb_25': 1.5, 'pb_75': 3.0, 'ps_25': 2, 'ps_75': 6}
            
            percentiles = {
                'pe_25': df['pe_ttm'].quantile(0.25),
                'pe_75': df['pe_ttm'].quantile(0.75),
                'pb_25': df['pb'].quantile(0.25),
                'pb_75': df['pb'].quantile(0.75),
                'ps_25': df['ps_ttm'].quantile(0.25),
                'ps_75': df['ps_ttm'].quantile(0.75)
            }
            
            self.industry_percentiles[cache_key] = percentiles
            return percentiles
            
        except Exception as e:
            print(f"获取行业 {industry} 估值分位数失败: {e}")
            return {'pe_25': 15, 'pe_75': 30, 'pb_25': 1.5, 'pb_75': 3.0, 'ps_25': 2, 'ps_75': 6}
    
    def calculate_pe_score(self, pe_ttm: float, industry_percentiles: Dict[str, float]) -> float:
        """计算PE评分（越低越好）"""
        if pe_ttm <= 0 or pe_ttm > 100:
            return 30.0  # 异常值给予低分
        
        pe_25 = industry_percentiles['pe_25']
        pe_75 = industry_percentiles['pe_75']
        
        if pe_ttm <= pe_25:
            return 80.0  # 低估值高分
        elif pe_ttm <= pe_75:
            # 线性插值
            score = 80.0 - (pe_ttm - pe_25) / (pe_75 - pe_25) * 30.0
            return max(50.0, score)
        else:
            # 高估值低分
            penalty = min((pe_ttm - pe_75) / pe_75 * 20.0, 20.0)
            return max(30.0, 50.0 - penalty)
    
    def calculate_pb_score(self, pb: float, industry_percentiles: Dict[str, float]) -> float:
        """计算PB评分（越低越好）"""
        if pb <= 0 or pb > 10:
            return 30.0
        
        pb_25 = industry_percentiles['pb_25']
        pb_75 = industry_percentiles['pb_75']
        
        if pb <= pb_25:
            return 80.0
        elif pb <= pb_75:
            score = 80.0 - (pb - pb_25) / (pb_75 - pb_25) * 30.0
            return max(50.0, score)
        else:
            penalty = min((pb - pb_75) / pb_75 * 20.0, 20.0)
            return max(30.0, 50.0 - penalty)
    
    def calculate_ps_score(self, ps_ttm: float, industry_percentiles: Dict[str, float]) -> float:
        """计算PS评分（越低越好）"""
        if ps_ttm <= 0 or ps_ttm > 20:
            return 30.0
        
        ps_25 = industry_percentiles['ps_25']
        ps_75 = industry_percentiles['ps_75']
        
        if ps_ttm <= ps_25:
            return 80.0
        elif ps_ttm <= ps_75:
            score = 80.0 - (ps_ttm - ps_25) / (ps_75 - ps_25) * 30.0
            return max(50.0, score)
        else:
            penalty = min((ps_ttm - ps_75) / ps_75 * 20.0, 20.0)
            return max(30.0, 50.0 - penalty)
    
    def calculate(self, stock_data: StockData, market_data: Dict) -> float:
        """计算估值因子综合得分"""
        try:
            if not all([stock_data.pe_ttm, stock_data.pb, stock_data.ps_ttm]):
                return 50.0  # 数据缺失给予中性分
            
            # 获取股票行业信息
            industry_query = """
                SELECT industry FROM securities WHERE code = ?
            """
            industry_df = pd.read_sql_query(industry_query, self.conn, params=[stock_data.code])
            
            if industry_df.empty:
                industry = "其他"
            else:
                industry = industry_df.iloc[0]['industry'] or "其他"
            
            # 获取行业分位数
            percentiles = self.get_industry_percentiles(industry, stock_data.trade_date)
            
            # 计算各指标得分
            pe_score = self.calculate_pe_score(stock_data.pe_ttm, percentiles)
            pb_score = self.calculate_pb_score(stock_data.pb, percentiles)
            ps_score = self.calculate_ps_score(stock_data.ps_ttm, percentiles)
            
            # 加权平均（PE权重最高）
            valuation_score = pe_score * 0.5 + pb_score * 0.3 + ps_score * 0.2
            
            return min(100.0, max(0.0, valuation_score))
            
        except Exception as e:
            print(f"计算估值因子失败: {e}")
            return 50.0
    
    def get_factor_name(self) -> str:
        return "ValuationFactor"

class ProfitabilityFactor(FactorCalculator):
    """盈利能力因子计算器"""
    
    def __init__(self, db_path: str):
        """初始化盈利能力因子计算器"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
    
    def get_financial_indicators(self, stock_code: str, trade_date: str) -> Dict[str, float]:
        """获取财务指标"""
        try:
            query = """
                SELECT roe, roa, net_profit_yoy, revenue_yoy, gross_margin
                FROM financial_indicator fi
                JOIN securities s ON fi.security_id = s.id
                WHERE s.code = ? AND fi.trade_date <= ?
                ORDER BY fi.trade_date DESC
                LIMIT 1
            """
            
            df = pd.read_sql_query(query, self.conn, params=[stock_code, trade_date])
            
            if df.empty:
                return {}
            
            row = df.iloc[0]
            return {
                'roe': row.get('roe'),
                'roa': row.get('roa'),
                'net_profit_yoy': row.get('net_profit_yoy'),
                'revenue_yoy': row.get('revenue_yoy'),
                'gross_margin': row.get('gross_margin')
            }
            
        except Exception as e:
            print(f"获取股票 {stock_code} 财务指标失败: {e}")
            return {}
    
    def calculate_roe_score(self, roe: Optional[float]) -> float:
        """计算ROE得分"""
        if roe is None or roe < 0:
            return 30.0
        
        if roe >= 20:
            return 90.0
        elif roe >= 15:
            return 80.0
        elif roe >= 10:
            return 70.0
        elif roe >= 5:
            return 60.0
        else:
            return 40.0
    
    def calculate_roa_score(self, roa: Optional[float]) -> float:
        """计算ROA得分"""
        if roa is None or roa < 0:
            return 30.0
        
        if roa >= 10:
            return 90.0
        elif roa >= 7:
            return 80.0
        elif roa >= 5:
            return 70.0
        elif roa >= 3:
            return 60.0
        else:
            return 40.0
    
    def calculate_growth_score(self, net_profit_yoy: Optional[float], revenue_yoy: Optional[float]) -> float:
        """计算成长性得分"""
        scores = []
        
        if net_profit_yoy is not None:
            if net_profit_yoy >= 30:
                scores.append(90.0)
            elif net_profit_yoy >= 20:
                scores.append(80.0)
            elif net_profit_yoy >= 10:
                scores.append(70.0)
            elif net_profit_yoy >= 0:
                scores.append(60.0)
            else:
                scores.append(30.0)
        
        if revenue_yoy is not None:
            if revenue_yoy >= 20:
                scores.append(85.0)
            elif revenue_yoy >= 15:
                scores.append(75.0)
            elif revenue_yoy >= 10:
                scores.append(65.0)
            elif revenue_yoy >= 0:
                scores.append(55.0)
            else:
                scores.append(35.0)
        
        return np.mean(scores) if scores else 50.0
    
    def calculate(self, stock_data: StockData, market_data: Dict) -> float:
        """计算盈利能力因子综合得分"""
        try:
            # 获取财务指标
            financial_data = self.get_financial_indicators(stock_data.code, stock_data.trade_date)
            
            if not financial_data:
                return 50.0
            
            # 计算各项得分
            roe_score = self.calculate_roe_score(financial_data.get('roe'))
            roa_score = self.calculate_roa_score(financial_data.get('roa'))
            growth_score = self.calculate_growth_score(
                financial_data.get('net_profit_yoy'),
                financial_data.get('revenue_yoy')
            )
            
            # 加权平均
            profitability_score = roe_score * 0.4 + roa_score * 0.3 + growth_score * 0.3
            
            return min(100.0, max(0.0, profitability_score))
            
        except Exception as e:
            print(f"计算盈利能力因子失败: {e}")
            return 50.0
    
    def get_factor_name(self) -> str:
        return "ProfitabilityFactor"

class FinancialHealthFactor(FactorCalculator):
    """财务健康度因子计算器"""
    
    def __init__(self, db_path: str):
        """初始化财务健康度因子计算器"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
    
    def get_financial_health_data(self, stock_code: str, trade_date: str) -> Dict[str, float]:
        """获取财务健康度数据"""
        try:
            query = """
                SELECT debt_to_assets, current_ratio, quick_ratio, 
                       cash_ratio, operating_cash_flow_yoy
                FROM financial_indicator fi
                JOIN securities s ON fi.security_id = s.id
                WHERE s.code = ? AND fi.trade_date <= ?
                ORDER BY fi.trade_date DESC
                LIMIT 1
            """
            
            df = pd.read_sql_query(query, self.conn, params=[stock_code, trade_date])
            
            if df.empty:
                return {}
            
            row = df.iloc[0]
            return {
                'debt_to_assets': row.get('debt_to_assets'),
                'current_ratio': row.get('current_ratio'),
                'quick_ratio': row.get('quick_ratio'),
                'cash_ratio': row.get('cash_ratio'),
                'operating_cash_flow_yoy': row.get('operating_cash_flow_yoy')
            }
            
        except Exception as e:
            print(f"获取股票 {stock_code} 财务健康度数据失败: {e}")
            return {}
    
    def calculate_debt_score(self, debt_to_assets: Optional[float]) -> float:
        """计算负债率得分（越低越好）"""
        if debt_to_assets is None:
            return 50.0
        
        if debt_to_assets < 0.3:
            return 90.0
        elif debt_to_assets < 0.5:
            return 75.0
        elif debt_to_assets < 0.7:
            return 60.0
        else:
            return 30.0
    
    def calculate_liquidity_score(self, current_ratio: Optional[float], 
                                  quick_ratio: Optional[float]) -> float:
        """计算流动性得分"""
        scores = []
        
        if current_ratio is not None:
            if current_ratio >= 2.0:
                scores.append(85.0)
            elif current_ratio >= 1.5:
                scores.append(75.0)
            elif current_ratio >= 1.0:
                scores.append(60.0)
            else:
                scores.append(30.0)
        
        if quick_ratio is not None:
            if quick_ratio >= 1.5:
                scores.append(85.0)
            elif quick_ratio >= 1.0:
                scores.append(75.0)
            elif quick_ratio >= 0.8:
                scores.append(60.0)
            else:
                scores.append(35.0)
        
        return np.mean(scores) if scores else 50.0
    
    def calculate_cash_flow_score(self, operating_cash_flow_yoy: Optional[float]) -> float:
        """计算现金流得分"""
        if operating_cash_flow_yoy is None:
            return 50.0
        
        if operating_cash_flow_yoy >= 20:
            return 85.0
        elif operating_cash_flow_yoy >= 0:
            return 70.0
        elif operating_cash_flow_yoy >= -10:
            return 50.0
        else:
            return 30.0
    
    def calculate(self, stock_data: StockData, market_data: Dict) -> float:
        """计算财务健康度因子综合得分"""
        try:
            # 获取财务健康度数据
            health_data = self.get_financial_health_data(stock_data.code, stock_data.trade_date)
            
            if not health_data:
                return 50.0
            
            # 计算各项得分
            debt_score = self.calculate_debt_score(health_data.get('debt_to_assets'))
            liquidity_score = self.calculate_liquidity_score(
                health_data.get('current_ratio'),
                health_data.get('quick_ratio')
            )
            cash_flow_score = self.calculate_cash_flow_score(health_data.get('operating_cash_flow_yoy'))
            
            # 加权平均
            health_score = debt_score * 0.4 + liquidity_score * 0.3 + cash_flow_score * 0.3
            
            return min(100.0, max(0.0, health_score))
            
        except Exception as e:
            print(f"计算财务健康度因子失败: {e}")
            return 50.0
    
    def get_factor_name(self) -> str:
        return "FinancialHealthFactor"

class CompositeFundamentalFactor(FactorCalculator):
    """基本面综合因子计算器"""
    
    def __init__(self, db_path: str):
        """初始化基本面综合因子计算器"""
        self.valuation_factor = ValuationFactor(db_path)
        self.profitability_factor = ProfitabilityFactor(db_path)
        self.financial_health_factor = FinancialHealthFactor(db_path)
    
    def calculate(self, stock_data: StockData, market_data: Dict) -> float:
        """计算基本面综合得分"""
        try:
            # 计算各项基本面得分
            valuation_score = self.valuation_factor.calculate(stock_data, market_data)
            profitability_score = self.profitability_factor.calculate(stock_data, market_data)
            health_score = self.financial_health_factor.calculate(stock_data, market_data)
            
            # 权重配置：估值40%，盈利40%，健康度20%
            fundamental_score = (
                valuation_score * 0.4 +
                profitability_score * 0.4 +
                health_score * 0.2
            )
            
            return min(100.0, max(0.0, fundamental_score))
            
        except Exception as e:
            print(f"计算基本面综合因子失败: {e}")
            return 50.0
    
    def get_factor_name(self) -> str:
        return "CompositeFundamentalFactor"

if __name__ == "__main__":
    # 测试代码
    from datetime import datetime
    
    # 创建测试数据
    test_stock = StockData(
        code="000001",
        name="平安银行",
        trade_date="2025-08-01",
        open=10.0,
        high=10.5,
        low=9.8,
        close=10.2,
        volume=100000,
        pe_ttm=8.5,
        pb=0.6,
        ps_ttm=2.1
    )
    
    # 测试基本面因子
    factor = CompositeFundamentalFactor("data_adapter/stock_data.db")
    score = factor.calculate(test_stock, {"trade_date": "2025-08-01"})
    
    print(f"✅ 基本面因子计算完成")
    print(f"📊 测试股票 {test_stock.code} 基本面得分: {score:.2f}")