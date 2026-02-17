#!/usr/bin/env python3
"""
新一代量化评分核心框架
Enhanced Multi-Factor Scoring System Core Framework

基于相关性分析发现的问题，重新设计评分体系架构：
- 技术面(30%) + 基本面(35%) + 资金面(20%) + 市场面(15%)
- 动态权重调整
- 市场状态识别
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path

@dataclass
class ScoringConfig:
    """评分配置类"""
    # 基础权重
    technical_weight: float = 0.30  # 技术面权重
    fundamental_weight: float = 0.35  # 基本面权重
    capital_weight: float = 0.20  # 资金面权重
    market_weight: float = 0.15  # 市场面权重
    
    # 动态权重调整参数
    volatility_threshold: float = 0.02  # 波动率阈值
    volume_threshold: float = 1.5  # 成交量阈值
    
    # 市场状态参数
    bull_market_threshold: float = 0.05  # 牛市判断阈值
    bear_market_threshold: float = -0.05  # 熊市判断阈值
    
    # 评分范围
    min_score: float = 0.0
    max_score: float = 100.0

class MarketState:
    """市场状态枚举"""
    BULL = "bull"  # 牛市
    BEAR = "bear"  # 熊市
    SIDEWAYS = "sideways"  # 震荡市

@dataclass
class StockData:
    """股票数据结构"""
    code: str
    name: str
    trade_date: str
    
    # 价格数据
    open: float
    high: float
    low: float
    close: float
    volume: int
    
    # 基本面数据
    pe_ttm: Optional[float] = None
    pb: Optional[float] = None
    ps_ttm: Optional[float] = None
    market_cap: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    
    # 技术指标
    ma5: Optional[float] = None
    ma10: Optional[float] = None
    ma20: Optional[float] = None
    ma60: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[float] = None
    kdj_k: Optional[float] = None
    kdj_d: Optional[float] = None
    kdj_j: Optional[float] = None

@dataclass
class ScoringResult:
    """评分结果结构"""
    stock_code: str
    stock_name: str
    trade_date: str
    
    # 各维度得分
    technical_score: float
    fundamental_score: float
    capital_score: float
    market_score: float
    
    # 综合得分
    total_score: float
    
    # 权重信息
    weights: Dict[str, float]
    
    # 详细信息
    market_state: str
    risk_level: str
    recommendation: str

class FactorCalculator(ABC):
    """因子计算器抽象基类"""
    
    @abstractmethod
    def calculate(self, stock_data: StockData, market_data: Dict) -> float:
        """计算因子得分"""
        pass
    
    @abstractmethod
    def get_factor_name(self) -> str:
        """获取因子名称"""
        pass

class ScoringEngine:
    """评分引擎核心类"""
    
    def __init__(self, config: ScoringConfig, db_path: str = "data_adapter/stock_data.db"):
        """
        初始化评分引擎
        
        Args:
            config: 评分配置
            db_path: 数据库路径
        """
        self.config = config
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        
        # 因子计算器注册表
        self.technical_factors: List[FactorCalculator] = []
        self.fundamental_factors: List[FactorCalculator] = []
        self.capital_factors: List[FactorCalculator] = []
        self.market_factors: List[FactorCalculator] = []
        
        # 市场状态缓存
        self.market_state_cache: Dict[str, str] = {}
        
    def register_factor(self, factor_calculator: FactorCalculator, factor_type: str):
        """
        注册因子计算器
        
        Args:
            factor_calculator: 因子计算器实例
            factor_type: 因子类型 ('technical', 'fundamental', 'capital', 'market')
        """
        if factor_type == 'technical':
            self.technical_factors.append(factor_calculator)
        elif factor_type == 'fundamental':
            self.fundamental_factors.append(factor_calculator)
        elif factor_type == 'capital':
            self.capital_factors.append(factor_calculator)
        elif factor_type == 'market':
            self.market_factors.append(factor_calculator)
        else:
            raise ValueError(f"Unsupported factor type: {factor_type}")
    
    def get_stock_data(self, stock_code: str, trade_date: str) -> Optional[StockData]:
        """获取股票数据"""
        try:
            # 查询基础价格数据
            price_query = """
                SELECT dq.open, dq.high, dq.low, dq.close, dq.volume,
                       dq.ma5, dq.ma10, dq.ma20, dq.ma60,
                       s.name
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                WHERE s.code = ? AND dq.trade_date = ?
            """
            
            price_df = pd.read_sql_query(price_query, self.conn, params=[stock_code, trade_date])
            
            if price_df.empty:
                return None
            
            price_row = price_df.iloc[0]
            
            # 查询基本面数据
            fundamental_query = """
                SELECT pe_ttm, pb, ps_ttm, market_cap
                FROM daily_basic db
                JOIN securities s ON db.security_id = s.id
                WHERE s.code = ? AND db.trade_date = ?
            """
            
            fundamental_df = pd.read_sql_query(fundamental_query, self.conn, params=[stock_code, trade_date])
            
            # 查询技术指标数据
            technical_query = """
                SELECT rsi, macd_dif, kdj_k, kdj_d, kdj_j
                FROM technical_indicators ti
                JOIN securities s ON ti.security_id = s.id
                WHERE s.code = ? AND ti.trade_date = ?
            """
            
            technical_df = pd.read_sql_query(technical_query, self.conn, params=[stock_code, trade_date])
            
            # 构建StockData对象
            stock_data = StockData(
                code=stock_code,
                name=price_row['name'],
                trade_date=trade_date,
                open=price_row['open'],
                high=price_row['high'],
                low=price_row['low'],
                close=price_row['close'],
                volume=price_row['volume'],
                ma5=price_row.get('ma5'),
                ma10=price_row.get('ma10'),
                ma20=price_row.get('ma20'),
                ma60=price_row.get('ma60')
            )
            
            # 添加基本面数据
            if not fundamental_df.empty:
                fund_row = fundamental_df.iloc[0]
                stock_data.pe_ttm = fund_row.get('pe_ttm')
                stock_data.pb = fund_row.get('pb')
                stock_data.ps_ttm = fund_row.get('ps_ttm')
                stock_data.market_cap = fund_row.get('market_cap')
            
            # 添加技术指标数据
            if not technical_df.empty:
                tech_row = technical_df.iloc[0]
                stock_data.rsi = tech_row.get('rsi')
                stock_data.macd = tech_row.get('macd_dif')
                stock_data.kdj_k = tech_row.get('kdj_k')
                stock_data.kdj_d = tech_row.get('kdj_d')
                stock_data.kdj_j = tech_row.get('kdj_j')
            
            return stock_data
            
        except Exception as e:
            print(f"获取股票 {stock_code} 数据失败: {e}")
            return None
    
    def detect_market_state(self, trade_date: str) -> str:
        """
        识别市场状态
        
        Args:
            trade_date: 交易日期
            
        Returns:
            市场状态字符串
        """
        if trade_date in self.market_state_cache:
            return self.market_state_cache[trade_date]
        
        try:
            # 查询大盘指数数据（上证指数）
            index_query = """
                SELECT dq.close, dq.price_change_pct
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                WHERE s.code = '000001' AND s.type = '指数'
                AND dq.trade_date <= ?
                ORDER BY dq.trade_date DESC
                LIMIT 20
            """
            
            df = pd.read_sql_query(index_query, self.conn, params=[trade_date])
            
            if len(df) < 10:
                return MarketState.SIDEWAYS
            
            # 计算近期平均涨跌幅
            avg_change = df['price_change_pct'].head(10).mean()
            
            # 判断市场状态
            if avg_change > self.config.bull_market_threshold:
                state = MarketState.BULL
            elif avg_change < self.config.bear_market_threshold:
                state = MarketState.BEAR
            else:
                state = MarketState.SIDEWAYS
            
            self.market_state_cache[trade_date] = state
            return state
            
        except Exception as e:
            print(f"识别市场状态失败: {e}")
            return MarketState.SIDEWAYS
    
    def calculate_dynamic_weights(self, stock_data: StockData, market_state: str) -> Dict[str, float]:
        """
        计算动态权重
        
        Args:
            stock_data: 股票数据
            market_state: 市场状态
            
        Returns:
            动态权重字典
        """
        # 基础权重
        weights = {
            'technical': self.config.technical_weight,
            'fundamental': self.config.fundamental_weight,
            'capital': self.config.capital_weight,
            'market': self.config.market_weight
        }
        
        # 根据市场状态调整权重
        if market_state == MarketState.BULL:
            # 牛市：增加技术面和资金面权重
            weights['technical'] *= 1.2
            weights['capital'] *= 1.3
            weights['fundamental'] *= 0.8
            weights['market'] *= 0.9
        elif market_state == MarketState.BEAR:
            # 熊市：增加基本面和市场面权重
            weights['fundamental'] *= 1.3
            weights['market'] *= 1.2
            weights['technical'] *= 0.7
            weights['capital'] *= 0.8
        else:
            # 震荡市：均衡权重
            pass
        
        # 计算个股波动率调整权重
        if stock_data.ma20 and stock_data.close:
            volatility = abs(stock_data.close - stock_data.ma20) / stock_data.ma20
            if volatility > self.config.volatility_threshold:
                # 高波动股票降低技术面权重
                weights['technical'] *= 0.8
                weights['fundamental'] *= 1.1
        
        # 标准化权重
        total_weight = sum(weights.values())
        for key in weights:
            weights[key] /= total_weight
        
        return weights
    
    def calculate_factor_scores(self, stock_data: StockData, market_data: Dict) -> Dict[str, float]:
        """计算各维度因子得分"""
        scores = {}
        
        # 计算技术面得分
        technical_scores = []
        for factor in self.technical_factors:
            try:
                score = factor.calculate(stock_data, market_data)
                if score is not None:
                    technical_scores.append(score)
            except Exception as e:
                print(f"技术因子 {factor.get_factor_name()} 计算失败: {e}")
        
        scores['technical'] = np.mean(technical_scores) if technical_scores else 50.0
        
        # 计算基本面得分
        fundamental_scores = []
        for factor in self.fundamental_factors:
            try:
                score = factor.calculate(stock_data, market_data)
                if score is not None:
                    fundamental_scores.append(score)
            except Exception as e:
                print(f"基本面因子 {factor.get_factor_name()} 计算失败: {e}")
        
        scores['fundamental'] = np.mean(fundamental_scores) if fundamental_scores else 50.0
        
        # 计算资金面得分
        capital_scores = []
        for factor in self.capital_factors:
            try:
                score = factor.calculate(stock_data, market_data)
                if score is not None:
                    capital_scores.append(score)
            except Exception as e:
                print(f"资金面因子 {factor.get_factor_name()} 计算失败: {e}")
        
        scores['capital'] = np.mean(capital_scores) if capital_scores else 50.0
        
        # 计算市场面得分
        market_scores = []
        for factor in self.market_factors:
            try:
                score = factor.calculate(stock_data, market_data)
                if score is not None:
                    market_scores.append(score)
            except Exception as e:
                print(f"市场面因子 {factor.get_factor_name()} 计算失败: {e}")
        
        scores['market'] = np.mean(market_scores) if market_scores else 50.0
        
        return scores
    
    def calculate_risk_level(self, stock_data: StockData, total_score: float) -> str:
        """计算风险等级"""
        if total_score >= 80:
            return "低风险"
        elif total_score >= 60:
            return "中等风险"
        else:
            return "高风险"
    
    def generate_recommendation(self, total_score: float, risk_level: str) -> str:
        """生成投资建议"""
        if total_score >= 75 and risk_level in ["低风险", "中等风险"]:
            return "强烈买入"
        elif total_score >= 65:
            return "买入"
        elif total_score >= 55:
            return "谨慎买入"
        elif total_score >= 45:
            return "观望"
        else:
            return "回避"
    
    def score_stock(self, stock_code: str, trade_date: str) -> Optional[ScoringResult]:
        """
        对单只股票进行评分
        
        Args:
            stock_code: 股票代码
            trade_date: 交易日期
            
        Returns:
            评分结果
        """
        try:
            # 获取股票数据
            stock_data = self.get_stock_data(stock_code, trade_date)
            if not stock_data:
                return None
            
            # 识别市场状态
            market_state = self.detect_market_state(trade_date)
            
            # 计算动态权重
            weights = self.calculate_dynamic_weights(stock_data, market_state)
            
            # 获取市场数据（用于因子计算）
            market_data = {
                'market_state': market_state,
                'trade_date': trade_date
            }
            
            # 计算各维度得分
            factor_scores = self.calculate_factor_scores(stock_data, market_data)
            
            # 计算综合得分
            total_score = (
                factor_scores['technical'] * weights['technical'] +
                factor_scores['fundamental'] * weights['fundamental'] +
                factor_scores['capital'] * weights['capital'] +
                factor_scores['market'] * weights['market']
            )
            
            # 确保得分在合理范围内
            total_score = max(self.config.min_score, min(self.config.max_score, total_score))
            
            # 计算风险等级和投资建议
            risk_level = self.calculate_risk_level(stock_data, total_score)
            recommendation = self.generate_recommendation(total_score, risk_level)
            
            return ScoringResult(
                stock_code=stock_code,
                stock_name=stock_data.name,
                trade_date=trade_date,
                technical_score=factor_scores['technical'],
                fundamental_score=factor_scores['fundamental'],
                capital_score=factor_scores['capital'],
                market_score=factor_scores['market'],
                total_score=total_score,
                weights=weights,
                market_state=market_state,
                risk_level=risk_level,
                recommendation=recommendation
            )
            
        except Exception as e:
            print(f"评分股票 {stock_code} 失败: {e}")
            return None
    
    def batch_score(self, stock_codes: List[str], trade_date: str) -> List[ScoringResult]:
        """
        批量评分
        
        Args:
            stock_codes: 股票代码列表
            trade_date: 交易日期
            
        Returns:
            评分结果列表
        """
        results = []
        
        for stock_code in stock_codes:
            result = self.score_stock(stock_code, trade_date)
            if result:
                results.append(result)
        
        # 按总分降序排序
        results.sort(key=lambda x: x.total_score, reverse=True)
        
        return results
    
    def save_results_to_db(self, results: List[ScoringResult], table_name: str = "enhanced_stock_scores"):
        """将评分结果保存到数据库"""
        try:
            # 创建表（如果不存在）
            create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stock_code VARCHAR(10) NOT NULL,
                    stock_name VARCHAR(50),
                    trade_date DATE NOT NULL,
                    technical_score DECIMAL(5,2),
                    fundamental_score DECIMAL(5,2),
                    capital_score DECIMAL(5,2),
                    market_score DECIMAL(5,2),
                    total_score DECIMAL(5,2),
                    technical_weight DECIMAL(5,4),
                    fundamental_weight DECIMAL(5,4),
                    capital_weight DECIMAL(5,4),
                    market_weight DECIMAL(5,4),
                    market_state VARCHAR(20),
                    risk_level VARCHAR(20),
                    recommendation VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(stock_code, trade_date)
                )
            """
            
            self.conn.execute(create_table_sql)
            
            # 插入数据
            for result in results:
                insert_sql = f"""
                    INSERT OR REPLACE INTO {table_name} 
                    (stock_code, stock_name, trade_date, technical_score, fundamental_score, 
                     capital_score, market_score, total_score, technical_weight, fundamental_weight,
                     capital_weight, market_weight, market_state, risk_level, recommendation)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                
                self.conn.execute(insert_sql, (
                    result.stock_code, result.stock_name, result.trade_date,
                    result.technical_score, result.fundamental_score,
                    result.capital_score, result.market_score, result.total_score,
                    result.weights['technical'], result.weights['fundamental'],
                    result.weights['capital'], result.weights['market'],
                    result.market_state, result.risk_level, result.recommendation
                ))
            
            self.conn.commit()
            print(f"成功保存 {len(results)} 条评分结果到数据库")
            
        except Exception as e:
            print(f"保存评分结果到数据库失败: {e}")
            self.conn.rollback()
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'conn'):
            self.conn.close()

def create_default_config() -> ScoringConfig:
    """创建默认配置"""
    return ScoringConfig(
        technical_weight=0.30,
        fundamental_weight=0.35,
        capital_weight=0.20,
        market_weight=0.15,
        volatility_threshold=0.02,
        volume_threshold=1.5,
        bull_market_threshold=0.05,
        bear_market_threshold=-0.05,
        min_score=0.0,
        max_score=100.0
    )

if __name__ == "__main__":
    # 测试代码
    config = create_default_config()
    
    # 使用相对路径
    db_path = "../data_adapter/stock_data.db"
    if not Path(db_path).exists():
        print("⚠️ 数据库文件未找到，使用模拟模式")
        db_path = ":memory:"  # 使用内存数据库进行测试
    
    try:
        engine = ScoringEngine(config, db_path)
        print("✅ 新一代量化评分核心框架已初始化")
        print(f"📊 配置信息:")
        print(f"  - 技术面权重: {config.technical_weight}")
        print(f"  - 基本面权重: {config.fundamental_weight}")
        print(f"  - 资金面权重: {config.capital_weight}")
        print(f"  - 市场面权重: {config.market_weight}")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        print("请确保数据库文件存在或检查路径配置")