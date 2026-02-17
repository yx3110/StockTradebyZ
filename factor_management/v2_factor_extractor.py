#!/usr/bin/env python3
"""
V2选股系统因子提取器
将现有的scoring系统因子映射到因子管理框架
"""

import pandas as pd
import numpy as np
import sqlite3
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path

# 导入现有的因子计算器
import sys
sys.path.append('..')
from scoring.factor_calculator import FactorCalculator as V2FactorCalculator
from factor_manager import FactorManager

class V2FactorExtractor:
    """V2选股系统因子提取器"""
    
    def __init__(self, db_path: str = "data_adapter/stock_data.db"):
        self.db_path = db_path
        self.v2_calculator = V2FactorCalculator(db_path)
        self.factor_manager = FactorManager(db_path)
        self.logger = self._setup_logger()
        
        # 注册V2系统的所有因子
        self._register_v2_factors()
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger("V2FactorExtractor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _register_v2_factors(self):
        """注册V2系统的所有因子到因子管理框架"""
        
        # 1. 动量相关因子 (40%权重)
        self.factor_manager.register_factor(
            name="v2_price_momentum",
            category="technical",
            description="V2价格动量因子(短期+中期+加速度)",
            dependencies=["close"],
            calculator=self._calculate_v2_price_momentum,
            version="2.0"
        )
        
        self.factor_manager.register_factor(
            name="v2_volume_momentum", 
            category="technical",
            description="V2成交量动量因子(量价配合)",
            dependencies=["volume", "close"],
            calculator=self._calculate_v2_volume_momentum,
            version="2.0"
        )
        
        self.factor_manager.register_factor(
            name="v2_tech_momentum",
            category="technical", 
            description="V2技术指标动量(RSI+MACD)",
            dependencies=["rsi", "macd_dif", "macd_dea"],
            calculator=self._calculate_v2_tech_momentum,
            version="2.0"
        )
        
        self.factor_manager.register_factor(
            name="v2_trend_consistency",
            category="technical",
            description="V2趋势一致性因子",
            dependencies=["bbi", "close"],
            calculator=self._calculate_v2_trend_consistency,
            version="2.0"
        )
        
        # 2. 均值回归因子 (25%权重)
        self.factor_manager.register_factor(
            name="v2_mean_reversion",
            category="technical",
            description="V2均值回归因子(价格修复能力)",
            dependencies=["close", "bbi"],
            calculator=self._calculate_v2_mean_reversion,
            version="2.0"
        )
        
        # 3. 量价突破因子 (20%权重)
        self.factor_manager.register_factor(
            name="v2_volume_breakout",
            category="technical",
            description="V2量价突破因子(突破确认)",
            dependencies=["close", "volume", "high", "low"],
            calculator=self._calculate_v2_volume_breakout,
            version="2.0"
        )
        
        # 4. 相对强度因子 (10%权重)
        self.factor_manager.register_factor(
            name="v2_relative_performance",
            category="market",
            description="V2相对强度因子(相对市场表现)",
            dependencies=["close", "market_index"],
            calculator=self._calculate_v2_relative_performance,
            version="2.0"
        )
        
        # 5. 稳定性因子 (5%权重)
        self.factor_manager.register_factor(
            name="v2_stability",
            category="technical",
            description="V2稳定性因子(波动率控制)",
            dependencies=["close"],
            calculator=self._calculate_v2_stability,
            version="2.0"
        )
        
        # 6. V2综合评分
        self.factor_manager.register_factor(
            name="v2_composite_score",
            category="composite",
            description="V2综合评分(基于实际表现优化)",
            dependencies=["v2_momentum", "v2_mean_reversion", "v2_volume_breakout", 
                         "v2_relative_performance", "v2_stability"],
            calculator=self._calculate_v2_composite_score,
            version="2.0"
        )
        
        self.logger.info("已注册6个V2系统因子到因子管理框架")
    
    def _calculate_v2_price_momentum(self, df: pd.DataFrame) -> pd.Series:
        """计算V2价格动量因子"""
        if len(df) < 10:
            return pd.Series([50.0] * len(df), index=df.index)
        
        results = []
        for i in range(len(df)):
            if i < 5:
                results.append(50.0)
                continue
                
            # 使用V2计算器的逻辑
            window_df = df.iloc[max(0, i-19):i+1]
            
            # 短期动量 (5日)
            if len(window_df) >= 5:
                short_return = (window_df['close'].iloc[-1] / window_df['close'].iloc[-5] - 1) * 100
            else:
                short_return = 0
            
            # 中期动量 (10日)  
            if len(window_df) >= 10:
                medium_return = (window_df['close'].iloc[-1] / window_df['close'].iloc[-10] - 1) * 100
            else:
                medium_return = short_return
                
            # 动量加速度
            if len(window_df) >= 6:
                recent_momentum = (window_df['close'].iloc[-1] / window_df['close'].iloc[-3] - 1) * 100
                early_momentum = (window_df['close'].iloc[-3] / window_df['close'].iloc[-6] - 1) * 100
                acceleration = recent_momentum - early_momentum
            else:
                acceleration = 0
            
            # 综合评分
            momentum = short_return * 0.4 + medium_return * 0.4 + acceleration * 0.2
            score = 50 + np.tanh(momentum / 10) * 40
            results.append(max(0, min(100, score)))
        
        return pd.Series(results, index=df.index)
    
    def _calculate_v2_volume_momentum(self, df: pd.DataFrame) -> pd.Series:
        """计算V2成交量动量因子"""
        if len(df) < 5:
            return pd.Series([50.0] * len(df), index=df.index)
        
        results = []
        for i in range(len(df)):
            if i < 5:
                results.append(50.0)
                continue
            
            window_df = df.iloc[max(0, i-19):i+1]
            
            # 近期成交量 vs 历史平均
            recent_volume = window_df['volume'].tail(3).mean()
            avg_volume = window_df['volume'].mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            # 量价配合度
            if len(window_df) >= 5:
                price_change = window_df['close'].iloc[-1] / window_df['close'].iloc[-5] - 1
            else:
                price_change = 0
                
            volume_score = 50
            if price_change > 0 and volume_ratio > 1.2:  # 上涨配合放量
                volume_score = 70 + min(30, (volume_ratio - 1) * 30)
            elif price_change < 0 and volume_ratio < 0.8:  # 下跌伴随缩量
                volume_score = 30 + max(-20, volume_ratio * 25)
                
            results.append(max(0, min(100, volume_score)))
        
        return pd.Series(results, index=df.index)
    
    def _calculate_v2_tech_momentum(self, df: pd.DataFrame) -> pd.Series:
        """计算V2技术指标动量因子"""
        results = []
        for i in range(len(df)):
            score = 50
            
            # RSI动量
            if 'rsi' in df.columns and pd.notna(df['rsi'].iloc[i]):
                current_rsi = df['rsi'].iloc[i]
                if 30 < current_rsi < 70:  # RSI在健康区间
                    score += 10
                elif current_rsi < 30:  # 超卖
                    score += 15
                elif current_rsi > 80:  # 超买
                    score -= 10
            
            # MACD动量
            if 'macd_dif' in df.columns and 'macd_dea' in df.columns:
                current_macd = df['macd_dif'].iloc[i] if pd.notna(df['macd_dif'].iloc[i]) else 0
                current_signal = df['macd_dea'].iloc[i] if pd.notna(df['macd_dea'].iloc[i]) else 0
                
                if current_macd > current_signal and current_macd > 0:
                    score += 15
                elif current_macd < current_signal:
                    score -= 10
            
            results.append(max(0, min(100, score)))
        
        return pd.Series(results, index=df.index)
    
    def _calculate_v2_trend_consistency(self, df: pd.DataFrame) -> pd.Series:
        """计算V2趋势一致性因子"""
        results = []
        for i in range(len(df)):
            if i < 5:
                results.append(50.0)
                continue
            
            score = 50
            
            # BBI vs 收盘价
            if 'bbi' in df.columns and pd.notna(df['bbi'].iloc[i]):
                if df['close'].iloc[i] > df['bbi'].iloc[i]:
                    score += 20  # 价格在BBI上方
                else:
                    score -= 15  # 价格在BBI下方
            
            # 趋势连续性
            if i >= 3:
                recent_closes = df['close'].iloc[i-2:i+1]
                if (recent_closes.diff() > 0).sum() >= 2:  # 多数上涨
                    score += 10
                elif (recent_closes.diff() < 0).sum() >= 2:  # 多数下跌
                    score -= 10
            
            results.append(max(0, min(100, score)))
        
        return pd.Series(results, index=df.index)
    
    def _calculate_v2_mean_reversion(self, df: pd.DataFrame) -> pd.Series:
        """计算V2均值回归因子"""
        # 这里简化实现，实际应该调用V2的mean_reversion_factor方法
        results = []
        for i in range(len(df)):
            if i < 10:
                results.append(50.0)
                continue
            
            # 价格偏离度
            window_df = df.iloc[max(0, i-19):i+1]
            current_price = window_df['close'].iloc[-1]
            
            if 'bbi' in window_df.columns and pd.notna(window_df['bbi'].iloc[-1]):
                mean_price = window_df['bbi'].iloc[-1]
            else:
                mean_price = window_df['close'].rolling(10).mean().iloc[-1]
            
            deviation = (current_price - mean_price) / mean_price if mean_price > 0 else 0
            
            # 均值回归分数：偏离越大，回归潜力越高
            reversion_score = 50 + np.tanh(-deviation * 5) * 30
            results.append(max(0, min(100, reversion_score)))
        
        return pd.Series(results, index=df.index)
    
    def _calculate_v2_volume_breakout(self, df: pd.DataFrame) -> pd.Series:
        """计算V2量价突破因子"""
        results = []
        for i in range(len(df)):
            if i < 10:
                results.append(50.0)
                continue
            
            window_df = df.iloc[max(0, i-19):i+1]
            
            # 价格突破
            current_close = window_df['close'].iloc[-1]
            recent_high = window_df['high'].tail(5).max()
            price_breakout = current_close >= recent_high * 0.98
            
            # 成交量确认
            current_volume = window_df['volume'].iloc[-1]
            avg_volume = window_df['volume'].rolling(10).mean().iloc[-1]
            volume_confirm = current_volume > avg_volume * 1.2
            
            # 突破分数
            breakout_score = 50
            if price_breakout and volume_confirm:
                breakout_score = 85
            elif price_breakout:
                breakout_score = 70
            elif volume_confirm:
                breakout_score = 65
            
            results.append(max(0, min(100, breakout_score)))
        
        return pd.Series(results, index=df.index)
    
    def _calculate_v2_relative_performance(self, df: pd.DataFrame) -> pd.Series:
        """计算V2相对强度因子"""
        # 简化实现，应该与市场指数对比
        results = []
        for i in range(len(df)):
            if i < 10:
                results.append(50.0)
                continue
            
            window_df = df.iloc[max(0, i-19):i+1]
            
            # 相对收益率（这里简化为绝对收益）
            if len(window_df) >= 10:
                stock_return = (window_df['close'].iloc[-1] / window_df['close'].iloc[-10] - 1) * 100
                # 假设市场平均收益为2%
                market_return = 2.0
                relative_return = stock_return - market_return
                
                relative_score = 50 + np.tanh(relative_return / 5) * 30
            else:
                relative_score = 50
            
            results.append(max(0, min(100, relative_score)))
        
        return pd.Series(results, index=df.index)
    
    def _calculate_v2_stability(self, df: pd.DataFrame) -> pd.Series:
        """计算V2稳定性因子"""
        results = []
        for i in range(len(df)):
            if i < 10:
                results.append(50.0)
                continue
            
            window_df = df.iloc[max(0, i-19):i+1]
            
            # 波动率
            returns = window_df['close'].pct_change().dropna()
            if len(returns) > 0:
                volatility = returns.std() * np.sqrt(252)  # 年化波动率
                # 稳定性分数：波动率越低，稳定性越高
                stability_score = 50 + np.tanh(-volatility + 0.3) * 25
            else:
                stability_score = 50
            
            results.append(max(0, min(100, stability_score)))
        
        return pd.Series(results, index=df.index)
    
    def _calculate_v2_composite_score(self, df: pd.DataFrame) -> pd.Series:
        """计算V2综合评分"""
        # V2权重配置
        weights = {
            'momentum': 0.40,        # 动量因子
            'mean_reversion': 0.25,  # 均值回归  
            'volume_breakout': 0.20, # 量价突破
            'relative_performance': 0.10,  # 相对强度
            'stability': 0.05        # 稳定性
        }
        
        # 计算各子因子
        momentum = (
            self._calculate_v2_price_momentum(df) * 0.30 +
            self._calculate_v2_volume_momentum(df) * 0.25 +
            self._calculate_v2_tech_momentum(df) * 0.25 +
            self._calculate_v2_trend_consistency(df) * 0.20
        )
        
        mean_reversion = self._calculate_v2_mean_reversion(df)
        volume_breakout = self._calculate_v2_volume_breakout(df)
        relative_performance = self._calculate_v2_relative_performance(df)
        stability = self._calculate_v2_stability(df)
        
        # 综合评分
        composite_score = (
            momentum * weights['momentum'] +
            mean_reversion * weights['mean_reversion'] +
            volume_breakout * weights['volume_breakout'] +
            relative_performance * weights['relative_performance'] +
            stability * weights['stability']
        )
        
        return composite_score
    
    def extract_factors_for_stock(self, stock_code: str, 
                                 start_date: str, 
                                 end_date: str) -> pd.DataFrame:
        """为单只股票提取所有V2因子"""
        
        self.logger.info(f"提取股票 {stock_code} 的V2因子数据...")
        
        # 获取股票原始数据
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT 
                    dq.trade_date,
                    dq.close,
                    dq.high,
                    dq.low, 
                    dq.volume,
                    ti.bbi,
                    ti.rsi6 as rsi,
                    ti.macd_dif,
                    ti.macd_dea
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                LEFT JOIN technical_indicators ti ON ti.security_id = s.id AND ti.trade_date = dq.trade_date
                WHERE s.code = ? 
                AND dq.trade_date BETWEEN ? AND ?
                ORDER BY dq.trade_date
            """
            
            df = pd.read_sql_query(query, conn, params=[stock_code, start_date, end_date])
        
        if df.empty:
            self.logger.warning(f"未找到股票 {stock_code} 的数据")
            return pd.DataFrame()
        
        # 计算所有V2因子
        result_df = df[['trade_date']].copy()
        result_df['stock_code'] = stock_code
        
        # 添加各因子
        result_df['v2_price_momentum'] = self._calculate_v2_price_momentum(df)
        result_df['v2_volume_momentum'] = self._calculate_v2_volume_momentum(df)
        result_df['v2_tech_momentum'] = self._calculate_v2_tech_momentum(df)
        result_df['v2_trend_consistency'] = self._calculate_v2_trend_consistency(df)
        result_df['v2_mean_reversion'] = self._calculate_v2_mean_reversion(df)
        result_df['v2_volume_breakout'] = self._calculate_v2_volume_breakout(df)
        result_df['v2_relative_performance'] = self._calculate_v2_relative_performance(df)
        result_df['v2_stability'] = self._calculate_v2_stability(df)
        result_df['v2_composite_score'] = self._calculate_v2_composite_score(df)
        
        return result_df
    
    def batch_extract_factors(self, stock_codes: List[str],
                            start_date: str,
                            end_date: str,
                            batch_size: int = 100) -> pd.DataFrame:
        """批量提取V2因子数据"""
        
        self.logger.info(f"批量提取 {len(stock_codes)} 只股票的V2因子...")
        
        all_results = []
        
        for i in range(0, len(stock_codes), batch_size):
            batch_codes = stock_codes[i:i+batch_size]
            batch_results = []
            
            for code in batch_codes:
                try:
                    factor_data = self.extract_factors_for_stock(code, start_date, end_date)
                    if not factor_data.empty:
                        batch_results.append(factor_data)
                except Exception as e:
                    self.logger.error(f"提取股票 {code} 因子失败: {e}")
                    continue
            
            if batch_results:
                batch_df = pd.concat(batch_results, ignore_index=True)
                all_results.append(batch_df)
                
            self.logger.info(f"完成批次 {i//batch_size + 1}/{(len(stock_codes)-1)//batch_size + 1}")
        
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            self.logger.info(f"成功提取 {len(final_df)} 条V2因子记录")
            return final_df
        else:
            return pd.DataFrame()
    
    def save_factors_to_database(self, factor_data: pd.DataFrame):
        """将V2因子数据保存到数据库"""
        
        self.logger.info(f"保存 {len(factor_data)} 条V2因子数据到数据库...")
        
        with sqlite3.connect(self.db_path) as conn:
            # 创建v2_factors表（如果不存在）
            conn.execute("""
                CREATE TABLE IF NOT EXISTS v2_factors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    security_id INTEGER,
                    trade_date DATE,
                    v2_price_momentum REAL,
                    v2_volume_momentum REAL,
                    v2_tech_momentum REAL,
                    v2_trend_consistency REAL,
                    v2_mean_reversion REAL,
                    v2_volume_breakout REAL,
                    v2_relative_performance REAL,
                    v2_stability REAL,
                    v2_composite_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(security_id, trade_date)
                )
            """)
            
            # 为每条记录获取security_id
            for _, row in factor_data.iterrows():
                try:
                    # 获取security_id
                    security_id_query = "SELECT id FROM securities WHERE code = ?"
                    security_result = conn.execute(security_id_query, [row['stock_code']]).fetchone()
                    
                    if security_result:
                        security_id = security_result[0]
                        
                        # 插入因子数据
                        conn.execute("""
                            INSERT OR REPLACE INTO v2_factors 
                            (security_id, trade_date, v2_price_momentum, v2_volume_momentum,
                             v2_tech_momentum, v2_trend_consistency, v2_mean_reversion,
                             v2_volume_breakout, v2_relative_performance, v2_stability, 
                             v2_composite_score)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            security_id,
                            row['trade_date'],
                            row['v2_price_momentum'],
                            row['v2_volume_momentum'],
                            row['v2_tech_momentum'],
                            row['v2_trend_consistency'],
                            row['v2_mean_reversion'],
                            row['v2_volume_breakout'],
                            row['v2_relative_performance'],
                            row['v2_stability'],
                            row['v2_composite_score']
                        ))
                except Exception as e:
                    self.logger.error(f"保存股票 {row['stock_code']} 数据失败: {e}")
                    continue
            
            conn.commit()
            self.logger.info("V2因子数据保存完成")


def main():
    """测试V2因子提取器"""
    
    extractor = V2FactorExtractor()
    
    # 测试单只股票
    print("测试提取单只股票因子...")
    factor_data = extractor.extract_factors_for_stock(
        '000001', '2025-01-01', '2025-08-18'
    )
    print(f"提取到 {len(factor_data)} 条记录")
    print(factor_data.head())
    
    # 测试批量提取
    print("\n测试批量提取因子...")
    test_stocks = ['000001', '000002', '000858']
    batch_data = extractor.batch_extract_factors(
        test_stocks, '2025-08-01', '2025-08-18'
    )
    print(f"批量提取到 {len(batch_data)} 条记录")
    
    # 保存到数据库
    if not batch_data.empty:
        extractor.save_factors_to_database(batch_data)


if __name__ == "__main__":
    main()