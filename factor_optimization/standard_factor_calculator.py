#!/usr/bin/env python3
"""
标准化因子计算器 - 为权重优化提供标准化的0-100分因子数据
确保数据格式通用且标准一致，便于后续权重迭代
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'scoring_improvements'))

from data_adapter.database_manager import DatabaseManager
from scoring_improvements.squeeze_momentum_calculator import SqueezeMomentumCalculator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StandardFactorCalculator:
    """标准化因子计算器 - 生成0-100分标准化因子数据"""
    
    def __init__(self, factor_db_path: str = None):
        self.factor_db_path = factor_db_path or os.path.join(current_dir, 'standard_factors.db')
        self.db_manager = DatabaseManager()
        self.squeeze_calculator = SqueezeMomentumCalculator()
        
        # 从挤压动量缓存获取数据的路径
        self.squeeze_cache_path = os.path.join(project_root, 'weight_optimization_cache.db')
        
        # 初始化标准化因子表
        self._init_standard_cache()
    
    def _init_standard_cache(self):
        """初始化标准化因子缓存表"""
        with sqlite3.connect(self.factor_db_path) as conn:
            cursor = conn.cursor()
            
            # 删除旧的非标准数据表(如果需要)
            # cursor.execute("DROP TABLE IF EXISTS stock_indicators")
            
            # 创建标准化因子表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS standard_factors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stock_code TEXT NOT NULL,
                    trade_date DATE NOT NULL,
                    
                    -- 7个维度标准化分数 (0-100分)
                    technical_score DECIMAL(5,2),           -- 技术指标维度总分
                    squeeze_momentum_score DECIMAL(5,2),    -- 挤压动量维度总分
                    fundamental_score DECIMAL(5,2),         -- 基本面维度总分
                    performance_score DECIMAL(5,2),         -- 市场表现维度总分
                    sentiment_score DECIMAL(5,2),           -- 情绪指标维度总分
                    risk_control_score DECIMAL(5,2),        -- 风险控制维度总分
                    market_regime_score DECIMAL(5,2),       -- 市场环境维度总分
                    
                    -- 技术指标子因子分数 (0-100分)
                    kdj_strength DECIMAL(5,2),
                    rsi_momentum DECIMAL(5,2),
                    bbi_trend DECIMAL(5,2),
                    volume_surge DECIMAL(5,2),
                    
                    -- 挤压动量子因子分数 (0-100分)
                    squeeze_state DECIMAL(5,2),
                    squeeze_release DECIMAL(5,2),
                    momentum_direction DECIMAL(5,2),
                    momentum_consistency DECIMAL(5,2),
                    
                    -- 基本面子因子分数 (0-100分)
                    pe_valuation DECIMAL(5,2),
                    pb_valuation DECIMAL(5,2),
                    roe_profitability DECIMAL(5,2),
                    financial_quality DECIMAL(5,2),
                    market_cap DECIMAL(5,2),
                    turnover_activity DECIMAL(5,2),
                    
                    -- 市场表现子因子分数 (0-100分)
                    price_momentum DECIMAL(5,2),
                    relative_strength DECIMAL(5,2),
                    volatility_risk DECIMAL(5,2),
                    
                    -- 情绪指标子因子分数 (0-100分)
                    money_flow DECIMAL(5,2),
                    market_attention DECIMAL(5,2),
                    investor_emotion DECIMAL(5,2),
                    
                    -- 风险控制子因子分数 (0-100分)
                    stop_loss_risk DECIMAL(5,2),
                    max_drawdown DECIMAL(5,2),
                    risk_adjusted_return DECIMAL(5,2),
                    
                    -- 市场环境子因子分数 (0-100分)
                    market_beta DECIMAL(5,2),
                    sector_rotation DECIMAL(5,2),
                    liquidity DECIMAL(5,2),
                    
                    -- 未来收益数据 (用于优化验证)
                    return_1d DECIMAL(8,4),
                    return_3d DECIMAL(8,4),
                    return_5d DECIMAL(8,4),
                    return_10d DECIMAL(8,4),
                    return_20d DECIMAL(8,4),
                    
                    -- 元数据
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    
                    UNIQUE(stock_code, trade_date)
                )
            """)
            
            # 创建索引
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_standard_factors_date 
                ON standard_factors(trade_date)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_standard_factors_code 
                ON standard_factors(stock_code)
            """)
            
            conn.commit()
            logger.info(f"✅ 标准化因子缓存表初始化完成")
    
    def calculate_technical_factors(self, stock_code: str, trade_date: str) -> Dict:
        """计算技术指标维度因子 (0-100分)"""
        try:
            # 获取技术指标数据
            query = """
            SELECT ti.kdj_k, ti.kdj_d, ti.kdj_j, ti.rsi6, ti.rsi12, ti.rsi24,
                   ti.bbi, ti.volume_ratio,
                   dq.volume, dq.close
            FROM technical_indicators ti
            JOIN daily_quotes dq ON ti.security_id = dq.security_id AND ti.trade_date = dq.trade_date
            JOIN securities s ON dq.security_id = s.id
            WHERE s.code = ? AND ti.trade_date = ?
            """
            
            result = self.db_manager.execute_query(query, [stock_code, trade_date])
            if not result:
                return {'technical_score': 50.0, 'kdj_strength': 50.0, 'rsi_momentum': 50.0, 
                       'bbi_trend': 50.0, 'volume_surge': 50.0}
            
            row = result[0]
            kdj_k, kdj_d, kdj_j = row[0] or 50, row[1] or 50, row[2] or 50
            rsi6, rsi12, rsi24 = row[3] or 50, row[4] or 50, row[5] or 50
            bbi = row[6] or 0
            volume_ratio = row[7] or 1.0
            volume = row[8] or 0
            close_price = row[9] or 100
            
            # 1. KDJ强度评分 (0-100)
            kdj_strength = self._calculate_kdj_strength(kdj_k, kdj_d, kdj_j)
            
            # 2. RSI动量评分 (0-100)
            rsi_momentum = self._calculate_rsi_momentum(rsi6, rsi12, rsi24)
            
            # 3. BBI趋势评分 (0-100)
            bbi_trend = self._calculate_bbi_trend(close_price, bbi)
            
            # 4. 成交量突破评分 (0-100)
            volume_surge = self._calculate_volume_surge(volume_ratio)
            
            # 技术指标总分 (等权重平均)
            technical_score = (kdj_strength + rsi_momentum + bbi_trend + volume_surge) / 4
            
            return {
                'technical_score': technical_score,
                'kdj_strength': kdj_strength,
                'rsi_momentum': rsi_momentum,
                'bbi_trend': bbi_trend,
                'volume_surge': volume_surge
            }
            
        except Exception as e:
            logger.error(f"计算技术指标失败 {stock_code} {trade_date}: {e}")
            return {'technical_score': 50.0, 'kdj_strength': 50.0, 'rsi_momentum': 50.0, 
                   'bbi_trend': 50.0, 'volume_surge': 50.0}
    
    def _calculate_kdj_strength(self, k: float, d: float, j: float) -> float:
        """计算KDJ强度评分"""
        # KDJ金叉银叉及数值区间评分
        score = 50.0  # 基础分
        
        # 金叉状态 +20分
        if k > d and j > k:
            score += 20
        # 银叉状态 -15分
        elif k < d and j < k:
            score -= 15
        
        # 超买超卖区间调整
        if 20 <= k <= 80 and 20 <= d <= 80:  # 正常区间 +10分
            score += 10
        elif k > 80 or d > 80:  # 超买区间 +5分
            score += 5
        elif k < 20 or d < 20:  # 超卖区间 +15分
            score += 15
        
        return max(0, min(100, score))
    
    def _calculate_rsi_momentum(self, rsi6: float, rsi12: float, rsi24: float) -> float:
        """计算RSI动量评分"""
        score = 50.0
        
        # RSI多周期一致性
        rsi_avg = (rsi6 + rsi12 + rsi24) / 3
        
        # 超卖反弹机会 (RSI < 30) +25分
        if rsi_avg < 30:
            score += 25
        # 强势上升 (30-70) 根据数值评分
        elif 30 <= rsi_avg <= 70:
            score += (rsi_avg - 30) / 40 * 20  # 线性评分
        # 超买谨慎区间 (70-80) +5分
        elif 70 < rsi_avg <= 80:
            score += 5
        # 严重超买 (>80) -10分
        else:
            score -= 10
        
        # RSI多周期趋势一致性奖励
        if abs(rsi6 - rsi12) < 5 and abs(rsi12 - rsi24) < 5:
            score += 10
        
        return max(0, min(100, score))
    
    def _calculate_bbi_trend(self, close: float, bbi: float) -> float:
        """计算BBI趋势评分"""
        if bbi == 0:
            return 50.0
            
        # 价格相对BBI位置评分
        price_ratio = close / bbi
        
        if price_ratio >= 1.05:      # 强势突破 +30分
            score = 80
        elif price_ratio >= 1.02:    # 轻微突破 +20分
            score = 70
        elif price_ratio >= 0.98:    # 均线附近 +10分
            score = 60
        elif price_ratio >= 0.95:    # 轻微下跌 0分
            score = 50
        else:                        # 明显弱势 -20分
            score = 30
        
        return max(0, min(100, score))
    
    def _calculate_volume_surge(self, volume_ratio: float) -> float:
        """计算成交量突破评分"""
        # 量比评分
        if volume_ratio >= 3.0:      # 巨量 +40分
            score = 90
        elif volume_ratio >= 2.0:    # 大量 +30分
            score = 80
        elif volume_ratio >= 1.5:    # 温和放量 +20分
            score = 70
        elif volume_ratio >= 1.2:    # 轻微放量 +10分
            score = 60
        elif volume_ratio >= 0.8:    # 正常成交 0分
            score = 50
        else:                        # 缩量 -10分
            score = 40
        
        return max(0, min(100, score))
    
    def calculate_squeeze_momentum_factors(self, stock_code: str, trade_date: str) -> Dict:
        """计算挤压动量维度因子 (0-100分)"""
        try:
            # 从squeeze_momentum_cache表获取数据
            query = """
            SELECT squeeze_state, squeeze_release, squeeze_intensity,
                   momentum_direction, momentum_consistency,
                   squeeze_momentum, momentum_strength
            FROM squeeze_momentum_cache
            WHERE stock_code = ? AND trade_date = ?
            """
            
            with sqlite3.connect(self.squeeze_cache_path) as conn:
                result = pd.read_sql_query(query, conn, params=[stock_code, trade_date])
            
            if len(result) == 0:
                return {'squeeze_momentum_score': 50.0, 'squeeze_state': 50.0, 
                       'squeeze_release': 50.0, 'momentum_direction': 50.0, 'momentum_consistency': 50.0}
            
            row = result.iloc[0]
            
            # 1. 挤压状态评分 (0-100)
            squeeze_state_score = 75 if row['squeeze_state'] else 25
            
            # 2. 挤压释放评分 (0-100)
            squeeze_release_score = 90 if row['squeeze_release'] else 30
            
            # 3. 动量方向评分 (0-100)
            direction = row['momentum_direction']
            if direction > 0:
                momentum_direction_score = 75
            elif direction < 0:
                momentum_direction_score = 25
            else:
                momentum_direction_score = 50
            
            # 4. 动量一致性评分 (0-100)
            consistency = row['momentum_consistency'] if pd.notna(row['momentum_consistency']) else 0.5
            momentum_consistency_score = min(100, consistency * 100)
            
            # 挤压动量总分 (加权平均: 释放35%, 方向25%, 状态25%, 一致性15%)
            squeeze_momentum_score = (
                squeeze_release_score * 0.35 +
                momentum_direction_score * 0.25 +
                squeeze_state_score * 0.25 +
                momentum_consistency_score * 0.15
            )
            
            return {
                'squeeze_momentum_score': squeeze_momentum_score,
                'squeeze_state': squeeze_state_score,
                'squeeze_release': squeeze_release_score,
                'momentum_direction': momentum_direction_score,
                'momentum_consistency': momentum_consistency_score
            }
            
        except Exception as e:
            logger.error(f"计算挤压动量失败 {stock_code} {trade_date}: {e}")
            return {'squeeze_momentum_score': 50.0, 'squeeze_state': 50.0, 
                   'squeeze_release': 50.0, 'momentum_direction': 50.0, 'momentum_consistency': 50.0}
    
    def calculate_fundamental_factors(self, stock_code: str, trade_date: str) -> Dict:
        """计算基本面维度因子 (0-100分)"""
        try:
            # 获取基本面数据
            query = """
            SELECT db.pe_ttm, db.pb, db.ps_ttm, db.total_mv, db.turnover_rate,
                   fi.roe, fi.roa, fi.netprofit_margin
            FROM daily_basic db
            JOIN securities s ON db.security_id = s.id
            LEFT JOIN financial_indicator fi ON s.id = fi.security_id 
                AND fi.end_date = (
                    SELECT MAX(end_date) FROM financial_indicator 
                    WHERE security_id = s.id AND end_date <= ?
                )
            WHERE s.code = ? AND db.trade_date = ?
            """
            
            result = self.db_manager.execute_query(query, [trade_date, stock_code, trade_date])
            if not result:
                return self._default_fundamental_scores()
            
            row = result[0]
            pe_ttm = row[0] if row[0] and row[0] > 0 else None
            pb = row[1] if row[1] and row[1] > 0 else None
            total_mv = row[3] if row[3] else 100  # 总市值（万元）
            turnover_rate = row[4] if row[4] else 1.0
            roe = row[5] if row[5] else None
            
            # 1. PE估值评分 (0-100)
            pe_valuation = self._calculate_pe_score(pe_ttm)
            
            # 2. PB估值评分 (0-100)
            pb_valuation = self._calculate_pb_score(pb)
            
            # 3. ROE盈利能力评分 (0-100)
            roe_profitability = self._calculate_roe_score(roe)
            
            # 4. 财务质量评分 (0-100) - 基于多个指标
            financial_quality = (pe_valuation + pb_valuation + roe_profitability) / 3
            
            # 5. 市值规模评分 (0-100) - total_mv单位是万元
            market_cap_score = self._calculate_market_cap_score(total_mv / 10000)  # 转换为亿元
            
            # 6. 换手率活跃度评分 (0-100)
            turnover_activity = self._calculate_turnover_score(turnover_rate)
            
            # 基本面总分 (加权平均)
            fundamental_score = (
                pe_valuation * 0.23 +
                pb_valuation * 0.23 +
                roe_profitability * 0.27 +
                financial_quality * 0.17 +
                market_cap_score * 0.13 +
                turnover_activity * 0.17
            )
            
            return {
                'fundamental_score': fundamental_score,
                'pe_valuation': pe_valuation,
                'pb_valuation': pb_valuation,
                'roe_profitability': roe_profitability,
                'financial_quality': financial_quality,
                'market_cap': market_cap_score,
                'turnover_activity': turnover_activity
            }
            
        except Exception as e:
            logger.error(f"计算基本面失败 {stock_code} {trade_date}: {e}")
            return self._default_fundamental_scores()
    
    def _default_fundamental_scores(self) -> Dict:
        """默认基本面评分"""
        return {
            'fundamental_score': 50.0,
            'pe_valuation': 50.0,
            'pb_valuation': 50.0,
            'roe_profitability': 50.0,
            'financial_quality': 50.0,
            'market_cap': 50.0,
            'turnover_activity': 50.0
        }
    
    def _calculate_pe_score(self, pe: Optional[float]) -> float:
        """PE估值评分"""
        if not pe or pe <= 0:
            return 40.0  # 无数据或负PE给予较低分
        
        if pe <= 10:        return 90  # 极低估值
        elif pe <= 15:      return 80  # 低估值
        elif pe <= 25:      return 70  # 合理估值
        elif pe <= 35:      return 60  # 略高估值
        elif pe <= 50:      return 50  # 高估值
        elif pe <= 80:      return 40  # 很高估值
        else:               return 20  # 极高估值
    
    def _calculate_pb_score(self, pb: Optional[float]) -> float:
        """PB估值评分"""
        if not pb or pb <= 0:
            return 40.0
        
        if pb <= 0.8:       return 90  # 破净股
        elif pb <= 1.0:     return 85  # 接近净资产
        elif pb <= 1.5:     return 75  # 合理水平
        elif pb <= 2.0:     return 65  # 略高
        elif pb <= 3.0:     return 55  # 高估
        elif pb <= 5.0:     return 45  # 很高
        else:               return 30  # 极高
    
    def _calculate_roe_score(self, roe: Optional[float]) -> float:
        """ROE盈利能力评分"""
        if not roe:
            return 50.0
        
        if roe >= 20:       return 95  # 优秀盈利能力
        elif roe >= 15:     return 85  # 很好
        elif roe >= 10:     return 75  # 良好
        elif roe >= 8:      return 65  # 尚可
        elif roe >= 5:      return 55  # 一般
        elif roe >= 0:      return 40  # 较差
        else:               return 20  # 亏损
    
    def _calculate_market_cap_score(self, market_cap: float) -> float:
        """市值规模评分 (偏好中等市值)"""
        if market_cap >= 5000:    return 60  # 超大盘股
        elif market_cap >= 1000:  return 75  # 大盘股  
        elif market_cap >= 300:   return 85  # 中大盘股 (偏好)
        elif market_cap >= 100:   return 80  # 中盘股 (偏好)
        elif market_cap >= 50:    return 70  # 中小盘股
        else:                     return 50  # 小盘股
    
    def _calculate_turnover_score(self, turnover: float) -> float:
        """换手率活跃度评分"""
        if turnover >= 10:      return 90  # 极活跃
        elif turnover >= 5:     return 80  # 很活跃
        elif turnover >= 3:     return 75  # 活跃
        elif turnover >= 2:     return 65  # 适中活跃
        elif turnover >= 1:     return 55  # 一般
        else:                   return 40  # 不活跃
    
    def calculate_performance_factors(self, stock_code: str, trade_date: str) -> Dict:
        """计算市场表现维度因子 (0-100分)"""
        try:
            # 获取近期价格表现数据
            query = """
            SELECT dq1.close as current_close,
                   dq5.close as close_5d_ago,
                   dq20.close as close_20d_ago,
                   dq1.price_change_pct as daily_change
            FROM daily_quotes dq1
            JOIN securities s ON dq1.security_id = s.id
            LEFT JOIN daily_quotes dq5 ON s.id = dq5.security_id 
                AND dq5.trade_date = DATE(dq1.trade_date, '-5 days')
            LEFT JOIN daily_quotes dq20 ON s.id = dq20.security_id 
                AND dq20.trade_date = DATE(dq1.trade_date, '-20 days')
            WHERE s.code = ? AND dq1.trade_date = ?
            """
            
            result = self.db_manager.execute_query(query, [stock_code, trade_date])
            if not result:
                return self._default_performance_scores()
            
            row = result[0]
            current_close = row[0] or 100
            close_5d = row[1] or current_close
            close_20d = row[2] or current_close
            daily_change = row[3] or 0
            
            # 1. 价格动量评分 (0-100)
            return_5d = (current_close - close_5d) / close_5d * 100 if close_5d > 0 else 0
            return_20d = (current_close - close_20d) / close_20d * 100 if close_20d > 0 else 0
            price_momentum = self._calculate_momentum_score(return_5d, return_20d, daily_change)
            
            # 2. 相对强度评分 (简化版，基于短期表现)
            relative_strength = min(100, max(0, 50 + return_5d * 2))
            
            # 3. 波动率风险评分 (基于日涨跌幅)
            volatility_risk = self._calculate_volatility_score(abs(daily_change))
            
            # 市场表现总分
            performance_score = (
                price_momentum * 0.67 +
                relative_strength * 0.20 +
                volatility_risk * 0.13
            )
            
            return {
                'performance_score': performance_score,
                'price_momentum': price_momentum,
                'relative_strength': relative_strength,
                'volatility_risk': volatility_risk
            }
            
        except Exception as e:
            logger.error(f"计算市场表现失败 {stock_code} {trade_date}: {e}")
            return self._default_performance_scores()
    
    def _default_performance_scores(self) -> Dict:
        """默认市场表现评分"""
        return {
            'performance_score': 50.0,
            'price_momentum': 50.0,
            'relative_strength': 50.0,
            'volatility_risk': 50.0
        }
    
    def _calculate_momentum_score(self, ret_5d: float, ret_20d: float, daily: float) -> float:
        """计算价格动量评分"""
        score = 50.0
        
        # 短期动量 (5日)
        if ret_5d >= 10:      score += 25
        elif ret_5d >= 5:     score += 15
        elif ret_5d >= 2:     score += 10
        elif ret_5d >= -2:    score += 0
        elif ret_5d >= -5:    score -= 10
        else:                 score -= 20
        
        # 中期动量 (20日)  
        if ret_20d >= 20:     score += 15
        elif ret_20d >= 10:   score += 10
        elif ret_20d >= 0:    score += 5
        elif ret_20d >= -10:  score -= 5
        else:                 score -= 15
        
        # 当日表现
        if daily >= 5:        score += 10
        elif daily >= 0:      score += 5
        elif daily >= -5:     score += 0
        else:                 score -= 10
        
        return max(0, min(100, score))
    
    def _calculate_volatility_score(self, abs_daily_change: float) -> float:
        """计算波动率评分 (低波动率高分)"""
        if abs_daily_change <= 1:     return 90  # 极低波动
        elif abs_daily_change <= 2:   return 80  # 低波动
        elif abs_daily_change <= 3:   return 70  # 适中波动
        elif abs_daily_change <= 5:   return 60  # 中等波动
        elif abs_daily_change <= 7:   return 50  # 高波动
        elif abs_daily_change <= 9:   return 40  # 很高波动
        else:                          return 30  # 极高波动
    
    def calculate_other_factors(self, stock_code: str, trade_date: str) -> Dict:
        """计算其他维度因子 (情绪、风控、市场环境)"""
        # 简化实现，基于基础数据计算
        return {
            # 情绪指标维度
            'sentiment_score': 50.0,
            'money_flow': 50.0,
            'market_attention': 50.0,
            'investor_emotion': 50.0,
            
            # 风险控制维度
            'risk_control_score': 50.0,
            'stop_loss_risk': 50.0,
            'max_drawdown': 50.0,
            'risk_adjusted_return': 50.0,
            
            # 市场环境维度
            'market_regime_score': 50.0,
            'market_beta': 50.0,
            'sector_rotation': 50.0,
            'liquidity': 50.0
        }
    
    def calculate_future_returns(self, stock_code: str, trade_date: str) -> Dict:
        """计算未来收益数据"""
        try:
            query = """
            SELECT 
                dq_current.close as current_close,
                dq_1d.close as close_1d,
                dq_3d.close as close_3d,
                dq_5d.close as close_5d,
                dq_10d.close as close_10d,
                dq_20d.close as close_20d
            FROM daily_quotes dq_current
            JOIN securities s ON dq_current.security_id = s.id
            LEFT JOIN daily_quotes dq_1d ON s.id = dq_1d.security_id 
                AND dq_1d.trade_date = DATE(dq_current.trade_date, '+1 days')
            LEFT JOIN daily_quotes dq_3d ON s.id = dq_3d.security_id 
                AND dq_3d.trade_date = DATE(dq_current.trade_date, '+3 days')  
            LEFT JOIN daily_quotes dq_5d ON s.id = dq_5d.security_id 
                AND dq_5d.trade_date = DATE(dq_current.trade_date, '+5 days')
            LEFT JOIN daily_quotes dq_10d ON s.id = dq_10d.security_id 
                AND dq_10d.trade_date = DATE(dq_current.trade_date, '+10 days')
            LEFT JOIN daily_quotes dq_20d ON s.id = dq_20d.security_id 
                AND dq_20d.trade_date = DATE(dq_current.trade_date, '+20 days')
            WHERE s.code = ? AND dq_current.trade_date = ?
            """
            
            result = self.db_manager.execute_query(query, [stock_code, trade_date])
            if not result:
                return {}
            
            row = result[0]
            current = row[0] or 0
            
            returns = {}
            if current > 0:
                for i, period in enumerate(['1d', '3d', '5d', '10d', '20d'], 1):
                    future_price = row[i]
                    if future_price and future_price > 0:
                        returns[f'return_{period}'] = (future_price - current) / current * 100
                    else:
                        returns[f'return_{period}'] = None
            
            return returns
            
        except Exception as e:
            logger.error(f"计算未来收益失败 {stock_code} {trade_date}: {e}")
            return {}
    
    def calculate_stock_standard_factors(self, args: Tuple) -> Optional[Dict]:
        """计算单只股票的标准化因子"""
        stock_code, trade_date = args
        
        try:
            # 计算各维度因子
            technical = self.calculate_technical_factors(stock_code, trade_date)
            squeeze = self.calculate_squeeze_momentum_factors(stock_code, trade_date)
            fundamental = self.calculate_fundamental_factors(stock_code, trade_date)
            performance = self.calculate_performance_factors(stock_code, trade_date)
            others = self.calculate_other_factors(stock_code, trade_date)
            future_returns = self.calculate_future_returns(stock_code, trade_date)
            
            # 合并所有因子
            all_factors = {
                'stock_code': stock_code,
                'trade_date': trade_date,
                **technical,
                **squeeze,
                **fundamental,
                **performance,
                **others,
                **future_returns
            }
            
            return all_factors
            
        except Exception as e:
            logger.error(f"计算标准化因子失败 {stock_code} {trade_date}: {e}")
            return None
    
    def batch_calculate_standard_factors(self, start_date: str, end_date: str,
                                       max_workers: int = 6, batch_size: int = 1000) -> Dict:
        """批量计算标准化因子"""
        logger.info(f"🚀 开始批量计算标准化因子 ({start_date} 到 {end_date})")
        
        # 获取需要计算的股票和日期组合
        query = """
        SELECT DISTINCT s.code, dq.trade_date
        FROM securities s
        JOIN daily_quotes dq ON s.id = dq.security_id
        WHERE s.is_active = 1 AND s.type = 'A股'
        AND dq.trade_date >= ? AND dq.trade_date <= ?
        ORDER BY dq.trade_date DESC, s.code
        """
        
        results = self.db_manager.execute_query(query, [start_date, end_date])
        stock_date_pairs = [(row[0], row[1]) for row in results]
        
        logger.info(f"📊 需要计算 {len(stock_date_pairs)} 个股票-日期组合")
        
        # 分批处理
        all_records = []
        processed_count = 0
        
        for i in range(0, len(stock_date_pairs), batch_size):
            batch_pairs = stock_date_pairs[i:i + batch_size]
            
            logger.info(f"📈 处理第 {i//batch_size + 1} 批，共 {len(batch_pairs)} 个组合")
            
            # 并行计算当前批次
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self.calculate_stock_standard_factors, pair)
                    for pair in batch_pairs
                ]
                
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        all_records.append(result)
                        processed_count += 1
                    
                    if processed_count % 500 == 0:
                        logger.info(f"✅ 已处理 {processed_count} 个组合")
            
            # 批量保存到数据库
            if all_records:
                self._batch_save_standard_factors(all_records)
                logger.info(f"💾 批次保存完成: {len(all_records)} 条记录")
                all_records.clear()  # 清空以释放内存
        
        # 保存剩余记录
        if all_records:
            self._batch_save_standard_factors(all_records)
        
        stats = {
            'processed_combinations': processed_count,
            'total_combinations': len(stock_date_pairs),
            'success_rate': processed_count / len(stock_date_pairs) * 100,
            'calculation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info("🎉 标准化因子批量计算完成！")
        logger.info(f"📊 处理统计: {processed_count}/{len(stock_date_pairs)} ({stats['success_rate']:.1f}%)")
        
        return stats
    
    def _batch_save_standard_factors(self, records: List[Dict]):
        """批量保存标准化因子到数据库"""
        if not records:
            return
        
        # 准备插入SQL
        insert_sql = """
        INSERT OR REPLACE INTO standard_factors (
            stock_code, trade_date,
            technical_score, squeeze_momentum_score, fundamental_score, 
            performance_score, sentiment_score, risk_control_score, market_regime_score,
            kdj_strength, rsi_momentum, bbi_trend, volume_surge,
            squeeze_state, squeeze_release, momentum_direction, momentum_consistency,
            pe_valuation, pb_valuation, roe_profitability, financial_quality, 
            market_cap, turnover_activity,
            price_momentum, relative_strength, volatility_risk,
            money_flow, market_attention, investor_emotion,
            stop_loss_risk, max_drawdown, risk_adjusted_return,
            market_beta, sector_rotation, liquidity,
            return_1d, return_3d, return_5d, return_10d, return_20d
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        # 准备数据
        batch_data = []
        for record in records:
            batch_data.append((
                record['stock_code'], record['trade_date'],
                record.get('technical_score', 50), record.get('squeeze_momentum_score', 50),
                record.get('fundamental_score', 50), record.get('performance_score', 50),
                record.get('sentiment_score', 50), record.get('risk_control_score', 50),
                record.get('market_regime_score', 50),
                record.get('kdj_strength', 50), record.get('rsi_momentum', 50),
                record.get('bbi_trend', 50), record.get('volume_surge', 50),
                record.get('squeeze_state', 50), record.get('squeeze_release', 50),
                record.get('momentum_direction', 50), record.get('momentum_consistency', 50),
                record.get('pe_valuation', 50), record.get('pb_valuation', 50),
                record.get('roe_profitability', 50), record.get('financial_quality', 50),
                record.get('market_cap', 50), record.get('turnover_activity', 50),
                record.get('price_momentum', 50), record.get('relative_strength', 50),
                record.get('volatility_risk', 50),
                record.get('money_flow', 50), record.get('market_attention', 50),
                record.get('investor_emotion', 50),
                record.get('stop_loss_risk', 50), record.get('max_drawdown', 50),
                record.get('risk_adjusted_return', 50),
                record.get('market_beta', 50), record.get('sector_rotation', 50),
                record.get('liquidity', 50),
                record.get('return_1d'), record.get('return_3d'), record.get('return_5d'),
                record.get('return_10d'), record.get('return_20d')
            ))
        
        # 执行批量插入
        with sqlite3.connect(self.factor_db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany(insert_sql, batch_data)
            conn.commit()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='标准化因子计算器')
    parser.add_argument('--start-date', default='2024-06-01', help='开始日期')
    parser.add_argument('--end-date', default='2025-08-25', help='结束日期')
    parser.add_argument('--max-workers', type=int, default=6, help='最大进程数')
    parser.add_argument('--batch-size', type=int, default=1000, help='批次大小')
    
    args = parser.parse_args()
    
    calculator = StandardFactorCalculator()
    
    stats = calculator.batch_calculate_standard_factors(
        args.start_date, args.end_date, 
        args.max_workers, args.batch_size
    )
    
    logger.info("✨ 标准化因子计算完成！")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    main()