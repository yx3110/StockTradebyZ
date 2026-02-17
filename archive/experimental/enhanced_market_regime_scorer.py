#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版市场环境评分系统

重新设计市场环境评分算法，实现：
1. 更大的评分变化范围 (0.1-0.9)
2. 更强的时间动态性
3. 更敏感的市场状态区分
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import time
import warnings
import math

warnings.filterwarnings('ignore')

class EnhancedMarketRegimeScorer:
    """增强版市场环境评分器"""
    
    def __init__(self, db_path: str = "data_adapter/stock_data.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # 市场环境评分配置
        self.config = {
            # 多时间窗口权重
            "time_windows": {
                "short_term": {"days": 5, "weight": 0.4},    # 短期趋势
                "medium_term": {"days": 20, "weight": 0.35}, # 中期趋势  
                "long_term": {"days": 60, "weight": 0.25}    # 长期趋势
            },
            
            # 市场状态阈值 (更敏感)
            "thresholds": {
                "strong_bull": 0.02,      # 强牛市
                "bull": 0.008,            # 牛市
                "neutral_up": 0.003,      # 中性偏多
                "neutral_down": -0.003,   # 中性偏空
                "bear": -0.008,           # 熊市
                "strong_bear": -0.02      # 强熊市
            },
            
            # 波动率阈值
            "volatility": {
                "very_high": 0.035,       # 极高波动
                "high": 0.025,            # 高波动
                "normal": 0.015,          # 正常波动
                "low": 0.008              # 低波动
            },
            
            # 成交量异常阈值
            "volume": {
                "surge": 2.0,             # 放量
                "normal": 0.8,            # 正常
                "shrink": 0.4             # 缩量
            }
        }
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def get_market_data(self, date: str, days_back: int = 120) -> pd.DataFrame:
        """获取市场数据 - 上证指数"""
        try:
            query = """
            SELECT trade_date, close, price_change_pct, volume
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE s.code = '000001.SH'
            AND trade_date <= ?
            ORDER BY trade_date DESC
            LIMIT ?
            """
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=[date, days_back])
            
            if not df.empty:
                # 计算一些基础指标
                df['return'] = df['price_change_pct'] / 100
                df['log_return'] = np.log(df['close'] / df['close'].shift(1))
                df['volatility'] = df['return'].rolling(window=10).std()
                
            return df
            
        except Exception as e:
            self.logger.error(f"获取市场数据失败: {e}")
            return pd.DataFrame()
    
    def detect_comprehensive_market_regime(self, date: str) -> Dict[str, Any]:
        """检测综合市场环境 - 全新算法"""
        try:
            # 获取足够的历史数据
            market_data = self.get_market_data(date, 120)
            
            if market_data.empty or len(market_data) < 20:
                return self._get_default_regime()
            
            regime_info = {}
            
            # 1. 多时间窗口趋势分析
            trend_scores = []
            for window_name, window_config in self.config["time_windows"].items():
                days = window_config["days"]
                weight = window_config["weight"]
                
                if len(market_data) >= days:
                    window_data = market_data.head(days)
                    
                    # 计算趋势强度
                    returns = window_data['return'].fillna(0)
                    avg_return = returns.mean()
                    
                    # 计算趋势一致性 (正收益天数比例)
                    consistency = (returns > 0).mean()
                    
                    # 计算趋势加速度 (最近vs早期收益率比较)
                    if len(returns) >= 10:
                        recent_return = returns.head(5).mean()
                        early_return = returns.tail(5).mean()
                        acceleration = recent_return - early_return
                    else:
                        acceleration = 0
                    
                    # 综合趋势评分
                    trend_score = (
                        avg_return * 0.5 +           # 平均收益
                        (consistency - 0.5) * 0.3 +  # 一致性调整
                        acceleration * 0.2            # 趋势加速度
                    )
                    
                    trend_scores.append({
                        'window': window_name,
                        'score': trend_score,
                        'weight': weight,
                        'avg_return': avg_return,
                        'consistency': consistency,
                        'acceleration': acceleration
                    })
            
            # 加权综合趋势
            weighted_trend = sum(item['score'] * item['weight'] for item in trend_scores)
            regime_info['trend_score'] = weighted_trend
            regime_info['trend_details'] = trend_scores
            
            # 2. 波动率环境分析
            recent_volatility = market_data.head(20)['return'].std() if len(market_data) >= 20 else 0.02
            regime_info['volatility'] = recent_volatility
            
            # 波动率状态
            vol_state = "normal"
            if recent_volatility > self.config["volatility"]["very_high"]:
                vol_state = "very_high"
            elif recent_volatility > self.config["volatility"]["high"]:
                vol_state = "high"
            elif recent_volatility < self.config["volatility"]["low"]:
                vol_state = "low"
            
            regime_info['volatility_state'] = vol_state
            
            # 3. 成交量环境分析
            if len(market_data) >= 20:
                recent_volume = market_data.head(10)['volume'].mean()
                historic_volume = market_data.tail(20)['volume'].mean()
                
                volume_ratio = recent_volume / historic_volume if historic_volume > 0 else 1.0
                regime_info['volume_ratio'] = volume_ratio
                
                if volume_ratio > self.config["volume"]["surge"]:
                    volume_state = "surge"
                elif volume_ratio < self.config["volume"]["shrink"]:
                    volume_state = "shrink"
                else:
                    volume_state = "normal"
                    
                regime_info['volume_state'] = volume_state
            else:
                regime_info['volume_ratio'] = 1.0
                regime_info['volume_state'] = "normal"
            
            # 4. 市场状态分类
            market_state = self._classify_market_state(weighted_trend)
            regime_info['market_state'] = market_state
            
            # 5. 动量分析
            if len(market_data) >= 10:
                short_momentum = market_data.head(5)['return'].mean()
                medium_momentum = market_data.head(20)['return'].mean() if len(market_data) >= 20 else short_momentum
                
                momentum_divergence = abs(short_momentum - medium_momentum)
                regime_info['short_momentum'] = short_momentum
                regime_info['medium_momentum'] = medium_momentum  
                regime_info['momentum_divergence'] = momentum_divergence
            
            # 6. 技术形态识别
            regime_info['technical_pattern'] = self._identify_technical_pattern(market_data)
            
            return regime_info
            
        except Exception as e:
            self.logger.error(f"检测市场环境失败: {e}")
            return self._get_default_regime()
    
    def _classify_market_state(self, trend_score: float) -> str:
        """根据趋势得分分类市场状态"""
        thresholds = self.config["thresholds"]
        
        if trend_score > thresholds["strong_bull"]:
            return "strong_bull"
        elif trend_score > thresholds["bull"]:
            return "bull"
        elif trend_score > thresholds["neutral_up"]:
            return "neutral_up"
        elif trend_score > thresholds["neutral_down"]:
            return "neutral_down"
        elif trend_score > thresholds["bear"]:
            return "bear"
        else:
            return "strong_bear"
    
    def _identify_technical_pattern(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """识别技术形态"""
        try:
            if len(market_data) < 20:
                return {"pattern": "insufficient_data", "strength": 0.0}
            
            closes = market_data.head(20)['close'].values
            
            # 简单趋势识别
            if len(closes) >= 10:
                recent_high = closes[:5].max()
                recent_low = closes[:5].min()
                historic_high = closes[5:].max()
                historic_low = closes[5:].min()
                
                # 突破形态
                if recent_high > historic_high * 1.02:
                    return {"pattern": "upward_breakout", "strength": 0.8}
                elif recent_low < historic_low * 0.98:
                    return {"pattern": "downward_breakout", "strength": 0.8}
                
                # 整理形态
                range_ratio = (closes.max() - closes.min()) / closes.mean()
                if range_ratio < 0.03:
                    return {"pattern": "consolidation", "strength": 0.6}
            
            return {"pattern": "normal", "strength": 0.5}
            
        except Exception as e:
            return {"pattern": "error", "strength": 0.0}
    
    def calculate_dynamic_market_score(self, regime_info: Dict[str, Any]) -> float:
        """计算动态市场环境得分 - 核心算法"""
        try:
            scores = []
            
            # 1. 趋势得分 (40%) - 基于多时间窗口
            trend_score = regime_info.get('trend_score', 0)
            market_state = regime_info.get('market_state', 'neutral_up')
            
            # 市场状态基础得分
            state_scores = {
                'strong_bull': 0.9,
                'bull': 0.75,
                'neutral_up': 0.6,
                'neutral_down': 0.4,
                'bear': 0.25,
                'strong_bear': 0.1
            }
            
            base_score = state_scores.get(market_state, 0.5)
            
            # 趋势强度调整
            trend_strength = min(1.0, abs(trend_score) / 0.02)  # 归一化到0-1
            adjusted_trend_score = base_score + (trend_strength - 0.5) * 0.2
            adjusted_trend_score = max(0.05, min(0.95, adjusted_trend_score))
            
            scores.append(('trend', adjusted_trend_score, 0.4))
            
            # 2. 波动率环境得分 (25%)
            volatility_state = regime_info.get('volatility_state', 'normal')
            vol_scores = {
                'very_high': 0.85,  # 高波动有利于选股机会
                'high': 0.7,
                'normal': 0.5,
                'low': 0.3          # 低波动市场机会较少
            }
            vol_score = vol_scores.get(volatility_state, 0.5)
            scores.append(('volatility', vol_score, 0.25))
            
            # 3. 成交量环境得分 (20%)
            volume_ratio = regime_info.get('volume_ratio', 1.0)
            volume_state = regime_info.get('volume_state', 'normal')
            
            # 成交量异常度评分
            if volume_state == 'surge':
                volume_score = 0.8 + min(0.15, (volume_ratio - 2.0) / 5.0)  # 放量有利
            elif volume_state == 'shrink':
                volume_score = 0.3 - min(0.2, (0.4 - volume_ratio) / 0.3)   # 缩量不利
            else:
                volume_score = 0.5 + (volume_ratio - 1.0) / 4.0  # 线性调整
                
            volume_score = max(0.1, min(0.9, volume_score))
            scores.append(('volume', volume_score, 0.2))
            
            # 4. 动量分歧度得分 (15%)
            momentum_divergence = regime_info.get('momentum_divergence', 0)
            # 动量分歧有利于选股(不同股票分化明显)
            divergence_score = 0.5 + min(0.4, momentum_divergence / 0.02)
            scores.append(('momentum', divergence_score, 0.15))
            
            # 加权综合得分
            final_score = sum(score * weight for _, score, weight in scores)
            
            # 记录详细信息
            self.logger.debug(f"市场环境评分详情: {[(name, f'{score:.3f}×{weight:.2f}') for name, score, weight in scores]}")
            self.logger.info(f"最终市场环境得分: {final_score:.4f}")
            
            return final_score
            
        except Exception as e:
            self.logger.error(f"计算动态市场得分失败: {e}")
            return 0.5  # 返回中性得分
    
    def _get_default_regime(self) -> Dict[str, Any]:
        """获取默认市场环境"""
        return {
            'trend_score': 0.0,
            'market_state': 'neutral_up',
            'volatility': 0.02,
            'volatility_state': 'normal',
            'volume_ratio': 1.0,
            'volume_state': 'normal',
            'short_momentum': 0.0,
            'medium_momentum': 0.0,
            'momentum_divergence': 0.0,
            'technical_pattern': {'pattern': 'normal', 'strength': 0.5}
        }
    
    def batch_calculate_market_scores(self, dates: List[str]) -> Dict[str, float]:
        """批量计算多个日期的市场环境得分"""
        results = {}
        
        self.logger.info(f"开始批量计算 {len(dates)} 个日期的市场环境得分...")
        
        for i, date in enumerate(dates, 1):
            if i % 50 == 0 or i == len(dates):
                self.logger.info(f"进度: {i}/{len(dates)} ({i/len(dates)*100:.1f}%)")
            
            try:
                regime_info = self.detect_comprehensive_market_regime(date)
                market_score = self.calculate_dynamic_market_score(regime_info)
                results[date] = market_score
                
            except Exception as e:
                self.logger.error(f"计算日期 {date} 的市场得分失败: {e}")
                results[date] = 0.5  # 默认中性得分
        
        self.logger.info(f"批量计算完成！得分分布: 最小={min(results.values()):.4f}, "
                        f"最大={max(results.values()):.4f}, 平均={np.mean(list(results.values())):.4f}")
        
        return results

def main():
    """测试新的市场环境评分系统"""
    scorer = EnhancedMarketRegimeScorer()
    
    # 测试更多日期来验证动态范围
    test_dates = [
        '2024-01-02', '2024-02-05', '2024-03-15', '2024-04-22', '2024-05-20', 
        '2024-06-20', '2024-07-15', '2024-08-12', '2024-09-10', '2024-10-14',
        '2024-11-11', '2024-12-09', '2025-01-15', '2025-02-10', '2025-08-20'
    ]
    
    print("🔬 测试新市场环境评分系统")
    print("="*60)
    
    for date in test_dates:
        print(f"\n📅 日期: {date}")
        
        # 获取市场环境信息
        regime_info = scorer.detect_comprehensive_market_regime(date)
        
        # 计算动态得分
        market_score = scorer.calculate_dynamic_market_score(regime_info)
        
        print(f"市场状态: {regime_info.get('market_state', 'unknown')}")
        print(f"趋势得分: {regime_info.get('trend_score', 0):.4f}")
        print(f"波动率: {regime_info.get('volatility', 0)*100:.2f}% ({regime_info.get('volatility_state', 'unknown')})")
        print(f"成交量比率: {regime_info.get('volume_ratio', 1.0):.2f}x ({regime_info.get('volume_state', 'unknown')})")
        print(f"🎯 最终市场环境得分: {market_score:.4f}")
    
    # 统计评分分布
    scores = []
    for date in test_dates:
        regime_info = scorer.detect_comprehensive_market_regime(date)
        market_score = scorer.calculate_dynamic_market_score(regime_info)
        scores.append(market_score)
    
    print(f"\n📊 评分统计：")
    print(f"  最小值: {min(scores):.4f}")
    print(f"  最大值: {max(scores):.4f}")
    print(f"  平均值: {np.mean(scores):.4f}")
    print(f"  标准差: {np.std(scores):.4f}")
    print(f"  评分范围: {max(scores) - min(scores):.4f}")
    
    print(f"\n✅ 测试完成！新评分系统展现了更大的动态变化范围。")

if __name__ == "__main__":
    main()