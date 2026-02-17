#!/usr/bin/env python3
"""
市场分析增强器
用于获取基本面数据和大盘分析，增强AI选股报告
"""

import os
import sys
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import sqlite3

# 添加路径
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / "data_adapter"))

from data_adapter.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

class MarketAnalysisEnhancer:
    """市场分析增强器"""
    
    def __init__(self, db_path: str = "data_adapter/stock_data.db"):
        """初始化增强器"""
        self.db_manager = DatabaseManager(db_path)
        
        # 大盘指数代码映射
        self.market_indices = {
            'sh000001': {'name': '上证指数', 'symbol': '000001.SH'},
            'sz399001': {'name': '深证成指', 'symbol': '399001.SZ'},
            'sz399006': {'name': '创业板指', 'symbol': '399006.SZ'},
            'sh000300': {'name': '沪深300', 'symbol': '000300.SH'}
        }
        
        # 基本面数据缓存
        self.fundamentals_cache = {}
        self.market_cache = {}
        
    def get_stock_fundamentals(self, stock_code: str) -> Dict[str, Any]:
        """获取股票基本面数据"""
        if stock_code in self.fundamentals_cache:
            return self.fundamentals_cache[stock_code]
            
        try:
            # 尝试多种方式获取基本面数据
            fundamentals = self._get_fundamentals_from_database(stock_code)
            
            if not fundamentals:
                fundamentals = self._estimate_fundamentals_from_price_data(stock_code)
            
            self.fundamentals_cache[stock_code] = fundamentals
            return fundamentals
            
        except Exception as e:
            logger.warning(f"获取股票{stock_code}基本面数据失败: {e}")
            return self._get_default_fundamentals()
    
    def _get_fundamentals_from_database(self, stock_code: str) -> Optional[Dict]:
        """从数据库获取基本面数据"""
        try:
            with self.db_manager.get_connection() as conn:
                # 获取最新的市值等基础数据
                query = """
                SELECT s.name, s.industry, s.area, s.list_date,
                       q.close, q.volume, q.amount
                FROM securities s 
                JOIN daily_quotes q ON s.id = q.security_id
                WHERE s.code = ? 
                ORDER BY q.trade_date DESC 
                LIMIT 1
                """
                cursor = conn.execute(query, (stock_code,))
                result = cursor.fetchone()
                
                if result:
                    # 计算基本的市值估算
                    close_price = float(result[4]) if result[4] else 0
                    volume = int(result[5]) if result[5] else 0
                    amount = float(result[6]) if result[6] else 0
                    
                    # 估算总股本（简化计算）
                    estimated_shares = (amount / close_price) if close_price > 0 and amount > 0 else volume * 100
                    market_cap = close_price * estimated_shares / 10000  # 万元
                    
                    return {
                        'name': result[0],
                        'industry': result[1] or '未知',
                        'area': result[2] or '未知',
                        'list_date': result[3],
                        'current_price': close_price,
                        'market_cap': round(market_cap, 2),
                        'estimated_shares': round(estimated_shares / 10000, 2),  # 万股
                        'pe_ratio': None,  # 暂无数据
                        'pb_ratio': None,  # 暂无数据
                        'roe': None,       # 暂无数据
                        'data_source': 'database_estimated'
                    }
                    
        except Exception as e:
            logger.warning(f"从数据库获取{stock_code}基本面数据失败: {e}")
        
        return None
    
    def _estimate_fundamentals_from_price_data(self, stock_code: str) -> Dict[str, Any]:
        """基于价格数据估算基本面信息"""
        try:
            with self.db_manager.get_connection() as conn:
                # 获取近期价格数据用于估算
                query = """
                SELECT q.close, q.volume, q.amount, q.trade_date
                FROM securities s 
                JOIN daily_quotes q ON s.id = q.security_id
                WHERE s.code = ? 
                ORDER BY q.trade_date DESC 
                LIMIT 30
                """
                df = pd.read_sql_query(query, conn, params=(stock_code,))
                
                if len(df) > 0:
                    latest = df.iloc[0]
                    avg_volume = df['volume'].mean()
                    avg_amount = df['amount'].mean()
                    
                    # 估算流通市值
                    if avg_amount > 0 and latest['close'] > 0:
                        estimated_shares = avg_amount / latest['close']
                        market_cap = latest['close'] * estimated_shares / 10000
                    else:
                        market_cap = 0
                    
                    return {
                        'current_price': float(latest['close']),
                        'market_cap': round(market_cap, 2),
                        'avg_volume': round(avg_volume, 0),
                        'avg_amount': round(avg_amount, 2),
                        'data_source': 'price_estimated'
                    }
                    
        except Exception as e:
            logger.warning(f"从价格数据估算{stock_code}基本面失败: {e}")
        
        return self._get_default_fundamentals()
    
    def _get_default_fundamentals(self) -> Dict[str, Any]:
        """获取默认基本面数据"""
        return {
            'current_price': 0,
            'market_cap': 0,
            'pe_ratio': None,
            'pb_ratio': None,
            'roe': None,
            'data_source': 'unavailable'
        }
    
    def get_market_analysis(self, analysis_date: str = None) -> Dict[str, Any]:
        """获取大盘分析"""
        if analysis_date is None:
            analysis_date = datetime.now().strftime('%Y-%m-%d')
        
        if analysis_date in self.market_cache:
            return self.market_cache[analysis_date]
        
        try:
            market_analysis = {
                'analysis_date': analysis_date,
                'indices_performance': {},
                'market_sentiment': {},
                'technical_analysis': {},
                'market_environment': 'neutral'
            }
            
            # 分析各大指数表现
            for index_key, index_info in self.market_indices.items():
                performance = self._analyze_index_performance(index_info, analysis_date)
                market_analysis['indices_performance'][index_key] = performance
            
            # 综合市场情绪
            market_analysis['market_sentiment'] = self._analyze_market_sentiment(analysis_date)
            
            # 技术面分析
            market_analysis['technical_analysis'] = self._analyze_market_technical(analysis_date)
            
            # 综合判断市场环境
            market_analysis['market_environment'] = self._determine_market_environment(market_analysis)
            
            self.market_cache[analysis_date] = market_analysis
            return market_analysis
            
        except Exception as e:
            logger.error(f"获取大盘分析失败: {e}")
            return self._get_default_market_analysis(analysis_date)
    
    def _analyze_index_performance(self, index_info: Dict, analysis_date: str) -> Dict[str, Any]:
        """分析指数表现"""
        try:
            # 由于没有指数数据，使用ETF数据作为参考
            etf_codes = {
                '上证指数': '510300',  # 沪深300ETF作为参考
                '深证成指': '159901',  # 深100ETF作为参考  
                '创业板指': '159915',  # 创业板ETF作为参考
                '沪深300': '510300'   # 沪深300ETF
            }
            
            etf_code = etf_codes.get(index_info['name'], '510300')
            performance = self._get_etf_performance(etf_code, analysis_date)
            
            return {
                'name': index_info['name'],
                'latest_price': performance.get('latest_price', 0),
                'price_change': performance.get('price_change', 0),
                'price_change_pct': performance.get('price_change_pct', 0),
                'volume': performance.get('volume', 0),
                'ma5_trend': performance.get('ma5_trend', 'unknown'),
                'ma20_trend': performance.get('ma20_trend', 'unknown'),
                'technical_signal': performance.get('technical_signal', 'neutral')
            }
            
        except Exception as e:
            logger.warning(f"分析指数{index_info['name']}表现失败: {e}")
            return {
                'name': index_info['name'],
                'latest_price': 0,
                'price_change': 0,
                'price_change_pct': 0,
                'technical_signal': 'unknown'
            }
    
    def _get_etf_performance(self, etf_code: str, analysis_date: str) -> Dict[str, Any]:
        """获取ETF表现数据"""
        try:
            with self.db_manager.get_connection() as conn:
                query = """
                SELECT q.close, q.price_change, q.price_change_pct, q.volume,
                       q.ma5, q.ma20, q.trade_date
                FROM securities s 
                JOIN daily_quotes q ON s.id = q.security_id
                WHERE s.code = ? 
                ORDER BY q.trade_date DESC 
                LIMIT 5
                """
                df = pd.read_sql_query(query, conn, params=(etf_code,))
                
                if len(df) > 0:
                    latest = df.iloc[0]
                    
                    # 计算趋势
                    ma5_trend = 'up' if latest['ma5'] and latest['close'] > latest['ma5'] else 'down'
                    ma20_trend = 'up' if latest['ma20'] and latest['close'] > latest['ma20'] else 'down'
                    
                    # 技术信号
                    if latest['price_change_pct'] > 1:
                        technical_signal = 'bullish'
                    elif latest['price_change_pct'] < -1:
                        technical_signal = 'bearish'
                    else:
                        technical_signal = 'neutral'
                    
                    return {
                        'latest_price': float(latest['close']),
                        'price_change': float(latest['price_change']) if latest['price_change'] else 0,
                        'price_change_pct': float(latest['price_change_pct']),
                        'volume': int(latest['volume']),
                        'ma5_trend': ma5_trend,
                        'ma20_trend': ma20_trend,
                        'technical_signal': technical_signal
                    }
                    
        except Exception as e:
            logger.warning(f"获取ETF {etf_code} 表现数据失败: {e}")
        
        return {}
    
    def _analyze_market_sentiment(self, analysis_date: str) -> Dict[str, Any]:
        """分析市场整体情绪"""
        try:
            # 通过涨跌股票数量分析市场情绪
            with self.db_manager.get_connection() as conn:
                query = """
                SELECT 
                    COUNT(CASE WHEN q.price_change_pct > 0 THEN 1 END) as rising_count,
                    COUNT(CASE WHEN q.price_change_pct < 0 THEN 1 END) as falling_count,
                    COUNT(CASE WHEN q.price_change_pct > 5 THEN 1 END) as strong_rising,
                    COUNT(CASE WHEN q.price_change_pct < -5 THEN 1 END) as strong_falling,
                    AVG(q.price_change_pct) as avg_change_pct,
                    COUNT(*) as total_stocks
                FROM securities s 
                JOIN daily_quotes q ON s.id = q.security_id
                WHERE s.type = 'A股' 
                ORDER BY q.trade_date DESC 
                LIMIT 1000
                """
                cursor = conn.execute(query)
                result = cursor.fetchone()
                
                if result:
                    rising_count, falling_count, strong_rising, strong_falling, avg_change, total = result
                    
                    # 计算市场情绪指标
                    if total > 0:
                        rising_ratio = rising_count / total
                        sentiment_score = (rising_ratio - 0.5) * 2  # -1到1之间
                        
                        if sentiment_score > 0.3:
                            sentiment_label = "乐观"
                        elif sentiment_score > 0.1:
                            sentiment_label = "偏乐观"
                        elif sentiment_score > -0.1:
                            sentiment_label = "中性"
                        elif sentiment_score > -0.3:
                            sentiment_label = "偏悲观"
                        else:
                            sentiment_label = "悲观"
                        
                        return {
                            'sentiment_score': round(sentiment_score, 3),
                            'sentiment_label': sentiment_label,
                            'rising_count': rising_count,
                            'falling_count': falling_count,
                            'rising_ratio': round(rising_ratio, 3),
                            'strong_rising': strong_rising,
                            'strong_falling': strong_falling,
                            'avg_change_pct': round(avg_change, 2) if avg_change else 0,
                            'total_analyzed': total
                        }
                        
        except Exception as e:
            logger.warning(f"分析市场情绪失败: {e}")
        
        return {
            'sentiment_label': '未知',
            'sentiment_score': 0,
            'rising_ratio': 0.5
        }
    
    def _analyze_market_technical(self, analysis_date: str) -> Dict[str, Any]:
        """分析市场技术面"""
        try:
            # 基于主要ETF的技术指标分析整体技术面
            major_etfs = ['510300', '159915', '510050']  # 沪深300、创业板、上证50
            
            technical_signals = []
            volume_trends = []
            
            for etf_code in major_etfs:
                etf_tech = self._get_etf_technical_analysis(etf_code)
                if etf_tech:
                    technical_signals.append(etf_tech.get('signal_score', 0))
                    volume_trends.append(etf_tech.get('volume_trend', 0))
            
            if technical_signals:
                avg_signal = np.mean(technical_signals)
                avg_volume = np.mean(volume_trends)
                
                if avg_signal > 0.5:
                    tech_environment = "技术面偏强"
                elif avg_signal > 0:
                    tech_environment = "技术面偏弱但稳定"
                elif avg_signal > -0.5:
                    tech_environment = "技术面偏弱"
                else:
                    tech_environment = "技术面较弱"
                
                return {
                    'technical_environment': tech_environment,
                    'signal_score': round(avg_signal, 2),
                    'volume_trend': round(avg_volume, 2),
                    'market_momentum': 'positive' if avg_signal > 0 else 'negative'
                }
                
        except Exception as e:
            logger.warning(f"分析市场技术面失败: {e}")
        
        return {
            'technical_environment': '技术面中性',
            'signal_score': 0,
            'market_momentum': 'neutral'
        }
    
    def _get_etf_technical_analysis(self, etf_code: str) -> Optional[Dict]:
        """获取ETF技术分析"""
        try:
            with self.db_manager.get_connection() as conn:
                query = """
                SELECT close, volume, ma5, ma20, price_change_pct
                FROM securities s 
                JOIN daily_quotes q ON s.id = q.security_id
                WHERE s.code = ? 
                ORDER BY q.trade_date DESC 
                LIMIT 20
                """
                df = pd.read_sql_query(query, conn, params=(etf_code,))
                
                if len(df) >= 5:
                    latest = df.iloc[0]
                    
                    # 计算技术信号评分
                    signal_score = 0
                    
                    # MA信号
                    if latest['ma5'] and latest['ma20']:
                        if latest['close'] > latest['ma5'] > latest['ma20']:
                            signal_score += 0.5
                        elif latest['close'] < latest['ma5'] < latest['ma20']:
                            signal_score -= 0.5
                    
                    # 价格动量
                    if latest['price_change_pct'] > 0:
                        signal_score += 0.3
                    else:
                        signal_score -= 0.3
                    
                    # 成交量趋势
                    recent_volume = df['volume'].head(5).mean()
                    older_volume = df['volume'].tail(5).mean()
                    volume_trend = (recent_volume - older_volume) / older_volume if older_volume > 0 else 0
                    
                    return {
                        'signal_score': signal_score,
                        'volume_trend': volume_trend
                    }
                    
        except Exception as e:
            logger.warning(f"获取ETF {etf_code} 技术分析失败: {e}")
        
        return None
    
    def _determine_market_environment(self, market_analysis: Dict) -> str:
        """综合判断市场环境"""
        try:
            sentiment = market_analysis.get('market_sentiment', {})
            technical = market_analysis.get('technical_analysis', {})
            
            sentiment_score = sentiment.get('sentiment_score', 0)
            signal_score = technical.get('signal_score', 0)
            
            # 综合评分
            combined_score = (sentiment_score + signal_score) / 2
            
            if combined_score > 0.3:
                return "市场环境良好"
            elif combined_score > 0:
                return "市场环境一般"
            elif combined_score > -0.3:
                return "市场环境偏弱"
            else:
                return "市场环境较差"
                
        except Exception as e:
            logger.warning(f"判断市场环境失败: {e}")
        
        return "市场环境未知"
    
    def _get_default_market_analysis(self, analysis_date: str) -> Dict[str, Any]:
        """获取默认市场分析"""
        return {
            'analysis_date': analysis_date,
            'market_environment': '数据不足',
            'market_sentiment': {'sentiment_label': '未知', 'sentiment_score': 0},
            'technical_analysis': {'technical_environment': '技术面未知', 'signal_score': 0},
            'indices_performance': {}
        }
    
    def enhance_stock_with_fundamentals(self, stock: Dict[str, Any]) -> Dict[str, Any]:
        """为股票添加基本面信息"""
        stock_code = stock.get('code', '')
        if not stock_code:
            return stock
        
        fundamentals = self.get_stock_fundamentals(stock_code)
        
        # 合并基本面数据
        enhanced_stock = stock.copy()
        enhanced_stock['fundamentals'] = fundamentals
        
        return enhanced_stock
    
    def adjust_recommendation_by_market(self, recommendation: Dict[str, Any], 
                                      market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """根据大盘表现调整个股建议"""
        try:
            market_env = market_analysis.get('market_environment', '市场环境未知')
            sentiment_score = market_analysis.get('market_sentiment', {}).get('sentiment_score', 0)
            technical_score = market_analysis.get('technical_analysis', {}).get('signal_score', 0)
            
            # 调整投资建议
            original_advice = recommendation.get('advice_type', '持有')
            adjusted_advice = original_advice
            
            adjustment_reason = []
            
            # 市场环境较差时，降低买入建议
            if sentiment_score < -0.3 or technical_score < -0.3:
                if '买入' in original_advice:
                    adjusted_advice = '谨慎持有'
                    adjustment_reason.append('大盘环境偏弱，建议谨慎')
                elif '持有' in original_advice:
                    adjustment_reason.append('大盘偏弱，持有需谨慎')
            
            # 市场环境良好时，可适当提升建议
            elif sentiment_score > 0.3 and technical_score > 0.3:
                if '持有' in original_advice:
                    adjusted_advice = '适量买入'
                    adjustment_reason.append('大盘环境良好，可适量增持')
            
            # 更新建议
            adjusted_recommendation = recommendation.copy()
            adjusted_recommendation['advice_type'] = adjusted_advice
            adjusted_recommendation['market_adjustment'] = {
                'original_advice': original_advice,
                'market_environment': market_env,
                'adjustment_reason': adjustment_reason
            }
            
            return adjusted_recommendation
            
        except Exception as e:
            logger.warning(f"根据大盘调整建议失败: {e}")
            return recommendation


def test_market_enhancer():
    """测试市场分析增强器"""
    enhancer = MarketAnalysisEnhancer()
    
    print("🧪 测试市场分析增强器")
    
    # 测试基本面数据获取
    print("\n=== 测试基本面数据 ===")
    fundamentals = enhancer.get_stock_fundamentals('000001')
    print(f"平安银行基本面: {fundamentals}")
    
    # 测试大盘分析
    print("\n=== 测试大盘分析 ===") 
    market_analysis = enhancer.get_market_analysis()
    print(f"市场环境: {market_analysis['market_environment']}")
    print(f"市场情绪: {market_analysis['market_sentiment'].get('sentiment_label', '未知')}")
    
    # 测试建议调整
    print("\n=== 测试建议调整 ===")
    sample_recommendation = {
        'advice_type': '买入',
        'confidence': '中等',
        'reasons': ['量化指标良好']
    }
    
    adjusted = enhancer.adjust_recommendation_by_market(sample_recommendation, market_analysis)
    print(f"原建议: {sample_recommendation['advice_type']}")
    print(f"调整后: {adjusted['advice_type']}")
    if 'market_adjustment' in adjusted:
        print(f"调整原因: {adjusted['market_adjustment'].get('adjustment_reason', [])}")


if __name__ == "__main__":
    test_market_enhancer()