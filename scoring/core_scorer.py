#!/usr/bin/env python3
"""
核心评分器
Core Scorer Module

基于实际选股表现优化的股票综合评分系统
"""

import sqlite3
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd

from .config import ScoringConfig, DEFAULT_CONFIG
from .factor_calculator import FactorCalculator

class StockScorer:
    """股票评分器 - 生产级评分系统"""
    
    def __init__(self, config: Optional[ScoringConfig] = None, db_path: str = "data_adapter/stock_data.db"):
        self.config = config or DEFAULT_CONFIG
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.factor_calculator = FactorCalculator(db_path)
        
        # 验证配置
        self.config.validate()
    
    def calculate_score(self, stock_code: str, trade_date: str) -> Dict:
        """
        计算单只股票的综合评分
        
        Args:
            stock_code: 股票代码
            trade_date: 交易日期
            
        Returns:
            包含评分详情的字典
        """
        try:
            # 计算各因子分数
            factor_scores = self.factor_calculator.calculate_all_factors(stock_code, trade_date)
            
            # 加权综合评分
            composite_score = (
                factor_scores['momentum'] * self.config.momentum_factor_weight +
                factor_scores['mean_reversion'] * self.config.mean_reversion_weight +
                factor_scores['volume_breakout'] * self.config.volume_breakout_weight +
                factor_scores['relative_performance'] * self.config.relative_performance_weight +
                factor_scores['stability'] * self.config.stability_factor_weight
            )
            
            # 确定推荐等级和置信度
            recommendation, confidence = self._determine_recommendation(composite_score)
            
            # 获取股票基本信息
            stock_info = self._get_stock_info(stock_code)
            
            return {
                'stock_code': stock_code,
                'stock_name': stock_info.get('name', 'Unknown'),
                'trade_date': trade_date,
                'composite_score': round(composite_score, 1),
                'recommendation': recommendation,
                'confidence': confidence,
                'factor_scores': {k: round(v, 1) for k, v in factor_scores.items()},
                'factor_weights': {
                    'momentum': self.config.momentum_factor_weight,
                    'mean_reversion': self.config.mean_reversion_weight,
                    'volume_breakout': self.config.volume_breakout_weight,
                    'relative_performance': self.config.relative_performance_weight,
                    'stability': self.config.stability_factor_weight
                },
                'industry': stock_info.get('industry', 'Unknown'),
                'market_cap': stock_info.get('market_cap', None)
            }
            
        except Exception as e:
            print(f"计算评分失败 {stock_code}: {e}")
            return self._get_default_score_result(stock_code, trade_date)
    
    def _determine_recommendation(self, score: float) -> Tuple[str, str]:
        """确定推荐等级和置信度"""
        if score >= self.config.buy_threshold:
            return "买入", "高"
        elif score >= self.config.cautious_buy_threshold:
            return "谨慎买入", "中"
        elif score >= self.config.watch_threshold:
            return "观望", "低"
        else:
            return "回避", "低"
    
    def _get_stock_info(self, stock_code: str) -> Dict:
        """获取股票基本信息"""
        try:
            query = """
                SELECT s.name, s.industry, db.total_mv as market_cap
                FROM securities s
                LEFT JOIN daily_basic db ON s.id = db.security_id
                WHERE s.code = ?
                ORDER BY db.trade_date DESC
                LIMIT 1
            """
            
            result = self.conn.execute(query, [stock_code]).fetchone()
            
            if result:
                return {
                    'name': result[0] or 'Unknown',
                    'industry': result[1] or 'Unknown',
                    'market_cap': result[2]
                }
            else:
                return {'name': 'Unknown', 'industry': 'Unknown', 'market_cap': None}
                
        except Exception as e:
            print(f"获取股票信息失败 {stock_code}: {e}")
            return {'name': 'Unknown', 'industry': 'Unknown', 'market_cap': None}
    
    def _get_default_score_result(self, stock_code: str, trade_date: str) -> Dict:
        """获取默认评分结果"""
        return {
            'stock_code': stock_code,
            'stock_name': 'Unknown',
            'trade_date': trade_date,
            'composite_score': 50.0,
            'recommendation': '观望',
            'confidence': '低',
            'factor_scores': {
                'momentum': 50.0,
                'mean_reversion': 50.0,
                'volume_breakout': 50.0,
                'relative_performance': 50.0,
                'stability': 50.0
            },
            'factor_weights': {
                'momentum': self.config.momentum_factor_weight,
                'mean_reversion': self.config.mean_reversion_weight,
                'volume_breakout': self.config.volume_breakout_weight,
                'relative_performance': self.config.relative_performance_weight,
                'stability': self.config.stability_factor_weight
            },
            'industry': 'Unknown',
            'market_cap': None
        }
    
    def batch_score(self, stock_codes: List[str], trade_date: str, 
                   show_progress: bool = True) -> List[Dict]:
        """
        批量评分
        
        Args:
            stock_codes: 股票代码列表
            trade_date: 交易日期
            show_progress: 是否显示进度
            
        Returns:
            评分结果列表
        """
        results = []
        total = len(stock_codes)
        
        for i, code in enumerate(stock_codes):
            if show_progress and i % 50 == 0:
                print(f"评分进度: {i+1}/{total}")
            
            try:
                result = self.calculate_score(code, trade_date)
                results.append(result)
            except Exception as e:
                print(f"评分失败 {code}: {e}")
                continue
        
        return results
    
    def get_top_picks(self, stock_codes: List[str], trade_date: str, 
                     top_n: int = 20) -> List[Dict]:
        """
        获取评分最高的股票
        
        Args:
            stock_codes: 候选股票列表
            trade_date: 交易日期
            top_n: 返回前N只股票
            
        Returns:
            评分最高的N只股票
        """
        all_scores = self.batch_score(stock_codes, trade_date)
        
        # 按评分排序
        sorted_scores = sorted(all_scores, key=lambda x: x['composite_score'], reverse=True)
        
        return sorted_scores[:top_n]
    
    def get_buy_recommendations(self, stock_codes: List[str], trade_date: str) -> List[Dict]:
        """
        获取买入推荐股票
        
        Args:
            stock_codes: 候选股票列表
            trade_date: 交易日期
            
        Returns:
            买入推荐股票列表
        """
        all_scores = self.batch_score(stock_codes, trade_date)
        
        # 筛选买入推荐
        buy_recommendations = [
            score for score in all_scores 
            if score['recommendation'] in ['买入', '谨慎买入']
        ]
        
        # 按评分排序
        return sorted(buy_recommendations, key=lambda x: x['composite_score'], reverse=True)
    
    def analyze_score_distribution(self, results: List[Dict]) -> Dict:
        """
        分析评分分布
        
        Args:
            results: 评分结果列表
            
        Returns:
            分布统计信息
        """
        if not results:
            return {}
        
        scores = [r['composite_score'] for r in results]
        recommendations = [r['recommendation'] for r in results]
        
        # 评分统计
        score_stats = {
            'total_count': len(scores),
            'mean_score': sum(scores) / len(scores),
            'max_score': max(scores),
            'min_score': min(scores),
            'score_ranges': {}
        }
        
        # 评分区间统计
        ranges = [(80, 100, '80+'), (70, 80, '70-80'), (60, 70, '60-70'), (0, 60, '<60')]
        for min_val, max_val, label in ranges:
            count = len([s for s in scores if min_val <= s < max_val])
            score_stats['score_ranges'][label] = {
                'count': count,
                'percentage': count / len(scores) * 100
            }
        
        # 推荐等级统计
        recommendation_stats = {}
        for rec in set(recommendations):
            count = recommendations.count(rec)
            recommendation_stats[rec] = {
                'count': count,
                'percentage': count / len(recommendations) * 100
            }
        
        return {
            'score_distribution': score_stats,
            'recommendation_distribution': recommendation_stats
        }
    
    def __del__(self):
        """清理数据库连接"""
        if hasattr(self, 'conn'):
            self.conn.close()