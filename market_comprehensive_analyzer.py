#!/usr/bin/env python3
"""
市场综合分析引擎
整合大盘数据、行业板块、新闻等，提供技术面、基本面、消息面综合分析和市场评级
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

# 添加项目路径
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

# 导入数据获取器
from pure_tushare_news_fetcher import PureTushareNewsFetcher

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketComprehensiveAnalyzer:
    """市场综合分析引擎"""
    
    def __init__(self, config_path: str = "config.json"):
        """初始化分析引擎"""
        self.config = self._load_config(config_path)
        self.project_root = Path(__file__).parent.absolute()
        
        # 初始化数据获取器
        self.data_fetcher = PureTushareNewsFetcher(config_path)
        
        # 分析权重配置
        self.analysis_weights = {
            'technical': 0.35,      # 技术面权重 35%
            'fundamental': 0.30,    # 基本面权重 30%
            'sentiment': 0.20,      # 市场情绪权重 20%
            'news': 0.15           # 消息面权重 15%
        }
        
        # 评级标准
        self.rating_levels = {
            (80, 100): "强烈看多",
            (60, 80): "看多", 
            (40, 60): "中性",
            (20, 40): "看空",
            (0, 20): "强烈看空"
        }

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"加载配置文件失败: {e}")
            return {}

    def analyze_comprehensive_market(self, analysis_date: str = None, days: int = 5) -> Dict[str, Any]:
        """综合市场分析"""
        if analysis_date is None:
            analysis_date = datetime.now().strftime("%Y-%m-%d")
            
        logger.info(f"开始综合市场分析 - {analysis_date}")
        
        try:
            # 获取原始市场数据
            market_data = self.data_fetcher.get_comprehensive_market_data(days=days)
            
            if 'error' in market_data:
                return {
                    'error': f"数据获取失败: {market_data['error']}",
                    'analysis_date': analysis_date
                }
            
            # 进行各维度分析
            technical_analysis = self._analyze_technical_aspect(market_data)
            fundamental_analysis = self._analyze_fundamental_aspect(market_data)
            sentiment_analysis = self._analyze_sentiment_aspect(market_data)
            news_analysis = self._analyze_news_aspect(market_data)
            
            # 综合评分和评级
            comprehensive_score = self._calculate_comprehensive_score(
                technical_analysis, fundamental_analysis, sentiment_analysis, news_analysis
            )
            
            market_rating = self._determine_market_rating(comprehensive_score)
            
            # 生成交易指导
            trading_guidance = self._generate_trading_guidance(
                comprehensive_score, market_rating, technical_analysis, 
                fundamental_analysis, sentiment_analysis
            )
            
            # 整合最终分析结果
            result = {
                'analysis_date': analysis_date,
                'analysis_timestamp': datetime.now().isoformat(),
                'data_range_days': days,
                'raw_market_data': market_data,
                'technical_analysis': technical_analysis,
                'fundamental_analysis': fundamental_analysis,
                'sentiment_analysis': sentiment_analysis,
                'news_analysis': news_analysis,
                'comprehensive_score': comprehensive_score,
                'market_rating': market_rating,
                'trading_guidance': trading_guidance,
                'analysis_summary': self._generate_analysis_summary(
                    technical_analysis, fundamental_analysis, sentiment_analysis, 
                    news_analysis, comprehensive_score, market_rating
                ),
                'data_quality': self._assess_data_quality(market_data)
            }
            
            logger.info(f"综合分析完成 - 市场评级: {market_rating['rating']}, 综合评分: {comprehensive_score:.1f}")
            return result
            
        except Exception as e:
            logger.error(f"综合市场分析失败: {e}")
            return {
                'error': str(e),
                'analysis_date': analysis_date,
                'analysis_timestamp': datetime.now().isoformat()
            }

    def _analyze_technical_aspect(self, market_data: Dict) -> Dict[str, Any]:
        """技术面分析"""
        logger.info("进行技术面分析...")
        
        indices_data = market_data.get('indices', {})
        sector_data = market_data.get('sectors', {})
        sector_ranking = market_data.get('sector_ranking', {})
        
        technical_score = 50  # 基准分
        signals = []
        key_observations = []
        
        # 1. 主要指数技术分析
        if indices_data:
            # 计算主要指数平均表现
            major_indices = ['000001.SH', '399001.SZ', '399006.SZ', '000300.SH', '000905.SH']
            index_performances = []
            
            for code, data in indices_data.items():
                if code in major_indices:
                    change_pct = data.get('change_pct', 0)
                    index_performances.append(change_pct)
                    
                    # 单个指数分析
                    if change_pct > 1:
                        signals.append(f"{data['name']}强势上涨{change_pct:+.2f}%")
                    elif change_pct < -1:
                        signals.append(f"{data['name']}走弱下跌{change_pct:+.2f}%")
            
            avg_performance = np.mean(index_performances) if index_performances else 0
            
            # 技术评分调整
            technical_score += avg_performance * 15  # 指数表现权重较高
            
            key_observations.append(f"主要指数平均表现{avg_performance:+.2f}%")
            
            # 市场宽度分析
            rising_indices = sum(1 for data in indices_data.values() if data.get('change_pct', 0) > 0)
            market_breadth = rising_indices / len(indices_data)
            
            technical_score += (market_breadth - 0.5) * 20
            key_observations.append(f"市场宽度{market_breadth:.1%}指数上涨")
        
        # 2. 行业板块技术分析
        if sector_data:
            rising_sectors = sum(1 for data in sector_data.values() if data.get('change_pct', 0) > 0)
            sector_breadth = rising_sectors / len(sector_data)
            
            technical_score += (sector_breadth - 0.5) * 15
            key_observations.append(f"行业宽度{sector_breadth:.1%}行业上涨")
            
            # 强弱板块分析
            if sector_ranking.get('strong_sectors'):
                strong_sectors = sector_ranking['strong_sectors'][:3]
                signals.extend([f"{s['name']}表现强势{s['change_pct']:+.2f}%" for s in strong_sectors])
                
                # 检查是否有明显热点
                top_performance = strong_sectors[0]['change_pct'] if strong_sectors else 0
                if top_performance > 2:
                    technical_score += 5
                    signals.append("市场存在明显热点板块")
        
        # 3. 量能分析
        total_turnover = 0
        volume_signals = []
        
        for code, data in indices_data.items():
            turnover = data.get('turnover', 0)
            total_turnover += turnover
            
            # 简化的量能判断
            if turnover > 500000000000:  # 5000亿以上
                volume_signals.append(f"{data['name']}成交活跃")
        
        if volume_signals:
            signals.extend(volume_signals[:2])  # 最多显示2个量能信号
        
        # 限制技术评分范围
        technical_score = max(0, min(100, technical_score))
        
        # 技术等级
        if technical_score >= 70:
            technical_level = "强势"
        elif technical_score >= 55:
            technical_level = "偏强"
        elif technical_score >= 45:
            technical_level = "中性"
        elif technical_score >= 30:
            technical_level = "偏弱"
        else:
            technical_level = "弱势"
        
        return {
            'score': round(technical_score, 1),
            'level': technical_level,
            'key_observations': key_observations,
            'technical_signals': signals,
            'analysis_text': f"技术面呈现{technical_level}格局。{' '.join(key_observations[:2])}。" + 
                           (f"主要信号：{signals[0]}。" if signals else ""),
            'details': {
                'index_count': len(indices_data),
                'sector_count': len(sector_data),
                'rising_ratio': market_breadth if 'market_breadth' in locals() else 0
            }
        }

    def _analyze_fundamental_aspect(self, market_data: Dict) -> Dict[str, Any]:
        """基本面分析"""
        logger.info("进行基本面分析...")
        
        # 基本面分析主要基于宏观经济和政策环境
        # 由于缺乏具体基本面数据，主要通过行业结构和市场表现推断
        
        indices_data = market_data.get('indices', {})
        sector_data = market_data.get('sectors', {})
        sector_ranking = market_data.get('sector_ranking', {})
        
        fundamental_score = 50  # 基准分
        key_factors = []
        risk_factors = []
        
        # 1. 市场规模指数分析（反映经济基本面）
        large_cap_performance = 0
        small_cap_performance = 0
        
        for code, data in indices_data.items():
            change_pct = data.get('change_pct', 0)
            
            # 大盘股指标
            if code in ['000016.SH', '000300.SH']:  # 上证50, 沪深300
                large_cap_performance += change_pct
            
            # 小盘股指标  
            elif code in ['000852.SH', '932000.CSI']:  # 中证1000, 中证2000
                small_cap_performance += change_pct
        
        # 大小盘比较分析
        if large_cap_performance > small_cap_performance:
            fundamental_score += 5
            key_factors.append("大盘股相对强势，资金偏好稳健")
        elif small_cap_performance > large_cap_performance + 0.5:
            fundamental_score += 3
            key_factors.append("小盘股表现活跃，市场风险偏好提升")
        
        # 2. 行业结构分析
        if sector_ranking.get('strong_sectors'):
            strong_sectors = sector_ranking['strong_sectors']
            
            # 分析强势行业类型
            cyclical_sectors = ['钢铁', '有色', '化工', '建材', '机械']
            defensive_sectors = ['食品饮料', '医药', '公用事业', '银行']
            growth_sectors = ['电子', '计算机', '新能源', '军工']
            
            strong_sector_names = [s['name'] for s in strong_sectors]
            
            cyclical_count = sum(1 for name in strong_sector_names if any(c in name for c in cyclical_sectors))
            defensive_count = sum(1 for name in strong_sector_names if any(d in name for d in defensive_sectors))
            growth_count = sum(1 for name in strong_sector_names if any(g in name for g in growth_sectors))
            
            if cyclical_count >= 2:
                fundamental_score += 8
                key_factors.append("周期性行业表现强势，经济复苏预期增强")
            elif defensive_count >= 2:
                fundamental_score -= 3
                key_factors.append("防御性行业领涨，市场偏向避险")
            elif growth_count >= 2:
                fundamental_score += 5
                key_factors.append("成长性行业活跃，创新驱动明显")
        
        # 3. 金融板块分析（经济基本面的重要指标）
        financial_performance = []
        for code, data in sector_data.items():
            if any(keyword in data.get('name', '') for keyword in ['银行', '证券', '保险']):
                financial_performance.append(data.get('change_pct', 0))
        
        if financial_performance:
            avg_financial_performance = np.mean(financial_performance)
            if avg_financial_performance > 0.5:
                fundamental_score += 6
                key_factors.append("金融板块表现良好，流动性环境改善")
            elif avg_financial_performance < -0.5:
                fundamental_score -= 4
                risk_factors.append("金融板块承压，需关注流动性风险")
        
        # 4. 消费板块分析（内需基本面）
        consumer_performance = []
        for code, data in sector_data.items():
            if any(keyword in data.get('name', '') for keyword in ['食品', '家电', '汽车', '零售']):
                consumer_performance.append(data.get('change_pct', 0))
        
        if consumer_performance:
            avg_consumer_performance = np.mean(consumer_performance)
            if avg_consumer_performance > 0:
                fundamental_score += 4
                key_factors.append("消费相关板块稳定，内需基本面支撑")
            else:
                risk_factors.append("消费板块疲弱，内需有待改善")
        
        # 限制基本面评分范围
        fundamental_score = max(0, min(100, fundamental_score))
        
        # 基本面等级
        if fundamental_score >= 70:
            fundamental_level = "良好"
        elif fundamental_score >= 55:
            fundamental_level = "偏好"
        elif fundamental_score >= 45:
            fundamental_level = "中性"
        elif fundamental_score >= 30:
            fundamental_level = "偏弱"
        else:
            fundamental_level = "疲弱"
        
        return {
            'score': round(fundamental_score, 1),
            'level': fundamental_level,
            'key_factors': key_factors,
            'risk_factors': risk_factors,
            'analysis_text': f"基本面整体{fundamental_level}。" + 
                           (f"{key_factors[0]}。" if key_factors else "") +
                           (f"需要关注：{risk_factors[0]}。" if risk_factors else ""),
            'details': {
                'large_cap_vs_small_cap': 'large_cap' if large_cap_performance > small_cap_performance else 'small_cap',
                'key_factors_count': len(key_factors),
                'risk_factors_count': len(risk_factors)
            }
        }

    def _analyze_sentiment_aspect(self, market_data: Dict) -> Dict[str, Any]:
        """市场情绪分析"""
        logger.info("进行市场情绪分析...")
        
        market_sentiment = market_data.get('market_sentiment', {})
        sector_ranking = market_data.get('sector_ranking', {})
        indices_data = market_data.get('indices', {})
        
        # 直接使用数据获取器计算的情绪指标
        sentiment_score = market_sentiment.get('sentiment_score', 50)
        sentiment_level = market_sentiment.get('sentiment_level', '中性')
        
        sentiment_indicators = []
        market_characteristics = []
        
        # 1. 情绪指标解读
        avg_index_change = market_sentiment.get('avg_index_change', 0)
        rising_sector_ratio = market_sentiment.get('rising_sector_ratio', 0.5)
        
        sentiment_indicators.append(f"主要指数平均{avg_index_change:+.2f}%")
        sentiment_indicators.append(f"{rising_sector_ratio:.1%}行业上涨")
        
        # 2. 市场特征分析
        if rising_sector_ratio > 0.7:
            market_characteristics.append("普涨格局，市场情绪高涨")
        elif rising_sector_ratio < 0.3:
            market_characteristics.append("普跌格局，市场情绪低迷")
        else:
            market_characteristics.append("结构性行情，情绪分化")
        
        # 3. 极值指数分析（创业板和科创50代表风险偏好）
        growth_indices_performance = []
        for code, data in indices_data.items():
            if code in ['399006.SZ', '000688.SH']:  # 创业板指、科创50
                growth_indices_performance.append(data.get('change_pct', 0))
        
        if growth_indices_performance:
            avg_growth_performance = np.mean(growth_indices_performance)
            if avg_growth_performance > 0.5:
                market_characteristics.append("成长股活跃，风险偏好提升")
            elif avg_growth_performance < -1:
                market_characteristics.append("成长股承压，风险偏好下降")
        
        # 4. 板块轮动分析
        if sector_ranking.get('strong_sectors') and sector_ranking.get('weak_sectors'):
            strong_sectors = sector_ranking['strong_sectors']
            weak_sectors = sector_ranking['weak_sectors']
            
            performance_gap = strong_sectors[0]['change_pct'] - weak_sectors[-1]['change_pct']
            
            if performance_gap > 3:
                market_characteristics.append("板块分化明显，资金集中流入热点")
            elif performance_gap < 1:
                market_characteristics.append("板块走势趋同，市场缺乏明确方向")
        
        return {
            'score': round(sentiment_score, 1),
            'level': sentiment_level,
            'sentiment_indicators': sentiment_indicators,
            'market_characteristics': market_characteristics,
            'analysis_text': f"市场情绪{sentiment_level}。{sentiment_indicators[0]}，{sentiment_indicators[1]}。" +
                           (f"{market_characteristics[0]}。" if market_characteristics else ""),
            'details': {
                'avg_index_change': avg_index_change,
                'rising_sector_ratio': rising_sector_ratio,
                'sentiment_description': market_sentiment.get('description', '')
            }
        }

    def _analyze_news_aspect(self, market_data: Dict) -> Dict[str, Any]:
        """消息面分析"""
        logger.info("进行消息面分析...")
        
        news_headlines = market_data.get('news_headlines', [])
        
        news_score = 50  # 基准分
        key_news = []
        news_themes = []
        importance_distribution = {'high': 0, 'medium': 0, 'low': 0}
        
        # 分析新闻
        if news_headlines:
            for news in news_headlines:
                importance = news.get('importance', 'low')
                importance_distribution[importance] += 1
                
                # 重要新闻影响评分
                if importance == 'high':
                    news_score += 3
                elif importance == 'medium':
                    news_score += 1
                
                # 收集关键新闻（只保留高重要性的）
                if importance == 'high' and len(key_news) < 3:
                    key_news.append(news['title'][:50] + '...')
                
                # 提取新闻主题
                keywords = news.get('keywords', [])
                news_themes.extend(keywords)
        
        # 主题统计
        theme_counts = {}
        for theme in news_themes:
            theme_counts[theme] = theme_counts.get(theme, 0) + 1
        
        # 获取主要主题
        main_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # 新闻数量调整
        if len(news_headlines) > 15:
            news_score += 2  # 新闻丰富
        elif len(news_headlines) < 5:
            news_score -= 3  # 新闻较少
        
        # 限制新闻评分范围
        news_score = max(0, min(100, news_score))
        
        # 新闻面等级
        if news_score >= 65:
            news_level = "积极"
        elif news_score >= 55:
            news_level = "偏积极"
        elif news_score >= 45:
            news_level = "中性"
        elif news_score >= 35:
            news_level = "偏消极"
        else:
            news_level = "消极"
        
        return {
            'score': round(news_score, 1),
            'level': news_level,
            'key_news': key_news,
            'main_themes': [theme for theme, count in main_themes],
            'importance_distribution': importance_distribution,
            'analysis_text': f"消息面{news_level}。共收集到{len(news_headlines)}条财经新闻，" +
                           f"其中高重要性{importance_distribution['high']}条。" +
                           (f"主要主题：{main_themes[0][0]}。" if main_themes else ""),
            'details': {
                'total_news_count': len(news_headlines),
                'theme_diversity': len(theme_counts)
            }
        }

    def _calculate_comprehensive_score(self, technical: Dict, fundamental: Dict, 
                                     sentiment: Dict, news: Dict) -> float:
        """计算综合评分"""
        
        comprehensive_score = (
            technical['score'] * self.analysis_weights['technical'] +
            fundamental['score'] * self.analysis_weights['fundamental'] + 
            sentiment['score'] * self.analysis_weights['sentiment'] +
            news['score'] * self.analysis_weights['news']
        )
        
        return round(comprehensive_score, 1)

    def _determine_market_rating(self, comprehensive_score: float) -> Dict[str, Any]:
        """确定市场评级"""
        
        rating = "中性"
        for (min_score, max_score), level in self.rating_levels.items():
            if min_score <= comprehensive_score < max_score:
                rating = level
                break
        
        # 风险等级
        if comprehensive_score >= 70:
            risk_level = "低风险"
        elif comprehensive_score >= 50:
            risk_level = "中等风险"
        else:
            risk_level = "高风险"
        
        # 投资建议
        if comprehensive_score >= 70:
            investment_advice = "积极配置，适度增仓"
        elif comprehensive_score >= 55:
            investment_advice = "谨慎乐观，选择性配置"
        elif comprehensive_score >= 45:
            investment_advice = "保持中性仓位，观望为主"
        elif comprehensive_score >= 30:
            investment_advice = "控制仓位，防范风险"
        else:
            investment_advice = "减少仓位，回避风险"
        
        return {
            'rating': rating,
            'score': comprehensive_score,
            'risk_level': risk_level,
            'investment_advice': investment_advice,
            'confidence': min(90, max(60, comprehensive_score))  # 置信度
        }

    def _generate_trading_guidance(self, comprehensive_score: float, market_rating: Dict,
                                 technical: Dict, fundamental: Dict, sentiment: Dict) -> Dict[str, Any]:
        """生成交易指导"""
        
        guidance = {
            'overall_strategy': market_rating['investment_advice'],
            'position_suggestion': '',
            'sector_focus': [],
            'risk_management': [],
            'market_timing': '',
            'key_levels': {},
            'next_trading_day_outlook': ''
        }
        
        # 仓位建议
        if comprehensive_score >= 70:
            guidance['position_suggestion'] = "建议仓位70-80%"
        elif comprehensive_score >= 55:
            guidance['position_suggestion'] = "建议仓位50-70%"
        elif comprehensive_score >= 45:
            guidance['position_suggestion'] = "建议仓位30-50%"
        else:
            guidance['position_suggestion'] = "建议仓位20-30%"
        
        # 板块关注（基于技术和基本面分析）
        if technical.get('technical_signals'):
            guidance['sector_focus'].extend(technical['technical_signals'][:2])
        
        # 风险管理
        if comprehensive_score < 50:
            guidance['risk_management'].append("严格止损，控制单股仓位")
            guidance['risk_management'].append("关注系统性风险")
        
        if sentiment['score'] < 40:
            guidance['risk_management'].append("市场情绪低迷，避免追涨杀跌")
        
        # 市场时机判断
        if technical['score'] > 60 and sentiment['score'] > 60:
            guidance['market_timing'] = "技术面和情绪面共振，关注做多机会"
        elif technical['score'] < 40 and sentiment['score'] < 40:
            guidance['market_timing'] = "技术面和情绪面偏弱，谨慎参与"
        else:
            guidance['market_timing'] = "市场信号分化，等待明确方向"
        
        # 下一交易日展望
        if comprehensive_score >= 60:
            guidance['next_trading_day_outlook'] = "预计下一交易日延续强势，可关注热点板块"
        elif comprehensive_score >= 40:
            guidance['next_trading_day_outlook'] = "预计下一交易日维持震荡，结构性机会为主"
        else:
            guidance['next_trading_day_outlook'] = "预计下一交易日偏弱，以防守为主"
        
        return guidance

    def _generate_analysis_summary(self, technical: Dict, fundamental: Dict, 
                                 sentiment: Dict, news: Dict, 
                                 comprehensive_score: float, market_rating: Dict) -> str:
        """生成分析摘要"""
        
        summary_parts = [
            f"**市场评级**: {market_rating['rating']} (综合评分: {comprehensive_score})",
            f"**技术面**: {technical['level']} ({technical['score']}分)",
            f"**基本面**: {fundamental['level']} ({fundamental['score']}分)", 
            f"**市场情绪**: {sentiment['level']} ({sentiment['score']}分)",
            f"**消息面**: {news['level']} ({news['score']}分)",
            "",
            f"**投资建议**: {market_rating['investment_advice']}",
            f"**风险等级**: {market_rating['risk_level']}",
            "",
            "**关键要点**:",
            f"- 技术面：{technical['analysis_text']}",
            f"- 基本面：{fundamental['analysis_text']}",
            f"- 市场情绪：{sentiment['analysis_text']}",
            f"- 消息面：{news['analysis_text']}"
        ]
        
        return "\n".join(summary_parts)

    def _assess_data_quality(self, market_data: Dict) -> Dict[str, Any]:
        """评估数据质量"""
        
        indices_count = len(market_data.get('indices', {}))
        sectors_count = len(market_data.get('sectors', {}))
        news_count = len(market_data.get('news_headlines', []))
        
        # 数据完整性评分
        data_score = 0
        if indices_count >= 8:
            data_score += 30
        elif indices_count >= 5:
            data_score += 20
        elif indices_count >= 3:
            data_score += 10
        
        if sectors_count >= 20:
            data_score += 40
        elif sectors_count >= 10:
            data_score += 25
        elif sectors_count >= 5:
            data_score += 15
        
        if news_count >= 10:
            data_score += 30
        elif news_count >= 5:
            data_score += 20
        elif news_count >= 1:
            data_score += 10
        
        # 数据质量等级
        if data_score >= 80:
            quality_level = "优秀"
        elif data_score >= 60:
            quality_level = "良好"
        elif data_score >= 40:
            quality_level = "一般"
        else:
            quality_level = "较差"
        
        return {
            'quality_score': data_score,
            'quality_level': quality_level,
            'indices_count': indices_count,
            'sectors_count': sectors_count,
            'news_count': news_count,
            'completeness': f"指数{indices_count}个，行业{sectors_count}个，新闻{news_count}条"
        }

    def save_analysis_report(self, analysis_result: Dict, analysis_date: str = None) -> str:
        """保存分析报告"""
        
        if analysis_date is None:
            analysis_date = datetime.now().strftime("%Y-%m-%d")
        
        # 创建报告目录
        reports_dir = self.project_root / "reports" / "market_comprehensive"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存JSON格式的详细数据
        date_str = analysis_date.replace('-', '')
        json_path = reports_dir / f"市场综合分析_{date_str}.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2, default=str)
        
        # 生成Markdown格式的报告
        md_path = reports_dir / f"市场综合分析报告_{date_str}.md"
        
        md_content = self._generate_markdown_report(analysis_result)
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"分析报告已保存: JSON({json_path}), MD({md_path})")
        return str(md_path)

    def _generate_markdown_report(self, analysis_result: Dict) -> str:
        """生成Markdown格式报告"""
        
        lines = [
            "# 📊 市场综合分析报告",
            "",
            f"**分析日期**: {analysis_result.get('analysis_date', '')}",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**数据范围**: 最近{analysis_result.get('data_range_days', 5)}个交易日",
            "",
            "## 🎯 核心结论",
            "",
            analysis_result.get('analysis_summary', ''),
            "",
            "## 📈 详细分析",
            "",
        ]
        
        # 各维度分析
        analyses = ['technical_analysis', 'fundamental_analysis', 'sentiment_analysis', 'news_analysis']
        analysis_names = ['技术面分析', '基本面分析', '市场情绪分析', '消息面分析']
        
        for analysis_key, analysis_name in zip(analyses, analysis_names):
            analysis = analysis_result.get(analysis_key, {})
            if analysis:
                lines.extend([
                    f"### {analysis_name}",
                    "",
                    f"**评分**: {analysis.get('score', 'N/A')}分 | **等级**: {analysis.get('level', 'N/A')}",
                    "",
                    analysis.get('analysis_text', ''),
                    ""
                ])
        
        # 交易指导
        trading_guidance = analysis_result.get('trading_guidance', {})
        if trading_guidance:
            lines.extend([
                "## 🎯 交易指导",
                "",
                f"**整体策略**: {trading_guidance.get('overall_strategy', '')}",
                f"**仓位建议**: {trading_guidance.get('position_suggestion', '')}",
                f"**市场时机**: {trading_guidance.get('market_timing', '')}",
                f"**下日展望**: {trading_guidance.get('next_trading_day_outlook', '')}",
                ""
            ])
            
            if trading_guidance.get('risk_management'):
                lines.extend([
                    "**风险管理要点**:",
                    *[f"- {risk}" for risk in trading_guidance['risk_management']],
                    ""
                ])
        
        # 数据质量
        data_quality = analysis_result.get('data_quality', {})
        if data_quality:
            lines.extend([
                "## 📊 数据质量评估",
                "",
                f"**数据质量**: {data_quality.get('quality_level', '')} ({data_quality.get('quality_score', 0)}分)",
                f"**数据完整性**: {data_quality.get('completeness', '')}",
                ""
            ])
        
        # 免责声明
        lines.extend([
            "---",
            "",
            "## ⚠️ 重要声明",
            "",
            "本报告基于公开市场数据和技术分析方法生成，仅供参考，不构成投资建议。",
            "市场存在不确定性，投资者应根据自身情况谨慎决策，风险自负。",
            "",
            f"*报告由Claude AI自动生成 | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        ])
        
        return "\n".join(lines)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="市场综合分析引擎")
    parser.add_argument("--date", type=str, default=None,
                       help="分析日期 (YYYY-MM-DD，默认为今天)")
    parser.add_argument("--days", type=int, default=5,
                       help="数据回溯天数，默认5天")
    parser.add_argument("--save", action="store_true",
                       help="保存分析报告")
    parser.add_argument("--config", type=str, default="config.json",
                       help="配置文件路径")
    
    args = parser.parse_args()
    
    print(f"🚀 启动市场综合分析引擎")
    print(f"📅 分析日期: {args.date or '今日'}")
    print(f"📈 数据天数: {args.days}天")
    
    # 创建分析引擎
    analyzer = MarketComprehensiveAnalyzer(args.config)
    
    # 进行综合分析
    print(f"🔄 开始综合市场分析...\n")
    result = analyzer.analyze_comprehensive_market(args.date, args.days)
    
    if 'error' in result:
        print(f"❌ 分析失败: {result['error']}")
        return
    
    # 显示核心结果
    market_rating = result['market_rating']
    print(f"🎯 市场评级: {market_rating['rating']}")
    print(f"📊 综合评分: {market_rating['score']}")
    print(f"⚠️  风险等级: {market_rating['risk_level']}")
    print(f"💡 投资建议: {market_rating['investment_advice']}")
    
    # 显示各维度评分
    print(f"\n📈 各维度评分:")
    print(f"  技术面: {result['technical_analysis']['score']}分 ({result['technical_analysis']['level']})")
    print(f"  基本面: {result['fundamental_analysis']['score']}分 ({result['fundamental_analysis']['level']})")
    print(f"  市场情绪: {result['sentiment_analysis']['score']}分 ({result['sentiment_analysis']['level']})")
    print(f"  消息面: {result['news_analysis']['score']}分 ({result['news_analysis']['level']})")
    
    # 保存报告
    if args.save:
        report_path = analyzer.save_analysis_report(result, args.date)
        print(f"\n💾 分析报告已保存: {report_path}")
    
    print(f"\n✅ 市场综合分析完成!")


if __name__ == "__main__":
    main()