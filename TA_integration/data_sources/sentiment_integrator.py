#!/usr/bin/env python3
"""
中国股票情绪数据整合器
整合东方财富、雪球等多个中文社交媒体平台的情绪数据
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

# 添加路径
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent / "agents"))

try:
    from eastmoney_api import EastMoneyAPI
    from xueqiu_api import XueqiuAPI  # 新增雪球API
    from chinese_sentiment_analyst import ChineseMarketSentimentAnalyst
except ImportError as e:
    print(f"导入失败: {e}")
    # 创建dummy类以避免运行时错误
    class EastMoneyAPI:
        def __init__(self):
            pass
        def get_stock_sentiment_summary(self, stock_code):
            return {'avg_sentiment': 0.0, 'summary': '情绪数据获取失败', 'total_posts': 0}
    
    class XueqiuAPI:
        def __init__(self):
            pass
        def get_stock_sentiment_summary(self, stock_code):
            return {'avg_sentiment': 0.0, 'summary': '雪球数据获取失败', 'total_posts': 0}
    
    class ChineseMarketSentimentAnalyst:
        def __init__(self):
            pass
        def analyze_sentiment_with_filtering(self, posts):
            return {}

logger = logging.getLogger(__name__)

class ChineseSentimentIntegrator:
    """中国股票情绪数据整合器"""
    
    def __init__(self):
        """初始化整合器"""
        self.eastmoney_api = EastMoneyAPI()
        self.xueqiu_api = XueqiuAPI()  # 重新启用雪球API
        self.sentiment_analyst = ChineseMarketSentimentAnalyst()
        
        # 数据源权重配置
        self.source_weights = {
            'eastmoney': 0.6,      # 东方财富股吧权重
            'xueqiu': 0.4,         # 雪球权重（重新启用）
            'other': 0.0           # 其他数据源权重（已禁用）
        }
        
        # 缓存配置
        self.cache_dir = current_dir.parent / "data" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expire_hours = 2  # 缓存2小时过期
    
    def get_china_stock_sentiment(self, stock_code: str, analysis_date: str, 
                                look_back_days: int = 7) -> Dict[str, Any]:
        """获取中国股票综合情绪分析 - 简化版本，跳过实际数据获取"""
        try:
            logger.info(f"情绪分析已禁用，返回默认数据: {stock_code}")
            
            # 直接返回无情绪数据的结果
            return self._create_no_sentiment_result(stock_code)
            
        except Exception as e:
            logger.error(f"获取股票{stock_code}情绪分析失败: {e}")
            return self._error_result(stock_code, str(e))
    
    def _get_xueqiu_sentiment(self, stock_code: str, look_back_days: int) -> Dict[str, Any]:
        """获取雪球情绪数据（已禁用）"""
        # 雪球API相关功能已移除
        logger.info(f"雪球数据源已禁用: {stock_code}")
        return self._empty_sentiment_data()
    
    def _try_pysnowball_api(self, stock_code: str) -> Dict[str, Any]:
        """雪球API已禁用"""
        logger.info(f"雪球API已禁用: {stock_code}")
        return self._empty_sentiment_data()
    
    def _mock_xueqiu_data(self, stock_code: str) -> Dict[str, Any]:
        """雪球数据已禁用"""
        logger.info(f"雪球数据源已禁用: {stock_code}")
        return self._empty_sentiment_data()
    
    def _integrate_sentiment_data(self, stock_code: str, sentiment_data: Dict) -> Dict[str, Any]:
        """整合多源情绪数据"""
        
        # 提取各数据源的情绪指标
        eastmoney = sentiment_data.get('eastmoney', {})
        xueqiu = sentiment_data.get('xueqiu', {})
        
        # 计算加权平均情绪
        total_weight = 0
        weighted_sentiment = 0
        
        # 东方财富数据
        if eastmoney.get('total_posts', 0) > 0:
            em_sentiment = eastmoney.get('avg_sentiment', 0.0)
            em_weight = self.source_weights['eastmoney']
            weighted_sentiment += em_sentiment * em_weight
            total_weight += em_weight
        
        # 雪球数据（重新启用）
        if xueqiu.get('total_posts', 0) > 0:
            xq_sentiment = xueqiu.get('avg_sentiment', 0.0)
            xq_weight = self.source_weights['xueqiu']
            weighted_sentiment += xq_sentiment * xq_weight
            total_weight += xq_weight
        
        # 计算最终情绪指数
        final_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0.0
        
        # 合并讨论数量（东方财富 + 雪球）
        total_discussions = eastmoney.get('total_posts', 0) + xueqiu.get('total_posts', 0)
        
        # 提取热门话题（东方财富 + 雪球）
        hot_topics = []
        if eastmoney.get('hot_topics'):
            hot_topics.extend(eastmoney['hot_topics'][:3])  # 东方财富取3个
        if xueqiu.get('hot_topics'):
            hot_topics.extend(xueqiu['hot_topics'][:2])     # 雪球取2个
        
        # 情绪分布统计（东方财富 + 雪球）
        em_dist = eastmoney.get('sentiment_distribution', {'positive': 0, 'neutral': 0, 'negative': 0})
        xq_dist = xueqiu.get('sentiment_distribution', {'positive': 0, 'neutral': 0, 'negative': 0})
        sentiment_distribution = {
            'positive': em_dist['positive'] + xq_dist['positive'],
            'neutral': em_dist['neutral'] + xq_dist['neutral'],
            'negative': em_dist['negative'] + xq_dist['negative']
        }
        
        # 生成综合分析报告
        analysis_summary = self._generate_integrated_summary(
            stock_code, final_sentiment, total_discussions, sentiment_data
        )
        
        # 使用高级情绪分析师进行后处理
        posts_for_analysis = self._prepare_posts_for_analysis(sentiment_data)
        if posts_for_analysis:
            advanced_analysis = self.sentiment_analyst.analyze_sentiment_with_filtering(posts_for_analysis)
        else:
            advanced_analysis = {}
        
        return {
            'stock_code': stock_code,
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            'integrated_sentiment': round(final_sentiment, 3),
            'total_discussions': total_discussions,
            'sentiment_distribution': sentiment_distribution,
            'hot_topics': hot_topics[:5],  # 限制5个热门话题
            'data_sources': {
                'eastmoney': eastmoney,
                'xueqiu': xueqiu,
                'source_weights': self.source_weights
            },
            'advanced_analysis': advanced_analysis,
            'analysis_summary': analysis_summary,
            'confidence_level': self._calculate_confidence_level(sentiment_data),
            'data_freshness': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _merge_sentiment_distributions(self, distributions: List[Dict]) -> Dict[str, int]:
        """合并情绪分布"""
        merged = {'positive': 0, 'neutral': 0, 'negative': 0}
        
        for dist in distributions:
            if dist:
                merged['positive'] += dist.get('positive', 0)
                merged['neutral'] += dist.get('neutral', 0) 
                merged['negative'] += dist.get('negative', 0)
        
        return merged
    
    def _prepare_posts_for_analysis(self, sentiment_data: Dict) -> List[Dict]:
        """准备帖子数据供高级分析"""
        posts = []
        
        # 从各数据源提取帖子（如果有详细数据）
        for source_name, source_data in sentiment_data.items():
            if source_name in ['eastmoney', 'xueqiu'] and isinstance(source_data, dict):
                # 这里需要详细的帖子数据，目前的API返回的是汇总数据
                # 可以考虑在各API中返回详细帖子数据
                pass
        
        return posts
    
    def _generate_integrated_summary(self, stock_code: str, sentiment: float, 
                                   discussions: int, sentiment_data: Dict) -> str:
        """生成综合分析摘要"""
        
        # 情绪标签
        if sentiment > 0.3:
            sentiment_label = "强烈看好"
        elif sentiment > 0.15:
            sentiment_label = "偏向看好"
        elif sentiment > 0.05:
            sentiment_label = "略微看好"
        elif sentiment < -0.3:
            sentiment_label = "强烈看空"
        elif sentiment < -0.15:
            sentiment_label = "偏向看空"
        elif sentiment < -0.05:
            sentiment_label = "略微看空"
        else:
            sentiment_label = "中性"
        
        # 讨论热度
        if discussions > 200:
            heat_label = "讨论非常活跃"
        elif discussions > 100:
            heat_label = "讨论较为活跃"
        elif discussions > 50:
            heat_label = "有一定讨论"
        else:
            heat_label = "讨论较少"
        
        # 数据源情况
        eastmoney_posts = sentiment_data.get('eastmoney', {}).get('total_posts', 0)
        xueqiu_posts = sentiment_data.get('xueqiu', {}).get('total_posts', 0)
        
        # 构建数据源说明
        source_info = []
        if eastmoney_posts > 0:
            source_info.append(f"东方财富{eastmoney_posts}条")
        if xueqiu_posts > 0:
            source_info.append(f"雪球{xueqiu_posts}条")
        
        source_str = "、".join(source_info) if source_info else "各平台"
        
        summary = f"股票{stock_code}在{source_str}讨论中整体情绪{sentiment_label}，{heat_label}。"
        summary += f"共分析{discussions}条讨论。"
        summary += f"综合情绪指数{sentiment:.3f}。"
        
        return summary
    
    def _calculate_confidence_level(self, sentiment_data: Dict) -> str:
        """计算综合置信度"""
        total_posts = 0
        data_sources_count = 0
        
        for source_name, source_data in sentiment_data.items():
            if source_name in ['eastmoney', 'xueqiu'] and isinstance(source_data, dict):
                posts = source_data.get('total_posts', 0)
                if posts > 0:
                    total_posts += posts
                    data_sources_count += 1
        
        # 基于数据量和数据源数量判断置信度
        if total_posts > 100 and data_sources_count >= 2:
            return "高"
        elif total_posts > 50 or data_sources_count >= 2:
            return "中"
        elif total_posts > 20:
            return "中"
        else:
            return "低"
    
    def _empty_sentiment_data(self) -> Dict[str, Any]:
        """空情绪数据"""
        return {
            'total_posts': 0,
            'avg_sentiment': 0.0,
            'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
            'hot_topics': [],
            'summary': '暂无数据'
        }
    
    def _create_no_sentiment_result(self, stock_code: str) -> Dict[str, Any]:
        """创建无情绪数据的标准结果"""
        return {
            'stock_code': stock_code,
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            'integrated_sentiment': 0.0,
            'total_discussions': 0,
            'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
            'hot_topics': [],
            'data_sources': {
                'eastmoney': self._empty_sentiment_data(),
                'xueqiu': self._empty_sentiment_data(),
                'source_weights': self.source_weights
            },
            'advanced_analysis': {},
            'analysis_summary': f'股票{stock_code}情绪分析功能已禁用，使用默认中性情绪',
            'confidence_level': '无',
            'data_freshness': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _error_result(self, stock_code: str, error_msg: str) -> Dict[str, Any]:
        """错误结果"""
        return {
            'stock_code': stock_code,
            'error': error_msg,
            'integrated_sentiment': 0.0,
            'total_discussions': 0,
            'analysis_summary': f'股票{stock_code}情绪分析失败: {error_msg}',
            'confidence_level': '低'
        }
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """从缓存获取数据"""
        try:
            cache_file = self.cache_dir / f"sentiment_{cache_key}.json"
            if cache_file.exists():
                # 检查缓存时间
                file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if datetime.now() - file_time < timedelta(hours=self.cache_expire_hours):
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                    logger.debug(f"从缓存读取情绪数据: {cache_key}")
                    return cached_data
                else:
                    # 删除过期缓存
                    cache_file.unlink()
        except Exception as e:
            logger.warning(f"读取缓存失败: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, data: Dict):
        """保存数据到缓存"""
        try:
            cache_file = self.cache_dir / f"sentiment_{cache_key}.json"
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(f"情绪数据已缓存: {cache_key}")
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")


# 全局实例
_sentiment_integrator = None

def get_china_stock_sentiment(stock_code: str, analysis_date: str, look_back_days: int = 7) -> Dict[str, Any]:
    """获取中国股票情绪分析的全局函数"""
    global _sentiment_integrator
    
    if _sentiment_integrator is None:
        _sentiment_integrator = ChineseSentimentIntegrator()
    
    return _sentiment_integrator.get_china_stock_sentiment(stock_code, analysis_date, look_back_days)


def test_sentiment_integrator():
    """测试情绪数据整合器"""
    integrator = ChineseSentimentIntegrator()
    
    # 测试几只股票
    test_stocks = ['000001', '000002', '300401']
    
    for stock_code in test_stocks:
        print(f"\n🧪 测试股票: {stock_code}")
        result = integrator.get_china_stock_sentiment(stock_code, datetime.now().strftime('%Y-%m-%d'))
        
        print(f"综合情绪: {result.get('integrated_sentiment', 0):.3f}")
        print(f"讨论总数: {result.get('total_discussions', 0)}")
        print(f"置信度: {result.get('confidence_level', '未知')}")
        print(f"摘要: {result.get('analysis_summary', '无')}")


if __name__ == "__main__":
    test_sentiment_integrator()