#!/usr/bin/env python3
"""
雪球股票讨论数据获取模块 - 合规版本
基于现有cookie配置获取雪球股票讨论数据
"""

import requests
import json
import time
import random
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from urllib.parse import quote

try:
    from xueqiu_config import get_xueqiu_cookie, validate_cookie_format
except ImportError:
    def get_xueqiu_cookie():
        return ""
    def validate_cookie_format(cookie):
        return False

logger = logging.getLogger(__name__)

class XueqiuAPI:
    """雪球股票讨论数据获取API"""
    
    def __init__(self):
        self.base_url = "https://xueqiu.com"
        self.api_base = "https://stock.xueqiu.com"
        self.session = requests.Session()
        
        # 缓存机制
        self._cache = {}
        self._cache_timeout = 1800  # 30分钟缓存
        
        # 获取cookie配置
        self.cookie = get_xueqiu_cookie()
        self.cookie_valid = validate_cookie_format(self.cookie)
        
        # 设置请求头
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://xueqiu.com/',
            'Origin': 'https://xueqiu.com',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-site',
        })
        
        # 设置cookie
        if self.cookie:
            self.session.headers['Cookie'] = self.cookie
        
        # 统计信息
        self.success_stocks = set()
        self.failed_stocks = set()
        
        logger.info(f"雪球API初始化，Cookie状态: {'有效' if self.cookie_valid else '无效或未配置'}")
    
    def get_stock_sentiment_summary(self, stock_code: str) -> Dict[str, Any]:
        """
        获取股票在雪球的讨论情绪汇总
        """
        if not self.cookie_valid:
            logger.warning(f"雪球: Cookie无效，跳过 {stock_code}")
            return self._create_no_data_result(stock_code, "Cookie配置无效")
        
        try:
            # 检查缓存
            cache_key = f"xueqiu_sentiment_{stock_code}"
            if cache_key in self._cache:
                cached_time, cached_data = self._cache[cache_key]
                if time.time() - cached_time < self._cache_timeout:
                    logger.info(f"雪球: 使用缓存数据 {stock_code}")
                    return cached_data
            
            # 获取股票讨论数据
            discussions = self._get_stock_discussions(stock_code)
            
            if discussions:
                # 分析讨论情绪
                result = self._analyze_discussions(discussions, stock_code)
                self._cache[cache_key] = (time.time(), result)
                self.success_stocks.add(stock_code)
                return result
            else:
                # 无讨论数据
                self.failed_stocks.add(stock_code)
                return self._create_no_data_result(stock_code, "暂无讨论数据")
                
        except Exception as e:
            logger.error(f"雪球: 获取{stock_code}数据时发生异常: {e}")
            self.failed_stocks.add(stock_code)
            return self._create_no_data_result(stock_code, f"获取失败: {str(e)}")
    
    def _get_stock_discussions(self, stock_code: str, limit: int = 20) -> List[Dict]:
        """
        获取股票讨论列表
        """
        try:
            # 转换股票代码格式
            symbol = self._convert_stock_code(stock_code)
            if not symbol:
                logger.warning(f"雪球: 无效的股票代码格式 {stock_code}")
                return []
            
            # API端点
            api_url = f"{self.api_base}/v5/stock/timeline/list.json"
            
            params = {
                'symbol': symbol,
                'count': limit,
                'source': 'all',
                '_': int(time.time() * 1000)  # 时间戳防缓存
            }
            
            logger.debug(f"雪球: 请求讨论数据 {symbol}")
            
            # 添加随机延迟
            time.sleep(random.uniform(1, 3))
            
            response = self.session.get(api_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('error_code') == 0 and 'list' in data:
                    discussions = []
                    for item in data['list']:
                        discussion = self._parse_discussion_item(item, stock_code)
                        if discussion:
                            discussions.append(discussion)
                    
                    logger.info(f"雪球: 获取到{len(discussions)}条讨论数据")
                    return discussions
                else:
                    error_msg = data.get('error_description', '未知错误')
                    logger.warning(f"雪球: API返回错误 {stock_code}: {error_msg}")
                    return []
            else:
                logger.warning(f"雪球: HTTP错误 {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"雪球: 获取讨论数据失败 {stock_code}: {e}")
            return []
    
    def _convert_stock_code(self, stock_code: str) -> str:
        """
        转换股票代码为雪球格式
        """
        clean_code = stock_code.replace('.SH', '').replace('.SZ', '')
        
        if clean_code.startswith('6'):
            return f"SH{clean_code}"
        elif clean_code.startswith(('0', '3')):
            return f"SZ{clean_code}"
        elif clean_code.startswith(('4', '8')):  # 新三板
            return f"OC{clean_code}"
        else:
            return f"SZ{clean_code}"  # 默认深圳
    
    def _parse_discussion_item(self, item: Dict, stock_code: str) -> Optional[Dict]:
        """
        解析单个讨论项目
        """
        try:
            # 提取基本信息
            title = item.get('title', '') or item.get('text', '')
            if not title or len(title) < 5:
                return None
            
            # 清理HTML标签
            title = re.sub(r'<[^>]+>', '', title).strip()
            if not title:
                return None
            
            # 提取其他信息
            user_info = item.get('user', {})
            created_at = item.get('created_at', 0)
            
            # 转换时间戳
            if created_at:
                created_time = datetime.fromtimestamp(created_at / 1000).strftime('%Y-%m-%d %H:%M:%S')
            else:
                created_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 情绪分析
            sentiment_score = self._simple_sentiment_analysis(title)
            
            return {
                'platform': '雪球',
                'stock_code': stock_code,
                'title': title,
                'content': title,  # 使用标题作为内容
                'author': user_info.get('screen_name', '匿名'),
                'created_time': created_time,
                'like_count': item.get('like_count', 0),
                'comment_count': item.get('reply_count', 0),
                'view_count': item.get('view_count', 0),
                'sentiment_score': sentiment_score
            }
            
        except Exception as e:
            logger.debug(f"雪球: 解析讨论项目失败: {e}")
            return None
    
    def _simple_sentiment_analysis(self, text: str) -> float:
        """
        简单的情绪分析
        """
        if not text:
            return 0.0
        
        positive_words = [
            '看好', '买入', '上涨', '涨', '利好', '推荐', '机会', '强势',
            '突破', '支撑', '反弹', '底部', '低估', '价值', '牛市',
            '优质', '成长', '潜力', '未来可期'
        ]
        
        negative_words = [
            '看空', '卖出', '下跌', '跌', '利空', '风险', '危险',
            '高估', '泡沫', '暴跌', '套牢', '割肉', '破位', '调整',
            '熊市', '垃圾', '退市', '爆雷'
        ]
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        return (positive_count - negative_count) / total
    
    def _analyze_discussions(self, discussions: List[Dict], stock_code: str) -> Dict[str, Any]:
        """
        分析讨论数据
        """
        if not discussions:
            return self._create_no_data_result(stock_code, "无讨论数据")
        
        # 情绪分析
        sentiments = [d['sentiment_score'] for d in discussions]
        avg_sentiment = sum(sentiments) / len(sentiments)
        
        positive = sum(1 for s in sentiments if s > 0.1)
        negative = sum(1 for s in sentiments if s < -0.1)
        neutral = len(sentiments) - positive - negative
        
        # 热门话题
        hot_topics = []
        sorted_discussions = sorted(discussions, 
                                  key=lambda x: x['like_count'] + x['comment_count'], 
                                  reverse=True)
        for disc in sorted_discussions[:5]:
            if disc['title']:
                topic = f"「{disc['title'][:50]}」"
                if disc['like_count'] or disc['comment_count']:
                    topic += f" (👍{disc['like_count']} 💬{disc['comment_count']})"
                hot_topics.append(topic)
        
        return {
            'stock_code': stock_code,
            'platform': '雪球',
            'total_posts': len(discussions),
            'avg_sentiment': round(avg_sentiment, 3),
            'sentiment_distribution': {
                'positive': positive,
                'neutral': neutral,
                'negative': negative
            },
            'hot_topics': hot_topics,
            'summary': self._generate_summary(avg_sentiment, positive, neutral, negative, len(discussions))
        }
    
    def _generate_summary(self, avg_sentiment: float, positive: int, 
                         neutral: int, negative: int, total: int) -> str:
        """
        生成情绪摘要
        """
        if avg_sentiment > 0.2:
            sentiment_label = "偏向看好"
        elif avg_sentiment > 0.05:
            sentiment_label = "略微看好"
        elif avg_sentiment < -0.2:
            sentiment_label = "偏向看空"
        elif avg_sentiment < -0.05:
            sentiment_label = "略微看空"
        else:
            sentiment_label = "情绪中性"
        
        pos_pct = round(positive / total * 100, 1)
        neg_pct = round(negative / total * 100, 1)
        neu_pct = round(neutral / total * 100, 1)
        
        return (f"基于{total}条雪球讨论分析，整体情绪{sentiment_label}。"
                f"看好占{pos_pct}%，中性占{neu_pct}%，看空占{neg_pct}%。"
                f"平均情绪指数{avg_sentiment:.3f}。")
    
    def _create_no_data_result(self, stock_code: str, reason: str) -> Dict[str, Any]:
        """
        创建无数据结果
        """
        return {
            'stock_code': stock_code,
            'platform': '雪球',
            'total_posts': 0,
            'avg_sentiment': 0.0,
            'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
            'hot_topics': [],
            'summary': f'暂无雪球讨论数据（{reason}）'
        }
    
    def get_success_rate(self) -> Dict[str, Any]:
        """
        获取数据获取成功率统计
        """
        total = len(self.success_stocks) + len(self.failed_stocks)
        if total == 0:
            return {'success_rate': 0, 'total_attempts': 0}
        
        return {
            'success_rate': len(self.success_stocks) / total,
            'successful_stocks': len(self.success_stocks),
            'failed_stocks': len(self.failed_stocks),
            'total_attempts': total,
            'success_list': list(self.success_stocks),
            'failed_list': list(self.failed_stocks)
        }


def test_xueqiu_api():
    """测试雪球API"""
    api = XueqiuAPI()
    
    test_stocks = ['000001', '600519', '000002', '300658']
    
    for stock_code in test_stocks:
        print(f"\n{'='*50}")
        print(f"测试股票: {stock_code}")
        print('='*50)
        
        result = api.get_stock_sentiment_summary(stock_code)
        
        print(f"讨论数量: {result['total_posts']}")
        print(f"情绪指数: {result['avg_sentiment']}")
        print(f"摘要: {result['summary']}")
        
        if result['hot_topics']:
            print("热门话题:")
            for topic in result['hot_topics'][:2]:
                print(f"  - {topic}")
        
        time.sleep(3)  # 避免请求过快
    
    # 显示统计信息
    stats = api.get_success_rate()
    print(f"\n雪球数据获取统计:")
    print(f"成功率: {stats['success_rate']:.1%}")
    print(f"成功获取: {stats['successful_stocks']}只")
    print(f"获取失败: {stats['failed_stocks']}只")
    if stats['success_list']:
        print(f"成功的股票: {stats['success_list']}")


if __name__ == "__main__":
    test_xueqiu_api()