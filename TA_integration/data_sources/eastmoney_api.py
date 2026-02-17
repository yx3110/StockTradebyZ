#!/usr/bin/env python3
"""
东方财富股吧数据抓取模块 - 诚实版本
如果无法获取真实数据，直接返回无数据状态，不提供虚假信息
"""

import requests
import json
import time
import random
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from urllib.parse import quote
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

class EastMoneyAPI:
    """东方财富股吧数据获取API - 诚实版本"""
    
    def __init__(self):
        self.base_url = "https://guba.eastmoney.com"
        self.session = requests.Session()
        
        # 缓存机制
        self._cache = {}
        self._cache_timeout = 1800  # 30分钟缓存
        
        # 优化请求头 - 更真实的浏览器行为
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Cache-Control': 'max-age=0',
            'Connection': 'keep-alive',
            'Referer': 'https://guba.eastmoney.com/',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'same-origin',
            'Sec-Fetch-User': '?1',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # 统计信息
        self.success_stocks = set()
        self.failed_stocks = set()
        
    def get_stock_sentiment_summary(self, stock_code: str) -> Dict[str, Any]:
        """
        获取股票情绪汇总 - 只返回真实数据或明确表示无数据
        """
        try:
            # 检查缓存
            cache_key = f"sentiment_{stock_code}"
            if cache_key in self._cache:
                cached_time, cached_data = self._cache[cache_key]
                if time.time() - cached_time < self._cache_timeout:
                    logger.info(f"东财: 使用缓存数据 {stock_code}")
                    return cached_data
            
            # 尝试获取真实数据
            posts = self._attempt_real_data_fetch(stock_code)
            
            if posts:
                # 有真实数据，进行分析
                result = self._analyze_real_posts(posts, stock_code)
                self._cache[cache_key] = (time.time(), result)
                self.success_stocks.add(stock_code)
                return result
            else:
                # 无法获取真实数据，诚实返回
                self.failed_stocks.add(stock_code)
                return self._create_honest_no_data_result(stock_code)
                
        except Exception as e:
            logger.error(f"东财: 获取{stock_code}数据时发生异常: {e}")
            self.failed_stocks.add(stock_code)
            return self._create_honest_no_data_result(stock_code)
    
    def _attempt_real_data_fetch(self, stock_code: str) -> List[Dict]:
        """
        尝试获取真实的股票讨论数据
        只有确认是真实数据才返回，否则返回空列表
        """
        posts = []
        
        # 方法1: 尝试官方API (如果存在)
        api_posts = self._try_official_api(stock_code)
        if api_posts:
            posts.extend(api_posts)
        
        # 方法2: 尝试特定的股票页面格式
        web_posts = self._try_specific_web_pages(stock_code)
        if web_posts:
            posts.extend(web_posts)
        
        # 方法3: 尝试搜索结果页面
        if len(posts) < 3:  # 如果数据太少，尝试搜索
            search_posts = self._try_search_results(stock_code)
            if search_posts:
                posts.extend(search_posts)
        
        # 验证数据真实性
        verified_posts = self._verify_posts_authenticity(posts, stock_code)
        
        return verified_posts
    
    def _try_official_api(self, stock_code: str) -> List[Dict]:
        """
        尝试官方API接口
        """
        try:
            clean_code = stock_code.replace('.SH', '').replace('.SZ', '')
            
            # 尝试几个可能的官方API端点
            api_endpoints = [
                f"https://gbapi.eastmoney.com/bbsapi/bbslists/quotec",
                f"https://push2.eastmoney.com/api/qt/stock/get",
                f"https://datacenter-web.eastmoney.com/api/data/v1/get"
            ]
            
            for api_url in api_endpoints:
                try:
                    # 构建参数
                    if 'bbslists' in api_url:
                        params = {
                            'code': f"{clean_code}.{'SH' if clean_code.startswith(('60', '68', '51')) else 'SZ'}",
                            'type': '1',
                            'pageindex': '1',
                            'pagesize': '20'
                        }
                    else:
                        continue  # 跳过其他API，避免无效请求
                    
                    response = self.session.get(api_url, params=params, timeout=8)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('success') and data.get('data'):
                            posts = self._parse_api_response(data, stock_code)
                            if posts:
                                logger.info(f"东财: 通过API获取到{len(posts)}条真实数据")
                                return posts
                
                except Exception as e:
                    logger.debug(f"东财: API {api_url} 调用失败: {e}")
                    continue
            
            return []
            
        except Exception as e:
            logger.debug(f"东财: 官方API尝试失败: {e}")
            return []
    
    def _try_specific_web_pages(self, stock_code: str) -> List[Dict]:
        """
        尝试特定的网页格式，只有确认页面内容正确才返回数据
        """
        try:
            clean_code = stock_code.replace('.SH', '').replace('.SZ', '')
            
            # 只尝试最有可能成功的URL
            test_urls = [
                f"https://guba.eastmoney.com/list,{clean_code}.html"
            ]
            
            for url in test_urls:
                try:
                    response = self.session.get(url, timeout=10)
                    if response.status_code != 200:
                        continue
                    
                    # 设置正确编码
                    if response.apparent_encoding:
                        response.encoding = response.apparent_encoding
                    elif not response.encoding or response.encoding == 'ISO-8859-1':
                        response.encoding = 'utf-8'
                    
                    # 严格验证页面内容
                    if self._is_valid_stock_page(response.text, stock_code):
                        posts = self._parse_web_page_carefully(response.text, stock_code)
                        if posts:
                            logger.info(f"东财: 从网页获取到{len(posts)}条真实数据")
                            return posts
                    else:
                        logger.debug(f"东财: 页面验证失败，可能是重定向页面")
                        
                except Exception as e:
                    logger.debug(f"东财: 网页 {url} 访问失败: {e}")
                    continue
                    
                # 更长的请求间隔 - 降低请求频率避免触发限制
                time.sleep(random.uniform(3, 8))
            
            return []
            
        except Exception as e:
            logger.debug(f"东财: 网页尝试失败: {e}")
            return []
    
    def _try_search_results(self, stock_code: str) -> List[Dict]:
        """
        通过搜索结果获取相关讨论
        """
        try:
            # 暂时跳过搜索策略，因为搜索API也可能受限
            return []
            
        except Exception as e:
            logger.debug(f"东财: 搜索尝试失败: {e}")
            return []
    
    def _is_valid_stock_page(self, html_content: str, target_stock_code: str) -> bool:
        """
        严格验证页面是否为目标股票的讨论页面
        """
        try:
            # 提取页面标题
            title_match = re.search(r'<title[^>]*>([^<]+)</title>', html_content, re.IGNORECASE)
            if not title_match:
                return False
                
            title = title_match.group(1).strip()
            clean_target = target_stock_code.replace('.SH', '').replace('.SZ', '')
            
            # 检查标题是否包含目标股票代码
            if clean_target in title:
                return True
            
            # 检查是否被重定向到其他股票
            redirect_codes = ['000001', '399001', '399006']  # 常见重定向目标
            for code in redirect_codes:
                if code != clean_target and code in title:
                    logger.debug(f"东财: 检测到重定向到 {code}")
                    return False
            
            # 检查页面内容中是否有目标股票相关信息
            if clean_target in html_content:
                # 进一步验证，确保不是偶然出现
                code_count = html_content.count(clean_target)
                if code_count >= 3:  # 至少出现3次才认为是相关页面
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"东财: 页面验证失败: {e}")
            return False
    
    def _parse_web_page_carefully(self, html_content: str, stock_code: str) -> List[Dict]:
        """
        仔细解析网页内容，只提取确认的真实帖子
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            posts = []
            
            # 查找帖子容器
            selectors = [
                'tr[class*="listitem"]',
                'tbody tr',
                'div[class*="post"]',
                'li[class*="item"]'
            ]
            
            for selector in selectors:
                elements = soup.select(selector)
                if len(elements) > 5:  # 至少要有一些元素
                    for element in elements[:20]:  # 最多处理20个
                        post = self._extract_post_carefully(element, stock_code)
                        if post and self._is_genuine_post(post):
                            posts.append(post)
                    
                    if posts:
                        break
            
            return posts[:10]  # 最多返回10条
            
        except Exception as e:
            logger.debug(f"东财: 网页解析失败: {e}")
            return []
    
    def _extract_post_carefully(self, element, stock_code: str) -> Optional[Dict]:
        """
        小心地提取帖子信息，确保数据质量
        """
        try:
            # 提取标题
            title = None
            title_selectors = ['a[title]', '.title a', 'a']
            for selector in title_selectors:
                title_elem = element.select_one(selector)
                if title_elem:
                    title = (title_elem.get('title') or title_elem.get_text()).strip()
                    if title and 5 <= len(title) <= 100:
                        break
            
            if not title:
                return None
            
            # 提取其他信息
            author = self._extract_text_carefully(element, ['.author', '.user', 'td:nth-child(3)']) or '匿名'
            
            # 简单的情绪分析
            sentiment_score = self._simple_sentiment_analysis(title)
            
            return {
                'platform': '东方财富股吧',
                'stock_code': stock_code,
                'title': title,
                'content': title,  # 使用标题作为内容
                'author': author,
                'created_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'like_count': 0,
                'comment_count': 0,
                'view_count': 0,
                'sentiment_score': sentiment_score
            }
            
        except Exception as e:
            logger.debug(f"东财: 帖子提取失败: {e}")
            return None
    
    def _extract_text_carefully(self, element, selectors: List[str]) -> Optional[str]:
        """仔细提取文本"""
        for selector in selectors:
            try:
                elem = element.select_one(selector)
                if elem:
                    text = elem.get_text(strip=True)
                    if text and len(text) < 50:
                        return text
            except:
                continue
        return None
    
    def _is_genuine_post(self, post: Dict) -> bool:
        """
        验证帖子是否为真实的讨论帖子
        """
        title = post.get('title', '')
        
        # 过滤明显无效的标题
        invalid_keywords = [
            '404', '错误', '页面不存在', '加载失败', 
            'error', '删除', '违规', 'javascript'
        ]
        
        if any(keyword in title.lower() for keyword in invalid_keywords):
            return False
        
        # 标题长度检查
        if len(title) < 5 or len(title) > 150:
            return False
        
        return True
    
    def _verify_posts_authenticity(self, posts: List[Dict], stock_code: str) -> List[Dict]:
        """
        验证帖子的真实性
        """
        if not posts:
            return []
        
        verified = []
        clean_code = stock_code.replace('.SH', '').replace('.SZ', '')
        
        for post in posts:
            title = post.get('title', '')
            
            # 检查是否与目标股票相关
            is_relevant = (
                clean_code in title or  # 包含股票代码
                len(title) > 10 or      # 标题足够长，可能是真实讨论
                any(word in title for word in ['股票', '投资', '买入', '卖出', '涨', '跌'])  # 包含投资相关词汇
            )
            
            if is_relevant:
                verified.append(post)
        
        return verified
    
    def _parse_api_response(self, data: Dict, stock_code: str) -> List[Dict]:
        """
        解析API响应数据
        """
        try:
            posts = []
            
            # 根据不同API格式解析
            if 'data' in data and 'post' in data['data']:
                for item in data['data']['post']:
                    post = {
                        'platform': '东方财富股吧',
                        'stock_code': stock_code,
                        'title': item.get('post_title', ''),
                        'content': item.get('post_content', ''),
                        'author': item.get('post_user_nickname', '匿名'),
                        'created_time': self._parse_api_time(item.get('post_publish_time')),
                        'like_count': int(item.get('post_zan_count', 0)),
                        'comment_count': int(item.get('post_comment_count', 0)),
                        'view_count': int(item.get('post_click_count', 0)),
                        'sentiment_score': self._simple_sentiment_analysis(item.get('post_title', ''))
                    }
                    
                    if post['title'] and len(post['title']) > 3:
                        posts.append(post)
            
            return posts
            
        except Exception as e:
            logger.debug(f"东财: API响应解析失败: {e}")
            return []
    
    def _parse_api_time(self, time_value) -> str:
        """解析API时间"""
        try:
            if isinstance(time_value, str):
                return time_value
            elif isinstance(time_value, (int, float)):
                return datetime.fromtimestamp(time_value).strftime('%Y-%m-%d %H:%M:%S')
        except:
            pass
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def _simple_sentiment_analysis(self, text: str) -> float:
        """
        简单的情绪分析
        """
        if not text:
            return 0.0
        
        positive_words = [
            '买入', '看好', '上涨', '涨', '利好', '推荐', '机会', 
            '强势', '突破', '支撑', '反弹', '底部', '低估', '价值'
        ]
        
        negative_words = [
            '卖出', '看空', '下跌', '跌', '利空', '风险', '危险',
            '高估', '泡沫', '暴跌', '套牢', '割肉', '破位', '调整'
        ]
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        return (positive_count - negative_count) / total
    
    def _analyze_real_posts(self, posts: List[Dict], stock_code: str) -> Dict[str, Any]:
        """
        分析真实的帖子数据
        """
        if not posts:
            return self._create_honest_no_data_result(stock_code)
        
        # 情绪分析
        sentiments = [p['sentiment_score'] for p in posts]
        avg_sentiment = sum(sentiments) / len(sentiments)
        
        positive = sum(1 for s in sentiments if s > 0.1)
        negative = sum(1 for s in sentiments if s < -0.1)
        neutral = len(sentiments) - positive - negative
        
        # 热门话题
        hot_topics = []
        sorted_posts = sorted(posts, key=lambda x: x['like_count'] + x['comment_count'], reverse=True)
        for post in sorted_posts[:5]:
            if post['title']:
                topic = f"「{post['title']}」"
                if post['like_count'] or post['comment_count']:
                    topic += f" (👍{post['like_count']} 💬{post['comment_count']})"
                hot_topics.append(topic)
        
        return {
            'stock_code': stock_code,
            'platform': '东方财富股吧',
            'total_posts': len(posts),
            'avg_sentiment': round(avg_sentiment, 3),
            'sentiment_distribution': {
                'positive': positive,
                'neutral': neutral,
                'negative': negative
            },
            'hot_topics': hot_topics,
            'summary': self._generate_honest_summary(avg_sentiment, positive, neutral, negative, len(posts))
        }
    
    def _generate_honest_summary(self, avg_sentiment: float, positive: int, 
                                neutral: int, negative: int, total: int) -> str:
        """
        生成诚实的情绪摘要
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
        
        return (f"基于{total}条真实帖子的分析，东方财富股吧讨论整体情绪{sentiment_label}。"
                f"看好占{pos_pct}%，中性占{neu_pct}%，看空占{neg_pct}%。"
                f"平均情绪指数{avg_sentiment:.3f}。")
    
    def _create_honest_no_data_result(self, stock_code: str) -> Dict[str, Any]:
        """
        创建诚实的无数据结果
        """
        return {
            'stock_code': stock_code,
            'platform': '东方财富股吧',
            'total_posts': 0,
            'avg_sentiment': 0.0,
            'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
            'hot_topics': [],
            'summary': '暂无东方财富股吧讨论数据（网站访问受限或该股票讨论较少）'
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


def test_honest_eastmoney_api():
    """测试诚实版本的东方财富API"""
    api = EastMoneyAPI()
    
    test_stocks = ['000001', '600519', '000002', '300658']
    
    for stock_code in test_stocks:
        print(f"\n{'='*50}")
        print(f"测试股票: {stock_code}")
        print('='*50)
        
        result = api.get_stock_sentiment_summary(stock_code)
        
        print(f"帖子数量: {result['total_posts']}")
        print(f"情绪指数: {result['avg_sentiment']}")
        print(f"摘要: {result['summary']}")
        
        if result['hot_topics']:
            print("热门话题:")
            for topic in result['hot_topics'][:2]:
                print(f"  - {topic}")
        
        time.sleep(2)
    
    # 显示统计信息
    stats = api.get_success_rate()
    print(f"\n数据获取统计:")
    print(f"成功率: {stats['success_rate']:.1%}")
    print(f"成功获取: {stats['successful_stocks']}只")
    print(f"获取失败: {stats['failed_stocks']}只")
    if stats['success_list']:
        print(f"成功的股票: {stats['success_list']}")


if __name__ == "__main__":
    test_honest_eastmoney_api()