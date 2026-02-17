#!/usr/bin/env python3
"""
中国市场情绪分析师Agent
专门处理中国社交媒体平台的特殊情况，包括水军识别和反话检测
"""

import re
import json
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta


class ChineseMarketSentimentAnalyst:
    """中国市场情绪分析师"""
    
    def __init__(self):
        # 水军检测关键词
        self.water_army_patterns = [
            r'[\u4e00-\u9fff]*老师[\u4e00-\u9fff]*推荐',  # 老师推荐
            r'[\u4e00-\u9fff]*群[\u4e00-\u9fff]*\d+',      # 群+数字
            r'[\u4e00-\u9fff]*微信[\u4e00-\u9fff]*',       # 微信相关
            r'[\u4e00-\u9fff]*QQ[\u4e00-\u9fff]*\d+',      # QQ+数字
            r'[\u4e00-\u9fff]*收费[\u4e00-\u9fff]*',       # 收费相关
            r'[\u4e00-\u9fff]*代客[\u4e00-\u9fff]*',       # 代客理财
            r'[\u4e00-\u9fff]*内幕[\u4e00-\u9fff]*',       # 内幕消息
            r'[\u4e00-\u9fff]*必涨[\u4e00-\u9fff]*',       # 必涨
            r'[\u4e00-\u9fff]*稳赚[\u4e00-\u9fff]*',       # 稳赚
            r'[\u4e00-\u9fff]*保底[\u4e00-\u9fff]*',       # 保底
            r'[\u4e00-\u9fff]*分成[\u4e00-\u9fff]*',       # 分成
        ]
        
        # 反话识别模式
        self.reverse_talk_patterns = [
            (r'[\u4e00-\u9fff]*不看好[\u4e00-\u9fff]*但[\u4e00-\u9fff]*', 'reverse'),  # 不看好但...
            (r'[\u4e00-\u9fff]*虽然[\u4e00-\u9fff]*不过[\u4e00-\u9fff]*', 'reverse'),  # 虽然...不过...
            (r'[\u4e00-\u9fff]*说[\u4e00-\u9fff]*跌[\u4e00-\u9fff]*涨[\u4e00-\u9fff]*', 'reverse'),  # 说跌就涨
            (r'[\u4e00-\u9fff]*反正[\u4e00-\u9fff]*', 'sarcasm'),  # 反正...
            (r'[\u4e00-\u9fff]*呵呵[\u4e00-\u9fff]*', 'sarcasm'),  # 呵呵
            (r'[\u4e00-\u9fff]*就是[\u4e00-\u9fff]*好[\u4e00-\u9fff]*', 'sarcasm'),  # 就是好(讽刺)
        ]
        
        # 真实情绪关键词
        self.genuine_sentiment_keywords = {
            'positive': [
                '基本面改善', '业绩超预期', '技术突破', '成交量放大', '主力资金', 
                '政策利好', '行业景气', '估值合理', '分红高', '成长性好',
                '盈利能力强', '现金流好', '毛利率提升', '市占率增加', '创新能力'
            ],
            'negative': [
                '业绩下滑', '亏损扩大', '资金流出', '技术破位', '估值过高',
                '政策风险', '行业下行', '竞争激烈', '成本上升', '债务压力',
                '现金流紧张', '市场份额下降', '监管风险', '商誉减值', '经营困难'
            ],
            'neutral': [
                '观望', '等待', '不确定', '谨慎', '平稳', '震荡', '整理',
                '横盘', '分歧', '争议', '待定', '研究中', '关注'
            ]
        }
        
        # 可信度评分权重
        self.credibility_weights = {
            'post_length': 0.1,      # 帖子长度
            'technical_content': 0.2, # 技术分析内容
            'data_reference': 0.25,   # 数据引用
            'logical_reasoning': 0.2, # 逻辑推理
            'interaction_quality': 0.15, # 互动质量
            'account_history': 0.1    # 账号历史
        }
    
    def analyze_post_credibility(self, post_content: str, post_info: Dict[str, Any]) -> float:
        """分析帖子可信度"""
        credibility_score = 0.0
        
        # 检测水军
        if self._is_water_army(post_content, post_info):
            return 0.1  # 水军帖子给极低分
        
        # 检测反话/讽刺
        reverse_type = self._detect_reverse_talk(post_content)
        if reverse_type:
            credibility_score *= 0.6  # 反话打折
        
        # 评估帖子长度 (50-500字较合理)
        length = len(post_content)
        if 50 <= length <= 500:
            credibility_score += self.credibility_weights['post_length']
        elif length > 500:
            credibility_score += self.credibility_weights['post_length'] * 0.8
        
        # 检查技术分析内容
        if self._has_technical_analysis(post_content):
            credibility_score += self.credibility_weights['technical_content']
        
        # 检查数据引用
        if self._has_data_reference(post_content):
            credibility_score += self.credibility_weights['data_reference']
        
        # 检查逻辑推理
        if self._has_logical_reasoning(post_content):
            credibility_score += self.credibility_weights['logical_reasoning']
        
        # 评估互动质量
        interaction_score = self._evaluate_interaction_quality(post_info)
        credibility_score += interaction_score * self.credibility_weights['interaction_quality']
        
        # 账号历史评估 (简化版)
        account_score = self._evaluate_account_history(post_info)
        credibility_score += account_score * self.credibility_weights['account_history']
        
        return min(1.0, max(0.0, credibility_score))
    
    def _is_water_army(self, content: str, post_info: Dict[str, Any]) -> bool:
        """检测是否为水军"""
        # 检查水军关键词
        for pattern in self.water_army_patterns:
            if re.search(pattern, content):
                return True
        
        # 检查发帖频率 (如果有数据)
        if 'post_frequency' in post_info and post_info['post_frequency'] > 20:  # 每日超过20帖
            return True
        
        # 检查内容重复度 (简化检查)
        if '复制' in content or '转发' in content:
            return True
        
        # 检查账号创建时间
        if 'account_age_days' in post_info and post_info['account_age_days'] < 30:
            return True
        
        return False
    
    def _detect_reverse_talk(self, content: str) -> str:
        """检测反话类型"""
        for pattern, talk_type in self.reverse_talk_patterns:
            if re.search(pattern, content):
                return talk_type
        return ''
    
    def _has_technical_analysis(self, content: str) -> bool:
        """检查是否包含技术分析"""
        technical_terms = [
            'MACD', 'KDJ', 'RSI', 'BOLL', '均线', '支撑', '阻力', '突破',
            '金叉', '死叉', '背离', '形态', '成交量', '换手率', '量价',
            'K线', '蜡烛图', '趋势线', '压力位', '支撑位'
        ]
        return any(term in content for term in technical_terms)
    
    def _has_data_reference(self, content: str) -> bool:
        """检查是否包含数据引用"""
        data_patterns = [
            r'\d+\.?\d*%',  # 百分比
            r'\d+\.?\d*元',  # 价格
            r'\d+\.?\d*亿',  # 金额
            r'\d+\.?\d*万',  # 金额
            r'\d{4}年\d{1,2}月',  # 日期
            r'财报', r'年报', r'季报', r'业绩', r'营收', r'净利润'
        ]
        return any(re.search(pattern, content) for pattern in data_patterns)
    
    def _has_logical_reasoning(self, content: str) -> bool:
        """检查是否包含逻辑推理"""
        reasoning_words = [
            '因为', '所以', '由于', '导致', '因此', '综合', '分析',
            '预计', '预期', '估计', '判断', '认为', '基于', '考虑到'
        ]
        return any(word in content for word in reasoning_words)
    
    def _evaluate_interaction_quality(self, post_info: Dict[str, Any]) -> float:
        """评估互动质量"""
        likes = post_info.get('likes', 0)
        comments = post_info.get('comments', 0)
        shares = post_info.get('shares', 0)
        
        # 简单的互动质量评分
        interaction_score = 0.0
        if likes > 5:
            interaction_score += 0.3
        if comments > 2:
            interaction_score += 0.4
        if shares > 1:
            interaction_score += 0.3
        
        return min(1.0, interaction_score)
    
    def _evaluate_account_history(self, post_info: Dict[str, Any]) -> float:
        """评估账号历史"""
        account_age = post_info.get('account_age_days', 0)
        follower_count = post_info.get('followers', 0)
        
        # 账号年龄评分
        age_score = min(1.0, account_age / 365.0)  # 1年以上满分
        
        # 粉丝数评分 (粉丝太多或太少都可疑)
        if 100 <= follower_count <= 10000:
            follower_score = 1.0
        elif follower_count > 10000:
            follower_score = 0.7  # 大V可能有偏向性
        else:
            follower_score = 0.3  # 新账号或僵尸粉
        
        return (age_score + follower_score) / 2
    
    def analyze_sentiment_with_filtering(self, posts_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析情绪并过滤水军和反话"""
        total_posts = len(posts_data)
        if total_posts == 0:
            return self._empty_sentiment_result()
        
        filtered_posts = []
        water_army_count = 0
        reverse_talk_count = 0
        
        for post in posts_data:
            content = post.get('content', '')
            
            # 计算可信度
            credibility = self.analyze_post_credibility(content, post)
            post['credibility'] = credibility
            
            # 过滤低可信度帖子
            if credibility < 0.3:
                if self._is_water_army(content, post):
                    water_army_count += 1
                continue
            
            # 检测反话
            reverse_type = self._detect_reverse_talk(content)
            if reverse_type:
                reverse_talk_count += 1
                post['reverse_type'] = reverse_type
                # 反转情绪标签
                if post.get('sentiment') == 'positive':
                    post['sentiment'] = 'negative'
                elif post.get('sentiment') == 'negative':
                    post['sentiment'] = 'positive'
            
            filtered_posts.append(post)
        
        # 计算加权情绪分数
        weighted_sentiment = self._calculate_weighted_sentiment(filtered_posts)
        
        # 生成分析报告
        return {
            'total_posts': total_posts,
            'filtered_posts': len(filtered_posts),
            'water_army_detected': water_army_count,
            'reverse_talk_detected': reverse_talk_count,
            'filter_rate': (total_posts - len(filtered_posts)) / total_posts if total_posts > 0 else 0,
            'weighted_sentiment': weighted_sentiment,
            'sentiment_distribution': self._calculate_sentiment_distribution(filtered_posts),
            'confidence_level': self._calculate_confidence_level(filtered_posts),
            'genuine_posts': filtered_posts,
            'analysis_summary': self._generate_analysis_summary(
                total_posts, len(filtered_posts), water_army_count, reverse_talk_count, weighted_sentiment
            )
        }
    
    def _empty_sentiment_result(self) -> Dict[str, Any]:
        """空结果"""
        return {
            'total_posts': 0,
            'filtered_posts': 0,
            'water_army_detected': 0,
            'reverse_talk_detected': 0,
            'filter_rate': 0,
            'weighted_sentiment': 0,
            'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
            'confidence_level': 'low',
            'genuine_posts': [],
            'analysis_summary': '暂无讨论数据'
        }
    
    def _calculate_weighted_sentiment(self, posts: List[Dict[str, Any]]) -> float:
        """计算加权情绪分数"""
        if not posts:
            return 0.0
        
        total_weight = 0
        weighted_sum = 0
        
        for post in posts:
            credibility = post.get('credibility', 0.5)
            sentiment = post.get('sentiment', 'neutral')
            
            # 情绪转数值
            if sentiment == 'positive':
                sentiment_score = 1.0
            elif sentiment == 'negative':
                sentiment_score = -1.0
            else:
                sentiment_score = 0.0
            
            # 加权计算
            weight = credibility
            weighted_sum += sentiment_score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _calculate_sentiment_distribution(self, posts: List[Dict[str, Any]]) -> Dict[str, int]:
        """计算情绪分布"""
        distribution = {'positive': 0, 'neutral': 0, 'negative': 0}
        
        for post in posts:
            sentiment = post.get('sentiment', 'neutral')
            if sentiment in distribution:
                distribution[sentiment] += 1
        
        return distribution
    
    def _calculate_confidence_level(self, posts: List[Dict[str, Any]]) -> str:
        """计算置信度等级"""
        if not posts:
            return 'low'
        
        avg_credibility = sum(post.get('credibility', 0) for post in posts) / len(posts)
        post_count = len(posts)
        
        if avg_credibility > 0.7 and post_count >= 20:
            return 'high'
        elif avg_credibility > 0.5 and post_count >= 10:
            return 'medium'
        else:
            return 'low'
    
    def _generate_analysis_summary(self, total: int, filtered: int, water_army: int, 
                                 reverse_talk: int, sentiment: float) -> str:
        """生成分析摘要"""
        filter_rate = (total - filtered) / total * 100 if total > 0 else 0
        
        sentiment_label = "中性"
        if sentiment > 0.2:
            sentiment_label = "偏乐观"
        elif sentiment > 0.4:
            sentiment_label = "乐观"
        elif sentiment < -0.2:
            sentiment_label = "偏悲观"
        elif sentiment < -0.4:
            sentiment_label = "悲观"
        
        summary = f"共分析{total}条讨论，过滤{total-filtered}条低质量内容（过滤率{filter_rate:.1f}%）。"
        summary += f"其中检测到{water_army}条疑似水军，{reverse_talk}条反话表达。"
        summary += f"基于{filtered}条高质量讨论，市场情绪呈{sentiment_label}态势（{sentiment:.2f}）。"
        
        return summary


def create_chinese_sentiment_analyst() -> ChineseMarketSentimentAnalyst:
    """创建中国市场情绪分析师实例"""
    return ChineseMarketSentimentAnalyst()


if __name__ == "__main__":
    # 测试中国市场情绪分析师
    analyst = ChineseMarketSentimentAnalyst()
    
    # 模拟测试数据
    test_posts = [
        {
            'content': '这只股票基本面不错，MACD即将金叉，建议关注',
            'sentiment': 'positive',
            'likes': 10,
            'comments': 5,
            'account_age_days': 365,
            'followers': 500
        },
        {
            'content': '老师推荐的股票，必涨无疑，微信群999',
            'sentiment': 'positive',
            'likes': 2,
            'comments': 0,
            'account_age_days': 15,
            'followers': 10
        },
        {
            'content': '说跌就涨，呵呵，反正我是不信',
            'sentiment': 'negative',
            'likes': 8,
            'comments': 3,
            'account_age_days': 500,
            'followers': 1000
        }
    ]
    
    result = analyst.analyze_sentiment_with_filtering(test_posts)
    print("🤖 中国市场情绪分析结果:")
    print(f"总帖子数: {result['total_posts']}")
    print(f"过滤后: {result['filtered_posts']}")
    print(f"水军检测: {result['water_army_detected']}条")
    print(f"反话检测: {result['reverse_talk_detected']}条")
    print(f"加权情绪: {result['weighted_sentiment']:.3f}")
    print(f"置信度: {result['confidence_level']}")
    print(f"分析摘要: {result['analysis_summary']}")