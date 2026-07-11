#!/usr/bin/env python3
"""
纯Tushare新闻获取器
仅使用Tushare Pro官方接口获取真实的财经数据，不包含任何本地编造内容
"""

import tushare as ts
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import time
import re

logger = logging.getLogger(__name__)

class PureTushareNewsFetcher:
    """纯Tushare Pro新闻获取器 - 只使用真实API数据"""
    
    def __init__(self, config_path: str = "config.json"):
        """初始化Tushare新闻获取器"""
        self.config_path = config_path
        self._load_config()
        self._init_tushare()
        
        # API调用间隔
        self.api_delay = self.config.get('tushare', {}).get('rate_limit_delay', 0.3)
        self.last_api_call = 0
    
    def _load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
    
    def _init_tushare(self):
        """初始化Tushare Pro API (token 优先从 core.config/.env 获取)"""
        try:
            try:
                from core.config import get_tushare_token
                token = get_tushare_token()
            except ImportError:
                token = self.config['tushare']['token']
            ts.set_token(token)
            self.pro = ts.pro_api()
            logger.info("Tushare Pro API初始化成功")
        except Exception as e:
            logger.error(f"Tushare Pro API初始化失败: {e}")
            raise
    
    def _rate_limit(self):
        """API调用频率限制"""
        current_time = time.time()
        elapsed = current_time - self.last_api_call
        
        if elapsed < self.api_delay:
            time.sleep(self.api_delay - elapsed)
        
        self.last_api_call = time.time()
    
    def get_stock_news(self, stock_code: str, stock_name: str, days: int = 15) -> Dict[str, any]:
        """获取股票相关新闻 - 仅使用真实Tushare数据
        
        Args:
            stock_code: 股票代码，如 '000001'
            stock_name: 股票名称，如 '平安银行'
            days: 获取最近几天的新闻，默认15天
            
        Returns:
            包含新闻数据的字典
        """
        try:
            logger.info(f"开始通过Tushare获取{stock_code} {stock_name}的真实新闻数据")
            
            # 计算日期范围
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            all_news = []
            success_sources = []
            
            # 1. 获取真实新闻数据（使用news接口）
            news_items = self._fetch_real_news(start_date, end_date)
            if news_items:
                # 过滤与股票相关的新闻
                relevant_news = self._filter_stock_relevant_news(news_items, stock_code, stock_name)
                all_news.extend(relevant_news)
                if relevant_news:
                    success_sources.append('Tushare真实新闻')
            
            # 2. 获取财经日历事件
            eco_events = self._fetch_economic_calendar(start_date, end_date)
            if eco_events:
                all_news.extend(eco_events)
                success_sources.append('Tushare财经日历')
            
            # 3. 获取披露日期信息
            disclosure_info = self._fetch_disclosure_dates(stock_code)
            if disclosure_info:
                all_news.extend(disclosure_info)
                success_sources.append('Tushare披露日期')
            
            # 4. 获取股票基础信息上下文
            stock_context = self._fetch_stock_context(stock_code, stock_name)
            if stock_context:
                all_news.extend(stock_context)
                success_sources.append('Tushare股票信息')
            
            # 去重和排序
            unique_news = self._deduplicate_news(all_news)
            sorted_news = sorted(unique_news, key=lambda x: self._parse_datetime(x.get('publish_time', '')), reverse=True)
            
            # 限制数量
            final_news = sorted_news[:25]
            
            result = {
                'stock_code': stock_code,
                'stock_name': stock_name,
                'news_count': len(final_news),
                'success_sources': success_sources,
                'date_range': f"{start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}",
                'news_items': final_news,
                'summary': self._generate_news_summary(final_news, stock_name),
                'data_source': 'Tushare Pro API - 100%真实数据'
            }
            
            logger.info(f"成功通过Tushare获取{stock_code}真实新闻，共{len(final_news)}条，来源：{', '.join(success_sources)}")
            return result
            
        except Exception as e:
            logger.error(f"通过Tushare获取{stock_code}新闻失败: {e}")
            return {
                'stock_code': stock_code,
                'stock_name': stock_name,
                'news_count': 0,
                'success_sources': [],
                'news_items': [],
                'summary': 'Tushare新闻获取失败',
                'error': str(e),
                'data_source': 'Tushare Pro API'
            }
    
    def _fetch_real_news(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """获取真实新闻数据"""
        try:
            self._rate_limit()
            
            # 使用Tushare news接口获取真实新闻
            start_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
            end_str = end_date.strftime('%Y-%m-%d %H:%M:%S')
            
            # 获取最近的新闻数据
            news_df = self.pro.news(src='sina', start_date=start_str, end_date=end_str)
            
            if news_df.empty:
                logger.debug("未获取到Tushare真实新闻数据")
                return []
            
            news_items = []
            for _, row in news_df.iterrows():
                try:
                    # 处理真实新闻数据
                    datetime_str = str(row['datetime']) if pd.notna(row['datetime']) else ''
                    title = str(row.get('title', '')) if pd.notna(row.get('title')) else ''
                    content = str(row.get('content', '')) if pd.notna(row.get('content')) else ''
                    
                    # 如果没有标题，使用内容前50字符作为标题
                    if not title and content:
                        title = content[:50] + ('...' if len(content) > 50 else '')
                    elif not title and not content:
                        continue
                    
                    # 确保有有效内容
                    if not content:
                        content = title
                    
                    news_item = {
                        'title': title,
                        'content': content,
                        'publish_time': datetime_str,
                        'source': 'Tushare真实新闻',
                        'url': '',
                        'type': 'real_news',
                        'relevance_score': 1  # 将由过滤函数重新计算
                    }
                    news_items.append(news_item)
                
                except Exception as e:
                    logger.debug(f"解析Tushare真实新闻条目失败: {e}")
                    continue
            
            logger.info(f"从Tushare获取到{len(news_items)}条真实新闻数据")
            return news_items
            
        except Exception as e:
            logger.debug(f"获取Tushare真实新闻失败: {e}")
            return []
    
    def _fetch_economic_calendar(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """获取财经日历事件"""
        try:
            self._rate_limit()
            
            # 获取日期范围内的财经日历
            date_list = []
            current_date = start_date
            while current_date <= end_date:
                date_list.append(current_date.strftime('%Y%m%d'))
                current_date += timedelta(days=1)
            
            all_events = []
            for date_str in date_list[-7:]:  # 只获取最近7天的数据，避免API调用过多
                try:
                    self._rate_limit()
                    eco_df = self.pro.eco_cal(date=date_str)
                    
                    if not eco_df.empty:
                        for _, row in eco_df.iterrows():
                            try:
                                event_date = str(row['date'])
                                event_time = str(row['time']) if pd.notna(row['time']) else '00:00'
                                event_name = str(row['event']) if pd.notna(row['event']) else ''
                                country = str(row['country']) if pd.notna(row['country']) else ''
                                value = str(row['value']) if pd.notna(row['value']) else ''
                                
                                if event_name:
                                    # 格式化发布时间
                                    try:
                                        if len(event_time) == 5:  # HH:MM格式
                                            pub_time = datetime.strptime(f"{event_date} {event_time}", '%Y%m%d %H:%M').strftime('%Y-%m-%d %H:%M:%S')
                                        else:
                                            pub_time = datetime.strptime(event_date, '%Y%m%d').strftime('%Y-%m-%d %H:%M:%S')
                                    except Exception:
                                        pub_time = datetime.strptime(event_date, '%Y%m%d').strftime('%Y-%m-%d %H:%M:%S')
                                    
                                    # 生成财经日历新闻
                                    title = f"{country}{event_name}" if country else event_name
                                    content = f"财经日历：{event_name}"
                                    if value and value != 'nan':
                                        content += f"，公布值：{value}"
                                    
                                    news_item = {
                                        'title': title,
                                        'content': content,
                                        'publish_time': pub_time,
                                        'source': 'Tushare财经日历',
                                        'url': 'https://tushare.pro/document/2?doc_id=233',
                                        'type': 'economic_calendar',
                                        'relevance_score': 3
                                    }
                                    all_events.append(news_item)
                            
                            except Exception as e:
                                logger.debug(f"解析财经日历事件失败: {e}")
                                continue
                
                except Exception as e:
                    logger.debug(f"获取{date_str}财经日历失败: {e}")
                    continue
            
            logger.info(f"从Tushare获取到{len(all_events)}个财经日历事件")
            return all_events[:10]  # 限制财经日历事件数量
            
        except Exception as e:
            logger.debug(f"获取Tushare财经日历失败: {e}")
            return []
    
    def _fetch_disclosure_dates(self, stock_code: str) -> List[Dict]:
        """获取披露日期信息"""
        try:
            self._rate_limit()
            
            # 获取最近的披露日期信息
            ts_code = self._convert_to_ts_code(stock_code)
            disc_df = self.pro.disclosure_date(ts_code=ts_code)
            
            if disc_df.empty:
                return []
            
            disclosure_news = []
            for _, row in disc_df.head(3).iterrows():  # 只取前3条
                try:
                    ts_code_val = str(row['ts_code']) if pd.notna(row['ts_code']) else ''
                    end_date = str(row['end_date']) if pd.notna(row['end_date']) else ''
                    pre_date = str(row['pre_date']) if pd.notna(row['pre_date']) else ''
                    actual_date = str(row['actual_date']) if pd.notna(row['actual_date']) else ''
                    
                    if ts_code_val and end_date:
                        # 格式化日期
                        try:
                            end_date_formatted = datetime.strptime(end_date, '%Y%m%d').strftime('%Y年%m月%d日')
                            period = end_date_formatted[:7]  # 2024年12月 -> 2024年12
                        except Exception:
                            end_date_formatted = end_date
                            period = end_date
                        
                        title = f"{ts_code_val}财务报告披露计划"
                        content = f"报告期：{end_date_formatted}"
                        
                        if pre_date and pre_date != 'nan':
                            try:
                                pre_date_formatted = datetime.strptime(pre_date, '%Y%m%d').strftime('%Y年%m月%d日')
                                content += f"，预计披露日期：{pre_date_formatted}"
                            except Exception:
                                content += f"，预计披露日期：{pre_date}"
                        
                        if actual_date and actual_date != 'nan':
                            try:
                                actual_date_formatted = datetime.strptime(actual_date, '%Y%m%d').strftime('%Y年%m月%d日')
                                content += f"，实际披露日期：{actual_date_formatted}"
                            except Exception:
                                content += f"，实际披露日期：{actual_date}"
                        
                        news_item = {
                            'title': title,
                            'content': content,
                            'publish_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'source': 'Tushare披露日期',
                            'url': 'https://tushare.pro/document/2?doc_id=162',
                            'type': 'disclosure_schedule',
                            'relevance_score': 8
                        }
                        disclosure_news.append(news_item)
                
                except Exception as e:
                    logger.debug(f"处理披露日期信息失败: {e}")
                    continue
            
            logger.info(f"生成了{len(disclosure_news)}条披露日期信息")
            return disclosure_news
            
        except Exception as e:
            logger.debug(f"获取披露日期信息失败: {e}")
            return []
    
    def _fetch_stock_context(self, stock_code: str, stock_name: str) -> List[Dict]:
        """获取股票基础信息上下文"""
        try:
            self._rate_limit()
            
            ts_code = self._convert_to_ts_code(stock_code)
            basic_df = self.pro.stock_basic(ts_code=ts_code, fields='ts_code,symbol,name,area,industry,market,list_date')
            
            if basic_df.empty:
                return []
            
            context_news = []
            for _, row in basic_df.iterrows():
                try:
                    industry = str(row['industry']) if pd.notna(row['industry']) else ''
                    area = str(row['area']) if pd.notna(row['area']) else ''
                    market = str(row['market']) if pd.notna(row['market']) else ''
                    list_date = str(row['list_date']) if pd.notna(row['list_date']) else ''
                    
                    if industry:
                        # 格式化上市日期
                        try:
                            list_date_formatted = datetime.strptime(list_date, '%Y%m%d').strftime('%Y年%m月%d日')
                        except Exception:
                            list_date_formatted = list_date
                        
                        # 生成股票基础信息
                        title = f"{stock_name}基本信息"
                        content = f"{stock_name}属于{industry}行业，注册地在{area}"
                        if market:
                            content += f"，在{market}市场交易"
                        if list_date_formatted:
                            content += f"，于{list_date_formatted}上市"
                        content += "。"
                        
                        context_item = {
                            'title': title,
                            'content': content,
                            'publish_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'source': 'Tushare股票信息',
                            'url': 'https://tushare.pro/document/2?doc_id=25',
                            'type': 'stock_context',
                            'relevance_score': 5
                        }
                        context_news.append(context_item)
                
                except Exception as e:
                    logger.debug(f"处理股票基础信息失败: {e}")
                    continue
            
            logger.info(f"生成了{len(context_news)}条股票基础信息")
            return context_news
            
        except Exception as e:
            logger.debug(f"获取股票基础信息失败: {e}")
            return []
    
    def _filter_stock_relevant_news(self, news_list: List[Dict], stock_code: str, stock_name: str) -> List[Dict]:
        """过滤与股票相关的新闻"""
        relevant_news = []
        
        # 扩展关键词匹配
        keywords = [
            stock_name, stock_code,
            '银行', '保险', '证券', '基金', '信托',  # 金融行业
            '房地产', '建筑', '钢铁', '煤炭', '电力',  # 传统行业
            '科技', '互联网', '新能源', '医药', '消费',  # 新兴行业
            '央行', '货币政策', '利率', '存款准备金',  # 货币政策
            'GDP', 'CPI', 'PMI', '经济数据',  # 经济指标
            '股市', '上涨', '下跌', '涨停', '跌停', '成交量',  # 市场相关
            '财报', '业绩', '分红', '重组', '并购', 'IPO'  # 公司相关
        ]
        
        for news in news_list:
            title = news.get('title', '')
            content = news.get('content', '')
            combined_text = f"{title} {content}".lower()
            
            # 计算相关性分数
            relevance_score = 0
            
            # 直接匹配股票名称和代码
            if stock_name in combined_text:
                relevance_score += 20
            if stock_code in combined_text:
                relevance_score += 20
            
            # 关键词匹配
            for keyword in keywords:
                if keyword in combined_text:
                    relevance_score += 1
            
            # 设置相关性阈值
            if relevance_score >= 3:
                news['relevance_score'] = relevance_score
                relevant_news.append(news)
        
        # 按相关性排序
        relevant_news.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        logger.info(f"从{len(news_list)}条真实新闻中筛选出{len(relevant_news)}条相关新闻")
        return relevant_news[:10]  # 限制相关新闻数量
    
    def _convert_to_ts_code(self, stock_code: str) -> str:
        """将股票代码转换为Tushare格式"""
        if stock_code.startswith('60'):
            return f"{stock_code}.SH"
        elif stock_code.startswith(('00', '30')):
            return f"{stock_code}.SZ"
        else:
            return stock_code
    
    def _deduplicate_news(self, news_list: List[Dict]) -> List[Dict]:
        """去除重复新闻"""
        seen_titles = set()
        unique_news = []
        
        for news in news_list:
            title = news.get('title', '').strip()
            # 简化标题用于去重
            simplified_title = re.sub(r'[：:：\s]+', '', title)[:50]
            
            if simplified_title and simplified_title not in seen_titles:
                seen_titles.add(simplified_title)
                unique_news.append(news)
        
        return unique_news
    
    def _parse_datetime(self, datetime_str: str) -> datetime:
        """解析日期时间字符串"""
        try:
            # 尝试多种日期格式
            formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d',
                '%Y/%m/%d',
                '%Y%m%d'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(datetime_str, fmt)
                except Exception:
                    continue
            
            # 如果都失败，返回当前时间
            return datetime.now()
            
        except Exception:
            return datetime.now()
    
    def _generate_news_summary(self, news_items: List[Dict], stock_name: str) -> str:
        """生成新闻摘要"""
        if not news_items:
            return f"{stock_name}近期暂无相关新闻"
        
        # 分析新闻类型
        type_counts = {}
        for news in news_items:
            news_type = news.get('type', 'unknown')
            type_counts[news_type] = type_counts.get(news_type, 0) + 1
        
        summary_parts = []
        
        # 数量统计
        summary_parts.append(f"通过Tushare Pro获取{len(news_items)}条真实信息")
        
        # 来源分析
        if type_counts.get('real_news', 0) > 0:
            summary_parts.append(f"包含{type_counts['real_news']}条真实新闻")
        if type_counts.get('economic_calendar', 0) > 0:
            summary_parts.append(f"{type_counts['economic_calendar']}条财经日历")
        if type_counts.get('disclosure_schedule', 0) > 0:
            summary_parts.append(f"{type_counts['disclosure_schedule']}条披露信息")
        if type_counts.get('stock_context', 0) > 0:
            summary_parts.append(f"{type_counts['stock_context']}条基础信息")
        
        # 最新消息
        if news_items:
            latest_news = news_items[0]
            summary_parts.append(f"最新：{latest_news.get('title', '')[:25]}...")
        
        return " | ".join(summary_parts)

    def get_market_indices_data(self, days: int = 5) -> Dict[str, Any]:
        """获取主要市场指数数据"""
        logger.info("获取主要市场指数数据...")
        
        # 重要A股指数（包含中证2000和中证全指）
        important_indices = {
            '000001.SH': '上证指数',
            '399001.SZ': '深证成指', 
            '399006.SZ': '创业板指',
            '000688.SH': '科创50',
            '000016.SH': '上证50',
            '000300.SH': '沪深300',
            '000905.SH': '中证500',
            '000852.SH': '中证1000',
            '932000.CSI': '中证2000',
            '000985.SH': '中证全指'
        }
        
        indices_data = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        for ts_code, name in important_indices.items():
            try:
                self._rate_limit()
                
                df = self.pro.index_daily(
                    ts_code=ts_code,
                    start_date=start_date.strftime('%Y%m%d'),
                    end_date=end_date.strftime('%Y%m%d')
                )
                
                if not df.empty:
                    df = df.sort_values('trade_date')
                    latest = df.iloc[-1]
                    prev = df.iloc[-2] if len(df) > 1 else latest
                    
                    indices_data[ts_code] = {
                        'name': name,
                        'latest_price': float(latest['close']),
                        'change': float(latest['close'] - prev['close']),
                        'change_pct': float(latest['pct_chg']),
                        'volume': float(latest['vol']) if 'vol' in latest else 0,
                        'turnover': float(latest['amount']) if 'amount' in latest else 0,
                        'date': latest['trade_date'],
                        'high': float(latest['high']),
                        'low': float(latest['low']),
                        'open': float(latest['open']),
                        'market': ts_code.split('.')[1] if '.' in ts_code else 'CSI'
                    }
                    
                    logger.info(f"获取{name}数据成功: {latest['close']:.2f} ({latest['pct_chg']:+.2f}%)")
                    
            except Exception as e:
                logger.warning(f"获取{name}({ts_code})数据失败: {e}")
                
        return indices_data

    def get_sector_performance_data(self, days: int = 5) -> Dict[str, Any]:
        """获取申万二级行业板块表现数据 - 暂时禁用（权限不足）"""
        logger.info("申万行业板块数据获取已暂时禁用（API权限限制）")
        
        return {}
        
        # 申万二级行业指数代码（暂时注释，等获得权限后启用）
        """
        sw_l2_indices = {
            # 农林牧渔
            '801014.SI': '饲料',
            '801016.SI': '种植业',
            '801017.SI': '养殖业',
            
            # 基础化工
            '801032.SI': '化学纤维',
            '801033.SI': '化学原料',
            '801034.SI': '化学制品',
            '801036.SI': '塑料',
            '801037.SI': '橡胶',
            
            # 钢铁
            '801043.SI': '冶钢原料',
            '801044.SI': '普钢',
            
            # 有色金属
            '801051.SI': '金属新材料',
            '801053.SI': '贵金属',
            '801054.SI': '小金属',
            
            # 电子
            '801081.SI': '半导体',
            '801082.SI': '消费电子',
            '801083.SI': '其他电子',
            '801084.SI': '电子化学品',
            
            # 家用电器
            '801111.SI': '白色家电',
            '801112.SI': '黑色家电',
            '801113.SI': '小家电',
            
            # 食品饮料
            '801121.SI': '白酒',
            '801122.SI': '其他酒类',
            '801123.SI': '食品加工',
            '801124.SI': '调味发酵品',
            '801125.SI': '乳品',
            
            # 医药生物
            '801151.SI': '化学制药',
            '801152.SI': '中药',
            '801153.SI': '生物制品',
            '801154.SI': '医疗器械',
            '801155.SI': '医疗服务',
            '801156.SI': '医药商业',
            
            # 公用事业
            '801161.SI': '电力',
            '801162.SI': '水务',
            '801163.SI': '燃气',
            '801164.SI': '环保工程及服务',
            
            # 交通运输  
            '801171.SI': '航空运输',
            '801172.SI': '航运',
            '801173.SI': '港口',
            '801174.SI': '高速公路',
            '801175.SI': '物流',
            
            # 房地产
            '801181.SI': '房地产开发',
            '801182.SI': '园区开发',
            '801183.SI': '物业管理',
            
            # 建筑材料
            '801711.SI': '水泥制造',
            '801712.SI': '玻璃制造',
            '801713.SI': '其他建材',
            
            # 建筑装饰
            '801721.SI': '房屋建设',
            '801722.SI': '装修装饰',
            '801723.SI': '基础建设',
            
            # 电气设备
            '801731.SI': '电机',
            '801732.SI': '电气自动化设备',
            '801733.SI': '电网设备',
            
            # 国防军工
            '801741.SI': '航天装备',
            '801742.SI': '航空装备',
            '801743.SI': '地面兵装',
            '801744.SI': '船舶制造',
            
            # 计算机
            '801751.SI': '计算机设备',
            '801752.SI': '计算机应用',
            
            # 传媒
            '801761.SI': '文化传媒',
            '801762.SI': '游戏',
            '801763.SI': '营销传播',
            
            # 通信
            '801771.SI': '通信设备',
            '801772.SI': '通信服务',
            
            # 银行
            '801781.SI': '银行',
            
            # 非银金融
            '801791.SI': '证券',
            '801792.SI': '保险',
            '801793.SI': '多元金融',
            
            # 汽车
            '801881.SI': '汽车整车',
            '801882.SI': '汽车零部件',
            '801883.SI': '汽车服务',
            
            # 机械设备
            '801891.SI': '专用设备',
            '801892.SI': '通用机械',
            '801893.SI': '工程机械',
            '801894.SI': '重型机械'
        }
        """

    def get_financial_news_headlines(self, days: int = 5) -> List[Dict[str, Any]]:
        """获取财经新闻头条"""
        logger.info("获取财经新闻头条...")
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            self._rate_limit()
            
            # 获取新闻数据
            news_df = self.pro.news(
                src='sina',
                start_date=start_date.strftime('%Y-%m-%d %H:%M:%S'),
                end_date=end_date.strftime('%Y-%m-%d %H:%M:%S')
            )
            
            news_headlines = []
            if not news_df.empty:
                for _, row in news_df.head(20).iterrows():  # 限制20条
                    try:
                        title = str(row.get('title', '')) if pd.notna(row.get('title')) else ''
                        content = str(row.get('content', '')) if pd.notna(row.get('content')) else ''
                        datetime_str = str(row['datetime']) if pd.notna(row['datetime']) else ''
                        
                        if title or content:
                            # 简化处理
                            if not title and content:
                                title = content[:50] + ('...' if len(content) > 50 else '')
                            
                            importance = self._assess_news_importance(title, content)
                            
                            news_headlines.append({
                                'title': title,
                                'content': content[:200] + ('...' if len(content) > 200 else ''),
                                'publish_time': datetime_str,
                                'source': 'Tushare新浪财经',
                                'importance': importance,
                                'keywords': self._extract_keywords(title, content)
                            })
                    
                    except Exception as e:
                        logger.debug(f"处理新闻条目失败: {e}")
                        continue
            
            logger.info(f"获取到{len(news_headlines)}条财经新闻")
            return news_headlines
            
        except Exception as e:
            logger.warning(f"获取财经新闻失败: {e}")
            return []

    def _assess_news_importance(self, title: str, content: str) -> str:
        """评估新闻重要性"""
        combined_text = f"{title} {content}".lower()
        
        # 高重要性关键词
        high_importance_keywords = [
            '央行', '货币政策', '降准', '降息', '加息', 'gdp', 'cpi', 'pmi',
            '证监会', '银保监会', '国务院', '发改委', '财政部',
            '科创板', '创业板', '主板', '北交所', 'ipo', '退市'
        ]
        
        # 中等重要性关键词
        medium_importance_keywords = [
            '业绩', '财报', '分红', '重组', '并购', '增持', '减持',
            '基金', '保险', '银行', '券商', '信托'
        ]
        
        for keyword in high_importance_keywords:
            if keyword in combined_text:
                return 'high'
        
        for keyword in medium_importance_keywords:
            if keyword in combined_text:
                return 'medium'
        
        return 'low'

    def _extract_keywords(self, title: str, content: str) -> List[str]:
        """提取新闻关键词"""
        combined_text = f"{title} {content}"
        
        # 简单的关键词提取
        keywords = []
        key_terms = [
            '上涨', '下跌', '涨停', '跌停', '突破', '回调',
            '利好', '利空', '业绩', '重组', '并购', '分红',
            '央行', '降准', '降息', 'GDP', 'CPI', 'PMI',
            '新能源', '人工智能', '芯片', '医药', '地产'
        ]
        
        for term in key_terms:
            if term in combined_text:
                keywords.append(term)
        
        return keywords[:5]  # 最多返回5个关键词

    def get_comprehensive_market_data(self, days: int = 5) -> Dict[str, Any]:
        """获取综合市场数据"""
        logger.info(f"开始获取综合市场数据 ({days}天)")
        
        try:
            # 获取各类数据
            indices_data = self.get_market_indices_data(days)
            sector_data = self.get_sector_performance_data(days)
            news_data = self.get_financial_news_headlines(days)
            
            # 计算市场情绪指标
            market_sentiment = self._calculate_market_sentiment_from_data(indices_data, sector_data)
            
            # 识别强弱板块
            sector_ranking = self._rank_sectors_by_performance(sector_data)
            
            result = {
                'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                'data_range_days': days,
                'indices': indices_data,
                'sectors': sector_data,
                'sector_ranking': sector_ranking,
                'news_headlines': news_data,
                'market_sentiment': market_sentiment,
                'data_source': 'Tushare Pro API',
                'update_timestamp': datetime.now().isoformat(),
                'summary': {
                    'indices_count': len(indices_data),
                    'sectors_count': len(sector_data),
                    'news_count': len(news_data),
                    'strong_sectors_count': len(sector_ranking.get('strong_sectors', [])),
                    'weak_sectors_count': len(sector_ranking.get('weak_sectors', []))
                }
            }
            
            logger.info(f"综合市场数据获取完成：指数{len(indices_data)}个，行业{len(sector_data)}个，新闻{len(news_data)}条")
            return result
            
        except Exception as e:
            logger.error(f"获取综合市场数据失败: {e}")
            return {
                'error': str(e),
                'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                'data_source': 'Tushare Pro API'
            }

    def _calculate_market_sentiment_from_data(self, indices_data: Dict, sector_data: Dict) -> Dict[str, Any]:
        """基于数据计算市场情绪"""
        if not indices_data:
            return {'sentiment_score': 50, 'sentiment_level': '中性', 'description': '数据不足'}
        
        # 计算主要指数平均涨跌幅
        major_indices_changes = []
        for code, data in indices_data.items():
            if code in ['000001.SH', '399001.SZ', '399006.SZ', '000300.SH', '000905.SH']:
                major_indices_changes.append(data['change_pct'])
        
        avg_index_change = np.mean(major_indices_changes) if major_indices_changes else 0
        
        # 计算行业上涨比例
        if sector_data:
            rising_sectors = sum(1 for data in sector_data.values() if data['change_pct'] > 0)
            sector_ratio = rising_sectors / len(sector_data)
        else:
            sector_ratio = 0.5
        
        # 综合情绪评分
        sentiment_score = 50 + avg_index_change * 10 + (sector_ratio - 0.5) * 40
        sentiment_score = max(0, min(100, sentiment_score))
        
        # 情绪等级
        if sentiment_score >= 70:
            sentiment_level = "乐观"
        elif sentiment_score >= 55:
            sentiment_level = "偏乐观"  
        elif sentiment_score >= 45:
            sentiment_level = "中性"
        elif sentiment_score >= 30:
            sentiment_level = "偏悲观"
        else:
            sentiment_level = "悲观"
        
        return {
            'sentiment_score': round(sentiment_score, 1),
            'sentiment_level': sentiment_level,
            'avg_index_change': round(avg_index_change, 2),
            'rising_sector_ratio': round(sector_ratio, 2),
            'description': f"主要指数平均{avg_index_change:+.2f}%，{sector_ratio:.1%}行业上涨"
        }

    def _rank_sectors_by_performance(self, sector_data: Dict) -> Dict[str, List]:
        """根据表现对行业板块排名"""
        if not sector_data:
            return {'strong_sectors': [], 'weak_sectors': []}
        
        # 按日涨跌幅排序
        sectors_list = []
        for code, data in sector_data.items():
            sectors_list.append({
                'code': code,
                'name': data['name'],
                'change_pct': data['change_pct'],
                'change_5d_pct': data.get('change_5d_pct', 0)
            })
        
        # 按日涨跌幅排序
        sectors_list.sort(key=lambda x: x['change_pct'], reverse=True)
        
        # 取前5强和后5弱
        strong_sectors = sectors_list[:5]
        weak_sectors = sectors_list[-5:]
        
        return {
            'strong_sectors': strong_sectors,
            'weak_sectors': weak_sectors,
            'total_sectors': len(sectors_list)
        }


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    fetcher = PureTushareNewsFetcher()
    
    # 测试几个股票
    test_stocks = [
        ('000001', '平安银行'),
        ('600036', '招商银行'),
        ('000002', '万科A'),
        ('600519', '贵州茅台'),
        ('000858', '五粮液')
    ]
    
    for code, name in test_stocks:
        print(f"\n=== 测试 {code} {name} ===")
        news_data = fetcher.get_stock_news(code, name, days=7)  # 减少天数以提高API效率
        
        print(f"📊 新闻数量: {news_data['news_count']}")
        print(f"📡 数据来源: {', '.join(news_data['success_sources'])}")
        print(f"📝 摘要: {news_data['summary']}")
        print(f"🔗 数据源: {news_data.get('data_source', 'Unknown')}")
        
        if news_data['news_items']:
            print("🔥 相关信息:")
            for i, news in enumerate(news_data['news_items'][:5], 1):
                print(f"  {i}. {news['title'][:50]}...")
                print(f"     ⏰ {news['publish_time']} | 📺 {news['source']} | ⭐ {news.get('relevance_score', 0)}分")
        else:
            print("❌ 未获取到相关信息")
        
        print("-" * 60)