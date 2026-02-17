#!/usr/bin/env python3
"""
市场走势分析器 - 获取和分析大盘、板块、全球市场等数据
支持：中国A股指数、香港市场、全球主要市场、大宗商品、汇率、财经新闻
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import yfinance as yf
import tushare as ts
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketTrendAnalyzer:
    """市场走势分析器"""
    
    def __init__(self, config_path: str = "config.json"):
        """初始化分析器"""
        self.config = self._load_config(config_path)
        self.project_root = Path(__file__).parent.absolute()
        
        # 初始化Tushare
        tushare_token = self.config.get('tushare', {}).get('token')
        if tushare_token:
            ts.set_token(tushare_token)
            self.ts_api = ts.pro_api()
        else:
            logger.warning("未找到Tushare API token，部分功能可能不可用")
            self.ts_api = None
        
        # 中国A股主要指数
        self.china_indices = {
            '000001.SH': '上证指数',
            '399001.SZ': '深证成指', 
            '399006.SZ': '创业板指',
            '000688.SH': '科创50',
            '000016.SH': '上证50',
            '000300.SH': '沪深300',
            '000905.SH': '中证500',
            '000852.SH': '中证1000',
            '399005.SZ': '中小板指',
            '000009.SH': '上证380',
            '000010.SH': '上证180',
            '000043.SH': '超大盘',
            '399101.SZ': '中小300',
            '399330.SZ': '深证100'
        }
        
        # 香港市场指数
        self.hk_indices = {
            '^HSI': '恒生指数',
            '^HSCE': '恒生国企指数',
            '^HSTECH': '恒生科技指数',
            '^HST': '恒生科技30',
            '^HSHARES': 'H股指数'
        }
        
        # 全球主要指数
        self.global_indices = {
            '^GSPC': '标普500',
            '^DJI': '道琼斯',
            '^IXIC': '纳斯达克',
            '^RUT': '罗素2000',
            '^N225': '日经225',
            '^FTSE': '富时100',
            '^GDAXI': '德国DAX',
            '^FCHI': '法国CAC40',
            'IMOEX.ME': '俄罗斯RTS',
            '^BSESN': '印度孟买',
            '^BVSP': '巴西IBOV',
            '^KS11': '韩国KOSPI',
            '^TWII': '台湾加权',
            '^AORD': '澳大利亚全股',
            '^AXJO': '澳洲200'
        }
        
        # 大宗商品
        self.commodities = {
            'GC=F': '黄金期货',
            'SI=F': '白银期货',
            'CL=F': '原油期货',
            'BZ=F': '布伦特原油',
            'NG=F': '天然气',
            'ZC=F': '玉米期货',
            'ZS=F': '大豆期货',
            'ZW=F': '小麦期货',
            'HG=F': '铜期货',
            'PL=F': '铂金期货'
        }
        
        # 主要汇率
        self.currencies = {
            'USDCNY=X': '美元/人民币',
            'CNY=X': '人民币指数',
            'EURUSD=X': '欧元/美元',
            'GBPUSD=X': '英镑/美元',
            'USDJPY=X': '美元/日元',
            'AUDUSD=X': '澳元/美元',
            'USDCAD=X': '美元/加元',
            'USDCHF=X': '美元/瑞郎',
            'DX-Y.NYB': '美元指数'
        }
        
        # A股行业板块 (申万一级行业)
        self.sw_industries = [
            '801010.SI',  # 农林牧渔
            '801020.SI',  # 采掘
            '801030.SI',  # 化工  
            '801040.SI',  # 钢铁
            '801050.SI',  # 有色金属
            '801080.SI',  # 电子
            '801110.SI',  # 家用电器
            '801120.SI',  # 食品饮料
            '801130.SI',  # 纺织服装
            '801140.SI',  # 轻工制造
            '801150.SI',  # 医药生物
            '801160.SI',  # 公用事业
            '801170.SI',  # 交通运输
            '801180.SI',  # 房地产
            '801200.SI',  # 商业贸易
            '801210.SI',  # 休闲服务
            '801230.SI',  # 综合
            '801710.SI',  # 建筑材料
            '801720.SI',  # 建筑装饰
            '801730.SI',  # 电气设备
            '801740.SI',  # 国防军工
            '801750.SI',  # 计算机
            '801760.SI',  # 传媒
            '801770.SI',  # 通信
            '801780.SI',  # 银行
            '801790.SI',  # 非银金融
            '801880.SI',  # 汽车
            '801890.SI'   # 机械设备
        ]

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"加载配置文件失败: {e}")
            return {}

    def get_china_indices_data(self, days: int = 5) -> Dict[str, Any]:
        """获取中国A股主要指数数据"""
        logger.info("获取中国A股指数数据...")
        
        indices_data = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        for code, name in self.china_indices.items():
            try:
                # 使用Tushare获取指数数据
                if self.ts_api:
                    df = self.ts_api.index_daily(
                        ts_code=code,
                        start_date=start_date.strftime('%Y%m%d'),
                        end_date=end_date.strftime('%Y%m%d')
                    )
                    
                    if not df.empty:
                        df = df.sort_values('trade_date')
                        latest = df.iloc[-1]
                        prev = df.iloc[-2] if len(df) > 1 else latest
                        
                        indices_data[code] = {
                            'name': name,
                            'latest_price': float(latest['close']),
                            'change': float(latest['close'] - prev['close']),
                            'change_pct': float(latest['pct_chg']),
                            'volume': float(latest['vol']) if 'vol' in latest else 0,
                            'turnover': float(latest['amount']) if 'amount' in latest else 0,
                            'date': latest['trade_date'],
                            'high': float(latest['high']),
                            'low': float(latest['low']),
                            'open': float(latest['open'])
                        }
                        
                        logger.info(f"获取{name}数据成功: {latest['close']:.2f} ({latest['pct_chg']:+.2f}%)")
                
            except Exception as e:
                logger.warning(f"获取{name}({code})数据失败: {e}")
                
        return indices_data

    def get_hk_market_data(self, days: int = 5) -> Dict[str, Any]:
        """获取香港市场数据"""
        logger.info("获取香港市场数据...")
        
        hk_data = {}
        
        for symbol, name in self.hk_indices.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=f"{days}d")
                
                if not hist.empty:
                    latest = hist.iloc[-1]
                    prev = hist.iloc[-2] if len(hist) > 1 else latest
                    
                    change = latest['Close'] - prev['Close']
                    change_pct = (change / prev['Close']) * 100
                    
                    hk_data[symbol] = {
                        'name': name,
                        'latest_price': float(latest['Close']),
                        'change': float(change),
                        'change_pct': float(change_pct),
                        'volume': int(latest['Volume']),
                        'high': float(latest['High']),
                        'low': float(latest['Low']),
                        'open': float(latest['Open'])
                    }
                    
                    logger.info(f"获取{name}数据成功: {latest['Close']:.2f} ({change_pct:+.2f}%)")
                    
            except Exception as e:
                logger.warning(f"获取{name}({symbol})数据失败: {e}")
                
        return hk_data

    def get_global_markets_data(self, days: int = 5) -> Dict[str, Any]:
        """获取全球主要市场数据"""
        logger.info("获取全球市场数据...")
        
        global_data = {}
        
        for symbol, name in self.global_indices.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=f"{days}d")
                
                if not hist.empty:
                    latest = hist.iloc[-1]
                    prev = hist.iloc[-2] if len(hist) > 1 else latest
                    
                    change = latest['Close'] - prev['Close']
                    change_pct = (change / prev['Close']) * 100
                    
                    global_data[symbol] = {
                        'name': name,
                        'latest_price': float(latest['Close']),
                        'change': float(change),
                        'change_pct': float(change_pct),
                        'volume': int(latest['Volume']) if not pd.isna(latest['Volume']) else 0,
                        'high': float(latest['High']),
                        'low': float(latest['Low']),
                        'open': float(latest['Open'])
                    }
                    
                    logger.info(f"获取{name}数据成功: {latest['Close']:.2f} ({change_pct:+.2f}%)")
                    
            except Exception as e:
                logger.warning(f"获取{name}({symbol})数据失败: {e}")
                
        return global_data

    def get_commodities_data(self, days: int = 5) -> Dict[str, Any]:
        """获取大宗商品数据"""
        logger.info("获取大宗商品数据...")
        
        commodities_data = {}
        
        for symbol, name in self.commodities.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=f"{days}d")
                
                if not hist.empty:
                    latest = hist.iloc[-1]
                    prev = hist.iloc[-2] if len(hist) > 1 else latest
                    
                    change = latest['Close'] - prev['Close']
                    change_pct = (change / prev['Close']) * 100
                    
                    commodities_data[symbol] = {
                        'name': name,
                        'latest_price': float(latest['Close']),
                        'change': float(change),
                        'change_pct': float(change_pct),
                        'volume': int(latest['Volume']) if not pd.isna(latest['Volume']) else 0,
                        'high': float(latest['High']),
                        'low': float(latest['Low']),
                        'open': float(latest['Open'])
                    }
                    
                    logger.info(f"获取{name}数据成功: {latest['Close']:.2f} ({change_pct:+.2f}%)")
                    
            except Exception as e:
                logger.warning(f"获取{name}({symbol})数据失败: {e}")
                
        return commodities_data

    def get_currency_data(self, days: int = 5) -> Dict[str, Any]:
        """获取主要汇率数据"""
        logger.info("获取汇率数据...")
        
        currency_data = {}
        
        for symbol, name in self.currencies.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=f"{days}d")
                
                if not hist.empty:
                    latest = hist.iloc[-1]
                    prev = hist.iloc[-2] if len(hist) > 1 else latest
                    
                    change = latest['Close'] - prev['Close']
                    change_pct = (change / prev['Close']) * 100
                    
                    currency_data[symbol] = {
                        'name': name,
                        'latest_price': float(latest['Close']),
                        'change': float(change),
                        'change_pct': float(change_pct),
                        'high': float(latest['High']),
                        'low': float(latest['Low']),
                        'open': float(latest['Open'])
                    }
                    
                    logger.info(f"获取{name}数据成功: {latest['Close']:.4f} ({change_pct:+.2f}%)")
                    
            except Exception as e:
                logger.warning(f"获取{name}({symbol})数据失败: {e}")
                
        return currency_data

    def get_sector_analysis(self, days: int = 5) -> Dict[str, Any]:
        """获取行业板块分析"""
        logger.info("获取行业板块数据...")
        
        if not self.ts_api:
            logger.warning("Tushare API不可用，跳过板块分析")
            return {}
            
        sector_data = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            # 获取申万一级行业数据
            for code in self.sw_industries[:10]:  # 限制前10个行业避免API限制
                try:
                    df = self.ts_api.index_daily(
                        ts_code=code,
                        start_date=start_date.strftime('%Y%m%d'),
                        end_date=end_date.strftime('%Y%m%d')
                    )
                    
                    if not df.empty:
                        df = df.sort_values('trade_date')
                        latest = df.iloc[-1]
                        
                        # 获取行业名称
                        industry_name = self._get_industry_name(code)
                        
                        sector_data[code] = {
                            'name': industry_name,
                            'latest_price': float(latest['close']),
                            'change_pct': float(latest['pct_chg']),
                            'volume': float(latest['vol']) if 'vol' in latest else 0,
                            'turnover': float(latest['amount']) if 'amount' in latest else 0
                        }
                        
                        logger.info(f"获取{industry_name}数据成功: {latest['pct_chg']:+.2f}%")
                        
                    time.sleep(0.2)  # 控制API调用频率
                    
                except Exception as e:
                    logger.warning(f"获取行业{code}数据失败: {e}")
                    
        except Exception as e:
            logger.error(f"获取行业板块数据失败: {e}")
            
        return sector_data

    def _get_industry_name(self, code: str) -> str:
        """根据行业代码获取行业名称"""
        industry_names = {
            '801010.SI': '农林牧渔',
            '801020.SI': '采掘',
            '801030.SI': '化工',
            '801040.SI': '钢铁',
            '801050.SI': '有色金属',
            '801080.SI': '电子',
            '801110.SI': '家用电器',
            '801120.SI': '食品饮料',
            '801130.SI': '纺织服装',
            '801140.SI': '轻工制造',
            '801150.SI': '医药生物',
            '801160.SI': '公用事业',
            '801170.SI': '交通运输',
            '801180.SI': '房地产',
            '801200.SI': '商业贸易'
        }
        return industry_names.get(code, f'行业{code}')

    def get_financial_news(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取重要财经新闻（模拟数据，实际需要接入新闻API）"""
        logger.info("获取财经新闻...")
        
        # 这里应该接入真实的新闻API，如新浪财经、东方财富等
        # 目前提供模拟数据结构
        mock_news = [
            {
                'title': '央行宣布降准0.25个百分点',
                'summary': '央行决定于近期下调存款准备金率0.25个百分点，释放流动性约5000亿元',
                'source': '央行官网',
                'publish_time': '2025-08-08 09:30:00',
                'importance': 'high',
                'sentiment': 'positive',
                'impact_markets': ['股市', '债市', '汇率']
            },
            {
                'title': '美联储维持利率不变',
                'summary': '美联储宣布维持联邦基金利率在5.25%-5.50%区间不变',
                'source': '美联储',
                'publish_time': '2025-08-08 02:00:00',
                'importance': 'high',
                'sentiment': 'neutral',
                'impact_markets': ['全球股市', '美元', '大宗商品']
            }
        ]
        
        return mock_news[:limit]

    def analyze_market_comprehensive(self, analysis_date: str = None) -> Dict[str, Any]:
        """综合市场分析"""
        if analysis_date is None:
            analysis_date = datetime.now().strftime("%Y-%m-%d")
            
        logger.info(f"开始综合市场分析 - {analysis_date}")
        
        # 获取所有市场数据
        china_data = self.get_china_indices_data(days=5)
        hk_data = self.get_hk_market_data(days=5)
        global_data = self.get_global_markets_data(days=5)
        commodities_data = self.get_commodities_data(days=5)
        currency_data = self.get_currency_data(days=5)
        sector_data = self.get_sector_analysis(days=5)
        news_data = self.get_financial_news(limit=10)
        
        # 计算市场情绪指标
        market_sentiment = self._calculate_market_sentiment(china_data, global_data, commodities_data)
        
        # 识别强势和弱势板块
        sector_ranking = self._rank_sectors(sector_data)
        
        # 全球市场联动分析
        global_correlation = self._analyze_global_correlation(china_data, hk_data, global_data)
        
        return {
            'analysis_date': analysis_date,
            'china_indices': china_data,
            'hk_market': hk_data,
            'global_markets': global_data,
            'commodities': commodities_data,
            'currencies': currency_data,
            'sectors': sector_data,
            'sector_ranking': sector_ranking,
            'financial_news': news_data,
            'market_sentiment': market_sentiment,
            'global_correlation': global_correlation,
            'analysis_timestamp': datetime.now().isoformat()
        }

    def _calculate_market_sentiment(self, china_data: Dict, global_data: Dict, commodities_data: Dict) -> Dict[str, Any]:
        """计算市场情绪指标"""
        
        # 计算A股整体表现
        china_changes = [data['change_pct'] for data in china_data.values() if 'change_pct' in data]
        china_avg_change = np.mean(china_changes) if china_changes else 0
        
        # 计算全球市场表现
        global_changes = [data['change_pct'] for data in global_data.values() if 'change_pct' in data]
        global_avg_change = np.mean(global_changes) if global_changes else 0
        
        # 计算风险资产表现（黄金作为避险指标）
        gold_change = 0
        if 'GC=F' in commodities_data:
            gold_change = commodities_data['GC=F'].get('change_pct', 0)
        
        # 综合情绪评分（0-100分）
        sentiment_score = 50  # 基准分
        sentiment_score += china_avg_change * 2  # A股权重更大
        sentiment_score += global_avg_change * 1
        sentiment_score -= gold_change * 0.5  # 黄金上涨通常表示避险情绪
        
        sentiment_score = max(0, min(100, sentiment_score))  # 限制在0-100之间
        
        # 情绪等级
        if sentiment_score >= 70:
            sentiment_level = "极度乐观"
        elif sentiment_score >= 55:
            sentiment_level = "乐观"
        elif sentiment_score >= 45:
            sentiment_level = "中性"
        elif sentiment_score >= 30:
            sentiment_level = "悲观"
        else:
            sentiment_level = "极度悲观"
        
        return {
            'overall_score': round(sentiment_score, 1),
            'sentiment_level': sentiment_level,
            'china_avg_change': round(china_avg_change, 2),
            'global_avg_change': round(global_avg_change, 2),
            'risk_appetite': "高" if gold_change < -1 else "中" if gold_change < 1 else "低"
        }

    def _rank_sectors(self, sector_data: Dict) -> Dict[str, List]:
        """对行业板块进行排名"""
        if not sector_data:
            return {'strong_sectors': [], 'weak_sectors': []}
        
        # 按涨跌幅排序
        sectors_list = []
        for code, data in sector_data.items():
            sectors_list.append({
                'code': code,
                'name': data['name'],
                'change_pct': data['change_pct']
            })
        
        sectors_list.sort(key=lambda x: x['change_pct'], reverse=True)
        
        # 取前3强和后3弱
        strong_sectors = sectors_list[:3]
        weak_sectors = sectors_list[-3:]
        
        return {
            'strong_sectors': strong_sectors,
            'weak_sectors': weak_sectors
        }

    def _analyze_global_correlation(self, china_data: Dict, hk_data: Dict, global_data: Dict) -> Dict[str, Any]:
        """分析全球市场联动性"""
        
        # 获取主要指数涨跌幅
        correlations = {}
        
        # A股代表
        if '000001.SH' in china_data:
            sh_change = china_data['000001.SH'].get('change_pct', 0)
        else:
            sh_change = 0
        
        # 香港代表
        if '^HSI' in hk_data:
            hsi_change = hk_data['^HSI'].get('change_pct', 0)
            correlations['港股联动'] = "正向" if sh_change * hsi_change > 0 else "负向"
        
        # 美股代表
        if '^GSPC' in global_data:
            sp500_change = global_data['^GSPC'].get('change_pct', 0)
            correlations['美股联动'] = "正向" if sh_change * sp500_change > 0 else "负向"
        
        return correlations


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="市场走势综合分析器")
    parser.add_argument("--date", type=str, default=None,
                       help="分析日期 (YYYY-MM-DD，默认为今天)")
    parser.add_argument("--config", type=str, default="config.json",
                       help="配置文件路径")
    parser.add_argument("--save", action="store_true",
                       help="保存分析结果到文件")
    
    args = parser.parse_args()
    
    print(f"🌍 启动市场走势综合分析器")
    print(f"📅 分析日期: {args.date or '今日'}")
    
    # 创建分析器
    analyzer = MarketTrendAnalyzer(args.config)
    
    # 进行综合分析
    print(f"🔄 开始综合市场分析...\n")
    result = analyzer.analyze_market_comprehensive(args.date)
    
    # 显示结果摘要
    print(f"📊 市场分析完成!")
    print(f"📈 A股指数获取: {len(result['china_indices'])}个")
    print(f"🏙️  香港市场: {len(result['hk_market'])}个指数")
    print(f"🌏 全球市场: {len(result['global_markets'])}个指数")
    print(f"🥇 大宗商品: {len(result['commodities'])}个品种")
    print(f"💱 汇率数据: {len(result['currencies'])}个货币对")
    print(f"🏭 行业板块: {len(result['sectors'])}个行业")
    print(f"📰 财经新闻: {len(result['financial_news'])}条")
    
    if result.get('market_sentiment'):
        sentiment = result['market_sentiment']
        print(f"\n💫 市场情绪: {sentiment['sentiment_level']} (评分: {sentiment['overall_score']})")
    
    # 保存结果
    if args.save:
        output_dir = Path("reports") / "market_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        date_str = (args.date or datetime.now().strftime("%Y-%m-%d")).replace('-', '')
        output_file = output_dir / f"市场走势分析_{date_str}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"💾 分析结果已保存到: {output_file}")


if __name__ == "__main__":
    main()