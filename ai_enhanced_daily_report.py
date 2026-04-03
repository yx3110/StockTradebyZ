#!/usr/bin/env python3
"""
AI增强每日选股报告生成器
使用Claude作为核心分析引擎，统一进行技术面、基本面分析和打分
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from multiprocessing import cpu_count

# 添加必要的路径
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))
# 导入Claude分析器和市场综合分析器
try:
    from archive.experimental.claude_driven_analyzer import ClaudeDrivenAnalyzer
except ImportError:
    ClaudeDrivenAnalyzer = None

try:
    from market_comprehensive_analyzer import MarketComprehensiveAnalyzer
except ImportError:
    MarketComprehensiveAnalyzer = None

class EnhancedReportParser:
    """简易报告解析器 (TA_integration已移除)"""
    def parse_report(self, path):
        return {'stocks': [], 'error': '解析器不可用'}

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIEnhancedDailyReport:
    """AI增强每日选股报告生成器"""
    
    def __init__(self, config_path: str = "config.json", max_workers: int = None, scoring_version: str = "v3"):
        """初始化生成器"""
        self.config = self._load_config(config_path)
        self.project_root = Path(__file__).parent.absolute()
        self.scoring_version = scoring_version  # 评分版本: "v3" 或 "v3.1"
        
        # 并行处理配置
        self.max_workers = max_workers or min(3, cpu_count())
        self.api_call_delay = 0.5  # API调用间隔
        self.retry_attempts = 3
        self.retry_delay = 2
        
        # 统计信息
        self.stats = {
            'total_stocks': 0,
            'success_count': 0,
            'failed_count': 0,
            'retry_count': 0,
            'start_time': None,
            'end_time': None
        }
        
        # 线程安全锁
        self.api_lock = threading.Lock()
        
        # 初始化Claude分析器和市场综合分析器
        self.claude_analyzer = ClaudeDrivenAnalyzer()
        self.market_analyzer = MarketComprehensiveAnalyzer(config_path)
        
        # 情绪分析组件已禁用
        # self.eastmoney_api = EastMoneyAPI()
        self.report_parser = EnhancedReportParser()
        
        # 报告配置
        self.report_config = {
            'show_detailed_analysis': True,
            'analysis_text_length': 200,
            'include_trading_advice': True,
            'include_price_targets': True
        }
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"加载配置文件失败: {e}")
            return {}
    
    def generate_daily_report(self, analysis_date: str = None) -> Dict[str, Any]:
        """生成每日AI增强选股报告 - 支持并行处理"""
        if analysis_date is None:
            analysis_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"开始并行生成{analysis_date}的AI增强选股报告")
        
        # 重置统计
        self.stats = {
            'total_stocks': 0,
            'success_count': 0,
            'failed_count': 0,
            'retry_count': 0,
            'start_time': time.time(),
            'end_time': None
        }
        
        try:
            # 步骤1: 获取市场综合分析
            logger.info("获取市场综合分析...")
            market_analysis = self.market_analyzer.analyze_comprehensive_market(analysis_date, days=5)
            
            # 步骤2: 获取基础量化选股结果
            quantitative_results = self._get_quantitative_results(analysis_date)
            if not quantitative_results:
                logger.error("未能获取量化选股结果")
                return {"error": "未能获取量化选股结果"}
            
            # 步骤3: 获取所有推荐股票
            stocks_to_analyze = quantitative_results['stocks']
            self.stats['total_stocks'] = len(stocks_to_analyze)
            
            logger.info(f"开始并行分析{len(stocks_to_analyze)}只股票，使用{self.max_workers}个线程")
            
            # 步骤4: 并行分析所有股票
            enhanced_stocks = self._parallel_analyze_stocks(stocks_to_analyze, analysis_date)
                    
            # 步骤5: 生成包含市场分析的最终报告
            final_report = self._generate_final_report_with_market(enhanced_stocks, analysis_date, market_analysis)
            
            # 记录结束时间
            self.stats['end_time'] = time.time()
            total_time = self.stats['end_time'] - self.stats['start_time']
            
            # 步骤6: 保存报告
            self._save_reports(final_report, analysis_date)
            
            logger.info(f"AI增强选股报告生成完成：")
            logger.info(f"  - 总股票数: {self.stats['total_stocks']}只")
            logger.info(f"  - 成功分析: {self.stats['success_count']}只")
            logger.info(f"  - 失败分析: {self.stats['failed_count']}只")
            logger.info(f"  - 重试次数: {self.stats['retry_count']}次")
            logger.info(f"  - 总耗时: {total_time:.1f}秒")
            logger.info(f"  - 平均每股: {total_time/len(enhanced_stocks):.1f}秒")
            
            return {
                "success": True,
                "analysis_date": analysis_date,
                "total_stocks": len(enhanced_stocks),
                "detailed_analysis_count": len(enhanced_stocks),
                "enhanced_stocks": enhanced_stocks,
                "final_report": final_report,
                "market_analysis": market_analysis,
                "summary": self._generate_summary_stats(enhanced_stocks),
                "stats": self.stats
            }
            
        except Exception as e:
            logger.error(f"生成AI增强报告失败: {e}")
            return {"error": str(e)}
    
    def _get_quantitative_results(self, analysis_date: str) -> Optional[Dict]:
        """获取量化选股结果"""
        try:
            # 首先尝试从JSON文件读取详细的分析结果
            date_str = analysis_date.replace('-', '')
            
            # 根据评分版本确定数据路径
            if self.scoring_version == "v3.1":
                json_paths = [
                    self.project_root / "reports" / "daily_selection_v3.1" / f"analysis_data_{date_str}.json",
                    self.project_root / f"analysis_data_{date_str}.json"
                ]
            else:
                json_paths = [
                    self.project_root / "reports" / "daily_selection" / f"analysis_data_{date_str}.json",
                    self.project_root / f"analysis_data_{date_str}.json"
                ]
            
            for json_path in json_paths:
                if json_path.exists():
                    logger.info(f"找到量化分析JSON数据: {json_path}")
                    with open(json_path, 'r', encoding='utf-8') as f:
                        analysis_data = json.load(f)
                    
                    # 从JSON数据中提取需要分析的股票
                    stocks_to_analyze = []
                    
                    # 获取需要详细分析的股票（标记为needs_detailed_analysis的）
                    if 'top_recommendations' in analysis_data:
                        for stock in analysis_data['top_recommendations']:
                            if stock.get('needs_detailed_analysis', True):  # 默认为True以兼容旧版本
                                stocks_to_analyze.append({
                                    'rank': len(stocks_to_analyze) + 1,
                                    'code': stock['stock_code'],
                                    'name': stock.get('stock_name', '未知'),
                                    'industry': stock.get('industry', '未知'),
                                    'area': stock.get('area', '未知'),
                                    'market': stock.get('market', '未知'),
                                    'comprehensive_score': stock.get('score', 0),
                                    'strategies': stock.get('strategies', []),
                                    'close_price': stock.get('close_price', 0),
                                    'price_change_pct': stock.get('price_change_pct', 0),
                                    'kdj_k': stock.get('kdj_k', 0),
                                    'kdj_d': stock.get('kdj_d', 0),
                                    'kdj_j': stock.get('kdj_j', 0),
                                    'bbi': stock.get('bbi', 0),
                                    'volume': stock.get('volume', 0),
                                    'market_cap': stock.get('market_cap', 0),
                                    'needs_detailed_analysis': True,
                                    'analysis_reason': stock.get('analysis_reason', '选中分析')
                                })
                    
                    logger.info(f"从JSON数据中获取到 {len(stocks_to_analyze)} 只需要详细分析的股票")
                    return {
                        'stocks': stocks_to_analyze,
                        'total_count': len(stocks_to_analyze),
                        'analysis_date': analysis_date
                    }
            
            # 如果没有JSON文件，尝试从MD报告中解析
            logger.info("未找到JSON数据文件，尝试从MD报告解析...")
            
            # 根据评分版本确定MD报告路径
            if self.scoring_version == "v3.1":
                possible_paths = [
                    self.project_root / "reports" / "daily_selection_v3.1" / f"选股分析报告_{date_str}.md",
                    self.project_root / "daily_result" / f"选股分析报告_{date_str}.md",
                    self.project_root / f"选股分析报告_{date_str}.md"
                ]
            else:
                possible_paths = [
                    self.project_root / "reports" / "daily_selection" / f"选股分析报告_{date_str}.md",
                    self.project_root / "daily_result" / f"选股分析报告_{date_str}.md",
                    self.project_root / f"选股分析报告_{date_str}.md"
                ]
            
            for report_path in possible_paths:
                if report_path.exists():
                    logger.info(f"找到量化选股报告: {report_path}")
                    return self._parse_quantitative_report(str(report_path))
            
            logger.warning(f"未找到{analysis_date}的量化选股报告")
            return None
            
        except Exception as e:
            logger.error(f"获取量化选股结果失败: {e}")
            return None
    
    def _parse_quantitative_report(self, report_path: str) -> Dict:
        """解析量化选股报告"""
        try:
            parsed_result = self.report_parser.parse_report(report_path)
            
            if parsed_result.get('error'):
                logger.error(f"解析报告失败: {parsed_result['error']}")
                return {'stocks': [], 'total_count': 0}
            
            # 转换StockInfo对象为字典格式
            stocks = []
            for stock_info in parsed_result['stocks']:
                stock = {
                    'rank': stock_info.rank,
                    'code': stock_info.code,
                    'name': stock_info.name,
                    'industry': stock_info.industry,
                    'area': stock_info.area,
                    'market': stock_info.market,
                    'comprehensive_score': stock_info.comprehensive_score,
                    'strategies': stock_info.strategies,
                    'close_price': stock_info.close_price,
                    'price_change_pct': stock_info.price_change_pct,
                    'kdj_k': stock_info.kdj_k,
                    'kdj_d': stock_info.kdj_d,
                    'kdj_j': stock_info.kdj_j,
                    'bbi': stock_info.bbi,
                    'volume': stock_info.volume,
                    'market_cap': stock_info.market_cap
                }
                stocks.append(stock)
            
            return {
                'stocks': stocks,
                'total_count': len(stocks),
                'report_info': parsed_result.get('report_info', {}),
                'analysis_date': parsed_result.get('report_info', {}).get('analysis_date', '')
            }
            
        except Exception as e:
            logger.error(f"解析量化报告失败: {e}")
            return {'stocks': [], 'total_count': 0}
    
    def _parallel_analyze_stocks(self, stocks_to_analyze: List[Dict], analysis_date: str) -> List[Dict]:
        """并行分析所有股票"""
        enhanced_stocks = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_stock = {}
            for stock in stocks_to_analyze:
                future = executor.submit(self._perform_claude_analysis_with_retry, stock, analysis_date, 0)
                future_to_stock[future] = stock
            
            # 收集结果
            for future in as_completed(future_to_stock):
                stock = future_to_stock[future]
                try:
                    enhanced_stock = future.result(timeout=300)  # 5分钟超时
                    enhanced_stocks.append(enhanced_stock)
                    
                    # 实时显示进度
                    progress = len(enhanced_stocks)
                    if progress % 5 == 0 or progress == self.stats['total_stocks']:
                        logger.info(f"进度: {progress}/{self.stats['total_stocks']} - 成功:{self.stats['success_count']} 失败:{self.stats['failed_count']} 重试:{self.stats['retry_count']}")
                
                except Exception as e:
                    logger.error(f"并行分析异常 {stock['code']}: {e}")
                    enhanced_stock = self._create_failed_analysis_stock(stock, str(e), 0)
                    enhanced_stocks.append(enhanced_stock)
                    self.stats['failed_count'] += 1
        
        # 按原始排序恢复
        enhanced_stocks.sort(key=lambda x: x.get('rank', 999))
        return enhanced_stocks
    
    def _rate_limit_api_call(self):
        """控制API调用频率"""
        with self.api_lock:
            time.sleep(self.api_call_delay)
    
    def _perform_claude_analysis_with_retry(self, stock: Dict, analysis_date: str, retry_count: int) -> Dict:
        """带重试机制的Claude分析"""
        stock_code = stock['code']
        stock_name = stock.get('name', '')
        
        try:
            # 控制API调用频率
            self._rate_limit_api_call()
            
            logger.info(f"[{stock['rank']}/{self.stats['total_stocks']}] 开始分析: {stock_code} - {stock_name}")
            
            # 获取综合数据
            stock_data = self.claude_analyzer.get_comprehensive_stock_data(stock_code, days=30)
            if not stock_data:
                raise Exception("无法获取股票数据")
            
            # 进行Claude分析
            claude_result = self.claude_analyzer.analyze_stock_with_claude(stock_data, None)
            
            # 合并结果
            enhanced_stock = {
                **stock,
                'claude_analysis': claude_result,
                'sentiment_data': None,
                'has_detailed_analysis': True,
                'analysis_type': 'claude_driven',
                'retry_count': retry_count
            }
            
            # 检查是否有错误
            if 'error' in claude_result:
                raise Exception(claude_result['error'])
            
            logger.info(f"[{stock['rank']}/{self.stats['total_stocks']}] 完成分析: {stock_code} - 评分{claude_result.get('overall_score', 'N/A')}")
            self.stats['success_count'] += 1
            
            return enhanced_stock
            
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"[{stock['rank']}/{self.stats['total_stocks']}] 分析失败 {stock_code}: {error_msg}")
            
            # 检查是否需要重试
            if retry_count < self.retry_attempts:
                # 特定错误类型判断
                if any(keyword in error_msg.lower() for keyword in ['rate limit', 'timeout', '429', '503']):
                    logger.info(f"[{stock['rank']}/{self.stats['total_stocks']}] 等待{self.retry_delay}秒后重试...")
                    time.sleep(self.retry_delay * (retry_count + 1))  # 递增延迟
                    self.stats['retry_count'] += 1
                    return self._perform_claude_analysis_with_retry(stock, analysis_date, retry_count + 1)
            
            # 返回失败结果
            self.stats['failed_count'] += 1
            return self._create_failed_analysis_stock(stock, error_msg, retry_count)
    
    def _perform_claude_analysis(self, stock: Dict, analysis_date: str) -> Dict:
        """对单只股票进行Claude详细分析 (保留为兼容性)"""
        return self._perform_claude_analysis_with_retry(stock, analysis_date, 0)
    
    
    # 情绪数据获取方法已移除
    
    # 情绪评分计算方法已移除
    
    def _create_failed_analysis_stock(self, stock: Dict, error_msg: str, retry_count: int = 0) -> Dict:
        """创建分析失败的股票对象"""
        return {
            **stock,
            'claude_analysis': {
                'overall_score': 50,
                'rating': '分析失败',
                'confidence': 0.0,
                'technical_analysis': {
                    'score': 50,
                    'analysis_text': f'技术分析失败: {error_msg}',
                    'key_levels': {'support': 0, 'resistance': 0},
                    'signals': ['分析过程出现错误']
                },
                'fundamental_analysis': {
                    'score': 50,
                    'analysis_text': f'基本面分析失败: {error_msg}',
                    'valuation_level': '未知',
                    'key_strengths': ['分析失败'],
                    'key_risks': ['无法获取分析结果']
                },
                'trading_recommendation': {
                    'action': '暂停操作',
                    'rationale': f'分析系统异常: {error_msg}'
                },
                'error': error_msg
            },
            'sentiment_data': None,
            'has_detailed_analysis': True,
            'analysis_type': 'failed',
            'retry_count': retry_count
        }
    
    def _generate_final_report(self, enhanced_stocks: List[Dict], analysis_date: str) -> str:
        """生成最终报告"""
        
        # 按Claude评分或量化评分排序
        def get_sort_key(stock):
            if stock.get('claude_analysis') and stock['claude_analysis'].get('overall_score'):
                return stock['claude_analysis']['overall_score']
            return stock.get('comprehensive_score', 0)
        
        enhanced_stocks.sort(key=get_sort_key, reverse=True)
        
        report_lines = [
            "# 🤖 AI增强选股分析报告\n",
            f"## 📊 分析概览\n",
            f"- **分析日期**: {analysis_date}",
            f"- **分析引擎**: Claude 4 + 技术分析 + 基本面分析",
            f"- **量化评分版本**: {self.scoring_version.upper()} {'优化权重版' if self.scoring_version == 'v3.1' else ''}",
            f"- **数据源**: SQLite数据库 + 量化选股策略 (情绪分析已暂停)",
            f"- **分析股票数**: {len(enhanced_stocks)}",
            f"- **详细分析数**: {sum(1 for s in enhanced_stocks if s.get('has_detailed_analysis'))}",
            f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        ]
        
        # 添加投资建议汇总
        claude_analyzed = [s for s in enhanced_stocks if s.get('claude_analysis')]
        buy_count = sum(1 for s in claude_analyzed if s['claude_analysis'].get('trading_recommendation', {}).get('action') in ['买入', 'BUY'])
        hold_count = sum(1 for s in claude_analyzed if s['claude_analysis'].get('trading_recommendation', {}).get('action') in ['持有', 'HOLD'])
        sell_count = sum(1 for s in claude_analyzed if s['claude_analysis'].get('trading_recommendation', {}).get('action') in ['卖出', 'SELL'])
        
        report_lines.extend([
            "## 📈 Claude分析投资建议汇总\n",
            f"- **买入建议**: {buy_count}只股票",
            f"- **持有建议**: {hold_count}只股票", 
            f"- **卖出建议**: {sell_count}只股票",
            f"- **其他建议**: {len(claude_analyzed) - buy_count - hold_count - sell_count}只股票\n"
        ])
        
        # 添加详细股票分析
        report_lines.append("## 🎯 详细分析报告\n")
        
        for i, stock in enumerate(enhanced_stocks, 1):
            stock_code = stock['code']
            stock_name = stock.get('name', '未知')
            claude_result = stock.get('claude_analysis')
            
            report_lines.extend([
                f"### {i}. {stock_code} - {stock_name}",
                ""
            ])
            
            if claude_result and 'error' not in claude_result:
                # 有Claude分析结果
                self._add_claude_analysis_section(report_lines, stock, claude_result)
            else:
                # 没有Claude分析，使用量化结果
                self._add_quantitative_analysis_section(report_lines, stock)
            
            report_lines.append("---\n")
        
        # 添加免责声明
        report_lines.extend([
            "## ⚠️ 重要声明\n",
            "本报告由Claude AI自动生成，仅供参考，不构成投资建议。",
            "所有价格预测和操作建议均基于历史数据和技术分析，市场存在不确定性。",
            "投资有风险，入市需谨慎。请结合个人风险承受能力做出决策。\n",
            "---",
            f"*报告由Claude AI自动生成，生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        ])
        
        return "\n".join(report_lines)
    
    def _generate_final_report_with_market(self, enhanced_stocks: List[Dict], analysis_date: str, market_analysis: Dict) -> str:
        """生成包含市场分析的最终报告"""
        
        # 按Claude评分或量化评分排序
        def get_sort_key(stock):
            if stock.get('claude_analysis') and stock['claude_analysis'].get('overall_score'):
                return stock['claude_analysis']['overall_score']
            return stock.get('comprehensive_score', 0)
        
        enhanced_stocks.sort(key=get_sort_key, reverse=True)
        
        report_lines = [
            "# 🌍 A股市场综合分析与选股报告\n",
        ]
        
        # 1. 市场综合分析部分（放在最上面）
        if 'error' not in market_analysis:
            report_lines.extend(self._add_market_analysis_section(market_analysis, analysis_date))
        else:
            report_lines.extend([
                "## 🔴 市场分析获取失败\n",
                f"市场分析数据获取失败：{market_analysis.get('error', '未知错误')}\n",
                "---\n"
            ])
        
        # 2. 选股分析部分
        report_lines.extend([
            "# 🤖 AI增强选股分析\n",
            f"## 📊 选股概览\n",
            f"- **分析日期**: {analysis_date}",
            f"- **分析引擎**: Claude 4 + 技术分析 + 基本面分析",
            f"- **量化评分版本**: {self.scoring_version.upper()} {'优化权重版' if self.scoring_version == 'v3.1' else ''}",
            f"- **数据源**: SQLite数据库 + 量化选股策略 + 市场综合分析",
            f"- **分析股票数**: {len(enhanced_stocks)}",
            f"- **详细分析数**: {sum(1 for s in enhanced_stocks if s.get('has_detailed_analysis'))}",
            f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        ])
        
        # 添加投资建议汇总
        claude_analyzed = [s for s in enhanced_stocks if s.get('claude_analysis')]
        buy_count = sum(1 for s in claude_analyzed if s['claude_analysis'].get('trading_recommendation', {}).get('action') in ['买入', 'BUY'])
        hold_count = sum(1 for s in claude_analyzed if s['claude_analysis'].get('trading_recommendation', {}).get('action') in ['持有', 'HOLD'])
        sell_count = sum(1 for s in claude_analyzed if s['claude_analysis'].get('trading_recommendation', {}).get('action') in ['卖出', 'SELL'])
        
        report_lines.extend([
            "## 📈 Claude分析投资建议汇总\n",
            f"- **买入建议**: {buy_count}只股票",
            f"- **持有建议**: {hold_count}只股票", 
            f"- **卖出建议**: {sell_count}只股票",
            f"- **其他建议**: {len(claude_analyzed) - buy_count - hold_count - sell_count}只股票\n"
        ])
        
        # 添加详细股票分析
        report_lines.append("## 🎯 详细选股分析\n")
        
        for i, stock in enumerate(enhanced_stocks, 1):
            stock_code = stock['code']
            stock_name = stock.get('name', '未知')
            claude_result = stock.get('claude_analysis')
            
            report_lines.extend([
                f"### {i}. {stock_code} - {stock_name}",
                ""
            ])
            
            if claude_result and 'error' not in claude_result:
                # 有Claude分析结果
                self._add_claude_analysis_section(report_lines, stock, claude_result)
            else:
                # 没有Claude分析，使用量化结果
                self._add_quantitative_analysis_section(report_lines, stock)
            
            report_lines.append("---\n")
        
        # 添加免责声明
        report_lines.extend([
            "## ⚠️ 重要声明\n",
            "本报告由Claude AI自动生成，整合了市场综合分析和个股量化筛选，仅供参考，不构成投资建议。",
            "所有价格预测和操作建议均基于历史数据和技术分析，市场存在不确定性。",
            "投资有风险，入市需谨慎。请结合个人风险承受能力和市场综合分析做出决策。\n",
            "---",
            f"*报告由Claude AI自动生成，包含市场综合分析 | 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        ])
        
        return "\n".join(report_lines)
    
    def _add_market_analysis_section(self, market_analysis: Dict, analysis_date: str) -> List[str]:
        """添加市场分析部分"""
        
        lines = [
            "## 🌍 市场综合分析\n",
            f"**分析日期**: {analysis_date} | **数据范围**: 最近{market_analysis.get('data_range_days', 5)}个交易日\n"
        ]
        
        # 核心结论
        market_rating = market_analysis.get('market_rating', {})
        if market_rating:
            lines.extend([
                "### 🎯 核心结论\n",
                f"- **市场评级**: {market_rating.get('rating', 'N/A')} (综合评分: {market_rating.get('score', 'N/A')})",
                f"- **风险等级**: {market_rating.get('risk_level', 'N/A')}",
                f"- **投资建议**: {market_rating.get('investment_advice', 'N/A')}\n"
            ])
        
        # 各维度分析
        dimensions = [
            ('technical_analysis', '📈 技术面', 'technical'),
            ('fundamental_analysis', '💰 基本面', 'fundamental'), 
            ('sentiment_analysis', '😊 市场情绪', 'sentiment'),
            ('news_analysis', '📰 消息面', 'news')
        ]
        
        lines.append("### 📊 四维分析\n")
        
        for analysis_key, analysis_name, _ in dimensions:
            analysis = market_analysis.get(analysis_key, {})
            if analysis:
                score = analysis.get('score', 'N/A')
                level = analysis.get('level', 'N/A')
                analysis_text = analysis.get('analysis_text', '暂无分析')
                
                lines.extend([
                    f"**{analysis_name}**: {level} ({score}分)",
                    f"{analysis_text}",
                    ""
                ])
        
        # 交易指导
        trading_guidance = market_analysis.get('trading_guidance', {})
        if trading_guidance:
            lines.extend([
                "### 🎯 交易指导\n",
                f"- **整体策略**: {trading_guidance.get('overall_strategy', '')}",
                f"- **仓位建议**: {trading_guidance.get('position_suggestion', '')}",
                f"- **市场时机**: {trading_guidance.get('market_timing', '')}",
                f"- **下日展望**: {trading_guidance.get('next_trading_day_outlook', '')}\n"
            ])
            
            # 风险管理要点
            risk_management = trading_guidance.get('risk_management', [])
            if risk_management:
                lines.extend([
                    "**风险管理要点**:",
                    *[f"- {risk}" for risk in risk_management],
                    ""
                ])
        
        # 数据质量
        data_quality = market_analysis.get('data_quality', {})
        if data_quality:
            lines.extend([
                f"**数据质量**: {data_quality.get('quality_level', '')} - {data_quality.get('completeness', '')}",
                ""
            ])
        
        lines.append("---\n")
        
        return lines
    
    def _add_claude_analysis_section(self, report_lines: List[str], stock: Dict, claude_result: Dict):
        """添加Claude分析部分"""
        
        technical = claude_result.get('technical_analysis', {})
        fundamental = claude_result.get('fundamental_analysis', {})
        sentiment = claude_result.get('sentiment_analysis', {})
        trading = claude_result.get('trading_recommendation', {})
        risk = claude_result.get('risk_assessment', {})
        
        # 基本信息
        report_lines.extend([
            "**📊 综合评估**",
            f"- **Claude评分**: {claude_result.get('overall_score', 0):.1f}分",
            f"- **投资评级**: {claude_result.get('rating', 'N/A')}",
            f"- **分析置信度**: {claude_result.get('confidence', 0):.1%}",
            f"- **风险等级**: {risk.get('risk_level', 'N/A')}",
            ""
        ])
        
        # 技术分析
        if technical.get('analysis_text'):
            report_lines.extend([
                f"**📈 技术分析** (评分: {technical.get('score', 'N/A')}分)",
                f"{technical['analysis_text']}",
                ""
            ])
            
            # 关键价位
            key_levels = technical.get('key_levels', {})
            if key_levels.get('support') or key_levels.get('resistance'):
                report_lines.extend([
                    "**关键价位**:",
                    f"- 支撑位: {key_levels.get('support', 'N/A')}元",
                    f"- 阻力位: {key_levels.get('resistance', 'N/A')}元",
                    ""
                ])
        
        # 基本面分析
        if fundamental.get('analysis_text'):
            report_lines.extend([
                f"**💰 基本面分析** (评分: {fundamental.get('score', 'N/A')}分)",
                f"{fundamental['analysis_text']}",
                ""
            ])
            
            # 估值水平
            if fundamental.get('valuation_level'):
                report_lines.extend([
                    f"**估值水平**: {fundamental['valuation_level']}",
                    ""
                ])
        
        # 情绪分析部分已完全移除
        
        # 新闻分析
        news_analysis = claude_result.get('news_analysis', {})
        if news_analysis.get('analysis_text'):
            report_lines.extend([
                f"**📰 新闻面分析** (评分: {news_analysis.get('score', 'N/A')}分)",
                f"{news_analysis['analysis_text']}",
                ""
            ])
            
            # 显示关键事件
            if news_analysis.get('key_events'):
                report_lines.extend([
                    "**重要事件**:",
                    *[f"- {event}" for event in news_analysis['key_events'][:3]],
                    ""
                ])
        
        # 交易建议
        if trading:
            report_lines.extend([
                "**🎯 操作建议**",
                f"- **建议操作**: {trading.get('action', 'N/A')}",
                f"- **目标价位**: {trading.get('target_price', 'N/A')}元",
                f"- **止损价位**: {trading.get('stop_loss', 'N/A')}元",
                f"- **建议仓位**: {trading.get('position_size', 'N/A')}",
                f"- **持有期**: {trading.get('holding_period', 'N/A')}",
                ""
            ])
            
            if trading.get('rationale'):
                report_lines.extend([
                    f"**操作理由**: {trading['rationale']}",
                    ""
                ])
        
        # 风险提示
        if risk.get('main_risks'):
            report_lines.extend([
                "**⚠️ 风险提示**",
                *[f"- {risk_item}" for risk_item in risk['main_risks']],
                ""
            ])
    
    def _add_quantitative_analysis_section(self, report_lines: List[str], stock: Dict):
        """添加量化分析部分（Claude分析不可用时）"""
        report_lines.extend([
            "**📊 量化分析**",
            f"- **量化评分**: {stock.get('comprehensive_score', 0):.1f}分",
            f"- **策略验证**: {', '.join(stock.get('strategies', []))}",
            f"- **收盘价**: {stock.get('close_price', 'N/A')}元",
            f"- **涨跌幅**: {stock.get('price_change_pct', 0):.2f}%",
            "",
            "**注意**: 该股票未进行Claude详细分析，仅提供量化指标参考。",
            ""
        ])
    
    def _save_reports(self, report_content: str, analysis_date: str):
        """保存报告文件"""
        try:
            # 创建报告目录
            reports_dir = self.project_root / "reports" / "ai_enhanced" 
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存详细报告
            date_str = analysis_date.replace('-', '')
            version_suffix = "_V31" if self.scoring_version == "v3.1" else ""
            report_path = reports_dir / f"AI增强选股报告_{date_str}{version_suffix}.md"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"AI增强报告已保存到: {report_path}")
            
        except Exception as e:
            logger.error(f"保存报告失败: {e}")
    
    def _generate_summary_stats(self, enhanced_stocks: List[Dict]) -> Dict:
        """生成统计摘要"""
        if not enhanced_stocks:
            return {}
        
        claude_analyzed = [s for s in enhanced_stocks if s.get('claude_analysis') and 'error' not in s['claude_analysis']]
        
        if claude_analyzed:
            claude_scores = [s['claude_analysis']['overall_score'] for s in claude_analyzed]
            avg_claude_score = np.mean(claude_scores)
        else:
            avg_claude_score = 0
        
        return {
            'total_stocks': len(enhanced_stocks),
            'claude_analyzed_count': len(claude_analyzed),
            'avg_claude_score': round(avg_claude_score, 1),
            'success_rate': len(claude_analyzed) / len(enhanced_stocks) if enhanced_stocks else 0,
            'buy_recommendations': sum(1 for s in claude_analyzed 
                                    if s['claude_analysis'].get('trading_recommendation', {}).get('action') in ['买入', 'BUY'])
        }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI增强每日选股报告生成器 - 支持并行处理")
    parser.add_argument("--date", type=str, default=None, 
                       help="分析日期 (YYYY-MM-DD，默认为今天)")
    parser.add_argument("--config", type=str, default="config.json",
                       help="配置文件路径")
    parser.add_argument("--max-workers", type=int, default=None,
                       help="最大并行线程数 (默认为8或CPU核数)")
    parser.add_argument("--scoring-version", choices=["v3", "v3.1"], default="v3",
                       help="量化评分版本: v3(标准) 或 v3.1(优化权重版)")
    parser.add_argument("--verbose", action="store_true",
                       help="显示详细日志")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print(f"🚀 启动AI增强选股分析系统 (并行版)")
    print(f"📅 分析日期: {args.date or '今日'}")
    print(f"📊 评分版本: {args.scoring_version.upper()}")
    print(f"💻 并行线程: {args.max_workers or '自动检测'}")
    print(f"⚙️  并行优化: 支持重试机制和API频率控制")
    
    # 创建报告生成器
    generator = AIEnhancedDailyReport(args.config, max_workers=args.max_workers, scoring_version=args.scoring_version)
    
    # 生成报告 - 分析所有推荐股票
    print(f"🔄 开始并行分析...\n")
    result = generator.generate_daily_report(args.date)
    
    if result.get("success"):
        print(f"\n🎉 AI增强选股报告生成成功！")
        print(f"📅 分析日期: {result['analysis_date']}")
        print(f"📈 分析股票: {result['total_stocks']}只")
        print(f"🤖 Claude详细分析: {result['detailed_analysis_count']}只")
        
        # 显示并行处理统计
        if result.get('stats'):
            stats = result['stats']
            total_time = stats['end_time'] - stats['start_time']
            print(f"\n🚀 并行处理统计:")
            print(f"  ✅ 成功分析: {stats['success_count']}只")
            print(f"  ❌ 失败分析: {stats['failed_count']}只")
            print(f"  🔄 重试次数: {stats['retry_count']}次")
            print(f"  ⏱️  总耗时: {total_time:.1f}秒")
            print(f"  ⚡ 平均每股: {total_time/result['total_stocks']:.1f}秒")
            success_rate = stats['success_count'] / stats['total_stocks'] if stats['total_stocks'] > 0 else 0
            print(f"  🎯 成功率: {success_rate:.1%}")
        
        if result.get('summary'):
            summary = result['summary']
            print(f"\n💡 分析结果:")
            print(f"  🔥 买入建议: {summary.get('buy_recommendations', 0)}只")
            print(f"  🎯 Claude平均评分: {summary.get('avg_claude_score', 0)}分")
        
        print(f"\n📄 报告已保存在 reports/ai_enhanced/ 目录")
    else:
        print(f"\n❌ 报告生成失败: {result.get('error', '未知错误')}")


if __name__ == "__main__":
    main()