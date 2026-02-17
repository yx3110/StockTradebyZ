#!/usr/bin/env python3
"""
中国市场TradingAgents适配器
修改TradingAgents以支持中国股票分析
"""

import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# 添加TradingAgents到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
trading_agents_path = os.path.join(project_root, 'TradingAgents')
sys.path.append(trading_agents_path)

try:
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.agents.utils.agent_utils import Toolkit
    from langchain_openai import ChatOpenAI
except ImportError as e:
    print(f"请确保TradingAgents已正确安装: {e}")
    sys.exit(1)

try:
    from .china_stock_adapter import ChinaStockAdapter, ChinaMarketDataProvider
except ImportError:
    from china_stock_adapter import ChinaStockAdapter, ChinaMarketDataProvider

class ChinaTradingAgents:
    """中国市场TradingAgents"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化中国市场TradingAgents"""
        self.config = config or self._create_china_config()
        self.adapter = ChinaStockAdapter()
        self.data_provider = ChinaMarketDataProvider(self.adapter)
        
        # 初始化TradingAgents
        self.ta_graph = None
        self._initialize_ta_graph()
    
    def _create_china_config(self) -> Dict[str, Any]:
        """创建中国市场配置"""
        config = DEFAULT_CONFIG.copy()
        config.update({
            "market": "china",
            "currency": "CNY",
            "trading_hours": {
                "open": "09:30",
                "close": "15:00",
                "break_start": "11:30",
                "break_end": "13:00"
            },
            "trading_rules": {
                "price_limit": 0.10,  # 涨跌停10%
                "st_limit": 0.05,     # ST股票5%
                "trading_unit": 100,  # 手
                "t_plus": 1          # T+1
            },
            "online_tools": False,    # 使用离线数据
            "data_dir": "TA_integration/data"
        })
        return config
    
    def _initialize_ta_graph(self):
        """初始化TradingAgents图"""
        try:
            self.ta_graph = TradingAgentsGraph(
                debug=False,
                config=self.config
            )
            
            # 替换数据获取函数
            self._patch_data_functions()
            
        except Exception as e:
            print(f"初始化TradingAgents失败: {e}")
    
    def _patch_data_functions(self):
        """替换数据获取函数以支持中国股票"""
        if self.ta_graph and hasattr(self.ta_graph, 'toolkit'):
            # 替换价格数据获取函数
            original_get_yf_data = self.ta_graph.toolkit.get_YFin_data
            
            def get_china_stock_data(symbol, start_date, end_date):
                """获取中国股票数据"""
                return self.data_provider.get_stock_data_for_ta(symbol, start_date, end_date)
            
            def get_china_sentiment_data(symbol, curr_date, look_back_days=7):
                """获取中国股票情绪数据"""
                try:
                    # 导入中文情绪分析模块
                    import sys
                    from pathlib import Path
                    current_dir = Path(__file__).parent.parent
                    sys.path.append(str(current_dir / "data_sources"))
                    
                    from sentiment_integrator import get_china_stock_sentiment
                    return get_china_stock_sentiment(symbol, curr_date, look_back_days)
                except Exception as e:
                    return f"获取{symbol}中文情绪数据失败: {e}"
            
            # 替换数据获取函数
            self.ta_graph.toolkit.get_YFin_data = get_china_stock_data
            self.ta_graph.toolkit.get_YFin_data_online = get_china_stock_data
            
            # 替换情绪分析函数
            if hasattr(self.ta_graph.toolkit, 'get_reddit_stock_info'):
                self.ta_graph.toolkit.get_reddit_stock_info = get_china_sentiment_data
            if hasattr(self.ta_graph.toolkit, 'get_reddit_company_news'):
                self.ta_graph.toolkit.get_reddit_company_news = get_china_sentiment_data
    
    def analyze_stock(self, stock_code: str, analysis_date: str, stock_name: str = "") -> Dict[str, Any]:
        """分析中国股票"""
        if not self.ta_graph:
            return {"error": "TradingAgents未正确初始化"}
        
        try:
            # 确保数据已准备
            self._prepare_stock_data(stock_code, stock_name)
            
            # 运行TradingAgents分析
            final_state, decision = self.ta_graph.propagate(stock_code, analysis_date)
            
            # 解析结果
            result = self._parse_ta_result(final_state, decision, stock_code)
            return result
            
        except Exception as e:
            return {"error": f"分析股票 {stock_code} 失败: {e}"}
    
    def _prepare_stock_data(self, stock_code: str, stock_name: str = ""):
        """准备股票数据"""
        # 检查数据是否存在
        availability = self.adapter.check_data_availability(stock_code)
        
        if not availability['ta_data_exists']:
            # 转换数据
            success = self.adapter.convert_stock_data(stock_code, stock_name)
            if not success:
                raise ValueError(f"无法准备股票 {stock_code} 的数据")
    
    def _parse_ta_result(self, final_state: Dict, decision: str, stock_code: str) -> Dict[str, Any]:
        """解析TradingAgents结果"""
        return {
            "stock_code": stock_code,
            "analysis_date": final_state.get("trade_date", ""),
            "decision": decision,
            "confidence": self._calculate_confidence(final_state),
            "market_analysis": final_state.get("market_report", ""),
            "sentiment_analysis": final_state.get("sentiment_report", ""),
            "news_analysis": final_state.get("news_report", ""),
            "fundamental_analysis": final_state.get("fundamentals_report", ""),
            "investment_plan": final_state.get("investment_plan", ""),
            "risk_assessment": final_state.get("risk_assessment", ""),
            "bull_arguments": self._extract_bull_arguments(final_state),
            "bear_arguments": self._extract_bear_arguments(final_state),
            "final_recommendation": decision
        }
    
    def _calculate_confidence(self, final_state: Dict) -> float:
        """计算置信度"""
        # 基于各个分析师的一致性计算置信度
        # 这里是简化版本，实际可以更复杂
        return 0.75  # 默认75%置信度
    
    def _extract_bull_arguments(self, final_state: Dict) -> List[str]:
        """提取看涨论据"""
        bull_args = []
        
        # 从不同报告中提取积极因素
        market_report = final_state.get("market_report", "")
        if "上涨" in market_report or "买入" in market_report:
            bull_args.append("技术面显示上涨趋势")
        
        sentiment_report = final_state.get("sentiment_report", "")
        if "积极" in sentiment_report or "乐观" in sentiment_report:
            bull_args.append("市场情绪积极")
        
        return bull_args
    
    def _extract_bear_arguments(self, final_state: Dict) -> List[str]:
        """提取看跌论据"""
        bear_args = []
        
        # 从不同报告中提取消极因素
        market_report = final_state.get("market_report", "")
        if "下跌" in market_report or "卖出" in market_report:
            bear_args.append("技术面显示下跌风险")
        
        return bear_args
    
    def batch_analyze_stocks(self, stock_list: List[Dict[str, str]], analysis_date: str) -> Dict[str, Dict]:
        """批量分析股票"""
        results = {}
        
        for stock_info in stock_list:
            stock_code = stock_info['code']
            stock_name = stock_info.get('name', '')
            
            print(f"正在分析 {stock_code} - {stock_name}...")
            
            result = self.analyze_stock(stock_code, analysis_date, stock_name)
            results[stock_code] = result
        
        return results
    
    def create_enhanced_report(self, original_stocks: List[Dict], ta_results: Dict[str, Dict]) -> str:
        """创建增强的分析报告"""
        report_lines = [
            "# 🤖 AI增强选股分析报告\n",
            f"## 📊 分析概览",
            f"- **分析日期**: {datetime.now().strftime('%Y-%m-%d')}",
            f"- **分析股票数**: {len(ta_results)}",
            f"- **AI分析引擎**: TradingAgents + 量化选股系统\n"
        ]
        
        # 按决策分组
        buy_stocks = []
        hold_stocks = []
        sell_stocks = []
        error_stocks = []
        
        for stock_code, result in ta_results.items():
            if 'error' in result:
                error_stocks.append((stock_code, result))
                continue
                
            decision = result.get('decision', 'HOLD').upper()
            if 'BUY' in decision:
                buy_stocks.append((stock_code, result))
            elif 'SELL' in decision:
                sell_stocks.append((stock_code, result))
            else:
                hold_stocks.append((stock_code, result))
        
        # 添加推荐买入部分
        if buy_stocks:
            report_lines.append("## 🚀 AI推荐买入股票\n")
            for i, (stock_code, result) in enumerate(buy_stocks, 1):
                # 尝试匹配原始股票信息，支持字典和对象两种格式
                original_stock = None
                for s in original_stocks:
                    if hasattr(s, 'code'):  # StockInfo对象
                        if s.code == stock_code:
                            original_stock = s.__dict__
                            break
                    elif isinstance(s, dict) and s.get('code') == stock_code:  # 字典格式
                        original_stock = s
                        break
                
                if original_stock is None:
                    original_stock = {'name': '未找到', 'comprehensive_score': 0, 'strategies': []}
                
                report_lines.extend([
                    f"### {i}. {stock_code} - {original_stock.get('name', '未找到股票名称')}",
                    "",
                    "**基本信息**",
                    f"- **所属行业**: {original_stock.get('industry', '未知')}",
                    f"- **注册地区**: {original_stock.get('area', '未知')}",
                    f"- **交易板块**: {original_stock.get('board', '未知')}",
                    f"- **当前价格**: {original_stock.get('close_price', 0):.2f}元",
                    f"- **涨跌幅**: {original_stock.get('price_change_pct', 0):+.2f}%",
                    "",
                    "**量化分析**",
                    f"- **综合评分**: {original_stock.get('comprehensive_score', 0):.1f}分",
                    f"- **策略支持**: {', '.join(original_stock.get('strategies', []))}",
                    f"- **KDJ指标**: K={original_stock.get('kdj_k', 0):.1f}, D={original_stock.get('kdj_d', 0):.1f}, J={original_stock.get('kdj_j', 0):.1f}",
                    f"- **BBI指标**: {original_stock.get('bbi', 0):.2f}",
                    "",
                    "**AI智能分析**",
                    f"- **AI决策**: {result['decision']}",
                    f"- **置信度**: {result['confidence']:.1%}",
                    "",
                    "**AI分析摘要**:",
                    result.get('investment_plan', '')[:200] + "...",
                    "",
                    "**看涨理由**:",
                    "\n".join([f"- {arg}" for arg in result.get('bull_arguments', [])]),
                    "",
                    "---\n"
                ])
        
        # 添加需要关注部分
        if sell_stocks:
            report_lines.append("## ⚠️ AI建议谨慎的股票\n")
            for i, (stock_code, result) in enumerate(sell_stocks, 1):
                # 尝试匹配原始股票信息
                original_stock = None
                for s in original_stocks:
                    if hasattr(s, 'code'):  # StockInfo对象
                        if s.code == stock_code:
                            original_stock = s.__dict__
                            break
                    elif isinstance(s, dict) and s.get('code') == stock_code:  # 字典格式
                        original_stock = s
                        break
                
                if original_stock is None:
                    original_stock = {'name': '未找到', 'comprehensive_score': 0, 'strategies': []}
                
                report_lines.extend([
                    f"### {i}. {stock_code} - {original_stock.get('name', '未找到股票名称')}",
                    "",
                    "**基本信息**",
                    f"- **所属行业**: {original_stock.get('industry', '未知')}",
                    f"- **注册地区**: {original_stock.get('area', '未知')}",
                    f"- **交易板块**: {original_stock.get('board', '未知')}",
                    f"- **当前价格**: {original_stock.get('close_price', 0):.2f}元",
                    f"- **涨跌幅**: {original_stock.get('price_change_pct', 0):+.2f}%",
                    "",
                    "**量化分析**",
                    f"- **综合评分**: {original_stock.get('comprehensive_score', 0):.1f}分",
                    f"- **策略支持**: {', '.join(original_stock.get('strategies', []))}",
                    "",
                    "**AI智能分析**",
                    f"- **AI决策**: {result['decision']}",
                    f"- **置信度**: {result['confidence']:.1%}",
                    "",
                    "**AI风险警示**:",
                    result.get('investment_plan', '')[:200] + "...",
                    "",
                    "**看跌理由**:",
                    "\n".join([f"- {arg}" for arg in result.get('bear_arguments', [])]),
                    "",
                    "---\n"
                ])
        
        # 添加AI分析失败但仍需关注的股票
        if error_stocks:
            report_lines.append("## 📊 量化推荐股票 (AI分析暂不可用)\n")
            for i, (stock_code, result) in enumerate(error_stocks, 1):
                # 尝试匹配原始股票信息
                original_stock = None
                for s in original_stocks:
                    if hasattr(s, 'code'):  # StockInfo对象
                        if s.code == stock_code:
                            original_stock = s.__dict__
                            break
                    elif isinstance(s, dict) and s.get('code') == stock_code:  # 字典格式
                        original_stock = s
                        break
                
                if original_stock is None:
                    original_stock = {'name': '未找到', 'comprehensive_score': 0, 'strategies': []}
                
                report_lines.extend([
                    f"### {i}. {stock_code} - {original_stock.get('name', '未找到股票名称')}",
                    "",
                    "**基本信息**",
                    f"- **所属行业**: {original_stock.get('industry', '未知')}",
                    f"- **注册地区**: {original_stock.get('area', '未知')}",
                    f"- **交易板块**: {original_stock.get('board', '未知')}",
                    f"- **当前价格**: {original_stock.get('close_price', 0):.2f}元",
                    f"- **涨跌幅**: {original_stock.get('price_change_pct', 0):+.2f}%",
                    "",
                    "**量化分析**",
                    f"- **综合评分**: {original_stock.get('comprehensive_score', 0):.1f}分",
                    f"- **策略支持**: {', '.join(original_stock.get('strategies', []))}",
                    f"- **KDJ指标**: K={original_stock.get('kdj_k', 0):.1f}, D={original_stock.get('kdj_d', 0):.1f}, J={original_stock.get('kdj_j', 0):.1f}",
                    f"- **BBI指标**: {original_stock.get('bbi', 0):.2f}",
                    "",
                    "**AI分析状态**",
                    f"- **状态**: 分析失败",
                    f"- **原因**: {result.get('error', '未知错误')}",
                    f"- **建议**: 基于量化指标，该股票仍具备投资价值",
                    "",
                    "---\n"
                ])
        
        return "\n".join(report_lines)

class ChinaStockAnalyzer:
    """中国股票分析器 - 集成量化选股和AI分析"""
    
    def __init__(self, config=None):
        self.china_ta = ChinaTradingAgents(config=config)
        
    def analyze_daily_report(self, report_date: str, top_n: int = 10) -> Dict[str, Any]:
        """分析每日选股报告"""
        import sys
        import os
        from pathlib import Path
        
        # 添加项目根目录到路径
        current_dir = Path(__file__).parent.parent.parent
        sys.path.append(str(current_dir))
        
        from TA_integration.core.report_parser import ReportParser
        
        # 解析选股报告
        parser = ReportParser()
        # 修复报告路径 - 从项目根目录查找reports
        root_dir = Path(__file__).parent.parent.parent
        report_path = root_dir / f"reports/daily_selection/选股分析报告_{report_date.replace('-', '')}.md"
        
        if not os.path.exists(report_path):
            return {"error": f"选股报告不存在: {report_path}"}
        
        parsed_data = parser.parse_report(report_path)
        if not parsed_data:
            return {"error": "解析选股报告失败"}
        
        # 获取要分析的股票
        stocks_to_analyze = []
        
        if top_n == -1:
            # 分析全部股票，优先多策略，再按评分排序
            multi_strategy_stocks = parser.get_multi_strategy_stocks(parsed_data['stocks'])
            single_strategy_stocks = [s for s in parsed_data['stocks'] if not s.is_multi_strategy]
            sorted_single = sorted(single_strategy_stocks, 
                                 key=lambda x: x.comprehensive_score, reverse=True)
            stocks_to_analyze = multi_strategy_stocks + sorted_single
            print(f"将分析全部 {len(stocks_to_analyze)} 只股票")
        else:
            # 分析指定数量的股票
            # 优先分析多策略股票
            multi_strategy_stocks = parser.get_multi_strategy_stocks(parsed_data['stocks'])
            stocks_to_analyze.extend(multi_strategy_stocks)
            
            # 添加高分单策略股票
            remaining_count = top_n - len(multi_strategy_stocks)
            if remaining_count > 0:
                single_strategy_stocks = [s for s in parsed_data['stocks'] if not s.is_multi_strategy]
                top_single = sorted(single_strategy_stocks, 
                                  key=lambda x: x.comprehensive_score, reverse=True)[:remaining_count]
                stocks_to_analyze.extend(top_single)
        
        # 转换为AI分析所需格式
        stock_list = [{"code": s.code, "name": s.name} for s in stocks_to_analyze]
        
        # 运行AI分析
        ta_results = self.china_ta.batch_analyze_stocks(stock_list, report_date)
        
        # 生成增强报告
        enhanced_report = self.china_ta.create_enhanced_report(
            [s.__dict__ for s in stocks_to_analyze], 
            ta_results
        )
        
        return {
            "original_data": parsed_data,
            "ai_results": ta_results,
            "enhanced_report": enhanced_report,
            "summary": {
                "total_analyzed": len(ta_results),
                "buy_recommendations": len([r for r in ta_results.values() 
                                          if 'BUY' in r.get('decision', '').upper()]),
                "sell_warnings": len([r for r in ta_results.values() 
                                    if 'SELL' in r.get('decision', '').upper()])
            }
        }

if __name__ == "__main__":
    # 测试中国市场TradingAgents
    analyzer = ChinaStockAnalyzer()
    
    # 分析最新选股报告
    result = analyzer.analyze_daily_report("2025-07-31", top_n=5)
    
    if "error" not in result:
        print("AI增强分析完成！")
        print(f"总分析股票数: {result['summary']['total_analyzed']}")
        print(f"买入推荐: {result['summary']['buy_recommendations']}")
        print(f"谨慎提醒: {result['summary']['sell_warnings']}")
        
        # 保存增强报告
        with open("TA_integration/enhanced_report.md", "w", encoding="utf-8") as f:
            f.write(result['enhanced_report'])
        print("增强报告已保存到 TA_integration/enhanced_report.md")
    else:
        print(f"分析失败: {result['error']}")