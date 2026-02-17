#!/usr/bin/env python3
"""
改进的中国股票分析器
集成量化选股系统和AI分析，提供统一的分析接口
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

# 添加路径
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent))
sys.path.append(str(current_dir.parent / "core"))

from china_trading_agents import ChinaTradingAgents
from claude_config import ClaudeConfig, create_claude_trading_config
from report_parser import ReportParser

logger = logging.getLogger(__name__)

class ChinaStockAnalyzer:
    """中国股票综合分析器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化分析器"""
        self.config = config or create_claude_trading_config("balanced")
        self.china_ta = None
        self.report_parser = ReportParser()
        
        # 初始化TradingAgents
        self._initialize_trading_agents()
    
    def _initialize_trading_agents(self):
        """初始化TradingAgents"""
        try:
            self.china_ta = ChinaTradingAgents(self.config)
            logger.info("TradingAgents初始化成功")
        except Exception as e:
            logger.warning(f"TradingAgents初始化失败: {e}")
            self.china_ta = None
    
    def analyze_daily_report(self, analysis_date: str, top_n: int = -1) -> Dict[str, Any]:
        """分析每日选股报告"""
        try:
            # 解析选股报告
            report_data = self._load_daily_report(analysis_date)
            if not report_data:
                return {"error": f"无法加载{analysis_date}的选股报告"}
            
            # 确定要分析的股票
            stocks_to_analyze = report_data['stocks']
            if top_n > 0:
                stocks_to_analyze = stocks_to_analyze[:top_n]
                logger.info(f"限制分析前{top_n}只股票")
            else:
                logger.info(f"分析全部{len(stocks_to_analyze)}只股票")
            
            # 执行AI分析
            ai_results = {}
            successful_analyses = 0
            failed_analyses = 0
            
            if self.china_ta:
                for stock in stocks_to_analyze:
                    try:
                        ai_result = self.china_ta.analyze_stock(
                            stock.code, 
                            analysis_date, 
                            stock.name
                        )
                        
                        if "error" not in ai_result:
                            ai_results[stock.code] = ai_result
                            successful_analyses += 1
                        else:
                            logger.warning(f"股票{stock.code}分析失败: {ai_result['error']}")
                            ai_results[stock.code] = {"error": ai_result["error"]}
                            failed_analyses += 1
                            
                    except Exception as e:
                        logger.error(f"分析股票{stock.code}时发生异常: {e}")
                        ai_results[stock.code] = {"error": str(e)}
                        failed_analyses += 1
            else:
                logger.warning("TradingAgents未初始化，跳过AI分析")
            
            # 生成增强报告
            enhanced_report = self._generate_enhanced_report(
                report_data, ai_results, analysis_date
            )
            
            # 统计结果
            buy_recommendations = sum(1 for r in ai_results.values() 
                                    if r.get("decision", "").upper() == "BUY")
            sell_warnings = sum(1 for r in ai_results.values() 
                              if r.get("decision", "").upper() == "SELL")
            
            return {
                "mode": "enhance",
                "analysis_date": analysis_date,
                "original_data": report_data,
                "ai_results": ai_results,
                "enhanced_report": enhanced_report,
                "summary": {
                    "total_stocks": len(stocks_to_analyze),
                    "total_analyzed": successful_analyses,
                    "failed_analyses": failed_analyses,
                    "buy_recommendations": buy_recommendations,
                    "sell_warnings": sell_warnings
                }
            }
            
        except Exception as e:
            logger.error(f"分析每日报告时发生错误: {e}")
            return {"error": str(e)}
    
    def _load_daily_report(self, analysis_date: str) -> Optional[Dict]:
        """加载每日选股报告"""
        # 获取项目根目录
        project_root = Path(__file__).parent.parent.parent
        date_str = analysis_date.replace('-', '')
        
        # 尝试多个可能的报告路径
        possible_paths = [
            project_root / "reports" / "daily_selection" / f"选股分析报告_{date_str}.md",
            project_root / "daily_result" / f"选股分析报告_{date_str}.md",
            project_root / f"选股分析报告_{date_str}.md"
        ]
        
        for report_path in possible_paths:
            if report_path.exists():
                logger.info(f"找到报告文件: {report_path}")
                return self.report_parser.parse_report(str(report_path))
        
        logger.error(f"未找到{analysis_date}的选股报告")
        return None
    
    def _generate_enhanced_report(self, original_data: Dict, ai_results: Dict, analysis_date: str) -> str:
        """生成增强报告"""
        lines = [
            "# 🤖 AI增强选股分析报告\n",
            "## 📊 分析概览",
            f"- **分析日期**: {analysis_date}",
            f"- **分析股票数**: {len(original_data['stocks'])}",
            f"- **AI分析引擎**: TradingAgents + 量化选股系统\n"
        ]
        
        # 分类显示结果
        if ai_results:
            successful_ai = {k: v for k, v in ai_results.items() if "error" not in v}
            failed_ai = {k: v for k, v in ai_results.items() if "error" in v}
            
            if successful_ai:
                lines.extend([
                    "## 🎯 AI成功分析股票\n"
                ])
                
                for i, stock in enumerate(original_data['stocks'], 1):
                    if stock.code in successful_ai:
                        ai_result = successful_ai[stock.code]
                        lines.extend(self._format_successful_analysis(stock, ai_result, i))
            
            if failed_ai:
                lines.extend([
                    "\n## 📊 量化推荐股票 (AI分析暂不可用)\n"
                ])
                
                for i, stock in enumerate(original_data['stocks'], 1):
                    if stock.code in failed_ai:
                        ai_result = failed_ai[stock.code]
                        lines.extend(self._format_failed_analysis(stock, ai_result, i))
        else:
            lines.extend([
                "## 📊 量化推荐股票 (AI分析暂不可用)\n"
            ])
            
            for i, stock in enumerate(original_data['stocks'], 1):
                lines.extend(self._format_quantitative_only(stock, i))
        
        return "\n".join(lines)
    
    def _format_successful_analysis(self, stock, ai_result: Dict, index: int) -> List[str]:
        """格式化成功的AI分析结果"""
        return [
            f"### {index}. {stock.code} - {stock.name}",
            "",
            "**基本信息**",
            f"- **所属行业**: {getattr(stock, 'industry', '未知')}",
            f"- **注册地区**: {getattr(stock, 'area', '未知')}",
            f"- **交易板块**: {getattr(stock, 'market', '未知')}",
            f"- **当前价格**: {stock.close_price}元",
            f"- **涨跌幅**: {stock.price_change_pct:+.2f}%",
            "",
            "**量化分析**",
            f"- **综合评分**: {stock.comprehensive_score:.1f}分",
            f"- **策略支持**: {', '.join(stock.strategies)}",
            f"- **KDJ指标**: K={stock.kdj_k:.1f}, D={stock.kdj_d:.1f}, J={stock.kdj_j:.1f}",
            f"- **BBI指标**: {stock.bbi:.2f}",
            "",
            "**AI分析结果**",
            f"- **AI决策**: {ai_result.get('decision', 'HOLD')}",
            f"- **置信度**: {ai_result.get('confidence', 0.5):.1%}",
            f"- **市场分析**: {ai_result.get('market_analysis', '暂无')[:100]}...",
            f"- **风险评估**: {ai_result.get('risk_assessment', '暂无')[:100]}...",
            "",
            "---",
            ""
        ]
    
    def _format_failed_analysis(self, stock, ai_result: Dict, index: int) -> List[str]:
        """格式化失败的AI分析结果"""
        return [
            f"### {index}. {stock.code} - {stock.name}",
            "",
            "**基本信息**",
            f"- **所属行业**: {getattr(stock, 'industry', '未知')}",
            f"- **注册地区**: {getattr(stock, 'area', '未知')}",
            f"- **交易板块**: {getattr(stock, 'market', '未知')}",
            f"- **当前价格**: {stock.close_price}元",
            f"- **涨跌幅**: {stock.price_change_pct:+.2f}%",
            "",
            "**量化分析**",
            f"- **综合评分**: {stock.comprehensive_score:.1f}分",
            f"- **策略支持**: {', '.join(stock.strategies)}",
            f"- **KDJ指标**: K={stock.kdj_k:.1f}, D={stock.kdj_d:.1f}, J={stock.kdj_j:.1f}",
            f"- **BBI指标**: {stock.bbi:.2f}",
            "",
            "**AI分析状态**",
            "- **状态**: 分析失败",
            f"- **原因**: {ai_result.get('error', '未知错误')}",
            "- **建议**: 基于量化指标，该股票仍具备投资价值",
            "",
            "---",
            ""
        ]
    
    def _format_quantitative_only(self, stock, index: int) -> List[str]:
        """格式化仅量化分析的结果"""
        return [
            f"### {index}. {stock.code} - {stock.name}",
            "",
            "**基本信息**",
            f"- **所属行业**: {getattr(stock, 'industry', '未知')}",
            f"- **注册地区**: {getattr(stock, 'area', '未知')}",
            f"- **交易板块**: {getattr(stock, 'market', '未知')}",
            f"- **当前价格**: {stock.close_price}元",
            f"- **涨跌幅**: {stock.price_change_pct:+.2f}%",
            "",
            "**量化分析**",
            f"- **综合评分**: {stock.comprehensive_score:.1f}分",
            f"- **策略支持**: {', '.join(stock.strategies)}",
            f"- **KDJ指标**: K={stock.kdj_k:.1f}, D={stock.kdj_d:.1f}, J={stock.kdj_j:.1f}",
            f"- **BBI指标**: {stock.bbi:.2f}",
            "",
            "**AI分析状态**",
            "- **状态**: AI分析服务暂不可用",
            "- **建议**: 基于量化指标进行投资决策",
            "",
            "---",
            ""
        ]