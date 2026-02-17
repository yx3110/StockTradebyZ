#!/usr/bin/env python3
"""
选股报告解析器
解析每日选股报告，提取股票信息用于TradingAgents分析
"""

import re
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

@dataclass
class StockInfo:
    """股票信息数据类"""
    code: str
    name: str
    exchange: str
    board: str
    industry: str
    area: str
    close_price: float
    price_change: float
    price_change_pct: float
    volume: int
    volatility: float
    kdj_k: float
    kdj_d: float
    kdj_j: float
    bbi: float
    macd_dif: float
    strategies: List[str]
    strategy_count: int
    comprehensive_score: float
    is_multi_strategy: bool

class ReportParser:
    """选股报告解析器"""
    
    def __init__(self):
        self.pattern_stock_section = re.compile(r'### \d+\. (\d+) - (.+?)\n(.*?)(?=### \d+\.|---|\n## |$)', re.DOTALL)
        self.pattern_multi_strategy = re.compile(r'### \d+个策略选中的股票 \((\d+)只\)\n(.+?)\n', re.DOTALL)
    
    def parse_report(self, report_path: str) -> Dict[str, Any]:
        """解析选股报告文件"""
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析基本信息
            analysis_date = self._extract_analysis_date(content)
            multi_strategy_stocks = self._extract_multi_strategy_stocks(content)
            stock_list = self._extract_stock_details(content)
            
            return {
                'analysis_date': analysis_date,
                'multi_strategy_stocks': multi_strategy_stocks,
                'total_stocks': len(stock_list),
                'stocks': stock_list
            }
        except Exception as e:
            print(f"解析报告出错: {e}")
            return None
    
    def _extract_analysis_date(self, content: str) -> str:
        """提取分析日期"""
        match = re.search(r'- \*\*分析日期\*\*: (\d{4}-\d{2}-\d{2})', content)
        return match.group(1) if match else ""
    
    def _extract_multi_strategy_stocks(self, content: str) -> List[str]:
        """提取多策略选中的股票代码"""
        multi_codes = []
        matches = self.pattern_multi_strategy.findall(content)
        for count, codes_text in matches:
            if int(count) > 0:
                codes = [code.strip() for code in codes_text.split(',')]
                multi_codes.extend(codes)
        return multi_codes
    
    def _extract_stock_details(self, content: str) -> List[StockInfo]:
        """提取股票详细信息"""
        stocks = []
        matches = self.pattern_stock_section.findall(content)
        
        for code, name, details in matches:
            try:
                stock_info = self._parse_stock_details(code, name, details)
                if stock_info:
                    stocks.append(stock_info)
            except Exception as e:
                print(f"解析股票 {code} 失败: {e}")
                continue
        
        return stocks
    
    def _parse_stock_details(self, code: str, name: str, details: str) -> StockInfo:
        """解析单只股票的详细信息"""
        # 基本信息
        exchange_match = re.search(r'- \*\*交易所板块\*\*: (.+)', details)
        industry_match = re.search(r'- \*\*所属行业\*\*: (.+)', details)
        area_match = re.search(r'- \*\*注册地\*\*: (.+)', details)
        
        # 市场表现
        close_price_match = re.search(r'- \*\*收盘价\*\*: ([\d.]+)元', details)
        price_change_match = re.search(r'- \*\*涨跌幅\*\*: ([+-]?[\d.]+)元 \(([+-]?[\d.]+)%\)', details)
        volume_match = re.search(r'- \*\*成交量\*\*: ([\d,]+)手', details)
        volatility_match = re.search(r'- \*\*波动率\*\*: ([\d.]+)%', details)
        
        # 技术指标
        kdj_match = re.search(r'- \*\*KDJ\*\*: K=([\d.-]+), D=([\d.-]+), J=([\d.-]+)', details)
        bbi_match = re.search(r'- \*\*BBI\*\*: ([\d.-]+)', details)
        macd_match = re.search(r'- \*\*MACD DIF\*\*: ([\d.-]+)', details)
        
        # 策略信息
        strategy_count_match = re.search(r'- \*\*通过策略数\*\*: (\d+)个', details)
        strategies_match = re.search(r'- \*\*策略名称\*\*: (.+)', details)
        score_match = re.search(r'- \*\*综合评分\*\*: ([\d.]+)分', details)
        
        # 构建StockInfo对象
        stock_info = StockInfo(
            code=code,
            name=name,
            exchange=exchange_match.group(1) if exchange_match else "",
            board=exchange_match.group(1) if exchange_match else "",
            industry=industry_match.group(1) if industry_match else "",
            area=area_match.group(1) if area_match else "",
            close_price=float(close_price_match.group(1)) if close_price_match else 0.0,
            price_change=float(price_change_match.group(1)) if price_change_match else 0.0,
            price_change_pct=float(price_change_match.group(2)) if price_change_match else 0.0,
            volume=int(volume_match.group(1).replace(',', '')) if volume_match else 0,
            volatility=float(volatility_match.group(1)) if volatility_match else 0.0,
            kdj_k=float(kdj_match.group(1)) if kdj_match else 0.0,
            kdj_d=float(kdj_match.group(2)) if kdj_match else 0.0,
            kdj_j=float(kdj_match.group(3)) if kdj_match else 0.0,
            bbi=float(bbi_match.group(1)) if bbi_match else 0.0,
            macd_dif=float(macd_match.group(1)) if macd_match else 0.0,
            strategies=strategies_match.group(1).split(', ') if strategies_match else [],
            strategy_count=int(strategy_count_match.group(1)) if strategy_count_match else 1,
            comprehensive_score=float(score_match.group(1)) if score_match else 0.0,
            is_multi_strategy=int(strategy_count_match.group(1)) > 1 if strategy_count_match else False
        )
        
        return stock_info
    
    def get_top_stocks(self, stocks: List[StockInfo], top_n: int = 10) -> List[StockInfo]:
        """获取评分最高的前N只股票"""
        return sorted(stocks, key=lambda x: x.comprehensive_score, reverse=True)[:top_n]
    
    def get_multi_strategy_stocks(self, stocks: List[StockInfo]) -> List[StockInfo]:
        """获取多策略选中的股票"""
        return [stock for stock in stocks if stock.is_multi_strategy]
    
    def export_to_json(self, data: Dict[str, Any], output_path: str):
        """导出数据为JSON格式"""
        # 转换StockInfo对象为字典
        if 'stocks' in data:
            data['stocks'] = [stock.__dict__ for stock in data['stocks']]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 测试解析器
    parser = ReportParser()
    report_path = "../reports/daily_selection/选股分析报告_20250731.md"
    
    if Path(report_path).exists():
        result = parser.parse_report(report_path)
        if result:
            print(f"解析成功！")
            print(f"分析日期: {result['analysis_date']}")
            print(f"总股票数: {result['total_stocks']}")
            print(f"多策略股票: {len(parser.get_multi_strategy_stocks(result['stocks']))}")
            
            # 导出为JSON
            parser.export_to_json(result, "../TA_integration/temp/parsed_report.json")
            print("已导出为JSON格式")
    else:
        print(f"报告文件不存在: {report_path}")