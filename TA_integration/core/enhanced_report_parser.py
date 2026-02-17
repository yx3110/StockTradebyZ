#!/usr/bin/env python3
"""
增强版报告解析器
专门用于解析中国股票选股报告，提取详细的股票信息和技术指标
"""

import re
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, NamedTuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class StockInfo(NamedTuple):
    """股票信息数据结构"""
    rank: int
    code: str
    name: str
    industry: str
    area: str
    market: str
    close_price: float
    price_change_pct: float
    comprehensive_score: float
    strategies: List[str]
    kdj_k: float
    kdj_d: float
    kdj_j: float
    bbi: float
    volume: Optional[int] = None
    market_cap: Optional[float] = None

class EnhancedReportParser:
    """增强版报告解析器"""
    
    def __init__(self):
        """初始化解析器"""
        self.stock_pattern_mappings = {
            # 股票基本信息模式
            'stock_header': r'### (\d+)\. (\d{6}) - ([^\n]+)',
            'industry': r'- \*\*所属行业\*\*[：:]\s*([^\n]+)',
            'area': r'- \*\*注册地区\*\*[：:]\s*([^\n]+)', 
            'market': r'- \*\*交易板块\*\*[：:]\s*([^\n]+)',
            'price': r'- \*\*当前价格\*\*[：:]\s*([\d.]+)元',
            'price_change': r'- \*\*涨跌幅\*\*[：:]\s*([+-]?[\d.]+)%',
            
            # 量化分析数据
            'score': r'- \*\*综合评分\*\*[：:]\s*([\d.]+)分',
            'strategies': r'- \*\*策略支持\*\*[：:]\s*([^\n]+)',
            'kdj': r'- \*\*KDJ指标\*\*[：:]\s*K=([\d.-]+),\s*D=([\d.-]+),\s*J=([\d.-]+)',
            'bbi': r'- \*\*BBI指标\*\*[：:]\s*([\d.-]+)',
            
            # 额外技术指标
            'macd': r'- \*\*MACD\*\*[：:]\s*([^\n]+)',
            'rsi': r'- \*\*RSI\*\*[：:]\s*([\d.-]+)',
            'volume': r'- \*\*成交量\*\*[：:]\s*([\d.]+)([万亿]?手?)',
            'market_cap': r'- \*\*市值\*\*[：:]\s*([\d.]+)([万亿]?)元',
        }
        
        # 策略名称映射
        self.strategy_mapping = {
            '少负战法': 'BBIKDJSelector',
            '补票战法': 'BBIShortLongSelector', 
            'TePu战法': 'BreakoutVolumeKDJSelector',
            '填坑战法': 'PeakKDJSelector'
        }
    
    def parse_report(self, report_path: str) -> Dict[str, Any]:
        """解析选股报告"""
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"解析报告文件: {report_path}")
            
            # 提取报告元信息
            report_info = self._extract_report_info(content)
            
            # 提取股票信息
            stocks = self._extract_stocks(content)
            
            logger.info(f"成功解析 {len(stocks)} 只股票")
            
            return {
                'report_info': report_info,
                'stocks': stocks,
                'total_count': len(stocks),
                'parsed_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"解析报告失败: {e}")
            return {'stocks': [], 'total_count': 0, 'error': str(e)}
    
    def _extract_report_info(self, content: str) -> Dict[str, Any]:
        """提取报告基本信息"""
        info = {}
        
        # 提取报告标题和日期
        title_match = re.search(r'# ([^\n]+)', content)
        if title_match:
            info['title'] = title_match.group(1).strip()
        
        # 提取分析日期
        date_patterns = [
            r'分析日期[：:]\s*(\d{4}-\d{2}-\d{2})',
            r'日期[：:]\s*(\d{4}-\d{2}-\d{2})',
            r'(\d{4}年\d{1,2}月\d{1,2}日)',
            r'选股分析报告_(\d{8})'
        ]
        
        for pattern in date_patterns:
            date_match = re.search(pattern, content)
            if date_match:
                date_str = date_match.group(1)
                try:
                    if '年' in date_str:
                        # 处理中文日期格式
                        date_str = re.sub(r'(\d{4})年(\d{1,2})月(\d{1,2})日', r'\1-\2-\3', date_str)
                    elif len(date_str) == 8:
                        # 处理YYYYMMDD格式
                        date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                    
                    info['analysis_date'] = date_str
                    break
                except:
                    continue
        
        # 提取股票总数
        count_patterns = [
            r'共筛选出\s*(\d+)\s*只股票',
            r'筛选股票数[：:]\s*(\d+)',
            r'分析股票数[：:]\s*(\d+)'
        ]
        
        for pattern in count_patterns:
            count_match = re.search(pattern, content)
            if count_match:
                info['total_stocks'] = int(count_match.group(1))
                break
        
        return info
    
    def _extract_stocks(self, content: str) -> List[StockInfo]:
        """提取股票详细信息"""
        stocks = []
        
        # 按股票分割内容 
        stock_sections = re.split(r'### \d+\.', content)[1:]  # 跳过第一个空部分
        
        for i, section in enumerate(stock_sections):
            try:
                # 重新添加被分割掉的标题标记
                section = f"### {i+1}." + section
                stock_info = self._parse_stock_section(section, i+1)
                if stock_info:
                    stocks.append(stock_info)
            except Exception as e:
                logger.warning(f"解析第{i+1}只股票失败: {e}")
                continue
        
        return stocks
    
    def _parse_stock_section(self, section: str, rank: int) -> Optional[StockInfo]:
        """解析单个股票信息段落"""
        try:
            # 提取基本信息
            header_match = re.search(self.stock_pattern_mappings['stock_header'], section)
            if not header_match:
                return None
            
            _, code, name = header_match.groups()
            
            # 提取详细信息
            industry = self._extract_field(section, 'industry') or '未知'
            area = self._extract_field(section, 'area') or '未知'
            market = self._extract_field(section, 'market') or '未知'
            
            # 提取价格信息
            close_price = float(self._extract_field(section, 'price') or 0)
            price_change_str = self._extract_field(section, 'price_change') or '0'
            price_change_pct = float(price_change_str.replace('+', ''))
            
            # 提取量化分析数据
            comprehensive_score = float(self._extract_field(section, 'score') or 70.0)
            
            strategies_str = self._extract_field(section, 'strategies') or ''
            strategies = [s.strip() for s in strategies_str.split(',') if s.strip()]
            
            # 提取KDJ指标
            kdj_match = re.search(self.stock_pattern_mappings['kdj'], section)
            if kdj_match:
                kdj_k = float(kdj_match.group(1))
                kdj_d = float(kdj_match.group(2))
                kdj_j = float(kdj_match.group(3))
            else:
                kdj_k = kdj_d = kdj_j = 0.0
            
            # 提取BBI指标
            bbi = float(self._extract_field(section, 'bbi') or 0.0)
            
            # 提取其他可选信息
            volume = self._parse_volume(self._extract_field(section, 'volume'))
            market_cap = self._parse_market_cap(self._extract_field(section, 'market_cap'))
            
            return StockInfo(
                rank=rank,
                code=code,
                name=name,
                industry=industry,
                area=area,
                market=market,
                close_price=close_price,
                price_change_pct=price_change_pct,
                comprehensive_score=comprehensive_score,
                strategies=strategies,
                kdj_k=kdj_k,
                kdj_d=kdj_d,
                kdj_j=kdj_j,
                bbi=bbi,
                volume=volume,
                market_cap=market_cap
            )
            
        except Exception as e:
            logger.warning(f"解析股票段落失败: {e}")
            return None
    
    def _extract_field(self, section: str, field_name: str) -> Optional[str]:
        """从段落中提取指定字段"""
        pattern = self.stock_pattern_mappings.get(field_name)
        if not pattern:
            return None
        
        match = re.search(pattern, section)
        return match.group(1) if match else None
    
    def _parse_volume(self, volume_str: Optional[str]) -> Optional[int]:
        """解析成交量"""
        if not volume_str:
            return None
        
        try:
            # 移除非数字字符，保留数字和小数点
            num_str = re.sub(r'[^\d.]', '', volume_str)
            volume = float(num_str)
            
            # 处理单位
            if '万' in volume_str:
                volume *= 10000
            elif '亿' in volume_str:
                volume *= 100000000
            
            return int(volume)
        except:
            return None
    
    def _parse_market_cap(self, cap_str: Optional[str]) -> Optional[float]:
        """解析市值"""
        if not cap_str:
            return None
        
        try:
            # 移除非数字字符，保留数字和小数点
            num_str = re.sub(r'[^\d.]', '', cap_str)
            cap = float(num_str)
            
            # 处理单位
            if '万' in cap_str:
                cap *= 10000
            elif '亿' in cap_str:
                cap *= 100000000
            
            return cap
        except:
            return None
    
    def to_dataframe(self, stocks: List[StockInfo]) -> pd.DataFrame:
        """转换为DataFrame格式"""
        if not stocks:
            return pd.DataFrame()
        
        # 转换为字典列表
        data = []
        for stock in stocks:
            stock_dict = stock._asdict()
            # 将策略列表转换为字符串
            stock_dict['strategies'] = ','.join(stock.strategies)
            data.append(stock_dict)
        
        return pd.DataFrame(data)
    
    def export_to_csv(self, stocks: List[StockInfo], output_path: str):
        """导出为CSV文件"""
        df = self.to_dataframe(stocks)
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"股票数据已导出到: {output_path}")
    
    def get_stocks_by_strategy(self, stocks: List[StockInfo], strategy_name: str) -> List[StockInfo]:
        """根据策略筛选股票"""
        return [stock for stock in stocks if strategy_name in stock.strategies]
    
    def get_top_stocks(self, stocks: List[StockInfo], top_n: int = 10) -> List[StockInfo]:
        """获取评分最高的前N只股票"""
        sorted_stocks = sorted(stocks, key=lambda x: x.comprehensive_score, reverse=True)
        return sorted_stocks[:top_n]


def test_enhanced_parser():
    """测试增强版解析器"""
    parser = EnhancedReportParser()
    
    # 查找最新的报告文件
    from pathlib import Path
    
    report_dirs = [
        Path("reports/daily_selection"),
        Path("daily_result")
    ]
    
    latest_report = None
    for report_dir in report_dirs:
        if report_dir.exists():
            report_files = list(report_dir.glob("选股分析报告_*.md"))
            if report_files:
                latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
                break
    
    if latest_report:
        print(f"🧪 测试解析报告: {latest_report}")
        result = parser.parse_report(str(latest_report))
        
        if result.get('stocks'):
            print(f"✅ 成功解析 {len(result['stocks'])} 只股票")
            
            # 显示前5只股票信息
            for i, stock in enumerate(result['stocks'][:5], 1):
                print(f"\n{i}. {stock.code} - {stock.name}")
                print(f"   综合评分: {stock.comprehensive_score:.1f}")
                print(f"   策略: {', '.join(stock.strategies)}")
                print(f"   当前价格: {stock.close_price}元")
        else:
            print("❌ 解析失败或无股票数据")
    else:
        print("❌ 未找到报告文件")


if __name__ == "__main__":
    test_enhanced_parser()