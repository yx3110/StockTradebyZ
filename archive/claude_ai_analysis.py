#!/usr/bin/env python3
"""
使用Claude API直接进行股票AI分析
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from anthropic import Anthropic

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class ClaudeStockAnalyzer:
    """Claude股票分析器"""
    
    def __init__(self, api_key: str = None):
        """初始化分析器"""
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            # 从配置文件读取
            config_path = Path('config.json')
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.api_key = config.get('anthropic', {}).get('api_key')
        
        if not self.api_key:
            raise ValueError("未找到Anthropic API密钥")
        
        self.client = Anthropic(api_key=self.api_key)
        logger.info("Claude API客户端初始化成功")
    
    def analyze_stock(self, stock_code: str, stock_info: Dict[str, Any]) -> Dict[str, Any]:
        """分析单只股票"""
        prompt = f"""你是一位专业的中国A股投资分析师。请分析以下股票的投资价值：

股票代码：{stock_code}
股票名称：{stock_info.get('name', '未知')}
行业：{stock_info.get('industry', '未知')}
选中策略：{stock_info.get('strategies', [])}
综合评分：{stock_info.get('score', 0)}

技术指标：
- 收盘价：{stock_info.get('close', 0)}元
- 涨跌幅：{stock_info.get('change_pct', 0)}%
- KDJ: K={stock_info.get('k', 0)}, D={stock_info.get('d', 0)}, J={stock_info.get('j', 0)}
- BBI: {stock_info.get('bbi', 0)}

请提供：
1. 技术面分析（20字以内）
2. 投资建议（买入/持有/卖出）
3. 主要风险（20字以内）
4. 目标价位预测
5. 信心评分（0-100）

请用JSON格式返回，包含：
{{
    "technical_analysis": "技术面分析",
    "recommendation": "BUY/HOLD/SELL",
    "main_risk": "主要风险",
    "target_price": 数字,
    "confidence": 数字
}}"""

        try:
            response = self.client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=500,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # 解析响应
            content = response.content[0].text
            # 尝试提取JSON
            import re
            json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                logger.warning(f"无法解析{stock_code}的AI响应")
                return {
                    "technical_analysis": "分析失败",
                    "recommendation": "HOLD",
                    "main_risk": "数据不足",
                    "target_price": stock_info.get('close', 0),
                    "confidence": 50
                }
                
        except Exception as e:
            logger.error(f"分析{stock_code}时出错: {e}")
            return {
                "technical_analysis": "分析错误",
                "recommendation": "HOLD",
                "main_risk": str(e)[:20],
                "target_price": stock_info.get('close', 0),
                "confidence": 0
            }
    
    def batch_analyze(self, stocks_data: List[Dict[str, Any]], max_stocks: int = 10) -> List[Dict[str, Any]]:
        """批量分析股票"""
        results = []
        total = min(len(stocks_data), max_stocks)
        
        logger.info(f"开始AI分析 {total} 只股票")
        
        for i, stock_data in enumerate(stocks_data[:total]):
            stock_code = stock_data.get('stock_code', '')
            logger.info(f"分析进度: {i+1}/{total} - {stock_code}")
            
            ai_result = self.analyze_stock(stock_code, stock_data)
            
            # 合并结果
            enhanced_data = {**stock_data, **ai_result}
            results.append(enhanced_data)
        
        return results
    
    def generate_ai_report(self, analysis_results: List[Dict[str, Any]], report_date: str) -> str:
        """生成AI增强报告"""
        report = f"""# 🤖 AI增强版选股分析报告

## 📊 报告概览
- **分析日期**: {report_date}
- **AI分析模型**: Claude 3.5 Haiku
- **分析股票数**: {len(analysis_results)}
- **分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🎯 AI推荐汇总

### 强烈推荐买入（AI信心>80）
"""
        # 按AI信心评分排序
        sorted_results = sorted(analysis_results, 
                              key=lambda x: x.get('confidence', 0), 
                              reverse=True)
        
        # 强烈推荐
        strong_buy = [s for s in sorted_results if s.get('recommendation') == 'BUY' and s.get('confidence', 0) > 80]
        if strong_buy:
            for stock in strong_buy[:5]:
                report += f"\n#### {stock['stock_code']} - {stock.get('name', '未知')}"
                report += f"\n- **AI信心**: {stock.get('confidence', 0)}%"
                report += f"\n- **技术分析**: {stock.get('technical_analysis', '')}"
                report += f"\n- **目标价**: {stock.get('target_price', 0):.2f}元"
                report += f"\n- **潜在收益**: {((stock.get('target_price', 0) / stock.get('close', 1) - 1) * 100):.1f}%"
                report += f"\n- **主要风险**: {stock.get('main_risk', '')}\n"
        else:
            report += "\n*暂无高信心推荐*\n"
        
        # 一般推荐
        report += "\n### 一般推荐买入（AI信心60-80）\n"
        normal_buy = [s for s in sorted_results if s.get('recommendation') == 'BUY' and 60 <= s.get('confidence', 0) <= 80]
        if normal_buy:
            for stock in normal_buy[:5]:
                report += f"- {stock['stock_code']} {stock.get('name', '')} - 信心{stock.get('confidence')}%\n"
        else:
            report += "*暂无*\n"
        
        # AI分析统计
        report += "\n## 📈 AI分析统计\n"
        buy_count = len([s for s in analysis_results if s.get('recommendation') == 'BUY'])
        hold_count = len([s for s in analysis_results if s.get('recommendation') == 'HOLD'])
        sell_count = len([s for s in analysis_results if s.get('recommendation') == 'SELL'])
        
        report += f"- **买入建议**: {buy_count}只 ({buy_count/len(analysis_results)*100:.1f}%)\n"
        report += f"- **持有建议**: {hold_count}只 ({hold_count/len(analysis_results)*100:.1f}%)\n"
        report += f"- **卖出建议**: {sell_count}只 ({sell_count/len(analysis_results)*100:.1f}%)\n"
        
        avg_confidence = sum(s.get('confidence', 0) for s in analysis_results) / len(analysis_results)
        report += f"- **平均信心度**: {avg_confidence:.1f}%\n"
        
        # 详细分析结果
        report += "\n## 📋 完整AI分析结果\n"
        report += "\n| 股票代码 | 股票名称 | 量化评分 | AI建议 | AI信心 | 目标价 | 技术分析 | 主要风险 |\n"
        report += "|---------|---------|---------|--------|--------|--------|----------|----------|\n"
        
        for stock in sorted_results[:20]:  # 显示前20只
            report += f"| {stock['stock_code']} "
            report += f"| {stock.get('name', '-')} "
            report += f"| {stock.get('score', 0):.1f} "
            report += f"| {stock.get('recommendation', '-')} "
            report += f"| {stock.get('confidence', 0)}% "
            report += f"| {stock.get('target_price', 0):.2f} "
            report += f"| {stock.get('technical_analysis', '-')[:15]} "
            report += f"| {stock.get('main_risk', '-')[:15]} |\n"
        
        if len(sorted_results) > 20:
            report += f"\n*... 还有{len(sorted_results)-20}只股票未显示*\n"
        
        report += "\n## ⚠️ 风险提示\n"
        report += "- AI分析仅供参考，不构成投资建议\n"
        report += "- 投资有风险，入市需谨慎\n"
        report += "- 建议结合基本面和市场环境综合判断\n"
        
        return report


def main():
    """主函数"""
    logger.info("开始生成AI增强报告")
    
    # 1. 读取今日选股报告数据
    report_date = datetime.now().strftime('%Y-%m-%d')
    report_date_fmt = report_date.replace('-', '')
    
    # 从生成的报告中提取股票信息（这里简化处理）
    stocks_to_analyze = [
        {"stock_code": "300401", "name": "花园生物", "score": 83.80, "close": 15.87, "j": -6.09, "strategies": ["少妇战法", "填坑战法"]},
        {"stock_code": "000528", "name": "柳工", "score": 82.87, "close": 11.02, "j": -0.99, "strategies": ["少妇战法", "填坑战法"]},
        {"stock_code": "002633", "name": "申科股份", "score": 81.02, "close": 16.57, "j": -8.93, "strategies": ["少妇战法", "填坑战法"]},
        {"stock_code": "000789", "name": "万年青", "score": 78.24, "close": 6.11, "j": -6.28, "strategies": ["少妇战法", "TePu战法"]},
        {"stock_code": "000401", "name": "冀东水泥", "score": 78.24, "close": 4.87, "j": -10.8, "strategies": ["少妇战法", "TePu战法"]},
        {"stock_code": "601179", "name": "中国西电", "score": 77.31, "close": 6.55, "j": -7.75, "strategies": ["少妇战法", "TePu战法"]},
        {"stock_code": "600563", "name": "法拉电子", "score": 76.39, "close": 109.0, "j": -2.36, "strategies": ["少妇战法", "TePu战法"]},
        {"stock_code": "002271", "name": "东方雨虹", "score": 76.39, "close": 11.93, "j": -1.43, "strategies": ["少妇战法", "TePu战法"]},
        {"stock_code": "603338", "name": "浙江鼎力", "score": 76.39, "close": 48.91, "j": -12.42, "strategies": ["少妇战法", "填坑战法"]},
        {"stock_code": "515680", "name": "央企创新ETF基金", "score": 76.39, "close": 1.52, "j": -10.81, "strategies": ["少妇战法", "TePu战法"]}
    ]
    
    # 2. 初始化AI分析器
    try:
        analyzer = ClaudeStockAnalyzer()
        
        # 3. 批量AI分析
        ai_results = analyzer.batch_analyze(stocks_to_analyze, max_stocks=10)
        
        # 4. 生成AI报告
        ai_report = analyzer.generate_ai_report(ai_results, report_date)
        
        # 5. 保存报告
        output_dir = Path('reports')
        output_dir.mkdir(exist_ok=True)
        
        report_path = output_dir / f'AI增强选股报告_{report_date_fmt}.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(ai_report)
        
        logger.info(f"AI报告已保存: {report_path}")
        
        # 6. 打印报告预览
        print("\n" + "="*60)
        print("AI报告预览：")
        print("="*60)
        print(ai_report[:1000] + "...")
        
    except Exception as e:
        logger.error(f"生成AI报告失败: {e}")
        raise


if __name__ == '__main__':
    main()