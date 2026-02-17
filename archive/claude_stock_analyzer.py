#!/usr/bin/env python3
"""
Claude股票分析器
直接使用Claude API分析每日选股报告中的股票
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

try:
    from anthropic import Anthropic
except ImportError:
    print("请安装anthropic包: pip install anthropic")
    exit(1)

class ClaudeStockAnalyzer:
    """Claude股票分析器"""
    
    def __init__(self, api_key: str = None):
        """初始化分析器"""
        if not api_key:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                # 尝试从config.json读取
                try:
                    config_path = Path('TA_integration/config/config.json')
                    if config_path.exists():
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                            api_key = config.get('claude', {}).get('api_key')
                except:
                    pass
        
        if not api_key:
            raise ValueError("需要提供Claude API密钥")
        
        self.client = Anthropic(api_key=api_key)
        
    def parse_daily_report(self, date: str) -> List[Dict[str, Any]]:
        """解析每日选股报告"""
        report_path = Path(f"reports/daily_selection/选股分析报告_{date.replace('-', '')}.md")
        
        if not report_path.exists():
            print(f"未找到报告文件: {report_path}")
            return []
        
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 简单解析提取股票信息
        stocks = []
        lines = content.split('\n')
        current_stock = None
        
        for line in lines:
            line = line.strip()
            
            # 检测股票标题行 (例如: ### 1. 300401 - 花园生物)
            if line.startswith('###') and ' - ' in line and any(c.isdigit() for c in line):
                parts = line.split(' - ', 1)
                if len(parts) == 2:
                    # 提取股票代码
                    code_part = parts[0].split()[-1]  # 获取最后一个部分作为代码
                    name = parts[1]
                    
                    if current_stock:
                        stocks.append(current_stock)
                    
                    current_stock = {
                        'stock_code': code_part,
                        'stock_name': name,
                        'analysis_data': {}
                    }
            
            elif current_stock and line.startswith('- **'):
                # 解析股票信息行
                if ':' in line:
                    key_part = line.split(':', 1)[0].replace('- **', '').replace('**', '')
                    value_part = line.split(':', 1)[1].strip()
                    
                    # 映射中文字段到英文
                    field_mapping = {
                        '收盘价': 'close_price',
                        '所属行业': 'industry', 
                        '注册地': 'area',
                        '股票类型': 'stock_type',
                        '涨跌幅': 'price_change_pct',
                        '成交量': 'volume',
                        '综合评分': 'comprehensive_score',
                        '通过策略数': 'strategy_count',
                        '策略名称': 'strategies'
                    }
                    
                    if key_part in field_mapping:
                        field_name = field_mapping[key_part]
                        # 处理数值
                        if field_name in ['close_price', 'comprehensive_score']:
                            try:
                                value_part = float(value_part.replace('元', '').replace('分', ''))
                            except:
                                pass
                        elif field_name == 'strategy_count':
                            try:
                                value_part = int(value_part.replace('个', ''))
                            except:
                                pass
                        
                        current_stock['analysis_data'][field_name] = value_part
        
        if current_stock:
            stocks.append(current_stock)
        
        return stocks[:10]  # 返回前10只股票
    
    def analyze_stock_with_claude(self, stock_info: Dict[str, Any]) -> Dict[str, Any]:
        """使用Claude分析单只股票"""
        
        stock_code = stock_info.get('stock_code', '')
        stock_name = stock_info.get('stock_name', '')
        analysis_data = stock_info.get('analysis_data', {})
        
        # 构建分析提示词
        prompt = f"""
请作为专业的中国股票投资分析师，对以下股票进行全面分析：

股票信息：
- 代码：{stock_code}
- 名称：{stock_name}
- 行业：{analysis_data.get('industry', '未知')}
- 地区：{analysis_data.get('area', '未知')}
- 收盘价：{analysis_data.get('close_price', '未知')}
- 涨跌幅：{analysis_data.get('price_change_pct', '未知')}
- 成交量：{analysis_data.get('volume', '未知')}
- 量化评分：{analysis_data.get('comprehensive_score', '未知')}分
- 选中策略：{analysis_data.get('strategies', '未知')}

请从以下维度进行分析并给出0-100分的AI评分：

1. **基本面分析** (权重25%)：
   - 行业前景和发展趋势
   - 公司在行业中的地位
   - 财务健康状况评估

2. **技术面分析** (权重25%)：
   - 当前价格走势判断
   - 支撑阻力位分析  
   - 成交量配合度

3. **市场情绪** (权重20%)：
   - 市场对该行业的关注度
   - 投资者情绪判断
   - 资金流向分析

4. **投资建议** (权重30%)：
   - 短期投资价值(1-3个月)
   - 中期投资潜力(3-12个月)
   - 风险提示和注意事项

请用JSON格式返回分析结果：
{{
    "ai_score": <0-100的整数评分>,
    "confidence": <0-1的置信度>,
    "analysis": {{
        "fundamentals": "<基本面分析>",
        "technical": "<技术面分析>", 
        "sentiment": "<市场情绪分析>",
        "recommendation": "<投资建议>",
        "risks": "<风险提示>",
        "time_horizon": {{
            "short_term": "<短期展望>",
            "medium_term": "<中期展望>"
        }}
    }},
    "final_recommendation": "<BUY/HOLD/SELL>",
    "reasoning": "<总体推理过程>"
}}
"""
        
        try:
            message = self.client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=2000,
                temperature=0.1,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # 提取JSON内容
            response_text = message.content[0].text
            
            # 尝试提取JSON
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    return result
                except json.JSONDecodeError:
                    pass
            
            # 如果无法解析JSON，返回基本结果
            return {
                "ai_score": 50,
                "confidence": 0.5,
                "analysis": {
                    "fundamentals": "基本面分析暂不可用",
                    "technical": "技术面分析暂不可用",
                    "sentiment": "情绪分析暂不可用", 
                    "recommendation": "投资建议暂不可用",
                    "risks": "风险分析暂不可用"
                },
                "final_recommendation": "HOLD",
                "reasoning": "分析结果解析失败",
                "raw_response": response_text
            }
            
        except Exception as e:
            print(f"Claude API调用失败: {e}")
            return {
                "ai_score": 0,
                "confidence": 0.0,
                "analysis": {
                    "fundamentals": f"分析失败: {e}",
                    "technical": "技术分析不可用",
                    "sentiment": "情绪分析不可用",
                    "recommendation": "无法提供建议",
                    "risks": "风险评估不可用"
                },
                "final_recommendation": "HOLD",
                "reasoning": f"API调用异常: {e}"
            }
    
    def generate_enhanced_report(self, date: str, stocks_with_ai: List[Dict[str, Any]], 
                               output_dir: str = "reports") -> str:
        """生成增强版报告"""
        
        output_path = Path(output_dir) / "ai_enhanced" 
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_file = output_path / f"AI增强选股报告_{date.replace('-', '')}.md"
        
        # 按AI评分排序
        stocks_with_ai.sort(key=lambda x: x.get('ai_analysis', {}).get('ai_score', 0), reverse=True)
        
        report_content = f"""# 🤖 AI增强股票分析报告

## 📊 分析概览
- **分析日期**: {date}
- **AI模型**: Claude 3.5 Haiku
- **分析股票数**: {len(stocks_with_ai)}只
- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🏆 AI评分排行榜

"""
        
        for i, stock in enumerate(stocks_with_ai, 1):
            ai_analysis = stock.get('ai_analysis', {})
            analysis_detail = ai_analysis.get('analysis', {})
            
            ai_score = ai_analysis.get('ai_score', 0)
            confidence = ai_analysis.get('confidence', 0)
            recommendation = ai_analysis.get('final_recommendation', 'HOLD')
            
            # 设置推荐标记
            rec_emoji = "🟢" if recommendation == "BUY" else "🔴" if recommendation == "SELL" else "🟡"
            
            report_content += f"""### {i}. {stock['stock_code']} - {stock['stock_name']} {rec_emoji}

**AI评分**: {ai_score}/100 | **置信度**: {confidence:.2f} | **建议**: {recommendation}

**基本面分析**:
{analysis_detail.get('fundamentals', '暂无')}

**技术面分析**:
{analysis_detail.get('technical', '暂无')}

**市场情绪**:
{analysis_detail.get('sentiment', '暂无')}

**投资建议**:
{analysis_detail.get('recommendation', '暂无')}

**风险提示**:
{analysis_detail.get('risks', '暂无')}

**分析推理**:
{ai_analysis.get('reasoning', '暂无')}

---

"""
        
        report_content += """## ⚠️ 重要声明

- 本分析由AI生成，仅供参考，不构成投资建议
- 投资有风险，决策需谨慎
- 请结合自身风险承受能力做出投资决策
- AI分析可能存在偏差，请以实际市场表现为准

---

🤖 Generated with Claude 3.5 Haiku
"""
        
        # 保存报告
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return str(report_file)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Claude股票分析器")
    parser.add_argument("--date", type=str, 
                       default=datetime.now().strftime("%Y-%m-%d"),
                       help="分析日期 (YYYY-MM-DD)")
    parser.add_argument("--top-n", type=int, default=10,
                       help="分析前N只股票")
    parser.add_argument("--output-dir", type=str, default="reports",
                       help="输出目录")
    parser.add_argument("--api-key", type=str,
                       help="Claude API密钥")
    
    args = parser.parse_args()
    
    try:
        # 初始化分析器
        analyzer = ClaudeStockAnalyzer(api_key=args.api_key)
        print(f"✅ Claude分析器初始化成功")
        
        # 解析每日报告
        print(f"📊 解析{args.date}的选股报告...")
        stocks = analyzer.parse_daily_report(args.date)
        
        if not stocks:
            print("❌ 未找到股票数据")
            return
        
        print(f"✅ 找到{len(stocks)}只股票，开始AI分析...")
        
        # 限制分析数量
        stocks_to_analyze = stocks[:args.top_n]
        stocks_with_ai = []
        
        # 逐一分析股票
        for i, stock in enumerate(stocks_to_analyze, 1):
            print(f"🔍 分析第{i}只股票: {stock['stock_code']} - {stock['stock_name']}")
            
            ai_analysis = analyzer.analyze_stock_with_claude(stock)
            stock['ai_analysis'] = ai_analysis
            stocks_with_ai.append(stock)
            
            ai_score = ai_analysis.get('ai_score', 0)
            recommendation = ai_analysis.get('final_recommendation', 'HOLD')
            print(f"   AI评分: {ai_score}/100, 建议: {recommendation}")
        
        # 生成增强报告
        print("📝 生成AI增强报告...")
        report_file = analyzer.generate_enhanced_report(args.date, stocks_with_ai, args.output_dir)
        
        print(f"✅ 分析完成！报告已保存到: {report_file}")
        
        # 显示总结
        avg_score = sum(s['ai_analysis']['ai_score'] for s in stocks_with_ai) / len(stocks_with_ai)
        buy_count = sum(1 for s in stocks_with_ai if s['ai_analysis']['final_recommendation'] == 'BUY')
        
        print(f"\n📈 分析总结:")
        print(f"   平均AI评分: {avg_score:.1f}/100")
        print(f"   BUY建议: {buy_count}只")
        print(f"   HOLD建议: {len(stocks_with_ai) - buy_count}只")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()