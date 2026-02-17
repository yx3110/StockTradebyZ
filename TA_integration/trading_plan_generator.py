#!/usr/bin/env python3
"""
基于ChatGPT Micro-Cap实验思路的交易计划生成器
结合现有持仓和最新选股报告，生成下一交易日的具体操作建议
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import os
import logging
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TA_integration.ai_portfolio_manager import AIPortfolioManager, Position

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingPlanGenerator:
    """交易计划生成器 - ChatGPT Micro-Cap风格"""
    
    def __init__(self, cash_balance: float = 50000.0, max_position_size: float = 0.1):
        """
        初始化交易计划生成器
        
        Args:
            cash_balance: 可用现金余额
            max_position_size: 单个仓位最大占比
        """
        self.cash_balance = cash_balance
        self.max_position_size = max_position_size
        self.ai_manager = AIPortfolioManager()
        
    def analyze_existing_positions(self, portfolio_file: str) -> Dict:
        """分析现有持仓"""
        return self.ai_manager.analyze_current_portfolio(portfolio_file)
    
    def parse_stock_selection_report(self, report_file: str) -> List[Dict]:
        """解析选股报告，提取新机会"""
        candidates = []
        
        with open(report_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        current_stock = None
        
        for line in lines:
            # 解析股票基本信息
            if line.startswith('### ') and ' - ' in line and '未知' not in line:
                try:
                    # 提取股票代码和名称
                    parts = line.split(' - ')
                    if len(parts) >= 2:
                        header = parts[0].replace('###', '').strip()
                        stock_name = parts[1].strip()
                        # 提取股票代码 (去掉序号)
                        code_part = header.split('.')[-1]
                        if code_part.isdigit() and len(code_part) == 6:
                            stock_code = code_part
                        else:
                            # 尝试其他解析方式
                            import re
                            code_match = re.search(r'(\d{6})', header)
                            if code_match:
                                stock_code = code_match.group(1)
                            else:
                                continue
                        
                        current_stock = {
                            'stock_code': stock_code,
                            'stock_name': stock_name
                        }
                except Exception as e:
                    logger.warning(f"解析股票信息失败: {line}, 错误: {e}")
                    continue
            
            elif current_stock:
                # 解析价格信息
                if '**收盘价**:' in line:
                    try:
                        price = float(line.split(':')[1].replace('元', '').strip())
                        current_stock['current_price'] = price
                    except:
                        pass
                
                # 解析综合评分
                elif '**综合评分**:' in line:
                    try:
                        score = float(line.split(':')[1].replace('分', '').strip())
                        current_stock['score'] = score
                    except:
                        pass
                
                # 解析投资建议
                elif '**建议买入价**:' in line:
                    try:
                        buy_price = float(line.split(':')[1].replace('元', '').strip())
                        current_stock['suggested_buy_price'] = buy_price
                    except:
                        pass
                
                elif '**建议止损价**:' in line:
                    try:
                        stop_price = float(line.split(':')[1].replace('元', '').strip())
                        current_stock['suggested_stop_price'] = stop_price
                    except:
                        pass
                
                elif '**建议止盈价**:' in line:
                    try:
                        target_price = float(line.split(':')[1].replace('元', '').strip())
                        current_stock['suggested_target_price'] = target_price
                    except:
                        pass
                
                elif '**目标收益**:' in line:
                    try:
                        target_return = line.split(':')[1].replace('%', '').replace('+', '').strip()
                        current_stock['target_return'] = float(target_return) / 100
                    except:
                        pass
                
                elif '**策略名称**:' in line:
                    try:
                        strategies = line.split(':')[1].strip()
                        current_stock['strategies'] = strategies
                    except:
                        pass
                
                # 检查是否到达股票信息结尾
                elif line.strip() == '---' and current_stock:
                    # 验证必要字段
                    if all(key in current_stock for key in ['stock_code', 'stock_name', 'current_price', 'score']):
                        candidates.append(current_stock.copy())
                    current_stock = None
        
        # 按评分排序
        candidates = sorted(candidates, key=lambda x: x.get('score', 0), reverse=True)
        
        logger.info(f"从选股报告中解析出 {len(candidates)} 个投资候选")
        return candidates
    
    def generate_trading_decisions(self, current_analysis: Dict, 
                                 new_candidates: List[Dict]) -> Dict:
        """生成交易决策 - ChatGPT Micro-Cap风格"""
        
        # 现有持仓分析
        current_positions = current_analysis['all_positions']
        total_portfolio_value = sum(pos.value for pos in current_positions) + self.cash_balance
        
        decisions = {
            'sell_decisions': [],
            'buy_decisions': [],
            'hold_decisions': [],
            'position_adjustments': []
        }
        
        # 1. 处理现有持仓的卖出决策 (ChatGPT风格: 严格止损)
        for pos in current_positions:
            decision = self._evaluate_existing_position(pos, total_portfolio_value)
            if decision['action'] == 'SELL':
                decisions['sell_decisions'].append(decision)
            elif decision['action'] == 'HOLD':
                decisions['hold_decisions'].append(decision)
            elif decision['action'] == 'REDUCE':
                decisions['position_adjustments'].append(decision)
        
        # 2. 评估新买入机会
        available_cash = self.cash_balance
        for sell_decision in decisions['sell_decisions']:
            available_cash += sell_decision.get('proceeds', 0)
        
        # 筛选高质量新候选 (ChatGPT风格: 深度研究 + 高确信度)
        qualified_candidates = self._screen_new_candidates(new_candidates, current_positions)
        
        # 3. 生成买入决策 (集中投资风格)
        buy_decisions = self._generate_buy_decisions(qualified_candidates, available_cash, total_portfolio_value)
        decisions['buy_decisions'] = buy_decisions
        
        return decisions
    
    def _evaluate_existing_position(self, position: Position, total_value: float) -> Dict:
        """评估现有持仓 - ChatGPT风格决策"""
        
        # 计算当前仓位占比
        position_pct = position.value / total_value
        
        # ChatGPT风格: 基于信号强度和技术面做决策
        if position.signal == "AVOID" and position.signal_strength >= 8:
            # 强烈卖出信号
            return {
                'action': 'SELL',
                'stock_code': position.stock_code,
                'stock_name': position.stock_name,
                'shares': position.shares,
                'current_price': position.current_price,
                'proceeds': position.value,
                'reason': f"技术面极度弱势，信号强度{position.signal_strength}/10",
                'priority': 'HIGH'
            }
        
        elif position.signal == "AVOID" and position.signal_strength >= 6:
            # 减仓决策
            reduce_shares = int(position.shares * 0.5)  # 减仓50%
            return {
                'action': 'REDUCE',
                'stock_code': position.stock_code,
                'stock_name': position.stock_name,
                'shares': reduce_shares,
                'current_price': position.current_price,
                'proceeds': reduce_shares * position.current_price,
                'reason': f"技术面偏弱，建议减仓50%",
                'priority': 'MEDIUM'
            }
        
        elif position_pct > 0.4:
            # ChatGPT风格: 避免过度集中
            reduce_shares = int(position.shares * 0.3)  # 减仓30%
            return {
                'action': 'REDUCE',
                'stock_code': position.stock_code,
                'stock_name': position.stock_name,
                'shares': reduce_shares,
                'current_price': position.current_price,
                'proceeds': reduce_shares * position.current_price,
                'reason': f"仓位过度集中({position_pct:.1%})，风险控制",
                'priority': 'HIGH'
            }
        
        else:
            # 持有决策
            return {
                'action': 'HOLD',
                'stock_code': position.stock_code,
                'stock_name': position.stock_name,
                'shares': position.shares,
                'current_price': position.current_price,
                'reason': f"信号: {position.signal}, 继续持有观察",
                'priority': 'LOW'
            }
    
    def _screen_new_candidates(self, candidates: List[Dict], 
                             current_positions: List[Position]) -> List[Dict]:
        """筛选新候选股票 - ChatGPT深度研究风格"""
        
        # 排除已持有的股票
        held_codes = {pos.stock_code for pos in current_positions}
        candidates = [c for c in candidates if c['stock_code'] not in held_codes]
        
        qualified = []
        
        for candidate in candidates:
            # ChatGPT风格筛选标准:
            # 1. 高评分 (>= 85分)
            # 2. 多策略验证
            # 3. 良好的风险收益比
            
            score = candidate.get('score', 0)
            target_return = candidate.get('target_return', 0)
            strategies = candidate.get('strategies', '')
            
            # 基础筛选条件
            if score >= 85:  # 高评分要求
                qualification_score = 0
                reasons = []
                
                # 评分加权
                if score >= 90:
                    qualification_score += 10
                    reasons.append(f"超高评分({score}分)")
                elif score >= 85:
                    qualification_score += 7
                    reasons.append(f"高评分({score}分)")
                
                # 多策略验证
                if ',' in strategies:  # 多策略选中
                    qualification_score += 8
                    reasons.append("多策略验证")
                
                # 收益潜力
                if target_return >= 0.12:  # 12%以上目标收益
                    qualification_score += 5
                    reasons.append(f"高收益潜力({target_return:.1%})")
                elif target_return >= 0.08:
                    qualification_score += 3
                    reasons.append(f"良好收益潜力({target_return:.1%})")
                
                # 综合评估
                if qualification_score >= 15:  # 高标准
                    candidate['qualification_score'] = qualification_score
                    candidate['selection_reasons'] = reasons
                    qualified.append(candidate)
        
        # 按综合评分排序
        qualified = sorted(qualified, key=lambda x: x.get('qualification_score', 0), reverse=True)
        
        logger.info(f"高质量候选股票筛选: {len(qualified)}只通过严格筛选")
        return qualified[:10]  # ChatGPT风格: 专注少数高确信度机会
    
    def _generate_buy_decisions(self, candidates: List[Dict], 
                              available_cash: float, total_value: float) -> List[Dict]:
        """生成买入决策 - 集中投资风格"""
        
        buy_decisions = []
        remaining_cash = available_cash
        
        # ChatGPT风格: 集中投资于少数高确信度标的
        for i, candidate in enumerate(candidates[:3]):  # 最多3只新股票
            
            # 计算建议仓位大小
            if i == 0:  # 最高确信度
                target_position_pct = 0.15  # 15%
            elif i == 1:  # 次高确信度
                target_position_pct = 0.12  # 12%
            else:  # 第三高确信度
                target_position_pct = 0.08  # 8%
            
            target_value = total_value * target_position_pct
            buy_price = candidate.get('suggested_buy_price', candidate.get('current_price', 0))
            
            if buy_price > 0 and target_value <= remaining_cash:
                shares = int(target_value / buy_price)
                actual_cost = shares * buy_price
                
                if shares > 0:
                    buy_decisions.append({
                        'action': 'BUY',
                        'stock_code': candidate['stock_code'],
                        'stock_name': candidate['stock_name'],
                        'shares': shares,
                        'buy_price': buy_price,
                        'cost': actual_cost,
                        'stop_loss_price': candidate.get('suggested_stop_price', buy_price * 0.9),
                        'target_price': candidate.get('suggested_target_price', buy_price * 1.15),
                        'target_return': candidate.get('target_return', 0.15),
                        'score': candidate.get('score', 0),
                        'strategies': candidate.get('strategies', ''),
                        'selection_reasons': candidate.get('selection_reasons', []),
                        'position_pct': target_position_pct,
                        'priority': 'HIGH' if i == 0 else 'MEDIUM'
                    })
                    
                    remaining_cash -= actual_cost
                    logger.info(f"新买入决策: {candidate['stock_name']} {shares}股 @ ¥{buy_price:.2f}")
        
        return buy_decisions
    
    def generate_trading_plan_report(self, decisions: Dict, trading_date: str) -> str:
        """生成交易计划报告 - ChatGPT风格"""
        
        total_sells = len(decisions['sell_decisions'])
        total_buys = len(decisions['buy_decisions'])
        total_holds = len(decisions['hold_decisions'])
        total_adjustments = len(decisions['position_adjustments'])
        
        # 计算资金变动
        sell_proceeds = sum(d.get('proceeds', 0) for d in decisions['sell_decisions'])
        buy_costs = sum(d.get('cost', 0) for d in decisions['buy_decisions'])
        adjustment_proceeds = sum(d.get('proceeds', 0) for d in decisions['position_adjustments'])
        
        net_cash_change = sell_proceeds + adjustment_proceeds - buy_costs
        
        report = f"""# 🎯 ChatGPT风格交易计划报告

## 📅 交易日期: {trading_date}

### 💡 投资哲学
基于ChatGPT Micro-Cap实验的核心理念:
- **集中投资**: 持有少数高确信度股票
- **严格止损**: 10%止损纪律，控制下行风险
- **深度研究**: AI多维度分析 + 量化验证
- **动态调整**: 根据技术面变化及时调整仓位

## 📊 交易概览
- **卖出操作**: {total_sells}只股票
- **买入操作**: {total_buys}只股票  
- **持有观察**: {total_holds}只股票
- **仓位调整**: {total_adjustments}只股票
- **净现金变动**: ¥{net_cash_change:,.0f}

## 🚨 卖出决策 ({total_sells}只)

"""
        
        # 卖出决策详情
        for i, decision in enumerate(decisions['sell_decisions'], 1):
            report += f"""### {i}. {decision['stock_code']} - {decision['stock_name']} 【{decision['priority']}优先级】
- **操作**: 全部卖出 {decision['shares']}股
- **当前价格**: ¥{decision['current_price']:.2f}
- **预期回款**: ¥{decision['proceeds']:,.0f}
- **决策理由**: {decision['reason']}

"""
        
        # 仓位调整
        if total_adjustments > 0:
            report += f"## ⚖️ 仓位调整 ({total_adjustments}只)\n\n"
            for i, decision in enumerate(decisions['position_adjustments'], 1):
                report += f"""### {i}. {decision['stock_code']} - {decision['stock_name']}
- **操作**: 减仓 {decision['shares']}股
- **当前价格**: ¥{decision['current_price']:.2f}
- **回款金额**: ¥{decision['proceeds']:,.0f}
- **调整理由**: {decision['reason']}

"""
        
        # 买入决策
        if total_buys > 0:
            report += f"## 💰 买入决策 ({total_buys}只)\n\n"
            for i, decision in enumerate(decisions['buy_decisions'], 1):
                reasons_text = '; '.join(decision.get('selection_reasons', []))
                report += f"""### {i}. {decision['stock_code']} - {decision['stock_name']} 【{decision['priority']}优先级】
- **买入数量**: {decision['shares']}股
- **建议买入价**: ¥{decision['buy_price']:.2f}
- **投资金额**: ¥{decision['cost']:,.0f}
- **目标仓位**: {decision['position_pct']:.1%}
- **止损价**: ¥{decision['stop_loss_price']:.2f} (-{(1-decision['stop_loss_price']/decision['buy_price']):.1%})
- **目标价**: ¥{decision['target_price']:.2f} (+{decision['target_return']:.1%})
- **量化评分**: {decision['score']:.1f}分
- **选中策略**: {decision['strategies']}
- **选择理由**: {reasons_text}

"""
        
        # 持有决策
        if total_holds > 0:
            report += f"## ⏸️ 持有观察 ({total_holds}只)\n\n"
            for i, decision in enumerate(decisions['hold_decisions'], 1):
                report += f"**{i}. {decision['stock_code']} - {decision['stock_name']}**: {decision['reason']}\n"
        
        # 执行要点
        report += f"""

## 🎯 执行要点

### 优先级顺序
1. **HIGH优先级**: 立即执行，开盘优先处理
2. **MEDIUM优先级**: 市场平稳时执行
3. **LOW优先级**: 观察为主，谨慎操作

### 风险控制
- **止损纪律**: 严格执行10%止损，避免情绪化决策
- **仓位控制**: 单股仓位不超过15%，总仓位不超过85%
- **市场适应**: 密切关注大盘走势，必要时调整执行节奏

### 执行建议
- **分批买入**: 建议分2-3批买入，降低时点风险
- **止损挂单**: 买入后立即设置止损价格
- **动态监控**: 每日收盘后评估持仓表现

## 📊 投资组合展望

### 风险评估
- **预期波动**: 中等（集中投资增加波动性）
- **最大回撤**: 控制在15%以内
- **收益目标**: 15-25%年化收益率

### 成功要素
1. **严格执行**: 按计划执行，避免随意更改
2. **情绪控制**: 不被短期波动影响决策
3. **持续学习**: 根据市场反馈调整策略

---
🤖 Generated by ChatGPT-Style Trading Plan Generator | 灵感来源: ChatGPT Micro-Cap Experiment
"""
        
        return report


def main():
    """主函数：生成明日交易计划"""
    # 初始化交易计划生成器
    generator = TradingPlanGenerator(cash_balance=100000.0)  # 10万现金
    
    # 使用示例持仓数据
    sample_positions = [
        Position("000001", "平安银行", 1000, 12.50, 12500, "BUY", 8, "建议加仓", "技术面强势突破"),
        Position("600036", "招商银行", 500, 41.20, 20600, "BUY", 9, "强烈买入", "业绩超预期"),
        Position("002594", "比亚迪", 200, 250.80, 50160, "SELL", 3, "建议减仓", "估值过高")
    ]
    current_analysis = generator.ai_manager.analyze_sample_portfolio(sample_positions)
    
    # 分析最新选股报告
    selection_report = "/Users/yangxu/StockTradebyZ/reports/daily_selection/选股分析报告_20250801.md"
    new_candidates = generator.parse_stock_selection_report(selection_report)
    
    # 生成交易决策
    trading_decisions = generator.generate_trading_decisions(current_analysis, new_candidates)
    
    # 生成交易计划报告
    tomorrow_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    report = generator.generate_trading_plan_report(trading_decisions, tomorrow_date)
    
    # 保存报告
    output_dir = "reports/trading_plans"
    os.makedirs(output_dir, exist_ok=True)
    
    report_file = os.path.join(output_dir, f"交易计划_{tomorrow_date.replace('-', '')}.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"交易计划已生成: {report_file}")
    
    # 输出关键信息
    print(f"\n🎯 {tomorrow_date} 交易计划已生成")
    print(f"📊 卖出: {len(trading_decisions['sell_decisions'])}只")
    print(f"💰 买入: {len(trading_decisions['buy_decisions'])}只")
    print(f"⏸️  持有: {len(trading_decisions['hold_decisions'])}只")
    print(f"📝 报告位置: {report_file}")
    
    return trading_decisions


if __name__ == "__main__":
    main()