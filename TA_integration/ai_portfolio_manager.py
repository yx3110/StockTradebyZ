#!/usr/bin/env python3
"""
AI驱动的集中投资组合管理系统
基于ChatGPT Micro-Cap Experiment的思路，使用AI分析进行投资决策
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import os
import logging
from dataclasses import dataclass, asdict
import yfinance as yf

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Position:
    """持仓信息"""
    stock_code: str
    stock_name: str
    shares: int
    current_price: float
    value: float
    signal: str
    signal_strength: int
    action: str
    reason: str
    priority: str = "medium"
    stop_loss_price: Optional[float] = None
    target_price: Optional[float] = None
    entry_date: Optional[str] = None
    
class AIPortfolioManager:
    """AI驱动的投资组合管理器"""
    
    def __init__(self, 
                 portfolio_size: int = 5,
                 stop_loss_pct: float = 0.1,
                 position_size_range: Tuple[float, float] = (0.2, 0.4),
                 config_path: str = "TA_integration/config/ai_portfolio_config.json"):
        """
        初始化AI投资组合管理器
        
        Args:
            portfolio_size: 集中持仓数量（默认5只）
            stop_loss_pct: 止损百分比（默认10%）
            position_size_range: 每个仓位占比范围（默认20-40%）
            config_path: 配置文件路径
        """
        self.portfolio_size = portfolio_size
        self.stop_loss_pct = stop_loss_pct
        self.position_size_range = position_size_range
        self.config_path = config_path
        
        # 加载或创建配置
        self.config = self._load_config()
        
        # 初始化性能跟踪
        self.performance_history = []
        self.trade_log = []
        
    def _load_config(self) -> Dict:
        """加载配置文件"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 默认配置
            default_config = {
                "benchmark_indices": ["000300.SH", "399006.SZ"],  # 沪深300, 创业板指
                "catalyst_keywords": ["重组", "并购", "新产品", "大订单", "业绩预增"],
                "risk_keywords": ["诉讼", "处罚", "亏损", "减持", "质押"],
                "max_drawdown": 0.2,
                "rebalance_frequency": "weekly",
                "cash_reserve": 0.05
            }
            return default_config
    
    def analyze_current_portfolio(self, portfolio_file: str) -> Dict:
        """分析当前持仓并生成AI建议"""
        logger.info(f"分析持仓文件: {portfolio_file}")
        
        # 解析现有持仓
        positions = self._parse_portfolio_file(portfolio_file)
        
        # 为所有持仓计算止损和目标价
        all_positions_with_targets = []
        for position in positions:
            position_with_targets = self._calculate_price_targets(position)
            all_positions_with_targets.append(position_with_targets)
        
        # 生成集中投资组合建议
        concentrated_portfolio = self._generate_concentrated_portfolio(positions)
        
        # 为集中投资组合计算止损和目标价
        for position in concentrated_portfolio:
            position = self._calculate_price_targets(position)
        
        # 生成交易计划
        trading_plan = self._generate_trading_plan(all_positions_with_targets)
        
        return {
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current_positions": len(positions),
            "all_positions": all_positions_with_targets,  # 添加所有持仓
            "concentrated_portfolio": concentrated_portfolio,
            "trading_plan": trading_plan,
            "risk_metrics": self._calculate_risk_metrics(concentrated_portfolio)
        }
    
    def _parse_portfolio_file(self, file_path: str) -> List[Position]:
        """解析持仓文件"""
        positions = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 简单解析逻辑，提取股票信息
        lines = content.split('\n')
        current_position = None
        
        for line in lines:
            if line.startswith('###') and '- ' in line:
                # 解析股票代码和名称
                parts = line.split('-')
                if len(parts) >= 2:
                    code_part = parts[0].replace('###', '').strip()
                    name_part = parts[1].strip()
                    # 提取纯数字代码 (去掉可能的后缀)
                    stock_code = code_part.split('.')[-1]
                    stock_name = name_part
                    current_position = {
                        "stock_code": stock_code,
                        "stock_name": stock_name
                    }
            
            elif current_position and '**持仓**:' in line:
                shares = int(line.split(':')[1].replace('股', '').strip())
                current_position['shares'] = shares
            
            elif current_position and '**当前价格**:' in line:
                price = float(line.split('¥')[1].strip())
                current_position['current_price'] = price
            
            elif current_position and '**持仓价值**:' in line:
                value = float(line.split('¥')[1].split()[0].replace(',', ''))
                current_position['value'] = value
                
            elif current_position and '**交易信号**:' in line:
                signal_parts = line.split(':')[1].strip().split('(')
                signal = signal_parts[0].strip()
                if len(signal_parts) > 1:
                    # 解析强度: "强度: 9/10)" -> 9
                    strength_text = signal_parts[1].replace(')', '')
                    if '强度:' in strength_text:
                        strength_num = strength_text.split('强度:')[1].split('/')[0].strip()
                        try:
                            strength = int(strength_num)
                        except ValueError:
                            strength = 5  # 默认强度
                    else:
                        strength = 5
                else:
                    strength = 5
                current_position['signal'] = signal
                current_position['signal_strength'] = strength
                
            elif current_position and '**建议操作**:' in line:
                action = line.split(':')[1].strip()
                current_position['action'] = action
                
            elif current_position and '**操作理由**:' in line:
                reason = line.split(':')[1].strip()
                current_position['reason'] = reason
                
                # 创建Position对象
                positions.append(Position(**current_position))
                current_position = None
        
        return positions
    
    def _generate_concentrated_portfolio(self, positions: List[Position]) -> List[Position]:
        """生成集中投资组合（3-5只高确信度股票）"""
        logger.info("生成AI驱动的集中投资组合")
        
        # 评分逻辑：结合信号强度、持仓占比和AI分析
        scored_positions = []
        total_value = sum(p.value for p in positions)
        
        for pos in positions:
            # 基础分数
            score = 0
            score_details = []
            
            # 1. 信号评分 (权重最高)
            if pos.signal == "HOLD":
                signal_score = 5
                score_details.append(f"HOLD信号: +{signal_score}")
            elif pos.signal == "BUY":
                signal_score = 10
                score_details.append(f"BUY信号: +{signal_score}")
            elif pos.signal == "AVOID":
                signal_score = -pos.signal_strength
                score_details.append(f"AVOID信号(强度{pos.signal_strength}): {signal_score}")
            else:
                signal_score = 0
                score_details.append("未知信号: 0")
            score += signal_score
            
            # 2. 持仓价值占比评分
            value_pct = pos.value / total_value
            if 0.05 <= value_pct <= 0.25:
                value_score = 5
                score_details.append(f"合理仓位({value_pct:.1%}): +{value_score}")
            elif value_pct > 0.5:
                value_score = -8
                score_details.append(f"过度集中({value_pct:.1%}): {value_score}")
            elif value_pct < 0.05:
                value_score = -2
                score_details.append(f"仓位过小({value_pct:.1%}): {value_score}")
            else:
                value_score = 2
                score_details.append(f"仓位适中({value_pct:.1%}): +{value_score}")
            score += value_score
            
            # 3. 技术面关键词分析
            reason_lower = pos.reason.lower()
            if any(keyword in reason_lower for keyword in ["支撑", "金叉", "上行", "强势"]):
                tech_score = 3
                score_details.append("技术面积极: +3")
            elif any(keyword in reason_lower for keyword in ["死叉", "弱势", "下行", "破位"]):
                tech_score = -3
                score_details.append("技术面消极: -3")
            else:
                tech_score = 0
                score_details.append("技术面中性: 0")
            score += tech_score
            
            # 4. 行动建议评分
            if "保持" in pos.action or "持有" in pos.action:
                action_score = 3
                score_details.append("建议持有: +3")
            elif "加仓" in pos.action:
                action_score = 5
                score_details.append("建议加仓: +5")
            elif "减仓" in pos.action:
                if "小幅" in pos.action:
                    action_score = -2
                    score_details.append("建议小幅减仓: -2")
                else:
                    action_score = -5
                    score_details.append("建议大幅减仓: -5")
            else:
                action_score = 0
                score_details.append("行动中性: 0")
            score += action_score
            
            # 记录评分详情
            pos.score_details = score_details
            pos.total_score = score
            scored_positions.append((score, pos))
            
            logger.info(f"{pos.stock_code}-{pos.stock_name}: 总分{score}, 详情: {'; '.join(score_details)}")
        
        # 排序并选择前N个
        scored_positions.sort(key=lambda x: x[0], reverse=True)
        concentrated = [pos for score, pos in scored_positions[:self.portfolio_size]]
        
        logger.info(f"核心持仓选择完成，TOP{self.portfolio_size}: {[f'{p.stock_code}-{p.stock_name}(分数:{p.total_score})' for p in concentrated]}")
        
        # 调整仓位大小
        total_value = sum(pos.value for pos in concentrated)
        for pos in concentrated:
            target_pct = min(max(pos.value / total_value, self.position_size_range[0]), 
                           self.position_size_range[1])
            pos.target_shares = int(total_value * target_pct / pos.current_price)
        
        return concentrated
    
    def _calculate_price_targets(self, position: Position) -> Position:
        """计算止损和目标价"""
        # 止损价：当前价格下跌10%
        position.stop_loss_price = position.current_price * (1 - self.stop_loss_pct)
        
        # 目标价：根据技术面和信号强度
        if position.signal == "HOLD":
            position.target_price = position.current_price * 1.15  # 15%目标
        elif position.signal == "BUY":
            position.target_price = position.current_price * 1.25  # 25%目标
        else:
            position.target_price = position.current_price * 1.05  # 5%目标
        
        return position
    
    def _generate_trading_plan(self, portfolio: List[Position]) -> Dict:
        """生成具体交易计划"""
        plan = {
            "immediate_actions": [],
            "watch_list": [],
            "stop_loss_orders": []
        }
        
        for pos in portfolio:
            # 立即行动项
            if pos.signal == "AVOID" and pos.signal_strength >= 8:
                plan["immediate_actions"].append({
                    "stock_code": pos.stock_code,
                    "action": "SELL",
                    "shares": int(pos.shares * 0.6),  # 卖出60%
                    "reason": f"技术面极度弱势 (信号强度: {pos.signal_strength}/10)",
                    "priority": "HIGH"
                })
            
            # 观察列表
            elif pos.signal == "AVOID" and pos.signal_strength >= 6:
                plan["watch_list"].append({
                    "stock_code": pos.stock_code,
                    "action": "REDUCE",
                    "shares": int(pos.shares * 0.3),  # 减仓30%
                    "reason": pos.reason,
                    "trigger": f"跌破{pos.stop_loss_price:.2f}"
                })
            
            # 止损订单
            plan["stop_loss_orders"].append({
                "stock_code": pos.stock_code,
                "stop_price": pos.stop_loss_price,
                "shares": pos.shares,
                "current_price": pos.current_price
            })
        
        return plan
    
    def _calculate_risk_metrics(self, portfolio: List[Position]) -> Dict:
        """计算风险指标"""
        total_value = sum(pos.value for pos in portfolio)
        
        # 集中度风险
        concentrations = [pos.value / total_value for pos in portfolio]
        max_concentration = max(concentrations)
        herfindahl_index = sum(c**2 for c in concentrations)
        
        # 下行风险
        potential_losses = sum(pos.value * self.stop_loss_pct for pos in portfolio)
        
        return {
            "total_portfolio_value": total_value,
            "number_of_positions": len(portfolio),
            "max_position_concentration": f"{max_concentration:.1%}",
            "herfindahl_index": f"{herfindahl_index:.3f}",
            "potential_stop_loss": potential_losses,
            "stop_loss_percentage": f"{potential_losses/total_value:.1%}"
        }
    
    def generate_daily_report(self, portfolio_analysis: Dict) -> str:
        """生成每日AI投资组合报告"""
        all_positions = portfolio_analysis.get('all_positions', [])
        concentrated_portfolio = portfolio_analysis['concentrated_portfolio']
        
        report = f"""# 🤖 AI驱动投资组合完整分析报告

## 📊 概览
- **分析时间**: {portfolio_analysis['analysis_date']}
- **当前持仓数**: {portfolio_analysis['current_positions']}只
- **建议集中持仓**: {len(concentrated_portfolio)}只
- **投资理念**: 基于AI深度分析的高确信度集中投资

## 📈 全部持仓AI分析

"""
        # 按AI评分排序显示所有持仓
        for i, pos in enumerate(all_positions, 1):
            # 判断是否在集中投资组合中
            is_concentrated = any(cp.stock_code == pos.stock_code for cp in concentrated_portfolio)
            status_emoji = "🌟" if is_concentrated else "📊"
            status_text = " (核心持仓)" if is_concentrated else ""
            
            report += f"""### {i}. {pos.stock_code} - {pos.stock_name}{status_text} {status_emoji}
- **当前仓位**: {pos.shares}股 (¥{pos.value:,.0f})
- **当前价格**: ¥{pos.current_price:.2f}
- **AI信号**: {pos.signal} (强度: {pos.signal_strength}/10)
- **止损价**: ¥{pos.stop_loss_price:.2f} (-{self.stop_loss_pct:.0%})
- **目标价**: ¥{pos.target_price:.2f} (+{(pos.target_price/pos.current_price-1):.0%})
- **建议操作**: {pos.action}
- **AI分析**: {pos.reason}

"""

        # 添加集中投资组合建议
        report += f"""
## 🎯 AI推荐集中持仓 (TOP {len(concentrated_portfolio)})

基于综合AI评分系统，推荐以下核心持仓：

"""
        for i, pos in enumerate(concentrated_portfolio, 1):
            score_detail = '; '.join(getattr(pos, 'score_details', ['评分详情未计算']))
            report += f"**{i}. {pos.stock_code} - {pos.stock_name}** (总分: {getattr(pos, 'total_score', '未知')})\n"
            report += f"   - 信号: {pos.signal} (强度: {pos.signal_strength}/10)\n"
            report += f"   - 建议: {pos.action}\n"
            report += f"   - 评分详情: {score_detail}\n\n"
        
        # 添加评分说明
        report += """
### 📊 评分体系说明
- **信号评分**: BUY(+10) > HOLD(+5) > AVOID(-强度值)
- **仓位评分**: 合理仓位(5-25%)+5分，过度集中(>50%)-8分
- **技术面评分**: 积极关键词+3分，消极关键词-3分
- **行动评分**: 加仓+5分，持有+3分，小幅减仓-2分，大幅减仓-5分
"""

        # 添加交易计划
        plan = portfolio_analysis['trading_plan']
        report += f"""## 📋 交易执行计划

### 🚨 立即执行 ({len(plan['immediate_actions'])}项)
"""
        for action in plan['immediate_actions']:
            report += f"- **{action['stock_code']}**: {action['action']} {action['shares']}股 - {action['reason']}\n"
        
        report += f"""
### 👀 重点观察 ({len(plan['watch_list'])}项)
"""
        for watch in plan['watch_list']:
            report += f"- **{watch['stock_code']}**: {watch['action']} {watch['shares']}股 (触发条件: {watch['trigger']})\n"
        
        # 添加风险指标
        risk = portfolio_analysis['risk_metrics']
        report += f"""
## 📊 风险管理指标

- **组合总价值**: ¥{risk['total_portfolio_value']:,.0f}
- **最大单一持仓**: {risk['max_position_concentration']}
- **集中度指数**: {risk['herfindahl_index']}
- **潜在止损金额**: ¥{risk['potential_stop_loss']:,.0f} ({risk['stop_loss_percentage']})

## 💡 AI投资哲学

1. **集中投资**: 持有3-5只高确信度股票
2. **深度研究**: 每只股票都经过AI多维度分析
3. **严格止损**: 所有仓位设置{self.stop_loss_pct:.0%}止损
4. **动态调整**: 每周重新评估并调整组合

## ⚠️ 风险提示

- 集中投资风险较高，请确保风险承受能力
- 严格执行止损纪律，避免情绪化交易
- AI分析仅供参考，需结合市场实际情况

🤖 Generated by AI Portfolio Manager | 灵感来源: ChatGPT Micro-Cap Experiment
"""
        return report
    
    def track_performance(self, current_prices: Dict[str, float], 
                         benchmark_prices: Dict[str, float]) -> Dict:
        """跟踪投资组合表现"""
        # 计算收益率
        portfolio_return = self._calculate_portfolio_return(current_prices)
        
        # 计算基准收益率
        benchmark_returns = {
            idx: (price - self.config.get('benchmark_start_prices', {}).get(idx, price)) / 
                  self.config.get('benchmark_start_prices', {}).get(idx, price)
            for idx, price in benchmark_prices.items()
        }
        
        # 计算夏普比率和其他指标
        metrics = {
            "portfolio_return": portfolio_return,
            "benchmark_returns": benchmark_returns,
            "excess_return": portfolio_return - np.mean(list(benchmark_returns.values())),
            "tracking_date": datetime.now().strftime("%Y-%m-%d")
        }
        
        self.performance_history.append(metrics)
        return metrics
    
    def _calculate_portfolio_return(self, current_prices: Dict[str, float]) -> float:
        """计算组合收益率"""
        # 这里需要实现实际的收益率计算逻辑
        # 暂时返回模拟值
        return 0.05
    
    def analyze_sample_portfolio(self, positions: List[Position]) -> Dict:
        """分析示例持仓并生成AI建议"""
        logger.info(f"分析 {len(positions)} 个持仓")
        
        # 为所有持仓计算止损和目标价
        all_positions_with_targets = []
        for position in positions:
            position_with_targets = self._calculate_price_targets(position)
            all_positions_with_targets.append(position_with_targets)
        
        # 生成集中投资组合建议
        concentrated_portfolio = self._generate_concentrated_portfolio(all_positions_with_targets)
        
        total_value = sum(p.value for p in positions)
        
        return {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_positions': len(positions),
            'all_positions': all_positions_with_targets,
            'concentrated_portfolio': concentrated_portfolio,
            'total_value': total_value,
            'trading_plan': {
                'buy_orders': [p for p in all_positions_with_targets if p.signal == 'BUY'],
                'sell_orders': [p for p in all_positions_with_targets if p.signal == 'SELL'],
                'hold_positions': [p for p in all_positions_with_targets if p.signal == 'HOLD'],
                'immediate_actions': [
                    {'stock_code': p.stock_code, 'action': p.action, 'shares': p.shares, 'reason': p.reason}
                    for p in all_positions_with_targets if p.signal in ['BUY', 'SELL']
                ],
                'watch_list': [
                    {'stock_code': p.stock_code, 'action': '观察', 'shares': p.shares, 'trigger': f'价格{p.target_price:.2f}'}
                    for p in all_positions_with_targets if p.signal == 'HOLD'
                ]
            },
            'risk_metrics': {
                'total_portfolio_value': total_value,
                'max_position_concentration': f"{max(p.value/total_value for p in positions):.1%}",
                'herfindahl_index': f"{sum((p.value/total_value)**2 for p in positions):.3f}",
                'potential_stop_loss': total_value * self.stop_loss_pct,
                'stop_loss_percentage': f"{self.stop_loss_pct:.1%}"
            }
        }


def main():
    """主函数：运行AI投资组合分析"""
    # 创建管理器
    manager = AIPortfolioManager(
        portfolio_size=5,
        stop_loss_pct=0.1,
        position_size_range=(0.2, 0.4)
    )
    
    # 使用示例持仓数据
    sample_positions = [
        Position("000001", "平安银行", 1000, 12.50, 12500, "BUY", 8, "建议加仓", "技术面强势突破"),
        Position("000002", "万科A", 2000, 8.30, 16600, "HOLD", 6, "维持持有", "基本面稳定"),
        Position("600036", "招商银行", 500, 41.20, 20600, "BUY", 9, "强烈买入", "业绩超预期"),
        Position("002594", "比亚迪", 200, 250.80, 50160, "SELL", 3, "建议减仓", "估值过高"),
        Position("000858", "五粮液", 100, 180.50, 18050, "HOLD", 5, "观望", "白酒板块调整")
    ]
    
    # 生成分析
    analysis = manager.analyze_sample_portfolio(sample_positions)
    
    # 生成报告
    report = manager.generate_daily_report(analysis)
    
    # 保存报告
    output_dir = "reports/ai_portfolio"
    os.makedirs(output_dir, exist_ok=True)
    
    report_file = os.path.join(output_dir, f"AI集中投资组合_{datetime.now().strftime('%Y%m%d')}.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"AI投资组合报告已生成: {report_file}")
    
    # 输出关键信息
    print(f"\n🤖 AI投资组合分析完成")
    print(f"📊 建议集中持有: {len(analysis['concentrated_portfolio'])}只股票")
    print(f"🚨 立即行动项: {len(analysis['trading_plan']['immediate_actions'])}项")
    print(f"📝 报告已保存至: {report_file}")


if __name__ == "__main__":
    main()