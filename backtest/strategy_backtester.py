#!/usr/bin/env python3
"""
策略回测引擎
集成高级交易策略系统的专业回测器
基于选股报告和策略决策进行回测
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import json

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from strategy.strategy_manager import StrategyManager
from strategy.base_strategy import TradeAction, DecisionReason

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("backtest/logs/strategy_backtest.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("strategy_backtester")

class StrategyBacktester:
    """策略驱动的回测引擎"""
    
    def __init__(self, initial_capital: float = 1000000, strategy_name: str = "平衡型策略"):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.strategy_manager = StrategyManager()
        self.strategy_name = strategy_name
        
        # 设置策略
        if not self.strategy_manager.set_active_strategy(strategy_name):
            logger.warning(f"策略 {strategy_name} 不存在，使用默认平衡型策略")
            self.strategy_manager.set_active_strategy("平衡型策略")
            self.strategy_name = "平衡型策略"
        
        # 持仓和交易记录
        self.positions: Dict[str, Dict] = {}  # {stock_code: position_info}
        self.trades: List[Dict] = []
        self.daily_portfolio: List[Dict] = []
        self.strategy_decisions: List[Dict] = []
        
        # 交易成本（A股标准）
        self.commission_rate = 0.0003
        self.stamp_tax = 0.001
        self.transfer_fee = 0.00002
        self.min_commission = 5.0
        
        logger.info(f"策略回测器初始化完成，策略: {self.strategy_name}，初始资金: {initial_capital:,.0f}元")
    
    def load_data(self, data_dir: str, reports_dir: str, start_date: str, end_date: str) -> Tuple[pd.DataFrame, Dict]:
        """
        加载股票数据和选股报告
        
        Args:
            data_dir: 股票数据目录
            reports_dir: 选股报告目录
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            (股票数据, 选股报告字典)
        """
        logger.info(f"加载数据: {start_date} 至 {end_date}")
        
        # 1. 加载股票数据
        stock_data = self._load_stock_data(data_dir, start_date, end_date)
        
        # 2. 加载选股报告
        reports_data = self._load_reports_data(reports_dir, start_date, end_date)
        
        logger.info(f"数据加载完成: 股票数据 {len(stock_data)} 条，选股报告 {len(reports_data)} 个")
        return stock_data, reports_data
    
    def _load_stock_data(self, data_dir: str, start_date: str, end_date: str) -> pd.DataFrame:
        """加载股票数据"""
        data_path = Path(data_dir)
        all_data = []
        
        csv_files = list(data_path.glob("*.csv"))
        logger.info(f"发现 {len(csv_files)} 个股票数据文件")
        
        for i, csv_file in enumerate(csv_files):
            if i % 1000 == 0:
                logger.info(f"已加载 {i}/{len(csv_files)} 个文件...")
            
            try:
                # 跳过非股票文件
                if csv_file.name.startswith("securities_list"):
                    continue
                
                stock_code = csv_file.stem.split('_')[0]
                df = pd.read_csv(csv_file)
                
                if df.empty:
                    continue
                
                df['stock_code'] = stock_code
                df['date'] = pd.to_datetime(df['date'])
                
                # 过滤日期范围
                mask = (df['date'] >= start_date) & (df['date'] <= end_date)
                df_filtered = df.loc[mask]
                
                if not df_filtered.empty:
                    all_data.append(df_filtered)
                    
            except Exception as e:
                logger.warning(f"加载 {csv_file} 失败: {e}")
                continue
        
        if not all_data:
            raise ValueError("未能加载任何有效的股票数据")
        
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"股票数据加载完成: {len(combined_data)} 条记录")
        
        return combined_data.sort_values(['date', 'stock_code']).reset_index(drop=True)
    
    def _load_reports_data(self, reports_dir: str, start_date: str, end_date: str) -> Dict[str, str]:
        """加载选股报告"""
        reports_path = Path(reports_dir)
        reports_data = {}
        
        report_files = list(reports_path.glob("选股分析报告_*.md"))
        logger.info(f"发现 {len(report_files)} 个选股报告文件")
        
        for report_file in report_files:
            try:
                # 从文件名提取日期
                date_str = report_file.stem.split('_')[1]
                report_date = datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
                
                # 检查日期范围
                if start_date <= report_date <= end_date:
                    with open(report_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 检查是否为交易日
                    if "不是交易日" not in content:
                        reports_data[report_date] = content
                        
            except Exception as e:
                logger.warning(f"加载报告 {report_file} 失败: {e}")
                continue
        
        logger.info(f"选股报告加载完成: {len(reports_data)} 个有效报告")
        return reports_data
    
    def run_backtest(self, stock_data: pd.DataFrame, reports_data: Dict[str, str]) -> Dict:
        """
        运行策略回测
        
        Args:
            stock_data: 股票数据
            reports_data: 选股报告数据
            
        Returns:
            回测结果
        """
        logger.info("开始执行策略回测...")
        
        # 获取所有交易日期
        all_dates = sorted(stock_data['date'].unique())
        
        logger.info(f"回测期间: {all_dates[0]} 至 {all_dates[-1]}")
        logger.info(f"总交易日数: {len(all_dates)}, 选股报告日数: {len(reports_data)}")
        
        # 逐日执行回测
        for i, current_date in enumerate(all_dates):
            if i % 20 == 0:
                progress = i / len(all_dates) * 100
                logger.info(f"回测进度: {progress:.1f}% ({current_date.strftime('%Y-%m-%d')})")
            
            current_date_str = current_date.strftime('%Y-%m-%d')
            
            # 获取当日股票数据
            daily_prices = stock_data[stock_data['date'] == current_date]
            daily_prices_dict = daily_prices.set_index('stock_code').to_dict('index')
            
            # 如果有选股报告，执行策略分析
            if current_date_str in reports_data:
                self._execute_strategy_decisions(
                    current_date,
                    reports_data[current_date_str],
                    daily_prices_dict
                )
            
            # 更新持仓和计算组合价值
            self._update_positions_and_portfolio(current_date, daily_prices_dict)
        
        logger.info("策略回测执行完成！")
        
        # 计算绩效指标
        return self._calculate_performance_metrics()
    
    def _execute_strategy_decisions(self, current_date: datetime, report_content: str, daily_prices: Dict):
        """执行策略决策"""
        try:
            # 构建当前持仓信息
            current_positions = self._build_current_positions_data(daily_prices)
            
            # 计算组合价值
            total_value = self.current_capital + sum(
                pos['shares'] * daily_prices.get(code, {}).get('close', 0)
                for code, pos in self.positions.items()
                if code in daily_prices
            )
            
            # 使用策略管理器分析
            analysis_result = self.strategy_manager.analyze_and_decide(
                report_content=report_content,
                current_positions=current_positions,
                portfolio_value=total_value,
                cash_amount=self.current_capital,
                market_conditions=self._get_market_conditions(current_date, daily_prices)
            )
            
            if 'error' in analysis_result:
                logger.warning(f"策略分析失败 {current_date}: {analysis_result['error']}")
                return
            
            # 记录策略决策
            self.strategy_decisions.append({
                'date': current_date,
                'decisions_count': len(analysis_result['trade_decisions']),
                'recommended_stocks': len(analysis_result['recommended_stocks']),
                'portfolio_analysis': analysis_result['portfolio_analysis']
            })
            
            # 执行交易决策
            self._execute_trade_decisions(current_date, analysis_result['trade_decisions'], daily_prices)
            
        except Exception as e:
            logger.error(f"执行策略决策失败 {current_date}: {e}")
    
    def _build_current_positions_data(self, daily_prices: Dict) -> List[Dict]:
        """构建当前持仓数据"""
        positions_data = []
        
        for stock_code, position in self.positions.items():
            if stock_code in daily_prices:
                current_price = daily_prices[stock_code]['close']
                
                positions_data.append({
                    'stock_code': stock_code,
                    'stock_name': position.get('stock_name', stock_code),
                    'shares': position['shares'],
                    'cost_price': position['cost_price'],
                    'current_price': current_price,
                    'market_value': position['shares'] * current_price,
                    'holding_days': position['holding_days']
                })
        
        return positions_data
    
    def _get_market_conditions(self, current_date: datetime, daily_prices: Dict) -> Dict:
        """获取市场条件（简化版）"""
        # 简化的市场条件评估
        return {
            'sentiment': 'NEUTRAL',
            'volatility': 'NORMAL',
            'date': current_date.strftime('%Y-%m-%d')
        }
    
    def _execute_trade_decisions(self, current_date: datetime, decisions: List[Dict], daily_prices: Dict):
        """执行交易决策"""
        for decision in decisions:
            stock_code = decision['stock_code']
            action = decision['action']
            
            if stock_code not in daily_prices:
                continue  # 没有价格数据，跳过
            
            current_price = daily_prices[stock_code]['close']
            
            try:
                if action == 'BUY':
                    self._execute_buy(current_date, stock_code, decision, current_price)
                elif action == 'SELL':
                    self._execute_sell(current_date, stock_code, decision, current_price)
                elif action == 'REDUCE':
                    self._execute_reduce(current_date, stock_code, decision, current_price)
                elif action == 'ADD':
                    self._execute_add(current_date, stock_code, decision, current_price)
                    
            except Exception as e:
                logger.warning(f"执行交易决策失败 {current_date} {stock_code} {action}: {e}")
    
    def _execute_buy(self, date: datetime, stock_code: str, decision: Dict, price: float):
        """执行买入"""
        target_shares = decision.get('target_shares', 0)
        if target_shares <= 0:
            return
        
        # 计算交易成本
        trade_value = target_shares * price
        commission = max(trade_value * self.commission_rate, self.min_commission)
        transfer_fee = trade_value * self.transfer_fee
        total_cost = trade_value + commission + transfer_fee
        
        if total_cost > self.current_capital:
            return  # 资金不足
        
        # 执行买入
        self.positions[stock_code] = {
            'shares': target_shares,
            'cost_price': price,
            'entry_date': date,
            'holding_days': 0,
            'stock_name': stock_code  # 简化处理
        }
        
        self.current_capital -= total_cost
        
        # 记录交易
        self.trades.append({
            'date': date,
            'stock_code': stock_code,
            'action': 'BUY',
            'shares': target_shares,
            'price': price,
            'amount': total_cost,
            'commission': commission,
            'reason': decision.get('reason', 'Unknown'),
            'strategy': self.strategy_name
        })
    
    def _execute_sell(self, date: datetime, stock_code: str, decision: Dict, price: float):
        """执行卖出"""
        if stock_code not in self.positions:
            return
        
        position = self.positions[stock_code]
        shares = position['shares']
        
        # 计算交易成本
        trade_value = shares * price
        commission = max(trade_value * self.commission_rate, self.min_commission)
        stamp_tax = trade_value * self.stamp_tax
        transfer_fee = trade_value * self.transfer_fee
        total_cost = commission + stamp_tax + transfer_fee
        
        proceeds = trade_value - total_cost
        profit = proceeds - (position['cost_price'] * shares)
        
        # 执行卖出
        self.current_capital += proceeds
        del self.positions[stock_code]
        
        # 记录交易
        self.trades.append({
            'date': date,
            'stock_code': stock_code,
            'action': 'SELL',
            'shares': shares,
            'price': price,
            'amount': proceeds,
            'commission': commission,
            'stamp_tax': stamp_tax,
            'profit': profit,
            'profit_pct': profit / (position['cost_price'] * shares),
            'reason': decision.get('reason', 'Unknown'),
            'strategy': self.strategy_name
        })
    
    def _execute_reduce(self, date: datetime, stock_code: str, decision: Dict, price: float):
        """执行减仓"""
        if stock_code not in self.positions:
            return
        
        position = self.positions[stock_code]
        reduce_shares = min(decision.get('target_shares', 0), position['shares'])
        
        if reduce_shares <= 0:
            return
        
        # 计算交易成本
        trade_value = reduce_shares * price
        commission = max(trade_value * self.commission_rate, self.min_commission)
        stamp_tax = trade_value * self.stamp_tax
        transfer_fee = trade_value * self.transfer_fee
        total_cost = commission + stamp_tax + transfer_fee
        
        proceeds = trade_value - total_cost
        profit = proceeds - (position['cost_price'] * reduce_shares)
        
        # 更新持仓
        position['shares'] -= reduce_shares
        self.current_capital += proceeds
        
        # 记录交易
        self.trades.append({
            'date': date,
            'stock_code': stock_code,
            'action': 'REDUCE',
            'shares': reduce_shares,
            'price': price,
            'amount': proceeds,
            'commission': commission,
            'stamp_tax': stamp_tax,
            'profit': profit,
            'profit_pct': profit / (position['cost_price'] * reduce_shares),
            'reason': decision.get('reason', 'Unknown'),
            'strategy': self.strategy_name
        })
    
    def _execute_add(self, date: datetime, stock_code: str, decision: Dict, price: float):
        """执行加仓"""
        if stock_code not in self.positions:
            return
        
        add_shares = decision.get('target_shares', 0)
        if add_shares <= 0:
            return
        
        # 计算交易成本
        trade_value = add_shares * price
        commission = max(trade_value * self.commission_rate, self.min_commission)
        transfer_fee = trade_value * self.transfer_fee
        total_cost = trade_value + commission + transfer_fee
        
        if total_cost > self.current_capital:
            return  # 资金不足
        
        # 更新持仓（重新计算平均成本）
        position = self.positions[stock_code]
        total_cost_basis = position['shares'] * position['cost_price'] + trade_value
        total_shares = position['shares'] + add_shares
        new_avg_cost = total_cost_basis / total_shares
        
        position['shares'] = total_shares
        position['cost_price'] = new_avg_cost
        
        self.current_capital -= total_cost
        
        # 记录交易
        self.trades.append({
            'date': date,
            'stock_code': stock_code,
            'action': 'ADD',
            'shares': add_shares,
            'price': price,
            'amount': total_cost,
            'commission': commission,
            'reason': decision.get('reason', 'Unknown'),
            'strategy': self.strategy_name
        })
    
    def _update_positions_and_portfolio(self, current_date: datetime, daily_prices: Dict):
        """更新持仓和组合价值"""
        # 更新持仓天数
        for stock_code, position in self.positions.items():
            position['holding_days'] = (current_date - position['entry_date']).days
        
        # 计算组合价值
        positions_value = 0
        for stock_code, position in self.positions.items():
            if stock_code in daily_prices:
                current_price = daily_prices[stock_code]['close']
                positions_value += position['shares'] * current_price
        
        total_value = self.current_capital + positions_value
        
        # 记录每日组合状况
        self.daily_portfolio.append({
            'date': current_date,
            'total_value': total_value,
            'cash': self.current_capital,
            'positions_value': positions_value,
            'positions_count': len(self.positions),
            'daily_return': (total_value - self.initial_capital) / self.initial_capital
        })
    
    def _calculate_performance_metrics(self) -> Dict:
        """计算绩效指标"""
        logger.info("计算回测绩效指标...")
        
        if not self.daily_portfolio:
            return {'error': '无回测数据'}
        
        portfolio_df = pd.DataFrame(self.daily_portfolio)
        trades_df = pd.DataFrame(self.trades)
        
        # 基础指标
        final_value = portfolio_df['total_value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        trading_days = len(portfolio_df)
        annual_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0
        
        # 波动率和夏普比率
        portfolio_df['daily_returns'] = portfolio_df['total_value'].pct_change()
        daily_returns = portfolio_df['daily_returns'].dropna()
        volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0
        sharpe_ratio = (annual_return - 0.03) / volatility if volatility > 0 else 0
        
        # 最大回撤
        rolling_max = portfolio_df['total_value'].expanding().max()
        drawdown = (portfolio_df['total_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # 交易统计
        if not trades_df.empty:
            sell_trades = trades_df[trades_df['action'].isin(['SELL', 'REDUCE'])]
            profitable_trades = len(sell_trades[sell_trades['profit'] > 0]) if not sell_trades.empty else 0
            win_rate = profitable_trades / len(sell_trades) if len(sell_trades) > 0 else 0
            
            avg_profit = sell_trades[sell_trades['profit'] > 0]['profit'].mean() if profitable_trades > 0 else 0
            avg_loss = abs(sell_trades[sell_trades['profit'] < 0]['profit'].mean()) if len(sell_trades[sell_trades['profit'] < 0]) > 0 else 1
            profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 0
        else:
            win_rate = 0
            profit_loss_ratio = 0
        
        # 策略统计
        strategy_stats = pd.DataFrame(self.strategy_decisions) if self.strategy_decisions else pd.DataFrame()
        
        results = {
            'strategy_name': self.strategy_name,
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'total_trades': len(trades_df),
            'trading_days': trading_days,
            'final_value': final_value,
            'initial_capital': self.initial_capital,
            'strategy_decisions_count': len(self.strategy_decisions),
            'avg_decisions_per_day': len(self.strategy_decisions) / trading_days if self.strategy_decisions and trading_days > 0 else 0,
            'portfolio_df': portfolio_df,
            'trades_df': trades_df,
            'strategy_stats': strategy_stats
        }
        
        logger.info("绩效指标计算完成")
        return results
    
    def generate_strategy_report(self, results: Dict) -> str:
        """生成策略回测报告"""
        report = f"""
# 📊 {results['strategy_name']} 回测性能报告

## 回测概览
- **策略名称**: {results['strategy_name']}
- **回测期间**: {results['portfolio_df']['date'].min().strftime('%Y-%m-%d')} 至 {results['portfolio_df']['date'].max().strftime('%Y-%m-%d')}
- **初始资金**: {results['initial_capital']:,.0f}元
- **最终资金**: {results['final_value']:,.0f}元
- **交易日数**: {results['trading_days']}天

## 收益表现
| 指标 | 策略表现 | 行业标准 | 评级 |
|------|----------|----------|------|
| 累计收益率 | {results['total_return']:.2%} | >15% | {'A' if results['total_return'] > 0.15 else 'B' if results['total_return'] > 0.08 else 'C'} |
| 年化收益率 | {results['annual_return']:.2%} | >12% | {'A' if results['annual_return'] > 0.12 else 'B' if results['annual_return'] > 0.08 else 'C'} |
| 交易胜率 | {results['win_rate']:.2%} | >50% | {'A' if results['win_rate'] > 0.5 else 'B' if results['win_rate'] > 0.4 else 'C'} |

## 风险指标
| 指标 | 策略数值 | 行业标准 | 评级 |
|------|----------|----------|------|
| 年化波动率 | {results['volatility']:.2%} | <25% | {'A' if results['volatility'] < 0.25 else 'B' if results['volatility'] < 0.35 else 'C'} |
| 最大回撤 | {results['max_drawdown']:.2%} | <15% | {'A' if results['max_drawdown'] > -0.15 else 'B' if results['max_drawdown'] > -0.25 else 'C'} |
| 夏普比率 | {results['sharpe_ratio']:.2f} | >1.0 | {'A' if results['sharpe_ratio'] > 1.0 else 'B' if results['sharpe_ratio'] > 0.5 else 'C'} |

## 策略执行统计
- **总交易次数**: {results['total_trades']}次
- **策略决策次数**: {results['strategy_decisions_count']}次
- **平均每日决策数**: {results.get('avg_decisions_per_day', 0):.1f}个
- **交易胜率**: {results['win_rate']:.2%}
- **盈亏比**: {results['profit_loss_ratio']:.2f}

## 策略特色分析
本次回测使用了**{results['strategy_name']}**，该策略的特点是基于选股报告和当前持仓情况做出智能交易决策，相比传统的机械化交易策略，具有更强的适应性和决策灵活性。

## 📈 策略评级
"""
        
        # 计算综合评级
        score = 0
        if results['total_return'] > 0.15: score += 25
        elif results['total_return'] > 0.08: score += 15
        
        if results['max_drawdown'] > -0.15: score += 25
        elif results['max_drawdown'] > -0.25: score += 15
        
        if results['sharpe_ratio'] > 1.0: score += 25
        elif results['sharpe_ratio'] > 0.5: score += 15
        
        if results['win_rate'] > 0.5: score += 25
        elif results['win_rate'] > 0.4: score += 15
        
        if score >= 80:
            grade = "A+ (优秀)"
        elif score >= 60:
            grade = "A (良好)"
        elif score >= 40:
            grade = "B (一般)"
        else:
            grade = "C (需改进)"
        
        report += f"**综合评级**: {grade} (评分: {score}/100)\n\n"
        
        report += f"""
## ⚠️ 风险提示
- 本回测结果基于历史数据和选股报告，不代表未来表现
- 策略决策的有效性依赖于选股报告的质量
- 实盘交易存在滑点、冲击成本等额外风险
- 建议从小资金开始验证策略有效性

## 📊 数据来源
- **策略类型**: {results['strategy_name']}
- **选股信号**: 基于量化选股报告的策略决策
- **交易成本**: A股实际成本模型（佣金万三+印花税千一）
- **分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🤖 Generated by Strategy Backtester
"""
        
        return report
    
    def save_results(self, results: Dict, output_dir: str = "backtest/strategy_results"):
        """保存策略回测结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        strategy_name_safe = self.strategy_name.replace('型策略', '').replace(' ', '_')
        
        # 保存组合净值数据
        portfolio_file = output_path / f"portfolio_{strategy_name_safe}_{timestamp}.csv"
        results['portfolio_df'].to_csv(portfolio_file, index=False, encoding='utf-8')
        
        # 保存交易记录
        trades_file = output_path / f"trades_{strategy_name_safe}_{timestamp}.csv"
        results['trades_df'].to_csv(trades_file, index=False, encoding='utf-8')
        
        # 保存策略决策记录
        if not results['strategy_stats'].empty:
            strategy_file = output_path / f"strategy_decisions_{strategy_name_safe}_{timestamp}.csv"
            results['strategy_stats'].to_csv(strategy_file, index=False, encoding='utf-8')
        
        # 保存回测报告
        report = self.generate_strategy_report(results)
        report_file = output_path / f"strategy_report_{strategy_name_safe}_{timestamp}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"策略回测结果已保存到: {output_path}")
        return output_path

if __name__ == "__main__":
    # 示例使用
    print("策略回测引擎已初始化")
    print("使用方法:")
    print("1. backtester = StrategyBacktester(strategy_name='平衡型策略')")
    print("2. stock_data, reports_data = backtester.load_data(...)")
    print("3. results = backtester.run_backtest(stock_data, reports_data)")
    print("4. backtester.save_results(results)")