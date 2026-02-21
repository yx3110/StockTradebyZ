#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
50万资金虚拟交易回测系统
从2025年8月1日开始交易，生成详细的交易流程和持仓报告
"""

import sys
import os
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data_adapter.database_manager import DatabaseManager
from stock_selctor.Selector import (
    BBIKDJSelector, BBIShortLongSelector, 
    BreakoutVolumeKDJSelector, PeakKDJSelector
)

class VirtualTradingAccount:
    """虚拟交易账户管理"""
    
    def __init__(self, initial_capital: float = 500000):
        """
        初始化账户
        Args:
            initial_capital: 初始资金（默认50万）
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # {stock_code: {'shares': 数量, 'cost': 成本, 'entry_date': 日期}}
        self.trade_history = []  # 交易记录
        self.daily_netvalue = []  # 每日净值记录
        self.db_manager = DatabaseManager()
        
    def get_stock_price(self, stock_code: str, date: str) -> float:
        """获取股票在指定日期的收盘价"""
        # 转换日期格式 20250801 -> 2025-08-01
        formatted_date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
        
        with self.db_manager.get_connection() as conn:
            query = """
            SELECT close FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE s.code = ? AND date(dq.trade_date) = ?
            """
            result = pd.read_sql_query(query, conn, params=[stock_code, formatted_date])
            if not result.empty:
                return float(result.iloc[0]['close'])
        return 0
    
    def buy_stock(self, stock_code: str, shares: int, price: float, date: str) -> bool:
        """买入股票"""
        cost = shares * price * 1.001  # 加上0.1%手续费
        
        if cost > self.cash:
            return False  # 资金不足
        
        self.cash -= cost
        
        if stock_code in self.positions:
            # 加仓
            old_shares = self.positions[stock_code]['shares']
            old_cost = self.positions[stock_code]['cost']
            new_shares = old_shares + shares
            new_cost = (old_cost * old_shares + price * shares) / new_shares
            self.positions[stock_code] = {
                'shares': new_shares,
                'cost': new_cost,
                'entry_date': self.positions[stock_code]['entry_date']
            }
        else:
            # 新建仓
            self.positions[stock_code] = {
                'shares': shares,
                'cost': price,
                'entry_date': date
            }
        
        # 记录交易
        self.trade_history.append({
            'date': date,
            'stock_code': stock_code,
            'action': 'BUY',
            'shares': shares,
            'price': price,
            'cost': cost,
            'cash_after': self.cash
        })
        
        return True
    
    def sell_stock(self, stock_code: str, shares: int, price: float, date: str) -> bool:
        """卖出股票"""
        if stock_code not in self.positions:
            return False
        
        if self.positions[stock_code]['shares'] < shares:
            return False
        
        # 计算收入（扣除手续费和印花税）
        income = shares * price * (1 - 0.001 - 0.001)  # 0.1%手续费 + 0.1%印花税
        self.cash += income
        
        # 更新持仓
        self.positions[stock_code]['shares'] -= shares
        if self.positions[stock_code]['shares'] == 0:
            del self.positions[stock_code]
        
        # 记录交易
        self.trade_history.append({
            'date': date,
            'stock_code': stock_code,
            'action': 'SELL',
            'shares': shares,
            'price': price,
            'income': income,
            'cash_after': self.cash
        })
        
        return True
    
    def get_portfolio_value(self, date: str) -> float:
        """计算组合总市值"""
        total_value = self.cash
        
        for stock_code, position in self.positions.items():
            current_price = self.get_stock_price(stock_code, date)
            if current_price > 0:
                total_value += position['shares'] * current_price
        
        return total_value
    
    def record_daily_netvalue(self, date: str):
        """记录每日净值"""
        portfolio_value = self.get_portfolio_value(date)
        netvalue = portfolio_value / self.initial_capital
        
        self.daily_netvalue.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'netvalue': netvalue,
            'cash': self.cash,
            'positions_value': portfolio_value - self.cash,
            'positions_count': len(self.positions)
        })

class AITradingSimulator:
    """AI驱动的交易模拟器"""
    
    def __init__(self, initial_capital: float = 500000):
        self.account = VirtualTradingAccount(initial_capital)
        self.db_manager = DatabaseManager()
        self.selectors = {
            'bbi_kdj': BBIKDJSelector(),
            'bbi_shortlong': BBIShortLongSelector(),
            'breakout_volume': BreakoutVolumeKDJSelector(),
            'peak_kdj': PeakKDJSelector()
        }
        
    def get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取交易日期列表"""
        # 转换日期格式 20250801 -> 2025-08-01
        formatted_start = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
        formatted_end = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
        
        with self.db_manager.get_connection() as conn:
            query = """
            SELECT DISTINCT date(trade_date) as trade_date
            FROM daily_quotes 
            WHERE date(trade_date) >= ? AND date(trade_date) <= ?
            GROUP BY date(trade_date)
            HAVING COUNT(*) > 1000
            ORDER BY trade_date
            """
            result = pd.read_sql_query(query, conn, params=[formatted_start, formatted_end])
            # 转换回YYYYMMDD格式
            dates = []
            for date_str in result['trade_date'].tolist():
                dates.append(date_str.replace('-', ''))
            return dates
    
    def load_ai_report(self, date: str) -> Dict:
        """加载AI报告中的推荐股票"""
        # 转换日期格式
        formatted_date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
        report_path = f"reports/ai_enhanced/AI增强选股报告_{formatted_date}.md"
        
        recommendations = {}
        
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 解析AI报告
            lines = content.split('\n')
            current_stock = None
            
            for line in lines:
                if '### ' in line and ' - ' in line:
                    parts = line.split(' - ')
                    if len(parts) >= 2:
                        stock_code = parts[0].replace('### ', '').strip()
                        current_stock = stock_code
                        recommendations[stock_code] = {
                            'action': 'HOLD',
                            'score': 50,
                            'target_price': 0,
                            'stop_loss': 0
                        }
                
                elif current_stock:
                    if '**Claude评分**:' in line:
                        try:
                            score = float(line.split('**Claude评分**:')[1].split('分')[0].strip())
                            recommendations[current_stock]['score'] = score
                        except:
                            pass
                    
                    elif '**建议操作**:' in line or '**投资评级**:' in line:
                        if '买入' in line:
                            recommendations[current_stock]['action'] = 'BUY'
                        elif '卖出' in line:
                            recommendations[current_stock]['action'] = 'SELL'
                    
                    elif '**目标价位**:' in line:
                        try:
                            price = float(line.split('**目标价位**:')[1].split('元')[0].strip())
                            recommendations[current_stock]['target_price'] = price
                        except:
                            pass
        
        return recommendations
    
    def generate_trading_signals(self, date: str) -> List[Dict]:
        """生成交易信号"""
        signals = []
        
        # 1. 加载AI报告推荐
        ai_recommendations = self.load_ai_report(date)
        
        # 2. 获取当日可交易的股票
        # 转换日期格式
        formatted_date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
        
        with self.db_manager.get_connection() as conn:
            query = """
            SELECT DISTINCT s.code, s.name
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE date(dq.trade_date) = ? AND s.type = 'A股'
            LIMIT 100
            """
            stocks = pd.read_sql_query(query, conn, params=[formatted_date])
        
        # 3. 生成信号
        for _, stock in stocks.iterrows():
            stock_code = stock['code']
            
            # 默认信号
            signal = {
                'stock_code': stock_code,
                'stock_name': stock['name'],
                'action': 'HOLD',
                'score': 50,
                'confidence': 0.5
            }
            
            # 如果在AI推荐中
            if stock_code in ai_recommendations:
                ai_rec = ai_recommendations[stock_code]
                signal['action'] = ai_rec['action']
                signal['score'] = ai_rec['score']
                signal['confidence'] = ai_rec['score'] / 100
            
            # 只添加买入或卖出信号
            if signal['action'] != 'HOLD':
                signals.append(signal)
        
        # 按评分排序
        signals.sort(key=lambda x: x['score'], reverse=True)
        
        return signals[:10]  # 返回前10个信号
    
    def execute_trading_day(self, date: str):
        """执行单日交易"""
        print(f"\n{'='*60}")
        print(f"📅 交易日期: {date}")
        print(f"{'='*60}")
        
        # 1. 记录开盘前状态
        portfolio_value = self.account.get_portfolio_value(date)
        print(f"💰 账户资金: {self.account.cash:,.0f}元")
        print(f"📊 持仓市值: {portfolio_value - self.account.cash:,.0f}元")
        print(f"💎 总市值: {portfolio_value:,.0f}元")
        print(f"📈 收益率: {(portfolio_value/self.account.initial_capital - 1)*100:.2f}%")
        
        # 2. 生成交易信号
        signals = self.generate_trading_signals(date)
        
        if not signals:
            print("🔍 今日无交易信号")
            self.account.record_daily_netvalue(date)
            return
        
        print(f"\n🎯 今日交易信号 ({len(signals)}个):")
        
        # 3. 执行交易
        for signal in signals:
            stock_code = signal['stock_code']
            stock_name = signal['stock_name']
            action = signal['action']
            score = signal['score']
            
            current_price = self.account.get_stock_price(stock_code, date)
            if current_price <= 0:
                continue
            
            if action == 'BUY' and score > 65:
                # 计算买入数量（每次使用5%资金）
                position_size = min(self.account.cash * 0.05, 50000)  # 最多5万每只
                shares = int(position_size / current_price / 100) * 100
                
                if shares >= 100:
                    if self.account.buy_stock(stock_code, shares, current_price, date):
                        print(f"  ✅ 买入: {stock_code} {stock_name} "
                              f"{shares}股 @ {current_price:.2f}元 (评分:{score:.0f})")
            
            elif action == 'SELL' and stock_code in self.account.positions:
                position = self.account.positions[stock_code]
                shares = position['shares']
                
                if self.account.sell_stock(stock_code, shares, current_price, date):
                    pnl = (current_price - position['cost']) * shares
                    pnl_pct = (current_price / position['cost'] - 1) * 100
                    print(f"  ❌ 卖出: {stock_code} {stock_name} "
                          f"{shares}股 @ {current_price:.2f}元 "
                          f"(盈亏:{pnl:+,.0f}元 {pnl_pct:+.1f}%)")
        
        # 4. 风险控制：检查止损
        for stock_code in list(self.account.positions.keys()):
            position = self.account.positions[stock_code]
            current_price = self.account.get_stock_price(stock_code, date)
            
            if current_price > 0:
                loss_pct = (current_price / position['cost'] - 1) * 100
                
                # 8%止损
                if loss_pct <= -8:
                    shares = position['shares']
                    if self.account.sell_stock(stock_code, shares, current_price, date):
                        print(f"  🛑 止损: {stock_code} {shares}股 @ {current_price:.2f}元 "
                              f"(亏损:{loss_pct:.1f}%)")
        
        # 5. 记录每日净值
        self.account.record_daily_netvalue(date)
        
        # 6. 显示当前持仓
        if self.account.positions:
            print(f"\n📋 当前持仓 ({len(self.account.positions)}只):")
            for stock_code, position in list(self.account.positions.items())[:5]:
                current_price = self.account.get_stock_price(stock_code, date)
                if current_price > 0:
                    pnl_pct = (current_price / position['cost'] - 1) * 100
                    market_value = position['shares'] * current_price
                    print(f"  • {stock_code}: {position['shares']}股 "
                          f"成本{position['cost']:.2f} 现价{current_price:.2f} "
                          f"({pnl_pct:+.1f}%) 市值{market_value:,.0f}元")
    
    def run_simulation(self, start_date: str, end_date: str):
        """运行完整模拟"""
        print(f"\n{'='*80}")
        print(f"🚀 启动50万资金虚拟交易系统")
        print(f"{'='*80}")
        print(f"📅 回测期间: {start_date} 至 {end_date}")
        print(f"💰 初始资金: {self.account.initial_capital:,.0f}元")
        
        # 获取所有交易日
        trading_dates = self.get_trading_dates(start_date, end_date)
        print(f"📊 交易日数: {len(trading_dates)}天")
        
        # 逐日执行交易
        for i, date in enumerate(trading_dates, 1):
            print(f"\n进度: {i}/{len(trading_dates)}")
            self.execute_trading_day(date)
            
            # 每5天显示一次汇总
            if i % 5 == 0:
                portfolio_value = self.account.get_portfolio_value(date)
                total_return = (portfolio_value / self.account.initial_capital - 1) * 100
                print(f"\n📊 阶段汇总: 总收益率 {total_return:+.2f}% "
                      f"总市值 {portfolio_value:,.0f}元")
        
        # 生成最终报告
        self.generate_final_report(end_date)
    
    def generate_final_report(self, end_date: str):
        """生成最终报告"""
        portfolio_value = self.account.get_portfolio_value(end_date)
        total_return = (portfolio_value / self.account.initial_capital - 1) * 100
        
        report = f"""
{'='*80}
📊 50万资金虚拟交易最终报告
{'='*80}

## 📈 收益统计
- 初始资金: {self.account.initial_capital:,.0f}元
- 最终市值: {portfolio_value:,.0f}元
- 总收益: {portfolio_value - self.account.initial_capital:+,.0f}元
- 收益率: {total_return:+.2f}%
- 现金余额: {self.account.cash:,.0f}元
- 持仓市值: {portfolio_value - self.account.cash:,.0f}元

## 📊 交易统计
- 总交易次数: {len(self.account.trade_history)}
- 买入次数: {sum(1 for t in self.account.trade_history if t['action'] == 'BUY')}
- 卖出次数: {sum(1 for t in self.account.trade_history if t['action'] == 'SELL')}
- 当前持仓: {len(self.account.positions)}只

## 📋 最终持仓明细
"""
        
        for stock_code, position in self.account.positions.items():
            current_price = self.account.get_stock_price(stock_code, end_date)
            if current_price > 0:
                pnl = (current_price - position['cost']) * position['shares']
                pnl_pct = (current_price / position['cost'] - 1) * 100
                market_value = position['shares'] * current_price
                
                report += f"""
股票代码: {stock_code}
- 持仓数量: {position['shares']}股
- 成本价格: {position['cost']:.2f}元
- 当前价格: {current_price:.2f}元
- 持仓盈亏: {pnl:+,.0f}元 ({pnl_pct:+.1f}%)
- 市值: {market_value:,.0f}元
"""
        
        # 计算风险指标
        if self.account.daily_netvalue:
            netvalues = [d['netvalue'] for d in self.account.daily_netvalue]
            max_drawdown = self.calculate_max_drawdown(netvalues)
            sharpe_ratio = self.calculate_sharpe_ratio(netvalues)
            
            report += f"""
## 📊 风险指标
- 最大回撤: {max_drawdown:.2f}%
- 夏普比率: {sharpe_ratio:.3f}
- 交易天数: {len(self.account.daily_netvalue)}
"""
        
        # 保存报告
        report_path = f"reports/backtest/虚拟交易报告_50万_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        print(f"\n📄 详细报告已保存: {report_path}")
        
        # 保存交易记录
        if self.account.trade_history:
            trades_df = pd.DataFrame(self.account.trade_history)
            trades_path = f"reports/backtest/交易记录_50万_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            trades_df.to_csv(trades_path, index=False, encoding='utf-8-sig')
            print(f"📄 交易记录已保存: {trades_path}")
        
        # 保存净值曲线
        if self.account.daily_netvalue:
            netvalue_df = pd.DataFrame(self.account.daily_netvalue)
            netvalue_path = f"reports/backtest/净值曲线_50万_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            netvalue_df.to_csv(netvalue_path, index=False, encoding='utf-8-sig')
            print(f"📄 净值曲线已保存: {netvalue_path}")
    
    def calculate_max_drawdown(self, netvalues: List[float]) -> float:
        """计算最大回撤"""
        peak = netvalues[0]
        max_dd = 0
        
        for value in netvalues:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def calculate_sharpe_ratio(self, netvalues: List[float], risk_free_rate: float = 0.03) -> float:
        """计算夏普比率"""
        if len(netvalues) < 2:
            return 0
        
        returns = [(netvalues[i] - netvalues[i-1]) / netvalues[i-1] 
                  for i in range(1, len(netvalues))]
        
        if not returns:
            return 0
        
        avg_return = np.mean(returns) * 252  # 年化
        std_return = np.std(returns) * np.sqrt(252)  # 年化
        
        if std_return == 0:
            return 0
        
        return (avg_return - risk_free_rate) / std_return

def main():
    """主函数"""
    # 创建模拟器
    simulator = AITradingSimulator(initial_capital=500000)
    
    # 设置回测期间（使用8月数据）
    start_date = '20250801'
    end_date = '20250811'
    
    # 运行模拟
    simulator.run_simulation(start_date, end_date)
    
    return simulator

if __name__ == "__main__":
    simulator = main()