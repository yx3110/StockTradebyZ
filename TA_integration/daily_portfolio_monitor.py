#!/usr/bin/env python3
"""
每日投资组合监控脚本
基于ChatGPT Micro-Cap Experiment的思路，实现自动止损和每日更新
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import os
import logging
import sys
import argparse

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_portfolio_manager import AIPortfolioManager, Position

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DailyPortfolioMonitor:
    """每日投资组合监控器"""
    
    def __init__(self, 
                 portfolio_csv: str = "TA_integration/data/ai_portfolio.csv",
                 trade_log_csv: str = "TA_integration/data/ai_trade_log.csv",
                 data_dir: str = "full_securities_data"):
        """
        初始化监控器
        
        Args:
            portfolio_csv: AI投资组合CSV文件路径
            trade_log_csv: 交易日志CSV文件路径
            data_dir: 股票数据目录
        """
        self.portfolio_csv = portfolio_csv
        self.trade_log_csv = trade_log_csv
        self.data_dir = data_dir
        self.today = datetime.now().strftime("%Y-%m-%d")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(portfolio_csv), exist_ok=True)
        os.makedirs(os.path.dirname(trade_log_csv), exist_ok=True)
        
        # 初始化AI管理器
        self.ai_manager = AIPortfolioManager()
    
    def load_portfolio(self) -> pd.DataFrame:
        """加载当前投资组合"""
        if os.path.exists(self.portfolio_csv):
            return pd.read_csv(self.portfolio_csv)
        else:
            # 创建空的投资组合
            logger.warning(f"投资组合文件不存在，创建新文件: {self.portfolio_csv}")
            df = pd.DataFrame(columns=['ticker', 'shares', 'buy_price', 'stop_loss', 'target_price', 'entry_date'])
            df.to_csv(self.portfolio_csv, index=False)
            return df
    
    def get_latest_price(self, ticker: str) -> Optional[Tuple[float, str]]:
        """获取股票最新价格和日期"""
        # 从本地数据文件读取
        possible_files = [
            os.path.join(self.data_dir, f"{ticker}_A股.csv"),
            os.path.join(self.data_dir, f"{ticker}_ETF.csv"),
            os.path.join(self.data_dir, f"{ticker}_基金.csv"),
            os.path.join(self.data_dir, f"{ticker}.csv")  # 通用格式
        ]
        
        for stock_file in possible_files:
            if os.path.exists(stock_file):
                try:
                    df = pd.read_csv(stock_file)
                    if not df.empty:
                        # 获取最新日期的收盘价
                        df['date'] = pd.to_datetime(df['date'])
                        latest_row = df.sort_values('date').iloc[-1]
                        price = float(latest_row['close'])
                        date = latest_row['date'].strftime('%Y-%m-%d')
                        return price, date
                except Exception as e:
                    logger.error(f"读取{ticker}价格失败: {e}")
                    continue
        
        # 如果本地文件不存在，尝试使用tushare实时获取
        try:
            import tushare as ts
            from config import config
            
            # 初始化tushare
            token = config.get('tushare', {}).get('token')
            if token:
                ts.set_token(token)
                pro = ts.pro_api()
                
                # 获取最新交易日数据
                df = pro.daily(ts_code=f"{ticker}.SZ", limit=1)
                if df.empty:
                    df = pro.daily(ts_code=f"{ticker}.SH", limit=1)
                
                if not df.empty:
                    price = float(df.iloc[0]['close'])
                    date = df.iloc[0]['trade_date']
                    # 转换日期格式 20250801 -> 2025-08-01
                    date = pd.to_datetime(date).strftime('%Y-%m-%d')
                    return price, date
        except Exception as e:
            logger.warning(f"Tushare获取{ticker}价格失败: {e}")
        
        return None
    
    def check_stop_loss(self, portfolio: pd.DataFrame) -> List[Dict]:
        """检查止损触发"""
        triggered_stops = []
        
        for idx, row in portfolio.iterrows():
            ticker = row['ticker']
            shares = row['shares']
            stop_loss = row['stop_loss']
            buy_price = row['buy_price']
            
            # 获取最新价格
            price_data = self.get_latest_price(ticker)
            if price_data is None:
                logger.warning(f"无法获取{ticker}的最新价格")
                continue
            
            current_price, price_date = price_data
            
            # 检查是否触发止损
            if current_price <= stop_loss:
                triggered_stops.append({
                    'ticker': ticker,
                    'shares': shares,
                    'current_price': current_price,
                    'stop_loss': stop_loss,
                    'buy_price': buy_price,
                    'loss_pct': (current_price - buy_price) / buy_price,
                    'reason': '自动止损触发'
                })
                logger.info(f"⚠️ {ticker}触发止损: 当前价{current_price:.2f} <= 止损价{stop_loss:.2f}")
        
        return triggered_stops
    
    def log_trade(self, trade_type: str, ticker: str, shares: int, 
                  price: float, reason: str, pnl: float = 0.0):
        """记录交易日志"""
        trade_log = {
            'Date': self.today,
            'Type': trade_type,
            'Ticker': ticker,
            'Shares': shares,
            'Price': price,
            'PnL': pnl,
            'Reason': reason
        }
        
        # 追加到交易日志
        if os.path.exists(self.trade_log_csv):
            df = pd.read_csv(self.trade_log_csv)
            df = pd.concat([df, pd.DataFrame([trade_log])], ignore_index=True)
        else:
            df = pd.DataFrame([trade_log])
        
        df.to_csv(self.trade_log_csv, index=False)
        logger.info(f"交易记录: {trade_type} {shares}股 {ticker} @ ¥{price:.2f}")
    
    def execute_stop_loss(self, portfolio: pd.DataFrame, triggered_stops: List[Dict]) -> pd.DataFrame:
        """执行止损卖出"""
        for stop in triggered_stops:
            ticker = stop['ticker']
            shares = stop['shares']
            price = stop['current_price']
            buy_price = stop['buy_price']
            pnl = (price - buy_price) * shares
            
            # 记录卖出交易
            self.log_trade('SELL', ticker, shares, price, stop['reason'], pnl)
            
            # 从投资组合中移除
            portfolio = portfolio[portfolio['ticker'] != ticker]
            
            logger.info(f"💔 止损卖出: {ticker} {shares}股 @ ¥{price:.2f}, 损失¥{-pnl:.2f}")
        
        return portfolio
    
    def update_portfolio_metrics(self, portfolio: pd.DataFrame) -> pd.DataFrame:
        """更新投资组合指标"""
        # 获取每只股票的最新价格
        def get_price_only(ticker):
            price_data = self.get_latest_price(ticker)
            return price_data[0] if price_data else np.nan
        
        portfolio['current_price'] = portfolio['ticker'].apply(get_price_only)
        portfolio['value'] = portfolio['shares'] * portfolio['current_price']
        portfolio['pnl'] = (portfolio['current_price'] - portfolio['buy_price']) * portfolio['shares']
        portfolio['pnl_pct'] = (portfolio['current_price'] - portfolio['buy_price']) / portfolio['buy_price']
        portfolio['distance_to_stop'] = (portfolio['current_price'] - portfolio['stop_loss']) / portfolio['current_price']
        portfolio['distance_to_target'] = (portfolio['target_price'] - portfolio['current_price']) / portfolio['current_price']
        
        return portfolio
    
    def generate_daily_summary(self, portfolio: pd.DataFrame, 
                             triggered_stops: List[Dict],
                             cash_balance: float = 10000.0) -> str:
        """生成每日监控报告"""
        # 计算组合统计
        total_value = portfolio['value'].sum() if not portfolio.empty else 0
        total_equity = total_value + cash_balance
        
        report = f"""# 📊 每日投资组合监控报告

## 📅 监控日期: {self.today}

### 💼 持仓概览
- **持仓数量**: {len(portfolio)}只
- **持仓市值**: ¥{total_value:,.2f}
- **现金余额**: ¥{cash_balance:,.2f}
- **总权益**: ¥{total_equity:,.2f}

### ⚠️ 止损触发 ({len(triggered_stops)}只)
"""
        
        if triggered_stops:
            for stop in triggered_stops:
                report += f"""
**{stop['ticker']}**:
- 触发价格: ¥{stop['current_price']:.2f}
- 止损价格: ¥{stop['stop_loss']:.2f}
- 损失比例: {stop['loss_pct']:.1%}
"""
        else:
            report += "\n无止损触发\n"
        
        # 当前持仓详情
        report += "\n### 📈 当前持仓\n"
        if not portfolio.empty:
            portfolio_sorted = portfolio.sort_values('pnl_pct', ascending=False)
            
            for _, pos in portfolio_sorted.iterrows():
                emoji = "🟢" if pos['pnl_pct'] > 0 else "🔴"
                report += f"""
**{pos['ticker']}** {emoji}
- 当前价: ¥{pos['current_price']:.2f} ({pos['pnl_pct']:+.1%})
- 持仓: {int(pos['shares'])}股 (¥{pos['value']:,.0f})
- 止损距离: {pos['distance_to_stop']:.1%}
- 目标距离: {pos['distance_to_target']:+.1%}
"""
        
        # 性能指标
        report += f"""
### 📊 性能指标
- **当日涨跌**: 待计算
- **累计收益**: 待计算
- **最大回撤**: 待计算
- **夏普比率**: 待计算

### 💡 AI建议
基于当前市场状况和技术指标，建议：
1. 密切关注接近止损的股票
2. 考虑对表现优异的股票加仓
3. 评估是否需要调整止损位置

---
🤖 自动监控系统 | 灵感来源: ChatGPT Micro-Cap Experiment
"""
        return report
    
    def run_daily_update(self, execute_trades: bool = False):
        """运行每日更新"""
        logger.info(f"开始每日投资组合监控 - {self.today}")
        
        # 加载投资组合
        portfolio = self.load_portfolio()
        
        if portfolio.empty:
            logger.info("投资组合为空，跳过监控")
            return
        
        # 检查止损
        triggered_stops = self.check_stop_loss(portfolio)
        
        # 执行止损（如果启用）
        if execute_trades and triggered_stops:
            portfolio = self.execute_stop_loss(portfolio, triggered_stops)
            # 保存更新后的投资组合
            portfolio.to_csv(self.portfolio_csv, index=False)
        
        # 更新投资组合指标
        portfolio = self.update_portfolio_metrics(portfolio)
        
        # 生成报告
        report = self.generate_daily_summary(portfolio, triggered_stops)
        
        # 保存报告
        report_dir = "reports/daily_monitor"
        os.makedirs(report_dir, exist_ok=True)
        report_file = os.path.join(report_dir, f"monitor_{self.today.replace('-', '')}.md")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"监控报告已保存: {report_file}")
        
        # 输出关键信息
        print(f"\n📊 每日监控完成 - {self.today}")
        print(f"⚠️  止损触发: {len(triggered_stops)}只")
        print(f"💼 当前持仓: {len(portfolio)}只")
        print(f"📝 报告位置: {report_file}")
        
        return portfolio, triggered_stops


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='每日投资组合监控')
    parser.add_argument('--execute', action='store_true', 
                       help='执行实际交易（卖出触发止损的股票）')
    parser.add_argument('--portfolio', type=str, 
                       default="TA_integration/data/ai_portfolio.csv",
                       help='投资组合CSV文件路径')
    
    args = parser.parse_args()
    
    # 创建监控器
    monitor = DailyPortfolioMonitor(portfolio_csv=args.portfolio)
    
    # 运行每日更新
    monitor.run_daily_update(execute_trades=args.execute)
    
    # 提示用户
    if not args.execute:
        print("\n💡 提示: 使用 --execute 参数来执行实际的止损卖出")


if __name__ == "__main__":
    main()