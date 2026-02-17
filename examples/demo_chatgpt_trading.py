#!/usr/bin/env python3
"""
ChatGPT辅助交易功能演示
展示ChatGPT-Micro-Cap-Experiment项目的功能
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path

def demo_chatgpt_trading_features():
    """演示ChatGPT交易功能"""
    
    print("🤖 ChatGPT辅助交易系统演示")
    print("=" * 50)
    
    # 1. 读取现有的交易日志
    print("\n📊 1. 查看历史交易记录")
    trade_log_path = Path("ChatGPT-Micro-Cap-Experiment/Scripts and CSV Files/chatgpt_trade_log.csv")
    
    if trade_log_path.exists():
        trade_log = pd.read_csv(trade_log_path)
        print(f"   找到 {len(trade_log)} 条交易记录")
        
        if len(trade_log) > 0:
            print("\n   最近5笔交易:")
            # 检查实际的列名
            print("   列名:", list(trade_log.columns))
            if 'Date' in trade_log.columns and 'Ticker' in trade_log.columns:
                display_cols = [col for col in ['Date', 'Ticker', 'Reason', 'PnL'] if col in trade_log.columns]
                print(trade_log[display_cols].tail())
            
            # 统计交易表现 - 基于实际的数据结构
            buy_trades = trade_log[trade_log['Shares Bought'].notna()]
            sell_trades = trade_log[trade_log['Shares Sold'].notna()]
            
            print(f"\n   交易统计:")
            print(f"   - 买入交易: {len(buy_trades)} 笔")
            print(f"   - 卖出交易: {len(sell_trades)} 笔")
            
            if len(sell_trades) > 0 and 'PnL' in trade_log.columns:
                # 只统计卖出交易的盈亏
                sell_pnl = sell_trades['PnL'].dropna()
                if len(sell_pnl) > 0:
                    total_pnl = sell_pnl.sum()
                    win_trades = len(sell_pnl[sell_pnl > 0])
                    win_rate = win_trades / len(sell_pnl) * 100
                    
                    print(f"   - 总盈亏: ${total_pnl:.2f}")
                    print(f"   - 胜率: {win_rate:.1f}%")
    else:
        print("   未找到交易记录文件")
    
    # 2. 读取投资组合更新
    print("\n💼 2. 查看投资组合表现")
    portfolio_path = Path("ChatGPT-Micro-Cap-Experiment/Scripts and CSV Files/chatgpt_portfolio_update.csv")
    
    if portfolio_path.exists():
        portfolio_df = pd.read_csv(portfolio_path)
        print(f"   找到 {len(portfolio_df)} 个投资组合更新记录")
        
        if len(portfolio_df) > 0:
            latest = portfolio_df.iloc[-1]
            first = portfolio_df.iloc[0]
            
            print(f"\n   投资组合概览:")
            print(f"   - 起始日期: {first['Date']}")
            print(f"   - 最新日期: {latest['Date']}")
            
            # 检查可用的列
            print("   投资组合文件列名:", list(portfolio_df.columns))
            
            if 'Total Equity' in portfolio_df.columns:
                first_equity = first['Total Equity'] if pd.notna(first['Total Equity']) else 0
                latest_equity = latest['Total Equity'] if pd.notna(latest['Total Equity']) else 0
                
                if first_equity > 0 and latest_equity > 0:
                    print(f"   - 初始资产: ${first_equity:.2f}")
                    print(f"   - 当前资产: ${latest_equity:.2f}")
                    total_return = (latest_equity - first_equity) / first_equity * 100
                    print(f"   - 总收益率: {total_return:.2f}%")
            
            # 显示最新持仓 - 基于实际的CSV结构
            print(f"\n   最新持仓情况:")
            # 按日期分组获取最新的持仓
            latest_date = portfolio_df['Date'].max()
            latest_positions = portfolio_df[portfolio_df['Date'] == latest_date]
            
            if len(latest_positions) > 0:
                for _, position in latest_positions.iterrows():
                    if position['Shares'] > 0:
                        ticker = position['Ticker']
                        shares = position['Shares']
                        price = position['Current Price']
                        value = position['Total Value']
                        pnl = position['PnL']
                        
                        print(f"   - {ticker}: {shares:.0f}股 @ ${price:.2f} = ${value:.2f} (盈亏: ${pnl:.2f})")
    else:
        print("   未找到投资组合文件")
    
    # 3. 演示股票数据获取功能
    print("\n📈 3. 演示实时股票数据获取")
    
    # 选择一些示例股票
    demo_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    print(f"   获取示例股票数据: {', '.join(demo_tickers)}")
    
    try:
        for ticker in demo_tickers[:3]:  # 只演示前3个避免API限制
            stock = yf.Ticker(ticker)
            info = stock.info
            
            current_price = info.get('currentPrice', 0)
            market_cap = info.get('marketCap', 0)
            pe_ratio = info.get('trailingPE', 0)
            
            print(f"   - {ticker}: ${current_price:.2f}, 市值: ${market_cap/1e9:.1f}B, P/E: {pe_ratio:.1f}")
            
    except Exception as e:
        print(f"   股票数据获取失败: {e}")
    
    # 4. 演示风险管理功能
    print("\n⚠️  4. 风险管理演示")
    
    # 模拟一个简单的投资组合
    demo_portfolio = {
        'AAPL': {'shares': 10, 'buy_price': 150.0, 'stop_loss': 135.0},
        'MSFT': {'shares': 15, 'buy_price': 300.0, 'stop_loss': 270.0},
        'GOOGL': {'shares': 5, 'buy_price': 120.0, 'stop_loss': 108.0}
    }
    
    print("   示例投资组合风险分析:")
    
    total_investment = 0
    total_risk = 0
    
    for ticker, position in demo_portfolio.items():
        investment = position['shares'] * position['buy_price']
        max_loss = position['shares'] * (position['buy_price'] - position['stop_loss'])
        risk_pct = (max_loss / investment) * 100
        
        total_investment += investment
        total_risk += max_loss
        
        print(f"   - {ticker}: 投资${investment:.0f}, 最大亏损${max_loss:.0f} ({risk_pct:.1f}%)")
    
    portfolio_risk_pct = (total_risk / total_investment) * 100
    print(f"\n   投资组合总风险: ${total_risk:.0f} ({portfolio_risk_pct:.1f}%)")
    
    # 5. ChatGPT分析提示词示例
    print("\n🧠 5. ChatGPT分析提示词示例")
    
    sample_prompt = """
请作为专业的股票分析师，对以下股票进行分析：

股票: AAPL (苹果公司)
当前价格: $175.50
近期表现: -2.3% (本周)
市值: $2.8T
P/E比率: 28.5

请从以下角度分析：
1. 基本面分析 (财务状况、业务前景)
2. 技术面分析 (价格趋势、支撑阻力)
3. 市场情绪 (投资者关注度、新闻影响)
4. 投资建议 (买入/持有/卖出，目标价位)
5. 风险评估 (潜在风险因素)

请给出具体的投资建议和风险控制策略。
"""
    
    print("   ChatGPT分析提示词示例:")
    print("   " + "="*40)
    print(sample_prompt.strip())
    print("   " + "="*40)
    
    # 6. 系统功能总结
    print("\n✨ 6. ChatGPT辅助交易系统功能总结")
    print("""
   核心功能:
   ✅ 实时股票数据获取 (yfinance)
   ✅ 投资组合管理和追踪
   ✅ 交易记录和盈亏计算
   ✅ 自动止损触发检测
   ✅ 投资组合表现对比 (vs S&P 500)
   ✅ 风险管理和仓位控制
   ✅ ChatGPT深度研究整合
   
   工作流程:
   1. ChatGPT进行股票深度研究
   2. 基于AI分析制定投资决策
   3. 系统执行交易并记录
   4. 实时监控止损触发
   5. 定期评估投资组合表现
   6. 生成详细的交易报告
   
   适用场景:
   - 个人投资者的AI辅助决策
   - 小资金量的精选股票投资
   - 基于深度研究的价值投资
   - 风险控制和资金管理
   """)
    
    print("\n🎯 演示完成！")
    print("ChatGPT辅助交易系统为投资者提供了AI驱动的投资决策支持。")

if __name__ == "__main__":
    demo_chatgpt_trading_features()