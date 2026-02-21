#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI自动交易系统回测脚本
快速测试和验证系统有效性
"""

import sys
import os
from datetime import datetime, timedelta

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from strategy.ai_auto_trading_system import create_sample_backtest, AIBacktestEngine, AITradingStrategy

def quick_backtest():
    """快速回测测试"""
    print("🚀 启动AI自动交易系统快速回测")
    print("="*60)
    
    try:
        # 创建回测引擎
        engine = AIBacktestEngine()
        
        # 使用最近AI报告推荐的股票进行测试
        test_stocks = [
            '300679',  # 电连技术 - AI推荐买入
            '002594',  # 比亚迪 - 持续关注
            '000858',  # 五粮液 - 防御性配置
            '600519',  # 贵州茅台 - 价值股
            '000001'   # 平安银行 - 金融蓝筹
        ]
        
        # 设置回测期间 (最近1个月)
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        
        print(f"📅 回测期间: {start_date} - {end_date}")
        print(f"📈 测试股票: {', '.join(test_stocks)}")
        print()
        
        # 添加股票数据
        for stock in test_stocks:
            engine.add_stock_data(stock, start_date, end_date)
        
        # 配置策略参数
        strategy_params = {
            'max_positions': 3,        # 最大持仓3只
            'position_size': 0.15,     # 每只股票15%仓位
            'stop_loss_pct': 0.08,     # 8%止损
            'take_profit_pct': 0.20,   # 20%止盈
            'rebalance_days': 3        # 3天调仓一次
        }
        
        print("⚙️  策略参数:")
        for key, value in strategy_params.items():
            print(f"   {key}: {value}")
        print()
        
        # 添加策略
        engine.add_strategy(AITradingStrategy, **strategy_params)
        
        # 运行回测
        results = engine.run_backtest()
        
        # 生成并保存报告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        report_path = f"reports/backtest/AI交易快速回测_{timestamp}.md"
        
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        report = engine.generate_report(report_path)
        
        print(f"📊 详细报告已保存: {report_path}")
        
        # 显示关键指标
        print("\n📋 回测摘要:")
        print(f"   初始资金: 1,000,000 元")
        print(f"   最终资金: {engine.cerebro.broker.getvalue():,.0f} 元")
        total_return = (engine.cerebro.broker.getvalue() / 1000000 - 1) * 100
        print(f"   总收益率: {total_return:.2f}%")
        
        if hasattr(results, 'analyzers'):
            trades = results.analyzers.trades.get_analysis()
            if 'total' in trades and trades.total.total > 0:
                win_rate = (trades.won.total / trades.total.total) * 100 if 'won' in trades else 0
                print(f"   交易次数: {trades.total.total}")
                print(f"   胜率: {win_rate:.1f}%")
            
            drawdown = results.analyzers.drawdown.get_analysis()
            if hasattr(drawdown, 'max') and drawdown.max.drawdown:
                print(f"   最大回撤: {drawdown.max.drawdown:.2f}%")
        
        return engine, results
        
    except Exception as e:
        print(f"❌ 回测失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def comprehensive_backtest():
    """全面回测测试"""
    print("🔬 启动AI自动交易系统全面回测")
    print("="*60)
    
    try:
        # 使用更长的回测期间和更多股票
        engine = AIBacktestEngine()
        
        # 从最新AI报告中获取更多推荐股票
        comprehensive_stocks = [
            '300679', '002594', '000858', '600519', '000001',  # 核心推荐
            '600036', '000002', '601318', '000568', '002415'   # 补充标的
        ]
        
        # 3个月回测期间
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')
        
        print(f"📅 回测期间: {start_date} - {end_date}")
        print(f"📈 测试股票: {len(comprehensive_stocks)}只")
        print()
        
        # 添加数据
        for stock in comprehensive_stocks:
            engine.add_stock_data(stock, start_date, end_date)
        
        # 更保守的策略参数
        strategy_params = {
            'max_positions': 5,
            'position_size': 0.1,      # 降低单股仓位
            'stop_loss_pct': 0.06,     # 更严格止损
            'take_profit_pct': 0.15,   # 更保守止盈
            'rebalance_days': 5        # 降低调仓频率
        }
        
        engine.add_strategy(AITradingStrategy, **strategy_params)
        
        # 运行回测
        results = engine.run_backtest()
        
        # 保存报告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        report_path = f"reports/backtest/AI交易全面回测_{timestamp}.md"
        report = engine.generate_report(report_path)
        
        print(f"📊 详细报告已保存: {report_path}")
        
        return engine, results
        
    except Exception as e:
        print(f"❌ 全面回测失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    print("🤖 AI自动交易系统回测工具")
    print("请选择回测模式:")
    print("1. 快速回测 (1个月, 5只股票)")
    print("2. 全面回测 (3个月, 10只股票)")
    print("3. 示例回测 (使用默认配置)")
    
    choice = input("\n请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        engine, results = quick_backtest()
    elif choice == "2":
        engine, results = comprehensive_backtest()
    elif choice == "3":
        engine, results, report = create_sample_backtest()
    else:
        print("❌ 无效选择")
        exit(1)
    
    if engine and results:
        print("\n✅ 回测完成! 系统运行正常")
        print("\n💡 提示:")
        print("- 查看 reports/backtest/ 目录获取详细报告")
        print("- 可以调整 strategy_params 参数优化策略")
        print("- 建议在更长时间段进行回测验证")
    else:
        print("\n❌ 回测失败，请检查数据和配置")