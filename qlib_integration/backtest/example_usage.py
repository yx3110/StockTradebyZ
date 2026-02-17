#!/usr/bin/env python3
"""
StockTradebyZ x Qlib 回测集成使用示例

展示如何使用集成的回测系统进行各种分析：
1. 单策略回测
2. 多策略对比
3. 参数优化
4. 性能分析
"""

import os
import sys
from datetime import datetime, timedelta

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from backtest_runner import BacktestRunner, run_simple_backtest


def example_single_strategy_backtest():
    """示例1: 单策略回测"""
    print("=" * 60)
    print("示例1: 单策略回测 - BBI+KDJ策略")
    print("=" * 60)
    
    try:
        # 运行简单回测
        result = run_simple_backtest(
            strategies=['bbikdj'],
            start_date='2024-01-01',
            end_date='2024-06-30',
            initial_cash=1000000
        )
        
        print(f"✅ 回测完成！")
        print(f"📊 年化收益率: {result.get('annual_return', 0):.2%}")
        print(f"📊 夏普比率: {result.get('sharpe_ratio', 0):.2f}")
        print(f"📊 最大回撤: {result.get('max_drawdown', 0):.2%}")
        print(f"📊 胜率: {result.get('win_rate', 0):.2%}")
        
    except Exception as e:
        print(f"❌ 单策略回测失败: {e}")


def example_multi_strategy_comparison():
    """示例2: 多策略对比回测"""
    print("\n" + "=" * 60)
    print("示例2: 多策略对比回测")
    print("=" * 60)
    
    try:
        runner = BacktestRunner()
        
        # 定义多个策略配置
        strategy_configs = [
            {
                'strategies': ['bbikdj'],
                'max_positions': 10,
                'position_size': 0.1,
                'min_score': 70.0,
                'rebalance_freq': 5
            },
            {
                'strategies': ['breakout'],
                'max_positions': 8,
                'position_size': 0.12,
                'min_score': 75.0,
                'rebalance_freq': 3
            },
            {
                'strategies': ['bbikdj', 'breakout'],
                'max_positions': 12,
                'position_size': 0.08,
                'min_score': 65.0,
                'rebalance_freq': 7
            }
        ]
        
        backtest_config = {
            'start_time': '2024-01-01',
            'end_time': '2024-06-30',
            'account': 1000000,
            'freq': 'day'
        }
        
        # 运行多策略对比
        comparison = runner.run_multi_strategy_comparison(
            strategy_configs=strategy_configs,
            backtest_config=backtest_config
        )
        
        print("✅ 多策略对比完成！")
        print("\n📊 策略对比结果:")
        
        for row in comparison['comparison_table']:
            print(f"🔸 {row['strategy']}")
            print(f"   年化收益: {row['annual_return']:.2%}")
            print(f"   夏普比率: {row['sharpe_ratio']:.2f}")
            print(f"   最大回撤: {row['max_drawdown']:.2%}")
            print("")
        
    except Exception as e:
        print(f"❌ 多策略对比失败: {e}")


def example_parameter_optimization():
    """示例3: 参数优化"""
    print("\n" + "=" * 60)
    print("示例3: 参数优化 - 不同仓位配置")
    print("=" * 60)
    
    try:
        runner = BacktestRunner()
        
        # 测试不同的仓位配置
        position_sizes = [0.08, 0.10, 0.12, 0.15]
        results = []
        
        for pos_size in position_sizes:
            print(f"🔄 测试仓位大小: {pos_size:.1%}")
            
            strategy_config = {
                'strategies': ['bbikdj', 'breakout'],
                'max_positions': 10,
                'position_size': pos_size,
                'min_score': 70.0,
                'stop_loss': 0.08,
                'take_profit': 0.15
            }
            
            backtest_config = {
                'start_time': '2024-01-01',
                'end_time': '2024-03-31',
                'account': 1000000,
                'freq': 'day'
            }
            
            result = runner.run_backtest(
                strategy_config=strategy_config,
                backtest_config=backtest_config,
                save_results=False
            )
            
            results.append({
                'position_size': pos_size,
                'annual_return': result.get('annual_return', 0),
                'sharpe_ratio': result.get('sharpe_ratio', 0),
                'max_drawdown': result.get('max_drawdown', 0)
            })
        
        print("\n✅ 参数优化完成！")
        print("\n📊 不同仓位配置结果:")
        print("仓位大小 | 年化收益 | 夏普比率 | 最大回撤")
        print("-" * 45)
        
        for res in results:
            print(f"{res['position_size']:.1%}      | "
                  f"{res['annual_return']:.2%}    | "
                  f"{res['sharpe_ratio']:.2f}      | "
                  f"{res['max_drawdown']:.2%}")
        
        # 找出最佳配置
        best_config = max(results, key=lambda x: x['sharpe_ratio'])
        print(f"\n🏆 最佳配置（按夏普比率）: 仓位大小 {best_config['position_size']:.1%}")
        
    except Exception as e:
        print(f"❌ 参数优化失败: {e}")


def example_advanced_analysis():
    """示例4: 高级分析"""
    print("\n" + "=" * 60)
    print("示例4: 高级分析 - 策略组合优化")
    print("=" * 60)
    
    try:
        runner = BacktestRunner()
        
        # 测试不同策略组合
        strategy_combinations = [
            ['bbikdj'],
            ['breakout'],
            ['peak'],
            ['bbikdj', 'breakout'],
            ['bbikdj', 'peak'],
            ['breakout', 'peak'],
            ['bbikdj', 'breakout', 'peak']
        ]
        
        results = []
        
        for strategies in strategy_combinations:
            combo_name = '+'.join(strategies)
            print(f"🔄 测试策略组合: {combo_name}")
            
            try:
                strategy_config = {
                    'strategies': strategies,
                    'max_positions': 10,
                    'position_size': 0.1,
                    'min_score': 70.0
                }
                
                backtest_config = {
                    'start_time': '2024-01-01',
                    'end_time': '2024-03-31',
                    'account': 1000000,
                    'freq': 'day'
                }
                
                result = runner.run_backtest(
                    strategy_config=strategy_config,
                    backtest_config=backtest_config,
                    save_results=False
                )
                
                results.append({
                    'strategies': combo_name,
                    'count': len(strategies),
                    'annual_return': result.get('annual_return', 0),
                    'sharpe_ratio': result.get('sharpe_ratio', 0),
                    'max_drawdown': result.get('max_drawdown', 0),
                    'win_rate': result.get('win_rate', 0)
                })
                
            except Exception as e:
                print(f"   ⚠️ 策略组合 {combo_name} 测试失败: {e}")
        
        if results:
            print("\n✅ 策略组合分析完成！")
            print("\n📊 策略组合对比:")
            
            # 按夏普比率排序
            results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
            
            print("排名 | 策略组合 | 年化收益 | 夏普比率 | 最大回撤 | 胜率")
            print("-" * 65)
            
            for i, res in enumerate(results[:5], 1):  # 显示前5名
                print(f"{i:2d}   | {res['strategies']:15s} | "
                      f"{res['annual_return']:8.2%} | "
                      f"{res['sharpe_ratio']:8.2f} | "
                      f"{res['max_drawdown']:8.2%} | "
                      f"{res['win_rate']:6.2%}")
        
    except Exception as e:
        print(f"❌ 高级分析失败: {e}")


def example_risk_analysis():
    """示例5: 风险分析"""
    print("\n" + "=" * 60)
    print("示例5: 风险分析 - 不同市场环境下的表现")
    print("=" * 60)
    
    try:
        runner = BacktestRunner()
        
        # 测试不同时间段（代表不同市场环境）
        time_periods = [
            {'name': '2024年1月', 'start': '2024-01-01', 'end': '2024-01-31'},
            {'name': '2024年2月', 'start': '2024-02-01', 'end': '2024-02-29'},
            {'name': '2024年3月', 'start': '2024-03-01', 'end': '2024-03-31'},
            {'name': '2024年4月', 'start': '2024-04-01', 'end': '2024-04-30'},
            {'name': '2024年5月', 'start': '2024-05-01', 'end': '2024-05-31'},
        ]
        
        strategy_config = {
            'strategies': ['bbikdj', 'breakout'],
            'max_positions': 10,
            'position_size': 0.1,
            'min_score': 70.0
        }
        
        period_results = []
        
        for period in time_periods:
            print(f"🔄 分析{period['name']}市场表现")
            
            try:
                backtest_config = {
                    'start_time': period['start'],
                    'end_time': period['end'],
                    'account': 1000000,
                    'freq': 'day'
                }
                
                result = runner.run_backtest(
                    strategy_config=strategy_config,
                    backtest_config=backtest_config,
                    save_results=False
                )
                
                period_results.append({
                    'period': period['name'],
                    'total_return': result.get('total_return', 0),
                    'sharpe_ratio': result.get('sharpe_ratio', 0),
                    'max_drawdown': result.get('max_drawdown', 0),
                    'win_rate': result.get('win_rate', 0)
                })
                
            except Exception as e:
                print(f"   ⚠️ {period['name']}分析失败: {e}")
        
        if period_results:
            print("\n✅ 风险分析完成！")
            print("\n📊 不同时期表现:")
            print("时期     | 总收益率 | 夏普比率 | 最大回撤 | 胜率")
            print("-" * 50)
            
            for res in period_results:
                print(f"{res['period']:8s} | "
                      f"{res['total_return']:8.2%} | "
                      f"{res['sharpe_ratio']:8.2f} | "
                      f"{res['max_drawdown']:8.2%} | "
                      f"{res['win_rate']:6.2%}")
            
            # 计算统计指标
            returns = [r['total_return'] for r in period_results]
            if returns:
                avg_return = sum(returns) / len(returns)
                return_std = (sum([(r - avg_return) ** 2 for r in returns]) / len(returns)) ** 0.5
                
                print(f"\n📈 跨期统计:")
                print(f"平均收益率: {avg_return:.2%}")
                print(f"收益率标准差: {return_std:.2%}")
                print(f"收益率稳定性: {(1 - return_std/abs(avg_return) if avg_return != 0 else 0):.2%}")
        
    except Exception as e:
        print(f"❌ 风险分析失败: {e}")


def run_all_examples():
    """运行所有示例"""
    print("🚀 StockTradebyZ x Qlib 回测集成系统演示")
    print("=" * 60)
    print("本演示将展示回测系统的各种功能")
    print("⚠️  注意：演示使用短期数据，实际使用建议更长时间段")
    print("=" * 60)
    
    # 运行所有示例
    example_single_strategy_backtest()
    example_multi_strategy_comparison()
    example_parameter_optimization() 
    example_advanced_analysis()
    example_risk_analysis()
    
    print("\n" + "=" * 60)
    print("🎉 演示完成！")
    print("=" * 60)
    print("📁 详细报告已保存到 reports/qlib_backtest/ 目录")
    print("📊 可以查看生成的Markdown和JSON报告")
    print("🔧 配置文件位置: qlib_integration/backtest/config/")
    print("📖 更多用法请查看各模块的文档字符串")


if __name__ == "__main__":
    # 检查依赖
    try:
        import qlib
        print(f"✅ Qlib版本: {qlib.__version__}")
    except ImportError:
        print("❌ 请先安装qlib: pip install qlib")
        sys.exit(1)
    
    try:
        import yaml
        print("✅ PyYAML已安装")
    except ImportError:
        print("⚠️  建议安装PyYAML: pip install pyyaml")
    
    # 运行演示
    run_all_examples()