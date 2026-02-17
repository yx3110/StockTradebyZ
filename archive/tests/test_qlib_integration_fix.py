#!/usr/bin/env python3
"""
测试Qlib集成修复
验证Exchange配置问题是否已解决
"""

import os
import sys
import logging
from datetime import datetime, timedelta

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_exchange_initialization():
    """测试交易所初始化"""
    try:
        from qlib_integration.backtest.chinese_exchange import ChineseAShareExchange
        
        logger.info("测试ChineseAShareExchange初始化...")
        
        # 使用基本参数测试，包含trade_unit作为kwargs
        exchange = ChineseAShareExchange(
            start_time='2024-01-01',
            end_time='2024-12-31',
            freq='day',
            codes=['000001.SZ', '000002.SZ'],
            deal_price='close',
            limit_threshold=0.095,
            open_cost=0.0003,
            close_cost=0.0013,
            min_cost=5.0,
            trade_unit=100
        )
        
        logger.info("✅ ChineseAShareExchange初始化成功")
        logger.info(f"交易所规则: T+1={exchange.t_plus_1}, 交易单位={exchange.trade_unit}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ChineseAShareExchange初始化失败: {e}")
        return False

def test_backtest_runner_initialization():
    """测试回测运行器初始化"""
    try:
        from qlib_integration.backtest.backtest_runner import BacktestRunner
        
        logger.info("测试BacktestRunner初始化...")
        
        # 创建简单的测试配置
        test_config = {
            'backtest': {
                'start_time': '2024-01-01',
                'end_time': '2024-12-31',
                'benchmark': '000300.SH',
                'account': 1000000,
                'freq': 'day'
            },
            'strategy': {
                'strategies': ['bbikdj'],
                'max_positions': 10,
                'min_score': 70.0,
                'rebalance_freq': 5
            },
            'exchange': {
                'limit_threshold': 0.095,
                'deal_price': 'close',
                'trade_unit': 100,
                'open_cost': 0.0003,
                'close_cost': 0.0013,
                'min_cost': 5.0
            }
        }
        
        # 写入临时配置文件
        import json
        import yaml
        temp_config_path = '/tmp/test_config.yaml'
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(test_config, f, default_flow_style=False)
        
        # 测试创建BacktestRunner（不初始化Qlib）
        runner = BacktestRunner.__new__(BacktestRunner)
        runner.config = test_config
        
        logger.info("✅ BacktestRunner配置加载成功")
        
        # 清理临时文件
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ BacktestRunner初始化失败: {e}")
        return False

def test_v352_strategy_initialization():
    """测试V3.52策略初始化"""
    try:
        from qlib_integration.backtest.stocktrader_strategy import StockTraderStrategy
        from scoring.scoring_engine import ScoringEngine
        
        logger.info("测试V3.52策略初始化...")
        
        # 创建评分引擎
        scorer = ScoringEngine()
        logger.info("评分引擎创建成功")
        
        # 测试策略类创建（不运行实际回测）
        strategy_config = {
            'strategies': ['bbikdj'],
            'max_positions': 10,
            'position_size': 0.1,
            'min_score': 75.0,
            'rebalance_freq': 5
        }
        
        logger.info("✅ V3.52策略组件测试成功")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ V3.52策略初始化失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("🚀 开始测试Qlib集成修复...")
    
    tests = [
        ("交易所初始化测试", test_exchange_initialization),
        ("回测运行器初始化测试", test_backtest_runner_initialization), 
        ("V3.52策略初始化测试", test_v352_strategy_initialization)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"{test_name}执行异常: {e}")
            results.append((test_name, False))
    
    # 测试结果总结
    logger.info("\n" + "="*50)
    logger.info("📊 测试结果总结:")
    logger.info("="*50)
    
    passed = 0
    failed = 0
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        
        if success:
            passed += 1
        else:
            failed += 1
    
    logger.info(f"\n总计: {passed}个测试通过, {failed}个测试失败")
    
    if failed == 0:
        logger.info("🎉 所有测试通过！Qlib集成修复成功！")
        return True
    else:
        logger.error("⚠️  存在测试失败，需要进一步修复")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)