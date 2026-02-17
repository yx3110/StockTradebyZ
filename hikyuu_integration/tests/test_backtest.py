#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtest单元测试

测试Portfolio, Broker, BacktestEngine等组件
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import unittest
from hikyuu_integration import (
    HikyuuStyleDataAdapter, Query,
    Portfolio, Position, Trade, Broker,
    HikyuuStyleBacktestEngine, BacktestResult,
    BBISignal, KDJSignal, MLScoringSignal,
    MM_FixedCount, MM_FixedPercent, MM_FixedRisk,
    ST_FixedPercent, ST_ProfitGoal
)


class TestPortfolio(unittest.TestCase):
    """测试Portfolio组件"""

    def setUp(self):
        """初始化测试环境"""
        self.portfolio = Portfolio(initial_cash=100000)

    def test_01_buy_stock(self):
        """测试买入股票"""
        success = self.portfolio.buy('000001', 1000, 10.0, '2025-09-30', reason='测试买入')

        self.assertTrue(success)
        self.assertEqual(self.portfolio.get_position_count(), 1)
        self.assertTrue(self.portfolio.has_position('000001'))

        pos = self.portfolio.get_position('000001')
        self.assertEqual(pos.shares, 1000)
        self.assertEqual(pos.entry_price, 10.0)

        print(f"✅ 买入成功: {pos}")

    def test_02_sell_stock(self):
        """测试卖出股票"""
        # 先买入
        self.portfolio.buy('000001', 1000, 10.0, '2025-09-30')

        # 更新价格
        self.portfolio.update_prices({'000001': 11.0})

        # 卖出
        success = self.portfolio.sell('000001', 1000, 11.0, '2025-10-01', reason='测试卖出')

        self.assertTrue(success)
        self.assertEqual(self.portfolio.get_position_count(), 0)
        self.assertFalse(self.portfolio.has_position('000001'))

        print(f"✅ 卖出成功, 现金={self.portfolio.cash:.2f}")

    def test_03_portfolio_stats(self):
        """测试组合统计"""
        # 买入两只股票
        self.portfolio.buy('000001', 1000, 10.0, '2025-09-30')
        self.portfolio.buy('000002', 500, 20.0, '2025-09-30')

        # 更新价格
        self.portfolio.update_prices({'000001': 11.0, '000002': 22.0})

        # 记录价值
        self.portfolio.record_value('2025-09-30')

        stats = self.portfolio.get_stats()

        self.assertEqual(stats['position_count'], 2)
        self.assertGreater(stats['total_value'], stats['initial_cash'])
        self.assertGreater(stats['total_pnl'], 0)

        print(f"✅ 组合统计:")
        print(f"   持仓数: {stats['position_count']}")
        print(f"   总资产: {stats['total_value']:,.2f}")
        print(f"   盈亏:   {stats['total_pnl']:,.2f} ({stats['total_pnl_pct']:.2f}%)")

    def test_04_insufficient_cash(self):
        """测试资金不足"""
        # 尝试买入超过资金的股票
        success = self.portfolio.buy('000001', 100000, 10.0, '2025-09-30')

        self.assertFalse(success)
        self.assertEqual(self.portfolio.get_position_count(), 0)

        print(f"✅ 资金不足检测正常")

    def test_05_partial_sell(self):
        """测试部分卖出"""
        # 买入1000股
        self.portfolio.buy('000001', 1000, 10.0, '2025-09-30')

        # 卖出500股
        success = self.portfolio.sell('000001', 500, 11.0, '2025-10-01')

        self.assertTrue(success)
        pos = self.portfolio.get_position('000001')
        self.assertEqual(pos.shares, 500)

        print(f"✅ 部分卖出: 剩余{pos.shares}股")


class TestBroker(unittest.TestCase):
    """测试Broker组件"""

    @classmethod
    def setUpClass(cls):
        """初始化测试环境"""
        cls.adapter = HikyuuStyleDataAdapter()
        cls.portfolio = Portfolio(initial_cash=100000)
        cls.broker = Broker(cls.portfolio, cls.adapter, enable_t1=True, enable_limit_check=False)

    def test_01_buy_stock(self):
        """测试Broker买入"""
        success = self.broker.buy('000001', 1000, '2025-09-25', reason='Broker买入测试')

        self.assertTrue(success)
        self.assertEqual(self.portfolio.get_position_count(), 1)

        print(f"✅ Broker买入成功")

    def test_02_t1_restriction(self):
        """测试T+1限制"""
        # 买入
        self.broker.buy('000002', 1000, '2025-09-25')

        # 尝试当天卖出（应该失败）
        success = self.broker.sell('000002', 1000, '2025-09-25')

        self.assertFalse(success)
        print(f"✅ T+1限制正常: 当天无法卖出")

        # 次日卖出（应该成功）
        success2 = self.broker.sell('000002', 1000, '2025-09-26')

        self.assertTrue(success2)
        print(f"✅ T+1限制正常: 次日可以卖出")

    def test_03_t1_locks(self):
        """测试T+1锁定"""
        # 买入（使用000651，我们知道这个股票有数据）
        self.broker.buy('000651', 1000, '2025-09-25')

        # 检查T+1锁定
        self.assertTrue(self.broker.has_t1_lock('000651'))

        locks = self.broker.get_t1_locks()
        self.assertIn('000651', locks)

        print(f"✅ T+1锁定正常: {locks}")


class TestBacktestEngine(unittest.TestCase):
    """测试回测引擎"""

    @classmethod
    def setUpClass(cls):
        """初始化测试环境"""
        cls.adapter = HikyuuStyleDataAdapter()

    def test_01_simple_backtest(self):
        """测试简单回测"""
        # 创建回测引擎
        engine = HikyuuStyleBacktestEngine(
            data_adapter=self.adapter,
            signal=BBISignal(),
            money_manager=MM_FixedPercent(0.2),  # 每次20%
            initial_cash=100000,
            max_positions=3,
            enable_t1=False,  # 关闭T+1简化测试
            enable_limit_check=False
        )

        # 运行回测
        result = engine.run(
            stock_list=['000001', '000002'],
            start_date='2025-09-01',
            end_date='2025-09-30'
        )

        # 验证结果
        self.assertIsInstance(result, BacktestResult)
        self.assertGreater(result.trade_count, 0)

        print(f"\n✅ 简单回测完成:")
        print(f"   交易次数: {result.trade_count}")
        print(f"   收益:     {result.total_return_pct:.2f}%")
        print(f"   年化:     {result.annualized_return:.2f}%")
        print(f"   最大回撤: {result.max_drawdown_pct:.2f}%")

    def test_02_backtest_with_stop_loss(self):
        """测试带止损的回测"""
        engine = HikyuuStyleBacktestEngine(
            data_adapter=self.adapter,
            signal=KDJSignal(),
            money_manager=MM_FixedPercent(0.1),
            stop_loss=ST_FixedPercent(0.08),  # 8%止损
            initial_cash=100000,
            max_positions=5,
            enable_t1=False,
            enable_limit_check=False
        )

        result = engine.run(
            stock_list=['000001', '000002', '000651'],
            start_date='2025-08-01',
            end_date='2025-09-30'
        )

        self.assertIsInstance(result, BacktestResult)

        print(f"\n✅ 止损回测完成:")
        print(f"   交易次数: {result.trade_count}")
        print(f"   收益:     {result.total_return_pct:.2f}%")
        print(f"   胜率:     {result.win_rate:.2f}%")

    def test_03_ml_backtest(self):
        """测试ML信号回测"""
        engine = HikyuuStyleBacktestEngine(
            data_adapter=self.adapter,
            signal=MLScoringSignal(ml_version='v3.81', min_score=80),
            money_manager=MM_FixedRisk(risk_pct=0.02, stop_loss_pct=0.08),
            stop_loss=ST_ProfitGoal(0.20),  # 20%止盈
            initial_cash=100000,
            max_positions=10,
            enable_t1=False,
            enable_limit_check=False
        )

        result = engine.run(
            stock_list=['000001', '000002', '000651', '002594'],
            start_date='2025-08-01',
            end_date='2025-09-30'
        )

        self.assertIsInstance(result, BacktestResult)

        print(f"\n✅ ML回测完成:")
        print(f"   交易次数: {result.trade_count}")
        print(f"   收益:     {result.total_return_pct:.2f}%")
        print(f"   夏普比率: {result.sharpe_ratio:.2f}")

    def test_04_backtest_result_export(self):
        """测试回测结果导出"""
        engine = HikyuuStyleBacktestEngine(
            data_adapter=self.adapter,
            signal=BBISignal(),
            money_manager=MM_FixedPercent(0.1),
            initial_cash=100000,
            enable_t1=False,
            enable_limit_check=False
        )

        result = engine.run(
            stock_list=['000001'],
            start_date='2025-09-01',
            end_date='2025-09-30'
        )

        # 导出交易记录
        trades_df = result.get_trades_df()

        self.assertIsNotNone(trades_df)
        print(f"\n✅ 交易记录导出:")
        if len(trades_df) > 0:
            print(trades_df.head())
        else:
            print("   无交易记录")


class TestBacktestMetrics(unittest.TestCase):
    """测试回测指标计算"""

    @classmethod
    def setUpClass(cls):
        """初始化测试环境"""
        cls.adapter = HikyuuStyleDataAdapter()

    def test_01_return_calculation(self):
        """测试收益率计算"""
        portfolio = Portfolio(initial_cash=100000)

        # 模拟交易
        portfolio.buy('000001', 1000, 10.0, '2025-09-01')
        portfolio.update_prices({'000001': 12.0})
        portfolio.record_value('2025-09-30')

        # 计算收益
        pnl = portfolio.get_total_pnl()
        pnl_pct = portfolio.get_total_pnl_pct()

        print(f"\n✅ 收益率计算:")
        print(f"   盈亏: {pnl:,.2f}")
        print(f"   收益率: {pnl_pct:.2f}%")

        self.assertGreater(pnl, 0)
        self.assertGreater(pnl_pct, 0)

    def test_02_commission_calculation(self):
        """测试手续费计算"""
        portfolio = Portfolio(
            initial_cash=100000,
            commission_rate=0.0003,
            min_commission=5.0
        )

        # 买卖股票
        portfolio.buy('000001', 1000, 10.0, '2025-09-01')
        portfolio.sell('000001', 1000, 11.0, '2025-09-02')

        stats = portfolio.get_stats()

        print(f"\n✅ 手续费计算:")
        print(f"   总手续费: {stats['total_commission']:.2f}")

        self.assertGreater(stats['total_commission'], 0)


def run_tests():
    """运行所有测试"""
    print("=" * 80)
    print("🧪 Phase 3: 回测引擎测试")
    print("=" * 80)

    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加测试
    suite.addTests(loader.loadTestsFromTestCase(TestPortfolio))
    suite.addTests(loader.loadTestsFromTestCase(TestBroker))
    suite.addTests(loader.loadTestsFromTestCase(TestBacktestEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestBacktestMetrics))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 输出总结
    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print("✅ 所有测试通过！Phase 3 完成")
    else:
        print(f"❌ 测试失败: {len(result.failures)} 个失败, {len(result.errors)} 个错误")

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
