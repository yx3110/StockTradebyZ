#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Signal单元测试

测试Signal, MoneyManager, StopLoss等组件
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import unittest
from hikyuu_integration import (
    HikyuuStyleDataAdapter, Query,
    BBISignal, KDJSignal, CompositeSignal,
    MLScoringSignal, MLCombinedSignal,
    MM_FixedCount, MM_FixedPercent, MM_FixedRisk,
    ST_FixedPercent, ST_ProfitGoal, ST_Trailing
)


class TestBasicSignals(unittest.TestCase):
    """测试基础Signal"""

    @classmethod
    def setUpClass(cls):
        """初始化测试环境"""
        cls.adapter = HikyuuStyleDataAdapter()
        cls.kdata = cls.adapter.get_kdata('000001', Query(-50))

    def test_01_bbi_signal(self):
        """测试BBI信号"""
        signal = BBISignal()
        signal.calculate(self.kdata)

        buy_dates = signal.get_all_buy_dates()
        sell_dates = signal.get_all_sell_dates()

        print(f"✅ BBI信号: 买入{len(buy_dates)}次, 卖出{len(sell_dates)}次")

        if buy_dates:
            print(f"   最近买入: {buy_dates[-1]}")
        if sell_dates:
            print(f"   最近卖出: {sell_dates[-1]}")

        self.assertIsInstance(buy_dates, list)
        self.assertIsInstance(sell_dates, list)

    def test_02_kdj_signal(self):
        """测试KDJ信号"""
        signal = KDJSignal(oversold=20, overbought=80)
        signal.calculate(self.kdata)

        buy_dates = signal.get_all_buy_dates()
        sell_dates = signal.get_all_sell_dates()

        print(f"✅ KDJ信号: 买入{len(buy_dates)}次, 卖出{len(sell_dates)}次")

        self.assertIsInstance(buy_dates, list)
        self.assertIsInstance(sell_dates, list)

    def test_03_composite_signal(self):
        """测试组合信号"""
        bbi_signal = BBISignal()
        kdj_signal = KDJSignal()

        # AND组合
        composite_and = CompositeSignal([bbi_signal, kdj_signal], mode='AND')
        composite_and.calculate(self.kdata)

        buy_dates_and = composite_and.get_all_buy_dates()
        print(f"✅ 组合信号(AND): 买入{len(buy_dates_and)}次")

        # OR组合
        composite_or = CompositeSignal([bbi_signal, kdj_signal], mode='OR')
        composite_or.calculate(self.kdata)

        buy_dates_or = composite_or.get_all_buy_dates()
        print(f"✅ 组合信号(OR): 买入{len(buy_dates_or)}次")

        # AND信号应该少于或等于OR信号
        self.assertLessEqual(len(buy_dates_and), len(buy_dates_or))


class TestMLScoringSignal(unittest.TestCase):
    """测试ML评分Signal"""

    @classmethod
    def setUpClass(cls):
        """初始化测试环境"""
        cls.adapter = HikyuuStyleDataAdapter()
        cls.kdata = cls.adapter.get_kdata('000001', Query(-30))

    def test_01_ml_signal_v37(self):
        """测试v3.7 ML信号"""
        signal = MLScoringSignal(ml_version='v3.7', min_score=80)
        signal.calculate(self.kdata)

        buy_dates = signal.get_all_buy_dates()
        scores = signal.get_all_scores()

        print(f"✅ v3.7 ML信号测试:")
        print(f"   买入信号: {len(buy_dates)}次")
        print(f"   评分缓存: {len(scores)}条")

        if scores:
            avg_score = sum(scores.values()) / len(scores)
            print(f"   平均评分: {avg_score:.1f}")

        self.assertIsInstance(buy_dates, list)
        self.assertEqual(len(scores), len(self.kdata))

    def test_02_ml_signal_v38(self):
        """测试v3.8 ML信号"""
        signal = MLScoringSignal(ml_version='v3.8', min_score=80)
        signal.calculate(self.kdata)

        buy_dates = signal.get_all_buy_dates()
        print(f"✅ v3.8 ML信号: 买入{len(buy_dates)}次")

        self.assertIsInstance(buy_dates, list)

    def test_03_ml_signal_v381(self):
        """测试v3.81 ML信号"""
        signal = MLScoringSignal(ml_version='v3.81', min_score=80)
        signal.calculate(self.kdata)

        buy_dates = signal.get_all_buy_dates()
        sell_dates = signal.get_all_sell_dates()

        print(f"✅ v3.81 ML信号: 买入{len(buy_dates)}次, 卖出{len(sell_dates)}次")

        self.assertIsInstance(buy_dates, list)
        self.assertIsInstance(sell_dates, list)

    def test_04_ml_combined_signal(self):
        """测试ML组合信号"""
        signal = MLCombinedSignal(
            ml_version='v3.81',
            min_score=80,
            use_bbi=True,
            use_kdj=True
        )
        signal.calculate(self.kdata)

        buy_dates = signal.get_all_buy_dates()
        print(f"✅ ML组合信号(ML+BBI+KDJ): 买入{len(buy_dates)}次")

        # 组合信号应该比单纯ML信号更严格
        ml_only = MLScoringSignal(ml_version='v3.81', min_score=80)
        ml_only.calculate(self.kdata)

        self.assertLessEqual(len(buy_dates), len(ml_only.get_all_buy_dates()))


class TestMoneyManager(unittest.TestCase):
    """测试资金管理"""

    def test_01_fixed_count(self):
        """测试固定股数"""
        mm = MM_FixedCount(1000)

        buy_num = mm.get_buy_num('2025-09-30', '000001', 10.0, 50000)
        self.assertEqual(buy_num, 1000)
        print(f"✅ 固定股数(1000): 买入{buy_num}股")

        # 资金不足
        buy_num2 = mm.get_buy_num('2025-09-30', '000001', 100.0, 50000)
        self.assertEqual(buy_num2, 0)
        print(f"✅ 固定股数(资金不足): 买入{buy_num2}股")

    def test_02_fixed_percent(self):
        """测试固定比例"""
        mm = MM_FixedPercent(0.1)  # 10%

        buy_num = mm.get_buy_num('2025-09-30', '000001', 10.0, 100000)
        # 10% = 10000, 10000/10 = 1000股
        self.assertEqual(buy_num, 1000)
        print(f"✅ 固定比例(10%): 买入{buy_num}股")

    def test_03_fixed_risk(self):
        """测试固定风险"""
        mm = MM_FixedRisk(risk_pct=0.02, stop_loss_pct=0.08)

        buy_num = mm.get_buy_num('2025-09-30', '000001', 10.0, 100000)
        print(f"✅ 固定风险(2%): 买入{buy_num}股")

        self.assertGreater(buy_num, 0)
        self.assertEqual(buy_num % 100, 0)  # 必须是100的整数倍


class TestStopLoss(unittest.TestCase):
    """测试止损策略"""

    def test_01_fixed_percent_stop(self):
        """测试固定百分比止损"""
        st = ST_FixedPercent(0.08)  # 8%止损

        # 未触发止损
        result1 = st.should_stop('2025-09-30', 10.0, 9.5, '2025-09-20')
        self.assertFalse(result1)
        print(f"✅ 固定止损: 跌5%未触发")

        # 触发止损
        result2 = st.should_stop('2025-09-30', 10.0, 9.0, '2025-09-20')
        self.assertTrue(result2)
        print(f"✅ 固定止损: 跌10%已触发")

    def test_02_profit_goal(self):
        """测试止盈"""
        st = ST_ProfitGoal(0.20)  # 20%止盈

        # 未触发止盈
        result1 = st.should_stop('2025-09-30', 10.0, 11.5, '2025-09-20')
        self.assertFalse(result1)
        print(f"✅ 止盈: 涨15%未触发")

        # 触发止盈
        result2 = st.should_stop('2025-09-30', 10.0, 12.5, '2025-09-20')
        self.assertTrue(result2)
        print(f"✅ 止盈: 涨25%已触发")


def run_tests():
    """运行所有测试"""
    print("=" * 80)
    print("🧪 Phase 2: Signal组件测试")
    print("=" * 80)

    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加测试
    suite.addTests(loader.loadTestsFromTestCase(TestBasicSignals))
    suite.addTests(loader.loadTestsFromTestCase(TestMLScoringSignal))
    suite.addTests(loader.loadTestsFromTestCase(TestMoneyManager))
    suite.addTests(loader.loadTestsFromTestCase(TestStopLoss))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 输出总结
    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print("✅ 所有测试通过！Phase 2 完成")
    else:
        print(f"❌ 测试失败: {len(result.failures)} 个失败, {len(result.errors)} 个错误")

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
