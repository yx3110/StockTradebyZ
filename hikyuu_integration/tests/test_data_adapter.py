#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据适配器单元测试

测试HikyuuStyleDataAdapter的核心功能
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import unittest
from hikyuu_integration import HikyuuStyleDataAdapter, Query, Stock, KData


class TestHikyuuStyleDataAdapter(unittest.TestCase):
    """测试数据适配器"""

    @classmethod
    def setUpClass(cls):
        """初始化测试环境"""
        cls.adapter = HikyuuStyleDataAdapter()

    def test_01_adapter_initialization(self):
        """测试适配器初始化"""
        self.assertIsNotNone(self.adapter)
        self.assertIsNotNone(self.adapter.db)
        print("✅ 适配器初始化成功")

    def test_02_get_stock(self):
        """测试获取股票对象"""
        stock = self.adapter.get_stock('000001')

        self.assertIsInstance(stock, Stock)
        self.assertEqual(stock.code, '000001')
        self.assertIsNotNone(stock.name)
        print(f"✅ 获取股票对象成功: {stock}")

    def test_03_query_object(self):
        """测试Query对象"""
        # 最近150天
        query1 = Query(-150)
        self.assertEqual(query1.get_days_count(), 150)
        self.assertTrue(query1.is_recent_days())

        # 日期区间
        query2 = Query(start='2024-01-01', end='2025-09-30')
        self.assertEqual(query2.start_date, '2024-01-01')
        self.assertEqual(query2.end_date, '2025-09-30')
        self.assertFalse(query2.is_recent_days())

        print("✅ Query对象测试通过")

    def test_04_get_kdata_recent_days(self):
        """测试获取最近N天的K线数据"""
        kdata = self.adapter.get_kdata('000001', Query(-50))

        self.assertIsInstance(kdata, KData)
        self.assertGreater(len(kdata), 0)
        self.assertEqual(kdata.stock_code, '000001')

        # 检查数据属性
        self.assertIsNotNone(kdata.open)
        self.assertIsNotNone(kdata.close)
        self.assertIsNotNone(kdata.datetime)

        print(f"✅ 获取最近50天K线成功: {kdata}")
        print(f"   数据长度: {len(kdata)}")
        print(f"   日期范围: {kdata.datetime[0]} 至 {kdata.datetime[-1]}")

    def test_05_get_kdata_date_range(self):
        """测试获取指定日期范围的K线数据"""
        kdata = self.adapter.get_kdata(
            '000001',
            Query(start='2025-01-01', end='2025-09-30')
        )

        self.assertIsInstance(kdata, KData)
        self.assertGreater(len(kdata), 0)

        print(f"✅ 获取日期范围K线成功: {kdata}")

    def test_06_kdata_access(self):
        """测试KData数据访问"""
        kdata = self.adapter.get_kdata('000001', Query(-10))

        # 测试索引访问
        first_bar = kdata[0]
        self.assertIn('datetime', first_bar)
        self.assertIn('open', first_bar)
        self.assertIn('close', first_bar)

        last_bar = kdata[-1]
        self.assertIsNotNone(last_bar['close'])

        # 测试数组访问
        self.assertEqual(len(kdata.close), len(kdata))
        self.assertEqual(len(kdata.open), len(kdata))

        print(f"✅ KData数据访问测试通过")
        print(f"   第一根K线: {first_bar['datetime']} 收盘价={first_bar['close']}")
        print(f"   最后一根K线: {last_bar['datetime']} 收盘价={last_bar['close']}")

    def test_07_get_indicator(self):
        """测试获取技术指标"""
        kdata = self.adapter.get_kdata('000001', Query(-50))

        # 获取MA20
        ma20 = kdata.get_indicator('MA', n=20)
        if ma20 is not None:
            self.assertEqual(len(ma20), len(kdata))
            print(f"✅ 获取MA20成功，最新值: {ma20[-1]:.2f}")
        else:
            print("⚠️ MA20数据不存在（可能technical_indicators表无数据）")

        # 获取BBI
        bbi = kdata.get_indicator('BBI')
        if bbi is not None:
            print(f"✅ 获取BBI成功，最新值: {bbi[-1]:.2f}")

        # 获取KDJ
        kdj_k = kdata.get_indicator('KDJ_K')
        if kdj_k is not None:
            print(f"✅ 获取KDJ_K成功，最新值: {kdj_k[-1]:.2f}")

    def test_08_stock_methods(self):
        """测试Stock对象方法"""
        stock = self.adapter.get_stock('000001')

        # 测试get_kdata
        kdata = stock.get_kdata(Query(-30))
        self.assertIsInstance(kdata, KData)
        self.assertEqual(kdata.stock_code, '000001')

        # 测试is_valid
        self.assertTrue(stock.is_valid())

        print(f"✅ Stock对象方法测试通过")

    def test_09_get_all_stocks(self):
        """测试获取所有股票"""
        stocks = self.adapter.get_all_stocks('A股')

        self.assertIsInstance(stocks, list)
        self.assertGreater(len(stocks), 0)

        print(f"✅ 获取所有A股成功，共{len(stocks)}只")

    def test_10_get_trading_dates(self):
        """测试获取交易日期"""
        dates = self.adapter.get_trading_dates('2025-01-01', '2025-09-30')

        self.assertIsInstance(dates, list)
        self.assertGreater(len(dates), 0)

        print(f"✅ 获取交易日期成功，共{len(dates)}个交易日")
        print(f"   范围: {dates[0]} 至 {dates[-1]}")

    def test_11_stock_info(self):
        """测试股票信息"""
        stock = self.adapter.get_stock('000001')

        print(f"✅ 股票信息:")
        print(f"   代码: {stock.code}")
        print(f"   名称: {stock.name}")
        print(f"   类型: {stock.type}")
        print(f"   交易所: {stock.exchange}")
        print(f"   行业: {stock.industry}")


class TestPerformance(unittest.TestCase):
    """性能测试"""

    @classmethod
    def setUpClass(cls):
        """初始化测试环境"""
        cls.adapter = HikyuuStyleDataAdapter()

    def test_preload_performance(self):
        """测试预加载性能"""
        import time

        stock_list = ['000001', '000002', '600000', '600036', '601318']

        print("\n📊 性能测试：预加载 vs 逐个查询")

        # 方法1: 逐个查询
        start = time.time()
        for stock in stock_list:
            kdata = self.adapter.get_kdata(stock, Query(start='2025-01-01', end='2025-09-30'))
        time_individual = time.time() - start

        print(f"   逐个查询: {time_individual:.3f}秒")

        # 方法2: 预加载
        adapter2 = HikyuuStyleDataAdapter()
        start = time.time()
        adapter2.preload_data(stock_list, '2025-01-01', '2025-09-30')
        # 这里实际应该使用缓存读取，但我们的实现还没有完全优化
        time_preload = time.time() - start

        print(f"   预加载: {time_preload:.3f}秒")
        print(f"   ✅ 预加载测试完成")


def run_tests():
    """运行所有测试"""
    print("=" * 80)
    print("🧪 Hikyuu数据适配器测试")
    print("=" * 80)

    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加基础测试
    suite.addTests(loader.loadTestsFromTestCase(TestHikyuuStyleDataAdapter))

    # 添加性能测试
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 输出总结
    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print("✅ 所有测试通过！")
    else:
        print(f"❌ 测试失败: {len(result.failures)} 个失败, {len(result.errors)} 个错误")

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
