# Hikyuu风格回测框架 - 完整使用指南

**版本**: v0.2.1
**作者**: StockTradebyZ Team
**日期**: 2025-10-10

---

## 📚 目录

1. [简介](#简介)
2. [快速开始](#快速开始)
3. [核心概念](#核心概念)
4. [API参考](#api参考)
5. [使用示例](#使用示例)
6. [性能优化](#性能优化)
7. [最佳实践](#最佳实践)
8. [常见问题](#常见问题)
9. [附录](#附录)

---

## 简介

### 什么是Hikyuu风格回测框架？

Hikyuu风格回测框架是一个**轻量级但功能完整**的Python回测系统，借鉴了Hikyuu的优秀设计思想，专为中国A股市场设计。

### 核心特点

✅ **轻量级**: 纯Python实现，无需编译C++
✅ **高性能**: SmartCacheManager + 并行回测
✅ **灵活**: 多种Signal/MM/SL策略可组合
✅ **完整**: 涵盖数据、信号、资金管理、止损、组合管理
✅ **T+1规则**: 完整支持中国股市T+1交易规则
✅ **生产就绪**: 无TODO/mock，完整测试覆盖

### 适用场景

- 技术分析驱动的交易策略
- 中短线交易策略回测
- 多股票组合回测
- Signal策略开发和验证
- 快速原型开发

---

## 快速开始

### 安装依赖

```bash
pip install pandas numpy
```

### 最简单的回测

```python
from hikyuu_integration import (
    HikyuuStyleBacktestEngine,
    BBISignal,
    MM_FixedPercent,
    ST_FixedPercent,
    HikyuuStyleDataAdapter
)
from data_adapter.database_manager import DatabaseManager

# 1. 创建数据适配器
db = DatabaseManager(db_path='data_adapter/stock_data.db')
adapter = HikyuuStyleDataAdapter(db_manager=db)

# 2. 创建回测引擎
engine = HikyuuStyleBacktestEngine(
    data_adapter=adapter,
    signal=BBISignal(),                # BBI信号
    money_manager=MM_FixedPercent(0.2),  # 20%资金
    stop_loss=ST_FixedPercent(0.08),    # 8%止损
    initial_cash=100000,                # 10万初始资金
    max_positions=10                    # 最多10只股票
)

# 3. 运行回测
result = engine.run(
    stock_list=['000001', '000002', '600000'],
    start_date='2025-01-01',
    end_date='2025-09-30'
)

# 4. 查看结果
print(f"总收益率: {result.total_return:.2f}%")
print(f"夏普比率: {result.sharpe_ratio:.2f}")
print(f"最大回撤: {result.max_drawdown:.2f}%")
print(f"胜率: {result.win_rate:.2f}%")
```

---

## 核心概念

### 1. 数据适配器 (DataAdapter)

**作用**: 连接SQLite数据库，提供K线数据和股票信息

```python
from hikyuu_integration import HikyuuStyleDataAdapter
from data_adapter.database_manager import DatabaseManager

# 创建适配器
db = DatabaseManager(db_path='data_adapter/stock_data.db')
adapter = HikyuuStyleDataAdapter(
    db_manager=db,
    cache_capacity=1000  # 缓存容量
)

# 获取股票列表
stocks = adapter.get_all_stocks('A股')

# 获取K线数据
kdata = adapter.get_kdata('000001', Query(start='2025-01-01'))

# 数据预加载(性能优化)
adapter.preload_data(stocks, '2025-01-01', '2025-09-30')
```

**关键方法**:
- `get_stock(code)`: 获取股票对象
- `get_kdata(code, query)`: 获取K线数据
- `preload_data(stocks, start, end)`: 批量预加载数据
- `get_cache_stats()`: 获取缓存统计
- `print_cache_stats()`: 打印缓存统计

### 2. Query对象

**作用**: 定义数据查询条件

```python
from hikyuu_integration import Query

# 最近150天
q1 = Query(-150)

# 日期区间
q2 = Query(start='2025-01-01', end='2025-09-30')

# 从某日期开始
q3 = Query(start='2025-01-01')
```

### 3. Signal (交易信号)

**作用**: 生成买入/卖出信号

#### 内置Signal

**BBISignal** - BBI指标信号
```python
from hikyuu_integration import BBISignal

signal = BBISignal()
```
- **买入**: 收盘价上穿BBI
- **卖出**: 收盘价下穿BBI

**KDJSignal** - KDJ指标信号
```python
from hikyuu_integration import KDJSignal

signal = KDJSignal(
    k_period=9,      # K周期
    d_period=3,      # D周期
    j_period=3,      # J周期
    oversold=20,     # 超卖阈值
    overbought=80    # 超买阈值
)
```
- **买入**: K值<20且K上穿D
- **卖出**: K值>80且K下穿D

**CompositeSignal** - 组合信号
```python
from hikyuu_integration import CompositeSignal, BBISignal, KDJSignal

signal = CompositeSignal([
    BBISignal(),
    KDJSignal()
])
```
- **买入**: 所有子信号都发出买入
- **卖出**: 任一子信号发出卖出

#### 自定义Signal

```python
from hikyuu_integration import SignalBase

class MySignal(SignalBase):
    def __init__(self, param1=10):
        super().__init__()
        self.param1 = param1

    def calculate(self, kdata):
        """计算指标"""
        # 添加你的指标计算
        self.add_indicator('my_indicator', values)

    def should_buy(self, date, kdata):
        """买入判断"""
        # 返回True/False
        return condition

    def should_sell(self, date, kdata):
        """卖出判断"""
        # 返回True/False
        return condition
```

### 4. 资金管理 (MoneyManager)

**作用**: 控制每次买入的资金量和股数

#### MM_FixedPercent - 固定百分比
```python
from hikyuu_integration import MM_FixedPercent

mm = MM_FixedPercent(0.2)  # 每次使用20%可用资金
```

#### MM_FixedCount - 固定股数
```python
from hikyuu_integration import MM_FixedCount

mm = MM_FixedCount(1000)  # 每次买入1000股
```

#### MM_FixedRisk - 固定风险
```python
from hikyuu_integration import MM_FixedRisk

mm = MM_FixedRisk(
    risk_percent=0.02,   # 每次风险2%
    stop_loss_percent=0.08  # 止损8%
)
```

### 5. 止损策略 (StopLoss)

**作用**: 自动止损和止盈

#### ST_FixedPercent - 固定百分比止损
```python
from hikyuu_integration import ST_FixedPercent

sl = ST_FixedPercent(0.08)  # 亏损8%止损
```

#### ST_ProfitGoal - 目标止盈
```python
from hikyuu_integration import ST_ProfitGoal

sl = ST_ProfitGoal(
    profit_target=0.20,  # 盈利20%止盈
    trailing_stop=0.10    # 回撤10%止盈
)
```

#### ST_Trailing - 移动止损
```python
from hikyuu_integration import ST_Trailing

sl = ST_Trailing(
    initial_stop=0.08,   # 初始止损8%
    trailing_pct=0.05    # 跟踪5%
)
```

#### ST_Composite - 组合止损
```python
from hikyuu_integration import ST_Composite

sl = ST_Composite([
    ST_FixedPercent(0.08),      # 固定止损8%
    ST_ProfitGoal(0.20, 0.10)   # 20%止盈+10%回撤
])
```

### 6. 回测引擎 (BacktestEngine)

**作用**: 执行回测，管理交易流程

```python
from hikyuu_integration import HikyuuStyleBacktestEngine

engine = HikyuuStyleBacktestEngine(
    data_adapter=adapter,           # 数据适配器
    signal=signal,                  # 交易信号
    money_manager=mm,               # 资金管理
    stop_loss=sl,                   # 止损策略
    initial_cash=100000,            # 初始资金
    max_positions=10                # 最大持仓数
)

result = engine.run(
    stock_list=['000001', '000002'],
    start_date='2025-01-01',
    end_date='2025-09-30'
)
```

### 7. 并行回测引擎 (ParallelBacktestEngine)

**作用**: 多进程并行回测，适合大规模回测

```python
from hikyuu_integration import ParallelBacktestEngine

engine = ParallelBacktestEngine(
    data_adapter=adapter,
    signal=BBISignal(),
    money_manager=MM_FixedPercent(0.2),
    stop_loss=ST_FixedPercent(0.08),
    initial_cash=100000,
    max_workers=4  # 4个并行进程
)

result = engine.run(
    stock_list=stocks,  # 可以是100+只股票
    start_date='2025-01-01',
    end_date='2025-09-30'
)
```

**注意**: 并行回测适合>100只股票的场景

---

## API参考

### HikyuuStyleDataAdapter

#### 构造函数
```python
__init__(db_manager=None, cache_capacity=1000)
```
- `db_manager`: DatabaseManager实例
- `cache_capacity`: 缓存容量，默认1000

#### 主要方法

**get_stock(code: str) → Stock**
- 获取股票对象

**get_kdata(code: str, query: Query) → KData**
- 获取K线数据

**preload_data(stock_list: List[str], start_date: str, end_date: str)**
- 批量预加载数据到缓存

**get_all_stocks(stock_type: str = 'A股') → List[str]**
- 获取所有股票代码

**get_trading_dates(start_date: str, end_date: str) → List[str]**
- 获取交易日期列表

**get_cache_stats() → Dict**
- 获取缓存统计信息

**print_cache_stats()**
- 打印缓存统计

**clear_cache()**
- 清空所有缓存

---

### HikyuuStyleBacktestEngine

#### 构造函数
```python
__init__(
    data_adapter,
    signal,
    money_manager=None,
    stop_loss=None,
    initial_cash=100000,
    max_positions=10
)
```

#### 主要方法

**run(stock_list: List[str], start_date: str, end_date: str) → BacktestResult**
- 运行回测
- 返回BacktestResult对象

---

### ParallelBacktestEngine

#### 构造函数
```python
__init__(
    data_adapter,
    signal,
    money_manager=None,
    stop_loss=None,
    initial_cash=100000,
    max_positions=10,
    max_workers=None  # 默认=CPU核心数
)
```

#### 主要方法

**run(stock_list: List[str], start_date: str, end_date: str) → Dict**
- 运行并行回测
- 返回合并后的结果字典

---

### BacktestResult

#### 属性

- `total_return`: 总收益率 (%)
- `annualized_return`: 年化收益率 (%)
- `sharpe_ratio`: 夏普比率
- `max_drawdown`: 最大回撤 (%)
- `win_rate`: 胜率 (%)
- `portfolio`: Portfolio对象
- `start_date`: 回测开始日期
- `end_date`: 回测结束日期

#### 方法

**print_summary()**
- 打印回测摘要

---

## 使用示例

### 示例1: 基本回测

```python
from hikyuu_integration import *
from data_adapter.database_manager import DatabaseManager

# 创建适配器
db = DatabaseManager(db_path='data_adapter/stock_data.db')
adapter = HikyuuStyleDataAdapter(db_manager=db)

# 获取股票列表
stocks = adapter.get_all_stocks('A股')[:20]

# 创建引擎
engine = HikyuuStyleBacktestEngine(
    data_adapter=adapter,
    signal=BBISignal(),
    money_manager=MM_FixedPercent(0.2),
    stop_loss=ST_FixedPercent(0.08),
    initial_cash=100000,
    max_positions=10
)

# 运行回测
result = engine.run(
    stock_list=stocks,
    start_date='2025-07-01',
    end_date='2025-09-30'
)

# 打印结果
result.print_summary()
```

### 示例2: 组合Signal策略

```python
from hikyuu_integration import *

# 创建组合信号
signal = CompositeSignal([
    BBISignal(),
    KDJSignal(oversold=30, overbought=70)
])

engine = HikyuuStyleBacktestEngine(
    data_adapter=adapter,
    signal=signal,
    money_manager=MM_FixedPercent(0.15),
    stop_loss=ST_Composite([
        ST_FixedPercent(0.08),
        ST_ProfitGoal(0.20, 0.10)
    ]),
    initial_cash=100000,
    max_positions=8
)

result = engine.run(stocks, start_date, end_date)
```

### 示例3: 并行回测

```python
from hikyuu_integration import *

# 获取大量股票
stocks = adapter.get_all_stocks('A股')[:200]

# 创建并行引擎
engine = ParallelBacktestEngine(
    data_adapter=adapter,
    signal=BBISignal(),
    money_manager=MM_FixedPercent(0.2),
    stop_loss=ST_FixedPercent(0.08),
    initial_cash=100000,
    max_workers=8  # 8个进程并行
)

# 运行并行回测
result = engine.run(
    stock_list=stocks,
    start_date='2025-01-01',
    end_date='2025-09-30'
)

# 查看结果
print(f"总收益率: {result['total_return']:.2f}%")
print(f"总交易次数: {result['total_trades']}")
print(f"测试股票数: {result['num_stocks']}")
```

### 示例4: 数据预加载优化

```python
from hikyuu_integration import *

# 创建适配器
adapter = HikyuuStyleDataAdapter(db_manager=db, cache_capacity=500)

# 预加载数据 (大幅提升性能)
stocks = adapter.get_all_stocks('A股')[:100]
adapter.preload_data(stocks, '2025-01-01', '2025-09-30')

# 后续get_kdata将直接从缓存读取
engine = HikyuuStyleBacktestEngine(
    data_adapter=adapter,
    signal=BBISignal(),
    money_manager=MM_FixedPercent(0.2),
    stop_loss=ST_FixedPercent(0.08),
    initial_cash=100000
)

result = engine.run(stocks, '2025-01-01', '2025-09-30')

# 查看缓存统计
adapter.print_cache_stats()
```

### 示例5: 自定义Signal

```python
from hikyuu_integration import SignalBase

class MASignal(SignalBase):
    """均线交叉信号"""

    def __init__(self, short_period=5, long_period=20):
        super().__init__()
        self.short_period = short_period
        self.long_period = long_period

    def calculate(self, kdata):
        """计算均线"""
        df = kdata.to_dataframe()

        # 短期均线
        ma_short = df['close'].rolling(window=self.short_period).mean()
        self.add_indicator('ma_short', ma_short.tolist())

        # 长期均线
        ma_long = df['close'].rolling(window=self.long_period).mean()
        self.add_indicator('ma_long', ma_long.tolist())

    def should_buy(self, date, kdata):
        """金叉买入"""
        if date not in kdata.trade_dates:
            return False

        idx = kdata.trade_dates.index(date)
        if idx < 1:
            return False

        ma_short = self.get_indicator_value('ma_short', idx)
        ma_short_prev = self.get_indicator_value('ma_short', idx - 1)
        ma_long = self.get_indicator_value('ma_long', idx)
        ma_long_prev = self.get_indicator_value('ma_long', idx - 1)

        if ma_short is None or ma_long is None:
            return False

        # 金叉：短期均线上穿长期均线
        if ma_short_prev <= ma_long_prev and ma_short > ma_long:
            return True

        return False

    def should_sell(self, date, kdata):
        """死叉卖出"""
        if date not in kdata.trade_dates:
            return False

        idx = kdata.trade_dates.index(date)
        if idx < 1:
            return False

        ma_short = self.get_indicator_value('ma_short', idx)
        ma_short_prev = self.get_indicator_value('ma_short', idx - 1)
        ma_long = self.get_indicator_value('ma_long', idx)
        ma_long_prev = self.get_indicator_value('ma_long', idx - 1)

        if ma_short is None or ma_long is None:
            return False

        # 死叉：短期均线下穿长期均线
        if ma_short_prev >= ma_long_prev and ma_short < ma_long:
            return True

        return False


# 使用自定义Signal
signal = MASignal(short_period=5, long_period=20)

engine = HikyuuStyleBacktestEngine(
    data_adapter=adapter,
    signal=signal,
    money_manager=MM_FixedPercent(0.2),
    stop_loss=ST_FixedPercent(0.08),
    initial_cash=100000
)

result = engine.run(stocks, start_date, end_date)
```

---

## 性能优化

### 1. 数据预加载

**问题**: 每次get_kdata都查询数据库很慢

**解决**: 使用preload_data批量预加载

```python
# ❌ 慢: 每次查询数据库
for code in stocks:
    kdata = adapter.get_kdata(code, Query(start='2025-01-01'))

# ✅ 快: 批量预加载
adapter.preload_data(stocks, '2025-01-01', '2025-09-30')
for code in stocks:
    kdata = adapter.get_kdata(code, Query(start='2025-01-01'))
```

**性能提升**: 3-10倍

### 2. 智能缓存

**SmartCacheManager自动优化**:
- LRU淘汰策略
- 子范围查询匹配
- O(1)时间复杂度

**查看缓存效率**:
```python
stats = adapter.get_cache_stats()
print(f"命中率: {stats['hit_rate']:.1f}%")
print(f"缓存大小: {stats['size']}/{stats['capacity']}")
```

### 3. 并行回测

**使用场景**: >100只股票

```python
# 单线程: 适合<50只股票
engine = HikyuuStyleBacktestEngine(...)

# 并行: 适合>100只股票
engine = ParallelBacktestEngine(..., max_workers=8)
```

**预期加速比**:
- 100只股票: 2-3x
- 500只股票: 3-4x
- 1000只股票: 3-4x

### 4. 缓存容量调整

```python
# 小规模回测 (<50只股票)
adapter = HikyuuStyleDataAdapter(db_manager=db, cache_capacity=100)

# 中规模回测 (50-200只股票)
adapter = HikyuuStyleDataAdapter(db_manager=db, cache_capacity=500)

# 大规模回测 (>200只股票)
adapter = HikyuuStyleDataAdapter(db_manager=db, cache_capacity=2000)
```

---

## 最佳实践

### 1. 策略开发流程

```
1. 定义Signal逻辑
   ↓
2. 小规模测试 (5-10只股票, 1个月)
   ↓
3. 参数优化
   ↓
4. 中规模验证 (50只股票, 3个月)
   ↓
5. 大规模回测 (200+只股票, 6个月+)
   ↓
6. 实盘模拟
```

### 2. 参数设置建议

**初始资金**:
- 测试: 100,000元
- 实盘模拟: 实际资金

**最大持仓数**:
- 小资金 (<10万): 3-5只
- 中等资金 (10-50万): 5-10只
- 大资金 (>50万): 10-20只

**资金管理**:
- 保守: MM_FixedPercent(0.10)  # 10%
- 中等: MM_FixedPercent(0.20)  # 20%
- 激进: MM_FixedPercent(0.30)  # 30%

**止损设置**:
- 保守: ST_FixedPercent(0.05)  # 5%
- 中等: ST_FixedPercent(0.08)  # 8%
- 激进: ST_FixedPercent(0.10)  # 10%

### 3. Signal组合策略

**趋势+震荡**:
```python
signal = CompositeSignal([
    BBISignal(),      # 趋势
    KDJSignal()       # 震荡
])
```

**多周期确认**:
```python
signal = CompositeSignal([
    MASignal(5, 20),   # 短期
    MASignal(20, 60)   # 中期
])
```

### 4. 风险控制

**多层止损**:
```python
stop_loss = ST_Composite([
    ST_FixedPercent(0.08),      # 固定止损
    ST_ProfitGoal(0.20, 0.10),  # 止盈+回撤
    ST_Trailing(0.08, 0.05)     # 移动止损
])
```

**仓位控制**:
- 单只股票不超过20%
- 同行业不超过30%
- 保留5-10%现金储备

### 5. 回测注意事项

**避免过拟合**:
- 使用样本外数据验证
- 不要过度优化参数
- 关注夏普比率而非收益率

**考虑交易成本**:
- 手续费: 默认万三
- 滑点: 实盘可能有1-2跳
- 印花税: 卖出千一

**市场环境**:
- 不同市场环境策略表现不同
- 定期评估策略有效性
- 及时调整参数

---

## 常见问题

### Q1: 如何处理停牌股票？

**A**: 框架自动处理停牌，停牌期间不会买入也不会卖出。

### Q2: T+1规则是如何实现的？

**A**: Broker类自动检查，当日买入的股票次日才能卖出。

```python
# 2025-09-01买入
portfolio.buy('000001', 1000, 10.0, '2025-09-01')

# 2025-09-01无法卖出 (T+1)
# 2025-09-02可以卖出
portfolio.sell('000001', 1000, 10.5, '2025-09-02')
```

### Q3: 如何处理涨跌停？

**A**: Broker类自动检查涨跌停限制，涨停无法买入，跌停无法卖出。

### Q4: 并行回测结果为什么与单线程不同？

**A**:
- 单线程: 多只股票共享Portfolio，有持仓数量限制
- 并行: 每只股票独立Portfolio，无相互影响

建议:
- 策略评估: 使用并行
- 组合模拟: 使用单线程

### Q5: 缓存命中率低怎么办？

**A**:
1. 增加cache_capacity
2. 使用preload_data预加载
3. 查看缓存统计分析原因

```python
adapter.print_cache_stats()
```

### Q6: 如何保存回测结果？

**A**:
```python
import pickle

# 保存
with open('result.pkl', 'wb') as f:
    pickle.dump(result, f)

# 加载
with open('result.pkl', 'rb') as f:
    result = pickle.load(f)
```

### Q7: 可以回测期货吗？

**A**: 目前仅支持A股，期货需要修改Broker类实现保证金和杠杆机制。

### Q8: 如何获得更详细的日志？

**A**:
```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

---

## 附录

### A. 完整示例代码

参见: `hikyuu_integration/test_comprehensive_backtest.py`

### B. 测试用例

参见: `hikyuu_integration/test_hikyuu_integration.py`

### C. 性能基准

参见: `hikyuu_integration/benchmark_cache.py`

### D. 框架对比

参见: `HIKYUU_PHASE4_4_FRAMEWORK_COMPARISON.md`

### E. 版本历史

**v0.2.1** (2025-10-10):
- ✅ SmartCacheManager (LRU + 智能匹配)
- ✅ ParallelBacktestEngine (多进程并行)
- ✅ 修复datetime类型问题
- ✅ Trade.pnl字段
- ✅ 完整测试覆盖

**v0.2.0** (2025-10-09):
- ✅ 基础框架完成
- ✅ Signal/MM/SL系统
- ✅ Broker + Portfolio
- ✅ 基本回测引擎

### F. 联系方式

- 项目地址: `/Users/yangxu/StockTradebyZ/hikyuu_integration/`
- 文档地址: `/Users/yangxu/StockTradebyZ/HIKYUU_*.md`
- 作者: StockTradebyZ Team

---

**最后更新**: 2025-10-10
**版本**: v0.2.1
**状态**: Production Ready 🚀
