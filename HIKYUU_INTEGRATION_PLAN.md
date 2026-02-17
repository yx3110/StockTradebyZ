# Hikyuu量化框架整合计划

## 📋 项目概况

### 目标
将Hikyuu量化回测框架与现有StockTradebyZ系统整合，利用Hikyuu的高性能回测引擎加速我们的V3.7/V3.8/V3.81评分系统的验证和优化。

### 背景
- **现有系统**: StockTradebyZ - 成熟的A股量化交易系统，包含7,111只股票数据，SQLite数据库，3个ML评分系统
- **新框架**: Hikyuu - 基于C++/Python的高性能量化框架，专为A股市场设计，支持快速回测
- **性能优势**: AMD 7950x测试显示，1913万日K线计算20日MA仅需166毫秒

## 🏗️ Hikyuu架构分析

### 核心组件
```
Hikyuu Framework
├── C++核心库 (hikyuu_cpp/)
│   ├── 数据存储 (HDF5/MySQL/SQLite)
│   ├── 指标计算引擎
│   └── 回测引擎
├── Python包装层 (hikyuu_pywrap/)
│   └── pybind11接口
├── Python API (hikyuu/)
│   ├── trade_sys/ - 交易系统
│   ├── indicator/ - 技术指标
│   ├── data/ - 数据接口
│   └── trade_manage/ - 交易管理
└── 交互工具
    └── Jupyter Notebook示例
```

### 回测系统组件
Hikyuu将系统化交易抽象为以下可组合组件：
1. **Environment** (EV): 市场环境判断策略
2. **Condition** (CN): 系统有效条件
3. **Signal** (SG): 信号指示器 ⭐核心
4. **MoneyManager** (MM): 资金管理策略
5. **StopLoss** (ST): 止损/止盈策略
6. **ProfitGoal** (PG): 盈利目标策略
7. **Slippage** (SP): 移滑价差算法
8. **Selector** (SE): 交易对象选择算法 ⭐关键
9. **AllocateFunds** (AF): 资金分配策略

## 🎯 整合策略

### 方案选择

#### 方案A: 完整安装Hikyuu (❌ 不推荐)
**优点**:
- 获得完整的Hikyuu功能
- 可以使用Hikyuu的所有内置指标和策略

**缺点**:
- 需要编译C++代码，依赖xmake
- 编译时间长，维护成本高
- 可能与现有系统产生依赖冲突
- 数据需要转换为HDF5或MySQL格式

#### 方案B: 轻量级适配器模式 (✅ 推荐)
**优点**:
- 无需编译，快速上手
- 保持现有SQLite数据库
- 复用Hikyuu的设计思想，不依赖其实现
- 灵活可控，易于维护

**缺点**:
- 需要自己实现部分功能
- 无法使用Hikyuu的C++性能优势

**推荐原因**:
1. 我们已有完善的数据管理系统(DatabaseManager)
2. 我们已有成熟的ML评分系统(v3.7/v3.8/v3.81)
3. 我们需要的是回测框架设计思想，不是C++性能
4. Python实现更灵活，更易于与现有系统整合

### 最终方案: **借鉴Hikyuu设计+适配器模式**

## 🔧 技术实现方案

### 1. 数据适配层

#### 目标
将现有SQLite数据库适配为Hikyuu风格的数据接口

#### 实现
```python
# hikyuu_integration/data_adapter.py

class HikyuuStyleDataAdapter:
    """
    将StockTradebyZ的SQLite数据适配为Hikyuu风格接口
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.cache = {}

    def get_stock(self, code: str):
        """获取股票对象 (类似hikyuu的sm['sh000001'])"""
        return Stock(code, self)

    def get_kdata(self, code: str, query: Query):
        """
        获取K线数据 (类似hikyuu的s.get_kdata(Query(-150)))

        返回KData对象，包含:
        - datetime列表
        - open, high, low, close, volume数据
        - 支持通过索引访问: kdata[i]
        """
        pass

    def get_indicator_data(self, code: str, indicator_name: str, params: dict):
        """
        获取技术指标数据

        支持的指标:
        - MA, EMA, RSI, MACD, KDJ, BBI等
        """
        pass

class Stock:
    """股票对象，类似hikyuu的Stock"""
    def __init__(self, code: str, adapter):
        self.code = code
        self.adapter = adapter

    def get_kdata(self, query):
        return self.adapter.get_kdata(self.code, query)

class Query:
    """查询对象，类似hikyuu的Query"""
    def __init__(self, days=None, start_date=None, end_date=None):
        self.days = days  # -150表示最近150天
        self.start_date = start_date
        self.end_date = end_date
```

### 2. 评分器信号适配器

#### 目标
将我们的ML评分系统(v3.7/v3.8/v3.81)转换为Hikyuu风格的Signal

#### 实现
```python
# hikyuu_integration/ml_signal_adapter.py

from hikyuu_integration.signal_base import SignalBase

class MLScoringSignal(SignalBase):
    """
    ML评分系统作为Hikyuu Signal使用

    支持v3.7, v3.8, v3.81三个版本
    """

    def __init__(self, ml_version='v3.81', min_score=80.0):
        super().__init__(name=f'{ml_version}_MLSignal')
        self.ml_version = ml_version
        self.min_score = min_score
        self.ml_system = self._init_ml_system()

    def _init_ml_system(self):
        """初始化对应版本的ML系统"""
        if self.ml_version == 'v3.7':
            from ml_models.v37 import V370AdvancedMLSystem
            return V370AdvancedMLSystem()
        elif self.ml_version == 'v3.8':
            from ml_models.v38 import V380AdvancedIncrementalMLSystem
            return V380AdvancedIncrementalMLSystem()
        elif self.ml_version == 'v3.81':
            from ml_models.v381 import V380Level4IntegratedSystem
            return V380Level4IntegratedSystem()
        else:
            raise ValueError(f"Unknown ML version: {self.ml_version}")

    def _calculate(self, kdata):
        """
        计算信号

        Hikyuu回测引擎会调用这个方法来计算买入/卖出信号
        """
        for i in range(len(kdata)):
            date = kdata.get_datetime(i)
            stock_code = kdata.stock.code

            # 使用ML系统计算评分
            score = self.ml_system.calculate_score(stock_code, date)

            if score >= self.min_score:
                self._add_buy_signal(date)
            elif score < self.min_score - 10:  # 评分低于阈值10分，卖出
                self._add_sell_signal(date)
```

### 3. 快速回测引擎

#### 目标
借鉴Hikyuu的设计，创建轻量级但功能完整的回测引擎

#### 实现
```python
# hikyuu_integration/backtest_engine.py

class HikyuuStyleBacktestEngine:
    """
    借鉴Hikyuu设计的快速回测引擎

    支持:
    - 多种Signal组合测试
    - 资金管理策略
    - 止损止盈
    - 并行回测
    - 性能分析
    """

    def __init__(self, data_adapter: HikyuuStyleDataAdapter):
        self.data = data_adapter
        self.portfolio = Portfolio()
        self.broker = Broker()

    def run_system(self,
                   signal: SignalBase,
                   money_manager: MoneyManagerBase,
                   stop_loss: StopLossBase,
                   stock_list: List[str],
                   start_date: str,
                   end_date: str):
        """
        运行交易系统

        参数:
        - signal: 信号指示器 (如MLScoringSignal)
        - money_manager: 资金管理
        - stop_loss: 止损策略
        - stock_list: 股票池
        - start_date/end_date: 回测期间
        """

        # 1. 初始化
        self.portfolio.reset(initial_cash=5000000)

        # 2. 获取交易日期
        trading_dates = self._get_trading_dates(start_date, end_date)

        # 3. 回测循环
        for date in trading_dates:
            # 3.1 检查信号
            for stock_code in stock_list:
                kdata = self.data.get_kdata(stock_code, Query(end_date=date))
                signal.calculate(kdata)

                if signal.should_buy(date):
                    # 计算买入数量
                    buy_num = money_manager.get_buy_num(
                        date, stock_code,
                        price=kdata.get_close(date),
                        available_cash=self.portfolio.cash
                    )
                    # 执行买入
                    self.broker.buy(stock_code, date, buy_num)

                elif signal.should_sell(date):
                    # 执行卖出
                    position = self.portfolio.get_position(stock_code)
                    if position:
                        self.broker.sell(stock_code, date, position.shares)

            # 3.2 检查止损
            stop_loss.check(self.portfolio, date)

            # 3.3 更新组合价值
            self.portfolio.update_value(date, self.data)

        # 4. 计算绩效
        return self._calculate_performance()

    def parallel_test(self,
                     signal_configs: List[dict],
                     stock_list: List[str],
                     start_date: str,
                     end_date: str):
        """
        并行测试多个信号配置

        示例:
        signal_configs = [
            {'ml_version': 'v3.7', 'min_score': 80},
            {'ml_version': 'v3.8', 'min_score': 80},
            {'ml_version': 'v3.81', 'min_score': 80},
        ]
        """
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=len(signal_configs)) as executor:
            futures = []
            for config in signal_configs:
                signal = MLScoringSignal(**config)
                future = executor.submit(
                    self.run_system,
                    signal,
                    MM_FixedCount(1000),  # 固定每次买入1000股
                    ST_FixedPercent(0.08),  # 固定8%止损
                    stock_list,
                    start_date,
                    end_date
                )
                futures.append(future)

            results = [f.result() for f in futures]

        return self._compare_results(results)
```

### 4. 项目结构

```
StockTradebyZ/
├── hikyuu/                          # Hikyuu源码 (仅参考，不编译)
├── hikyuu_integration/              # 🆕 Hikyuu整合模块
│   ├── __init__.py
│   ├── data_adapter.py              # 数据适配层
│   ├── signal_base.py               # Signal基类
│   ├── ml_signal_adapter.py         # ML评分Signal适配器
│   ├── money_manager.py             # 资金管理策略
│   ├── stop_loss.py                 # 止损策略
│   ├── backtest_engine.py           # 快速回测引擎
│   ├── portfolio.py                 # 组合管理
│   ├── broker.py                    # 交易执行
│   └── performance.py               # 绩效分析
├── examples/                        # 🆕 使用示例
│   ├── quick_backtest_example.py    # 快速回测示例
│   ├── ml_signal_test.py            # ML信号测试
│   └── strategy_comparison.py       # 策略对比
└── reports/
    └── hikyuu_backtest/             # 🆕 Hikyuu风格回测报告
```

## 📊 使用示例

### 示例1: 快速回测单个ML版本

```python
from hikyuu_integration import HikyuuStyleBacktestEngine, HikyuuStyleDataAdapter
from hikyuu_integration import MLScoringSignal, MM_FixedCount, ST_FixedPercent
from data_adapter.database_manager import DatabaseManager

# 1. 初始化数据适配器
db = DatabaseManager()
data_adapter = HikyuuStyleDataAdapter(db)

# 2. 创建回测引擎
engine = HikyuuStyleBacktestEngine(data_adapter)

# 3. 创建ML信号
signal = MLScoringSignal(ml_version='v3.81', min_score=80)

# 4. 运行回测
results = engine.run_system(
    signal=signal,
    money_manager=MM_FixedCount(1000),    # 每次买入1000股
    stop_loss=ST_FixedPercent(0.08),      # 8%止损
    stock_list=['000001', '000002', ...],  # 股票池
    start_date='2024-01-01',
    end_date='2025-09-30'
)

# 5. 查看结果
print(f"总收益率: {results['total_return']:.2%}")
print(f"夏普比率: {results['sharpe_ratio']:.2f}")
print(f"最大回撤: {results['max_drawdown']:.2%}")
```

### 示例2: 并行对比多个ML版本

```python
# 并行测试v3.7, v3.8, v3.81
signal_configs = [
    {'ml_version': 'v3.7', 'min_score': 80},
    {'ml_version': 'v3.8', 'min_score': 80},
    {'ml_version': 'v3.81', 'min_score': 80},
]

comparison = engine.parallel_test(
    signal_configs=signal_configs,
    stock_list=get_all_stocks(),
    start_date='2024-01-01',
    end_date='2025-09-30'
)

# 输出对比结果
print(comparison.to_markdown())
```

### 示例3: 自定义Signal

```python
from hikyuu_integration.signal_base import SignalBase

class CustomSignal(SignalBase):
    """自定义信号: 结合量化策略和ML评分"""

    def __init__(self):
        super().__init__(name='CustomSignal')
        self.ml_signal = MLScoringSignal(ml_version='v3.81')

    def _calculate(self, kdata):
        # 1. 计算ML评分
        ml_score = self.ml_signal.calculate_score(kdata)

        # 2. 计算量化指标
        bbi = kdata.get_indicator('BBI', n=10)
        kdj_k = kdata.get_indicator('KDJ_K', n=9)

        # 3. 组合信号
        for i in range(len(kdata)):
            if (ml_score[i] >= 80 and
                kdata.close[i] > bbi[i] and
                kdj_k[i] < 20):
                self._add_buy_signal(kdata.datetime[i])
```

## ⚡ 性能优化

### 数据预加载
```python
# 批量预加载数据到缓存
data_adapter.preload_kdata(stock_list, start_date, end_date)

# 回测时直接从缓存读取，避免重复数据库查询
```

### 并行计算
```python
# 多进程并行计算不同股票的信号
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=8) as executor:
    signals = executor.map(calculate_signal, stock_list)
```

### 向量化计算
```python
# 使用numpy向量化计算技术指标
import numpy as np
closes = np.array(kdata.close)
ma20 = np.convolve(closes, np.ones(20)/20, mode='valid')
```

## 📈 预期收益

### 开发时间
- **数据适配层**: 2-3天
- **Signal适配器**: 1-2天
- **回测引擎**: 3-4天
- **测试优化**: 2-3天
- **总计**: 8-12天

### 性能提升
- 回测速度: 预计比现有`extensible_backtest_engine.py`快2-3倍
- 代码复用: Hikyuu设计模式可用于未来策略开发
- 灵活性: 可快速测试新的Signal组合

### 功能增强
1. **标准化回测流程**: 统一的Signal/MoneyManager/StopLoss接口
2. **策略组合能力**: 轻松组合不同的策略组件
3. **并行测试**: 同时测试多个策略配置
4. **详细分析**: 更丰富的回测报告和性能指标

## 🚀 实施计划

### Phase 1: 基础框架 (3-4天)
- [ ] 创建`hikyuu_integration/`目录结构
- [ ] 实现`HikyuuStyleDataAdapter`
- [ ] 实现`SignalBase`基类
- [ ] 编写数据适配器单元测试

### Phase 2: Signal适配 (2-3天)
- [ ] 实现`MLScoringSignal`适配器
- [ ] 支持v3.7/v3.8/v3.81三个版本
- [ ] 测试Signal计算正确性

### Phase 3: 回测引擎 (3-4天)
- [ ] 实现`HikyuuStyleBacktestEngine`
- [ ] 实现`Portfolio`组合管理
- [ ] 实现`Broker`交易执行
- [ ] 添加止损止盈逻辑

### Phase 4: 优化测试 (2-3天)
- [ ] 性能优化(缓存、并行)
- [ ] 完整回测测试
- [ ] 与`extensible_backtest_engine.py`对比验证
- [ ] 编写使用文档和示例

## 🎓 学习资源

### Hikyuu文档
- 官方文档: https://hikyuu.readthedocs.io/
- 项目首页: https://hikyuu.org/
- GitHub: https://github.com/fasiondog/hikyuu
- 入门示例: hikyuu/hikyuu/examples/notebook/

### 参考文件
- `hikyuu/hikyuu/trade_sys/trade_sys.py` - Signal/MM/ST创建函数
- `hikyuu/docs/examples/quick_crtsg.py` - Turtle策略示例
- `hikyuu/hikyuu/__init__.py` - API导出

## 📝 结论

通过借鉴Hikyuu的优秀设计思想，结合适配器模式，我们可以在**不编译C++代码**的前提下，获得一个**高效、灵活、易维护**的回测框架。这个框架将：

1. ✅ 完美整合现有的SQLite数据库
2. ✅ 无缝集成v3.7/v3.8/v3.81评分系统
3. ✅ 提供统一的策略组件接口
4. ✅ 支持快速并行回测
5. ✅ 保持代码简洁和可维护性

**推荐立即开始Phase 1实施！** 🚀
