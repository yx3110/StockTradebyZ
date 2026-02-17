# 策略与回测引擎解耦设计方案

**设计时间**: 2025-10-12
**核心理念**: 策略即插件，回测引擎只负责执行

---

## 🎯 设计目标

### 当前问题

```python
# ❌ 当前架构 - 策略硬编码在引擎中
class ExtensibleBacktestEngine:
    def __init__(self):
        self.take_profit_pct = 0.15      # 策略参数混在引擎里
        self.stop_loss_pct = 0.08
        self.max_holding_days = 20
        # ...

    def _check_take_profit(self):        # 策略逻辑在引擎方法中
        # 硬编码的止盈逻辑

    def _execute_rebalance(self):        # 调仓逻辑在引擎方法中
        # 硬编码的调仓逻辑
```

**问题**:
- ❌ 无法轻松切换策略
- ❌ 无法对比多种策略
- ❌ 参数调优困难
- ❌ 策略逻辑分散

### 理想架构

```python
# ✅ 解耦后架构 - 策略是独立组件
class TradingStrategy(ABC):
    """策略抽象基类"""
    @abstractmethod
    def should_sell_position(self, position, current_price, date, selected_stocks) -> Tuple[bool, str]:
        """决策：是否卖出某个持仓"""

    @abstractmethod
    def should_take_profit(self, position, current_price) -> bool:
        """决策：是否止盈"""

    @abstractmethod
    def should_stop_loss(self, position, current_price) -> bool:
        """决策：是否止损"""

class ExtensibleBacktestEngine:
    def __init__(self, strategy: TradingStrategy):
        self.strategy = strategy  # 注入策略

    def _check_take_profit(self):
        for position in positions:
            if self.strategy.should_take_profit(position, price):
                self._execute_sell(...)  # 引擎只负责执行
```

**优势**:
- ✅ 策略可插拔
- ✅ 轻松对比多种策略
- ✅ 独立优化参数
- ✅ 清晰的职责分离

---

## 🏗️ 架构设计

### 1. 策略层次结构

```
TradingStrategy (抽象基类)
├── ConservativeStrategy (保守策略)
│   ├── 止盈: 10%
│   ├── 止损: 5%
│   └── 持仓: 30天
│
├── BalancedStrategy (平衡策略) ← 当前实现
│   ├── 止盈: 15%
│   ├── 止损: 8%
│   └── 持仓: 20天
│
├── AggressiveStrategy (激进策略)
│   ├── 止盈: 20%
│   ├── 止损: 10%
│   └── 持仓: 10天
│
└── CustomStrategy (自定义策略)
    └── 用户自定义参数
```

### 2. 策略接口设计

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass

@dataclass
class Position:
    """持仓信息"""
    stock_code: str
    shares: int
    avg_cost: float
    entry_date: str
    entry_score: float

@dataclass
class StrategyConfig:
    """策略配置"""
    # 止盈止损
    take_profit_pct: float = 0.15
    stop_loss_pct: float = 0.08

    # 持仓管理
    max_holding_days: int = 20
    min_score_for_hold: float = 75.0

    # 调仓配置
    enable_rebalance_sell: bool = True
    rebalance_frequency: int = 5

    # 仓位管理
    max_positions: int = 10
    max_position_pct: float = 0.15

    # 策略元数据
    name: str = "未命名策略"
    description: str = ""
    risk_level: str = "中等"  # 低/中等/高

class TradingStrategy(ABC):
    """交易策略抽象基类"""

    def __init__(self, config: StrategyConfig):
        self.config = config

    # ========== 核心决策接口 ==========

    @abstractmethod
    def should_sell_on_rebalance(
        self,
        position: Position,
        current_price: float,
        current_date: str,
        selected_stocks: List[Dict]
    ) -> Tuple[bool, str]:
        """
        调仓时是否卖出某个持仓

        Returns:
            (should_sell, reason)
        """
        pass

    @abstractmethod
    def should_take_profit(
        self,
        position: Position,
        current_price: float
    ) -> bool:
        """是否止盈"""
        pass

    @abstractmethod
    def should_stop_loss(
        self,
        position: Position,
        current_price: float
    ) -> bool:
        """是否止损"""
        pass

    @abstractmethod
    def should_check_holding_period(
        self,
        position: Position,
        current_date: str
    ) -> bool:
        """是否超过持仓期限"""
        pass

    @abstractmethod
    def calculate_position_size(
        self,
        stock_code: str,
        stock_price: float,
        available_capital: float,
        current_positions: int
    ) -> int:
        """计算买入股数"""
        pass

    # ========== 辅助方法 ==========

    def calculate_profit_pct(self, position: Position, current_price: float) -> float:
        """计算盈亏百分比"""
        return (current_price - position.avg_cost) / position.avg_cost

    def calculate_holding_days(self, entry_date: str, current_date: str) -> int:
        """计算持仓天数"""
        entry = datetime.strptime(entry_date, '%Y-%m-%d')
        current = datetime.strptime(current_date, '%Y-%m-%d')
        return (current - entry).days

    def get_config(self) -> StrategyConfig:
        """获取策略配置"""
        return self.config

    def get_info(self) -> Dict:
        """获取策略信息"""
        return {
            'name': self.config.name,
            'description': self.config.description,
            'risk_level': self.config.risk_level,
            'config': self.config.__dict__
        }
```

### 3. 具体策略实现示例

```python
class BalancedStrategy(TradingStrategy):
    """平衡策略 - 中等风险收益"""

    def __init__(self):
        config = StrategyConfig(
            take_profit_pct=0.15,
            stop_loss_pct=0.08,
            max_holding_days=20,
            min_score_for_hold=75.0,
            enable_rebalance_sell=True,
            rebalance_frequency=5,
            max_positions=10,
            name="平衡策略",
            description="15%止盈，8%止损，最长持仓20天",
            risk_level="中等"
        )
        super().__init__(config)

    def should_sell_on_rebalance(self, position, current_price, current_date, selected_stocks):
        selected_codes = {s['stock_code'] for s in selected_stocks}
        selected_scores = {s['stock_code']: s['score'] for s in selected_stocks}

        # 检查1: 不在新选股列表
        if position.stock_code not in selected_codes:
            return True, "rebalance_not_selected"

        # 检查2: 评分过低
        if position.stock_code in selected_scores:
            if selected_scores[position.stock_code] < self.config.min_score_for_hold:
                return True, f"rebalance_low_score_{selected_scores[position.stock_code]:.1f}"

        # 检查3: 持仓过久
        holding_days = self.calculate_holding_days(position.entry_date, current_date)
        if holding_days > self.config.max_holding_days:
            return True, f"rebalance_max_holding_{holding_days}d"

        return False, ""

    def should_take_profit(self, position, current_price):
        profit_pct = self.calculate_profit_pct(position, current_price)
        return profit_pct > self.config.take_profit_pct

    def should_stop_loss(self, position, current_price):
        profit_pct = self.calculate_profit_pct(position, current_price)
        return profit_pct < -self.config.stop_loss_pct

    def should_check_holding_period(self, position, current_date):
        holding_days = self.calculate_holding_days(position.entry_date, current_date)
        return holding_days > self.config.max_holding_days

    def calculate_position_size(self, stock_code, stock_price, available_capital, current_positions):
        # 平均分配资金
        max_new_positions = self.config.max_positions - current_positions
        if max_new_positions <= 0:
            return 0

        target_allocation = available_capital / max_new_positions
        shares = int(target_allocation / (stock_price * 100)) * 100
        return shares


class ConservativeStrategy(TradingStrategy):
    """保守策略 - 低风险"""

    def __init__(self):
        config = StrategyConfig(
            take_profit_pct=0.10,      # 更低的止盈
            stop_loss_pct=0.05,        # 更严格的止损
            max_holding_days=30,       # 更长的持仓周期
            min_score_for_hold=80.0,   # 更高的评分要求
            max_positions=8,           # 更少的持仓数
            name="保守策略",
            description="10%止盈，5%止损，高评分，长周期",
            risk_level="低"
        )
        super().__init__(config)

    # 实现相同的接口，但参数更保守...


class AggressiveStrategy(TradingStrategy):
    """激进策略 - 高风险高收益"""

    def __init__(self):
        config = StrategyConfig(
            take_profit_pct=0.20,      # 更高的止盈目标
            stop_loss_pct=0.10,        # 更宽松的止损
            max_holding_days=10,       # 更短的持仓周期
            min_score_for_hold=70.0,   # 更低的评分要求
            max_positions=15,          # 更多的持仓数
            rebalance_frequency=3,     # 更频繁的调仓
            name="激进策略",
            description="20%止盈，10%止损，高频交易",
            risk_level="高"
        )
        super().__init__(config)

    # 实现相同的接口，但参数更激进...
```

### 4. 回测引擎改造

```python
class ExtensibleBacktestEngine:
    """可扩展回测引擎 - 策略解耦版"""

    def __init__(
        self,
        strategy: TradingStrategy,  # 🆕 注入策略
        initial_capital: float = 5000000,
        max_workers: int = 6,
        commission_rate: float = 0.0003,
        stamp_tax: float = 0.001,
        min_score_threshold: float = 80.0
    ):
        self.strategy = strategy  # 🆕 策略实例
        self.initial_capital = initial_capital
        # ... 其他引擎参数

        # 🆕 从策略获取配置
        self.max_positions = strategy.config.max_positions
        self.rebalance_freq = strategy.config.rebalance_frequency

    def _check_take_profit(self, date: str):
        """检查止盈 - 委托给策略"""
        positions_to_sell = []

        for stock_code, pos_dict in self.positions.items():
            if pos_dict['shares'] > 0:
                # 转换为Position对象
                position = Position(
                    stock_code=stock_code,
                    shares=pos_dict['shares'],
                    avg_cost=pos_dict['avg_cost'],
                    entry_date=pos_dict['entry_date'],
                    entry_score=pos_dict.get('entry_score', 0)
                )

                current_price = self._get_stock_price(stock_code, date)
                if current_price:
                    # 🆕 委托给策略决策
                    if self.strategy.should_take_profit(position, current_price):
                        positions_to_sell.append(stock_code)

        for stock_code in positions_to_sell:
            self._execute_sell(stock_code, date, "take_profit")

    def _check_stop_loss(self, date: str):
        """检查止损 - 委托给策略"""
        positions_to_sell = []

        for stock_code, pos_dict in self.positions.items():
            if pos_dict['shares'] > 0:
                position = Position(
                    stock_code=stock_code,
                    shares=pos_dict['shares'],
                    avg_cost=pos_dict['avg_cost'],
                    entry_date=pos_dict['entry_date'],
                    entry_score=pos_dict.get('entry_score', 0)
                )

                current_price = self._get_stock_price(stock_code, date)
                if current_price:
                    # 🆕 委托给策略决策
                    if self.strategy.should_stop_loss(position, current_price):
                        positions_to_sell.append(stock_code)

        for stock_code in positions_to_sell:
            self._execute_sell(stock_code, date, "stop_loss")

    def _rebalance_sell_positions(self, date: str, selected_stocks: List[Dict]):
        """调仓卖出 - 委托给策略"""
        positions_to_sell = []

        for stock_code, pos_dict in self.positions.items():
            if pos_dict['shares'] > 0:
                position = Position(
                    stock_code=stock_code,
                    shares=pos_dict['shares'],
                    avg_cost=pos_dict['avg_cost'],
                    entry_date=pos_dict['entry_date'],
                    entry_score=pos_dict.get('entry_score', 0)
                )

                current_price = self._get_stock_price(stock_code, date)
                if current_price:
                    # 🆕 委托给策略决策
                    should_sell, reason = self.strategy.should_sell_on_rebalance(
                        position, current_price, date, selected_stocks
                    )

                    if should_sell:
                        positions_to_sell.append((stock_code, reason))

        for stock_code, reason in positions_to_sell:
            self._execute_sell(stock_code, date, reason)

        return len(positions_to_sell)
```

---

## 🎯 使用示例

### 单策略回测

```python
from trading_strategy import BalancedStrategy
from extensible_backtest_engine import ExtensibleBacktestEngine

# 创建策略
strategy = BalancedStrategy()

# 注入策略到引擎
engine = ExtensibleBacktestEngine(
    strategy=strategy,
    initial_capital=1000000
)

# 运行回测
results = engine.run_backtest(
    versions=['V3.7'],
    start_date='2025-07-01',
    end_date='2025-09-30'
)
```

### 多策略对比

```python
from trading_strategy import ConservativeStrategy, BalancedStrategy, AggressiveStrategy

strategies = {
    'conservative': ConservativeStrategy(),
    'balanced': BalancedStrategy(),
    'aggressive': AggressiveStrategy()
}

results = {}

for name, strategy in strategies.items():
    engine = ExtensibleBacktestEngine(strategy=strategy)
    result = engine.run_backtest(
        versions=['V3.7'],
        start_date='2025-07-01',
        end_date='2025-09-30'
    )
    results[name] = result

# 对比分析
compare_strategies(results)
```

### 自定义策略

```python
from trading_strategy import TradingStrategy, StrategyConfig

class MyCustomStrategy(TradingStrategy):
    def __init__(self):
        # 自定义参数
        config = StrategyConfig(
            take_profit_pct=0.12,
            stop_loss_pct=0.06,
            max_holding_days=15,
            name="我的自定义策略"
        )
        super().__init__(config)

    def should_take_profit(self, position, current_price):
        # 自定义止盈逻辑
        profit = self.calculate_profit_pct(position, current_price)

        # 例如：根据持仓天数调整止盈线
        holding_days = self.calculate_holding_days(position.entry_date, datetime.now().strftime('%Y-%m-%d'))

        if holding_days < 5:
            return profit > 0.15  # 短期持仓要求更高收益
        else:
            return profit > 0.10  # 长期持仓降低要求

    # 实现其他方法...

# 使用自定义策略
strategy = MyCustomStrategy()
engine = ExtensibleBacktestEngine(strategy=strategy)
```

---

## 📊 策略对比分析框架

```python
def compare_strategies(results: Dict[str, Dict]) -> pd.DataFrame:
    """对比多个策略的回测结果"""

    comparison = []

    for strategy_name, result in results.items():
        comparison.append({
            '策略名称': strategy_name,
            '总收益率': result['total_return'],
            '年化收益': result['annual_return'],
            '夏普比率': result['sharpe_ratio'],
            '最大回撤': result['max_drawdown'],
            '胜率': result['win_rate'],
            '交易次数': result['total_trades'],
            '风险等级': result['strategy_info']['risk_level']
        })

    df = pd.DataFrame(comparison)
    df = df.sort_values('夏普比率', ascending=False)

    return df
```

---

## ✅ 优势总结

### 1. 灵活性
- 轻松切换策略
- 快速实验新想法
- 支持自定义策略

### 2. 可维护性
- 清晰的职责分离
- 策略逻辑集中管理
- 易于测试和调试

### 3. 可扩展性
- 添加新策略无需修改引擎
- 支持策略组合
- 支持动态参数调整

### 4. 可对比性
- 同一模型不同策略对比
- 同一策略不同模型对比
- 多维度性能分析

### 5. 优化效率
- 策略参数独立优化
- 网格搜索最优参数
- A/B测试不同策略

---

## 🚀 实施计划

### Phase 1: 基础架构 (优先级最高)
- [ ] 创建 `trading_strategy.py` (抽象基类)
- [ ] 实现 `BalancedStrategy` (当前策略)
- [ ] 改造回测引擎支持策略注入

### Phase 2: 策略库
- [ ] 实现 `ConservativeStrategy`
- [ ] 实现 `AggressiveStrategy`
- [ ] 实现策略对比工具

### Phase 3: 高级功能
- [ ] 策略参数网格搜索
- [ ] 策略性能可视化
- [ ] 策略组合优化

---

**设计者**: Claude Code
**复审**: 待用户确认
**优先级**: 🔥 高 (用户正确指出了架构问题)
