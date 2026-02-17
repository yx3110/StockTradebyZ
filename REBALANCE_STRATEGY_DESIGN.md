# 完整调仓策略设计文档

**版本**: 2.0
**设计时间**: 2025-10-12
**目的**: 实现真正的主动调仓策略，替代"买入持有+止损"的被动策略

---

## 🎯 设计目标

1. **真正的调仓**: 定期卖出低评分/不在新列表的股票，买入高评分股票
2. **止盈机制**: 不让盈利股票"坐过山车"
3. **时间轮换**: 避免长期持有低效股票
4. **资金效率**: 充分利用现金和卖出所得

---

## 📊 策略架构

### 核心参数配置

```python
class ExtensibleBacktestEngine:
    def __init__(self, ...):
        # 原有参数
        self.max_position_pct = 0.15      # 单只股票最大仓位15%
        self.max_positions = 10           # 最多持仓10只
        self.stop_loss_pct = 0.08         # 止损线8%
        self.rebalance_freq = 5           # 调仓频率5天

        # 🆕 新增参数
        self.take_profit_pct = 0.15       # 止盈线15%
        self.max_holding_days = 20        # 最长持仓20天
        self.min_score_for_hold = 75.0    # 继续持有的最低评分
        self.enable_rebalance_sell = True # 启用调仓卖出
```

### 调仓日逻辑 (每5天)

```
┌─────────────────────────────────────────┐
│        开始调仓 (Day 0, 5, 10...)       │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  1. 模型选股 (获取高评分股票列表)        │
│     - 从全市场筛选                      │
│     - 评分 >= min_score_threshold       │
│     - 按评分排序                        │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  2. 评估现有持仓                        │
│     For each 持仓股票:                  │
│       ├─ 不在新选股列表? → 标记卖出     │
│       ├─ 评分 < min_score_for_hold? → 卖│
│       └─ 持仓 > max_holding_days? → 卖  │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  3. 卖出决策执行                        │
│     - 卖出所有标记的股票                │
│     - 理由: "rebalance_exit"           │
│     - 计算卖出所得                      │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  4. 计算可用资金                        │
│     total_available = 现金 + 卖出所得   │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  5. 买入新股票                          │
│     - 选择top N只高评分股票             │
│     - N = max_positions - 保留持仓数    │
│     - 均匀分配资金                      │
│     - 执行买入                          │
└─────────────────────────────────────────┘
```

### 非调仓日逻辑 (每天)

```
┌─────────────────────────────────────────┐
│         日常监控 (每个交易日)           │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  1. 更新持仓市值                        │
│     - 获取当前价格                      │
│     - 计算组合价值                      │
│     - 记录日收益率                      │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  2. 止损检查 (优先级最高)               │
│     For each 持仓:                      │
│       if (current_price - avg_cost) / avg_cost < -8%:
│         卖出, 理由: "stop_loss"         │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  3. 🆕 止盈检查 (新增)                  │
│     For each 持仓:                      │
│       if (current_price - avg_cost) / avg_cost > +15%:
│         卖出, 理由: "take_profit"       │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  4. 🆕 持仓期限检查 (新增)              │
│     For each 持仓:                      │
│       if holding_days > max_holding_days:
│         卖出, 理由: "max_holding"       │
└─────────────────────────────────────────┘
```

---

## 🔧 实现细节

### 1. 改造 `_execute_rebalance`

**当前问题**:
```python
def _execute_rebalance(self, date: str, selected_stocks: List[Dict]):
    total_available = self.current_capital  # ❌ 只用现金
    # ❌ 不卖出旧持仓
    # 只买入新股票
```

**优化方案**:
```python
def _execute_rebalance(self, date: str, selected_stocks: List[Dict], adapter: MLModelAdapter):
    # 1. 评估现有持仓，决定卖出
    if self.enable_rebalance_sell:
        self._rebalance_sell_positions(date, selected_stocks, adapter)

    # 2. 计算可用资金 (现金 + 卖出所得)
    total_available = self.current_capital

    # 3. 买入新股票
    self._rebalance_buy_stocks(date, selected_stocks, total_available)
```

### 2. 新增 `_rebalance_sell_positions`

```python
def _rebalance_sell_positions(self, date: str, selected_stocks: List[Dict],
                              adapter: MLModelAdapter):
    """调仓时卖出不符合条件的持仓"""

    selected_codes = {stock['stock_code'] for stock in selected_stocks}
    selected_scores = {stock['stock_code']: stock['score'] for stock in selected_stocks}

    positions_to_sell = []

    for stock_code, position in self.positions.items():
        if position['shares'] <= 0:
            continue

        should_sell, reason = self._should_sell_position(
            stock_code, position, date, selected_codes, selected_scores
        )

        if should_sell:
            positions_to_sell.append((stock_code, reason))

    # 执行卖出
    for stock_code, reason in positions_to_sell:
        self._execute_sell(stock_code, date, reason)
```

### 3. 新增 `_should_sell_position`

```python
def _should_sell_position(self, stock_code: str, position: Dict, date: str,
                         selected_codes: set, selected_scores: Dict) -> Tuple[bool, str]:
    """判断持仓是否应该卖出"""

    # 检查1: 不在新选股列表
    if stock_code not in selected_codes:
        return True, "rebalance_not_selected"

    # 检查2: 评分低于持有阈值
    if stock_code in selected_scores:
        if selected_scores[stock_code] < self.min_score_for_hold:
            return True, "rebalance_low_score"

    # 检查3: 持仓时间过长
    holding_days = self._calculate_holding_days(position['entry_date'], date)
    if holding_days > self.max_holding_days:
        return True, "rebalance_max_holding"

    return False, ""
```

### 4. 新增 `_check_take_profit`

```python
def _check_take_profit(self, date: str):
    """检查止盈"""
    positions_to_sell = []

    for stock_code, position in self.positions.items():
        if position['shares'] > 0:
            current_price = self._get_stock_price(stock_code, date)
            if current_price and current_price > 0:
                cost_basis = position['avg_cost']
                profit_pct = (current_price - cost_basis) / cost_basis

                if profit_pct > self.take_profit_pct:
                    positions_to_sell.append(stock_code)

    for stock_code in positions_to_sell:
        self._execute_sell(stock_code, date, "take_profit")
```

### 5. 新增 `_check_holding_period`

```python
def _check_holding_period(self, date: str):
    """检查持仓期限"""
    positions_to_sell = []

    for stock_code, position in self.positions.items():
        if position['shares'] > 0:
            holding_days = self._calculate_holding_days(position['entry_date'], date)

            if holding_days > self.max_holding_days:
                positions_to_sell.append(stock_code)

    for stock_code in positions_to_sell:
        self._execute_sell(stock_code, date, "max_holding")

def _calculate_holding_days(self, entry_date: str, current_date: str) -> int:
    """计算持仓天数"""
    entry = datetime.strptime(entry_date, '%Y-%m-%d')
    current = datetime.strptime(current_date, '%Y-%m-%d')
    return (current - entry).days
```

### 6. 修改 `_update_portfolio_value`

```python
def _update_portfolio_value(self, date: str):
    """更新组合价值"""
    # 原有逻辑...

    # 风控检查 (优先级顺序很重要)
    self._check_stop_loss(date)        # 优先级1: 止损
    self._check_take_profit(date)      # 优先级2: 止盈
    self._check_holding_period(date)   # 优先级3: 超期
```

---

## 📈 预期改进效果

### 原策略 vs 新策略对比

| 指标 | 原策略 | 新策略 | 改进 |
|------|--------|--------|------|
| **交易特性** | 买入持有+止损 | 主动调仓 | ✅ |
| **卖出原因** | 100%止损 | 止损/止盈/调仓/超期 | ✅ |
| **资金利用** | 第1次后闲置 | 持续循环使用 | ✅ |
| **预期胜率** | 0% | 30-50% | ✅ |
| **收益特性** | 被动 | 主动优化 | ✅ |

### 卖出原因分布 (预期)

```
原策略:
├─ stop_loss: 100% (26/26)
└─ 其他: 0%

新策略 (预期):
├─ take_profit: 30-40%        ← 盈利了结
├─ rebalance_not_selected: 20-30%  ← 调仓退出
├─ stop_loss: 20-30%          ← 止损
├─ max_holding: 10-20%        ← 超期轮换
└─ rebalance_low_score: 0-10% ← 降级退出
```

---

## 🧪 测试计划

1. **单元测试**:
   - 测试 `_should_sell_position` 各种场景
   - 测试 `_check_take_profit` 触发条件
   - 测试 `_check_holding_period` 计算准确性

2. **集成测试**:
   - 模拟5天调仓周期
   - 验证卖出→买入资金流转
   - 验证持仓数量不超过max_positions

3. **回测对比**:
   - 重新运行V3.7 vs V3.81回测
   - 对比原策略 vs 新策略
   - 分析胜率、收益率、夏普比率变化

---

## 🎛️ 参数调优建议

### 保守型配置 (低风险)
```python
take_profit_pct = 0.20      # 止盈20%
stop_loss_pct = 0.05        # 止损5%
max_holding_days = 30       # 最长30天
min_score_for_hold = 80.0   # 持有分数80+
```

### 激进型配置 (高频交易)
```python
take_profit_pct = 0.10      # 止盈10%
stop_loss_pct = 0.08        # 止损8%
max_holding_days = 10       # 最长10天
min_score_for_hold = 75.0   # 持有分数75+
```

### 默认配置 (平衡型)
```python
take_profit_pct = 0.15      # 止盈15%
stop_loss_pct = 0.08        # 止损8%
max_holding_days = 20       # 最长20天
min_score_for_hold = 75.0   # 持有分数75+
```

---

## ✅ 实现检查清单

- [ ] 添加新参数到 `__init__`
- [ ] 实现 `_rebalance_sell_positions`
- [ ] 实现 `_should_sell_position`
- [ ] 实现 `_check_take_profit`
- [ ] 实现 `_check_holding_period`
- [ ] 实现 `_calculate_holding_days`
- [ ] 修改 `_execute_rebalance` (添加adapter参数)
- [ ] 修改 `_update_portfolio_value` (添加新检查)
- [ ] 修改 `_run_single_version_backtest` (传递adapter)
- [ ] 更新日志输出
- [ ] 单元测试
- [ ] 集成测试
- [ ] 完整回测验证

---

**设计者**: Claude Code
**复审**: 待用户确认
**实施时间**: 2025-10-12
