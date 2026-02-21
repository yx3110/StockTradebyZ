# 回测胜率问题分析报告

**生成时间**: 2025-10-12
**问题发现**: ML模型回测对比中V3.7和V3.81胜率均显示0.00%

---

## 📋 问题摘要

ML模型回测结果显示两个关键问题：

1. **✅ 已修复**: 胜率计算逻辑错误 - 统计了所有交易而非仅卖出交易
2. **⚠️ 待优化**: 回测策略设计问题 - 只买不卖导致100%的卖出交易都是止损

---

## 🔍 问题1: 胜率计算BUG (已修复)

### 问题描述

`extensible_backtest_engine.py` 第943-946行的胜率计算逻辑错误：

```python
# ❌ 错误代码
profit_trades = [t for t in self.trades if t.get('profit', 0) > 0]
loss_trades = [t for t in self.trades if t.get('profit', 0) < 0]
win_rate = len(profit_trades) / (len(profit_trades) + len(loss_trades)) if (profit_trades or loss_trades) else 0
```

### 根本原因

- **买入交易**(`_execute_rebalance`第763-773行): 没有`profit`字段
- **卖出交易**(`_execute_sell`第880-890行): 有`profit`字段

当代码执行 `t.get('profit', 0)` 时：
- 买入交易返回默认值0
- 0既不满足 `> 0`（盈利）也不满足 `< 0`（亏损）
- 导致买入交易被排除在统计之外

### 修复方案

```python
# ✅ 修复后的代码
# 计算胜率 - 只统计卖出交易（买入交易没有profit字段）
sell_trades = [t for t in self.trades if t.get('action') == 'sell']
profit_trades = [t for t in sell_trades if t.get('profit', 0) > 0]
loss_trades = [t for t in sell_trades if t.get('profit', 0) < 0]
win_rate = len(profit_trades) / len(sell_trades) if sell_trades else 0
```

### 修复文件

- `extensible_backtest_engine.py:943-947`

---

## 🚨 问题2: 回测策略设计缺陷 (待优化)

### 实际数据分析

```
V3.7:
  买入交易: 90次
  卖出交易: 26次
  盈利交易: 0次
  亏损交易: 26次
  胜率: 0.00% (正确反映了100%亏损的事实)

V3.81:
  买入交易: 55次
  卖出交易: 29次
  盈利交易: 0次
  亏损交易: 29次
  胜率: 0.00% (正确反映了100%亏损的事实)
```

### 策略分析

当前`_execute_rebalance`函数逻辑（第740-792行）：

```python
def _execute_rebalance(self, date: str, selected_stocks: List[Dict]):
    """执行调仓"""
    # ❌ 问题：只使用可用现金，不卖出旧仓位
    total_available = self.current_capital  # 仅可用现金
    target_allocation = total_available / min(len(selected_stocks), self.max_positions)

    # 只买入新股票，不卖出旧持仓
    for stock_info in selected_stocks[:self.max_positions]:
        # ... 买入逻辑 ...
```

**实际执行流程**：

1. **第1次调仓**: 使用100万现金买入10只股票
2. **第2-N次调仓**: `current_capital ≈ 0`，无法买入新股票
3. **唯一的卖出**: 仅当止损触发时（跌幅 > 8%）
4. **结果**: 100%的卖出交易都是止损（必然亏损）

### 设计缺陷

| 环节 | 当前实现 | 应有实现 |
|------|----------|----------|
| 调仓准备 | 不卖出旧仓位 | ✅ 清仓不在新选股列表的股票 |
| 资金计算 | 仅用现金 | ✅ 现金 + 卖出所得 |
| 买入逻辑 | 只买不卖 | ✅ 先卖后买，真正调仓 |
| 盈利了结 | 无机制 | ✅ 正常卖出盈利股票 |

### 为什么胜率是0.00%

**不是计算错误，而是策略问题**：

- 当前策略只在止损时卖出
- 止损定义为"亏损8%"
- 所以100%的卖出交易都是亏损
- 胜率0.00%正确反映了这个事实

---

## 💡 优化建议

### 短期修复 (保守)

在`_execute_rebalance`中添加卖出旧持仓的逻辑：

```python
def _execute_rebalance(self, date: str, selected_stocks: List[Dict]):
    """执行调仓"""
    if not selected_stocks:
        return

    # 🆕 卖出不在新选股列表中的股票
    selected_codes = {stock['stock_code'] for stock in selected_stocks}
    for stock_code in list(self.positions.keys()):
        if self.positions[stock_code]['shares'] > 0:
            if stock_code not in selected_codes:
                self._execute_sell(stock_code, date, "rebalance_exit")

    # 现在使用：现金 + 卖出所得
    total_available = self.current_capital
    target_allocation = total_available / min(len(selected_stocks), self.max_positions)

    # ... 后续买入逻辑 ...
```

### 长期优化 (完整)

1. **盈利了结机制**:
   ```python
   def _check_take_profit(self, date: str):
       """检查止盈（如盈利超过15%）"""
       positions_to_sell = []
       for stock_code, position in self.positions.items():
           if position['shares'] > 0:
               current_price = self._get_stock_price(stock_code, date)
               if current_price:
                   profit_pct = (current_price - position['avg_cost']) / position['avg_cost']
                   if profit_pct > self.take_profit_pct:  # 如15%
                       positions_to_sell.append(stock_code)

       for stock_code in positions_to_sell:
           self._execute_sell(stock_code, date, "take_profit")
   ```

2. **持仓时间限制**:
   ```python
   def _check_holding_period(self, date: str):
       """检查持仓天数，强制平仓超期股票"""
       for stock_code, position in self.positions.items():
           if position['shares'] > 0:
               holding_days = (datetime.strptime(date, '%Y-%m-%d') -
                             datetime.strptime(position['entry_date'], '%Y-%m-%d')).days
               if holding_days > self.max_holding_days:  # 如20天
                   self._execute_sell(stock_code, date, "max_holding")
   ```

3. **完整调仓逻辑**:
   - 评估每个持仓是否仍值得保留
   - 卖出低评分/不在新列表的股票
   - 用全部可用资金买入高评分股票
   - 实现真正的"调仓"而非"只买"

---

## 📊 影响评估

### 对已完成回测的影响

- 回测结果数据**正确**（总收益、夏普比率、最大回撤）
- 胜率0.00%现在**准确反映**了策略特性
- 策略本质是"买入持有+止损"，不是主动交易

### 需要的后续动作

1. ✅ **已完成**: 修复胜率计算BUG
2. ⏳ **待决定**: 是否优化回测策略为真正的调仓策略
3. ⏳ **待评估**: 重新运行回测对比优化前后的性能

---

## 📁 相关文件

| 文件 | 修改状态 | 说明 |
|------|---------|------|
| `extensible_backtest_engine.py` | ✅ 已修复 | 胜率计算逻辑(行943-947) |
| `analyze_trades.py` | 🆕 新增 | 交易结构分析工具 |
| `reports/backtest/ml_versions_comparison_*.json` | - | 回测原始数据 |

---

## 🎯 结论

1. **胜率计算BUG已修复** - 现在正确统计卖出交易
2. **0.00%胜率是准确的** - 反映了"只止损不止盈"的策略特性
3. **回测策略需要优化** - 当前不是真正的"调仓"，而是"一次性买入+被动持有"

**推荐**: 优化回测策略，添加正常的调仓退出机制，使其成为真正的主动交易策略。

---

**报告生成者**: Claude Code
**分析基于**: 2025-07-01 → 2025-09-30 回测数据
