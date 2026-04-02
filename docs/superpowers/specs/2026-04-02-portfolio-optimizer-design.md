# Portfolio Optimizer: 自适应价格锚定 + 动态止损止盈 + 风险预算仓位

**日期**: 2026-04-02
**目标**: 替代现有固定百分比价格/仓位逻辑，实现对沪深300和中证2000的超额收益
**验证**: A/B对比回测，日级OHLC止损/止盈模拟

---

## 1. 问题总结

现有 `_enhance_prices_with_ml()` 的五个问题:

| 字段 | 现状 | 问题 |
|------|------|------|
| 买入价 | pred≥0→折扣0%, pred<0→0.5~2.5% | 几乎等于收盘价，无技术面锚点 |
| 止损价 | 主板-10%, 创科-15% | 一刀切，银行股太松小盘股太紧 |
| 目标价 | 主板+8%, 创科+12% | 固定百分比，无阻力位参考 |
| 仓位 | 强烈买入15%/买入8%/谨慎3%/观望1% | 离散分档，不考虑波动率，不与市场环境联动 |
| 持仓期 | 固定15日 | 不响应止损/止盈信号，白白承受回撤或错过锁利 |

## 2. 设计总览

新建独立模块 `portfolio_optimizer.py`，包含6个子系统:

```
┌─────────────────────────────────────────────────┐
│              portfolio_optimizer.py               │
│                                                   │
│  ┌──────────────┐  ┌──────────────┐              │
│  │ AdaptiveEntry │  │ AdaptiveStop │              │
│  │  (买入价)      │  │  (止损价)     │              │
│  └──────┬───────┘  └──────┬───────┘              │
│         │                  │                      │
│  ┌──────┴──────────────────┴───────┐             │
│  │       AdaptiveTarget             │             │
│  │        (目标价)                   │             │
│  └──────────────┬──────────────────┘             │
│                  │                                │
│  ┌──────────────┴──────────────────┐             │
│  │     RiskBudgetAllocator          │             │
│  │  (信号强度×波动率倒数仓位)        │             │
│  └──────────────┬──────────────────┘             │
│                  │                                │
│  ┌──────────────┴──────────────────┐             │
│  │    SignalStrengthFilter          │             │
│  │  (动态cutoff, 非固定Top N)       │             │
│  └─────────────────────────────────┘             │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│     backtest/dynamic_sl_tp_engine.py             │
│                                                   │
│  日级OHLC止损/止盈模拟 + 移动止盈(trailing stop)  │
│  + A/B对比回测框架                                │
└─────────────────────────────────────────────────┘
```

## 3. 模块详细设计

### 3.1 AdaptiveEntry — 自适应买入价

**输入**: close, atr_20d, support_level, pred_10d
**输出**: buy_price

```python
def compute_entry_price(close, atr_20d, support_level, pred_10d):
    # 因子1: ATR折扣 — 波动大的股票等更深的回调
    atr_ratio = atr_20d / close
    atr_discount = atr_ratio * 0.5  # ATR的一半作为折扣

    # 因子2: 支撑距离折扣 — 离支撑远的等回踩
    support_gap = max(0, (close - support_level) / close)
    support_discount = support_gap * 0.3

    # 因子3: ML信号调节 — 强看涨少折扣追着买
    if pred_10d > 0.005:
        ml_mult = 0.5   # 强看涨: 折扣减半
    elif pred_10d < -0.005:
        ml_mult = 1.5   # 看跌: 折扣加大
    else:
        ml_mult = 1.0   # 中性

    adaptive_discount = (atr_discount + support_discount) * ml_mult
    adaptive_discount = clip(adaptive_discount, 0.0, 0.03)  # 0~3%

    return round(close * (1 - adaptive_discount), 2)
```

**支撑位计算**:
```python
def compute_support(df_20d):
    rolling_low = df_20d['low'].rolling(20).min().iloc[-1]
    ma20 = df_20d['close'].rolling(20).mean().iloc[-1]
    ma60 = df_20d['close'].rolling(60).mean().iloc[-1]
    # 取rolling low和均线中离收盘价较近的作为支撑
    candidates = [rolling_low, ma20, ma60]
    close = df_20d['close'].iloc[-1]
    valid = [c for c in candidates if c < close * 0.995]
    return max(valid) if valid else close * 0.97
```

### 3.2 AdaptiveStop — 自适应止损价

**输入**: buy_price, close, atr_20d, support_level, env_score, board_type
**输出**: stop_price

```python
def compute_stop_price(buy_price, close, atr_20d, support_level, env_score, is_wide_limit):
    individual_atr_pct = atr_20d / close

    # 基础止损 = 2倍ATR (覆盖~95%正常日内波动)
    base_stop_pct = individual_atr_pct * 2.0

    # 市场环境调节
    if env_score >= 60:
        env_mult = 0.85   # 偏多: 止损收紧
    elif env_score >= 40:
        env_mult = 1.0    # 中性
    else:
        env_mult = 1.2    # 偏空: 止损放宽

    # 板块硬限制
    min_stop = 0.03
    max_stop = 0.15 if is_wide_limit else 0.10

    stop_pct = clip(base_stop_pct * env_mult, min_stop, max_stop)
    stop_price = buy_price * (1 - stop_pct)

    # 支撑位保护: 支撑位在止损区间内 → 止损移到支撑下方0.5%
    if support_level > stop_price and support_level < buy_price * 0.99:
        stop_price = support_level * 0.995

    return round(stop_price, 2)
```

### 3.3 AdaptiveTarget — 自适应目标价

**输入**: buy_price, stop_price, close, resistance, pred_10d
**输出**: target_price

```python
def compute_target_price(buy_price, stop_price, close, resistance, pred_10d):
    risk = buy_price - stop_price

    # 1. 技术目标: 阻力位下方2%
    tech_target = resistance * 0.98 if resistance > close * 1.01 else None

    # 2. ML目标: 预测收益(校准后)
    ml_target = close * (1 + pred_10d) if pred_10d > 0 else None

    # 3. 风险收益比下限: 至少2:1
    min_rr_target = buy_price + risk * 2.0

    # 4. 中位数选择
    candidates = [c for c in [tech_target, ml_target, min_rr_target] if c is not None]
    if len(candidates) >= 2:
        candidates.sort()
        target = candidates[len(candidates) // 2]  # 中位数
    elif candidates:
        target = candidates[0]
    else:
        target = buy_price + risk * 2.0

    # 硬约束: 3%~15%涨幅
    target = clip(target, buy_price * 1.03, buy_price * 1.15)

    return round(target, 2)
```

**阻力位计算**:
```python
def compute_resistance(df_20d):
    rolling_high = df_20d['high'].rolling(20).max().iloc[-1]
    ma20 = df_20d['close'].rolling(20).mean().iloc[-1]
    ma60 = df_20d['close'].rolling(60).mean().iloc[-1]
    close = df_20d['close'].iloc[-1]
    candidates = [rolling_high, ma20, ma60]
    valid = [c for c in candidates if c > close * 1.005]
    return min(valid) if valid else close * 1.08
```

### 3.4 SignalStrengthFilter — 动态持仓数量

**现有**: 固定Top 10。
**新设计**: 基于信号强度cutoff，动态决定持仓数量。

```python
def filter_by_signal_strength(stocks_sorted_by_composite, env_score):
    """
    动态cutoff规则:
    - composite > 0: 信号为正的全部入选 (ML认为有正期望)
    - 数量约束: 最少3只(避免过度集中), 最多20只(避免过度分散)
    - 环境恶劣时(env_score<30)进一步收紧: 最多5只
    """
    positive_signal = [s for s in stocks_sorted_by_composite if s['composite'] > 0]

    if env_score < 30:
        max_n = 5
    elif env_score < 50:
        max_n = 12
    else:
        max_n = 20

    min_n = 3
    n = clip(len(positive_signal), min_n, max_n)

    # 如果正信号不足min_n, 补足到min_n (但标记为低信心)
    selected = stocks_sorted_by_composite[:n]
    return selected
```

### 3.5 RiskBudgetAllocator — 风险预算仓位

**输入**: selected_stocks (含composite_rank_pct, atr_20d, close), env_score
**输出**: 每只股票的position_pct

```python
def allocate_positions(selected_stocks, env_score):
    # Step 1: 总仓位 — 由市场环境评分决定
    if env_score >= 80:
        total_exposure = 0.90
    elif env_score >= 60:
        total_exposure = 0.65
    elif env_score >= 40:
        total_exposure = 0.40
    elif env_score >= 20:
        total_exposure = 0.20
    else:
        total_exposure = 0.05

    # Step 2: 个股原始权重 = 信号强度 × 波动率倒数
    for s in selected_stocks:
        atr_pct = max(s['atr_20d'] / s['close'], 0.01)
        signal = s['composite_rank_pct']  # 0~1, 1=最强
        s['raw_weight'] = signal * (1.0 / atr_pct)

    # Step 3: 归一化
    total_raw = sum(s['raw_weight'] for s in selected_stocks)
    for s in selected_stocks:
        s['norm_weight'] = s['raw_weight'] / total_raw if total_raw > 0 else 1/len(selected_stocks)

    # Step 4: 个股仓位 = 归一化权重 × 总仓位
    for s in selected_stocks:
        raw_pct = s['norm_weight'] * total_exposure * 100  # 转百分比
        s['position_pct'] = round(clip(raw_pct, 1.0, 15.0), 1)

    # Step 5: 等风险贡献校验 — 削减风险贡献过高的
    avg_risk = np.mean([s['position_pct'] * s['atr_20d'] / s['close'] for s in selected_stocks])
    for s in selected_stocks:
        rc = s['position_pct'] * s['atr_20d'] / s['close']
        if rc > avg_risk * 2.0:
            s['position_pct'] = round(s['position_pct'] * (avg_risk * 2.0 / rc), 1)

    # Step 6: 总仓位校验 — 不超过总仓位限制
    actual_total = sum(s['position_pct'] for s in selected_stocks)
    if actual_total > total_exposure * 100:
        scale = total_exposure * 100 / actual_total
        for s in selected_stocks:
            s['position_pct'] = round(s['position_pct'] * scale, 1)

    return selected_stocks
```

### 3.6 DynamicSLTPEngine — 日级止损/止盈回测引擎

**核心**: 替代固定持仓期，逐日判断止损/止盈/移动止盈触发。

```python
def simulate_trade(buy_date, buy_price, stop_price, target_price, 
                   daily_ohlc, max_hold_days=15):
    """
    逐日模拟一笔交易，返回 (exit_date, exit_price, exit_reason)
    
    exit_reason: 'stop_loss' | 'take_profit' | 'trailing_stop' | 'max_hold'
    """
    current_stop = stop_price
    highest_since_entry = buy_price

    for day_idx in range(max_hold_days):
        day = daily_ohlc[day_idx]  # {open, high, low, close}

        # T+1: 买入当天不能卖
        if day_idx == 0:
            highest_since_entry = max(highest_since_entry, day['high'])
            continue

        # 止损检查 (优先于止盈 — 日级别保守假设)
        if day['low'] <= current_stop:
            return day['date'], current_stop, 'stop_loss'

        # 止盈检查
        if day['high'] >= target_price:
            return day['date'], target_price, 'take_profit'

        # 更新最高价
        highest_since_entry = max(highest_since_entry, day['high'])

        # 移动止盈 (trailing stop):
        # 浮盈超过目标的60% → 止损上移到成本价(保本)
        profit_threshold = buy_price + (target_price - buy_price) * 0.6
        if highest_since_entry >= profit_threshold:
            # 保本线
            new_stop = max(current_stop, buy_price)
            # 进一步: 浮盈每增加1个ATR, 止损跟进0.5个ATR
            # (ATR需从外部传入或预计算)
            current_stop = new_stop

        # 更激进的trailing: 最高价回撤2倍ATR则出场
        trailing_stop = highest_since_entry - (target_price - buy_price) * 0.4
        current_stop = max(current_stop, trailing_stop)

    # 到期: 按最后一天收盘价卖出
    last_day = daily_ohlc[min(max_hold_days-1, len(daily_ohlc)-1)]
    return last_day['date'], last_day['close'], 'max_hold'
```

### 3.7 A/B对比回测

新文件 `backtest/ab_compare.py`:

```python
def run_ab_comparison(report_dir, start_date, end_date, benchmark='000905.SH'):
    """
    A方案: 现有逻辑 (固定止损止盈, 固定15日持仓, 离散仓位)
    B方案: 新portfolio_optimizer (自适应价格, 动态SLTP, 风险预算)

    对比指标:
    - 绝对年化收益, Sharpe, MaxDD, Calmar
    - 对沪深300(000300.SH)超额收益
    - 对中证2000(932000.CSI)超额收益
    - 月度胜率, 日均换手率
    - 止损触发率, 止盈触发率, 到期卖出率
    """
```

输出示例:
```
| 指标 | A (现有) | B (新) | 差异 |
|------|----------|--------|------|
| 年化收益 | 124.5% | ? | ? |
| 对300超额 | +xx% | ? | ? |
| 对2000超额 | +xx% | ? | ? |
| Sharpe | 3.14 | ? | ? |
| MaxDD | -6.4% | ? | ? |
| 止损触发率 | N/A | xx% | - |
| 止盈触发率 | N/A | xx% | - |
| 到期卖出率 | 100% | xx% | - |
```

## 4. 数据依赖

所有数据均已在数据库中可用:

| 数据 | 来源 | 说明 |
|------|------|------|
| OHLC | daily_quotes | open/high/low/close, 回测用 |
| ATR(20) | technical_indicators 或实时计算 | 20日ATR |
| MA20/MA60 | technical_indicators | 均线支撑/阻力 |
| pred_10d | ML scorer输出 | 10日预测收益 |
| composite | ML scorer输出 | 综合评分 |
| env_score | 市场环境评分模块 | 当日环境分 |

## 5. 集成方式

### tomorrow_stock_selector.py 改动

```python
# 在 _enhance_prices_with_ml() 中:
from portfolio_optimizer import PortfolioOptimizer

optimizer = PortfolioOptimizer(env_score=self.env_score)

# 替代原有价格计算
for stock in selected_stocks:
    stock = optimizer.compute_prices(stock, df_recent)
    # compute_prices 内部调用:
    # - compute_entry_price
    # - compute_stop_price
    # - compute_target_price

# 替代原有仓位计算
selected_stocks = optimizer.filter_and_allocate(selected_stocks)
# filter_and_allocate 内部调用:
# - filter_by_signal_strength
# - allocate_positions
```

### 报告格式变化

表头新增列:
```
| 排名 | 股票代码 | 名称 | Composite | 建议 | 收盘价 | 买入价 | 止损价 | 目标价 | 仓位 | 止损% | R:R | ATR% |
```

新增: 止损%(个股实际止损幅度), R:R(风险收益比), ATR%(标准化波动率)

## 6. 文件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `portfolio_optimizer.py` | 新建 | 核心模块: 价格锚定+止损止盈+仓位优化 |
| `backtest/dynamic_sl_tp_engine.py` | 新建 | 日级OHLC止损/止盈回测引擎 |
| `backtest/ab_compare.py` | 新建 | A/B对比回测脚本 |
| `tomorrow_stock_selector.py` | 修改 | 调用portfolio_optimizer替代原有逻辑 |

## 7. 验证标准

改进方案**必须同时满足**:
1. 对沪深300年化超额 > 现有方案
2. 对中证2000年化超额 > 现有方案
3. MaxDD 不恶化超过2个百分点
4. Sharpe ≥ 现有方案的90%

如果不满足，回退到现有逻辑，不会破坏生产。

## 8. 风险与回退

- **portfolio_optimizer.py 完全独立**，不修改原有价格计算函数，通过开关切换
- 回测失败 → 保留原有逻辑，新模块作为实验代码
- `tomorrow_stock_selector.py` 通过参数 `--optimizer v2` 启用新逻辑，默认仍走旧路径
