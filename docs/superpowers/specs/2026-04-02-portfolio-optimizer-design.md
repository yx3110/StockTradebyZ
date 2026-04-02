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

## 3. 超参数校准策略

**核心原则**: 所有超参数通过回测网格搜索确定，不拍脑袋。

### 3.0 待校准参数总表

下表列出所有超参数、搜索范围和校准方法。实施时先用默认值跑通流程，再用网格搜索在回测中找最优值。

| 模块 | 参数 | 搜索范围 | 默认初始值 | 校准指标 |
|------|------|----------|-----------|----------|
| **Entry** | `atr_discount_mult` | [0.1, 0.3, 0.5, 0.7, 1.0] | 0.5 | 成交率×期望收益 |
| **Entry** | `support_discount_mult` | [0.1, 0.2, 0.3, 0.5] | 0.3 | 成交率×期望收益 |
| **Entry** | `ml_bullish_threshold` | [0.002, 0.005, 0.01, 0.02] | 0.005 | 成交率×期望收益 |
| **Entry** | `ml_bullish_mult` | [0.3, 0.5, 0.7] | 0.5 | 成交率×期望收益 |
| **Entry** | `ml_bearish_mult` | [1.2, 1.5, 2.0] | 1.5 | 成交率×期望收益 |
| **Entry** | `max_discount` | [0.02, 0.03, 0.05] | 0.03 | 成交率×期望收益 |
| **Stop** | `atr_multiplier` | [1.0, 1.5, 2.0, 2.5, 3.0] | 2.0 | 止损触发率×持仓期收益 |
| **Stop** | `env_mult_bullish` | [0.7, 0.85, 1.0] | 0.85 | Sharpe |
| **Stop** | `env_mult_bearish` | [1.0, 1.2, 1.5] | 1.2 | MaxDD |
| **Stop** | `min_stop_pct` | [0.02, 0.03, 0.04] | 0.03 | 噪声扫出率 |
| **Stop** | `max_stop_main` | [0.08, 0.10, 0.12] | 0.10 | 尾部损失 |
| **Stop** | `max_stop_wide` | [0.12, 0.15, 0.18] | 0.15 | 尾部损失 |
| **Target** | `min_rr_ratio` | [1.5, 2.0, 2.5, 3.0] | 2.0 | 年化收益 |
| **Target** | `target_clip_min` | [0.02, 0.03, 0.05] | 0.03 | 止盈触发率 |
| **Target** | `target_clip_max` | [0.10, 0.15, 0.20] | 0.15 | 止盈触发率 |
| **Filter** | `composite_cutoff` | [0, 0.0001, 0.0005, 0.001] | 0 | 超额收益 |
| **Filter** | `min_n` | [3, 5, 8] | 3 | 分散度 |
| **Filter** | `max_n_bull` | [10, 15, 20, 30] | 20 | 超额收益 |
| **Filter** | `max_n_bear` | [3, 5, 8] | 5 | MaxDD |
| **Trailing** | `trailing_trigger_pct` | [0.3, 0.5, 0.6, 0.8] | 0.6 | 盈利交易平均收益 |
| **Trailing** | `trailing_fallback_pct` | [0.2, 0.3, 0.4, 0.5] | 0.4 | 盈利交易平均收益 |
| **Hold** | `max_hold_days` | [5, 10, 15, 20, 30] | 15 | 年化收益/换手率 |

**校准方法**: 
1. **两阶段搜索**: 先粗搜(每参数3-5个值)，锁定方向，再细搜(±20%范围)
2. **分组校准**: Entry参数一组、Stop参数一组、Target参数一组、Filter参数一组，组内联合搜索，组间顺序校准
3. **校准集**: 用2022-01-01~2025-06-30回测(IS)，2025-07-01~2026-03-31验证(OOS)
4. **校准指标**: 各组有不同的主要优化指标(见上表)，全局用 Sharpe×(1 - MaxDD/100) 作为综合目标
5. **过拟合防护**: OOS表现 < IS的60% → 判定过拟合，回退到IS中位数参数

### 3.0.1 校准脚本设计

```python
# scripts/calibrate_optimizer_params.py
def calibrate(param_group, search_grid, report_dir, start_date, end_date):
    """
    对一组参数做网格搜索, 返回最优参数组合
    
    流程:
    1. 对grid中每个参数组合, 用dynamic_sl_tp_engine回测
    2. 计算 Sharpe × (1 - MaxDD/100) 综合目标
    3. 按综合目标排序, 取Top 3
    4. 对Top 3在OOS期验证, 取OOS最优
    5. 输出: best_params.json + 搜索过程log
    """

# 校准顺序:
# Phase 1: Stop参数 (止损是风控底线, 先定)
# Phase 2: Target参数 (目标价依赖止损)
# Phase 3: Entry参数 (买入价依赖止损目标)
# Phase 4: Filter + 仓位参数 (组合层面)
# Phase 5: Trailing参数 (精调)
# Phase 6: 全局联合微调 (Phase1-5最优值±1格)
```

## 4. 模块详细设计

### 4.1 AdaptiveEntry — 自适应买入价

**输入**: close, atr_20d, support_level, pred_10d, params
**输出**: buy_price

所有数值系数来自 `params` (由校准脚本输出), 代码中不硬编码。

```python
def compute_entry_price(close, atr_20d, support_level, pred_10d, params):
    # 因子1: ATR折扣 — 波动大的股票等更深的回调
    atr_ratio = atr_20d / close
    atr_discount = atr_ratio * params['atr_discount_mult']

    # 因子2: 支撑距离折扣 — 离支撑远的等回踩
    support_gap = max(0, (close - support_level) / close)
    support_discount = support_gap * params['support_discount_mult']

    # 因子3: ML信号调节 — 强看涨少折扣追着买
    if pred_10d > params['ml_bullish_threshold']:
        ml_mult = params['ml_bullish_mult']
    elif pred_10d < -params['ml_bullish_threshold']:
        ml_mult = params['ml_bearish_mult']
    else:
        ml_mult = 1.0

    adaptive_discount = (atr_discount + support_discount) * ml_mult
    adaptive_discount = clip(adaptive_discount, 0.0, params['max_discount'])

    return round(close * (1 - adaptive_discount), 2)
```

**支撑位计算** (无超参数, 纯技术面计算):
```python
def compute_support(df_60d):
    rolling_low = df_60d['low'].rolling(20).min().iloc[-1]
    ma20 = df_60d['close'].rolling(20).mean().iloc[-1]
    ma60 = df_60d['close'].rolling(60).mean().iloc[-1]
    close = df_60d['close'].iloc[-1]
    candidates = [rolling_low, ma20, ma60]
    valid = [c for c in candidates if c < close * 0.995]
    return max(valid) if valid else close * 0.97
```

### 4.2 AdaptiveStop — 自适应止损价

**输入**: buy_price, close, atr_20d, support_level, env_score, is_wide_limit, params
**输出**: stop_price

```python
def compute_stop_price(buy_price, close, atr_20d, support_level, env_score, is_wide_limit, params):
    individual_atr_pct = atr_20d / close

    # 基础止损 = N倍ATR
    base_stop_pct = individual_atr_pct * params['atr_multiplier']

    # 市场环境调节
    if env_score >= 60:
        env_mult = params['env_mult_bullish']
    elif env_score >= 40:
        env_mult = 1.0
    else:
        env_mult = params['env_mult_bearish']

    # 板块硬限制
    min_stop = params['min_stop_pct']
    max_stop = params['max_stop_wide'] if is_wide_limit else params['max_stop_main']

    stop_pct = clip(base_stop_pct * env_mult, min_stop, max_stop)
    stop_price = buy_price * (1 - stop_pct)

    # 支撑位保护: 支撑位在止损区间内 → 止损移到支撑下方0.5%
    if support_level > stop_price and support_level < buy_price * 0.99:
        stop_price = support_level * 0.995

    return round(stop_price, 2)
```

### 4.3 AdaptiveTarget — 自适应目标价

**输入**: buy_price, stop_price, close, resistance, pred_10d, params
**输出**: target_price

```python
def compute_target_price(buy_price, stop_price, close, resistance, pred_10d, params):
    risk = buy_price - stop_price

    # 1. 技术目标: 阻力位下方2%
    tech_target = resistance * 0.98 if resistance > close * 1.01 else None

    # 2. ML目标: 预测收益
    ml_target = close * (1 + pred_10d) if pred_10d > 0 else None

    # 3. 风险收益比下限
    min_rr_target = buy_price + risk * params['min_rr_ratio']

    # 4. 中位数选择
    candidates = [c for c in [tech_target, ml_target, min_rr_target] if c is not None]
    if len(candidates) >= 2:
        candidates.sort()
        target = candidates[len(candidates) // 2]
    elif candidates:
        target = candidates[0]
    else:
        target = buy_price + risk * params['min_rr_ratio']

    # 硬约束
    target = clip(target, buy_price * (1 + params['target_clip_min']),
                          buy_price * (1 + params['target_clip_max']))

    return round(target, 2)
```

**阻力位计算** (无超参数):
```python
def compute_resistance(df_60d):
    rolling_high = df_60d['high'].rolling(20).max().iloc[-1]
    ma20 = df_60d['close'].rolling(20).mean().iloc[-1]
    ma60 = df_60d['close'].rolling(60).mean().iloc[-1]
    close = df_60d['close'].iloc[-1]
    candidates = [rolling_high, ma20, ma60]
    valid = [c for c in candidates if c > close * 1.005]
    return min(valid) if valid else close * 1.08
```

### 4.4 SignalStrengthFilter — 动态持仓数量

**现有**: 固定Top 10。
**新设计**: 基于信号强度cutoff，动态决定持仓数量。超参数由回测校准。

```python
def filter_by_signal_strength(stocks_sorted_by_composite, env_score, params):
    """
    动态cutoff: composite > cutoff 的入选, 数量受min_n/max_n约束
    """
    cutoff = params['composite_cutoff']
    above_cutoff = [s for s in stocks_sorted_by_composite if s['composite'] > cutoff]

    if env_score < 30:
        max_n = params['max_n_bear']
    elif env_score < 50:
        max_n = (params['max_n_bear'] + params['max_n_bull']) // 2
    else:
        max_n = params['max_n_bull']

    min_n = params['min_n']
    n = clip(len(above_cutoff), min_n, max_n)

    selected = stocks_sorted_by_composite[:n]
    return selected
```

### 4.5 RiskBudgetAllocator — 风险预算仓位

**输入**: selected_stocks (含composite_rank_pct, atr_20d, close), env_score
**输出**: 每只股票的position_pct

仓位分配逻辑本身是确定性公式(信号×波动率倒数归一化), 无需校准超参数。
总仓位由市场环境评分决定(复用现有逻辑), 个股上限(15%)和下限(1%)是风控硬约束。

```python
def allocate_positions(selected_stocks, env_score):
    # Step 1: 总仓位 — 由市场环境评分决定 (复用现有分档)
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
        raw_pct = s['norm_weight'] * total_exposure * 100
        s['position_pct'] = round(clip(raw_pct, 1.0, 15.0), 1)

    # Step 5: 等风险贡献校验 — 削减风险贡献过高的
    avg_risk = np.mean([s['position_pct'] * s['atr_20d'] / s['close'] for s in selected_stocks])
    for s in selected_stocks:
        rc = s['position_pct'] * s['atr_20d'] / s['close']
        if rc > avg_risk * 2.0:
            s['position_pct'] = round(s['position_pct'] * (avg_risk * 2.0 / rc), 1)

    # Step 6: 总仓位校验
    actual_total = sum(s['position_pct'] for s in selected_stocks)
    if actual_total > total_exposure * 100:
        scale = total_exposure * 100 / actual_total
        for s in selected_stocks:
            s['position_pct'] = round(s['position_pct'] * scale, 1)

    return selected_stocks
```

### 4.6 DynamicSLTPEngine — 日级止损/止盈回测引擎

**核心**: 替代固定持仓期，逐日判断止损/止盈/移动止盈触发。超参数由回测校准。

```python
def simulate_trade(buy_date, buy_price, stop_price, target_price, 
                   daily_ohlc, params):
    """
    逐日模拟一笔交易，返回 (exit_date, exit_price, exit_reason)
    
    exit_reason: 'stop_loss' | 'take_profit' | 'trailing_stop' | 'max_hold'
    """
    max_hold_days = params['max_hold_days']
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
        # 浮盈超过目标利润的 trailing_trigger_pct → 止损上移到成本价
        profit_range = target_price - buy_price
        if highest_since_entry >= buy_price + profit_range * params['trailing_trigger_pct']:
            new_stop = max(current_stop, buy_price)  # 至少保本
            current_stop = new_stop

        # Trailing: 最高价回撤 trailing_fallback_pct × profit_range 则出场
        trailing_stop = highest_since_entry - profit_range * params['trailing_fallback_pct']
        current_stop = max(current_stop, trailing_stop)

    # 到期: 按最后一天收盘价卖出
    last_day = daily_ohlc[min(max_hold_days-1, len(daily_ohlc)-1)]
    return last_day['date'], last_day['close'], 'max_hold'
```

### 4.7 A/B对比回测

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

## 5. 数据依赖

所有数据均已在数据库中可用:

| 数据 | 来源 | 说明 |
|------|------|------|
| OHLC | daily_quotes | open/high/low/close, 回测用 |
| ATR(20) | technical_indicators 或实时计算 | 20日ATR |
| MA20/MA60 | technical_indicators | 均线支撑/阻力 |
| pred_10d | ML scorer输出 | 10日预测收益 |
| composite | ML scorer输出 | 综合评分 |
| env_score | 市场环境评分模块 | 当日环境分 |

## 6. 集成方式

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

## 7. 文件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `portfolio_optimizer.py` | 新建 | 核心模块: 价格锚定+止损止盈+仓位优化 |
| `backtest/dynamic_sl_tp_engine.py` | 新建 | 日级OHLC止损/止盈回测引擎 |
| `backtest/ab_compare.py` | 新建 | A/B对比回测脚本 |
| `scripts/calibrate_optimizer_params.py` | 新建 | 超参数网格搜索校准脚本 |
| `optimizer_params.json` | 新建 | 校准后的最优超参数 (由校准脚本输出) |
| `tomorrow_stock_selector.py` | 修改 | 调用portfolio_optimizer替代原有逻辑 |

## 8. 验证标准

改进方案**必须同时满足**:
1. 对沪深300年化超额 > 现有方案
2. 对中证2000年化超额 > 现有方案
3. MaxDD 不恶化超过2个百分点
4. Sharpe ≥ 现有方案的90%

如果不满足，回退到现有逻辑，不会破坏生产。

## 9. 风险与回退

- **portfolio_optimizer.py 完全独立**，不修改原有价格计算函数，通过开关切换
- 回测失败 → 保留原有逻辑，新模块作为实验代码
- `tomorrow_stock_selector.py` 通过参数 `--optimizer v2` 启用新逻辑，默认仍走旧路径
