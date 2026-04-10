# NG v1.0.8 — 低换手组合构建

**Date**: 2026-04-10
**Base**: ng1.0.1 信号 (不改) + ng1.0.5 风控 (可选叠加)
**Target**: 换手 44x→15-20x, Sharpe 1.93→2.2+, Pre-2020 保持 A+
**Principle**: 全部固定参数，无数据驱动优化，零过拟合风险

## 背景

ng1.0.7 迭代教训：从模型信号层面改进(条件标签、市场特征)容易过拟合训练期regime模式(Pre-2020=C级)。ng1.0.1 是唯一双A+模型(WF-OOS + Pre-2020)，信号不动。

当前瓶颈是**换手率44x过高**——每次调仓几乎全换，交易成本年化6.6%吃掉alpha。解决方案：在组合构建层面加入4个固定规则降换手。

## 四个组合规则

### Rule 1: 持仓缓冲 (Hysteresis)

买卖不对称门槛，避免边缘股票反复进出：

- **买入**: 新股必须排进全市场 Top-`buy_threshold` 才入选 (default: 8)
- **卖出**: 现有持仓跌出 Top-`sell_threshold` 才卖出 (default: 20)
- 持仓数维持 `target_n`=10，当持仓 <10 时从 Top-8 补入，>10 时不强制卖出

```
if stock in portfolio:
    sell if rank > sell_threshold (20)
else:
    buy if rank <= buy_threshold (8) and len(portfolio) < target_n
```

**参数**: `buy_threshold=8`, `sell_threshold=20`, `target_n=10` (固定常数)

### Rule 2: 分批调仓 (Staggered Rebalancing)

10只持仓分2组，错开调仓：

- A组(slot 0-4): 在 rebalance_day=0, 10, 20, ... 调仓
- B组(slot 5-9): 在 rebalance_day=5, 15, 25, ... 调仓
- 每次调仓只评估本组5只持仓是否需要替换

```
day % 10 == 0: rebalance group A (slots 0-4)
day % 10 == 5: rebalance group B (slots 5-9)
```

**效果**: 单次最多换5只(50%)，等效 holding_period=10d 但换手减半

**参数**: `n_groups=2` (固定)

### Rule 3: 最小持有期 (Min Holding Period)

买入后至少持有 N 天，避免噪声换手：

- 买入日期记录在 `entry_date[stock]`
- 调仓时检查: `if today - entry_date < min_hold_days: skip sell`
- 即使排名跌出 sell_threshold 也继续持有直到满足最小持有期

**参数**: `min_hold_days=5` (固定)

### Rule 4: 成本感知排名 (Cost-Aware Ranking)

在排名时对新买入股票施加成本惩罚：

```python
for stock in candidates:
    if stock in current_portfolio:
        adj_score = raw_score  # 无成本
    else:
        adj_score = raw_score - cost_penalty
# cost_penalty = 单次交易成本 (买入佣金+卖出佣金+滑点)
# 固定值: 0.003 (0.3%)
```

排名用 `adj_score` 而非 `raw_score`，新股必须 score 显著高于现有持仓才能替换。

**参数**: `cost_penalty=0.003` (固定, 对应0.3%单次交易成本)

## 实现方案

### 修改文件

| 文件 | 改动 |
|------|------|
| `backtest/backtest_report_based.py` | `run_single_backtest()` 新增4个参数，在调仓逻辑中实现4条规则 |
| `backtest/run_north_star_eval.py` | CLI新增 `--hysteresis`, `--staggered`, `--min-hold`, `--cost-penalty` 参数 |

**不改的文件**: ng_trainer.py, ng_production_scorer.py, ng_cache_updater.py, tomorrow_stock_selector.py

### 回测逻辑变更

当前 `run_single_backtest` 的调仓逻辑:
```
每 focus_days 天:
  1. 加载当天报告 → 按 rank_field 排序
  2. 取 Top-N → 新持仓
  3. 计算换手成本
```

改为:
```
每天检查是否需要调仓:
  1. 确定当前调仓组 (day % 10 < 5 → A组, else → B组)
  2. 加载当天报告 → 按 rank_field 排序
  3. 对已持有股票: adj_score = raw_score
     对新候选股票: adj_score = raw_score - cost_penalty
  4. 按 adj_score 重排
  5. 对当前组的每个slot:
     - if 持仓排名 > sell_threshold AND 已持有 >= min_hold_days: 卖出
     - if slot空 AND 候选排名 <= buy_threshold: 买入
  6. 计算换手成本
```

### 参数汇总

| 参数 | 默认值 | 含义 | 可调范围 |
|------|:------:|------|:--------:|
| `buy_threshold` | 8 | 新股买入门槛(排名) | 5-15 |
| `sell_threshold` | 20 | 持仓卖出门槛(排名) | 15-30 |
| `target_n` | 10 | 目标持仓数 | 不变 |
| `n_groups` | 2 | 分批组数 | 2-3 |
| `min_hold_days` | 5 | 最小持有天数 | 3-7 |
| `cost_penalty` | 0.003 | 新股成本惩罚 | 0.002-0.005 |

所有参数为**固定常数**，不从数据优化。

## 评估计划

### Step 1: 消融实验 (WF-OOS)

在 ng1.0.1 报告上逐个开启规则，测量换手和Sharpe变化:

| 实验 | 规则 | 预期换手 |
|------|------|:--------:|
| Baseline | 无 | 44x |
| +Hysteresis | Rule 1 only | ~30x |
| +Cost-Aware | Rule 1+4 | ~25x |
| +Min-Hold | Rule 1+3+4 | ~22x |
| **Full (A+C)** | **Rule 1+2+3+4** | **15-20x** |

### Step 2: Pre-2020 验证

用最优配置跑 Pre-2020(2018-2019) 评估，必须保持 A+ 级别。

### Step 3: 叠加 ng1.0.5 风控

在低换手组合上叠加三层风控(SL6%+RG-agg+VT20%+CPPI)，看能否同时满足:
- 换手 < 20x
- Sharpe > 2.5
- MaxDD < 15%
- Pre-2020 A+

## 预期效果

| 指标 | ng1.0.1裸 | ng1.0.8(预期) | 目标 |
|------|:---:|:---:|:---:|
| 换手(年化) | 44x | **15-20x** | <20x |
| Sharpe | 1.93 | **2.2-2.5** | >2.5 |
| 年化(净) | 76.9% | **78-82%** | >80% |
| Pre-2020 V5.2 | 73.7% A+ | **A+** | A+ |

## 版本号

**ng1.0.8** — 低换手组合构建 (信号=ng1.0.1, 组合规则=4条固定规则)
