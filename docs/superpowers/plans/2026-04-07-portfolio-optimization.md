# 组合优化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建组合回测框架，网格搜索最优组合规则，更新生产配置——提升Sharpe、降MaxDD、提超额

**Architecture:** 单文件 `scripts/portfolio_backtest.py`(~400行)，读报告JSON+行情数据，模拟组合规则(止损/CPPI/持仓缓冲/加权)，支持网格搜索。在IS(2024-2026)和OOS(2018-2020)双期验证后更新production_config.json。

**Tech Stack:** Python, SQLite, numpy, pandas, json, argparse

---

### Task 1: 实现PortfolioBacktester核心引擎

**Files:**
- Create: `scripts/portfolio_backtest.py`

- [ ] **Step 1: 创建完整的portfolio_backtest.py**

这是一个单文件，包含全部逻辑。由于这是数据脚本而非库代码，直接写完整实现（不走TDD，用end-to-end验证）。

完整代码见下方。核心类 `PortfolioBacktester`：
- `__init__`: 加载报告JSON + 行情数据到内存
- `run(config)`: 按配置跑一次回测，返回metrics dict
- `grid_search(grid)`: 遍历参数组合，双期验证，排序输出

关键实现细节：
- ATR从OHLC自行计算(14日EMA of True Range)，不依赖technical_indicators表
- 涨跌停检测: 主板±9.5%, 创业板/科创板±19.5%, 北交所±29.5%
- 止损每日检查(不等调仓日)
- CPPI基于peak_nav动态计算exposure
- 排名用`rank_score`(优先)或`pred_10d`，跳过值为0的股票

- [ ] **Step 2: 验证单配置回测**

Run:
```bash
python3 scripts/portfolio_backtest.py \
  --report-dir reports/daily_selection_ng102_pre2020_v2 \
  --top-n 5 --rebal-days 10
```
Expected: 输出年化收益、超额、Sharpe、MaxDD等指标。Top-5超额应接近+2.2%(与之前内联回测一致)。

- [ ] **Step 3: Commit**

```bash
git add scripts/portfolio_backtest.py
git commit -m "feat: 组合回测框架 — 止损/CPPI/持仓缓冲/网格搜索"
```

---

### Task 2: 网格搜索最优配置

**Files:**
- No code changes — 运行已实现的grid_search

- [ ] **Step 1: 运行网格搜索**

Run:
```bash
python3 scripts/portfolio_backtest.py \
  --is-dir reports/daily_selection_ng102 \
  --oos-dir reports/daily_selection_ng102_pre2020_v2 \
  --grid
```
Expected: 输出Top-10配置，按 `0.5×IS_Sharpe + 0.5×OOS_Sharpe` 排序。两期都要Sharpe>0且超额>0。

- [ ] **Step 2: 记录最优配置**

从输出中选择排名第1的配置，记录所有参数和指标。

---

### Task 3: 更新生产配置

**Files:**
- Modify: `production_config.json`

- [ ] **Step 1: 将最优组合规则写入production_config.json**

在 `portfolio` section 中更新最优参数（具体值来自Task 2的网格搜索结果）。

- [ ] **Step 2: Commit**

```bash
git add production_config.json
git commit -m "feat: 组合优化 — 网格搜索最优配置更新生产"
```

- [ ] **Step 3: 输出最终对比表**

```
| 配置 | IS年化 | IS Sharpe | IS MaxDD | OOS年化 | OOS超额 | OOS Sharpe | OOS MaxDD |
|------|--------|----------|---------|---------|---------|-----------|---------|
| 基线(等权Top5) | ? | ? | ? | +16.4% | +2.2% | 0.78 | -20.3% |
| 最优配置 | ? | ? | ? | ? | ? | ? | ? |
```
