# CPPI实盘调仓顾问 设计文档

**日期**: 2026-04-05
**目标**: 将CPPI仓位管理集成到EastMoneyTrader，自动生成含杠杆控制的调仓指令

## 1. 概述

在EastMoneyTrader现有架构（OCR读持仓 + 自动下单）上增加CPPI层，实现：
- 基于净值追踪自动计算仓位比例（含融资融券杠杆）
- 集成NG模型选股，输出具体调仓指令（卖出X股、买入Y股）
- 极端市场机会时自动提升杠杆上限
- A股约束：T+1、100股整手、涨跌停、担保比例安全阀

## 2. 架构

```
StockTradebyZ NG模型
  → reports/daily_selection_ng102/*.json (Top-10推荐)
        ↓
EastMoneyTrader
  ├── OCR读取: 普通账户持仓 + 信用账户持仓/负债/担保比例
  ├── CPPIManager: 计算净值 → exposure → target持仓
  ├── RebalancePlan: 对比target vs current → 生成卖出/买入列表
  └── TradeExecutor: (可选)自动执行调仓
```

## 3. 核心组件：CPPIManager

### 3.1 净值计算

```python
净资产 = 信用账户总市值 - 融资负债 + 普通账户市值 + 现金余额
```

每次调仓前通过OCR从东方财富读取：
- 信用账户：持仓市值、融资余额、可用余额、担保比例
- 普通账户：持仓市值、可用余额

### 3.2 CPPI Exposure计算

```python
peak_nav = max(历史所有净资产记录)
floor_nav = peak_nav * (1 - cppi_floor)  # cppi_floor=0.05
cushion = 净资产 - floor_nav
exposure = min(max_leverage, cppi_multiplier * cushion / 净资产)
exposure = max(0.0, exposure)  # 不做空

# exposure含义:
#   1.5 → 总敞口=净资产×1.5（融资50%）
#   1.0 → 满仓无杠杆
#   0.4 → 40%仓位，60%现金
```

### 3.3 双模式杠杆

**正常模式**: max_leverage = 1.3

**机会模式**: 当市场极端超卖时，max_leverage提升到1.6
触发条件（≥2个同时满足）:
- `market_drawdown < -0.15`（大盘从60日高点跌>15%）
- `market_breadth < 0.15`（全市场<15%股票上涨）
- `vix_proxy > 1.8`（短期波动率远超长期）

退出条件: `market_breadth > 0.5` 连续3天，恢复正常max_leverage

市场指标从NG的ng102_feature_cache中读取（已有10个市场特征）。

### 3.4 安全约束（层叠优先级）

1. **CPPI计算** → target_exposure
2. **最大杠杆上限** → exposure ≤ max_leverage (1.3正常/1.6机会)
3. **担保比例安全阀**:
   - 担保比例 < 170% → 强制 exposure ≤ 1.0（停止融资）
   - 担保比例 < 150% → 强制减仓使担保比例回到 170%
   - 担保比例 < 130% → 紧急全部平仓（券商强制平仓线）
4. **单日最大变动** → 单次调仓exposure变化 ≤ 0.3（防止剧烈调仓）

## 4. 调仓计划生成

### 4.1 输入
- 当前持仓（OCR读取）: [{code, qty, available_qty, cost_price, current_price}]
- NG推荐Top-10: [{code, score, rank_score}]
- CPPI exposure: 0.0 ~ 1.6
- 总可投资金 = 净资产 × exposure

### 4.2 调仓逻辑

```
Step 1: 确定目标持仓
  target_value_per_stock = 总可投资金 / 10
  对每只推荐股票: target_qty = floor(target_value / 现价 / 100) * 100  # 100股整手

Step 2: 生成卖出列表（先卖后买）
  - 不在Top-10中的持仓 → 全部卖出（用available_qty，T+1约束）
  - 在Top-10中但持仓过多 → 卖出多余部分

Step 3: 生成买入列表
  - 新进入Top-10的 → 买入target_qty
  - 已持有但不足 → 补买差额
  
Step 4: 约束检查
  - 跳过涨停股（买不进）
  - 跳过跌停股（卖不出）
  - available_qty=0的不卖（T+1限制，今天买的）
  - 买入金额 ≤ 可用余额 + 可融资额度
```

### 4.3 输出格式

```
═══════════════════════════════════════════
CPPI调仓计划 2026-04-05
═══════════════════════════════════════════
净资产: 50.0万  |  历史最高: 52.0万  |  Floor: 49.4万
Cushion: 0.6万  |  Exposure: 24.0%  |  模式: 正常

目标投资: 12.0万 (净资产×24%)  |  当前持仓: 40.0万

【卖出】
  600519.SH  贵州茅台  卖出200股  @1850.00  ≈37.0万
  000001.SZ  平安银行  卖出300股  @15.20    ≈0.5万

【买入】
  002001.SZ  新和成    买入200股  @25.50    ≈0.5万
  601857.SH  中国石油  买入1000股 @9.80     ≈1.0万
  ...

【融资操作】
  偿还融资: 8.0万（降低杠杆）

预计调仓后: 持仓12.0万, 现金30.0万, 融资0万
担保比例: N/A (无融资)
═══════════════════════════════════════════
```

## 5. 状态持久化

`config/cppi_state.json`:
```json
{
  "initial_capital": 500000,
  "initial_date": "2026-04-05",
  "peak_nav": 520000,
  "peak_date": "2026-04-03",
  "current_mode": "normal",
  "opportunity_trigger_date": null,
  "nav_history": [
    {"date": "2026-04-05", "nav": 500000, "exposure": 1.0}
  ]
}
```

每次调仓自动更新。

## 6. CLI集成

在EastMoneyTrader的`trade.py`中添加命令:

```bash
# 初始化CPPI（首次使用）
python3 trade.py cppi-init --capital 500000

# 查看当前CPPI状态（OCR读持仓+计算exposure）
python3 trade.py cppi-status

# 生成调仓计划（读取最新NG报告 + OCR持仓 + CPPI计算）
python3 trade.py cppi-rebalance

# 执行调仓（需确认）
python3 trade.py cppi-rebalance --execute --confirm
```

## 7. 与StockTradebyZ的集成

- 报告路径: `reports/daily_selection_ng102/analysis_data_{YYYYMMDD}.json`
- 市场指标: 从`ng102_feature_cache`读取当日market_*字段（机会模式判断）
- CPPI参数: 从`production_config.json`读取（cppi_floor, cppi_multiplier）

## 8. 文件变更清单（EastMoneyTrader repo）

| 文件 | 改动 |
|------|------|
| `core/cppi_manager.py` | 新建：CPPI核心计算+净值追踪+状态持久化 |
| `core/rebalancer.py` | 修改：集成CPPI exposure到调仓逻辑 |
| `core/strategy_bridge.py` | 修改：支持NG报告格式+市场指标读取 |
| `core/screen_reader.py` | 修改：增加信用账户融资余额/担保比例OCR |
| `trade.py` | 修改：添加cppi-init/cppi-status/cppi-rebalance命令 |
| `config/cppi_state.json` | 新建：CPPI状态持久化 |
| `config/settings.json` | 修改：添加CPPI参数配置 |

## 9. 参数配置

```json
{
  "cppi": {
    "floor": 0.05,
    "multiplier": 20,
    "max_leverage_normal": 1.3,
    "max_leverage_opportunity": 1.6,
    "max_exposure_change_per_day": 0.3,
    "margin_safety_ratio": 1.7,
    "margin_warning_ratio": 1.5,
    "margin_emergency_ratio": 1.3
  }
}
```
