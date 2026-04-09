# 0AMV 全市场活跃市值指标 + 牛熊体制切换设计

## 背景

0AMV（活筹指数）是指南针软件的核心市场级指标，用DMA（动态移动平均）以换手率为平滑因子，从成交额中分离"活跃资金"和"死筹"。本方案将其复刻为Python实现，并基于0AMV构建牛熊体制判断信号，在牛市/熊市分别使用不同NG模型（ng1.0.1 vs ng1.0.4）进行选股。

## 一、0AMV指标计算

### 1.1 数据源

全市场每日成交额 = 上证指数(000001.SH) amount + 深证成指(399001.SZ) amount

数据获取方式：
- **历史回填**：tushare `pro.index_daily()` 补全2018-01至今
- **每日更新**：修改 `quick_daily_update.py` 的 `update_market_indices()` 确保每日抓取指数amount

全市场流通市值 = `SUM(circ_mv)` from `daily_basic` 表（每日聚合全A股）

市场换手率 = 全市场成交额 / 全市场流通市值

### 1.2 核心算法

通达信原版公式（改造为全市场版）：

```
Var1 := SMA(market_amount, 10, 1) / 1e7

C5   := DMA(SMA(Var1, 3, 1),  market_turnover / 0.02)
C13  := DMA(SMA(Var1, 3, 1),  market_turnover / 0.10)
C34  := DMA(SMA(Var1, 8, 1),  market_turnover / 0.18)
INF  := DMA(Var1,             market_turnover / 1.10)

MA60 := MA(Var1, 60)

DIF  := EMA(Var1, 12) - EMA(Var1, 26)
DEA  := EMA(DIF, 9)
MACD := (DIF - DEA) * 2
```

其中：
- `SMA(X, N, M)` = 中国式SMA: `Y = (M * X + (N-M) * Y_prev) / N`，即 `ewm(alpha=M/N)`
- `DMA(X, A)` = 动态移动平均: `Y = A * X + (1-A) * Y_prev`，A为动态系数（换手率缩放）
- `market_turnover` = 当日全市场换手率

### 1.3 输出字段

| 字段 | 含义 |
|------|------|
| `var1` | 成交额平滑值（千万元） |
| `amv_c5` | 5日活跃成本线 |
| `amv_c13` | 13日活跃成本线 |
| `amv_c34` | 34日活跃成本线 |
| `amv_inf` | 无穷成本线（全历史加权） |
| `amv_ma60` | var1的60日简单均线 |
| `amv_dif` | MACD快线 |
| `amv_dea` | MACD慢线 |
| `amv_macd` | MACD柱状 = (DIF-DEA)*2 |

### 1.4 存储

新建 `market_amv` 表：

```sql
CREATE TABLE IF NOT EXISTS market_amv (
    trade_date DATE PRIMARY KEY,
    market_amount REAL,        -- 全市场成交额（元）
    market_circ_mv REAL,       -- 全市场流通市值（万元）
    market_turnover REAL,      -- 全市场换手率
    var1 REAL,
    amv_c5 REAL,
    amv_c13 REAL,
    amv_c34 REAL,
    amv_inf REAL,
    amv_ma60 REAL,
    amv_dif REAL,
    amv_dea REAL,
    amv_macd REAL,
    amv_regime INTEGER         -- 1=牛市, -1=熊市
);
```

## 二、牛熊体制判断

### 2.1 规则

**转牛条件**（三者同时满足）：
1. var1 单日涨幅 ≥ +4.3%
2. var1 > amv_ma60（站上60日线）
3. amv_macd 从 <0 转为 >0（MACD上穿零轴）

**转熊条件**（三者同时满足）：
1. var1 单日跌幅 ≤ -2.3%
2. var1 < amv_ma60（跌破60日线）
3. amv_macd 从 >0 转为 <0（MACD下穿零轴）

**状态机**：
- 初始状态：根据首日条件判定
- 状态一旦确定，维持不变直到触发对向切换条件
- 熊市中出现大阳线但 var1 < amv_ma60 且 macd < 0 → 维持熊市（超跌反弹）

### 2.2 输出

`amv_regime` 字段：`1` = 牛市, `-1` = 熊市

## 三、双模型切换回测

### 3.1 目标

验证假设：牛市用ng1.0.1（高收益高波动）、熊市用ng1.0.4-3seed（稳健低回撤）的切换策略，是否优于单独使用任一模型。

### 3.2 方法

1. 计算2020-01-01至2026-03-31的每日 `amv_regime`
2. 对每个交易日，根据当日regime选择对应模型的报告：
   - regime=1 → 使用 `reports/daily_selection_ng101/` 的选股
   - regime=-1 → 使用 `reports/daily_selection_ng104_ensemble_3seed/` 的选股
3. 合并为一套新的报告序列
4. 用标准北极星评估（`run_north_star_eval.py`）对比：
   - 纯ng1.0.1
   - 纯ng1.0.4-3s
   - 切换策略

### 3.3 评估指标

重点关注：
- 年化收益（切换策略能否接近ng1.0.1的129.7%？）
- MaxDD（切换策略能否接近ng1.0.4的-23.1%？）
- Sharpe（能否同时改善收益和风险？）
- 牛熊切换次数（太频繁说明信号不稳）

## 四、实现步骤

### Step 1: 数据准备
- 回填上证+深证指数的历史amount到 `daily_quotes`
- 修改 `quick_daily_update.py` 确保每日更新指数amount
- 聚合每日全市场流通市值

### Step 2: 0AMV计算模块
- 新建 `indicators/market_amv.py`
- 实现 SMA(中国式)、DMA(动态)、标准EMA/MA
- 实现完整0AMV四条线 + MA60 + MACD
- 实现牛熊状态机
- 计算结果写入 `market_amv` 表

### Step 3: 集成到每日更新
- `quick_daily_update.py` 每日更新后自动调用0AMV计算
- 增量计算（只算新增日期）

### Step 4: 双模型切换回测
- 新建 `backtest/regime_switch_backtest.py`
- 读取 `market_amv` 表的 `amv_regime`
- 按regime合并两个模型的报告
- 调用北极星评估对比三种策略

### Step 5: 验证与调参
- 检查牛熊切换时点是否符合直觉（对照大盘K线）
- 如有需要微调4.3%/-2.3%阈值
