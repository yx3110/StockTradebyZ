# NG v1.1.0 设计: 三方向联合升级

**日期**: 2026-04-05
**基线**: ng1.0.2 (V5.2=74.0% A+, 69因子, 行业超额标签, 下行风险模型+CPPI)
**目标**: 通过信号质量+标签工程+训练框架三方向升级突破 V5.2 > 78%

## 方案概述

采用 **方案C (独立fast-check + 合并)**: 一次性实现三个方向代码，通过CLI开关分别验证独立增益，确认正向后合并为 ng1.1.0 完整训练。

### 三个方向

| 方向 | 改动 | CLI开关 | 预期增益 |
|------|------|---------|---------|
| 1. 信号质量 | +8资金流因子 + IC筛选交互因子 | `--enable-moneyflow`, `--enable-interaction` | IC提升 |
| 2. 标签工程 | 风格因子残差标签(去size/mom/vol) | `--residual-label` | 纯alpha, 降低因子暴露 |
| 3. 训练框架 | 6-8个WF窗口 + 市况样本加权 | `--wf-windows N`, `--regime-weight` | OOS稳健性 |

---

## 方向1: 资金流因子 (8个新因子)

### 数据源

- `moneyflow`: 个股资金流（4档: 小/中/大/特大单），按日期批量查询，5183只/天
- `moneyflow_hsgt`: 北向资金日度汇总（沪股通/深股通）
- `moneyflow_ind`: 不可用（权限不足），从个股聚合替代

### 新因子定义

| # | 因子名 | 计算方式 | 经济逻辑 |
|---|--------|---------|---------|
| 1 | `net_mf_ratio_5d` | 5日净资金流入 / 5日成交额 | 资金净流入强度(归一化) |
| 2 | `big_order_ratio` | (大单+特大单净买入额) / 总成交额 | 主力资金占比 |
| 3 | `big_order_trend_5d` | 5日大单净买入的线性回归斜率 | 主力加仓加速/减速 |
| 4 | `small_vs_big_divergence` | sign(小单净买) × sign(大单净买) 的5日均值 | 散户vs主力分歧(反向指标) |
| 5 | `mf_concentration` | (特大单买入+卖出) / 总成交额 | 资金集中度 |
| 6 | `mf_momentum_10d` | 净资金流MA5 - 净资金流MA10 | 资金流趋势动量 |
| 7 | `northbound_stock_5d` | 个股5日北向净买入(从hsgt_top10，无数据=0) | 外资偏好 |
| 8 | `mf_volume_divergence` | sign(净资金流) × sign(涨跌幅) 的5日均值 | 量价背离预警 |

### 原始数据存储

新增表 `moneyflow_daily`:
```sql
CREATE TABLE IF NOT EXISTS moneyflow_daily (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code VARCHAR(10) NOT NULL,
    trade_date DATE NOT NULL,
    buy_sm_amount REAL,
    sell_sm_amount REAL,
    buy_md_amount REAL,
    sell_md_amount REAL,
    buy_lg_amount REAL,
    sell_lg_amount REAL,
    buy_elg_amount REAL,
    sell_elg_amount REAL,
    net_mf_amount REAL,
    UNIQUE(code, trade_date)
);
CREATE INDEX idx_moneyflow_daily_date ON moneyflow_daily(trade_date);
CREATE INDEX idx_moneyflow_daily_code_date ON moneyflow_daily(code, trade_date);
```

---

## 方向1b: 非线性交互因子 (IC筛选)

### 候选交互组合 (8个预定义)

| # | 交互因子名 | 公式 | 经济逻辑 |
|---|-----------|------|---------|
| 1 | `ix_vol_pullback` | volume_ratio_5d × pullback_to_ma20 | 放量回踩入场信号 |
| 2 | `ix_big_trend` | big_order_ratio × trend_strength_20d | 主力加仓+趋势确认 |
| 3 | `ix_rsi_mf` | rsi_14 × mf_momentum_10d | 技术超卖+资金回流 |
| 4 | `ix_ind_big` | industry_relative_strength × big_order_ratio | 强势行业+主力介入 |
| 5 | `ix_mf_efficiency` | net_mf_ratio_5d / (turnover_rate + 1e-8) | 资金流入效率 |
| 6 | `ix_vol_surge_pullback` | cs_rank_volume_surge × pullback_from_high | 异常放量+回撤 |
| 7 | `ix_alpha_conc` | residual_return_20d × mf_concentration | 纯alpha+主力集中 |
| 8 | `ix_north_cap` | northbound_stock_5d × log_market_cap | 外资蓝筹偏好 |

### IC 筛选规则

1. 在训练集截面上计算 Rank IC vs label_10d
2. 保留条件: |IC| > 0.02 **且** 与现有因子最大 |corr| < 0.7
3. 预计保留 3-5 个, 由数据驱动, 非硬编码
4. 筛选在训练阶段执行, 保留的因子名写入模型 pkl

---

## 方向2: 风格因子残差标签

### 当前问题

ng1.0.1 行业超额标签: `excess = stock_return - industry_median`

仍暴露:
- **Size**: 小盘股系统性跑赢大盘时标签偏高
- **Momentum**: 动量股标签偏高 (V5.0重建时 β_UMD=3.029 的教训)
- **Volatility**: 高波动股标签方差大, 噪声信号混入

### 方案: 日截面 Fama-French 式回归取残差

```python
# 每日截面回归 (N~5000只股票)
for date in trading_dates:
    X = df[['log_market_cap', 'momentum_20d', 'volatility_20d']]  # 3个风格因子
    y = df['excess_return_Xd']  # 行业超额收益
    model = LinearRegression().fit(X, y)
    df['residual_Xd'] = y - model.predict(X)  # 残差 = 纯alpha
```

- 对 3d/5d/10d/15d 四个窗口分别做
- `downside_10d = max(0, -residual_10d)`
- 保留原始 `label_raw_Xd` (行业超额) 用于对比
- 回归系数不保存, 每次从数据重新计算

### CLI开关

`--residual-label`: 启用残差标签 (默认 off, 使用行业超额标签)

---

## 方向3: 训练框架升级

### 3a: WF窗口增加

| 参数 | 当前 (ng1.0.2) | 升级 (ng1.1.0) |
|------|---------------|----------------|
| WF窗口数 | 3 | 6-8 (CLI控制) |
| Step | 120天 | 90天 (更密) |
| Test window | 120天 | 120天 (不变) |
| Min train | 900天 | 900天 (不变) |
| Mode | Expanding | Expanding (不变) |

CLI: `--wf-windows N` (默认3, 建议8)

### 3b: 市况样本加权

基于 `market_return_20d` 将训练样本分为三档:

| 市况 | 条件 | 权重 | 占比(估计) |
|------|------|------|-----------|
| 牛市 | market_return_20d > +5% | 0.8 | ~25% |
| 震荡 | -5% ≤ market_return_20d ≤ +5% | 1.0 | ~55% |
| 熊市 | market_return_20d < -5% | 1.2 | ~20% |

- 权重通过 `sample_weight` 参数传入 LightGBM/XGBoost/CatBoost
- RandomForest 和 HistGB 也支持 sample_weight
- LambdaRank 的 sample_weight 作用于 group 内排序

CLI: `--regime-weight` (默认 off)

---

## 缓存表 Schema

### ng110_feature_cache

```sql
CREATE TABLE IF NOT EXISTS ng110_feature_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code VARCHAR(10) NOT NULL,
    trade_date DATE NOT NULL,
    features_json TEXT NOT NULL,     -- 77+因子 (69原有 + 8资金流 + 3-5交互)
    label_3d REAL,                   -- 残差标签 (--residual-label) 或 行业超额
    label_5d REAL,
    label_10d REAL,
    label_15d REAL,
    label_raw_3d REAL,               -- 原始行业超额标签 (始终保留)
    label_raw_5d REAL,
    label_raw_10d REAL,
    label_raw_15d REAL,
    downside_10d REAL,
    market_return_5d REAL,
    market_return_20d REAL,
    market_volatility_20d REAL,
    market_breadth REAL,
    market_new_high_ratio REAL,
    northbound_flow_5d REAL,
    market_volume_ratio REAL,
    market_drawdown REAL,
    vix_proxy REAL,
    market_momentum_diff REAL,
    UNIQUE(code, trade_date)
);
CREATE INDEX idx_ng110_date ON ng110_feature_cache(trade_date);
CREATE INDEX idx_ng110_code_date ON ng110_feature_cache(code, trade_date);
```

---

## fast-check 验证矩阵

| 实验 | 资金流 | 交互 | 残差标签 | WF窗口 | 市况权重 | 对比基线 |
|------|--------|------|---------|--------|---------|---------|
| baseline | ❌ | ❌ | ❌ | 3 | ❌ | = ng1.0.2 |
| exp1_signal | ✅ | ✅ | ❌ | 3 | ❌ | 因子增益 |
| exp2_label | ❌ | ❌ | ✅ | 3 | ❌ | 标签增益 |
| exp3_train | ❌ | ❌ | ❌ | 8 | ✅ | 框架增益 |
| final | 正向✅ | 正向✅ | 正向✅ | 正向✅ | 正向✅ | 合并验证 |

每个 fast-check ~2min, 总验证 ~10min。合并后完整训练 ~3-5h。

---

## 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `ml_models/ng/ng_schema.py` | 修改 | 新增 moneyflow_daily + ng110_feature_cache 表定义 |
| `ml_models/ng/ng_feature_calculator.py` | 修改 | 新增8资金流因子 + 8交互因子计算 |
| `ml_models/ng/ng_cache_updater.py` | 修改 | 支持 ng1.1.0 版本, 残差标签, moneyflow数据获取 |
| `ml_models/ng/ng_trainer.py` | 修改 | WF窗口参数化, 市况加权, 交互因子IC筛选, CLI开关 |
| `ml_models/ng/ng_production_scorer.py` | 修改 | 支持 ng110 表, 交互因子推理 |
| `fetch_data/quick_daily_update.py` | 修改 | 新增 moneyflow_daily 日常更新步骤 |
| `backtest/batch_generate_v395_reports.py` | 修改 | 注册 ng1.1.0 版本 |
| `production_config.json` | 修改 | 升级版本号 (验证通过后) |

## 成功标准

- fast-check 至少2个方向有正向增益 (10d ICIR提升)
- 合并后完整训练 V5.2 > 76% (vs 当前74.0%)
- L3 风控 ≥ 55% (不低于ng1.0.2)
- 无负向方向混入最终模型
