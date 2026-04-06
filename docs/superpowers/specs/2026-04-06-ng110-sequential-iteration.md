# NG v1.1.0 顺序迭代设计: 残差标签优先 + OOS硬门槛

**日期**: 2026-04-06
**基线**: ng1.0.2 (V5.2=74.0% A+, 2018-2020 OOS: 年化超额-7.6%)
**目标**: 2018-2020 OOS超额 > 0% (泛化能力修复), 同时不损失2024+ in-sample表现

## 问题诊断

ng1.0.2在2024-2026 in-sample表现优异(年化+117%, 超额+107%), 但2018-2020真正OOS完全失效(年化+6.7%, 超额-7.6%)。

**根因**: β_UMD = 3.029 — 模型对动量因子有3倍隐性暴露。
- 训练期(2020-2026)是牛市, 动量持续溢价 → 模型学到"买动量"
- 2018-2020是熊市/震荡, 动量反转 → 模型选股被反噬
- 行业超额标签去除了行业beta, 但**没有去除动量/市值/波动率暴露**

## 核心策略: 一次一变量, OOS硬门槛

不同时改多个东西。按因果优先级逐步验证, 每步必须通过2018-2020 OOS门槛。

```
Step 1: 残差标签 (根因修复)
  → fast-check + 2018-2020 OOS回测
  ├─ PASS (超额>0%) → Step 2
  └─ FAIL → 停下诊断, 不继续堆料

Step 2: 残差标签 + 资金流因子 (增量信号)
  → fast-check + 2018-2020 OOS回测
  ├─ 增量>0 → Step 3
  └─ 增量≤0 → 只保留残差标签

Step 3: 残差标签 + 资金流 + WF8/regime (鲁棒性)
  → 完整训练 + 双向评估(WF-OOS + Pre-2020)
```

---

## Step 1: 残差标签 (方向2单独验证)

### 改动
- 训练时使用 `label_Xd` (风格残差) 而非 `label_raw_Xd` (行业超额)
- ng110_feature_cache 中已预计算: `label_Xd = excess_return - β×[log_mcap, momentum_20d, volatility_20d]`
- 特征集不变: 69个(59股票+10市场), 与ng1.0.2相同

### 预期效果
- 去除对市值/动量/波动率的隐性依赖
- 2018-2020 OOS应改善(动量反转不再伤害)
- In-sample可能略降(去掉了"免费"的动量beta)

### 验证命令
```bash
# Step 1a: fast-check (2个WF窗口, ~2min)
python3 ml_models/ng/ng_trainer.py \
  --start-date 2020-01-01 \
  --fast-check \
  --purge-days 15

# Step 1b: 完整训练 (3个WF窗口, ~1-2h)
python3 ml_models/ng/ng_trainer.py \
  --start-date 2020-01-01 \
  --purge-days 15

# Step 1c: 生成2018-2020报告
python3 backtest/batch_generate_v395_reports.py \
  --version ng1.1.0 \
  --start-date 2018-04-02 --end-date 2020-12-31 \
  --output-dir reports/daily_selection_ng110_pre2020

# Step 1d: 2018-2020 OOS回测 (用本会话已有的回测脚本逻辑)
# 期望: 年化超额 > 0% (vs ng1.0.2的-7.6%)
```

### 通过条件
- **硬门槛**: 2018-2020 OOS 年化超额 > 0%
- **软指标**: fast-check 10d ICIR > 0.5, 2024+ 10d ICIR不低于ng1.0.2的80%

---

## Step 2: 残差标签 + 资金流因子

### 前置条件
Step 1 通过硬门槛。

### 改动
- 在Step 1基础上启用8个资金流因子: `--enable-moneyflow`
- 特征集: 69 + 7 = 76个 (northbound_stock_5d 全为0, 实际7个有效因子)
- 不启用交互因子(保持一次一变量)

### 资金流因子列表
1. `net_mf_ratio_5d` — 5日净流入强度
2. `big_order_ratio` — 主力资金占比
3. `big_order_trend_5d` — 主力加仓趋势
4. `small_vs_big_divergence` — 散户vs主力分歧
5. `mf_concentration` — 资金集中度
6. `mf_momentum_10d` — 资金流动量
7. `mf_volume_divergence` — 量价背离

### Bug fix needed
ng_trainer.py MONEYFLOW_FEATURE_NAMES 缺少 `northbound_stock_5d`, INTERACTION_FEATURE_NAMES 缺少 `ix_north_cap`。需要修复(即使这两个是placeholder)。

### 验证命令
```bash
# Step 2a: fast-check
python3 ml_models/ng/ng_trainer.py \
  --start-date 2020-01-01 \
  --fast-check \
  --purge-days 15 \
  --enable-moneyflow

# Step 2b: 完整训练 + 2018-2020 OOS
# (同Step 1b-1d流程)
```

### 通过条件
- **硬门槛**: 2018-2020 OOS 年化超额 > Step 1结果 (增量为正)
- **软指标**: fast-check 10d ICIR > Step 1的ICIR

---

## Step 3: 残差标签 + 资金流 + WF8 + 市况加权

### 前置条件
Step 2有正增量, 或Step 1通过但Step 2无增量(则退回Step 1配置)。

### 改动
- WF窗口: 3 → 8 (90天步长, 更密覆盖)
- 可选: `--regime-weight` (熊市样本权重1.2, 牛市0.8)
- 可选: `--enable-interaction` (IC>0.02的交互因子)

### 验证命令
```bash
# 完整训练 (~3-5h)
python3 ml_models/ng/ng_trainer.py \
  --start-date 2020-01-01 \
  --purge-days 15 \
  --enable-moneyflow \
  --wf-windows 8 \
  --regime-weight

# 双向评估
# 1. WF-OOS (向前泛化)
# 2. Pre-2020 (向后泛化) 
# 3. 2018-2020 顺序回测 (实际P&L)
```

### 最终通过条件
- 2018-2020 OOS 年化超额 > 0%
- WF-OOS 北极星 > B级
- 2024+ in-sample不低于ng1.0.2的70%
- 顺序回测MaxDD < -30%

---

## 技术细节

### ng110_feature_cache 数据确认
- 行数: 3,176,524 (2020-01-02 ~ 2026-04-03)
- label_Xd: 风格残差标签 ✅ (已预计算)
- label_raw_Xd: 原始行业超额标签 ✅ (保留用于对比)
- 资金流因子: 7个有效 ✅ (northbound_stock_5d=0)
- 交互因子: 8个 ✅

### ⚠️ 前置任务: 回填2018-2020 ng110缓存
ng110_feature_cache 当前仅覆盖2020-01-02起, **无2018-2020数据**。
OOS回测需要先回填:
```bash
python3 ml_models/ng/ng_cache_updater.py \
  --start-date 2018-01-01 --end-date 2019-12-31 \
  --version ng1.1.0
```
预计耗时: ~15-30分钟 (约500个交易日 × ~2秒/日)
注意: moneyflow_daily已覆盖2018-01-02起, 无需额外回填资金流原始数据。

### 2018-2020 OOS回测方法
与本会话已验证的方法一致:
- 等权Top-N, 10日调仓
- 双边0.3%交易成本
- 基准: 沪深300
- 指标: 年化收益, 年化超额, Sharpe, MaxDD, 胜率

### 不做的事
- 不改特征工程(Step 1/2用现有69/76特征)
- 不改模型结构(LightGBM+XGBoost+CatBoost+RF+HGB+LambdaRank)
- 不改ICIR自适应权重机制
- 不改scorer/推理逻辑
- 不同时改多个方向

---

## 风险与回退

| 风险 | 概率 | 应对 |
|------|------|------|
| Step 1 FAIL: 残差标签也不解决OOS | 中 | 回查残差回归是否正确去除了动量; 考虑扩展训练集到2018 |
| Step 2 资金流无增量 | 高 | 正常, GBDT可能已从价量因子学到类似信号; 跳过, 只用Step1 |
| In-sample显著下降 | 低 | 可接受, 因为目标是OOS修复而非IS最大化 |
| 训练时间过长(>5h) | 中 | 用fast-check先排除明显失败的配置 |
