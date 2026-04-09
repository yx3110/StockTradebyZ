# NG v1.0.7 — 条件化单模型 + Pareto回撤过滤

**Date**: 2026-04-09
**Target**: 年化 > 80%, Sharpe > 2.5, MaxDD < 15% (裸信号，无后置风控)
**Base**: NG1.0.1 (69特征, 年化129.7%, Sharpe=3.17, MaxDD=-25.4%)

## 核心问题诊断

NG1.0.1信号最强但回撤最大，根因：
1. 模型不感知市场环境 — 10个market特征太粗糙，GBDT无法学会"熊市选防御、牛市选进攻"
2. 熊市信号反转 — worst 60d ICIR=-0.249，危机期间选的股票反而跌更多
3. 回撤预测(NG1.0.2)只是软折扣，高alpha但高回撤的股票仍被选中

## 方案概述

**Part A (条件化单模型)**: 把市场regime作为连续特征+交叉特征喂入单一模型，让GBDT自己学会环境自适应选股。

**Part B (Pareto回撤过滤)**: 独立风险模型预测个股maxdd_10d，硬过滤高风险标的后再按alpha排序。

## Part A: 条件化特征工程

### A1. 新增市场状态连续特征 (+8个)

在现有10个市场特征基础上新增，使用0AMV指标和扩展市场统计：

| # | 特征名 | 计算方法 | 含义 |
|---|--------|----------|------|
| 1 | `amv_var1` | 0AMV var1连续值 (from market_amv表) | 市场活跃度指数 |
| 2 | `amv_macd` | 0AMV MACD值 | 市场活跃度动量 |
| 3 | `amv_regime_days` | 当前regime持续天数 / 60 (归一化) | regime稳定性 |
| 4 | `market_ret_60d` | 沪深300 60日收益率 | 中期趋势 |
| 5 | `market_vol_ratio` | market_vol_5d / market_vol_60d | 短期vs长期波动 |
| 6 | `breadth_momentum_5d` | market_breadth 5日变化 | 涨跌面动量 |
| 7 | `market_skewness_20d` | 20日市场收益偏度 | 尾部风险 |
| 8 | `liquidity_stress` | 当日市场成交额 / 60日均值 | 流动性压力 |

**数据来源**: market_amv表(0AMV) + daily_quotes(沪深300, 全A) — 无需新API

### A2. 新增股票×市场交叉特征 (+7个)

让GBDT直接看到"个股特性在当前市场环境下的意义"：

| # | 特征名 | 计算公式 | 含义 |
|---|--------|----------|------|
| 1 | `cx_beta_mkt_vol` | idiosyncratic_volatility × market_volatility_20d | 高市场波动时高beta更危险 |
| 2 | `cx_momentum_trend` | cs_rank_return_5d × market_return_20d | 动量与大盘同向放大 |
| 3 | `cx_ind_mkt_dir` | industry_relative_strength × sign(market_return_20d) | 行业强度×大盘方向 |
| 4 | `cx_vol_stress` | volume_ratio_5d × liquidity_stress | 个股放量×市场缩量 = 异常 |
| 5 | `cx_drawdown_regime` | current_drawdown × amv_regime_days | 个股回撤在regime稳定时更有意义 |
| 6 | `cx_value_bear` | pe_percentile_60d × (1 - market_breadth) | 弱市时低估值更重要 |
| 7 | `cx_quality_stress` | roe_ttm × market_vol_ratio | 高波动时质量因子避风港效应 |

**注意**: 交叉特征使用IC筛选(min_ic=0.02, max_corr=0.7)，训练前自动过滤，只保留有效的。

### A3. 条件化标签

当前NG1.0.1统一使用行业超额标签(stock_return - industry_median)。问题：熊市时所有行业都跌，超额标签接近0，模型学不到有用信号。

**改进**:
- 熊市(market_return_20d < -5%): label = 截面排名百分位 (rank_pct)，即"跌得少 = 好"
- 非熊市: 保留行业超额标签 (现有方法)
- 混合方式: `label = (1 - bear_weight) × excess_return + bear_weight × rank_pct`
  - `bear_weight = max(0, min(1, -market_return_20d / 0.10))` — 连续插值，不硬切换

**实现**: 在ng_cache_updater中计算conditional_label_Xd，存入cache表。

### A4. 增强样本加权

当前: bull=0.8, sideways=1.0, bear=1.2
**改为**: bull=0.7, sideways=1.0, bear=1.5, crisis(mkt_ret_20d < -10%)=2.0

根据0AMV regime细化:
- AMV牛市: 0.7
- AMV横盘: 1.0
- AMV熊市: 1.5
- AMV熊市+大跌(20d ret < -10%): 2.0

## Part B: Pareto回撤过滤

### B1. 独立风险模型

使用全部特征(含Part A新增)训练独立LightGBM回归模型，目标: maxdd_10d。

与NG1.0.2的区别:
- NG1.0.2: 风险模型共享alpha模型特征，结果作为score的折扣
- **NG1.0.7**: 风险模型独立训练(独立WF窗口)，结果作为**硬过滤门槛**

### B2. Pareto选股逻辑

1. Alpha模型产出: pred_10d (所有股票)
2. Risk模型产出: pred_maxdd_10d (所有股票)
3. **硬过滤**: pred_maxdd_10d < quantile_20(当日全市场) → 排除 (最差20%风险)
4. **排名**: 剩余80%中按pred_10d排序，取Top-N

可调参数: `risk_filter_quantile` (默认0.20，即过滤最差20%)

### B3. 回测评估时自动做消融实验

报告自动输出三组对比:
- Pure alpha (无风险过滤)
- Pareto filtered (20% cutoff)
- Pareto filtered (30% cutoff)

## 特征总结

| 类别 | NG1.0.1 | NG1.0.7 | 增量 |
|------|:-------:|:-------:|:----:|
| 股票特征 | 56 | 56 | 0 |
| 市场特征 | 10 | 18 | +8 |
| 交叉特征 | 0 | 7 (IC筛选后可能减少) | +7 |
| **总计** | **66** | **81** (最多) | **+15** |

注: 使用NG1.0.3 base (56 stock, 去掉3个FLIP因子)，不用NG1.0.4的9个smoothing特征(避免特征膨胀)。

## 版本号

**ng1.0.7** — 条件化单模型 + Pareto回撤过滤

## 文件变更

| 文件 | 改动 |
|------|------|
| `ml_models/ng/ng_schema.py` | 添加ng1.0.7表映射 + conditional_label列 |
| `ml_models/ng/ng_feature_calculator.py` | 添加 `compute_amv_features()`, `compute_conditional_interaction_features()`, `compute_extended_market_features()` |
| `ml_models/ng/ng_cache_updater.py` | v1.0.7支持: 计算新特征+条件化标签, 存入cache |
| `ml_models/ng/ng_trainer.py` | v1.0.7特征列表, 增强样本加权, Pareto过滤逻辑 |
| `ml_models/ng/ng_production_scorer.py` | v1.0.7推理: Pareto过滤 + 新特征加载 |
| `tomorrow_stock_selector.py` | 添加 `--scoring-version ng1.0.7` 支持 |

## 评估计划

1. **Fast-check** (2min): 2个WF窗口验证IC/ICIR方向
2. **完整训练** (~3-5h): 全量WF + downside模型
3. **报告生成**: `batch_generate_v395_reports.py --version ng1.0.7`
4. **北极星评估**: WF-OOS + Pre-2020双向评估
5. **消融实验**: 单独评估Part A (条件化模型) 和Part B (Pareto过滤) 的贡献

## 预期效果

| 指标 | NG1.0.1 | NG1.0.7 预期 | 改进机制 |
|------|:-------:|:----------:|----------|
| 年化收益 | 129.7% | 80-100% | 回撤过滤会牺牲部分高风险高收益标的 |
| Sharpe | 3.17 | 2.5-3.5 | 条件化模型减少熊市错误选股 |
| MaxDD | -25.4% | -12%~-15% | Pareto硬过滤 + 条件化标签 |
| Worst 60d ICIR | -0.249 | >0 | 市场状态感知让模型不在熊市犯大错 |
| 超额胜率 | 70.1% | 65-72% | 交叉特征提升regime特异性选股准确率 |
