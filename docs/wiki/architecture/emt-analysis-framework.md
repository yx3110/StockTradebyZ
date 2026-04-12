# EastMoneyTrader 量化分析框架

> 2026-04-11/12 迭代 — 在 EastMoneyTrader 侧构建完整的 Level 1 量化分析基础设施：
> 因子有效性诊断、组合优化、回测验证、信号驱动调仓、幸存者偏差修复。

## 动机

原始的 EastMoneyTrader 只是执行层：读 StockTradebyZ 报告 → 生成调仓计划 → GUI 自动化。
缺乏对模型信号本身的独立诊断能力：

- 评分 IC 到底多强？IC 会衰减到哪一天？
- 分层收益单调吗？真 Alpha 还是行业 Beta？
- 现在的日频调仓是不是过度交易？
- 回测结果是否经过 Walk-Forward 校验？是否调过多重检验？

本次迭代在 EMT 侧搭建 `analysis/` 模块回答这些问题，不依赖 StockTradebyZ 内部工具，保持执行层诊断独立性。

## 模块结构

```
EastMoneyTrader/analysis/
├── data_loader.py          — 统一数据加载（评分报告 + SQLite 价格 + 市值）
├── ic_analyzer.py          — IC/IR、IC衰减、分层回测、中性化、拥挤度、时间趋势
├── portfolio_optimizer.py  — 等权/评分加权/风险平价/Markowitz/Black-Litterman
├── backtest_framework.py   — Walk-Forward、蒙特卡洛、多重检验校正
├── feature_audit.py        — NG 模型特征审计 (gain + SHAP + 单因子IC + 相关性) [2026-04-12]
└── feature_validator.py    — 候选新特征四关验证 (IC/分组/冗余/LGB增量) [2026-04-12]
```

CLI 入口:
- `python3 trade.py analyze --mode full --start 2024-01-02` — 完整 Level 1 分析
- `python3 scripts/audit_ng_features.py --targets 3d 5d 10d 15d [--model-path ...]` — 审计现有 NG 模型特征
- `python3 scripts/compare_targets.py --model-tag ng110` — 跨 target 对比
- `python3 scripts/validate_feature.py --candidate <csv> --name <feat_name>` — 候选新特征四关验证

## Level 1 覆盖矩阵

### 因子分析
| 项目 | 状态 | 实现 |
|------|------|------|
| IC/IR 计算 | ✅ | `compute_daily_ic` + `compute_ic_summary`（Spearman 秩相关 + t-stat + p-value） |
| IC 衰减曲线 | ✅ | `compute_ic_decay`（1/3/5/10/20 日） |
| 分层回测 | ✅ | `compute_quantile_returns`（5 组 + 单调性检验 + 交易成本扣除） |
| Winsorize ±3σ | ✅ | `winsorize_factor`（截面截断，默认启用） |
| 因子中性化 | ✅ | `neutralize_factor`（行业/市值/双重 OLS 残差） |
| 因子拥挤度 | ✅ | `compute_factor_autocorrelation`（排名自相关） |
| IC 时间趋势 | ✅ | `compute_ic_time_trend`（60 日滚动 + 前后半段对比） |
| 分市场环境 IC | ✅ | `compute_regime_ic`（牛/熊/震荡，无 lookahead） |

### 组合优化
| 项目 | 状态 | 实现 |
|------|------|------|
| 等权 / 评分加权 / 评分/波动率加权 | ✅ | `equal_weight` / `score_weight` / `score_vol_weight` |
| 风险平价 | ✅ | `risk_parity`（1/σ）+ `risk_parity_tilted`（带评分倾斜） |
| Markowitz MVO | ✅ | `markowitz`（SLSQP，带 max_weight 约束） |
| Black-Litterman | ✅ | `black_litterman`（市场均衡 + 观点贝叶斯后验） |
| 换手率约束 | ✅ | `max_turnover` 参数（LP 线性化 + trust-constr） |
| 流动性约束 | ✅ | `max_position_value` + `total_budget`（按日成交额 × 参与率） |
| 行业集中度约束 | ✅ | `check_industry_concentration`（默认 ≤25%） |
| 协方差估计 | ✅ | `estimate_covariance`（sklearn LedoitWolf 自适应收缩） |

### 回测框架
| 项目 | 状态 | 实现 |
|------|------|------|
| Walk-Forward | ✅ | `WalkForwardAnalyzer`（训练 500 日 / 测试 60 日 + WFE 指标） |
| 蒙特卡洛 Bootstrap | ✅ | `bootstrap_returns`（10000 次，返回百分位分布） |
| 交易序列打乱 | ✅ | `trade_shuffle`（保留单笔盈亏，打破时间顺序） |
| 参数扰动 | ✅ | `parameter_perturbation`（加噪测稳健性） |
| Bonferroni / Holm-Bonferroni | ✅ | `MultipleTestingCorrector` |
| Sharpe haircut | ✅ | Bailey & de Prado (2014) 公式 |

### 回测检查清单
| 项目 | 状态 | 备注 |
|------|------|------|
| 复权价格 | ✅ | `adj_factor` 计算 `adj_close`，覆盖不足时 fallback 原始 close |
| 前视偏差 | ✅ | 报告日期 T 匹配 T→T+N 前向收益，无未来数据 |
| 停牌过滤 | ✅ | `is_suspend` 从 volume=0 计算 |
| 涨跌停过滤 | ✅ | 按板块阈值：主板 ±10% / 创业板/科创板 ±20% / 北交所 ±30% |
| 交易成本 | ✅ | 单边 0.15%、往返 0.30%，扣除后对比 |
| 分市场环境 | ✅ | 过去 20 日市场均值划分牛/熊/震荡 |
| **幸存者偏差** | ✅ | **2026-04-12 从 tushare 导入 322 只退市股 + 257K 条日线** |

### 特征审计 (2026-04-12 新增)
| 项目 | 状态 | 实现 |
|------|------|------|
| LGB gain importance | ✅ | 从 booster 直取 + 归一化 |
| SHAP importance | ✅ | TreeExplainer 在 5000 行采样上, 支持多 target 共享样本 |
| 单因子 Rank IC | ✅ | `compute_univariate_ic_batched`（一次 groupby 算所有特征, 复用 ic_analyzer）|
| 特征相关性矩阵 | ✅ | `compute_correlation_matrix` (Spearman) + `find_redundant_pairs` (默认 \|相关\|≥0.8) |
| 综合排名 + 决策 | ✅ | `build_audit_report` 三维 rank_pct 均值 → STRONG/NORMAL/WEAK/UNKNOWN |
| 多 target 对比 | ✅ | `compare_targets.py` 找"全 horizon STRONG"核心特征 / "全 WEAK" 剔除候选 |

### 候选新特征验证 (2026-04-12 新增)
| 关卡 | 通过标准 | 实现 |
|------|---------|------|
| 1. IC 分析 | ic_mean ≥ 0.02 + ICIR ≥ 0.3 + 胜率 ≥ 55% | 复用 `ICAnalyzer.compute_daily_ic` |
| 2. 分组回测 | 单调 + Long-Short 年化 ≥ 5% + Sharpe ≥ 1 | 复用 `compute_quantile_returns` |
| 3. 冗余/正交性 | max|相关| < 0.8 + 正交化 IC 保留 ≥ 50% | sklearn LinearRegression 残差 |
| 4. LGB 增量 | ΔIC ≥ 0.005 (带/不带候选特征训两版 LGB 对比 OOS IC) | lightgbm.train + early stopping |

决策: 4/4 → ACCEPT, 3/4 → MARGINAL (watch list), ≤2/4 或 Gate 1 挂 → REJECT.

## 关键诊断发现（基于 ng1.0.1）

### 1. 因子中性化：真 Alpha 占比 53%

```
原始 IC：          0.1635
行业中性化：       0.0961  (-41%)
市值中性化：       0.1238  (-24%)
双重中性化：       0.0861  (-47%)
```

**结论**：模型约一半收益来自行业轮动+小盘效应（Beta），另一半才是纯选股 Alpha。
并非失效，但需要知道自己承担的风险结构。

### 2. IC 衰减曲线：最优持有期 10-20 日

```
持有期   IC均值    IC_IR
1日      0.044     0.52
3日      0.071     0.95
5日      0.089     1.26
10日     0.116     1.75   ← 峰值
20日     0.114     1.47
```

**结论**：每天换股浪费交易成本。信号要 10 天才完全实现。

### 3. IC 时间趋势：2024 年轻微衰减

```
早期 IC（H1 2024）:  0.245
近期 IC（H2 2024）:  0.143
比值：               0.58  → ⚠️ 轻微下降
```

**建议**：定期跑 `analyze --mode ic` 监测，若比值 < 0.5 考虑暂停或重训。

### 4. 因子排名自相关：信号切换健康

```
1日自相关：  0.80
5日自相关：  0.59
10日自相关： 0.43
20日自相关： 0.13
```

**结论**：20 日后信号完全刷新，与 IC 衰减曲线一致。

### 5. 特征审计：发现 4 个 feature 定义 bug + 7 个僵尸特征 (2026-04-12)

首次运行 `audit_ng_features.py` 对 ng1.0.1 + ng1.1.0 全量审计, 发现:

**ng1.0.1 (69 特征) — 4 对相关 >= 0.999 的冗余特征**:
```
volume_contraction          = volume_ratio_5d            (都是 vol5/vol20)
industry_return_5d          = sw_index_return_5d         (参数名误导)
industry_relative_strength  = residual_return_20d        (都是 stock-industry)
revenue_growth              ≡ profit_to_gr (margin)      (命名错误, 不是真 growth)
```

**ng1.1.0 (声称 77 特征) — 7 个僵尸特征**:
`cx_beta_mkt_vol`, `cx_drawdown_regime`, `cx_ind_mkt_dir`, `cx_momentum_trend`, `cx_quality_stress`, `cx_value_bear`, `cx_vol_stress` 全部 gain=0 + SHAP=0 — 模型元信息声称使用但实际在 `ng101_feature_cache` 里不存在, 训练时全 NaN, LGB 从未分裂. **实际有效特征只有 70 个, 不是 77 个**.

**结论**: 这些 bug 在 factor_quality_*.json 里看不出来 (自相关和 gain 归一化抹掉了零值的绝对含义), 必须靠**独立 SHAP + 单因子 IC + 相关性矩阵三角验证**才能捕获.

**修复**: ng_feature_calculator.py 改 4 个定义 + cache 回填 7 个 cx_* + 重训. 详见 [lessons/known-pitfalls.md](../lessons/known-pitfalls.md#特征工程类) 和 [models/ng-factor-quality.md](../models/ng-factor-quality.md#emt-独立审计结果-2026-04-12).

## 信号驱动调仓

取代日频调仓的核心机制，在 `core/rebalancer.py` 内通过 `HoldingsStateManager`
实现：

### 触发规则（优先级由高到低）
1. **止损**：亏损 > 6% 强制卖出（无视持有期）
2. **min_hold 保护**：持有 < 7 日阻止非止损卖出
3. **评分跌破地板**：评分 < 25 → 卖出
4. **评分衰减**：从入场跌 >30% 且持续下滑 → 卖出
5. **超持有期**：持有 > 20 日且评分 < 35 → 卖出
6. **买入信号**：评分 >= 60 且不在冷却期（7 日）

### 状态文件：`config/holdings_state.json`
- 每股记录 `entry_date`、`entry_price`、`entry_score`、`score_history`
- 冷却期登记避免卖完立刻买回
- Phase 2 OCR 对账，Phase 5 交易后状态更新

## 退市股数据修复（幸存者偏差）

### 问题
- 原始 DB `securities.delist_date` 全空、`is_active=1`
- 历史 IC 分析只看到"还活着"的股票，高估因子表现

### 修复
tushare `stock_basic(list_status='D')` → 322 只退市股
- 227 只已存在 securities 记录，`UPDATE` 添加 `delist_date` + `is_active=0`
- 95 只新记录 `INSERT`
- 日线数据：`pro.daily()` + `adj_factor()`，导入 **257,416 条**

脚本：`EastMoneyTrader/scripts/import_delisted_stocks.py`

覆盖范围：
```
2020: 16 只退市
2021: 20 只
2022: 46 只
2023: 45 只
2024: 52 只
2025: 29 只
2026: 4 只（至 2026-03-31）
```

### 实测对比：IC 无变化

导入前后对 2024 全年做 IC 对比：

| 指标 | 导入前 | 导入后 |
|------|--------|--------|
| IC 均值 | 0.1902 | 0.1902 |
| IC_IR | 1.25 | 1.25 |
| 双重中性后真 Alpha 占比 | 47% | 47% |

**根因**：StockTradebyZ 在生成 `analysis_data_*.json` 时已过滤退市股
（样本 `600260 *ST凯乐` 等完全不出现在 `all_stocks_with_scores` 中）。
**历史报告级 IC 分析本来就没有幸存者偏差**。

这次导入是**防御性修复**，价值在于：
1. **下次 NG 模型训练时可用完整样本做 OOS**：能检验模型是否能在退市前 6 个月识别 underweight
2. **手工构造 watchlist 的风险评估**：若误选后来退市的股票，现在能算出真实损失轨迹
3. **未来任何不过滤退市股的回测脚本**默认会使用完整宇宙

## 已知局限

1. **日线复权因子稀疏**（2020-2024 约 35% 覆盖），LedoitWolf 需样本 ≥ n+10
2. **Markowitz 对输入极敏感**，生产建议用 BL 或 risk_parity
3. **中性化 `np.linalg.lstsq` 每日独立回归**，全周期分析 ~3 秒/年
4. **换手率约束在起点 infeasible 时退回 current_weights**，不硬失败

## 相关文件

- `/Users/yangxu/EastMoneyTrader/analysis/` — 全部分析模块
- `/Users/yangxu/EastMoneyTrader/scripts/import_delisted_stocks.py` — 退市股导入
- `/Users/yangxu/EastMoneyTrader/trade.py` 的 `analyze` 子命令 — CLI 入口
- `/Users/yangxu/EastMoneyTrader/core/rebalancer.py` `HoldingsStateManager` — 信号驱动状态

## 交叉引用
- [自动调仓系统](auto-rebalancing.md) — EMT 与 StockTradebyZ 的整体联动
- [回测方法论](../evaluation/backtesting.md) — StockTradebyZ 内部回测对应物
- [北极星评估体系](../evaluation/north-star.md) — V5.2 评分与此处 IC 诊断的关系
