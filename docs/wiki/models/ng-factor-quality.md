# NG 系列因子质量与权重分布

记录每个 NG 模型训练后的因子质量指标和权重分布，用于跟踪因子健康度、发现冗余因子、指导特征工程方向。

> **维护规则**: 每次训练新 NG 模型后，运行 `_log_factor_quality` 自动生成 `factor_quality_{timestamp}.json`，据此更新本页。另外, **EMT 侧的 `scripts/audit_ng_features.py` 提供独立第三方审计** (gain/SHAP/单因子 IC/相关性四维), 用于验证训练日志的声明是否和模型实际行为一致.

---

## EMT 独立审计结果 (2026-04-12)

EastMoneyTrader 侧建立了 `analysis/feature_audit.py` 作为**独立于训练 pipeline 的第三方审计**. 运行在 `ng101_feature_cache` 的 50k 采样 + ng101 / ng110 模型上, 交叉检查模型的 feature_importance 是否与数据一致.

### 审计指标框架

每个特征在 4 个 target (3d/5d/10d/15d) 下分别评估:
- **gain_importance**: LGB booster 的 gain-based importance (归一化到 sum=1)
- **shap_importance**: SHAP TreeExplainer 的 mean(|SHAP value|) (归一化到 sum=1)
- **ic_mean**: 单因子与 label_{target} 的 Rank IC 均值 (绝对值越大越好)
- **composite_score**: 三维 rank_pct 均值, 决策 STRONG/NORMAL/WEAK

### ng1.0.1 审计结果

| 维度 | 数值 |
|---|---|
| 声明特征数 | 69 (59 stock + 10 market) |
| 实际有效特征 | 66 (减去 3 对冗余) |
| 全 horizon STRONG | 22 |
| 全 horizon WEAK | 0 |
| 冗余特征对 (相关 >= 0.8) | 17 对, 其中 4 对 >= 0.999 (详见 lessons/known-pitfalls.md) |

**核心 alpha 来源** (all-STRONG 前 10):
1. `market_volatility_20d` (0.986)
2. `industry_relative_strength` (0.945)
3. `northbound_flow_5d` (0.940)
4. `turnover_rate` (0.928)
5. `cs_rank_volatility` (0.903)
6. `market_new_high_ratio` (0.900)
7. `market_drawdown` (0.884)
8. `vix_proxy` (0.880)
9. `relative_strength_vs_peers` (0.879)
10. `market_volume_ratio` (0.870)

→ NG 本质是**技术 + 市场情绪模型**, 基本面只有 `roe_ttm` 和 `pe_ttm` 进榜.

### ng1.1.0 审计结果 (发现问题)

| 维度 | 数值 | 对 ng1.0.1 |
|---|---|---|
| 声明特征数 | 77 | +8 |
| **实际有效特征** | **70** | +4 (cx_* 7 个 gain=0 被剔除) |
| 全 horizon STRONG | 28 | +6 |
| 全 horizon WEAK | 1 (`industry_rank_return_5d`) | +1 |
| UNKNOWN | **7 (全部是 cx_\*)** | +7 |
| 冗余特征对 | 17 对 | 未修 (ng101 的 4 对完全保留) |

**关键问题**:
1. 7 个新增的 `cx_*` 交互特征 (cx_beta_mkt_vol, cx_drawdown_regime, cx_ind_mkt_dir, cx_momentum_trend, cx_quality_stress, cx_value_bear, cx_vol_stress) **训练时全 NaN** (cache 未回填), 模型从未用到它们
2. ng1.0.1 的 4 个 feature 定义 bug 原封不动保留 (volume_contraction=volume_ratio_5d, industry_return_5d=sw_index_return_5d, industry_relative_strength=residual_return_20d, revenue_growth≡profit_to_gr margin)

**修复** (2026-04-12, 已完成):
- 4个feature定义bug已修复: volume_contraction/sw_index_return_5d/industry_relative_strength删除, revenue_growth修正为真or_yoy
- cx_*僵尸特征根因修复: `_get_active_stock_features()`用`self._cond_ix_cols`代替`version_ge`硬编码
- ng101_feature_cache已回填profit_margin_ratio + 真revenue_growth + peg_proxy(用真or_yoy)
- **ng1.0.1重训结果: V5.2=72.1% A+, 年化165.7%, MaxDD=-11.7%, Sharpe=2.753**
- **ng1.1.0重训结果: V5.2=70.4% A+, 年化122.8%, MaxDD=-12.5%, Sharpe=2.065**

Bug修复效果对比 (同时段2024-2026, 10d持仓, composite排名):

| 指标 | ng1.0.1原版 | ng1.0.1 bugfix | ng1.1.0 bugfix |
|---|---|---|---|
| V5.2 | 73.7% A+ | **72.1% A+** | 70.4% A+ |
| 年化 | 98.3% | **165.7%** | 122.8% |
| Sharpe | 2.367 | **2.753** | 2.065 |
| MaxDD | -27.2% | **-11.7%** | -12.5% |
| 特征数 | 69 | 66 (3冗余NaN) | 68 |

**Bug#4(revenue_growth修正)是最大贡献**: 模型首次获得真正的成长性信号。

### 审计工具位置

| 工具 | 路径 | 用途 |
|---|---|---|
| 审计 CLI | `EastMoneyTrader/scripts/audit_ng_features.py` | 一键审计任意 ng 模型 (支持多 target) |
| 跨 target 对比 | `EastMoneyTrader/scripts/compare_targets.py` | 找"全 horizon STRONG"核心特征 |
| 候选特征验证 | `EastMoneyTrader/scripts/validate_feature.py` | 新特征上线前的四关验证 |
| 审计报告输出 | `EastMoneyTrader/logs/feature_audit/` | Markdown + CSV, 按 date × model_tag × target 存档 |

**建议使用节奏**:
- 每次新训练一个 NG 版本, 跑 `audit_ng_features.py --model-path <pkl>` 验证:
  - 没有 gain=0 的僵尸特征
  - 没有 |相关| >= 0.95 的冗余对
  - 新加入的特征确实进入 all-STRONG (或至少 NORMAL)

---

## ng1.1.0 迭代结果 (2026-04-12)

基于全面评估的两项改进:
- **P0**: 移除8个多版本验证无用因子 (81→73特征)
- **P1**: Ensemble权重shrinkage(70%IC+30%等权) + floor(≥5%)

### 全量训练 WF ICIR (3窗口)

| 周期 | ng1.0.7 (81feat) | ng1.1.0 (73feat) | 变化 |
|---|---|---|---|
| 3d | 0.850 | 0.754 | -11.3% |
| 5d | 0.940 | 0.811 | -13.7% |
| **10d** | 1.059 | **1.018** | -3.9% |
| **15d** | 1.098 | **1.176** | **+7.1%** |

**结论**: 中长周期(10d/15d)维持甚至提升，短周期(3d/5d)下降。精简的因子对短周期有微弱信号，但对中长周期是噪声。主要使用10d/15d做选股，结果正面。

### Ensemble 权重改善 (10d target)

| 指标 | ng1.0.7 | ng1.1.0 P1 |
|---|---|---|
| 最大权重 | 35.5% (hgb) | 19.0% (xgb) |
| 最小权重 | 1.0% (lgb) | 15.1% (lgb_rank) |
| 极差 | 34.5pp | 3.9pp |

P1 shrinkage 彻底消除了算法独裁问题。

### 北极星评估 (最终版, 基于ng1.0.1)
- **V5.2 = 49.9% B级** (composite排名, 10d持仓, 2024-2026)
- 与ng1.0.1在相同时段表现一致——移除的3个因子权重本就近零(roe_change 0.008%, n_sectors_strong 0%, days_since_breakout 0.012%)
- ICIR adaptive weights正确生效: 3d=0.200, 5d=0.216, 10d=0.271, 15d=0.313
- WF ICIR: 3d=0.754, 5d=0.811, 10d=1.018, 15d=1.176, WFER=0.588
- 模型文件: `ng110_seed42_multi_target_20260412_161546.pkl`

### 关键教训
1. **P0精简对预测无实际影响**: 移除的3个因子importance≈0，去掉后分数不变。价值在于代码精简
2. **P1权重均衡化是WF层面有效的改动**: ensemble极差从34.5pp→3.9pp
3. **V485 pipeline的PRUNE_FEATURES与NG特征互不干扰**: V485裁剪的是V4.7.x命名体系的特征，NG的features_json特征不受影响
4. **基准版本选择很重要**: ng1.1.0必须基于ng1.0.1(69feat)而非ng1.0.7(81feat)，因为分析是在ng1.0.1上做的
5. **2024-2026时段(含大熊市)裸信号V5.2≈B级是正常水平**: 和ng1.0.1一致，生产部署需叠加CPPI/止损等风控才能达到A+

### P2/P3 迭代结果 (2026-04-12)

在ng1.1.0(P0+P1)基础上继续:
- **P2**: 新增4个alpha因子(cs_rank_pb, cs_rank_dv, peg_proxy, pb_roe_ratio), 77特征
- **P3**: Stock因子对market因子做OLS正交化, 减少market依赖

Fast-check对比 (2窗口平均ICIR):

| 配置 | 3d | 5d | 10d | 15d | 平均 |
|---|---|---|---|---|---|
| ng1.0.1 基线 | 0.629 | 0.835 | 0.931 | 1.060 | 0.864 |
| P2 (77feat) | 0.756 | 0.888 | 1.058 | 0.981 | **0.921 (+7%)** |
| P3 orth (77feat) | 0.774 | 0.881 | 0.959 | 1.076 | **0.923 (+7%)** |

P2全量WF (3窗口): 3d=0.681, 5d=0.836, 10d=0.879, 15d=1.060 (vs ng1.0.1基本持平)

**结论**: P2和P3各带来约7%提升(vs ng1.0.1)，但增量有限。P3正交化没有额外效果——industry excess标签已部分中性化market。新增的4个P2因子效果中性，可保留不必移除。

### 后续方向
- ng1.1.0最终配置: P0(精简3因子) + P1(权重均衡) + P2(4新alpha因子) + P3(market正交化), 70特征
- 在ng1.0.8基础上应用P1权重均衡化可能是更有效的方向

---

## 全面评估总结 (2026-04-11)

基于 5 个 NG 版本 (ng1.0.0 ~ ng1.0.7) 的因子权重全面分析。

### 核心发现

1. **Market 组仍是最大权重来源**，但已从 ng1.0.0 的 81% 降至 ng1.0.7 的 21%。改善主要归功于 cs_rank 和 industry 组的引入。
2. **14 个核心因子** (Mean≥2%, CV<1.0) 贡献了约 60% 的模型信息量，其中 market 组占 8 个。
3. **ng1.0.3 (41特征) 是因子效率最高的版本**: 近零因子仅 1 个，15d ICIR=1.229 为所有版本最高。多并非好——精简反而提升信号纯度。
4. **新增因子组成功率分化严重**: cs_rank(11%) 和 residual(14%) 大成功; smoothing(9.2%) 中等; ext_market 8个因子中5个全零——扩market信号已饱和。
5. **算法权重极端倾斜**: ICIR优化导致部分算法被压到1%，ng1.0.4的rf占94%，模型多样性丧失。

---

## 跨版本对比

### WF OOS ICIR + 组权重

| 版本 | 特征数 | 10d ICIR | 15d ICIR | market | fund | industry | cs_rank | residual | 近零数 |
|---|---|---|---|---|---|---|---|---|---|
| ng1.0.0 | 62 | 0.515 | 0.681 | **72.3%** | 13.3% | 2.0% | — | — | 21 |
| ng1.0.1 | 69 | **0.931** | 1.060 | 38.4% | 12.6% | 9.3% | 11.0% | 14.0% | 9 |
| **ng1.0.3** | 41 | 1.014 | **1.229** | 25.6% | 23.6% | 13.4% | 15.3% | 11.4% | **1** |
| ng1.0.4 | 75 | 0.935 | 1.069 | 24.1% | 17.5% | 11.8% | 13.7% | 5.5% | 4 |
| ng1.0.7 | 81 | **1.059** | 1.098 | **20.7%** | 14.3% | 14.6% | 14.1% | 5.3% | 8 |

**趋势**: 特征数增加→market占比降低(好)，但近零因子也增加(说明新增因子有水分)。

### 因子组效率 (组贡献% / 因子数 = 每因子平均贡献)

| 组 | 效率 | 平均总贡献 | 平均因子数 | 评级 |
|---|---|---|---|---|
| market | 3.62%/因子 | 36.2% | 10 | 高效核心 |
| residual | 1.81%/因子 | 9.1% | 5 | 效率良好 |
| cs_rank | 1.81%/因子 | 13.5% | 8 | 效率良好 |
| industry | 1.59%/因子 | 10.2% | 9 | 效率良好 |
| fundamental | 1.22%/因子 | 16.3% | 13 | 中等(因子太多) |
| volume | 1.08%/因子 | 7.3% | 7 | 中等 |
| smoothing | 1.02%/因子 | 9.2% | 9 | 中等(仅ng1.0.4) |
| ext_market | 0.95%/因子 | 7.6% | 8 | 中等(5/8全零) |
| cond_ix | 0.91%/因子 | 6.4% | 7 | 中等(仅ng1.0.7) |
| pullback | 0.76%/因子 | 4.6% | 6 | 偏低 |
| trend | 0.63%/因子 | 2.7% | 4 | 偏低 |

---

## 核心因子 (14个, Mean≥2%, CV<1.0)

跨所有NG版本始终重要且稳定的因子:

| 因子 | 均值 | CV | 组 | 角色 |
|---|---|---|---|---|
| **vix_proxy** | 7.8% | 0.36 | market | 短期恐慌指标，所有版本Top5 |
| **cs_rank_turnover** | 7.1% | 0.41 | cs_rank | 行业内换手率排名，v1.0.1+后第一cs_rank因子 |
| **market_volatility_20d** | 5.5% | 0.48 | market | 市场波动率，与vix_proxy互补 |
| **industry_return_20d** | 5.4% | 0.79 | industry | 行业中期动量，v1.0.3+后常居Top3 |
| **northbound_flow_5d** | 4.4% | 0.65 | market | 北向资金，外资定价权指标 |
| **roe_ttm** | 4.2% | 0.24 | fundamental | **最稳定因子(CV=0.24)**，唯一核心基本面 |
| **market_return_20d** | 3.8% | 0.34 | market | 中期市场趋势 |
| **market_drawdown** | 3.7% | 0.89 | market | 市场回撤 |
| **turnover_rate** | 3.3% | 0.40 | volume | 换手率，流动性+活跃度 |
| **idiosyncratic_volatility** | 3.0% | 0.38 | residual | 个股特异波动率 |
| **market_volume_ratio** | 2.8% | 0.61 | market | 市场量能 |
| **market_new_high_ratio** | 2.8% | 0.64 | market | 创新高比例，市场强度 |
| **relative_strength_vs_peers** | 2.6% | 0.96 | residual | 相对行业强弱 |
| **market_return_5d** | 2.4% | 0.88 | market | 短期市场趋势 |

**观察**: 14个核心因子中8个是market, 说明模型仍高度依赖市场状态做选股。个股alpha信号(roe_ttm, cs_rank_turnover, turnover_rate, idiosyncratic_volatility)仅6个。

---

## 稳定贡献因子 (Mean 0.5~2%, CV<0.8)

| 因子 | 均值 | CV | 组 |
|---|---|---|---|
| market_momentum_diff | 1.9% | 0.71 | market |
| residual_return_20d | 1.9% | 0.77 | residual |
| pullback_to_ma10 | 1.7% | 0.47 | pullback |
| log_adv_20d | 1.6% | 0.65 | fundamental |
| trend_strength_20d | 1.5% | 0.75 | trend |
| cs_rank_volatility | 1.4% | 0.19 | cs_rank |
| industry_hhi | 1.4% | 0.44 | industry |
| revenue_growth | 1.3% | 0.32 | fundamental |
| log_amount_ma5 | 1.3% | 0.54 | volume |
| net_profit_margin | 1.2% | 0.33 | fundamental |
| pb | 1.2% | 0.32 | fundamental |
| cs_rank_rsi | 1.1% | 0.27 | cs_rank |
| cs_rank_return_5d | 1.1% | 0.08 | cs_rank |
| pe_ttm | 1.1% | 0.38 | fundamental |
| market_breadth | 1.0% | 0.59 | market |
| debt_to_assets | 1.0% | 0.36 | fundamental |
| current_ratio | 1.0% | 0.26 | fundamental |
| pe_percentile_60d | 0.9% | 0.35 | fundamental |
| cs_rank_pe | 0.9% | 0.35 | cs_rank |
| dv_ratio | 0.9% | 0.48 | fundamental |
| residual_volume | 0.8% | 0.40 | residual |
| free_float_ratio | 0.7% | 0.65 | fundamental |
| volume_price_corr | 0.7% | 0.35 | volume |
| ocf_quality | 0.7% | 0.35 | fundamental |
| residual_skewness | 0.7% | 0.26 | residual |
| pullback_to_ma20 | 0.6% | 0.07 | pullback |
| volume_cv | 0.6% | 0.53 | volume |
| cs_rank_pullback | 0.6% | 0.25 | cs_rank |
| cs_rank_return_20d | 0.6% | 0.36 | cs_rank |

---

## 可移除因子 (始终近零)

所有出现版本中均 <0.5% 的因子:

| 因子 | 出现版本数 | 均值 | 组 | 判定 |
|---|---|---|---|---|
| **n_sectors_strong** | 3 | 0.000% | industry | **移除** — 3版本全零 |
| **roe_change** | 5 | 0.094% | fundamental | **移除** — 5版本全近零 |
| **days_since_breakout** | 4 | 0.171% | trend | **移除** — 信号被adx_proxy替代 |
| industry_breadth | 4 | 0.153% | industry | 观察 — 被industry_return_20d替代 |
| sector_breadth_vs_market | 3 | 0.162% | industry | 观察 — 被industry_return_20d替代 |
| volume_ratio_5d | 4 | 0.267% | volume | 观察 — 与volume_breakout高度相关 |
| sector_volume_vs_market | 3 | 0.305% | industry | 观察 |
| sw_index_return_5d | 4 | 0.308% | industry | 观察 — 可能被industry_return_5d替代 |

**ng1.0.7 ext_market 特有的全零因子 (5个)**: market_ret_60d, market_vol_ratio, breadth_momentum_5d, market_skewness_20d, liquidity_stress — 这些因子的信息已被现有market因子覆盖。

---

## 新增因子贡献评估

| 版本 | 新增组 | 总贡献 | 活跃/总数 | 最佳因子 | 评价 |
|---|---|---|---|---|---|
| ng1.0.1 | cs_rank | 11.0% | 10/10 | cs_rank_turnover 2.7% | 大成功，全部活跃 |
| ng1.0.1 | residual | 14.0% | 5/5 | relative_strength 6.8% | 大成功，每因子效率高 |
| ng1.0.4 | smoothing | 9.2% | 8/9 | ma60_distance 2.5% | 中等，vol_regime近零 |
| ng1.0.7 | ext_market | 7.6% | **3/8** | amv_var1 4.2% | 分化：AMV有效，其余5个全零 |
| ng1.0.7 | cond_ix | 6.4% | 7/7 | cx_beta_mkt_vol 2.5% | 中等，全部活跃但贡献分散 |

---

## 算法权重异常

10d target ensemble 权重:

| 版本 | 异常 | 问题 |
|---|---|---|
| ng1.0.0 | 均衡(~16.7%每个) | 信号太弱无法区分算法 |
| ng1.0.1 | lgb=1%, cb=1%, hgb=35% | 中等倾斜 |
| ng1.0.3 | xgb=50%, rf=1% | 明显倾斜 |
| **ng1.0.4** | **rf=94%**, 其余5个≤1.8% | **严重问题** — 单算法独裁 |
| ng1.0.7 | xgb=1%, cb=1%, rf=32% | 中等倾斜 |

**根因**: ICIR优化在有限WF窗口(3个)上容易过拟合到单一算法。ng1.0.4的rf=94%意味着模型实质上退化为单一RandomForest。

---

## 迭代方向建议

### 方向1: 精简因子集 (高确信度，推荐优先执行)

**目标**: 去掉多版本验证无用的因子，减少噪声，参考ng1.0.3的成功经验。

具体操作:
- **立即移除 (3个)**: `roe_change`, `n_sectors_strong`, `days_since_breakout`
- **ng1.0.7的5个全零ext_market因子移除**: `market_ret_60d`, `market_vol_ratio`, `breadth_momentum_5d`, `market_skewness_20d`, `liquidity_stress` — 保留有效的 `amv_var1`, `amv_macd`, `amv_regime_days`
- **合并冗余industry因子**: `industry_breadth` + `sector_breadth_vs_market` 信息已被 `industry_return_20d` 覆盖

**预期效果**: 减少8-10个因子→降低模型复杂度→减少过拟合→算法权重更均衡

### 方向2: 增强个股alpha因子 (中等确信度)

**问题**: 14个核心因子中8个是market组，个股alpha信号不足。

可探索的方向:
- **深化cs_rank组**: cs_rank_turnover(7.1%)是最成功的新因子，可考虑增加行业内动量变化率、行业内alpha排名
- **强化资金流因子**: amv_var1(4.2%)首次进入即为Top6，说明活跃筹码有信息增量。可探索个股级AMV(而非全市场)
- **估值-质量交互**: roe_ttm(4.2%)是唯一核心基本面，但与pe_ttm(1.1%)的交互(如PEG)可能有增量

### 方向3: 修复算法权重极端倾斜 (高确信度)

**问题**: ng1.0.4的rf=94%说明ICIR优化过拟合。

可选方案:
- **权重下限约束**: ensemble权重≥5%或≥10%，防止算法被完全丢弃
- **增加WF窗口数**: 从3个增加到5个，使ICIR估计更稳定
- **Shrinkage**: 将ICIR优化权重与等权(1/N)做加权平均(如70%ICIR+30%等权)

### 方向4: 降低market组依赖 (长期方向)

**现状**: market组从81%降到21%已是显著改善，但仍是最大单一组。

可探索:
- **Market-neutral标签**: 使用 `个股收益 - 市场收益` 而非 `个股收益 - 行业收益`，从标签层面消除market因子的择时贡献
- **Market因子正交化**: 训练时对stock因子做market因子的残差化，强制模型从stock因子中提取信号
- **风险**: 完全去market可能降低牛熊择时能力(ng1.0.6的AMV牛熊切换正依赖此能力)

### 优先级排序

| 优先级 | 方向 | 预期工作量 | 预期收益 | 风险 |
|---|---|---|---|---|
| P0 | 精简因子集 | 小(改config) | 中(减噪声+过拟合) | 低 |
| P1 | 修复算法权重倾斜 | 小(改优化逻辑) | 中(模型稳健性) | 低 |
| P2 | 增强个股alpha因子 | 中(新特征工程) | 高(个股区分度) | 中(新因子可能无效) |
| P3 | 降低market依赖 | 大(标签/训练框架) | 高(alpha纯度) | 高(可能降低择时) |

---

## 各版本因子详情

### NG v1.0.0 (62 特征, baseline)

**WF**: 10d IC=0.069, ICIR=0.515 | **标签**: absolute return

**Top 10 (占 82.5%)**: vix_proxy(15.9%), market_drawdown(12.0%), market_volatility_20d(11.5%), northbound_flow_5d(11.4%), market_return_5d(7.8%), market_volume_ratio(6.3%), market_new_high_ratio(5.1%), market_momentum_diff(4.9%), market_return_20d(4.6%), roe_ttm(2.9%)

**问题**: Top 9全是market因子(81.3%)。近零因子21个。

### NG v1.0.1 (69 特征, +cs_rank +residual)

**WF**: 10d IC=0.091, ICIR=0.931 | **标签**: industry_excess_return

**Top 10 (占 61.5%)**: market_volatility_20d(10.8%), vix_proxy(10.1%), relative_strength_vs_peers(6.3%), industry_relative_strength(5.6%), roe_ttm(5.5%), market_return_20d(5.4%), market_new_high_ratio(5.1%), northbound_flow_5d(4.4%), market_drawdown(4.3%), turnover_rate(4.1%)

**改善**: cs_rank(11%) + residual(14%)引入后market降至38.4%。近零9个。

### NG v1.0.3 (41 特征, 精简版 — 最高效率)

**WF**: 10d IC=0.077, ICIR=1.014 | **标签**: industry_excess_return

**Top 10 (占 64.4%)**: industry_return_20d(11.9%), cs_rank_turnover(11.0%), vix_proxy(10.5%), roe_ttm(7.5%), idiosyncratic_volatility(5.5%), market_volatility_20d(4.7%), northbound_flow_5d(4.0%), turnover_rate(3.6%), log_adv_20d(3.2%), market_return_20d(2.4%)

**特点**: 近零仅1个(roe_change)。Top10覆盖5个组，分布最均匀。15d ICIR=1.229为全系列最高。

### NG v1.0.4 (75 特征, +smoothing)

**WF**: 10d IC=0.066, ICIR=0.935 | **标签**: industry_excess_return (RA penalty=1.5)

**Top 10 (占 39.9%)**: vix_proxy(6.9%), cs_rank_turnover(6.5%), industry_return_20d(6.0%), market_volatility_20d(3.5%), roe_ttm(3.4%), northbound_flow_5d(3.1%), trend_strength_20d(3.0%), idiosyncratic_volatility(2.5%), ma60_distance(2.5%), log_adv_20d(2.5%)

**特点**: 权重分散度最佳(Top10仅39.9%)。但rf=94% ensemble问题严重。近零4个。

### NG v1.0.7 (81 特征, +AMV +cond_ix)

**WF**: 10d IC=0.082, ICIR=1.059 | **标签**: industry_excess_return

**Top 10 (占 54.1%)**: industry_return_20d(11.6%), cs_rank_turnover(9.9%), market_return_20d(6.2%), roe_ttm(5.1%), vix_proxy(4.9%), amv_var1(4.2%), turnover_rate(3.5%), market_volatility_20d(3.0%), pullback_to_ma10(2.7%), idiosyncratic_volatility(2.6%)

**特点**: amv_var1直接进Top6。ext_market的8个因子中5个全零(信息已饱和)。近零8个。

---

## Ensemble 算法权重

### 10d Target

| 算法 | ng1.0.0 | ng1.0.1 | ng1.0.3 | ng1.0.4 | ng1.0.7 |
|---|---|---|---|---|---|
| lgb | 16.7% | 1.0% | 13.6% | 41.9% | 29.6% |
| xgb | 16.8% | 27.2% | **50.2%** | 1.0% | 1.0% |
| cb | 16.7% | 1.0% | 13.7% | 41.7% | 1.0% |
| rf | 16.6% | 27.3% | 1.0% | **94.1%** | 32.4% |
| hgb | 16.6% | 35.5% | 9.9% | 1.0% | 21.1% |
| lgb_rank | 16.6% | 8.0% | 11.5% | 1.8% | 14.9% |

### 自适应 Target 权重

| 版本 | 3d | 5d | 10d | 15d |
|---|---|---|---|---|
| ng1.0.0 | 22.9% | 20.9% | 24.2% | 32.0% |
| ng1.0.1 | 18.2% | 24.2% | 26.9% | 30.7% |
| ng1.0.3 | 19.4% | 21.7% | 26.6% | 32.3% |
| ng1.0.4 | 21.3% | 22.0% | 26.4% | 30.3% |
| ng1.0.7 | 21.5% | 23.8% | 26.8% | 27.8% |

---

## 相关页面

- [NG 系列详解](ng-series.md) — 版本改动与性能
- [特征指南](../features/feature-guide.md) — 因子含义与计算方式
- [ML 管线](../architecture/ml-pipeline.md) — 训练流程
