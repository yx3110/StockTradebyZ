# ng1.1.1 因子迭代设计（EMT 四关验证 + 清理+新增）

- **版本代号**：ng1.1.1
- **日期**：2026-04-13
- **硬目标**：MaxDD < -10% AND Sharpe > 2.5（纯因子路径，不叠加风控）
- **Base**：ng1.0.1 bugfix（66 特征）+ 继承 ng1.1.0 的 P0 移除 + P1 权重 shrinkage
- **迭代策略**：清理现役 + 四关筛选候选新因子

---

## 1. 背景与动机

### 1.1 ng1.0.1 bugfix vs ng1.1.0 的 paradox
wiki 数据表明 ng1.0.1 bugfix（66 特征）在 Sharpe=2.753、年化=165.7%、MaxDD=-11.7%，**反而强于 ng1.1.0**（68 特征，Sharpe=2.065, 年化=122.8%, MaxDD=-12.5%）。ng1.1.0 的 P0 精简"无实际影响"（被移除因子 importance≈0）、P1 权重均衡化 ensemble 极差 34.5→3.9pp（唯一有效改动）、P2 四新因子"效果中性"未经四关验证、P3 市场正交化边际。

### 1.2 现有工具：EMT 四关验证框架
`EastMoneyTrader/analysis/feature_validator.py` 提供：
- **Gate 1 单因子 IC**：`|ic_mean|≥0.02 AND |ICIR|≥0.3 AND 胜率≥55% AND n≥30`
- **Gate 2 五分位分组回测**：单调 + 多空年化≥5% + 多空 Sharpe≥1
- **Gate 3 冗余/正交性**：max_corr<0.8 AND 正交化后 IC 保留≥50%
- **Gate 4 增量 LGB 价值**：带/不带候选 80/20 切分，ΔIC≥0.005
- 决策：Gate1 必过；全过=ACCEPT；过 N-1=MARGINAL；其它=REJECT

### 1.3 硬目标可达性说明
wiki 记录只有 ng1.0.5（SL=6% + regime-gate-aggressive + VT=20% + CPPI(F0.08,M20)）达到 MaxDD=-9.1%，那是叠加风控。纯因子路径达 MaxDD<-10%+Sharpe>2.5 主要依赖**淘汰 regime-unstable 因子 + 加入低波动/低尾部风险因子降低尾部损失**。如果最终指标不达标，此次迭代仍产出 ng1.1.1 候选，由用户决定是否合入生产。

---

## 2. 架构：两阶段 Funnel + 重训

```
ng1.0.1 base 66 feat (bugfix 后)
     ↓ 继承 ng1.1.0 P0 移除 3 近零因子
[63 现役特征]
     ↓ Phase 1: EMT Gate 1+3 审计 (现役特征)
[~50-55 精简特征]
     ↓ Phase 2: 14 个候选新因子 × EMT Gate 1-4
[~53-60 增强特征]  (只留 ACCEPT；MARGINAL 也 REJECT)
     ↓ Phase 3: 重训 (继承 ng1.1.0 P1 权重 shrinkage) + WF + 三层评估
ng1.1.1 模型
```

### 2.1 继承的改动
- **ng1.0.1 bugfix 3 个冗余移除**：`volume_contraction`（=volume_ratio_5d）、`sw_index_return_5d`（=industry_return_5d）、`industry_relative_strength`（=residual_return_20d）
- **ng1.1.0 P0 近零移除 3 个**：`roe_change`（5 版本全近零）、`n_sectors_strong`（3 版本全零）、`days_since_breakout`（信号被 adx_proxy 替代）
- **ng1.1.0 P1 权重 shrinkage**：ensemble 权重 = 70% ICIR 自适应 + 30% 等权，floor≥5%

### 2.2 不继承的改动
- ng1.1.0 P2 四因子（cs_rank_pb、cs_rank_dv、peg_proxy、pb_roe_ratio）— 未经四关认证，按"MARGINAL 也 REJECT"原则一并下架，若通过 Phase 2 四关再加回
- ng1.1.0 P3 市场正交化 — wiki 明确"边际效果"且行业超额标签已部分中性化 market

---

## 3. Phase 1：现役特征清理审计

### 3.1 审计范围
对 63 个现役特征（59 股票 + 10 市场 − 3 P0 − 3 bugfix）逐个跑 EMT Gate 1+3。Gate 2/4 成本高（Gate 4 每个特征几分钟 LGB 训练），现役特征已间接通过 ng1.0.1 训练验证，只做严格复查。

### 3.2 淘汰规则
淘汰任一触发：
- **Gate 1 不过**（阈值：`|ICIR|<0.3 OR |ic_mean|<0.01 OR 胜率<52%`）
  - 比 EMT 默认（ICIR≥0.3, ic_mean≥0.02, 胜率≥55%）稍松 — **Phase 1 用于淘汰明显无用因子，避免误伤有用 market 因子（截面 IC 本就小）**。Phase 2 候选筛选继续用 EMT 默认严格阈值。
- **Gate 3 不过**（阈值：max_corr≥0.85 AND 正交化后 IC 保留<50%）
  - 比 EMT 默认 max_corr<0.8 稍松，因为 NG 基座本就有 17 对 corr≥0.8 的兄弟，全砍会动到核心因子；用 0.85 作为 Phase 1 冗余阈值，只处理高度冗余的。冗余对中只保留 IC 更强的一个。
- **市场特征特殊处理**：截面 IC 无意义（每日常数），跳过 Gate 1；用 `gain_importance<0.5% AND SHAP rank 后 30%` 作为淘汰信号（参考 EMT audit 工具 `compute_gain_importance` + `compute_shap_importance`）

### 3.3 预期淘汰清单（待验证）
基于 ng-factor-quality wiki 历史审计：
- `industry_rank_return_5d`（ng1.1.0 audit 为 **全 horizon WEAK**）
- `industry_breadth`、`sector_breadth_vs_market`、`sector_volume_vs_market`（3 版本均值<0.2%）
- `volume_ratio_5d`（4 版本均值 0.267%，高度相关 volume_breakout）
- `industry_hhi` 若 IC 不过关
- ng1.0.1 17 对冗余里的弱侧（待四关定量判断）

### 3.4 审计产出
`reports/ng111/phase1_audit.csv`：列 = feature / ic_mean / ic_ir / pct_positive / max_corr / redundant_with / decision (KEEP/DROP)。
`reports/ng111/phase1_audit.md`：人类可读摘要 + 淘汰清单 + 理由。

### 3.5 止损
如果淘汰特征 < 3 个 → 清理阶段价值低，**直接进 Phase 2**；仍产出报告归档。

---

## 4. Phase 2：候选新因子四关筛选

### 4.1 候选池（14 个，全部零数据成本）

| # | 候选名 | 定义 | 数据源 | 理论/经验支撑 |
|---|---|---|---|---|
| F1 | `amount_acceleration` | d(log(amount_ma5))/Δt | daily_quotes | ng1.0.7 `amv_var1` 全 horizon STRONG，贡献 4.2% |
| F2 | `turnover_volatility_20d` | std(turnover)/mean(turnover), 20d | daily_basic | `cs_rank_turnover` 是 7.1% 核心；波动是互补维度 |
| F3 | `close_to_ma60_pct` | close/MA60 - 1 | technical_indicators | ng1.0.4 `ma60_distance` 贡献 2.5% |
| F4 | `up_days_ratio_20d` | 上涨天数 / 20 | daily_quotes | 动量持续性，与点位动量互补 |
| F5 | `moneyflow_net_5d_z` | (net_mf_5d - μ_60) / σ_60 | moneyflow_daily | ng1.1.0 遗产 8.8M 行数据，zscore 标准化是关键 |
| F6 | `signal_trust_score_60d` | signal_trust 系统输出 | signal_trust 表 | 新上线的假信号过滤器 |
| F7 | `return_skew_20d` | skew(daily_return, 20d) | daily_quotes | 负偏（左尾暴跌多）关联 MaxDD 目标 |
| F8 | `illiq_amihud_20d` | mean(\|return\|/amount, 20d) | daily_quotes | 经典 Amihud (2002) 非流动性溢价 |
| F9 | `log_close_vs_vwap_20d` | log(close / VWAP_20d) | daily_quotes | 相对成交均价位置 |
| F10 | `max_drawdown_60d` | (max_price - min_price) / max_price, 60d | daily_quotes | 回撤状态，直接关联 MaxDD 目标 |
| F11 | `beta_to_market_60d` | cov(r_i, r_mkt)/var(r_mkt), 60d | daily_quotes | 低 beta 偏好降组合波动 |
| F12 | `accrual_ratio` | (net_profit - ocf) / total_assets | financial_indicator | 经典 Sloan (1996) 应计项因子 |
| F13 | `industry_alpha_20d` | r_i - r_industry_mean, 20d 累积 | daily_quotes + industry | 行业内相对收益（residual_return_20d 是 market 残差，区别） |
| F14 | `overnight_return_20d` | mean((open_t - close_{t-1}) / close_{t-1}), 20d | daily_quotes | 隔夜收益捕捉消息驱动信号 |

### 4.2 验证流程
每个候选：
1. 从 StockTradebyZ 数据源计算该因子，shift(1) 避免未来函数，输出 `(date, stock_code, value)` CSV
2. 调用 `EastMoneyTrader/scripts/validate_feature.py --candidate <csv> --name <f_name> --target 10d`（主 target 用 10d，因为生产 focus_days=10-15）
3. 记录 Gate 1-4 结果 + decision

### 4.3 采纳规则（严格模式）
- **只采纳 ACCEPT**（4/4 关全过）
- **MARGINAL 也 REJECT**（因为硬目标严，任何一关不过都可能是信号不稳）
- 若一个候选在多个 target（5d/10d/15d）都 ACCEPT → 优先加入
- 若两个 ACCEPT 候选互相高相关（Gate 3 说的 max_corr）→ 只加 IC 更强的

### 4.4 审计产出
- `reports/ng111/phase2_candidates/<feature_name>.md`：每个候选一份 EMT summary
- `reports/ng111/phase2_summary.md`：全候选对比表 + 最终 ACCEPT 清单
- `reports/ng111/phase2_accepted_features.json`：待加入 ng1.1.1 的因子定义

### 4.5 止损
- 如果 0 个 ACCEPT → ng1.1.1 = 纯清理版（继续 Phase 3）
- 如果 1-2 个 ACCEPT → 正常
- 如果 ≥3 个 ACCEPT → 记录稀有事件（EMT 文档说头部量化淘汰率 95%），Phase 3 前人工复核一遍避免数据泄露

---

## 5. Phase 3：重训与评估

### 5.1 训练配置
- **Trainer**：`ml_models/ng/ng_trainer.py`（继承 ng1.1.0 的 P1 shrinkage）
- **Version flag**：`--version ng1.1.1`
- **Feature cache 表**：新建 `ng111_feature_cache`，schema 与 ng101 一致（通过 `_cache_version='ng1.1.1'` 区分存储格式）
- **训练窗口**：`--start-date 2020-01-01 --purge-days 15`
- **Base model type**：与 ng1.0.1 一致（LGB + XGB + CatBoost + RF + HGB + LambdaRank），使用 ensemble 权重 shrinkage 70%/30%

### 5.2 三层评估

**Layer 1 — WF OOS**
- 训练自动跑 walk-forward，输出 3 窗口 ICIR（3d/5d/10d/15d）
- 目标：10d ICIR ≥ 0.93（不低于 ng1.0.1），15d ICIR ≥ 1.10（不低于 ng1.0.1 的 1.060）

**Layer 2 — Pre-2020 跨 regime 泛化** (2026-04-20 订正)
- 用训练完的模型反推 2018-2019 的 OOS 预测
- ~~目标：V5.2 ≥ 60% A 级（ng1.0.1 Pre-2020 是 73.7% A+，不能大幅退步）~~
- **新目标: 同向 alpha — 净年化 ≥ 0% AND 超额胜率 ≥ 60%** (ng1.0.1 bugfix 实测 Pre-2020 V5.2=45.5% B, 净年化 -19%, 之前的 73.7% 是 4-10 评估 bug 遗留 ghost number, 不能作为泛化基线)

**Layer 3 — 生产级评估（主要判据）**
- 命令：`python3 backtest/run_north_star_eval.py --production --version ng1.1.1`
- 目标（硬约束）：MaxDD < -10% AND Sharpe > 2.5
- 如果未达标：记录最接近的配置，不 merge 为生产默认

### 5.3 决策矩阵

**比较基准**：所有"退步 / 不退步"判断对比 **ng1.0.1 bugfix**（Sharpe=2.753, MaxDD=-11.7%, 年化=165.7%, V5.2=72.1-73.4% A+, 10d WF ICIR=0.931, 15d=1.060, ~~Pre-2020 V5.2=73.7% A+~~ **Pre-2020 V5.2=45.5% B 实测 2026-04-20**）。

| 结果 | 决策 |
|---|---|
| 达成硬约束（MaxDD<-10% AND Sharpe>2.5）AND 信号质量不退步（10d WF ICIR ≥ 0.93 AND Pre-2020 V5.2 ≥ 60% A 级） | Merge 为新生产默认，更新 MEMORY/wiki |
| 未达硬约束，但 Sharpe>2.5 OR MaxDD<-10% 至少一项达成 | 作为候选保留，wiki 记录权衡，不切生产默认 |
| 硬约束一项都未达成 AND Sharpe <= ng1.0.1 bugfix 的 2.753 AND MaxDD >= ng1.0.1 bugfix 的 -11.7% | 放弃 ng1.1.1，仅保留审计报告作为因子池知识沉淀 |

---

## 6. 实施顺序与时间估算

| Step | 内容 | 时长 |
|------|------|------|
| 1 | EMT 四关工具与 StockTradebyZ 数据库接入验证（跑一个 sanity 候选） | 0.5d |
| 2 | Phase 1 现役 63 特征 Gate 1+3 审计，产出淘汰清单 | 0.5d |
| 3 | Phase 2 计算 14 候选因子 + 四关验证（每个 ~5-15min） | 1-1.5d |
| 4 | Phase 3 训练 ng1.1.1 + WF OOS + Pre-2020 + 生产回测 | 3-6h |
| 5 | 评估 + 决策 + wiki/MEMORY 更新 | 0.5d |
| **总计** | | **2.5-3d** |

---

## 7. 关键文件清单

### 新增
- `ml_models/ng/ng111_feature_calculator.py` — 新因子计算器（只写被 ACCEPT 的因子）
- `scripts/audit_ng111_current_features.py` — Phase 1 现役特征审计脚本
- `scripts/validate_ng111_candidates.py` — Phase 2 候选 14 因子批量验证脚本
- `reports/ng111/` — 全部审计报告

### 修改
- `ml_models/ng/ng_cache_updater.py` — 新增 ng1.1.1 cache_version 分支
- `ml_models/ng/ng_trainer.py` — 新增 `--version ng1.1.1` 分支（features_json 字段对齐）
- `ml_models/ng/ng_schema.py` — 如需新表
- `tomorrow_stock_selector.py` — SCORER_REGISTRY 注册 ng1.1.1
- `fetch_data/quick_daily_update.py` — daily update 同步更新 ng1.1.1 cache

### 不动
- ng1.0.1 / ng1.0.3 / ng1.0.4 / ng1.0.8 / ng1.0.6 / ng1.1.0 — 全部保留可复现
- 其它版本的 cache 表 — 全部保留

---

## 8. 风险与 Mitigation

| 风险 | 概率 | Mitigation |
|------|------|------------|
| 四关淘汰现役核心特征（如 roe_ttm） | 低 | Gate 1 阈值放宽到 0.01（默认 0.02），Market 特征不走 Gate 1 |
| 候选 0 ACCEPT → ng1.1.1 ≈ 清理版 | 中（头部量化淘汰率 95%） | 候选池扩到 14，期望 ACCEPT 率 > 0 |
| Phase 2 某候选有未来函数 | 中 | 所有候选强制 shift(1)；Gate 1 IC 异常高（>0.15）时人工复查 |
| 新因子计算 bug 导致 audit 虚高 | 低 | Phase 2 前 sanity 跑一个已知的 `close_to_ma60_pct`（ng1.0.4 已验证贡献 2.5%），预期至少进 MARGINAL 以验证工具链 |
| MaxDD<-10% 达不到 | 高 | 决策矩阵明确"未达标可不 merge"，且保留所有中间结果 |
| 重训破坏现有生产系统 | 低 | ng1.1.1 cache 独立表，生产 selector 显式按 version 路由，互不影响 |

---

## 9. 成功标准

### 必须达成
- [ ] Phase 1 审计报告完整（63 特征全覆盖）
- [ ] Phase 2 14 候选全部跑完四关并归档
- [ ] ng1.1.1 模型文件 + 特征清单 + 训练日志完整归档
- [ ] 三层评估（WF / Pre-2020 / 生产）全部产出

### 期望达成（决定是否 merge 为生产默认）
- [ ] MaxDD < -10%
- [ ] Sharpe > 2.5
- [ ] 10d WF ICIR ≥ 0.93
- [ ] Pre-2020 V5.2 ≥ 60% A 级

### 知识资产沉淀（无论模型是否 merge 都必须完成）
- [ ] 更新 `docs/wiki/models/ng-series.md` 加入 ng1.1.1 章节
- [ ] 更新 `docs/wiki/models/ng-factor-quality.md` 加入 Phase 1/2 审计结论
- [ ] 更新 MEMORY 简短一行指向本 spec + 结果
- [ ] 14 候选因子的 EMT 四关结果永久归档，作为未来候选池的参考
