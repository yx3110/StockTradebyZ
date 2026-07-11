# NG v1.5.0 Design Spec: Regime-Refined + MaxDD Hardening

**Date**: 2026-04-20
**Status**: Draft → Pending User Review
**Base**: ng1.4.0 (73 features, Stage 3.5 PASS 68%, Stage 4a pending) / ng1.0.1 (当前生产, WF-OOS 73.4% A+)
**Goal**: V5.2 ≥ 75% AND MaxDD ≤ -15% AND Pre-2020 年化 ≥ 0%, **同时达成跨 regime 稳健**
**Codename**: "RREG" (Regime-Refined with CVaR Gate)

---

## 0. 核心洞察 (2026-04-20 审计总结)

今天一次完整审计把 NG 系列的真实地形图画清楚了。订正后的数字:

### 各版本实测 (统一口径, 10d hold, V5.2, 当前评分卡)

| 版本 | 核心创新 | WF-OOS V5.2 | WF-OOS MaxDD | Pre-2020 年化 | Pre-2020 V5.2 | β_UMD |
|---|---|---:|---:|---:|---:|---:|
| **ng1.0.1** (4-12 bugfix) | 66 特征 MSE 单头 | 73.4% A+ | **-11.7%** ⭐ | **-19.0%** | 45.5% B | +0.38 |
| **ng1.0.6** (外部 regime) | 牛→ng101, 熊→ng104-3s | **78.9% A+** ⭐ | -22.9% | **+0.7%** ⭐ | 41.1% C | **+0.005** ⭐ |
| ng1.0.3 | ng101 - 翻转因子 | — | — | -33.4% | 42.5% C | -4.26 |
| ng1.0.4 | 75 特征 + RA label | 78.1% A+ | -32.6% | -9.2% | 38.8% C | -1.65 |
| ng1.0.7 | 条件标签 + 18 市场特征 | 76.8% A+ | — | -35.7% | 34.7% C | -2.57 |
| ng1.1.0 | ng101 精简 + 4 P2 | 70.4% A+ | -12.5% | — | — | — |
| ng1.2.x (0/1/2/3/4) | 改 loss/label | 41-53% C | -54%~-99% | 22-35% | — | — |
| ng1.3.0 | Dual-head (excess+downside) | 68% (S3.5) / 48% (S4a) | — | — | — | — |
| **ng1.4.0** | ng101 + Tier A (4 downside + 3 AMV) | 68% (S3.5 PASS) | pending | pending | pending | pending |

### 五条硬洞察

**I1. Regime 意识是真 alpha 来源**
ng1.0.6 (外部 regime switch) 在 Pre-2020 是**唯一**正收益的版本 (+0.7%, Sharpe +0.18)。β_UMD=+0.005 比 ng1.0.1 的 +0.38 干净 76 倍。机制是: 牛市用 ng1.0.1 (动量+), 熊市用 ng1.0.4-3s (动量-), 净 β_UMD ≈ 0。这不是巧合, 是 regime 动态切换带来的结构性清洁。

**I2. MaxDD 和跨 regime 泛化是两个正交短板**
- MaxDD 最小: ng1.0.1 (-11.7%)
- 跨 regime 最稳: ng1.0.6 (Pre-2020 +0.7%)
- 两者**互斥** — ng1.0.6 MaxDD 差 ng1.0.1 一倍, ng1.0.1 Pre-2020 年化 -19%

**I3. Label/Loss 层创新独立行动全部失败**
- ng1.2.0 margin hinge: V5.2=41.8% C, Sharpe 负
- ng1.2.1 vol-norm rank: V5.2=40% C
- ng1.2.2 quintile CE: V5.2=41.8% C
- ng1.2.3 contrarian + mined + soft downside: WF 45% / Pre-2020 31%
- ng1.2.4 极保守 mf 增量: Stage 3.5 V5.2=48.5% B
- ng1.0.4 RA label (penalty 1.5): Pre-2020 38.8% C 不泛化
- ng1.0.7 conditional label + AMV: Pre-2020 34.7% C 不泛化

共同点: **不改特征只改标签/Loss = 自废武功**。

**I4. Dual-head 不是银弹**
ng1.3.0 Stage 3.5 PASS 68% 但 Stage 4a REJECTED 48%, β ablation 证实 β 不关键。dual-head 的 Pareto 冲突严重, excess head 和 downside head 在牛市/熊市权重分配永远打架。

**I5. 特征层 + 单头 + MSE + industry excess 是最稳的底座**
所有成功版本 (ng1.0.1, ng1.0.6, ng1.4.0) 都遵循: 单头回归, MSE loss, industry excess 标签, 只在特征层做创新。失败版本 (ng1.2.x, ng1.3.0) 都打破了这个结构。

---

## 1. 方向候选对比

基于 I1-I5 洞察, 列了 6 个候选方向, 逐一评估:

| 方向 | 描述 | 违反的洞察 | 风险 | 潜在收益 |
|---|---|---|---|---|
| A. Internalize regime (单模型 + regime 特征) | 把 AMV 特征喂给单模型, 学 regime 依赖 | — | 中 (ng1.0.7 同路线已败) | 中-高 |
| B. Fine-grained regime switch | ng1.0.6 扩展到 3-4 状态 (牛/震荡/熊急/熊缓) | — | 高 (切换成本+overfit) | 低-中 |
| C. Multi-task 3-head (excess/downside/regime) | ng1.3.0 方向继续深化 | I4 (dual-head 已败) | 高 | 低 |
| D. Regime-aware ensemble (ng101+ng106 composite) | 双模型并行, 动态加权 | — | 低 | 中 |
| E. CVaR loss regularization | 单模型 + tail-risk 惩罚 loss 项 | I3 (改 loss) | 中-高 (ng1.2.x 教训) | 中 |
| **F. ng1.4.0 base + Tier B regime-refined 特征 + CVaR (stage B)** | 特征层扩展 + 可选 loss 层增强 | 最少 | **低** | **高** |

### 选中: 方向 F

**为什么选 F**:
- 完全遵循 I5 (特征层创新优先)
- ng1.4.0 已训完, Stage 3.5 PASS, 继承已验证底座
- CVaR 作为可选 Stage B, 主路径失败才启用, 风险隔离
- 预期兼得 ng1.0.1 的 MaxDD 优势 + ng1.0.6 的 regime 稳健

**为什么不选 A/B/C/D/E**:
- A: ng1.0.7 已证伪 "内化 regime 特征" 单步不够, 需要更精细的 regime 特征设计
- B: 切换成本 + overfitting, 方向 F 的 5 个新特征里已包含 regime-refinement 精华
- C: ng1.3.0 证伪, 不重复
- D: 不是模型迭代, 是组合层 overlay (保留为 ng1.5.x patch 或 ng1.5.0 失败后的 fallback)
- E: 直接改 loss 风险高 (I3), 降级为 ng1.5.0 Phase B 可选

---

## 2. ng1.5.0 技术设计

### 2.1 特征层 (78 总, ng1.4.0 73 + 5 新 Tier B)

**继承 ng1.4.0 基础 (73 特征)**:
- ng1.0.1 base: 56 stock + 10 market
- ng1.4.0 Tier A: 4 downside (current_drawdown, downside_vol_20d, recovery_speed_20d, gap_risk_20d) + 3 AMV (amv_var1, amv_macd, amv_regime_days)

**新增 ng1.5.0 Tier B (5 特征, 精选 regime-refine + MaxDD-hardening)**:

| 特征名 | 类别 | 计算 | 作用 |
|---|---|---|---|
| `amv_regime_bull_prob` | market regime | logit(var1/ma60 - 1) + logit(amv_macd); 0-1 软概率 | 替代硬 0/1 regime, 让模型学渐进过渡 |
| `industry_regime_agreement` | industry regime | 本股行业 5d 收益 vs 大盘 5d 收益方向一致性 (60d 滚动) | ng1.0.6 "牛→行业跟涨" 成功的关键内化 |
| `recent_maxdd_60d` | risk | 60d 窗口内当前价/窗口高点-1 (和 current_drawdown 20d 互补) | 直接给模型 MaxDD 敏感信号 |
| `volatility_skew_20d` | risk | downside_vol_20d / (upside_vol_20d + ε) | 下行波动占比, RA 标签失败后的替代 |
| `upside_capture_60d` | regime + risk | 大盘涨日本股涨幅 / 大盘涨幅 (60d 均值) | 牛市跟涨能力, 识别 "熊市只跌, 牛市不涨" 的陷阱股 |

**排除清单 (禁止加)**:
- ❌ Contrarian 因子 (ng1.2.3 教训)
- ❌ Mined 算子暴力搜索 (ng1.2.3 教训, top-30 0 命中)
- ❌ Moneyflow 细分 (ng1.3.0 ablation 证无用)
- ❌ Market cap features (ng1.0.1 翻转因子已删)

### 2.2 标签层 (不动)

- `label_3d`, `label_5d`, `label_10d`, `label_15d` = **industry excess return** (ng1.0.1 原样)
- 拒绝 RA label (I3: ng1.0.4 失败), conditional label (I3: ng1.0.7 失败)

### 2.3 训练层 (继承 ng1.0.1)

- **6-algo MSE ensemble**: LGB / XGB / CB / RF / HGB / LGB_Rank
- **3-seed**: 42, 123, 456 (memory 里 seed 传播 bug 已修)
- **Walk-forward**: Auto-WF (turbo-check 3 配置: expanding / sliding-720d / sliding-500d+decay730), 选 10d ICIR 最优
- **ICIR 自适应权重**: 同 ng1.0.1
- **purge_days**: 15 (覆盖 label_15d horizon)
- **`--target-parallel 4`**: M5 Max 实测 1.38x 加速

### 2.4 Phase 划分

**Phase A (主路径)**: 特征层升级, 不动 loss
- 预期效果: V5.2 +2~5pp, MaxDD -2~4pp, Pre-2020 +5~15pp 年化
- 风险: 低 (只加特征, 单头 MSE 不改)
- 成功判据: Stage 4a V5.2 ≥ 75%, MaxDD ≤ -15%, Pre-2020 年化 ≥ 0%

**Phase B (可选, 仅 Phase A 未达 MaxDD 目标时启用)**: CVaR 正则化
- Loss = MSE(y, ŷ) + λ · CVaR_5%(per-stock squared loss)
- λ 网格: {0 (回退), 0.1, 0.3} — 不做 0.5+ (overfit 风险)
- 只在 label_10d/15d 加 (3d/5d 对 MaxDD 不敏感)
- 触发条件: Phase A V5.2 达标 但 MaxDD > -18%

**Phase C (fallback, 仅 Phase A+B 均失败)**: 降级到 ng1.5.0-ensemble
- ng1.5.0 模型 + ng1.0.1 模型并行评分, 用 amv_regime_bull_prob 动态加权
- 相当于 "内置 ng1.0.6"
- 不是首选 — 是 safety net

### 2.5 Schema + Cache

**新建** `ng150_feature_cache` 表 (不复用 ng130, 避免 schema 污染):
- 基础列: code, trade_date, features_json, 4 labels
- 新增: label_raw_Nd × 4 (ng1.0.3+ 延续)
- 不加: downside/ra/cond 标签 (ng1.5.0 只用 industry excess)

**Schema 注册**:
```python
VERSION_TABLE_MAP['ng1.5.0'] = 'ng150_feature_cache'
SCHEMA_VERSION_MAP['ng1.5.0'] = 'ng1.5.0'  # 自己的 schema
```

Cache 回填时间预算: 2018-01-01 → 2026-04-20 约 8 年, 单日 ~4s → **40-60 min** 顺序 (Pool 死锁风险, 用 `--num-workers 0`)。

### 2.6 锁死的设计约束 (不许改)

1. ✅ Industry excess label (I3, I5)
2. ✅ 单头架构 (I4)
3. ✅ MSE loss (Phase A 阶段)
4. ✅ 6-algo ensemble
5. ✅ 3-seed ensemble
6. ✅ Auto-WF 模式选择
7. ✅ purge_days=15
8. ✅ β 归因 artefact 写入 pkl (Check 9 硬性要求)
9. ✅ git_commit_hash + host + duration 写入 pkl (Check 9)
10. ✅ 每步完成后 3 轮 `/simplify` (feedback_simplify_after_each_step.md)

---

## 3. 成功准则 (Pre-flight Check 4 写死)

### 3.1 最小成功 (必须达到, 否则不切生产)

- WF-OOS V5.2 ≥ 72% (不低于 ng1.0.1 73.4% × 0.98)
- WF-OOS Sharpe ≥ 2.5 (不低于 ng1.0.1 2.75 × 0.9)
- WF-OOS MaxDD ≤ -15% (允许 ng1.0.1 -11.7% × 1.3)
- Pre-2020 净年化 ≥ 0% (**跨 regime 同向 alpha, 严格优于 ng1.0.1 的 -19%**)
- β_UMD ≤ 0.5 (不严于 ng1.0.1 0.38)
- β_SMB ≤ 1.3 (不严于 ng1.0.6 1.54)

### 3.2 目标成功 (推荐切生产门槛)

- WF-OOS V5.2 ≥ **75%** A+ (超 ng1.0.1 1.6pp)
- WF-OOS Sharpe ≥ **2.8** (超 ng1.0.1 0.05)
- WF-OOS MaxDD ≤ **-13%** (超 ng1.0.1 -11.7% 恶化 <1.3pp)
- Pre-2020 净年化 ≥ **+5%** (超 ng1.0.6 +0.7% 6pp)
- β_UMD ≤ **0.25**
- Alpha t ≥ 5.0 (保持 ng1.0.1 水平)

### 3.3 理想成功 (Stretch)

- WF-OOS V5.2 ≥ **78%** A+ (超 ng1.0.6 78.9% 几乎持平)
- WF-OOS Sharpe ≥ **3.0**
- WF-OOS MaxDD ≤ **-10%** (优于 ng1.0.1)
- Pre-2020 净年化 ≥ **+10%**
- **同时** 达到 ng1.0.1 (MaxDD) + ng1.0.6 (V5.2/Pre-2020) 的全部优势

---

## 4. 实施阶段 + Gates (严格顺序)

### Stage 0: Pre-flight (1h)

10 项 Pre-flight Check 全部通过 (CLAUDE.md 规定):
- [ ] Check 1: Schema 一致性 (DB ⇔ Training ⇔ Scorer 三方对齐)
- [ ] Check 2: Backfill 逻辑正确 (5 个新特征的公式 grep, 无 `shift(-`)
- [ ] Check 3: 训练最高效路径 (auto-WF + target-parallel 4 启)
- [ ] Check 4: Acceptance criteria (本 spec 3.2 节)
- [ ] Check 5: Baseline 对比表建好 (ng1.0.1 / ng1.4.0 / ng1.5.0)
- [ ] Check 6: Checkpointing (caffeinate + tee log + WF checkpoint)
- [ ] Check 7: 数据泄露预扫 (β 初验 + shift grep)
- [ ] Check 8: 资源预算 (df -h, ps aux, RAM)
- [ ] Check 9: Pkl 元数据 (git_commit_hash + host + seed + duration)
- [ ] Check 10: trainer + cache_updater 3 轮 `/simplify`

**用户要看到的 Pre-flight 输出格式**:
> ✅ Check 1-10 全部通过, kickoff 条件成立

### Stage 1: Feature Engineering (2-3h)

实现 5 个 Tier B 特征 in `ml_models/ng/ng_feature_calculator.py`:
- 在 `build_features()` 添加 5 个计算函数
- 单因子 IC 验证 (60d 滚动, 全 4 horizon): **|IC| ≥ 0.02** 留, 否则删
- 因子间 Pearson 相关性 < 0.8 (否则去冗余)
- **ABORT trigger**: 5 个特征中 3 个 |IC| < 0.02 → 方向 F 失败, 转 Phase C ensemble

### Stage 2: Cache Backfill (40-60 min)

```bash
python3 ml_models/ng/ng_cache_updater.py \
    --start-date 2018-01-01 --end-date 2026-04-20 \
    --version ng1.5.0
```

覆盖率检查: avg ≥ 1200 股票/天 (和 ng101 覆盖率一致)。
**ABORT**: 若覆盖率 < 1000 → 特征计算 bug, 不训。

### Stage 3: Fast-check (2 min)

```bash
python3 ml_models/ng/ng_trainer.py --version ng1.5.0 --seed 42 \
    --fast-check --target-parallel 4
```

2 WF 窗口 × start=2022, 看 10d OOS IC 方向。
- **PASS**: 10d OOS IC ≥ 0.05 且方向正
- **ABORT**: IC < 0.03 或 负方向 → 特征设计有问题

### Stage 4: Full train 3-seed (6-10h)

```bash
for SEED in 42 123 456; do
  caffeinate -i python3 ml_models/ng/ng_trainer.py \
    --version ng1.5.0 --seed ${SEED} --target-parallel 4 \
    2>&1 | tee logs/ng150_seed${SEED}_$(date +%Y%m%d_%H%M%S).log
done
```

每 seed 约 45-60 min (cache 命中后)。**3 个 seed 串行跑** (避免 WF 窗口的 ThreadPoolExecutor 互相抢锁)。

### Stage 4.5: Sanity (15 min)

- **Seed 传播**: 3 seed pred_10d 两两 corr ∈ [0.85, 0.95]
- **Schema 一致**: cache 列名 = trainer `feature_names` = pkl 里 `feature_names` (三方全等)
- **β artefact**: `python3 backtest/factor_returns.py --version ng1.5.0` 写入 `ml_models/trained_models/ng/factor_attribution_ng150.json`, 验证 β_UMD ≤ 0.5

### Stage 5: Stage 3.5 Gate (45 min, 2025 全年)

```bash
python3 scripts/ng130_stage35_gate.py --version ng1.5.0 \
    --start 2025-01-01 --end 2025-12-31
```
(需给 gate 脚本加 `--version` 通用参数支持, ng1.4.0 plan 也有这个依赖)

**Gate**: V5.2 ≥ **68%** (ng1.4.0 Stage 3.5 水平)
**ABORT**: V5.2 < 60% → 彻底失败, 转 Phase C ensemble

### Stage 6: Stage 4a Gate (1h, 2024-2026 跨 regime)

```bash
python3 scripts/ng130_stage35_gate.py --version ng1.5.0 \
    --start 2024-01-01 --end 2026-04-17
```

**Gate**:
- V5.2 ≥ 75% (目标成功)
- MaxDD ≤ -15% (目标成功)
- Sharpe ≥ 2.8 (目标成功)
- 月度 10d IC 负比例 < 25%

**若 V5.2 ≥ 72% 但 MaxDD > -18%** → 启动 **Phase B: CVaR 正则化**
**若 V5.2 < 70%** → 转 Phase C ensemble

### Stage 7: Stage 4b Gate (2h, Pre-2020)

```bash
python3 backtest/batch_generate_v395_reports.py --version ng1.5.0 \
    --model-path {最新 pkl} \
    --start-date 2018-01-01 --end-date 2019-12-31 \
    --output-dir reports/daily_selection_ng150_pre2020 --force
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_ng150_pre2020 \
    --label NG150-PRE2020 --scoring-version ng1.5.0 \
    --rank-field composite --top-n 10 --focus-days 10 --score-version v52
```

**Gate** (同向 alpha):
- 净年化 ≥ 0%
- Sharpe ≥ 0
- V5.2 ≥ 50% (ng1.0.1 45.5% + 5pp)

**若 Pre-2020 年化 < -5%** → Pre-2020 退化严重, 应降级为 ng1.5.0-conditional (只牛市用), 不切生产。

### Stage 8: 观察期 (1-2 周)

并行 ng1.0.1 + ng1.5.0 日报对比:
- 单股 picks 相关性 ≥ 0.5 (避免完全不同选股, 降低切换风险)
- 行业分布偏差 < 20pp (单行业占比差距)
- 无"惊吓股" (突然出现 ng1.0.1 未进 Top50 但 ng1.5.0 进 Top10 的情况, 人工审查 3-5 个样本)

### Stage 9: 切换生产

修改 `ml_models/ng/ng_schema.py`:
```python
PRODUCTION_VERSION = 'ng1.5.0'  # 从 'ng1.0.1' 改
```

更新:
- `CLAUDE.md` ML Scoring Systems 段
- `MEMORY.md` 顶部 + `ng_production_switch.md`
- `docs/wiki/models/ng-series.md`
- `docs/wiki/log.md` (加 model 类条目)

---

## 5. 风险矩阵

| # | 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|---|
| R1 | 5 个新特征有数据泄露 (shift(-) / 同日窥视) | 低 | 高 | Stage 0 Check 7 预扫 + Stage 4.5 β 验证 |
| R2 | `amv_regime_bull_prob` 用到未来 regime | 低 | 高 | 公式严格 t-1 截面, code review |
| R3 | 回填死锁 (Pool pipe buffer) | 低 | 中 | 顺序 `--num-workers 0`, ng_cache_updater 默认已是 |
| R4 | Pre-2020 仍负年化 | 中 | 中 | Stage 4b ABORT + 接受"不切生产但保留 artefact" |
| R5 | MaxDD > -18% | 中 | 高 | Phase B CVaR 启动 (fallback 机制) |
| R6 | 换手率 > 50x (ng1.0.1 是 45x) | 低 | 中 | 叠加 ng1.0.8 sell50 overlay |
| R7 | 3-seed pred corr < 0.85 (seed bug) | 低 | 高 | sanity_tests.py 强制检查 (ng1.3.0 已有) |
| R8 | Stage 4a V5.2 < 70% | 中 | 高 | Phase C fallback = ng1.5.0 + ng1.0.1 ensemble |
| R9 | 训练超 10h | 中 | 低 | target-parallel 4 + 3 seed 串行, 总 6-8h 估计内 |
| R10 | Pkl 缺 metadata (git_commit_hash) | 低 | 低 | Stage 0 Check 9 代码预检 |

---

## 6. 备选降级路径

### Phase B: CVaR 正则化 (触发: Stage 6 V5.2 达标 但 MaxDD > -18%)

- 在 `ng_trainer.py` 的 LGB/XGB/HGB 损失里加 CVaR 惩罚 (RF/CB 不改, 因为库内不好加)
- LGB 自定义 `fobj` 返回 grad = MSE_grad + λ * CVaR_grad_per_stock
- λ = 0.1 (不要太大, ng1.2.x margin 教训)
- 只在 label_10d/label_15d 加 (horizon 越长 MaxDD 越敏感)
- 估时: 改 trainer 2h + 重训 6h + eval 2h = 10h

### Phase C: Regime-Aware Ensemble (触发: Stage 6 V5.2 < 70% 或 Phase B 也失败)

- 不训新模型, 只做 composite 层 overlay
- 公式: `final_score = w_bull * ng101_score + w_bear * ng106_score` 其中 `w_bull = amv_regime_bull_prob`
- 实现在 `ml_models/ng/ng_production_scorer.py` 的一个新 scorer class `NG150EnsembleScorer`
- 估时: 2-3h (纯代码, 无训练)
- 预期: 兼得 ng1.0.1 (MaxDD) + ng1.0.6 (V5.2) 部分优势, 但比不上真正的 Phase A 成功

---

## 7. 不做什么 (根据教训显式排除)

| 不做 | 理由 |
|---|---|
| Dual-head 架构 | ng1.3.0 Stage 4a 48% 已证伪 (I4) |
| Contrarian/mined factors | ng1.2.3 双崩 (I3) |
| RA label (penalty 1.5) | ng1.0.4 Pre-2020 C 级不泛化 (I3) |
| Conditional label | ng1.0.7 Pre-2020 C 级不泛化 (I3) |
| Margin / quintile CE loss | ng1.2.0/1.2.2 均 C 级 (I3) |
| Moneyflow 细分 | ng1.3.0 ablation top-30 0 命中 |
| Fast-check 作 go/no-go 主决策 | ng1.0.9 ICIR 虚高教训 |
| IC screening 筛选特征 | ng1.3.0 数据泄露路径 |
| Mined 算子暴力搜索 | ng1.2.3 64d IC 0.73 → 370d 0.28 小样本假象 |
| 追求 V5.2 ≥ 80% | 过拟合风险, 目标成功 75% 已够 |
| 训练时改 purge_days | 15 已足够覆盖 label_15d |

---

## 8. 时间预算 + 依赖

**总预算**: **18-25h** (不含 Phase B/C 降级)

| 阶段 | 估时 | 关键依赖 |
|---|---|---|
| Stage 0 Pre-flight | 1h | 代码 review |
| Stage 1 特征实现 | 2-3h | ng_feature_calculator.py 修改 + IC 验证 |
| Stage 2 Cache 回填 | 40-60min | 磁盘 ~300MB, DB lock 时间 |
| Stage 3 Fast-check | 2min | — |
| Stage 4 Full train | 6-10h | `--target-parallel 4`, 3 seed 串行 |
| Stage 4.5 Sanity | 15min | β artefact 要 factor_returns.py 跑通 |
| Stage 5 Stage 3.5 | 45min | 需 ng130_stage35_gate.py 加 --version 通用支持 |
| Stage 6 Stage 4a | 1h | 同上 |
| Stage 7 Stage 4b | 2h | Pre-2020 报告生成 |
| Stage 8 观察期 | 1-2 周 | 人工 |
| Stage 9 切生产 | 30min | 文档更新 |

**Phase B** 若触发: +10h
**Phase C** 若触发: +3h

---

## 9. 开工前 Prerequisites

在 ng1.5.0 kickoff 之前, 建议先完成:

1. **ng1.4.0 Stage 4a 评估** (pending, 已有 552 reports 但未跑 gate)
   - 如果 ng1.4.0 Stage 4a **PASS V5.2 ≥ 70%** → **ng1.4.0 可能已经够, ng1.5.0 可以降级为"ng1.4.0 + 2 精选新特征"小迭代**
   - 如果 ng1.4.0 Stage 4a **FAIL** → ng1.5.0 必须做全量 (5 新特征 + Phase B 备选)
   - 估时: 1h (Stage 4a + 4b gate)

2. **ng130_stage35_gate.py `--version` 通用化** (ng1.4.0 plan 也提过的依赖)
   - 目前硬编码 ng1.3.0, 改为接受任意 version + scorer 版本
   - 估时: 1h

3. **pkl 元数据 Check 9 硬化** (当前 ng101 pkl 就缺 git_commit_hash)
   - `ng_trainer.py` 在 pkl dump 前写入 `{git_commit_hash, host, training_duration_sec, seed, schema_version, feature_names}`
   - 估时: 30min

---

## 10. 一句话概括

**ng1.5.0 = ng1.4.0 稳定底座 + 5 个 regime-refined 特征 + MSE 单头不动 + Phase B CVaR 正则化 fallback**, 目标同时拿下 ng1.0.1 的 MaxDD 优势 + ng1.0.6 的跨 regime 稳健, 不重复 ng1.2.x/1.3.0 已证伪的 loss/label/dual-head 陷阱。

**End of spec.**
