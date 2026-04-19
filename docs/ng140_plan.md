# ng1.4.0 + ng1.3.1 路线图 — 2026-04-20

**触发**: ng1.3.0 Stage 4a REJECTED (V5.2=48% 跨 regime), β ablation 证实 β 不是瓶颈。
**目标**: 先做 B (ng1.4.0), 失败则做 C (ng1.3.1 regime-conditional)。
**生产状态**: 保持 `ng1.0.1` (V5.2=72.1% A+) 直到新版本全通 Stage 3.5 + 4a。

---

## Phase B: ng1.4.0 — ng1.0.1 基座 + Tier A 新特征 (优先, ~4-5h)

### 设计

把 ng1.3.x 学到的**有用部分**搬到 ng1.0.1 稳定底座, 不要 ng1.3.x 的不稳定部分。

**保留 ng1.0.1**:
- 66 特征 (56 stock + 10 market) 基础集
- Industry excess labels (跨 regime 稳定的标签)
- 6-algo MSE ensemble (lgb/xgb/cb/rf/hgb/lgb_rank)
- 3-seed 平均 (seed 42/123/456)
- ICIR 自适应权重

**新增 (Tier A 验证有用)**:
- 4 downside stock features: `current_drawdown, downside_vol_20d, recovery_speed_20d, gap_risk_20d`
- 3 AMV market features: `amv_var1, amv_macd, amv_regime_days`

**弃用 (ng1.3.x 验证无用)**:
- ❌ Dual-head 架构 (Stage 4a 证伪)
- ❌ β composite (ablation 证伪: β=0 vs β=0.3 只差 2pp)
- ❌ Downside label swap (downside head 对 2024 泛化失败)
- ❌ 3 moneyflow features (sanity 证实 0/10 top-30 命中)
- ❌ 行业中性化 at inference (可后加)

**最终 ng1.4.0**: 56 stock + 4 downside + 10 market + 3 AMV = **73 特征**

### 执行步骤

#### B.1 Schema + 缓存复用 (~30min)

复用 `ng130_feature_cache` 表 (已有 4 downside + AMV, 2020-2026 1522 天覆盖)。
- `ng_schema.py`: 新增 `ng1.4.0 → 'ng130_feature_cache'` 映射 (schema 层共用)
- `NG140_STOCK_FEATURES` = `STOCK_FEATURE_NAMES + NG130_TIER_A_DOWNSIDE` (56 + 4, 不包含 mf)
- `NG140_MARKET_FEATURES` = `MARKET_FEATURE_NAMES + NG130_TIER_A_AMV` (10 + 3)
- `NG140_ALL_FEATURES` = 73 total

**Pre-flight 验证**: Schema 列与 cache 一致 (ng130 cache 已有这些列)。

#### B.2 Trainer 配置 (~30min)

`ng_trainer.py`:
- 在 `version_feature_table` 加 `('ng1.4.0', NG140_ALL_FEATURES, NG140_STOCK_FEATURES, NG140_MARKET_FEATURES, [])`
- 在 `load_data` / `prepare_features` 的 ng1.3.x 分支前加 ng1.4.x 分支
- **不走 dual-head 路径** (是单头 excess)
- 保留 ng1.0.1 的 6-algo ensemble + 3-seed

#### B.3 Fast-check (~2min)

```bash
python3 ml_models/ng/ng_trainer.py --version ng1.4.0 --fast-check --target-parallel 4
```

验证: 特征矩阵 shape = `(N, 73)`, 2-window WF 10d OOS IC > 0.05 方向正。

#### B.4 Full train 3 seeds (~3-4h, 并行训练)

```bash
for SEED in 42 123 456; do
  python3 ml_models/ng/ng_trainer.py --version ng1.4.0 --seed ${SEED} --target-parallel 4
done
```

Cache 命中 (同 ng130 表), 每 seed ~45-60 min。

#### B.5 Stage 3.5 Gate (~45min)

```bash
python3 scripts/ng130_stage35_gate.py --version ng1.4.0 --start 2025-01-01 --end 2025-12-31
```

Gate 脚本需要加 `--version` 参数支持 (目前 hardcoded ng1.3.0)。

#### B.6 Stage 4a Gate — 关键判断点 (~1h)

```bash
python3 scripts/ng130_stage35_gate.py --version ng1.4.0 --start 2024-01-01 --end 2026-04-17
```

**决策点**:
- **V5.2 ≥ 70%** → PASS, 继续 Stage 4b + 生产切换准备
- **V5.2 65-70%** → 与 ng1.0.1 相当, 但不明显优势, 保持生产
- **V5.2 < 65%** → FAIL, 说明 ng1.3.x 特征本质有问题, 放弃 B 转 C

### B 成功标准

- Stage 3.5 V5.2 ≥ 65% (2025)
- Stage 4a V5.2 ≥ 70% (2024-2026, 跨 regime)
- 稳定性: 月度 10d IC 负 IC 比例 < 25% (ng1.3.0 Stage 4a 是 33%)

### B 总时间预算

~4-5h (Schema 1h + Train 3.5h + Gate 2h)

---

## Phase C: ng1.3.1 — regime-conditional weights (备选, ~3h)

**触发条件**: B 失败 (V5.2 < 65%) 或 B 成功但差距明显 (V5.2 65-70%, 想再榨一把 ng1.3.x)。

**不重训**, 只改推断层。

### 设计

ng1.3.0 pkl 已有每个 WF 窗口的 OOS 预测和 ICIR。用 AMV regime 信息将 ICIR 权重拆分:

```
算法权重 per regime:
  bull:     {lgb: w_bull_lgb, xgb: w_bull_xgb, ...}
  sideways: {lgb: w_side_lgb, ...}
  bear:     {lgb: w_bear_lgb, ...}
```

推断时: 查当天 AMV regime → 用该 regime 的算法权重组合 6-algo 预测。

### 执行步骤

#### C.1 Per-regime ICIR 分析 (~1h)

- Load 3 seeds × 2 heads 的 pkl
- 在 2024-2026 OOS 窗口对每个算法 × target × regime 计算 IC
- 计算 per-regime ICIR
- 输出: `ml_models/ng/ng131_regime_weights.json`

#### C.2 NG130DualHeadScorer 改造 (~1h)

- `__init__` 加载 `ng131_regime_weights.json`
- `predict()` 里获取每天的 AMV regime
- 用 per-regime 权重替代当前 ICIR 权重
- 缺省 regime = sideways (安全选项)

**注意**: 不改 pkl, 只改 scorer 层。所有 ng1.3.0 pkl 继续可用。

#### C.3 Stage 4a gate + 对比 (~1h)

```bash
python3 scripts/ng130_stage35_gate.py --version ng1.3.1 --start 2024-01-01 --end 2026-04-17
```

**决策点**:
- V5.2 ≥ 55% (从 48% 提升 7+pp) → regime-conditional 有效, 继续打磨
- V5.2 < 55% → ng1.3.x 死, 接受结论

### C 总时间预算

~3h (Analysis 1h + 改造 1h + Gate 1h)

---

## 执行顺序

```
开始 B (ng1.4.0)
├── 成功 (V5.2 Stage 4a ≥ 70%) → Stage 4b + 生产切换 (停在这里, 不做 C)
├── 部分成功 (65-70%) → 停在这里, 不做 C (ng1.0.1 已够好)
└── 失败 (<65%) → 做 C (ng1.3.1)
    ├── 成功 (V5.2 Stage 4a ≥ 55%) → 记为 ng1.3.x 可行方向, 但不是生产替代
    └── 失败 (<55%) → 最终放弃 ng1.3.x, 接受 ng1.0.1 作为长期生产
```

---

## ABORT 触发 (任一命中停止当前 phase)

### B (ng1.4.0):
1. Fast-check 10d OOS IC < 0.03 → 说明新特征没信号, 放弃
2. Full train 中 WF window 1 的 10d OOS IC < 0.05 → 放弃本 seed
3. Stage 3.5 V5.2 < 55% → B 彻底失败, 跳 C

### C (ng1.3.1):
1. Per-regime ICIR 分析显示跨 regime IC 差异 < 0.02 → regime 信号太弱, 放弃 C
2. Stage 4a V5.2 < 50% → C 无效, 彻底放弃

---

## 成功后的下一步

### 如 ng1.4.0 PASS:
1. Stage 4b (Pre-2020 cross-regime) — 需 ng130_feature_cache 回填 2018-2019
2. 1-2 周生产观察 (ng1.4.0 vs ng1.0.1 并行评分对比)
3. 切 `PRODUCTION_VERSION = 'ng1.4.0'`
4. 更新 CLAUDE.md + MEMORY.md + docs/wiki

### 如 ng1.3.1 (regime-conditional) PASS:
1. 不直接上生产 (ng1.4.0 是更干净的路线)
2. 把 regime-conditional 机制提炼成独立模块, 可用于 ng1.4.0 + 未来版本

---

## 文档状态

- Plan: 本文件
- Postmortem PASS (Stage 3.5): `reports/ng130/stage35_pass.md`
- Postmortem REJECTED (Stage 4a): `reports/ng130/stage35_rejected_v2.md` (最新版覆盖, 需保存)
- Memory: `ng130_stage35_pass.md`, `ng130_stage35_v2_rejected.md` (过期, 需更新)

---

**End of plan.**
