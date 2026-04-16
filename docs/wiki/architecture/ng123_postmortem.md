# ng1.2.3 Postmortem

**Date**: 2026-04-16
**Status**: ❌ REJECTED — `PRODUCTION_VERSION` 保持 `ng1.0.1`
**Spec**: `docs/superpowers/specs/2026-04-14-ng123-design.md`
**Plan**: `docs/superpowers/plans/2026-04-14-ng123-implementation.md`

## 决策摘要

ng1.2.3 在双向北极星评估中**双崩**:

| 指标 | ng1.0.1 baseline | ng1.2.3 WF-OOS | ng1.2.3 Pre-2020 |
|---|---|---|---|
| **V5.2 总分** | **72.1%** A+ | **45%** | **31%** |
| **年化收益** | +165.7% | **-8.8%** | **-63.3%** |
| **Sharpe** | 2.753 | **-0.36** | **-2.29** |
| **超额年化** | +130%+ | **-11.2%** | **-49.3%** |
| **超额 MaxDD** | -11.7% | **-60.6%** | **-69.9%** |

WF-OOS 应≥73% 接受（spec §8.1），实际 45%（差 28pp）。Pre-2020 应≥70%，实际 31%（差 39pp）。

## 关键 fast-check 结果（这些都通过了！）

| Stage | 结果 |
|---|---|
| Stage 1 (Moneyflow IC) | ✅ PASS — 6 factors at ng101 median+ percentile |
| Stage 2 (Mined Factors) | ✅ PASS — 14/30 cross-regime stable, top 6 selected |
| Stage 3 (λ Ablation) | ✅ PASS — λ=0.6 selected (3 of 3 above threshold) |
| **Training WF ICIR (3 seeds)** | **10d ICIR=1.26 vs ng1.0.1 baseline 0.93 (+35%)** |

**核心矛盾**: WF 训练 ICIR 极高 (1.26)，但 production 信号在回测中产生**负超额**。

## Root Cause Analysis

### 假设 1: λ=0.6 downside penalty 过重 ✦ 最可能

Spec §5.2 警告 ng1.0.4 用 λ=1.5 失败（rank order 翻号）。Stage 3 ablation 选 λ=0.6 仅基于 WF training ICIR 最大化（1.266 vs 0.0=1.159），未在 OOS production 模拟中验证。

λ=0.6 实际上：
- excess=+0.05, downside=0.10 → label = 0.05 - 0.06 = **-0.01**（翻号）
- excess=+0.10, downside=0.20 → label = 0.10 - 0.12 = **-0.02**（翻号）

A 股很多上涨股票路径波动较大（downside 0.10-0.20 普遍），λ=0.6 把多数"涨股"标记为负 label。模型学到"避免上涨股票"。

WF ICIR 高是因为 in-sample 模型学到了 transformed label 的方向；但 OOS production 用 raw return 衡量，模型预测和真实涨跌方向相反。

### 假设 2: Mined factors 全部 sign-flipped (contrarian) 累积反向偏差

6 mined factors 全部 `sign_flip=True`（A 股 contrarian）。组合后可能形成 systematic 反向预测。

### 假设 3: Moneyflow factors 也是 contrarian (Stage 1 发现)

6 moneyflow factors IC 均为负，本质 contrarian。叠加 mined 也 contrarian → 信号高度同方向，但模型在 production 中 score 反向。

## What We Learned

### ✅ 工程层面（成功的部分）

1. **Pipeline 健全**: 8 个真 bug 在 fast-check 阶段就被抓到（schema 泄漏、cache key、ordering trap、gates miscalibration、mining JSON 丢字段、cx_* 泄漏 etc）
2. **优化效果显著**: R1 simplify 让 Phase 2 backfill 从估计 8-11h 降到 2.5h（4× 加速）
3. **3-seed 一致性**: 三个 seed WF ICIR 差异 <0.005，说明随机性不是问题源
4. **fast-check 流程价值**: 提前发现 Stage 1 IC 是 contrarian 的事实, 但**未提前发现 production 信号会反向**

### ❌ 方法论层面（核心教训）

1. **Stage 3 λ ablation 用错指标**: 优化 WF training ICIR 而非 OOS production score。training ICIR 高不等于 production 收益高。下次应在 production 模拟（小样本回测）上选 λ。
2. **Contrarian 因子组合是危险信号**: 当所有候选因子都是 contrarian (negative IC) 时，模型组合后可能产生 systematic 反向 signal。需要**单因子 production 验证**而非聚合 ICIR。
3. **Stage 2 用 raw return 而非 industry-excess**: simplify R1 反思过这点（保留 raw return）。可能这是 Stage 2 PASS 但 production 失败的 root cause 之一。
4. **Spec §6.4 决策门槛只看 fast-check ICIR**: 应该加上**production 回测 spot check**（小样本 30-90 天）作为 Stage 4 gate，而非直接进 Phase 2 全量 backfill。

### 🎯 改进 ng1.2.4 的建议

1. **保留 6 moneyflow 因子作为 contrarian signal**（IC 一致负确实有信号）
2. **去掉所有 mined factors**（contrarian 叠加反向）
3. **去掉 downside label penalty**（用纯 industry-excess label）
4. **加 Stage 3.5 production spot check**: 训完后跑 90-day 回测，看 V5 估算分 ≥ 60% 才进 Phase 2
5. **重新做 Stage 3 ablation**: 不在 WF ICIR 选 λ，在 90-day production 回测的 V5 上选

## Artifacts Preserved

| 路径 | 内容 |
|---|---|
| `ml_models/trained_models/ng/ng123_seed{42,123,456}_*.pkl` | 3 个训练好的模型 (~77MB each) |
| `ml_models/ng/ng123_*.py` | 4 个 ng123 模块（label transform, moneyflow, mined, drop helpers）|
| `ml_models/ng/tests/test_ng123_*.py` | 152 个单元测试 |
| `scripts/ng123/*.py` | Stage 1/2/3 验证脚本 + decision aggregator |
| `data_adapter/stock_data.db:ng123_feature_cache` | 1880 dates × 71 features × ~1960 stocks/day = 3.69M rows |
| `reports/daily_selection_ng1.2.3_wf_oos/` | 1520 daily reports (2020-2026) |
| `reports/daily_selection_ng1.2.3_pre2020/` | 360 daily reports (2018-2019) |
| `reports/ng123/fastcheck/decision.md` | Fast-check Go decision (条件 A) |
| `logs/ng123_*.log` | 完整训练 + 评估日志 |

## 诊断访问

```bash
# Run ng1.2.3 scorer on any date for diagnosis (PRODUCTION 不受影响)
python3 tomorrow_stock_selector.py 2026-04-15 --scoring-version ng1.2.3
```

## Production Status

`ml_models/ng/ng_schema.py:34` `PRODUCTION_VERSION = 'ng1.0.1'` 保持不变 ✅
- 现存 ng1.0.1 模型继续 daily 选股
- daily_update.py + tomorrow_stock_selector.py 不受影响
- 0 production downtime, 0 user-facing impact

## 累计修复的 8 个真 bug（永久价值）

虽然 ng1.2.3 失败，但过程中修复的 bugs 都对 ng1.0.1 production 也有益：

1. ng1.2.1 vn_label schema 泄漏 (ng_schema.py)
2. cx_* 从 ng1.0.7 漏进 ng1.2.3 (ng_cache_updater.py)
3. dv_ratio filter-before-read ordering (ng_cache_updater.py)
4. Stage 1 gates miscalibration (ng101 baseline 校准)
5. factor_mining dedup IndexError (factor_mining_pipeline.py)
6. factor_mining JSON 缺 spec 字段 (factor_mining_pipeline.py)
7. NGTrainer cache key 不含 λ (ng_trainer.py + train_v395_multi_target.py)
8. cache_updater operands 6× 重复 → 1× (4× backfill 加速)

## 时间消耗总结

| Phase | 计划 | 实际 |
|---|---|---|
| Phase 0 (setup) | ½ day | ½ day |
| Phase 1.1 (12 mf factors + tests) | 1 day | 1 day |
| Phase 1.2-1.3 (fast-check) | 1 day | 2 days (含 7 个 bug 修复) |
| Phase 2 (backfill) | 6-8h | **2.5h** (R1 优化省 5h) |
| Phase 3 (training) | 2-6h | 4h |
| Phase 4 (eval + decision) | ½ day | ½ day |
| **总计** | **4-5 days** | **~5 days** |
