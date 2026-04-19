# ng1.3.1 系统排查 + 效率提升计划

> **生成日期**: 2026-04-19
> **触发**: ng1.3.0 Stage 3.5 REJECTED (V5.2=60.6% < 65% gate)
> **目标**: 在 ng1.3.1 重训前一次性排查所有已知/潜在问题, 同时大幅提升训练 + 评估效率

---

## 📋 Part 1: 已知问题 (必修)

### P0: 阻塞 ng1.3.1 训练正确性

#### P0-1. `prepare_features` 覆盖 NG130 stock_feature_cols
**症状**: NGTrainer init 设 `self.stock_feature_cols = NG130_STOCK_FEATURES (63)`, 但训练日志显示 "56 stock + 18 market + 7 cond_ix" — 说明被 V485Trainer.prepare_features 重置.

**根因**: `prepare_features` 是 V485Trainer (parent class) 方法, 它根据 version_ge 检查重新构造 `active_market_cols = NG107_MARKET_FEATURES if version >= ng1.0.7`, 同时也对 stock_feature_cols 做了 STOCK_FEATURE_NAMES fallback。

**修法**: 在 NGTrainer override prepare_features (或在调用前 patch self.feature_names + stock_feature_cols)。具体:
```python
def prepare_features(self, df):
    if _is_1_3_branch(self._ng_version):
        # 不让父类用 NG107/NG110 fallback 覆盖
        saved_stock = self.stock_feature_cols
        saved_macro = self.macro_feature_cols
        saved_feat_names = self.feature_names
        result = super().prepare_features(df)
        # 强制恢复 NG130 配置
        self.stock_feature_cols = saved_stock
        self.macro_feature_cols = saved_macro
        self.feature_names = saved_feat_names
        return result
    return super().prepare_features(df)
```

或更彻底: 让 NGTrainer 的 `_get_active_stock_features` 显式返回 NG130_STOCK_FEATURES, 不依赖 super().

**验证**: 训练日志必须显示 "63 stock + 13 market + 0 cond_ix = 76 total"。

#### P0-2. β 从未做 WF 网格搜索 (Phase 5)
**症状**: composite scorer 用默认 β=0.3, 没有按 spec §8 WF 网格搜索 {0.1, 0.2, 0.3, 0.4, 0.5}。

**修法**: 写 `scripts/ng130_beta_search.py` (plan 中已设计):
- 5 个 β × 3 WF windows = 15 backtests
- 选 max(WF Sharpe) 满足 cv<0.5 + 2/3 Top-2 约束
- 写入 `reports/ng130/beta_search.md` + 更新 ng130_composite.DEFAULT_BETA

**预估**: 2-4h (主要是 5×242天 报告生成时间)。

#### P0-3. 训练日志 vs pkl feature_names 不一致
**症状**: pkl `feature_names` 列了 81 项 (含 Tier A/B + ext_market 5), 但实际 booster 在前 56 stock + 18 market + 7 cond_ix 上训练. Inference 时 scorer 给 81 列, booster 拿前 N 列 → 错位.

**根因**: ng_trainer.py:1546 `model_data['feature_names'] = self.feature_names` — self.feature_names 在 init 时是 NG130_ALL_FEATURES (含未训练的 Tier A/B), 但训练实际用的是 prepare_features 重置后的列表.

**修法**: 在 pkl save 前同步 `self.feature_names = self.stock_feature_cols + self.macro_feature_cols + self._cond_ix_cols` (即训练后实际用的)。

---

### P1: 影响 ng1.3.1 评估准确性

#### P1-1. `label_raw_*` 写入但不读取 (dead column)
**症状**: ng_cache_updater 写 4 个 label_raw_*d 列, 训练 / scorer 都不读. 浪费 SQLite 空间 ~30%。

**修法 (low priority)**: 评估完成后从 schema 移除, 或新版本不再写。

#### P1-2. `--target-parallel` 是 dead parameter
**症状**: 命令行 `--target-parallel 4` 设了 trainer._target_parallel, 但全文 grep 该属性只有这一处, 没读取. 实际仍是 serial 4-target 训练.

**修法**: 实现 ThreadPoolExecutor 并行 4 targets:
```python
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=self._target_parallel) as ex:
    futures = {target: ex.submit(self._train_target, target, ...) for target in ['3d','5d','10d','15d']}
    results = {t: f.result() for t, f in futures.items()}
```

**注意**: M5 Max 实测 1.38x 加速 (memory中已有). 但需 thread-safe 的 random_seed 处理 (sklearn RF / lgb 的 random_state 是 instance-level OK)。

**预估**: 7h training → 5h. 但 multi-seed loop 串行总时间 (3 seeds × 5h) = 15h vs 21h.

#### P1-3. EMT validator 采样偏差 (500 stocks vs 6000)
**症状**: 当前 EMT 只采样 500 A股 (按 code 字典序前 500). 这偏向 000xxx 类深圳股票, 排除大量 600xxx 上证 / 30xxxx 创业板。

**修法**: 改成随机采样 500 (with seed) 或全 universe scan 加采样后 IC bootstrap。

#### P1-4. Stage 3.5 1 年窗口高方差
**症状**: 242 天 spot check 给 V5.2 raw 60.6%, 但 plan §9.2 提到 ng1.2.4 教训: 61 天 IC 0.73 → 370 天 IC 0.28 (假象)。242 天可能仍不够稳定。

**修法**: gate 窗口扩展到 2024-01 → 2026-04 (~16 个月, 320+ days), 或干脆并发跑 Pre-2020 + WF-OOS 替代 Stage 3.5 (无 spot check)。

---

### P2: 长期技术债

#### P2-1. NG130DualHeadScorer 不验证 6 个 pkl feature_names 一致
**症状**: scorer 只从 `models[(seeds[0], 'excess')]` 取 feature_names, 假设其他 5 个一致. 若混合不同版本 pkl (e.g. 一个 ng1.3.0 + 一个 ng1.3.1) 会静默错位.

**修法**: __init__ 末尾 assert 6 个 pkl feature_names 完全一致。

#### P2-2. compute_ng130_mf_factors 与 ng123_moneyflow_factors aggregate 重复
**症状**: 30 行重复 buy/sell 求和逻辑. 两者签名差异 (输入 dict 字段) 让合并复杂。

**修法 (defer)**: 抽 `ml_models/ng/mf_aggregation.py` 共用 helper. 等 ng1.4.x 系列时一并做。

#### P2-3. quick_daily_update.py 未集成 ng1.3.x 缓存更新
**症状**: `update_ng_feature_cache()` 循环只 refresh `[PRODUCTION_VERSION, 'ng1.0.3', 'ng1.0.4']`. 切到 ng1.3.0 production 前需手动加 'ng1.3.0' 到 versions list, 否则当天特征缺失.

**修法**: 加 'ng1.3.0' 到 versions 数组, 或用 SCORER_REGISTRY 动态枚举 active versions。

---

## 📋 Part 2: 潜在问题 (推测/未验证)

### S0: 数据完整性

#### S0-1. label swap 可能与 V485 内部 label 转换交互
**疑点**: ng_trainer.py:993-1001 swap downside_Nd → label_Nd. 但 V485Trainer 在 walk_forward_train 内部可能再做 residual / industry-excess conversion. 若发生, downside head 的 label 实际是 "downside excess return industry-residual" (语义不明)。

**验证方法**: 在 swap 后 + walk_forward_train 调用前后, log `result['label_10d'].describe()`. 比较 mean/std 看是否变化.

#### S0-2. prepare_features 对 cs_rank/residual 的重新计算
**疑点**: prepare_features 内部可能重新计算 cs_rank_*, residual_* 等, 用 active_market_cols 而不是 self.macro_feature_cols. 如果 cs_rank 计算依赖未训练的 ext_market features, 结果不准。

**验证**: 对比 ng1.0.1 和 ng1.3.0 的 cs_rank_volatility 输出一致性。

#### S0-3. AMV ffill 跨 stock/date 边界
**疑点**: load_data 对 market features 做 `result.sort_values('trade_date'); for col: result[col] = ffill()`. AMV features 已是 cross-sectional 同值, ffill 不影响. 但若某日 AMV 计算失败 (market_amv 表 gap), ffill 会用前一天值, 可能掩盖数据缺失。

**验证**: SELECT * FROM market_amv WHERE trade_date BETWEEN ... AND ... GROUP BY trade_date HAVING COUNT(*)=0; 找 gap.

### S1: 模型行为

#### S1-1. composite rank tie-breaking
**疑点**: `pd.Series.rank(pct=True)` 默认 method='average'. 平局时给均值, 不是 first/dense. 在跨 stock 评分时若多 stocks pred_excess 完全相等, rank 会重叠 → composite_rank 不严格 1-N。

**验证**: 输出某天 scored DataFrame 看 composite_rank 是否 1, 2, 3, ... 还是有重复。

**修法**: scorer 用 `rank(method='first')` 强制 break ties。

#### S1-2. Seed diversity 异常 (2 个 seed corr=0.97)
**症状**: corr(123, 456)=0.97 略超出 [0.85, 0.95] 上界. 可能 seed=123 和 seed=456 在 cb/rf algos 上行为相似 (deterministic 部分高)。

**验证**: 分别看 lgb/xgb/cb/rf 的 corr per seed, 找哪个 algo 主导。

#### S1-3. xgb 在 NG130DualHeadScorer 偶发失败 (DMatrix 类型问题)
**症状**: 之前 sanity test 显示 xgb predict 报错 "Expecting data to be a DMatrix object". scorer 已加 try/except 跳过, 但 xgb 失败意味着该算法权重被 fallback 到等权或其他 algos 比例放大。

**修法**: scorer._predict_head 对 xgb 显式构造 DMatrix。已在 Round 1 提到但未实施。

### S2: Eval 框架

#### S2-1. north_star V5.2 的 5 个不同 config 不知谁 valid
**症状**: Stage 3.5 log 输出 5 个 V5.2 评分 (61.6%/55.5%/61.4%/54.6%/60.6%), 没有清晰标注哪个对应 plan 的 "composite top10 10d focus" gate. 只能猜测最后一个对应。

**修法**: 自定义 Stage 3.5 gate script (plan 中设计的), 只跑一个 spec 配置 + 输出明确 PASS/FAIL。

#### S2-2. 242 天折扣因子 0.66 是否合理
**症状**: V5.2 raw 60.6% 被打折到 39.9%. 但 ng1.0.1 / ng1.0.6 等 baseline 通常报 5+ 年 backtest, 折扣 = 1.0. 不公平比较。

**修法**: Gate threshold 应该定义为 raw V5.2, 不被折扣干扰. 或扩展 spot check 窗口到 500+ days。

---

## 🚀 Part 3: 训练效率提升

### E0: 当前训练时间分析 (per 1-seed run, ~70 min)
- Auto-WF turbo check: 6 min (3 configs × WF window scan)
- Data load + prepare: 3 min (cache hit) / 8 min (cache miss)
- 4 targets × 6 algos = 24 boosters
  - LGB / LGB-Rank: ~4-5 min each
  - XGB: ~6 min
  - CatBoost: ~12 min ← **slowest**
  - RandomForest: ~6 min  
  - HistGBM: ~5 min
  - 总 booster time: ~40-45 min
- Save + WF summary: 5 min
- Per WF window × 3 windows: 总 60-65 min training time

### E1: 高优先级改进 (实施成本 < 4h, 提速 30%+)

#### E1-1. 实现 `--target-parallel` (P1-2 fix 后)
- 4 targets 并行, M5 Max 实测 1.38x → 70min → 51min per seed
- 6 seed runs = 6 × 51 = 5.1h vs 7h (-27%)
- **预估收益**: 节省 ~2h per full retrain

#### E1-2. WF Cache 跨 seed 共用
**现状**: 每 seed 独立 load_data + prepare_features (~10 min per seed × 6 = 60 min wasted)
**改进**: 用 hashed cache key (without seed) 让 X/y_Nd 在 seed 之间复用 (seed 只影响 booster 训练随机性)
**预估收益**: 节省 ~50 min per full retrain (实际看 cache hit rate)

#### E1-3. 跳过 CatBoost 或换为 LightGBM-mode CatBoost
**现状**: CatBoost ~12 min/booster, ICIR weight 经常被压到 1% (e.g. ng1.0.1)
**改进**: 对 ng1.3.x 砍掉 CB, 5 algos × 4 targets = 20 boosters (vs 24)
**预估收益**: 节省 ~12 min per WF window × 3 = 36 min per seed

#### E1-4. Cache 预热 (训练前一次性 load 全 universe)
**现状**: prepare_features 每次 X cross-sectional Z-score 重新算 (~3 min)
**改进**: 预先算好全数据 stats 缓存到 disk, 训练时 load
**预估收益**: 节省 ~6-9 min per seed (3 windows × 2-3 min)

### E2: 中优先级 (4-8h 实施, 50%+ 提速)

#### E2-1. Multi-seed 并行 (multi-process)
- 用 multiprocessing 并行 3 seeds (各自 fork)
- 注意 SQLite 并发: 用 read-only mode + WAL
- 内存: 3 × ~3GB peak = 9GB (M5 Max 64GB 充足)
- **收益**: 7h → ~3.5h (single seed time + overhead)

#### E2-2. Booster early stopping more aggressive
- LGB / XGB / HGB 早停 patience 从 30 → 15
- CB 从 50 → 25
- **预估**: 减少 20-30% per booster

#### E2-3. 减小 CV folds 至 2 (vs 3) for fast iteration
- ng1.3.x 的 WF 是 3 windows. 改 2 windows for fast-check
- 用于 ablation, 不用于 production model
- 收益: 33% time reduction

### E3: 长期 (重构, 8h+)

#### E3-1. 训练 pipeline 切换到 LightGBM-only
- LGB/LGB-Rank 在 ng1.0.1 实证有效, 占 ensemble 权重 50%+
- 删 XGB/CB/RF/HGB, 只保留 LGB + LGB-Rank
- 4 targets × 2 algos = 8 boosters vs 24 boosters
- **收益**: ~70% time reduction (7h → 2h)
- **风险**: 失去 ensemble 多样性, 但 ng1.0.1 已表明 LGB 权重最高时仍 work

#### E3-2. 用 Polars 替代 Pandas
- prepare_features 大量 groupby + reindex, Polars 通常 5-10x 快
- 改造范围: load_data + prepare_features + 部分 V485 内部
- **收益**: data prep 节省 ~40%
- **风险**: 兼容性 (Polars API ≠ Pandas)

#### E3-3. GPU 训练 (LGB GPU mode)
- M5 Max 有 GPU but LGB GPU mode 需要 OpenCL build
- 大数据 (>10M rows) 才有显著提速
- ng130 cache ~3M rows, GPU 收益边际
- **预估**: 不推荐 (开发成本 > 收益)

---

## 📋 Part 4: 评估流程效率

### V1. 报告生成 (242 day = 21 min) 已经可接受
- batch_generate_v395_reports 已用 fast mode
- 240 days × 5s = 20 min 合理

### V2. north_star eval 跑 5 个 config 浪费时间
- Stage 3.5 不需要 5 个 config, 只要 plan-spec 的 "composite top10 10d"
- 加 CLI flag `--gate-config-only` 跳过其他 4 个

### V3. β 网格搜索可并行 5 个 β
- 每个 β 独立报告生成 + 评估
- 用 ProcessPoolExecutor 并行 5 个进程
- 收益: 5h serial → 1-2h parallel

---

## 🎯 ng1.3.1 实施清单 (按优先级排序)

### Day 1 (准备 + 修)
- [ ] **Task 1.1**: 修 P0-1 (prepare_features 覆盖) → 单元测试验证 NG130 features 真正进训练
- [ ] **Task 1.2**: 修 P0-3 (pkl feature_names 同步) → 集成测试验证 scorer feature 对齐
- [ ] **Task 1.3**: 加 **3 轮 /simplify** (per memory feedback)
- [ ] **Task 1.4**: 实施 P1-2 (--target-parallel ThreadPoolExecutor) — 节省 27% time
- [ ] **Task 1.5**: 加 S0-1 验证 (label swap mean/std log)
- [ ] **Task 1.6**: 加 S1-1 fix (rank tie-breaking method='first')

### Day 2 (训练 + 评估)
- [ ] **Task 2.1**: 重训 6 runs (~5h with --target-parallel)
- [ ] **Task 2.2**: 训练后 sanity 三件套:
  - [ ] Head diversity test (excess vs downside corr should be in [-0.5, 0.5])
  - [ ] Seed diversity test (corr in [0.85, 0.95])
  - [ ] Feature importance check (Tier A + B 都进 top 30)
- [ ] **Task 2.3**: 写自定义 `scripts/ng130_stage35_gate.py` (V2 fix), 跑单个 spec config

### Day 3 (β + 全 gate)
- [ ] **Task 3.1**: β 网格搜索 (5 β × 3 WF = 15 evals, 用并行)
- [ ] **Task 3.2**: 选 β* + 更新 NG130DualHeadScorer.DEFAULT_BETA
- [ ] **Task 3.3**: Stage 3.5 重测 (with β*)
- [ ] **Task 3.4**: 如 Stage 3.5 PASS, 跑 Stage 4a (WF-OOS 2020-2026) + Stage 4b (Pre-2020)

### Day 4 (集成 + 部署)
- [ ] **Task 4.1**: P2-3 fix (quick_daily_update 加 ng1.3.0)
- [ ] **Task 4.2**: P2-1 fix (scorer assert feature_names 一致)
- [ ] **Task 4.3**: 1-2 周生产观察, 然后 PRODUCTION_VERSION='ng1.3.0'

---

## ⏱ 时间预算

| Phase | 估时 | 累计 |
|---|---|---|
| Day 1 (修+测试) | 8h | 8h |
| Day 2 (训练+sanity) | 5-6h (含训练 5h overnight) | 14h |
| Day 3 (β+gate) | 5h | 19h |
| Day 4 (集成+部署) | 2h + 观察期 | 21h + 观察 |

vs ng1.3.0 attempt (~40h trace + train + eval): **节省 50%+**

---

## 🛡 Risk Mitigation

### 关键 ABORT 触发 (不要硬撑)
1. P0-1 修复后, 训练日志显示 stock features < 60 → 修复未生效, ABORT
2. Sanity test head diversity > 0.95 → label swap 失败, ABORT  
3. β 网格全 5 个都 < 0.85 mean Sharpe → 模型本质有问题, ABORT
4. Stage 3.5 V5.2 < 65% (with β*) → ABORT, 写第二份 rejected.md
5. Stage 4b Pre-2020 V5.2 < 60% → 跨 regime 不泛化, ABORT

### Rollback 路径
- 任何阶段 fail → PRODUCTION_VERSION 保持 'ng1.0.1'
- 6 个 ng1.3.0 pkls 已 archive (不删, 留 postmortem)
- 6 个 ng1.3.1 pkls 训练完先放 ng/ 目录, gate 全 pass 再正式启用

---

## 📚 文档状态

- ✅ Spec: `docs/superpowers/specs/2026-04-18-ng130-multitask-design.md` (latest)
- ✅ Plan v1: `docs/superpowers/plans/2026-04-18-ng130-multitask.md` (executed, 含 ng1.3.0 实际偏差)
- ✅ Postmortem: `reports/ng130/stage35_rejected.md` (新写)
- ✅ **本文档** (ng1.3.1 systematic review plan)

---

## 🎬 下一轮 Session 起点

打开新 session 时, 让 Claude 阅读本文件作为 context, 然后:
1. 先看 ng1.3.0 已 commit 的 fixes (4 个 critical bugs 已修)
2. 按本文档 Day 1 Task 1.1 开始
3. 每完成一步打勾, 更新本文档

---

**End of plan.**
