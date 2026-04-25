# ng2.0b Pre-flight Checklist (per CLAUDE.md 10-item)

**Goal:** ng2.0b = ng1.0.1 base schema + `--regime-weight {bull, bear}` sample weighting.
- ng2.0b-bull: bull-regime samples ×2, bear ×0.5
- ng2.0b-bear: bear-regime samples ×2, bull ×0.5

**Date:** 2026-04-26
**Git HEAD:** 0a9c464b

---

## ✅ Check 1: Schema 一致性

- ng2.0b 复用 ng1.0.1 schema (66 features), no new schema branch
- DB 表 `ng101_feature_cache` 18 列: id/code/trade_date/features_json (56 stock features) + 4 labels (3d/5d/10d/15d) + 10 market_* cols
- Trainer 读 `features_json` JSON 列, 无 schema 改动需求
- **PASS** — no schema work

## ✅ Check 2: Feature Backfill 逻辑

- `ng101_feature_cache`: 3,668,887 行 (2018-04-02 → 2026-04-24)
- Pre-2020 (2018-04 → 2019-12): 452,401 行 ✓ (够跑 Pre-2020 OOS eval)
- 2020+: 3,216,486 行 ✓ (够跑 WF-OOS eval)
- ng1.0.1 backfill 早已完成, sample-weighting 不动 features
- **PASS** — no backfill work

## ✅ Check 3: Efficiency

- Trainer: `ml_models/training/train_v395_multi_target.py` (existing)
- `--target-parallel 4` ✓ (per memory `training_target_parallel.md`, M5 Max 1.38x speedup)
- Auto-WF on by default (turbo-check 3 configs, 6min overhead)
- Predicted full-train: 30-60 min per sub-model on M5 Max (per CLAUDE.md ng1.0.1 reference)
- **PASS**

## ✅ Check 4: Acceptance Criteria + ABORT line

- **Final gates** (per plan):
  - WF-OOS V5.2 ≥ 81% (vs ng2.0a baseline 79.3% A+, +1.7pp)
  - WF-OOS MaxDD ≤ -18% (vs ng2.0a -17.6%, ≥ existing)
  - WF-OOS Sharpe ≥ 2.5 (vs ng2.0a 2.751)
  - Pre-2020 net annual ≥ -5% (vs ng2.0a -17.3%, +12.3pp)
- **Mid-train ABORT line**: first WF window 10d ICIR < 0.4 → kill
- **PASS** — gates documented

## ✅ Check 5: Baseline Fair Comparison

- 对比基线: ng2.0a (= ng1.0.1 bull + ng1.0.4 bear, baseline regime)
- 同 WF mode (expanding) + 同 purge=15 + 同 seed=42
- 评估窗口: WF-OOS 2020-2026 + Pre-2020 2018-2019 (与 ng2.0a Phase C 完全相同)
- **PASS**

## ✅ Check 6: Checkpointing + Logging

- `caffeinate -i` 防 sleep ✓
- `tee logs/train_ng2_0b_*_$(timestamp).log` ✓
- Trainer 默认每个 WF 窗口 `joblib.dump` checkpoint
- PID 写入 logs/, tail -f 可查
- **PASS** — protocol 已定

## ✅ Check 7: Leakage Pre-scan

- ng1.0.1 schema clean: β_UMD=+0.38 t=5.4 (per memory `ng101_pre2020_audit_2026_04_20.md`), 1/8 of v4.9.0.1
- Sample weights 不引入新 features → 不引入新 leakage
- Purge=15 days 与 15d label horizon 一致 ✓
- **PASS**

## ✅ Check 8: Resources

- Disk: 710 GB free / 1.8 TB (>>20 GB threshold) ✓
- RAM: ~24 GB free (M5 Max 64GB total, ng1.0.1 训练峰值 ~20GB per memory) ✓
- 无竞争训练任务: `ps aux | grep -E "train_v395|cache_updater"` returned empty ✓
- **PASS**

## ✅ Check 9: 可重现性元数据

- pickle 写入 (verify post-train):
  - `git_commit_hash`: 0a9c464b
  - `schema_version`: ng1.0.1
  - `regime_weight_mode`: bull / bear (新加, Task B2 需写入)
  - `seed`: 42
  - `wf_mode`: auto (or expanding fallback)
  - `purge_days`: 15
  - `training_duration_sec`: ?
  - `host`: M5Max-local
- **CONDITIONAL PASS** — Task B2 必须确保 `regime_weight_mode` 写入 pickle metadata. Task B3 Step 3 验证.

## ⏳ Check 10: /simplify

- Trainer 改动 (Task B2) — 写完 + smoke 之后 /simplify 3 轮 (per CLAUDE.md memory `feedback_simplify_after_each_step.md`)
- **DEFERRED** — Task B2 收尾时执行

---

## 综合 verdict

10/10 PASS or CONDITIONAL/DEFERRED (B2 收尾时 close):
1. Schema ✓
2. Backfill ✓
3. Efficiency ✓
4. Acceptance ✓
5. Baseline ✓
6. Checkpointing ✓
7. Leakage ✓
8. Resources ✓
9. Metadata (B2 + B3 close)
10. /simplify (B2 close)

**KICKOFF APPROVED** for Phase B2 (trainer code change + smoke).
After B2 smoke + /simplify pass: Phase B3 long train.
