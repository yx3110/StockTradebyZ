# ng1.2.4 实施计划 — 极保守增量 + Stage 3.5 production gate

> **For agentic workers:** 沿用 superpowers:subagent-driven-development 模式. 严格 ng1.2.3 教训:
> 1. 不引入新代码模块 — 复用 ng123_moneyflow_factors.py
> 2. 不动 label — 沿用 ng1.0.1 industry-excess
> 3. **必跑 Stage 3.5 spot check** 在 Phase 2 全量 backfill 之前

**Goal**: ng1.0.1 production baseline + 2 个 P90 mf factors + 0 label 改动. 验收 V5.2 ≥ 73% AND Pre-2020 ≥ 70% AND MaxDD ≤ -12%.

**Architecture**: 复用 ng1.2.3 已有的 moneyflow factor 计算模块 + cache infrastructure, 仅缩减输出至 top 2 factors. 新增 Stage 3.5 production spot check 作为 Phase 2 进入 gate.

**Tech Stack**: Python 3.13, LightGBM/XGBoost/CatBoost 3-seed ensemble, SQLite, NG trainer (`ml_models/ng/ng_trainer.py`). 复用 ng123 module 95% 不变.

**Spec**: `docs/superpowers/specs/2026-04-16-ng124-design.md`

---

## 前置: 文件结构

**新增文件**: 0 (零新模块)

**修改文件**:
- `ml_models/ng/ng_schema.py` — 注册 `ng1.2.4` → `ng124_feature_cache` (own schema, 无 downside cols)
- `ml_models/ng/ng123_moneyflow_factors.py` — 加 `ACCEPTED_MF_FACTORS_NG124` frozenset (top 2)
- `ml_models/ng/ng_cache_updater.py` — ng1.2.4 分支用 ng124 accepted set, 跳过 mined factor 计算
- `ml_models/ng/ng_trainer.py` — ng1.2.4 用纯 industry-excess label (跳过 downside penalty)
- `backtest/batch_generate_v395_reports.py` — 加 'ng1.2.4' choice
- `tests/`: 加 1 个测试 `test_ng124_acceptance` 验证 accepted set

**输出**:
- `reports/ng124/spot_check/` — Stage 3.5 90-day reports + V5 eval
- `reports/daily_selection_ng1.2.4_wf_oos/` — Phase 2 后 1520 daily reports
- `reports/daily_selection_ng1.2.4_pre2020/` — 360 daily reports
- `reports/ng124/decision.md` — 综合决策

---

## Phase 0: 基础设施 + Schema (½ day)

### Task 1: 注册 ng1.2.4 schema

**Files:**
- Modify: `ml_models/ng/ng_schema.py`

- [ ] **Step 1: VERSION_TABLE_MAP 加 ng1.2.4**

```python
'ng1.2.4': 'ng124_feature_cache',  # ng1.0.1 + 2 top mf factors only
```

- [ ] **Step 2: SCHEMA_VERSION_MAP 加 ng1.2.4 (own schema, NO downside cols)**

```python
'ng1.2.4': 'ng1.2.4',  # own schema (similar to ng1.2.3 but no downside_kd cols)
```

- [ ] **Step 3: `_schema_sql` 加 ng1.2.4 block (NO downside cols)**

```python
# ng1.2.4 has NO additional cols beyond ng1.0.1 base — just label_3d/5d/10d/15d + market
# (No downside_kd because no penalty applied; label remains industry-excess)
# So no additional schema block needed; ng1.2.4 falls through with existing base schema
```

⚠️ 关键: ng1.2.4 **不**触发 ng1.2.3 block (downside_3d/5d/10d/15d), 不触发 ng1.2.1 block (vn_label etc). 仅复用 ng1.0.1 base + label_raw.

需要在 `_schema_sql` 加上界守卫 ng1.2.3 block:
```python
if is_12 and version_ge(ver, 'ng1.2.3') and not version_ge(ver, 'ng1.2.4'):
    extra_cols += _real_cols('downside_3d', 'downside_5d', 'downside_10d', 'downside_15d')
```

- [ ] **Step 4: Smoke test**

```bash
python3 -c "
from ml_models.ng.ng_schema import _schema_sql
sql = _schema_sql('ng124_feature_cache', version='ng1.2.4')
# Must NOT include downside_*, vn_label_*, ra_label_*, maxdd_*, cond_label_*, amv_*
forbidden = ['downside_3d', 'vn_label_3d', 'ra_label_3d', 'maxdd_3d', 'cond_label_3d', 'amv_var1']
for c in forbidden:
    assert c not in sql, f'{c} should NOT be in ng1.2.4 schema'
# Must include base + label_raw
required = ['label_3d', 'label_raw_3d', 'features_json', 'market_return_5d']
for c in required:
    assert c in sql, f'{c} MUST be in ng1.2.4 schema'
print('ng1.2.4 schema OK: base + label_raw, no penalty cols')
"
python3 ml_models/ng/ng_schema.py ng1.2.4
```

- [ ] **Step 5: Test + Commit**

```bash
python3 -m pytest ml_models/ng/tests/test_ng123_schema.py -v  # ensure existing tests pass
# Add new tests to test_ng123_schema.py for ng1.2.4 (forbidden + required cols)

git add ml_models/ng/ng_schema.py ml_models/ng/tests/test_ng123_schema.py
git commit -m "feat(ng124): 注册 ng1.2.4 schema (无 downside_kd, 复用 ng1.0.1 base)"
```

---

### Task 2: 定义 ACCEPTED_MF_FACTORS_NG124 (top 2 only)

**Files:**
- Modify: `ml_models/ng/ng123_moneyflow_factors.py`

- [ ] **Step 1: 添加 ng1.2.4 专用 frozenset**

After `ACCEPTED_MF_FACTORS` block, add:

```python
# ng1.2.4: 极保守 — 仅 top 2 P90 因子 (|ICIR| > 0.5 in ng101 baseline percentile)
# Rationale: ng1.2.3 失败教训 = contrarian factors 累积反向, 减少到 2 个最强降低 risk
ACCEPTED_MF_FACTORS_NG124 = frozenset([
    'mf_net_elg_20d_ratio',     # |ICIR|=0.524, P90
    'cs_rank_mf_net_elg_20d',   # |ICIR|=0.511, P90
])
```

Add to `__all__`.

- [ ] **Step 2: 修改 `compute_all_moneyflow_factors`** to support ng1.2.4 selection:

```python
def compute_all_moneyflow_factors(
    rows: List[Dict],
    stock_scalars: Dict[str, float] = None,
    peer_scalars: Dict[str, np.ndarray] = None,
    accepted_only: bool = True,
    ng124_mode: bool = False,  # NEW
) -> Dict[str, float]:
    """...
    Args:
        ng124_mode: If True, return only ACCEPTED_MF_FACTORS_NG124 (top 2). Overrides accepted_only.
    """
    # ... existing body ...
    if ng124_mode:
        result = {k: v for k, v in result.items() if k in ACCEPTED_MF_FACTORS_NG124}
    elif accepted_only:
        result = {k: v for k, v in result.items() if k in ACCEPTED_MF_FACTORS}
    return result
```

- [ ] **Step 3: Test**

```python
def test_ng124_returns_2_factors():
    rows = [_mk_row(buy_elg=200, sell_elg=100, buy_sm=400, sell_sm=400)] * 20
    res = compute_all_moneyflow_factors(rows, ng124_mode=True)
    expected = {'mf_net_elg_20d_ratio', 'cs_rank_mf_net_elg_20d'}
    assert set(res.keys()) == expected
```

- [ ] **Step 4: Commit**

```bash
git add ml_models/ng/ng123_moneyflow_factors.py ml_models/ng/tests/test_ng123_moneyflow_factors.py
git commit -m "feat(ng124): ACCEPTED_MF_FACTORS_NG124 frozenset (top 2 P90 only)"
```

---

## Phase 1: Cache Updater + Trainer 集成 (½ day)

### Task 3: ng_cache_updater 分支 ng1.2.4 (跳过 mined, 用 NG124 accepted set)

**Files:**
- Modify: `ml_models/ng/ng_cache_updater.py`

Find the existing ng1.2.3 integration block (around lines 1460-1485). Add ng1.2.4 sister branch:

```python
# ng1.2.4: top 2 mf factors, NO mined, NO downside penalty
elif self.version == 'ng1.2.4':
    # Reuse moneyflow infrastructure (peer_mf_scalars_per_industry already computed)
    code_industry = universe[sec_id].get('industry', 'UNKNOWN')
    peer_mf = peer_mf_scalars_per_industry.get(code_industry, {})
    stock_scalars = stock_mf_scalars_per_code.get(code, {})
    mf_factors = compute_all_moneyflow_factors(
        moneyflow_rows,
        stock_scalars=stock_scalars,
        peer_scalars=peer_mf,
        ng124_mode=True,  # ← top 2 only
    )
    all_feats.update(mf_factors)
    # NO mined factors. NO downside computation. NO ng1.0.1 feature filtering.
    # ng1.2.4 = ng1.0.1 features + 2 mf factors, same label
```

- [ ] **Step 1: Modify** (per above)
- [ ] **Step 2: Smoke-test single date**
```bash
python3 ml_models/ng/ng_cache_updater.py --start-date 2024-06-03 --end-date 2024-06-03 --version ng1.2.4
python3 -c "
import sqlite3, json
c = sqlite3.connect('data_adapter/stock_data.db')
row = c.execute('SELECT features_json FROM ng124_feature_cache WHERE trade_date=\"2024-06-03\" LIMIT 1').fetchone()
feats = json.loads(row[0])
mf_keys = sorted([k for k in feats if k.startswith('mf_') or k.startswith('cs_rank_mf_')])
print(f'Total: {len(feats)} (expect ~76 = 64 ng1.0.1 + 2 mf + 10 market)')
print(f'mf keys ({len(mf_keys)}): {mf_keys}')
# Expect: 2 keys: mf_net_elg_20d_ratio, cs_rank_mf_net_elg_20d
"
```

- [ ] **Step 3: Commit**

---

### Task 4: ng_trainer 分支 ng1.2.4 (skip downside penalty)

**Files:**
- Modify: `ml_models/ng/ng_trainer.py`

Find `if _is_1_2_branch(self.schema_version) and version_ge(self.schema_version, 'ng1.2.3'):` blocks. Add upper-bound guard so ng1.2.4 does NOT enter:

```python
# Was: version_ge(..., 'ng1.2.3')
# New: _version_in_range(..., 'ng1.2.3', 'ng1.2.4')
if _is_1_2_branch(self.schema_version) and _version_in_range(self.schema_version, 'ng1.2.3', 'ng1.2.4'):
    # downside-related label transformation only for ng1.2.3
```

Specifically the SELECT extra_select for downside cols (line ~786) and the label penalty application (line ~944).

- [ ] **Step 1: Modify SELECT block**
- [ ] **Step 2: Modify label penalty block**
- [ ] **Step 3: Smoke-test trainer can read ng1.2.4 cache**
```bash
python3 ml_models/ng/ng_trainer.py --version ng1.2.4 --start-date 2024-01-01 --end-date 2024-06-30 \
  --fast-check --purge-days 15 2>&1 | head -20
# Should NOT log "ng1.2.3: applied downside penalty"
# Should load ng124_feature_cache successfully
```

- [ ] **Step 4: Commit**

---

## Phase 2: Mini Backfill for Stage 3.5 (½ day)

### Task 5: Mini backfill 2020-2024 for spot check training

- [ ] **Step 1: Backfill 2020-01 → 2024-12 (5 years)**

```bash
nohup python3 ml_models/ng/ng_cache_updater.py \
  --start-date 2020-01-01 --end-date 2024-12-31 --version ng1.2.4 \
  > logs/ng124_mini_backfill_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

Estimate ~2-3h with ng1.2.3-tested optimizations.

- [ ] **Step 2: Verify completion**

```bash
python3 -c "
import sqlite3
c = sqlite3.connect('data_adapter/stock_data.db')
s = c.execute('SELECT MIN(trade_date), MAX(trade_date), COUNT(DISTINCT trade_date), COUNT(*) FROM ng124_feature_cache').fetchone()
print(f'Mini cache: {s[0]} to {s[1]}, {s[2]} days, {s[3]:,} rows')
"
```

---

## Phase 3: Stage 3.5 Production Spot Check (1 day — KEY GATE)

### Task 6: Train mini-model for spot check

Since we want quick validation, train ONE seed (42) on 2020-01 → 2024-09 only.

- [ ] **Step 1: Quick single-seed train**

```bash
nohup python3 ml_models/ng/ng_trainer.py \
  --version ng1.2.4 --seed 42 --purge-days 15 \
  --start-date 2020-01-01 --end-date 2024-09-30 \
  > logs/ng124_spot_train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

Estimate ~1-1.5h.

- [ ] **Step 2: Verify model saved**

```bash
ls -lt ml_models/trained_models/ng/ng124_seed42_*.pkl
```

### Task 7: Generate 90-day spot check reports

- [ ] **Step 1: Add 'ng1.2.4' to batch_generate choices**

```bash
sed -i '' "s/'ng1.2.0', 'ng1.2.1', 'ng1.2.2', 'ng1.2.3'/'ng1.2.0', 'ng1.2.1', 'ng1.2.2', 'ng1.2.3', 'ng1.2.4'/" backtest/batch_generate_v395_reports.py
```

- [ ] **Step 2: Generate 90-day reports for 2024-10-01 → 2024-12-31**

```bash
nohup python3 backtest/batch_generate_v395_reports.py \
  --version ng1.2.4 --start-date 2024-10-01 --end-date 2024-12-31 \
  --output-dir reports/ng124/spot_check/daily \
  > logs/ng124_spot_reports_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

Estimate ~10-15min for ~60 trading days.

### Task 8: Spot check evaluation + GATE decision

- [ ] **Step 1: Run north star eval on spot check reports**

```bash
python3 backtest/run_north_star_eval.py --backtest \
  --report-dir reports/ng124/spot_check/daily \
  --label NG124-SPOTCHECK --top-n 10 --focus-days 10 --rank-field composite \
  > reports/ng124/spot_check/eval.log 2>&1

# Extract V5.2 estimate
grep -E "V5|总分|年化|Sharpe|MaxDD" reports/ng124/spot_check/eval.log | tail -20
```

- [ ] **Step 2: GATE decision**

| V5 score | Action |
|---|---|
| **≥ 65%** | ✅ PROCEED to Phase 4 (full backfill + train) |
| **60-65%** | ⚠️ User review required before continuing |
| **< 60%** | ❌ ABORT — write postmortem, ng1.2.4 REJECTED |

- [ ] **Step 3: Commit spot check artifacts**

```bash
git add reports/ng124/spot_check/eval.log reports/ng124/spot_check/decision.md
git commit -m "checkpoint(ng124): Stage 3.5 spot check V5=XX% — DECISION"
```

---

## Phase 4: Full Backfill (1 day, only if Stage 3.5 PASS)

### Task 9: Full backfill 2018-2026

Same as ng1.2.3 Phase 2, with ng1.2.4 as version:

```bash
# Clean mini cache first to avoid duplicates
python3 -c "import sqlite3; c=sqlite3.connect('data_adapter/stock_data.db'); n=c.execute('DELETE FROM ng124_feature_cache').rowcount; c.commit(); print(f'Cleaned {n} rows')"

# Full backfill
nohup python3 ml_models/ng/ng_cache_updater.py \
  --start-date 2018-01-01 --end-date 2026-04-16 --version ng1.2.4 \
  > logs/ng124_full_backfill_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

Estimate ~2.5h (with R1 optimizations from ng1.2.3).

---

## Phase 5: Full 3-Seed Training (1 day)

### Task 10: Train 3 seeds

```bash
nohup python3 ml_models/ng/ng_trainer.py \
  --version ng1.2.4 \
  --seeds 42,123,456 \
  --purge-days 15 \
  > logs/ng124_train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

Estimate ~4-5h.

---

## Phase 6: Bidirectional Eval + Decision (½ day)

### Task 11: Generate WF-OOS + Pre-2020 reports

```bash
# WF-OOS (2020-2026)
nohup python3 backtest/batch_generate_v395_reports.py \
  --version ng1.2.4 --start-date 2020-01-01 --end-date 2026-04-15 \
  --output-dir reports/daily_selection_ng1.2.4_wf_oos &

# Pre-2020 (2018-04 → 2019-12)
nohup python3 backtest/batch_generate_v395_reports.py \
  --version ng1.2.4 --start-date 2018-04-01 --end-date 2019-12-31 \
  --output-dir reports/daily_selection_ng1.2.4_pre2020 &
```

### Task 12: Bidirectional north star eval

```bash
python3 backtest/run_north_star_eval.py --backtest \
  --report-dir reports/daily_selection_ng1.2.4_wf_oos \
  --label NG124-WFOOS --top-n 10 --focus-days 10 --rank-field composite

python3 backtest/run_north_star_eval.py --backtest \
  --report-dir reports/daily_selection_ng1.2.4_pre2020 \
  --label NG124-PRE2020 --top-n 10 --focus-days 10 --rank-field composite
```

### Task 13: Apply acceptance criteria + DECISION

Per spec §8 thresholds:
- WF-OOS V5.2 ≥ 73% AND Pre-2020 V5.2 ≥ 70% AND MaxDD ≤ -12%

| Outcome | Action |
|---|---|
| ALL PASS | ✅ Promote: `PRODUCTION_VERSION = 'ng1.2.4'` |
| Partial (V5≥72 but Pre-2020 ok) | User review |
| FAIL | ❌ Postmortem, keep ng1.0.1 |

---

## Phase 7A: Production Switch (if accepted)

Same as ng1.2.3 Phase 5A: change `PRODUCTION_VERSION`, update CLAUDE.md + Wiki + MEMORY.

## Phase 7B: Postmortem (if rejected)

Same as ng1.2.3 Phase 5B: docs/wiki/architecture/ng124_postmortem.md.

---

## Quick Reference: Spec §6.4 Decision Conditions

```
Phase 3 (Stage 3.5 spot check)
   ├─ V5 ≥ 65% ──► Phase 4 + 5 + 6 (full pipeline)
   ├─ V5 60-65% ──► User review
   └─ V5 < 60% ──► ABORT (Postmortem, save ~10h work)

Phase 6 (final eval)
   ├─ ALL pass ──► Promote to production
   ├─ Partial ──► User review
   └─ FAIL ──► Postmortem
```

## Lessons Applied from ng1.2.3

| ng1.2.3 mistake | ng1.2.4 mitigation |
|---|---|
| 改 3 轴 (features + factors + label) | 仅改 1 轴 (+2 factors) |
| Stage 3 用 WF training ICIR 选 λ | 不改 label, λ ablation 不需要 |
| 直接 8h+ full backfill 后才发现失败 | Stage 3.5 spot check 1h gate |
| Mined factors 累积反向 | 不加 mined |
| Downside penalty 翻号 | 不加 penalty |

**Total time saved on failure path**: 失败时 (Stage 3.5 fail) 仅花 ~3h (mini backfill + spot train + eval), 而非 ng1.2.3 的 12+h.
