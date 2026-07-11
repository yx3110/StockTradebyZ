# NG v1.5.0 Implementation Plan

**Created**: 2026-04-20
**Spec**: `docs/superpowers/specs/2026-04-20-ng150-regime-refined-design.md` (**MUST READ FIRST**)
**Goal**: Build ng1.5.0 model — ng1.4.0 base + 5 regime-refined features, target V5.2 ≥ 75% + MaxDD ≤ -15% + Pre-2020 年化 ≥ 0%
**Time budget**: 18-25h (Phases 0-9 main path)
**Audience**: Fresh session with cleared context. Read every referenced file before acting.

---

## 0. Mandatory Pre-Read (do this first before any action)

Load the following files into context **in order**. Do not skim — these contain the 2026-04-20 审计结论 that drive every design decision:

1. `docs/superpowers/specs/2026-04-20-ng150-regime-refined-design.md` — the full spec (why, what, not-do)
2. `CLAUDE.md` sections: `🛑 模型迭代 Pre-flight Checklist` (lines ~12-170) + `ML Scoring Systems` (lines 570-600)
3. `/Users/yangxu/.claude/projects/-Users-yangxu-StockTradebyZ/memory/MEMORY.md` (index) + specifically:
   - `ng101_pre2020_audit_2026_04_20.md` (73.7% A+ is a ghost number, full audit)
   - `ng106_amv_regime.md` (2026-04-20 update: ng1.0.6 is actually综合最优)
   - `ng_production_switch.md`
4. `docs/wiki/models/ng-series.md` Pre-2020 OOS 小节 + WF-OOS 完整复核 小节 (all 实测数字)
5. `docs/ng140_plan.md` — ng1.4.0 context (this builds on ng1.4.0)
6. `ml_models/ng/ng_schema.py` — schema registry (you'll add ng1.5.0 entries here)
7. `ml_models/ng/ng_trainer.py` — trainer (you'll add ng1.5.0 feature list + branch)
8. `ml_models/ng/ng_feature_calculator.py` — feature computation (you'll add 5 new features)
9. `ml_models/ng/ng_cache_updater.py` — cache backfill (you'll extend for ng1.5.0)

**Key facts you MUST know after pre-read**:
- Current production: ng1.0.1 (pkl `ng101_seed42_multi_target_20260412_233749.pkl`), WF-OOS V5.2=73.4% A+, MaxDD=-11.7%, **Pre-2020 V5.2=45.5% B (not 73.7% A+ — that was a ghost number)**
- ng1.0.6 (0AMV regime switch) is actually the best综合 model: WF-OOS V5.2=78.9% A+, β_UMD=+0.005, Pre-2020 唯一正年化 (+0.7%), but MaxDD=-22.9% (twice ng1.0.1)
- ng1.4.0 (ng1.0.1 + Tier A 4 downside + 3 AMV = 73 features) training完成, Stage 3.5 PASS 68%, **Stage 4a pending**
- ng1.3.0 (dual-head) REJECTED, ng1.2.x 全 REJECTED, ng1.0.4/1.0.7 Pre-2020 不泛化
- 所有 loss/label 创新 independently 失败 → ng1.5.0 坚持 MSE + industry excess

**Red flags to watch for** (past bugs that keep reappearing):
- Schema 一致性 (DB ⇔ trainer ⇔ scorer 三方对齐), past bugs: ng1.1.0 走 ng1.0.7 超集, revenue_growth 用 profit_to_gr
- Seed 传播 (bug discovered in ensemble_iteration_2026_04_08.md), must verify 3-seed pred corr ∈ [0.85, 0.95]
- Fast-check 虚高 (ng1.0.9 教训), 不作 go/no-go 主决策
- Multi-process Pool 死锁 (memory), 回填用 `--num-workers 0` 顺序
- 数据泄露 (v4.9.0.1 β_UMD=3.029 事后才发现), 必须跑 `factor_returns.py` 落 β artefact

---

## 1. Checkpoint Strategy

**This plan is session-safe**: each phase ends with a git commit so the next session can resume by running `git log --oneline | head -20` to find where it stopped.

Commit message prefix convention: `[ng150-plan] P{N}: <what>`

Examples:
- `[ng150-plan] P0: prerequisites complete` (after phase 0)
- `[ng150-plan] P1: feature engineering done, IC validated`
- `[ng150-plan] P5: Stage 3.5 gate PASS V5.2=X%`

**Never skip the commit** — it's the only reliable handoff to a fresh session.

---

## Phase 0: Prerequisites (4h, do before main work)

Three prerequisites identified in spec Section 9. Do them in this order:

### 0.1 Evaluate ng1.4.0 Stage 4a + 4b (1h)

ng1.4.0 already trained (3 seeds, pkls in `ml_models/trained_models/ng/ng140_seed{42,123,456}_multi_target_20260420_*.pkl`) with 552 reports in `reports/daily_selection_ng1.4.0_stage4a/` but **gate evaluation not yet run**. Result determines ng1.5.0 scope.

```bash
# Stage 4a (2024-2026, 跨 regime)
caffeinate -i python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_ng1.4.0_stage4a \
    --label NG140-STAGE4A \
    --scoring-version ng1.4.0 \
    --rank-field composite \
    --top-n 10 \
    --focus-days 10 \
    --score-version v52 > logs/ng140_stage4a_eval_$(date +%Y%m%d_%H%M%S).log 2>&1

# Extract V5.2
f=$(ls -t logs/ng140_stage4a_eval_*.log | head -1)
grep -E "MKT=|加权评分.*→ 等级|年化收益\(净\)|Sharpe:|最大回撤" "$f" | tail -15
```

**Decision tree based on Stage 4a V5.2**:

| ng1.4.0 Stage 4a V5.2 | Decision |
|---|---|
| **≥ 75% A+** | ng1.4.0 可能已够, ng1.5.0 降级为"ng1.4.0 + 2 精选新特征"小迭代 (节省 ~50% 工作量). 见 §0.1.a |
| **70-75%** | ng1.5.0 按原计划走 (加 5 新特征), 作为 incremental improvement |
| **< 70%** | ng1.5.0 必须做全量 (5 特征 + Phase B CVaR 备选启用概率高) |

Also run Pre-2020 (Stage 4b) for ng1.4.0 if cache has 2018-2019 for ng130_feature_cache (need to check):

```bash
# Check ng130 cache 2018-2019 coverage
python3 -c "
import sqlite3
c = sqlite3.connect('data_adapter/stock_data.db')
n = c.execute('SELECT COUNT(*), COUNT(DISTINCT trade_date) FROM ng130_feature_cache WHERE trade_date BETWEEN ? AND ?', ('2018-01-01', '2019-12-31')).fetchone()
print(f'ng130_feature_cache 2018-2019: {n[0]} rows, {n[1]} dates')
"

# If dates >= 200, proceed; else skip 4b or backfill first
```

**Checkpoint**: `git commit -am "[ng150-plan] P0.1: ng140 Stage 4a eval complete, V5.2=X%"`

#### 0.1.a Scope Adjustment (if ng1.4.0 Stage 4a PASS ≥ 75%)

Re-read spec Section 2.1. Pick only **2 of the 5 new features** for ng1.5.0:
- Keep: `amv_regime_bull_prob` + `recent_maxdd_60d` (highest-priority Tier B)
- Drop: `industry_regime_agreement`, `volatility_skew_20d`, `upside_capture_60d`
- Rationale: ng1.4.0 already strong; minimal marginal features reduce overfitting risk
- Time budget adjustment: Phase 1 cuts to 1.5h, Phase 4 unchanged

Update: `git commit -am "[ng150-plan] P0.1a: scope cut to 2 features (ng140 Stage 4a PASSED)"`

### 0.2 Generalize `scripts/ng130_stage35_gate.py` (1h)

Current state: script is hardcoded for ng1.3.0. Must accept `--version` to work with ng1.4.0, ng1.5.0, and future versions.

```bash
# Inspect current interface
python3 scripts/ng130_stage35_gate.py --help 2>&1 | head -30
grep -n "ng1.3\|ng130\|NGProductionScorer\|NG130DualHeadScorer" scripts/ng130_stage35_gate.py | head -10
```

**What to change** (pattern, adjust to actual code):
1. Add `--version` CLI arg (default 'ng1.3.0' for backward compat)
2. Route scorer loading by version:
   - `ng1.3.x` → `NG130DualHeadScorer`
   - `ng1.4.x`, `ng1.5.x`, `ng1.0.x`, `ng1.1.x` → `NGProductionScorer(version=args.version)`
3. Parameterize output dir: `reports/daily_selection_{version_dotted}_stage35` / `_stage4a`
4. Keep V5.2 threshold logic the same

**Validation**: re-run with ng1.3.0 first (must match prior output), then test ng1.4.0:

```bash
# Smoke: ng1.3.0 still works
python3 scripts/ng130_stage35_gate.py --version ng1.3.0 --start 2025-01-01 --end 2025-06-30 --dry-run
# Real: ng1.4.0 works
python3 scripts/ng130_stage35_gate.py --version ng1.4.0 --start 2025-01-01 --end 2025-12-31
```

**Checkpoint**: `git commit -am "[ng150-plan] P0.2: generalize stage35_gate.py --version support"`

### 0.3 Harden pkl metadata (CLAUDE.md Check 9) (30min)

Current bug: ng1.0.1 pkl lacks `git_commit_hash`, `host`, `training_duration_sec` (per audit memory). Fix **before** training ng1.5.0.

**Edit** `ml_models/ng/ng_trainer.py`, find the `joblib.dump(model_data, path)` call. Add metadata:

```python
import subprocess, socket, time

# Before joblib.dump
model_data['git_commit_hash'] = subprocess.check_output(
    ['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, text=True
).strip()
model_data['host'] = socket.gethostname()
model_data['training_duration_sec'] = time.time() - train_start_ts
model_data['schema_version'] = self.schema_version  # already has
# Verify these already present: seed, feature_names, targets, wf_mode, purge_days
```

Also add sanity check at load time in `ng_production_scorer.py`:
```python
# Warn if metadata missing (backward compat)
for field in ['git_commit_hash', 'host', 'schema_version']:
    if field not in model_data:
        logger.warning(f"pkl missing {field} — pre-2026-04-20 model")
```

**Test**: train a throwaway 1-window model, verify pkl has all metadata:
```bash
python3 ml_models/ng/ng_trainer.py --version ng1.0.1 --fast-check --seed 42
python3 -c "
import joblib
m = joblib.load(sorted([p for p in __import__('pathlib').Path('ml_models/trained_models/ng').glob('ng101_seed42_*.pkl')])[-1])
for k in ['git_commit_hash', 'host', 'training_duration_sec', 'seed', 'schema_version']:
    print(f'  {k}: {m.get(k, \"MISSING\")}')
"
```

**Checkpoint**: `git commit -am "[ng150-plan] P0.3: pkl metadata hardened (git_commit/host/duration)"`

---

## Phase 1: Feature Engineering (2-3h)

Implement 5 new Tier B features (or 2 if §0.1.a). Reference: spec Section 2.1.

### 1.1 Add features to `ml_models/ng/ng_feature_calculator.py`

Each new feature gets its own function. Strict rules:
- No future info: computed on data up to and including `trade_date`, no `shift(-...)`
- Handle NaN gracefully: return np.nan if insufficient history
- Unit test at top of function docstring with expected shape

#### `amv_regime_bull_prob`

```python
def _compute_amv_regime_bull_prob(amv_var1: pd.Series, amv_macd: pd.Series,
                                    amv_ma60: pd.Series) -> pd.Series:
    """
    Soft bull probability from 0AMV state.
      bull_score = 0.6 * tanh((var1/ma60 - 1) * 10) + 0.4 * tanh(macd * 5)
      bull_prob = (bull_score + 1) / 2  # map [-1, 1] → [0, 1]

    No future info: all inputs are t-1 snapshot (amv_var1 is current day close).
    Expected: 0.5 around regime transition, 0.0 in deep bear, 1.0 in strong bull.
    """
    var1_ratio = amv_var1 / amv_ma60 - 1.0
    score = 0.6 * np.tanh(var1_ratio * 10) + 0.4 * np.tanh(amv_macd * 5)
    return (score + 1) / 2
```

#### `industry_regime_agreement`

```python
def _compute_industry_regime_agreement(
    stock_ret_5d: pd.Series,  # 本股过去 5 日收益
    industry_ret_5d: pd.Series,  # 本股行业过去 5 日收益
    market_ret_5d: pd.Series,  # 大盘过去 5 日收益
    window: int = 60
) -> pd.Series:
    """
    60 日滚动窗口: 行业与大盘方向一致天数占比.
      sign_agree = (sign(industry_ret) == sign(market_ret)).astype(int)
      agreement = rolling_mean(sign_agree, window=60)

    ng1.0.6 成功的关键: 当行业和大盘同向 → 跟牛/抗熊都有结构性 α.
    """
    agree = (np.sign(industry_ret_5d) == np.sign(market_ret_5d)).astype(float)
    return agree.rolling(window).mean()
```

#### `recent_maxdd_60d`

```python
def _compute_recent_maxdd_60d(close: pd.Series) -> pd.Series:
    """
    60 日窗口 当前价相对窗口最高点的回撤.
      dd[t] = close[t] / max(close[t-59:t+1]) - 1  # 总是 <= 0
    """
    rolling_max = close.rolling(60, min_periods=20).max()
    return close / rolling_max - 1.0
```

#### `volatility_skew_20d`

```python
def _compute_volatility_skew_20d(returns_1d: pd.Series) -> pd.Series:
    """
    下行波动占比: downside_vol / (upside_vol + ε).
      downside_vol = std(returns where returns < 0, 20d)
      upside_vol = std(returns where returns > 0, 20d)
    """
    def _skew(window):
        neg = window[window < 0]
        pos = window[window > 0]
        if len(neg) < 2 or len(pos) < 2:
            return np.nan
        return neg.std() / (pos.std() + 1e-6)
    return returns_1d.rolling(20, min_periods=10).apply(_skew, raw=False)
```

#### `upside_capture_60d`

```python
def _compute_upside_capture_60d(
    stock_ret_1d: pd.Series, market_ret_1d: pd.Series, window: int = 60
) -> pd.Series:
    """
    大盘涨日本股涨幅 / 大盘涨幅 平均.
      bull_days = market_ret_1d > 0
      capture[t] = mean(stock_ret / market_ret) over last 60 bull days (up to t)

    识别陷阱股: 牛市不跟涨 + 熊市跟跌的股票.
    """
    def _cap(df_window):
        bull = df_window[df_window['mkt'] > 0]
        if len(bull) < 10:
            return np.nan
        return (bull['stock'] / (bull['mkt'] + 1e-6)).mean()
    combined = pd.DataFrame({'stock': stock_ret_1d, 'mkt': market_ret_1d})
    return combined.rolling(window, min_periods=20).apply(_cap).iloc[:, 0]
```

### 1.2 Wire into `build_features()`

Find the existing `build_features()` call in `ng_feature_calculator.py`. Add 5 new feature assignments in a block guarded by `version_ge(ver, 'ng1.5.0')`.

### 1.3 Register feature names in `ng_trainer.py`

Append to `ml_models/ng/ng_trainer.py`:

```python
# ng1.5.0: ng1.4.0 + 5 Tier B regime-refined features
NG150_TIER_B_FEATURES: List[str] = [
    'amv_regime_bull_prob',
    'industry_regime_agreement',
    'recent_maxdd_60d',
    'volatility_skew_20d',
    'upside_capture_60d',
]

NG150_STOCK_FEATURES: List[str] = (
    STOCK_FEATURE_NAMES
    + NG130_TIER_A_DOWNSIDE  # 4 downside
    + NG150_TIER_B_FEATURES  # 5 new
)
NG150_MARKET_FEATURES: List[str] = (
    MARKET_FEATURE_NAMES
    + NG130_TIER_A_AMV  # 3 AMV
)
NG150_ALL_FEATURES: List[str] = NG150_STOCK_FEATURES + NG150_MARKET_FEATURES
NG150_VERSION = 'ng1.5.0'
# Total: 56 + 4 + 5 + 10 + 3 = 78 features
```

Update `version_feature_table` in `NGTrainer.__init__` to prepend ng1.5.0:

```python
version_feature_table = [
    ('ng1.5.0', NG150_ALL_FEATURES, NG150_STOCK_FEATURES, NG150_MARKET_FEATURES, []),
    ('ng1.4.0', NG140_ALL_FEATURES, ...),
    # ...
]
```

### 1.4 Single-factor IC validation

Before committing features, run a smoke IC check (each new feature vs forward 10d return):

```bash
python3 -c "
import sqlite3, numpy as np, pandas as pd
# Pull 2024 Q4 samples + next 10d return
# For each new feature: compute rank IC
# PASS: |IC| >= 0.02 on at least 3 of 5 new features
# ABORT: < 3 features pass
" > logs/ng150_feature_ic_$(date +%Y%m%d_%H%M%S).log 2>&1
```

If 3+ of 5 pass, proceed. If not, re-examine feature formulas (may have data leakage or bug).

### 1.5 `/simplify` 3 rounds

Per `feedback_simplify_after_each_step.md`: run `/simplify ml_models/ng/ng_feature_calculator.py` and `/simplify ml_models/ng/ng_trainer.py` three times each. Fix each issue before continuing.

**Checkpoint**: `git commit -am "[ng150-plan] P1: 5 Tier B features implemented + IC validated"`

---

## Phase 2: Cache Backfill (40-60 min)

### 2.1 Register ng1.5.0 schema

Edit `ml_models/ng/ng_schema.py`:

```python
VERSION_TABLE_MAP['ng1.5.0'] = 'ng150_feature_cache'
SCHEMA_VERSION_MAP['ng1.5.0'] = 'ng1.5.0'  # own schema (not reuse)

# In _schema_sql() add a branch for ng1.5.x:
def _is_1_5_branch(ver: str) -> bool:
    return ver.startswith('ng1.5.')

# In the extra_cols logic:
if _is_1_5_branch(ver):
    extra_cols = _real_cols('label_raw_3d', 'label_raw_5d', 'label_raw_10d', 'label_raw_15d')
    # No downside/ra/cond labels for ng1.5.0
```

Verify the new table creates:

```bash
python3 -c "
from ml_models.ng.ng_schema import create_table
create_table(version='ng1.5.0')
"
python3 -c "
import sqlite3
c = sqlite3.connect('data_adapter/stock_data.db')
cols = [r[1] for r in c.execute('PRAGMA table_info(ng150_feature_cache)').fetchall()]
print(f'{len(cols)} columns: {cols[:5]}...{cols[-5:]}')
"
```

### 2.2 Wire `ng_cache_updater.py` for ng1.5.0

The updater needs to know how to compute the 5 new features during pass-1. Reuse AMV and industry return reads from ng1.3.x/ng1.4.x branches.

### 2.3 Single-day backfill test

```bash
python3 ml_models/ng/ng_cache_updater.py --date 2024-06-03 --version ng1.5.0 2>&1 | tail -10
python3 -c "
import sqlite3, json
c = sqlite3.connect('data_adapter/stock_data.db')
row = c.execute('SELECT features_json FROM ng150_feature_cache WHERE trade_date=? LIMIT 1', ('2024-06-03',)).fetchone()
feats = json.loads(row[0])
expected = ['amv_regime_bull_prob', 'industry_regime_agreement', 'recent_maxdd_60d', 'volatility_skew_20d', 'upside_capture_60d']
for f in expected:
    val = feats.get(f, 'MISSING')
    print(f'  {f}: {val}')
"
```

**ABORT**: any new feature returns None / all-NaN on a healthy day. Fix formula.

### 2.4 Full backfill 2018-2026

```bash
caffeinate -i python3 ml_models/ng/ng_cache_updater.py \
    --start-date 2018-01-01 --end-date 2026-04-20 \
    --version ng1.5.0 > logs/ng150_backfill_$(date +%Y%m%d_%H%M%S).log 2>&1
```

Watch: ~4s/day × 2030 days = ~2h (sequential, Pool deadlock avoided).

Validate coverage:
```bash
python3 -c "
import sqlite3
c = sqlite3.connect('data_adapter/stock_data.db')
r = c.execute('SELECT COUNT(*), COUNT(DISTINCT trade_date), AVG(cnt) FROM (SELECT COUNT(*) as cnt FROM ng150_feature_cache GROUP BY trade_date)').fetchone()
print(f'rows={r[0]}, dates={r[1]}, avg_per_day={r[2]:.0f}')
# Expect: ~2M rows, ~2030 dates, ~1100-1300 stocks/day (same filter as ng1.0.1)
"
```

**ABORT**: avg < 1000 stocks/day → feature calculation bug. Debug single day first.

**Checkpoint**: `git commit -am "[ng150-plan] P2: ng150 cache backfilled 2018-2026"`

---

## Phase 3: Fast-check (2 min)

```bash
python3 ml_models/ng/ng_trainer.py --version ng1.5.0 --seed 42 \
    --fast-check --target-parallel 4 2>&1 | tee logs/ng150_fastcheck_$(date +%Y%m%d_%H%M%S).log
```

**PASS**: 10d OOS IC ≥ 0.05 AND ≥ 0 direction (printed at end)
**ABORT**: IC < 0.03 OR negative direction → features have no signal or wrong direction. Review §1.1 formulas.

**Checkpoint**: `git commit -am "[ng150-plan] P3: fast-check PASS IC=X"`

---

## Phase 4: Full training 3-seed (6-10h)

**Critical**: run seeds **serially** (not parallel) — each seed uses `--target-parallel 4` for intra-window parallelism already.

```bash
mkdir -p logs/ng150_training_$(date +%Y%m%d_%H%M%S)
LOGDIR=logs/ng150_training_$(date +%Y%m%d_%H%M%S)

for SEED in 42 123 456; do
  echo "Starting seed $SEED at $(date)"
  caffeinate -i python3 ml_models/ng/ng_trainer.py \
      --version ng1.5.0 --seed ${SEED} --target-parallel 4 \
      2>&1 | tee ${LOGDIR}/ng150_seed${SEED}.log
  echo "Seed $SEED done at $(date)"
done
```

Check each seed's WF ICIR in factor_quality JSON:
```bash
for SEED in 42 123 456; do
  f=$(ls -t ml_models/trained_models/ng/factor_quality_*.json | head -1)
  python3 -c "
import json
d = json.load(open('$f'))
wf = d.get('wf_icir', {})
print(f'seed $SEED 10d ICIR: {wf.get(\"10d\", {}).get(\"icir\", \"MISSING\")}')"
done
```

**PASS**: all 3 seeds 10d ICIR ≥ 0.70 (ng1.4.0 seed456 was 1.09)
**ABORT**: any seed 10d ICIR < 0.50 → likely seed bug or data issue, debug before continuing

**Checkpoint**: `git commit -am "[ng150-plan] P4: 3-seed training complete, ICIRs=[X,Y,Z]"`

---

## Phase 4.5: Sanity Checks (15-30 min)

Run the sanity script (create if doesn't exist, pattern from `scripts/sanity_tests.py` for ng1.3.0):

```bash
python3 -c "
import joblib, glob, numpy as np
from pathlib import Path

pkls = sorted(Path('ml_models/trained_models/ng/').glob('ng150_seed*_multi_target_*.pkl'))
assert len(pkls) >= 3, f'Expected 3 pkls, got {len(pkls)}'

# 1. metadata check (Check 9)
for p in pkls[-3:]:
    m = joblib.load(p)
    for field in ['git_commit_hash', 'host', 'training_duration_sec', 'seed', 'feature_names']:
        assert field in m, f'{p.name} missing {field}'
    print(f'  {p.name}: seed={m[\"seed\"]}, commit={m[\"git_commit_hash\"][:8]}, duration={m[\"training_duration_sec\"]:.0f}s')

# 2. feature_names match across seeds
feat_lists = [tuple(joblib.load(p)['feature_names']) for p in pkls[-3:]]
assert all(f == feat_lists[0] for f in feat_lists), 'feature_names differ across seeds'
print(f'  feature_names consistent: {len(feat_lists[0])} features')

# 3. Count should be 78
assert len(feat_lists[0]) == 78, f'Expected 78 features, got {len(feat_lists[0])}'
print('  sanity PASS')
"
```

### Seed propagation check

Generate reports for 2024-06-03 with each seed, check pred correlation:

```bash
for SEED in 42 123 456; do
  # Force load specific pkl
  pkl=$(ls -t ml_models/trained_models/ng/ng150_seed${SEED}_multi_target_*.pkl | head -1)
  python3 backtest/batch_generate_v395_reports.py \
      --version ng1.5.0 --model-path "$pkl" \
      --start-date 2024-06-03 --end-date 2024-06-03 \
      --output-dir reports/_ng150_sanity_seed${SEED} --force
done

python3 -c "
import json, os, numpy as np
preds = []
for seed in [42, 123, 456]:
    with open(f'reports/_ng150_sanity_seed{seed}/analysis_data_20240603.json') as f:
        d = json.load(f)
    p = {s['stock_code']: float(s.get('pred_10d', 0) or 0) for s in d['all_stocks_with_scores']}
    preds.append(p)

# Common keys
keys = set(preds[0]) & set(preds[1]) & set(preds[2])
if len(keys) < 100:
    print(f'WARN only {len(keys)} common stocks')
else:
    from scipy.stats import spearmanr
    arrs = [np.array([preds[i][k] for k in keys]) for i in range(3)]
    for i in range(3):
        for j in range(i+1, 3):
            c, _ = spearmanr(arrs[i], arrs[j])
            print(f'seed {[42,123,456][i]} vs {[42,123,456][j]}: corr={c:.3f}')
# PASS: all correlations in [0.85, 0.95]
# FAIL < 0.85: seed propagation bug OR features too noisy
# FAIL > 0.95: seeds effectively identical, ensemble pointless
"
```

### β attribution artefact

```bash
# First, generate a small WF-OOS report sample (2024-2026)
# (Uses ensemble of 3 seeds — standard scorer does this automatically)
python3 backtest/batch_generate_v395_reports.py --version ng1.5.0 \
    --start-date 2024-01-01 --end-date 2026-04-17 \
    --output-dir reports/daily_selection_ng1.5.0_stage4a --force

# Then run eval to extract β
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_ng1.5.0_stage4a \
    --label NG150-STAGE4A --scoring-version ng1.5.0 \
    --rank-field composite --top-n 10 --focus-days 10 --score-version v52 \
    > logs/ng150_stage4a_eval_$(date +%Y%m%d_%H%M%S).log 2>&1
```

Extract β and verify:
```bash
f=$(ls -t logs/ng150_stage4a_eval_*.log | head -1)
grep -E "MKT=|SMB=|HML=|UMD=|Alpha\(年化\)|加权评分.*→ 等级" "$f"
```

**Gate**: β_UMD ≤ 0.5 absolute (per spec 3.1 minimum)

**Checkpoint**: `git commit -am "[ng150-plan] P4.5: sanity pass, β_UMD=X"`

---

## Phase 5: Stage 3.5 Gate (45 min, 2025 subset)

```bash
python3 scripts/ng130_stage35_gate.py --version ng1.5.0 \
    --start 2025-01-01 --end 2025-12-31 \
    2>&1 | tee logs/ng150_stage35_gate_$(date +%Y%m%d_%H%M%S).log
```

**PASS**: V5.2 raw ≥ 68% (ng1.4.0 Stage 3.5 baseline)
**WARN (60-68%)**: marginal, document concerns, proceed to Stage 4a cautiously
**ABORT (< 60%)**: ng1.5.0 Phase A failed. Proceed to §10 fallback decision.

Write postmortem: `reports/ng150/stage35_pass.md` or `stage35_rejected.md` (mirror ng140 pattern).

**Checkpoint**: `git commit -am "[ng150-plan] P5: Stage 3.5 V5.2=X%"`

---

## Phase 6: Stage 4a Gate (1h, 2024-2026 cross-regime)

```bash
python3 scripts/ng130_stage35_gate.py --version ng1.5.0 \
    --start 2024-01-01 --end 2026-04-17 \
    2>&1 | tee logs/ng150_stage4a_gate_$(date +%Y%m%d_%H%M%S).log
```

**Decision matrix (check against spec Section 3)**:

| V5.2 | MaxDD | Sharpe | Decision |
|---|---|---|---|
| ≥ 75% | ≤ -15% | ≥ 2.8 | 🎯 Target success, proceed to Phase 7 |
| ≥ 72% | ≤ -15% | ≥ 2.5 | ✅ Minimum success, proceed cautiously |
| ≥ 72% | -15% ~ -18% | any | ⚠️ Launch Phase B (CVaR loss), spec §6 |
| < 72% | any | any | ❌ Launch Phase C (ensemble fallback), spec §6 |

Also check monthly IC negative ratio < 25% (ng1.3.0 was 33% which signals overfitting).

Write: `reports/ng150/stage4a_{pass,warn,rejected}.md`.

**Checkpoint**: `git commit -am "[ng150-plan] P6: Stage 4a V5.2=X%, MaxDD=Y%"`

---

## Phase 7: Stage 4b Gate (2h, Pre-2020)

Check ng150 cache has 2018-2019:
```bash
python3 -c "
import sqlite3
c = sqlite3.connect('data_adapter/stock_data.db')
n = c.execute('SELECT COUNT(*), COUNT(DISTINCT trade_date) FROM ng150_feature_cache WHERE trade_date BETWEEN ? AND ?', ('2018-01-01', '2019-12-31')).fetchone()
print(f'Pre-2020 cache: {n[0]} rows, {n[1]} dates')
"
# If dates < 200: backfill first (Phase 2 should have covered, but verify)
```

Generate reports + evaluate:

```bash
python3 backtest/batch_generate_v395_reports.py --version ng1.5.0 \
    --start-date 2018-01-01 --end-date 2019-12-31 \
    --output-dir reports/daily_selection_ng1.5.0_pre2020 --force

python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_ng1.5.0_pre2020 \
    --label NG150-PRE2020 --scoring-version ng1.5.0 \
    --rank-field composite --top-n 10 --focus-days 10 --score-version v52 \
    2>&1 | tee logs/ng150_stage4b_$(date +%Y%m%d_%H%M%S).log
```

**Gate** (spec 3.1 minimum):
- 净年化 ≥ 0% (strict improvement over ng1.0.1's -19%)
- Sharpe ≥ 0
- V5.2 ≥ 50%

**If 净年化 < -5%**: Pre-2020 退化严重, don't switch production even if Stage 4a PASS. Log it, treat ng1.5.0 as "WF-OOS-only" model.

**Checkpoint**: `git commit -am "[ng150-plan] P7: Stage 4b 年化=X%, Sharpe=Y"`

---

## Phase 8: Observation Period (1-2 weeks, human)

During this phase, the model stays at ng1.0.1 in production but ng1.5.0 generates parallel reports daily.

**Automation** (add to daily cron / `run_daily_update.sh`):

```bash
# After normal ng1.0.1 report generated, also run ng1.5.0
python3 backtest/batch_generate_v395_reports.py \
    --version ng1.5.0 \
    --start-date "$(date +%Y-%m-%d)" --end-date "$(date +%Y-%m-%d)" \
    --output-dir reports/daily_selection_ng150_shadow --force
```

Daily diff check (Human-in-loop):
```bash
python3 -c "
import json, os
d = '$(date +%Y%m%d)'
with open(f'reports/daily_selection_ng101/analysis_data_{d}.json') as f:
    a = json.load(f)
with open(f'reports/daily_selection_ng150_shadow/analysis_data_{d}.json') as f:
    b = json.load(f)
# Top 10 overlap
top_a = {s['stock_code'] for s in a['all_stocks_with_scores'][:10]}
top_b = {s['stock_code'] for s in b['all_stocks_with_scores'][:10]}
overlap = top_a & top_b
print(f'Top10 overlap: {len(overlap)}/10 = {overlap}')
print(f'ng101 only: {top_a - top_b}')
print(f'ng150 only: {top_b - top_a}')
"
```

**Flags to investigate**:
- Top10 overlap < 3 → too disruptive, review
- Single industry concentration > 40% in ng1.5.0 picks
- Any "surprise" stock with no apparent reason (human judgment)

**Criteria for proceeding to Phase 9**:
- At least 10 trading days of shadow
- Top10 overlap avg ≥ 5/10
- No crashed pipelines
- User approves the 差异 pattern

---

## Phase 9: Production Switch (30 min)

### 9.1 Code change

`ml_models/ng/ng_schema.py`:
```python
PRODUCTION_VERSION = 'ng1.5.0'  # was 'ng1.0.1'
```

Verify downstream (should auto-pick up via `get_table_name()` and selector default):
```bash
grep -rn "PRODUCTION_VERSION\|'ng1.0.1'" ml_models/ data_adapter/ tomorrow_stock_selector.py | head -20
```

### 9.2 Documentation updates

Update (details in each file's 2026-04-20 audit reference):

- `CLAUDE.md` ML Scoring Systems section: swap ng1.0.1 ↔ ng1.5.0 as the 🏆 production entry
- `MEMORY.md`: add new top entry "ng1.5.0 切换生产 (yyyy-MM-dd)", update `ng_production_switch.md` detail file
- `docs/wiki/models/ng-series.md`: add ng1.5.0 to rankings
- `docs/wiki/log.md`: add `yyyy-MM-dd | model | NG v1.5.0 发布 — ...`

### 9.3 Smoke test

```bash
python3 tomorrow_stock_selector.py $(date +%Y-%m-%d) --scoring-version ng1.5.0 2>&1 | tail -20
# Should complete without error, produce top-10 picks
```

### 9.4 Close plan

**Final checkpoint**: `git commit -am "[ng150-plan] P9: production switched to ng1.5.0"`

Write: `reports/ng150/production_switch.md` summarizing:
- Final metrics (WF-OOS, Pre-2020, β)
- Rollback procedure (revert PRODUCTION_VERSION to 'ng1.0.1' + restart daily cron)
- Known limitations

---

## 10. Fallback Decisions

### If Stage 3.5 FAIL (Phase 5, V5.2 < 60%)

→ **Phase A failed**. 5 new features are not additive over ng1.4.0 base. Actions:
1. Do not proceed to Phase 6/7
2. Run ablation: drop each new feature in turn, re-eval Stage 3.5
3. Identify which feature(s) harmful
4. Write `reports/ng150/stage35_rejected.md` with postmortem
5. Two options:
   - **Option X**: keep ng1.5.0 = ng1.4.0 + only harmless new features (if any), restart from Phase 3
   - **Option Y**: accept defeat, cancel ng1.5.0, wait for next iteration design

### If Stage 4a fails with MaxDD > -18% (Phase 6)

→ Launch **Phase B: CVaR Regularization** (spec §6). Estimated +10h.

Implementation sketch:
1. In `ng_trainer.py`, only for LGB / XGB / HGB (skip RF/CB which don't support custom loss):
   - Define custom `fobj(preds, dataset)` returning `grad = MSE_grad + λ * CVaR_5%_grad_per_stock`
   - λ values: 0 (sanity), 0.1, 0.3
2. Re-train seed 42 with each λ
3. Pick best λ by Stage 3.5 V5.2 (tiebreak: MaxDD)
4. Retrain full 3-seed with chosen λ
5. Re-run Phase 5-7 gates

### If Stage 4a fails with V5.2 < 70% (Phase 6)

→ Launch **Phase C: Regime-Aware Ensemble** (spec §6). Estimated +3h, no retraining.

Implementation sketch:
1. Create `ml_models/ng/ng_production_scorer.py::NG150EnsembleScorer`:
   - Load both ng1.0.1 pkl and ng1.0.6 switchover (its two internal pkls ng101 + ng104-3s)
   - For each prediction, compute `bull_prob` from AMV features
   - Return `final = bull_prob * ng101_score + (1 - bull_prob) * ng106_score`
2. Register in `batch_generate_v395_reports.py` (version handler block)
3. Re-run Stage 6+7 with `--scoring-version ng1.5.0` routing to ensemble
4. If PASS: ng1.5.0 IS the ensemble (not a new model). Document as such in wiki.

### If Stage 4b fails (Pre-2020 净年化 < -5%) (Phase 7)

→ Don't switch production. Options:
1. Keep ng1.0.1 as production, archive ng1.5.0 as "WF-OOS-only" candidate
2. Continue Phase 8 observation for WF-OOS regime only, revisit in 3-6 months
3. Consider Phase C ensemble (if not already tried)

---

## 11. Reference Index (for the fresh session)

### Key files to know
- Spec: `docs/superpowers/specs/2026-04-20-ng150-regime-refined-design.md`
- This plan: `docs/superpowers/plans/2026-04-20-ng150-implementation.md`
- ng1.4.0 plan: `docs/ng140_plan.md`
- ng1.4.0 Stage 3.5 result: `reports/ng140/stage35_pass.md`
- ng1.0.1 audit: `memory/ng101_pre2020_audit_2026_04_20.md`
- 2026-04-20 log entries: `docs/wiki/log.md` lines 9-11
- Failed iterations postmortems: `memory/ng123_rejected.md`, `memory/ng124_plan.md`, `memory/ng111_abandoned.md`, `memory/ng12x_iteration_2026_04_14.md`

### Key commands reference
```bash
# Training
python3 ml_models/ng/ng_trainer.py --version ng1.5.0 --seed 42 --target-parallel 4

# Cache backfill (--num-workers 0 implicit for ng_cache_updater)
python3 ml_models/ng/ng_cache_updater.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD --version ng1.5.0

# Report generation
python3 backtest/batch_generate_v395_reports.py --version ng1.5.0 --start-date ... --end-date ... --output-dir ... --force

# Gate
python3 scripts/ng130_stage35_gate.py --version ng1.5.0 --start ... --end ...

# Full eval
python3 backtest/run_north_star_eval.py --backtest --report-dir ... --scoring-version ng1.5.0 --rank-field composite --top-n 10 --focus-days 10 --score-version v52
```

### Key numerical baselines (from 2026-04-20 audit, do not trust older numbers)

| Model | WF-OOS V5.2 | WF-OOS Sharpe (10d) | MaxDD | Pre-2020 年化 | β_UMD |
|---|---|---|---|---|---|
| ng1.0.1 (4-12 bugfix) | 73.4% A+ | 2.367 | **-11.7%** | -19.0% | +0.38 |
| ng1.0.6 (0AMV switch) | **78.9% A+** | **2.808** | -22.9% | **+0.7%** | +0.005 |
| ng1.4.0 (Stage 3.5) | 68% (partial) | — | — | — | — |
| **ng1.5.0 target** | **≥ 75%** | **≥ 2.8** | **≤ -15%** | **≥ 0%** | **≤ 0.25** |

### Long-running commands use `run_in_background: true`

For training (6-10h) / backfill (2h) / Stage 4a eval (1h+): always use background + Monitor (Claude Code tool). Do not block the main conversation.

---

## 12. Session Handoff Notes

If you're the session picking this up:

1. Run `git log --oneline | head -30 | grep 'ng150-plan'` — last commit shows exact phase completed
2. Read last log file under `logs/ng150_*` to see where things left off
3. Do the mandatory pre-read (§0) — the 2026-04-20 audit is **essential** context, do not skip
4. Validate current state:
   ```bash
   # Is ng150 cache built?
   python3 -c "import sqlite3; c=sqlite3.connect('data_adapter/stock_data.db'); print(c.execute('SELECT COUNT(*) FROM ng150_feature_cache').fetchone())"

   # Are pkls trained?
   ls ml_models/trained_models/ng/ng150_seed*_multi_target_*.pkl 2>/dev/null | wc -l

   # Reports generated?
   ls reports/daily_selection_ng1.5.0_* 2>/dev/null
   ```

5. Pick up at the next phase.

**Do not** retrain or re-backfill if artefacts already exist — check mtime and contents first.

---

**End of plan.**
