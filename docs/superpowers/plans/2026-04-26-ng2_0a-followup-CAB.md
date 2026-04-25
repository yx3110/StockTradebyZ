# ng2.0a Follow-up Implementation Plan: C → A → B

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 三阶段闭环 ng2.0a 余下工作 — (C) 紧缩 B1/B2 calibration 修 Pre-2020 over-call → (A) 把 v2 regime + ng1.0.1 bull 配置接进生产 selector → (B) ng2.0b sample-weighted sub-model 重训 拿 ≤-18% MaxDD 目标。

**Architecture:** Phase C 试 4 个 calibration 变体扫描出最优 hysteresis/streak 参数, 写回 `market_regime_signals`。Phase A 在 `tomorrow_stock_selector.py` 加 `--scoring-version ng2.0a` 路由 (regime_v2 → bull=ng1.0.1 / bear=ng1.0.4-3s), 灰度对比生产 ng106v2。Phase B 用现有 `compute_sample_weights` hook 加 regime 权重维度, 重训 ng2.0b-bull (bull regime ×2) 和 ng2.0b-bear (bear regime ×2), 跑 Step B+ eval。

**Tech Stack:** Python 3, SQLite, pandas, numpy, pytest, 现有 `indicators/` + `ml_models/ng/` + `backtest/` + `tomorrow_stock_selector.py` 框架

**Spec:** `docs/superpowers/specs/2026-04-25-ng200-regime-refined-architecture-design.md`

**Predecessor plan:** `docs/superpowers/plans/2026-04-25-ng200a-multi-beta-regime.md` — Tasks 1-7 完成 (B1/B2/regime_v2 模块 + DB + 回填 + Step A PASS + regime-version dispatch); Task 8 Step B 评估发现关键洞察 (见下文 Context)。

---

## Context (REQUIRED READING — 新 session 必读, 决策依赖此处)

### ng2.0a 已完成的事实

1. **Step A PASS (commit `35cec18d`)**: regime_v2 (V11+B1+B2 hard vote) vs V11 baseline (regime_v1):
   - 1526 trading days (2020-2026): 84.2% agreement, ratio v2/v1 flips=0.87x (calmer), Δ%bull=+7.14pp
   - 2018 Q4 sanity: v2 bear days 37/59 (v1: 38) — 都识别熊市
   - **2019 Q1 sanity: v2 bull days 40/56 (v1: 26) — v2 抓住反弹起点早 14 天**

2. **Step B 原版 (commit `1916874b`, 用 ng106v2 sub-models = ng1.0.7 bull / ng1.0.4-3s bear)**:
   - WF-OOS 2020-2026: V5.2 v1=80.4%S, v2=80.2%S (Δ=-0.2pp 噪声), Sharpe v2 +0.20, MaxDD v2 +1.4pp
   - Pre-2020: V5.2 v1=33.4%C, v2=32.5%C (Δ=-0.9pp), MaxDD v1=-31.7% / v2=-45.6% (Δ=-13.9pp)
   - implementer 第一遍判 ABORT, 但用户挑战
   
3. **Step B Option C 复测 (commit `0ce52658`, 换 ng106v1 sub-models = ng1.0.1 bull / ng1.0.4-3s bear)**:
   - WF-OOS: V5.2 v1=76.5%A+, **v2=79.3%A+ (Δ=+2.8pp 真实)**, MaxDD v1=-26.8% / **v2=-17.6% (Δ=+9.2pp)**
   - Pre-2020: V5.2 v1=40.7%C, v2=37.2%C (Δ=-3.5pp), 净年化 v1=+3.0% / v2=-10.3% (Δ=-13.3pp)
   - **v2 regime 配 ng1.0.1 bull 在 WF-OOS 上明显赢生产, MaxDD 大幅改善, Sharpe 改善**

4. **Day-by-day forward-return 验证 (current session)**: v2 在 WF-OOS 175 天 v2-added-bull 上的 bull-vs-bear 选股差异 -0.13pp/day (噪声), 在 Pre-2020 42 天 v2-added-bull 上 +0.32pp/day (v2 略对). 说明 **v2 regime 信号本身是 sound 的**, Pre-2020 -45.6% MaxDD 是 sub-model 交互 + cost penalty 的 north-star eval artifact, 不是 regime 错判。

### 当前生产配置 (ng106v2 = `--scoring-version ng1.0.62`)

- `tomorrow_stock_selector.py:5775` 把 `ng1.0.62` 进入 ng106 mode, bull=ng1.0.7, bear=ng1.0.4-3s
- 报告写 `reports/daily_selection_ng106v2/`
- WF-OOS V5.2=80.4% S, MaxDD=-23.7%

### 用户决策 (2026-04-25)

- 喜欢 v2+ng101 的 MaxDD 改善 (-17.6% vs -23.7% 生产 = 6pp 改善)
- 选择 C → A → B 的执行顺序:
  - **C 先**: 修 B1/B2 在 2018-2019 的 over-call (33 多余 bull 天) 试图把 Pre-2020 拉回正
  - **A 中**: 把验证过的 v2+ng101 配置 (或 C 优化后的 v2+ng101) 接进 selector 作为新 scoring-version `ng2.0a`
  - **B 后**: ng2.0b 重训 bull/bear sub-model 用 sample weights, 拿 ≤-18% MaxDD 目标

---

## File Structure

### Phase C (B1/B2 tightening)

**New files:**
- `scripts/ng2_0a_calibration_sweep.py` — 一次性 sweep 4 个 variant 写报告
- `scripts/build_regime_v2_history_variant.py` — 参数化 build_regime_v2 backfill (基于 Task 5)
- `reports/ng2_0a_calibration_sweep_results.md` — sweep 结果对比表

**Modified files:**
- `indicators/breadth.py` — 已支持 `hysteresis_lo/hi/streak_days` 参数, 无需修改
- `indicators/realized_vol.py` — 已支持参数, 无需修改
- `indicators/regime_classifier.py:compute_regime_v2` — 已支持 `system_streak`, 加 `vote_threshold` 参数 (默认 2 = 多数; 改 3 = 全票)
- `data_adapter/stock_data.db` — 用 `--variant {v2_default,v2_strict_b1,v2_strict_b2,v2_streak5,v2_unanimous}` 命名后缀新建/复用 `market_regime_signals_<variant>` 表 OR 在主表加列

### Phase A (production wiring)

**New files:**
- `ml_models/ng/ng_2_0a_scorer.py` — `NG200aScorer` 类, 实现 v2 regime 路由
- `docs/wiki/architecture/ng2_0a_multi_beta_regime.md` — wiki 文档

**Modified files:**
- `tomorrow_stock_selector.py` (line 5775 区段, line 6204 valid version list) — 加 `ng2.0a` 路由
- `CLAUDE.md` — 加 `ng2.0a` 到 ML Scoring Systems 活跃版本列表
- `~/.claude/projects/-Users-yangxu-StockTradebyZ/memory/MEMORY.md` + `ng2_0a_production.md` — 记录上线

### Phase B (sample-weighted retrain)

**New files:**
- `ml_models/ng/ng_2_0b_trainer.py` — wrapper script 调用现有 `train_v395_multi_target.py` + 加 regime weight
- `docs/superpowers/plans/2026-05-XX-ng2_0b-sample-weighted-retrain.md` — 详细子 plan (在 Phase A 收尾后写, 跑训练前再展开)

**Modified files:**
- `ml_models/training/train_v395_multi_target.py:2034` (`compute_sample_weights`) — 加 `--regime-weight bull|bear|none` CLI flag, 默认 none 保 backward compat
- 训练后产物: `ml_models/trained_models/ng/ng2_0b_bull_*.pkl`, `ng2_0b_bear_*.pkl`

---

# PHASE C: B1/B2 Calibration Tightening

**Goal:** 试 4 个 variant 找最优 calibration, 把 v2 regime 在 Pre-2020 的 over-call (currently +33 bull 天) 减少, 同时不丢 WF-OOS V5.2 (currently v2+ng101 = 79.3%)。

**Acceptance criteria:**
- Pre-2020 V5.2 ≥ 38% (from current 37.2%, +0.8pp)
- Pre-2020 net annual ≥ -8% (from current -10.3%, +2.3pp)
- WF-OOS V5.2 ≥ 78% (允许从 79.3% 损失 ≤ 1.3pp 换 Pre-2020 改善)
- 至少 1 个 variant 通过, 否则 Phase C 整体 ABORT, 走 default v2 → Phase A

**Variants to test:**

| Variant | B1 (lo, hi) | B2 (lo, hi) | system_streak | vote_threshold |
|---|---|---|---|---|
| V0 baseline (current) | (0.45, 0.55) | (0.30, 0.70) | 3 | 2 |
| V1 strict_b1 | (0.40, 0.65) | (0.30, 0.70) | 3 | 2 |
| V2 strict_b2 | (0.45, 0.55) | (0.25, 0.75) | 3 | 2 |
| V3 streak5 | (0.45, 0.55) | (0.30, 0.70) | 5 | 2 |
| V4 unanimous | (0.45, 0.55) | (0.30, 0.70) | 3 | 3 |

---

## Task C1: Add `vote_threshold` parameter to compute_regime_v2

**Files:**
- Modify: `indicators/regime_classifier.py:compute_regime_v2`
- Modify: `tests/indicators/test_regime_v2.py` — 加 unanimous variant test

- [ ] **Step 1: 写失败测试**

Append to `tests/indicators/test_regime_v2.py`:

```python
def test_regime_v2_unanimous_vote_requires_all_3():
    """vote_threshold=3 means all 3 signals must be bull to flip bull."""
    n = 30
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    # 2-of-3 bull (V11+B1, B2 bear) — should NOT trigger bull when threshold=3
    v11 = pd.Series([1] * n, index=idx)
    b1 = pd.Series([1] * n, index=idx)
    b2 = pd.Series([0] * n, index=idx)
    out = compute_regime_v2(v11, b1, b2, system_streak=3, vote_threshold=3)
    assert out['vote_count'].iloc[-1] == 2
    # raw should be -1 (bear) since vote=2 < threshold=3
    assert out['regime_v2_raw'].iloc[-1] == -1
    assert out['regime_v2'].iloc[-1] == -1


def test_regime_v2_unanimous_vote_3_of_3_bull():
    """All 3 bull with threshold=3 should still produce bull."""
    n = 30
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    v11 = pd.Series([1] * n, index=idx)
    b1 = pd.Series([1] * n, index=idx)
    b2 = pd.Series([1] * n, index=idx)
    out = compute_regime_v2(v11, b1, b2, system_streak=3, vote_threshold=3)
    assert out['vote_count'].iloc[-1] == 3
    assert (out['regime_v2'].iloc[3:] == 1).all()


def test_regime_v2_default_threshold_is_majority_2():
    """Default vote_threshold=2 should preserve backward-compat (majority vote)."""
    n = 30
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    v11 = pd.Series([1] * n, index=idx)
    b1 = pd.Series([1] * n, index=idx)
    b2 = pd.Series([0] * n, index=idx)
    # Don't pass vote_threshold — should default to 2
    out = compute_regime_v2(v11, b1, b2, system_streak=3)
    assert out['regime_v2_raw'].iloc[-1] == 1  # 2 of 3 = bull at default threshold
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/indicators/test_regime_v2.py -v -k unanimous
```

Expected: `TypeError: compute_regime_v2() got an unexpected keyword argument 'vote_threshold'`

- [ ] **Step 3: Add `vote_threshold` parameter to compute_regime_v2**

Edit `indicators/regime_classifier.py`. Find `def compute_regime_v2(` (around line 209). Update signature and `raw_bull_int` calculation:

```python
def compute_regime_v2(
    v11_bull: pd.Series,
    b1_bull: pd.Series,
    b2_bull: pd.Series,
    system_streak: int = 3,
    vote_threshold: int = 2,
) -> pd.DataFrame:
    """ng2.0a: hard vote across 3 binary signals + system-level streak.

    Args:
        v11_bull, b1_bull, b2_bull: each 1=bull, 0=bear, NaN allowed (treated as 0).
            All three Series must share the same DatetimeIndex.
        system_streak: streak_days at vote-output level (default 3).
        vote_threshold: minimum bull count to call bull (default 2 = majority;
            3 = unanimous).

    Returns:
        DataFrame indexed same as inputs with columns:
            vote_count: int 0..3 (count of bulls)
            regime_v2_raw: +1 if vote >= vote_threshold else -1
            regime_v2_streak: consecutive raw days at current majority side
            regime_v2: +1/-1 (after system_streak applied to raw)
    """
    if not (v11_bull.index.equals(b1_bull.index) and v11_bull.index.equals(b2_bull.index)):
        raise ValueError('v11_bull / b1_bull / b2_bull indices must match')
    if vote_threshold not in (1, 2, 3):
        raise ValueError(f'vote_threshold={vote_threshold} must be in {{1, 2, 3}}')

    v = v11_bull.fillna(0).astype(int)
    b1 = b1_bull.fillna(0).astype(int)
    b2 = b2_bull.fillna(0).astype(int)

    vote = v + b1 + b2  # 0..3
    raw_bull_int = (vote >= vote_threshold).astype(int)

    raw_arr = raw_bull_int.to_numpy()
    confirmed = persist_n(raw_arr, n_days=system_streak)

    grp = (raw_bull_int != raw_bull_int.shift()).cumsum()
    streak = (raw_bull_int.groupby(grp).cumcount() + 1).astype(np.int32)

    out = pd.DataFrame({
        'vote_count': vote.astype('Int8'),
        'regime_v2_raw': np.where(raw_bull_int == 1, 1, -1).astype(np.int8),
        'regime_v2_streak': streak,
        'regime_v2': confirmed.astype(np.int8),
    }, index=v11_bull.index)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python3 -m pytest tests/indicators/ -v
```

Expected: All previous tests + 3 new tests pass (19/19 total).

- [ ] **Step 5: /simplify pass**

Re-read the function. Concerns to check:
- Is `vote_threshold ∈ {1, 2, 3}` validation correct? (Yes — vote sum is 0..3, threshold of 0 would always-bull and 4 would always-bear, both useless.)
- Backward compat: existing callers passing only positional args still get default `vote_threshold=2` → identical behavior.

- [ ] **Step 6: Commit**

```bash
git add indicators/regime_classifier.py tests/indicators/test_regime_v2.py
git commit -m "feat(ng2.0a-v2): add vote_threshold param to compute_regime_v2 (default=2 majority, 3=unanimous)"
```

---

## Task C2: Build parameterized regime backfill helper

**Files:**
- Create: `scripts/build_regime_v2_history_variant.py`

- [ ] **Step 1: Write the script**

Create `scripts/build_regime_v2_history_variant.py`:

```python
"""Backfill market_regime_signals_<variant> for a parameterized B1/B2/vote setting.

Difference from build_regime_v2_history.py:
  - Accepts B1/B2/vote params via CLI
  - Writes to a per-variant table name `market_regime_signals_{variant}` so the original
    `market_regime_signals` table (= V0 baseline) remains untouched.

Usage:
    python3 scripts/build_regime_v2_history_variant.py \\
        --variant strict_b1 \\
        --b1-lo 0.40 --b1-hi 0.65 \\
        --start 2018-01-01 --end 2026-04-25
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indicators.breadth import compute_breadth_signal
from indicators.realized_vol import compute_realized_vol_signal
from indicators.regime_classifier import compute_regime_v2

DB_PATH = Path(__file__).resolve().parents[1] / 'data_adapter' / 'stock_data.db'
HS300_CODE = '000300.SH'

# Reuse the data-load helpers from build_regime_v2_history.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_regime_v2_history import load_close_panel, load_index_close, load_v11_regime


DDL_TEMPLATE = """
CREATE TABLE IF NOT EXISTS {table} (
    trade_date TEXT PRIMARY KEY,
    v11_var1 REAL,
    v11_ma60 REAL,
    v11_macd REAL,
    v11_bull INTEGER,
    v11_streak INTEGER,
    b1_pct_above_ma20 REAL,
    b1_pct_above_ma60 REAL,
    b1_adv_dec_ratio REAL,
    b1_score REAL,
    b1_bull INTEGER,
    b1_streak INTEGER,
    b2_rv_60d REAL,
    b2_rv_percentile_252 REAL,
    b2_bull INTEGER,
    b2_streak INTEGER,
    vote_count INTEGER,
    regime_v2_raw INTEGER,
    regime_v2_streak INTEGER,
    regime_v2 INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', required=True,
                   choices=['baseline', 'strict_b1', 'strict_b2', 'streak5', 'unanimous'])
    p.add_argument('--b1-lo', type=float, default=0.45)
    p.add_argument('--b1-hi', type=float, default=0.55)
    p.add_argument('--b2-lo', type=float, default=0.30)
    p.add_argument('--b2-hi', type=float, default=0.70)
    p.add_argument('--system-streak', type=int, default=3)
    p.add_argument('--vote-threshold', type=int, default=2)
    p.add_argument('--start', default='2018-01-01')
    p.add_argument('--end', default='2026-04-25')
    args = p.parse_args()

    table = f'market_regime_signals_{args.variant}'
    print(f'Variant: {args.variant} -> table {table}')
    print(f'  B1=({args.b1_lo}, {args.b1_hi}) B2=({args.b2_lo}, {args.b2_hi}) '
          f'streak={args.system_streak} vote_threshold={args.vote_threshold}')

    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.execute('PRAGMA busy_timeout=30000')
    conn.executescript(DDL_TEMPLATE.format(table=table))

    try:
        v11_df = load_v11_regime(conn, args.start, args.end)
        lookback_b1 = (pd.Timestamp(args.start) - pd.Timedelta(days=120)).strftime('%Y-%m-%d')
        panel = load_close_panel(conn, lookback_b1, args.end)
        b1 = compute_breadth_signal(
            panel, ma_short=20, ma_long=60,
            streak_days=3,  # B1's own streak fixed; system_streak applies at vote level
            hysteresis_lo=args.b1_lo, hysteresis_hi=args.b1_hi,
        )
        b1 = b1.loc[b1.index >= pd.Timestamp(args.start)]

        lookback_b2 = (pd.Timestamp(args.start) - pd.Timedelta(days=500)).strftime('%Y-%m-%d')
        idx_close = load_index_close(conn, HS300_CODE, lookback_b2, args.end)
        b2 = compute_realized_vol_signal(
            idx_close, rv_window=60, percentile_window=252, streak_days=3,
            hysteresis_lo=args.b2_lo, hysteresis_hi=args.b2_hi,
        )
        b2 = b2.loc[b2.index >= pd.Timestamp(args.start)]

        common = v11_df.index.intersection(b1.index).intersection(b2.index)
        v11_a = v11_df.loc[common]
        b1_a = b1.loc[common]
        b2_a = b2.loc[common]
        vote = compute_regime_v2(
            v11_a['v11_bull'].astype(int),
            b1_a['b1_bull'].fillna(0).astype(int),
            b2_a['b2_bull'].fillna(0).astype(int),
            system_streak=args.system_streak,
            vote_threshold=args.vote_threshold,
        )

        merged = pd.concat([v11_a, b1_a, b2_a, vote], axis=1)
        merged.index.name = 'trade_date'
        merged = merged.reset_index()
        merged['trade_date'] = merged['trade_date'].dt.strftime('%Y-%m-%d')
        merged['b1_adv_dec_ratio'] = None

        cols = [
            'trade_date',
            'var1', 'ma60', 'macd', 'v11_bull', 'v11_streak',
            'pct_above_ma20', 'pct_above_ma60', 'b1_adv_dec_ratio', 'b1_score', 'b1_bull', 'b1_streak',
            'rv_60d', 'rv_percentile_252', 'b2_bull', 'b2_streak',
            'vote_count', 'regime_v2_raw', 'regime_v2_streak', 'regime_v2',
        ]
        db_cols = [
            'trade_date',
            'v11_var1', 'v11_ma60', 'v11_macd', 'v11_bull', 'v11_streak',
            'b1_pct_above_ma20', 'b1_pct_above_ma60', 'b1_adv_dec_ratio', 'b1_score', 'b1_bull', 'b1_streak',
            'b2_rv_60d', 'b2_rv_percentile_252', 'b2_bull', 'b2_streak',
            'vote_count', 'regime_v2_raw', 'regime_v2_streak', 'regime_v2',
        ]
        out = merged[cols].rename(columns=dict(zip(cols, db_cols)))

        # Idempotent: delete then insert
        conn.execute(f'DELETE FROM {table}')
        out.to_sql(table, conn, if_exists='append', index=False)
        conn.commit()

        bull_n = int((out['regime_v2'] == 1).sum())
        bear_n = int((out['regime_v2'] == -1).sum())
        print(f'  written {len(out)} rows: bull={bull_n}, bear={bear_n}')

        # Pre-2020 vs WF-OOS bull breakdown for quick visibility
        pre = out[out['trade_date'] < '2020-01-01']
        wfo = out[out['trade_date'] >= '2020-01-01']
        print(f'  Pre-2020: bull={int((pre["regime_v2"] == 1).sum())}/{len(pre)}')
        print(f'  WF-OOS:   bull={int((wfo["regime_v2"] == 1).sum())}/{len(wfo)}')
    finally:
        conn.close()


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Smoke-test on baseline (must reproduce existing market_regime_signals counts)**

```bash
mkdir -p logs
python3 scripts/build_regime_v2_history_variant.py --variant baseline --start 2018-01-01 --end 2026-04-25 \
    2>&1 | tee logs/ng2_0a_phase_c_smoke_baseline.log
```

Expected: writes 1943 rows; bull=745 / bear=1198 (matches predecessor `market_regime_signals` from Task 5 of previous plan).

Verify:
```bash
python3 -c "
import sqlite3
c = sqlite3.connect('data_adapter/stock_data.db', timeout=30)
n_b = c.execute('SELECT COUNT(*) FROM market_regime_signals WHERE regime_v2=1').fetchone()[0]
n_v = c.execute('SELECT COUNT(*) FROM market_regime_signals_baseline WHERE regime_v2=1').fetchone()[0]
print(f'main bull={n_b}, baseline-variant bull={n_v}, match={n_b == n_v}')
assert n_b == n_v, 'baseline variant must match main table'
"
```

Expected: prints `main bull=745, baseline-variant bull=745, match=True`.

- [ ] **Step 3: Commit**

```bash
git add scripts/build_regime_v2_history_variant.py
git commit -m "feat(ng2.0a-v2): parameterized regime backfill (per-variant table)"
```

---

## Task C3: Run all 4 variant backfills

- [ ] **Step 1: V1 strict_b1 (tighter B1 hysteresis)**

```bash
python3 scripts/build_regime_v2_history_variant.py --variant strict_b1 \
    --b1-lo 0.40 --b1-hi 0.65 \
    --start 2018-01-01 --end 2026-04-25 \
    2>&1 | tee logs/ng2_0a_phase_c_backfill_strict_b1.log
```

Expected: writes 1943 rows. Note Pre-2020 bull count — should be < 104 (V0 baseline) since B1 is now harder to flip bull.

- [ ] **Step 2: V2 strict_b2 (tighter B2 percentile)**

```bash
python3 scripts/build_regime_v2_history_variant.py --variant strict_b2 \
    --b2-lo 0.25 --b2-hi 0.75 \
    --start 2018-01-01 --end 2026-04-25 \
    2>&1 | tee logs/ng2_0a_phase_c_backfill_strict_b2.log
```

- [ ] **Step 3: V3 streak5 (longer system streak)**

```bash
python3 scripts/build_regime_v2_history_variant.py --variant streak5 \
    --system-streak 5 \
    --start 2018-01-01 --end 2026-04-25 \
    2>&1 | tee logs/ng2_0a_phase_c_backfill_streak5.log
```

- [ ] **Step 4: V4 unanimous (vote >= 3 required)**

```bash
python3 scripts/build_regime_v2_history_variant.py --variant unanimous \
    --vote-threshold 3 \
    --start 2018-01-01 --end 2026-04-25 \
    2>&1 | tee logs/ng2_0a_phase_c_backfill_unanimous.log
```

- [ ] **Step 5: Verify all 5 variant tables present + bull-day counts spread**

```bash
python3 -c "
import sqlite3
c = sqlite3.connect('data_adapter/stock_data.db', timeout=30)
for v in ('baseline', 'strict_b1', 'strict_b2', 'streak5', 'unanimous'):
    t = f'market_regime_signals_{v}'
    pre_b = c.execute(f\"SELECT COUNT(*) FROM {t} WHERE trade_date < '2020-01-01' AND regime_v2 = 1\").fetchone()[0]
    pre_n = c.execute(f\"SELECT COUNT(*) FROM {t} WHERE trade_date < '2020-01-01'\").fetchone()[0]
    wfo_b = c.execute(f\"SELECT COUNT(*) FROM {t} WHERE trade_date >= '2020-01-01' AND regime_v2 = 1\").fetchone()[0]
    wfo_n = c.execute(f\"SELECT COUNT(*) FROM {t} WHERE trade_date >= '2020-01-01'\").fetchone()[0]
    print(f'{v:12} Pre-2020: {pre_b}/{pre_n} bull, WF-OOS: {wfo_b}/{wfo_n} bull')
"
```

Expected: `unanimous` Pre-2020 bull count is much lower (~30-50?) than `baseline` (104). `strict_b1` / `strict_b2` somewhere in between.

- [ ] **Step 6: Commit logs**

```bash
git add logs/ng2_0a_phase_c_backfill_*.log
git commit -m "chore(ng2.0a-v2): backfill 4 calibration variant tables (strict_b1/b2/streak5/unanimous)"
```

(Note: `logs/` is gitignored per `.gitignore`, so this commit will be empty if log files don't get staged. That's OK — skip the commit if `git diff --cached` shows nothing. The DB tables ARE the durable artifact.)

---

## Task C4: Add `--regime-table` flag to regime_switch_backtest.py

**Files:**
- Modify: `backtest/regime_switch_backtest.py`

The current `load_regime` uses hardcoded table names (`market_amv` for v1, `market_regime_signals` for v2). For variant testing we need to read from `market_regime_signals_<variant>`.

- [ ] **Step 1: Read existing load_regime function**

```bash
grep -n "def load_regime" backtest/regime_switch_backtest.py
```

Expected: function defined around line 22-42 (after Task 7 of predecessor plan modified it to support `version='v2'`).

- [ ] **Step 2: Modify load_regime to accept a custom table name**

Edit `backtest/regime_switch_backtest.py`. Find `def load_regime(db_path=None, version: str = 'v1'):` and update:

```python
def load_regime(db_path=None, version: str = 'v1', table: str = None):
    """Load daily regime from DB.

    version='v1': read market_amv.amv_regime
    version='v2': read market_regime_signals.regime_v2 (or `table` arg if given)
    table: optional override for v2 (e.g. 'market_regime_signals_unanimous')
    """
    if db_path is None:
        db_path = DB_PATH
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute('PRAGMA busy_timeout=30000')
    try:
        if version == 'v1':
            cur = conn.execute('SELECT trade_date, amv_regime FROM market_amv ORDER BY trade_date')
        elif version == 'v2':
            tbl = table or 'market_regime_signals'
            cur = conn.execute(
                f'SELECT trade_date, regime_v2 FROM {tbl} '
                f'WHERE regime_v2 IS NOT NULL ORDER BY trade_date'
            )
        else:
            raise ValueError(f'unknown regime version: {version!r}')
        return {date_str: int(r) for date_str, r in cur.fetchall()}
    finally:
        conn.close()
```

Find `run_comparison(...)` and add `regime_table=None` parameter, forward to `load_regime`.

In argparse main, add:
```python
parser.add_argument('--regime-table', default=None,
    help='Custom v2 table name (e.g. market_regime_signals_unanimous). Default: market_regime_signals')
```

And forward to `run_comparison(regime_table=args.regime_table)`.

- [ ] **Step 3: Smoke-test custom table loads**

```bash
python3 -c "
import sys
sys.path.insert(0, 'backtest')
from regime_switch_backtest import load_regime
r_def = load_regime(version='v2')
r_un = load_regime(version='v2', table='market_regime_signals_unanimous')
print(f'default v2 bull: {sum(1 for v in r_def.values() if v==1)}')
print(f'unanimous v2 bull: {sum(1 for v in r_un.values() if v==1)}')
assert sum(1 for v in r_un.values() if v==1) < sum(1 for v in r_def.values() if v==1), \\
    'unanimous should call fewer bull days'
print('OK')
"
```

Expected: unanimous bull count < default. Specific numbers depend on Task C3 results.

- [ ] **Step 4: Commit**

```bash
git add backtest/regime_switch_backtest.py
git commit -m "feat(ng2.0a-v2): regime_switch_backtest --regime-table for variant tables"
```

---

## Task C5: Build Phase C sweep script

**Files:**
- Create: `scripts/ng2_0a_calibration_sweep.py`

This script automates: for each variant, run merge_v2 + Step B north-star eval (WF-OOS + Pre-2020), extract metrics, build comparison table.

- [ ] **Step 1: Write the sweep script**

Create `scripts/ng2_0a_calibration_sweep.py`:

```python
"""Phase C calibration sweep: run regime-switch backtest + north-star eval for each B1/B2 variant.

Outputs a markdown comparison table to reports/ng2_0a_calibration_sweep_results.md.

Sub-models used (per user decision 2026-04-25):
  bull = ng1.0.1 (reports/daily_selection_ng101 for WF-OOS, _pre2020 for Pre-2020)
  bear = ng1.0.4-3s (reports/daily_selection_ng104_ensemble_3seed for WF-OOS, _pre2020 for Pre-2020)
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

VARIANTS = [
    'baseline',     # V0: B1=(0.45,0.55), B2=(0.30,0.70), streak=3, vote=2
    'strict_b1',    # V1: B1=(0.40,0.65)
    'strict_b2',    # V2: B2=(0.25,0.75)
    'streak5',      # V3: system_streak=5
    'unanimous',    # V4: vote_threshold=3
]

REPO = Path(__file__).resolve().parents[1]


def run(cmd: list, log_path: Path) -> str:
    """Run shell command, tee to log, return stdout."""
    print(f'  > {" ".join(cmd)}')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    full = res.stdout + res.stderr
    log_path.write_text(full)
    if res.returncode != 0:
        print(f'  ! exited {res.returncode}, see {log_path}')
    return full


def extract_metrics(log_text: str) -> dict:
    """Extract V5.2 final score, 10d Sharpe/MaxDD/annual from north-star eval log."""
    m = {}
    # Last 加权评分 line in the log (V5.2 295-point card, the gate metric)
    matches = re.findall(r'加权评分:\s*([\d.]+)%\s*(?:×\s*[\d.]+\s*=\s*([\d.]+)%)?\s*→\s*等级\s*(\S+)', log_text)
    if matches:
        # Take the last one (V5.2 final, after preceding V4/V5/V5.1 cards)
        last = matches[-1]
        # If discount applied (Pre-2020), use post-discount score
        m['v52_score'] = float(last[1]) if last[1] else float(last[0])
        m['v52_grade'] = last[2]
    # 10日持仓 metrics — find the section then extract
    sec = re.search(r'\(10日持仓\)(.+?)(?=北极星评分卡|$)', log_text, re.DOTALL)
    if sec:
        block = sec.group(1)
        for key, label in [
            ('sharpe', r'Sharpe:\s*([-\d.]+)'),
            ('maxdd', r'最大回撤:\s*([-\d.]+)%'),
            ('annual_gross', r'年化收益\(毛\):\s*([-\d.]+)%'),
            ('annual_net', r'年化收益\(净\):\s*([-\d.]+)%'),
            ('icir', r'ICIR.*?([-\d.]+)'),
        ]:
            mm = re.search(label, block)
            if mm:
                m[key] = float(mm.group(1))
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variants', nargs='+', default=VARIANTS,
                   help='Subset of variants to test')
    p.add_argument('--skip-merge', action='store_true',
                   help='Skip regime_switch merge step (assume merged dirs already exist)')
    args = p.parse_args()

    bull_wfo = 'reports/daily_selection_ng101'
    bear_wfo = 'reports/daily_selection_ng104_ensemble_3seed'
    bull_pre = 'reports/daily_selection_ng101_pre2020'
    bear_pre = 'reports/daily_selection_ng104_pre2020'

    rows = []  # (variant, window, metrics_dict)

    for variant in args.variants:
        table = f'market_regime_signals_{variant}'
        for window, start, end, bd, brd in [
            ('WF-OOS', '2020-01-01', '2026-04-25', bull_wfo, bear_wfo),
            ('Pre-2020', '2018-01-01', '2019-12-31', bull_pre, bear_pre),
        ]:
            tag = f'{variant}_{window.lower().replace("-", "_")}'
            out_dir = f'reports/daily_selection_regime_switch_{tag}'

            if not args.skip_merge:
                print(f'\n[Merge] variant={variant} window={window}')
                run([
                    'python3', 'backtest/regime_switch_backtest.py',
                    '--regime-version', 'v2',
                    '--regime-table', table,
                    '--bull-dir', bd,
                    '--bear-dir', brd,
                    '--out-dir', out_dir,
                ], Path(f'logs/ng2_0a_phase_c_merge_{tag}.log'))

            print(f'\n[Eval] variant={variant} window={window}')
            log = Path(f'logs/ng2_0a_phase_c_eval_{tag}.log')
            run([
                'python3', 'backtest/run_north_star_eval.py', '--backtest',
                '--report-dir', out_dir,
                '--label', f'ng2.0a-v2-{variant}-{window}',
                '--top-n', '10', '--focus-days', '10', '--rank-field', 'composite',
                '--start-date', start, '--end-date', end,
            ], log)
            metrics = extract_metrics(log.read_text())
            rows.append((variant, window, metrics))
            print(f'  -> V5.2={metrics.get("v52_score", "N/A")}%, '
                  f'Sharpe={metrics.get("sharpe", "N/A")}, '
                  f'MaxDD={metrics.get("maxdd", "N/A")}%')

    # Write markdown report
    out = Path('reports/ng2_0a_calibration_sweep_results.md')
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ['# ng2.0a Calibration Sweep Results\n']
    lines.append('Sub-models: bull=ng1.0.1, bear=ng1.0.4-3s (per Step B Option C decision)\n')
    lines.append('## WF-OOS 2020-2026\n')
    lines.append('| Variant | V5.2 | Grade | Sharpe | MaxDD | Annual gross | Annual net | ICIR |')
    lines.append('|---|---:|---|---:|---:|---:|---:|---:|')
    for v, w, m in rows:
        if w != 'WF-OOS': continue
        lines.append(f'| {v} | {m.get("v52_score", "?")}% | {m.get("v52_grade", "?")} | '
                     f'{m.get("sharpe", "?")} | {m.get("maxdd", "?")}% | '
                     f'{m.get("annual_gross", "?")}% | {m.get("annual_net", "?")}% | '
                     f'{m.get("icir", "?")} |')
    lines.append('\n## Pre-2020 2018-2019\n')
    lines.append('| Variant | V5.2 (×0.85) | Grade | Sharpe | MaxDD | Annual gross |')
    lines.append('|---|---:|---|---:|---:|---:|')
    for v, w, m in rows:
        if w != 'Pre-2020': continue
        lines.append(f'| {v} | {m.get("v52_score", "?")}% | {m.get("v52_grade", "?")} | '
                     f'{m.get("sharpe", "?")} | {m.get("maxdd", "?")}% | '
                     f'{m.get("annual_gross", "?")}% |')

    # Acceptance gate evaluation (per plan section)
    lines.append('\n## Gate evaluation (Phase C acceptance)\n')
    lines.append('Acceptance: WF-OOS V5.2 ≥ 78% AND Pre-2020 V5.2 ≥ 38% AND Pre-2020 net annual ≥ -8%\n')
    lines.append('| Variant | WF-OOS V5.2 ≥78 | Pre-2020 V5.2 ≥38 | Pre-2020 annual_net ≥-8 | Verdict |')
    lines.append('|---|---|---|---|---|')
    for v in args.variants:
        wfo = next((m for vv, w, m in rows if vv == v and w == 'WF-OOS'), {})
        pre = next((m for vv, w, m in rows if vv == v and w == 'Pre-2020'), {})
        g1 = wfo.get('v52_score', 0) >= 78
        g2 = pre.get('v52_score', 0) >= 38
        g3 = pre.get('annual_net', -100) >= -8
        verdict = 'PASS' if (g1 and g2 and g3) else 'FAIL'
        lines.append(f'| {v} | {"OK" if g1 else "X"} {wfo.get("v52_score", "?")}% | '
                     f'{"OK" if g2 else "X"} {pre.get("v52_score", "?")}% | '
                     f'{"OK" if g3 else "X"} {pre.get("annual_net", "?")}% | **{verdict}** |')

    out.write_text('\n'.join(lines) + '\n')
    print(f'\nReport: {out}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run the sweep**

```bash
python3 scripts/ng2_0a_calibration_sweep.py 2>&1 | tee logs/ng2_0a_phase_c_sweep.log
```

This runs 5 variants × 2 windows = 10 merge + 10 eval invocations. Expected duration: ~15-25 minutes.

- [ ] **Step 3: Read the comparison table**

```bash
cat reports/ng2_0a_calibration_sweep_results.md
```

Expected: 2 markdown tables (WF-OOS + Pre-2020) + a gate evaluation table.

- [ ] **Step 4: Commit**

```bash
git add scripts/ng2_0a_calibration_sweep.py
git commit -m "feat(ng2.0a-v2): Phase C calibration sweep — 4 variant comparison"
```

---

## Task C6: GATE — Phase C decision

**This is a user-review GATE.** Stop here, present the sweep results, get the user's go/no-go on which variant proceeds to Phase A.

- [ ] **Step 1: Surface the sweep results to the user**

Read `reports/ng2_0a_calibration_sweep_results.md` and present a tight summary in the chat:
- Which variants PASS the acceptance gate?
- Which is the "best": highest WF-OOS V5.2 + Pre-2020 V5.2 ≥ 38% + Pre-2020 annual_net ≥ -8%
- If multiple PASS: recommend the one closest to baseline behavior (least change to working production candidate)
- If NONE pass: recommend default `baseline` variant proceed to Phase A as-is, document Phase C as "no improvement found"

- [ ] **Step 2: Wait for user decision**

User chooses ONE variant (could be `baseline` if none improved). Record the choice as `PHASE_C_WINNING_VARIANT` in subsequent commits and Phase A wiring.

- [ ] **Step 3: Commit decision**

```bash
git add reports/ng2_0a_calibration_sweep_results.md
git commit -m "feat(ng2.0a-v2): Phase C complete — winning variant: <VARIANT>"
```

(Replace `<VARIANT>` with user-chosen variant name.)

---

# PHASE A: Production Wiring of ng2.0a Scoring Version

**Goal:** 把 v2 regime + ng1.0.1 bull + ng1.0.4-3s bear (with Phase C 选定的 calibration variant) 接进 `tomorrow_stock_selector.py` 作为 `--scoring-version ng2.0a`. 灰度对比生产 ng106v2 至少 1 天 reports.

**Acceptance criteria:**
- `python3 tomorrow_stock_selector.py 2026-04-24 --scoring-version ng2.0a` runs end-to-end without error
- Output report has correct structure (`all_stocks_with_scores` populated, `pred_10d` has non-zero values per CLAUDE.md OOS check rule)
- Side-by-side Top-10 overlap with ng1.0.62 production ≥ 70%
- Documentation updated: CLAUDE.md, wiki, MEMORY

**Note:** `<WINNING_VARIANT>` placeholder below = the variant chosen in Task C6. If C6 found NO winning variant, use `baseline`.

---

## Task A1: Inspect existing ng106 dispatch pattern

**Files:**
- Read: `tomorrow_stock_selector.py` (around lines 5775-6062 for ng106 mode)
- Read: `ml_models/ng/ng_production_scorer.py` (or whichever NG scorer file is invoked)

- [ ] **Step 1: Find ng106 dispatch entry point**

```bash
grep -n "ng106_mode\|ng1.0.6\|ng1.0.62" tomorrow_stock_selector.py | head -30
```

Expected: shows `if scoring_version in ("ng1.0.6", "ng1.0.62")` block + `_ng106_mode` flag wiring + report dir routing.

- [ ] **Step 2: Find what classes/functions ng106 mode actually invokes**

```bash
grep -n "amv_regime\|market_amv\|regime_switch\|_ng106" tomorrow_stock_selector.py | head -40
```

Expected: identifies the load_regime / merge call site for ng106. Read carefully (~50 lines around the dispatch) to understand:
1. How does ng106 load the regime per-day?
2. How does it call bull-model scorer vs bear-model scorer?
3. How does it merge into a single Top-10 for the day?
4. Where does the report dir get written?

- [ ] **Step 3: Document findings inline**

Write a 5-line note in the chat (or scratch file) mapping:
- Bull model: `ng1.0.7` → loaded via `__??__` from `__??__`
- Bear model: `ng1.0.4-3s` → loaded via `__??__` from `__??__`
- Regime: read from `market_amv.amv_regime` for current trade_date
- Merge: pick bull Top-10 if regime=1, bear Top-10 if regime=-1
- Report: `reports/daily_selection_ng106v2/`

This grounds the next task. Do NOT skip — guessing the structure leads to hours of debugging later.

---

## Task A2: Add `ng2.0a` scoring-version dispatch

**Files:**
- Modify: `tomorrow_stock_selector.py`

- [ ] **Step 1: Add ng2.0a to the valid scoring-versions list**

Edit `tomorrow_stock_selector.py` line 6204 (the long argparse `choices=[...]` list). Add `'ng2.0a'` to the list.

- [ ] **Step 2: Add ng2.0a to the ng106-style mode wiring**

In the ng106 dispatch block (around line 5775):

```python
# CURRENT:
ng106_mode = False
if scoring_version in ("ng1.0.6", "ng1.0.62"):
    ng106_mode = True
    bull_model = "ng1.0.7" if scoring_version == "ng1.0.62" else "ng1.0.1"
    ...
```

Extend to handle ng2.0a:

```python
ng106_mode = False
ng200a_mode = False
if scoring_version in ("ng1.0.6", "ng1.0.62"):
    ng106_mode = True
    bull_model = "ng1.0.7" if scoring_version == "ng1.0.62" else "ng1.0.1"
    bear_model = "ng1.0.4-3s"
elif scoring_version == "ng2.0a":
    ng200a_mode = True
    bull_model = "ng1.0.1"          # Phase A default per Step B Option C result
    bear_model = "ng1.0.4-3s"
    regime_table = "market_regime_signals_<WINNING_VARIANT>"  # Phase C decision
```

(Replace `<WINNING_VARIANT>` with Phase C result. If C found nothing, use `market_regime_signals` (default V0 baseline) since Task C2 smoke verified `_baseline` table matches the main table.)

- [ ] **Step 3: Wire ng200a_mode through selector init**

Find the line ~5829-5831:
```python
if ng106_mode:
    selector._ng106_mode = True
    selector._ng106_tag = version_tag
```

Add:
```python
if ng200a_mode:
    selector._ng200a_mode = True
    selector._ng200a_bull_model = bull_model
    selector._ng200a_bear_model = bear_model
    selector._ng200a_regime_table = regime_table
```

- [ ] **Step 4: Wire report dir for ng2.0a**

Find the report_dir block around line 6057-6062:
```python
if ng106_mode:
    report_dir = Path("reports/daily_selection_ng106v2"
                      if version_tag == 'ng1.0.62'
                      else "reports/daily_selection_ng106")
```

Add elif:
```python
elif ng200a_mode:
    report_dir = Path("reports/daily_selection_ng2_0a")
```

- [ ] **Step 5: Implement the actual ng200a scoring path**

This is the substantive change. Identify where ng106_mode actually computes the Top-10 list per-day (search for `_ng106_mode` usage in the rest of the file). The pattern likely:

1. Loads bull-model scorer
2. Loads bear-model scorer
3. For target trade_date, looks up `regime` from market_amv
4. If regime=1: use bull scorer's Top-N; else: use bear scorer's Top-N

For ng200a_mode, do the SAME but read regime from `market_regime_signals_<WINNING_VARIANT>` instead of `market_amv`. Specifically:

```python
# Wherever ng106_mode reads regime:
if getattr(self, '_ng200a_mode', False):
    import sqlite3
    db = sqlite3.connect('data_adapter/stock_data.db', timeout=30)
    db.execute('PRAGMA busy_timeout=30000')
    table = self._ng200a_regime_table
    row = db.execute(
        f'SELECT regime_v2 FROM {table} WHERE trade_date = ?', (target_date,)
    ).fetchone()
    db.close()
    if row is None or row[0] is None:
        regime = -1  # default bear if no regime data (pre-warmup)
    else:
        regime = int(row[0])
elif getattr(self, '_ng106_mode', False):
    # existing ng106 regime load
    ...
```

(Exact integration depends on Task A1 findings. The intent is "swap the regime source from `market_amv` to `market_regime_signals_<variant>` while keeping the bull/bear scorer dispatch logic identical to ng106v1 = ng1.0.1 bull / ng1.0.4-3s bear.")

- [ ] **Step 6: Update titling logic**

Find the `_ng106_mode` title override around line 5060:
```python
if getattr(self, '_ng106_mode', False):
    # ng1.0.6 覆盖标题
```

Add:
```python
if getattr(self, '_ng200a_mode', False):
    # ng2.0a multi-beta vote regime
    title = f"NG2.0a 选股 (v2 regime + ng1.0.1 bull + ng1.0.4-3s bear) — {target_date}"
```

- [ ] **Step 7: Commit**

```bash
git add tomorrow_stock_selector.py
git commit -m "feat(ng2.0a): add --scoring-version ng2.0a dispatch (v2 regime + ng101 bull + ng104-3s bear)"
```

---

## Task A3: Smoke-test end-to-end

**Files:**
- Read: `reports/daily_selection_ng2_0a/analysis_data_20260424.json` (after run)

- [ ] **Step 1: Run for one trading day**

```bash
python3 tomorrow_stock_selector.py 2026-04-24 --scoring-version ng2.0a 2>&1 | tee logs/ng2_0a_phase_a_smoke.log
```

Expected: completes in under 5 minutes, prints Top-10 list, writes to `reports/daily_selection_ng2_0a/analysis_data_20260424.json`.

- [ ] **Step 2: Verify report structure**

```bash
python3 -c "
import json
data = json.load(open('reports/daily_selection_ng2_0a/analysis_data_20260424.json'))
print('Keys:', list(data.keys())[:8])
stocks = data.get('all_stocks_with_scores', [])
print(f'Total scored: {len(stocks)}')
nonzero = sum(1 for s in stocks if float(s.get('pred_10d', 0) or 0) != 0)
print(f'pred_10d non-zero: {nonzero}/{len(stocks)}')
assert len(stocks) >= 1000, 'expected 1000+ scored stocks'
assert nonzero >= len(stocks) * 0.5, 'expected >50% with non-zero pred_10d (per CLAUDE.md OOS rule)'
print('OK')
print('Top-10:')
for s in stocks[:10]:
    print(f'  {s.get(\"stock_code\")}: score={s.get(\"score\"):.2f}, pred_10d={s.get(\"pred_10d\")}')
"
```

Expected: 1000+ stocks, ≥50% non-zero pred_10d, Top-10 prints with non-trivial scores.

- [ ] **Step 3: Verify regime was correctly read from variant table**

```bash
python3 -c "
import sqlite3
c = sqlite3.connect('data_adapter/stock_data.db', timeout=30)
# Use the WINNING_VARIANT chosen in Phase C
table = 'market_regime_signals_<WINNING_VARIANT>'
r = c.execute(f'SELECT regime_v2 FROM {table} WHERE trade_date = ?', ('2026-04-24',)).fetchone()
print(f'regime_v2 on 2026-04-24 (variant={table}): {r[0] if r else None}')
"
```

(If the smoke run produced sensible output, this regime value correctly drove bull-vs-bear sub-model routing.)

- [ ] **Step 4: Commit logs (if not gitignored)**

```bash
git add logs/ng2_0a_phase_a_smoke.log 2>/dev/null  # may be gitignored
git diff --cached
# If diff is empty, skip the commit; otherwise:
git commit -m "chore(ng2.0a): Phase A smoke-test log for 2026-04-24" || echo "no changes (gitignored)"
```

---

## Task A4: Side-by-side Top-10 vs ng1.0.62 production

- [ ] **Step 1: Run ng1.0.62 production on the same day**

```bash
python3 tomorrow_stock_selector.py 2026-04-24 --scoring-version ng1.0.62 2>&1 | tail -30
```

(Should be fast — likely already cached. Output writes to `reports/daily_selection_ng106v2/`.)

- [ ] **Step 2: Compare Top-10 overlap**

```bash
python3 -c "
import json
def top10(d):
    p = f'reports/daily_selection_{d}/analysis_data_20260424.json'
    data = json.load(open(p))
    return [s.get('stock_code') for s in data.get('all_stocks_with_scores', [])[:10]]

t1 = top10('ng106v2')
t2 = top10('ng2_0a')
overlap = len(set(t1) & set(t2))
print(f'ng106v2 Top10: {t1}')
print(f'ng2.0a  Top10: {t2}')
print(f'Overlap: {overlap}/10 = {overlap*10}%')
assert overlap >= 7, f'expected ≥70% overlap, got {overlap*10}%'
print('OK — overlap ≥ 70%, ng2.0a is a sensible variant of production not a wild departure')
"
```

Expected: ≥7/10 overlap. If less, investigate (likely a bug in ng2.0a wiring) before proceeding.

- [ ] **Step 3: Commit**

```bash
git add scripts/ng2_0a_top10_overlap.py 2>/dev/null
git commit -m "test(ng2.0a): Phase A side-by-side overlap with ng1.0.62 production" 2>/dev/null || echo "no new files"
```

---

## Task A5: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` (ML Scoring Systems 活跃版本 section)

- [ ] **Step 1: Find the existing ML Scoring Systems entry**

```bash
grep -n "ML Scoring Systems\|NG v1.0.6\|ng1.0.6\|ng106" CLAUDE.md | head -10
```

Expected: section "## ML Scoring Systems (活跃版本)" with numbered list (1, 2, 3...).

- [ ] **Step 2: Insert ng2.0a entry**

Add a new entry at the top of the activist list, BEFORE the existing `🎯 NG v1.0.6 (0AMV 牛熊切换)` entry. Use this exact text (replace placeholders with actual numbers from Phase C/A results):

```markdown
1. **🆕 NG v2.0a (multi-beta regime + ng106v1 sub-model)** (2026-04-26, 灰度评估中):
   - 核心: V11 (0AMV) + B1 (% 股票上方 MA20/MA60) + B2 (沪深300 60d RV percentile) hard vote (2-of-3) + 3d streak
   - Sub-model: bull → ng1.0.1 (= ng106v1 bull, ng1.0.7 因 Pre-2020 弱被换下), bear → ng1.0.4-3s
   - 性能 (vs ng1.0.62 生产 = ng106v2):
     - WF-OOS V5.2 = XX.X% (vs 80.4% S 生产)
     - **WF-OOS MaxDD = -XX.X% (vs -23.7% 生产, 改善 ~6pp)**
     - WF-OOS Sharpe = X.XX (vs 2.39 生产)
     - Pre-2020 V5.2 = XX.X% C, 净年化 XX.X%
   - Calibration: variant `<WINNING_VARIANT>` (Phase C 选定)
   - 选股: `python3 tomorrow_stock_selector.py YYYY-MM-DD --scoring-version ng2.0a`
   - 报告: `reports/daily_selection_ng2_0a/`
   - regime 表: `market_regime_signals_<WINNING_VARIANT>`
   - 后续: ng2.0b sample-weighted sub-model retrain (Phase B 待 plan)
```

(Fill XX.X% with actuals from Task A3/A4 + Phase C eval.)

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(ng2.0a): add ng2.0a to CLAUDE.md ML Scoring Systems section"
```

---

## Task A6: Wiki + Memory updates

**Files:**
- Create: `docs/wiki/architecture/ng2_0a_multi_beta_regime.md`
- Create: `~/.claude/projects/-Users-yangxu-StockTradebyZ/memory/ng2_0a_production.md`
- Modify: `~/.claude/projects/-Users-yangxu-StockTradebyZ/memory/MEMORY.md`

- [ ] **Step 1: Create wiki page**

Create `docs/wiki/architecture/ng2_0a_multi_beta_regime.md`:

```markdown
# NG v2.0a Multi-Beta Regime Classifier

**Status:** 灰度对比生产 ng106v2 (2026-04-26 上线候选)

## 核心架构

3 个独立信号 hard vote (2-of-3 多数, 系统级 3d streak) → 路由 ng1.0.1 (bull) 或 ng1.0.4-3s (bear).

| 信号 | 计算 | hysteresis | streak |
|---|---|---|---|
| V11 | 0AMV 位置 + MACD (existing regime_classifier.py preset) | — | 3d |
| B1 | % stocks closing above MA20/MA60 (panel) | (0.45, 0.55) | 3d |
| B2 | 沪深300 60d realized vol → 252d percentile | (0.30, 0.70) inverted | 3d |

vote_threshold=2 (多数), system_streak=3 (default), Phase C 可调。

## 性能 vs 生产 ng106v2

(填入 Phase A 实测数字)

## 落地

- selector: `--scoring-version ng2.0a`
- 报告: `reports/daily_selection_ng2_0a/`
- regime 数据: `market_regime_signals[_<variant>]` table
- 模块: `indicators/breadth.py`, `indicators/realized_vol.py`, `indicators/regime_classifier.py:compute_regime_v2`

## 关键决策日志

- 2026-04-25 Step A PASS (84% agreement vs V11 baseline, 0.87x flips)
- 2026-04-25 Step B 原版 (ng1.0.7 bull) 实质平局 (V5.2 -0.2pp)
- 2026-04-25 Step B Option C (ng1.0.1 bull) v2 +2.8pp V5.2 + 9pp MaxDD 改善 → 决策切换 sub-model
- 2026-04-26 Phase C calibration sweep, 选 variant <WINNING_VARIANT>
- 2026-04-26 Phase A 接生产, 灰度起跑

## 已知限制

- B1/B2 在长期熊市末段 (2018Q4 → 2019Q1) 倾向 over-call bull (Pre-2020 +33 多余 bull 天 vs V11). Phase C 部分缓解, ng2.0b sub-model 重训进一步修。
- regime 数据需要 daily 更新: `python3 scripts/build_regime_v2_history_variant.py --variant <WINNING_VARIANT> --start <yesterday> --end <today>` 应加入 daily_update workflow。

## 关联

- spec: `docs/superpowers/specs/2026-04-25-ng200-regime-refined-architecture-design.md`
- plan: `docs/superpowers/plans/2026-04-25-ng200a-multi-beta-regime.md` (ng2.0a 主线), `docs/superpowers/plans/2026-04-26-ng2_0a-followup-CAB.md` (后续)
- 上游: `regime_classifier_v1` (V11 baseline)
- 下游: `ng2_0b` (sub-model 重训, 待 plan)
```

- [ ] **Step 2: Create memory detail file**

Create `/Users/yangxu/.claude/projects/-Users-yangxu-StockTradebyZ/memory/ng2_0a_production.md`:

```markdown
---
name: ng2.0a 上线 (灰度)
description: ng2.0a multi-beta regime + ng1.0.1 bull / ng1.0.4-3s bear 配置接进 selector, 灰度对比 ng106v2
type: project
---

## 配置
- scoring-version: `ng2.0a`
- regime: `market_regime_signals_<WINNING_VARIANT>` (Phase C calibration)
- bull: ng1.0.1 (Step B Option C 验证比 ng1.0.7 好搭配 v2 regime)
- bear: ng1.0.4-3s
- 报告: `reports/daily_selection_ng2_0a/`

## 性能
- WF-OOS V5.2 = XX.X% (vs 80.4% 生产 ng106v2)
- WF-OOS MaxDD = -XX.X% (vs -23.7% 生产, 改善 ~6pp 是用户想要的)
- WF-OOS Sharpe = X.XX (vs 2.39 生产)
- Pre-2020 净年化 XX.X% (vs -10.8% 生产)

## Why
- 用户偏好低回撤 (用户原话 "回撤控制上的提升是很好的, 我个人会更喜欢这个模型" 2026-04-25)
- ng106v2 在 WF-OOS V5.2 上微胜 (~1pp), 但 MaxDD 差很多, ng2.0a 是 risk-adjusted 更好

## How to apply
- 灰度阶段: 同时跑 ng1.0.62 + ng2.0a, 对比每日 Top-10 overlap (目标 ≥70%) + 1-4 周累计 forward return
- 切换决定权: 用户. infrastructure 已就绪, 不切换也不影响 ng106v2 生产
- 后续 ng2.0b 重训如果通过 (V5.2 ≥ 81% + MaxDD ≤ -18%), 再考虑切换 ng2.0a → ng2.0b
```

- [ ] **Step 3: Add MEMORY.md index entry**

Append to `/Users/yangxu/.claude/projects/-Users-yangxu-StockTradebyZ/memory/MEMORY.md` at the top of the existing list:

```markdown
## 🆕 ng2.0a 灰度上线 (2026-04-26) → 详见 `ng2_0a_production.md`
- v2 regime (V11+B1+B2 hard vote, calibration variant `<WINNING_VARIANT>`) + ng1.0.1 bull + ng1.0.4-3s bear
- WF-OOS V5.2=XX% (vs 80.4 生产), MaxDD=-XX% (vs -23.7 生产, 改善 ~6pp). Phase C 选 `<WINNING_VARIANT>` 修 Pre-2020 over-call
- 选股: `--scoring-version ng2.0a`. ng2.0b 重训 plan 待写
```

- [ ] **Step 4: Commit**

```bash
git add docs/wiki/architecture/ng2_0a_multi_beta_regime.md
git commit -m "docs(ng2.0a): wiki + memory for production candidate ng2.0a"
```

(Memory files in `~/.claude/...` are not in repo; the index update happens at runtime by Claude itself.)

---

## Task A7: Update daily_update workflow to refresh regime table

**Files:**
- Modify: `fetch_data/quick_daily_update.py` OR `run_daily_update.sh` (whichever orchestrates daily updates)

The new `market_regime_signals_<WINNING_VARIANT>` table must be refreshed daily, otherwise the next trading day's selector will use stale regime.

- [ ] **Step 1: Find the daily update orchestrator**

```bash
grep -rn "build_regime_v2\|market_regime_signals\|indicators/market_amv" run_daily_update.sh fetch_data/quick_daily_update.py 2>/dev/null | head
```

- [ ] **Step 2: Add backfill call after market_amv refresh**

After whichever step refreshes `market_amv`, add (one shell line OR one Python subprocess call):

```bash
python3 scripts/build_regime_v2_history_variant.py \
    --variant <WINNING_VARIANT> \
    --start $(date -v-7d +%Y-%m-%d) \
    --end $(date +%Y-%m-%d)
```

(Backfill 7 days to ensure idempotent overlap; the script DELETEs then INSERTs in range.)

If `quick_daily_update.py` is the orchestrator, add as a subprocess call near the end (after market_amv update, before stock-selection trigger).

- [ ] **Step 3: Smoke-test daily refresh**

```bash
python3 scripts/build_regime_v2_history_variant.py --variant <WINNING_VARIANT> --start 2026-04-20 --end 2026-04-25
```

Verify the recent days are present:
```bash
python3 -c "
import sqlite3
c = sqlite3.connect('data_adapter/stock_data.db', timeout=30)
n = c.execute(\"SELECT COUNT(*) FROM market_regime_signals_<WINNING_VARIANT> WHERE trade_date >= '2026-04-20'\").fetchone()[0]
print(f'recent days: {n}')
assert n >= 4
print('OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add run_daily_update.sh fetch_data/quick_daily_update.py 2>/dev/null
git commit -m "feat(ng2.0a): add daily refresh of market_regime_signals_<WINNING_VARIANT> to daily workflow"
```

---

# PHASE B: ng2.0b Sample-Weighted Sub-Model Retrain

**Goal:** 重训 bull sub-model (`ng2.0b-bull`) 用 bull-regime sample 加权 ×2, 重训 bear sub-model (`ng2.0b-bear`) 用 bear-regime sample 加权 ×2. 用 Phase A 选定的 v2 regime 路由这两个新 sub-model. 目标 V5.2 ≥ 81% + MaxDD ≤ -18% + Pre-2020 净年化 ≥ -5%.

**Acceptance criteria (per spec):**
- WF-OOS V5.2 ≥ 81% (vs ng2.0a 的 79.3% baseline + 1.7pp)
- WF-OOS MaxDD ≤ -18% (vs ng2.0a 的 -17.6%, ≥ 现有水平)
- Pre-2020 net annual ≥ -5% (vs ng2.0a 的 -10.3%, +5pp 改善)
- WF-OOS Sharpe ≥ 2.5
- 不引入新数据泄露 (β_UMD < 1.0)

**Estimated total duration:** 6-12 hours (training is dominant)

**This phase has its own pre-flight checklist per CLAUDE.md "Model Iteration Pre-flight Checklist (10 项)".**

---

## Task B1: Pre-flight checklist (10 items per CLAUDE.md)

**This task does NOT write code — it produces a checklist report.** Run before any training.

- [ ] **Step 1: Run all 10 pre-flight checks**

For each of the 10 items in CLAUDE.md `## 🛑 模型迭代 Pre-flight Checklist` (Schema, Backfill, Efficiency, Acceptance, Baseline, Checkpointing, 泄露 Pre-scan, 资源, 元数据, simplify), produce a one-line answer.

Critical for ng2.0b specifically:
- **Check 1 Schema**: ng2.0b uses ng1.0.1 schema (66 features). No schema change needed → skip-applicable.
- **Check 2 Backfill**: ng1.0.1 feature cache (`ng101_feature_cache`) already populated. No backfill needed.
- **Check 3 Efficiency**: trainer is `ml_models/training/train_v395_multi_target.py`. Use `--target-parallel 4` per memory `training_target_parallel.md`. Auto-WF on by default.
- **Check 4 Acceptance**: WF-OOS V5.2 ≥ 81% / MaxDD ≤ -18% / Pre-2020 net annual ≥ -5%. ABORT if first WF window 10d ICIR < 0.4.
- **Check 5 Baseline**: compare against ng1.0.1 (current bull sub-model in ng106v1) and ng1.0.4-3s (current bear sub-model). Same WF mode (expanding), purge=15, seeds (42 and one of 123/456 for variance check).
- **Check 6 Checkpointing**: `caffeinate -i python3 ...` + `tee logs/train_ng2_0b_*.log`. Save WF window checkpoints.
- **Check 7 泄露**: ng1.0.1 schema is clean (β_UMD=+0.38 t=5.4 per memory). Sample weights don't introduce features. Should be safe.
- **Check 8 资源**: `df -h` ≥ 20GB free, no concurrent training (`ps aux | grep train_v395`).
- **Check 9 元数据**: pickle includes git_commit_hash, schema_version=ng1.0.1, sample_weight_mode={bull,bear}, seed, wf_mode, purge_days, training_duration_sec.
- **Check 10 /simplify**: trainer changes (Task B3) pass /simplify.

- [ ] **Step 2: Output the checklist as Claude turn**

User must see all 10 items pass before training kicks off. If any fail, fix or abort.

- [ ] **Step 3: Commit checklist**

```bash
mkdir -p reports/ng2_0b
cat > reports/ng2_0b/preflight_checklist.md << 'EOF'
# ng2.0b Pre-flight Checklist
[paste the 10-item output here]
EOF
git add reports/ng2_0b/preflight_checklist.md
git commit -m "chore(ng2.0b): pre-flight checklist passed before training kickoff"
```

---

## Task B2: Add `--regime-weight` CLI flag to trainer

**Files:**
- Modify: `ml_models/training/train_v395_multi_target.py`

The trainer already has `compute_sample_weights` infrastructure (line 2034). We add an optional regime multiplier on top.

- [ ] **Step 1: Find compute_sample_weights**

```bash
grep -n "compute_sample_weights\|sample_w =" ml_models/training/train_v395_multi_target.py | head
```

Read the function (around line 2034-2074) to understand the current weight scheme (likely time-decay + IC-based).

- [ ] **Step 2: Add a regime-weight multiplier**

Modify `compute_sample_weights` to optionally multiply weights by `2.0` for samples in the chosen regime, `0.5` for the opposite. Add the regime data load + multiplication at the end of the function, gated by `self.regime_weight_mode`.

```python
# At the END of compute_sample_weights, after the existing weight computation:
if getattr(self, 'regime_weight_mode', None) in ('bull', 'bear'):
    import sqlite3
    db = sqlite3.connect('data_adapter/stock_data.db', timeout=30)
    db.execute('PRAGMA busy_timeout=30000')
    # Use the V11 0AMV regime as the labeling source (stable, predates ng2.0a)
    rows = db.execute('SELECT trade_date, amv_regime FROM market_amv').fetchall()
    db.close()
    regime_map = {d: int(r) for d, r in rows}

    target_regime = 1 if self.regime_weight_mode == 'bull' else -1
    multipliers = np.ones(len(df), dtype=np.float64)
    for i, td in enumerate(df['trade_date'].astype(str)):
        r = regime_map.get(td)
        if r == target_regime:
            multipliers[i] = 2.0
        elif r == -target_regime:
            multipliers[i] = 0.5
        # If regime missing, leave at 1.0
    sample_w = sample_w * multipliers
    print(f'  [regime_weight] mode={self.regime_weight_mode}: '
          f'{int((multipliers == 2.0).sum())} ×2, '
          f'{int((multipliers == 0.5).sum())} ×0.5, '
          f'{int((multipliers == 1.0).sum())} ×1.0')
return sample_w
```

(Adjust attribute access pattern to match the trainer class — `self.regime_weight_mode` is set from CLI in Step 3 below.)

- [ ] **Step 3: Add CLI flag**

In the trainer's argparse section:

```python
parser.add_argument('--regime-weight', choices=['none', 'bull', 'bear'], default='none',
    help='Sample weight by regime: bull=×2 bull samples / ×0.5 bear; bear=opposite. '
         'Default: none (no regime weighting).')
```

Wire into the trainer:
```python
trainer.regime_weight_mode = None if args.regime_weight == 'none' else args.regime_weight
```

- [ ] **Step 4: Smoke-test on 1 WF window with --fast-check**

```bash
python3 ml_models/training/train_v395_multi_target.py \
    --version ng1.0.1 \
    --regime-weight bull \
    --fast-check \
    --purge-days 15 \
    --target-parallel 4 \
    2>&1 | tee logs/ng2_0b_smoke_bull.log | tail -40
```

Expected: prints `[regime_weight] mode=bull: NNN ×2, MMM ×0.5, KKK ×1.0` line, completes in ~2 min, prints IC > 0 (sanity).

- [ ] **Step 5: Commit**

```bash
git add ml_models/training/train_v395_multi_target.py
git commit -m "feat(ng2.0b): add --regime-weight {bull,bear,none} CLI flag to trainer"
```

---

## Task B3: Train ng2.0b-bull (full WF, sample weight bull ×2)

- [ ] **Step 1: Caffeinate + train**

```bash
caffeinate -i python3 ml_models/training/train_v395_multi_target.py \
    --version ng1.0.1 \
    --regime-weight bull \
    --purge-days 15 \
    --target-parallel 4 \
    --seed 42 \
    2>&1 | tee logs/train_ng2_0b_bull_$(date +%Y%m%d_%H%M%S).log
```

Expected duration: 30-60 min (auto-WF on, M5 Max).

ABORT line per Pre-flight Check 4: if first WF window 10d ICIR < 0.4, kill the run and investigate.

- [ ] **Step 2: Verify output pickle**

```bash
ls -lah ml_models/trained_models/ng/ | grep -E "ng101.*$(date +%Y%m%d)" | tail
```

Expected: a recent .pkl with size 50-80MB.

Rename for clarity:
```bash
NEW_PKL=$(ls -t ml_models/trained_models/ng/ng101_*.pkl | head -1)
NEW_NAME="ml_models/trained_models/ng/ng2_0b_bull_seed42_$(date +%Y%m%d_%H%M%S).pkl"
mv "$NEW_PKL" "$NEW_NAME"
echo "Renamed to $NEW_NAME"
```

- [ ] **Step 3: Verify metadata**

```bash
python3 -c "
import joblib
m = joblib.load('$NEW_NAME')
print('keys:', list(m.keys()) if isinstance(m, dict) else type(m))
print('regime_weight_mode:', m.get('regime_weight_mode', 'NOT SAVED — ADD TO TRAINER METADATA'))
print('schema_version:', m.get('schema_version', '?'))
print('git_commit:', m.get('git_commit_hash', '?'))
"
```

If `regime_weight_mode` is `NOT SAVED`, return to Task B2 and add it to the pickle metadata dict.

---

## Task B4: Train ng2.0b-bear (full WF, sample weight bear ×2)

- [ ] **Step 1: Train**

```bash
caffeinate -i python3 ml_models/training/train_v395_multi_target.py \
    --version ng1.0.1 \
    --regime-weight bear \
    --purge-days 15 \
    --target-parallel 4 \
    --seed 42 \
    2>&1 | tee logs/train_ng2_0b_bear_$(date +%Y%m%d_%H%M%S).log
```

(Note: we use `--version ng1.0.1` here because ng1.0.1 is the BASE schema. The bear sub-model will be a different model trained on the same schema, just with different sample weights. If the trainer's output naming doesn't distinguish, rename per Task B3 Step 2.)

- [ ] **Step 2: Rename pickle**

```bash
NEW_PKL=$(ls -t ml_models/trained_models/ng/ng101_*.pkl | head -1)
NEW_NAME="ml_models/trained_models/ng/ng2_0b_bear_seed42_$(date +%Y%m%d_%H%M%S).pkl"
mv "$NEW_PKL" "$NEW_NAME"
```

---

## Task B5: Generate report dirs for both new sub-models

The new sub-models need full-history report dirs to feed into regime_switch_backtest.

- [ ] **Step 1: Generate ng2.0b-bull reports**

```bash
python3 backtest/batch_generate_v395_reports.py \
    --version ng2.0b-bull \
    --model-pkl ml_models/trained_models/ng/ng2_0b_bull_seed42_*.pkl \
    --start 2018-01-01 --end 2026-04-25 \
    2>&1 | tee logs/ng2_0b_bull_reports.log
```

(If `batch_generate_v395_reports.py` doesn't accept `--model-pkl` or `--version ng2.0b-bull`, you'll need to extend it OR write a thin wrapper. Look at how it currently routes by `--version` and add an entry for ng2.0b-bull pointing at the new pickle.)

Expected: writes ~1500-2000 reports under `reports/daily_selection_ng2_0b_bull/`.

- [ ] **Step 2: Generate ng2.0b-bear reports**

```bash
python3 backtest/batch_generate_v395_reports.py \
    --version ng2.0b-bear \
    --model-pkl ml_models/trained_models/ng/ng2_0b_bear_seed42_*.pkl \
    --start 2018-01-01 --end 2026-04-25 \
    2>&1 | tee logs/ng2_0b_bear_reports.log
```

- [ ] **Step 3: Verify counts**

```bash
ls reports/daily_selection_ng2_0b_bull | wc -l
ls reports/daily_selection_ng2_0b_bear | wc -l
```

Expected: ≥ 1500 each.

---

## Task B6: Step B+ end-to-end eval (v2 regime + ng2.0b-bull + ng2.0b-bear)

This is the core validation. Use the same regime variant chosen in Phase C.

- [ ] **Step 1: Run regime_switch_backtest with new sub-models**

```bash
# WF-OOS
python3 backtest/regime_switch_backtest.py \
    --regime-version v2 \
    --regime-table market_regime_signals_<WINNING_VARIANT> \
    --bull-dir reports/daily_selection_ng2_0b_bull \
    --bear-dir reports/daily_selection_ng2_0b_bear \
    --out-dir reports/daily_selection_regime_switch_ng2_0b_wfoos \
    2>&1 | tee logs/ng2_0b_step_b_merge_wfoos.log

# Pre-2020 (use the same dirs — they cover full range)
python3 backtest/regime_switch_backtest.py \
    --regime-version v2 \
    --regime-table market_regime_signals_<WINNING_VARIANT> \
    --bull-dir reports/daily_selection_ng2_0b_bull \
    --bear-dir reports/daily_selection_ng2_0b_bear \
    --out-dir reports/daily_selection_regime_switch_ng2_0b_pre2020 \
    2>&1 | tee logs/ng2_0b_step_b_merge_pre2020.log
```

- [ ] **Step 2: North-star evals**

```bash
# WF-OOS
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_regime_switch_ng2_0b_wfoos \
    --label "ng2.0b-WF-OOS" --top-n 10 --focus-days 10 --rank-field composite \
    --start-date 2020-01-01 --end-date 2026-04-25 \
    2>&1 | tee logs/ng2_0b_eval_wfoos.log

# Pre-2020
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_regime_switch_ng2_0b_pre2020 \
    --label "ng2.0b-PRE-2020" --top-n 10 --focus-days 10 --rank-field composite \
    --start-date 2018-01-01 --end-date 2019-12-31 \
    2>&1 | tee logs/ng2_0b_eval_pre2020.log
```

- [ ] **Step 3: Extract V5.2/Sharpe/MaxDD/annual from each log**

Use the same regex pattern as Task C5 (`extract_metrics`).

- [ ] **Step 4: Build comparison table**

Create `reports/ng2_0b/ng2_0b_step_b_results.md`:

```markdown
# ng2.0b Step B Results

## WF-OOS 2020-2026

| Metric | Production ng106v2 | ng2.0a (v2+ng101) | **ng2.0b (v2+ng2.0b-bull/bear)** | Δ vs production |
|---|---:|---:|---:|---:|
| V5.2 | 80.4% S | XX.X% A+ | XX.X% | +/- pp |
| Sharpe (10d) | 2.39 | XX | XX | +/- |
| MaxDD (10d) | -23.7% | -XX.X% | -XX.X% | +/- pp |
| Annual gross | 104.3% | XX.X% | XX.X% | +/- pp |
| ICIR | 0.89 | 0.72 | XX | +/- |

## Pre-2020 2018-2019

| Metric | Production ng106v2 | ng2.0a (v2+ng101) | **ng2.0b** | Δ vs production |
|---|---:|---:|---:|---:|
| V5.2 (×0.85) | 33.4% C | 37.2% C | XX.X% | +/- pp |
| Sharpe (10d) | -0.50 | -0.44 | XX | +/- |
| MaxDD (10d) | -31.7% | -33.9% | -XX.X% | +/- pp |
| Annual net | -17.8% | XX.X% | XX.X% | +/- pp |

## Acceptance gate (per Phase B spec)

| Gate | Target | ng2.0b actual | PASS? |
|---|---|---|---|
| WF-OOS V5.2 ≥ 81% | 81% | XX% | ? |
| WF-OOS MaxDD ≤ -18% | -18% | -XX% | ? |
| WF-OOS Sharpe ≥ 2.5 | 2.5 | XX | ? |
| Pre-2020 net annual ≥ -5% | -5% | XX% | ? |

## Verdict

(Fill PASS / ABORT / PARTIAL based on gates)
```

- [ ] **Step 5: Commit results**

```bash
git add reports/ng2_0b/ng2_0b_step_b_results.md
git commit -m "feat(ng2.0b): Step B+ eval with sample-weighted sub-models — see report"
```

---

## Task B7: GATE — ng2.0b PASS / PARTIAL / ABORT decision

**This is a user-review GATE.** Stop, present results, get decision.

- [ ] **Step 1: Surface results**

Read `reports/ng2_0b/ng2_0b_step_b_results.md` and present in chat:
- All 4 gates: PASS / FAIL each
- Recommendation: PASS = production swap candidate; PARTIAL = keep ng2.0a as production, ng2.0b is shelved finding; ABORT = revert any changes, document why retraining didn't help

- [ ] **Step 2: Wait for user decision**

- [ ] **Step 3: Apply decision**

If PASS:
- Update `tomorrow_stock_selector.py` to add `ng2.0b` scoring-version (similar pattern to ng2.0a Task A2)
- Update CLAUDE.md, wiki, MEMORY (similar to ng2.0a Task A5/A6)
- Commit

If PARTIAL or ABORT:
- Document in MEMORY.md why ng2.0b didn't pass
- Keep ng2.0a as the new production candidate (no further action)
- Commit

```bash
git add CLAUDE.md docs/wiki/ tomorrow_stock_selector.py 2>/dev/null
git commit -m "docs(ng2.0b): <PASS|PARTIAL|ABORT> per Step B+ gate evaluation"
```

---

## Self-review checklist (run after Phase B completion)

- [ ] All Phase C variants tested + winning one chosen + wired in Phase A
- [ ] Phase A: ng2.0a callable via `--scoring-version ng2.0a`, smoke-test passed, ≥70% Top-10 overlap with ng1.0.62
- [ ] Phase A: CLAUDE.md, wiki, MEMORY updated with real numbers (no XX placeholders)
- [ ] Phase A: daily refresh of `market_regime_signals_<variant>` wired into daily workflow
- [ ] Phase B: pre-flight 10-checklist passed BEFORE training
- [ ] Phase B: ng2.0b-bull and ng2.0b-bear trained, pickles include regime_weight_mode metadata
- [ ] Phase B: Step B+ eval ran end-to-end, results table has real numbers
- [ ] Phase B: PASS/PARTIAL/ABORT decision documented + applied
- [ ] All commits use `feat(ng2.0a):` / `feat(ng2.0a-v2):` / `feat(ng2.0b):` / `docs(ng2.0X):` convention
- [ ] All run logs in `logs/ng2_0[ab]_*` (gitignored OK)
- [ ] No git add -A used anywhere — only files mentioned in each task

---

## Notes for the executing session

- The repo's CLAUDE.md mandates `/simplify` after each non-trivial code change. Run it manually (re-read with critical eye) — there's no slash command available to subagents.
- The repo's CLAUDE.md prohibits `git add -A` due to dirty working tree (catboost_info, training artifacts, webapp.db). Always stage explicit files.
- `reports/` and `logs/` are gitignored. Markdown reports (`reports/ng2_0a_*.md`) ARE writable but won't be tracked. The COMMIT MESSAGES are the durable record for Phase outcomes.
- The repo commits ng2.0a-related work directly to `main` (per recent commit history `feat(ng2.0a): ...`). No feature branch needed.
- DB path: `data_adapter/stock_data.db`. Always use `timeout=30` + `PRAGMA busy_timeout=30000` per CLAUDE.md SQLite concurrency rule.
- Use `caffeinate -i` for any training to prevent macOS sleep throttling.

---

## Failure modes and how to recover

**Phase C fails (no variant passes gate):** Document as "calibration sweep no-op" in `reports/ng2_0a_calibration_sweep_results.md` verdict. Skip Phase C wiring; Phase A uses default `market_regime_signals` (= V0 baseline). Continue to Phase A.

**Phase A smoke-test fails (selector errors / Top-10 overlap < 70%):** Likely a scorer wiring bug. Check that ng200a_mode actually loads the bull/bear models (not just sets the flag). The `_ng106_mode` reference implementation is the cleanest pattern to copy — diff your ng200a code against it.

**Phase B trainer crashes:** Most likely the `compute_sample_weights` modification has a column-name bug. Check the trainer's `df` columns (likely `trade_date` is a datetime not a string — adjust the `.astype(str)` call).

**Phase B Step B+ eval shows ng2.0b WORSE than ng2.0a:** This is a possible outcome — sample weighting may overfit bull/bear samples. Document the finding. Ng2.0a stays as the production candidate. Phase B becomes a reject log, not a regression.

**User pulls /clear mid-execution:** This plan is fully self-contained. New session reads this file + CLAUDE.md + Context section above. Pick up at the first unchecked checkbox `- [ ]` in the plan.
