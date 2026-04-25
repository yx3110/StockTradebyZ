# NG v2.0a Implementation Plan: Multi-Beta Regime Classifier

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 升级 regime classifier 到 multi-beta hard-vote (V11 + B1 breadth + B2 realized vol), 用现成 ng106 v2 sub-model 端到端验证, 跑 Step A (regime alone) + Step B (end-to-end) 双闸门.

**Architecture:** 三个独立信号 (V11 0AMV / B1 % stocks above MA / B2 沪深300 60d realized vol percentile) 各自含 3-day streak, 系统层 majority vote (2-of-3) + 再 3-day streak. 所有 regime 计算结果落库 `market_regime_signals` 新表, regime_switch_backtest 读新表路由到 ng1.0.7 (bull) / ng104-3s (bear) 现成 sub-model.

**Tech Stack:** Python 3, SQLite, pandas, numpy, pytest, 现有 indicators/ + ml_models/ng/ + backtest/ 框架

**Spec:** `docs/superpowers/specs/2026-04-25-ng200-regime-refined-architecture-design.md`

**Out of scope (后续 ng2.0b 单独 plan):** sub-model 重训 (sample-weighted bull/bear)

---

## File Structure

**New files:**
- `indicators/breadth.py` — B1 信号 (% stocks above MA20/MA60, advance-decline, 3d streak)
- `indicators/realized_vol.py` — B2 信号 (沪深300 60d RV percentile, hysteresis 30/70, 3d streak)
- `tests/indicators/__init__.py` — pytest 包标识
- `tests/indicators/test_breadth.py` — B1 单测
- `tests/indicators/test_realized_vol.py` — B2 单测
- `tests/indicators/test_regime_v2.py` — vote 合成单测
- `scripts/build_regime_v2_history.py` — 一次性回算 2018-2026 写库
- `scripts/compare_regime_v1_v2.py` — Step A 验证 (flip / 状态分布 / 重合矩阵 / 2018-2019 sanity)

**Modified files:**
- `indicators/regime_classifier.py` — 新增 `compute_regime_v2(v11, b1, b2, system_streak=3)` 函数
- `backtest/regime_switch_backtest.py` — 加 `--regime-version {v1, v2}`, v2 读 `market_regime_signals`
- `data_adapter/database_manager.py` (或新 migration 脚本) — 建表 `market_regime_signals`

---

## Task 1: 创建 B1 市场广度信号模块

**Files:**
- Create: `indicators/breadth.py`
- Create: `tests/indicators/__init__.py` (empty)
- Create: `tests/indicators/test_breadth.py`

- [ ] **Step 1: 写失败测试**

Create `tests/indicators/__init__.py` (empty file).

Create `tests/indicators/test_breadth.py`:

```python
"""B1 breadth signal unit tests."""
import numpy as np
import pandas as pd
import pytest

from indicators.breadth import compute_breadth_signal


def _fake_close_panel(n_dates=80, n_stocks=100, seed=42):
    """Generate synthetic close panel: rows=date, cols=stock_code."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2024-01-01', periods=n_dates, freq='B')
    codes = [f'00000{i}.SZ' for i in range(n_stocks)]
    # Random walk per stock
    rets = rng.normal(0, 0.02, size=(n_dates, n_stocks))
    prices = 10.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=codes)


def test_breadth_returns_dataframe_with_required_cols():
    panel = _fake_close_panel()
    out = compute_breadth_signal(panel, ma_short=20, ma_long=60, streak_days=3)
    assert {'pct_above_ma20', 'pct_above_ma60', 'b1_score', 'b1_bull', 'b1_streak'} <= set(out.columns)
    assert len(out) == len(panel)


def test_breadth_bull_when_majority_above_ma():
    """All stocks trending up → pct_above_ma should rise and b1_bull=1 after streak."""
    n = 80
    dates = pd.date_range('2024-01-01', periods=n, freq='B')
    codes = ['A.SZ', 'B.SZ', 'C.SZ']
    # Strict uptrend in last 30 days
    prices = np.linspace(10, 20, n).reshape(-1, 1).repeat(3, axis=1)
    panel = pd.DataFrame(prices, index=dates, columns=codes)
    out = compute_breadth_signal(panel, ma_short=20, ma_long=60, streak_days=3)
    assert out['b1_bull'].iloc[-1] == 1


def test_breadth_bear_when_majority_below_ma():
    n = 80
    dates = pd.date_range('2024-01-01', periods=n, freq='B')
    codes = ['A.SZ', 'B.SZ', 'C.SZ']
    prices = np.linspace(20, 10, n).reshape(-1, 1).repeat(3, axis=1)
    panel = pd.DataFrame(prices, index=dates, columns=codes)
    out = compute_breadth_signal(panel, ma_short=20, ma_long=60, streak_days=3)
    assert out['b1_bull'].iloc[-1] == 0


def test_breadth_hysteresis_keeps_prev_in_neutral_zone():
    """When score is in 0.45-0.55 hysteresis band, b1_bull stays at previous value."""
    # Build score series oscillating in hysteresis band
    n = 30
    dates = pd.date_range('2024-01-01', periods=n, freq='B')
    codes = ['A.SZ']
    # 50/50 above MA → score ~0.5 (in band) → should sticky
    panel = pd.DataFrame(np.full((n, 1), 10.0), index=dates, columns=codes)
    out = compute_breadth_signal(panel, ma_short=20, ma_long=60, streak_days=3)
    # No information → values should be deterministic (bull or bear, but stable)
    assert out['b1_bull'].iloc[-5:].nunique() == 1


def test_breadth_streak_required_for_flip():
    """Single-day above-band shouldn't flip if previous regime was bear."""
    # Construct synthetic score series that just barely crosses threshold for 1 day
    # by directly testing internal _apply_threshold_with_hysteresis if exposed,
    # or testing via end-to-end with engineered prices is too brittle.
    # Use the exported helper:
    from indicators.breadth import apply_threshold_with_hysteresis_and_streak
    score = pd.Series([0.30, 0.30, 0.30, 0.60, 0.30, 0.30],  # one-day spike
                     index=pd.date_range('2024-01-01', periods=6, freq='B'))
    bull, streak = apply_threshold_with_hysteresis_and_streak(
        score, lo=0.45, hi=0.55, streak_days=3
    )
    # 0.60 spike: 1 day raw_bull=1, streak=1, but persist_n requires 3 → stay bear
    assert bull.iloc[3] == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/indicators/test_breadth.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'indicators.breadth'`.

- [ ] **Step 3: 实现 indicators/breadth.py**

```python
"""B1 市场广度信号 (% stocks above MA + advance-decline + hysteresis + streak).

输入: 全 A 股每日收盘价面板 (DataFrame: index=trade_date, columns=stock_code)
输出: DataFrame with columns:
    pct_above_ma20, pct_above_ma60: 当日 % stocks 收盘 > 自身 MA20 / MA60
    b1_score: 0.5 * pct_above_ma20 + 0.5 * pct_above_ma60 ∈ [0, 1]
    b1_bull: 1=bull, 0=bear (含 hysteresis 0.45/0.55 + 3d streak)
    b1_streak: 当前 regime 已连续天数

设计:
    - 滚动 MA 用 pandas rolling, min_periods=ma_window 保证前 N-1 日 NaN
    - hysteresis: score > 0.55 → 倾向 bull; < 0.45 → 倾向 bear; 0.45-0.55 沿用前值
    - streak: 倾向值连续 N 日才确认
    - 数据不足前 max(ma_long, streak_days) 日返回 NaN/0 (调用方需 dropna)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_threshold_with_hysteresis_and_streak(
    score: pd.Series,
    lo: float = 0.45,
    hi: float = 0.55,
    streak_days: int = 3,
    initial: int = 0,
) -> tuple[pd.Series, pd.Series]:
    """对 score 应用 hysteresis + streak, 返回 (bull, streak).

    raw_bull[t] = 1 if score[t] > hi else (0 if score[t] < lo else raw_bull[t-1])
    confirmed_bull[t] flips only after streak_days consecutive raw_bull[t] same as candidate.
    """
    n = len(score)
    raw = np.full(n, initial, dtype=np.int8)
    prev = initial
    for i, s in enumerate(score.values):
        if np.isnan(s):
            raw[i] = prev
            continue
        if s > hi:
            raw[i] = 1
        elif s < lo:
            raw[i] = 0
        else:
            raw[i] = prev
        prev = raw[i]

    # streak: count consecutive same-value days; flip confirmed only after streak_days
    confirmed = np.full(n, initial, dtype=np.int8)
    streak = np.zeros(n, dtype=np.int32)
    cur = initial
    cur_streak = 0
    for i in range(n):
        if raw[i] == cur:
            cur_streak += 1
        else:
            # candidate flip
            if cur_streak < 0:
                cur_streak = 1
            else:
                cur_streak = 1
        # check if we should flip cur to raw[i]
        if raw[i] != cur:
            # build run length of raw[i] from i backwards
            run = 1
            j = i - 1
            while j >= 0 and raw[j] == raw[i]:
                run += 1
                j -= 1
            if run >= streak_days:
                cur = raw[i]
                cur_streak = run
        confirmed[i] = cur
        streak[i] = cur_streak

    return (
        pd.Series(confirmed, index=score.index, name='confirmed_bull'),
        pd.Series(streak, index=score.index, name='streak'),
    )


def compute_breadth_signal(
    close_panel: pd.DataFrame,
    ma_short: int = 20,
    ma_long: int = 60,
    weight_short: float = 0.5,
    streak_days: int = 3,
    hysteresis_lo: float = 0.45,
    hysteresis_hi: float = 0.55,
) -> pd.DataFrame:
    """B1 breadth signal.

    close_panel: DataFrame (index=trade_date, columns=stock_code). NaN allowed.
    Returns: DataFrame indexed by trade_date with regime columns.
    """
    if close_panel.empty:
        raise ValueError('close_panel is empty')
    if not (0 <= weight_short <= 1):
        raise ValueError(f'weight_short={weight_short} must be in [0,1]')

    ma_s = close_panel.rolling(ma_short, min_periods=ma_short).mean()
    ma_l = close_panel.rolling(ma_long, min_periods=ma_long).mean()

    above_s = (close_panel > ma_s)
    above_l = (close_panel > ma_l)

    # Per date: % of stocks with valid MA (not NaN) that are above MA
    valid_s = ma_s.notna() & close_panel.notna()
    valid_l = ma_l.notna() & close_panel.notna()
    pct_s = above_s.where(valid_s).sum(axis=1) / valid_s.sum(axis=1).replace(0, np.nan)
    pct_l = above_l.where(valid_l).sum(axis=1) / valid_l.sum(axis=1).replace(0, np.nan)

    score = weight_short * pct_s + (1.0 - weight_short) * pct_l

    bull, streak = apply_threshold_with_hysteresis_and_streak(
        score, lo=hysteresis_lo, hi=hysteresis_hi, streak_days=streak_days, initial=0
    )

    out = pd.DataFrame({
        'pct_above_ma20': pct_s,
        'pct_above_ma60': pct_l,
        'b1_score': score,
        'b1_bull': bull.astype('Int8'),
        'b1_streak': streak.astype('Int32'),
    })
    return out
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python3 -m pytest tests/indicators/test_breadth.py -v
```

Expected: 5 PASS.

- [ ] **Step 5: /simplify 一轮**

```bash
# In Claude Code session: run /simplify on indicators/breadth.py
```

Expected: review for DRY/YAGNI/clarity, fix any flagged issues.

- [ ] **Step 6: Commit**

```bash
git add indicators/breadth.py tests/indicators/__init__.py tests/indicators/test_breadth.py
git commit -m "feat(ng2.0a): add B1 breadth signal (% stocks above MA20/MA60 + hysteresis + 3d streak)"
```

---

## Task 2: 创建 B2 已实现波动率信号模块

**Files:**
- Create: `indicators/realized_vol.py`
- Create: `tests/indicators/test_realized_vol.py`

- [ ] **Step 1: 写失败测试**

Create `tests/indicators/test_realized_vol.py`:

```python
"""B2 realized vol signal unit tests."""
import numpy as np
import pandas as pd
import pytest

from indicators.realized_vol import compute_realized_vol_signal


def _fake_index_close(n=400, vol=0.012, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2023-01-01', periods=n, freq='B')
    rets = rng.normal(0, vol, size=n)
    prices = 3000.0 * np.exp(np.cumsum(rets))
    return pd.Series(prices, index=dates, name='close')


def test_realized_vol_returns_required_cols():
    s = _fake_index_close()
    out = compute_realized_vol_signal(s, rv_window=60, percentile_window=252, streak_days=3)
    assert {'rv_60d', 'rv_percentile_252', 'b2_bull', 'b2_streak'} <= set(out.columns)
    assert len(out) == len(s)


def test_realized_vol_low_vol_is_bull():
    """Constant-low-vol series → b2_bull=1 after streak."""
    n = 400
    dates = pd.date_range('2023-01-01', periods=n, freq='B')
    rng = np.random.default_rng(1)
    # Mix: high vol first 252 days (calibration), then low vol
    rets_high = rng.normal(0, 0.020, size=252)
    rets_low = rng.normal(0, 0.005, size=n - 252)
    rets = np.concatenate([rets_high, rets_low])
    prices = 3000.0 * np.exp(np.cumsum(rets))
    s = pd.Series(prices, index=dates)
    out = compute_realized_vol_signal(s, rv_window=60, percentile_window=252, streak_days=3)
    # End of low-vol period: percentile should be very low → b2_bull=1
    assert out['b2_bull'].iloc[-1] == 1
    assert out['rv_percentile_252'].iloc[-1] < 0.30


def test_realized_vol_high_vol_is_bear():
    n = 400
    dates = pd.date_range('2023-01-01', periods=n, freq='B')
    rng = np.random.default_rng(2)
    rets_low = rng.normal(0, 0.005, size=252)
    rets_high = rng.normal(0, 0.030, size=n - 252)
    rets = np.concatenate([rets_low, rets_high])
    prices = 3000.0 * np.exp(np.cumsum(rets))
    s = pd.Series(prices, index=dates)
    out = compute_realized_vol_signal(s, rv_window=60, percentile_window=252, streak_days=3)
    assert out['b2_bull'].iloc[-1] == 0
    assert out['rv_percentile_252'].iloc[-1] > 0.70


def test_realized_vol_no_lookahead():
    """Output at t must only use data up to t (no future-looking)."""
    s = _fake_index_close(n=350)
    out_full = compute_realized_vol_signal(s, rv_window=60, percentile_window=252, streak_days=3)
    out_truncated = compute_realized_vol_signal(s.iloc[:300], rv_window=60, percentile_window=252, streak_days=3)
    # Values at common indices must be identical
    common = out_full.index.intersection(out_truncated.index)
    assert (out_full.loc[common, 'rv_60d'] == out_truncated.loc[common, 'rv_60d']).all()
    assert (out_full.loc[common, 'rv_percentile_252'].round(8)
            == out_truncated.loc[common, 'rv_percentile_252'].round(8)).all()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/indicators/test_realized_vol.py -v
```

Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: 实现 indicators/realized_vol.py**

```python
"""B2 已实现波动率信号 (60d RV + 252d percentile + hysteresis + 3d streak).

输入: 沪深 300 收盘价 Series (index=trade_date)
输出: DataFrame with columns rv_60d, rv_percentile_252, b2_bull, b2_streak

设计:
    - rv_60d = std(log_returns) over 60d, annualized 不必 (相对 percentile 不变)
    - rv_percentile_252 = rolling rank of rv_60d in past 252d window
    - hysteresis: percentile < 30 → 倾向 bull (低波好); > 70 → 倾向 bear (高波坏); 30-70 沿用前值
    - 3d streak 才切换
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from indicators.breadth import apply_threshold_with_hysteresis_and_streak


def compute_realized_vol_signal(
    close: pd.Series,
    rv_window: int = 60,
    percentile_window: int = 252,
    streak_days: int = 3,
    hysteresis_lo: float = 0.30,
    hysteresis_hi: float = 0.70,
) -> pd.DataFrame:
    """B2 realized vol signal."""
    if close.empty:
        raise ValueError('close is empty')
    if (close <= 0).any():
        raise ValueError('close contains non-positive values')

    log_ret = np.log(close).diff()
    rv = log_ret.rolling(rv_window, min_periods=rv_window).std()

    # Rolling percentile of rv within past percentile_window days
    pct = rv.rolling(percentile_window, min_periods=percentile_window).rank(pct=True)

    # Note: low percentile = low vol = bull-friendly.
    # We want: pct < 0.30 → bull (1); pct > 0.70 → bear (0).
    # apply_threshold expects: score > hi → 1 (bull), < lo → 0 (bear).
    # So we feed score = 1 - pct (inverted).
    score = 1.0 - pct
    bull, streak = apply_threshold_with_hysteresis_and_streak(
        score,
        lo=1.0 - hysteresis_hi,  # = 0.30 (when pct > 0.70 → score < 0.30 → bear)
        hi=1.0 - hysteresis_lo,  # = 0.70 (when pct < 0.30 → score > 0.70 → bull)
        streak_days=streak_days,
        initial=1,  # default neutral-bull (low vol baseline)
    )

    out = pd.DataFrame({
        'rv_60d': rv,
        'rv_percentile_252': pct,
        'b2_bull': bull.astype('Int8'),
        'b2_streak': streak.astype('Int32'),
    })
    return out
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python3 -m pytest tests/indicators/test_realized_vol.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: /simplify 一轮**

- [ ] **Step 6: Commit**

```bash
git add indicators/realized_vol.py tests/indicators/test_realized_vol.py
git commit -m "feat(ng2.0a): add B2 realized vol signal (60d RV + 252d percentile + 3d streak)"
```

---

## Task 3: regime_classifier.py 加 compute_regime_v2 投票函数

**Files:**
- Modify: `indicators/regime_classifier.py` (append)
- Create: `tests/indicators/test_regime_v2.py`

- [ ] **Step 1: 写失败测试**

Create `tests/indicators/test_regime_v2.py`:

```python
"""ng2.0a multi-beta vote regime unit tests."""
import numpy as np
import pandas as pd
import pytest

from indicators.regime_classifier import compute_regime_v2


def test_regime_v2_unanimous_bull():
    n = 30
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    v11 = pd.Series([1] * n, index=idx)
    b1 = pd.Series([1] * n, index=idx)
    b2 = pd.Series([1] * n, index=idx)
    out = compute_regime_v2(v11, b1, b2, system_streak=3)
    # After streak_days=3, all should be bull (+1)
    assert (out['regime_v2'].iloc[3:] == 1).all()


def test_regime_v2_unanimous_bear():
    n = 30
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    v11 = pd.Series([0] * n, index=idx)
    b1 = pd.Series([0] * n, index=idx)
    b2 = pd.Series([0] * n, index=idx)
    out = compute_regime_v2(v11, b1, b2, system_streak=3)
    assert (out['regime_v2'].iloc[3:] == -1).all()


def test_regime_v2_majority_vote():
    n = 30
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    # 2-of-3 bull (V11+B1, B2 bear)
    v11 = pd.Series([1] * n, index=idx)
    b1 = pd.Series([1] * n, index=idx)
    b2 = pd.Series([0] * n, index=idx)
    out = compute_regime_v2(v11, b1, b2, system_streak=3)
    assert out['vote_count'].iloc[-1] == 2
    assert out['regime_v2_raw'].iloc[-1] == 1
    assert out['regime_v2'].iloc[-1] == 1


def test_regime_v2_minority_is_bear():
    n = 30
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    v11 = pd.Series([1] * n, index=idx)
    b1 = pd.Series([0] * n, index=idx)
    b2 = pd.Series([0] * n, index=idx)
    out = compute_regime_v2(v11, b1, b2, system_streak=3)
    assert out['vote_count'].iloc[-1] == 1
    assert out['regime_v2_raw'].iloc[-1] == -1


def test_regime_v2_streak_blocks_one_day_flip():
    """Single day vote-flip shouldn't propagate without streak."""
    n = 10
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    # Mostly bear, 1 day bull majority on day 5
    v11 = pd.Series([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], index=idx)
    b1 = pd.Series([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], index=idx)
    b2 = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], index=idx)
    out = compute_regime_v2(v11, b1, b2, system_streak=3)
    # Day 5 raw flips to +1 but streak<3 → confirmed regime_v2 should stay -1
    assert out['regime_v2'].iloc[4] == -1


def test_regime_v2_returns_required_columns():
    n = 30
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    v11 = pd.Series([1] * n, index=idx)
    b1 = pd.Series([1] * n, index=idx)
    b2 = pd.Series([1] * n, index=idx)
    out = compute_regime_v2(v11, b1, b2)
    assert {'vote_count', 'regime_v2_raw', 'regime_v2', 'regime_v2_streak'} <= set(out.columns)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/indicators/test_regime_v2.py -v
```

Expected: FAIL with `ImportError: cannot import name 'compute_regime_v2'`.

- [ ] **Step 3: 实现 compute_regime_v2 (追加到 indicators/regime_classifier.py 末尾)**

Append to `indicators/regime_classifier.py`:

```python
# ============================================================
# ng2.0a: Multi-beta vote regime
# ============================================================

def compute_regime_v2(
    v11_bull: pd.Series,
    b1_bull: pd.Series,
    b2_bull: pd.Series,
    system_streak: int = 3,
) -> pd.DataFrame:
    """ng2.0a: hard vote across 3 binary signals + system-level streak.

    Args:
        v11_bull, b1_bull, b2_bull: each 1=bull, 0=bear, NaN allowed (will be treated as 0)
            All three Series must share the same DatetimeIndex.
        system_streak: streak_days at vote-output level (default 3)

    Returns:
        DataFrame indexed same as inputs with columns:
            vote_count: int 0..3 (count of bulls)
            regime_v2_raw: +1 if vote ≥ 2 else -1
            regime_v2_streak: consecutive days at current regime
            regime_v2: +1/-1 (after system_streak applied to raw)
    """
    if not (v11_bull.index.equals(b1_bull.index) and v11_bull.index.equals(b2_bull.index)):
        raise ValueError('v11_bull / b1_bull / b2_bull indices must match')

    v = v11_bull.fillna(0).astype(int)
    b1 = b1_bull.fillna(0).astype(int)
    b2 = b2_bull.fillna(0).astype(int)

    vote = v + b1 + b2  # 0..3
    raw_bull_int = (vote >= 2).astype(int)  # 1=bull, 0=bear

    # Apply system-level streak using existing persist_n
    raw_arr = raw_bull_int.to_numpy()
    confirmed = persist_n(raw_arr, n_days=system_streak)  # +1/-1

    # Track streak length
    n = len(raw_arr)
    streak = np.zeros(n, dtype=np.int32)
    cur_run = 1
    for i in range(1, n):
        if raw_arr[i] == raw_arr[i - 1]:
            cur_run += 1
        else:
            cur_run = 1
        streak[i] = cur_run
    if n > 0:
        streak[0] = 1

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
python3 -m pytest tests/indicators/test_regime_v2.py -v
```

Expected: 6 PASS.

- [ ] **Step 5: /simplify 一轮**

- [ ] **Step 6: Commit**

```bash
git add indicators/regime_classifier.py tests/indicators/test_regime_v2.py
git commit -m "feat(ng2.0a): add compute_regime_v2 hard-vote across V11+B1+B2 with system streak"
```

---

## Task 4: 创建 market_regime_signals DB 表 + migration

**Files:**
- Create: `scripts/migrations/2026_04_25_add_market_regime_signals.py`

- [ ] **Step 1: 检查表是否已存在**

```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('data_adapter/stock_data.db', timeout=30)
r = conn.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name='market_regime_signals'\").fetchone()
print('exists' if r else 'absent')
"
```

Expected: `absent`.

- [ ] **Step 2: 写 migration 脚本**

Create `scripts/migrations/2026_04_25_add_market_regime_signals.py`:

```python
"""Migration: create market_regime_signals table for ng2.0a.

Idempotent: safe to re-run, will not drop or alter existing data.
"""
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / 'data_adapter' / 'stock_data.db'

DDL = """
CREATE TABLE IF NOT EXISTS market_regime_signals (
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

INDEX_DDL = "CREATE INDEX IF NOT EXISTS idx_mrs_regime_v2 ON market_regime_signals(regime_v2);"


def main():
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.execute('PRAGMA busy_timeout=30000')
    try:
        conn.executescript(DDL)
        conn.execute(INDEX_DDL)
        conn.commit()
        print(f'OK: market_regime_signals created in {DB_PATH}')
    finally:
        conn.close()


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: 跑 migration**

```bash
mkdir -p scripts/migrations
python3 scripts/migrations/2026_04_25_add_market_regime_signals.py
```

Expected: `OK: market_regime_signals created in ...`

- [ ] **Step 4: 验证表结构**

```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('data_adapter/stock_data.db', timeout=30)
cols = [r[1] for r in conn.execute('PRAGMA table_info(market_regime_signals)').fetchall()]
print('cols:', cols)
assert 'regime_v2' in cols
assert 'vote_count' in cols
print('OK')
"
```

Expected: prints column list, ends with `OK`.

- [ ] **Step 5: Commit**

```bash
git add scripts/migrations/2026_04_25_add_market_regime_signals.py
git commit -m "feat(ng2.0a): add market_regime_signals table migration"
```

---

## Task 5: 创建 build_regime_v2_history.py 历史回算脚本

**Files:**
- Create: `scripts/build_regime_v2_history.py`

- [ ] **Step 1: 写脚本**

Create `scripts/build_regime_v2_history.py`:

```python
"""Backfill market_regime_signals for ng2.0a (V11 + B1 + B2 + vote).

Runs once over 2018-2026 to populate the table with all historical regime states.

Usage:
    python3 scripts/build_regime_v2_history.py --start 2018-01-01 --end 2026-04-25
    python3 scripts/build_regime_v2_history.py --replace   # drop existing rows first
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indicators import market_amv
from indicators.breadth import compute_breadth_signal
from indicators.realized_vol import compute_realized_vol_signal
from indicators.regime_classifier import compute_regime_v2

DB_PATH = Path(__file__).resolve().parents[1] / 'data_adapter' / 'stock_data.db'

# 沪深 300 在 securities 表里
HS300_CODE = '000300.SH'


def load_close_panel(conn, start: str, end: str) -> pd.DataFrame:
    """Load A-share close panel (excluding ETFs/indices). 排除 ST 名称的列由调用方决定."""
    q = """
        SELECT s.code AS code, q.trade_date AS date, q.close AS close
        FROM daily_quotes q
        JOIN securities s ON s.id = q.security_id
        WHERE s.type = 'A股'
          AND q.trade_date BETWEEN ? AND ?
          AND q.close IS NOT NULL
    """
    df = pd.read_sql(q, conn, params=(start, end))
    if df.empty:
        raise RuntimeError(f'No A-share data between {start} and {end}')
    df['date'] = pd.to_datetime(df['date'])
    panel = df.pivot(index='date', columns='code', values='close').sort_index()
    return panel


def load_index_close(conn, code: str, start: str, end: str) -> pd.Series:
    q = """
        SELECT q.trade_date AS date, q.close AS close
        FROM daily_quotes q
        JOIN securities s ON s.id = q.security_id
        WHERE s.code = ?
          AND q.trade_date BETWEEN ? AND ?
        ORDER BY q.trade_date
    """
    df = pd.read_sql(q, conn, params=(code, start, end))
    if df.empty:
        raise RuntimeError(f'No data for {code} between {start} and {end}')
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date')['close'].astype(float)


def load_v11_regime(conn, start: str, end: str) -> pd.DataFrame:
    """Load existing V11 0AMV regime from market_amv table."""
    q = """
        SELECT trade_date, var1, ma60, macd, amv_regime
        FROM market_amv
        WHERE trade_date BETWEEN ? AND ?
        ORDER BY trade_date
    """
    df = pd.read_sql(q, conn, params=(start, end))
    if df.empty:
        raise RuntimeError('market_amv is empty; run indicators/market_amv.py first')
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.set_index('trade_date')
    # amv_regime: +1=bull, -1=bear → convert to 1/0
    df['v11_bull'] = (df['amv_regime'] == 1).astype(int)
    # streak: count consecutive same-value
    streak = np.zeros(len(df), dtype=int)
    cur_run = 1
    for i in range(1, len(df)):
        if df['amv_regime'].iat[i] == df['amv_regime'].iat[i - 1]:
            cur_run += 1
        else:
            cur_run = 1
        streak[i] = cur_run
    if len(df) > 0:
        streak[0] = 1
    df['v11_streak'] = streak
    return df[['var1', 'ma60', 'macd', 'v11_bull', 'v11_streak']]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--start', default='2018-01-01')
    p.add_argument('--end', default='2026-04-25')
    p.add_argument('--replace', action='store_true', help='delete existing rows in date range first')
    args = p.parse_args()

    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.execute('PRAGMA busy_timeout=30000')

    try:
        print(f'[1/5] Load V11 regime from market_amv table ({args.start}..{args.end})')
        v11_df = load_v11_regime(conn, args.start, args.end)
        print(f'      V11 rows: {len(v11_df)}')

        print(f'[2/5] Load A-share close panel (need 60+ days lookback for B1)')
        # B1 needs 60d MA → fetch from 90 days before start
        lookback_start = (pd.Timestamp(args.start) - pd.Timedelta(days=120)).strftime('%Y-%m-%d')
        panel = load_close_panel(conn, lookback_start, args.end)
        print(f'      panel: {panel.shape[0]} dates × {panel.shape[1]} stocks')

        print(f'[3/5] Compute B1 breadth signal')
        b1 = compute_breadth_signal(panel, ma_short=20, ma_long=60, streak_days=3)
        # Trim to actual date range
        b1 = b1.loc[b1.index >= pd.Timestamp(args.start)]
        print(f'      B1 rows: {len(b1)}, bull days: {int(b1["b1_bull"].fillna(0).sum())}')

        print(f'[4/5] Compute B2 realized vol signal on {HS300_CODE}')
        # B2 needs 60+252 = 312 days lookback
        b2_lookback_start = (pd.Timestamp(args.start) - pd.Timedelta(days=500)).strftime('%Y-%m-%d')
        idx_close = load_index_close(conn, HS300_CODE, b2_lookback_start, args.end)
        b2 = compute_realized_vol_signal(idx_close, rv_window=60, percentile_window=252, streak_days=3)
        b2 = b2.loc[b2.index >= pd.Timestamp(args.start)]
        print(f'      B2 rows: {len(b2)}, bull days: {int(b2["b2_bull"].fillna(0).sum())}')

        print(f'[5/5] Compute regime_v2 vote + write to DB')
        # Align all three signals by date intersection
        common = v11_df.index.intersection(b1.index).intersection(b2.index)
        v11_aligned = v11_df.loc[common]
        b1_aligned = b1.loc[common]
        b2_aligned = b2.loc[common]
        vote = compute_regime_v2(
            v11_aligned['v11_bull'].astype(int),
            b1_aligned['b1_bull'].fillna(0).astype(int),
            b2_aligned['b2_bull'].fillna(0).astype(int),
            system_streak=3,
        )

        merged = pd.concat([v11_aligned, b1_aligned, b2_aligned, vote], axis=1)
        merged.index.name = 'trade_date'
        merged = merged.reset_index()
        merged['trade_date'] = merged['trade_date'].dt.strftime('%Y-%m-%d')

        # Match DB column names; b1_adv_dec_ratio currently NULL (not computed in v1, can add later)
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

        if args.replace:
            conn.execute(
                'DELETE FROM market_regime_signals WHERE trade_date BETWEEN ? AND ?',
                (args.start, args.end),
            )
        out.to_sql('market_regime_signals', conn, if_exists='append', index=False)
        conn.commit()

        bull_n = int((out['regime_v2'] == 1).sum())
        bear_n = int((out['regime_v2'] == -1).sum())
        print(f'      written {len(out)} rows: bull={bull_n}, bear={bear_n}')
    finally:
        conn.close()


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 跑回算 (2018-2026)**

```bash
python3 scripts/build_regime_v2_history.py --start 2018-01-01 --end 2026-04-25 --replace 2>&1 | tee logs/ng2_0a_build_regime_history.log
```

Expected: prints `[1/5]` through `[5/5]`, final line shows `written N rows: bull=X, bear=Y` with N ~1900-2000 trading days, both X and Y > 0.

- [ ] **Step 3: 验证写入数据**

```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('data_adapter/stock_data.db', timeout=30)
n = conn.execute('SELECT COUNT(*) FROM market_regime_signals').fetchone()[0]
bull = conn.execute('SELECT COUNT(*) FROM market_regime_signals WHERE regime_v2=1').fetchone()[0]
bear = conn.execute('SELECT COUNT(*) FROM market_regime_signals WHERE regime_v2=-1').fetchone()[0]
nan = conn.execute('SELECT COUNT(*) FROM market_regime_signals WHERE regime_v2 IS NULL').fetchone()[0]
print(f'rows={n}, bull={bull}, bear={bear}, null={nan}')
assert n > 1500, 'expected >1500 trading days'
assert bull > 0 and bear > 0, 'expected both bull and bear days'
print('OK')
"
```

Expected: rows in 1500-2200 range, both bull and bear > 0, null=0.

- [ ] **Step 4: /simplify 一轮**

- [ ] **Step 5: Commit**

```bash
git add scripts/build_regime_v2_history.py
git commit -m "feat(ng2.0a): backfill market_regime_signals 2018-2026 (V11+B1+B2 vote)"
```

---

## Task 6: Step A 验证脚本 + 跑验证 + 决策

**Files:**
- Create: `scripts/compare_regime_v1_v2.py`

- [ ] **Step 1: 写脚本**

Create `scripts/compare_regime_v1_v2.py`:

```python
"""Step A validation: compare V11 (regime_v1) vs ng2.0a multi-beta vote (regime_v2).

Outputs:
  1. Regime distribution: % bull / % bear, total trading days
  2. Flip count: regime transitions in 2020-2026
  3. Agreement matrix (V11 × ng2.0a): how often they agree
  4. 2018-2019 sanity check: did ng2.0a identify 2018Q4 bear & 2019Q1 bull rebound?
  5. PASS/ABORT decision per spec section 6.1

Usage: python3 scripts/compare_regime_v1_v2.py
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).resolve().parents[1] / 'data_adapter' / 'stock_data.db'


def load_regimes(start: str, end: str) -> pd.DataFrame:
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    q = """
        SELECT mrs.trade_date AS date,
               ma.amv_regime AS regime_v1,
               mrs.regime_v2 AS regime_v2,
               mrs.v11_bull, mrs.b1_bull, mrs.b2_bull, mrs.vote_count
        FROM market_regime_signals mrs
        JOIN market_amv ma ON ma.trade_date = mrs.trade_date
        WHERE mrs.trade_date BETWEEN ? AND ?
        ORDER BY mrs.trade_date
    """
    df = pd.read_sql(q, conn, params=(start, end))
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date')


def count_flips(regime: pd.Series) -> int:
    return int((regime.diff().fillna(0) != 0).sum())


def regime_distribution(regime: pd.Series) -> dict:
    n = len(regime)
    return {
        'total_days': n,
        'bull_days': int((regime == 1).sum()),
        'bear_days': int((regime == -1).sum()),
        'pct_bull': round(100.0 * (regime == 1).sum() / n, 2) if n else 0,
        'pct_bear': round(100.0 * (regime == -1).sum() / n, 2) if n else 0,
    }


def agreement_matrix(v1: pd.Series, v2: pd.Series) -> dict:
    n = len(v1)
    both_bull = int(((v1 == 1) & (v2 == 1)).sum())
    both_bear = int(((v1 == -1) & (v2 == -1)).sum())
    v1_bull_v2_bear = int(((v1 == 1) & (v2 == -1)).sum())
    v1_bear_v2_bull = int(((v1 == -1) & (v2 == 1)).sum())
    agree = both_bull + both_bear
    return {
        'total': n,
        'agree': agree,
        'agree_pct': round(100.0 * agree / n, 2) if n else 0,
        'both_bull': both_bull,
        'both_bear': both_bear,
        'v1_bull_v2_bear': v1_bull_v2_bear,
        'v1_bear_v2_bull': v1_bear_v2_bull,
    }


def sanity_2018_2019(df_18_19: pd.DataFrame) -> dict:
    """Did v2 identify 2018 bear (Q4 selloff) + 2019 Q1 rebound bull?"""
    q4_2018 = df_18_19.loc['2018-10-01':'2018-12-31']
    q1_2019 = df_18_19.loc['2019-01-01':'2019-03-31']
    return {
        '2018_q4_bear_days_v2': int((q4_2018['regime_v2'] == -1).sum()),
        '2018_q4_total': len(q4_2018),
        '2018_q4_bear_days_v1': int((q4_2018['regime_v1'] == -1).sum()),
        '2019_q1_bull_days_v2': int((q1_2019['regime_v2'] == 1).sum()),
        '2019_q1_total': len(q1_2019),
        '2019_q1_bull_days_v1': int((q1_2019['regime_v1'] == 1).sum()),
    }


def main():
    print('=' * 70)
    print('Step A Validation: regime_v1 (V11) vs regime_v2 (ng2.0a multi-beta vote)')
    print('=' * 70)

    # Main analysis window: 2020-2026
    df = load_regimes('2020-01-01', '2026-04-25')
    print(f'\n[Main window 2020-2026: {len(df)} trading days]\n')

    # 1. Distributions
    dist_v1 = regime_distribution(df['regime_v1'])
    dist_v2 = regime_distribution(df['regime_v2'])
    print('1. Regime distribution:')
    print(f'   V11 (v1):    bull={dist_v1["pct_bull"]}%  bear={dist_v1["pct_bear"]}%  total={dist_v1["total_days"]}d')
    print(f'   ng2.0a (v2): bull={dist_v2["pct_bull"]}%  bear={dist_v2["pct_bear"]}%  total={dist_v2["total_days"]}d')
    pct_diff = abs(dist_v1['pct_bull'] - dist_v2['pct_bull'])
    print(f'   Δ%bull: {pct_diff:.2f}pp')

    # 2. Flip count
    flips_v1 = count_flips(df['regime_v1'])
    flips_v2 = count_flips(df['regime_v2'])
    print('\n2. Flip count (transitions):')
    print(f'   V11 (v1):    {flips_v1} flips')
    print(f'   ng2.0a (v2): {flips_v2} flips')
    flip_ratio = flips_v2 / flips_v1 if flips_v1 else float('inf')
    print(f'   ratio v2/v1: {flip_ratio:.2f}x')

    # 3. Agreement
    agree = agreement_matrix(df['regime_v1'], df['regime_v2'])
    print('\n3. Agreement matrix (V11 × ng2.0a):')
    print(f'   agree: {agree["agree"]}/{agree["total"]} = {agree["agree_pct"]}%')
    print(f'   both_bull={agree["both_bull"]}, both_bear={agree["both_bear"]}')
    print(f'   v1_bull_v2_bear={agree["v1_bull_v2_bear"]}, v1_bear_v2_bull={agree["v1_bear_v2_bull"]}')

    # 4. Sanity 2018-2019
    df_18_19 = load_regimes('2018-01-01', '2019-12-31')
    sanity = sanity_2018_2019(df_18_19)
    print('\n4. 2018-2019 sanity check:')
    print(f'   2018 Q4 (selloff expected): v2 bear days = {sanity["2018_q4_bear_days_v2"]}/{sanity["2018_q4_total"]} '
          f'(v1: {sanity["2018_q4_bear_days_v1"]})')
    print(f'   2019 Q1 (rebound expected): v2 bull days = {sanity["2019_q1_bull_days_v2"]}/{sanity["2019_q1_total"]} '
          f'(v1: {sanity["2019_q1_bull_days_v1"]})')

    # 5. Decision per spec section 6.1
    print('\n5. PASS/ABORT decision (spec gates):')
    issues = []
    if flip_ratio > 1.5:
        issues.append(f'ABORT: flip ratio {flip_ratio:.2f}x > 1.5x (whipsaw)')
    if pct_diff > 25:
        issues.append(f'ABORT: %bull diff {pct_diff:.2f}pp > 25pp')
    if agree['agree_pct'] < 50:
        issues.append(f'ABORT: agreement {agree["agree_pct"]}% < 50%')
    if sanity['2018_q4_bear_days_v2'] == 0:
        issues.append('FAIL sanity: ng2.0a missed 2018 Q4 bear')
    if sanity['2019_q1_bull_days_v2'] == 0:
        issues.append('FAIL sanity: ng2.0a missed 2019 Q1 bull rebound')

    if issues:
        print('   ❌ STEP A NOT PASSING:')
        for i in issues:
            print(f'     - {i}')
        sys.exit(1)
    else:
        print('   ✅ Step A primary gates PASS — proceed to Step B end-to-end backtest')

    # Save report
    report_path = Path('reports') / 'ng2_0a_step_a_report.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(f'# ng2.0a Step A Report\n\n')
        f.write(f'**Date**: 2026-04-25\n\n')
        f.write(f'## Distributions\n- V11: bull {dist_v1["pct_bull"]}%, bear {dist_v1["pct_bear"]}%\n')
        f.write(f'- ng2.0a: bull {dist_v2["pct_bull"]}%, bear {dist_v2["pct_bear"]}%\n\n')
        f.write(f'## Flips (2020-2026)\n- V11: {flips_v1}\n- ng2.0a: {flips_v2}\n- ratio: {flip_ratio:.2f}x\n\n')
        f.write(f'## Agreement\n- {agree["agree_pct"]}% agree ({agree["agree"]}/{agree["total"]})\n\n')
        f.write(f'## 2018-2019 Sanity\n- 2018 Q4 v2 bear: {sanity["2018_q4_bear_days_v2"]}/{sanity["2018_q4_total"]}\n')
        f.write(f'- 2019 Q1 v2 bull: {sanity["2019_q1_bull_days_v2"]}/{sanity["2019_q1_total"]}\n')
    print(f'\n📄 Report saved to {report_path}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 跑验证**

```bash
python3 scripts/compare_regime_v1_v2.py 2>&1 | tee logs/ng2_0a_step_a.log
```

Expected output: prints sections 1-5, ends with either:
- `✅ Step A primary gates PASS` → proceed
- `❌ STEP A NOT PASSING` with specific issues → STOP, fix and re-run

If ABORT: investigate flip ratio (likely needs hysteresis tuning in B1 or B2). Iterate on:
- B1 thresholds: try (0.40, 0.60) or (0.50, 0.50) → re-run from Task 5 backfill
- B2 thresholds: try (0.25, 0.75) → re-run from Task 5 backfill
- system_streak: increase to 5 → re-run from Task 5 backfill

After two failed iterations, kill ng2.0a and revert (per spec section 13 termination condition #1).

- [ ] **Step 3: 用户审阅 Step A 报告**

Show user `reports/ng2_0a_step_a_report.md` and `logs/ng2_0a_step_a.log`. Wait for go/no-go before proceeding to Task 7.

- [ ] **Step 4: Commit**

```bash
git add scripts/compare_regime_v1_v2.py reports/ng2_0a_step_a_report.md logs/ng2_0a_step_a.log
git commit -m "feat(ng2.0a): Step A validation passed — multi-beta regime stable vs V11 baseline"
```

---

## Task 7: regime_switch_backtest.py 加 --regime-version v2

**Files:**
- Modify: `backtest/regime_switch_backtest.py`

- [ ] **Step 1: 看现有 load_regime 定义并替换**

Current (line 22-35) reads from `market_amv.amv_regime`. Replace with version-dispatched loader.

Edit `backtest/regime_switch_backtest.py`:

Replace the existing `load_regime` function with:

```python
def load_regime(db_path=None, version: str = 'v1'):
    """加载每日 regime, 返回 {date_str: regime_int}.

    version='v1': use market_amv.amv_regime (legacy V11 0AMV)
    version='v2': use market_regime_signals.regime_v2 (ng2.0a multi-beta vote)
    """
    if db_path is None:
        db_path = Path(__file__).resolve().parents[1] / 'data_adapter' / 'stock_data.db'
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.execute('PRAGMA busy_timeout=30000')
    try:
        if version == 'v1':
            cur = conn.execute(
                'SELECT trade_date, amv_regime FROM market_amv ORDER BY trade_date'
            )
        elif version == 'v2':
            cur = conn.execute(
                'SELECT trade_date, regime_v2 FROM market_regime_signals '
                'WHERE regime_v2 IS NOT NULL ORDER BY trade_date'
            )
        else:
            raise ValueError(f'unknown regime version: {version!r}')
        regime = {}
        for row in cur.fetchall():
            date_str, r = row[0], row[1]
            regime[date_str] = int(r)
        return regime
    finally:
        conn.close()
```

Then in the `main()` function (around line 119), find the `regime = load_regime()` line and update with argparse-driven version + caller updates:

```python
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--regime-version', choices=['v1', 'v2'], default='v1',
                        help='v1=V11 0AMV (legacy), v2=ng2.0a multi-beta vote')
    parser.add_argument('--bull-dir', default=None, help='override bull report dir')
    parser.add_argument('--bear-dir', default=None, help='override bear report dir')
    parser.add_argument('--out-dir', default=None, help='override output dir')
    args = parser.parse_args()

    regime = load_regime(version=args.regime_version)
    if not regime:
        print(f'ERROR: regime version={args.regime_version} table empty; '
              'run indicators/market_amv.py or scripts/build_regime_v2_history.py first')
        return

    # ... (rest of main, replacing hardcoded paths with args.bull_dir / args.bear_dir / args.out_dir)
```

If existing main has hardcoded report dirs, keep those as defaults but allow CLI override. The `--out-dir` should default to `reports/daily_selection_regime_switch_v{version}` to keep v1 and v2 outputs separate.

- [ ] **Step 2: 检查 Path import 存在**

```bash
grep -n "^from pathlib import Path\|^import pathlib" backtest/regime_switch_backtest.py
```

If absent: add `from pathlib import Path` to imports at top of file.

- [ ] **Step 3: smoke-test v1 仍工作 (regression)**

```bash
python3 backtest/regime_switch_backtest.py --regime-version v1 --out-dir /tmp/regime_v1_smoke 2>&1 | head -20
```

Expected: prints `体制信号: Nd, 牛市Xd, 熊市Yd` matching pre-change behavior.

- [ ] **Step 4: smoke-test v2 加载**

```bash
python3 -c "
import sys
sys.path.insert(0, 'backtest')
from regime_switch_backtest import load_regime
r = load_regime(version='v2')
bull = sum(1 for v in r.values() if v == 1)
bear = sum(1 for v in r.values() if v == -1)
print(f'v2 loaded: total={len(r)}, bull={bull}, bear={bear}')
assert len(r) > 1500
assert bull > 0 and bear > 0
print('OK')
"
```

Expected: total > 1500, bull > 0, bear > 0.

- [ ] **Step 5: /simplify 一轮**

- [ ] **Step 6: Commit**

```bash
git add backtest/regime_switch_backtest.py
git commit -m "feat(ng2.0a): regime_switch_backtest --regime-version {v1, v2} dispatch"
```

---

## Task 8: Step B 端到端回测 (2020-2026 + Pre-2020)

**Files (read/write):**
- Read: existing `reports/daily_selection_ng107/` (bull) and `reports/daily_selection_ng104_3s/` (bear)
- Write: `reports/daily_selection_regime_switch_v2/` (merged)
- Write: `reports/ng2_0a_step_b_eval.md`

- [ ] **Step 1: 确认现成 sub-model 报告目录存在**

```bash
ls -d reports/daily_selection_ng107* reports/daily_selection_ng104_3s* 2>&1 | head
```

If missing, regenerate them first:
```bash
python3 backtest/batch_generate_v395_reports.py --version ng1.0.7 --start 2018-01-01 --end 2026-04-25
python3 backtest/batch_generate_v395_reports.py --version ng1.0.4-3s --start 2018-01-01 --end 2026-04-25
```

(Skip this step if dirs already populated from prior ng106 v2 runs — check with `ls reports/daily_selection_ng107 | wc -l` should be > 1500.)

- [ ] **Step 2: 跑端到端 v2 merge**

```bash
mkdir -p logs
python3 backtest/regime_switch_backtest.py \
    --regime-version v2 \
    --bull-dir reports/daily_selection_ng107 \
    --bear-dir reports/daily_selection_ng104_3s \
    --out-dir reports/daily_selection_regime_switch_v2 \
    2>&1 | tee logs/ng2_0a_step_b_merge.log
```

Expected: prints 牛市X天 / 熊市Y天 of merged v2; reports written.

- [ ] **Step 3: 跑 north-star 评估 — WF-OOS 2020-2026**

```bash
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_regime_switch_v2 \
    --label ng2.0a-WF-OOS \
    --top-n 10 --focus-days 10 --rank-field composite \
    --start-date 2020-01-01 --end-date 2026-04-25 \
    2>&1 | tee logs/ng2_0a_step_b_wfoos.log
```

Expected: prints V5.2 score, Sharpe 10d, MaxDD, annualized return.

- [ ] **Step 4: 跑 north-star 评估 — Pre-2020 OOS 2018-2019**

```bash
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_regime_switch_v2 \
    --label ng2.0a-PRE-2020 \
    --top-n 10 --focus-days 10 --rank-field composite \
    --start-date 2018-01-01 --end-date 2019-12-31 \
    2>&1 | tee logs/ng2_0a_step_b_pre2020.log
```

Expected: prints V5.2, net annualized return, Sharpe.

- [ ] **Step 5: 计算 Top-10 与 ng106 v2 重合度**

Create `scripts/ng2_0a_top10_overlap.py`:

```python
"""Compute Top-10 daily overlap between ng2.0a (v2 merged) and ng106 v2 (v1 merged)."""
import json
from pathlib import Path

import pandas as pd

V2_DIR = Path('reports/daily_selection_regime_switch_v2')
V1_DIR = Path('reports/daily_selection_regime_switch')  # ng106 v2 (V1 merged)


def load_top10(report_dir: Path, date_compact: str) -> set:
    p = report_dir / f'选股报告_{date_compact}.json'
    if not p.exists():
        return set()
    with open(p) as f:
        data = json.load(f)
    stocks = data.get('all_stocks_with_scores', [])[:10]
    return {s.get('code') for s in stocks if s.get('code')}


def main():
    if not V1_DIR.exists() or not V2_DIR.exists():
        raise SystemExit(f'Missing dirs: V1={V1_DIR.exists()}, V2={V2_DIR.exists()}')

    files_v2 = sorted(V2_DIR.glob('选股报告_*.json'))
    overlaps = []
    for fp in files_v2:
        date_compact = fp.stem.replace('选股报告_', '')
        # Restrict to 2020-2026 main window
        if date_compact < '20200101' or date_compact > '20260425':
            continue
        s_v1 = load_top10(V1_DIR, date_compact)
        s_v2 = load_top10(V2_DIR, date_compact)
        if not s_v1 or not s_v2:
            continue
        overlap_pct = 100.0 * len(s_v1 & s_v2) / 10.0
        overlaps.append((date_compact, overlap_pct))

    if not overlaps:
        raise SystemExit('No comparable dates found')

    df = pd.DataFrame(overlaps, columns=['date', 'overlap_pct'])
    avg = df['overlap_pct'].mean()
    median = df['overlap_pct'].median()
    pct_below_50 = 100.0 * (df['overlap_pct'] < 50).sum() / len(df)
    print(f'Top-10 overlap (ng2.0a v2 vs ng106 v2 v1):')
    print(f'  comparable dates: {len(df)}')
    print(f'  mean overlap: {avg:.2f}%')
    print(f'  median overlap: {median:.2f}%')
    print(f'  % days with overlap < 50%: {pct_below_50:.2f}%')
    if avg < 50:
        print('  ❌ ABORT: avg overlap < 50% (list drift)')
        raise SystemExit(1)
    print('  ✅ Top-10 overlap gate PASS')


if __name__ == '__main__':
    main()
```

Run:

```bash
python3 scripts/ng2_0a_top10_overlap.py 2>&1 | tee logs/ng2_0a_top10_overlap.log
```

Expected: avg ≥ 50%.

- [ ] **Step 6: 写 Step B 评估报告 + 决策**

Create `reports/ng2_0a_step_b_eval.md` summarizing:

```markdown
# ng2.0a Step B Evaluation Report (2026-04-25)

**Status:** [PASS / ABORT]
**Baseline:** ng106 v2 (V11 + ng1.0.7 bull + ng104-3s bear)

## Results vs gates

| Gate | Baseline | ng2.0a | PASS? |
|---|---:|---:|---|
| V5.2 (10d, WF-OOS 2020-2026) | 79% | XX% | ✅/❌ |
| 10d Sharpe | 2.808 | X.XXX | ✅/❌ |
| MaxDD | -22.9% | -XX.X% | ✅/❌ |
| Pre-2020 净年化 | TBD-baseline | +X.X% | ✅/❌ |
| Pre-2020 V5.2 | TBD-baseline | XX% | ✅/❌ |
| Top-10 overlap | 100% | XX% | ✅/❌ |

(填入数字)

## Decision

[PASS → 生产候选 ng2.0a, 写 wiki/CLAUDE.md, 触发 ng2.0b plan]
[ABORT → 终止 ng2.0a, 维持 ng106 v2 生产, 写 lessons learned]

## Hand-off

[If PASS] → Trigger ng2.0b implementation plan (sample-weighted sub-model retrain).
[If ABORT] → Document failure mode in MEMORY.md.
```

Fill in actuals from prior log files. Show user, get approval.

- [ ] **Step 7: Commit**

```bash
git add reports/ng2_0a_step_b_eval.md scripts/ng2_0a_top10_overlap.py reports/daily_selection_regime_switch_v2/ logs/ng2_0a_step_b*.log logs/ng2_0a_top10_overlap.log
git commit -m "feat(ng2.0a): Step B end-to-end eval (V5.2/Sharpe/MaxDD/Pre-2020/Top-10 overlap)"
```

---

## Task 9: 文档更新 (Wiki + CLAUDE.md + MEMORY)

This task runs in two flavors based on Step B outcome.

**If Step B PASSES:**

- [ ] **Step 1: Update CLAUDE.md ML Scoring Systems section**

Add ng2.0a entry to `## ML Scoring Systems (活跃版本)` listing in CLAUDE.md, between ng1.0.6 and ng1.0.1 entries:

```markdown
1. **🆕 NG v2.0a (multi-beta regime + ng106 v2 sub-model)** (2026-04-25, Step B PASSED):
   - 核心: V11 + B1 市场广度 + B2 已实现波动率 hard-vote regime, sub-model 沿用 ng106 v2 (ng1.0.7 bull / ng104-3s bear)
   - 性能 (WF-OOS 2020-2026): V5.2=XX% A+, 10d Sharpe=X.XXX, MaxDD=-XX%, Pre-2020 净年化=+X%
   - vs ng106 v2 baseline: ΔV5.2=+Xpp, ΔMaxDD=-Xpp, Top-10 overlap=XX%
   - 控制: regime 表 `market_regime_signals.regime_v2`, 选股 `--scoring-version ng2.0a`
   - 后续: ng2.0b sample-weighted sub-model retrain (待 plan)
```

(Fill X with actuals.)

- [ ] **Step 2: Update Wiki**

Edit `docs/wiki/models/ng-factor-quality.md` to add ng2.0a row to ML version comparison table.
Create `docs/wiki/models/ng2_0a_multi_beta_regime.md` with:
  - Architecture diagram
  - B1/B2 signal definitions
  - Step A + Step B results summary
  - Link to spec + plan

- [ ] **Step 3: Update MEMORY**

Append to `~/.claude/projects/-Users-yangxu-StockTradebyZ/memory/MEMORY.md`:

```markdown
## ✅ ng2.0a multi-beta regime PASS (2026-04-25) → 详见 `ng20a_multi_beta_regime.md`
- V11 + B1 (市场广度 % above MA20/MA60) + B2 (沪深300 60d RV percentile) hard-vote + 3d streak
- Step B 端到端 vs ng106 v2: V5.2 [PASS], MaxDD [PASS], Pre-2020 [PASS], Top-10 overlap XX%
- 生产: `--scoring-version ng2.0a`. ng2.0b 重训 plan 触发
```

Create memory detail file `~/.claude/projects/-Users-yangxu-StockTradebyZ/memory/ng20a_multi_beta_regime.md`.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md docs/wiki/
git commit -m "docs(ng2.0a): Step B PASS — production candidate, ng2.0b retrain triggered"
```

**If Step B ABORTS (record lesson, don't pollute production docs):**

- [ ] **Step 1: Record failure in MEMORY**

Append concise lesson to `MEMORY.md`:

```markdown
## ❌ ng2.0a REJECTED (2026-04-25) → 详见 `ng20a_rejected.md`
- Step B [V5.2 / MaxDD / Pre-2020 / Top-10] gate fail
- 生产保持 ng106 v2. infra (B1/B2/regime_v2 表) 永久保留
```

Detail file `ng20a_rejected.md` with failure mode + lessons.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/ docs/superpowers/plans/ reports/ng2_0a_step_b_eval.md
git commit -m "docs(ng2.0a): REJECTED at Step B — keep ng106 v2 production, infra preserved"
```

---

## Self-review checklist (run after Task 9 completion)

- [ ] Spec section 0-13 every requirement has a task implementing or validating it
- [ ] Step A gates (spec 6.1) coded in `compare_regime_v1_v2.py` ABORT logic
- [ ] Step B gates (spec 6.2) coded in `ng2_0a_top10_overlap.py` + `step_b_eval.md`
- [ ] No new feature schema changes (per spec Pre-flight Check 1)
- [ ] No future leakage in B1/B2 (per spec Pre-flight Check 7)
- [ ] All new modules pass /simplify
- [ ] All commits follow `feat(ng2.0a): ...` / `docs(ng2.0a): ...` convention
- [ ] All run logs saved to `logs/ng2_0a_*.log`
- [ ] All evaluation reports saved to `reports/ng2_0a_*.md` or `reports/daily_selection_regime_switch_v2/`
