# Daily Selection NG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new factor-based stock selection model ("NG") with 62 economically-motivated factors organized around right-side trend trading, fully integrated into the existing training and evaluation pipeline.

**Architecture:** A new `ml_models/ng/` module contains the feature calculator, trainer, and scorer. A new `ng_feature_cache` SQLite table stores precomputed features. The NG trainer inherits from V485Trainer to reuse WF, ensemble, and LambdaRank machinery. The batch report generator and north star eval are extended to support the `ng` version.

**Tech Stack:** Python 3, SQLite, LightGBM, XGBoost, CatBoost, scikit-learn, numpy, pandas

---

### Task 1: Create `ng_feature_cache` database table

**Files:**
- Create: `ml_models/ng/__init__.py`
- Create: `ml_models/ng/ng_schema.py`

- [ ] **Step 1: Create the ng module directory and init**

```python
# ml_models/ng/__init__.py
"""Daily Selection NG — Next Generation trend-following factor model."""
```

- [ ] **Step 2: Write schema creation script**

```python
# ml_models/ng/ng_schema.py
"""Create and manage the ng_feature_cache table."""
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data_adapter', 'stock_data.db')

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ng_feature_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code VARCHAR(10) NOT NULL,
    trade_date DATE NOT NULL,
    features_json TEXT NOT NULL,
    label_3d REAL,
    label_5d REAL,
    label_10d REAL,
    -- market features stored as columns for fast filtering
    market_return_5d REAL,
    market_return_20d REAL,
    market_volatility_20d REAL,
    market_breadth REAL,
    market_new_high_ratio REAL,
    northbound_flow_5d REAL,
    market_volume_ratio REAL,
    market_drawdown REAL,
    vix_proxy REAL,
    market_momentum_diff REAL,
    UNIQUE(code, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_ng_fc_date ON ng_feature_cache(trade_date);
CREATE INDEX IF NOT EXISTS idx_ng_fc_code_date ON ng_feature_cache(code, trade_date);
"""


def create_table(db_path: str = None):
    """Create ng_feature_cache table if not exists."""
    path = db_path or DB_PATH
    with sqlite3.connect(path, timeout=30) as conn:
        conn.executescript(SCHEMA_SQL)
    print(f"ng_feature_cache table ready: {path}")


if __name__ == '__main__':
    create_table()
```

- [ ] **Step 3: Run to create table**

Run: `python3 ml_models/ng/ng_schema.py`
Expected: `ng_feature_cache table ready: data_adapter/stock_data.db`

- [ ] **Step 4: Verify table exists**

Run: `python3 -c "import sqlite3; c=sqlite3.connect('data_adapter/stock_data.db'); print([r[0] for r in c.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name='ng_feature_cache'\").fetchall()])"`
Expected: `['ng_feature_cache']`

- [ ] **Step 5: Commit**

```bash
git add ml_models/ng/
git commit -m "feat(ng): create ng_feature_cache table schema"
```

---

### Task 2: Implement stock-level feature calculator (factors 1-30)

**Files:**
- Create: `ml_models/ng/ng_feature_calculator.py`

This is the core file. It computes all 62 factors for a single stock on a single date, given preloaded price/volume/technical data. We split implementation into two tasks: stock-level factors (1-30) and fundamental+market+industry factors (31-62).

- [ ] **Step 1: Write the feature calculator class with factors 1-30**

```python
# ml_models/ng/ng_feature_calculator.py
"""NG Feature Calculator — 62 economically-motivated factors for trend trading.

Factor groups:
  1-12:  Trend state         — "Is this stock in an uptrend?"
  13-22: Pullback entry      — "Is now a good pullback buy point?"
  23-30: Volume confirmation — "Is smart money entering?"
  31-44: Fundamental quality — "Is this stock worth buying?"
  45-54: Market environment  — "Does the market support longs?"
  55-62: Industry rotation   — "Is this sector in play?"
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional


def compute_stock_features(
    closes: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    amounts: np.ndarray,
    ma5: np.ndarray,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    atr_14: float,
    macd_macd: np.ndarray,
    kdj_j: float,
    boll_upper: float,
    boll_lower: float,
    rsi_12: float,
    rsi_24: float,
) -> Dict[str, float]:
    """Compute factors 1-30 from price/volume/technical arrays.

    All array inputs are ordered oldest-first, length >= 60.
    The last element is "today".

    Returns dict of {factor_name: value}. Missing values are np.nan.
    """
    n = len(closes)
    if n < 60:
        return {}

    c = closes[-1]       # today's close
    features = {}

    # ── Group 1: Trend State (12 factors) ──

    # 1. price_above_ma20
    features['price_above_ma20'] = c / ma20[-1] - 1 if ma20[-1] > 0 else np.nan

    # 2. price_above_ma60
    features['price_above_ma60'] = c / ma60[-1] - 1 if ma60[-1] > 0 else np.nan

    # 3. ma_alignment — degree of bullish alignment
    pairs = [
        (ma5[-1] > ma10[-1]),
        (ma10[-1] > ma20[-1]),
        (ma20[-1] > ma60[-1]),
    ]
    alignment_count = sum(pairs) / 3.0
    spread = (ma5[-1] - ma60[-1]) / c if c > 0 else 0
    features['ma_alignment'] = alignment_count * spread

    # 4. trend_strength_20d — linear regression slope / std
    if n >= 20:
        window = closes[-20:]
        x = np.arange(20, dtype=float)
        slope = np.polyfit(x, window, 1)[0]
        std = np.std(window)
        features['trend_strength_20d'] = slope / std if std > 1e-8 else 0.0
    else:
        features['trend_strength_20d'] = np.nan

    # 5. new_high_20d
    high_20d = np.max(highs[-20:])
    features['new_high_20d'] = c / high_20d if high_20d > 0 else np.nan

    # 6. new_high_60d
    high_60d = np.max(highs[-60:])
    features['new_high_60d'] = c / high_60d if high_60d > 0 else np.nan

    # 7. days_since_breakout — days since close exceeded prior 20d high
    prev_high_20d = np.max(highs[-21:-1]) if n >= 21 else np.nan
    if not np.isnan(prev_high_20d) and c > prev_high_20d:
        # count backwards: how many consecutive days close > prev rolling 20d high
        days = 0
        for i in range(n - 1, max(n - 61, -1), -1):
            rolling_high = np.max(highs[max(0, i - 20):i]) if i > 0 else highs[0]
            if closes[i] > rolling_high:
                days += 1
            else:
                break
        features['days_since_breakout'] = float(min(days, 60))
    else:
        features['days_since_breakout'] = 0.0

    # 8. adx_proxy — |ma5 - ma20| / atr
    features['adx_proxy'] = abs(ma5[-1] - ma20[-1]) / atr_14 if atr_14 > 1e-8 else 0.0

    # 9. macd_histogram
    features['macd_histogram'] = float(macd_macd[-1])

    # 10. macd_acceleration — 5d change in MACD histogram
    if len(macd_macd) >= 6:
        features['macd_acceleration'] = float(macd_macd[-1] - macd_macd[-6])
    else:
        features['macd_acceleration'] = np.nan

    # 11. price_channel_position
    low_20d = np.min(lows[-20:])
    channel_range = high_20d - low_20d
    features['price_channel_position'] = (c - low_20d) / channel_range if channel_range > 1e-8 else 0.5

    # 12. cumulative_return_60d
    features['cumulative_return_60d'] = c / closes[-60] - 1 if closes[-60] > 0 else np.nan

    # ── Group 2: Pullback Entry (10 factors) ──

    # 13. pullback_from_high — drawdown from 5d high
    high_5d = np.max(closes[-5:])
    features['pullback_from_high'] = 1.0 - c / high_5d if high_5d > 0 else 0.0

    # 14. pullback_to_ma10
    features['pullback_to_ma10'] = c / ma10[-1] - 1 if ma10[-1] > 0 else np.nan

    # 15. pullback_to_ma20
    features['pullback_to_ma20'] = c / ma20[-1] - 1 if ma20[-1] > 0 else np.nan

    # 16. rsi_14 — approximate from rsi_12 and rsi_24
    features['rsi_14'] = 0.6 * rsi_12 + 0.4 * rsi_24 if not (np.isnan(rsi_12) or np.isnan(rsi_24)) else np.nan

    # 17. kdj_j_value
    features['kdj_j_value'] = float(kdj_j)

    # 18. volume_contraction — 5d avg vol / 20d avg vol
    vol_5d = np.mean(volumes[-5:])
    vol_20d = np.mean(volumes[-20:])
    features['volume_contraction'] = vol_5d / vol_20d if vol_20d > 0 else np.nan

    # 19. lower_shadow_ratio
    hl_range = highs[-1] - lows[-1]
    features['lower_shadow_ratio'] = (c - lows[-1]) / hl_range if hl_range > 1e-8 else 0.5

    # 20. consecutive_down_days
    down_days = 0
    for i in range(n - 1, max(n - 21, 0), -1):
        if closes[i] < closes[i - 1]:
            down_days += 1
        else:
            break
    features['consecutive_down_days'] = float(down_days)

    # 21. bollinger_position
    boll_range = boll_upper - boll_lower
    features['bollinger_position'] = (c - boll_lower) / boll_range if boll_range > 1e-8 else 0.5

    # 22. intraday_recovery — 5d average of (close-low)/(high-low)
    recoveries = []
    for i in range(-5, 0):
        hl = highs[i] - lows[i]
        if hl > 1e-8:
            recoveries.append((closes[i] - lows[i]) / hl)
    features['intraday_recovery'] = float(np.mean(recoveries)) if recoveries else 0.5

    # ── Group 3: Volume Confirmation (8 factors) ──

    # 23. volume_ratio_5d
    features['volume_ratio_5d'] = vol_5d / vol_20d if vol_20d > 0 else np.nan

    # 24. volume_price_corr — 20d correlation of close and volume
    if n >= 20:
        c20 = closes[-20:]
        v20 = volumes[-20:].astype(float)
        if np.std(c20) > 1e-8 and np.std(v20) > 1e-8:
            features['volume_price_corr'] = float(np.corrcoef(c20, v20)[0, 1])
        else:
            features['volume_price_corr'] = 0.0
    else:
        features['volume_price_corr'] = np.nan

    # 25. obv_trend — OBV 20d linear regression slope (normalized)
    if n >= 20:
        signs = np.sign(np.diff(closes[-21:]))
        obv = np.cumsum(signs * volumes[-20:])
        x = np.arange(20, dtype=float)
        slope = np.polyfit(x, obv, 1)[0]
        features['obv_trend'] = slope / (np.mean(volumes[-20:]) + 1e-8)
    else:
        features['obv_trend'] = np.nan

    # 26. volume_breakout — today's volume / 20d avg
    features['volume_breakout'] = float(volumes[-1]) / vol_20d if vol_20d > 0 else np.nan

    # 27. log_amount_ma5
    amt_5d = np.mean(amounts[-5:])
    features['log_amount_ma5'] = float(np.log(amt_5d + 1))

    # 28. turnover_rate — passed in from daily_basic, set externally
    features['turnover_rate'] = np.nan  # placeholder, set by caller

    # 29. up_volume_ratio — fraction of 20d volume on up days
    if n >= 20:
        up_mask = closes[-20:] > opens[-20:]
        total_vol = np.sum(volumes[-20:])
        up_vol = np.sum(volumes[-20:][up_mask])
        features['up_volume_ratio'] = up_vol / total_vol if total_vol > 0 else 0.5
    else:
        features['up_volume_ratio'] = np.nan

    # 30. volume_cv — coefficient of variation of 20d volume
    if n >= 20:
        v_std = np.std(volumes[-20:])
        features['volume_cv'] = v_std / vol_20d if vol_20d > 0 else np.nan
    else:
        features['volume_cv'] = np.nan

    return features
```

- [ ] **Step 2: Write a quick smoke test**

```python
# Test with synthetic data
python3 -c "
import numpy as np
from ml_models.ng.ng_feature_calculator import compute_stock_features

np.random.seed(42)
n = 80
closes = 10 + np.cumsum(np.random.randn(n) * 0.1)
opens = closes - np.random.rand(n) * 0.05
highs = closes + np.random.rand(n) * 0.2
lows = closes - np.random.rand(n) * 0.2
volumes = np.random.randint(1e6, 1e7, n).astype(float)
amounts = volumes * closes
ma5 = np.convolve(closes, np.ones(5)/5, 'same')
ma10 = np.convolve(closes, np.ones(10)/10, 'same')
ma20 = np.convolve(closes, np.ones(20)/20, 'same')
ma60 = np.convolve(closes, np.ones(60)/60, 'same')
macd = np.random.randn(n) * 0.01

f = compute_stock_features(
    closes, opens, highs, lows, volumes, amounts,
    ma5, ma10, ma20, ma60,
    atr_14=0.3, macd_macd=macd, kdj_j=45.0,
    boll_upper=closes[-1]+0.5, boll_lower=closes[-1]-0.5,
    rsi_12=55.0, rsi_24=50.0,
)
print(f'Factors computed: {len(f)}')
for k, v in sorted(f.items()):
    print(f'  {k}: {v:.4f}' if not np.isnan(v) else f'  {k}: NaN')
"
```

Expected: `Factors computed: 30` with reasonable values, no crashes.

- [ ] **Step 3: Commit**

```bash
git add ml_models/ng/ng_feature_calculator.py
git commit -m "feat(ng): implement stock-level factors 1-30 (trend+pullback+volume)"
```

---

### Task 3: Add fundamental, market, and industry factors (31-62)

**Files:**
- Modify: `ml_models/ng/ng_feature_calculator.py`

- [ ] **Step 1: Add fundamental factors function**

Add to `ng_feature_calculator.py`:

```python
def compute_fundamental_features(
    pe_ttm: float,
    pb: float,
    dv_ratio: float,
    circ_mv: float,
    free_share: float,
    total_share: float,
    turnover_rate: float,
    adv_20d: float,
    pe_ttm_history_60d: np.ndarray,
    roe: float,
    roe_prev_year: float,
    profit_to_gr: float,
    netprofit_margin: float,
    ocf_to_profit: float,
    debt_to_assets: float,
    current_ratio: float,
) -> Dict[str, float]:
    """Compute factors 31-44: fundamental quality."""
    features = {}

    # 31. roe_ttm
    features['roe_ttm'] = float(roe) if roe is not None and not np.isnan(roe) else np.nan

    # 32. roe_change
    if roe is not None and roe_prev_year is not None and not (np.isnan(roe) or np.isnan(roe_prev_year)):
        features['roe_change'] = float(roe - roe_prev_year)
    else:
        features['roe_change'] = np.nan

    # 33. revenue_growth (use profit_to_gr as proxy)
    features['revenue_growth'] = float(profit_to_gr) if profit_to_gr is not None and not np.isnan(profit_to_gr) else np.nan

    # 34. net_profit_margin
    features['net_profit_margin'] = float(netprofit_margin) if netprofit_margin is not None and not np.isnan(netprofit_margin) else np.nan

    # 35. ocf_quality
    features['ocf_quality'] = float(ocf_to_profit) if ocf_to_profit is not None and not np.isnan(ocf_to_profit) else np.nan

    # 36. pe_ttm
    features['pe_ttm'] = float(pe_ttm) if pe_ttm is not None and not np.isnan(pe_ttm) else np.nan

    # 37. pb
    features['pb'] = float(pb) if pb is not None and not np.isnan(pb) else np.nan

    # 38. pe_percentile_60d
    if pe_ttm_history_60d is not None and len(pe_ttm_history_60d) >= 10:
        valid = pe_ttm_history_60d[~np.isnan(pe_ttm_history_60d)]
        if len(valid) >= 10 and pe_ttm is not None and not np.isnan(pe_ttm):
            features['pe_percentile_60d'] = float(np.mean(valid <= pe_ttm))
        else:
            features['pe_percentile_60d'] = np.nan
    else:
        features['pe_percentile_60d'] = np.nan

    # 39. debt_to_assets
    features['debt_to_assets'] = float(debt_to_assets) if debt_to_assets is not None and not np.isnan(debt_to_assets) else np.nan

    # 40. current_ratio
    features['current_ratio'] = float(current_ratio) if current_ratio is not None and not np.isnan(current_ratio) else np.nan

    # 41. log_market_cap
    features['log_market_cap'] = float(np.log(circ_mv + 1)) if circ_mv is not None and circ_mv > 0 else np.nan

    # 42. log_adv_20d
    features['log_adv_20d'] = float(np.log(adv_20d + 1)) if adv_20d is not None and adv_20d > 0 else np.nan

    # 43. free_float_ratio
    if free_share is not None and total_share is not None and total_share > 0:
        features['free_float_ratio'] = float(free_share / total_share)
    else:
        features['free_float_ratio'] = np.nan

    # 44. dv_ratio
    features['dv_ratio'] = float(dv_ratio) if dv_ratio is not None and not np.isnan(dv_ratio) else np.nan

    # Also set turnover_rate (was placeholder in stock features)
    features['turnover_rate'] = float(turnover_rate) if turnover_rate is not None and not np.isnan(turnover_rate) else np.nan

    return features


def compute_market_features(
    benchmark_closes: np.ndarray,
    all_stock_returns: np.ndarray,
    all_stock_highs_20d_ratio: np.ndarray,
    total_market_amount: np.ndarray,
    northbound_net_buy_5d: float,
    northbound_std: float,
) -> Dict[str, float]:
    """Compute factors 45-54: market environment.

    Args:
        benchmark_closes: CSI300 close prices, length >= 60, oldest first
        all_stock_returns: 1d returns of all stocks today, shape (N,)
        all_stock_highs_20d_ratio: close/max(high,20d) for all stocks, shape (N,)
        total_market_amount: daily total market turnover, length >= 20
        northbound_net_buy_5d: sum of northbound net buy over 5 days (RMB)
        northbound_std: std of northbound daily net buy (for z-score)
    """
    features = {}
    bm = benchmark_closes
    n = len(bm)

    # 45. market_return_5d
    features['market_return_5d'] = bm[-1] / bm[-6] - 1 if n >= 6 else np.nan

    # 46. market_return_20d
    features['market_return_20d'] = bm[-1] / bm[-21] - 1 if n >= 21 else np.nan

    # 47. market_volatility_20d
    if n >= 21:
        log_rets = np.diff(np.log(bm[-21:]))
        features['market_volatility_20d'] = float(np.std(log_rets) * np.sqrt(252))
    else:
        features['market_volatility_20d'] = np.nan

    # 48. market_breadth
    if all_stock_returns is not None and len(all_stock_returns) > 0:
        features['market_breadth'] = float(np.mean(all_stock_returns > 0))
    else:
        features['market_breadth'] = np.nan

    # 49. market_new_high_ratio
    if all_stock_highs_20d_ratio is not None and len(all_stock_highs_20d_ratio) > 0:
        features['market_new_high_ratio'] = float(np.mean(all_stock_highs_20d_ratio > 0.98))
    else:
        features['market_new_high_ratio'] = np.nan

    # 50. northbound_flow_5d (z-score)
    if northbound_std > 0 and not np.isnan(northbound_net_buy_5d):
        features['northbound_flow_5d'] = northbound_net_buy_5d / northbound_std
    else:
        features['northbound_flow_5d'] = np.nan

    # 51. market_volume_ratio
    if total_market_amount is not None and len(total_market_amount) >= 20:
        avg_20 = np.mean(total_market_amount[-20:])
        features['market_volume_ratio'] = total_market_amount[-1] / avg_20 if avg_20 > 0 else np.nan
    else:
        features['market_volume_ratio'] = np.nan

    # 52. market_drawdown
    if n >= 60:
        peak_60d = np.max(bm[-60:])
        features['market_drawdown'] = bm[-1] / peak_60d - 1
    else:
        features['market_drawdown'] = np.nan

    # 53. vix_proxy
    if n >= 61:
        vol_20 = np.std(np.diff(np.log(bm[-21:]))) * np.sqrt(252)
        vol_60 = np.std(np.diff(np.log(bm[-61:]))) * np.sqrt(252)
        features['vix_proxy'] = vol_20 / vol_60 if vol_60 > 1e-8 else 1.0
    else:
        features['vix_proxy'] = np.nan

    # 54. market_momentum_diff
    r5 = features.get('market_return_5d', np.nan)
    r20 = features.get('market_return_20d', np.nan)
    features['market_momentum_diff'] = r5 - r20 if not (np.isnan(r5) or np.isnan(r20)) else np.nan

    return features


def compute_industry_features(
    stock_return_20d: float,
    industry_stock_returns_1d: np.ndarray,
    industry_stock_returns_5d: np.ndarray,
    industry_stock_returns_20d: np.ndarray,
    industry_amounts_5d: np.ndarray,
    industry_amounts_20d: np.ndarray,
    all_industry_returns_5d: np.ndarray,
    sw_index_return_5d: float,
) -> Dict[str, float]:
    """Compute factors 55-62: industry rotation.

    Args:
        stock_return_20d: this stock's 20d return
        industry_stock_returns_*d: arrays of returns for all stocks in the same industry
        industry_amounts_*d: arrays of total amount for the industry over different windows
        all_industry_returns_5d: array of 5d returns for all 31 SW L1 industries
        sw_index_return_5d: SW industry index 5d return for this stock's industry
    """
    features = {}

    # 55. industry_return_5d
    if industry_stock_returns_5d is not None and len(industry_stock_returns_5d) > 0:
        features['industry_return_5d'] = float(np.mean(industry_stock_returns_5d))
    else:
        features['industry_return_5d'] = np.nan

    # 56. industry_return_20d
    if industry_stock_returns_20d is not None and len(industry_stock_returns_20d) > 0:
        features['industry_return_20d'] = float(np.mean(industry_stock_returns_20d))
    else:
        features['industry_return_20d'] = np.nan

    # 57. industry_relative_strength
    ind_20d = features.get('industry_return_20d', np.nan)
    if not np.isnan(ind_20d) and not np.isnan(stock_return_20d):
        features['industry_relative_strength'] = stock_return_20d - ind_20d
    else:
        features['industry_relative_strength'] = np.nan

    # 58. industry_breadth
    if industry_stock_returns_1d is not None and len(industry_stock_returns_1d) > 0:
        features['industry_breadth'] = float(np.mean(industry_stock_returns_1d > 0))
    else:
        features['industry_breadth'] = np.nan

    # 59. industry_volume_change
    if (industry_amounts_5d is not None and industry_amounts_20d is not None
            and len(industry_amounts_5d) > 0 and len(industry_amounts_20d) > 0):
        avg_5 = np.mean(industry_amounts_5d)
        avg_20 = np.mean(industry_amounts_20d)
        features['industry_volume_change'] = avg_5 / avg_20 if avg_20 > 0 else np.nan
    else:
        features['industry_volume_change'] = np.nan

    # 60. industry_rank_return_5d (0-1 rank among all industries)
    ind_5d = features.get('industry_return_5d', np.nan)
    if all_industry_returns_5d is not None and len(all_industry_returns_5d) > 1 and not np.isnan(ind_5d):
        features['industry_rank_return_5d'] = float(np.mean(all_industry_returns_5d <= ind_5d))
    else:
        features['industry_rank_return_5d'] = np.nan

    # 61. sw_index_return_5d
    features['sw_index_return_5d'] = float(sw_index_return_5d) if not np.isnan(sw_index_return_5d) else np.nan

    # 62. industry_hhi
    if industry_stock_returns_1d is not None and len(industry_stock_returns_1d) > 1:
        abs_rets = np.abs(industry_stock_returns_1d)
        total = np.sum(abs_rets)
        if total > 1e-8:
            shares = abs_rets / total
            features['industry_hhi'] = float(np.sum(shares ** 2))
        else:
            features['industry_hhi'] = np.nan
    else:
        features['industry_hhi'] = np.nan

    return features
```

- [ ] **Step 2: Smoke test with synthetic data**

```bash
python3 -c "
import numpy as np
from ml_models.ng.ng_feature_calculator import compute_fundamental_features, compute_market_features, compute_industry_features

# Fundamental
f = compute_fundamental_features(
    pe_ttm=25.0, pb=3.5, dv_ratio=1.2, circ_mv=8e9, free_share=5e8,
    total_share=1e9, turnover_rate=3.5, adv_20d=1e8,
    pe_ttm_history_60d=np.random.uniform(20, 30, 60),
    roe=15.0, roe_prev_year=12.0, profit_to_gr=25.0, netprofit_margin=10.0,
    ocf_to_profit=0.8, debt_to_assets=45.0, current_ratio=1.5,
)
print(f'Fundamental: {len(f)} factors')

# Market
bm = 3800 + np.cumsum(np.random.randn(80) * 10)
m = compute_market_features(
    benchmark_closes=bm, all_stock_returns=np.random.randn(3000)*0.02,
    all_stock_highs_20d_ratio=np.random.uniform(0.85, 1.02, 3000),
    total_market_amount=np.random.uniform(8e11, 1.2e12, 80),
    northbound_net_buy_5d=5e9, northbound_std=3e9,
)
print(f'Market: {len(m)} factors')

# Industry
ind = compute_industry_features(
    stock_return_20d=0.05,
    industry_stock_returns_1d=np.random.randn(50)*0.02,
    industry_stock_returns_5d=np.random.randn(50)*0.03,
    industry_stock_returns_20d=np.random.randn(50)*0.06,
    industry_amounts_5d=np.random.uniform(1e9, 5e9, 5),
    industry_amounts_20d=np.random.uniform(1e9, 5e9, 20),
    all_industry_returns_5d=np.random.randn(31)*0.02,
    sw_index_return_5d=0.015,
)
print(f'Industry: {len(ind)} factors')
print(f'Total: {len(f) + len(m) + len(ind)} factors')
"
```

Expected: `Fundamental: 15 factors`, `Market: 10 factors`, `Industry: 8 factors`, `Total: 33 factors`. Combined with Task 2's 30 = 63 total (62 unique + turnover_rate appears in both).

- [ ] **Step 3: Commit**

```bash
git add ml_models/ng/ng_feature_calculator.py
git commit -m "feat(ng): add fundamental(31-44), market(45-54), industry(55-62) factors"
```

---

### Task 4: Build the batch feature cache updater

**Files:**
- Create: `ml_models/ng/ng_cache_updater.py`

This file orchestrates: load raw data from DB → compute all 62 factors per stock per date → write to `ng_feature_cache`.

- [ ] **Step 1: Write the cache updater**

```python
# ml_models/ng/ng_cache_updater.py
"""Batch updater for ng_feature_cache table.

Usage:
    # Update a single date
    python3 ml_models/ng/ng_cache_updater.py --date 2025-01-15

    # Backfill a date range
    python3 ml_models/ng/ng_cache_updater.py --start-date 2020-01-01 --end-date 2025-12-31
"""
import argparse
import json
import sqlite3
import sys
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml_models.ng.ng_feature_calculator import (
    compute_stock_features, compute_fundamental_features,
    compute_market_features, compute_industry_features,
)
from ml_models.ng.ng_schema import create_table

DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')


class NGCacheUpdater:
    """Precompute and cache NG features for all stocks on given dates."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        create_table(self.db_path)

    def get_trading_dates(self, start_date: str, end_date: str) -> list:
        """Get trading dates from daily_quotes."""
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            rows = conn.execute(
                "SELECT DISTINCT trade_date FROM daily_quotes "
                "WHERE trade_date BETWEEN ? AND ? ORDER BY trade_date",
                (start_date, end_date)
            ).fetchall()
        return [r[0] for r in rows]

    def update_single_date(self, date: str) -> int:
        """Compute and cache features for all stocks on a single date.

        Returns number of stocks processed.
        """
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA busy_timeout = 30000")

        try:
            # 1. Load all stock data for this date (+ lookback)
            lookback_start = self._offset_date(date, -90)
            stock_data = self._load_stock_data(conn, lookback_start, date)
            if stock_data.empty:
                return 0

            # 2. Load technical indicators for this date
            tech_data = self._load_tech_data(conn, date)

            # 3. Load daily_basic for this date
            basic_data = self._load_basic_data(conn, date)

            # 4. Load financial indicators (latest available)
            fin_data = self._load_financial_data(conn, date)

            # 5. Compute market-level features (once per date)
            market_features = self._compute_market_features_for_date(conn, stock_data, date)

            # 6. Compute industry-level data
            industry_data = self._compute_industry_data(stock_data, date, conn)

            # 7. Get labels (future returns)
            labels = self._compute_labels(conn, date)

            # 8. For each stock, compute all factors
            today_stocks = stock_data[stock_data['trade_date'] == date]['code'].unique()
            rows = []
            for code in today_stocks:
                stock_df = stock_data[stock_data['code'] == code].sort_values('trade_date')
                if len(stock_df) < 60:
                    continue

                # Filter: market cap >= 50bn (50亿)
                mc = basic_data.get(code, {}).get('circ_mv', 0) or 0
                if mc < 5e9:  # circ_mv is in RMB (元), 50亿=5e9
                    continue

                # Stock-level features (1-30)
                tech = tech_data.get(code, {})
                sf = compute_stock_features(
                    closes=stock_df['close'].values,
                    opens=stock_df['open'].values,
                    highs=stock_df['high'].values,
                    lows=stock_df['low'].values,
                    volumes=stock_df['volume'].values,
                    amounts=stock_df['amount'].values,
                    ma5=stock_df['ma5'].values,
                    ma10=stock_df['ma10'].values,
                    ma20=stock_df['ma20'].values,
                    ma60=stock_df['ma60'].values,
                    atr_14=tech.get('atr_14', 0.0),
                    macd_macd=stock_df['macd_macd'].values if 'macd_macd' in stock_df else np.zeros(len(stock_df)),
                    kdj_j=tech.get('kdj_j', 50.0),
                    boll_upper=tech.get('boll_upper', stock_df['close'].values[-1] + 1),
                    boll_lower=tech.get('boll_lower', stock_df['close'].values[-1] - 1),
                    rsi_12=tech.get('rsi12', 50.0),
                    rsi_24=tech.get('rsi24', 50.0),
                )
                if not sf:
                    continue

                # Fundamental features (31-44)
                bd = basic_data.get(code, {})
                fd = fin_data.get(code, {})
                pe_hist = self._get_pe_history_60d(conn, code, date)
                adv_20d = float(np.mean(stock_df['amount'].values[-20:])) if len(stock_df) >= 20 else 0

                ff = compute_fundamental_features(
                    pe_ttm=bd.get('pe_ttm'), pb=bd.get('pb'),
                    dv_ratio=bd.get('dv_ratio', bd.get('dv_ttm')),
                    circ_mv=bd.get('circ_mv'), free_share=bd.get('free_share'),
                    total_share=bd.get('total_share'), turnover_rate=bd.get('turnover_rate'),
                    adv_20d=adv_20d, pe_ttm_history_60d=pe_hist,
                    roe=fd.get('roe'), roe_prev_year=fd.get('roe_prev_year'),
                    profit_to_gr=fd.get('profit_to_gr'),
                    netprofit_margin=fd.get('netprofit_margin'),
                    ocf_to_profit=fd.get('ocf_to_profit'),
                    debt_to_assets=fd.get('debt_to_assets'),
                    current_ratio=fd.get('current_ratio'),
                )

                # Industry features (55-62)
                ind_code = self._get_industry(conn, code)
                ifd = industry_data.get(ind_code, {})
                stock_ret_20d = sf.get('cumulative_return_60d', np.nan)  # use 20d approx
                if len(stock_df) >= 20:
                    stock_ret_20d = stock_df['close'].values[-1] / stock_df['close'].values[-20] - 1

                inf = compute_industry_features(
                    stock_return_20d=stock_ret_20d,
                    industry_stock_returns_1d=ifd.get('returns_1d'),
                    industry_stock_returns_5d=ifd.get('returns_5d'),
                    industry_stock_returns_20d=ifd.get('returns_20d'),
                    industry_amounts_5d=ifd.get('amounts_5d'),
                    industry_amounts_20d=ifd.get('amounts_20d'),
                    all_industry_returns_5d=industry_data.get('__all_5d__', np.array([])),
                    sw_index_return_5d=ifd.get('sw_index_return_5d', np.nan),
                )

                # Merge all + market features
                all_features = {**sf, **ff, **inf}
                # Override turnover_rate from fundamental (sf had placeholder)
                all_features['turnover_rate'] = ff.get('turnover_rate', np.nan)

                stock_labels = labels.get(code, {})
                row = {
                    'code': code,
                    'trade_date': date,
                    'features_json': json.dumps(all_features, default=_json_default),
                    'label_3d': stock_labels.get('label_3d'),
                    'label_5d': stock_labels.get('label_5d'),
                    'label_10d': stock_labels.get('label_10d'),
                    **{k: v for k, v in market_features.items()},
                }
                rows.append(row)

            # 9. Write to database
            if rows:
                self._write_cache(conn, rows)

            return len(rows)

        finally:
            conn.close()

    def _offset_date(self, date: str, days: int) -> str:
        """Offset a date string by N calendar days."""
        from datetime import datetime, timedelta
        dt = datetime.strptime(date, '%Y-%m-%d')
        return (dt + timedelta(days=days)).strftime('%Y-%m-%d')

    def _load_stock_data(self, conn, start_date, end_date) -> pd.DataFrame:
        """Load OHLCV + MA data for all A-stocks in date range."""
        sql = """
        SELECT s.code, q.trade_date, q.open, q.high, q.low, q.close,
               q.volume, q.amount, q.ma5, q.ma10, q.ma20, q.ma60,
               q.price_change_pct, q.is_limit_up, q.is_limit_down, q.is_st
        FROM daily_quotes q
        JOIN securities s ON q.security_id = s.id
        WHERE s.type = 'A股' AND q.trade_date BETWEEN ? AND ?
          AND q.volume > 0
        ORDER BY s.code, q.trade_date
        """
        return pd.read_sql(sql, conn, params=(start_date, end_date))

    def _load_tech_data(self, conn, date) -> dict:
        """Load technical indicators for a single date, keyed by code."""
        sql = """
        SELECT s.code, t.kdj_j, t.macd_macd, t.rsi12, t.rsi24,
               t.boll_upper, t.boll_lower, t.atr_14
        FROM technical_indicators t
        JOIN securities s ON t.security_id = s.id
        WHERE t.trade_date = ?
        """
        df = pd.read_sql(sql, conn, params=(date,))
        return {row['code']: row.to_dict() for _, row in df.iterrows()}

    def _load_basic_data(self, conn, date) -> dict:
        """Load daily_basic for a single date, keyed by code."""
        sql = """
        SELECT s.code, b.pe_ttm, b.pb, b.ps_ttm, b.turnover_rate,
               b.total_mv, b.circ_mv, b.dv_ratio, b.dv_ttm,
               b.total_share, b.float_share, b.free_share
        FROM daily_basic b
        JOIN securities s ON b.security_id = s.id
        WHERE b.trade_date = ?
        """
        df = pd.read_sql(sql, conn, params=(date,))
        return {row['code']: row.to_dict() for _, row in df.iterrows()}

    def _load_financial_data(self, conn, date) -> dict:
        """Load latest financial indicators for each stock as of date."""
        sql = """
        SELECT s.code, f.roe, f.profit_to_gr, f.netprofit_margin,
               f.ocf_to_profit, f.debt_to_assets, f.current_ratio
        FROM financial_indicator f
        JOIN securities s ON f.security_id = s.id
        WHERE f.ann_date <= ? AND f.ann_date >= ?
        ORDER BY f.ann_date DESC
        """
        # Look back 1 year for latest filing
        start = self._offset_date(date, -400)
        df = pd.read_sql(sql, conn, params=(date, start))
        result = {}
        for _, row in df.iterrows():
            code = row['code']
            if code not in result:
                result[code] = row.to_dict()
        return result

    def _compute_market_features_for_date(self, conn, stock_data, date) -> dict:
        """Compute the 10 market-environment features for a date."""
        # Load CSI300 benchmark
        bm_sql = """
        SELECT q.close FROM daily_quotes q
        JOIN securities s ON q.security_id = s.id
        WHERE s.code = '000300.SH' AND q.trade_date <= ?
        ORDER BY q.trade_date DESC LIMIT 80
        """
        bm_rows = conn.execute(bm_sql, (date,)).fetchall()
        bm_closes = np.array([r[0] for r in reversed(bm_rows)]) if bm_rows else np.array([])

        # Today's stock returns
        today = stock_data[stock_data['trade_date'] == date]
        returns_1d = today['price_change_pct'].values if 'price_change_pct' in today else np.array([])

        # Stocks near 20d high (approximate)
        highs_ratio = np.array([])  # simplified; full impl uses rolling max

        # Total market amount
        amt_sql = """
        SELECT q.trade_date, SUM(q.amount) as total_amount
        FROM daily_quotes q
        JOIN securities s ON q.security_id = s.id
        WHERE s.type = 'A股' AND q.trade_date <= ? AND q.trade_date >= ?
        GROUP BY q.trade_date ORDER BY q.trade_date
        """
        start_30 = self._offset_date(date, -45)
        amt_rows = conn.execute(amt_sql, (date, start_30)).fetchall()
        total_amounts = np.array([r[1] for r in amt_rows]) if amt_rows else np.array([])

        # Northbound flow
        nb_sql = """
        SELECT net_buy FROM hsgt_daily WHERE trade_date <= ?
        ORDER BY trade_date DESC LIMIT 60
        """
        nb_rows = conn.execute(nb_sql, (date,)).fetchall()
        nb_values = [r[0] for r in reversed(nb_rows)] if nb_rows else []
        nb_5d = sum(nb_values[-5:]) if len(nb_values) >= 5 else np.nan
        nb_std = float(np.std(nb_values)) if len(nb_values) >= 20 else 1.0

        mf = compute_market_features(
            benchmark_closes=bm_closes,
            all_stock_returns=returns_1d,
            all_stock_highs_20d_ratio=highs_ratio,
            total_market_amount=total_amounts,
            northbound_net_buy_5d=nb_5d,
            northbound_std=nb_std,
        )
        return mf

    def _compute_industry_data(self, stock_data, date, conn) -> dict:
        """Precompute industry-level aggregates for all industries."""
        # Load industry mapping
        ind_sql = "SELECT code, industry FROM securities WHERE type = 'A股' AND industry IS NOT NULL"
        ind_map = dict(conn.execute(ind_sql).fetchall())

        today = stock_data[stock_data['trade_date'] == date].copy()
        today['industry'] = today['code'].map(ind_map)

        result = {}
        all_ind_5d = []

        for ind, group in today.groupby('industry'):
            codes = group['code'].values
            # 1d returns
            returns_1d = group['price_change_pct'].values if 'price_change_pct' in group else np.array([])

            # 5d/20d returns from stock_data lookback
            returns_5d = []
            returns_20d = []
            for code in codes:
                sdf = stock_data[stock_data['code'] == code].sort_values('trade_date')
                if len(sdf) >= 5:
                    returns_5d.append(sdf['close'].values[-1] / sdf['close'].values[-5] - 1)
                if len(sdf) >= 20:
                    returns_20d.append(sdf['close'].values[-1] / sdf['close'].values[-20] - 1)

            avg_5d = float(np.mean(returns_5d)) if returns_5d else np.nan
            all_ind_5d.append(avg_5d)

            result[ind] = {
                'returns_1d': returns_1d,
                'returns_5d': np.array(returns_5d),
                'returns_20d': np.array(returns_20d),
                'amounts_5d': None,  # simplified
                'amounts_20d': None,
                'sw_index_return_5d': np.nan,  # TODO: load from SW index
            }

        result['__all_5d__'] = np.array(all_ind_5d)
        return result

    def _compute_labels(self, conn, date) -> dict:
        """Compute future return labels for all stocks on date."""
        # Get next N trading dates
        future_sql = """
        SELECT DISTINCT trade_date FROM daily_quotes
        WHERE trade_date > ? ORDER BY trade_date LIMIT 16
        """
        future_dates = [r[0] for r in conn.execute(future_sql, (date,)).fetchall()]
        if len(future_dates) < 11:
            return {}

        t1 = future_dates[0]   # T+1
        t3 = future_dates[2] if len(future_dates) > 2 else None
        t5 = future_dates[4] if len(future_dates) > 4 else None
        t10 = future_dates[9] if len(future_dates) > 9 else None

        # Load opens at T+1 and closes at T+3/5/10
        labels = {}
        prices_sql = """
        SELECT s.code, q.trade_date, q.open, q.close
        FROM daily_quotes q JOIN securities s ON q.security_id = s.id
        WHERE s.type = 'A股' AND q.trade_date IN ({})
        """.format(','.join('?' * len(set([t1] + [d for d in [t3, t5, t10] if d]))))
        dates_needed = list(set([t1] + [d for d in [t3, t5, t10] if d]))
        price_df = pd.read_sql(prices_sql, conn, params=dates_needed)

        for code, grp in price_df.groupby('code'):
            grp = grp.set_index('trade_date')
            if t1 not in grp.index:
                continue
            base = grp.loc[t1, 'open']
            if base <= 0:
                continue
            l = {}
            if t3 and t3 in grp.index:
                l['label_3d'] = float(grp.loc[t3, 'close'] / base - 1)
            if t5 and t5 in grp.index:
                l['label_5d'] = float(grp.loc[t5, 'close'] / base - 1)
            if t10 and t10 in grp.index:
                l['label_10d'] = float(grp.loc[t10, 'close'] / base - 1)
            labels[code] = l

        return labels

    def _get_pe_history_60d(self, conn, code, date) -> np.ndarray:
        """Get 60-day PE history for a stock."""
        sql = """
        SELECT b.pe_ttm FROM daily_basic b
        JOIN securities s ON b.security_id = s.id
        WHERE s.code = ? AND b.trade_date <= ? AND b.trade_date >= ?
        ORDER BY b.trade_date
        """
        start = self._offset_date(date, -90)
        rows = conn.execute(sql, (code, date, start)).fetchall()
        return np.array([r[0] for r in rows[-60:] if r[0] is not None])

    def _get_industry(self, conn, code) -> str:
        """Get SW L1 industry for a stock."""
        row = conn.execute(
            "SELECT industry FROM securities WHERE code = ?", (code,)
        ).fetchone()
        return row[0] if row else 'unknown'

    def _write_cache(self, conn, rows):
        """Insert or replace rows into ng_feature_cache."""
        sql = """
        INSERT OR REPLACE INTO ng_feature_cache
            (code, trade_date, features_json, label_3d, label_5d, label_10d,
             market_return_5d, market_return_20d, market_volatility_20d,
             market_breadth, market_new_high_ratio, northbound_flow_5d,
             market_volume_ratio, market_drawdown, vix_proxy, market_momentum_diff)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        data = []
        for r in rows:
            data.append((
                r['code'], r['trade_date'], r['features_json'],
                r.get('label_3d'), r.get('label_5d'), r.get('label_10d'),
                r.get('market_return_5d'), r.get('market_return_20d'),
                r.get('market_volatility_20d'), r.get('market_breadth'),
                r.get('market_new_high_ratio'), r.get('northbound_flow_5d'),
                r.get('market_volume_ratio'), r.get('market_drawdown'),
                r.get('vix_proxy'), r.get('market_momentum_diff'),
            ))
        conn.executemany(sql, data)
        conn.commit()

    def backfill(self, start_date: str, end_date: str):
        """Backfill feature cache for a date range."""
        dates = self.get_trading_dates(start_date, end_date)
        print(f"Backfilling {len(dates)} dates: {start_date} → {end_date}")
        for i, date in enumerate(dates):
            t0 = time.time()
            count = self.update_single_date(date)
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(dates)}] {date}: {count} stocks ({elapsed:.1f}s)")


def _json_default(obj):
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return str(obj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NG Feature Cache Updater')
    parser.add_argument('--date', help='Single date (YYYY-MM-DD)')
    parser.add_argument('--start-date', help='Backfill start date')
    parser.add_argument('--end-date', help='Backfill end date')
    args = parser.parse_args()

    updater = NGCacheUpdater()

    if args.date:
        count = updater.update_single_date(args.date)
        print(f"Updated {count} stocks for {args.date}")
    elif args.start_date and args.end_date:
        updater.backfill(args.start_date, args.end_date)
    else:
        print("Usage: --date YYYY-MM-DD or --start-date/--end-date for backfill")
```

- [ ] **Step 2: Test with a single date**

Run: `python3 ml_models/ng/ng_cache_updater.py --date 2025-06-15`
Expected: prints stock count, no crashes. Verify with:
`python3 -c "import sqlite3; c=sqlite3.connect('data_adapter/stock_data.db'); print(c.execute('SELECT COUNT(*) FROM ng_feature_cache WHERE trade_date=\"2025-06-15\"').fetchone())"`

- [ ] **Step 3: Commit**

```bash
git add ml_models/ng/ng_cache_updater.py
git commit -m "feat(ng): batch feature cache updater with 62-factor computation"
```

---

### Task 5: Implement NG Trainer (inheriting V485Trainer)

**Files:**
- Create: `ml_models/ng/ng_trainer.py`

The trainer loads data from `ng_feature_cache`, applies preprocessing, then delegates to V485Trainer's WF + ensemble machinery.

- [ ] **Step 1: Write the NG trainer**

```python
# ml_models/ng/ng_trainer.py
"""NG Trainer — inherits V485Trainer, overrides feature loading to use ng_feature_cache."""
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml_models.training.train_v395_multi_target import V485Trainer

logger = logging.getLogger(__name__)

DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')

# 52 stock-level features (non-market, non-industry-rank features)
STOCK_FEATURE_NAMES = [
    # Trend state (12)
    'price_above_ma20', 'price_above_ma60', 'ma_alignment', 'trend_strength_20d',
    'new_high_20d', 'new_high_60d', 'days_since_breakout', 'adx_proxy',
    'macd_histogram', 'macd_acceleration', 'price_channel_position', 'cumulative_return_60d',
    # Pullback entry (10)
    'pullback_from_high', 'pullback_to_ma10', 'pullback_to_ma20', 'rsi_14',
    'kdj_j_value', 'volume_contraction', 'lower_shadow_ratio', 'consecutive_down_days',
    'bollinger_position', 'intraday_recovery',
    # Volume confirmation (8)
    'volume_ratio_5d', 'volume_price_corr', 'obv_trend', 'volume_breakout',
    'log_amount_ma5', 'turnover_rate', 'up_volume_ratio', 'volume_cv',
    # Fundamental quality (14)
    'roe_ttm', 'roe_change', 'revenue_growth', 'net_profit_margin', 'ocf_quality',
    'pe_ttm', 'pb', 'pe_percentile_60d', 'debt_to_assets', 'current_ratio',
    'log_market_cap', 'log_adv_20d', 'free_float_ratio', 'dv_ratio',
    # Industry (8)
    'industry_return_5d', 'industry_return_20d', 'industry_relative_strength',
    'industry_breadth', 'industry_volume_change', 'industry_rank_return_5d',
    'sw_index_return_5d', 'industry_hhi',
]

MARKET_FEATURE_NAMES = [
    'market_return_5d', 'market_return_20d', 'market_volatility_20d',
    'market_breadth', 'market_new_high_ratio', 'northbound_flow_5d',
    'market_volume_ratio', 'market_drawdown', 'vix_proxy', 'market_momentum_diff',
]

ALL_FEATURE_NAMES = STOCK_FEATURE_NAMES + MARKET_FEATURE_NAMES  # 62 total


class NGTrainer(V485Trainer):
    """NG model trainer. Loads from ng_feature_cache, uses 62 economically-motivated factors."""

    VERSION_TAG = 'ng'

    # Composite weights: focus on 5d (trend trading 5-10d hold)
    TARGET_WEIGHTS = {'label_3d': 0.15, 'label_5d': 0.50, 'label_10d': 0.35}

    def load_data(self, start_date=None, end_date=None) -> pd.DataFrame:
        """Load training data from ng_feature_cache."""
        import sqlite3
        conn = sqlite3.connect(self.db_path or DB_PATH, timeout=30)

        sql = """
        SELECT code, trade_date, features_json,
               label_3d, label_5d, label_10d,
               market_return_5d, market_return_20d, market_volatility_20d,
               market_breadth, market_new_high_ratio, northbound_flow_5d,
               market_volume_ratio, market_drawdown, vix_proxy, market_momentum_diff
        FROM ng_feature_cache
        WHERE trade_date BETWEEN ? AND ?
          AND label_5d IS NOT NULL
        ORDER BY trade_date, code
        """
        start = start_date or '2020-01-01'
        end = end_date or '2099-12-31'
        df = pd.read_sql(sql, conn, params=(start, end))
        conn.close()

        if df.empty:
            logger.warning("No data loaded from ng_feature_cache!")
            return df

        # Parse features_json into columns
        parsed = df['features_json'].apply(json.loads).tolist()
        feat_df = pd.DataFrame(parsed)

        # Ensure all expected stock features exist
        for col in STOCK_FEATURE_NAMES:
            if col not in feat_df.columns:
                feat_df[col] = np.nan

        # Merge stock features + market features (from columns) + labels
        result = pd.DataFrame()
        result['code'] = df['code'].values
        result['trade_date'] = df['trade_date'].values

        for col in STOCK_FEATURE_NAMES:
            result[col] = feat_df[col].values.astype(float)

        for col in MARKET_FEATURE_NAMES:
            result[col] = df[col].values.astype(float)

        result['label_3d'] = df['label_3d'].values.astype(float)
        result['label_5d'] = df['label_5d'].values.astype(float)
        result['label_10d'] = df['label_10d'].values.astype(float)

        logger.info(f"NG data loaded: {len(result)} samples, "
                    f"{len(ALL_FEATURE_NAMES)} features, "
                    f"dates: {result['trade_date'].min()} → {result['trade_date'].max()}")
        return result

    def get_feature_names(self):
        return ALL_FEATURE_NAMES

    def get_stock_feature_names(self):
        return STOCK_FEATURE_NAMES

    def get_market_feature_names(self):
        return MARKET_FEATURE_NAMES
```

- [ ] **Step 2: Verify it can load data (after cache is built)**

Run: `python3 -c "from ml_models.ng.ng_trainer import NGTrainer; t = NGTrainer(); df = t.load_data('2025-06-01', '2025-06-30'); print(f'Loaded: {len(df)} rows, {df.columns.tolist()[:5]}...')"`

- [ ] **Step 3: Commit**

```bash
git add ml_models/ng/ng_trainer.py
git commit -m "feat(ng): NG trainer inheriting V485, loads from ng_feature_cache"
```

---

### Task 6: Implement NG Production Scorer

**Files:**
- Create: `ml_models/ng/ng_production_scorer.py`

- [ ] **Step 1: Write the scorer**

```python
# ml_models/ng/ng_production_scorer.py
"""NG Production Scorer — loads NG model, scores all stocks for a date."""
import sys
import json
import logging
import numpy as np
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml_models.ng.ng_trainer import ALL_FEATURE_NAMES, STOCK_FEATURE_NAMES, MARKET_FEATURE_NAMES

logger = logging.getLogger(__name__)
MODEL_DIR = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'ng'


class NGProductionScorer:
    """Score stocks using the NG model."""

    def __init__(self):
        self.model_data = None
        self._load_models()

    def _load_models(self):
        """Load the latest NG model."""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        pkl_files = sorted(MODEL_DIR.glob('ng_*.pkl'), key=lambda f: f.stat().st_mtime)
        if not pkl_files:
            logger.warning("No NG model found. Run training first.")
            return
        latest = pkl_files[-1]
        self.model_data = joblib.load(latest)
        logger.info(f"NG model loaded: {latest.name}")

    def predict_scores(self, stock_codes: list, date: str) -> dict:
        """Score all stocks for a date.

        Returns: {code: {score, pred_3d, pred_5d, pred_10d, rank_score, recommendation}}
        """
        if self.model_data is None:
            return {}

        import sqlite3
        db_path = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')
        conn = sqlite3.connect(db_path, timeout=30)

        # Load features from ng_feature_cache
        placeholders = ','.join('?' * len(stock_codes))
        sql = f"""
        SELECT code, features_json,
               market_return_5d, market_return_20d, market_volatility_20d,
               market_breadth, market_new_high_ratio, northbound_flow_5d,
               market_volume_ratio, market_drawdown, vix_proxy, market_momentum_diff
        FROM ng_feature_cache
        WHERE trade_date = ? AND code IN ({placeholders})
        """
        rows = conn.execute(sql, [date] + list(stock_codes)).fetchall()
        conn.close()

        if not rows:
            return {}

        # Build feature matrix
        results = {}
        codes = []
        X_list = []

        for row in rows:
            code = row[0]
            features = json.loads(row[1])
            market_vals = row[2:]  # 10 market features

            feature_vec = []
            for fname in STOCK_FEATURE_NAMES:
                val = features.get(fname, np.nan)
                feature_vec.append(float(val) if val is not None else np.nan)
            for i, fname in enumerate(MARKET_FEATURE_NAMES):
                val = market_vals[i]
                feature_vec.append(float(val) if val is not None else np.nan)

            codes.append(code)
            X_list.append(feature_vec)

        X = np.array(X_list)

        # Apply winsorization bounds from training
        if 'winsorize_bounds' in self.model_data:
            bounds = self.model_data['winsorize_bounds']
            for j, (lo, hi) in enumerate(bounds):
                if j < X.shape[1]:
                    X[:, j] = np.clip(X[:, j], lo, hi)

        # Predict with ensemble for each target
        predictions = {}
        for target in ['label_3d', 'label_5d', 'label_10d']:
            target_models = self.model_data.get(f'models_{target}', {})
            if not target_models:
                continue
            preds = np.zeros(len(codes))
            total_weight = 0
            for name, model in target_models.items():
                try:
                    p = model.predict(X)
                    w = self.model_data.get(f'weights_{target}', {}).get(name, 1.0)
                    preds += w * p
                    total_weight += w
                except Exception:
                    continue
            if total_weight > 0:
                preds /= total_weight
            predictions[target] = preds

        # Composite ranking: 5d=0.50, 10d=0.35, 3d=0.15
        composite = np.zeros(len(codes))
        if 'label_5d' in predictions:
            composite += 0.50 * predictions['label_5d']
        if 'label_10d' in predictions:
            composite += 0.35 * predictions['label_10d']
        if 'label_3d' in predictions:
            composite += 0.15 * predictions['label_3d']

        for i, code in enumerate(codes):
            results[code] = {
                'pred_3d': float(predictions.get('label_3d', np.zeros(1))[i]),
                'pred_5d': float(predictions.get('label_5d', np.zeros(1))[i]),
                'pred_10d': float(predictions.get('label_10d', np.zeros(1))[i]),
                'rank_score': float(composite[i]),
                'score': 0.0,  # will be filled as percentile
                'recommendation': '',
            }

        # Convert rank_score to percentile score (0-100)
        if results:
            all_scores = [v['rank_score'] for v in results.values()]
            for code in results:
                pct = np.mean([s <= results[code]['rank_score'] for s in all_scores]) * 100
                results[code]['score'] = round(pct, 1)
                if pct >= 95:
                    results[code]['recommendation'] = '强烈买入'
                elif pct >= 85:
                    results[code]['recommendation'] = '买入'
                elif pct >= 70:
                    results[code]['recommendation'] = '谨慎买入'
                else:
                    results[code]['recommendation'] = '观望'

        return results
```

- [ ] **Step 2: Commit**

```bash
git add ml_models/ng/ng_production_scorer.py
git commit -m "feat(ng): production scorer with composite 5d/10d/3d ranking"
```

---

### Task 7: Register NG in batch report generator

**Files:**
- Modify: `backtest/batch_generate_v395_reports.py`

- [ ] **Step 1: Add 'ng' to SUPPORTED_VERSIONS and scorer mapping**

Find the `SUPPORTED_VERSIONS` set and add `'ng'`. Find the scorer mapping and add:

```python
from ml_models.ng.ng_production_scorer import NGProductionScorer
# In scorer_map:
'ng': NGProductionScorer,
```

Also in `get_trading_dates()`, add ng_feature_cache as the table source when version is 'ng'.

- [ ] **Step 2: Commit**

```bash
git add backtest/batch_generate_v395_reports.py
git commit -m "feat(ng): register NG version in batch report generator"
```

---

### Task 8: Backfill feature cache and run first training

- [ ] **Step 1: Backfill ng_feature_cache (2020-2025)**

Run: `python3 ml_models/ng/ng_cache_updater.py --start-date 2020-01-01 --end-date 2025-12-31`
Expected: ~1400 trading dates, ~2000-3000 stocks per date. May take 30-60 minutes.

- [ ] **Step 2: Verify cache contents**

Run: `python3 -c "import sqlite3; c=sqlite3.connect('data_adapter/stock_data.db'); print(c.execute('SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM ng_feature_cache').fetchone())"`
Expected: ~3-4 million rows, dates from 2020-01 to 2025-12.

- [ ] **Step 3: Run fast-check training**

The NG trainer inherits V485's walk-forward machinery. Run with `--fast-check`:

```bash
python3 -c "
from ml_models.ng.ng_trainer import NGTrainer
t = NGTrainer()
df = t.load_data('2020-01-01', '2025-12-31')
print(f'Loaded {len(df)} samples')
# Fast-check: small WF to verify pipeline works
# Full training uses the CLI: python3 ml_models/training/train_v395_multi_target.py --ng --fast-check
"
```

Note: the `--ng` CLI flag needs to be added to `train_v395_multi_target.py` to select NGTrainer. This is a small addition to the argparse section.

- [ ] **Step 4: Commit cache and any fixes**

```bash
git commit -m "feat(ng): backfill ng_feature_cache and verify training pipeline"
```

---

### Task 9: Generate WF OOS reports and run V5.2 evaluation

- [ ] **Step 1: Run full training with WF OOS report generation**

```bash
python3 ml_models/training/train_v395_multi_target.py --ng --sharpe-blend 0.3
```

This produces: model file + WF OOS reports in `reports/daily_selection_ng_wf_oos/`.

- [ ] **Step 2: Generate batch reports for extended evaluation**

```bash
python3 backtest/batch_generate_v395_reports.py \
    --version ng --start-date 2024-05-01 --end-date 2025-10-31 \
    --output-dir reports/daily_selection_ng_wf_oos
```

- [ ] **Step 3: Run V5.2 no-leakage evaluation**

```bash
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_ng_wf_oos \
    --label "NG-v1" --top-n 10 --focus-days 10 \
    --retention-bonus 0.2 --cppi-floor 0.08 --cppi-multiplier 20 \
    --score-floor 30 --ema-alpha 0.7 --min-market-cap 50 \
    --score-version v52
```

Expected: V5.2 score. **Success criterion: > 64.0% (beat V4901 baseline).**

- [ ] **Step 4: Commit results**

```bash
git add reports/ ml_models/trained_models/ng/
git commit -m "feat(ng): first NG model trained, V5.2 evaluation complete"
```
