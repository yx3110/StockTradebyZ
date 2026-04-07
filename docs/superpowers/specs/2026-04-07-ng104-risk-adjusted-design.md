# NG v1.0.4 Design Spec: Risk-Adjusted Labels + Multi-Seed Ensemble + IC Deep Screening

**Date**: 2026-04-07
**Status**: Approved
**Base**: ng1.0.3 (66 features, 5-model ensemble, industry excess labels)
**Goal**: MaxDD < 10% (hard), Sharpe >= 0.60, turnover <= 30x, excess return >= +15%

---

## 1. Problem Statement

ng1.0.3 achieves +18.1% annualized / +24.8% excess / 0.45 Sharpe in 2018-2020 OOS, but:
- MaxDD is not explicitly controlled (no hard constraint in model)
- Turnover ~43x/year erodes net returns via transaction costs
- IC stability screening was manual (only pre/post-2020, 2 regimes)
- Model labels optimize for raw excess return, ignoring intra-period drawdown

## 2. Design Overview

Four independent improvement dimensions, all at model level:

```
                    ng1.0.3 (baseline)
                          |
          +---------------+---------------+
          |               |               |
    [Risk-Adjusted   [Multi-Seed    [IC Stability
     Labels]          Ensemble]      Deep Screen]
          |               |               |
          +-------+-------+-------+-------+
                  |               |
           [Signal Smoothing Features]
                  |
              ng1.0.4
```

Each dimension is independently beneficial. If any single dimension underperforms in fast-check, it can be disabled without affecting others.

## 3. Risk-Adjusted Labels

### 3.1 Current Label (ng1.0.3)
```python
label_Nd = close[T+1+N] / open[T+1] - 1          # absolute return
excess_label_Nd = label_Nd - industry_median_Nd     # industry excess
```

### 3.2 New Label (ng1.0.4)
```python
# Step 1: Compute forward N-day max drawdown from close prices
# For each date T, look at close prices from T+1 to T+1+N
peak = close[T+1]
maxDD_Nd = 0.0
for t in range(T+2, T+1+N+1):
    peak = max(peak, close[t])
    dd = close[t] / peak - 1    # negative when in drawdown
    maxDD_Nd = min(maxDD_Nd, dd)
# maxDD_Nd in [-1, 0], more negative = deeper drawdown

# Step 2: Industry excess (same as ng1.0.3)
excess_return_Nd = label_Nd - industry_median_Nd

# Step 3: Risk-adjusted label (multiplicative penalty)
risk_adjusted_label_Nd = excess_return_Nd * (1 + maxDD_Nd) ** penalty_power
```

### 3.3 Parameters
- `penalty_power`: Controls drawdown penalty strength
  - 0.0 = no penalty (degenerates to ng1.0.3)
  - 1.0 = linear penalty (10% DD -> label shrinks 10%)
  - 1.5 = recommended starting point (10% DD -> label shrinks ~15%)
  - 2.0 = aggressive penalty (10% DD -> label shrinks ~19%)
- Fast-check grid: [0.0, 0.5, 1.0, 1.5, 2.0]
- Applied uniformly to all 4 targets (3d/5d/10d/15d), maxDD window matches target horizon

### 3.4 Edge Cases
- Stock suspended during forward window: skip (no label), same as ng1.0.3
- maxDD_Nd = 0 (stock only went up): penalty factor = 1.0, no change
- Negative excess return with deep DD: penalty makes it more negative (double punishment for losers that also had deep drawdowns — desired behavior)

### 3.5 Rationale
- Multiplicative form avoids Calmar division instability (maxDD near zero)
- GBDT sees higher labels for stocks with same return but lower drawdown
- Model learns to prefer "smooth uptrend" over "volatile spike" patterns
- penalty_power=0 provides clean A/B baseline comparison

## 4. Multi-Seed Ensemble

### 4.1 Current State
- Implemented in ng1.0.2 session: `scripts/ensemble_predict.py`
- 3 seeds (42, 123, 456), separate training + post-hoc averaging
- Seed propagation fixed in commit 3ce87d3b (all ML libs receive seed)

### 4.2 ng1.0.4 Upgrade
- **5 seeds**: [42, 123, 456, 789, 2024]
  - Variance reduction: 3 seeds -> 5 seeds = -40% prediction variance
  - Diminishing returns beyond 5 (var ~ 1/N)
- **Integrated into ng_trainer.py**: `--seeds 42,123,456,789,2024` trains all sequentially
  - Or parallel: 5 separate processes with `--seed X` each
- **Integrated into NGProductionScorer**: Auto-loads all seed models for a version, averages internally
  - Model naming: `ng104_seed{N}.pkl`
  - Scorer discovers: `glob(f'ng104_seed*.pkl')`

### 4.3 Ensemble Method
- Simple arithmetic mean of continuous predictions per target:
  ```python
  pred_Nd_ensemble = mean([model_seed_i.predict(features) for i in seeds])
  ```
- ICIR-adaptive composite weights computed per-seed, then averaged
- Final recommendation thresholds applied to ensemble score (not per-seed)

### 4.4 Expected Impact
- Turnover: ~43x -> ~25-30x (smoothed daily score changes)
- MaxDD: indirect reduction via more stable Top-N membership
- Alpha: preserved or improved (noise cancellation, signal retention)

## 5. IC Stability Deep Screening

### 5.1 Motivation
ng1.0.3 manually identified 3 IC-flipping factors by comparing pre/post-2020.
ng1.0.4 automates this with 6 market regimes for comprehensive coverage.

### 5.2 Regime Definitions
Derived from existing market features (no new data needed):
```python
regimes = {
    'bull':           market_return_20d > +5%,
    'bear':           market_return_20d < -5%,
    'sideways':       abs(market_return_20d) <= 5%,
    'high_vol':       market_volatility_20d > percentile_75,
    'small_dominant': csi1000_return > csi300_return,
    'large_dominant': csi300_return > csi1000_return,
}
```
Note: regimes can overlap (e.g., bull + high_vol). This is intentional — we want IC measured under each condition independently.

### 5.3 IC Stability Scoring Algorithm
For each feature f:
1. Compute Spearman IC(f, risk_adjusted_label_10d) within each regime
2. Evaluate:
   - `sign_consistency`: Do all regimes with |IC| > 0.01 have the same sign?
   - `ic_cv`: std(regime_ICs) / (|mean(regime_ICs)| + 1e-8)
3. Classification:
   - FLIP: sign_consistency = False -> **remove** (strong recommendation)
   - UNSTABLE: ic_cv > 2.0 -> **candidate for removal** (verify via fast-check)
   - STABLE: keep

### 5.4 Implementation
New script: `scripts/ic_stability_analyzer.py`
```bash
python3 scripts/ic_stability_analyzer.py \
  --cache-table ng103_feature_cache \
  --label label_10d \
  --output reports/ic_stability_analysis.md
```
- Output: markdown table with per-feature IC across 6 regimes + stability flag
- Run once before training, results manually reviewed, then hardcoded into ng1.0.4 feature list
- NOT automated in training loop (human-in-the-loop for feature decisions)

### 5.5 Expected Outcome
- Identify 3-8 additional unstable features beyond ng1.0.3's 3
- Final feature count: ~58-63 (from current 66) + new signal smoothing features
- Net feature count: ~63-71

## 6. Signal Smoothing Features

### 6.1 New Feature Candidates (8-10)

**Group 1: Long-Horizon Trend (3)**
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `trend_strength_60d` | Linear regression slope of 60d close prices | Long-term trend inertia, slow-changing signal |
| `ma60_distance` | (close - MA60) / MA60 | Distance from 60d mean, stable reference |
| `price_channel_pos_40d` | (close - low_40d) / (high_40d - low_40d) | Position within 40d channel, smooth [0,1] |

**Group 2: Volatility Regime (3)**
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `vol_ratio_5d_60d` | realized_vol_5d / realized_vol_60d | Short/long vol ratio, volatility clustering |
| `vol_regime` | percentile_rank(vol_20d, vol_250d_history) | Current vol regime [0,1], very stable |
| `downside_vol_20d` | std(returns[returns<0]) over 20d | Downside-only volatility, risk-specific |

**Group 3: Drawdown State (2-3)**
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `current_drawdown` | close / max(close_60d) - 1 | Current distance from peak, drawdown risk |
| `recovery_speed_20d` | (close - low_20d) / (high_20d - low_20d) | Rebound momentum from trough |
| `gap_risk_20d` | count(abs(open/prev_close-1) > 2%) / 20 | Gap frequency, tail risk proxy |

### 6.2 Data Requirements
All computable from existing `daily_quotes` table (open, high, low, close, volume).
No new API calls or data sources needed.

### 6.3 Feature Qualification
New features must pass IC stability screening (Section 5) before inclusion:
1. Compute feature values and add to ng104_feature_cache
2. Run IC stability analyzer on new + existing features
3. Only STABLE new features enter the final model
4. fast-check validates net improvement

### 6.4 Expected Impact on Turnover
- 60d trend features: high autocorrelation -> daily predictions change slowly
- Volatility regime: changes on weekly scale, not daily
- Drawdown state: monotonic during drawdowns, smooth recovery
- Combined with multi-seed averaging: double smoothing effect

## 7. Cache Architecture

### 7.1 New Cache Table
- Table name: `ng104_feature_cache`
- Columns: all ng1.0.3 features + 8-10 new features + risk-adjusted labels
- Additional label columns: `maxdd_3d`, `maxdd_5d`, `maxdd_10d`, `maxdd_15d` (raw maxDD for each horizon)
- Risk-adjusted labels stored as: `ra_label_3d`, `ra_label_5d`, `ra_label_10d`, `ra_label_15d`
- Original excess labels retained as: `label_3d`, `label_5d`, `label_10d`, `label_15d` (for A/B comparison)

### 7.2 Backfill
```bash
python3 ml_models/ng/ng_cache_updater.py \
  --start-date 2018-01-01 \
  --end-date 2026-04-07 \
  --version ng1.0.4
```

## 8. Training Pipeline

### 8.1 Step-by-Step

```
Phase 1: Data Preparation
  1. Implement new features in ng_feature_calculator.py
  2. Implement risk-adjusted label computation in ng_cache_updater.py
  3. Add ng1.0.4 to ng_schema.py (table mapping + feature list)
  4. Backfill ng104_feature_cache (2018-01-01 ~ today)

Phase 2: Feature Selection
  5. Run ic_stability_analyzer.py on all candidate features
  6. Review results, decide final feature set
  7. Update ng_trainer.py with ng1.0.4 feature list

Phase 3: Hyperparameter Tuning
  8. fast-check penalty_power grid: [0.0, 0.5, 1.0, 1.5, 2.0]
     - 2 WF windows, ~2min per run, single seed
     - Compare 10d ICIR across penalty values
  9. Select best penalty_power

Phase 4: Multi-Seed Training
  10. Train 5 seeds with best penalty_power
      - Sequential: ~5-10h total
      - Parallel (5 processes): ~1-2h
  11. Each seed produces: ng104_seed{N}.pkl

Phase 5: Evaluation
  12. Generate ensemble reports (WF-OOS + Pre-2020)
  13. Run north_star_eval.py on ensemble predictions
  14. Compare vs ng1.0.3 baseline on all target metrics
```

### 8.2 CLI Interface

```bash
# Training (single seed)
python3 ml_models/ng/ng_trainer.py \
  --version ng1.0.4 \
  --seed 42 \
  --penalty-power 1.5 \
  --purge-days 15

# Training (all seeds, sequential)
python3 ml_models/ng/ng_trainer.py \
  --version ng1.0.4 \
  --seeds 42,123,456,789,2024 \
  --penalty-power 1.5 \
  --purge-days 15

# Fast-check
python3 ml_models/ng/ng_trainer.py \
  --version ng1.0.4 \
  --fast-check \
  --penalty-power 1.5

# IC Stability Analysis
python3 scripts/ic_stability_analyzer.py \
  --cache-table ng104_feature_cache \
  --label ra_label_10d \
  --output reports/ic_stability_ng104.md

# Ensemble Reports
python3 scripts/ensemble_predict.py \
  --version ng1.0.4 \
  --seeds 42,123,456,789,2024 \
  --start-date 2018-04-02 \
  --end-date 2020-12-31 \
  --output-dir reports/daily_selection_ng104_ensemble_pre2020
```

## 9. Success Criteria

| Metric | ng1.0.3 Baseline | ng1.0.4 Target | Priority |
|--------|:-----------------:|:--------------:|:--------:|
| MaxDD (2018-2020 OOS) | uncontrolled | **< 10%** | P0 hard |
| Annualized Excess Return | +24.8% | >= +15% | P1 |
| Sharpe Ratio | 0.45 | >= 0.60 | P1 |
| Turnover | ~43x | <= 30x | P2 |
| Information Ratio | 0.89 | >= 0.80 | P2 |

## 10. Fallback Strategy

If risk-adjusted labels underperform (fast-check: penalty_power=0 wins):
- Keep original excess labels
- Still apply: multi-seed ensemble + IC deep screening + signal smoothing features
- Three independent dimensions still provide value

If multi-seed doesn't reduce turnover enough:
- Can stack with portfolio-level EMA smoothing (alpha=0.7, existing in V4.9.0.1)
- But this is portfolio construction layer, not model layer

## 11. Files to Modify

| File | Changes |
|------|---------|
| `ml_models/ng/ng_schema.py` | Add ng1.0.4 table mapping + feature list |
| `ml_models/ng/ng_feature_calculator.py` | Add 8-10 new features (trend/vol/drawdown) |
| `ml_models/ng/ng_cache_updater.py` | Add maxDD computation + risk-adjusted labels |
| `ml_models/ng/ng_trainer.py` | Add --version ng1.0.4, --penalty-power, --seeds |
| `ml_models/ng/ng_production_scorer.py` | Multi-seed auto-loading + ensemble averaging |
| `scripts/ic_stability_analyzer.py` | **NEW**: 6-regime IC stability analysis tool |
| `scripts/ensemble_predict.py` | Support --version, auto-discover seed models |
| `ml_models/ng/__init__.py` | Export ng1.0.4 components |
| `docs/wiki/models/ng-series.md` | Document ng1.0.4 design + results |

## 12. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|:----------:|:------:|------------|
| Calmar label noise (maxDD estimation) | Medium | Medium | penalty_power=0 fallback, fast-check validation |
| IC screening removes too many features | Low | Medium | Conservative threshold (ic_cv > 2.0), manual review |
| 5-seed training time too long | Low | Low | Parallel training, fast-check with single seed first |
| New features overfit specific regimes | Medium | Medium | IC stability screening applied to new features too |
| MaxDD target not achievable at model level alone | Medium | High | Can combine with CPPI at portfolio level if needed |
