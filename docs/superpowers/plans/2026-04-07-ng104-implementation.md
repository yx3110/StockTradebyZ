# NG v1.0.4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement ng1.0.4 with risk-adjusted labels, 5-seed ensemble, IC stability deep screening, and signal smoothing features to achieve MaxDD<10% and Sharpe>=0.60.

**Architecture:** Extend ng1.0.3's 66-feature pipeline with 9 new signal-smoothing features, risk-adjusted labels (multiplicative DD penalty), automated IC stability screening across 6 market regimes, and integrated 5-seed ensemble training/scoring. All changes are in the model layer — no portfolio construction changes.

**Tech Stack:** Python 3, LightGBM, XGBoost, CatBoost, scikit-learn, SQLite, NumPy, Pandas, SciPy

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `ml_models/ng/ng_schema.py` | Modify | Add ng1.0.4 table mapping with new columns |
| `ml_models/ng/ng_feature_calculator.py` | Modify | Add 9 signal smoothing feature functions |
| `ml_models/ng/ng_cache_updater.py` | Modify | Add maxDD label computation + new features + risk-adjusted labels |
| `ml_models/ng/ng_trainer.py` | Modify | Add --version ng1.0.4, --penalty-power, --seeds flags; load RA labels |
| `ml_models/ng/ng_production_scorer.py` | Modify | Multi-seed auto-loading and ensemble averaging |
| `scripts/ic_stability_analyzer.py` | Create | 6-regime IC stability analysis tool |
| `scripts/ensemble_predict.py` | Modify | Add --version flag, auto-discover seed models |
| `ml_models/ng/__init__.py` | Modify | Update version constant |

---

### Task 1: Schema — Add ng1.0.4 Table Definition

**Files:**
- Modify: `ml_models/ng/ng_schema.py`

- [ ] **Step 1: Add ng1.0.4 to VERSION_TABLE_MAP and schema**

In `ml_models/ng/ng_schema.py`, add the ng1.0.4 entry:

```python
# In VERSION_TABLE_MAP (line 19-24), add:
VERSION_TABLE_MAP = {
    'ng1.0.0': 'ng_feature_cache',
    'ng1.0.1': 'ng101_feature_cache',
    'ng1.0.2': 'ng102_feature_cache',
    'ng1.0.3': 'ng103_feature_cache',
    'ng1.0.4': 'ng104_feature_cache',
}
```

In `_schema_sql()`, add ng1.0.4 extra columns after the ng1.0.3 block (after line 45):

```python
    if ver >= 'ng1.0.4':
        extra_cols += '\n    maxdd_3d REAL,'
        extra_cols += '\n    maxdd_5d REAL,'
        extra_cols += '\n    maxdd_10d REAL,'
        extra_cols += '\n    maxdd_15d REAL,'
        extra_cols += '\n    ra_label_3d REAL,'
        extra_cols += '\n    ra_label_5d REAL,'
        extra_cols += '\n    ra_label_10d REAL,'
        extra_cols += '\n    ra_label_15d REAL,'
```

- [ ] **Step 2: Run schema creation to verify**

Run: `python3 -c "from ml_models.ng.ng_schema import create_table; create_table(version='ng1.0.4')"`
Expected: `ng104_feature_cache table ready: .../stock_data.db`

- [ ] **Step 3: Commit**

```bash
git add ml_models/ng/ng_schema.py
git commit -m "feat(ng104): schema — ng104_feature_cache表定义 (maxdd + ra_label列)"
```

---

### Task 2: Feature Calculator — Add 9 Signal Smoothing Features

**Files:**
- Modify: `ml_models/ng/ng_feature_calculator.py`

- [ ] **Step 1: Add `compute_smoothing_features()` function**

Add this new function after `compute_stock_features()` (after line 267) in `ml_models/ng/ng_feature_calculator.py`:

```python
# ---------------------------------------------------------------------------
# Function 1b: Signal smoothing features (9 factors, ng1.0.4)
# ---------------------------------------------------------------------------

def compute_smoothing_features(
    closes: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
) -> Dict[str, float]:
    """
    Compute 9 signal smoothing features for ng1.0.4:
      Group 1 - Long-horizon trend (3): trend_strength_60d, ma60_distance, price_channel_pos_40d
      Group 2 - Volatility regime (3): vol_ratio_5d_60d, vol_regime, downside_vol_20d
      Group 3 - Drawdown state (3): current_drawdown, recovery_speed_20d, gap_risk_20d
    """
    result: Dict[str, float] = {}
    close = float(closes[-1])

    # ---- Group 1: Long-Horizon Trend (3) ----

    # 1. trend_strength_60d
    if len(closes) >= 60:
        c60 = closes[-60:].astype(float)
        slope = _linreg_slope(c60)
        std60 = c60.std()
        result['trend_strength_60d'] = slope / (std60 + 1e-8) if not np.isnan(slope) else np.nan
    else:
        result['trend_strength_60d'] = np.nan

    # 2. ma60_distance
    if len(closes) >= 60:
        ma60 = float(np.mean(closes[-60:]))
        result['ma60_distance'] = close / (ma60 + 1e-8) - 1.0
    else:
        result['ma60_distance'] = np.nan

    # 3. price_channel_pos_40d
    if len(closes) >= 40:
        high_40d = float(np.max(highs[-40:]))
        low_40d = float(np.min(lows[-40:]))
        channel_range = high_40d - low_40d
        result['price_channel_pos_40d'] = (close - low_40d) / (channel_range + 1e-8)
    else:
        result['price_channel_pos_40d'] = np.nan

    # ---- Group 2: Volatility Regime (3) ----

    # 4. vol_ratio_5d_60d
    if len(closes) >= 60:
        rets = np.diff(np.log(closes[-60:].astype(float) + 1e-8))
        vol_5d = float(np.std(rets[-5:])) if len(rets) >= 5 else np.nan
        vol_60d = float(np.std(rets))
        result['vol_ratio_5d_60d'] = vol_5d / (vol_60d + 1e-8) if not np.isnan(vol_5d) else np.nan
    else:
        result['vol_ratio_5d_60d'] = np.nan

    # 5. vol_regime — 20d realized vol percentile in 250d history
    if len(closes) >= 250:
        log_rets = np.diff(np.log(closes.astype(float) + 1e-8))
        # Rolling 20d vol for last 250 days
        if len(log_rets) >= 250:
            vols_history = []
            for i in range(len(log_rets) - 249, len(log_rets) - 19):
                vols_history.append(float(np.std(log_rets[i:i+20])))
            current_vol = float(np.std(log_rets[-20:]))
            if vols_history:
                result['vol_regime'] = float(np.mean(np.array(vols_history) < current_vol))
            else:
                result['vol_regime'] = np.nan
        else:
            result['vol_regime'] = np.nan
    elif len(closes) >= 60:
        # Fallback: use 60d history
        log_rets = np.diff(np.log(closes[-60:].astype(float) + 1e-8))
        vol_20d = float(np.std(log_rets[-20:])) if len(log_rets) >= 20 else np.nan
        vol_60d = float(np.std(log_rets))
        result['vol_regime'] = 0.5 if np.isnan(vol_20d) else float(vol_20d > vol_60d)
    else:
        result['vol_regime'] = np.nan

    # 6. downside_vol_20d
    if len(closes) >= 21:
        daily_rets = np.diff(closes[-21:].astype(float)) / (closes[-21:-1].astype(float) + 1e-8)
        neg_rets = daily_rets[daily_rets < 0]
        result['downside_vol_20d'] = float(np.std(neg_rets)) if len(neg_rets) >= 3 else 0.0
    else:
        result['downside_vol_20d'] = np.nan

    # ---- Group 3: Drawdown State (3) ----

    # 7. current_drawdown
    if len(closes) >= 60:
        peak_60d = float(np.max(closes[-60:]))
        result['current_drawdown'] = close / (peak_60d + 1e-8) - 1.0
    else:
        result['current_drawdown'] = np.nan

    # 8. recovery_speed_20d
    if len(closes) >= 20:
        high_20d = float(np.max(closes[-20:]))
        low_20d = float(np.min(closes[-20:]))
        channel = high_20d - low_20d
        result['recovery_speed_20d'] = (close - low_20d) / (channel + 1e-8)
    else:
        result['recovery_speed_20d'] = np.nan

    # 9. gap_risk_20d
    if len(closes) >= 21 and len(opens) >= 20:
        gap_count = 0
        for i in range(-20, 0):
            prev_close = float(closes[i - 1])
            today_open = float(opens[i])
            if prev_close > 1e-8 and abs(today_open / prev_close - 1.0) > 0.02:
                gap_count += 1
        result['gap_risk_20d'] = gap_count / 20.0
    else:
        result['gap_risk_20d'] = np.nan

    return result
```

- [ ] **Step 2: Verify function is importable**

Run: `python3 -c "from ml_models.ng.ng_feature_calculator import compute_smoothing_features; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add ml_models/ng/ng_feature_calculator.py
git commit -m "feat(ng104): 9个信号平滑特征 — 长周期趋势+波动率regime+回撤状态"
```

---

### Task 3: Cache Updater — MaxDD Labels + Risk-Adjusted Labels + New Features

**Files:**
- Modify: `ml_models/ng/ng_cache_updater.py`

- [ ] **Step 1: Add maxDD computation function**

Add this function near the top of `ng_cache_updater.py` (near `compute_labels_from_future_prices` around line 48):

```python
def compute_maxdd_from_future_prices(
    base_open: float,
    future_closes: Dict[int, float],
    horizons: tuple = (3, 5, 10, 15),
) -> Dict[str, float]:
    """Compute max drawdown for each horizon from future close prices.

    MaxDD_Nd = min over t in [1..N] of (close_t / peak_so_far - 1)
    Returns dict like {'maxdd_3d': -0.05, 'maxdd_5d': -0.08, ...}
    Values are in [-1, 0], more negative = deeper drawdown.
    """
    result = {}
    for h in horizons:
        # Collect close prices from day 1 to day h (inclusive)
        prices = []
        for t in range(0, h + 1):
            if t in future_closes and not np.isnan(future_closes[t]):
                prices.append(future_closes[t])
        if not prices:
            result[f'maxdd_{h}d'] = np.nan
            continue

        # Track peak and max drawdown
        peak = prices[0]
        max_dd = 0.0
        for p in prices:
            if p > peak:
                peak = p
            dd = p / (peak + 1e-8) - 1.0
            if dd < max_dd:
                max_dd = dd
        result[f'maxdd_{h}d'] = float(max_dd)

    return result
```

- [ ] **Step 2: Modify `_compute_labels` to also compute maxDD**

In the `_compute_labels` method (around line 602), after the existing label computation, add maxDD:

```python
    def _compute_labels(
        self, future_dates, future_prices, security_ids
    ):
        # ... existing code stays the same ...
        # After computing labels_all via compute_labels_from_future_prices,
        # also compute maxDD for each security

        # Existing result dict already built...
        # Add maxDD computation
        if self.version >= 'ng1.0.4':
            for sid in security_ids:
                fp = future_prices.get(sid, {})
                if not fp or future_dates[0] not in fp:
                    continue
                future_closes = {}
                for n in LABEL_HORIZONS:
                    if n < len(future_dates) and future_dates[n] in fp:
                        future_closes[n] = fp[future_dates[n]].get('close', np.nan)
                # Also include day 0 close for maxDD tracking
                if future_dates[0] in fp:
                    future_closes[0] = fp[future_dates[0]].get('close', np.nan)

                maxdd = compute_maxdd_from_future_prices(
                    base_open=fp[future_dates[0]].get('open', np.nan),
                    future_closes=future_closes,
                    horizons=tuple(LABEL_HORIZONS),
                )
                if sid in result:
                    result[sid].update(maxdd)

        return result
```

- [ ] **Step 3: Add risk-adjusted label conversion**

Add a new method `_convert_labels_to_risk_adjusted` in the NGCacheUpdater class (after `_convert_labels_to_residual`, around line 777):

```python
    def _convert_labels_to_risk_adjusted(
        self,
        labels_all: Dict[int, Dict[str, float]],
        penalty_power: float = 1.5,
    ) -> Dict[int, Dict[str, float]]:
        """
        ng1.0.4: Convert excess labels to risk-adjusted labels.
        ra_label_Nd = excess_label_Nd * (1 + maxDD_Nd) ^ penalty_power

        Stores both original excess (label_Xd) and risk-adjusted (ra_label_Xd).
        """
        for sid, labs in labels_all.items():
            for h in LABEL_HORIZONS:
                excess_key = f'label_{h}d'
                maxdd_key = f'maxdd_{h}d'
                ra_key = f'ra_label_{h}d'

                excess = labs.get(excess_key, np.nan)
                maxdd = labs.get(maxdd_key, np.nan)

                if np.isnan(excess) or np.isnan(maxdd):
                    labs[ra_key] = np.nan
                    continue

                # (1 + maxdd) in (0, 1] since maxdd in [-1, 0]
                penalty = (1.0 + maxdd) ** penalty_power
                labs[ra_key] = excess * penalty

        return labels_all
```

- [ ] **Step 4: Integrate smoothing features into `update_single_date`**

In the `update_single_date` method, after `compute_stock_features()` call (around line 1018), add smoothing features:

```python
                # --- ng1.0.4: Compute smoothing features (9) ---
                smooth_feats = {}
                if self.version >= 'ng1.0.4':
                    from ml_models.ng.ng_feature_calculator import compute_smoothing_features
                    try:
                        smooth_feats = compute_smoothing_features(
                            closes=closes, opens=opens, highs=highs,
                            lows=lows, volumes=volumes,
                        )
                    except Exception as e:
                        print(f"    WARN: smoothing_features failed for {code}: {e}")
                        smooth_feats = {}
```

Then in the feature assembly section (where `eligible_stocks[sid]` is built, around line 1111), merge smooth_feats:

```python
                eligible_stocks[sid] = {
                    # ... existing fields ...
                    'smooth_feats': smooth_feats,  # ng1.0.4
                }
```

And in the final feature JSON assembly (where features_json is built), merge smooth_feats into the dict.

- [ ] **Step 5: Integrate maxDD labels and risk-adjusted labels in label flow**

After `_convert_labels_to_excess` and `_convert_labels_to_residual` calls (around line 926), add:

```python
            # ng1.0.4: Compute risk-adjusted labels
            if self.version >= 'ng1.0.4':
                labels_all = self._convert_labels_to_risk_adjusted(
                    labels_all, penalty_power=self.penalty_power
                )
```

- [ ] **Step 6: Update INSERT to include ng1.0.4 columns**

In the INSERT statement that writes to the cache table, add the new columns for ng1.0.4:

```python
                if self.version >= 'ng1.0.4':
                    row_data['maxdd_3d'] = labels.get('maxdd_3d', None)
                    row_data['maxdd_5d'] = labels.get('maxdd_5d', None)
                    row_data['maxdd_10d'] = labels.get('maxdd_10d', None)
                    row_data['maxdd_15d'] = labels.get('maxdd_15d', None)
                    row_data['ra_label_3d'] = labels.get('ra_label_3d', None)
                    row_data['ra_label_5d'] = labels.get('ra_label_5d', None)
                    row_data['ra_label_10d'] = labels.get('ra_label_10d', None)
                    row_data['ra_label_15d'] = labels.get('ra_label_15d', None)
```

- [ ] **Step 7: Add --penalty-power CLI arg to cache updater**

In the CLI section of `ng_cache_updater.py`, add:

```python
    parser.add_argument('--penalty-power', type=float, default=1.5,
                        help='Risk-adjusted label penalty power (default: 1.5)')
```

And pass it to the updater: `updater.penalty_power = args.penalty_power`

- [ ] **Step 8: Commit**

```bash
git add ml_models/ng/ng_cache_updater.py
git commit -m "feat(ng104): maxDD标签+风险调整标签+信号平滑特征 集成到cache updater"
```

---

### Task 4: Trainer — Version Flag + Risk-Adjusted Label Loading + Seeds

**Files:**
- Modify: `ml_models/ng/ng_trainer.py`

- [ ] **Step 1: Add ng1.0.4 feature names constant**

After `ALL_FEATURE_NAMES` (line 81), add ng1.0.4 smoothing features:

```python
SMOOTHING_FEATURE_NAMES: List[str] = [
    # Long-horizon trend (3)
    'trend_strength_60d', 'ma60_distance', 'price_channel_pos_40d',
    # Volatility regime (3)
    'vol_ratio_5d_60d', 'vol_regime', 'downside_vol_20d',
    # Drawdown state (3)
    'current_drawdown', 'recovery_speed_20d', 'gap_risk_20d',
]

# ng1.0.4 = ng1.0.3 base (56 stock) + 9 smoothing = 65 stock features + 10 market = 75 total
# (before IC stability screening, which may remove some)
NG104_STOCK_FEATURES: List[str] = STOCK_FEATURE_NAMES + SMOOTHING_FEATURE_NAMES
NG104_ALL_FEATURES: List[str] = NG104_STOCK_FEATURES + MARKET_FEATURE_NAMES
```

- [ ] **Step 2: Modify NGTrainer to support ng1.0.4 version**

Update `__init__` to accept version parameter:

```python
    def __init__(self, db_path: str = DB_PATH, version: str = 'ng1.0.3'):
        super().__init__(db_path)
        self._ng_version = version
        self.target_weights = dict(self.TARGET_WEIGHTS)
        self._turbo_skip_etf = True
        self.cache_table = get_table_name(version)
        # Select feature set by version
        if version >= 'ng1.0.4':
            self.feature_names = list(NG104_ALL_FEATURES)
            self.stock_feature_cols = list(NG104_STOCK_FEATURES)
        else:
            self.feature_names = list(ALL_FEATURE_NAMES)
            self.stock_feature_cols = list(STOCK_FEATURE_NAMES)
        self.macro_feature_cols = list(MARKET_FEATURE_NAMES)
        # ... rest of init ...
```

- [ ] **Step 3: Modify load_data to read risk-adjusted labels for ng1.0.4**

In `load_data` method, when version is ng1.0.4, read `ra_label_Xd` columns and use them as training labels:

```python
        # ng1.0.4: Use risk-adjusted labels if available
        if self._ng_version >= 'ng1.0.4':
            for h in ['3d', '5d', '10d', '15d']:
                ra_col = f'ra_label_{h}'
                if ra_col in df_raw.columns:
                    ra_vals = pd.to_numeric(df_raw[ra_col], errors='coerce')
                    # Use RA label where available, fallback to excess
                    mask = ra_vals.notna()
                    result.loc[mask, f'label_{h}'] = ra_vals[mask].values
```

- [ ] **Step 4: Add --version and --seeds CLI arguments**

Update the CLI parser (around line 784):

```python
    parser.add_argument('--version', default='ng1.0.3',
                        help='NG version (ng1.0.3 or ng1.0.4)')
    parser.add_argument('--penalty-power', type=float, default=1.5,
                        help='Risk-adjusted label penalty power (ng1.0.4)')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated seeds for multi-seed training (e.g., 42,123,456,789,2024)')
```

And in the main block, handle --seeds:

```python
    if args.seeds:
        seed_list = [int(s.strip()) for s in args.seeds.split(',')]
        for i, seed in enumerate(seed_list):
            logger.info(f"\n{'='*60}")
            logger.info(f"Training seed {seed} ({i+1}/{len(seed_list)})")
            logger.info(f"{'='*60}")
            # Set seed
            import random
            random.seed(seed)
            np.random.seed(seed)
            import ml_models.training.train_v395_multi_target as _trainer_mod
            _trainer_mod._GLOBAL_RANDOM_SEED = seed

            trainer = NGTrainer(version=args.version)
            # ... set switches ...
            model_data, history = trainer.walk_forward_train(...)
    elif args.seed is not None:
        # ... existing single seed logic ...
```

- [ ] **Step 5: Update model save naming for ng1.0.4 + seed**

In `walk_forward_train`, when saving the model, use version-aware naming:

```python
            # Version-aware naming
            version_tag = self._ng_version.replace('.', '')  # ng104
            seed_tag = f'_seed{_trainer_mod._GLOBAL_RANDOM_SEED}' if hasattr(_trainer_mod, '_GLOBAL_RANDOM_SEED') and _trainer_mod._GLOBAL_RANDOM_SEED != 42 else ''
            new_path = ng_dir / f'{version_tag}{seed_tag}_multi_target_{timestamp}.pkl'
```

- [ ] **Step 6: Commit**

```bash
git add ml_models/ng/ng_trainer.py
git commit -m "feat(ng104): trainer支持 --version ng1.0.4 + --seeds多种子 + 风险调整标签加载"
```

---

### Task 5: Production Scorer — Multi-Seed Auto-Loading

**Files:**
- Modify: `ml_models/ng/ng_production_scorer.py`

- [ ] **Step 1: Add multi-seed model discovery and loading**

Add a new method `_load_ensemble_models` to NGProductionScorer:

```python
    def _load_ensemble_models(self, version: str = None):
        """Load all seed models for a given version and average predictions."""
        ver = version or 'ng1.0.4'
        ver_tag = ver.replace('.', '')  # ng104

        seed_files = sorted(
            self.model_dir.glob(f'{ver_tag}_seed*_multi_target_*.pkl'),
            key=lambda f: f.stat().st_mtime
        )

        if not seed_files:
            # Fallback: look for single model without seed tag
            single_files = sorted(
                self.model_dir.glob(f'{ver_tag}_multi_target_*.pkl'),
                key=lambda f: f.stat().st_mtime
            )
            if single_files:
                self._load_model(str(single_files[-1]))
                self._ensemble_scorers = None
                return
            logger.warning(f"No {ver_tag} models found")
            return

        # Load all seed models as separate scorers
        self._ensemble_scorers = []
        for sf in seed_files:
            scorer = NGProductionScorer.__new__(NGProductionScorer)
            scorer.db_path = self.db_path
            scorer.model_dir = self.model_dir
            scorer.models = {}
            scorer.weights = {}
            scorer.feature_names = list(ALL_FEATURE_NAMES)
            scorer.stock_feature_cols = list(STOCK_FEATURE_NAMES)
            scorer.macro_feature_cols = list(MARKET_FEATURE_NAMES)
            scorer.winsorize_bounds = None
            scorer.global_quantiles = None
            scorer.recommendation_thresholds = None
            scorer.target_weights = dict(DEFAULT_COMPOSITE_WEIGHTS)
            scorer.cache_table = get_table_name(ver)
            scorer.downside_model = None
            scorer.lambda_risk = 0.5
            scorer._ensemble_scorers = None
            scorer._load_model(str(sf))
            self._ensemble_scorers.append(scorer)

        # Use first scorer's metadata for this scorer
        first = self._ensemble_scorers[0]
        self.feature_names = first.feature_names
        self.stock_feature_cols = first.stock_feature_cols
        self.macro_feature_cols = first.macro_feature_cols
        self.cache_table = first.cache_table
        self.target_weights = first.target_weights
        self.winsorize_bounds = first.winsorize_bounds
        self.global_quantiles = first.global_quantiles

        print(f"  Multi-seed ensemble: {len(self._ensemble_scorers)} models loaded")
```

- [ ] **Step 2: Override predict_scores to average across seeds**

Add a wrapper that checks for ensemble mode:

```python
    def predict_scores(self, stock_codes, date):
        if getattr(self, '_ensemble_scorers', None):
            return self._ensemble_predict_scores(stock_codes, date)
        return super().predict_scores(stock_codes, date)  # actually call existing method

    def _ensemble_predict_scores(self, stock_codes, date):
        """Average predictions from all seed models."""
        # Load features once
        features_df = self._load_features(stock_codes, date)

        all_results = []
        for scorer in self._ensemble_scorers:
            if features_df is not None and len(features_df) > 0:
                results = scorer.predict_scores_from_preloaded(
                    stock_codes, date, features_df.copy())
            else:
                results = scorer.predict_scores(stock_codes, date)
            all_results.append(results)

        # Average numeric fields
        merged = {}
        for code in stock_codes:
            preds = [r.get(code, {}) for r in all_results
                     if code in r and r[code].get('exec_filter') != 'no_data']
            if not preds:
                merged[code] = {
                    'score': 50.0, 'pred_3d': 0, 'pred_5d': 0,
                    'pred_10d': 0, 'pred_15d': 0, 'rank_score': 0,
                    'recommendation': '观望', 'exec_filter': 'no_data',
                }
                continue

            avg = {}
            for key in ['pred_3d', 'pred_5d', 'pred_10d', 'pred_15d', 'rank_score', 'score']:
                vals = [p.get(key, 0) for p in preds if p.get(key) is not None]
                avg[key] = float(np.mean(vals)) if vals else 0.0
            avg['recommendation'] = self._get_recommendation(avg['score'])
            merged[code] = avg

        return merged
```

- [ ] **Step 3: Update `__init__` to accept version parameter for auto-loading**

```python
    def __init__(self, db_path=None, model_path=None, version=None):
        # ... existing init code ...
        self._ensemble_scorers = None

        if version and version >= 'ng1.0.4' and model_path is None:
            self._load_ensemble_models(version)
        else:
            self._load_model(model_path)
```

- [ ] **Step 4: Commit**

```bash
git add ml_models/ng/ng_production_scorer.py
git commit -m "feat(ng104): scorer多种子自动发现+ensemble平均"
```

---

### Task 6: IC Stability Analyzer Script

**Files:**
- Create: `scripts/ic_stability_analyzer.py`

- [ ] **Step 1: Create the script**

Create `scripts/ic_stability_analyzer.py`:

```python
#!/usr/bin/env python3
"""
IC Stability Analyzer — 6-regime IC analysis for feature screening.

Computes Spearman IC for each feature across 6 market regimes and flags
FLIP (IC sign changes) and UNSTABLE (high IC variance) features.

Usage:
  python3 scripts/ic_stability_analyzer.py \
    --cache-table ng104_feature_cache \
    --label ra_label_10d \
    --output reports/ic_stability_ng104.md
"""

import sys
import os
import json
import sqlite3
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DB_PATH = os.path.join(PROJECT_ROOT, 'data_adapter', 'stock_data.db')

try:
    import orjson
    _json_loads = orjson.loads
except ImportError:
    _json_loads = json.loads


def load_data(cache_table, label_col, db_path=DB_PATH):
    """Load features + labels + market features from cache."""
    conn = sqlite3.connect(db_path, timeout=30)
    query = f"""
    SELECT code, trade_date, features_json,
           {label_col},
           market_return_5d, market_return_20d, market_volatility_20d
    FROM {cache_table}
    WHERE {label_col} IS NOT NULL
    ORDER BY trade_date
    """
    df = pd.read_sql(query, conn)
    conn.close()

    if df.empty:
        print(f"No data in {cache_table} with {label_col}")
        return None

    # Parse features
    parsed = df['features_json'].apply(_json_loads).tolist()
    features_df = pd.DataFrame(parsed)
    features_df['trade_date'] = df['trade_date'].values
    features_df['label'] = pd.to_numeric(df[label_col], errors='coerce').values
    features_df['market_return_20d'] = pd.to_numeric(df['market_return_20d'], errors='coerce').values
    features_df['market_volatility_20d'] = pd.to_numeric(df['market_volatility_20d'], errors='coerce').values

    return features_df


def define_regimes(df):
    """Define 6 market regimes from market features."""
    mkt_ret = df['market_return_20d'].values
    mkt_vol = df['market_volatility_20d'].values

    vol_p75 = np.nanpercentile(mkt_vol, 75) if not np.all(np.isnan(mkt_vol)) else 0.05

    regimes = {
        'bull': mkt_ret > 0.05,
        'bear': mkt_ret < -0.05,
        'sideways': np.abs(mkt_ret) <= 0.05,
        'high_vol': mkt_vol > vol_p75,
        'low_vol': mkt_vol <= vol_p75,
    }

    # Note: small/large dominant requires CSI1000 vs CSI300 data
    # For now, use vol-based proxy; can extend later with index data

    return regimes


def compute_regime_ic(df, feature_col, regimes):
    """Compute Spearman IC for a feature in each regime."""
    results = {}
    x_all = pd.to_numeric(df[feature_col], errors='coerce').values
    y_all = df['label'].values

    for regime_name, mask in regimes.items():
        x = x_all[mask]
        y = y_all[mask]
        valid = ~np.isnan(x) & ~np.isnan(y)
        n_valid = valid.sum()

        if n_valid < 500:
            results[regime_name] = {'ic': np.nan, 'n': int(n_valid)}
            continue

        ic, pval = spearmanr(x[valid], y[valid])
        results[regime_name] = {'ic': float(ic), 'n': int(n_valid), 'pval': float(pval)}

    return results


def classify_stability(regime_ics):
    """Classify feature as STABLE, UNSTABLE, or FLIP."""
    ics = [v['ic'] for v in regime_ics.values()
           if not np.isnan(v.get('ic', np.nan)) and abs(v['ic']) > 0.01]

    if len(ics) < 2:
        return 'INSUFFICIENT', 0.0

    signs = [np.sign(ic) for ic in ics]
    sign_set = set(signs)
    mean_ic = np.mean(ics)
    std_ic = np.std(ics)
    ic_cv = std_ic / (abs(mean_ic) + 1e-8)

    if len(sign_set) > 1:
        return 'FLIP', ic_cv
    elif ic_cv > 2.0:
        return 'UNSTABLE', ic_cv
    else:
        return 'STABLE', ic_cv


def generate_report(results, output_path):
    """Generate markdown report."""
    lines = ['# IC Stability Analysis Report\n']
    lines.append(f'Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}\n')

    # Summary
    n_stable = sum(1 for r in results if r['flag'] == 'STABLE')
    n_flip = sum(1 for r in results if r['flag'] == 'FLIP')
    n_unstable = sum(1 for r in results if r['flag'] == 'UNSTABLE')
    n_insuf = sum(1 for r in results if r['flag'] == 'INSUFFICIENT')

    lines.append(f'## Summary\n')
    lines.append(f'- STABLE: {n_stable}')
    lines.append(f'- FLIP: {n_flip} (recommend remove)')
    lines.append(f'- UNSTABLE: {n_unstable} (verify via fast-check)')
    lines.append(f'- INSUFFICIENT: {n_insuf}\n')

    # Detailed table
    lines.append('## Feature IC by Regime\n')
    regime_names = ['bull', 'bear', 'sideways', 'high_vol', 'low_vol']
    header = '| Feature | ' + ' | '.join(regime_names) + ' | Mean IC | CV | Flag |'
    separator = '|' + '|'.join([':---:'] * (len(regime_names) + 4)) + '|'
    lines.append(header)
    lines.append(separator)

    # Sort: FLIP first, then UNSTABLE, then STABLE
    flag_order = {'FLIP': 0, 'UNSTABLE': 1, 'INSUFFICIENT': 2, 'STABLE': 3}
    results.sort(key=lambda r: (flag_order.get(r['flag'], 9), -r['ic_cv']))

    for r in results:
        ic_cells = []
        for rn in regime_names:
            ic = r['regime_ics'].get(rn, {}).get('ic', np.nan)
            if np.isnan(ic):
                ic_cells.append('-')
            else:
                ic_cells.append(f'{ic:.4f}')

        mean_ic = np.nanmean([r['regime_ics'].get(rn, {}).get('ic', np.nan) for rn in regime_names])
        flag_emoji = {'FLIP': 'FLIP', 'UNSTABLE': 'UNSTABLE', 'STABLE': 'STABLE', 'INSUFFICIENT': 'INSUF'}

        row = f"| {r['feature']} | {' | '.join(ic_cells)} | {mean_ic:.4f} | {r['ic_cv']:.2f} | {flag_emoji.get(r['flag'], r['flag'])} |"
        lines.append(row)

    report = '\n'.join(lines)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nReport saved: {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description='IC Stability Analyzer')
    parser.add_argument('--cache-table', default='ng104_feature_cache',
                        help='Feature cache table name')
    parser.add_argument('--label', default='ra_label_10d',
                        help='Label column for IC computation')
    parser.add_argument('--output', default='reports/ic_stability_analysis.md',
                        help='Output report path')
    args = parser.parse_args()

    print(f"IC Stability Analyzer")
    print(f"  Cache table: {args.cache_table}")
    print(f"  Label: {args.label}")

    # Load data
    df = load_data(args.cache_table, args.label)
    if df is None:
        return

    print(f"  Loaded: {len(df):,} rows, {df['trade_date'].nunique()} dates")

    # Define regimes
    regimes = define_regimes(df)
    for name, mask in regimes.items():
        print(f"  Regime '{name}': {mask.sum():,} samples")

    # Get feature columns (exclude metadata)
    exclude = {'trade_date', 'label', 'market_return_20d', 'market_volatility_20d',
               'market_return_5d', 'code'}
    feature_cols = [c for c in df.columns if c not in exclude and not c.startswith('label')]

    print(f"\n  Analyzing {len(feature_cols)} features...")

    results = []
    for feat in feature_cols:
        regime_ics = compute_regime_ic(df, feat, regimes)
        flag, ic_cv = classify_stability(regime_ics)
        results.append({
            'feature': feat,
            'regime_ics': regime_ics,
            'flag': flag,
            'ic_cv': ic_cv,
        })

    # Print summary
    print(f"\n  Results:")
    for r in sorted(results, key=lambda x: x['flag']):
        if r['flag'] in ('FLIP', 'UNSTABLE'):
            ics = {k: f"{v['ic']:.4f}" for k, v in r['regime_ics'].items()
                   if not np.isnan(v.get('ic', np.nan))}
            print(f"    [{r['flag']}] {r['feature']}: CV={r['ic_cv']:.2f}, ICs={ics}")

    # Generate report
    generate_report(results, args.output)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Verify script runs (help mode)**

Run: `python3 scripts/ic_stability_analyzer.py --help`
Expected: Shows argument parser help with --cache-table, --label, --output

- [ ] **Step 3: Commit**

```bash
git add scripts/ic_stability_analyzer.py
git commit -m "feat(ng104): IC稳定性分析器 — 6-regime自动化筛选FLIP/UNSTABLE特征"
```

---

### Task 7: Update ensemble_predict.py — Version Support

**Files:**
- Modify: `scripts/ensemble_predict.py`

- [ ] **Step 1: Add --version flag and auto-discovery**

Replace the `--models` required arg with optional, add `--version`:

```python
    parser.add_argument('--models', nargs='+', default=None,
                        help='Model .pkl file paths (auto-discovered if --version used)')
    parser.add_argument('--version', default=None,
                        help='NG version for auto model discovery (e.g., ng1.0.4)')
    parser.add_argument('--seeds', default='42,123,456,789,2024',
                        help='Seed list for auto-discovery (comma-separated)')
```

Add auto-discovery logic before scorer loading:

```python
    if args.models:
        model_paths = args.models
    elif args.version:
        ver_tag = args.version.replace('.', '')
        model_dir = Path(PROJECT_ROOT) / 'ml_models' / 'trained_models' / 'ng'
        model_paths = sorted(model_dir.glob(f'{ver_tag}_seed*_multi_target_*.pkl'))
        if not model_paths:
            model_paths = sorted(model_dir.glob(f'{ver_tag}_multi_target_*.pkl'))
        model_paths = [str(p) for p in model_paths]
        if not model_paths:
            print(f"No models found for {args.version}")
            return
    else:
        parser.error("Either --models or --version is required")
```

- [ ] **Step 2: Update scoring_version in report JSON**

In `build_report_json` (line 149), make version dynamic:

```python
    return {
        'analysis_date': date,
        'scoring_version': f'{version}_ensemble' if version else 'ng_ensemble',
        'all_stocks_with_scores': stocks_list,
    }
```

- [ ] **Step 3: Update cache_table to be version-aware**

```python
    if args.version:
        from ml_models.ng.ng_schema import get_table_name
        cache_table = get_table_name(args.version)
    else:
        cache_table = scorers[0].cache_table
```

- [ ] **Step 4: Commit**

```bash
git add scripts/ensemble_predict.py
git commit -m "feat(ng104): ensemble_predict支持 --version自动发现seed模型"
```

---

### Task 8: Update __init__.py + Version Constants

**Files:**
- Modify: `ml_models/ng/__init__.py`
- Modify: `ml_models/ng/ng_trainer.py` (version constant)

- [ ] **Step 1: Add NG104 version constant**

In `ng_trainer.py`, add after `NG_VERSION = 'ng1.0.3'` (line 97):

```python
NG104_VERSION = 'ng1.0.4'
```

- [ ] **Step 2: Update __init__.py exports**

```python
"""Daily Selection NG — Next Generation trend-following factor model.

v1.0.4: Risk-adjusted labels, multi-seed ensemble, IC stability screening, signal smoothing.
"""

from .ng_trainer import (
    NGTrainer, ALL_FEATURE_NAMES, STOCK_FEATURE_NAMES, MARKET_FEATURE_NAMES,
    MONEYFLOW_FEATURE_NAMES, INTERACTION_FEATURE_NAMES,
    SMOOTHING_FEATURE_NAMES, NG104_STOCK_FEATURES, NG104_ALL_FEATURES,
    NG_VERSION, NG_V1_VERSION, NG104_VERSION,
)
from .ng_production_scorer import NGProductionScorer

__all__ = [
    'NGTrainer', 'NGProductionScorer',
    'ALL_FEATURE_NAMES', 'STOCK_FEATURE_NAMES', 'MARKET_FEATURE_NAMES',
    'MONEYFLOW_FEATURE_NAMES', 'INTERACTION_FEATURE_NAMES',
    'SMOOTHING_FEATURE_NAMES', 'NG104_STOCK_FEATURES', 'NG104_ALL_FEATURES',
    'NG_VERSION', 'NG_V1_VERSION', 'NG104_VERSION',
]
```

- [ ] **Step 3: Commit**

```bash
git add ml_models/ng/__init__.py ml_models/ng/ng_trainer.py
git commit -m "feat(ng104): 版本常量 + __init__导出更新"
```

---

### Task 9: Backfill Cache + IC Stability Screening

**Files:**
- No new files; use existing CLI tools

- [ ] **Step 1: Backfill ng104_feature_cache (2018-01-01 ~ today)**

Run:
```bash
python3 ml_models/ng/ng_cache_updater.py \
  --start-date 2018-01-01 \
  --end-date 2026-04-07 \
  --version ng1.0.4 \
  --penalty-power 1.5
```

Expected: Processes ~1500 trading dates, ~5000 stocks/date. This takes 1-3 hours.

- [ ] **Step 2: Verify cache has data**

Run:
```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('data_adapter/stock_data.db', timeout=30)
r = conn.execute('SELECT COUNT(*), COUNT(ra_label_10d), MIN(trade_date), MAX(trade_date) FROM ng104_feature_cache').fetchone()
print(f'Rows: {r[0]:,}, RA labels: {r[1]:,}, Date range: {r[2]} ~ {r[3]}')
# Verify non-zero RA labels
r2 = conn.execute('SELECT COUNT(*) FROM ng104_feature_cache WHERE ra_label_10d != 0 AND ra_label_10d IS NOT NULL').fetchone()
print(f'Non-zero RA labels: {r2[0]:,}')
conn.close()
"
```

Expected: Millions of rows, non-zero RA label count should be > 50% of total.

- [ ] **Step 3: Run IC stability analysis**

Run:
```bash
python3 scripts/ic_stability_analyzer.py \
  --cache-table ng104_feature_cache \
  --label ra_label_10d \
  --output reports/ic_stability_ng104.md
```

Expected: Report showing STABLE/FLIP/UNSTABLE classification for all ~75 features.

- [ ] **Step 4: Review IC report and decide final feature set**

Read the report, identify FLIP features. Update `NG104_STOCK_FEATURES` in `ng_trainer.py` to remove any FLIP features.

- [ ] **Step 5: Commit feature set update**

```bash
git add ml_models/ng/ng_trainer.py reports/ic_stability_ng104.md
git commit -m "feat(ng104): IC稳定性筛选结果 — 最终特征集确定"
```

---

### Task 10: Fast-Check Penalty Power Grid Search

**Files:**
- No new files; use existing fast-check infrastructure

- [ ] **Step 1: Run fast-check with penalty_power=0.0 (baseline)**

```bash
python3 ml_models/ng/ng_trainer.py \
  --version ng1.0.4 \
  --fast-check \
  --penalty-power 0.0 \
  --start-date 2020-01-01 \
  --purge-days 15
```

Record 10d ICIR output.

- [ ] **Step 2: Run fast-check with penalty_power=1.0**

```bash
python3 ml_models/ng/ng_trainer.py \
  --version ng1.0.4 \
  --fast-check \
  --penalty-power 1.0 \
  --start-date 2020-01-01 \
  --purge-days 15
```

- [ ] **Step 3: Run fast-check with penalty_power=1.5**

```bash
python3 ml_models/ng/ng_trainer.py \
  --version ng1.0.4 \
  --fast-check \
  --penalty-power 1.5 \
  --start-date 2020-01-01 \
  --purge-days 15
```

- [ ] **Step 4: Run fast-check with penalty_power=2.0**

```bash
python3 ml_models/ng/ng_trainer.py \
  --version ng1.0.4 \
  --fast-check \
  --penalty-power 2.0 \
  --start-date 2020-01-01 \
  --purge-days 15
```

- [ ] **Step 5: Compare results and select best penalty_power**

Compare 10d ICIR across all 4 runs. Select the penalty_power with highest ICIR. If 0.0 wins, fall back to excess labels (risk-adjusted provides no benefit).

- [ ] **Step 6: Commit grid search results**

```bash
git commit --allow-empty -m "docs(ng104): fast-check penalty_power网格搜索完成 — 最优值=X.X"
```

---

### Task 11: Multi-Seed Full Training

**Files:**
- No new files; use existing training infrastructure

- [ ] **Step 1: Train seed 42**

```bash
python3 ml_models/ng/ng_trainer.py \
  --version ng1.0.4 \
  --seed 42 \
  --penalty-power {BEST_PP} \
  --start-date 2020-01-01 \
  --purge-days 15
```

Expected: Model saved as `ng104_seed42_multi_target_*.pkl`

- [ ] **Step 2: Train seeds 123, 456, 789, 2024 (can be parallelized)**

Run each in a separate terminal or use `--seeds`:

```bash
python3 ml_models/ng/ng_trainer.py --version ng1.0.4 --seed 123 --penalty-power {BEST_PP} --start-date 2020-01-01 --purge-days 15
python3 ml_models/ng/ng_trainer.py --version ng1.0.4 --seed 456 --penalty-power {BEST_PP} --start-date 2020-01-01 --purge-days 15
python3 ml_models/ng/ng_trainer.py --version ng1.0.4 --seed 789 --penalty-power {BEST_PP} --start-date 2020-01-01 --purge-days 15
python3 ml_models/ng/ng_trainer.py --version ng1.0.4 --seed 2024 --penalty-power {BEST_PP} --start-date 2020-01-01 --purge-days 15
```

- [ ] **Step 3: Verify all 5 models exist**

```bash
ls -la ml_models/trained_models/ng/ng104_seed*
```

Expected: 5 .pkl files, each 40-70MB.

---

### Task 12: Evaluation — Ensemble Reports + North Star

**Files:**
- No new files; use existing evaluation tools

- [ ] **Step 1: Generate Pre-2020 ensemble reports**

```bash
python3 scripts/ensemble_predict.py \
  --version ng1.0.4 \
  --start-date 2018-04-02 \
  --end-date 2020-12-31 \
  --output-dir reports/daily_selection_ng104_ensemble_pre2020
```

- [ ] **Step 2: Verify reports have non-zero pred_10d**

```bash
python3 -c "
import json, os, glob
files = sorted(glob.glob('reports/daily_selection_ng104_ensemble_pre2020/analysis_data_*.json'))
print(f'Reports: {len(files)}')
if files:
    with open(files[0]) as f:
        d = json.load(f)
    nonzero = sum(1 for s in d['all_stocks_with_scores'] if float(s.get('pred_10d', 0) or 0) != 0)
    print(f'First report non-zero pred_10d: {nonzero}/{len(d[\"all_stocks_with_scores\"])}')
"
```

Expected: Non-zero count > 1000 (most stocks should have predictions).

- [ ] **Step 3: Run North Star evaluation**

```bash
python3 backtest/run_north_star_eval.py --backtest \
  --report-dir reports/daily_selection_ng104_ensemble_pre2020 \
  --label NG104-ENSEMBLE-PRE2020 \
  --top-n 5 --focus-days 10 --rank-field score
```

Record: V5.2 score, annual return, excess return, MaxDD, Sharpe, turnover.

- [ ] **Step 4: Compare vs ng1.0.3 baseline**

| Metric | ng1.0.3 | ng1.0.4 ensemble | Target |
|--------|---------|-------------------|--------|
| MaxDD | ? | ? | < 10% |
| Excess Return | +24.8% | ? | >= 15% |
| Sharpe | 0.45 | ? | >= 0.60 |
| Turnover | ~43x | ? | <= 30x |

- [ ] **Step 5: Commit evaluation results**

```bash
git add -A
git commit -m "eval(ng104): 双向评估结果 — MaxDD=X%, Sharpe=Y, 换手=Zx"
```

---

### Task 13: Wiki + Documentation Update

**Files:**
- Modify: `docs/wiki/models/ng-series.md`
- Modify: `docs/wiki/log.md`

- [ ] **Step 1: Update ng-series.md with ng1.0.4 entry**

Add ng1.0.4 section documenting: feature set, label design, penalty_power, multi-seed architecture, IC screening results, and evaluation metrics.

- [ ] **Step 2: Update log.md with iteration entry**

Add date-stamped entry to `docs/wiki/log.md`.

- [ ] **Step 3: Commit**

```bash
git add docs/wiki/models/ng-series.md docs/wiki/log.md
git commit -m "docs(ng104): wiki更新 — ng1.0.4设计+评估结果"
```

---

## Dependency Graph

```
Task 1 (schema) ──┐
Task 2 (features) ─┼─→ Task 3 (cache updater) → Task 9 (backfill + IC screen)
                   │                                       │
Task 4 (trainer) ──┘                     Task 6 (IC analyzer)
                                                           │
Task 5 (scorer) ──────────────────→ Task 10 (fast-check grid) → Task 11 (5-seed training)
                                                                          │
Task 7 (ensemble_predict) ──────────────────────────→ Task 12 (evaluation)
                                                                          │
Task 8 (init + version) ──────────────────────────────→ Task 13 (docs)
```

Tasks 1, 2, 4, 5, 6, 7, 8 can be developed in parallel.
Task 3 depends on Tasks 1 + 2.
Task 9 depends on Task 3 + 6.
Task 10 depends on Task 9 + 4.
Task 11 depends on Task 10.
Task 12 depends on Task 11 + 7.
Task 13 depends on Task 12.
