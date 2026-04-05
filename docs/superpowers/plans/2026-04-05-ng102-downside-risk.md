# NG v1.0.2 Downside Risk Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a downside risk prediction target to NG, discounting high-risk stocks in the composite score to reduce MaxDD from -27.2% toward -15%.

**Architecture:** Train a separate downside_10d model (LightGBM) using the same WF windows as the main ensemble. At scoring time, subtract `lambda_risk * pred_downside_10d` from the return composite. The downside model trains on `max(0, -excess_return_10d)` — a zero-inflated target where non-negative excess returns produce zero labels. Features are identical (69 factors).

**Tech Stack:** LightGBM, existing V485 ensemble machinery, SQLite cache tables.

---

### Task 1: Schema — add ng102_feature_cache table

**Files:**
- Modify: `ml_models/ng/ng_schema.py`

- [ ] **Step 1: Add ng1.0.2 to VERSION_TABLE_MAP and update DEFAULT_VERSION**

```python
# In ng_schema.py, update these:
VERSION_TABLE_MAP = {
    'ng1.0.0': 'ng_feature_cache',
    'ng1.0.1': 'ng101_feature_cache',
    'ng1.0.2': 'ng102_feature_cache',
}

DEFAULT_VERSION = 'ng1.0.2'
```

- [ ] **Step 2: Add downside_10d column to ng102 schema**

Override `_schema_sql` to support a version parameter, or create a v102-specific schema function. The ng102 table adds one column `downside_10d REAL` after `label_15d`:

```python
def _schema_sql(table_name: str, version: str = None) -> str:
    """Generate CREATE TABLE SQL. v1.0.2+ includes downside_10d column."""
    extra_col = "\n    downside_10d REAL," if version and version >= 'ng1.0.2' else ""
    return f"""
CREATE TABLE IF NOT EXISTS {table_name} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code VARCHAR(10) NOT NULL,
    trade_date DATE NOT NULL,
    features_json TEXT NOT NULL,
    label_3d REAL,
    label_5d REAL,
    label_10d REAL,
    label_15d REAL,{extra_col}
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

CREATE INDEX IF NOT EXISTS idx_{table_name}_date ON {table_name}(trade_date);
CREATE INDEX IF NOT EXISTS idx_{table_name}_code_date ON {table_name}(code, trade_date);
"""
```

Update `create_table` to pass version through:
```python
def create_table(db_path: str = None, version: str = None):
    path = db_path or DB_PATH
    ver = version or DEFAULT_VERSION
    table_name = get_table_name(ver)
    with sqlite3.connect(path, timeout=30) as conn:
        conn.executescript(_schema_sql(table_name, version=ver))
    print(f"{table_name} table ready: {path}")
```

- [ ] **Step 3: Verify table creation**

```bash
python3 -c "
from ml_models.ng.ng_schema import create_table, get_table_name
print(get_table_name('ng1.0.2'))  # ng102_feature_cache
create_table(version='ng1.0.2')
"
```

Then verify the column exists:
```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('data_adapter/stock_data.db')
cur = conn.execute('PRAGMA table_info(ng102_feature_cache)')
cols = [r[1] for r in cur.fetchall()]
print(cols)
assert 'downside_10d' in cols, 'downside_10d column missing!'
print('OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add ml_models/ng/ng_schema.py
git commit -m "feat(ng): ng102 schema with downside_10d column"
```

---

### Task 2: Cache updater — compute downside_10d label

**Files:**
- Modify: `ml_models/ng/ng_cache_updater.py`

The downside label is derived from the industry excess label_10d that we already compute:
```python
downside_10d = max(0, -excess_return_10d)
```

- [ ] **Step 1: Update version and imports**

Change default version in `NGCacheUpdater.__init__`:
```python
def __init__(self, db_path: str = None, version: str = 'ng1.0.2'):
```

- [ ] **Step 2: Compute downside_10d in `_convert_labels_to_excess`**

After computing `excess_labels`, add `downside_10d` for each stock:

```python
# At the end of _convert_labels_to_excess, after the excess loop:
for sid, labs in excess_labels.items():
    label_10d = labs.get('label_10d', np.nan)
    if not np.isnan(label_10d):
        labs['downside_10d'] = max(0.0, -label_10d)
    else:
        labs['downside_10d'] = np.nan
```

- [ ] **Step 3: Update INSERT statement to include downside_10d**

The insert row tuple gains one more value after `label_15d`. Update the INSERT SQL:

```python
conn.executemany(
    f'''INSERT OR REPLACE INTO {self.table_name}
       (code, trade_date, features_json,
        label_3d, label_5d, label_10d, label_15d, downside_10d,
        market_return_5d, market_return_20d, market_volatility_20d,
        market_breadth, market_new_high_ratio, northbound_flow_5d,
        market_volume_ratio, market_drawdown, vix_proxy,
        market_momentum_diff)
       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
    insert_rows
)
```

And update the tuple construction to insert `_to_sql(stock_labels.get('downside_10d'))` after `label_15d`:

```python
insert_rows.append((
    data['code'],
    date,
    features_json,
    _to_sql(stock_labels.get('label_3d')),
    _to_sql(stock_labels.get('label_5d')),
    _to_sql(stock_labels.get('label_10d')),
    _to_sql(stock_labels.get('label_15d')),
    _to_sql(stock_labels.get('downside_10d')),  # NEW
    _to_sql(market_feats.get('market_return_5d')),
    # ... rest unchanged
))
```

- [ ] **Step 4: Smoke test on one date**

```bash
python3 ml_models/ng/ng_cache_updater.py --date 2025-06-13 --version ng1.0.2
python3 -c "
import sqlite3
conn = sqlite3.connect('data_adapter/stock_data.db')
cur = conn.execute('SELECT downside_10d, label_10d FROM ng102_feature_cache WHERE trade_date=\"2025-06-13\" LIMIT 5')
for r in cur.fetchall():
    print(f'downside={r[0]:.4f}, label_10d={r[1]:.4f}, check: downside==max(0,-label)? {abs(r[0] - max(0,-r[1])) < 1e-6}')
"
```

Expected: downside_10d = max(0, -label_10d) for each row.

- [ ] **Step 5: Commit**

```bash
git add ml_models/ng/ng_cache_updater.py
git commit -m "feat(ng): compute downside_10d label in ng102 cache"
```

---

### Task 3: Trainer — add downside_10d training target

**Files:**
- Modify: `ml_models/ng/ng_trainer.py`

Key design: V485's `_train_single_window` is hardcoded for 4 targets. Instead of modifying V485, we train a standalone LightGBM model for downside_10d in a separate pass after the main WF training completes, using the same train/test splits.

- [ ] **Step 1: Update version constants**

```python
NG_V1_VERSION = 'ng1.0.0'
NG_VERSION = 'ng1.0.2'
```

- [ ] **Step 2: Update load_data to include downside_10d**

In `load_data`, add downside_10d to the SELECT query and result DataFrame:

```python
query = f"""
SELECT code, trade_date, features_json,
       label_3d, label_5d, label_10d, label_15d,
       downside_10d,
       market_return_5d, ...
FROM {self.cache_table}
WHERE label_5d IS NOT NULL {date_filter}
ORDER BY trade_date, code
"""
```

After building result DataFrame, add:
```python
result['downside_10d'] = pd.to_numeric(df_raw.get('downside_10d'), errors='coerce').values
```

- [ ] **Step 3: Update prepare_features to return downside_10d**

In `prepare_features`, after building y_15d, add:
```python
y_downside = pd.to_numeric(df['downside_10d'], errors='coerce').fillna(0.0).values
```

Change return to: `return X, y_3d, y_5d, y_10d, y_15d, y_downside, df`

- [ ] **Step 4: Add `_train_downside_model` method**

```python
def _train_downside_model(self, X_train, y_train, X_val, y_val):
    """Train a standalone LightGBM for downside_10d prediction."""
    import lightgbm as lgb
    
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 200,
        'verbose': -1,
    }
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    
    model = lgb.train(
        params, dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    
    return model
```

- [ ] **Step 5: Integrate downside training into walk_forward_train**

After the main `super().walk_forward_train()` call returns model_data and history, add a second pass that trains downside models using the same WF window splits. Store the downside model in `model_data['downside_model']`:

```python
# After main training, before model save
# Train downside model using same WF windows
if not model_data.get('fast_check'):
    logger.info("Training downside_10d model...")
    df = self.load_data(start_date=start_date, end_date=end_date)
    X, y_3d, y_5d, y_10d, y_15d, y_downside, df = self.prepare_features(df)
    
    # Use the last WF window's train/val split
    dates = df['trade_date'].values
    unique_dates = np.unique(dates)
    n = len(unique_dates)
    val_start = n - test_days - val_days
    train_mask = np.isin(dates, unique_dates[:val_start])
    val_mask = np.isin(dates, unique_dates[val_start:val_start + val_days])
    
    downside_model = self._train_downside_model(
        X[train_mask], y_downside[train_mask],
        X[val_mask], y_downside[val_mask],
    )
    model_data['downside_model'] = downside_model
    
    # Compute downside IC on test set
    test_mask = np.isin(dates, unique_dates[val_start + val_days:])
    if test_mask.sum() > 0:
        pred_ds = downside_model.predict(X[test_mask])
        from scipy.stats import spearmanr
        ic, _ = spearmanr(pred_ds, y_downside[test_mask])
        logger.info(f"  Downside 10d OOS IC: {ic:.4f}")
        model_data['downside_ic'] = float(ic)
```

- [ ] **Step 6: Store lambda_risk in model metadata**

```python
model_data['lambda_risk'] = 0.5  # default, tunable
model_data['ng_innovations']['downside_model'] = True
model_data['ng_innovations']['lambda_risk'] = 0.5
```

- [ ] **Step 7: CLI — add --lambda-risk and --asymmetric-loss flags**

```python
parser.add_argument('--lambda-risk', type=float, default=0.5,
                    help='Risk discount factor (default: 0.5)')
parser.add_argument('--asymmetric-loss', action='store_true',
                    help='Use asymmetric MSE for LightGBM return targets')
```

- [ ] **Step 8: Commit**

```bash
git add ml_models/ng/ng_trainer.py
git commit -m "feat(ng): train downside_10d model in ng1.0.2"
```

---

### Task 4: Scorer — risk-discounted composite

**Files:**
- Modify: `ml_models/ng/ng_production_scorer.py`

- [ ] **Step 1: Load downside model from pkl**

In `_load_model`, after loading main models, add:
```python
self.downside_model = model_data.get('downside_model')
self.lambda_risk = model_data.get('lambda_risk', 0.5)
```

- [ ] **Step 2: Add downside prediction to scoring**

In `predict_scores`, after computing `combined_pred` for return targets, add risk discount:

```python
# After: combined_pred = sum of return predictions
# Risk discount
if self.downside_model is not None:
    pred_downside = self.downside_model.predict(feature_matrix)
    pred_downside = np.clip(pred_downside, 0, None)  # downside is non-negative
    combined_pred = combined_pred - self.lambda_risk * pred_downside
```

Also add `pred_downside_10d` to the output dict per stock:
```python
all_results[code] = {
    ...existing fields...,
    'pred_downside_10d': float(pred_downside[i]) if self.downside_model else 0.0,
}
```

- [ ] **Step 3: Same changes for `predict_scores_from_preloaded`**

Mirror the risk discount logic in the preloaded scoring path.

- [ ] **Step 4: Commit**

```bash
git add ml_models/ng/ng_production_scorer.py
git commit -m "feat(ng): risk-discounted composite scoring in ng1.0.2"
```

---

### Task 5: Batch report generator + version wiring

**Files:**
- Modify: `backtest/batch_generate_v395_reports.py`
- Modify: `ml_models/ng/__init__.py`

- [ ] **Step 1: Add ng1.0.2 to batch generator choices**

In the argparse choices list, add `'ng1.0.2'`. In the table auto-detection logic, add:
```python
'ng102_feature_cache' if args.version == 'ng1.0.2'
```

In the scorer initialization section:
```python
elif args.version in ('ng1.0.0', 'ng1.0.1', 'ng1.0.2'):
    from ml_models.ng.ng_production_scorer import NGProductionScorer
    scorer = NGProductionScorer(model_path=getattr(args, 'model_path', None))
```

- [ ] **Step 2: Update __init__.py version**

```python
NG_VERSION = 'ng1.0.2'
```

- [ ] **Step 3: Commit**

```bash
git add backtest/batch_generate_v395_reports.py ml_models/ng/__init__.py
git commit -m "feat(ng): wire ng1.0.2 into batch reports and version exports"
```

---

### Task 6: Backfill ng102_feature_cache

**Files:** No code changes, operational step.

- [ ] **Step 1: Copy features from ng101 and add downside labels**

Since features_json is identical between ng101 and ng102, we can copy from ng101 and only recompute labels. But the simplest approach is running the full updater:

```bash
nohup python3 ml_models/ng/ng_cache_updater.py \
    --start-date 2018-01-01 --end-date 2026-04-03 --version ng1.0.2 \
    > logs/ng102_backfill.log 2>&1 &
```

Expected: ~35 minutes, ~3.2M rows.

- [ ] **Step 2: Verify backfill**

```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('data_adapter/stock_data.db')
cur = conn.execute('SELECT COUNT(*), COUNT(DISTINCT trade_date), MIN(trade_date), MAX(trade_date) FROM ng102_feature_cache')
print(cur.fetchone())
# Check downside_10d distribution
cur = conn.execute('SELECT AVG(downside_10d), MAX(downside_10d) FROM ng102_feature_cache WHERE downside_10d > 0')
print('Avg downside (when >0):', cur.fetchone())
"
```

---

### Task 7: Fast-check validation

**Files:** No code changes, validation step.

- [ ] **Step 1: Run fast-check**

```bash
python3 ml_models/ng/ng_trainer.py \
    --start-date 2020-01-01 --purge-days 15 --fast-check \
    --lambda-risk 0.5
```

Expected: 2 WF windows, ~5 minutes. Check output for:
- Downside 10d OOS IC > 0 (positive means model predicts downside correctly)
- Return ICIR still reasonable (> 0.5)

- [ ] **Step 2: Lambda grid search via fast-check**

Run 4 fast-checks with different lambda values:
```bash
for lambda in 0.3 0.5 0.7 1.0; do
    python3 ml_models/ng/ng_trainer.py \
        --start-date 2020-01-01 --purge-days 15 --fast-check \
        --lambda-risk $lambda 2>&1 | grep -E "ICIR|Downside|lambda"
done
```

Pick the lambda with best combined ICIR.

---

### Task 8: Full training + V5 comparison

**Files:** No code changes, execution step.

- [ ] **Step 1: Full WF training with optimal lambda**

```bash
nohup python3 ml_models/ng/ng_trainer.py \
    --start-date 2020-01-01 --purge-days 15 \
    --lambda-risk <optimal_lambda> \
    > logs/ng102_training.log 2>&1 &
```

- [ ] **Step 2: Generate reports**

```bash
python3 backtest/batch_generate_v395_reports.py \
    --version ng1.0.2 --start-date auto --end-date auto \
    --output-dir reports/daily_selection_ng102
```

- [ ] **Step 3: V5 comparison**

```bash
echo "=== ng1.0.1 ==="
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_ng101 \
    --label NG101 --top-n 10 --focus-days 10 --rank-field composite 2>&1 | \
    grep -E "V5|加权|L[1-6].*权重"

echo "=== ng1.0.2 ==="
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_ng102 \
    --label NG102 --top-n 10 --focus-days 10 --rank-field composite 2>&1 | \
    grep -E "V5|加权|L[1-6].*权重"
```

Target: V5 加权 > 75% (S级), L3 风控 > 60%.

- [ ] **Step 4: Commit final results**

```bash
git add -A
git commit -m "feat(ng): v1.0.2 完成 — 多目标下行风险, V5=XX%"
git push
```
