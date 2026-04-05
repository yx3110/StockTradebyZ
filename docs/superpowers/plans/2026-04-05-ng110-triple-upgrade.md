# NG v1.1.0 三方向联合升级实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 ng1.0.2 基础上通过资金流因子+风格残差标签+WF升级三个方向提升模型性能，目标 V5.2 > 76%

**Architecture:** 所有改动在现有 NG 代码上增量修改（ng_schema → ng_feature_calculator → ng_cache_updater → ng_trainer → ng_production_scorer），通过 CLI 开关控制每个方向的开启/关闭，实现独立 fast-check 验证后合并

**Tech Stack:** Python 3, SQLite, LightGBM, XGBoost, CatBoost, scikit-learn, numpy, pandas, Tushare API

---

## 文件清单

| 文件 | 操作 | 职责 |
|------|------|------|
| `ml_models/ng/ng_schema.py` | 修改 | 新增 moneyflow_daily 表 + ng110_feature_cache 表(含 label_raw 列) |
| `ml_models/ng/ng_feature_calculator.py` | 修改 | 新增8资金流因子 + 8交互因子计算函数 |
| `ml_models/ng/ng_cache_updater.py` | 修改 | 新增 moneyflow 数据获取 + 残差标签计算 + ng1.1.0 版本支持 |
| `ml_models/ng/ng_trainer.py` | 修改 | WF窗口参数化 + 市况加权 + 交互因子IC筛选 + CLI开关 |
| `ml_models/ng/ng_production_scorer.py` | 修改 | 支持 ng110 表 + 动态因子列表(含交互因子) |
| `fetch_data/quick_daily_update.py` | 修改 | 新增 moneyflow_daily 日常更新步骤 |
| `backtest/batch_generate_v395_reports.py` | 修改 | 注册 ng1.1.0 版本 |

---

### Task 1: Schema 扩展 — moneyflow_daily + ng110_feature_cache

**Files:**
- Modify: `ml_models/ng/ng_schema.py`

- [ ] **Step 1: 在 VERSION_TABLE_MAP 新增 ng1.1.0 映射**

在 `ng_schema.py` 的 `VERSION_TABLE_MAP` 字典中添加:

```python
VERSION_TABLE_MAP = {
    'ng1.0.0': 'ng_feature_cache',
    'ng1.0.1': 'ng101_feature_cache',
    'ng1.0.2': 'ng102_feature_cache',
    'ng1.1.0': 'ng110_feature_cache',
}
```

- [ ] **Step 2: 修改 _schema_sql 支持 ng1.1.0 的 label_raw 列**

修改 `_schema_sql` 函数，在 `ver >= 'ng1.1.0'` 时添加 `label_raw_*` 列:

```python
def _schema_sql(table_name: str, version: str = None) -> str:
    ver = version or DEFAULT_VERSION
    extra_cols = ''
    if ver >= 'ng1.0.2':
        extra_cols += '\n    downside_10d REAL,'
    if ver >= 'ng1.1.0':
        extra_cols += '\n    label_raw_3d REAL,'
        extra_cols += '\n    label_raw_5d REAL,'
        extra_cols += '\n    label_raw_10d REAL,'
        extra_cols += '\n    label_raw_15d REAL,'
    return f"""
CREATE TABLE IF NOT EXISTS {table_name} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code VARCHAR(10) NOT NULL,
    trade_date DATE NOT NULL,
    features_json TEXT NOT NULL,
    label_3d REAL,
    label_5d REAL,
    label_10d REAL,
    label_15d REAL,{extra_cols}
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

- [ ] **Step 3: 新增 moneyflow_daily 表创建函数**

在 `ng_schema.py` 文件末尾 `if __name__` 之前添加:

```python
MONEYFLOW_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS moneyflow_daily (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code VARCHAR(10) NOT NULL,
    trade_date DATE NOT NULL,
    buy_sm_amount REAL,
    sell_sm_amount REAL,
    buy_md_amount REAL,
    sell_md_amount REAL,
    buy_lg_amount REAL,
    sell_lg_amount REAL,
    buy_elg_amount REAL,
    sell_elg_amount REAL,
    net_mf_amount REAL,
    UNIQUE(code, trade_date)
);
CREATE INDEX IF NOT EXISTS idx_moneyflow_daily_date ON moneyflow_daily(trade_date);
CREATE INDEX IF NOT EXISTS idx_moneyflow_daily_code_date ON moneyflow_daily(code, trade_date);
"""


def create_moneyflow_table(db_path: str = None):
    """Create moneyflow_daily table (if not exists)."""
    path = db_path or DB_PATH
    with sqlite3.connect(path, timeout=30) as conn:
        conn.executescript(MONEYFLOW_SCHEMA_SQL)
    print(f"moneyflow_daily table ready: {path}")
```

- [ ] **Step 4: 验证 schema**

Run: `python3 -c "from ml_models.ng.ng_schema import create_table, create_moneyflow_table, get_table_name; create_table(version='ng1.1.0'); create_moneyflow_table(); print(get_table_name('ng1.1.0'))"`
Expected: `ng110_feature_cache table ready: ...`, `moneyflow_daily table ready: ...`, `ng110_feature_cache`

- [ ] **Step 5: Commit**

```bash
git add ml_models/ng/ng_schema.py
git commit -m "feat(ng): ng110 schema — moneyflow_daily + label_raw columns"
```

---

### Task 2: Moneyflow 数据获取 — fetch_data 集成

**Files:**
- Modify: `fetch_data/quick_daily_update.py`
- Modify: `ml_models/ng/ng_cache_updater.py` (添加 moneyflow 获取方法)

- [ ] **Step 1: 在 ng_cache_updater.py 添加 moneyflow 数据获取方法**

在 `NGCacheUpdater` 类中添加 `_fetch_and_store_moneyflow` 方法（放在 `_connect` 方法之后）:

```python
def _fetch_and_store_moneyflow(self, conn: sqlite3.Connection, date: str) -> int:
    """Fetch moneyflow data from Tushare and store in moneyflow_daily table.
    Returns number of rows inserted."""
    import json as _json
    try:
        with open(str(Path(__file__).resolve().parent.parent.parent / 'config.json')) as f:
            config = _json.load(f)
        import tushare as ts
        pro = ts.pro_api(config['tushare']['token'])
    except Exception as e:
        print(f"    WARN: Cannot init Tushare: {e}")
        return 0

    date_str = date.replace('-', '')

    # Check if data already exists for this date
    existing = conn.execute(
        "SELECT COUNT(*) FROM moneyflow_daily WHERE trade_date = ?", (date,)
    ).fetchone()[0]
    if existing > 0:
        return existing

    try:
        df = pro.moneyflow(trade_date=date_str)
        if df is None or len(df) == 0:
            return 0

        rows = []
        for _, row in df.iterrows():
            code = row['ts_code']
            rows.append((
                code, date,
                float(row.get('buy_sm_amount') or 0),
                float(row.get('sell_sm_amount') or 0),
                float(row.get('buy_md_amount') or 0),
                float(row.get('sell_md_amount') or 0),
                float(row.get('buy_lg_amount') or 0),
                float(row.get('sell_lg_amount') or 0),
                float(row.get('buy_elg_amount') or 0),
                float(row.get('sell_elg_amount') or 0),
                float(row.get('net_mf_amount') or 0),
            ))

        conn.executemany("""
            INSERT OR REPLACE INTO moneyflow_daily
            (code, trade_date, buy_sm_amount, sell_sm_amount,
             buy_md_amount, sell_md_amount, buy_lg_amount, sell_lg_amount,
             buy_elg_amount, sell_elg_amount, net_mf_amount)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        conn.commit()
        return len(rows)
    except Exception as e:
        print(f"    WARN: moneyflow fetch failed for {date}: {e}")
        return 0
```

- [ ] **Step 2: 添加 moneyflow 历史数据加载方法**

在 `NGCacheUpdater` 类中添加 `_load_moneyflow_data` 方法（用于因子计算时从已存储数据读取）:

```python
def _load_moneyflow_data(self, conn: sqlite3.Connection, date: str,
                          security_ids: List[int], n_days: int = 20
                          ) -> Dict[str, List[dict]]:
    """Load recent moneyflow data from moneyflow_daily table.
    Returns {code: [{date, buy_sm_amount, ..., net_mf_amount}, ...]}.
    """
    # Map security_id → code
    sid_to_code = {}
    for sid in security_ids:
        row = conn.execute(
            "SELECT code FROM securities WHERE id = ?", (sid,)
        ).fetchone()
        if row:
            sid_to_code[sid] = row[0]

    if not sid_to_code:
        return {}

    codes = list(sid_to_code.values())
    result: Dict[str, List[dict]] = {}

    chunk_size = 900
    for i in range(0, len(codes), chunk_size):
        chunk = codes[i:i + chunk_size]
        placeholders = ','.join('?' * len(chunk))
        query = f"""
        SELECT code, trade_date, buy_sm_amount, sell_sm_amount,
               buy_md_amount, sell_md_amount, buy_lg_amount, sell_lg_amount,
               buy_elg_amount, sell_elg_amount, net_mf_amount
        FROM moneyflow_daily
        WHERE code IN ({placeholders})
          AND trade_date <= ?
        ORDER BY trade_date DESC
        LIMIT ?
        """
        rows = conn.execute(query, chunk + [date, len(chunk) * n_days]).fetchall()
        for r in rows:
            code = r[0]
            if code not in result:
                result[code] = []
            result[code].append({
                'trade_date': r[1],
                'buy_sm_amount': r[2] or 0, 'sell_sm_amount': r[3] or 0,
                'buy_md_amount': r[4] or 0, 'sell_md_amount': r[5] or 0,
                'buy_lg_amount': r[6] or 0, 'sell_lg_amount': r[7] or 0,
                'buy_elg_amount': r[8] or 0, 'sell_elg_amount': r[9] or 0,
                'net_mf_amount': r[10] or 0,
            })

    # Sort by date ascending
    for code in result:
        result[code].sort(key=lambda x: x['trade_date'])

    return result
```

- [ ] **Step 3: 在 quick_daily_update.py 新增 moneyflow 更新步骤**

在 `quick_daily_update()` 函数中，步骤12（NG特征缓存）之前添加 moneyflow 获取:

```python
# 11.4. 更新资金流数据 (NG v1.1.0)
logger.info("【步骤11.5/15】更新资金流数据...")
try:
    from ml_models.ng.ng_schema import create_moneyflow_table, DB_PATH as NG_DB_PATH
    import sqlite3
    create_moneyflow_table(NG_DB_PATH)
    
    import json as _json
    with open(os.path.join(os.path.dirname(__file__), '..', 'config.json')) as f:
        cfg = _json.load(f)
    import tushare as ts
    pro = ts.pro_api(cfg['tushare']['token'])
    
    date_str = date.replace('-', '')
    df_mf = pro.moneyflow(trade_date=date_str)
    if df_mf is not None and len(df_mf) > 0:
        conn_mf = sqlite3.connect(NG_DB_PATH, timeout=30)
        rows_mf = []
        for _, row in df_mf.iterrows():
            rows_mf.append((
                row['ts_code'], date,
                float(row.get('buy_sm_amount') or 0),
                float(row.get('sell_sm_amount') or 0),
                float(row.get('buy_md_amount') or 0),
                float(row.get('sell_md_amount') or 0),
                float(row.get('buy_lg_amount') or 0),
                float(row.get('sell_lg_amount') or 0),
                float(row.get('buy_elg_amount') or 0),
                float(row.get('sell_elg_amount') or 0),
                float(row.get('net_mf_amount') or 0),
            ))
        conn_mf.executemany("""
            INSERT OR REPLACE INTO moneyflow_daily
            (code, trade_date, buy_sm_amount, sell_sm_amount,
             buy_md_amount, sell_md_amount, buy_lg_amount, sell_lg_amount,
             buy_elg_amount, sell_elg_amount, net_mf_amount)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows_mf)
        conn_mf.commit()
        conn_mf.close()
        stats['moneyflow'] = len(rows_mf)
        logger.info(f"  资金流数据更新完成: {len(rows_mf)} 条")
    else:
        stats['moneyflow'] = 0
        logger.info("  无资金流数据")
except Exception as e:
    logger.warning(f"  资金流数据更新失败: {e}")
    stats['moneyflow'] = 0
```

在统计报告部分添加一行: `logger.info(f"资金流数据: {stats.get('moneyflow', 0):,} 条")`

- [ ] **Step 4: 验证 moneyflow 获取**

Run: `python3 -c "
from ml_models.ng.ng_schema import create_moneyflow_table, DB_PATH
create_moneyflow_table()
import sqlite3
conn = sqlite3.connect(DB_PATH, timeout=30)
print('Table exists:', conn.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name='moneyflow_daily'\").fetchone())
conn.close()
"`
Expected: `Table exists: ('moneyflow_daily',)`

- [ ] **Step 5: Commit**

```bash
git add ml_models/ng/ng_cache_updater.py fetch_data/quick_daily_update.py
git commit -m "feat(ng): moneyflow数据获取 — Tushare API集成 + daily update"
```

---

### Task 3: 资金流因子计算 (8个新因子)

**Files:**
- Modify: `ml_models/ng/ng_feature_calculator.py`

- [ ] **Step 1: 添加 compute_moneyflow_features 函数**

在 `ng_feature_calculator.py` 末尾（`compute_residual_features` 之后）添加:

```python
def compute_moneyflow_features(
    mf_rows: list,  # List[dict] with keys: net_mf_amount, buy_lg_amount, sell_lg_amount, buy_elg_amount, sell_elg_amount, buy_sm_amount, sell_sm_amount
    amounts: np.ndarray,  # Daily trading amounts (for normalization)
    price_changes: np.ndarray,  # Daily price change pct (for divergence)
) -> Dict[str, float]:
    """
    Compute 8 money flow factors from moneyflow_daily data.
    
    Args:
        mf_rows: Recent moneyflow records sorted by date ascending (up to 20 days)
        amounts: Daily trading amounts for the stock (same period)
        price_changes: Daily price change percentages
    
    Returns dict with 8 factors.
    """
    result = {
        'net_mf_ratio_5d': np.nan,
        'big_order_ratio': np.nan,
        'big_order_trend_5d': np.nan,
        'small_vs_big_divergence': np.nan,
        'mf_concentration': np.nan,
        'mf_momentum_10d': np.nan,
        'northbound_stock_5d': np.nan,  # placeholder, filled externally
        'mf_volume_divergence': np.nan,
    }
    
    if not mf_rows or len(mf_rows) < 3:
        return result
    
    n = len(mf_rows)
    
    # Extract arrays
    net_mf = np.array([r['net_mf_amount'] for r in mf_rows])
    big_buy = np.array([r['buy_lg_amount'] + r['buy_elg_amount'] for r in mf_rows])
    big_sell = np.array([r['sell_lg_amount'] + r['sell_elg_amount'] for r in mf_rows])
    big_net = big_buy - big_sell
    sm_buy = np.array([r['buy_sm_amount'] for r in mf_rows])
    sm_sell = np.array([r['sell_sm_amount'] for r in mf_rows])
    sm_net = sm_buy - sm_sell
    total_amount = np.array([
        r['buy_sm_amount'] + r['sell_sm_amount'] +
        r['buy_md_amount'] + r['sell_md_amount'] +
        r['buy_lg_amount'] + r['sell_lg_amount'] +
        r['buy_elg_amount'] + r['sell_elg_amount']
        for r in mf_rows
    ])
    elg_amount = np.array([r['buy_elg_amount'] + r['sell_elg_amount'] for r in mf_rows])
    sm_md_amount = np.array([
        r['buy_sm_amount'] + r['sell_sm_amount'] +
        r['buy_md_amount'] + r['sell_md_amount']
        for r in mf_rows
    ])
    
    # 1. net_mf_ratio_5d: 5日净资金流入 / 5日成交额
    if n >= 5:
        amt_sum = np.sum(amounts[-5:]) if len(amounts) >= 5 else np.sum(amounts)
        result['net_mf_ratio_5d'] = float(np.sum(net_mf[-5:]) / (amt_sum + 1e-8))
    
    # 2. big_order_ratio: 今日(大单+特大单净买) / 今日总成交额
    if total_amount[-1] > 1e-8:
        result['big_order_ratio'] = float(big_net[-1] / total_amount[-1])
    
    # 3. big_order_trend_5d: 5日大单净买入的线性回归斜率
    if n >= 5:
        result['big_order_trend_5d'] = float(_linreg_slope(big_net[-5:]))
    
    # 4. small_vs_big_divergence: 散户vs主力分歧 (5日平均)
    if n >= 5:
        signs = np.sign(sm_net[-5:]) * np.sign(big_net[-5:])
        result['small_vs_big_divergence'] = float(np.mean(signs))
    
    # 5. mf_concentration: 特大单占比 / (小单+中单占比)
    if total_amount[-1] > 1e-8 and sm_md_amount[-1] > 1e-8:
        elg_ratio = elg_amount[-1] / total_amount[-1]
        sm_md_ratio = sm_md_amount[-1] / total_amount[-1]
        result['mf_concentration'] = float(elg_ratio / (sm_md_ratio + 1e-8))
    
    # 6. mf_momentum_10d: 净资金流MA5 - MA10
    if n >= 10:
        ma5_mf = np.mean(net_mf[-5:])
        ma10_mf = np.mean(net_mf[-10:])
        denom = np.std(net_mf[-10:]) + 1e-8
        result['mf_momentum_10d'] = float((ma5_mf - ma10_mf) / denom)
    elif n >= 5:
        result['mf_momentum_10d'] = float(np.mean(net_mf[-5:]) / (np.std(net_mf) + 1e-8))
    
    # 7. northbound_stock_5d: filled externally (set to 0 here as default)
    result['northbound_stock_5d'] = 0.0
    
    # 8. mf_volume_divergence: sign(净资金流) × sign(涨跌幅) 的5日均值
    if n >= 5 and len(price_changes) >= 5:
        mf_sign = np.sign(net_mf[-5:])
        price_sign = np.sign(price_changes[-5:])
        result['mf_volume_divergence'] = float(np.mean(mf_sign * price_sign))
    
    return result
```

- [ ] **Step 2: 添加 compute_interaction_features 函数**

```python
def compute_interaction_features(
    stock_feats: Dict[str, float],
    mf_feats: Dict[str, float],
    industry_feats: Dict[str, float],
    residual_feats: Dict[str, float],
    cs_rank_feats: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute 8 candidate interaction factors.
    IC screening will select the best ones during training.
    
    Returns dict with ix_* prefixed keys.
    """
    def _safe_mul(a, b):
        if np.isnan(a) or np.isnan(b):
            return np.nan
        return float(a * b)
    
    def _safe_div(a, b):
        if np.isnan(a) or np.isnan(b) or abs(b) < 1e-8:
            return np.nan
        return float(a / b)
    
    return {
        'ix_vol_pullback': _safe_mul(
            stock_feats.get('volume_ratio_5d', np.nan),
            stock_feats.get('pullback_to_ma20', np.nan)),
        'ix_big_trend': _safe_mul(
            mf_feats.get('big_order_ratio', np.nan),
            stock_feats.get('trend_strength_20d', np.nan)),
        'ix_rsi_mf': _safe_mul(
            stock_feats.get('rsi_14', np.nan),
            mf_feats.get('mf_momentum_10d', np.nan)),
        'ix_ind_big': _safe_mul(
            industry_feats.get('industry_relative_strength', np.nan),
            mf_feats.get('big_order_ratio', np.nan)),
        'ix_mf_efficiency': _safe_div(
            mf_feats.get('net_mf_ratio_5d', np.nan),
            stock_feats.get('turnover_rate', np.nan)),
        'ix_vol_surge_pullback': _safe_mul(
            cs_rank_feats.get('cs_rank_volume_surge', np.nan),
            stock_feats.get('pullback_from_high', np.nan)),
        'ix_alpha_conc': _safe_mul(
            residual_feats.get('residual_return_20d', np.nan),
            mf_feats.get('mf_concentration', np.nan)),
        'ix_north_cap': _safe_mul(
            mf_feats.get('northbound_stock_5d', np.nan),
            stock_feats.get('log_market_cap', np.nan) if 'log_market_cap' in stock_feats
            else stock_feats.get('log_market_cap', np.nan)),
    }
```

- [ ] **Step 3: Commit**

```bash
git add ml_models/ng/ng_feature_calculator.py
git commit -m "feat(ng): 8资金流因子 + 8交互因子计算函数"
```

---

### Task 4: Cache Updater 升级 — 资金流因子 + 残差标签 + ng1.1.0

**Files:**
- Modify: `ml_models/ng/ng_cache_updater.py`

- [ ] **Step 1: 更新常量和导入**

在 `ng_cache_updater.py` 顶部导入区域添加:

```python
from ml_models.ng.ng_feature_calculator import compute_moneyflow_features, compute_interaction_features
from ml_models.ng.ng_schema import create_moneyflow_table
from sklearn.linear_model import LinearRegression
```

更新 `MONEYFLOW_FEATURE_NAMES` 和 `INTERACTION_FEATURE_NAMES` 常量:

```python
MONEYFLOW_FEATURE_NAMES = [
    'net_mf_ratio_5d', 'big_order_ratio', 'big_order_trend_5d',
    'small_vs_big_divergence', 'mf_concentration', 'mf_momentum_10d',
    'northbound_stock_5d', 'mf_volume_divergence',
]

INTERACTION_FEATURE_NAMES = [
    'ix_vol_pullback', 'ix_big_trend', 'ix_rsi_mf', 'ix_ind_big',
    'ix_mf_efficiency', 'ix_vol_surge_pullback', 'ix_alpha_conc', 'ix_north_cap',
]
```

- [ ] **Step 2: 在 update_single_date 中集成 moneyflow 数据**

在 `update_single_date` 方法的步骤8（northbound flow）之后，添加 moneyflow 加载:

```python
# 8.5. Fetch and store moneyflow data (ng1.1.0)
if self.version >= 'ng1.1.0':
    create_moneyflow_table(self.db_path)
    mf_count = self._fetch_and_store_moneyflow(conn, date)
    print(f"  [{date}] Moneyflow data: {mf_count} rows")
    
    # Load moneyflow for factor computation
    mf_data = self._load_moneyflow_data(conn, date, active_sids, n_days=20)
else:
    mf_data = {}
```

- [ ] **Step 3: 在 pass 1 循环中计算资金流因子和交互因子**

在每个 eligible stock 的特征计算循环中（`eligible_stocks[sid]` 赋值之前），添加:

```python
# --- Compute moneyflow features (8, NEW in ng1.1.0) ---
mf_feats = {}
if self.version >= 'ng1.1.0':
    code_for_mf = info['code']
    mf_rows_for_stock = mf_data.get(code_for_mf, [])
    # price_changes from closes
    if len(closes) >= 2:
        price_changes = np.diff(closes) / (closes[:-1] + 1e-8)
    else:
        price_changes = np.array([0.0])
    
    try:
        mf_feats = compute_moneyflow_features(
            mf_rows=mf_rows_for_stock,
            amounts=amounts,
            price_changes=price_changes,
        )
    except Exception as e:
        print(f"    WARN: moneyflow_features failed for {code}: {e}")
        mf_feats = {}
```

在 CS rank + residual 计算之后，添加交互因子:

```python
# --- Compute interaction features (8 candidates, NEW in ng1.1.0) ---
ix_feats = {}
if self.version >= 'ng1.1.0':
    try:
        ix_feats = compute_interaction_features(
            stock_feats=stock_feats,
            mf_feats=mf_feats,
            industry_feats=ind_feats,
            residual_feats=residual_feats,
            cs_rank_feats=cs_rank_feats,
        )
    except Exception as e:
        print(f"    WARN: interaction_features failed for {code}: {e}")
        ix_feats = {}
```

将 `mf_feats` 和 `ix_feats` 合并到 `features_json`:

```python
all_feats = {**stock_feats, **fund_feats, **ind_feats, **cs_rank_feats, **residual_feats}
if self.version >= 'ng1.1.0':
    all_feats.update(mf_feats)
    all_feats.update(ix_feats)
```

- [ ] **Step 4: 添加风格因子残差标签计算**

在 `_convert_labels_to_excess` 方法末尾（return 之前），添加残差标签逻辑:

```python
def _convert_labels_to_residual(
    self,
    excess_labels: Dict[int, Dict[str, float]],
    universe: Dict[int, dict],
    price_data: Dict[int, List[dict]],
    returns_20d: Dict[int, float],
    stock_volatilities: Dict[int, float],
) -> Dict[int, Dict[str, float]]:
    """
    ng1.1.0: Convert industry excess labels to STYLE RESIDUAL labels.
    Removes size/momentum/volatility exposure via cross-sectional regression.
    
    residual = excess_return - β_size × log_mcap - β_mom × mom_20d - β_vol × vol_20d
    """
    # Collect cross-sectional data
    sids_with_data = []
    X_rows = []
    
    for sid, labs in excess_labels.items():
        info = universe.get(sid)
        if info is None:
            continue
        
        rows = price_data.get(sid, [])
        if not rows:
            continue
        
        # log_market_cap proxy from daily_basic
        last_row = rows[-1]
        close_price = last_row.get('close', 0)
        if close_price <= 0:
            continue
        
        mom_20d = returns_20d.get(sid, np.nan)
        vol_20d = stock_volatilities.get(sid, np.nan)
        
        # Use circ_mv from the universe data if available
        circ_mv = info.get('circ_mv', np.nan)
        if np.isnan(circ_mv) or circ_mv <= 0:
            continue
        log_mcap = np.log(circ_mv + 1)
        
        if np.isnan(mom_20d) or np.isnan(vol_20d):
            continue
        
        sids_with_data.append(sid)
        X_rows.append([log_mcap, mom_20d, vol_20d])
    
    if len(sids_with_data) < 100:
        # Not enough data for regression, return excess as-is
        return excess_labels
    
    X = np.array(X_rows)
    
    residual_labels = {}
    for horizon in ['label_3d', 'label_5d', 'label_10d', 'label_15d']:
        y = np.array([excess_labels[sid].get(horizon, np.nan) for sid in sids_with_data])
        valid = ~np.isnan(y)
        if valid.sum() < 100:
            continue
        
        reg = LinearRegression().fit(X[valid], y[valid])
        predicted = reg.predict(X)
        
        for i, sid in enumerate(sids_with_data):
            if sid not in residual_labels:
                residual_labels[sid] = dict(excess_labels.get(sid, {}))
            if valid[i]:
                # Store residual as label, keep raw excess as label_raw
                residual_labels[sid][f'label_raw_{horizon.replace("label_", "")}'] = residual_labels[sid].get(horizon, np.nan)
                residual_labels[sid][horizon] = float(y[i] - predicted[i])
    
    # Fill in stocks that weren't in regression (keep excess labels)
    for sid, labs in excess_labels.items():
        if sid not in residual_labels:
            residual_labels[sid] = dict(labs)
            for horizon in ['label_3d', 'label_5d', 'label_10d', 'label_15d']:
                residual_labels[sid][f'label_raw_{horizon.replace("label_", "")}'] = labs.get(horizon, np.nan)
    
    # Update downside_10d from residual label_10d
    for sid, labs in residual_labels.items():
        label_10d = labs.get('label_10d', np.nan)
        if label_10d is not None and not np.isnan(label_10d):
            labs['downside_10d'] = max(0.0, -label_10d)
        else:
            labs['downside_10d'] = np.nan
    
    return residual_labels
```

- [ ] **Step 5: 在 update_single_date 中调用残差标签**

在标签计算部分（步骤13之后），如果版本 >= ng1.1.0，调用残差转换:

```python
# Convert to industry excess returns
labels_all = self._convert_labels_to_excess(labels_abs, universe)

# ng1.1.0: Further convert to style residual labels
if self.version >= 'ng1.1.0':
    # Need circ_mv in universe for regression
    for sid in universe:
        db = daily_basic.get(sid)
        if db:
            universe[sid]['circ_mv'] = db.get('circ_mv', np.nan)
    
    labels_all = self._convert_labels_to_residual(
        labels_all, universe, price_data, returns_20d, stock_volatilities
    )
```

- [ ] **Step 6: 更新 INSERT 语句支持 label_raw 列**

在写入 cache 表的 INSERT 语句中，如果 version >= ng1.1.0，添加 label_raw 列:

```python
if self.version >= 'ng1.1.0':
    insert_sql = f"""
    INSERT OR REPLACE INTO {self.table_name}
    (code, trade_date, features_json,
     label_3d, label_5d, label_10d, label_15d, downside_10d,
     label_raw_3d, label_raw_5d, label_raw_10d, label_raw_15d,
     market_return_5d, market_return_20d, market_volatility_20d,
     market_breadth, market_new_high_ratio, northbound_flow_5d,
     market_volume_ratio, market_drawdown, vix_proxy, market_momentum_diff)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    # In the row tuple, add label_raw values after downside_10d
    row_data = (
        code, date, features_json_str,
        labels.get('label_3d'), labels.get('label_5d'),
        labels.get('label_10d'), labels.get('label_15d'),
        labels.get('downside_10d'),
        labels.get('label_raw_3d'), labels.get('label_raw_5d'),
        labels.get('label_raw_10d'), labels.get('label_raw_15d'),
        *market_values,
    )
```

- [ ] **Step 7: 更新默认版本和 argparse**

将 argparse 的 `--version` default 改为 `ng1.1.0`:

```python
parser.add_argument('--version', default='ng1.1.0', help='NG version (default: ng1.1.0)')
```

- [ ] **Step 8: 验证 cache updater (单日)**

Run: `python3 ml_models/ng/ng_cache_updater.py --date 2026-04-03 --version ng1.1.0`
Expected: 处理 ~4500 只股票，无 error

- [ ] **Step 9: Commit**

```bash
git add ml_models/ng/ng_cache_updater.py
git commit -m "feat(ng): ng1.1.0 cache — 资金流因子 + 交互因子 + 风格残差标签"
```

---

### Task 5: Trainer 升级 — WF参数化 + 市况加权 + IC筛选 + CLI开关

**Files:**
- Modify: `ml_models/ng/ng_trainer.py`

- [ ] **Step 1: 更新版本常量和特征名**

```python
NG_VERSION = 'ng1.1.0'

# 在 STOCK_FEATURE_NAMES 列表末尾追加资金流因子
MONEYFLOW_FEATURE_NAMES: List[str] = [
    'net_mf_ratio_5d', 'big_order_ratio', 'big_order_trend_5d',
    'small_vs_big_divergence', 'mf_concentration', 'mf_momentum_10d',
    'northbound_stock_5d', 'mf_volume_divergence',
]

INTERACTION_FEATURE_NAMES: List[str] = [
    'ix_vol_pullback', 'ix_big_trend', 'ix_rsi_mf', 'ix_ind_big',
    'ix_mf_efficiency', 'ix_vol_surge_pullback', 'ix_alpha_conc', 'ix_north_cap',
]

# ALL will be dynamically constructed based on CLI switches
BASE_ALL_FEATURE_NAMES: List[str] = STOCK_FEATURE_NAMES + MARKET_FEATURE_NAMES  # 68
```

- [ ] **Step 2: 添加 IC 筛选交互因子方法**

在 `NGTrainer` 类中添加:

```python
def _select_interaction_features(self, df: pd.DataFrame, label_col: str = 'label_10d',
                                  min_ic: float = 0.02, max_corr: float = 0.7
                                  ) -> List[str]:
    """IC-based selection of interaction features.
    Returns list of interaction feature names that pass the filter."""
    from scipy.stats import spearmanr
    
    existing_cols = [c for c in BASE_ALL_FEATURE_NAMES if c in df.columns]
    candidate_cols = [c for c in INTERACTION_FEATURE_NAMES if c in df.columns]
    
    if not candidate_cols or label_col not in df.columns:
        return []
    
    y = df[label_col].values
    valid = ~np.isnan(y)
    
    selected = []
    for col in candidate_cols:
        x = df[col].values
        both_valid = valid & ~np.isnan(x)
        if both_valid.sum() < 1000:
            continue
        
        ic, _ = spearmanr(x[both_valid], y[both_valid])
        if abs(ic) < min_ic:
            logger.info(f"  IX {col}: IC={ic:.4f} < {min_ic}, SKIP")
            continue
        
        # Check max correlation with existing features
        max_abs_corr = 0.0
        for ecol in existing_cols:
            ex = df[ecol].values
            both = both_valid & ~np.isnan(ex)
            if both.sum() < 100:
                continue
            corr, _ = spearmanr(x[both], ex[both])
            max_abs_corr = max(max_abs_corr, abs(corr))
        
        if max_abs_corr > max_corr:
            logger.info(f"  IX {col}: IC={ic:.4f}, max_corr={max_abs_corr:.3f} > {max_corr}, SKIP")
            continue
        
        logger.info(f"  IX {col}: IC={ic:.4f}, max_corr={max_abs_corr:.3f} → SELECTED")
        selected.append(col)
    
    return selected
```

- [ ] **Step 3: 添加市况样本加权方法**

```python
def _compute_regime_weights(self, df: pd.DataFrame) -> np.ndarray:
    """Compute sample weights based on market regime.
    Bull (+5%): 0.8, Sideways: 1.0, Bear (-5%): 1.2"""
    mkt_ret = df['market_return_20d'].values if 'market_return_20d' in df.columns else np.zeros(len(df))
    weights = np.ones(len(df))
    weights[mkt_ret > 0.05] = 0.8   # 牛市降权
    weights[mkt_ret < -0.05] = 1.2  # 熊市增权
    logger.info(f"  Regime weights: bull={np.sum(weights == 0.8):,}, "
                f"sideways={np.sum(weights == 1.0):,}, bear={np.sum(weights == 1.2):,}")
    return weights
```

- [ ] **Step 4: 修改 load_data 支持动态特征列表**

在 `load_data` 方法中，解析 features_json 后动态添加资金流和交互因子:

```python
# After df_stock_features construction:
# Dynamically extend feature list based on switches
active_stock_features = list(STOCK_FEATURE_NAMES)
if getattr(self, '_enable_moneyflow', False):
    active_stock_features += MONEYFLOW_FEATURE_NAMES
if getattr(self, '_enable_interaction', False):
    active_stock_features += INTERACTION_FEATURE_NAMES

for col in active_stock_features:
    if col not in df_stock_features.columns:
        df_stock_features[col] = np.nan

df_stock_features = df_stock_features[[c for c in active_stock_features if c in df_stock_features.columns]]

# Update instance feature names
self.feature_names = active_stock_features + list(MARKET_FEATURE_NAMES)
self.stock_feature_cols = active_stock_features
```

- [ ] **Step 5: 修改 walk_forward_train 集成 IC筛选和市况加权**

在 `walk_forward_train` 方法中，在调用 `super().walk_forward_train()` 之前:

```python
# IC screen interaction features (before training)
if getattr(self, '_enable_interaction', False):
    df_full = self.load_data(start_date=start_date, end_date=end_date)
    selected_ix = self._select_interaction_features(df_full)
    if selected_ix:
        # Remove non-selected interaction features
        remove_ix = [c for c in INTERACTION_FEATURE_NAMES if c not in selected_ix]
        self.feature_names = [c for c in self.feature_names if c not in remove_ix]
        self.stock_feature_cols = [c for c in self.stock_feature_cols if c not in remove_ix]
        logger.info(f"  Selected interaction features: {selected_ix}")
    else:
        # Remove all interaction features
        self.feature_names = [c for c in self.feature_names if c not in INTERACTION_FEATURE_NAMES]
        self.stock_feature_cols = [c for c in self.stock_feature_cols if c not in INTERACTION_FEATURE_NAMES]
        logger.info("  No interaction features passed IC screen")

# Regime sample weights
if getattr(self, '_regime_weight', False):
    self._sample_weight_fn = self._compute_regime_weights
```

将 `sample_weight_fn` 传递给 V485Trainer 的训练循环需要在 prepare_features 返回时附加:

```python
def prepare_features(self, df: pd.DataFrame) -> tuple:
    # ... existing code ...
    
    # Attach regime weights if enabled
    if hasattr(self, '_sample_weight_fn') and self._sample_weight_fn is not None:
        self._current_sample_weights = self._sample_weight_fn(df)
    
    return X, y_3d, y_5d, y_10d, y_15d, df
```

- [ ] **Step 6: 更新 CLI argparse 添加新开关**

```python
parser.add_argument('--enable-moneyflow', action='store_true',
                    help='Enable moneyflow features (8 factors)')
parser.add_argument('--enable-interaction', action='store_true',
                    help='Enable interaction features with IC screening')
parser.add_argument('--residual-label', action='store_true',
                    help='Use style-residual labels instead of industry excess')
parser.add_argument('--wf-windows', type=int, default=3,
                    help='Number of WF windows (default: 3, recommended: 8)')
parser.add_argument('--regime-weight', action='store_true',
                    help='Enable market regime sample weighting')
```

在 trainer 初始化后设置开关:

```python
trainer._enable_moneyflow = args.enable_moneyflow
trainer._enable_interaction = args.enable_interaction
trainer._regime_weight = args.regime_weight

# WF windows: adjust step_days to create more windows
if args.wf_windows > 3:
    # step_days = 90 gives ~8 windows for 5-year data
    args.step_days = 90
    logger.info(f"WF windows target: {args.wf_windows}, step_days adjusted to 90")
```

- [ ] **Step 7: 更新 model metadata 记录所用开关**

在 `walk_forward_train` 保存 model_data 时:

```python
model_data['ng_innovations']['ng110_switches'] = {
    'moneyflow': getattr(self, '_enable_moneyflow', False),
    'interaction': getattr(self, '_enable_interaction', False),
    'interaction_selected': selected_ix if getattr(self, '_enable_interaction', False) else [],
    'residual_label': self.cache_table == get_table_name('ng1.1.0'),
    'regime_weight': getattr(self, '_regime_weight', False),
    'wf_step_days': step_days,
}
```

- [ ] **Step 8: Commit**

```bash
git add ml_models/ng/ng_trainer.py
git commit -m "feat(ng): ng1.1.0 trainer — WF参数化 + 市况加权 + IC筛选 + CLI开关"
```

---

### Task 6: Production Scorer 升级

**Files:**
- Modify: `ml_models/ng/ng_production_scorer.py`

- [ ] **Step 1: 更新特征名常量**

在 `ng_production_scorer.py` 导入区域添加:

```python
from ml_models.ng.ng_trainer import MONEYFLOW_FEATURE_NAMES, INTERACTION_FEATURE_NAMES
```

- [ ] **Step 2: 修改 _load_model 支持动态特征列表**

在 `_load_model` 方法中，从模型 pkl 读取实际使用的特征名:

```python
# After loading model_data:
# Use feature_names from model (may include moneyflow + selected interaction)
if 'feature_names' in model_data:
    self.feature_names = model_data['feature_names']
if 'stock_feature_cols' in model_data:
    self.stock_feature_cols = model_data['stock_feature_cols']
if 'macro_feature_cols' in model_data:
    self.macro_feature_cols = model_data['macro_feature_cols']
```

- [ ] **Step 3: 修改 _load_features 扩展 features_json 解析**

在 `_load_features` 中，解析 features_json 后补齐可能存在的新因子列:

```python
# After parsing features_json:
for col in self.stock_feature_cols:
    if col not in df_features.columns:
        df_features[col] = 0.0  # fillna for missing moneyflow/interaction features
```

- [ ] **Step 4: Commit**

```bash
git add ml_models/ng/ng_production_scorer.py
git commit -m "feat(ng): ng1.1.0 scorer — 动态特征列表 + moneyflow/interaction支持"
```

---

### Task 7: Batch Report 注册 + __init__ 更新

**Files:**
- Modify: `backtest/batch_generate_v395_reports.py`
- Modify: `ml_models/ng/__init__.py`

- [ ] **Step 1: 在 batch_generate_v395_reports.py 注册 ng1.1.0**

在 argparse choices 列表中添加 `'ng1.1.0'`:

```python
choices=[..., 'ng1.0.2', 'ng1.1.0']
```

在 version-to-scorer 映射的 ng 分支中添加 ng1.1.0:

```python
elif args.version in ('ng1.0.0', 'ng1.0.1', 'ng1.0.2', 'ng1.1.0'):
    from ml_models.ng.ng_production_scorer import NGProductionScorer
    scorer = NGProductionScorer(model_path=getattr(args, 'model_path', None))
```

在 `_ng_tables` 映射中添加:

```python
_ng_tables = {
    'ng1.0.0': 'ng_feature_cache',
    'ng1.0.1': 'ng101_feature_cache',
    'ng1.0.2': 'ng102_feature_cache',
    'ng1.1.0': 'ng110_feature_cache',
}
```

- [ ] **Step 2: 更新 __init__.py**

更新 docstring 和导出:

```python
"""Daily Selection NG — Next Generation trend-following factor model.

v1.1.0: Moneyflow factors, style-residual labels, WF upgrade, regime weighting.
"""
```

- [ ] **Step 3: Commit**

```bash
git add backtest/batch_generate_v395_reports.py ml_models/ng/__init__.py
git commit -m "feat(ng): 注册ng1.1.0版本 — batch report + __init__"
```

---

### Task 8: Moneyflow 历史数据回填

**Files:**
- 使用现有 `ng_cache_updater.py`

- [ ] **Step 1: 回填 moneyflow_daily 历史数据**

创建回填脚本（一次性使用，不需要新文件，直接在终端运行）:

```bash
python3 -c "
import sys, json, time, sqlite3
sys.path.insert(0, '.')
from ml_models.ng.ng_schema import DB_PATH, create_moneyflow_table
create_moneyflow_table()

with open('config.json') as f:
    config = json.load(f)
import tushare as ts
pro = ts.pro_api(config['tushare']['token'])

conn = sqlite3.connect(DB_PATH, timeout=30)
# Get all trading dates from 2018-01-01
dates = [r[0] for r in conn.execute(
    'SELECT DISTINCT trade_date FROM daily_quotes WHERE trade_date >= ? ORDER BY trade_date',
    ('2018-01-01',)
).fetchall()]

# Check existing
existing = set(r[0] for r in conn.execute(
    'SELECT DISTINCT trade_date FROM moneyflow_daily'
).fetchall())

missing = [d for d in dates if d not in existing]
print(f'Total dates: {len(dates)}, existing: {len(existing)}, to fetch: {len(missing)}')

for i, d in enumerate(missing):
    try:
        d_str = d.replace('-', '')
        df = pro.moneyflow(trade_date=d_str)
        if df is not None and len(df) > 0:
            rows = [(r['ts_code'], d,
                     float(r.get('buy_sm_amount') or 0), float(r.get('sell_sm_amount') or 0),
                     float(r.get('buy_md_amount') or 0), float(r.get('sell_md_amount') or 0),
                     float(r.get('buy_lg_amount') or 0), float(r.get('sell_lg_amount') or 0),
                     float(r.get('buy_elg_amount') or 0), float(r.get('sell_elg_amount') or 0),
                     float(r.get('net_mf_amount') or 0))
                    for _, r in df.iterrows()]
            conn.executemany('INSERT OR REPLACE INTO moneyflow_daily '
                           '(code,trade_date,buy_sm_amount,sell_sm_amount,'
                           'buy_md_amount,sell_md_amount,buy_lg_amount,sell_lg_amount,'
                           'buy_elg_amount,sell_elg_amount,net_mf_amount) '
                           'VALUES (?,?,?,?,?,?,?,?,?,?,?)', rows)
            conn.commit()
            if (i+1) % 10 == 0:
                print(f'  [{i+1}/{len(missing)}] {d}: {len(rows)} rows')
        time.sleep(0.15)  # rate limit
    except Exception as e:
        print(f'  ERROR {d}: {e}')
        time.sleep(1)

conn.close()
print('Done!')
"
```

注意: 这个回填可能需要 30-60 分钟（~2000 天 × 0.15s/天）。在后台运行。

- [ ] **Step 2: 回填 ng110_feature_cache**

在 moneyflow 回填完成后:

```bash
python3 ml_models/ng/ng_cache_updater.py \
    --start-date 2020-01-01 --end-date 2026-04-05 --version ng1.1.0
```

- [ ] **Step 3: 验证缓存完整性**

```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('data_adapter/stock_data.db', timeout=30)
mf = conn.execute('SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM moneyflow_daily').fetchone()
ng = conn.execute('SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM ng110_feature_cache').fetchone()
print(f'moneyflow_daily: {mf[0]:,} rows, {mf[1]} ~ {mf[2]}')
print(f'ng110_feature_cache: {ng[0]:,} rows, {ng[1]} ~ {ng[2]}')
conn.close()
"
```

---

### Task 9: Fast-check 验证矩阵

**Files:**
- 使用修改后的 `ng_trainer.py`

- [ ] **Step 1: Baseline fast-check (= ng1.0.2 级别)**

```bash
python3 ml_models/ng/ng_trainer.py \
    --fast-check --start-date 2022-01-01 \
    --lambda-risk 0.5
```

记录 10d IC 和 ICIR 作为 baseline。

- [ ] **Step 2: exp1 — 资金流 + 交互因子**

```bash
python3 ml_models/ng/ng_trainer.py \
    --fast-check --start-date 2022-01-01 \
    --enable-moneyflow --enable-interaction \
    --lambda-risk 0.5
```

对比 baseline，记录 10d IC/ICIR 变化。

- [ ] **Step 3: exp2 — 残差标签**

需要先确保 ng110 cache 中已有残差标签数据（Task 8 完成后）:

```bash
python3 ml_models/ng/ng_trainer.py \
    --fast-check --start-date 2022-01-01 \
    --lambda-risk 0.5
```

注: 残差标签是在 cache_updater 阶段写入的，trainer 读取时自动使用 ng110 表中的 label 列（已经是残差）。需要临时切换 `NG_VERSION = 'ng1.1.0'` 以读取 ng110 表。

- [ ] **Step 4: exp3 — WF8 + 市况加权**

```bash
python3 ml_models/ng/ng_trainer.py \
    --fast-check --start-date 2022-01-01 \
    --wf-windows 8 --regime-weight \
    --lambda-risk 0.5
```

- [ ] **Step 5: 分析结果，确定合并方案**

对比4个实验的 10d ICIR:
- 正向（ICIR 提升 > 0.05）: 保留
- 负向或噪声: 排除

- [ ] **Step 6: Final — 合并所有正向改动，完整训练**

```bash
python3 ml_models/ng/ng_trainer.py \
    --start-date 2020-01-01 --purge-days 15 \
    --enable-moneyflow --enable-interaction \
    --wf-windows 8 --regime-weight \
    --lambda-risk 0.5
```

（根据 step 5 结果调整开关）

---

### Task 10: 北极星评估 + 生产配置更新

**Files:**
- 使用现有回测工具
- Modify: `production_config.json`

- [ ] **Step 1: 生成批量报告**

```bash
python3 backtest/batch_generate_v395_reports.py \
    --version ng1.1.0 \
    --start-date 2020-01-01 --end-date 2026-04-05 \
    --output-dir reports/daily_selection_ng110
```

- [ ] **Step 2: 北极星 V5.2 评估**

```bash
python3 backtest/run_north_star_eval.py \
    --backtest --report-dir reports/daily_selection_ng110 \
    --label NG110-PROD --top-n 10 --focus-days 10 \
    --rank-field composite \
    --cppi-floor 0.05 --cppi-multiplier 20
```

Target: V5.2 > 76% (vs ng1.0.2 的 74.0%)

- [ ] **Step 3: 如果达标，更新 production_config.json**

更新版本号、模型路径、特征数、缓存表等字段。

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat(ng): ng1.1.0完整训练 — 资金流因子+残差标签+WF升级, V5.2=XX%"
```
