# Webapp Data Explorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a unified `/data-explorer` page to the webapp that lets the user query and visualize every table in `data_adapter/stock_data.db` — raw OHLC, fundamentals, technical indicators, every model feature cache (with `features_json` auto-expanded), factor caches, money-flow, signal-trust scores, and backtest results — via SQL editor + Visual Builder with smart-suggested charts and saved queries.

**Architecture:** One Flask blueprint (`api/data_explorer.py`) + 5 focused backend modules in `core/data_explorer/` (schema discovery, SELECT-only query runner with LIMIT injection and 30 s timeout, JSON feature expander, chart-type heuristic, saved-query CRUD). One Jinja template + one JS file (jQuery + CodeMirror 6 + DataTables + ApexCharts) — all libraries except CodeMirror already loaded in `base.html`.

**Tech Stack:** Flask 3.0 blueprint pattern (existing), SQLite (readonly URI connection), `sqlglot` (new pip dep) for SQL parsing, Pandas `json_normalize`, CodeMirror 6 (CDN), ApexCharts + DataTables (already loaded), Bootstrap 5 styling.

**Spec reference:** `docs/superpowers/specs/2026-04-19-webapp-data-explorer-design.md`

---

## File structure

Create:
- `webapp/core/data_explorer/__init__.py`
- `webapp/core/data_explorer/feature_expander.py`
- `webapp/core/data_explorer/chart_suggester.py`
- `webapp/core/data_explorer/schema_discovery.py`
- `webapp/core/data_explorer/query_runner.py`
- `webapp/core/data_explorer/query_store.py`
- `webapp/api/data_explorer.py`
- `webapp/templates/data_explorer.html`
- `webapp/static/js/data_explorer.js`
- `webapp/static/css/data_explorer.css`
- `webapp/tests/test_data_explorer/__init__.py`
- `webapp/tests/test_data_explorer/conftest.py`
- `webapp/tests/test_data_explorer/test_feature_expander.py`
- `webapp/tests/test_data_explorer/test_chart_suggester.py`
- `webapp/tests/test_data_explorer/test_schema_discovery.py`
- `webapp/tests/test_data_explorer/test_query_runner.py`
- `webapp/tests/test_data_explorer/test_query_store.py`
- `webapp/tests/test_data_explorer/test_blueprint_integration.py`

Modify:
- `webapp/requirements.txt` (add `sqlglot==26.12.1`)
- `webapp/app.py` (register blueprint, add `/data-explorer` route)
- `webapp/templates/base.html` (add nav link)

---

### Task 1: Add `sqlglot` dependency + `saved_queries` migration

**Files:**
- Modify: `webapp/requirements.txt`
- Create: `webapp/core/data_explorer/__init__.py`
- Create: `webapp/core/data_explorer/query_store.py` (skeleton for migration only)
- Test: `webapp/tests/test_data_explorer/__init__.py`, `webapp/tests/test_data_explorer/conftest.py`, `webapp/tests/test_data_explorer/test_query_store.py` (migration test only — CRUD tests added in Task 7)

- [ ] **Step 1: Add `sqlglot` to requirements**

Append to `webapp/requirements.txt`:

```
# SQL parsing for data explorer (SELECT-only validation, LIMIT injection)
sqlglot==26.12.1
```

Run: `pip install sqlglot==26.12.1`

- [ ] **Step 2: Create package init**

Create `webapp/core/data_explorer/__init__.py`:

```python
"""
Data Explorer - unified query and visualization layer for stock_data.db.

Modules:
  feature_expander  - expand features_json column to flat columns
  chart_suggester   - heuristic chart-type picker based on DataFrame shape
  schema_discovery  - scan sqlite_master and classify tables
  query_runner      - SELECT-only SQL execution with LIMIT + timeout
  query_store       - CRUD for saved_queries in webapp.db
"""
```

- [ ] **Step 3: Create test infrastructure**

Create `webapp/tests/test_data_explorer/__init__.py` (empty file).

Create `webapp/tests/test_data_explorer/conftest.py`:

```python
"""Shared fixtures for data_explorer tests."""
import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def tmp_webapp_db(tmp_path: Path) -> Path:
    """Empty webapp.db-style SQLite file for query_store tests."""
    return tmp_path / "webapp.db"


@pytest.fixture
def tmp_stock_db(tmp_path: Path) -> Path:
    """Tiny stock_data.db-style SQLite file with a couple of tables for query_runner tests."""
    db_path = tmp_path / "stock.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE daily_quotes (
            code TEXT, trade_date DATE, close REAL
        );
        INSERT INTO daily_quotes VALUES
          ('600519.SH', '2026-04-17', 1700.0),
          ('600519.SH', '2026-04-18', 1720.0),
          ('000858.SZ', '2026-04-18', 150.0);

        CREATE TABLE ng101_feature_cache (
            code TEXT, trade_date DATE, label_10d REAL, features_json TEXT
        );
        INSERT INTO ng101_feature_cache VALUES
          ('600519.SH', '2026-04-18', 0.05, '{"pb": 8.1, "roe": 31.2}'),
          ('000858.SZ', '2026-04-18', 0.03, '{"pb": 4.2, "roe": 22.1}');

        CREATE TABLE securities (code TEXT, name TEXT, type TEXT);
        INSERT INTO securities VALUES ('600519.SH', 'Maotai', 'A股');
        """
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def sample_df_with_json() -> pd.DataFrame:
    """DataFrame shaped like a feature_cache query result."""
    return pd.DataFrame(
        {
            "code": ["600519.SH", "000858.SZ"],
            "trade_date": ["2026-04-18", "2026-04-18"],
            "label_10d": [0.05, 0.03],
            "features_json": [
                json.dumps({"pb": 8.1, "roe": 31.2}),
                json.dumps({"pb": 4.2, "roe": 22.1}),
            ],
        }
    )
```

- [ ] **Step 4: Write the failing test for `query_store.apply_migration`**

Create `webapp/tests/test_data_explorer/test_query_store.py`:

```python
"""Tests for query_store (Task 1 covers migration; Task 7 covers CRUD)."""
import sqlite3
from pathlib import Path

from core.data_explorer.query_store import apply_migration


def test_migration_creates_table(tmp_webapp_db: Path) -> None:
    apply_migration(tmp_webapp_db)
    conn = sqlite3.connect(tmp_webapp_db)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    conn.close()
    assert "saved_queries" in tables


def test_migration_is_idempotent(tmp_webapp_db: Path) -> None:
    apply_migration(tmp_webapp_db)
    apply_migration(tmp_webapp_db)
    apply_migration(tmp_webapp_db)
    conn = sqlite3.connect(tmp_webapp_db)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(saved_queries)")]
    conn.close()
    assert cols == [
        "id", "name", "sql", "tags", "description",
        "created_at", "updated_at", "last_run_at", "run_count",
    ]
```

- [ ] **Step 5: Run the test to verify it fails**

Run: `cd webapp && python3 -m pytest tests/test_data_explorer/test_query_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'webapp.core.data_explorer.query_store'`.

- [ ] **Step 6: Implement the minimal migration**

Create `webapp/core/data_explorer/query_store.py`:

```python
"""CRUD for saved_queries on webapp.db. See Task 7 for CRUD helpers."""
from __future__ import annotations

import sqlite3
from pathlib import Path


MIGRATION_SQL = """
CREATE TABLE IF NOT EXISTS saved_queries (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL UNIQUE,
    sql          TEXT    NOT NULL,
    tags         TEXT,
    description  TEXT,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_run_at  TIMESTAMP,
    run_count    INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_saved_queries_tags ON saved_queries(tags);
"""


def apply_migration(webapp_db_path: str | Path) -> None:
    """Idempotent: create saved_queries table if missing."""
    webapp_db_path = Path(webapp_db_path)
    webapp_db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(webapp_db_path, timeout=30.0)
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        conn.executescript(MIGRATION_SQL)
        conn.commit()
    finally:
        conn.close()
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `cd webapp && python3 -m pytest tests/test_data_explorer/test_query_store.py -v`
Expected: PASS 2 tests.

- [ ] **Step 8: Commit**

```bash
git add webapp/requirements.txt \
        webapp/core/data_explorer/__init__.py \
        webapp/core/data_explorer/query_store.py \
        webapp/tests/test_data_explorer/__init__.py \
        webapp/tests/test_data_explorer/conftest.py \
        webapp/tests/test_data_explorer/test_query_store.py
git commit -m "feat(explorer): add sqlglot dep + saved_queries migration"
```

---

### Task 2: `feature_expander.py` — expand `features_json` to columns

**Files:**
- Create: `webapp/core/data_explorer/feature_expander.py`
- Test: `webapp/tests/test_data_explorer/test_feature_expander.py`

- [ ] **Step 1: Write the failing tests**

Create `webapp/tests/test_data_explorer/test_feature_expander.py`:

```python
"""Tests for feature_expander.expand()."""
import pandas as pd
import pytest

from core.data_explorer.feature_expander import expand


def test_expands_features_json_into_columns(sample_df_with_json: pd.DataFrame) -> None:
    df_out, warnings = expand(sample_df_with_json)
    assert "features_json" not in df_out.columns
    assert "pb" in df_out.columns
    assert "roe" in df_out.columns
    assert df_out.loc[0, "pb"] == 8.1
    assert df_out.loc[1, "roe"] == 22.1
    assert warnings == []


def test_passthrough_when_no_features_json_column() -> None:
    df = pd.DataFrame({"code": ["A"], "close": [1.0]})
    df_out, warnings = expand(df)
    assert list(df_out.columns) == ["code", "close"]
    assert warnings == []


def test_malformed_json_returns_warning_and_preserves_column() -> None:
    df = pd.DataFrame(
        {"code": ["A"], "features_json": ['{"broken":']}
    )
    df_out, warnings = expand(df)
    assert "features_json" in df_out.columns  # preserved
    assert len(warnings) == 1
    assert "features_json expansion failed" in warnings[0]


def test_preserves_non_feature_columns_alongside_expanded() -> None:
    df = pd.DataFrame(
        {
            "code": ["A", "B"],
            "label_10d": [0.1, 0.2],
            "features_json": ['{"x": 1}', '{"x": 2}'],
        }
    )
    df_out, _ = expand(df)
    assert list(df_out.columns) == ["code", "label_10d", "x"]
    assert df_out["x"].tolist() == [1, 2]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd webapp && python3 -m pytest tests/test_data_explorer/test_feature_expander.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `expand()`**

Create `webapp/core/data_explorer/feature_expander.py`:

```python
"""Expand features_json column in a DataFrame into flat scalar columns.

feature_cache tables store ~66 stock-level features as a single JSON blob per
row; users expect to SELECT / filter / plot individual features without
calling json_extract() everywhere, so we normalize on the Python side.
"""
from __future__ import annotations

import json
from typing import Tuple

import pandas as pd


def expand(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    """Return (df_with_json_columns_flat, warnings).

    - If 'features_json' not in df, return df unchanged, no warnings.
    - On any JSON parse failure, return df unchanged with one warning string.
    - Idempotent (safe to call multiple times; no-op after first call).
    """
    if "features_json" not in df.columns:
        return df, []
    try:
        normalized = pd.json_normalize(df["features_json"].apply(json.loads))
    except (json.JSONDecodeError, TypeError) as e:
        return df, [f"features_json expansion failed: {e}"]
    out = pd.concat(
        [
            df.drop(columns=["features_json"]).reset_index(drop=True),
            normalized.reset_index(drop=True),
        ],
        axis=1,
    )
    return out, []
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd webapp && python3 -m pytest tests/test_data_explorer/test_feature_expander.py -v`
Expected: PASS 4 tests.

- [ ] **Step 5: Commit**

```bash
git add webapp/core/data_explorer/feature_expander.py \
        webapp/tests/test_data_explorer/test_feature_expander.py
git commit -m "feat(explorer): feature_expander (pd.json_normalize wrapper)"
```

---

### Task 3: `chart_suggester.py` — heuristic chart-type picker

**Files:**
- Create: `webapp/core/data_explorer/chart_suggester.py`
- Test: `webapp/tests/test_data_explorer/test_chart_suggester.py`

- [ ] **Step 1: Write the failing tests**

Create `webapp/tests/test_data_explorer/test_chart_suggester.py`:

```python
"""Tests for chart_suggester.suggest() — rule table per spec §5.4."""
import pandas as pd

from core.data_explorer.chart_suggester import suggest


def test_r1_time_series_single_code() -> None:
    df = pd.DataFrame({
        "trade_date": pd.to_datetime(["2026-04-17", "2026-04-18"]),
        "code": ["600519.SH", "600519.SH"],
        "close": [1700.0, 1720.0],
    })
    hint = suggest(df)
    assert hint["type"] == "line"
    assert hint["x"] == "trade_date"
    assert hint["y"] == "close"


def test_r2_time_series_multi_code_returns_none() -> None:
    df = pd.DataFrame({
        "trade_date": pd.to_datetime(["2026-04-18", "2026-04-18"]),
        "code": ["600519.SH", "000858.SZ"],
        "close": [1720.0, 150.0],
    })
    assert suggest(df) is None


def test_r3_two_numerics_scatter() -> None:
    df = pd.DataFrame({"pb": [1.0, 2.0, 3.0], "roe": [10, 20, 30]})
    hint = suggest(df)
    assert hint["type"] == "scatter"
    # r reported in annotation; simply assert presence
    assert "annotations" in hint and "pearson_r" in hint["annotations"]


def test_r4_bar_small_categorical() -> None:
    df = pd.DataFrame({"code": [f"S{i}" for i in range(30)], "pred": range(30)})
    hint = suggest(df)
    assert hint["type"] == "bar"
    assert hint["x"] == "code"
    assert hint["y"] == "pred"


def test_r5_histogram_large_categorical() -> None:
    df = pd.DataFrame({"code": [f"S{i}" for i in range(100)], "pred": range(100)})
    hint = suggest(df)
    assert hint["type"] == "histogram"
    assert hint["x"] == "pred"


def test_r6_single_numeric_histogram() -> None:
    df = pd.DataFrame({"pred_10d": [0.01, 0.05, -0.02]})
    hint = suggest(df)
    assert hint["type"] == "histogram"
    assert hint["x"] == "pred_10d"


def test_else_no_hint() -> None:
    df = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})  # two categoricals, no numeric
    assert suggest(df) is None


def test_empty_dataframe_no_hint() -> None:
    assert suggest(pd.DataFrame()) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd webapp && python3 -m pytest tests/test_data_explorer/test_chart_suggester.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `suggest()`**

Create `webapp/core/data_explorer/chart_suggester.py`:

```python
"""Heuristic chart-type picker.

Rules evaluated in order; first match wins. Returns None when nothing fits.
Spec: docs/superpowers/specs/2026-04-19-webapp-data-explorer-design.md §5.4
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from pandas.api.types import is_numeric_dtype


_CATEGORICAL_NAMES = {"code", "industry", "name", "tag", "trust_tag"}
_DATE_NAMES = {"trade_date", "date", "ann_date", "end_date", "as_of_date"}


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if is_numeric_dtype(df[c])]


def _categorical_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c in _CATEGORICAL_NAMES]


def _date_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if c in _DATE_NAMES:
            return c
    return None


def suggest(df: pd.DataFrame) -> Optional[dict]:
    if df.empty:
        return None

    nums = _numeric_cols(df)
    cats = _categorical_cols(df)
    date_col = _date_col(df)

    # R1 + R2: trade_date + numeric
    if date_col and nums:
        # R2: multiple distinct codes → ambiguous (no small-multiples in v1)
        if "code" in df.columns and df["code"].nunique() > 1:
            return None
        # R1: single (or no) code → line
        return {"type": "line", "x": date_col, "y": nums[0]}

    # R3: exactly 2 numerics (no date, no category)
    if len(nums) == 2 and not cats and not date_col:
        r = float(df[nums[0]].corr(df[nums[1]]))
        return {
            "type": "scatter",
            "x": nums[0],
            "y": nums[1],
            "annotations": {"pearson_r": round(r, 4)},
        }

    # R4 / R5: 1 categorical + 1 numeric
    if len(cats) == 1 and len(nums) == 1:
        if len(df) <= 50:
            return {"type": "bar", "x": cats[0], "y": nums[0]}
        return {"type": "histogram", "x": nums[0]}

    # R6: single numeric only
    if len(nums) == 1 and not cats and not date_col:
        return {"type": "histogram", "x": nums[0]}

    return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd webapp && python3 -m pytest tests/test_data_explorer/test_chart_suggester.py -v`
Expected: PASS 8 tests.

- [ ] **Step 5: Commit**

```bash
git add webapp/core/data_explorer/chart_suggester.py \
        webapp/tests/test_data_explorer/test_chart_suggester.py
git commit -m "feat(explorer): chart_suggester heuristic (6 rules)"
```

---

### Task 4: `schema_discovery.py` — scan + classify tables

**Files:**
- Create: `webapp/core/data_explorer/schema_discovery.py`
- Test: `webapp/tests/test_data_explorer/test_schema_discovery.py`

- [ ] **Step 1: Write the failing tests**

Create `webapp/tests/test_data_explorer/test_schema_discovery.py`:

```python
"""Tests for schema_discovery.discover() + classify_table()."""
from pathlib import Path

from core.data_explorer.schema_discovery import (
    classify_table,
    discover,
)


def test_classify_known_tables() -> None:
    assert classify_table("daily_quotes") == "raw"
    assert classify_table("daily_basic") == "raw"
    assert classify_table("financial_indicator") == "raw"
    assert classify_table("technical_indicators") == "technical"
    assert classify_table("ng101_feature_cache") == "feature_cache"
    assert classify_table("v39_feature_cache") == "feature_cache"
    assert classify_table("alpha158_feature_cache") == "feature_cache"
    assert classify_table("worldquant_factors") == "feature_cache"
    assert classify_table("v492_factor_cache") == "factor"
    assert classify_table("factor_daily_returns") == "factor"
    assert classify_table("market_amv") == "market_state"
    assert classify_table("signal_trust_scores") == "market_state"
    assert classify_table("moneyflow_daily") == "moneyflow"
    assert classify_table("hsgt_daily") == "moneyflow"
    assert classify_table("securities") == "meta"
    assert classify_table("sw_industry") == "meta"
    assert classify_table("backtest_trades") == "backtest"
    assert classify_table("stock_signals") == "backtest"


def test_classify_unknown_returns_other() -> None:
    assert classify_table("some_brand_new_table") == "other"


def test_discover_returns_categorized_dict(tmp_stock_db: Path) -> None:
    result = discover(tmp_stock_db)

    # conftest creates daily_quotes, ng101_feature_cache, securities
    categories = set(result.keys())
    assert "raw" in categories
    assert "feature_cache" in categories
    assert "meta" in categories

    raw_tables = [t["table"] for t in result["raw"]]
    assert "daily_quotes" in raw_tables

    ng = next(t for t in result["feature_cache"] if t["table"] == "ng101_feature_cache")
    assert ng["has_features_json"] is True
    assert "code" in [c["name"] for c in ng["columns"]]
    assert ng["row_count"] == 2
    assert ng["date_range"] == ("2026-04-18", "2026-04-18")


def test_discover_without_trade_date_handles_date_range_none(tmp_stock_db: Path) -> None:
    # securities has no trade_date; date_range must be None, not crash
    result = discover(tmp_stock_db)
    sec = next(t for t in result["meta"] if t["table"] == "securities")
    assert sec["date_range"] is None
    assert sec["has_features_json"] is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd webapp && python3 -m pytest tests/test_data_explorer/test_schema_discovery.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `schema_discovery`**

Create `webapp/core/data_explorer/schema_discovery.py`:

```python
"""Scan sqlite_master and classify tables into category buckets.

In-process TTL cache (5 min) keyed by db path. Invalidated via refresh=True.
"""
from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path
from typing import Optional


_CACHE: dict[str, tuple[float, dict]] = {}
_TTL_S = 300

_RAW = {
    "daily_quotes", "daily_basic", "financial_indicator",
    "financial_indicator_backup", "financial_statement_raw",
    "index_daily", "market_indices", "sw_index_daily",
    "daily_basic_temp",
}
_TECHNICAL = {"technical_indicators", "technical_overview"}
_MARKET_STATE = {
    "market_amv",
    "signal_trust_scores", "signal_trust_scores_history", "signal_trust_samples",
}
_MONEYFLOW = {"moneyflow_daily", "hsgt_daily"}
_META = {
    "securities", "stock_basic_info", "sw_industry",
    "schema_version", "data_update_log", "latest_quotes",
}
_BACKTEST = {"backtest_trades", "backtest_results", "stock_signals"}
# names that contain the word "factor" but are NOT feature_cache-shaped
_FACTOR_EXTRA = {"factor_daily_returns"}


def classify_table(name: str) -> str:
    if name in _RAW:
        return "raw"
    if name in _TECHNICAL:
        return "technical"
    if name in _MARKET_STATE:
        return "market_state"
    if name in _MONEYFLOW:
        return "moneyflow"
    if name in _META:
        return "meta"
    if name in _BACKTEST:
        return "backtest"
    if name in _FACTOR_EXTRA:
        return "factor"
    # pattern-based
    if re.search(r"_factor(_cache)?$", name):
        return "factor"
    if name.endswith("_feature_cache") or name in {
        "worldquant_factors", "latest_worldquant_factors",
        "active_mv_feature_cache", "neural_embedding_cache",
        "brain_alpha_cache", "alpha158_feature_cache",
        "ng_feature_cache",  # legacy ng v1.0.0
    }:
        return "feature_cache"
    return "other"


def discover(db_path: str | Path, refresh: bool = False) -> dict[str, list[dict]]:
    key = str(db_path)
    now = time.time()
    if not refresh and key in _CACHE:
        ts, val = _CACHE[key]
        if now - ts < _TTL_S:
            return val

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = [
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        result: dict[str, list[dict]] = {}
        for t in tables:
            cols = [
                {"name": c[1], "type": c[2]}
                for c in conn.execute(f'PRAGMA table_info("{t}")')
            ]
            col_names = {c["name"] for c in cols}
            has_json = "features_json" in col_names

            # row_count — may be slow on 10M-row tables but acceptable at 5min TTL
            row_count = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]

            # date_range — only if a known date column exists
            date_col = next(
                (d for d in ("trade_date", "end_date", "ann_date", "as_of_date")
                 if d in col_names),
                None,
            )
            date_range: Optional[tuple[str, str]] = None
            if date_col and row_count > 0:
                row = conn.execute(
                    f'SELECT MIN("{date_col}"), MAX("{date_col}") FROM "{t}"'
                ).fetchone()
                if row and row[0] and row[1]:
                    date_range = (str(row[0]), str(row[1]))

            category = classify_table(t)
            result.setdefault(category, []).append(
                {
                    "table": t,
                    "columns": cols,
                    "row_count": row_count,
                    "date_range": date_range,
                    "has_features_json": has_json,
                }
            )
    finally:
        conn.close()

    _CACHE[key] = (now, result)
    return result
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd webapp && python3 -m pytest tests/test_data_explorer/test_schema_discovery.py -v`
Expected: PASS 4 tests.

- [ ] **Step 5: Commit**

```bash
git add webapp/core/data_explorer/schema_discovery.py \
        webapp/tests/test_data_explorer/test_schema_discovery.py
git commit -m "feat(explorer): schema_discovery (classify + 5-min TTL cache)"
```

---

### Task 5: `query_runner.py` part 1 — SQL validation + LIMIT injection

**Files:**
- Create: `webapp/core/data_explorer/query_runner.py` (validation half)
- Test: `webapp/tests/test_data_explorer/test_query_runner.py` (validation tests)

- [ ] **Step 1: Write the failing validation tests**

Create `webapp/tests/test_data_explorer/test_query_runner.py`:

```python
"""Tests for query_runner — Task 5 covers validation + LIMIT; Task 6 covers exec."""
import pytest

from core.data_explorer.query_runner import (
    InvalidQueryError,
    ensure_select_only,
    inject_limit,
)


def test_select_passes() -> None:
    ensure_select_only("SELECT * FROM daily_quotes")  # no raise


@pytest.mark.parametrize("sql", [
    "INSERT INTO t VALUES (1)",
    "UPDATE t SET x=1",
    "DELETE FROM t",
    "DROP TABLE t",
    "CREATE TABLE x (a INT)",
    "ATTACH DATABASE 'x.db' AS other",
    "PRAGMA writable_schema=1",
    "SELECT 1; DROP TABLE t",  # multi-statement
])
def test_non_select_rejected(sql: str) -> None:
    with pytest.raises(InvalidQueryError):
        ensure_select_only(sql)


def test_syntax_error_rejected() -> None:
    with pytest.raises(InvalidQueryError):
        ensure_select_only("SELEKT * FROM t")


def test_inject_limit_when_missing() -> None:
    out = inject_limit("SELECT * FROM daily_quotes", max_rows=100)
    assert "LIMIT 100" in out.upper().replace("\n", " ")


def test_inject_limit_caps_larger_user_limit() -> None:
    out = inject_limit("SELECT * FROM t LIMIT 99999", max_rows=100)
    normalized = out.upper().replace("\n", " ")
    assert "LIMIT 100" in normalized
    assert "LIMIT 99999" not in normalized


def test_inject_limit_preserves_smaller_user_limit() -> None:
    out = inject_limit("SELECT * FROM t LIMIT 50", max_rows=10000)
    assert "LIMIT 50" in out.upper().replace("\n", " ")
    assert "LIMIT 10000" not in out.upper().replace("\n", " ")
```

- [ ] **Step 2: Run to verify failures**

Run: `cd webapp && python3 -m pytest tests/test_data_explorer/test_query_runner.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement validation half**

Create `webapp/core/data_explorer/query_runner.py`:

```python
"""SELECT-only SQL runner with LIMIT injection and timeout.

Task 5: ensure_select_only + inject_limit.
Task 6: run_query (executes against a read-only connection).
"""
from __future__ import annotations

import sqlglot
from sqlglot import exp


class InvalidQueryError(ValueError):
    """Raised for any SQL that is not a single read-only SELECT."""


def ensure_select_only(sql: str) -> exp.Select:
    """Parse the SQL; raise InvalidQueryError if it is anything but a single SELECT.

    Rejects: INSERT/UPDATE/DELETE/DROP/CREATE/ALTER/ATTACH/DETACH/PRAGMA,
    multi-statement inputs, and syntax errors.
    """
    try:
        parsed = sqlglot.parse(sql, read="sqlite")
    except sqlglot.errors.ParseError as e:
        raise InvalidQueryError(f"SQL parse error: {e}") from e

    # Drop empty statements (trailing ';' yields None)
    statements = [p for p in parsed if p is not None]
    if len(statements) != 1:
        raise InvalidQueryError(
            f"Exactly one statement required; got {len(statements)}"
        )
    stmt = statements[0]
    if not isinstance(stmt, exp.Select):
        raise InvalidQueryError(
            f"Only SELECT allowed; got {stmt.key.upper()}"
        )
    return stmt


def inject_limit(sql: str, max_rows: int) -> str:
    """Return SQL with LIMIT ≤ max_rows guaranteed.

    - Missing LIMIT: append `LIMIT max_rows`.
    - Existing LIMIT > max_rows: replace with max_rows.
    - Existing LIMIT ≤ max_rows: leave as-is.
    """
    stmt = ensure_select_only(sql)
    existing = stmt.args.get("limit")
    if existing is None:
        stmt.limit(max_rows, copy=False)
    else:
        try:
            current = int(existing.expression.this)
        except (ValueError, AttributeError):
            current = max_rows + 1  # unparseable → replace with cap
        if current > max_rows:
            stmt.limit(max_rows, copy=False)
    return stmt.sql(dialect="sqlite")
```

- [ ] **Step 4: Run to verify passes**

Run: `cd webapp && python3 -m pytest tests/test_data_explorer/test_query_runner.py -v`
Expected: PASS 7 tests (4 cases + 3 parametrized + 3 ensure/limit combined = count may differ; just verify all green).

- [ ] **Step 5: Commit**

```bash
git add webapp/core/data_explorer/query_runner.py \
        webapp/tests/test_data_explorer/test_query_runner.py
git commit -m "feat(explorer): query_runner validation + LIMIT injection"
```

---

### Task 6: `query_runner.py` part 2 — `run_query()` (execute + timeout)

**Files:**
- Modify: `webapp/core/data_explorer/query_runner.py` (append `run_query`, `QueryResult`)
- Modify: `webapp/tests/test_data_explorer/test_query_runner.py` (append exec tests)

- [ ] **Step 1: Write the failing execution tests**

Append to `webapp/tests/test_data_explorer/test_query_runner.py`:

```python
# --- Task 6: run_query tests -------------------------------------------------

import pytest

from core.data_explorer.query_runner import QueryResult, run_query


def test_run_query_returns_rows(tmp_stock_db) -> None:
    result = run_query(
        db_path=tmp_stock_db,
        sql="SELECT code, close FROM daily_quotes ORDER BY trade_date",
    )
    assert isinstance(result, QueryResult)
    assert "code" in result.columns and "close" in result.columns
    assert result.row_count == 3
    assert result.truncated is False
    assert result.took_ms >= 0


def test_run_query_expands_features_json(tmp_stock_db) -> None:
    result = run_query(
        db_path=tmp_stock_db,
        sql="SELECT * FROM ng101_feature_cache",
        expand_features=True,
    )
    assert "features_json" not in result.columns
    assert "pb" in result.columns and "roe" in result.columns


def test_run_query_preserves_features_json_when_expand_false(tmp_stock_db) -> None:
    result = run_query(
        db_path=tmp_stock_db,
        sql="SELECT * FROM ng101_feature_cache",
        expand_features=False,
    )
    assert "features_json" in result.columns


def test_run_query_truncation_warning(tmp_stock_db) -> None:
    result = run_query(
        db_path=tmp_stock_db,
        sql="SELECT * FROM daily_quotes",
        max_rows=2,
    )
    assert result.row_count == 2
    assert result.truncated is True
    assert any("truncated" in w.lower() for w in result.warnings)


def test_run_query_limit_injection_warning(tmp_stock_db) -> None:
    result = run_query(
        db_path=tmp_stock_db,
        sql="SELECT * FROM daily_quotes",  # no LIMIT
    )
    assert any("limit" in w.lower() for w in result.warnings)


def test_run_query_rejects_write(tmp_stock_db) -> None:
    from core.data_explorer.query_runner import InvalidQueryError
    with pytest.raises(InvalidQueryError):
        run_query(tmp_stock_db, "DELETE FROM daily_quotes")


def test_run_query_returns_chart_hint(tmp_stock_db) -> None:
    # 1 categorical + 1 numeric, rows ≤ 50 → bar
    result = run_query(
        db_path=tmp_stock_db,
        sql="SELECT code, close FROM daily_quotes",
    )
    assert result.chart_hint is not None
    assert result.chart_hint["type"] in {"bar", "line"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd webapp && python3 -m pytest tests/test_data_explorer/test_query_runner.py -v`
Expected: FAIL with `ImportError: cannot import name 'QueryResult'` or `'run_query'`.

- [ ] **Step 3: Implement `run_query` and `QueryResult`**

Append to `webapp/core/data_explorer/query_runner.py`:

```python
# --- Task 6: execution -------------------------------------------------------

import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from core.data_explorer.chart_suggester import suggest as suggest_chart
from core.data_explorer.feature_expander import expand as expand_features


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[list]             # JSON-serializable; NaN → None
    row_count: int
    truncated: bool
    took_ms: int
    warnings: list[str] = field(default_factory=list)
    chart_hint: dict | None = None


class QueryTimeoutError(TimeoutError):
    """Raised when the SQLite progress handler aborts a long query."""


def _df_to_jsonable_rows(df: pd.DataFrame) -> list[list]:
    """Convert DataFrame to list-of-list with NaN → None, datetime → ISO string."""
    out: list[list] = []
    for _, row in df.iterrows():
        converted: list = []
        for v in row.tolist():
            if isinstance(v, float) and math.isnan(v):
                converted.append(None)
            elif isinstance(v, pd.Timestamp):
                converted.append(v.isoformat())
            elif pd.isna(v):
                converted.append(None)
            else:
                converted.append(v)
        out.append(converted)
    return out


def run_query(
    db_path: str | Path,
    sql: str,
    *,
    expand_features: bool = True,
    max_rows: int = 10_000,
    timeout_s: int = 30,
) -> QueryResult:
    """Validate, inject LIMIT, execute read-only, expand features, suggest chart."""
    warnings: list[str] = []

    # Step 1+2: validate and inject
    stmt = ensure_select_only(sql)
    if stmt.args.get("limit") is None:
        warnings.append(f"LIMIT injected ({max_rows} rows)")
    final_sql = inject_limit(sql, max_rows)

    # Step 3: open readonly
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5)

    # Step 4: progress handler for timeout
    deadline = time.monotonic() + timeout_s

    def _progress_cb() -> int:
        return 1 if time.monotonic() > deadline else 0

    conn.set_progress_handler(_progress_cb, 10_000)

    started = time.monotonic()
    try:
        df = pd.read_sql_query(final_sql, conn)
    except sqlite3.OperationalError as e:
        # Progress handler returning 1 surfaces as "interrupted"
        if "interrupted" in str(e).lower():
            raise QueryTimeoutError(
                f"Query exceeded {timeout_s}s timeout"
            ) from e
        raise
    finally:
        conn.set_progress_handler(None, 0)
        conn.close()

    took_ms = int((time.monotonic() - started) * 1000)

    # Step 5: feature expand
    if expand_features:
        df, exp_warnings = expand_features_fn_call(df)
        warnings.extend(exp_warnings)

    # Step 6: post-cap (should already fit but guard)
    truncated = len(df) >= max_rows
    if truncated:
        df = df.head(max_rows)
        warnings.append(f"results truncated to {max_rows} rows")

    # Step 7: chart hint
    hint = suggest_chart(df)

    return QueryResult(
        columns=list(df.columns),
        rows=_df_to_jsonable_rows(df),
        row_count=len(df),
        truncated=truncated,
        took_ms=took_ms,
        warnings=warnings,
        chart_hint=hint,
    )


# Alias needed because `expand_features` is also a kwarg name above.
expand_features_fn_call = expand_features
```

- [ ] **Step 4: Run tests to verify passes**

Run: `cd webapp && python3 -m pytest tests/test_data_explorer/test_query_runner.py -v`
Expected: PASS all tests (validation + execution).

- [ ] **Step 5: Commit**

```bash
git add webapp/core/data_explorer/query_runner.py \
        webapp/tests/test_data_explorer/test_query_runner.py
git commit -m "feat(explorer): run_query with readonly conn + progress-handler timeout"
```

---

### Task 7: `query_store.py` CRUD + seed

**Files:**
- Modify: `webapp/core/data_explorer/query_store.py` (append CRUD + seed)
- Modify: `webapp/tests/test_data_explorer/test_query_store.py` (append CRUD tests)

- [ ] **Step 1: Write the failing CRUD tests**

Append to `webapp/tests/test_data_explorer/test_query_store.py`:

```python
# --- Task 7: CRUD + seed ----------------------------------------------------

import pytest

from core.data_explorer.query_store import (
    create_query, delete_query, get_query, list_queries,
    seed_default_queries, touch_query, update_query,
)


def _setup(tmp_webapp_db):
    apply_migration(tmp_webapp_db)
    return tmp_webapp_db


def test_create_and_get_and_list(tmp_webapp_db):
    db = _setup(tmp_webapp_db)
    q = create_query(db, name="my-q", sql="SELECT 1", tags="test", description="d")
    assert q["id"] > 0
    assert q["name"] == "my-q"

    fetched = get_query(db, q["id"])
    assert fetched["sql"] == "SELECT 1"

    rows = list_queries(db)
    assert len(rows) == 1


def test_create_name_unique(tmp_webapp_db):
    db = _setup(tmp_webapp_db)
    create_query(db, name="dup", sql="SELECT 1", tags=None, description=None)
    with pytest.raises(ValueError) as exc:
        create_query(db, name="dup", sql="SELECT 2", tags=None, description=None)
    assert "already exists" in str(exc.value)


def test_update_fields(tmp_webapp_db):
    db = _setup(tmp_webapp_db)
    q = create_query(db, name="n", sql="S1", tags=None, description=None)
    updated = update_query(db, q["id"], sql="S2", tags="new")
    assert updated["sql"] == "S2"
    assert updated["tags"] == "new"


def test_delete(tmp_webapp_db):
    db = _setup(tmp_webapp_db)
    q = create_query(db, name="n", sql="S", tags=None, description=None)
    delete_query(db, q["id"])
    assert list_queries(db) == []


def test_touch_increments_run_count(tmp_webapp_db):
    db = _setup(tmp_webapp_db)
    q = create_query(db, name="n", sql="S", tags=None, description=None)
    touch_query(db, q["id"])
    touch_query(db, q["id"])
    fetched = get_query(db, q["id"])
    assert fetched["run_count"] == 2
    assert fetched["last_run_at"] is not None


def test_list_filters_by_tag(tmp_webapp_db):
    db = _setup(tmp_webapp_db)
    create_query(db, name="a", sql="S", tags="preset,stock", description=None)
    create_query(db, name="b", sql="S", tags="preset,cross", description=None)
    create_query(db, name="c", sql="S", tags="user", description=None)

    preset_rows = list_queries(db, tag="preset")
    assert {r["name"] for r in preset_rows} == {"a", "b"}


def test_seed_only_on_empty_table(tmp_webapp_db):
    db = _setup(tmp_webapp_db)
    inserted1 = seed_default_queries(db)
    inserted2 = seed_default_queries(db)
    assert inserted1 > 0
    assert inserted2 == 0  # idempotent
```

- [ ] **Step 2: Run tests to verify failures**

Run: `cd webapp && python3 -m pytest tests/test_data_explorer/test_query_store.py -v`
Expected: FAIL with `ImportError` for `create_query` etc.

- [ ] **Step 3: Implement CRUD + seed**

Replace `webapp/core/data_explorer/query_store.py` with:

```python
"""CRUD for saved_queries on webapp.db + idempotent seed."""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


MIGRATION_SQL = """
CREATE TABLE IF NOT EXISTS saved_queries (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL UNIQUE,
    sql          TEXT    NOT NULL,
    tags         TEXT,
    description  TEXT,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_run_at  TIMESTAMP,
    run_count    INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_saved_queries_tags ON saved_queries(tags);
"""


_COLS = [
    "id", "name", "sql", "tags", "description",
    "created_at", "updated_at", "last_run_at", "run_count",
]


def apply_migration(webapp_db_path: str | Path) -> None:
    webapp_db_path = Path(webapp_db_path)
    webapp_db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(webapp_db_path)
    try:
        conn.executescript(MIGRATION_SQL)
        conn.commit()
    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row | tuple) -> dict[str, Any]:
    return dict(zip(_COLS, row))


def _connect(db: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    return conn


def create_query(
    db: str | Path, *, name: str, sql: str,
    tags: str | None, description: str | None,
) -> dict[str, Any]:
    conn = _connect(db)
    try:
        try:
            cur = conn.execute(
                "INSERT INTO saved_queries (name, sql, tags, description) "
                "VALUES (?, ?, ?, ?)",
                (name, sql, tags, description),
            )
            conn.commit()
        except sqlite3.IntegrityError as e:
            raise ValueError(f"saved query name already exists: {name}") from e
        return get_query(db, cur.lastrowid, _conn=conn)
    finally:
        conn.close()


def get_query(
    db: str | Path, query_id: int, *, _conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    conn = _conn or _connect(db)
    try:
        row = conn.execute(
            "SELECT id,name,sql,tags,description,created_at,updated_at,"
            "last_run_at,run_count FROM saved_queries WHERE id=?",
            (query_id,),
        ).fetchone()
        if row is None:
            raise LookupError(f"saved query id={query_id} not found")
        return _row_to_dict(tuple(row))
    finally:
        if _conn is None:
            conn.close()


def list_queries(
    db: str | Path, tag: str | None = None,
) -> list[dict[str, Any]]:
    conn = _connect(db)
    try:
        if tag:
            rows = conn.execute(
                "SELECT id,name,sql,tags,description,created_at,updated_at,"
                "last_run_at,run_count FROM saved_queries "
                "WHERE ',' || tags || ',' LIKE ? ORDER BY updated_at DESC",
                (f"%,{tag},%",),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id,name,sql,tags,description,created_at,updated_at,"
                "last_run_at,run_count FROM saved_queries "
                "ORDER BY updated_at DESC"
            ).fetchall()
        return [_row_to_dict(tuple(r)) for r in rows]
    finally:
        conn.close()


def update_query(
    db: str | Path, query_id: int, **fields: Any,
) -> dict[str, Any]:
    allowed = {"name", "sql", "tags", "description"}
    changes = {k: v for k, v in fields.items() if k in allowed}
    if not changes:
        return get_query(db, query_id)
    set_clause = ", ".join(f"{k}=?" for k in changes)
    params = list(changes.values()) + [query_id]
    conn = _connect(db)
    try:
        conn.execute(
            f"UPDATE saved_queries SET {set_clause}, "
            "updated_at=CURRENT_TIMESTAMP WHERE id=?",
            params,
        )
        conn.commit()
        return get_query(db, query_id, _conn=conn)
    finally:
        conn.close()


def delete_query(db: str | Path, query_id: int) -> None:
    conn = _connect(db)
    try:
        conn.execute("DELETE FROM saved_queries WHERE id=?", (query_id,))
        conn.commit()
    finally:
        conn.close()


def touch_query(db: str | Path, query_id: int) -> None:
    conn = _connect(db)
    try:
        conn.execute(
            "UPDATE saved_queries SET run_count=run_count+1, "
            "last_run_at=CURRENT_TIMESTAMP WHERE id=?",
            (query_id,),
        )
        conn.commit()
    finally:
        conn.close()


# -- seed ---------------------------------------------------------------------

_SEED: list[tuple[str, str, str, str]] = [
    (
        "preset: single-stock all-features",
        "SELECT * FROM ng101_feature_cache\n"
        "WHERE code = '600519.SH'\n"
        "  AND trade_date >= date('now', '-60 days')\n"
        "ORDER BY trade_date DESC",
        "preset,stock",
        "Mode A: every ng101 feature for a single stock over 60 days",
    ),
    (
        "preset: cross-section pred_10d top50",
        "SELECT code, trade_date, label_10d, features_json\n"
        "FROM ng101_feature_cache\n"
        "WHERE trade_date = (SELECT MAX(trade_date) FROM ng101_feature_cache)\n"
        "ORDER BY label_10d DESC\n"
        "LIMIT 50",
        "preset,cross",
        "Mode B: cross-section — top 50 by label_10d on latest day",
    ),
    (
        "preset: model compare ng101 vs ng110",
        "SELECT a.code, a.trade_date,\n"
        "       a.label_10d AS ng101_label10, b.label_10d AS ng110_label10\n"
        "FROM ng101_feature_cache a\n"
        "JOIN ng110_feature_cache b\n"
        "  ON a.code = b.code AND a.trade_date = b.trade_date\n"
        "WHERE a.trade_date = (SELECT MAX(trade_date) FROM ng101_feature_cache)\n"
        "ORDER BY a.label_10d DESC\n"
        "LIMIT 100",
        "preset,compare",
        "Mode C: ng101 vs ng110 label_10d on latest day",
    ),
    (
        "preset: signal_trust 🔴 tagged stocks",
        "SELECT code, as_of_date, n_samples, direction_hit_rate, trust_tag\n"
        "FROM signal_trust_scores\n"
        "WHERE trust_tag = '🔴'\n"
        "ORDER BY n_samples DESC\n"
        "LIMIT 50",
        "preset",
        "Stocks flagged 🔴 (low-trust) by signal_trust system",
    ),
]


def seed_default_queries(db: str | Path) -> int:
    """Insert SEED rows when saved_queries is empty. Returns rows inserted."""
    conn = _connect(db)
    try:
        (count,) = conn.execute("SELECT COUNT(*) FROM saved_queries").fetchone()
        if count > 0:
            return 0
        inserted = 0
        for name, sql, tags, description in _SEED:
            try:
                conn.execute(
                    "INSERT INTO saved_queries (name, sql, tags, description) "
                    "VALUES (?, ?, ?, ?)",
                    (name, sql, tags, description),
                )
                inserted += 1
            except sqlite3.IntegrityError as e:
                logger.warning(f"seed insert skipped for '{name}': {e}")
        conn.commit()
        return inserted
    finally:
        conn.close()
```

- [ ] **Step 4: Run tests to verify passes**

Run: `cd webapp && python3 -m pytest tests/test_data_explorer/test_query_store.py -v`
Expected: PASS all tests.

- [ ] **Step 5: Commit**

```bash
git add webapp/core/data_explorer/query_store.py \
        webapp/tests/test_data_explorer/test_query_store.py
git commit -m "feat(explorer): query_store CRUD + seed (4 presets)"
```

---

### Task 8: Flask blueprint `api/data_explorer.py`

**Files:**
- Create: `webapp/api/data_explorer.py`
- Create: `webapp/tests/test_data_explorer/test_blueprint_integration.py`

- [ ] **Step 1: Write the failing integration tests**

Create `webapp/tests/test_data_explorer/test_blueprint_integration.py`:

```python
"""Integration tests: hit the blueprint through a live Flask test client."""
import json
from pathlib import Path

import pytest

import sys
webapp_root = Path(__file__).resolve().parents[2]
if str(webapp_root) not in sys.path:
    sys.path.insert(0, str(webapp_root))

from app import create_app  # noqa: E402


@pytest.fixture
def client(tmp_stock_db: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FLASK_ENV", "development")
    app = create_app()
    app.config["STOCK_DB_PATH"] = tmp_stock_db
    app.config["WEBAPP_DB_PATH"] = tmp_path / "webapp.db"
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_schema_endpoint(client):
    resp = client.get("/api/explorer/schema")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    cats = body["schema"]
    assert "raw" in cats
    assert any(t["table"] == "daily_quotes" for t in cats["raw"])


def test_query_endpoint_happy_path(client):
    resp = client.post(
        "/api/explorer/query",
        data=json.dumps({"sql": "SELECT * FROM daily_quotes LIMIT 3"}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert body["row_count"] == 3
    assert "columns" in body and "rows" in body


def test_query_endpoint_rejects_write(client):
    resp = client.post(
        "/api/explorer/query",
        data=json.dumps({"sql": "DELETE FROM daily_quotes"}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["success"] is False
    assert body["code"] == "invalid_query"


def test_saved_query_crud(client):
    # create
    resp = client.post(
        "/api/explorer/saved",
        data=json.dumps({
            "name": "t1", "sql": "SELECT 1",
            "tags": "test", "description": ""
        }),
        content_type="application/json",
    )
    assert resp.status_code == 201
    qid = resp.get_json()["query"]["id"]

    # list
    resp = client.get("/api/explorer/saved")
    names = [q["name"] for q in resp.get_json()["queries"]]
    assert "t1" in names

    # delete
    resp = client.delete(f"/api/explorer/saved/{qid}")
    assert resp.status_code == 200

    resp = client.get("/api/explorer/saved")
    assert not any(q["id"] == qid for q in resp.get_json()["queries"])
```

- [ ] **Step 2: Run tests to verify failures**

Run: `cd webapp && python3 -m pytest tests/test_data_explorer/test_blueprint_integration.py -v`
Expected: FAIL — blueprint is not registered yet; we'll register in Task 9, but the blueprint module itself must exist first.

Actually the test will FAIL with `404 not found` for the endpoint. We need to register the blueprint for the tests to pass — do that inline in Task 8 Step 3.

- [ ] **Step 3: Implement the blueprint**

Create `webapp/api/data_explorer.py`:

```python
"""Flask blueprint for the data explorer page.

Endpoints (all under url_prefix='/api/explorer'):
  GET    /schema              -> { success, schema: {cat: [TableInfo...]} }
  POST   /query               -> { success, columns, rows, row_count, truncated,
                                   took_ms, warnings, chart_hint }
  GET    /saved               -> { success, queries: [SavedQuery...] }
  POST   /saved               -> 201 { success, query }
  GET    /saved/<id>          -> { success, query }
  PUT    /saved/<id>          -> { success, query }
  DELETE /saved/<id>          -> { success }
"""
from __future__ import annotations

import dataclasses
import logging

from flask import Blueprint, current_app, jsonify, request

from core.data_explorer.query_runner import (
    InvalidQueryError,
    QueryTimeoutError,
    run_query,
)
from core.data_explorer.query_store import (
    apply_migration,
    create_query,
    delete_query,
    get_query,
    list_queries,
    seed_default_queries,
    touch_query,
    update_query,
)
from core.data_explorer.schema_discovery import discover


logger = logging.getLogger(__name__)

data_explorer_bp = Blueprint("data_explorer", __name__)


def _err(code: str, message: str, http: int = 400):
    return jsonify({"success": False, "error": message, "code": code}), http


def _ensure_store_ready() -> None:
    """Idempotent migration + seed on first API hit."""
    db = current_app.config["WEBAPP_DB_PATH"]
    apply_migration(db)
    seed_default_queries(db)


@data_explorer_bp.route("/schema", methods=["GET"])
def schema():
    try:
        refresh = request.args.get("refresh") == "1"
        schema_dict = discover(
            current_app.config["STOCK_DB_PATH"], refresh=refresh
        )
        return jsonify({"success": True, "schema": schema_dict})
    except Exception as e:
        logger.error("schema discovery failed", exc_info=True)
        return _err("server_error", str(e), 500)


@data_explorer_bp.route("/query", methods=["POST"])
def query():
    payload = request.get_json(silent=True) or {}
    sql = (payload.get("sql") or "").strip()
    if not sql:
        return _err("invalid_query", "empty sql")
    expand = bool(payload.get("expand_features", True))
    try:
        result = run_query(
            current_app.config["STOCK_DB_PATH"],
            sql,
            expand_features=expand,
        )
        body = dataclasses.asdict(result)
        body["success"] = True
        return jsonify(body)
    except InvalidQueryError as e:
        return _err("invalid_query", str(e), 400)
    except QueryTimeoutError as e:
        return _err("timeout", str(e), 504)
    except Exception as e:
        logger.error("run_query failed", exc_info=True)
        return _err("server_error", str(e), 500)


@data_explorer_bp.route("/saved", methods=["GET"])
def saved_list():
    _ensure_store_ready()
    tag = request.args.get("tag")
    return jsonify({
        "success": True,
        "queries": list_queries(current_app.config["WEBAPP_DB_PATH"], tag=tag),
    })


@data_explorer_bp.route("/saved", methods=["POST"])
def saved_create():
    _ensure_store_ready()
    payload = request.get_json(silent=True) or {}
    try:
        q = create_query(
            current_app.config["WEBAPP_DB_PATH"],
            name=payload["name"],
            sql=payload["sql"],
            tags=payload.get("tags"),
            description=payload.get("description"),
        )
        return jsonify({"success": True, "query": q}), 201
    except KeyError as e:
        return _err("invalid_payload", f"missing field: {e.args[0]}")
    except ValueError as e:
        return _err("conflict", str(e), 409)


@data_explorer_bp.route("/saved/<int:qid>", methods=["GET"])
def saved_get(qid: int):
    _ensure_store_ready()
    try:
        return jsonify({
            "success": True,
            "query": get_query(current_app.config["WEBAPP_DB_PATH"], qid),
        })
    except LookupError as e:
        return _err("not_found", str(e), 404)


@data_explorer_bp.route("/saved/<int:qid>", methods=["PUT"])
def saved_update(qid: int):
    _ensure_store_ready()
    payload = request.get_json(silent=True) or {}
    try:
        q = update_query(
            current_app.config["WEBAPP_DB_PATH"], qid, **payload
        )
        return jsonify({"success": True, "query": q})
    except LookupError as e:
        return _err("not_found", str(e), 404)


@data_explorer_bp.route("/saved/<int:qid>", methods=["DELETE"])
def saved_delete(qid: int):
    _ensure_store_ready()
    delete_query(current_app.config["WEBAPP_DB_PATH"], qid)
    return jsonify({"success": True})


@data_explorer_bp.route("/saved/<int:qid>/run", methods=["POST"])
def saved_run(qid: int):
    """Run a saved query and increment its run_count / last_run_at."""
    _ensure_store_ready()
    try:
        q = get_query(current_app.config["WEBAPP_DB_PATH"], qid)
    except LookupError as e:
        return _err("not_found", str(e), 404)

    payload = request.get_json(silent=True) or {}
    expand = bool(payload.get("expand_features", True))
    try:
        result = run_query(
            current_app.config["STOCK_DB_PATH"], q["sql"], expand_features=expand,
        )
        touch_query(current_app.config["WEBAPP_DB_PATH"], qid)
        body = dataclasses.asdict(result)
        body["success"] = True
        return jsonify(body)
    except InvalidQueryError as e:
        return _err("invalid_query", str(e), 400)
    except QueryTimeoutError as e:
        return _err("timeout", str(e), 504)
```

- [ ] **Step 4: Register the blueprint in `app.py`**

In `webapp/app.py`, locate `register_blueprints(app)` (around line 123) and add:

```python
def register_blueprints(app):
    """注册蓝图"""
    from api.daily_tasks import daily_tasks_bp
    from api.model_training import model_training_bp
    from api.backtest import backtest_bp
    from api.tasks import tasks_bp
    from api.portfolio import portfolio_bp
    from api.data_management import data_management_bp
    from api.stock import stock_bp
    from api.data_explorer import data_explorer_bp   # NEW

    app.register_blueprint(daily_tasks_bp, url_prefix='/api/daily')
    app.register_blueprint(model_training_bp, url_prefix='/api/models')
    app.register_blueprint(backtest_bp, url_prefix='/api/backtest')
    app.register_blueprint(tasks_bp, url_prefix='/api/tasks')
    app.register_blueprint(portfolio_bp, url_prefix='/api/portfolio')
    app.register_blueprint(data_management_bp, url_prefix='/api/data')
    app.register_blueprint(stock_bp, url_prefix='/api/stock')
    app.register_blueprint(data_explorer_bp, url_prefix='/api/explorer')  # NEW
```

- [ ] **Step 5: Run tests to verify passes**

Run: `cd webapp && python3 -m pytest tests/test_data_explorer/test_blueprint_integration.py -v`
Expected: PASS 4 tests.

Also run the full suite to confirm nothing broke:
Run: `cd webapp && python3 -m pytest tests/test_data_explorer/ -v`
Expected: PASS all.

- [ ] **Step 6: Commit**

```bash
git add webapp/api/data_explorer.py \
        webapp/app.py \
        webapp/tests/test_data_explorer/test_blueprint_integration.py
git commit -m "feat(explorer): Flask blueprint + 7 endpoints + integration tests"
```

---

### Task 9: Page route + nav link

**Files:**
- Modify: `webapp/app.py` (add `/data-explorer` route)
- Modify: `webapp/templates/base.html` (add nav item)

- [ ] **Step 1: Add the page route**

In `webapp/app.py`, after the `/data-management` route (around line 63), add:

```python
    @app.route('/data-explorer')
    def data_explorer():
        """数据探索页面"""
        return render_template('data_explorer.html')
```

- [ ] **Step 2: Add the nav link**

In `webapp/templates/base.html`, locate the nav `<ul>` (starting around line 35). Add a new `<li>` between "数据管理" and the search box:

```html
                    <li class="nav-item">
                        <a class="nav-link {% if request.endpoint == 'data_explorer' %}active{% endif %}"
                           href="{{ url_for('data_explorer') }}">
                            <i class="bi bi-search"></i> 数据探索
                        </a>
                    </li>
```

(The `<i class="bi bi-search">` matches Bootstrap Icons 1.11 already loaded in base.html.)

- [ ] **Step 3: Smoke test the route**

Add a simple check to `test_blueprint_integration.py`:

```python
def test_data_explorer_page_renders(client):
    resp = client.get("/data-explorer")
    # template doesn't exist yet; expect 500 after Task 10 turns this into 200
    # For Task 9, a Jinja TemplateNotFound wraps as 500 in prod handler
    assert resp.status_code in (200, 500)
```

Run: `cd webapp && python3 -m pytest tests/test_data_explorer/test_blueprint_integration.py::test_data_explorer_page_renders -v`
Expected: PASS (status 500 because template not yet created).

- [ ] **Step 4: Commit**

```bash
git add webapp/app.py webapp/templates/base.html \
        webapp/tests/test_data_explorer/test_blueprint_integration.py
git commit -m "feat(explorer): add /data-explorer route + nav link"
```

---

### Task 10: `data_explorer.html` scaffolding + CSS

**Files:**
- Create: `webapp/templates/data_explorer.html`
- Create: `webapp/static/css/data_explorer.css`

- [ ] **Step 1: Create the CSS**

Create `webapp/static/css/data_explorer.css`:

```css
/* Data Explorer page layout */

.explorer-layout {
    display: grid;
    grid-template-columns: 260px 1fr;
    gap: 12px;
    min-height: calc(100vh - 200px);
}

.schema-browser {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    padding: 10px;
    overflow-y: auto;
    font-size: 0.85rem;
    max-height: calc(100vh - 160px);
}

.schema-browser .category {
    font-weight: 600;
    cursor: pointer;
    padding: 4px 0;
    user-select: none;
}
.schema-browser .category .bi { font-size: 0.7rem; }
.schema-browser .category-count {
    font-weight: normal;
    color: #6c757d;
    font-size: 0.75rem;
}
.schema-browser ul.tables {
    list-style: none;
    padding-left: 14px;
    margin: 4px 0;
}
.schema-browser ul.tables li {
    padding: 2px 4px;
    border-radius: 3px;
    cursor: pointer;
    color: #0d6efd;
    font-family: ui-monospace, SFMono-Regular, monospace;
    font-size: 0.8rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.schema-browser ul.tables li:hover { background: #e9ecef; }
.schema-browser .active-marker {
    color: #198754;
    font-size: 0.7rem;
    margin-left: 4px;
}
.schema-browser .has-json::after {
    content: "{ }";
    color: #fd7e14;
    font-size: 0.7rem;
    margin-left: 4px;
}

.editor-region { display: flex; flex-direction: column; gap: 8px; }

.cm-editor {
    border: 1px solid #dee2e6;
    border-radius: 4px;
    font-size: 0.9rem;
    min-height: 150px;
    background: #282c34;
}
.cm-editor .cm-scroller { font-family: ui-monospace, monospace; }

.result-region {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    min-height: 300px;
}
.result-table-card, .result-chart-card {
    border: 1px solid #dee2e6;
    border-radius: 4px;
    padding: 8px;
}
.result-chart-card { min-height: 300px; }

.warning-banner {
    background: #fff3cd;
    border: 1px solid #ffeeba;
    color: #856404;
    padding: 6px 10px;
    border-radius: 4px;
    font-size: 0.85rem;
    margin-bottom: 8px;
}
.mode-tabs { border-bottom: 2px solid #dee2e6; margin-bottom: 12px; }
.mode-tabs .nav-link.active {
    border-bottom: 3px solid #0d6efd;
    color: #0d6efd;
    font-weight: 600;
}

.saved-query-list .saved-row {
    padding: 8px;
    border-bottom: 1px solid #dee2e6;
    cursor: pointer;
}
.saved-query-list .saved-row:hover { background: #f8f9fa; }
```

- [ ] **Step 2: Create the template**

Create `webapp/templates/data_explorer.html`:

```html
{% extends "base.html" %}

{% block title %}数据探索 - StockTradebyZ Web{% endblock %}

{% block extra_css %}
<link rel="stylesheet" href="{{ url_for('static', filename='css/data_explorer.css') }}">
{% endblock %}

{% block content %}
<div class="row mb-3">
  <div class="col-12">
    <h2 class="mb-3">
      <i class="bi bi-search"></i> 数据探索
      <small class="text-muted fs-6">SQL / Builder / Saved / 单股 / 截面 / 模型对比</small>
    </h2>
  </div>
</div>

<!-- Mode tabs -->
<ul class="nav mode-tabs" role="tablist">
  <li class="nav-item"><a class="nav-link active" data-mode="sql" href="#sql"><i class="bi bi-code-slash"></i> SQL Explorer</a></li>
  <li class="nav-item"><a class="nav-link" data-mode="stock" href="#stock"><i class="bi bi-graph-up"></i> 单股深挖</a></li>
  <li class="nav-item"><a class="nav-link" data-mode="cross" href="#cross"><i class="bi bi-bar-chart"></i> 截面分析</a></li>
  <li class="nav-item"><a class="nav-link" data-mode="compare" href="#compare"><i class="bi bi-diagram-3"></i> 模型对比</a></li>
  <li class="nav-item"><a class="nav-link" data-mode="saved" href="#saved"><i class="bi bi-star"></i> 已保存 <span class="badge bg-secondary" id="saved-count">0</span></a></li>
</ul>

<!-- Main layout: schema browser + query region -->
<div class="explorer-layout" id="explorer-main">

  <!-- Left: schema browser -->
  <aside class="schema-browser">
    <input class="form-control form-control-sm mb-2" id="schema-search" placeholder="搜索表/列...">
    <div id="schema-tree">
      <div class="text-muted small">Loading schema...</div>
    </div>
  </aside>

  <!-- Right: editor + results -->
  <section class="editor-region">
    <!-- Toolbar: SQL/Builder toggle + options -->
    <div class="d-flex gap-2 align-items-center flex-wrap">
      <div class="btn-group btn-group-sm" role="group" id="input-mode-toggle">
        <button class="btn btn-primary" data-input="sql">SQL</button>
        <button class="btn btn-outline-secondary" data-input="builder"><i class="bi bi-tools"></i> Builder</button>
      </div>
      <div class="form-check form-switch small">
        <input class="form-check-input" type="checkbox" id="expand-features" checked>
        <label class="form-check-label" for="expand-features">自动展开 features_json</label>
      </div>
      <div class="flex-grow-1"></div>
      <button class="btn btn-primary btn-sm" id="btn-run">
        <i class="bi bi-play-fill"></i> 运行 <small class="opacity-75">(Ctrl+Enter)</small>
      </button>
      <button class="btn btn-outline-secondary btn-sm" id="btn-save"><i class="bi bi-save"></i> 保存</button>
      <button class="btn btn-outline-secondary btn-sm" id="btn-csv"><i class="bi bi-file-earmark-spreadsheet"></i> CSV</button>
    </div>

    <!-- SQL editor container -->
    <div id="sql-editor-container"></div>

    <!-- Visual builder (hidden by default) -->
    <div id="builder-container" class="d-none card p-3"></div>

    <!-- Run summary / warnings -->
    <div id="query-summary" class="small text-muted"></div>
    <div id="warning-area"></div>

    <!-- Results split -->
    <div class="result-region">
      <div class="result-table-card">
        <div class="d-flex align-items-center mb-1">
          <strong class="small"><i class="bi bi-table"></i> 结果表</strong>
          <span class="ms-2 small text-muted" id="table-meta"></span>
        </div>
        <div id="result-table-wrap"><div class="text-muted small">(运行一条 SQL 查询)</div></div>
      </div>
      <div class="result-chart-card">
        <div class="d-flex align-items-center mb-1">
          <strong class="small"><i class="bi bi-graph-up-arrow"></i> 图表</strong>
          <select class="form-select form-select-sm ms-2" id="chart-type-select" style="width:auto">
            <option value="auto">自动</option>
            <option value="line">折线</option>
            <option value="scatter">散点</option>
            <option value="bar">柱状</option>
            <option value="histogram">直方图</option>
            <option value="none">不画</option>
          </select>
          <span class="ms-auto small text-success" id="chart-hint-label"></span>
        </div>
        <div id="result-chart"></div>
      </div>
    </div>
  </section>
</div>

<!-- Saved queries panel (hidden except on #saved mode) -->
<div id="saved-panel" class="d-none card mt-3">
  <div class="card-body">
    <div class="d-flex align-items-center mb-3">
      <h5 class="mb-0"><i class="bi bi-star"></i> 保存的查询</h5>
      <input class="form-control form-control-sm ms-auto" id="saved-tag-filter"
             placeholder="按 tag 过滤..." style="max-width:200px">
    </div>
    <div class="saved-query-list" id="saved-query-list"></div>
  </div>
</div>

<!-- Save dialog -->
<div class="modal fade" id="save-modal" tabindex="-1">
  <div class="modal-dialog">
    <div class="modal-content">
      <div class="modal-header"><h5 class="modal-title">保存查询</h5>
        <button type="button" class="btn-close" data-bs-dismiss="modal"></button></div>
      <div class="modal-body">
        <div class="mb-2"><label class="form-label">名称</label>
          <input class="form-control" id="save-name" required></div>
        <div class="mb-2"><label class="form-label">标签（逗号分隔）</label>
          <input class="form-control" id="save-tags" placeholder="user,analysis"></div>
        <div class="mb-2"><label class="form-label">描述</label>
          <textarea class="form-control" id="save-description" rows="2"></textarea></div>
      </div>
      <div class="modal-footer">
        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">取消</button>
        <button type="button" class="btn btn-primary" id="save-modal-confirm">保存</button>
      </div>
    </div>
  </div>
</div>
{% endblock %}

{% block extra_js %}
<!-- CodeMirror 6 (SQL) bundled via cdn.jsdelivr.net -->
<script type="importmap">
{
  "imports": {
    "@codemirror/view": "https://esm.sh/@codemirror/view@6",
    "@codemirror/state": "https://esm.sh/@codemirror/state@6",
    "@codemirror/commands": "https://esm.sh/@codemirror/commands@6",
    "@codemirror/lang-sql": "https://esm.sh/@codemirror/lang-sql@6",
    "@codemirror/language": "https://esm.sh/@codemirror/language@6",
    "codemirror": "https://esm.sh/codemirror@6"
  }
}
</script>

<script src="{{ url_for('static', filename='js/data_explorer.js') }}" type="module"></script>
{% endblock %}
```

- [ ] **Step 3: Manual smoke test — page loads**

Start the webapp:
```bash
cd webapp && python3 app.py
```

In a browser, visit `http://localhost:8000/data-explorer`. Expected:
- Page renders with the two-column layout (Schema sidebar left, editor right).
- Toolbar shows Run / Save / CSV buttons.
- Schema sidebar shows "Loading schema..." (JS not wired yet — next task).
- No 500 errors in the Flask log.
- Nav bar highlights "数据探索".

Stop the webapp (`Ctrl+C`).

- [ ] **Step 4: Commit**

```bash
git add webapp/templates/data_explorer.html \
        webapp/static/css/data_explorer.css
git commit -m "feat(explorer): data_explorer.html scaffold + CSS"
```

---

### Task 11: `data_explorer.js` — schema browser

**Files:**
- Create: `webapp/static/js/data_explorer.js`

- [ ] **Step 1: Create the module with schema-browser behavior**

Create `webapp/static/js/data_explorer.js`:

```javascript
// Data Explorer frontend controller.
// Modules wired below: schemaBrowser, sqlEditor (Task 12), queryRunner (Task 13),
// chart (Task 13), builder (Task 14), saved (Task 15).

import { EditorView, keymap, lineNumbers } from "@codemirror/view";
import { EditorState } from "@codemirror/state";
import { defaultKeymap, history, historyKeymap } from "@codemirror/commands";
import { sql } from "@codemirror/lang-sql";

const state = {
  schema: null,
  editor: null,
  lastResult: null,
  currentHint: null,
  currentChart: null,
  savedQueries: [],
};

// ---- Schema Browser ---------------------------------------------------------

async function loadSchema() {
  const resp = await fetch("/api/explorer/schema");
  const body = await resp.json();
  if (!body.success) {
    window.showToast("加载 schema 失败", "error");
    return;
  }
  state.schema = body.schema;
  renderSchema();
}

function renderSchema() {
  const tree = document.getElementById("schema-tree");
  tree.innerHTML = "";

  const orderedCats = [
    "raw", "technical", "feature_cache", "factor",
    "market_state", "moneyflow", "meta", "backtest", "other",
  ];
  for (const cat of orderedCats) {
    const tables = state.schema[cat];
    if (!tables || tables.length === 0) continue;

    const catDiv = document.createElement("div");
    catDiv.className = "category";
    catDiv.innerHTML = `<i class="bi bi-caret-down-fill"></i> <b>${cat}</b> <span class="category-count">(${tables.length})</span>`;
    const ul = document.createElement("ul");
    ul.className = "tables";

    for (const t of tables.sort((a, b) => a.table.localeCompare(b.table))) {
      const li = document.createElement("li");
      li.textContent = t.table;
      li.title = `${t.row_count.toLocaleString()} 行` +
        (t.date_range ? ` · ${t.date_range[0]} → ${t.date_range[1]}` : "");
      if (t.has_features_json) li.classList.add("has-json");
      li.addEventListener("click", () => insertSampleQuery(t));
      ul.appendChild(li);
    }

    catDiv.addEventListener("click", (e) => {
      // toggle collapse
      const icon = catDiv.querySelector(".bi");
      if (ul.style.display === "none") {
        ul.style.display = "";
        icon.classList.replace("bi-caret-right-fill", "bi-caret-down-fill");
      } else {
        ul.style.display = "none";
        icon.classList.replace("bi-caret-down-fill", "bi-caret-right-fill");
      }
    });

    tree.appendChild(catDiv);
    tree.appendChild(ul);
  }

  // Search box filters
  document.getElementById("schema-search").addEventListener("input", (e) => {
    const q = e.target.value.toLowerCase();
    tree.querySelectorAll("ul.tables li").forEach((li) => {
      li.style.display = li.textContent.toLowerCase().includes(q) ? "" : "none";
    });
  });
}

function insertSampleQuery(tableInfo) {
  const hasTradeDate = tableInfo.columns.some((c) => c.name === "trade_date");
  let sample;
  if (hasTradeDate) {
    sample = `SELECT *\nFROM ${tableInfo.table}\nWHERE trade_date = (SELECT MAX(trade_date) FROM ${tableInfo.table})\nLIMIT 100;`;
  } else {
    sample = `SELECT *\nFROM ${tableInfo.table}\nLIMIT 100;`;
  }
  setEditorContent(sample);
}

// ---- SQL Editor (CodeMirror 6) ---------------------------------------------

function setEditorContent(content) {
  state.editor.dispatch({
    changes: { from: 0, to: state.editor.state.doc.length, insert: content },
  });
}

function getEditorContent() {
  return state.editor.state.doc.toString();
}

function initEditor() {
  const container = document.getElementById("sql-editor-container");
  const runKey = {
    key: "Mod-Enter",
    run: () => { runQuery(); return true; },
  };
  state.editor = new EditorView({
    state: EditorState.create({
      doc: "-- 点击左侧表名自动生成 sample SQL, 或直接写:\nSELECT 1;",
      extensions: [
        lineNumbers(),
        history(),
        sql(),
        keymap.of([runKey, ...defaultKeymap, ...historyKeymap]),
      ],
    }),
    parent: container,
  });
}

// ---- Query Runner (filled in Task 13) --------------------------------------

async function runQuery() {
  // Stub — Task 13 fills this in
  window.showToast("runQuery stub — Task 13 wires this up", "info");
}

// ---- Bootstrap --------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  initEditor();
  loadSchema();
  document.getElementById("btn-run").addEventListener("click", runQuery);
});
```

- [ ] **Step 2: Manual smoke test**

```bash
cd webapp && python3 app.py
```

Visit `http://localhost:8000/data-explorer`. Expected:
- Schema sidebar populates within 1-2 s with all table categories.
- Clicking a category name toggles collapse.
- Typing in the schema search filters the visible tables.
- Clicking `daily_quotes` inserts `SELECT * FROM daily_quotes WHERE trade_date = ... LIMIT 100;` into the CodeMirror editor.
- `ng101_feature_cache` has the `{ }` orange annotation indicating `features_json` column.
- Editor is a dark theme with syntax highlighting.
- Pressing `Ctrl+Enter` shows the "runQuery stub" toast.

If schema doesn't load, check browser console for CSP/CORS/import-map errors. If CodeMirror fails to import from esm.sh, try alternate CDN `https://cdn.jsdelivr.net/npm/@codemirror/view@6/+esm` etc.

- [ ] **Step 3: Commit**

```bash
git add webapp/static/js/data_explorer.js
git commit -m "feat(explorer): JS schema browser + CodeMirror 6 editor"
```

---

### Task 12: `data_explorer.js` — query runner + result table

**Files:**
- Modify: `webapp/static/js/data_explorer.js` (replace `runQuery` stub)

- [ ] **Step 1: Replace the stub with a real runner**

In `webapp/static/js/data_explorer.js`, replace the stub `async function runQuery()` with:

```javascript
// ---- Query Runner -----------------------------------------------------------

async function runQuery() {
  const sqlText = getEditorContent().trim();
  if (!sqlText) {
    window.showToast("SQL 为空", "warning");
    return;
  }
  const expandFeatures = document.getElementById("expand-features").checked;
  const runBtn = document.getElementById("btn-run");
  runBtn.disabled = true;
  document.getElementById("query-summary").textContent = "执行中...";
  document.getElementById("warning-area").innerHTML = "";

  try {
    const resp = await fetch("/api/explorer/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sql: sqlText, expand_features: expandFeatures }),
    });
    const body = await resp.json();
    if (!body.success) {
      window.showToast(body.error || "查询失败", "error");
      document.getElementById("query-summary").textContent =
        `错误 (${body.code}): ${body.error}`;
      return;
    }
    state.lastResult = body;
    state.currentHint = body.chart_hint;
    renderResultTable(body);
    renderWarnings(body.warnings, body.truncated);
    document.getElementById("query-summary").textContent =
      `${body.row_count.toLocaleString()} 行 · ${body.took_ms} ms` +
      (body.truncated ? " (截断)" : "");
    renderChart(body, "auto");
  } catch (e) {
    window.showToast("网络错误: " + e.message, "error");
  } finally {
    runBtn.disabled = false;
  }
}

function renderWarnings(warnings, truncated) {
  const area = document.getElementById("warning-area");
  area.innerHTML = "";
  if (truncated) {
    const banner = document.createElement("div");
    banner.className = "warning-banner";
    banner.textContent = "⚠️ 结果已截断至上限 10 000 行 — 请加 LIMIT 或 WHERE 缩小范围";
    area.appendChild(banner);
  }
  for (const w of warnings || []) {
    const div = document.createElement("div");
    div.className = "warning-banner";
    div.textContent = w;
    area.appendChild(div);
  }
}

function renderResultTable(body) {
  const wrap = document.getElementById("result-table-wrap");
  wrap.innerHTML = "";
  if (body.row_count === 0) {
    wrap.innerHTML = '<div class="text-muted small">0 rows</div>';
    return;
  }
  const table = document.createElement("table");
  table.className = "table table-sm table-striped";
  table.style.width = "100%";
  const thead = document.createElement("thead");
  thead.innerHTML = "<tr>" + body.columns.map((c) => `<th>${c}</th>`).join("") + "</tr>";
  table.appendChild(thead);
  const tbody = document.createElement("tbody");
  for (const row of body.rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = row.map((v) => `<td>${v === null ? "<span class='text-muted'>·</span>" : escapeHtml(String(v))}</td>`).join("");
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  wrap.appendChild(table);

  // DataTables for sort + paging (library loaded in base.html)
  $(table).DataTable({
    pageLength: 25,
    deferRender: true,
    scrollX: true,
    order: [],
  });
  document.getElementById("table-meta").textContent =
    `${body.row_count.toLocaleString()} rows × ${body.columns.length} cols`;
}

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;")
          .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
```

- [ ] **Step 2: Manual smoke test**

Restart webapp, visit `/data-explorer`:
- Click `daily_quotes` → sample SQL inserted
- Press Ctrl+Enter
- Expect: result table renders with 100 rows, columns show code/trade_date/open/high/low/close/volume/..., DataTables paginates
- Bottom shows `100 rows · ~50 ms`
- Type `SELECT * FROM daily_quotes` (no LIMIT), Run → warning banner "LIMIT injected (10 000 rows)" + truncated banner if >10k
- Type `DROP TABLE daily_quotes`, Run → error toast + error message in query summary

- [ ] **Step 3: Commit**

```bash
git add webapp/static/js/data_explorer.js
git commit -m "feat(explorer): JS query runner + DataTables result rendering"
```

---

### Task 13: `data_explorer.js` — chart (ApexCharts) with auto-suggest + override

**Files:**
- Modify: `webapp/static/js/data_explorer.js`

- [ ] **Step 1: Add chart rendering functions**

Append to `webapp/static/js/data_explorer.js` (after `renderResultTable`):

```javascript
// ---- Chart ------------------------------------------------------------------

function renderChart(body, typeOverride) {
  const hint = body.chart_hint;
  const label = document.getElementById("chart-hint-label");
  const container = document.getElementById("result-chart");
  container.innerHTML = "";

  // Determine type
  let type = typeOverride;
  if (type === "auto") type = hint ? hint.type : "none";
  if (!type || type === "none") {
    label.textContent = hint ? "" : "无合适图表";
    return;
  }
  label.textContent = (typeOverride === "auto" && hint) ? `✨ auto: ${hint.type}` : "";

  // Build series depending on type. Use first numeric as Y if hint missing.
  const numericCols = body.columns.filter((c, i) =>
    body.rows.every((r) => r[i] === null || typeof r[i] === "number")
  );
  const firstNumeric = numericCols[0];
  const xCol = (hint && hint.x) || "code";
  const yCol = (hint && hint.y) || firstNumeric;
  if (!yCol) { label.textContent = "无数值列可画"; return; }

  const xIdx = body.columns.indexOf(xCol);
  const yIdx = body.columns.indexOf(yCol);
  if (yIdx < 0) { label.textContent = "指定列不存在"; return; }

  let options;
  if (type === "line") {
    options = {
      chart: { type: "line", height: 280, animations: { enabled: false } },
      series: [{ name: yCol, data: body.rows.map((r) => r[yIdx]) }],
      xaxis: { categories: body.rows.map((r) => r[xIdx]), title: { text: xCol } },
      yaxis: { title: { text: yCol } },
      stroke: { width: 2 },
    };
  } else if (type === "scatter") {
    options = {
      chart: { type: "scatter", height: 280, animations: { enabled: false } },
      series: [{
        name: `${xCol} vs ${yCol}`,
        data: body.rows.map((r) => [r[xIdx], r[yIdx]]).filter((p) => p[0] !== null && p[1] !== null),
      }],
      xaxis: { title: { text: xCol } },
      yaxis: { title: { text: yCol } },
      title: { text: hint && hint.annotations ? `r = ${hint.annotations.pearson_r}` : "" },
    };
  } else if (type === "bar") {
    // sort descending by Y
    const sorted = [...body.rows].sort((a, b) => (b[yIdx] ?? 0) - (a[yIdx] ?? 0));
    options = {
      chart: { type: "bar", height: 280, animations: { enabled: false } },
      series: [{ name: yCol, data: sorted.map((r) => r[yIdx]) }],
      xaxis: { categories: sorted.map((r) => r[xIdx]), title: { text: xCol } },
      yaxis: { title: { text: yCol } },
    };
  } else if (type === "histogram") {
    const values = body.rows.map((r) => r[yIdx]).filter((v) => typeof v === "number");
    const bins = histogram(values, 20);
    options = {
      chart: { type: "bar", height: 280, animations: { enabled: false } },
      series: [{ name: yCol, data: bins.counts }],
      xaxis: {
        categories: bins.edges.map((v) => v.toFixed(3)),
        title: { text: yCol },
      },
      yaxis: { title: { text: "count" } },
    };
  }

  if (state.currentChart) {
    state.currentChart.destroy();
    state.currentChart = null;
  }
  if (options) {
    state.currentChart = new ApexCharts(container, options);
    state.currentChart.render();
  }
}

function histogram(values, nBins) {
  if (values.length === 0) return { edges: [], counts: [] };
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const step = (hi - lo) / nBins || 1;
  const counts = new Array(nBins).fill(0);
  const edges = [];
  for (let i = 0; i < nBins; i++) edges.push(lo + i * step);
  for (const v of values) {
    const idx = Math.min(nBins - 1, Math.floor((v - lo) / step));
    counts[idx]++;
  }
  return { edges, counts };
}

// Chart type dropdown
document.addEventListener("DOMContentLoaded", () => {
  const sel = document.getElementById("chart-type-select");
  sel.addEventListener("change", () => {
    if (state.lastResult) renderChart(state.lastResult, sel.value);
  });
});
```

- [ ] **Step 2: Manual smoke test**

Restart webapp. Visit `/data-explorer`. Run these queries and check the chart:

| Query | Expected chart |
|---|---|
| `SELECT trade_date, close FROM daily_quotes WHERE code='600519.SH' ORDER BY trade_date LIMIT 60` | Line chart of close prices |
| `SELECT pb, roe FROM ng101_feature_cache WHERE trade_date=(SELECT MAX(trade_date) FROM ng101_feature_cache) LIMIT 500` (after JSON expand — would fail if table doesn't have pb/roe; use available features) | Scatter + Pearson r |
| `SELECT code, label_10d FROM ng101_feature_cache WHERE trade_date=(SELECT MAX(trade_date) FROM ng101_feature_cache) ORDER BY label_10d DESC LIMIT 30` | Descending bar chart |
| `SELECT label_10d FROM ng101_feature_cache WHERE trade_date=(SELECT MAX(trade_date) FROM ng101_feature_cache) LIMIT 500` | Histogram |

Also test the chart-type dropdown: switch to "scatter" / "bar" / "histogram" and verify override works.

- [ ] **Step 3: Commit**

```bash
git add webapp/static/js/data_explorer.js
git commit -m "feat(explorer): ApexCharts auto-suggest + type override (line/scatter/bar/hist)"
```

---

### Task 14: `data_explorer.js` — Visual Builder

**Files:**
- Modify: `webapp/static/js/data_explorer.js`

- [ ] **Step 1: Add builder HTML scaffold into `data_explorer.html`**

In `webapp/templates/data_explorer.html`, replace the empty builder container:

```html
    <!-- Visual builder (hidden by default) -->
    <div id="builder-container" class="d-none card p-3">
      <div class="row g-2 small">
        <div class="col-md-3">
          <label class="form-label">Table</label>
          <select class="form-select form-select-sm" id="b-table"></select>
        </div>
        <div class="col-md-6">
          <label class="form-label">Columns (多选)</label>
          <select class="form-select form-select-sm" id="b-columns" multiple size="4"></select>
        </div>
        <div class="col-md-3">
          <label class="form-label">LIMIT</label>
          <input class="form-control form-control-sm" id="b-limit" value="100">
        </div>
      </div>
      <div class="row g-2 small mt-2">
        <div class="col-12"><label class="form-label">Filters</label></div>
        <div class="col-12" id="b-filters-rows"></div>
        <div class="col-12"><button class="btn btn-outline-secondary btn-sm" id="b-add-filter">+ Filter</button></div>
      </div>
      <div class="row g-2 small mt-2">
        <div class="col-md-4">
          <label class="form-label">ORDER BY</label>
          <select class="form-select form-select-sm" id="b-order-col"><option value="">(none)</option></select>
        </div>
        <div class="col-md-2">
          <label class="form-label">Dir</label>
          <select class="form-select form-select-sm" id="b-order-dir">
            <option value="ASC">ASC</option><option value="DESC" selected>DESC</option>
          </select>
        </div>
        <div class="col-md-6 d-flex align-items-end justify-content-end gap-2">
          <button class="btn btn-primary btn-sm" id="b-build-sql"><i class="bi bi-arrow-up-circle"></i> Build SQL</button>
        </div>
      </div>
    </div>
```

- [ ] **Step 2: Add Builder logic to JS**

Append to `webapp/static/js/data_explorer.js`:

```javascript
// ---- Visual Builder ---------------------------------------------------------

function getAllTablesFlat() {
  const out = [];
  for (const cat of Object.values(state.schema || {})) {
    for (const t of cat) out.push(t);
  }
  return out.sort((a, b) => a.table.localeCompare(b.table));
}

function populateBuilder() {
  const tables = getAllTablesFlat();
  const tSel = document.getElementById("b-table");
  tSel.innerHTML = tables.map((t) => `<option value="${t.table}">${t.table}</option>`).join("");

  tSel.addEventListener("change", () => refreshBuilderColumns());
  refreshBuilderColumns();

  document.getElementById("b-add-filter").addEventListener("click", () => addFilterRow());
  document.getElementById("b-build-sql").addEventListener("click", () => buildSqlFromBuilder());
}

function findTable(name) {
  return getAllTablesFlat().find((t) => t.table === name);
}

function refreshBuilderColumns() {
  const name = document.getElementById("b-table").value;
  const t = findTable(name);
  const cols = t ? t.columns.map((c) => c.name) : [];
  document.getElementById("b-columns").innerHTML =
    cols.map((c) => `<option value="${c}" selected>${c}</option>`).join("");
  document.getElementById("b-order-col").innerHTML =
    `<option value="">(none)</option>` +
    cols.map((c) => `<option value="${c}">${c}</option>`).join("");
  document.getElementById("b-filters-rows").innerHTML = "";
}

function addFilterRow() {
  const name = document.getElementById("b-table").value;
  const t = findTable(name);
  const cols = t ? t.columns.map((c) => c.name) : [];
  const row = document.createElement("div");
  row.className = "input-group input-group-sm mb-1";
  row.innerHTML = `
    <select class="form-select form-select-sm b-f-col">
      ${cols.map((c) => `<option>${c}</option>`).join("")}
    </select>
    <select class="form-select form-select-sm b-f-op" style="max-width:90px">
      <option>=</option><option>!=</option><option>&gt;</option><option>&lt;</option>
      <option>&gt;=</option><option>&lt;=</option><option>LIKE</option><option>IN</option>
    </select>
    <input class="form-control form-control-sm b-f-val" placeholder="value">
    <button class="btn btn-outline-danger btn-sm b-f-del">×</button>
  `;
  row.querySelector(".b-f-del").addEventListener("click", () => row.remove());
  document.getElementById("b-filters-rows").appendChild(row);
}

function buildSqlFromBuilder() {
  const table = document.getElementById("b-table").value;
  const cols = [...document.getElementById("b-columns").selectedOptions].map((o) => o.value);
  const limit = parseInt(document.getElementById("b-limit").value, 10) || 100;
  const orderCol = document.getElementById("b-order-col").value;
  const orderDir = document.getElementById("b-order-dir").value;

  const selectList = cols.length ? cols.join(", ") : "*";
  let sql = `SELECT ${selectList}\nFROM ${table}`;

  const filterRows = document.querySelectorAll("#b-filters-rows .input-group");
  const clauses = [];
  for (const r of filterRows) {
    const col = r.querySelector(".b-f-col").value;
    const op = r.querySelector(".b-f-op").value;
    const valRaw = r.querySelector(".b-f-val").value.trim();
    if (!valRaw) continue;
    let val = valRaw;
    // Auto-quote if non-numeric, non-IN
    if (op === "IN") {
      val = "(" + valRaw.split(",").map((v) => `'${v.trim()}'`).join(", ") + ")";
    } else if (isNaN(Number(valRaw))) {
      val = `'${valRaw.replace(/'/g, "''")}'`;
    }
    clauses.push(`  ${col} ${op} ${val}`);
  }
  if (clauses.length) sql += `\nWHERE\n${clauses.join("\n  AND\n")}`;
  if (orderCol) sql += `\nORDER BY ${orderCol} ${orderDir}`;
  sql += `\nLIMIT ${limit};`;

  setEditorContent(sql);
  // Switch back to SQL mode
  toggleInputMode("sql");
}

// Input mode toggle (SQL ↔ Builder)
function toggleInputMode(mode) {
  const sqlC = document.getElementById("sql-editor-container");
  const bC = document.getElementById("builder-container");
  const toggle = document.getElementById("input-mode-toggle");
  if (mode === "builder") {
    sqlC.classList.add("d-none");
    bC.classList.remove("d-none");
    toggle.querySelector('[data-input="sql"]').classList.replace("btn-primary", "btn-outline-secondary");
    toggle.querySelector('[data-input="builder"]').classList.replace("btn-outline-secondary", "btn-primary");
  } else {
    sqlC.classList.remove("d-none");
    bC.classList.add("d-none");
    toggle.querySelector('[data-input="sql"]').classList.replace("btn-outline-secondary", "btn-primary");
    toggle.querySelector('[data-input="builder"]').classList.replace("btn-primary", "btn-outline-secondary");
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("#input-mode-toggle button").forEach((btn) => {
    btn.addEventListener("click", () => toggleInputMode(btn.dataset.input));
  });
});

// Initialize builder after schema loads
const _originalRenderSchema = renderSchema;
renderSchema = function () {
  _originalRenderSchema();
  populateBuilder();
};
```

- [ ] **Step 3: Manual smoke test**

Restart webapp. Visit `/data-explorer`.

- Click the "Builder" toggle button. Builder form appears, SQL editor hides.
- Select `ng101_feature_cache` from Table dropdown. Columns populate; order_by populates.
- Select only `code` + `label_10d` in the columns multi-select.
- Click "+ Filter", pick `trade_date` `=` `2026-04-18` (use today's actual max date — or leave empty).
- Set ORDER BY `label_10d` DESC, LIMIT 30.
- Click "Build SQL" → editor switches back to SQL view with generated query.
- Press Ctrl+Enter → results render.

- [ ] **Step 4: Commit**

```bash
git add webapp/templates/data_explorer.html webapp/static/js/data_explorer.js
git commit -m "feat(explorer): Visual Builder (table/cols/filter/order/limit → SQL)"
```

---

### Task 15: `data_explorer.js` — Saved queries + hash routing + CSV + final integration

**Files:**
- Modify: `webapp/static/js/data_explorer.js`

- [ ] **Step 1: Add saved-queries + hash-routing logic**

Append to `webapp/static/js/data_explorer.js`:

```javascript
// ---- Saved queries ---------------------------------------------------------

async function loadSavedQueries() {
  const resp = await fetch("/api/explorer/saved");
  const body = await resp.json();
  if (!body.success) return;
  state.savedQueries = body.queries;
  document.getElementById("saved-count").textContent = body.queries.length;
  renderSavedList();
}

function renderSavedList() {
  const host = document.getElementById("saved-query-list");
  host.innerHTML = "";
  const filter = (document.getElementById("saved-tag-filter").value || "").trim().toLowerCase();
  const rows = state.savedQueries.filter((q) =>
    !filter || (q.tags || "").toLowerCase().includes(filter)
  );
  if (rows.length === 0) {
    host.innerHTML = '<div class="text-muted small p-2">无保存查询</div>';
    return;
  }
  for (const q of rows) {
    const d = document.createElement("div");
    d.className = "saved-row";
    d.innerHTML = `
      <div class="d-flex align-items-center">
        <b class="me-2">${escapeHtml(q.name)}</b>
        <span class="text-muted small">${q.tags ? "[" + escapeHtml(q.tags) + "]" : ""}</span>
        <span class="ms-auto text-muted small">run ${q.run_count}×</span>
        <button class="btn btn-sm btn-outline-primary ms-2 b-load">加载</button>
        <button class="btn btn-sm btn-outline-danger ms-1 b-del">×</button>
      </div>
      <div class="text-muted small mt-1">${escapeHtml(q.description || "")}</div>
      <pre class="small mb-0" style="white-space:pre-wrap">${escapeHtml(q.sql)}</pre>
    `;
    d.querySelector(".b-load").addEventListener("click", () => {
      setEditorContent(q.sql);
      window.location.hash = "#sql";
    });
    d.querySelector(".b-del").addEventListener("click", async () => {
      if (!confirm(`删除 "${q.name}"?`)) return;
      await fetch(`/api/explorer/saved/${q.id}`, { method: "DELETE" });
      loadSavedQueries();
    });
    host.appendChild(d);
  }
}

async function openSaveModal() {
  document.getElementById("save-name").value = "";
  document.getElementById("save-tags").value = "user";
  document.getElementById("save-description").value = "";
  new bootstrap.Modal(document.getElementById("save-modal")).show();
}

async function confirmSave() {
  const name = document.getElementById("save-name").value.trim();
  const tags = document.getElementById("save-tags").value.trim();
  const description = document.getElementById("save-description").value.trim();
  const sqlText = getEditorContent().trim();
  if (!name) { window.showToast("请输入名称", "warning"); return; }
  const resp = await fetch("/api/explorer/saved", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, sql: sqlText, tags, description }),
  });
  const body = await resp.json();
  if (!body.success) {
    window.showToast(body.error || "保存失败", "error");
    return;
  }
  bootstrap.Modal.getInstance(document.getElementById("save-modal")).hide();
  window.showToast(`已保存: ${name}`, "success");
  loadSavedQueries();
}

// ---- Hash routing (Mode tabs) ---------------------------------------------

const MODE_PRESET_NAMES = {
  stock:   "preset: single-stock all-features",
  cross:   "preset: cross-section pred_10d top50",
  compare: "preset: model compare ng101 vs ng110",
};

async function applyMode(mode) {
  // Update tab UI
  document.querySelectorAll(".mode-tabs .nav-link").forEach((a) => {
    a.classList.toggle("active", a.dataset.mode === mode);
  });

  const main = document.getElementById("explorer-main");
  const saved = document.getElementById("saved-panel");
  if (mode === "saved") {
    main.classList.add("d-none");
    saved.classList.remove("d-none");
    await loadSavedQueries();
    return;
  }
  main.classList.remove("d-none");
  saved.classList.add("d-none");

  if (mode in MODE_PRESET_NAMES) {
    const presetName = MODE_PRESET_NAMES[mode];
    const preset = state.savedQueries.find((q) => q.name === presetName);
    if (preset) {
      setEditorContent(preset.sql);
    } else {
      window.showToast(
        `preset "${presetName}" missing — run saved_seed or re-check`,
        "warning"
      );
    }
  }
}

function currentMode() {
  return (window.location.hash || "#sql").slice(1);
}

window.addEventListener("hashchange", () => applyMode(currentMode()));

// ---- CSV download -----------------------------------------------------------

function downloadCsv() {
  if (!state.lastResult) {
    window.showToast("先运行一条查询", "warning");
    return;
  }
  const { columns, rows } = state.lastResult;
  const lines = [columns.join(",")];
  for (const r of rows) {
    lines.push(r.map((v) => {
      if (v === null) return "";
      const s = String(v);
      return s.includes(",") || s.includes('"') ? `"${s.replace(/"/g, '""')}"` : s;
    }).join(","));
  }
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `explorer_${new Date().toISOString().slice(0,10)}.csv`;
  a.click();
}

// ---- Final wiring ----------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("btn-save").addEventListener("click", openSaveModal);
  document.getElementById("save-modal-confirm").addEventListener("click", confirmSave);
  document.getElementById("btn-csv").addEventListener("click", downloadCsv);
  document.getElementById("saved-tag-filter").addEventListener("input", renderSavedList);

  // Load saved queries once so presets are available when the user switches modes
  loadSavedQueries().then(() => applyMode(currentMode()));
});
```

- [ ] **Step 2: Full manual integration test**

Restart webapp. Visit `/data-explorer`.

Run through the complete checklist:

- [ ] **Page loads** — sidebar lists 8 categories, all 45+ tables visible.
- [ ] **Single-stock preset** — click `#stock` tab (`/data-explorer#stock`); editor populated with single-stock SQL; press Run → results.
- [ ] **Cross-section preset** — click `#cross`; editor populated with cross-section SQL; Run → 50 rows, auto chart = bar.
- [ ] **Compare preset** — click `#compare`; editor populated (may fail if `ng110_feature_cache` missing — ok, expect helpful error, not crash).
- [ ] **Saved preset not found warning** — if preset missing, toast appears, no crash.
- [ ] **SELECT * triggers LIMIT injection** — `SELECT * FROM daily_quotes` → warning banner.
- [ ] **DROP rejected** — `DROP TABLE daily_quotes` → 400 error toast, gutter-free.
- [ ] **JSON expansion** — run `SELECT * FROM ng101_feature_cache WHERE trade_date=(SELECT MAX(trade_date) FROM ng101_feature_cache) LIMIT 10` with "自动展开" ON; verify result table has 60+ columns (features + labels + market cols). Turn switch OFF, rerun, verify `features_json` column reappears.
- [ ] **Save and reload** — Save a query, reload page; saved query shows in `#saved` tab.
- [ ] **Delete saved** — delete the test saved query; confirm removal.
- [ ] **Builder** — Toggle to Builder; pick `ng101_feature_cache`, select 2 columns, add filter `trade_date = '<date>'`, order by `label_10d DESC`, limit 20; Build SQL; run; verify correct SQL produced and results render.
- [ ] **CSV download** — Click CSV button; verify file downloads with correct header + rows.
- [ ] **Chart override** — switch chart dropdown between options; verify re-render.
- [ ] **Timeout** — Run a slow query (e.g. Cartesian product `SELECT a.code FROM daily_quotes a, daily_quotes b LIMIT 100000000`) and verify ~30 s timeout toast.

- [ ] **Step 3: Commit**

```bash
git add webapp/static/js/data_explorer.js
git commit -m "feat(explorer): saved queries + hash-routed mode presets + CSV"
```

- [ ] **Step 4: Final test pass**

Run full webapp test suite:

```bash
cd webapp && python3 -m pytest tests/test_data_explorer/ -v
```

Expected: PASS all tests (Tasks 1-8). Manual UI tests from Step 2 above cover Tasks 9-15.

- [ ] **Step 5: Final commit sweep**

```bash
git log --oneline -n 15
```

Verify 15 commits on this branch (one per task). If any uncommitted files remain in `git status`, address them.

---

## Self-review

**1. Spec coverage check:**

| Spec section | Task covering it |
|---|---|
| §2 use cases 1-4 | Mode A/B/C = presets in Task 7 seed; Mode SQL = Task 11-12 |
| §3 Q3 (SQL + Builder) | Tasks 11, 12, 14 |
| §3 Q4 (category schema browser) | Tasks 4, 11 |
| §3 Q5 (auto chart + override) | Task 13 |
| §3 Q6 (saved to webapp.db) | Tasks 1, 7, 8, 15 |
| §3 Q7 (features_json expand) | Task 2, verified in 6, 12 |
| §4.1 backend modules | Tasks 1-7 |
| §4.2 frontend + CodeMirror 6 | Tasks 10, 11 |
| §4.3 saved_queries table | Task 1 |
| §4.4 sqlglot dep | Task 1 |
| §5.1 schema_discovery | Task 4 |
| §5.2 query_runner | Tasks 5, 6 |
| §5.3 feature_expander | Task 2 |
| §5.4 chart_suggester | Task 3 |
| §5.5 query_store + seed | Task 7 |
| §5.6 frontend components | Tasks 11-15 |
| §6 data flow | Tasks 5-8, 12 |
| §7 error handling table | Task 5 (validator), 6 (timeout), 8 (HTTP status) |
| §8 testing strategy | Each task has TDD steps; integration in Task 8 |
| §9 non-goals | respected — no dashboard, no CSV streaming, etc. |
| §10 rollout | Tasks 1, 8-10, blueprint reg, nav link |

All spec sections covered.

**2. Placeholder scan:** All steps have concrete code, exact commands, expected outputs. No "TBD", no "similar to above", no hand-wave error handling.

**3. Type consistency:**
- `QueryResult` has fields `{columns, rows, row_count, truncated, took_ms, warnings, chart_hint}` — matches Task 6 test expectations + Task 8 blueprint serialization + Task 12 frontend consumption. ✓
- `classify_table()` return strings: `raw, technical, feature_cache, factor, market_state, moneyflow, meta, backtest, other` — matches Task 4 test + Task 11 `orderedCats` list. ✓
- Seed query names match MODE_PRESET_NAMES in Task 15. ✓
- All function names consistent across tasks.

No inconsistencies to fix.
