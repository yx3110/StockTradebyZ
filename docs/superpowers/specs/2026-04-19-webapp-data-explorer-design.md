# Webapp Data Explorer — Design Spec

**Date**: 2026-04-19
**Status**: Design approved, ready for implementation plan
**Author**: Claude (via brainstorming skill)
**Approver**: yangxu

---

## 1. Goal

Add a unified data exploration page to the StockTradebyZ webapp that lets the
user query and visualize **all** local data — raw market data, fundamentals,
technical indicators, every model feature cache, factor caches, market-state
tables, money flow, labels, and backtest results — from one place.

Primary pain point today: each data source lives in its own table (45+ tables),
and there is no generic query / plot UI. The existing `/data-management` page
is completeness-focused; `/stock/<code>` only shows K-line + a fixed set of
technicals for one stock.

## 2. User & Scope

- **User**: yangxu, solo researcher, running the webapp on localhost.
- **Access**: local only (no auth); the SQL editor is gated to read-only by
  server-side enforcement.
- **Use cases, in frequency order**:
  1. **SQL self-service query** (most frequent) — ad-hoc `SELECT` across any
     table, sort/filter, visualize.
  2. **Single-stock deep-dive** — "show me every column and every model's
     features+predictions for 600519 on 2026-04-18".
  3. **Cross-section** — "distribution of `pred_10d` across the whole market
     today; top/bottom 50".
  4. **Model side-by-side** — "ng101 vs ng110 vs v39 predictions for the same
     universe/day".

  Use case 1 is the core UI. Use cases 2–4 are delivered as pre-built SQL
  **snippet presets** that populate the same editor — zero new specialized
  components in v1.

## 3. Decisions locked during brainstorming

| # | Decision |
|---|----------|
| Q1 | Scope = all four modes (E), but SQL is primary; 2–4 ship as snippets |
| Q2 | SQL self-service is the highest-frequency mode |
| Q3 | Input methods = SQL editor **and** Visual Query Builder (both present) |
| Q4 | Schema browser organized by smart category (raw/feature_cache/factor/…) |
| Q5 | Chart = smart-suggested default + manual override dropdown |
| Q6 | Saved queries persisted to `webapp.db` (naming + tags) — no dashboard in v1 |
| Q7 | `features_json` auto-expanded to columns at query time via Pandas |
| Approach | **A: In-house Flask blueprint**, native to existing webapp |

Hard-coded defaults (no UI switch, enforced server-side):

- SELECT-only (non-SELECT → 400)
- Auto-inject `LIMIT 10000` if user omits a LIMIT
- Query timeout: 30 s
- Result payload capped at 10 000 rows regardless of user LIMIT

## 4. Architecture

### 4.1 Backend modules

```
webapp/
├── api/
│   └── data_explorer.py          # new Flask blueprint, url_prefix='/api/explorer'
├── core/
│   └── data_explorer/            # new package
│       ├── __init__.py
│       ├── schema_discovery.py   # sqlite_master scan + category classifier
│       ├── query_runner.py       # readonly conn + sqlglot SELECT-only + LIMIT + timeout
│       ├── feature_expander.py   # pd.json_normalize for features_json
│       ├── chart_suggester.py    # heuristic chart-type picker
│       └── query_store.py        # saved_queries CRUD on webapp.db
└── templates/
    └── data_explorer.html        # single page, jQuery + Bootstrap + CodeMirror 6
```

### 4.2 Frontend

One template, one page, five tabs driven by URL hash:

- `/data-explorer`  (default → `#sql`)
- `/data-explorer#sql`      — SQL Explorer
- `/data-explorer#stock`    — Single-stock deep-dive (snippet preset)
- `/data-explorer#cross`    — Cross-section (snippet preset)
- `/data-explorer#compare`  — Model compare (snippet preset)
- `/data-explorer#saved`    — Saved queries list

JS libraries:

- **Already loaded** (in `base.html`): jQuery 3.7, Bootstrap 5.3, DataTables
  1.13, ApexCharts.
- **New**: CodeMirror 6 (CDN, ~150 KB bundle incl. sql-lang extension).
  Loaded only in `data_explorer.html` via `{% block extra_js %}`, not in
  `base.html`, so other pages stay unchanged.

No new frontend framework (no Vue, Alpine, React). State lives in a module-
scoped closure inside `data_explorer.js`.

### 4.3 Database additions (`webapp.db`)

```sql
CREATE TABLE IF NOT EXISTS saved_queries (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL UNIQUE,
    sql          TEXT    NOT NULL,
    tags         TEXT,             -- comma-separated
    description  TEXT,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_run_at  TIMESTAMP,
    run_count    INTEGER DEFAULT 0
);
CREATE INDEX idx_saved_queries_tags ON saved_queries(tags);
```

Seeded with snippets for the three preset modes at first run.

### 4.4 New dependency

- `sqlglot` — added to `webapp/requirements.txt`. Used to parse SQL and
  verify it is a single `SELECT` statement before execution.

## 5. Component detail

### 5.1 `schema_discovery.py`

- `discover() -> dict[category, list[TableInfo]]`
- `TableInfo = {table, columns: [(name, type)], row_count, date_range, has_features_json, active?}`
- Classification rules (by table name):
  - `raw`: `daily_quotes`, `daily_basic`, `financial_indicator`,
    `financial_indicator_backup`, `financial_statement_raw`, `index_daily`,
    `market_indices`, `sw_index_daily`
  - `technical`: `technical_indicators`, `technical_overview`
  - `feature_cache`: `*_feature_cache`, `neural_embedding_cache`,
    `alpha158_feature_cache`, `brain_alpha_cache`, `latest_worldquant_factors`,
    `worldquant_factors`, `active_mv_feature_cache`
  - `factor`: `*_factor_cache`, `factor_daily_returns`
  - `market_state`: `market_amv`, `signal_trust_scores`, `signal_trust_scores_history`,
    `signal_trust_samples`
  - `moneyflow`: `moneyflow_daily`, `hsgt_daily`
  - `meta`: `securities`, `stock_basic_info`, `sw_industry`, `schema_version`,
    `data_update_log`, `latest_quotes`
  - `backtest`: `backtest_trades`, `backtest_results`, `stock_signals`
  - Unmatched → `other`
- `active?` flag: true for the currently-production model cache
  (read `ml_models/ng/ng_schema.py:PRODUCTION_VERSION` and map to table).
- Cached in-process for 5 min; invalidated on `?refresh=1` query param.

### 5.2 `query_runner.py`

Signature:

```python
def run_query(sql: str,
              expand_features: bool = True,
              max_rows: int = 10_000,
              timeout_s: int = 30) -> QueryResult
```

Steps:

1. **Parse** with `sqlglot.parse_one(sql, read='sqlite')`. Must be exactly one
   `exp.Select` root. Else raise `InvalidQueryError` → API 400.
2. **Inject LIMIT** if root has no `limit` (sqlglot AST); cap at `max_rows`
   even if user-provided limit is larger.
3. **Open read-only connection**: `sqlite3.connect('file:' + db_path + '?mode=ro', uri=True)`.
4. **Install progress callback** to enforce timeout: `conn.set_progress_handler(cb, 10_000)`
   where `cb` raises after elapsed > `timeout_s`.
5. **Execute** via `pd.read_sql_query(sql, conn)`.
6. **Feature expand** (if `expand_features` and `features_json` column present)
   — delegate to `feature_expander.expand(df)`.
7. **Post-run safety**: truncate to `max_rows` rows, add `warning` if truncated.
8. Return:

```python
@dataclass
class QueryResult:
    columns:      list[str]           # post-expand
    rows:         list[list]          # JSON-serializable, NaN → None
    row_count:    int                 # rows actually returned
    truncated:    bool                # True iff row_count == max_rows (caller-observed cap)
    took_ms:      int
    warnings:     list[str]           # ["LIMIT injected", "rows truncated", ...]
    chart_hint:   dict | None         # from chart_suggester
```

We deliberately do not compute a pre-LIMIT count (would cost a second pass
over the same data). The UI shows "10 000+ rows (truncated)" whenever
`truncated=True`.

API: `POST /api/explorer/query` body `{sql, expand_features, max_rows?}`.

### 5.3 `feature_expander.py`

```python
def expand(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if 'features_json' not in df.columns:
        return df, []
    try:
        normalized = pd.json_normalize(df['features_json'].apply(json.loads))
    except Exception as e:
        return df, [f"features_json expansion failed: {e}"]
    return (
        pd.concat([df.drop(columns=['features_json']).reset_index(drop=True),
                   normalized.reset_index(drop=True)], axis=1),
        []
    )
```

Idempotent; safe when column absent.

### 5.4 `chart_suggester.py`

Heuristic purely on DataFrame dtypes + column names. Rules are evaluated in
order; first match wins:

| Rule | Precondition | Suggestion |
|------|-------------|------------|
| R1 | `trade_date` present + ≥1 numeric + single `code` (or no `code`) | `line` (X=`trade_date`, Y=first numeric) |
| R2 | `trade_date` + `code` (multiple codes) + 1 numeric | `null` — ambiguous in v1; table only |
| R3 | exactly 2 numerics, no obvious key | `scatter` + Pearson r in title |
| R4 | 1 categorical (`code`/`industry`/`name`/`tag`) + 1 numeric + rows ≤ 50 | `bar` descending |
| R5 | 1 categorical + 1 numeric + rows > 50 | `histogram` of the numeric |
| R6 | exactly 1 numeric, no date, no category | `histogram` (20 bins) + mean/median overlay |
| else |  | `null` → table only |

Returns `{type, x, y, annotations?}` or `None`.

### 5.5 `query_store.py`

Thin CRUD on `saved_queries`:

- `list(tag=None) -> list[SavedQuery]`
- `get(id) -> SavedQuery`
- `create(name, sql, tags, description) -> SavedQuery`
- `update(id, **fields) -> SavedQuery`
- `delete(id) -> None`
- `touch(id)` — `run_count += 1`, `last_run_at = now`

On app startup, if `saved_queries` has 0 rows, insert the following seed
queries. Subsequent restarts skip this block (idempotent).

```python
SEED_QUERIES = [
  # snippet powering Mode A (single-stock deep-dive) — parameterize 600519/date in UI
  ("preset: single-stock all-features",
   """SELECT * FROM ng101_feature_cache
      WHERE code = '600519.SH'
        AND trade_date >= date('now', '-60 days')
      ORDER BY trade_date DESC""",
   "preset,stock"),

  # snippet powering Mode B (cross-section)
  ("preset: cross-section pred_10d top50",
   """SELECT code, trade_date, label_10d, features_json
      FROM ng101_feature_cache
      WHERE trade_date = (SELECT MAX(trade_date) FROM ng101_feature_cache)
      ORDER BY label_10d DESC
      LIMIT 50""",
   "preset,cross"),

  # snippet powering Mode C (model compare)
  ("preset: model compare ng101 vs ng110",
   """SELECT a.code, a.trade_date,
             a.label_10d AS ng101_label10, b.label_10d AS ng110_label10
      FROM ng101_feature_cache a
      JOIN ng110_feature_cache b
        ON a.code = b.code AND a.trade_date = b.trade_date
      WHERE a.trade_date = (SELECT MAX(trade_date) FROM ng101_feature_cache)
      ORDER BY a.label_10d DESC
      LIMIT 100""",
   "preset,compare"),

  # useful general-purpose snippet
  ("preset: signal_trust 🔴 tagged stocks",
   """SELECT code, as_of_date, n_samples, direction_hit_rate, trust_tag
      FROM signal_trust_scores
      WHERE trust_tag = '🔴'
      ORDER BY n_samples DESC
      LIMIT 50""",
   "preset"),
]
```

`ng110_feature_cache` may not exist in all deployments — the seed insert is
wrapped in a per-row `try/except`; a failed insert logs a warning and skips
that seed.

### 5.6 Frontend components (in `static/js/data_explorer.js`)

- `initSchemaBrowser()` — fetches `/api/explorer/schema`, renders collapsible
  tree. Clicking a table inserts `SELECT * FROM <t> WHERE trade_date = (SELECT MAX(trade_date) FROM <t>) LIMIT 100`.
- `initEditor()` — CodeMirror 6 with SQL mode. `Ctrl+Enter` → `runQuery()`.
- `initBuilder()` — Bootstrap form: table select, column multi-select,
  filter rows (column / op / value), order by, limit. `Build SQL` button
  updates the editor.
- `runQuery()` — POST `/api/explorer/query`; on success, render into
  DataTables + ApexCharts; on error, toast + inline annotation in editor
  gutter.
- `renderChart(hint, overrideType?)` — reads the returned `chart_hint`; user
  can override via dropdown (line / scatter / bar / histogram / none).
- `initSaved()` — lists from `/api/explorer/saved`; supports filter by tag;
  click a row → load into editor.
- Hash routing: on `hashchange`, swap the preset SQL and run.

## 6. Data flow (SQL query lifecycle)

```
Browser                       Flask                      SQLite
   │                             │                          │
   │ POST /api/explorer/query    │                          │
   │   {sql, expand_features}    │                          │
   ├────────────────────────────►│                          │
   │                             │ sqlglot.parse            │
   │                             │ validate SELECT-only     │
   │                             │ inject LIMIT if missing  │
   │                             │                          │
   │                             │ open ro conn, pd.read_sql│
   │                             ├─────────────────────────►│
   │                             │                          │
   │                             │◄──── DataFrame ──────────┤
   │                             │                          │
   │                             │ feature_expander.expand  │
   │                             │ truncate to max_rows     │
   │                             │ chart_suggester.suggest  │
   │                             │                          │
   │◄──────── JSON ──────────────┤                          │
   │   {columns, rows,           │                          │
   │    total_rows, took_ms,     │                          │
   │    warnings, chart_hint}    │                          │
   │                             │                          │
   │ DataTables.render           │                          │
   │ ApexCharts.render(chart_hint)                          │
```

## 7. Error handling & safety

| Situation | Response |
|-----------|----------|
| Non-SELECT SQL (INSERT/UPDATE/…/PRAGMA/ATTACH) | 400 `invalid_query`, toast + gutter marker |
| Multiple statements (`;`-separated) | 400 `invalid_query` |
| Syntax error (sqlglot raises) | 400 with parse error message |
| Query timeout (>30 s, progress handler aborts) | 504 `timeout`, toast |
| SQLite runtime error (no such column, etc.) | 400 with DB error message |
| Empty result | 200 with `row_count=0`, table shows "No rows"; no chart |
| `features_json` JSON parse failure | 200 with warning; un-expanded column preserved |
| Row count == max_rows (10 000) | 200 with `truncated=True`; client banner "results truncated, add LIMIT/WHERE to narrow" |
| Saved query name collision | 409 `conflict`, toast "name already exists" |
| Concurrent query (user clicks Run twice) | Latest wins; UI disables button during in-flight |
| `sqlite3.OperationalError: database is locked` | Retry once after 500 ms (the main app is a writer) |

All API errors come back as `{success: false, error: str, code: str}` —
consistent with other blueprints in the project.

## 8. Testing strategy

Unit tests in `webapp/tests/test_data_explorer/`:

- `test_schema_discovery.py` — classifies known tables correctly, handles
  unknown table name (→ `other`), handles missing `trade_date` column.
- `test_query_runner.py` — SELECT passes; INSERT / DROP / PRAGMA rejected;
  missing LIMIT → injected; explicit LIMIT > max_rows → capped; 30s timeout
  triggered via `time.sleep` in progress handler stub.
- `test_feature_expander.py` — expands real ng101 sample; idempotent on
  absent column; graceful failure on malformed JSON.
- `test_chart_suggester.py` — 5 shapes produce expected hints; no-hint
  returns `None`.
- `test_query_store.py` — CRUD happy path; name unique; seed idempotent.

Integration test: a smoke test that starts the Flask app, hits
`GET /api/explorer/schema`, `POST /api/explorer/query` with a canonical query,
and asserts shape of the response (no DB content assertions — those would
rot).

Manual test checklist in plan:

- Load page; sidebar renders every table (45+) across the 8 defined categories
  plus `other` for any unclassified.
- Click `ng101_feature_cache`; editor populated; Run; table has 66+ feature
  columns after JSON expand; chart auto-renders a histogram or scatter per
  the shape.
- Paste `DROP TABLE daily_quotes;` → 400.
- Paste unbounded `SELECT * FROM daily_quotes` → returns 10 000 rows +
  "LIMIT injected" and `truncated=True` banner.
- Paste a 30+ second query (e.g. `WITH RECURSIVE cnt AS (...)` loop) →
  timeout toast at 30 s, server log shows progress-handler abort.
- Save a query; reload page; saved query appears in `#saved` tab.
- Switch to `#stock` hash; editor populated with the single-stock preset;
  change `600519.SH` to another code; Run; result reflects the new code.

## 9. Out of scope (v1) — explicit non-goals

- **Dashboard / pinned widgets / scheduled refresh** — saved queries are
  listed only, not rendered as cards.
- **Query sharing / URL permalinks** — in-browser only.
- **Independent UI for Modes A/B/C** — they are SQL snippet presets; a
  dedicated page can be added later if snippets prove insufficient.
- **Multi-statement / CTE-heavy query editor affordances** — plain SQL
  editor only.
- **Result CSV streaming beyond 10 000 rows** — truncated download only.
- **Schema discovery for external DBs** — stock_data.db only.
- **Authentication / multi-user** — single local user assumption.
- **Write operations from UI** — all data mutation stays in the existing
  backfill/training flows.

## 10. Rollout

Single PR merging:

1. New Python modules in `webapp/core/data_explorer/` + blueprint.
2. `webapp/templates/data_explorer.html` + `static/js/data_explorer.js` +
   `static/css/data_explorer.css`.
3. Nav link added to `base.html` between "数据管理" and the search box.
4. `sqlglot` added to `webapp/requirements.txt`.
5. Seed migration for `saved_queries` on first boot.
6. Tests.

No data migration risk (all new surface area, no existing tables touched).

## 11. Open questions / explicit deferrals

None blocking v1.

Deferred for v2 consideration:

- CSV upload → temp table query (if ad-hoc external datasets ever needed).
- "Facet pill" UI à la Datasette on the result table.
- Small-multiples / faceted chart rendering.
- `EXPLAIN QUERY PLAN` panel to show index usage.
