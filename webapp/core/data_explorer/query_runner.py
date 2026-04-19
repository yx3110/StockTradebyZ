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
    """Return SQL with LIMIT <= max_rows guaranteed.

    - Missing LIMIT: append LIMIT max_rows.
    - Existing LIMIT > max_rows: replace with max_rows.
    - Existing LIMIT <= max_rows: leave as-is.
    """
    stmt = ensure_select_only(sql)
    existing = stmt.args.get("limit")
    if existing is None:
        stmt.limit(max_rows, copy=False)
    else:
        try:
            current = int(existing.expression.this)
        except (ValueError, AttributeError):
            current = max_rows + 1  # unparseable -> replace with cap
        if current > max_rows:
            stmt.limit(max_rows, copy=False)
    return stmt.sql(dialect="sqlite")


# --- Task 6: execution -------------------------------------------------------

import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from core.data_explorer.chart_suggester import suggest as suggest_chart
from core.data_explorer.feature_expander import expand as _expand_features


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
            else:
                try:
                    if pd.isna(v):
                        converted.append(None)
                        continue
                except (TypeError, ValueError):
                    pass
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
        df, exp_warnings = _expand_features(df)
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
