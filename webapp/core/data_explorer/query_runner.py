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
