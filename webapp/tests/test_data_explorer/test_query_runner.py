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


# --- Task 6: run_query tests -------------------------------------------------

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


# --- Fix 2: Timeout test -------------------------------------------------------

def test_run_query_timeout(tmp_stock_db) -> None:
    """Verify the 30s timeout path actually fires. Use a 2s timeout + a
    recursive CTE that counts forever so the progress handler trips fast.

    We include a LIMIT so inject_limit returns the original SQL unchanged
    (Fix 5), avoiding sqlglot's CTE column-alias rewriting bug.
    """
    from core.data_explorer.query_runner import QueryTimeoutError
    # Recursive CTE using AS alias (sqlglot-safe) + huge LIMIT so inject_limit
    # keeps original text via Fix 5 early-return path
    slow_sql = (
        "WITH RECURSIVE cnt AS "
        "(SELECT 1 AS x UNION ALL SELECT x+1 FROM cnt) "
        "SELECT x FROM cnt LIMIT 1000000000"
    )
    with pytest.raises(QueryTimeoutError) as exc:
        run_query(tmp_stock_db, slow_sql, timeout_s=2, max_rows=1_000_000_000)
    assert "timeout" in str(exc.value).lower()


# --- Fix 4: UNION/INTERSECT/EXCEPT acceptance ---------------------------------

def test_union_accepted() -> None:
    ensure_select_only("SELECT 1 UNION SELECT 2")  # no raise


def test_union_gets_limit_injected() -> None:
    out = inject_limit("SELECT 1 UNION SELECT 2", max_rows=100)
    assert "LIMIT 100" in out.upper().replace("\n", " ")


# --- Fix 5: inject_limit preserves original text when limit already valid -----

def test_inject_limit_preserves_original_text_when_limit_valid() -> None:
    # sqlglot would normally rewrite json_extract → -> operator.
    # We keep the user's exact text because their LIMIT is already valid.
    original = "SELECT json_extract(features_json, '$.pb') FROM ng101_feature_cache LIMIT 50"
    out = inject_limit(original, max_rows=10000)
    assert out == original
