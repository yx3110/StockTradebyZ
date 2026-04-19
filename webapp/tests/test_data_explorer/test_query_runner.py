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
