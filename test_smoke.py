"""Smoke test for subtask-4-1.

Verifies:
1. quick_daily_update.py contains NO to_csv() calls.
2. DatabaseManager can create a SQLite DB, insert synthetic kline data,
   and retrieve it correctly.
3. select_stock.py --db flag loads data from the SQLite DB without error.
"""
from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

# ──────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────

def _fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def _ok(msg: str) -> None:
    print(f"OK  : {msg}")


# ──────────────────────────────────────────────
# Test 1 – no to_csv() calls in quick_daily_update.py
# ──────────────────────────────────────────────

def test_no_csv_writes() -> None:
    """Parse the AST of quick_daily_update.py and assert to_csv is never called."""
    script = Path("quick_daily_update.py")
    if not script.exists():
        _fail("quick_daily_update.py not found")

    source = script.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(script))

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "to_csv"
        ):
            _fail(
                f"Found to_csv() call at line {node.lineno} in quick_daily_update.py – "
                "redundant CSV write not eliminated!"
            )

    _ok("quick_daily_update.py has no to_csv() calls")


# ──────────────────────────────────────────────
# Test 2 – DatabaseManager CRUD with synthetic data
# ──────────────────────────────────────────────

def _make_synthetic_df(n: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "open": [10.0 + i * 0.1 for i in range(n)],
            "close": [10.1 + i * 0.1 for i in range(n)],
            "high": [10.5 + i * 0.1 for i in range(n)],
            "low": [9.9 + i * 0.1 for i in range(n)],
            "volume": [1_000_000.0 + i * 10_000 for i in range(n)],
        }
    )


def test_db_manager_crud() -> None:
    """Create an in-memory SQLite DB via DatabaseManager, insert, and read back."""
    from db_manager import DatabaseManager

    # Use ':memory:' for a temporary in-memory database
    db = DatabaseManager(":memory:")

    code = "000001"
    df_in = _make_synthetic_df(10)

    # Insert via upsert_klines (the method called by quick_daily_update)
    db.upsert_klines(code, df_in)

    # Verify get_last_date returns the last date
    last = db.get_last_date(code)
    expected_last = df_in["date"].max().strftime("%Y-%m-%d")
    if last != expected_last:
        _fail(f"get_last_date returned '{last}', expected '{expected_last}'")
    _ok(f"get_last_date correct: {last}")

    # Verify load_stock_data returns full data
    df_out = db.load_stock_data(code)
    if len(df_out) != len(df_in):
        _fail(f"load_stock_data returned {len(df_out)} rows, expected {len(df_in)}")
    _ok(f"load_stock_data returned {len(df_out)} rows")

    # Verify load_all_stocks_data with explicit code list
    all_data = db.load_all_stocks_data([code])
    if code not in all_data:
        _fail(f"load_all_stocks_data missing code '{code}'")
    _ok(f"load_all_stocks_data contains code '{code}'")

    # Verify get_all_codes
    codes = db.get_all_codes()
    if code not in codes:
        _fail(f"get_all_codes missing '{code}'")
    _ok(f"get_all_codes: {codes}")

    db.close_all()
    _ok("DatabaseManager CRUD smoke test passed")


# ──────────────────────────────────────────────
# Test 3 – select_stock.py --db flag works
# ──────────────────────────────────────────────

def test_select_stock_db_flag() -> None:
    """Create a temp SQLite DB with two stocks, run select_stock.py --db, check exit 0."""
    from db_manager import DatabaseManager

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_smoke.db"

        # Populate DB with two synthetic stocks
        db = DatabaseManager(db_path)
        for code in ("000001", "600000"):
            db.upsert_klines(code, _make_synthetic_df(20))
        db.close_all()

        # Run select_stock.py with --db flag (no --config needed since it
        # defaults to ./configs.json which exists in the working directory)
        result = subprocess.run(
            [
                sys.executable,
                "select_stock.py",
                "--db",
                str(db_path),
                "--tickers",
                "000001,600000",
            ],
            capture_output=True,
            text=True,
        )

        # A non-zero exit is a failure
        if result.returncode != 0:
            _fail(
                f"select_stock.py --db exited with code {result.returncode}.\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )

        # Verify DB data was loaded (logged by load_data_from_db)
        combined = result.stdout + result.stderr
        if "从数据库加载了" not in combined:
            _fail(
                "select_stock.py --db did not log DB load message.\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )

        _ok("select_stock.py --db flag works correctly")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Smoke Test: subtask-4-1 verification")
    print("=" * 60)

    test_no_csv_writes()
    test_db_manager_crud()
    test_select_stock_db_flag()

    print()
    print("ALL SMOKE TESTS PASSED")
