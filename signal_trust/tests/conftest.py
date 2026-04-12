"""共享 fixtures：临时 DB + mock 报告工厂。"""
import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from signal_trust.db import connect, migrate


@pytest.fixture
def tmp_db(tmp_path):
    """临时 DB 文件，含 securities/daily_quotes/daily_basic 三张基础表 + signal_trust 两张表。"""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""CREATE TABLE securities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        code TEXT UNIQUE NOT NULL,
        name TEXT,
        industry TEXT
    )""")
    conn.execute("""CREATE TABLE daily_quotes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        security_id INTEGER NOT NULL,
        trade_date TEXT NOT NULL,
        close REAL,
        amount REAL,
        UNIQUE(security_id, trade_date)
    )""")
    conn.execute("""CREATE TABLE daily_basic (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        security_id INTEGER NOT NULL,
        trade_date TEXT NOT NULL,
        circ_mv REAL,
        UNIQUE(security_id, trade_date)
    )""")
    conn.commit()
    conn.close()
    migrate(str(db_path))
    return str(db_path)


def _seed_stock(db_path: str, code: str, industry: str,
                quotes: list[tuple[str, float, float]],
                circ_mv_by_date: dict[str, float] | None = None) -> int:
    """写入一只股票和它的日线。quotes: [(date, close, amount), ...]"""
    conn = connect(db_path)
    cur = conn.execute("INSERT INTO securities(code, industry) VALUES (?, ?)", (code, industry))
    sid = cur.lastrowid
    for d, close, amount in quotes:
        conn.execute(
            "INSERT INTO daily_quotes(security_id, trade_date, close, amount) VALUES (?, ?, ?, ?)",
            (sid, d, close, amount),
        )
    if circ_mv_by_date:
        for d, mv in circ_mv_by_date.items():
            conn.execute(
                "INSERT INTO daily_basic(security_id, trade_date, circ_mv) VALUES (?, ?, ?)",
                (sid, d, mv),
            )
    conn.commit()
    conn.close()
    return sid


@pytest.fixture
def seed_stock(tmp_db):
    def _factory(code: str, industry: str = "计算机",
                 quotes: list[tuple[str, float, float]] = None,
                 circ_mv_by_date: dict[str, float] | None = None) -> int:
        return _seed_stock(tmp_db, code, industry, quotes or [], circ_mv_by_date)
    return _factory


def _write_report(report_dir: Path, date: str, stocks: list[dict], version: str = "ng101"):
    """写一份 mock analysis_data JSON。stocks: [{stock_code, pred_10d, ...}, ...]"""
    d = report_dir / f"daily_selection_{version}"
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis_date": date,
        "scoring_version": version,
        "all_stocks_with_scores": stocks,
    }
    (d / f"analysis_data_{date.replace('-', '')}.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


@pytest.fixture
def write_report(tmp_path):
    reports_root = tmp_path / "reports"
    reports_root.mkdir(exist_ok=True)
    def _factory(date: str, stocks: list[dict], version: str = "ng101"):
        _write_report(reports_root, date, stocks, version)
        return reports_root / f"daily_selection_{version}"
    return _factory
