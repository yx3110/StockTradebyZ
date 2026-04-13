import sqlite3

import pytest

from signal_trust.db import migrate, connect


def test_migrate_creates_both_tables(tmp_path):
    db_path = str(tmp_path / "new.db")
    migrate(db_path)
    conn = sqlite3.connect(db_path)
    try:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
    finally:
        conn.close()
    assert "signal_trust_samples" in tables
    assert "signal_trust_scores" in tables


def test_migrate_idempotent(tmp_path):
    db_path = str(tmp_path / "new.db")
    migrate(db_path)
    migrate(db_path)  # 再跑一次不应报错
    conn = connect(db_path)
    try:
        # 能正常查询
        conn.execute("SELECT COUNT(*) FROM signal_trust_samples").fetchone()
    finally:
        conn.close()


def test_samples_primary_key(tmp_path):
    db_path = str(tmp_path / "new.db")
    migrate(db_path)
    conn = connect(db_path)
    try:
        conn.execute(
            "INSERT INTO signal_trust_samples(code, trade_date, sample_end_date, pred_10d, version) "
            "VALUES (?, ?, ?, ?, ?)",
            ("000001.SZ", "2026-01-01", "2026-01-15", 0.02, "ng101"),
        )
        # 同 (code, trade_date) 第二次应失败
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO signal_trust_samples(code, trade_date, sample_end_date, pred_10d, version) "
                "VALUES (?, ?, ?, ?, ?)",
                ("000001.SZ", "2026-01-01", "2026-01-15", 0.03, "ng106"),
            )
    finally:
        conn.close()
