import json
from pathlib import Path

import pytest

from signal_trust.report_appender import append_trust_tags
from signal_trust.db import connect
from signal_trust.constants import TAG_GREEN, TAG_NO_DATA


def _write_report_file(path: Path, stocks: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "analysis_date": "2026-04-12",
        "all_stocks_with_scores": stocks,
    }, ensure_ascii=False), encoding="utf-8")


def _seed_score(db_path: str, code: str, tag: str, n: int):
    conn = connect(db_path)
    try:
        conn.execute(
            "INSERT INTO signal_trust_scores "
            "(code, as_of_date, n_samples, direction_hit_rate, systematic_bias, "
            " high_pred_realize_rate, trust_tag) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (code, "2026-04-12", n, 0.6, -0.01, 0.5, tag),
        )
        conn.commit()
    finally:
        conn.close()


def test_appends_trust_fields(tmp_db, tmp_path):
    report = tmp_path / "report.json"
    _write_report_file(report, [
        {"stock_code": "A.SZ", "rank_score": 95, "pred_10d": 0.015},
        {"stock_code": "B.SZ", "rank_score": 90, "pred_10d": 0.012},
    ])
    _seed_score(tmp_db, "A.SZ", TAG_GREEN, 42)
    n = append_trust_tags(str(report), db_path=tmp_db, top_n=50)
    assert n == 2
    data = json.loads(report.read_text(encoding="utf-8"))
    a = next(s for s in data["all_stocks_with_scores"] if s["stock_code"] == "A.SZ")
    b = next(s for s in data["all_stocks_with_scores"] if s["stock_code"] == "B.SZ")
    assert a["trust_tag"] == TAG_GREEN
    assert a["trust_samples"] == 42
    assert a["trust_details"]["direction_hit_rate"] == 0.6
    assert b["trust_tag"] == TAG_NO_DATA
    assert b["trust_samples"] == 0


def test_only_top_n_tagged(tmp_db, tmp_path):
    report = tmp_path / "r.json"
    stocks = [{"stock_code": f"S{i}.SZ", "rank_score": 100 - i, "pred_10d": 0.015}
              for i in range(100)]
    _write_report_file(report, stocks)
    _seed_score(tmp_db, "S0.SZ", TAG_GREEN, 20)
    _seed_score(tmp_db, "S99.SZ", TAG_GREEN, 20)
    n = append_trust_tags(str(report), db_path=tmp_db, top_n=10)
    assert n == 10
    data = json.loads(report.read_text(encoding="utf-8"))
    # 前 10 应有 trust_tag, 后 90 没有
    tagged = [s for s in data["all_stocks_with_scores"] if "trust_tag" in s]
    assert len(tagged) == 10


def test_missing_scores_table_graceful(tmp_path):
    """scores 表不存在时不应抛. 为日报流程容错保命."""
    import sqlite3
    db_path = tmp_path / "empty.db"
    sqlite3.connect(str(db_path)).close()
    report = tmp_path / "r.json"
    _write_report_file(report, [{"stock_code": "A.SZ", "rank_score": 100, "pred_10d": 0.015}])
    n = append_trust_tags(str(report), db_path=str(db_path), top_n=50)
    # 返回 1 (尝试贴标签但 DB 没有数据, 落入 TAG_NO_DATA 分支)
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["all_stocks_with_scores"][0]["stock_code"] == "A.SZ"


def test_atomic_write_on_crash(tmp_db, tmp_path, monkeypatch):
    """写中途失败原文件保留."""
    report = tmp_path / "r.json"
    original_content = json.dumps({
        "analysis_date": "2026-04-12",
        "all_stocks_with_scores": [{"stock_code": "A.SZ", "rank_score": 100, "pred_10d": 0.015}],
    }, ensure_ascii=False)
    report.write_text(original_content, encoding="utf-8")
    _seed_score(tmp_db, "A.SZ", TAG_GREEN, 20)

    # patch Path.replace 抛异常
    def boom(self, target):
        raise OSError("disk full")
    monkeypatch.setattr(Path, "replace", boom)
    with pytest.raises(OSError):
        append_trust_tags(str(report), db_path=tmp_db, top_n=50)
    # 原文件应未变
    assert report.read_text(encoding="utf-8") == original_content
