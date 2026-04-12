from signal_trust.global_stats import (
    aggregate_by_bucket, format_markdown_report,
)
from signal_trust.db import connect


def _seed(db_path: str, rows: list[dict]):
    conn = connect(db_path)
    try:
        for r in rows:
            conn.execute(
                "INSERT INTO signal_trust_samples "
                "(code, trade_date, sample_end_date, pred_10d, actual_10d, version, "
                " market_cap_bucket, industry, liquidity_bucket) "
                "VALUES (?, ?, ?, ?, ?, 'ng106', ?, ?, ?)",
                (r["code"], r["trade_date"], r["sample_end_date"],
                 r["pred_10d"], r["actual_10d"],
                 r.get("market_cap_bucket"), r.get("industry"), r.get("liquidity_bucket")),
            )
        conn.commit()
    finally:
        conn.close()


def test_aggregate_by_market_cap(tmp_db):
    rows = []
    # 微盘 5 条 方向命中率 0.4 (2/5)
    for i in range(5):
        rows.append({
            "code": f"M{i}.SZ", "trade_date": f"2025-01-{i+1:02d}",
            "sample_end_date": f"2025-01-{i+15:02d}",
            "pred_10d": 0.02, "actual_10d": 0.02 if i < 2 else -0.01,
            "market_cap_bucket": "微盘", "industry": "计算机", "liquidity_bucket": "低",
        })
    # 大盘 10 条 方向命中率 0.9 (9/10)
    for i in range(10):
        rows.append({
            "code": f"L{i}.SZ", "trade_date": f"2025-02-{i+1:02d}",
            "sample_end_date": f"2025-02-{i+15:02d}",
            "pred_10d": 0.02, "actual_10d": 0.03 if i < 9 else -0.01,
            "market_cap_bucket": "大盘", "industry": "银行", "liquidity_bucket": "高",
        })
    _seed(tmp_db, rows)
    agg = aggregate_by_bucket(tmp_db, "market_cap_bucket", as_of_date="2026-04-12")
    by_b = {r["bucket"]: r for r in agg}
    assert by_b["微盘"]["n_samples"] == 5
    assert abs(by_b["微盘"]["direction_hit_rate"] - 0.4) < 1e-9
    assert by_b["大盘"]["n_samples"] == 10
    assert abs(by_b["大盘"]["direction_hit_rate"] - 0.9) < 1e-9


def test_aggregate_by_bucket_invalid_column(tmp_db):
    import pytest
    with pytest.raises(ValueError):
        aggregate_by_bucket(tmp_db, "bad_column", as_of_date="2026-04-12")


def test_format_markdown_contains_all_sections(tmp_db):
    rows = [{
        "code": "A.SZ", "trade_date": "2025-01-01",
        "sample_end_date": "2025-01-15",
        "pred_10d": 0.02, "actual_10d": 0.03,
        "market_cap_bucket": "小盘", "industry": "计算机", "liquidity_bucket": "中高",
    }]
    _seed(tmp_db, rows)
    md = format_markdown_report(tmp_db, as_of_date="2026-04-12")
    assert "按市值分组" in md
    assert "按行业" in md
    assert "按流动性分组" in md
    assert "小盘" in md
