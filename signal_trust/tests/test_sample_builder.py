import pytest
from pathlib import Path
from signal_trust.sample_builder import scan_reports, dedupe_by_version
from signal_trust.constants import PRED_THRESHOLD


def test_scan_reports_filters_threshold(write_report):
    report_dir = write_report(
        "2026-01-10",
        [
            {"stock_code": "000001.SZ", "pred_10d": 0.015},   # 入选
            {"stock_code": "000002.SZ", "pred_10d": 0.005},   # 低于阈值
            {"stock_code": "000003.SZ", "pred_10d": 0.0},     # 0 被过滤
            {"stock_code": "000004.SZ", "pred_10d": None},    # None 被过滤
            {"stock_code": "000005.SZ"},                       # 缺字段
        ],
    )
    records = list(scan_reports([str(report_dir.parent)]))
    codes = {r["code"] for r in records}
    assert codes == {"000001.SZ"}


def test_scan_reports_extracts_version_from_dir(write_report):
    d106 = write_report("2026-01-10", [{"stock_code": "A.SZ", "pred_10d": 0.02}], version="ng106")
    d101 = write_report("2026-01-10", [{"stock_code": "A.SZ", "pred_10d": 0.02}], version="ng101")
    records = list(scan_reports([str(d106.parent)]))
    versions = {r["version"] for r in records}
    assert versions == {"ng106", "ng101"}


def test_dedupe_keeps_highest_priority():
    records = [
        {"code": "A.SZ", "trade_date": "2026-01-10", "pred_10d": 0.02, "version": "ng101"},
        {"code": "A.SZ", "trade_date": "2026-01-10", "pred_10d": 0.03, "version": "ng106"},
        {"code": "A.SZ", "trade_date": "2026-01-10", "pred_10d": 0.025, "version": "v39"},
    ]
    out = dedupe_by_version(records)
    assert len(out) == 1
    assert out[0]["version"] == "ng106"
    assert out[0]["pred_10d"] == 0.03


def test_dedupe_unknown_version_lowest_priority():
    records = [
        {"code": "A.SZ", "trade_date": "2026-01-10", "pred_10d": 0.02, "version": "未知xyz"},
        {"code": "A.SZ", "trade_date": "2026-01-10", "pred_10d": 0.03, "version": "ng101"},
    ]
    out = dedupe_by_version(records)
    assert out[0]["version"] == "ng101"


def test_dedupe_preserves_different_dates():
    records = [
        {"code": "A.SZ", "trade_date": "2026-01-10", "pred_10d": 0.02, "version": "ng101"},
        {"code": "A.SZ", "trade_date": "2026-01-11", "pred_10d": 0.03, "version": "ng101"},
    ]
    out = dedupe_by_version(records)
    assert len(out) == 2


def test_scan_reports_skips_malformed_json(tmp_path):
    """坏JSON文件不应崩溃, 静默跳过."""
    d = tmp_path / "reports" / "daily_selection_ng101"
    d.mkdir(parents=True)
    (d / "analysis_data_20260110.json").write_text("{not valid json")
    records = list(scan_reports([str(tmp_path / "reports")]))
    assert records == []


def test_scan_reports_date_fallback_from_filename(tmp_path):
    """JSON 缺 analysis_date 字段时, 从文件名 YYYYMMDD 推断."""
    d = tmp_path / "reports" / "daily_selection_ng101"
    d.mkdir(parents=True)
    import json as _json
    (d / "analysis_data_20260115.json").write_text(_json.dumps({
        "scoring_version": "ng101",
        "all_stocks_with_scores": [{"stock_code": "A.SZ", "pred_10d": 0.02}],
    }, ensure_ascii=False))
    records = list(scan_reports([str(tmp_path / "reports")]))
    assert len(records) == 1
    assert records[0]["trade_date"] == "2026-01-15"


def test_scan_reports_skips_nan_pred(tmp_path):
    """NaN 的 pred_10d 不应通过阈值过滤."""
    d = tmp_path / "reports" / "daily_selection_ng101"
    d.mkdir(parents=True)
    # float("nan") 是 NaN；用字符串 "nan" 触发该路径
    (d / "analysis_data_20260115.json").write_text(
        '{"analysis_date": "2026-01-15", "all_stocks_with_scores": ['
        '{"stock_code": "A.SZ", "pred_10d": "nan"},'
        '{"stock_code": "B.SZ", "pred_10d": 0.02}'
        ']}'
    )
    records = list(scan_reports([str(tmp_path / "reports")]))
    codes = {r["code"] for r in records}
    assert codes == {"B.SZ"}  # NaN A.SZ 被跳过, 正常 B.SZ 通过


# ---------------------------------------------------------------------------
# compute_actual_10d tests
# ---------------------------------------------------------------------------
from signal_trust.sample_builder import compute_actual_10d


def test_actual_10d_normal_case(seed_stock, tmp_db):
    # 10 个交易日后, close 上涨 5%
    quotes = []
    start_close = 100.0
    for i in range(15):
        d = f"2026-01-{i+1:02d}"
        quotes.append((d, start_close * (1 + 0.005 * i), 1e8))
    seed_stock("A.SZ", "计算机", quotes)
    # T = 2026-01-01 (idx 0), T+10 trading days = 2026-01-11 (idx 10)
    actual = compute_actual_10d(tmp_db, "A.SZ", "2026-01-01")
    expected = (quotes[10][1] - quotes[0][1]) / quotes[0][1]
    assert abs(actual - expected) < 1e-9


def test_actual_10d_missing_future_returns_none(seed_stock, tmp_db):
    # 只有 5 个交易日数据, T+10 不存在
    quotes = [(f"2026-01-{i+1:02d}", 100.0 + i, 1e8) for i in range(5)]
    seed_stock("A.SZ", "计算机", quotes)
    assert compute_actual_10d(tmp_db, "A.SZ", "2026-01-01") is None


def test_actual_10d_suspended_days_still_count(seed_stock, tmp_db):
    # 数据库里只存在交易日(停牌日无记录). T+10 指的是"数据库里第 10 个后续交易日"
    quotes = [(f"2026-01-{i+1:02d}", 100.0 + i * 2, 1e8) for i in range(12)]
    seed_stock("A.SZ", "计算机", quotes)
    actual = compute_actual_10d(tmp_db, "A.SZ", "2026-01-01")
    expected = (quotes[10][1] - quotes[0][1]) / quotes[0][1]
    assert abs(actual - expected) < 1e-9


def test_actual_10d_stock_not_found(tmp_db):
    assert compute_actual_10d(tmp_db, "NONEXIST.SZ", "2026-01-01") is None


# ---------------------------------------------------------------------------
# compute_sample_end_date tests
# ---------------------------------------------------------------------------
from signal_trust.sample_builder import compute_sample_end_date


def test_sample_end_date_normal(seed_stock, tmp_db):
    """12 trading days available, T+10 = idx 10 date."""
    quotes = [(f"2026-01-{i+1:02d}", 100.0, 1e8) for i in range(12)]
    seed_stock("A.SZ", "计算机", quotes)
    end = compute_sample_end_date(tmp_db, "2026-01-01")
    assert end == "2026-01-11"  # idx 10 of 12


def test_sample_end_date_insufficient_returns_none(seed_stock, tmp_db):
    """Only 5 days → return None."""
    quotes = [(f"2026-01-{i+1:02d}", 100.0, 1e8) for i in range(5)]
    seed_stock("A.SZ", "计算机", quotes)
    assert compute_sample_end_date(tmp_db, "2026-01-01") is None


# ---------------------------------------------------------------------------
# compute_market_cap_bucket / compute_liquidity_bucket / upsert_samples tests
# ---------------------------------------------------------------------------
from signal_trust.sample_builder import (
    compute_market_cap_bucket, compute_liquidity_bucket, upsert_samples,
)


def test_market_cap_bucket_thresholds(seed_stock, tmp_db):
    # daily_basic.circ_mv 单位是万元, 30亿 = 30_0000 万元
    seed_stock("A.SZ", "计算机",
               quotes=[("2026-01-01", 100, 1e8)],
               circ_mv_by_date={"2026-01-01": 20_0000})  # 20亿 → 微盘
    seed_stock("B.SZ", "计算机",
               quotes=[("2026-01-01", 100, 1e8)],
               circ_mv_by_date={"2026-01-01": 50_0000})  # 50亿 → 小盘
    seed_stock("C.SZ", "计算机",
               quotes=[("2026-01-01", 100, 1e8)],
               circ_mv_by_date={"2026-01-01": 200_0000})  # 200亿 → 中盘
    seed_stock("D.SZ", "计算机",
               quotes=[("2026-01-01", 100, 1e8)],
               circ_mv_by_date={"2026-01-01": 800_0000})  # 800亿 → 大盘
    assert compute_market_cap_bucket(tmp_db, "A.SZ", "2026-01-01") == "微盘"
    assert compute_market_cap_bucket(tmp_db, "B.SZ", "2026-01-01") == "小盘"
    assert compute_market_cap_bucket(tmp_db, "C.SZ", "2026-01-01") == "中盘"
    assert compute_market_cap_bucket(tmp_db, "D.SZ", "2026-01-01") == "大盘"


def test_market_cap_bucket_missing_data(seed_stock, tmp_db):
    seed_stock("X.SZ", "计算机", quotes=[("2026-01-01", 100, 1e8)])
    assert compute_market_cap_bucket(tmp_db, "X.SZ", "2026-01-01") == "未知"


def test_liquidity_bucket_uses_30day_mean(seed_stock, tmp_db):
    # 股票 X 的 30 日均成交额 5e7, 股票 Y 的 30 日均成交额 5e9
    qx = [(f"2026-01-{i+1:02d}", 100, 5e7) for i in range(30)]
    qy = [(f"2026-01-{i+1:02d}", 100, 5e9) for i in range(30)]
    seed_stock("X.SZ", "计算机", qx)
    seed_stock("Y.SZ", "计算机", qy)
    thresholds = (1e8, 3e8, 1e9)  # p25/p50/p75 假设阈值
    bx = compute_liquidity_bucket(tmp_db, "X.SZ", "2026-01-30", thresholds)
    by = compute_liquidity_bucket(tmp_db, "Y.SZ", "2026-01-30", thresholds)
    assert bx == "低"
    assert by == "高"


def test_upsert_samples_idempotent(tmp_db):
    from signal_trust.db import connect
    rows = [{
        "code": "A.SZ", "trade_date": "2026-01-10", "sample_end_date": "2026-01-24",
        "pred_10d": 0.015, "actual_10d": 0.02, "version": "ng106",
        "market_cap_bucket": "小盘", "industry": "计算机", "liquidity_bucket": "中高",
    }]
    upsert_samples(tmp_db, rows)
    upsert_samples(tmp_db, rows)  # 重跑
    conn = connect(tmp_db)
    try:
        (n,) = conn.execute("SELECT COUNT(*) FROM signal_trust_samples").fetchone()
    finally:
        conn.close()
    assert n == 1


def test_upsert_backfills_actual_10d(tmp_db):
    from signal_trust.db import connect
    # 首次入库 actual_10d=None, 二次更新应覆盖
    rows1 = [{
        "code": "A.SZ", "trade_date": "2026-01-10", "sample_end_date": "2026-01-24",
        "pred_10d": 0.015, "actual_10d": None, "version": "ng106",
        "market_cap_bucket": "小盘", "industry": "计算机", "liquidity_bucket": "中高",
    }]
    upsert_samples(tmp_db, rows1)
    rows2 = [{**rows1[0], "actual_10d": 0.025}]
    upsert_samples(tmp_db, rows2, update_actual=True)
    conn = connect(tmp_db)
    try:
        row = conn.execute(
            "SELECT actual_10d FROM signal_trust_samples WHERE code='A.SZ'"
        ).fetchone()
    finally:
        conn.close()
    assert abs(row["actual_10d"] - 0.025) < 1e-9
