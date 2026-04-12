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
