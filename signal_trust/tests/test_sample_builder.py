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
