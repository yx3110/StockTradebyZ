"""P3.1 单测: scoring_router 路由逻辑保护 (重构前后语义不变)."""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from stock_selctor.scoring_router import (
    RouteResult,
    route_scoring_version,
    route_ng106,
    route_ng200a,
    route_ng21,
)


@pytest.fixture
def db_with_regime():
    """临时 DB 含 market_amv + market_regime_signals 表."""
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "test.db"
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE market_amv (trade_date TEXT, amv_regime INTEGER)")
        conn.executemany(
            "INSERT INTO market_amv VALUES (?, ?)",
            [("2026-04-20", 1), ("2026-04-25", -1), ("2026-04-26", -1)],
        )
        conn.execute(
            "CREATE TABLE market_regime_signals (trade_date TEXT, regime_v2 INTEGER, b2_rv_percentile_252 REAL)")
        conn.executemany(
            "INSERT INTO market_regime_signals VALUES (?, ?, ?)",
            [("2026-04-20", 1, 0.5), ("2026-04-25", -1, 0.95), ("2026-04-26", -1, 0.92)],
        )
        # 交易日历 (staleness 按交易日计数): 4-26 后到 6-30 再排若干交易日
        conn.execute("CREATE TABLE daily_quotes (trade_date TEXT)")
        conn.executemany(
            "INSERT INTO daily_quotes VALUES (?)",
            [("2026-04-20",), ("2026-04-25",), ("2026-04-26",),
             ("2026-04-27",), ("2026-04-28",), ("2026-04-29",), ("2026-06-30",)],
        )
        conn.commit()
        conn.close()
        yield str(db)


# ────────────────────────────────────────────────────────────
# ng106 路由
# ────────────────────────────────────────────────────────────

def test_ng106_bull_regime(db_with_regime):
    res = route_ng106("ng1.0.6", "2026-04-20", db_with_regime)
    assert res.scoring_version == "ng1.0.1"
    assert res.bull_model == "ng1.0.1"
    assert res.bear_model == "ng1.0.4"
    assert res.ng106_mode is True
    # P0.1 (2026-04-27): overlay default-on for ng1.0.6 base.
    assert res.ng106_overlay_mode is True
    assert res.ng106_alt_mode is False
    assert res.ng106_overlay_regime == "bull"


def test_ng106_bear_regime(db_with_regime):
    res = route_ng106("ng1.0.6", "2026-04-25", db_with_regime)
    assert res.scoring_version == "ng1.0.4"
    assert res.ng106_overlay_regime == "bear"


def test_ng1062_bull(db_with_regime):
    """ng1.0.62 牛市 → ng1.0.7."""
    res = route_ng106("ng1.0.62", "2026-04-20", db_with_regime)
    assert res.scoring_version == "ng1.0.7"
    assert res.bull_model == "ng1.0.7"


def test_ng106_with_overlay(db_with_regime):
    res = route_ng106("ng1.0.6+overlay", "2026-04-20", db_with_regime)
    assert res.ng106_overlay_mode is True
    assert res.ng106_alt_mode is False
    assert res.scoring_version == "ng1.0.1"
    assert res.version_tag == "ng1.0.6+overlay"


def test_ng106_with_alt(db_with_regime):
    """+alt → bull 切换到 ng1.7.0."""
    res = route_ng106("ng1.0.6+alt", "2026-04-20", db_with_regime)
    assert res.ng106_alt_mode is True
    assert res.bull_model == "ng1.7.0"
    assert res.scoring_version == "ng1.7.0"


def test_ng106_alt_overlay_combo(db_with_regime):
    """+alt+overlay 组合."""
    res = route_ng106("ng1.0.6+alt+overlay", "2026-04-20", db_with_regime)
    assert res.ng106_alt_mode is True
    assert res.ng106_overlay_mode is True
    assert res.bull_model == "ng1.7.0"
    assert res.scoring_version == "ng1.7.0"


def test_ng106_db_error_fail_defensive_to_bear():
    """DB 不存在时, fail-defensive 到 bear + 降级标记 (2026-07-11 行为变更).

    旧行为 fail-open 到 bull = 数据管道故障日用最激进配置; 数据故障与市场
    异动日高度相关, 故障必须选防御侧, 且降级要可见。
    """
    res = route_ng106("ng1.0.6", "2026-04-20", "/non/existent/path.db")
    assert res.scoring_version == res.bear_model
    assert res.ng106_overlay_regime == "bear"
    assert res.regime_degraded  # 降级原因非空


def test_ng106_stale_regime_fail_defensive_to_bear(db_with_regime):
    """regime 信号陈旧 (> REGIME_MAX_STALE_DAYS) 时按熊市防御路由."""
    # db_with_regime 的最新 amv 行日期是 2026-04-20 附近; 用远未来 target_date 触发 staleness
    res = route_ng106("ng1.0.6", "2026-06-30", db_with_regime)
    assert res.scoring_version == res.bear_model
    assert "陈旧" in res.regime_degraded


# ────────────────────────────────────────────────────────────
# ng2.0a / ng2.1 路由
# ────────────────────────────────────────────────────────────

def test_ng200a_bull(db_with_regime):
    res = route_ng200a("ng2.0a", "2026-04-20", db_with_regime)
    assert res.scoring_version == "ng1.0.1"
    assert res.ng200a_mode is True
    assert res.bull_model == "ng1.0.1"


def test_ng200a_bear(db_with_regime):
    res = route_ng200a("ng2.0a", "2026-04-25", db_with_regime)
    assert res.scoring_version == "ng1.0.4"


def test_ng21_bull(db_with_regime):
    res = route_ng21("ng2.1", "2026-04-20", db_with_regime)
    assert res.scoring_version == "ng2.1-bull"
    assert res.ng21_mode is True
    assert res.ng21_regime == "bull"


def test_ng21_bear(db_with_regime):
    res = route_ng21("ng2.1", "2026-04-25", db_with_regime)
    assert res.scoring_version == "ng2.1-bear"
    assert res.ng21_regime == "bear"


# ────────────────────────────────────────────────────────────
# Top-level dispatch
# ────────────────────────────────────────────────────────────

def test_dispatch_routes_correctly(db_with_regime):
    cases = [
        ("ng1.0.6", "ng1.0.1"),
        ("ng1.0.6+alt", "ng1.7.0"),
        ("ng2.0a", "ng1.0.1"),
        ("ng2.1", "ng2.1-bull"),
        ("v3.95", "v3.95"),  # 透传
    ]
    for input_ver, expected_out in cases:
        res = route_scoring_version(input_ver, "2026-04-20", db_with_regime)
        assert res.scoring_version == expected_out, \
            f"input={input_ver} expected={expected_out} got={res.scoring_version}"


def test_dispatch_passthrough_for_non_moe():
    """非 MOE 版本 (v3.9, v4.x, ng1.0.1 等) 应透传不查 DB."""
    res = route_scoring_version("ng1.0.1", "2026-04-20", "/non/existent.db")
    assert res.scoring_version == "ng1.0.1"
    assert res.ng106_mode is False
    assert res.ng200a_mode is False
    assert res.ng21_mode is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
