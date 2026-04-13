"""Tests for signal_trust.bulk_loader.BulkEnricher."""
import pytest

from signal_trust.bulk_loader import BulkEnricher


def _make_quotes(n: int) -> list[tuple[str, float, float]]:
    """Generate n consecutive dates starting 2026-01-01."""
    import datetime
    base = datetime.date(2026, 1, 1)
    result = []
    for i in range(n):
        d = base + datetime.timedelta(days=i)
        close = 100.0 + i * 0.5
        amount = 1e8 + i * 1e6
        result.append((d.isoformat(), close, amount))
    return result


def test_bulk_enricher_basic(seed_stock, tmp_db):
    """Seed 2 stocks × 15 days; verify actual_10d + market_cap_bucket + sample_end_date."""
    quotes = _make_quotes(15)
    dates = [d for d, _, _ in quotes]

    # A.SZ: 大盘 (circ_mv = 500_0000 万元 >= 500_0000 threshold)
    circ_a = {d: 500_0000.0 for d in dates}
    seed_stock("A.SZ", "银行", quotes, circ_mv_by_date=circ_a)

    # B.SZ: 微盘 (circ_mv = 20_0000 万元 < 30_0000 threshold)
    circ_b = {d: 20_0000.0 for d in dates}
    seed_stock("B.SZ", "计算机", quotes, circ_mv_by_date=circ_b)

    enr = BulkEnricher(tmp_db)
    enr.load()

    # actual_10d for T=day0, T+10=day10
    a = enr.actual_10d("A.SZ", quotes[0][0])
    expected = (quotes[10][1] - quotes[0][1]) / quotes[0][1]
    assert a is not None
    assert abs(a - expected) < 1e-9

    # B.SZ market cap = 20_0000 → 微盘 (< 30_0000)
    assert enr.market_cap_bucket("B.SZ", quotes[0][0]) == "微盘"

    # A.SZ market cap = 500_0000 → 大盘 (>= 500_0000)
    assert enr.market_cap_bucket("A.SZ", quotes[0][0]) == "大盘"

    # sample_end_date: day0 → day10 (HOLD_DAYS=10 trading days later)
    end = enr.sample_end_date(quotes[0][0])
    assert end == quotes[10][0]

    # Insufficient data: day14 (last day) has no T+10 → None
    assert enr.actual_10d("A.SZ", quotes[14][0]) is None


def test_bulk_enricher_actual_10d_missing_date(seed_stock, tmp_db):
    """actual_10d returns None when trade_date is not in data."""
    quotes = _make_quotes(15)
    seed_stock("C.SZ", "医药", quotes)
    enr = BulkEnricher(tmp_db)
    enr.load()
    assert enr.actual_10d("C.SZ", "1990-01-01") is None
    assert enr.actual_10d("UNKNOWN.SZ", quotes[0][0]) is None


def test_bulk_enricher_sample_end_date_boundary(seed_stock, tmp_db):
    """sample_end_date returns None for dates without T+HOLD_DAYS days of future data."""
    quotes = _make_quotes(12)
    seed_stock("D.SZ", "电子", quotes)
    enr = BulkEnricher(tmp_db)
    enr.load()
    # day0 has exactly 12 days total; day0+10=day10 exists → not None
    assert enr.sample_end_date(quotes[0][0]) == quotes[10][0]
    # day2+10=day12 does not exist (only 12 days, indices 0-11) → None
    assert enr.sample_end_date(quotes[2][0]) is None


def test_bulk_enricher_liquidity_bucket(seed_stock, tmp_db):
    """Liquidity bucket classifies by 30-day mean amount vs thresholds."""
    # amount = 5e7 on every day → mean = 5e7
    quotes = [(f"2026-01-{d:02d}", 100.0, 5e7) for d in range(1, 31)]
    seed_stock("X.SZ", "计算机", quotes)
    enr = BulkEnricher(tmp_db)
    enr.load()
    thresholds = (1e8, 3e8, 1e9)
    b = enr.liquidity_bucket("X.SZ", "2026-01-30", thresholds)
    assert b == "低"  # 5e7 < 1e8

    # Unknown code → "未知"
    assert enr.liquidity_bucket("NOPE.SZ", "2026-01-30", thresholds) == "未知"


def test_bulk_enricher_compute_liquidity_thresholds(seed_stock, tmp_db):
    """compute_liquidity_thresholds returns a tuple of 3 floats."""
    quotes = _make_quotes(35)
    seed_stock("E.SZ", "银行", quotes)
    enr = BulkEnricher(tmp_db)
    enr.load()
    thresholds = enr.compute_liquidity_thresholds(quotes[-1][0])
    assert len(thresholds) == 3
    p25, p50, p75 = thresholds
    assert p25 <= p50 <= p75


def test_bulk_enricher_sparse_liquidity_thresholds(tmp_db):
    """compute_liquidity_thresholds falls back to defaults when data is sparse."""
    enr = BulkEnricher(tmp_db)
    enr.load()
    thresholds = enr.compute_liquidity_thresholds("2026-01-01")
    assert thresholds == (1e8, 3e8, 1e9)
