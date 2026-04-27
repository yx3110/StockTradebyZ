"""P1.3 单测: signal_trust 🟡 软扣分."""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def trust_db():
    """临时 DB 含 signal_trust_scores 表 + 4 种 tag 样本."""
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "test.db"
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE signal_trust_scores (code TEXT PRIMARY KEY, trust_tag TEXT)")
        conn.executemany(
            "INSERT INTO signal_trust_scores VALUES (?, ?)",
            [
                ("000001", "🟢可信"),
                ("000002", "🟡存疑"),
                ("000003", "🔴高风险"),
                ("000004", "⚪数据不足"),
                ("000005", "🟡存疑"),
            ],
        )
        conn.commit()
        conn.close()
        yield str(db)


def _stock(code, score=0.10, industry="科技"):
    return {
        "stock_code": code,
        "stock_name": f"S{code}",
        "stock_type": "A股",
        "industry": industry,
        "rank_score": score,
        "composite": score,
    }


def test_yellow_penalty_applied(trust_db):
    from stock_selctor.post_filters import penalize_unreliable_by_trust_yellow

    stocks = [_stock("000001"), _stock("000002"), _stock("000003"), _stock("000005")]
    out, penalized = penalize_unreliable_by_trust_yellow(stocks, trust_db, penalty_factor=0.7)

    # 🟡 stocks 000002, 000005 should be penalized (×0.7)
    pen_codes = {s["stock_code"] for s in penalized}
    assert pen_codes == {"000002", "000005"}, f"unexpected penalized: {pen_codes}"

    # rank_score for 🟡 should be 0.07 = 0.10 × 0.7
    s2 = next(s for s in out if s["stock_code"] == "000002")
    assert abs(s2["rank_score"] - 0.07) < 1e-6
    assert abs(s2["_orig_rank_score"] - 0.10) < 1e-6
    assert s2["_trust_penalty_applied"] is True

    # 🟢 stock 000001 untouched
    s1 = next(s for s in out if s["stock_code"] == "000001")
    assert abs(s1["rank_score"] - 0.10) < 1e-6
    assert "_orig_rank_score" not in s1


def test_yellow_penalty_disabled_when_factor_one(trust_db):
    """penalty_factor=1.0 等价于不扣分."""
    from stock_selctor.post_filters import apply_post_filters

    stocks = [_stock("000001", 0.10), _stock("000002", 0.09), _stock("000005", 0.08)]
    summary = apply_post_filters(
        stocks, trust_db,
        enable_trust_filter=False,  # 关掉 🔴 过滤避免干扰
        industry_cap=0,
        enable_trust_yellow_penalty=True,
        yellow_penalty_factor=1.0,
    )
    # factor=1.0 时不应触发扣分
    assert summary["trust_penalized"] == []


def test_yellow_penalty_OFF_by_default(trust_db):
    """默认 OFF: 不传 enable_trust_yellow_penalty 时, 🟡 不应被扣分 (修复 silent prod 行为变更)."""
    from stock_selctor.post_filters import apply_post_filters

    stocks = [_stock("000001", 0.10), _stock("000002", 0.09)]   # 🟢 + 🟡
    summary = apply_post_filters(
        stocks, trust_db,
        enable_trust_filter=False,
        industry_cap=0,
        # NOT passing enable_trust_yellow_penalty
    )
    # 默认 False, 🟡 stocks 应保持原 score
    s2 = next(s for s in summary["stocks"] if s["stock_code"] == "000002")
    assert abs(s2["rank_score"] - 0.09) < 1e-6, "默认 OFF 时 🟡 不应被扣分"
    assert summary["trust_penalized"] == []


def test_full_pipeline_red_drop_yellow_penalize_industry_cap(trust_db):
    """完整路径: 🔴 剔除 + 🟡 扣分 + industry cap."""
    from stock_selctor.post_filters import apply_post_filters

    stocks = [
        _stock("000001", 0.20, "科技"),    # 🟢 高分
        _stock("000002", 0.18, "科技"),    # 🟡 0.18 → 0.126
        _stock("000003", 0.15, "科技"),    # 🔴 剔除
        _stock("000005", 0.14, "科技"),    # 🟡 0.14 → 0.098
        _stock("000006", 0.10, "金融"),    # unknown (not in DB)
    ]
    summary = apply_post_filters(
        stocks, trust_db,
        enable_trust_filter=True,
        industry_cap=2,                  # 科技最多 2 只
        enable_trust_yellow_penalty=True,
        yellow_penalty_factor=0.7,
    )

    # 🔴 000003 被剔除
    assert {s["stock_code"] for s in summary["trust_dropped"]} == {"000003"}
    # 🟡 000002, 000005 被扣分
    assert {s["stock_code"] for s in summary["trust_penalized"]} == {"000002", "000005"}

    # 重排 + industry cap=2 后, 科技应留 000001 (0.20) + 000002 (0.126) -> 不, 等等:
    # 重排后: 000001=0.20, 000002=0.126, 000005=0.098, 000006=0.10
    # 科技排序: 000001 > 000002 > 000005 (cap=2 → drop 000005)
    final_codes = [s["stock_code"] for s in summary["stocks"]]
    # 期望 industry_cap 让 000005 出局 (科技第 3)
    assert "000003" not in final_codes  # 🔴
    assert "000005" not in final_codes  # 🟡 因 industry cap 出局


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
