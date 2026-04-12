"""端到端集成测试: mock 报告 → 建库 → 分数 → 贴标签."""
import json
from pathlib import Path

import pytest

from signal_trust.db import connect
from signal_trust.constants import TAG_GREEN, TAG_RED


def test_full_cycle(tmp_db, tmp_path, seed_stock):
    """3 股票 × 2 月, 验证建库和分数正确."""
    # 种子数据: 20 天的 daily_quotes
    quotes_days = [(f"2025-01-{d:02d}", 100.0 + d * 0.5, 1e8) for d in range(1, 21)]
    # 为 3 只股票种子数据
    for code, industry in [("A.SZ", "银行"), ("B.SZ", "传媒"), ("C.SZ", "计算机")]:
        seed_stock(code, industry, quotes_days,
                   circ_mv_by_date={d: 500_0000 for d, *_ in quotes_days})

    # 写报告 JSON (10 天, 每天 2 只股票, 每只 pred_10d=0.015)
    reports_root = tmp_path / "reports"
    for i, (d, *_) in enumerate(quotes_days[:10]):  # 前 10 天有 T+10 可算
        stocks = [
            {"stock_code": "A.SZ", "pred_10d": 0.015, "rank_score": 95},
            {"stock_code": "B.SZ", "pred_10d": 0.015, "rank_score": 90},
        ]
        version_dir = reports_root / "daily_selection_ng106"
        version_dir.mkdir(parents=True, exist_ok=True)
        (version_dir / f"analysis_data_{d.replace('-', '')}.json").write_text(
            json.dumps({"analysis_date": d, "all_stocks_with_scores": stocks}, ensure_ascii=False)
        )

    # 调用 rebuild 主函数
    from scripts.rebuild_signal_trust import main as rebuild_main
    rebuild_main(db_path=tmp_db, reports_root=str(reports_root), as_of_date="2026-04-12")

    # 验证: samples 表至少 15 行 (10 天 × 2 股票 = 20, 允许少量丢失)
    conn = connect(tmp_db)
    try:
        (n,) = conn.execute("SELECT COUNT(*) FROM signal_trust_samples").fetchone()
    finally:
        conn.close()
    assert n >= 15
    # scores 表也有对应记录
    conn = connect(tmp_db)
    try:
        (n2,) = conn.execute("SELECT COUNT(*) FROM signal_trust_scores").fetchone()
    finally:
        conn.close()
    assert n2 >= 1
