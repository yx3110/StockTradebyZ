#!/usr/bin/env python3
"""每日增量: (A) 新报告入库, (B) 回填 actual_10d, (C) 刷新分数."""
import argparse
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_trust.db import connect, migrate
from signal_trust.sample_builder import (
    scan_reports,
    dedupe_by_version,
    compute_actual_10d,
    compute_market_cap_bucket,
    compute_liquidity_bucket,
    compute_liquidity_thresholds,
    compute_sample_end_date,
    _industry_lookup,
    upsert_samples,
)
from signal_trust.scorer import compute_scores, upsert_scores
from signal_trust.constants import DEFAULT_DB_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

END_DATE_SENTINEL = "9999-12-31"


def _new_samples_for_date(db_path: str, reports_root: str, trade_date: str) -> int:
    """(A) 扫当日新报告, 入库 (actual_10d 可能 NULL)."""
    all_records = []
    for r in scan_reports([reports_root]):
        if r["trade_date"] == trade_date:
            all_records.append(r)
    deduped = dedupe_by_version(all_records)
    if not deduped:
        logger.info(f"  (A) 无新样本 @ {trade_date}")
        return 0

    codes = [r["code"] for r in deduped]
    industry_map = _industry_lookup(db_path, codes)
    end_date = compute_sample_end_date(db_path, trade_date)
    liq_thr = compute_liquidity_thresholds(db_path, trade_date)

    enriched = []
    for r in deduped:
        actual = compute_actual_10d(db_path, r["code"], r["trade_date"])
        mc = compute_market_cap_bucket(db_path, r["code"], r["trade_date"])
        liq = compute_liquidity_bucket(db_path, r["code"], r["trade_date"], liq_thr)
        enriched.append({
            "code": r["code"],
            "trade_date": r["trade_date"],
            "sample_end_date": end_date if end_date else END_DATE_SENTINEL,
            "pred_10d": r["pred_10d"],
            "actual_10d": actual,
            "version": r["version"],
            "market_cap_bucket": mc,
            "industry": industry_map.get(r["code"], "未分类"),
            "liquidity_bucket": liq,
        })
    n = upsert_samples(db_path, enriched, update_actual=False)
    logger.info(f"  (A) 入库 {n} 条 @ {trade_date}")
    return n


def _backfill_actuals(db_path: str) -> int:
    """(B) 对所有 actual_10d IS NULL 样本, 尝试计算实际收益 + 补 sample_end_date."""
    conn = connect(db_path)
    try:
        pending = conn.execute(
            "SELECT code, trade_date FROM signal_trust_samples "
            "WHERE actual_10d IS NULL"
        ).fetchall()
    finally:
        conn.close()
    if not pending:
        logger.info("  (B) 无回填任务")
        return 0
    logger.info(f"  (B) 尝试回填 {len(pending)} 条")
    updated = 0
    conn = connect(db_path)
    try:
        for row in pending:
            actual = compute_actual_10d(db_path, row["code"], row["trade_date"])
            if actual is None:
                continue
            end_date = compute_sample_end_date(db_path, row["trade_date"])
            if end_date is None:
                continue
            conn.execute(
                "UPDATE signal_trust_samples SET actual_10d = ?, sample_end_date = ? "
                "WHERE code = ? AND trade_date = ? AND actual_10d IS NULL",
                (actual, end_date, row["code"], row["trade_date"]),
            )
            updated += 1
        conn.commit()
    finally:
        conn.close()
    logger.info(f"  (B) 实际回填 {updated} 条")
    return updated


def main(
    db_path: str = DEFAULT_DB_PATH,
    reports_root: str = "reports",
    trade_date: str | None = None,
) -> None:
    migrate(db_path)
    today = trade_date or datetime.today().strftime("%Y-%m-%d")
    logger.info(f"=== Signal Trust 增量 @ {today} ===")

    _new_samples_for_date(db_path, reports_root, today)
    _backfill_actuals(db_path)

    logger.info(f"(C) 刷新全部分数 as_of_date={today}")
    scores = compute_scores(db_path, as_of_date=today)
    n = upsert_scores(db_path, scores)
    logger.info(f"  刷新 {n} 条 scores")

    dist = Counter(s["trust_tag"] for s in scores.values())
    logger.info(f"标签分布: {dict(dist)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Signal Trust 每日增量更新")
    ap.add_argument("--db-path", default=DEFAULT_DB_PATH)
    ap.add_argument("--reports-root", default="reports")
    ap.add_argument("--date", default=None, help="trade_date; 默认今日")
    args = ap.parse_args()
    main(args.db_path, args.reports_root, args.date)
