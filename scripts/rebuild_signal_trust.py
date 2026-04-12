#!/usr/bin/env python3
"""首次建库: 扫所有历史报告 → samples 表 → scores 表."""
import argparse
import logging
import sys
from pathlib import Path

# 允许从项目根目录运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_trust.db import migrate
from signal_trust.sample_builder import (
    scan_reports, dedupe_by_version, compute_actual_10d,
    compute_market_cap_bucket, compute_liquidity_bucket,
    compute_liquidity_thresholds, compute_sample_end_date,
    _industry_lookup, upsert_samples,
)
from signal_trust.scorer import compute_scores, upsert_scores
from signal_trust.constants import DEFAULT_DB_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(db_path: str = DEFAULT_DB_PATH, reports_root: str = "reports",
         as_of_date: str | None = None):
    logger.info("迁移 DB schema...")
    migrate(db_path)

    logger.info(f"扫描报告目录: {reports_root}")
    raw = list(scan_reports([reports_root]))
    logger.info(f"  原始记录: {len(raw):,}")
    deduped = dedupe_by_version(raw)
    logger.info(f"  去重后: {len(deduped):,}")

    # 确定 as_of_date
    if as_of_date is not None:
        as_of = as_of_date
    elif deduped:
        as_of = max(r["trade_date"] for r in deduped)
    else:
        logger.warning("无样本, 退出")
        return

    # 计算 sample_end_date + actual_10d + 分组标签
    logger.info("计算 actual_10d + 分组...")
    codes = list({r["code"] for r in deduped})
    industry_map = _industry_lookup(db_path, codes)
    liq_thresholds = compute_liquidity_thresholds(db_path, as_of)

    # T+10 未到的样本用哨兵日期占位, 后续 daily update 的 _backfill_actuals 会补齐.
    END_DATE_SENTINEL = "9999-12-31"
    enriched = []
    for i, r in enumerate(deduped):
        if i % 10000 == 0 and i > 0:
            logger.info(f"  进度: {i:,}/{len(deduped):,}")
        end_date = compute_sample_end_date(db_path, r["trade_date"])
        actual = compute_actual_10d(db_path, r["code"], r["trade_date"])
        mc = compute_market_cap_bucket(db_path, r["code"], r["trade_date"])
        liq = compute_liquidity_bucket(db_path, r["code"], r["trade_date"], liq_thresholds)
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

    logger.info(f"入库: {len(enriched):,} 条")
    upsert_samples(db_path, enriched, update_actual=True)

    logger.info(f"计算全市场分数 as_of_date={as_of}...")
    scores = compute_scores(db_path, as_of_date=as_of)
    upsert_scores(db_path, scores)
    logger.info(f"  {len(scores):,} 只股票已打分")

    # 标签分布
    from collections import Counter
    dist = Counter(s["trust_tag"] for s in scores.values())
    logger.info(f"标签分布: {dict(dist)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", default=DEFAULT_DB_PATH)
    ap.add_argument("--reports-root", default="reports")
    ap.add_argument("--as-of-date", default=None)
    args = ap.parse_args()
    main(args.db_path, args.reports_root, args.as_of_date)
