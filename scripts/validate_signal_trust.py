#!/usr/bin/env python3
"""数据健康检查 + 泄露自检."""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_trust.db import connect
from signal_trust.constants import DEFAULT_DB_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(db_path: str = DEFAULT_DB_PATH):
    conn = connect(db_path)
    try:
        cur = conn.cursor()

        # 样本统计
        (total,) = cur.execute("SELECT COUNT(*) FROM signal_trust_samples").fetchone()
        (with_actual,) = cur.execute(
            "SELECT COUNT(*) FROM signal_trust_samples WHERE actual_10d IS NOT NULL"
        ).fetchone()
        (date_min, date_max) = cur.execute(
            "SELECT MIN(trade_date), MAX(trade_date) FROM signal_trust_samples"
        ).fetchone()
        logger.info(f"样本总数: {total:,} (含 actual: {with_actual:,}, NULL: {total-with_actual:,})")
        logger.info(f"覆盖期: {date_min} ~ {date_max}")

        # 版本贡献
        logger.info("各版本样本数:")
        for row in cur.execute(
            "SELECT version, COUNT(*) AS n FROM signal_trust_samples "
            "GROUP BY version ORDER BY n DESC"
        ).fetchall():
            logger.info(f"  {row['version']}: {row['n']:,}")

        # 标签分布
        tags = cur.execute(
            "SELECT trust_tag, COUNT(*) AS n FROM signal_trust_scores GROUP BY trust_tag"
        ).fetchall()
        logger.info("标签分布: " + ", ".join(f"{r['trust_tag']}={r['n']}" for r in tags))

        # 🚨 泄露自检
        today = datetime.today().strftime("%Y-%m-%d")
        (leak,) = cur.execute(
            "SELECT COUNT(*) FROM signal_trust_samples "
            "WHERE sample_end_date >= ? AND actual_10d IS NOT NULL",
            (today,),
        ).fetchone()
        if leak > 0:
            logger.warning(f"⚠️ 疑似泄露: {leak} 条未来样本已有 actual_10d (应为 NULL)")
        else:
            logger.info("✓ 泄露自检通过")
    finally:
        conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", default=DEFAULT_DB_PATH)
    args = ap.parse_args()
    main(args.db_path)
