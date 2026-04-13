#!/usr/bin/env python3
"""周度全局失效统计, 输出 Markdown 报告."""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_trust.global_stats import format_markdown_report
from signal_trust.constants import DEFAULT_DB_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(db_path: str = DEFAULT_DB_PATH, out_dir: str = "reports/signal_trust",
         as_of_date: str | None = None):
    d = as_of_date or datetime.today().strftime("%Y-%m-%d")
    md = format_markdown_report(db_path, as_of_date=d)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"global_stats_{d.replace('-', '')}.md"
    f.write_text(md, encoding="utf-8")
    logger.info(f"已写入: {f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", default=DEFAULT_DB_PATH)
    ap.add_argument("--out-dir", default="reports/signal_trust")
    ap.add_argument("--as-of-date", default=None)
    args = ap.parse_args()
    main(args.db_path, args.out_dir, args.as_of_date)
