#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
个股 price_change_pct 回填 (2026-07-11 北极星 P0 修复配套, P0.1b)

背景: 个股 price_change_pct 2020 年 95.65% NULL、2021-2024 100% NULL —
北极星回测的涨停剔除在该段完全失效, 与 Pre-2020/2025-26 (生效) 口径不对称。

计算: pct = (close_t × adj_t) / (close_{t-1} × adj_{t-1}) − 1  (复权价, 小数制)
- 与 2018/2019/2025 现有 Tushare pct_chg 语义一致 (pre_close 除权口径, 无除权 artifact)
- 仅填 price_change_pct IS NULL 的行, 不覆盖已有值
- adj_factor 任一侧缺失的行保持 NULL (诚实缺失, 不用原始价顶替 —
  原始价在除权日会产生 -30%~-50% 假涨跌幅)
- 前置依赖: fetch_data/backfill_adj_factor.py 已跑完

用法:
  python3 fetch_data/backfill_price_change_pct.py                    # 2019-12..2025-01 全段
  python3 fetch_data/backfill_price_change_pct.py --start 2019-12-01 --end 2025-01-31
  python3 fetch_data/backfill_price_change_pct.py --dry-run
"""

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import get_db_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="个股 price_change_pct 复权回填")
    # 默认段前后各带 buffer: 缺口是 2020-01..2024-12, 需要 2019 年末的前收作基准
    parser.add_argument("--start", default="2019-12-01")
    parser.add_argument("--end", default="2025-01-31")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    conn = sqlite3.connect(str(get_db_path()))
    conn.execute("PRAGMA busy_timeout=30000")

    logger.info(f"加载 {args.start} ~ {args.end} A股 close/adj_factor ...")
    df = pd.read_sql_query(
        """
        SELECT dq.security_id, dq.trade_date, dq.close, dq.adj_factor,
               (dq.price_change_pct IS NULL) AS pct_null
        FROM daily_quotes dq JOIN securities s ON s.id = dq.security_id
        WHERE s.type = 'A股' AND dq.trade_date BETWEEN ? AND ?
          AND dq.close IS NOT NULL AND dq.close > 0
        ORDER BY dq.security_id, dq.trade_date
        """,
        conn, params=(args.start, args.end),
    )
    logger.info(f"共 {len(df)} 行, 其中 pct NULL {int(df['pct_null'].sum())} 行")

    # 复权价 + 组内前收 (adj_factor=1.0 是占位符, 视同缺失)
    adj = df["adj_factor"].where(df["adj_factor"] != 1.0)
    df["adj_close"] = df["close"] * adj
    g = df.groupby("security_id", sort=False)
    df["prev_adj_close"] = g["adj_close"].shift(1)

    fillable = df[df["pct_null"].astype(bool)
                  & df["adj_close"].notna()
                  & df["prev_adj_close"].notna()
                  & (df["prev_adj_close"] > 0)].copy()
    fillable["pct"] = fillable["adj_close"] / fillable["prev_adj_close"] - 1.0
    n_unfillable = int(df["pct_null"].sum()) - len(fillable)
    logger.info(f"可回填 {len(fillable)} 行; 因 adj_factor 缺失/首行无前收跳过 {n_unfillable} 行 (保持 NULL)")

    # 极端值 sanity: 复权口径下 |pct|>0.44 (新股首日上限) 视为异常, 记录不回填
    extreme = fillable[fillable["pct"].abs() > 0.44]
    if len(extreme) > 0:
        logger.warning(f"异常涨跌幅 {len(extreme)} 行 (|pct|>44%, 多为数据质量问题), 跳过不回填; "
                       f"样例: {extreme[['security_id','trade_date','pct']].head(3).values.tolist()}")
        fillable = fillable[fillable["pct"].abs() <= 0.44]

    if args.dry_run:
        logger.info(f"[dry-run] 将回填 {len(fillable)} 行, 样例:\n{fillable.head(5)}")
        return

    params = list(zip(fillable["pct"].astype(float),
                      fillable["security_id"].astype(int),
                      fillable["trade_date"]))
    CHUNK = 200_000
    total = 0
    for i in range(0, len(params), CHUNK):
        cur = conn.executemany(
            "UPDATE daily_quotes SET price_change_pct=? "
            "WHERE security_id=? AND trade_date=? AND price_change_pct IS NULL",
            params[i:i + CHUNK],
        )
        conn.commit()
        total += cur.rowcount
        logger.info(f"进度 {min(i + CHUNK, len(params))}/{len(params)}: 累计更新 {total} 行")

    # 收尾验证: 分年 NULL 率 + 涨停检出数
    rows = conn.execute(
        """
        SELECT substr(dq.trade_date,1,4) y, COUNT(*) n,
               SUM(dq.price_change_pct IS NULL) n_null,
               SUM(dq.price_change_pct >= 0.095) n_limit
        FROM daily_quotes dq JOIN securities s ON s.id = dq.security_id
        WHERE s.type='A股' AND dq.trade_date BETWEEN ? AND ?
        GROUP BY y ORDER BY y
        """, (args.start, args.end)).fetchall()
    for y, n, n_null, n_limit in rows:
        logger.info(f"  {y}: NULL {n_null}/{n} ({n_null/n:.1%}), 涨幅≥9.5% 检出 {n_limit}")


if __name__ == "__main__":
    main()
