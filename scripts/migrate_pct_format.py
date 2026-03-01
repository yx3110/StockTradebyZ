#!/usr/bin/env python3
"""
一次性迁移脚本: 将 daily_quotes.price_change_pct 从百分比格式(9.5=9.5%)统一为小数格式(0.095=9.5%)

分析结果:
  - 2025-09 之前: 百分比格式 (avg |pct| ~ 1.0-2.0)
  - 2025-09 及之后: 小数格式 (avg |pct| ~ 0.015-0.020)

迁移策略:
  1. trade_date < '2025-09-01' 的所有记录: 除以100
  2. trade_date >= '2025-09-01' 且 |pct| > 0.50 的记录: 除以100 (少量异常/新股)
  3. 其余: 保持不变 (已是小数格式)

用法:
    python3 scripts/migrate_pct_format.py           # dry-run
    python3 scripts/migrate_pct_format.py --execute  # 实际执行
"""

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / 'data_adapter' / 'stock_data.db'

# 边界日期: 数据分析确认 2025-08-25 开始是小数格式
# (2025-08-22及之前avg|pct|~1.5, 2025-08-25起avg|pct|~0.018)
BOUNDARY_DATE = '2025-08-25'


def main():
    execute = '--execute' in sys.argv

    if not DB_PATH.exists():
        print(f"数据库不存在: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    # 统计
    cur.execute("SELECT COUNT(*) FROM daily_quotes WHERE price_change_pct IS NOT NULL")
    total = cur.fetchone()[0]

    cur.execute(f"SELECT COUNT(*) FROM daily_quotes WHERE trade_date < '{BOUNDARY_DATE}' AND price_change_pct IS NOT NULL")
    old_count = cur.fetchone()[0]

    cur.execute(f"SELECT COUNT(*) FROM daily_quotes WHERE trade_date >= '{BOUNDARY_DATE}' AND ABS(price_change_pct) > 0.50")
    new_outliers = cur.fetchone()[0]

    need_migrate = old_count + new_outliers

    print(f"数据库: {DB_PATH}")
    print(f"总记录数 (非NULL): {total:,}")
    print(f"百分比格式 (< {BOUNDARY_DATE}): {old_count:,}")
    print(f"新数据异常 (>= {BOUNDARY_DATE}, |pct| > 0.50): {new_outliers:,}")
    print(f"需要迁移总计: {need_migrate:,}")
    print()

    # 显示各阶段样例
    cur.execute(f"""
        SELECT trade_date,
               AVG(ABS(price_change_pct)) as avg_abs,
               COUNT(*) as cnt
        FROM daily_quotes
        WHERE trade_date IN (
            SELECT trade_date FROM daily_quotes
            WHERE trade_date < '{BOUNDARY_DATE}' AND price_change_pct IS NOT NULL
            GROUP BY trade_date
            ORDER BY trade_date DESC LIMIT 3
        )
        GROUP BY trade_date ORDER BY trade_date DESC
    """)
    print("旧数据样例 (百分比 → 小数):")
    for row in cur.fetchall():
        print(f"  {row[0]}: avg|pct|={row[1]:.4f} → {row[1]/100:.6f} ({row[2]:,} stocks)")

    cur.execute(f"""
        SELECT trade_date,
               AVG(ABS(price_change_pct)) as avg_abs,
               COUNT(*) as cnt
        FROM daily_quotes
        WHERE trade_date IN (
            SELECT trade_date FROM daily_quotes
            WHERE trade_date >= '{BOUNDARY_DATE}' AND price_change_pct IS NOT NULL
            GROUP BY trade_date
            ORDER BY trade_date LIMIT 3
        )
        GROUP BY trade_date ORDER BY trade_date
    """)
    print("\n新数据样例 (已是小数, 不迁移):")
    for row in cur.fetchall():
        print(f"  {row[0]}: avg|pct|={row[1]:.6f} ({row[2]:,} stocks)")
    print()

    if not execute:
        print("这是 dry-run 模式。使用 --execute 参数执行实际迁移。")
        conn.close()
        return

    # 执行迁移
    print("=" * 60)
    print("开始迁移...")

    # Step 1: 旧数据全部除以100
    cur.execute(f"""
        UPDATE daily_quotes
        SET price_change_pct = price_change_pct / 100.0
        WHERE trade_date < '{BOUNDARY_DATE}' AND price_change_pct IS NOT NULL
    """)
    step1 = cur.rowcount
    print(f"Step 1: 旧数据 (< {BOUNDARY_DATE}): 迁移 {step1:,} 条")

    # Step 2: 新数据中的异常值 (|pct| > 0.50, 即>50%变动不可能)
    cur.execute(f"""
        UPDATE daily_quotes
        SET price_change_pct = price_change_pct / 100.0
        WHERE trade_date >= '{BOUNDARY_DATE}' AND ABS(price_change_pct) > 0.50
    """)
    step2 = cur.rowcount
    print(f"Step 2: 新数据异常 (>= {BOUNDARY_DATE}, |pct| > 0.50): 迁移 {step2:,} 条")

    conn.commit()
    print(f"\n总迁移: {step1 + step2:,} 条")

    # 验证
    cur.execute("SELECT COUNT(*) FROM daily_quotes WHERE ABS(price_change_pct) > 0.50")
    remaining = cur.fetchone()[0]
    print(f"迁移后 |pct| > 0.50 的记录: {remaining} (应为0或极少)")

    # 抽样验证
    for month in ['2018-06', '2024-06', '2025-08', '2025-10', '2026-01']:
        cur.execute(f"""
            SELECT AVG(ABS(price_change_pct)) FROM daily_quotes
            WHERE SUBSTR(trade_date, 1, 7) = '{month}' AND price_change_pct IS NOT NULL
        """)
        r = cur.fetchone()
        if r and r[0]:
            print(f"  {month}: avg|pct| = {r[0]:.6f} (正常范围: 0.010-0.025)")

    conn.close()
    print("\n迁移完成!")


if __name__ == '__main__':
    main()
