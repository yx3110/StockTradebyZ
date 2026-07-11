#!/usr/bin/env python3
"""financial_indicator 日期格式统一迁移 (2026-07-11).

背景: 表内 ann_date/end_date 存在两种格式 —
  - TEXT 'YYYY-MM-DD' (166,750 行, backfill_financial_indicator.py 写入)
  - INTEGER YYYYMMDD  (37,776 行, quick_daily_update.py 写入 Tushare 原始值)
SQLite 类型排序中 INTEGER 恒小于 TEXT, 导致:
  - NG 生产管线的整数边界查询 (ng_cache_updater._load_financial_data) 对 81% 的
    TEXT 行完全不可见 → roe/or_yoy 等生产特征大面积静默缺失
  - v471/v473 的 TEXT 比较把所有 INTEGER 行(含未来公告)误判为 <= 任意日期
  - 同一份财报双格式共存 37,556 对, UNIQUE(security_id, end_date) 被击穿

迁移步骤 (单事务):
  1. 备份: CREATE TABLE financial_indicator_backup_YYYYMMDD AS SELECT *
  2. 去重: 同 (security_id, 归一化 end_date) 的重复组, 保留非空字段最多的行
     (实测 INTEGER 行 99/167 非空 vs TEXT 行 23/167, 通常保留 INTEGER 行)
     平手时保留 ann_date 较新者, 再平手保留 id 较大者
  3. 归一: 全部 ann_date/end_date 统一为 TEXT 'YYYY-MM-DD'
  4. 校验: typeof 全 text、无重复组、行数 = 原行数 - 删除数

默认 dry-run 只打印计划; 加 --apply 才实际写库。
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / 'data_adapter' / 'stock_data.db'


def norm_date(v) -> str:
    """任意格式 → 'YYYY-MM-DD'; None → ''"""
    if v is None:
        return ''
    s = str(v)[:10]
    if '-' in s:
        return s
    s = s[:8]
    return f'{s[:4]}-{s[4:6]}-{s[6:8]}' if len(s) == 8 else s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--apply', action='store_true', help='实际执行迁移 (默认 dry-run)')
    ap.add_argument('--db', default=str(DB_PATH))
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.execute('PRAGMA busy_timeout=30000')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    total = c.execute('SELECT COUNT(*) FROM financial_indicator').fetchone()[0]
    fmt_counts = dict(c.execute(
        'SELECT typeof(ann_date), COUNT(*) FROM financial_indicator GROUP BY typeof(ann_date)'
    ).fetchall())
    print(f'总行数: {total}, ann_date 格式分布: {fmt_counts}')
    if fmt_counts.get('integer', 0) == 0:
        print('没有 INTEGER 格式行, 无需迁移。')
        return 0

    # --- 找重复组并选 keeper ---
    print('扫描重复组...')
    dup_rows = c.execute('''
        SELECT * FROM financial_indicator WHERE (security_id, replace(CAST(end_date AS TEXT), '-', '')) IN (
            SELECT security_id, replace(CAST(end_date AS TEXT), '-', '')
            FROM financial_indicator
            GROUP BY security_id, replace(CAST(end_date AS TEXT), '-', '')
            HAVING COUNT(*) > 1
        )
    ''').fetchall()
    groups: dict[tuple, list] = {}
    for r in dup_rows:
        key = (r['security_id'], norm_date(r['end_date']))
        groups.setdefault(key, []).append(r)

    to_delete: list[int] = []
    for key, rows in groups.items():
        # keeper: 非空字段最多 → ann_date 最新 → id 最大
        rows.sort(key=lambda r: (
            sum(1 for v in tuple(r) if v is not None),
            norm_date(r['ann_date']),
            r['id'],
        ))
        to_delete.extend(r['id'] for r in rows[:-1])

    n_int = c.execute(
        "SELECT COUNT(*) FROM financial_indicator WHERE typeof(ann_date)='integer'"
    ).fetchone()[0]
    print(f'重复组: {len(groups)}, 将删除冗余行: {len(to_delete)}')
    print(f'将归一化 INTEGER 行: ~{n_int} (去重后剩余部分)')

    if not args.apply:
        print('\n[dry-run] 未做任何修改。加 --apply 执行。')
        return 0

    backup_name = f"financial_indicator_backup_{datetime.now().strftime('%Y%m%d')}"
    exists = c.execute(
        "SELECT 1 FROM sqlite_master WHERE name=?", (backup_name,)
    ).fetchone()
    if exists:
        print(f'备份表 {backup_name} 已存在, 中止 (先确认/删除旧备份再跑)')
        return 1

    try:
        c.execute('BEGIN IMMEDIATE')
        print(f'1/4 备份 → {backup_name}')
        c.execute(f'CREATE TABLE {backup_name} AS SELECT * FROM financial_indicator')

        print(f'2/4 删除 {len(to_delete)} 条冗余重复行')
        for i in range(0, len(to_delete), 500):
            chunk = to_delete[i:i + 500]
            c.execute(
                f"DELETE FROM financial_indicator WHERE id IN ({','.join('?' * len(chunk))})",
                chunk,
            )

        print('3/4 归一化日期格式 → TEXT YYYY-MM-DD')
        c.execute('''
            UPDATE financial_indicator
            SET ann_date = CASE WHEN ann_date IS NULL THEN NULL ELSE
                    substr(CAST(ann_date AS TEXT),1,4) || '-' ||
                    substr(CAST(ann_date AS TEXT),5,2) || '-' ||
                    substr(CAST(ann_date AS TEXT),7,2) END,
                end_date = substr(CAST(end_date AS TEXT),1,4) || '-' ||
                    substr(CAST(end_date AS TEXT),5,2) || '-' ||
                    substr(CAST(end_date AS TEXT),7,2)
            WHERE typeof(ann_date) = 'integer' OR typeof(end_date) = 'integer'
        ''')

        print('4/4 校验')
        fmt_after = dict(c.execute(
            'SELECT typeof(ann_date), COUNT(*) FROM financial_indicator '
            'WHERE ann_date IS NOT NULL GROUP BY typeof(ann_date)'
        ).fetchall())
        remaining_dups = c.execute('''
            SELECT COUNT(*) FROM (
                SELECT 1 FROM financial_indicator
                GROUP BY security_id, end_date HAVING COUNT(*) > 1)
        ''').fetchone()[0]
        n_after = c.execute('SELECT COUNT(*) FROM financial_indicator').fetchone()[0]
        bad_fmt = c.execute(
            "SELECT COUNT(*) FROM financial_indicator WHERE end_date NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'"
        ).fetchone()[0]
        print(f'  格式分布: {fmt_after}, 剩余重复组: {remaining_dups}, '
              f'行数 {total}→{n_after} (-{total - n_after}), 非法格式: {bad_fmt}')
        if fmt_after.get('integer', 0) or remaining_dups or bad_fmt or (total - n_after) != len(to_delete):
            raise RuntimeError('校验失败, 回滚')
        conn.commit()
        print(f'✅ 迁移完成。备份表: {backup_name} (确认无误后可手动 DROP)')
    except Exception as e:
        conn.rollback()
        print(f'❌ 迁移失败已回滚: {e}')
        return 1
    finally:
        conn.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
