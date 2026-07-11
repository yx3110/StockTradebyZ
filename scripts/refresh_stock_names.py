#!/usr/bin/env python3
"""Refresh securities.name from Tushare namechange.

DB 里的 stock name 在 ST 戴帽 / 摘帽 / 改名后不会自动更新.
调用 `pro.namechange()` 按日期范围拉取最近变更, 把每只票的"最新"名字写回
`securities.name`, 让下游 (candidate_ranker 的 ST 过滤 / single_stock_review 的
展示等) 看到准确的 ST 标记.

Usage:
    python3 scripts/refresh_stock_names.py                  # 默认回溯 540 天
    python3 scripts/refresh_stock_names.py --lookback-days 365
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import tushare as ts

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DB_PATH = PROJECT_ROOT / 'data_adapter' / 'stock_data.db'
CONFIG_PATH = PROJECT_ROOT / 'config.json'


def fetch_latest_names(lookback_days: int) -> dict[str, str]:
    """Query Tushare namechange, return {code_no_suffix: latest_name}."""
    # token 优先从 core.config/.env 获取
    try:
        from core.config import get_tushare_token
        token = get_tushare_token()
    except ImportError:
        token = json.load(open(CONFIG_PATH))['tushare']['token']
    ts.set_token(token)
    pro = ts.pro_api()

    end = datetime.today()
    start = end - timedelta(days=lookback_days)
    df = pro.namechange(start_date=start.strftime('%Y%m%d'),
                        end_date=end.strftime('%Y%m%d'))
    if df is None or df.empty:
        return {}
    df = df.sort_values(['ts_code', 'start_date'])
    latest = df.groupby('ts_code').tail(1)
    return {row.ts_code.split('.')[0]: row.name for row in latest.itertuples()}


def update_db_names(latest: dict[str, str]) -> tuple[int, int]:
    """Apply name updates to securities table. Returns (n_changed, n_total_seen)."""
    if not latest:
        return 0, 0
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.execute('PRAGMA busy_timeout=30000')
    cur = conn.cursor()
    changed = 0
    seen = 0
    for code, new_name in latest.items():
        row = cur.execute('SELECT name FROM securities WHERE code=?', (code,)).fetchone()
        if not row:
            continue
        seen += 1
        if row[0] == new_name:
            continue
        cur.execute('UPDATE securities SET name=? WHERE code=?', (new_name, code))
        print(f'  {code}: {row[0]!r} -> {new_name!r}')
        changed += 1
    conn.commit()
    conn.close()
    return changed, seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lookback-days', type=int, default=540,
                    help='refresh names changed within the last N days (default 540)')
    args = ap.parse_args()

    print(f'[1/2] fetching namechange from Tushare (last {args.lookback_days} days)...')
    latest = fetch_latest_names(args.lookback_days)
    print(f'       {len(latest)} codes with name changes')

    print('[2/2] applying updates to securities table...')
    changed, seen = update_db_names(latest)
    print(f'\n✅ updated {changed} / {seen} matched rows (of {len(latest)} fetched)')


if __name__ == '__main__':
    main()
