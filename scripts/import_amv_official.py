"""
导入指南针官方 0AMV (活跃市值) 日线 OHLCV 到 market_amv_official 表

用法:
    python3 scripts/import_amv_official.py [--csv PATH]

默认 CSV: ~/Downloads/活跃市值0AMV_日线OHLCV.csv
导入后真实行 is_simulated=0, 覆盖同日期的外推行 (is_simulated=1)。
"""
import argparse
import os
import sqlite3
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from indicators.market_amv import DB_PATH, ensure_official_table  # noqa: E402

DEFAULT_CSV = os.path.expanduser('~/Downloads/活跃市值0AMV_日线OHLCV.csv')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=DEFAULT_CSV)
    parser.add_argument('--db', default=DB_PATH)
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f'CSV 不存在: {args.csv}')
        sys.exit(1)

    df = pd.read_csv(args.csv, encoding='utf-8-sig')
    df.columns = [c.strip().lower() for c in df.columns]
    expected = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']
    if list(df.columns) != expected:
        print(f'列名不符合预期: {list(df.columns)}')
        sys.exit(1)

    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df = df.dropna(subset=['date', 'close']).drop_duplicates(subset=['date'], keep='last')

    conn = sqlite3.connect(args.db, timeout=30)
    conn.execute('PRAGMA busy_timeout=30000')
    ensure_official_table(conn)
    conn.executemany(
        'INSERT OR REPLACE INTO market_amv_official VALUES (?,?,?,?,?,?,?,0)',
        df[expected].itertuples(index=False, name=None),
    )
    conn.commit()

    n, dmin, dmax = conn.execute(
        'SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM market_amv_official'
    ).fetchone()
    conn.close()
    print(f'导入完成: {len(df)} 行写入, 表内共 {n} 行, {dmin} ~ {dmax}')


if __name__ == '__main__':
    main()
