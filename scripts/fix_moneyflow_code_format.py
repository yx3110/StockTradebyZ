#!/usr/bin/env python3
"""
Fix moneyflow_daily.code format mismatch with ng cache.
Adds `code_6` column (6-digit code without .SZ/.SH suffix) + index.

Problem (from memory/data_pitfalls_2026_04_13.md):
  moneyflow_daily.code = '000001.SZ'
  ng101_feature_cache.code = '000001'
  → JOIN fails silently, factor values all NaN

Solution: Add derived column `code_6 = substr(code, 1, 6)` with its own index.
"""
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data_adapter', 'stock_data.db')


def main():
    with sqlite3.connect(DB_PATH, timeout=60) as conn:
        cur = conn.cursor()

        cur.execute('PRAGMA table_info(moneyflow_daily)')
        cols = [r[1] for r in cur.fetchall()]
        if 'code_6' in cols:
            print('✓ moneyflow_daily.code_6 already exists')
        else:
            print('Adding code_6 column...')
            cur.execute('ALTER TABLE moneyflow_daily ADD COLUMN code_6 VARCHAR(6)')
            print('  Populating code_6 = substr(code, 1, 6)...')
            cur.execute("UPDATE moneyflow_daily SET code_6 = substr(code, 1, 6)")
            conn.commit()
            print('  Done.')

        cur.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='moneyflow_daily'")
        idxs = [r[0] for r in cur.fetchall()]
        if 'idx_moneyflow_code_6_date' not in idxs:
            print('Creating idx_moneyflow_code_6_date...')
            cur.execute('CREATE INDEX idx_moneyflow_code_6_date ON moneyflow_daily(code_6, trade_date)')
            conn.commit()
            print('  Done.')
        else:
            print('✓ idx_moneyflow_code_6_date already exists')

        cur.execute('SELECT COUNT(*) FROM moneyflow_daily WHERE code_6 IS NULL')
        null_count = cur.fetchone()[0]
        cur.execute('SELECT COUNT(*) FROM moneyflow_daily')
        total = cur.fetchone()[0]
        print(f'\nVerification: {total:,} rows, {null_count:,} NULL code_6')
        assert null_count == 0, f'{null_count} rows have NULL code_6!'

        cur.execute("SELECT code, code_6 FROM moneyflow_daily LIMIT 3")
        for row in cur.fetchall():
            print(f'  code={row[0]} → code_6={row[1]}')


if __name__ == '__main__':
    main()
