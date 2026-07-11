"""回填上证指数+深证成指的历史amount数据到daily_quotes表"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import sqlite3
import tushare as ts
import pandas as pd
import time


def backfill_index_amount(start_date='20180101', end_date=None):
    # token 优先从 core.config/.env 获取
    try:
        from core.config import get_tushare_token
        token = get_tushare_token()
    except ImportError:
        with open('config.json') as f:
            token = json.load(f)['tushare']['token']
    pro = ts.pro_api(token)

    if end_date is None:
        end_date = pd.Timestamp.now().strftime('%Y%m%d')

    indices = {
        '000001.SH': '上证指数',
        '399001.SZ': '深证成指',
    }

    conn = sqlite3.connect('data_adapter/stock_data.db', timeout=30)
    try:
        sec_map = {}
        for code in indices:
            row = conn.execute(
                'SELECT id FROM securities WHERE code = ?', (code,)
            ).fetchone()
            if row:
                sec_map[code] = row[0]
            else:
                print(f'WARNING: {code} not in securities table, skip')

        for ts_code, name in indices.items():
            if ts_code not in sec_map:
                continue
            sid = sec_map[ts_code]
            print(f'\n回填 {name} ({ts_code}) amount...')

            df = pro.index_daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,trade_date,amount'
            )
            if df.empty:
                print(f'  无数据')
                continue

            print(f'  获取 {len(df)} 天数据')

            updated = 0
            for _, row in df.iterrows():
                trade_date = pd.to_datetime(
                    row['trade_date'], format='%Y%m%d'
                ).strftime('%Y-%m-%d')
                amount = row.get('amount')
                if pd.isna(amount):
                    continue
                conn.execute(
                    'UPDATE daily_quotes SET amount = ? '
                    'WHERE security_id = ? AND trade_date = ?',
                    (amount, sid, trade_date)
                )
                updated += 1

            conn.commit()
            print(f'  更新 {updated} 条')
            time.sleep(0.5)

        print('\n=== 验证 ===')
        for ts_code, name in indices.items():
            if ts_code not in sec_map:
                continue
            r = conn.execute('''
                SELECT MIN(trade_date), MAX(trade_date), COUNT(*)
                FROM daily_quotes
                WHERE security_id = ? AND amount IS NOT NULL AND amount > 0
            ''', (sec_map[ts_code],)).fetchone()
            print(f'{name}: {r[0]} ~ {r[1]}, {r[2]}条')
    finally:
        conn.close()
    print('\n完成')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='回填指数amount历史数据')
    parser.add_argument('--start-date', default='20180101')
    parser.add_argument('--end-date', default=None)
    args = parser.parse_args()
    backfill_index_amount(args.start_date, args.end_date)
