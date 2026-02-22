#!/usr/bin/env python3
"""Targeted CCI_14 + ATR_14 backfill - much faster than full recalculation"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import argparse

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'data_adapter', 'stock_data.db')


def compute_cci_atr_for_stock(code: str, start_date: str, end_date: str) -> list:
    """Compute CCI_14 and ATR_14 for a single stock, return list of (cci, atr, code, date) tuples"""
    conn = sqlite3.connect(DB_PATH)
    try:
        # Load enough history before start_date for 14-day lookback
        df = pd.read_sql_query("""
            SELECT dq.trade_date, dq.high, dq.low, dq.close
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE s.code = ? AND dq.trade_date >= date(?, '-30 days') AND dq.trade_date <= ?
            ORDER BY dq.trade_date
        """, conn, params=(code, start_date, end_date))
        conn.close()

        if len(df) < 15:
            return []

        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        close = df['close'].values.astype(float)
        dates = df['trade_date'].values

        results = []

        # Compute CCI_14
        typical_price = (high + low + close) / 3.0
        tp_series = pd.Series(typical_price)
        tp_sma = tp_series.rolling(window=14, min_periods=14).mean().values
        tp_mad = tp_series.rolling(window=14, min_periods=14).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        ).values

        # Compute ATR_14
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
        atr_series = pd.Series(tr).ewm(span=14, adjust=False).mean().values

        for i in range(len(df)):
            d = str(dates[i])
            if d < start_date:
                continue

            cci = None
            if not np.isnan(tp_sma[i]) and tp_mad[i] != 0 and not np.isnan(tp_mad[i]):
                cci = float((typical_price[i] - tp_sma[i]) / (0.015 * tp_mad[i]))

            atr = float(atr_series[i]) if not np.isnan(atr_series[i]) else None

            if cci is not None or atr is not None:
                results.append((cci, atr, code, d))

        return results

    except Exception as e:
        try:
            conn.close()
        except:
            pass
        return []


def main():
    parser = argparse.ArgumentParser(description='Backfill CCI_14 and ATR_14 only')
    parser.add_argument('--start-date', default='2024-01-01')
    parser.add_argument('--end-date', default='2026-02-22')
    parser.add_argument('--max-workers', type=int, default=8)
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)

    # Get all stock codes that have technical_indicators data
    codes = [row[0] for row in conn.execute("""
        SELECT DISTINCT s.code FROM technical_indicators ti
        JOIN securities s ON ti.security_id = s.id
        WHERE ti.trade_date >= ? AND ti.trade_date <= ?
    """, (args.start_date, args.end_date)).fetchall()]
    conn.close()

    print(f"Backfilling CCI_14 + ATR_14 for {len(codes)} stocks ({args.start_date} to {args.end_date})")

    t0 = time.time()
    all_updates = []
    completed = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(compute_cci_atr_for_stock, code, args.start_date, args.end_date): code
                   for code in codes}

        for future in as_completed(futures):
            completed += 1
            results = future.result()
            all_updates.extend(results)
            if completed % 500 == 0:
                print(f"  Computed {completed}/{len(codes)} stocks, {len(all_updates)} updates so far...")

    print(f"Computation done: {len(all_updates)} updates in {time.time()-t0:.1f}s")

    # Batch UPDATE
    if all_updates:
        print("Writing to database...")
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA journal_mode=WAL")

        batch_size = 5000
        for i in range(0, len(all_updates), batch_size):
            batch = all_updates[i:i+batch_size]
            conn.executemany("""
                UPDATE technical_indicators
                SET cci_14 = ?, atr_14 = ?
                WHERE security_id = (SELECT id FROM securities WHERE code = ?)
                AND trade_date = ?
            """, batch)
            conn.commit()
            if (i // batch_size) % 20 == 0:
                print(f"  Written {min(i+batch_size, len(all_updates))}/{len(all_updates)} updates...")

        conn.close()

    elapsed = time.time() - t0
    print(f"Done! {len(all_updates)} rows updated in {elapsed:.1f}s")


if __name__ == '__main__':
    main()
