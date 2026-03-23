#!/usr/bin/env python3
"""
Phase 1: Pre-compute ML scores for all stocks across all dates.
Save to a pickle cache so eval_stop_target.py runs in <60 seconds.

Usage:
    python3 scripts/cache_scores_for_eval.py --version v4.8.1
"""
import sys
import os
import time
import json
import sqlite3
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / 'data_adapter' / 'stock_data.db'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default='v4.8.1')
    parser.add_argument('--start-date', default='2024-01-01')
    parser.add_argument('--top-n', type=int, default=20)
    parser.add_argument('--hold-days', type=int, default=10)
    args = parser.parse_args()

    from backtest.batch_generate_v395_reports import (
        get_trading_dates,
        fast_preload_feature_cache,
        load_securities_info,
        preload_daily_basic_bulk,
        score_all_stocks_from_preloaded,
    )
    from backtest.backtest_stop_target_direct import (
        load_scorer, preload_all_quotes,
        get_recommendation, get_risk_level,
    )

    t0 = time.time()

    # Get end date
    conn = sqlite3.connect(str(DB_PATH))
    row = conn.execute("SELECT MAX(trade_date) FROM v39_feature_cache").fetchone()
    conn.close()
    end_date = row[0] if row else '2026-03-15'

    print(f"[cache] Scoring {args.version} from {args.start_date} to {end_date}", flush=True)

    # Load scorer
    scorer = load_scorer(args.version)

    # Get dates & preload features
    dates = get_trading_dates(args.start_date, end_date, args.version)
    print(f"[cache] {len(dates)} trading dates", flush=True)

    features_cache = fast_preload_feature_cache(dates)
    daily_basic_cache = preload_daily_basic_bulk(dates)
    securities_info = load_securities_info()

    # Preload quotes
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    quotes_end = (end_dt + timedelta(days=args.hold_days + 30)).strftime('%Y-%m-%d')
    all_quotes = preload_all_quotes(args.start_date, quotes_end)

    # Build close price lookup
    raw_quotes = all_quotes.reset_index()
    close_by_date = {}
    for date_val, grp in raw_quotes.groupby('trade_date'):
        close_by_date[date_val] = dict(zip(grp['code'], grp['close']))

    t1 = time.time()
    print(f"[cache] Data loaded in {t1-t0:.1f}s", flush=True)

    # Score all dates
    cache_data = {
        'version': args.version,
        'start_date': args.start_date,
        'end_date': end_date,
        'top_n': args.top_n,
        'hold_days': args.hold_days,
        'dates': [],
        'all_quotes_index': all_quotes,  # store for trade simulation
    }

    daily_entries = []

    for i, date in enumerate(dates):
        features_df = features_cache.get(date)
        daily_basic_df = daily_basic_cache.get(date)
        if features_df is None or len(features_df) == 0:
            continue

        scored = score_all_stocks_from_preloaded(
            scorer, features_df, date, daily_basic_df, args.version
        )
        if not scored:
            continue

        close_prices = close_by_date.get(date, {})

        # Build per-stock entries with predictions + close
        stocks = []
        for code, data in scored.items():
            close = close_prices.get(code, 0)
            if close <= 0:
                continue

            pred_3d = data.get('pred_3d', 0)
            pred_5d = data.get('pred_5d', 0)
            pred_10d = data.get('pred_10d', 0)
            pred_15d = data.get('pred_15d', 0)

            rec = get_recommendation(pred_3d, pred_5d, pred_10d, pred_15d, scorer)
            risk = get_risk_level(pred_3d, pred_5d, pred_10d, pred_15d, scorer)

            stocks.append({
                'code': code,
                'name': securities_info.get(code, {}).get('name', ''),
                'close': close,
                'pred_3d': pred_3d,
                'pred_5d': pred_5d,
                'pred_10d': pred_10d,
                'pred_15d': pred_15d,
                'recommendation': rec,
                'risk_level': risk,
            })

        # Sort by pred_10d and take top-N
        stocks.sort(key=lambda x: x['pred_10d'], reverse=True)
        top_stocks = stocks[:args.top_n]

        daily_entries.append({
            'date': date,
            'stocks': top_stocks,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t1
            print(f"  [{i+1}/{len(dates)}] {len(daily_entries)} valid days, "
                  f"{elapsed:.0f}s elapsed", flush=True)

    cache_data['daily_entries'] = daily_entries
    cache_data['dates'] = [e['date'] for e in daily_entries]

    t2 = time.time()
    print(f"[cache] Scoring done: {len(daily_entries)} valid days, "
          f"{sum(len(e['stocks']) for e in daily_entries)} stock-days, "
          f"{t2-t1:.0f}s", flush=True)

    # Save cache
    cache_path = PROJECT_ROOT / 'scripts' / f'.score_cache_{args.version}.pkl'
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f, protocol=4)
    size_mb = cache_path.stat().st_size / 1024 / 1024
    print(f"[cache] Saved to {cache_path} ({size_mb:.1f} MB)", flush=True)
    print(f"[cache] Total time: {t2-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
