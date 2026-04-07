#!/usr/bin/env python3
"""
Multi-seed ensemble predictor for NG models.

Loads N model files, averages their predictions, generates report JSONs.
Use after training multiple seeds with:
  python3 ml_models/ng/ng_trainer.py --seed 42
  python3 ml_models/ng/ng_trainer.py --seed 123
  python3 ml_models/ng/ng_trainer.py --seed 456

Then generate ensemble reports:
  python3 scripts/ensemble_predict.py \
    --models ng_seed42.pkl ng_seed123.pkl ng_seed456.pkl \
    --start-date 2018-04-02 --end-date 2020-12-31 \
    --output-dir reports/daily_selection_ng102_ensemble_pre2020
"""

import sys
import os
import json
import time
import sqlite3
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

DB_PATH = os.path.join(PROJECT_ROOT, 'data_adapter', 'stock_data.db')


def load_scorer(model_path):
    """Load an NGProductionScorer with a specific model file."""
    from ml_models.ng.ng_production_scorer import NGProductionScorer
    return NGProductionScorer(model_path=model_path)


def ensemble_predict(scorers, stock_codes, date):
    """Average predictions from multiple scorers.
    Loads features once, then calls predict_scores_from_preloaded for each scorer."""
    # Load features once (all scorers use same cache table)
    features_df = scorers[0]._load_features(stock_codes, date)

    all_results = []
    for scorer in scorers:
        if features_df is not None and len(features_df) > 0:
            results = scorer.predict_scores_from_preloaded(stock_codes, date, features_df.copy())
        else:
            results = scorer.predict_scores(stock_codes, date)
        all_results.append(results)

    # Merge: average numeric fields across scorers
    merged = {}
    for code in stock_codes:
        preds = []
        for r in all_results:
            if code in r and r[code].get('exec_filter') != 'no_data':
                preds.append(r[code])

        if not preds:
            merged[code] = {
                'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0,
                'pred_15d': 0, 'rank_score': 0, 'recommendation': '观望',
                'exec_filter': 'no_data',
            }
            continue

        avg = {}
        for key in ['pred_3d', 'pred_5d', 'pred_10d', 'pred_15d', 'rank_score', 'score']:
            vals = [p.get(key, 0) for p in preds if p.get(key) is not None]
            avg[key] = float(np.mean(vals)) if vals else 0.0

        avg['recommendation'] = preds[0].get('recommendation', '观望')
        merged[code] = avg

    return merged


def get_trading_dates(start_date, end_date, cache_table='ng102_feature_cache'):
    """Get trading dates from feature cache."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        dates = [r[0] for r in conn.execute(f"""
            SELECT DISTINCT trade_date FROM {cache_table}
            WHERE trade_date >= ? AND trade_date <= ?
            ORDER BY trade_date
        """, (start_date, end_date)).fetchall()]
        return dates
    finally:
        conn.close()


def load_securities_info():
    """Load securities basic info from securities + stock_basic_info tables."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")
    info = {}
    try:
        # Primary: securities table (most complete, has industry for A-stocks)
        rows = conn.execute("""
            SELECT code, name, industry FROM securities WHERE industry IS NOT NULL AND industry != ''
        """).fetchall()
        for r in rows:
            code = r[0].split('.')[0] if '.' in r[0] else r[0]
            info[code] = {'name': r[1] or '', 'industry': r[2] or ''}
        # Fallback: stock_basic_info for any missing
        rows2 = conn.execute("""
            SELECT code, name, industry FROM stock_basic_info WHERE industry IS NOT NULL AND industry != ''
        """).fetchall()
        for r in rows2:
            code = r[0].split('.')[0] if '.' in r[0] else r[0]
            if code not in info:
                info[code] = {'name': r[1] or '', 'industry': r[2] or ''}
    except Exception:
        pass
    finally:
        conn.close()
    return info


def build_report_json(scored_stocks, date, securities_info):
    """Build analysis_data JSON from scored stocks."""
    stocks_list = []
    for code, data in scored_stocks.items():
        info = securities_info.get(code, {})
        entry = {
            'stock_code': code,
            'stock_name': info.get('name', ''),
            'industry': info.get('industry', ''),
            'score': data.get('score', 50.0),
            'pred_3d': data.get('pred_3d', 0),
            'pred_5d': data.get('pred_5d', 0),
            'pred_10d': data.get('pred_10d', 0),
            'pred_15d': data.get('pred_15d', 0),
            'rank_score': data.get('rank_score', 0),
            'analysis_date': date,
        }
        stocks_list.append(entry)

    # Sort by rank_score descending
    stocks_list.sort(key=lambda x: float(x.get('rank_score', 0) or 0), reverse=True)

    return {
        'analysis_date': date,
        'scoring_version': 'ng1.0.2_ensemble',
        'all_stocks_with_scores': stocks_list,
    }


def main():
    parser = argparse.ArgumentParser(description='Multi-seed ensemble predictor')
    parser.add_argument('--models', nargs='+', required=True,
                        help='Model .pkl file paths')
    parser.add_argument('--start-date', required=True)
    parser.add_argument('--end-date', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing reports')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load scorers
    print(f"Loading {len(args.models)} models...")
    scorers = []
    for mp in args.models:
        print(f"  Loading: {mp}")
        scorer = load_scorer(mp)
        scorers.append(scorer)
    print(f"  Ensemble of {len(scorers)} models ready\n")

    # Get trading dates
    # Use the cache table from the first scorer
    cache_table = scorers[0].cache_table
    dates = get_trading_dates(args.start_date, args.end_date, cache_table)
    print(f"Trading dates: {len(dates)} ({dates[0]} ~ {dates[-1]})")

    # Determine dates to generate
    dates_to_gen = []
    for date in dates:
        date_str = date.replace('-', '')
        json_file = output_dir / f'analysis_data_{date_str}.json'
        if json_file.exists() and not args.force:
            continue
        dates_to_gen.append(date)
    print(f"To generate: {len(dates_to_gen)} (skip {len(dates) - len(dates_to_gen)} existing)")

    # Load stock codes from cache for each date (use first scorer's method)
    securities_info = load_securities_info()

    # Get all stock codes from cache
    conn = sqlite3.connect(DB_PATH, timeout=30)
    all_codes_by_date = {}
    for date in dates_to_gen:
        codes = [r[0] for r in conn.execute(f"""
            SELECT DISTINCT code FROM {cache_table} WHERE trade_date = ?
        """, (date,)).fetchall()]
        all_codes_by_date[date] = codes
    conn.close()

    t0 = time.time()
    done = 0
    for i, date in enumerate(dates_to_gen):
        t_day = time.time()
        codes = all_codes_by_date[date]

        # Ensemble predict
        scored = ensemble_predict(scorers, codes, date)

        # Build and save JSON
        analysis = build_report_json(scored, date, securities_info)
        date_str = date.replace('-', '')
        json_file = output_dir / f'analysis_data_{date_str}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2, default=str)

        elapsed = time.time() - t_day
        done += 1
        eta = (time.time() - t0) / done * (len(dates_to_gen) - done) if done > 0 else 0
        n_scored = sum(1 for v in scored.values() if v.get('exec_filter') != 'no_data')
        print(f"  [{i+1}/{len(dates_to_gen)}] {date}: {n_scored} scored, "
              f"{elapsed:.1f}s (ETA: {eta:.0f}s)")

    total = time.time() - t0
    print(f"\nDone! {done} reports in {total:.1f}s ({total/60:.1f}min)")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    main()
