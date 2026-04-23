#!/usr/bin/env python3
"""按每日 0AMV regime 把 ng1.0.7 / ng1.0.4 的历史报告拼装成 ng1.0.62 (MOE v2).

ng1.0.62 是运行时 router:
  regime=1 (牛市) → ng1.0.7
  regime=-1 (熊市) → ng1.0.4

对每个交易日查 market_amv 的当日 regime, 从对应子专家的 `daily_selection_*_fast/`
读取 analysis_data_YYYYMMDD.json, 写到 `daily_selection_ng106v2_fullmarket/`.

Usage:
    python3 scripts/merge_ng1062_historical_reports.py \\
        --start-date 2024-01-01 --end-date 2026-04-22
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / 'data_adapter' / 'stock_data.db'
BULL_DIR = PROJECT_ROOT / 'reports' / 'daily_selection_ng1.0.7_fast'
BEAR_DIR = PROJECT_ROOT / 'reports' / 'daily_selection_ng1.0.4_fast'
OUT_DIR = PROJECT_ROOT / 'reports' / 'daily_selection_ng106v2_fullmarket'


def load_regimes(start: str, end: str) -> dict[str, int]:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            'SELECT trade_date, amv_regime FROM market_amv '
            'WHERE trade_date >= ? AND trade_date <= ? ORDER BY trade_date',
            (start, end),
        ).fetchall()
    return {d: r for d, r in rows}


def merge(start: str, end: str, overwrite: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    regimes = load_regimes(start, end)
    print(f'regime rows: {len(regimes)} ({start} .. {end})')

    stats = {'bull': 0, 'bear': 0, 'missing_src': 0, 'skipped_existing': 0}
    for date, regime in regimes.items():
        date_str = date.replace('-', '')
        src_dir, label = (BULL_DIR, 'bull') if regime == 1 else (BEAR_DIR, 'bear')
        src = src_dir / f'analysis_data_{date_str}.json'
        dst = OUT_DIR / f'analysis_data_{date_str}.json'

        if dst.exists() and not overwrite:
            stats['skipped_existing'] += 1
            continue
        if not src.exists():
            stats['missing_src'] += 1
            print(f'  MISS {date} [{label}]: {src.name} not in {src_dir.name}')
            continue

        # Inject routing metadata so downstream readers know this is a merged MOE report.
        with open(src) as f:
            data = json.load(f)
        data['scoring_version'] = 'ng1.0.62'
        data['ng1062_router'] = {
            'trade_date': date,
            'amv_regime': regime,
            'sub_expert': 'ng1.0.7' if regime == 1 else 'ng1.0.4',
            'merged_from': src_dir.name,
        }
        with open(dst, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        stats[label] += 1

    print(f'bull days written: {stats["bull"]}')
    print(f'bear days written: {stats["bear"]}')
    print(f'skipped (existing): {stats["skipped_existing"]}')
    print(f'missing source:    {stats["missing_src"]}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--start-date', required=True, help='YYYY-MM-DD')
    ap.add_argument('--end-date', required=True, help='YYYY-MM-DD')
    ap.add_argument('--overwrite', action='store_true',
                    help='overwrite existing ng106v2 reports (default: skip)')
    args = ap.parse_args()
    merge(args.start_date, args.end_date, overwrite=args.overwrite)


if __name__ == '__main__':
    main()
