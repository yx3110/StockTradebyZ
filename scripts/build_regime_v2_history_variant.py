"""Backfill market_regime_signals_<variant> for a parameterized B1/B2/vote setting.

Difference from build_regime_v2_history.py:
  - Accepts B1/B2/vote params via CLI
  - Writes to a per-variant table name `market_regime_signals_{variant}` so the original
    `market_regime_signals` table (= V0 baseline) remains untouched.

Usage:
    python3 scripts/build_regime_v2_history_variant.py \\
        --variant strict_b1 \\
        --b1-lo 0.40 --b1-hi 0.65 \\
        --start 2018-01-01 --end 2026-04-25
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from indicators.breadth import compute_breadth_signal
from indicators.realized_vol import compute_realized_vol_signal
from indicators.regime_classifier import compute_regime_v2
from build_regime_v2_history import load_close_panel, load_index_close, load_v11_regime

DB_PATH = Path(__file__).resolve().parents[1] / 'data_adapter' / 'stock_data.db'
HS300_CODE = '000300.SH'


DDL_TEMPLATE = """
CREATE TABLE IF NOT EXISTS {table} (
    trade_date TEXT PRIMARY KEY,
    v11_var1 REAL,
    v11_ma60 REAL,
    v11_macd REAL,
    v11_bull INTEGER,
    v11_streak INTEGER,
    b1_pct_above_ma20 REAL,
    b1_pct_above_ma60 REAL,
    b1_adv_dec_ratio REAL,
    b1_score REAL,
    b1_bull INTEGER,
    b1_streak INTEGER,
    b2_rv_60d REAL,
    b2_rv_percentile_252 REAL,
    b2_bull INTEGER,
    b2_streak INTEGER,
    vote_count INTEGER,
    regime_v2_raw INTEGER,
    regime_v2_streak INTEGER,
    regime_v2 INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', required=True,
                   choices=['baseline', 'strict_b1', 'strict_b2', 'streak5', 'unanimous'])
    p.add_argument('--b1-lo', type=float, default=0.45)
    p.add_argument('--b1-hi', type=float, default=0.55)
    p.add_argument('--b2-lo', type=float, default=0.30)
    p.add_argument('--b2-hi', type=float, default=0.70)
    p.add_argument('--system-streak', type=int, default=3)
    p.add_argument('--vote-threshold', type=int, default=2)
    p.add_argument('--start', default='2018-01-01')
    p.add_argument('--end', default='2026-04-25')
    args = p.parse_args()

    table = f'market_regime_signals_{args.variant}'
    print(f'Variant: {args.variant} -> table {table}')
    print(f'  B1=({args.b1_lo}, {args.b1_hi}) B2=({args.b2_lo}, {args.b2_hi}) '
          f'streak={args.system_streak} vote_threshold={args.vote_threshold}')

    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.execute('PRAGMA busy_timeout=30000')
    conn.executescript(DDL_TEMPLATE.format(table=table))

    try:
        v11_df = load_v11_regime(conn, args.start, args.end)
        lookback_b1 = (pd.Timestamp(args.start) - pd.Timedelta(days=120)).strftime('%Y-%m-%d')
        panel = load_close_panel(conn, lookback_b1, args.end)
        b1 = compute_breadth_signal(
            panel, ma_short=20, ma_long=60,
            streak_days=3,  # B1's own streak fixed; system_streak applies at vote level
            hysteresis_lo=args.b1_lo, hysteresis_hi=args.b1_hi,
        )
        b1 = b1.loc[b1.index >= pd.Timestamp(args.start)]

        lookback_b2 = (pd.Timestamp(args.start) - pd.Timedelta(days=500)).strftime('%Y-%m-%d')
        idx_close = load_index_close(conn, HS300_CODE, lookback_b2, args.end)
        b2 = compute_realized_vol_signal(
            idx_close, rv_window=60, percentile_window=252, streak_days=3,
            hysteresis_lo=args.b2_lo, hysteresis_hi=args.b2_hi,
        )
        b2 = b2.loc[b2.index >= pd.Timestamp(args.start)]

        common = v11_df.index.intersection(b1.index).intersection(b2.index)
        v11_a = v11_df.loc[common]
        b1_a = b1.loc[common]
        b2_a = b2.loc[common]
        vote = compute_regime_v2(
            v11_a['v11_bull'].astype(int),
            b1_a['b1_bull'].fillna(0).astype(int),
            b2_a['b2_bull'].fillna(0).astype(int),
            system_streak=args.system_streak,
            vote_threshold=args.vote_threshold,
        )

        merged = pd.concat([v11_a, b1_a, b2_a, vote], axis=1)
        merged.index.name = 'trade_date'
        merged = merged.reset_index()
        merged['trade_date'] = merged['trade_date'].dt.strftime('%Y-%m-%d')
        merged['b1_adv_dec_ratio'] = None

        cols = [
            'trade_date',
            'var1', 'ma60', 'macd', 'v11_bull', 'v11_streak',
            'pct_above_ma20', 'pct_above_ma60', 'b1_adv_dec_ratio', 'b1_score', 'b1_bull', 'b1_streak',
            'rv_60d', 'rv_percentile_252', 'b2_bull', 'b2_streak',
            'vote_count', 'regime_v2_raw', 'regime_v2_streak', 'regime_v2',
        ]
        db_cols = [
            'trade_date',
            'v11_var1', 'v11_ma60', 'v11_macd', 'v11_bull', 'v11_streak',
            'b1_pct_above_ma20', 'b1_pct_above_ma60', 'b1_adv_dec_ratio', 'b1_score', 'b1_bull', 'b1_streak',
            'b2_rv_60d', 'b2_rv_percentile_252', 'b2_bull', 'b2_streak',
            'vote_count', 'regime_v2_raw', 'regime_v2_streak', 'regime_v2',
        ]
        out = merged[cols].rename(columns=dict(zip(cols, db_cols)))

        # Idempotent: delete then insert
        conn.execute(f'DELETE FROM {table}')
        out.to_sql(table, conn, if_exists='append', index=False)
        conn.commit()

        bull_n = int((out['regime_v2'] == 1).sum())
        bear_n = int((out['regime_v2'] == -1).sum())
        print(f'  written {len(out)} rows: bull={bull_n}, bear={bear_n}')

        # Pre-2020 vs WF-OOS bull breakdown for quick visibility
        pre = out[out['trade_date'] < '2020-01-01']
        wfo = out[out['trade_date'] >= '2020-01-01']
        print(f'  Pre-2020: bull={int((pre["regime_v2"] == 1).sum())}/{len(pre)}')
        print(f'  WF-OOS:   bull={int((wfo["regime_v2"] == 1).sum())}/{len(wfo)}')
    finally:
        conn.close()


if __name__ == '__main__':
    main()
