"""Backfill market_regime_signals for ng2.0a (V11 + B1 + B2 + vote).

Runs once over 2018-2026 to populate the table with all historical regime states.

Usage:
    python3 scripts/build_regime_v2_history.py --start 2018-01-01 --end 2026-04-25
    python3 scripts/build_regime_v2_history.py --replace   # delete existing rows in date range first
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indicators.breadth import compute_breadth_signal
from indicators.realized_vol import compute_realized_vol_signal
from indicators.regime_classifier import compute_regime_v2

DB_PATH = Path(__file__).resolve().parents[1] / 'data_adapter' / 'stock_data.db'

HS300_CODE = '000300.SH'


def load_close_panel(conn, start: str, end: str) -> pd.DataFrame:
    """Load A-share close panel (excluding ETFs/indices)."""
    q = """
        SELECT s.code AS code, q.trade_date AS date, q.close AS close
        FROM daily_quotes q
        JOIN securities s ON s.id = q.security_id
        WHERE s.type = 'A股'
          AND q.trade_date BETWEEN ? AND ?
          AND q.close IS NOT NULL
    """
    df = pd.read_sql(q, conn, params=(start, end))
    if df.empty:
        raise RuntimeError(f'No A-share data between {start} and {end}')
    df['date'] = pd.to_datetime(df['date'])
    panel = df.pivot(index='date', columns='code', values='close').sort_index()
    return panel


def load_index_close(conn, code: str, start: str, end: str) -> pd.Series:
    q = """
        SELECT q.trade_date AS date, q.close AS close
        FROM daily_quotes q
        JOIN securities s ON s.id = q.security_id
        WHERE s.code = ?
          AND q.trade_date BETWEEN ? AND ?
        ORDER BY q.trade_date
    """
    df = pd.read_sql(q, conn, params=(code, start, end))
    if df.empty:
        raise RuntimeError(f'No data for {code} between {start} and {end}')
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date')['close'].astype(float)


def load_v11_regime(conn, start: str, end: str) -> pd.DataFrame:
    """Load existing V11 0AMV regime from market_amv table.

    Note: market_amv columns are var1, amv_ma60, amv_macd, amv_regime.
    We rename amv_ma60→ma60 and amv_macd→macd for downstream consumption.
    """
    q = """
        SELECT trade_date, var1, amv_ma60, amv_macd, amv_regime
        FROM market_amv
        WHERE trade_date BETWEEN ? AND ?
        ORDER BY trade_date
    """
    df = pd.read_sql(q, conn, params=(start, end))
    if df.empty:
        raise RuntimeError('market_amv is empty; run indicators/market_amv.py first')
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.set_index('trade_date').rename(columns={'amv_ma60': 'ma60', 'amv_macd': 'macd'})
    df['v11_bull'] = (df['amv_regime'] == 1).astype(int)
    grp = (df['amv_regime'] != df['amv_regime'].shift()).cumsum()
    df['v11_streak'] = (df['amv_regime'].groupby(grp).cumcount() + 1).astype(int)
    return df[['var1', 'ma60', 'macd', 'v11_bull', 'v11_streak']]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--start', default='2018-01-01')
    p.add_argument('--end', default='2026-04-25')
    p.add_argument('--replace', action='store_true', help='delete existing rows in date range first')
    args = p.parse_args()

    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.execute('PRAGMA busy_timeout=30000')

    try:
        print(f'[1/5] Load V11 regime from market_amv table ({args.start}..{args.end})')
        v11_df = load_v11_regime(conn, args.start, args.end)
        print(f'      V11 rows: {len(v11_df)}')

        print(f'[2/5] Load A-share close panel (need 60+ days lookback for B1)')
        # B1 needs 60d MA → fetch from 120 days before start to be safe
        lookback_start = (pd.Timestamp(args.start) - pd.Timedelta(days=120)).strftime('%Y-%m-%d')
        panel = load_close_panel(conn, lookback_start, args.end)
        print(f'      panel: {panel.shape[0]} dates × {panel.shape[1]} stocks')

        print(f'[3/5] Compute B1 breadth signal')
        b1 = compute_breadth_signal(panel, ma_short=20, ma_long=60, streak_days=3)
        b1 = b1.loc[b1.index >= pd.Timestamp(args.start)]
        print(f'      B1 rows: {len(b1)}, bull days: {int(b1["b1_bull"].fillna(0).sum())}')

        print(f'[4/5] Compute B2 realized vol signal on {HS300_CODE}')
        # B2 needs 60+252 = 312 days lookback
        b2_lookback_start = (pd.Timestamp(args.start) - pd.Timedelta(days=500)).strftime('%Y-%m-%d')
        idx_close = load_index_close(conn, HS300_CODE, b2_lookback_start, args.end)
        b2 = compute_realized_vol_signal(idx_close, rv_window=60, percentile_window=252, streak_days=3)
        b2 = b2.loc[b2.index >= pd.Timestamp(args.start)]
        print(f'      B2 rows: {len(b2)}, bull days: {int(b2["b2_bull"].fillna(0).sum())}')

        print(f'[5/5] Compute regime_v2 vote + write to DB')
        common = v11_df.index.intersection(b1.index).intersection(b2.index)
        v11_aligned = v11_df.loc[common]
        b1_aligned = b1.loc[common]
        b2_aligned = b2.loc[common]
        vote = compute_regime_v2(
            v11_aligned['v11_bull'].astype(int),
            b1_aligned['b1_bull'].fillna(0).astype(int),
            b2_aligned['b2_bull'].fillna(0).astype(int),
            system_streak=3,
        )

        merged = pd.concat([v11_aligned, b1_aligned, b2_aligned, vote], axis=1)
        merged.index.name = 'trade_date'
        merged = merged.reset_index()
        merged['trade_date'] = merged['trade_date'].dt.strftime('%Y-%m-%d')

        # future field — not computed in v1, column exists in schema for later
        merged['b1_adv_dec_ratio'] = None

        # DataFrame col → DB col mapping (order must match schema)
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

        if args.replace:
            conn.execute(
                'DELETE FROM market_regime_signals WHERE trade_date BETWEEN ? AND ?',
                (args.start, args.end),
            )
        out.to_sql('market_regime_signals', conn, if_exists='append', index=False)
        conn.commit()

        bull_n = int((out['regime_v2'] == 1).sum())
        bear_n = int((out['regime_v2'] == -1).sum())
        print(f'      written {len(out)} rows: bull={bull_n}, bear={bear_n}')
    finally:
        conn.close()


if __name__ == '__main__':
    main()
