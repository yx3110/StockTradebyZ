#!/usr/bin/env python3
"""
ng1.3.0 Tier C EMT 4-Gate Validator.

Gates (per spec §5.4):
  Gate 1: 10d rank IC ≥ threshold (amihud/vwap ≥ 0.15; accel/tail ≥ 0.10)
  Gate 2: IC direction stability ≥ 52% (or ≤ 48% if consistently negative)
  Gate 3: Max |corr| with other Tier C candidates < 0.7
  Gate 4 (deferred): Gain importance > 1% (post-training audit)

Output: reports/ng130/emt_validation/tier_c_results.{md,json} + accepted_factors.json
"""
import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

from ml_models.ng.ng130_amount_factors import (
    compute_amihud_illiq_20d, compute_vwap_close_ratio_20d,
    compute_amount_acceleration_5d, compute_tail_beta_60d,
)

DB_PATH = str(PROJ_ROOT / 'data_adapter' / 'stock_data.db')

GATE_THRESHOLDS = {
    'amihud_illiq_20d': 0.15,
    'vwap_close_ratio_20d': 0.15,
    'amount_acceleration_5d': 0.10,
    'tail_beta_60d': 0.10,
}
IC_MIN_DIRECTION = 0.52
CORR_MAX = 0.7


def _load_stock_history(conn, security_id: int, start: str, end: str) -> pd.DataFrame:
    """Load daily data for one stock by security_id."""
    sql = """
    SELECT trade_date, close, amount, volume
    FROM daily_quotes
    WHERE security_id = ? AND trade_date BETWEEN ? AND ?
    ORDER BY trade_date
    """
    df = pd.read_sql(sql, conn, params=[security_id, start, end])
    return df


def _load_market_rets(conn, start: str, end: str) -> pd.Series:
    """Load 上证指数 daily returns. Start earlier than target to allow 60d warm-up."""
    sql = """
    SELECT trade_date, close FROM daily_quotes dq
    JOIN securities s ON dq.security_id = s.id
    WHERE s.code = '000001.SH' AND trade_date BETWEEN ? AND ?
    ORDER BY trade_date
    """
    df = pd.read_sql(sql, conn, params=[start, end])
    if df.empty:
        return pd.Series(dtype=float)
    df['ret'] = df['close'].pct_change()
    return df.set_index('trade_date')['ret'].dropna()


def _compute_factor_samples(
    code: str, df: pd.DataFrame, market_rets: pd.Series,
) -> list:
    """For each valid date, compute 4 factors + 10d forward ret."""
    if len(df) < 70:
        return []

    closes = df['close'].values.astype(np.float64)
    amounts = df['amount'].values.astype(np.float64)
    volumes = df['volume'].values.astype(np.float64)
    dates = df['trade_date'].tolist()

    samples = []
    for i in range(60, len(df) - 10):
        t = dates[i]
        c_hist = closes[:i+1]
        a_hist = amounts[:i+1]
        v_hist = volumes[:i+1]
        stock_rets = np.diff(c_hist) / (c_hist[:-1] + 1e-8)

        if t not in market_rets.index:
            continue
        try:
            m_idx = market_rets.index.get_loc(t)
        except KeyError:
            continue
        if m_idx < 60:
            continue
        m_hist = market_rets.iloc[m_idx-59:m_idx+1].values

        factors = {
            'amihud_illiq_20d': compute_amihud_illiq_20d(c_hist, a_hist),
            'vwap_close_ratio_20d': compute_vwap_close_ratio_20d(c_hist, a_hist, v_hist),
            'amount_acceleration_5d': compute_amount_acceleration_5d(a_hist),
            'tail_beta_60d': compute_tail_beta_60d(stock_rets, m_hist),
        }

        fwd_ret = closes[i+10] / closes[i] - 1.0
        samples.append({'code': code, 'trade_date': t, 'fwd_ret_10d': float(fwd_ret), **factors})

    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start-date', default='2020-01-01')
    ap.add_argument('--end-date', default='2024-12-31')
    ap.add_argument('--sample-size', type=int, default=50000)
    ap.add_argument('--max-codes', type=int, default=500,
                    help='Max stocks to sample from (each contributes many samples)')
    ap.add_argument('--output-dir', default='reports/ng130/emt_validation')
    args = ap.parse_args()

    out_dir = PROJ_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH, timeout=60)

    codes_df = pd.read_sql(
        "SELECT id, code FROM securities WHERE type='A股' ORDER BY code LIMIT ?",
        conn, params=[args.max_codes],
    )
    print(f'Sampling from {len(codes_df)} A-share codes (max_codes={args.max_codes})...')

    # Load market once (extend start by 90 trading days for warm-up)
    market_rets = _load_market_rets(conn, args.start_date, args.end_date)
    print(f'Market returns loaded: {len(market_rets)} days')

    all_samples = []
    for i, row in codes_df.iterrows():
        if len(all_samples) >= args.sample_size:
            break
        if i % 50 == 0:
            print(f'  [{i}/{len(codes_df)}] collected {len(all_samples)} samples')
        df = _load_stock_history(conn, row['id'], args.start_date, args.end_date)
        samples = _compute_factor_samples(row['code'], df, market_rets)
        all_samples.extend(samples)

    conn.close()

    if not all_samples:
        print('✗ No samples collected! Check data ranges.')
        sys.exit(1)

    df_all = pd.DataFrame(all_samples).dropna(subset=['fwd_ret_10d'])
    # For each factor, drop rows where that factor is NaN, but keep those rows for other factors
    print(f'\nTotal raw samples: {len(df_all)}')

    results = {}

    # Gate 1 (|IC|) + Gate 2 (direction) per factor
    for factor, threshold in GATE_THRESHOLDS.items():
        df_f = df_all.dropna(subset=[factor])
        if len(df_f) < 100:
            results[factor] = {
                'rank_ic': None, 'abs_ic': None, 'ic_direction': None,
                'samples_used': len(df_f),
                'gate_1_threshold': threshold,
                'gate_1_pass': False, 'gate_2_pass': False,
            }
            continue

        ic = df_f[factor].corr(df_f['fwd_ret_10d'], method='spearman')

        # Per-date IC for direction stability
        def _per_date_ic(g):
            if len(g) < 5:
                return np.nan
            return g[factor].corr(g['fwd_ret_10d'], method='spearman')

        by_date = df_f.groupby('trade_date').apply(_per_date_ic).dropna()
        if len(by_date) > 0:
            ic_direction = (by_date > 0).mean()
        else:
            ic_direction = 0.5

        results[factor] = {
            'rank_ic': float(ic) if not np.isnan(ic) else None,
            'abs_ic': float(abs(ic)) if not np.isnan(ic) else None,
            'ic_direction': float(ic_direction),
            'samples_used': len(df_f),
            'date_windows': len(by_date),
            'gate_1_threshold': threshold,
            'gate_1_pass': (abs(ic) >= threshold) if not np.isnan(ic) else False,
            'gate_2_pass': (ic_direction >= IC_MIN_DIRECTION) or (ic_direction <= (1 - IC_MIN_DIRECTION)),
        }

    # Gate 3: Pairwise correlation between 4 Tier C candidates
    factor_names = list(GATE_THRESHOLDS.keys())
    df_all_full = df_all.dropna(subset=factor_names)
    if len(df_all_full) >= 100:
        corr_matrix = df_all_full[factor_names].corr().abs()
        for factor in factor_names:
            others = [c for c in corr_matrix.columns if c != factor]
            max_corr = float(corr_matrix.loc[factor, others].max())
            results[factor]['max_corr_tierc'] = max_corr
            results[factor]['gate_3_pass_tierc'] = max_corr < CORR_MAX
    else:
        for factor in factor_names:
            results[factor]['max_corr_tierc'] = None
            results[factor]['gate_3_pass_tierc'] = False

    # Overall accepted: pass all 3 current gates
    for factor, r in results.items():
        r['accepted'] = r.get('gate_1_pass', False) and r.get('gate_2_pass', False) and r.get('gate_3_pass_tierc', False)

    # Sanitize numpy types for JSON serialization
    def _to_python(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    def _sanitize(d):
        return {k: _to_python(v) for k, v in d.items()}

    results_clean = {f: _sanitize(r) for f, r in results.items()}

    # Write JSON
    with open(out_dir / 'tier_c_results.json', 'w') as f:
        json.dump(results_clean, f, indent=2, ensure_ascii=False)

    # Write Markdown
    md_path = out_dir / 'tier_c_results.md'
    with open(md_path, 'w') as f:
        f.write(f'# ng1.3.0 Tier C EMT Validation — {datetime.now():%Y-%m-%d %H:%M}\n\n')
        f.write(f'**Samples** (raw): {len(df_all)}\n')
        f.write(f'**Window**: {args.start_date} → {args.end_date}\n')
        f.write(f'**Codes sampled**: {len(codes_df)}\n\n')
        f.write('| Factor | IC | |IC| | IC dir | N | Gate1 (|IC|) | Gate2 (dir) | Gate3 (corr) | Accept |\n')
        f.write('|---|---|---|---|---|---|---|---|---|\n')
        for factor in factor_names:
            r = results[factor]
            ic_str = f'{r["rank_ic"]:.4f}' if r.get('rank_ic') is not None else 'N/A'
            abs_ic_str = f'{r["abs_ic"]:.4f}' if r.get('abs_ic') is not None else 'N/A'
            dir_str = f'{r["ic_direction"]:.2%}' if r.get('ic_direction') is not None else 'N/A'
            g1 = '✓' if r.get('gate_1_pass') else '✗'
            g2 = '✓' if r.get('gate_2_pass') else '✗'
            g3 = '✓' if r.get('gate_3_pass_tierc') else '✗'
            acc = '**✓**' if r.get('accepted') else '✗'
            n = r.get('samples_used', 0)
            f.write(f'| {factor} | {ic_str} | {abs_ic_str} | {dir_str} | {n:,} | {g1} ({r.get("gate_1_threshold")}) | {g2} | {g3} | {acc} |\n')

    # Write accepted factors
    accepted_list = [f for f, r in results.items() if r.get('accepted')]
    with open(out_dir / 'accepted_factors.json', 'w') as f:
        json.dump(accepted_list, f, indent=2)

    print(f'\n✓ Reports: {md_path}')
    print(f'  accepted_factors.json: {accepted_list}')


if __name__ == '__main__':
    main()
