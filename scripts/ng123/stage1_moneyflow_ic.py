#!/usr/bin/env python3
"""ng1.2.3 Stage 1: Moneyflow factor IC validation + orthogonality vs ng101.

Per spec §6.1 — Pass criteria (ALL):
  - >= 6 factors with |IC| > 0.02 AND |ICIR| > 0.3
  - >= 4 factors with |IC| > 0.04 AND |ICIR| > 0.5
  - >= 8 factors with max |corr| < 0.5 vs all 64 ng101 features
  - >= 2 of 4 cs_rank factors pass

Output: reports/ng123/fastcheck/stage1_moneyflow_ic.csv
        reports/ng123/fastcheck/stage1_status.json

Estimated runtime: ~4 hours on full universe × 2022-01..2026-04.
"""
import argparse
import json
import pickle
import sys
import sqlite3
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml_models.ng.ng123_moneyflow_factors import (
    compute_group_a_factors,
    compute_group_b_factors,
    compute_group_c_factors,
)

DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')
OUTPUT_DIR = PROJECT_ROOT / 'reports' / 'ng123' / 'fastcheck'
CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoint'

# Pass thresholds (spec §6.1)
MIN_IC = 0.02
MIN_ICIR = 0.3
STRONG_IC = 0.04
STRONG_ICIR = 0.5
MAX_CORR_VS_NG101 = 0.5


def _strip_suffix(code: str) -> str:
    """Normalize '000001.SZ' → '000001'; leave '000001' unchanged."""
    return code.split('.')[0] if '.' in code else code


def load_moneyflow_per_stock(start_date: str, end_date: str, n_stocks: int = None,
                              seed: int = 42) -> Dict[str, List[Dict]]:
    """Load moneyflow rows per stock for the date range, returning chronological lists.

    Returns dict keyed by SUFFIXED code (e.g. '000001.SZ') because the factor
    functions only need raw rows. Code normalization for merging happens later.
    """
    print(f"  Loading moneyflow {start_date} → {end_date}...", flush=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.row_factory = sqlite3.Row

    # A-stocks with moneyflow coverage in the date range
    codes_q = """
        SELECT DISTINCT mf.code
        FROM moneyflow_daily mf
        JOIN securities s ON s.code = substr(mf.code, 1, 6)
        WHERE s.type = 'A股'
          AND mf.trade_date BETWEEN ? AND ?
    """
    codes = [r[0] for r in conn.execute(codes_q, (start_date, end_date)).fetchall()]
    if n_stocks and len(codes) > n_stocks:
        codes = list(np.random.RandomState(seed).choice(codes, n_stocks, replace=False))
    print(f"  Universe: {len(codes)} stocks", flush=True)

    # Bulk load in chunks
    chunk = 800
    per_stock: Dict[str, List[Dict]] = defaultdict(list)
    for i in range(0, len(codes), chunk):
        batch = codes[i:i + chunk]
        rows = conn.execute(
            f"""SELECT code, trade_date,
                       buy_sm_amount, sell_sm_amount,
                       buy_md_amount, sell_md_amount,
                       buy_lg_amount, sell_lg_amount,
                       buy_elg_amount, sell_elg_amount,
                       net_mf_amount
                FROM moneyflow_daily
                WHERE trade_date BETWEEN ? AND ?
                  AND code IN ({','.join('?' * len(batch))})
                ORDER BY code, trade_date""",
            [start_date, end_date] + batch
        ).fetchall()
        for r in rows:
            per_stock[r['code']].append(dict(r))

    conn.close()
    print(f"  Loaded {sum(len(v) for v in per_stock.values()):,} total moneyflow rows",
          flush=True)
    return dict(per_stock)


def load_label_10d(start_date: str, end_date: str) -> pd.DataFrame:
    """Load 10d industry-excess label from ng101_feature_cache.

    Returns DataFrame with columns: code_norm (6-digit), trade_date, label_10d.
    """
    print("  Loading label_10d from ng101_feature_cache...", flush=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout = 30000")
    df = pd.read_sql(
        """SELECT code, trade_date, label_10d
           FROM ng101_feature_cache
           WHERE trade_date BETWEEN ? AND ?
             AND label_10d IS NOT NULL""",
        conn, params=[start_date, end_date])
    conn.close()
    # ng101 stores 6-digit codes; normalize defensively
    df['code_norm'] = df['code'].apply(_strip_suffix)
    print(f"  Labels: {len(df):,} rows, "
          f"sample code format: {df['code'].iloc[0] if len(df) else 'N/A'}", flush=True)
    return df


def load_ng101_features(start_date: str, end_date: str) -> pd.DataFrame:
    """Load ng101 features_json for orthogonality check (capped at 100k rows for speed)."""
    print("  Loading ng101 features (sample up to 100k rows)...", flush=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout = 30000")
    rows = conn.execute(
        """SELECT code, trade_date, features_json
           FROM ng101_feature_cache
           WHERE trade_date BETWEEN ? AND ?
             AND features_json IS NOT NULL
           LIMIT 100000""",
        (start_date, end_date)).fetchall()
    conn.close()

    parsed = []
    for code, td, fjson in rows:
        try:
            d = json.loads(fjson)
            d['code_norm'] = _strip_suffix(code)
            d['trade_date'] = td
            parsed.append(d)
        except (json.JSONDecodeError, TypeError):
            continue
    df = pd.DataFrame(parsed)
    print(f"  ng101 features: {len(df):,} rows, {len(df.columns)} columns", flush=True)
    return df


def load_industry_map() -> Dict[str, str]:
    """Load code → industry mapping from securities table (6-digit codes)."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout = 30000")
    rows = conn.execute(
        "SELECT code, industry FROM securities WHERE industry IS NOT NULL AND industry != ''"
    ).fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}


def compute_factors_for_universe(
    mf_per_stock: Dict[str, List[Dict]],
    universe: List[str],
) -> pd.DataFrame:
    """Compute Groups A+B+C (8 factors) and scalars per (code, date).

    cs_rank (Group D) is computed separately after all stocks are processed,
    so peer arrays can be built per (industry, date).

    Returns DataFrame with columns: code_norm (6-digit), trade_date, 8 factor cols,
    4 scalar cols (prefixed '_scalar_').
    """
    print(f"  Computing Groups A+B+C (8 factors) for {len(universe)} stocks...",
          flush=True)
    records = []
    for i, code in enumerate(universe):
        stock_rows = mf_per_stock.get(code, [])
        if len(stock_rows) < 5:
            continue
        code_norm = _strip_suffix(code)
        for j in range(5, len(stock_rows)):
            window = stock_rows[max(0, j - 19):j + 1]  # up to 20-day rolling window
            a = compute_group_a_factors(window)
            b = compute_group_b_factors(window)
            factors: Dict[str, object] = {}
            factors.update(a)
            factors.update(b)
            factors.update(compute_group_c_factors(window))
            # Extract scalars from already-computed Group A/B results (avoids double-call)
            factors['_scalar_net_elg_5d_ratio']    = a['mf_net_elg_5d_ratio']
            factors['_scalar_net_elg_20d_ratio']   = a['mf_net_elg_20d_ratio']
            factors['_scalar_smart_net_share_20d'] = a['mf_smart_net_share_20d']
            factors['_scalar_persistence_20d']     = b['mf_elg_persistence_20d']
            factors['code_norm'] = code_norm
            factors['trade_date'] = stock_rows[j]['trade_date']
            records.append(factors)
        if i % 200 == 0:
            print(f"    progress: {i}/{len(universe)} stocks", flush=True)

    df = pd.DataFrame(records)
    print(f"  Raw factor rows: {len(df):,}", flush=True)
    return df


def add_cs_rank_factors(df_factors: pd.DataFrame,
                        industry_map: Dict[str, str]) -> pd.DataFrame:
    """Compute cs_rank Group D factors via pandas groupby rank.

    Adds 4 cs_rank columns; drops the _scalar_ helper columns and 'industry'.
    Mutates df_factors in-place (caller owns the freshly-built DataFrame).
    """
    print("  Computing cs_rank Group D factors via groupby...", flush=True)
    df_factors['industry'] = df_factors['code_norm'].map(industry_map).fillna('UNKNOWN')

    scalar_to_rank = [
        ('_scalar_net_elg_5d_ratio',    'cs_rank_mf_net_elg_5d'),
        ('_scalar_net_elg_20d_ratio',   'cs_rank_mf_net_elg_20d'),
        ('_scalar_smart_net_share_20d', 'cs_rank_mf_smart_net_share_20d'),
        ('_scalar_persistence_20d',     'cs_rank_mf_elg_persistence_20d'),
    ]
    for scalar_col, rank_col in scalar_to_rank:
        if scalar_col in df_factors.columns:
            df_factors[rank_col] = (
                df_factors
                .groupby(['industry', 'trade_date'])[scalar_col]
                .rank(pct=True)
            )

    scalar_cols = [s for s, _ in scalar_to_rank if s in df_factors.columns]
    df_factors = df_factors.drop(columns=scalar_cols + ['industry'])
    return df_factors


def compute_ic_per_factor(df_factors: pd.DataFrame, df_labels: pd.DataFrame,
                           min_stocks_per_day: int = 100) -> pd.DataFrame:
    """Compute Spearman IC per factor, daily cross-section, then aggregate to mean+ICIR.

    Args:
        min_stocks_per_day: Minimum stocks per day to include that day in IC computation.
            Default 100 for full-universe runs. Set to 30 for smoke-tests with ~200 stocks.
    """
    print(f"  Computing IC per factor (Spearman, daily cross-section, "
          f"min_stocks_per_day={min_stocks_per_day})...", flush=True)

    # Both df_factors and df_labels use 'code_norm' + 'trade_date' as keys
    merged = df_factors.merge(df_labels[['code_norm', 'trade_date', 'label_10d']],
                               on=['code_norm', 'trade_date'], how='inner')
    print(f"    Merged rows: {len(merged):,}", flush=True)

    factor_cols = [
        c for c in df_factors.columns
        if c not in ('code_norm', 'trade_date')
        and not c.startswith('_scalar_')
    ]
    print(f"    Factors to evaluate: {factor_cols}", flush=True)

    # Pre-group once; reused across all factor columns
    grouped = merged.groupby('trade_date')

    results = []
    for fc in factor_cols:
        ics = []
        for date, grp in grouped:
            sub = grp[[fc, 'label_10d']].dropna()
            if len(sub) < min_stocks_per_day:
                continue
            try:
                ic_val, _ = spearmanr(sub[fc].values, sub['label_10d'].values)
                if np.isfinite(ic_val):
                    ics.append(ic_val)
            except Exception:
                pass
        if len(ics) < 20:
            results.append({
                'factor': fc, 'n_days': len(ics),
                'ic_mean': np.nan, 'ic_std': np.nan, 'icir': np.nan,
                'ic_positive_pct': np.nan,
            })
            continue
        ic_arr = np.array(ics)
        ic_mean = float(np.mean(ic_arr))
        ic_std = float(np.std(ic_arr))
        icir = ic_mean / max(ic_std, 1e-8)
        results.append({
            'factor': fc,
            'n_days': len(ics),
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'icir': icir,
            'ic_positive_pct': float(np.mean(ic_arr > 0)),
        })
    return pd.DataFrame(results)


def compute_max_corr_vs_ng101(df_factors: pd.DataFrame, df_ng101: pd.DataFrame) -> pd.DataFrame:
    """For each new factor, compute max |Pearson corr| vs all ng101 feature columns."""
    print("  Computing max |corr| vs ng101 features...", flush=True)

    # df_ng101 has 'code_norm' + 'trade_date'; df_factors same
    merged = df_factors.merge(df_ng101, on=['code_norm', 'trade_date'], how='inner')
    print(f"    Merged for corr: {len(merged):,} rows", flush=True)

    if len(merged) < 500:
        print("    WARN: too few rows for corr check; skipping", flush=True)
        return pd.DataFrame(columns=['factor', 'max_abs_corr', 'worst_ng101_feature'])

    factor_cols = [
        c for c in df_factors.columns
        if c not in ('code_norm', 'trade_date')
        and not c.startswith('_scalar_')
    ]
    # ng101 feature columns: exclude key cols and the new factor cols
    ng101_cols = [
        c for c in df_ng101.columns
        if c not in ('code_norm', 'trade_date')
        and c not in factor_cols
        and not c.startswith('label_')
    ]
    print(f"    ng101 features for corr: {len(ng101_cols)}", flush=True)

    rows = []
    for fc in factor_cols:
        max_abs_corr = 0.0
        worst_pair = ''
        for nc in ng101_cols:
            sub = merged[[fc, nc]].dropna()
            if len(sub) < 100:
                continue
            try:
                c = sub[fc].corr(sub[nc])
                if pd.notna(c) and abs(c) > max_abs_corr:
                    max_abs_corr = abs(c)
                    worst_pair = nc
            except Exception:
                continue
        rows.append({
            'factor': fc,
            'max_abs_corr': max_abs_corr,
            'worst_ng101_feature': worst_pair,
        })
    return pd.DataFrame(rows)


def evaluate_pass_criteria(ic_df: pd.DataFrame,
                            corr_df: pd.DataFrame) -> Dict[str, object]:
    """Apply spec §6.1 pass criteria. Returns per-criterion bool + count dict."""
    # Drop rows with NaN IC (too few days)
    valid_ic = ic_df.dropna(subset=['ic_mean', 'icir'])

    n_pass_basic = int(((valid_ic['ic_mean'].abs() > MIN_IC)
                        & (valid_ic['icir'].abs() > MIN_ICIR)).sum())
    n_pass_strong = int(((valid_ic['ic_mean'].abs() > STRONG_IC)
                         & (valid_ic['icir'].abs() > STRONG_ICIR)).sum())

    if len(corr_df) > 0:
        n_pass_corr = int((corr_df['max_abs_corr'] < MAX_CORR_VS_NG101).sum())
    else:
        n_pass_corr = 0

    # cs_rank factors: those starting with 'cs_rank_mf'
    cs_rank_rows = valid_ic[valid_ic['factor'].str.startswith('cs_rank_mf')]
    n_cs_pass = int(((cs_rank_rows['ic_mean'].abs() > MIN_IC)
                     & (cs_rank_rows['icir'].abs() > MIN_ICIR)).sum())

    return {
        'criterion_1_basic_count': n_pass_basic,
        'criterion_1_basic_pass': n_pass_basic >= 6,
        'criterion_2_strong_count': n_pass_strong,
        'criterion_2_strong_pass': n_pass_strong >= 4,
        'criterion_3_corr_count': n_pass_corr,
        'criterion_3_corr_pass': n_pass_corr >= 8,
        'criterion_4_csrank_count': n_cs_pass,
        'criterion_4_csrank_pass': n_cs_pass >= 2,
    }


def main():
    p = argparse.ArgumentParser(description='ng1.2.3 Stage 1 moneyflow IC validation')
    p.add_argument('--start-date', default='2022-01-01',
                   help='Start date for IC/corr computation (default: 2022-01-01)')
    p.add_argument('--end-date', default=None,
                   help='End date (default: today)')
    p.add_argument('--n-stocks', type=int, default=None,
                   help='Sample n stocks randomly (default: full universe; use 50 for smoke-test)')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed for stock sampling (default: 42)')
    p.add_argument('--min-stocks-per-day', type=int, default=100,
                   help='Minimum stocks per day for IC computation (default: 100 for full run, '
                        'use 30 for smoke-tests with --n-stocks ~200)')
    p.add_argument('--no-checkpoint', action='store_true',
                   help='Disable checkpoint/resume — always recompute df_factors')
    args = p.parse_args()

    if args.end_date is None:
        args.end_date = date.today().strftime('%Y-%m-%d')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Stage 1: Moneyflow IC Validation ===")
    print(f"  Date range: {args.start_date} → {args.end_date}")
    print(f"  n_stocks: {args.n_stocks or 'full universe'}")
    print(f"  min_stocks_per_day: {args.min_stocks_per_day}")
    print()

    mf_per_stock = load_moneyflow_per_stock(
        args.start_date, args.end_date, args.n_stocks, args.seed)
    df_labels = load_label_10d(args.start_date, args.end_date)
    df_ng101 = load_ng101_features(args.start_date, args.end_date)
    industry_map = load_industry_map()
    print(f"  Industry map: {len(industry_map):,} entries", flush=True)

    universe = list(mf_per_stock.keys())

    checkpoint_key = (
        f"factors_{args.start_date}_{args.end_date}_"
        f"{args.n_stocks or 'full'}.pkl"
    )
    checkpoint_path = CHECKPOINT_DIR / checkpoint_key

    if not args.no_checkpoint and checkpoint_path.exists():
        print(f"  Loading factor cache from {checkpoint_path} "
              f"(skip ~3.5h computation)", flush=True)
        with open(checkpoint_path, 'rb') as fh:
            df_factors = pickle.load(fh)
    else:
        df_raw = compute_factors_for_universe(mf_per_stock, universe)

        if len(df_raw) == 0:
            print("ERROR: No factor rows computed — check moneyflow data availability!")
            sys.exit(1)

        df_factors = add_cs_rank_factors(df_raw, industry_map)

        if not args.no_checkpoint:
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, 'wb') as fh:
                pickle.dump(df_factors, fh, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  Saved factor checkpoint: {checkpoint_path}", flush=True)

    factor_cols = [c for c in df_factors.columns if c not in ('code_norm', 'trade_date')]
    print(f"\n  Factor columns ({len(factor_cols)}): {factor_cols}")

    universe_norms = set(df_factors['code_norm'].unique())
    df_labels_filtered = df_labels[df_labels['code_norm'].isin(universe_norms)]
    print(f"\n  Labels after universe filter: {len(df_labels_filtered):,} rows", flush=True)

    ic_df = compute_ic_per_factor(df_factors, df_labels_filtered,
                                   min_stocks_per_day=args.min_stocks_per_day)
    print("\n=== Factor IC Summary ===")
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 120)
    print(ic_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    df_ng101_filtered = df_ng101[df_ng101['code_norm'].isin(universe_norms)]
    corr_df = compute_max_corr_vs_ng101(df_factors, df_ng101_filtered)
    if len(corr_df) > 0:
        print("\n=== Factor Orthogonality (max |corr| vs ng101) ===")
        print(corr_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    criteria = evaluate_pass_criteria(ic_df, corr_df)
    print("\n=== Pass Criteria (spec §6.1) ===")
    for k, v in criteria.items():
        print(f"  {k}: {v}")

    overall_pass = all(v for k, v in criteria.items() if k.endswith('_pass'))
    verdict = 'PASS' if overall_pass else 'FAIL'
    print(f"\nSTAGE 1 OVERALL: {verdict}")

    out_csv = OUTPUT_DIR / 'stage1_moneyflow_ic.csv'
    if len(corr_df) > 0:
        combined = ic_df.merge(corr_df, on='factor', how='outer')
    else:
        combined = ic_df.copy()
        combined['max_abs_corr'] = np.nan
        combined['worst_ng101_feature'] = ''
    combined.to_csv(out_csv, index=False)
    print(f"\nSaved CSV: {out_csv}")

    status = {
        'stage': 1,
        'overall_pass': overall_pass,
        'verdict': verdict,
        'criteria': criteria,
        'n_factors': int(len(ic_df)),
        'n_stocks_used': int(df_factors['code_norm'].nunique()),
        'n_factor_rows': int(len(df_factors)),
        'config': {
            'start_date': args.start_date,
            'end_date': args.end_date,
            'n_stocks': args.n_stocks,
            'min_stocks_per_day': args.min_stocks_per_day,
        },
    }
    status_path = OUTPUT_DIR / 'stage1_status.json'
    with open(status_path, 'w') as f:
        json.dump(status, f, indent=2)
    print(f"Saved status JSON: {status_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
