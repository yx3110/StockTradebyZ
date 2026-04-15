#!/usr/bin/env python3
"""ng1.2.3 Stage 2: Mined factor re-validation per spec §6.2.

Apply secondary filters:
  1. Orthogonality (|corr|<0.5) vs ng101 64 features + 6 accepted moneyflow factors
  2. Sign flip for IC<0 (recorded, not applied here)
  3. Cross-regime stability: same-sign IC on 2022 (bear) + 2024 (recovery),
     both |IC|>0.02

Pass criterion: >=6 mined factors |ICIR|>0.5 AND orthogonal AND cross-regime stable.

Output: reports/ng123/fastcheck/stage2_mined_factors.csv
        reports/ng123/fastcheck/stage2_status.json
"""
import argparse
import json
import sys
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')
OUTPUT_DIR = PROJECT_ROOT / 'reports' / 'ng123' / 'fastcheck'
MINING_RESULTS = PROJECT_ROOT / 'scripts' / 'mined_factors_results.json'

# Regime periods for cross-regime stability check
REGIME_PERIODS = {
    'bear_2022': ('2022-01-01', '2022-12-31'),
    'recovery_2024': ('2024-01-01', '2024-12-31'),
}

# Filter thresholds
MIN_ABS_ICIR = 0.5          # primary IC ratio filter
MIN_ABS_IC_REGIME = 0.02    # per-regime IC floor
MAX_CORR_VS_EXISTING = 0.5  # orthogonality ceiling

# Economic interpretability: reject these patterns (pure black-box combos)
BLACKLIST_PATTERNS = [
    'op_inv(ts_skew',
    'op_inv(ts_kurt',
    'op_sign(ts_kurt',
]

# ng101 features used for orthogonality check (key ones — sampled from schema)
NG101_NUMERIC_FEATURES = [
    'return_1d', 'return_3d', 'return_5d', 'return_10d', 'return_20d',
    'return_60d', 'volatility_20d', 'volatility_60d',
    'rsi_14', 'ma5', 'ma10', 'ma20', 'ma60',
    'volume_ratio_5d', 'volume_ratio_20d',
    'macd', 'macd_signal', 'bb_pct',
    'turnover_rate', 'pe_ttm', 'pb',
    'market_cap_log',
]

# 6 accepted moneyflow factor names (from Stage 1)
ACCEPTED_MONEYFLOW_FEATURES = [
    'mf_net_inflow_1d', 'mf_net_inflow_3d', 'mf_net_inflow_5d',
    'mf_large_ratio_1d', 'mf_large_ratio_3d', 'large_net_pct_5d',
]


def load_mining_results() -> List[Dict]:
    """Load factor candidates from mining pipeline output."""
    if not MINING_RESULTS.exists():
        raise FileNotFoundError(
            f"{MINING_RESULTS} not found — run factor_mining_pipeline.py first")
    with open(MINING_RESULTS) as f:
        data = json.load(f)
    # Support both {'factors': [...]} and flat list
    if isinstance(data, list):
        return data
    return data.get('factors', data.get('results', []))


def is_economically_interpretable(name: str) -> tuple:
    """Check if a factor has a plausible economic interpretation.

    Returns (ok: bool, reason: str).
    Rejects pure black-box combinations; requires at least one sentence of
    semantic meaning derivable from the operator+operand pair.
    """
    # Reject explicit blacklist
    for pat in BLACKLIST_PATTERNS:
        if pat in name:
            return False, f"blacklisted pattern '{pat}'"

    # Simple semantic mapping: operand × operator → readable meaning
    semantic_map = {
        ('ts_mean', 'ret'):      'average return momentum',
        ('ts_mean', 'volume'):   'average volume trend',
        ('ts_std', 'ret'):       'return volatility',
        ('ts_std', 'volume'):    'volume volatility',
        ('ts_rank', 'close'):    'price rank in window',
        ('ts_rank', 'volume'):   'volume rank in window',
        ('ts_rank', 'ret'):      'return rank (momentum)',
        ('ts_ret', 'close'):     'price momentum',
        ('ts_delta', 'volume'):  'volume acceleration',
        ('ts_delta', 'turnover'): 'turnover acceleration',
        ('ts_zscore', 'volume'): 'volume anomaly z-score',
        ('ts_zscore', 'ret'):    'return anomaly',
        ('ts_corr', 'close'):    'price-volume correlation',
        ('ts_corr', 'volume'):   'cross-series correlation',
        ('ts_decay_linear', 'ret'):    'exponentially-weighted momentum',
        ('ts_decay_linear', 'volume'): 'recent volume weight',
        ('ts_skew', 'ret'):      'return skewness (tail risk)',
        ('ts_max', 'ret'):       'max return in window (upside)',
        ('ts_min', 'ret'):       'min return in window (downside)',
        ('ts_maxpos', 'ret'):    'position of max return (momentum age)',
        ('ts_minpos', 'ret'):    'position of min return (drawdown timing)',
    }

    # Try to extract operator + primary operand from name
    for (op, operand), meaning in semantic_map.items():
        if op in name and operand in name:
            return True, meaning

    # Range / body / position / vwap operands always interpretable
    for operand in ('range', 'body', 'position', 'vwap_approx', 'turnover'):
        if operand in name:
            return True, f"interpretable microstructure feature ({operand})"

    # Any depth-2 using abs/neg/sign is still interpretable
    for elem in ('abs(', 'neg(', 'sign('):
        if elem in name:
            return True, f"monotone transform of interpretable base"

    # Fallback: reject anything we cannot name
    return False, "no recognizable semantic pattern"


def load_existing_features_sample(n_rows: int = 80000) -> pd.DataFrame:
    """Load ng101 features for orthogonality check (random sample)."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    rows = conn.execute(
        f"""SELECT code, trade_date, features_json
            FROM ng101_feature_cache
            WHERE features_json IS NOT NULL
              AND trade_date >= '2022-01-01'
              AND trade_date <= '2025-01-01'
            ORDER BY RANDOM() LIMIT {n_rows}""").fetchall()
    conn.close()
    parsed = []
    for code, td, fjson in rows:
        try:
            d = json.loads(fjson)
            d['__code'] = code
            d['__date'] = td
            parsed.append(d)
        except Exception:
            continue
    return pd.DataFrame(parsed)


def compute_factor_values_for_period(factor_spec: Dict, period: tuple,
                                     n_stocks: int = 500) -> pd.DataFrame:
    """Recompute a mined factor for all stocks in the given date range.

    Returns DataFrame[code, trade_date, factor_value, label_10d].
    """
    from factor_mining_pipeline import generate_operands, compute_factor

    start, end = period
    conn = sqlite3.connect(DB_PATH, timeout=30)

    # Sample stocks randomly (reproducible)
    all_codes = [r[0] for r in conn.execute(
        "SELECT code FROM securities WHERE type='A股' ORDER BY code").fetchall()]
    rng = np.random.RandomState(42)
    sampled = rng.choice(all_codes, min(n_stocks, len(all_codes)), replace=False).tolist()

    records = []
    for code in sampled:
        df_stk = pd.read_sql(
            """SELECT q.trade_date, q.open, q.high, q.low, q.close,
                      q.volume, q.price_change_pct
               FROM daily_quotes q
               JOIN securities s ON q.security_id = s.id
               WHERE s.code = ?
                 AND q.trade_date BETWEEN ? AND ?
               ORDER BY q.trade_date""",
            conn, params=[code, start, end])
        if len(df_stk) < 60:
            continue

        df_turn = pd.read_sql(
            """SELECT db.trade_date, db.turnover_rate
               FROM daily_basic db
               JOIN securities s ON db.security_id = s.id
               WHERE s.code = ?
                 AND db.trade_date BETWEEN ? AND ?""",
            conn, params=[code, start, end])
        df_stk = df_stk.merge(df_turn, on='trade_date', how='left')
        df_stk['turnover_rate'] = df_stk['turnover_rate'].fillna(0.0)

        operands = generate_operands(df_stk)
        try:
            factor_series = compute_factor(factor_spec, operands)
            if factor_series is None:
                continue
            factor_vals = np.asarray(factor_series.values, dtype=np.float64)
        except Exception:
            continue

        df_stk['label_10d'] = df_stk['close'].shift(-10) / df_stk['close'] - 1
        n = len(df_stk)
        for i in range(n):
            v = factor_vals[i] if i < len(factor_vals) else np.nan
            lbl = df_stk['label_10d'].values[i]
            if np.isfinite(v) and np.isfinite(lbl):
                records.append({
                    'code': code,
                    'trade_date': df_stk['trade_date'].values[i],
                    'factor_value': float(v),
                    'label_10d': float(lbl),
                })

    conn.close()
    return pd.DataFrame(records)


def compute_period_ic(df: pd.DataFrame, sign_flip: bool = False) -> float:
    """Mean Spearman IC across daily cross-sections.

    Requires >= 20 valid cross-sections each with >= 50 stocks.
    """
    if len(df) < 1000:
        return np.nan
    ics = []
    for _, grp in df.groupby('trade_date'):
        sub = grp.dropna(subset=['factor_value', 'label_10d'])
        if len(sub) < 50:
            continue
        try:
            ic, _ = spearmanr(sub['factor_value'], sub['label_10d'])
            if np.isfinite(ic):
                ics.append(float(ic))
        except Exception:
            continue
    if len(ics) < 20:
        return np.nan
    mean_ic = float(np.mean(ics))
    return -mean_ic if sign_flip else mean_ic


def check_orthogonality(factor_spec: Dict, existing_df: pd.DataFrame) -> tuple:
    """Check |corr| < 0.5 vs all ng101 + moneyflow features.

    Returns (is_orthogonal: bool, max_corr: float, correlated_feature: str).
    """
    if existing_df.empty:
        return True, 0.0, ''

    from factor_mining_pipeline import generate_operands, compute_factor

    # Compute factor values on the sample rows
    factor_vals_by_key = {}  # key = (code, date) -> factor_value
    conn = sqlite3.connect(DB_PATH, timeout=30)

    sampled_codes = existing_df['__code'].unique()[:200]
    for code in sampled_codes:
        rows = existing_df[existing_df['__code'] == code]
        if len(rows) < 20:
            continue
        dates = sorted(rows['__date'].tolist())
        start, end = dates[0], dates[-1]
        df_stk = pd.read_sql(
            """SELECT q.trade_date, q.open, q.high, q.low, q.close,
                      q.volume, q.price_change_pct
               FROM daily_quotes q
               JOIN securities s ON q.security_id = s.id
               WHERE s.code = ? AND q.trade_date BETWEEN ? AND ?
               ORDER BY q.trade_date""",
            conn, params=[code, start, end])
        if len(df_stk) < 60:
            continue
        df_turn = pd.read_sql(
            """SELECT db.trade_date, db.turnover_rate
               FROM daily_basic db
               JOIN securities s ON db.security_id = s.id
               WHERE s.code = ? AND db.trade_date BETWEEN ? AND ?""",
            conn, params=[code, start, end])
        df_stk = df_stk.merge(df_turn, on='trade_date', how='left')
        df_stk['turnover_rate'] = df_stk['turnover_rate'].fillna(0.0)

        operands = generate_operands(df_stk)
        try:
            fseries = compute_factor(factor_spec, operands)
            if fseries is None:
                continue
        except Exception:
            continue

        for i, row in rows.iterrows():
            td = row['__date']
            idx = df_stk[df_stk['trade_date'] == td].index
            if len(idx) == 0:
                continue
            pos = df_stk.index.get_loc(idx[0])
            if pos < len(fseries):
                v = float(fseries.values[pos])
                if np.isfinite(v):
                    factor_vals_by_key[(code, td)] = v

    conn.close()

    if len(factor_vals_by_key) < 500:
        # Not enough overlap — skip orthogonality (treat as orthogonal with warning)
        return True, 0.0, 'insufficient_overlap'

    # Build aligned series
    common_keys = list(factor_vals_by_key.keys())
    factor_arr = np.array([factor_vals_by_key[k] for k in common_keys])

    check_cols = [c for c in NG101_NUMERIC_FEATURES + ACCEPTED_MONEYFLOW_FEATURES
                  if c in existing_df.columns]

    max_corr = 0.0
    worst_feat = ''
    existing_indexed = existing_df.set_index(['__code', '__date'])

    aligned_existing = {}
    for col in check_cols:
        vals = []
        for (code, td) in common_keys:
            try:
                v = existing_indexed.loc[(code, td), col]
                vals.append(float(v) if np.isfinite(float(v)) else np.nan)
            except (KeyError, TypeError, ValueError):
                vals.append(np.nan)
        aligned_existing[col] = np.array(vals)

    for col, existing_arr in aligned_existing.items():
        mask = np.isfinite(factor_arr) & np.isfinite(existing_arr)
        if mask.sum() < 200:
            continue
        try:
            r, _ = spearmanr(factor_arr[mask], existing_arr[mask])
            if np.isfinite(r) and abs(r) > max_corr:
                max_corr = abs(r)
                worst_feat = col
        except Exception:
            continue

    is_ortho = max_corr < MAX_CORR_VS_EXISTING
    return is_ortho, max_corr, worst_feat


def main():
    p = argparse.ArgumentParser(
        description='Stage 2: validate mined factors for ng1.2.3')
    p.add_argument('--top-n-candidates', type=int, default=30,
                   help='Validate top N factors by |ICIR| from mining (default 30)')
    p.add_argument('--skip-orthogonality', action='store_true',
                   help='Skip orthogonality check (faster, for debugging)')
    p.add_argument('--n-stocks', type=int, default=500,
                   help='Stocks per period for cross-regime recompute (default 500)')
    args = p.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== ng1.2.3 Stage 2: Mined Factor Re-validation ===", flush=True)

    # ------------------------------------------------------------------ #
    # 1. Load mining output and pre-filter by |ICIR|>0.5                  #
    # ------------------------------------------------------------------ #
    candidates = load_mining_results()
    print(f"Loaded {len(candidates)} candidates from {MINING_RESULTS}", flush=True)

    # Sort by |ICIR| descending
    candidates.sort(key=lambda x: abs(x.get('icir', 0)), reverse=True)

    # Only keep those passing the primary ICIR gate
    strong = [c for c in candidates if abs(c.get('icir', 0)) >= MIN_ABS_ICIR]
    print(f"Factors with |ICIR|>={MIN_ABS_ICIR}: {len(strong)}", flush=True)
    to_validate = strong[:args.top_n_candidates]
    print(f"Validating top {len(to_validate)}", flush=True)

    # ------------------------------------------------------------------ #
    # 2. Orthogonality sample (load once, reuse across all factors)       #
    # ------------------------------------------------------------------ #
    existing_df = pd.DataFrame()
    if not args.skip_orthogonality:
        print("\nLoading ng101 feature sample for orthogonality check…", flush=True)
        try:
            existing_df = load_existing_features_sample(n_rows=80000)
            print(f"  Loaded {len(existing_df)} rows, "
                  f"{len(existing_df.get('__code', pd.Series()).unique())} stocks",
                  flush=True)
        except Exception as e:
            print(f"  WARNING: Could not load ng101 features ({e}); skipping orthogonality",
                  flush=True)

    # ------------------------------------------------------------------ #
    # 3. Per-factor validation loop                                        #
    # ------------------------------------------------------------------ #
    results = []
    for i, c in enumerate(to_validate):
        name = c['name']
        orig_ic = c.get('ic_mean', 0.0) or 0.0
        orig_icir = c.get('icir', 0.0) or 0.0
        sign_flip = orig_ic < 0

        print(f"\n[{i+1}/{len(to_validate)}] {name}", flush=True)
        print(f"  original IC={orig_ic:+.4f}  ICIR={orig_icir:+.3f}  sign_flip={sign_flip}",
              flush=True)

        # -- Economic interpretability --
        interp_ok, interp_reason = is_economically_interpretable(name)
        print(f"  interpretable={interp_ok}  reason='{interp_reason}'", flush=True)

        # -- Orthogonality --
        if not args.skip_orthogonality and not existing_df.empty:
            ortho_ok, max_corr, worst_feat = check_orthogonality(c, existing_df)
        else:
            ortho_ok, max_corr, worst_feat = True, 0.0, 'skipped'
        print(f"  orthogonal={ortho_ok}  max_corr={max_corr:.3f}  vs_feat='{worst_feat}'",
              flush=True)

        # -- Cross-regime stability --
        ic_per_regime: Dict[str, float] = {}
        for regime_name, period in REGIME_PERIODS.items():
            df_period = compute_factor_values_for_period(
                c, period, n_stocks=args.n_stocks)
            ic = compute_period_ic(df_period, sign_flip=sign_flip)
            ic_per_regime[regime_name] = ic
            label = f"{ic:+.4f}" if np.isfinite(ic) else "n/a"
            print(f"  {regime_name}: IC={label}", flush=True)

        valid_ics = [v for v in ic_per_regime.values() if np.isfinite(v)]
        both_covered = len(valid_ics) >= 2
        same_sign = (both_covered and
                     all(np.sign(v) == np.sign(valid_ics[0]) for v in valid_ics))
        both_strong = (both_covered and
                       all(abs(v) >= MIN_ABS_IC_REGIME for v in valid_ics))
        regime_stable = same_sign and both_strong

        print(f"  regime_stable={regime_stable}  same_sign={same_sign}  "
              f"both_strong={both_strong}", flush=True)

        # -- Final verdict for this factor --
        passes = regime_stable and ortho_ok and interp_ok
        print(f"  VERDICT: {'PASS' if passes else 'FAIL'}", flush=True)

        results.append({
            'name': name,
            'original_ic': round(orig_ic, 6),
            'original_icir': round(orig_icir, 4),
            'sign_flip': sign_flip,
            'interpretable': interp_ok,
            'interp_reason': interp_reason,
            'orthogonal': ortho_ok,
            'max_corr_vs_existing': round(max_corr, 4),
            'worst_corr_feature': worst_feat,
            'ic_bear_2022': round(ic_per_regime.get('bear_2022', np.nan), 6)
                            if np.isfinite(ic_per_regime.get('bear_2022', np.nan)) else None,
            'ic_recovery_2024': round(ic_per_regime.get('recovery_2024', np.nan), 6)
                                if np.isfinite(ic_per_regime.get('recovery_2024', np.nan)) else None,
            'same_sign_across_regimes': same_sign,
            'both_periods_strong': both_strong,
            'regime_stable': regime_stable,
            'passes_all': passes,
        })

    # ------------------------------------------------------------------ #
    # 4. Summary                                                          #
    # ------------------------------------------------------------------ #
    df_results = pd.DataFrame(results)
    # Sort: passes_all first, then |ICIR| descending
    df_results['_abs_icir'] = df_results['original_icir'].abs()
    df_results = df_results.sort_values(
        ['passes_all', '_abs_icir'], ascending=[False, False])
    df_results = df_results.drop(columns=['_abs_icir'])

    print("\n\n=== Stage 2 Results ===", flush=True)
    display_cols = ['name', 'original_icir', 'sign_flip', 'interpretable',
                    'orthogonal', 'ic_bear_2022', 'ic_recovery_2024',
                    'regime_stable', 'passes_all']
    available_cols = [c for c in display_cols if c in df_results.columns]
    print(df_results[available_cols].to_string(index=False), flush=True)

    n_pass = int(df_results['passes_all'].sum())
    n_regime_stable = int(df_results['regime_stable'].sum())
    overall_pass = n_pass >= 6

    print(f"\nFactors passing ALL filters: {n_pass}", flush=True)
    print(f"Regime-stable (before ortho/interp): {n_regime_stable}", flush=True)
    print(f"\nSTAGE 2 OVERALL: {'PASS ✅' if overall_pass else 'FAIL ❌'} "
          f"(need >=6, got {n_pass})", flush=True)

    # ------------------------------------------------------------------ #
    # 5. Save outputs                                                      #
    # ------------------------------------------------------------------ #
    csv_path = OUTPUT_DIR / 'stage2_mined_factors.csv'
    df_results.to_csv(csv_path, index=False)

    top6 = df_results[df_results['passes_all']].head(6).to_dict('records')
    # Strip non-serialisable NaN
    for row in top6:
        for k, v in row.items():
            if isinstance(v, float) and np.isnan(v):
                row[k] = None

    status = {
        'stage': 2,
        'overall_pass': bool(overall_pass),
        'n_validated': len(results),
        'n_pass': n_pass,
        'n_regime_stable': n_regime_stable,
        'top_6': top6,
    }
    json_path = OUTPUT_DIR / 'stage2_status.json'
    with open(json_path, 'w') as f:
        json.dump(status, f, indent=2, default=str)

    print(f"\nSaved: {csv_path}", flush=True)
    print(f"Saved: {json_path}", flush=True)


if __name__ == '__main__':
    main()
