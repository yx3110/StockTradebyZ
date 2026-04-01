#!/usr/bin/env python3
"""
V4.8.2 单因子IC验证脚本

对V4.8.2新增的21个因子逐一计算:
- Rank IC (Spearman相关性 vs forward returns)
- ICIR (IC均值/IC标准差)
- IC>0比例
- 因子间相关性矩阵 (与V4.8.1现有因子去重)

用法:
  python3 scripts/ic_test_v482_factors.py [--start-date 2023-01-01] [--end-date 2025-12-31]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
import argparse
import warnings
warnings.filterwarnings('ignore')

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'data_adapter', 'stock_data.db')

# V4.8.2 新因子定义
V482_PHASE1 = [
    'industry_adj_str', 'turnover_reversal', 'max5_lottery',
    'retail_crowding', 'chaikin_mf_20d', 'residual_momentum',
    'ksft_5d', 'sumd_20d', 'obv_price_div', 'limit_proximity_5d',
]
V482_PHASE2 = [
    'cfp', 'gpoa_approx', 'accruals_quality',
    'delta_roe_yoy', 'delta_leverage_yoy', 'rev_growth_consistency',
]
V482_PHASE3 = [
    'high_52w_ratio', 'trend_rsquared_20d', 'imxd_20d',
    'realized_skew_20d', 'trend_strength_60d',
]
ALL_FACTORS = V482_PHASE1 + V482_PHASE2 + V482_PHASE3


def load_ohlcv_data(conn, start_date, end_date):
    """Load OHLCV data with 370-day lookback for 52w high"""
    ext_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=370)).strftime('%Y-%m-%d')
    # Also need forward 20 days for forward returns
    ext_end = (datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=30)).strftime('%Y-%m-%d')

    query = """
    SELECT s.code, q.trade_date, q.open, q.high, q.low, q.close,
           q.volume, q.price_change_pct
    FROM daily_quotes q
    JOIN securities s ON q.security_id = s.id
    WHERE s.type = 'A股' AND q.trade_date >= ? AND q.trade_date <= ?
      AND q.volume > 0 AND q.close > 0
    ORDER BY s.code, q.trade_date
    """
    df = pd.read_sql(query, conn, params=[ext_start, ext_end])
    print(f"  OHLCV: {len(df):,} rows, {df['code'].nunique()} stocks")
    return df


def load_basic_data(conn, start_date, end_date):
    """Load daily_basic (turnover, market cap)"""
    ext_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=90)).strftime('%Y-%m-%d')
    ext_end = (datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=30)).strftime('%Y-%m-%d')

    query = """
    SELECT s.code, db.trade_date, db.turnover_rate, db.total_mv
    FROM daily_basic db
    JOIN securities s ON db.security_id = s.id
    WHERE db.trade_date >= ? AND db.trade_date <= ?
    """
    df = pd.read_sql(query, conn, params=[ext_start, ext_end])
    print(f"  daily_basic: {len(df):,} rows")
    return df


def load_financial_data(conn):
    """Load financial_indicator for Phase 2 factors"""
    query = """
    SELECT s.code, fi.end_date, fi.roe, fi.ocfps, fi.gross_margin,
           fi.assets_turn, fi.ocf_to_or, fi.or_yoy, fi.debt_to_assets
    FROM financial_indicator fi
    JOIN securities s ON fi.security_id = s.id
    WHERE fi.end_date >= '20180101'
    ORDER BY s.code, fi.end_date
    """
    df = pd.read_sql(query, conn)
    print(f"  financial_indicator: {len(df):,} rows")
    return df


def load_industry_map(conn):
    """Load industry mapping"""
    query = "SELECT code, industry FROM securities WHERE type = 'A股' AND industry IS NOT NULL"
    df = pd.read_sql(query, conn)
    return dict(zip(df['code'], df['industry']))


def compute_forward_returns(df_ohlcv, horizons=[3, 5, 10, 15]):
    """Compute forward returns for each stock"""
    parts = []
    for code, grp in df_ohlcv.groupby('code'):
        grp = grp.sort_values('trade_date').copy()
        close_s = grp['close'].reset_index(drop=True)
        for h in horizons:
            grp[f'fwd_ret_{h}d'] = close_s.shift(-h).values / close_s.values - 1
        parts.append(grp)
    return pd.concat(parts, ignore_index=True)


def compute_phase1_factors(df_ohlcv, df_basic, code_to_industry):
    """Compute Phase 1 price-volume factors (vectorized where possible)"""
    print("\n  Computing Phase 1 (price-volume) factors...")

    # Merge basic data
    df = df_ohlcv.merge(df_basic[['code', 'trade_date', 'turnover_rate', 'total_mv']],
                        on=['code', 'trade_date'], how='left')
    df['turnover_rate'] = df['turnover_rate'].fillna(0.0)
    df['total_mv'] = df['total_mv'].fillna(0.0)
    df['industry'] = df['code'].map(code_to_industry)

    # Pre-compute industry median 5d returns
    print("    Computing industry median returns...")
    ret5d_parts = []
    for code, grp in df.groupby('code'):
        grp = grp.sort_values('trade_date')
        r5d = grp['close'].pct_change(5)
        ret5d_parts.append(pd.DataFrame({
            'code': code, 'trade_date': grp['trade_date'].values,
            'ret5d': r5d.values, 'industry': grp['industry'].values
        }))
    df_ret5d = pd.concat(ret5d_parts, ignore_index=True)
    industry_med_ret5d = df_ret5d.groupby(['trade_date', 'industry'])['ret5d'].median()
    ind_med_dict = industry_med_ret5d.to_dict()

    # Market daily return
    mkt_ret = df.groupby('trade_date')['price_change_pct'].median().to_dict()

    factor_parts = []
    n_stocks = df['code'].nunique()
    processed = 0

    for code, grp in df.groupby('code'):
        grp = grp.sort_values('trade_date').copy()
        n = len(grp)
        if n < 25:
            continue

        close = grp['close'].values.astype(float)
        open_ = grp['open'].values.astype(float)
        high = grp['high'].values.astype(float)
        low = grp['low'].values.astype(float)
        volume = grp['volume'].values.astype(float)
        pct = pd.to_numeric(grp['price_change_pct'], errors='coerce').fillna(0).values.astype(float)
        turnover = grp['turnover_rate'].values.astype(float)
        mcap = grp['total_mv'].values.astype(float)
        dates = grp['trade_date'].values
        industry = code_to_industry.get(code, '')

        out = pd.DataFrame({'code': code, 'trade_date': dates})

        close_s = pd.Series(close)
        pct_s = pd.Series(pct)
        turn_s = pd.Series(turnover)
        vol_s = pd.Series(volume)

        # 1. industry_adj_str
        ret5d = close_s.pct_change(5).values
        ind_med = np.array([ind_med_dict.get((d, industry), 0.0) for d in dates])
        out['industry_adj_str'] = -(ret5d - ind_med)

        # 2. turnover_reversal
        out['turnover_reversal'] = -turn_s.rolling(20, min_periods=5).mean().values

        # 3. max5_lottery
        def _max5(x):
            if len(x) < 5: return 0.0
            return -np.mean(np.sort(x)[-5:])
        out['max5_lottery'] = pct_s.rolling(20, min_periods=10).apply(_max5, raw=True).values

        # 4. retail_crowding (placeholder - cross-sectional later)
        turn_rank = turn_s.rolling(60, min_periods=20).rank(pct=True).values
        mcap_s = pd.Series(mcap)
        mcap_rank_inv = 1.0 - mcap_s.rolling(60, min_periods=20).rank(pct=True).values
        out['retail_crowding'] = -(turn_rank * mcap_rank_inv)

        # 5. chaikin_mf_20d
        hl_range = high - low
        hl_safe = np.where(hl_range > 1e-8, hl_range, 1e-8)
        mf_mult = ((close - low) - (high - close)) / hl_safe
        mf_vol = mf_mult * volume
        mf_sum = pd.Series(mf_vol).rolling(20, min_periods=5).sum().values
        vol_sum = vol_s.rolling(20, min_periods=5).sum().values
        vol_sum_safe = np.where(vol_sum > 0, vol_sum, 1e-8)
        out['chaikin_mf_20d'] = mf_sum / vol_sum_safe

        # 6. residual_momentum
        mkt_rets = np.array([mkt_ret.get(d, 0.0) for d in dates])
        resid_mom = np.full(n, np.nan)
        for i in range(25, n):
            stock_r = pct[i-25:i]
            mkt_r = mkt_rets[i-25:i]
            mkt_var = np.var(mkt_r)
            if mkt_var > 1e-12:
                beta = np.cov(stock_r, mkt_r)[0,1] / mkt_var
            else:
                beta = 0.0
            residual = stock_r - beta * mkt_r
            resid_mom[i] = np.sum(residual[:20])
        out['residual_momentum'] = resid_mom

        # 7. ksft_5d
        ksft_daily = (2 * close - high - low) / hl_safe
        out['ksft_5d'] = pd.Series(ksft_daily).rolling(5, min_periods=3).mean().values

        # 8. sumd_20d
        gains = np.where(pct > 0, pct, 0.0)
        losses = np.where(pct < 0, -pct, 0.0)
        sum_gains = pd.Series(gains).rolling(20, min_periods=5).sum().values
        sum_losses = pd.Series(losses).rolling(20, min_periods=5).sum().values
        denom = sum_gains + sum_losses
        denom_safe = np.where(denom > 1e-8, denom, 1e-8)
        out['sumd_20d'] = (sum_gains - sum_losses) / denom_safe

        # 9. obv_price_div
        obv_sign = np.where(pct > 0, 1, np.where(pct < 0, -1, 0))
        obv = np.cumsum(obv_sign * volume)
        obv_s = pd.Series(obv)
        obv_ret = (obv_s / obv_s.shift(20).replace(0, np.nan) - 1).values
        price_ret = close_s.pct_change(20).values
        out['obv_price_div'] = obv_ret - price_ret

        # 10. limit_proximity_5d
        if code.startswith('3') or code.startswith('688'):
            limit_pct = 0.20
        elif code.startswith(('4', '8')):
            limit_pct = 0.30
        else:
            limit_pct = 0.10
        limit_prox = np.abs(pct) / limit_pct
        out['limit_proximity_5d'] = pd.Series(limit_prox).rolling(5, min_periods=3).mean().values

        factor_parts.append(out)
        processed += 1
        if processed % 1000 == 0:
            print(f"      Phase 1: {processed}/{n_stocks} stocks")

    df_p1 = pd.concat(factor_parts, ignore_index=True) if factor_parts else pd.DataFrame()
    print(f"    Phase 1 done: {len(df_p1):,} rows, {processed} stocks")

    # Cross-sectional normalization
    for col in ['turnover_reversal', 'retail_crowding']:
        if col in df_p1.columns:
            df_p1[col] = df_p1.groupby('trade_date')[col].rank(pct=True)
            df_p1[col] = 1.0 - df_p1[col]

    return df_p1


def compute_phase2_factors(df_fi, df_ohlcv):
    """Compute Phase 2 financial quality factors"""
    print("\n  Computing Phase 2 (financial quality) factors...")

    if len(df_fi) == 0:
        return pd.DataFrame()

    df_fi['end_date'] = df_fi['end_date'].str.replace('-', '')
    fi_factors = []

    for code, grp_fi in df_fi.groupby('code'):
        grp_fi = grp_fi.sort_values('end_date').copy()
        if len(grp_fi) < 2:
            continue

        roe_vals = grp_fi['roe'].values
        ocfps_vals = grp_fi['ocfps'].values
        gm_vals = grp_fi['gross_margin'].values
        at_vals = grp_fi['assets_turn'].values
        ocf_or_vals = grp_fi['ocf_to_or'].values
        or_yoy_vals = grp_fi['or_yoy'].values
        dta_vals = grp_fi['debt_to_assets'].values
        end_dates = grp_fi['end_date'].values

        for i in range(len(grp_fi)):
            row = {'code': code, 'fi_end_date': end_dates[i]}

            # cfp will be 'ocfps' — divide by close later after forward-fill
            row['cfp'] = float(ocfps_vals[i]) if pd.notna(ocfps_vals[i]) else np.nan

            gm = float(gm_vals[i]) if pd.notna(gm_vals[i]) else np.nan
            at = float(at_vals[i]) if pd.notna(at_vals[i]) else np.nan
            row['gpoa_approx'] = gm * at / 100.0 if pd.notna(gm) and pd.notna(at) else np.nan

            ocf_or = float(ocf_or_vals[i]) if pd.notna(ocf_or_vals[i]) else np.nan
            row['accruals_quality'] = -(1 - ocf_or) if pd.notna(ocf_or) else np.nan

            if i >= 4 and pd.notna(roe_vals[i]) and pd.notna(roe_vals[i-4]):
                row['delta_roe_yoy'] = float(roe_vals[i] - roe_vals[i-4])
            else:
                row['delta_roe_yoy'] = np.nan

            if i >= 4 and pd.notna(dta_vals[i]) and pd.notna(dta_vals[i-4]):
                row['delta_leverage_yoy'] = float(dta_vals[i] - dta_vals[i-4])
            else:
                row['delta_leverage_yoy'] = np.nan

            if i >= 3:
                last4 = [float(v) for v in or_yoy_vals[max(0,i-3):i+1] if pd.notna(v)]
                if len(last4) >= 2:
                    mean_g = np.mean(last4)
                    std_g = np.std(last4)
                    row['rev_growth_consistency'] = mean_g / std_g if std_g > 1e-8 else np.sign(mean_g) * 10.0
                else:
                    row['rev_growth_consistency'] = np.nan
            else:
                row['rev_growth_consistency'] = np.nan

            fi_factors.append(row)

    if not fi_factors:
        return pd.DataFrame()

    df_fi_computed = pd.DataFrame(fi_factors)
    print(f"    Financial factors computed: {len(df_fi_computed):,} quarterly rows")

    # Forward-fill to daily via simple per-code mapping
    # For each code: sort quarterly by end_date, then for each trade_date find latest fi_end_date <= trade_date
    code_dates = df_ohlcv[['code', 'trade_date']].drop_duplicates()
    code_dates['td_str'] = code_dates['trade_date'].str.replace('-', '')

    fi_merged_parts = []
    for code, fi_grp in df_fi_computed.groupby('code'):
        fi_grp = fi_grp.dropna(subset=['fi_end_date']).sort_values('fi_end_date')
        cd = code_dates[code_dates['code'] == code].sort_values('td_str').copy()
        if len(cd) == 0 or len(fi_grp) == 0:
            continue
        # Use searchsorted for efficient lookup
        fi_dates = fi_grp['fi_end_date'].values
        td_strs = cd['td_str'].values
        indices = np.searchsorted(fi_dates, td_strs, side='right') - 1
        # Map each trade_date to its matching fi row
        valid_mask = indices >= 0
        rows = []
        for col in V482_PHASE2:
            if col in fi_grp.columns:
                fi_vals = fi_grp[col].values
                mapped = np.where(valid_mask, fi_vals[np.clip(indices, 0, len(fi_vals)-1)], np.nan)
                cd[col] = mapped
            else:
                cd[col] = np.nan
        fi_merged_parts.append(cd[['code', 'trade_date'] + V482_PHASE2])

    if not fi_merged_parts:
        return pd.DataFrame()

    df_daily = pd.concat(fi_merged_parts, ignore_index=True)

    # cfp currently holds ocfps — convert to ocfps/close
    close_lookup = df_ohlcv.groupby(['code', 'trade_date'])['close'].last()
    df_daily = df_daily.merge(
        close_lookup.reset_index().rename(columns={'close': '_close'}),
        on=['code', 'trade_date'], how='left')
    close_safe = df_daily['_close'].replace(0, np.nan)
    df_daily['cfp'] = pd.to_numeric(df_daily['cfp'], errors='coerce') / close_safe

    print(f"    Phase 2 forward-filled: {len(df_daily):,} daily rows")
    return df_daily[['code', 'trade_date'] + V482_PHASE2]


def compute_phase3_factors(df_ohlcv):
    """Compute Phase 3 long-horizon + academic factors"""
    print("\n  Computing Phase 3 (academic) factors...")

    factor_parts = []
    n_stocks = df_ohlcv['code'].nunique()
    processed = 0

    for code, grp in df_ohlcv.groupby('code'):
        grp = grp.sort_values('trade_date').copy()
        n = len(grp)
        if n < 60:
            continue

        close = grp['close'].values.astype(float)
        high = grp['high'].values.astype(float)
        low = grp['low'].values.astype(float)
        pct = pd.to_numeric(grp['price_change_pct'], errors='coerce').fillna(0).values.astype(float)
        dates = grp['trade_date'].values

        out = pd.DataFrame({'code': code, 'trade_date': dates})
        close_s = pd.Series(close)
        pct_s = pd.Series(pct)

        # 17. high_52w_ratio
        high_s = pd.Series(high)
        max_252 = high_s.rolling(252, min_periods=60).max().values
        max_safe = np.where(max_252 > 1e-8, max_252, 1e-8)
        out['high_52w_ratio'] = close / max_safe

        # 18. trend_rsquared_20d
        rsq = np.full(n, np.nan)
        for i in range(19, n):
            y = close[i-19:i+1]
            x = np.arange(20, dtype=float)
            y_mean = np.mean(y)
            ss_tot = np.sum((y - y_mean) ** 2)
            if ss_tot > 1e-12:
                slope = np.sum((x - x.mean()) * (y - y_mean)) / np.sum((x - x.mean()) ** 2)
                intercept = y_mean - slope * x.mean()
                y_pred = slope * x + intercept
                ss_res = np.sum((y - y_pred) ** 2)
                rsq[i] = 1 - ss_res / ss_tot
            else:
                rsq[i] = 0.0
        out['trend_rsquared_20d'] = rsq

        # 19. imxd_20d
        imxd = np.full(n, np.nan)
        for i in range(19, n):
            h_w = high[i-19:i+1]
            l_w = low[i-19:i+1]
            imxd[i] = np.argmax(h_w) / 19.0 - np.argmin(l_w) / 19.0
        out['imxd_20d'] = imxd

        # 20. realized_skew_20d
        skew = np.full(n, np.nan)
        for i in range(19, n):
            r = pct[i-19:i+1]
            mu = np.mean(r)
            sigma = np.std(r)
            if sigma > 1e-8:
                skew[i] = np.mean(((r - mu) / sigma) ** 3)
            else:
                skew[i] = 0.0
        out['realized_skew_20d'] = skew

        # 21. trend_strength_60d
        ret_60d = close_s.pct_change(60).values
        vol_60d = pct_s.rolling(60, min_periods=20).std().values
        vol_safe = np.where(vol_60d > 1e-8, vol_60d, 1e-8)
        out['trend_strength_60d'] = ret_60d / (vol_safe * np.sqrt(60))

        factor_parts.append(out)
        processed += 1
        if processed % 1000 == 0:
            print(f"      Phase 3: {processed}/{n_stocks} stocks")

    df_p3 = pd.concat(factor_parts, ignore_index=True) if factor_parts else pd.DataFrame()
    print(f"    Phase 3 done: {len(df_p3):,} rows, {processed} stocks")
    return df_p3


def compute_rank_ic(df, factor_col, return_col):
    """Compute daily Spearman Rank IC between factor and forward return"""
    ics = []
    for date, grp in df.groupby('trade_date'):
        valid = grp[[factor_col, return_col]].dropna()
        if len(valid) < 30:
            continue
        ic, _ = stats.spearmanr(valid[factor_col], valid[return_col])
        if np.isfinite(ic):
            ics.append(ic)
    if len(ics) < 10:
        return {'mean_ic': np.nan, 'icir': np.nan, 'ic_pos_ratio': np.nan, 'n_days': 0}

    mean_ic = np.mean(ics)
    std_ic = np.std(ics)
    icir = mean_ic / std_ic if std_ic > 1e-8 else 0.0
    ic_pos = np.mean([1 if ic > 0 else 0 for ic in ics])

    return {
        'mean_ic': mean_ic,
        'icir': icir,
        'ic_pos_ratio': ic_pos,
        'n_days': len(ics),
    }


def main():
    parser = argparse.ArgumentParser(description='V4.8.2 单因子IC验证')
    parser.add_argument('--start-date', default='2023-01-01', help='Start date')
    parser.add_argument('--end-date', default='2025-12-31', help='End date')
    args = parser.parse_args()

    print(f"V4.8.2 单因子IC验证: {args.start_date} → {args.end_date}")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)

    # Load data
    print("\n1. 加载数据...")
    df_ohlcv = load_ohlcv_data(conn, args.start_date, args.end_date)
    df_basic = load_basic_data(conn, args.start_date, args.end_date)
    df_fi = load_financial_data(conn)
    code_to_industry = load_industry_map(conn)
    conn.close()

    # Compute forward returns
    print("\n2. 计算Forward Returns...")
    df_ohlcv = compute_forward_returns(df_ohlcv, horizons=[3, 5, 10, 15])
    print(f"  Forward returns computed for {len(df_ohlcv):,} rows")

    # Compute all factors
    print("\n3. 计算21个V4.8.2因子...")
    df_p1 = compute_phase1_factors(df_ohlcv, df_basic, code_to_industry)
    df_p2 = compute_phase2_factors(df_fi, df_ohlcv)
    df_p3 = compute_phase3_factors(df_ohlcv)

    # Merge all factors
    print("\n4. 合并因子...")
    # Start from ohlcv (has forward returns)
    df_all = df_ohlcv[['code', 'trade_date', 'fwd_ret_3d', 'fwd_ret_5d', 'fwd_ret_10d', 'fwd_ret_15d']].copy()

    # Filter to target date range
    df_all = df_all[(df_all['trade_date'] >= args.start_date) & (df_all['trade_date'] <= args.end_date)]

    if len(df_p1) > 0:
        df_all = df_all.merge(df_p1, on=['code', 'trade_date'], how='left')
    if len(df_p2) > 0:
        df_all = df_all.merge(df_p2, on=['code', 'trade_date'], how='left')
    if len(df_p3) > 0:
        df_all = df_all.merge(df_p3, on=['code', 'trade_date'], how='left')

    print(f"  合并后: {len(df_all):,} 行, {df_all['code'].nunique()} stocks")

    # Factor coverage report
    print("\n5. 因子覆盖率:")
    for f in ALL_FACTORS:
        if f in df_all.columns:
            non_null = df_all[f].notna().sum()
            pct = non_null / len(df_all) * 100
            print(f"  {f:30s}: {non_null:>10,} / {len(df_all):,} ({pct:5.1f}%)")
        else:
            print(f"  {f:30s}: NOT COMPUTED")

    # IC testing
    print("\n6. 单因子IC测试:")
    print("=" * 110)
    print(f"{'Factor':<30s} | {'3d IC':>8s} {'ICIR':>6s} | {'5d IC':>8s} {'ICIR':>6s} | "
          f"{'10d IC':>8s} {'ICIR':>6s} | {'15d IC':>8s} {'ICIR':>6s} | {'IC>0%':>5s} {'Days':>5s}")
    print("-" * 110)

    results = {}
    horizons = [3, 5, 10, 15]

    for f in ALL_FACTORS:
        if f not in df_all.columns:
            continue

        row_str = f"{f:<30s} |"
        f_results = {}

        for h in horizons:
            ret_col = f'fwd_ret_{h}d'
            ic_data = compute_rank_ic(df_all, f, ret_col)
            f_results[f'{h}d'] = ic_data

            if np.isnan(ic_data['mean_ic']):
                row_str += f" {'N/A':>8s} {'N/A':>6s} |"
            else:
                row_str += f" {ic_data['mean_ic']:>8.4f} {ic_data['icir']:>6.2f} |"

        # Use 10d results for summary
        ic10 = f_results.get('10d', {})
        ic_pos = ic10.get('ic_pos_ratio', 0)
        n_days = ic10.get('n_days', 0)
        row_str += f" {ic_pos*100:>4.0f}% {n_days:>5d}"

        # Quality markers
        ic10_val = ic10.get('mean_ic', 0) or 0
        icir10 = ic10.get('icir', 0) or 0
        if abs(ic10_val) >= 0.03 and abs(icir10) >= 0.5:
            row_str += "  ★★★"
        elif abs(ic10_val) >= 0.02 and abs(icir10) >= 0.3:
            row_str += "  ★★"
        elif abs(ic10_val) >= 0.01 and abs(icir10) >= 0.2:
            row_str += "  ★"

        print(row_str)
        results[f] = f_results

    # Correlation matrix between factors
    print("\n\n7. 因子间相关性矩阵 (Spearman, 取截面均值):")
    available_factors = [f for f in ALL_FACTORS if f in df_all.columns and df_all[f].notna().sum() > 1000]

    if len(available_factors) >= 2:
        # Compute average cross-sectional rank correlation
        corr_sum = pd.DataFrame(0.0, index=available_factors, columns=available_factors)
        corr_count = 0

        # Sample dates for speed
        all_dates = df_all['trade_date'].unique()
        sample_dates = np.random.choice(all_dates, min(50, len(all_dates)), replace=False)

        for date in sample_dates:
            day_data = df_all[df_all['trade_date'] == date][available_factors].dropna(axis=1, how='all')
            if len(day_data) < 30:
                continue
            day_corr = day_data.corr(method='spearman')
            corr_sum = corr_sum.add(day_corr.reindex(index=available_factors, columns=available_factors).fillna(0))
            corr_count += 1

        if corr_count > 0:
            corr_avg = corr_sum / corr_count

            # Print high correlations (>0.7)
            print(f"  (基于{corr_count}个交易日的截面相关性均值)")
            print(f"\n  ⚠️ 高相关性因子对 (|corr| > 0.5):")
            found_high = False
            for i, f1 in enumerate(available_factors):
                for j, f2 in enumerate(available_factors):
                    if i < j and abs(corr_avg.loc[f1, f2]) > 0.5:
                        print(f"    {f1} <-> {f2}: {corr_avg.loc[f1, f2]:.3f}")
                        found_high = True
            if not found_high:
                print(f"    无 (所有因子间|corr| < 0.5)")

    # Summary
    print("\n\n8. 总结推荐:")
    print("=" * 70)
    print(f"{'Factor':<30s} | {'10d IC':>8s} | {'10d ICIR':>8s} | {'Verdict':>10s}")
    print("-" * 70)

    pass_factors = []
    fail_factors = []
    for f in ALL_FACTORS:
        if f not in results:
            print(f"{f:<30s} | {'N/A':>8s} | {'N/A':>8s} | {'SKIP':>10s}")
            continue
        ic10 = results[f].get('10d', {})
        ic_val = ic10.get('mean_ic', 0) or 0
        icir_val = ic10.get('icir', 0) or 0

        if abs(ic_val) >= 0.01 and abs(icir_val) >= 0.2:
            verdict = "✅ PASS"
            pass_factors.append(f)
        else:
            verdict = "❌ FAIL"
            fail_factors.append(f)

        print(f"{f:<30s} | {ic_val:>8.4f} | {icir_val:>8.3f} | {verdict}")

    print(f"\n通过: {len(pass_factors)}/{len(ALL_FACTORS)} 因子")
    print(f"推荐保留: {pass_factors}")
    if fail_factors:
        print(f"建议移除: {fail_factors}")


if __name__ == '__main__':
    main()
