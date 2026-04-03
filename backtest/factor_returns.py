"""
A股4因子收益构建模块

从SQLite数据库构建 Fama-French 风格的4因子日收益:
- MKT: 市场因子 (中证500超额收益)
- SMB: 规模因子 (小盘-大盘)
- HML: 价值因子 (高B/P - 低B/P)
- UMD: 动量因子 (Winner - Loser, 12-1月)
"""

import numpy as np
import pandas as pd
import sqlite3
import os
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, 'data_adapter', 'stock_data.db')
RISK_FREE_RATE = 0.02  # 年化2%


def _to_db_date(date_str: str) -> str:
    """Convert date string to YYYY-MM-DD format used in DB."""
    s = date_str.replace('-', '')
    if len(s) == 8:
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return date_str


def build_factor_returns(start_date: str, end_date: str,
                         db_path: str = None) -> pd.DataFrame:
    if db_path is None:
        db_path = DB_PATH
    sd = _to_db_date(start_date)
    ed = _to_db_date(end_date)
    # Load stock data with 14-month lookback so UMD can be computed
    lookback_sd = (pd.Timestamp(sd) - pd.DateOffset(months=14)).strftime('%Y-%m-%d')
    conn = sqlite3.connect(db_path, timeout=30)
    mkt_df = _build_mkt_factor(conn, sd, ed)
    stock_data = _load_stock_data(conn, lookback_sd, ed)
    conn.close()
    if stock_data.empty:
        logger.warning("No stock data found for factor construction")
        return pd.DataFrame(columns=['MKT', 'SMB', 'HML', 'UMD'])
    # SMB and HML only need current period data
    current_stock_data = stock_data[stock_data['trade_date'] >= sd]
    smb_df = _build_smb_factor(current_stock_data)
    hml_df = _build_hml_factor(current_stock_data)
    # UMD needs full lookback data
    umd_df = _build_umd_factor(stock_data)
    # Filter UMD to requested date range
    if not umd_df.empty:
        umd_df = umd_df[umd_df.index >= sd]
    result = mkt_df
    for factor_df in [smb_df, hml_df, umd_df]:
        if not factor_df.empty and len(factor_df.columns) > 0:
            result = result.join(factor_df, how='left')
    # Ensure all 4 factor columns are always present
    for col in ['MKT', 'SMB', 'HML', 'UMD']:
        if col not in result.columns:
            result[col] = 0.0
    result = result[['MKT', 'SMB', 'HML', 'UMD']].fillna(0.0)
    return result


def _build_mkt_factor(conn, sd, ed):
    # Fetch index close with one extra day lookback for return computation
    lookback_sd = (pd.Timestamp(sd) - pd.DateOffset(days=5)).strftime('%Y-%m-%d')
    query = """
        SELECT dq.trade_date,
               COALESCE(
                   CASE WHEN dq.price_change_pct IS NOT NULL THEN CAST(dq.price_change_pct AS REAL) END,
                   CASE WHEN LAG(dq.close) OVER (ORDER BY dq.trade_date) IS NOT NULL
                        THEN dq.close / LAG(dq.close) OVER (ORDER BY dq.trade_date) - 1 END
               ) AS ret
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.code = '000905.SH'
          AND dq.trade_date BETWEEN ? AND ?
        ORDER BY dq.trade_date
    """
    df = pd.read_sql(query, conn, params=[lookback_sd, ed])
    if df.empty:
        return pd.DataFrame(columns=['MKT'])
    df['trade_date'] = df['trade_date'].astype(str)
    df['ret'] = pd.to_numeric(df['ret'], errors='coerce')
    df = df.dropna(subset=['ret'])
    df = df[df['trade_date'] >= sd]
    df = df.set_index('trade_date')
    df['MKT'] = df['ret'] - RISK_FREE_RATE / 252
    return df[['MKT']]


def _load_stock_data(conn, sd, ed):
    # Compute returns from close prices using LAG window function
    # Use COALESCE: prefer stored price_change_pct (decimal), fall back to computed
    query = """
        WITH base AS (
            SELECT dq.trade_date, s.code,
                   dq.price_change_pct,
                   dq.close,
                   LAG(dq.close) OVER (PARTITION BY dq.security_id ORDER BY dq.trade_date) AS prev_close,
                   db.total_mv, db.pb
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            LEFT JOIN daily_basic db ON db.security_id = dq.security_id
                                     AND db.trade_date = dq.trade_date
            WHERE s.type = 'A股'
              AND dq.trade_date BETWEEN ? AND ?
              AND dq.volume > 0
              AND db.total_mv > 0
              AND db.pb > 0
        )
        SELECT trade_date, code,
               COALESCE(
                   CASE WHEN price_change_pct IS NOT NULL THEN CAST(price_change_pct AS REAL) END,
                   CASE WHEN prev_close IS NOT NULL AND prev_close > 0
                        THEN (close / prev_close - 1) END
               ) AS price_change_pct,
               total_mv, pb
        FROM base
        WHERE close > 0
        ORDER BY trade_date, code
    """
    df = pd.read_sql(query, conn, params=[sd, ed])
    df['trade_date'] = df['trade_date'].astype(str)
    df['price_change_pct'] = pd.to_numeric(df['price_change_pct'], errors='coerce')
    # Drop rows where we couldn't compute return (first date for each stock)
    df = df.dropna(subset=['price_change_pct'])
    return df


def _build_smb_factor(stock_data):
    result = {}
    for date, group in stock_data.groupby('trade_date'):
        if len(group) < 50:
            continue
        q20 = group['total_mv'].quantile(0.2)
        q80 = group['total_mv'].quantile(0.8)
        small = group[group['total_mv'] <= q20]['price_change_pct'].mean()
        big = group[group['total_mv'] >= q80]['price_change_pct'].mean()
        result[date] = small - big
    s = pd.Series(result, name='SMB')
    s.index.name = 'trade_date'
    return s.to_frame()


def _build_hml_factor(stock_data):
    result = {}
    for date, group in stock_data.groupby('trade_date'):
        if len(group) < 50:
            continue
        group = group.copy()
        group['bp'] = 1.0 / group['pb']
        q20 = group['bp'].quantile(0.2)
        q80 = group['bp'].quantile(0.8)
        high_bp = group[group['bp'] >= q80]['price_change_pct'].mean()
        low_bp = group[group['bp'] <= q20]['price_change_pct'].mean()
        result[date] = high_bp - low_bp
    s = pd.Series(result, name='HML')
    s.index.name = 'trade_date'
    return s.to_frame()


def _build_umd_factor(stock_data):
    dates = sorted(stock_data['trade_date'].unique())
    if len(dates) < 252:
        logger.warning("Not enough dates for UMD factor (need 252+)")
        return pd.DataFrame(columns=['UMD'])
    pivot = stock_data.pivot_table(
        index='trade_date', columns='code',
        values='price_change_pct', aggfunc='first'
    )
    cum_ret = (1 + pivot).cumprod()
    result = {}
    for i, date in enumerate(dates):
        if i < 252:
            continue
        date_12m = dates[max(0, i - 252)]
        date_1m = dates[max(0, i - 21)]
        if date_12m not in cum_ret.index or date_1m not in cum_ret.index:
            continue
        mom = cum_ret.loc[date_1m] / cum_ret.loc[date_12m] - 1
        mom = mom.dropna()
        if len(mom) < 50:
            continue
        q20 = mom.quantile(0.2)
        q80 = mom.quantile(0.8)
        day_ret = pivot.loc[date].dropna()
        common = mom.index.intersection(day_ret.index)
        winners = day_ret[common[mom[common] >= q80]].mean()
        losers = day_ret[common[mom[common] <= q20]].mean()
        if not np.isnan(winners) and not np.isnan(losers):
            result[date] = winners - losers
    s = pd.Series(result, name='UMD')
    s.index.name = 'trade_date'
    return s.to_frame()


def load_or_build_factors(start_date: str, end_date: str,
                          db_path: str = None,
                          cache_table: str = 'factor_daily_returns') -> pd.DataFrame:
    if db_path is None:
        db_path = DB_PATH
    sd = _to_db_date(start_date)
    ed = _to_db_date(end_date)
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        cached = pd.read_sql(
            f"SELECT * FROM {cache_table} WHERE trade_date BETWEEN ? AND ? ORDER BY trade_date",
            conn, params=[sd, ed]
        )
        if len(cached) > 0:
            cached = cached.set_index('trade_date')
            cached.index = pd.to_datetime(cached.index)
            conn.close()
            logger.info(f"Loaded {len(cached)} factor returns from cache")
            return cached[['MKT', 'SMB', 'HML', 'UMD']]
    except Exception:
        pass
    conn.close()
    lookback_sd = pd.Timestamp(sd) - pd.DateOffset(months=14)
    lookback_sd_str = lookback_sd.strftime('%Y-%m-%d')
    df = build_factor_returns(lookback_sd_str, ed, db_path)
    df = df[df.index >= sd]
    if df.empty:
        return df
    try:
        conn = sqlite3.connect(db_path, timeout=30)
        cache_df = df.reset_index()
        cache_df.columns = ['trade_date', 'MKT', 'SMB', 'HML', 'UMD']
        cache_df.to_sql(cache_table, conn, if_exists='replace', index=False)
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{cache_table}_date ON {cache_table}(trade_date)")
        conn.commit()
        conn.close()
        logger.info(f"Cached {len(df)} factor returns to {cache_table}")
    except Exception as e:
        logger.warning(f"Failed to cache factor returns: {e}")
    return df
