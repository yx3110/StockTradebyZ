#!/usr/bin/env python3
"""
回填V492因子缓存到DB — 完全复用V482Trainer的rolling逻辑

将10个V492新因子预计算存到 v492_factor_cache 表:
  V482: high_52w_ratio, residual_momentum, turnover_reversal, sumd_20d, realized_skew_20d
  Alpha158: corr_close_vol_20d, cntp_20d, rsqr_20d, ksft, imax_20d

存储格式: (code, trade_date, factor1, factor2, ..., factor10)
推理时 scorer 直接 SELECT 即可，无需实时计算。
"""

import numpy as np
import pandas as pd
import sqlite3
import logging
import json
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / 'data_adapter' / 'stock_data.db'

V492_FACTORS = [
    'high_52w_ratio', 'residual_momentum', 'turnover_reversal',
    'sumd_20d', 'realized_skew_20d',
    'corr_close_vol_20d', 'cntp_20d', 'rsqr_20d', 'ksft', 'imax_20d',
]


def create_table(conn):
    """创建v492_factor_cache表"""
    conn.execute(f"""
    CREATE TABLE IF NOT EXISTS v492_factor_cache (
        code TEXT NOT NULL,
        trade_date TEXT NOT NULL,
        {', '.join(f'{f} REAL DEFAULT 0' for f in V492_FACTORS)},
        PRIMARY KEY (code, trade_date)
    )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_v492_fc_date ON v492_factor_cache(trade_date)")
    conn.commit()
    logger.info("v492_factor_cache 表已创建/确认")


def compute_all_factors(start_date='2023-06-01', end_date='2026-12-31'):
    """计算所有因子 (与V482Trainer.load_data完全一致的rolling逻辑)

    需要252天lookback, 所以 start_date 往前扩展370天
    """
    conn = sqlite3.connect(str(DB_PATH), timeout=30)

    from datetime import datetime as dt_cls, timedelta as td_cls
    try:
        dt_start = dt_cls.strptime(start_date, '%Y-%m-%d')
    except ValueError:
        dt_start = dt_cls.strptime(start_date, '%Y%m%d')
    ext_start = (dt_start - td_cls(days=370)).strftime('%Y-%m-%d')

    logger.info(f"加载OHLCV数据: {ext_start} ~ {end_date}")
    t0 = time.time()

    ohlcv_query = """
    SELECT s.code, q.trade_date, q.open, q.high, q.low, q.close,
           q.volume, q.price_change_pct
    FROM daily_quotes q
    JOIN securities s ON q.security_id = s.id
    WHERE s.type = 'A股' AND q.trade_date >= ? AND q.trade_date <= ?
    ORDER BY s.code, q.trade_date
    """
    df_ohlcv = pd.read_sql(ohlcv_query, conn, params=[ext_start, end_date])
    logger.info(f"  OHLCV: {len(df_ohlcv):,} 行, {df_ohlcv['code'].nunique()} 只股票, 耗时{time.time()-t0:.1f}s")

    # 换手率
    basic_query = """
    SELECT s.code, db.trade_date, db.turnover_rate
    FROM daily_basic db JOIN securities s ON db.security_id = s.id
    WHERE db.trade_date >= ? AND db.trade_date <= ?
    """
    df_basic = pd.read_sql(basic_query, conn, params=[ext_start, end_date])
    logger.info(f"  换手率: {len(df_basic):,} 行")

    df_ohlcv = df_ohlcv.merge(df_basic[['code', 'trade_date', 'turnover_rate']],
                               on=['code', 'trade_date'], how='left')
    df_ohlcv['turnover_rate'] = df_ohlcv['turnover_rate'].fillna(0.0)

    # 全市场每日中位数收益 (for residual_momentum)
    logger.info("  计算全市场每日中位数收益...")
    market_daily_ret = {}
    for td, grp in df_ohlcv.groupby('trade_date'):
        pcts = pd.to_numeric(grp['price_change_pct'], errors='coerce').dropna()
        if len(pcts) > 0:
            market_daily_ret[td] = float(pcts.median())

    conn.close()

    # 逐股计算因子
    logger.info(f"逐股计算10个因子...")
    factor_parts = []
    n_stocks = df_ohlcv['code'].nunique()
    processed = 0
    t0 = time.time()

    for code, grp in df_ohlcv.groupby('code'):
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
        dates = grp['trade_date'].values

        out = pd.DataFrame({'code': code, 'trade_date': dates})
        close_s = pd.Series(close)
        pct_s = pd.Series(pct)

        # === V482 factors (完全复用训练逻辑) ===

        # 1. high_52w_ratio: close / max(high, 252d)
        high_s = pd.Series(high)
        max_252 = high_s.rolling(252, min_periods=60).max().values
        max_252_safe = np.where(max_252 > 1e-8, max_252, 1e-8)
        out['high_52w_ratio'] = close / max_252_safe

        # 2. residual_momentum
        mkt_rets = np.array([market_daily_ret.get(d, 0.0) for d in dates])
        resid_mom = np.full(n, np.nan)
        for i in range(25, n):
            stock_r = pct[i-25:i]
            mkt_r = mkt_rets[i-25:i]
            mkt_var = np.var(mkt_r)
            if mkt_var > 1e-12:
                beta = np.cov(stock_r, mkt_r)[0, 1] / mkt_var
            else:
                beta = 0.0
            residual = stock_r - beta * mkt_r
            resid_mom[i] = np.sum(residual[:20])
        out['residual_momentum'] = resid_mom

        # 3. turnover_reversal
        turn_s = pd.Series(turnover)
        out['turnover_reversal'] = -turn_s.rolling(20, min_periods=5).mean().values

        # 4. sumd_20d
        gains = np.where(pct > 0, pct, 0.0)
        losses = np.where(pct < 0, -pct, 0.0)
        sum_gains = pd.Series(gains).rolling(20, min_periods=5).sum().values
        sum_losses = pd.Series(losses).rolling(20, min_periods=5).sum().values
        denom = sum_gains + sum_losses
        denom_safe = np.where(denom > 1e-8, denom, 1e-8)
        out['sumd_20d'] = (sum_gains - sum_losses) / denom_safe

        # 5. realized_skew_20d
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

        # === Alpha158 factors ===

        # 6. corr_close_vol_20d
        log_vol = np.log(volume + 1)
        corr_vals = np.full(n, np.nan)
        for i in range(19, n):
            c_w = close[i-19:i+1]
            v_w = log_vol[i-19:i+1]
            if np.std(c_w) > 1e-8 and np.std(v_w) > 1e-8:
                corr_vals[i] = np.corrcoef(c_w, v_w)[0, 1]
        out['corr_close_vol_20d'] = corr_vals

        # 7. cntp_20d
        up_flag = (close > np.concatenate([[close[0]], close[:-1]])).astype(float)
        out['cntp_20d'] = pd.Series(up_flag).rolling(20, min_periods=5).mean().values

        # 8. rsqr_20d
        rsqr = np.full(n, np.nan)
        for i in range(19, n):
            y = close[i-19:i+1]
            x = np.arange(20)
            if np.std(y) > 1e-8:
                corr = np.corrcoef(x, y)[0, 1]
                rsqr[i] = corr ** 2
        out['rsqr_20d'] = rsqr

        # 9. ksft: (2*close - high - low) / open
        open_safe = np.where(open_ > 1e-8, open_, 1e-8)
        out['ksft'] = (2 * close - high - low) / open_safe

        # 10. imax_20d
        imax = np.full(n, np.nan)
        for i in range(19, n):
            h_window = high[i-19:i+1]
            imax[i] = np.argmax(h_window) / 19.0
        out['imax_20d'] = imax

        # 只保留start_date之后的行
        out = out[out['trade_date'] >= start_date].copy()
        if len(out) > 0:
            factor_parts.append(out)

        processed += 1
        if processed % 500 == 0:
            elapsed = time.time() - t0
            rate = processed / elapsed
            eta = (n_stocks - processed) / rate
            logger.info(f"  已处理 {processed}/{n_stocks} ({processed/n_stocks*100:.1f}%), "
                       f"速度={rate:.0f}股/秒, ETA={eta/60:.0f}分钟")

    if not factor_parts:
        logger.error("无有效数据!")
        return

    df_all = pd.concat(factor_parts, ignore_index=True)
    df_all = df_all.fillna(0.0)
    logger.info(f"因子计算完成: {len(df_all):,} 行, {df_all['code'].nunique()} 只股票, "
                f"{df_all['trade_date'].nunique()} 交易日")

    # 写入DB
    logger.info("写入数据库...")
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    create_table(conn)

    # 分批写入 (每批5000行, 避免SQLite变量限制)
    batch_size = 5000
    total_written = 0
    for start in range(0, len(df_all), batch_size):
        batch = df_all.iloc[start:start+batch_size]
        batch.to_sql('v492_factor_cache', conn, if_exists='append', index=False)
        total_written += len(batch)
        if total_written % 50000 < batch_size:
            logger.info(f"  已写入 {total_written:,}/{len(df_all):,}")

    conn.commit()

    # 验证
    count = conn.execute("SELECT COUNT(*) FROM v492_factor_cache").fetchone()[0]
    n_dates = conn.execute("SELECT COUNT(DISTINCT trade_date) FROM v492_factor_cache").fetchone()[0]
    conn.close()

    logger.info(f"回填完成: {count:,} 行, {n_dates} 交易日")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', default='2024-01-01')
    parser.add_argument('--end-date', default='2026-03-30')
    args = parser.parse_args()

    compute_all_factors(start_date=args.start_date, end_date=args.end_date)
