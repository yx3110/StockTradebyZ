"""
0AMV 全市场活跃市值指标

复刻指南针(Compass)活筹指数，改造为全市场版本。
数据源: 上证指数+深证成指 成交额之和
"""
import numpy as np
import pandas as pd
import sqlite3
import os
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, 'data_adapter', 'stock_data.db')

# === 通达信函数复刻 ===

def tdx_sma(series: np.ndarray, n: int, m: int = 1) -> np.ndarray:
    """中国式SMA: Y = (M * X + (N - M) * Y_prev) / N"""
    alpha = m / n
    result = np.zeros(len(series))
    result[0] = series[0]
    for i in range(1, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
    return result


def tdx_dma(series: np.ndarray, a: np.ndarray) -> np.ndarray:
    """动态移动平均: Y = A * X + (1 - A) * Y_prev, A clamp到(0,1)"""
    a_clamped = np.clip(a, 0.001, 1.0)
    result = np.zeros(len(series))
    result[0] = series[0]
    for i in range(1, len(series)):
        result[i] = a_clamped[i] * series[i] + (1 - a_clamped[i]) * result[i - 1]
    return result


def ema(series: np.ndarray, n: int) -> np.ndarray:
    """标准EMA"""
    s = pd.Series(series)
    return s.ewm(span=n, adjust=False).mean().values


def ma(series: np.ndarray, n: int) -> np.ndarray:
    """简单移动平均，前n-1个值用累积均值填充"""
    s = pd.Series(series)
    return s.rolling(n, min_periods=1).mean().values


# === 数据加载 ===

def load_market_amount(conn: sqlite3.Connection) -> pd.DataFrame:
    """加载上证+深证每日成交额"""
    query = """
        SELECT dq.trade_date, SUM(dq.amount) as market_amount
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.code IN ('000001.SH', '399001.SZ')
          AND dq.amount IS NOT NULL AND dq.amount > 0
        GROUP BY dq.trade_date
        HAVING COUNT(*) = 2
        ORDER BY dq.trade_date
    """
    df = pd.read_sql(query, conn)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df


def load_market_circ_mv(conn: sqlite3.Connection) -> pd.DataFrame:
    """加载每日全A股流通市值合计"""
    query = """
        SELECT db.trade_date, SUM(db.circ_mv) as market_circ_mv
        FROM daily_basic db
        JOIN securities s ON db.security_id = s.id
        WHERE s.type = 'A股'
          AND db.circ_mv IS NOT NULL AND db.circ_mv > 0
        GROUP BY db.trade_date
        ORDER BY db.trade_date
    """
    df = pd.read_sql(query, conn)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df


# === 0AMV核心计算 ===

def compute_amv(market_amount: np.ndarray, market_turnover: np.ndarray) -> dict:
    """计算完整0AMV指标"""
    var1 = tdx_sma(market_amount, 10, 1) / 1e7

    sma_var1_3 = tdx_sma(var1, 3, 1)
    sma_var1_8 = tdx_sma(var1, 8, 1)

    a_c5 = market_turnover / 0.02
    a_c13 = market_turnover / 0.10
    a_c34 = market_turnover / 0.18
    a_inf = market_turnover / 1.10

    c5 = tdx_dma(sma_var1_3, a_c5)
    c13 = tdx_dma(sma_var1_3, a_c13)
    c34 = tdx_dma(sma_var1_8, a_c34)
    inf_line = tdx_dma(var1, a_inf)

    ma60 = ma(var1, 60)

    ema12 = ema(var1, 12)
    ema26 = ema(var1, 26)
    dif = ema12 - ema26
    dea = ema(dif, 9)
    macd_hist = (dif - dea) * 2

    return {
        'var1': var1, 'amv_c5': c5, 'amv_c13': c13, 'amv_c34': c34,
        'amv_inf': inf_line, 'amv_ma60': ma60,
        'amv_dif': dif, 'amv_dea': dea, 'amv_macd': macd_hist,
    }


# === 牛熊状态机 ===

def compute_regime(var1: np.ndarray, ma60: np.ndarray, macd: np.ndarray,
                   slow_bear_days: int = 10) -> np.ndarray:
    """牛熊体制判断

    转牛 (急涨): var1涨>=4.3% AND var1>ma60 AND macd>0
    转熊 (急跌): var1跌<=-2.3% AND var1<ma60 AND macd<0
    转熊 (缓跌): 连续N天 var1<ma60 AND macd<0 → 强制转熊

    Args:
        slow_bear_days: 缓跌转熊的连续天数阈值 (默认10)
    """
    n = len(var1)
    regime = np.zeros(n, dtype=int)

    pct_change = np.zeros(n)
    pct_change[1:] = (var1[1:] - var1[:-1]) / (var1[:-1] + 1e-15)

    regime[0] = 1 if var1[0] > ma60[0] else -1

    bear_streak = 0  # 连续满足var1<ma60且macd<0的天数

    for i in range(1, n):
        prev_regime = regime[i - 1]

        if prev_regime == -1:
            bear_streak = 0
            bull_signal = (
                pct_change[i] >= 0.043
                and var1[i] > ma60[i]
                and macd[i] > 0
            )
            regime[i] = 1 if bull_signal else -1
        elif prev_regime == 1:
            # 急跌转熊
            bear_signal = (
                pct_change[i] <= -0.023
                and var1[i] < ma60[i]
                and macd[i] < 0
            )
            # 缓跌计数
            if var1[i] < ma60[i] and macd[i] < 0:
                bear_streak += 1
            else:
                bear_streak = 0

            if bear_signal or bear_streak >= slow_bear_days:
                regime[i] = -1
                bear_streak = 0
            else:
                regime[i] = 1

    return regime


# === DB读写 ===

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS market_amv (
    trade_date DATE PRIMARY KEY,
    market_amount REAL,
    market_circ_mv REAL,
    market_turnover REAL,
    var1 REAL,
    amv_c5 REAL,
    amv_c13 REAL,
    amv_c34 REAL,
    amv_inf REAL,
    amv_ma60 REAL,
    amv_dif REAL,
    amv_dea REAL,
    amv_macd REAL,
    amv_regime INTEGER
)
"""


def save_to_db(conn: sqlite3.Connection, df: pd.DataFrame):
    """将计算结果写入market_amv表"""
    conn.execute(CREATE_TABLE_SQL)
    rows = []
    for _, r in df.iterrows():
        rows.append((
            r['trade_date'].strftime('%Y-%m-%d') if hasattr(r['trade_date'], 'strftime')
            else str(r['trade_date']),
            r['market_amount'], r['market_circ_mv'], r['market_turnover'],
            r['var1'], r['amv_c5'], r['amv_c13'], r['amv_c34'], r['amv_inf'],
            r['amv_ma60'], r['amv_dif'], r['amv_dea'], r['amv_macd'],
            int(r['amv_regime']),
        ))
    conn.executemany(
        'INSERT OR REPLACE INTO market_amv VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
        rows
    )
    conn.commit()
    logger.info(f'Saved {len(rows)} rows to market_amv')


# === 主入口 ===

def compute_and_save(db_path: str = None):
    """完整计算流程"""
    if db_path is None:
        db_path = DB_PATH
    conn = sqlite3.connect(db_path, timeout=30)

    amount_df = load_market_amount(conn)
    circ_mv_df = load_market_circ_mv(conn)

    if amount_df.empty:
        logger.error('No market amount data. Run backfill_index_amount.py first.')
        conn.close()
        return None

    df = amount_df.merge(circ_mv_df, on='trade_date', how='inner')
    df = df.sort_values('trade_date').reset_index(drop=True)

    # tushare index_daily amount单位千元, circ_mv单位万元
    df['market_turnover'] = (df['market_amount'] * 1000) / (df['market_circ_mv'] * 10000 + 1e-15)

    logger.info(f'Data: {len(df)} days, {df["trade_date"].min()} ~ {df["trade_date"].max()}')

    amv = compute_amv(df['market_amount'].values, df['market_turnover'].values)
    for key, arr in amv.items():
        df[key] = arr

    df['amv_regime'] = compute_regime(amv['var1'], amv['amv_ma60'], amv['amv_macd'])

    save_to_db(conn, df)
    conn.close()

    bull_days = (df['amv_regime'] == 1).sum()
    bear_days = (df['amv_regime'] == -1).sum()
    switches = (df['amv_regime'].diff().abs() > 0).sum()
    print(f'\n0AMV计算完成: {len(df)}天')
    print(f'  牛市: {bull_days}天 ({100*bull_days/len(df):.1f}%)')
    print(f'  熊市: {bear_days}天 ({100*bear_days/len(df):.1f}%)')
    print(f'  切换次数: {switches}')
    print(f'  最新regime: {"牛市" if df.iloc[-1]["amv_regime"]==1 else "熊市"}')
    print(f'  最新var1: {df.iloc[-1]["var1"]:.2f}')
    print(f'  最新ma60: {df.iloc[-1]["amv_ma60"]:.2f}')

    return df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    compute_and_save()
