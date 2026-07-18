"""
重新拟合官方0AMV锚定外推模型参数 (indicators/market_amv.py 的 AMV_OFF_* 常量)

模型: Y_t = A * Y_{t-1} * (1 + BETA * r_中证全指,t) + K * 当日成交额(元)
方法: 固定 BETA 网格, 对 (A, K) 做一步预测最小二乘, 取 RMSE 最优。

用法 (官方CSV更新导入后, 若外推漂移变大可重跑并回填常量):
    python3 scripts/fit_amv_official.py
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from indicators.market_amv import DB_PATH  # noqa: E402


def main():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    df = pd.read_sql(
        """
        SELECT o.trade_date, o.close AS official, ma.market_amount, dq.close AS idx_close
        FROM market_amv_official o
        JOIN market_amv ma ON ma.trade_date = o.trade_date
        JOIN securities s ON s.code = '000985.SH'
        JOIN daily_quotes dq ON dq.security_id = s.id AND dq.trade_date = o.trade_date
        WHERE o.is_simulated = 0
        ORDER BY o.trade_date
        """,
        conn,
    )
    conn.close()
    if len(df) < 200:
        print(f'重叠样本太少 ({len(df)} 天), 先跑 scripts/import_amv_official.py')
        sys.exit(1)

    y = df['official'].to_numpy()
    amt = df['market_amount'].to_numpy() * 1000.0  # 千元→元
    r = df['idx_close'].to_numpy()
    r = np.concatenate([[0.0], r[1:] / r[:-1] - 1])

    best = None
    for beta in np.arange(0.8, 1.8, 0.05):
        X = np.column_stack([y[:-1] * (1 + beta * r[1:]), amt[1:]])
        coef, *_ = np.linalg.lstsq(X, y[1:], rcond=None)
        rmse = np.sqrt(np.mean((X @ coef - y[1:]) ** 2))
        if best is None or rmse < best[0]:
            best = (rmse, beta, coef)
    rmse, beta, (a, k) = best

    pred = a * y[:-1] * (1 + beta * r[1:]) + k * amt[1:]
    mape = np.mean(np.abs(pred - y[1:]) / y[1:])
    print(f'样本: {len(df)} 天 ({df.trade_date.iloc[0]} ~ {df.trade_date.iloc[-1]})')
    print(f'AMV_OFF_A = {a:.5f}      # 留存率 (日衰减 {1-a:.2%})')
    print(f'AMV_OFF_BETA = {beta:.2f}')
    print(f'AMV_OFF_K = {k:.4g}')
    print(f'一步预测 MAPE = {mape:.3%}, RMSE = {rmse:.1f}')
    print('如与 indicators/market_amv.py 现值差异明显, 手动回填常量并更新注释。')


if __name__ == '__main__':
    main()
