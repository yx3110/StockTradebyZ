#!/usr/bin/env python3
"""
从V4.8.7报告构建回测数据pickle (供阈值优化用)

读取 reports/daily_selection_v4.8.7/analysis_data_*.json,
计算 composite (0.6*pred_10d + 0.4*pred_15d),
合并未来N天真实收益,
输出 /tmp/v487_full_backtest.pkl

列: date, code, composite, rank, pred_3d, pred_5d, pred_10d, pred_15d,
    fwd_3d, fwd_5d, fwd_10d, fwd_15d
"""
import json, os, sys, sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / 'reports' / 'daily_selection_v4.8.7'
DB_PATH = PROJECT_ROOT / 'data_adapter' / 'stock_data.db'
OUTPUT_PATH = '/tmp/v487_full_backtest.pkl'

TRANSACTION_COST = 0.00302


def load_reports():
    """加载所有analysis_data JSON, 返回DataFrame"""
    rows = []
    for json_file in sorted(REPORT_DIR.glob('analysis_data_*.json')):
        date_str = json_file.stem.replace('analysis_data_', '')
        date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except Exception:
            continue

        stocks = data.get('all_stocks_with_scores', [])
        for s in stocks:
            code = s.get('stock_code', '')
            if not code or len(code) != 6:
                continue
            p3 = s.get('pred_3d', 0) or 0
            p5 = s.get('pred_5d', 0) or 0
            p10 = s.get('pred_10d', 0) or 0
            p15 = s.get('pred_15d', 0) or 0
            composite = 0.6 * p10 + 0.4 * p15
            rows.append({
                'date': date,
                'code': code,
                'composite': composite,
                'pred_3d': p3,
                'pred_5d': p5,
                'pred_10d': p10,
                'pred_15d': p15,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # 计算每天的排名 (越小越好)
    df['rank'] = df.groupby('date')['composite'].rank(ascending=False, method='first').astype(int)
    return df


def add_forward_returns(df):
    """从数据库加载未来收益"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA busy_timeout = 30000")

    # 获取所有交易日
    all_dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT trade_date FROM daily_quotes ORDER BY trade_date"
    ).fetchall()]
    date_idx = {d: i for i, d in enumerate(all_dates)}

    # 批量查询收盘价
    all_codes = df['code'].unique().tolist()
    min_date = df['date'].min()

    # 分块查询避免SQL过大
    CHUNK = 500
    price_frames = []
    for i in range(0, len(all_codes), CHUNK):
        chunk_codes = all_codes[i:i + CHUNK]
        codes_str = ','.join(f"'{c}'" for c in chunk_codes)
        query = f"""
            SELECT s.code, dq.trade_date, dq.close
            FROM daily_quotes dq
            JOIN securities s ON s.id = dq.security_id
            WHERE s.code IN ({codes_str})
              AND dq.trade_date >= '{min_date}'
            ORDER BY s.code, dq.trade_date
        """
        price_frames.append(pd.read_sql(query, conn))

    conn.close()

    if not price_frames:
        return df

    prices = pd.concat(price_frames, ignore_index=True)

    # 构建价格查找表
    price_map = {}
    for _, row in prices.iterrows():
        price_map[(row['code'], row['trade_date'])] = row['close']

    # 计算前瞻收益
    for hold_days, col_name in [(3, 'fwd_3d'), (5, 'fwd_5d'), (10, 'fwd_10d'), (15, 'fwd_15d')]:
        fwd_returns = []
        for _, row in df.iterrows():
            date = row['date']
            code = row['code']
            if date not in date_idx:
                fwd_returns.append(np.nan)
                continue
            idx = date_idx[date]
            future_idx = idx + hold_days
            if future_idx >= len(all_dates):
                fwd_returns.append(np.nan)
                continue
            future_date = all_dates[future_idx]
            buy_price = price_map.get((code, date))
            sell_price = price_map.get((code, future_date))
            if buy_price and sell_price and buy_price > 0:
                ret = (sell_price - buy_price) / buy_price - 2 * TRANSACTION_COST
                fwd_returns.append(ret)
            else:
                fwd_returns.append(np.nan)
        df[col_name] = fwd_returns

    return df


def main():
    print(f"加载V4.8.7报告: {REPORT_DIR}")
    df = load_reports()
    if df.empty:
        print("ERROR: 无报告数据")
        sys.exit(1)

    n_dates = df['date'].nunique()
    n_stocks = len(df)
    print(f"  报告: {n_dates}天, {n_stocks}条记录")
    print(f"  日期范围: {df['date'].min()} ~ {df['date'].max()}")
    print(f"  Composite分布: mean={df['composite'].mean():.6f}, "
          f"median={df['composite'].median():.6f}, "
          f"P95={df['composite'].quantile(0.95):.6f}")

    print(f"\n加载未来收益数据...")
    df = add_forward_returns(df)

    for col in ['fwd_3d', 'fwd_5d', 'fwd_10d', 'fwd_15d']:
        valid = df[col].notna().sum()
        if valid > 0:
            mean_ret = df[col].dropna().mean()
            print(f"  {col}: {valid}条有效 (mean={mean_ret:+.4f})")

    df.to_pickle(OUTPUT_PATH)
    print(f"\n✅ 已保存: {OUTPUT_PATH} ({len(df)}行)")


if __name__ == '__main__':
    main()
