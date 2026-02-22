#!/usr/bin/env python3
"""
分析隔夜跳空对策略收益的影响。

读取已有的 strategy_signals CSV，从数据库补充 close[T] 和 open[T+1]，
按不同跳空阈值过滤后重新计算收益指标。
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_adapter.stock_data_loader import StockDataLoader


def main():
    csv_path = PROJECT_ROOT / "reports" / "backtest" / "strategy_signals_20260222.csv"
    print(f"加载信号: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["date"], dtype={"code": str})
    print(f"总信号数: {len(df)}")

    # ── 加载行情数据，计算 gap ──
    print("加载行情数据...")
    loader = StockDataLoader(str(PROJECT_ROOT / "data_adapter" / "stock_data.db"))
    data = loader.load_all_stock_data_wide(
        start_date="2025-01-01", end_date="2026-02-13",
        lookback_days=10, security_types=["A股"]
    )
    print(f"加载 {len(data)} 只股票")

    # 构建 (code, date) → (close_T, open_T1) 查找表
    gap_records = []
    for code, stock_df in data.items():
        dates = stock_df["date"].values
        closes = stock_df["close"].values
        opens = stock_df["open"].values
        for i in range(len(stock_df) - 1):
            close_t = closes[i]
            open_t1 = opens[i + 1]
            if close_t > 0 and open_t1 > 0:
                gap_pct = (open_t1 - close_t) / close_t * 100
                gap_records.append({
                    "code": code,
                    "date": dates[i],
                    "gap_pct": gap_pct,
                })

    gap_df = pd.DataFrame(gap_records)
    gap_df["date"] = pd.to_datetime(gap_df["date"])
    print(f"Gap记录: {len(gap_df)}")

    # 合并
    df = df.merge(gap_df, on=["code", "date"], how="left")
    df_with_gap = df[df["gap_pct"].notna()].copy()
    print(f"匹配到gap的信号: {len(df_with_gap)}/{len(df)}")

    # ── 按阈值分析 ──
    thresholds = [None, 0, 1, 3, 5]  # None = 不过滤
    holding_periods = [3, 5, 10]
    strategies = df_with_gap["strategy"].unique()

    for hp in holding_periods:
        ret_col = f"ret_{hp}d"
        print(f"\n{'='*90}")
        print(f"  {hp}日持仓 — 不同跳空阈值下的收益对比")
        print(f"{'='*90}")

        # 表头
        header = f"{'策略':>12} |"
        for t in thresholds:
            label = "无限制" if t is None else f"<{t}%"
            header += f" {label:>20} |"
        print(header)
        print("-" * len(header))

        for strat in sorted(strategies):
            strat_df = df_with_gap[df_with_gap["strategy"] == strat]
            row = f"{strat:>12} |"

            for t in thresholds:
                if t is None:
                    filtered = strat_df
                else:
                    filtered = strat_df[strat_df["gap_pct"] < t]

                valid = filtered[filtered[ret_col].notna()]
                n = len(valid)
                if n == 0:
                    row += f" {'--':>20} |"
                    continue

                avg_ret = valid[ret_col].mean() * 100
                hit_rate = (valid[ret_col] > 0).mean() * 100
                row += f" {avg_ret:+.2f}% {hit_rate:.0f}% n={n:<4d} |"

            print(row)

    # ── 跳空分布统计 ──
    print(f"\n{'='*90}")
    print(f"  各策略选股的次日跳空分布")
    print(f"{'='*90}")
    print(f"{'策略':>12} | {'平均gap':>8} | {'中位gap':>8} | {'gap>0%':>8} | {'gap>3%':>8} | {'gap>5%':>8}")
    print("-" * 75)

    for strat in sorted(strategies):
        strat_df = df_with_gap[df_with_gap["strategy"] == strat]
        gap = strat_df["gap_pct"]
        avg_gap = gap.mean()
        med_gap = gap.median()
        pct_pos = (gap > 0).mean() * 100
        pct_3 = (gap > 3).mean() * 100
        pct_5 = (gap > 5).mean() * 100
        print(f"{strat:>12} | {avg_gap:+.2f}%  | {med_gap:+.2f}%  | {pct_pos:.1f}%   | {pct_3:.1f}%   | {pct_5:.1f}%")

    # ── 每个阈值的收益提升/下降百分比 ──
    print(f"\n{'='*90}")
    print(f"  过滤高开后的收益变化（vs 无限制基准）")
    print(f"{'='*90}")

    for hp in holding_periods:
        ret_col = f"ret_{hp}d"
        print(f"\n--- {hp}日持仓 ---")
        print(f"{'策略':>12} | {'<0% vs基准':>14} | {'<1% vs基准':>14} | {'<3% vs基准':>14} | {'<5% vs基准':>14}")
        print("-" * 80)

        for strat in sorted(strategies):
            strat_df = df_with_gap[df_with_gap["strategy"] == strat]
            base_valid = strat_df[strat_df[ret_col].notna()]
            if len(base_valid) == 0:
                continue
            base_ret = base_valid[ret_col].mean()

            row = f"{strat:>12} |"
            for t in [0, 1, 3, 5]:
                filtered = strat_df[strat_df["gap_pct"] < t]
                valid = filtered[filtered[ret_col].notna()]
                if len(valid) == 0 or base_ret == 0:
                    row += f" {'--':>14} |"
                    continue
                new_ret = valid[ret_col].mean()
                delta = (new_ret - base_ret) * 100  # 百分点变化
                pct_kept = len(valid) / len(base_valid) * 100
                row += f" {delta:+.2f}pp ({pct_kept:.0f}%) |"
            print(row)


if __name__ == "__main__":
    main()
