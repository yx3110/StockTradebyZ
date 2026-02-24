#!/usr/bin/env python3
"""
ML评分器快速回测
直接使用 V390/V395 ProductionScorer + v39_feature_cache 进行快速回测
跳过量化策略选股，纯ML评分排名 → top N 买入 → 持仓N天 → 统计收益

用法:
    python3 backtest/backtest_ml_scorer.py --version v3.9 --start-date 2025-09-01 --top-n 20
    python3 backtest/backtest_ml_scorer.py --version v3.95 --start-date 2025-09-01 --top-n 20
    python3 backtest/backtest_ml_scorer.py --version both --start-date 2025-09-01
"""
import sys
import os
import sqlite3
import argparse
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

DB_PATH = os.path.join(PROJECT_ROOT, 'data_adapter', 'stock_data.db')


def get_trading_dates(start_date, end_date):
    """获取有缓存数据的交易日列表"""
    conn = sqlite3.connect(DB_PATH)
    dates = [r[0] for r in conn.execute("""
        SELECT DISTINCT trade_date FROM v39_feature_cache
        WHERE trade_date >= ? AND trade_date <= ?
        ORDER BY trade_date
    """, (start_date, end_date)).fetchall()]
    conn.close()
    return dates


def get_all_stock_codes_for_date(date):
    """获取某日有缓存特征的所有A股代码（排除ETF/基金）"""
    conn = sqlite3.connect(DB_PATH)
    codes = [r[0] for r in conn.execute("""
        SELECT DISTINCT code FROM v39_feature_cache
        WHERE trade_date = ?
          AND code NOT LIKE '1%'
          AND code NOT LIKE '5%'
    """, (date,)).fetchall()]
    conn.close()
    return codes


def get_future_returns(codes, trade_date, holding_days_list=[1, 3, 5, 10]):
    """获取股票在trade_date之后N天的实际收益率"""
    conn = sqlite3.connect(DB_PATH)

    # 获取trade_date之后的交易日
    future_dates = [r[0] for r in conn.execute("""
        SELECT DISTINCT trade_date FROM daily_quotes
        WHERE trade_date > ?
        ORDER BY trade_date
        LIMIT ?
    """, (trade_date, max(holding_days_list) + 1)).fetchall()]

    if not future_dates:
        conn.close()
        return {}

    # 获取当日收盘价
    placeholders = ','.join(['?' for _ in codes])
    today_prices = {}
    rows = conn.execute(f"""
        SELECT s.code, dq.close
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.code IN ({placeholders}) AND dq.trade_date = ?
    """, list(codes) + [trade_date]).fetchall()
    for code, close in rows:
        today_prices[code] = close

    # 获取未来各天收盘价
    results = {}
    for days in holding_days_list:
        if days - 1 >= len(future_dates):
            continue
        future_date = future_dates[days - 1]

        future_rows = conn.execute(f"""
            SELECT s.code, dq.close
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE s.code IN ({placeholders}) AND dq.trade_date = ?
        """, list(codes) + [future_date]).fetchall()

        for code, future_close in future_rows:
            if code in today_prices and today_prices[code] > 0:
                ret = (future_close - today_prices[code]) / today_prices[code]
                if code not in results:
                    results[code] = {}
                results[code][f'return_{days}d'] = ret

    conn.close()
    return results


def run_backtest(version, start_date, end_date, top_n=20, holding_days=5):
    """运行ML评分器回测"""
    print(f"\n{'='*80}")
    print(f"  ML评分器回测: {version}")
    print(f"  日期范围: {start_date} ~ {end_date}")
    print(f"  Top N: {top_n}, 持仓天数: {holding_days}")
    print(f"{'='*80}\n")

    # 初始化评分器
    if version == 'v3.9':
        from ml_models.v39.v390_production_scorer import V390ProductionScorer
        scorer = V390ProductionScorer()
    elif version == 'v3.95':
        from ml_models.v39.v395_production_scorer import V395ProductionScorer
        scorer = V395ProductionScorer(model_type='small_data')
    else:
        raise ValueError(f"不支持的版本: {version}")

    trading_dates = get_trading_dates(start_date, end_date)
    print(f"交易日数: {len(trading_dates)}")

    # 回测结果
    daily_results = []
    all_picks = []

    t0 = time.time()

    for i, date in enumerate(trading_dates):
        # 获取当日所有股票
        codes = get_all_stock_codes_for_date(date)
        if not codes:
            continue

        # 批量评分
        if version == 'v3.9':
            scores = scorer.predict_scores(codes, date)
        elif version == 'v3.95':
            scores = scorer.predict_scores(codes, date)

        if not scores:
            continue

        # 按分数排序取 top N
        scored_list = [(code, info.get('score', 0)) for code, info in scores.items()]
        scored_list.sort(key=lambda x: x[1], reverse=True)
        top_stocks = scored_list[:top_n]
        bottom_stocks = scored_list[-top_n:]

        # 获取实际收益
        top_codes = [c for c, _ in top_stocks]
        bottom_codes = [c for c, _ in bottom_stocks]
        all_codes = list(set(top_codes + bottom_codes))

        future_returns = get_future_returns(all_codes, date, [1, 3, 5, 10])

        # 计算 top N 组合平均收益
        for days in [1, 3, 5, 10]:
            key = f'return_{days}d'
            top_returns = [future_returns.get(c, {}).get(key, 0) for c in top_codes
                          if key in future_returns.get(c, {})]
            bottom_returns = [future_returns.get(c, {}).get(key, 0) for c in bottom_codes
                             if key in future_returns.get(c, {})]

            if top_returns:
                avg_top = np.mean(top_returns)
                avg_bottom = np.mean(bottom_returns) if bottom_returns else 0

                daily_results.append({
                    'date': date,
                    'days': days,
                    'avg_top_return': avg_top,
                    'avg_bottom_return': avg_bottom,
                    'spread': avg_top - avg_bottom,
                    'top_positive_pct': np.mean([r > 0 for r in top_returns]),
                    'n_top': len(top_returns),
                    'n_bottom': len(bottom_returns)
                })

        # 记录选股
        for code, score in top_stocks:
            pick = {
                'date': date,
                'code': code,
                'score': score,
            }
            for days in [1, 3, 5, 10]:
                key = f'return_{days}d'
                pick[key] = future_returns.get(code, {}).get(key, None)
            all_picks.append(pick)

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (len(trading_dates) - i - 1) / rate if rate > 0 else 0

        if (i + 1) % 10 == 0 or i == 0:
            top_score = top_stocks[0][1] if top_stocks else 0
            print(f"  [{i+1}/{len(trading_dates)}] {date}: "
                  f"{len(codes)} stocks scored, top={top_score:.1f}, "
                  f"({rate:.1f} dates/s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"\n评分完成: {len(trading_dates)} 天, 耗时 {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")

    # 汇总统计
    if not daily_results:
        print("无回测结果!")
        return None

    df = pd.DataFrame(daily_results)
    picks_df = pd.DataFrame(all_picks)

    print(f"\n{'='*80}")
    print(f"  回测结果汇总: {version}")
    print(f"{'='*80}")

    for days in [1, 3, 5, 10]:
        sub = df[df['days'] == days]
        if len(sub) == 0:
            continue

        avg_top = sub['avg_top_return'].mean() * 100
        avg_bottom = sub['avg_bottom_return'].mean() * 100
        avg_spread = sub['spread'].mean() * 100
        win_rate = (sub['avg_top_return'] > 0).mean() * 100
        avg_positive_pct = sub['top_positive_pct'].mean() * 100

        # IC (Information Coefficient): 评分与收益的Spearman相关性
        from scipy.stats import spearmanr
        sub_picks = picks_df[picks_df[f'return_{days}d'].notna()]
        if len(sub_picks) > 10:
            ic, p_val = spearmanr(sub_picks['score'], sub_picks[f'return_{days}d'])
        else:
            ic, p_val = 0, 1

        # Top N 累计收益 (假设每天等权重买入top N持仓days天)
        cumulative = (1 + sub['avg_top_return']).prod() - 1

        print(f"\n  📊 {days}日持仓:")
        print(f"    Top{top_n} 平均收益:  {avg_top:+.3f}%")
        print(f"    Bottom{top_n} 平均收益: {avg_bottom:+.3f}%")
        print(f"    多空价差:           {avg_spread:+.3f}%")
        print(f"    Top{top_n} 盈利天数比:  {win_rate:.1f}%")
        print(f"    Top{top_n} 内盈利股比:  {avg_positive_pct:.1f}%")
        print(f"    IC (Spearman):      {ic:.4f} (p={p_val:.4f})")
        print(f"    累计收益 (等权):    {cumulative*100:+.2f}%")

    # 保存结果
    report_dir = Path(f'reports/backtest')
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 保存选股明细
    picks_file = report_dir / f'ml_backtest_{version.replace(".", "")}_{timestamp}_picks.csv'
    picks_df.to_csv(picks_file, index=False, encoding='utf-8-sig')

    # 生成报告
    report_lines = [
        f"# ML评分器回测报告 - {version}",
        f"",
        f"## 回测参数",
        f"- **评分版本**: {version}",
        f"- **日期范围**: {start_date} ~ {end_date}",
        f"- **交易日数**: {len(trading_dates)}",
        f"- **Top N**: {top_n}",
        f"- **回测时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## 回测结果",
        f"",
        f"| 持仓天数 | Top{top_n}均收益 | Bottom{top_n}均收益 | 多空价差 | 盈利天数比 | IC | 累计收益 |",
        f"|:--------:|:----------:|:------------:|:--------:|:--------:|:---:|:--------:|",
    ]

    for days in [1, 3, 5, 10]:
        sub = df[df['days'] == days]
        if len(sub) == 0:
            continue
        avg_top = sub['avg_top_return'].mean() * 100
        avg_bottom = sub['avg_bottom_return'].mean() * 100
        avg_spread = sub['spread'].mean() * 100
        win_rate = (sub['avg_top_return'] > 0).mean() * 100

        sub_picks = picks_df[picks_df[f'return_{days}d'].notna()]
        if len(sub_picks) > 10:
            ic, _ = spearmanr(sub_picks['score'], sub_picks[f'return_{days}d'])
        else:
            ic = 0

        cumulative = (1 + sub['avg_top_return']).prod() - 1

        report_lines.append(
            f"| {days}d | {avg_top:+.3f}% | {avg_bottom:+.3f}% | "
            f"{avg_spread:+.3f}% | {win_rate:.1f}% | {ic:.4f} | {cumulative*100:+.2f}% |"
        )

    # 月度收益统计
    report_lines.extend([
        f"",
        f"## 月度收益 (5日持仓)",
        f"",
        f"| 月份 | Top{top_n}均收益 | 盈利天数比 | 选股天数 |",
        f"|:----:|:----------:|:--------:|:------:|",
    ])

    sub5 = df[df['days'] == 5].copy()
    if len(sub5) > 0:
        sub5['month'] = pd.to_datetime(sub5['date']).dt.to_period('M')
        for month, group in sub5.groupby('month'):
            monthly_ret = group['avg_top_return'].mean() * 100
            monthly_win = (group['avg_top_return'] > 0).mean() * 100
            report_lines.append(
                f"| {month} | {monthly_ret:+.3f}% | {monthly_win:.1f}% | {len(group)} |"
            )

    # Top 10 最佳/最差选股日
    report_lines.extend([
        f"",
        f"## Top 10 最佳选股日 (5日持仓)",
        f"",
    ])
    if len(sub5) > 0:
        best_days = sub5.nlargest(10, 'avg_top_return')
        for _, row in best_days.iterrows():
            report_lines.append(f"- {row['date']}: {row['avg_top_return']*100:+.2f}%")

    report_lines.extend([
        f"",
        f"## Top 10 最差选股日 (5日持仓)",
        f"",
    ])
    if len(sub5) > 0:
        worst_days = sub5.nsmallest(10, 'avg_top_return')
        for _, row in worst_days.iterrows():
            report_lines.append(f"- {row['date']}: {row['avg_top_return']*100:+.2f}%")

    report = '\n'.join(report_lines)
    report_file = report_dir / f'ml_backtest_{version.replace(".", "")}_{timestamp}.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n📊 回测报告: {report_file}")
    print(f"📊 选股明细: {picks_file}")

    return {
        'version': version,
        'daily_results': df,
        'picks': picks_df,
        'report_file': str(report_file)
    }


def main():
    parser = argparse.ArgumentParser(description='ML评分器快速回测')
    parser.add_argument('--version', default='v3.9', choices=['v3.9', 'v3.95', 'both'])
    parser.add_argument('--start-date', default='2025-09-01')
    parser.add_argument('--end-date', default='2026-02-13')
    parser.add_argument('--top-n', type=int, default=20)
    parser.add_argument('--holding-days', type=int, default=5)
    args = parser.parse_args()

    versions = ['v3.9', 'v3.95'] if args.version == 'both' else [args.version]

    results = {}
    for ver in versions:
        result = run_backtest(ver, args.start_date, args.end_date, args.top_n, args.holding_days)
        if result:
            results[ver] = result

    if len(results) == 2:
        print(f"\n{'='*80}")
        print(f"  模型对比")
        print(f"{'='*80}")
        for days in [1, 3, 5, 10]:
            print(f"\n  {days}日持仓:")
            for ver in versions:
                df = results[ver]['daily_results']
                sub = df[df['days'] == days]
                if len(sub) > 0:
                    avg = sub['avg_top_return'].mean() * 100
                    spread = sub['spread'].mean() * 100
                    print(f"    {ver}: Top20 {avg:+.3f}%, 价差 {spread:+.3f}%")


if __name__ == '__main__':
    main()
