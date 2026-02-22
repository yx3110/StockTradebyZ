#!/usr/bin/env python3
"""
基于报告的回测引擎
读取选股报告JSON文件，提取Top N股票，计算实际收益，评估模型表现

用法:
    # 回测新模型v3.9
    python3 backtest/backtest_report_based.py --report-dir reports/daily_selection_v3.9_model20260222 --label "v3.9新模型"

    # 对比新旧模型
    python3 backtest/backtest_report_based.py \
        --report-dir reports/daily_selection_v3.9_model20260222 \
        --compare-dir reports/daily_selection_v3.9 \
        --label "v3.9新模型" --compare-label "v3.9旧模型"

    # 对比v3.9 vs v3.95 (新模型)
    python3 backtest/backtest_report_based.py \
        --report-dir reports/daily_selection_v3.9_model20260222 \
        --compare-dir reports/daily_selection_v3.95_model20260221 \
        --label "v3.9新模型" --compare-label "v3.95新模型"

    # 四模型全面对比
    python3 backtest/backtest_report_based.py --all
"""
import sys
import os
import sqlite3
import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from scipy.stats import spearmanr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

DB_PATH = os.path.join(PROJECT_ROOT, 'data_adapter', 'stock_data.db')

HOLDING_DAYS = [1, 3, 5, 10, 15]


def load_reports(report_dir):
    """加载所有JSON报告，返回 {date: [{code, score, predicted_return_5d}, ...]}"""
    report_dir = Path(report_dir)
    reports = {}

    for json_file in sorted(report_dir.glob('analysis_data_*.json')):
        date_str = json_file.stem.replace('analysis_data_', '')
        # 转为 YYYY-MM-DD 格式
        date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  跳过 {json_file.name}: {e}")
            continue

        stocks = data.get('all_stocks_with_scores', [])
        if not stocks:
            continue

        stock_list = []
        for s in stocks:
            code = s.get('stock_code', '')
            score = s.get('score', 0)
            pred_ret = s.get('predicted_return_5d', None)
            if code and score > 0:
                stock_list.append({
                    'code': code,
                    'score': score,
                    'predicted_return_5d': pred_ret,
                    'strategies': s.get('strategies', []),
                    'n_strategies': s.get('selected_by_strategies', 1),
                })

        if stock_list:
            # 按分数排序
            stock_list.sort(key=lambda x: x['score'], reverse=True)
            reports[date] = stock_list

    return reports


def get_next_trading_date(trade_date):
    """获取下一个交易日（报告日期的次日开盘买入）"""
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("""
        SELECT DISTINCT trade_date FROM daily_quotes
        WHERE trade_date > ?
        ORDER BY trade_date
        LIMIT 1
    """, (trade_date,)).fetchone()
    conn.close()
    return row[0] if row else None


def get_future_returns(codes, buy_date, holding_days_list=None):
    if holding_days_list is None:
        holding_days_list = HOLDING_DAYS
    """获取买入日买入后N天的实际收益率（以买入日开盘价买入，持仓N天后收盘价卖出）"""
    conn = sqlite3.connect(DB_PATH)

    # 获取buy_date当天及之后的交易日
    future_dates = [r[0] for r in conn.execute("""
        SELECT DISTINCT trade_date FROM daily_quotes
        WHERE trade_date >= ?
        ORDER BY trade_date
        LIMIT ?
    """, (buy_date, max(holding_days_list) + 2)).fetchall()]

    if not future_dates or future_dates[0] != buy_date:
        conn.close()
        return {}

    codes_str = ','.join([f"'{c}'" for c in codes])

    # 获取买入日开盘价（次日开盘买入）
    buy_prices = {}
    rows = conn.execute(f"""
        SELECT s.code, dq.open
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.code IN ({codes_str}) AND dq.trade_date = ?
    """, (buy_date,)).fetchall()
    for code, open_price in rows:
        if open_price and open_price > 0:
            buy_prices[code] = open_price

    # 获取未来各天收盘价
    results = {}
    for days in holding_days_list:
        idx = days  # future_dates[0] = buy_date, future_dates[days] = buy+days天
        if idx >= len(future_dates):
            continue
        sell_date = future_dates[idx]

        sell_rows = conn.execute(f"""
            SELECT s.code, dq.close
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE s.code IN ({codes_str}) AND dq.trade_date = ?
        """, (sell_date,)).fetchall()

        for code, close in sell_rows:
            if code in buy_prices and buy_prices[code] > 0:
                ret = (close - buy_prices[code]) / buy_prices[code]
                if code not in results:
                    results[code] = {}
                results[code][f'return_{days}d'] = ret

    conn.close()
    return results


def run_single_backtest(reports, label, top_n=20):
    """运行单个报告目录的回测"""
    print(f"\n{'='*80}")
    print(f"  报告回测: {label}")
    print(f"  报告天数: {len(reports)}, Top N: {top_n}")
    print(f"{'='*80}\n")

    daily_results = []
    all_picks = []
    skipped = 0

    dates = sorted(reports.keys())
    for i, date in enumerate(dates):
        stocks = reports[date]
        top_stocks = stocks[:top_n]
        bottom_stocks = stocks[-top_n:] if len(stocks) >= top_n * 2 else stocks[-(len(stocks)//2):]

        # 买入日 = 报告日的下一个交易日
        buy_date = get_next_trading_date(date)
        if not buy_date:
            skipped += 1
            continue

        top_codes = [s['code'] for s in top_stocks]
        bottom_codes = [s['code'] for s in bottom_stocks]
        # 查询所有候选股票的收益（用于逐日IC计算）
        all_codes = list(set([s['code'] for s in stocks]))

        future_returns = get_future_returns(all_codes, buy_date, HOLDING_DAYS)

        if not future_returns:
            skipped += 1
            continue

        for days in HOLDING_DAYS:
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
                    'buy_date': buy_date,
                    'days': days,
                    'avg_top_return': avg_top,
                    'avg_bottom_return': avg_bottom,
                    'spread': avg_top - avg_bottom,
                    'top_positive_pct': np.mean([r > 0 for r in top_returns]),
                    'n_top': len(top_returns),
                    'n_bottom': len(bottom_returns),
                    'n_total_stocks': len(stocks),
                })

        # 记录所有候选股票明细（用于逐日IC计算）
        top_code_set = set(top_codes)
        for s in stocks:
            pick = {
                'date': date,
                'buy_date': buy_date,
                'code': s['code'],
                'score': s['score'],
                'predicted_return_5d': s.get('predicted_return_5d'),
                'n_strategies': s.get('n_strategies', 1),
                'is_top': s['code'] in top_code_set,
            }
            for days in HOLDING_DAYS:
                key = f'return_{days}d'
                pick[key] = future_returns.get(s['code'], {}).get(key, None)
            all_picks.append(pick)

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(dates)}] {date} → 买入{buy_date}: "
                  f"{len(stocks)}只候选, top{min(top_n, len(stocks))}只")

    if skipped:
        print(f"  跳过 {skipped} 天（无交易数据）")

    if not daily_results:
        print("  无回测结果!")
        return None

    df = pd.DataFrame(daily_results)
    picks_df = pd.DataFrame(all_picks)

    # 汇总打印
    print(f"\n{'─'*70}")
    print(f"  {label} 回测结果")
    print(f"{'─'*70}")

    summary = {}
    daily_ic_series = {}  # {days: DataFrame with date, ic}

    for days in HOLDING_DAYS:
        sub = df[df['days'] == days]
        if len(sub) == 0:
            continue

        avg_top = sub['avg_top_return'].mean() * 100
        avg_bottom = sub['avg_bottom_return'].mean() * 100
        avg_spread = sub['spread'].mean() * 100
        win_rate = (sub['avg_top_return'] > 0).mean() * 100
        avg_positive_pct = sub['top_positive_pct'].mean() * 100

        # 全局IC
        sub_picks = picks_df[picks_df[f'return_{days}d'].notna()]
        if len(sub_picks) > 10:
            ic, p_val = spearmanr(sub_picks['score'], sub_picks[f'return_{days}d'])
        else:
            ic, p_val = 0, 1

        # 逐日IC序列
        ic_records = []
        for date in sorted(picks_df['date'].unique()):
            day_picks = picks_df[(picks_df['date'] == date) & (picks_df[f'return_{days}d'].notna())]
            if len(day_picks) >= 5:
                day_ic, day_p = spearmanr(day_picks['score'], day_picks[f'return_{days}d'])
                ic_records.append({'date': date, 'ic': day_ic, 'p_val': day_p, 'n_stocks': len(day_picks)})

        ic_df = pd.DataFrame(ic_records) if ic_records else pd.DataFrame()
        daily_ic_series[days] = ic_df

        # ICIR = mean(daily_IC) / std(daily_IC)
        if len(ic_df) > 5:
            ic_mean = ic_df['ic'].mean()
            ic_std = ic_df['ic'].std()
            icir = ic_mean / ic_std if ic_std > 0 else 0
            ic_positive_pct = (ic_df['ic'] > 0).mean() * 100
        else:
            ic_mean, ic_std, icir, ic_positive_pct = ic, 0, 0, 0

        # 累计收益
        cumulative = (1 + sub['avg_top_return']).prod() - 1

        summary[days] = {
            'avg_top': avg_top,
            'avg_bottom': avg_bottom,
            'spread': avg_spread,
            'win_rate': win_rate,
            'positive_pct': avg_positive_pct,
            'ic': ic,
            'ic_p': p_val,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'icir': icir,
            'ic_positive_pct': ic_positive_pct,
            'cumulative': cumulative * 100,
            'n_days': len(sub),
            'n_ic_days': len(ic_df),
        }

        print(f"\n  📊 {days}日持仓 ({len(sub)}天):")
        print(f"    Top{top_n} 日均收益:    {avg_top:+.3f}%")
        print(f"    Bottom 日均收益:      {avg_bottom:+.3f}%")
        print(f"    多空价差:             {avg_spread:+.3f}%")
        print(f"    Top{top_n} 盈利天数占比: {win_rate:.1f}%")
        print(f"    Top{top_n} 内盈利股占比: {avg_positive_pct:.1f}%")
        print(f"    全局IC (Spearman):    {ic:.4f} (p={p_val:.4f})")
        print(f"    逐日IC均值:           {ic_mean:.4f} ± {ic_std:.4f}")
        print(f"    ICIR:                 {icir:.4f}")
        print(f"    IC>0天数占比:         {ic_positive_pct:.1f}% ({len(ic_df)}天)")

    # 月度分解 (5日持仓)
    sub5 = df[df['days'] == 5].copy()
    if len(sub5) > 0:
        print(f"\n  📅 月度收益 (5日持仓):")
        sub5['month'] = pd.to_datetime(sub5['date']).dt.to_period('M')
        for month, group in sub5.groupby('month'):
            monthly_ret = group['avg_top_return'].mean() * 100
            monthly_win = (group['avg_top_return'] > 0).mean() * 100
            n_days = len(group)
            print(f"    {month}: {monthly_ret:+.3f}% (盈利{monthly_win:.0f}%, {n_days}天)")

    # 月度IC分解
    for days in [5, 10]:
        ic_df = daily_ic_series.get(days)
        if ic_df is not None and len(ic_df) > 0:
            print(f"\n  📅 月度IC ({days}日持仓):")
            ic_df_copy = ic_df.copy()
            ic_df_copy['month'] = pd.to_datetime(ic_df_copy['date']).dt.to_period('M')
            for month, group in ic_df_copy.groupby('month'):
                m_ic = group['ic'].mean()
                m_std = group['ic'].std()
                m_icir = m_ic / m_std if m_std > 0 else 0
                m_pos = (group['ic'] > 0).mean() * 100
                print(f"    {month}: IC={m_ic:+.4f} ±{m_std:.4f}, ICIR={m_icir:+.3f}, IC>0={m_pos:.0f}% ({len(group)}天)")

    return {
        'label': label,
        'summary': summary,
        'daily_results': df,
        'picks': picks_df,
        'daily_ic_series': daily_ic_series,
    }


def compare_results(result_a, result_b):
    """对比两个回测结果"""
    label_a = result_a['label']
    label_b = result_b['label']

    print(f"\n{'='*80}")
    print(f"  模型对比: {label_a} vs {label_b}")
    print(f"{'='*80}")

    header = f"| 指标 | {label_a} | {label_b} | 差异 | 优胜 |"
    sep = f"|:-----|:----------:|:----------:|:------:|:----:|"

    for days in HOLDING_DAYS:
        sa = result_a['summary'].get(days)
        sb = result_b['summary'].get(days)
        if not sa or not sb:
            continue

        print(f"\n  📊 {days}日持仓对比:")
        print(f"  {header}")
        print(f"  {sep}")

        metrics = [
            ('日均收益', 'avg_top', '%', True),
            ('多空价差', 'spread', '%', True),
            ('盈利天数比', 'win_rate', '%', True),
            ('盈利股占比', 'positive_pct', '%', True),
            ('全局IC', 'ic', '', True),
            ('逐日IC均值', 'ic_mean', '', True),
            ('ICIR', 'icir', '', True),
            ('IC>0占比', 'ic_positive_pct', '%', True),
            ('累计收益', 'cumulative', '%', True),
        ]

        for name, key, unit, higher_better in metrics:
            va = sa[key]
            vb = sb[key]
            diff = va - vb
            if higher_better:
                winner = label_a if va > vb else label_b if vb > va else "平"
            else:
                winner = label_a if va < vb else label_b if vb < va else "平"

            fmt = '+.3f' if key in ('avg_top', 'spread', 'cumulative') else '.1f' if key in ('win_rate', 'positive_pct') else '.4f'
            print(f"  | {name} | {va:{fmt}}{unit} | {vb:{fmt}}{unit} | {diff:+.3f} | {winner} |")


def generate_report(results, output_dir='reports/backtest'):
    """生成Markdown回测报告"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    labels = [r['label'] for r in results]
    label_str = '_vs_'.join([l.replace(' ', '').replace('.', '') for l in labels])

    report_lines = [
        f"# 选股报告回测对比",
        f"",
        f"**回测时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## 回测参数",
        f"",
    ]

    for r in results:
        summary = r['summary']
        n_days = summary.get(5, {}).get('n_days', 0) if summary else 0
        report_lines.append(f"- **{r['label']}**: {n_days} 交易日")

    report_lines.extend([
        f"",
        f"## 综合对比",
        f"",
    ])

    for days in HOLDING_DAYS:
        report_lines.extend([
            f"### {days}日持仓",
            f"",
            f"| 模型 | 日均收益 | 多空价差 | 盈利天数比 | 盈利股占比 | 全局IC | 逐日IC均值 | ICIR | IC>0占比 | 累计收益 |",
            f"|:-----|:--------:|:--------:|:----------:|:----------:|:------:|:----------:|:----:|:--------:|:--------:|",
        ])

        for r in results:
            s = r['summary'].get(days)
            if not s:
                continue
            report_lines.append(
                f"| {r['label']} | {s['avg_top']:+.3f}% | {s['spread']:+.3f}% | "
                f"{s['win_rate']:.1f}% | {s['positive_pct']:.1f}% | "
                f"{s['ic']:.4f} | {s.get('ic_mean', 0):.4f}±{s.get('ic_std', 0):.4f} | "
                f"{s.get('icir', 0):.4f} | {s.get('ic_positive_pct', 0):.1f}% | "
                f"{s['cumulative']:+.2f}% |"
            )
        report_lines.append("")

    # 月度对比 (5日持仓)
    report_lines.extend([
        f"## 月度收益对比 (5日持仓)",
        f"",
    ])

    months_header = "| 月份 |"
    months_sep = "|:----:|"
    for r in results:
        months_header += f" {r['label']} |"
        months_sep += ":----------:|"
    report_lines.append(months_header)
    report_lines.append(months_sep)

    # 收集所有月份
    all_months = set()
    monthly_data = {}
    for r in results:
        df = r['daily_results']
        sub5 = df[df['days'] == 5].copy()
        if len(sub5) > 0:
            sub5['month'] = pd.to_datetime(sub5['date']).dt.to_period('M')
            for month, group in sub5.groupby('month'):
                all_months.add(month)
                if month not in monthly_data:
                    monthly_data[month] = {}
                monthly_data[month][r['label']] = group['avg_top_return'].mean() * 100

    for month in sorted(all_months):
        row = f"| {month} |"
        for r in results:
            val = monthly_data.get(month, {}).get(r['label'])
            if val is not None:
                row += f" {val:+.3f}% |"
            else:
                row += " - |"
        report_lines.append(row)

    report_lines.append("")

    # 保存
    report_file = output_dir / f'report_backtest_{label_str}_{timestamp}.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    # 保存选股明细CSV
    for r in results:
        csv_file = output_dir / f'report_backtest_{r["label"].replace(" ", "").replace(".", "")}_{timestamp}_picks.csv'
        r['picks'].to_csv(csv_file, index=False, encoding='utf-8-sig')

    print(f"\n📊 回测报告: {report_file}")
    return str(report_file)


def main():
    parser = argparse.ArgumentParser(description='基于报告的回测引擎')
    parser.add_argument('--report-dir', help='主报告目录')
    parser.add_argument('--compare-dir', help='对比报告目录')
    parser.add_argument('--label', default='模型A', help='主报告标签')
    parser.add_argument('--compare-label', default='模型B', help='对比报告标签')
    parser.add_argument('--top-n', type=int, default=20, help='每日选取Top N只 (default: 20)')
    parser.add_argument('--all', action='store_true', help='四模型全面对比')
    args = parser.parse_args()

    if args.all:
        # 四模型全面对比
        configs = [
            ('reports/daily_selection_v3.9', 'v3.9旧模型'),
            ('reports/daily_selection_v3.9_model20260222', 'v3.9新模型'),
            ('reports/daily_selection_v3.95', 'v3.95旧模型'),
            ('reports/daily_selection_v3.95_model20260221', 'v3.95新模型'),
        ]

        results = []
        for dir_path, label in configs:
            if not Path(dir_path).exists():
                print(f"  跳过 {label}: {dir_path} 不存在")
                continue
            reports = load_reports(dir_path)
            print(f"加载 {label}: {len(reports)} 天报告")
            result = run_single_backtest(reports, label, args.top_n)
            if result:
                results.append(result)

        if len(results) >= 2:
            # 两两对比
            for i in range(len(results)):
                for j in range(i + 1, len(results)):
                    compare_results(results[i], results[j])

            generate_report(results)

    elif args.report_dir:
        reports_a = load_reports(args.report_dir)
        print(f"加载 {args.label}: {len(reports_a)} 天报告")

        result_a = run_single_backtest(reports_a, args.label, args.top_n)

        results = [result_a] if result_a else []

        if args.compare_dir:
            reports_b = load_reports(args.compare_dir)
            print(f"加载 {args.compare_label}: {len(reports_b)} 天报告")
            result_b = run_single_backtest(reports_b, args.compare_label, args.top_n)

            if result_a and result_b:
                results.append(result_b)
                compare_results(result_a, result_b)

        if results:
            generate_report(results)
    else:
        parser.print_help()
        print("\n示例:")
        print("  python3 backtest/backtest_report_based.py --all")
        print("  python3 backtest/backtest_report_based.py --report-dir reports/daily_selection_v3.9_model20260222 --label 'v3.9新模型'")


if __name__ == '__main__':
    main()
