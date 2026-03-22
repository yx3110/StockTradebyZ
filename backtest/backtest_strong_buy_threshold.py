#!/usr/bin/env python3
"""
强烈买入阈值回测: 各模型只买"强烈买入"(composite >= strong_buy)的股票，统计实际收益

Composite公式 (按模型版本区分):
  - V4.6+: 0.6 * pred_10d + 0.4 * pred_15d  (rank_score, 与scorer一致)
  - 旧版本: pred_3d * 0.1 + pred_5d * 0.2 + pred_10d * 0.4 + pred_15d * 0.3

阈值来自各模型的 recommendation_thresholds.json (训练时基于历史百分位校准)
"""
import sys, os, json, sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

DB_PATH = os.path.join(PROJECT_ROOT, 'data_adapter', 'stock_data.db')

# Model configs: model_key -> (report_dirs_list, threshold_dir)
# 多目录合并: 先加载extended，再用最新报告补充缺失日期
# Model configs: model_key -> (report_dirs_list, threshold_dir, composite_mode)
# composite_mode: 'rank' = 0.6*10d + 0.4*15d (V4.6+), 'legacy' = 0.1*3d + 0.2*5d + 0.4*10d + 0.3*15d
MODEL_CONFIGS = {
    'V3.96': (['reports/daily_selection_v3.96_merged_extended',
               'reports/daily_selection_v3.96'], 'ml_models/trained_models/v396', 'legacy'),
    'V4.3':  (['reports/daily_selection_v4.3_merged_extended',
               'reports/daily_selection_v4.3'], 'ml_models/trained_models/v43', 'legacy'),
    'V4.4':  (['reports/daily_selection_v4.4_v2_merged_extended',
               'reports/daily_selection_v4.4_merged_extended',
               'reports/daily_selection_v4.4_bugfix',
               'reports/daily_selection_v4.4'], 'ml_models/trained_models/v44', 'legacy'),
    'V4.6':  (['reports/daily_selection_v4.6_merged_extended',
               'reports/daily_selection_v4.6'], 'ml_models/trained_models/v46', 'rank'),
    'V4.7.3':(['reports/daily_selection_v4.7.3_merged_extended',
               'reports/daily_selection_v4.7.3'], 'ml_models/trained_models/v473', 'rank'),
    'V4.8':  (['reports/daily_selection_v4.8'], 'ml_models/trained_models/v48', 'rank'),
    'V4.8.1':(['reports/daily_selection_v4.8.1'], 'ml_models/trained_models/v481', 'rank'),
}

HOLDING_DAYS = [1, 3, 5, 10, 15, 20]
TRANSACTION_COST = 0.00302  # 单边: 佣金+印花税+滑点


def load_thresholds(threshold_dir):
    path = Path(threshold_dir) / 'recommendation_thresholds.json'
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_reports_with_composite(report_dirs, composite_mode='legacy'):
    """加载报告, 计算composite, 返回 {date: [{code, composite, score, pred_Xd}, ...]}

    Args:
        report_dirs: 单个目录字符串或目录列表。多目录时后面的目录只补充缺失日期。
        composite_mode: 'rank' = 0.6*10d + 0.4*15d (V4.6+与scorer一致),
                        'legacy' = 0.1*3d + 0.2*5d + 0.4*10d + 0.3*15d (旧版本)
    """
    if isinstance(report_dirs, str):
        report_dirs = [report_dirs]

    reports = {}
    for report_dir in report_dirs:
        report_dir = Path(report_dir)
        if not report_dir.is_dir():
            continue
        for json_file in sorted(report_dir.glob('analysis_data_*.json')):
            date_str = json_file.stem.replace('analysis_data_', '')
            date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            if date in reports:
                continue  # 已有该日期数据，跳过
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                continue

            stocks = data.get('all_stocks_with_scores', [])
            if not stocks:
                continue

            stock_list = []
            for s in stocks:
                code = s.get('stock_code', '')
                if not code or len(code) != 6:
                    continue
                pred_3d = s.get('pred_3d', 0) or 0
                pred_5d = s.get('pred_5d', 0) or 0
                pred_10d = s.get('pred_10d', 0) or 0
                pred_15d = s.get('pred_15d')

                if composite_mode == 'rank':
                    # V4.6+: 与scorer的rank_score一致
                    composite = 0.6 * pred_10d + 0.4 * (pred_15d if pred_15d else 0)
                else:
                    # 旧版本: 4-weight composite
                    if pred_15d is not None and pred_15d != 0:
                        composite = pred_3d * 0.1 + pred_5d * 0.2 + pred_10d * 0.4 + pred_15d * 0.3
                    else:
                        composite = pred_3d * 0.15 + pred_5d * 0.25 + pred_10d * 0.60

                stock_list.append({
                    'code': code,
                    'composite': composite,
                    'score': s.get('score', 0),
                    'pred_3d': pred_3d,
                    'pred_5d': pred_5d,
                    'pred_10d': pred_10d,
                })
            if stock_list:
                reports[date] = stock_list
    return reports


def load_future_returns(dates, codes_by_date, max_hold=20):
    """从数据库批量加载未来收益"""
    conn = sqlite3.connect(DB_PATH)

    # 获取所有交易日
    all_dates = pd.read_sql("SELECT DISTINCT trade_date FROM daily_quotes ORDER BY trade_date",
                            conn)['trade_date'].tolist()
    date_set = set(all_dates)
    date_idx = {d: i for i, d in enumerate(all_dates)}

    # 收集所有需要的code
    all_codes = set()
    for codes in codes_by_date.values():
        all_codes.update(codes)

    if not all_codes:
        conn.close()
        return {}

    # 批量查询收盘价
    code_list_str = ','.join(f"'{c}'" for c in all_codes)
    min_date = min(dates)
    query = f"""
        SELECT s.code, dq.trade_date, dq.close
        FROM daily_quotes dq
        JOIN securities s ON s.id = dq.security_id
        WHERE s.code IN ({code_list_str})
          AND dq.trade_date >= '{min_date}'
        ORDER BY s.code, dq.trade_date
    """
    df = pd.read_sql(query, conn)
    conn.close()

    # 构建 {code: {date: close}} 查找表
    price_map = defaultdict(dict)
    for _, row in df.iterrows():
        price_map[row['code']][row['trade_date']] = row['close']

    # 计算未来N天收益
    results = {}  # {(date, code): {hold_days: return}}
    for date in dates:
        if date not in date_idx:
            continue
        idx = date_idx[date]
        codes = codes_by_date.get(date, [])
        for code in codes:
            returns = {}
            buy_price = price_map.get(code, {}).get(date)
            if buy_price is None or buy_price <= 0:
                continue
            for hd in HOLDING_DAYS:
                future_idx = idx + hd
                if future_idx >= len(all_dates):
                    continue
                future_date = all_dates[future_idx]
                future_price = price_map.get(code, {}).get(future_date)
                if future_price is not None and future_price > 0:
                    raw_ret = (future_price - buy_price) / buy_price
                    # 扣除双边交易成本
                    net_ret = raw_ret - 2 * TRANSACTION_COST
                    returns[hd] = net_ret
            if returns:
                results[(date, code)] = returns
    return results


def run_backtest():
    print("=" * 80)
    print("  强烈买入阈值回测: 各模型只买 composite >= strong_buy 阈值的股票")
    print("  V4.6+: composite = 0.6*10d + 0.4*15d (rank_score)")
    print("  旧版本: composite = 0.1*3d + 0.2*5d + 0.4*10d + 0.3*15d")
    print(f"  交易成本: {TRANSACTION_COST*100:.3f}% x2 (双边)")
    print("=" * 80)

    all_results = {}
    cached_stats = {}  # model_name -> {avg_strong, n_days, date_range}

    for model_name, (report_dirs, threshold_dir, composite_mode) in MODEL_CONFIGS.items():
        # 检查至少有一个报告目录存在
        valid_dirs = [d for d in report_dirs if os.path.isdir(d)]
        if not valid_dirs:
            print(f"\n{model_name}: 报告目录不存在, 跳过")
            continue

        thresholds = load_thresholds(threshold_dir)
        if not thresholds:
            print(f"\n{model_name}: 无recommendation_thresholds, 跳过")
            continue

        strong_buy_threshold = thresholds['strong_buy']
        buy_threshold = thresholds['buy']

        print(f"\n{'='*60}")
        print(f"  {model_name}")
        print(f"  强烈买入阈值: composite >= {strong_buy_threshold:.6f}")
        print(f"  买入阈值:     composite >= {buy_threshold:.6f}")
        print(f"  Composite模式: {composite_mode} ({'0.6*10d+0.4*15d' if composite_mode == 'rank' else '0.1*3d+0.2*5d+0.4*10d+0.3*15d'})")
        print(f"  报告来源: {', '.join(os.path.basename(d) for d in valid_dirs)}")
        print(f"{'='*60}")

        reports = load_reports_with_composite(valid_dirs, composite_mode=composite_mode)
        all_dates_sorted = sorted(reports.keys())
        date_range = f"{all_dates_sorted[0]}~{all_dates_sorted[-1]}" if all_dates_sorted else "N/A"
        print(f"  加载报告: {len(reports)} 天 ({date_range})")

        # 筛选强烈买入的股票
        strong_buy_by_date = {}
        buy_by_date = {}
        codes_needed = defaultdict(set)
        total_strong = 0
        total_buy = 0

        for date, stocks in reports.items():
            sb = [s for s in stocks if s['composite'] >= strong_buy_threshold]
            b = [s for s in stocks if s['composite'] >= buy_threshold]
            if sb:
                strong_buy_by_date[date] = sb
                total_strong += len(sb)
                for s in sb:
                    codes_needed[date].add(s['code'])
            if b:
                buy_by_date[date] = b
                total_buy += len(b)
                for s in b:
                    codes_needed[date].add(s['code'])

        dates_with_data = sorted(codes_needed.keys())
        n_strong_days = len(strong_buy_by_date)
        n_buy_days = len(buy_by_date)
        avg_strong = total_strong / max(n_strong_days, 1)
        avg_buy = total_buy / max(n_buy_days, 1)

        print(f"  强烈买入: {n_strong_days}天有信号, 共{total_strong}只, 日均{avg_strong:.1f}只")
        print(f"  买入:     {n_buy_days}天有信号, 共{total_buy}只, 日均{avg_buy:.1f}只")

        cached_stats[model_name] = {
            'avg_strong': avg_strong,
            'n_days': len(reports),
            'date_range': date_range,
        }

        if not dates_with_data:
            print(f"  无可回测数据")
            continue

        # 加载未来收益
        print(f"  加载价格数据...")
        future_returns = load_future_returns(dates_with_data, dict(codes_needed))

        # 统计各持仓天数的收益
        for level_name, level_data, level_threshold in [
            ('强烈买入', strong_buy_by_date, strong_buy_threshold),
            ('买入', buy_by_date, buy_threshold),
        ]:
            print(f"\n  --- {level_name} (composite >= {level_threshold:.6f}) ---")

            returns_by_hold = defaultdict(list)
            daily_returns = defaultdict(list)  # for computing Sharpe

            for date, stocks in sorted(level_data.items()):
                day_returns_by_hold = defaultdict(list)
                for s in stocks:
                    key = (date, s['code'])
                    if key in future_returns:
                        for hd, ret in future_returns[key].items():
                            returns_by_hold[hd].append(ret)
                            day_returns_by_hold[hd].append(ret)
                # 等权日均收益
                for hd, rets in day_returns_by_hold.items():
                    daily_returns[hd].append(np.mean(rets))

            if not returns_by_hold:
                print(f"    无价格数据")
                continue

            print(f"    {'持仓天数':>8} | {'样本数':>6} | {'平均收益':>8} | {'中位收益':>8} | {'胜率':>6} | {'年化收益':>10} | {'Sharpe':>7}")
            print(f"    {'-'*8:>8}-+-{'-'*6:>6}-+-{'-'*8:>8}-+-{'-'*8:>8}-+-{'-'*6:>6}-+-{'-'*10:>10}-+-{'-'*7:>7}")

            hold_results = {}
            for hd in HOLDING_DAYS:
                if hd not in returns_by_hold:
                    continue
                rets = np.array(returns_by_hold[hd])
                d_rets = np.array(daily_returns[hd])
                n = len(rets)
                mean_ret = np.mean(rets)
                median_ret = np.median(rets)
                win_rate = np.mean(rets > 0)

                # 年化: 非重叠累积NAV
                periods_per_year = 244.0 / hd
                non_overlap = rets[::max(hd, 1)]
                nav = np.cumprod(1 + non_overlap)
                total_years = n * hd / 244.0  # approximate
                annualized = nav[-1] ** (1 / total_years) - 1 if total_years > 0 and nav[-1] > 0 else 0

                # Sharpe: Newey-West adjusted for overlapping returns
                if len(d_rets) > 1:
                    d_mean = np.mean(d_rets)
                    d_demeaned = d_rets - d_mean
                    n_d = len(d_rets)
                    gamma0 = np.sum(d_demeaned**2) / n_d
                    nw_var = gamma0
                    for lag in range(1, min(hd, n_d)):
                        gamma_lag = np.sum(d_demeaned[lag:] * d_demeaned[:-lag]) / n_d
                        nw_var += 2 * (1 - lag / hd) * gamma_lag
                    nw_std = np.sqrt(max(nw_var, 0))
                    sharpe = d_mean / nw_std * np.sqrt(periods_per_year) if nw_std > 0 else 0
                else:
                    sharpe = 0

                hold_results[hd] = {
                    'n': n, 'mean': mean_ret, 'median': median_ret,
                    'win_rate': win_rate, 'annualized': annualized, 'sharpe': sharpe,
                }
                print(f"    {hd:>8}d | {n:>6} | {mean_ret:>+8.2%} | {median_ret:>+8.2%} | {win_rate:>6.1%} | {annualized:>+10.1%} | {sharpe:>7.2f}")

            all_results[(model_name, level_name)] = hold_results

    # Summary comparison table
    print(f"\n\n{'='*100}")
    print(f"  综合对比: 各模型 强烈买入 信号的10天持仓净收益")
    print(f"{'='*100}")
    print(f"  {'模型':>8} | {'日均只数':>8} | {'平均收益':>8} | {'中位收益':>8} | {'胜率':>6} | {'年化收益':>10} | {'Sharpe':>7} | 日期范围")
    print(f"  {'-'*8:>8}-+-{'-'*8:>8}-+-{'-'*8:>8}-+-{'-'*8:>8}-+-{'-'*6:>6}-+-{'-'*10:>10}-+-{'-'*7:>7}-+-{'-'*25}")

    for model_name in MODEL_CONFIGS:
        key = (model_name, '强烈买入')
        if key not in all_results or 10 not in all_results[key]:
            continue
        r = all_results[key][10]
        avg_n = cached_stats.get(model_name, {}).get('avg_strong', 0)
        n_days = cached_stats.get(model_name, {}).get('n_days', 0)
        date_range = cached_stats.get(model_name, {}).get('date_range', '')
        print(f"  {model_name:>8} | {avg_n:>8.1f} | {r['mean']:>+8.2%} | {r['median']:>+8.2%} | {r['win_rate']:>6.1%} | {r['annualized']:>+10.1%} | {r['sharpe']:>7.2f} | {date_range}")

    # 对比 Top-10 vs 强烈买入
    print(f"\n\n{'='*100}")
    print(f"  对比: 强烈买入 vs 买入 信号 (10天持仓)")
    print(f"{'='*100}")
    print(f"  {'模型':>8} | {'级别':>8} | {'平均收益':>8} | {'胜率':>6} | {'年化收益':>10} | {'Sharpe':>7}")
    print(f"  {'-'*8:>8}-+-{'-'*8:>8}-+-{'-'*8:>8}-+-{'-'*6:>6}-+-{'-'*10:>10}-+-{'-'*7:>7}")

    for model_name in MODEL_CONFIGS:
        for level in ['强烈买入', '买入']:
            key = (model_name, level)
            if key not in all_results or 10 not in all_results[key]:
                continue
            r = all_results[key][10]
            print(f"  {model_name:>8} | {level:>8} | {r['mean']:>+8.2%} | {r['win_rate']:>6.1%} | {r['annualized']:>+10.1%} | {r['sharpe']:>7.2f}")


if __name__ == '__main__':
    run_backtest()
