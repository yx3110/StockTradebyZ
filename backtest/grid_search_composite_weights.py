#!/usr/bin/env python3
"""Fast grid search for optimal composite ranking weights (pred_3d/5d/10d/15d).

Preloads all JSON report data + price data once, then re-ranks for each weight combo.
Computes IC, annual return, Sharpe, MaxDD WITHOUT full backtest engine overhead.
"""

import sys
import json
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, rankdata

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')


def load_raw_data(report_dir):
    """Load all JSON reports into a flat structure: {date: [{code, pred_3d, ..., score}, ...]}"""
    report_dir = Path(report_dir)
    all_data = {}

    for json_file in sorted(report_dir.glob('analysis_data_*.json')):
        date_str = json_file.stem.replace('analysis_data_', '')
        date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

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
            if not code or s.get('score', 0) <= 0:
                continue
            stock_list.append({
                'code': code,
                'pred_3d': s.get('pred_3d', 0) or 0,
                'pred_5d': s.get('pred_5d', 0) or 0,
                'pred_10d': s.get('pred_10d', 0) or 0,
                'pred_15d': s.get('pred_15d', 0) or 0,
                'score': s.get('score', 0),
            })

        if stock_list:
            all_data[date] = stock_list

    return all_data


def load_price_data(dates, all_codes):
    """Load forward returns for all dates and codes from DB."""
    conn = sqlite3.connect(DB_PATH)

    # Get all trading dates
    all_trade_dates = pd.read_sql(
        "SELECT DISTINCT trade_date FROM daily_quotes ORDER BY trade_date",
        conn
    )['trade_date'].tolist()
    date_idx = {d: i for i, d in enumerate(all_trade_dates)}

    # Load price data for relevant codes
    min_date = min(dates)
    max_date = max(dates)
    # Need prices up to ~15 trading days after max_date
    max_idx = date_idx.get(max_date, len(all_trade_dates)-1)
    end_idx = min(max_idx + 20, len(all_trade_dates)-1)
    end_date = all_trade_dates[end_idx]

    # Get security_id mapping
    code_list = list(all_codes)
    placeholders = ','.join(['?'] * len(code_list))
    sec_df = pd.read_sql(
        f"SELECT id, code FROM securities WHERE code IN ({placeholders})",
        conn, params=code_list
    )
    sec_id_map = dict(zip(sec_df['code'], sec_df['id']))
    id_code_map = dict(zip(sec_df['id'], sec_df['code']))

    sec_ids = list(sec_id_map.values())
    if not sec_ids:
        conn.close()
        return {}, all_trade_dates, date_idx

    placeholders_ids = ','.join(['?'] * len(sec_ids))
    price_df = pd.read_sql(
        f"""SELECT security_id, trade_date, close, price_change_pct
            FROM daily_quotes
            WHERE trade_date >= ? AND trade_date <= ?
            AND security_id IN ({placeholders_ids})""",
        conn, params=[min_date, end_date] + sec_ids
    )
    conn.close()

    # Build price lookup: {code: {date: close}}
    price_lookup = {}
    for _, row in price_df.iterrows():
        code = id_code_map.get(row['security_id'])
        if code:
            if code not in price_lookup:
                price_lookup[code] = {}
            price_lookup[code][row['trade_date']] = row['close']

    return price_lookup, all_trade_dates, date_idx


def get_forward_return(code, date, hold_days, price_lookup, all_dates, date_idx):
    """Get hold_days forward return for a stock."""
    prices = price_lookup.get(code)
    if not prices:
        return None

    idx = date_idx.get(date)
    if idx is None:
        return None

    # Buy at next trading day's close (T+1)
    buy_idx = idx + 1
    sell_idx = buy_idx + hold_days

    if buy_idx >= len(all_dates) or sell_idx >= len(all_dates):
        return None

    buy_date = all_dates[buy_idx]
    sell_date = all_dates[sell_idx]

    buy_price = prices.get(buy_date)
    sell_price = prices.get(sell_date)

    if buy_price and sell_price and buy_price > 0:
        return (sell_price - buy_price) / buy_price
    return None


def apply_composite_ranking(stock_list, weights):
    """Apply composite ranking with given weights, return top-N codes sorted."""
    n = len(stock_list)
    if n < 2:
        return stock_list

    fields = ['pred_3d', 'pred_5d', 'pred_10d', 'pred_15d']
    weight_list = [weights.get(f, 0) for f in fields]

    # Vectorized ranking
    pred_matrix = np.array([[s[f] for f in fields] for s in stock_list])
    rank_matrix = np.zeros_like(pred_matrix)
    for j in range(4):
        ranks = rankdata(pred_matrix[:, j], method='average')
        rank_matrix[:, j] = (ranks - 1) / max(n - 1, 1)

    composite = rank_matrix @ np.array(weight_list)

    for i, s in enumerate(stock_list):
        s['rank_score'] = composite[i]

    stock_list.sort(key=lambda x: x['rank_score'], reverse=True)
    return stock_list


def evaluate_weights(all_data, weights, price_lookup, all_dates, date_idx,
                     top_n=10, hold_days=10):
    """Evaluate a single weight combo: compute IC, return, Sharpe, MaxDD."""
    dates = sorted(all_data.keys())

    daily_ics = []
    daily_returns = []
    portfolio_values = [1.0]

    for date in dates:
        stocks = all_data[date]
        stocks_copy = [dict(s) for s in stocks]  # don't mutate original

        # Apply ranking
        apply_composite_ranking(stocks_copy, weights)

        # Top-N picks
        top_picks = stocks_copy[:top_n]

        # Compute IC: rank correlation between rank_score and actual forward return
        # Use all stocks (or a sample) for IC
        sample = stocks_copy[:200]  # top 200 for speed
        pred_scores = []
        actual_returns = []
        for s in sample:
            ret = get_forward_return(s['code'], date, hold_days,
                                     price_lookup, all_dates, date_idx)
            if ret is not None:
                pred_scores.append(s['rank_score'])
                actual_returns.append(ret)

        if len(pred_scores) >= 10:
            ic, _ = spearmanr(pred_scores, actual_returns)
            if not np.isnan(ic):
                daily_ics.append(ic)

        # Portfolio return: equal-weight top-N
        pick_returns = []
        for s in top_picks:
            ret = get_forward_return(s['code'], date, hold_days,
                                     price_lookup, all_dates, date_idx)
            if ret is not None:
                pick_returns.append(ret)

        if pick_returns:
            port_ret = np.mean(pick_returns)
            daily_returns.append(port_ret)
            # Approximate: each pick held for hold_days, so we step forward
            portfolio_values.append(portfolio_values[-1] * (1 + port_ret))

    if not daily_returns or not daily_ics:
        return None

    # Metrics
    ic_mean = np.mean(daily_ics)
    ic_std = np.std(daily_ics)
    icir = ic_mean / ic_std if ic_std > 0 else 0
    ic_pos = np.mean([1 for ic in daily_ics if ic > 0]) if daily_ics else 0

    # Annualized return (approximate)
    total_ret = portfolio_values[-1] / portfolio_values[0] - 1
    n_periods = len(daily_returns)
    periods_per_year = 252 / hold_days
    ann_ret = (1 + total_ret) ** (periods_per_year / max(n_periods, 1)) - 1

    # Sharpe (Newey-West adjusted for overlapping returns)
    ret_arr = np.array(daily_returns)
    if len(ret_arr) > 1:
        r_mean = ret_arr.mean()
        demeaned = ret_arr - r_mean
        n_r = len(ret_arr)
        gamma0 = np.sum(demeaned**2) / n_r
        nw_var = gamma0
        for lag in range(1, min(hold_days, n_r)):
            gamma_lag = np.sum(demeaned[lag:] * demeaned[:-lag]) / n_r
            nw_var += 2 * (1 - lag / hold_days) * gamma_lag
        nw_std = np.sqrt(max(nw_var, 0))
        sharpe = r_mean / nw_std * np.sqrt(periods_per_year) if nw_std > 0 else 0
    else:
        sharpe = 0

    # MaxDD
    values = np.array(portfolio_values)
    peak = np.maximum.accumulate(values)
    dd = (values - peak) / peak
    max_dd = dd.min()

    # Mean return per period
    mean_ret = np.mean(daily_returns)

    return {
        'ic_mean': ic_mean,
        'icir': icir,
        'ic_pos': ic_pos,
        'ann_ret': ann_ret,
        'total_ret': total_ret,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'mean_ret': mean_ret,
        'n_periods': n_periods,
    }


def grid_search(report_dir, top_n=10, focus_days=10):
    report_dir = Path(report_dir)

    print("Loading report data...")
    all_data = load_raw_data(report_dir)
    print(f"  {len(all_data)} dates loaded")

    # Collect all stock codes
    all_codes = set()
    for stocks in all_data.values():
        for s in stocks:
            all_codes.add(s['code'])
    print(f"  {len(all_codes)} unique stocks")

    print("Loading price data from DB...")
    dates = sorted(all_data.keys())
    price_lookup, all_dates, date_idx = load_price_data(dates, all_codes)
    print(f"  Price data for {len(price_lookup)} stocks")

    # Generate weight grid
    step = 0.05
    candidates = []
    for w3 in [round(i * step, 2) for i in range(0, 7)]:       # 0.00 - 0.30
        for w5 in [round(i * step, 2) for i in range(0, 9)]:   # 0.00 - 0.40
            for w10 in [round(i * step, 2) for i in range(4, 13)]:  # 0.20 - 0.60
                w15 = round(1.0 - w3 - w5 - w10, 2)
                if w15 < -0.001 or w15 > 0.601:
                    continue
                w15 = max(0.0, w15)
                candidates.append({'pred_3d': w3, 'pred_5d': w5,
                                   'pred_10d': w10, 'pred_15d': w15})

    print(f"\nGrid search: {len(candidates)} weight combos, top-{top_n}, {focus_days}d hold")
    print(f"{'='*95}")

    results = []
    for i, weights in enumerate(candidates):
        r = evaluate_weights(all_data, weights, price_lookup, all_dates, date_idx,
                            top_n=top_n, hold_days=focus_days)
        if r is None:
            continue

        entry = {**weights, **r}
        results.append(entry)

        if (i + 1) % 50 == 0 or i == len(candidates) - 1:
            best = max(results, key=lambda x: x['icir'])
            print(f"  [{i+1}/{len(candidates)}] best ICIR so far: "
                  f"({best['pred_3d']:.2f},{best['pred_5d']:.2f},"
                  f"{best['pred_10d']:.2f},{best['pred_15d']:.2f}) "
                  f"ICIR={best['icir']:.3f} AnnRet={best['ann_ret']:.1%} "
                  f"Sharpe={best['sharpe']:.3f}")

    # Sort by ICIR (primary), then Sharpe (secondary)
    results.sort(key=lambda x: (x['icir'], x['sharpe']), reverse=True)

    print(f"\n{'='*100}")
    print(f"TOP 20 COMPOSITE WEIGHT COMBOS (by ICIR)")
    print(f"{'='*100}")
    print(f"{'Rank':>4} {'3d':>5} {'5d':>5} {'10d':>5} {'15d':>5} | "
          f"{'ICIR':>6} {'IC':>6} {'IC>0':>5} | "
          f"{'AnnRet':>8} {'TotalRet':>9} {'MaxDD':>7} {'Sharpe':>7} {'Periods':>7}")
    print(f"{'-'*100}")

    for rank, r in enumerate(results[:20], 1):
        print(f"{rank:>4} {r['pred_3d']:>5.2f} {r['pred_5d']:>5.2f} "
              f"{r['pred_10d']:>5.2f} {r['pred_15d']:>5.2f} | "
              f"{r['icir']:>6.3f} {r['ic_mean']:>6.4f} {r['ic_pos']*100:>4.0f}% | "
              f"{r['ann_ret']*100:>7.1f}% {r['total_ret']*100:>8.1f}% "
              f"{r['max_dd']*100:>6.1f}% {r['sharpe']:>7.3f} {r['n_periods']:>7}")

    # Also sort by Sharpe
    results_by_sharpe = sorted(results, key=lambda x: x['sharpe'], reverse=True)
    print(f"\n{'='*100}")
    print(f"TOP 10 BY SHARPE")
    print(f"{'='*100}")
    for rank, r in enumerate(results_by_sharpe[:10], 1):
        print(f"{rank:>4} {r['pred_3d']:>5.2f} {r['pred_5d']:>5.2f} "
              f"{r['pred_10d']:>5.2f} {r['pred_15d']:>5.2f} | "
              f"ICIR={r['icir']:>6.3f} IC={r['ic_mean']:>6.4f} | "
              f"AnnRet={r['ann_ret']*100:>7.1f}% MaxDD={r['max_dd']*100:>6.1f}% "
              f"Sharpe={r['sharpe']:>7.3f}")

    # Also sort by AnnRet
    results_by_ret = sorted(results, key=lambda x: x['ann_ret'], reverse=True)
    print(f"\n{'='*100}")
    print(f"TOP 10 BY ANNUAL RETURN")
    print(f"{'='*100}")
    for rank, r in enumerate(results_by_ret[:10], 1):
        print(f"{rank:>4} {r['pred_3d']:>5.2f} {r['pred_5d']:>5.2f} "
              f"{r['pred_10d']:>5.2f} {r['pred_15d']:>5.2f} | "
              f"ICIR={r['icir']:>6.3f} IC={r['ic_mean']:>6.4f} | "
              f"AnnRet={r['ann_ret']*100:>7.1f}% MaxDD={r['max_dd']*100:>6.1f}% "
              f"Sharpe={r['sharpe']:>7.3f}")

    # Current default
    current = [r for r in results if abs(r['pred_3d']-0.10)<0.01 and abs(r['pred_5d']-0.20)<0.01
               and abs(r['pred_10d']-0.40)<0.01 and abs(r['pred_15d']-0.30)<0.01]
    if current:
        c = current[0]
        cur_rank_icir = next(i for i, r in enumerate(results, 1) if r is c)
        cur_rank_sharpe = next(i for i, r in enumerate(results_by_sharpe, 1) if r is c)
        print(f"\nCURRENT DEFAULT (0.10/0.20/0.40/0.30) — ICIR rank #{cur_rank_icir}, Sharpe rank #{cur_rank_sharpe}:")
        print(f"  ICIR={c['icir']:.3f} IC={c['ic_mean']:.4f} IC>0={c['ic_pos']*100:.0f}%")
        print(f"  AnnRet={c['ann_ret']*100:.1f}% TotalRet={c['total_ret']*100:.1f}% "
              f"MaxDD={c['max_dd']*100:.1f}% Sharpe={c['sharpe']:.3f}")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Grid search composite ranking weights')
    parser.add_argument('--report-dir', default='reports/daily_selection_v4.7.5')
    parser.add_argument('--top-n', type=int, default=10)
    parser.add_argument('--focus-days', type=int, default=10)
    args = parser.parse_args()

    report_dir = PROJECT_ROOT / args.report_dir
    results = grid_search(report_dir, top_n=args.top_n, focus_days=args.focus_days)
    print(f"\nTotal combos evaluated: {len(results)}")
