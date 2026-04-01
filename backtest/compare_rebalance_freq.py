#!/usr/bin/env python3
"""
调仓频率对比: 每天 vs 每5天 vs 每10天
模拟50万小资金实际持仓，对比不同调仓频率的真实净值曲线

核心区别：
- 每天调仓: 总是持有最新Top-10，但频繁交易
- 每N天调仓: 持有Top-10不动N天，然后切换
- 小资金(50万)滑点≈0，成本主要是佣金+印花税
"""

import sys, os, json, time, sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
DB_PATH = os.path.join(PROJECT_ROOT, 'data_adapter', 'stock_data.db')


def get_trading_dates(start_date, end_date):
    conn = sqlite3.connect(DB_PATH)
    dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT trade_date FROM v39_feature_cache WHERE trade_date >= ? AND trade_date <= ? ORDER BY trade_date",
        (start_date, end_date)).fetchall()]
    conn.close()
    return dates


def preload_features(dates):
    result = {}
    conn = sqlite3.connect(DB_PATH)
    for i in range(0, len(dates), 50):
        chunk = dates[i:i+50]
        ph = ','.join(['?'] * len(chunk))
        df = pd.read_sql_query(f"""
            SELECT code, trade_date, features_json,
                   market_return_20d, market_return_10d, market_return_5d,
                   market_volatility_20d, market_volatility_10d,
                   market_up_ratio_20d, market_up_ratio_10d,
                   market_drawdown_20d, market_volume_ratio,
                   market_position_20d, market_momentum_20d, market_momentum_5d
            FROM v39_feature_cache WHERE trade_date IN ({ph})
        """, conn, params=chunk)
        if df.empty:
            continue
        parsed = df['features_json'].apply(json.loads)
        features_all = pd.DataFrame(parsed.tolist())
        features_all['code'] = df['code'].values
        features_all['trade_date'] = df['trade_date'].values
        for col in [c for c in df.columns if c.startswith('market_')]:
            features_all[col] = df[col].values
        for date, group in features_all.groupby('trade_date'):
            result[date] = group.drop(columns=['trade_date']).reset_index(drop=True)
    conn.close()
    return result


def preload_next_day_returns(dates):
    """加载每只股票的次日收益 (T→T+1 close-to-close)"""
    conn = sqlite3.connect(DB_PATH)
    all_dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT trade_date FROM daily_quotes ORDER BY trade_date").fetchall()]
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    result = {}
    for date in dates:
        idx = date_to_idx.get(date)
        if idx is None or idx + 2 >= len(all_dates):
            continue
        # T日收盘选股 → T+1开盘买入 → T+1收盘的收益
        # 简化: 用T+1 close vs T close作为当日持仓收益
        buy_date = all_dates[idx + 1]
        sell_date = all_dates[idx + 2] if idx + 2 < len(all_dates) else None
        if sell_date is None:
            continue
        rows = conn.execute("""
            SELECT s.code, q1.close, q2.close
            FROM daily_quotes q1
            JOIN daily_quotes q2 ON q1.security_id = q2.security_id
            JOIN securities s ON q1.security_id = s.id
            WHERE q1.trade_date = ? AND q2.trade_date = ?
              AND q1.close > 0 AND q2.close > 0
        """, (buy_date, sell_date)).fetchall()
        result[date] = {code: (p2 - p1) / p1 for code, p1, p2 in rows}
    conn.close()
    return result


def preload_market_caps(dates):
    """加载市值数据用于过滤微盘"""
    conn = sqlite3.connect(DB_PATH)
    result = {}
    for i in range(0, len(dates), 50):
        chunk = dates[i:i+50]
        ph = ','.join(['?'] * len(chunk))
        rows = conn.execute(f"""
            SELECT s.code, db.trade_date, db.total_mv
            FROM daily_basic db
            JOIN securities s ON db.security_id = s.id
            WHERE db.trade_date IN ({ph}) AND db.total_mv > 0
        """, chunk).fetchall()
        for code, date, mv in rows:
            if date not in result:
                result[date] = {}
            result[date][code] = mv / 10000  # 万→亿
    conn.close()
    return result


def score_date(scorer, features_df, date):
    all_codes = features_df['code'].tolist()
    results = scorer.predict_scores_from_preloaded(all_codes, date, features_df)
    return results


def simulate_rebalance(daily_scores, daily_1d_returns, dates, rebalance_every,
                       top_n=10, capital=500000,
                       commission=0.00025, stamp_tax=0.0005, transfer_fee=0.00002,
                       slippage=0.0,  # 50万小资金滑点≈0
                       market_caps=None, min_cap=0):
    """
    模拟不同调仓频率的净值曲线

    rebalance_every: 1=每天, 5=每5天, 10=每10天
    """
    nav = capital
    nav_history = [nav]
    date_history = [dates[0]]
    current_holdings = set()
    total_trades = 0
    total_cost = 0
    daily_returns = []

    round_trip = commission * 2 + stamp_tax + transfer_fee * 2 + slippage * 2

    for i, date in enumerate(dates[:-1]):  # 最后一天不操作
        scores = daily_scores.get(date)
        next_day_rets = daily_1d_returns.get(date)
        if scores is None or next_day_rets is None:
            continue

        # 是否调仓
        should_rebalance = (i % rebalance_every == 0)

        if should_rebalance:
            # 按rank_score排名
            ranked = sorted(scores.items(), key=lambda x: x[1].get('rank_score', 0), reverse=True)

            # 过滤微盘股
            if market_caps and min_cap > 0:
                caps = market_caps.get(date, {})
                ranked = [(c, s) for c, s in ranked if caps.get(c, 999) >= min_cap]

            new_top = set(c for c, _ in ranked[:top_n])
        else:
            new_top = current_holdings

        # 计算换手
        if current_holdings:
            sold = current_holdings - new_top
            bought = new_top - current_holdings
            n_trades = len(sold) + len(bought)
        else:
            sold = set()
            bought = new_top
            n_trades = len(bought)  # 首次建仓只算买入

        # 交易成本 (对调仓部分)
        if current_holdings and n_trades > 0:
            # 卖出+买入各一半
            trade_fraction = len(sold) / top_n  # 换手比例
            cost = nav * trade_fraction * round_trip
            nav -= cost
            total_cost += cost
            total_trades += n_trades

        current_holdings = new_top

        # 持仓收益 (等权)
        if current_holdings:
            holding_rets = []
            for code in current_holdings:
                r = next_day_rets.get(code)
                if r is not None:
                    holding_rets.append(r)
            if holding_rets:
                port_ret = np.mean(holding_rets)
                nav *= (1 + port_ret)
                daily_returns.append(port_ret)

        nav_history.append(nav)
        date_history.append(date)

    # 计算指标
    nav_arr = np.array(nav_history)
    total_return = nav_arr[-1] / nav_arr[0] - 1
    n_days = len(daily_returns)
    total_years = n_days / 245.0

    if total_years > 0 and nav_arr[-1] > 0:
        ann_return = (nav_arr[-1] / nav_arr[0]) ** (1 / total_years) - 1
    else:
        ann_return = 0

    # MaxDD
    peak = np.maximum.accumulate(nav_arr)
    dd = (nav_arr - peak) / peak
    max_dd = dd.min()

    # Sharpe (日频, 无需NW修正因为是真实日收益)
    daily_returns = np.array(daily_returns)
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(245)
    else:
        sharpe = 0

    # 月度胜率
    monthly_rets = []
    for m_start in range(0, len(daily_returns), 21):
        m_end = min(m_start + 21, len(daily_returns))
        if m_end > m_start:
            m_ret = np.prod(1 + daily_returns[m_start:m_end]) - 1
            monthly_rets.append(m_ret)
    monthly_win = np.mean([r > 0 for r in monthly_rets]) if monthly_rets else 0

    return {
        'total_return': total_return,
        'ann_return': ann_return,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'total_trades': total_trades,
        'total_cost': total_cost,
        'cost_pct': total_cost / capital,
        'annual_cost_pct': total_cost / capital / total_years if total_years > 0 else 0,
        'daily_returns': daily_returns,
        'nav_history': nav_arr,
        'monthly_win': monthly_win,
        'n_months': len(monthly_rets),
        'n_days': n_days,
    }


def main():
    start_date = '2024-01-01'
    end_date = '2026-02-13'
    top_n = 10
    capital = 500000  # 50万

    print(f"{'='*70}")
    print(f"  调仓频率对比: V4.7.5 + 50万资金")
    print(f"  {start_date} ~ {end_date}, Top-{top_n}")
    print(f"{'='*70}\n")

    # Load scorer
    from ml_models.v39.v475_production_scorer import V475ProductionScorer
    scorer = V475ProductionScorer()

    dates = get_trading_dates(start_date, end_date)
    print(f"Loading features ({len(dates)} dates)...")
    sys.stdout.flush()
    features_cache = preload_features(dates)
    print(f"Loading 1-day forward returns...")
    sys.stdout.flush()
    daily_1d_returns = preload_next_day_returns(dates)
    print(f"Loading market caps...")
    sys.stdout.flush()
    market_caps = preload_market_caps(dates)

    # Score all dates
    print(f"Scoring all dates...")
    sys.stdout.flush()
    t0 = time.time()
    daily_scores = {}
    for di, date in enumerate(dates):
        fdf = features_cache.get(date)
        if fdf is None or len(fdf) < 100:
            continue
        results = score_date(scorer, fdf, date)
        if len(results) > 50:
            daily_scores[date] = results
        if (di + 1) % 100 == 0:
            print(f"  {di+1}/{len(dates)} ({time.time()-t0:.0f}s)")
            sys.stdout.flush()
    print(f"  Done: {len(daily_scores)} dates in {time.time()-t0:.0f}s\n")
    sys.stdout.flush()

    # 成本场景
    cost_scenarios = {
        '50万(无滑点)': {'slippage': 0.0},
        '500万(0.05%滑点)': {'slippage': 0.0005},
        '5000万(0.15%滑点)': {'slippage': 0.0015},
    }

    # 调仓频率
    frequencies = [1, 2, 3, 5, 10, 20]

    for scenario_name, cost_params in cost_scenarios.items():
        print(f"\n{'='*70}")
        print(f"  场景: {scenario_name}")
        print(f"{'='*70}")
        print(f"  {'频率':>10} {'年化(净)':>10} {'总收益':>10} {'MaxDD':>8} {'Sharpe':>8} "
              f"{'交易次数':>8} {'年成本':>8} {'月胜率':>8}")
        print(f"  {'-'*78}")

        results = {}
        for freq in frequencies:
            r = simulate_rebalance(
                daily_scores, daily_1d_returns, dates,
                rebalance_every=freq, top_n=top_n, capital=capital,
                slippage=cost_params['slippage'],
                market_caps=market_caps, min_cap=0
            )
            results[freq] = r
            print(f"  每{freq:>2}天调仓  {r['ann_return']:>+9.1%} {r['total_return']:>+9.1%} "
                  f"{r['max_dd']:>+7.1%} {r['sharpe']:>8.3f} "
                  f"{r['total_trades']:>8} {r['annual_cost_pct']:>7.1%} "
                  f"{r['monthly_win']:>7.1%}")
        sys.stdout.flush()

        # 最优频率
        best_freq = max(results, key=lambda f: results[f]['ann_return'])
        print(f"\n  ✓ 最优: 每{best_freq}天调仓, "
              f"年化{results[best_freq]['ann_return']:+.1%}, "
              f"Sharpe={results[best_freq]['sharpe']:.3f}")

    # 过滤微盘股的影响
    print(f"\n{'='*70}")
    print(f"  微盘股过滤影响 (每天调仓, 50万无滑点)")
    print(f"{'='*70}")
    print(f"  {'最低市值':>10} {'年化(净)':>10} {'MaxDD':>8} {'Sharpe':>8}")
    print(f"  {'-'*40}")
    for min_cap in [0, 20, 30, 50]:
        r = simulate_rebalance(
            daily_scores, daily_1d_returns, dates,
            rebalance_every=1, top_n=top_n, capital=capital,
            slippage=0.0, market_caps=market_caps, min_cap=min_cap
        )
        label = f"≥{min_cap}亿" if min_cap > 0 else "不过滤"
        print(f"  {label:>10} {r['ann_return']:>+9.1%} {r['max_dd']:>+7.1%} {r['sharpe']:>8.3f}")
    sys.stdout.flush()

    # 详细的每天调仓 vs 每10天调仓 季度对比
    print(f"\n{'='*70}")
    print(f"  季度收益对比: 每天 vs 每10天 (50万无滑点)")
    print(f"{'='*70}")

    r1 = results.get(1, simulate_rebalance(daily_scores, daily_1d_returns, dates,
                                            rebalance_every=1, top_n=top_n, capital=capital))
    r10 = results.get(10, simulate_rebalance(daily_scores, daily_1d_returns, dates,
                                              rebalance_every=10, top_n=top_n, capital=capital))

    # 按季度分析
    rets_1 = r1['daily_returns']
    rets_10 = r10['daily_returns']
    min_len = min(len(rets_1), len(rets_10))

    # 用dates做季度映射
    valid_dates = [d for d in dates if d in daily_scores and d in daily_1d_returns]
    print(f"\n  {'季度':<10} {'每天调仓':>10} {'每10天调仓':>12} {'差异':>8}")
    print(f"  {'-'*44}")

    # 简单按63天(~1季度)分段
    for q_start in range(0, min_len, 63):
        q_end = min(q_start + 63, min_len)
        if q_end - q_start < 20:
            break
        q_ret_1 = np.prod(1 + rets_1[q_start:q_end]) - 1
        q_ret_10 = np.prod(1 + rets_10[q_start:q_end]) - 1
        q_idx = q_start // 63 + 1
        q_year = 2024 + (q_start // 252)
        q_num = ((q_start % 252) // 63) + 1
        print(f"  Q{q_idx:>2}       {q_ret_1:>+9.1%} {q_ret_10:>+11.1%} {q_ret_1-q_ret_10:>+7.1%}")

    print(f"\n  总计      {r1['ann_return']:>+9.1%}(年化) {r10['ann_return']:>+11.1%}(年化)")
    sys.stdout.flush()


if __name__ == '__main__':
    main()
