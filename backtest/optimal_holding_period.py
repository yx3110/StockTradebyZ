#!/usr/bin/env python3
"""
最优持仓天数分析：买入Top-10后，持有N天卖出，哪个N最优？

方法：
- 用V4.7.5评分，每个调仓日选Top-10
- 分别测试持有 1/2/3/5/7/10/15/20 天后卖出
- 使用非重叠持仓周期（持有期结束后才开始下一轮）
- 50万资金，无滑点
"""

import sys, os, json, time, sqlite3, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
DB_PATH = os.path.join(PROJECT_ROOT, 'data_adapter', 'stock_data.db')
SCORE_CACHE = os.path.join(PROJECT_ROOT, 'backtest', '.v475_score_cache.pkl')


def get_trading_dates(start_date, end_date):
    conn = sqlite3.connect(DB_PATH)
    dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT trade_date FROM v39_feature_cache WHERE trade_date >= ? AND trade_date <= ? ORDER BY trade_date",
        (start_date, end_date)).fetchall()]
    conn.close()
    return dates


def get_all_trading_dates():
    conn = sqlite3.connect(DB_PATH)
    dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT trade_date FROM daily_quotes ORDER BY trade_date").fetchall()]
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


def preload_forward_returns_multi(dates, all_trade_dates, hold_days_list):
    """一次性加载所有持仓天数的forward returns"""
    conn = sqlite3.connect(DB_PATH)
    date_to_idx = {d: i for i, d in enumerate(all_trade_dates)}
    result = {hd: {} for hd in hold_days_list}

    for date in dates:
        idx = date_to_idx.get(date)
        if idx is None:
            continue
        buy_idx = idx + 1  # T+1买入

        for hd in hold_days_list:
            sell_idx = buy_idx + hd
            if sell_idx >= len(all_trade_dates):
                continue
            buy_date = all_trade_dates[buy_idx]
            sell_date = all_trade_dates[sell_idx]
            rows = conn.execute("""
                SELECT s.code, q_buy.close, q_sell.close
                FROM daily_quotes q_buy
                JOIN daily_quotes q_sell ON q_buy.security_id = q_sell.security_id
                JOIN securities s ON q_buy.security_id = s.id
                WHERE q_buy.trade_date = ? AND q_sell.trade_date = ?
                  AND q_buy.close > 0 AND q_sell.close > 0
            """, (buy_date, sell_date)).fetchall()
            result[hd][date] = {code: (sp - bp) / bp for code, bp, sp in rows}

    conn.close()
    return result


def score_all_dates(dates, features_cache):
    """评分所有日期，带缓存"""
    # 尝试加载缓存
    if os.path.exists(SCORE_CACHE):
        try:
            with open(SCORE_CACHE, 'rb') as f:
                cached = pickle.load(f)
            if cached.get('dates_hash') == hash(tuple(dates)):
                print(f"  Using cached scores ({len(cached['scores'])} dates)")
                return cached['scores']
        except:
            pass

    from ml_models.v39.v475_production_scorer import V475ProductionScorer
    scorer = V475ProductionScorer()

    t0 = time.time()
    daily_scores = {}
    for di, date in enumerate(dates):
        fdf = features_cache.get(date)
        if fdf is None or len(fdf) < 100:
            continue
        all_codes = fdf['code'].tolist()
        results = scorer.predict_scores_from_preloaded(all_codes, date, fdf)
        if len(results) > 50:
            daily_scores[date] = results
        if (di + 1) % 100 == 0:
            print(f"  {di+1}/{len(dates)} ({time.time()-t0:.0f}s)")
            sys.stdout.flush()
    print(f"  Done: {len(daily_scores)} dates in {time.time()-t0:.0f}s")

    # 保存缓存
    try:
        with open(SCORE_CACHE, 'wb') as f:
            pickle.dump({'dates_hash': hash(tuple(dates)), 'scores': daily_scores}, f)
        print(f"  Scores cached to {SCORE_CACHE}")
    except:
        pass

    return daily_scores


def get_top_n(scores, top_n=10):
    """从scores字典中取rank_score最高的top_n"""
    ranked = sorted(scores.items(), key=lambda x: x[1].get('rank_score', 0), reverse=True)
    return [code for code, _ in ranked[:top_n]]


def main():
    start_date = '2024-01-01'
    end_date = '2026-02-13'
    top_n = 10
    capital = 500000
    hold_days_list = [1, 2, 3, 5, 7, 10, 15, 20]

    # 交易成本 (50万小资金)
    commission = 0.00025
    stamp_tax = 0.0005
    transfer_fee = 0.00002
    round_trip = commission * 2 + stamp_tax + transfer_fee * 2  # ~0.104%

    print(f"{'='*70}")
    print(f"  最优持仓天数分析: V4.7.5 + 50万资金")
    print(f"  {start_date} ~ {end_date}, Top-{top_n}")
    print(f"  交易成本: {round_trip:.3%}/次 (无滑点)")
    print(f"{'='*70}\n")

    dates = get_trading_dates(start_date, end_date)
    all_trade_dates = get_all_trading_dates()

    print(f"Loading features ({len(dates)} dates)...")
    sys.stdout.flush()
    features_cache = preload_features(dates)

    print(f"Loading forward returns for {hold_days_list}...")
    sys.stdout.flush()
    fwd_returns = preload_forward_returns_multi(dates, all_trade_dates, hold_days_list)

    print(f"Scoring...")
    sys.stdout.flush()
    daily_scores = score_all_dates(dates, features_cache)

    # ============================================================
    # 分析1：非重叠持仓周期（最真实的模拟）
    # 每轮：选股 → 持有N天 → 卖出 → 下一轮选股
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  分析1: 非重叠持仓周期 (买入→持有N天→卖出→下一轮)")
    print(f"{'='*70}")
    print(f"  {'持仓天数':>8} {'轮数':>6} {'平均收益':>8} {'中位收益':>8} {'胜率':>6} "
          f"{'年化(净)':>10} {'Sharpe':>8} {'MaxDD':>8} {'年换手':>6}")
    print(f"  {'-'*82}")

    best_sharpe = 0
    best_hd = 0

    for hd in hold_days_list:
        fwd = fwd_returns[hd]
        scored_dates = sorted(daily_scores.keys())

        # 非重叠选股
        nav = capital
        nav_history = [nav]
        period_returns = []
        i = 0
        while i < len(scored_dates):
            date = scored_dates[i]
            if date not in fwd:
                i += 1
                continue

            scores = daily_scores[date]
            top_codes = get_top_n(scores, top_n)

            # 持仓收益
            rets = [fwd[date].get(c, np.nan) for c in top_codes]
            valid_rets = [r for r in rets if not np.isnan(r)]
            if valid_rets:
                port_ret = np.mean(valid_rets)
                # 扣交易成本 (买+卖一次)
                cost = round_trip
                net_ret = port_ret - cost
                nav *= (1 + net_ret)
                period_returns.append(net_ret)
                nav_history.append(nav)

            # 跳过持仓天数
            i += hd

        if not period_returns:
            continue

        period_returns = np.array(period_returns)
        nav_arr = np.array(nav_history)

        # 指标计算
        total_return = nav_arr[-1] / nav_arr[0] - 1
        n_periods = len(period_returns)
        total_years = n_periods * hd / 245.0
        ann_return = (nav_arr[-1] / nav_arr[0]) ** (1 / total_years) - 1 if total_years > 0 else 0

        # Sharpe (per-period)
        periods_per_year = 245.0 / hd
        if len(period_returns) > 1 and period_returns.std() > 0:
            sharpe = period_returns.mean() / period_returns.std() * np.sqrt(periods_per_year)
        else:
            sharpe = 0

        # MaxDD
        peak = np.maximum.accumulate(nav_arr)
        dd = (nav_arr - peak) / peak
        max_dd = dd.min()

        # 年化换手次数
        annual_trades = periods_per_year  # 每轮全换

        win_rate = np.mean(period_returns > 0)
        mean_ret = np.mean(period_returns)
        median_ret = np.median(period_returns)

        print(f"  {hd:>6}天  {n_periods:>6} {mean_ret:>+7.2%} {median_ret:>+7.2%} {win_rate:>5.1%} "
              f"{ann_return:>+9.1%} {sharpe:>8.3f} {max_dd:>+7.1%} {annual_trades:>5.1f}次")

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_hd = hd

    print(f"\n  ✓ Sharpe最优持仓: {best_hd}天")
    sys.stdout.flush()

    # ============================================================
    # 分析2：按信号强度分层
    # 只看rank_score最高的那些（真正的"强烈买入"）
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  分析2: 信号强度分层 (Top-5 vs Top-10 vs Top-20)")
    print(f"{'='*70}")

    for top_k in [5, 10, 20]:
        print(f"\n  --- Top-{top_k} ---")
        print(f"  {'持仓天数':>8} {'轮数':>6} {'平均收益':>8} {'胜率':>6} {'年化(净)':>10} {'Sharpe':>8}")
        print(f"  {'-'*54}")

        for hd in [3, 5, 7, 10, 15]:
            fwd = fwd_returns[hd]
            scored_dates = sorted(daily_scores.keys())
            period_returns = []
            nav = capital
            i = 0
            while i < len(scored_dates):
                date = scored_dates[i]
                if date not in fwd:
                    i += 1
                    continue
                scores = daily_scores[date]
                top_codes = get_top_n(scores, top_k)
                rets = [fwd[date].get(c, np.nan) for c in top_codes]
                valid_rets = [r for r in rets if not np.isnan(r)]
                if valid_rets:
                    net_ret = np.mean(valid_rets) - round_trip
                    nav *= (1 + net_ret)
                    period_returns.append(net_ret)
                i += hd

            if not period_returns:
                continue
            period_returns = np.array(period_returns)
            total_years = len(period_returns) * hd / 245.0
            ann_return = (nav / capital) ** (1 / total_years) - 1 if total_years > 0 else 0
            periods_per_year = 245.0 / hd
            sharpe = period_returns.mean() / period_returns.std() * np.sqrt(periods_per_year) if period_returns.std() > 0 else 0
            print(f"  {hd:>6}天  {len(period_returns):>6} {np.mean(period_returns):>+7.2%} "
                  f"{np.mean(period_returns>0):>5.1%} {ann_return:>+9.1%} {sharpe:>8.3f}")
    sys.stdout.flush()

    # ============================================================
    # 分析3：逐日收益衰减曲线
    # 买入Top-10后，第1天、第2天...第20天各赚多少？
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  分析3: Alpha衰减曲线 (买入后第N天的边际收益)")
    print(f"{'='*70}")

    # 计算从买入后第1天到第20天的累积收益
    cumulative_by_day = {}
    for hd in hold_days_list:
        fwd = fwd_returns[hd]
        all_rets = []
        for date in sorted(daily_scores.keys()):
            if date not in fwd:
                continue
            scores = daily_scores[date]
            top_codes = get_top_n(scores, top_n)
            rets = [fwd[date].get(c, np.nan) for c in top_codes]
            valid_rets = [r for r in rets if not np.isnan(r)]
            if valid_rets:
                all_rets.append(np.mean(valid_rets))
        if all_rets:
            cumulative_by_day[hd] = np.mean(all_rets)

    print(f"\n  累积收益 (买入后持有到第N天):")
    print(f"  {'持有天数':>8} {'累积收益':>10} {'日均收益':>10} {'边际收益':>10}")
    print(f"  {'-'*42}")

    prev_cum = 0
    sorted_days = sorted(cumulative_by_day.keys())
    for hd in sorted_days:
        cum = cumulative_by_day[hd]
        daily_avg = cum / hd
        marginal = cum - prev_cum
        prev_day = sorted_days[sorted_days.index(hd) - 1] if sorted_days.index(hd) > 0 else 0
        days_diff = hd - prev_day
        marginal_per_day = marginal / days_diff if days_diff > 0 else 0

        bar_len = int(cum * 200) if cum > 0 else 0
        bar = '█' * min(bar_len, 40)
        print(f"  {hd:>6}天  {cum:>+9.3%} {daily_avg:>+9.4%} {marginal_per_day:>+9.4%}/天 {bar}")
        prev_cum = cum

    # 找到边际收益为负的拐点
    print(f"\n  解读:")
    if len(sorted_days) >= 2:
        for i in range(1, len(sorted_days)):
            hd = sorted_days[i]
            prev_hd = sorted_days[i-1]
            marginal = cumulative_by_day[hd] - cumulative_by_day[prev_hd]
            days_diff = hd - prev_hd
            if marginal / days_diff < cumulative_by_day[prev_hd] / prev_hd * 0.5:
                print(f"  ⚠ 在第{prev_hd}→{hd}天之间，边际alpha大幅衰减")
                print(f"     前{prev_hd}天日均: {cumulative_by_day[prev_hd]/prev_hd:+.4%}/天")
                print(f"     第{prev_hd+1}-{hd}天日均: {marginal/days_diff:+.4%}/天")
                break

    # ============================================================
    # 分析4：不同调仓频率下的最优持仓期
    # 结合之前的调仓频率分析
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  分析4: 实际操作建议 (50万)")
    print(f"{'='*70}")

    print(f"""
  场景A: 纯买入持有型 (不盯盘)
    → 每5天选一次股，持有5天后全卖再买
    → 适合上班族，每周操作一次

  场景B: 滚动持仓型 (每天看一眼)
    → 每天检查，只换掉排名跌出Top-15的股票
    → 未跌出的继续持有 (减少无意义换手)

  场景C: 信号触发型 (看到强信号才动)
    → 平时持有，只在rank_score显著变化时调仓
    → 最低换手，最接近"强烈买入时买入"
""")
    sys.stdout.flush()

    # 场景B模拟: 每天调仓但有缓冲区 (只换跌出top-15的)
    print(f"  场景B详细模拟 (缓冲区调仓):")
    print(f"  {'缓冲区':>8} {'年化(净)':>10} {'MaxDD':>8} {'Sharpe':>8} {'日换手':>8}")
    print(f"  {'-'*48}")

    # 加载1d returns
    fwd_1d = preload_forward_returns_multi(dates, all_trade_dates, [1])[1]

    for buffer_size in [10, 12, 15, 20, 30]:
        nav = capital
        nav_history = [nav]
        current_holdings = []
        daily_rets = []
        turnovers = []

        for date in sorted(daily_scores.keys()):
            if date not in fwd_1d:
                continue
            scores = daily_scores[date]
            new_top = get_top_n(scores, top_n)

            if not current_holdings:
                current_holdings = new_top
            else:
                # 只换掉跌出buffer_size的
                keep = [c for c in current_holdings if c in get_top_n(scores, buffer_size)]
                need = top_n - len(keep)
                # 从new_top中补充不在keep中的
                additions = [c for c in new_top if c not in keep][:need]
                sold = [c for c in current_holdings if c not in keep]

                turnover = len(sold) / top_n
                turnovers.append(turnover)

                # 交易成本
                cost = turnover * round_trip
                nav -= nav * cost

                current_holdings = keep + additions

            # 持仓收益
            rets = [fwd_1d[date].get(c, np.nan) for c in current_holdings]
            valid = [r for r in rets if not np.isnan(r)]
            if valid:
                port_ret = np.mean(valid)
                nav *= (1 + port_ret)
                daily_rets.append(port_ret)
                nav_history.append(nav)

        daily_rets = np.array(daily_rets)
        nav_arr = np.array(nav_history)
        total_years = len(daily_rets) / 245.0
        ann = (nav_arr[-1] / nav_arr[0]) ** (1 / total_years) - 1 if total_years > 0 else 0
        sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(245) if daily_rets.std() > 0 else 0
        peak = np.maximum.accumulate(nav_arr)
        max_dd = ((nav_arr - peak) / peak).min()
        avg_turnover = np.mean(turnovers) if turnovers else 0

        label = f"Top-{buffer_size}"
        print(f"  {label:>8} {ann:>+9.1%} {max_dd:>+7.1%} {sharpe:>8.3f} {avg_turnover:>7.1%}")

    sys.stdout.flush()


if __name__ == '__main__':
    main()
