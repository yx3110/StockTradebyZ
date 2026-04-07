#!/usr/bin/env python3
"""
Portfolio Backtest Engine — 组合回测框架

支持功能:
- 多种选股规则 (top_n, score_floor, holding_buffer)
- 多种加权方式 (equal, score_weighted, inv_volatility)
- 固定周期调仓 (rebal_days)
- 每日止损检测 (ATR止损 + 百分比止损, 跌停延迟卖出)
- CPPI风险预算 (动态仓位缩放)
- 交易成本扣减 (涨停不可买)
- 网格搜索 + IS/OOS双向验证

用法:
    # 单配置回测
    python3 scripts/portfolio_backtest.py \\
      --report-dir reports/daily_selection_ng102 \\
      --top-n 5 --rebal-days 10 --atr-stop 2.0 --cppi-floor 0.05

    # 网格搜索 (IS + OOS)
    python3 scripts/portfolio_backtest.py \\
      --is-dir reports/daily_selection_ng102 \\
      --oos-dir reports/daily_selection_ng102_oos --grid
"""
import sys
import os
import json
import glob
import sqlite3
import argparse
import itertools
import numpy as np
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

DB_PATH = os.path.join(PROJECT_ROOT, 'data_adapter', 'stock_data.db')

# 涨跌停阈值 (板块区分)
LIMIT_MAIN = 0.095      # 主板 ±10%
LIMIT_GEM_STAR = 0.195  # 创业板(30x)/科创板(688x) ±20%

DEFAULT_CONFIG = {
    'top_n': 5,
    'score_floor': 0,
    'holding_buffer': 0,
    'weighting': 'equal',
    'rebal_days': 10,
    'atr_stop_mult': 0,
    'max_loss_pct': 0,
    'cppi_floor': 0,
    'cppi_multiplier': 20,
    'cost_per_side': 0.0015,
}

DEFAULT_GRID = {
    'top_n': [5, 10],
    'holding_buffer': [0, 3, 5],
    'weighting': ['equal', 'score_weighted'],
    'atr_stop_mult': [0, 1.5, 2.0],
    'max_loss_pct': [0, 0.08],
    'cppi_floor': [0, 0.05, 0.08],
    'cppi_multiplier': [15, 20],
    'rebal_days': [5, 10, 15],
}


def _limit_threshold(code: str) -> float:
    """根据股票代码返回涨跌停阈值"""
    if code.startswith('3') or code.startswith('68'):
        return LIMIT_GEM_STAR
    return LIMIT_MAIN


def _is_limit_up(code: str, pct: float) -> bool:
    return pct >= _limit_threshold(code)


def _is_limit_down(code: str, pct: float) -> bool:
    return pct <= -_limit_threshold(code)


def _normalize_code(code: str) -> str:
    """报告中的裸代码 -> 标准化为6位裸代码"""
    code = code.strip()
    if '.' in code:
        code = code.split('.')[0]
    return code


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_reports(report_dir: str) -> dict:
    """加载所有JSON报告文件, 返回 {date_str: [(code, rank_val), ...]}

    date_str格式: 'YYYY-MM-DD'
    code: 6位裸代码
    按rank_val降序排列
    """
    pattern = os.path.join(report_dir, 'analysis_data_*.json')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No report files found in {report_dir}")

    reports = {}
    for fp in files:
        fname = os.path.basename(fp)
        ds = fname.replace('analysis_data_', '').replace('.json', '')
        date_str = f"{ds[:4]}-{ds[4:6]}-{ds[6:8]}"

        with open(fp, 'r', encoding='utf-8') as f:
            data = json.load(f)

        stocks = data.get('all_stocks_with_scores', [])
        picks = []
        for s in stocks:
            code = _normalize_code(s.get('stock_code', ''))
            if len(code) > 6 or not code:
                continue
            # A-stock filter: must have non-empty industry
            industry = s.get('industry', '')
            if not industry or industry == '':
                continue

            rank_val = s.get('rank_score', 0) or 0
            if rank_val == 0:
                rank_val = s.get('pred_10d', 0) or 0
            if rank_val == 0:
                continue
            picks.append((code, float(rank_val)))

        if picks:
            picks.sort(key=lambda x: x[1], reverse=True)
            reports[date_str] = picks

    print(f"  Loaded {len(reports)} report days from {report_dir}")
    return reports


def load_price_data(db_path: str, start_date: str, end_date: str,
                    code_universe: set = None) -> dict:
    """加载价格数据, 返回 {code: {date: {open, close, high, low, pct}}}

    code: 6位裸代码 (去掉.SZ/.SH后缀)
    date: 'YYYY-MM-DD'
    code_universe: 只加载这些代码的数据(大幅减少内存和IO)
    """
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")

    if code_universe and len(code_universe) < 5000:
        placeholders = ','.join('?' for _ in code_universe)
        cursor = conn.execute(f"""
            SELECT s.code, dq.trade_date, dq.open, dq.close, dq.high, dq.low, dq.price_change_pct
            FROM daily_quotes dq JOIN securities s ON s.id = dq.security_id
            WHERE dq.trade_date >= ? AND dq.trade_date <= ?
              AND s.code IN ({placeholders})
            ORDER BY s.code, dq.trade_date
        """, (start_date, end_date, *code_universe))
    else:
        cursor = conn.execute("""
            SELECT s.code, dq.trade_date, dq.open, dq.close, dq.high, dq.low, dq.price_change_pct
            FROM daily_quotes dq JOIN securities s ON s.id = dq.security_id
            WHERE dq.trade_date >= ? AND dq.trade_date <= ?
              AND s.code NOT LIKE '%.%'
            ORDER BY s.code, dq.trade_date
        """, (start_date, end_date))

    prices = defaultdict(dict)
    for row in cursor:
        code, td, o, c, h, l, pct = row
        prices[code][td] = {
            'open': float(o) if o else 0,
            'close': float(c) if c else 0,
            'high': float(h) if h else 0,
            'low': float(l) if l else 0,
            'pct': float(pct) if pct else 0,
        }
    conn.close()
    print(f"  Loaded price data for {len(prices)} stocks")
    return dict(prices)


def load_benchmark(db_path: str, start_date: str, end_date: str,
                   bench_code: str = '000300.SH') -> dict:
    """加载基准指数收盘价, 返回 {date: close}"""
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")
    cursor = conn.execute("""
        SELECT dq.trade_date, dq.close
        FROM daily_quotes dq JOIN securities s ON s.id = dq.security_id
        WHERE s.code = ? AND dq.trade_date >= ? AND dq.trade_date <= ?
        ORDER BY dq.trade_date
    """, (bench_code, start_date, end_date))

    bench = {}
    for td, c in cursor:
        bench[td] = float(c)
    conn.close()
    print(f"  Loaded benchmark ({bench_code}): {len(bench)} days")
    return bench


def compute_atr14(prices: dict) -> dict:
    """为所有股票计算ATR14, 返回 {code: {date: atr14_value}}"""
    atr_data = {}
    for code, daily in prices.items():
        dates = sorted(daily.keys())
        if len(dates) < 2:
            continue
        atr_series = {}
        prev_close = None
        ema = None
        alpha = 2.0 / (14 + 1)
        for d in dates:
            bar = daily[d]
            h, l, c = bar['high'], bar['low'], bar['close']
            if prev_close is not None and h > 0 and l > 0:
                tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
                if ema is None:
                    ema = tr
                else:
                    ema = alpha * tr + (1 - alpha) * ema
                atr_series[d] = ema
            prev_close = c
        if atr_series:
            atr_data[code] = atr_series
    print(f"  Computed ATR14 for {len(atr_data)} stocks")
    return atr_data


# ---------------------------------------------------------------------------
# PortfolioBacktester
# ---------------------------------------------------------------------------

class PortfolioBacktester:
    """组合回测引擎"""

    def __init__(self, report_dir: str, db_path: str = DB_PATH):
        print(f"Initializing PortfolioBacktester...")
        self.reports = load_reports(report_dir)
        self.trading_days = sorted(self.reports.keys())
        if not self.trading_days:
            raise ValueError("No valid trading days found")

        # Collect code universe from reports (only load prices for these stocks)
        code_universe = set()
        for picks in self.reports.values():
            for code, _ in picks[:200]:  # top 200 per day is enough
                code_universe.add(code)

        # Date range with buffer for ATR warmup
        first, last = self.trading_days[0], self.trading_days[-1]
        from datetime import datetime, timedelta
        start_dt = datetime.strptime(first, '%Y-%m-%d') - timedelta(days=60)
        start_ext = start_dt.strftime('%Y-%m-%d')

        self.prices = load_price_data(db_path, start_ext, last, code_universe)
        self.benchmark = load_benchmark(db_path, start_ext, last)
        self.atr = compute_atr14(self.prices)
        print(f"  Trading period: {first} ~ {last} ({len(self.trading_days)} days)\n")

    def run(self, config: dict = None) -> dict:
        """执行回测, 返回结果字典"""
        cfg = {**DEFAULT_CONFIG, **(config or {})}

        top_n = cfg['top_n']
        score_floor = cfg['score_floor']
        hold_buf = cfg['holding_buffer']
        weighting = cfg['weighting']
        rebal_days = cfg['rebal_days']
        atr_stop = cfg['atr_stop_mult']
        max_loss = cfg['max_loss_pct']
        cppi_floor = cfg['cppi_floor']
        cppi_mult = cfg['cppi_multiplier']
        cost_rate = cfg['cost_per_side']

        days = self.trading_days
        nav = 1.0
        peak_nav = 1.0
        cash = 1.0
        # holdings: {code: {shares, entry_price, entry_date, weight}}
        holdings = {}
        # entry ATR for stop-loss
        entry_atr = {}

        nav_series = []   # [(date, nav)]
        monthly_rets = defaultdict(float)  # 'YYYY-MM' -> cumulative
        monthly_start = {}  # 'YYYY-MM' -> nav at month start
        yearly_rets = {}
        yearly_start = {}

        n_stops = 0
        n_cppi_days = 0
        total_turnover = 0.0
        n_rebal = 0
        days_since_rebal = rebal_days  # Force rebalance on first day

        # Deferred sells (from limit-down days)
        deferred_stops = set()   # stop-loss sells deferred due to limit-down
        deferred_rebal = set()   # rebalance sells deferred due to limit-down

        for i, date in enumerate(days):
            # ---------------------------------------------------------------
            # 1. Mark-to-market: update NAV from holdings
            # ---------------------------------------------------------------
            portfolio_val = 0.0
            stopped_codes = set()

            for code in list(holdings.keys()):
                pos = holdings[code]
                bar = self.prices.get(code, {}).get(date)
                if bar is None or bar['close'] <= 0:
                    # No price data: keep at last known value
                    portfolio_val += pos['shares'] * pos['entry_price']
                    continue

                cur_price = bar['close']
                pos_val = pos['shares'] * cur_price
                portfolio_val += pos_val

                # Check stop-loss (daily, not just on rebal)
                if atr_stop > 0 or max_loss > 0:
                    stop_price = 0
                    if atr_stop > 0 and code in entry_atr:
                        stop_price = pos['entry_price'] - atr_stop * entry_atr[code]
                    if max_loss > 0:
                        pct_stop = pos['entry_price'] * (1 - max_loss)
                        stop_price = max(stop_price, pct_stop)
                    if stop_price > 0 and cur_price <= stop_price:
                        stopped_codes.add(code)

            nav = cash + portfolio_val

            # ---------------------------------------------------------------
            # 2. Execute stop-loss sells + deferred sells (check limit-down)
            # ---------------------------------------------------------------
            # Add deferred stops back for retry
            for code in list(deferred_stops):
                if code in holdings:
                    stopped_codes.add(code)

            # Execute deferred rebalance sells first
            for code in list(deferred_rebal):
                if code not in holdings:
                    deferred_rebal.discard(code)
                    continue
                bar = self.prices.get(code, {}).get(date)
                if bar is None:
                    continue
                if _is_limit_down(code, bar['pct']):
                    continue  # Still limit-down, keep deferred
                pos = holdings[code]
                sell_val = pos['shares'] * bar['close']
                cost = sell_val * cost_rate
                cash += sell_val - cost
                total_turnover += sell_val
                del holdings[code]
                deferred_rebal.discard(code)

            # Execute stop-loss sells
            for code in stopped_codes:
                if code not in holdings:
                    continue
                bar = self.prices.get(code, {}).get(date)
                if bar is None:
                    continue
                if _is_limit_down(code, bar['pct']):
                    deferred_stops.add(code)
                    continue
                pos = holdings[code]
                sell_val = pos['shares'] * bar['close']
                cost = sell_val * cost_rate
                cash += sell_val - cost
                total_turnover += sell_val
                del holdings[code]
                deferred_stops.discard(code)
                n_stops += 1

            # Recalculate NAV after stops
            portfolio_val = sum(
                h['shares'] * self.prices.get(c, {}).get(date, {}).get('close', h['entry_price'])
                for c, h in holdings.items()
            )
            nav = cash + portfolio_val

            # ---------------------------------------------------------------
            # 3. CPPI exposure calculation
            # ---------------------------------------------------------------
            exposure = 1.0
            if cppi_floor > 0:
                peak_nav = max(peak_nav, nav)
                floor_nav = peak_nav * (1 - cppi_floor)
                cushion = nav - floor_nav
                if cushion <= 0:
                    exposure = 0.0
                else:
                    exposure = min(1.0, cppi_mult * cushion / nav)
                if exposure < 1.0:
                    n_cppi_days += 1

            # ---------------------------------------------------------------
            # 4. Rebalance check
            # ---------------------------------------------------------------
            days_since_rebal += 1
            if days_since_rebal >= rebal_days:
                days_since_rebal = 0
                n_rebal += 1

                # Get today's rankings
                ranked = self.reports.get(date, [])
                if score_floor > 0:
                    ranked = [(c, v) for c, v in ranked if v >= score_floor]

                # Determine target holdings with holding buffer
                current_codes = set(holdings.keys())
                target_codes = []
                # First pass: select up to top_n, respecting buffer for current
                extended_top = ranked[:top_n + hold_buf] if hold_buf > 0 else ranked[:top_n]
                extended_codes = {c for c, v in extended_top}

                # Keep current holdings if within buffer range
                kept = set()
                if hold_buf > 0:
                    for c in current_codes:
                        if c in extended_codes:
                            kept.add(c)

                # Fill remaining slots with top-ranked new stocks
                for c, v in ranked:
                    if len(target_codes) >= top_n:
                        break
                    if c in kept or c not in current_codes:
                        target_codes.append((c, v))
                    if c in kept:
                        kept.discard(c)

                # If we still have buffer-kept stocks not yet added, add them
                for c in list(current_codes):
                    if len(target_codes) >= top_n:
                        break
                    if c in extended_codes and c not in {x[0] for x in target_codes}:
                        # Find rank_val for this code
                        rv = next((v for cc, v in ranked if cc == c), 0)
                        if rv > 0:
                            target_codes.append((c, rv))

                target_dict = {c: v for c, v in target_codes}
                target_set = set(target_dict.keys())

                # Compute weights
                weights = self._compute_weights(
                    target_codes, weighting, date
                )

                # Sell positions not in target
                for code in list(holdings.keys()):
                    if code not in target_set:
                        bar = self.prices.get(code, {}).get(date)
                        if bar is None:
                            continue
                        if _is_limit_down(code, bar['pct']):
                            deferred_rebal.add(code)
                            continue
                        pos = holdings[code]
                        sell_val = pos['shares'] * bar['close']
                        cost = sell_val * cost_rate
                        cash += sell_val - cost
                        total_turnover += sell_val
                        del holdings[code]
                        deferred_rebal.discard(code)

                # Recalculate available capital
                portfolio_val = sum(
                    h['shares'] * self.prices.get(c, {}).get(date, {}).get('close', h['entry_price'])
                    for c, h in holdings.items()
                )
                total_val = cash + portfolio_val
                target_equity = total_val * exposure

                # Buy/rebalance into target weights
                for code, w in weights.items():
                    bar = self.prices.get(code, {}).get(date)
                    if bar is None or bar['close'] <= 0:
                        continue
                    # Skip if limit-up (cannot buy)
                    if code not in holdings and _is_limit_up(code, bar['pct']):
                        continue

                    target_val = target_equity * w
                    cur_val = 0
                    if code in holdings:
                        cur_val = holdings[code]['shares'] * bar['close']

                    diff = target_val - cur_val
                    # Skip tiny rebalances (< 0.5% of total value)
                    if abs(diff) < total_val * 0.005:
                        continue

                    if diff > 0:
                        # Buy
                        if _is_limit_up(code, bar['pct']):
                            continue
                        buy_val = min(diff, cash)
                        if buy_val < total_val * 0.005:
                            continue
                        cost = buy_val * cost_rate
                        shares_add = (buy_val - cost) / bar['close']
                        if code in holdings:
                            old = holdings[code]
                            total_shares = old['shares'] + shares_add
                            # Weighted average entry price
                            old_val = old['shares'] * old['entry_price']
                            new_val = shares_add * bar['close']
                            holdings[code] = {
                                'shares': total_shares,
                                'entry_price': (old_val + new_val) / total_shares,
                                'entry_date': old['entry_date'],
                            }
                        else:
                            holdings[code] = {
                                'shares': shares_add,
                                'entry_price': bar['close'],
                                'entry_date': date,
                            }
                        # Record ATR at entry for stop-loss
                        atr_val = self.atr.get(code, {}).get(date)
                        if atr_val:
                            entry_atr[code] = atr_val
                        cash -= buy_val
                        total_turnover += buy_val
                    else:
                        # Partial sell
                        sell_val = -diff
                        if _is_limit_down(code, bar['pct']):
                            continue
                        shares_sell = sell_val / bar['close']
                        shares_sell = min(shares_sell, holdings[code]['shares'])
                        actual_sell = shares_sell * bar['close']
                        cost = actual_sell * cost_rate
                        cash += actual_sell - cost
                        total_turnover += actual_sell
                        holdings[code]['shares'] -= shares_sell
                        if holdings[code]['shares'] < 1e-6:
                            del holdings[code]

            # ---------------------------------------------------------------
            # 5. End-of-day NAV
            # ---------------------------------------------------------------
            portfolio_val = sum(
                h['shares'] * self.prices.get(c, {}).get(date, {}).get('close', h['entry_price'])
                for c, h in holdings.items()
            )
            nav = cash + portfolio_val
            peak_nav = max(peak_nav, nav)
            nav_series.append((date, nav))

            # Track monthly/yearly
            ym = date[:7]
            if ym not in monthly_start:
                monthly_start[ym] = nav
            monthly_rets[ym] = nav / monthly_start[ym] - 1

            yr = date[:4]
            if yr not in yearly_start:
                yearly_start[yr] = nav
            yearly_rets[yr] = nav / yearly_start[yr] - 1

        # ---------------------------------------------------------------
        # Compute summary metrics
        # ---------------------------------------------------------------
        navs = np.array([n for _, n in nav_series])
        dates_list = [d for d, _ in nav_series]
        n_days = len(navs)
        if n_days < 2:
            return {'error': 'Not enough data'}

        daily_rets = np.diff(navs) / navs[:-1]
        years = n_days / 252.0

        total_ret = navs[-1] / navs[0] - 1
        annual_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0

        # Benchmark returns
        bench_navs = []
        for d in dates_list:
            if d in self.benchmark:
                bench_navs.append(self.benchmark[d])
        if len(bench_navs) >= 2:
            bench_total = bench_navs[-1] / bench_navs[0] - 1
            bench_annual = (1 + bench_total) ** (1 / years) - 1 if years > 0 else 0
        else:
            bench_total = 0
            bench_annual = 0

        excess = annual_ret - bench_annual

        # Sharpe (annualized, rf=0)
        if len(daily_rets) > 1 and np.std(daily_rets) > 0:
            sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)
        else:
            sharpe = 0

        # Max drawdown
        running_max = np.maximum.accumulate(navs)
        drawdowns = (navs - running_max) / running_max
        max_dd = float(np.min(drawdowns))

        calmar = annual_ret / abs(max_dd) if abs(max_dd) > 0 else 0

        # Win rate (monthly)
        monthly_wins = sum(1 for v in monthly_rets.values() if v > 0)
        monthly_total = len(monthly_rets)
        win_rate = monthly_wins / monthly_total if monthly_total > 0 else 0

        # Turnover (annualized, relative to avg NAV)
        avg_nav = float(np.mean(navs))
        turnover = (total_turnover / avg_nav) / years if years > 0 and avg_nav > 0 else 0

        return {
            'annual_return': annual_ret,
            'bench_annual': bench_annual,
            'excess': excess,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'calmar': calmar,
            'win_rate': win_rate,
            'turnover': turnover,
            'n_stops': n_stops,
            'n_cppi_days': n_cppi_days,
            'n_rebal': n_rebal,
            'n_days': n_days,
            'years': years,
            'monthly_returns': dict(monthly_rets),
            'yearly_returns': yearly_rets,
            'nav_series': nav_series,
            'config': cfg,
        }

    def _compute_weights(self, target_codes: list, method: str, date: str) -> dict:
        """计算目标持仓权重, 返回 {code: weight}, 权重和为1"""
        if not target_codes:
            return {}

        codes = [c for c, v in target_codes]
        vals = [max(v, 0) for c, v in target_codes]

        if method == 'score_weighted':
            total = sum(vals)
            if total > 0:
                return {c: v / total for c, v in zip(codes, vals)}

        elif method == 'inv_volatility':
            inv_vols = []
            for c in codes:
                atr_val = self.atr.get(c, {}).get(date)
                price = self.prices.get(c, {}).get(date, {}).get('close', 0)
                if atr_val and price > 0:
                    atr_pct = atr_val / price
                    inv_vols.append(1.0 / max(atr_pct, 1e-6))
                else:
                    inv_vols.append(1.0)  # Fallback to equal
            total = sum(inv_vols)
            if total > 0:
                return {c: v / total for c, v in zip(codes, inv_vols)}

        # Default: equal weight
        w = 1.0 / len(codes)
        return {c: w for c in codes}

    def grid_search(self, grid: dict = None, oos_bt: 'PortfolioBacktester' = None) -> list:
        """网格搜索: 遍历所有参数组合, 返回Top10"""
        grid = grid or DEFAULT_GRID

        # Build all combinations
        keys = sorted(grid.keys())
        combos = list(itertools.product(*(grid[k] for k in keys)))
        print(f"Grid search: {len(combos)} combinations")

        results = []
        for idx, combo in enumerate(combos):
            cfg = {k: v for k, v in zip(keys, combo)}
            # Fill defaults
            for k, v in DEFAULT_CONFIG.items():
                if k not in cfg:
                    cfg[k] = v

            is_res = self.run(cfg)
            if 'error' in is_res:
                continue

            entry = {
                'config': cfg,
                'is_sharpe': is_res['sharpe'],
                'is_excess': is_res['excess'],
                'is_annual': is_res['annual_return'],
                'is_max_dd': is_res['max_dd'],
            }

            if oos_bt:
                oos_res = oos_bt.run(cfg)
                if 'error' in oos_res:
                    continue
                entry['oos_sharpe'] = oos_res['sharpe']
                entry['oos_excess'] = oos_res['excess']
                entry['oos_annual'] = oos_res['annual_return']
                entry['oos_max_dd'] = oos_res['max_dd']

            results.append(entry)

            if (idx + 1) % 50 == 0:
                print(f"  ... {idx + 1}/{len(combos)} done")

        # Filter: both IS and OOS must have Sharpe > 0 and excess > 0
        if oos_bt:
            valid = [r for r in results
                     if r['is_sharpe'] > 0 and r['is_excess'] > 0
                     and r['oos_sharpe'] > 0 and r['oos_excess'] > 0]
            valid.sort(key=lambda r: 0.5 * r['is_sharpe'] + 0.5 * r['oos_sharpe'],
                       reverse=True)
        else:
            valid = [r for r in results if r['is_sharpe'] > 0]
            valid.sort(key=lambda r: r['is_sharpe'], reverse=True)

        return valid[:10]


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _cfg_str(cfg: dict) -> str:
    """简短配置描述"""
    parts = [
        f"top{cfg['top_n']}",
        f"rebal{cfg['rebal_days']}",
        cfg['weighting'][:5],
    ]
    if cfg.get('holding_buffer', 0) > 0:
        parts.append(f"buf{cfg['holding_buffer']}")
    if cfg.get('atr_stop_mult', 0) > 0:
        parts.append(f"atr{cfg['atr_stop_mult']}")
    if cfg.get('max_loss_pct', 0) > 0:
        parts.append(f"loss{cfg['max_loss_pct']}")
    if cfg.get('cppi_floor', 0) > 0:
        parts.append(f"cppi{cfg['cppi_floor']}x{cfg['cppi_multiplier']}")
    return ','.join(parts)


def print_single_result(res: dict):
    """打印单次回测结果"""
    cfg = res['config']
    days = res['n_days']
    years = res['years']

    print(f"\n{'='*50}")
    print(f"  Portfolio Backtest")
    print(f"{'='*50}")
    print(f"Config: top_n={cfg['top_n']}, rebal_days={cfg['rebal_days']}, "
          f"weighting={cfg['weighting']}, atr_stop={cfg['atr_stop_mult']}, "
          f"cppi_floor={cfg['cppi_floor']}, cppi_mult={cfg['cppi_multiplier']}")
    if cfg.get('holding_buffer', 0) > 0:
        print(f"        holding_buffer={cfg['holding_buffer']}")
    if cfg.get('max_loss_pct', 0) > 0:
        print(f"        max_loss_pct={cfg['max_loss_pct']}")

    nav_series = res['nav_series']
    print(f"Period: {nav_series[0][0]} ~ {nav_series[-1][0]} "
          f"({years:.1f}yr, {res['n_rebal']} rebal periods)")
    print()
    print(f"  年化收益: {res['annual_return']:+.1%}  (沪深300: {res['bench_annual']:+.1%})")
    print(f"  年化超额: {res['excess']:+.1%}")
    print(f"  Sharpe:   {res['sharpe']:.2f}")
    print(f"  MaxDD:    {res['max_dd']:.1%}")
    print(f"  Calmar:   {res['calmar']:.2f}")
    print(f"  胜率:     {res['win_rate']:.1%}")
    print(f"  换手率:   {res['turnover']:.0f}%/年")
    print(f"  止损触发: {res['n_stops']}次")
    if cfg.get('cppi_floor', 0) > 0:
        cppi_pct = res['n_cppi_days'] / days * 100 if days > 0 else 0
        print(f"  CPPI减仓: {res['n_cppi_days']}天 ({cppi_pct:.1f}%)")

    yr = res.get('yearly_returns', {})
    if yr:
        parts = [f"{y}={v:+.1%}" for y, v in sorted(yr.items())]
        print(f"\n分年: {' '.join(parts)}")
    print()


def print_grid_results(results: list, n_total: int, has_oos: bool):
    """打印网格搜索结果"""
    n_valid = len(results)
    print(f"\n{'='*100}")
    label = "IS+OOS both Sharpe>0 & excess>0" if has_oos else "IS Sharpe>0"
    print(f"  Grid Search: {n_total} combinations, {n_valid} valid ({label})")
    print(f"{'='*100}")

    if has_oos:
        header = (f"  {'Rank':>4} | {'Score':>6} | {'IS_Shrp':>7} | {'IS_Excs':>8} | "
                  f"{'IS_MaxDD':>8} | {'OOS_Shrp':>8} | {'OOS_Excs':>9} | "
                  f"{'OOS_MaxDD':>9} | Config")
        print(header)
        print(f"  {'-'*4}-+-{'-'*6}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-"
              f"{'-'*9}-+-{'-'*9}-+-{'-'*30}")
        for i, r in enumerate(results):
            score = 0.5 * r['is_sharpe'] + 0.5 * r['oos_sharpe']
            print(f"  {i+1:>4} | {score:>6.3f} | {r['is_sharpe']:>7.2f} | "
                  f"{r['is_excess']:>+7.1%} | {r['is_max_dd']:>7.1%} | "
                  f"{r['oos_sharpe']:>8.2f} | {r['oos_excess']:>+8.1%} | "
                  f"{r['oos_max_dd']:>8.1%} | {_cfg_str(r['config'])}")
    else:
        header = (f"  {'Rank':>4} | {'Sharpe':>6} | {'Annual':>8} | {'Excess':>8} | "
                  f"{'MaxDD':>8} | Config")
        print(header)
        print(f"  {'-'*4}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*30}")
        for i, r in enumerate(results):
            print(f"  {i+1:>4} | {r['is_sharpe']:>6.2f} | {r['is_annual']:>+7.1%} | "
                  f"{r['is_excess']:>+7.1%} | {r['is_max_dd']:>7.1%} | "
                  f"{_cfg_str(r['config'])}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Portfolio Backtest Engine')

    # Mode
    parser.add_argument('--grid', action='store_true', help='Run grid search')

    # Directories
    parser.add_argument('--report-dir', type=str, help='Report directory (single run)')
    parser.add_argument('--is-dir', type=str, help='In-sample report dir (grid search)')
    parser.add_argument('--oos-dir', type=str, help='Out-of-sample report dir (grid search)')
    parser.add_argument('--db-path', type=str, default=DB_PATH)

    # Single-run parameters
    parser.add_argument('--top-n', type=int, default=5)
    parser.add_argument('--score-floor', type=float, default=0)
    parser.add_argument('--holding-buffer', type=int, default=0)
    parser.add_argument('--weighting', type=str, default='equal',
                        choices=['equal', 'score_weighted', 'inv_volatility'])
    parser.add_argument('--rebal-days', type=int, default=10)
    parser.add_argument('--atr-stop', type=float, default=0)
    parser.add_argument('--max-loss', type=float, default=0)
    parser.add_argument('--cppi-floor', type=float, default=0)
    parser.add_argument('--cppi-mult', type=int, default=20)
    parser.add_argument('--cost', type=float, default=0.0015)

    args = parser.parse_args()

    if args.grid:
        # Grid search mode
        is_dir = args.is_dir or args.report_dir
        if not is_dir:
            parser.error("--is-dir or --report-dir required for grid search")

        print("=== In-Sample ===")
        is_bt = PortfolioBacktester(is_dir, args.db_path)

        oos_bt = None
        if args.oos_dir:
            print("=== Out-of-Sample ===")
            oos_bt = PortfolioBacktester(args.oos_dir, args.db_path)

        top10 = is_bt.grid_search(oos_bt=oos_bt)
        n_total = 1
        for vals in DEFAULT_GRID.values():
            n_total *= len(vals)
        print_grid_results(top10, n_total, oos_bt is not None)

    else:
        # Single run mode
        report_dir = args.report_dir
        if not report_dir:
            parser.error("--report-dir required for single run")

        bt = PortfolioBacktester(report_dir, args.db_path)
        cfg = {
            'top_n': args.top_n,
            'score_floor': args.score_floor,
            'holding_buffer': args.holding_buffer,
            'weighting': args.weighting,
            'rebal_days': args.rebal_days,
            'atr_stop_mult': args.atr_stop,
            'max_loss_pct': args.max_loss,
            'cppi_floor': args.cppi_floor,
            'cppi_multiplier': args.cppi_mult,
            'cost_per_side': args.cost,
        }
        res = bt.run(cfg)
        if 'error' in res:
            print(f"Error: {res['error']}")
            sys.exit(1)
        print_single_result(res)


if __name__ == '__main__':
    main()
