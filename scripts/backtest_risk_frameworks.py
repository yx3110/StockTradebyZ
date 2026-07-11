#!/usr/bin/env python3
"""
Backtest multiple risk management frameworks on ng1.0.2 reports.

Strategies compared:
  A. No Risk Mgmt — 100% always, equal weight Top-10
  B. Simple CPPI — floor=5%, mult=20 (death spiral prone)
  C. Webapp-style — regime + circuit breaker + ATR stops + risk parity
  D. Hybrid V2 — fixed-fractional + regime + trailing stop + recovery
     (the one we want to deploy)

Hybrid V2 design:
  - Base: Kelly/Fixed-fractional (always invest at least min_exposure=20%)
  - Regime overlay: bull=100%, neutral=80%, bear=50%
  - Drawdown dampening: gradual, not binary
      exposure *= max(0.3, 1 - 2 * drawdown)
      at 10% DD → 80%, at 20% DD → 60%, at 35% DD → 30% (floor)
  - ATR trailing stop: per-stock, exit if close < highest_since_entry - 2.5*ATR
      but allow RE-ENTRY at next rebalance if stock is still in Top-10
  - Recovery: after DD recovery to within 3% of peak, restore full exposure
  - Sector cap: max 30% per industry
  - Risk parity: inverse volatility weighting
"""

import json, os, sqlite3, glob
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd

PROJECT = "/Users/yangxu/StockTradebyZ"
REPORT_DIR = os.path.join(PROJECT, "reports/daily_selection_ng102")
DB_PATH = os.path.join(PROJECT, "data_adapter/stock_data.db")
CSI300_CODE = "000300.SH"
TOP_N = 10
FOCUS_DAYS = 10
COST_PER_LEG = 0.0015
INITIAL_NAV = 1_000_000.0


def load_data():
    """Load all necessary data."""
    conn = sqlite3.connect(DB_PATH, timeout=30)

    # Reports
    files = sorted(glob.glob(os.path.join(REPORT_DIR, "analysis_data_*.json")))
    reports = {}
    for f in files:
        date = os.path.basename(f).replace("analysis_data_", "").replace(".json", "")
        with open(f) as fh:
            d = json.load(fh)
        stocks = d.get('all_stocks_with_scores', [])
        top = sorted(stocks, key=lambda x: x.get('rank_score', x.get('score', 0)), reverse=True)[:TOP_N]
        codes = [s.get('stock_code', '') for s in top if s.get('stock_code')]
        if codes:
            reports[date] = codes

    # Price data
    print(f"Loading prices...")
    df_prices = pd.read_sql("""
        SELECT s.code, dq.trade_date, dq.open, dq.close, dq.high, dq.low
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.type = 'A股' AND dq.trade_date >= '2018-01-01'
        ORDER BY dq.trade_date
    """, conn)

    # CSI300
    df_bench = pd.read_sql(f"""
        SELECT dq.trade_date, dq.close as bench_close
        FROM daily_quotes dq JOIN securities s ON dq.security_id = s.id
        WHERE s.code = '{CSI300_CODE}' AND dq.trade_date >= '2018-01-01'
        ORDER BY dq.trade_date
    """, conn)

    # ATR
    df_atr = pd.read_sql("""
        SELECT s.code, ti.trade_date, ti.atr_14
        FROM technical_indicators ti
        JOIN securities s ON ti.security_id = s.id
        WHERE ti.trade_date >= '2018-01-01' AND ti.atr_14 IS NOT NULL
    """, conn)

    # Industry
    df_ind = pd.read_sql("SELECT code, industry FROM securities WHERE type='A股'", conn)
    industry_map = dict(zip(df_ind['code'], df_ind['industry']))

    conn.close()

    # Build lookups
    price_map = {}
    for _, row in df_prices.iterrows():
        key = (row['code'], row['trade_date'])
        price_map[key] = {'open': row['open'], 'close': row['close'],
                          'high': row['high'], 'low': row['low']}

    bench_map = dict(zip(df_bench['trade_date'], df_bench['bench_close']))

    atr_map = {}
    for _, row in df_atr.iterrows():
        atr_map[(row['code'], row['trade_date'])] = row['atr_14']

    # Trading dates
    trading_dates = sorted(df_bench['trade_date'].unique())

    # Volatility: 20d std of returns per stock
    vol_data = df_prices.pivot_table(index='trade_date', columns='code', values='close')
    vol_20d = vol_data.pct_change().rolling(20).std()

    return reports, price_map, bench_map, atr_map, industry_map, trading_dates, vol_20d


def get_regime(bench_map, date, trading_dates):
    """Detect market regime from CSI300 20d return."""
    idx = trading_dates.index(date) if date in trading_dates else -1
    if idx < 20:
        return 'neutral'
    d20_ago = trading_dates[idx - 20]
    p_now = bench_map.get(date, 0)
    p_20 = bench_map.get(d20_ago, 0)
    if p_20 <= 0:
        return 'neutral'
    ret = p_now / p_20 - 1
    if ret > 0.03:
        return 'bull'
    elif ret < -0.03:
        return 'bear'
    return 'neutral'


class Strategy:
    """Base strategy."""
    def __init__(self, name):
        self.name = name
        self.nav = INITIAL_NAV
        self.cash = INITIAL_NAV
        self.positions = {}  # code -> {qty, entry_price, highest_price}
        self.peak_nav = INITIAL_NAV
        self.nav_history = []
        self.trade_count = 0
        self.wins = 0
        self.rebalances = 0

    def get_exposure(self, date, bench_map, trading_dates):
        return 1.0

    def get_weights(self, codes, date, vol_20d, industry_map):
        """Equal weight."""
        n = len(codes)
        return {c: 1.0 / n for c in codes} if n > 0 else {}

    def check_stops(self, date, price_map, atr_map):
        """Check stop losses. Return list of codes to exit."""
        return []

    def rebalance(self, date, codes, price_map, bench_map, atr_map,
                  industry_map, trading_dates, vol_20d):
        """Execute rebalance."""
        self.rebalances += 1

        # Check stops first
        stopped = self.check_stops(date, price_map, atr_map)
        for code in stopped:
            if code in self.positions:
                pos = self.positions[code]
                close = price_map.get((code, date), {}).get('close', 0)
                if close > 0:
                    self.cash += pos['qty'] * close * (1 - COST_PER_LEG)
                    self.trade_count += 1
                del self.positions[code]

        # Get target exposure
        exposure = self.get_exposure(date, bench_map, trading_dates)

        # Get weights
        weights = self.get_weights(codes, date, vol_20d, industry_map)

        # Calculate target portfolio
        target_invested = self.nav * exposure
        target_positions = {}
        for code, w in weights.items():
            price = price_map.get((code, date), {}).get('open', 0)
            if price > 0:
                target_value = target_invested * w
                qty = int(target_value / price / 100) * 100
                if qty >= 100:
                    target_positions[code] = {'qty': qty, 'price': price}

        # Pre-rebalance NAV for win tracking
        pre_nav = self.nav

        # Sell positions not in target
        for code in list(self.positions.keys()):
            if code not in target_positions:
                pos = self.positions[code]
                sell_price = price_map.get((code, date), {}).get('open', 0)
                if sell_price > 0:
                    self.cash += pos['qty'] * sell_price * (1 - COST_PER_LEG)
                    self.trade_count += 1
                del self.positions[code]

        # Buy new targets
        for code, target in target_positions.items():
            current_qty = self.positions.get(code, {}).get('qty', 0)
            if current_qty == target['qty']:
                continue  # no change

            # Sell excess
            if current_qty > target['qty']:
                diff = current_qty - target['qty']
                sell_price = target['price']
                self.cash += diff * sell_price * (1 - COST_PER_LEG)
                self.positions[code]['qty'] = target['qty']
                self.trade_count += 1

            # Buy deficit
            elif current_qty < target['qty']:
                diff = target['qty'] - current_qty
                buy_cost = diff * target['price'] * (1 + COST_PER_LEG)
                if buy_cost <= self.cash:
                    self.cash -= buy_cost
                    if code in self.positions:
                        self.positions[code]['qty'] = target['qty']
                    else:
                        self.positions[code] = {
                            'qty': target['qty'],
                            'entry_price': target['price'],
                            'highest_price': target['price'],
                        }
                    self.trade_count += 1

    def update_daily(self, date, price_map):
        """Update NAV and position tracking daily."""
        total_mv = 0
        for code, pos in self.positions.items():
            close = price_map.get((code, date), {}).get('close', 0)
            if close > 0:
                total_mv += pos['qty'] * close
                # Update highest price for trailing stops
                if close > pos.get('highest_price', 0):
                    pos['highest_price'] = close

        self.nav = self.cash + total_mv
        if self.nav > self.peak_nav:
            self.peak_nav = self.nav
        self.nav_history.append({'date': date, 'nav': self.nav})

    def metrics(self):
        """Compute performance metrics."""
        if not self.nav_history:
            return {}
        navs = np.array([h['nav'] for h in self.nav_history])
        returns = np.diff(navs) / navs[:-1]
        returns = returns[~np.isnan(returns)]

        ann_ret = (navs[-1] / navs[0]) ** (252 / max(len(navs), 1)) - 1
        max_dd = 0
        peak = navs[0]
        dd_days = 0
        max_dd_days = 0
        for n in navs:
            if n > peak:
                peak = n
                dd_days = 0
            dd = (peak - n) / peak
            dd_days += 1
            if dd > max_dd:
                max_dd = dd
            if dd_days > max_dd_days and dd > 0.001:
                max_dd_days = dd_days

        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252) if len(returns) > 10 else 0
        cvar = np.mean(returns[returns <= np.percentile(returns, 5)]) * np.sqrt(252) if len(returns) > 20 else 0

        avg_exposure = 0
        if self.nav_history:
            exposures = [(h['nav'] - INITIAL_NAV * 0.01) / h['nav'] for h in self.nav_history]
            # Rough: invested / nav
            pass

        return {
            'Annual Return': f"{ann_ret*100:.1f}%",
            'Total Return': f"{(navs[-1]/navs[0]-1)*100:.1f}%",
            'Max Drawdown': f"{max_dd*100:.1f}%",
            'Sharpe': f"{sharpe:.3f}",
            'Trades': self.trade_count,
            'Rebalances': self.rebalances,
            'Final NAV': f"{navs[-1]:,.0f}",
            'CVaR 5%': f"{cvar*100:.2f}%",
            'Longest DD': f"{max_dd_days}d",
        }


class NoRiskStrategy(Strategy):
    """A. Always 100% invested, equal weight."""
    def __init__(self):
        super().__init__("A. 无风控 (100%满仓)")


class SimpleCPPI(Strategy):
    """B. Simple CPPI."""
    def __init__(self):
        super().__init__("B. Simple CPPI")
        self.floor = 0.05
        self.mult = 20

    def get_exposure(self, date, bench_map, trading_dates):
        floor_nav = self.peak_nav * (1 - self.floor)
        cushion = self.nav - floor_nav
        if self.nav <= 0:
            return 0
        exp = self.mult * cushion / self.nav
        return min(1.3, max(0, exp))


class WebappStyle(Strategy):
    """C. Webapp-style: regime + circuit breaker + ATR stops + risk parity."""
    def __init__(self):
        super().__init__("C. Webapp风控")
        self.floor = 0.05
        self.mult = 20

    def get_exposure(self, date, bench_map, trading_dates):
        # CPPI base
        floor_nav = self.peak_nav * (1 - self.floor)
        cushion = self.nav - floor_nav
        cppi_exp = self.mult * cushion / (self.nav + 1e-8)
        cppi_exp = max(0, cppi_exp)

        # Regime cap
        regime = get_regime(bench_map, date, trading_dates)
        regime_cap = {'bull': 0.85, 'neutral': 0.70, 'bear': 0.40}.get(regime, 0.70)

        # Circuit breaker
        dd = (self.peak_nav - self.nav) / (self.peak_nav + 1e-8)
        cb_mult = 1.0
        if dd > 0.15:
            cb_mult = 0.4
        elif dd > 0.10:
            cb_mult = 0.8

        return min(cppi_exp, regime_cap) * cb_mult

    def get_weights(self, codes, date, vol_20d, industry_map):
        # Risk parity: inverse volatility
        weights = {}
        for code in codes:
            vol = vol_20d.get((date, code), None) if isinstance(vol_20d, dict) else None
            if vol is None:
                try:
                    vol = vol_20d.loc[date, code] if date in vol_20d.index and code in vol_20d.columns else 0.05
                except Exception:
                    vol = 0.05
            if pd.isna(vol) or vol < 0.001:
                vol = 0.05
            weights[code] = 1.0 / vol

        # Sector cap: 30%
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        # Apply sector cap
        sector_weights = defaultdict(float)
        for code, w in weights.items():
            ind = industry_map.get(code, 'other')
            sector_weights[ind] += w
        for ind, sw in sector_weights.items():
            if sw > 0.30:
                scale = 0.30 / sw
                for code in list(weights.keys()):
                    if industry_map.get(code, 'other') == ind:
                        weights[code] *= scale
        # Re-normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        return weights

    def check_stops(self, date, price_map, atr_map):
        stopped = []
        for code, pos in self.positions.items():
            close = price_map.get((code, date), {}).get('close', 0)
            atr = atr_map.get((code, date), 0)
            if close > 0 and atr > 0:
                stop_level = pos.get('highest_price', pos['entry_price']) - 2.5 * atr
                if close < stop_level:
                    stopped.append(code)
        return stopped


class HybridV2(Strategy):
    """D. Hybrid V2: fixed-fractional + regime + trailing stops + recovery."""
    def __init__(self):
        super().__init__("D. Hybrid V2 (推荐)")
        self.min_exposure = 0.20  # Never go below 20%

    def get_exposure(self, date, bench_map, trading_dates):
        # Regime overlay
        regime = get_regime(bench_map, date, trading_dates)
        regime_mult = {'bull': 1.0, 'neutral': 0.80, 'bear': 0.50}.get(regime, 0.80)

        # Drawdown dampening (gradual, not binary)
        dd = (self.peak_nav - self.nav) / (self.peak_nav + 1e-8)
        dd_mult = max(0.3, 1.0 - 2.0 * dd)
        # dd=0% → 1.0, dd=5% → 0.9, dd=10% → 0.8, dd=20% → 0.6, dd=35% → 0.3

        # Combined
        exposure = regime_mult * dd_mult

        # Floor: never below min_exposure (prevents death spiral)
        exposure = max(self.min_exposure, exposure)

        return min(1.3, exposure)  # max leverage cap

    def get_weights(self, codes, date, vol_20d, industry_map):
        # Risk parity + sector cap (same as Webapp)
        weights = {}
        for code in codes:
            try:
                vol = vol_20d.loc[date, code] if date in vol_20d.index and code in vol_20d.columns else 0.05
            except Exception:
                vol = 0.05
            if pd.isna(vol) or vol < 0.001:
                vol = 0.05
            weights[code] = 1.0 / vol

        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        # Sector cap 30%
        sector_weights = defaultdict(float)
        for code, w in weights.items():
            ind = industry_map.get(code, 'other')
            sector_weights[ind] += w
        for ind, sw in sector_weights.items():
            if sw > 0.30:
                scale = 0.30 / sw
                for code in list(weights.keys()):
                    if industry_map.get(code, 'other') == ind:
                        weights[code] *= scale
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        return weights

    def check_stops(self, date, price_map, atr_map):
        """Trailing stop: exit if close < highest - 2.5*ATR.
        But allow re-entry at next rebalance (unlike Webapp which permanently exits)."""
        stopped = []
        for code, pos in self.positions.items():
            close = price_map.get((code, date), {}).get('close', 0)
            atr = atr_map.get((code, date), 0)
            if close > 0 and atr > 0:
                stop_level = pos.get('highest_price', pos['entry_price']) - 2.5 * atr
                if close < stop_level:
                    stopped.append(code)
        return stopped


def run_backtest(strategies, reports, price_map, bench_map, atr_map,
                 industry_map, trading_dates, vol_20d):
    """Run backtest for all strategies."""
    report_dates = sorted(reports.keys())

    # Group into rebalance periods
    rebalance_dates = []
    i = 0
    while i < len(report_dates):
        d = report_dates[i]
        # Find next trading date after report date
        td_idx = next((j for j, td in enumerate(trading_dates) if td >= d), None)
        if td_idx is not None and td_idx + 1 < len(trading_dates):
            entry_date = trading_dates[td_idx + 1]  # buy next day
            rebalance_dates.append((d, entry_date))
        i += FOCUS_DAYS  # skip ahead

    print(f"Rebalance periods: {len(rebalance_dates)}")

    # Run daily
    reb_idx = 0
    next_reb_date = rebalance_dates[0][1] if rebalance_dates else None

    for date in trading_dates:
        if date < '2018-04-01':
            continue

        # Check if rebalance day
        if reb_idx < len(rebalance_dates) and date >= rebalance_dates[reb_idx][1]:
            report_date = rebalance_dates[reb_idx][0]
            codes = reports.get(report_date, [])

            for strat in strategies:
                if codes:
                    strat.rebalance(date, codes, price_map, bench_map, atr_map,
                                    industry_map, trading_dates, vol_20d)

            reb_idx += 1

        # Daily update
        for strat in strategies:
            # Check stops on non-rebalance days too
            if date != next_reb_date:
                stopped = strat.check_stops(date, price_map, atr_map)
                for code in stopped:
                    if code in strat.positions:
                        pos = strat.positions[code]
                        close = price_map.get((code, date), {}).get('close', 0)
                        if close > 0:
                            strat.cash += pos['qty'] * close * (1 - COST_PER_LEG)
                            strat.trade_count += 1
                        del strat.positions[code]

            strat.update_daily(date, price_map)

        if reb_idx < len(rebalance_dates):
            next_reb_date = rebalance_dates[reb_idx][1]

    return strategies


def main():
    print("Loading data...")
    reports, price_map, bench_map, atr_map, industry_map, trading_dates, vol_20d = load_data()
    print(f"Reports: {len(reports)}, Trading dates: {len(trading_dates)}")

    strategies = [
        NoRiskStrategy(),
        SimpleCPPI(),
        WebappStyle(),
        HybridV2(),
    ]

    print("Running backtest...")
    strategies = run_backtest(strategies, reports, price_map, bench_map, atr_map,
                              industry_map, trading_dates, vol_20d)

    # Print results
    print("\n" + "=" * 90)
    print("RISK FRAMEWORK COMPARISON — ng1.0.2 Reports, 2018-2026")
    print("=" * 90)

    metrics_keys = ['Annual Return', 'Total Return', 'Max Drawdown', 'Sharpe',
                    'Trades', 'Final NAV', 'CVaR 5%', 'Longest DD']

    header = f"{'Metric':<20}"
    for s in strategies:
        header += f" {s.name:>20}"
    print(header)
    print("-" * (20 + 21 * len(strategies)))

    all_metrics = [s.metrics() for s in strategies]
    for key in metrics_keys:
        row = f"{key:<20}"
        for m in all_metrics:
            row += f" {m.get(key, 'N/A'):>20}"
        print(row)

    # Save markdown
    md = ["# 风控框架回测对比", ""]
    md.append(f"**数据**: ng1.0.2 reports, 2018-04 → 2026-04, Top-10, Hold 10d")
    md.append(f"**初始资金**: {INITIAL_NAV:,.0f}元")
    md.append("")
    md.append("## 结果对比")
    md.append("")
    md.append("| 指标 | " + " | ".join(s.name for s in strategies) + " |")
    md.append("|------|" + "|".join("------" for _ in strategies) + "|")
    for key in metrics_keys:
        row = f"| {key} |"
        for m in all_metrics:
            row += f" {m.get(key, 'N/A')} |"
        md.append(row)

    md.append("")
    md.append("## 策略说明")
    md.append("")
    md.append("### A. 无风控")
    md.append("100%满仓，等权分配Top-10，无止损无仓位控制")
    md.append("")
    md.append("### B. Simple CPPI")
    md.append("Floor=5%, Multiplier=20. 亏超5%后exposure=0%永久锁死（死亡螺旋）")
    md.append("")
    md.append("### C. Webapp风控")
    md.append("CPPI + 市况适应 + 熔断 + ATR止损 + 风险平价. 改善了死亡螺旋但仍受CPPI限制")
    md.append("")
    md.append("### D. Hybrid V2 (推荐)")
    md.append("- **无CPPI**: 用渐进式回撤衰减替代 (dd_mult = max(0.3, 1-2*dd))")
    md.append("- **最低仓位20%**: 永远不会锁死到0%")
    md.append("- **市况适应**: 牛市100%, 中性80%, 熊市50%")
    md.append("- **ATR追踪止损**: 止损后下次调仓可重新入场")
    md.append("- **风险平价**: 反波动率加权 + 行业30%上限")
    md.append("- **回撤衰减公式**: 0%回撤→满仓, 10%→80%, 20%→60%, 35%→30%(底)")
    md.append("")

    out_path = os.path.join(PROJECT, "reports/risk_framework_comparison.md")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print(f"\nReport saved: {out_path}")


if __name__ == '__main__':
    main()
