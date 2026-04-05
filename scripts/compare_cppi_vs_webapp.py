#!/usr/bin/env python3
"""
compare_cppi_vs_webapp.py
=========================
Backtest comparison: Simple CPPI vs Webapp-style Advanced Risk Management
on the same ng1.0.2 stock selection reports.

Strategy A: Simple CPPI (EastMoneyTrader style)
  - Floor=5%, Multiplier=20, equal-weight Top-10, no stops

Strategy B: Webapp Advanced Risk Management
  - Market regime detection (CSI300 20d return)
  - Circuit breaker (drawdown 10%/15%)
  - ATR stop-loss (2.5 * ATR_14)
  - Risk-parity weighting (inverse 20d volatility)
  - Sector limit (25% max per industry)
  - CPPI base with regime cap

Usage:
  python3 scripts/compare_cppi_vs_webapp.py
"""

import json
import os
import sqlite3
import glob
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd

# ── paths ───────────────────────────────────────────────────────────────────
PROJECT = "/Users/yangxu/StockTradebyZ"
REPORT_DIR = os.path.join(PROJECT, "reports/daily_selection_ng102")
DB_PATH = os.path.join(PROJECT, "data_adapter/stock_data.db")
OUTPUT_PATH = os.path.join(PROJECT, "reports/cppi_vs_webapp_comparison.md")
CSI300_CODE = "000300.SH"
TOP_N = 10
FOCUS_DAYS = 10
TRADE_COST = 0.0015  # 0.15% per leg
INITIAL_NAV = 1_000_000.0

# ── database helpers ────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def load_security_map(conn):
    """Return dict: bare code (no suffix) -> (security_id, exchange, industry)."""
    cur = conn.execute("SELECT id, code, exchange, industry FROM securities")
    mapping = {}
    for r in cur:
        code = r["code"]
        bare = code.split(".")[0] if "." in code else code
        mapping[bare] = (r["id"], r["exchange"] or "", r["industry"] or "")
    return mapping


def load_all_quotes(conn):
    """Load daily_quotes into DataFrame."""
    df = pd.read_sql_query(
        "SELECT security_id, trade_date, open, high, low, close FROM daily_quotes",
        conn,
    )
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def load_csi300(conn, sec_map):
    """Load CSI300 close series."""
    bare = CSI300_CODE.split(".")[0]
    if bare in sec_map:
        sid = sec_map[bare][0]
    else:
        cur = conn.execute("SELECT id FROM securities WHERE code=?", (CSI300_CODE,))
        row = cur.fetchone()
        if not row:
            raise RuntimeError(f"CSI300 ({CSI300_CODE}) not found")
        sid = row["id"]

    df = pd.read_sql_query(
        "SELECT trade_date, close FROM daily_quotes WHERE security_id=? ORDER BY trade_date",
        conn, params=(sid,),
    )
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["close"]


def load_atr_map(conn):
    """Return dict: (security_id, trade_date_str) -> atr_14."""
    cur = conn.execute(
        "SELECT security_id, trade_date, atr_14 FROM technical_indicators WHERE atr_14 IS NOT NULL"
    )
    return {(r["security_id"], r["trade_date"]): r["atr_14"] for r in cur}


# ── report loading ──────────────────────────────────────────────────────────

def load_reports():
    """Return list of (date_str, top-10 stocks sorted by rank_score desc)."""
    pattern = os.path.join(REPORT_DIR, "analysis_data_*.json")
    files = sorted(glob.glob(pattern))
    reports = []
    for fp in files:
        basename = os.path.basename(fp)
        date_str = basename.replace("analysis_data_", "").replace(".json", "")
        try:
            with open(fp) as f:
                data = json.load(f)
        except Exception:
            continue
        stocks = data.get("all_stocks_with_scores", [])
        if not stocks:
            continue
        stocks_sorted = sorted(stocks, key=lambda s: s.get("rank_score", 0), reverse=True)
        reports.append((date_str, stocks_sorted[:TOP_N]))
    return reports


# ── trading calendar ────────────────────────────────────────────────────────

def build_trading_calendar(quotes_df):
    dates = sorted(quotes_df["trade_date"].unique())
    return pd.DatetimeIndex(dates)


def next_n_trading_days(cal, date, n):
    idx = cal.searchsorted(date, side="right")
    return cal[idx: idx + n]


# ── price lookup ────────────────────────────────────────────────────────────

class PriceLookup:
    """Fast price lookup via per-security date-indexed DataFrames."""

    def __init__(self, quotes_df):
        self._cache = {}
        for sid, grp in quotes_df.groupby("security_id"):
            self._cache[sid] = grp.set_index("trade_date").sort_index()

    def get(self, sid, date, field="open"):
        g = self._cache.get(sid)
        if g is None or date not in g.index:
            return None
        return float(g.loc[date, field])

    def get_volatility_20d(self, sid, date):
        """20-day realized vol (std of daily returns)."""
        g = self._cache.get(sid)
        if g is None:
            return None
        idx = g.index.searchsorted(date, side="right")
        if idx < 21:
            return None
        rets = g.iloc[idx - 21: idx]["close"].pct_change().dropna()
        if len(rets) < 10:
            return None
        return float(rets.std())

    def get_hl_range_20d(self, sid, date):
        """Fallback ATR: avg(high-low) over past 20 days."""
        g = self._cache.get(sid)
        if g is None:
            return None
        idx = g.index.searchsorted(date, side="right")
        if idx < 20:
            return None
        window = g.iloc[idx - 20: idx]
        return float((window["high"] - window["low"]).mean())


# ── Portfolio base class ────────────────────────────────────────────────────

class Portfolio:
    """Shared portfolio accounting."""

    def __init__(self, initial_nav=INITIAL_NAV):
        self.cash = initial_nav
        self.positions = {}   # sid -> dict(shares, entry_price, ...)
        self.nav_history = []  # [(date, nav)]
        self.trade_count = 0
        self.peak_nav = initial_nav
        self.rebalance_navs = []  # NAV at start of each holding period

    def _position_value(self, prices_lookup, date):
        """Sum of current position values at close."""
        total = 0.0
        for sid, pos in self.positions.items():
            price = prices_lookup.get(sid, date, "close")
            if price is None:
                price = pos["entry_price"]
            total += pos["shares"] * price
        return total

    def get_nav(self, prices_lookup, date):
        return self.cash + self._position_value(prices_lookup, date)

    def mark_to_market(self, date, prices_lookup):
        nav = self.get_nav(prices_lookup, date)
        self.peak_nav = max(self.peak_nav, nav)
        self.nav_history.append((date, nav))
        return nav

    def liquidate_all(self, prices_lookup, date, price_field="open"):
        """Sell all positions at given price field. Return total proceeds after cost."""
        if not self.positions:
            return 0.0
        proceeds = 0.0
        for sid, pos in list(self.positions.items()):
            price = prices_lookup.get(sid, date, price_field)
            if price is None:
                price = pos["entry_price"]
            value = pos["shares"] * price
            cost = value * TRADE_COST
            proceeds += value - cost
            self.trade_count += 1
        self.positions = {}
        self.cash += proceeds
        return proceeds


# ── Strategy A: Simple CPPI ────────────────────────────────────────────────

class SimpleCPPI(Portfolio):
    """Floor=5%, Multiplier=20, equal-weight, no stops."""

    def __init__(self, floor_pct=0.05, multiplier=20):
        super().__init__()
        self.floor_pct = floor_pct
        self.multiplier = multiplier

    def compute_exposure(self):
        nav = self.cash  # called after liquidation, cash = total NAV
        floor_nav = self.peak_nav * (1 - self.floor_pct)
        if nav <= 0:
            return 0.0
        cushion = max(0, nav - floor_nav) / nav
        return min(1.3, max(0, self.multiplier * cushion))

    def rebalance(self, target_sids, prices_lookup, buy_date):
        # sell everything at open
        self.liquidate_all(prices_lookup, buy_date, "open")

        # update peak after liquidation
        self.peak_nav = max(self.peak_nav, self.cash)

        exposure = self.compute_exposure()
        invest = self.cash * exposure

        if invest < 1000:  # skip if negligible
            return

        # filter valid stocks
        valid = [(sid, prices_lookup.get(sid, buy_date, "open"))
                 for sid in target_sids]
        valid = [(sid, p) for sid, p in valid if p and p > 0]
        if not valid:
            return

        w = 1.0 / len(valid)
        for sid, price in valid:
            alloc = invest * w
            if alloc < 100:
                continue
            shares = alloc / price
            cost = alloc * TRADE_COST
            self.cash -= (alloc + cost)
            self.positions[sid] = {"shares": shares, "entry_price": price}
            self.trade_count += 1


# ── Strategy B: Webapp Advanced ────────────────────────────────────────────

class WebappAdvanced(Portfolio):
    """Regime + circuit breaker + ATR stop + risk-parity + sector cap + CPPI."""

    def __init__(self, floor_pct=0.05, multiplier=20):
        super().__init__()
        self.floor_pct = floor_pct
        self.multiplier = multiplier

    def detect_regime(self, csi300, date):
        idx = csi300.index.searchsorted(date, side="right")
        if idx < 20:
            return "neutral", 0.70
        recent = csi300.iloc[idx - 20: idx]
        if len(recent) < 2:
            return "neutral", 0.70
        ret = recent.iloc[-1] / recent.iloc[0] - 1
        if ret > 0.03:
            return "bull", 0.85
        elif ret < -0.03:
            return "bear", 0.40
        return "neutral", 0.70

    def circuit_breaker(self):
        if self.peak_nav <= 0:
            return 1.0
        dd = (self.peak_nav - self.cash) / self.peak_nav  # cash = NAV after liquidation
        if dd > 0.15:
            return 0.4
        elif dd > 0.10:
            return 0.8
        return 1.0

    def compute_exposure(self, regime_cap):
        nav = self.cash
        floor_nav = self.peak_nav * (1 - self.floor_pct)
        if nav <= 0:
            return 0.0
        cushion = max(0, nav - floor_nav) / nav
        cppi = min(1.3, max(0, self.multiplier * cushion))
        exposure = min(cppi, regime_cap)
        exposure *= self.circuit_breaker()
        return exposure

    def rebalance(self, target_sids, weights, prices_lookup, buy_date,
                  csi300, atr_map):
        self.liquidate_all(prices_lookup, buy_date, "open")
        self.peak_nav = max(self.peak_nav, self.cash)

        _, regime_cap = self.detect_regime(csi300, buy_date)
        exposure = self.compute_exposure(regime_cap)
        invest = self.cash * exposure

        if invest < 1000:  # skip if negligible
            return

        for sid in target_sids:
            w = weights.get(sid, 0)
            if w <= 0:
                continue
            price = prices_lookup.get(sid, buy_date, "open")
            if not price or price <= 0:
                continue

            alloc = invest * w
            if alloc < 100:
                continue
            shares = alloc / price
            cost = alloc * TRADE_COST
            self.cash -= (alloc + cost)

            # ATR stop
            date_str = buy_date.strftime("%Y-%m-%d")
            atr = atr_map.get((sid, date_str))
            if atr is None:
                atr = prices_lookup.get_hl_range_20d(sid, buy_date)
            stop = (price - 2.5 * atr) if atr and atr > 0 else None

            self.positions[sid] = {
                "shares": shares, "entry_price": price, "stop": stop
            }
            self.trade_count += 1

    def check_stops(self, date, prices_lookup):
        """Execute ATR stops: sell at close if low < stop."""
        triggered = []
        for sid, pos in self.positions.items():
            stop = pos.get("stop")
            if stop is None:
                continue
            low = prices_lookup.get(sid, date, "low")
            if low is not None and low < stop:
                close = prices_lookup.get(sid, date, "close")
                if close is None:
                    close = pos["entry_price"]
                value = pos["shares"] * close
                cost = value * TRADE_COST
                self.cash += value - cost
                self.trade_count += 1
                triggered.append(sid)
        for sid in triggered:
            del self.positions[sid]


# ── risk-parity + sector limit weights ──────────────────────────────────────

def compute_risk_parity_weights(sids, prices_lookup, date, sid_to_industry, sector_limit=0.25):
    """Inverse-volatility weights with 25% sector cap."""
    vols = {}
    for sid in sids:
        v = prices_lookup.get_volatility_20d(sid, date)
        vols[sid] = v if v and v > 0 else 0.03

    inv = {sid: 1.0 / v for sid, v in vols.items()}
    total = sum(inv.values())
    if total <= 0:
        w = 1.0 / len(sids) if sids else 0
        return {sid: w for sid in sids}

    weights = {sid: iv / total for sid, iv in inv.items()}

    # iterative sector cap
    for _ in range(5):
        sector_w = defaultdict(float)
        sector_m = defaultdict(list)
        for sid in sids:
            ind = sid_to_industry.get(sid, "Unknown")
            sector_w[ind] += weights.get(sid, 0)
            sector_m[ind].append(sid)

        capped = False
        for ind, sw in sector_w.items():
            if sw > sector_limit:
                scale = sector_limit / sw
                for sid in sector_m[ind]:
                    weights[sid] *= scale
                capped = True

        if not capped:
            break
        total = sum(weights.values())
        if total > 0:
            weights = {sid: w / total for sid, w in weights.items()}

    return weights


# ── metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(nav_history, trade_count, initial_nav=INITIAL_NAV):
    if not nav_history:
        return {}

    dates, navs = zip(*nav_history)
    navs = np.array(navs, dtype=float)
    dates = list(dates)

    total_days = (dates[-1] - dates[0]).days
    years = total_days / 365.25 if total_days > 0 else 1

    final_nav = navs[-1]
    total_ret = final_nav / initial_nav - 1
    annual_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else total_ret

    daily_rets = np.diff(navs) / navs[:-1]
    daily_rets = daily_rets[np.isfinite(daily_rets)]

    sharpe = 0.0
    if len(daily_rets) > 1 and np.std(daily_rets) > 0:
        sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)

    running_max = np.maximum.accumulate(navs)
    drawdowns = (running_max - navs) / running_max
    max_dd = float(np.max(drawdowns))

    # longest drawdown (trading days)
    in_dd = drawdowns > 0.001
    longest_dd = current_dd = 0
    for v in in_dd:
        if v:
            current_dd += 1
            longest_dd = max(longest_dd, current_dd)
        else:
            current_dd = 0

    # win rate: computed from per-rebalance returns (every FOCUS_DAYS)
    # sample NAV at rebalance boundaries
    rebal_rets = []
    step = FOCUS_DAYS
    for i in range(0, len(navs) - step, step):
        r = (navs[i + step] - navs[i]) / navs[i]
        rebal_rets.append(r)
    wins = sum(1 for r in rebal_rets if r > 0)
    win_rate = wins / len(rebal_rets) if rebal_rets else 0

    # CVaR 5%
    cvar_5 = 0.0
    if len(daily_rets) > 20:
        sorted_rets = np.sort(daily_rets)
        n5 = max(1, int(len(sorted_rets) * 0.05))
        cvar_5 = float(np.mean(sorted_rets[:n5]))

    annual_turnover = trade_count / years if years > 0 else trade_count

    return {
        "Annual Return (%)": annual_ret * 100,
        "Total Return (%)": total_ret * 100,
        "Max Drawdown (%)": max_dd * 100,
        "Sharpe Ratio": sharpe,
        "Win Rate (%)": win_rate * 100,
        "CVaR 5% (%)": cvar_5 * 100,
        "Annual Turnover": annual_turnover,
        "Number of Trades": trade_count,
        "Longest DD Period (days)": longest_dd,
        "Final NAV": final_nav,
    }


# ── main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("CPPI vs Webapp Advanced Risk Management Comparison")
    print("=" * 70)

    print("\n[1/5] Loading reports...")
    reports = load_reports()
    print(f"  Found {len(reports)} report dates ({reports[0][0]} to {reports[-1][0]})")

    print("[2/5] Loading database...")
    conn = get_db()
    sec_map = load_security_map(conn)
    sid_to_industry = {sid: ind for _, (sid, _, ind) in sec_map.items()}

    print("[3/5] Loading price data...")
    quotes_df = load_all_quotes(conn)
    prices = PriceLookup(quotes_df)
    cal = build_trading_calendar(quotes_df)
    print(f"  Trading calendar: {len(cal)} days")

    print("[4/5] Loading CSI300 & ATR...")
    csi300 = load_csi300(conn, sec_map)
    atr_map = load_atr_map(conn)
    print(f"  CSI300: {len(csi300)} days, ATR entries: {len(atr_map)}")
    conn.close()

    def resolve_sids(top_stocks):
        result = []
        for s in top_stocks:
            code = s.get("stock_code", "")
            if code in sec_map:
                result.append(sec_map[code][0])
        return result

    # build report date map
    report_map = {}
    for date_str, top in reports:
        dt = pd.Timestamp(datetime.strptime(date_str, "%Y%m%d"))
        report_map[dt] = top
    sorted_report_dates = sorted(report_map.keys())

    def find_latest_report(trade_date):
        idx = pd.DatetimeIndex(sorted_report_dates).searchsorted(trade_date, side="right") - 1
        return sorted_report_dates[idx] if idx >= 0 else None

    # find first valid trading day
    first_report = sorted_report_dates[0]
    next_days = next_n_trading_days(cal, first_report, 1)
    if len(next_days) == 0:
        print("ERROR: No trading days after first report")
        return
    all_trade_dates = cal[cal >= next_days[0]]

    # init strategies
    strat_a = SimpleCPPI()
    strat_b = WebappAdvanced()

    print(f"\n[5/5] Running backtest (rebalance every {FOCUS_DAYS} trading days)...")

    days_since_rebalance = FOCUS_DAYS  # trigger immediately
    num_rebalances = 0
    active_rebalances_a = 0  # rebalances where A actually invested
    active_rebalances_b = 0
    exposure_history_a = []  # (date, exposure)
    exposure_history_b = []
    stop_triggers_b = 0
    regime_counts = {"bull": 0, "neutral": 0, "bear": 0}

    for td in all_trade_dates:
        # Strategy B: check ATR stops daily
        before_stops = len(strat_b.positions)
        strat_b.check_stops(td, prices)
        stop_triggers_b += (before_stops - len(strat_b.positions))

        if days_since_rebalance >= FOCUS_DAYS:
            rpt_date = find_latest_report(td)
            if rpt_date is None:
                days_since_rebalance += 1
                strat_a.mark_to_market(td, prices)
                strat_b.mark_to_market(td, prices)
                continue

            top = report_map[rpt_date]
            sids = resolve_sids(top)
            if not sids:
                days_since_rebalance += 1
                strat_a.mark_to_market(td, prices)
                strat_b.mark_to_market(td, prices)
                continue

            # rebalance A
            strat_a.rebalance(sids, prices, td)

            # rebalance B with risk-parity weights
            rp_weights = compute_risk_parity_weights(
                sids, prices, td, sid_to_industry
            )
            strat_b.rebalance(sids, rp_weights, prices, td, csi300, atr_map)

            # track exposure
            pos_val_a = sum(p["shares"] * (prices.get(s, td, "open") or p["entry_price"])
                           for s, p in strat_a.positions.items())
            nav_a_approx = strat_a.cash + pos_val_a
            exp_a = pos_val_a / nav_a_approx if nav_a_approx > 0 else 0

            pos_val_b = sum(p["shares"] * (prices.get(s, td, "open") or p["entry_price"])
                           for s, p in strat_b.positions.items())
            nav_b_approx = strat_b.cash + pos_val_b
            exp_b = pos_val_b / nav_b_approx if nav_b_approx > 0 else 0

            exposure_history_a.append((td, exp_a))
            exposure_history_b.append((td, exp_b))

            if len(strat_a.positions) > 0:
                active_rebalances_a += 1
            if len(strat_b.positions) > 0:
                active_rebalances_b += 1

            # track regime
            regime, _ = strat_b.detect_regime(csi300, td)
            regime_counts[regime] += 1

            days_since_rebalance = 0
            num_rebalances += 1

        days_since_rebalance += 1
        strat_a.mark_to_market(td, prices)
        strat_b.mark_to_market(td, prices)

    # results
    metrics_a = compute_metrics(strat_a.nav_history, strat_a.trade_count)
    metrics_b = compute_metrics(strat_b.nav_history, strat_b.trade_count)

    # exposure stats
    exp_a_vals = [e for _, e in exposure_history_a]
    exp_b_vals = [e for _, e in exposure_history_b]
    avg_exp_a = np.mean(exp_a_vals) if exp_a_vals else 0
    avg_exp_b = np.mean(exp_b_vals) if exp_b_vals else 0
    # find last active date for each
    last_active_a = "never"
    last_active_b = "never"
    for dt, e in reversed(exposure_history_a):
        if e > 0.01:
            last_active_a = dt.strftime("%Y-%m-%d")
            break
    for dt, e in reversed(exposure_history_b):
        if e > 0.01:
            last_active_b = dt.strftime("%Y-%m-%d")
            break

    print(f"\n  Total rebalance points: {num_rebalances}")
    print(f"  Active rebalances (with positions): A={active_rebalances_a}, B={active_rebalances_b}")
    print(f"  Avg exposure: A={avg_exp_a:.1%}, B={avg_exp_b:.1%}")
    print(f"  Last active date: A={last_active_a}, B={last_active_b}")
    print(f"  ATR stop triggers (B): {stop_triggers_b}")
    print(f"  Regimes: bull={regime_counts['bull']}, neutral={regime_counts['neutral']}, bear={regime_counts['bear']}")
    print(f"  Trading days simulated: {len(strat_a.nav_history)}")

    # add derived metrics
    metrics_a["Active Rebalances"] = active_rebalances_a
    metrics_b["Active Rebalances"] = active_rebalances_b
    metrics_a["Avg Exposure (%)"] = avg_exp_a * 100
    metrics_b["Avg Exposure (%)"] = avg_exp_b * 100
    metrics_b["ATR Stop Triggers"] = stop_triggers_b
    metrics_a["ATR Stop Triggers"] = 0

    # ── print table ─────────────────────────────────────────────────────────
    metric_order = [
        "Annual Return (%)",
        "Total Return (%)",
        "Max Drawdown (%)",
        "Sharpe Ratio",
        "Win Rate (%)",
        "CVaR 5% (%)",
        "Avg Exposure (%)",
        "Active Rebalances",
        "ATR Stop Triggers",
        "Annual Turnover",
        "Number of Trades",
        "Longest DD Period (days)",
        "Final NAV",
    ]

    header = f"{'Metric':<30} {'Simple CPPI':>18} {'Webapp Advanced':>18}"
    sep = "-" * 70
    lines = [
        "",
        "=" * 70,
        "BACKTEST RESULTS: CPPI vs Webapp Advanced Risk Management",
        f"Reports: ng1.0.2 | Top-{TOP_N} | Hold {FOCUS_DAYS}d | Cost {TRADE_COST*100:.2f}%/leg",
        f"Period: {reports[0][0]} to {reports[-1][0]} | Rebalances: {num_rebalances}",
        "=" * 70,
        "",
        header,
        sep,
    ]

    def fmt_val(m, v):
        if m == "Sharpe Ratio":
            return f"{v:>18.3f}"
        if m in ("Number of Trades", "Longest DD Period (days)", "Active Rebalances", "ATR Stop Triggers"):
            return f"{v:>18.0f}"
        if m == "Final NAV":
            return f"{v:>18,.0f}"
        if m == "Annual Turnover":
            return f"{v:>18.1f}"
        return f"{v:>18.2f}"

    for m in metric_order:
        va, vb = metrics_a.get(m, 0), metrics_b.get(m, 0)
        lines.append(f"{m:<30} {fmt_val(m, va)} {fmt_val(m, vb)}")

    lines.append(sep)

    better_ret = "Simple CPPI" if metrics_a.get("Annual Return (%)", 0) > metrics_b.get("Annual Return (%)", 0) else "Webapp Advanced"
    better_sharpe = "Simple CPPI" if metrics_a.get("Sharpe Ratio", 0) > metrics_b.get("Sharpe Ratio", 0) else "Webapp Advanced"
    better_dd = "Simple CPPI" if metrics_a.get("Max Drawdown (%)", 0) < metrics_b.get("Max Drawdown (%)", 0) else "Webapp Advanced"

    lines += [
        "",
        "WINNER SUMMARY:",
        f"  Best Annual Return:  {better_ret}",
        f"  Best Sharpe Ratio:   {better_sharpe}",
        f"  Lowest Max Drawdown: {better_dd}",
        "",
        "STRATEGY DESCRIPTIONS:",
        "  Strategy A (Simple CPPI):",
        "    - Floor=5%, Multiplier=20",
        "    - Equal weight across Top-10",
        "    - No stops, no regime adaptation",
        "",
        "  Strategy B (Webapp Advanced):",
        "    - Market regime (CSI300 20d): Bull>3%=85%, Neutral=70%, Bear<-3%=40%",
        "    - Circuit breaker: DD>15%->0.4x, DD>10%->0.8x exposure",
        "    - ATR stop-loss: entry - 2.5*ATR_14",
        "    - Risk-parity weighting (inverse 20d volatility)",
        "    - Sector limit: max 25% per industry",
        "    - CPPI base: floor=5%, multiplier=20, capped by regime",
    ]

    print("\n".join(lines))

    # ── save markdown ───────────────────────────────────────────────────────
    def fmt_md(m, v):
        if m == "Sharpe Ratio":
            return f"{v:.3f}"
        if m in ("Number of Trades", "Longest DD Period (days)", "Active Rebalances", "ATR Stop Triggers"):
            return f"{v:.0f}"
        if m == "Final NAV":
            return f"{v:,.0f}"
        if m == "Annual Turnover":
            return f"{v:.1f}"
        return f"{v:.2f}"

    md = [
        "# CPPI vs Webapp Advanced Risk Management Comparison",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Reports**: ng1.0.2 | Top-{TOP_N} | Hold {FOCUS_DAYS}d | Cost {TRADE_COST*100:.2f}%/leg",
        f"**Period**: {reports[0][0]} to {reports[-1][0]} | Rebalances: {num_rebalances}",
        "",
        "## Results",
        "",
        f"| {'Metric':<30} | {'Simple CPPI':>18} | {'Webapp Advanced':>18} |",
        f"|{'-'*32}|{'-'*20}|{'-'*20}|",
    ]
    for m in metric_order:
        va, vb = metrics_a.get(m, 0), metrics_b.get(m, 0)
        md.append(f"| {m:<30} | {fmt_md(m, va):>18} | {fmt_md(m, vb):>18} |")

    md += [
        "",
        "## Winner Summary",
        "",
        f"- **Best Annual Return**: {better_ret}",
        f"- **Best Sharpe Ratio**: {better_sharpe}",
        f"- **Lowest Max Drawdown**: {better_dd}",
        "",
        "## Strategy Descriptions",
        "",
        "### Strategy A: Simple CPPI (EastMoneyTrader)",
        "- Floor=5%, Multiplier=20",
        "- Equal weight across Top-10",
        "- No stops, no regime adaptation",
        "",
        "### Strategy B: Webapp Advanced Risk Management",
        "- **Market Regime** (CSI300 20d return): Bull >3% = 85% cap, Neutral = 70%, Bear <-3% = 40%",
        "- **Circuit Breaker**: DD >15% -> 0.4x exposure, DD >10% -> 0.8x",
        "- **ATR Stop-Loss**: Exit if price < entry - 2.5 * ATR_14",
        "- **Risk-Parity Weighting**: Inverse 20d volatility (lower vol = higher weight)",
        "- **Sector Limit**: Max 25% in any single industry",
        "- **CPPI Base**: Floor=5%, Multiplier=20, capped by regime exposure",
        "",
        "## Diagnostic Info",
        "",
        f"- **Regime distribution**: Bull={regime_counts['bull']}, Neutral={regime_counts['neutral']}, Bear={regime_counts['bear']}",
        f"- **Last active date**: A={last_active_a}, B={last_active_b}",
        f"- **ATR stop triggers**: {stop_triggers_b}",
        "",
        "### CPPI Death Spiral Note",
        "",
        "With floor=5% and multiplier=20, the CPPI formula gives 100% exposure when NAV=peak,",
        "but 0% exposure once NAV drops below 95% of peak. Since peak_nav only increases,",
        "a single bad period (>5% loss) permanently locks exposure at 0%.",
        "Strategy A hit this after the first 10-day period; Strategy B survived longer",
        "because the regime cap limited initial exposure, but eventually also locked out.",
    ]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"\nReport saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
