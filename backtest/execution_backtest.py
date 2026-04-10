#!/usr/bin/env python3
"""
执行逻辑回测 — 量化实际交易中的 gap 过滤、开盘价执行、减仓策略对收益的影响

对比配置:
  baseline:    NG1.0.5 标准回测（买入开盘价，卖出收盘价，无gap过滤）
  open_exec:   买卖均用开盘价
  gap_filter:  叠加高开/低开过滤（参数扫描）
  reduce_50:   减仓50%策略（低分持仓不全卖）

所有配置用同一份 NG1.0.1 报告，北极星 V5.2 评分对比。

用法:
    cd /Users/yangxu/StockTradebyZ
    python3 backtest/execution_backtest.py
    python3 backtest/execution_backtest.py --gap-scan       # 扫描最优gap阈值
    python3 backtest/execution_backtest.py --reduce-scan    # 扫描减仓策略
"""

import sys
import json
import sqlite3
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')
REPORT_DIR = PROJECT_ROOT / 'reports' / 'daily_selection_ng101'

# 交易成本
COMMISSION = 0.0003      # 佣金 0.03%
STAMP_TAX = 0.001        # 印花税 0.1% (卖出)
TRANSFER_FEE = 0.00002   # 过户费 0.002%
MIN_COMMISSION = 5.0      # 最低佣金 5元


def load_reports(report_dir: Path, start_date: str = '2024-01-01',
                 end_date: str = '2026-12-31') -> dict:
    """加载报告: {date_str: [{stock_code, score, ...}]}"""
    reports = {}
    for f in sorted(report_dir.glob('analysis_data_*.json')):
        date_str = f.stem.replace('analysis_data_', '')
        # 转为 YYYY-MM-DD 格式比较
        if len(date_str) == 8:
            date_cmp = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        else:
            date_cmp = date_str
        if date_cmp < start_date or date_cmp > end_date:
            continue
        with open(f, 'r') as fh:
            data = json.load(fh)
        stocks = data.get('all_stocks_with_scores', data.get('top_recommendations', []))
        if stocks:
            reports[date_cmp] = stocks
    return reports


def load_price_data(start_date: str, end_date: str) -> pd.DataFrame:
    """加载OHLC + 涨跌幅数据"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT s.code, dq.trade_date, dq.open, dq.close, dq.high, dq.low,
               dq.price_change_pct, dq.is_limit_up, dq.is_limit_down
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE dq.trade_date >= ? AND dq.trade_date <= ?
          AND dq.open > 0 AND dq.close > 0
          AND s.type = 'A股'
    """, conn, params=[start_date, end_date])
    conn.close()
    return df


def get_trading_dates(price_df: pd.DataFrame) -> list:
    """获取排序后的交易日列表"""
    return sorted(price_df['trade_date'].unique())


def get_limit_up_threshold(code: str) -> float:
    """涨停阈值: ST 5%, 创业板/科创板 20%, 其他 10%"""
    if code.startswith('3') or code.startswith('68'):
        return 19.5
    return 9.5


def compute_transaction_cost(amount: float, direction: str) -> float:
    """计算交易成本"""
    cost = max(amount * COMMISSION, MIN_COMMISSION)  # 佣金
    cost += amount * TRANSFER_FEE                     # 过户费
    if direction == 'sell':
        cost += amount * STAMP_TAX                    # 印花税(卖出)
    return cost


class ExecutionBacktester:
    """执行逻辑回测器"""

    def __init__(self, reports: dict, price_df: pd.DataFrame,
                 top_n: int = 10, focus_days: int = 15,
                 score_floor: float = 30.0,
                 stop_loss_pct: float = 0.06,
                 cppi_floor: float = 0.08, cppi_multiplier: float = 20.0):
        self.reports = reports
        self.price_df = price_df
        self.trading_dates = get_trading_dates(price_df)
        self.top_n = top_n
        self.focus_days = focus_days
        self.score_floor = score_floor
        self.stop_loss_pct = stop_loss_pct
        self.cppi_floor = cppi_floor
        self.cppi_multiplier = cppi_multiplier

        # 构建价格查找表: {(date, code): {open, close, pct, ...}}
        self._prices = {}
        for _, row in price_df.iterrows():
            key = (row['trade_date'], row['code'])
            self._prices[key] = {
                'open': row['open'],
                'close': row['close'],
                'high': row['high'],
                'low': row['low'],
                'pct': row['price_change_pct'] if pd.notna(row['price_change_pct']) else 0,
            }

    def get_price(self, date: str, code: str, field: str = 'close') -> float:
        key = (date, code)
        if key in self._prices:
            return self._prices[key].get(field, 0)
        return 0

    def get_prev_close(self, date: str, code: str) -> float:
        """获取前一交易日收盘价"""
        idx = self.trading_dates.index(date) if date in self.trading_dates else -1
        if idx <= 0:
            return 0
        prev_date = self.trading_dates[idx - 1]
        return self.get_price(prev_date, code, 'close')

    def is_limit_up(self, date: str, code: str) -> bool:
        pct = self.get_price(date, code, 'pct')
        return pct >= get_limit_up_threshold(code)

    def run(self, exec_mode: str = 'baseline',
            buy_gap_skip: float = 0.0,
            buy_gap_reduce: float = 0.0,
            sell_gap_skip: float = 0.0,
            reduce_pct: float = 0.0) -> dict:
        """
        运行回测

        exec_mode:
          'baseline' — 买入开盘价, 卖出收盘价 (现有回测模型)
          'open_exec' — 买卖均用开盘价
          'gap_filter' — open_exec + gap过滤
          'reduce_mode' — gap_filter + 部分减仓

        buy_gap_skip: 高开超过此比例跳过买入 (如 0.03 = 3%)
        buy_gap_reduce: 高开超过此比例减半买入
        sell_gap_skip: 低开超过此比例暂缓卖出
        reduce_pct: >0时低分持仓减仓而非全卖 (如 0.5 = 减50%)
        """
        initial_capital = 1_000_000.0
        cash = initial_capital
        positions = {}  # {code: {qty, cost, buy_date, score}}
        nav_history = []  # [{date, nav, cash, positions_value}]
        trade_log = []

        report_dates = sorted(self.reports.keys())
        rebalance_counter = 0

        for i, date in enumerate(self.trading_dates):
            # 更新持仓市值
            positions_value = 0
            for code, pos in list(positions.items()):
                price = self.get_price(date, code, 'close')
                if price > 0:
                    pos['current_price'] = price
                    pos['market_value'] = pos['qty'] * price
                    positions_value += pos['market_value']
                else:
                    positions_value += pos.get('market_value', 0)

            nav = cash + positions_value
            nav_history.append({
                'date': date,
                'nav': nav,
                'cash': cash,
                'positions_value': positions_value,
                'n_positions': len(positions),
            })

            # 止损检查 (每天)
            if self.stop_loss_pct > 0:
                for code in list(positions.keys()):
                    pos = positions[code]
                    if pos['cost'] > 0:
                        ret = (pos['current_price'] - pos['cost']) / pos['cost']
                        if ret <= -self.stop_loss_pct:
                            sell_price = self._get_exec_price(date, code, 'sell', exec_mode)
                            if sell_price > 0:
                                proceeds = pos['qty'] * sell_price
                                cost = compute_transaction_cost(proceeds, 'sell')
                                cash += proceeds - cost
                                trade_log.append({
                                    'date': date, 'code': code, 'action': 'stop_loss',
                                    'price': sell_price, 'qty': pos['qty'],
                                    'reason': f'止损 {ret*100:.1f}%',
                                })
                                del positions[code]

            # 调仓日检查: 找到最近的报告日 <= 今天
            latest_report_date = None
            for rd in report_dates:
                if rd <= date:
                    latest_report_date = rd
                else:
                    break

            if latest_report_date is None:
                continue

            # 每 focus_days 天调仓一次
            rebalance_counter += 1
            if rebalance_counter < self.focus_days and i > 0:
                continue
            rebalance_counter = 0

            # 获取推荐列表
            stocks = self.reports[latest_report_date]
            ranked = sorted(stocks, key=lambda s: s.get('score', 0), reverse=True)
            # 过滤 score_floor
            ranked = [s for s in ranked if s.get('score', 0) >= self.score_floor]
            target_codes = [s['stock_code'] for s in ranked[:self.top_n]]

            # CPPI 敞口控制
            peak_nav = max(h['nav'] for h in nav_history) if nav_history else initial_capital
            floor_nav = initial_capital * (1 - self.cppi_floor)
            cushion = max(0, nav - floor_nav)
            target_exposure = min(1.0, self.cppi_multiplier * cushion / nav) if nav > 0 else 0
            target_invested = nav * target_exposure

            # 决定卖出
            held_codes = set(positions.keys())
            sell_codes = held_codes - set(target_codes)

            # 执行卖出
            next_date = self.trading_dates[i + 1] if i + 1 < len(self.trading_dates) else None
            if next_date:
                for code in sell_codes:
                    pos = positions[code]
                    prev_close = self.get_price(date, code, 'close')
                    open_price = self.get_price(next_date, code, 'open')

                    if open_price <= 0:
                        continue

                    # Gap 过滤 (卖出)
                    if sell_gap_skip > 0 and prev_close > 0:
                        gap = (open_price - prev_close) / prev_close
                        if gap < -sell_gap_skip:
                            continue  # 低开太多，暂缓

                    sell_price = self._get_exec_price(next_date, code, 'sell', exec_mode)
                    if sell_price <= 0:
                        continue

                    # 减仓模式: 只卖一部分
                    sell_qty = pos['qty']
                    if reduce_pct > 0 and reduce_pct < 1.0:
                        # 根据评分动态减仓: 评分越低减越多
                        stock_score = 0
                        for s in ranked:
                            if s['stock_code'] == code:
                                stock_score = s.get('score', 0)
                                break
                        if stock_score > 20:  # 不是最差的，只减一部分
                            sell_qty = int(pos['qty'] * reduce_pct)

                    if sell_qty <= 0:
                        continue

                    proceeds = sell_qty * sell_price
                    cost = compute_transaction_cost(proceeds, 'sell')
                    cash += proceeds - cost

                    if sell_qty >= pos['qty']:
                        del positions[code]
                    else:
                        positions[code]['qty'] -= sell_qty
                        positions[code]['market_value'] = positions[code]['qty'] * pos['current_price']

                    trade_log.append({
                        'date': next_date, 'code': code, 'action': 'sell',
                        'price': sell_price, 'qty': sell_qty,
                    })

                # 执行买入
                buy_candidates = [c for c in target_codes if c not in positions]
                if buy_candidates and target_invested > positions_value:
                    buy_budget = min(cash, target_invested - positions_value)
                    per_stock = buy_budget / len(buy_candidates) if buy_candidates else 0

                    for code in buy_candidates:
                        if per_stock < 1000:
                            break

                        if self.is_limit_up(next_date, code):
                            continue  # 涨停买不进

                        prev_close = self.get_price(date, code, 'close')
                        open_price = self.get_price(next_date, code, 'open')

                        if open_price <= 0:
                            continue

                        # Gap 过滤 (买入)
                        if buy_gap_skip > 0 and prev_close > 0:
                            gap = (open_price - prev_close) / prev_close
                            if gap > buy_gap_skip:
                                continue  # 高开太多，跳过
                            if buy_gap_reduce > 0 and gap > buy_gap_reduce:
                                per_stock *= 0.5  # 高开，减半

                        buy_price = self._get_exec_price(next_date, code, 'buy', exec_mode)
                        if buy_price <= 0:
                            continue

                        qty = int(per_stock / buy_price / 100) * 100
                        if qty < 100:
                            continue

                        amount = qty * buy_price
                        cost = compute_transaction_cost(amount, 'buy')

                        if cash >= amount + cost:
                            cash -= amount + cost
                            positions[code] = {
                                'qty': qty,
                                'cost': buy_price,
                                'buy_date': next_date,
                                'current_price': buy_price,
                                'market_value': amount,
                            }
                            trade_log.append({
                                'date': next_date, 'code': code, 'action': 'buy',
                                'price': buy_price, 'qty': qty,
                            })

        # 计算指标
        return self._compute_metrics(nav_history, trade_log, initial_capital)

    def _get_exec_price(self, date: str, code: str, direction: str,
                         exec_mode: str) -> float:
        """根据执行模式获取执行价格"""
        if exec_mode == 'baseline':
            if direction == 'buy':
                return self.get_price(date, code, 'open')
            else:
                return self.get_price(date, code, 'close')
        else:
            # open_exec / gap_filter / reduce_mode: 买卖均用开盘价
            return self.get_price(date, code, 'open')

    def _compute_metrics(self, nav_history: list, trade_log: list,
                          initial_capital: float) -> dict:
        """计算回测指标"""
        if not nav_history:
            return {}

        navs = [h['nav'] for h in nav_history]
        dates = [h['date'] for h in nav_history]
        returns = pd.Series(navs).pct_change().dropna()

        # 基础指标
        total_return = (navs[-1] / initial_capital) - 1
        n_years = len(navs) / 242  # 约242个交易日/年
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # 最大回撤
        peak = pd.Series(navs).expanding().max()
        drawdown = (pd.Series(navs) - peak) / peak
        max_drawdown = drawdown.min()

        # Sharpe (无风险利率 3%)
        rf_daily = 0.03 / 242
        excess_returns = returns - rf_daily
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(242) if len(excess_returns) > 10 else 0

        # Sortino
        downside = excess_returns[excess_returns < 0]
        sortino = excess_returns.mean() / downside.std() * np.sqrt(242) if len(downside) > 5 else 0

        # Calmar
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # 月度胜率
        nav_series = pd.Series(navs, index=pd.to_datetime(dates))
        monthly = nav_series.resample('ME').last().pct_change().dropna()
        monthly_win_rate = (monthly > 0).mean() if len(monthly) > 0 else 0

        # 交易统计
        n_trades = len(trade_log)
        n_buys = sum(1 for t in trade_log if t['action'] == 'buy')
        n_sells = sum(1 for t in trade_log if t['action'] in ('sell', 'stop_loss'))
        n_stop_loss = sum(1 for t in trade_log if t['action'] == 'stop_loss')

        # 换手率 (年化)
        total_traded = sum(t['price'] * t['qty'] for t in trade_log)
        avg_nav = np.mean(navs)
        turnover = total_traded / avg_nav / n_years if avg_nav > 0 and n_years > 0 else 0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe': sharpe,
            'sortino': sortino,
            'calmar': calmar,
            'monthly_win_rate': monthly_win_rate,
            'n_trades': n_trades,
            'n_buys': n_buys,
            'n_sells': n_sells,
            'n_stop_loss': n_stop_loss,
            'turnover': turnover,
            'final_nav': navs[-1],
            'n_days': len(navs),
            'nav_history': nav_history,
            'trade_log': trade_log,
        }


def format_result(label: str, result: dict) -> str:
    """格式化单个结果"""
    lines = [
        f"  {label}:",
        f"    年化收益: {result['annual_return']*100:>8.1f}%",
        f"    最大回撤: {result['max_drawdown']*100:>8.1f}%",
        f"    Sharpe:   {result['sharpe']:>8.2f}",
        f"    Sortino:  {result['sortino']:>8.2f}",
        f"    Calmar:   {result['calmar']:>8.2f}",
        f"    月度胜率: {result['monthly_win_rate']*100:>8.1f}%",
        f"    换手率:   {result['turnover']:>8.1f}x/年",
        f"    交易笔数: {result['n_trades']:>8d} (买{result['n_buys']} 卖{result['n_sells']} 止损{result['n_stop_loss']})",
    ]
    return '\n'.join(lines)


def run_comparison():
    """运行基线对比"""
    print(f"\n{'='*70}")
    print(f"  执行逻辑回测 — NG1.0.5 (2024-01 ~ 2026-04)")
    print(f"{'='*70}\n")

    print("加载报告和价格数据...")
    reports = load_reports(REPORT_DIR, '2024-01-01', '2026-12-31')
    print(f"  报告: {len(reports)} 个交易日")

    price_df = load_price_data('2024-01-01', '2026-12-31')
    print(f"  价格: {len(price_df)} 条记录")

    bt = ExecutionBacktester(
        reports, price_df,
        top_n=10, focus_days=15, score_floor=30,
        stop_loss_pct=0.06, cppi_floor=0.08, cppi_multiplier=20,
    )

    configs = {
        'A. Baseline (买开卖收)': dict(exec_mode='baseline'),
        'B. Open Exec (买卖均开盘)': dict(exec_mode='open_exec'),
        'C. Gap Filter 3%/1%': dict(exec_mode='gap_filter', buy_gap_skip=0.03, buy_gap_reduce=0.01, sell_gap_skip=0.03),
        'D. Gap Filter 5%/2%': dict(exec_mode='gap_filter', buy_gap_skip=0.05, buy_gap_reduce=0.02, sell_gap_skip=0.05),
        'E. Gap 3%/1% + 减仓50%': dict(exec_mode='reduce_mode', buy_gap_skip=0.03, buy_gap_reduce=0.01, sell_gap_skip=0.03, reduce_pct=0.5),
    }

    results = {}
    for label, params in configs.items():
        print(f"\n运行: {label}...")
        result = bt.run(**params)
        results[label] = result
        print(format_result(label, result))

    # 对比表格
    print(f"\n{'='*70}")
    print(f"  对比总结")
    print(f"{'='*70}")
    print(f"  {'配置':<28} {'年化':>8} {'MaxDD':>8} {'Sharpe':>8} {'换手':>8}")
    print(f"  {'─'*64}")
    for label, result in results.items():
        print(f"  {label:<28} {result['annual_return']*100:>7.1f}% {result['max_drawdown']*100:>7.1f}% "
              f"{result['sharpe']:>8.2f} {result['turnover']:>7.1f}x")

    return results


def run_gap_scan():
    """扫描最优 gap 过滤阈值"""
    print(f"\n{'='*70}")
    print(f"  Gap 阈值扫描 (2024-01 ~ 2026-04)")
    print(f"{'='*70}\n")

    reports = load_reports(REPORT_DIR, '2024-01-01', '2026-12-31')
    price_df = load_price_data('2024-01-01', '2026-12-31')

    bt = ExecutionBacktester(
        reports, price_df,
        top_n=10, focus_days=15, score_floor=30,
        stop_loss_pct=0.06, cppi_floor=0.08, cppi_multiplier=20,
    )

    print(f"  {'买入跳过':>10} {'买入减半':>10} {'卖出暂缓':>10} {'年化':>8} {'MaxDD':>8} {'Sharpe':>8} {'换手':>8}")
    print(f"  {'─'*74}")

    best_sharpe = -999
    best_config = {}

    for buy_skip in [0, 0.02, 0.03, 0.04, 0.05]:
        for buy_reduce in [0, 0.01, 0.02]:
            if buy_reduce >= buy_skip and buy_skip > 0:
                continue
            for sell_skip in [0, 0.02, 0.03, 0.05]:
                result = bt.run(
                    exec_mode='gap_filter',
                    buy_gap_skip=buy_skip,
                    buy_gap_reduce=buy_reduce,
                    sell_gap_skip=sell_skip,
                )
                label = f"{buy_skip*100:.0f}%/{buy_reduce*100:.0f}%/{sell_skip*100:.0f}%"
                print(f"  {buy_skip*100:>9.0f}% {buy_reduce*100:>9.0f}% {sell_skip*100:>9.0f}% "
                      f"{result['annual_return']*100:>7.1f}% {result['max_drawdown']*100:>7.1f}% "
                      f"{result['sharpe']:>8.2f} {result['turnover']:>7.1f}x")

                if result['sharpe'] > best_sharpe:
                    best_sharpe = result['sharpe']
                    best_config = {
                        'buy_gap_skip': buy_skip,
                        'buy_gap_reduce': buy_reduce,
                        'sell_gap_skip': sell_skip,
                        'result': result,
                    }

    print(f"\n  最优配置: 买入跳过>{best_config['buy_gap_skip']*100:.0f}% "
          f"买入减半>{best_config['buy_gap_reduce']*100:.0f}% "
          f"卖出暂缓>{best_config['sell_gap_skip']*100:.0f}%")
    print(f"  Sharpe={best_sharpe:.2f}")

    return best_config


def run_reduce_scan():
    """扫描减仓策略"""
    print(f"\n{'='*70}")
    print(f"  减仓策略扫描 (2024-01 ~ 2026-04)")
    print(f"{'='*70}\n")

    reports = load_reports(REPORT_DIR, '2024-01-01', '2026-12-31')
    price_df = load_price_data('2024-01-01', '2026-12-31')

    bt = ExecutionBacktester(
        reports, price_df,
        top_n=10, focus_days=15, score_floor=30,
        stop_loss_pct=0.06, cppi_floor=0.08, cppi_multiplier=20,
    )

    print(f"  {'减仓比例':>10} {'年化':>8} {'MaxDD':>8} {'Sharpe':>8} {'换手':>8}")
    print(f"  {'─'*50}")

    for reduce_pct in [0, 0.3, 0.5, 0.7, 1.0]:
        label = f"{reduce_pct*100:.0f}%" if reduce_pct > 0 else "全卖"
        result = bt.run(
            exec_mode='gap_filter',
            buy_gap_skip=0.03, buy_gap_reduce=0.01, sell_gap_skip=0.03,
            reduce_pct=reduce_pct,
        )
        print(f"  {label:>10} {result['annual_return']*100:>7.1f}% {result['max_drawdown']*100:>7.1f}% "
              f"{result['sharpe']:>8.2f} {result['turnover']:>7.1f}x")


def main():
    parser = argparse.ArgumentParser(description='执行逻辑回测')
    parser.add_argument('--gap-scan', action='store_true', help='扫描最优gap阈值')
    parser.add_argument('--reduce-scan', action='store_true', help='扫描减仓策略')
    args = parser.parse_args()

    if args.gap_scan:
        run_gap_scan()
    elif args.reduce_scan:
        run_reduce_scan()
    else:
        run_comparison()


if __name__ == '__main__':
    main()
