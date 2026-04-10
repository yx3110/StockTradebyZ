#!/usr/bin/env python3
"""
执行逻辑回测 V2 — 基于 Gap×Score 分析的智能执行策略

策略来自 gap_signal_analysis.py 的发现:
  - 高分股低开是最强买入信号 (gap<-3%, 10d收益6.7%)
  - 平开是最弱信号 (10d仅0.4%)
  - 低分股低开应等日内反弹再卖

新增配置 'smart_exec':
  买入: 根据 score×gap 动态调仓位
    gap<-3%: 加仓150% (抄底信号)
    gap -3~-1%: 正常100%
    gap -1~1%: 减仓70% (平开弱信号)
    gap 1~3%: 正常100%
    gap >5%: 对70-85分减仓50% (高分除外)

  卖出: 低开>3%等日内反弹（用收盘价而非开盘价卖）
"""

import sys
import json
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 复用 execution_backtest 的基础设施
from backtest.execution_backtest import (
    load_reports, load_price_data, ExecutionBacktester,
    format_result, REPORT_DIR,
    COMMISSION, STAMP_TAX, TRANSFER_FEE, MIN_COMMISSION,
    compute_transaction_cost, get_limit_up_threshold, get_trading_dates,
)

DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')


class SmartExecutionBacktester(ExecutionBacktester):
    """基于 Gap×Score 分析的智能执行回测器"""

    def run_smart(self,
                  # 买入仓位调整系数 (相对于默认等权)
                  gap_down_big_mult: float = 1.5,    # gap<-3%: 加仓
                  gap_down_small_mult: float = 1.0,   # gap -3~-1%: 正常
                  gap_flat_mult: float = 0.7,          # gap -1~1%: 减仓
                  gap_up_small_mult: float = 1.0,      # gap 1~3%: 正常
                  gap_up_big_mult: float = 0.5,        # gap >5%: 减仓(非高分)
                  gap_up_big_highscore_mult: float = 1.0,  # gap >5% 高分: 正常
                  # 卖出: 低开时用收盘价（等反弹）
                  sell_gap_down_use_close: float = 0.03,  # 低开>3%用收盘价卖
                  ) -> dict:
        """智能执行模式"""
        initial_capital = 1_000_000.0
        cash = initial_capital
        positions = {}
        nav_history = []
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
                'date': date, 'nav': nav, 'cash': cash,
                'positions_value': positions_value,
                'n_positions': len(positions),
            })

            # 止损
            if self.stop_loss_pct > 0:
                for code in list(positions.keys()):
                    pos = positions[code]
                    if pos['cost'] > 0:
                        ret = (pos['current_price'] - pos['cost']) / pos['cost']
                        if ret <= -self.stop_loss_pct:
                            sell_price = self.get_price(date, code, 'open')
                            if sell_price > 0:
                                proceeds = pos['qty'] * sell_price
                                cost = compute_transaction_cost(proceeds, 'sell')
                                cash += proceeds - cost
                                trade_log.append({
                                    'date': date, 'code': code, 'action': 'stop_loss',
                                    'price': sell_price, 'qty': pos['qty'],
                                })
                                del positions[code]

            # 调仓日
            latest_report_date = None
            for rd in report_dates:
                if rd <= date:
                    latest_report_date = rd
                else:
                    break
            if latest_report_date is None:
                continue

            rebalance_counter += 1
            if rebalance_counter < self.focus_days and i > 0:
                continue
            rebalance_counter = 0

            stocks = self.reports[latest_report_date]
            ranked = sorted(stocks, key=lambda s: s.get('score', 0), reverse=True)
            ranked = [s for s in ranked if s.get('score', 0) >= self.score_floor]
            target_codes = [s['stock_code'] for s in ranked[:self.top_n]]
            score_map = {s['stock_code']: s.get('score', 0) for s in ranked}

            # CPPI
            peak_nav = max(h['nav'] for h in nav_history) if nav_history else initial_capital
            floor_nav = initial_capital * (1 - self.cppi_floor)
            cushion = max(0, nav - floor_nav)
            target_exposure = min(1.0, self.cppi_multiplier * cushion / nav) if nav > 0 else 0
            target_invested = nav * target_exposure

            # 卖出
            held_codes = set(positions.keys())
            sell_codes = held_codes - set(target_codes)

            next_date = self.trading_dates[i + 1] if i + 1 < len(self.trading_dates) else None
            if next_date:
                for code in sell_codes:
                    pos = positions[code]
                    prev_close = self.get_price(date, code, 'close')
                    open_price = self.get_price(next_date, code, 'open')
                    close_price = self.get_price(next_date, code, 'close')

                    if open_price <= 0:
                        continue

                    # 智能卖出: 低开>3% 用收盘价（等日内反弹）
                    gap = (open_price - prev_close) / prev_close if prev_close > 0 else 0
                    if gap < -sell_gap_down_use_close and close_price > 0:
                        sell_price = close_price  # 等反弹
                    else:
                        sell_price = open_price

                    if sell_price <= 0:
                        continue

                    proceeds = pos['qty'] * sell_price
                    cost = compute_transaction_cost(proceeds, 'sell')
                    cash += proceeds - cost
                    trade_log.append({
                        'date': next_date, 'code': code, 'action': 'sell',
                        'price': sell_price, 'qty': pos['qty'],
                    })
                    del positions[code]

                # 买入
                buy_candidates = [c for c in target_codes if c not in positions]
                if buy_candidates and target_invested > positions_value:
                    buy_budget = min(cash, target_invested - positions_value)
                    base_per_stock = buy_budget / len(buy_candidates)

                    for code in buy_candidates:
                        if self.is_limit_up(next_date, code):
                            continue

                        prev_close = self.get_price(date, code, 'close')
                        open_price = self.get_price(next_date, code, 'open')
                        if open_price <= 0 or prev_close <= 0:
                            continue

                        gap = (open_price - prev_close) / prev_close
                        score = score_map.get(code, 0)

                        # 智能仓位: 根据 gap×score 调整
                        if gap < -0.03:
                            mult = gap_down_big_mult
                        elif gap < -0.01:
                            mult = gap_down_small_mult
                        elif gap < 0.01:
                            mult = gap_flat_mult
                        elif gap < 0.03:
                            mult = gap_up_small_mult
                        elif gap > 0.05:
                            mult = gap_up_big_highscore_mult if score >= 85 else gap_up_big_mult
                        else:
                            mult = 1.0

                        allocated = base_per_stock * mult
                        buy_price = open_price
                        qty = int(allocated / buy_price / 100) * 100
                        if qty < 100:
                            continue

                        amount = qty * buy_price
                        cost = compute_transaction_cost(amount, 'buy')
                        if cash >= amount + cost:
                            cash -= amount + cost
                            positions[code] = {
                                'qty': qty, 'cost': buy_price,
                                'buy_date': next_date,
                                'current_price': buy_price,
                                'market_value': amount,
                            }
                            trade_log.append({
                                'date': next_date, 'code': code, 'action': 'buy',
                                'price': buy_price, 'qty': qty,
                                'gap': f"{gap*100:.1f}%", 'mult': f"{mult:.1f}x",
                            })

        return self._compute_metrics(nav_history, trade_log, initial_capital)


def main():
    print(f"\n{'='*70}")
    print(f"  执行逻辑回测 V2 — Smart Execution (2024-01 ~ 2026-04)")
    print(f"{'='*70}\n")

    reports = load_reports(REPORT_DIR, '2024-01-01', '2026-12-31')
    price_df = load_price_data('2024-01-01', '2026-12-31')
    print(f"  报告: {len(reports)} 天, 价格: {len(price_df)} 条\n")

    bt = SmartExecutionBacktester(
        reports, price_df,
        top_n=10, focus_days=15, score_floor=30,
        stop_loss_pct=0.06, cppi_floor=0.08, cppi_multiplier=20,
    )

    configs = {
        'A. Baseline (V1最优)': lambda: bt.run(exec_mode='open_exec'),
        'B. Smart V2 (默认)': lambda: bt.run_smart(),
        'C. Smart V2 (激进抄底)': lambda: bt.run_smart(
            gap_down_big_mult=2.0, gap_flat_mult=0.5),
        'D. Smart V2 (保守)': lambda: bt.run_smart(
            gap_down_big_mult=1.2, gap_flat_mult=0.8, gap_up_big_mult=0.3),
        'E. Smart V2 (卖出等反弹5%)': lambda: bt.run_smart(
            sell_gap_down_use_close=0.05),
    }

    results = {}
    for label, run_fn in configs.items():
        print(f"运行: {label}...")
        result = run_fn()
        results[label] = result
        print(format_result(label, result))

    print(f"\n{'='*70}")
    print(f"  对比总结")
    print(f"{'='*70}")
    print(f"  {'配置':<28} {'年化':>8} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8} {'换手':>8}")
    print(f"  {'─'*72}")
    for label, result in results.items():
        print(f"  {label:<28} {result['annual_return']*100:>7.1f}% {result['max_drawdown']*100:>7.1f}% "
              f"{result['sharpe']:>8.2f} {result['calmar']:>8.2f} {result['turnover']:>7.1f}x")


if __name__ == '__main__':
    main()
