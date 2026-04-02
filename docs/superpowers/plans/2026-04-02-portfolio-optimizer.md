# Portfolio Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fixed-percentage price/position logic with adaptive ATR-based prices, dynamic stop-loss/take-profit, risk-budget position sizing, and grid-search parameter calibration — to achieve excess returns over CSI300 and CSI2000.

**Architecture:** New standalone `portfolio_optimizer.py` module with parameterized price/position calculations. New `backtest/dynamic_sl_tp_backtest.py` extending existing `backtest_stop_target_direct.py` with trailing stop and parameter sweep. All hyperparameters loaded from `optimizer_params.json`. Integration via `--optimizer v2` flag in `tomorrow_stock_selector.py`.

**Tech Stack:** Python 3, pandas, numpy, sqlite3, existing backtest infrastructure (`backtest_stop_target_direct.py`, `backtest_report_based.py`)

**Spec:** `docs/superpowers/specs/2026-04-02-portfolio-optimizer-design.md`

---

### Task 1: PortfolioOptimizer core module — price calculations

**Files:**
- Create: `portfolio_optimizer.py`
- Create: `optimizer_params.json` (default initial params)

- [ ] **Step 1: Create default parameters file**

```json
// optimizer_params.json
{
  "_comment": "Default initial params — to be calibrated via grid search",
  "entry": {
    "atr_discount_mult": 0.5,
    "support_discount_mult": 0.3,
    "ml_bullish_threshold": 0.005,
    "ml_bullish_mult": 0.5,
    "ml_bearish_mult": 1.5,
    "max_discount": 0.03
  },
  "stop": {
    "atr_multiplier": 2.0,
    "env_mult_bullish": 0.85,
    "env_mult_bearish": 1.2,
    "min_stop_pct": 0.03,
    "max_stop_main": 0.10,
    "max_stop_wide": 0.15
  },
  "target": {
    "min_rr_ratio": 2.0,
    "target_clip_min": 0.03,
    "target_clip_max": 0.15
  },
  "filter": {
    "composite_cutoff": 0,
    "min_n": 3,
    "max_n_bull": 20,
    "max_n_bear": 5
  },
  "trailing": {
    "trailing_trigger_pct": 0.6,
    "trailing_fallback_pct": 0.4
  },
  "hold": {
    "max_hold_days": 15
  }
}
```

- [ ] **Step 2: Create portfolio_optimizer.py with support/resistance helpers**

```python
#!/usr/bin/env python3
"""
Portfolio Optimizer: 自适应价格锚定 + 动态止损止盈 + 风险预算仓位

所有超参数从 optimizer_params.json 加载, 通过回测网格搜索校准.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any

PROJECT_ROOT = Path(__file__).parent
DEFAULT_PARAMS_PATH = PROJECT_ROOT / 'optimizer_params.json'


def load_params(path: Optional[str] = None) -> dict:
    """加载超参数配置"""
    p = Path(path) if path else DEFAULT_PARAMS_PATH
    if p.exists():
        with open(p) as f:
            return json.load(f)
    raise FileNotFoundError(f"参数文件不存在: {p}")


def compute_support(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
    """计算支撑位: 近20日最低价、MA20、MA60中离收盘价最近且低于收盘价的

    Args:
        highs: 近60+日最高价数组
        lows: 近60+日最低价数组
        closes: 近60+日收盘价数组
    """
    close = closes[-1]
    rolling_low_20 = np.min(lows[-20:]) if len(lows) >= 20 else np.min(lows)
    ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else np.mean(closes)
    ma60 = np.mean(closes[-60:]) if len(closes) >= 60 else np.mean(closes)

    candidates = [c for c in [rolling_low_20, ma20, ma60] if c < close * 0.995]
    return max(candidates) if candidates else close * 0.97


def compute_resistance(highs: np.ndarray, closes: np.ndarray) -> float:
    """计算阻力位: 近20日最高价、MA20、MA60中离收盘价最近且高于收盘价的"""
    close = closes[-1]
    rolling_high_20 = np.max(highs[-20:]) if len(highs) >= 20 else np.max(highs)
    ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else np.mean(closes)
    ma60 = np.mean(closes[-60:]) if len(closes) >= 60 else np.mean(closes)

    candidates = [c for c in [rolling_high_20, ma20, ma60] if c > close * 1.005]
    return min(candidates) if candidates else close * 1.08


def compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                period: int = 20) -> float:
    """计算ATR(period)"""
    if len(closes) < period + 1:
        return 0.0
    prev_close = np.roll(closes, 1)[1:]  # shift(1)
    h = highs[1:]
    l = lows[1:]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_close), np.abs(l - prev_close)))
    return float(np.mean(tr[-period:])) if len(tr) >= period else float(np.mean(tr))
```

- [ ] **Step 3: Add entry/stop/target price calculation functions**

Append to `portfolio_optimizer.py`:

```python
def compute_entry_price(close: float, atr_20d: float, support: float,
                        pred_10d: float, params: dict) -> float:
    """自适应买入价: ATR折扣 + 支撑距离 + ML信号调节"""
    p = params['entry']
    atr_ratio = atr_20d / close if close > 0 else 0

    atr_discount = atr_ratio * p['atr_discount_mult']
    support_gap = max(0, (close - support) / close) if close > 0 else 0
    support_discount = support_gap * p['support_discount_mult']

    threshold = p['ml_bullish_threshold']
    if pred_10d > threshold:
        ml_mult = p['ml_bullish_mult']
    elif pred_10d < -threshold:
        ml_mult = p['ml_bearish_mult']
    else:
        ml_mult = 1.0

    adaptive_discount = (atr_discount + support_discount) * ml_mult
    adaptive_discount = np.clip(adaptive_discount, 0.0, p['max_discount'])

    return round(close * (1 - adaptive_discount), 2)


def compute_stop_price(buy_price: float, close: float, atr_20d: float,
                       support: float, env_score: float,
                       is_wide_limit: bool, params: dict) -> float:
    """自适应止损: N倍ATR + 市场环境调节 + 支撑位保护"""
    p = params['stop']
    atr_pct = atr_20d / close if close > 0 else 0.05

    base_stop_pct = atr_pct * p['atr_multiplier']

    if env_score >= 60:
        env_mult = p['env_mult_bullish']
    elif env_score >= 40:
        env_mult = 1.0
    else:
        env_mult = p['env_mult_bearish']

    min_stop = p['min_stop_pct']
    max_stop = p['max_stop_wide'] if is_wide_limit else p['max_stop_main']

    stop_pct = np.clip(base_stop_pct * env_mult, min_stop, max_stop)
    stop_price = buy_price * (1 - stop_pct)

    # 支撑位保护
    if support > stop_price and support < buy_price * 0.99:
        stop_price = support * 0.995

    return round(stop_price, 2)


def compute_target_price(buy_price: float, stop_price: float, close: float,
                         resistance: float, pred_10d: float, params: dict) -> float:
    """自适应目标价: 技术阻力位 + ML预测 + 风险收益比约束"""
    p = params['target']
    risk = buy_price - stop_price
    if risk <= 0:
        return round(buy_price * 1.05, 2)

    tech_target = resistance * 0.98 if resistance > close * 1.01 else None
    ml_target = close * (1 + pred_10d) if pred_10d > 0 else None
    min_rr_target = buy_price + risk * p['min_rr_ratio']

    candidates = [c for c in [tech_target, ml_target, min_rr_target] if c is not None]
    if len(candidates) >= 2:
        candidates.sort()
        target = candidates[len(candidates) // 2]
    elif candidates:
        target = candidates[0]
    else:
        target = min_rr_target

    target = np.clip(target,
                     buy_price * (1 + p['target_clip_min']),
                     buy_price * (1 + p['target_clip_max']))

    return round(float(target), 2)
```

- [ ] **Step 4: Add signal filter and risk budget allocator**

Append to `portfolio_optimizer.py`:

```python
def filter_by_signal_strength(stocks: List[dict], env_score: float,
                              params: dict) -> List[dict]:
    """动态信号强度cutoff: composite > cutoff 的入选, 数量受约束"""
    p = params['filter']
    cutoff = p['composite_cutoff']
    above = [s for s in stocks if s.get('composite', 0) > cutoff]

    if env_score < 30:
        max_n = p['max_n_bear']
    elif env_score < 50:
        max_n = (p['max_n_bear'] + p['max_n_bull']) // 2
    else:
        max_n = p['max_n_bull']

    min_n = p['min_n']
    n = int(np.clip(len(above), min_n, max_n))

    # 按composite降序排列, 取前n只
    sorted_stocks = sorted(stocks, key=lambda s: s.get('composite', 0), reverse=True)
    return sorted_stocks[:n]


def allocate_positions(stocks: List[dict], env_score: float) -> List[dict]:
    """风险预算仓位: 信号强度 × 波动率倒数, 市场环境总仓位约束"""
    if not stocks:
        return stocks

    # 总仓位
    if env_score >= 80:
        total_exposure = 0.90
    elif env_score >= 60:
        total_exposure = 0.65
    elif env_score >= 40:
        total_exposure = 0.40
    elif env_score >= 20:
        total_exposure = 0.20
    else:
        total_exposure = 0.05

    n = len(stocks)
    # composite_rank_pct: 排名百分位 (1=最强)
    for i, s in enumerate(stocks):
        s['composite_rank_pct'] = 1.0 - i / max(n, 1)

    # 原始权重 = 信号强度 × 波动率倒数
    for s in stocks:
        atr_pct = max(s.get('atr_pct', 0.03), 0.01)
        signal = s['composite_rank_pct']
        s['raw_weight'] = signal * (1.0 / atr_pct)

    total_raw = sum(s['raw_weight'] for s in stocks)
    if total_raw <= 0:
        for s in stocks:
            s['position_pct'] = round(total_exposure * 100 / n, 1)
        return stocks

    # 归一化 → 个股仓位
    for s in stocks:
        norm_w = s['raw_weight'] / total_raw
        raw_pct = norm_w * total_exposure * 100
        s['position_pct'] = round(np.clip(raw_pct, 1.0, 15.0), 1)

    # 等风险贡献校验
    risk_contributions = [s['position_pct'] * s.get('atr_pct', 0.03) for s in stocks]
    avg_rc = np.mean(risk_contributions)
    for i, s in enumerate(stocks):
        if risk_contributions[i] > avg_rc * 2.0 and avg_rc > 0:
            s['position_pct'] = round(s['position_pct'] * (avg_rc * 2.0 / risk_contributions[i]), 1)

    # 总仓位约束
    actual = sum(s['position_pct'] for s in stocks)
    if actual > total_exposure * 100:
        scale = total_exposure * 100 / actual
        for s in stocks:
            s['position_pct'] = round(s['position_pct'] * scale, 1)

    return stocks
```

- [ ] **Step 5: Add PortfolioOptimizer facade class**

Append to `portfolio_optimizer.py`:

```python
class PortfolioOptimizer:
    """统一接口: 价格计算 + 信号筛选 + 仓位分配"""

    def __init__(self, params_path: Optional[str] = None, params: Optional[dict] = None):
        if params:
            self.params = params
        else:
            self.params = load_params(params_path)

    def compute_prices(self, stock_info: dict, highs: np.ndarray,
                       lows: np.ndarray, closes: np.ndarray,
                       env_score: float) -> dict:
        """为单只股票计算买入价/止损/目标价

        Args:
            stock_info: 必须含 stock_code, close_price, pred_10d
            highs/lows/closes: 近60+日K线数据
            env_score: 市场环境评分 0-100
        """
        close = stock_info.get('close_price', 0)
        if close <= 0:
            return stock_info

        pred_10d = stock_info.get('pred_10d', 0) or 0
        stock_code = stock_info.get('stock_code', '')
        is_wide_limit = stock_code.startswith('30') or stock_code.startswith('688')

        atr = compute_atr(highs, lows, closes, period=20)
        support = compute_support(highs, lows, closes)
        resistance = compute_resistance(highs, closes)

        buy_price = compute_entry_price(close, atr, support, pred_10d, self.params)
        stop_price = compute_stop_price(buy_price, close, atr, support,
                                        env_score, is_wide_limit, self.params)
        target_price = compute_target_price(buy_price, stop_price, close,
                                            resistance, pred_10d, self.params)

        stock_info['suggested_buy_price'] = buy_price
        stock_info['stop_loss_price'] = stop_price
        stock_info['take_profit_price'] = target_price
        stock_info['atr_20'] = round(atr, 4)
        stock_info['atr_pct'] = round(atr / close, 4) if close > 0 else 0
        stock_info['support_level'] = round(support, 2)
        stock_info['resistance_level'] = round(resistance, 2)

        risk = buy_price - stop_price
        reward = target_price - buy_price
        stock_info['risk_pct'] = round((risk / buy_price) * 100, 2) if buy_price > 0 else 0
        stock_info['reward_pct'] = round((reward / buy_price) * 100, 2) if buy_price > 0 else 0
        stock_info['risk_reward_ratio'] = round(reward / risk, 2) if risk > 0 else 0

        return stock_info

    def filter_and_allocate(self, stocks: List[dict], env_score: float) -> List[dict]:
        """信号筛选 + 仓位分配"""
        selected = filter_by_signal_strength(stocks, env_score, self.params)
        selected = allocate_positions(selected, env_score)
        return selected
```

- [ ] **Step 6: Verify module imports correctly**

Run: `python3 -c "from portfolio_optimizer import PortfolioOptimizer, load_params; p = load_params(); print(f'Loaded {len(p)} param groups'); opt = PortfolioOptimizer(); print('OK')"`
Expected: `Loaded 6 param groups` then `OK`

- [ ] **Step 7: Commit**

```bash
git add portfolio_optimizer.py optimizer_params.json
git commit -m "feat: PortfolioOptimizer核心模块 — 自适应价格+风险预算仓位"
```

---

### Task 2: Dynamic SL/TP backtest engine with trailing stop

**Files:**
- Create: `backtest/dynamic_sl_tp_backtest.py`

This builds on the existing `backtest/backtest_stop_target_direct.py` infrastructure (`preload_all_quotes`, `get_trading_dates`) but adds: trailing stop, parameterized SL/TP, portfolio-level simulation with position sizing.

- [ ] **Step 1: Create backtest engine with trade simulation**

```python
#!/usr/bin/env python3
"""
动态止损/止盈回测引擎 — 支持trailing stop + 参数化SL/TP + 组合仓位

基于 backtest_stop_target_direct.py 的 preload_all_quotes 基础设施,
新增: 移动止盈, portfolio-level仿真, A/B对比.
"""

import sys
import os
import json
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

DB_PATH = os.path.join(PROJECT_ROOT, 'data_adapter', 'stock_data.db')


def simulate_trade_with_trailing(
    code: str,
    buy_date: str,
    buy_price: float,
    stop_loss: float,
    target_price: float,
    all_quotes: pd.DataFrame,
    params: dict,
) -> Optional[dict]:
    """
    模拟单笔交易: 限价买入 → trailing stop / 止盈 / 止损 / 到期退出

    与 backtest_stop_target_direct.simulate_trade 的区别:
    1. 支持移动止盈 (trailing stop)
    2. T+1: 买入当天不可卖
    3. 超参数全部从params读取
    4. 返回更详细的退出信息
    """
    max_hold_days = params['hold']['max_hold_days']
    trailing_trigger = params['trailing']['trailing_trigger_pct']
    trailing_fallback = params['trailing']['trailing_fallback_pct']

    try:
        code_quotes = all_quotes.loc[code]
    except KeyError:
        return None

    # 找到 buy_date 之后的交易日 (buy_date是分析日, 次日开盘买入)
    future = code_quotes[code_quotes.index > buy_date].head(max_hold_days + 1)
    if future.empty or buy_price <= 0 or stop_loss <= 0 or target_price <= 0:
        return None

    # Step 1: 限价买入检查 (第一个交易日)
    first_day = future.iloc[0]
    if first_day['low'] > buy_price * 1.005:
        # 无法成交
        return {'outcome': 'no_fill', 'trade_return': 0, 'hold_days': 0,
                'actual_entry': 0, 'exit_price': 0}

    actual_entry = min(first_day['open'], buy_price)
    if actual_entry <= 0:
        return None

    # Step 2: 逐日模拟 (T+1, 从第二天开始可卖)
    current_stop = stop_loss
    highest_since_entry = max(actual_entry, first_day['high'])
    profit_range = target_price - actual_entry

    remaining = future.iloc[1:]  # 跳过买入日
    for day_idx, (date, row) in enumerate(remaining.iterrows()):
        # 止损检查 (优先)
        if row['low'] <= current_stop:
            exit_price = current_stop
            ret = (exit_price - actual_entry) / actual_entry
            return {'outcome': 'stop_loss', 'trade_return': ret,
                    'hold_days': day_idx + 1, 'actual_entry': actual_entry,
                    'exit_price': exit_price, 'exit_date': date}

        # 止盈检查
        if row['high'] >= target_price:
            exit_price = target_price
            ret = (exit_price - actual_entry) / actual_entry
            return {'outcome': 'take_profit', 'trade_return': ret,
                    'hold_days': day_idx + 1, 'actual_entry': actual_entry,
                    'exit_price': exit_price, 'exit_date': date}

        # 更新最高价
        highest_since_entry = max(highest_since_entry, row['high'])

        # 移动止盈: 浮盈超过目标的trigger% → 至少保本
        if profit_range > 0 and highest_since_entry >= actual_entry + profit_range * trailing_trigger:
            current_stop = max(current_stop, actual_entry)  # 保本线

        # Trailing: 最高价回撤 fallback% × profit_range
        if profit_range > 0:
            trail = highest_since_entry - profit_range * trailing_fallback
            current_stop = max(current_stop, trail)

    # 到期: 按最后一天收盘价卖出
    last_row = remaining.iloc[-1] if len(remaining) > 0 else first_day
    exit_price = last_row['close']
    ret = (exit_price - actual_entry) / actual_entry
    return {'outcome': 'max_hold', 'trade_return': ret,
            'hold_days': len(remaining), 'actual_entry': actual_entry,
            'exit_price': exit_price, 'exit_date': remaining.index[-1] if len(remaining) > 0 else buy_date}
```

- [ ] **Step 2: Add portfolio-level backtest function**

Append to `backtest/dynamic_sl_tp_backtest.py`:

```python
def run_portfolio_backtest(
    report_dir: str,
    params: dict,
    benchmark_codes: List[str] = None,
    start_date: str = None,
    end_date: str = None,
    use_optimizer: bool = True,
    label: str = 'Optimized',
) -> dict:
    """
    组合级别回测: 读取报告 → 计算价格 → 模拟交易 → 聚合收益

    Args:
        report_dir: 含 analysis_data_*.json 的报告目录
        params: 超参数 (来自optimizer_params.json)
        benchmark_codes: 基准指数列表, 默认 ['000300.SH', '932000.CSI']
        use_optimizer: True=新逻辑, False=旧逻辑(对照组)
        label: 回测标签

    Returns:
        {
            'label': str,
            'trades': List[dict],  # 所有交易记录
            'daily_nav': pd.Series,  # 日级净值曲线
            'metrics': dict,  # 年化收益/Sharpe/MaxDD/超额等
            'exit_stats': dict,  # 止损/止盈/到期退出比例
        }
    """
    if benchmark_codes is None:
        benchmark_codes = ['000300.SH', '932000.CSI']

    # 导入现有基础设施
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'backtest'))
    from backtest_report_based import load_reports, HOLDING_DAYS

    # 加载报告
    reports = load_reports(report_dir, rank_field='composite')
    dates = sorted(reports.keys())
    if start_date:
        dates = [d for d in dates if d >= start_date]
    if end_date:
        dates = [d for d in dates if d <= end_date]

    if not dates:
        print(f"  无可用报告: {report_dir}")
        return {}

    print(f"\n{'='*60}")
    print(f"  回测: {label}")
    print(f"  报告: {len(dates)}天, {dates[0]} → {dates[-1]}")
    print(f"  模式: {'新Optimizer' if use_optimizer else '旧逻辑'}")

    # 预加载日线数据
    from backtest_stop_target_direct import preload_all_quotes
    # 需要报告期+持仓期的数据
    all_quotes = preload_all_quotes(dates[0], '2026-12-31')

    # 预加载60日K线用于ATR/支撑/阻力计算
    conn = sqlite3.connect(DB_PATH)
    # 找到start_date前80个交易日
    lookback_dates = [r[0] for r in conn.execute("""
        SELECT DISTINCT trade_date FROM daily_quotes
        WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT 80
    """, (dates[0],)).fetchall()]
    lookback_start = lookback_dates[-1] if lookback_dates else dates[0]
    conn.close()

    kline_quotes = preload_all_quotes(lookback_start, '2026-12-31')

    # 导入PortfolioOptimizer
    if use_optimizer:
        from portfolio_optimizer import PortfolioOptimizer
        optimizer = PortfolioOptimizer(params=params)

    # 模拟交易
    all_trades = []
    trade_cost = 0.00302  # 双边交易成本0.302%
    max_hold = params['hold']['max_hold_days']

    for date in dates:
        stocks = reports[date]
        if not stocks:
            continue

        if use_optimizer:
            # 新逻辑: 动态价格+仓位
            env_score = 50.0  # 默认值, 下方从JSON覆盖

            # 读取env_score (如果JSON中有)
            json_path = Path(report_dir) / f"analysis_data_{date.replace('-','')}.json"
            if json_path.exists():
                try:
                    with open(json_path) as f:
                        data = json.load(f)
                    te = data.get('trading_environment', {})
                    env_score = te.get('total_score', 50.0)
                except:
                    pass

            # 为每只股票计算自适应价格
            for s in stocks:
                code = s.get('code', '')
                try:
                    kline = kline_quotes.loc[code]
                    mask = kline.index <= date
                    kline_before = kline[mask].tail(80)
                    if len(kline_before) < 20:
                        continue
                    highs = kline_before['high'].values
                    lows = kline_before['low'].values
                    closes = kline_before['close'].values

                    stock_info = {
                        'stock_code': code,
                        'close_price': closes[-1],
                        'pred_10d': s.get('pred_10d', 0),
                    }
                    stock_info = optimizer.compute_prices(stock_info, highs, lows, closes, env_score)
                    s['buy_price'] = stock_info['suggested_buy_price']
                    s['stop_loss'] = stock_info['stop_loss_price']
                    s['target'] = stock_info['take_profit_price']
                    s['atr_pct'] = stock_info.get('atr_pct', 0.03)
                    s['composite'] = s.get('rank_score', s.get('score', 0))
                except (KeyError, IndexError):
                    s['buy_price'] = s.get('rank_score', 0)
                    continue

            # 信号筛选+仓位分配
            valid = [s for s in stocks if s.get('buy_price', 0) > 0]
            selected = optimizer.filter_and_allocate(valid, env_score)

        else:
            # 旧逻辑: 固定百分比 (Top 10, 等权)
            sorted_stocks = sorted(stocks, key=lambda s: s.get('rank_score', 0), reverse=True)
            selected = sorted_stocks[:10]
            for s in selected:
                code = s.get('code', '')
                close = s.get('rank_score', 0)  # 需要从quotes获取close
                try:
                    kline = kline_quotes.loc[code]
                    mask = kline.index <= date
                    close = kline[mask].iloc[-1]['close'] if len(kline[mask]) > 0 else 0
                except:
                    close = 0
                pred_10d = s.get('pred_10d', 0)
                is_wide = code.startswith('30') or code.startswith('688')
                s['buy_price'] = round(close * (1 - (0 if (pred_10d or 0) >= 0 else 0.025)), 2)
                base_stop = 0.15 if is_wide else 0.10
                s['stop_loss'] = round(close * (1 - base_stop), 2)
                base_target = 0.12 if is_wide else 0.08
                s['target'] = round(close * (1 + base_target), 2)
                s['position_pct'] = 10.0  # 等权10%

        # 执行交易模拟
        for s in selected:
            code = s.get('code', '')
            result = simulate_trade_with_trailing(
                code, date,
                s.get('buy_price', 0),
                s.get('stop_loss', 0),
                s.get('target', 0),
                all_quotes, params,
            )
            if result and result['outcome'] != 'no_fill':
                result['code'] = code
                result['date'] = date
                result['position_pct'] = s.get('position_pct', 10.0)
                result['trade_return_net'] = result['trade_return'] - trade_cost
                all_trades.append(result)

    # 聚合结果
    if not all_trades:
        print("  无有效交易")
        return {}

    trades_df = pd.DataFrame(all_trades)
    metrics = _compute_metrics(trades_df, max_hold, dates)

    # 基准收益
    for bench_code in benchmark_codes:
        bench_ret = _get_benchmark_return(bench_code, dates[0], dates[-1], max_hold)
        metrics[f'benchmark_{bench_code}_annual'] = bench_ret
        metrics[f'excess_{bench_code}'] = metrics.get('annual_return', 0) - bench_ret

    # 退出统计
    exit_stats = trades_df['outcome'].value_counts(normalize=True).to_dict()

    print(f"\n  交易笔数: {len(all_trades)}")
    print(f"  年化收益: {metrics.get('annual_return', 0):.1%}")
    print(f"  Sharpe: {metrics.get('sharpe', 0):.3f}")
    print(f"  MaxDD: {metrics.get('max_drawdown', 0):.1%}")
    for bc in benchmark_codes:
        print(f"  超额({bc}): {metrics.get(f'excess_{bc}', 0):.1%}")
    print(f"  止损率: {exit_stats.get('stop_loss', 0):.1%}")
    print(f"  止盈率: {exit_stats.get('take_profit', 0):.1%}")
    print(f"  到期率: {exit_stats.get('max_hold', 0):.1%}")

    return {
        'label': label,
        'trades': all_trades,
        'metrics': metrics,
        'exit_stats': exit_stats,
    }


def _compute_metrics(trades_df: pd.DataFrame, hold_days: int,
                     report_dates: list) -> dict:
    """从交易记录计算组合级别指标"""
    # 加权收益 (按仓位权重)
    trades_df['weighted_return'] = trades_df['trade_return_net'] * trades_df['position_pct'] / 100

    # 按日期聚合
    daily_returns = trades_df.groupby('date')['weighted_return'].sum()

    # 年化
    n_periods = len(report_dates)
    periods_per_year = 252 / hold_days
    total_return = (1 + daily_returns).prod() - 1
    n_holding_periods = n_periods  # 每个报告日一个持仓期
    annual_return = (1 + total_return) ** (periods_per_year / max(n_holding_periods, 1)) - 1

    # Sharpe
    if len(daily_returns) > 1:
        mean_ret = daily_returns.mean()
        std_ret = daily_returns.std()
        sharpe = (mean_ret * periods_per_year - 0.02) / (std_ret * np.sqrt(periods_per_year)) if std_ret > 0 else 0
    else:
        sharpe = 0

    # MaxDD
    cumulative = (1 + daily_returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min()

    # 月度胜率
    if len(daily_returns) >= 2:
        monthly_groups = trades_df.groupby(trades_df['date'].str[:7])['weighted_return'].sum()
        monthly_win_rate = (monthly_groups > 0).mean()
    else:
        monthly_win_rate = 0

    return {
        'annual_return': annual_return,
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'monthly_win_rate': monthly_win_rate,
        'n_trades': len(trades_df),
        'avg_return': trades_df['trade_return_net'].mean(),
        'win_rate': (trades_df['trade_return_net'] > 0).mean(),
        'avg_hold_days': trades_df['hold_days'].mean(),
    }


def _get_benchmark_return(index_code: str, start_date: str, end_date: str,
                          hold_days: int) -> float:
    """获取基准年化收益"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT dq.trade_date, dq.close
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.code = ? AND dq.trade_date >= ? AND dq.trade_date <= ?
        ORDER BY dq.trade_date
    """, conn, params=[index_code, start_date, end_date])
    conn.close()

    if len(df) < 2:
        return 0

    total_ret = df['close'].iloc[-1] / df['close'].iloc[0] - 1
    n_days = len(df)
    annual_ret = (1 + total_ret) ** (252 / max(n_days, 1)) - 1
    return annual_ret
```

- [ ] **Step 3: Verify engine loads and runs**

Run: `python3 -c "from backtest.dynamic_sl_tp_backtest import simulate_trade_with_trailing; print('Import OK')"`
Expected: `Import OK`

- [ ] **Step 4: Commit**

```bash
git add backtest/dynamic_sl_tp_backtest.py
git commit -m "feat: 动态SL/TP回测引擎 — trailing stop + portfolio仿真"
```

---

### Task 3: Parameter calibration script

**Files:**
- Create: `scripts/calibrate_optimizer_params.py`

- [ ] **Step 1: Create calibration script with grid search**

```python
#!/usr/bin/env python3
"""
超参数网格搜索校准

校准顺序: Stop → Target → Entry → Filter → Trailing → 全局微调
IS: 2022-01-01~2025-06-30, OOS: 2025-07-01~2026-03-31
综合目标: Sharpe × (1 - |MaxDD|)
"""

import sys
import os
import json
import itertools
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from portfolio_optimizer import load_params
from backtest.dynamic_sl_tp_backtest import run_portfolio_backtest


# ============================================================
# 搜索网格定义
# ============================================================
SEARCH_GRIDS = {
    'stop': {
        'stop.atr_multiplier': [1.0, 1.5, 2.0, 2.5, 3.0],
        'stop.env_mult_bullish': [0.7, 0.85, 1.0],
        'stop.env_mult_bearish': [1.0, 1.2, 1.5],
        'stop.min_stop_pct': [0.02, 0.03, 0.04],
        'stop.max_stop_main': [0.08, 0.10, 0.12],
        'stop.max_stop_wide': [0.12, 0.15, 0.18],
    },
    'target': {
        'target.min_rr_ratio': [1.5, 2.0, 2.5, 3.0],
        'target.target_clip_min': [0.02, 0.03, 0.05],
        'target.target_clip_max': [0.10, 0.15, 0.20],
    },
    'entry': {
        'entry.atr_discount_mult': [0.1, 0.3, 0.5, 0.7, 1.0],
        'entry.support_discount_mult': [0.1, 0.2, 0.3, 0.5],
        'entry.ml_bullish_threshold': [0.002, 0.005, 0.01],
        'entry.ml_bullish_mult': [0.3, 0.5, 0.7],
        'entry.ml_bearish_mult': [1.2, 1.5, 2.0],
        'entry.max_discount': [0.02, 0.03, 0.05],
    },
    'filter': {
        'filter.composite_cutoff': [0, 0.0001, 0.0005, 0.001],
        'filter.min_n': [3, 5, 8],
        'filter.max_n_bull': [10, 15, 20, 30],
        'filter.max_n_bear': [3, 5, 8],
    },
    'trailing': {
        'trailing.trailing_trigger_pct': [0.3, 0.5, 0.6, 0.8],
        'trailing.trailing_fallback_pct': [0.2, 0.3, 0.4, 0.5],
    },
    'hold': {
        'hold.max_hold_days': [5, 10, 15, 20, 30],
    },
}


def set_nested(d: dict, dotted_key: str, value):
    """设置嵌套dict值: 'stop.atr_multiplier' → d['stop']['atr_multiplier']"""
    keys = dotted_key.split('.')
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def objective(metrics: dict) -> float:
    """综合优化目标: Sharpe × (1 - |MaxDD|)"""
    sharpe = metrics.get('sharpe', 0)
    max_dd = abs(metrics.get('max_drawdown', 0))
    return sharpe * (1 - max_dd)


def calibrate_group(group_name: str, grid: dict, base_params: dict,
                    report_dir: str, is_dates: Tuple[str, str],
                    oos_dates: Tuple[str, str]) -> Tuple[dict, list]:
    """
    对一组参数做网格搜索

    Returns: (best_params, search_log)
    """
    param_names = list(grid.keys())
    param_values = list(grid.values())
    combos = list(itertools.product(*param_values))

    print(f"\n{'='*60}")
    print(f"  校准: {group_name} ({len(combos)} 组合)")
    print(f"  IS: {is_dates[0]} → {is_dates[1]}")

    results = []
    for i, combo in enumerate(combos):
        params = json.loads(json.dumps(base_params))  # deep copy
        for name, val in zip(param_names, combo):
            set_nested(params, name, val)

        t0 = time.time()
        result = run_portfolio_backtest(
            report_dir, params,
            start_date=is_dates[0], end_date=is_dates[1],
            label=f'{group_name}_{i}',
        )
        elapsed = time.time() - t0

        if not result:
            continue

        obj = objective(result['metrics'])
        entry = {
            'combo': dict(zip(param_names, combo)),
            'metrics': result['metrics'],
            'objective': obj,
            'time': elapsed,
        }
        results.append(entry)

        if (i + 1) % 10 == 0 or i == len(combos) - 1:
            print(f"  [{i+1}/{len(combos)}] best_obj={max(r['objective'] for r in results):.4f}")

    if not results:
        print("  无有效结果!")
        return base_params, []

    # 排序, 取Top 3在OOS验证
    results.sort(key=lambda r: r['objective'], reverse=True)
    top3 = results[:3]

    print(f"\n  IS Top 3:")
    for j, r in enumerate(top3):
        print(f"    #{j+1}: obj={r['objective']:.4f}, "
              f"Sharpe={r['metrics'].get('sharpe',0):.3f}, "
              f"MaxDD={r['metrics'].get('max_drawdown',0):.1%}")

    # OOS验证
    best_oos = None
    best_oos_obj = -999

    for j, r in enumerate(top3):
        params = json.loads(json.dumps(base_params))
        for name, val in r['combo'].items():
            set_nested(params, name, val)

        oos_result = run_portfolio_backtest(
            report_dir, params,
            start_date=oos_dates[0], end_date=oos_dates[1],
            label=f'{group_name}_OOS_{j}',
        )
        if not oos_result:
            continue

        oos_obj = objective(oos_result['metrics'])
        is_obj = r['objective']

        print(f"    OOS #{j+1}: obj={oos_obj:.4f} (IS={is_obj:.4f}, ratio={oos_obj/is_obj:.1%})")

        # 过拟合检测: OOS < IS × 60%
        if oos_obj < is_obj * 0.6:
            print(f"    ⚠️ 过拟合! OOS/IS={oos_obj/is_obj:.1%} < 60%")
            continue

        if oos_obj > best_oos_obj:
            best_oos_obj = oos_obj
            best_oos = r['combo']

    # 如果所有Top3都过拟合, 用IS中位数参数
    if best_oos is None:
        print("  ⚠️ 全部过拟合, 使用IS中位数参数")
        mid = results[len(results) // 2]
        best_oos = mid['combo']

    # 应用最优参数
    best_params = json.loads(json.dumps(base_params))
    for name, val in best_oos.items():
        set_nested(best_params, name, val)

    print(f"\n  ✅ {group_name} 最优: {best_oos}")
    return best_params, results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='超参数校准')
    parser.add_argument('--report-dir', required=True, help='报告目录')
    parser.add_argument('--is-start', default='2022-01-01', help='IS开始日期')
    parser.add_argument('--is-end', default='2025-06-30', help='IS结束日期')
    parser.add_argument('--oos-start', default='2025-07-01', help='OOS开始日期')
    parser.add_argument('--oos-end', default='2026-03-31', help='OOS结束日期')
    parser.add_argument('--output', default='optimizer_params_calibrated.json', help='输出文件')
    parser.add_argument('--phase', choices=['all', 'stop', 'target', 'entry', 'filter', 'trailing', 'hold'],
                       default='all', help='校准阶段 (默认全部)')
    args = parser.parse_args()

    base_params = load_params()
    is_dates = (args.is_start, args.is_end)
    oos_dates = (args.oos_start, args.oos_end)

    phases = ['stop', 'target', 'entry', 'filter', 'trailing', 'hold']
    if args.phase != 'all':
        phases = [args.phase]

    all_logs = {}
    for phase in phases:
        if phase not in SEARCH_GRIDS:
            continue
        base_params, logs = calibrate_group(
            phase, SEARCH_GRIDS[phase], base_params,
            args.report_dir, is_dates, oos_dates,
        )
        all_logs[phase] = logs

    # 保存校准结果
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(base_params, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 校准完成, 保存到: {output_path}")

    # 保存搜索日志
    log_path = output_path.with_suffix('.log.json')
    with open(log_path, 'w') as f:
        json.dump(all_logs, f, indent=2, default=str)
    print(f"  搜索日志: {log_path}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Verify script loads**

Run: `python3 scripts/calibrate_optimizer_params.py --help`
Expected: Shows argparse help with `--report-dir`, `--phase`, etc.

- [ ] **Step 3: Commit**

```bash
git add scripts/calibrate_optimizer_params.py
git commit -m "feat: 超参数网格搜索校准脚本 — 6阶段IS/OOS验证"
```

---

### Task 4: A/B comparison script

**Files:**
- Create: `backtest/ab_compare.py`

- [ ] **Step 1: Create A/B comparison script**

```python
#!/usr/bin/env python3
"""
A/B对比回测: 现有逻辑 vs 新Portfolio Optimizer

用法:
    python3 backtest/ab_compare.py \
        --report-dir reports/daily_selection_v4.9.0.2_fullmarket \
        --params optimizer_params.json
"""

import sys
import os
import json
import argparse
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from portfolio_optimizer import load_params
from backtest.dynamic_sl_tp_backtest import run_portfolio_backtest


def run_ab(report_dir: str, params_path: str = None,
           start_date: str = None, end_date: str = None):
    """运行A/B对比"""
    params = load_params(params_path) if params_path else load_params()
    benchmarks = ['000300.SH', '932000.CSI']

    # A: 旧逻辑 (固定百分比, Top 10等权, 固定持仓)
    print("\n" + "="*80)
    print("  A方案: 现有逻辑 (固定SL/TP, Top 10等权)")
    result_a = run_portfolio_backtest(
        report_dir, params, benchmarks,
        start_date=start_date, end_date=end_date,
        use_optimizer=False, label='A: Baseline',
    )

    # B: 新逻辑 (自适应价格, 动态SLTP, 风险预算仓位)
    print("\n" + "="*80)
    print("  B方案: Portfolio Optimizer (自适应价格+动态SLTP+风险预算)")
    result_b = run_portfolio_backtest(
        report_dir, params, benchmarks,
        start_date=start_date, end_date=end_date,
        use_optimizer=True, label='B: Optimized',
    )

    # 对比报告
    print("\n" + "="*80)
    print("  A/B 对比结果")
    print("="*80)

    if not result_a or not result_b:
        print("  某方案无结果, 无法对比")
        return

    ma = result_a['metrics']
    mb = result_b['metrics']

    rows = [
        ('年化收益', 'annual_return', '.1%'),
        ('Sharpe', 'sharpe', '.3f'),
        ('MaxDD', 'max_drawdown', '.1%'),
        ('月度胜率', 'monthly_win_rate', '.1%'),
        ('胜率', 'win_rate', '.1%'),
        ('平均收益', 'avg_return', '.2%'),
        ('交易笔数', 'n_trades', 'd'),
        ('平均持仓天数', 'avg_hold_days', '.1f'),
    ]

    for bench in benchmarks:
        rows.append((f'超额({bench})', f'excess_{bench}', '.1%'))

    print(f"\n{'指标':<20} {'A (Baseline)':>15} {'B (Optimized)':>15} {'差异':>12}")
    print("-" * 65)
    for label, key, fmt in rows:
        va = ma.get(key, 0)
        vb = mb.get(key, 0)
        diff = vb - va
        fmt_str = f'{{{fmt}}}'
        print(f"{label:<20} {format(va, fmt):>15} {format(vb, fmt):>15} {format(diff, fmt):>12}")

    # 退出方式对比
    print(f"\n{'退出方式':<20} {'A':>15} {'B':>15}")
    print("-" * 55)
    for outcome in ['stop_loss', 'take_profit', 'max_hold']:
        va = result_a['exit_stats'].get(outcome, 0)
        vb = result_b['exit_stats'].get(outcome, 0)
        print(f"{outcome:<20} {va:>15.1%} {vb:>15.1%}")

    # 判定
    print(f"\n{'='*60}")
    excess_300_a = ma.get('excess_000300.SH', 0)
    excess_300_b = mb.get('excess_000300.SH', 0)
    excess_2000_a = ma.get('excess_932000.CSI', 0)
    excess_2000_b = mb.get('excess_932000.CSI', 0)
    dd_a = abs(ma.get('max_drawdown', 0))
    dd_b = abs(mb.get('max_drawdown', 0))
    sharpe_a = ma.get('sharpe', 0)
    sharpe_b = mb.get('sharpe', 0)

    pass_300 = excess_300_b > excess_300_a
    pass_2000 = excess_2000_b > excess_2000_a
    pass_dd = dd_b <= dd_a + 0.02  # MaxDD不恶化超过2pp
    pass_sharpe = sharpe_b >= sharpe_a * 0.9  # Sharpe >= 90%

    all_pass = pass_300 and pass_2000 and pass_dd and pass_sharpe

    print(f"  对300超额: {'✅' if pass_300 else '❌'} B={excess_300_b:.1%} vs A={excess_300_a:.1%}")
    print(f"  对2000超额: {'✅' if pass_2000 else '❌'} B={excess_2000_b:.1%} vs A={excess_2000_a:.1%}")
    print(f"  MaxDD: {'✅' if pass_dd else '❌'} B={dd_b:.1%} vs A={dd_a:.1%} (容忍+2pp)")
    print(f"  Sharpe: {'✅' if pass_sharpe else '❌'} B={sharpe_b:.3f} vs A={sharpe_a:.3f} (≥90%)")
    print(f"\n  {'🎉 B方案通过! 建议切换到新逻辑' if all_pass else '⚠️ B方案未全部通过, 保留A方案'}")


def main():
    parser = argparse.ArgumentParser(description='A/B对比回测')
    parser.add_argument('--report-dir', required=True, help='报告目录')
    parser.add_argument('--params', default=None, help='参数文件 (默认optimizer_params.json)')
    parser.add_argument('--start-date', default=None, help='开始日期')
    parser.add_argument('--end-date', default=None, help='结束日期')
    args = parser.parse_args()

    run_ab(args.report_dir, args.params, args.start_date, args.end_date)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Verify imports**

Run: `python3 -c "from backtest.ab_compare import run_ab; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add backtest/ab_compare.py
git commit -m "feat: A/B对比回测脚本 — 基线vs新Optimizer"
```

---

### Task 5: Integration with tomorrow_stock_selector.py

**Files:**
- Modify: `tomorrow_stock_selector.py:6100-6129` (argparse)
- Modify: `tomorrow_stock_selector.py:3779-3783` (`_enhance_prices_with_ml` call site)
- Modify: `tomorrow_stock_selector.py:1592-1695` (add optimizer v2 branch)

- [ ] **Step 1: Add `--optimizer` CLI argument**

In `tomorrow_stock_selector.py`, after line 6122 (the `--no-full-market` arg), add:

```python
    parser.add_argument('--optimizer', choices=['v1', 'v2'], default='v1',
                       help='价格/仓位优化器版本: v1=现有逻辑(默认), v2=自适应价格+风险预算')
    parser.add_argument('--optimizer-params', default=None,
                       help='v2优化器参数文件路径 (默认optimizer_params.json)')
```

And update the `main()` call at line 6127-6129 to pass the new args:

```python
    main(target_date=args.date, scoring_version=args.scoring_version,
         stocks_only=args.stocks_only, skip_strategies=args.skip_strategies,
         full_market=full_market, optimizer_version=getattr(args, 'optimizer', 'v1'),
         optimizer_params_path=getattr(args, 'optimizer_params', None))
```

- [ ] **Step 2: Add optimizer_version parameter to main() and StockSelector**

Find the `main()` function definition and add `optimizer_version='v1', optimizer_params_path=None` parameters. Pass them to the `StockSelector` constructor.

In `StockSelector.__init__()`, store: `self.optimizer_version = optimizer_version`.

If `optimizer_version == 'v2'`, initialize:
```python
from portfolio_optimizer import PortfolioOptimizer
self.portfolio_optimizer = PortfolioOptimizer(params_path=optimizer_params_path)
```

- [ ] **Step 3: Add v2 branch in _enhance_prices_with_ml**

At the top of `_enhance_prices_with_ml()` (line 1592), add early return for v2:

```python
    def _enhance_prices_with_ml(self, stock_info: dict) -> dict:
        # V2 optimizer: 使用portfolio_optimizer模块
        if getattr(self, 'optimizer_version', 'v1') == 'v2' and hasattr(self, 'portfolio_optimizer'):
            return self._enhance_prices_with_optimizer_v2(stock_info)

        # ... existing v1 logic unchanged ...
```

Add new method `_enhance_prices_with_optimizer_v2()`:

```python
    def _enhance_prices_with_optimizer_v2(self, stock_info: dict) -> dict:
        """V2: 使用portfolio_optimizer计算自适应价格"""
        close = stock_info.get('close_price', 0)
        if close <= 0:
            return stock_info

        code = stock_info.get('stock_code', '')
        env_score = 50.0
        if hasattr(self, '_cached_env_score'):
            env_score = self._cached_env_score

        # 获取近60日K线
        try:
            from data_adapter.database_manager import DatabaseManager
            db = DatabaseManager()
            df = db.get_stock_daily_data(code, limit=80)
            if df is None or len(df) < 20:
                return stock_info
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
        except Exception:
            return stock_info

        stock_info = self.portfolio_optimizer.compute_prices(
            stock_info, highs, lows, closes, env_score)

        # 仓位由filter_and_allocate在批量阶段处理, 这里给默认值
        stock_info['position_pct'] = stock_info.get('position_pct', 5)
        return stock_info
```

- [ ] **Step 4: Cache env_score for v2 optimizer**

In the `analyze_trading_environment()` call site (around line 5175 in `generate_report()`), after getting env_result, store:

```python
self._cached_env_score = env_result.get('total_score', 50.0)
```

- [ ] **Step 5: Add v2 batch position allocation**

After all stocks are scored (around line 3784, after the loop), add v2 batch allocation:

```python
        # V2: 批量信号筛选+仓位分配
        if getattr(self, 'optimizer_version', 'v1') == 'v2' and hasattr(self, 'portfolio_optimizer'):
            env_score = getattr(self, '_cached_env_score', 50.0)
            stock_with_scores = self.portfolio_optimizer.filter_and_allocate(
                stock_with_scores, env_score)
```

- [ ] **Step 6: Test v2 flag runs without error**

Run: `python3 tomorrow_stock_selector.py 2026-04-01 --optimizer v2 --scoring-version v4.9.0.1 2>&1 | head -30`
Expected: Report generates (may differ in prices from v1). No crash.

- [ ] **Step 7: Commit**

```bash
git add tomorrow_stock_selector.py
git commit -m "feat: --optimizer v2 集成portfolio_optimizer到选股流程"
```

---

### Task 6: Run initial A/B backtest with default params

**Files:** None (execution only)

- [ ] **Step 1: Run A/B comparison with default params**

```bash
python3 backtest/ab_compare.py \
    --report-dir reports/daily_selection_v4.9.0.2_fullmarket \
    --start-date 2024-01-01 --end-date 2026-03-31
```

Expected: Prints comparison table showing A vs B metrics. This is the **baseline** before calibration.

- [ ] **Step 2: Save baseline results**

Redirect output to file:
```bash
python3 backtest/ab_compare.py \
    --report-dir reports/daily_selection_v4.9.0.2_fullmarket \
    --start-date 2024-01-01 --end-date 2026-03-31 \
    2>&1 | tee reports/backtest/ab_compare_default_params.txt
```

- [ ] **Step 3: Commit baseline**

```bash
git add reports/backtest/ab_compare_default_params.txt
git commit -m "eval: A/B对比回测基线 (默认参数)"
```

---

### Task 7: Run parameter calibration

**Files:** None (execution only)

- [ ] **Step 1: Run Stop parameters calibration first**

```bash
python3 scripts/calibrate_optimizer_params.py \
    --report-dir reports/daily_selection_v4.9.0.2_fullmarket \
    --phase stop \
    --output optimizer_params_calibrated.json
```

Expected: Grid search over 5×3×3×3×3×3 = 1215 combinations (may take a while), outputs best stop params.

- [ ] **Step 2: Continue with remaining phases sequentially**

```bash
# Each phase reads the previously calibrated params and adds its optimized group
for phase in target entry filter trailing hold; do
    python3 scripts/calibrate_optimizer_params.py \
        --report-dir reports/daily_selection_v4.9.0.2_fullmarket \
        --phase $phase \
        --output optimizer_params_calibrated.json
done
```

- [ ] **Step 3: Run A/B with calibrated params**

```bash
python3 backtest/ab_compare.py \
    --report-dir reports/daily_selection_v4.9.0.2_fullmarket \
    --params optimizer_params_calibrated.json \
    --start-date 2024-01-01 --end-date 2026-03-31 \
    2>&1 | tee reports/backtest/ab_compare_calibrated.txt
```

- [ ] **Step 4: Check pass criteria**

Review output: all 4 criteria must pass (excess over 300, excess over 2000, MaxDD, Sharpe). If B passes, adopt calibrated params. If not, diagnose which module underperforms and iterate.

- [ ] **Step 5: If passes, promote calibrated params**

```bash
cp optimizer_params_calibrated.json optimizer_params.json
git add optimizer_params.json optimizer_params_calibrated.json
git add reports/backtest/ab_compare_calibrated.txt
git commit -m "feat: 校准后超参数 — A/B验证通过"
```

---

### Task 8: Report format update (new columns)

**Files:**
- Modify: `tomorrow_stock_selector.py:5256-5257` (table header)
- Modify: `tomorrow_stock_selector.py:5546-5567` (table row)

- [ ] **Step 1: Update table header for v2 optimizer**

At line 5256 (the v4.4+ header), wrap in a condition:

```python
if getattr(self, 'optimizer_version', 'v1') == 'v2':
    header = "| 排名 | 股票代码 | 股票名称 | 选中策略 | Composite | 投资建议 | 预测10d | 收盘价 | 买入价 | 止损价 | 目标价 | 仓位 | 止损% | R:R | ATR% |"
    separator = "|------|----------|----------|----------|-----------|----------|---------|--------|--------|--------|--------|------|-------|-----|------|"
else:
    # existing header unchanged
```

- [ ] **Step 2: Update table row for v2 optimizer**

At line 5556 (v4.4+ row), add v2 branch:

```python
if getattr(self, 'optimizer_version', 'v1') == 'v2':
    stop_pct = stock.get('risk_pct', 0)
    rr = stock.get('risk_reward_ratio', 0)
    atr_pct = stock.get('atr_pct', 0) * 100
    report += f"| {i+1} | {stock_code} | {stock_name} | {strategies_str} | {composite_val:.6f} | {recommendation} | {pred_10d*100:+.2f}% | {close_price:.2f} | {buy_price:.2f} | {stop_loss:.2f} | {target:.2f} | {pos_str} | {stop_pct:.1f}% | {rr:.1f} | {atr_pct:.1f}% |\n"
else:
    # existing row unchanged
```

- [ ] **Step 3: Generate a sample v2 report**

Run: `python3 tomorrow_stock_selector.py 2026-04-01 --optimizer v2 --scoring-version v4.9.0.1`

Verify new columns (止损%, R:R, ATR%) appear in the report.

- [ ] **Step 4: Commit**

```bash
git add tomorrow_stock_selector.py
git commit -m "feat: V2报告增加止损%/R:R/ATR%列"
```
