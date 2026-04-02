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
