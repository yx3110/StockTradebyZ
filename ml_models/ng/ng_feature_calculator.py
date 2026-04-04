"""
Daily Selection NG — 62-Factor Feature Calculator
==================================================
Factor groups:
  Group 1 (1-12):  Trend State          — price vs MAs, momentum, MACD, channel position
  Group 2 (13-22): Pullback Entry       — RSI, KDJ, volume contraction, shadow ratios
  Group 3 (23-30): Volume Confirmation  — OBV, volume ratios, turnover, up-volume bias
  Group 4 (31-44): Fundamental          — ROE, margins, valuation, size, liquidity
  Group 5 (45-54): Market Environment   — benchmark trend, breadth, northbound flow
  Group 6 (55-62): Industry Rotation    — sector strength, breadth, HHI, relative rank

All functions return Dict[str, float].  Missing values are represented as np.nan.
All divisions are guarded with +1e-8 or explicit checks to avoid ZeroDivisionError.
Dependencies: numpy only (no pandas).
"""

import numpy as np
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _linreg_slope(arr: np.ndarray) -> float:
    """Return OLS slope of arr ~ index, normalised by nothing."""
    n = len(arr)
    if n < 2:
        return np.nan
    x = np.arange(n, dtype=float)
    xm = x - x.mean()
    ym = arr - arr.mean()
    denom = (xm * xm).sum()
    if denom < 1e-12:
        return np.nan
    return float((xm * ym).sum() / denom)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation with safety guards."""
    if len(a) < 3 or len(a) != len(b):
        return np.nan
    a_std = a.std()
    b_std = b.std()
    if a_std < 1e-12 or b_std < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _percentile_rank(value: float, history: np.ndarray) -> float:
    """Return fraction of history values strictly below value (0..1)."""
    if history is None or len(history) == 0:
        return np.nan
    return float(np.mean(history < value))


# ---------------------------------------------------------------------------
# Function 1: Factors 1-30 — Stock-level price/volume/technical features
# ---------------------------------------------------------------------------

def compute_stock_features(
    closes: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    amounts: np.ndarray,
    ma5: np.ndarray,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    atr_14: float,
    macd_macd: np.ndarray,  # MACD histogram values, same length as closes
    kdj_j: float,           # today's KDJ J value
    boll_upper: float,      # today's Bollinger upper band
    boll_lower: float,      # today's Bollinger lower band
    rsi_12: float,          # today's RSI(12)
    rsi_24: float,          # today's RSI(24)
) -> Dict[str, float]:
    """
    Compute 30 stock-level features covering trend state (1-12),
    pullback entry signals (13-22), and volume confirmation (23-30).

    Parameters
    ----------
    closes, opens, highs, lows, volumes, amounts : np.ndarray
        OHLCV arrays, oldest-first, length >= 60.
    ma5/ma10/ma20/ma60 : np.ndarray
        Moving averages, same length as closes.
    atr_14 : float
        14-day Average True Range for today.
    macd_macd : np.ndarray
        MACD histogram (DIF - DEA), same length as closes.
    kdj_j : float
        Today's KDJ J value.
    boll_upper / boll_lower : float
        Today's Bollinger Band upper/lower.
    rsi_12 / rsi_24 : float
        Today's RSI(12) and RSI(24).

    Returns
    -------
    Dict[str, float] with exactly 30 keys.
    """
    result: Dict[str, float] = {}

    close = float(closes[-1])
    high_today = float(highs[-1])
    low_today = float(lows[-1])
    open_today = float(opens[-1])
    vol_today = float(volumes[-1])

    # ---- Group 1: Trend State (factors 1-12) --------------------------------

    # 1. price_above_ma20
    _ma20 = float(ma20[-1])
    result['price_above_ma20'] = close / (_ma20 + 1e-8) - 1.0

    # 2. price_above_ma60
    _ma60 = float(ma60[-1])
    result['price_above_ma60'] = close / (_ma60 + 1e-8) - 1.0

    # 3. ma_alignment
    _ma5 = float(ma5[-1])
    _ma10 = float(ma10[-1])
    alignment_score = float(np.mean([_ma5 > _ma10, _ma10 > _ma20, _ma20 > _ma60]))
    result['ma_alignment'] = alignment_score * (_ma5 - _ma60) / (close + 1e-8)

    # 4. trend_strength_20d
    if len(closes) >= 20:
        c20 = closes[-20:].astype(float)
        slope = _linreg_slope(c20)
        std20 = c20.std()
        result['trend_strength_20d'] = slope / (std20 + 1e-8) if not np.isnan(slope) else np.nan
    else:
        result['trend_strength_20d'] = np.nan

    # 5. new_high_20d
    if len(highs) >= 20:
        result['new_high_20d'] = close / (float(np.max(highs[-20:])) + 1e-8)
    else:
        result['new_high_20d'] = np.nan

    # 6. new_high_60d
    if len(highs) >= 60:
        result['new_high_60d'] = close / (float(np.max(highs[-60:])) + 1e-8)
    else:
        result['new_high_60d'] = np.nan

    # 7. days_since_breakout — consecutive days close > prior 20d high, max 60
    if len(closes) >= 21:
        # Need at least 21 bars to compute "prior 20d high" one day back
        breakout_days = 0
        for i in range(1, min(len(closes), 61)):
            idx = len(closes) - i          # today going backwards
            if idx < 20:
                break
            prior_high = float(np.max(highs[idx - 20: idx]))
            if closes[idx] > prior_high:
                breakout_days += 1
            else:
                break
        result['days_since_breakout'] = float(breakout_days)
    else:
        result['days_since_breakout'] = 0.0

    # 8. adx_proxy
    result['adx_proxy'] = abs(_ma5 - _ma20) / (float(atr_14) + 1e-8)

    # 9. macd_histogram
    result['macd_histogram'] = float(macd_macd[-1]) if len(macd_macd) >= 1 else np.nan

    # 10. macd_acceleration
    if len(macd_macd) >= 6:
        result['macd_acceleration'] = float(macd_macd[-1]) - float(macd_macd[-6])
    else:
        result['macd_acceleration'] = np.nan

    # 11. price_channel_position
    if len(lows) >= 20 and len(highs) >= 20:
        chan_lo = float(np.min(lows[-20:]))
        chan_hi = float(np.max(highs[-20:]))
        result['price_channel_position'] = (close - chan_lo) / (chan_hi - chan_lo + 1e-8)
    else:
        result['price_channel_position'] = np.nan

    # 12. cumulative_return_60d
    if len(closes) >= 60:
        result['cumulative_return_60d'] = close / (float(closes[-60]) + 1e-8) - 1.0
    else:
        result['cumulative_return_60d'] = np.nan

    # ---- Group 2: Pullback Entry (factors 13-22) ----------------------------

    # 13. pullback_from_high (relative to recent 5-day high of closes)
    if len(closes) >= 5:
        recent_high = float(np.max(closes[-5:]))
        result['pullback_from_high'] = 1.0 - close / (recent_high + 1e-8)
    else:
        result['pullback_from_high'] = np.nan

    # 14. pullback_to_ma10
    result['pullback_to_ma10'] = close / (_ma10 + 1e-8) - 1.0

    # 15. pullback_to_ma20
    result['pullback_to_ma20'] = close / (_ma20 + 1e-8) - 1.0

    # 16. rsi_14 (approximation: 0.6*rsi_12 + 0.4*rsi_24)
    result['rsi_14'] = 0.6 * float(rsi_12) + 0.4 * float(rsi_24)

    # 17. kdj_j_value
    result['kdj_j_value'] = float(kdj_j)

    # 18. volume_contraction
    if len(volumes) >= 20:
        vol5_mean = float(np.mean(volumes[-5:]))
        vol20_mean = float(np.mean(volumes[-20:]))
        result['volume_contraction'] = vol5_mean / (vol20_mean + 1e-8)
    else:
        result['volume_contraction'] = np.nan

    # 19. lower_shadow_ratio
    hl_range = high_today - low_today + 1e-8
    result['lower_shadow_ratio'] = (close - low_today) / hl_range

    # 20. consecutive_down_days (max 20)
    down_count = 0
    for i in range(1, min(len(closes), 21)):
        idx = len(closes) - i
        if idx < 1:
            break
        if closes[idx] < closes[idx - 1]:
            down_count += 1
        else:
            break
    result['consecutive_down_days'] = float(down_count)

    # 21. bollinger_position
    result['bollinger_position'] = (close - float(boll_lower)) / (float(boll_upper) - float(boll_lower) + 1e-8)

    # 22. intraday_recovery (mean over last 5 completed bars, i.e. -5 to -1 inclusive)
    if len(closes) >= 5 and len(highs) >= 5 and len(lows) >= 5:
        recoveries = []
        for i in range(-5, 0):
            h = float(highs[i])
            l = float(lows[i])
            c = float(closes[i])
            recoveries.append((c - l) / (h - l + 1e-8))
        result['intraday_recovery'] = float(np.mean(recoveries))
    else:
        result['intraday_recovery'] = np.nan

    # ---- Group 3: Volume Confirmation (factors 23-30) -----------------------

    # 23. volume_ratio_5d
    if len(volumes) >= 20:
        v5 = float(np.mean(volumes[-5:]))
        v20 = float(np.mean(volumes[-20:]))
        result['volume_ratio_5d'] = v5 / (v20 + 1e-8)
    else:
        result['volume_ratio_5d'] = np.nan

    # 24. volume_price_corr
    if len(closes) >= 20 and len(volumes) >= 20:
        result['volume_price_corr'] = _safe_corr(closes[-20:].astype(float), volumes[-20:].astype(float))
    else:
        result['volume_price_corr'] = np.nan

    # 25. obv_trend
    if len(closes) >= 20 and len(volumes) >= 20:
        price_diff = np.diff(closes[-20:].astype(float))
        signs = np.where(price_diff > 0, 1.0, np.where(price_diff < 0, -1.0, 0.0))
        obv = np.cumsum(signs * volumes[-19:].astype(float))
        slope = _linreg_slope(obv)
        v20_mean = float(np.mean(volumes[-20:]))
        result['obv_trend'] = slope / (v20_mean + 1e-8) if not np.isnan(slope) else np.nan
    else:
        result['obv_trend'] = np.nan

    # 26. volume_breakout
    if len(volumes) >= 20:
        v20_mean = float(np.mean(volumes[-20:]))
        result['volume_breakout'] = vol_today / (v20_mean + 1e-8)
    else:
        result['volume_breakout'] = np.nan

    # 27. log_amount_ma5
    if len(amounts) >= 5:
        result['log_amount_ma5'] = float(np.log(float(np.mean(amounts[-5:])) + 1.0))
    else:
        result['log_amount_ma5'] = np.nan

    # 28. turnover_rate — placeholder, overridden by compute_fundamental_features
    result['turnover_rate'] = np.nan

    # 29. up_volume_ratio
    if len(closes) >= 20 and len(volumes) >= 20 and len(opens) >= 20:
        up_mask = closes[-20:] > opens[-20:]
        total_vol = float(volumes[-20:].sum())
        up_vol = float(volumes[-20:][up_mask].sum())
        result['up_volume_ratio'] = up_vol / (total_vol + 1e-8)
    else:
        result['up_volume_ratio'] = np.nan

    # 30. volume_cv
    if len(volumes) >= 20:
        v_arr = volumes[-20:].astype(float)
        result['volume_cv'] = float(v_arr.std()) / (float(v_arr.mean()) + 1e-8)
    else:
        result['volume_cv'] = np.nan

    return result


# ---------------------------------------------------------------------------
# Function 2: Factors 31-44 — Fundamental / valuation features
# ---------------------------------------------------------------------------

def compute_fundamental_features(
    pe_ttm: float,
    pb: float,
    dv_ratio: float,
    circ_mv: float,
    free_share: float,
    total_share: float,
    turnover_rate: float,
    adv_20d: float,
    pe_ttm_history_60d: np.ndarray,
    roe: float,
    roe_prev_year: float,
    profit_to_gr: float,
    netprofit_margin: float,
    ocf_to_profit: float,
    debt_to_assets: float,
    current_ratio: float,
) -> Dict[str, float]:
    """
    Compute 14 fundamental features (31-44).

    Returns
    -------
    Dict[str, float] with exactly 14 keys.
    The 'turnover_rate' key here overrides the placeholder set in
    compute_stock_features().
    """
    result: Dict[str, float] = {}

    # 31. roe_ttm
    result['roe_ttm'] = float(roe)

    # 32. roe_change
    result['roe_change'] = float(roe) - float(roe_prev_year)

    # 33. revenue_growth
    result['revenue_growth'] = float(profit_to_gr)

    # 34. net_profit_margin
    result['net_profit_margin'] = float(netprofit_margin)

    # 35. ocf_quality
    result['ocf_quality'] = float(ocf_to_profit)

    # 36. pe_ttm
    result['pe_ttm'] = float(pe_ttm)

    # 37. pb
    result['pb'] = float(pb)

    # 38. pe_percentile_60d
    if pe_ttm_history_60d is not None and len(pe_ttm_history_60d) > 0:
        result['pe_percentile_60d'] = _percentile_rank(float(pe_ttm), pe_ttm_history_60d.astype(float))
    else:
        result['pe_percentile_60d'] = np.nan

    # 39. debt_to_assets
    result['debt_to_assets'] = float(debt_to_assets)

    # 40. current_ratio
    result['current_ratio'] = float(current_ratio)

    # 41. log_market_cap
    result['log_market_cap'] = float(np.log(float(circ_mv) + 1.0))

    # 42. log_adv_20d
    result['log_adv_20d'] = float(np.log(float(adv_20d) + 1.0))

    # 43. free_float_ratio
    _total = float(total_share)
    result['free_float_ratio'] = float(free_share) / (_total + 1e-8)

    # 44. dv_ratio
    result['dv_ratio'] = float(dv_ratio)

    # Also override the turnover_rate placeholder from compute_stock_features
    result['turnover_rate'] = float(turnover_rate)

    return result


# ---------------------------------------------------------------------------
# Function 3: Factors 45-54 — Market environment features
# ---------------------------------------------------------------------------

def compute_market_features(
    benchmark_closes: np.ndarray,       # CSI300 closes, >= 60, oldest first
    all_stock_returns: np.ndarray,       # 1d returns of all stocks today
    all_stock_highs_20d_ratio: np.ndarray,  # close/max(high,20d) for all stocks
    total_market_amount: np.ndarray,    # daily total turnover, >= 20
    northbound_net_buy_5d: float,
    northbound_std: float,
) -> Dict[str, float]:
    """
    Compute 10 market-environment features (45-54).

    Returns
    -------
    Dict[str, float] with exactly 10 keys.
    """
    result: Dict[str, float] = {}

    bm = benchmark_closes.astype(float)
    amount = total_market_amount.astype(float)

    # 45. market_return_5d
    if len(bm) >= 6:
        result['market_return_5d'] = bm[-1] / (bm[-6] + 1e-8) - 1.0
    else:
        result['market_return_5d'] = np.nan

    # 46. market_return_20d
    if len(bm) >= 21:
        result['market_return_20d'] = bm[-1] / (bm[-21] + 1e-8) - 1.0
    else:
        result['market_return_20d'] = np.nan

    # 47. market_volatility_20d (annualised)
    if len(bm) >= 21:
        log_rets = np.diff(np.log(bm[-21:] + 1e-8))
        vol_20d = float(log_rets.std()) * np.sqrt(252)
        result['market_volatility_20d'] = vol_20d
    else:
        result['market_volatility_20d'] = np.nan
        vol_20d = np.nan

    # 48. market_breadth
    if all_stock_returns is not None and len(all_stock_returns) > 0:
        result['market_breadth'] = float(np.mean(all_stock_returns > 0))
    else:
        result['market_breadth'] = np.nan

    # 49. market_new_high_ratio
    if all_stock_highs_20d_ratio is not None and len(all_stock_highs_20d_ratio) > 0:
        result['market_new_high_ratio'] = float(np.mean(all_stock_highs_20d_ratio > 0.98))
    else:
        result['market_new_high_ratio'] = np.nan

    # 50. northbound_flow_5d (z-score)
    _nb_std = float(northbound_std)
    result['northbound_flow_5d'] = float(northbound_net_buy_5d) / (_nb_std + 1e-8)

    # 51. market_volume_ratio
    if len(amount) >= 20:
        result['market_volume_ratio'] = float(amount[-1]) / (float(np.mean(amount[-20:])) + 1e-8)
    else:
        result['market_volume_ratio'] = np.nan

    # 52. market_drawdown
    if len(bm) >= 60:
        result['market_drawdown'] = bm[-1] / (float(np.max(bm[-60:])) + 1e-8) - 1.0
    else:
        result['market_drawdown'] = np.nan

    # 53. vix_proxy (short-term vol / long-term vol)
    if len(bm) >= 61:
        log_rets_60 = np.diff(np.log(bm[-61:] + 1e-8))
        vol_60d = float(log_rets_60.std()) * np.sqrt(252)
        if not np.isnan(vol_20d):
            result['vix_proxy'] = vol_20d / (vol_60d + 1e-8)
        else:
            result['vix_proxy'] = np.nan
    else:
        result['vix_proxy'] = np.nan

    # 54. market_momentum_diff
    r5 = result.get('market_return_5d', np.nan)
    r20 = result.get('market_return_20d', np.nan)
    if not (np.isnan(r5) or np.isnan(r20)):
        result['market_momentum_diff'] = r5 - r20
    else:
        result['market_momentum_diff'] = np.nan

    return result


# ---------------------------------------------------------------------------
# Function 4: Factors 55-62 — Industry rotation features
# ---------------------------------------------------------------------------

def compute_industry_features(
    stock_return_20d: float,
    industry_stock_returns_1d: np.ndarray,   # 1d returns of all stocks in industry
    industry_stock_returns_5d: np.ndarray,   # 5d returns of all stocks in industry
    industry_stock_returns_20d: np.ndarray,  # 20d returns of all stocks in industry
    industry_amounts_5d: np.ndarray,         # recent 5d daily total amount of industry
    industry_amounts_20d: np.ndarray,        # recent 20d daily total amount of industry
    all_industry_returns_5d: np.ndarray,     # 5d returns for all 31 SW industries
    sw_index_return_5d: float,
) -> Dict[str, float]:
    """
    Compute 8 industry-rotation features (55-62).

    Returns
    -------
    Dict[str, float] with exactly 8 keys.
    """
    result: Dict[str, float] = {}

    # 55. industry_return_5d
    if industry_stock_returns_5d is not None and len(industry_stock_returns_5d) > 0:
        ind_ret5 = float(np.mean(industry_stock_returns_5d))
        result['industry_return_5d'] = ind_ret5
    else:
        ind_ret5 = np.nan
        result['industry_return_5d'] = np.nan

    # 56. industry_return_20d
    if industry_stock_returns_20d is not None and len(industry_stock_returns_20d) > 0:
        ind_ret20 = float(np.mean(industry_stock_returns_20d))
        result['industry_return_20d'] = ind_ret20
    else:
        ind_ret20 = np.nan
        result['industry_return_20d'] = np.nan

    # 57. industry_relative_strength
    if not np.isnan(ind_ret20):
        result['industry_relative_strength'] = float(stock_return_20d) - ind_ret20
    else:
        result['industry_relative_strength'] = np.nan

    # 58. industry_breadth
    if industry_stock_returns_1d is not None and len(industry_stock_returns_1d) > 0:
        result['industry_breadth'] = float(np.mean(industry_stock_returns_1d > 0))
    else:
        result['industry_breadth'] = np.nan

    # 59. industry_volume_change
    if (industry_amounts_5d is not None and len(industry_amounts_5d) > 0 and
            industry_amounts_20d is not None and len(industry_amounts_20d) > 0):
        mean5 = float(np.mean(industry_amounts_5d))
        mean20 = float(np.mean(industry_amounts_20d))
        result['industry_volume_change'] = mean5 / (mean20 + 1e-8)
    else:
        result['industry_volume_change'] = np.nan

    # 60. industry_rank_return_5d — percentile of this industry's 5d return among all industries
    if (all_industry_returns_5d is not None and len(all_industry_returns_5d) > 0
            and not np.isnan(ind_ret5)):
        result['industry_rank_return_5d'] = _percentile_rank(ind_ret5, all_industry_returns_5d.astype(float))
    else:
        result['industry_rank_return_5d'] = np.nan

    # 61. sw_index_return_5d
    result['sw_index_return_5d'] = float(sw_index_return_5d)

    # 62. industry_hhi — Herfindahl-Hirschman Index of abs(individual returns)
    if industry_stock_returns_20d is not None and len(industry_stock_returns_20d) > 0:
        abs_rets = np.abs(industry_stock_returns_20d.astype(float))
        total = abs_rets.sum()
        if total < 1e-12:
            result['industry_hhi'] = np.nan
        else:
            shares = abs_rets / total
            result['industry_hhi'] = float((shares ** 2).sum())
    else:
        result['industry_hhi'] = np.nan

    return result
