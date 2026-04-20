"""
Daily Selection NG v1.1.0 — Feature Calculator
===============================================
v1.1.0 changes from v1.0.0:
  - REMOVED 11 low cross-sectional-discrimination factors:
    price_above_ma20/60, ma_alignment, new_high_20d/60d,
    macd_histogram/acceleration, price_channel_position,
    cumulative_return_60d, bollinger_position, consecutive_down_days
  - ADDED 10 cross-sectional rank factors (industry-relative percentile)
  - ADDED 5 residual factors (market/industry-neutralized signals)
  - ADDED 3 sector activity factors

Factor groups:
  Group 1 (5):   Trend State          — trend_strength, breakout, adx
  Group 2 (6):   Pullback Entry       — RSI, KDJ, volume contraction, shadow
  Group 3 (7):   Volume Confirmation  — OBV, volume ratios, turnover
  Group 4 (14):  Fundamental          — ROE, margins, valuation, size, liquidity
  Group 5 (10):  Market Environment   — benchmark trend, breadth, northbound flow
  Group 6 (11):  Industry Rotation    — sector strength, breadth, HHI + 3 sector activity
  Group 7 (10):  Cross-Sectional Rank — industry-relative percentile ranks
  Group 8 (5):   Residual Factors     — market/industry-neutralized alpha signals

Total: 5+6+7+14+10+11+10+5 = 68 factors (58 stock + 10 market)

All functions return Dict[str, float].  Missing values are represented as np.nan.
All divisions are guarded with +1e-8 or explicit checks to avoid ZeroDivisionError.
Dependencies: numpy only (no pandas).
"""

import numpy as np
from typing import Dict, List, Optional


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


def _skewness(arr: np.ndarray) -> float:
    """Compute skewness of an array. Returns 0.0 if std is near zero."""
    if len(arr) < 3:
        return np.nan
    std_r = arr.std()
    if std_r < 1e-8:
        return 0.0
    return float(np.mean(((arr - arr.mean()) / std_r) ** 3))


def _industry_percentile_rank(value: float, peer_values: np.ndarray) -> float:
    """Return percentile rank of value among peer_values (0..1).
    Used for cross-sectional rank factors within an industry."""
    if peer_values is None or len(peer_values) < 2:
        return 0.5  # neutral if no peers
    valid = peer_values[~np.isnan(peer_values)]
    if len(valid) < 2:
        return 0.5
    return float(np.mean(valid < value))


# ---------------------------------------------------------------------------
# Function 1: Stock-level price/volume/technical features (19 factors)
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
    macd_macd: np.ndarray,
    kdj_j: float,
    boll_upper: float,
    boll_lower: float,
    rsi_12: float,
    rsi_24: float,
) -> Dict[str, float]:
    """
    Compute 19 stock-level features covering:
      - Trend state (5): trend_strength_20d, days_since_breakout, adx_proxy (3 kept from v1.0.0)
        + pullback_from_high, pullback_to_ma10 moved to trend for clarity... no.

    Actually: 5 trend + 6 pullback + 7 volume + 1 intraday_recovery = 19 stock features.

    v1.1.0 REMOVED:
      price_above_ma20/60, ma_alignment, new_high_20d/60d,
      macd_histogram/acceleration, price_channel_position,
      cumulative_return_60d, bollinger_position, consecutive_down_days

    Parameters same as v1.0.0 (macd_macd etc still passed for potential future use,
    but the removed factors are not computed).
    """
    result: Dict[str, float] = {}

    close = float(closes[-1])
    high_today = float(highs[-1])
    low_today = float(lows[-1])
    open_today = float(opens[-1])
    vol_today = float(volumes[-1])

    _ma5 = float(ma5[-1])
    _ma10 = float(ma10[-1])
    _ma20 = float(ma20[-1])
    _ma60 = float(ma60[-1])

    # ---- Group 1: Trend State (5 factors, was 12 in v1.0.0) -----------------

    # 1. trend_strength_20d
    if len(closes) >= 20:
        c20 = closes[-20:].astype(float)
        slope = _linreg_slope(c20)
        std20 = c20.std()
        result['trend_strength_20d'] = slope / (std20 + 1e-8) if not np.isnan(slope) else np.nan
    else:
        result['trend_strength_20d'] = np.nan

    # 2. days_since_breakout — consecutive days close > prior 20d high, max 60
    if len(closes) >= 21:
        breakout_days = 0
        for i in range(1, min(len(closes), 61)):
            idx = len(closes) - i
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

    # 3. adx_proxy
    result['adx_proxy'] = abs(_ma5 - _ma20) / (float(atr_14) + 1e-8)

    # 4. pullback_from_high (relative to recent 5-day high of closes)
    if len(closes) >= 5:
        recent_high = float(np.max(closes[-5:]))
        result['pullback_from_high'] = 1.0 - close / (recent_high + 1e-8)
    else:
        result['pullback_from_high'] = np.nan

    # 5. volume_contraction — REMOVED (Bug #1: identical to volume_ratio_5d)

    # ---- Group 2: Pullback Entry (6 factors, was 10 in v1.0.0) ---------------
    # REMOVED: bollinger_position, consecutive_down_days

    # 6. pullback_to_ma10
    result['pullback_to_ma10'] = close / (_ma10 + 1e-8) - 1.0

    # 7. pullback_to_ma20
    result['pullback_to_ma20'] = close / (_ma20 + 1e-8) - 1.0

    # 8. rsi_14 (approximation: 0.6*rsi_12 + 0.4*rsi_24)
    result['rsi_14'] = 0.6 * float(rsi_12) + 0.4 * float(rsi_24)

    # 9. kdj_j_value
    result['kdj_j_value'] = float(kdj_j)

    # 10. lower_shadow_ratio
    hl_range = high_today - low_today + 1e-8
    result['lower_shadow_ratio'] = (close - low_today) / hl_range

    # 11. intraday_recovery (mean over last 5 bars)
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

    # ---- Group 3: Volume Confirmation (7 factors, was 8 in v1.0.0) -----------

    # 12. volume_ratio_5d
    if len(volumes) >= 20:
        v5 = float(np.mean(volumes[-5:]))
        v20 = float(np.mean(volumes[-20:]))
        result['volume_ratio_5d'] = v5 / (v20 + 1e-8)
    else:
        result['volume_ratio_5d'] = np.nan

    # 13. volume_price_corr
    if len(closes) >= 20 and len(volumes) >= 20:
        result['volume_price_corr'] = _safe_corr(closes[-20:].astype(float), volumes[-20:].astype(float))
    else:
        result['volume_price_corr'] = np.nan

    # 14. obv_trend
    if len(closes) >= 20 and len(volumes) >= 20:
        price_diff = np.diff(closes[-20:].astype(float))
        signs = np.where(price_diff > 0, 1.0, np.where(price_diff < 0, -1.0, 0.0))
        obv = np.cumsum(signs * volumes[-19:].astype(float))
        slope = _linreg_slope(obv)
        v20_mean = float(np.mean(volumes[-20:]))
        result['obv_trend'] = slope / (v20_mean + 1e-8) if not np.isnan(slope) else np.nan
    else:
        result['obv_trend'] = np.nan

    # 15. volume_breakout
    if len(volumes) >= 20:
        v20_mean = float(np.mean(volumes[-20:]))
        result['volume_breakout'] = vol_today / (v20_mean + 1e-8)
    else:
        result['volume_breakout'] = np.nan

    # 16. log_amount_ma5
    if len(amounts) >= 5:
        result['log_amount_ma5'] = float(np.log(float(np.mean(amounts[-5:])) + 1.0))
    else:
        result['log_amount_ma5'] = np.nan

    # 17. turnover_rate — placeholder, overridden by compute_fundamental_features
    result['turnover_rate'] = np.nan

    # 18. up_volume_ratio
    if len(closes) >= 20 and len(volumes) >= 20 and len(opens) >= 20:
        up_mask = closes[-20:] > opens[-20:]
        total_vol = float(volumes[-20:].sum())
        up_vol = float(volumes[-20:][up_mask].sum())
        result['up_volume_ratio'] = up_vol / (total_vol + 1e-8)
    else:
        result['up_volume_ratio'] = np.nan

    # 19. volume_cv
    if len(volumes) >= 20:
        v_arr = volumes[-20:].astype(float)
        result['volume_cv'] = float(v_arr.std()) / (float(v_arr.mean()) + 1e-8)
    else:
        result['volume_cv'] = np.nan

    return result


# ---------------------------------------------------------------------------
# Function 1b: Signal smoothing features (9 factors, ng1.0.4)
# ---------------------------------------------------------------------------

def compute_smoothing_features(
    closes: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
) -> Dict[str, float]:
    """
    Compute 9 signal smoothing features for ng1.0.4:
      Group 1 - Long-horizon trend (3): trend_strength_60d, ma60_distance, price_channel_pos_40d
      Group 2 - Volatility regime (3): vol_ratio_5d_60d, vol_regime, downside_vol_20d
      Group 3 - Drawdown state (3): current_drawdown, recovery_speed_20d, gap_risk_20d
    """
    result: Dict[str, float] = {}
    close = float(closes[-1])

    # ---- Group 1: Long-Horizon Trend (3) ----

    # 1. trend_strength_60d — linear regression slope normalized by std
    if len(closes) >= 60:
        c60 = closes[-60:].astype(float)
        slope = _linreg_slope(c60)
        std60 = c60.std()
        result['trend_strength_60d'] = slope / (std60 + 1e-8) if not np.isnan(slope) else np.nan
    else:
        result['trend_strength_60d'] = np.nan

    # 2. ma60_distance — relative distance from 60-day moving average
    if len(closes) >= 60:
        ma60 = float(np.mean(closes[-60:]))
        result['ma60_distance'] = close / (ma60 + 1e-8) - 1.0
    else:
        result['ma60_distance'] = np.nan

    # 3. price_channel_pos_40d — position within 40-day high-low channel [0, 1]
    if len(closes) >= 40:
        high_40d = float(np.max(highs[-40:]))
        low_40d = float(np.min(lows[-40:]))
        channel_range = high_40d - low_40d
        result['price_channel_pos_40d'] = (close - low_40d) / (channel_range + 1e-8)
    else:
        result['price_channel_pos_40d'] = np.nan

    # ---- Group 2: Volatility Regime (3) ----

    # Pre-compute shared data for volatility features
    rets_60 = None
    if len(closes) >= 60:
        rets_60 = np.diff(np.log(closes[-60:].astype(float) + 1e-8))

    # 4. vol_ratio_5d_60d — short-term vs long-term realized volatility ratio
    if rets_60 is not None:
        vol_5d = float(np.std(rets_60[-5:])) if len(rets_60) >= 5 else np.nan
        vol_60d = float(np.std(rets_60))
        result['vol_ratio_5d_60d'] = vol_5d / (vol_60d + 1e-8) if not np.isnan(vol_5d) else np.nan
    else:
        result['vol_ratio_5d_60d'] = np.nan

    # 5. vol_regime — 20d realized vol percentile in 250d history [0, 1]
    if len(closes) >= 250:
        log_rets = np.diff(np.log(closes.astype(float) + 1e-8))
        if len(log_rets) >= 250:
            # Vectorized: sliding window of 20d std over last 250 returns
            window_view = np.lib.stride_tricks.sliding_window_view(
                log_rets[-(250):], 20)
            all_vols = window_view.std(axis=1)  # shape: (231,)
            current_vol = all_vols[-1]
            result['vol_regime'] = float(np.mean(all_vols[:-1] < current_vol))
        else:
            result['vol_regime'] = np.nan
    elif rets_60 is not None:
        vol_20d = float(np.std(rets_60[-20:])) if len(rets_60) >= 20 else np.nan
        vol_60d = float(np.std(rets_60))
        result['vol_regime'] = 0.5 if np.isnan(vol_20d) else float(vol_20d > vol_60d)
    else:
        result['vol_regime'] = np.nan

    # 6. downside_vol_20d — std of negative returns over 20d
    if len(closes) >= 21:
        daily_rets = np.diff(closes[-21:].astype(float)) / (closes[-21:-1].astype(float) + 1e-8)
        neg_rets = daily_rets[daily_rets < 0]
        result['downside_vol_20d'] = float(np.std(neg_rets)) if len(neg_rets) >= 3 else 0.0
    else:
        result['downside_vol_20d'] = np.nan

    # ---- Group 3: Drawdown State (3) ----

    # 7. current_drawdown — distance from 60d peak, in [-1, 0]
    if len(closes) >= 60:
        peak_60d = float(np.max(closes[-60:]))
        result['current_drawdown'] = close / (peak_60d + 1e-8) - 1.0
    else:
        result['current_drawdown'] = np.nan

    # 8. recovery_speed_20d — position within 20d high-low range [0, 1]
    if len(closes) >= 20:
        high_20d = float(np.max(closes[-20:]))
        low_20d = float(np.min(closes[-20:]))
        channel = high_20d - low_20d
        result['recovery_speed_20d'] = (close - low_20d) / (channel + 1e-8)
    else:
        result['recovery_speed_20d'] = np.nan

    # 9. gap_risk_20d — vectorized
    if len(closes) >= 21 and len(opens) >= 20:
        prev_closes = closes[-21:-1].astype(float)
        today_opens = opens[-20:].astype(float)
        valid = prev_closes > 1e-8
        gaps = np.abs(today_opens / (prev_closes + 1e-8) - 1.0) > 0.02
        result['gap_risk_20d'] = float(np.sum(gaps & valid)) / 20.0
    else:
        result['gap_risk_20d'] = np.nan

    return result


# ---------------------------------------------------------------------------
# Function 2: Fundamental / valuation features (14 factors, unchanged)
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
    or_yoy: float = np.nan,          # Bug #4 fix: 真正的营收同比增长率
    netprofit_margin: float = np.nan,
    ocf_to_profit: float = np.nan,
    debt_to_assets: float = np.nan,
    current_ratio: float = np.nan,
) -> Dict[str, float]:
    """Compute fundamental features."""
    result: Dict[str, float] = {}

    result['roe_ttm'] = float(roe)
    result['roe_change'] = float(roe) - float(roe_prev_year)
    # Bug #4 fix: profit_to_gr is margin ratio, not growth; or_yoy is real revenue growth
    result['profit_margin_ratio'] = float(profit_to_gr)
    result['revenue_growth'] = float(or_yoy) if not np.isnan(or_yoy) else np.nan
    result['net_profit_margin'] = float(netprofit_margin)
    result['ocf_quality'] = float(ocf_to_profit)
    result['pe_ttm'] = float(pe_ttm)
    result['pb'] = float(pb)

    if pe_ttm_history_60d is not None and len(pe_ttm_history_60d) > 0:
        result['pe_percentile_60d'] = _percentile_rank(float(pe_ttm), pe_ttm_history_60d.astype(float))
    else:
        result['pe_percentile_60d'] = np.nan

    result['debt_to_assets'] = float(debt_to_assets)
    result['current_ratio'] = float(current_ratio)
    result['log_market_cap'] = float(np.log(float(circ_mv) + 1.0))
    result['log_adv_20d'] = float(np.log(float(adv_20d) + 1.0))

    _total = float(total_share)
    result['free_float_ratio'] = float(free_share) / (_total + 1e-8)
    result['dv_ratio'] = float(dv_ratio)
    result['turnover_rate'] = float(turnover_rate)

    # ng1.1.0 P2: composition factors
    _roe = float(roe)
    _pe = float(pe_ttm)
    _pb = float(pb)
    _growth = float(or_yoy) if not np.isnan(or_yoy) else 0.0
    if abs(_growth) > 1.0 and abs(_pe) < 1000:
        result['peg_proxy'] = _pe / (_growth + 1e-8)  # PE / revenue_growth% (or_yoy in %)
    else:
        result['peg_proxy'] = np.nan
    if abs(_roe) > 0.01:
        result['pb_roe_ratio'] = _pb / (_roe + 1e-8)  # PB / ROE
    else:
        result['pb_roe_ratio'] = np.nan

    return result


# ---------------------------------------------------------------------------
# Function 3: Market environment features (10 factors, unchanged)
# ---------------------------------------------------------------------------

def compute_market_features(
    benchmark_closes: np.ndarray,
    all_stock_returns: np.ndarray,
    all_stock_highs_20d_ratio: np.ndarray,
    total_market_amount: np.ndarray,
    northbound_net_buy_5d: float,
    northbound_std: float,
) -> Dict[str, float]:
    """Compute 10 market-environment features. Unchanged from v1.0.0."""
    result: Dict[str, float] = {}

    bm = benchmark_closes.astype(float)
    amount = total_market_amount.astype(float)

    if len(bm) >= 6:
        result['market_return_5d'] = bm[-1] / (bm[-6] + 1e-8) - 1.0
    else:
        result['market_return_5d'] = np.nan

    if len(bm) >= 21:
        result['market_return_20d'] = bm[-1] / (bm[-21] + 1e-8) - 1.0
    else:
        result['market_return_20d'] = np.nan

    if len(bm) >= 21:
        log_rets = np.diff(np.log(bm[-21:] + 1e-8))
        vol_20d = float(log_rets.std()) * np.sqrt(252)
        result['market_volatility_20d'] = vol_20d
    else:
        result['market_volatility_20d'] = np.nan
        vol_20d = np.nan

    if all_stock_returns is not None and len(all_stock_returns) > 0:
        result['market_breadth'] = float(np.mean(all_stock_returns > 0))
    else:
        result['market_breadth'] = np.nan

    if all_stock_highs_20d_ratio is not None and len(all_stock_highs_20d_ratio) > 0:
        result['market_new_high_ratio'] = float(np.mean(all_stock_highs_20d_ratio > 0.98))
    else:
        result['market_new_high_ratio'] = np.nan

    _nb_std = float(northbound_std)
    result['northbound_flow_5d'] = float(northbound_net_buy_5d) / (_nb_std + 1e-8)

    if len(amount) >= 20:
        result['market_volume_ratio'] = float(amount[-1]) / (float(np.mean(amount[-20:])) + 1e-8)
    else:
        result['market_volume_ratio'] = np.nan

    if len(bm) >= 60:
        result['market_drawdown'] = bm[-1] / (float(np.max(bm[-60:])) + 1e-8) - 1.0
    else:
        result['market_drawdown'] = np.nan

    if len(bm) >= 61:
        log_rets_60 = np.diff(np.log(bm[-61:] + 1e-8))
        vol_60d = float(log_rets_60.std()) * np.sqrt(252)
        if not np.isnan(vol_20d):
            result['vix_proxy'] = vol_20d / (vol_60d + 1e-8)
        else:
            result['vix_proxy'] = np.nan
    else:
        result['vix_proxy'] = np.nan

    r5 = result.get('market_return_5d', np.nan)
    r20 = result.get('market_return_20d', np.nan)
    if not (np.isnan(r5) or np.isnan(r20)):
        result['market_momentum_diff'] = r5 - r20
    else:
        result['market_momentum_diff'] = np.nan

    return result


# ---------------------------------------------------------------------------
# Function 4: Industry rotation features (11 factors, was 8 + 3 new sector activity)
# ---------------------------------------------------------------------------

def compute_industry_features(
    stock_return_20d: float,
    industry_stock_returns_1d: np.ndarray,
    industry_stock_returns_5d: np.ndarray,
    industry_stock_returns_20d: np.ndarray,
    industry_amounts_5d: np.ndarray,
    industry_amounts_20d: np.ndarray,
    all_industry_returns_5d: np.ndarray,
    sw_index_return_5d: float,
    # v1.1.0 new params for sector activity features
    market_breadth: float = np.nan,
    market_volume_ratio: float = np.nan,
) -> Dict[str, float]:
    """
    Compute 11 industry-rotation features.
    v1.1.0: +3 sector activity features (sector_breadth_vs_market,
    sector_volume_vs_market, n_sectors_strong).
    """
    result: Dict[str, float] = {}

    # 1. industry_return_5d
    if industry_stock_returns_5d is not None and len(industry_stock_returns_5d) > 0:
        ind_ret5 = float(np.mean(industry_stock_returns_5d))
        result['industry_return_5d'] = ind_ret5
    else:
        ind_ret5 = np.nan
        result['industry_return_5d'] = np.nan

    # 2. industry_return_20d
    if industry_stock_returns_20d is not None and len(industry_stock_returns_20d) > 0:
        ind_ret20 = float(np.mean(industry_stock_returns_20d))
        result['industry_return_20d'] = ind_ret20
    else:
        ind_ret20 = np.nan
        result['industry_return_20d'] = np.nan

    # 3. industry_relative_strength — REMOVED (Bug #3: identical to residual_return_20d)

    # 4. industry_breadth
    if industry_stock_returns_1d is not None and len(industry_stock_returns_1d) > 0:
        ind_breadth = float(np.mean(industry_stock_returns_1d > 0))
        result['industry_breadth'] = ind_breadth
    else:
        ind_breadth = np.nan
        result['industry_breadth'] = np.nan

    # 5. industry_volume_change
    if (industry_amounts_5d is not None and len(industry_amounts_5d) > 0 and
            industry_amounts_20d is not None and len(industry_amounts_20d) > 0):
        mean5 = float(np.mean(industry_amounts_5d))
        mean20 = float(np.mean(industry_amounts_20d))
        ind_vol_change = mean5 / (mean20 + 1e-8)
        result['industry_volume_change'] = ind_vol_change
    else:
        ind_vol_change = np.nan
        result['industry_volume_change'] = np.nan

    # 6. industry_rank_return_5d (= sector_momentum_rank)
    if (all_industry_returns_5d is not None and len(all_industry_returns_5d) > 0
            and not np.isnan(ind_ret5)):
        result['industry_rank_return_5d'] = _percentile_rank(ind_ret5, all_industry_returns_5d.astype(float))
    else:
        result['industry_rank_return_5d'] = np.nan

    # 7. sw_index_return_5d — REMOVED (Bug #2: identical to industry_return_5d)

    # 8. industry_hhi
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

    # ---- v1.1.0: Sector Activity Features (3 new) ---

    # 9. sector_breadth_vs_market — industry breadth / market breadth
    if not np.isnan(ind_breadth) and not np.isnan(market_breadth) and market_breadth > 1e-8:
        result['sector_breadth_vs_market'] = ind_breadth / market_breadth
    else:
        result['sector_breadth_vs_market'] = np.nan

    # 10. sector_volume_vs_market — industry volume change / market volume ratio
    if not np.isnan(ind_vol_change) and not np.isnan(market_volume_ratio) and market_volume_ratio > 1e-8:
        result['sector_volume_vs_market'] = ind_vol_change / market_volume_ratio
    else:
        result['sector_volume_vs_market'] = np.nan

    # 11. n_sectors_strong — count of industries with 5d return > 2%
    if all_industry_returns_5d is not None and len(all_industry_returns_5d) > 0:
        result['n_sectors_strong'] = float(np.sum(all_industry_returns_5d > 0.02))
    else:
        result['n_sectors_strong'] = np.nan

    return result


# ---------------------------------------------------------------------------
# Function 5: Cross-Sectional Rank Features (10 factors, NEW in v1.1.0)
# ---------------------------------------------------------------------------

def compute_cross_sectional_rank_features(
    stock_return_5d: float,
    stock_return_20d: float,
    stock_volume_surge: float,       # volume_ratio_5d for this stock
    stock_turnover: float,           # turnover_rate for this stock
    stock_rsi: float,                # rsi_14 for this stock
    stock_new_high_dist: float,      # close / 20d_high for this stock
    stock_pullback: float,           # pullback_from_high for this stock
    stock_volatility: float,         # 20d return std for this stock
    stock_market_cap: float,         # log_market_cap for this stock
    stock_pe: float,                 # pe_ttm for this stock
    # Industry peer arrays (all stocks in same industry, including self)
    peer_returns_5d: np.ndarray,
    peer_returns_20d: np.ndarray,
    peer_volume_surges: np.ndarray,
    peer_turnovers: np.ndarray,
    peer_rsis: np.ndarray,
    peer_new_high_dists: np.ndarray,
    peer_pullbacks: np.ndarray,
    peer_volatilities: np.ndarray,
    peer_market_caps: np.ndarray,
    peer_pes: np.ndarray,
    # ng1.1.0 P2: new dimensions (optional for backward compat)
    stock_pb: float = np.nan,
    stock_dv: float = np.nan,
    peer_pbs: np.ndarray = None,
    peer_dvs: np.ndarray = None,
) -> Dict[str, float]:
    """
    Compute cross-sectional rank features — percentile rank of this stock's
    characteristics within its industry peers. Range: [0, 1].

    These answer "within the same industry, where does this stock rank?" and
    eliminate the industry-level common movement that caused UMD=3.7 in v1.0.0.
    """
    result: Dict[str, float] = {}

    result['cs_rank_return_5d'] = _industry_percentile_rank(stock_return_5d, peer_returns_5d)
    result['cs_rank_return_20d'] = _industry_percentile_rank(stock_return_20d, peer_returns_20d)
    result['cs_rank_volume_surge'] = _industry_percentile_rank(stock_volume_surge, peer_volume_surges)
    result['cs_rank_turnover'] = _industry_percentile_rank(stock_turnover, peer_turnovers)
    result['cs_rank_rsi'] = _industry_percentile_rank(stock_rsi, peer_rsis)
    result['cs_rank_new_high'] = _industry_percentile_rank(stock_new_high_dist, peer_new_high_dists)
    result['cs_rank_pullback'] = _industry_percentile_rank(stock_pullback, peer_pullbacks)
    result['cs_rank_volatility'] = _industry_percentile_rank(stock_volatility, peer_volatilities)
    result['cs_rank_market_cap'] = _industry_percentile_rank(stock_market_cap, peer_market_caps)
    result['cs_rank_pe'] = _industry_percentile_rank(stock_pe, peer_pes)

    # ng1.1.0 P2: additional cs_rank dimensions
    if peer_pbs is not None and len(peer_pbs) > 0:
        result['cs_rank_pb'] = _industry_percentile_rank(stock_pb, peer_pbs)
    if peer_dvs is not None and len(peer_dvs) > 0:
        result['cs_rank_dv'] = _industry_percentile_rank(stock_dv, peer_dvs)

    return result


# ---------------------------------------------------------------------------
# Function 6: Residual Factors (5 factors, NEW in v1.1.0)
# ---------------------------------------------------------------------------

def compute_residual_features(
    stock_daily_returns: np.ndarray,    # 20d daily log returns for this stock
    market_daily_returns: np.ndarray,   # 20d daily log returns for market (CSI300)
    industry_return_20d: float,         # industry mean 20d return
    stock_avg_volume_5d: float,         # stock's 5d average volume
    industry_avg_volume_5d: float,      # industry average of 5d avg volumes
    stock_return_20d: float,            # stock's 20d cumulative return
    market_return_20d: float,           # market (CSI300) 20d return
    industry_equal_weight_index: Optional[np.ndarray] = None,  # industry EW index (20d)
) -> Dict[str, float]:
    """
    Compute 5 residual factors — signals after removing market and industry effects.

    These capture pure stock-specific alpha, not market or sector momentum.
    """
    result: Dict[str, float] = {}

    # 1. residual_return_20d: stock_return - beta*market_return - industry_return
    #    Simple decomposition: alpha = R_stock - R_industry (skip beta for simplicity/stability)
    if not np.isnan(stock_return_20d) and not np.isnan(industry_return_20d):
        result['residual_return_20d'] = stock_return_20d - industry_return_20d
    else:
        result['residual_return_20d'] = np.nan

    # 2. residual_volume: stock volume deviation from industry average
    if stock_avg_volume_5d > 0 and industry_avg_volume_5d > 0:
        result['residual_volume'] = np.log(stock_avg_volume_5d + 1) - np.log(industry_avg_volume_5d + 1)
    else:
        result['residual_volume'] = np.nan

    # 3. idiosyncratic_volatility: std of residual returns (stock - market)
    if (stock_daily_returns is not None and market_daily_returns is not None
            and len(stock_daily_returns) >= 10 and len(market_daily_returns) >= 10):
        n = min(len(stock_daily_returns), len(market_daily_returns))
        residuals = stock_daily_returns[-n:] - market_daily_returns[-n:]
        result['idiosyncratic_volatility'] = float(np.std(residuals)) * np.sqrt(252)
    else:
        result['idiosyncratic_volatility'] = np.nan

    # 4. residual_skewness: skewness of residual returns
    if (stock_daily_returns is not None and market_daily_returns is not None
            and len(stock_daily_returns) >= 10 and len(market_daily_returns) >= 10):
        n = min(len(stock_daily_returns), len(market_daily_returns))
        residuals = stock_daily_returns[-n:] - market_daily_returns[-n:]
        result['residual_skewness'] = _skewness(residuals)
    else:
        result['residual_skewness'] = np.nan

    # 5. relative_strength_vs_peers: stock / industry equal-weight index ratio
    #    Simplified: use cumulative return ratio
    if not np.isnan(stock_return_20d) and not np.isnan(industry_return_20d):
        # (1 + R_stock) / (1 + R_industry) - 1
        denom = 1.0 + industry_return_20d
        if abs(denom) > 1e-8:
            result['relative_strength_vs_peers'] = (1.0 + stock_return_20d) / denom - 1.0
        else:
            result['relative_strength_vs_peers'] = np.nan
    else:
        result['relative_strength_vs_peers'] = np.nan

    return result


# ---------------------------------------------------------------------------
# Helper utilities for interaction features
# ---------------------------------------------------------------------------

def _safe_mul(a, b) -> float:
    """Multiply two values, return np.nan if either is nan or None."""
    if a is None or b is None:
        return np.nan
    a_f, b_f = float(a), float(b)
    if np.isnan(a_f) or np.isnan(b_f):
        return np.nan
    return a_f * b_f


def _safe_div(a, b, eps: float = 1e-8) -> float:
    """Divide a by b, return np.nan if either is nan or None."""
    if a is None or b is None:
        return np.nan
    a_f, b_f = float(a), float(b)
    if np.isnan(a_f) or np.isnan(b_f):
        return np.nan
    return a_f / (b_f + eps)


# ---------------------------------------------------------------------------
# Function 7: Money Flow Features (8 factors, NEW in v1.1.0)
# ---------------------------------------------------------------------------

def compute_moneyflow_features(
    mf_rows: list,       # List[dict] with keys: net_mf_amount, buy_lg_amount, sell_lg_amount,
                          #   buy_elg_amount, sell_elg_amount, buy_sm_amount, sell_sm_amount,
                          #   buy_md_amount, sell_md_amount
    amounts: np.ndarray,  # Daily trading amounts (for normalization)
    price_changes: np.ndarray,  # Daily price change pct (for divergence)
) -> Dict[str, float]:
    """
    Compute 8 money-flow factors from Tushare moneyflow data.

    Factors:
      1. net_mf_ratio_5d       — net money flow / trading amount (5d)
      2. big_order_ratio        — (big+elg net buy today) / total amount today
      3. big_order_trend_5d     — linreg slope of big net over 5d
      4. small_vs_big_divergence — sign agreement of small vs big net (5d mean)
      5. mf_concentration       — elg share / (sm+md share) today
      6. mf_momentum_10d        — (MA5 - MA10 of net_mf) / std(net_mf 10d)
      7. northbound_stock_5d    — placeholder (filled externally)
      8. mf_volume_divergence   — sign agreement of net_mf vs price_change (5d)

    Returns np.nan for any factor where data is insufficient (< 3 rows).
    All divisions guarded with +1e-8.
    """
    result: Dict[str, float] = {}
    n = len(mf_rows) if mf_rows else 0

    # Insufficient data guard
    if n < 3:
        return {
            'net_mf_ratio_5d': np.nan,
            'big_order_ratio': np.nan,
            'big_order_trend_5d': np.nan,
            'small_vs_big_divergence': np.nan,
            'mf_concentration': np.nan,
            'mf_momentum_10d': np.nan,
            'northbound_stock_5d': 0.0,
            'mf_volume_divergence': np.nan,
        }

    # Extract arrays from mf_rows
    net_mf = np.array([float(r.get('net_mf_amount', 0) or 0) for r in mf_rows])
    buy_lg = np.array([float(r.get('buy_lg_amount', 0) or 0) for r in mf_rows])
    sell_lg = np.array([float(r.get('sell_lg_amount', 0) or 0) for r in mf_rows])
    buy_elg = np.array([float(r.get('buy_elg_amount', 0) or 0) for r in mf_rows])
    sell_elg = np.array([float(r.get('sell_elg_amount', 0) or 0) for r in mf_rows])
    buy_sm = np.array([float(r.get('buy_sm_amount', 0) or 0) for r in mf_rows])
    sell_sm = np.array([float(r.get('sell_sm_amount', 0) or 0) for r in mf_rows])
    buy_md = np.array([float(r.get('buy_md_amount', 0) or 0) for r in mf_rows])
    sell_md = np.array([float(r.get('sell_md_amount', 0) or 0) for r in mf_rows])

    big_net = (buy_lg + buy_elg) - (sell_lg + sell_elg)
    sm_net = buy_sm - sell_sm

    # 1. net_mf_ratio_5d = sum(net_mf[-5:]) / sum(amounts[-5:])
    tail5 = min(n, 5)
    amt_tail5 = amounts[-tail5:] if amounts is not None and len(amounts) >= tail5 else None
    if amt_tail5 is not None and len(amt_tail5) > 0:
        result['net_mf_ratio_5d'] = float(net_mf[-tail5:].sum()) / (float(amt_tail5.sum()) + 1e-8)
    else:
        result['net_mf_ratio_5d'] = np.nan

    # 2. big_order_ratio = (big+elg net buy today) / total_amount today
    today_amt = float(amounts[-1]) if amounts is not None and len(amounts) > 0 else 0.0
    result['big_order_ratio'] = float(big_net[-1]) / (today_amt + 1e-8)

    # 3. big_order_trend_5d = linreg slope of big_net[-5:]
    if tail5 >= 2:
        result['big_order_trend_5d'] = _linreg_slope(big_net[-tail5:])
    else:
        result['big_order_trend_5d'] = np.nan

    # 4. small_vs_big_divergence = mean(sign(sm_net) * sign(big_net)) over 5d
    if tail5 >= 3:
        signs = np.sign(sm_net[-tail5:]) * np.sign(big_net[-tail5:])
        result['small_vs_big_divergence'] = float(signs.mean())
    else:
        result['small_vs_big_divergence'] = np.nan

    # 5. mf_concentration = (elg_amount today / total) / (sm_md_amount today / total)
    elg_today = float(buy_elg[-1]) + float(sell_elg[-1])
    sm_md_today = float(buy_sm[-1]) + float(sell_sm[-1]) + float(buy_md[-1]) + float(sell_md[-1])
    result['mf_concentration'] = elg_today / (sm_md_today + 1e-8)

    # 6. mf_momentum_10d = (MA5 - MA10 of net_mf) / std(net_mf[-10:])
    tail10 = min(n, 10)
    if tail10 >= 5:
        ma5_mf = float(net_mf[-5:].mean())
        ma10_mf = float(net_mf[-tail10:].mean())
        std10 = float(net_mf[-tail10:].std())
        result['mf_momentum_10d'] = (ma5_mf - ma10_mf) / (std10 + 1e-8)
    else:
        result['mf_momentum_10d'] = np.nan

    # 7. northbound_stock_5d — placeholder, filled externally
    result['northbound_stock_5d'] = 0.0

    # 8. mf_volume_divergence = mean(sign(net_mf) * sign(price_change)) over 5d
    if (price_changes is not None and len(price_changes) >= tail5
            and tail5 >= 3):
        pc = price_changes[-tail5:].astype(float)
        signs_div = np.sign(net_mf[-tail5:]) * np.sign(pc)
        result['mf_volume_divergence'] = float(signs_div.mean())
    else:
        result['mf_volume_divergence'] = np.nan

    return result


# ---------------------------------------------------------------------------
# Function 8: Interaction Features (8 factors, NEW in v1.1.0)
# ---------------------------------------------------------------------------

def compute_interaction_features(
    stock_feats: Dict[str, float],
    mf_feats: Dict[str, float],
    industry_feats: Dict[str, float],
    residual_feats: Dict[str, float],
    cs_rank_feats: Dict[str, float],
    fund_feats: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute 8 candidate interaction factors (to be IC-screened during training).

    These capture non-linear cross-group interactions that single factors miss.
    All factor names prefixed with ix_.

    Uses _safe_mul / _safe_div helpers so that any nan input propagates correctly.
    """
    result: Dict[str, float] = {}

    # 1. ix_vol_pullback = volume_ratio_5d * pullback_to_ma20
    result['ix_vol_pullback'] = _safe_mul(
        stock_feats.get('volume_ratio_5d'),
        stock_feats.get('pullback_to_ma20'),
    )

    # 2. ix_big_trend = big_order_ratio * trend_strength_20d
    result['ix_big_trend'] = _safe_mul(
        mf_feats.get('big_order_ratio'),
        stock_feats.get('trend_strength_20d'),
    )

    # 3. ix_rsi_mf = rsi_14 * mf_momentum_10d
    result['ix_rsi_mf'] = _safe_mul(
        stock_feats.get('rsi_14'),
        mf_feats.get('mf_momentum_10d'),
    )

    # 4. ix_ind_big = industry_relative_strength * big_order_ratio
    result['ix_ind_big'] = _safe_mul(
        industry_feats.get('industry_relative_strength'),
        mf_feats.get('big_order_ratio'),
    )

    # 5. ix_mf_efficiency = net_mf_ratio_5d / (turnover_rate + 1e-8)
    # turnover_rate lives in fund_feats (compute_fundamental_features), not stock_feats
    _turnover = fund_feats.get('turnover_rate') if fund_feats else stock_feats.get('turnover_rate')
    result['ix_mf_efficiency'] = _safe_div(
        mf_feats.get('net_mf_ratio_5d'),
        _turnover,
    )

    # 6. ix_vol_surge_pullback = cs_rank_volume_surge * pullback_from_high
    result['ix_vol_surge_pullback'] = _safe_mul(
        cs_rank_feats.get('cs_rank_volume_surge'),
        stock_feats.get('pullback_from_high'),
    )

    # 7. ix_alpha_conc = residual_return_20d * mf_concentration
    result['ix_alpha_conc'] = _safe_mul(
        residual_feats.get('residual_return_20d'),
        mf_feats.get('mf_concentration'),
    )

    # 8. ix_north_cap = northbound_stock_5d * log_market_cap
    # log_market_cap lives in fund_feats (compute_fundamental_features), not stock_feats
    _log_mcap = fund_feats.get('log_market_cap') if fund_feats else stock_feats.get('log_market_cap')
    result['ix_north_cap'] = _safe_mul(
        mf_feats.get('northbound_stock_5d'),
        _log_mcap,
    )

    return result


# ---------------------------------------------------------------------------
# Function 9: Extended Market State Features (8 factors, NEW in ng1.0.7)
# ---------------------------------------------------------------------------

def compute_extended_market_features(
    benchmark_closes: np.ndarray,
    total_market_amount: np.ndarray,
    amv_var1: float,
    amv_macd: float,
    amv_regime_days: int,
    market_breadth_history_5d: np.ndarray,
) -> Dict[str, float]:
    """
    Compute 8 extended market-state features for ng1.0.7.
    These give the model continuous market regime information.
    """
    result: Dict[str, float] = {}

    # 1. amv_var1 — 0AMV activity index (continuous)
    result['amv_var1'] = float(amv_var1) if not np.isnan(amv_var1) else 0.0

    # 2. amv_macd — 0AMV MACD value (momentum of activity)
    result['amv_macd'] = float(amv_macd) if not np.isnan(amv_macd) else 0.0

    # 3. amv_regime_days — days in current regime, normalized by 60
    result['amv_regime_days'] = float(amv_regime_days) / 60.0

    # 4. market_ret_60d — 60-day benchmark return
    bm = benchmark_closes.astype(float) if benchmark_closes is not None else np.array([])
    if len(bm) >= 61:
        result['market_ret_60d'] = bm[-1] / (bm[-61] + 1e-8) - 1.0
    else:
        result['market_ret_60d'] = np.nan

    # 5. market_vol_ratio — short-term vs long-term market volatility
    if len(bm) >= 61:
        log_rets = np.diff(np.log(bm[-61:] + 1e-8))
        vol_5d = float(np.std(log_rets[-5:])) if len(log_rets) >= 5 else np.nan
        vol_60d = float(np.std(log_rets))
        result['market_vol_ratio'] = vol_5d / (vol_60d + 1e-8) if not np.isnan(vol_5d) else np.nan
    else:
        result['market_vol_ratio'] = np.nan

    # 6. breadth_momentum_5d — 5-day change in market breadth
    if market_breadth_history_5d is not None and len(market_breadth_history_5d) >= 2:
        arr = market_breadth_history_5d.astype(float)
        result['breadth_momentum_5d'] = arr[-1] - arr[0]
    else:
        result['breadth_momentum_5d'] = np.nan

    # 7. market_skewness_20d — skewness of 20d market returns
    if len(bm) >= 21:
        log_rets_20 = np.diff(np.log(bm[-21:] + 1e-8))
        result['market_skewness_20d'] = _skewness(log_rets_20)
    else:
        result['market_skewness_20d'] = np.nan

    # 8. liquidity_stress — current market amount / 60d average
    amount = total_market_amount.astype(float) if total_market_amount is not None else np.array([])
    if len(amount) >= 60:
        result['liquidity_stress'] = float(amount[-1]) / (float(np.mean(amount[-60:])) + 1e-8)
    elif len(amount) >= 20:
        result['liquidity_stress'] = float(amount[-1]) / (float(np.mean(amount[-20:])) + 1e-8)
    else:
        result['liquidity_stress'] = np.nan

    return result


# ---------------------------------------------------------------------------
# Function 10: Conditional Interaction Features (7 factors, NEW in ng1.0.7)
# ---------------------------------------------------------------------------

def compute_conditional_interaction_features(
    stock_feats: Dict[str, float],
    fund_feats: Dict[str, float],
    market_feats: Dict[str, float],
    ext_market_feats: Dict[str, float],
    industry_feats: Dict[str, float],
    residual_feats: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute 7 stock×market conditional interaction features for ng1.0.7.
    These let GBDT directly learn regime-conditional stock preferences.
    """
    result: Dict[str, float] = {}

    # 1. cx_beta_mkt_vol = idiosyncratic_volatility × market_volatility_20d
    result['cx_beta_mkt_vol'] = _safe_mul(
        residual_feats.get('idiosyncratic_volatility'),
        market_feats.get('market_volatility_20d'),
    )

    # 2. cx_momentum_trend = cs_rank_return_5d × market_return_20d
    result['cx_momentum_trend'] = _safe_mul(
        stock_feats.get('cs_rank_return_5d',
                        stock_feats.get('pullback_to_ma10', np.nan)),
        market_feats.get('market_return_20d'),
    )

    # 3. cx_ind_mkt_dir = industry_relative_strength × sign(market_return_20d)
    mkt_ret_20d = market_feats.get('market_return_20d', 0.0)
    ind_rs = industry_feats.get('industry_relative_strength', np.nan)
    if not np.isnan(ind_rs) and not np.isnan(mkt_ret_20d):
        result['cx_ind_mkt_dir'] = ind_rs * np.sign(mkt_ret_20d)
    else:
        result['cx_ind_mkt_dir'] = np.nan

    # 4. cx_vol_stress = volume_ratio_5d × liquidity_stress
    result['cx_vol_stress'] = _safe_mul(
        stock_feats.get('volume_ratio_5d'),
        ext_market_feats.get('liquidity_stress'),
    )

    # 5. cx_drawdown_regime = current_drawdown × amv_regime_days
    #    Use pullback_to_ma20 as drawdown proxy (always available)
    result['cx_drawdown_regime'] = _safe_mul(
        stock_feats.get('pullback_to_ma20',
                        stock_feats.get('current_drawdown', np.nan)),
        ext_market_feats.get('amv_regime_days'),
    )

    # 6. cx_value_bear = pe_percentile_60d × (1 - market_breadth)
    pe_pct = fund_feats.get('pe_percentile_60d', np.nan)
    mkt_breadth = market_feats.get('market_breadth', np.nan)
    if not np.isnan(pe_pct) and not np.isnan(mkt_breadth):
        result['cx_value_bear'] = pe_pct * (1.0 - mkt_breadth)
    else:
        result['cx_value_bear'] = np.nan

    # 7. cx_quality_stress = roe_ttm × market_vol_ratio
    result['cx_quality_stress'] = _safe_mul(
        fund_feats.get('roe_ttm'),
        ext_market_feats.get('market_vol_ratio'),
    )

    return result


# ---------------------------------------------------------------------------
# ng1.2.3 helpers: drop-list filter + 70-feature entry point
# ---------------------------------------------------------------------------

NG123_DROP_FEATURES = frozenset([
    'lower_shadow_ratio',
    'volume_cv',
    'volume_contraction',
    'volume_price_corr',
    'industry_hhi',
    'industry_volume_change',
    'n_sectors_strong',
    'peg_proxy',
    'pb_roe_ratio',
    'dv_ratio',
    'up_volume_ratio',
    'ocf_quality',
])


def get_ng123_drop_features() -> frozenset:
    """Return the 12 stock features dropped from ng1.0.1 in ng1.2.3 (per spec §4.3)."""
    return NG123_DROP_FEATURES


def filter_ng123_features(features_dict: Dict[str, float]) -> Dict[str, float]:
    """Remove the 12 dropped features from a ng1.0.1 feature dict."""
    return {k: v for k, v in features_dict.items() if k not in NG123_DROP_FEATURES}


# ---------------------------------------------------------------------------
# ng1.5.0 — Tier B Regime-Refined Features (5 factors)
#   Stock-level (4): industry_regime_agreement, recent_maxdd_60d,
#                     volatility_skew_20d, upside_capture_60d
#   Market-level (1): amv_regime_bull_prob
#
# All functions strict t-snapshot (no `shift(-N)`, no future info).
# ---------------------------------------------------------------------------

NG150_STOCK_REGIME_FEATURES: List[str] = [
    'industry_regime_agreement',
    'recent_maxdd_60d',
    'volatility_skew_20d',
    'upside_capture_60d',
]
NG150_MARKET_REGIME_FEATURES: List[str] = [
    'amv_regime_bull_prob',
]


def compute_ng150_regime_stock_features(
    closes: np.ndarray,
    stock_returns_1d: np.ndarray,
    benchmark_returns_1d: np.ndarray,
    industry_returns_5d_history: np.ndarray,
) -> Dict[str, float]:
    """
    Compute 4 stock-level Tier B regime-refined features.

    Args:
        closes: stock close price array (>= 20 days).
        stock_returns_1d: stock daily log-returns (>= 60 obs).
        benchmark_returns_1d: benchmark daily log-returns aligned with stock_returns_1d;
            the benchmark 5d series is derived internally from these 1d rets.
        industry_returns_5d_history: stock's industry mean 5d return, one per
            trading day over the rolling window (>= 60 obs, last obs = today).

    Returns dict with the 4 feature names; missing values are np.nan.
    All computations use t-snapshot data only (no future info).
    """
    result: Dict[str, float] = {}

    # Derive benchmark 5d cumret from 1d rets (rolling product of (1+r) over 5d).
    mkt5 = None
    if benchmark_returns_1d is not None and len(benchmark_returns_1d) >= 5:
        br = np.asarray(benchmark_returns_1d, dtype=float)
        br = np.where(np.isfinite(br), br, 0.0)
        sw = np.lib.stride_tricks.sliding_window_view(br, 5)
        mkt5 = np.prod(1.0 + sw, axis=1) - 1.0  # len = N - 4

    # 1. industry_regime_agreement — 60d fraction of days where industry 5d
    #    ret direction matches benchmark 5d ret direction. Captures ng1.0.6
    #    "牛→行业跟涨" mechanism.
    ind = industry_returns_5d_history
    if ind is not None and mkt5 is not None and len(ind) >= 20 and len(mkt5) >= 20:
        n = min(len(ind), len(mkt5), 60)
        ind_tail = np.asarray(ind[-n:], dtype=float)
        mkt_tail = np.asarray(mkt5[-n:], dtype=float)
        valid = (~np.isnan(ind_tail)) & (~np.isnan(mkt_tail))
        if valid.sum() >= 10:
            agree = (np.sign(ind_tail[valid]) == np.sign(mkt_tail[valid])).astype(float)
            result['industry_regime_agreement'] = float(agree.mean())
        else:
            result['industry_regime_agreement'] = np.nan
    else:
        result['industry_regime_agreement'] = np.nan

    # 2. recent_maxdd_60d — worst PATH-DEPENDENT drawdown inside 60d window
    #    (peak-to-trough inside window, not current-vs-peak). This differs from
    #    ng1.4.0 `current_drawdown` (= close/peak60-1, snapshot). Value <= 0.
    c = closes.astype(float) if closes is not None else np.array([])
    if len(c) >= 20:
        window = c[-60:] if len(c) >= 60 else c
        running_peak = np.maximum.accumulate(window)
        dd = window / (running_peak + 1e-8) - 1.0  # <= 0 each day
        result['recent_maxdd_60d'] = float(dd.min())
    else:
        result['recent_maxdd_60d'] = np.nan

    # 3. volatility_skew_20d — downside_vol / upside_vol ratio.
    #    Proxy for left-tail dominance when RA label failed (I3).
    r = stock_returns_1d
    if r is not None and len(r) >= 10:
        tail = np.asarray(r[-20:], dtype=float)
        tail = tail[~np.isnan(tail)]
        neg = tail[tail < 0]
        pos = tail[tail > 0]
        if len(neg) >= 3 and len(pos) >= 3:
            result['volatility_skew_20d'] = float(neg.std()) / (float(pos.std()) + 1e-6)
        else:
            result['volatility_skew_20d'] = np.nan
    else:
        result['volatility_skew_20d'] = np.nan

    # 4. upside_capture_60d — on bull days (benchmark up), mean(stock_ret / mkt_ret).
    #    Identifies "bull-inert, bear-sticky" trap stocks.
    if (r is not None and benchmark_returns_1d is not None
            and len(r) >= 20 and len(benchmark_returns_1d) >= 20):
        n = min(len(r), len(benchmark_returns_1d), 60)
        s_tail = np.asarray(r[-n:], dtype=float)
        m_tail = np.asarray(benchmark_returns_1d[-n:], dtype=float)
        valid = (~np.isnan(s_tail)) & (~np.isnan(m_tail)) & (m_tail > 1e-4)
        if valid.sum() >= 5:
            ratios = s_tail[valid] / (m_tail[valid] + 1e-6)
            # clip extreme outliers (single-stock limit-up on flat benchmark day)
            ratios = np.clip(ratios, -10.0, 10.0)
            result['upside_capture_60d'] = float(ratios.mean())
        else:
            result['upside_capture_60d'] = np.nan
    else:
        result['upside_capture_60d'] = np.nan

    return result


def compute_ng150_regime_market_features(
    amv_var1: float,
    amv_macd: float,
    amv_var1_ma60: float,
) -> Dict[str, float]:
    """
    Compute 1 market-level Tier B feature: soft bull probability from 0AMV state.

      bull_score = 0.6 * tanh((var1/ma60 - 1) * 10) + 0.4 * tanh(macd * 5)
      bull_prob  = (bull_score + 1) / 2    # map [-1, 1] → [0, 1]

    amv_var1_ma60 is the 60-day moving average of amv_var1 (computed by the
    caller). Provides a continuous regime signal (vs ng1.0.6's hard 0/1 switch).
    """
    result: Dict[str, float] = {}
    if (np.isnan(amv_var1) or np.isnan(amv_macd) or np.isnan(amv_var1_ma60)
            or amv_var1_ma60 <= 0):
        result['amv_regime_bull_prob'] = np.nan
        return result
    var1_ratio = float(amv_var1) / float(amv_var1_ma60) - 1.0
    score = 0.6 * np.tanh(var1_ratio * 10.0) + 0.4 * np.tanh(float(amv_macd) * 5.0)
    result['amv_regime_bull_prob'] = float((score + 1.0) / 2.0)
    return result
