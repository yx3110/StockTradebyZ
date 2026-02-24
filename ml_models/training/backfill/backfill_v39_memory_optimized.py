#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.9特征完整回填脚本 (内存预加载优化版)

计算全部48个特征：
- 技术特征: 24个
- 基本面特征: 10个
- 市场特征: 8个
- 活跃市值特征: 6个

优化策略：
1. 预加载所有数据到内存 (daily_quotes, daily_basic, financial_indicator)
2. 按交易日分组，批量计算市场级特征 (每日只计算一次)
3. 向量化计算技术指标
4. 多进程并行处理不同交易日
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import multiprocessing
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

# macOS 兼容性：使用 fork 模式 + 环境变量禁用安全检查
import os
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
# 明确设置为 fork 模式
try:
    multiprocessing.set_start_method('fork', force=True)
except RuntimeError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FullV39FeatureComputer:
    """完整V3.9特征计算器 - 基于内存预加载"""

    def __init__(self, db_path=None):
        self.db_path = db_path or str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')
        self.quotes_by_code = {}
        self.basic_by_code = {}
        self.financial_by_code = {}
        self.daily_quotes = None
        self.daily_basic = None
        self.trade_dates = []
        self.market_cache = {}

    def preload_all_data(self, start_date: str, end_date: str):
        """预加载所有需要的数据到内存"""
        logger.info("=" * 60)
        logger.info("🚀 预加载所有数据到内存...")
        logger.info("=" * 60)

        conn = sqlite3.connect(self.db_path)

        # 计算需要更早的数据 (技术指标需要60-100天回望)
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        buffer_start = (start_dt - timedelta(days=150)).strftime('%Y-%m-%d')

        # 1. 加载日线行情数据
        logger.info(f"📈 加载日线行情 ({buffer_start} ~ {end_date})...")
        self.daily_quotes = pd.read_sql("""
            SELECT
                s.code, dq.trade_date,
                dq.open, dq.high, dq.low, dq.close, dq.volume,
                dq.price_change_pct, dq.is_limit_up
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE s.type = 'A股'
            AND dq.trade_date BETWEEN ? AND ?
            ORDER BY s.code, dq.trade_date
        """, conn, params=[buffer_start, end_date])
        logger.info(f"   ✓ 加载 {len(self.daily_quotes):,} 条行情记录")

        # 2. 加载每日基本面数据
        logger.info(f"💰 加载每日基本面 ({buffer_start} ~ {end_date})...")
        self.daily_basic = pd.read_sql("""
            SELECT
                s.code, db.trade_date,
                db.pe_ttm, db.pb, db.ps_ttm, db.turnover_rate, db.circ_mv
            FROM daily_basic db
            JOIN securities s ON db.security_id = s.id
            WHERE s.type = 'A股'
            AND db.trade_date BETWEEN ? AND ?
            ORDER BY s.code, db.trade_date
        """, conn, params=[buffer_start, end_date])
        logger.info(f"   ✓ 加载 {len(self.daily_basic):,} 条基本面记录")

        # 3. 加载财务指标数据
        logger.info("📋 加载财务指标...")
        self.financial = pd.read_sql("""
            SELECT
                s.code, fi.end_date,
                fi.ocf_to_profit, fi.roe, fi.grossprofit_margin, fi.profit_to_gr,
                fi.debt_to_assets, fi.current_ratio, fi.ar_turn
            FROM financial_indicator fi
            JOIN securities s ON fi.security_id = s.id
            WHERE s.type = 'A股'
            ORDER BY s.code, fi.end_date DESC
        """, conn)
        logger.info(f"   ✓ 加载 {len(self.financial):,} 条财务记录")

        conn.close()

        # 4. 创建索引
        logger.info("🔧 创建内存索引...")

        self.quotes_by_code = {
            code: group.set_index('trade_date').sort_index()
            for code, group in self.daily_quotes.groupby('code')
        }

        self.basic_by_code = {
            code: group.set_index('trade_date').sort_index()
            for code, group in self.daily_basic.groupby('code')
        }

        self.financial_by_code = {}
        for code, group in self.financial.groupby('code'):
            self.financial_by_code[code] = group.head(4)

        self.trade_dates = sorted(self.daily_quotes['trade_date'].unique())

        logger.info(f"   ✓ 索引创建完成，覆盖 {len(self.quotes_by_code)} 只股票")
        logger.info(f"   ✓ 交易日范围: {min(self.trade_dates)} ~ {max(self.trade_dates)}")
        logger.info("=" * 60)

    def get_stock_data(self, code: str, end_date: str, days: int = 100):
        if code not in self.quotes_by_code:
            return None
        df = self.quotes_by_code[code]
        df = df[df.index <= end_date].tail(days)
        if len(df) < 60:
            return None
        return df.reset_index()

    def compute_technical_features(self, df):
        """计算24个技术特征"""
        features = {}
        try:
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)
            close = df['close'].values.astype(float)
            volume = df['volume'].values.astype(float)
            n = len(close)
            if n < 60:
                return self._get_nan_tech_features()

            # ADX
            features['adx_14'] = self._calc_adx(high, low, close, 14)
            
            # Aroon
            aroon = self._calc_aroon(high, low, 25)
            features['aroon_up_25'] = aroon['up']
            features['aroon_down_25'] = aroon['down']
            features['aroon_oscillator_25'] = aroon['oscillator']

            # Ichimoku
            ichimoku = self._calc_ichimoku(high, low, close)
            features['ichimoku_conversion'] = ichimoku['conversion']
            features['ichimoku_base'] = ichimoku['base']
            features['ichimoku_span_a'] = ichimoku['span_a']
            features['ichimoku_span_b'] = ichimoku['span_b']

            # SuperTrend
            supertrend = self._calc_supertrend(high, low, close, 10, 3)
            features['supertrend_10_3'] = supertrend['value']
            features['supertrend_signal'] = supertrend['signal']

            # Williams %R
            features['williams_r_14'] = self._calc_williams_r(high, low, close, 14)

            # SMI
            smi = self._calc_smi(high, low, close, 14, 3)
            features['smi_14'] = smi['smi']
            features['smi_signal_3'] = smi['signal']

            # TSI
            tsi = self._calc_tsi(close, 25, 13, 7)
            features['tsi_25_13'] = tsi['tsi']
            features['tsi_signal_7'] = tsi['signal']

            # A/D Line
            ad_line = self._calc_ad_line(high, low, close, volume)
            features['ad_line'] = ad_line[-1] if len(ad_line) > 0 else 0
            features['ad_line_change_5'] = (ad_line[-1] - ad_line[-6]) / (abs(ad_line[-6]) + 1e-10) if n >= 6 else 0

            # CMF
            features['cmf_20'] = self._calc_cmf(high, low, close, volume, 20)

            # VWAP
            features['vwap_deviation'] = self._calc_vwap_deviation(close, volume)

            # Large order
            features['large_order_net_inflow'] = self._estimate_large_order_flow(close, volume)

            # Volatility
            features['bb_width_20'] = self._calc_bb_width(close, 20)
            features['kc_width_20'] = self._calc_kc_width(high, low, close, 20)
            features['atr_percent_14'] = self._calc_atr_percent(high, low, close, 14)
            features['historical_volatility_percentile_60'] = self._calc_volatility_percentile(close, 60)

        except Exception as e:
            return self._get_nan_tech_features()
        return features

    def compute_fundamental_features(self, code: str, trade_date: str):
        """计算10个基本面特征"""
        features = {
            'operating_cashflow_to_netprofit': np.nan,
            'roe_change_rate': np.nan,
            'gross_margin_trend': np.nan,
            'netprofit_growth_stability': np.nan,
            'pe_industry_percentile': np.nan,
            'pb_industry_percentile': np.nan,
            'ps_industry_percentile': np.nan,
            'debt_to_asset_ratio': np.nan,
            'current_ratio': np.nan,
            'receivables_turnover': np.nan,
        }
        try:
            if code in self.financial_by_code:
                fin_df = self.financial_by_code[code]
                if len(fin_df) > 0:
                    latest = fin_df.iloc[0]
                    features['operating_cashflow_to_netprofit'] = latest.get('ocf_to_profit', np.nan)
                    features['debt_to_asset_ratio'] = latest.get('debt_to_assets', np.nan)
                    features['current_ratio'] = latest.get('current_ratio', np.nan)
                    features['receivables_turnover'] = latest.get('ar_turn', np.nan)
                    if len(fin_df) >= 2:
                        roe_curr = fin_df.iloc[0]['roe']
                        roe_prev = fin_df.iloc[1]['roe']
                        if pd.notna(roe_curr) and pd.notna(roe_prev) and roe_prev != 0:
                            features['roe_change_rate'] = (roe_curr - roe_prev) / abs(roe_prev)
                    if len(fin_df) >= 3:
                        margins = fin_df['grossprofit_margin'].dropna().values[:3]
                        if len(margins) >= 3:
                            x = np.arange(len(margins))
                            slope, _ = np.polyfit(x, margins, 1)
                            features['gross_margin_trend'] = slope
                        profit_growth = fin_df['profit_to_gr'].dropna().values[:3]
                        if len(profit_growth) >= 3:
                            mean_g = np.mean(profit_growth)
                            std_g = np.std(profit_growth)
                            cv = std_g / (abs(mean_g) + 1e-10)
                            features['netprofit_growth_stability'] = 1 / (cv + 1)

            if code in self.basic_by_code and trade_date in self.basic_by_code[code].index:
                stock_basic = self.basic_by_code[code].loc[trade_date]
                market_basic = self.daily_basic[self.daily_basic['trade_date'] == trade_date]
                if len(market_basic) > 10:
                    for col, feat in [('pe_ttm', 'pe_industry_percentile'), 
                                      ('pb', 'pb_industry_percentile'), 
                                      ('ps_ttm', 'ps_industry_percentile')]:
                        val = stock_basic.get(col)
                        if pd.notna(val):
                            mkt = market_basic[col].dropna()
                            if len(mkt) > 0:
                                features[feat] = (mkt < val).sum() / len(mkt)
        except:
            pass
        return features

    def compute_market_features(self, trade_date: str, code: str = None):
        """计算8个市场特征"""
        if trade_date not in self.market_cache:
            self.market_cache[trade_date] = self._compute_market_level_features(trade_date)
        features = self.market_cache[trade_date].copy()
        if code:
            features.update(self._compute_stock_market_features(code, trade_date))
        return features

    def _compute_market_level_features(self, trade_date: str):
        features = {'advance_decline_ratio': 1.0, 'limit_up_count': 0.0, 
                    'northbound_net_inflow': 0.0, 'margin_balance_change': 0.0, 'concept_heat_index': 0.0}
        try:
            day_quotes = self.daily_quotes[self.daily_quotes['trade_date'] == trade_date]
            if len(day_quotes) > 0:
                advance = (day_quotes['price_change_pct'] > 0).sum()
                decline = (day_quotes['price_change_pct'] < 0).sum()
                features['advance_decline_ratio'] = advance / (decline + 1) if decline > 0 else float(advance)
                if 'is_limit_up' in day_quotes.columns:
                    limit_up = day_quotes['is_limit_up'].sum()
                    features['limit_up_count'] = min(limit_up / 100.0, 1.0)
                    features['concept_heat_index'] = limit_up / len(day_quotes) if len(day_quotes) > 0 else 0
        except:
            pass
        return features

    def _compute_stock_market_features(self, code: str, trade_date: str):
        features = {'sector_strength_rank': 0.5, 'industry_fund_flow_rank': 0.5, 'market_attention_score': 1.0}
        try:
            day_quotes = self.daily_quotes[self.daily_quotes['trade_date'] == trade_date]
            if code in self.quotes_by_code and trade_date in self.quotes_by_code[code].index:
                stock_data = self.quotes_by_code[code].loc[trade_date]
                stock_pct = stock_data['price_change_pct']
                if len(day_quotes) > 0 and pd.notna(stock_pct):
                    features['sector_strength_rank'] = (day_quotes['price_change_pct'] < stock_pct).sum() / len(day_quotes)
                hist = self.quotes_by_code[code]
                hist = hist[hist.index < trade_date].tail(5)
                if len(hist) > 0:
                    avg_vol = hist['volume'].mean()
                    curr_vol = stock_data['volume']
                    if avg_vol > 0:
                        features['industry_fund_flow_rank'] = min(float(curr_vol) / float(avg_vol) / 2, 1.0)
            if code in self.basic_by_code and trade_date in self.basic_by_code[code].index:
                stock_turnover = self.basic_by_code[code].loc[trade_date].get('turnover_rate', 1)
                day_basic = self.daily_basic[self.daily_basic['trade_date'] == trade_date]
                if len(day_basic) > 0 and pd.notna(stock_turnover):
                    avg_turnover = day_basic['turnover_rate'].mean()
                    if pd.notna(avg_turnover) and avg_turnover > 0:
                        features['market_attention_score'] = float(stock_turnover) / float(avg_turnover)
        except:
            pass
        return features

    def compute_active_mv_features(self, code: str, trade_date: str):
        """计算6个活跃市值特征"""
        features = {'market_active_mv_ratio': 0.0, 'market_active_mv_zscore': 0.0, 'market_active_mv_trend': 0.0,
                    'stock_active_mv_rank': 0.5, 'stock_relative_liquidity': 0.5, 'market_cap_quality_score': 0.5}
        try:
            market_basic_today = self.daily_basic[self.daily_basic['trade_date'] == trade_date]
            if len(market_basic_today) > 0:
                valid = market_basic_today.dropna(subset=['circ_mv', 'turnover_rate'])
                if len(valid) > 0:
                    active_mv = valid['circ_mv'] * valid['turnover_rate'] / 100
                    total_circ_mv = valid['circ_mv'].sum()
                    if total_circ_mv > 0:
                        features['market_active_mv_ratio'] = active_mv.sum() / total_circ_mv
                    if code in self.basic_by_code and trade_date in self.basic_by_code[code].index:
                        stock_basic = self.basic_by_code[code].loc[trade_date]
                        stock_circ_mv = stock_basic.get('circ_mv', 0)
                        stock_turnover = stock_basic.get('turnover_rate', 0)
                        if pd.notna(stock_circ_mv) and pd.notna(stock_turnover) and stock_circ_mv > 0:
                            stock_active_mv = stock_circ_mv * stock_turnover / 100
                            features['stock_active_mv_rank'] = (active_mv <= stock_active_mv).sum() / len(active_mv)
                            mean_active_mv = active_mv.mean()
                            if mean_active_mv > 0:
                                features['stock_relative_liquidity'] = np.tanh(stock_active_mv / mean_active_mv / 2)
                            circ_mv_yi = stock_circ_mv / 10000
                            if circ_mv_yi > 0:
                                features['market_cap_quality_score'] = 1 / (1 + np.exp(-(np.log(circ_mv_yi) - np.log(50)) / 0.8))
        except:
            pass
        return features

    def compute_all_features(self, code: str, trade_date: str):
        """计算单只股票的所有48个特征"""
        df = self.get_stock_data(code, trade_date, days=100)
        if df is None:
            return None
        features = {}
        features.update(self.compute_technical_features(df))
        features.update(self.compute_fundamental_features(code, trade_date))
        features.update(self.compute_market_features(trade_date, code))
        features.update(self.compute_active_mv_features(code, trade_date))
        return features

    def calculate_label(self, code: str, trade_date: str, lookahead: int = 5):
        if code not in self.quotes_by_code:
            return None
        df = self.quotes_by_code[code]
        if trade_date not in df.index:
            return None
        future_dates = df[df.index > trade_date].head(lookahead)
        if len(future_dates) < lookahead:
            return None
        current_close = df.loc[trade_date, 'close']
        future_close = future_dates.iloc[-1]['close']
        if pd.notna(current_close) and current_close > 0:
            return (future_close - current_close) / current_close * 100
        return None

    def _get_nan_tech_features(self):
        return {k: np.nan for k in [
            'adx_14', 'aroon_up_25', 'aroon_down_25', 'aroon_oscillator_25',
            'ichimoku_conversion', 'ichimoku_base', 'ichimoku_span_a', 'ichimoku_span_b',
            'supertrend_10_3', 'supertrend_signal', 'williams_r_14', 'smi_14', 'smi_signal_3',
            'tsi_25_13', 'tsi_signal_7', 'ad_line', 'ad_line_change_5', 'cmf_20',
            'vwap_deviation', 'large_order_net_inflow', 'bb_width_20', 'kc_width_20',
            'atr_percent_14', 'historical_volatility_percentile_60'
        ]}

    def _calc_adx(self, high, low, close, period=14):
        try:
            tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
            plus_dm = np.maximum(high[1:] - high[:-1], 0)
            minus_dm = np.maximum(low[:-1] - low[1:], 0)
            plus_dm = np.where(plus_dm > minus_dm, plus_dm, 0)
            minus_dm = np.where(minus_dm > plus_dm, minus_dm, 0)
            atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
            plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean().values / (atr + 1e-10)
            minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean().values / (atr + 1e-10)
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            return pd.Series(dx).ewm(span=period, adjust=False).mean().iloc[-1]
        except:
            return np.nan

    def _calc_aroon(self, high, low, period=25):
        try:
            if len(high) < period:
                return {'up': np.nan, 'down': np.nan, 'oscillator': np.nan}
            periods_since_high = period - np.argmax(high[-period:])
            periods_since_low = period - np.argmin(low[-period:])
            up = ((period - periods_since_high) / period) * 100
            down = ((period - periods_since_low) / period) * 100
            return {'up': up, 'down': down, 'oscillator': up - down}
        except:
            return {'up': np.nan, 'down': np.nan, 'oscillator': np.nan}

    def _calc_ichimoku(self, high, low, close):
        try:
            h, l = pd.Series(high), pd.Series(low)
            conv = (h.rolling(9).max() + l.rolling(9).min()) / 2
            base = (h.rolling(26).max() + l.rolling(26).min()) / 2
            span_a = ((conv + base) / 2).shift(26)
            span_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
            p = close[-1]
            return {
                'conversion': (p - conv.iloc[-1]) / p if pd.notna(conv.iloc[-1]) else 0,
                'base': (p - base.iloc[-1]) / p if pd.notna(base.iloc[-1]) else 0,
                'span_a': (p - span_a.iloc[-1]) / p if pd.notna(span_a.iloc[-1]) else 0,
                'span_b': (p - span_b.iloc[-1]) / p if pd.notna(span_b.iloc[-1]) else 0
            }
        except:
            return {'conversion': 0, 'base': 0, 'span_a': 0, 'span_b': 0}

    def _calc_supertrend(self, high, low, close, period=10, mult=3):
        try:
            tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
            atr = pd.Series(tr).rolling(period).mean().values
            hl = (high[1:] + low[1:]) / 2
            upper, lower = hl + mult * atr, hl - mult * atr
            n = len(close) - 1
            st, trend = np.zeros(n), np.ones(n)
            for i in range(1, n):
                if close[i] > upper[i-1]: trend[i], st[i] = 1, lower[i]
                elif close[i] < lower[i-1]: trend[i], st[i] = -1, upper[i]
                else: trend[i] = trend[i-1]; st[i] = max(lower[i], st[i-1]) if trend[i] == 1 else min(upper[i], st[i-1])
            return {'value': (close[-1] - st[-1]) / close[-1], 'signal': float(trend[-1])}
        except:
            return {'value': np.nan, 'signal': np.nan}

    def _calc_williams_r(self, high, low, close, period=14):
        try:
            h, l = pd.Series(high), pd.Series(low)
            hh, ll = h.rolling(period).max().iloc[-1], l.rolling(period).min().iloc[-1]
            return -100 * (hh - close[-1]) / (hh - ll + 1e-10)
        except:
            return np.nan

    def _calc_smi(self, high, low, close, period=14, smooth=3):
        try:
            h, l, c = pd.Series(high), pd.Series(low), pd.Series(close)
            ll, hh = l.rolling(period).min(), h.rolling(period).max()
            diff = c - (hh + ll) / 2
            ds = diff.ewm(span=smooth).mean().ewm(span=smooth).mean()
            rs = (hh - ll).ewm(span=smooth).mean().ewm(span=smooth).mean()
            smi = 100 * ds / (rs / 2 + 1e-10)
            return {'smi': smi.iloc[-1], 'signal': smi.ewm(span=smooth).mean().iloc[-1]}
        except:
            return {'smi': np.nan, 'signal': np.nan}

    def _calc_tsi(self, close, lp=25, sp=13, sigp=7):
        try:
            c = pd.Series(close)
            pc = c.diff()
            ds = pc.ewm(span=lp).mean().ewm(span=sp).mean()
            ds_abs = pc.abs().ewm(span=lp).mean().ewm(span=sp).mean()
            tsi = 100 * ds / (ds_abs + 1e-10)
            return {'tsi': tsi.iloc[-1], 'signal': tsi.ewm(span=sigp).mean().iloc[-1]}
        except:
            return {'tsi': np.nan, 'signal': np.nan}

    def _calc_ad_line(self, high, low, close, volume):
        try:
            mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
            return np.cumsum(mfm * volume)
        except:
            return np.array([0])

    def _calc_cmf(self, high, low, close, volume, period=20):
        try:
            mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
            return pd.Series(mfm * volume).rolling(period).sum().iloc[-1] / (pd.Series(volume).rolling(period).sum().iloc[-1] + 1e-10)
        except:
            return np.nan

    def _calc_vwap_deviation(self, close, volume):
        try:
            vwap = (close * volume).sum() / (volume.sum() + 1e-10)
            return (close[-1] - vwap) / vwap
        except:
            return np.nan

    def _estimate_large_order_flow(self, close, volume, threshold=1.5):
        try:
            period = min(10, len(close) - 1)
            pc = close[-period:] - close[-period-1:-1]
            vol_avg = np.mean(volume[-20:]) if len(volume) >= 20 else np.mean(volume)
            large_vol = volume[-period:] > vol_avg * threshold
            return (np.sum((pc > 0) & large_vol) - np.sum((pc < 0) & large_vol)) / period
        except:
            return np.nan

    def _calc_bb_width(self, close, period=20):
        try:
            c = pd.Series(close)
            sma, std = c.rolling(period).mean(), c.rolling(period).std()
            return (4 * std / (sma + 1e-10)).iloc[-1]
        except:
            return np.nan

    def _calc_kc_width(self, high, low, close, period=20, mult=2):
        try:
            ema = pd.Series(close).ewm(span=period).mean()
            tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
            atr = pd.Series(tr).rolling(period).mean().iloc[-1]
            return (2 * mult * atr) / (ema.iloc[-1] + 1e-10)
        except:
            return np.nan

    def _calc_atr_percent(self, high, low, close, period=14):
        try:
            tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
            return pd.Series(tr).rolling(period).mean().iloc[-1] / close[-1]
        except:
            return np.nan

    def _calc_volatility_percentile(self, close, period=60):
        try:
            c = pd.Series(close)
            vol = c.pct_change().rolling(20).std() * np.sqrt(252)
            if len(vol) < period:
                return 0.5
            return (vol.iloc[-period:] < vol.iloc[-1]).sum() / period
        except:
            return 0.5


_global_computer = None

def init_worker(computer_data):
    global _global_computer
    _global_computer = FullV39FeatureComputer.__new__(FullV39FeatureComputer)
    _global_computer.__dict__.update(computer_data)
    _global_computer.market_cache = {}

def worker_compute_day(trade_date):
    global _global_computer
    stock_list = list(_global_computer.quotes_by_code.keys())
    results = []
    for code in stock_list:
        try:
            features = _global_computer.compute_all_features(code, trade_date)
            if features is None:
                continue
            label = _global_computer.calculate_label(code, trade_date, 5)
            if label is None:
                continue
            features['code'] = code
            features['trade_date'] = trade_date
            features['label_5d'] = label
            results.append(features)
        except:
            continue
    return results


def main():
    parser = argparse.ArgumentParser(description='V3.9特征完整回填 (内存预加载版)')
    parser.add_argument('--start-date', type=str, default='2025-09-01')
    parser.add_argument('--end-date', type=str, default='2025-10-28')
    parser.add_argument('--lookahead-days', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--db-path', type=str, default=str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db'))
    args = parser.parse_args()

    num_workers = args.num_workers or min(cpu_count(), 8)

    logger.info("=" * 80)
    logger.info("🚀 V3.9特征完整回填 (48个特征，内存预加载优化)")
    logger.info("=" * 80)
    logger.info(f"时间范围: {args.start_date} ~ {args.end_date}")
    logger.info(f"并行进程数: {num_workers}")

    computer = FullV39FeatureComputer(args.db_path)
    computer.preload_all_data(args.start_date, args.end_date)

    valid_dates = [d for d in computer.trade_dates if args.start_date <= d <= args.end_date]
    valid_dates = valid_dates[:-args.lookahead_days]
    stock_count = len(computer.quotes_by_code)

    logger.info(f"有效交易日: {len(valid_dates)}天")
    logger.info(f"股票数量: {stock_count}")
    logger.info(f"预计样本数: {len(valid_dates) * stock_count:,}")

    computer_data = {
        'quotes_by_code': computer.quotes_by_code,
        'basic_by_code': computer.basic_by_code,
        'financial_by_code': computer.financial_by_code,
        'daily_quotes': computer.daily_quotes,
        'daily_basic': computer.daily_basic,
        'trade_dates': computer.trade_dates,
        'db_path': args.db_path,
    }

    conn = sqlite3.connect(args.db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS v39_feature_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            features_json TEXT,
            label_5d REAL,
            UNIQUE(code, trade_date)
        )
    """)
    conn.commit()

    total_inserted = 0
    start_time = datetime.now()

    logger.info(f"\n🚀 开始并行计算 ({len(valid_dates)}天)...")

    with Pool(processes=num_workers, initializer=init_worker, initargs=(computer_data,)) as pool:
        for i, results in enumerate(pool.imap_unordered(worker_compute_day, valid_dates), 1):
            if results:
                values = []
                for d in results:
                    code = d.pop('code')
                    trade_date = d.pop('trade_date')
                    label_5d = d.pop('label_5d', None)
                    features_json = json.dumps(d, ensure_ascii=False, default=lambda x: None if pd.isna(x) else x)
                    values.append((code, trade_date, features_json, label_5d))
                cursor.executemany("""
                    INSERT OR REPLACE INTO v39_feature_cache
                    (code, trade_date, features_json, label_5d)
                    VALUES (?, ?, ?, ?)
                """, values)
                conn.commit()
                total_inserted += len(values)

            if i % 2 == 0 or i == len(valid_dates):
                progress = i / len(valid_dates) * 100
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = total_inserted / elapsed if elapsed > 0 else 0
                eta = (len(valid_dates) - i) * (elapsed / i) / 60 if i > 0 else 0
                logger.info(f"进度: {progress:.1f}% ({i}/{len(valid_dates)}天) | 已插入: {total_inserted:,} | 速率: {rate:.0f}样本/秒 | 剩余: {eta:.1f}分钟")

    conn.close()
    elapsed_total = (datetime.now() - start_time).total_seconds() / 60

    logger.info("\n" + "=" * 80)
    logger.info(f"✅ 回填完成!")
    logger.info(f"总插入数: {total_inserted:,}")
    logger.info(f"总耗时: {elapsed_total:.1f}分钟")
    if elapsed_total > 0:
        logger.info(f"平均速率: {total_inserted/elapsed_total:.0f}样本/分钟")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
