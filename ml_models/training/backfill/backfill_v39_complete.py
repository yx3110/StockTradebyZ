#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.9特征完整回填脚本 (全部65个特征)
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
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CompleteV39FeatureComputer:
    """Complete V3.9 Feature Computer - 65 features"""

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
        """Preload all data into memory"""
        logger.info("=" * 60)
        logger.info("🚀 Preloading all data into memory...")
        conn = sqlite3.connect(self.db_path)

        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        buffer_start = (start_dt - timedelta(days=150)).strftime('%Y-%m-%d')

        logger.info(f"📈 Loading daily quotes ({buffer_start} ~ {end_date})...")
        self.daily_quotes = pd.read_sql("""
            SELECT s.code, dq.trade_date, dq.open, dq.high, dq.low, dq.close, dq.volume,
                   dq.price_change_pct, dq.is_limit_up
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE s.type = 'A股' AND dq.trade_date BETWEEN ? AND ?
            ORDER BY s.code, dq.trade_date
        """, conn, params=[buffer_start, end_date])
        logger.info(f"   ✓ Loaded {len(self.daily_quotes):,} quote records")

        logger.info(f"💰 Loading daily basic...")
        self.daily_basic = pd.read_sql("""
            SELECT s.code, db.trade_date, db.pe_ttm, db.pb, db.ps_ttm, db.turnover_rate, db.circ_mv
            FROM daily_basic db
            JOIN securities s ON db.security_id = s.id
            WHERE s.type = 'A股' AND db.trade_date BETWEEN ? AND ?
        """, conn, params=[buffer_start, end_date])
        logger.info(f"   ✓ Loaded {len(self.daily_basic):,} basic records")

        logger.info("📋 Loading financials...")
        self.financial = pd.read_sql("""
            SELECT s.code, fi.end_date, fi.ocf_to_profit, fi.roe, fi.grossprofit_margin,
                   fi.profit_to_gr, fi.debt_to_assets, fi.current_ratio, fi.ar_turn
            FROM financial_indicator fi
            JOIN securities s ON fi.security_id = s.id WHERE s.type = 'A股'
            ORDER BY s.code, fi.end_date DESC
        """, conn)
        logger.info(f"   ✓ Loaded {len(self.financial):,} financial records")
        conn.close()

        logger.info("🔧 Creating indexes...")
        self.quotes_by_code = {code: group.set_index('trade_date').sort_index()
                               for code, group in self.daily_quotes.groupby('code')}
        self.basic_by_code = {code: group.set_index('trade_date').sort_index()
                              for code, group in self.daily_basic.groupby('code')}
        self.financial_by_code = {code: group.head(4) for code, group in self.financial.groupby('code')}
        self.trade_dates = sorted(self.daily_quotes['trade_date'].unique())
        logger.info(f"   ✓ Indexed {len(self.quotes_by_code)} stocks")
        logger.info("=" * 60)

    def get_stock_data(self, code, end_date, days=100):
        if code not in self.quotes_by_code:
            return None
        df = self.quotes_by_code[code]
        df = df[df.index <= end_date].tail(days)
        return df.reset_index() if len(df) >= 30 else None

    def compute_all_features(self, code: str, trade_date: str):
        """Compute all 65 features for a stock"""
        df = self.get_stock_data(code, trade_date)
        if df is None:
            return None

        features = {}
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        close = df['close'].values.astype(float)
        volume = df['volume'].values.astype(float)
        n = len(close)

        # === Basic Technical Features ===
        # Momentum
        features['momentum_5d'] = (close[-1] / close[-5] - 1) if n >= 5 else 0
        features['momentum_10d'] = (close[-1] / close[-10] - 1) if n >= 10 else 0
        features['momentum_20d'] = (close[-1] / close[-20] - 1) if n >= 20 else 0

        # MA Ratios
        ma5, ma10, ma20 = np.mean(close[-5:]), np.mean(close[-10:]), np.mean(close[-20:])
        features['price_ma5_ratio'] = close[-1] / ma5 - 1 if ma5 > 0 else 0
        features['price_ma10_ratio'] = close[-1] / ma10 - 1 if ma10 > 0 else 0
        features['price_ma20_ratio'] = close[-1] / ma20 - 1 if ma20 > 0 else 0
        features['price_ma_ratio_squared'] = features['price_ma5_ratio'] ** 2

        # Volatility
        returns = np.diff(close) / close[:-1] if n > 1 else np.array([0])
        features['volatility_10d'] = np.std(returns[-10:]) if len(returns) >= 10 else 0
        features['volatility_20d'] = np.std(returns[-20:]) if len(returns) >= 20 else 0

        # Volatility asymmetry (upside vs downside)
        if len(returns) >= 20:
            up_returns = returns[-20:][returns[-20:] > 0]
            down_returns = returns[-20:][returns[-20:] < 0]
            up_vol = np.std(up_returns) if len(up_returns) > 1 else 0
            down_vol = np.std(down_returns) if len(down_returns) > 1 else 0
            features['volatility_asymmetry'] = (up_vol - down_vol) / (up_vol + down_vol + 1e-10)
        else:
            features['volatility_asymmetry'] = 0

        # RSI
        gains = np.maximum(np.diff(close), 0) if n > 1 else np.array([0])
        losses = np.maximum(-np.diff(close), 0) if n > 1 else np.array([0])
        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
        features['rsi_14'] = 100 - 100/(1 + avg_gain/(avg_loss + 1e-10))

        # BB Position
        bb_mid = ma20
        bb_std = np.std(close[-20:]) if n >= 20 else 1
        features['bb_position'] = (close[-1] - bb_mid) / (2 * bb_std + 1e-10)

        # Volume ratios
        vol_ma5 = np.mean(volume[-5:]) if n >= 5 else volume[-1]
        vol_ma20 = np.mean(volume[-20:]) if n >= 20 else volume[-1]
        features['volume_ratio_5d'] = volume[-1] / (vol_ma5 + 1e-10)
        features['volume_ratio_20d'] = volume[-1] / (vol_ma20 + 1e-10)
        features['volume_trend'] = vol_ma5 / (vol_ma20 + 1e-10)

        # ATR
        tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        features['atr_14'] = np.mean(tr[-14:]) / close[-1] if len(tr) >= 14 else 0

        # Momentum strength
        if n >= 20:
            pos_momentum = sum(1 for i in range(-20, 0) if close[i] > close[i-1])
            features['momentum_strength'] = pos_momentum / 20
        else:
            features['momentum_strength'] = 0.5

        # === Advanced Technical Features (24) ===
        tech = self._compute_advanced_technical(high, low, close, volume)
        features.update(tech)

        # === Fundamental Features ===
        if code in self.basic_by_code and trade_date in self.basic_by_code[code].index:
            basic = self.basic_by_code[code].loc[trade_date]
            features['pe_ttm'] = basic.get('pe_ttm', np.nan) or np.nan
            features['pb'] = basic.get('pb', np.nan) or np.nan
            features['turnover_rate'] = basic.get('turnover_rate', np.nan) or np.nan
            circ_mv = basic.get('circ_mv', 0) or 0
            features['market_cap'] = np.log1p(circ_mv / 10000) if circ_mv > 0 else 0
        else:
            features['pe_ttm'] = np.nan
            features['pb'] = np.nan
            features['turnover_rate'] = np.nan
            features['market_cap'] = 0

        fund = self._compute_fundamental(code, trade_date)
        features.update(fund)

        # === Market Features ===
        mkt = self._compute_market(trade_date, code)
        features.update(mkt)

        # === Active MV Features ===
        amv = self._compute_active_mv(code, trade_date)
        features.update(amv)

        # === Interaction Features ===
        roe = features.get('roe_change_rate', 0) or 0
        features['roe_momentum_interaction'] = roe * features.get('momentum_5d', 0)

        # Relative strength to market
        if trade_date in self.market_cache:
            mkt_data = self.market_cache[trade_date]
            mkt_return = mkt_data.get('market_return', 0)
            features['relative_strength_to_market'] = features['momentum_5d'] - mkt_return
        else:
            features['relative_strength_to_market'] = 0

        return features

    def _compute_advanced_technical(self, high, low, close, volume):
        """Compute 24 advanced technical features"""
        features = {}
        n = len(close)
        if n < 60:
            return {k: np.nan for k in ['adx_14', 'aroon_up_25', 'aroon_down_25', 'aroon_oscillator_25',
                                        'ichimoku_conversion', 'ichimoku_base', 'ichimoku_span_a', 'ichimoku_span_b',
                                        'supertrend_10_3', 'supertrend_signal', 'williams_r_14', 'smi_14', 'smi_signal_3',
                                        'tsi_25_13', 'tsi_signal_7', 'ad_line', 'ad_line_change_5', 'cmf_20',
                                        'vwap_deviation', 'large_order_net_inflow', 'bb_width_20', 'kc_width_20',
                                        'atr_percent_14', 'historical_volatility_percentile_60']}
        try:
            # ADX
            tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
            plus_dm = np.where(np.maximum(high[1:] - high[:-1], 0) > np.maximum(low[:-1] - low[1:], 0),
                              np.maximum(high[1:] - high[:-1], 0), 0)
            minus_dm = np.where(np.maximum(low[:-1] - low[1:], 0) > np.maximum(high[1:] - high[:-1], 0),
                               np.maximum(low[:-1] - low[1:], 0), 0)
            atr = pd.Series(tr).ewm(span=14).mean().values
            plus_di = 100 * pd.Series(plus_dm).ewm(span=14).mean().values / (atr + 1e-10)
            minus_di = 100 * pd.Series(minus_dm).ewm(span=14).mean().values / (atr + 1e-10)
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            features['adx_14'] = pd.Series(dx).ewm(span=14).mean().iloc[-1]

            # Aroon
            period = 25
            features['aroon_up_25'] = ((period - (period - np.argmax(high[-period:]))) / period) * 100
            features['aroon_down_25'] = ((period - (period - np.argmin(low[-period:]))) / period) * 100
            features['aroon_oscillator_25'] = features['aroon_up_25'] - features['aroon_down_25']

            # Ichimoku
            h, l = pd.Series(high), pd.Series(low)
            conv = (h.rolling(9).max() + l.rolling(9).min()) / 2
            base = (h.rolling(26).max() + l.rolling(26).min()) / 2
            span_a = ((conv + base) / 2).shift(26)
            span_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
            p = close[-1]
            features['ichimoku_conversion'] = (p - conv.iloc[-1]) / p if pd.notna(conv.iloc[-1]) else 0
            features['ichimoku_base'] = (p - base.iloc[-1]) / p if pd.notna(base.iloc[-1]) else 0
            features['ichimoku_span_a'] = (p - span_a.iloc[-1]) / p if pd.notna(span_a.iloc[-1]) else 0
            features['ichimoku_span_b'] = (p - span_b.iloc[-1]) / p if pd.notna(span_b.iloc[-1]) else 0

            # SuperTrend
            atr_st = pd.Series(tr).rolling(10).mean().values
            hl = (high[1:] + low[1:]) / 2
            upper, lower = hl + 3 * atr_st, hl - 3 * atr_st
            st, trend = np.zeros(len(close)-1), np.ones(len(close)-1)
            for i in range(1, len(st)):
                if close[i] > upper[i-1]: trend[i], st[i] = 1, lower[i]
                elif close[i] < lower[i-1]: trend[i], st[i] = -1, upper[i]
                else: trend[i] = trend[i-1]; st[i] = max(lower[i], st[i-1]) if trend[i] == 1 else min(upper[i], st[i-1])
            features['supertrend_10_3'] = (close[-1] - st[-1]) / close[-1]
            features['supertrend_signal'] = float(trend[-1])

            # Williams %R
            hh, ll = pd.Series(high).rolling(14).max(), pd.Series(low).rolling(14).min()
            features['williams_r_14'] = -100 * (hh.iloc[-1] - close[-1]) / (hh.iloc[-1] - ll.iloc[-1] + 1e-10)

            # SMI
            c = pd.Series(close)
            diff = c - (pd.Series(high).rolling(14).max() + pd.Series(low).rolling(14).min()) / 2
            ds = diff.ewm(span=3).mean().ewm(span=3).mean()
            rs = (pd.Series(high).rolling(14).max() - pd.Series(low).rolling(14).min()).ewm(span=3).mean().ewm(span=3).mean()
            smi = 100 * ds / (rs / 2 + 1e-10)
            features['smi_14'] = smi.iloc[-1]
            features['smi_signal_3'] = smi.ewm(span=3).mean().iloc[-1]

            # TSI
            pc = c.diff()
            ds2 = pc.ewm(span=25).mean().ewm(span=13).mean()
            ds2_abs = pc.abs().ewm(span=25).mean().ewm(span=13).mean()
            tsi = 100 * ds2 / (ds2_abs + 1e-10)
            features['tsi_25_13'] = tsi.iloc[-1]
            features['tsi_signal_7'] = tsi.ewm(span=7).mean().iloc[-1]

            # A/D Line
            mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
            ad = np.cumsum(mfm * volume)
            features['ad_line'] = ad[-1]
            features['ad_line_change_5'] = (ad[-1] - ad[-6]) / (abs(ad[-6]) + 1e-10) if n >= 6 else 0

            # CMF
            features['cmf_20'] = pd.Series(mfm * volume).rolling(20).sum().iloc[-1] / (pd.Series(volume).rolling(20).sum().iloc[-1] + 1e-10)

            # VWAP deviation
            vwap = (close * volume).sum() / (volume.sum() + 1e-10)
            features['vwap_deviation'] = (close[-1] - vwap) / vwap

            # Large order flow
            period = min(10, n - 1)
            pc_arr = close[-period:] - close[-period-1:-1]
            vol_avg = np.mean(volume[-20:]) if n >= 20 else np.mean(volume)
            large_vol = volume[-period:] > vol_avg * 1.5
            features['large_order_net_inflow'] = (np.sum((pc_arr > 0) & large_vol) - np.sum((pc_arr < 0) & large_vol)) / period

            # BB Width
            sma = pd.Series(close).rolling(20).mean()
            std = pd.Series(close).rolling(20).std()
            features['bb_width_20'] = (4 * std / (sma + 1e-10)).iloc[-1]

            # KC Width
            ema = pd.Series(close).ewm(span=20).mean()
            features['kc_width_20'] = (4 * pd.Series(tr).rolling(20).mean().iloc[-1]) / (ema.iloc[-1] + 1e-10)

            # ATR%
            features['atr_percent_14'] = pd.Series(tr).rolling(14).mean().iloc[-1] / close[-1]

            # Volatility percentile
            vol = pd.Series(close).pct_change().rolling(20).std() * np.sqrt(252)
            features['historical_volatility_percentile_60'] = (vol.iloc[-60:] < vol.iloc[-1]).sum() / 60 if len(vol) >= 60 else 0.5

        except Exception as e:
            pass
        return features

    def _compute_fundamental(self, code, trade_date):
        features = {k: np.nan for k in ['operating_cashflow_to_netprofit', 'roe_change_rate', 'gross_margin_trend',
                                        'netprofit_growth_stability', 'pe_industry_percentile', 'pb_industry_percentile',
                                        'ps_industry_percentile', 'debt_to_asset_ratio', 'current_ratio', 'receivables_turnover']}
        try:
            if code in self.financial_by_code:
                fin = self.financial_by_code[code]
                if len(fin) > 0:
                    features['operating_cashflow_to_netprofit'] = fin.iloc[0].get('ocf_to_profit', np.nan)
                    features['debt_to_asset_ratio'] = fin.iloc[0].get('debt_to_assets', np.nan)
                    features['current_ratio'] = fin.iloc[0].get('current_ratio', np.nan)
                    features['receivables_turnover'] = fin.iloc[0].get('ar_turn', np.nan)
                    if len(fin) >= 2:
                        r0, r1 = fin.iloc[0]['roe'], fin.iloc[1]['roe']
                        if pd.notna(r0) and pd.notna(r1) and r1 != 0:
                            features['roe_change_rate'] = (r0 - r1) / abs(r1)
                    if len(fin) >= 3:
                        m = fin['grossprofit_margin'].dropna().values[:3]
                        if len(m) >= 3: features['gross_margin_trend'] = np.polyfit(np.arange(len(m)), m, 1)[0]
                        pg = fin['profit_to_gr'].dropna().values[:3]
                        if len(pg) >= 3:
                            cv = np.std(pg) / (abs(np.mean(pg)) + 1e-10)
                            features['netprofit_growth_stability'] = 1 / (cv + 1)
            if code in self.basic_by_code and trade_date in self.basic_by_code[code].index:
                sb = self.basic_by_code[code].loc[trade_date]
                mb = self.daily_basic[self.daily_basic['trade_date'] == trade_date]
                if len(mb) > 10:
                    for col, feat in [('pe_ttm', 'pe_industry_percentile'), ('pb', 'pb_industry_percentile'), ('ps_ttm', 'ps_industry_percentile')]:
                        v = sb.get(col)
                        if pd.notna(v):
                            mkt = mb[col].dropna()
                            if len(mkt) > 0: features[feat] = (mkt < v).sum() / len(mkt)
        except Exception:
            pass
        return features

    def _compute_market(self, trade_date, code=None):
        if trade_date not in self.market_cache:
            dq = self.daily_quotes[self.daily_quotes['trade_date'] == trade_date]
            mkt = {'advance_decline_ratio': 1.0, 'limit_up_count': 0.0, 'northbound_net_inflow': 0.0,
                   'margin_balance_change': 0.0, 'concept_heat_index': 0.0, 'market_return': 0.0}
            if len(dq) > 0:
                adv, dec = (dq['price_change_pct'] > 0).sum(), (dq['price_change_pct'] < 0).sum()
                mkt['advance_decline_ratio'] = adv / (dec + 1) if dec > 0 else float(adv)
                mkt['market_return'] = dq['price_change_pct'].mean() if len(dq) > 0 else 0
                if 'is_limit_up' in dq.columns:
                    lu = dq['is_limit_up'].sum()
                    mkt['limit_up_count'] = min(lu / 100.0, 1.0)
                    mkt['concept_heat_index'] = lu / len(dq) if len(dq) > 0 else 0
            self.market_cache[trade_date] = mkt
        features = self.market_cache[trade_date].copy()
        if code:
            features.update(self._compute_stock_market(code, trade_date))
        return features

    def _compute_stock_market(self, code, trade_date):
        features = {'sector_strength_rank': 0.5, 'industry_fund_flow_rank': 0.5, 'market_attention_score': 1.0}
        try:
            dq = self.daily_quotes[self.daily_quotes['trade_date'] == trade_date]
            if code in self.quotes_by_code and trade_date in self.quotes_by_code[code].index:
                sd = self.quotes_by_code[code].loc[trade_date]
                sp = sd['price_change_pct']
                if len(dq) > 0 and pd.notna(sp):
                    features['sector_strength_rank'] = (dq['price_change_pct'] < sp).sum() / len(dq)
                hist = self.quotes_by_code[code]
                hist = hist[hist.index < trade_date].tail(5)
                if len(hist) > 0:
                    avg_vol = hist['volume'].mean()
                    if avg_vol > 0: features['industry_fund_flow_rank'] = min(float(sd['volume']) / float(avg_vol) / 2, 1.0)
            if code in self.basic_by_code and trade_date in self.basic_by_code[code].index:
                st = self.basic_by_code[code].loc[trade_date].get('turnover_rate', 1)
                db = self.daily_basic[self.daily_basic['trade_date'] == trade_date]
                if len(db) > 0 and pd.notna(st):
                    at = db['turnover_rate'].mean()
                    if pd.notna(at) and at > 0: features['market_attention_score'] = float(st) / float(at)
        except Exception:
            pass
        return features

    def _compute_active_mv(self, code, trade_date):
        features = {'market_active_mv_ratio': 0.0, 'market_active_mv_zscore': 0.0, 'market_active_mv_trend': 0.0,
                    'stock_active_mv_rank': 0.5, 'stock_relative_liquidity': 0.5, 'market_cap_quality_score': 0.5}
        try:
            mb = self.daily_basic[self.daily_basic['trade_date'] == trade_date]
            if len(mb) > 0:
                valid = mb.dropna(subset=['circ_mv', 'turnover_rate'])
                if len(valid) > 0:
                    amv = valid['circ_mv'] * valid['turnover_rate'] / 100
                    tcmv = valid['circ_mv'].sum()
                    if tcmv > 0: features['market_active_mv_ratio'] = amv.sum() / tcmv
                    if code in self.basic_by_code and trade_date in self.basic_by_code[code].index:
                        sb = self.basic_by_code[code].loc[trade_date]
                        scmv, str_ = sb.get('circ_mv', 0), sb.get('turnover_rate', 0)
                        if pd.notna(scmv) and pd.notna(str_) and scmv > 0:
                            samv = scmv * str_ / 100
                            features['stock_active_mv_rank'] = (amv <= samv).sum() / len(amv)
                            mamv = amv.mean()
                            if mamv > 0: features['stock_relative_liquidity'] = np.tanh(samv / mamv / 2)
                            cmvy = scmv / 10000
                            if cmvy > 0: features['market_cap_quality_score'] = 1 / (1 + np.exp(-(np.log(cmvy) - np.log(50)) / 0.8))
        except Exception:
            pass
        return features

    def calculate_label(self, code, trade_date, lookahead=5):
        if code not in self.quotes_by_code: return None
        df = self.quotes_by_code[code]
        if trade_date not in df.index: return None
        future = df[df.index > trade_date].head(lookahead)
        if len(future) < lookahead: return None
        cc = df.loc[trade_date, 'close']
        fc = future.iloc[-1]['close']
        return (fc - cc) / cc * 100 if pd.notna(cc) and cc > 0 else None


_global_computer = None

def init_worker(computer_data):
    global _global_computer
    _global_computer = CompleteV39FeatureComputer.__new__(CompleteV39FeatureComputer)
    _global_computer.__dict__.update(computer_data)
    _global_computer.market_cache = {}

def worker_compute_day(trade_date):
    global _global_computer
    results = []
    for code in list(_global_computer.quotes_by_code.keys()):
        try:
            features = _global_computer.compute_all_features(code, trade_date)
            if features is None: continue
            label = _global_computer.calculate_label(code, trade_date, 5)
            if label is None: continue
            features['code'] = code
            features['trade_date'] = trade_date
            features['label_5d'] = label
            results.append(features)
        except Exception:
            continue
    return results


def main():
    parser = argparse.ArgumentParser(description='V3.9 Complete Feature Backfill (65 features)')
    parser.add_argument('--start-date', default='2025-09-01')
    parser.add_argument('--end-date', default='2025-10-28')
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--db-path', default=str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db'))
    args = parser.parse_args()

    num_workers = args.num_workers or min(cpu_count(), 8)
    logger.info(f"🚀 V3.9 Complete Feature Backfill (65 features)")
    logger.info(f"Date range: {args.start_date} ~ {args.end_date}, Workers: {num_workers}")

    computer = CompleteV39FeatureComputer(args.db_path)
    computer.preload_all_data(args.start_date, args.end_date)

    valid_dates = [d for d in computer.trade_dates if args.start_date <= d <= args.end_date][:-5]
    logger.info(f"Trading days: {len(valid_dates)}, Stocks: {len(computer.quotes_by_code)}")

    computer_data = {k: v for k, v in computer.__dict__.items() if k != 'market_cache'}

    conn = sqlite3.connect(args.db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS v39_feature_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT, code TEXT NOT NULL, trade_date TEXT NOT NULL,
            features_json TEXT, label_5d REAL, UNIQUE(code, trade_date))
    """)
    conn.commit()

    total_inserted = 0
    start_time = datetime.now()

    with Pool(processes=num_workers, initializer=init_worker, initargs=(computer_data,)) as pool:
        for i, results in enumerate(pool.imap_unordered(worker_compute_day, valid_dates), 1):
            if results:
                values = [(d.pop('code'), d.pop('trade_date'), json.dumps({k: (None if pd.isna(v) else v) for k, v in d.items() if k != 'label_5d'}, ensure_ascii=False), d.get('label_5d')) for d in results]
                cursor.executemany("INSERT OR REPLACE INTO v39_feature_cache (code, trade_date, features_json, label_5d) VALUES (?, ?, ?, ?)", values)
                conn.commit()
                total_inserted += len(values)
            if i % 2 == 0 or i == len(valid_dates):
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.info(f"Progress: {i}/{len(valid_dates)} days | Inserted: {total_inserted:,} | Rate: {total_inserted/elapsed:.0f}/s | ETA: {(len(valid_dates)-i)*(elapsed/i)/60:.1f}min")

    conn.close()
    logger.info(f"✅ Done! Total: {total_inserted:,} in {(datetime.now()-start_time).total_seconds()/60:.1f} min")


if __name__ == "__main__":
    main()
