#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V4.0 Cross-Sectional Alpha Model 特征缓存更新器

核心理念: 学习"哪只股票能跑赢行业/市场"，而非"大盘涨不涨"
- ~55个特征，其中31个为行业内cross-sectional排名 (0-1)
- 标签为超额收益 (个股收益 - 沪深300收益)
- 技术指标从technical_indicators表直接读取

基于 v39_feature_cache_updater.py 的批量加载架构 fork。
"""

import os
import sys
import json
import sqlite3
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import bisect
import time

from fetch_data.label_utils import compute_aligned_labels
from fetch_data.shared_data_loader import batch_load_stock_data, load_sw_industry_mapping

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class V40FeatureCacheUpdater:
    """V4.0 Cross-Sectional Alpha Model 特征缓存更新器"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')
        self.db_path = db_path

        # 行业映射缓存
        self._sw_industry_mapping = None   # code -> l1_name
        self._sw_l1_label_encoding = None  # l1_name -> int
        self._sw_l1_codes = None           # l1_name -> l1_code

    # ================================================================
    # 行业映射
    # ================================================================

    def _load_sw_industry_mapping(self):
        """加载申万行业映射 (委托 shared_data_loader)"""
        if self._sw_industry_mapping is not None:
            return

        mapping, encoding, codes = load_sw_industry_mapping(db_path=self.db_path)
        self._sw_industry_mapping = mapping
        self._sw_l1_label_encoding = encoding
        self._sw_l1_codes = codes

    # ================================================================
    # 数据加载
    # ================================================================

    def _batch_load_stock_data(self, date: str, lookback: int = 60) -> Dict[str, pd.DataFrame]:
        """批量加载所有A股的历史行情数据 (委托 shared_data_loader)"""
        return batch_load_stock_data(
            date=date,
            lookback=lookback,
            db_path=self.db_path,
            stock_types=('A股',),
        )

    def _batch_load_technical_indicators(self, date: str) -> Dict[str, Dict]:
        """
        批量加载当天所有A股的技术指标 (从 technical_indicators 表)

        Returns:
            {code: {kdj_k, kdj_j, macd_dif, macd_dea, macd_macd, rsi6, rsi12,
                    boll_upper, boll_middle, boll_lower, cci_14, atr_14,
                    squeeze_state, squeeze_days, momentum_direction,
                    zhixing_short_trend}}
        """
        conn = sqlite3.connect(self.db_path)
        query = """
        SELECT s.code,
               ti.kdj_k, ti.kdj_j, ti.kdj_d,
               ti.macd_dif, ti.macd_dea, ti.macd_macd,
               ti.rsi6, ti.rsi12,
               ti.boll_upper, ti.boll_middle, ti.boll_lower,
               ti.cci_14, ti.atr_14,
               ti.squeeze_state, ti.squeeze_days,
               ti.momentum_direction,
               ti.zhixing_short_trend
        FROM technical_indicators ti
        JOIN securities s ON ti.security_id = s.id
        WHERE s.type = 'A股' AND ti.trade_date = ?
        """
        df = pd.read_sql_query(query, conn, params=(date,))
        conn.close()

        result = {}
        for _, row in df.iterrows():
            result[row['code']] = row.to_dict()
        result.pop('', None)  # 安全删除空键

        logger.info(f"技术指标加载: {len(result)} 只股票 ({date})")
        return result

    def _batch_load_daily_basic(self, date: str) -> Dict[str, Dict]:
        """
        批量加载当天的daily_basic数据 (换手率/市值/PE/PB/PS)

        Returns:
            {code: {turnover_rate, total_mv, pe_ttm, pb, ps_ttm, circ_mv}}
        """
        conn = sqlite3.connect(self.db_path)
        query = """
        SELECT s.code, db.turnover_rate, db.total_mv, db.pe_ttm, db.pb, db.ps_ttm, db.circ_mv
        FROM daily_basic db
        JOIN securities s ON db.security_id = s.id
        WHERE s.type = 'A股'
        AND db.trade_date = (SELECT MAX(trade_date) FROM daily_basic WHERE trade_date <= ?)
        """
        df = pd.read_sql_query(query, conn, params=(date,))
        conn.close()

        result = {}
        for _, row in df.iterrows():
            result[row['code']] = row.to_dict()

        logger.info(f"基本面加载: {len(result)} 只股票")
        return result

    def _load_hs300_closes(self, start_date: str, end_date: str) -> pd.DataFrame:
        """加载沪深300收盘价 (用于超额收益标签计算)"""
        conn = sqlite3.connect(self.db_path)
        query = """
        SELECT q.trade_date, q.close
        FROM daily_quotes q
        JOIN securities s ON q.security_id = s.id
        WHERE (s.code = '000300.SH' OR s.name = '沪深300')
        AND q.trade_date >= ? AND q.trade_date <= ?
        ORDER BY q.trade_date
        """
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()
        return df

    def _load_northbound_flow(self, date: str) -> float:
        """加载北向资金5日z-score"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='hsgt_daily'")
        if not cursor.fetchone():
            conn.close()
            return 0.0

        cursor.execute("""
            SELECT north_money FROM hsgt_daily
            WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT 60
        """, (date,))
        rows = cursor.fetchall()
        conn.close()

        if len(rows) < 5:
            return 0.0

        # 5日累计 (万元 -> 亿元)
        flow_5d = sum(float(r[0]) for r in rows[:5] if r[0] is not None) / 10000.0
        # 60日均值和标准差
        all_flows = [float(r[0]) / 10000.0 for r in rows if r[0] is not None]
        if len(all_flows) >= 20:
            mean_60d = np.mean(all_flows)
            std_60d = np.std(all_flows)
            if std_60d > 0:
                return (flow_5d - mean_60d) / std_60d
        return 0.0

    # ================================================================
    # Cross-Sectional 特征计算核心
    # ================================================================

    def _compute_cross_sectional_features(self, date: str,
                                           stock_data_map: Dict[str, pd.DataFrame],
                                           tech_indicators: Dict[str, Dict],
                                           daily_basic: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        计算所有股票的cross-sectional特征

        对当天全部A股统一计算：
        1. 按申万一级行业分组
        2. 对每个指标计算行业内百分位排名 (0-1)
        3. 计算行业中位数超额值

        Returns:
            {code: {feature_name: value, ...}}
        """
        self._load_sw_industry_mapping()

        # ========== 收集原始指标 ==========
        raw_data = {}  # code -> {metric: value}

        for code, stock_df in stock_data_map.items():
            if stock_df is None or len(stock_df) < 20:
                continue

            # 验证最后一行是当天数据
            last_date = str(stock_df.iloc[-1]['trade_date'])[:10]
            if last_date != date:
                continue

            closes = stock_df['close'].values
            volumes = stock_df['volume'].values
            highs = stock_df['high'].values
            lows = stock_df['low'].values

            entry = {}
            entry['l1_name'] = self._sw_industry_mapping.get(code)

            # --- A. 动量收益率 ---
            if len(closes) >= 5 and closes[-5] > 0:
                entry['return_5d'] = closes[-1] / closes[-5] - 1
            if len(closes) >= 10 and closes[-10] > 0:
                entry['return_10d'] = closes[-1] / closes[-10] - 1
            if len(closes) >= 20 and closes[-20] > 0:
                entry['return_20d'] = closes[-1] / closes[-20] - 1

            # --- B. 成交量/流动性 ---
            if len(volumes) >= 20:
                avg_vol_5 = np.mean(volumes[-5:])
                avg_vol_20 = np.mean(volumes[-20:])
                entry['volume_ratio'] = volumes[-1] / avg_vol_5 if avg_vol_5 > 0 else 1.0

                basic = daily_basic.get(code, {})
                turnover = basic.get('turnover_rate')
                if turnover is not None and turnover > 0:
                    entry['turnover_rate'] = turnover

                # 换手率变化
                # (近5日均换手率) / (近20日均换手率) — 用成交量代替
                entry['turnover_change'] = avg_vol_5 / avg_vol_20 if avg_vol_20 > 0 else 1.0

                # 量价相关性 10日
                if len(closes) >= 11:
                    vol_changes = np.diff(volumes[-11:]) / np.maximum(volumes[-11:-1], 1)
                    price_changes = np.diff(closes[-11:]) / closes[-11:-1]
                    if len(vol_changes) >= 2:
                        corr = np.corrcoef(vol_changes, price_changes)[0, 1]
                        entry['volume_price_corr_10d'] = corr if not np.isnan(corr) else 0.0

                # Amihud非流动性
                if len(closes) >= 21:
                    log_returns = np.abs(np.diff(np.log(closes[-21:])))
                    vol_slice = volumes[-20:]
                    vol_nonzero = np.maximum(vol_slice, 1)
                    entry['amihud_illiquidity'] = float(np.mean(log_returns / vol_nonzero))

            # --- C. 波动率/风险 ---
            if len(closes) >= 11:
                log_returns = np.diff(np.log(closes[-11:]))
                entry['volatility_10d'] = float(np.std(log_returns) * np.sqrt(252))
            if len(closes) >= 21:
                log_returns = np.diff(np.log(closes[-21:]))
                entry['volatility_20d'] = float(np.std(log_returns) * np.sqrt(252))

                # 最大回撤
                rolling_max = np.maximum.accumulate(closes[-20:])
                drawdowns = closes[-20:] / rolling_max - 1
                entry['max_drawdown_20d'] = float(np.min(drawdowns))

                # 涨跌不对称
                daily_rets = np.diff(closes[-21:]) / closes[-21:-1]
                pos_rets = daily_rets[daily_rets > 0]
                neg_rets = daily_rets[daily_rets < 0]
                if len(neg_rets) > 0 and len(pos_rets) > 0:
                    entry['updown_asymmetry_10d'] = float(
                        np.mean(pos_rets) / np.mean(np.abs(neg_rets)))
                else:
                    entry['updown_asymmetry_10d'] = 1.0

            # --- D. 市值 ---
            basic = daily_basic.get(code, {})
            total_mv = basic.get('total_mv')
            if total_mv is not None and total_mv > 0:
                entry['total_mv'] = total_mv

            # --- E. 估值 ---
            pe_ttm = basic.get('pe_ttm')
            pb = basic.get('pb')
            ps_ttm = basic.get('ps_ttm')
            if pe_ttm is not None and pe_ttm > 0:
                entry['pe_ttm'] = pe_ttm
            if pb is not None and pb > 0:
                entry['pb'] = pb
            if ps_ttm is not None and ps_ttm > 0:
                entry['ps_ttm'] = ps_ttm

            # --- F. 技术指标 (从 technical_indicators 表) ---
            ti = tech_indicators.get(code, {})
            for key in ['kdj_j', 'kdj_k', 'macd_macd', 'rsi6', 'rsi12', 'cci_14']:
                val = ti.get(key)
                if val is not None:
                    entry[key] = float(val)

            macd_dif = ti.get('macd_dif')
            macd_dea = ti.get('macd_dea')
            if macd_dif is not None and macd_dea is not None:
                entry['macd_cross_signal'] = 1.0 if macd_dif > macd_dea else -1.0

            boll_upper = ti.get('boll_upper')
            boll_lower = ti.get('boll_lower')
            boll_middle = ti.get('boll_middle')
            if boll_upper is not None and boll_lower is not None and boll_upper > boll_lower:
                entry['boll_position'] = float(
                    (closes[-1] - boll_lower) / (boll_upper - boll_lower))

            atr_14 = ti.get('atr_14')
            if atr_14 is not None and closes[-1] > 0:
                entry['atr_14_pct'] = float(atr_14 / closes[-1])

            # 直接使用的指标
            for key in ['squeeze_state', 'squeeze_days', 'momentum_direction', 'zhixing_short_trend']:
                val = ti.get(key)
                if val is not None:
                    entry[key] = float(val)

            # --- I. 基础特征 (非排名) ---
            # RSI (直接值)
            rsi12_val = ti.get('rsi12')
            entry['rsi_14'] = float(rsi12_val) if rsi12_val is not None else 50.0

            # 20日高低位置
            if len(closes) >= 20:
                h20 = np.max(highs[-20:])
                l20 = np.min(lows[-20:])
                entry['price_position_20d'] = float(
                    (closes[-1] - l20) / (h20 - l20)) if h20 > l20 else 0.5

            # 均线交叉
            if len(closes) >= 20:
                ma5 = np.mean(closes[-5:])
                ma10 = np.mean(closes[-10:])
                ma20 = np.mean(closes[-20:])
                entry['ma_cross'] = 1 if ma5 > ma10 > ma20 else (-1 if ma5 < ma10 < ma20 else 0)

            # 涨跌幅
            if len(closes) >= 6:
                pct_changes = np.diff(closes[-6:]) / closes[-6:-1]
                entry['avg_pct_change_5d'] = float(np.mean(pct_changes))
                entry['min_pct_change_5d'] = float(np.min(pct_changes))

            raw_data[code] = entry

        # ========== Cross-Sectional 排名计算 ==========
        logger.info(f"计算cross-sectional排名: {len(raw_data)} 只股票")

        # 转为DataFrame便于分组排名
        raw_df = pd.DataFrame.from_dict(raw_data, orient='index')
        raw_df.index.name = 'code'

        # 行业内排名指标
        industry_rank_metrics = {
            'return_5d': 'xs_return_5d_rank',
            'return_10d': 'xs_return_10d_rank',
            'return_20d': 'xs_return_20d_rank',
            'volume_ratio': 'xs_volume_ratio_rank',
            'turnover_rate': 'xs_turnover_rank',
            'kdj_j': 'xs_kdj_j_rank',
            'kdj_k': 'xs_kdj_k_rank',
            'macd_macd': 'xs_macd_hist_rank',
            'rsi6': 'xs_rsi6_rank',
            'rsi12': 'xs_rsi12_rank',
            'boll_position': 'xs_boll_position_rank',
            'cci_14': 'xs_cci14_rank',
            'atr_14_pct': 'xs_atr14_pct_rank',
            'pe_ttm': 'xs_pe_rank',
            'pb': 'xs_pb_rank',
            'ps_ttm': 'xs_ps_rank',
            'total_mv': 'xs_market_cap_rank',
        }

        # 全市场排名指标 (不分行业)
        market_rank_metrics = {
            'volatility_10d': 'xs_volatility_10d_rank',
            'volatility_20d': 'xs_volatility_20d_rank',
        }

        # 按行业分组计算排名
        if 'l1_name' in raw_df.columns:
            for raw_col, rank_col in industry_rank_metrics.items():
                if raw_col in raw_df.columns:
                    raw_df[rank_col] = raw_df.groupby('l1_name')[raw_col].rank(pct=True)

        # 全市场排名
        for raw_col, rank_col in market_rank_metrics.items():
            if raw_col in raw_df.columns:
                raw_df[rank_col] = raw_df[raw_col].rank(pct=True)

        # ========== 行业超额收益 ==========
        if 'l1_name' in raw_df.columns:
            for period in ['5d', '10d', '20d']:
                ret_col = f'return_{period}'
                excess_col = f'xs_return_{period}_excess'
                if ret_col in raw_df.columns:
                    industry_median = raw_df.groupby('l1_name')[ret_col].transform('median')
                    raw_df[excess_col] = raw_df[ret_col] - industry_median

        # ========== 行业上下文特征 ==========
        if 'l1_name' in raw_df.columns and len(raw_df) > 0:
            # 行业涨跌比 (breadth)
            if 'return_5d' in raw_df.columns:
                # 使用当日涨跌 (price_change_pct) 来计算breadth
                pass

            # 行业KDJ均值
            if 'kdj_j' in raw_df.columns:
                raw_df['industry_kdj_avg'] = raw_df.groupby('l1_name')['kdj_j'].transform('mean')

            # 行业MACD多头比例
            if 'macd_cross_signal' in raw_df.columns:
                raw_df['industry_macd_bullish_pct'] = raw_df.groupby('l1_name')[
                    'macd_cross_signal'].transform(lambda x: (x > 0).mean())

            # 行业集中度 (HHI)
            if 'total_mv' in raw_df.columns:
                def hhi(x):
                    s = x.sum()
                    if s > 0:
                        shares = x / s
                        return (shares ** 2).sum()
                    return 0.03
                raw_df['industry_concentration'] = raw_df.groupby('l1_name')[
                    'total_mv'].transform(hhi)

            # 行业动量排名 (跨行业)
            if 'return_5d' in raw_df.columns:
                industry_ret = raw_df.groupby('l1_name')['return_5d'].mean()
                industry_ret_rank = industry_ret.rank(pct=True)
                raw_df['industry_momentum_rank'] = raw_df['l1_name'].map(industry_ret_rank)

                # 行业轮动信号
                if 'return_20d' in raw_df.columns:
                    industry_ret_20d = raw_df.groupby('l1_name')['return_20d'].mean()
                    rotation = industry_ret - industry_ret_20d
                    raw_df['industry_rotation_signal'] = raw_df['l1_name'].map(rotation)

        # ========== 组装最终特征 ==========
        result = {}
        for code in raw_data.keys():
            if code not in raw_df.index:
                continue

            row = raw_df.loc[code]
            features = {}

            # A. Cross-Sectional 动量排名 (6)
            for col in ['xs_return_5d_rank', 'xs_return_10d_rank', 'xs_return_20d_rank',
                        'xs_return_5d_excess', 'xs_return_10d_excess', 'xs_return_20d_excess']:
                features[col] = _safe_float(row.get(col), 0.5 if 'rank' in col else 0.0)

            # B. Cross-Sectional 成交量/流动性 (5)
            features['xs_volume_ratio_rank'] = _safe_float(row.get('xs_volume_ratio_rank'), 0.5)
            features['xs_turnover_rank'] = _safe_float(row.get('xs_turnover_rank'), 0.5)
            features['xs_turnover_change'] = _safe_float(row.get('turnover_change'), 1.0)
            features['volume_price_corr_10d'] = _safe_float(row.get('volume_price_corr_10d'), 0.0)
            features['amihud_illiquidity'] = _safe_float(row.get('amihud_illiquidity'), 0.0)

            # C. 技术指标 Cross-Sectional (14)
            for col in ['xs_kdj_j_rank', 'xs_kdj_k_rank', 'xs_macd_hist_rank',
                        'xs_rsi6_rank', 'xs_rsi12_rank', 'xs_boll_position_rank',
                        'xs_cci14_rank', 'xs_atr14_pct_rank']:
                features[col] = _safe_float(row.get(col), 0.5)
            features['macd_cross_signal'] = _safe_float(row.get('macd_cross_signal'), 0.0)
            features['boll_position'] = _safe_float(row.get('boll_position'), 0.5)
            features['squeeze_state'] = _safe_float(row.get('squeeze_state'), 0.0)
            features['squeeze_days'] = _safe_float(row.get('squeeze_days'), 0.0)
            features['momentum_direction'] = _safe_float(row.get('momentum_direction'), 0.0)
            features['zhixing_short_trend'] = _safe_float(row.get('zhixing_short_trend'), 0.0)

            # D. Cross-Sectional 波动率/风险 (5)
            features['xs_volatility_10d_rank'] = _safe_float(row.get('xs_volatility_10d_rank'), 0.5)
            features['xs_volatility_20d_rank'] = _safe_float(row.get('xs_volatility_20d_rank'), 0.5)
            features['max_drawdown_20d'] = _safe_float(row.get('max_drawdown_20d'), 0.0)
            features['updown_asymmetry_10d'] = _safe_float(row.get('updown_asymmetry_10d'), 1.0)
            features['xs_market_cap_rank'] = _safe_float(row.get('xs_market_cap_rank'), 0.5)

            # E. 估值排名 (3)
            features['xs_pe_rank'] = _safe_float(row.get('xs_pe_rank'), 0.5)
            features['xs_pb_rank'] = _safe_float(row.get('xs_pb_rank'), 0.5)
            features['xs_ps_rank'] = _safe_float(row.get('xs_ps_rank'), 0.5)

            # F. 行业上下文 (8)
            l1_name = self._sw_industry_mapping.get(code)
            if l1_name and self._sw_l1_label_encoding:
                features['sw_l1_code'] = self._sw_l1_label_encoding.get(l1_name, -1)
            else:
                features['sw_l1_code'] = -1

            # 行业涨跌比和成交量变化: 从stock_data_map中统计
            features['industry_breadth'] = _safe_float(
                self._compute_industry_breadth(code, stock_data_map, date), 0.5)
            features['industry_volume_change'] = _safe_float(
                self._compute_industry_volume_change(code, stock_data_map), 1.0)
            features['industry_kdj_avg'] = _safe_float(row.get('industry_kdj_avg'), 50.0)
            features['industry_macd_bullish_pct'] = _safe_float(row.get('industry_macd_bullish_pct'), 0.5)
            features['industry_concentration'] = _safe_float(row.get('industry_concentration'), 0.03)
            features['industry_momentum_rank'] = _safe_float(row.get('industry_momentum_rank'), 0.5)
            features['industry_rotation_signal'] = _safe_float(row.get('industry_rotation_signal'), 0.0)

            # G. 市场状态 (6, 大幅精简)
            # 这些在外层统一计算后传入

            # H. 交互特征 (5)
            xs_ret5 = features['xs_return_5d_rank']
            xs_vol = features['xs_volume_ratio_rank']
            xs_macd = features['xs_macd_hist_rank']
            features['momentum_volume_confirm'] = xs_ret5 * xs_vol
            features['tech_momentum_confirm'] = xs_macd * xs_ret5
            features['squeeze_momentum'] = features['squeeze_state'] * xs_ret5
            features['contrarian_signal'] = (1 - _safe_float(row.get('xs_return_20d_rank'), 0.5)) * xs_vol
            features['trend_consensus'] = (
                features['macd_cross_signal'] *
                features['momentum_direction'] *
                np.sign(features['zhixing_short_trend']) if features['zhixing_short_trend'] != 0 else 0
            )

            # I. 原始基础特征 (5)
            features['rsi_14'] = _safe_float(row.get('rsi_14'), 50.0)
            features['price_position_20d'] = _safe_float(row.get('price_position_20d'), 0.5)
            features['ma_cross'] = _safe_float(row.get('ma_cross'), 0)
            features['avg_pct_change_5d'] = _safe_float(row.get('avg_pct_change_5d'), 0.0)
            features['min_pct_change_5d'] = _safe_float(row.get('min_pct_change_5d'), 0.0)

            result[code] = features

        logger.info(f"Cross-sectional特征计算完成: {len(result)} 只股票")
        return result

    def _compute_industry_breadth(self, code: str,
                                   stock_data_map: Dict[str, pd.DataFrame],
                                   date: str) -> float:
        """计算该股票所在行业的涨跌比"""
        if not hasattr(self, '_industry_breadth_cache'):
            self._industry_breadth_cache = {}

        l1_name = self._sw_industry_mapping.get(code) if self._sw_industry_mapping else None
        if not l1_name:
            return 0.5

        if l1_name in self._industry_breadth_cache:
            return self._industry_breadth_cache[l1_name]

        # 统计同行业股票涨跌
        up = 0
        total = 0
        for c, name in self._sw_industry_mapping.items():
            if name == l1_name:
                df = stock_data_map.get(c)
                if df is not None and len(df) >= 2:
                    last_date = str(df.iloc[-1]['trade_date'])[:10]
                    if last_date == date:
                        pct = (df.iloc[-1]['close'] / df.iloc[-2]['close'] - 1)
                        if pct > 0:
                            up += 1
                        total += 1

        breadth = up / total if total > 0 else 0.5
        self._industry_breadth_cache[l1_name] = breadth
        return breadth

    def _compute_industry_volume_change(self, code: str,
                                         stock_data_map: Dict[str, pd.DataFrame]) -> float:
        """计算该股票所在行业的成交量变化 (今日/5日均量)"""
        if not hasattr(self, '_industry_vol_cache'):
            self._industry_vol_cache = {}

        l1_name = self._sw_industry_mapping.get(code) if self._sw_industry_mapping else None
        if not l1_name:
            return 1.0

        if l1_name in self._industry_vol_cache:
            return self._industry_vol_cache[l1_name]

        total_today = 0
        total_avg5 = 0
        for c, name in self._sw_industry_mapping.items():
            if name == l1_name:
                df = stock_data_map.get(c)
                if df is not None and len(df) >= 5:
                    vols = df['volume'].values
                    total_today += vols[-1]
                    total_avg5 += np.mean(vols[-5:])

        vol_change = total_today / total_avg5 if total_avg5 > 0 else 1.0
        self._industry_vol_cache[l1_name] = vol_change
        return vol_change

    # ================================================================
    # 市场状态特征 (精简版)
    # ================================================================

    def _compute_market_features(self, date: str, stock_data_map: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        计算6个精简版市场状态特征 (分类变量为主，限制信息量)
        """
        features = {
            'market_regime': 1,        # 0=bear, 1=neutral, 2=bull
            'market_vol_regime': 1,    # 0=low, 1=medium, 2=high
            'market_breadth_5d': 0.5,
            'northbound_flow_zscore': 0.0,
            'market_volume_regime': 1,  # 0=shrinking, 1=normal, 2=expanding
            'market_trend_strength': 0.0,
        }

        conn = sqlite3.connect(self.db_path)
        market_df = pd.read_sql_query("""
            SELECT q.trade_date, q.close, q.volume
            FROM daily_quotes q JOIN securities s ON q.security_id = s.id
            WHERE (s.code = '000300.SH' OR s.name = '沪深300')
            AND q.trade_date <= ?
            ORDER BY q.trade_date DESC LIMIT 40
        """, conn, params=(date,))

        # 市场涨跌比 (5日平均)
        breadth_df = pd.read_sql_query("""
            SELECT q.trade_date,
                   SUM(CASE WHEN q.price_change_pct > 0 THEN 1 ELSE 0 END) as up_count,
                   COUNT(*) as total
            FROM daily_quotes q JOIN securities s ON q.security_id = s.id
            WHERE s.type = 'A股' AND q.trade_date <= ?
            GROUP BY q.trade_date
            ORDER BY q.trade_date DESC LIMIT 5
        """, conn, params=(date,))
        conn.close()

        if market_df.empty or len(market_df) < 20:
            return features

        market_df = market_df.sort_values('trade_date')
        closes = market_df['close'].values
        volumes = market_df['volume'].values

        # market_regime: 基于20d return
        ret_20d = closes[-1] / closes[0] - 1 if closes[0] > 0 else 0
        if ret_20d > 0.05:
            features['market_regime'] = 2  # bull
        elif ret_20d < -0.05:
            features['market_regime'] = 0  # bear
        else:
            features['market_regime'] = 1  # neutral

        # market_vol_regime: 基于20d波动率
        log_returns = np.diff(np.log(closes[-21:]))
        vol_20d = np.std(log_returns) * np.sqrt(252) if len(log_returns) >= 20 else 0.2
        if vol_20d < 0.15:
            features['market_vol_regime'] = 0  # low
        elif vol_20d > 0.30:
            features['market_vol_regime'] = 2  # high
        else:
            features['market_vol_regime'] = 1  # medium

        # market_breadth_5d
        if not breadth_df.empty:
            ratios = breadth_df['up_count'] / breadth_df['total'].replace(0, 1)
            features['market_breadth_5d'] = float(ratios.mean())

        # northbound_flow_zscore
        features['northbound_flow_zscore'] = self._load_northbound_flow(date)

        # market_volume_regime
        if len(volumes) >= 20:
            vol_ratio = np.mean(volumes[-5:]) / np.mean(volumes[-20:])
            if vol_ratio < 0.8:
                features['market_volume_regime'] = 0  # shrinking
            elif vol_ratio > 1.2:
                features['market_volume_regime'] = 2  # expanding
            else:
                features['market_volume_regime'] = 1  # normal

        # market_trend_strength (信噪比)
        if len(closes) >= 20 and vol_20d > 0:
            features['market_trend_strength'] = float(abs(ret_20d) / vol_20d)

        return features

    # ================================================================
    # 单日更新
    # ================================================================

    def update_single_date(self, date: str, stock_data_cache: dict = None) -> int:
        """更新单日的V4.0特征缓存

        Args:
            date: 交易日期 YYYY-MM-DD
            stock_data_cache: 可选的预加载股票数据 {code: DataFrame}，避免重复查询
        """
        start_time = time.time()
        logger.info(f"开始更新 {date} 的V4.0特征缓存...")

        # 清除行业聚合缓存
        self._industry_breadth_cache = {}
        self._industry_vol_cache = {}

        # 1. 加载行业映射
        self._load_sw_industry_mapping()

        # 2. 批量预加载数据 (如果有缓存则复用)
        if stock_data_cache is not None:
            stock_data_map = stock_data_cache
            logger.info(f"复用股票数据缓存: {len(stock_data_map)} 只股票")
        else:
            stock_data_map = self._batch_load_stock_data(date, lookback=60)
        tech_indicators = self._batch_load_technical_indicators(date)
        daily_basic = self._batch_load_daily_basic(date)

        # 3. 计算cross-sectional特征
        all_features = self._compute_cross_sectional_features(
            date, stock_data_map, tech_indicators, daily_basic)

        # 4. 计算市场状态特征 (共享)
        market_features = self._compute_market_features(date, stock_data_map)

        # 5. 组装结果
        results = []
        for code, features in all_features.items():
            # 添加市场特征
            features.update(market_features)
            results.append({
                'code': code,
                'trade_date': date,
                'features': features,
            })

        # 6. 写入数据库
        if results:
            inserted = self._batch_insert(results)
            elapsed = time.time() - start_time
            logger.info(f"写入 {inserted} 条记录到 v40_feature_cache, 总耗时 {elapsed:.1f}秒")
            return inserted

        return 0

    # ================================================================
    # 数据库操作
    # ================================================================

    def _ensure_table(self, conn):
        """确保 v40_feature_cache 表存在"""
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS v40_feature_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT NOT NULL,
            trade_date DATE NOT NULL,
            features_json TEXT,
            label_3d_excess REAL,
            label_5d_excess REAL,
            label_10d_excess REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(code, trade_date)
        )
        """)
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_v40_code_date ON v40_feature_cache(code, trade_date)
        """)
        conn.commit()

    def _batch_insert(self, results: List[Dict]) -> int:
        """批量插入特征到数据库"""
        conn = sqlite3.connect(self.db_path)
        self._ensure_table(conn)
        cursor = conn.cursor()

        sql = """
        INSERT OR REPLACE INTO v40_feature_cache
        (code, trade_date, features_json, label_3d_excess, label_5d_excess, label_10d_excess)
        VALUES (?, ?, ?, ?, ?, ?)
        """

        values = []
        for r in results:
            features_json = json.dumps(r['features'], ensure_ascii=False)
            labels = r.get('labels', {})
            values.append((
                r['code'], r['trade_date'], features_json,
                labels.get('label_3d_excess'),
                labels.get('label_5d_excess'),
                labels.get('label_10d_excess'),
            ))

        cursor.executemany(sql, values)
        conn.commit()
        conn.close()
        return len(values)

    # ================================================================
    # 快速批量回填 (一次性预加载 + 滑动窗口 + 超额收益标签)
    # ================================================================

    def update_date_range_fast(self, start_date: str, end_date: str) -> int:
        """
        快速批量回填V4.0特征缓存

        一次性加载所有数据，滑动窗口计算特征+标签
        """
        overall_start = time.time()

        # 获取交易日列表
        conn = sqlite3.connect(self.db_path)
        df_dates = pd.read_sql_query(
            "SELECT DISTINCT trade_date FROM daily_quotes "
            "WHERE trade_date >= ? AND trade_date <= ? ORDER BY trade_date",
            conn, params=(start_date, end_date)
        )
        conn.close()
        trading_dates = df_dates['trade_date'].tolist()

        if not trading_dates:
            logger.warning(f"日期范围 {start_date} ~ {end_date} 无交易日")
            return 0

        logger.info("=" * 80)
        logger.info(f"V4.0 Cross-Sectional特征缓存快速批量回填")
        logger.info(f"  日期范围: {start_date} ~ {end_date} ({len(trading_dates)} 个交易日)")
        logger.info("=" * 80)

        # Phase 1: 一次性预加载
        logger.info("Phase 1: 一次性预加载全部数据...")
        phase1_start = time.time()

        lookback_dt = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=120)
        future_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=20)
        lookback_start = lookback_dt.strftime('%Y-%m-%d')
        future_end = future_dt.strftime('%Y-%m-%d')

        conn = sqlite3.connect(self.db_path)

        # 全部A股行情
        logger.info(f"  加载行情数据...")
        all_quotes = pd.read_sql_query("""
            SELECT s.code, q.trade_date, q.open, q.high, q.low, q.close, q.volume, q.price_change_pct
            FROM daily_quotes q JOIN securities s ON q.security_id = s.id
            WHERE s.type = 'A股' AND q.trade_date >= ? AND q.trade_date <= ?
            ORDER BY s.code, q.trade_date
        """, conn, params=(lookback_start, future_end))
        logger.info(f"  行情: {len(all_quotes):,} 条")

        # 全部技术指标
        logger.info(f"  加载技术指标...")
        all_tech = pd.read_sql_query("""
            SELECT s.code, ti.trade_date,
                   ti.kdj_k, ti.kdj_j, ti.kdj_d,
                   ti.macd_dif, ti.macd_dea, ti.macd_macd,
                   ti.rsi6, ti.rsi12,
                   ti.boll_upper, ti.boll_middle, ti.boll_lower,
                   ti.cci_14, ti.atr_14,
                   ti.squeeze_state, ti.squeeze_days,
                   ti.momentum_direction, ti.zhixing_short_trend
            FROM technical_indicators ti
            JOIN securities s ON ti.security_id = s.id
            WHERE s.type = 'A股' AND ti.trade_date >= ? AND ti.trade_date <= ?
        """, conn, params=(start_date, end_date))
        logger.info(f"  技术指标: {len(all_tech):,} 条")

        # daily_basic
        logger.info(f"  加载基本面数据...")
        all_basic = pd.read_sql_query("""
            SELECT s.code, db.trade_date, db.turnover_rate, db.total_mv,
                   db.pe_ttm, db.pb, db.ps_ttm, db.circ_mv
            FROM daily_basic db JOIN securities s ON db.security_id = s.id
            WHERE s.type = 'A股' AND db.trade_date >= ? AND db.trade_date <= ?
        """, conn, params=(lookback_start, end_date))
        logger.info(f"  基本面: {len(all_basic):,} 条")

        # 沪深300收盘价 (超额收益标签)
        hs300_df = pd.read_sql_query("""
            SELECT q.trade_date, q.close
            FROM daily_quotes q JOIN securities s ON q.security_id = s.id
            WHERE (s.code = '000300.SH' OR s.name = '沪深300')
            AND q.trade_date >= ? AND q.trade_date <= ?
            ORDER BY q.trade_date
        """, conn, params=(lookback_start, future_end))

        conn.close()

        # 按股票分组
        stock_data_all = {}
        stock_dates_idx = {}
        for code, group in all_quotes.groupby('code'):
            df = group.reset_index(drop=True)
            stock_data_all[code] = df
            stock_dates_idx[code] = df['trade_date'].tolist()

        # 技术指标按日期分组
        tech_by_date = {}
        for d, g in all_tech.groupby('trade_date'):
            tech_dict = {}
            for _, row in g.iterrows():
                tech_dict[row['code']] = row.to_dict()
            tech_by_date[d] = tech_dict

        # daily_basic按日期分组
        basic_by_date = {}
        for d, g in all_basic.groupby('trade_date'):
            basic_dict = {}
            for _, row in g.iterrows():
                basic_dict[row['code']] = row.to_dict()
            basic_by_date[d] = basic_dict
        basic_dates_sorted = sorted(basic_by_date.keys())

        # 沪深300索引
        hs300_dates = hs300_df['trade_date'].tolist()
        hs300_closes = hs300_df['close'].values

        total_rows = len(all_quotes)
        del all_quotes, all_tech, all_basic

        phase1_time = time.time() - phase1_start
        logger.info(f"Phase 1 完成: {len(stock_data_all)} 只股票, 耗时 {phase1_time:.1f}秒")

        # Phase 2: 加载行业映射
        self._load_sw_industry_mapping()

        # Phase 3: 逐日计算特征 + 标签
        logger.info("Phase 3: 逐日计算特征+标签...")
        phase3_start = time.time()

        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA busy_timeout=60000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")
        self._ensure_table(conn)

        total_inserted = 0

        for i, date in enumerate(trading_dates):
            # 清除行业聚合缓存
            self._industry_breadth_cache = {}
            self._industry_vol_cache = {}

            # 滑动窗口切片
            stock_data_map = {}
            for code, dates_list in stock_dates_idx.items():
                pos = bisect.bisect_right(dates_list, date)
                if pos < 20:
                    continue
                if dates_list[pos - 1] != date:
                    continue
                start_idx = max(0, pos - 60)
                stock_data_map[code] = stock_data_all[code].iloc[start_idx:pos]

            # 获取当天技术指标
            tech_indicators = tech_by_date.get(date, {})

            # 获取最近的daily_basic
            basic_pos = bisect.bisect_right(basic_dates_sorted, date)
            if basic_pos > 0:
                closest_basic_date = basic_dates_sorted[basic_pos - 1]
                daily_basic = basic_by_date.get(closest_basic_date, {})
            else:
                daily_basic = {}

            # 计算cross-sectional特征
            all_features = self._compute_cross_sectional_features(
                date, stock_data_map, tech_indicators, daily_basic)

            # 计算市场特征 (精简)
            market_features = self._compute_market_features(date, stock_data_map)

            # 计算超额收益标签
            hs300_pos = bisect.bisect_left(hs300_dates, date)

            results = []
            for code, features in all_features.items():
                features.update(market_features)

                # 计算超额收益标签
                labels = self._compute_excess_labels(
                    code, date, stock_data_all.get(code),
                    stock_dates_idx.get(code, []),
                    hs300_dates, hs300_closes)

                results.append({
                    'code': code,
                    'trade_date': date,
                    'features': features,
                    'labels': labels,
                })

            # 批量插入
            if results:
                cursor = conn.cursor()
                sql = """
                INSERT OR REPLACE INTO v40_feature_cache
                (code, trade_date, features_json, label_3d_excess, label_5d_excess, label_10d_excess)
                VALUES (?, ?, ?, ?, ?, ?)
                """
                values = []
                for r in results:
                    features_json = json.dumps(r['features'], ensure_ascii=False)
                    labels = r.get('labels', {})
                    values.append((
                        r['code'], r['trade_date'], features_json,
                        labels.get('label_3d_excess'),
                        labels.get('label_5d_excess'),
                        labels.get('label_10d_excess'),
                    ))
                cursor.executemany(sql, values)
                total_inserted += len(values)

            # 定期commit
            if (i + 1) % 10 == 0:
                conn.commit()

            # 进度报告
            if (i + 1) % 20 == 0 or (i + 1) == len(trading_dates):
                elapsed = time.time() - phase3_start
                rate = (i + 1) / elapsed
                eta = (len(trading_dates) - i - 1) / rate if rate > 0 else 0
                logger.info(f"  [{i+1}/{len(trading_dates)}] {date}: "
                           f"{len(results)} stocks, 总计 {total_inserted:,}, "
                           f"{rate:.1f} 天/秒, ETA {eta:.0f}秒")

        conn.commit()
        conn.close()

        phase3_time = time.time() - phase3_start
        overall_time = time.time() - overall_start

        logger.info("=" * 80)
        logger.info(f"V4.0特征缓存批量回填完成!")
        logger.info(f"  日期范围: {trading_dates[0]} ~ {trading_dates[-1]} ({len(trading_dates)} 天)")
        logger.info(f"  总记录: {total_inserted:,}")
        logger.info(f"  Phase 1 (数据预加载): {phase1_time:.1f}秒")
        logger.info(f"  Phase 3 (特征+标签): {phase3_time:.1f}秒 ({phase3_time/len(trading_dates):.2f}秒/天)")
        logger.info(f"  总耗时: {overall_time:.1f}秒 ({overall_time/60:.1f}分钟)")
        logger.info("=" * 80)

        return total_inserted

    def _compute_excess_labels(self, code: str, date: str,
                                stock_df_full: Optional[pd.DataFrame],
                                dates_list: List[str],
                                hs300_dates: List[str],
                                hs300_closes: np.ndarray) -> Dict[str, Optional[float]]:
        """计算超额收益标签 (个股收益 - 沪深300收益)

        标签定义与回测执行对齐 (股票与基准同一持仓期):
          个股: open[T+1] → close[T+1+N]
          基准: close[T+1] → close[T+1+N] (同持仓期, 指数用close-to-close)
          excess = 个股收益 - 基准同期收益
        """
        labels = {'label_3d_excess': None, 'label_5d_excess': None, 'label_10d_excess': None}

        if stock_df_full is None or len(dates_list) == 0:
            return labels

        pos = bisect.bisect_left(dates_list, date)
        if pos >= len(dates_list) or dates_list[pos] != date:
            return labels

        volumes = stock_df_full['volume'].values

        # 报告日停牌检测
        if volumes[pos] == 0:
            return labels

        # 需要至少 pos+2 的数据
        remaining = len(dates_list) - pos
        if remaining < 3:
            return labels

        # 使用统一标签计算个股收益 (current_idx=pos 即报告日)
        stock_labels = compute_aligned_labels(
            opens=stock_df_full['open'].values,
            closes=stock_df_full['close'].values,
            volumes=volumes,
            current_idx=pos,
            horizons=(3, 5, 10),
        )
        # 如果个股标签全为 nan，说明买入日无法交易
        if all(np.isnan(stock_labels.get(f'label_{n}d', np.nan)) for n in (3, 5, 10)):
            return labels

        # 沪深300基准: 同持仓期 close[T+1] → close[T+1+N]
        hs300_pos = bisect.bisect_left(hs300_dates, date)
        if hs300_pos >= len(hs300_dates) or hs300_dates[hs300_pos] != date:
            return labels
        # 基准也从 T+1 起算
        hs300_buy_pos = hs300_pos + 1
        if hs300_buy_pos >= len(hs300_closes):
            return labels
        hs300_base = hs300_closes[hs300_buy_pos]
        if hs300_base <= 0:
            return labels

        hs300_remaining = len(hs300_closes) - hs300_buy_pos

        for n, key in [(3, 'label_3d_excess'), (5, 'label_5d_excess'), (10, 'label_10d_excess')]:
            stock_ret = stock_labels.get(f'label_{n}d', np.nan)
            if not np.isnan(stock_ret) and hs300_remaining > n:
                market_ret = hs300_closes[hs300_buy_pos + n] / hs300_base - 1
                labels[key] = float(stock_ret - market_ret)

        return labels

    def backfill_labels(self, batch_size: int = 1000) -> int:
        """回填已有缓存中的超额收益标签"""
        logger.info("=" * 80)
        logger.info("📊 回填V4.0超额收益标签...")
        logger.info("=" * 80)
        start_time = time.time()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 找截止日期 (>= 11个交易日前)
        cursor.execute("""
            SELECT DISTINCT trade_date FROM daily_quotes
            ORDER BY trade_date DESC LIMIT 15
        """)
        recent_dates = [r[0] for r in cursor.fetchall()]
        if len(recent_dates) < 11:
            logger.warning("交易日不足，无法回填标签")
            conn.close()
            return 0
        cutoff_date = recent_dates[10]

        # 查找需要回填的记录 (用最长horizon确保所有label都被填充)
        cursor.execute("""
            SELECT id, code, trade_date FROM v40_feature_cache
            WHERE label_10d_excess IS NULL AND trade_date <= ?
            ORDER BY trade_date, code
        """, (cutoff_date,))
        rows = cursor.fetchall()
        total = len(rows)
        logger.info(f"需要回填: {total:,} 条记录 (截至 {cutoff_date})")

        if total == 0:
            conn.close()
            return 0

        # 加载security_id映射
        cursor.execute("SELECT code, id FROM securities WHERE type = 'A股'")
        code_to_sid = {r[0]: r[1] for r in cursor.fetchall()}

        # 加载沪深300的security_id
        cursor.execute("""
            SELECT s.id FROM securities s WHERE s.code = '000300.SH' OR s.name = '沪深300' LIMIT 1
        """)
        hs300_row = cursor.fetchone()
        hs300_sid = hs300_row[0] if hs300_row else None
        if not hs300_sid:
            logger.warning("找不到沪深300的security_id，无法回填超额标签")
            conn.close()
            return 0

        # --- 批量预加载价格数据 (替代 N+1 per-row 查询) ---
        needed_codes = set(row[1] for row in rows)
        needed_sids = [code_to_sid[c] for c in needed_codes if c in code_to_sid]
        min_date = min(row[2] for row in rows)
        logger.info(f"批量预加载价格数据: {len(needed_sids)} 只股票, 起始日期 {min_date}")

        # 股票价格: sid → sorted list of (trade_date, open, close, volume)
        price_data = {}
        for chunk_start in range(0, len(needed_sids), 900):
            chunk = needed_sids[chunk_start:chunk_start + 900]
            placeholders = ','.join('?' * len(chunk))
            price_rows = cursor.execute(f"""
                SELECT security_id, trade_date, open, close, volume
                FROM daily_quotes
                WHERE security_id IN ({placeholders})
                  AND trade_date >= ?
                ORDER BY security_id, trade_date
            """, chunk + [min_date]).fetchall()
            for sid, td, op, cl, vol in price_rows:
                if sid not in price_data:
                    price_data[sid] = []
                price_data[sid].append((td, op, cl, vol))

        # 股票 trade_date → index 快速查找
        sid_date_idx = {}
        for sid, prices in price_data.items():
            sid_date_idx[sid] = {prices[i][0]: i for i in range(len(prices))}

        # 沪深300价格: 一次性加载全部 (替代 N 次 per-row 查询)
        hs300_prices = cursor.execute("""
            SELECT trade_date, close FROM daily_quotes
            WHERE security_id = ? AND trade_date >= ?
            ORDER BY trade_date
        """, (hs300_sid, min_date)).fetchall()
        # trade_date → (index, close) 的快速查找
        hs300_date_idx = {hs300_prices[i][0]: i for i in range(len(hs300_prices))}

        load_elapsed = time.time() - start_time
        logger.info(f"价格数据预加载完成: {sum(len(v) for v in price_data.values()):,} 条股票 + "
                    f"{len(hs300_prices):,} 条HS300, 耗时 {load_elapsed:.1f}秒")

        updated = 0
        for i in range(0, total, batch_size):
            batch = rows[i:i + batch_size]
            for cache_id, code, trade_date in batch:
                sid = code_to_sid.get(code)
                if not sid or sid not in price_data:
                    continue

                prices = price_data[sid]
                idx = sid_date_idx[sid].get(trade_date)
                if idx is None:
                    continue

                stock_rows = prices[idx:idx + 17]

                # 沪深300从同一日期切片
                hs300_idx = hs300_date_idx.get(trade_date)
                if hs300_idx is None:
                    continue
                hs300_rows = hs300_prices[hs300_idx:hs300_idx + 15]

                # 至少需要报告日 + 买入日 + 1天
                if len(stock_rows) < 3 or len(hs300_rows) < 3:
                    continue

                # 报告日停牌检测
                base_vol = stock_rows[0][3]
                if base_vol is not None and base_vol == 0:
                    continue

                # 买入日 (T+1) 的开盘价作为 base
                buy_open = stock_rows[1][1]  # open
                buy_vol = stock_rows[1][3]   # volume
                # 基准也从 T+1 起算 (同持仓期)
                hs300_base = hs300_rows[1][1]  # close[T+1]

                if not buy_open or buy_open <= 0 or (buy_vol is not None and buy_vol == 0):
                    continue
                if not hs300_base or hs300_base <= 0:
                    continue

                label_3d = label_5d = label_10d = None
                for n, label_idx in [(3, 0), (5, 1), (10, 2)]:
                    if len(stock_rows) > 1 + n and len(hs300_rows) > 1 + n:
                        stock_ret = stock_rows[1 + n][2] / buy_open - 1  # [2]=close
                        mkt_ret = hs300_rows[1 + n][1] / hs300_base - 1  # 基准同持仓期
                        val = stock_ret - mkt_ret
                        if label_idx == 0:
                            label_3d = val
                        elif label_idx == 1:
                            label_5d = val
                        else:
                            label_10d = val

                if label_3d is not None:
                    cursor.execute("""
                        UPDATE v40_feature_cache
                        SET label_3d_excess = ?, label_5d_excess = ?, label_10d_excess = ?
                        WHERE id = ?
                    """, (label_3d, label_5d, label_10d, cache_id))
                    updated += 1

            conn.commit()
            if (i + batch_size) % 10000 == 0 or i + batch_size >= total:
                elapsed = time.time() - start_time
                progress = min(i + batch_size, total)
                logger.info(f"进度: {progress:,}/{total:,}, 已更新: {updated:,}, 耗时: {elapsed:.0f}秒")

        conn.close()
        elapsed = time.time() - start_time
        logger.info(f"✅ 标签回填完成: {updated:,} 条, 耗时: {elapsed:.1f}秒")
        return updated


# ================================================================
# 工具函数
# ================================================================

def _safe_float(val, default: float = 0.0) -> float:
    """安全转换为float"""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def update_v40_feature_cache(date_str: str, stock_data_cache: dict = None) -> int:
    """
    每日更新入口函数 (供quick_daily_update.py调用)

    Args:
        date_str: 日期字符串，格式 YYYYMMDD
        stock_data_cache: 可选的预加载股票数据 {code: DataFrame}
    """
    date_dash = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    try:
        updater = V40FeatureCacheUpdater()
        count = updater.update_single_date(date_dash, stock_data_cache=stock_data_cache)
        return count
    except Exception as e:
        logger.error(f"更新V4.0特征缓存失败: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description='V4.0 Cross-Sectional特征缓存更新')
    parser.add_argument('--date', type=str, help='单日更新日期 (YYYY-MM-DD)')
    parser.add_argument('--start-date', type=str, help='批量更新开始日期')
    parser.add_argument('--end-date', type=str, help='批量更新结束日期')
    parser.add_argument('--backfill-labels', action='store_true',
                        help='回填已有缓存的超额收益标签')

    args = parser.parse_args()

    updater = V40FeatureCacheUpdater()

    if args.backfill_labels:
        updater.backfill_labels()
    elif args.date:
        updater.update_single_date(args.date)
    elif args.start_date and args.end_date:
        updater.update_date_range_fast(args.start_date, args.end_date)
    else:
        today = datetime.now().strftime('%Y-%m-%d')
        updater.update_single_date(today)


if __name__ == '__main__':
    main()
