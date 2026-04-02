#!/usr/bin/env python3
"""
V4.9.2 production scorer — V4.9.0底座 + 10新因子推理 + 预测EMA平滑

模型: V4.9.2 (66特征 = 61-5弱+5V482+5Alpha158)
推理:
  1. 基础61特征 (V4.8.5 pipeline)
  2. 注入10新因子 (从DB实时计算)
  3. 删除5弱因子 (model不使用的列)
  4. 模型预测
  5. EMA平滑 (alpha=0.7)
继承: V4.9.0的Q95 + 市场门控
"""

import numpy as np
import pandas as pd
import sqlite3
import logging
from datetime import datetime as dt_cls, timedelta as td_cls
from pathlib import Path
from typing import Dict, List, Optional

from .v490_production_scorer import V490ProductionScorer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# V492新增因子
V492_NEW_FACTORS = [
    'high_52w_ratio', 'residual_momentum', 'turnover_reversal',
    'sumd_20d', 'realized_skew_20d',
    'corr_close_vol_20d', 'cntp_20d', 'rsqr_20d', 'ksft', 'imax_20d',
]

# V492删除的弱因子
V492_PRUNE = ['dv_ttm', 'max_pct_change_5d', 'brain_roll_spread', 'cci_14', 'macd_hist']


class PredictionSmoother:
    """EMA smoother for stock predictions across calls."""

    def __init__(self, alpha: float = 0.7):
        self.alpha = alpha
        self._cache = {}

    def smooth_batch(self, codes: list, raw_preds: np.ndarray) -> np.ndarray:
        result = np.empty_like(raw_preds)
        for i, (code, pred) in enumerate(zip(codes, raw_preds)):
            if code in self._cache:
                result[i] = self.alpha * pred + (1 - self.alpha) * self._cache[code]
            else:
                result[i] = pred
            self._cache[code] = float(result[i])
        return result

    def reset(self):
        self._cache.clear()


def _compute_v492_factors_for_date(codes: list, date: str, db_path: str) -> pd.DataFrame:
    """计算10个V492新因子 (从DB实时加载OHLCV)

    5个V482因子: high_52w_ratio, residual_momentum, turnover_reversal, sumd_20d, realized_skew_20d
    5个Alpha158因子: corr_close_vol_20d, cntp_20d, rsqr_20d, ksft, imax_20d
    """
    conn = sqlite3.connect(db_path)

    # 需要252天历史 (for high_52w_ratio)
    try:
        dt = dt_cls.strptime(date, '%Y-%m-%d')
    except ValueError:
        dt = dt_cls.strptime(date, '%Y%m%d')
    lookback_start = (dt - td_cls(days=380)).strftime('%Y-%m-%d')

    # Strip exchange suffix for DB query (000001.SZ → 000001)
    codes_stripped = set(c.split('.')[0] if '.' in c else c for c in codes)
    # Build reverse map: stripped → original (with suffix)
    code_map = {}
    for c in codes:
        stripped = c.split('.')[0] if '.' in c else c
        code_map[stripped] = c

    # 全A股OHLCV查询 (比5000+ IN更快) — 只取最近30天用于因子计算
    short_lookback = (dt - td_cls(days=40)).strftime('%Y-%m-%d')
    # 252天lookback只用于high_52w_ratio, 用单独查询
    query = """
    SELECT s.code, q.trade_date, q.open, q.high, q.low, q.close,
           q.volume, q.price_change_pct
    FROM daily_quotes q
    JOIN securities s ON q.security_id = s.id
    WHERE s.type = 'A股'
      AND q.trade_date >= ? AND q.trade_date <= ?
      AND q.volume > 0
    ORDER BY s.code, q.trade_date
    """
    df_ohlcv = pd.read_sql(query, conn, params=[short_lookback, date])
    # 过滤到目标股票
    df_ohlcv = df_ohlcv[df_ohlcv['code'].isin(codes_stripped)].copy()

    # 252天高点 (只查high用于high_52w_ratio)
    high_query = """
    SELECT s.code, MAX(q.high) as max_high_252
    FROM daily_quotes q JOIN securities s ON q.security_id = s.id
    WHERE s.type = 'A股' AND q.trade_date >= ? AND q.trade_date <= ? AND q.volume > 0
    GROUP BY s.code
    """
    df_high252 = pd.read_sql(high_query, conn, params=[lookback_start, date])
    high252_map = dict(zip(df_high252['code'], df_high252['max_high_252']))

    # 换手率
    turnover_query = """
    SELECT s.code, db.trade_date, db.turnover_rate
    FROM daily_basic db JOIN securities s ON db.security_id = s.id
    WHERE db.trade_date >= ? AND db.trade_date <= ?
    """
    df_tr = pd.read_sql(turnover_query, conn, params=[short_lookback, date])
    df_tr = df_tr[df_tr['code'].isin(codes_stripped)].copy()

    # 全市场平均收益 (for residual_momentum) — 只取近30天加速查询
    mkt_start = (dt - td_cls(days=40)).strftime('%Y-%m-%d')
    mkt_query = """
    SELECT q.trade_date, AVG(q.price_change_pct) as mkt_ret
    FROM daily_quotes q JOIN securities s ON q.security_id = s.id
    WHERE s.type = 'A股' AND q.trade_date >= ? AND q.trade_date <= ? AND q.volume > 0
    GROUP BY q.trade_date
    """
    df_mkt = pd.read_sql(mkt_query, conn, params=[mkt_start, date])
    conn.close()

    if df_ohlcv.empty:
        return pd.DataFrame({'code': codes, **{f: 0.0 for f in V492_NEW_FACTORS}})

    df_ohlcv = df_ohlcv.merge(df_tr[['code', 'trade_date', 'turnover_rate']],
                               on=['code', 'trade_date'], how='left')
    df_ohlcv['turnover_rate'] = df_ohlcv['turnover_rate'].fillna(0.0)

    mkt_ret_map = dict(zip(df_mkt['trade_date'], df_mkt['mkt_ret']))

    results = []
    for code_raw, grp in df_ohlcv.groupby('code'):
        code = code_map.get(code_raw, code_raw)  # map back to original format
        grp = grp.sort_values('trade_date').reset_index(drop=True)
        n = len(grp)
        if n < 25:
            results.append({'code': code, **{f: 0.0 for f in V492_NEW_FACTORS}})
            continue

        close = grp['close'].values.astype(float)
        open_ = grp['open'].values.astype(float)
        high = grp['high'].values.astype(float)
        low = grp['low'].values.astype(float)
        volume = grp['volume'].values.astype(float)
        pct = pd.to_numeric(grp['price_change_pct'], errors='coerce').fillna(0).values.astype(float)
        turnover = grp['turnover_rate'].values.astype(float)
        dates = grp['trade_date'].values

        row = {'code': code}

        # 1. high_52w_ratio: close[-1] / max(high, 252d)
        code_stripped = code.split('.')[0] if '.' in code else code
        max_high_252 = high252_map.get(code_stripped, np.max(high))
        row['high_52w_ratio'] = close[-1] / max(max_high_252, 1e-8)

        # 2. residual_momentum: 去市场残差动量 (25d window, skip last 5d)
        if n >= 25:
            stock_r = pct[-25:]
            mkt_r = np.array([float(mkt_ret_map.get(d, 0) or 0) for d in dates[-25:]])
            mkt_var = np.var(mkt_r)
            if mkt_var > 1e-12:
                beta = np.cov(stock_r, mkt_r)[0, 1] / mkt_var
            else:
                beta = 0.0
            residual = stock_r - beta * mkt_r
            row['residual_momentum'] = float(np.sum(residual[:20]))
        else:
            row['residual_momentum'] = 0.0

        # 3. turnover_reversal: -mean(turnover, 20d)
        window = min(20, n)
        row['turnover_reversal'] = -float(np.mean(turnover[-window:]))

        # 4. sumd_20d: (gains - losses) / (gains + losses)
        window = min(20, n)
        r = pct[-window:]
        sum_g = np.sum(r[r > 0])
        sum_l = np.sum(-r[r < 0])
        denom = sum_g + sum_l
        row['sumd_20d'] = (sum_g - sum_l) / max(denom, 1e-8)

        # 5. realized_skew_20d
        window = min(20, n)
        r = pct[-window:]
        mu, sigma = np.mean(r), np.std(r)
        if sigma > 1e-8:
            row['realized_skew_20d'] = float(np.mean(((r - mu) / sigma) ** 3))
        else:
            row['realized_skew_20d'] = 0.0

        # 6. corr_close_vol_20d: Corr(close, log(volume), 20d)
        window = min(20, n)
        c_w = close[-window:]
        v_w = np.log(volume[-window:] + 1)
        if np.std(c_w) > 1e-8 and np.std(v_w) > 1e-8:
            row['corr_close_vol_20d'] = float(np.corrcoef(c_w, v_w)[0, 1])
        else:
            row['corr_close_vol_20d'] = 0.0

        # 7. cntp_20d: fraction of up-days
        window = min(20, n)
        c_w = close[-window:]
        c_prev = np.concatenate([[c_w[0]], c_w[:-1]])
        row['cntp_20d'] = float(np.mean(c_w > c_prev))

        # 8. rsqr_20d: R-squared of linear fit
        window = min(20, n)
        c_w = close[-window:]
        x = np.arange(len(c_w))
        if np.std(c_w) > 1e-8:
            corr = np.corrcoef(x, c_w)[0, 1]
            row['rsqr_20d'] = corr ** 2
        else:
            row['rsqr_20d'] = 0.0

        # 9. ksft: (2*close - high - low) / open (today only)
        row['ksft'] = (2 * close[-1] - high[-1] - low[-1]) / max(open_[-1], 1e-8)

        # 10. imax_20d: position of max high in 20d (0=oldest, 1=newest)
        window = min(20, n)
        h_w = high[-window:]
        row['imax_20d'] = float(np.argmax(h_w)) / max(len(h_w) - 1, 1)

        results.append(row)

    df_result = pd.DataFrame(results)

    # 补齐缺失code (用原始格式codes)
    missing_codes = set(codes) - set(df_result['code'])
    if missing_codes:
        missing_df = pd.DataFrame({'code': list(missing_codes),
                                    **{f: 0.0 for f in V492_NEW_FACTORS}})
        df_result = pd.concat([df_result, missing_df], ignore_index=True)

    return df_result


class V492ProductionScorer(V490ProductionScorer):
    """V4.9.2 scorer — 66特征 + EMA平滑 + Q95 + 市场门控"""

    def __init__(self, model_type: str = 'small_data', smooth_alpha: float = 0.7):
        self._smooth_alpha = smooth_alpha
        self._smoother_10d = PredictionSmoother(alpha=smooth_alpha)
        self._smoother_15d = PredictionSmoother(alpha=smooth_alpha)
        self._v492_model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v492'
        self._v492_factor_cache = {}  # date -> DataFrame
        super().__init__(model_type=model_type)
        logger.info(f"  V4.9.2 预测平滑: alpha={smooth_alpha}")

    def _load_models(self):
        """优先加载v492模型(66特征), fallback到v490"""
        v492_files = list(self._v492_model_dir.glob('v492_*.pkl'))
        if v492_files:
            self.model_dir = self._v492_model_dir
            latest = max(v492_files, key=lambda f: f.stat().st_mtime)
            self._load_model_from_file(latest, label='V4.9.2')
            return
        super()._load_models()

    def _compute_v492_new_factors(self, features_df: pd.DataFrame, date: str) -> pd.DataFrame:
        """注入10个V492新因子 (优先从DB缓存读) + 删除5个弱因子"""
        if date in self._v492_factor_cache:
            df_factors = self._v492_factor_cache[date]
        else:
            # 优先从DB缓存读 (与训练完全一致)
            df_factors = self._load_v492_factors_from_db(date)
            if df_factors is None or len(df_factors) == 0:
                # 回退: 实时计算 (可能不精确)
                codes = features_df['code'].tolist()
                db_path = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')
                df_factors = _compute_v492_factors_for_date(codes, date, db_path)
            self._v492_factor_cache[date] = df_factors

        if df_factors is not None and len(df_factors) > 0:
            for col in V492_NEW_FACTORS:
                if col in df_factors.columns:
                    factor_map = dict(zip(df_factors['code'], df_factors[col]))
                    features_df[col] = features_df['code'].map(factor_map).fillna(0.0)
                else:
                    features_df[col] = 0.0

        # 删除弱因子
        for col in V492_PRUNE:
            if col in features_df.columns:
                features_df[col] = 0.0  # 填0而非drop, 保持列索引对齐

        return features_df

    def _load_v492_factors_from_db(self, date: str) -> Optional[pd.DataFrame]:
        """从v492_factor_cache表读取预计算因子"""
        try:
            import sqlite3
            conn = sqlite3.connect(str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db'))
            cols = ', '.join(V492_NEW_FACTORS)
            df = pd.read_sql_query(
                f"SELECT code, {cols} FROM v492_factor_cache WHERE trade_date = ?",
                conn, params=[date])
            conn.close()
            if len(df) > 0:
                return df
        except Exception as e:
            logger.debug(f"v492_factor_cache miss for {date}: {e}")
        return None

    def predict_scores(self, stock_codes: List[str], date: str) -> Dict[str, Dict]:
        """V4.9.2: 注入新因子 → V4.9.0评分 → EMA平滑"""
        # monkey-patch: 在V481因子计算后追加V492因子
        original_compute = self._compute_v481_new_factors

        def _compute_with_v492(features_df, date_arg):
            features_df = original_compute(features_df, date_arg)
            features_df = self._compute_v492_new_factors(features_df, date_arg)
            return features_df

        self._compute_v481_new_factors = _compute_with_v492
        try:
            results = super().predict_scores(stock_codes, date)
        finally:
            self._compute_v481_new_factors = original_compute

        # EMA平滑
        codes_with_pred = [(c, d) for c, d in results.items()
                           if d.get('pred_10d') is not None and d.get('score', 0) > 0]

        if codes_with_pred:
            codes = [c for c, _ in codes_with_pred]
            raw_10d = np.array([d['pred_10d'] for _, d in codes_with_pred])
            raw_15d = np.array([d.get('pred_15d', 0) for _, d in codes_with_pred])

            smoothed_10d = self._smoother_10d.smooth_batch(codes, raw_10d)
            smoothed_15d = self._smoother_15d.smooth_batch(codes, raw_15d)

            for i, (code, _) in enumerate(codes_with_pred):
                results[code]['pred_10d_raw'] = results[code]['pred_10d']
                results[code]['pred_10d'] = float(smoothed_10d[i])
                results[code]['pred_15d_raw'] = results[code].get('pred_15d', 0)
                results[code]['pred_15d'] = float(smoothed_15d[i])
                results[code]['rank_score'] = 0.6 * smoothed_10d[i] + 0.4 * smoothed_15d[i]

        return results

    def reset_smoothers(self):
        self._smoother_10d.reset()
        self._smoother_15d.reset()
        self._v492_factor_cache.clear()
