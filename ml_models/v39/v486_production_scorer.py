#!/usr/bin/env python3
"""
V4.8.6 production scorer -- 64 features + RRF ensemble (头部区分度优化)

Architecture:
  Model: V4.8.6 trained model (64 features, A股+ETF, LambdaRank truncation)
  Scorer: V4.8.5 base + 3 BRAIN factors + RRF ensemble aggregation
  Key innovation: Reciprocal Rank Fusion replaces weighted average → head discrimination

New factors (TopK Sharpe validated, +4.2% ICIR):
  brain_high_low_ratio, brain_close_to_high, brain_momentum_decay10

Head discrimination fixes:
  1. RRF ensemble: rank-based fusion immune to score compression
  2. LambdaRank truncation_level=15 (training-side, gradient focus on top-15)
  3. Tree params: num_leaves=63, min_data=50 (finer head resolution)

Fallback chain: v486 model -> v485 model -> v484 model -> v481 model -> v475 model
"""

import json
import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from .v485_production_scorer import V485ProductionScorer

import logging
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class V486ProductionScorer(V485ProductionScorer):
    """V4.8.6 scorer -- 64 features + RRF ensemble (head discrimination fix)"""

    V486_BRAIN_FACTORS = [
        'brain_high_low_ratio',       # Top-K Sharpe验证
        'brain_close_to_high',        # Top-K Sharpe验证
        'brain_momentum_decay10',     # Top-K Sharpe验证
    ]

    def __init__(self, model_type: str = 'small_data'):
        self._v486_model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v486'
        self._brain_factors_cache = {}
        super().__init__(model_type=model_type)

    def _load_models(self):
        """Try v486 model first, fallback to v485"""
        v486_files = list(self._v486_model_dir.glob('v486_*.pkl'))
        if v486_files:
            self.model_dir = self._v486_model_dir
            latest = max(v486_files, key=lambda f: f.stat().st_mtime)
            self._load_model_from_file(latest, label='V4.8.6')
            return
        super()._load_models()

    def _load_brain_factors(self, date: str) -> Dict[str, Dict[str, float]]:
        """Load 3 BRAIN factors from cache for a given date."""
        if date in self._brain_factors_cache:
            return self._brain_factors_cache[date]

        result = {}
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "SELECT code, features_json FROM brain_alpha_cache WHERE trade_date = ?",
                (date,)
            )
            for code, fj in cursor:
                try:
                    parsed = json.loads(fj)
                    factors = {}
                    for f in self.V486_BRAIN_FACTORS:
                        factors[f] = float(parsed.get(f, 0))
                    result[code] = factors
                except Exception:
                    pass
            conn.close()
        except Exception as e:
            logger.warning(f"V4.8.6 brain factors load failed for {date}: {e}")

        self._brain_factors_cache[date] = result
        return result

    def _compute_v486_brain_factors(self, features_df: pd.DataFrame, date: str) -> pd.DataFrame:
        """Inject 3 BRAIN factors into features DataFrame."""
        factors_data = self._load_brain_factors(date)

        if factors_data:
            for f in self.V486_BRAIN_FACTORS:
                features_df[f] = features_df['code'].map(
                    lambda c, _f=f: factors_data.get(c, {}).get(_f, 0.0)
                ).fillna(0.0)
        else:
            for f in self.V486_BRAIN_FACTORS:
                features_df[f] = 0.0

        return features_df

    def _cascade_ensemble_predict(self, X_input, models, weights):
        """V4.8.6: RRF ensemble (排名融合, 免疫分数压缩)"""
        from scipy.stats import rankdata

        preds = {}
        for name, model in models.items():
            try:
                preds[name] = model.predict(X_input)
            except Exception:
                continue

        if not preds:
            return np.zeros(len(X_input)), False
        if len(preds) == 1:
            return list(preds.values())[0], True

        n = len(X_input)
        k = 60

        rrf_scores = np.zeros(n)
        for name, pred in preds.items():
            ranks = rankdata(pred, method='ordinal')
            desc_ranks = n + 1 - ranks
            rrf_scores += 1.0 / (k + desc_ranks)

        # 映射回均值预测尺度
        rrf_ranks = rankdata(rrf_scores, method='ordinal')
        all_preds = np.column_stack(list(preds.values()))
        mean_pred = np.mean(all_preds, axis=1)
        sorted_mean = np.sort(mean_pred)
        rrf_calibrated = sorted_mean[rrf_ranks - 1]

        return rrf_calibrated, True

    V482_ALL_FACTORS = [
        'limit_proximity_5d', 'trend_strength_60d', 'high_52w_ratio',
        'max5_lottery', 'imxd_20d',
    ]

    def _compute_v482_top5_factors(self, features_df: pd.DataFrame, date: str) -> pd.DataFrame:
        """计算13个V4.8.2 IC验证因子 (从DB加载OHLCV, per-stock计算)"""
        try:
            conn = sqlite3.connect(self.db_path)
            from datetime import datetime as dt_cls, timedelta as td_cls
            ext_start = (dt_cls.strptime(date, '%Y-%m-%d') - td_cls(days=280)).strftime('%Y-%m-%d')

            codes_str = ','.join(f"'{c}'" for c in features_df['code'].unique())
            ohlcv = pd.read_sql(f"""
                SELECT s.code, q.trade_date, q.high, q.low, q.close, q.volume, q.price_change_pct
                FROM daily_quotes q JOIN securities s ON q.security_id = s.id
                WHERE s.type = 'A股' AND q.trade_date >= ? AND q.trade_date <= ?
                  AND s.code IN ({codes_str})
                ORDER BY s.code, q.trade_date
            """, conn, params=[ext_start, date])

            # turnover_rate
            df_turn = pd.read_sql(f"""
                SELECT s.code, db.trade_date, db.turnover_rate
                FROM daily_basic db JOIN securities s ON db.security_id = s.id
                WHERE db.trade_date >= ? AND db.trade_date <= ?
                  AND s.code IN ({codes_str})
            """, conn, params=[ext_start, date])
            conn.close()

            if not df_turn.empty:
                ohlcv = ohlcv.merge(df_turn, on=['code', 'trade_date'], how='left')
                ohlcv['turnover_rate'] = ohlcv['turnover_rate'].fillna(0.0)
            else:
                ohlcv['turnover_rate'] = 0.0

            # 市场中位数收益 (industry_adj_str, residual_momentum)
            mkt_ret = ohlcv.groupby('trade_date')['price_change_pct'].median().to_dict()

            factor_map = {}
            for code, grp in ohlcv.groupby('code'):
                grp = grp.sort_values('trade_date')
                if len(grp) < 60:
                    continue
                close = grp['close'].values
                high = grp['high'].values
                low = grp['low'].values
                pct = grp['price_change_pct'].values
                volume = grp['volume'].values.astype(float)
                turnover = grp['turnover_rate'].values
                n = len(close)

                vals = {}
                # 1. limit_proximity_5d
                lp = 0.20 if (code.startswith('3') or code.startswith('688')) else (0.30 if code.startswith(('4', '8')) else 0.10)
                vals['limit_proximity_5d'] = float(np.mean(np.abs(pct[-5:]) / max(lp, 1e-8)))

                # 2. trend_strength_60d
                if n >= 61:
                    ret60 = close[-1] / close[-61] - 1 if close[-61] > 0 else 0
                    vol60 = max(np.std(pct[-60:]), 1e-8)
                    vals['trend_strength_60d'] = ret60 / (vol60 * np.sqrt(60))
                else:
                    vals['trend_strength_60d'] = 0.0

                # 3. high_52w_ratio
                max_high = np.max(high[-min(252, n):])
                vals['high_52w_ratio'] = close[-1] / max(max_high, 1e-8)

                # 4. max5_lottery
                vals['max5_lottery'] = -float(np.mean(np.sort(pct[-20:])[-5:])) if n >= 20 else 0.0

                # 5. imxd_20d
                if n >= 20:
                    vals['imxd_20d'] = np.argmax(high[-20:]) / 19.0 - np.argmin(low[-20:]) / 19.0
                else:
                    vals['imxd_20d'] = 0.0

                # 6. sumd_20d
                if n >= 20:
                    g = np.sum(np.where(pct[-20:] > 0, pct[-20:], 0))
                    l = np.sum(np.where(pct[-20:] < 0, -pct[-20:], 0))
                    vals['sumd_20d'] = (g - l) / max(g + l, 1e-8)
                else:
                    vals['sumd_20d'] = 0.0

                # 7. industry_adj_str
                if n >= 6:
                    ret5 = close[-1] / close[-6] - 1 if close[-6] > 0 else 0
                    dates_list = grp['trade_date'].values
                    mkt_med = mkt_ret.get(dates_list[-1], 0.0)
                    vals['industry_adj_str'] = -(ret5 - mkt_med)
                else:
                    vals['industry_adj_str'] = 0.0

                # 8. turnover_reversal
                vals['turnover_reversal'] = -float(np.mean(turnover[-20:])) if n >= 20 else 0.0

                # 9. residual_momentum
                if n >= 25:
                    stock_r = pct[-25:]
                    dates_arr = grp['trade_date'].values[-25:]
                    mkt_r = np.array([mkt_ret.get(d, 0.0) for d in dates_arr])
                    mkt_var = np.var(mkt_r)
                    beta = np.cov(stock_r, mkt_r)[0, 1] / mkt_var if mkt_var > 1e-12 else 0.0
                    vals['residual_momentum'] = float(np.sum((stock_r - beta * mkt_r)[:20]))
                else:
                    vals['residual_momentum'] = 0.0

                # 10. realized_skew_20d
                if n >= 20:
                    r = pct[-20:]
                    mu, sigma = np.mean(r), max(np.std(r), 1e-8)
                    vals['realized_skew_20d'] = float(np.mean(((r - mu) / sigma) ** 3))
                else:
                    vals['realized_skew_20d'] = 0.0

                # 11. retail_crowding (simplified)
                vals['retail_crowding'] = -float(np.mean(turnover[-20:])) if n >= 20 else 0.0

                # 12. obv_price_div
                if n >= 20:
                    obv_sign = np.where(pct > 0, 1, np.where(pct < 0, -1, 0))
                    obv = np.cumsum(obv_sign * volume)
                    obv_ret = (obv[-1] / obv[-21] - 1) if abs(obv[-21]) > 0 else 0
                    price_ret = close[-1] / close[-21] - 1 if close[-21] > 0 else 0
                    vals['obv_price_div'] = obv_ret - price_ret
                else:
                    vals['obv_price_div'] = 0.0

                # 13. delta_roe_yoy (placeholder)
                vals['delta_roe_yoy'] = 0.0

                factor_map[code] = vals

            for f in self.V482_ALL_FACTORS:
                features_df[f] = features_df['code'].map(
                    lambda c, _f=f: factor_map.get(c, {}).get(_f, 0.0)
                ).fillna(0.0)

        except Exception as e:
            logger.warning(f"V4.8.6 V482 factors failed: {e}")
            for f in self.V482_ALL_FACTORS:
                features_df[f] = 0.0

        return features_df

    def predict_scores(self, stock_codes: List[str], date: str) -> Dict[str, Dict]:
        """V4.8.6 scoring — V4.8.5 + 3 BRAIN + 5 V482 + 头部加权训练."""
        original_v481_compute = self._compute_v481_new_factors

        def _compute_v481_and_extras(features_df, date_arg):
            features_df = original_v481_compute(features_df, date_arg)
            features_df = self._compute_v484_roll_spread(features_df, date_arg)
            features_df = self._compute_v486_brain_factors(features_df, date_arg)
            features_df = self._compute_v482_top5_factors(features_df, date_arg)
            return features_df

        self._compute_v481_new_factors = _compute_v481_and_extras
        try:
            from .v481_production_scorer import V481ProductionScorer
            results = V481ProductionScorer.predict_scores(self, stock_codes, date)
        finally:
            self._compute_v481_new_factors = original_v481_compute

        # Apply V485 ETF flagging
        etf_prefixes = ('510', '511', '512', '513', '515', '516', '517', '518',
                        '560', '561', '562', '563', '588',
                        '159', '160', '161', '162', '163', '164', '165',
                        '166', '167', '168', '169')
        for code, data in results.items():
            if code[:3] in etf_prefixes:
                data['etf'] = True
                data['etf_note'] = 'ETF预测信心较低(ICIR~0.31 vs A股~0.84)'
            else:
                data['etf'] = False

        return results
