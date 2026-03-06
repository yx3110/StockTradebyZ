#!/usr/bin/env python3
"""
V4.7.1 生产评分器 — 底层信号质量提升

继承V4.4的成熟9步pipeline, 仅做4项变更:
1. 模型目录: v471/
2. Bug 3修复: _get_market_return_20d() 统一使用000300.SH
3. 新特征加载: 财务质量因子 + daily_basic扩展 + 微观结构 + 反转 + 风险
4. predict_scores(): 在tech features之后插入新特征加载步骤
"""

import numpy as np
import pandas as pd
import sqlite3
import pickle
import joblib
from pathlib import Path
from typing import Dict, List, Optional

from .v44_production_scorer import V44ProductionScorer

import logging
logger = logging.getLogger(__name__)


class V471ProductionScorer(V44ProductionScorer):
    """V4.7.1 生产评分器 — V4.4底座 + Bug修复 + 17新特征"""

    def __init__(self, model_type: str = 'small_data'):
        self._v471_model_dir = Path(__file__).parent.parent.parent / 'ml_models' / 'trained_models' / 'v471'
        self._financial_cache = {}  # date -> DataFrame
        self._micro_cache = {}  # date -> DataFrame
        super().__init__(model_type=model_type)

    def _load_models(self):
        """覆盖加载方法, 使用 v471 模型目录"""
        self.model_dir = self._v471_model_dir
        if self.model_type == 'rolling':
            self._load_rolling_models()
        else:
            self._load_v471_model()

    def _load_v471_model(self):
        """加载v4.7.1模型"""
        model_files = list(self.model_dir.glob('v471_*.pkl'))
        if not model_files:
            print(f"V4.7.1 未找到模型文件: {self.model_dir}/v471_*.pkl")
            return

        latest = max(model_files, key=lambda f: f.stat().st_mtime)
        try:
            model_data = joblib.load(latest)
        except Exception:
            with open(latest, 'rb') as f:
                model_data = pickle.load(f)

        raw_models = model_data.get('models', {})
        self.models = {}
        self.weights = model_data.get('ensemble_weights', {})
        for target, target_data in raw_models.items():
            if isinstance(target_data, dict) and 'models' in target_data:
                self.models[target] = target_data['models']
                if not self.weights:
                    self.weights[f'label_{target}'] = target_data.get('weights', {})
            else:
                self.models[target] = target_data

        self.scaler = model_data.get('scaler')
        self.feature_cols = model_data.get('feature_names', model_data.get('feature_cols', []))
        self.market_feature_cols = model_data.get('market_features', model_data.get('market_feature_cols', []))
        self.target_weights = model_data.get('target_weights', {
            'label_3d': 0.20, 'label_5d': 0.25, 'label_10d': 0.35, 'label_15d': 0.20
        })

        # 元数据
        self.cascade = False
        self.cascade_feature_names = None
        self.dual_stream = False
        self.rank_normalized = False
        self.robust_zscore = model_data.get('robust_zscore', True)
        self.extra_features_from_daily_basic = model_data.get('extra_features_from_daily_basic', None)
        self.extra_tech_features = model_data.get('extra_features_from_tech_indicators', None)
        self.stock_rank_cols = model_data.get('stock_feature_cols', None)

        # V4.7.1 新增特征列表
        self.extra_features_financial = model_data.get('extra_features_financial', [])
        self.extra_features_microstructure = model_data.get('extra_features_microstructure', [])
        self.extra_features_reversal = model_data.get('extra_features_reversal', [])
        self.extra_features_risk = model_data.get('extra_features_risk', [])

        # Winsorize bounds
        raw_bounds = model_data.get('winsorize_bounds')
        if raw_bounds:
            self.winsorize_bounds = {k: tuple(v) for k, v in raw_bounds.items()} if isinstance(raw_bounds, dict) else raw_bounds

        # V4.4 组件
        self.bear_models = model_data.get('bear_models', {})
        self.isotonic_calibration = model_data.get('isotonic_calibration', {})

        # 全局分位数
        raw_quantiles = model_data.get('global_quantiles')
        if raw_quantiles is not None:
            self.global_quantiles = np.array(raw_quantiles)
        else:
            quantiles_path = self.model_dir / 'global_quantiles.npy'
            if quantiles_path.exists():
                self.global_quantiles = np.load(quantiles_path)

        wf = model_data.get('walk_forward_metrics', {})

        gq_status = "全局评分" if self.global_quantiles is not None else "截面评分"
        print(f"V4.7.1 模型加载完成: {list(self.models.keys())} [V4.4底座+Bug修复+17新特征+{gq_status}]")
        print(f"  模型文件: {latest.name}")
        print(f"  特征数: {len(self.feature_cols)}")
        print(f"  新特征: 财务{len(self.extra_features_financial)}+微观{len(self.extra_features_microstructure)}"
              f"+反转{len(self.extra_features_reversal)}+风险{len(self.extra_features_risk)}")
        print(f"  熊市专家: {list(self.bear_models.keys()) if self.bear_models else '无'}")
        print(f"  保序校准: {list(self.isotonic_calibration.keys()) if self.isotonic_calibration else '无'}")
        if wf:
            for t, m in wf.items():
                print(f"  WF {t}: ICIR={m.get('mean_icir', 0):.4f}±{m.get('std_icir', 0):.4f}")

    def _get_market_return_20d(self, date: str) -> Optional[float]:
        """Bug 3修复: 统一使用000300.SH (沪深300), 与训练侧一致"""
        if date in self._market_return_cache:
            return self._market_return_cache[date]

        conn = sqlite3.connect(self.db_path)
        try:
            query = """
            SELECT q.close
            FROM daily_quotes q
            JOIN securities s ON q.security_id = s.id
            WHERE s.code = '000300.SH' AND q.trade_date <= ?
            ORDER BY q.trade_date DESC
            LIMIT 21
            """
            df = pd.read_sql_query(query, conn, params=[date])
        finally:
            conn.close()

        if len(df) < 21:
            self._market_return_cache[date] = None
            return None

        ret = (df['close'].iloc[0] / df['close'].iloc[20]) - 1
        self._market_return_cache[date] = float(ret)
        return float(ret)

    def _load_financial_features(self, features_df: pd.DataFrame, date: str) -> pd.DataFrame:
        """加载财务质量因子 (point-in-time from financial_indicator)"""
        if not self.extra_features_financial:
            return features_df

        if date in self._financial_cache:
            df_fi = self._financial_cache[date]
        else:
            conn = sqlite3.connect(self.db_path)
            try:
                # 获取截至date的最新财报数据 (point-in-time)
                query = """
                SELECT s.code, fi.roe, fi.gross_margin, fi.current_ratio,
                       fi.assets_turn, fi.netprofit_yoy, fi.or_yoy
                FROM financial_indicator fi
                JOIN securities s ON fi.security_id = s.id
                WHERE fi.ann_date <= ? AND fi.ann_date IS NOT NULL AND fi.ann_date != ''
                AND fi.id IN (
                    SELECT MAX(fi2.id) FROM financial_indicator fi2
                    WHERE fi2.security_id = fi.security_id AND fi2.ann_date <= ?
                    AND fi2.ann_date IS NOT NULL AND fi2.ann_date != ''
                )
                """
                df_fi = pd.read_sql_query(query, conn, params=[date, date])
            finally:
                conn.close()
            self._financial_cache[date] = df_fi

        if len(df_fi) > 0:
            features_df = features_df.merge(df_fi, on='code', how='left')
        else:
            for col in self.extra_features_financial:
                if col not in features_df.columns:
                    features_df[col] = 0.0

        # 填充缺失值: 截面中位数
        for col in self.extra_features_financial:
            if col in features_df.columns:
                median_val = features_df[col].median()
                features_df[col] = features_df[col].fillna(median_val if not pd.isna(median_val) else 0.0)

        return features_df

    def _load_daily_basic_extra(self, features_df: pd.DataFrame, date: str) -> pd.DataFrame:
        """加载daily_basic扩展特征 (dv_ttm, turnover_rate_f, float_ratio)"""
        conn = sqlite3.connect(self.db_path)
        try:
            query = """
            SELECT s.code, db.dv_ttm, db.turnover_rate_f, db.circ_mv, db.total_mv
            FROM daily_basic db
            JOIN securities s ON db.security_id = s.id
            WHERE db.trade_date = ?
            """
            df_extra = pd.read_sql_query(query, conn, params=[date])
        finally:
            conn.close()

        if len(df_extra) > 0:
            df_extra['float_ratio'] = df_extra['circ_mv'] / df_extra['total_mv'].clip(lower=1e-8)
            df_extra.drop(columns=['circ_mv', 'total_mv'], inplace=True)
            features_df = features_df.merge(df_extra, on='code', how='left')

        for col in ['dv_ttm', 'turnover_rate_f', 'float_ratio']:
            if col in features_df.columns:
                median_val = features_df[col].median()
                features_df[col] = features_df[col].fillna(median_val if not pd.isna(median_val) else 0.0)
            else:
                features_df[col] = 0.0

        return features_df

    def _compute_microstructure_features(self, features_df: pd.DataFrame, date: str) -> pd.DataFrame:
        """计算微观结构/反转/风险因子 (从OHLCV)"""
        all_new_cols = (self.extra_features_microstructure +
                        self.extra_features_reversal +
                        self.extra_features_risk)
        if not all_new_cols:
            return features_df

        if date in self._micro_cache:
            df_micro = self._micro_cache[date]
        else:
            conn = sqlite3.connect(self.db_path)
            try:
                # 需要前30天数据用于滚动窗口
                query = """
                SELECT s.code, q.trade_date, q.close, q.volume, q.price_change_pct
                FROM daily_quotes q
                JOIN securities s ON q.security_id = s.id
                WHERE s.type = 'A股' AND q.trade_date <= ?
                AND q.trade_date >= date(?, '-40 days')
                ORDER BY s.code, q.trade_date
                """
                df_ohlcv = pd.read_sql_query(query, conn, params=[date, date])
            finally:
                conn.close()

            if len(df_ohlcv) == 0:
                for col in all_new_cols:
                    features_df[col] = 0.0
                return features_df

            results_list = []
            for code, grp in df_ohlcv.groupby('code'):
                grp = grp.sort_values('trade_date')
                if len(grp) < 5:
                    continue

                close = grp['close'].values.astype(float)
                volume = grp['volume'].values.astype(float)
                pct = pd.to_numeric(grp['price_change_pct'], errors='coerce').fillna(0).values.astype(float)

                row = {'code': code}

                # 微观结构
                if self.extra_features_microstructure:
                    # Amihud
                    abs_ret = np.abs(pct[-20:]) if len(pct) >= 20 else np.abs(pct)
                    vol_safe = np.where(volume[-20:] > 0, volume[-20:], 1e-8) if len(volume) >= 20 else np.where(volume > 0, volume, 1e-8)
                    row['amihud_illiquidity'] = float(np.mean(abs_ret / vol_safe))

                    # Volume-price correlation (10d)
                    n = min(10, len(close))
                    if n >= 5:
                        corr = np.corrcoef(close[-n:], volume[-n:])[0, 1]
                        row['volume_price_corr_10d'] = float(corr) if not np.isnan(corr) else 0.0
                    else:
                        row['volume_price_corr_10d'] = 0.0

                    # Max drawdown 20d
                    n_dd = min(20, len(close))
                    window = close[-n_dd:]
                    running_max = np.maximum.accumulate(window)
                    dd = (window - running_max) / np.where(running_max > 0, running_max, 1e-8)
                    row['max_drawdown_20d'] = float(np.min(dd))

                    # Up/down volume asymmetry
                    n_ud = min(10, len(pct))
                    up_vol = np.sum(volume[-n_ud:][pct[-n_ud:] > 0])
                    dn_vol = np.sum(volume[-n_ud:][pct[-n_ud:] < 0])
                    row['updown_volume_asymmetry'] = float(up_vol / max(dn_vol, 1e-8))

                # 反转因子
                if self.extra_features_reversal:
                    row['return_1d'] = float(close[-1] / close[-2] - 1) if len(close) >= 2 else 0.0
                    row['return_3d'] = float(close[-1] / close[-4] - 1) if len(close) >= 4 else 0.0

                # 风险因子
                if self.extra_features_risk:
                    n_risk = min(20, len(close))
                    daily_ret = np.diff(close[-n_risk:]) / close[-n_risk:-1]
                    if len(daily_ret) >= 5:
                        demeaned = daily_ret - np.mean(daily_ret)
                        row['idio_volatility_20d'] = float(np.std(demeaned))
                        neg_ret = daily_ret[daily_ret < 0]
                        row['downside_deviation_20d'] = float(np.std(neg_ret)) if len(neg_ret) > 0 else 0.0
                    else:
                        row['idio_volatility_20d'] = 0.0
                        row['downside_deviation_20d'] = 0.0

                results_list.append(row)

            df_micro = pd.DataFrame(results_list)
            self._micro_cache[date] = df_micro

        if len(df_micro) > 0:
            features_df = features_df.merge(df_micro, on='code', how='left')

        for col in all_new_cols:
            if col in features_df.columns:
                features_df[col] = features_df[col].fillna(0.0)
            else:
                features_df[col] = 0.0

        return features_df

    def predict_scores(self, stock_codes: List[str], date: str) -> Dict[str, Dict]:
        """V4.7.1 评分管线: V4.4基础 + 新特征加载"""
        # 日期格式标准化
        if isinstance(date, str) and len(date) == 8 and date.isdigit():
            date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        results = {}

        # Step 1: V4.3 基础特征 (全截面)
        features_df = self._get_features(stock_codes, date, load_full_cross_section=True)
        if features_df is not None and len(features_df) > 0:
            features_df = self._robust_zscore_normalize_features(features_df)
            features_df = self._load_daily_basic_features(features_df, date)
            features_df = self._load_technical_features(features_df, date)

            # V4.7.1: 新特征加载
            features_df = self._load_financial_features(features_df, date)
            features_df = self._load_daily_basic_extra(features_df, date)
            features_df = self._compute_microstructure_features(features_df, date)

            features_df = features_df[features_df['code'].isin(stock_codes)].copy()

        if features_df is None or len(features_df) == 0:
            for code in stock_codes:
                results[code] = {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0, 'pred_15d': 0,
                                 'exec_filter': 'no_data'}
            return results

        # 准备特征矩阵
        exclude_cols = {'code', 'trade_date'}
        if self.feature_cols:
            missing = [c for c in self.feature_cols if c not in features_df.columns]
            if missing:
                if len(missing) > len(self.feature_cols) * 0.3:
                    logger.warning(f"V4.7.1: {len(missing)}/{len(self.feature_cols)} 特征缺失: {missing[:5]}...")
                for col in missing:
                    features_df[col] = 0
            available_cols = self.feature_cols
        else:
            available_cols = [c for c in features_df.columns if c not in exclude_cols]

        X = features_df[available_cols].fillna(0).values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        codes = features_df['code'].tolist()

        # 独立预测 4 目标
        model_predictions_success = False
        predictions = {
            '3d': np.zeros(len(X)), '5d': np.zeros(len(X)),
            '10d': np.zeros(len(X)), '15d': np.zeros(len(X))
        }

        for target in ['3d', '5d', '10d', '15d']:
            if target not in self.models or not self.models[target]:
                continue
            target_pred = np.zeros(len(X))
            total_weight = 0
            success_count = 0

            for name, model in self.models[target].items():
                try:
                    pred = model.predict(X)
                    weight = self.weights.get(f'label_{target}', {}).get(name, 0.2)
                    target_pred += weight * pred
                    total_weight += weight
                    success_count += 1
                except Exception:
                    continue

            if total_weight > 0:
                target_pred /= total_weight
                predictions[target] = target_pred
                if success_count > 0:
                    model_predictions_success = True

        # 市况自适应目标权重
        regime_weights = self._get_regime_target_weights(date)
        if model_predictions_success:
            combined_pred = (
                regime_weights.get('label_3d', 0.20) * predictions['3d'] +
                regime_weights.get('label_5d', 0.25) * predictions['5d'] +
                regime_weights.get('label_10d', 0.35) * predictions['10d'] +
                regime_weights.get('label_15d', 0.20) * predictions['15d']
            )
        else:
            combined_pred = self._calculate_fallback_scores(features_df, available_cols)
            predictions = self._estimate_predictions_from_features(features_df, available_cols)

        scores = self._to_global_score(combined_pred)

        for i, code in enumerate(codes):
            results[code] = {
                'score': float(scores[i]),
                'pred_3d': float(predictions['3d'][i]) if i < len(predictions['3d']) else 0,
                'pred_5d': float(predictions['5d'][i]) if i < len(predictions['5d']) else 0,
                'pred_10d': float(predictions['10d'][i]) if i < len(predictions['10d']) else 0,
                'pred_15d': float(predictions.get('15d', np.zeros(1))[min(i, len(predictions.get('15d', [0]))-1)]) if '15d' in predictions else 0,
            }

        # Step 2: Module C — 熊市专家混合
        results = self._blend_bear_specialist(results, date, X, codes)

        # Step 3: Module A — 保序回归校准
        results = self._apply_isotonic_calibration(results, codes)

        # 校准后重新评分
        if model_predictions_success and self.isotonic_calibration:
            new_combined = np.zeros(len(codes))
            for i, code in enumerate(codes):
                if code in results:
                    r = results[code]
                    new_combined[i] = (
                        regime_weights.get('label_3d', 0.20) * r.get('pred_3d', 0) +
                        regime_weights.get('label_5d', 0.25) * r.get('pred_5d', 0) +
                        regime_weights.get('label_10d', 0.35) * r.get('pred_10d', 0) +
                        regime_weights.get('label_15d', 0.20) * r.get('pred_15d', 0)
                    )
            new_scores = self._to_global_score(new_combined)
            if len(new_scores) > 0:
                for i, code in enumerate(codes):
                    if code in results:
                        results[code]['score'] = float(new_scores[i])

        # Step 4: Module E — 可执行性过滤
        results = self._apply_executability_filters(results, date)

        for code in stock_codes:
            if code not in results:
                results[code] = {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0, 'pred_15d': 0,
                                 'exec_filter': 'no_data'}

        # Module F: 市况信息
        regime_info = self._get_regime_info(date)
        for code in results:
            results[code]['regime_info'] = regime_info

        return results

    def predict_scores_from_preloaded(self, stock_codes: List[str], date: str,
                                       features_df: Optional[pd.DataFrame]) -> Dict[str, Dict]:
        """V4.7.1: 使用预加载特征评分"""
        results = {}

        if features_df is None or len(features_df) == 0:
            for code in stock_codes:
                results[code] = {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0, 'pred_15d': 0,
                                 'exec_filter': 'no_data'}
            return results

        # robust z-score + daily_basic + tech features
        features_df = self._robust_zscore_normalize_features(features_df.copy())
        features_df = self._load_daily_basic_features(features_df, date)
        features_df = self._load_technical_features(features_df, date)

        # V4.7.1: 新特征加载
        features_df = self._load_financial_features(features_df, date)
        features_df = self._load_daily_basic_extra(features_df, date)
        features_df = self._compute_microstructure_features(features_df, date)

        mask = features_df['code'].isin(stock_codes)
        filtered_df = features_df[mask].copy()

        if len(filtered_df) == 0:
            for code in stock_codes:
                results[code] = {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0, 'pred_15d': 0,
                                 'exec_filter': 'no_data'}
            return results

        # 准备特征矩阵
        exclude_cols = {'code', 'trade_date'}
        if self.feature_cols:
            for col in self.feature_cols:
                if col not in filtered_df.columns:
                    filtered_df[col] = 0
            available_cols = self.feature_cols
        else:
            available_cols = [c for c in filtered_df.columns if c not in exclude_cols]

        X = filtered_df[available_cols].fillna(0).values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        codes = filtered_df['code'].tolist()

        model_predictions_success = False
        predictions = {
            '3d': np.zeros(len(X)), '5d': np.zeros(len(X)),
            '10d': np.zeros(len(X)), '15d': np.zeros(len(X))
        }

        for target in ['3d', '5d', '10d', '15d']:
            if target not in self.models or not self.models[target]:
                continue
            target_pred = np.zeros(len(X))
            total_weight = 0
            success_count = 0

            for name, model in self.models[target].items():
                try:
                    pred = model.predict(X)
                    weight = self.weights.get(f'label_{target}', {}).get(name, 0.2)
                    target_pred += weight * pred
                    total_weight += weight
                    success_count += 1
                except Exception:
                    continue

            if total_weight > 0:
                target_pred /= total_weight
                predictions[target] = target_pred
                if success_count > 0:
                    model_predictions_success = True

        regime_weights = self._get_regime_target_weights(date)
        if model_predictions_success:
            combined_pred = (
                regime_weights.get('label_3d', 0.20) * predictions['3d'] +
                regime_weights.get('label_5d', 0.25) * predictions['5d'] +
                regime_weights.get('label_10d', 0.35) * predictions['10d'] +
                regime_weights.get('label_15d', 0.20) * predictions['15d']
            )
        else:
            combined_pred = self._calculate_fallback_scores(filtered_df, available_cols)
            predictions = self._estimate_predictions_from_features(filtered_df, available_cols)

        scores = self._to_global_score(combined_pred)

        for i, code in enumerate(codes):
            results[code] = {
                'score': float(scores[i]),
                'pred_3d': float(predictions['3d'][i]) if i < len(predictions['3d']) else 0,
                'pred_5d': float(predictions['5d'][i]) if i < len(predictions['5d']) else 0,
                'pred_10d': float(predictions['10d'][i]) if i < len(predictions['10d']) else 0,
                'pred_15d': float(predictions.get('15d', np.zeros(1))[min(i, len(predictions.get('15d', [0]))-1)]) if '15d' in predictions else 0,
            }

        # Step 2-4: V4.4 pipeline (bear specialist, isotonic, executability)
        results = self._blend_bear_specialist(results, date, X, codes)
        results = self._apply_isotonic_calibration(results, codes)

        if model_predictions_success and self.isotonic_calibration:
            new_combined = np.zeros(len(codes))
            for i, code in enumerate(codes):
                if code in results:
                    r = results[code]
                    new_combined[i] = (
                        regime_weights.get('label_3d', 0.20) * r.get('pred_3d', 0) +
                        regime_weights.get('label_5d', 0.25) * r.get('pred_5d', 0) +
                        regime_weights.get('label_10d', 0.35) * r.get('pred_10d', 0) +
                        regime_weights.get('label_15d', 0.20) * r.get('pred_15d', 0)
                    )
            new_scores = self._to_global_score(new_combined)
            if len(new_scores) > 0:
                for i, code in enumerate(codes):
                    if code in results:
                        results[code]['score'] = float(new_scores[i])

        results = self._apply_executability_filters(results, date)

        for code in stock_codes:
            if code not in results:
                results[code] = {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0, 'pred_15d': 0,
                                 'exec_filter': 'no_data'}

        return results
