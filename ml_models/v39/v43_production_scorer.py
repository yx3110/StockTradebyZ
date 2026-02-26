#!/usr/bin/env python3
"""
V4.3 生产评分器
基于 V3.96 Robust Z-Score + Industry-Excess Labels 架构

改进点 (相比 V3.96):
  - 扩展特征 49→59: KDJ/MACD/Bollinger/ATR/高低位/上影线/偏度
  - 4 目标: 3d/5d/10d/15d (新增 15d)
  - 等权集成 (更稳健)
  - 强正则化训练 (min_data_in_leaf=500, L1=1.0, L2=5.0)
  - Walk-Forward 验证
  - 样本加权 (涨跌停/极端值降权)
"""

import json
import pickle
import joblib
import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
from .v395_production_scorer import V395ProductionScorer


class V43ProductionScorer(V395ProductionScorer):
    """V4.3 生产评分器 — 扩展特征 + 强正则 + 4目标 + 等权集成"""

    def __init__(self, model_type: str = 'small_data'):
        self._v43_model_dir = Path(__file__).parent.parent.parent / 'ml_models' / 'trained_models' / 'v43'
        self.extra_tech_features = None
        self._tech_feature_cache = {}  # date -> DataFrame
        super().__init__(model_type=model_type)

    def _load_models(self):
        """覆盖加载方法, 使用 v43 模型目录"""
        self.model_dir = self._v43_model_dir
        if self.model_type == 'rolling':
            self._load_rolling_models()
        else:
            self._load_v43_model()

    def _load_v43_model(self):
        """加载v4.3模型"""
        model_files = list(self.model_dir.glob('v43_*.pkl'))
        if not model_files:
            print(f"V4.3 未找到模型文件: {self.model_dir}/v43_*.pkl")
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
            'label_3d': 0.25, 'label_5d': 0.30, 'label_10d': 0.25, 'label_15d': 0.20
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

        # Winsorization bounds
        raw_bounds = model_data.get('winsorize_bounds')
        if raw_bounds and self.feature_cols:
            if isinstance(raw_bounds, dict):
                self.winsorize_bounds = raw_bounds
            elif isinstance(raw_bounds, list) and len(raw_bounds) == len(self.feature_cols):
                self.winsorize_bounds = {
                    col: bounds for col, bounds in zip(self.feature_cols, raw_bounds)
                    if bounds[0] != bounds[1]
                }
            else:
                self.winsorize_bounds = None
        else:
            self.winsorize_bounds = None

        # 全局分位数
        raw_quantiles = model_data.get('global_quantiles')
        if raw_quantiles is not None:
            self.global_quantiles = np.array(raw_quantiles)
        else:
            quantiles_path = self.model_dir / 'global_quantiles.npy'
            if quantiles_path.exists():
                self.global_quantiles = np.load(quantiles_path)

        # Walk-forward 结果
        wf = model_data.get('walk_forward_metrics', {})

        gq_status = f"全局评分" if self.global_quantiles is not None else "截面评分"
        print(f"V4.3 模型加载完成: {list(self.models.keys())} [robust_zscore+4targets+{gq_status}]")
        print(f"  模型文件: {latest.name}")
        if wf:
            for t, m in wf.items():
                print(f"  WF {t}: ICIR={m.get('mean_icir', 0):.4f}±{m.get('std_icir', 0):.4f}")

    def _load_technical_features(self, features_df: pd.DataFrame, date: str) -> pd.DataFrame:
        """从 technical_indicators + daily_quotes 加载技术指标特征"""
        if not self.extra_tech_features:
            return features_df

        # 使用缓存
        if date in self._tech_feature_cache:
            df_tech = self._tech_feature_cache[date]
        else:
            conn = sqlite3.connect(self.db_path)
            try:
                codes = features_df['code'].tolist()
                placeholders = ','.join(['?' for _ in codes])
                query = f"""
                SELECT s.code,
                       ti.kdj_k, ti.kdj_j, ti.macd_dif, ti.macd_dea, ti.macd_macd,
                       ti.boll_upper, ti.boll_lower, ti.atr_14,
                       q.close, q.high, q.low
                FROM technical_indicators ti
                JOIN securities s ON ti.security_id = s.id
                JOIN daily_quotes q ON q.security_id = s.id AND q.trade_date = ti.trade_date
                WHERE s.code IN ({placeholders}) AND ti.trade_date = ?
                """
                df_tech = pd.read_sql_query(query, conn, params=codes + [date])
            finally:
                conn.close()
            self._tech_feature_cache[date] = df_tech

        if len(df_tech) == 0:
            for col in self.extra_tech_features:
                features_df[col] = 0.0
            return features_df

        # 计算衍生特征
        df_tech = df_tech.copy()
        df_tech['macd_hist'] = df_tech['macd_macd']
        boll_range = df_tech['boll_upper'] - df_tech['boll_lower']
        df_tech['boll_position'] = np.where(
            boll_range > 1e-6,
            (df_tech['close'] - df_tech['boll_lower']) / boll_range,
            0.5
        )
        df_tech['atr_14_pct'] = np.where(
            df_tech['close'] > 0,
            df_tech['atr_14'] / df_tech['close'],
            0.0
        )
        hl_range = df_tech['high'] - df_tech['low']
        df_tech['high_low_position'] = np.where(
            hl_range > 1e-6,
            (df_tech['close'] - df_tech['low']) / hl_range,
            0.5
        )
        df_tech['upper_shadow_ratio'] = np.where(
            hl_range > 1e-6,
            (df_tech['high'] - df_tech['close']) / hl_range,
            0.0
        )

        # Merge tech features
        tech_cols = ['code', 'kdj_k', 'kdj_j', 'macd_dif', 'macd_dea', 'macd_hist',
                     'boll_position', 'atr_14_pct', 'high_low_position', 'upper_shadow_ratio']
        features_df = features_df.merge(df_tech[tech_cols], on='code', how='left')

        # return_skewness_proxy
        if 'max_pct_change_5d' in features_df.columns and 'avg_pct_change_5d' in features_df.columns:
            vol = features_df.get('volatility_10d', pd.Series(1e-6, index=features_df.index))
            vol = vol.replace(0, np.nan).fillna(1e-6)
            features_df['return_skewness_proxy'] = (
                (features_df['max_pct_change_5d'] - features_df['avg_pct_change_5d']) / vol
            ).clip(-10, 10).fillna(0)
        else:
            features_df['return_skewness_proxy'] = 0.0

        # 填充缺失
        for col in self.extra_tech_features:
            if col in features_df.columns:
                features_df[col] = features_df[col].fillna(features_df[col].median() if features_df[col].notna().any() else 0)
            else:
                features_df[col] = 0.0

        return features_df

    def predict_scores(self, stock_codes: List[str], date: str) -> Dict[str, Dict]:
        """预测股票评分 — V4.3: 全截面 + robust_zscore + daily_basic + tech features + 4目标"""
        results = {}

        # 加载全截面 → robust z-score → daily_basic → tech features → 过滤
        features_df = self._get_features(stock_codes, date, load_full_cross_section=True)
        if features_df is not None and len(features_df) > 0:
            features_df = self._robust_zscore_normalize_features(features_df)
            features_df = self._load_daily_basic_features(features_df, date)
            features_df = self._load_technical_features(features_df, date)
            features_df = features_df[features_df['code'].isin(stock_codes)].copy()

        if features_df is None or len(features_df) == 0:
            for code in stock_codes:
                results[code] = {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0, 'pred_15d': 0}
            return results

        # 准备特征矩阵
        exclude_cols = {'code', 'trade_date'}
        if self.feature_cols:
            missing = [c for c in self.feature_cols if c not in features_df.columns]
            if missing:
                if len(missing) > len(self.feature_cols) * 0.3:
                    logger.warning(f"⚠️ V4.3: {len(missing)}/{len(self.feature_cols)} 特征缺失: {missing[:5]}...")
                for col in missing:
                    features_df[col] = 0
            available_cols = self.feature_cols
        else:
            available_cols = [c for c in features_df.columns if c not in exclude_cols]

        X = features_df[available_cols].fillna(0).values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # 应用训练时保存的Winsorization bounds
        X = self._apply_winsorization(X, available_cols)

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

        # 融合 4 目标
        if model_predictions_success:
            combined_pred = (
                self.target_weights.get('label_3d', 0.25) * predictions['3d'] +
                self.target_weights.get('label_5d', 0.30) * predictions['5d'] +
                self.target_weights.get('label_10d', 0.25) * predictions['10d'] +
                self.target_weights.get('label_15d', 0.20) * predictions['15d']
            )
        else:
            combined_pred = self._calculate_fallback_scores(features_df, available_cols)
            predictions = self._estimate_predictions_from_features(features_df, available_cols)

        # 全局百分位评分 (or 截面百分位 fallback)
        scores = self._to_global_score(combined_pred)

        # 构建结果
        codes = features_df['code'].tolist()
        for i, code in enumerate(codes):
            results[code] = {
                'score': float(scores[i]),
                'pred_3d': float(predictions['3d'][i]) if i < len(predictions['3d']) else 0,
                'pred_5d': float(predictions['5d'][i]) if i < len(predictions['5d']) else 0,
                'pred_10d': float(predictions['10d'][i]) if i < len(predictions['10d']) else 0,
                'pred_15d': float(predictions.get('15d', np.zeros(1))[min(i, len(predictions.get('15d', [0]))-1)]) if '15d' in predictions else 0,
            }

        for code in stock_codes:
            if code not in results:
                results[code] = {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0, 'pred_15d': 0}

        return results

    def predict_scores_from_preloaded(self, stock_codes: List[str], date: str,
                                       features_df: Optional[pd.DataFrame]) -> Dict[str, Dict]:
        """使用预加载特征评分 — V4.3 版本"""
        results = {}

        if features_df is None or len(features_df) == 0:
            for code in stock_codes:
                results[code] = {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0, 'pred_15d': 0}
            return results

        # robust z-score + daily_basic + tech features
        features_df = self._robust_zscore_normalize_features(features_df.copy())
        features_df = self._load_daily_basic_features(features_df, date)
        features_df = self._load_technical_features(features_df, date)

        mask = features_df['code'].isin(stock_codes)
        filtered_df = features_df[mask].copy()

        if len(filtered_df) == 0:
            for code in stock_codes:
                results[code] = {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0, 'pred_15d': 0}
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

        # 应用训练时保存的Winsorization bounds
        X = self._apply_winsorization(X, available_cols)

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

        if model_predictions_success:
            combined_pred = (
                self.target_weights.get('label_3d', 0.25) * predictions['3d'] +
                self.target_weights.get('label_5d', 0.30) * predictions['5d'] +
                self.target_weights.get('label_10d', 0.25) * predictions['10d'] +
                self.target_weights.get('label_15d', 0.20) * predictions['15d']
            )
        else:
            combined_pred = self._calculate_fallback_scores(filtered_df, available_cols)
            predictions = self._estimate_predictions_from_features(filtered_df, available_cols)

        scores = self._to_global_score(combined_pred)

        codes = filtered_df['code'].tolist()
        for i, code in enumerate(codes):
            results[code] = {
                'score': float(scores[i]),
                'pred_3d': float(predictions['3d'][i]) if i < len(predictions['3d']) else 0,
                'pred_5d': float(predictions['5d'][i]) if i < len(predictions['5d']) else 0,
                'pred_10d': float(predictions['10d'][i]) if i < len(predictions['10d']) else 0,
                'pred_15d': float(predictions.get('15d', np.zeros(1))[min(i, len(predictions.get('15d', [0]))-1)]) if '15d' in predictions else 0,
            }

        for code in stock_codes:
            if code not in results:
                results[code] = {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0, 'pred_15d': 0}

        return results

    def preload_feature_cache(self, dates: List[str]) -> Dict[str, pd.DataFrame]:
        """批量预加载特征缓存 + 技术指标"""
        # 先调用父类预加载基础特征
        result = super().preload_feature_cache(dates)

        if not self.extra_tech_features:
            return result

        # 批量预加载技术指标
        valid_dates = [d for d in dates if result.get(d) is not None]
        if not valid_dates:
            return result

        conn = sqlite3.connect(self.db_path)
        try:
            placeholders = ','.join(['?' for _ in valid_dates])
            query = f"""
            SELECT s.code, ti.trade_date,
                   ti.kdj_k, ti.kdj_j, ti.macd_dif, ti.macd_dea, ti.macd_macd,
                   ti.boll_upper, ti.boll_lower, ti.atr_14,
                   q.close, q.high, q.low
            FROM technical_indicators ti
            JOIN securities s ON ti.security_id = s.id
            JOIN daily_quotes q ON q.security_id = s.id AND q.trade_date = ti.trade_date
            WHERE ti.trade_date IN ({placeholders}) AND s.type = 'A股'
            """
            df_tech_all = pd.read_sql_query(query, conn, params=valid_dates)
        finally:
            conn.close()

        if df_tech_all.empty:
            return result

        # 按日期缓存
        for date, date_df in df_tech_all.groupby('trade_date'):
            self._tech_feature_cache[date] = date_df.copy()

        total_tech = len(df_tech_all)
        print(f"V4.3技术指标预加载完成: {len(valid_dates)}天, {total_tech}条记录")

        return result
