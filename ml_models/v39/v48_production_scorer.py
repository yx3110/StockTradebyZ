#!/usr/bin/env python3
"""
V4.8 生产评分器 — V4.7.2底座 + 12个新财务质量特征 + ListNet + 模型置信度加权

继承V4.7.2: V4.6管线(ICIR/Meta-Learner/Combined Isotonic) + V4.7.1特征(17个)
         + Bug 1/2/3修复 + 无小盘加成 + 000300.SH统一
新增三个创新轴:
  1. 12个新财务质量特征 (盈利质量/财务安全/增长动量)
  2. ListNet排名模型 + 回归/排名alpha融合
  3. 模型一致性置信度加权
"""

import numpy as np
import pandas as pd
import sqlite3
import pickle
import joblib
from pathlib import Path
from typing import Dict, List, Optional
from .v472_production_scorer import V472ProductionScorer

import logging
logger = logging.getLogger(__name__)


class V48ProductionScorer(V472ProductionScorer):
    """V4.8 生产评分器 — V4.7.2底座 + 财务质量扩展 + ListNet融合 + 置信度加权"""

    # V4.8新增的12个财务质量特征 (不与V4.7.1的17个重叠)
    EXTRA_FINANCIAL_QUALITY = [
        # Tier 1 - 盈利质量
        'netprofit_margin', 'ocf_to_opincome', 'salescash_to_or', 'roe_dt', 'fcfe_ps',
        # Tier 2 - 财务安全 (dv_ttm已在V4.7.1中, 不重复)
        'debt_to_eqt', 'ebit_to_interest', 'quick_ratio',
        # Tier 3 - 增长动量 (netprofit_yoy已在V4.7.1中, 不重复)
        'basic_eps_yoy', 'op_yoy', 'q_profit_yoy', 'netprofit_yoy_accel',
    ]

    def __init__(self, model_type: str = 'small_data'):
        self._v48_model_dir = Path(__file__).parent.parent.parent / 'ml_models' / 'trained_models' / 'v48'
        self._fq_cache = {}  # date -> DataFrame (financial quality)
        super().__init__(model_type=model_type)

    def _load_models(self):
        """覆盖加载方法, 使用 v48 模型目录"""
        self.model_dir = self._v48_model_dir
        self._v472_model_dir = self._v48_model_dir  # 让V4.7.2父类方法也用v48目录
        if self.model_type == 'rolling':
            self._load_rolling_models()
        else:
            self._load_v48_model()

    def _load_v48_model(self):
        """加载v4.8模型 — V4.7.2结构 + V4.8扩展"""
        model_files = list(self.model_dir.glob('v48_*.pkl'))
        if not model_files:
            print(f"V4.8 未找到模型文件: {self.model_dir}/v48_*.pkl")
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

        # V4.7.1 特征列表 (继承)
        self.extra_features_financial = model_data.get('extra_features_financial', [])
        self.extra_features_microstructure = model_data.get('extra_features_microstructure', [])
        self.extra_features_reversal = model_data.get('extra_features_reversal', [])
        self.extra_features_risk = model_data.get('extra_features_risk', [])

        # V4.8 新增财务质量特征
        self.extra_financial_quality_features = model_data.get('extra_financial_quality_features',
                                                                self.EXTRA_FINANCIAL_QUALITY)

        # Winsorize bounds
        raw_bounds = model_data.get('winsorize_bounds')
        if raw_bounds:
            self.winsorize_bounds = {k: tuple(v) for k, v in raw_bounds.items()} if isinstance(raw_bounds, dict) else raw_bounds

        # V4.4 组件
        self.bear_models = model_data.get('bear_models', {})
        self.isotonic_calibration = model_data.get('isotonic_calibration', {})

        # V4.6 组件
        self.combined_isotonic = model_data.get('combined_isotonic')
        self.meta_learner = model_data.get('meta_learner')
        self.meta_feature_names = model_data.get('meta_feature_names')

        # V4.8 新组件
        self.ranking_alpha = model_data.get('ranking_alpha', 0.5)
        self.has_ranking_models = model_data.get('has_ranking_models', False)
        self.use_confidence_weighting = model_data.get('use_confidence_weighting', True)

        # 全局分位数
        raw_quantiles = model_data.get('global_quantiles')
        if raw_quantiles is not None:
            self.global_quantiles = np.array(raw_quantiles)
        else:
            quantiles_path = self.model_dir / 'global_quantiles.npy'
            if quantiles_path.exists():
                self.global_quantiles = np.load(quantiles_path)

        # 投资建议阈值
        self.recommendation_thresholds = model_data.get('recommendation_thresholds')
        if not self.recommendation_thresholds:
            rec_path = self.model_dir / 'recommendation_thresholds.json'
            if rec_path.exists():
                import json as _json
                with open(rec_path, 'r') as f:
                    self.recommendation_thresholds = _json.load(f)

        wf = model_data.get('walk_forward_metrics', {})

        # V4.6.1: ICIR权重clip
        self.weights = self._clip_icir_weights(self.weights)

        gq_status = "全局评分" if self.global_quantiles is not None else "截面评分"
        meta_status = "有" if self.meta_learner is not None else "无"
        ciso_status = "有" if self.combined_isotonic is not None else "无"
        rank_status = "有" if self.has_ranking_models else "无"
        n_v471_feat = (len(self.extra_features_financial) + len(self.extra_features_microstructure) +
                       len(self.extra_features_reversal) + len(self.extra_features_risk))
        print(f"V4.8 模型加载完成: {list(self.models.keys())} [V4.7.2底座+ListNet+财务质量+{gq_status}]")
        print(f"  模型文件: {latest.name}")
        print(f"  特征数: {len(self.feature_cols)}")
        print(f"  V4.7.1特征: {n_v471_feat}个 | V4.8新增财务: {len(self.extra_financial_quality_features)}个")
        print(f"  排名模型: {rank_status} (alpha={self.ranking_alpha:.2f})")
        print(f"  置信度加权: {'启用' if self.use_confidence_weighting else '禁用'}")
        print(f"  Combined Isotonic: {ciso_status} | Meta-Learner: {meta_status}")
        if wf:
            for t, m in wf.items():
                print(f"  WF {t}: ICIR={m.get('mean_icir', 0):.4f}±{m.get('std_icir', 0):.4f}")

    def _load_financial_quality_features(self, features_df: pd.DataFrame, date: str) -> pd.DataFrame:
        """加载V4.8新增的12个财务质量特征 (point-in-time, 不与V4.7.1重叠)"""
        if not self.extra_financial_quality_features:
            return features_df

        if date in self._fq_cache:
            df_cached = self._fq_cache[date]
            if df_cached is not None and len(df_cached) > 0:
                features_df = features_df.merge(df_cached, on='code', how='left')
            for col in self.extra_financial_quality_features:
                if col in features_df.columns:
                    median_val = features_df[col].median()
                    features_df[col] = features_df[col].fillna(median_val if not pd.isna(median_val) else 0.0)
                else:
                    features_df[col] = 0.0
            return features_df

        conn = sqlite3.connect(self.db_path)
        try:
            # 盈利质量 + 财务安全 + 增长动量 (from financial_indicator, point-in-time)
            fi_cols = [c for c in self.extra_financial_quality_features
                       if c not in ('dv_ttm', 'netprofit_yoy_accel')]
            # Map feature names to DB column names (most are direct)
            fi_db_cols = [c for c in fi_cols if c != 'netprofit_yoy_accel']

            fi_query = f"""
            SELECT s.code, {', '.join(f'fi.{c}' for c in fi_db_cols)}
            FROM financial_indicator fi
            JOIN securities s ON fi.security_id = s.id
            WHERE fi.ann_date <= ? AND fi.ann_date IS NOT NULL AND fi.ann_date != ''
            AND fi.id IN (
                SELECT MAX(fi2.id) FROM financial_indicator fi2
                WHERE fi2.security_id = fi.security_id AND fi2.ann_date <= ?
                AND fi2.ann_date IS NOT NULL AND fi2.ann_date != ''
            )
            """
            df_fi = pd.read_sql_query(fi_query, conn, params=[date, date])
        finally:
            conn.close()

        self._fq_cache[date] = df_fi

        if len(df_fi) > 0:
            features_df = features_df.merge(df_fi, on='code', how='left')

        # Fill missing with cross-sectional median
        for col in self.extra_financial_quality_features:
            if col in features_df.columns:
                median_val = features_df[col].median()
                features_df[col] = features_df[col].fillna(median_val if not pd.isna(median_val) else 0.0)
            else:
                features_df[col] = 0.0

        return features_df

    def _compute_confidence_weight(self, model_preds_list: List[np.ndarray]) -> np.ndarray:
        """计算模型一致性置信度权重

        多模型预测值的标准差 → 一致性度量
        标准差越小 → 置信度越高 → 分数保持
        标准差越大 → 置信度越低 → 分数衰减
        """
        if len(model_preds_list) < 3:
            return np.ones(model_preds_list[0].shape[0])

        pred_stack = np.stack(model_preds_list, axis=0)
        model_std = np.std(pred_stack, axis=0)

        # confidence = 1 / (1 + std * 10): std=0 → 1.0, std=0.01 → 0.91, std=0.05 → 0.67
        confidence = 1.0 / (1.0 + model_std * 10)

        # weight = 0.7 + 0.3 * confidence: confidence=1→1.0, confidence=0.5→0.85, confidence=0→0.7
        weight = 0.7 + 0.3 * confidence
        return weight

    def predict_scores(self, stock_codes: List[str], date: str) -> Dict[str, Dict]:
        """V4.8 评分管线:
        Step 1: V4.3 基础特征 + V4.7.1 17特征 + V4.8 12财务质量特征
        Step 2: 5-7模型×4目标 (3d=5回归, 5d/10d/15d=5回归+LambdaRank+ListNet)
        Step 2.5: 回归/排名alpha融合
        Step 3: Meta-learner → Combined isotonic
        Step 3.5: 模型置信度加权
        Step 4: 全局百分位评分
        Step 5-7: 熊市混合 → 保序校准 → 可执行性过滤
        (无小盘加成 — 继承V4.7.2)
        """
        results = {}

        # Step 1: 基础特征 + V4.7.1特征 + V4.8财务质量特征
        features_df = self._get_features(stock_codes, date, load_full_cross_section=True)
        if features_df is not None and len(features_df) > 0:
            features_df = self._robust_zscore_normalize_features(features_df)
            features_df = self._load_daily_basic_features(features_df, date)
            features_df = self._load_technical_features(features_df, date)
            # V4.7.1 特征 (继承自V4.7.2)
            features_df = self._load_financial_features(features_df, date)
            features_df = self._load_daily_basic_extra(features_df, date)
            features_df = self._compute_microstructure_features(features_df, date)
            # V4.8 新增财务质量特征
            features_df = self._load_financial_quality_features(features_df, date)
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
                    logger.warning(f"V4.8: {len(missing)}/{len(self.feature_cols)} 特征缺失: {missing[:5]}...")
                for col in missing:
                    features_df[col] = 0
            available_cols = self.feature_cols
        else:
            available_cols = [c for c in features_df.columns if c not in exclude_cols]

        X = features_df[available_cols].fillna(0).values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        codes = features_df['code'].tolist()

        # Step 2: 模型预测 + 回归/排名融合
        model_predictions_success = False
        predictions = {
            '3d': np.zeros(len(X)), '5d': np.zeros(len(X)),
            '10d': np.zeros(len(X)), '15d': np.zeros(len(X))
        }
        all_model_preds = []  # For confidence weighting

        for target in ['3d', '5d', '10d', '15d']:
            if target not in self.models or not self.models[target]:
                continue

            reg_pred = np.zeros(len(X))
            reg_weight = 0
            rank_pred = np.zeros(len(X))
            rank_weight = 0
            target_model_preds = []

            for name, model in self.models[target].items():
                try:
                    if name == 'xgb':
                        import xgboost as xgb_lib
                        pred = model.predict(xgb_lib.DMatrix(X))
                    else:
                        pred = model.predict(X)

                    weight = self.weights.get(f'label_{target}', {}).get(name, 0.2)
                    target_model_preds.append(pred)

                    if name in ('lgb_rank', 'lgb_listnet'):
                        rank_pred += weight * pred
                        rank_weight += weight
                    else:
                        reg_pred += weight * pred
                        reg_weight += weight
                except Exception:
                    continue

            # Fuse regression and ranking predictions
            if reg_weight > 0:
                reg_pred /= reg_weight
            if rank_weight > 0:
                rank_pred /= rank_weight
                # Rescale rank预测到回归模型尺度 (LambdaRank/ListNet输出排名分数,尺度不同)
                if reg_weight > 0:
                    rp_std = max(np.std(rank_pred), 1e-8)
                    rank_pred = (rank_pred - np.mean(rank_pred)) / rp_std * max(np.std(reg_pred), 1e-8) + np.mean(reg_pred)

            if reg_weight > 0 and rank_weight > 0 and self.has_ranking_models:
                alpha = self.ranking_alpha
                target_pred = alpha * reg_pred + (1 - alpha) * rank_pred
            elif reg_weight > 0:
                target_pred = reg_pred
            elif rank_weight > 0:
                target_pred = rank_pred
            else:
                target_pred = np.zeros(len(X))

            predictions[target] = target_pred
            all_model_preds.extend(target_model_preds)

            if len(target_model_preds) > 0:
                model_predictions_success = True

        # Step 3: Meta-learner 或 ICIR权重加权
        regime_weights = self._get_regime_target_weights(date)
        if model_predictions_success:
            meta_pred = self._get_meta_predictions(X)
            if meta_pred is not None:
                combined_pred = meta_pred
            else:
                combined_pred = (
                    regime_weights.get('label_3d', 0.20) * predictions['3d'] +
                    regime_weights.get('label_5d', 0.25) * predictions['5d'] +
                    regime_weights.get('label_10d', 0.35) * predictions['10d'] +
                    regime_weights.get('label_15d', 0.20) * predictions['15d']
                )
        else:
            combined_pred = self._calculate_fallback_scores(features_df, available_cols)
            predictions = self._estimate_predictions_from_features(features_df, available_cols)

        # Combined isotonic calibration
        combined_pred = self._apply_combined_isotonic(combined_pred)

        # Step 3.5: 模型置信度加权
        confidence_weight = None
        if self.use_confidence_weighting and len(all_model_preds) >= 3:
            confidence_weight = self._compute_confidence_weight(all_model_preds)
            combined_pred = combined_pred * confidence_weight

        # Step 4: 全局百分位评分
        scores = self._to_global_score(combined_pred)

        for i, code in enumerate(codes):
            results[code] = {
                'score': float(scores[i]),
                'pred_3d': float(predictions['3d'][i]) if i < len(predictions['3d']) else 0,
                'pred_5d': float(predictions['5d'][i]) if i < len(predictions['5d']) else 0,
                'pred_10d': float(predictions['10d'][i]) if i < len(predictions['10d']) else 0,
                'pred_15d': float(predictions.get('15d', np.zeros(1))[min(i, len(predictions.get('15d', [0]))-1)]) if '15d' in predictions else 0,
            }

        # Step 5: 熊市专家混合
        results = self._blend_bear_specialist(results, date, X, codes)

        # Step 6: Per-target 保序回归校准
        results = self._apply_isotonic_calibration(results, codes)

        # 校准后重新计算综合分数
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

            new_combined = self._apply_combined_isotonic(new_combined)
            if confidence_weight is not None:
                new_combined = new_combined * confidence_weight
            new_scores = self._to_global_score(new_combined)
            if len(new_scores) > 0:
                for i, code in enumerate(codes):
                    if code in results:
                        results[code]['score'] = float(new_scores[i])

        # Step 7: 增强可执行性过滤
        results = self._apply_enhanced_executability_filters(results, date)

        # (无小盘加成 — 继承V4.7.2的no-op)

        # 补全缺失code
        for code in stock_codes:
            if code not in results:
                results[code] = {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0, 'pred_15d': 0,
                                 'exec_filter': 'no_data'}

        # 市况信息
        regime_info = self._get_regime_info(date)
        for code in results:
            results[code]['regime_info'] = regime_info

        return results

    def predict_scores_from_preloaded(self, stock_codes: List[str], date: str,
                                       features_df: Optional[pd.DataFrame]) -> Dict[str, Dict]:
        """使用预加载特征评分 — V4.8 版本 (批量评分用)"""
        results = {}

        if features_df is None or len(features_df) == 0:
            for code in stock_codes:
                results[code] = {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0, 'pred_15d': 0,
                                 'exec_filter': 'no_data'}
            return results

        # 特征加载: V4.3基础 + V4.7.1扩展 + V4.8财务质量
        features_df = self._robust_zscore_normalize_features(features_df.copy())
        features_df = self._load_daily_basic_features(features_df, date)
        features_df = self._load_technical_features(features_df, date)
        features_df = self._load_financial_features(features_df, date)
        features_df = self._load_daily_basic_extra(features_df, date)
        features_df = self._compute_microstructure_features(features_df, date)
        features_df = self._load_financial_quality_features(features_df, date)

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
        all_model_preds = []

        for target in ['3d', '5d', '10d', '15d']:
            if target not in self.models or not self.models[target]:
                continue

            reg_pred = np.zeros(len(X))
            reg_weight = 0
            rank_pred = np.zeros(len(X))
            rank_weight = 0
            target_model_preds = []

            for name, model in self.models[target].items():
                try:
                    if name == 'xgb':
                        import xgboost as xgb_lib
                        pred = model.predict(xgb_lib.DMatrix(X))
                    else:
                        pred = model.predict(X)

                    weight = self.weights.get(f'label_{target}', {}).get(name, 0.2)
                    target_model_preds.append(pred)

                    if name in ('lgb_rank', 'lgb_listnet'):
                        rank_pred += weight * pred
                        rank_weight += weight
                    else:
                        reg_pred += weight * pred
                        reg_weight += weight
                except Exception:
                    continue

            if reg_weight > 0:
                reg_pred /= reg_weight
            if rank_weight > 0:
                rank_pred /= rank_weight

            if reg_weight > 0 and rank_weight > 0 and self.has_ranking_models:
                alpha = self.ranking_alpha
                target_pred = alpha * reg_pred + (1 - alpha) * rank_pred
            elif reg_weight > 0:
                target_pred = reg_pred
            elif rank_weight > 0:
                target_pred = rank_pred
            else:
                target_pred = np.zeros(len(X))

            predictions[target] = target_pred
            all_model_preds.extend(target_model_preds)

            if len(target_model_preds) > 0:
                model_predictions_success = True

        # Fusion
        regime_weights = self._get_regime_target_weights(date)
        if model_predictions_success:
            meta_pred = self._get_meta_predictions(X)
            if meta_pred is not None:
                combined_pred = meta_pred
            else:
                combined_pred = (
                    regime_weights.get('label_3d', 0.20) * predictions['3d'] +
                    regime_weights.get('label_5d', 0.25) * predictions['5d'] +
                    regime_weights.get('label_10d', 0.35) * predictions['10d'] +
                    regime_weights.get('label_15d', 0.20) * predictions['15d']
                )
        else:
            combined_pred = self._calculate_fallback_scores(filtered_df, available_cols)
            predictions = self._estimate_predictions_from_features(filtered_df, available_cols)

        combined_pred = self._apply_combined_isotonic(combined_pred)

        # Confidence weighting
        confidence_weight = None
        if self.use_confidence_weighting and len(all_model_preds) >= 3:
            confidence_weight = self._compute_confidence_weight(all_model_preds)
            combined_pred = combined_pred * confidence_weight

        scores = self._to_global_score(combined_pred)

        for i, code in enumerate(codes):
            results[code] = {
                'score': float(scores[i]),
                'pred_3d': float(predictions['3d'][i]) if i < len(predictions['3d']) else 0,
                'pred_5d': float(predictions['5d'][i]) if i < len(predictions['5d']) else 0,
                'pred_10d': float(predictions['10d'][i]) if i < len(predictions['10d']) else 0,
                'pred_15d': float(predictions.get('15d', np.zeros(1))[min(i, len(predictions.get('15d', [0]))-1)]) if '15d' in predictions else 0,
            }

        # Post-processing pipeline
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

            new_combined = self._apply_combined_isotonic(new_combined)
            if confidence_weight is not None:
                new_combined = new_combined * confidence_weight
            new_scores = self._to_global_score(new_combined)
            if len(new_scores) > 0:
                for i, code in enumerate(codes):
                    if code in results:
                        results[code]['score'] = float(new_scores[i])

        results = self._apply_enhanced_executability_filters(results, date)

        for code in stock_codes:
            if code not in results:
                results[code] = {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0, 'pred_15d': 0,
                                 'exec_filter': 'no_data'}

        return results
