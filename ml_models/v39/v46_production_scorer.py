#!/usr/bin/env python3
"""
V4.6.1 生产评分器
基于 V4.6 模型 + scorer后处理调优 (不重训练)

V4.6.1 调整 (修复小盘过度偏移):
  - 移除小盘评分加成 (训练层1.5x已足够, scorer层+5%是double dipping)
  - 流动性折扣阈值 2.0% → 1.5%, 底线 0.1 → 0.2
  - ICIR权重clip到[0.08, 0.50]后重归一化, 防止单模型主导

V4.6 原改进点 (模型层, 保留):
  1A. ICIR最大化集成权重 (替代IC+单调性加权)
  1B. 小盘加权训练 (训练层已处理)
  1C. Combined-Score Isotonic (组合分数保序校准)
  1D. Stacking Meta-Learner (Ridge, 5模型×4目标meta features)
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


class V46ProductionScorer(V44ProductionScorer):
    """V4.6.1 生产评分器 — V4.6模型 + scorer调优 (移除小盘加成/流动性1.5%/ICIR clip)"""

    def __init__(self, model_type: str = 'small_data'):
        self._v46_model_dir = Path(__file__).parent.parent.parent / 'ml_models' / 'trained_models' / 'v46'
        self.combined_isotonic = None
        self.meta_learner = None
        self.meta_feature_names = None
        self._daily_basic_cache = {}  # date -> {code: circ_mv}
        super().__init__(model_type=model_type)

    def _load_models(self):
        """覆盖加载方法, 使用 v46 模型目录"""
        self.model_dir = self._v46_model_dir
        if self.model_type == 'rolling':
            self._load_rolling_models()
        else:
            self._load_v46_model()

    def _load_v46_model(self):
        """加载v4.6模型"""
        model_files = list(self.model_dir.glob('v46_*.pkl'))
        if not model_files:
            print(f"V4.6 未找到模型文件: {self.model_dir}/v46_*.pkl")
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
        # V4.6 optimal: 10d+15d after ablation (511 days, 2024-01~2026-02)
        # 10d+15d: AnnRet +77.5%, Sharpe 1.345 vs default: +68.5%, 1.212
        self.target_weights = {
            'label_3d': 0.00, 'label_5d': 0.00, 'label_10d': 0.60, 'label_15d': 0.40
        }

        # 元数据
        self.cascade = False
        self.cascade_feature_names = None
        self.dual_stream = False
        self.rank_normalized = False
        self.robust_zscore = model_data.get('robust_zscore', True)
        self.extra_features_from_daily_basic = model_data.get('extra_features_from_daily_basic', None)
        self.extra_tech_features = model_data.get('extra_features_from_tech_indicators', None)
        self.stock_rank_cols = model_data.get('stock_feature_cols', None)

        # Winsorize bounds
        raw_bounds = model_data.get('winsorize_bounds')
        if raw_bounds:
            self.winsorize_bounds = {k: tuple(v) for k, v in raw_bounds.items()} if isinstance(raw_bounds, dict) else raw_bounds

        # V4.4 组件 (继承)
        self.bear_models = model_data.get('bear_models', {})
        self.isotonic_calibration = model_data.get('isotonic_calibration', {})

        # V4.6 新增组件
        self.combined_isotonic = model_data.get('combined_isotonic')
        self.meta_learner = model_data.get('meta_learner')
        self.meta_feature_names = model_data.get('meta_feature_names')

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

        # V4.6.1: ICIR权重clip到[0.08, 0.50]后重归一化, 防止单模型主导
        self.weights = self._clip_icir_weights(self.weights)

        gq_status = "全局评分" if self.global_quantiles is not None else "截面评分"
        meta_status = "有" if self.meta_learner is not None else "无"
        ciso_status = "有" if self.combined_isotonic is not None else "无"
        print(f"V4.6 模型加载完成: {list(self.models.keys())} [V4.4底座+5增强+{gq_status}]")
        print(f"  模型文件: {latest.name}")
        print(f"  熊市专家: {list(self.bear_models.keys()) if self.bear_models else '无'}")
        print(f"  保序校准: {list(self.isotonic_calibration.keys()) if self.isotonic_calibration else '无'}")
        print(f"  Combined Isotonic: {ciso_status}")
        print(f"  Meta-Learner: {meta_status} ({len(self.meta_feature_names) if self.meta_feature_names else 0} features)")
        if wf:
            for t, m in wf.items():
                print(f"  WF {t}: ICIR={m.get('mean_icir', 0):.4f}±{m.get('std_icir', 0):.4f}")

    def _clip_icir_weights(self, weights: Dict) -> Dict:
        """V4.6.1: clip ICIR权重到[0.08, 0.50]后重归一化, 保持集成多样性"""
        clipped = {}
        for target_key, model_weights in weights.items():
            if not isinstance(model_weights, dict):
                clipped[target_key] = model_weights
                continue
            new_w = {}
            for name, w in model_weights.items():
                new_w[name] = np.clip(w, 0.08, 0.50)
            total = sum(new_w.values())
            if total > 0:
                new_w = {k: v / total for k, v in new_w.items()}
            clipped[target_key] = new_w
        return clipped

    def _get_meta_predictions(self, X: np.ndarray) -> Optional[np.ndarray]:
        """1D: 使用Meta-Learner生成combined_pred"""
        if self.meta_learner is None or not self.meta_feature_names:
            return None

        # 收集meta features: 与训练时相同的顺序
        meta_features = []
        for feat_name in self.meta_feature_names:
            # feat_name format: '{model_name}_{target_key}'
            parts = feat_name.rsplit('_', 1)
            if len(parts) != 2:
                return None
            model_name, target_key = parts

            if target_key not in self.models:
                return None
            if model_name not in self.models[target_key]:
                return None

            model = self.models[target_key][model_name]
            try:
                if model_name == 'xgb':
                    import xgboost as xgb_lib
                    pred = model.predict(xgb_lib.DMatrix(X))
                else:
                    pred = model.predict(X)
                meta_features.append(pred)
            except Exception:
                return None

        if len(meta_features) != len(self.meta_feature_names):
            return None

        X_meta = np.column_stack(meta_features)
        try:
            return self.meta_learner.predict(X_meta)
        except Exception:
            return None

    def _apply_combined_isotonic(self, combined_pred: np.ndarray) -> np.ndarray:
        """1C: Combined-Score Isotonic校准"""
        if self.combined_isotonic is None:
            return combined_pred
        try:
            return self.combined_isotonic.predict(combined_pred)
        except Exception:
            return combined_pred

    def _load_daily_basic_for_smallcap(self, date: str, codes: List[str]) -> Dict[str, float]:
        """加载当日circ_mv数据 (用于小盘加成)"""
        if date in self._daily_basic_cache:
            return self._daily_basic_cache[date]

        conn = sqlite3.connect(self.db_path)
        try:
            placeholders = ','.join(['?' for _ in codes])
            query = f"""
            SELECT s.code, db.circ_mv
            FROM daily_basic db
            JOIN securities s ON db.security_id = s.id
            WHERE s.code IN ({placeholders}) AND db.trade_date = ?
            """
            df = pd.read_sql_query(query, conn, params=codes + [date])
        finally:
            conn.close()

        result = {}
        for _, row in df.iterrows():
            if pd.notna(row.get('circ_mv')):
                result[row['code']] = float(row['circ_mv'])

        self._daily_basic_cache[date] = result
        return result

    def _apply_enhanced_executability_filters(self, results: Dict[str, Dict], date: str) -> Dict[str, Dict]:
        """V4.6.1: 增强可执行性过滤 + 连续流动性折扣

        vs V4.4: 流动性 turnover < 1.0% 阶梯折扣 → turnover < 1.5% 连续折扣 (V4.6.1温和版)
        """
        exec_data_t = self._load_executability_data(date, list(results.keys()))

        # 加载T+1数据 (买入日)
        next_date = self._get_next_trading_date(date)
        exec_data_t1 = self._load_executability_data(next_date, list(results.keys())) if next_date else {}

        for code in list(results.keys()):
            d_t = exec_data_t.get(code, {})
            d_t1 = exec_data_t1.get(code, {})

            # 涨停阈值 (小数形式: 0.095 = 9.5%)
            is_cyb_kc = code.startswith('30') or code.startswith('688')
            is_bse = code.startswith('8')
            limit_threshold = 0.195 if is_cyb_kc else (0.295 if is_bse else 0.095)

            # T+1实际涨停 → 评分清零
            pct_t1 = d_t1.get('pct_change', 0)
            if pct_t1 >= limit_threshold:
                results[code]['score'] = 0.0
                results[code]['exec_filter'] = 'limit_up_t1'
                continue

            # T日涨停 → 评分清零
            pct_t = d_t.get('pct_change', 0)
            if pct_t >= limit_threshold:
                results[code]['score'] = 0.0
                results[code]['exec_filter'] = 'limit_up'
                continue

            # T+1近涨停 (涨幅>5%) → 大幅降权
            if pct_t1 > 0.05:
                results[code]['score'] *= 0.2
                results[code]['exec_filter'] = 'near_limit_up_t1'
                continue

            # T日近涨停 (涨幅>5%) → 降权
            if pct_t > 0.05:
                results[code]['score'] *= 0.3
                results[code]['exec_filter'] = 'near_limit_up'
                continue

            # T日涨幅>3% → 轻度降权
            if pct_t > 0.03:
                results[code]['score'] *= 0.7
                results[code]['exec_filter'] = 'momentum_risk'
                continue

            # V4.6.1: 流动性折扣 — 阈值1.5%、底线0.2 (V4.6是2.0%/0.1, 太激进)
            # turnover < 1.5% → score *= max(0.2, turnover/1.5)
            turnover = d_t.get('turnover_rate', 999)
            if turnover < 1.5:
                discount = max(0.2, turnover / 1.5)
                results[code]['score'] *= discount
                results[code]['exec_filter'] = 'low_liquidity'
                continue

            results[code]['exec_filter'] = 'pass'

        return results

    def _apply_small_cap_bonus(self, results: Dict[str, Dict], date: str,
                                codes: List[str]) -> Dict[str, Dict]:
        """2E: 小盘评分加成 — 连续函数，中位数以下股票获得0-20%加成

        公式: score *= 1 + beta * max(0, 1 - mv/median_mv)
        beta=0.20: 市值=0时+20%, 市值=median时+0%, 线性过渡
        """
        circ_mv_data = self._load_daily_basic_for_smallcap(date, codes)
        if not circ_mv_data:
            return results

        mv_values = list(circ_mv_data.values())
        if not mv_values:
            return results

        median_mv = np.median(mv_values)
        if median_mv <= 0:
            return results

        beta = 0.20
        for code in codes:
            if code not in results or code not in circ_mv_data:
                continue
            ratio = circ_mv_data[code] / median_mv
            bonus = beta * max(0, 1 - ratio)
            if bonus > 0:
                results[code]['score'] *= (1 + bonus)

        return results

    def predict_scores(self, stock_codes: List[str], date: str) -> Dict[str, Dict]:
        """V4.6 评分管线:
        Step 1: V4.3 基础特征 (robust z-score + daily_basic + tech)
        Step 2: 5模型×4目标 base predictions
        Step 3: Meta-learner combined_pred 或 ICIR权重加权
        Step 4: Combined isotonic calibration
        Step 5: 全局百分位评分
        Step 6: 熊市专家混合
        Step 7: Per-target isotonic + 重算分数
        Step 8: 增强可执行性过滤 + 连续流动性折扣
        Step 9: 小盘评分加成
        """
        results = {}

        # Step 1: V4.3 基础特征
        features_df = self._get_features(stock_codes, date, load_full_cross_section=True)
        if features_df is not None and len(features_df) > 0:
            features_df = self._robust_zscore_normalize_features(features_df)
            features_df = self._load_daily_basic_features(features_df, date)
            features_df = self._load_technical_features(features_df, date)
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
                    logger.warning(f"⚠️ V4.6: {len(missing)}/{len(self.feature_cols)} 特征缺失: {missing[:5]}...")
                for col in missing:
                    features_df[col] = 0
            available_cols = self.feature_cols
        else:
            available_cols = [c for c in features_df.columns if c not in exclude_cols]

        X = features_df[available_cols].fillna(0).values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        codes = features_df['code'].tolist()

        # Step 2: 5模型×4目标 base predictions
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
                    if name == 'xgb':
                        import xgboost as xgb_lib
                        pred = model.predict(xgb_lib.DMatrix(X))
                    else:
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

        # Step 3: Meta-learner 或 ICIR权重加权
        regime_weights = self._get_regime_target_weights(date)
        if model_predictions_success:
            # 尝试Meta-learner
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

        # Step 4: Combined isotonic calibration
        combined_pred = self._apply_combined_isotonic(combined_pred)

        # Step 5: 全局百分位评分
        scores = self._to_global_score(combined_pred)

        for i, code in enumerate(codes):
            results[code] = {
                'score': float(scores[i]),
                'pred_3d': float(predictions['3d'][i]) if i < len(predictions['3d']) else 0,
                'pred_5d': float(predictions['5d'][i]) if i < len(predictions['5d']) else 0,
                'pred_10d': float(predictions['10d'][i]) if i < len(predictions['10d']) else 0,
                'pred_15d': float(predictions.get('15d', np.zeros(1))[min(i, len(predictions.get('15d', [0]))-1)]) if '15d' in predictions else 0,
            }

        # Step 6: Module C — 熊市专家混合
        results = self._blend_bear_specialist(results, date, X, codes)

        # Step 7: Module A — Per-target 保序回归校准
        results = self._apply_isotonic_calibration(results, codes)

        # 校准后重新计算综合分数 (全局百分位)
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

            # Re-apply combined isotonic
            new_combined = self._apply_combined_isotonic(new_combined)
            new_scores = self._to_global_score(new_combined)
            if len(new_scores) > 0:
                for i, code in enumerate(codes):
                    if code in results:
                        results[code]['score'] = float(new_scores[i])

        # Step 8: V4.6 增强可执行性过滤 + 连续流动性折扣
        results = self._apply_enhanced_executability_filters(results, date)

        # Step 9: 小盘评分加成 (连续函数, beta=0.20)
        results = self._apply_small_cap_bonus(results, date, codes)

        # 补全缺失code
        for code in stock_codes:
            if code not in results:
                results[code] = {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0, 'pred_15d': 0,
                                 'exec_filter': 'no_data'}

        # Module F: 附加市况信息 (不影响评分)
        regime_info = self._get_regime_info(date)
        for code in results:
            results[code]['regime_info'] = regime_info

        return results

    def predict_scores_from_preloaded(self, stock_codes: List[str], date: str,
                                       features_df: Optional[pd.DataFrame]) -> Dict[str, Dict]:
        """使用预加载特征评分 — V4.6 版本 (批量评分用)"""
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
                    if name == 'xgb':
                        import xgboost as xgb_lib
                        pred = model.predict(xgb_lib.DMatrix(X))
                    else:
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

        # Step 3: Meta-learner 或 加权融合
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

        # Step 4: Combined isotonic
        combined_pred = self._apply_combined_isotonic(combined_pred)

        # Step 5: 全局百分位
        scores = self._to_global_score(combined_pred)

        for i, code in enumerate(codes):
            results[code] = {
                'score': float(scores[i]),
                'pred_3d': float(predictions['3d'][i]) if i < len(predictions['3d']) else 0,
                'pred_5d': float(predictions['5d'][i]) if i < len(predictions['5d']) else 0,
                'pred_10d': float(predictions['10d'][i]) if i < len(predictions['10d']) else 0,
                'pred_15d': float(predictions.get('15d', np.zeros(1))[min(i, len(predictions.get('15d', [0]))-1)]) if '15d' in predictions else 0,
            }

        # Step 6: 熊市专家混合
        results = self._blend_bear_specialist(results, date, X, codes)

        # Step 7: 保序回归校准
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

            new_combined = self._apply_combined_isotonic(new_combined)
            new_scores = self._to_global_score(new_combined)
            if len(new_scores) > 0:
                for i, code in enumerate(codes):
                    if code in results:
                        results[code]['score'] = float(new_scores[i])

        # Step 8: V4.6 增强可执行性过滤
        results = self._apply_enhanced_executability_filters(results, date)

        # Step 9: 小盘评分加成 (连续函数, beta=0.20)
        results = self._apply_small_cap_bonus(results, date, codes)

        # 补全缺失code
        for code in stock_codes:
            if code not in results:
                results[code] = {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0, 'pred_15d': 0,
                                 'exec_filter': 'no_data'}

        return results
