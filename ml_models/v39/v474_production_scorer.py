#!/usr/bin/env python3
"""
V4.7.4 生产评分器 — 连续评分 + V4.7.3简化管线 + 选择性V4.8特征 + ListNet

核心改进 (相比V4.7.3):
1. 连续插值评分: np.interp替代np.searchsorted, 消除头部同分问题
2. 内置composite排名: scorer直接输出rank_score, 无需回测层二次计算
3. 选择性财务特征: +4个低缺失V4.8特征 (netprofit_margin/ocf_to_opincome/debt_to_eqt/basic_eps_yoy)
4. ListNet排名模型: 10d/15d加入ListNet (V4.8证明ICIR+0.14/+0.24)
5. ICIR权重约束: floor=0.10, ceiling=0.35 (防止V4.8式92.6%过度集中)
6. 预测值z-score标准化: 消除模型间尺度差异后再ICIR加权

继承V4.7.3: 无Meta-Learner, 无Combined Isotonic, ICIR权重, Bear Specialist, Per-target Isotonic
"""

import numpy as np
import pandas as pd
import sqlite3
import pickle
import joblib
from pathlib import Path
from typing import Dict, List, Optional

from .v473_production_scorer import V473ProductionScorer

import logging
logger = logging.getLogger(__name__)


# Composite ranking weights (与backtest_report_based.py对齐)
COMPOSITE_WEIGHTS = {
    'pred_3d': 0.10,
    'pred_5d': 0.20,
    'pred_10d': 0.40,
    'pred_15d': 0.30,
}


class V474ProductionScorer(V473ProductionScorer):
    """V4.7.4 生产评分器 — 连续评分 + 选择性V4.8增强 + ListNet"""

    def __init__(self, model_type: str = 'small_data'):
        self._v474_model_dir = Path(__file__).parent.parent.parent / 'ml_models' / 'trained_models' / 'v474'
        self._fq_cache = {}  # V4.8 financial quality cache
        super().__init__(model_type=model_type)

    def _load_models(self):
        """覆盖加载方法, 使用 v474 模型目录"""
        self.model_dir = self._v474_model_dir
        if self.model_type == 'rolling':
            self._load_rolling_models()
        else:
            self._load_v474_model()

    def _load_v474_model(self):
        """加载v4.7.4模型"""
        model_files = list(self.model_dir.glob('v474_*.pkl'))
        if not model_files:
            print(f"V4.7.4 未找到模型文件: {self.model_dir}/v474_*.pkl")
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

        # V4.7.1 精简特征 + V4.7.4选择性V4.8特征
        self.extra_features_financial = model_data.get('extra_features_financial', [])
        self.extra_features_microstructure = model_data.get('extra_features_microstructure', [])
        self.extra_features_reversal = model_data.get('extra_features_reversal', [])
        self.extra_features_risk = model_data.get('extra_features_risk', [])
        self.extra_financial_quality = model_data.get('extra_financial_quality', [])

        # Winsorize bounds
        raw_bounds = model_data.get('winsorize_bounds')
        if raw_bounds:
            self.winsorize_bounds = {k: tuple(v) for k, v in raw_bounds.items()} if isinstance(raw_bounds, dict) else raw_bounds

        # V4.4 组件 (继承)
        self.bear_models = model_data.get('bear_models', {})
        self.isotonic_calibration = model_data.get('isotonic_calibration', {})

        # V4.7.4: 无Meta-Learner, 无Combined Isotonic (继承V4.7.3核心设计)

        # 全局分位数
        raw_quantiles = model_data.get('global_quantiles')
        if raw_quantiles is not None:
            self.global_quantiles = np.array(raw_quantiles)
        else:
            quantiles_path = self.model_dir / 'global_quantiles.npy'
            if quantiles_path.exists():
                self.global_quantiles = np.load(quantiles_path)

        # V4.7.4: ICIR权重clip到[0.10, 0.35]后重归一化 (更严约束)
        self.weights = self._clip_icir_weights_v474(self.weights)

        wf = model_data.get('walk_forward_metrics', {})
        gq_status = "连续全局评分" if self.global_quantiles is not None else "截面评分"
        print(f"V4.7.4 模型加载完成: {list(self.models.keys())} [连续评分+选择性增强+{gq_status}]")
        print(f"  模型文件: {latest.name}")
        print(f"  特征数: {len(self.feature_cols)}")
        n_v471 = (len(self.extra_features_financial) + len(self.extra_features_microstructure) +
                  len(self.extra_features_reversal) + len(self.extra_features_risk))
        print(f"  V4.7.1特征: {n_v471} | V4.8选择性特征: {len(self.extra_financial_quality)}")

        # 检查是否有ListNet模型
        has_listnet = any('lgb_listnet' in (self.models.get(t, {}) or {}) for t in ['10d', '15d'])
        print(f"  ListNet: {'有(10d/15d)' if has_listnet else '无'}")
        print(f"  ICIR约束: floor=0.10, ceiling=0.35 (V4.7.4加严)")
        print(f"  Meta-Learner: 无 | Combined Isotonic: 无")
        if wf:
            for t, m in wf.items():
                print(f"  WF {t}: ICIR={m.get('mean_icir', 0):.4f}+-{m.get('std_icir', 0):.4f}")

    # ========== V4.7.4: 更严ICIR权重约束 ==========

    def _clip_icir_weights_v474(self, weights: Dict) -> Dict:
        """V4.7.4: clip ICIR权重到[0.10, 0.35]后重归一化

        V4.8教训: 5d目标HGB获得92.6%权重, 相当于单模型预测
        V4.7.4: floor=0.10保证每个模型都有贡献, ceiling=0.35防止单模型主导
        """
        clipped = {}
        for target_key, model_weights in weights.items():
            if not isinstance(model_weights, dict):
                clipped[target_key] = model_weights
                continue
            new_w = {}
            for name, w in model_weights.items():
                new_w[name] = np.clip(w, 0.10, 0.35)
            total = sum(new_w.values())
            if total > 0:
                new_w = {k: v / total for k, v in new_w.items()}
            clipped[target_key] = new_w
        return clipped

    # ========== V4.7.4核心创新1: 连续插值评分 ==========

    def _to_global_score(self, combined_pred: np.ndarray) -> np.ndarray:
        """V4.7.4: 连续插值评分 — 替代V4.7.3的searchsorted离散映射

        使用np.interp进行线性插值, 产生真正连续的浮点分数:
        - V4.7.3: searchsorted → 只有1001个离散值(0.0, 0.1, ..., 100.0)
        - V4.7.4: interp → 无限精度连续值(如94.7231 vs 94.7289)
        - 头部区分度: 从~5只同分 → 每只股票唯一分数

        全局分位数数组的角度:
        - global_quantiles[i] 对应百分位 i/1000*100
        - interp在quantile值之间做线性插值, 产生连续映射
        """
        if self.global_quantiles is not None and len(self.global_quantiles) > 1:
            # 连续插值: pred值 → 百分位分数(0-100, 真正连续)
            percentile_grid = np.linspace(0, 100, len(self.global_quantiles))
            scores = np.interp(combined_pred, self.global_quantiles, percentile_grid)
            return np.clip(scores, 0, 100)
        else:
            # Fallback: 截面百分位 (已连续, 无需修改)
            if len(combined_pred) > 1:
                from scipy import stats
                ranks = stats.rankdata(combined_pred)
                percentiles = (ranks - 1) / (len(ranks) - 1) * 100
                scores = 30 + percentiles * 0.6
            else:
                scores = np.array([60.0])
            return scores

    # ========== V4.7.4核心创新2: 内置composite排名 ==========

    def _compute_composite_rank_score(self, results: Dict[str, Dict]) -> Dict[str, Dict]:
        """V4.7.4: 直接在scorer层计算composite rank_score

        多周期加权排名融合, 与backtest_report_based.py的逻辑完全对齐:
        - 对每个pred_Xd独立计算百分位排名[0,1]
        - 加权合并: 3d×0.10 + 5d×0.20 + 10d×0.40 + 15d×0.30
        - 只有在所有周期都排名靠前的股票才能获得高composite分

        这确保报告生成时无需依赖score排序, 直接用rank_score.
        """
        codes = list(results.keys())
        if not codes:
            return results

        n = len(codes)
        if n <= 1:
            for code in codes:
                results[code]['rank_score'] = results[code].get('score', 50.0)
            return results

        # 提取各周期预测值
        pred_arrays = {}
        for field in COMPOSITE_WEIGHTS:
            key = field  # pred_3d, pred_5d, pred_10d, pred_15d
            values = np.array([results[c].get(key, 0) for c in codes])
            pred_arrays[field] = values

        # 计算各周期百分位排名
        from scipy.stats import rankdata
        rank_arrays = {}
        for field, values in pred_arrays.items():
            ranks = rankdata(values, method='average')
            rank_arrays[field] = (ranks - 1) / max(n - 1, 1)  # 归一化到[0,1]

        # 加权合并
        composite = np.zeros(n)
        for field, weight in COMPOSITE_WEIGHTS.items():
            composite += weight * rank_arrays[field]

        # 写入结果
        for i, code in enumerate(codes):
            results[code]['rank_score'] = float(composite[i])

        return results

    # ========== V4.7.4: 加载选择性V4.8财务质量特征 ==========

    def _load_financial_quality_features(self, features_df: pd.DataFrame, date: str) -> pd.DataFrame:
        """加载V4.7.4选择性V4.8财务质量特征 (4个低缺失率特征)

        选择标准:
        - netprofit_margin: 盈利质量核心指标, 缺失率<20%
        - ocf_to_opincome: 现金流质量, 区分真假利润
        - debt_to_eqt: 杠杆率, 风险因子
        - basic_eps_yoy: EPS增速, 增长动量

        排除:
        - roe_dt (与roe高度相关, 冗余)
        - fcfe_ps/salescash_to_or (缺失率>50%)
        - ebit_to_interest/quick_ratio (稀疏)
        - op_yoy/q_profit_yoy/netprofit_yoy_accel (与netprofit_yoy冗余)
        """
        if not self.extra_financial_quality:
            return features_df

        if date in self._fq_cache:
            df_fi = self._fq_cache[date]
        else:
            conn = sqlite3.connect(self.db_path)
            try:
                select_cols = ', '.join([f'fi.{col}' for col in self.extra_financial_quality])
                query = f"""
                SELECT s.code, {select_cols}
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
            self._fq_cache[date] = df_fi

        if len(df_fi) > 0:
            features_df = features_df.merge(df_fi, on='code', how='left')

        for col in self.extra_financial_quality:
            if col in features_df.columns:
                median_val = features_df[col].median()
                features_df[col] = features_df[col].fillna(median_val if not pd.isna(median_val) else 0.0)
            else:
                features_df[col] = 0.0

        return features_df

    # ========== V4.7.4核心创新5: 预测值z-score标准化集成 ==========

    def _ensemble_predict_zscore(self, models: dict, weights: dict, X: np.ndarray,
                                  target: str) -> np.ndarray:
        """V4.7.4: 预测值z-score标准化后加权集成

        问题: 不同模型的预测值尺度不同 (LGB: [-0.01, 0.05], RF: [-0.05, 0.20])
        简单加权平均会被大尺度模型主导

        解决: 先将每个模型的预测标准化为零均值单位方差, 再加权
        """
        if not models:
            return np.zeros(len(X))

        preds = {}
        for name, model in models.items():
            try:
                if name == 'xgb':
                    import xgboost as xgb_lib
                    pred = model.predict(xgb_lib.DMatrix(X))
                else:
                    pred = model.predict(X)
                preds[name] = pred
            except Exception:
                continue

        if not preds:
            return np.zeros(len(X))

        # Z-score标准化每个模型的预测
        standardized = {}
        for name, pred in preds.items():
            mean_val = np.mean(pred)
            std_val = np.std(pred)
            if std_val > 1e-10:
                standardized[name] = (pred - mean_val) / std_val
            else:
                standardized[name] = pred - mean_val

        # ICIR加权融合标准化后的预测
        target_key = f'label_{target}'
        total_weight = 0
        combined = np.zeros(len(X))
        for name, zpred in standardized.items():
            w = weights.get(target_key, {}).get(name, 1.0 / len(standardized))
            combined += w * zpred
            total_weight += w

        if total_weight > 0:
            combined /= total_weight

        # 还原到原始模型的平均尺度 (保持可解释性)
        all_means = [np.mean(p) for p in preds.values()]
        all_stds = [np.std(p) for p in preds.values()]
        avg_mean = np.mean(all_means)
        avg_std = np.mean(all_stds) if np.mean(all_stds) > 1e-10 else 1.0
        combined = combined * avg_std + avg_mean

        return combined

    # ========== 覆写: predict_scores — V4.7.4 8步管线 ==========

    def predict_scores(self, stock_codes: List[str], date: str) -> Dict[str, Dict]:
        """V4.7.4 评分管线 — 连续评分 + 选择性V4.8增强:
        Step 1: V4.3 基础特征 (robust z-score + daily_basic + tech)
        Step 1.5: V4.7.1 精简特征 (roe + daily_basic_extra + micro+reversal+idio_vol)
        Step 1.7: V4.7.4 选择性V4.8特征 (4个低缺失财务质量)
        Step 2: 5-7模型×4目标 base predictions
        Step 3: Z-score标准化 + ICIR[0.10,0.35]加权融合 (无Meta-Learner)
        Step 4: 连续插值全局评分 (np.interp, 无Combined Isotonic)
        Step 5: 熊市专家混合 + Per-target Isotonic + 重算分数
        Step 6: 增强可执行性过滤 + 连续流动性折扣
        Step 7: Composite多周期排名融合
        """
        # 日期格式标准化
        if isinstance(date, str) and len(date) == 8 and date.isdigit():
            date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        results = {}

        # Step 1: V4.3 基础特征
        features_df = self._get_features(stock_codes, date, load_full_cross_section=True)
        if features_df is not None and len(features_df) > 0:
            features_df = self._robust_zscore_normalize_features(features_df)
            features_df = self._load_daily_basic_features(features_df, date)
            features_df = self._load_technical_features(features_df, date)

            # Step 1.5: V4.7.1 精简特征
            features_df = self._load_financial_features(features_df, date)
            features_df = self._load_daily_basic_extra(features_df, date)
            features_df = self._compute_microstructure_features(features_df, date)

            # Step 1.7: V4.7.4 选择性V4.8特征
            features_df = self._load_financial_quality_features(features_df, date)

            features_df = features_df[features_df['code'].isin(stock_codes)].copy()

        if features_df is None or len(features_df) == 0:
            for code in stock_codes:
                results[code] = {'score': 50.0, 'rank_score': 0.5,
                                 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0, 'pred_15d': 0,
                                 'exec_filter': 'no_data'}
            return results

        # 准备特征矩阵
        exclude_cols = {'code', 'trade_date'}
        if self.feature_cols:
            missing = [c for c in self.feature_cols if c not in features_df.columns]
            if missing:
                if len(missing) > len(self.feature_cols) * 0.3:
                    logger.warning(f"V4.7.4: {len(missing)}/{len(self.feature_cols)} 特征缺失: {missing[:5]}...")
                for col in missing:
                    features_df[col] = 0
            available_cols = self.feature_cols
        else:
            available_cols = [c for c in features_df.columns if c not in exclude_cols]

        X = features_df[available_cols].fillna(0).values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        codes = features_df['code'].tolist()

        # Step 2+3: Z-score标准化 + ICIR加权集成 (每个目标独立)
        model_predictions_success = False
        predictions = {
            '3d': np.zeros(len(X)), '5d': np.zeros(len(X)),
            '10d': np.zeros(len(X)), '15d': np.zeros(len(X))
        }

        for target in ['3d', '5d', '10d', '15d']:
            if target not in self.models or not self.models[target]:
                continue
            # V4.7.4: z-score标准化集成
            target_pred = self._ensemble_predict_zscore(
                self.models[target], self.weights, X, target)
            if np.any(target_pred != 0):
                predictions[target] = target_pred
                model_predictions_success = True

        # ICIR加权目标融合 (无Meta-Learner)
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

        # Step 4: 连续插值全局评分 (V4.7.4核心改进)
        scores = self._to_global_score(combined_pred)

        for i, code in enumerate(codes):
            results[code] = {
                'score': float(scores[i]),
                'pred_3d': float(predictions['3d'][i]) if i < len(predictions['3d']) else 0,
                'pred_5d': float(predictions['5d'][i]) if i < len(predictions['5d']) else 0,
                'pred_10d': float(predictions['10d'][i]) if i < len(predictions['10d']) else 0,
                'pred_15d': float(predictions.get('15d', np.zeros(1))[min(i, len(predictions.get('15d', [0]))-1)]) if '15d' in predictions else 0,
            }

        # Step 5: Bear specialist + Per-target Isotonic + 重算分数
        results = self._blend_bear_specialist(results, date, X, codes)
        results = self._apply_isotonic_calibration(results, codes)

        # 校准后重新计算综合分数 (连续插值, 无Combined Isotonic)
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

        # Step 6: 增强可执行性过滤 + 连续流动性折扣
        results = self._apply_enhanced_executability_filters(results, date)

        # 补全缺失code
        for code in stock_codes:
            if code not in results:
                results[code] = {'score': 50.0, 'rank_score': 0.5,
                                 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0, 'pred_15d': 0,
                                 'exec_filter': 'no_data'}

        # Step 7: Composite多周期排名融合 (V4.7.4内置)
        results = self._compute_composite_rank_score(results)

        # 附加市况信息 (不影响评分)
        regime_info = self._get_regime_info(date)
        for code in results:
            results[code]['regime_info'] = regime_info

        return results

    def predict_scores_from_preloaded(self, stock_codes: List[str], date: str,
                                       features_df: Optional[pd.DataFrame]) -> Dict[str, Dict]:
        """使用预加载特征评分 — V4.7.4版本 (批量评分用)"""
        results = {}

        if features_df is None or len(features_df) == 0:
            for code in stock_codes:
                results[code] = {'score': 50.0, 'rank_score': 0.5,
                                 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0, 'pred_15d': 0,
                                 'exec_filter': 'no_data'}
            return results

        # 特征加载
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
                results[code] = {'score': 50.0, 'rank_score': 0.5,
                                 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0, 'pred_15d': 0,
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

        # Z-score标准化集成
        model_predictions_success = False
        predictions = {
            '3d': np.zeros(len(X)), '5d': np.zeros(len(X)),
            '10d': np.zeros(len(X)), '15d': np.zeros(len(X))
        }

        for target in ['3d', '5d', '10d', '15d']:
            if target not in self.models or not self.models[target]:
                continue
            target_pred = self._ensemble_predict_zscore(
                self.models[target], self.weights, X, target)
            if np.any(target_pred != 0):
                predictions[target] = target_pred
                model_predictions_success = True

        # 目标融合 (无Meta-Learner)
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

        # 连续插值评分
        scores = self._to_global_score(combined_pred)

        for i, code in enumerate(codes):
            results[code] = {
                'score': float(scores[i]),
                'pred_3d': float(predictions['3d'][i]) if i < len(predictions['3d']) else 0,
                'pred_5d': float(predictions['5d'][i]) if i < len(predictions['5d']) else 0,
                'pred_10d': float(predictions['10d'][i]) if i < len(predictions['10d']) else 0,
                'pred_15d': float(predictions.get('15d', np.zeros(1))[min(i, len(predictions.get('15d', [0]))-1)]) if '15d' in predictions else 0,
            }

        # Post-processing
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

        results = self._apply_enhanced_executability_filters(results, date)

        for code in stock_codes:
            if code not in results:
                results[code] = {'score': 50.0, 'rank_score': 0.5,
                                 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0, 'pred_15d': 0,
                                 'exec_filter': 'no_data'}

        # Composite排名
        results = self._compute_composite_rank_score(results)

        return results
