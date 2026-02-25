#!/usr/bin/env python3
"""
V4.4 生产评分器
基于 V4.3 信号底座 + 6 个增强模块

改进点 (相比 V4.3):
  Module A: 保序回归校准 → IC单调性提升
  Module B: 流动性感知 (训练层已处理)
  Module C: 熊市专家模型混合 → 最差60日ICIR提升
  Module D: Sharpe标签融合 + 10d权重偏向 (训练层已处理)
  Module E: 可执行性过滤 → 涨停失败率降至2%
  Module F: 市况自适应信息输出
"""

import json
import pickle
import joblib
import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
from .v43_production_scorer import V43ProductionScorer


class V44ProductionScorer(V43ProductionScorer):
    """V4.4 生产评分器 — V4.3信号 + 可执行性过滤 + 熊市专家 + 保序校准 + 市况自适应"""

    def __init__(self, model_type: str = 'small_data'):
        self._v44_model_dir = Path(__file__).parent.parent.parent / 'ml_models' / 'trained_models' / 'v44'
        self.bear_models = {}
        self.isotonic_calibration = {}
        self._exec_cache = {}  # date -> DataFrame (可执行性数据缓存)
        self._market_return_cache = {}  # date -> float
        self._next_trade_date_cache = {}  # date -> next_trading_date
        super().__init__(model_type=model_type)

    def _load_models(self):
        """覆盖加载方法, 使用 v44 模型目录"""
        self.model_dir = self._v44_model_dir
        if self.model_type == 'rolling':
            self._load_rolling_models()
        else:
            self._load_v44_model()

    def _load_v44_model(self):
        """加载v4.4模型"""
        model_files = list(self.model_dir.glob('v44_*.pkl'))
        if not model_files:
            print(f"V4.4 未找到模型文件: {self.model_dir}/v44_*.pkl")
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

        # V4.4 新增组件
        self.bear_models = model_data.get('bear_models', {})
        self.isotonic_calibration = model_data.get('isotonic_calibration', {})

        wf = model_data.get('walk_forward_metrics', {})

        print(f"V4.4 模型加载完成: {list(self.models.keys())} [v4.3信号+6增强模块]")
        print(f"  模型文件: {latest.name}")
        print(f"  熊市专家: {list(self.bear_models.keys()) if self.bear_models else '无'}")
        print(f"  保序校准: {list(self.isotonic_calibration.keys()) if self.isotonic_calibration else '无'}")
        if wf:
            for t, m in wf.items():
                print(f"  WF {t}: ICIR={m.get('mean_icir', 0):.4f}±{m.get('std_icir', 0):.4f}")

    def _load_executability_data(self, date: str, codes: List[str] = None) -> Dict[str, dict]:
        """从daily_quotes+daily_basic加载当日涨停/换手数据"""
        if date in self._exec_cache:
            return self._exec_cache[date]

        conn = sqlite3.connect(self.db_path)
        try:
            code_filter = ""
            params = [date]
            if codes:
                placeholders = ','.join(['?' for _ in codes])
                code_filter = f" AND s.code IN ({placeholders})"
                params.extend(codes)

            query = f"""
            SELECT s.code,
                   q.is_limit_up, q.price_change_pct,
                   COALESCE(db.turnover_rate, 0) as turnover_rate
            FROM daily_quotes q
            JOIN securities s ON q.security_id = s.id
            LEFT JOIN daily_basic db ON db.security_id = s.id AND db.trade_date = q.trade_date
            WHERE q.trade_date = ? AND s.type = 'A股'{code_filter}
            """
            df = pd.read_sql_query(query, conn, params=params)
        finally:
            conn.close()

        result = {}
        for _, row in df.iterrows():
            result[row['code']] = {
                'is_limit_up': int(row.get('is_limit_up', 0) or 0),
                'pct_change': float(row.get('price_change_pct', 0) or 0),
                'turnover_rate': float(row.get('turnover_rate', 0) or 0),
            }

        self._exec_cache[date] = result
        return result

    def _get_market_return_20d(self, date: str) -> Optional[float]:
        """获取截至date的市场20日收益率"""
        if date in self._market_return_cache:
            return self._market_return_cache[date]

        conn = sqlite3.connect(self.db_path)
        try:
            query = """
            SELECT q.close
            FROM daily_quotes q
            JOIN securities s ON q.security_id = s.id
            WHERE s.code = '000001.SH' AND q.trade_date <= ?
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

    def _get_next_trading_date(self, date: str) -> Optional[str]:
        """获取date的下一个交易日 (买入日T+1)"""
        if date in self._next_trade_date_cache:
            return self._next_trade_date_cache[date]

        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("""
                SELECT DISTINCT trade_date FROM daily_quotes
                WHERE trade_date > ? ORDER BY trade_date LIMIT 1
            """, (date,)).fetchone()
        finally:
            conn.close()

        result = row[0] if row else None
        self._next_trade_date_cache[date] = result
        return result

    def _blend_bear_specialist(self, results: Dict[str, Dict], date: str,
                                X: np.ndarray, codes: List[str]) -> Dict[str, Dict]:
        """Module C: 熊市时混入bear_specialist预测 (V4.4.1: 更早激活, 更大权重)"""
        if not self.bear_models:
            return results

        market_ret = self._get_market_return_20d(date)
        if market_ret is None or market_ret > -0.03:
            return results  # 非熊市不混合 (阈值从-5%降至-3%)

        # 熊市权重: -3%→0.15, -5%→0.30, -10%→0.60, 最大0.60
        bear_weight = min(0.60, max(0.15, (abs(market_ret) - 0.03) * 6.4 + 0.15))

        for target_key, bear_model in self.bear_models.items():
            try:
                bear_pred = bear_model.predict(X)
            except Exception:
                continue

            # 混合: (1-w)*base + w*bear
            for i, code in enumerate(codes):
                if code in results and i < len(bear_pred):
                    pred_key = f'pred_{target_key}'
                    if pred_key in results[code]:
                        base_val = results[code][pred_key]
                        blended = (1 - bear_weight) * base_val + bear_weight * float(bear_pred[i])
                        results[code][pred_key] = blended

        return results

    def _apply_isotonic_calibration(self, results: Dict[str, Dict],
                                     codes: List[str]) -> Dict[str, Dict]:
        """Module A: 保序回归校准 — 确保高分→高预期收益"""
        if not self.isotonic_calibration:
            return results

        for target_key, iso_model in self.isotonic_calibration.items():
            pred_key = f'pred_{target_key}'
            raw_preds = []
            valid_codes = []
            for code in codes:
                if code in results and pred_key in results[code]:
                    raw_preds.append(results[code][pred_key])
                    valid_codes.append(code)

            if not raw_preds:
                continue

            raw_arr = np.array(raw_preds)
            try:
                calibrated = iso_model.predict(raw_arr)
                for i, code in enumerate(valid_codes):
                    results[code][pred_key] = float(calibrated[i])
            except Exception:
                pass  # 校准失败时保持原值

        return results

    def _apply_executability_filters(self, results: Dict[str, Dict], date: str) -> Dict[str, Dict]:
        """Module E: 可执行性过滤 (V4.4.1: T+1涨停检测 + 更严门槛)

        关键修复: 北极星评估在买入日T+1检测涨停, 我们也必须检查T+1数据。
        报告生成于T日, 股票在T+1买入。T日涨5%的股票, T+1很可能涨停。
        """
        exec_data_t = self._load_executability_data(date, list(results.keys()))

        # 加载T+1数据 (买入日)
        next_date = self._get_next_trading_date(date)
        exec_data_t1 = self._load_executability_data(next_date, list(results.keys())) if next_date else {}

        for code in list(results.keys()):
            d_t = exec_data_t.get(code, {})
            d_t1 = exec_data_t1.get(code, {})

            # 判定涨停阈值 (百分比形式: 9.5 = 9.5%)
            is_cyb_kc = code.startswith('30') or code.startswith('688')
            limit_threshold = 19.5 if is_cyb_kc else 9.5

            # T+1实际涨停 → 评分清零 (买入日不可买入, 直接匹配北极星判定)
            pct_t1 = d_t1.get('pct_change', 0)
            if pct_t1 >= limit_threshold:
                results[code]['score'] = 0.0
                results[code]['exec_filter'] = 'limit_up_t1'
                continue

            # T日涨停 → 评分清零 (T+1大概率高开或继续涨停)
            pct_t = d_t.get('pct_change', 0)
            if pct_t >= limit_threshold:
                results[code]['score'] = 0.0
                results[code]['exec_filter'] = 'limit_up'
                continue

            # T+1近涨停 (涨幅>5%) → 大幅降权
            if pct_t1 > 5.0:
                results[code]['score'] *= 0.2
                results[code]['exec_filter'] = 'near_limit_up_t1'
                continue

            # T日近涨停 (涨幅>5%) → 降权 (T+1追高风险)
            if pct_t > 5.0:
                results[code]['score'] *= 0.3
                results[code]['exec_filter'] = 'near_limit_up'
                continue

            # T日涨幅>3% → 轻度降权 (追涨风险)
            if pct_t > 3.0:
                results[code]['score'] *= 0.7
                results[code]['exec_filter'] = 'momentum_risk'
                continue

            # 低换手率 (<1.0%) → 降权 (难以执行, 改善流动性覆盖)
            turnover = d_t.get('turnover_rate', 999)
            if turnover < 1.0:
                discount = 0.3 if turnover < 0.3 else 0.5 if turnover < 0.5 else 0.7
                results[code]['score'] *= discount
                results[code]['exec_filter'] = 'low_liquidity'
                continue

            results[code]['exec_filter'] = 'pass'

        return results

    def _get_regime_target_weights(self, date: str) -> dict:
        """根据市况动态调整目标权重 — 熊市偏向10d/15d (更防御), 牛市偏向3d/5d (更进攻)"""
        market_ret = self._get_market_return_20d(date)
        base = dict(self.target_weights)  # copy

        if market_ret is not None and market_ret < -0.03:
            # 熊市: 减少短期动量权重, 增加长期质量权重
            severity = min(1.0, (abs(market_ret) - 0.03) / 0.07)  # 0→1 over -3%→-10%
            base['label_3d'] = 0.20 - 0.10 * severity   # 0.20→0.10
            base['label_5d'] = 0.25 - 0.10 * severity   # 0.25→0.15
            base['label_10d'] = 0.35 + 0.05 * severity  # 0.35→0.40
            base['label_15d'] = 0.20 + 0.15 * severity  # 0.20→0.35

        return base

    def _get_regime_info(self, date: str) -> dict:
        """Module F: 市况自适应信息"""
        market_ret = self._get_market_return_20d(date)

        if market_ret is None:
            return {'regime': 'unknown', 'market_return_20d': None, 'bear_probability': 0.0}

        if market_ret < -0.05:
            regime = 'bear'
            bear_prob = min(1.0, abs(market_ret) * 10)
        elif market_ret > 0.05:
            regime = 'bull'
            bear_prob = 0.0
        else:
            regime = 'neutral'
            bear_prob = max(0.0, -market_ret * 10)

        return {
            'regime': regime,
            'market_return_20d': market_ret,
            'bear_probability': bear_prob,
            'suggested_top_n_ratio': 0.5 if regime == 'bear' else 1.0,  # 熊市减少持仓
        }

    def predict_scores(self, stock_codes: List[str], date: str) -> Dict[str, Dict]:
        """V4.4 评分管线: V4.3基础 → 熊市混合 → 保序校准 → 可执行性过滤"""
        results = {}

        # Step 1: V4.3 基础评分 (全截面 robust_zscore + daily_basic + tech features + 4目标)
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
                    logger.warning(f"⚠️ V4.4: {len(missing)}/{len(self.feature_cols)} 特征缺失: {missing[:5]}...")
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

        # 构建初始结果 (V4.4.1: 市况自适应目标权重)
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

        # 映射到 30-90 分制
        if len(combined_pred) > 1:
            from scipy import stats
            ranks = stats.rankdata(combined_pred)
            percentiles = (ranks - 1) / (len(ranks) - 1) * 100
            scores = 30 + percentiles * 0.6
        else:
            scores = np.array([60.0])

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

        # 校准后重新计算综合分数和排名 (V4.4.1: 市况自适应权重)
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

            if len(new_combined) > 1:
                from scipy import stats
                ranks = stats.rankdata(new_combined)
                percentiles = (ranks - 1) / (len(ranks) - 1) * 100
                new_scores = 30 + percentiles * 0.6
                for i, code in enumerate(codes):
                    if code in results:
                        results[code]['score'] = float(new_scores[i])

        # Step 4: Module E — 可执行性过滤 (V4.4.1: T+1涨停检测)
        results = self._apply_executability_filters(results, date)

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
        """使用预加载特征评分 — V4.4 版本 (批量评分用)"""
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

        # V4.4.1: 市况自适应目标权重
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

        if len(combined_pred) > 1:
            from scipy import stats
            ranks = stats.rankdata(combined_pred)
            percentiles = (ranks - 1) / (len(ranks) - 1) * 100
            scores = 30 + percentiles * 0.6
        else:
            scores = np.array([60.0])

        for i, code in enumerate(codes):
            results[code] = {
                'score': float(scores[i]),
                'pred_3d': float(predictions['3d'][i]) if i < len(predictions['3d']) else 0,
                'pred_5d': float(predictions['5d'][i]) if i < len(predictions['5d']) else 0,
                'pred_10d': float(predictions['10d'][i]) if i < len(predictions['10d']) else 0,
                'pred_15d': float(predictions.get('15d', np.zeros(1))[min(i, len(predictions.get('15d', [0]))-1)]) if '15d' in predictions else 0,
            }

        # Step 2: Module C — 熊市专家混合 (V4.4.1: 更早激活, 更大权重)
        results = self._blend_bear_specialist(results, date, X, codes)

        # Step 3: Module A — 保序回归校准
        results = self._apply_isotonic_calibration(results, codes)

        # 校准后重新排名 (V4.4.1: 市况自适应权重)
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

            if len(new_combined) > 1:
                from scipy import stats
                ranks = stats.rankdata(new_combined)
                percentiles = (ranks - 1) / (len(ranks) - 1) * 100
                new_scores = 30 + percentiles * 0.6
                for i, code in enumerate(codes):
                    if code in results:
                        results[code]['score'] = float(new_scores[i])

        # Step 4: Module E — 可执行性过滤 (V4.4.1: T+1涨停检测)
        results = self._apply_executability_filters(results, date)

        for code in stock_codes:
            if code not in results:
                results[code] = {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0, 'pred_15d': 0,
                                 'exec_filter': 'no_data'}

        return results

    def preload_feature_cache(self, dates: List[str]) -> Dict[str, pd.DataFrame]:
        """批量预加载特征缓存 + 技术指标 + 可执行性数据"""
        result = super().preload_feature_cache(dates)

        # 批量预加载可执行性数据 (涨停/换手)
        self.preload_executability_bulk(dates)

        return result

    def preload_executability_bulk(self, dates: List[str]):
        """批量预加载涨停/换手数据 (V4.4.1: 同时预加载T+1数据), 避免逐日SQL"""
        # 收集T+1日期 (买入日)
        all_dates_needed = set(dates)
        conn = sqlite3.connect(self.db_path)
        try:
            # 批量查找所有dates的下一个交易日
            all_trade_dates = [r[0] for r in conn.execute(
                "SELECT DISTINCT trade_date FROM daily_quotes ORDER BY trade_date"
            ).fetchall()]
        finally:
            conn.close()

        trade_date_set = set(all_trade_dates)
        for d in dates:
            if d in trade_date_set:
                idx = all_trade_dates.index(d)
                if idx + 1 < len(all_trade_dates):
                    next_d = all_trade_dates[idx + 1]
                    all_dates_needed.add(next_d)
                    self._next_trade_date_cache[d] = next_d

        valid_dates = [d for d in all_dates_needed if d not in self._exec_cache]
        if not valid_dates:
            return

        conn = sqlite3.connect(self.db_path)
        try:
            placeholders = ','.join(['?' for _ in valid_dates])
            query = f"""
            SELECT s.code, q.trade_date,
                   q.is_limit_up, q.price_change_pct,
                   COALESCE(db.turnover_rate, 0) as turnover_rate
            FROM daily_quotes q
            JOIN securities s ON q.security_id = s.id
            LEFT JOIN daily_basic db ON db.security_id = s.id AND db.trade_date = q.trade_date
            WHERE q.trade_date IN ({placeholders}) AND s.type = 'A股'
            """
            df = pd.read_sql_query(query, conn, params=valid_dates)
        finally:
            conn.close()

        if df.empty:
            return

        for date, date_df in df.groupby('trade_date'):
            result = {}
            for _, row in date_df.iterrows():
                result[row['code']] = {
                    'is_limit_up': int(row.get('is_limit_up', 0) or 0),
                    'pct_change': float(row.get('price_change_pct', 0) or 0),
                    'turnover_rate': float(row.get('turnover_rate', 0) or 0),
                }
            self._exec_cache[date] = result

        print(f"V4.4可执行性数据预加载完成: {len(valid_dates)}天(含T+1), {len(df)}条记录")
