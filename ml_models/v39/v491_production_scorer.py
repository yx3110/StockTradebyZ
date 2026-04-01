#!/usr/bin/env python3
"""
V4.9.1 production scorer — V4.8.5底座 + 三管齐下改进

A) 基准超额标签训练 → 模型输出已经是超额收益预测
B) 市场门控: 内置LGB二分类器, confidence<0.3时不推荐任何股票
C) 排名平滑特征: 推理时计算3个平滑特征, 偏好信号稳定的股票

核心改进:
  - predict_scores() 前先运行市场门控, 低信心时返回空推荐
  - 在特征准备阶段注入排名平滑特征
  - 继承V4.8.5的Q95 Widen-then-Concentrate pipeline

Fallback chain: v491 model -> v485 model -> v484 model
"""

import numpy as np
import pandas as pd
import sqlite3
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .v485_production_scorer import V485ProductionScorer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 市场门控阈值
GATE_DONT_BUY = 0.30     # confidence < 0.30 → 不推荐任何股票
GATE_REDUCE = 0.50        # 0.30 ≤ confidence < 0.50 → top_n减半
GATE_FULL = 0.50          # confidence ≥ 0.50 → 正常推荐

# 排名平滑特征
SMOOTHING_FEATURES = ['feature_momentum_5d', 'signal_consistency_5d', 'consecutive_strength_5d']


class V491ProductionScorer(V485ProductionScorer):
    """V4.9.1 scorer — 市场门控 + 排名平滑 + 超额收益"""

    def __init__(self, model_type: str = 'small_data'):
        self._v491_model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v491'
        self._market_gate = None     # LGB binary classifier
        self._gate_features = None   # feature names for gate model
        self._last_gate_confidence = None  # 上一次市场信心值
        super().__init__(model_type=model_type)

    def _load_models(self):
        """Load v491 model, fallback to v485"""
        # V4.9.1 original
        v491_files = list(self._v491_model_dir.glob('v491_*.pkl'))
        if v491_files:
            self.model_dir = self._v491_model_dir
            latest = max(v491_files, key=lambda f: f.stat().st_mtime)
            self._load_model_from_file(latest, label='V4.9.1')

            # 加载市场门控模型
            model_data = joblib.load(str(latest))
            gate_data = model_data.get('market_gate')
            if gate_data and gate_data.get('model'):
                self._market_gate = gate_data['model']
                self._gate_features = gate_data['feature_names']
                logger.info(f"  市场门控模型已加载 (AUC={gate_data.get('auc', 'N/A')}, "
                           f"{len(self._gate_features)}个特征)")
            else:
                logger.warning("  V4.9.1: 市场门控模型未找到")
            return
        super()._load_models()

    def _evaluate_market_gate(self, date: str) -> float:
        """运行市场门控模型, 返回confidence ∈ [0, 1]

        confidence = P(未来10天市场正收益)
        """
        if self._market_gate is None or self._gate_features is None:
            return 1.0  # 无门控模型 → 全开

        try:
            # 获取市场特征
            conn = sqlite3.connect(str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db'))

            # 从v39_feature_cache获取市场特征 (任意A股的该日记录)
            query = """
            SELECT market_return_20d, market_return_10d, market_return_5d,
                   market_volatility_20d, market_volatility_10d,
                   market_up_ratio_20d, market_up_ratio_10d,
                   market_drawdown_20d, market_volume_ratio,
                   market_position_20d, market_momentum_20d, market_momentum_5d
            FROM v39_feature_cache
            WHERE trade_date = ?
            LIMIT 1
            """
            row = conn.execute(query, (date,)).fetchone()
            conn.close()

            if row is None:
                # 尝试YYYYMMDD格式
                date_alt = date.replace('-', '')
                conn = sqlite3.connect(str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db'))
                row = conn.execute(query, (date_alt,)).fetchone()
                conn.close()

            if row is None:
                logger.warning(f"  市场门控: {date} 无市场特征, 默认confidence=0.6")
                return 0.6

            # 构建特征向量
            col_names = ['market_return_20d', 'market_return_10d', 'market_return_5d',
                        'market_volatility_20d', 'market_volatility_10d',
                        'market_up_ratio_20d', 'market_up_ratio_10d',
                        'market_drawdown_20d', 'market_volume_ratio',
                        'market_position_20d', 'market_momentum_20d', 'market_momentum_5d']
            row_dict = dict(zip(col_names, row))

            X_gate = np.array([[row_dict.get(f, 0.0) for f in self._gate_features]])
            confidence = float(self._market_gate.predict(X_gate)[0])
            confidence = np.clip(confidence, 0.0, 1.0)

            return confidence

        except Exception as e:
            logger.warning(f"  市场门控评估失败: {e}, 默认confidence=0.6")
            return 0.6

    def _compute_smoothing_features_inference(self, features_df: pd.DataFrame,
                                               date: str) -> pd.DataFrame:
        """推理时计算排名平滑特征 (从DB加载历史数据)"""
        if features_df is None or len(features_df) == 0:
            return features_df

        # 如果模型没有用排名平滑特征, 直接返回
        if not any(f in (self.feature_cols or []) for f in SMOOTHING_FEATURES):
            for col in SMOOTHING_FEATURES:
                features_df[col] = 0.0
            return features_df

        try:
            conn = sqlite3.connect(str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db'))

            # 获取过去10天的核心特征用于计算变化率
            codes = features_df['code'].tolist()
            if not codes:
                conn.close()
                for col in SMOOTHING_FEATURES:
                    features_df[col] = 0.0
                return features_df

            # 用v39_feature_cache获取5天前的特征
            from datetime import datetime as dt_cls, timedelta as td_cls
            try:
                dt = dt_cls.strptime(date, '%Y-%m-%d')
            except ValueError:
                dt = dt_cls.strptime(date, '%Y%m%d')
            date_5d_ago = (dt - td_cls(days=10)).strftime('%Y-%m-%d')  # 多取几天确保有5个交易日

            # 获取每个code最近5天的features_json
            placeholders = ','.join(['?'] * len(codes))
            hist_query = f"""
            SELECT code, trade_date, features_json
            FROM v39_feature_cache
            WHERE code IN ({placeholders})
              AND trade_date >= ? AND trade_date <= ?
            ORDER BY code, trade_date
            """
            params = codes + [date_5d_ago, date]
            df_hist = pd.read_sql(hist_query, conn, params=params)
            conn.close()

            if df_hist.empty:
                for col in SMOOTHING_FEATURES:
                    features_df[col] = 0.0
                return features_df

            # 解析features_json
            try:
                import orjson
                _loads = orjson.loads
            except ImportError:
                import json
                _loads = json.loads

            core_cols = ['close_to_ma5', 'close_to_ma20', 'rsi_14',
                        'volume_ratio_5', 'kdj_j', 'macd_hist']

            # 每个code: 计算feature_momentum_5d
            code_feature_momentum = {}
            code_signal_consistency = {}
            code_consec_strength = {}

            for code, grp in df_hist.groupby('code'):
                grp = grp.sort_values('trade_date')
                if len(grp) < 2:
                    continue

                parsed = grp['features_json'].apply(_loads).tolist()
                feat_df = pd.DataFrame(parsed)

                available_core = [c for c in core_cols if c in feat_df.columns]
                if not available_core:
                    continue

                vals = feat_df[available_core].values

                # feature_momentum_5d: 最新一行 vs 5天前一行的绝对差
                if len(vals) >= 2:
                    diff = np.abs(vals[-1] - vals[0])
                    code_feature_momentum[code] = float(np.nanmean(diff))

                # signal_consistency_5d: proxy score的std
                proxy = feat_df[available_core].rank(pct=True).mean(axis=1).values
                if len(proxy) >= 2:
                    std_val = np.std(proxy)
                    code_signal_consistency[code] = 1.0 / (1.0 + std_val)

                # consecutive_strength_5d: 暂用proxy最后几天是否在top quartile
                if len(proxy) >= 2:
                    # 用最近5天proxy的平均值作为代理 (越高越好)
                    recent_avg = np.mean(proxy[-5:]) if len(proxy) >= 5 else np.mean(proxy)
                    code_consec_strength[code] = float(recent_avg)

            # 映射到features_df
            med_momentum = np.median(list(code_feature_momentum.values())) if code_feature_momentum else 0.5
            med_consistency = np.median(list(code_signal_consistency.values())) if code_signal_consistency else 0.5
            med_strength = np.median(list(code_consec_strength.values())) if code_consec_strength else 0.5

            features_df['feature_momentum_5d'] = features_df['code'].map(
                code_feature_momentum).fillna(med_momentum)
            features_df['signal_consistency_5d'] = features_df['code'].map(
                code_signal_consistency).fillna(med_consistency)
            features_df['consecutive_strength_5d'] = features_df['code'].map(
                code_consec_strength).fillna(med_strength)

        except Exception as e:
            logger.warning(f"  排名平滑特征计算失败: {e}")
            for col in SMOOTHING_FEATURES:
                features_df[col] = 0.0

        return features_df

    def predict_scores(self, stock_codes: List[str], date: str) -> Dict[str, Dict]:
        """V4.9.1 scoring pipeline:

        1. 市场门控: 评估market_confidence
           - <0.30: 返回空推荐 + 市场警告
           - 0.30~0.50: 正常评分但标记"减仓"
           - >0.50: 正常评分
        2. V4.8.5基础评分 (含Q95)
        3. 注入排名平滑特征
        4. 附加gate_confidence到每只股票的结果中
        """
        # Date format normalization
        if isinstance(date, str) and len(date) == 8 and date.isdigit():
            date_dash = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        else:
            date_dash = date

        # Step 0: 市场门控
        confidence = self._evaluate_market_gate(date_dash)
        self._last_gate_confidence = confidence

        regime = 'normal'
        if confidence < GATE_DONT_BUY:
            regime = 'dont_buy'
            logger.info(f"  🚫 市场门控: confidence={confidence:.3f} < {GATE_DONT_BUY} → 不推荐买入")
        elif confidence < GATE_REDUCE:
            regime = 'reduce'
            logger.info(f"  ⚠️ 市场门控: confidence={confidence:.3f} < {GATE_REDUCE} → 减仓模式")
        else:
            logger.info(f"  ✅ 市场门控: confidence={confidence:.3f} ≥ {GATE_FULL} → 正常推荐")

        # 空仓信号: 返回所有股票score=0, 标记不推荐
        if regime == 'dont_buy':
            results = {}
            for code in stock_codes:
                results[code] = {
                    'score': 0.0,
                    'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0, 'pred_15d': 0,
                    'rank_score': 0,
                    'gate_confidence': confidence,
                    'gate_regime': 'dont_buy',
                    'gate_message': f'市场信心不足({confidence:.1%}), 建议空仓观望',
                    'head_rank': 9999,
                    'in_head_pool': False,
                }
            return results

        # Step 1: 注入排名平滑特征 (通过monkey-patch)
        original_compute = self._compute_v481_new_factors

        def _compute_with_smoothing(features_df, date_arg):
            features_df = original_compute(features_df, date_arg)
            features_df = self._compute_smoothing_features_inference(features_df, date_arg)
            return features_df

        self._compute_v481_new_factors = _compute_with_smoothing
        try:
            # Step 2: V4.8.5基础评分 (含Q95 Widen-then-Concentrate)
            results = super().predict_scores(stock_codes, date)
        finally:
            self._compute_v481_new_factors = original_compute

        # Step 3: 附加门控信息
        for code, data in results.items():
            data['gate_confidence'] = confidence
            data['gate_regime'] = regime
            if regime == 'reduce':
                data['gate_message'] = f'市场信心偏低({confidence:.1%}), 建议减仓'

        return results

    def get_gate_confidence(self) -> Optional[float]:
        """返回上一次predict_scores的市场门控信心值"""
        return self._last_gate_confidence

    def get_recommended_top_n(self, base_top_n: int = 10) -> int:
        """根据市场门控信心值调整推荐数量

        confidence < 0.30 → 0 (不推荐)
        0.30 ≤ confidence < 0.50 → base_top_n // 2
        confidence ≥ 0.50 → base_top_n
        """
        if self._last_gate_confidence is None:
            return base_top_n

        c = self._last_gate_confidence
        if c < GATE_DONT_BUY:
            return 0
        elif c < GATE_REDUCE:
            return max(1, base_top_n // 2)
        else:
            return base_top_n
