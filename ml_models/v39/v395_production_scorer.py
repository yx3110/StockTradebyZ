#!/usr/bin/env python3
"""
V3.95 生产评分器
支持多目标预测（3d, 5d, 10d收益）和市场状态特征
"""

import os
import sys
import json
import pickle
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sqlite3
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from core.config import get_db_path as _get_db_path
    _DEFAULT_DB_PATH = _get_db_path()
except ImportError:
    _DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / 'data_adapter' / 'stock_data.db'


class V395ProductionScorer:
    """V3.95 生产评分器"""

    def __init__(self, model_type: str = 'rolling'):
        """
        初始化V3.95评分器

        Args:
            model_type: 'rolling' 或 'small_data'
        """
        self.model_type = model_type
        self.model_dir = Path(__file__).parent.parent.parent / 'ml_models' / 'trained_models' / 'v395'
        self.db_path = _DEFAULT_DB_PATH

        # 加载模型配置
        self.models = {}
        self.weights = {}
        self.scaler = None
        self.feature_cols = None
        self.market_feature_cols = None
        self.target_weights = {'label_3d': 0.4, 'label_5d': 0.35, 'label_10d': 0.25}

        # 截面改进标志 (新模型会设置为True)
        self.rank_normalized = False
        self.cross_sectional_neutralization = False
        self.stock_rank_cols = None  # 需要rank归一化的个股特征列表

        # 双流特征标志 (dual_stream模型同时使用raw+rank特征)
        self.dual_stream = False
        self.stock_feature_cols_raw = None
        self.stock_feature_cols_rank = None

        # 级联Rank标志 (cascade模型顺序预测 3d→5d→10d)
        self.cascade = False
        self.cascade_feature_names = None

        # Robust Z-Score标志 (v2模型: 保留幅度信息的截面归一化)
        self.robust_zscore = False
        self.extra_features_from_daily_basic = None

        self._load_models()

    def _load_models(self):
        """加载模型"""
        if self.model_type == 'rolling':
            self._load_rolling_models()
        else:
            self._load_small_data_model()

    def _load_rolling_models(self):
        """加载滚动训练模型"""
        # 加载权重配置
        weights_path = self.model_dir / 'v395_rolling_weights.json'
        if weights_path.exists():
            with open(weights_path, 'r') as f:
                config = json.load(f)
                self.weights = config.get('ensemble_weights', {})
                self.feature_cols = config.get('feature_cols', [])
                self.market_feature_cols = config.get('market_feature_cols', [])
                self.target_weights = config.get('target_weights', self.target_weights)

        # 加载scaler
        scaler_path = self.model_dir / 'v395_rolling_scaler.pkl'
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

        # 加载各目标的模型
        model_names = ['lgb', 'xgb', 'cb', 'rf', 'hgb', 'gb']
        for target in ['3d', '5d', '10d']:
            self.models[target] = {}
            for name in model_names:
                model_path = self.model_dir / f'v395_rolling_{target}_{name}.pkl'
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        self.models[target][name] = pickle.load(f)

        print(f"V3.95 Rolling模型加载完成: {len(self.models)} 个目标")

    def _load_small_data_model(self):
        """加载小数据版模型"""
        # 查找最新的小数据模型
        model_files = list(self.model_dir.glob('v395_multi_target_*.pkl'))
        if model_files:
            latest = max(model_files, key=lambda f: f.stat().st_mtime)
            try:
                model_data = joblib.load(latest)
            except Exception:
                with open(latest, 'rb') as f:
                    model_data = pickle.load(f)
            raw_models = model_data.get('models', {})
            # 处理嵌套结构: {target: {models: {name: model}, weights: {...}}}
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
            # Phase 2: 优先使用ICIR动态权重（如果模型中包含）
            self.target_weights = model_data.get('dynamic_weights',
                                                  model_data.get('target_weights', self.target_weights))

            # 级联Rank元数据 (最新模型)
            self.cascade = model_data.get('cascade', False)
            self.cascade_feature_names = model_data.get('cascade_feature_names', None)

            # 双流特征元数据
            self.dual_stream = model_data.get('dual_stream', False)
            self.stock_feature_cols_raw = model_data.get('stock_feature_cols_raw', None)
            self.stock_feature_cols_rank = model_data.get('stock_feature_cols_rank', None)

            # 截面改进元数据 (旧模型兼容)
            self.rank_normalized = model_data.get('rank_normalized', False)
            self.cross_sectional_neutralization = model_data.get('cross_sectional_neutralization', False)
            self.stock_rank_cols = model_data.get('stock_feature_cols', None)

            # Robust Z-Score 元数据 (v2模型)
            self.robust_zscore = model_data.get('robust_zscore', False)
            self.extra_features_from_daily_basic = model_data.get('extra_features_from_daily_basic', None)

        if self.robust_zscore:
            suffix = " [robust_zscore+industry_excess]"
        elif self.cascade:
            suffix = " [cascade_rank]"
        elif self.dual_stream:
            suffix = " [dual_stream]"
        elif self.rank_normalized:
            suffix = " [rank_normalized]"
        else:
            suffix = ""
        print(f"V3.95 SmallData模型加载完成: {list(self.models.keys())}{suffix}")

    def _rank_normalize_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """对个股特征做截面Rank归一化（宏观特征保持原值）"""
        if not self.stock_rank_cols:
            return features_df
        rank_cols = [c for c in self.stock_rank_cols if c in features_df.columns]
        if rank_cols:
            features_df[rank_cols] = features_df[rank_cols].rank(pct=True)
            features_df[rank_cols] = features_df[rank_cols].fillna(0.5)
        return features_df

    def _robust_zscore_normalize_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """对个股特征做截面Robust Z-Score归一化 (保留幅度信息)

        z = (x - median) / (MAD * 1.4826), clip[-3, 3]
        优势 vs Rank: PE=5 和 PE=50 有不同的 z 值, 而 rank 只有序数差
        """
        if not self.stock_rank_cols:
            return features_df
        zscore_cols = [c for c in self.stock_rank_cols if c in features_df.columns]
        if not zscore_cols:
            return features_df

        data = features_df[zscore_cols].values.copy()
        for j in range(data.shape[1]):
            col_data = data[:, j]
            valid = col_data[~np.isnan(col_data)]
            if len(valid) < 5:
                data[:, j] = 0.0
                continue
            median = np.nanmedian(col_data)
            mad = np.nanmedian(np.abs(col_data - median)) * 1.4826
            if mad < 1e-8:
                data[:, j] = 0.0
            else:
                data[:, j] = np.clip((col_data - median) / mad, -3, 3)
        features_df[zscore_cols] = data
        features_df[zscore_cols] = features_df[zscore_cols].fillna(0.0)
        return features_df

    def _load_daily_basic_features(self, features_df: pd.DataFrame, date: str) -> pd.DataFrame:
        """从daily_basic加载额外特征 (pe_ttm, pb, ps_ttm, turnover_rate, log_market_cap)"""
        if not self.extra_features_from_daily_basic:
            return features_df

        conn = sqlite3.connect(self.db_path)
        try:
            codes = features_df['code'].tolist()
            placeholders = ','.join(['?' for _ in codes])
            query = f"""
            SELECT s.code, db.pe_ttm, db.pb, db.ps_ttm, db.turnover_rate, db.circ_mv
            FROM daily_basic db
            JOIN securities s ON db.security_id = s.id
            WHERE s.code IN ({placeholders}) AND db.trade_date = ?
            """
            df_basic = pd.read_sql_query(query, conn, params=codes + [date])
        finally:
            conn.close()

        if len(df_basic) > 0:
            features_df = features_df.merge(df_basic, on='code', how='left')
            features_df['log_market_cap'] = np.log1p(features_df['circ_mv'].fillna(0))
            features_df.drop(columns=['circ_mv'], inplace=True, errors='ignore')
            for col in ['pe_ttm', 'pb', 'ps_ttm', 'turnover_rate', 'log_market_cap']:
                if col in features_df.columns:
                    features_df[col] = features_df[col].fillna(features_df[col].median())
        else:
            # 无daily_basic数据, 填0
            for col in ['pe_ttm', 'pb', 'ps_ttm', 'turnover_rate', 'log_market_cap']:
                features_df[col] = 0.0

        return features_df

    def _create_dual_stream_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        双流特征: 保留原始值(raw) + 新增截面Rank列(_rank)

        向后兼容: 仅当 self.dual_stream=True 且 stock_feature_cols_raw 存在时使用
        """
        if not self.stock_feature_cols_raw:
            return features_df
        for col in self.stock_feature_cols_raw:
            rank_col = f"{col}_rank"
            if col in features_df.columns:
                features_df[rank_col] = features_df[col].rank(pct=True).fillna(0.5)
        return features_df

    def _get_features(self, stock_codes: Optional[List[str]], date: str,
                      load_full_cross_section: bool = False) -> Optional[pd.DataFrame]:
        """获取股票特征"""
        conn = sqlite3.connect(self.db_path)

        # 构建查询
        if load_full_cross_section:
            # 加载全截面数据 (用于rank归一化)
            query = """
            SELECT code, trade_date, features_json,
                   market_return_20d, market_return_10d, market_return_5d,
                   market_volatility_20d, market_volatility_10d,
                   market_up_ratio_20d, market_up_ratio_10d,
                   market_drawdown_20d, market_volume_ratio,
                   market_position_20d, market_momentum_20d, market_momentum_5d
            FROM v39_feature_cache
            WHERE trade_date = ?
            """
            params = [date]
        else:
            placeholders = ','.join(['?' for _ in stock_codes])
            query = f"""
            SELECT code, trade_date, features_json,
                   market_return_20d, market_return_10d, market_return_5d,
                   market_volatility_20d, market_volatility_10d,
                   market_up_ratio_20d, market_up_ratio_10d,
                   market_drawdown_20d, market_volume_ratio,
                   market_position_20d, market_momentum_20d, market_momentum_5d
            FROM v39_feature_cache
            WHERE code IN ({placeholders})
              AND trade_date = ?
            """
            params = list(stock_codes) + [date]

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if len(df) == 0:
            return None

        # 解析features_json (向量化)
        parsed = df['features_json'].apply(
            lambda s: json.loads(s) if isinstance(s, str) else None
        )
        valid_mask = parsed.notna()
        if not valid_mask.any():
            return None

        features_df = pd.DataFrame(parsed[valid_mask].tolist())
        features_df['code'] = df.loc[valid_mask, 'code'].values

        # 添加市场特征
        market_cols = [c for c in df.columns if c.startswith('market_')]
        df_market = df[df['code'].isin(valid_codes)][['code'] + market_cols].reset_index(drop=True)
        features_df = features_df.merge(df_market, on='code', how='left')

        return features_df

    def _calculate_fallback_scores(self, features_df: pd.DataFrame, available_cols: List[str]) -> np.ndarray:
        """
        备用评分计算：使用缓存中实际存在的16个特征计算有意义的综合评分

        特征权重设计（基于对未来收益的预测能力）：
        - 动量因子 (return_5d/10d/20d): 40% - 短期动量是最强预测因子
        - 技术面 (rsi_14, ma_cross, price_position): 25% - 技术指标的辅助判断
        - 成交量 (volume_ratio, volume_trend): 20% - 量价配合验证
        - 风险调整 (volatility): 15% - 适度波动为佳
        """
        n_stocks = len(features_df)
        scores = np.zeros(n_stocks)

        # 1. 动量因子评分 (40%) - 使用return_*d字段
        momentum_score = np.zeros(n_stocks)
        momentum_weights = {'return_5d': 0.5, 'return_10d': 0.3, 'return_20d': 0.2}
        total_momentum_weight = 0

        for col, weight in momentum_weights.items():
            if col in available_cols:
                # 动量越高越好，使用percentile转换
                values = features_df[col].fillna(0).values
                momentum_score += weight * values
                total_momentum_weight += weight

        if total_momentum_weight > 0:
            momentum_score /= total_momentum_weight

        # 2. 技术面评分 (25%)
        tech_score = np.zeros(n_stocks)

        # RSI评分：50附近最优，远离50惩罚
        if 'rsi_14' in available_cols:
            rsi = features_df['rsi_14'].fillna(50).values
            # RSI 30-70区间给正分，超买超卖给负分
            rsi_score = 1 - np.abs(rsi - 50) / 50  # 归一化到0-1
            tech_score += 0.3 * rsi_score

        # 均线位置评分：价格在均线上方为正
        ma_ratio_cols = ['ma5_ratio', 'ma10_ratio', 'ma20_ratio']
        ma_count = 0
        for col in ma_ratio_cols:
            if col in available_cols:
                ma_ratio = features_df[col].fillna(0).values
                # 价格在均线上方(正值)给正分，控制在-0.1到0.1范围
                tech_score += 0.2 * np.clip(ma_ratio * 10, -1, 1)
                ma_count += 1

        # 价格位置评分：中位附近较好
        if 'price_position_20d' in available_cols:
            pos = features_df['price_position_20d'].fillna(0.5).values
            # 0.3-0.7区间最优
            pos_score = 1 - 2 * np.abs(pos - 0.5)
            tech_score += 0.1 * np.clip(pos_score, 0, 1)

        # 3. 成交量评分 (20%)
        volume_score = np.zeros(n_stocks)

        if 'volume_ratio' in available_cols:
            vol_ratio = features_df['volume_ratio'].fillna(1).values
            # 放量(1.0-2.5)为佳，缩量或过度放量不佳
            vol_score = np.where(
                (vol_ratio >= 1.0) & (vol_ratio <= 2.5),
                (vol_ratio - 1) / 1.5,  # 1.0-2.5映射到0-1
                np.where(vol_ratio < 1.0, vol_ratio - 1, -(vol_ratio - 2.5) / 2)
            )
            volume_score += 0.6 * np.clip(vol_score, -1, 1)

        if 'volume_trend' in available_cols:
            vol_trend = features_df['volume_trend'].fillna(0).values
            # 正向趋势为佳
            volume_score += 0.4 * np.clip(vol_trend, -1, 1)

        # 4. 风险调整评分 (15%)
        risk_score = np.zeros(n_stocks)

        volatility_cols = ['volatility_10d', 'volatility_20d']
        vol_count = 0
        for col in volatility_cols:
            if col in available_cols:
                volatility = features_df[col].fillna(0.1).values
                # 适度波动(0.05-0.15)为佳，过高或过低都不好
                vol_optimal = np.where(
                    (volatility >= 0.05) & (volatility <= 0.15),
                    1 - np.abs(volatility - 0.1) / 0.05,
                    np.where(volatility < 0.05, volatility / 0.05, 0.15 / volatility)
                )
                risk_score += vol_optimal
                vol_count += 1

        if vol_count > 0:
            risk_score /= vol_count

        # 综合评分
        scores = (
            0.40 * momentum_score +   # 动量最重要
            0.25 * tech_score +       # 技术面次之
            0.20 * volume_score +     # 成交量配合
            0.15 * risk_score         # 风险调整
        )

        return scores

    def _estimate_predictions_from_features(self, features_df: pd.DataFrame, available_cols: List[str]) -> Dict[str, np.ndarray]:
        """
        基于现有特征估算3d/5d/10d预测收益
        使用简单的线性外推和均值回归结合
        """
        n_stocks = len(features_df)
        predictions = {
            '3d': np.zeros(n_stocks),
            '5d': np.zeros(n_stocks),
            '10d': np.zeros(n_stocks)
        }

        # 获取历史收益率
        return_5d = features_df['return_5d'].fillna(0).values if 'return_5d' in available_cols else np.zeros(n_stocks)
        return_10d = features_df['return_10d'].fillna(0).values if 'return_10d' in available_cols else np.zeros(n_stocks)
        return_20d = features_df['return_20d'].fillna(0).values if 'return_20d' in available_cols else np.zeros(n_stocks)

        # 计算动量强度
        momentum = 0.5 * return_5d + 0.3 * return_10d + 0.2 * return_20d

        # 均值回归因子（过去涨太多可能回调）
        mean_reversion = -0.1 * return_20d

        # 短期动量延续 + 轻微均值回归
        # 3日预测：强动量延续
        predictions['3d'] = 0.6 * return_5d / 5 * 3 + 0.4 * momentum * 0.3

        # 5日预测：动量延续 + 均值回归
        predictions['5d'] = 0.5 * return_5d + 0.3 * momentum * 0.5 + 0.2 * mean_reversion

        # 10日预测：更多均值回归
        predictions['10d'] = 0.4 * return_10d + 0.3 * momentum * 0.8 + 0.3 * mean_reversion

        # 限制预测范围在合理区间 (-10%, +10%)
        for target in predictions:
            predictions[target] = np.clip(predictions[target], -0.10, 0.10)

        return predictions

    def _cascade_ensemble_predict(self, X_input, models, weights):
        """
        对单个目标的所有模型做加权集成预测

        Args:
            X_input: 特征矩阵
            models: {model_name: model} 字典
            weights: {model_name: weight} 字典

        Returns:
            (ensemble_pred, success): 集成预测结果和是否成功标志
        """
        target_pred = np.zeros(len(X_input))
        total_weight = 0
        success_count = 0

        for name, model in models.items():
            try:
                pred = model.predict(X_input)
                weight = weights.get(name, 0.2)
                target_pred += weight * pred
                total_weight += weight
                success_count += 1
            except Exception:
                continue

        if total_weight > 0:
            target_pred /= total_weight
            return target_pred, success_count > 0
        return target_pred, False

    def predict_scores(self, stock_codes: List[str], date: str) -> Dict[str, Dict]:
        """
        预测股票评分

        Args:
            stock_codes: 股票代码列表
            date: 交易日期 (YYYY-MM-DD 或 YYYYMMDD)

        Returns:
            Dict[股票代码, {score, pred_3d, pred_5d, pred_10d}]
        """
        results = {}

        if not stock_codes:
            return results

        # 日期格式标准化
        if isinstance(date, str) and len(date) == 8 and date.isdigit():
            date = f"{date[:4]}-{date[4:6]}-{date[6:]}"

        # 获取特征 — 五种路径:
        # 1. robust_zscore: 全截面加载 → z-score归一化 → 加载daily_basic → 过滤
        # 2. cascade: 全截面加载 → rank替换 → 过滤 → 级联推理
        # 3. dual_stream: 全截面加载 → 生成raw+rank双流特征 → 过滤
        # 4. rank_normalized: 全截面加载 → rank替换 → 过滤
        # 5. raw: 仅加载目标股票
        if self.robust_zscore:
            features_df = self._get_features(stock_codes, date, load_full_cross_section=True)
            if features_df is not None and len(features_df) > 0:
                features_df = self._robust_zscore_normalize_features(features_df)
                features_df = self._load_daily_basic_features(features_df, date)
                features_df = features_df[features_df['code'].isin(stock_codes)].copy()
        elif self.cascade or self.rank_normalized:
            features_df = self._get_features(stock_codes, date, load_full_cross_section=True)
            if features_df is not None and len(features_df) > 0:
                features_df = self._create_dual_stream_features(features_df)
                features_df = features_df[features_df['code'].isin(stock_codes)].copy()
        else:
            features_df = self._get_features(stock_codes, date)

        if features_df is None or len(features_df) == 0:
            for code in stock_codes:
                results[code] = {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0}
            return results

        # 准备特征矩阵 - 使用模型训练时的特征列顺序
        exclude_cols = {'code', 'trade_date'}
        if self.feature_cols:
            missing = [c for c in self.feature_cols if c not in features_df.columns]
            if missing:
                if len(missing) > len(self.feature_cols) * 0.3:
                    logger.warning(f"⚠️ {len(missing)}/{len(self.feature_cols)} 特征缺失, 预测质量可能下降: {missing[:5]}...")
                for col in missing:
                    features_df[col] = 0
            available_cols = self.feature_cols
        else:
            available_cols = [c for c in features_df.columns if c not in exclude_cols]

        X = features_df[available_cols].fillna(0).values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # 尝试使用训练好的模型预测
        model_predictions_success = False
        predictions = {'3d': np.zeros(len(X)), '5d': np.zeros(len(X)), '10d': np.zeros(len(X))}

        if self.cascade:
            # 级联推理: 3d → 5d → 10d (顺序预测, 前级预测作为后级输入)
            # 1. 3d预测 (基础特征)
            if '3d' in self.models and self.models['3d']:
                w3d = self.weights.get('label_3d', self.weights.get('3d', {}))
                if isinstance(w3d, dict):
                    predictions['3d'], ok_3d = self._cascade_ensemble_predict(X, self.models['3d'], w3d)
                    if ok_3d:
                        model_predictions_success = True

            # 2. 5d预测 (基础特征 + cascade_pred_3d)
            if '5d' in self.models and self.models['5d']:
                X_5d = np.column_stack([X, predictions['3d']])
                w5d = self.weights.get('label_5d', self.weights.get('5d', {}))
                if isinstance(w5d, dict):
                    predictions['5d'], ok_5d = self._cascade_ensemble_predict(X_5d, self.models['5d'], w5d)
                    if ok_5d:
                        model_predictions_success = True

            # 3. 10d预测 (基础特征 + cascade_pred_3d + cascade_pred_5d)
            if '10d' in self.models and self.models['10d']:
                X_10d = np.column_stack([X, predictions['3d'], predictions['5d']])
                w10d = self.weights.get('label_10d', self.weights.get('10d', {}))
                if isinstance(w10d, dict):
                    predictions['10d'], ok_10d = self._cascade_ensemble_predict(X_10d, self.models['10d'], w10d)
                    if ok_10d:
                        model_predictions_success = True
        else:
            # 非级联: 独立预测各目标 (旧模型兼容)
            for target in ['3d', '5d', '10d']:
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

        # 计算综合评分
        if model_predictions_success:
            combined_pred = (
                self.target_weights['label_3d'] * predictions['3d'] +
                self.target_weights['label_5d'] * predictions['5d'] +
                self.target_weights['label_10d'] * predictions['10d']
            )
        else:
            combined_pred = self._calculate_fallback_scores(features_df, available_cols)
            predictions = self._estimate_predictions_from_features(features_df, available_cols)

        # 转换为百分制评分 (使用百分位排名)
        if len(combined_pred) > 1:
            from scipy import stats
            ranks = stats.rankdata(combined_pred)
            percentiles = (ranks - 1) / (len(ranks) - 1) * 100
            scores = 30 + percentiles * 0.6
        else:
            scores = np.array([60.0])

        # 构建结果
        codes = features_df['code'].tolist()
        for i, code in enumerate(codes):
            results[code] = {
                'score': float(scores[i]),
                'pred_3d': float(predictions['3d'][i]) if i < len(predictions['3d']) else 0,
                'pred_5d': float(predictions['5d'][i]) if i < len(predictions['5d']) else 0,
                'pred_10d': float(predictions['10d'][i]) if i < len(predictions['10d']) else 0
            }

        for code in stock_codes:
            if code not in results:
                results[code] = {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0}

        return results

    def preload_feature_cache(self, dates: List[str]) -> Dict[str, pd.DataFrame]:
        """
        批量预加载多个日期的特征缓存（用于批量报告生成）

        一条 SQL WHERE trade_date IN (...) 查询所有日期的 v39_feature_cache，
        包含 features_json 和 market_* 列。

        Args:
            dates: 日期列表 ['YYYY-MM-DD', ...]

        Returns:
            {date: features_DataFrame} 字典，每个 DataFrame 包含 code 列和所有特征列
        """
        result = {d: None for d in dates}
        if not dates:
            return result

        conn = sqlite3.connect(self.db_path)
        try:
            placeholders = ','.join(['?' for _ in dates])
            query = f"""
            SELECT code, trade_date, features_json,
                   market_return_20d, market_return_10d, market_return_5d,
                   market_volatility_20d, market_volatility_10d,
                   market_up_ratio_20d, market_up_ratio_10d,
                   market_drawdown_20d, market_volume_ratio,
                   market_position_20d, market_momentum_20d, market_momentum_5d
            FROM v39_feature_cache
            WHERE trade_date IN ({placeholders})
            """
            df = pd.read_sql_query(query, conn, params=dates)
            conn.close()

            if df.empty:
                return result

            # 按日期分组处理
            for date, date_df in df.groupby('trade_date'):
                features_list = []
                parsed = date_df['features_json'].apply(
                    lambda s: json.loads(s) if isinstance(s, str) else None
                )
                valid_mask = parsed.notna()
                if not valid_mask.any():
                    continue

                features_df = pd.DataFrame(parsed[valid_mask].tolist())
                valid_codes = date_df.loc[valid_mask, 'code'].values
                features_df['code'] = valid_codes

                # 添加市场特征
                market_cols = [c for c in date_df.columns if c.startswith('market_')]
                df_market = date_df[date_df['code'].isin(valid_codes)][['code'] + market_cols].reset_index(drop=True)
                features_df = features_df.merge(df_market, on='code', how='left')

                result[date] = features_df

            total = sum(len(v) for v in result.values() if v is not None)
            print(f"V3.95特征缓存预加载完成: {len(dates)}天, {total}条记录")

        except Exception as e:
            print(f"V3.95特征缓存预加载失败: {e}")
            if conn:
                conn.close()

        return result

    def predict_scores_from_preloaded(self, stock_codes: List[str], date: str,
                                       features_df: Optional[pd.DataFrame]) -> Dict[str, Dict]:
        """
        使用预加载的特征进行评分预测（跳过SQL查询）

        Args:
            stock_codes: 股票代码列表
            date: 交易日期
            features_df: 预加载的特征 DataFrame（含 code 列），可为 None

        Returns:
            Dict[股票代码, {score, pred_3d, pred_5d, pred_10d}]
        """
        results = {}

        # 如果没有预加载数据，过滤出请求的股票
        if features_df is None or len(features_df) == 0:
            for code in stock_codes:
                results[code] = {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0}
            return results

        # 五种路径: robust_zscore > cascade > dual_stream > rank_normalized > raw
        if self.robust_zscore:
            features_df = self._robust_zscore_normalize_features(features_df.copy())
            features_df = self._load_daily_basic_features(features_df, date)
        elif self.cascade or self.rank_normalized:
            features_df = self._rank_normalize_features(features_df.copy())
        elif self.dual_stream:
            features_df = self._create_dual_stream_features(features_df.copy())

        # 过滤出请求的股票代码
        mask = features_df['code'].isin(stock_codes)
        filtered_df = features_df[mask].copy()

        if len(filtered_df) == 0:
            for code in stock_codes:
                results[code] = {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0}
            return results

        # 以下逻辑复用 predict_scores 的核心路径
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

        model_predictions_success = False
        predictions = {'3d': np.zeros(len(X)), '5d': np.zeros(len(X)), '10d': np.zeros(len(X))}

        if self.cascade:
            # 级联推理: 3d → 5d → 10d
            if '3d' in self.models and self.models['3d']:
                w3d = self.weights.get('label_3d', self.weights.get('3d', {}))
                if isinstance(w3d, dict):
                    predictions['3d'], ok_3d = self._cascade_ensemble_predict(X, self.models['3d'], w3d)
                    if ok_3d:
                        model_predictions_success = True

            if '5d' in self.models and self.models['5d']:
                X_5d = np.column_stack([X, predictions['3d']])
                w5d = self.weights.get('label_5d', self.weights.get('5d', {}))
                if isinstance(w5d, dict):
                    predictions['5d'], ok_5d = self._cascade_ensemble_predict(X_5d, self.models['5d'], w5d)
                    if ok_5d:
                        model_predictions_success = True

            if '10d' in self.models and self.models['10d']:
                X_10d = np.column_stack([X, predictions['3d'], predictions['5d']])
                w10d = self.weights.get('label_10d', self.weights.get('10d', {}))
                if isinstance(w10d, dict):
                    predictions['10d'], ok_10d = self._cascade_ensemble_predict(X_10d, self.models['10d'], w10d)
                    if ok_10d:
                        model_predictions_success = True
        else:
            # 非级联: 独立预测各目标
            for target in ['3d', '5d', '10d']:
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
                self.target_weights['label_3d'] * predictions['3d'] +
                self.target_weights['label_5d'] * predictions['5d'] +
                self.target_weights['label_10d'] * predictions['10d']
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

        codes = filtered_df['code'].tolist()
        for i, code in enumerate(codes):
            results[code] = {
                'score': float(scores[i]),
                'pred_3d': float(predictions['3d'][i]) if i < len(predictions['3d']) else 0,
                'pred_5d': float(predictions['5d'][i]) if i < len(predictions['5d']) else 0,
                'pred_10d': float(predictions['10d'][i]) if i < len(predictions['10d']) else 0
            }

        for code in stock_codes:
            if code not in results:
                results[code] = {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0}

        return results

    def predict_scores_with_ranking(self, stock_codes: List[str], date: str) -> Dict[str, Dict]:
        """
        使用百分位排名的评分预测（与V3.94接口兼容）
        """
        return self.predict_scores(stock_codes, date)


def test_scorer():
    """测试评分器"""
    scorer = V395ProductionScorer(model_type='rolling')

    # 测试股票
    test_codes = ['000001', '000002', '600000', '600519', '000858']
    test_date = '2025-11-28'

    results = scorer.predict_scores(test_codes, test_date)

    print(f"\n测试日期: {test_date}")
    print("=" * 60)
    for code, data in results.items():
        print(f"{code}: 综合评分={data['score']:.1f}, "
              f"3d={data['pred_3d']:.4f}, 5d={data['pred_5d']:.4f}, 10d={data['pred_10d']:.4f}")


if __name__ == '__main__':
    test_scorer()
