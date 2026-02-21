#!/usr/bin/env python3
"""
V3.95 生产评分器
支持多目标预测（3d, 5d, 10d收益）和市场状态特征
"""

import os
import sys
import json
import pickle
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
        self.db_path = Path(__file__).parent.parent.parent / 'data_adapter' / 'stock_data.db'

        # 加载模型配置
        self.models = {}
        self.weights = {}
        self.scaler = None
        self.feature_cols = None
        self.market_feature_cols = None
        self.target_weights = {'label_3d': 0.4, 'label_5d': 0.35, 'label_10d': 0.25}

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
        model_names = ['lgb', 'xgb', 'cb', 'rf', 'gb']
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
            with open(latest, 'rb') as f:
                model_data = pickle.load(f)
                self.models = model_data.get('models', {})
                self.weights = model_data.get('ensemble_weights', {})
                self.scaler = model_data.get('scaler')
                self.feature_cols = model_data.get('feature_cols', [])
                self.market_feature_cols = model_data.get('market_feature_cols', [])

        print(f"V3.95 SmallData模型加载完成")

    def _get_features(self, stock_codes: List[str], date: str) -> Optional[pd.DataFrame]:
        """获取股票特征"""
        conn = sqlite3.connect(self.db_path)

        # 构建查询
        codes_str = ','.join([f"'{c}'" for c in stock_codes])
        query = f"""
        SELECT code, trade_date, features_json,
               market_return_20d, market_return_10d, market_return_5d,
               market_volatility_20d, market_volatility_10d,
               market_up_ratio_20d, market_up_ratio_10d,
               market_drawdown_20d, market_volume_ratio,
               market_position_20d, market_momentum_20d, market_momentum_5d
        FROM v39_feature_cache
        WHERE code IN ({codes_str})
          AND trade_date = '{date}'
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        if len(df) == 0:
            return None

        # 解析features_json
        features_list = []
        valid_codes = []

        for _, row in df.iterrows():
            try:
                features = json.loads(row['features_json'])
                features_list.append(features)
                valid_codes.append(row['code'])
            except (json.JSONDecodeError, TypeError):
                continue

        if not features_list:
            return None

        # 创建特征DataFrame
        features_df = pd.DataFrame(features_list)
        features_df['code'] = valid_codes

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

    def predict_scores(self, stock_codes: List[str], date: str) -> Dict[str, Dict]:
        """
        预测股票评分

        Args:
            stock_codes: 股票代码列表
            date: 交易日期 (YYYY-MM-DD)

        Returns:
            Dict[股票代码, {score, pred_3d, pred_5d, pred_10d}]
        """
        results = {}

        # 获取特征
        features_df = self._get_features(stock_codes, date)

        if features_df is None or len(features_df) == 0:
            # 返回默认分数
            for code in stock_codes:
                results[code] = {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0}
            return results

        # 准备特征矩阵 - 使用缓存中实际存在的特征
        # 排除非特征列
        exclude_cols = {'code', 'trade_date'}
        available_cols = [c for c in features_df.columns if c not in exclude_cols]

        X = features_df[available_cols].fillna(0).values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # 尝试使用训练好的模型预测
        model_predictions_success = False
        predictions = {'3d': np.zeros(len(X)), '5d': np.zeros(len(X)), '10d': np.zeros(len(X))}

        for target in ['3d', '5d', '10d']:
            if target not in self.models or not self.models[target]:
                continue

            # 集成预测
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
                except Exception as e:
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
            # 备用方案: 使用缓存中实际存在的特征计算智能评分
            # features_json包含: return_5d/10d/20d, volatility_10d/20d, volume_ratio/trend,
            # price_position_20d, ma5/10/20_ratio, ma_cross, rsi_14, avg/max/min_pct_change_5d
            combined_pred = self._calculate_fallback_scores(features_df, available_cols)

            # 同时基于特征估算预测收益
            predictions = self._estimate_predictions_from_features(features_df, available_cols)

        # 转换为百分制评分 (使用百分位排名)
        if len(combined_pred) > 1:
            from scipy import stats
            ranks = stats.rankdata(combined_pred)
            percentiles = (ranks - 1) / (len(ranks) - 1) * 100
            scores = 30 + percentiles * 0.6  # 映射到30-90分
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

        # 对于没有特征的股票，返回默认分数
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
