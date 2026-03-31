#!/usr/bin/env python3
"""
V3.91 市场自适应版本 - Mixture of Experts

核心思路：
1. 不抛弃任何特征，而是让模型学会在不同市场环境下使用不同特征
2. 市场状态检测器 - 识别当前是什么市场环境
3. 多专家模型 - 每个专家擅长特定市场环境
4. 门控网络 - 动态决定如何混合专家预测

目标：IC差距 < 30%，同时保持较高的验证IC
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import spearmanr, rankdata
import warnings

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings('ignore')

# 设置日志
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/v391_adaptive_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """市场状态检测器"""

    # 用于检测市场状态的特征
    REGIME_FEATURES = [
        'advance_decline_ratio',   # 市场广度
        'market_attention_score',  # 市场关注度
        'atr_percent_14',          # 波动率
        'adx_14',                  # 趋势强度
        'rsi_14',                  # 超买超卖
        'bb_width_20',             # 布林带宽度
    ]

    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        self.regime_names = ['低波震荡', '趋势上涨', '高波下跌']

    def fit(self, df: pd.DataFrame, feature_cols: List[str]):
        """训练市场状态检测器"""
        # 提取市场状态特征
        regime_cols = [c for c in self.REGIME_FEATURES if c in feature_cols]
        if len(regime_cols) < 3:
            logger.warning(f"市场状态特征不足，使用全部特征的聚类")
            regime_cols = feature_cols[:6]

        # 按日期聚合市场状态
        market_state = df.groupby('trade_date')[regime_cols].mean()

        # 标准化
        X = self.scaler.fit_transform(market_state.values)
        X = np.nan_to_num(X, nan=0.0)

        # 聚类
        self.kmeans.fit(X)

        # 分析每个状态的特征
        labels = self.kmeans.labels_
        for i in range(self.n_regimes):
            mask = labels == i
            if mask.sum() > 0:
                cluster_mean = market_state.iloc[mask].mean()
                logger.info(f"  状态 {i} ({mask.sum()}天): "
                           f"广度={cluster_mean.get('advance_decline_ratio', 0):.2f}, "
                           f"波动={cluster_mean.get('atr_percent_14', 0):.2f}, "
                           f"趋势={cluster_mean.get('adx_14', 0):.2f}")

        return self

    def predict(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """预测每个样本的市场状态"""
        regime_cols = [c for c in self.REGIME_FEATURES if c in feature_cols]
        if len(regime_cols) < 3:
            regime_cols = feature_cols[:6]

        # 按日期聚合
        market_state = df.groupby('trade_date')[regime_cols].mean()

        # 标准化
        X = self.scaler.transform(market_state.values)
        X = np.nan_to_num(X, nan=0.0)

        # 预测
        date_regimes = dict(zip(market_state.index, self.kmeans.predict(X)))

        # 映射回每个样本
        return np.array([date_regimes.get(d, 0) for d in df['trade_date']])


class ExpertModel:
    """单个专家模型"""

    def __init__(self, name: str, alpha: float = 10.0):
        self.name = name
        self.alpha = alpha
        self.model = Ridge(alpha=alpha, random_state=42)
        self.scaler = RobustScaler()
        self.feature_weights = None  # 学习到的特征重要性

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        """训练专家，支持样本权重"""
        X_scaled = self.scaler.fit_transform(X)
        y_rank = (rankdata(y) - 1) / (len(y) - 1) - 0.5
        self.model.fit(X_scaled, y_rank, sample_weight=sample_weight)
        self.feature_weights = np.abs(self.model.coef_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class GatingNetwork:
    """门控网络 - 决定如何混合专家"""

    def __init__(self, n_experts: int):
        self.n_experts = n_experts
        self.model = LogisticRegression(
            multi_class='multinomial',
            max_iter=1000,
            random_state=42
        )
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, expert_performances: np.ndarray):
        """
        训练门控网络
        X: 市场状态特征
        expert_performances: 每个专家在每个样本上的表现 (n_samples, n_experts)
        """
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        # 选择每个样本表现最好的专家作为标签
        best_experts = np.argmax(expert_performances, axis=1)

        self.model.fit(X_scaled, best_experts)
        return self

    def predict_weights(self, X: np.ndarray) -> np.ndarray:
        """预测专家权重（软分配）"""
        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        return self.model.predict_proba(X_scaled)


class AdaptiveV391Trainer:
    """市场自适应V3.91训练器"""

    PERIOD_WEIGHTS = {'5d': 0.40, '10d': 0.35, '15d': 0.25}

    def compute_time_decay_weights(self, dates: pd.Series, half_life_days: int = 180) -> np.ndarray:
        """
        计算时间衰减权重

        half_life_days: 半衰期（天）- 多少天前的数据权重衰减到一半
        """
        dates_dt = pd.to_datetime(dates)
        max_date = dates_dt.max()
        days_ago = np.array([(max_date - d).days for d in dates_dt])

        # 指数衰减: weight = exp(-ln(2) * days_ago / half_life)
        decay_rate = np.log(2) / half_life_days
        weights = np.exp(-decay_rate * days_ago)

        # 归一化，使权重均值为1
        weights = weights / weights.mean()

        return weights

    # 按特征类型分组
    FEATURE_GROUPS = {
        '动量特征': ['rsi_14', 'momentum_20', 'roc_10', 'tsi', 'willr_14',
                   'stoch_k', 'stoch_d', 'cci_20', 'cmo_14'],
        '趋势特征': ['supertrend_signal', 'adx_14', 'aroon_up_25', 'aroon_down_25',
                   'ichimoku_base', 'ichimoku_conv', 'psar_trend', 'vortex_pos', 'vortex_neg'],
        '波动特征': ['atr_percent_14', 'bb_width_20', 'kc_width_20', 'donchian_width_20',
                   'volatility_20', 'true_range_pct'],
        '成交量特征': ['cmf_20', 'ad_line_change_5', 'obv_slope_10', 'vwap_deviation',
                    'volume_ratio_20', 'mfi_14'],
        '市场特征': ['advance_decline_ratio', 'market_attention_score', 'sector_momentum'],
        '基本面特征': ['pe_percentile', 'pb_percentile', 'ps_percentile', 'market_cap_rank',
                    'turnover_rate_percentile', 'gross_margin_trend', 'roe_trend']
    }

    def __init__(self, db_path: str = 'data_adapter/stock_data.db', n_experts: int = 3):
        self.db_path = db_path
        self.n_experts = n_experts
        self.regime_detector = MarketRegimeDetector(n_regimes=n_experts)

    def load_data(self, start_date: str = '2023-01-01') -> Dict:
        """加载数据"""
        import sqlite3

        logger.info("=" * 80)
        logger.info(f"📥 加载数据 (从 {start_date} 开始)")
        logger.info("=" * 80)

        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT code, trade_date, features_json as features,
                   label_5d as future_return_5d,
                   label_10d as future_return_10d,
                   label_15d as future_return_15d
            FROM v39_feature_cache
            WHERE trade_date >= ?
            AND features_json IS NOT NULL
            AND label_5d IS NOT NULL
            AND label_10d IS NOT NULL
            AND label_15d IS NOT NULL
        """

        df = pd.read_sql_query(query, conn, params=[start_date])
        conn.close()

        logger.info(f"✅ 加载 {len(df):,} 个样本")

        # 解析特征
        features_list = []
        valid_indices = []

        for idx, row in df.iterrows():
            try:
                features = json.loads(row['features'])
                if isinstance(features, dict):
                    features_list.append(features)
                    valid_indices.append(idx)
            except Exception:
                continue

        df_valid = df.loc[valid_indices].copy()
        feature_df = pd.DataFrame(features_list)

        # 获取所有可用特征
        all_features = []
        for group_features in self.FEATURE_GROUPS.values():
            for f in group_features:
                if f in feature_df.columns and f not in all_features:
                    all_features.append(f)

        # 添加其他数值特征
        for col in feature_df.columns:
            if col not in all_features and feature_df[col].dtype in ['float64', 'int64']:
                all_features.append(col)

        logger.info(f"📊 使用 {len(all_features)} 个特征")

        for col in all_features:
            if col in feature_df.columns:
                df_valid[col] = feature_df[col].values

        # 按时间排序
        df_valid = df_valid.sort_values('trade_date').reset_index(drop=True)

        # 时序分割
        unique_dates = df_valid['trade_date'].unique()
        split_idx = int(len(unique_dates) * 0.8)
        split_date = unique_dates[split_idx]

        train_mask = df_valid['trade_date'] < split_date
        val_mask = df_valid['trade_date'] >= split_date

        logger.info(f"📅 时间范围: {unique_dates[0]} ~ {unique_dates[-1]}")
        logger.info(f"📊 训练集: {train_mask.sum():,} 样本 (< {split_date})")
        logger.info(f"📊 验证集: {val_mask.sum():,} 样本 (>= {split_date})")

        return {
            'df': df_valid,
            'feature_names': all_features,
            'train_mask': train_mask,
            'val_mask': val_mask,
            'split_date': split_date
        }

    def train_period_adaptive(self, period: str, data: Dict,
                               half_life_days: int = 180) -> Dict:
        """使用自适应专家混合训练单个周期"""

        logger.info(f"\n{'='*60}")
        logger.info(f"🔹 训练 {period} 周期 (Mixture of Experts + 时间衰减)")
        logger.info(f"    半衰期: {half_life_days}天")
        logger.info(f"{'='*60}")

        df = data['df']
        feature_names = data['feature_names']
        train_mask = data['train_mask']
        val_mask = data['val_mask']
        label_col = f'future_return_{period}'

        # 提取特征和标签
        X_all = df[feature_names].values
        y_all = df[label_col].values
        X_all = np.nan_to_num(X_all, nan=0.0)

        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        X_val = X_all[val_mask]
        y_val = y_all[val_mask]

        df_train = df[train_mask].copy()
        df_val = df[val_mask].copy()

        # 计算时间衰减权重
        train_weights = self.compute_time_decay_weights(
            df_train['trade_date'], half_life_days
        )
        logger.info(f"  时间权重范围: {train_weights.min():.3f} ~ {train_weights.max():.3f}")

        # Step 1: 检测市场状态
        logger.info("\n📊 Step 1: 检测市场状态")
        self.regime_detector.fit(df_train, feature_names)
        train_regimes = self.regime_detector.predict(df_train, feature_names)
        val_regimes = self.regime_detector.predict(df_val, feature_names)

        for i in range(self.n_experts):
            train_count = (train_regimes == i).sum()
            val_count = (val_regimes == i).sum()
            logger.info(f"  状态 {i}: 训练集 {train_count:,} 样本, 验证集 {val_count:,} 样本")

        # Step 2: 训练每个状态的专家模型 (带时间衰减权重)
        logger.info("\n📊 Step 2: 训练专家模型 (使用时间衰减权重)")
        experts = []
        expert_train_preds = np.zeros((len(X_train), self.n_experts))
        expert_val_preds = np.zeros((len(X_val), self.n_experts))

        for i in range(self.n_experts):
            regime_mask = train_regimes == i
            if regime_mask.sum() < 100:
                logger.warning(f"  专家 {i}: 样本不足 ({regime_mask.sum()}), 使用全部数据")
                regime_mask = np.ones(len(X_train), dtype=bool)

            expert = ExpertModel(f'Expert_{i}', alpha=10.0)
            # 使用时间衰减权重训练
            expert.fit(X_train[regime_mask], y_train[regime_mask],
                      sample_weight=train_weights[regime_mask])
            experts.append(expert)

            # 所有样本的预测
            expert_train_preds[:, i] = expert.predict(X_train)
            expert_val_preds[:, i] = expert.predict(X_val)

            # 评估专家
            train_ic = spearmanr(expert_train_preds[:, i], y_train)[0]
            val_ic = spearmanr(expert_val_preds[:, i], y_val)[0]
            logger.info(f"  专家 {i}: 训练IC={train_ic:.4f}, 验证IC={val_ic:.4f}")

        # Step 3: 计算每个专家在每个样本上的表现（用于训练门控网络）
        logger.info("\n📊 Step 3: 训练门控网络")

        # 用排名相关性衡量每个专家的表现
        # 为每个样本选择IC最高的专家
        expert_performances = np.zeros((len(X_train), self.n_experts))

        # 使用滚动窗口计算每个专家的局部表现
        window_size = 500
        for i in range(0, len(X_train), window_size):
            end_idx = min(i + window_size, len(X_train))
            for j in range(self.n_experts):
                window_preds = expert_train_preds[i:end_idx, j]
                window_actual = y_train[i:end_idx]
                if len(window_preds) > 10:
                    ic = spearmanr(window_preds, window_actual)[0]
                    expert_performances[i:end_idx, j] = ic if not np.isnan(ic) else 0

        # 提取门控网络的输入特征（市场状态特征）
        gate_features = [c for c in MarketRegimeDetector.REGIME_FEATURES if c in feature_names]
        if len(gate_features) < 3:
            gate_features = feature_names[:6]

        X_gate_train = df_train[gate_features].values
        X_gate_val = df_val[gate_features].values
        X_gate_train = np.nan_to_num(X_gate_train, nan=0.0)
        X_gate_val = np.nan_to_num(X_gate_val, nan=0.0)

        gating = GatingNetwork(self.n_experts)
        gating.fit(X_gate_train, expert_performances)

        # Step 4: 混合预测
        logger.info("\n📊 Step 4: 混合专家预测")

        train_weights = gating.predict_weights(X_gate_train)
        val_weights = gating.predict_weights(X_gate_val)

        # 加权平均
        train_pred = np.sum(expert_train_preds * train_weights, axis=1)
        val_pred = np.sum(expert_val_preds * val_weights, axis=1)

        # 计算最终IC
        ic_train = spearmanr(train_pred, y_train)[0]
        ic_val = spearmanr(val_pred, y_val)[0]
        ic_gap = (ic_train - ic_val) / ic_train * 100 if ic_train > 0 else 0

        logger.info(f"\n📊 {period} 周期最终结果:")
        logger.info(f"  训练IC: {ic_train:.4f}")
        logger.info(f"  验证IC: {ic_val:.4f}")
        logger.info(f"  IC差距: {ic_gap:.1f}%")

        # 计算Top收益
        top_n = 100
        top_indices = np.argsort(val_pred)[-top_n:]
        top_return = y_val[top_indices].mean() * 100
        logger.info(f"  Top{top_n}收益: {top_return:.2f}%")

        # 计算方向准确率
        direction_correct = np.sum((val_pred > 0) == (y_val > 0)) / len(y_val)
        logger.info(f"  方向准确率: {direction_correct:.2%}")

        return {
            'experts': experts,
            'gating': gating,
            'regime_detector': self.regime_detector,
            'gate_features': gate_features,
            'train_ic': ic_train,
            'val_ic': ic_val,
            'ic_gap': ic_gap,
            'top_return': top_return,
            'direction_accuracy': direction_correct
        }

    def train_all_periods(self, start_date: str = '2023-01-01',
                          half_life_days: int = 180) -> Dict:
        """训练所有周期"""

        logger.info("=" * 80)
        logger.info("🚀 V3.91 市场自适应版本训练 (Mixture of Experts + 时间衰减)")
        logger.info("=" * 80)
        logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"数据起始日期: {start_date}")
        logger.info(f"专家数量: {self.n_experts}")
        logger.info(f"时间衰减半衰期: {half_life_days}天")
        logger.info("=" * 80)

        # 加载数据
        data = self.load_data(start_date)

        # 训练各周期
        period_results = {}
        for period in ['5d', '10d', '15d']:
            result = self.train_period_adaptive(period, data, half_life_days)
            period_results[period] = result

        # 综合评估
        composite_train_ic = sum(
            period_results[p]['train_ic'] * self.PERIOD_WEIGHTS[p]
            for p in ['5d', '10d', '15d']
        )
        composite_val_ic = sum(
            period_results[p]['val_ic'] * self.PERIOD_WEIGHTS[p]
            for p in ['5d', '10d', '15d']
        )
        composite_gap = (composite_train_ic - composite_val_ic) / composite_train_ic * 100

        logger.info("\n" + "=" * 80)
        logger.info("📊 综合评估")
        logger.info("=" * 80)
        logger.info(f"综合训练IC: {composite_train_ic:.4f}")
        logger.info(f"综合验证IC: {composite_val_ic:.4f}")
        logger.info(f"综合IC差距: {composite_gap:.1f}%")

        if composite_gap < 20:
            logger.info("✅ 过拟合已完全控制! (差距 < 20%)")
        elif composite_gap < 30:
            logger.info("✅ 过拟合控制良好! (差距 < 30%)")
        elif composite_gap < 40:
            logger.info("⚠️ 轻度过拟合 (差距 30-40%)")
        else:
            logger.info("❌ 过拟合较严重 (差距 > 40%)")

        # 计算评分
        score = self.calculate_score(composite_val_ic, composite_gap, period_results)
        logger.info(f"\n🎯 模型评分: {score:.1f}/100")

        return {
            'period_results': period_results,
            'composite_train_ic': composite_train_ic,
            'composite_val_ic': composite_val_ic,
            'composite_gap': composite_gap,
            'feature_names': data['feature_names'],
            'start_date': start_date,
            'n_experts': self.n_experts,
            'score': score
        }

    def calculate_score(self, val_ic: float, gap: float, period_results: Dict) -> float:
        """计算综合评分"""

        # IC评分 (0-40分)
        if val_ic >= 0.15:
            ic_score = 40
        elif val_ic >= 0.10:
            ic_score = 30 + (val_ic - 0.10) / 0.05 * 10
        elif val_ic >= 0.05:
            ic_score = 20 + (val_ic - 0.05) / 0.05 * 10
        elif val_ic >= 0:
            ic_score = val_ic / 0.05 * 20
        else:
            ic_score = 0

        # 过拟合控制评分 (0-30分)
        if gap <= 20:
            gap_score = 30
        elif gap <= 30:
            gap_score = 25
        elif gap <= 50:
            gap_score = 15
        else:
            gap_score = max(0, 15 - (gap - 50) / 10)

        # Top收益评分 (0-30分)
        avg_top_return = np.mean([r['top_return'] for r in period_results.values()])
        if avg_top_return >= 3:
            return_score = 30
        elif avg_top_return >= 2:
            return_score = 25
        elif avg_top_return >= 1:
            return_score = 20
        elif avg_top_return >= 0:
            return_score = 10
        else:
            return_score = 0

        return ic_score + gap_score + return_score

    def save_model(self, results: Dict, version: str = 'adaptive'):
        """保存模型"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_dir = 'ml_models/trained_models/v391'
        os.makedirs(model_dir, exist_ok=True)

        filename = f'v391_{version}_{timestamp}.pkl'
        filepath = os.path.join(model_dir, filename)

        save_data = {
            'version': f'v3.91-{version}',
            'timestamp': timestamp,
            'period_models': {},
            'period_weights': self.PERIOD_WEIGHTS,
            'feature_names': results['feature_names'],
            'start_date': results['start_date'],
            'n_experts': results['n_experts'],
            'metrics': {
                'composite_train_ic': results['composite_train_ic'],
                'composite_val_ic': results['composite_val_ic'],
                'composite_gap': results['composite_gap'],
                'score': results['score']
            },
            'architecture': 'Mixture of Experts'
        }

        for period, period_result in results['period_results'].items():
            save_data['period_models'][period] = {
                'experts': period_result['experts'],
                'gating': period_result['gating'],
                'regime_detector': period_result['regime_detector'],
                'gate_features': period_result['gate_features'],
                'train_ic': period_result['train_ic'],
                'val_ic': period_result['val_ic'],
                'top_return': period_result['top_return']
            }

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        logger.info(f"\n✅ 模型已保存: {filepath}")

        # 更新版本历史
        version_file = os.path.join(model_dir, 'VERSION_HISTORY.json')
        if os.path.exists(version_file):
            with open(version_file, 'r') as f:
                version_history = json.load(f)
        else:
            version_history = {'versions': []}

        version_history['updated'] = datetime.now().isoformat()
        version_history['versions'].insert(0, {
            'filename': filename,
            'version': f'v3.91-{version}',
            'timestamp': timestamp,
            'start_date': results['start_date'],
            'features': len(results['feature_names']),
            'n_experts': results['n_experts'],
            'composite_train_ic': round(results['composite_train_ic'], 4),
            'composite_val_ic': round(results['composite_val_ic'], 4),
            'composite_gap': f"{results['composite_gap']:.1f}%",
            'score': round(results['score'], 1),
            'architecture': 'Mixture of Experts',
            'description': f"{results['n_experts']}专家混合模型，自适应市场状态"
        })

        with open(version_file, 'w') as f:
            json.dump(version_history, f, indent=2)

        return filepath


def main():
    """主函数"""
    trainer = AdaptiveV391Trainer(n_experts=3)

    # 使用全量数据 (2022年开始) + 时间衰减权重
    # half_life_days=180: 6个月前的数据权重衰减到一半
    results = trainer.train_all_periods(
        start_date='2022-01-01',
        half_life_days=180  # 半衰期6个月
    )

    trainer.save_model(results, version='adaptive_moe_decay')

    logger.info("\n" + "=" * 80)
    logger.info("🎉 训练完成!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
