#!/usr/bin/env python3
"""
V3.91 极简抗过拟合版本

核心思路：
1. 极少特征 (10-15个)
2. 使用线性模型 (Ridge回归) 作为主力
3. 极强正则化
4. 噪声注入
5. 只使用最近的数据 (2024年以后)

目标：IC差距 < 30%
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import RobustScaler, StandardScaler
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
        logging.FileHandler(f'{log_dir}/v391_ultra_simple_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class UltraSimpleV391Trainer:
    """极简抗过拟合V3.91训练器"""

    PERIOD_WEIGHTS = {'5d': 0.40, '10d': 0.35, '15d': 0.25}

    # 核心特征 (只选择最稳定、最有预测力的)
    CORE_FEATURES = [
        'supertrend_signal',      # 趋势信号
        'aroon_down_25',          # 趋势强度
        'williams_r_14',          # 超买超卖
        'vwap_deviation',         # 价格偏离
        'cmf_20',                 # 资金流向
        'ichimoku_base',          # 支撑阻力
        'ad_line_change_5',       # 成交量趋势
        'bb_width_20',            # 波动率
        'market_attention_score', # 市场关注度
        'advance_decline_ratio',  # 市场广度
    ]

    def __init__(self, db_path: str = 'data_adapter/stock_data.db'):
        self.db_path = db_path
        self.scaler = RobustScaler()

    def load_recent_data(self, start_date: str = '2024-01-01') -> Dict:
        """只加载最近的数据，减少分布漂移"""
        import sqlite3

        logger.info("=" * 80)
        logger.info(f"📥 加载最近数据 (从 {start_date} 开始)")
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

        # 只选择核心特征
        available_core = [f for f in self.CORE_FEATURES if f in feature_df.columns]
        logger.info(f"📊 使用 {len(available_core)} 个核心特征")

        for col in available_core:
            df_valid[col] = feature_df[col].values

        # 按时间排序
        df_valid = df_valid.sort_values('trade_date').reset_index(drop=True)

        # 时序分割 (80% 训练, 20% 验证)
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
            'feature_names': available_core,
            'train_mask': train_mask,
            'val_mask': val_mask,
            'split_date': split_date
        }

    def add_noise(self, X: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """添加高斯噪声增强鲁棒性"""
        noise = np.random.randn(*X.shape) * noise_level * np.std(X, axis=0, keepdims=True)
        return X + noise

    def rank_transform(self, y: np.ndarray) -> np.ndarray:
        """排名转换，降低异常值影响"""
        return (rankdata(y) - 1) / (len(y) - 1) - 0.5

    def train_period_simple(self, period: str, data: Dict,
                            noise_level: float = 0.05,
                            alpha: float = 10.0) -> Dict:
        """使用极简模型训练单个周期"""

        logger.info(f"\n{'='*60}")
        logger.info(f"🔹 训练 {period} 周期 (线性模型 + 强正则化)")
        logger.info(f"{'='*60}")

        df = data['df']
        feature_names = data['feature_names']
        train_mask = data['train_mask']
        val_mask = data['val_mask']

        label_col = f'future_return_{period}'

        # 提取特征和标签
        X_all = df[feature_names].values
        y_all = df[label_col].values

        # 处理缺失值
        X_all = np.nan_to_num(X_all, nan=0.0)

        # 分割
        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        X_val = X_all[val_mask]
        y_val = y_all[val_mask]

        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # 标签排名转换
        y_train_rank = self.rank_transform(y_train)

        # 添加训练噪声
        X_train_noisy = self.add_noise(X_train_scaled, noise_level)

        # 多次训练取平均 (Bootstrap aggregating 简化版)
        n_models = 5
        val_predictions = np.zeros(len(X_val))
        train_predictions = np.zeros(len(X_train))

        alphas = [alpha * (1 + i * 0.5) for i in range(n_models)]  # 不同正则化强度

        for i, a in enumerate(alphas):
            # Ridge回归 (极强正则化)
            model = Ridge(alpha=a, random_state=42 + i)
            model.fit(X_train_noisy, y_train_rank)

            val_predictions += model.predict(X_val_scaled) / n_models
            train_predictions += model.predict(X_train_scaled) / n_models

        # 计算IC
        ic_train = spearmanr(train_predictions, y_train)[0]
        ic_val = spearmanr(val_predictions, y_val)[0]
        ic_gap = (ic_train - ic_val) / ic_train * 100 if ic_train > 0 else 0

        logger.info(f"  训练IC: {ic_train:.4f}")
        logger.info(f"  验证IC: {ic_val:.4f}")
        logger.info(f"  IC差距: {ic_gap:.1f}%")

        # 计算Top收益
        top_n = 100
        top_indices = np.argsort(val_predictions)[-top_n:]
        top_return = y_val[top_indices].mean() * 100

        logger.info(f"  Top{top_n}收益: {top_return:.2f}%")

        # 训练最终模型 (用于生产)
        final_model = Ridge(alpha=alpha, random_state=42)
        final_model.fit(X_train_scaled, y_train_rank)

        return {
            'model': final_model,
            'scaler': self.scaler,
            'feature_names': feature_names,
            'train_ic': ic_train,
            'val_ic': ic_val,
            'ic_gap': ic_gap,
            'top_return': top_return
        }

    def train_all_periods(self, start_date: str = '2024-01-01',
                          noise_level: float = 0.05,
                          alpha: float = 10.0) -> Dict:
        """训练所有周期"""

        logger.info("=" * 80)
        logger.info("🚀 V3.91 极简抗过拟合版本训练")
        logger.info("=" * 80)
        logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"数据起始日期: {start_date}")
        logger.info(f"噪声水平: {noise_level}")
        logger.info(f"正则化强度 (alpha): {alpha}")
        logger.info(f"核心特征: {len(self.CORE_FEATURES)}个")
        logger.info("=" * 80)

        # 加载数据
        data = self.load_recent_data(start_date)

        # 训练各周期
        period_results = {}
        for period in ['5d', '10d', '15d']:
            result = self.train_period_simple(
                period, data,
                noise_level=noise_level,
                alpha=alpha
            )
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

        return {
            'period_results': period_results,
            'composite_train_ic': composite_train_ic,
            'composite_val_ic': composite_val_ic,
            'composite_gap': composite_gap,
            'feature_names': data['feature_names'],
            'start_date': start_date,
            'params': {
                'noise_level': noise_level,
                'alpha': alpha
            }
        }

    def save_model(self, results: Dict, version: str = 'ultra_simple'):
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
            'metrics': {
                'composite_train_ic': results['composite_train_ic'],
                'composite_val_ic': results['composite_val_ic'],
                'composite_gap': results['composite_gap']
            },
            'params': results['params']
        }

        for period, period_result in results['period_results'].items():
            save_data['period_models'][period] = {
                'model': period_result['model'],
                'scaler': period_result['scaler'],
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
            'composite_train_ic': results['composite_train_ic'],
            'composite_val_ic': results['composite_val_ic'],
            'composite_gap': f"{results['composite_gap']:.1f}%",
            'model_type': 'Ridge Linear (极简)'
        })

        with open(version_file, 'w') as f:
            json.dump(version_history, f, indent=2)

        return filepath


def main():
    """主函数"""
    trainer = UltraSimpleV391Trainer()

    # 使用2024年以后的数据，减少分布漂移
    results = trainer.train_all_periods(
        start_date='2024-01-01',
        noise_level=0.05,
        alpha=10.0
    )

    trainer.save_model(results, version='ultra_simple')

    logger.info("\n" + "=" * 80)
    logger.info("🎉 训练完成!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
