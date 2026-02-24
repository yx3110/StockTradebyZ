#!/usr/bin/env python3
"""
V3.95 滚动训练脚本
使用最近6个月数据训练，解决市场regime变化问题
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sqlite3
from scipy import stats

warnings.filterwarnings('ignore')

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


class V395RollingTrainer:
    """V3.95 滚动训练器"""

    def __init__(self,
                 train_months: int = 6,
                 val_months: int = 1,
                 test_months: int = 1):
        """
        初始化滚动训练器

        Args:
            train_months: 训练数据月数
            val_months: 验证数据月数
            test_months: 测试数据月数
        """
        self.train_months = train_months
        self.val_months = val_months
        self.test_months = test_months

        # 目标权重
        self.target_weights = {
            'label_3d': 0.4,
            'label_5d': 0.35,
            'label_10d': 0.25
        }

        # 模型目录
        self.model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v395'
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # 数据库路径
        self.db_path = PROJECT_ROOT / 'data_adapter' / 'stock_data.db'

        # 特征列表（从v39_feature_cache）
        self.feature_cols = None
        self.market_feature_cols = [
            'market_return_20d', 'market_return_10d', 'market_return_5d',
            'market_volatility_20d', 'market_volatility_10d',
            'market_up_ratio_20d', 'market_up_ratio_10d',
            'market_drawdown_20d', 'market_volume_ratio',
            'market_position_20d', 'market_momentum_20d', 'market_momentum_5d'
        ]

        # 模型配置
        self.models = {}
        self.scalers = {}
        self.ensemble_weights = {}

    def load_data(self, end_date: str = None) -> pd.DataFrame:
        """
        加载最近N个月的数据

        Args:
            end_date: 结束日期，默认为今天
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        # 计算起始日期
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        total_months = self.train_months + self.val_months + self.test_months
        start_dt = end_dt - timedelta(days=total_months * 30 + 30)  # 多加30天buffer
        start_date = start_dt.strftime('%Y-%m-%d')

        print(f"加载数据: {start_date} ~ {end_date}")

        conn = sqlite3.connect(self.db_path)

        # 构建查询 - features_json存储为JSON字符串
        query = """
        SELECT code, trade_date, features_json,
               label_3d, label_5d, label_10d,
               market_return_20d, market_return_10d, market_return_5d,
               market_volatility_20d, market_volatility_10d,
               market_up_ratio_20d, market_up_ratio_10d,
               market_drawdown_20d, market_volume_ratio,
               market_position_20d, market_momentum_20d, market_momentum_5d
        FROM v39_feature_cache
        WHERE trade_date >= ?
          AND trade_date <= ?
          AND label_3d IS NOT NULL
          AND label_5d IS NOT NULL
          AND label_10d IS NOT NULL
          AND market_return_20d IS NOT NULL
        ORDER BY trade_date, code
        """

        df = pd.read_sql_query(query, conn, params=[start_date, end_date])
        conn.close()

        print(f"原始数据量: {len(df)}")

        # 解析features_json列
        print("解析特征JSON...")
        features_list = []
        valid_indices = []

        for idx, row in df.iterrows():
            try:
                features = json.loads(row['features_json'])
                features_list.append(features)
                valid_indices.append(idx)
            except (json.JSONDecodeError, TypeError):
                continue

        # 创建特征DataFrame
        features_df = pd.DataFrame(features_list, index=valid_indices)

        # 确定特征列
        self.feature_cols = list(features_df.columns)
        print(f"特征列数: {len(self.feature_cols)}")
        print(f"市场特征列数: {len(self.market_feature_cols)}")

        # 合并特征到原始数据
        df = df.loc[valid_indices].reset_index(drop=True)
        features_df = features_df.reset_index(drop=True)

        # 添加特征列
        for col in self.feature_cols:
            df[col] = features_df[col]

        # 删除features_json列
        df = df.drop(columns=['features_json'])

        print(f"有效数据量: {len(df)}")
        print(f"日期范围: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
        print(f"股票数: {df['code'].nunique()}")

        return df

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """按时间分割数据"""
        dates = sorted(df['trade_date'].unique())
        n_dates = len(dates)

        # 计算分割点
        total_months = self.train_months + self.val_months + self.test_months
        train_ratio = self.train_months / total_months
        val_ratio = self.val_months / total_months

        train_end_idx = int(n_dates * train_ratio)
        val_end_idx = int(n_dates * (train_ratio + val_ratio))

        train_dates = dates[:train_end_idx]
        val_dates = dates[train_end_idx:val_end_idx]
        test_dates = dates[val_end_idx:]

        train_df = df[df['trade_date'].isin(train_dates)]
        val_df = df[df['trade_date'].isin(val_dates)]
        test_df = df[df['trade_date'].isin(test_dates)]

        print(f"\n数据分割:")
        print(f"  训练集: {len(train_df)} 样本, {len(train_dates)} 天")
        print(f"    日期: {train_dates[0]} ~ {train_dates[-1]}")
        print(f"  验证集: {len(val_df)} 样本, {len(val_dates)} 天")
        print(f"    日期: {val_dates[0]} ~ {val_dates[-1]}")
        print(f"  测试集: {len(test_df)} 样本, {len(test_dates)} 天")
        print(f"    日期: {test_dates[0]} ~ {test_dates[-1]}")

        return train_df, val_df, test_df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """准备特征和标签"""
        # 特征
        all_feat_cols = self.feature_cols + self.market_feature_cols
        X = df[all_feat_cols].fillna(0).values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # 标签
        y = {
            'label_3d': df['label_3d'].values,
            'label_5d': df['label_5d'].values,
            'label_10d': df['label_10d'].values
        }

        return X, y

    def create_models(self) -> Dict[str, object]:
        """创建基础模型"""
        return {
            'lgb': LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1,
                force_col_wise=True
            ),
            'xgb': XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbosity=0
            ),
            'cb': CatBoostRegressor(
                iterations=200,
                depth=6,
                learning_rate=0.05,
                l2_leaf_reg=3,
                random_seed=42,
                verbose=False
            ),
            'rf': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'gb': HistGradientBoostingRegressor(
                max_iter=200,
                max_depth=6,
                learning_rate=0.05,
                min_samples_leaf=20,
                l2_regularization=0.1,
                random_state=42
            )
        }

    def train_target(self, target_name: str,
                     X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Dict, Dict, np.ndarray]:
        """训练单个目标的所有模型"""
        print(f"\n训练目标: {target_name}")

        models = self.create_models()
        predictions = {}

        for name, model in models.items():
            print(f"  训练 {name}...", end=' ')
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            predictions[name] = pred

            # 计算验证集指标
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            ic = stats.spearmanr(pred, y_val)[0]
            print(f"RMSE={rmse:.4f}, IC={ic:.4f}")

        # 计算集成权重（基于验证集IC）
        ics = {}
        for name, pred in predictions.items():
            ic = stats.spearmanr(pred, y_val)[0]
            ics[name] = max(ic, 0.01)  # 避免负权重

        total_ic = sum(ics.values())
        weights = {name: ic / total_ic for name, ic in ics.items()}

        print(f"  集成权重: {weights}")

        # 计算集成预测
        ensemble_pred = np.zeros(len(y_val))
        for name, pred in predictions.items():
            ensemble_pred += weights[name] * pred

        return models, weights, ensemble_pred

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                 df: pd.DataFrame = None) -> Dict:
        """评估预测结果"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        ic = stats.spearmanr(y_pred, y_true)[0]

        # 方向准确率
        direction_acc = np.mean((y_pred > 0) == (y_true > 0))

        # Top10收益
        top10_return = 0
        top20_return = 0

        if df is not None and 'trade_date' in df.columns:
            df_eval = df.copy()
            df_eval['pred'] = y_pred
            df_eval['actual'] = y_true

            top10_returns = []
            top20_returns = []

            for date in df_eval['trade_date'].unique():
                day_df = df_eval[df_eval['trade_date'] == date]
                if len(day_df) >= 20:
                    sorted_df = day_df.sort_values('pred', ascending=False)
                    top10 = sorted_df.head(10)['actual'].mean()
                    top20 = sorted_df.head(20)['actual'].mean()
                    top10_returns.append(top10)
                    top20_returns.append(top20)

            if top10_returns:
                top10_return = np.mean(top10_returns)
                top20_return = np.mean(top20_returns)

        return {
            'rmse': rmse,
            'ic': ic,
            'direction_accuracy': direction_acc,
            'top10_return': top10_return,
            'top20_return': top20_return
        }

    def train(self, end_date: str = None):
        """执行滚动训练"""
        print("=" * 60)
        print("V3.95 滚动训练")
        print("=" * 60)

        start_time = datetime.now()

        # 加载数据
        df = self.load_data(end_date)

        if len(df) < 1000:
            print("数据量不足，需要至少1000条记录")
            return None

        # 分割数据
        train_df, val_df, test_df = self.split_data(df)

        # 准备特征
        X_train, y_train = self.prepare_features(train_df)
        X_val, y_val = self.prepare_features(val_df)
        X_test, y_test = self.prepare_features(test_df)

        # 标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        self.scalers['main'] = scaler

        # 训练每个目标
        all_models = {}
        all_weights = {}
        val_predictions = {}
        test_predictions = {}

        for target in ['label_3d', 'label_5d', 'label_10d']:
            models, weights, val_pred = self.train_target(
                target,
                X_train, y_train[target],
                X_val, y_val[target]
            )

            all_models[target] = models
            all_weights[target] = weights
            val_predictions[target] = val_pred

            # 测试集预测
            test_pred = np.zeros(len(X_test))
            for name, model in models.items():
                test_pred += weights[name] * model.predict(X_test)
            test_predictions[target] = test_pred

        self.models = all_models
        self.ensemble_weights = all_weights

        # 评估
        print("\n" + "=" * 60)
        print("验证集评估:")
        print("=" * 60)

        val_metrics = {}
        for target in ['label_3d', 'label_5d', 'label_10d']:
            metrics = self.evaluate(y_val[target], val_predictions[target], val_df)
            val_metrics[target] = metrics
            print(f"\n{target}:")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  IC: {metrics['ic']:.4f}")
            print(f"  方向准确率: {metrics['direction_accuracy']:.2%}")
            print(f"  Top10收益: {metrics['top10_return']:.4f}")
            print(f"  Top20收益: {metrics['top20_return']:.4f}")

        print("\n" + "=" * 60)
        print("测试集评估:")
        print("=" * 60)

        test_metrics = {}
        for target in ['label_3d', 'label_5d', 'label_10d']:
            metrics = self.evaluate(y_test[target], test_predictions[target], test_df)
            test_metrics[target] = metrics
            print(f"\n{target}:")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  IC: {metrics['ic']:.4f}")
            print(f"  方向准确率: {metrics['direction_accuracy']:.2%}")
            print(f"  Top10收益: {metrics['top10_return']:.4f}")
            print(f"  Top20收益: {metrics['top20_return']:.4f}")

        # 保存模型
        self.save_models()

        # 保存训练历史
        end_time = datetime.now()
        history = {
            'version': 'v3.95-rolling',
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': (end_time - start_time).total_seconds(),
            'status': 'completed',
            'config': {
                'train_months': self.train_months,
                'val_months': self.val_months,
                'test_months': self.test_months
            },
            'data_info': {
                'train_samples': len(train_df),
                'val_samples': len(val_df),
                'test_samples': len(test_df),
                'train_dates': f"{train_df['trade_date'].min()} ~ {train_df['trade_date'].max()}",
                'val_dates': f"{val_df['trade_date'].min()} ~ {val_df['trade_date'].max()}",
                'test_dates': f"{test_df['trade_date'].min()} ~ {test_df['trade_date'].max()}"
            },
            'feature_count': len(self.feature_cols) + len(self.market_feature_cols),
            'val_metrics': {k: {kk: float(vv) for kk, vv in v.items()}
                          for k, v in val_metrics.items()},
            'test_metrics': {k: {kk: float(vv) for kk, vv in v.items()}
                           for k, v in test_metrics.items()},
            'target_weights': self.target_weights,
            'ensemble_weights': {k: {kk: float(vv) for kk, vv in v.items()}
                                for k, v in all_weights.items()}
        }

        history_path = self.model_dir / 'training_history_rolling.json'
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        print(f"\n训练历史已保存: {history_path}")

        return test_metrics

    def save_models(self):
        """保存模型"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存每个目标的模型
        for target in ['label_3d', 'label_5d', 'label_10d']:
            target_suffix = target.replace('label_', '')

            for model_name, model in self.models[target].items():
                path = self.model_dir / f'v395_rolling_{target_suffix}_{model_name}.pkl'
                with open(path, 'wb') as f:
                    pickle.dump(model, f)

        # 保存scaler
        scaler_path = self.model_dir / 'v395_rolling_scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scalers, f)

        # 保存权重
        weights_path = self.model_dir / 'v395_rolling_weights.json'
        with open(weights_path, 'w') as f:
            json.dump({
                'ensemble_weights': {k: {kk: float(vv) for kk, vv in v.items()}
                                    for k, v in self.ensemble_weights.items()},
                'target_weights': self.target_weights,
                'feature_cols': self.feature_cols,
                'market_feature_cols': self.market_feature_cols
            }, f, indent=2)

        print(f"\n模型已保存到: {self.model_dir}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='V3.95 滚动训练')
    parser.add_argument('--train-months', type=int, default=6, help='训练数据月数')
    parser.add_argument('--val-months', type=int, default=1, help='验证数据月数')
    parser.add_argument('--test-months', type=int, default=1, help='测试数据月数')
    parser.add_argument('--end-date', type=str, default=None, help='结束日期 (YYYY-MM-DD)')

    args = parser.parse_args()

    trainer = V395RollingTrainer(
        train_months=args.train_months,
        val_months=args.val_months,
        test_months=args.test_months
    )

    trainer.train(end_date=args.end_date)


if __name__ == '__main__':
    main()
