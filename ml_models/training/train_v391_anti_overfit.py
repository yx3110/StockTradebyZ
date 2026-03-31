#!/usr/bin/env python3
"""
V3.91 抗过拟合版本训练脚本

核心改进策略：
1. 更强的正则化参数
2. 特征选择和降噪
3. Walk-Forward交叉验证
4. 标签平滑和排名目标
5. 随机特征Dropout
6. 简化模型架构
7. 时间衰减权重

目标：让训练集和验证集IC差距控制在20%以内
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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
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
        logging.FileHandler(f'{log_dir}/v391_anti_overfit_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AntiOverfitV391Trainer:
    """抗过拟合V3.91训练器"""

    # 周期权重
    PERIOD_WEIGHTS = {'5d': 0.40, '10d': 0.35, '15d': 0.25}

    # 抗过拟合的正则化参数
    REGULARIZED_PARAMS = {
        'lgb': {
            'objective': 'regression',
            'metric': 'mse',
            'boosting_type': 'gbdt',
            'num_leaves': 15,          # 减少 (原31)
            'max_depth': 4,            # 减少 (原6)
            'learning_rate': 0.01,     # 减少 (原0.05)
            'n_estimators': 500,       # 增加迭代次数
            'min_child_samples': 100,  # 增加 (原20)
            'subsample': 0.6,          # 减少 (原0.8)
            'colsample_bytree': 0.6,   # 减少 (原0.8)
            'reg_alpha': 0.5,          # 增加L1正则化 (原0.1)
            'reg_lambda': 1.0,         # 增加L2正则化 (原0.1)
            'min_split_gain': 0.01,    # 增加最小分裂增益
            'verbose': -1,
            'random_state': 42,
            'n_jobs': -1
        },
        'xgb': {
            'objective': 'reg:squarederror',
            'max_depth': 3,            # 减少 (原5)
            'learning_rate': 0.01,     # 减少 (原0.05)
            'n_estimators': 500,
            'min_child_weight': 50,    # 增加 (原10)
            'subsample': 0.6,
            'colsample_bytree': 0.6,
            'reg_alpha': 0.5,
            'reg_lambda': 1.0,
            'gamma': 0.1,              # 增加
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        },
        'cat': {
            'iterations': 300,
            'depth': 4,                # 减少 (原6)
            'learning_rate': 0.02,     # 减少 (原0.05)
            'l2_leaf_reg': 10,         # 增加 (原3)
            'min_data_in_leaf': 100,   # 增加 (原20)
            'random_seed': 42,
            'verbose': False,
            'thread_count': -1
        },
        'rf': {
            'n_estimators': 100,       # 减少 (原200)
            'max_depth': 6,            # 减少 (原10)
            'min_samples_split': 50,   # 增加 (原10)
            'min_samples_leaf': 25,    # 增加 (原5)
            'max_features': 0.5,       # 减少 (原auto)
            'random_state': 42,
            'n_jobs': -1,
            'oob_score': True
        }
    }

    def __init__(self, db_path: str = 'data_adapter/stock_data.db'):
        """初始化训练器"""
        self.db_path = db_path
        self.feature_names = None
        self.feature_selector = None
        self.scaler = RobustScaler()  # 使用RobustScaler处理异常值

    def load_data_with_walk_forward(self,
                                     start_date: str = '2023-01-01',
                                     n_splits: int = 5) -> Dict:
        """
        使用Walk-Forward方式加载和分割数据

        Walk-Forward验证：
        Split 1: Train [0-60%], Val [60-70%]
        Split 2: Train [0-70%], Val [70-80%]
        Split 3: Train [0-80%], Val [80-90%]
        Split 4: Train [0-90%], Val [90-100%]
        """
        import sqlite3

        logger.info("=" * 80)
        logger.info("📥 加载训练数据 (Walk-Forward模式)")
        logger.info("=" * 80)

        conn = sqlite3.connect(self.db_path)

        # 加载v3.9特征缓存
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
        logger.info("📊 解析特征...")
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

            if len(features_list) % 50000 == 0 and len(features_list) > 0:
                logger.info(f"  已处理: {len(features_list):,}/{len(df):,}")

        df_valid = df.loc[valid_indices].copy()
        feature_df = pd.DataFrame(features_list)

        # 合并特征
        for col in feature_df.columns:
            df_valid[col] = feature_df[col].values

        self.feature_names = list(feature_df.columns)
        logger.info(f"✅ 特征矩阵: ({len(df_valid)}, {len(self.feature_names)})")

        # 按时间排序
        df_valid = df_valid.sort_values('trade_date').reset_index(drop=True)
        unique_dates = df_valid['trade_date'].unique()
        logger.info(f"📅 时间范围: {unique_dates[0]} ~ {unique_dates[-1]}")

        # 创建Walk-Forward splits
        n_dates = len(unique_dates)
        splits = []

        # 使用扩展窗口 (Expanding Window) 而非滑动窗口
        for i in range(n_splits):
            train_end_idx = int(n_dates * (0.6 + i * 0.1))
            val_end_idx = min(int(n_dates * (0.7 + i * 0.1)), n_dates)

            train_end_date = unique_dates[train_end_idx]
            val_start_date = unique_dates[train_end_idx]
            val_end_date = unique_dates[val_end_idx - 1]

            train_mask = df_valid['trade_date'] < train_end_date
            val_mask = (df_valid['trade_date'] >= val_start_date) & \
                       (df_valid['trade_date'] <= val_end_date)

            splits.append({
                'train_mask': train_mask,
                'val_mask': val_mask,
                'train_end_date': train_end_date,
                'val_dates': f"{val_start_date} ~ {val_end_date}"
            })

            logger.info(f"  Split {i+1}: 训练 < {train_end_date}, 验证 {val_start_date} ~ {val_end_date}")

        return {
            'df': df_valid,
            'feature_names': self.feature_names,
            'splits': splits
        }

    def select_stable_features(self, X: np.ndarray, y: np.ndarray,
                               feature_names: List[str],
                               top_k: int = 30) -> List[int]:
        """
        特征选择：选择与目标相关性高且稳定的特征

        稳定性标准：
        1. 与目标的Spearman相关性
        2. 低方差过滤
        3. 低相关性过滤（避免冗余）
        """
        logger.info(f"🔍 特征选择: 从 {len(feature_names)} 个特征中选择 {top_k} 个")

        correlations = []
        for i in range(X.shape[1]):
            # 计算与目标的Spearman相关性
            valid_mask = ~np.isnan(X[:, i]) & ~np.isnan(y)
            if valid_mask.sum() > 100:
                corr, _ = spearmanr(X[valid_mask, i], y[valid_mask])
                correlations.append(abs(corr) if not np.isnan(corr) else 0)
            else:
                correlations.append(0)

        # 按相关性排序选择top_k
        sorted_indices = np.argsort(correlations)[::-1]
        selected = []

        for idx in sorted_indices:
            if len(selected) >= top_k:
                break

            # 检查与已选特征的相关性，避免冗余
            is_redundant = False
            for sel_idx in selected:
                valid_mask = ~np.isnan(X[:, idx]) & ~np.isnan(X[:, sel_idx])
                if valid_mask.sum() > 100:
                    corr, _ = spearmanr(X[valid_mask, idx], X[valid_mask, sel_idx])
                    if abs(corr) > 0.8:  # 高相关性阈值
                        is_redundant = True
                        break

            if not is_redundant:
                selected.append(idx)
                logger.info(f"  选择特征 {feature_names[idx]}: corr={correlations[idx]:.4f}")

        logger.info(f"✅ 选择了 {len(selected)} 个特征")
        return selected

    def smooth_labels(self, y: np.ndarray,
                      clip_percentile: float = 1.0,
                      use_rank: bool = True) -> np.ndarray:
        """
        标签平滑处理

        1. 截断极端值
        2. 可选：转换为排名（更稳健）
        """
        # 截断极端值
        lower = np.percentile(y, clip_percentile)
        upper = np.percentile(y, 100 - clip_percentile)
        y_clipped = np.clip(y, lower, upper)

        if use_rank:
            # 转换为排名百分比 (0-1)
            y_rank = rankdata(y_clipped) / len(y_clipped)
            # 中心化到 -0.5 ~ 0.5
            return y_rank - 0.5
        else:
            return y_clipped

    def add_time_decay_weights(self, dates: np.ndarray,
                                half_life_days: int = 180) -> np.ndarray:
        """
        添加时间衰减权重，最近的数据权重更高
        """
        # 转换为距最新日期的天数
        dates_dt = pd.to_datetime(dates)
        max_date = dates_dt.max()
        days_ago = np.array([(max_date - d).days for d in dates_dt])

        # 指数衰减
        weights = np.exp(-np.log(2) * days_ago / half_life_days)

        # 归一化
        weights = weights / weights.mean()

        return weights

    def train_single_model_with_dropout(self,
                                        model_type: str,
                                        X_train: np.ndarray,
                                        y_train: np.ndarray,
                                        X_val: np.ndarray,
                                        y_val: np.ndarray,
                                        sample_weights: np.ndarray = None,
                                        feature_dropout: float = 0.2) -> Tuple:
        """
        训练单个模型，带特征Dropout

        特征Dropout：随机屏蔽一部分特征，增强泛化能力
        """
        import lightgbm as lgb
        import xgboost as xgb
        from catboost import CatBoostRegressor
        from sklearn.ensemble import RandomForestRegressor

        # 随机特征dropout
        n_features = X_train.shape[1]
        if feature_dropout > 0:
            n_drop = int(n_features * feature_dropout)
            keep_indices = np.random.choice(n_features, n_features - n_drop, replace=False)
            keep_indices = np.sort(keep_indices)
            X_train_dropped = X_train[:, keep_indices]
            X_val_dropped = X_val[:, keep_indices]
        else:
            X_train_dropped = X_train
            X_val_dropped = X_val
            keep_indices = np.arange(n_features)

        params = self.REGULARIZED_PARAMS[model_type].copy()

        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train_dropped, y_train,
                sample_weight=sample_weights,
                eval_set=[(X_val_dropped, y_val)],
                callbacks=[lgb.early_stopping(100, verbose=False)]
            )
            best_iter = model.best_iteration_

        elif model_type == 'xgb':
            model = xgb.XGBRegressor(**params, early_stopping_rounds=100)
            model.fit(
                X_train_dropped, y_train,
                sample_weight=sample_weights,
                eval_set=[(X_val_dropped, y_val)],
                verbose=False
            )
            best_iter = model.best_iteration if hasattr(model, 'best_iteration') and model.best_iteration else params['n_estimators']

        elif model_type == 'cat':
            model = CatBoostRegressor(**params)
            model.fit(
                X_train_dropped, y_train,
                sample_weight=sample_weights,
                eval_set=(X_val_dropped, y_val),
                early_stopping_rounds=100,
                verbose=False
            )
            best_iter = model.get_best_iteration()

        elif model_type == 'rf':
            model = RandomForestRegressor(**params)
            model.fit(X_train_dropped, y_train)
            best_iter = params['n_estimators']

        # 评估
        pred_train = model.predict(X_train_dropped)
        pred_val = model.predict(X_val_dropped)

        ic_train = spearmanr(pred_train, y_train)[0]
        ic_val = spearmanr(pred_val, y_val)[0]

        return model, keep_indices, ic_train, ic_val, best_iter

    def train_period_with_walk_forward(self,
                                       period: str,
                                       data: Dict,
                                       use_label_smoothing: bool = True,
                                       use_feature_selection: bool = True,
                                       feature_dropout: float = 0.1) -> Dict:
        """
        使用Walk-Forward验证训练单个周期的模型
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🔹 训练 {period} 周期模型 (Walk-Forward)")
        logger.info(f"{'='*80}")

        df = data['df']
        feature_names = data['feature_names']
        splits = data['splits']

        label_col = f'future_return_{period}'

        # 准备完整数据
        X_all = df[feature_names].values
        y_all = df[label_col].values
        dates_all = df['trade_date'].values

        # 处理缺失值
        X_all = np.nan_to_num(X_all, nan=0.0)

        # 特征选择（使用第一个split的训练数据）
        if use_feature_selection:
            first_train_mask = splits[0]['train_mask']
            X_first_train = X_all[first_train_mask]
            y_first_train = y_all[first_train_mask]
            selected_features = self.select_stable_features(
                X_first_train, y_first_train, feature_names, top_k=25
            )
            self.feature_selector = selected_features
            X_all = X_all[:, selected_features]
            feature_names_selected = [feature_names[i] for i in selected_features]
        else:
            feature_names_selected = feature_names

        # Walk-Forward验证
        all_val_predictions = []
        all_val_actuals = []
        all_train_ics = []
        all_val_ics = []

        models_per_split = []

        for split_idx, split in enumerate(splits):
            logger.info(f"\n--- Split {split_idx + 1}/{len(splits)} ---")

            train_mask = split['train_mask']
            val_mask = split['val_mask']

            X_train = X_all[train_mask]
            y_train = y_all[train_mask]
            dates_train = dates_all[train_mask]

            X_val = X_all[val_mask]
            y_val = y_all[val_mask]

            # 标签平滑
            if use_label_smoothing:
                y_train_smooth = self.smooth_labels(y_train, clip_percentile=2.0, use_rank=True)
            else:
                y_train_smooth = y_train

            # 时间衰减权重
            sample_weights = self.add_time_decay_weights(dates_train, half_life_days=180)

            # 训练多个模型
            split_models = {}
            split_predictions = np.zeros(len(X_val))
            model_weights = {'lgb': 0.35, 'xgb': 0.35, 'rf': 0.30}  # 简化：去掉CatBoost

            for model_type in ['lgb', 'xgb', 'rf']:
                model, keep_idx, ic_train, ic_val, best_iter = \
                    self.train_single_model_with_dropout(
                        model_type, X_train, y_train_smooth, X_val, y_val,
                        sample_weights=sample_weights,
                        feature_dropout=feature_dropout
                    )

                # 验证集预测
                pred_val = model.predict(X_val[:, keep_idx] if feature_dropout > 0 else X_val)
                split_predictions += pred_val * model_weights[model_type]

                split_models[model_type] = {
                    'model': model,
                    'keep_indices': keep_idx,
                    'ic_train': ic_train,
                    'ic_val': ic_val
                }

                logger.info(f"    {model_type}: IC_train={ic_train:.4f}, IC_val={ic_val:.4f}, "
                           f"轮数={best_iter}")

            # 记录验证结果
            final_ic_val = spearmanr(split_predictions, y_val)[0]
            logger.info(f"  Split {split_idx + 1} 综合 IC_val: {final_ic_val:.4f}")

            all_val_predictions.extend(split_predictions)
            all_val_actuals.extend(y_val)
            all_val_ics.append(final_ic_val)

            # 计算训练集IC
            train_preds = np.zeros(len(X_train))
            for model_type, model_info in split_models.items():
                if feature_dropout > 0:
                    train_preds += model_info['model'].predict(
                        X_train[:, model_info['keep_indices']]
                    ) * model_weights[model_type]
                else:
                    train_preds += model_info['model'].predict(X_train) * model_weights[model_type]
            train_ic = spearmanr(train_preds, y_train)[0]
            all_train_ics.append(train_ic)

            models_per_split.append(split_models)

        # 汇总结果
        mean_train_ic = np.mean(all_train_ics)
        mean_val_ic = np.mean(all_val_ics)
        ic_gap = (mean_train_ic - mean_val_ic) / mean_train_ic * 100

        logger.info(f"\n📊 {period} 周期汇总:")
        logger.info(f"   平均训练IC: {mean_train_ic:.4f}")
        logger.info(f"   平均验证IC: {mean_val_ic:.4f}")
        logger.info(f"   IC差距: {ic_gap:.1f}%")

        # 使用最后一个split的模型作为最终模型
        final_models = models_per_split[-1]

        return {
            'models': final_models,
            'feature_selector': self.feature_selector if use_feature_selection else None,
            'feature_names': feature_names_selected,
            'train_ic': mean_train_ic,
            'val_ic': mean_val_ic,
            'ic_gap': ic_gap,
            'all_train_ics': all_train_ics,
            'all_val_ics': all_val_ics
        }

    def train_all_periods(self,
                          start_date: str = '2023-01-01',
                          use_label_smoothing: bool = True,
                          use_feature_selection: bool = True,
                          feature_dropout: float = 0.1) -> Dict:
        """训练所有周期的模型"""

        logger.info("=" * 80)
        logger.info("🚀 V3.91 抗过拟合版本训练")
        logger.info("=" * 80)
        logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"训练数据起始日期: {start_date}")
        logger.info(f"标签平滑: {use_label_smoothing}")
        logger.info(f"特征选择: {use_feature_selection}")
        logger.info(f"特征Dropout: {feature_dropout}")
        logger.info("=" * 80)

        # 加载数据
        data = self.load_data_with_walk_forward(start_date, n_splits=4)

        # 训练各周期
        period_results = {}
        for period in ['5d', '10d', '15d']:
            result = self.train_period_with_walk_forward(
                period, data,
                use_label_smoothing=use_label_smoothing,
                use_feature_selection=use_feature_selection,
                feature_dropout=feature_dropout
            )
            period_results[period] = result

        # 计算综合指标
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

        # 判断是否过拟合
        if composite_gap < 20:
            logger.info("✅ 过拟合已控制! (差距 < 20%)")
        elif composite_gap < 40:
            logger.info("⚠️ 轻度过拟合 (差距 20-40%)")
        else:
            logger.info("❌ 严重过拟合 (差距 > 40%)")

        return {
            'period_results': period_results,
            'composite_train_ic': composite_train_ic,
            'composite_val_ic': composite_val_ic,
            'composite_gap': composite_gap,
            'feature_names': data['feature_names'],
            'start_date': start_date,
            'training_params': {
                'use_label_smoothing': use_label_smoothing,
                'use_feature_selection': use_feature_selection,
                'feature_dropout': feature_dropout
            }
        }

    def save_model(self, results: Dict, version: str = 'anti_overfit'):
        """保存模型"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_dir = 'ml_models/trained_models/v391'
        os.makedirs(model_dir, exist_ok=True)

        filename = f'v391_{version}_{timestamp}.pkl'
        filepath = os.path.join(model_dir, filename)

        # 构建保存数据
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
            'training_params': results['training_params']
        }

        # 保存各周期模型
        for period, period_result in results['period_results'].items():
            save_data['period_models'][period] = {
                'models': period_result['models'],
                'feature_selector': period_result['feature_selector'],
                'train_ic': period_result['train_ic'],
                'val_ic': period_result['val_ic']
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
            'composite_gap': f"{results['composite_gap']:.1f}%"
        })

        with open(version_file, 'w') as f:
            json.dump(version_history, f, indent=2)

        return filepath


def main():
    """主函数"""
    trainer = AntiOverfitV391Trainer()

    # 训练模型
    results = trainer.train_all_periods(
        start_date='2023-01-01',
        use_label_smoothing=True,
        use_feature_selection=True,
        feature_dropout=0.1
    )

    # 保存模型
    trainer.save_model(results, version='anti_overfit')

    logger.info("\n" + "=" * 80)
    logger.info("🎉 训练完成!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
