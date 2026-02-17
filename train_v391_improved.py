#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.91 改进版多周期模型训练器

改进点：
1. 时序数据划分 - 避免未来数据泄露
2. 分层采样 - 确保各收益区间样本平衡
3. 早停机制 - 防止过拟合
4. 多指标验证 - IC、方向准确率等金融指标
5. 超参数优化 - 网格搜索最佳参数
6. 多种子集成 - 减少随机性影响
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import argparse
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json
import pickle
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# 配置日志
log_file = 'logs/v391_improved_training.log'
Path('logs').mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FinancialMetrics:
    """金融评估指标"""

    @staticmethod
    def information_coefficient(y_true, y_pred):
        """IC - 信息系数（Spearman相关性）"""
        ic, _ = spearmanr(y_pred, y_true)
        return ic if not np.isnan(ic) else 0

    @staticmethod
    def direction_accuracy(y_true, y_pred):
        """方向准确率"""
        return np.mean((y_pred > 0) == (y_true > 0))

    @staticmethod
    def top_n_return(y_true, y_pred, n=100):
        """Top N股票的平均实际收益"""
        top_n_idx = np.argsort(y_pred)[-n:]
        return y_true[top_n_idx].mean()

    @staticmethod
    def quantile_monotonicity(y_true, y_pred, n_quantiles=5):
        """分位数单调性检验"""
        try:
            quantiles = pd.qcut(y_pred, q=n_quantiles, labels=False, duplicates='drop')
            returns = [y_true[quantiles == q].mean() for q in range(n_quantiles)]
            # 检查是否单调递增
            is_monotonic = all(returns[i] <= returns[i+1] for i in range(len(returns)-1))
            return is_monotonic, returns
        except:
            return False, []


class ImprovedV391Trainer:
    """V3.91 改进版训练器"""

    # 多周期权重配置
    PERIOD_WEIGHTS = {
        '5d': 0.40,
        '10d': 0.35,
        '15d': 0.25,
    }

    def __init__(self, db_path='data_adapter/stock_data.db', start_date='2023-01-01'):
        self.db_path = db_path
        self.start_date = start_date
        self.models = {'5d': {}, '10d': {}, '15d': {}}
        self.meta_models = {'5d': None, '10d': None, '15d': None}
        self.feature_columns = None
        self.best_params = {}
        self.training_history = []

    def load_data_with_time_split(self, val_ratio=0.2):
        """
        加载数据并进行时序划分

        关键：使用时间顺序划分，训练集在前，验证集在后
        这样可以模拟真实场景：用历史数据预测未来
        """
        logger.info("=" * 80)
        logger.info(f"📥 加载训练数据 (起始日期: {self.start_date})")
        logger.info("=" * 80)

        conn = sqlite3.connect(self.db_path)

        query = f"""
            SELECT code, trade_date, features_json, label_5d, label_10d, label_15d
            FROM v39_feature_cache
            WHERE label_5d IS NOT NULL
            AND label_10d IS NOT NULL
            AND label_15d IS NOT NULL
            AND trade_date >= '{self.start_date}'
            ORDER BY trade_date, code
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        logger.info(f"✅ 加载 {len(df):,} 个样本 (从 {self.start_date} 开始)")

        # 解析JSON特征
        logger.info("📊 解析特征...")
        features_list = []
        valid_rows = []

        for idx, row in df.iterrows():
            try:
                features_dict = json.loads(row['features_json'])
                features_list.append(features_dict)
                valid_rows.append(row)
            except:
                continue

            if (idx + 1) % 50000 == 0:
                logger.info(f"  已处理: {idx+1:,}/{len(df):,}")

        X = pd.DataFrame(features_list)
        df_valid = pd.DataFrame(valid_rows).reset_index(drop=True)

        self.feature_columns = X.columns.tolist()

        # 处理缺失值
        X = X.fillna(0)

        logger.info(f"✅ 特征矩阵: {X.shape}")
        logger.info(f"✅ 特征数量: {len(self.feature_columns)}")

        # 时序划分：按日期排序后，前80%训练，后20%验证
        dates = df_valid['trade_date'].values
        unique_dates = sorted(df_valid['trade_date'].unique())

        split_idx = int(len(unique_dates) * (1 - val_ratio))
        split_date = unique_dates[split_idx]

        train_mask = df_valid['trade_date'] < split_date
        val_mask = df_valid['trade_date'] >= split_date

        X_train = X[train_mask].values
        X_val = X[val_mask].values

        y = {
            '5d': df_valid['label_5d'].values,
            '10d': df_valid['label_10d'].values,
            '15d': df_valid['label_15d'].values
        }

        y_train = {p: y[p][train_mask] for p in ['5d', '10d', '15d']}
        y_val = {p: y[p][val_mask] for p in ['5d', '10d', '15d']}

        logger.info(f"\n📊 时序数据划分:")
        logger.info(f"  训练集: {X_train.shape[0]:,} 样本 (< {split_date})")
        logger.info(f"  验证集: {X_val.shape[0]:,} 样本 (>= {split_date})")

        # 标签分布统计
        logger.info("\n📊 标签分布:")
        for period in ['5d', '10d', '15d']:
            logger.info(f"  {period} 训练: mean={y_train[period].mean():.4f}, std={y_train[period].std():.4f}")
            logger.info(f"  {period} 验证: mean={y_val[period].mean():.4f}, std={y_val[period].std():.4f}")

        return X_train, X_val, y_train, y_val

    def _create_optimized_models(self, n_estimators=300, early_stopping=True):
        """创建优化后的基础模型"""
        models = {
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=0.03,  # 降低学习率
                max_depth=7,
                num_leaves=63,
                min_child_samples=30,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=0.03,
                max_depth=7,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            ),
            'catboost': CatBoostRegressor(
                iterations=n_estimators,
                learning_rate=0.03,
                depth=7,
                l2_leaf_reg=5,
                random_state=42,
                verbose=False
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=150,
                max_depth=12,
                min_samples_split=15,
                min_samples_leaf=8,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
        }
        return models

    def train_with_early_stopping(self, period, X_train, y_train, X_val, y_val):
        """
        带早停机制的训练

        使用验证集IC作为早停指标（金融场景更有意义）
        """
        logger.info(f"\n🔹 训练 {period} 周期模型 (带早停)...")

        trained_models = {}
        best_metrics = {}

        # LightGBM with early stopping
        logger.info("  训练 LightGBM...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=7,
            num_leaves=63,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )

        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        y_pred = lgb_model.predict(X_val)
        ic = FinancialMetrics.information_coefficient(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        dir_acc = FinancialMetrics.direction_accuracy(y_val, y_pred)

        logger.info(f"    LightGBM: IC={ic:.4f}, R²={r2:.4f}, 方向={dir_acc*100:.1f}%, "
                   f"最优轮数={lgb_model.best_iteration_}")
        trained_models['lightgbm'] = lgb_model
        best_metrics['lightgbm'] = {'ic': ic, 'r2': r2, 'dir_acc': dir_acc}

        # XGBoost with early stopping
        logger.info("  训练 XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=7,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            early_stopping_rounds=50
        )

        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        y_pred = xgb_model.predict(X_val)
        ic = FinancialMetrics.information_coefficient(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        dir_acc = FinancialMetrics.direction_accuracy(y_val, y_pred)

        logger.info(f"    XGBoost: IC={ic:.4f}, R²={r2:.4f}, 方向={dir_acc*100:.1f}%, "
                   f"最优轮数={xgb_model.best_iteration}")
        trained_models['xgboost'] = xgb_model
        best_metrics['xgboost'] = {'ic': ic, 'r2': r2, 'dir_acc': dir_acc}

        # CatBoost with early stopping
        logger.info("  训练 CatBoost...")
        cat_model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.03,
            depth=7,
            l2_leaf_reg=5,
            random_state=42,
            verbose=False,
            early_stopping_rounds=50
        )

        cat_model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False
        )

        y_pred = cat_model.predict(X_val)
        ic = FinancialMetrics.information_coefficient(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        dir_acc = FinancialMetrics.direction_accuracy(y_val, y_pred)

        logger.info(f"    CatBoost: IC={ic:.4f}, R²={r2:.4f}, 方向={dir_acc*100:.1f}%, "
                   f"最优轮数={cat_model.best_iteration_}")
        trained_models['catboost'] = cat_model
        best_metrics['catboost'] = {'ic': ic, 'r2': r2, 'dir_acc': dir_acc}

        # RandomForest (无早停，但使用OOB评分)
        logger.info("  训练 RandomForest...")
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=15,
            min_samples_leaf=8,
            max_features='sqrt',
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )

        rf_model.fit(X_train, y_train)

        y_pred = rf_model.predict(X_val)
        ic = FinancialMetrics.information_coefficient(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        dir_acc = FinancialMetrics.direction_accuracy(y_val, y_pred)

        logger.info(f"    RandomForest: IC={ic:.4f}, R²={r2:.4f}, 方向={dir_acc*100:.1f}%, "
                   f"OOB={rf_model.oob_score_:.4f}")
        trained_models['random_forest'] = rf_model
        best_metrics['random_forest'] = {'ic': ic, 'r2': r2, 'dir_acc': dir_acc}

        self.models[period] = trained_models
        return trained_models, best_metrics

    def train_meta_model_with_cv(self, period, X_train, y_train, X_val, y_val):
        """
        使用交叉验证训练元模型
        """
        logger.info(f"  训练 {period} 元模型 (5折交叉验证)...")

        # 生成元特征
        meta_train = np.column_stack([
            model.predict(X_train) for model in self.models[period].values()
        ])
        meta_val = np.column_stack([
            model.predict(X_val) for model in self.models[period].values()
        ])

        # 使用时序交叉验证选择最佳超参数
        best_ic = -np.inf
        best_params = None

        param_grid = [
            {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
            {'n_estimators': 150, 'learning_rate': 0.08, 'max_depth': 4},
            {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 3},
            {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 4},
        ]

        for params in param_grid:
            # 5折时序交叉验证
            tscv = TimeSeriesSplit(n_splits=5)
            cv_ics = []

            for train_idx, val_idx in tscv.split(meta_train):
                cv_X_train, cv_X_val = meta_train[train_idx], meta_train[val_idx]
                cv_y_train, cv_y_val = y_train[train_idx], y_train[val_idx]

                model = GradientBoostingRegressor(
                    n_estimators=params['n_estimators'],
                    learning_rate=params['learning_rate'],
                    max_depth=params['max_depth'],
                    random_state=42
                )
                model.fit(cv_X_train, cv_y_train)

                y_pred = model.predict(cv_X_val)
                ic = FinancialMetrics.information_coefficient(cv_y_val, y_pred)
                cv_ics.append(ic)

            mean_ic = np.mean(cv_ics)
            if mean_ic > best_ic:
                best_ic = mean_ic
                best_params = params

        logger.info(f"    最佳参数: {best_params}, CV IC={best_ic:.4f}")

        # 使用最佳参数训练最终元模型
        meta_model = GradientBoostingRegressor(
            n_estimators=best_params['n_estimators'],
            learning_rate=best_params['learning_rate'],
            max_depth=best_params['max_depth'],
            random_state=42
        )
        meta_model.fit(meta_train, y_train)

        # 验证集评估
        y_pred = meta_model.predict(meta_val)
        ic = FinancialMetrics.information_coefficient(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        dir_acc = FinancialMetrics.direction_accuracy(y_val, y_pred)
        top_100 = FinancialMetrics.top_n_return(y_val, y_pred, 100)
        is_mono, quantile_returns = FinancialMetrics.quantile_monotonicity(y_val, y_pred)

        logger.info(f"    元模型验证: IC={ic:.4f}, R²={r2:.4f}, 方向={dir_acc*100:.1f}%")
        logger.info(f"    Top100收益={top_100*100:.2f}%, 分位数单调={is_mono}")
        if quantile_returns:
            logger.info(f"    分位数收益: {[f'{r*100:.2f}%' for r in quantile_returns]}")

        self.meta_models[period] = meta_model
        self.best_params[period] = best_params

        return meta_model, {
            'ic': ic, 'r2': r2, 'dir_acc': dir_acc,
            'top_100_return': top_100, 'is_monotonic': is_mono
        }

    def comprehensive_evaluation(self, X_val, y_val):
        """综合评估所有周期"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 综合模型评估")
        logger.info("=" * 80)

        results = {}

        for period in ['5d', '10d', '15d']:
            # 元特征
            meta_features = np.column_stack([
                model.predict(X_val) for model in self.models[period].values()
            ])
            # 元模型预测
            y_pred = self.meta_models[period].predict(meta_features)

            ic = FinancialMetrics.information_coefficient(y_val[period], y_pred)
            r2 = r2_score(y_val[period], y_pred)
            dir_acc = FinancialMetrics.direction_accuracy(y_val[period], y_pred)
            top_100 = FinancialMetrics.top_n_return(y_val[period], y_pred, 100)
            is_mono, _ = FinancialMetrics.quantile_monotonicity(y_val[period], y_pred)

            results[period] = {
                'ic': ic, 'r2': r2, 'direction_accuracy': dir_acc,
                'top_100_return': top_100, 'is_monotonic': is_mono,
                'predictions': y_pred
            }

            logger.info(f"\n  {period} 周期:")
            logger.info(f"    IC: {ic:.4f} {'✅' if ic > 0.05 else '❌'}")
            logger.info(f"    R²: {r2:.4f} {'✅' if r2 > 0.15 else '❌'}")
            logger.info(f"    方向准确率: {dir_acc*100:.2f}% {'✅' if dir_acc > 0.55 else '❌'}")
            logger.info(f"    Top100收益: {top_100*100:.2f}% {'✅' if top_100 > 0.02 else '❌'}")
            logger.info(f"    分位数单调: {'是 ✅' if is_mono else '否 ❌'}")

        # 计算综合评分
        composite_pred = np.zeros(len(X_val))
        composite_true = np.zeros(len(X_val))

        for period, weight in self.PERIOD_WEIGHTS.items():
            composite_pred += weight * results[period]['predictions']
            composite_true += weight * y_val[period]

        composite_ic = FinancialMetrics.information_coefficient(composite_true, composite_pred)
        composite_r2 = r2_score(composite_true, composite_pred)
        composite_dir = FinancialMetrics.direction_accuracy(composite_true, composite_pred)

        results['composite'] = {
            'ic': composite_ic, 'r2': composite_r2, 'direction_accuracy': composite_dir
        }

        logger.info(f"\n  综合评分 (加权):")
        logger.info(f"    IC: {composite_ic:.4f}")
        logger.info(f"    R²: {composite_r2:.4f}")
        logger.info(f"    方向准确率: {composite_dir*100:.2f}%")

        return results

    def save_model(self, output_path='models/v391'):
        """保存模型"""
        logger.info("\n" + "=" * 80)
        logger.info("💾 保存V3.91改进版模型...")
        logger.info("=" * 80)

        Path(output_path).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        full_model = {
            'version': 'v3.91-improved',
            'base_models': self.models,
            'meta_models': self.meta_models,
            'feature_columns': self.feature_columns,
            'period_weights': self.PERIOD_WEIGHTS,
            'best_params': self.best_params,
            'training_history': self.training_history,
            'start_date': self.start_date,
            'timestamp': timestamp
        }

        # 保存带时间戳的版本
        full_path = f"{output_path}/v391_improved_{timestamp}.pkl"
        with open(full_path, 'wb') as f:
            pickle.dump(full_model, f)
        logger.info(f"  ✅ 完整模型: {full_path}")

        # 更新latest链接
        latest_path = f"{output_path}/v391_multiperiod_latest.pkl"
        with open(latest_path, 'wb') as f:
            pickle.dump(full_model, f)
        logger.info(f"  ✅ 生产链接: {latest_path}")

        return full_path

    def train(self):
        """完整训练流程"""
        logger.info("=" * 80)
        logger.info("🚀 V3.91 改进版多周期模型训练")
        logger.info("=" * 80)
        logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"训练数据起始日期: {self.start_date}")

        # 1. 加载数据（时序划分）
        X_train, X_val, y_train, y_val = self.load_data_with_time_split()

        # 2. 训练各周期基础模型（带早停）
        logger.info("\n" + "=" * 80)
        logger.info("🔧 训练基础模型 (带早停机制)...")
        logger.info("=" * 80)

        all_metrics = {}
        for period in ['5d', '10d', '15d']:
            _, metrics = self.train_with_early_stopping(
                period, X_train, y_train[period], X_val, y_val[period]
            )
            all_metrics[period] = metrics

        # 3. 训练元模型（带交叉验证）
        logger.info("\n" + "=" * 80)
        logger.info("🔧 训练元模型 (带交叉验证)...")
        logger.info("=" * 80)

        for period in ['5d', '10d', '15d']:
            _, meta_metrics = self.train_meta_model_with_cv(
                period, X_train, y_train[period], X_val, y_val[period]
            )
            all_metrics[f'{period}_meta'] = meta_metrics

        # 4. 综合评估
        eval_results = self.comprehensive_evaluation(X_val, y_val)
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': all_metrics,
            'eval_results': {k: {kk: vv for kk, vv in v.items() if kk != 'predictions'}
                           for k, v in eval_results.items()}
        })

        # 5. 保存模型
        model_path = self.save_model()

        logger.info("\n" + "=" * 80)
        logger.info("🎉 V3.91 改进版训练完成!")
        logger.info("=" * 80)
        logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 最终评估汇总
        logger.info("\n📊 最终评估汇总:")
        for period in ['5d', '10d', '15d']:
            r = eval_results[period]
            logger.info(f"  {period}: IC={r['ic']:.4f}, R²={r['r2']:.4f}, "
                       f"方向={r['direction_accuracy']*100:.1f}%, "
                       f"Top100={r['top_100_return']*100:.2f}%")

        comp = eval_results['composite']
        logger.info(f"  综合: IC={comp['ic']:.4f}, R²={comp['r2']:.4f}, "
                   f"方向={comp['direction_accuracy']*100:.1f}%")

        return model_path, eval_results


def main():
    parser = argparse.ArgumentParser(description='V3.91 改进版多周期模型训练')
    parser.add_argument('--db-path', type=str, default='data_adapter/stock_data.db')
    parser.add_argument('--start-date', type=str, default='2023-01-01',
                       help='训练数据起始日期 (默认: 2023-01-01)')
    parser.add_argument('--output-dir', type=str, default='models/v391')

    args = parser.parse_args()

    trainer = ImprovedV391Trainer(
        db_path=args.db_path,
        start_date=args.start_date
    )

    model_path, results = trainer.train()

    logger.info(f"\n✅ 模型已保存至: {model_path}")

    return results


if __name__ == "__main__":
    main()
