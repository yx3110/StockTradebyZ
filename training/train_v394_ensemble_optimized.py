#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.9.4 三层Ensemble训练脚本 - 优化版

优化内容:
1. 移除MLP (对标准化敏感，容易失败)
2. 使用加权Ensemble替代Ridge回归
3. 增加LightGBM/XGBoost的正则化
4. 添加特征裁剪避免极端值
5. 基于验证集性能自动调整权重

作者: Claude Code
创建时间: 2025-12-14
"""

import numpy as np
import pandas as pd
import sqlite3
import json
from datetime import datetime, timedelta
import logging
from pathlib import Path
import joblib
from tqdm import tqdm
import argparse
import warnings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings('ignore')

# ML库
import lightgbm as lgb
import xgboost as xgb
try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("Warning: CatBoost not installed")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')


class TrainingHistoryRecorder:
    """训练历史记录器"""

    def __init__(self, version="v3.94"):
        self.version = version
        self.start_time = datetime.now()
        self.models = {}
        self.meta_model = None
        self.summary = {}

    def record_model(self, model_name, train_losses, val_losses, iterations=None,
                     best_iteration=None, metric_name='rmse', additional_metrics=None):
        if iterations is None:
            iterations = list(range(1, len(train_losses) + 1))

        self.models[model_name] = {
            'metric_name': metric_name,
            'train_losses': [float(x) for x in train_losses],
            'val_losses': [float(x) for x in val_losses],
            'iterations': iterations,
            'best_iteration': best_iteration or len(iterations),
            'final_train_loss': float(train_losses[-1]) if train_losses else None,
            'final_val_loss': float(val_losses[-1]) if val_losses else None,
            'additional_metrics': additional_metrics or {},
            'recorded_at': datetime.now().isoformat()
        }

    def record_cv_model(self, model_name, cv_scores, metric_name='cv_score'):
        self.models[model_name] = {
            'metric_name': metric_name,
            'cv_scores': [float(x) for x in cv_scores],
            'mean_cv_score': float(np.mean(cv_scores)),
            'std_cv_score': float(np.std(cv_scores)),
            'final_val_loss': float(np.mean(cv_scores)),
            'recorded_at': datetime.now().isoformat()
        }

    def record_meta_model(self, weights, model_rmses):
        self.meta_model = {
            'type': 'weighted_ensemble',
            'weights': {k: float(v) for k, v in weights.items()},
            'model_rmses': {k: float(v) for k, v in model_rmses.items()},
            'recorded_at': datetime.now().isoformat()
        }

    def set_summary(self, training_samples, validation_samples, feature_count,
                    final_metrics, training_params=None):
        self.summary = {
            'training_samples': training_samples,
            'validation_samples': validation_samples,
            'feature_count': feature_count,
            'final_metrics': {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                            for k, v in final_metrics.items()},
            'training_params': training_params or {}
        }

    def save(self, output_dir):
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        history = {
            'version': self.version,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'status': 'completed',
            'duration_seconds': int(duration),
            'models': self.models,
            'meta_model': self.meta_model,
            'summary': self.summary
        }

        output_path = Path(output_dir) / 'training_history_latest.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = Path(output_dir) / f'training_history_{timestamp}.json'
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        logger.info(f"训练历史已保存: {output_path}")
        return output_path


class V394OptimizedEnsembleTrainer:
    """V3.9.4 优化版Ensemble训练器"""

    def __init__(self):
        self.version = "V3.9.4-Optimized"
        self.scaler = RobustScaler()  # 使用RobustScaler，对异常值更鲁棒
        self.feature_columns = None
        self.base_models = {}
        self.model_weights = {}
        self.model_rmses = {}

        # 优化后的LightGBM参数
        self.lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,  # 减少复杂度
            'learning_rate': 0.03,  # 降低学习率
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'lambda_l1': 0.1,  # L1正则化
            'lambda_l2': 0.1,  # L2正则化
            'min_child_samples': 50,  # 增加最小样本数
            'verbose': -1,
            'n_estimators': 1000,
            'early_stopping_rounds': 100
        }

        # 优化后的XGBoost参数
        self.xgb_params = {
            'objective': 'reg:squarederror',
            'max_depth': 5,  # 减少深度
            'learning_rate': 0.03,
            'n_estimators': 1000,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.1,  # L1正则化
            'reg_lambda': 0.1,  # L2正则化
            'min_child_weight': 50,
            'early_stopping_rounds': 100,
            'verbosity': 0
        }

        # 优化后的RandomForest参数
        self.rf_params = {
            'n_estimators': 300,
            'max_depth': 8,
            'min_samples_split': 50,
            'min_samples_leaf': 20,
            'max_features': 'sqrt',
            'n_jobs': -1,
            'random_state': 42
        }

        # CatBoost参数
        if HAS_CATBOOST:
            self.cb_params = {
                'iterations': 1000,
                'learning_rate': 0.03,
                'depth': 5,
                'l2_leaf_reg': 5,
                'verbose': 0,
                'early_stopping_rounds': 100,
                'random_seed': 42
            }

        # GradientBoosting (替代MLP)
        self.gb_params = {
            'n_estimators': 300,
            'learning_rate': 0.03,
            'max_depth': 4,
            'min_samples_split': 50,
            'min_samples_leaf': 20,
            'subsample': 0.7,
            'random_state': 42
        }

    def load_merged_features(self, min_date=None, max_date=None):
        """加载并合并特征数据"""
        conn = sqlite3.connect(DB_PATH)
        logger.info("加载并合并特征数据...")

        date_filter = ""
        if min_date:
            date_filter += f" AND v.trade_date >= '{min_date}'"
        if max_date:
            date_filter += f" AND v.trade_date <= '{max_date}'"

        query = f"""
        SELECT
            v.code, v.trade_date, v.features_json, v.label_5d,
            a.market_active_mv_ratio, a.market_active_mv_zscore,
            a.market_active_mv_trend, a.stock_active_mv_rank,
            a.stock_relative_liquidity, a.market_cap_quality_score
        FROM v39_feature_cache v
        INNER JOIN active_mv_feature_cache a
            ON v.code = a.code AND v.trade_date = a.trade_date
        WHERE v.label_5d IS NOT NULL {date_filter}
        ORDER BY v.trade_date, v.code
        """

        df = pd.read_sql(query, conn)
        logger.info(f"  合并数据: {len(df):,} 条")
        conn.close()

        if df.empty:
            return None

        logger.info("  解析JSON特征...")
        features_list = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="解析特征"):
            try:
                features = json.loads(row['features_json'])
                features['code'] = row['code']
                features['trade_date'] = row['trade_date']
                features['label_5d'] = row['label_5d']
                features['market_active_mv_ratio'] = row['market_active_mv_ratio']
                features['market_active_mv_zscore'] = row['market_active_mv_zscore']
                features['market_active_mv_trend'] = row['market_active_mv_trend']
                features['stock_active_mv_rank'] = row['stock_active_mv_rank']
                features['stock_relative_liquidity'] = row['stock_relative_liquidity']
                features['market_cap_quality_score'] = row['market_cap_quality_score']
                features_list.append(features)
            except:
                continue

        result_df = pd.DataFrame(features_list)
        logger.info(f"  解析完成: {len(result_df):,} 条有效数据")
        return result_df

    def prepare_data(self, df, train_end, val_start, val_end, test_start):
        """准备数据集"""
        logger.info("准备数据集...")

        exclude_cols = ['code', 'trade_date', 'label_5d']
        self.feature_columns = [c for c in df.columns if c not in exclude_cols]
        logger.info(f"  特征数量: {len(self.feature_columns)}")

        train_df = df[df['trade_date'] <= train_end].copy()
        val_df = df[(df['trade_date'] >= val_start) & (df['trade_date'] <= val_end)].copy()
        test_df = df[df['trade_date'] >= test_start].copy()

        logger.info(f"  训练集: {len(train_df):,} 样本")
        logger.info(f"  验证集: {len(val_df):,} 样本")
        logger.info(f"  测试集: {len(test_df):,} 样本")

        X_train = train_df[self.feature_columns].values
        y_train = train_df['label_5d'].values
        X_val = val_df[self.feature_columns].values
        y_val = val_df['label_5d'].values
        X_test = test_df[self.feature_columns].values
        y_test = test_df['label_5d'].values

        # 处理NaN和Inf，并裁剪极端值
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        X_val = np.nan_to_num(X_val, nan=0, posinf=0, neginf=0)
        X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

        # 裁剪极端值 (1st-99th percentile)
        for i in range(X_train.shape[1]):
            p1, p99 = np.percentile(X_train[:, i], [1, 99])
            X_train[:, i] = np.clip(X_train[:, i], p1, p99)
            X_val[:, i] = np.clip(X_val[:, i], p1, p99)
            X_test[:, i] = np.clip(X_test[:, i], p1, p99)

        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        return {
            'train': (X_train, X_train_scaled, y_train, train_df),
            'val': (X_val, X_val_scaled, y_val, val_df),
            'test': (X_test, X_test_scaled, y_test, test_df)
        }

    def train_base_models(self, X_train, y_train, X_val, y_val, history_recorder=None):
        """训练基础模型"""
        logger.info("\n" + "="*60)
        logger.info("训练基础模型 (优化版)")
        logger.info("="*60)

        # 1. LightGBM
        logger.info("\n[1/5] 训练 LightGBM...")
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

        lgb_train_losses, lgb_val_losses = [], []

        def lgb_callback(env):
            if env.evaluation_result_list:
                lgb_val_losses.append(env.evaluation_result_list[0][2])
                train_pred = env.model.predict(X_train)
                lgb_train_losses.append(np.sqrt(mean_squared_error(y_train, train_pred)))

        self.base_models['lgb'] = lgb.train(
            {k: v for k, v in self.lgb_params.items() if k not in ['n_estimators', 'early_stopping_rounds']},
            lgb_train,
            num_boost_round=self.lgb_params['n_estimators'],
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(self.lgb_params['early_stopping_rounds'], verbose=False), lgb_callback]
        )

        lgb_pred = self.base_models['lgb'].predict(X_val)
        lgb_rmse = np.sqrt(mean_squared_error(y_val, lgb_pred))
        self.model_rmses['lgb'] = lgb_rmse
        logger.info(f"  LightGBM RMSE: {lgb_rmse:.4f}, 树数: {self.base_models['lgb'].num_trees()}")

        if history_recorder and lgb_train_losses:
            history_recorder.record_model('lightgbm', lgb_train_losses, lgb_val_losses,
                best_iteration=self.base_models['lgb'].best_iteration)

        # 2. XGBoost
        logger.info("\n[2/5] 训练 XGBoost...")
        self.base_models['xgb'] = xgb.XGBRegressor(**self.xgb_params)
        self.base_models['xgb'].fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)

        xgb_train_losses, xgb_val_losses = [], []
        if hasattr(self.base_models['xgb'], 'evals_result'):
            evals = self.base_models['xgb'].evals_result()
            xgb_train_losses = evals.get('validation_0', {}).get('rmse', [])
            xgb_val_losses = evals.get('validation_1', {}).get('rmse', [])

        xgb_pred = self.base_models['xgb'].predict(X_val)
        xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_pred))
        self.model_rmses['xgb'] = xgb_rmse
        logger.info(f"  XGBoost RMSE: {xgb_rmse:.4f}")

        if history_recorder and xgb_val_losses:
            history_recorder.record_model('xgboost', xgb_train_losses, xgb_val_losses)

        # 3. CatBoost
        if HAS_CATBOOST:
            logger.info("\n[3/5] 训练 CatBoost...")
            self.base_models['cb'] = cb.CatBoostRegressor(**self.cb_params)
            self.base_models['cb'].fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

            cb_train_losses, cb_val_losses = [], []
            if hasattr(self.base_models['cb'], 'get_evals_result'):
                evals = self.base_models['cb'].get_evals_result()
                cb_train_losses = evals.get('learn', {}).get('RMSE', [])
                cb_val_losses = evals.get('validation', {}).get('RMSE', [])

            cb_pred = self.base_models['cb'].predict(X_val)
            cb_rmse = np.sqrt(mean_squared_error(y_val, cb_pred))
            self.model_rmses['cb'] = cb_rmse
            logger.info(f"  CatBoost RMSE: {cb_rmse:.4f}")

            if history_recorder and cb_val_losses:
                history_recorder.record_model('catboost', cb_train_losses, cb_val_losses)
        else:
            logger.info("\n[3/5] 跳过 CatBoost")

        # 4. RandomForest
        logger.info("\n[4/5] 训练 RandomForest...")
        self.base_models['rf'] = RandomForestRegressor(**self.rf_params)
        self.base_models['rf'].fit(X_train, y_train)

        rf_pred = self.base_models['rf'].predict(X_val)
        rf_rmse = np.sqrt(mean_squared_error(y_val, rf_pred))
        self.model_rmses['rf'] = rf_rmse
        logger.info(f"  RandomForest RMSE: {rf_rmse:.4f}")

        if history_recorder:
            history_recorder.record_cv_model('random_forest', [rf_rmse])

        # 5. GradientBoosting (替代MLP)
        logger.info("\n[5/5] 训练 GradientBoosting (替代MLP)...")
        self.base_models['gb'] = GradientBoostingRegressor(**self.gb_params)
        self.base_models['gb'].fit(X_train, y_train)

        gb_pred = self.base_models['gb'].predict(X_val)
        gb_rmse = np.sqrt(mean_squared_error(y_val, gb_pred))
        self.model_rmses['gb'] = gb_rmse
        logger.info(f"  GradientBoosting RMSE: {gb_rmse:.4f}")

        if history_recorder:
            history_recorder.record_cv_model('gradient_boosting', [gb_rmse])

        logger.info("\n基础模型训练完成!")

    def compute_ensemble_weights(self):
        """基于RMSE计算加权Ensemble权重"""
        logger.info("\n" + "="*60)
        logger.info("计算Ensemble权重")
        logger.info("="*60)

        # 使用RMSE的倒数作为权重
        inv_rmses = {k: 1.0 / v for k, v in self.model_rmses.items()}
        total = sum(inv_rmses.values())
        self.model_weights = {k: v / total for k, v in inv_rmses.items()}

        logger.info("  模型权重分配:")
        for name, weight in sorted(self.model_weights.items(), key=lambda x: -x[1]):
            logger.info(f"    {name}: {weight:.3f} (RMSE: {self.model_rmses[name]:.4f})")

    def predict(self, X):
        """加权Ensemble预测"""
        predictions = {}

        predictions['lgb'] = self.base_models['lgb'].predict(X)
        predictions['xgb'] = self.base_models['xgb'].predict(X)
        if 'cb' in self.base_models:
            predictions['cb'] = self.base_models['cb'].predict(X)
        predictions['rf'] = self.base_models['rf'].predict(X)
        predictions['gb'] = self.base_models['gb'].predict(X)

        # 加权平均
        final_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            final_pred += self.model_weights[name] * pred

        return final_pred

    def evaluate(self, X, y_true, df, split_name="测试集"):
        """评估模型性能"""
        logger.info(f"\n{'='*60}")
        logger.info(f"评估 {split_name}")
        logger.info("="*60)

        predictions = self.predict(X)

        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        mae = mean_absolute_error(y_true, predictions)
        r2 = r2_score(y_true, predictions)
        ic = np.corrcoef(predictions, y_true)[0, 1]
        direction_acc = np.mean((predictions > 0) == (y_true > 0))

        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  IC: {ic:.4f}")
        logger.info(f"  方向准确率: {direction_acc*100:.2f}%")

        # Top N分析
        df_eval = df.copy()
        df_eval['prediction'] = predictions
        df_eval['actual'] = y_true

        results_by_date = []
        for date, group in df_eval.groupby('trade_date'):
            if len(group) >= 20:
                top10 = group.nlargest(10, 'prediction')
                top20 = group.nlargest(20, 'prediction')
                results_by_date.append({
                    'top10_return': top10['actual'].mean(),
                    'top10_win_rate': (top10['actual'] > 0).mean(),
                    'top20_return': top20['actual'].mean(),
                    'top20_win_rate': (top20['actual'] > 0).mean()
                })

        if results_by_date:
            results_df = pd.DataFrame(results_by_date)
            logger.info(f"\nTop N 分析 ({len(results_df)} 个交易日):")
            logger.info(f"  Top 10 平均收益: {results_df['top10_return'].mean()*100:.2f}%")
            logger.info(f"  Top 10 平均胜率: {results_df['top10_win_rate'].mean()*100:.2f}%")
            logger.info(f"  Top 20 平均收益: {results_df['top20_return'].mean()*100:.2f}%")
            logger.info(f"  Top 20 平均胜率: {results_df['top20_win_rate'].mean()*100:.2f}%")

            # 日均IC
            daily_ic = []
            for date, group in df_eval.groupby('trade_date'):
                if len(group) >= 20:
                    ic_d = np.corrcoef(group['prediction'], group['actual'])[0, 1]
                    if not np.isnan(ic_d):
                        daily_ic.append(ic_d)
            if daily_ic:
                logger.info(f"  日均IC: {np.mean(daily_ic):.4f}")

        return {
            'rmse': rmse, 'mae': mae, 'r2': r2, 'ic': ic,
            'direction_acc': direction_acc, 'predictions': predictions,
            'top10_return': results_df['top10_return'].mean() if results_by_date else 0,
            'top20_return': results_df['top20_return'].mean() if results_by_date else 0,
            'top10_win_rate': results_df['top10_win_rate'].mean() if results_by_date else 0,
            'top20_win_rate': results_df['top20_win_rate'].mean() if results_by_date else 0
        }

    def save_model(self, output_path):
        """保存模型"""
        model_data = {
            'version': self.version,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler,
            'base_models': self.base_models,
            'model_weights': self.model_weights,
            'model_rmses': self.model_rmses,
            'created_at': datetime.now().isoformat()
        }

        joblib.dump(model_data, output_path)
        import os
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"模型已保存: {output_path} ({size_mb:.1f} MB)")

    def get_feature_importance(self):
        """获取特征重要性"""
        if 'lgb' not in self.base_models:
            return None

        importance = self.base_models['lgb'].feature_importance(importance_type='gain')
        return pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)


def main():
    parser = argparse.ArgumentParser(description='V3.9.4 优化版Ensemble训练')
    parser.add_argument('--train-end', default='2025-06-30')
    parser.add_argument('--val-start', default='2025-07-08')
    parser.add_argument('--val-end', default='2025-09-15')
    parser.add_argument('--test-start', default='2025-09-23')
    parser.add_argument('--output-dir', default=str(PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v394'))

    args = parser.parse_args()

    logger.info("="*70)
    logger.info("V3.9.4 优化版Ensemble模型训练")
    logger.info("="*70)

    trainer = V394OptimizedEnsembleTrainer()
    history_recorder = TrainingHistoryRecorder(version="v3.94-optimized")

    # 加载数据
    df = trainer.load_merged_features()
    if df is None or len(df) == 0:
        logger.error("无法加载特征数据")
        return

    logger.info(f"总数据量: {len(df):,} 条")

    # 准备数据
    data = trainer.prepare_data(df, args.train_end, args.val_start, args.val_end, args.test_start)

    X_train, _, y_train, train_df = data['train']
    X_val, _, y_val, val_df = data['val']
    X_test, _, y_test, test_df = data['test']

    # 训练基础模型
    trainer.train_base_models(X_train, y_train, X_val, y_val, history_recorder)

    # 计算Ensemble权重
    trainer.compute_ensemble_weights()

    # 评估
    val_results = trainer.evaluate(X_val, y_val, val_df, "验证集")
    test_results = trainer.evaluate(X_test, y_test, test_df, "测试集")

    # 特征重要性
    logger.info("\n" + "="*60)
    logger.info("特征重要性 Top 15")
    logger.info("="*60)
    feature_imp = trainer.get_feature_importance()
    if feature_imp is not None:
        for i, (_, row) in enumerate(feature_imp.head(15).iterrows()):
            logger.info(f"  #{i+1}: {row['feature']} ({row['importance']:.1f})")

    # 记录训练摘要
    history_recorder.record_meta_model(trainer.model_weights, trainer.model_rmses)
    history_recorder.set_summary(
        training_samples=len(train_df),
        validation_samples=len(val_df),
        feature_count=len(trainer.feature_columns),
        final_metrics={
            'val_rmse': test_results['rmse'],
            'ic': test_results['ic'],
            'direction_accuracy': test_results['direction_acc'],
            'top10_return': test_results['top10_return'],
            'top20_return': test_results['top20_return']
        }
    )

    # 保存模型和历史
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f"{args.output_dir}/v394_optimized_{timestamp}.pkl"
    trainer.save_model(model_path)
    history_recorder.save(args.output_dir)

    logger.info("\n" + "="*70)
    logger.info("训练完成!")
    logger.info("="*70)


if __name__ == '__main__':
    main()
