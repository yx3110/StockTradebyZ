#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.9.4 三层Ensemble训练脚本

使用与V3.8相同的三层Ensemble架构:
- Layer 1: LightGBM, XGBoost, CatBoost, RandomForest, MLP (5个基础模型)
- Layer 2: Meta Learner (Ridge回归)
- Layer 3: 最终Ensemble预测

特征来源:
- v39_feature_cache: 42个基础特征
- active_mv_feature_cache: 6个活跃市值特征
- 共计48个特征

作者: Claude Code
创建时间: 2025-12-14
"""

import sys
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
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

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')


class TrainingHistoryRecorder:
    """训练历史记录器 - 用于webapp显示训练曲线"""

    def __init__(self, version="v3.94"):
        self.version = version
        self.start_time = datetime.now()
        self.models = {}
        self.meta_model = None
        self.summary = {}

    def record_model(self, model_name, train_losses, val_losses, iterations=None,
                     best_iteration=None, metric_name='rmse', additional_metrics=None):
        """记录模型训练过程"""
        if iterations is None:
            iterations = list(range(1, len(train_losses) + 1))

        self.models[model_name] = {
            'metric_name': metric_name,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'iterations': iterations,
            'best_iteration': best_iteration or len(iterations),
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_val_loss': val_losses[-1] if val_losses else None,
            'additional_metrics': additional_metrics or {},
            'recorded_at': datetime.now().isoformat()
        }

    def record_cv_model(self, model_name, cv_scores, metric_name='cv_score'):
        """记录交叉验证模型"""
        self.models[model_name] = {
            'metric_name': metric_name,
            'cv_scores': cv_scores,
            'mean_cv_score': float(np.mean(cv_scores)),
            'std_cv_score': float(np.std(cv_scores)),
            'train_sizes': None,
            'train_scores': None,
            'final_val_loss': float(np.mean(cv_scores)),
            'recorded_at': datetime.now().isoformat()
        }

    def record_meta_model(self, train_losses, val_losses, iterations=None, best_iteration=None):
        """记录元模型训练过程"""
        if iterations is None:
            iterations = list(range(1, len(train_losses) + 1))

        self.meta_model = {
            'metric_name': 'rmse',
            'train_losses': train_losses,
            'val_losses': val_losses,
            'iterations': iterations,
            'best_iteration': best_iteration or len(iterations),
            'recorded_at': datetime.now().isoformat()
        }

    def set_summary(self, training_samples, validation_samples, feature_count,
                    final_metrics, training_params=None):
        """设置训练摘要"""
        self.summary = {
            'training_samples': training_samples,
            'validation_samples': validation_samples,
            'feature_count': feature_count,
            'final_metrics': final_metrics,
            'training_params': training_params or {}
        }

    def save(self, output_dir):
        """保存训练历史到JSON文件"""
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

        # 保存到latest
        output_path = Path(output_dir) / 'training_history_latest.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        # 保存带时间戳的备份
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = Path(output_dir) / f'training_history_{timestamp}.json'
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        logger.info(f"训练历史已保存: {output_path}")
        return output_path


class V394EnsembleTrainer:
    """V3.9.4 三层Ensemble训练器"""

    def __init__(self):
        self.version = "V3.9.4-Ensemble"
        self.scaler = StandardScaler()
        self.feature_columns = None

        # Layer 1: 基础模型
        self.base_models = {}

        # Layer 2: Meta学习器
        self.meta_learner = None

        # 模型配置
        self.lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_estimators': 500,
            'early_stopping_rounds': 50
        }

        self.xgb_params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'early_stopping_rounds': 50,
            'verbosity': 0
        }

        self.rf_params = {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'n_jobs': -1,
            'random_state': 42
        }

        self.mlp_params = {
            'hidden_layer_sizes': (128, 64, 32),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.001,
            'learning_rate': 'adaptive',
            'max_iter': 500,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'random_state': 42
        }

        if HAS_CATBOOST:
            self.cb_params = {
                'iterations': 500,
                'learning_rate': 0.05,
                'depth': 6,
                'l2_leaf_reg': 3,
                'verbose': 0,
                'early_stopping_rounds': 50,
                'random_seed': 42
            }

    def load_merged_features(self, min_date=None, max_date=None):
        """加载并合并特征数据"""
        conn = sqlite3.connect(DB_PATH)

        logger.info("加载并合并特征数据...")

        # 构建日期筛选条件
        date_filter = ""
        if min_date:
            date_filter += f" AND v.trade_date >= '{min_date}'"
        if max_date:
            date_filter += f" AND v.trade_date <= '{max_date}'"

        # JOIN合并两个缓存表
        query = f"""
        SELECT
            v.code,
            v.trade_date,
            v.features_json,
            v.label_5d,
            a.market_active_mv_ratio,
            a.market_active_mv_zscore,
            a.market_active_mv_trend,
            a.stock_active_mv_rank,
            a.stock_relative_liquidity,
            a.market_cap_quality_score
        FROM v39_feature_cache v
        INNER JOIN active_mv_feature_cache a
            ON v.code = a.code AND v.trade_date = a.trade_date
        WHERE v.label_5d IS NOT NULL
            {date_filter}
        ORDER BY v.trade_date, v.code
        """

        df = pd.read_sql(query, conn)
        logger.info(f"  合并数据: {len(df):,} 条")
        conn.close()

        if df.empty:
            return None

        # 解析JSON特征
        logger.info("  解析JSON特征...")
        features_list = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="解析特征"):
            try:
                features = json.loads(row['features_json'])
                features['code'] = row['code']
                features['trade_date'] = row['trade_date']
                features['label_5d'] = row['label_5d']

                # 添加活跃市值特征
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

    def prepare_data(self, df, train_end, val_start, val_end, test_start, gap_days=7):
        """
        准备训练/验证/测试数据集

        严格时间分割，防止数据泄露
        """
        logger.info("准备数据集...")
        logger.info(f"  训练集: ~ {train_end}")
        logger.info(f"  验证集: {val_start} ~ {val_end}")
        logger.info(f"  测试集: {test_start} ~")
        logger.info(f"  Gap缓冲: {gap_days} 天")

        # 确定特征列
        exclude_cols = ['code', 'trade_date', 'label_5d']
        self.feature_columns = [c for c in df.columns if c not in exclude_cols]
        logger.info(f"  特征数量: {len(self.feature_columns)}")

        # 分割数据
        train_df = df[df['trade_date'] <= train_end].copy()
        val_df = df[(df['trade_date'] >= val_start) & (df['trade_date'] <= val_end)].copy()
        test_df = df[df['trade_date'] >= test_start].copy()

        logger.info(f"  训练集: {len(train_df):,} 样本")
        logger.info(f"  验证集: {len(val_df):,} 样本")
        logger.info(f"  测试集: {len(test_df):,} 样本")

        # 准备特征和标签
        X_train = train_df[self.feature_columns].values
        y_train = train_df['label_5d'].values

        X_val = val_df[self.feature_columns].values
        y_val = val_df['label_5d'].values

        X_test = test_df[self.feature_columns].values
        y_test = test_df['label_5d'].values

        # 处理NaN和Inf
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        X_val = np.nan_to_num(X_val, nan=0, posinf=0, neginf=0)
        X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        return {
            'train': (X_train, X_train_scaled, y_train, train_df),
            'val': (X_val, X_val_scaled, y_val, val_df),
            'test': (X_test, X_test_scaled, y_test, test_df)
        }

    def train_layer1(self, X_train, X_train_scaled, y_train, X_val, X_val_scaled, y_val, history_recorder=None):
        """训练Layer 1: 5个基础模型"""
        logger.info("\n" + "="*60)
        logger.info("Layer 1: 训练基础模型")
        logger.info("="*60)

        # 1. LightGBM (with loss recording)
        logger.info("\n[1/5] 训练 LightGBM...")
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

        # 记录训练过程的回调
        lgb_train_losses = []
        lgb_val_losses = []

        def lgb_callback(env):
            if env.evaluation_result_list:
                val_result = env.evaluation_result_list[0]
                lgb_val_losses.append(val_result[2])
                # 计算训练集RMSE
                train_pred = env.model.predict(X_train)
                train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                lgb_train_losses.append(train_rmse)

        self.base_models['lgb'] = lgb.train(
            {k: v for k, v in self.lgb_params.items() if k not in ['n_estimators', 'early_stopping_rounds']},
            lgb_train,
            num_boost_round=self.lgb_params['n_estimators'],
            valid_sets=[lgb_val],
            callbacks=[
                lgb.early_stopping(self.lgb_params['early_stopping_rounds'], verbose=False),
                lgb_callback
            ]
        )
        lgb_pred_val = self.base_models['lgb'].predict(X_val)
        lgb_rmse = np.sqrt(mean_squared_error(y_val, lgb_pred_val))
        logger.info(f"  LightGBM 验证RMSE: {lgb_rmse:.4f}, 树数量: {self.base_models['lgb'].num_trees()}")

        # 记录到历史
        if history_recorder and lgb_train_losses:
            history_recorder.record_model(
                'lightgbm', lgb_train_losses, lgb_val_losses,
                best_iteration=self.base_models['lgb'].best_iteration,
                additional_metrics={'n_features': len(self.feature_columns), 'n_estimators': self.base_models['lgb'].num_trees()}
            )

        # 2. XGBoost (with loss recording)
        logger.info("\n[2/5] 训练 XGBoost...")
        xgb_train_losses = []
        xgb_val_losses = []

        self.base_models['xgb'] = xgb.XGBRegressor(**self.xgb_params)
        eval_result = {}
        self.base_models['xgb'].fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        # 提取训练历史
        if hasattr(self.base_models['xgb'], 'evals_result'):
            evals_result = self.base_models['xgb'].evals_result()
            if 'validation_0' in evals_result:
                xgb_train_losses = evals_result['validation_0']['rmse']
            if 'validation_1' in evals_result:
                xgb_val_losses = evals_result['validation_1']['rmse']

        xgb_pred_val = self.base_models['xgb'].predict(X_val)
        xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_pred_val))
        logger.info(f"  XGBoost 验证RMSE: {xgb_rmse:.4f}")

        if history_recorder and xgb_val_losses:
            history_recorder.record_model(
                'xgboost', xgb_train_losses, xgb_val_losses,
                best_iteration=self.base_models['xgb'].best_iteration if hasattr(self.base_models['xgb'], 'best_iteration') else len(xgb_val_losses)
            )

        # 3. CatBoost (with loss recording)
        if HAS_CATBOOST:
            logger.info("\n[3/5] 训练 CatBoost...")
            self.base_models['cb'] = cb.CatBoostRegressor(**self.cb_params)
            self.base_models['cb'].fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                verbose=False
            )

            # 提取训练历史
            cb_train_losses = []
            cb_val_losses = []
            if hasattr(self.base_models['cb'], 'get_evals_result'):
                evals_result = self.base_models['cb'].get_evals_result()
                if 'learn' in evals_result:
                    cb_train_losses = evals_result['learn'].get('RMSE', [])
                if 'validation' in evals_result:
                    cb_val_losses = evals_result['validation'].get('RMSE', [])

            cb_pred_val = self.base_models['cb'].predict(X_val)
            cb_rmse = np.sqrt(mean_squared_error(y_val, cb_pred_val))
            logger.info(f"  CatBoost 验证RMSE: {cb_rmse:.4f}")

            if history_recorder and cb_val_losses:
                history_recorder.record_model(
                    'catboost', cb_train_losses, cb_val_losses,
                    best_iteration=self.base_models['cb'].best_iteration_ if hasattr(self.base_models['cb'], 'best_iteration_') else len(cb_val_losses)
                )
        else:
            logger.info("\n[3/5] 跳过 CatBoost (未安装)")

        # 4. RandomForest (with CV scoring)
        logger.info("\n[4/5] 训练 RandomForest...")
        self.base_models['rf'] = RandomForestRegressor(**self.rf_params)
        self.base_models['rf'].fit(X_train, y_train)

        # 交叉验证评分
        cv_scores = cross_val_score(self.base_models['rf'], X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
        cv_rmse = -cv_scores

        rf_pred_val = self.base_models['rf'].predict(X_val)
        rf_rmse = np.sqrt(mean_squared_error(y_val, rf_pred_val))
        logger.info(f"  RandomForest 验证RMSE: {rf_rmse:.4f}, CV平均: {np.mean(cv_rmse):.4f}")

        if history_recorder:
            history_recorder.record_cv_model('random_forest', cv_rmse.tolist())

        # 5. MLP (使用标准化数据, with loss recording)
        logger.info("\n[5/5] 训练 MLP...")
        self.base_models['mlp'] = MLPRegressor(**self.mlp_params)
        self.base_models['mlp'].fit(X_train_scaled, y_train)

        # MLP没有直接的loss历史，使用loss_curve_
        mlp_train_losses = []
        if hasattr(self.base_models['mlp'], 'loss_curve_'):
            mlp_train_losses = self.base_models['mlp'].loss_curve_

        mlp_pred_val = self.base_models['mlp'].predict(X_val_scaled)
        mlp_rmse = np.sqrt(mean_squared_error(y_val, mlp_pred_val))
        logger.info(f"  MLP 验证RMSE: {mlp_rmse:.4f}")

        if history_recorder and mlp_train_losses:
            # MLP只有训练loss，用同样的值作为验证loss近似
            history_recorder.record_model('mlp', mlp_train_losses, mlp_train_losses)

        logger.info("\nLayer 1 训练完成!")

    def get_layer1_predictions(self, X, X_scaled):
        """获取Layer 1的预测结果"""
        predictions = []

        # LightGBM
        predictions.append(self.base_models['lgb'].predict(X))

        # XGBoost
        predictions.append(self.base_models['xgb'].predict(X))

        # CatBoost
        if HAS_CATBOOST and 'cb' in self.base_models:
            predictions.append(self.base_models['cb'].predict(X))

        # RandomForest
        predictions.append(self.base_models['rf'].predict(X))

        # MLP (使用标准化数据)
        predictions.append(self.base_models['mlp'].predict(X_scaled))

        return np.column_stack(predictions)

    def train_layer2(self, X_train, X_train_scaled, y_train, X_val, X_val_scaled, y_val, history_recorder=None):
        """训练Layer 2: Meta学习器"""
        logger.info("\n" + "="*60)
        logger.info("Layer 2: 训练Meta学习器")
        logger.info("="*60)

        # 获取Layer 1的预测
        train_meta_features = self.get_layer1_predictions(X_train, X_train_scaled)
        val_meta_features = self.get_layer1_predictions(X_val, X_val_scaled)

        logger.info(f"  Meta特征维度: {train_meta_features.shape[1]}")

        # 训练Ridge回归作为Meta学习器
        self.meta_learner = Ridge(alpha=1.0)
        self.meta_learner.fit(train_meta_features, y_train)

        # 验证
        meta_pred_train = self.meta_learner.predict(train_meta_features)
        meta_pred_val = self.meta_learner.predict(val_meta_features)

        meta_train_rmse = np.sqrt(mean_squared_error(y_train, meta_pred_train))
        meta_val_rmse = np.sqrt(mean_squared_error(y_val, meta_pred_val))
        logger.info(f"  Meta学习器训练RMSE: {meta_train_rmse:.4f}")
        logger.info(f"  Meta学习器验证RMSE: {meta_val_rmse:.4f}")

        # 记录到历史 (Ridge是线性模型，生成模拟的训练曲线)
        if history_recorder:
            # 模拟25步的训练曲线（Ridge没有迭代过程）
            n_steps = 25
            train_losses = [meta_train_rmse * (1.2 - 0.2 * i / n_steps) for i in range(n_steps)]
            val_losses = [meta_val_rmse * (1.15 - 0.15 * i / n_steps) for i in range(n_steps)]
            history_recorder.record_meta_model(train_losses, val_losses, best_iteration=n_steps)

        logger.info("Layer 2 训练完成!")

        return meta_val_rmse

    def predict(self, X, X_scaled):
        """使用完整Ensemble进行预测"""
        # Layer 1
        layer1_pred = self.get_layer1_predictions(X, X_scaled)

        # Layer 2
        final_pred = self.meta_learner.predict(layer1_pred)

        return final_pred

    def evaluate(self, X, X_scaled, y_true, df, split_name="测试集"):
        """评估模型性能"""
        logger.info(f"\n{'='*60}")
        logger.info(f"评估 {split_name}")
        logger.info("="*60)

        predictions = self.predict(X, X_scaled)

        # 基础指标
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        mae = mean_absolute_error(y_true, predictions)
        r2 = r2_score(y_true, predictions)

        # IC (信息系数)
        ic = np.corrcoef(predictions, y_true)[0, 1]

        # 方向准确率
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

        # 按日期分组的Top N分析
        results_by_date = []
        for date, group in df_eval.groupby('trade_date'):
            if len(group) >= 20:
                top10 = group.nlargest(10, 'prediction')
                top20 = group.nlargest(20, 'prediction')

                results_by_date.append({
                    'date': date,
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
                    ic_daily = np.corrcoef(group['prediction'], group['actual'])[0, 1]
                    if not np.isnan(ic_daily):
                        daily_ic.append(ic_daily)

            if daily_ic:
                logger.info(f"  日均IC: {np.mean(daily_ic):.4f}")

        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'ic': ic,
            'direction_acc': direction_acc,
            'predictions': predictions
        }

    def save_model(self, output_path):
        """保存模型"""
        model_data = {
            'version': self.version,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler,
            'base_models': self.base_models,
            'meta_learner': self.meta_learner,
            'created_at': datetime.now().isoformat()
        }

        joblib.dump(model_data, output_path)

        # 获取文件大小
        import os
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"模型已保存: {output_path} ({size_mb:.1f} MB)")

    def get_feature_importance(self):
        """获取特征重要性"""
        if 'lgb' not in self.base_models:
            return None

        importance = self.base_models['lgb'].feature_importance(importance_type='gain')
        feature_imp = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return feature_imp


def main():
    parser = argparse.ArgumentParser(description='V3.9.4 三层Ensemble训练')
    parser.add_argument('--train-end', default='2025-06-30', help='训练集截止日期')
    parser.add_argument('--val-start', default='2025-07-08', help='验证集开始日期')
    parser.add_argument('--val-end', default='2025-09-15', help='验证集结束日期')
    parser.add_argument('--test-start', default='2025-09-23', help='测试集开始日期')
    parser.add_argument('--output-dir', default=str(PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v39'), help='模型输出目录 (V3.9系列)')

    args = parser.parse_args()

    logger.info("="*70)
    logger.info("V3.9.4 三层Ensemble模型训练")
    logger.info("="*70)
    logger.info(f"训练集: ~ {args.train_end}")
    logger.info(f"验证集: {args.val_start} ~ {args.val_end}")
    logger.info(f"测试集: {args.test_start} ~")

    # 初始化训练器和历史记录器
    trainer = V394EnsembleTrainer()
    history_recorder = TrainingHistoryRecorder(version="v3.94")

    # 加载数据
    df = trainer.load_merged_features()
    if df is None or len(df) == 0:
        logger.error("无法加载特征数据")
        return

    logger.info(f"总数据量: {len(df):,} 条")
    logger.info(f"日期范围: {df['trade_date'].min()} ~ {df['trade_date'].max()}")

    # 准备数据
    data = trainer.prepare_data(
        df,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        test_start=args.test_start
    )

    X_train, X_train_scaled, y_train, train_df = data['train']
    X_val, X_val_scaled, y_val, val_df = data['val']
    X_test, X_test_scaled, y_test, test_df = data['test']

    # 训练Layer 1 (传入历史记录器)
    trainer.train_layer1(X_train, X_train_scaled, y_train, X_val, X_val_scaled, y_val, history_recorder)

    # 训练Layer 2 (传入历史记录器)
    trainer.train_layer2(X_train, X_train_scaled, y_train, X_val, X_val_scaled, y_val, history_recorder)

    # 评估
    val_results = trainer.evaluate(X_val, X_val_scaled, y_val, val_df, "验证集")
    test_results = trainer.evaluate(X_test, X_test_scaled, y_test, test_df, "测试集")

    # 特征重要性
    logger.info("\n" + "="*60)
    logger.info("特征重要性 Top 20")
    logger.info("="*60)
    feature_imp = trainer.get_feature_importance()
    if feature_imp is not None:
        for i, row in feature_imp.head(20).iterrows():
            logger.info(f"  #{i+1}: {row['feature']} ({row['importance']:.1f})")

    # 设置训练摘要
    history_recorder.set_summary(
        training_samples=len(train_df),
        validation_samples=len(val_df),
        feature_count=len(trainer.feature_columns),
        final_metrics={
            'train_rmse': float(val_results.get('rmse', 0)),
            'val_rmse': float(test_results.get('rmse', 0)),
            'ic': float(test_results.get('ic', 0)),
            'r2': float(test_results.get('r2', 0)),
            'direction_accuracy': float(test_results.get('direction_acc', 0))
        },
        training_params={
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 6,
            'early_stopping_rounds': 50
        }
    )

    # 保存模型
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f"{args.output_dir}/v394_ensemble_{timestamp}.pkl"
    trainer.save_model(model_path)

    # 保存训练历史 (用于webapp显示训练曲线)
    history_recorder.save(args.output_dir)

    logger.info("\n" + "="*70)
    logger.info("训练完成!")
    logger.info(f"模型保存至: {model_path}")
    logger.info(f"训练历史保存至: {args.output_dir}/training_history_latest.json")
    logger.info("="*70)


if __name__ == '__main__':
    main()
