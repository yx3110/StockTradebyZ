#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 Level 4 Quality Meta-learner
基于LightGBM的质量评分元学习器，解决V3.8质量评分聚集问题

核心功能：
1. 🤖 LightGBM回归模型架构
2. 📊 时间序列交叉验证
3. 🔧 自动超参数优化
4. 📈 模型性能评估
5. 🔍 特征重要性分析
6. 💾 模型序列化和加载
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Level4QualityMetaLearner:
    """Level 4 Quality Meta-learner - LightGBM质量评分模型"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        self.best_params = None
        self.training_history = {}
        self.model_metrics = {}

    def create_model(self, params: Optional[Dict] = None) -> lgb.LGBMRegressor:
        """
        创建LightGBM模型

        Args:
            params: 自定义参数，如果为None则使用默认参数

        Returns:
            配置好的LightGBM模型
        """
        try:
            # 默认参数 - 针对质量评分回归任务优化
            default_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 20,
                'min_child_weight': 1e-3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': self.random_state,
                'verbosity': -1,
                'force_row_wise': True  # 避免警告
            }

            # 合并自定义参数
            if params:
                default_params.update(params)

            model = lgb.LGBMRegressor(**default_params)
            logger.info(f"✅ LightGBM模型创建成功，参数: {len(default_params)}个")

            return model

        except Exception as e:
            logger.error(f"❌ 模型创建失败: {e}")
            raise

    def prepare_data(self, train_path: str, val_path: str, test_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        准备训练数据

        Args:
            train_path: 训练集路径
            val_path: 验证集路径
            test_path: 测试集路径

        Returns:
            (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        try:
            logger.info("📂 加载训练数据...")

            # 读取数据集
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            test_df = pd.read_csv(test_path)

            logger.info(f"   训练集: {len(train_df)} 条记录")
            logger.info(f"   验证集: {len(val_df)} 条记录")
            logger.info(f"   测试集: {len(test_df)} 条记录")

            # 提取特征列
            feature_columns = [col for col in train_df.columns if col.startswith('feature_')]
            self.feature_names = feature_columns

            if len(feature_columns) == 0:
                raise ValueError("未找到特征列（以'feature_'开头）")

            logger.info(f"   特征维度: {len(feature_columns)}")

            # 提取特征和标签
            X_train = train_df[feature_columns].values
            X_val = val_df[feature_columns].values
            X_test = test_df[feature_columns].values

            # 检查目标变量
            target_column = 'quality_overall'
            if target_column not in train_df.columns:
                raise ValueError(f"未找到目标变量列: {target_column}")

            y_train = train_df[target_column].values
            y_val = val_df[target_column].values
            y_test = test_df[target_column].values

            # 数据质量检查
            self._validate_data_quality(X_train, y_train, "训练集")
            self._validate_data_quality(X_val, y_val, "验证集")
            self._validate_data_quality(X_test, y_test, "测试集")

            logger.info("✅ 数据准备完成")
            return X_train, y_train, X_val, y_val, X_test, y_test

        except Exception as e:
            logger.error(f"❌ 数据准备失败: {e}")
            raise

    def _validate_data_quality(self, X: np.ndarray, y: np.ndarray, dataset_name: str):
        """验证数据质量"""
        try:
            # 检查形状
            logger.info(f"   {dataset_name}: X{X.shape}, y{y.shape}")

            # 检查缺失值
            nan_count_X = np.isnan(X).sum()
            nan_count_y = np.isnan(y).sum()
            if nan_count_X > 0:
                logger.warning(f"   ⚠️ {dataset_name} X中有{nan_count_X}个NaN值")
            if nan_count_y > 0:
                logger.warning(f"   ⚠️ {dataset_name} y中有{nan_count_y}个NaN值")

            # 检查目标变量分布
            y_stats = {
                'mean': np.mean(y),
                'std': np.std(y),
                'min': np.min(y),
                'max': np.max(y)
            }
            logger.info(f"   {dataset_name} 目标分布: mean={y_stats['mean']:.3f}, std={y_stats['std']:.3f}, range=[{y_stats['min']:.3f}, {y_stats['max']:.3f}]")

        except Exception as e:
            logger.warning(f"⚠️ {dataset_name}数据质量检查失败: {e}")

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   early_stopping_rounds: int = 50,
                   verbose_eval: int = 100) -> Dict[str, Any]:
        """
        训练模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            early_stopping_rounds: 早停轮数
            verbose_eval: 打印间隔

        Returns:
            训练历史信息
        """
        try:
            logger.info("🚀 开始模型训练...")

            # 创建模型
            self.model = self.create_model()

            # 🔧 LightGBM 4.x版本使用callbacks参数
            callbacks = [
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=verbose_eval)
            ]

            # 训练模型
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                eval_names=['train', 'val'],
                callbacks=callbacks
            )

            # 记录训练历史
            self.training_history = {
                'best_iteration': self.model.best_iteration_,
                'best_score': self.model.best_score_,
                'feature_importance': self.model.feature_importances_.tolist(),
                'n_features': len(self.feature_names) if self.feature_names else X_train.shape[1]
            }

            # 特征重要性
            if self.feature_names:
                self.feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
            else:
                # 如果没有特征名，使用默认命名
                feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
                self.feature_names = feature_names
                self.feature_importance = dict(zip(feature_names, self.model.feature_importances_))

            logger.info(f"✅ 模型训练完成，最佳迭代: {self.model.best_iteration_}")
            logger.info(f"   最佳验证分数: {self.model.best_score_}")

            return self.training_history

        except Exception as e:
            logger.error(f"❌ 模型训练失败: {e}")
            raise

    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                n_iter: int = 50, cv_folds: int = 3) -> Dict[str, Any]:
        """
        超参数优化

        Args:
            X_train: 训练特征
            y_train: 训练标签
            n_iter: 随机搜索迭代次数
            cv_folds: 交叉验证折数

        Returns:
            最佳参数
        """
        try:
            logger.info("🔧 开始超参数优化...")

            # 参数搜索空间
            param_distributions = {
                'num_leaves': [15, 31, 63, 127],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'feature_fraction': [0.7, 0.8, 0.9, 1.0],
                'bagging_fraction': [0.7, 0.8, 0.9, 1.0],
                'min_child_samples': [10, 20, 30, 50],
                'reg_alpha': [0.0, 0.1, 0.5, 1.0],
                'reg_lambda': [0.0, 0.1, 0.5, 1.0],
                'subsample': [0.7, 0.8, 0.9, 1.0]
            }

            # 基础模型
            base_model = self.create_model()

            # 时间序列交叉验证
            tscv = TimeSeriesSplit(n_splits=cv_folds)

            # 随机搜索
            random_search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_distributions,
                n_iter=n_iter,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                random_state=self.random_state,
                verbose=1
            )

            # 执行搜索
            random_search.fit(X_train, y_train)

            # 保存最佳参数
            self.best_params = random_search.best_params_
            best_score = -random_search.best_score_  # 转换为正值

            logger.info(f"✅ 超参数优化完成")
            logger.info(f"   最佳CV分数 (RMSE): {best_score:.6f}")
            logger.info(f"   最佳参数: {len(self.best_params)}个")

            # 显示关键参数
            key_params = ['num_leaves', 'learning_rate', 'feature_fraction', 'reg_alpha']
            for param in key_params:
                if param in self.best_params:
                    logger.info(f"     {param}: {self.best_params[param]}")

            return self.best_params

        except Exception as e:
            logger.error(f"❌ 超参数优化失败: {e}")
            raise

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray,
                      X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, float]:
        """
        评估模型性能

        Args:
            X_test: 测试特征
            y_test: 测试标签
            X_val: 验证特征（可选）
            y_val: 验证标签（可选）

        Returns:
            评估指标字典
        """
        try:
            logger.info("📊 开始模型评估...")

            if self.model is None:
                raise ValueError("模型尚未训练")

            # 预测
            y_pred_test = self.model.predict(X_test)

            # 计算基础指标
            test_metrics = self._calculate_metrics(y_test, y_pred_test, "测试集")

            # 如果提供验证集，也进行评估
            val_metrics = {}
            if X_val is not None and y_val is not None:
                y_pred_val = self.model.predict(X_val)
                val_metrics = self._calculate_metrics(y_val, y_pred_val, "验证集")

            # 合并指标
            all_metrics = {
                'test_rmse': test_metrics['rmse'],
                'test_mae': test_metrics['mae'],
                'test_r2': test_metrics['r2'],
                'test_pearson_corr': test_metrics['pearson_corr'],
                'test_spearman_corr': test_metrics['spearman_corr'],
                'test_prediction_std': test_metrics['prediction_std']
            }

            if val_metrics:
                all_metrics.update({
                    'val_rmse': val_metrics['rmse'],
                    'val_mae': val_metrics['mae'],
                    'val_r2': val_metrics['r2'],
                    'val_pearson_corr': val_metrics['pearson_corr'],
                    'val_spearman_corr': val_metrics['spearman_corr'],
                    'val_prediction_std': val_metrics['prediction_std']
                })

            self.model_metrics = all_metrics

            # 🎯 关键指标验证
            logger.info("🎯 关键指标验证:")
            pred_std = test_metrics['prediction_std']
            target_std = np.std(y_test)

            if pred_std > 0.15:
                logger.info(f"   ✅ 预测差异化: std={pred_std:.3f} (目标>0.15)")
            else:
                logger.warning(f"   ⚠️ 预测差异化不足: std={pred_std:.3f} (目标>0.15)")

            if test_metrics['pearson_corr'] > 0.3:
                logger.info(f"   ✅ 相关性: r={test_metrics['pearson_corr']:.3f} (目标>0.3)")
            else:
                logger.warning(f"   ⚠️ 相关性不足: r={test_metrics['pearson_corr']:.3f} (目标>0.3)")

            logger.info("✅ 模型评估完成")
            return all_metrics

        except Exception as e:
            logger.error(f"❌ 模型评估失败: {e}")
            raise

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str) -> Dict[str, float]:
        """计算评估指标"""
        try:
            # 基础回归指标
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            # 相关性指标
            pearson_corr, pearson_p = pearsonr(y_true, y_pred)
            spearman_corr, spearman_p = spearmanr(y_true, y_pred)

            # 预测分布指标
            prediction_std = np.std(y_pred)
            prediction_range = np.max(y_pred) - np.min(y_pred)

            metrics = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'pearson_corr': pearson_corr,
                'pearson_p': pearson_p,
                'spearman_corr': spearman_corr,
                'spearman_p': spearman_p,
                'prediction_std': prediction_std,
                'prediction_range': prediction_range
            }

            # 打印指标
            logger.info(f"   {dataset_name}指标:")
            logger.info(f"     RMSE: {rmse:.6f}")
            logger.info(f"     MAE: {mae:.6f}")
            logger.info(f"     R²: {r2:.6f}")
            logger.info(f"     Pearson相关性: {pearson_corr:.6f} (p={pearson_p:.6f})")
            logger.info(f"     预测std: {prediction_std:.6f}")
            logger.info(f"     预测范围: {prediction_range:.6f}")

            return metrics

        except Exception as e:
            logger.error(f"❌ {dataset_name}指标计算失败: {e}")
            return {}

    def analyze_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        分析特征重要性

        Args:
            top_n: 显示前N个重要特征

        Returns:
            特征重要性DataFrame
        """
        try:
            if self.model is None or self.feature_importance is None:
                raise ValueError("模型尚未训练或特征重要性不可用")

            logger.info("🔍 分析特征重要性...")

            # 创建特征重要性DataFrame
            importance_df = pd.DataFrame({
                'feature': list(self.feature_importance.keys()),
                'importance': list(self.feature_importance.values())
            }).sort_values('importance', ascending=False)

            # 计算重要性百分比
            total_importance = importance_df['importance'].sum()
            importance_df['importance_pct'] = (importance_df['importance'] / total_importance) * 100

            # 显示前N个重要特征
            logger.info(f"   Top {top_n} 重要特征:")
            for i, row in importance_df.head(top_n).iterrows():
                feature_name = row['feature'].replace('feature_', '')
                logger.info(f"     {i+1:>2}. {feature_name:<20}: {row['importance']:>8.1f} ({row['importance_pct']:>5.1f}%)")

            # 分析特征重要性分布
            top_10_pct = importance_df.head(10)['importance_pct'].sum()
            logger.info(f"   前10个特征重要性占比: {top_10_pct:.1f}%")

            return importance_df

        except Exception as e:
            logger.error(f"❌ 特征重要性分析失败: {e}")
            return pd.DataFrame()

    def save_model(self, model_path: str = "models/level4_quality_meta_learner.pkl",
                  metadata_path: str = "models/level4_model_metadata.json"):
        """
        保存模型和元数据

        Args:
            model_path: 模型保存路径
            metadata_path: 元数据保存路径
        """
        try:
            logger.info("💾 保存模型...")

            if self.model is None:
                raise ValueError("模型尚未训练")

            # 创建目录
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)

            # 保存模型
            joblib.dump(self.model, model_path)

            # 🔧 修复JSON序列化问题，转换numpy类型
            def convert_numpy_types(obj):
                """递归转换numpy类型为Python原生类型"""
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj

            # 保存元数据
            metadata = {
                'feature_names': self.feature_names,
                'feature_importance': convert_numpy_types(self.feature_importance),
                'best_params': convert_numpy_types(self.best_params),
                'training_history': convert_numpy_types(self.training_history),
                'model_metrics': convert_numpy_types(self.model_metrics),
                'random_state': self.random_state
            }

            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"✅ 模型保存成功:")
            logger.info(f"   模型文件: {model_path}")
            logger.info(f"   元数据: {metadata_path}")

        except Exception as e:
            logger.error(f"❌ 模型保存失败: {e}")
            raise

    def load_model(self, model_path: str = "models/level4_quality_meta_learner.pkl",
                  metadata_path: str = "models/level4_model_metadata.json"):
        """
        加载模型和元数据

        Args:
            model_path: 模型文件路径
            metadata_path: 元数据文件路径
        """
        try:
            logger.info("📂 加载模型...")

            # 加载模型
            self.model = joblib.load(model_path)

            # 加载元数据
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            self.feature_names = metadata.get('feature_names')
            self.feature_importance = metadata.get('feature_importance')
            self.best_params = metadata.get('best_params')
            self.training_history = metadata.get('training_history')
            self.model_metrics = metadata.get('model_metrics')

            logger.info(f"✅ 模型加载成功:")
            logger.info(f"   特征数量: {len(self.feature_names) if self.feature_names else 0}")
            logger.info(f"   最佳迭代: {self.training_history.get('best_iteration', 'N/A')}")

        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise

    def predict_quality_score(self, features: np.ndarray) -> np.ndarray:
        """
        预测质量评分

        Args:
            features: 特征数组 (n_samples, 25)

        Returns:
            质量评分预测 (n_samples,)
        """
        try:
            if self.model is None:
                raise ValueError("模型尚未训练或加载")

            predictions = self.model.predict(features)

            # 确保预测值在合理范围内
            predictions = np.clip(predictions, 0.0, 1.0)

            return predictions

        except Exception as e:
            logger.error(f"❌ 质量评分预测失败: {e}")
            raise

# 使用示例和测试
if __name__ == "__main__":
    # 创建Level 4模型
    level4_model = Level4QualityMetaLearner(random_state=42)

    try:
        # 准备数据
        X_train, y_train, X_val, y_val, X_test, y_test = level4_model.prepare_data(
            "level4_training_dataset_v2_train.csv",
            "level4_training_dataset_v2_val.csv",
            "level4_training_dataset_v2_test.csv"
        )

        # 超参数优化 (可选，比较耗时)
        print("\n🔧 是否进行超参数优化? (耗时较长)")
        optimize = input("输入 'y' 进行优化，其他键跳过: ").lower() == 'y'

        if optimize:
            best_params = level4_model.optimize_hyperparameters(X_train, y_train, n_iter=30, cv_folds=3)
            # 使用最佳参数重新创建模型
            level4_model.model = level4_model.create_model(best_params)

        # 训练模型
        training_history = level4_model.train_model(X_train, y_train, X_val, y_val)

        # 评估模型
        metrics = level4_model.evaluate_model(X_test, y_test, X_val, y_val)

        # 特征重要性分析
        importance_df = level4_model.analyze_feature_importance(top_n=15)

        # 保存模型
        level4_model.save_model()

        print("\n🎉 Level 4 Quality Meta-learner 训练完成!")
        print("✅ 模型已保存到 models/ 目录")

    except Exception as e:
        logger.error(f"❌ Level 4模型训练失败: {e}")
        import traceback
        traceback.print_exc()