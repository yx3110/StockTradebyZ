#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.8增量学习引擎核心组件
实现在线学习算法，支持模型的增量更新和自适应优化

Phase 3: 增量学习机制
- IncrementalLearner: 主增量学习器
- 支持LightGBM, XGBoost, CatBoost的增量训练
- 自适应学习率调整
- 性能监控和模型版本管理

Created: 2025-09-16
Author: Claude Code
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# ML模型 - 优雅处理依赖问题
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"Warning: LightGBM not available - {e}")
    lgb = None
    LGB_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"Warning: XGBoost not available - {e}")
    xgb = None
    XGB_AVAILABLE = False

try:
    import catboost as cb
    CB_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"Warning: CatBoost not available - {e}")
    cb = None
    CB_AVAILABLE = False

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# 添加项目路径
sys.path.append('/Users/yangxu/StockTradebyZ')

class IncrementalLearner:
    """
    增量学习器核心组件

    功能：
    1. 支持多种ML模型的增量训练
    2. 自适应学习率调整
    3. 模型性能监控
    4. 版本管理和回滚
    """

    def __init__(self,
                 models_config: Dict,
                 learning_rates: Dict[str, float],
                 forgetting_factors: Dict[str, float],
                 logger: logging.Logger,
                 model_save_path: str = 'incremental_learning/models/'):

        self.models_config = models_config
        self.learning_rates = learning_rates
        self.forgetting_factors = forgetting_factors
        self.logger = logger
        self.model_save_path = model_save_path

        # 确保模型保存路径存在
        os.makedirs(model_save_path, exist_ok=True)

        # 初始化模型存储
        self.base_models = {}
        self.model_versions = {}
        self.model_performance_history = {}

        # 增量学习参数
        self.update_counters = {}
        self.last_update_time = {}

        # 性能阈值
        self.performance_thresholds = {
            'r2_min': 0.3,
            'mse_max_increase': 0.5,
            'mae_max_increase': 0.3
        }

        self.logger.info("🚀 增量学习引擎初始化完成")

    def initialize_base_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        初始化基础模型
        """
        self.logger.info(f"📊 初始化基础模型，训练样本: {len(X_train)} 条")

        results = {}

        for model_name, config in self.models_config.items():
            try:
                if model_name == 'lightgbm':
                    model = self._init_lightgbm(X_train, y_train, config)
                elif model_name == 'xgboost':
                    model = self._init_xgboost(X_train, y_train, config)
                elif model_name == 'catboost':
                    model = self._init_catboost(X_train, y_train, config)
                elif model_name == 'random_forest':
                    model = self._init_random_forest(X_train, y_train, config)
                elif model_name == 'neural_network':
                    model = self._init_neural_network(X_train, y_train, config)
                else:
                    self.logger.warning(f"⚠️ 未支持的模型类型: {model_name}")
                    continue

                # 保存模型
                self.base_models[model_name] = model
                self.model_versions[model_name] = 1
                self.update_counters[model_name] = 0
                self.last_update_time[model_name] = datetime.now()

                # 评估初始性能
                y_pred = self._predict_model(model, X_train, model_name)
                performance = self._calculate_performance_metrics(y_train, y_pred)

                self.model_performance_history[model_name] = [performance]

                results[model_name] = {
                    'model': model,
                    'performance': performance,
                    'status': 'initialized'
                }

                self.logger.info(f"✅ {model_name} 初始化完成 - R²: {performance['r2']:.4f}")

            except Exception as e:
                self.logger.error(f"❌ {model_name} 初始化失败: {e}")
                results[model_name] = {'status': 'failed', 'error': str(e)}

        # 保存初始模型
        self._save_models_snapshot('initial')

        return results

    def _init_lightgbm(self, X_train, y_train, config):
        """初始化LightGBM模型"""
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': config.get('num_leaves', 31),
            'learning_rate': config.get('learning_rate', 0.05),
            'feature_fraction': config.get('feature_fraction', 0.9),
            'bagging_fraction': config.get('bagging_fraction', 0.8),
            'bagging_freq': config.get('bagging_freq', 5),
            'verbose': -1,
            'random_state': 42
        }

        train_data = lgb.Dataset(X_train, label=y_train)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=config.get('num_rounds', 100),
            valid_sets=[train_data],
            callbacks=[lgb.log_evaluation(0)]
        )

        return model

    def _init_xgboost(self, X_train, y_train, config):
        """初始化XGBoost模型"""
        params = {
            'objective': 'reg:squarederror',
            'max_depth': config.get('max_depth', 6),
            'learning_rate': config.get('learning_rate', 0.05),
            'subsample': config.get('subsample', 0.8),
            'colsample_bytree': config.get('colsample_bytree', 0.8),
            'random_state': 42,
            'verbosity': 0
        }

        dtrain = xgb.DMatrix(X_train, label=y_train)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=config.get('num_rounds', 100),
            evals=[(dtrain, 'train')],
            verbose_eval=False
        )

        return model

    def _init_catboost(self, X_train, y_train, config):
        """初始化CatBoost模型"""
        if cb is None:
            raise ImportError("CatBoost not available")

        model = cb.CatBoostRegressor(
            iterations=config.get('iterations', 100),
            learning_rate=config.get('learning_rate', 0.05),
            depth=config.get('depth', 6),
            l2_leaf_reg=config.get('l2_leaf_reg', 3),
            random_seed=42,
            verbose=False
        )

        model.fit(X_train, y_train, verbose=False)
        return model

    def _init_random_forest(self, X_train, y_train, config):
        """初始化随机森林模型"""
        model = RandomForestRegressor(
            n_estimators=config.get('n_estimators', 100),
            max_depth=config.get('max_depth', None),
            min_samples_split=config.get('min_samples_split', 2),
            min_samples_leaf=config.get('min_samples_leaf', 1),
            random_state=42
        )

        model.fit(X_train, y_train)
        return model

    def _init_neural_network(self, X_train, y_train, config):
        """初始化神经网络模型"""
        model = MLPRegressor(
            hidden_layer_sizes=config.get('hidden_layers', (100, 50)),
            activation=config.get('activation', 'relu'),
            solver=config.get('solver', 'adam'),
            alpha=config.get('alpha', 0.001),
            learning_rate=config.get('learning_rate', 'adaptive'),
            max_iter=config.get('max_iter', 500),
            random_state=42
        )

        model.fit(X_train, y_train)
        return model

    def incremental_update(self,
                          new_features: pd.DataFrame,
                          new_targets: pd.Series,
                          update_type: str = 'daily') -> Dict[str, Any]:
        """
        增量更新所有模型

        Args:
            new_features: 新的特征数据
            new_targets: 新的目标值
            update_type: 更新类型 ('daily', 'realtime', 'weekly')

        Returns:
            更新结果统计
        """
        self.logger.info(f"🔄 开始{update_type}增量更新，新数据: {len(new_features)} 条")

        results = {}

        for model_name, model in self.base_models.items():
            try:
                # 计算自适应学习率
                adaptive_lr = self._calculate_adaptive_learning_rate(
                    model_name, update_type
                )

                # 执行增量更新
                updated_model, update_stats = self._update_single_model(
                    model, model_name, new_features, new_targets,
                    adaptive_lr, update_type
                )

                if updated_model is not None:
                    # 验证更新效果
                    validation_result = self._validate_updated_model(
                        updated_model, model_name, new_features, new_targets
                    )

                    if validation_result['accept_update']:
                        # 接受更新
                        self.base_models[model_name] = updated_model
                        self.model_versions[model_name] += 1
                        self.update_counters[model_name] += 1
                        self.last_update_time[model_name] = datetime.now()

                        # 记录性能
                        self.model_performance_history[model_name].append(
                            validation_result['performance']
                        )

                        results[model_name] = {
                            'status': 'updated',
                            'version': self.model_versions[model_name],
                            'performance': validation_result['performance'],
                            'learning_rate': adaptive_lr,
                            'update_stats': update_stats
                        }

                        self.logger.info(f"✅ {model_name} 更新成功 v{self.model_versions[model_name]} - R²: {validation_result['performance']['r2']:.4f}")
                    else:
                        # 拒绝更新
                        results[model_name] = {
                            'status': 'rejected',
                            'reason': validation_result['rejection_reason'],
                            'performance': validation_result['performance']
                        }

                        self.logger.warning(f"⚠️ {model_name} 更新被拒绝: {validation_result['rejection_reason']}")
                else:
                    results[model_name] = {
                        'status': 'failed',
                        'reason': 'Model update returned None'
                    }

            except Exception as e:
                self.logger.error(f"❌ {model_name} 增量更新失败: {e}")
                results[model_name] = {
                    'status': 'error',
                    'error': str(e)
                }

        # 保存更新后的模型快照
        if any(r['status'] == 'updated' for r in results.values()):
            self._save_models_snapshot(f'{update_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

        return results

    def _calculate_adaptive_learning_rate(self,
                                        model_name: str,
                                        update_type: str) -> float:
        """
        计算自适应学习率
        """
        base_lr = self.learning_rates.get(model_name, 0.01)

        # 获取历史性能
        performance_history = self.model_performance_history.get(model_name, [])

        if len(performance_history) < 2:
            return base_lr

        # 计算性能变化趋势
        recent_performance = [p['r2'] for p in performance_history[-5:]]
        if len(recent_performance) >= 2:
            performance_trend = np.mean(np.diff(recent_performance))
        else:
            performance_trend = 0

        # 根据更新类型调整
        type_multiplier = {
            'daily': 1.0,
            'realtime': 0.5,  # 实时更新更保守
            'weekly': 1.5     # 周度更新更激进
        }.get(update_type, 1.0)

        # 根据性能趋势调整
        if performance_trend < -0.01:  # 性能下降
            trend_multiplier = 1.5  # 增加学习率
        elif performance_trend > 0.01:  # 性能提升
            trend_multiplier = 0.8  # 减少学习率
        else:
            trend_multiplier = 1.0

        # 根据更新次数调整（防止过拟合）
        update_count = self.update_counters.get(model_name, 0)
        count_multiplier = max(0.5, 1.0 - update_count * 0.01)

        adaptive_lr = base_lr * type_multiplier * trend_multiplier * count_multiplier
        adaptive_lr = np.clip(adaptive_lr, 0.001, 0.1)  # 限制范围

        return adaptive_lr

    def _update_single_model(self,
                           model: Any,
                           model_name: str,
                           new_features: pd.DataFrame,
                           new_targets: pd.Series,
                           learning_rate: float,
                           update_type: str) -> Tuple[Any, Dict]:
        """
        更新单个模型
        """
        update_stats = {
            'samples_used': len(new_features),
            'learning_rate': learning_rate,
            'update_type': update_type,
            'update_time': datetime.now().isoformat()
        }

        try:
            if model_name == 'lightgbm':
                updated_model = self._update_lightgbm(
                    model, new_features, new_targets, learning_rate, update_type
                )
            elif model_name == 'xgboost':
                updated_model = self._update_xgboost(
                    model, new_features, new_targets, learning_rate, update_type
                )
            elif model_name == 'catboost':
                updated_model = self._update_catboost(
                    model, new_features, new_targets, learning_rate, update_type
                )
            elif model_name in ['random_forest', 'neural_network']:
                # 对于不支持增量学习的模型，使用重训练
                updated_model = self._retrain_model(
                    model, model_name, new_features, new_targets
                )
            else:
                return None, update_stats

            update_stats['status'] = 'success'
            return updated_model, update_stats

        except Exception as e:
            update_stats['status'] = 'failed'
            update_stats['error'] = str(e)
            return None, update_stats

    def _update_lightgbm(self, model, new_features, new_targets, learning_rate, update_type):
        """更新LightGBM模型"""
        # LightGBM的增量训练
        new_train_data = lgb.Dataset(new_features, label=new_targets)

        # 获取原模型参数
        params = model.params.copy()
        params['learning_rate'] = learning_rate

        # 确定增量训练轮数
        num_rounds = {
            'daily': 20,
            'realtime': 5,
            'weekly': 50
        }.get(update_type, 20)

        # 增量训练
        updated_model = lgb.train(
            params,
            new_train_data,
            init_model=model,
            num_boost_round=num_rounds,
            callbacks=[lgb.log_evaluation(0)]  # 替换verbose_eval参数
        )

        return updated_model

    def _update_xgboost(self, model, new_features, new_targets, learning_rate, update_type):
        """更新XGBoost模型"""
        # XGBoost的增量训练
        new_dtrain = xgb.DMatrix(new_features, label=new_targets)

        # 确定增量训练轮数
        num_rounds = {
            'daily': 20,
            'realtime': 5,
            'weekly': 50
        }.get(update_type, 20)

        # 更新参数
        params = {
            'objective': 'reg:squarederror',
            'learning_rate': learning_rate,
            'verbosity': 0
        }

        # 增量训练
        updated_model = xgb.train(
            params,
            new_dtrain,
            num_boost_round=num_rounds,
            xgb_model=model,
            verbose_eval=False
        )

        return updated_model

    def _update_catboost(self, model, new_features, new_targets, learning_rate, update_type):
        """更新CatBoost模型"""
        if cb is None:
            raise ImportError("CatBoost not available")

        # CatBoost的增量训练
        num_iterations = {
            'daily': 20,
            'realtime': 5,
            'weekly': 50
        }.get(update_type, 20)

        # 创建新模型进行增量训练
        updated_model = cb.CatBoostRegressor(
            iterations=num_iterations,
            learning_rate=learning_rate,
            depth=model.get_params()['depth'],
            random_seed=42,
            verbose=False
        )

        # 增量训练
        updated_model.fit(
            new_features, new_targets,
            init_model=model,
            verbose=False
        )

        return updated_model

    def _retrain_model(self, model, model_name, new_features, new_targets):
        """重训练不支持增量学习的模型"""
        # 对于随机森林和神经网络，使用新数据重新训练
        # 这里可以考虑与历史数据混合训练

        if model_name == 'random_forest':
            # 获取原模型参数
            params = model.get_params()
            new_model = RandomForestRegressor(**params)
            new_model.fit(new_features, new_targets)
            return new_model

        elif model_name == 'neural_network':
            # 神经网络可以使用partial_fit进行增量学习
            if hasattr(model, 'partial_fit'):
                model.partial_fit(new_features, new_targets)
                return model
            else:
                # 重新训练
                params = model.get_params()
                new_model = MLPRegressor(**params)
                new_model.fit(new_features, new_targets)
                return new_model

        return model

    def _validate_updated_model(self,
                               updated_model: Any,
                               model_name: str,
                               validation_features: pd.DataFrame,
                               validation_targets: pd.Series) -> Dict:
        """
        验证更新后的模型
        """
        try:
            # 预测
            y_pred = self._predict_model(updated_model, validation_features, model_name)

            # 计算性能指标
            performance = self._calculate_performance_metrics(validation_targets, y_pred)

            # 获取历史性能用于比较
            historical_performance = self.model_performance_history.get(model_name, [])

            if not historical_performance:
                # 没有历史数据，接受更新
                return {
                    'accept_update': True,
                    'performance': performance,
                    'rejection_reason': None
                }

            # 与历史性能比较
            recent_avg_r2 = np.mean([p['r2'] for p in historical_performance[-3:]])
            recent_avg_mse = np.mean([p['mse'] for p in historical_performance[-3:]])

            # 决策规则
            accept_update = True
            rejection_reason = None

            # R²不能过低
            if performance['r2'] < self.performance_thresholds['r2_min']:
                accept_update = False
                rejection_reason = f"R² too low: {performance['r2']:.4f}"

            # R²不能显著下降
            elif performance['r2'] < recent_avg_r2 - 0.05:
                accept_update = False
                rejection_reason = f"R² decreased significantly: {performance['r2']:.4f} vs {recent_avg_r2:.4f}"

            # MSE不能显著增加
            elif performance['mse'] > recent_avg_mse * (1 + self.performance_thresholds['mse_max_increase']):
                accept_update = False
                rejection_reason = f"MSE increased too much: {performance['mse']:.4f} vs {recent_avg_mse:.4f}"

            return {
                'accept_update': accept_update,
                'performance': performance,
                'rejection_reason': rejection_reason,
                'comparison': {
                    'current_r2': performance['r2'],
                    'recent_avg_r2': recent_avg_r2,
                    'current_mse': performance['mse'],
                    'recent_avg_mse': recent_avg_mse
                }
            }

        except Exception as e:
            return {
                'accept_update': False,
                'performance': None,
                'rejection_reason': f"Validation error: {str(e)}"
            }

    def _predict_model(self, model: Any, features: pd.DataFrame, model_name: str) -> np.ndarray:
        """
        使用模型进行预测
        """
        if model_name == 'lightgbm':
            return model.predict(features)
        elif model_name == 'xgboost':
            dtest = xgb.DMatrix(features)
            return model.predict(dtest)
        elif model_name in ['catboost', 'random_forest', 'neural_network']:
            return model.predict(features)
        else:
            raise ValueError(f"Unsupported model type: {model_name}")

    def _calculate_performance_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算模型性能指标
        """
        return {
            'r2': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
        }

    def _save_models_snapshot(self, snapshot_name: str):
        """
        保存模型快照
        """
        try:
            snapshot_path = os.path.join(self.model_save_path, snapshot_name)
            os.makedirs(snapshot_path, exist_ok=True)

            for model_name, model in self.base_models.items():
                model_file = os.path.join(snapshot_path, f"{model_name}.pkl")

                if model_name in ['lightgbm', 'xgboost', 'catboost']:
                    # 使用模型自带的保存方法
                    if model_name == 'lightgbm':
                        model.save_model(model_file.replace('.pkl', '.txt'))
                    elif model_name == 'xgboost':
                        model.save_model(model_file.replace('.pkl', '.json'))
                    elif model_name == 'catboost':
                        model.save_model(model_file.replace('.pkl', '.cbm'))
                else:
                    # 使用joblib保存sklearn模型
                    joblib.dump(model, model_file)

            # 保存元数据
            metadata = {
                'snapshot_name': snapshot_name,
                'timestamp': datetime.now().isoformat(),
                'model_versions': self.model_versions.copy(),
                'update_counters': self.update_counters.copy(),
                'performance_history': {
                    name: history[-1] if history else None
                    for name, history in self.model_performance_history.items()
                }
            }

            metadata_file = os.path.join(snapshot_path, 'metadata.json')
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            self.logger.info(f"📁 模型快照已保存: {snapshot_name}")

        except Exception as e:
            self.logger.error(f"❌ 保存模型快照失败: {e}")

    def get_model_status(self) -> Dict[str, Any]:
        """
        获取所有模型的状态信息
        """
        status = {}

        for model_name in self.base_models.keys():
            performance_history = self.model_performance_history.get(model_name, [])
            current_performance = performance_history[-1] if performance_history else None

            status[model_name] = {
                'version': self.model_versions.get(model_name, 0),
                'update_count': self.update_counters.get(model_name, 0),
                'last_update': self.last_update_time.get(model_name, None),
                'current_performance': current_performance,
                'performance_trend': self._calculate_performance_trend(model_name)
            }

        return status

    def _calculate_performance_trend(self, model_name: str) -> str:
        """
        计算性能趋势
        """
        performance_history = self.model_performance_history.get(model_name, [])

        if len(performance_history) < 2:
            return 'insufficient_data'

        recent_r2 = [p['r2'] for p in performance_history[-5:]]
        trend = np.polyfit(range(len(recent_r2)), recent_r2, 1)[0]

        if trend > 0.01:
            return 'improving'
        elif trend < -0.01:
            return 'declining'
        else:
            return 'stable'

def main():
    """测试增量学习器"""
    print("🚀 测试增量学习器...")

    # 配置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 模型配置
    models_config = {
        'lightgbm': {
            'num_leaves': 31,
            'learning_rate': 0.05,
            'num_rounds': 100
        },
        'xgboost': {
            'max_depth': 6,
            'learning_rate': 0.05,
            'num_rounds': 100
        }
    }

    learning_rates = {
        'lightgbm': 0.01,
        'xgboost': 0.01,
        'catboost': 0.01
    }

    forgetting_factors = {
        'short': 0.95,
        'medium': 0.98,
        'long': 0.99
    }

    # 创建增量学习器
    learner = IncrementalLearner(
        models_config=models_config,
        learning_rates=learning_rates,
        forgetting_factors=forgetting_factors,
        logger=logger
    )

    # 生成测试数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_train = pd.Series(
        X_train.sum(axis=1) + np.random.randn(n_samples) * 0.1,
        name='target'
    )

    # 初始化模型
    print("\n📊 初始化基础模型...")
    init_results = learner.initialize_base_models(X_train, y_train)

    for model_name, result in init_results.items():
        if result['status'] == 'initialized':
            print(f"✅ {model_name}: R² = {result['performance']['r2']:.4f}")
        else:
            print(f"❌ {model_name}: {result.get('error', 'Failed')}")

    # 生成新数据进行增量更新
    print("\n🔄 测试增量更新...")
    X_new = pd.DataFrame(
        np.random.randn(100, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_new = pd.Series(
        X_new.sum(axis=1) + np.random.randn(100) * 0.1,
        name='target'
    )

    # 增量更新
    update_results = learner.incremental_update(X_new, y_new, update_type='daily')

    for model_name, result in update_results.items():
        if result['status'] == 'updated':
            print(f"✅ {model_name} 更新成功: v{result['version']} - R² = {result['performance']['r2']:.4f}")
        else:
            print(f"⚠️ {model_name}: {result['status']} - {result.get('reason', 'Unknown')}")

    # 显示模型状态
    print("\n📊 模型状态:")
    status = learner.get_model_status()
    for model_name, info in status.items():
        print(f"{model_name}:")
        print(f"  版本: v{info['version']}")
        print(f"  更新次数: {info['update_count']}")
        print(f"  性能趋势: {info['performance_trend']}")
        if info['current_performance']:
            print(f"  当前R²: {info['current_performance']['r2']:.4f}")

    print("\n✅ 增量学习器测试完成！")

if __name__ == "__main__":
    main()