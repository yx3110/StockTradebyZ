#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练历史记录器 - 记录模型训练过程中的loss曲线等指标

用于在webapp中展示训练过程的可视化
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class TrainingHistoryRecorder:
    """
    训练历史记录器

    记录训练过程中的loss、metrics等曲线数据，保存到JSON文件供webapp展示
    """

    def __init__(self, model_version: str, output_dir: str = 'models'):
        """
        初始化记录器

        Args:
            model_version: 模型版本 (如 'v3.9', 'v3.91')
            output_dir: 输出目录
        """
        self.model_version = model_version
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 训练历史数据
        self.history = {
            'version': model_version,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'status': 'training',
            'models': {},  # 各个模型的训练曲线
            'meta_model': None,  # 元模型的训练曲线
            'summary': {}  # 汇总信息
        }

    def record_model_training(
        self,
        model_name: str,
        train_losses: List[float],
        val_losses: List[float],
        metric_name: str = 'rmse',
        best_iteration: Optional[int] = None,
        additional_metrics: Optional[Dict[str, Any]] = None
    ):
        """
        记录单个模型的训练历史

        Args:
            model_name: 模型名称 (如 'lightgbm', 'xgboost', 'catboost')
            train_losses: 训练集loss列表
            val_losses: 验证集loss列表
            metric_name: 指标名称
            best_iteration: 最佳迭代次数
            additional_metrics: 额外的指标
        """
        self.history['models'][model_name] = {
            'metric_name': metric_name,
            'train_losses': [float(x) if not np.isnan(x) else None for x in train_losses],
            'val_losses': [float(x) if not np.isnan(x) else None for x in val_losses],
            'iterations': list(range(1, len(train_losses) + 1)),
            'best_iteration': best_iteration,
            'final_train_loss': float(train_losses[-1]) if train_losses and not np.isnan(train_losses[-1]) else None,
            'final_val_loss': float(val_losses[-1]) if val_losses and not np.isnan(val_losses[-1]) else None,
            'additional_metrics': additional_metrics or {},
            'recorded_at': datetime.now().isoformat()
        }

        logger.info(f"记录模型 {model_name} 训练历史: {len(train_losses)} 轮迭代")

    def record_lgb_training(self, model_name: str, lgb_model, metric_name: str = 'rmse'):
        """
        从LightGBM模型中提取训练历史

        Args:
            model_name: 模型名称
            lgb_model: 训练好的LightGBM模型
            metric_name: 评估指标名称
        """
        try:
            evals_result = lgb_model.evals_result_ if hasattr(lgb_model, 'evals_result_') else None

            if evals_result:
                # LightGBM的evals_result格式: {'valid_0': {'rmse': [...]}}
                train_key = 'training' if 'training' in evals_result else list(evals_result.keys())[0]
                val_key = 'valid_0' if 'valid_0' in evals_result else (
                    list(evals_result.keys())[1] if len(evals_result) > 1 else train_key
                )

                train_losses = evals_result.get(train_key, {}).get(metric_name, [])
                val_losses = evals_result.get(val_key, {}).get(metric_name, [])

                self.record_model_training(
                    model_name=model_name,
                    train_losses=train_losses if train_losses else val_losses,
                    val_losses=val_losses,
                    metric_name=metric_name,
                    best_iteration=lgb_model.best_iteration_ if hasattr(lgb_model, 'best_iteration_') else None,
                    additional_metrics={
                        'n_features': lgb_model.n_features_ if hasattr(lgb_model, 'n_features_') else None,
                        'n_estimators': lgb_model.n_estimators_ if hasattr(lgb_model, 'n_estimators_') else None
                    }
                )
            else:
                logger.warning(f"LightGBM模型 {model_name} 没有evals_result_属性")
        except Exception as e:
            logger.error(f"提取LightGBM训练历史失败: {e}")

    def record_xgb_training(self, model_name: str, xgb_model, metric_name: str = 'rmse'):
        """
        从XGBoost模型中提取训练历史

        Args:
            model_name: 模型名称
            xgb_model: 训练好的XGBoost模型
            metric_name: 评估指标名称
        """
        try:
            evals_result = xgb_model.evals_result() if hasattr(xgb_model, 'evals_result') else None

            if evals_result:
                # XGBoost的evals_result格式: {'validation_0': {'rmse': [...]}}
                keys = list(evals_result.keys())
                val_key = keys[0] if keys else None

                if val_key:
                    val_losses = evals_result[val_key].get(metric_name, [])
                    train_losses = val_losses  # XGBoost默认只记录eval_set

                    self.record_model_training(
                        model_name=model_name,
                        train_losses=train_losses,
                        val_losses=val_losses,
                        metric_name=metric_name,
                        best_iteration=xgb_model.best_iteration if hasattr(xgb_model, 'best_iteration') else None
                    )
            else:
                logger.warning(f"XGBoost模型 {model_name} 没有evals_result")
        except Exception as e:
            logger.error(f"提取XGBoost训练历史失败: {e}")

    def record_catboost_training(self, model_name: str, cat_model, metric_name: str = 'RMSE'):
        """
        从CatBoost模型中提取训练历史

        Args:
            model_name: 模型名称
            cat_model: 训练好的CatBoost模型
            metric_name: 评估指标名称
        """
        try:
            evals_result = cat_model.get_evals_result() if hasattr(cat_model, 'get_evals_result') else None

            if evals_result:
                # CatBoost的evals_result格式: {'learn': {'RMSE': [...]}, 'validation': {'RMSE': [...]}}
                train_losses = evals_result.get('learn', {}).get(metric_name, [])
                val_losses = evals_result.get('validation', {}).get(metric_name, [])

                self.record_model_training(
                    model_name=model_name,
                    train_losses=train_losses if train_losses else val_losses,
                    val_losses=val_losses,
                    metric_name=metric_name.lower(),
                    best_iteration=cat_model.best_iteration_ if hasattr(cat_model, 'best_iteration_') else None
                )
            else:
                logger.warning(f"CatBoost模型 {model_name} 没有evals_result")
        except Exception as e:
            logger.error(f"提取CatBoost训练历史失败: {e}")

    def record_sklearn_cv_scores(
        self,
        model_name: str,
        cv_scores: List[float],
        train_sizes: Optional[List[int]] = None,
        train_scores: Optional[List[float]] = None
    ):
        """
        记录Sklearn模型的交叉验证分数

        Args:
            model_name: 模型名称
            cv_scores: 交叉验证分数列表
            train_sizes: 训练集大小列表 (用于学习曲线)
            train_scores: 训练分数列表
        """
        self.history['models'][model_name] = {
            'metric_name': 'cv_score',
            'cv_scores': [float(x) for x in cv_scores],
            'mean_cv_score': float(np.mean(cv_scores)),
            'std_cv_score': float(np.std(cv_scores)),
            'train_sizes': train_sizes,
            'train_scores': [float(x) for x in train_scores] if train_scores else None,
            'recorded_at': datetime.now().isoformat()
        }

    def record_meta_model_training(
        self,
        train_losses: List[float],
        val_losses: List[float],
        metric_name: str = 'rmse',
        best_iteration: Optional[int] = None
    ):
        """
        记录元模型(Ensemble最终模型)的训练历史
        """
        self.history['meta_model'] = {
            'metric_name': metric_name,
            'train_losses': [float(x) if not np.isnan(x) else None for x in train_losses],
            'val_losses': [float(x) if not np.isnan(x) else None for x in val_losses],
            'iterations': list(range(1, len(train_losses) + 1)),
            'best_iteration': best_iteration,
            'recorded_at': datetime.now().isoformat()
        }

    def set_summary(
        self,
        training_samples: int,
        validation_samples: int,
        feature_count: int,
        final_metrics: Dict[str, float],
        training_params: Optional[Dict[str, Any]] = None
    ):
        """
        设置训练汇总信息

        Args:
            training_samples: 训练样本数
            validation_samples: 验证样本数
            feature_count: 特征数量
            final_metrics: 最终评估指标
            training_params: 训练参数
        """
        self.history['summary'] = {
            'training_samples': training_samples,
            'validation_samples': validation_samples,
            'feature_count': feature_count,
            'final_metrics': final_metrics,
            'training_params': training_params or {}
        }

    def finish(self, status: str = 'completed'):
        """
        完成记录并保存

        Args:
            status: 最终状态 ('completed', 'failed', 'interrupted')
        """
        self.history['end_time'] = datetime.now().isoformat()
        self.history['status'] = status

        # 计算总训练时间
        start = datetime.fromisoformat(self.history['start_time'])
        end = datetime.fromisoformat(self.history['end_time'])
        self.history['duration_seconds'] = (end - start).total_seconds()

        self.save()

    def save(self):
        """保存训练历史到JSON文件"""
        # 确定版本目录
        version_clean = self.model_version.replace('.', '')
        version_dir = self.output_dir / version_clean
        version_dir.mkdir(parents=True, exist_ok=True)

        # 使用时间戳命名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'training_history_{timestamp}.json'
        filepath = version_dir / filename

        # 同时保存一个latest版本
        latest_filepath = version_dir / 'training_history_latest.json'

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)

            with open(latest_filepath, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)

            logger.info(f"训练历史已保存: {filepath}")

        except Exception as e:
            logger.error(f"保存训练历史失败: {e}")

    @classmethod
    def load_latest(cls, model_version: str, models_dir: str = 'models') -> Optional[Dict]:
        """
        加载最新的训练历史

        Args:
            model_version: 模型版本
            models_dir: 模型目录

        Returns:
            训练历史字典，如果不存在返回None
        """
        version_clean = model_version.replace('.', '')
        filepath = Path(models_dir) / version_clean / 'training_history_latest.json'

        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载训练历史失败: {e}")

        return None

    @classmethod
    def list_all_histories(cls, model_version: str, models_dir: str = 'models') -> List[Dict]:
        """
        列出所有训练历史记录

        Args:
            model_version: 模型版本
            models_dir: 模型目录

        Returns:
            训练历史列表
        """
        version_clean = model_version.replace('.', '')
        version_dir = Path(models_dir) / version_clean

        histories = []
        if version_dir.exists():
            for filepath in sorted(version_dir.glob('training_history_*.json'), reverse=True):
                if 'latest' not in filepath.name:
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            history = json.load(f)
                            history['_filename'] = filepath.name
                            histories.append(history)
                    except Exception as e:
                        logger.error(f"加载 {filepath} 失败: {e}")

        return histories


class LightGBMCallback:
    """LightGBM训练回调，用于实时记录训练历史"""

    def __init__(self, recorder: TrainingHistoryRecorder, model_name: str):
        self.recorder = recorder
        self.model_name = model_name
        self.train_losses = []
        self.val_losses = []

    def __call__(self, env):
        """每轮迭代后调用"""
        if env.evaluation_result_list:
            for data_name, metric_name, value, is_higher_better in env.evaluation_result_list:
                if 'train' in data_name.lower():
                    self.train_losses.append(value)
                else:
                    self.val_losses.append(value)

    def finalize(self, best_iteration: Optional[int] = None):
        """训练完成后保存结果"""
        if self.val_losses:
            self.recorder.record_model_training(
                model_name=self.model_name,
                train_losses=self.train_losses if self.train_losses else self.val_losses,
                val_losses=self.val_losses,
                metric_name='rmse',
                best_iteration=best_iteration
            )
