#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 Level 4 质量评分后处理器
通过后处理技术扩展预测分布，解决差异化不足问题

核心技术:
1. 📈 分布拉伸 (Distribution Stretching)
2. 🎯 分位数映射 (Quantile Mapping)
3. 🔧 平滑约束 (Smooth Constraints)
4. ⚡ 实时优化 (Real-time Optimization)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class Level4QualityPostprocessor:
    """Level 4 质量评分后处理器"""

    def __init__(self, target_std: float = 0.15, target_range: Tuple[float, float] = (0.1, 0.9)):
        """
        初始化后处理器

        Args:
            target_std: 目标标准差
            target_range: 目标分布范围
        """
        self.target_std = target_std
        self.target_range = target_range
        self.fitted = False
        self.original_stats = {}
        self.transform_params = {}

    def fit_transform_parameters(self, raw_predictions: np.ndarray,
                                original_targets: np.ndarray = None) -> dict:
        """
        拟合变换参数

        Args:
            raw_predictions: 原始预测值
            original_targets: 原始目标值（用于参考分布）

        Returns:
            变换参数字典
        """
        try:
            logger.info("🔧 拟合后处理变换参数...")

            # 分析原始预测分布
            self.original_stats = {
                'mean': np.mean(raw_predictions),
                'std': np.std(raw_predictions),
                'min': np.min(raw_predictions),
                'max': np.max(raw_predictions),
                'range': np.max(raw_predictions) - np.min(raw_predictions),
                'q10': np.percentile(raw_predictions, 10),
                'q90': np.percentile(raw_predictions, 90)
            }

            logger.info(f"   原始分布: mean={self.original_stats['mean']:.3f}, std={self.original_stats['std']:.3f}")
            logger.info(f"   原始范围: [{self.original_stats['min']:.3f}, {self.original_stats['max']:.3f}]")

            # 🚀 方法1: 线性拉伸变换
            linear_params = self._fit_linear_stretch(raw_predictions)

            # 🚀 方法2: 分位数映射变换
            quantile_params = self._fit_quantile_mapping(raw_predictions)

            # 🚀 方法3: Beta分布变换
            beta_params = self._fit_beta_transform(raw_predictions)

            # 🚀 方法4: 混合变换
            hybrid_params = self._fit_hybrid_transform(raw_predictions)

            self.transform_params = {
                'linear': linear_params,
                'quantile': quantile_params,
                'beta': beta_params,
                'hybrid': hybrid_params
            }

            self.fitted = True
            logger.info("✅ 变换参数拟合完成")

            return self.transform_params

        except Exception as e:
            logger.error(f"❌ 变换参数拟合失败: {e}")
            raise

    def _fit_linear_stretch(self, predictions: np.ndarray) -> dict:
        """线性拉伸变换参数拟合"""
        try:
            current_std = np.std(predictions)
            current_mean = np.mean(predictions)

            # 计算拉伸因子
            stretch_factor = self.target_std / current_std if current_std > 0 else 1.0

            # 计算平移参数，确保均值合理
            target_mean = (self.target_range[0] + self.target_range[1]) / 2
            shift = target_mean - current_mean

            return {
                'stretch_factor': stretch_factor,
                'shift': shift,
                'original_mean': current_mean,
                'original_std': current_std
            }

        except Exception as e:
            logger.warning(f"线性拉伸参数拟合失败: {e}")
            return {'stretch_factor': 1.0, 'shift': 0.0}

    def _fit_quantile_mapping(self, predictions: np.ndarray) -> dict:
        """分位数映射变换参数拟合"""
        try:
            # 原始分位数
            percentiles = np.arange(1, 100, 1)  # 1%, 2%, ..., 99%
            original_quantiles = np.percentile(predictions, percentiles)

            # 目标分位数 (更均匀分布)
            target_quantiles = np.linspace(
                self.target_range[0],
                self.target_range[1],
                len(percentiles)
            )

            return {
                'percentiles': percentiles,
                'original_quantiles': original_quantiles,
                'target_quantiles': target_quantiles
            }

        except Exception as e:
            logger.warning(f"分位数映射参数拟合失败: {e}")
            return {}

    def _fit_beta_transform(self, predictions: np.ndarray) -> dict:
        """Beta分布变换参数拟合"""
        try:
            # 将预测值标准化到[0,1]
            pred_min, pred_max = np.min(predictions), np.max(predictions)
            if pred_max > pred_min:
                normalized_preds = (predictions - pred_min) / (pred_max - pred_min)
            else:
                normalized_preds = np.full_like(predictions, 0.5)

            # 拟合Beta分布参数
            try:
                alpha, beta, loc, scale = stats.beta.fit(normalized_preds, floc=0, fscale=1)
            except:
                # 如果拟合失败，使用默认参数
                alpha, beta = 2.0, 2.0

            return {
                'alpha': alpha,
                'beta': beta,
                'original_range': (pred_min, pred_max)
            }

        except Exception as e:
            logger.warning(f"Beta变换参数拟合失败: {e}")
            return {'alpha': 2.0, 'beta': 2.0}

    def _fit_hybrid_transform(self, predictions: np.ndarray) -> dict:
        """混合变换参数拟合 - 优化版本，避免极化"""
        try:
            # 结合线性拉伸和分位数映射的优点
            linear_params = self._fit_linear_stretch(predictions)

            # 分析预测分布特征
            current_std = np.std(predictions)
            current_range = np.max(predictions) - np.min(predictions)
            target_range_span = self.target_range[1] - self.target_range[0]

            # 计算需要的拉伸程度
            std_ratio = self.target_std / max(current_std, 1e-8)

            # 🔧 更保守的参数设置，避免过度变换
            if std_ratio > 5:  # 需要大幅拉伸
                range_expansion = min(1.8, target_range_span / max(current_range, 1e-8))
                nonlinear_power = 1.1
            elif std_ratio > 2:  # 中等拉伸
                range_expansion = min(1.4, target_range_span / max(current_range, 1e-8))
                nonlinear_power = 1.05
            else:  # 轻微调整
                range_expansion = min(1.2, target_range_span / max(current_range, 1e-8))
                nonlinear_power = 1.0

            # 确保参数在合理范围内
            range_expansion = max(1.0, min(2.0, range_expansion))

            return {
                **linear_params,
                'range_expansion': range_expansion,
                'nonlinear_power': nonlinear_power,
                'smooth_sigmoid': True  # 启用平滑sigmoid转换
            }

        except Exception as e:
            logger.warning(f"混合变换参数拟合失败: {e}")
            return {}

    def transform(self, predictions: np.ndarray, method: str = 'hybrid') -> np.ndarray:
        """
        应用后处理变换

        Args:
            predictions: 原始预测值
            method: 变换方法 ('linear', 'quantile', 'beta', 'hybrid')

        Returns:
            变换后的预测值
        """
        try:
            if not self.fitted:
                raise ValueError("后处理器尚未拟合，请先调用fit_transform_parameters")

            if method not in self.transform_params:
                logger.warning(f"变换方法{method}不可用，使用linear方法")
                method = 'linear'

            logger.info(f"🔄 应用{method}变换...")

            if method == 'linear':
                transformed = self._apply_linear_stretch(predictions)
            elif method == 'quantile':
                transformed = self._apply_quantile_mapping(predictions)
            elif method == 'beta':
                transformed = self._apply_beta_transform(predictions)
            elif method == 'hybrid':
                transformed = self._apply_hybrid_transform(predictions)
            else:
                transformed = predictions

            # 🔧 智能范围调整 (避免硬截断造成的极化)
            transformed = self._smart_range_adjustment(transformed)

            # 验证变换效果
            self._validate_transform_quality(predictions, transformed, method)

            return transformed

        except Exception as e:
            logger.error(f"❌ 变换应用失败: {e}")
            return predictions

    def _apply_linear_stretch(self, predictions: np.ndarray) -> np.ndarray:
        """应用线性拉伸变换"""
        params = self.transform_params['linear']

        # 中心化
        centered = predictions - params['original_mean']

        # 拉伸
        stretched = centered * params['stretch_factor']

        # 重新定位到目标范围
        target_center = (self.target_range[0] + self.target_range[1]) / 2
        transformed = stretched + target_center

        return transformed

    def _apply_quantile_mapping(self, predictions: np.ndarray) -> np.ndarray:
        """应用分位数映射变换"""
        params = self.transform_params['quantile']

        if not params:
            return predictions

        transformed = np.zeros_like(predictions)

        for i, pred in enumerate(predictions):
            # 找到最接近的分位数
            closest_idx = np.argmin(np.abs(params['original_quantiles'] - pred))
            transformed[i] = params['target_quantiles'][closest_idx]

        return transformed

    def _apply_beta_transform(self, predictions: np.ndarray) -> np.ndarray:
        """应用Beta分布变换"""
        params = self.transform_params['beta']

        # 标准化到[0,1]
        pred_min, pred_max = params['original_range']
        if pred_max > pred_min:
            normalized = (predictions - pred_min) / (pred_max - pred_min)
        else:
            normalized = np.full_like(predictions, 0.5)

        # 应用Beta分布的逆变换
        try:
            # 计算原始分布的CDF
            cdf_values = stats.beta.cdf(normalized, params['alpha'], params['beta'])

            # 使用均匀分布映射到目标范围
            transformed = self.target_range[0] + cdf_values * (self.target_range[1] - self.target_range[0])
        except:
            # 如果失败，使用线性映射
            transformed = self.target_range[0] + normalized * (self.target_range[1] - self.target_range[0])

        return transformed

    def _apply_hybrid_transform(self, predictions: np.ndarray) -> np.ndarray:
        """应用混合变换"""
        params = self.transform_params['hybrid']

        if not params:
            return self._apply_linear_stretch(predictions)

        # 第一步：线性拉伸
        centered = predictions - params['original_mean']
        stretched = centered * params['stretch_factor']

        # 第二步：非线性扩展
        sign = np.sign(stretched)
        abs_stretched = np.abs(stretched)
        power_expanded = sign * np.power(abs_stretched, params.get('nonlinear_power', 1.0))

        # 第三步：映射到目标范围
        target_center = (self.target_range[0] + self.target_range[1]) / 2
        transformed = power_expanded + target_center

        # 第四步：范围扩展
        range_center = np.mean(transformed)
        expanded = (transformed - range_center) * params.get('range_expansion', 1.0) + range_center

        return expanded

    def _smart_range_adjustment(self, predictions: np.ndarray) -> np.ndarray:
        """
        🔧 智能范围调整，避免硬截断造成的极化
        使用sigmoid映射替代硬截断
        """
        try:
            # 计算当前范围
            pred_min, pred_max = np.min(predictions), np.max(predictions)
            pred_range = pred_max - pred_min

            if pred_range < 1e-6:  # 几乎没有变化
                return np.full_like(predictions, 0.5)

            # 目标范围 [0.05, 0.95]
            target_min, target_max = 0.05, 0.95
            target_range = target_max - target_min

            # 如果已经在目标范围内，只需轻微调整
            if pred_min >= target_min and pred_max <= target_max:
                return predictions

            # 使用Sigmoid映射进行平滑转换
            # 先标准化到[-3, 3]范围（sigmoid在此范围内变化最明显）
            normalized = (predictions - np.mean(predictions)) / (np.std(predictions) + 1e-8)
            normalized = np.clip(normalized, -3, 3)  # 限制在sigmoid有效范围

            # 应用sigmoid函数
            sigmoid_values = 1 / (1 + np.exp(-normalized))

            # 映射到目标范围
            adjusted = target_min + sigmoid_values * target_range

            logger.info(f"   智能范围调整: [{pred_min:.3f}, {pred_max:.3f}] → [{np.min(adjusted):.3f}, {np.max(adjusted):.3f}]")

            return adjusted

        except Exception as e:
            logger.warning(f"智能范围调整失败: {e}，使用原始值")
            return np.clip(predictions, 0.05, 0.95)

    def _validate_transform_quality(self, original: np.ndarray, transformed: np.ndarray, method: str):
        """验证变换质量"""
        try:
            orig_std = np.std(original)
            trans_std = np.std(transformed)
            orig_range = np.max(original) - np.min(original)
            trans_range = np.max(transformed) - np.min(transformed)

            logger.info(f"   {method}变换效果:")
            logger.info(f"     标准差: {orig_std:.3f} → {trans_std:.3f} (目标>{self.target_std})")
            logger.info(f"     范围: {orig_range:.3f} → {trans_range:.3f}")

            # 评估是否达到目标
            std_achieved = trans_std >= self.target_std
            range_achieved = trans_range >= (self.target_range[1] - self.target_range[0]) * 0.7

            std_status = "✅" if std_achieved else "⚠️"
            range_status = "✅" if range_achieved else "⚠️"

            logger.info(f"     差异化目标: {std_status} {'达成' if std_achieved else '未达成'}")
            logger.info(f"     范围目标: {range_status} {'达成' if range_achieved else '未达成'}")

        except Exception as e:
            logger.warning(f"变换质量验证失败: {e}")

    def evaluate_all_methods(self, predictions: np.ndarray) -> pd.DataFrame:
        """
        评估所有变换方法

        Args:
            predictions: 原始预测值

        Returns:
            方法对比结果DataFrame
        """
        try:
            if not self.fitted:
                raise ValueError("后处理器尚未拟合")

            logger.info("📊 评估所有后处理方法...")

            methods = ['linear', 'quantile', 'beta', 'hybrid']
            results = []

            # 原始结果
            orig_std = np.std(predictions)
            orig_range = np.max(predictions) - np.min(predictions)
            results.append({
                'method': 'original',
                'std': orig_std,
                'range': orig_range,
                'std_improvement': 0.0,
                'target_achieved': orig_std >= self.target_std
            })

            # 各种方法
            for method in methods:
                try:
                    transformed = self.transform(predictions, method)
                    trans_std = np.std(transformed)
                    trans_range = np.max(transformed) - np.min(transformed)

                    results.append({
                        'method': method,
                        'std': trans_std,
                        'range': trans_range,
                        'std_improvement': (trans_std - orig_std) / orig_std * 100,
                        'target_achieved': trans_std >= self.target_std
                    })
                except Exception as e:
                    logger.warning(f"{method}方法评估失败: {e}")

            results_df = pd.DataFrame(results)

            # 显示结果
            logger.info("   方法对比结果:")
            for _, row in results_df.iterrows():
                status = "✅" if row['target_achieved'] else "❌"
                logger.info(f"     {row['method']:<10}: std={row['std']:.3f}, 改进={row['std_improvement']:+.1f}% {status}")

            return results_df

        except Exception as e:
            logger.error(f"❌ 方法评估失败: {e}")
            return pd.DataFrame()

    def get_best_method(self, predictions: np.ndarray) -> str:
        """
        获取最佳变换方法

        Args:
            predictions: 原始预测值

        Returns:
            最佳方法名称
        """
        try:
            results_df = self.evaluate_all_methods(predictions)

            if results_df.empty:
                return 'linear'

            # 筛选达到目标的方法
            achieved_methods = results_df[results_df['target_achieved'] == True]

            if not achieved_methods.empty:
                # 选择改进幅度最大的
                best_method = achieved_methods.loc[achieved_methods['std_improvement'].idxmax(), 'method']
            else:
                # 如果都没达到目标，选择最接近的
                best_method = results_df.loc[results_df['std'].idxmax(), 'method']

            logger.info(f"🎯 推荐最佳方法: {best_method}")
            return best_method

        except Exception as e:
            logger.error(f"❌ 最佳方法选择失败: {e}")
            return 'hybrid'

# 使用示例
if __name__ == "__main__":
    from level4_quality_meta_learner import Level4QualityMetaLearner

    # 加载模型和数据
    level4_model = Level4QualityMetaLearner()
    level4_model.load_model()

    X_train, y_train, X_val, y_val, X_test, y_test = level4_model.prepare_data(
        "level4_training_dataset_v2_train.csv",
        "level4_training_dataset_v2_val.csv",
        "level4_training_dataset_v2_test.csv"
    )

    # 获取原始预测
    y_pred_test = level4_model.predict_quality_score(X_test)

    print("🎯 Level 4 质量评分后处理优化")
    print(f"原始预测: mean={y_pred_test.mean():.3f}, std={y_pred_test.std():.3f}")

    # 创建后处理器
    postprocessor = Level4QualityPostprocessor(target_std=0.15, target_range=(0.1, 0.9))

    # 拟合变换参数
    postprocessor.fit_transform_parameters(y_pred_test, y_test)

    # 评估所有方法
    results_df = postprocessor.evaluate_all_methods(y_pred_test)

    # 获取最佳方法
    best_method = postprocessor.get_best_method(y_pred_test)

    # 应用最佳变换
    y_pred_optimized = postprocessor.transform(y_pred_test, best_method)

    print(f"\n✅ 优化后预测: mean={y_pred_optimized.mean():.3f}, std={y_pred_optimized.std():.3f}")
    print(f"   差异化改进: {(y_pred_optimized.std() - y_pred_test.std()) / y_pred_test.std() * 100:+.1f}%")
    print(f"   目标达成: {'✅' if y_pred_optimized.std() >= 0.15 else '❌'}")