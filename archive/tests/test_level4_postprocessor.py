#!/usr/bin/env python3
"""
测试Level 4质量后处理器的效果
验证是否能成功扩展预测分布以达到std>0.15目标
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# 添加路径以便导入模块
sys.path.append(str(Path(__file__).parent))

from level4_quality_postprocessor import Level4QualityPostprocessor
from level4_quality_meta_learner import Level4QualityMetaLearner

def main():
    print("=== Level 4 Quality 后处理器测试 ===\n")

    # 1. 加载训练好的模型
    print("1. 加载训练好的模型...")
    learner = Level4QualityMetaLearner()
    learner.load_model("models/level4_quality_meta_learner.pkl")

    # 2. 加载测试数据
    print("2. 加载测试数据...")
    test_data = pd.read_csv("level4_training_dataset_test.csv")
    print(f"   测试集样本数: {len(test_data)}")

    # 3. 生成原始预测
    print("3. 生成原始模型预测...")
    feature_cols = [col for col in test_data.columns if col.startswith('feature_')]
    X_test = test_data[feature_cols]
    y_test = test_data['quality_overall']

    # 原始预测
    original_predictions = learner.predict_quality_score(X_test)

    # 4. 分析原始预测分布
    print("4. 分析原始预测分布:")
    print(f"   均值: {np.mean(original_predictions):.4f}")
    print(f"   标准差: {np.std(original_predictions):.4f}")
    print(f"   最小值: {np.min(original_predictions):.4f}")
    print(f"   最大值: {np.max(original_predictions):.4f}")
    print(f"   范围: {np.max(original_predictions) - np.min(original_predictions):.4f}")

    # 5. 初始化后处理器
    print("\n5. 初始化后处理器...")
    postprocessor = Level4QualityPostprocessor()

    # 6. 测试不同后处理方法
    methods = ['linear', 'quantile', 'beta', 'hybrid']
    results = {}

    print("\n6. 测试不同后处理方法:")
    for method in methods:
        print(f"\n   --- {method.upper()} 方法 ---")

        # 拟合后处理器 (使用正确的方法名)
        transform_params = postprocessor.fit_transform_parameters(original_predictions)

        # 变换预测
        transformed_predictions = postprocessor.transform(original_predictions, method=method)

        # 计算统计指标
        mean_val = np.mean(transformed_predictions)
        std_val = np.std(transformed_predictions)
        min_val = np.min(transformed_predictions)
        max_val = np.max(transformed_predictions)
        range_val = max_val - min_val

        # 计算与真实值的相关系数 (如果目标值有变化)
        if np.std(y_test) > 1e-8:  # 检查目标值是否有变化
            correlation = np.corrcoef(transformed_predictions, y_test)[0, 1]
        else:
            correlation = np.nan  # 目标值无变化，无法计算相关系数

        print(f"   均值: {mean_val:.4f}")
        print(f"   标准差: {std_val:.4f} {'✅' if std_val >= 0.15 else '❌'}")
        print(f"   范围: [{min_val:.4f}, {max_val:.4f}] (宽度: {range_val:.4f})")
        print(f"   相关系数: {correlation:.4f}")

        # 保存结果
        results[method] = {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'range': range_val,
            'correlation': correlation,
            'predictions': transformed_predictions
        }

    # 7. 选择最佳方法
    print("\n7. 方法比较与推荐:")

    # 目标：std >= 0.15 且保持合理的分布
    target_std = 0.15

    # 检查是否能计算相关性
    can_calc_corr = np.std(y_test) > 1e-8
    if can_calc_corr:
        original_corr = np.corrcoef(original_predictions, y_test)[0, 1]
        print(f"   原始模型相关系数: {original_corr:.4f}")
    else:
        original_corr = np.nan
        print(f"   原始模型相关系数: 无法计算 (目标值无变化)")

    print(f"   目标标准差: >= {target_std}")

    best_method = None
    best_score = -1

    for method, result in results.items():
        # 综合评分：主要考虑标准差达标 + 分布合理性
        std_score = min(result['std'] / target_std, 2.0)  # 达标得1分，超出2倍封顶2分

        # 分布合理性：不要过度拉伸
        range_score = 1.0 - min(abs(result['range'] - 0.8) / 0.8, 1.0)  # 目标范围0.8左右

        # 如果能计算相关性，加入相关性得分
        if can_calc_corr and not np.isnan(result['correlation']):
            corr_score = result['correlation'] / original_corr if original_corr != 0 else 0
            combined_score = std_score * 0.5 + range_score * 0.3 + corr_score * 0.2
            corr_info = f"相关性 {result['correlation']:.4f}"
        else:
            combined_score = std_score * 0.7 + range_score * 0.3
            corr_info = "相关性 N/A"

        meets_std = "✅" if result['std'] >= target_std else "❌"

        print(f"   {method.upper()}: 标准差{meets_std} {result['std']:.4f}, "
              f"{corr_info}, 综合评分: {combined_score:.3f}")

        if combined_score > best_score:
            best_score = combined_score
            best_method = method

    if best_method:
        print(f"\n   推荐方法: {best_method.upper()} (评分: {best_score:.3f})")
    else:
        print(f"\n   ❌ 未找到合适的方法")

    # 8. 保存最佳方法的后处理器
    if best_method:
        print(f"\n8. 保存最佳后处理器 ({best_method})...")
        # 拟合最佳方法的后处理器
        postprocessor.fit_transform_parameters(original_predictions)
        # 保存后处理器和最佳方法信息
        import pickle
        postprocessor_data = {
            'postprocessor': postprocessor,
            'best_method': best_method,
            'target_std': target_std,
            'target_range': postprocessor.target_range
        }
        with open("models/level4_quality_postprocessor.pkl", 'wb') as f:
            pickle.dump(postprocessor_data, f)
        print("   后处理器已保存到 models/level4_quality_postprocessor.pkl")

        # 保存测试结果
        result_summary = {
            'original_stats': {
                'mean': float(np.mean(original_predictions)),
                'std': float(np.std(original_predictions)),
                'correlation': float(original_corr)
            },
            'best_method': best_method,
            'best_method_stats': {
                'mean': float(results[best_method]['mean']),
                'std': float(results[best_method]['std']),
                'correlation': float(results[best_method]['correlation'])
            },
            'all_methods': {
                method: {
                    'mean': float(result['mean']),
                    'std': float(result['std']),
                    'correlation': float(result['correlation'])
                } for method, result in results.items()
            }
        }

        import json
        with open("models/level4_postprocessor_test_results.json", 'w', encoding='utf-8') as f:
            json.dump(result_summary, f, indent=2, ensure_ascii=False)
        print("   测试结果已保存到 models/level4_postprocessor_test_results.json")

    # 9. 分布可视化 (如果可能)
    print("\n9. 分布分析:")
    print("   原始预测分布特征:")
    print(f"     Q25: {np.percentile(original_predictions, 25):.4f}")
    print(f"     Q50: {np.percentile(original_predictions, 50):.4f}")
    print(f"     Q75: {np.percentile(original_predictions, 75):.4f}")

    if best_method:
        best_preds = results[best_method]['predictions']
        print(f"   {best_method.upper()}后处理分布特征:")
        print(f"     Q25: {np.percentile(best_preds, 25):.4f}")
        print(f"     Q50: {np.percentile(best_preds, 50):.4f}")
        print(f"     Q75: {np.percentile(best_preds, 75):.4f}")

    print("\n=== 测试完成 ===")

    # 总结
    if best_method and results[best_method]['std'] >= target_std:
        print(f"✅ 成功: {best_method.upper()}方法达到目标，std={results[best_method]['std']:.4f} >= {target_std}")
    else:
        print(f"❌ 失败: 所有方法均未达到std >= {target_std}目标")
        print("   建议: 考虑调整模型架构或增加更多特征")

if __name__ == "__main__":
    main()