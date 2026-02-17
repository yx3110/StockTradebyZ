#!/usr/bin/env python3
"""
验证Level 4质量后处理器的完整功能
"""

import sys
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# 添加路径以便导入模块
sys.path.append(str(Path(__file__).parent))

from level4_quality_meta_learner import Level4QualityMetaLearner

def main():
    print("=== Level 4 Quality 后处理器验证 ===\n")

    # 1. 加载训练好的模型
    print("1. 加载训练好的模型...")
    learner = Level4QualityMetaLearner()
    learner.load_model("models/level4_quality_meta_learner.pkl")

    # 2. 加载后处理器
    print("2. 加载后处理器...")
    with open("models/level4_quality_postprocessor.pkl", 'rb') as f:
        postprocessor_data = pickle.load(f)

    postprocessor = postprocessor_data['postprocessor']
    best_method = postprocessor_data['best_method']
    target_std = postprocessor_data['target_std']

    print(f"   最佳方法: {best_method}")
    print(f"   目标标准差: {target_std}")

    # 3. 加载测试数据
    print("3. 加载测试数据...")
    test_data = pd.read_csv("level4_training_dataset_test.csv")
    feature_cols = [col for col in test_data.columns if col.startswith('feature_')]
    X_test = test_data[feature_cols]

    print(f"   测试集样本数: {len(test_data)}")
    print(f"   特征数量: {len(feature_cols)}")

    # 4. 完整的预测和后处理流程
    print("4. 完整预测流程...")

    # 4.1 原始模型预测
    raw_predictions = learner.predict_quality_score(X_test)
    print(f"   原始预测范围: [{np.min(raw_predictions):.4f}, {np.max(raw_predictions):.4f}]")
    print(f"   原始预测std: {np.std(raw_predictions):.4f}")

    # 4.2 应用后处理
    processed_predictions = postprocessor.transform(raw_predictions, method=best_method)
    print(f"   处理后范围: [{np.min(processed_predictions):.4f}, {np.max(processed_predictions):.4f}]")
    print(f"   处理后std: {np.std(processed_predictions):.4f}")

    # 5. 验证差异化效果
    print("5. 差异化效果验证:")
    improvement_ratio = np.std(processed_predictions) / np.std(raw_predictions)
    print(f"   标准差提升倍数: {improvement_ratio:.2f}x")
    print(f"   是否达到目标std>=0.15: {'✅' if np.std(processed_predictions) >= 0.15 else '❌'}")

    # 6. 分位数分析
    print("6. 分布分析:")
    raw_percentiles = np.percentile(raw_predictions, [10, 25, 50, 75, 90])
    processed_percentiles = np.percentile(processed_predictions, [10, 25, 50, 75, 90])

    print("   原始预测分位数:")
    print(f"     P10: {raw_percentiles[0]:.4f}, P25: {raw_percentiles[1]:.4f}")
    print(f"     P50: {raw_percentiles[2]:.4f}, P75: {raw_percentiles[3]:.4f}")
    print(f"     P90: {raw_percentiles[4]:.4f}")

    print("   处理后分位数:")
    print(f"     P10: {processed_percentiles[0]:.4f}, P25: {processed_percentiles[1]:.4f}")
    print(f"     P50: {processed_percentiles[2]:.4f}, P75: {processed_percentiles[3]:.4f}")
    print(f"     P90: {processed_percentiles[4]:.4f}")

    # 7. 示例股票质量评分
    print("7. 示例股票质量评分:")
    for i in [0, 100, 200, 300, 400]:
        if i < len(test_data):
            stock_code = test_data.iloc[i]['stock_code']
            stock_name = test_data.iloc[i]['stock_name']
            raw_score = raw_predictions[i]
            processed_score = processed_predictions[i]
            print(f"   {stock_code} {stock_name}: {raw_score:.4f} → {processed_score:.4f}")

    # 8. 保存完整的验证结果
    print("8. 保存验证结果...")
    validation_results = {
        'raw_predictions_stats': {
            'mean': float(np.mean(raw_predictions)),
            'std': float(np.std(raw_predictions)),
            'min': float(np.min(raw_predictions)),
            'max': float(np.max(raw_predictions)),
            'percentiles': [float(p) for p in raw_percentiles]
        },
        'processed_predictions_stats': {
            'mean': float(np.mean(processed_predictions)),
            'std': float(np.std(processed_predictions)),
            'min': float(np.min(processed_predictions)),
            'max': float(np.max(processed_predictions)),
            'percentiles': [float(p) for p in processed_percentiles]
        },
        'improvement_metrics': {
            'std_improvement_ratio': float(improvement_ratio),
            'target_std_achieved': bool(np.std(processed_predictions) >= 0.15),
            'best_method': best_method
        },
        'sample_predictions': [
            {
                'stock_code': str(test_data.iloc[i]['stock_code']),
                'stock_name': str(test_data.iloc[i]['stock_name']),
                'raw_score': float(raw_predictions[i]),
                'processed_score': float(processed_predictions[i])
            }
            for i in [0, 100, 200, 300, 400] if i < len(test_data)
        ]
    }

    import json
    with open("models/level4_postprocessor_validation.json", 'w', encoding='utf-8') as f:
        json.dump(validation_results, f, indent=2, ensure_ascii=False)

    print("   验证结果已保存到 models/level4_postprocessor_validation.json")

    print("\n=== 验证完成 ===")

    # 总结
    if np.std(processed_predictions) >= 0.15:
        print(f"✅ 成功: Level 4质量后处理器工作正常")
        print(f"   差异化目标达成: std={np.std(processed_predictions):.4f} >= 0.15")
        print(f"   改进倍数: {improvement_ratio:.2f}x")
    else:
        print(f"❌ 失败: 后处理器未能达到差异化目标")

if __name__ == "__main__":
    main()