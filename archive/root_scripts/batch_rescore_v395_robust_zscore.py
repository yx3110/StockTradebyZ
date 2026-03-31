#!/usr/bin/env python3
"""
批量用新v3.95 Robust Z-Score + Industry-Excess模型重新评分
复用已有报告的策略选股结果，只替换ML分数
"""
import json
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml_models.v39.v395_production_scorer import V395ProductionScorer


def main():
    # 源目录(已有策略选股+旧模型分数) → 目标目录(新模型分数)
    src_dir = PROJECT_ROOT / 'reports' / 'daily_selection_v3.95_model20260221'
    dst_dir = PROJECT_ROOT / 'reports' / 'daily_selection_v3.95_robust_zscore'
    dst_dir.mkdir(parents=True, exist_ok=True)

    # 加载新的Robust Z-Score模型 (自动选最新pkl)
    print("加载V3.95 Robust Z-Score模型...")
    scorer = V395ProductionScorer(model_type='small_data')
    print(f"  robust_zscore: {scorer.robust_zscore}")
    print(f"  extra_features: {scorer.extra_features_from_daily_basic}")
    print(f"  cascade: {scorer.cascade}")
    print(f"  rank_normalized: {scorer.rank_normalized}")
    print(f"  feature_cols count: {len(scorer.feature_cols) if scorer.feature_cols else 0}")

    if not scorer.robust_zscore:
        print("ERROR: 加载的模型不是robust_zscore类型! 请检查模型文件。")
        return

    # 收集所有日期和股票代码
    json_files = sorted(src_dir.glob('analysis_data_*.json'))
    print(f"找到 {len(json_files)} 份报告")

    # 预加载所有日期的特征缓存
    dates = []
    for f in json_files:
        date_str = f.stem.replace('analysis_data_', '')
        date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        dates.append(date)

    print(f"预加载 {len(dates)} 天特征缓存...")
    cache = scorer.preload_feature_cache(dates)

    # 逐日重新评分
    success = 0
    for json_file in json_files:
        date_str = json_file.stem.replace('analysis_data_', '')
        date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        stocks = data.get('all_stocks_with_scores', [])
        if not stocks:
            continue

        # 提取股票代码
        codes = [s['stock_code'] for s in stocks if s.get('stock_code')]

        # 用新模型评分 (使用预加载缓存)
        features_df = cache.get(date)
        results = scorer.predict_scores_from_preloaded(codes, date, features_df)

        # 更新分数
        for s in stocks:
            code = s.get('stock_code', '')
            if code in results:
                r = results[code]
                s['score'] = r['score']
                s['predicted_return_5d'] = r['pred_5d']
                s['pred_5d'] = r['pred_5d']
                if 'detailed_scoring' in s:
                    s['detailed_scoring']['final_score'] = r['score']
                    s['detailed_scoring']['pred_3d'] = r['pred_3d']
                    s['detailed_scoring']['pred_5d'] = r['pred_5d']
                    s['detailed_scoring']['pred_10d'] = r['pred_10d']
                    s['detailed_scoring']['scoring_method'] = 'V3.95_RobustZScore'
                if 'factor_scores' in s:
                    s['factor_scores']['predicted_return_5d'] = r['pred_5d']

        # 按新分数重新排序
        stocks.sort(key=lambda x: x.get('score', 0), reverse=True)
        data['all_stocks_with_scores'] = stocks

        # 保存到目标目录
        dst_json = dst_dir / json_file.name
        with open(dst_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # 复制对应的md文件(如果存在)
        md_src = src_dir / f"选股分析报告_{date_str}.md"
        if md_src.exists():
            import shutil
            shutil.copy2(md_src, dst_dir / md_src.name)

        success += 1
        if success % 20 == 0:
            print(f"  已完成 {success}/{len(json_files)} ...")

    print(f"\n完成! 成功重新评分 {success}/{len(json_files)} 份报告")
    print(f"输出目录: {dst_dir}")


if __name__ == '__main__':
    main()
