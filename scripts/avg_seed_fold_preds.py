#!/usr/bin/env python3
"""将多个 seed 的 WF fold-pred 报告按股票取均值, 拼成 ensemble fold-pred 目录.

用途: 3-seed ensemble 的真·WF-OOS 评估 (G3 gate) — 每个 seed 的 fold 模型只见过
test 窗口之前的数据, 逐股平均 pred/score 后即为 ensemble 的零泄漏 OOS 预测.

用法:
  python3 scripts/avg_seed_fold_preds.py \
      --seed-dirs reports/a/seed42 reports/b/seed123 reports/c/seed456 \
      --output-dir reports/daily_selection_ng101_3seed_wf_oos
"""
import argparse
import json
from pathlib import Path

AVG_FIELDS = ['pred_3d', 'pred_5d', 'pred_10d', 'pred_15d', 'score', 'rank_score']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed-dirs', nargs='+', required=True)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()

    seed_dirs = [Path(d) for d in args.seed_dirs]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 只处理所有 seed 都有的日期 (fold 布局相同时应完全一致)
    date_sets = [set(p.name for p in d.glob('analysis_data_*.json')) for d in seed_dirs]
    common = sorted(set.intersection(*date_sets))
    skipped_dates = set.union(*date_sets) - set(common)
    if skipped_dates:
        print(f'WARN: {len(skipped_dates)} 个日期并非所有 seed 都有, 跳过: {sorted(skipped_dates)[:3]}...')

    n_stock_mismatch = 0
    for fname in common:
        per_seed = []
        for d in seed_dirs:
            data = json.loads((d / fname).read_text())
            per_seed.append({s['stock_code']: s for s in data['all_stocks_with_scores']})
        codes = set.intersection(*(set(m) for m in per_seed))
        union = set.union(*(set(m) for m in per_seed))
        if codes != union:
            n_stock_mismatch += 1

        merged = []
        for code in codes:
            rows = [m[code] for m in per_seed]
            avg = dict(rows[0])  # 保留首 seed 的非数值字段 (industry 等)
            for f in AVG_FIELDS:
                vals = [float(r.get(f) or 0.0) for r in rows]
                avg[f] = sum(vals) / len(vals)
            merged.append(avg)
        merged.sort(key=lambda s: -s['score'])

        base = json.loads((seed_dirs[0] / fname).read_text())
        base['all_stocks_with_scores'] = merged
        (out_dir / fname).write_text(json.dumps(base, ensure_ascii=False))

    print(f'完成: {len(common)} 天 → {out_dir} (股票集不一致的日期: {n_stock_mismatch})')


if __name__ == '__main__':
    main()
