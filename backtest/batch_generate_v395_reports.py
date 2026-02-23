#!/usr/bin/env python3
"""
V3.95 快速批量报告生成器 (纯ML评分模式)

跳过8个量化策略、行情数据加载、技术指标预计算、Markdown报告生成，
只做ML评分并输出 analysis_data_YYYYMMDD.json，供 backtest_report_based.py 使用。

性能:
  - 旧方案 (subprocess): ~3.5分钟/天 = 112天约6.5小时
  - 中间方案 (batch_generate_reports.py): ~15秒/天 = 112天约28分钟
  - 本脚本 (纯ML评分): ~2秒/天 = 112天约4分钟 (不含初始化)

用法:
    # 生成报告到默认目录
    python3 backtest/batch_generate_v395_reports.py \\
        --start-date 2025-09-01 --end-date 2026-02-13

    # 生成到自定义目录
    python3 backtest/batch_generate_v395_reports.py \\
        --start-date 2025-09-01 --end-date 2026-02-13 \\
        --output-dir reports/daily_selection_v3.95_fast

    # 强制覆盖已有报告
    python3 backtest/batch_generate_v395_reports.py \\
        --start-date 2025-09-01 --end-date 2026-02-13 --force

    # 使用 v3.9 评分器
    python3 backtest/batch_generate_v395_reports.py \\
        --start-date 2025-09-01 --end-date 2026-02-13 --version v3.9

    # 同时输出 Markdown 概要
    python3 backtest/batch_generate_v395_reports.py \\
        --start-date 2025-09-01 --end-date 2026-02-13 --with-markdown
"""

import sys
import os
import json
import time
import sqlite3
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

DB_PATH = os.path.join(PROJECT_ROOT, 'data_adapter', 'stock_data.db')


def get_trading_dates(start_date: str, end_date: str) -> List[str]:
    """从 v39_feature_cache 获取交易日列表"""
    conn = sqlite3.connect(DB_PATH)
    dates = [r[0] for r in conn.execute("""
        SELECT DISTINCT trade_date FROM v39_feature_cache
        WHERE trade_date >= ? AND trade_date <= ?
        ORDER BY trade_date
    """, (start_date, end_date)).fetchall()]
    conn.close()
    return dates


def fast_preload_feature_cache(dates: List[str]) -> Dict[str, pd.DataFrame]:
    """
    高速批量预加载特征缓存 (替代 scorer.preload_feature_cache)

    优化方式:
    - pd.Series.apply(json.loads) 替代 iterrows (快3-5倍)
    - pd.json_normalize 批量展开 JSON (避免逐行 DataFrame 构造)
    - 分块加载避免内存峰值过高

    Args:
        dates: 日期列表

    Returns:
        {date: features_DataFrame} 含 code 列和所有特征+市场特征
    """
    result = {d: None for d in dates}
    if not dates:
        return result

    conn = sqlite3.connect(DB_PATH)

    # 分块加载: 每次最多50天，避免SQL太大 / 内存峰值过高
    CHUNK_SIZE = 50
    total_records = 0

    for chunk_start in range(0, len(dates), CHUNK_SIZE):
        chunk_dates = dates[chunk_start:chunk_start + CHUNK_SIZE]
        placeholders = ','.join(['?' for _ in chunk_dates])

        query = f"""
        SELECT code, trade_date, features_json,
               market_return_20d, market_return_10d, market_return_5d,
               market_volatility_20d, market_volatility_10d,
               market_up_ratio_20d, market_up_ratio_10d,
               market_drawdown_20d, market_volume_ratio,
               market_position_20d, market_momentum_20d, market_momentum_5d
        FROM v39_feature_cache
        WHERE trade_date IN ({placeholders})
        """
        df = pd.read_sql_query(query, conn, params=chunk_dates)

        if df.empty:
            continue

        # 向量化 JSON 解析: 比 iterrows 快 3-5 倍
        parsed = df['features_json'].apply(json.loads)
        features_all = pd.DataFrame(parsed.tolist())
        features_all['code'] = df['code'].values
        features_all['trade_date'] = df['trade_date'].values

        # 市场特征列
        market_cols = [c for c in df.columns if c.startswith('market_')]
        for col in market_cols:
            features_all[col] = df[col].values

        # 按日期分组
        for date, group in features_all.groupby('trade_date'):
            result[date] = group.drop(columns=['trade_date']).reset_index(drop=True)
            total_records += len(group)

    conn.close()
    return result


def load_securities_info() -> Dict[str, Dict]:
    """加载证券基本信息 (code -> {name, industry, ...})"""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT code, name, industry, area
        FROM securities
        WHERE type = 'A股'
    """).fetchall()
    conn.close()
    info = {}
    for code, name, industry, area in rows:
        info[code] = {
            'name': name or f'Stock_{code}',
            'industry': industry or '',
            'area': area or '',
        }
    return info


def preload_daily_basic_bulk(dates: List[str]) -> Dict[str, pd.DataFrame]:
    """
    批量预加载 daily_basic 数据 (pe_ttm, pb, ps_ttm, turnover_rate, circ_mv)

    分块查询避免 SQLite 变量数限制 (SQLITE_MAX_VARIABLE_NUMBER = 999)

    Returns:
        {date: DataFrame with columns [code, pe_ttm, pb, ps_ttm, turnover_rate, circ_mv]}
    """
    if not dates:
        return {}

    result = {}
    conn = sqlite3.connect(DB_PATH)
    CHUNK_SIZE = 50

    for chunk_start in range(0, len(dates), CHUNK_SIZE):
        chunk_dates = dates[chunk_start:chunk_start + CHUNK_SIZE]
        placeholders = ','.join(['?' for _ in chunk_dates])
        query = f"""
            SELECT s.code, db.trade_date, db.pe_ttm, db.pb, db.ps_ttm,
                   db.turnover_rate, db.circ_mv
            FROM daily_basic db
            JOIN securities s ON db.security_id = s.id
            WHERE db.trade_date IN ({placeholders})
        """
        df = pd.read_sql_query(query, conn, params=chunk_dates)

        if not df.empty:
            for date, group in df.groupby('trade_date'):
                result[date] = group.drop(columns=['trade_date']).reset_index(drop=True)

    conn.close()
    return result


def score_all_stocks_from_preloaded(
    scorer,
    features_df: Optional[pd.DataFrame],
    date: str,
    daily_basic_df: Optional[pd.DataFrame] = None,
    version: str = 'v3.95',
) -> Dict[str, Dict]:
    """
    对一天的全截面股票做ML评分。

    V3.95 路径: 直接操作预加载的 features_df，跳过所有SQL查询。
                支持 robust_zscore / cascade / dual_stream / rank_normalized / raw 五种路径。
    V3.9 路径:  使用 scorer.predict_scores_from_preloaded() 代理。

    Args:
        scorer: V395ProductionScorer 或 V390ProductionScorer 实例
        features_df: 预加载的特征 DataFrame (含 code 列及所有特征/市场特征)
        date: 交易日期 YYYY-MM-DD
        daily_basic_df: 预加载的 daily_basic DataFrame (含 code, pe_ttm, pb, ...)
        version: 'v3.95' 或 'v3.9'

    Returns:
        {code: {score, pred_3d, pred_5d, pred_10d}}
    """
    if features_df is None or len(features_df) == 0:
        return {}

    # V3.9: 使用 scorer 自带的 predict_scores_from_preloaded 方法
    # (V390 内部结构不同: base_models + meta_model, 非 models[target])
    if version == 'v3.9':
        all_codes = features_df['code'].tolist()
        results = scorer.predict_scores_from_preloaded(all_codes, date, features_df)
        # V390 返回格式: {code: {score, pred_3d, pred_5d, pred_10d}} 或
        # {code: {score, predicted_return, ...}}  - 需要标准化
        standardized = {}
        for code, data in results.items():
            standardized[code] = {
                'score': data.get('score', 50.0),
                'pred_3d': data.get('pred_3d', data.get('predicted_return', 0.0)),
                'pred_5d': data.get('pred_5d', data.get('predicted_return', 0.0)),
                'pred_10d': data.get('pred_10d', data.get('predicted_return', 0.0)),
            }
        return standardized

    # V3.95: 高效内联评分 (避免 predict_scores_from_preloaded 的额外开销)
    df = features_df.copy()

    # 五种截面归一化路径
    if getattr(scorer, 'robust_zscore', False):
        df = scorer._robust_zscore_normalize_features(df)
        # 使用预加载的 daily_basic 而非 SQL 查询
        if getattr(scorer, 'extra_features_from_daily_basic', None) and daily_basic_df is not None:
            df = _merge_daily_basic_features(df, daily_basic_df)
        elif getattr(scorer, 'extra_features_from_daily_basic', None):
            # 回退到 scorer 内置的 SQL 方法
            df = scorer._load_daily_basic_features(df, date)
    elif getattr(scorer, 'cascade', False) or getattr(scorer, 'rank_normalized', False):
        df = scorer._rank_normalize_features(df)
    elif getattr(scorer, 'dual_stream', False):
        df = scorer._create_dual_stream_features(df)

    # 准备特征矩阵
    exclude_cols = {'code', 'trade_date'}
    feature_cols = scorer.feature_cols
    if feature_cols:
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        available_cols = feature_cols
    else:
        available_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[available_cols].fillna(0).values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 模型预测
    model_predictions_success = False
    predictions = {
        '3d': np.zeros(len(X)),
        '5d': np.zeros(len(X)),
        '10d': np.zeros(len(X)),
    }

    if getattr(scorer, 'cascade', False):
        # 级联推理 3d -> 5d -> 10d
        if '3d' in scorer.models and scorer.models['3d']:
            w3d = scorer.weights.get('label_3d', scorer.weights.get('3d', {}))
            if isinstance(w3d, dict):
                predictions['3d'], ok = scorer._cascade_ensemble_predict(X, scorer.models['3d'], w3d)
                if ok:
                    model_predictions_success = True
        if '5d' in scorer.models and scorer.models['5d']:
            X_5d = np.column_stack([X, predictions['3d']])
            w5d = scorer.weights.get('label_5d', scorer.weights.get('5d', {}))
            if isinstance(w5d, dict):
                predictions['5d'], ok = scorer._cascade_ensemble_predict(X_5d, scorer.models['5d'], w5d)
                if ok:
                    model_predictions_success = True
        if '10d' in scorer.models and scorer.models['10d']:
            X_10d = np.column_stack([X, predictions['3d'], predictions['5d']])
            w10d = scorer.weights.get('label_10d', scorer.weights.get('10d', {}))
            if isinstance(w10d, dict):
                predictions['10d'], ok = scorer._cascade_ensemble_predict(X_10d, scorer.models['10d'], w10d)
                if ok:
                    model_predictions_success = True
    else:
        # 非级联: 独立预测各目标
        for target in ['3d', '5d', '10d']:
            if target not in scorer.models or not scorer.models[target]:
                continue
            target_pred = np.zeros(len(X))
            total_weight = 0
            success_count = 0
            for name, model in scorer.models[target].items():
                try:
                    pred = model.predict(X)
                    weight = scorer.weights.get(f'label_{target}', {}).get(name, 0.2)
                    target_pred += weight * pred
                    total_weight += weight
                    success_count += 1
                except Exception:
                    continue
            if total_weight > 0:
                target_pred /= total_weight
                predictions[target] = target_pred
                if success_count > 0:
                    model_predictions_success = True

    # 计算综合分数
    if model_predictions_success:
        tw = scorer.target_weights
        combined_pred = (
            tw['label_3d'] * predictions['3d'] +
            tw['label_5d'] * predictions['5d'] +
            tw['label_10d'] * predictions['10d']
        )
    else:
        combined_pred = scorer._calculate_fallback_scores(df, available_cols)
        predictions = scorer._estimate_predictions_from_features(df, available_cols)

    # 百分位排名 -> 30~90 分
    if len(combined_pred) > 1:
        from scipy import stats as sp_stats
        ranks = sp_stats.rankdata(combined_pred)
        percentiles = (ranks - 1) / (len(ranks) - 1) * 100
        scores = 30 + percentiles * 0.6
    else:
        scores = np.array([60.0])

    # 构建结果字典
    codes = df['code'].tolist()
    results = {}
    for i, code in enumerate(codes):
        results[code] = {
            'score': float(scores[i]),
            'pred_3d': float(predictions['3d'][i]) if i < len(predictions['3d']) else 0.0,
            'pred_5d': float(predictions['5d'][i]) if i < len(predictions['5d']) else 0.0,
            'pred_10d': float(predictions['10d'][i]) if i < len(predictions['10d']) else 0.0,
        }

    return results


def _merge_daily_basic_features(features_df: pd.DataFrame,
                                 daily_basic_df: pd.DataFrame) -> pd.DataFrame:
    """将预加载的 daily_basic 合并到特征 DataFrame"""
    if daily_basic_df is None or len(daily_basic_df) == 0:
        for col in ['pe_ttm', 'pb', 'ps_ttm', 'turnover_rate', 'log_market_cap']:
            features_df[col] = 0.0
        return features_df

    features_df = features_df.merge(daily_basic_df, on='code', how='left')
    features_df['log_market_cap'] = np.log1p(features_df['circ_mv'].fillna(0))
    features_df.drop(columns=['circ_mv'], inplace=True, errors='ignore')
    for col in ['pe_ttm', 'pb', 'ps_ttm', 'turnover_rate', 'log_market_cap']:
        if col in features_df.columns:
            features_df[col] = features_df[col].fillna(features_df[col].median())
    return features_df


def build_analysis_json(
    scored_stocks: Dict[str, Dict],
    date: str,
    securities_info: Dict[str, Dict],
    version: str = 'v3.95',
) -> dict:
    """
    构建 backtest_report_based.py load_reports() 兼容的 JSON 结构

    最小必需字段:
      all_stocks_with_scores: [{stock_code, score, predicted_return_5d,
                                strategies, selected_by_strategies}, ...]
    """
    all_stocks = []
    for code, data in scored_stocks.items():
        info = securities_info.get(code, {})
        all_stocks.append({
            'stock_code': code,
            'stock_name': info.get('name', f'Stock_{code}'),
            'industry': info.get('industry', ''),
            'score': round(data['score'], 2),
            'predicted_return_5d': round(data['pred_5d'], 6),
            'pred_3d': round(data['pred_3d'], 6),
            'pred_5d': round(data['pred_5d'], 6),
            'pred_10d': round(data['pred_10d'], 6),
            'selected_by_strategies': 1,
            'strategies': ['ML_Score'],
            'analysis_date': date,
        })

    # 按分数降序排列
    all_stocks.sort(key=lambda x: x['score'], reverse=True)

    return {
        'analysis_date': date,
        'scoring_version': version,
        'generation_mode': 'batch_fast_ml_only',
        'total_scored_stocks': len(all_stocks),
        'all_stocks_with_scores': all_stocks,
    }


def build_markdown_summary(analysis: dict, date: str) -> str:
    """生成简要 Markdown 报告 (可选)"""
    stocks = analysis.get('all_stocks_with_scores', [])
    date_str = date.replace('-', '')

    lines = [
        f"# V3.95 选股评分报告 {date}",
        f"",
        f"*批量快速生成 (纯ML评分模式) | {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        f"",
        f"## 评分概览",
        f"",
        f"- 评分股票数: {len(stocks)}",
    ]

    if stocks:
        scores = [s['score'] for s in stocks]
        lines.extend([
            f"- 最高分: {max(scores):.1f}",
            f"- 最低分: {min(scores):.1f}",
            f"- 中位数: {np.median(scores):.1f}",
            f"- >=85分: {sum(1 for s in scores if s >= 85)}",
            f"- >=80分: {sum(1 for s in scores if s >= 80)}",
        ])

    # Top 30 股票列表
    top_n = min(30, len(stocks))
    lines.extend([
        f"",
        f"## Top {top_n} 评分股票",
        f"",
        f"| 排名 | 代码 | 名称 | 行业 | 评分 | 3日预测 | 5日预测 | 10日预测 |",
        f"|:----:|:----:|:----:|:----:|:----:|:-------:|:-------:|:--------:|",
    ])

    for i, s in enumerate(stocks[:top_n]):
        lines.append(
            f"| {i+1} | {s['stock_code']} | {s['stock_name']} | {s.get('industry', '')} "
            f"| {s['score']:.1f} "
            f"| {s.get('pred_3d', 0)*100:+.2f}% "
            f"| {s.get('pred_5d', 0)*100:+.2f}% "
            f"| {s.get('pred_10d', 0)*100:+.2f}% |"
        )

    lines.append("")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='V3.95 快速批量报告生成器 (纯ML评分模式)')
    parser.add_argument('--start-date', default='2025-09-01',
                        help='开始日期 YYYY-MM-DD (default: 2025-09-01)')
    parser.add_argument('--end-date', default='2026-02-13',
                        help='结束日期 YYYY-MM-DD (default: 2026-02-13)')
    parser.add_argument('--output-dir', default=None,
                        help='输出目录 (default: reports/daily_selection_v{version}_fast)')
    parser.add_argument('--version', default='v3.95',
                        choices=['v3.9', 'v3.95'],
                        help='评分版本 (default: v3.95)')
    parser.add_argument('--force', action='store_true',
                        help='强制覆盖已有报告')
    parser.add_argument('--with-markdown', action='store_true',
                        help='同时生成 Markdown 概要报告')
    parser.add_argument('--suffix', default='',
                        help='报告目录后缀 (e.g. "robust_zscore")')
    args = parser.parse_args()

    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        dir_name = f'daily_selection_{args.version}_fast'
        if args.suffix:
            dir_name = f'daily_selection_{args.version}_{args.suffix}'
        output_dir = Path('reports') / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"V3.95 快速批量报告生成器")
    print(f"  版本:     {args.version}")
    print(f"  日期范围: {args.start_date} ~ {args.end_date}")
    print(f"  输出目录: {output_dir}")
    print()

    # ========== 1. 获取交易日列表 ==========
    t0 = time.time()
    dates = get_trading_dates(args.start_date, args.end_date)
    if not dates:
        print(f"未找到 {args.start_date} ~ {args.end_date} 范围内的交易日数据")
        return
    print(f"[1/5] 交易日: {len(dates)} 天 ({dates[0]} ~ {dates[-1]})")

    # 过滤已有报告
    if not args.force:
        existing = set()
        for f in output_dir.glob('analysis_data_*.json'):
            date_str = f.stem.replace('analysis_data_', '')
            if len(date_str) == 8:
                existing.add(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")
        dates_to_generate = [d for d in dates if d not in existing]
        if existing:
            print(f"      已有报告: {len(existing)}, 跳过")
    else:
        dates_to_generate = dates

    if not dates_to_generate:
        print(f"所有 {len(dates)} 份报告已存在，无需生成")
        return

    print(f"      待生成: {len(dates_to_generate)} 天")

    # ========== 2. 加载模型 ==========
    t_model = time.time()
    if args.version == 'v3.9':
        from ml_models.v39.v390_production_scorer import V390ProductionScorer
        scorer = V390ProductionScorer()
    else:
        from ml_models.v39.v395_production_scorer import V395ProductionScorer
        scorer = V395ProductionScorer(model_type='small_data')
    print(f"[2/5] 模型加载完成 ({time.time()-t_model:.1f}秒)")

    # ========== 3. 预加载特征缓存 (高速模式) ==========
    t_cache = time.time()
    preloaded_cache = fast_preload_feature_cache(dates_to_generate)
    n_records = sum(len(v) for v in preloaded_cache.values() if v is not None)
    print(f"[3/5] 特征缓存预加载完成: {n_records} 条记录 ({time.time()-t_cache:.1f}秒)")

    # ========== 4. 预加载 daily_basic (for robust_zscore) ==========
    daily_basic_cache = {}
    if getattr(scorer, 'robust_zscore', False) and getattr(scorer, 'extra_features_from_daily_basic', None):
        t_basic = time.time()
        daily_basic_cache = preload_daily_basic_bulk(dates_to_generate)
        n_basic = sum(len(v) for v in daily_basic_cache.values())
        print(f"[4/5] daily_basic 预加载完成: {n_basic} 条记录 ({time.time()-t_basic:.1f}秒)")
    else:
        print(f"[4/5] daily_basic 预加载: 跳过 (模型不需要)")

    # ========== 5. 加载证券信息 ==========
    t_info = time.time()
    securities_info = load_securities_info()
    print(f"[5/5] 证券信息加载完成: {len(securities_info)} 只 ({time.time()-t_info:.1f}秒)")

    t_init = time.time() - t0
    print(f"\n初始化完成, 总耗时: {t_init:.1f}秒")
    print(f"{'='*60}")

    # ========== 按日期循环评分 ==========
    done = 0
    failed = 0
    t_loop = time.time()

    for i, date in enumerate(dates_to_generate):
        t_day = time.time()

        try:
            features_df = preloaded_cache.get(date)
            daily_basic_df = daily_basic_cache.get(date)

            # 全截面评分
            scored_stocks = score_all_stocks_from_preloaded(
                scorer, features_df, date, daily_basic_df, version=args.version
            )

            if not scored_stocks:
                print(f"  [{i+1}/{len(dates_to_generate)}] {date}: 无特征数据，跳过")
                failed += 1
                continue

            # 构建 JSON
            analysis = build_analysis_json(scored_stocks, date, securities_info,
                                           version=args.version)

            # 保存 JSON
            date_str = date.replace('-', '')
            json_file = output_dir / f'analysis_data_{date_str}.json'
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2, default=str)

            # 可选: 保存 Markdown
            if args.with_markdown:
                md_content = build_markdown_summary(analysis, date)
                md_file = output_dir / f'选股分析报告_{date_str}.md'
                with open(md_file, 'w', encoding='utf-8') as f:
                    f.write(md_content)

            elapsed_day = time.time() - t_day
            elapsed_total = time.time() - t_loop
            rate = (done + 1) / elapsed_total if elapsed_total > 0 else 1
            remaining = len(dates_to_generate) - done - 1
            eta = remaining / rate if rate > 0 else 0

            n_stocks = len(scored_stocks)
            top_score = max(s['score'] for s in scored_stocks.values())
            print(f"  [{i+1}/{len(dates_to_generate)}] {date}: "
                  f"{n_stocks}只, top={top_score:.1f}, "
                  f"{elapsed_day:.2f}秒 (ETA: {eta:.0f}秒)")
            done += 1

        except Exception as e:
            print(f"  [{i+1}/{len(dates_to_generate)}] {date}: ERROR - {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # ========== 汇总 ==========
    total_time = time.time() - t0
    loop_time = time.time() - t_loop

    print(f"\n{'='*60}")
    print(f"批量报告生成完成!")
    print(f"  成功: {done}, 失败: {failed}")
    print(f"  总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    print(f"  评分耗时: {loop_time:.1f}秒 ({loop_time/60:.1f}分钟)")
    if done > 0:
        print(f"  平均: {loop_time/done:.2f}秒/天")
    print(f"  输出目录: {output_dir}")
    print(f"  报告数量: {len(list(output_dir.glob('analysis_data_*.json')))}")


if __name__ == '__main__':
    main()
