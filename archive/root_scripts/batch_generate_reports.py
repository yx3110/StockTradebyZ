#!/usr/bin/env python3
"""
批量生成历史选股报告（优化版）
资源单例复用：模型加载1次、数据查询1次、选股器初始化1次
只有策略计算和报告生成按日期循环

支持 v3.9 和 v3.95 两个版本的批量报告生成
"""
import sys
import os
import sqlite3
import argparse
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import pandas as pd

# 预计算指标列名，每个日期循环前需要清除以触发重新计算
PRECOMPUTED_COLS = {'BBI', 'K', 'D', 'J', 'DIF', 'ZXDQ', 'ZXDKX', 'MA60'}

logger = logging.getLogger(__name__)


def get_trading_dates(start_date, end_date, db_path='data_adapter/stock_data.db'):
    """获取交易日列表"""
    conn = sqlite3.connect(db_path)
    dates = [r[0] for r in conn.execute("""
        SELECT DISTINCT trade_date FROM v39_feature_cache
        WHERE trade_date >= ? AND trade_date <= ?
        ORDER BY trade_date
    """, (start_date, end_date)).fetchall()]
    conn.close()
    return dates


def strip_precomputed_indicators(data: Dict[str, pd.DataFrame]):
    """
    清除预计算指标列，使 precompute_indicators 在下一个日期重新计算

    precompute_indicators 使用 "BBI" not in hist.columns 作为guard，
    如果不清除，第二天的指标不会重新计算。
    """
    for code, df in data.items():
        cols_to_drop = PRECOMPUTED_COLS & set(df.columns)
        if cols_to_drop:
            df.drop(columns=list(cols_to_drop), inplace=True)


def generate_report_batch(dates: List[str], version: str, force: bool = False, suffix: str = ''):
    """
    批量生成报告 — 资源单例复用模式

    优化要点:
    1. 行情数据、模型、特征缓存只加载一次
    2. 技术指标（BBI/KDJ/MACD等）在全量数据上一次性预计算，利用因果性
    3. V3.9/V3.95 共用统一的特征缓存格式 {date: DataFrame}

    Args:
        dates: 交易日列表
        version: 'v3.9' 或 'v3.95'
        force: 是否强制覆盖已有报告
        suffix: 报告目录后缀，如 'model20260222' -> reports/daily_selection_v3.9_model20260222/
    """
    from data_adapter.stock_data_loader import StockDataLoader
    from tomorrow_stock_selector import TomorrowStockSelector

    dir_name = f'daily_selection_{version}_{suffix}' if suffix else f'daily_selection_{version}'
    report_dir = Path(f'reports/{dir_name}')
    report_dir.mkdir(parents=True, exist_ok=True)

    # 过滤已有报告（除非 --force）
    dates_to_generate = []
    for date in dates:
        date_str = date.replace('-', '')
        report_file = report_dir / f'选股分析报告_{date_str}.md'
        if report_file.exists() and not force:
            continue
        dates_to_generate.append(date)

    if not dates_to_generate:
        print(f"  {version}: 所有 {len(dates)} 份报告已存在，跳过")
        return 0, 0

    print(f"  {version}: 需要生成 {len(dates_to_generate)}/{len(dates)} 份报告")

    # ========== 一次性资源加载 ==========
    t0 = time.time()

    # 1. 数据加载器
    data_loader = StockDataLoader()

    # 2. 一次性加载全量行情数据
    print(f"  加载行情数据 ({dates_to_generate[0]} ~ {dates_to_generate[-1]}, lookback=200)...")
    t_data = time.time()
    all_data = data_loader.load_all_stock_data_wide(
        start_date=dates_to_generate[0],
        end_date=dates_to_generate[-1],
        lookback_days=200,
        security_types=['A股']
    )
    print(f"  行情数据加载完成: {len(all_data)} 只股票, 耗时 {time.time()-t_data:.1f}秒")

    # 3. 证券基本信息
    securities_info = data_loader.load_securities_info()

    # 4. 创建评分器（每版本一次）
    print(f"  加载 {version} 模型...")
    t_model = time.time()
    if version == 'v3.9':
        from ml_models.v39.v390_production_scorer import V390ProductionScorer
        scorer = V390ProductionScorer()
    elif version == 'v5.0':
        from ml_models.v39.v500_production_scorer import V500ProductionScorer
        scorer = V500ProductionScorer(model_type='small_data')
    else:
        from ml_models.v39.v395_production_scorer import V395ProductionScorer
        scorer = V395ProductionScorer(model_type='small_data')
    print(f"  模型加载完成, 耗时 {time.time()-t_model:.1f}秒")

    # 5. 预加载全部日期的特征缓存（一条SQL，V3.9和V3.95共用格式）
    print(f"  预加载特征缓存 ({len(dates_to_generate)} 天)...")
    t_cache = time.time()
    preloaded_cache = scorer.preload_feature_cache(dates_to_generate)
    print(f"  特征缓存预加载完成, 耗时 {time.time()-t_cache:.1f}秒")

    # 6. 一次性预计算技术指标（BBI/KDJ/DIF/ZXDQ/ZXDKX/MA60）
    #    这些指标都是因果计算（只看过去数据），在全量数据上计算一次即可
    #    截断后的副本会自动继承这些列，run_selectors 中的 precompute_indicators 检测到列已存在会跳过
    print(f"  一次性预计算技术指标 ({len(all_data)} 只股票)...")
    t_ind = time.time()
    from stock_selctor.Selector import precompute_indicators
    last_date = pd.Timestamp(dates_to_generate[-1])
    precompute_indicators(all_data, last_date)
    print(f"  技术指标预计算完成, 耗时 {time.time()-t_ind:.1f}秒")

    # 7. 创建选股器（每版本一次，注入资源）
    selector = TomorrowStockSelector.create_for_batch(
        scoring_version=version,
        stocks_only=True,
        scorer=scorer,
        data_loader=data_loader,
        securities_info=securities_info
    )

    t_init = time.time() - t0
    print(f"  初始化完成, 总耗时 {t_init:.1f}秒")

    # ========== 按日期循环 ==========
    done = 0
    failed = 0

    for i, date in enumerate(dates_to_generate):
        t_day = time.time()

        # 进度和ETA
        elapsed = time.time() - t0 - t_init  # 不算初始化
        rate = done / elapsed if elapsed > 0 and done > 0 else 0
        remaining = len(dates_to_generate) - done
        eta = remaining / rate if rate > 0 else 0

        print(f"  [{done+1}/{len(dates_to_generate)}] {version} {date} (ETA: {eta/60:.0f}min)", end='', flush=True)

        try:
            latest_date = pd.Timestamp(date)

            # 不再 strip_precomputed_indicators — 技术指标已一次性预计算
            # run_selectors 内部的 precompute_indicators 会检测到列已存在并跳过

            # 运行 8 个选股策略
            results = selector.run_selectors(all_data, latest_date)

            # 收集所有被策略选中的股票
            all_picks = set()
            for strategy, picks in results.items():
                all_picks.update(picks)
            all_picks_list = list(all_picks)

            # 用预加载特征做 ML 评分（统一格式: day_features 是 DataFrame 或 None）
            day_features = preloaded_cache.get(date)
            batch_results = scorer.predict_scores_from_preloaded(
                all_picks_list, date, day_features
            )
            if version == 'v3.9':
                selector.v39_batch_cache = batch_results
                selector.v395_batch_cache = {}
            elif version == 'v5.0':
                selector.v500_batch_cache = batch_results
            else:
                selector.v395_batch_cache = batch_results
                selector.v39_batch_cache = {}

            # 分析结果（使用已预填充的缓存，跳过内部batch scoring的DB查询）
            analysis = selector.analyze_results(results, all_data, latest_date)

            # 生成报告
            report = selector.generate_report(analysis, latest_date)

            # 保存报告
            date_str = date.replace('-', '')
            report_file = report_dir / f'选股分析报告_{date_str}.md'
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)

            # 保存分析数据JSON
            json_file = report_dir / f'analysis_data_{date_str}.json'
            try:
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(analysis, f, ensure_ascii=False, indent=2, default=str)
            except Exception:
                pass

            # 清空批处理缓存，为下一天准备
            selector.v39_batch_cache = {}
            selector.v395_batch_cache = {}
            if hasattr(selector, 'v500_batch_cache'):
                selector.v500_batch_cache = {}

            elapsed_day = time.time() - t_day
            n_picks = len(all_picks_list)
            print(f" -> {n_picks}只, {elapsed_day:.1f}秒")
            done += 1

        except Exception as e:
            print(f" -> ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            done += 1

    return done, failed


def _detect_model_date(version: str) -> str:
    """从模型文件名提取训练日期，如 v390_full_system_20260222_002439.pkl -> 20260222"""
    import re
    model_dir = Path('ml_models/trained_models')
    if version == 'v3.9':
        # 在 v39/ 子目录下查找最新的 v390_full_system_YYYYMMDD_*.pkl
        for f in sorted((model_dir / 'v39').glob('v390_full_system_*.pkl'), reverse=True):
            m = re.search(r'(\d{8})', f.name)
            if m:
                return m.group(1)
    elif version == 'v3.95':
        # 在 v395/ 子目录下查找最新的 v395_multi_target_YYYYMMDD_*.pkl
        for f in sorted((model_dir / 'v395').glob('v395_multi_target_*.pkl'), reverse=True):
            m = re.search(r'(\d{8})', f.name)
            if m:
                return m.group(1)
    return ''


def main():
    parser = argparse.ArgumentParser(description='批量生成选股报告（优化版）')
    parser.add_argument('--start-date', default='2025-09-01', help='开始日期 YYYY-MM-DD')
    parser.add_argument('--end-date', default='2026-02-13', help='结束日期 YYYY-MM-DD')
    parser.add_argument('--version', default='v3.9', choices=['v3.9', 'v3.95', 'v5.0', 'both'],
                       help='评分版本')
    parser.add_argument('--force', action='store_true', help='强制覆盖已有报告')
    parser.add_argument('--suffix', default='', help='报告目录后缀，如 model20260222')
    parser.add_argument('--auto-suffix', action='store_true',
                       help='自动从模型文件名提取训练日期作为后缀')
    args = parser.parse_args()

    dates = get_trading_dates(args.start_date, args.end_date)
    if not dates:
        print(f"未找到 {args.start_date} ~ {args.end_date} 范围内的交易日数据")
        return

    versions = ['v3.9', 'v3.95'] if args.version == 'both' else [args.version]

    # 确定每个版本的后缀
    version_suffixes = {}
    if args.auto_suffix:
        for v in versions:
            model_date = _detect_model_date(v)
            version_suffixes[v] = f'model{model_date}' if model_date else ''
            if model_date:
                print(f"  {v} 模型训练日期: {model_date}")
    else:
        for v in versions:
            version_suffixes[v] = args.suffix

    print(f"批量报告生成（优化版）: {len(dates)} 个交易日, 版本: {versions}")
    print(f"日期范围: {dates[0]} ~ {dates[-1]}")
    for v in versions:
        sfx = version_suffixes[v]
        dir_name = f'daily_selection_{v}_{sfx}' if sfx else f'daily_selection_{v}'
        print(f"  {v} -> reports/{dir_name}/")
    print()

    start_time = time.time()
    total_done = 0
    total_failed = 0

    for version in versions:
        print(f"=== {version} ===")
        done, failed = generate_report_batch(dates, version, force=args.force, suffix=version_suffixes[version])
        total_done += done
        total_failed += failed
        print()

    elapsed = time.time() - start_time
    print(f"完成: {total_done} 报告, 失败: {total_failed}, 总耗时: {elapsed/60:.1f}分钟")
    if total_done > 0:
        print(f"平均: {elapsed/total_done:.1f}秒/报告")


if __name__ == '__main__':
    main()
