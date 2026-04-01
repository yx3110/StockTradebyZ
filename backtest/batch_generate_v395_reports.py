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


def get_trading_dates(start_date: str, end_date: str, version: str = 'v3.95') -> List[str]:
    """从特征缓存获取交易日列表"""
    table = 'alpha158_feature_cache' if version == 'alpha158' else 'v39_feature_cache'
    conn = sqlite3.connect(DB_PATH)
    dates = [r[0] for r in conn.execute(f"""
        SELECT DISTINCT trade_date FROM {table}
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


def fast_preload_alpha158_cache(dates: List[str]) -> Dict[str, pd.DataFrame]:
    """高速批量预加载 Alpha158 特征缓存"""
    result = {d: None for d in dates}
    if not dates:
        return result

    conn = sqlite3.connect(DB_PATH)
    CHUNK_SIZE = 50
    total_records = 0

    for chunk_start in range(0, len(dates), CHUNK_SIZE):
        chunk_dates = dates[chunk_start:chunk_start + CHUNK_SIZE]
        placeholders = ','.join(['?' for _ in chunk_dates])

        query = f"""
        SELECT code, trade_date, features_json
        FROM alpha158_feature_cache
        WHERE trade_date IN ({placeholders})
        """
        df = pd.read_sql_query(query, conn, params=chunk_dates)

        if df.empty:
            continue

        parsed = df['features_json'].apply(json.loads)
        features_all = pd.DataFrame(parsed.tolist())
        features_all['code'] = df['code'].values
        features_all['trade_date'] = df['trade_date'].values

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

    # V4.8.3/V4.8.4/V4.8.5/V4.8.6: 使用 predict_scores() (需加载 brain_alpha_cache)
    if version in ('v4.8.3', 'v4.8.4', 'v4.8.5', 'v4.8.6', 'v4.8.7', 'v4.9.0', 'v4.9.0.1', 'v4.9.0.2', 'v4.9.1', 'v4.9.2'):
        all_codes = features_df['code'].tolist()
        return scorer.predict_scores(all_codes, date)

    # V3.9: 返回格式需要标准化 (V390 base_models + meta_model)
    if version == 'v3.9':
        all_codes = features_df['code'].tolist()
        results = scorer.predict_scores_from_preloaded(all_codes, date, features_df)
        standardized = {}
        for code, data in results.items():
            standardized[code] = {
                'score': data.get('score', 50.0),
                'pred_3d': data.get('pred_3d', data.get('predicted_return', 0.0)),
                'pred_5d': data.get('pred_5d', data.get('predicted_return', 0.0)),
                'pred_10d': data.get('pred_10d', data.get('predicted_return', 0.0)),
            }
        return standardized

    # 统一路径: alpha158, v4.4.2~v4.8.2 全部使用 predict_scores_from_preloaded
    # (各版本差异在scorer内部处理，无需外部分支)
    if version != 'v3.95':
        all_codes = features_df['code'].tolist()
        return scorer.predict_scores_from_preloaded(all_codes, date, features_df)

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

    # 模型预测 (动态支持3d/5d/10d/15d)
    model_predictions_success = False
    all_targets = sorted(scorer.models.keys()) if hasattr(scorer, 'models') else ['3d', '5d', '10d']
    predictions = {t: np.zeros(len(X)) for t in all_targets}

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
        # 非级联: 独立预测各目标 (含15d)
        for target in all_targets:
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

    # 计算综合分数 (动态target_weights)
    if model_predictions_success:
        tw = scorer.target_weights
        combined_pred = np.zeros(len(X))
        for t in all_targets:
            w = tw.get(f'label_{t}', 0)
            if w > 0 and t in predictions:
                combined_pred += w * predictions[t]
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
        p10 = float(predictions['10d'][i]) if i < len(predictions['10d']) else 0.0
        p15 = float(predictions['15d'][i]) if '15d' in predictions and i < len(predictions['15d']) else 0.0
        results[code] = {
            'score': float(scores[i]),
            'pred_3d': float(predictions['3d'][i]) if i < len(predictions['3d']) else 0.0,
            'pred_5d': float(predictions['5d'][i]) if i < len(predictions['5d']) else 0.0,
            'pred_10d': p10,
            'pred_15d': p15,
            'rank_score': 0.6 * p10 + 0.4 * p15,
        }

    return results


def bulk_preload_scorer_caches(scorer, dates: List[str]):
    """
    批量预加载 scorer 内部缓存，避免逐日 SQL 查询。

    预填充: _financial_cache, _micro_cache, _tech_feature_cache,
            _exec_cache, _next_trade_date_cache, _market_return_cache,
            _daily_basic_extra_cache (new)

    Speedup: ~450 per-date SQL → ~6 bulk SQL
    """
    if not dates:
        return

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout = 30000")

    # ── 0. 全交易日索引 (用于 next_trade_date) ──
    min_date = min(dates)
    all_td = [r[0] for r in conn.execute(
        "SELECT DISTINCT trade_date FROM daily_quotes WHERE trade_date >= ? ORDER BY trade_date",
        (min_date,)).fetchall()]
    td_idx = {d: i for i, d in enumerate(all_td)}

    # Pre-fill _next_trade_date_cache
    if hasattr(scorer, '_next_trade_date_cache'):
        import bisect
        for d in dates:
            if d in scorer._next_trade_date_cache:
                continue
            idx = bisect.bisect_right(all_td, d)
            scorer._next_trade_date_cache[d] = all_td[idx] if idx < len(all_td) else None

    # ── 1. Financial features (roe等) ──
    financial_cols = getattr(scorer, 'extra_features_financial', [])
    if financial_cols and hasattr(scorer, '_financial_cache'):
        t0 = time.time()
        uncached = [d for d in dates if d not in scorer._financial_cache]
        if uncached:
            select_cols = ', '.join([f'fi.{col}' for col in financial_cols])
            # 一次性加载所有财务数据 (按security_id+ann_date), 然后用pandas做"最近公告"逻辑
            query = f"""
            SELECT s.code, fi.security_id, fi.ann_date, fi.id, {select_cols}
            FROM financial_indicator fi
            JOIN securities s ON fi.security_id = s.id
            WHERE fi.ann_date IS NOT NULL AND fi.ann_date != ''
              AND fi.ann_date <= ?
            """
            df_fi_all = pd.read_sql_query(query, conn, params=[max(uncached)])
            if not df_fi_all.empty:
                # 确保 ann_date 是字符串类型 (DB中可能混合int/str)
                df_fi_all['ann_date'] = df_fi_all['ann_date'].astype(str)
                # 匹配原始SQL: MAX(id) per security_id WHERE ann_date <= date
                for date in uncached:
                    valid = df_fi_all[df_fi_all['ann_date'] <= date]
                    if len(valid) > 0:
                        # MAX(id) per security_id (与原始SQL一致)
                        idx = valid.groupby('security_id')['id'].idxmax()
                        latest = valid.loc[idx]
                        scorer._financial_cache[date] = latest[['code'] + financial_cols].reset_index(drop=True)
                    else:
                        scorer._financial_cache[date] = pd.DataFrame(columns=['code'] + financial_cols)
            else:
                for date in uncached:
                    scorer._financial_cache[date] = pd.DataFrame(columns=['code'] + financial_cols)
        print(f"  [bulk] Financial cache: {len(dates)} days ({time.time()-t0:.1f}s)")

    # ── 2. Daily basic extra (dv_ttm, turnover_rate_f, float_ratio) ──
    if hasattr(scorer, '_load_daily_basic_extra'):
        t0 = time.time()
        # 创建缓存字典 (scorer没有内置的)
        if not hasattr(scorer, '_daily_basic_extra_cache'):
            scorer._daily_basic_extra_cache = {}
        CHUNK = 100
        all_extra_dates = [d for d in dates if d not in scorer._daily_basic_extra_cache]
        for ci in range(0, len(all_extra_dates), CHUNK):
            chunk = all_extra_dates[ci:ci + CHUNK]
            ph = ','.join(['?' for _ in chunk])
            query = f"""
            SELECT s.code, db.trade_date, db.dv_ttm, db.turnover_rate_f, db.circ_mv, db.total_mv
            FROM daily_basic db
            JOIN securities s ON db.security_id = s.id
            WHERE db.trade_date IN ({ph})
            """
            df = pd.read_sql_query(query, conn, params=chunk)
            if not df.empty:
                for date, grp in df.groupby('trade_date'):
                    g = grp.drop(columns=['trade_date']).copy()
                    g['float_ratio'] = g['circ_mv'] / g['total_mv'].clip(lower=1e-8)
                    g.drop(columns=['circ_mv', 'total_mv'], inplace=True)
                    scorer._daily_basic_extra_cache[date] = g
            for d in chunk:
                if d not in scorer._daily_basic_extra_cache:
                    scorer._daily_basic_extra_cache[d] = pd.DataFrame()
        print(f"  [bulk] Daily basic extra: {len(dates)} days ({time.time()-t0:.1f}s)")

    # ── 3. Technical features ──
    tech_features = getattr(scorer, 'extra_tech_features', None)
    if tech_features and hasattr(scorer, '_tech_feature_cache'):
        t0 = time.time()
        all_tech_dates = [d for d in dates if d not in scorer._tech_feature_cache]
        CHUNK = 100
        for ci in range(0, len(all_tech_dates), CHUNK):
            chunk = all_tech_dates[ci:ci + CHUNK]
            ph = ','.join(['?' for _ in chunk])
            query = f"""
            SELECT s.code, ti.trade_date,
                   ti.kdj_k, ti.kdj_j, ti.macd_dif, ti.macd_dea, ti.macd_macd,
                   ti.boll_upper, ti.boll_lower, ti.atr_14,
                   q.close, q.high, q.low
            FROM technical_indicators ti
            JOIN securities s ON ti.security_id = s.id
            JOIN daily_quotes q ON q.security_id = s.id AND q.trade_date = ti.trade_date
            WHERE ti.trade_date IN ({ph})
            """
            df = pd.read_sql_query(query, conn, params=chunk)
            if not df.empty:
                for date, grp in df.groupby('trade_date'):
                    scorer._tech_feature_cache[date] = grp.drop(columns=['trade_date']).reset_index(drop=True)
            for d in chunk:
                if d not in scorer._tech_feature_cache:
                    scorer._tech_feature_cache[d] = pd.DataFrame()
        print(f"  [bulk] Tech features: {len(dates)} days ({time.time()-t0:.1f}s)")

    # ── 4. Executability data ──
    if hasattr(scorer, '_exec_cache'):
        t0 = time.time()
        all_exec_dates = [d for d in dates if d not in scorer._exec_cache]
        # 需要 T日 + T+1日 的数据
        all_needed_dates = set()
        date_to_next = {}
        for d in all_exec_dates:
            all_needed_dates.add(d)
            nd = scorer._next_trade_date_cache.get(d)
            if nd:
                all_needed_dates.add(nd)
                date_to_next[d] = nd
            else:
                date_to_next[d] = d
        needed_sorted = sorted(all_needed_dates)

        if needed_sorted:
            CHUNK = 100
            # 批量加载涨跌幅和换手率
            pct_data = {}  # {trade_date: {code: pct}}
            tr_data = {}   # {trade_date: {code: turnover_rate}}
            for ci in range(0, len(needed_sorted), CHUNK):
                chunk = needed_sorted[ci:ci + CHUNK]
                ph = ','.join(['?' for _ in chunk])
                query = f"""
                SELECT s.code, q.trade_date, q.price_change_pct
                FROM daily_quotes q
                JOIN securities s ON q.security_id = s.id
                WHERE s.type = 'A股' AND q.trade_date IN ({ph})
                """
                df = pd.read_sql_query(query, conn, params=chunk)
                for td, grp in df.groupby('trade_date'):
                    pct_data[td] = dict(zip(grp['code'], grp['price_change_pct']))

                query2 = f"""
                SELECT s.code, db.trade_date, db.turnover_rate
                FROM daily_basic db
                JOIN securities s ON db.security_id = s.id
                WHERE db.trade_date IN ({ph})
                """
                df2 = pd.read_sql_query(query2, conn, params=chunk)
                for td, grp in df2.groupby('trade_date'):
                    tr_data[td] = dict(zip(grp['code'], grp['turnover_rate']))

            # 组装 exec_cache
            all_codes_query = [r[0] for r in conn.execute(
                "SELECT code FROM securities WHERE type = 'A股'").fetchall()]
            for d in all_exec_dates:
                nd = date_to_next[d]
                records = []
                for code in all_codes_query:
                    records.append({
                        'code': code,
                        'pct_t': pct_data.get(d, {}).get(code),
                        'pct_t1': pct_data.get(nd, {}).get(code),
                        'turnover_rate': tr_data.get(d, {}).get(code),
                    })
                scorer._exec_cache[d] = pd.DataFrame(records)
        print(f"  [bulk] Executability: {len(all_exec_dates)} days ({time.time()-t0:.1f}s)")

    # ── 5. Microstructure features (最重的：40天滑动窗口) ──
    micro_cols = (getattr(scorer, 'extra_features_microstructure', []) +
                  getattr(scorer, 'extra_features_reversal', []) +
                  getattr(scorer, 'extra_features_risk', []))
    if micro_cols and hasattr(scorer, '_micro_cache'):
        t0 = time.time()
        all_micro_dates = sorted([d for d in dates if d not in scorer._micro_cache])
        if all_micro_dates:
            # 计算需要的日期范围 (最早日期 -60 天 到 最晚日期)
            earliest = all_micro_dates[0]
            latest = all_micro_dates[-1]
            query = """
            SELECT s.code, q.trade_date, q.close, q.volume, q.price_change_pct
            FROM daily_quotes q
            JOIN securities s ON q.security_id = s.id
            WHERE s.type = 'A股' AND q.trade_date >= date(?, '-60 days') AND q.trade_date <= ?
            ORDER BY s.code, q.trade_date
            """
            df_all = pd.read_sql_query(query, conn, params=[earliest, latest])
            print(f"  [bulk] Microstructure raw data: {len(df_all)} rows ({time.time()-t0:.1f}s)")

            if not df_all.empty:
                t1 = time.time()
                # 预处理: 转换类型
                df_all['close'] = pd.to_numeric(df_all['close'], errors='coerce')
                df_all['volume'] = pd.to_numeric(df_all['volume'], errors='coerce')
                df_all['price_change_pct'] = pd.to_numeric(df_all['price_change_pct'], errors='coerce').fillna(0)

                # 按 code 分组，一次性计算所有日期
                has_micro = bool(getattr(scorer, 'extra_features_microstructure', []))
                has_reversal = bool(getattr(scorer, 'extra_features_reversal', []))
                has_risk = bool(getattr(scorer, 'extra_features_risk', []))

                # 建立日期集合和交易日列表
                target_dates_set = set(all_micro_dates)
                all_trade_dates_in_data = sorted(df_all['trade_date'].unique())
                trade_date_positions = {d: i for i, d in enumerate(all_trade_dates_in_data)}

                # 按日期预分配结果
                micro_results = {d: [] for d in all_micro_dates}

                # 预计算每个target date的40日历天窗口起始日期
                from datetime import datetime as _dt, timedelta as _td
                target_window_start = {}
                for d in all_micro_dates:
                    dt_obj = _dt.strptime(d, '%Y-%m-%d')
                    target_window_start[d] = (dt_obj - _td(days=40)).strftime('%Y-%m-%d')

                for code, grp in df_all.groupby('code'):
                    grp = grp.sort_values('trade_date')
                    if len(grp) < 5:
                        continue
                    close_arr = grp['close'].values.astype(float)
                    vol_arr = grp['volume'].values.astype(float)
                    pct_arr = grp['price_change_pct'].values.astype(float)
                    grp_dates = grp['trade_date'].values

                    for j, td in enumerate(grp_dates):
                        if td not in target_dates_set:
                            continue
                        # 40日历天窗口 (匹配SQL: date(?, '-40 days'))
                        win_start = target_window_start[td]
                        start_j = j
                        while start_j > 0 and grp_dates[start_j - 1] >= win_start:
                            start_j -= 1
                        close = close_arr[start_j:j + 1]
                        volume = vol_arr[start_j:j + 1]
                        pct = pct_arr[start_j:j + 1]

                        if len(close) < 5:
                            continue

                        row = {'code': code}

                        if has_micro:
                            abs_ret = np.abs(pct[-20:]) if len(pct) >= 20 else np.abs(pct)
                            vol_safe = np.where(volume[-20:] > 0, volume[-20:], 1e-8) if len(volume) >= 20 else np.where(volume > 0, volume, 1e-8)
                            row['amihud_illiquidity'] = float(np.mean(abs_ret / vol_safe))

                            n = min(10, len(close))
                            if n >= 5:
                                corr = np.corrcoef(close[-n:], volume[-n:])[0, 1]
                                row['volume_price_corr_10d'] = float(corr) if not np.isnan(corr) else 0.0
                            else:
                                row['volume_price_corr_10d'] = 0.0

                            n_dd = min(20, len(close))
                            window = close[-n_dd:]
                            running_max = np.maximum.accumulate(window)
                            dd = (window - running_max) / np.where(running_max > 0, running_max, 1e-8)
                            row['max_drawdown_20d'] = float(np.min(dd))

                            n_ud = min(10, len(pct))
                            up_vol = np.sum(volume[-n_ud:][pct[-n_ud:] > 0])
                            dn_vol = np.sum(volume[-n_ud:][pct[-n_ud:] < 0])
                            row['updown_volume_asymmetry'] = float(up_vol / max(dn_vol, 1e-8))

                        if has_reversal:
                            row['return_1d'] = float(close[-1] / close[-2] - 1) if len(close) >= 2 else 0.0
                            row['return_3d'] = float(close[-1] / close[-4] - 1) if len(close) >= 4 else 0.0

                        if has_risk:
                            n_risk = min(20, len(close))
                            daily_ret = np.diff(close[-n_risk:]) / close[-n_risk:-1]
                            if len(daily_ret) >= 5:
                                demeaned = daily_ret - np.mean(daily_ret)
                                row['idio_volatility_20d'] = float(np.std(demeaned))
                            else:
                                row['idio_volatility_20d'] = 0.0

                        micro_results[td].append(row)

                for d in all_micro_dates:
                    scorer._micro_cache[d] = pd.DataFrame(micro_results[d]) if micro_results[d] else pd.DataFrame()
                print(f"  [bulk] Microstructure compute: {len(all_micro_dates)} days ({time.time()-t1:.1f}s)")

    # ── 6. Market return 20d ──
    if hasattr(scorer, '_market_return_cache'):
        t0 = time.time()
        query = """
        SELECT q.trade_date, q.close
        FROM daily_quotes q
        JOIN securities s ON q.security_id = s.id
        WHERE s.code = '000300.SH' AND q.trade_date >= date(?, '-45 days') AND q.trade_date <= ?
        ORDER BY q.trade_date
        """
        df_mkt = pd.read_sql_query(query, conn, params=[min_date, max(dates)])
        if not df_mkt.empty:
            mkt_close = df_mkt['close'].values.astype(float)
            mkt_dates = df_mkt['trade_date'].values
            mkt_idx = {d: i for i, d in enumerate(mkt_dates)}
            for d in dates:
                if d in scorer._market_return_cache:
                    continue
                if d in mkt_idx:
                    i = mkt_idx[d]
                    if i >= 20:
                        ret = mkt_close[i] / mkt_close[i - 20] - 1
                        scorer._market_return_cache[d] = float(ret)
                    else:
                        scorer._market_return_cache[d] = None
                else:
                    scorer._market_return_cache[d] = None
        print(f"  [bulk] Market return: {len(dates)} days ({time.time()-t0:.1f}s)")

    conn.close()


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
        entry = {
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
        }
        # V4.6+: save pred_15d and rank_score for correct composite/threshold
        if 'pred_15d' in data:
            entry['pred_15d'] = round(data['pred_15d'], 6)
        if 'rank_score' in data:
            entry['rank_score'] = round(data['rank_score'], 6)
        if 'head_rank' in data:
            entry['head_rank'] = data['head_rank']
        if 'q95_pred_10d' in data:
            entry['q95_pred_10d'] = round(data['q95_pred_10d'], 6)
        if 'in_head_pool' in data:
            entry['in_head_pool'] = data['in_head_pool']
        if 'gate_confidence' in data:
            entry['gate_confidence'] = round(data['gate_confidence'], 4)
        if 'gate_regime' in data:
            entry['gate_regime'] = data['gate_regime']
        all_stocks.append(entry)

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
    version = analysis.get('scoring_version', 'V3.95')
    date_str = date.replace('-', '')

    # 过滤: 只保留A股 (排除ETF/基金/指数, 代码6位纯数字 + .SZ/.SH)
    a_stocks = [s for s in stocks
                if s.get('stock_name', '').startswith('Stock_') is False
                and s.get('industry', '') != ''
                and len(s.get('stock_code', '')) <= 6]

    # 按 pred_10d 排序 (连续值, 不受全局百分位离散化影响)
    a_stocks.sort(key=lambda x: float(x.get('pred_10d', 0) or 0), reverse=True)

    lines = [
        f"# {version} 选股评分报告 {date}",
        f"",
        f"*批量快速生成 (纯ML评分模式) | {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        f"",
        f"## 评分概览",
        f"",
        f"- 评分A股数: {len(a_stocks)} (总评分: {len(stocks)})",
    ]

    if a_stocks:
        preds = [float(s.get('pred_10d', 0) or 0) for s in a_stocks]
        lines.extend([
            f"- Top-10 平均10d预测: {np.mean(preds[:10])*100:+.2f}%",
            f"- Top-50 平均10d预测: {np.mean(preds[:50])*100:+.2f}%",
            f"- 全市场中位10d预测: {np.median(preds)*100:+.2f}%",
        ])

    # Top 30 A股 (按 pred_10d)
    top_n = min(30, len(a_stocks))
    lines.extend([
        f"",
        f"## Top {top_n} A股 (按10日预测收益排名)",
        f"",
        f"| 排名 | 代码 | 名称 | 行业 | 评分 | 3日预测 | 5日预测 | 10日预测 | 15日预测 |",
        f"|:----:|:----:|:----:|:----:|:----:|:-------:|:-------:|:--------:|:--------:|",
    ])

    for i, s in enumerate(a_stocks[:top_n]):
        p3 = float(s.get('pred_3d', 0) or 0) * 100
        p5 = float(s.get('pred_5d', 0) or 0) * 100
        p10 = float(s.get('pred_10d', 0) or 0) * 100
        p15 = float(s.get('pred_15d', 0) or 0) * 100
        lines.append(
            f"| {i+1} | {s['stock_code']} | {s.get('stock_name', '')} | {s.get('industry', '')} "
            f"| {s['score']:.1f} "
            f"| {p3:+.2f}% "
            f"| {p5:+.2f}% "
            f"| {p10:+.2f}% "
            f"| {p15:+.2f}% |"
        )

    lines.append("")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='V3.95 快速批量报告生成器 (纯ML评分模式)')
    parser.add_argument('--start-date', default='auto',
                        help='开始日期 YYYY-MM-DD (default: auto, 从特征缓存检测)')
    parser.add_argument('--end-date', default='auto',
                        help='结束日期 YYYY-MM-DD (default: auto, 从特征缓存检测)')
    parser.add_argument('--output-dir', default=None,
                        help='输出目录 (default: reports/daily_selection_v{version}_fast)')
    parser.add_argument('--version', default='v3.95',
                        choices=['v3.9', 'v3.95', 'v3.96', 'v4.3', 'v4.4', 'v4.4.2', 'v4.6', 'v4.7', 'v4.7.1', 'v4.7.2', 'v4.7.3', 'v4.7.4', 'v4.7.5', 'v4.7.6', 'v4.7.7', 'v4.7.8', 'v4.7.9', 'v4.8.0', 'v4.8.1', 'v4.8.2', 'v4.8.3', 'v4.8.4', 'v4.8.5', 'v4.8.6', 'v4.8.7', 'v4.8.8', 'v4.9.0', 'v4.9.0.1', 'v4.9.0.2', 'v4.9.1', 'v4.9.2', 'v5.0', 'alpha158'],
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

    # auto 日期解析: 从特征缓存检测可用范围
    start_date = args.start_date
    end_date = args.end_date
    if start_date == 'auto' or end_date == 'auto':
        table = 'alpha158_feature_cache' if args.version == 'alpha158' else 'v39_feature_cache'
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(f"""
            SELECT MIN(trade_date), MAX(trade_date), COUNT(DISTINCT trade_date)
            FROM {table}
        """).fetchone()
        conn.close()
        if row and row[0]:
            if start_date == 'auto':
                start_date = row[0]
            if end_date == 'auto':
                end_date = row[1]
            print(f"  📅 自动检测特征缓存日期: {row[0]} → {row[1]} ({row[2]} 天)")
        else:
            print("  ⚠️ 无法从特征缓存检测日期, 请手动指定 --start-date / --end-date")
            return

    print(f"V3.95 快速批量报告生成器")
    print(f"  版本:     {args.version}")
    print(f"  日期范围: {start_date} ~ {end_date}")
    print(f"  输出目录: {output_dir}")
    print()

    # ========== 1. 获取交易日列表 ==========
    t0 = time.time()
    dates = get_trading_dates(start_date, end_date, version=args.version)
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
    if args.version == 'alpha158':
        from ml_models.v39.alpha158_production_scorer import Alpha158ProductionScorer
        scorer = Alpha158ProductionScorer()
    elif args.version == 'v3.9':
        from ml_models.v39.v390_production_scorer import V390ProductionScorer
        scorer = V390ProductionScorer()
    elif args.version == 'v3.96':
        from ml_models.v39.v396_production_scorer import V396ProductionScorer
        scorer = V396ProductionScorer(model_type='small_data')
    elif args.version == 'v4.9.2':
        from ml_models.v39.v492_production_scorer import V492ProductionScorer
        scorer = V492ProductionScorer(model_type='small_data', smooth_alpha=0.6)
    elif args.version == 'v4.9.1':
        from ml_models.v39.v491_production_scorer import V491ProductionScorer
        scorer = V491ProductionScorer(model_type='small_data')
    elif args.version == 'v4.9.0.2':
        from ml_models.v39.v4902_production_scorer import V4902ProductionScorer
        scorer = V4902ProductionScorer(model_type='small_data')
    elif args.version == 'v4.9.0.1':
        from ml_models.v39.v4901_production_scorer import V4901ProductionScorer
        scorer = V4901ProductionScorer(model_type='small_data')
    elif args.version == 'v4.9.0':
        from ml_models.v39.v490_production_scorer import V490ProductionScorer
        scorer = V490ProductionScorer(model_type='small_data')
    elif args.version == 'v4.8.8':
        from ml_models.v39.v488_production_scorer import V488ProductionScorer
        scorer = V488ProductionScorer()
    elif args.version == 'v4.8.7':
        from ml_models.v39.v487_production_scorer import V487ProductionScorer
        scorer = V487ProductionScorer()
    elif args.version == 'v4.8.6':
        from ml_models.v39.v486_production_scorer import V486ProductionScorer
        scorer = V486ProductionScorer()
    elif args.version == 'v4.8.5':
        from ml_models.v39.v485_production_scorer import V485ProductionScorer
        scorer = V485ProductionScorer()
    elif args.version == 'v4.8.4':
        from ml_models.v39.v484_production_scorer import V484ProductionScorer
        scorer = V484ProductionScorer()
    elif args.version == 'v4.8.3':
        from ml_models.v39.v483_production_scorer import V483ProductionScorer
        scorer = V483ProductionScorer()
    elif args.version == 'v4.8.2':
        from ml_models.v39.v482_production_scorer import V482ProductionScorer
        scorer = V482ProductionScorer()
    elif args.version == 'v4.8.1':
        from ml_models.v39.v481_production_scorer import V481ProductionScorer
        scorer = V481ProductionScorer()
    elif args.version == 'v4.8.0':
        from ml_models.v39.v480_production_scorer import V480ProductionScorer
        scorer = V480ProductionScorer()
    elif args.version == 'v4.7.9':
        from ml_models.v39.v479_production_scorer import V479ProductionScorer
        scorer = V479ProductionScorer()
    elif args.version == 'v4.7.8':
        from ml_models.v39.v478_production_scorer import V478ProductionScorer
        scorer = V478ProductionScorer()
    elif args.version == 'v4.7.7':
        from ml_models.v39.v477_production_scorer import V477ProductionScorer
        scorer = V477ProductionScorer()
    elif args.version == 'v4.7.6':
        from ml_models.v39.v476_production_scorer import V476ProductionScorer
        scorer = V476ProductionScorer()
    elif args.version == 'v4.7.5':
        from ml_models.v39.v475_production_scorer import V475ProductionScorer
        scorer = V475ProductionScorer()
    elif args.version == 'v4.7.4':
        from ml_models.v39.v474_production_scorer import V474ProductionScorer
        scorer = V474ProductionScorer()
    elif args.version == 'v4.7.3':
        from ml_models.v39.v473_production_scorer import V473ProductionScorer
        scorer = V473ProductionScorer()
    elif args.version == 'v4.7.2':
        from ml_models.v39.v472_production_scorer import V472ProductionScorer
        scorer = V472ProductionScorer()
    elif args.version == 'v4.7.1':
        from ml_models.v39.v471_production_scorer import V471ProductionScorer
        scorer = V471ProductionScorer()
    elif args.version == 'v4.7':
        from ml_models.v39.v47_production_scorer import V47ProductionScorer
        scorer = V47ProductionScorer()
    elif args.version == 'v4.6':
        from ml_models.v39.v46_production_scorer import V46ProductionScorer
        scorer = V46ProductionScorer()
    elif args.version == 'v4.4.2':
        from ml_models.v39.v44_production_scorer import V442ProductionScorer
        scorer = V442ProductionScorer()
    elif args.version == 'v4.4':
        from ml_models.v39.v44_production_scorer import V44ProductionScorer
        scorer = V44ProductionScorer()
    elif args.version == 'v4.3':
        from ml_models.v39.v43_production_scorer import V43ProductionScorer
        scorer = V43ProductionScorer()
    elif args.version == 'v5.0':
        from ml_models.v39.v500_production_scorer import V500ProductionScorer
        scorer = V500ProductionScorer()
    else:
        from ml_models.v39.v395_production_scorer import V395ProductionScorer
        scorer = V395ProductionScorer(model_type='small_data')
    print(f"[2/5] 模型加载完成 ({time.time()-t_model:.1f}秒)")

    # ========== 3. 预加载特征缓存 (高速模式) ==========
    t_cache = time.time()
    if args.version == 'alpha158':
        preloaded_cache = fast_preload_alpha158_cache(dates_to_generate)
    else:
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

    # ========== 5.5. 批量预加载 scorer 内部缓存 (避免逐日SQL) ==========
    if hasattr(scorer, '_micro_cache') or hasattr(scorer, '_financial_cache'):
        t_bulk = time.time()
        bulk_preload_scorer_caches(scorer, dates_to_generate)
        # Monkey-patch _load_daily_basic_extra 使用批量缓存
        if hasattr(scorer, '_daily_basic_extra_cache'):
            _orig_load_daily_basic_extra = scorer._load_daily_basic_extra
            def _patched_load_daily_basic_extra(features_df, date):
                cache = scorer._daily_basic_extra_cache
                if date in cache and len(cache[date]) > 0:
                    df_extra = cache[date]
                    features_df = features_df.merge(df_extra, on='code', how='left')
                    for col in ['dv_ttm', 'turnover_rate_f', 'float_ratio']:
                        if col in features_df.columns:
                            median_val = features_df[col].median()
                            features_df[col] = features_df[col].fillna(
                                median_val if not pd.isna(median_val) else 0.0)
                        else:
                            features_df[col] = 0.0
                    return features_df
                return _orig_load_daily_basic_extra(features_df, date)
            scorer._load_daily_basic_extra = _patched_load_daily_basic_extra
        print(f"[5.5] Scorer缓存批量预加载完成 ({time.time()-t_bulk:.1f}秒)")
    else:
        print(f"[5.5] Scorer缓存批量预加载: 跳过 (scorer不支持)")

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
