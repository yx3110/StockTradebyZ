#!/usr/bin/env python3
"""
V4.8.5 可行性评估: board_type 特征 + ETF 训练数据对 ICIR 影响

4组对比:
  A) 基线: A股 only, 无 board_type (当前 V4.8.4)
  B) A股 + board_type
  C) A股 + ETF, 无 board_type
  D) A股 + ETF + board_type

快速评估: 单窗口 train/test split, LightGBM 10d 目标
"""

import sys
import numpy as np
import pandas as pd
import sqlite3
import json
import logging
from pathlib import Path
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = PROJECT_ROOT / 'data_adapter' / 'stock_data.db'


def derive_board_type(code: str) -> int:
    """从股票代码(无交易所后缀)推导板块类型
    0=沪市主板(600/601/603/605), 1=深市主板(000/001/003),
    2=中小板(002), 3=创业板(300/301), 4=科创板(688/689),
    5=北交所(8xx/9xx/4xx), 6=ETF(510/511/512/513/515/516/517/518/159/16x/56x/588)
    """
    c3 = code[:3]

    # ETF
    if c3 in ('510', '511', '512', '513', '515', '516', '517', '518',
              '560', '561', '562', '563', '588',
              '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169'):
        return 6

    # 科创板
    if c3 in ('688', '689'):
        return 4

    # 创业板
    if c3 in ('300', '301'):
        return 3

    # 中小板
    if c3 == '002':
        return 2

    # 北交所 (8xx, 9xx 开头, 4xx)
    if code[0] in ('8', '9', '4'):
        return 5

    # 沪市主板 (6xx)
    if code[0] == '6':
        return 0

    # 深市主板 (000, 001, 003)
    if c3 in ('000', '001', '003'):
        return 1

    return 0  # fallback


def compute_etf_labels(conn, start_date: str, end_date: str) -> pd.DataFrame:
    """从 daily_quotes 为 ETF 计算 label_3d/5d/10d (未来N日收益率)"""
    logger.info("  计算 ETF labels (从 daily_quotes)...")

    query = """
    SELECT s.code, q.trade_date, q.close
    FROM daily_quotes q
    JOIN securities s ON q.security_id = s.id
    WHERE s.type = 'ETF_基金' AND q.volume > 0
    ORDER BY s.code, q.trade_date
    """
    df = pd.read_sql(query, conn)
    logger.info(f"    ETF 行情: {len(df):,} 条, {df['code'].nunique()} 只ETF")

    if df.empty:
        return pd.DataFrame()

    label_parts = []
    for code, grp in df.groupby('code'):
        grp = grp.sort_values('trade_date').reset_index(drop=True)
        close = grp['close'].values
        n = len(close)
        if n < 15:
            continue

        labels = pd.DataFrame({
            'code': grp['code'].values,
            'trade_date': grp['trade_date'].values,
        })
        for days, col in [(3, 'label_3d'), (5, 'label_5d'), (10, 'label_10d')]:
            future = np.full(n, np.nan)
            for i in range(n - days):
                future[i] = close[i + days] / close[i] - 1
            labels[col] = future

        label_parts.append(labels)

    if not label_parts:
        return pd.DataFrame()

    result = pd.concat(label_parts, ignore_index=True)
    result = result.dropna(subset=['label_3d', 'label_5d', 'label_10d'])

    # 过滤日期范围
    result = result[(result['trade_date'] >= start_date) & (result['trade_date'] <= end_date)]
    logger.info(f"    ETF labels 计算完成: {len(result):,} 条")
    return result


def load_feature_data(include_etf: bool, start_date: str = '2022-01-01',
                      end_date: str = '2026-03-20') -> pd.DataFrame:
    """加载 v39_feature_cache 数据 (A股 + 可选ETF)"""
    conn = sqlite3.connect(str(DB_PATH))

    try:
        import orjson
        _loads = orjson.loads
    except ImportError:
        _loads = json.loads

    # ===== 加载 A股 =====
    query_a = """
    SELECT v.code, v.trade_date, v.features_json,
           v.label_3d, v.label_5d, v.label_10d,
           v.market_return_20d, v.market_return_10d, v.market_return_5d,
           v.market_volatility_20d, v.market_volatility_10d,
           v.market_up_ratio_20d, v.market_up_ratio_10d,
           v.market_drawdown_20d, v.market_volume_ratio,
           v.market_position_20d, v.market_momentum_20d, v.market_momentum_5d
    FROM v39_feature_cache v
    JOIN securities s ON v.code = s.code
    WHERE v.label_3d IS NOT NULL
      AND v.label_5d IS NOT NULL
      AND v.label_10d IS NOT NULL
      AND s.type = 'A股'
      AND v.trade_date >= ? AND v.trade_date <= ?
    ORDER BY v.trade_date, v.code
    """
    df = pd.read_sql(query_a, conn, params=[start_date, end_date])
    logger.info(f"  A股记录: {len(df):,}")

    # 过滤交易日 <30 天
    code_counts = df.groupby('code').size()
    valid_codes = code_counts[code_counts >= 30].index
    df = df[df['code'].isin(valid_codes)].copy()
    logger.info(f"  A股过滤后: {len(df):,} ({df['code'].nunique()} 只)")

    # ===== 可选: 加载 ETF =====
    if include_etf:
        # ETF features from cache (no labels)
        query_etf = """
        SELECT v.code, v.trade_date, v.features_json,
               v.market_return_20d, v.market_return_10d, v.market_return_5d,
               v.market_volatility_20d, v.market_volatility_10d,
               v.market_up_ratio_20d, v.market_up_ratio_10d,
               v.market_drawdown_20d, v.market_volume_ratio,
               v.market_position_20d, v.market_momentum_20d, v.market_momentum_5d
        FROM v39_feature_cache v
        JOIN securities s ON v.code = s.code
        WHERE s.type = 'ETF_基金'
          AND v.trade_date >= ? AND v.trade_date <= ?
        """
        df_etf_feat = pd.read_sql(query_etf, conn, params=[start_date, end_date])
        logger.info(f"  ETF 特征记录: {len(df_etf_feat):,}")

        # 计算 ETF labels
        etf_labels = compute_etf_labels(conn, start_date, end_date)

        if not etf_labels.empty and not df_etf_feat.empty:
            # 合并 labels 到 ETF features
            df_etf = df_etf_feat.merge(etf_labels, on=['code', 'trade_date'], how='inner')
            logger.info(f"  ETF 合并后: {len(df_etf):,} ({df_etf['code'].nunique()} 只)")

            # 过滤 <30 天
            etf_counts = df_etf.groupby('code').size()
            valid_etf = etf_counts[etf_counts >= 30].index
            df_etf = df_etf[df_etf['code'].isin(valid_etf)].copy()
            logger.info(f"  ETF 过滤后: {len(df_etf):,} ({df_etf['code'].nunique()} 只)")

            # 合并到主df
            df = pd.concat([df, df_etf], ignore_index=True)
            logger.info(f"  合计: {len(df):,}")

    # ===== 解析 features_json =====
    parsed = df['features_json'].apply(_loads).tolist()
    df_feat = pd.DataFrame(parsed)
    for col in ['code', 'trade_date', 'label_3d', 'label_5d', 'label_10d']:
        df_feat[col] = df[col].values

    # 市场特征
    market_cols = ['market_return_20d', 'market_return_10d', 'market_return_5d',
                   'market_volatility_20d', 'market_volatility_10d',
                   'market_up_ratio_20d', 'market_up_ratio_10d',
                   'market_drawdown_20d', 'market_volume_ratio',
                   'market_position_20d', 'market_momentum_20d', 'market_momentum_5d']
    for col in market_cols:
        if col in df.columns:
            df_feat[col] = df[col].values

    # daily_basic (PE/PB/PS/turnover/market_cap)
    date_min, date_max = df_feat['trade_date'].min(), df_feat['trade_date'].max()
    basic_q = """
    SELECT s.code, db.trade_date, db.pe_ttm, db.pb, db.ps_ttm, db.turnover_rate, db.circ_mv
    FROM daily_basic db
    JOIN securities s ON db.security_id = s.id
    WHERE db.trade_date >= ? AND db.trade_date <= ?
    """
    df_basic = pd.read_sql(basic_q, conn, params=[date_min, date_max])
    df_feat = df_feat.merge(df_basic, on=['code', 'trade_date'], how='left')
    df_feat['log_market_cap'] = np.log1p(df_feat['circ_mv'].fillna(0))
    df_feat.drop(columns=['circ_mv'], inplace=True, errors='ignore')

    # 用截面中位数填充缺失
    for col in ['pe_ttm', 'pb', 'ps_ttm', 'turnover_rate', 'log_market_cap']:
        df_feat[col] = df_feat.groupby('trade_date')[col].transform(lambda x: x.fillna(x.median()))
        df_feat[col] = df_feat[col].fillna(0.0)

    conn.close()

    # 行业超额标签 — ETF(sw_l1_code=-1)单独作为一个行业组
    if 'sw_l1_code' in df_feat.columns:
        for lbl in ['label_3d', 'label_5d', 'label_10d']:
            med = df_feat.groupby(['trade_date', 'sw_l1_code'])[lbl].transform('median')
            df_feat[lbl] = df_feat[lbl] - med

    df_feat = df_feat.fillna(0.0)
    return df_feat


def add_board_type(df: pd.DataFrame, onehot: bool = True) -> pd.DataFrame:
    """添加 board_type + limit_pct 特征"""
    bt = df['code'].apply(derive_board_type)
    df['board_type'] = bt  # 保留整数版用于分析

    # limit_pct: 涨跌停幅度 (连续值, 编码板块唯一影响交易的差异)
    limit_map = {0: 0.10, 1: 0.10, 2: 0.10, 3: 0.20, 4: 0.20, 5: 0.30, 6: 0.10}
    df['limit_pct'] = bt.map(limit_map).astype(np.float32)

    if onehot:
        df['is_sh_main'] = (bt == 0).astype(np.float32)
        df['is_sz_main'] = (bt == 1).astype(np.float32)
        df['is_sme'] = (bt == 2).astype(np.float32)
        df['is_gem'] = (bt == 3).astype(np.float32)
        df['is_star'] = (bt == 4).astype(np.float32)
        df['is_bse'] = (bt == 5).astype(np.float32)
        df['is_etf'] = (bt == 6).astype(np.float32)
    return df


def prepare_and_train(df: pd.DataFrame, add_bt: bool, label: str):
    """准备特征, 训练 LightGBM, 计算 ICIR"""
    import lightgbm as lgb

    board_cols = {'board_type', 'is_sh_main', 'is_sz_main', 'is_sme', 'is_gem', 'is_star', 'is_bse', 'is_etf'}
    exclude = {'code', 'trade_date', 'label_3d', 'label_5d', 'label_10d', 'features_json'}
    # always exclude one-hot and integer board_type; limit_pct controlled by add_bt
    exclude.update(board_cols)
    if not add_bt:
        exclude.add('limit_pct')

    feat_cols = [c for c in df.columns if c not in exclude]
    target_col = 'label_10d'

    # 时间划分: 前70%训练, 后30%测试
    dates = np.sort(df['trade_date'].unique())
    split_idx = int(len(dates) * 0.7)
    train_end = dates[split_idx]
    test_start = dates[min(split_idx + 15, len(dates) - 1)]  # 15天purge gap

    train_mask = df['trade_date'] <= train_end
    test_mask = df['trade_date'] >= test_start

    X_train = df.loc[train_mask, feat_cols].values.astype(np.float32)
    y_train = df.loc[train_mask, target_col].values.astype(np.float32)
    X_test = df.loc[test_mask, feat_cols].values.astype(np.float32)
    y_test = df.loc[test_mask, target_col].values.astype(np.float32)
    test_dates = df.loc[test_mask, 'trade_date'].values
    test_codes = df.loc[test_mask, 'code'].values
    test_board = df.loc[test_mask, 'board_type'].values if 'board_type' in df.columns else None

    n_etf_train = (df.loc[train_mask, 'board_type'] == 6).sum() if 'board_type' in df.columns else 0
    n_etf_test = (df.loc[test_mask, 'board_type'] == 6).sum() if 'board_type' in df.columns else 0

    logger.info(f"  [{label}] 特征: {len(feat_cols)}, 训练: {len(X_train):,}"
                f" (ETF: {n_etf_train:,}), 测试: {len(X_test):,} (ETF: {n_etf_test:,})")

    # Winsorize
    lo = np.percentile(X_train, 1, axis=0)
    hi = np.percentile(X_train, 99, axis=0)
    X_train = np.clip(X_train, lo, hi)
    X_test = np.clip(X_test, lo, hi)

    # LightGBM
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'learning_rate': 0.02,
        'num_leaves': 31,
        'min_data_in_leaf': 200,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
    }

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feat_cols, free_raw_data=False)
    model = lgb.train(params, dtrain, num_boost_round=500)

    pred = model.predict(X_test)

    # ===== Daily IC / ICIR =====
    unique_dates = np.unique(test_dates)
    daily_ics = []
    top10_rets = []

    for d in unique_dates:
        mask = test_dates == d
        y_d = y_test[mask]
        p_d = pred[mask]
        if len(y_d) < 20:
            continue
        ic, _ = spearmanr(p_d, y_d)
        if not np.isnan(ic):
            daily_ics.append(ic)
        top_idx = np.argsort(p_d)[-10:]
        top10_rets.append(np.mean(y_d[top_idx]))

    mean_ic = np.mean(daily_ics)
    std_ic = np.std(daily_ics)
    icir = mean_ic / max(std_ic, 1e-8)
    ic_pos_rate = np.mean(np.array(daily_ics) > 0) * 100
    top10_mean = np.mean(top10_rets) * 100

    # limit_pct 重要度
    bt_pct = None
    if add_bt and 'limit_pct' in feat_cols:
        importances = model.feature_importance(importance_type='gain')
        lp_idx = feat_cols.index('limit_pct')
        bt_pct = importances[lp_idx] / importances.sum() * 100 if importances.sum() > 0 else 0

    # ETF 子集 ICIR (仅当含 ETF 时)
    etf_ic_info = None
    if test_board is not None:
        etf_mask_arr = test_board == 6
        if etf_mask_arr.sum() > 0:
            etf_daily_ics = []
            etf_top5_rets = []
            for d in unique_dates:
                mask = (test_dates == d) & etf_mask_arr
                y_d = y_test[mask]
                p_d = pred[mask]
                if len(y_d) < 5:
                    continue
                ic, _ = spearmanr(p_d, y_d)
                if not np.isnan(ic):
                    etf_daily_ics.append(ic)
                top_idx = np.argsort(p_d)[-5:]
                etf_top5_rets.append(np.mean(y_d[top_idx]))

            if etf_daily_ics:
                etf_ic_info = {
                    'mean_ic': np.mean(etf_daily_ics),
                    'icir': np.mean(etf_daily_ics) / max(np.std(etf_daily_ics), 1e-8),
                    'n_days': len(etf_daily_ics),
                    'top5_ret': np.mean(etf_top5_rets) * 100,
                }

    # A股子集 ICIR (当含 ETF 时 — 检查 ETF 是否污染了 A股预测)
    a_only_icir = None
    if test_board is not None:
        a_mask_arr = test_board != 6
        if a_mask_arr.sum() > 0:
            a_daily_ics = []
            for d in unique_dates:
                mask = (test_dates == d) & a_mask_arr
                y_d = y_test[mask]
                p_d = pred[mask]
                if len(y_d) < 20:
                    continue
                ic, _ = spearmanr(p_d, y_d)
                if not np.isnan(ic):
                    a_daily_ics.append(ic)
            if a_daily_ics:
                a_only_icir = np.mean(a_daily_ics) / max(np.std(a_daily_ics), 1e-8)

    # Top feature importances
    importances = model.feature_importance(importance_type='gain')
    top_feats = sorted(zip(feat_cols, importances), key=lambda x: -x[1])[:10]

    return {
        'label': label,
        'n_features': len(feat_cols),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'mean_ic': mean_ic,
        'icir': icir,
        'ic_pos_rate': ic_pos_rate,
        'top10_ret_pct': top10_mean,
        'n_test_days': len(daily_ics),
        'bt_importance_pct': bt_pct,
        'etf_ic_info': etf_ic_info,
        'a_only_icir': a_only_icir,
        'top_features': top_feats,
    }


def main():
    logger.info("=" * 70)
    logger.info("V4.8.5 可行性评估: board_type + ETF 对 ICIR 影响")
    logger.info("=" * 70)

    start_date = '2022-01-01'
    end_date = '2026-03-20'

    # ===== 加载数据 =====
    logger.info("\n[1/2] 加载 A股 数据")
    df_a = load_feature_data(include_etf=False, start_date=start_date, end_date=end_date)
    df_a = add_board_type(df_a)

    logger.info("\n[2/2] 加载 A股 + ETF 数据")
    df_ae = load_feature_data(include_etf=True, start_date=start_date, end_date=end_date)
    df_ae = add_board_type(df_ae)

    # 统计
    bt_names = {0: '沪市主板', 1: '深市主板', 2: '中小板', 3: '创业板', 4: '科创板', 5: '北交所', 6: 'ETF'}
    logger.info("\n板块分布 (A股+ETF):")
    for bt in sorted(bt_names.keys()):
        cnt = (df_ae['board_type'] == bt).sum()
        if cnt > 0:
            logger.info(f"  {bt_names[bt]}({bt}): {cnt:,} ({cnt/len(df_ae)*100:.1f}%)")

    # ===== 4组实验 =====
    results = []

    logger.info("\n" + "=" * 50)
    logger.info("实验A: 基线 (A股 only)")
    results.append(prepare_and_train(df_a, add_bt=False, label='A: 基线'))

    logger.info("\n" + "=" * 50)
    logger.info("实验B: A股 + limit_pct")
    results.append(prepare_and_train(df_a, add_bt=True, label='B: +limit_pct'))

    logger.info("\n" + "=" * 50)
    logger.info("实验C: A股 + ETF")
    results.append(prepare_and_train(df_ae, add_bt=False, label='C: +ETF'))

    logger.info("\n" + "=" * 50)
    logger.info("实验D: A股 + ETF + limit_pct")
    results.append(prepare_and_train(df_ae, add_bt=True, label='D: +ETF+limit'))

    # ===== 汇总 =====
    logger.info("\n" + "=" * 80)
    logger.info("汇总结果 (label_10d, LightGBM 500轮)")
    logger.info("=" * 80)
    logger.info(f"{'实验':<18} {'feat':>5} {'训练':>10} {'测试':>10} {'IC':>8} {'ICIR':>7} {'Δ':>8} {'IC>0':>6} {'Top10%':>8}")
    logger.info("-" * 80)

    baseline_icir = results[0]['icir']
    for r in results:
        delta = r['icir'] - baseline_icir
        delta_str = f"{delta:+.3f}" if r['label'] != 'A: 基线' else "   —"
        logger.info(f"{r['label']:<18} {r['n_features']:>5} {r['n_train']:>10,} {r['n_test']:>10,} "
                     f"{r['mean_ic']:>8.4f} {r['icir']:>7.3f} {delta_str:>8} {r['ic_pos_rate']:>5.1f}% "
                     f"{r['top10_ret_pct']:>7.2f}%")

    # board_type 重要度
    logger.info("\n--- limit_pct 特征重要度 ---")
    for r in results:
        if r['bt_importance_pct'] is not None:
            logger.info(f"  {r['label']}: limit_pct = {r['bt_importance_pct']:.3f}% (gain)")

    # A股子集 ICIR (检查 ETF 是否污染)
    logger.info("\n--- A股子集 ICIR (ETF是否影响A股预测质量) ---")
    for r in results:
        if r['a_only_icir'] is not None:
            delta = r['a_only_icir'] - baseline_icir
            logger.info(f"  {r['label']}: A股 ICIR={r['a_only_icir']:.3f} (vs 基线 {baseline_icir:.3f}, Δ={delta:+.3f})")

    # ETF 子集
    logger.info("\n--- ETF 子集预测能力 ---")
    for r in results:
        if r['etf_ic_info']:
            ei = r['etf_ic_info']
            logger.info(f"  {r['label']}: ETF IC={ei['mean_ic']:.4f}, ICIR={ei['icir']:.3f}, "
                         f"Top5 ret={ei['top5_ret']:.2f}%, {ei['n_days']}天")

    # Top features
    logger.info("\n--- Top 10 特征重要度 (实验D) ---")
    for feat, imp in results[-1]['top_features']:
        logger.info(f"  {feat:<30} {imp:>10.0f}")

    # ===== 结论 =====
    logger.info("\n" + "=" * 70)
    logger.info("结论")
    logger.info("=" * 70)

    b_delta = results[1]['icir'] - baseline_icir
    c_delta = results[2]['icir'] - baseline_icir
    d_delta = results[3]['icir'] - baseline_icir

    for name, delta in [('limit_pct', b_delta), ('ETF数据', c_delta), ('两者结合', d_delta)]:
        if delta > 0.05:
            logger.info(f"  ✅ {name} 明显有益: ICIR {delta:+.3f}")
        elif delta > 0.01:
            logger.info(f"  ✅ {name} 轻微有益: ICIR {delta:+.3f}")
        elif delta > -0.02:
            logger.info(f"  ➖ {name} 影响中性: ICIR {delta:+.3f}")
        else:
            logger.info(f"  ❌ {name} 有害: ICIR {delta:+.3f}")

    best_idx = np.argmax([r['icir'] for r in results])
    logger.info(f"\n  🏆 最佳: {results[best_idx]['label']} (ICIR={results[best_idx]['icir']:.3f})")


if __name__ == '__main__':
    main()
