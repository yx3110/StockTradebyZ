#!/usr/bin/env python3
"""
市场门控模型V2 — 增强版

30+特征 × 回归标签 × 扩展窗口CV × 规则混合

特征维度:
  1. 市场微观结构 (7): 涨跌停比/上涨占比/截面离散/新高新低
  2. 指数分化 (4): 大小盘强弱/成长价值/指数共振
  3. 估值资金面 (4): PE百分位/PB百分位/换手百分位/量比
  4. 长周期动量 (6): 60d/120d return/vol + drawdown + 连涨连跌
  5. 原始短周期 (8): 原V1的market_return/vol/momentum精选

训练: 回归预测market_return_10d, 推理时阈值化为confidence
"""

import numpy as np
import pandas as pd
import sqlite3
import logging
import joblib
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / 'data_adapter' / 'stock_data.db'
BENCHMARK_CODE = '000905.SH'


def compute_gate_features(start_date='2019-01-01', end_date='2026-12-31'):
    """计算所有门控特征, 返回per-day DataFrame"""
    conn = sqlite3.connect(str(DB_PATH))
    logger.info("加载基础数据...")

    # === 1. 基准指数日线 ===
    bm_query = """
    SELECT q.trade_date, q.close, q.volume
    FROM daily_quotes q JOIN securities s ON q.security_id = s.id
    WHERE s.code = ? AND q.trade_date >= ? AND q.trade_date <= ?
    ORDER BY q.trade_date
    """
    bm = pd.read_sql(bm_query, conn, params=[BENCHMARK_CODE, start_date, end_date])
    bm['trade_date'] = bm['trade_date'].astype(str)
    bm = bm.sort_values('trade_date').reset_index(drop=True)
    logger.info(f"  基准 {BENCHMARK_CODE}: {len(bm)} 天")

    # === 2. 多指数日线 (大小盘分化) ===
    index_codes = ['000300.SH', '000905.SH', '399006.SZ', '000852.SH']
    index_dfs = {}
    for code in index_codes:
        df = pd.read_sql(bm_query, conn, params=[code, start_date, end_date])
        df['trade_date'] = df['trade_date'].astype(str)
        df = df.sort_values('trade_date').set_index('trade_date')
        index_dfs[code] = df['close']
        logger.info(f"  指数 {code}: {len(df)} 天")

    # === 3. 全A股每日统计 (涨跌停/上涨比/截面std) ===
    logger.info("  计算全市场微观结构特征...")
    daily_stats_query = """
    SELECT q.trade_date,
           COUNT(*) as n_stocks,
           SUM(CASE WHEN q.price_change_pct > 0.095 THEN 1 ELSE 0 END) as limit_up,
           SUM(CASE WHEN q.price_change_pct < -0.095 THEN 1 ELSE 0 END) as limit_down,
           SUM(CASE WHEN q.price_change_pct > 0 THEN 1 ELSE 0 END) as up_stocks,
           AVG(q.price_change_pct) as avg_return,
           SUM(q.volume) as total_volume
    FROM daily_quotes q
    JOIN securities s ON q.security_id = s.id
    WHERE s.type = 'A股' AND q.trade_date >= ? AND q.trade_date <= ?
      AND q.volume > 0
    GROUP BY q.trade_date
    ORDER BY q.trade_date
    """
    ds = pd.read_sql(daily_stats_query, conn, params=[start_date, end_date])
    ds['trade_date'] = ds['trade_date'].astype(str)
    logger.info(f"  全市场日统计: {len(ds)} 天")

    # 截面收益率标准差 (需要单独query)
    logger.info("  计算截面收益率标准差...")
    cs_std_query = """
    SELECT q.trade_date,
           STDEV(q.price_change_pct) as cross_section_std
    FROM (
        SELECT trade_date, price_change_pct
        FROM daily_quotes q JOIN securities s ON q.security_id = s.id
        WHERE s.type = 'A股' AND q.trade_date >= ? AND q.trade_date <= ? AND q.volume > 0
    ) q
    GROUP BY q.trade_date
    """
    # SQLite没有STDEV, 用Python计算
    pct_query = """
    SELECT q.trade_date, q.price_change_pct
    FROM daily_quotes q JOIN securities s ON q.security_id = s.id
    WHERE s.type = 'A股' AND q.trade_date >= ? AND q.trade_date <= ? AND q.volume > 0
    """
    df_pct = pd.read_sql(pct_query, conn, params=[start_date, end_date])
    df_pct['trade_date'] = df_pct['trade_date'].astype(str)
    cs_std = df_pct.groupby('trade_date')['price_change_pct'].std().reset_index()
    cs_std.columns = ['trade_date', 'cross_section_std']
    logger.info(f"  截面std: {len(cs_std)} 天")

    # 新高新低 (20日新高/新低占比)
    logger.info("  计算新高新低比...")
    nh_nl_query = """
    SELECT q.trade_date, s.code, q.close
    FROM daily_quotes q JOIN securities s ON q.security_id = s.id
    WHERE s.type = 'A股' AND q.trade_date >= ? AND q.trade_date <= ? AND q.volume > 0
    ORDER BY s.code, q.trade_date
    """
    # 这个query太大，用近似方法: 从日统计推算
    # 改用: high_close_20d = rolling 20d max of close, 如果 close >= max → 新高
    # 太慢，换用简化版: 用price_change_pct > 0.06 (大涨日) 做代理
    # 这已经有涨停数据了，跳过新高新低

    # === 4. 全市场估值 ===
    logger.info("  计算全市场估值特征...")
    valuation_query = """
    SELECT db.trade_date,
           MEDIAN(db.pe_ttm) as median_pe,
           MEDIAN(db.pb) as median_pb,
           AVG(db.turnover_rate) as avg_turnover
    FROM daily_basic db
    JOIN securities s ON db.security_id = s.id
    WHERE s.type = 'A股' AND db.trade_date >= ? AND db.trade_date <= ?
      AND db.pe_ttm > 0 AND db.pe_ttm < 500
    GROUP BY db.trade_date
    """
    # SQLite没有MEDIAN, 用Python
    val_query = """
    SELECT db.trade_date, db.pe_ttm, db.pb, db.turnover_rate
    FROM daily_basic db JOIN securities s ON db.security_id = s.id
    WHERE s.type = 'A股' AND db.trade_date >= ? AND db.trade_date <= ?
      AND db.pe_ttm > 0 AND db.pe_ttm < 500
    """
    df_val = pd.read_sql(val_query, conn, params=[start_date, end_date])
    df_val['trade_date'] = df_val['trade_date'].astype(str)
    val_daily = df_val.groupby('trade_date').agg(
        median_pe=('pe_ttm', 'median'),
        median_pb=('pb', 'median'),
        avg_turnover=('turnover_rate', 'mean'),
    ).reset_index()
    logger.info(f"  估值数据: {len(val_daily)} 天")

    conn.close()

    # ========== 构建特征DataFrame ==========
    logger.info("构建特征矩阵...")

    # 基准为主轴
    feat = bm[['trade_date', 'close', 'volume']].copy()
    feat = feat.rename(columns={'close': 'bm_close', 'volume': 'bm_volume'})
    feat = feat.sort_values('trade_date').reset_index(drop=True)

    # --- 维度1: 微观结构 ---
    ds_merge = ds[['trade_date', 'n_stocks', 'limit_up', 'limit_down',
                   'up_stocks', 'total_volume']].copy()
    feat = feat.merge(ds_merge, on='trade_date', how='left')
    feat = feat.merge(cs_std, on='trade_date', how='left')

    feat['limit_up_ratio'] = feat['limit_up'] / feat['n_stocks'].clip(lower=1)
    feat['limit_down_ratio'] = feat['limit_down'] / feat['n_stocks'].clip(lower=1)
    feat['limit_ud_ratio'] = (feat['limit_up'] + 1) / (feat['limit_down'] + 1)
    feat['up_stock_ratio'] = feat['up_stocks'] / feat['n_stocks'].clip(lower=1)

    # 5日平均
    for col in ['limit_up_ratio', 'limit_down_ratio', 'limit_ud_ratio',
                'up_stock_ratio', 'cross_section_std']:
        feat[f'{col}_5d'] = feat[col].rolling(5, min_periods=1).mean()

    # --- 维度2: 指数分化 ---
    for code, series in index_dfs.items():
        tag = code.split('.')[0]
        feat = feat.merge(
            series.rename(f'idx_{tag}').reset_index(),
            on='trade_date', how='left'
        )

    # 相对强弱: 10日收益率之差
    for tag, close_col in [('000300', 'idx_000300'), ('000905', 'idx_000905'),
                            ('399006', 'idx_399006'), ('000852', 'idx_000852')]:
        feat[f'ret10d_{tag}'] = feat[close_col].pct_change(10)

    feat['hs300_vs_zz500'] = feat.get('ret10d_000300', 0) - feat.get('ret10d_000905', 0)
    feat['zz1000_vs_zz500'] = feat.get('ret10d_000852', 0) - feat.get('ret10d_000905', 0)
    feat['cyb_vs_hs300'] = feat.get('ret10d_399006', 0) - feat.get('ret10d_000300', 0)

    # 指数共振: 4个指数中几个10d收益为正
    idx_rets = feat[['ret10d_000300', 'ret10d_000905', 'ret10d_399006', 'ret10d_000852']]
    feat['index_breadth'] = (idx_rets > 0).sum(axis=1) / 4.0

    # --- 维度3: 估值 ---
    feat = feat.merge(val_daily, on='trade_date', how='left')

    # PE/PB的历史百分位 (750日≈3年)
    feat['pe_pctl_750d'] = feat['median_pe'].rolling(750, min_periods=60).rank(pct=True)
    feat['pb_pctl_750d'] = feat['median_pb'].rolling(750, min_periods=60).rank(pct=True)
    feat['turnover_pctl_60d'] = feat['avg_turnover'].rolling(60, min_periods=10).rank(pct=True)

    # 量比: 当日成交量/5日均量
    feat['volume_ma5_ratio'] = feat['total_volume'] / feat['total_volume'].rolling(5, min_periods=1).mean()

    # --- 维度4: 长周期动量 ---
    close = feat['bm_close']
    for w in [5, 10, 20, 60, 120]:
        feat[f'bm_return_{w}d'] = close.pct_change(w)

    for w in [20, 60]:
        log_ret = np.log(close / close.shift(1))
        feat[f'bm_vol_{w}d'] = log_ret.rolling(w, min_periods=5).std() * np.sqrt(252)

    # Drawdown from 60d peak
    feat['bm_peak_60d'] = close.rolling(60, min_periods=5).max()
    feat['bm_drawdown_60d'] = close / feat['bm_peak_60d'] - 1

    # 连涨连跌天数
    daily_ret = close.pct_change()
    up_streak = np.zeros(len(feat))
    down_streak = np.zeros(len(feat))
    for i in range(1, len(feat)):
        if daily_ret.iloc[i] > 0:
            up_streak[i] = up_streak[i-1] + 1
            down_streak[i] = 0
        elif daily_ret.iloc[i] < 0:
            down_streak[i] = down_streak[i-1] + 1
            up_streak[i] = 0
    feat['consecutive_up'] = up_streak
    feat['consecutive_down'] = down_streak

    # --- 维度5: 原始短周期 (精选, 非全部) ---
    # market_return_5d/10d/20d 已由bm_return计算
    # market_volatility 已由bm_vol计算
    # 额外: RSI-like for benchmark
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=5).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=5).mean()
    feat['bm_rsi_14'] = 100 - 100 / (1 + gain / loss.clip(lower=1e-8))

    # ========== 构建标签 ==========
    logger.info("构建回归标签...")

    # 回归标签: 基准未来10天收益
    feat['label_return_10d'] = close.shift(-10) / close - 1
    # 二分类标签 (备用)
    feat['label_positive_10d'] = (feat['label_return_10d'] > 0).astype(int)
    # 非对称标签: return > -1% (允许小亏)
    feat['label_safe_10d'] = (feat['label_return_10d'] > -0.01).astype(int)

    # ========== 清理 ==========
    # 选择最终特征列
    feature_cols = [
        # 微观结构
        'limit_up_ratio', 'limit_down_ratio', 'limit_ud_ratio',
        'up_stock_ratio', 'cross_section_std',
        'limit_up_ratio_5d', 'limit_down_ratio_5d', 'limit_ud_ratio_5d',
        'up_stock_ratio_5d', 'cross_section_std_5d',
        # 指数分化
        'hs300_vs_zz500', 'zz1000_vs_zz500', 'cyb_vs_hs300', 'index_breadth',
        # 估值
        'pe_pctl_750d', 'pb_pctl_750d', 'turnover_pctl_60d', 'volume_ma5_ratio',
        # 长周期
        'bm_return_5d', 'bm_return_10d', 'bm_return_20d',
        'bm_return_60d', 'bm_return_120d',
        'bm_vol_20d', 'bm_vol_60d',
        'bm_drawdown_60d', 'consecutive_up', 'consecutive_down',
        # RSI
        'bm_rsi_14',
    ]

    # 只保留有效行
    feat = feat.dropna(subset=['label_return_10d'])
    for col in feature_cols:
        if col not in feat.columns:
            feat[col] = 0.0
    feat[feature_cols] = feat[feature_cols].fillna(0.0)

    logger.info(f"最终数据: {len(feat)} 天, {len(feature_cols)} 特征")
    return feat, feature_cols


def train_gate_model_v2(feat, feature_cols):
    """训练增强版门控模型: 回归 + 扩展窗口CV + 规则混合"""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, mean_squared_error

    feat = feat.sort_values('trade_date').reset_index(drop=True)
    X = feat[feature_cols].values
    y_reg = feat['label_return_10d'].values
    y_cls = feat['label_positive_10d'].values
    y_safe = feat['label_safe_10d'].values
    dates = feat['trade_date'].values

    logger.info(f"\n训练门控模型V2 ({len(feat)} 天, {len(feature_cols)} 特征)")
    logger.info(f"  正类(return>0)比例: {y_cls.mean():.1%}")
    logger.info(f"  安全类(return>-1%)比例: {y_safe.mean():.1%}")

    # === 扩展窗口CV (4折) ===
    n = len(feat)
    fold_size = n // 5  # 20% per fold
    cv_results = []

    for fold in range(4):
        train_end = (fold + 1) * fold_size + fold_size  # 至少2折训练
        val_start = train_end
        val_end = min(val_start + fold_size, n)

        if val_end <= val_start:
            continue

        X_tr, y_tr = X[:train_end], y_reg[:train_end]
        X_val, y_val = X[val_start:val_end], y_reg[val_start:val_end]
        y_val_cls = y_cls[val_start:val_end]
        y_val_safe = y_safe[val_start:val_end]

        # 回归模型
        reg_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 12,
            'learning_rate': 0.03,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.8,
            'bagging_freq': 3,
            'reg_alpha': 2.0,
            'reg_lambda': 10.0,
            'min_data_in_leaf': 30,
            'verbose': -1,
        }

        ds_tr = lgb.Dataset(X_tr, y_tr)
        ds_val = lgb.Dataset(X_val, y_val, reference=ds_tr)

        model = lgb.train(
            reg_params, ds_tr,
            num_boost_round=300,
            valid_sets=[ds_val],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
        )

        pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, pred))

        # 用回归预测值做分类: pred > 0 → positive
        pred_cls = (pred > 0).astype(int)
        auc_cls = roc_auc_score(y_val_cls, pred)

        # 用回归预测值做安全分类: pred > -0.01 → safe
        pred_safe = pred > -0.01
        auc_safe = roc_auc_score(y_val_safe, pred)

        cv_results.append({
            'fold': fold, 'train_n': len(X_tr), 'val_n': len(X_val),
            'rmse': rmse, 'auc_cls': auc_cls, 'auc_safe': auc_safe,
            'trees': model.num_trees(),
            'val_dates': f"{dates[val_start]}~{dates[val_end-1]}",
        })
        logger.info(f"  Fold {fold}: train={len(X_tr)}, val={len(X_val)}, "
                     f"RMSE={rmse:.5f}, AUC(cls)={auc_cls:.3f}, AUC(safe)={auc_safe:.3f}, "
                     f"trees={model.num_trees()}")

    avg_auc = np.mean([r['auc_cls'] for r in cv_results])
    avg_auc_safe = np.mean([r['auc_safe'] for r in cv_results])
    logger.info(f"\n  CV平均: AUC(cls)={avg_auc:.3f}, AUC(safe)={avg_auc_safe:.3f}")

    # === 训练最终模型 (全量) ===
    logger.info("\n训练最终生产模型 (85% train + 15% val)...")
    split = int(n * 0.85)
    X_tr, y_tr = X[:split], y_reg[:split]
    X_val, y_val = X[split:], y_reg[split:]

    ds_tr = lgb.Dataset(X_tr, y_tr)
    ds_val = lgb.Dataset(X_val, y_val, reference=ds_tr)

    final_model = lgb.train(
        reg_params, ds_tr,
        num_boost_round=500,
        valid_sets=[ds_val],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
    )

    val_pred = final_model.predict(X_val)
    val_auc = roc_auc_score(y_cls[split:], val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    logger.info(f"  最终模型: {final_model.num_trees()} trees, Val AUC={val_auc:.3f}, RMSE={val_rmse:.5f}")

    # 特征重要性
    imp = final_model.feature_importance(importance_type='gain')
    imp_sorted = sorted(zip(feature_cols, imp), key=lambda x: -x[1])
    logger.info("\n  Top-10 特征重要性:")
    for name, score in imp_sorted[:10]:
        logger.info(f"    {name}: {score:.0f}")

    # === 校准: 回归值 → confidence ===
    # 用验证集的预测分布确定阈值
    all_pred = final_model.predict(X)
    p10 = np.percentile(all_pred, 10)
    p25 = np.percentile(all_pred, 25)
    p50 = np.percentile(all_pred, 50)
    p75 = np.percentile(all_pred, 75)
    logger.info(f"\n  全量预测分布: P10={p10:.4f}, P25={p25:.4f}, P50={p50:.4f}, P75={p75:.4f}")

    # confidence = percentile_rank(pred) in [0, 1]
    # 但要保存分布用于推理时的映射
    pred_quantiles = np.percentile(all_pred, np.linspace(0, 100, 101))

    # === 简单规则得分 ===
    logger.info("\n计算规则得分...")
    rule_scores = np.full(n, 0.5)
    for i in range(n):
        score = 0.5
        # 涨跌停比>2 → 偏牛
        if feat['limit_ud_ratio_5d'].iloc[i] > 2.0:
            score += 0.15
        elif feat['limit_ud_ratio_5d'].iloc[i] < 0.5:
            score -= 0.15

        # PE低位 → 偏牛
        pe_pctl = feat['pe_pctl_750d'].iloc[i]
        if pe_pctl < 0.3:
            score += 0.10
        elif pe_pctl > 0.8:
            score -= 0.10

        # 深度回撤 → 偏熊
        dd = feat['bm_drawdown_60d'].iloc[i]
        if dd < -0.15:
            score -= 0.15
        elif dd > -0.03:
            score += 0.05

        # 指数共振
        breadth = feat['index_breadth'].iloc[i]
        if breadth >= 1.0:
            score += 0.10
        elif breadth <= 0.0:
            score -= 0.10

        rule_scores[i] = np.clip(score, 0.0, 1.0)

    # 验证集上的规则AUC
    rule_auc = roc_auc_score(y_cls[split:], rule_scores[split:])
    logger.info(f"  规则AUC: {rule_auc:.3f}")

    # === 混合: 0.7×模型 + 0.3×规则 ===
    # 把模型的回归值转换到[0,1] via percentile mapping
    model_confidence = np.searchsorted(pred_quantiles, all_pred) / 100.0
    mixed = 0.7 * model_confidence + 0.3 * rule_scores
    mixed_auc = roc_auc_score(y_cls[split:], mixed[split:])
    logger.info(f"  混合AUC: {mixed_auc:.3f} (模型{val_auc:.3f} + 规则{rule_auc:.3f})")

    # === 保存 ===
    gate_data = {
        'model': final_model,
        'model_type': 'regression',
        'feature_names': feature_cols,
        'pred_quantiles': pred_quantiles,  # 用于推理时将回归值映射到confidence
        'rule_weights': {
            'model': 0.7,
            'rule': 0.3,
        },
        'rule_features': ['limit_ud_ratio_5d', 'pe_pctl_750d', 'bm_drawdown_60d', 'index_breadth'],
        'cv_auc': avg_auc,
        'cv_auc_safe': avg_auc_safe,
        'val_auc': val_auc,
        'mixed_auc': mixed_auc,
        'rule_auc': rule_auc,
        'rmse': val_rmse,
        'n_features': len(feature_cols),
        'n_days': n,
        'version': 'v2',
    }

    out_path = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'market_gate_model.pkl'
    joblib.dump(gate_data, str(out_path))
    logger.info(f"\n门控模型V2已保存: {out_path}")
    logger.info(f"  特征数: {len(feature_cols)}")
    logger.info(f"  CV AUC: {avg_auc:.3f}")
    logger.info(f"  Val AUC: {val_auc:.3f}")
    logger.info(f"  Mixed AUC: {mixed_auc:.3f}")

    return gate_data


if __name__ == '__main__':
    feat, feature_cols = compute_gate_features(start_date='2019-01-01')
    gate_data = train_gate_model_v2(feat, feature_cols)
