#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V5.0 Unified Feature Fusion 训练脚本

核心思路: 在 V3.96 (Robust Z-Score + Industry-Excess) 的成功基础上,
注入 V4.0 的独特高价值特征 (行业内排名、流动性、量价相关、交互特征等)

特征来源:
  - v39_feature_cache: 23 个基础股票特征 + 13 个市场特征 (V3.96 已验证)
  - v40_feature_cache: ~20 个精选独特特征 (流动性/风险/行业排名/交互信号)
  - daily_basic: 5 个基本面特征 (pe_ttm, pb, ps_ttm, turnover_rate, log_market_cap)

归一化策略:
  - v39 股票特征: Robust Z-Score (保留幅度信息)
  - v40 行业内排名特征 (0-1): 跳过归一化 (已归一化)
  - v40 连续值特征: Robust Z-Score
  - 市场/宏观特征: 保持原值
  - daily_basic 特征: Robust Z-Score

训练框架: 沿用 V3.96 (独立训练 3d/5d/10d, 5模型 Ensemble)
"""

import sys
import numpy as np
import pandas as pd
import sqlite3
import json
from datetime import datetime
import logging
from pathlib import Path
import joblib
from tqdm import tqdm
import argparse
import warnings
import gc

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings('ignore')

import lightgbm as lgb
import xgboost as xgb
try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("Warning: CatBoost not installed")

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')

# V4.0 独特特征精选 (V3.96 缺失的)
# 分为两类: 已归一化的rank特征 (0-1) 和需要 z-score 的连续特征
V40_RANK_FEATURES = [
    # 行业内技术排名 (0-1, 已归一化)
    'xs_kdj_j_rank', 'xs_rsi6_rank', 'xs_boll_position_rank',
    'xs_macd_hist_rank', 'xs_atr14_pct_rank',
    # 行业内基本面/成交排名
    'xs_turnover_rank', 'xs_volume_ratio_rank',
    # 行业内收益/波动排名
    'xs_return_5d_rank', 'xs_return_10d_rank',
    'xs_volatility_10d_rank',
    # 交互信号 (0-1 范围)
    'momentum_volume_confirm', 'tech_momentum_confirm',
    'contrarian_signal', 'boll_position',
]

V40_CONTINUOUS_FEATURES = [
    # 流动性
    'amihud_illiquidity', 'volume_price_corr_10d',
    # 风险
    'updown_asymmetry_10d', 'max_drawdown_20d',
    # 行业动态
    'industry_momentum_rank', 'industry_rotation_signal',
    'industry_breadth', 'industry_concentration',
    'industry_volume_change',
    # 策略信号
    'squeeze_momentum', 'zhixing_short_trend',
]

V40_SELECTED_FEATURES = V40_RANK_FEATURES + V40_CONTINUOUS_FEATURES

# 市场/宏观特征 (不做截面归一化)
MACRO_FEATURE_NAMES = [
    'market_return_20d', 'market_return_10d', 'market_return_5d',
    'market_volatility_20d', 'market_volatility_10d',
    'market_up_ratio_20d', 'market_up_ratio_10d',
    'market_drawdown_20d', 'market_volume_ratio',
    'market_position_20d', 'market_momentum_20d', 'market_momentum_5d',
    'northbound_flow_5d',
]


class V500UnifiedTrainer:
    """V5.0 Unified Feature Fusion 训练器"""

    def __init__(self, db_path: str = DB_PATH, v40_features: list = None,
                 include_neural: bool = False):
        self.db_path = db_path
        self.models = {}
        self.feature_names = None
        self.winsorize_bounds = None
        self.v40_features = v40_features or V40_SELECTED_FEATURES
        self.include_neural = include_neural  # Phase 3: GRU embeddings

        # Feature classification (set during prepare_features)
        self.stock_feature_cols = None
        self.macro_feature_cols = None
        self.v40_rank_cols = None
        self.v40_continuous_cols = None
        self.neural_cols = None

        self.target_weights = {
            'label_3d': 0.4,
            'label_5d': 0.35,
            'label_10d': 0.25
        }

    def load_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """加载 v39 + v40 + daily_basic 融合数据"""
        logger.info("=" * 80)
        logger.info("V5.0 数据加载: v39 + v40 + daily_basic 特征融合")
        logger.info("=" * 80)

        conn = sqlite3.connect(self.db_path)

        # --- Step 1: 加载 v39 基础特征 ---
        date_filter = ""
        date_params = []
        if start_date:
            date_filter += " AND v.trade_date >= ?"
            date_params.append(start_date)
        if end_date:
            date_filter += " AND v.trade_date <= ?"
            date_params.append(end_date)

        query_v39 = f"""
        SELECT
            v.code, v.trade_date, v.features_json,
            v.label_3d, v.label_5d, v.label_10d,
            v.market_return_20d, v.market_return_10d, v.market_return_5d,
            v.market_volatility_20d, v.market_volatility_10d,
            v.market_up_ratio_20d, v.market_up_ratio_10d,
            v.market_drawdown_20d, v.market_volume_ratio,
            v.market_position_20d, v.market_momentum_20d, v.market_momentum_5d
        FROM v39_feature_cache v
        JOIN securities s ON v.code = s.code
        JOIN daily_quotes q ON q.security_id = s.id AND q.trade_date = v.trade_date
        WHERE v.label_3d IS NOT NULL
          AND v.label_5d IS NOT NULL
          AND v.label_10d IS NOT NULL
          AND q.volume > 0
          AND v.code IN (
              SELECT s2.code FROM daily_quotes q2
              JOIN securities s2 ON q2.security_id = s2.id
              WHERE s2.type = 'A股'
              GROUP BY s2.code
              HAVING COUNT(*) >= 30
          )
          {date_filter}
        ORDER BY v.trade_date, v.code
        """

        df_v39 = pd.read_sql(query_v39, conn, params=date_params if date_params else None)
        logger.info(f"  v39 记录: {len(df_v39):,}")

        # 解析 v39 features_json
        v39_features_list = []
        for idx, row in tqdm(df_v39.iterrows(), total=len(df_v39), desc="解析v39特征"):
            try:
                features = json.loads(row['features_json'])
                features['code'] = row['code']
                features['trade_date'] = row['trade_date']
                features['label_3d'] = row['label_3d']
                features['label_5d'] = row['label_5d']
                features['label_10d'] = row['label_10d']
                # Add market features from columns
                for mc in MACRO_FEATURE_NAMES:
                    if mc in row.index and pd.notna(row[mc]):
                        features[mc] = row[mc]
                v39_features_list.append(features)
            except Exception:
                continue

        df = pd.DataFrame(v39_features_list)
        logger.info(f"  v39 解析成功: {len(df):,}, {len(df.columns)} 列")

        # --- Step 2: 加载 v40 精选特征 ---
        logger.info(f"  加载 v40 精选特征 ({len(self.v40_features)} 个)...")

        # v40 的日期范围 (可能比 v39 短)
        v40_date_filter = ""
        v40_params = []
        if start_date:
            v40_date_filter += " AND trade_date >= ?"
            v40_params.append(start_date)
        if end_date:
            v40_date_filter += " AND trade_date <= ?"
            v40_params.append(end_date)

        query_v40 = f"""
        SELECT code, trade_date, features_json
        FROM v40_feature_cache
        WHERE 1=1 {v40_date_filter}
        """
        df_v40_raw = pd.read_sql(query_v40, conn, params=v40_params if v40_params else None)
        logger.info(f"  v40 记录: {len(df_v40_raw):,}")

        # 解析 v40 并提取精选特征
        v40_records = []
        for _, row in tqdm(df_v40_raw.iterrows(), total=len(df_v40_raw), desc="解析v40特征"):
            try:
                v40_all = json.loads(row['features_json'])
                record = {'code': row['code'], 'trade_date': row['trade_date']}
                for feat in self.v40_features:
                    record[f'v40_{feat}'] = v40_all.get(feat, np.nan)
                v40_records.append(record)
            except Exception:
                continue

        df_v40 = pd.DataFrame(v40_records)
        logger.info(f"  v40 解析成功: {len(df_v40):,}")

        # --- Step 3: JOIN v39 + v40 ---
        before_merge = len(df)
        df = df.merge(df_v40, on=['code', 'trade_date'], how='inner')
        logger.info(f"  v39+v40 JOIN: {before_merge:,} → {len(df):,} (inner join)")

        # --- Step 4: 加载 daily_basic ---
        logger.info("  加载 daily_basic 额外特征...")
        date_min = df['trade_date'].min()
        date_max = df['trade_date'].max()
        query_basic = """
        SELECT s.code, db.trade_date,
               db.pe_ttm, db.pb, db.ps_ttm, db.turnover_rate, db.circ_mv
        FROM daily_basic db
        JOIN securities s ON db.security_id = s.id
        WHERE db.trade_date >= ? AND db.trade_date <= ?
        """
        df_basic = pd.read_sql(query_basic, conn, params=[date_min, date_max])
        conn.close()
        logger.info(f"  daily_basic 记录: {len(df_basic):,}")

        df = df.merge(df_basic, on=['code', 'trade_date'], how='left')
        df['log_market_cap'] = np.log1p(df['circ_mv'].fillna(0))
        df.drop(columns=['circ_mv'], inplace=True, errors='ignore')
        for col in ['pe_ttm', 'pb', 'ps_ttm', 'turnover_rate', 'log_market_cap']:
            missing = df[col].isnull().sum()
            if missing > 0:
                df[col] = df[col].fillna(df[col].median())

        # --- Step 5 (optional): 加载 GRU neural embeddings ---
        if self.include_neural:
            df = self._load_neural_embeddings(df)

        # --- Step 6: 行业超额收益标签 ---
        if 'sw_l1_code' in df.columns:
            logger.info("  计算行业超额收益标签...")
            for label_col in ['label_3d', 'label_5d', 'label_10d']:
                industry_median = df.groupby(['trade_date', 'sw_l1_code'])[label_col].transform('median')
                raw_mean = df[label_col].mean()
                df[label_col] = df[label_col] - industry_median
                excess_mean = df[label_col].mean()
                logger.info(f"    {label_col}: raw={raw_mean:.6f} → excess={excess_mean:.6f}")

        # --- Step 7: 标签 Winsorization (bounds将在split后仅从训练集计算) ---
        # 注意: 此处仅做占位标记, 实际winsorize在split_data()中执行以避免数据泄漏
        logger.info("  标签 winsorization 将在 train/test split 后执行(防泄漏)")

        # --- Step 8: 缺失值处理 ---
        # 市场特征 ffill
        market_cols = [c for c in MACRO_FEATURE_NAMES if c in df.columns]
        df = df.sort_values('trade_date')
        df[market_cols] = df[market_cols].ffill()
        df = df.dropna(subset=market_cols)

        # v40 特征: 缺失填0
        v40_cols = [c for c in df.columns if c.startswith('v40_')]
        df[v40_cols] = df[v40_cols].fillna(0)

        # 其他缺失填0
        df = df.fillna(0)

        logger.info(f"\n  最终数据集: {len(df):,} 样本, {len(df.columns)} 列")
        logger.info(f"  日期范围: {df['trade_date'].min()} ~ {df['trade_date'].max()}")

        return df

    def _load_neural_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """加载 GRU neural embeddings (Phase 3)"""
        try:
            from ml_models.neural.embedding_cache_manager import EmbeddingCacheManager
            cache_mgr = EmbeddingCacheManager(self.db_path)

            dates = df['trade_date'].unique().tolist()
            logger.info(f"  加载 GRU embeddings ({len(dates)} 天)...")

            emb_df = cache_mgr.load_date_range(
                min(dates), max(dates), model_version='gru_v1'
            )

            if emb_df is not None and len(emb_df) > 0:
                before = len(df)
                df = df.merge(emb_df, on=['code', 'trade_date'], how='left')

                # 嵌入列名: gru_emb_0 .. gru_emb_15
                emb_cols = [c for c in df.columns if c.startswith('gru_emb_')]
                df[emb_cols] = df[emb_cols].fillna(0)

                logger.info(f"    GRU embeddings: {len(emb_cols)} 维, "
                           f"覆盖率 {(df[emb_cols[0]] != 0).sum() / len(df) * 100:.1f}%")
            else:
                logger.warning("    GRU embeddings 无数据, 跳过")
        except ImportError:
            logger.warning("    neural 模块未安装, 跳过 GRU embeddings")
        except Exception as e:
            logger.warning(f"    GRU embeddings 加载失败: {e}")

        return df

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """准备特征: 分层归一化策略"""
        logger.info("\n" + "=" * 80)
        logger.info("V5.0 特征准备: 分层归一化")
        logger.info("=" * 80)

        exclude_cols = {'code', 'trade_date', 'label_3d', 'label_5d', 'label_10d', 'label_15d'}
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        # 分类特征
        self.macro_feature_cols = [c for c in MACRO_FEATURE_NAMES if c in feature_cols]

        # v40 rank features (0-1, skip z-score)
        self.v40_rank_cols = [f'v40_{f}' for f in V40_RANK_FEATURES if f'v40_{f}' in feature_cols]

        # v40 continuous features (need z-score)
        self.v40_continuous_cols = [f'v40_{f}' for f in V40_CONTINUOUS_FEATURES if f'v40_{f}' in feature_cols]

        # Neural embedding features (skip z-score)
        self.neural_cols = [c for c in feature_cols if c.startswith('gru_emb_')]

        # v39 stock features = everything else
        skip_set = set(self.macro_feature_cols + self.v40_rank_cols +
                      self.v40_continuous_cols + self.neural_cols)
        self.stock_feature_cols = [c for c in feature_cols if c not in skip_set]

        logger.info(f"  v39 股票特征: {len(self.stock_feature_cols)} (Robust Z-Score)")
        logger.info(f"  v40 排名特征: {len(self.v40_rank_cols)} (已归一化, 跳过)")
        logger.info(f"  v40 连续特征: {len(self.v40_continuous_cols)} (Robust Z-Score)")
        logger.info(f"  宏观特征: {len(self.macro_feature_cols)} (保持原值)")
        logger.info(f"  Neural特征: {len(self.neural_cols)} (跳过)")
        logger.info(f"  总特征数: {len(feature_cols)}")

        # --- Robust Z-Score on v39 stock features + v40 continuous features ---
        zscore_cols = self.stock_feature_cols + self.v40_continuous_cols
        logger.info(f"  截面 Robust Z-Score: {len(zscore_cols)} 个特征")

        zscore_data = df[zscore_cols].values.copy()
        dates_arr = df['trade_date'].values
        unique_dates = np.unique(dates_arr)

        for d in tqdm(unique_dates, desc="Robust Z-Score", leave=False):
            mask = dates_arr == d
            chunk = zscore_data[mask]
            median = np.nanmedian(chunk, axis=0)
            mad = np.nanmedian(np.abs(chunk - median), axis=0) * 1.4826
            mad[mad < 1e-8] = 1e-8
            zscore_data[mask] = np.clip((chunk - median) / mad, -3, 3)

        df[zscore_cols] = zscore_data
        df[zscore_cols] = df[zscore_cols].fillna(0.0)

        self.feature_names = feature_cols

        X = df[feature_cols].values
        y_3d = df['label_3d'].values
        y_5d = df['label_5d'].values
        y_10d = df['label_10d'].values

        # 特征 winsorization 推迟到 split_data() 中仅用训练集bounds
        logger.info(f"  特征 winsorization 将在 train/test split 后执行(防泄漏)")
        self.winsorize_bounds = None  # 占位, split_data中赋值

        return X, y_3d, y_5d, y_10d, df

    def _winsorize_features(self, X, lower_pct=1, upper_pct=99):
        """Per-feature winsorization (1st-99th percentile)"""
        X_w = X.copy()
        bounds = []
        for i in range(X.shape[1]):
            col = X[:, i]
            valid = col[~np.isnan(col)]
            if len(valid) == 0:
                bounds.append((0.0, 0.0))
                continue
            lo = float(np.percentile(valid, lower_pct))
            hi = float(np.percentile(valid, upper_pct))
            if lo == hi:
                bounds.append((lo, hi))
                continue
            X_w[:, i] = np.clip(col, lo, hi)
            bounds.append((lo, hi))
        return X_w, bounds

    def split_data(self, X, y_3d, y_5d, y_10d, df,
                   val_ratio=0.15, test_ratio=0.15, purge_days=10):
        """时间序列划分 + Purge Gap"""
        logger.info(f"\n划分数据集 (purge_gap={purge_days}天)...")

        dates = df['trade_date'].values
        unique_dates = np.sort(np.unique(dates))
        n_dates = len(unique_dates)

        train_end_idx = int(n_dates * (1 - val_ratio - test_ratio)) - 1
        val_end_idx = int(n_dates * (1 - test_ratio)) - 1

        train_date_end = unique_dates[train_end_idx]
        val_date_start = unique_dates[min(train_end_idx + 1 + purge_days, n_dates - 1)]
        val_date_end = unique_dates[val_end_idx]
        test_date_start = unique_dates[min(val_end_idx + 1 + purge_days, n_dates - 1)]

        train_mask = dates <= train_date_end
        val_mask = (dates >= val_date_start) & (dates <= val_date_end)
        test_mask = dates >= test_date_start

        self.val_dates = dates[val_mask]
        self.test_dates = dates[test_mask]

        result = {}
        for name, mask in [('train', train_mask), ('val', val_mask), ('test', test_mask)]:
            result[f'X_{name}'] = X[mask]
            result[f'y3d_{name}'] = y_3d[mask]
            result[f'y5d_{name}'] = y_5d[mask]
            result[f'y10d_{name}'] = y_10d[mask]

        # --- 特征 Winsorization: 仅用训练集计算bounds, 应用到所有集 ---
        result['X_train'], self.winsorize_bounds = self._winsorize_features(result['X_train'])
        for split_name in ['val', 'test']:
            X_s = result[f'X_{split_name}']
            for i, (lo, hi) in enumerate(self.winsorize_bounds):
                if lo != hi:
                    X_s[:, i] = np.clip(X_s[:, i], lo, hi)
            result[f'X_{split_name}'] = X_s
        logger.info(f"  特征 winsorization (train-only bounds): {len(self.winsorize_bounds)} 列")

        # --- 标签 Winsorization: 仅用训练集计算bounds ---
        self.label_winsorize_bounds = {}
        for label_key in ['y3d', 'y5d', 'y10d']:
            train_y = result[f'{label_key}_train']
            lo = float(np.percentile(train_y, 1))
            hi = float(np.percentile(train_y, 99))
            n_clipped = int(((train_y < lo) | (train_y > hi)).sum())
            self.label_winsorize_bounds[label_key] = (lo, hi)
            for split_name in ['train', 'val', 'test']:
                result[f'{label_key}_{split_name}'] = np.clip(result[f'{label_key}_{split_name}'], lo, hi)
            logger.info(f"  标签winsorize {label_key}: [{lo:.4f}, {hi:.4f}], train裁剪{n_clipped:,}个")

        logger.info(f"  训练集: {len(result['X_train']):,}, <= {train_date_end}")
        logger.info(f"  验证集: {len(result['X_val']):,}, {val_date_start} ~ {val_date_end}")
        logger.info(f"  测试集: {len(result['X_test']):,}, >= {test_date_start}")

        purged = len(X) - len(result['X_train']) - len(result['X_val']) - len(result['X_test'])
        logger.info(f"  Purge丢弃: {purged:,} 样本")

        return result

    def train_single_target_models(self, X_train, X_val, y_train, y_val, target_name: str):
        """为单个目标训练 5 个基础模型"""
        logger.info(f"\n训练 {target_name} 模型...")

        models = {}
        predictions_val = {}

        # 1. LightGBM
        logger.info(f"  LightGBM ({target_name})...")
        lgb_params = {
            'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
            'num_leaves': 31, 'learning_rate': 0.05,
            'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
            'reg_alpha': 0.1, 'reg_lambda': 0.1, 'verbose': -1,
        }
        lgb_train = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train, free_raw_data=True)
        lgb_model = lgb.train(
            lgb_params, lgb_train, num_boost_round=500,
            valid_sets=[lgb_train, lgb_val],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        models['lgb'] = lgb_model
        predictions_val['lgb'] = lgb_model.predict(X_val)
        del lgb_train, lgb_val; gc.collect()

        # 2. XGBoost
        logger.info(f"  XGBoost ({target_name})...")
        xgb_params = {
            'objective': 'reg:squarederror', 'eval_metric': 'rmse',
            'max_depth': 6, 'learning_rate': 0.05,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'reg_alpha': 0.1, 'reg_lambda': 0.1, 'verbosity': 0,
        }
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        xgb_model = xgb.train(
            xgb_params, dtrain, num_boost_round=500,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50, verbose_eval=False
        )
        models['xgb'] = xgb_model
        predictions_val['xgb'] = xgb_model.predict(dval)
        del dtrain, dval; gc.collect()

        # 3. CatBoost
        if HAS_CATBOOST:
            logger.info(f"  CatBoost ({target_name})...")
            cb_model = cb.CatBoostRegressor(
                iterations=500, learning_rate=0.05, depth=6,
                l2_leaf_reg=3, random_seed=42, verbose=False,
                early_stopping_rounds=50
            )
            cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
            models['cb'] = cb_model
            predictions_val['cb'] = cb_model.predict(X_val)

        # 4. RandomForest
        logger.info(f"  RandomForest ({target_name}, max_samples=200000)...")
        rf_model = RandomForestRegressor(
            n_estimators=100, max_depth=12,
            max_samples=min(200_000, X_train.shape[0]),
            max_features=0.8, min_samples_leaf=20,
            n_jobs=-1, random_state=42, verbose=0
        )
        rf_model.fit(X_train, y_train)
        models['rf'] = rf_model
        predictions_val['rf'] = rf_model.predict(X_val)

        # 5. HistGradientBoosting
        logger.info(f"  HistGradientBoosting ({target_name})...")
        hgb_model = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, max_depth=6, max_leaf_nodes=31,
            l2_regularization=0.1, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=50, random_state=42, verbose=0
        )
        hgb_model.fit(X_train, y_train)
        models['hgb'] = hgb_model
        predictions_val['hgb'] = hgb_model.predict(X_val)

        return models, predictions_val

    def calculate_ensemble_weights(self, predictions_val: dict, y_val) -> dict:
        """IC-based Ensemble 权重"""
        val_dates = getattr(self, 'val_dates', None)

        mean_ics = {}
        for name, pred in predictions_val.items():
            if val_dates is not None:
                daily_ics = []
                for date in np.unique(val_dates):
                    mask = val_dates == date
                    if mask.sum() < 10:
                        continue
                    ic, _ = spearmanr(pred[mask], y_val[mask])
                    if not np.isnan(ic):
                        daily_ics.append(ic)
                mean_ics[name] = float(np.mean(daily_ics)) if daily_ics else 0.0
            else:
                ic, _ = spearmanr(pred, y_val)
                mean_ics[name] = float(ic) if not np.isnan(ic) else 0.0

        logger.info(f"  模型IC: {', '.join(f'{k}={v:.4f}' for k, v in mean_ics.items())}")

        positive_ics = {k: v for k, v in mean_ics.items() if v > 0}
        if not positive_ics:
            n = len(mean_ics)
            weights = {k: 1.0 / n for k in mean_ics}
        else:
            total = sum(positive_ics.values())
            weights = {k: positive_ics.get(k, 0.0) / total for k in mean_ics}

        return weights, mean_ics

    def evaluate(self, split, all_results):
        """测试集评估: Daily IC, ICIR, Top-10%/20%"""
        logger.info("\n" + "=" * 80)
        logger.info("测试集评估")
        logger.info("=" * 80)

        X_test = split['X_test']
        test_dates = self.test_dates
        ensemble_predictions = {}

        for target_key in ['3d', '5d', '10d']:
            y_test = split[f'y{target_key}_test']
            pred_test = {}
            for name, model in all_results[target_key]['models'].items():
                if name == 'xgb':
                    pred_test[name] = model.predict(xgb.DMatrix(X_test))
                elif name == 'lgb':
                    pred_test[name] = model.predict(X_test)
                else:
                    pred_test[name] = model.predict(X_test)

            weights = all_results[target_key]['weights']
            ensemble_pred = np.zeros(len(X_test))
            for name, pred in pred_test.items():
                ensemble_pred += weights[name] * pred
            ensemble_predictions[target_key] = ensemble_pred

            # Daily IC / ICIR
            daily_ics = []
            for date in np.unique(test_dates):
                mask = test_dates == date
                if mask.sum() < 20:
                    continue
                ic, _ = spearmanr(ensemble_pred[mask], y_test[mask])
                if not np.isnan(ic):
                    daily_ics.append(ic)

            if daily_ics:
                mean_ic = np.mean(daily_ics)
                ic_ir = mean_ic / np.std(daily_ics) if np.std(daily_ics) > 0 else 0
                ic_pos = np.mean(np.array(daily_ics) > 0)
                logger.info(f"  {target_key}: IC={mean_ic:.4f}, ICIR={ic_ir:.4f}, IC>0={ic_pos:.1%}")

            # Top-10/20 excess return
            daily_top10_excess = []
            daily_top20_excess = []
            for date in np.unique(test_dates):
                mask = test_dates == date
                if mask.sum() < 20:
                    continue
                pred_day = ensemble_pred[mask]
                actual_day = y_test[mask]
                n = len(pred_day)
                top10_idx = np.argsort(pred_day)[-max(1, int(n * 0.1)):]
                top20_idx = np.argsort(pred_day)[-max(1, int(n * 0.2)):]
                daily_top10_excess.append(np.mean(actual_day[top10_idx]))
                daily_top20_excess.append(np.mean(actual_day[top20_idx]))

            if daily_top10_excess:
                logger.info(f"    Top-10% excess: {np.mean(daily_top10_excess):.4f}")
                logger.info(f"    Top-20% excess: {np.mean(daily_top20_excess):.4f}")

        # Fused prediction (5d evaluation)
        fused = (self.target_weights['label_3d'] * ensemble_predictions['3d'] +
                 self.target_weights['label_5d'] * ensemble_predictions['5d'] +
                 self.target_weights['label_10d'] * ensemble_predictions['10d'])

        daily_ics = []
        for date in np.unique(test_dates):
            mask = test_dates == date
            if mask.sum() < 20:
                continue
            ic, _ = spearmanr(fused[mask], split['y5d_test'][mask])
            if not np.isnan(ic):
                daily_ics.append(ic)

        if daily_ics:
            logger.info(f"\n  融合(→5d): IC={np.mean(daily_ics):.4f}, "
                       f"ICIR={np.mean(daily_ics)/np.std(daily_ics):.4f}")

    def _log_feature_importance(self, all_results: dict, top_n: int = 25):
        """特征重要性分析: 关注 v40 新增特征是否进入 Top-30"""
        logger.info("\n" + "=" * 80)
        logger.info("特征重要性分析")
        logger.info("=" * 80)

        for target_key, result in all_results.items():
            models = result.get('models', {})

            avg_importance = np.zeros(len(self.feature_names))
            n_models = 0

            for model_name, model in models.items():
                importance = None
                try:
                    if model_name == 'lgb' and hasattr(model, 'feature_importance'):
                        importance = model.feature_importance(importance_type='gain')
                    elif model_name == 'xgb':
                        score = model.get_score(importance_type='gain')
                        importance = np.zeros(len(self.feature_names))
                        for feat, val in score.items():
                            idx = int(feat.replace('f', ''))
                            if idx < len(importance):
                                importance[idx] = val
                    elif hasattr(model, 'feature_importances_'):
                        importance = model.feature_importances_
                except Exception:
                    continue

                if importance is not None and len(importance) == len(self.feature_names):
                    avg_importance += importance
                    n_models += 1

            if n_models > 0:
                avg_importance /= n_models
                feat_imp = sorted(zip(self.feature_names, avg_importance),
                                key=lambda x: x[1], reverse=True)

                logger.info(f"\n  {target_key} Top {top_n} (avg over {n_models} models):")
                v40_in_top = 0
                for rank, (feat, imp) in enumerate(feat_imp[:top_n], 1):
                    tag = " [v40]" if feat.startswith('v40_') else ""
                    tag += " [gru]" if feat.startswith('gru_emb_') else ""
                    if feat.startswith('v40_') and rank <= 30:
                        v40_in_top += 1
                    logger.info(f"    {rank:2d}. {feat:<40s} {imp:.4f}{tag}")

                logger.info(f"    v40特征在Top-30中: {v40_in_top}")

    def save_model(self, all_results: dict):
        """保存模型"""
        output_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v500'
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        model_data = {
            'models': all_results,
            'feature_names': self.feature_names,
            'target_weights': self.target_weights,
            'winsorize_bounds': self.winsorize_bounds,
            # V5.0 metadata
            'version': 'v5.0',
            'robust_zscore': True,
            'industry_excess_labels': True,
            'cascade': False,
            'rank_normalized': False,
            'dual_stream': False,
            # Feature classification
            'stock_feature_cols': self.stock_feature_cols,
            'macro_feature_cols': self.macro_feature_cols,
            'v40_rank_cols': self.v40_rank_cols,
            'v40_continuous_cols': self.v40_continuous_cols,
            'neural_cols': self.neural_cols or [],
            'v40_selected_features': self.v40_features,
            'extra_features_from_daily_basic': ['pe_ttm', 'pb', 'ps_ttm', 'turnover_rate', 'log_market_cap'],
            'include_neural': self.include_neural,
        }

        model_path = output_dir / f'v500_unified_{timestamp}.pkl'
        joblib.dump(model_data, model_path)
        logger.info(f"\n模型已保存: {model_path}")

        # latest link
        latest_path = output_dir / 'v500_unified_latest.pkl'
        joblib.dump(model_data, latest_path)

        # Training history
        history_path = output_dir / f'training_history_{timestamp}.json'
        history = {
            'version': 'v5.0-unified-fusion',
            'timestamp': timestamp,
            'feature_count': len(self.feature_names),
            'v39_stock_features': len(self.stock_feature_cols),
            'v40_rank_features': len(self.v40_rank_cols),
            'v40_continuous_features': len(self.v40_continuous_cols),
            'neural_features': len(self.neural_cols or []),
            'macro_features': len(self.macro_feature_cols),
            'target_weights': self.target_weights,
        }
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        return str(model_path)

    def train(self, start_date: str = None, end_date: str = None, purge_days: int = 10):
        """完整训练流程"""
        start_time = datetime.now()
        logger.info("=" * 80)
        logger.info("V5.0 Unified Feature Fusion 训练")
        logger.info(f"  v39基础 + v40精选({len(self.v40_features)}个) + daily_basic")
        if self.include_neural:
            logger.info("  + GRU Neural Embeddings")
        logger.info("=" * 80)

        # 1. 加载数据
        df = self.load_data(start_date, end_date)

        # 2. 准备特征
        X, y_3d, y_5d, y_10d, df = self.prepare_features(df)

        # 3. 划分数据
        split = self.split_data(X, y_3d, y_5d, y_10d, df, purge_days=purge_days)

        # 4. 独立训练 3d/5d/10d
        logger.info("\n" + "=" * 80)
        logger.info("独立模型训练: 3d, 5d, 10d")
        logger.info("=" * 80)

        all_results = {}
        targets = [
            ('3d', 'y3d_train', 'y3d_val', 'y3d_test'),
            ('5d', 'y5d_train', 'y5d_val', 'y5d_test'),
            ('10d', 'y10d_train', 'y10d_val', 'y10d_test'),
        ]

        for target_key, y_tr_key, y_va_key, y_te_key in targets:
            models, pred_val = self.train_single_target_models(
                split['X_train'], split['X_val'],
                split[y_tr_key], split[y_va_key], target_key
            )
            weights, ics = self.calculate_ensemble_weights(pred_val, split[y_va_key])
            all_results[target_key] = {
                'models': models, 'weights': weights, 'ics': ics
            }

        # 5. 评估
        self.evaluate(split, all_results)

        # 6. 特征重要性
        self._log_feature_importance(all_results)

        # 7. 保存
        model_path = self.save_model(all_results)

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"\nV5.0 训练完成! 总耗时: {duration:.0f}秒")

        return model_path


    def optuna_optimize(self, X_train, X_val, y_train, y_val, n_trials=50):
        """Phase 4: Optuna 超参数优化 (优化 LightGBM 5d 目标的 IC)"""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("Optuna required. Install: pip install optuna")

        logger.info(f"\n{'='*80}")
        logger.info(f"Phase 4: Optuna 超参数优化 ({n_trials} trials)")
        logger.info(f"{'='*80}")

        val_dates = self.val_dates

        def objective(trial):
            params = {
                'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True),
                'max_depth': trial.suggest_int('max_depth', 4, 8),
                'verbose': -1,
            }

            lgb_train = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
            lgb_val_ds = lgb.Dataset(X_val, label=y_val, reference=lgb_train, free_raw_data=True)

            model = lgb.train(
                params, lgb_train, num_boost_round=500,
                valid_sets=[lgb_val_ds],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            pred = model.predict(X_val)

            # Compute daily IC
            daily_ics = []
            if val_dates is not None:
                for date in np.unique(val_dates):
                    mask = val_dates == date
                    if mask.sum() < 10:
                        continue
                    ic, _ = spearmanr(pred[mask], y_val[mask])
                    if not np.isnan(ic):
                        daily_ics.append(ic)

            return np.mean(daily_ics) if daily_ics else 0.0

        study = optuna.create_study(direction='maximize',
                                     sampler=optuna.samplers.TPESampler(seed=42),
                                     pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        logger.info(f"\n  最优 IC: {study.best_value:.4f}")
        logger.info(f"  最优参数: {study.best_params}")

        # Save Optuna results
        output_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v500'
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(output_dir / f'optuna_results_{timestamp}.json', 'w') as f:
            json.dump({
                'best_ic': study.best_value,
                'best_params': study.best_params,
                'n_trials': n_trials,
            }, f, indent=2)

        return study.best_params


def main():
    parser = argparse.ArgumentParser(description='V5.0 Unified Feature Fusion 训练')
    parser.add_argument('--start-date', type=str, default='2022-01-01',
                       help='训练开始日期 (v40数据从2022-01-04开始)')
    parser.add_argument('--end-date', type=str, default=None, help='训练结束日期')
    parser.add_argument('--purge-days', type=int, default=10, help='Purge gap天数')
    parser.add_argument('--include-neural', action='store_true',
                       help='包含 GRU neural embeddings (Phase 3)')
    parser.add_argument('--v40-features', type=str, default=None,
                       help='JSON文件: 自定义v40特征列表 (覆盖默认)')
    parser.add_argument('--optuna', action='store_true',
                       help='Phase 4: 运行 Optuna 超参数优化 (50 trials)')
    parser.add_argument('--optuna-trials', type=int, default=50,
                       help='Optuna trial 数量')
    args = parser.parse_args()

    v40_features = None
    if args.v40_features:
        with open(args.v40_features) as f:
            v40_features = json.load(f)

    trainer = V500UnifiedTrainer(
        v40_features=v40_features,
        include_neural=args.include_neural,
    )

    if args.optuna:
        # Phase 4: 先加载数据和划分, 然后运行 Optuna
        df = trainer.load_data(args.start_date, args.end_date)
        X, y_3d, y_5d, y_10d, df = trainer.prepare_features(df)
        split = trainer.split_data(X, y_3d, y_5d, y_10d, df, purge_days=args.purge_days)
        best_params = trainer.optuna_optimize(
            split['X_train'], split['X_val'],
            split['y5d_train'], split['y5d_val'],
            n_trials=args.optuna_trials
        )
        logger.info(f"\n最优参数已保存. 可手动更新 train_single_target_models() 中的参数后重新训练.")
    else:
        trainer.train(
            start_date=args.start_date,
            end_date=args.end_date,
            purge_days=args.purge_days,
        )


if __name__ == '__main__':
    main()
