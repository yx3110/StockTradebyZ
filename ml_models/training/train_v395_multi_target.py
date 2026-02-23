#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.9.5 多目标 + 市场状态特征 训练脚本

核心改进 (v2 — Robust Z-Score + Industry-Excess Labels):
1. 多目标预测: 同时预测3天、5天、10天收益，使用加权融合
2. 市场状态特征: 加入大盘20日收益率、波动率、上涨比例、回撤等
3. 优化Ensemble: 基于IC的加权Ensemble架构
4. Robust Z-Score: 截面归一化保留幅度信息 (替代Rank归一化)
5. Industry-Excess Labels: 行业超额收益标签, 模型专注学习选股Alpha
6. 新增特征: pe_ttm, pb, ps_ttm, turnover_rate, log_market_cap (from daily_basic)
7. 独立训练: 各目标独立训练 (不使用级联)

作者: Claude Code
创建时间: 2025-12-27
"""

import sys
import numpy as np
import pandas as pd
import sqlite3
import json
from datetime import datetime, timedelta
import logging
from pathlib import Path
import joblib
from tqdm import tqdm
import argparse
import warnings

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings('ignore')

# ML库
import lightgbm as lgb
import xgboost as xgb
try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("Warning: CatBoost not installed")

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')


class MarketStateCalculator:
    """市场状态特征计算器"""

    def __init__(self, db_path: str = DB_PATH, lookback: int = 20):
        self.db_path = db_path
        self.lookback = lookback
        self.market_features = None

    def calculate_market_features(self) -> pd.DataFrame:
        """计算所有日期的市场状态特征"""
        logger.info("计算市场状态特征...")

        conn = sqlite3.connect(self.db_path)

        # 获取上证指数数据
        query = """
        SELECT q.trade_date, q.close, q.price_change_pct, q.volume
        FROM daily_quotes q
        JOIN securities s ON q.security_id = s.id
        WHERE s.code = '000001.SH'
        ORDER BY q.trade_date
        """
        df = pd.read_sql(query, conn)
        conn.close()

        # 计算市场状态特征
        lookback = self.lookback

        # 1. 市场收益率
        df['market_return_20d'] = df['close'].pct_change(lookback)
        df['market_return_10d'] = df['close'].pct_change(10)
        df['market_return_5d'] = df['close'].pct_change(5)

        # 2. 市场波动率
        df['market_volatility_20d'] = df['price_change_pct'].rolling(lookback).std()
        df['market_volatility_10d'] = df['price_change_pct'].rolling(10).std()

        # 3. 上涨天数比例
        df['market_up_ratio_20d'] = (df['price_change_pct'] > 0).rolling(lookback).mean()
        df['market_up_ratio_10d'] = (df['price_change_pct'] > 0).rolling(10).mean()

        # 4. 最大回撤
        df['market_max_20d'] = df['close'].rolling(lookback).max()
        df['market_drawdown_20d'] = (df['close'] / df['market_max_20d'] - 1)

        # 5. 成交量变化
        df['market_volume_ratio'] = df['volume'] / df['volume'].rolling(lookback).mean()

        # 6. 趋势强度 (简化版：使用价格相对位置)
        df['market_min_20d'] = df['close'].rolling(lookback).min()
        df['market_position_20d'] = (df['close'] - df['market_min_20d']) / \
                                     (df['market_max_20d'] - df['market_min_20d'] + 1e-8)

        # 7. 市场动量
        df['market_momentum_20d'] = df['close'] / df['close'].shift(lookback) - 1
        df['market_momentum_5d'] = df['close'] / df['close'].shift(5) - 1

        # 选择最终特征
        market_features = df[['trade_date',
                              'market_return_20d', 'market_return_10d', 'market_return_5d',
                              'market_volatility_20d', 'market_volatility_10d',
                              'market_up_ratio_20d', 'market_up_ratio_10d',
                              'market_drawdown_20d', 'market_volume_ratio',
                              'market_position_20d', 'market_momentum_20d', 'market_momentum_5d'
                             ]].copy()

        self.market_features = market_features.dropna()
        logger.info(f"  市场状态特征: {len(self.market_features)} 个交易日, {len(market_features.columns)-1} 个特征")

        return self.market_features


class V395MultiTargetTrainer:
    """V3.9.5 多目标训练器"""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.market_calculator = MarketStateCalculator(db_path)
        self.models = {}
        self.scaler = None
        self.feature_names = None

        self.winsorize_bounds = None

        # 多目标权重 (可调整)
        self.target_weights = {
            'label_3d': 0.4,   # 短期收益权重最高
            'label_5d': 0.35,  # 中期收益
            'label_10d': 0.25  # 长期收益
        }

    def load_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """加载训练数据"""
        logger.info("加载训练数据...")

        conn = sqlite3.connect(self.db_path)

        # 构建日期过滤
        date_filter = ""
        if start_date:
            date_filter += f" AND v.trade_date >= '{start_date}'"
        if end_date:
            date_filter += f" AND v.trade_date <= '{end_date}'"

        # 过滤:
        # 1. 三个标签都非空
        # 2. 排除停牌日 (volume=0)
        # 3. 排除交易日 <30 天的低历史股票
        query = f"""
        SELECT
            v.code, v.trade_date, v.features_json,
            v.label_3d, v.label_5d, v.label_10d
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

        df = pd.read_sql(query, conn)
        conn.close()

        logger.info(f"  原始记录: {len(df):,} (已过滤停牌日+低历史股票)")

        # 解析特征JSON
        features_list = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="解析特征"):
            try:
                features = json.loads(row['features_json'])
                features['code'] = row['code']
                features['trade_date'] = row['trade_date']
                features['label_3d'] = row['label_3d']
                features['label_5d'] = row['label_5d']
                features['label_10d'] = row['label_10d']
                features_list.append(features)
            except:
                continue

        df_features = pd.DataFrame(features_list)
        logger.info(f"  解析成功: {len(df_features):,}")

        # 合并市场状态特征
        market_features = self.market_calculator.calculate_market_features()
        df_features = df_features.merge(market_features, on='trade_date', how='left')

        # 缺失值处理: 市场特征用ffill(同日期的市场状态相同)，其余用0填充
        market_cols = [c for c in market_features.columns if c != 'trade_date']
        other_cols = [c for c in df_features.columns if c not in market_cols]

        # 先统计缺失
        missing_count = df_features.isnull().sum().sum()
        total_cells = df_features.shape[0] * df_features.shape[1]
        if missing_count > 0:
            missing_pct = missing_count / total_cells * 100
            logger.warning(f"  检测到 {missing_count:,} 个缺失值 ({missing_pct:.2f}%)")
            col_missing = df_features.isnull().sum()
            high_missing = col_missing[col_missing > 0].sort_values(ascending=False).head(10)
            for col, cnt in high_missing.items():
                logger.warning(f"    {col}: {cnt:,} 缺失 ({cnt/len(df_features)*100:.1f}%)")

        # 市场特征: 按trade_date排序后ffill (同一市场状态延续到下个交易日)
        df_features = df_features.sort_values('trade_date')
        df_features[market_cols] = df_features[market_cols].ffill()

        # 丢弃市场特征仍然为NaN的行 (仅最早几天无法ffill的)
        before_drop = len(df_features)
        df_features = df_features.dropna(subset=market_cols)
        dropped = before_drop - len(df_features)
        if dropped > 0:
            logger.info(f"  丢弃 {dropped:,} 行 (市场特征ffill后仍缺失，通常为最早几天)")

        # 非市场特征的缺失值用0填充
        remaining_missing = df_features.isnull().sum().sum()
        if remaining_missing > 0:
            logger.info(f"  剩余 {remaining_missing:,} 个非市场特征缺失值，用 0 填充")
            df_features = df_features.fillna(0)

        logger.info(f"  合并市场特征后: {len(df_features):,}")

        # ===== 新增: 从 daily_basic 加载额外特征 (pe_ttm, pb, ps_ttm, turnover_rate, circ_mv) =====
        logger.info("  加载 daily_basic 额外特征...")
        conn2 = sqlite3.connect(self.db_path)
        date_min = df_features['trade_date'].min()
        date_max = df_features['trade_date'].max()
        query_basic = f"""
        SELECT s.code, db.trade_date,
               db.pe_ttm, db.pb, db.ps_ttm, db.turnover_rate, db.circ_mv
        FROM daily_basic db
        JOIN securities s ON db.security_id = s.id
        WHERE db.trade_date >= '{date_min}' AND db.trade_date <= '{date_max}'
        """
        df_basic = pd.read_sql(query_basic, conn2)
        conn2.close()
        logger.info(f"    daily_basic 记录: {len(df_basic):,}")

        # 合并额外特征
        df_features = df_features.merge(df_basic, on=['code', 'trade_date'], how='left')

        # 计算 log_market_cap (对数市值, 消除量纲)
        df_features['log_market_cap'] = np.log1p(df_features['circ_mv'].fillna(0))
        df_features.drop(columns=['circ_mv'], inplace=True, errors='ignore')

        # 填充缺失的额外特征 (约5-10%股票可能缺少daily_basic数据)
        for col in ['pe_ttm', 'pb', 'ps_ttm', 'turnover_rate', 'log_market_cap']:
            missing = df_features[col].isnull().sum()
            if missing > 0:
                logger.info(f"    {col}: {missing:,} 缺失 ({missing/len(df_features)*100:.1f}%), 用中位数填充")
                df_features[col] = df_features[col].fillna(df_features[col].median())

        logger.info(f"  额外特征合并完成: +5 (pe_ttm, pb, ps_ttm, turnover_rate, log_market_cap)")

        # ===== 新增: 行业超额收益标签 =====
        # 原理: raw label 包含 市场方向+行业方向+个股Alpha, 模型浪费容量在预测大盘
        # industry-excess = stock_return - industry_median_return, 让模型专注学选股Alpha
        if 'sw_l1_code' in df_features.columns:
            logger.info("  计算行业超额收益标签...")
            for label_col in ['label_3d', 'label_5d', 'label_10d']:
                industry_median = df_features.groupby(['trade_date', 'sw_l1_code'])[label_col].transform('median')
                raw_mean = df_features[label_col].mean()
                df_features[label_col] = df_features[label_col] - industry_median
                excess_mean = df_features[label_col].mean()
                logger.info(f"    {label_col}: raw均值={raw_mean:.6f} → excess均值={excess_mean:.6f}")
        else:
            logger.warning("  sw_l1_code 列不存在, 跳过行业超额标签")

        # ===== 新增: 标签 Winsorization (每列1%/99%) =====
        for label_col in ['label_3d', 'label_5d', 'label_10d']:
            lo = df_features[label_col].quantile(0.01)
            hi = df_features[label_col].quantile(0.99)
            n_clipped = ((df_features[label_col] < lo) | (df_features[label_col] > hi)).sum()
            df_features[label_col] = df_features[label_col].clip(lo, hi)
            logger.info(f"  标签winsorize {label_col}: [{lo:.4f}, {hi:.4f}], 裁剪{n_clipped:,}个")

        return df_features

    def winsorize_features(self, X: np.ndarray, lower_pct: float = 1, upper_pct: float = 99) -> tuple:
        """
        Per-feature winsorization: 将每列裁剪到 [1st, 99th] percentile

        比硬裁剪 np.clip(-10, 10) 更好：
        - 保留极端行情信号的相对排序
        - 适应每个特征的自然尺度
        - 避免将所有异常值映射到同一个值

        Args:
            X: 特征矩阵 (n_samples, n_features)
            lower_pct: 下界百分位
            upper_pct: 上界百分位

        Returns:
            (X_winsorized, bounds): 裁剪后的矩阵和每列的 (lo, hi) bounds
        """
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
                # 避免常量列裁剪
                bounds.append((lo, hi))
                continue
            X_w[:, i] = np.clip(col, lo, hi)
            bounds.append((lo, hi))
        return X_w, bounds

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """准备特征和标签（纯Rank特征: 个股特征截面排名归一化）"""
        logger.info("准备特征和标签...")

        # 排除非特征列
        exclude_cols = ['code', 'trade_date', 'label_3d', 'label_5d', 'label_10d', 'label_15d']
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        # Step 1: 特征分层 — 宏观特征 vs 个股特征
        macro_feature_names = [
            'market_return_20d', 'market_return_10d', 'market_return_5d',
            'market_volatility_20d', 'market_volatility_10d',
            'market_up_ratio_20d', 'market_up_ratio_10d',
            'market_drawdown_20d', 'market_volume_ratio',
            'market_position_20d', 'market_momentum_20d', 'market_momentum_5d',
            'northbound_flow_5d'
        ]
        self.macro_feature_cols = [c for c in macro_feature_names if c in feature_cols]
        self.stock_feature_cols = [c for c in feature_cols if c not in self.macro_feature_cols]

        logger.info(f"  个股特征: {len(self.stock_feature_cols)}, 宏观特征: {len(self.macro_feature_cols)}")

        # Step 2: 截面Robust Z-Score — 个股特征 → (x - median) / (MAD * 1.4826), clip[-3,3]
        # 优势 vs Rank: 保留幅度信息 (PE=5 vs PE=50 有不同z值, 而rank只有序数差)
        # 这对5d/10d/15d预测至关重要: 绝对估值偏离幅度 → 回归潜力大小
        # 宏观特征保持原值 (所有股票同日值相同, z-score无意义)
        logger.info("  截面Robust Z-Score: 个股特征归一化 (保留幅度信息)")

        stock_data = df[self.stock_feature_cols].values.copy()
        dates_arr = df['trade_date'].values
        unique_dates = np.unique(dates_arr)

        for d in tqdm(unique_dates, desc="Robust Z-Score", leave=False):
            mask = dates_arr == d
            chunk = stock_data[mask]
            median = np.nanmedian(chunk, axis=0)
            mad = np.nanmedian(np.abs(chunk - median), axis=0) * 1.4826
            mad[mad < 1e-8] = 1e-8
            stock_data[mask] = np.clip((chunk - median) / mad, -3, 3)

        df[self.stock_feature_cols] = stock_data
        df[self.stock_feature_cols] = df[self.stock_feature_cols].fillna(0.0)
        self.rank_normalized = False
        self.robust_zscore = True
        self.dual_stream = False

        self.feature_names = feature_cols
        logger.info(f"  特征数量: {len(feature_cols)} (rank个股{len(self.stock_feature_cols)}+宏观{len(self.macro_feature_cols)})")

        X = df[feature_cols].values
        y_3d = df['label_3d'].values
        y_5d = df['label_5d'].values
        y_10d = df['label_10d'].values

        # Per-feature winsorization (rank后个股特征在[0,1]不受影响，主要裁剪宏观特征极端值)
        X, self.winsorize_bounds = self.winsorize_features(X)
        logger.info(f"  特征 winsorization: {len(self.winsorize_bounds)} 列, 1st/99th percentile")

        return X, y_3d, y_5d, y_10d, df

    def split_data(self, X, y_3d, y_5d, y_10d, df, val_ratio=0.15, test_ratio=0.15, purge_days=10):
        """
        时间序列划分数据集，带 Purge Gap

        Purge Gap: 在 train/val 和 val/test 边界丢弃 purge_days 个交易日的样本，
        避免标签窗口重叠（label_10d 使用未来10天价格，相邻样本标签高度相关）

        Args:
            purge_days: purge gap 交易日天数（应 >= 最大标签前瞻天数，label_10d 需要10天）
        """
        logger.info("划分数据集 (带 Purge Gap)...")

        # 使用 df 中的 trade_date 来做基于日期的划分
        dates = df['trade_date'].values
        unique_dates = np.sort(np.unique(dates))
        n_dates = len(unique_dates)

        # 按交易日数量划分边界
        train_date_end_idx = int(n_dates * (1 - val_ratio - test_ratio)) - 1
        val_date_end_idx = int(n_dates * (1 - test_ratio)) - 1

        train_date_end = unique_dates[train_date_end_idx]
        val_date_start = unique_dates[min(train_date_end_idx + 1 + purge_days, n_dates - 1)]
        val_date_end = unique_dates[val_date_end_idx]
        test_date_start = unique_dates[min(val_date_end_idx + 1 + purge_days, n_dates - 1)]

        # 按日期筛选样本 mask
        train_mask = dates <= train_date_end
        val_mask = (dates >= val_date_start) & (dates <= val_date_end)
        test_mask = dates >= test_date_start

        X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
        y_3d_train, y_3d_val, y_3d_test = y_3d[train_mask], y_3d[val_mask], y_3d[test_mask]
        y_5d_train, y_5d_val, y_5d_test = y_5d[train_mask], y_5d[val_mask], y_5d[test_mask]
        y_10d_train, y_10d_val, y_10d_test = y_10d[train_mask], y_10d[val_mask], y_10d[test_mask]

        purged_samples = len(X) - len(X_train) - len(X_val) - len(X_test)

        logger.info(f"  训练集: {len(X_train):,} 样本, <= {train_date_end}")
        logger.info(f"  验证集: {len(X_val):,} 样本, {val_date_start} ~ {val_date_end}")
        logger.info(f"  测试集: {len(X_test):,} 样本, >= {test_date_start}")
        logger.info(f"  Purge gap: {purge_days} 个交易日, 丢弃 {purged_samples:,} 个样本")

        # 保存验证集/测试集日期 (用于IC-based权重计算)
        self.val_dates = dates[val_mask]
        self.test_dates = dates[test_mask]

        return (X_train, X_val, X_test,
                y_3d_train, y_3d_val, y_3d_test,
                y_5d_train, y_5d_val, y_5d_test,
                y_10d_train, y_10d_val, y_10d_test)

    def train_single_target_models(self, X_train, X_val, y_train, y_val, target_name: str):
        """为单个目标训练所有基础模型"""
        logger.info(f"\n训练 {target_name} 模型...")

        models = {}
        predictions_train = {}
        predictions_val = {}

        # 1. LightGBM
        logger.info(f"  训练 LightGBM ({target_name})...")
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbose': -1
        }

        lgb_train = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train, free_raw_data=True)

        lgb_model = lgb.train(
            lgb_params, lgb_train,
            num_boost_round=500,
            valid_sets=[lgb_train, lgb_val],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        models['lgb'] = lgb_model
        predictions_train['lgb'] = lgb_model.predict(X_train)
        predictions_val['lgb'] = lgb_model.predict(X_val)
        del lgb_train, lgb_val  # 释放LGB数据集内存
        import gc; gc.collect()

        # 2. XGBoost
        logger.info(f"  训练 XGBoost ({target_name})...")
        xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbosity': 0
        }

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        xgb_model = xgb.train(
            xgb_params, dtrain,
            num_boost_round=500,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        models['xgb'] = xgb_model
        predictions_train['xgb'] = xgb_model.predict(dtrain)
        predictions_val['xgb'] = xgb_model.predict(dval)
        del dtrain, dval  # 释放XGB DMatrix内存
        gc.collect()

        # 3. CatBoost
        if HAS_CATBOOST:
            logger.info(f"  训练 CatBoost ({target_name})...")
            cb_model = cb.CatBoostRegressor(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3,
                random_seed=42,
                verbose=False,
                early_stopping_rounds=50
            )
            cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
            models['cb'] = cb_model
            predictions_train['cb'] = cb_model.predict(X_train)
            predictions_val['cb'] = cb_model.predict(X_val)

        # 4. RandomForest (子采样控制内存: 每棵树用max_samples样本)
        logger.info(f"  训练 RandomForest ({target_name}, max_samples=200000)...")
        n_samples = X_train.shape[0]
        rf_max_samples = min(200_000, n_samples)
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            max_samples=rf_max_samples,
            max_features=0.8,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        rf_model.fit(X_train, y_train)
        models['rf'] = rf_model
        predictions_train['rf'] = rf_model.predict(X_train)
        predictions_val['rf'] = rf_model.predict(X_val)

        # 5. HistGradientBoosting (sklearn直方图版，原生支持大数据集)
        logger.info(f"  训练 HistGradientBoosting ({target_name})...")
        hgb_model = HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.05,
            max_depth=6,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=50,
            random_state=42,
            verbose=0
        )
        hgb_model.fit(X_train, y_train)
        models['hgb'] = hgb_model
        predictions_train['hgb'] = hgb_model.predict(X_train)
        predictions_val['hgb'] = hgb_model.predict(X_val)

        return models, predictions_train, predictions_val

    def calculate_ensemble_weights(self, predictions_val: dict, y_val) -> dict:
        """基于验证集截面IC计算Ensemble权重"""
        val_dates = getattr(self, 'val_dates', None)

        if val_dates is None:
            # 回退到RMSE-based (无日期信息时)
            rmses = {}
            for name, pred in predictions_val.items():
                rmse = np.sqrt(mean_squared_error(y_val, pred))
                rmses[name] = rmse
            inv_rmses = {k: 1/v for k, v in rmses.items()}
            total = sum(inv_rmses.values())
            weights = {k: v/total for k, v in inv_rmses.items()}
            return weights, rmses

        # IC-based: 对每个trade_date计算Spearman IC，取mean_IC作为权重
        unique_dates = np.unique(val_dates)
        mean_ics = {}

        for name, pred in predictions_val.items():
            daily_ics = []
            for date in unique_dates:
                mask = val_dates == date
                n = mask.sum()
                if n < 10:  # 样本太少的日期跳过
                    continue
                ic, _ = spearmanr(pred[mask], y_val[mask])
                if not np.isnan(ic):
                    daily_ics.append(ic)
            mean_ics[name] = float(np.mean(daily_ics)) if daily_ics else 0.0

        logger.info(f"  模型IC: {', '.join(f'{k}={v:.4f}' for k, v in mean_ics.items())}")

        # 只保留IC > 0的模型
        positive_ics = {k: v for k, v in mean_ics.items() if v > 0}
        if not positive_ics:
            logger.warning("  所有模型IC <= 0, 回退到均等权重")
            n = len(mean_ics)
            weights = {k: 1.0 / n for k in mean_ics}
        else:
            total = sum(positive_ics.values())
            weights = {}
            for k in mean_ics:
                weights[k] = positive_ics.get(k, 0.0) / total

        return weights, mean_ics

    def ensemble_predict(self, predictions: dict, weights: dict) -> np.ndarray:
        """加权Ensemble预测"""
        result = np.zeros_like(list(predictions.values())[0])
        for name, pred in predictions.items():
            result += weights[name] * pred
        return result

    def _log_feature_importance(self, all_results: dict, top_n: int = 20):
        """
        提取并打印各目标、各模型的特征重要性

        级联模型中各目标的特征数不同:
        - 3d: 基础特征 (N列)
        - 5d: 基础特征 + cascade_pred_3d (N+1列)
        - 10d: 基础特征 + cascade_pred_3d + cascade_pred_5d (N+2列)

        Args:
            all_results: {'3d': {'models': {...}, ...}, '5d': ..., '10d': ...}
            top_n: 打印的前 N 个特征
        """
        logger.info("\n" + "=" * 60)
        logger.info("📊 特征重要性分析 (级联Rank)")
        logger.info("=" * 60)

        all_importances = {}

        # 获取每个目标的特征名
        cascade_names = getattr(self, 'cascade_feature_names', None)

        for target_name, result in all_results.items():
            # 确定该目标的特征名列表
            if cascade_names and target_name in cascade_names:
                target_feature_names = cascade_names[target_name]
            else:
                target_feature_names = self.feature_names

            models = result.get('models', {})
            for model_name, model in models.items():
                importance = None
                try:
                    if model_name == 'lgb' and hasattr(model, 'feature_importance'):
                        importance = model.feature_importance(importance_type='gain')
                    elif model_name == 'xgb':
                        score = model.get_score(importance_type='gain')
                        if target_feature_names:
                            importance = np.zeros(len(target_feature_names))
                            for feat, val in score.items():
                                idx = int(feat.replace('f', ''))
                                if idx < len(importance):
                                    importance[idx] = val
                    elif hasattr(model, 'feature_importances_'):
                        importance = model.feature_importances_
                except Exception as e:
                    logger.debug(f"  {target_name}/{model_name} 特征重要性提取失败: {e}")
                    continue

                if importance is None or target_feature_names is None:
                    continue

                # 确保importance长度匹配
                if len(importance) != len(target_feature_names):
                    logger.debug(f"  {target_name}/{model_name} 长度不匹配: "
                                 f"importance={len(importance)}, features={len(target_feature_names)}")
                    continue

                key = f"{target_name}_{model_name}"
                feat_imp = sorted(zip(target_feature_names, importance), key=lambda x: x[1], reverse=True)
                all_importances[key] = {f: float(v) for f, v in feat_imp}

                logger.info(f"\n🔹 {key} Top {top_n} ({len(target_feature_names)} features):")
                for rank, (feat, imp) in enumerate(feat_imp[:top_n], 1):
                    logger.info(f"  {rank:2d}. {feat:<35s} {imp:.4f}")

        # 计算全局平均 (只对基础特征)
        if all_importances and self.feature_names:
            avg_importance = {}
            for feat in self.feature_names:
                values = [imp.get(feat, 0) for imp in all_importances.values()]
                avg_importance[feat] = float(np.mean(values))
            avg_sorted = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)

            logger.info(f"\n🔹 全局平均特征重要性 Top {top_n} (基础特征):")
            for rank, (feat, imp) in enumerate(avg_sorted[:top_n], 1):
                logger.info(f"  {rank:2d}. {feat:<35s} {imp:.4f}")

            all_importances['global_average'] = dict(avg_sorted)

        # 级联特征重要性分析
        if cascade_names:
            logger.info(f"\n{'=' * 60}")
            logger.info("📊 级联特征贡献分析")
            logger.info("=" * 60)
            for target_name in ['5d', '10d']:
                cascade_cols = [c for c in (cascade_names.get(target_name, []))
                                if c.startswith('cascade_pred_')]
                if not cascade_cols:
                    continue
                for key, imp_dict in all_importances.items():
                    if not key.startswith(target_name):
                        continue
                    total_imp = sum(imp_dict.values())
                    if total_imp == 0:
                        continue
                    cascade_imp = sum(imp_dict.get(c, 0) for c in cascade_cols)
                    cascade_pct = cascade_imp / total_imp * 100
                    logger.info(f"  {key}: 级联特征贡献 = {cascade_pct:.1f}%")

        # 保存到 JSON
        output_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v395'
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        importance_path = output_dir / f"v395_feature_importance_{timestamp}.json"
        with open(importance_path, 'w', encoding='utf-8') as f:
            json.dump(all_importances, f, indent=2, ensure_ascii=False)
        logger.info(f"\n💾 特征重要性已保存: {importance_path}")

    def _generate_oof_predictions(self, X_train, y_train, train_dates, target_name, n_splits=5):
        """
        生成OOF预测 (用于级联特征, 防止信息泄露)

        使用时间序列分割: 按日期将训练集分为n_splits个fold,
        每个fold用之前的数据训练LightGBM, 预测当前fold.
        第一个fold无历史数据,使用全局均值填充.

        Args:
            X_train: 训练集特征矩阵
            y_train: 训练集标签
            train_dates: 训练集日期数组 (与X_train同长度)
            target_name: 目标名称 (用于日志)
            n_splits: fold数量

        Returns:
            oof_pred: OOF预测数组 (与X_train同长度)
        """
        logger.info(f"  生成 {target_name} OOF预测 ({n_splits}-fold 时间序列分割)...")

        unique_dates = np.sort(np.unique(train_dates))
        oof_pred = np.full(len(X_train), np.mean(y_train))  # 默认用全局均值
        fold_size = len(unique_dates) // n_splits

        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbose': -1
        }

        filled_count = 0
        for i in range(n_splits):
            # 时间序列fold定义
            val_start_idx = i * fold_size
            val_end_idx = (i + 1) * fold_size if i < n_splits - 1 else len(unique_dates)
            val_dates_set = set(unique_dates[val_start_idx:val_end_idx])

            val_mask = np.array([d in val_dates_set for d in train_dates])
            # 训练数据: 所有在当前fold之前的日期
            train_dates_set = set(unique_dates[:val_start_idx])
            fold_train_mask = np.array([d in train_dates_set for d in train_dates])

            n_train = fold_train_mask.sum()
            n_val = val_mask.sum()

            if n_train < 100:
                # 前几个fold训练数据不足,保持全局均值
                logger.info(f"    Fold {i+1}: 训练样本不足({n_train}), 使用全局均值")
                continue

            # 用LightGBM快速训练
            lgb_train = lgb.Dataset(X_train[fold_train_mask], label=y_train[fold_train_mask], free_raw_data=True)
            lgb_val = lgb.Dataset(X_train[val_mask], label=y_train[val_mask], reference=lgb_train, free_raw_data=True)

            fold_model = lgb.train(
                lgb_params, lgb_train,
                num_boost_round=300,
                valid_sets=[lgb_val],
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
            )

            oof_pred[val_mask] = fold_model.predict(X_train[val_mask])
            filled_count += n_val
            del lgb_train, lgb_val, fold_model

        import gc; gc.collect()
        logger.info(f"    OOF预测完成: {filled_count}/{len(X_train)} 样本由模型填充")
        return oof_pred

    def train(self, start_date: str = None, end_date: str = None, purge_days: int = 10):
        """完整训练流程"""
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("V3.95 Robust Z-Score + Industry-Excess 训练 (带 Purge Gap)")
        logger.info("=" * 60)

        # 1. 加载数据 (含daily_basic额外特征 + 行业超额标签)
        df = self.load_data(start_date, end_date)

        # 2. 准备特征 (Robust Z-Score: 保留幅度信息的截面归一化)
        X, y_3d, y_5d, y_10d, df = self.prepare_features(df)

        # 注: 不再需要标签中性化 — 行业超额标签已在load_data()中完成
        # industry-excess labels 移除了市场+行业系统性成分, 模型专注个股Alpha

        # 3. 划分数据（带 purge gap）
        (X_train, X_val, X_test,
         y_3d_train, y_3d_val, y_3d_test,
         y_5d_train, y_5d_val, y_5d_test,
         y_10d_train, y_10d_val, y_10d_test) = self.split_data(X, y_3d, y_5d, y_10d, df, purge_days=purge_days)

        # 4. 独立训练 (各目标独立, 无级联 — 避免噪声级联放大)
        logger.info("\n" + "=" * 60)
        logger.info("独立模型训练: 3d, 5d, 10d")
        logger.info("=" * 60)

        all_results = {}
        targets = [
            ('3d', y_3d_train, y_3d_val, y_3d_test, 'label_3d'),
            ('5d', y_5d_train, y_5d_val, y_5d_test, 'label_5d'),
            ('10d', y_10d_train, y_10d_val, y_10d_test, 'label_10d'),
        ]

        for target_key, y_tr, y_va, y_te, target_name in targets:
            models, pred_train, pred_val = self.train_single_target_models(
                X_train, X_val, y_tr, y_va, target_name)
            weights, rmses = self.calculate_ensemble_weights(pred_val, y_va)
            all_results[target_key] = {'models': models, 'weights': weights, 'rmses': rmses}

        # 5. 测试集评估 (独立推理)
        logger.info("\n" + "=" * 60)
        logger.info("测试集评估 (独立推理)")
        logger.info("=" * 60)

        final_metrics = {}
        ensemble_predictions = {}

        for target_key, y_tr, y_va, y_te, target_name in targets:
            pred_test = {}
            for name, model in all_results[target_key]['models'].items():
                if name == 'lgb':
                    pred_test[name] = model.predict(X_test)
                elif name == 'xgb':
                    pred_test[name] = model.predict(xgb.DMatrix(X_test))
                else:
                    pred_test[name] = model.predict(X_test)

            ensemble_pred = self.ensemble_predict(pred_test, all_results[target_key]['weights'])
            ensemble_predictions[target_key] = ensemble_pred

            rmse = np.sqrt(mean_squared_error(y_te, ensemble_pred))
            ic, _ = spearmanr(ensemble_pred, y_te)
            direction_acc = np.mean((ensemble_pred > 0) == (y_te > 0))
            top10_idx = np.argsort(ensemble_pred)[-int(len(ensemble_pred)*0.1):]
            top20_idx = np.argsort(ensemble_pred)[-int(len(ensemble_pred)*0.2):]
            top10_return = np.mean(y_te[top10_idx])
            top20_return = np.mean(y_te[top20_idx])

            logger.info(f"\n{target_key} 目标:")
            logger.info(f"  RMSE: {rmse:.4f}")
            logger.info(f"  IC: {ic:.4f}")
            logger.info(f"  方向准确率: {direction_acc:.4f}")
            logger.info(f"  Top10收益: {top10_return:.4f}")
            logger.info(f"  Top20收益: {top20_return:.4f}")

            final_metrics[target_key] = {
                'rmse': rmse, 'ic': ic,
                'direction_accuracy': direction_acc,
                'top10_return': top10_return,
                'top20_return': top20_return
            }

        # 6. 融合预测评估
        logger.info("\n" + "=" * 60)
        logger.info("融合预测评估 (加权平均)")
        logger.info("=" * 60)

        fused_pred = (
            self.target_weights['label_3d'] * ensemble_predictions['3d'] +
            self.target_weights['label_5d'] * ensemble_predictions['5d'] +
            self.target_weights['label_10d'] * ensemble_predictions['10d']
        )

        # 对5天收益评估
        rmse = np.sqrt(mean_squared_error(y_5d_test, fused_pred))
        ic, _ = spearmanr(fused_pred, y_5d_test)
        direction_acc = np.mean((fused_pred > 0) == (y_5d_test > 0))

        logger.info(f"融合预测 (5天收益):")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  IC: {ic:.4f}")
        logger.info(f"  方向准确率: {direction_acc:.4f}")

        # 7. 特征重要性分析
        self._log_feature_importance(all_results)

        # 8. 保存模型
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        output_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v395'
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        model_data = {
            'models': all_results,
            'feature_names': self.feature_names,
            'target_weights': self.target_weights,
            'market_features': list(self.market_calculator.market_features.columns[1:]),
            'winsorize_bounds': getattr(self, 'winsorize_bounds', None),
            # 模型类型标识 (v2: Robust Z-Score + Industry-Excess)
            'cascade': False,
            'rank_normalized': False,
            'robust_zscore': True,
            'industry_excess_labels': True,
            'dual_stream': False,
            'cross_sectional_neutralization': False,
            'macro_feature_cols': self.macro_feature_cols,
            'stock_feature_cols': self.stock_feature_cols,
            'extra_features_from_daily_basic': ['pe_ttm', 'pb', 'ps_ttm', 'turnover_rate', 'log_market_cap'],
        }

        model_path = output_dir / f'v395_multi_target_{timestamp}.pkl'
        joblib.dump(model_data, model_path)
        logger.info(f"\n模型已保存: {model_path}")

        # 保存训练历史
        history = {
            'version': 'v3.95-robust-zscore-industry-excess',
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'status': 'completed',
            'summary': {
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'test_samples': len(X_test),
                'feature_count': len(self.feature_names),
                'market_feature_count': len(self.market_calculator.market_features.columns) - 1,
                'final_metrics': final_metrics
            },
            'target_weights': self.target_weights,
            'ensemble_weights': {
                '3d': all_results['3d']['weights'],
                '5d': all_results['5d']['weights'],
                '10d': all_results['10d']['weights']
            }
        }

        history_path = output_dir / f'training_history_{timestamp}.json'
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        # 保存latest链接
        latest_path = output_dir / 'training_history_latest.json'
        with open(latest_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        logger.info(f"训练历史已保存: {history_path}")
        logger.info(f"\n训练完成! 总耗时: {duration:.0f}秒")

        return model_data, history


def main():
    parser = argparse.ArgumentParser(description='V3.95 Robust Z-Score + Industry-Excess 训练')
    parser.add_argument('--start-date', type=str, default='2020-01-01', help='训练开始日期')
    parser.add_argument('--end-date', type=str, default=None, help='训练结束日期')
    parser.add_argument('--purge-days', type=int, default=10, help='Purge gap天数 (应>=最大标签前瞻天数, label_10d需要10天)')
    args = parser.parse_args()

    trainer = V395MultiTargetTrainer()
    trainer.train(start_date=args.start_date, end_date=args.end_date, purge_days=args.purge_days)


if __name__ == '__main__':
    main()
