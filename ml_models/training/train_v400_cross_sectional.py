#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V4.2 Hybrid Alpha Model 训练脚本

V4.2 改进 (基于 V4.1 vs V3.96 回测分析):
1. 行业超额标签 (--industry-excess-label): label -= 行业中位数，去除行业beta
2. Robust Z-Score特征归一化 (--robust-zscore-features): 保留幅度信息(vs rank丢失)
3. Robust Z-Score标签归一化 (--robust-zscore-labels): 稳健去除每日标签尺度差异
4. V39市场特征注入 (--join-v39-market): 从v39_feature_cache加载12个连续市场特征
5. 5模型Ensemble (--add-hgb): +RF+HistGradientBoosting
6. 超参数优化: 1000棵+early stopping, depth=6, learning_rate=0.03

--v42 总开关: 自动激活所有改进

沿用 V4.0.1 的:
- Temporal split + 10天 purge gap
- Per-feature winsorization (1st-99th percentile)
- 市场/行业特征缩放
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
import lightgbm as lgb
import xgboost as xgb
try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class V400CrossSectionalTrainer:
    """V4.0 Cross-Sectional Alpha Model 训练器"""

    # 市场级特征 (训练时缩放以减少对共同因子的依赖)
    MARKET_FEATURES = {
        'market_regime', 'market_vol_regime', 'market_breadth_5d',
        'northbound_flow_zscore', 'market_volume_regime', 'market_trend_strength'
    }

    # 行业级特征
    INDUSTRY_FEATURES = {
        'sw_l1_code', 'industry_breadth', 'industry_volume_change',
        'industry_kdj_avg', 'industry_macd_bullish_pct',
        'industry_concentration', 'industry_momentum_rank',
        'industry_rotation_signal'
    }

    # V39市场特征名 (从v39_feature_cache加载)
    V39_MARKET_FEATURES = [
        'v39_market_return_5d', 'v39_market_return_10d', 'v39_market_return_20d',
        'v39_market_volatility_10d', 'v39_market_volatility_20d',
        'v39_market_up_ratio_10d', 'v39_market_up_ratio_20d',
        'v39_market_drawdown_20d', 'v39_market_volume_ratio',
        'v39_market_position_20d', 'v39_market_momentum_5d', 'v39_market_momentum_20d',
    ]

    def __init__(self, db_path=None, meta_model_type='ridge',
                 market_scale=0.3, industry_scale=0.5, skip_rf=False,
                 rank_label=False, neutralize_label=False, feature_select=False,
                 v42_mode=False, industry_excess_label=False,
                 robust_zscore_features=False, robust_zscore_labels=False,
                 join_v39_market=False, add_hgb=False):
        self.db_path = db_path or str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')
        self.models = {}
        self.meta_model = None
        self.meta_model_type = meta_model_type  # 'ridge', 'gbm', 'avg'
        self.feature_names = None
        self.winsorize_bounds = None
        self.market_scale = market_scale
        self.industry_scale = industry_scale
        self.skip_rf = skip_rf
        self.rank_label = rank_label
        self.neutralize_label = neutralize_label
        self.feature_select = feature_select
        self.selected_features = None  # IC筛选后的特征列表

        # V4.2 参数
        self.v42_mode = v42_mode
        self.industry_excess_label = industry_excess_label
        self.robust_zscore_features = robust_zscore_features
        self.robust_zscore_labels = robust_zscore_labels
        self.join_v39_market = join_v39_market
        self.add_hgb = add_hgb
        self.v39_market_feature_names = []  # 实际加载的v39特征名

        # V4.2 总开关: 自动激活所有改进
        if self.v42_mode:
            self.industry_excess_label = True
            self.robust_zscore_features = True
            self.robust_zscore_labels = True
            self.join_v39_market = True
            self.add_hgb = True
            self.skip_rf = False
            self.rank_label = False
            self.neutralize_label = False
            self.meta_model_type = 'avg'
            self.market_scale = 0.5
            logger.info("🔧 V4.2模式: 自动激活全部改进 (行业超额标签+RobustZScore+V39市场特征+5模型+超参优化)")

        # 多目标权重
        self.target_weights = {
            'label_3d_excess': 0.35,
            'label_5d_excess': 0.40,
            'label_10d_excess': 0.25,
        }

    def load_cached_features(self, min_samples=1000):
        """从v40_feature_cache加载预计算特征+超额收益标签"""
        logger.info("=" * 80)
        logger.info("📥 从v40_feature_cache加载Cross-Sectional特征...")
        logger.info("=" * 80)

        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT v.code, v.trade_date, v.features_json,
                   v.label_3d_excess, v.label_5d_excess, v.label_10d_excess
            FROM v40_feature_cache v
            JOIN securities s ON v.code = s.code
            JOIN daily_quotes q ON q.security_id = s.id AND q.trade_date = v.trade_date
            WHERE v.label_5d_excess IS NOT NULL
              AND q.volume > 0
            ORDER BY v.trade_date, v.code
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        logger.info(f"✅ 加载 {len(df):,} 个样本 (已过滤停牌日)")

        if len(df) < min_samples:
            raise ValueError(f"样本数不足！需要至少{min_samples}个，实际{len(df)}个")

        # 解析JSON特征
        logger.info("📊 解析JSON特征...")
        features_list = []
        labels_3d = []
        labels_5d = []
        labels_10d = []
        dates = []
        codes = []

        for idx, row in df.iterrows():
            try:
                features_dict = json.loads(row['features_json'])
                features_list.append(features_dict)
                labels_3d.append(row['label_3d_excess'])
                labels_5d.append(row['label_5d_excess'])
                labels_10d.append(row['label_10d_excess'])
                dates.append(row['trade_date'])
                codes.append(row['code'])
            except Exception as e:
                continue

            if (idx + 1) % 50000 == 0:
                logger.info(f"  已处理: {idx+1:,}/{len(df):,}")

        X = pd.DataFrame(features_list)
        y_3d = np.array(labels_3d)
        y_5d = np.array(labels_5d)
        y_10d = np.array(labels_10d)
        dates = np.array(dates)
        codes = np.array(codes)

        # 处理NaN (先用原始标签过滤)
        valid_mask = ~(np.isnan(y_3d) | np.isnan(y_5d) | np.isnan(y_10d))
        X = X.loc[valid_mask].reset_index(drop=True)
        y_3d = y_3d[valid_mask]
        y_5d = y_5d[valid_mask]
        y_10d = y_10d[valid_mask]
        dates = dates[valid_mask]
        codes = codes[valid_mask]

        logger.info(f"✅ 特征矩阵: {X.shape}")
        logger.info(f"✅ 有效样本: {len(y_3d):,}")
        logger.info(f"✅ 日期范围: {dates[0]} ~ {dates[-1]}")

        # 缺失值填0
        if X.isnull().any().any():
            logger.warning("⚠️ 检测到缺失值，使用0填充")
            X = X.fillna(0)

        # V4.2: 加载V39市场特征并合并
        if self.join_v39_market:
            v39_market_df = self._load_v39_market_features()
            if v39_market_df is not None:
                # 按trade_date合并
                X['_trade_date'] = dates
                X = X.merge(v39_market_df, left_on='_trade_date', right_on='trade_date', how='left')
                X = X.drop(columns=['_trade_date', 'trade_date'], errors='ignore')
                X[self.v39_market_feature_names] = X[self.v39_market_feature_names].fillna(0)

                # 删除V40的6个分类市场特征 (用V39的连续特征替代)
                v40_market_cat = ['market_regime', 'market_vol_regime', 'market_breadth_5d',
                                  'northbound_flow_zscore', 'market_volume_regime', 'market_trend_strength']
                dropped = [c for c in v40_market_cat if c in X.columns]
                if dropped:
                    X = X.drop(columns=dropped)
                    logger.info(f"  删除V40分类市场特征: {dropped}")

                logger.info(f"  ✅ 合并V39市场特征后: {X.shape}")

        self.feature_names = X.columns.tolist()

        # V4.2: 行业超额标签 (在加权融合之前)
        if self.industry_excess_label:
            y_3d, y_5d, y_10d = self._compute_industry_excess_labels(
                y_3d, y_5d, y_10d, dates, X)

        # 计算加权融合标签
        y = (self.target_weights['label_3d_excess'] * y_3d +
             self.target_weights['label_5d_excess'] * y_5d +
             self.target_weights['label_10d_excess'] * y_10d)

        # 保留原始5d超额收益用于评估 (即使transform了训练标签)
        y_5d_raw = y_5d.copy()

        # V4.2: Robust Z-Score标签归一化
        if self.robust_zscore_labels:
            y = self._robust_zscore_labels(y, dates)
            y_5d = self._robust_zscore_labels(y_5d, dates)

        # B2: 排名标签转换 (V4.2模式下自动跳过)
        if self.rank_label:
            y, y_5d = self._convert_to_rank_labels(y, y_5d, dates)

        # B3: 标签中性化 (V4.2模式下自动跳过)
        if self.neutralize_label:
            y = self._neutralize_labels(y, dates, codes, X)

        return X, y, y_5d, y_5d_raw, dates, codes

    def _convert_to_rank_labels(self, y, y_5d, dates):
        """
        B2: 将超额收益标签转为当日cross-sectional排名 (0-1)

        排名标签优势:
        - 分布稳定 (uniform)，不受牛熊市大幅波动影响
        - 与排名特征在同一空间
        - 模型只需预测相对排序，降低绝对值预测难度
        """
        logger.info("\n📊 B2: 转换为Cross-Sectional排名标签...")

        y_ranked = y.copy()
        y_5d_ranked = y_5d.copy()

        unique_dates = np.unique(dates)
        for date in unique_dates:
            mask = dates == date
            n = mask.sum()
            if n < 10:
                continue

            # 使用排名 / N 转为 0-1 uniform分布
            # 加0.5使rank居中: rank(1..N) -> (0.5/N, ..., (N-0.5)/N)
            day_y = y[mask]
            ranks = pd.Series(day_y).rank(method='average').values
            y_ranked[mask] = (ranks - 0.5) / n

            day_y5d = y_5d[mask]
            ranks_5d = pd.Series(day_y5d).rank(method='average').values
            y_5d_ranked[mask] = (ranks_5d - 0.5) / n

        logger.info(f"  ✅ 排名标签: mean={np.mean(y_ranked):.4f}, "
                     f"std={np.std(y_ranked):.4f}, "
                     f"min={np.min(y_ranked):.4f}, max={np.max(y_ranked):.4f}")
        logger.info(f"  原始标签: mean={np.mean(y):.6f}, std={np.std(y):.6f}")

        return y_ranked, y_5d_ranked

    def _neutralize_labels(self, y, dates, codes, X):
        """
        B3: 标签行业+市值中性化 (回归残差法)

        对每日标签做: y_neutral = y - beta_ind * industry_dummy - beta_size * log_mcap
        模型只学习行业/市值无法解释的alpha
        """
        from sklearn.linear_model import LinearRegression
        logger.info("\n📊 B3: 标签中性化 (行业+市值)...")

        y_neutral = y.copy()

        # 获取行业和市值信息
        has_industry = 'sw_l1_code' in X.columns
        has_mcap = 'xs_market_cap_rank' in X.columns

        if not has_industry and not has_mcap:
            logger.warning("  ⚠️ 无行业/市值特征，跳过中性化")
            return y

        unique_dates = np.unique(dates)
        neutralized_count = 0

        for date in unique_dates:
            mask = dates == date
            n = mask.sum()
            if n < 30:
                continue

            # 构建中性化因子矩阵
            factors = []
            if has_industry:
                # 行业哑变量
                ind_codes = X.loc[mask, 'sw_l1_code'].values
                unique_inds = np.unique(ind_codes)
                if len(unique_inds) > 1:
                    ind_dummies = np.zeros((n, len(unique_inds) - 1))
                    for i, ind in enumerate(unique_inds[1:]):
                        ind_dummies[:, i] = (ind_codes == ind).astype(float)
                    factors.append(ind_dummies)

            if has_mcap:
                # 市值排名作为size因子
                mcap = X.loc[mask, 'xs_market_cap_rank'].values.reshape(-1, 1)
                factors.append(mcap)

            if factors:
                Z = np.hstack(factors)
                reg = LinearRegression(fit_intercept=True)
                reg.fit(Z, y[mask])
                y_neutral[mask] = y[mask] - reg.predict(Z)
                neutralized_count += 1

        logger.info(f"  ✅ 中性化完成: {neutralized_count}/{len(unique_dates)} 天")
        logger.info(f"  中性化后: mean={np.mean(y_neutral):.6f}, std={np.std(y_neutral):.6f}")
        logger.info(f"  原始标签: mean={np.mean(y):.6f}, std={np.std(y):.6f}")

        return y_neutral

    def _load_v39_market_features(self):
        """
        从v39_feature_cache加载12个连续市场特征 (去重per date)

        V39的市场特征是连续值(return, volatility等)，比V40的分类变量(regime=0/1/2)信息量更大。
        """
        logger.info("\n📊 V4.2: 加载V39连续市场特征...")

        conn = sqlite3.connect(self.db_path)

        # v39_feature_cache的市场特征是per-date的列，每天所有股票共享同一值
        # 只需每天取1行即可
        v39_cols = [
            'market_return_5d', 'market_return_10d', 'market_return_20d',
            'market_volatility_10d', 'market_volatility_20d',
            'market_up_ratio_10d', 'market_up_ratio_20d',
            'market_drawdown_20d', 'market_volume_ratio',
            'market_position_20d', 'market_momentum_5d', 'market_momentum_20d',
        ]
        cols_sql = ', '.join(v39_cols)

        query = f"""
            SELECT trade_date, {cols_sql}
            FROM v39_feature_cache
            GROUP BY trade_date
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            logger.warning("  ⚠️ v39_feature_cache无数据，跳过V39市场特征注入")
            return None

        # 重命名: 加v39_前缀以区分
        rename_map = {col: f'v39_{col}' for col in v39_cols}
        df = df.rename(columns=rename_map)

        self.v39_market_feature_names = [f'v39_{col}' for col in v39_cols]
        logger.info(f"  ✅ 加载V39市场特征: {len(df)} 天, {len(self.v39_market_feature_names)} 个特征")
        logger.info(f"  特征: {self.v39_market_feature_names}")

        return df

    def _compute_industry_excess_labels(self, y_3d, y_5d, y_10d, dates, X):
        """
        V4.2: 行业超额标签 (label -= 行业中位数)

        比HS300-excess更精细：去除行业beta而非大盘beta
        每个标签独立处理
        """
        logger.info("\n📊 V4.2: 计算行业超额收益标签...")

        if 'sw_l1_code' not in X.columns:
            logger.warning("  ⚠️ 无sw_l1_code列，跳过行业超额")
            return y_3d, y_5d, y_10d

        sw_codes = X['sw_l1_code'].values
        unique_dates = np.unique(dates)

        y_3d_new = y_3d.copy()
        y_5d_new = y_5d.copy()
        y_10d_new = y_10d.copy()

        adjusted_count = 0
        for date in unique_dates:
            mask = dates == date
            n = mask.sum()
            if n < 20:
                continue

            day_sw = sw_codes[mask]
            unique_inds = np.unique(day_sw)

            for ind in unique_inds:
                ind_mask_in_day = day_sw == ind
                # 在全局mask中定位
                day_indices = np.where(mask)[0]
                ind_indices = day_indices[ind_mask_in_day]

                if len(ind_indices) < 3:
                    continue

                # 各标签减去行业中位数
                y_3d_new[ind_indices] -= np.median(y_3d[ind_indices])
                y_5d_new[ind_indices] -= np.median(y_5d[ind_indices])
                y_10d_new[ind_indices] -= np.median(y_10d[ind_indices])

            adjusted_count += 1

        logger.info(f"  ✅ 行业超额标签: {adjusted_count}/{len(unique_dates)} 天")
        logger.info(f"  3d标签: mean {np.mean(y_3d):.6f} → {np.mean(y_3d_new):.6f}")
        logger.info(f"  5d标签: mean {np.mean(y_5d):.6f} → {np.mean(y_5d_new):.6f}")
        logger.info(f"  10d标签: mean {np.mean(y_10d):.6f} → {np.mean(y_10d_new):.6f}")

        return y_3d_new, y_5d_new, y_10d_new

    def _robust_zscore_labels(self, y, dates):
        """
        V4.2: Robust Z-Score标签归一化

        z = (y - median) / (1.4826 * MAD), clip [-3, 3]
        稳健去除每日标签尺度差异，抗极端值
        """
        logger.info("\n📊 V4.2: Robust Z-Score标签归一化...")

        y_new = y.copy()
        unique_dates = np.unique(dates)
        transformed_count = 0

        for date in unique_dates:
            mask = dates == date
            n = mask.sum()
            if n < 20:
                continue

            day_y = y[mask]
            median = np.median(day_y)
            mad = np.median(np.abs(day_y - median))
            scale = 1.4826 * mad  # MAD to std conversion

            if scale < 1e-10:
                y_new[mask] = 0.0
            else:
                y_new[mask] = np.clip((day_y - median) / scale, -3, 3)

            transformed_count += 1

        logger.info(f"  ✅ Robust Z-Score标签: {transformed_count}/{len(unique_dates)} 天")
        logger.info(f"  归一化后: mean={np.mean(y_new):.6f}, std={np.std(y_new):.6f}, "
                     f"min={np.min(y_new):.4f}, max={np.max(y_new):.4f}")

        return y_new

    def _robust_zscore_features(self, X_train, X_val, X_test,
                                 dates_train, dates_val, dates_test):
        """
        V4.2: Cross-Sectional Robust Z-Score特征归一化

        对每个(split, date, feature)做: z = (x - median) / (1.4826 * MAD), clip [-3, 3]
        排除sw_l1_code（分类变量）
        各split独立归一化
        """
        logger.info("\n📊 V4.2: Cross-Sectional Robust Z-Score特征归一化...")

        categorical_cols = {'sw_l1_code'}
        numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

        total_transformed = 0
        for split_name, X_split, dates_split in [
            ('train', X_train, dates_train),
            ('val', X_val, dates_val),
            ('test', X_test, dates_test),
        ]:
            unique_dates = np.unique(dates_split)
            for date in unique_dates:
                mask = dates_split == date
                n = mask.sum()
                if n < 10:
                    continue

                for col in numeric_cols:
                    vals = X_split.loc[mask, col].values
                    median = np.median(vals)
                    mad = np.median(np.abs(vals - median))
                    scale = 1.4826 * mad

                    if scale < 1e-10:
                        X_split.loc[mask, col] = 0.0
                    else:
                        X_split.loc[mask, col] = np.clip((vals - median) / scale, -3, 3)

                total_transformed += 1

        logger.info(f"  ✅ 特征Robust Z-Score: {total_transformed} 个(split×date)组, "
                     f"{len(numeric_cols)} 个特征")

        return X_train, X_val, X_test

    def winsorize_features(self, X_train, X_val, X_test):
        """Per-feature winsorization (1st-99th percentile)"""
        logger.info("📊 特征Winsorization (1st-99th percentile)...")

        self.winsorize_bounds = {}
        for col in X_train.columns:
            lower = X_train[col].quantile(0.01)
            upper = X_train[col].quantile(0.99)
            self.winsorize_bounds[col] = (lower, upper)
            X_train[col] = X_train[col].clip(lower, upper)
            X_val[col] = X_val[col].clip(lower, upper)
            X_test[col] = X_test[col].clip(lower, upper)

        return X_train, X_val, X_test

    def scale_market_industry_features(self, X_train, X_val, X_test):
        """缩放市场和行业特征，减少共同因子对模型的干扰"""
        logger.info(f"📊 市场/行业特征缩放: market×{self.market_scale}, industry×{self.industry_scale}")

        # 动态扩展市场特征集合 (包含v39_*特征)
        market_features = set(self.MARKET_FEATURES)
        if self.v39_market_feature_names:
            market_features.update(self.v39_market_feature_names)

        scaled_market = 0
        scaled_industry = 0
        for col in X_train.columns:
            if col in market_features:
                X_train[col] = X_train[col] * self.market_scale
                X_val[col] = X_val[col] * self.market_scale
                X_test[col] = X_test[col] * self.market_scale
                scaled_market += 1
            elif col in self.INDUSTRY_FEATURES:
                X_train[col] = X_train[col] * self.industry_scale
                X_val[col] = X_val[col] * self.industry_scale
                X_test[col] = X_test[col] * self.industry_scale
                scaled_industry += 1

        logger.info(f"  缩放了 {scaled_market} 个市场特征, {scaled_industry} 个行业特征")
        return X_train, X_val, X_test

    def ic_feature_selection(self, X, y, dates, min_ic=0.005, min_ic_positive_pct=0.45):
        """
        B4: IC-based特征筛选

        对每个特征计算:
        1. 历史平均Daily IC (Spearman与标签的秩相关)
        2. IC > 0 的日期占比
        3. 移除 IC均值 < min_ic 或 IC>0占比 < min_ic_positive_pct 的特征

        Args:
            X: 特征矩阵
            y: 标签
            dates: 日期数组
            min_ic: 最低IC均值阈值
            min_ic_positive_pct: IC>0天数最低占比
        """
        logger.info("\n" + "=" * 80)
        logger.info(f"📊 B4: IC-based特征筛选 (min_ic={min_ic}, min_pos%={min_ic_positive_pct:.0%})")
        logger.info("=" * 80)

        unique_dates = np.unique(dates)
        feature_ics = {col: [] for col in X.columns}

        for date in unique_dates:
            mask = dates == date
            if mask.sum() < 50:
                continue

            y_day = y[mask]
            for col in X.columns:
                x_day = X.loc[mask, col].values
                # 跳过常数特征
                if np.std(x_day) < 1e-10:
                    continue
                ic, _ = spearmanr(x_day, y_day)
                if not np.isnan(ic):
                    feature_ics[col].append(ic)

        # 计算每个特征的IC统计
        feature_stats = []
        for col in X.columns:
            ics = feature_ics[col]
            if len(ics) < 20:
                feature_stats.append((col, 0.0, 0.0, len(ics), False))
                continue

            mean_ic = np.mean(ics)
            ic_pos_pct = np.mean(np.array(ics) > 0)
            ic_ir = mean_ic / np.std(ics) if np.std(ics) > 0 else 0

            keep = abs(mean_ic) >= min_ic and ic_pos_pct >= min_ic_positive_pct
            feature_stats.append((col, mean_ic, ic_pos_pct, len(ics), keep))

        # 排序显示
        feature_stats.sort(key=lambda x: abs(x[1]), reverse=True)

        kept = [f for f in feature_stats if f[4]]
        removed = [f for f in feature_stats if not f[4]]

        logger.info(f"\n  保留特征 ({len(kept)}):")
        for col, mean_ic, pos_pct, n_days, _ in kept[:30]:
            logger.info(f"    ✅ {col:<40s} IC={mean_ic:+.4f}, IC>0={pos_pct:.1%}")

        logger.info(f"\n  移除特征 ({len(removed)}):")
        for col, mean_ic, pos_pct, n_days, _ in removed:
            logger.info(f"    ❌ {col:<40s} IC={mean_ic:+.4f}, IC>0={pos_pct:.1%}")

        kept_cols = [f[0] for f in kept]
        if len(kept_cols) < 10:
            logger.warning(f"  ⚠️ 筛选后只剩 {len(kept_cols)} 个特征，太少! 放宽阈值...")
            # 取abs(IC)最大的前30个
            feature_stats.sort(key=lambda x: abs(x[1]), reverse=True)
            kept_cols = [f[0] for f in feature_stats[:30]]
            logger.info(f"  改为取IC绝对值最大的30个特征")

        self.selected_features = kept_cols
        self.feature_names = kept_cols

        X_selected = X[kept_cols].copy()
        logger.info(f"\n  ✅ 特征筛选: {X.shape[1]} → {X_selected.shape[1]} 个特征")

        return X_selected

    def temporal_split(self, X, y, y_5d, y_5d_raw, dates, codes,
                       val_ratio=0.15, test_ratio=0.15, purge_days=10):
        """时序划分 + purge gap (10天避免超额收益标签窗口重叠)"""
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

        result = {}
        for name, mask in [('train', train_mask), ('val', val_mask), ('test', test_mask)]:
            result[f'X_{name}'] = X.loc[mask].reset_index(drop=True)
            result[f'y_{name}'] = y[mask]
            result[f'y5d_{name}'] = y_5d[mask]
            result[f'y5d_raw_{name}'] = y_5d_raw[mask]
            result[f'dates_{name}'] = dates[mask]
            result[f'codes_{name}'] = codes[mask]

        logger.info(f"  时序划分 (purge_gap={purge_days}天):")
        logger.info(f"  训练集: {len(result['X_train']):,} 样本, 截至 {train_date_end}")
        logger.info(f"  验证集: {len(result['X_val']):,} 样本, {val_date_start} ~ {val_date_end}")
        logger.info(f"  测试集: {len(result['X_test']):,} 样本, {test_date_start} 起")

        return result

    def train_base_models(self, X_train, y_train, X_val, y_val):
        """训练基础模型 (V4.2: 5模型+early stopping+超参优化)"""
        logger.info("\n" + "=" * 80)
        if self.v42_mode:
            logger.info("🔧 V4.2: 训练5个基础模型 (1000棵+early stopping+depth=6)...")
        else:
            logger.info("🔧 训练基础模型 (更强正则化)...")
        logger.info("=" * 80)

        # V4.2 超参数优化
        if self.v42_mode:
            n_est = 1000
            lr = 0.03
            depth = 6
            n_leaves = 50
            min_child = 30
            subsample = 0.8
            colsample = 0.7
            reg_a = 0.1
            reg_l = 0.1
        else:
            n_est = 300
            lr = 0.03
            depth = 5
            n_leaves = 25
            min_child = 50
            subsample = 0.7
            colsample = 0.7
            reg_a = 0.3
            reg_l = 0.3

        models_config = {
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=n_est,
                learning_rate=lr,
                max_depth=depth,
                num_leaves=n_leaves,
                min_child_samples=min_child,
                subsample=subsample,
                colsample_bytree=colsample,
                reg_alpha=reg_a,
                reg_lambda=reg_l,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=n_est,
                learning_rate=lr,
                max_depth=depth,
                min_child_weight=min_child,
                subsample=subsample,
                colsample_bytree=colsample,
                reg_alpha=reg_a,
                reg_lambda=reg_l,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                early_stopping_rounds=50 if self.v42_mode else None,
            ),
        }

        if not self.skip_rf:
            models_config['random_forest'] = RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features=0.6,
                max_samples=200000,
                random_state=42,
                n_jobs=-1
            )
        else:
            logger.info("  ⏭️ 跳过 Random Forest (--skip-rf)")

        if HAS_CATBOOST:
            models_config['catboost'] = CatBoostRegressor(
                iterations=n_est,
                learning_rate=lr,
                depth=min(depth, 6),
                l2_leaf_reg=5,
                min_data_in_leaf=min_child,
                random_state=42,
                verbose=False,
                early_stopping_rounds=50 if self.v42_mode else None,
            )

        # V4.2: 新增 HistGradientBoosting
        if self.add_hgb:
            models_config['hist_gradient_boosting'] = HistGradientBoostingRegressor(
                max_iter=1000,
                learning_rate=lr,
                max_depth=depth,
                min_samples_leaf=min_child,
                l2_regularization=reg_l,
                early_stopping=False,
                random_state=42,
            )

        for name, model in models_config.items():
            logger.info(f"\n🔹 训练 {name}...")
            start_time = datetime.now()

            # V4.2: early stopping for LGB/XGB/CatBoost
            if self.v42_mode and name == 'lightgbm':
                model.fit(X_train, y_train,
                          eval_set=[(X_val, y_val)],
                          callbacks=[lgb.early_stopping(50, verbose=False),
                                     lgb.log_evaluation(0)])
            elif self.v42_mode and name == 'xgboost':
                model.fit(X_train, y_train,
                          eval_set=[(X_val, y_val)],
                          verbose=False)
            elif self.v42_mode and name == 'catboost':
                model.fit(X_train, y_train,
                          eval_set=(X_val, y_val))
            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            ic, _ = spearmanr(y_pred, y_val)

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"  ✅ {name}: MSE={mse:.6f}, MAE={mae:.6f}, IC={ic:.4f}, 耗时={elapsed:.1f}秒")

            # Log early stopping iterations if applicable
            if self.v42_mode:
                if name == 'lightgbm' and hasattr(model, 'best_iteration_'):
                    logger.info(f"     Early stopped at iteration {model.best_iteration_}")
                elif name == 'xgboost' and hasattr(model, 'best_iteration'):
                    logger.info(f"     Early stopped at iteration {model.best_iteration}")
                elif name == 'hist_gradient_boosting' and hasattr(model, 'n_iter_'):
                    logger.info(f"     Stopped at iteration {model.n_iter_}")

            self.models[name] = model

        return self.models

    def train_meta_model(self, X_val, y_val, X_test=None, y_test=None):
        """训练元模型 (Stacking) - 支持 Ridge/GBM/Average"""
        logger.info("\n" + "=" * 80)
        logger.info(f"🔧 训练元模型 (Stacking, type={self.meta_model_type})...")
        logger.info("=" * 80)

        model_names = list(self.models.keys())
        meta_features_val = np.column_stack([
            model.predict(X_val) for model in self.models.values()
        ])

        # 诊断 base model 在 val 上的预测分布
        self._diagnose_base_predictions(meta_features_val, y_val, model_names, "验证集")

        if self.meta_model_type == 'ridge':
            # Ridge: 线性模型，4个输入不会过拟合
            self.meta_model = Ridge(alpha=1.0)
            self.meta_model.fit(meta_features_val, y_val)

            # 记录 Ridge 系数 (应全为正值)
            logger.info(f"  Ridge 系数:")
            for name, coef in zip(model_names, self.meta_model.coef_):
                sign = "✅" if coef > 0 else "⚠️ 负值!"
                logger.info(f"    {name}: {coef:.6f} {sign}")
            logger.info(f"    intercept: {self.meta_model.intercept_:.6f}")

        elif self.meta_model_type == 'gbm':
            # 保留 GBM 选项 (已知过拟合风险)
            logger.warning("⚠️ GBM meta-model 有过拟合风险，建议使用 ridge")
            self.meta_model = GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
            )
            self.meta_model.fit(meta_features_val, y_val)

        elif self.meta_model_type == 'avg':
            # Simple average: 不训练，只保存权重
            self.meta_model = None  # predict 时直接取均值
            logger.info("  使用 Simple Average (等权) - 无需训练")

        else:
            raise ValueError(f"不支持的 meta_model_type: {self.meta_model_type}")

        # Simple average baseline 对比
        avg_pred_val = np.mean(meta_features_val, axis=1)
        avg_ic, _ = spearmanr(avg_pred_val, y_val)
        logger.info(f"\n  📊 Simple Average baseline IC (val): {avg_ic:.4f}")

        if X_test is not None and y_test is not None:
            meta_features_test = np.column_stack([
                model.predict(X_test) for model in self.models.values()
            ])

            # 诊断 base model 在 test 上的预测分布
            self._diagnose_base_predictions(meta_features_test, y_test, model_names, "测试集")

            # 检测 val→test 分布偏移
            self._diagnose_distribution_shift(meta_features_val, meta_features_test, model_names)

            # Meta-model 预测
            if self.meta_model is not None:
                y_pred = self.meta_model.predict(meta_features_test)
            else:
                y_pred = np.mean(meta_features_test, axis=1)

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            ic, _ = spearmanr(y_pred, y_test)
            direction_acc = np.mean((y_pred > 0) == (y_test > 0))

            logger.info(f"\n✅ 元模型 (测试集): MSE={mse:.6f}, MAE={mae:.6f}")
            logger.info(f"  IC={ic:.4f}, 超额方向准确率={direction_acc:.4f}")

            # 对比 simple average
            avg_pred_test = np.mean(meta_features_test, axis=1)
            avg_ic_test, _ = spearmanr(avg_pred_test, y_test)
            logger.info(f"  📊 Simple Average IC (test): {avg_ic_test:.4f}")

            # 检查正负分布
            pos_pct = np.mean(y_pred > 0) * 100
            logger.info(f"  预测正值比例: {pos_pct:.1f}%  (base models均值: {np.mean(meta_features_test):.6f})")

        return self.meta_model

    def _diagnose_base_predictions(self, meta_features, y_true, model_names, dataset_name):
        """诊断每个 base model 的预测分布"""
        logger.info(f"\n  📊 Base Model 预测分布 ({dataset_name}):")
        for i, name in enumerate(model_names):
            preds = meta_features[:, i]
            ic, _ = spearmanr(preds, y_true)
            pos_pct = np.mean(preds > 0) * 100
            logger.info(f"    {name}: mean={np.mean(preds):.6f}, std={np.std(preds):.6f}, "
                         f"pos%={pos_pct:.1f}%, IC={ic:.4f}")

    def _diagnose_distribution_shift(self, meta_val, meta_test, model_names):
        """检测 val→test 的预测分布偏移"""
        logger.info(f"\n  📊 Val→Test 分布偏移诊断:")
        for i, name in enumerate(model_names):
            val_mean = np.mean(meta_val[:, i])
            val_std = np.std(meta_val[:, i])
            test_mean = np.mean(meta_test[:, i])

            shift_sigma = abs(test_mean - val_mean) / val_std if val_std > 0 else 0
            status = "⚠️ WARNING" if shift_sigma > 2.0 else "✅ OK"
            logger.info(f"    {name}: val_mean={val_mean:.6f}, test_mean={test_mean:.6f}, "
                         f"shift={shift_sigma:.2f}σ {status}")

    def evaluate_cross_sectional(self, X_test, y_test, y5d_test, dates_test, codes_test):
        """
        Cross-Sectional评估指标

        - Daily IC: 每天的 Spearman rank correlation
        - IC_IR: IC均值 / IC标准差 (稳定性)
        - Top-10%/20% Precision: 预测前10%与实际前10%的重叠率
        - Top-10%/20% Excess Return: 预测排名前10%的平均超额收益
        """
        logger.info("\n" + "=" * 80)
        logger.info("📊 Cross-Sectional 评估指标")
        logger.info("=" * 80)

        # 获取预测值
        meta_features = np.column_stack([
            model.predict(X_test) for model in self.models.values()
        ])
        if self.meta_model is not None:
            y_pred = self.meta_model.predict(meta_features)
        else:
            y_pred = np.mean(meta_features, axis=1)

        # Cross-Sectional Demean: 每天减去当日均值，隔离选股能力
        logger.info("  📊 对预测值进行 Cross-Sectional Demean...")
        y_pred_demeaned = y_pred.copy()
        for date in np.unique(dates_test):
            mask = dates_test == date
            if mask.sum() > 1:
                y_pred_demeaned[mask] -= np.mean(y_pred[mask])

        # 使用 demeaned 预测进行评估
        y_pred = y_pred_demeaned

        # 按日期分组计算 Daily IC
        unique_dates = np.unique(dates_test)
        daily_ics = []
        daily_top10_excess = []
        daily_top20_excess = []
        daily_top10_precision = []
        daily_top20_precision = []

        for date in unique_dates:
            mask = dates_test == date
            if mask.sum() < 20:
                continue

            pred_day = y_pred[mask]
            actual_day = y5d_test[mask]  # 使用5d超额收益评估

            # Daily IC
            ic, _ = spearmanr(pred_day, actual_day)
            if not np.isnan(ic):
                daily_ics.append(ic)

            n = len(pred_day)
            top10_n = max(1, int(n * 0.1))
            top20_n = max(1, int(n * 0.2))

            # Top-10% excess return
            pred_top10_idx = np.argsort(pred_day)[-top10_n:]
            pred_top20_idx = np.argsort(pred_day)[-top20_n:]
            actual_top10_idx = set(np.argsort(actual_day)[-top10_n:])
            actual_top20_idx = set(np.argsort(actual_day)[-top20_n:])

            daily_top10_excess.append(np.mean(actual_day[pred_top10_idx]))
            daily_top20_excess.append(np.mean(actual_day[pred_top20_idx]))

            # Precision
            overlap_10 = len(set(pred_top10_idx) & actual_top10_idx) / top10_n
            overlap_20 = len(set(pred_top20_idx) & actual_top20_idx) / top20_n
            daily_top10_precision.append(overlap_10)
            daily_top20_precision.append(overlap_20)

        # 汇总
        metrics = {}

        if daily_ics:
            mean_ic = np.mean(daily_ics)
            std_ic = np.std(daily_ics)
            ic_ir = mean_ic / std_ic if std_ic > 0 else 0
            ic_positive_pct = np.mean(np.array(daily_ics) > 0)

            metrics['daily_ic_mean'] = mean_ic
            metrics['daily_ic_std'] = std_ic
            metrics['ic_ir'] = ic_ir
            metrics['ic_positive_pct'] = ic_positive_pct

            logger.info(f"  Daily IC: {mean_ic:.4f} ± {std_ic:.4f}")
            logger.info(f"  IC_IR: {ic_ir:.4f}")
            logger.info(f"  IC > 0 比例: {ic_positive_pct:.1%}")

        if daily_top10_excess:
            metrics['top10_excess_return_mean'] = np.mean(daily_top10_excess)
            metrics['top20_excess_return_mean'] = np.mean(daily_top20_excess)
            metrics['top10_precision_mean'] = np.mean(daily_top10_precision)
            metrics['top20_precision_mean'] = np.mean(daily_top20_precision)

            logger.info(f"  Top-10% 平均超额收益: {np.mean(daily_top10_excess):.4f} (5d)")
            logger.info(f"  Top-20% 平均超额收益: {np.mean(daily_top20_excess):.4f} (5d)")
            logger.info(f"  Top-10% Precision: {np.mean(daily_top10_precision):.2%}")
            logger.info(f"  Top-20% Precision: {np.mean(daily_top20_precision):.2%}")

        # 特征重要性分析: 个股/市场/行业比例
        self._analyze_feature_composition()

        return metrics

    def _analyze_feature_composition(self):
        """分析特征重要性中个股/行业/市场特征的占比"""
        if not self.feature_names:
            return

        logger.info("\n📊 特征来源占比分析:")

        market_features = set(self.MARKET_FEATURES)
        # 动态添加v39市场特征
        if self.v39_market_feature_names:
            market_features.update(self.v39_market_feature_names)
        industry_features = set(self.INDUSTRY_FEATURES)

        for name, model in self.models.items():
            importance = None
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'feature_importance'):
                importance = model.feature_importance()
            if importance is None:
                continue

            total = sum(importance)
            if total == 0:
                continue

            market_imp = sum(importance[i] for i, f in enumerate(self.feature_names) if f in market_features) / total
            industry_imp = sum(importance[i] for i, f in enumerate(self.feature_names) if f in industry_features) / total
            stock_imp = 1 - market_imp - industry_imp

            logger.info(f"  {name}: 个股={stock_imp:.1%}, 行业={industry_imp:.1%}, 市场={market_imp:.1%}")

    def _log_feature_importance(self, top_n: int = 20):
        """打印特征重要性并保存"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 特征重要性分析")
        logger.info("=" * 80)

        all_importances = {}

        for name, model in self.models.items():
            importance = None
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'feature_importance'):
                importance = model.feature_importance()
            if importance is None:
                continue

            feat_imp = sorted(zip(self.feature_names, importance), key=lambda x: x[1], reverse=True)
            all_importances[name] = {f: float(v) for f, v in feat_imp}

            logger.info(f"\n🔹 {name} Top {top_n}:")
            for rank, (feat, imp) in enumerate(feat_imp[:top_n], 1):
                logger.info(f"  {rank:2d}. {feat:<35s} {imp:.4f}")

        # 平均重要性
        if all_importances:
            avg_importance = {}
            for feat in self.feature_names:
                values = [imp.get(feat, 0) for imp in all_importances.values()]
                avg_importance[feat] = float(np.mean(values))
            avg_sorted = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)

            logger.info(f"\n🔹 平均特征重要性 Top {top_n}:")
            for rank, (feat, imp) in enumerate(avg_sorted[:top_n], 1):
                logger.info(f"  {rank:2d}. {feat:<35s} {imp:.4f}")

            all_importances['average'] = dict(avg_sorted)

        # 保存
        output_dir = Path(PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v400')
        output_dir.mkdir(parents=True, exist_ok=True)
        importance_path = output_dir / f"v400_feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(importance_path, 'w', encoding='utf-8') as f:
            json.dump(all_importances, f, indent=2, ensure_ascii=False)
        logger.info(f"\n💾 特征重要性已保存: {importance_path}")

    def save_model(self):
        """保存模型"""
        logger.info("\n" + "=" * 80)
        logger.info("💾 保存模型...")
        logger.info("=" * 80)

        output_dir = Path(PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v400')
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存基础模型
        base_models_path = output_dir / f"v400_base_models_{timestamp}.pkl"
        with open(base_models_path, 'wb') as f:
            pickle.dump(self.models, f)
        logger.info(f"  ✅ 基础模型: {base_models_path}")

        # 保存元模型
        meta_path = output_dir / f"v400_meta_model_{timestamp}.pkl"
        with open(meta_path, 'wb') as f:
            pickle.dump(self.meta_model, f)
        logger.info(f"  ✅ 元模型: {meta_path}")

        version = 'v4.2' if self.v42_mode else 'v4.1'

        # 保存权重和配置
        weights = {
            'feature_names': self.feature_names,
            'target_weights': self.target_weights,
            'winsorize_bounds': {k: (float(v[0]), float(v[1])) for k, v in self.winsorize_bounds.items()} if self.winsorize_bounds else None,
            'model_names': list(self.models.keys()),
            'meta_model_type': self.meta_model_type,
            'market_scale': self.market_scale,
            'industry_scale': self.industry_scale,
            'timestamp': timestamp,
            'version': version,
            'rank_label': self.rank_label,
            'neutralize_label': self.neutralize_label,
            'selected_features': self.selected_features,
            'v42_mode': self.v42_mode,
            'industry_excess_label': self.industry_excess_label,
            'robust_zscore_features': self.robust_zscore_features,
            'robust_zscore_labels': self.robust_zscore_labels,
            'join_v39_market': self.join_v39_market,
            'v39_market_feature_names': self.v39_market_feature_names,
        }
        weights_path = output_dir / f"v400_weights_{timestamp}.json"
        with open(weights_path, 'w') as f:
            json.dump(weights, f, indent=2)
        logger.info(f"  ✅ 权重配置: {weights_path}")

        # 保存完整系统 (兼容 V390 加载方式)
        full_model = {
            'base_models': self.models,
            'meta_model': self.meta_model,
            'feature_names': self.feature_names,
            'winsorize_bounds': self.winsorize_bounds,
            'target_weights': self.target_weights,
            'meta_model_type': self.meta_model_type,
            'market_scale': self.market_scale,
            'industry_scale': self.industry_scale,
            'timestamp': timestamp,
            'version': version,
            'rank_label': self.rank_label,
            'neutralize_label': self.neutralize_label,
            'selected_features': self.selected_features,
            'v42_mode': self.v42_mode,
            'industry_excess_label': self.industry_excess_label,
            'robust_zscore_features': self.robust_zscore_features,
            'robust_zscore_labels': self.robust_zscore_labels,
            'join_v39_market': self.join_v39_market,
            'v39_market_feature_names': self.v39_market_feature_names,
        }
        full_path = output_dir / f"v400_full_system_{timestamp}.pkl"
        with open(full_path, 'wb') as f:
            pickle.dump(full_model, f)
        logger.info(f"  ✅ 完整系统: {full_path}")

        # 创建latest符号链接
        latest_path = output_dir / "v400_full_system_latest.pkl"
        if latest_path.exists():
            latest_path.unlink()
        with open(latest_path, 'wb') as f:
            pickle.dump(full_model, f)
        logger.info(f"  ✅ Latest链接: {latest_path}")

        return str(full_path)

    def train(self, val_ratio=0.15, test_ratio=0.15, purge_days=10):
        """完整训练流程"""
        # 0. V4.2: 动态扩展MARKET_FEATURES以包含v39_*特征
        if self.v39_market_feature_names:
            self.MARKET_FEATURES = set(self.MARKET_FEATURES) | set(self.v39_market_feature_names)

        # 1. 加载数据 (含V4.2行业超额标签, Robust Z-Score标签)
        X, y, y_5d, y_5d_raw, dates, codes = self.load_cached_features()

        # 1.5 B4: IC-based特征筛选 (在split前对全量数据计算IC)
        if self.feature_select:
            X = self.ic_feature_selection(X, y, dates)

        # 2. 时序划分
        logger.info("\n" + "=" * 80)
        logger.info("📊 时序划分 (purge_gap=10天)...")
        logger.info("=" * 80)
        split = self.temporal_split(X, y, y_5d, y_5d_raw, dates, codes,
                                     val_ratio=val_ratio, test_ratio=test_ratio,
                                     purge_days=purge_days)

        # 3. Winsorization (V4.2 robust z-score模式下跳过，因为z-score已经归一化)
        if not self.robust_zscore_features:
            split['X_train'], split['X_val'], split['X_test'] = self.winsorize_features(
                split['X_train'], split['X_val'], split['X_test'])
        else:
            logger.info("📊 V4.2: 跳过Winsorization (将使用Robust Z-Score归一化)")
            # 仍然记录winsorize_bounds用于scorer的winsorize逻辑
            self.winsorize_bounds = {}
            for col in split['X_train'].columns:
                lower = split['X_train'][col].quantile(0.01)
                upper = split['X_train'][col].quantile(0.99)
                self.winsorize_bounds[col] = (lower, upper)

        # 3.2 V4.2: Cross-Sectional Robust Z-Score特征归一化
        if self.robust_zscore_features:
            split['X_train'], split['X_val'], split['X_test'] = self._robust_zscore_features(
                split['X_train'], split['X_val'], split['X_test'],
                split['dates_train'], split['dates_val'], split['dates_test'])

        # 3.5 市场/行业特征缩放
        split['X_train'], split['X_val'], split['X_test'] = self.scale_market_industry_features(
            split['X_train'], split['X_val'], split['X_test'])

        # 4. 训练基础模型
        self.train_base_models(split['X_train'], split['y_train'],
                               split['X_val'], split['y_val'])

        # 5. 训练元模型
        self.train_meta_model(split['X_val'], split['y_val'],
                              split['X_test'], split['y_test'])

        # 6. Cross-Sectional评估 (始终使用原始5d超额收益评估IC)
        metrics = self.evaluate_cross_sectional(
            split['X_test'], split['y_test'], split['y5d_raw_test'],
            split['dates_test'], split['codes_test'])

        # 7. 特征重要性
        self._log_feature_importance()

        # 8. 保存模型
        model_path = self.save_model()

        # 9. 保存评估报告
        self._save_evaluation_report(metrics)

        version_str = "V4.2 Hybrid Alpha" if self.v42_mode else "V4.0 Cross-Sectional Alpha"
        logger.info("\n" + "=" * 80)
        logger.info(f"🎉 {version_str} Model 训练完成!")
        logger.info("=" * 80)

        return model_path

    def _save_evaluation_report(self, metrics):
        """保存评估报告"""
        output_dir = Path(PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v400')
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / f"v400_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"💾 评估报告已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='V4.0/V4.2 Cross-Sectional Alpha Model 训练')
    parser.add_argument('--db-path', type=str,
                        default=str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db'),
                        help='数据库路径')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='验证集比例')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='测试集比例')
    parser.add_argument('--purge-days', type=int, default=10,
                        help='Purge gap天数 (应>=标签前瞻天数)')
    parser.add_argument('--meta-model', type=str, default='ridge',
                        choices=['ridge', 'gbm', 'avg'],
                        help='元模型类型: ridge(默认), gbm(已知过拟合), avg(等权平均)')
    parser.add_argument('--market-scale', type=float, default=0.3,
                        help='市场特征缩放因子 (默认0.3)')
    parser.add_argument('--industry-scale', type=float, default=0.5,
                        help='行业特征缩放因子 (默认0.5)')
    parser.add_argument('--skip-rf', action='store_true',
                        help='跳过Random Forest (训练慢且IC不稳定)')
    parser.add_argument('--rank-label', action='store_true',
                        help='B2: 将标签转为当日cross-sectional排名 (0-1)')
    parser.add_argument('--neutralize-label', action='store_true',
                        help='B3: 标签行业+市值中性化')
    parser.add_argument('--feature-select', action='store_true',
                        help='B4: IC-based特征筛选')

    # V4.2 新增参数
    parser.add_argument('--v42', action='store_true',
                        help='V4.2模式: 自动激活全部改进 (行业超额+RobustZScore+V39市场+5模型+超参)')
    parser.add_argument('--industry-excess-label', action='store_true',
                        help='V4.2: 行业超额标签 (label -= 行业中位数)')
    parser.add_argument('--robust-zscore-features', action='store_true',
                        help='V4.2: Robust Z-Score特征归一化 (cross-sectional per date)')
    parser.add_argument('--robust-zscore-labels', action='store_true',
                        help='V4.2: Robust Z-Score标签归一化')
    parser.add_argument('--join-v39-market', action='store_true',
                        help='V4.2: 从v39_feature_cache加载12个连续市场特征')
    parser.add_argument('--add-hgb', action='store_true',
                        help='V4.2: 添加HistGradientBoosting到模型Ensemble')

    args = parser.parse_args()

    # V4.2总开关覆盖冲突选项
    if args.v42:
        args.meta_model = 'avg'
        args.market_scale = 0.5
        args.skip_rf = False
        args.rank_label = False
        args.neutralize_label = False

    is_v42 = args.v42

    logger.info("=" * 80)
    if is_v42:
        logger.info("🚀 V4.2 Hybrid Alpha Model 训练")
        logger.info("   改进: 行业超额标签 + RobustZScore + V39市场特征 + 5模型Ensemble + 超参优化")
    else:
        logger.info("🚀 V4.1 Cross-Sectional Alpha Model 训练")
        logger.info("   目标: 学习个股相对强势信号，而非大盘方向")
    logger.info("=" * 80)
    logger.info(f"数据库: {args.db_path}")
    logger.info(f"元模型: {args.meta_model}")
    logger.info(f"市场特征缩放: {args.market_scale}")
    logger.info(f"行业特征缩放: {args.industry_scale}")
    if is_v42:
        logger.info(f"V4.2模式: ✅")
        logger.info(f"行业超额标签: {args.industry_excess_label or args.v42}")
        logger.info(f"RobustZScore特征: {args.robust_zscore_features or args.v42}")
        logger.info(f"RobustZScore标签: {args.robust_zscore_labels or args.v42}")
        logger.info(f"V39市场特征: {args.join_v39_market or args.v42}")
        logger.info(f"HGB模型: {args.add_hgb or args.v42}")
    else:
        logger.info(f"排名标签(B2): {args.rank_label}")
        logger.info(f"标签中性化(B3): {args.neutralize_label}")
        logger.info(f"特征筛选(B4): {args.feature_select}")
    logger.info(f"验证集比例: {args.val_ratio}")
    logger.info(f"测试集比例: {args.test_ratio}")
    logger.info(f"Purge gap: {args.purge_days} 天")

    trainer = V400CrossSectionalTrainer(
        db_path=args.db_path,
        meta_model_type=args.meta_model,
        market_scale=args.market_scale,
        industry_scale=args.industry_scale,
        skip_rf=args.skip_rf,
        rank_label=args.rank_label,
        neutralize_label=args.neutralize_label,
        feature_select=args.feature_select,
        v42_mode=args.v42,
        industry_excess_label=args.industry_excess_label,
        robust_zscore_features=args.robust_zscore_features,
        robust_zscore_labels=args.robust_zscore_labels,
        join_v39_market=args.join_v39_market,
        add_hgb=args.add_hgb,
    )
    model_path = trainer.train(
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        purge_days=args.purge_days
    )

    logger.info(f"\n✅ 模型已保存至: {model_path}")


if __name__ == "__main__":
    main()
