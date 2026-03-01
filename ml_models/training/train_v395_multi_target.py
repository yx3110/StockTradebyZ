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
        """计算所有日期的市场状态特征

        公式与 v39_feature_cache_updater._precompute_all_market_features() 对齐，
        确保训练和推理使用完全相同的特征定义。
        """
        logger.info("计算市场状态特征 (对齐DB缓存公式)...")

        conn = sqlite3.connect(self.db_path)

        # 获取沪深300指数数据 (与推理侧 v39_feature_cache 对齐)
        query = """
        SELECT q.trade_date, q.close, q.price_change_pct, q.volume
        FROM daily_quotes q
        JOIN securities s ON q.security_id = s.id
        WHERE s.code = '000300.SH'
        ORDER BY q.trade_date
        """
        df = pd.read_sql(query, conn)

        # 查询全市场每日上涨/总股票数 (用于 market_up_ratio)
        breadth_query = """
        SELECT q.trade_date,
               SUM(CASE WHEN q.price_change_pct > 0 THEN 1 ELSE 0 END) as up_count,
               COUNT(*) as total_count
        FROM daily_quotes q
        JOIN securities s ON q.security_id = s.id
        WHERE s.type = 'A股'
        GROUP BY q.trade_date
        ORDER BY q.trade_date
        """
        breadth_df = pd.read_sql(breadth_query, conn)
        conn.close()

        # 计算市场状态特征
        lookback = self.lookback

        # 1. 市场收益率 (与DB一致: pct_change)
        df['market_return_20d'] = df['close'].pct_change(lookback)
        df['market_return_10d'] = df['close'].pct_change(10)
        df['market_return_5d'] = df['close'].pct_change(5)

        # 2. 市场波动率 — 对齐DB: log-returns + 年化 sqrt(252)
        log_returns = np.log(df['close'] / df['close'].shift(1))
        df['market_volatility_20d'] = log_returns.rolling(lookback).std() * np.sqrt(252)
        df['market_volatility_10d'] = log_returns.rolling(10).std() * np.sqrt(252)

        # 3. 上涨比例 — 对齐DB: 全市场上涨股票比率 (非指数涨跌天数)
        breadth_df['up_ratio'] = breadth_df['up_count'] / breadth_df['total_count'].clip(lower=1)
        df = df.merge(breadth_df[['trade_date', 'up_ratio']], on='trade_date', how='left')
        df['up_ratio'] = df['up_ratio'].ffill()
        df['market_up_ratio_20d'] = df['up_ratio'].rolling(lookback).mean()
        df['market_up_ratio_10d'] = df['up_ratio'].rolling(10).mean()
        df.drop(columns=['up_ratio'], inplace=True)

        # 4. 最大回撤 — 对齐DB: 窗口内最大回撤 min((price - cummax) / cummax)
        def rolling_max_drawdown(closes, window):
            result = pd.Series(index=closes.index, dtype=float)
            for i in range(window - 1, len(closes)):
                window_closes = closes.iloc[i - window + 1:i + 1].values
                running_max = np.maximum.accumulate(window_closes)
                drawdowns = (window_closes - running_max) / running_max
                result.iloc[i] = np.min(drawdowns)
            return result

        df['market_drawdown_20d'] = rolling_max_drawdown(df['close'], lookback)

        # 5. 成交量变化 (与DB一致)
        df['market_volume_ratio'] = df['volume'] / df['volume'].rolling(lookback).mean()

        # 6. 趋势强度 (价格相对位置, 与DB一致)
        df['market_max_20d'] = df['close'].rolling(lookback).max()
        df['market_min_20d'] = df['close'].rolling(lookback).min()
        df['market_position_20d'] = (df['close'] - df['market_min_20d']) / \
                                     (df['market_max_20d'] - df['market_min_20d'] + 1e-8)

        # 7. 市场动量 — 对齐DB: 相对均线偏离 (非shift)
        df['market_momentum_20d'] = df['close'] / df['close'].rolling(lookback).mean() - 1
        df['market_momentum_5d'] = df['close'] / df['close'].rolling(5).mean() - 1

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
        # Phase 2: 风险调整标签融合比例 (0=纯收益, 0.3=推荐, 1=纯Sharpe)
        self.sharpe_label_blend = 0.3

    def _compute_global_quantiles(self, X: np.ndarray, all_results: dict,
                                    target_weights: dict, n_quantiles: int = 1001) -> np.ndarray:
        """计算全局 combined_pred 分位数分布 (用于全局百分位评分)

        在全量数据上运行训练好的集成模型, 收集所有 combined_pred,
        然后计算 n_quantiles 个分位点. 保存到模型文件后, 推理时用
        np.searchsorted 将新预测映射到全局百分位 (0-100).

        Args:
            X: 全量特征矩阵 (train+val+test)
            all_results: 训练好的模型 {target: {'models': {...}, 'weights': {...}}}
            target_weights: 目标权重 {'label_3d': 0.4, ...}
            n_quantiles: 分位点数量 (默认1001, 即0.0%到100.0%)

        Returns:
            np.ndarray: shape=(n_quantiles,), 分位数边界值 (从小到大排列)
        """
        logger.info(f"计算全局评分分位数 (n={X.shape[0]:,} 样本, {n_quantiles} 分位点)...")

        predictions = {}
        for target_key, result in all_results.items():
            target_pred = np.zeros(X.shape[0])
            total_weight = 0

            for name, model in result['models'].items():
                weight = result['weights'].get(name, 0.2)
                try:
                    if name == 'xgb':
                        import xgboost as xgb
                        pred = model.predict(xgb.DMatrix(X))
                    else:
                        pred = model.predict(X)
                    target_pred += weight * pred
                    total_weight += weight
                except Exception as e:
                    logger.warning(f"  全局分位数: {target_key}/{name} 预测失败: {e}")
                    continue

            if total_weight > 0:
                target_pred /= total_weight
            predictions[target_key] = target_pred

        # 计算 combined_pred (加权融合)
        combined_pred = np.zeros(X.shape[0])
        for target_key, pred in predictions.items():
            w = target_weights.get(f'label_{target_key}', 0)
            combined_pred += w * pred

        # 计算分位数
        quantile_points = np.linspace(0, 100, n_quantiles)
        global_quantiles = np.percentile(combined_pred, quantile_points)

        # 统计信息
        logger.info(f"  combined_pred 分布: min={combined_pred.min():.6f}, "
                     f"median={np.median(combined_pred):.6f}, max={combined_pred.max():.6f}")
        logger.info(f"  P1={global_quantiles[10]:.6f}, P25={global_quantiles[250]:.6f}, "
                     f"P50={global_quantiles[500]:.6f}, P75={global_quantiles[750]:.6f}, "
                     f"P99={global_quantiles[990]:.6f}")
        logger.info(f"  全局分位数计算完成, 将嵌入模型文件")

        return global_quantiles.tolist()

    def load_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """加载训练数据"""
        logger.info("加载训练数据...")

        conn = sqlite3.connect(self.db_path)

        # 构建日期过滤
        date_filter = ""
        date_params = []
        if start_date:
            date_filter += " AND v.trade_date >= ?"
            date_params.append(start_date)
        if end_date:
            date_filter += " AND v.trade_date <= ?"
            date_params.append(end_date)

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

        df = pd.read_sql(query, conn, params=date_params if date_params else None)
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
        query_basic = """
        SELECT s.code, db.trade_date,
               db.pe_ttm, db.pb, db.ps_ttm, db.turnover_rate, db.circ_mv
        FROM daily_basic db
        JOIN securities s ON db.security_id = s.id
        WHERE db.trade_date >= ? AND db.trade_date <= ?
        """
        df_basic = pd.read_sql(query_basic, conn2, params=[date_min, date_max])
        conn2.close()
        logger.info(f"    daily_basic 记录: {len(df_basic):,}")

        # 合并额外特征
        df_features = df_features.merge(df_basic, on=['code', 'trade_date'], how='left')

        # 计算 log_market_cap (对数市值, 消除量纲)
        df_features['log_market_cap'] = np.log1p(df_features['circ_mv'].fillna(0))
        df_features.drop(columns=['circ_mv'], inplace=True, errors='ignore')

        # 填充缺失的额外特征 (约5-10%股票可能缺少daily_basic数据)
        # 使用当日截面中位数填充，与推理侧 v395_production_scorer 保持一致
        for col in ['pe_ttm', 'pb', 'ps_ttm', 'turnover_rate', 'log_market_cap']:
            missing = df_features[col].isnull().sum()
            if missing > 0:
                logger.info(f"    {col}: {missing:,} 缺失 ({missing/len(df_features)*100:.1f}%), 用当日截面中位数填充")
                df_features[col] = df_features.groupby('trade_date')[col].transform(
                    lambda x: x.fillna(x.median())
                )
                # 如果某天全部缺失，用全局中位数兜底
                remaining = df_features[col].isnull().sum()
                if remaining > 0:
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

        # ===== 标签 Winsorization + Sharpe融合 延迟到 split_data() 后执行 =====
        # 原因: 在全量数据上计算quantile/std会导致test集信息泄漏到训练集
        # 正确做法: 仅在训练集上计算统计量，再应用到val/test
        # 标记: 这些步骤将在 _apply_label_transforms() 中执行
        self._pending_label_winsorize = True
        self._pending_sharpe_blend = self.sharpe_label_blend > 0
        logger.info(f"  标签Winsorization和Sharpe融合将在train/val/test分割后执行(防止数据泄漏)")

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

        # NOTE: Winsorization moved to split_data() to avoid data leakage
        # (bounds must be computed on training set only)

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

        # Per-feature winsorization: bounds computed on TRAIN only, applied to all splits
        X_train, self.winsorize_bounds = self.winsorize_features(X_train)
        logger.info(f"  特征 winsorization (训练集): {len(self.winsorize_bounds)} 列, 1st/99th percentile")
        # Apply train bounds to val/test
        for i, (lo, hi) in enumerate(self.winsorize_bounds):
            X_val[:, i] = np.clip(X_val[:, i], lo, hi)
            X_test[:, i] = np.clip(X_test[:, i], lo, hi)

        purged_samples = len(X) - len(X_train) - len(X_val) - len(X_test)

        logger.info(f"  训练集: {len(X_train):,} 样本, <= {train_date_end}")
        logger.info(f"  验证集: {len(X_val):,} 样本, {val_date_start} ~ {val_date_end}")
        logger.info(f"  测试集: {len(X_test):,} 样本, >= {test_date_start}")
        logger.info(f"  Purge gap: {purge_days} 个交易日, 丢弃 {purged_samples:,} 个样本")

        # 保存验证集/测试集日期 (用于IC-based权重计算)
        self.val_dates = dates[val_mask]
        self.test_dates = dates[test_mask]
        self.train_dates = dates[train_mask]

        # ===== 在split后应用标签变换(仅用训练集统计量，防止数据泄漏) =====
        label_sets = [
            (y_3d_train, y_3d_val, y_3d_test, 'label_3d'),
            (y_5d_train, y_5d_val, y_5d_test, 'label_5d'),
            (y_10d_train, y_10d_val, y_10d_test, 'label_10d'),
        ]

        if getattr(self, '_pending_label_winsorize', False):
            logger.info("  标签Winsorization (仅用训练集统计量)...")
            for y_tr, y_va, y_te, name in label_sets:
                lo = np.percentile(y_tr, 1)
                hi = np.percentile(y_tr, 99)
                n_clipped_tr = int(((y_tr < lo) | (y_tr > hi)).sum())
                y_tr[:] = np.clip(y_tr, lo, hi)
                y_va[:] = np.clip(y_va, lo, hi)
                y_te[:] = np.clip(y_te, lo, hi)
                logger.info(f"    {name}: [{lo:.4f}, {hi:.4f}], 训练集裁剪{n_clipped_tr:,}个")

        if getattr(self, '_pending_sharpe_blend', False):
            logger.info(f"  风险调整标签融合 (blend={self.sharpe_label_blend:.0%}, 仅用训练集统计量)...")
            for y_tr, y_va, y_te, name in label_sets:
                train_dates_arr = self.train_dates
                # 每日截面内的收益波动率 (仅训练集)
                unique_train_dates = np.unique(train_dates_arr)
                daily_vol_tr = np.zeros_like(y_tr)
                for d in unique_train_dates:
                    mask_d = train_dates_arr == d
                    std_d = np.std(y_tr[mask_d])
                    daily_vol_tr[mask_d] = std_d if std_d > 0 else 0
                # Sharpe-adjusted: 收益/波动
                sharpe_tr = y_tr / (daily_vol_tr + 1e-6)
                # 标准化尺度 (仅用训练集std)
                orig_std = np.std(y_tr)
                sharpe_std = np.std(sharpe_tr)
                if sharpe_std > 1e-8:
                    scale = orig_std / sharpe_std
                else:
                    scale = 1.0
                # 对训练集应用融合
                y_tr[:] = (1 - self.sharpe_label_blend) * y_tr + self.sharpe_label_blend * sharpe_tr * scale

                # 对val/test用训练集的mean daily_vol (防止数据泄漏)
                mean_daily_vol_train = np.mean([np.std(y_tr[train_dates_arr == d])
                    for d in unique_train_dates if (train_dates_arr == d).sum() > 1])
                mean_daily_vol_train = max(mean_daily_vol_train, 1e-6)

                for y_set, dates_set in [(y_va, self.val_dates), (y_te, self.test_dates)]:
                    if len(y_set) == 0:
                        continue
                    daily_vol_set = np.full_like(y_set, mean_daily_vol_train)
                    sharpe_set = y_set / (daily_vol_set + 1e-6)
                    y_set[:] = (1 - self.sharpe_label_blend) * y_set + self.sharpe_label_blend * sharpe_set * scale

                logger.info(f"    {name}: 训练集融合后std={np.std(y_tr):.6f}")

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

        # 5. 测试集评估 (独立推理 + 北极星Daily IC/ICIR)
        logger.info("\n" + "=" * 60)
        logger.info("测试集评估 (独立推理 + 北极星指标)")
        logger.info("=" * 60)

        final_metrics = {}
        ensemble_predictions = {}
        test_dates = self.test_dates  # 用于Daily IC计算

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

            # 过滤NaN用于评估
            valid_mask = ~(np.isnan(y_te) | np.isnan(ensemble_pred))
            y_te_valid = y_te[valid_mask]
            pred_valid = ensemble_pred[valid_mask]
            if valid_mask.sum() < len(valid_mask):
                logger.warning(f"  {target_key}: {(~valid_mask).sum()}个NaN值已跳过")

            rmse = np.sqrt(mean_squared_error(y_te_valid, pred_valid))
            ic_global, _ = spearmanr(pred_valid, y_te_valid)
            direction_acc = np.mean((pred_valid > 0) == (y_te_valid > 0))
            top10_idx = np.argsort(pred_valid)[-int(len(pred_valid)*0.1):]
            top20_idx = np.argsort(pred_valid)[-int(len(pred_valid)*0.2):]
            top10_return = np.mean(y_te_valid[top10_idx])
            top20_return = np.mean(y_te_valid[top20_idx])

            # 北极星: Daily IC / ICIR / IC>0%
            daily_ics = []
            unique_test_dates = np.unique(test_dates)
            for date in unique_test_dates:
                mask = test_dates == date
                n = mask.sum()
                if n < 20:  # 至少20只股票才有统计意义
                    continue
                day_ic, _ = spearmanr(ensemble_pred[mask], y_te[mask])
                if not np.isnan(day_ic):
                    daily_ics.append(day_ic)

            if daily_ics:
                daily_ic_mean = np.mean(daily_ics)
                daily_ic_std = np.std(daily_ics)
                daily_icir = daily_ic_mean / daily_ic_std if daily_ic_std > 1e-8 else 0
                daily_ic_pos_pct = np.mean(np.array(daily_ics) > 0) * 100
            else:
                daily_ic_mean = daily_ic_std = daily_icir = daily_ic_pos_pct = 0

            logger.info(f"\n{target_key} 目标:")
            logger.info(f"  RMSE: {rmse:.4f}")
            logger.info(f"  全局IC (single-shot): {ic_global:.4f}  ⚠️ 仅供参考,易虚高")
            logger.info(f"  ── 北极星 Daily IC ──")
            logger.info(f"  Daily IC均值: {daily_ic_mean:.4f} ± {daily_ic_std:.4f}  ({len(daily_ics)}天)")
            logger.info(f"  ICIR:         {daily_icir:.4f}")
            logger.info(f"  IC>0占比:     {daily_ic_pos_pct:.1f}%")
            logger.info(f"  ────────────────────")
            logger.info(f"  方向准确率: {direction_acc:.4f}")
            logger.info(f"  Top10收益: {top10_return:.4f}")
            logger.info(f"  Top20收益: {top20_return:.4f}")

            final_metrics[target_key] = {
                'rmse': rmse,
                'ic_global': ic_global,
                'daily_ic_mean': daily_ic_mean,
                'daily_ic_std': daily_ic_std,
                'daily_icir': daily_icir,
                'daily_ic_positive_pct': daily_ic_pos_pct,
                'direction_accuracy': direction_acc,
                'top10_return': top10_return,
                'top20_return': top20_return,
                'n_ic_days': len(daily_ics),
            }
            # 兼容旧字段
            final_metrics[target_key]['ic'] = daily_ic_mean

        # 6. Phase 2: 动态ICIR加权融合
        # 用测试集上各目标的ICIR重新分配权重，偏向信号更稳定的目标
        logger.info("\n" + "=" * 60)
        logger.info("融合预测评估 (ICIR动态加权 + 北极星指标)")
        logger.info("=" * 60)

        # 计算ICIR-based权重
        icir_map = {
            'label_3d': max(final_metrics.get('3d', {}).get('daily_icir', 0), 0),
            'label_5d': max(final_metrics.get('5d', {}).get('daily_icir', 0), 0),
            'label_10d': max(final_metrics.get('10d', {}).get('daily_icir', 0), 0),
        }
        icir_total = sum(icir_map.values())
        if icir_total > 0.01:
            dynamic_weights = {k: v / icir_total for k, v in icir_map.items()}
            # 混合: 50%静态 + 50%动态, 避免完全依赖样本内ICIR
            blended_weights = {}
            for k in self.target_weights:
                blended_weights[k] = 0.5 * self.target_weights[k] + 0.5 * dynamic_weights[k]
            logger.info(f"  静态权重: 3d={self.target_weights['label_3d']:.2f}, 5d={self.target_weights['label_5d']:.2f}, 10d={self.target_weights['label_10d']:.2f}")
            logger.info(f"  ICIR权重:  3d={dynamic_weights['label_3d']:.2f}, 5d={dynamic_weights['label_5d']:.2f}, 10d={dynamic_weights['label_10d']:.2f}")
            logger.info(f"  融合权重: 3d={blended_weights['label_3d']:.2f}, 5d={blended_weights['label_5d']:.2f}, 10d={blended_weights['label_10d']:.2f}")
        else:
            blended_weights = self.target_weights
            logger.info("  ICIR全为0, 使用静态权重")

        # 保存动态权重到模型（供生产scorer使用）
        self.dynamic_weights = blended_weights

        fused_pred = (
            blended_weights['label_3d'] * ensemble_predictions['3d'] +
            blended_weights['label_5d'] * ensemble_predictions['5d'] +
            blended_weights['label_10d'] * ensemble_predictions['10d']
        )

        # 对5天收益评估 (过滤NaN)
        fused_valid = ~(np.isnan(y_5d_test) | np.isnan(fused_pred))
        rmse = np.sqrt(mean_squared_error(y_5d_test[fused_valid], fused_pred[fused_valid]))
        ic_global, _ = spearmanr(fused_pred[fused_valid], y_5d_test[fused_valid])
        direction_acc = np.mean((fused_pred[fused_valid] > 0) == (y_5d_test[fused_valid] > 0))

        # 融合预测的Daily IC
        fused_daily_ics = []
        for date in unique_test_dates:
            mask = test_dates == date
            n = mask.sum()
            if n < 20:
                continue
            day_ic, _ = spearmanr(fused_pred[mask], y_5d_test[mask])
            if not np.isnan(day_ic):
                fused_daily_ics.append(day_ic)

        if fused_daily_ics:
            fused_ic_mean = np.mean(fused_daily_ics)
            fused_ic_std = np.std(fused_daily_ics)
            fused_icir = fused_ic_mean / fused_ic_std if fused_ic_std > 1e-8 else 0
            fused_ic_pos = np.mean(np.array(fused_daily_ics) > 0) * 100
        else:
            fused_ic_mean = fused_ic_std = fused_icir = fused_ic_pos = 0

        logger.info(f"融合预测 → 5天收益:")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  全局IC: {ic_global:.4f}  ⚠️ 仅供参考")
        logger.info(f"  ── 北极星 Daily IC ──")
        logger.info(f"  Daily IC均值: {fused_ic_mean:.4f} ± {fused_ic_std:.4f}  ({len(fused_daily_ics)}天)")
        logger.info(f"  ICIR:         {fused_icir:.4f}")
        logger.info(f"  IC>0占比:     {fused_ic_pos:.1f}%")
        logger.info(f"  ────────────────────")
        logger.info(f"  方向准确率: {direction_acc:.4f}")

        # 北极星达标评估
        logger.info("\n" + "=" * 60)
        logger.info("北极星目标达标评估")
        logger.info("=" * 60)
        for target_key in ['3d', '5d', '10d']:
            m = final_metrics[target_key]
            icir = m['daily_icir']
            ic = m['daily_ic_mean']
            status_ic = "✅" if ic >= 0.03 else "❌"
            status_icir = "✅" if icir >= 0.30 else "❌"
            logger.info(f"  {target_key}: DailyIC={ic:.4f} {status_ic}(≥0.03) | ICIR={icir:.4f} {status_icir}(≥0.30)")
        logger.info(f"  融合(5d): DailyIC={fused_ic_mean:.4f} {'✅' if fused_ic_mean >= 0.03 else '❌'}(≥0.03) | "
                    f"ICIR={fused_icir:.4f} {'✅' if fused_icir >= 0.30 else '❌'}(≥0.30)")

        final_metrics['fused'] = {
            'daily_ic_mean': fused_ic_mean,
            'daily_ic_std': fused_ic_std,
            'daily_icir': fused_icir,
            'daily_ic_positive_pct': fused_ic_pos,
            'ic_global': ic_global,
            'direction_accuracy': direction_acc,
            'rmse': rmse,
        }

        # 7. 特征重要性分析
        self._log_feature_importance(all_results)

        # 7.5 计算全局评分分位数 (仅用测试集，避免训练数据泄漏)
        global_quantiles = self._compute_global_quantiles(X_test, all_results, self.target_weights)

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
            'dynamic_weights': getattr(self, 'dynamic_weights', self.target_weights),
            'sharpe_label_blend': self.sharpe_label_blend,
            'market_features': list(self.market_calculator.market_features.columns[1:]),
            'winsorize_bounds': getattr(self, 'winsorize_bounds', None),
            'global_quantiles': global_quantiles,
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

        # 保存训练历史 (含北极星指标)
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
                'final_metrics': final_metrics,
            },
            'north_star_metrics': {
                '3d': {
                    'daily_ic': final_metrics.get('3d', {}).get('daily_ic_mean', 0),
                    'icir': final_metrics.get('3d', {}).get('daily_icir', 0),
                    'ic_positive_pct': final_metrics.get('3d', {}).get('daily_ic_positive_pct', 0),
                },
                '5d': {
                    'daily_ic': final_metrics.get('5d', {}).get('daily_ic_mean', 0),
                    'icir': final_metrics.get('5d', {}).get('daily_icir', 0),
                    'ic_positive_pct': final_metrics.get('5d', {}).get('daily_ic_positive_pct', 0),
                },
                '10d': {
                    'daily_ic': final_metrics.get('10d', {}).get('daily_ic_mean', 0),
                    'icir': final_metrics.get('10d', {}).get('daily_icir', 0),
                    'ic_positive_pct': final_metrics.get('10d', {}).get('daily_ic_positive_pct', 0),
                },
                'fused_5d': {
                    'daily_ic': final_metrics.get('fused', {}).get('daily_ic_mean', 0),
                    'icir': final_metrics.get('fused', {}).get('daily_icir', 0),
                    'ic_positive_pct': final_metrics.get('fused', {}).get('daily_ic_positive_pct', 0),
                },
            },
            'target_weights': self.target_weights,
            'dynamic_weights': getattr(self, 'dynamic_weights', self.target_weights),
            'sharpe_label_blend': self.sharpe_label_blend,
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


class V43Trainer(V395MultiTargetTrainer):
    """V4.3 训练器 — 在 V3.96 基础上增强

    改进:
    1. 扩展特征 (49→59): +KDJ/MACD/Bollinger/ATR/高低位/上影线/偏度
    2. 强正则化: num_leaves=20, min_data_in_leaf=500, L1=1.0, L2=5.0
    3. Walk-Forward 验证: 6 个滚动窗口, 报告 mean±std
    4. 样本加权: 涨跌停×0.3, 极端标签×0.5
    5. 等权集成: 所有模型 1/N (更稳健)
    6. 增加 15d 目标: 4 目标 3d/5d/10d/15d
    """

    def __init__(self, db_path: str = DB_PATH):
        super().__init__(db_path=db_path)
        # 4 目标权重
        self.target_weights = {
            'label_3d': 0.25,
            'label_5d': 0.30,
            'label_10d': 0.25,
            'label_15d': 0.20,
        }
        # 新增技术指标特征名
        self.extra_tech_feature_names = [
            'kdj_k', 'kdj_j', 'macd_dif', 'macd_dea', 'macd_hist',
            'boll_position', 'atr_14_pct',
            'high_low_position', 'upper_shadow_ratio',
            'return_skewness_proxy',
        ]

    def load_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """加载训练数据 — 扩展技术指标特征 + 15d 标签"""
        # 调用父类加载基础 49 特征 (含 industry-excess 标签)
        df = super().load_data(start_date, end_date)

        # ===== 新增: 15d 标签 =====
        logger.info("加载 15d 标签...")
        conn = sqlite3.connect(self.db_path)
        date_min = df['trade_date'].min()
        date_max = df['trade_date'].max()

        # 从 v39_feature_cache 获取 label_15d (可能不存在, 需要计算)
        # 先检查是否有 label_15d 列
        cursor = conn.execute("PRAGMA table_info(v39_feature_cache)")
        cache_cols = [row[1] for row in cursor.fetchall()]

        if 'label_15d' in cache_cols:
            query_15d = """
            SELECT code, trade_date, label_15d
            FROM v39_feature_cache
            WHERE trade_date >= ? AND trade_date <= ?
              AND label_15d IS NOT NULL
            """
            df_15d = pd.read_sql(query_15d, conn, params=[date_min, date_max])
            df = df.merge(df_15d, on=['code', 'trade_date'], how='left')
            logger.info(f"  label_15d 从缓存加载: {df_15d.shape[0]:,} 条")
        else:
            logger.info("  v39_feature_cache 无 label_15d 列, 从 daily_quotes 计算...")

        # 如果 label_15d 不存在或大量缺失, 从 daily_quotes 计算
        if 'label_15d' not in df.columns or df['label_15d'].isna().sum() > len(df) * 0.5:
            logger.info("  从 daily_quotes 计算 label_15d (未来15天收益)...")
            # 获取所有 A 股的 close 价格
            query_close = """
            SELECT s.code, q.trade_date, q.close
            FROM daily_quotes q
            JOIN securities s ON q.security_id = s.id
            WHERE s.type = 'A股' AND q.trade_date >= ?
            ORDER BY s.code, q.trade_date
            """
            df_close = pd.read_sql(query_close, conn, params=[date_min])
            df_close = df_close.sort_values(['code', 'trade_date'])
            df_close['future_close_15d'] = df_close.groupby('code')['close'].shift(-15)
            df_close['label_15d_calc'] = (df_close['future_close_15d'] / df_close['close'] - 1)
            df_close = df_close[['code', 'trade_date', 'label_15d_calc']].dropna()

            if 'label_15d' in df.columns:
                df.drop(columns=['label_15d'], inplace=True)
            df = df.merge(df_close, on=['code', 'trade_date'], how='left')
            df.rename(columns={'label_15d_calc': 'label_15d'}, inplace=True)
            logger.info(f"  label_15d 计算完成: {df['label_15d'].notna().sum():,} 条非空")

        # 行业超额收益标签 — 对 15d 也做
        if 'sw_l1_code' in df.columns and 'label_15d' in df.columns:
            industry_median = df.groupby(['trade_date', 'sw_l1_code'])['label_15d'].transform('median')
            raw_mean = df['label_15d'].mean()
            df['label_15d'] = df['label_15d'] - industry_median
            logger.info(f"  label_15d 行业超额: raw={raw_mean:.6f} → excess={df['label_15d'].mean():.6f}")

        # 标签 Winsorization — label_15d 延迟到 split 后执行 (防止数据泄漏)
        self._pending_label_15d_winsorize = 'label_15d' in df.columns
        if self._pending_label_15d_winsorize:
            logger.info("  label_15d Winsorization 将在 split 后执行 (防止数据泄漏)")

        # 丢弃 label_15d 为空的行
        before = len(df)
        df = df.dropna(subset=['label_15d'])
        logger.info(f"  过滤 label_15d 缺失后: {len(df):,} (丢弃 {before - len(df):,})")

        # ===== 新增: 技术指标特征 =====
        logger.info("加载技术指标特征 (KDJ/MACD/Bollinger/ATR)...")
        query_tech = """
        SELECT s.code, ti.trade_date,
               ti.kdj_k, ti.kdj_j, ti.macd_dif, ti.macd_dea, ti.macd_macd,
               ti.boll_upper, ti.boll_lower, ti.atr_14,
               q.close, q.high, q.low, q.is_limit_up, q.is_limit_down
        FROM technical_indicators ti
        JOIN securities s ON ti.security_id = s.id
        JOIN daily_quotes q ON q.security_id = s.id AND q.trade_date = ti.trade_date
        WHERE ti.trade_date >= ? AND ti.trade_date <= ?
          AND s.type = 'A股'
        """
        df_tech = pd.read_sql(query_tech, conn, params=[date_min, date_max])
        conn.close()
        logger.info(f"  技术指标记录: {len(df_tech):,}")

        # 计算衍生特征
        df_tech['macd_hist'] = df_tech['macd_macd']  # macd_macd 就是 MACD 柱状图 (DIF-DEA)*2
        boll_range = df_tech['boll_upper'] - df_tech['boll_lower']
        df_tech['boll_position'] = np.where(
            boll_range > 1e-6,
            (df_tech['close'] - df_tech['boll_lower']) / boll_range,
            0.5
        )
        df_tech['atr_14_pct'] = np.where(
            df_tech['close'] > 0,
            df_tech['atr_14'] / df_tech['close'],
            0.0
        )
        hl_range = df_tech['high'] - df_tech['low']
        df_tech['high_low_position'] = np.where(
            hl_range > 1e-6,
            (df_tech['close'] - df_tech['low']) / hl_range,
            0.5
        )
        df_tech['upper_shadow_ratio'] = np.where(
            hl_range > 1e-6,
            (df_tech['high'] - df_tech['close']) / hl_range,
            0.0
        )

        # 合并到主 DataFrame
        tech_merge_cols = ['code', 'trade_date',
                           'kdj_k', 'kdj_j', 'macd_dif', 'macd_dea', 'macd_hist',
                           'boll_position', 'atr_14_pct',
                           'high_low_position', 'upper_shadow_ratio',
                           'is_limit_up', 'is_limit_down']
        df = df.merge(df_tech[tech_merge_cols], on=['code', 'trade_date'], how='left')

        # 计算 return_skewness_proxy (from existing features)
        if 'max_pct_change_5d' in df.columns and 'avg_pct_change_5d' in df.columns and 'volatility_10d' in df.columns:
            vol = df['volatility_10d'].replace(0, np.nan).fillna(1e-6)
            df['return_skewness_proxy'] = (df['max_pct_change_5d'] - df['avg_pct_change_5d']) / vol
            df['return_skewness_proxy'] = df['return_skewness_proxy'].clip(-10, 10).fillna(0)
        else:
            df['return_skewness_proxy'] = 0.0
            logger.warning("  缺少 max/avg_pct_change_5d 或 volatility_10d, skewness_proxy 置 0")

        # 填充技术指标缺失值
        for col in self.extra_tech_feature_names:
            if col in df.columns:
                missing = df[col].isna().sum()
                if missing > 0:
                    df[col] = df[col].fillna(df[col].median() if df[col].notna().sum() > 0 else 0)
                    logger.info(f"    {col}: {missing:,} 缺失, 用中位数填充")

        # 填充涨跌停标志
        for col in ['is_limit_up', 'is_limit_down']:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)
            else:
                df[col] = 0

        logger.info(f"  技术指标特征合并完成: +{len(self.extra_tech_feature_names)} 特征")
        logger.info(f"  总样本: {len(df):,}, 总列数: {len(df.columns)}")

        return df

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """准备特征和标签 (含 label_15d)"""
        logger.info("准备特征和标签 (V4.3 扩展版)...")

        # 排除非特征列
        exclude_cols = ['code', 'trade_date', 'label_3d', 'label_5d', 'label_10d', 'label_15d',
                        'is_limit_up', 'is_limit_down']
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        # 特征分层 — 宏观 vs 个股
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

        # 截面 Robust Z-Score (个股特征)
        logger.info("  截面Robust Z-Score: 个股特征归一化")
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
        logger.info(f"  特征数量: {len(feature_cols)}")

        X = df[feature_cols].values
        y_3d = df['label_3d'].values
        y_5d = df['label_5d'].values
        y_10d = df['label_10d'].values
        y_15d = df['label_15d'].values

        # Per-feature winsorization
        X, self.winsorize_bounds = self.winsorize_features(X)

        return X, y_3d, y_5d, y_10d, y_15d, df

    def compute_sample_weights(self, df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """计算样本权重: 涨跌停×0.3, 极端标签×0.5"""
        weights = np.ones(len(y), dtype=np.float64)

        # 涨跌停降权
        if 'is_limit_up' in df.columns and 'is_limit_down' in df.columns:
            limit_mask = (df['is_limit_up'].values == 1) | (df['is_limit_down'].values == 1)
            weights[limit_mask] *= 0.3
            n_limit = limit_mask.sum()
            logger.info(f"    涨跌停降权: {n_limit:,} 样本 × 0.3")

        # 极端标签降权 (|z| > 3)
        if len(y) > 0:
            y_mean = np.nanmean(y)
            y_std = np.nanstd(y)
            if y_std > 1e-8:
                z_labels = np.abs((y - y_mean) / y_std)
                extreme_mask = z_labels > 3
                weights[extreme_mask] *= 0.5
                n_extreme = extreme_mask.sum()
                logger.info(f"    极端标签降权: {n_extreme:,} 样本 × 0.5")

        return weights

    def train_single_target_models(self, X_train, X_val, y_train, y_val, target_name: str,
                                    sample_weights_train=None):
        """为单个目标训练所有基础模型 — V4.3 强正则化"""
        logger.info(f"\n训练 {target_name} 模型 (V4.3 强正则化)...")

        models = {}
        predictions_train = {}
        predictions_val = {}

        import gc

        # 1. LightGBM — 强正则化
        logger.info(f"  训练 LightGBM ({target_name})...")
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 20,
            'learning_rate': 0.02,
            'feature_fraction': 0.6,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'reg_alpha': 1.0,
            'reg_lambda': 5.0,
            'min_data_in_leaf': 500,
            'min_gain_to_split': 0.01,
            'path_smooth': 10.0,
            'verbose': -1,
        }

        lgb_train = lgb.Dataset(X_train, label=y_train,
                                weight=sample_weights_train, free_raw_data=True)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train, free_raw_data=True)

        lgb_model = lgb.train(
            lgb_params, lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_train, lgb_val],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
        )
        models['lgb'] = lgb_model
        predictions_train['lgb'] = lgb_model.predict(X_train)
        predictions_val['lgb'] = lgb_model.predict(X_val)
        del lgb_train, lgb_val
        gc.collect()

        # 2. XGBoost — 强正则化
        logger.info(f"  训练 XGBoost ({target_name})...")
        xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 5,
            'learning_rate': 0.02,
            'subsample': 0.7,
            'colsample_bytree': 0.6,
            'reg_alpha': 1.0,
            'reg_lambda': 5.0,
            'min_child_weight': 100,
            'gamma': 0.1,
            'verbosity': 0,
        }

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        xgb_model = xgb.train(
            xgb_params, dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=30,
            verbose_eval=False
        )
        models['xgb'] = xgb_model
        predictions_train['xgb'] = xgb_model.predict(dtrain)
        predictions_val['xgb'] = xgb_model.predict(dval)
        del dtrain, dval
        gc.collect()

        # 3. CatBoost — 强正则化
        if HAS_CATBOOST:
            logger.info(f"  训练 CatBoost ({target_name})...")
            cb_model = cb.CatBoostRegressor(
                iterations=1000,
                learning_rate=0.02,
                depth=5,
                l2_leaf_reg=10,
                random_seed=42,
                verbose=False,
                early_stopping_rounds=30,
                min_data_in_leaf=500,
            )
            cb_pool_train = cb.Pool(X_train, label=y_train, weight=sample_weights_train)
            cb_pool_val = cb.Pool(X_val, label=y_val)
            cb_model.fit(cb_pool_train, eval_set=cb_pool_val, verbose=False)
            models['cb'] = cb_model
            predictions_train['cb'] = cb_model.predict(X_train)
            predictions_val['cb'] = cb_model.predict(X_val)
            del cb_pool_train, cb_pool_val
            gc.collect()

        # 4. RandomForest (子采样控制内存)
        logger.info(f"  训练 RandomForest ({target_name}, max_samples=200000)...")
        n_samples = X_train.shape[0]
        rf_max_samples = min(200_000, n_samples)
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            max_samples=rf_max_samples,
            max_features=0.6,
            min_samples_leaf=500,
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )
        rf_model.fit(X_train, y_train, sample_weight=sample_weights_train)
        models['rf'] = rf_model
        predictions_train['rf'] = rf_model.predict(X_train)
        predictions_val['rf'] = rf_model.predict(X_val)

        # 5. HistGradientBoosting — 强正则化
        logger.info(f"  训练 HistGradientBoosting ({target_name})...")
        hgb_model = HistGradientBoostingRegressor(
            max_iter=1000,
            learning_rate=0.02,
            max_depth=5,
            max_leaf_nodes=20,
            l2_regularization=5.0,
            min_samples_leaf=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=42,
            verbose=0,
        )
        # HGB 通过 sample_weight 参数传入
        hgb_model.fit(X_train, y_train, sample_weight=sample_weights_train)
        models['hgb'] = hgb_model
        predictions_train['hgb'] = hgb_model.predict(X_train)
        predictions_val['hgb'] = hgb_model.predict(X_val)

        return models, predictions_train, predictions_val

    def calculate_ensemble_weights(self, predictions_val: dict, y_val) -> dict:
        """等权平均 (更稳健, 不使用 IC-weighted)"""
        val_dates = getattr(self, 'val_dates', None)

        # 仍计算 IC 用于诊断日志
        mean_ics = {}
        if val_dates is not None:
            unique_dates = np.unique(val_dates)
            for name, pred in predictions_val.items():
                daily_ics = []
                for date in unique_dates:
                    mask = val_dates == date
                    n = mask.sum()
                    if n < 10:
                        continue
                    ic, _ = spearmanr(pred[mask], y_val[mask])
                    if not np.isnan(ic):
                        daily_ics.append(ic)
                mean_ics[name] = float(np.mean(daily_ics)) if daily_ics else 0.0
            logger.info(f"  模型IC (诊断): {', '.join(f'{k}={v:.4f}' for k, v in mean_ics.items())}")
        else:
            for name in predictions_val:
                mean_ics[name] = 0.0

        # 等权
        n = len(predictions_val)
        weights = {k: 1.0 / n for k in predictions_val}
        logger.info(f"  等权集成: {n} 模型, 每模型权重 = {1.0/n:.3f}")

        return weights, mean_ics

    def _calculate_daily_ic(self, pred, y, dates):
        """计算每日截面 IC 和 ICIR"""
        unique_dates = np.unique(dates)
        daily_ics = []
        for d in unique_dates:
            mask = dates == d
            n = mask.sum()
            if n < 10:
                continue
            ic, _ = spearmanr(pred[mask], y[mask])
            if not np.isnan(ic):
                daily_ics.append(ic)
        if not daily_ics:
            return 0.0, 0.0
        mean_ic = float(np.mean(daily_ics))
        std_ic = float(np.std(daily_ics))
        icir = mean_ic / std_ic if std_ic > 1e-8 else 0.0
        return mean_ic, icir

    def walk_forward_train(self, start_date: str = None, end_date: str = None,
                            purge_days: int = 15, min_train_days: int = 720,
                            val_days: int = 120, test_days: int = 120,
                            step_days: int = 120):
        """Walk-Forward 训练 + 验证"""
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("V4.3 Walk-Forward 训练")
        logger.info("=" * 60)

        # 1. 一次性加载全量数据
        df = self.load_data(start_date, end_date)
        X, y_3d, y_5d, y_10d, y_15d, df = self.prepare_features(df)

        dates = df['trade_date'].values
        unique_dates = np.sort(np.unique(dates))
        n_dates = len(unique_dates)
        logger.info(f"  总交易日: {n_dates}, 样本: {len(X):,}")

        # 2. 定义滚动窗口
        windows = []
        cursor = min_train_days
        while cursor + val_days + purge_days + test_days <= n_dates:
            train_end_idx = cursor - 1
            val_start_idx = cursor + purge_days
            val_end_idx = val_start_idx + val_days - 1
            test_start_idx = val_end_idx + 1 + purge_days
            test_end_idx = test_start_idx + test_days - 1

            if test_end_idx >= n_dates:
                break

            windows.append({
                'train_end': unique_dates[train_end_idx],
                'val_start': unique_dates[val_start_idx],
                'val_end': unique_dates[val_end_idx],
                'test_start': unique_dates[test_start_idx],
                'test_end': unique_dates[test_end_idx],
            })
            cursor += step_days

        logger.info(f"  Walk-Forward 窗口数: {len(windows)}")
        for i, w in enumerate(windows):
            logger.info(f"    窗口 {i+1}: train<='{w['train_end']}', val={w['val_start']}~{w['val_end']}, "
                         f"test={w['test_start']}~{w['test_end']}")

        # 3. 对每个窗口训练+评估
        wf_metrics = []
        import gc

        for wi, w in enumerate(windows):
            logger.info(f"\n{'='*50}")
            logger.info(f"Walk-Forward 窗口 {wi+1}/{len(windows)}")
            logger.info(f"{'='*50}")

            train_mask = dates <= w['train_end']
            val_mask = (dates >= w['val_start']) & (dates <= w['val_end'])
            test_mask = (dates >= w['test_start']) & (dates <= w['test_end'])

            X_train_w, X_val_w, X_test_w = X[train_mask].copy(), X[val_mask].copy(), X[test_mask].copy()
            y_3d_tr, y_3d_va, y_3d_te = y_3d[train_mask].copy(), y_3d[val_mask].copy(), y_3d[test_mask].copy()
            y_5d_tr, y_5d_va, y_5d_te = y_5d[train_mask].copy(), y_5d[val_mask].copy(), y_5d[test_mask].copy()
            y_10d_tr, y_10d_va, y_10d_te = y_10d[train_mask].copy(), y_10d[val_mask].copy(), y_10d[test_mask].copy()
            y_15d_tr, y_15d_va, y_15d_te = y_15d[train_mask].copy(), y_15d[val_mask].copy(), y_15d[test_mask].copy()
            test_dates_w = dates[test_mask]

            # Walk-Forward: 特征Winsorization (仅用窗口内训练集, 防止数据泄漏)
            X_train_w, wf_bounds = self.winsorize_features(X_train_w)
            for i, (lo, hi) in enumerate(wf_bounds):
                X_val_w[:, i] = np.clip(X_val_w[:, i], lo, hi)
                X_test_w[:, i] = np.clip(X_test_w[:, i], lo, hi)

            # Walk-Forward: 标签Winsorization (仅用训练集统计量, 防止数据泄漏)
            for y_tr_w, y_va_w, y_te_w in [(y_3d_tr, y_3d_va, y_3d_te),
                                             (y_5d_tr, y_5d_va, y_5d_te),
                                             (y_10d_tr, y_10d_va, y_10d_te),
                                             (y_15d_tr, y_15d_va, y_15d_te)]:
                lo = np.percentile(y_tr_w, 1)
                hi = np.percentile(y_tr_w, 99)
                y_tr_w[:] = np.clip(y_tr_w, lo, hi)
                y_va_w[:] = np.clip(y_va_w, lo, hi)
                y_te_w[:] = np.clip(y_te_w, lo, hi)

            logger.info(f"  train={X_train_w.shape[0]:,}, val={X_val_w.shape[0]:,}, test={X_test_w.shape[0]:,}")

            self.val_dates = dates[val_mask]

            # 训练 4 目标
            targets_w = [
                ('3d', y_3d_tr, y_3d_va, y_3d_te),
                ('5d', y_5d_tr, y_5d_va, y_5d_te),
                ('10d', y_10d_tr, y_10d_va, y_10d_te),
                ('15d', y_15d_tr, y_15d_va, y_15d_te),
            ]

            window_metrics = {}
            df_train_w = df[train_mask]

            for target_key, y_tr, y_va, y_te in targets_w:
                sample_w = self.compute_sample_weights(df_train_w, y_tr)
                models, pred_train, pred_val = self.train_single_target_models(
                    X_train_w, X_val_w, y_tr, y_va, f"label_{target_key}",
                    sample_weights_train=sample_w)
                weights, _ = self.calculate_ensemble_weights(pred_val, y_va)

                # Test set prediction
                pred_test = {}
                for name, model in models.items():
                    if name == 'xgb':
                        pred_test[name] = model.predict(xgb.DMatrix(X_test_w))
                    else:
                        pred_test[name] = model.predict(X_test_w)

                ensemble_pred = self.ensemble_predict(pred_test, weights)
                ic, icir = self._calculate_daily_ic(ensemble_pred, y_te, test_dates_w)
                window_metrics[target_key] = {'ic': ic, 'icir': icir}
                logger.info(f"  {target_key}: IC={ic:.4f}, ICIR={icir:.4f}")

                # 释放模型内存
                del models, pred_train, pred_val, pred_test
                gc.collect()

            wf_metrics.append(window_metrics)

        # 4. Walk-Forward 汇总
        logger.info("\n" + "=" * 60)
        logger.info("Walk-Forward 汇总")
        logger.info("=" * 60)

        wf_summary = {}
        for target_key in ['3d', '5d', '10d', '15d']:
            ics = [m[target_key]['ic'] for m in wf_metrics if target_key in m]
            icirs = [m[target_key]['icir'] for m in wf_metrics if target_key in m]
            summary = {
                'mean_ic': float(np.mean(ics)),
                'std_ic': float(np.std(ics)),
                'mean_icir': float(np.mean(icirs)),
                'std_icir': float(np.std(icirs)),
                'n_windows': len(ics),
            }
            wf_summary[target_key] = summary
            logger.info(f"  {target_key}: IC={summary['mean_ic']:.4f}±{summary['std_ic']:.4f}, "
                         f"ICIR={summary['mean_icir']:.4f}±{summary['std_icir']:.4f}")

        # 5. 训练最终生产模型 (85% train + 15% val for early stopping)
        logger.info("\n" + "=" * 60)
        logger.info("训练最终生产模型 (全量数据)")
        logger.info("=" * 60)

        # 按时间分割: 85% train, 15% val
        split_idx = int(n_dates * 0.85)
        split_date = unique_dates[split_idx]
        train_mask_final = dates <= split_date
        val_mask_final = dates > split_date

        X_train_f, X_val_f = X[train_mask_final], X[val_mask_final]
        self.val_dates = dates[val_mask_final]

        df_train_f = df[train_mask_final]
        all_results = {}

        targets_final = [
            ('3d', y_3d[train_mask_final].copy(), y_3d[val_mask_final].copy()),
            ('5d', y_5d[train_mask_final].copy(), y_5d[val_mask_final].copy()),
            ('10d', y_10d[train_mask_final].copy(), y_10d[val_mask_final].copy()),
            ('15d', y_15d[train_mask_final].copy(), y_15d[val_mask_final].copy()),
        ]

        # 生产模型: 标签Winsorization (仅用训练集统计量)
        for target_key, y_tr, y_va in targets_final:
            lo = np.percentile(y_tr, 1)
            hi = np.percentile(y_tr, 99)
            y_tr[:] = np.clip(y_tr, lo, hi)
            y_va[:] = np.clip(y_va, lo, hi)

        for target_key, y_tr, y_va in targets_final:
            sample_w = self.compute_sample_weights(df_train_f, y_tr)
            models, pred_train, pred_val = self.train_single_target_models(
                X_train_f, X_val_f, y_tr, y_va, f"label_{target_key}",
                sample_weights_train=sample_w)
            weights, rmses = self.calculate_ensemble_weights(pred_val, y_va)
            all_results[target_key] = {'models': models, 'weights': weights, 'rmses': rmses}

        # 6. 特征重要性分析
        self._log_feature_importance(all_results)

        # 6.5 计算全局评分分位数
        global_quantiles = self._compute_global_quantiles(X, all_results, self.target_weights)

        # 7. 保存模型
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        output_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v43'
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        model_data = {
            'version': 'v4.3',
            'models': all_results,
            'feature_names': self.feature_names,
            'target_weights': self.target_weights,
            'market_features': list(self.market_calculator.market_features.columns[1:]),
            'winsorize_bounds': getattr(self, 'winsorize_bounds', None),
            'global_quantiles': global_quantiles,
            # 模型类型标识
            'cascade': False,
            'rank_normalized': False,
            'robust_zscore': True,
            'industry_excess_labels': True,
            'dual_stream': False,
            'cross_sectional_neutralization': False,
            'macro_feature_cols': self.macro_feature_cols,
            'stock_feature_cols': self.stock_feature_cols,
            'extra_features_from_daily_basic': ['pe_ttm', 'pb', 'ps_ttm', 'turnover_rate', 'log_market_cap'],
            'extra_features_from_tech_indicators': self.extra_tech_feature_names,
            'targets': ['3d', '5d', '10d', '15d'],
            'ensemble_type': 'equal_weight',
            'sample_weighting': True,
            'walk_forward_metrics': wf_summary,
            'walk_forward_windows': len(windows),
            'regularization': {
                'num_leaves': 20, 'min_data_in_leaf': 500,
                'reg_alpha': 1.0, 'reg_lambda': 5.0,
                'path_smooth': 10.0, 'learning_rate': 0.02,
            },
        }

        model_path = output_dir / f'v43_multi_target_{timestamp}.pkl'
        joblib.dump(model_data, model_path)
        logger.info(f"\n模型已保存: {model_path}")

        # 保存训练历史
        history = {
            'version': 'v4.3',
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'status': 'completed',
            'summary': {
                'training_samples': int(train_mask_final.sum()),
                'validation_samples': int(val_mask_final.sum()),
                'feature_count': len(self.feature_names),
                'walk_forward_summary': wf_summary,
            },
            'target_weights': self.target_weights,
            'ensemble_weights': {k: all_results[k]['weights'] for k in all_results},
        }

        history_path = output_dir / f'training_history_{timestamp}.json'
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        latest_path = output_dir / 'training_history_latest.json'
        with open(latest_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        logger.info(f"训练历史已保存: {history_path}")
        logger.info(f"\nV4.3 训练完成! 总耗时: {duration:.0f}秒 ({duration/60:.1f}分钟)")

        return model_data, history


class V44Trainer(V43Trainer):
    """V4.4 训练器 — V4.3信号底座 + 6个增强模块

    目标: 北极星V2 80+/105 (76%+) A+级

    Module A: 单调性校准 (IC Monotonicity 2.9→3.8+)
      - 保序回归校准: 训练阶段在验证集拟合 score→return 的保序映射
      - 单调性感知集成权重: 60% IC + 40% quintile单调性

    Module B: 流动性感知 (Liquidity Coverage 80%→95%+)
      - 低换手率标签折扣: turnover_rate<0.5% 时 label×0.5

    Module C: 市况感知训练 (Worst 60d ICIR -0.001→0.10+)
      - 熊市样本加权: market_return_20d<-5% 时 weight×2.0
      - 熊市专家模型: 只用熊市样本训练单独LGB

    Module D: Sharpe标签融合 (来自V3.96)
      - 继承父类sharpe_label_blend=0.3, 调整目标权重偏向10d

    Module E: 可执行性过滤 (涨停失败率 11.5%→2%) → 在scorer层实现

    Module F: 市况自适应选股 (MaxDD→-10%) → 在scorer层实现
    """

    def __init__(self, db_path: str = DB_PATH):
        super().__init__(db_path=db_path)
        # Module D: Sharpe标签融合 + 调整权重偏向10d (对齐北极星10d评估)
        self.sharpe_label_blend = 0.3
        self.target_weights = {
            'label_3d': 0.20,   # 0.25→0.20
            'label_5d': 0.25,   # 0.30→0.25
            'label_10d': 0.35,  # 0.25→0.35 (增加, 对齐北极星10d评估)
            'label_15d': 0.20,  # 不变
        }

    def load_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """加载数据 — V4.3基础 + Module B 流动性折扣标签"""
        df = super().load_data(start_date, end_date)

        # Module B: 低换手率标签折扣 — 教模型"低流动性=低有效收益"
        if 'turnover_rate' in df.columns:
            low_liq = df['turnover_rate'] < 0.5  # 换手率<0.5%
            n_low = low_liq.sum()
            for col in ['label_3d', 'label_5d', 'label_10d', 'label_15d']:
                if col in df.columns:
                    df.loc[low_liq, col] *= 0.5
            logger.info(f"  Module B 流动性折扣: {n_low:,} 样本标签×0.5 (turnover_rate<0.5%)")
        else:
            logger.warning("  turnover_rate 不在数据中, 跳过流动性折扣")

        return df

    def compute_sample_weights(self, df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """Module C: 父类权重 + 熊市样本加权×2.0"""
        weights = super().compute_sample_weights(df, y)

        # 熊市样本加权 — 迫使模型学习逆境中选股
        if 'market_return_20d' in df.columns:
            bear = df['market_return_20d'].values < -0.05
            n_bear = bear.sum()
            weights[bear] *= 2.0
            logger.info(f"    熊市样本加权: {n_bear:,} 样本 × 2.0 (market_return_20d < -5%)")
        else:
            # 尝试从市场特征中找
            logger.info("    market_return_20d 未找到, 跳过熊市加权")

        return weights

    def calculate_ensemble_weights(self, predictions_val: dict, y_val) -> dict:
        """Module A: 60% IC加权 + 40% 单调性加权 (替代V4.3等权)"""
        val_dates = getattr(self, 'val_dates', None)

        if val_dates is None:
            return super().calculate_ensemble_weights(predictions_val, y_val)

        unique_dates = np.unique(val_dates)

        # 计算每个模型的 mean IC 和 quintile 单调性
        mean_ics = {}
        monotonicity_scores = {}

        for name, pred in predictions_val.items():
            # Daily IC
            daily_ics = []
            for date in unique_dates:
                mask = val_dates == date
                n = mask.sum()
                if n < 10:
                    continue
                ic, _ = spearmanr(pred[mask], y_val[mask])
                if not np.isnan(ic):
                    daily_ics.append(ic)
            mean_ics[name] = float(np.mean(daily_ics)) if daily_ics else 0.0

            # Quintile 单调性: 按预测分5组, 检查实际收益是否单调递增
            try:
                q_labels = pd.qcut(pred, q=5, labels=False, duplicates='drop')
                q_means = pd.Series(y_val).groupby(q_labels).mean()
                if len(q_means) >= 4:
                    # 计算相邻组均值差的符号: 理想情况全为正
                    diffs = np.diff(q_means.values)
                    monotonicity = np.mean(diffs > 0)  # 0~1, 1=完美单调
                else:
                    monotonicity = 0.5
            except Exception:
                monotonicity = 0.5
            monotonicity_scores[name] = monotonicity

        logger.info(f"  模型IC: {', '.join(f'{k}={v:.4f}' for k, v in mean_ics.items())}")
        logger.info(f"  单调性: {', '.join(f'{k}={v:.3f}' for k, v in monotonicity_scores.items())}")

        # 综合权重: 60% IC + 40% 单调性
        combined = {}
        for name in predictions_val:
            ic_score = max(mean_ics.get(name, 0), 0)
            mono_score = monotonicity_scores.get(name, 0.5)
            combined[name] = 0.6 * ic_score + 0.4 * mono_score

        total = sum(combined.values())
        if total < 1e-8:
            # 回退到等权
            n = len(predictions_val)
            weights = {k: 1.0 / n for k in predictions_val}
            logger.info(f"  综合权重为0, 回退等权: {n} 模型")
        else:
            weights = {k: v / total for k, v in combined.items()}
            logger.info(f"  IC+单调性权重: {', '.join(f'{k}={v:.3f}' for k, v in weights.items())}")

        return weights, mean_ics

    def _train_bear_specialist(self, X_train, y_train, df_train, target_key='10d'):
        """Module C: 熊市专家模型 — 只用熊市样本训练单LGB"""
        if 'market_return_20d' not in df_train.columns:
            logger.info("  熊市专家: market_return_20d 不可用, 跳过")
            return None

        bear_mask = df_train['market_return_20d'].values < -0.05
        n_bear = bear_mask.sum()
        if n_bear < 1000:
            logger.info(f"  熊市专家: 熊市样本不足 ({n_bear}), 跳过")
            return None

        X_bear = X_train[bear_mask]
        y_bear = y_train[bear_mask]

        logger.info(f"  训练熊市专家 ({target_key}): {n_bear:,} 样本...")

        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 15,      # 更简单的模型防过拟合
            'learning_rate': 0.03,
            'feature_fraction': 0.5,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'reg_alpha': 2.0,
            'reg_lambda': 10.0,
            'min_data_in_leaf': 300,
            'verbose': -1,
        }

        # 用随机20%做early stopping
        n = len(X_bear)
        split = int(n * 0.8)
        idx = np.random.RandomState(42).permutation(n)
        train_idx, val_idx = idx[:split], idx[split:]

        lgb_train = lgb.Dataset(X_bear[train_idx], label=y_bear[train_idx], free_raw_data=True)
        lgb_val = lgb.Dataset(X_bear[val_idx], label=y_bear[val_idx], reference=lgb_train, free_raw_data=True)

        bear_model = lgb.train(
            lgb_params, lgb_train,
            num_boost_round=500,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
        )

        # 评估
        pred_val = bear_model.predict(X_bear[val_idx])
        ic, _ = spearmanr(pred_val, y_bear[val_idx])
        logger.info(f"  熊市专家 {target_key}: bear val IC={ic:.4f}, rounds={bear_model.best_iteration}")

        del lgb_train, lgb_val
        import gc; gc.collect()

        return bear_model

    def _fit_isotonic_calibration(self, X_val, y_val_dict, all_results):
        """Module A: 在验证集上拟合保序回归校准"""
        from sklearn.isotonic import IsotonicRegression

        isotonic_models = {}

        for target_key in ['3d', '5d', '10d', '15d']:
            if target_key not in all_results:
                continue

            # 获取集成预测
            pred_val = {}
            models = all_results[target_key]['models']
            weights = all_results[target_key]['weights']

            for name, model in models.items():
                try:
                    if name == 'xgb':
                        import xgboost as xgb_lib
                        pred_val[name] = model.predict(xgb_lib.DMatrix(X_val))
                    else:
                        pred_val[name] = model.predict(X_val)
                except Exception:
                    continue

            if not pred_val:
                continue

            # 加权集成
            ensemble_pred = np.zeros(len(X_val))
            for name, pred in pred_val.items():
                w = weights.get(name, 1.0 / len(pred_val))
                ensemble_pred += w * pred
            total_w = sum(weights.get(n, 1.0/len(pred_val)) for n in pred_val)
            if total_w > 0:
                ensemble_pred /= total_w

            # 拟合保序回归: pred → actual return
            y_target = y_val_dict.get(target_key)
            if y_target is None:
                continue

            valid = ~(np.isnan(ensemble_pred) | np.isnan(y_target))
            if valid.sum() < 100:
                continue

            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(ensemble_pred[valid], y_target[valid])
            isotonic_models[target_key] = iso

            # 评估校准前后IC
            raw_ic, _ = spearmanr(ensemble_pred[valid], y_target[valid])
            calibrated_pred = iso.predict(ensemble_pred[valid])
            cal_ic, _ = spearmanr(calibrated_pred, y_target[valid])
            logger.info(f"  保序校准 {target_key}: IC {raw_ic:.4f} → {cal_ic:.4f}")

        return isotonic_models

    def walk_forward_train(self, start_date: str = None, end_date: str = None,
                            purge_days: int = 15, min_train_days: int = 900,
                            val_days: int = 120, test_days: int = 120,
                            step_days: int = 90):
        """V4.4 Walk-Forward 训练 — 增强版

        vs V4.3:
        - min_train_days: 720→900 (更多熊市样本)
        - step_days: 120→90 (更多窗口, 更平滑)
        - 新增: 熊市专家模型
        - 新增: 保序回归校准
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("V4.4 Walk-Forward 训练 (V4.3信号 + 6增强模块)")
        logger.info("=" * 60)
        logger.info(f"  参数: min_train={min_train_days}d, val={val_days}d, test={test_days}d, "
                     f"step={step_days}d, purge={purge_days}d")
        logger.info(f"  目标权重: {self.target_weights}")
        logger.info(f"  Sharpe融合: {self.sharpe_label_blend}")

        # 1. 一次性加载全量数据 (含V4.3扩展+Module B流动性折扣)
        df = self.load_data(start_date, end_date)
        X, y_3d, y_5d, y_10d, y_15d, df = self.prepare_features(df)

        dates = df['trade_date'].values
        unique_dates = np.sort(np.unique(dates))
        n_dates = len(unique_dates)
        logger.info(f"  总交易日: {n_dates}, 样本: {len(X):,}")

        # 2. 定义滚动窗口
        windows = []
        cursor = min_train_days
        while cursor + val_days + purge_days + test_days <= n_dates:
            train_end_idx = cursor - 1
            val_start_idx = cursor + purge_days
            val_end_idx = val_start_idx + val_days - 1
            test_start_idx = val_end_idx + 1 + purge_days
            test_end_idx = test_start_idx + test_days - 1

            if test_end_idx >= n_dates:
                break

            windows.append({
                'train_end': unique_dates[train_end_idx],
                'val_start': unique_dates[val_start_idx],
                'val_end': unique_dates[val_end_idx],
                'test_start': unique_dates[test_start_idx],
                'test_end': unique_dates[test_end_idx],
            })
            cursor += step_days

        logger.info(f"  Walk-Forward 窗口数: {len(windows)}")
        for i, w in enumerate(windows):
            logger.info(f"    窗口 {i+1}: train<='{w['train_end']}', val={w['val_start']}~{w['val_end']}, "
                         f"test={w['test_start']}~{w['test_end']}")

        # 3. 对每个窗口训练+评估
        wf_metrics = []
        import gc

        for wi, w in enumerate(windows):
            logger.info(f"\n{'='*50}")
            logger.info(f"Walk-Forward 窗口 {wi+1}/{len(windows)}")
            logger.info(f"{'='*50}")

            train_mask = dates <= w['train_end']
            val_mask = (dates >= w['val_start']) & (dates <= w['val_end'])
            test_mask = (dates >= w['test_start']) & (dates <= w['test_end'])

            X_train_w, X_val_w, X_test_w = X[train_mask].copy(), X[val_mask].copy(), X[test_mask].copy()
            y_3d_tr, y_3d_va, y_3d_te = y_3d[train_mask].copy(), y_3d[val_mask].copy(), y_3d[test_mask].copy()
            y_5d_tr, y_5d_va, y_5d_te = y_5d[train_mask].copy(), y_5d[val_mask].copy(), y_5d[test_mask].copy()
            y_10d_tr, y_10d_va, y_10d_te = y_10d[train_mask].copy(), y_10d[val_mask].copy(), y_10d[test_mask].copy()
            y_15d_tr, y_15d_va, y_15d_te = y_15d[train_mask].copy(), y_15d[val_mask].copy(), y_15d[test_mask].copy()
            test_dates_w = dates[test_mask]

            # Walk-Forward: 特征Winsorization (仅用窗口内训练集, 防止数据泄漏)
            X_train_w, wf_bounds = self.winsorize_features(X_train_w)
            for i, (lo, hi) in enumerate(wf_bounds):
                X_val_w[:, i] = np.clip(X_val_w[:, i], lo, hi)
                X_test_w[:, i] = np.clip(X_test_w[:, i], lo, hi)

            # Walk-Forward: 标签Winsorization (仅用训练集统计量, 防止数据泄漏)
            for y_tr_w, y_va_w, y_te_w in [(y_3d_tr, y_3d_va, y_3d_te),
                                             (y_5d_tr, y_5d_va, y_5d_te),
                                             (y_10d_tr, y_10d_va, y_10d_te),
                                             (y_15d_tr, y_15d_va, y_15d_te)]:
                lo = np.percentile(y_tr_w, 1)
                hi = np.percentile(y_tr_w, 99)
                y_tr_w[:] = np.clip(y_tr_w, lo, hi)
                y_va_w[:] = np.clip(y_va_w, lo, hi)
                y_te_w[:] = np.clip(y_te_w, lo, hi)

            logger.info(f"  train={X_train_w.shape[0]:,}, val={X_val_w.shape[0]:,}, test={X_test_w.shape[0]:,}")

            self.val_dates = dates[val_mask]

            # 训练 4 目标
            targets_w = [
                ('3d', y_3d_tr, y_3d_va, y_3d_te),
                ('5d', y_5d_tr, y_5d_va, y_5d_te),
                ('10d', y_10d_tr, y_10d_va, y_10d_te),
                ('15d', y_15d_tr, y_15d_va, y_15d_te),
            ]

            window_metrics = {}
            df_train_w = df[train_mask]

            for target_key, y_tr, y_va, y_te in targets_w:
                sample_w = self.compute_sample_weights(df_train_w, y_tr)
                models, pred_train, pred_val = self.train_single_target_models(
                    X_train_w, X_val_w, y_tr, y_va, f"label_{target_key}",
                    sample_weights_train=sample_w)
                weights, _ = self.calculate_ensemble_weights(pred_val, y_va)

                # Test set prediction
                pred_test = {}
                for name, model in models.items():
                    if name == 'xgb':
                        pred_test[name] = model.predict(xgb.DMatrix(X_test_w))
                    else:
                        pred_test[name] = model.predict(X_test_w)

                ensemble_pred = self.ensemble_predict(pred_test, weights)
                ic, icir = self._calculate_daily_ic(ensemble_pred, y_te, test_dates_w)
                window_metrics[target_key] = {'ic': ic, 'icir': icir}
                logger.info(f"  {target_key}: IC={ic:.4f}, ICIR={icir:.4f}")

                del models, pred_train, pred_val, pred_test
                gc.collect()

            wf_metrics.append(window_metrics)

        # 4. Walk-Forward 汇总
        logger.info("\n" + "=" * 60)
        logger.info("Walk-Forward 汇总")
        logger.info("=" * 60)

        wf_summary = {}
        for target_key in ['3d', '5d', '10d', '15d']:
            ics = [m[target_key]['ic'] for m in wf_metrics if target_key in m]
            icirs = [m[target_key]['icir'] for m in wf_metrics if target_key in m]
            summary = {
                'mean_ic': float(np.mean(ics)),
                'std_ic': float(np.std(ics)),
                'mean_icir': float(np.mean(icirs)),
                'std_icir': float(np.std(icirs)),
                'n_windows': len(ics),
            }
            wf_summary[target_key] = summary
            logger.info(f"  {target_key}: IC={summary['mean_ic']:.4f}±{summary['std_ic']:.4f}, "
                         f"ICIR={summary['mean_icir']:.4f}±{summary['std_icir']:.4f}")

        # 5. 训练最终生产模型 (85% train + 15% val for early stopping)
        logger.info("\n" + "=" * 60)
        logger.info("训练最终生产模型 (全量数据)")
        logger.info("=" * 60)

        split_idx = int(n_dates * 0.85)
        split_date = unique_dates[split_idx]
        train_mask_final = dates <= split_date
        val_mask_final = dates > split_date

        X_train_f, X_val_f = X[train_mask_final], X[val_mask_final]
        self.val_dates = dates[val_mask_final]

        df_train_f = df[train_mask_final]
        all_results = {}

        y_val_dict = {}
        targets_final = [
            ('3d', y_3d[train_mask_final].copy(), y_3d[val_mask_final].copy()),
            ('5d', y_5d[train_mask_final].copy(), y_5d[val_mask_final].copy()),
            ('10d', y_10d[train_mask_final].copy(), y_10d[val_mask_final].copy()),
            ('15d', y_15d[train_mask_final].copy(), y_15d[val_mask_final].copy()),
        ]

        # V44生产模型: 标签Winsorization (仅用训练集统计量)
        for target_key, y_tr, y_va in targets_final:
            lo = np.percentile(y_tr, 1)
            hi = np.percentile(y_tr, 99)
            y_tr[:] = np.clip(y_tr, lo, hi)
            y_va[:] = np.clip(y_va, lo, hi)

        for target_key, y_tr, y_va in targets_final:
            sample_w = self.compute_sample_weights(df_train_f, y_tr)
            models, pred_train, pred_val = self.train_single_target_models(
                X_train_f, X_val_f, y_tr, y_va, f"label_{target_key}",
                sample_weights_train=sample_w)
            weights, rmses = self.calculate_ensemble_weights(pred_val, y_va)
            all_results[target_key] = {'models': models, 'weights': weights, 'rmses': rmses}
            y_val_dict[target_key] = y_va

        # 6. Module C: 训练熊市专家模型 (只用10d目标)
        logger.info("\n" + "=" * 60)
        logger.info("Module C: 训练熊市专家模型")
        logger.info("=" * 60)
        bear_models = {}
        for target_key in ['10d', '15d']:
            y_tr_target = y_10d[train_mask_final] if target_key == '10d' else y_15d[train_mask_final]
            bear_model = self._train_bear_specialist(X_train_f, y_tr_target, df_train_f, target_key)
            if bear_model is not None:
                bear_models[target_key] = bear_model

        # 7. Module A: 保序回归校准
        logger.info("\n" + "=" * 60)
        logger.info("Module A: 保序回归校准")
        logger.info("=" * 60)
        isotonic_models = self._fit_isotonic_calibration(X_val_f, y_val_dict, all_results)

        # 8. 特征重要性分析
        self._log_feature_importance(all_results)

        # 8.5 计算全局评分分位数
        global_quantiles = self._compute_global_quantiles(X, all_results, self.target_weights)

        # 9. 保存模型
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        output_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v44'
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        model_data = {
            'version': 'v4.4',
            'models': all_results,
            'feature_names': self.feature_names,
            'target_weights': self.target_weights,
            'market_features': list(self.market_calculator.market_features.columns[1:]),
            'winsorize_bounds': getattr(self, 'winsorize_bounds', None),
            'global_quantiles': global_quantiles,
            # 模型类型标识
            'cascade': False,
            'rank_normalized': False,
            'robust_zscore': True,
            'industry_excess_labels': True,
            'dual_stream': False,
            'cross_sectional_neutralization': False,
            'macro_feature_cols': self.macro_feature_cols,
            'stock_feature_cols': self.stock_feature_cols,
            'extra_features_from_daily_basic': ['pe_ttm', 'pb', 'ps_ttm', 'turnover_rate', 'log_market_cap'],
            'extra_features_from_tech_indicators': self.extra_tech_feature_names,
            'targets': ['3d', '5d', '10d', '15d'],
            'ensemble_type': 'ic_monotonicity_weighted',
            'sample_weighting': True,
            'walk_forward_metrics': wf_summary,
            'walk_forward_windows': len(windows),
            'regularization': {
                'num_leaves': 20, 'min_data_in_leaf': 500,
                'reg_alpha': 1.0, 'reg_lambda': 5.0,
                'path_smooth': 10.0, 'learning_rate': 0.02,
            },
            # V4.4 新增组件
            'bear_models': bear_models,
            'isotonic_calibration': isotonic_models,
            'sharpe_label_blend': self.sharpe_label_blend,
            'liquidity_discount': True,
            'bear_sample_weighting': True,
            'min_train_days': min_train_days,
            'step_days': step_days,
        }

        model_path = output_dir / f'v44_multi_target_{timestamp}.pkl'
        joblib.dump(model_data, model_path)
        logger.info(f"\n模型已保存: {model_path}")
        logger.info(f"  大小: {model_path.stat().st_size / 1024 / 1024:.1f} MB")

        # 保存训练历史
        history = {
            'version': 'v4.4',
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'status': 'completed',
            'summary': {
                'training_samples': int(train_mask_final.sum()),
                'validation_samples': int(val_mask_final.sum()),
                'feature_count': len(self.feature_names),
                'walk_forward_summary': wf_summary,
                'bear_models': list(bear_models.keys()),
                'isotonic_targets': list(isotonic_models.keys()) if isotonic_models else [],
            },
            'target_weights': self.target_weights,
            'ensemble_weights': {k: all_results[k]['weights'] for k in all_results},
            'modules': {
                'A_monotonicity': True,
                'B_liquidity_discount': True,
                'C_bear_specialist': len(bear_models) > 0,
                'D_sharpe_blend': self.sharpe_label_blend,
                'E_executability_filter': 'scorer层实现',
                'F_regime_adaptive': 'scorer层实现',
            },
        }

        history_path = output_dir / f'training_history_{timestamp}.json'
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        latest_path = output_dir / 'training_history_latest.json'
        with open(latest_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        logger.info(f"训练历史已保存: {history_path}")
        logger.info(f"\nV4.4 训练完成! 总耗时: {duration:.0f}秒 ({duration/60:.1f}分钟)")

        return model_data, history


class V46Trainer(V44Trainer):
    """V4.6 训练器 — V4.4底座 + 5项弱指标针对性增强

    改进项:
    1A. ICIR最大化集成权重 (替代IC+单调性加权)
    1B. 小盘加权训练 (sample_weight × 1.5 for small-cap)
    1C. Combined-Score Isotonic (组合分数保序校准)
    1D. Stacking Meta-Learner (Ridge, 5模型×4目标=20维meta features)
    """

    def __init__(self, db_path: str = DB_PATH):
        super().__init__(db_path=db_path)

    def compute_sample_weights(self, df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """V4.6: 父类权重 + 小盘加权×1.5 + 低换手率降权×0.5"""
        weights = super().compute_sample_weights(df, y)

        # 1B: 小盘加权
        if 'log_market_cap' in df.columns:
            market_cap = df['log_market_cap'].values
            daily_median = np.nanmedian(market_cap)
            small_cap_mask = market_cap < daily_median
            weights[small_cap_mask] *= 1.5
            n_small = small_cap_mask.sum()
            logger.info(f"    小盘加权: {n_small:,} 样本 × 1.5 (log_market_cap < median)")

        # 低换手率降权 (训练层, 与评分层连续折扣配合)
        if 'turnover_rate' in df.columns:
            low_turnover = df['turnover_rate'].values < 1.0
            weights[low_turnover] *= 0.5
            n_low = low_turnover.sum()
            logger.info(f"    低换手降权: {n_low:,} 样本 × 0.5 (turnover_rate < 1.0%)")

        return weights

    def _optimize_icir_weights(self, all_results: dict, X_val: np.ndarray,
                                y_val_dict: dict, val_dates: np.ndarray) -> dict:
        """1A: ICIR最大化集成权重优化

        对每个target, 用scipy.optimize找到使 ICIR = mean(daily_IC)/std(daily_IC) 最大的权重组合
        """
        from scipy.optimize import minimize

        unique_dates = np.unique(val_dates)
        icir_weights = {}

        for target_key in ['3d', '5d', '10d', '15d']:
            if target_key not in all_results:
                continue

            models = all_results[target_key]['models']
            y_target = y_val_dict.get(target_key)
            if y_target is None:
                continue

            # 收集各模型在验证集上的预测
            model_preds = {}
            for name, model in models.items():
                try:
                    if name == 'xgb':
                        model_preds[name] = model.predict(xgb.DMatrix(X_val))
                    else:
                        model_preds[name] = model.predict(X_val)
                except Exception:
                    continue

            model_names = list(model_preds.keys())
            n_models = len(model_names)
            if n_models < 2:
                icir_weights[target_key] = all_results[target_key]['weights']
                continue

            pred_matrix = np.column_stack([model_preds[n] for n in model_names])

            def neg_icir(w):
                """负ICIR (用于最小化)"""
                w_norm = w / (w.sum() + 1e-10)
                ensemble = pred_matrix @ w_norm
                daily_ics = []
                for d in unique_dates:
                    mask = val_dates == d
                    if mask.sum() < 10:
                        continue
                    ic, _ = spearmanr(ensemble[mask], y_target[mask])
                    if not np.isnan(ic):
                        daily_ics.append(ic)
                if len(daily_ics) < 5:
                    return 0.0
                mean_ic = np.mean(daily_ics)
                std_ic = np.std(daily_ics)
                if std_ic < 1e-8:
                    return -mean_ic * 100
                return -(mean_ic / std_ic)

            # 优化
            w0 = np.ones(n_models) / n_models
            bounds = [(0.01, 1.0)] * n_models
            constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1.0}

            try:
                result = minimize(neg_icir, w0, method='SLSQP',
                                  bounds=bounds, constraints=constraints,
                                  options={'maxiter': 200, 'ftol': 1e-8})
                if result.success:
                    opt_w = result.x / result.x.sum()
                    weights = {name: float(opt_w[i]) for i, name in enumerate(model_names)}
                    icir_val = -result.fun
                    logger.info(f"  ICIR优化 {target_key}: ICIR={icir_val:.4f}, "
                                f"权重={', '.join(f'{k}={v:.3f}' for k, v in weights.items())}")
                else:
                    weights = all_results[target_key]['weights']
                    logger.info(f"  ICIR优化 {target_key}: 优化失败, 使用原权重")
            except Exception as e:
                weights = all_results[target_key]['weights']
                logger.info(f"  ICIR优化 {target_key}: 异常 {e}, 使用原权重")

            icir_weights[target_key] = weights

        return icir_weights

    def _fit_combined_isotonic(self, X_val: np.ndarray, y_val_dict: dict,
                                all_results: dict, icir_weights: dict) -> object:
        """1C: Combined-Score Isotonic — 对融合后的combined_pred拟合保序回归"""
        from sklearn.isotonic import IsotonicRegression

        # 计算验证集的combined_pred
        predictions = {}
        for target_key, result in all_results.items():
            target_pred = np.zeros(X_val.shape[0])
            total_weight = 0
            weights = icir_weights.get(target_key, result['weights'])

            for name, model in result['models'].items():
                try:
                    w = weights.get(name, 0.2)
                    if name == 'xgb':
                        pred = model.predict(xgb.DMatrix(X_val))
                    else:
                        pred = model.predict(X_val)
                    target_pred += w * pred
                    total_weight += w
                except Exception:
                    continue

            if total_weight > 0:
                target_pred /= total_weight
            predictions[target_key] = target_pred

        # 融合 combined_pred
        combined_pred = np.zeros(X_val.shape[0])
        for target_key, pred in predictions.items():
            w = self.target_weights.get(f'label_{target_key}', 0)
            combined_pred += w * pred

        # 融合 actual_weighted_return
        y_actual = np.zeros(X_val.shape[0])
        for target_key, y_val in y_val_dict.items():
            w = self.target_weights.get(f'label_{target_key}', 0)
            y_actual += w * y_val

        # 拟合保序回归
        valid = ~(np.isnan(combined_pred) | np.isnan(y_actual))
        if valid.sum() < 100:
            logger.info("  Combined Isotonic: 有效样本不足, 跳过")
            return None

        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(combined_pred[valid], y_actual[valid])

        # 评估
        raw_ic, _ = spearmanr(combined_pred[valid], y_actual[valid])
        cal_pred = iso.predict(combined_pred[valid])
        cal_ic, _ = spearmanr(cal_pred, y_actual[valid])
        logger.info(f"  Combined Isotonic: IC {raw_ic:.4f} → {cal_ic:.4f}")

        return iso

    def _train_meta_learner(self, X_val: np.ndarray, y_val_dict: dict,
                             all_results: dict) -> tuple:
        """1D: Stacking Meta-Learner — 5模型×4目标=20维meta features → Ridge"""
        from sklearn.linear_model import Ridge

        # 收集meta features: 每个模型对每个目标的预测
        meta_features = []
        meta_names = []

        for target_key in ['3d', '5d', '10d', '15d']:
            if target_key not in all_results:
                continue
            for name, model in all_results[target_key]['models'].items():
                try:
                    if name == 'xgb':
                        pred = model.predict(xgb.DMatrix(X_val))
                    else:
                        pred = model.predict(X_val)
                    meta_features.append(pred)
                    meta_names.append(f'{name}_{target_key}')
                except Exception:
                    continue

        if len(meta_features) < 4:
            logger.info(f"  Meta-Learner: 仅 {len(meta_features)} 个meta特征, 跳过")
            return None, None

        X_meta = np.column_stack(meta_features)

        # 目标: 加权实际收益
        y_actual = np.zeros(X_val.shape[0])
        for target_key, y_val in y_val_dict.items():
            w = self.target_weights.get(f'label_{target_key}', 0)
            y_actual += w * y_val

        # 训练 Ridge (强正则化)
        valid = ~np.isnan(y_actual)
        if valid.sum() < 100:
            logger.info("  Meta-Learner: 有效样本不足, 跳过")
            return None, None

        ridge = Ridge(alpha=10.0, fit_intercept=True)
        ridge.fit(X_meta[valid], y_actual[valid])

        # 评估
        meta_pred = ridge.predict(X_meta[valid])
        ic, _ = spearmanr(meta_pred, y_actual[valid])
        logger.info(f"  Meta-Learner: IC={ic:.4f}, {len(meta_names)} meta features, alpha=10.0")
        logger.info(f"    Top权重: {', '.join(f'{meta_names[i]}={ridge.coef_[i]:.4f}' for i in np.argsort(-np.abs(ridge.coef_))[:5])}")

        return ridge, meta_names

    def walk_forward_train(self, start_date: str = None, end_date: str = None,
                            purge_days: int = 15, min_train_days: int = 900,
                            val_days: int = 120, test_days: int = 120,
                            step_days: int = 90):
        """V4.6 Walk-Forward 训练 — V4.4底座 + ICIR权重 + Combined Isotonic + Meta-Learner"""
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("V4.6 Walk-Forward 训练 (V4.4底座 + 5项增强)")
        logger.info("=" * 60)
        logger.info(f"  参数: min_train={min_train_days}d, val={val_days}d, test={test_days}d, "
                     f"step={step_days}d, purge={purge_days}d")
        logger.info(f"  目标权重: {self.target_weights}")
        logger.info(f"  Sharpe融合: {self.sharpe_label_blend}")
        logger.info(f"  增强: ICIR权重 + 小盘加权 + Combined Isotonic + Meta-Learner")

        # 1. 一次性加载全量数据 (含V4.3扩展+Module B流动性折扣)
        df = self.load_data(start_date, end_date)
        X, y_3d, y_5d, y_10d, y_15d, df = self.prepare_features(df)

        dates = df['trade_date'].values
        unique_dates = np.sort(np.unique(dates))
        n_dates = len(unique_dates)
        logger.info(f"  总交易日: {n_dates}, 样本: {len(X):,}")

        # 2. 定义滚动窗口
        windows = []
        cursor = min_train_days
        while cursor + val_days + purge_days + test_days <= n_dates:
            train_end_idx = cursor - 1
            val_start_idx = cursor + purge_days
            val_end_idx = val_start_idx + val_days - 1
            test_start_idx = val_end_idx + 1 + purge_days
            test_end_idx = test_start_idx + test_days - 1

            if test_end_idx >= n_dates:
                break

            windows.append({
                'train_end': unique_dates[train_end_idx],
                'val_start': unique_dates[val_start_idx],
                'val_end': unique_dates[val_end_idx],
                'test_start': unique_dates[test_start_idx],
                'test_end': unique_dates[test_end_idx],
            })
            cursor += step_days

        logger.info(f"  Walk-Forward 窗口数: {len(windows)}")
        for i, w in enumerate(windows):
            logger.info(f"    窗口 {i+1}: train<='{w['train_end']}', val={w['val_start']}~{w['val_end']}, "
                         f"test={w['test_start']}~{w['test_end']}")

        # 3. 对每个窗口训练+评估
        wf_metrics = []
        import gc

        for wi, w in enumerate(windows):
            logger.info(f"\n{'='*50}")
            logger.info(f"Walk-Forward 窗口 {wi+1}/{len(windows)}")
            logger.info(f"{'='*50}")

            train_mask = dates <= w['train_end']
            val_mask = (dates >= w['val_start']) & (dates <= w['val_end'])
            test_mask = (dates >= w['test_start']) & (dates <= w['test_end'])

            X_train_w, X_val_w, X_test_w = X[train_mask].copy(), X[val_mask].copy(), X[test_mask].copy()
            y_3d_tr, y_3d_va, y_3d_te = y_3d[train_mask].copy(), y_3d[val_mask].copy(), y_3d[test_mask].copy()
            y_5d_tr, y_5d_va, y_5d_te = y_5d[train_mask].copy(), y_5d[val_mask].copy(), y_5d[test_mask].copy()
            y_10d_tr, y_10d_va, y_10d_te = y_10d[train_mask].copy(), y_10d[val_mask].copy(), y_10d[test_mask].copy()
            y_15d_tr, y_15d_va, y_15d_te = y_15d[train_mask].copy(), y_15d[val_mask].copy(), y_15d[test_mask].copy()
            test_dates_w = dates[test_mask]

            # Walk-Forward: 特征Winsorization (仅用窗口内训练集, 防止数据泄漏)
            X_train_w, wf_bounds = self.winsorize_features(X_train_w)
            for i, (lo, hi) in enumerate(wf_bounds):
                X_val_w[:, i] = np.clip(X_val_w[:, i], lo, hi)
                X_test_w[:, i] = np.clip(X_test_w[:, i], lo, hi)

            # Walk-Forward: 标签Winsorization (仅用训练集统计量, 防止数据泄漏)
            for y_tr_w, y_va_w, y_te_w in [(y_3d_tr, y_3d_va, y_3d_te),
                                             (y_5d_tr, y_5d_va, y_5d_te),
                                             (y_10d_tr, y_10d_va, y_10d_te),
                                             (y_15d_tr, y_15d_va, y_15d_te)]:
                lo = np.percentile(y_tr_w, 1)
                hi = np.percentile(y_tr_w, 99)
                y_tr_w[:] = np.clip(y_tr_w, lo, hi)
                y_va_w[:] = np.clip(y_va_w, lo, hi)
                y_te_w[:] = np.clip(y_te_w, lo, hi)

            logger.info(f"  train={X_train_w.shape[0]:,}, val={X_val_w.shape[0]:,}, test={X_test_w.shape[0]:,}")

            self.val_dates = dates[val_mask]

            # 训练 4 目标
            targets_w = [
                ('3d', y_3d_tr, y_3d_va, y_3d_te),
                ('5d', y_5d_tr, y_5d_va, y_5d_te),
                ('10d', y_10d_tr, y_10d_va, y_10d_te),
                ('15d', y_15d_tr, y_15d_va, y_15d_te),
            ]

            window_metrics = {}
            df_train_w = df[train_mask]

            for target_key, y_tr, y_va, y_te in targets_w:
                sample_w = self.compute_sample_weights(df_train_w, y_tr)
                models, pred_train, pred_val = self.train_single_target_models(
                    X_train_w, X_val_w, y_tr, y_va, f"label_{target_key}",
                    sample_weights_train=sample_w)
                weights, _ = self.calculate_ensemble_weights(pred_val, y_va)

                # Test set prediction
                pred_test = {}
                for name, model in models.items():
                    if name == 'xgb':
                        pred_test[name] = model.predict(xgb.DMatrix(X_test_w))
                    else:
                        pred_test[name] = model.predict(X_test_w)

                ensemble_pred = self.ensemble_predict(pred_test, weights)
                ic, icir = self._calculate_daily_ic(ensemble_pred, y_te, test_dates_w)
                window_metrics[target_key] = {'ic': ic, 'icir': icir}
                logger.info(f"  {target_key}: IC={ic:.4f}, ICIR={icir:.4f}")

                del models, pred_train, pred_val, pred_test
                gc.collect()

            wf_metrics.append(window_metrics)

        # 4. Walk-Forward 汇总
        logger.info("\n" + "=" * 60)
        logger.info("Walk-Forward 汇总")
        logger.info("=" * 60)

        wf_summary = {}
        for target_key in ['3d', '5d', '10d', '15d']:
            ics = [m[target_key]['ic'] for m in wf_metrics if target_key in m]
            icirs = [m[target_key]['icir'] for m in wf_metrics if target_key in m]
            summary = {
                'mean_ic': float(np.mean(ics)),
                'std_ic': float(np.std(ics)),
                'mean_icir': float(np.mean(icirs)),
                'std_icir': float(np.std(icirs)),
                'n_windows': len(ics),
            }
            wf_summary[target_key] = summary
            logger.info(f"  {target_key}: IC={summary['mean_ic']:.4f}±{summary['std_ic']:.4f}, "
                         f"ICIR={summary['mean_icir']:.4f}±{summary['std_icir']:.4f}")

        # 5. 训练最终生产模型 (85% train + 15% val)
        logger.info("\n" + "=" * 60)
        logger.info("训练最终V4.6生产模型 (全量数据)")
        logger.info("=" * 60)

        split_idx = int(n_dates * 0.85)
        split_date = unique_dates[split_idx]
        train_mask_final = dates <= split_date
        val_mask_final = dates > split_date

        X_train_f, X_val_f = X[train_mask_final], X[val_mask_final]
        self.val_dates = dates[val_mask_final]

        df_train_f = df[train_mask_final]
        all_results = {}

        y_val_dict = {}
        targets_final = [
            ('3d', y_3d[train_mask_final].copy(), y_3d[val_mask_final].copy()),
            ('5d', y_5d[train_mask_final].copy(), y_5d[val_mask_final].copy()),
            ('10d', y_10d[train_mask_final].copy(), y_10d[val_mask_final].copy()),
            ('15d', y_15d[train_mask_final].copy(), y_15d[val_mask_final].copy()),
        ]

        # V4.6生产模型: 标签Winsorization
        for target_key, y_tr, y_va in targets_final:
            lo = np.percentile(y_tr, 1)
            hi = np.percentile(y_tr, 99)
            y_tr[:] = np.clip(y_tr, lo, hi)
            y_va[:] = np.clip(y_va, lo, hi)

        for target_key, y_tr, y_va in targets_final:
            sample_w = self.compute_sample_weights(df_train_f, y_tr)
            models, pred_train, pred_val = self.train_single_target_models(
                X_train_f, X_val_f, y_tr, y_va, f"label_{target_key}",
                sample_weights_train=sample_w)
            weights, rmses = self.calculate_ensemble_weights(pred_val, y_va)
            all_results[target_key] = {'models': models, 'weights': weights, 'rmses': rmses}
            y_val_dict[target_key] = y_va

        # 6. Module C: 训练熊市专家模型
        logger.info("\n" + "=" * 60)
        logger.info("Module C: 训练熊市专家模型")
        logger.info("=" * 60)
        bear_models = {}
        for target_key in ['10d', '15d']:
            y_tr_target = y_10d[train_mask_final] if target_key == '10d' else y_15d[train_mask_final]
            bear_model = self._train_bear_specialist(X_train_f, y_tr_target, df_train_f, target_key)
            if bear_model is not None:
                bear_models[target_key] = bear_model

        # 7. Module A: 保序回归校准 (per-target)
        logger.info("\n" + "=" * 60)
        logger.info("Module A: Per-Target 保序回归校准")
        logger.info("=" * 60)
        isotonic_models = self._fit_isotonic_calibration(X_val_f, y_val_dict, all_results)

        # 8. V4.6增强: ICIR优化集成权重
        logger.info("\n" + "=" * 60)
        logger.info("V4.6 1A: ICIR最大化集成权重")
        logger.info("=" * 60)
        icir_weights = self._optimize_icir_weights(all_results, X_val_f, y_val_dict, self.val_dates)
        # 更新all_results中的权重
        for target_key, w in icir_weights.items():
            if target_key in all_results:
                all_results[target_key]['weights'] = w

        # 9. V4.6增强: Combined Isotonic
        logger.info("\n" + "=" * 60)
        logger.info("V4.6 1C: Combined-Score Isotonic")
        logger.info("=" * 60)
        combined_isotonic = self._fit_combined_isotonic(X_val_f, y_val_dict, all_results, icir_weights)

        # 10. V4.6增强: Stacking Meta-Learner
        logger.info("\n" + "=" * 60)
        logger.info("V4.6 1D: Stacking Meta-Learner")
        logger.info("=" * 60)
        meta_learner, meta_feature_names = self._train_meta_learner(X_val_f, y_val_dict, all_results)

        # 11. 特征重要性分析
        self._log_feature_importance(all_results)

        # 12. 计算全局评分分位数
        global_quantiles = self._compute_global_quantiles(X, all_results, self.target_weights)

        # 13. 保存模型
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        output_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v46'
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        model_data = {
            'version': 'v4.6',
            'models': all_results,
            'feature_names': self.feature_names,
            'target_weights': self.target_weights,
            'market_features': list(self.market_calculator.market_features.columns[1:]),
            'winsorize_bounds': getattr(self, 'winsorize_bounds', None),
            'global_quantiles': global_quantiles,
            # 模型类型标识
            'cascade': False,
            'rank_normalized': False,
            'robust_zscore': True,
            'industry_excess_labels': True,
            'dual_stream': False,
            'cross_sectional_neutralization': False,
            'macro_feature_cols': self.macro_feature_cols,
            'stock_feature_cols': self.stock_feature_cols,
            'extra_features_from_daily_basic': ['pe_ttm', 'pb', 'ps_ttm', 'turnover_rate', 'log_market_cap'],
            'extra_features_from_tech_indicators': self.extra_tech_feature_names,
            'targets': ['3d', '5d', '10d', '15d'],
            'ensemble_type': 'icir_optimized',
            'sample_weighting': True,
            'walk_forward_metrics': wf_summary,
            'walk_forward_windows': len(windows),
            'regularization': {
                'num_leaves': 20, 'min_data_in_leaf': 500,
                'reg_alpha': 1.0, 'reg_lambda': 5.0,
                'path_smooth': 10.0, 'learning_rate': 0.02,
            },
            # V4.4 组件 (继承)
            'bear_models': bear_models,
            'isotonic_calibration': isotonic_models,
            'sharpe_label_blend': self.sharpe_label_blend,
            'liquidity_discount': True,
            'bear_sample_weighting': True,
            'min_train_days': min_train_days,
            'step_days': step_days,
            # V4.6 新增组件
            'icir_optimized_weights': icir_weights,
            'combined_isotonic': combined_isotonic,
            'meta_learner': meta_learner,
            'meta_feature_names': meta_feature_names,
            'small_cap_weighting': True,
        }

        model_path = output_dir / f'v46_multi_target_{timestamp}.pkl'
        joblib.dump(model_data, model_path)
        logger.info(f"\n模型已保存: {model_path}")
        logger.info(f"  大小: {model_path.stat().st_size / 1024 / 1024:.1f} MB")

        # 保存训练历史
        history = {
            'version': 'v4.6',
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'status': 'completed',
            'summary': {
                'training_samples': int(train_mask_final.sum()),
                'validation_samples': int(val_mask_final.sum()),
                'feature_count': len(self.feature_names),
                'walk_forward_summary': wf_summary,
                'bear_models': list(bear_models.keys()),
                'isotonic_targets': list(isotonic_models.keys()) if isotonic_models else [],
                'meta_learner': meta_learner is not None,
                'combined_isotonic': combined_isotonic is not None,
                'icir_optimized': len(icir_weights) > 0,
            },
            'target_weights': self.target_weights,
            'ensemble_weights': {k: all_results[k]['weights'] for k in all_results},
            'modules': {
                'A_monotonicity': True,
                'B_liquidity_discount': True,
                'C_bear_specialist': len(bear_models) > 0,
                'D_sharpe_blend': self.sharpe_label_blend,
                'E_executability_filter': 'scorer层实现',
                'F_regime_adaptive': 'scorer层实现',
                'V46_icir_weights': True,
                'V46_small_cap_weighting': True,
                'V46_combined_isotonic': combined_isotonic is not None,
                'V46_meta_learner': meta_learner is not None,
            },
        }

        history_path = output_dir / f'training_history_{timestamp}.json'
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        latest_path = output_dir / 'training_history_latest.json'
        with open(latest_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        logger.info(f"训练历史已保存: {history_path}")
        logger.info(f"\nV4.6 训练完成! 总耗时: {duration:.0f}秒 ({duration/60:.1f}分钟)")

        return model_data, history


def main():
    parser = argparse.ArgumentParser(description='V3.95/V4.3/V4.4/V4.6 多目标训练')
    parser.add_argument('--start-date', type=str, default='2020-01-01', help='训练开始日期')
    parser.add_argument('--end-date', type=str, default=None, help='训练结束日期')
    parser.add_argument('--purge-days', type=int, default=10, help='Purge gap天数 (应>=最大标签前瞻天数, label_10d需要10天)')
    parser.add_argument('--sharpe-blend', type=float, default=0.3, help='Sharpe标签融合比例 (0=纯收益, 0.3=推荐, 1=纯Sharpe)')
    parser.add_argument('--v43', action='store_true', help='V4.3: 扩展特征+强正则+Walk-Forward')
    parser.add_argument('--v44', action='store_true', help='V4.4: V4.3信号+6增强模块 (单调性校准/流动性/熊市专家/Sharpe标签)')
    parser.add_argument('--v46', action='store_true', help='V4.6: V4.4+ICIR权重+小盘加权+Combined Isotonic+Meta-Learner')
    args = parser.parse_args()

    if args.v46:
        trainer = V46Trainer()
        trainer.walk_forward_train(
            start_date=args.start_date, end_date=args.end_date,
            purge_days=max(args.purge_days, 15))
    elif args.v44:
        trainer = V44Trainer()
        trainer.walk_forward_train(
            start_date=args.start_date, end_date=args.end_date,
            purge_days=max(args.purge_days, 15))
    elif args.v43:
        trainer = V43Trainer()
        trainer.walk_forward_train(
            start_date=args.start_date, end_date=args.end_date,
            purge_days=max(args.purge_days, 15))  # V4.3 需要 15d purge gap
    else:
        trainer = V395MultiTargetTrainer()
        trainer.sharpe_label_blend = args.sharpe_blend
        trainer.train(start_date=args.start_date, end_date=args.end_date, purge_days=args.purge_days)


if __name__ == '__main__':
    main()
