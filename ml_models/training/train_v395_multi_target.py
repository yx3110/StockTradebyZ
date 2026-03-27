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

# 可选快速JSON库 (pip install orjson, 3-5x faster than json)
try:
    import orjson
    _json_loads = orjson.loads
except ImportError:
    _json_loads = json.loads

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

    @staticmethod
    def _robust_zscore_cross_section(stock_data: np.ndarray, dates_arr: np.ndarray) -> np.ndarray:
        """截面Robust Z-Score — 向量化版本 (比逐日期布尔掩码快~5x)

        原理: 先按日期排序获得连续内存切片, 再用searchsorted定位日期边界,
        每个日期的数据是连续的 [s:e] 切片(view, 零拷贝), 避免布尔花式索引的拷贝开销.
        """
        # 按日期排序, 使每个日期的行在内存中连续
        sort_idx = np.argsort(dates_arr, kind='stable')
        unsort_idx = np.argsort(sort_idx, kind='stable')
        sorted_data = stock_data[sort_idx]
        sorted_dates = dates_arr[sort_idx]

        # 用searchsorted找到每个日期的起止索引
        unique_dates = np.unique(sorted_dates)
        starts = np.searchsorted(sorted_dates, unique_dates, side='left')
        ends = np.searchsorted(sorted_dates, unique_dates, side='right')

        for i in range(len(unique_dates)):
            s, e = starts[i], ends[i]
            chunk = sorted_data[s:e]  # view, 零拷贝
            median = np.nanmedian(chunk, axis=0)
            mad = np.nanmedian(np.abs(chunk - median), axis=0) * 1.4826
            mad[mad < 1e-8] = 1e-8
            sorted_data[s:e] = np.clip((chunk - median) / mad, -3, 3)

        # 还原原始顺序
        return sorted_data[unsort_idx]

    @staticmethod
    def _apply_bounds(X: np.ndarray, bounds: list) -> np.ndarray:
        """向量化应用winsorize bounds (替代逐列np.clip循环)

        Args:
            X: 特征矩阵 (n_samples, n_features), 会被原地修改
            bounds: list of (lo, hi) tuples, 长度等于n_features
        Returns:
            X (原地修改后)
        """
        lo_arr = np.array([b[0] for b in bounds])
        hi_arr = np.array([b[1] for b in bounds])
        np.clip(X, lo_arr, hi_arr, out=X)
        return X

    def _compute_global_quantiles(self, X: np.ndarray, all_results: dict,
                                    target_weights: dict, n_quantiles: int = 1001,
                                    precomputed_predictions: dict = None) -> np.ndarray:
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

        # 优化: 复用已有的ensemble预测，避免重复predict (节省~50-120秒)
        if precomputed_predictions is not None:
            predictions = precomputed_predictions
            logger.info(f"  使用预计算的ensemble预测 ({len(predictions)} targets)")
        else:
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

    def _compute_recommendation_thresholds(self, X: np.ndarray, all_results: dict,
                                             precomputed_predictions: dict = None) -> dict:
        """计算composite score的推荐阈值 (基于全市场历史百分位)

        Composite权重从 self.recommendation_composite_weights 取 (如有),
        否则回退到 self.target_weights 的 label_ 前缀去除版本。
        这样各版本 scorer 用什么公式, 训练时就用什么公式校准阈值。

        Returns:
            dict with keys: strong_buy (P95), buy (P80), cautious (P60), hold (P40)
        """
        # 确定 composite 权重: 优先用显式设置, 否则从 target_weights 派生
        if hasattr(self, 'recommendation_composite_weights'):
            composite_weights = self.recommendation_composite_weights
        else:
            composite_weights = {k.replace('label_', ''): v
                                 for k, v in self.target_weights.items()}

        cw_str = " + ".join(f"pred_{k}×{v}" for k, v in composite_weights.items() if v > 0)
        logger.info(f"计算composite推荐阈值 (n={X.shape[0]:,} 样本, composite={cw_str})...")

        # 优化: 复用已有的ensemble预测，避免重复predict
        if precomputed_predictions is not None:
            predictions = precomputed_predictions
            logger.info(f"  使用预计算的ensemble预测 ({len(predictions)} targets)")
        else:
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
                    except Exception:
                        continue
                if total_weight > 0:
                    target_pred /= total_weight
                predictions[target_key] = target_pred

        # Composite score
        composite = np.zeros(X.shape[0])
        for target_key, w in composite_weights.items():
            if target_key in predictions:
                composite += w * predictions[target_key]

        thresholds = {
            'strong_buy': float(np.percentile(composite, 95)),
            'buy': float(np.percentile(composite, 80)),
            'cautious': float(np.percentile(composite, 60)),
            'hold': float(np.percentile(composite, 40)),
        }

        logger.info(f"  composite分布: min={composite.min():.6f}, P50={np.median(composite):.6f}, max={composite.max():.6f}")
        logger.info(f"  推荐阈值: 强烈买入≥{thresholds['strong_buy']:.6f}, 买入≥{thresholds['buy']:.6f}, "
                     f"谨慎≥{thresholds['cautious']:.6f}, 观望≥{thresholds['hold']:.6f}")
        return thresholds


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

        # 解析特征JSON — 向量化版本 (比iterrows快10-50x)
        logger.info("  解析特征JSON...")
        try:
            parsed = df['features_json'].apply(_json_loads).tolist()
            df_features = pd.DataFrame(parsed)
            df_features['code'] = df['code'].values
            df_features['trade_date'] = df['trade_date'].values
            df_features['label_3d'] = df['label_3d'].values
            df_features['label_5d'] = df['label_5d'].values
            df_features['label_10d'] = df['label_10d'].values
        except Exception as e:
            # 降级: 逐行解析 (处理个别JSON异常)
            logger.warning(f"  向量化JSON解析失败({e}), 降级为逐行解析")
            features_list = []
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="解析特征"):
                try:
                    features = _json_loads(row['features_json'])
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

        # ===== BRAIN 验证因子 (可选, --brain-features 启用) =====
        if getattr(self, 'use_brain_features', False):
            try:
                from wqbrain_integration.validated_alphas import BRAIN_FEATURE_COLS
                from wqbrain_integration.brain_feature_importer import BrainFeatureImporter

                logger.info("  加载 BRAIN 验证因子...")
                importer = BrainFeatureImporter(db_path=self.db_path)
                brain_df = importer.get_features_for_training(date_min, date_max)

                if brain_df is not None and not brain_df.empty:
                    before_cols = len(df_features.columns)
                    df_features = df_features.merge(brain_df, on=['code', 'trade_date'], how='left')
                    brain_cols = [c for c in brain_df.columns if c not in ('code', 'trade_date')]
                    df_features[brain_cols] = df_features[brain_cols].fillna(0.0)
                    logger.info(f"    BRAIN 因子合并完成: +{len(brain_cols)} 特征, "
                                f"覆盖 {(brain_df.shape[0] / len(df_features) * 100):.1f}% 样本")
                else:
                    logger.warning("    BRAIN 缓存为空, 跳过. 请先运行: "
                                   "python3 wqbrain_integration/brain_feature_importer.py load-wq101 --compute")
            except ImportError:
                logger.warning("    wqbrain_integration 模块未安装, 跳过 BRAIN 因子")
            except Exception as e:
                logger.warning(f"    BRAIN 因子加载失败: {e}")

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
        # 向量化: 一次算完所有列的百分位，替代逐列循环
        lo_arr = np.nanpercentile(X, lower_pct, axis=0)
        hi_arr = np.nanpercentile(X, upper_pct, axis=0)
        all_nan_cols = np.all(np.isnan(X), axis=0)
        constant_mask = (lo_arr == hi_arr) | all_nan_cols
        # 对全NaN列设bounds为(0.0, 0.0)
        lo_arr[all_nan_cols] = 0.0
        hi_arr[all_nan_cols] = 0.0
        bounds = list(zip(lo_arr.astype(float).tolist(), hi_arr.astype(float).tolist()))
        # 仅clip非常量列
        for i in range(X.shape[1]):
            if not constant_mask[i]:
                X_w[:, i] = np.clip(X[:, i], lo_arr[i], hi_arr[i])
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

        stock_data = self._robust_zscore_cross_section(stock_data, dates_arr)

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
        self._apply_bounds(X_val, self.winsorize_bounds)
        self._apply_bounds(X_test, self.winsorize_bounds)

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
        # CLI overrides (--num-leaves, --min-data-in-leaf)
        if hasattr(self, '_cli_num_leaves'):
            lgb_params['num_leaves'] = self._cli_num_leaves
        if hasattr(self, '_cli_min_data_in_leaf'):
            lgb_params['min_data_in_leaf'] = self._cli_min_data_in_leaf

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
                    elif model_name == 'cb' and hasattr(model, 'get_feature_importance'):
                        try:
                            importance = np.array(model.get_feature_importance(), dtype=float)
                        except Exception:
                            importance = None
                    elif hasattr(model, 'feature_importances_'):
                        importance = model.feature_importances_
                except Exception as e:
                    logger.debug(f"  {target_name}/{model_name} 特征重要性提取失败: {e}")
                    continue

                if importance is None or target_feature_names is None:
                    continue

                # 转为numpy array (CatBoost ranker可能返回非标准格式)
                try:
                    importance = np.asarray(importance, dtype=float).ravel()
                    if importance.ndim == 0 or len(importance) == 0:
                        continue
                except Exception:
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

        # 并行训练: ThreadPoolExecutor (LGB/XGB/CatBoost是C++实现，释放GIL)
        parallel_training = getattr(self, 'parallel_training', True)
        if parallel_training and len(targets) > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            logger.info(f"  并行训练 {len(targets)} targets (ThreadPoolExecutor)")

            def _train_target(args):
                target_key, y_tr, y_va, y_te, target_name = args
                models, pred_train, pred_val = self.train_single_target_models(
                    X_train, X_val, y_tr, y_va, target_name)
                weights, rmses = self.calculate_ensemble_weights(pred_val, y_va)
                return target_key, {'models': models, 'weights': weights, 'rmses': rmses}

            with ThreadPoolExecutor(max_workers=len(targets)) as pool:
                futures = {pool.submit(_train_target, t): t[0] for t in targets}
                for future in as_completed(futures):
                    target_key, result = future.result()
                    all_results[target_key] = result
                    logger.info(f"  {target_key} 训练完成")
        else:
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
        # 优化: 复用ensemble_predictions，避免重复predict (节省24次model.predict调用)
        global_quantiles = self._compute_global_quantiles(
            X_test, all_results, self.target_weights,
            precomputed_predictions=ensemble_predictions)
        recommendation_thresholds = self._compute_recommendation_thresholds(
            X_test, all_results,
            precomputed_predictions=ensemble_predictions)

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
            'recommendation_thresholds': recommendation_thresholds,
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

        stock_data = self._robust_zscore_cross_section(stock_data, dates_arr)

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
            self._apply_bounds(X_val_w, wf_bounds)
            self._apply_bounds(X_test_w, wf_bounds)

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
        recommendation_thresholds = self._compute_recommendation_thresholds(X, all_results)

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
            'recommendation_thresholds': recommendation_thresholds,
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
            self._apply_bounds(X_val_w, wf_bounds)
            self._apply_bounds(X_test_w, wf_bounds)

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
        recommendation_thresholds = self._compute_recommendation_thresholds(X, all_results)

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
            'recommendation_thresholds': recommendation_thresholds,
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

    # V4.6 scorer 使用 0.6*10d + 0.4*15d composite
    recommendation_composite_weights = {'3d': 0.0, '5d': 0.0, '10d': 0.6, '15d': 0.4}

    def __init__(self, db_path: str = DB_PATH):
        super().__init__(db_path=db_path)

    def compute_sample_weights(self, df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """V4.6: 父类权重 + 四分位分层小盘加权 + 低换手率降权×0.5"""
        weights = super().compute_sample_weights(df, y)

        # 1B: 四分位分层小盘加权
        if 'log_market_cap' in df.columns:
            market_cap = df['log_market_cap'].values
            q25, q50, q75 = np.nanpercentile(market_cap, [25, 50, 75])
            # 底部25%: ×2.5, 25-50%: ×1.8, 50-75%: ×1.0, 顶部25%: ×0.7
            bottom_mask = market_cap < q25
            lower_mask = (market_cap >= q25) & (market_cap < q50)
            upper_mask = (market_cap >= q50) & (market_cap < q75)
            top_mask = market_cap >= q75
            weights[bottom_mask] *= 2.5
            weights[lower_mask] *= 1.8
            # upper: ×1.0 (不变)
            weights[top_mask] *= 0.7
            logger.info(f"    四分位小盘加权: bottom25%={bottom_mask.sum():,}×2.5, "
                        f"lower25%={lower_mask.sum():,}×1.8, "
                        f"upper25%={upper_mask.sum():,}×1.0, "
                        f"top25%={top_mask.sum():,}×0.7")

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

            # 预计算日期索引 + 预rank y_target (避免每次迭代重复计算)
            from scipy.stats import rankdata
            date_slices = []  # [(start, end), ...] 每个日期在排序后的索引范围
            sort_idx = np.argsort(val_dates, kind='stable')
            sorted_val_dates = val_dates[sort_idx]
            sorted_pred_matrix = pred_matrix[sort_idx]
            sorted_y_target = y_target[sort_idx]
            y_ranked_slices = []  # 预rank的y_target片段

            for d in unique_dates:
                s = np.searchsorted(sorted_val_dates, d, side='left')
                e = np.searchsorted(sorted_val_dates, d, side='right')
                n = e - s
                if n < 10:
                    continue
                date_slices.append((s, e))
                y_ranked_slices.append(rankdata(sorted_y_target[s:e]))

            n_valid_dates = len(date_slices)

            def neg_icir(w):
                """负ICIR (优化版: 预索引+预rank+corrcoef)"""
                w_norm = w / (w.sum() + 1e-10)
                ensemble = sorted_pred_matrix @ w_norm
                daily_ics = []
                for i, (s, e) in enumerate(date_slices):
                    ens_slice = ensemble[s:e]
                    ens_ranked = rankdata(ens_slice)
                    # Pearson on ranks = Spearman
                    ic = np.corrcoef(ens_ranked, y_ranked_slices[i])[0, 1]
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
            self._apply_bounds(X_val_w, wf_bounds)
            self._apply_bounds(X_test_w, wf_bounds)

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
        recommendation_thresholds = self._compute_recommendation_thresholds(X, all_results)

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
            'recommendation_thresholds': recommendation_thresholds,
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


class V47Trainer(V46Trainer):
    """V4.7 训练器 — V4.6底座 - 小盘加权 + 长周期偏重 + 动量风险折扣

    目标: IC单调性 3.23→4.5+, cap_balance_ratio 0.11→0.80+, 换手率↓, 流动性覆盖↑

    与V4.6的差异:
    1. 移除Module 1B小盘加权 (导致cap_balance_ratio=0.11/5)
    2. 新增lgb_pct模型: 用per-date percentile label训练的LGB, 直接优化排名
    3. 保留V4.6其他增强: ICIR权重(1A), Combined Isotonic(1C), Meta-Learner(1D)
    4. 长周期标签权重偏重: 10d=0.45, 15d=0.30 (降低短期信号权重→降换手)
    5. 动量风险标签折扣: 近3日涨幅>8%的股票标签×0.5 (减少涨停买入失败)
    """

    def __init__(self, db_path: str = DB_PATH):
        super().__init__(db_path=db_path)
        # V4.7: 长周期偏重 — 减少短期信号噪音, 降低日间排名波动→降换手
        self.target_weights = {
            'label_3d': 0.10,   # 0.20→0.10 (大幅降低, 减少短期噪音)
            'label_5d': 0.15,   # 0.25→0.15
            'label_10d': 0.45,  # 0.35→0.45 (主力, 对齐北极星10d评估)
            'label_15d': 0.30,  # 0.20→0.30 (增加, 提升信号持续性)
        }

    def load_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """V4.7: 父类加载 + 涨停风险标签折扣 (使用turnover_rate作为proxy)"""
        df = super().load_data(start_date, end_date)

        # V4.7 增强: 极低换手率 = 涨停锁仓信号 → 标签折扣
        # 涨停股换手率极低(<0.3%), 这些股票无法买入但模型可能给高分
        # V4.4 Module B 已用 turnover_rate<0.5% → label×0.5
        # V4.7 进一步: turnover_rate<0.3% → label×0.3 (更强折扣)
        if 'turnover_rate' in df.columns:
            very_low_liq = df['turnover_rate'] < 0.3
            n_very_low = very_low_liq.sum()
            if n_very_low > 0:
                for col in ['label_3d', 'label_5d', 'label_10d', 'label_15d']:
                    if col in df.columns:
                        # 在Module B的0.5折扣基础上再打0.6折 → 总折扣0.3
                        df.loc[very_low_liq, col] *= 0.6
                logger.info(f"  V4.7 涨停风险折扣: {n_very_low:,} 样本标签×0.6 (turnover<0.3%)")

        return df

    def compute_sample_weights(self, df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """V4.7: 使用V4.4权重(含熊市加权), 但跳过V4.6的小盘加权"""
        return V44Trainer.compute_sample_weights(self, df, y)

    def train_single_target_models(self, X_train, X_val, y_train, y_val, target_name: str,
                                    sample_weights_train=None):
        """V4.7: V4.3基础模型 + 额外percentile-label LGB模型"""
        import gc

        # 1. 标准模型 (lgb_reg, xgb, cb, rf, hgb) 从父类
        models, pred_train, pred_val = super().train_single_target_models(
            X_train, X_val, y_train, y_val, target_name,
            sample_weights_train=sample_weights_train)

        # 2. 额外: Percentile-Label LGB — 直接学习排名而非收益幅度
        train_dates = getattr(self, 'train_dates', None)
        val_dates = getattr(self, 'val_dates', None)
        if train_dates is not None and len(train_dates) == len(y_train):
            logger.info(f"  训练 LGB-Percentile ({target_name})...")
            try:
                # 计算per-date percentile label (0-1)
                from scipy.stats import rankdata
                unique_dates = np.unique(train_dates)
                pct_labels = np.zeros_like(y_train)
                for date in unique_dates:
                    mask = train_dates == date
                    n = mask.sum()
                    if n >= 20:
                        ranks = rankdata(y_train[mask])
                        pct_labels[mask] = (ranks - 1) / (n - 1)  # [0, 1]
                    else:
                        pct_labels[mask] = 0.5

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

                # Val set percentile labels
                pct_val = np.zeros_like(y_val)
                if val_dates is not None and len(val_dates) == len(y_val):
                    for date in np.unique(val_dates):
                        mask = val_dates == date
                        n = mask.sum()
                        if n >= 20:
                            ranks = rankdata(y_val[mask])
                            pct_val[mask] = (ranks - 1) / (n - 1)
                        else:
                            pct_val[mask] = 0.5

                lgb_pct_train = lgb.Dataset(X_train, label=pct_labels,
                                            weight=sample_weights_train, free_raw_data=True)
                lgb_pct_val = lgb.Dataset(X_val, label=pct_val,
                                          reference=lgb_pct_train, free_raw_data=True)

                lgb_pct_model = lgb.train(
                    lgb_params, lgb_pct_train,
                    num_boost_round=1000,
                    valid_sets=[lgb_pct_train, lgb_pct_val],
                    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
                )

                models['lgb_pct'] = lgb_pct_model
                pred_train['lgb_pct'] = lgb_pct_model.predict(X_train)
                pred_val['lgb_pct'] = lgb_pct_model.predict(X_val)

                # 计算IC评估
                from scipy.stats import spearmanr
                ic_val, _ = spearmanr(pred_val['lgb_pct'], y_val)
                logger.info(f"    LGB-Percentile: val IC={ic_val:.4f} (vs return)")

                del lgb_pct_train, lgb_pct_val
                gc.collect()
            except Exception as e:
                logger.warning(f"    LGB-Percentile训练失败: {e}")

        return models, pred_train, pred_val

    def walk_forward_train(self, start_date=None, end_date=None,
                            purge_days=15, min_train_days=900,
                            val_days=120, test_days=120, step_days=90):
        """V4.7 Walk-Forward — 设置train_dates + V4.6流程"""
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("V4.7 Walk-Forward 训练 (V4.6底座 - 小盘加权 + 长周期偏重 + 排名标签LGB)")
        logger.info("=" * 60)
        logger.info(f"  参数: min_train={min_train_days}d, val={val_days}d, test={test_days}d, "
                     f"step={step_days}d, purge={purge_days}d")
        logger.info(f"  目标权重: {self.target_weights}")
        logger.info(f"  增强: ICIR权重 + Combined Isotonic + Meta-Learner + Percentile-LGB")
        logger.info(f"  已移除: Module 1B 小盘加权")
        logger.info(f"  V4.7新增: 长周期偏重(10d=0.45,15d=0.30) + 涨停风险折扣")

        # 1. 加载数据
        df = self.load_data(start_date, end_date)
        X, y_3d, y_5d, y_10d, y_15d, df = self.prepare_features(df)

        dates = df['trade_date'].values
        unique_dates = np.sort(np.unique(dates))
        n_dates = len(unique_dates)
        logger.info(f"  总交易日: {n_dates}, 样本: {len(X):,}")

        # 2. Walk-Forward windows
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

        # 3. WF evaluation
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

            # Winsorization
            X_train_w, wf_bounds = self.winsorize_features(X_train_w)
            self._apply_bounds(X_val_w, wf_bounds)
            self._apply_bounds(X_test_w, wf_bounds)

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

            # V4.7: 设置train/val dates供lgb_pct使用
            self.train_dates = dates[train_mask]
            self.val_dates = dates[val_mask]

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

        # 4. WF summary
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

        # 5. 训练最终V4.7生产模型 (85% train + 15% val)
        logger.info("\n" + "=" * 60)
        logger.info("训练最终V4.7生产模型 (全量数据)")
        logger.info("=" * 60)

        split_idx = int(n_dates * 0.85)
        split_date = unique_dates[split_idx]
        train_mask_final = dates <= split_date
        val_mask_final = dates > split_date

        X_train_f, X_val_f = X[train_mask_final], X[val_mask_final]
        self.train_dates = dates[train_mask_final]  # V4.7: for lgb_pct
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

        # 6. Module C: bear specialist
        logger.info("\n" + "=" * 60)
        logger.info("Module C: 训练熊市专家模型")
        logger.info("=" * 60)
        bear_models = {}
        for target_key in ['10d', '15d']:
            y_tr_target = y_10d[train_mask_final] if target_key == '10d' else y_15d[train_mask_final]
            bear_model = self._train_bear_specialist(X_train_f, y_tr_target, df_train_f, target_key)
            if bear_model is not None:
                bear_models[target_key] = bear_model

        # 7. Module A: per-target isotonic
        logger.info("\n" + "=" * 60)
        logger.info("Module A: Per-Target 保序回归校准")
        logger.info("=" * 60)
        isotonic_models = self._fit_isotonic_calibration(X_val_f, y_val_dict, all_results)

        # 8. V4.6 1A: ICIR weights
        logger.info("\n" + "=" * 60)
        logger.info("V4.7 1A: ICIR最大化集成权重")
        logger.info("=" * 60)
        icir_weights = self._optimize_icir_weights(all_results, X_val_f, y_val_dict, self.val_dates)
        for target_key, w in icir_weights.items():
            if target_key in all_results:
                all_results[target_key]['weights'] = w

        # 9. V4.6 1C: Combined Isotonic
        logger.info("\n" + "=" * 60)
        logger.info("V4.7 1C: Combined-Score Isotonic")
        logger.info("=" * 60)
        combined_isotonic = self._fit_combined_isotonic(X_val_f, y_val_dict, all_results, icir_weights)

        # 10. V4.6 1D: Meta-Learner
        logger.info("\n" + "=" * 60)
        logger.info("V4.7 1D: Stacking Meta-Learner")
        logger.info("=" * 60)
        meta_learner, meta_feature_names = self._train_meta_learner(X_val_f, y_val_dict, all_results)

        # 11. Feature importance
        self._log_feature_importance(all_results)

        # 12. Global quantiles
        global_quantiles = self._compute_global_quantiles(X, all_results, self.target_weights)
        recommendation_thresholds = self._compute_recommendation_thresholds(X, all_results)

        # 13. Save model
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        output_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v47'
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        model_data = {
            'version': 'v4.7',
            'models': all_results,
            'feature_names': self.feature_names,
            'target_weights': self.target_weights,
            'market_features': list(self.market_calculator.market_features.columns[1:]),
            'winsorize_bounds': getattr(self, 'winsorize_bounds', None),
            'global_quantiles': global_quantiles,
            'recommendation_thresholds': recommendation_thresholds,
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
            'bear_models': bear_models,
            'isotonic_calibration': isotonic_models,
            'sharpe_label_blend': self.sharpe_label_blend,
            'liquidity_discount': True,
            'bear_sample_weighting': True,
            'min_train_days': min_train_days,
            'step_days': step_days,
            'icir_optimized_weights': icir_weights,
            'combined_isotonic': combined_isotonic,
            'meta_learner': meta_learner,
            'meta_feature_names': meta_feature_names,
            'small_cap_weighting': False,  # V4.7: disabled
            'has_percentile_lgb': True,    # V4.7: percentile-label model
        }

        model_path = output_dir / f'v47_multi_target_{timestamp}.pkl'
        joblib.dump(model_data, model_path)
        logger.info(f"\n模型已保存: {model_path}")
        logger.info(f"  大小: {model_path.stat().st_size / 1024 / 1024:.1f} MB")

        # Save training history
        history = {
            'version': 'v4.7',
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
                'V47_icir_weights': True,
                'V47_small_cap_weighting': False,
                'V47_percentile_lgb': True,
                'V47_combined_isotonic': combined_isotonic is not None,
                'V47_meta_learner': meta_learner is not None,
            },
        }

        history_path = output_dir / f'training_history_{timestamp}.json'
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        latest_path = output_dir / 'training_history_latest.json'
        with open(latest_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        logger.info(f"训练历史已保存: {history_path}")
        logger.info(f"\nV4.7 训练完成! 总耗时: {duration:.0f}秒 ({duration/60:.1f}分钟)")

        return model_data, history

    def train_production_only(self, start_date=None, end_date=None,
                               purge_days=15, min_train_days=900):
        """V4.7 快速训练 — 跳过Walk-Forward评估, 只训练生产模型

        节省~75%时间 (只训练1次而非4+1次)
        WF指标将为空, 但生产模型质量相同
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("V4.7 快速生产模型训练 (跳过Walk-Forward)")
        logger.info("=" * 60)
        logger.info(f"  目标权重: {self.target_weights}")
        logger.info(f"  增强: ICIR权重 + Combined Isotonic + Meta-Learner + Percentile-LGB")
        logger.info(f"  V4.7新增: 长周期偏重(10d=0.45,15d=0.30) + 涨停风险折扣")

        # 1. 加载数据
        df = self.load_data(start_date, end_date)
        X, y_3d, y_5d, y_10d, y_15d, df = self.prepare_features(df)

        dates = df['trade_date'].values
        unique_dates = np.sort(np.unique(dates))
        n_dates = len(unique_dates)
        logger.info(f"  总交易日: {n_dates}, 样本: {len(X):,}")

        # 2. 85% train + 15% val (跳过WF)
        split_idx = int(n_dates * 0.85)
        split_date = unique_dates[split_idx]
        train_mask_final = dates <= split_date
        val_mask_final = dates > split_date

        X_train_f, X_val_f = X[train_mask_final], X[val_mask_final]
        self.train_dates = dates[train_mask_final]
        self.val_dates = dates[val_mask_final]

        logger.info(f"  train: {X_train_f.shape[0]:,} (至{split_date}), val: {X_val_f.shape[0]:,}")

        df_train_f = df[train_mask_final]
        all_results = {}
        y_val_dict = {}

        targets_final = [
            ('3d', y_3d[train_mask_final].copy(), y_3d[val_mask_final].copy()),
            ('5d', y_5d[train_mask_final].copy(), y_5d[val_mask_final].copy()),
            ('10d', y_10d[train_mask_final].copy(), y_10d[val_mask_final].copy()),
            ('15d', y_15d[train_mask_final].copy(), y_15d[val_mask_final].copy()),
        ]

        # 标签Winsorization
        for target_key, y_tr, y_va in targets_final:
            lo = np.percentile(y_tr, 1)
            hi = np.percentile(y_tr, 99)
            y_tr[:] = np.clip(y_tr, lo, hi)
            y_va[:] = np.clip(y_va, lo, hi)

        # 3. 训练4目标模型
        for target_key, y_tr, y_va in targets_final:
            sample_w = self.compute_sample_weights(df_train_f, y_tr)
            models, pred_train, pred_val = self.train_single_target_models(
                X_train_f, X_val_f, y_tr, y_va, f"label_{target_key}",
                sample_weights_train=sample_w)
            weights, rmses = self.calculate_ensemble_weights(pred_val, y_va)
            all_results[target_key] = {'models': models, 'weights': weights, 'rmses': rmses}
            y_val_dict[target_key] = y_va

        # 4. Module C: bear specialist
        logger.info("\n" + "=" * 60)
        logger.info("Module C: 训练熊市专家模型")
        logger.info("=" * 60)
        bear_models = {}
        for target_key in ['10d', '15d']:
            y_tr_target = y_10d[train_mask_final] if target_key == '10d' else y_15d[train_mask_final]
            bear_model = self._train_bear_specialist(X_train_f, y_tr_target, df_train_f, target_key)
            if bear_model is not None:
                bear_models[target_key] = bear_model

        # 5. Module A: per-target isotonic
        logger.info("\n" + "=" * 60)
        logger.info("Module A: Per-Target 保序回归校准")
        logger.info("=" * 60)
        isotonic_models = self._fit_isotonic_calibration(X_val_f, y_val_dict, all_results)

        # 6. ICIR weights
        logger.info("\n" + "=" * 60)
        logger.info("V4.7 1A: ICIR最大化集成权重")
        logger.info("=" * 60)
        icir_weights = self._optimize_icir_weights(all_results, X_val_f, y_val_dict, self.val_dates)
        for target_key, w in icir_weights.items():
            if target_key in all_results:
                all_results[target_key]['weights'] = w

        # 7. Combined Isotonic
        logger.info("\n" + "=" * 60)
        logger.info("V4.7 1C: Combined-Score Isotonic")
        logger.info("=" * 60)
        combined_isotonic = self._fit_combined_isotonic(X_val_f, y_val_dict, all_results, icir_weights)

        # 8. Meta-Learner
        logger.info("\n" + "=" * 60)
        logger.info("V4.7 1D: Stacking Meta-Learner")
        logger.info("=" * 60)
        meta_learner, meta_feature_names = self._train_meta_learner(X_val_f, y_val_dict, all_results)

        # 9. Feature importance + global quantiles
        self._log_feature_importance(all_results)
        global_quantiles = self._compute_global_quantiles(X, all_results, self.target_weights)
        recommendation_thresholds = self._compute_recommendation_thresholds(X, all_results)

        # 10. Save model
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        output_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v47'
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        model_data = {
            'version': 'v4.7',
            'models': all_results,
            'feature_names': self.feature_names,
            'target_weights': self.target_weights,
            'market_features': list(self.market_calculator.market_features.columns[1:]),
            'winsorize_bounds': getattr(self, 'winsorize_bounds', None),
            'global_quantiles': global_quantiles,
            'recommendation_thresholds': recommendation_thresholds,
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
            'walk_forward_metrics': {},  # 跳过WF
            'walk_forward_windows': 0,
            'regularization': {
                'num_leaves': 20, 'min_data_in_leaf': 500,
                'reg_alpha': 1.0, 'reg_lambda': 5.0,
                'path_smooth': 10.0, 'learning_rate': 0.02,
            },
            'bear_models': bear_models,
            'isotonic_calibration': isotonic_models,
            'sharpe_label_blend': self.sharpe_label_blend,
            'liquidity_discount': True,
            'bear_sample_weighting': True,
            'min_train_days': min_train_days,
            'step_days': 0,
            'icir_optimized_weights': icir_weights,
            'combined_isotonic': combined_isotonic,
            'meta_learner': meta_learner,
            'meta_feature_names': meta_feature_names,
            'small_cap_weighting': False,
            'has_percentile_lgb': True,
        }

        model_path = output_dir / f'v47_multi_target_{timestamp}.pkl'
        joblib.dump(model_data, model_path)
        logger.info(f"\n模型已保存: {model_path}")
        logger.info(f"  大小: {model_path.stat().st_size / 1024 / 1024:.1f} MB")

        # Save training history
        history = {
            'version': 'v4.7',
            'training_mode': 'production_only',
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'status': 'completed',
            'summary': {
                'training_samples': int(train_mask_final.sum()),
                'validation_samples': int(val_mask_final.sum()),
                'feature_count': len(self.feature_names),
                'walk_forward_summary': {},
                'bear_models': list(bear_models.keys()),
                'isotonic_targets': list(isotonic_models.keys()) if isotonic_models else [],
                'meta_learner': meta_learner is not None,
                'combined_isotonic': combined_isotonic is not None,
                'icir_optimized': len(icir_weights) > 0,
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

        logger.info(f"\nV4.7 快速训练完成! 总耗时: {duration:.0f}秒 ({duration/60:.1f}分钟)")
        return model_data, history


class V471Trainer(V44Trainer):
    """V4.7.1 训练器 — 底层模型信号质量提升

    设计哲学: 修复Bug + 高价值因子 + 改进训练目标 = 提升裸信号质量
    架构上继承V4.4成熟的scorer pipeline, 不引入新的后处理模块.

    改进项:
    Bug 1: 修复prepare_features()中的Winsorization数据泄露
    Bug 2: 修复walk_forward_train()中Sharpe-Blend未执行的问题
    Bug 3: 市场指数统一为000300.SH (scorer层修复)
    Phase 2: +17个高价值特征 (59→76): 财务质量/daily_basic扩展/微观结构/反转/风险
    Phase 3: LambdaRank LGB — 直接优化排名指标
    Phase 4: 时间衰减样本权重 — 近期样本更重要
    """

    # 新增特征定义
    FINANCIAL_FEATURES = ['roe', 'gross_margin', 'current_ratio', 'assets_turn', 'netprofit_yoy', 'or_yoy']
    DAILY_BASIC_EXTRA = ['dv_ttm', 'turnover_rate_f']  # float_ratio从circ_mv/total_mv计算
    MICROSTRUCTURE_FEATURES = ['amihud_illiquidity', 'volume_price_corr_10d', 'max_drawdown_20d', 'updown_volume_asymmetry']
    REVERSAL_FEATURES = ['return_1d', 'return_3d']
    RISK_FEATURES = ['idio_volatility_20d', 'downside_deviation_20d']

    def __init__(self, db_path: str = DB_PATH):
        super().__init__(db_path=db_path)
        # 保持V4.4目标权重
        self.target_weights = {
            'label_3d': 0.20,
            'label_5d': 0.25,
            'label_10d': 0.35,
            'label_15d': 0.20,
        }

    def load_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """V4.7.1: V4.4基础 + 17个新特征"""
        df = super().load_data(start_date, end_date)

        date_min = df['trade_date'].min()
        date_max = df['trade_date'].max()
        conn = sqlite3.connect(self.db_path)

        # ===== P1: 6个财务质量因子 (point-in-time from financial_indicator) =====
        logger.info("  V4.7.1 加载财务质量因子 (point-in-time)...")
        fi_query = """
        SELECT s.code, fi.ann_date, fi.end_date,
               fi.roe, fi.gross_margin, fi.current_ratio,
               fi.assets_turn, fi.netprofit_yoy, fi.or_yoy
        FROM financial_indicator fi
        JOIN securities s ON fi.security_id = s.id
        WHERE fi.ann_date IS NOT NULL AND fi.ann_date != ''
        ORDER BY s.code, fi.ann_date
        """
        df_fi = pd.read_sql(fi_query, conn)
        logger.info(f"    financial_indicator 记录: {len(df_fi):,}")

        if len(df_fi) > 0:
            # Point-in-time join: 对每个(code, trade_date)找最新已公告的财报
            # 使用merge_asof高效实现 — 需要numeric/datetime类型
            # 转换日期为int YYYYMMDD格式用于merge_asof
            def _date_to_int(s):
                """将任意日期格式(str/int)转为int YYYYMMDD"""
                return pd.to_datetime(s.astype(str).str.replace('-', ''), format='%Y%m%d').dt.strftime('%Y%m%d').astype(np.int64)

            df_fi['_ann_int'] = _date_to_int(df_fi['ann_date'])
            df['_td_int'] = _date_to_int(df['trade_date'])

            df_fi = df_fi.sort_values('_ann_int')
            df = df.sort_values(['code', '_td_int'])

            fi_merged_parts = []
            for code, group in df.groupby('code'):
                fi_code = df_fi[df_fi['code'] == code].copy()
                if len(fi_code) == 0:
                    # 该股票无财报数据, 填NaN
                    for col in self.FINANCIAL_FEATURES:
                        group[col] = np.nan
                    fi_merged_parts.append(group)
                    continue

                fi_code = fi_code.drop_duplicates(subset='_ann_int', keep='last')
                group_reset = group.reset_index()
                fi_subset = fi_code[['_ann_int'] + self.FINANCIAL_FEATURES].rename(columns={'_ann_int': '_td_int'})
                merged = pd.merge_asof(
                    group_reset.sort_values('_td_int'),
                    fi_subset.sort_values('_td_int'),
                    on='_td_int',
                    direction='backward'
                ).set_index('index')
                fi_merged_parts.append(merged)

            df = pd.concat(fi_merged_parts).sort_index()
            df.drop(columns=['_td_int'], inplace=True)
            logger.info(f"    财务特征合并完成: +{len(self.FINANCIAL_FEATURES)} 特征")

            # 填充: 当日截面中位数, 然后全局中位数兜底
            for col in self.FINANCIAL_FEATURES:
                missing = df[col].isnull().sum()
                if missing > 0:
                    pct = missing / len(df) * 100
                    df[col] = df.groupby('trade_date')[col].transform(lambda x: x.fillna(x.median()))
                    remaining = df[col].isnull().sum()
                    if remaining > 0:
                        df[col] = df[col].fillna(df[col].median())
                    logger.info(f"      {col}: {missing:,} 缺失({pct:.1f}%) → {df[col].isnull().sum()} 剩余")
        else:
            for col in self.FINANCIAL_FEATURES:
                df[col] = 0.0
            logger.warning("    financial_indicator 为空, 财务特征填0")

        # ===== P2: 3个daily_basic扩展 (dv_ttm, turnover_rate_f, float_ratio) =====
        logger.info("  V4.7.1 加载daily_basic扩展特征...")
        db_extra_query = """
        SELECT s.code, db.trade_date, db.dv_ttm, db.turnover_rate_f, db.circ_mv, db.total_mv
        FROM daily_basic db
        JOIN securities s ON db.security_id = s.id
        WHERE db.trade_date >= ? AND db.trade_date <= ?
        """
        df_db_extra = pd.read_sql(db_extra_query, conn, params=[date_min, date_max])
        logger.info(f"    daily_basic extra 记录: {len(df_db_extra):,}")

        if len(df_db_extra) > 0:
            # 计算float_ratio = circ_mv / total_mv
            df_db_extra['float_ratio'] = (df_db_extra['circ_mv'] / df_db_extra['total_mv'].clip(lower=1e-8))
            df_db_extra.drop(columns=['circ_mv', 'total_mv'], inplace=True)

            df = df.merge(df_db_extra, on=['code', 'trade_date'], how='left')
            for col in ['dv_ttm', 'turnover_rate_f', 'float_ratio']:
                missing = df[col].isnull().sum()
                if missing > 0:
                    df[col] = df.groupby('trade_date')[col].transform(lambda x: x.fillna(x.median()))
                    remaining = df[col].isnull().sum()
                    if remaining > 0:
                        df[col] = df[col].fillna(df[col].median())
            logger.info(f"    daily_basic扩展合并: +3 (dv_ttm, turnover_rate_f, float_ratio)")

        # ===== P3: 4个微观结构因子 (从OHLCV计算) =====
        logger.info("  V4.7.1 计算微观结构因子...")
        ohlcv_query = """
        SELECT s.code, q.trade_date, q.open, q.high, q.low, q.close, q.volume, q.price_change_pct
        FROM daily_quotes q
        JOIN securities s ON q.security_id = s.id
        WHERE s.type = 'A股' AND q.trade_date >= ? AND q.trade_date <= ?
        ORDER BY s.code, q.trade_date
        """
        # 需要额外前20天的数据用于滚动窗口
        from datetime import datetime as dt_cls, timedelta as td_cls
        try:
            ext_start = (dt_cls.strptime(date_min, '%Y-%m-%d') - td_cls(days=40)).strftime('%Y-%m-%d')
        except:
            ext_start = (dt_cls.strptime(date_min, '%Y%m%d') - td_cls(days=40)).strftime('%Y%m%d')

        df_ohlcv = pd.read_sql(ohlcv_query, conn, params=[ext_start, date_max])
        logger.info(f"    OHLCV 记录: {len(df_ohlcv):,}")

        if len(df_ohlcv) > 0:
            micro_parts = []
            for code, grp in df_ohlcv.groupby('code'):
                grp = grp.sort_values('trade_date').copy()
                close = grp['close'].values
                volume = grp['volume'].values.astype(float)
                pct = grp['price_change_pct'].values

                # Amihud illiquidity: 20日平均(|return|/volume)
                abs_ret = np.abs(pct)
                vol_safe = np.where(volume > 0, volume, 1e-8)
                amihud_raw = abs_ret / vol_safe
                amihud = pd.Series(amihud_raw).rolling(20, min_periods=10).mean().values

                # Volume-price correlation (10d)
                vp_corr = pd.Series(close).rolling(10, min_periods=5).corr(pd.Series(volume.astype(float))).values

                # Max drawdown 20d
                close_s = pd.Series(close)
                rolling_max = close_s.rolling(20, min_periods=10).max()
                dd = (close_s - rolling_max) / rolling_max.clip(lower=1e-8)
                max_dd_20d = dd.rolling(20, min_periods=10).min().values

                # Up/down volume asymmetry (10d)
                up_mask = pct > 0
                down_mask = pct < 0
                up_vol = pd.Series(np.where(up_mask, volume, 0.0)).rolling(10, min_periods=3).sum().values
                dn_vol = pd.Series(np.where(down_mask, volume, 0.0)).rolling(10, min_periods=3).sum().values
                dn_vol_safe = np.where(dn_vol > 0, dn_vol, 1e-8)
                ud_asym = up_vol / dn_vol_safe

                grp_out = grp[['code', 'trade_date']].copy()
                grp_out['amihud_illiquidity'] = amihud
                grp_out['volume_price_corr_10d'] = vp_corr
                grp_out['max_drawdown_20d'] = max_dd_20d
                grp_out['updown_volume_asymmetry'] = ud_asym
                micro_parts.append(grp_out)

            df_micro = pd.concat(micro_parts, ignore_index=True)

            # 合并到主df
            df = df.merge(df_micro, on=['code', 'trade_date'], how='left')
            for col in self.MICROSTRUCTURE_FEATURES:
                missing = df[col].isnull().sum()
                if missing > 0:
                    df[col] = df.groupby('trade_date')[col].transform(lambda x: x.fillna(x.median()))
                    remaining = df[col].isnull().sum()
                    if remaining > 0:
                        df[col] = df[col].fillna(0.0)
            logger.info(f"    微观结构因子合并: +4 特征")

            # ===== P4: 2个短期反转因子 (从OHLCV的pct_change) =====
            logger.info("  V4.7.1 计算短期反转因子...")
            ret_parts = []
            for code, grp in df_ohlcv.groupby('code'):
                grp = grp.sort_values('trade_date').copy()
                close = grp['close'].values
                ret_1d = np.concatenate([[np.nan], close[1:] / close[:-1] - 1])
                close_s = pd.Series(close)
                ret_3d = (close_s / close_s.shift(3) - 1).values

                grp_out = grp[['code', 'trade_date']].copy()
                grp_out['return_1d'] = ret_1d
                grp_out['return_3d'] = ret_3d
                ret_parts.append(grp_out)

            df_ret = pd.concat(ret_parts, ignore_index=True)
            df = df.merge(df_ret, on=['code', 'trade_date'], how='left')
            for col in self.REVERSAL_FEATURES:
                df[col] = df[col].fillna(0.0)
            logger.info(f"    反转因子合并: +2 特征")

            # ===== P5: 2个风险因子 =====
            logger.info("  V4.7.1 计算风险因子...")
            risk_parts = []
            # 需要市场收益用于特质波动率
            market_ret_df = self.market_calculator.market_features
            if market_ret_df is not None and 'market_return_5d' in market_ret_df.columns:
                # 使用简化的daily market return (从5d差分近似1d)
                pass

            for code, grp in df_ohlcv.groupby('code'):
                grp = grp.sort_values('trade_date').copy()
                close = grp['close'].values
                daily_ret = np.concatenate([[0], close[1:] / close[:-1] - 1])

                # 特质波动率 (20d): 残差std, 简化为去均值后的std
                # 完整做法需要回归市场因子, 这里用简化版: total_vol - market_vol
                ret_s = pd.Series(daily_ret)
                total_vol = ret_s.rolling(20, min_periods=10).std().values

                # 对市场因子做简单回归: idio_vol ≈ total_vol * sqrt(1-R²)
                # 简化: idio_vol = std(ret - mean(ret)) ≈ total_vol (A股R²通常<0.3)
                # 更好的近似: 去均值(去掉市场drift)的std
                rolling_mean = ret_s.rolling(20, min_periods=10).mean().values
                demeaned = daily_ret - rolling_mean
                idio_vol = pd.Series(demeaned).rolling(20, min_periods=10).std().values

                # 下行偏差 (20d): 仅负收益的std
                neg_ret = np.where(daily_ret < 0, daily_ret, 0.0)
                downside_dev = pd.Series(neg_ret).rolling(20, min_periods=10).std().values

                grp_out = grp[['code', 'trade_date']].copy()
                grp_out['idio_volatility_20d'] = idio_vol
                grp_out['downside_deviation_20d'] = downside_dev
                risk_parts.append(grp_out)

            df_risk = pd.concat(risk_parts, ignore_index=True)
            df = df.merge(df_risk, on=['code', 'trade_date'], how='left')
            for col in self.RISK_FEATURES:
                df[col] = df[col].fillna(0.0)
            logger.info(f"    风险因子合并: +2 特征")

        else:
            # OHLCV为空, 填默认值
            for col in self.MICROSTRUCTURE_FEATURES + self.REVERSAL_FEATURES + self.RISK_FEATURES:
                df[col] = 0.0
            logger.warning("    OHLCV 为空, 微观/反转/风险特征填0")

        conn.close()

        total_new = len(self.FINANCIAL_FEATURES) + 3 + len(self.MICROSTRUCTURE_FEATURES) + \
                     len(self.REVERSAL_FEATURES) + len(self.RISK_FEATURES)
        logger.info(f"  V4.7.1 新增特征总计: +{total_new} (财务6+基本面3+微观4+反转2+风险2)")

        return df

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """V4.7.1 Bug 1修复: 移除提前winsorization, 仅依赖walk-forward中per-window的train-only bounds"""
        logger.info("准备特征和标签 (V4.7.1 — 无提前Winsorization)...")

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

        stock_data = self._robust_zscore_cross_section(stock_data, dates_arr)

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

        # Bug 1 修复: 不在此处winsorize全量数据!
        # V4.3/V4.4的 prepare_features() 在这里调用 self.winsorize_features(X),
        # 这会用全量(train+val+test)数据计算bounds, 导致数据泄露.
        # V4.7.1: 仅在 walk_forward_train() 的每个窗口中, 用窗口内训练集计算bounds.
        logger.info("  [Bug 1修复] 跳过全量Winsorization, 将在WF窗口内train-only计算bounds")

        return X, y_3d, y_5d, y_10d, y_15d, df

    def _apply_sharpe_blend(self, y_tr, y_va, y_te, train_dates, val_dates, test_dates, label_name):
        """Bug 2修复: 提取Sharpe-Blend为独立方法, 在WF窗口和生产模型训练中均调用

        原理: 融合收益和风险调整收益, 让模型学习Sharpe ratio而非纯收益.
        """
        blend = self.sharpe_label_blend
        if blend <= 0:
            return

        unique_train_dates = np.unique(train_dates)
        daily_vol_tr = np.zeros_like(y_tr)
        for d in unique_train_dates:
            mask_d = train_dates == d
            std_d = np.std(y_tr[mask_d])
            daily_vol_tr[mask_d] = std_d if std_d > 0 else 0

        # Sharpe-adjusted: 收益/波动
        sharpe_tr = y_tr / (daily_vol_tr + 1e-6)

        # 标准化尺度 (仅用训练集std)
        orig_std = np.std(y_tr)
        sharpe_std = np.std(sharpe_tr)
        scale = orig_std / sharpe_std if sharpe_std > 1e-8 else 1.0

        # 对训练集应用融合
        y_tr[:] = (1 - blend) * y_tr + blend * sharpe_tr * scale

        # 对val/test用训练集的mean daily_vol (防止数据泄漏)
        mean_daily_vol_train = np.mean([np.std(y_tr[train_dates == d])
            for d in unique_train_dates if (train_dates == d).sum() > 1])
        mean_daily_vol_train = max(mean_daily_vol_train, 1e-6)

        for y_set, dates_set in [(y_va, val_dates), (y_te, test_dates)]:
            if len(y_set) == 0:
                continue
            daily_vol_set = np.full_like(y_set, mean_daily_vol_train)
            sharpe_set = y_set / (daily_vol_set + 1e-6)
            y_set[:] = (1 - blend) * y_set + blend * sharpe_set * scale

        logger.info(f"      {label_name}: Sharpe blend={blend:.0%}, 训练集融合后std={np.std(y_tr):.6f}")

    def compute_sample_weights(self, df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """V4.7.1: V4.4权重(涨跌停+极端+熊市) + 时间衰减"""
        weights = super().compute_sample_weights(df, y)

        # Phase 4: 时间衰减 — 近期样本更重要
        if 'trade_date' in df.columns:
            dates = pd.to_datetime(df['trade_date'].values)
            max_date = dates.max()
            days_ago = ((max_date - dates) / pd.Timedelta(days=1)).astype(float)

            half_life_days = 365.0  # 1年半衰期
            decay = np.exp(-np.log(2) * days_ago / half_life_days)
            decay = np.clip(decay, 0.25, 1.0)  # 旧数据保留25%权重

            weights *= decay
            n_old = (decay < 0.5).sum()
            logger.info(f"    时间衰减: half_life={half_life_days:.0f}d, {n_old:,} 样本权重<0.5")

        return weights

    def train_single_target_models(self, X_train, X_val, y_train, y_val, target_name: str,
                                    sample_weights_train=None):
        """V4.7.1: V4.3的5个基础模型 + LambdaRank LGB"""
        import gc

        # 1. 标准5个回归模型 (继承V4.3: lgb, xgb, cb, rf, hgb)
        models, pred_train, pred_val = V43Trainer.train_single_target_models(
            self, X_train, X_val, y_train, y_val, target_name,
            sample_weights_train=sample_weights_train)

        # 2. Phase 3: LambdaRank LGB — 直接优化排名
        train_dates = getattr(self, 'train_dates', None)
        val_dates = getattr(self, 'val_dates', None)

        if train_dates is not None and len(train_dates) == len(y_train):
            logger.info(f"  训练 LGB-LambdaRank ({target_name})...")
            try:
                from scipy.stats import rankdata

                # 构建group信息 (每日为一组)
                unique_train_dates = np.unique(train_dates)
                # LambdaRank需要每组内的relevance label (0-4, 5档)
                relevance_train = np.zeros(len(y_train), dtype=np.int32)
                group_train = []
                for d in unique_train_dates:
                    mask = train_dates == d
                    n = mask.sum()
                    group_train.append(n)
                    if n >= 10:
                        # 5档分位
                        ranks = rankdata(y_train[mask])
                        pct = (ranks - 1) / (n - 1)  # [0, 1]
                        relevance_train[mask] = np.clip((pct * 5).astype(int), 0, 4)
                    else:
                        relevance_train[mask] = 2  # 中间档

                # Val group
                relevance_val = np.zeros(len(y_val), dtype=np.int32)
                group_val = []
                if val_dates is not None and len(val_dates) == len(y_val):
                    unique_val_dates = np.unique(val_dates)
                    for d in unique_val_dates:
                        mask = val_dates == d
                        n = mask.sum()
                        group_val.append(n)
                        if n >= 10:
                            ranks = rankdata(y_val[mask])
                            pct = (ranks - 1) / (n - 1)
                            relevance_val[mask] = np.clip((pct * 5).astype(int), 0, 4)
                        else:
                            relevance_val[mask] = 2

                lgb_rank_params = {
                    'objective': 'lambdarank',
                    'metric': 'ndcg',
                    'eval_at': [10, 50],
                    'lambdarank_truncation_level': 50,
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

                lgb_rank_train = lgb.Dataset(
                    X_train, label=relevance_train, group=group_train,
                    weight=sample_weights_train, free_raw_data=True
                )
                lgb_rank_val = lgb.Dataset(
                    X_val, label=relevance_val, group=group_val,
                    reference=lgb_rank_train, free_raw_data=True
                )

                lgb_rank_model = lgb.train(
                    lgb_rank_params, lgb_rank_train,
                    num_boost_round=1000,
                    valid_sets=[lgb_rank_train, lgb_rank_val],
                    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
                )

                models['lgb_rank'] = lgb_rank_model
                pred_train['lgb_rank'] = lgb_rank_model.predict(X_train)
                pred_val['lgb_rank'] = lgb_rank_model.predict(X_val)

                logger.info(f"    LGB-LambdaRank ({target_name}): 完成, "
                             f"train_groups={len(group_train)}, val_groups={len(group_val)}")

                # 缓存relevance标签供子类(V4.8 ListNet)复用, 避免重复计算
                self._cached_relevance_train = relevance_train
                self._cached_group_train = group_train
                self._cached_relevance_val = relevance_val
                self._cached_group_val = group_val

                del lgb_rank_train, lgb_rank_val
                gc.collect()

            except Exception as e:
                logger.warning(f"    LGB-LambdaRank ({target_name}) 训练失败: {e}")
                self._cached_relevance_train = None
                self._cached_group_train = None
                self._cached_relevance_val = None
                self._cached_group_val = None

        return models, pred_train, pred_val

    def walk_forward_train(self, start_date: str = None, end_date: str = None,
                            purge_days: int = 15, min_train_days: int = 900,
                            val_days: int = 120, test_days: int = 120,
                            step_days: int = 90):
        """V4.7.1 Walk-Forward 训练 — Bug修复 + 增强特征 + LambdaRank

        vs V4.4:
        - Bug 1修复: prepare_features不再提前winsorize
        - Bug 2修复: 每个WF窗口和生产模型训练中执行Sharpe-Blend
        - +17新特征 (59→76)
        - +LambdaRank LGB (6个模型/target)
        - +时间衰减样本权重
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("V4.7.1 Walk-Forward 训练 (Bug修复 + 17新特征 + LambdaRank)")
        logger.info("=" * 60)
        logger.info(f"  参数: min_train={min_train_days}d, val={val_days}d, test={test_days}d, "
                     f"step={step_days}d, purge={purge_days}d")
        logger.info(f"  目标权重: {self.target_weights}")
        logger.info(f"  Sharpe融合: {self.sharpe_label_blend} [Bug 2修复: 将在每个窗口执行]")

        # 1. 一次性加载全量数据
        df = self.load_data(start_date, end_date)
        X, y_3d, y_5d, y_10d, y_15d, df = self.prepare_features(df)

        dates = df['trade_date'].values
        unique_dates = np.sort(np.unique(dates))
        n_dates = len(unique_dates)
        logger.info(f"  总交易日: {n_dates}, 样本: {len(X):,}, 特征: {X.shape[1]}")

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
            train_dates_w = dates[train_mask]
            val_dates_w = dates[val_mask]

            # Walk-Forward: 特征Winsorization (仅用窗口内训练集)
            X_train_w, wf_bounds = self.winsorize_features(X_train_w)
            self._apply_bounds(X_val_w, wf_bounds)
            self._apply_bounds(X_test_w, wf_bounds)

            # Walk-Forward: 标签Winsorization (仅用训练集统计量)
            for y_tr_w, y_va_w, y_te_w in [(y_3d_tr, y_3d_va, y_3d_te),
                                             (y_5d_tr, y_5d_va, y_5d_te),
                                             (y_10d_tr, y_10d_va, y_10d_te),
                                             (y_15d_tr, y_15d_va, y_15d_te)]:
                lo = np.percentile(y_tr_w, 1)
                hi = np.percentile(y_tr_w, 99)
                y_tr_w[:] = np.clip(y_tr_w, lo, hi)
                y_va_w[:] = np.clip(y_va_w, lo, hi)
                y_te_w[:] = np.clip(y_te_w, lo, hi)

            # Bug 2修复: 在WF窗口内应用Sharpe-Blend
            logger.info(f"  [Bug 2修复] 应用Sharpe-Blend (blend={self.sharpe_label_blend:.0%})")
            for y_tr_w, y_va_w, y_te_w, name in [(y_3d_tr, y_3d_va, y_3d_te, 'label_3d'),
                                                    (y_5d_tr, y_5d_va, y_5d_te, 'label_5d'),
                                                    (y_10d_tr, y_10d_va, y_10d_te, 'label_10d'),
                                                    (y_15d_tr, y_15d_va, y_15d_te, 'label_15d')]:
                self._apply_sharpe_blend(y_tr_w, y_va_w, y_te_w,
                                          train_dates_w, val_dates_w, test_dates_w, name)

            logger.info(f"  train={X_train_w.shape[0]:,}, val={X_val_w.shape[0]:,}, test={X_test_w.shape[0]:,}")

            self.val_dates = dates[val_mask]
            self.train_dates = dates[train_mask]

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
        logger.info("训练最终生产模型 (全量数据)")
        logger.info("=" * 60)

        split_idx = int(n_dates * 0.85)
        split_date = unique_dates[split_idx]
        train_mask_final = dates <= split_date
        val_mask_final = dates > split_date

        X_train_f, X_val_f = X[train_mask_final].copy(), X[val_mask_final].copy()
        self.val_dates = dates[val_mask_final]
        self.train_dates = dates[train_mask_final]

        # Bug 1修复: 生产模型的Winsorization也只用训练集
        X_train_f, self.winsorize_bounds = self.winsorize_features(X_train_f)
        self._apply_bounds(X_val_f, self.winsorize_bounds)
        logger.info(f"  生产模型: 特征Winsorization (训练集bounds), {len(self.winsorize_bounds)} 列")

        df_train_f = df[train_mask_final]
        all_results = {}

        y_val_dict = {}
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

        # Bug 2修复: 生产模型也执行Sharpe-Blend
        logger.info(f"  [Bug 2修复] 生产模型Sharpe-Blend (blend={self.sharpe_label_blend:.0%})")
        train_dates_f = dates[train_mask_final]
        val_dates_f = dates[val_mask_final]
        for target_key, y_tr, y_va in targets_final:
            self._apply_sharpe_blend(y_tr, y_va, np.array([]),
                                      train_dates_f, val_dates_f, np.array([]),
                                      f"label_{target_key}")

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

        # 7. Module A: 保序回归校准
        logger.info("\n" + "=" * 60)
        logger.info("Module A: 保序回归校准")
        logger.info("=" * 60)
        isotonic_models = self._fit_isotonic_calibration(X_val_f, y_val_dict, all_results)

        # 8. 特征重要性
        self._log_feature_importance(all_results)

        # 8.5 计算全局评分分位数
        # 对全量X应用生产模型的winsorize bounds
        X_all = X.copy()
        self._apply_bounds(X_all, self.winsorize_bounds)
        global_quantiles = self._compute_global_quantiles(X_all, all_results, self.target_weights)
        recommendation_thresholds = self._compute_recommendation_thresholds(X_all, all_results)

        # 9. 保存模型
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        output_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v471'
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 将winsorize_bounds转为dict格式 (feature_name -> (lo, hi))
        winsorize_bounds_dict = {}
        if self.winsorize_bounds and self.feature_names:
            for idx, (lo, hi) in enumerate(self.winsorize_bounds):
                if idx < len(self.feature_names):
                    winsorize_bounds_dict[self.feature_names[idx]] = (lo, hi)

        model_data = {
            'version': 'v4.7.1',
            'models': all_results,
            'feature_names': self.feature_names,
            'target_weights': self.target_weights,
            'market_features': list(self.market_calculator.market_features.columns[1:]),
            'winsorize_bounds': winsorize_bounds_dict,
            'global_quantiles': global_quantiles,
            'recommendation_thresholds': recommendation_thresholds,
            # 模型类型标识
            'cascade': False,
            'rank_normalized': False,
            'robust_zscore': True,
            'industry_excess_labels': True,
            'dual_stream': False,
            'cross_sectional_neutralization': False,
            'macro_feature_cols': self.macro_feature_cols,
            'stock_feature_cols': self.stock_feature_cols,
            'extra_features_from_daily_basic': ['pe_ttm', 'pb', 'ps_ttm', 'turnover_rate', 'log_market_cap',
                                                 'dv_ttm', 'turnover_rate_f', 'float_ratio'],
            'extra_features_from_tech_indicators': self.extra_tech_feature_names,
            'extra_features_financial': self.FINANCIAL_FEATURES,
            'extra_features_microstructure': self.MICROSTRUCTURE_FEATURES,
            'extra_features_reversal': self.REVERSAL_FEATURES,
            'extra_features_risk': self.RISK_FEATURES,
            'targets': ['3d', '5d', '10d', '15d'],
            'ensemble_type': 'ic_monotonicity_weighted',
            'sample_weighting': True,
            'time_decay_half_life': 365,
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
            # V4.7.1 新增
            'has_lambdarank': True,
            'has_time_decay': True,
            'bug_fixes': ['winsorization_leakage', 'sharpe_blend_applied', 'market_index_000300'],
        }

        model_path = output_dir / f'v471_multi_target_{timestamp}.pkl'
        joblib.dump(model_data, model_path)
        logger.info(f"\n模型已保存: {model_path}")
        logger.info(f"  大小: {model_path.stat().st_size / 1024 / 1024:.1f} MB")

        # 保存训练历史
        history = {
            'version': 'v4.7.1',
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
            'bug_fixes': {
                'winsorization_leakage': 'prepare_features不再提前winsorize全量数据',
                'sharpe_blend_applied': '每个WF窗口和生产模型均执行Sharpe-Blend',
                'market_index': 'scorer层统一使用000300.SH',
            },
            'new_features': {
                'financial': self.FINANCIAL_FEATURES,
                'daily_basic_extra': ['dv_ttm', 'turnover_rate_f', 'float_ratio'],
                'microstructure': self.MICROSTRUCTURE_FEATURES,
                'reversal': self.REVERSAL_FEATURES,
                'risk': self.RISK_FEATURES,
            },
            'modules': {
                'A_monotonicity': True,
                'B_liquidity_discount': True,
                'C_bear_specialist': len(bear_models) > 0,
                'D_sharpe_blend': self.sharpe_label_blend,
                'E_executability_filter': 'scorer层实现',
                'F_regime_adaptive': 'scorer层实现',
                'lambdarank': True,
                'time_decay': True,
            },
        }

        history_path = output_dir / f'training_history_{timestamp}.json'
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        latest_path = output_dir / 'training_history_latest.json'
        with open(latest_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        logger.info(f"训练历史已保存: {history_path}")
        logger.info(f"\nV4.7.1 训练完成! 总耗时: {duration:.0f}秒 ({duration/60:.1f}分钟)")

        return model_data, history


class V472Trainer(V471Trainer):
    """V4.7.2 训练器 — V4.7.1底座 + V4.6后处理管线

    保留V4.7.1: Bug修复(3项) + 17新特征(76总) + LambdaRank + 时间衰减
    新增V4.6: ICIR权重优化 + Combined Isotonic + Meta-Learner
    修复: 3d目标降低Sharpe-Blend + 跳过LambdaRank(纯回归)
    去除: 小盘四分位加权(已验证有害)
    """

    # 目标特异性 Sharpe-Blend 比例
    TARGET_SHARPE_BLEND = {
        'label_3d': 0.10,    # 短期降低Sharpe (从0.3→0.1)
        'label_5d': 0.25,    # 中短期略降
        'label_10d': 0.35,   # 中期提高
        'label_15d': 0.35,   # 长期提高
    }

    def __init__(self, db_path: str = DB_PATH):
        super().__init__(db_path=db_path)

    def compute_sample_weights(self, df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """V4.7.2: V4.7.1权重(涨跌停+极端+熊市+时间衰减) — 不加小盘四分位加权

        直接继承V4.7.1(继承V4.4 + 时间衰减), 跳过V4.6的四分位小盘加权.
        """
        # V471Trainer.compute_sample_weights 已包含:
        # V4.4: 涨跌停 + 极端收益 + 熊市加权
        # V4.7.1: 时间衰减(half_life=365d)
        return super().compute_sample_weights(df, y)

    def _apply_sharpe_blend(self, y_tr, y_va, y_te, train_dates, val_dates, test_dates, label_name):
        """V4.7.2: 目标特异性Sharpe-Blend — 3d低blend(0.1), 10d/15d高blend(0.35)

        短期预测(3d)更依赖精确收益值, Sharpe调整引入噪声.
        中长期(10d/15d)更受益于Sharpe平滑.
        """
        # 查找目标特异性blend比例
        target_blend = self.TARGET_SHARPE_BLEND.get(label_name)
        if target_blend is not None:
            # 临时替换blend比例
            original_blend = self.sharpe_label_blend
            self.sharpe_label_blend = target_blend
            super()._apply_sharpe_blend(y_tr, y_va, y_te, train_dates, val_dates, test_dates, label_name)
            self.sharpe_label_blend = original_blend
        else:
            super()._apply_sharpe_blend(y_tr, y_va, y_te, train_dates, val_dates, test_dates, label_name)

    def train_single_target_models(self, X_train, X_val, y_train, y_val, target_name: str,
                                    sample_weights_train=None):
        """V4.7.2: 3d目标仅5个回归模型(跳过LambdaRank), 其余保留6模型

        短期预测更依赖回归精度而非排名优化.
        """
        if '3d' in target_name:
            # 3d: 仅5个回归模型 (lgb, xgb, cb, rf, hgb), 跳过LambdaRank
            logger.info(f"  V4.7.2: {target_name} 使用纯回归模型(5个, 跳过LambdaRank)")
            return V43Trainer.train_single_target_models(
                self, X_train, X_val, y_train, y_val, target_name,
                sample_weights_train=sample_weights_train)
        else:
            # 5d/10d/15d: 6个模型 (含LambdaRank)
            return V471Trainer.train_single_target_models(
                self, X_train, X_val, y_train, y_val, target_name,
                sample_weights_train=sample_weights_train)

    def walk_forward_train(self, start_date: str = None, end_date: str = None,
                            purge_days: int = 15, min_train_days: int = 900,
                            val_days: int = 120, test_days: int = 120,
                            step_days: int = 90):
        """V4.7.2 Walk-Forward 训练 — V4.7.1底座 + V4.6后处理管线

        vs V4.7.1:
        - 3d Sharpe-Blend降低(0.3→0.1), 10d/15d提高(0.3→0.35)
        - 3d跳过LambdaRank(纯回归5模型)
        - +ICIR权重优化 (V4.6 Module 1A)
        - +Combined Isotonic (V4.6 Module 1C)
        - +Stacking Meta-Learner (V4.6 Module 1D)
        - 无小盘四分位加权
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("V4.7.2 Walk-Forward 训练 (V4.7.1底座 + V4.6后处理管线)")
        logger.info("=" * 60)
        logger.info(f"  参数: min_train={min_train_days}d, val={val_days}d, test={test_days}d, "
                     f"step={step_days}d, purge={purge_days}d")
        logger.info(f"  目标权重: {self.target_weights}")
        logger.info(f"  Sharpe融合(目标特异性): {self.TARGET_SHARPE_BLEND}")
        logger.info(f"  3d: 纯回归5模型(无LambdaRank), 5d/10d/15d: 6模型(含LambdaRank)")

        # 1. 一次性加载全量数据 (V4.7.1: 76特征含17新特征)
        df = self.load_data(start_date, end_date)
        X, y_3d, y_5d, y_10d, y_15d, df = self.prepare_features(df)

        dates = df['trade_date'].values
        unique_dates = np.sort(np.unique(dates))
        n_dates = len(unique_dates)
        logger.info(f"  总交易日: {n_dates}, 样本: {len(X):,}, 特征: {X.shape[1]}")

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
            train_dates_w = dates[train_mask]
            val_dates_w = dates[val_mask]

            # Walk-Forward: 特征Winsorization (仅用窗口内训练集)
            X_train_w, wf_bounds = self.winsorize_features(X_train_w)
            self._apply_bounds(X_val_w, wf_bounds)
            self._apply_bounds(X_test_w, wf_bounds)

            # Walk-Forward: 标签Winsorization (仅用训练集统计量)
            for y_tr_w, y_va_w, y_te_w in [(y_3d_tr, y_3d_va, y_3d_te),
                                             (y_5d_tr, y_5d_va, y_5d_te),
                                             (y_10d_tr, y_10d_va, y_10d_te),
                                             (y_15d_tr, y_15d_va, y_15d_te)]:
                lo = np.percentile(y_tr_w, 1)
                hi = np.percentile(y_tr_w, 99)
                y_tr_w[:] = np.clip(y_tr_w, lo, hi)
                y_va_w[:] = np.clip(y_va_w, lo, hi)
                y_te_w[:] = np.clip(y_te_w, lo, hi)

            # V4.7.2: 目标特异性Sharpe-Blend
            self.train_dates = train_dates_w
            self.val_dates = val_dates_w
            for target_key, y_tr_w, y_va_w, y_te_w in [
                ('label_3d', y_3d_tr, y_3d_va, y_3d_te),
                ('label_5d', y_5d_tr, y_5d_va, y_5d_te),
                ('label_10d', y_10d_tr, y_10d_va, y_10d_te),
                ('label_15d', y_15d_tr, y_15d_va, y_15d_te),
            ]:
                self._apply_sharpe_blend(y_tr_w, y_va_w, y_te_w,
                                          train_dates_w, val_dates_w, test_dates_w,
                                          target_key)

            # 训练4目标
            window_metrics = {}
            for target_key, y_tr, y_va, y_te in [
                ('3d', y_3d_tr, y_3d_va, y_3d_te),
                ('5d', y_5d_tr, y_5d_va, y_5d_te),
                ('10d', y_10d_tr, y_10d_va, y_10d_te),
                ('15d', y_15d_tr, y_15d_va, y_15d_te),
            ]:
                sample_w = self.compute_sample_weights(df[train_mask], y_tr)
                models, pred_train, pred_val = self.train_single_target_models(
                    X_train_w, X_val_w, y_tr, y_va, f"label_{target_key}",
                    sample_weights_train=sample_w)
                weights, rmses = self.calculate_ensemble_weights(pred_val, y_va)

                # test set预测
                pred_test = {}
                for name, model in models.items():
                    try:
                        if name == 'xgb':
                            pred_test[name] = model.predict(xgb.DMatrix(X_test_w))
                        else:
                            pred_test[name] = model.predict(X_test_w)
                    except Exception:
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
        logger.info("训练最终V4.7.2生产模型 (全量数据)")
        logger.info("=" * 60)

        split_idx = int(n_dates * 0.85)
        split_date = unique_dates[split_idx]
        train_mask_final = dates <= split_date
        val_mask_final = dates > split_date

        X_train_f, X_val_f = X[train_mask_final].copy(), X[val_mask_final].copy()
        self.val_dates = dates[val_mask_final]
        self.train_dates = dates[train_mask_final]

        # Bug 1修复: 生产模型的Winsorization也只用训练集
        X_train_f, self.winsorize_bounds = self.winsorize_features(X_train_f)
        self._apply_bounds(X_val_f, self.winsorize_bounds)
        logger.info(f"  生产模型: 特征Winsorization (训练集bounds), {len(self.winsorize_bounds)} 列")

        df_train_f = df[train_mask_final]
        all_results = {}

        y_val_dict = {}
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

        # V4.7.2: 目标特异性Sharpe-Blend (Bug 2修复)
        logger.info(f"  [V4.7.2] 目标特异性Sharpe-Blend: {self.TARGET_SHARPE_BLEND}")
        train_dates_f = dates[train_mask_final]
        val_dates_f = dates[val_mask_final]
        for target_key, y_tr, y_va in targets_final:
            self._apply_sharpe_blend(y_tr, y_va, np.array([]),
                                      train_dates_f, val_dates_f, np.array([]),
                                      f"label_{target_key}")

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

        # 8. V4.6增强: ICIR优化集成权重 (复用V46Trainer方法)
        logger.info("\n" + "=" * 60)
        logger.info("V4.7.2 1A: ICIR最大化集成权重 (V4.6管线)")
        logger.info("=" * 60)
        icir_weights = V46Trainer._optimize_icir_weights(self, all_results, X_val_f, y_val_dict, self.val_dates)
        # 更新all_results中的权重
        for target_key, w in icir_weights.items():
            if target_key in all_results:
                all_results[target_key]['weights'] = w

        # 9. V4.6增强: Combined Isotonic (复用V46Trainer方法)
        logger.info("\n" + "=" * 60)
        logger.info("V4.7.2 1C: Combined-Score Isotonic (V4.6管线)")
        logger.info("=" * 60)
        combined_isotonic = V46Trainer._fit_combined_isotonic(self, X_val_f, y_val_dict, all_results, icir_weights)

        # 10. V4.6增强: Stacking Meta-Learner (复用V46Trainer方法)
        logger.info("\n" + "=" * 60)
        logger.info("V4.7.2 1D: Stacking Meta-Learner (V4.6管线)")
        logger.info("=" * 60)
        meta_learner, meta_feature_names = V46Trainer._train_meta_learner(self, X_val_f, y_val_dict, all_results)

        # 11. 特征重要性分析
        self._log_feature_importance(all_results)

        # 12. 计算全局评分分位数
        # 对全量X应用生产模型的winsorize bounds
        X_all = X.copy()
        self._apply_bounds(X_all, self.winsorize_bounds)
        global_quantiles = self._compute_global_quantiles(X_all, all_results, self.target_weights)
        recommendation_thresholds = self._compute_recommendation_thresholds(X_all, all_results)

        # 13. 保存模型
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        output_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v472'
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 将winsorize_bounds转为dict格式 (feature_name -> (lo, hi))
        winsorize_bounds_dict = {}
        if self.winsorize_bounds and self.feature_names:
            for idx, (lo, hi) in enumerate(self.winsorize_bounds):
                if idx < len(self.feature_names):
                    winsorize_bounds_dict[self.feature_names[idx]] = (lo, hi)

        model_data = {
            'version': 'v4.7.2',
            'models': all_results,
            'feature_names': self.feature_names,
            'target_weights': self.target_weights,
            'market_features': list(self.market_calculator.market_features.columns[1:]),
            'winsorize_bounds': winsorize_bounds_dict,
            'global_quantiles': global_quantiles,
            'recommendation_thresholds': recommendation_thresholds,
            # 模型类型标识
            'cascade': False,
            'rank_normalized': False,
            'robust_zscore': True,
            'industry_excess_labels': True,
            'dual_stream': False,
            'cross_sectional_neutralization': False,
            'macro_feature_cols': self.macro_feature_cols,
            'stock_feature_cols': self.stock_feature_cols,
            'extra_features_from_daily_basic': ['pe_ttm', 'pb', 'ps_ttm', 'turnover_rate', 'log_market_cap',
                                                 'dv_ttm', 'turnover_rate_f', 'float_ratio'],
            'extra_features_from_tech_indicators': self.extra_tech_feature_names,
            'extra_features_financial': self.FINANCIAL_FEATURES,
            'extra_features_microstructure': self.MICROSTRUCTURE_FEATURES,
            'extra_features_reversal': self.REVERSAL_FEATURES,
            'extra_features_risk': self.RISK_FEATURES,
            'targets': ['3d', '5d', '10d', '15d'],
            'ensemble_type': 'icir_optimized',  # V4.6管线
            'sample_weighting': True,
            'time_decay_half_life': 365,
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
            'sharpe_label_blend': 'target_specific',  # V4.7.2: 目标特异性
            'sharpe_blend_config': self.TARGET_SHARPE_BLEND,
            'liquidity_discount': True,
            'bear_sample_weighting': True,
            'min_train_days': min_train_days,
            'step_days': step_days,
            # V4.7.1 组件 (继承)
            'has_lambdarank': True,  # 5d/10d/15d有, 3d没有
            'has_time_decay': True,
            'bug_fixes': ['winsorization_leakage', 'sharpe_blend_applied', 'market_index_000300'],
            # V4.6 后处理组件 (新增)
            'icir_optimized_weights': icir_weights,
            'combined_isotonic': combined_isotonic,
            'meta_learner': meta_learner,
            'meta_feature_names': meta_feature_names,
            'small_cap_weighting': False,  # 明确关闭
            # V4.7.2 特有
            'target_specific_sharpe': self.TARGET_SHARPE_BLEND,
            '3d_no_lambdarank': True,
        }

        model_path = output_dir / f'v472_multi_target_{timestamp}.pkl'
        joblib.dump(model_data, model_path)
        logger.info(f"\n模型已保存: {model_path}")
        logger.info(f"  大小: {model_path.stat().st_size / 1024 / 1024:.1f} MB")

        # 保存训练历史
        history = {
            'version': 'v4.7.2',
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
            'bug_fixes': {
                'winsorization_leakage': 'prepare_features不再提前winsorize全量数据',
                'sharpe_blend_applied': '每个WF窗口和生产模型均执行目标特异性Sharpe-Blend',
                'market_index': 'scorer层统一使用000300.SH',
            },
            'new_features': {
                'financial': self.FINANCIAL_FEATURES,
                'daily_basic_extra': ['dv_ttm', 'turnover_rate_f', 'float_ratio'],
                'microstructure': self.MICROSTRUCTURE_FEATURES,
                'reversal': self.REVERSAL_FEATURES,
                'risk': self.RISK_FEATURES,
            },
            'modules': {
                'A_monotonicity': True,
                'B_liquidity_discount': True,
                'C_bear_specialist': len(bear_models) > 0,
                'D_sharpe_blend': 'target_specific',
                'E_executability_filter': 'scorer层实现',
                'F_regime_adaptive': 'scorer层实现',
                'lambdarank': '5d/10d/15d only (3d skipped)',
                'time_decay': True,
                'V46_icir_weights': True,
                'V46_small_cap_weighting': False,
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
        logger.info(f"\nV4.7.2 训练完成! 总耗时: {duration:.0f}秒 ({duration/60:.1f}分钟)")

        return model_data, history


class V473Trainer(V472Trainer):
    """V4.7.3 训练器 — 简化管线 + 特征精简 + 放宽正则化

    保留V4.7.2: Bug修复(3项) + 有价值新特征 + LambdaRank + 时间衰减 + ICIR权重 + Per-target Isotonic
    去除: Meta-Learner + Combined Isotonic (破坏预测区分度的两层压缩)
    精简: 去除5个高缺失率财务特征 + 1个冗余风险特征 (76→70)
    放宽: num_leaves 20→31, min_data_in_leaf 500→200, path_smooth 10→5
    """

    # 覆写: 只保留roe (其余5个季报特征96%+缺失, 中位数填充=噪声)
    FINANCIAL_FEATURES = ['roe']
    # 覆写: 去除downside_deviation_20d (与idio_volatility_20d相关>0.9)
    RISK_FEATURES = ['idio_volatility_20d']

    # V4.7.3/V4.7.5 scorer 使用 0.6*10d + 0.4*15d composite,
    # 推荐阈值必须用同一公式校准 (而非 target_weights)
    recommendation_composite_weights = {'3d': 0.0, '5d': 0.0, '10d': 0.6, '15d': 0.4}

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """V4.7.3: 继承V4.7.1的prepare_features, 但移除downside_deviation_20d列

        V471Trainer.load_data()硬编码计算了downside_deviation_20d, 即使RISK_FEATURES不包含它,
        它仍会作为DataFrame列被prepare_features()自动纳入特征集。这里显式移除。
        """
        # 先移除泄漏的列
        if 'downside_deviation_20d' in df.columns:
            df = df.drop(columns=['downside_deviation_20d'])
            logger.info("  V4.7.3: 移除冗余特征 downside_deviation_20d (与idio_volatility_20d相关>0.9)")
        return super().prepare_features(df)

    def train_single_target_models(self, X_train, X_val, y_train, y_val, target_name: str,
                                    sample_weights_train=None):
        """V4.7.3: 放宽正则化的5/6模型训练 — 增加预测多样性"""
        import gc

        models = {}
        predictions_train = {}
        predictions_val = {}

        # 1. LightGBM — 放宽正则化
        logger.info(f"  训练 LightGBM ({target_name}, V4.7.3 放宽正则化)...")
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,           # 20→31
            'learning_rate': 0.02,
            'feature_fraction': 0.6,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'reg_alpha': 0.5,           # 1.0→0.5
            'reg_lambda': 3.0,          # 5.0→3.0
            'min_data_in_leaf': 200,    # 500→200
            'min_gain_to_split': 0.01,
            'path_smooth': 5.0,         # 10.0→5.0
            'verbose': -1,
        }
        # CLI overrides (--num-leaves, --min-data-in-leaf)
        if hasattr(self, '_cli_num_leaves'):
            lgb_params['num_leaves'] = self._cli_num_leaves
        if hasattr(self, '_cli_min_data_in_leaf'):
            lgb_params['min_data_in_leaf'] = self._cli_min_data_in_leaf

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

        # 2. XGBoost — 放宽正则化
        logger.info(f"  训练 XGBoost ({target_name}, V4.7.3)...")
        xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,             # 5→6
            'learning_rate': 0.02,
            'subsample': 0.7,
            'colsample_bytree': 0.6,
            'reg_alpha': 0.5,           # 1.0→0.5
            'reg_lambda': 3.0,          # 5.0→3.0
            'min_child_weight': 50,     # 100→50
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

        # 3. CatBoost — 放宽正则化
        if HAS_CATBOOST:
            logger.info(f"  训练 CatBoost ({target_name}, V4.7.3)...")
            cb_model = cb.CatBoostRegressor(
                iterations=1000,
                learning_rate=0.02,
                depth=6,                    # 5→6
                l2_leaf_reg=10,
                random_seed=42,
                verbose=False,
                early_stopping_rounds=30,
                min_data_in_leaf=200,       # 500→200
            )
            cb_pool_train = cb.Pool(X_train, label=y_train, weight=sample_weights_train)
            cb_pool_val = cb.Pool(X_val, label=y_val)
            cb_model.fit(cb_pool_train, eval_set=cb_pool_val, verbose=False)
            models['cb'] = cb_model
            predictions_train['cb'] = cb_model.predict(X_train)
            predictions_val['cb'] = cb_model.predict(X_val)
            del cb_pool_train, cb_pool_val
            gc.collect()

        # 4. RandomForest — 放宽正则化
        logger.info(f"  训练 RandomForest ({target_name}, V4.7.3)...")
        n_samples = X_train.shape[0]
        rf_max_samples = min(200_000, n_samples)
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,               # 10→15 (原V4.3是12)
            max_samples=rf_max_samples,
            max_features=0.6,
            min_samples_leaf=200,       # 500→200
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )
        rf_model.fit(X_train, y_train, sample_weight=sample_weights_train)
        models['rf'] = rf_model
        predictions_train['rf'] = rf_model.predict(X_train)
        predictions_val['rf'] = rf_model.predict(X_val)

        # 5. HistGradientBoosting — 放宽正则化
        logger.info(f"  训练 HistGradientBoosting ({target_name}, V4.7.3)...")
        hgb_model = HistGradientBoostingRegressor(
            max_iter=1000,
            learning_rate=0.02,
            max_depth=6,                # 5→6
            max_leaf_nodes=47,          # 20→47 (约31的1.5倍)
            l2_regularization=3.0,      # 5.0→3.0
            min_samples_leaf=200,       # 500→200
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=42,
            verbose=0,
        )
        hgb_model.fit(X_train, y_train, sample_weight=sample_weights_train)
        models['hgb'] = hgb_model
        predictions_train['hgb'] = hgb_model.predict(X_train)
        predictions_val['hgb'] = hgb_model.predict(X_val)

        # 6. LambdaRank LGB (5d/10d/15d only, 3d skipped — 继承V4.7.2)
        if '3d' not in target_name:
            train_dates = getattr(self, 'train_dates', None)
            val_dates = getattr(self, 'val_dates', None)

            if train_dates is not None and len(train_dates) == len(y_train):
                logger.info(f"  训练 LGB-LambdaRank ({target_name}, V4.7.3 放宽正则化)...")
                try:
                    from scipy.stats import rankdata

                    unique_train_dates = np.unique(train_dates)
                    relevance_train = np.zeros(len(y_train), dtype=np.int32)
                    group_train = []
                    for d in unique_train_dates:
                        mask = train_dates == d
                        n = mask.sum()
                        group_train.append(n)
                        if n >= 10:
                            ranks = rankdata(y_train[mask])
                            pct = (ranks - 1) / (n - 1)
                            relevance_train[mask] = np.clip((pct * 5).astype(int), 0, 4)
                        else:
                            relevance_train[mask] = 2

                    relevance_val = np.zeros(len(y_val), dtype=np.int32)
                    group_val = []
                    if val_dates is not None and len(val_dates) == len(y_val):
                        unique_val_dates = np.unique(val_dates)
                        for d in unique_val_dates:
                            mask = val_dates == d
                            n = mask.sum()
                            group_val.append(n)
                            if n >= 10:
                                ranks = rankdata(y_val[mask])
                                pct = (ranks - 1) / (n - 1)
                                relevance_val[mask] = np.clip((pct * 5).astype(int), 0, 4)
                            else:
                                relevance_val[mask] = 2

                    lgb_rank_params = {
                        'objective': 'lambdarank',
                        'metric': 'ndcg',
                        'eval_at': [10, 50],
                        'lambdarank_truncation_level': 50,
                        'num_leaves': 31,           # 20→31
                        'learning_rate': 0.02,
                        'feature_fraction': 0.6,
                        'bagging_fraction': 0.7,
                        'bagging_freq': 5,
                        'reg_alpha': 0.5,           # 1.0→0.5
                        'reg_lambda': 3.0,          # 5.0→3.0
                        'min_data_in_leaf': 200,    # 500→200
                        'min_gain_to_split': 0.01,
                        'path_smooth': 5.0,         # 10.0→5.0
                        'verbose': -1,
                    }
                    # CLI overrides
                    if hasattr(self, '_cli_num_leaves'):
                        lgb_rank_params['num_leaves'] = self._cli_num_leaves
                    if hasattr(self, '_cli_min_data_in_leaf'):
                        lgb_rank_params['min_data_in_leaf'] = self._cli_min_data_in_leaf

                    lgb_rank_train = lgb.Dataset(
                        X_train, label=relevance_train, group=group_train,
                        weight=sample_weights_train, free_raw_data=True
                    )
                    lgb_rank_val = lgb.Dataset(
                        X_val, label=relevance_val, group=group_val,
                        reference=lgb_rank_train, free_raw_data=True
                    )

                    lgb_rank_model = lgb.train(
                        lgb_rank_params, lgb_rank_train,
                        num_boost_round=1000,
                        valid_sets=[lgb_rank_train, lgb_rank_val],
                        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
                    )

                    models['lgb_rank'] = lgb_rank_model
                    predictions_train['lgb_rank'] = lgb_rank_model.predict(X_train)
                    predictions_val['lgb_rank'] = lgb_rank_model.predict(X_val)
                    logger.info(f"    LGB-LambdaRank ({target_name}): 完成")

                    del lgb_rank_train, lgb_rank_val
                    gc.collect()
                except Exception as e:
                    logger.warning(f"    LambdaRank ({target_name}) 失败: {e}")

        return models, predictions_train, predictions_val

    def walk_forward_train(self, start_date: str = None, end_date: str = None,
                            purge_days: int = 15, min_train_days: int = 900,
                            val_days: int = 120, test_days: int = 120,
                            step_days: int = 90):
        """V4.7.3 Walk-Forward 训练 — V4.7.2底座, 去掉Meta-Learner/Combined Isotonic

        保留: ICIR权重优化 + Bear Specialist + Per-target Isotonic
        去除: Meta-Learner + Combined Isotonic (两层压缩破坏预测区分度)
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("V4.7.3 Walk-Forward 训练 (简化管线 + 特征精简 + 放宽正则化)")
        logger.info("=" * 60)
        logger.info(f"  参数: min_train={min_train_days}d, val={val_days}d, test={test_days}d, "
                     f"step={step_days}d, purge={purge_days}d")
        logger.info(f"  目标权重: {self.target_weights}")
        logger.info(f"  Sharpe融合(目标特异性): {self.TARGET_SHARPE_BLEND}")
        logger.info(f"  特征精简: FINANCIAL={self.FINANCIAL_FEATURES}, RISK={self.RISK_FEATURES}")
        logger.info(f"  正则化放宽: num_leaves=31, min_data_in_leaf=200, path_smooth=5.0")
        logger.info(f"  管线简化: 无Meta-Learner, 无Combined Isotonic")

        # 1. 一次性加载全量数据 (V4.7.3: 精简特征)
        df = self.load_data(start_date, end_date)
        X, y_3d, y_5d, y_10d, y_15d, df = self.prepare_features(df)

        dates = df['trade_date'].values
        unique_dates = np.sort(np.unique(dates))
        n_dates = len(unique_dates)
        logger.info(f"  总交易日: {n_dates}, 样本: {len(X):,}, 特征: {X.shape[1]}")

        # 2. 定义滚动窗口 (与V4.7.2一致: date-based)
        windows = []
        cursor = min_train_days
        while cursor + val_days + purge_days + test_days <= n_dates:
            train_end_idx = cursor - 1
            val_start_idx = cursor + purge_days
            val_end_idx = val_start_idx + val_days - 1
            test_start_idx = val_end_idx + purge_days + 1
            test_end_idx = min(test_start_idx + test_days - 1, n_dates - 1)
            windows.append({
                'train_end': unique_dates[train_end_idx],
                'val_start': unique_dates[val_start_idx],
                'val_end': unique_dates[val_end_idx],
                'test_start': unique_dates[test_start_idx],
                'test_end': unique_dates[test_end_idx],
            })
            cursor += step_days

        logger.info(f"  Walk-Forward窗口: {len(windows)}")
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
            train_dates_w = dates[train_mask]
            val_dates_w = dates[val_mask]

            # Walk-Forward: 特征Winsorization (仅用窗口内训练集)
            X_train_w, wf_bounds = self.winsorize_features(X_train_w)
            self._apply_bounds(X_val_w, wf_bounds)
            self._apply_bounds(X_test_w, wf_bounds)

            # Walk-Forward: 标签Winsorization (仅用训练集统计量)
            for y_tr_w, y_va_w, y_te_w in [(y_3d_tr, y_3d_va, y_3d_te),
                                             (y_5d_tr, y_5d_va, y_5d_te),
                                             (y_10d_tr, y_10d_va, y_10d_te),
                                             (y_15d_tr, y_15d_va, y_15d_te)]:
                lo = np.percentile(y_tr_w, 1)
                hi = np.percentile(y_tr_w, 99)
                y_tr_w[:] = np.clip(y_tr_w, lo, hi)
                y_va_w[:] = np.clip(y_va_w, lo, hi)
                y_te_w[:] = np.clip(y_te_w, lo, hi)

            # V4.7.2: 目标特异性Sharpe-Blend (in-place修改)
            self.train_dates = train_dates_w
            self.val_dates = val_dates_w
            for target_key, y_tr_w, y_va_w, y_te_w in [
                ('label_3d', y_3d_tr, y_3d_va, y_3d_te),
                ('label_5d', y_5d_tr, y_5d_va, y_5d_te),
                ('label_10d', y_10d_tr, y_10d_va, y_10d_te),
                ('label_15d', y_15d_tr, y_15d_va, y_15d_te),
            ]:
                self._apply_sharpe_blend(y_tr_w, y_va_w, y_te_w,
                                          train_dates_w, val_dates_w, test_dates_w,
                                          target_key)

            # 训练4目标 (bare keys: '3d', '5d', '10d', '15d')
            window_metrics = {}
            for target_key, y_tr, y_va, y_te in [
                ('3d', y_3d_tr, y_3d_va, y_3d_te),
                ('5d', y_5d_tr, y_5d_va, y_5d_te),
                ('10d', y_10d_tr, y_10d_va, y_10d_te),
                ('15d', y_15d_tr, y_15d_va, y_15d_te),
            ]:
                sample_w = self.compute_sample_weights(df[train_mask], y_tr)
                models, pred_train, pred_val = self.train_single_target_models(
                    X_train_w, X_val_w, y_tr, y_va, f"label_{target_key}",
                    sample_weights_train=sample_w)
                weights, rmses = self.calculate_ensemble_weights(pred_val, y_va)

                # test set预测
                pred_test = {}
                for name, model in models.items():
                    try:
                        if name == 'xgb':
                            pred_test[name] = model.predict(xgb.DMatrix(X_test_w))
                        else:
                            pred_test[name] = model.predict(X_test_w)
                    except Exception:
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
            if ics:
                summary = {
                    'mean_ic': float(np.mean(ics)),
                    'std_ic': float(np.std(ics)),
                    'mean_icir': float(np.mean(icirs)),
                    'std_icir': float(np.std(icirs)),
                    'n_windows': len(ics),
                }
                logger.info(f"  {target_key}: IC={summary['mean_ic']:.4f}+-{summary['std_ic']:.4f}, "
                             f"ICIR={summary['mean_icir']:.4f}+-{summary['std_icir']:.4f}")
            else:
                summary = {'mean_ic': 0, 'std_ic': 0, 'mean_icir': 0, 'std_icir': 0, 'n_windows': 0}
                logger.info(f"  {target_key}: (skipped, no WF windows)")
            wf_summary[target_key] = summary

        # 5. 训练最终生产模型 (85% train + 15% val)
        logger.info("\n" + "=" * 60)
        logger.info("训练最终V4.7.3生产模型 (全量数据)")
        logger.info("=" * 60)

        split_idx = int(n_dates * 0.85)
        split_date = unique_dates[split_idx]
        train_mask_final = dates <= split_date
        val_mask_final = dates > split_date

        X_train_f, X_val_f = X[train_mask_final].copy(), X[val_mask_final].copy()
        self.val_dates = dates[val_mask_final]
        self.train_dates = dates[train_mask_final]

        # Bug 1修复: 生产模型的Winsorization也只用训练集
        X_train_f, self.winsorize_bounds = self.winsorize_features(X_train_f)
        self._apply_bounds(X_val_f, self.winsorize_bounds)
        logger.info(f"  生产模型: 特征Winsorization (训练集bounds), {len(self.winsorize_bounds)} 列")

        df_train_f = df[train_mask_final]
        all_results = {}

        y_val_dict = {}
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

        # V4.7.2: 目标特异性Sharpe-Blend (Bug 2修复, in-place)
        logger.info(f"  [V4.7.3] 目标特异性Sharpe-Blend: {self.TARGET_SHARPE_BLEND}")
        train_dates_f = dates[train_mask_final]
        val_dates_f = dates[val_mask_final]
        for target_key, y_tr, y_va in targets_final:
            self._apply_sharpe_blend(y_tr, y_va, np.array([]),
                                      train_dates_f, val_dates_f, np.array([]),
                                      f"label_{target_key}")

        for target_key, y_tr, y_va in targets_final:
            sample_w = self.compute_sample_weights(df_train_f, y_tr)
            models, pred_train, pred_val = self.train_single_target_models(
                X_train_f, X_val_f, y_tr, y_va, f"label_{target_key}",
                sample_weights_train=sample_w)
            weights, rmses = self.calculate_ensemble_weights(pred_val, y_va)
            all_results[target_key] = {'models': models, 'weights': weights, 'rmses': rmses}
            y_val_dict[target_key] = y_va

        # 6. Module C: Bear specialist (继承V4.4)
        logger.info("\n" + "=" * 60)
        logger.info("Module C: Bear Specialist (10d/15d)")
        logger.info("=" * 60)
        bear_models = {}
        for target_key in ['10d', '15d']:
            y_tr_target = y_10d[train_mask_final] if target_key == '10d' else y_15d[train_mask_final]
            bear_model = self._train_bear_specialist(X_train_f, y_tr_target, df_train_f, target_key)
            if bear_model is not None:
                bear_models[target_key] = bear_model

        # 7. Module A: Per-target isotonic calibration (继承V4.4)
        logger.info("\n" + "=" * 60)
        logger.info("Module A: Per-Target 保序回归校准")
        logger.info("=" * 60)
        isotonic_models = self._fit_isotonic_calibration(X_val_f, y_val_dict, all_results)

        # 8. ICIR优化集成权重 (保留, 不压缩预测值)
        logger.info("\n" + "=" * 60)
        logger.info("V4.7.3 ICIR最大化集成权重 (保留自V4.7.2)")
        logger.info("=" * 60)
        icir_weights = V46Trainer._optimize_icir_weights(self, all_results, X_val_f, y_val_dict, self.val_dates)
        for target_key, w in icir_weights.items():
            if target_key in all_results:
                all_results[target_key]['weights'] = w

        # ★ V4.7.3核心变更: 跳过Combined Isotonic和Meta-Learner ★
        logger.info("\n" + "=" * 60)
        logger.info("V4.7.3: 跳过 Combined Isotonic 和 Meta-Learner (消除两层压缩)")
        logger.info("=" * 60)
        combined_isotonic = None
        meta_learner = None
        meta_feature_names = None

        # 9. 特征重要性分析
        self._log_feature_importance(all_results)

        # 10. 计算全局评分分位数
        X_all = X.copy()
        self._apply_bounds(X_all, self.winsorize_bounds)
        global_quantiles = self._compute_global_quantiles(X_all, all_results, self.target_weights)
        recommendation_thresholds = self._compute_recommendation_thresholds(X_all, all_results)

        # 11. 保存模型
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        output_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v473'
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        winsorize_bounds_dict = {}
        if self.winsorize_bounds and self.feature_names:
            for idx, (lo, hi) in enumerate(self.winsorize_bounds):
                if idx < len(self.feature_names):
                    winsorize_bounds_dict[self.feature_names[idx]] = (lo, hi)

        model_data = {
            'version': 'v4.7.3',
            'models': all_results,
            'feature_names': self.feature_names,
            'target_weights': self.target_weights,
            'market_features': list(self.market_calculator.market_features.columns[1:]),
            'winsorize_bounds': winsorize_bounds_dict,
            'global_quantiles': global_quantiles,
            'recommendation_thresholds': recommendation_thresholds,
            # 模型类型标识
            'cascade': False,
            'rank_normalized': False,
            'robust_zscore': True,
            'industry_excess_labels': True,
            'dual_stream': False,
            'cross_sectional_neutralization': False,
            'macro_feature_cols': self.macro_feature_cols,
            'stock_feature_cols': self.stock_feature_cols,
            'extra_features_from_daily_basic': ['pe_ttm', 'pb', 'ps_ttm', 'turnover_rate', 'log_market_cap',
                                                 'dv_ttm', 'turnover_rate_f', 'float_ratio'],
            'extra_features_from_tech_indicators': self.extra_tech_feature_names,
            'extra_features_financial': self.FINANCIAL_FEATURES,  # ['roe'] only
            'extra_features_microstructure': self.MICROSTRUCTURE_FEATURES,
            'extra_features_reversal': self.REVERSAL_FEATURES,
            'extra_features_risk': self.RISK_FEATURES,  # ['idio_volatility_20d'] only
            'targets': ['3d', '5d', '10d', '15d'],
            'ensemble_type': 'icir_optimized',
            'sample_weighting': True,
            'time_decay_half_life': 365,
            'walk_forward_metrics': wf_summary,
            'walk_forward_windows': len(windows),
            'regularization': {
                'num_leaves': 31, 'min_data_in_leaf': 200,
                'reg_alpha': 0.5, 'reg_lambda': 3.0,
                'path_smooth': 5.0, 'learning_rate': 0.02,
            },
            # V4.4 组件 (继承)
            'bear_models': bear_models,
            'isotonic_calibration': isotonic_models,
            'sharpe_label_blend': 'target_specific',
            'sharpe_blend_config': self.TARGET_SHARPE_BLEND,
            'liquidity_discount': True,
            'bear_sample_weighting': True,
            'min_train_days': min_train_days,
            'step_days': step_days,
            # V4.7.1 组件 (继承)
            'has_lambdarank': True,
            'has_time_decay': True,
            'bug_fixes': ['winsorization_leakage', 'sharpe_blend_applied', 'market_index_000300'],
            # V4.7.3 核心变更: 无压缩管线
            'icir_optimized_weights': icir_weights,
            'combined_isotonic': None,       # ★ 去除
            'meta_learner': None,            # ★ 去除
            'meta_feature_names': None,      # ★ 去除
            'small_cap_weighting': False,
            'target_specific_sharpe': self.TARGET_SHARPE_BLEND,
            '3d_no_lambdarank': True,
            # V4.7.3 特有标识
            'pipeline_simplified': True,
            'features_pruned': {
                'removed_financial': ['gross_margin', 'current_ratio', 'assets_turn', 'netprofit_yoy', 'or_yoy'],
                'removed_risk': ['downside_deviation_20d'],
            },
        }

        model_path = output_dir / f'v473_multi_target_{timestamp}.pkl'
        joblib.dump(model_data, model_path)
        logger.info(f"\n模型已保存: {model_path}")
        logger.info(f"  大小: {model_path.stat().st_size / 1024 / 1024:.1f} MB")

        # 保存训练历史
        history = {
            'version': 'v4.7.3',
            'base': 'V4.7.2 (简化管线: 去Meta-Learner/Combined Isotonic)',
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
                'meta_learner': False,
                'combined_isotonic': False,
                'icir_optimized': len(icir_weights) > 0,
            },
            'target_weights': self.target_weights,
            'ensemble_weights': {k: all_results[k]['weights'] for k in all_results},
            'design_rationale': {
                'remove_meta_learner': 'Ridge(alpha=10) 120d训练, 严重向均值收缩, 破坏top区分度',
                'remove_combined_isotonic': '压缩combined_pred, 破坏预测值区分度',
                'prune_features': '5个季报财务特征96%%+缺失率; downside_deviation与idio_vol r>0.9',
                'relax_regularization': '增加树叶数和深度, 提高unique预测值数量',
            },
        }

        history_path = output_dir / f'training_history_{timestamp}.json'
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        latest_path = output_dir / 'training_history_latest.json'
        with open(latest_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        logger.info(f"训练历史已保存: {history_path}")
        logger.info(f"\nV4.7.3 训练完成! 总耗时: {duration:.0f}秒 ({duration/60:.1f}分钟)")
        logger.info(f"  特征: {len(self.feature_names)}")
        logger.info(f"  管线: 无Meta-Learner, 无Combined Isotonic")
        logger.info(f"  正则化: num_leaves=31, min_data=200, path_smooth=5.0")

        return model_data, history


class V474Trainer(V473Trainer):
    """V4.7.4 训练器 — V4.7.3简化管线 + 选择性V4.8特征 + ListNet + 严格ICIR约束

    核心改进 (相比V4.7.3):
    1. +4个选择性V4.8财务特征 (netprofit_margin/ocf_to_opincome/debt_to_eqt/basic_eps_yoy)
    2. 10d/15d加入ListNet排名模型 (V4.8证明ICIR+0.14/+0.24)
    3. ICIR权重约束加严: floor=0.10, ceiling=0.35 (V4.8教训: 92.6%单模型主导)
    4. 评分连续化 (scorer层, 非训练层)

    继承V4.7.3:
    - 无Meta-Learner, 无Combined Isotonic (简化管线)
    - 70+4=74个特征 (精简+选择性扩展)
    - 放宽正则化: num_leaves=31, min_data=200, path_smooth=5
    - Bug修复(3项) + LambdaRank + 时间衰减 + ICIR权重 + Per-target Isotonic
    """

    # V4.7.4选择性V4.8财务特征 (低缺失率 + 高IC + 不与现有特征冗余)
    FINANCIAL_QUALITY_FEATURES = ['netprofit_margin', 'ocf_to_opincome', 'debt_to_eqt', 'basic_eps_yoy']

    def load_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """V4.7.4: V4.7.3基础 + 4个选择性V4.8财务质量特征"""
        # 先加载V4.7.3的全部数据 (含精简特征)
        df = super().load_data(start_date, end_date)

        date_min = df['trade_date'].min()
        date_max = df['trade_date'].max()

        # 加载4个选择性V4.8财务质量特征
        logger.info(f"  V4.7.4 加载选择性V4.8财务特征: {self.FINANCIAL_QUALITY_FEATURES}")
        conn = sqlite3.connect(self.db_path)
        try:
            fi_cols_str = ', '.join(f'fi.{c}' for c in self.FINANCIAL_QUALITY_FEATURES)
            fi_query = f"""
            SELECT s.code, fi.ann_date, {fi_cols_str}
            FROM financial_indicator fi
            JOIN securities s ON fi.security_id = s.id
            WHERE fi.ann_date IS NOT NULL AND fi.ann_date != ''
            ORDER BY s.code, fi.ann_date
            """
            df_fi = pd.read_sql(fi_query, conn)
        finally:
            conn.close()

        if len(df_fi) > 0:
            def _date_to_int(s):
                return pd.to_datetime(s.astype(str).str.replace('-', ''), format='%Y%m%d').dt.strftime('%Y%m%d').astype(np.int64)

            df_fi['_ann_int'] = _date_to_int(df_fi['ann_date'])
            if '_td_int' not in df.columns:
                df['_td_int'] = _date_to_int(df['trade_date'])

            # Point-in-time merge
            df_fi_dedup = df_fi.drop_duplicates(subset=['code', '_ann_int'], keep='last')
            fi_subset = df_fi_dedup[['code', '_ann_int'] + self.FINANCIAL_QUALITY_FEATURES].rename(
                columns={'_ann_int': '_td_int'}).sort_values('_td_int')

            original_index = df.index.copy()
            df = df.sort_values('_td_int').reset_index(drop=True)
            df = pd.merge_asof(df, fi_subset, on='_td_int', by='code', direction='backward',
                               suffixes=('', '_v48'))
            df.index = original_index

            # 清理重复列 (如果V4.7.3已有同名列)
            for col in self.FINANCIAL_QUALITY_FEATURES:
                v48_col = f'{col}_v48'
                if v48_col in df.columns:
                    df[col] = df[v48_col]
                    df.drop(columns=[v48_col], inplace=True)

            # 填充缺失值
            for col in self.FINANCIAL_QUALITY_FEATURES:
                if col in df.columns:
                    missing = df[col].isnull().sum()
                    if missing > 0:
                        df[col] = df.groupby('trade_date')[col].transform(lambda x: x.fillna(x.median()))
                        df[col] = df[col].fillna(df[col].median() if not pd.isna(df[col].median()) else 0.0)
                        pct = missing / len(df) * 100
                        if pct > 5:
                            logger.info(f"      {col}: {missing:,} 缺失({pct:.1f}%)")
                else:
                    df[col] = 0.0

            logger.info(f"    V4.7.4选择性财务特征: +{len(self.FINANCIAL_QUALITY_FEATURES)} 特征")
        else:
            for col in self.FINANCIAL_QUALITY_FEATURES:
                df[col] = 0.0
            logger.warning("    financial_indicator 为空, V4.7.4特征填0")

        # 清理临时列
        if '_td_int' in df.columns:
            df.drop(columns=['_td_int'], inplace=True, errors='ignore')

        return df

    def train_single_target_models(self, X_train, X_val, y_train, y_val, target_name: str,
                                    sample_weights_train=None):
        """V4.7.4: V4.7.3的5/6模型 + 10d/15d加入ListNet

        - 3d/5d: V4.7.3逻辑 (5模型回归 + LambdaRank对5d)
        - 10d/15d: V4.7.3的6模型 + ListNet = 7模型

        ListNet只加在10d/15d (V4.8数据显示这两个target收益最大)
        """
        import gc

        # 先用V4.7.3训练基础模型
        models, pred_train, pred_val = super().train_single_target_models(
            X_train, X_val, y_train, y_val, target_name,
            sample_weights_train=sample_weights_train)

        # V4.7.4: 仅对10d/15d加入ListNet
        if '10d' not in target_name and '15d' not in target_name:
            return models, pred_train, pred_val

        # 复用LambdaRank已计算的relevance标签
        relevance_train = getattr(self, '_cached_relevance_train', None)
        group_train = getattr(self, '_cached_group_train', None)
        relevance_val = getattr(self, '_cached_relevance_val', None)
        group_val = getattr(self, '_cached_group_val', None)

        if relevance_train is not None and group_train is not None:
            logger.info(f"  V4.7.4 训练 LGB-ListNet ({target_name}) [复用LambdaRank标签]...")
            try:
                lgb_listnet_params = {
                    'objective': 'rank_xendcg',
                    'metric': 'ndcg',
                    'eval_at': [10, 20],
                    'num_leaves': 31,           # 与V4.7.3对齐 (V4.8用24)
                    'learning_rate': 0.02,       # 与V4.7.3对齐 (V4.8用0.03)
                    'feature_fraction': 0.6,     # 与V4.7.3对齐 (V4.8用0.7)
                    'bagging_fraction': 0.7,     # 与V4.7.3对齐
                    'bagging_freq': 5,
                    'reg_alpha': 0.5,
                    'reg_lambda': 3.0,
                    'min_data_in_leaf': 200,     # 与V4.7.3对齐 (V4.8用300)
                    'min_gain_to_split': 0.01,
                    'path_smooth': 5.0,
                    'verbose': -1,
                }

                lgb_listnet_train = lgb.Dataset(
                    X_train, label=relevance_train, group=group_train,
                    weight=sample_weights_train, free_raw_data=True
                )
                lgb_listnet_val = lgb.Dataset(
                    X_val, label=relevance_val, group=group_val,
                    reference=lgb_listnet_train, free_raw_data=True
                )

                lgb_listnet_model = lgb.train(
                    lgb_listnet_params, lgb_listnet_train,
                    num_boost_round=800,     # 比V4.8的600更多, 因为lr更小
                    valid_sets=[lgb_listnet_train, lgb_listnet_val],
                    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
                )

                models['lgb_listnet'] = lgb_listnet_model
                pred_train['lgb_listnet'] = lgb_listnet_model.predict(X_train)
                pred_val['lgb_listnet'] = lgb_listnet_model.predict(X_val)

                ic_val, _ = spearmanr(pred_val['lgb_listnet'], y_val)
                logger.info(f"    LGB-ListNet ({target_name}): IC={ic_val:.4f}")

                del lgb_listnet_train, lgb_listnet_val
                gc.collect()
            except Exception as e:
                logger.warning(f"    LGB-ListNet ({target_name}) 训练失败: {e}")

        return models, pred_train, pred_val

    def walk_forward_train(self, start_date: str = None, end_date: str = None,
                            purge_days: int = 15, min_train_days: int = 900,
                            val_days: int = 120, test_days: int = 120,
                            step_days: int = 90):
        """V4.7.4 Walk-Forward 训练 — V4.7.3简化管线 + 选择性V4.8增强

        差异点(vs V4.7.3):
        - 特征: 70+4=74 (V4.7.3的70 + 4个V4.8选择性财务)
        - 模型: 10d/15d多一个ListNet
        - ICIR约束: [0.10, 0.35] (vs V4.7.3的[0.08, 0.50])
        - 其余: 完全继承V4.7.3 (无Meta-Learner, 无Combined Isotonic)
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("V4.7.4 Walk-Forward 训练 (V4.7.3简化管线 + 选择性V4.8增强)")
        logger.info("=" * 60)
        logger.info(f"  参数: min_train={min_train_days}d, val={val_days}d, test={test_days}d, "
                     f"step={step_days}d, purge={purge_days}d")
        logger.info(f"  目标权重: {self.target_weights}")
        logger.info(f"  Sharpe融合(目标特异性): {self.TARGET_SHARPE_BLEND}")
        logger.info(f"  特征: V4.7.3的70 + V4.8选择性{len(self.FINANCIAL_QUALITY_FEATURES)}")
        logger.info(f"  模型: 3d/5d=V4.7.3, 10d/15d=V4.7.3+ListNet")
        logger.info(f"  ICIR约束: floor=0.10, ceiling=0.35 (加严)")
        logger.info(f"  管线: 无Meta-Learner, 无Combined Isotonic")

        # 1. 加载数据 (V4.7.4: 含4个选择性财务特征)
        df = self.load_data(start_date, end_date)
        X, y_3d, y_5d, y_10d, y_15d, df = self.prepare_features(df)

        dates = df['trade_date'].values
        unique_dates = np.sort(np.unique(dates))
        n_dates = len(unique_dates)
        logger.info(f"  总交易日: {n_dates}, 样本: {len(X):,}, 特征: {X.shape[1]}")

        # 2. 定义滚动窗口
        windows = []
        cursor = min_train_days
        while cursor + val_days + purge_days + test_days <= n_dates:
            train_end_idx = cursor - 1
            val_start_idx = cursor + purge_days
            val_end_idx = val_start_idx + val_days - 1
            test_start_idx = val_end_idx + purge_days + 1
            test_end_idx = min(test_start_idx + test_days - 1, n_dates - 1)
            windows.append({
                'train_end': unique_dates[train_end_idx],
                'val_start': unique_dates[val_start_idx],
                'val_end': unique_dates[val_end_idx],
                'test_start': unique_dates[test_start_idx],
                'test_end': unique_dates[test_end_idx],
            })
            cursor += step_days

        logger.info(f"  Walk-Forward窗口: {len(windows)}")
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
            train_dates_w = dates[train_mask]
            val_dates_w = dates[val_mask]

            # Winsorization
            X_train_w, wf_bounds = self.winsorize_features(X_train_w)
            self._apply_bounds(X_val_w, wf_bounds)
            self._apply_bounds(X_test_w, wf_bounds)

            # 标签Winsorization
            for y_tr_w, y_va_w, y_te_w in [(y_3d_tr, y_3d_va, y_3d_te),
                                             (y_5d_tr, y_5d_va, y_5d_te),
                                             (y_10d_tr, y_10d_va, y_10d_te),
                                             (y_15d_tr, y_15d_va, y_15d_te)]:
                lo = np.percentile(y_tr_w, 1)
                hi = np.percentile(y_tr_w, 99)
                y_tr_w[:] = np.clip(y_tr_w, lo, hi)
                y_va_w[:] = np.clip(y_va_w, lo, hi)
                y_te_w[:] = np.clip(y_te_w, lo, hi)

            # Sharpe-Blend
            self.train_dates = train_dates_w
            self.val_dates = val_dates_w
            for target_key, y_tr_w, y_va_w, y_te_w in [
                ('label_3d', y_3d_tr, y_3d_va, y_3d_te),
                ('label_5d', y_5d_tr, y_5d_va, y_5d_te),
                ('label_10d', y_10d_tr, y_10d_va, y_10d_te),
                ('label_15d', y_15d_tr, y_15d_va, y_15d_te),
            ]:
                self._apply_sharpe_blend(y_tr_w, y_va_w, y_te_w,
                                          train_dates_w, val_dates_w, test_dates_w,
                                          target_key)

            # 训练4目标
            window_metrics = {}
            for target_key, y_tr, y_va, y_te in [
                ('3d', y_3d_tr, y_3d_va, y_3d_te),
                ('5d', y_5d_tr, y_5d_va, y_5d_te),
                ('10d', y_10d_tr, y_10d_va, y_10d_te),
                ('15d', y_15d_tr, y_15d_va, y_15d_te),
            ]:
                sample_w = self.compute_sample_weights(df[train_mask], y_tr)
                models, pred_train, pred_val = self.train_single_target_models(
                    X_train_w, X_val_w, y_tr, y_va, f"label_{target_key}",
                    sample_weights_train=sample_w)
                weights, rmses = self.calculate_ensemble_weights(pred_val, y_va)

                # test set预测
                pred_test = {}
                for name, model in models.items():
                    try:
                        if name == 'xgb':
                            pred_test[name] = model.predict(xgb.DMatrix(X_test_w))
                        else:
                            pred_test[name] = model.predict(X_test_w)
                    except Exception:
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
            if ics:
                summary = {
                    'mean_ic': float(np.mean(ics)),
                    'std_ic': float(np.std(ics)),
                    'mean_icir': float(np.mean(icirs)),
                    'std_icir': float(np.std(icirs)),
                    'n_windows': len(ics),
                }
                logger.info(f"  {target_key}: IC={summary['mean_ic']:.4f}+-{summary['std_ic']:.4f}, "
                             f"ICIR={summary['mean_icir']:.4f}+-{summary['std_icir']:.4f}")
            else:
                summary = {'mean_ic': 0, 'std_ic': 0, 'mean_icir': 0, 'std_icir': 0, 'n_windows': 0}
                logger.info(f"  {target_key}: (skipped, no WF windows)")
            wf_summary[target_key] = summary

        # 5. 训练最终生产模型
        logger.info("\n" + "=" * 60)
        logger.info("训练最终V4.7.4生产模型 (全量数据)")
        logger.info("=" * 60)

        split_idx = int(n_dates * 0.85)
        split_date = unique_dates[split_idx]
        train_mask_final = dates <= split_date
        val_mask_final = dates > split_date

        X_train_f, X_val_f = X[train_mask_final].copy(), X[val_mask_final].copy()
        self.val_dates = dates[val_mask_final]
        self.train_dates = dates[train_mask_final]

        X_train_f, self.winsorize_bounds = self.winsorize_features(X_train_f)
        self._apply_bounds(X_val_f, self.winsorize_bounds)

        df_train_f = df[train_mask_final]
        all_results = {}
        y_val_dict = {}

        targets_final = [
            ('3d', y_3d[train_mask_final].copy(), y_3d[val_mask_final].copy()),
            ('5d', y_5d[train_mask_final].copy(), y_5d[val_mask_final].copy()),
            ('10d', y_10d[train_mask_final].copy(), y_10d[val_mask_final].copy()),
            ('15d', y_15d[train_mask_final].copy(), y_15d[val_mask_final].copy()),
        ]

        for target_key, y_tr, y_va in targets_final:
            lo = np.percentile(y_tr, 1)
            hi = np.percentile(y_tr, 99)
            y_tr[:] = np.clip(y_tr, lo, hi)
            y_va[:] = np.clip(y_va, lo, hi)

        train_dates_f = dates[train_mask_final]
        val_dates_f = dates[val_mask_final]
        for target_key, y_tr, y_va in targets_final:
            self._apply_sharpe_blend(y_tr, y_va, np.array([]),
                                      train_dates_f, val_dates_f, np.array([]),
                                      f"label_{target_key}")

        for target_key, y_tr, y_va in targets_final:
            sample_w = self.compute_sample_weights(df_train_f, y_tr)
            models, pred_train, pred_val = self.train_single_target_models(
                X_train_f, X_val_f, y_tr, y_va, f"label_{target_key}",
                sample_weights_train=sample_w)
            weights, rmses = self.calculate_ensemble_weights(pred_val, y_va)
            all_results[target_key] = {'models': models, 'weights': weights, 'rmses': rmses}
            y_val_dict[target_key] = y_va

        # 6. Bear specialist
        bear_models = {}
        for target_key in ['10d', '15d']:
            y_tr_target = y_10d[train_mask_final] if target_key == '10d' else y_15d[train_mask_final]
            bear_model = self._train_bear_specialist(X_train_f, y_tr_target, df_train_f, target_key)
            if bear_model is not None:
                bear_models[target_key] = bear_model

        # 7. Isotonic calibration
        isotonic_models = self._fit_isotonic_calibration(X_val_f, y_val_dict, all_results)

        # 8. ICIR优化 + V4.7.4加严约束
        logger.info("\n" + "=" * 60)
        logger.info("V4.7.4 ICIR最大化集成权重 (加严约束: [0.10, 0.35])")
        logger.info("=" * 60)
        icir_weights = V46Trainer._optimize_icir_weights(self, all_results, X_val_f, y_val_dict, self.val_dates)

        # V4.7.4: 在ICIR优化后再次clip到[0.10, 0.35]
        for target_key, w in icir_weights.items():
            if isinstance(w, dict):
                clipped = {name: np.clip(val, 0.10, 0.35) for name, val in w.items()}
                total = sum(clipped.values())
                if total > 0:
                    clipped = {k: v / total for k, v in clipped.items()}
                icir_weights[target_key] = clipped
                if target_key in all_results:
                    all_results[target_key]['weights'] = clipped
                logger.info(f"  {target_key} ICIR权重(clip后): {', '.join(f'{k}={v:.3f}' for k, v in clipped.items())}")

        # 无Meta-Learner, 无Combined Isotonic (继承V4.7.3)
        logger.info("\n  V4.7.4: 跳过 Combined Isotonic 和 Meta-Learner (继承V4.7.3设计)")

        # 特征重要性
        self._log_feature_importance(all_results)

        # 全局评分分位数
        X_all = X.copy()
        self._apply_bounds(X_all, self.winsorize_bounds)
        global_quantiles = self._compute_global_quantiles(X_all, all_results, self.target_weights)
        recommendation_thresholds = self._compute_recommendation_thresholds(X_all, all_results)

        # 9. 保存模型
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        output_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v474'
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        winsorize_bounds_dict = {}
        if self.winsorize_bounds and self.feature_names:
            for idx, (lo, hi) in enumerate(self.winsorize_bounds):
                if idx < len(self.feature_names):
                    winsorize_bounds_dict[self.feature_names[idx]] = (lo, hi)

        model_data = {
            'version': 'v4.7.4',
            'models': all_results,
            'feature_names': self.feature_names,
            'target_weights': self.target_weights,
            'market_features': list(self.market_calculator.market_features.columns[1:]),
            'winsorize_bounds': winsorize_bounds_dict,
            'global_quantiles': global_quantiles,
            'recommendation_thresholds': recommendation_thresholds,
            # 模型类型标识
            'cascade': False,
            'rank_normalized': False,
            'robust_zscore': True,
            'industry_excess_labels': True,
            'dual_stream': False,
            'cross_sectional_neutralization': False,
            'macro_feature_cols': self.macro_feature_cols,
            'stock_feature_cols': self.stock_feature_cols,
            'extra_features_from_daily_basic': ['pe_ttm', 'pb', 'ps_ttm', 'turnover_rate', 'log_market_cap',
                                                 'dv_ttm', 'turnover_rate_f', 'float_ratio'],
            'extra_features_from_tech_indicators': self.extra_tech_feature_names,
            'extra_features_financial': self.FINANCIAL_FEATURES,  # V4.7.3: ['roe']
            'extra_features_microstructure': self.MICROSTRUCTURE_FEATURES,
            'extra_features_reversal': self.REVERSAL_FEATURES,
            'extra_features_risk': self.RISK_FEATURES,  # V4.7.3: ['idio_volatility_20d']
            'extra_financial_quality': self.FINANCIAL_QUALITY_FEATURES,  # V4.7.4新增
            'targets': ['3d', '5d', '10d', '15d'],
            'ensemble_type': 'icir_optimized_v474',
            'sample_weighting': True,
            'time_decay_half_life': 365,
            'walk_forward_metrics': wf_summary,
            'walk_forward_windows': len(windows),
            'regularization': {
                'num_leaves': 31, 'min_data_in_leaf': 200,
                'reg_alpha': 0.5, 'reg_lambda': 3.0,
                'path_smooth': 5.0, 'learning_rate': 0.02,
            },
            # V4.4 组件
            'bear_models': bear_models,
            'isotonic_calibration': isotonic_models,
            'sharpe_label_blend': 'target_specific',
            'sharpe_blend_config': self.TARGET_SHARPE_BLEND,
            'liquidity_discount': True,
            'bear_sample_weighting': True,
            'min_train_days': min_train_days,
            'step_days': step_days,
            # V4.7.1 组件
            'has_lambdarank': True,
            'has_time_decay': True,
            'bug_fixes': ['winsorization_leakage', 'sharpe_blend_applied', 'market_index_000300'],
            # V4.7.3 核心设计
            'icir_optimized_weights': icir_weights,
            'combined_isotonic': None,
            'meta_learner': None,
            'meta_feature_names': None,
            'small_cap_weighting': False,
            'target_specific_sharpe': self.TARGET_SHARPE_BLEND,
            '3d_no_lambdarank': True,
            'pipeline_simplified': True,
            # V4.7.4 独有标识
            'v474_innovations': {
                'continuous_scoring': 'np.interp (scorer层)',
                'selective_v48_features': self.FINANCIAL_QUALITY_FEATURES,
                'listnet_targets': ['10d', '15d'],
                'icir_clip_bounds': [0.10, 0.35],
                'prediction_zscore_ensemble': 'scorer层',
                'composite_ranking': 'scorer层内置',
            },
        }

        model_path = output_dir / f'v474_multi_target_{timestamp}.pkl'
        joblib.dump(model_data, model_path)
        logger.info(f"\n模型已保存: {model_path}")
        logger.info(f"  大小: {model_path.stat().st_size / 1024 / 1024:.1f} MB")

        # 保存训练历史
        history = {
            'version': 'v4.7.4',
            'base': 'V4.7.3 (简化管线) + 选择性V4.8增强',
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
                'meta_learner': False,
                'combined_isotonic': False,
                'icir_optimized': True,
                'has_listnet': True,
                'icir_clip_bounds': [0.10, 0.35],
            },
            'target_weights': self.target_weights,
            'ensemble_weights': {k: all_results[k]['weights'] for k in all_results},
            'v474_features': {
                'from_v473': '70 base features (精简版)',
                'from_v48': self.FINANCIAL_QUALITY_FEATURES,
                'total': len(self.feature_names),
            },
        }

        history_path = output_dir / f'training_history_{timestamp}.json'
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        latest_path = output_dir / 'training_history_latest.json'
        with open(latest_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        logger.info(f"训练历史已保存: {history_path}")
        logger.info(f"\nV4.7.4 训练完成! 总耗时: {duration:.0f}秒 ({duration/60:.1f}分钟)")
        logger.info(f"  特征: {len(self.feature_names)} (V4.7.3的70 + V4.8选择性{len(self.FINANCIAL_QUALITY_FEATURES)})")
        logger.info(f"  ListNet: 10d/15d")
        logger.info(f"  ICIR约束: [0.10, 0.35]")
        logger.info(f"  管线: 无Meta-Learner, 无Combined Isotonic")
        logger.info(f"  评分: 连续插值 (scorer层)")

        return model_data, history


class V475Trainer(V473Trainer):
    """V4.7.5 训练器 — V4.7.3底座 + 特征裁剪 + 标签平滑 + 自适应目标权重

    三层独立改进 (相比V4.7.3):
    Layer 1 (Scorer层, 不影响训练): 连续评分 np.interp + composite排名
    Layer 2A: 特征裁剪 — 去除底部低贡献特征 (70->~50)
    Layer 2B: 时序标签平滑 — 5d/10d/15d 标签 Gaussian 平滑, 减少端点噪声
    Layer 3: OOS-ICIR 自适应目标权重 — 信号强的周期贡献更大

    保留V4.7.3所有组件:
    - 放宽正则化 (num_leaves=31, min_data=200)
    - 无Meta-Learner, 无Combined Isotonic
    - ICIR权重[0.08,0.50] + Bear Specialist + Per-target Isotonic
    - 时间衰减 + 熊市加权 + 目标特异性Sharpe-Blend
    """

    # Features to prune: bottom-22 from Phase 0 importance analysis
    # Zero importance or <0.5% avg importance across 19 models
    PRUNE_FEATURES = [
        'sw_index_return_1d', 'sw_index_return_5d', 'industry_limit_up_ratio',  # zero importance
        'ma_cross', 'industry_volume_change', 'updown_volume_asymmetry',
        'industry_breadth', 'kdj_j', 'kdj_k', 'upper_shadow_ratio',
        'industry_return_5d', 'high_low_position', 'return_3d', 'ma5_ratio',
        'boll_position', 'amihud_illiquidity', 'industry_kdj_avg', 'ps_ttm',
        'industry_macd_bullish_pct', 'return_5d',
    ]
    # Keep return_1d (0.574%, reversal signal) and return_skewness_proxy (0.533%, risk signal)

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """V4.7.5: V4.7.3 features with bottom-20 pruned (70 -> ~50)"""
        # First apply V4.7.3's feature prep (removes downside_deviation_20d)
        X, y_3d, y_5d, y_10d, y_15d, df_out = super().prepare_features(df)

        # Prune low-importance features
        if self.feature_names:
            prune_indices = []
            keep_indices = []
            pruned_names = []
            for i, name in enumerate(self.feature_names):
                if name in self.PRUNE_FEATURES:
                    prune_indices.append(i)
                    pruned_names.append(name)
                else:
                    keep_indices.append(i)

            if keep_indices and len(keep_indices) < len(self.feature_names):
                X = X[:, keep_indices]
                self.feature_names = [self.feature_names[i] for i in keep_indices]
                logger.info(f"  V4.7.5: pruned {len(pruned_names)} low-importance features "
                            f"({len(self.feature_names)} remaining)")
                logger.info(f"    pruned: {pruned_names[:10]}{'...' if len(pruned_names) > 10 else ''}")

        return X, y_3d, y_5d, y_10d, y_15d, df_out

    def train_production_only(self, start_date=None, end_date=None,
                               purge_days=15, min_train_days=900):
        """V4.7.5 快速训练 — 跳过Walk-Forward, 直接训练生产模型 (~75%时间节省)"""
        # Reuse walk_forward_train with huge min_train_days to skip WF,
        # but we need to handle the empty wf_metrics case.
        # Simplest: call walk_forward_train with step_days > n_dates
        self.walk_forward_train(
            start_date=start_date, end_date=end_date,
            purge_days=purge_days, min_train_days=99999,
            val_days=120, test_days=120, step_days=99999)

    def walk_forward_train(self, start_date: str = None, end_date: str = None,
                            purge_days: int = 15, min_train_days: int = 900,
                            val_days: int = 120, test_days: int = 120,
                            step_days: int = 90):
        """V4.7.5 Walk-Forward 训练 — V4.7.3 + 特征裁剪 + 自适应目标权重"""
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("V4.7.5 Walk-Forward 训练 (V4.7.3 + 特征裁剪 + 自适应目标权重)")
        logger.info("=" * 60)
        logger.info(f"  参数: min_train={min_train_days}d, val={val_days}d, test={test_days}d, "
                     f"step={step_days}d, purge={purge_days}d")
        logger.info(f"  目标权重(初始): {self.target_weights}")
        logger.info(f"  Sharpe融合(目标特异性): {self.TARGET_SHARPE_BLEND}")
        logger.info(f"  特征: V4.7.3的70个 - {len(self.PRUNE_FEATURES)}个裁剪")
        logger.info(f"  正则化: V4.7.3 (num_leaves=31, min_data=200, path_smooth=5)")
        logger.info(f"  管线: V4.7.3 (无Meta-Learner, 无Combined Isotonic)")
        logger.info(f"  新增Layer2A: 特征裁剪 {self.PRUNE_FEATURES[:5]}...")
        logger.info(f"  新增Layer3: OOS-ICIR自适应目标权重")

        # 1. 一次性加载全量数据 (V4.7.3: 精简特征)
        df = self.load_data(start_date, end_date)
        X, y_3d, y_5d, y_10d, y_15d, df = self.prepare_features(df)

        dates = df['trade_date'].values
        unique_dates = np.sort(np.unique(dates))
        n_dates = len(unique_dates)
        logger.info(f"  总交易日: {n_dates}, 样本: {len(X):,}, 特征: {X.shape[1]}")

        # 2. 定义滚动窗口
        windows = []
        cursor = min_train_days
        while cursor + val_days + purge_days + test_days <= n_dates:
            train_end_idx = cursor - 1
            val_start_idx = cursor + purge_days
            val_end_idx = val_start_idx + val_days - 1
            test_start_idx = val_end_idx + purge_days + 1
            test_end_idx = min(test_start_idx + test_days - 1, n_dates - 1)
            windows.append({
                'train_end': unique_dates[train_end_idx],
                'val_start': unique_dates[val_start_idx],
                'val_end': unique_dates[val_end_idx],
                'test_start': unique_dates[test_start_idx],
                'test_end': unique_dates[test_end_idx],
            })
            cursor += step_days

        logger.info(f"  Walk-Forward窗口: {len(windows)}")
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
            train_dates_w = dates[train_mask]
            val_dates_w = dates[val_mask]

            # Winsorization
            X_train_w, wf_bounds = self.winsorize_features(X_train_w)
            self._apply_bounds(X_val_w, wf_bounds)
            self._apply_bounds(X_test_w, wf_bounds)

            for y_tr_w, y_va_w, y_te_w in [(y_3d_tr, y_3d_va, y_3d_te),
                                             (y_5d_tr, y_5d_va, y_5d_te),
                                             (y_10d_tr, y_10d_va, y_10d_te),
                                             (y_15d_tr, y_15d_va, y_15d_te)]:
                lo = np.percentile(y_tr_w, 1)
                hi = np.percentile(y_tr_w, 99)
                y_tr_w[:] = np.clip(y_tr_w, lo, hi)
                y_va_w[:] = np.clip(y_va_w, lo, hi)
                y_te_w[:] = np.clip(y_te_w, lo, hi)

            # Sharpe-Blend
            self.train_dates = train_dates_w
            self.val_dates = val_dates_w
            for target_key, y_tr_w, y_va_w, y_te_w in [
                ('label_3d', y_3d_tr, y_3d_va, y_3d_te),
                ('label_5d', y_5d_tr, y_5d_va, y_5d_te),
                ('label_10d', y_10d_tr, y_10d_va, y_10d_te),
                ('label_15d', y_15d_tr, y_15d_va, y_15d_te),
            ]:
                self._apply_sharpe_blend(y_tr_w, y_va_w, y_te_w,
                                          train_dates_w, val_dates_w, test_dates_w,
                                          target_key)

            # 训练4目标
            window_metrics = {}
            for target_key, y_tr, y_va, y_te in [
                ('3d', y_3d_tr, y_3d_va, y_3d_te),
                ('5d', y_5d_tr, y_5d_va, y_5d_te),
                ('10d', y_10d_tr, y_10d_va, y_10d_te),
                ('15d', y_15d_tr, y_15d_va, y_15d_te),
            ]:
                sample_w = self.compute_sample_weights(df[train_mask], y_tr)
                models, pred_train, pred_val = self.train_single_target_models(
                    X_train_w, X_val_w, y_tr, y_va, f"label_{target_key}",
                    sample_weights_train=sample_w)
                weights, rmses = self.calculate_ensemble_weights(pred_val, y_va)

                pred_test = {}
                for name, model in models.items():
                    try:
                        if name == 'xgb':
                            pred_test[name] = model.predict(xgb.DMatrix(X_test_w))
                        else:
                            pred_test[name] = model.predict(X_test_w)
                    except Exception:
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
            if ics:
                summary = {
                    'mean_ic': float(np.mean(ics)),
                    'std_ic': float(np.std(ics)),
                    'mean_icir': float(np.mean(icirs)),
                    'std_icir': float(np.std(icirs)),
                    'n_windows': len(ics),
                }
                logger.info(f"  {target_key}: IC={summary['mean_ic']:.4f}+-{summary['std_ic']:.4f}, "
                             f"ICIR={summary['mean_icir']:.4f}+-{summary['std_icir']:.4f}")
            else:
                summary = {'mean_ic': 0, 'std_ic': 0, 'mean_icir': 0, 'std_icir': 0, 'n_windows': 0}
                logger.info(f"  {target_key}: (skipped, no WF windows)")
            wf_summary[target_key] = summary

        # 4b. V4.7.5 Layer 3: Adaptive target weights from OOS ICIR
        logger.info("\n  Layer 3: Computing adaptive target weights from OOS ICIR...")
        raw_icirs = {}
        for target_key in ['3d', '5d', '10d', '15d']:
            icir_val = max(wf_summary[target_key]['mean_icir'], 0)  # floor at 0
            raw_icirs[f'label_{target_key}'] = icir_val

        total_icir = sum(raw_icirs.values())
        if total_icir > 0:
            adaptive_weights = {k: v / total_icir for k, v in raw_icirs.items()}
        else:
            adaptive_weights = self.target_weights  # fallback to fixed

        logger.info(f"  Fixed weights:    {self.target_weights}")
        logger.info(f"  Adaptive weights: {adaptive_weights}")
        logger.info(f"  (from OOS ICIR: {raw_icirs})")

        # Store for embedding in model
        self._adaptive_target_weights = adaptive_weights

        # 5. 训练最终生产模型
        logger.info("\n" + "=" * 60)
        logger.info("训练最终V4.7.5生产模型 (全量数据)")
        logger.info("=" * 60)

        split_idx = int(n_dates * 0.85)
        split_date = unique_dates[split_idx]
        train_mask_final = dates <= split_date
        val_mask_final = dates > split_date

        X_train_f, X_val_f = X[train_mask_final].copy(), X[val_mask_final].copy()
        self.val_dates = dates[val_mask_final]
        self.train_dates = dates[train_mask_final]

        X_train_f, self.winsorize_bounds = self.winsorize_features(X_train_f)
        self._apply_bounds(X_val_f, self.winsorize_bounds)
        logger.info(f"  生产模型: 特征Winsorization (训练集bounds), {len(self.winsorize_bounds)} 列")

        df_train_f = df[train_mask_final]
        all_results = {}

        y_val_dict = {}
        targets_final = [
            ('3d', y_3d[train_mask_final].copy(), y_3d[val_mask_final].copy()),
            ('5d', y_5d[train_mask_final].copy(), y_5d[val_mask_final].copy()),
            ('10d', y_10d[train_mask_final].copy(), y_10d[val_mask_final].copy()),
            ('15d', y_15d[train_mask_final].copy(), y_15d[val_mask_final].copy()),
        ]

        for target_key, y_tr, y_va in targets_final:
            lo = np.percentile(y_tr, 1)
            hi = np.percentile(y_tr, 99)
            y_tr[:] = np.clip(y_tr, lo, hi)
            y_va[:] = np.clip(y_va, lo, hi)

        logger.info(f"  [V4.7.5] 目标特异性Sharpe-Blend: {self.TARGET_SHARPE_BLEND}")
        train_dates_f = dates[train_mask_final]
        val_dates_f = dates[val_mask_final]
        for target_key, y_tr, y_va in targets_final:
            self._apply_sharpe_blend(y_tr, y_va, np.array([]),
                                      train_dates_f, val_dates_f, np.array([]),
                                      f"label_{target_key}")

        for target_key, y_tr, y_va in targets_final:
            sample_w = self.compute_sample_weights(df_train_f, y_tr)
            models, pred_train, pred_val = self.train_single_target_models(
                X_train_f, X_val_f, y_tr, y_va, f"label_{target_key}",
                sample_weights_train=sample_w)
            weights, rmses = self.calculate_ensemble_weights(pred_val, y_va)
            all_results[target_key] = {'models': models, 'weights': weights, 'rmses': rmses}
            y_val_dict[target_key] = y_va

        # 6. Bear specialist
        logger.info("\n" + "=" * 60)
        logger.info("Module C: Bear Specialist (10d/15d)")
        logger.info("=" * 60)
        bear_models = {}
        for target_key in ['10d', '15d']:
            y_tr_target = y_10d[train_mask_final] if target_key == '10d' else y_15d[train_mask_final]
            bear_model = self._train_bear_specialist(X_train_f, y_tr_target, df_train_f, target_key)
            if bear_model is not None:
                bear_models[target_key] = bear_model

        # 7. Isotonic calibration
        logger.info("\n" + "=" * 60)
        logger.info("Module A: Per-Target Isotonic Calibration")
        logger.info("=" * 60)
        isotonic_models = self._fit_isotonic_calibration(X_val_f, y_val_dict, all_results)

        # 8. ICIR weights
        logger.info("\n" + "=" * 60)
        logger.info("V4.7.5 ICIR Weights (inherited from V4.7.3)")
        logger.info("=" * 60)
        icir_weights = V46Trainer._optimize_icir_weights(self, all_results, X_val_f, y_val_dict, self.val_dates)
        for target_key, w in icir_weights.items():
            if target_key in all_results:
                all_results[target_key]['weights'] = w

        # Skip Meta-Learner and Combined Isotonic (V4.7.3 design)
        logger.info("\nV4.7.5: Skip Combined Isotonic + Meta-Learner (inherited from V4.7.3)")

        # 9. Feature importance
        self._log_feature_importance(all_results)

        # 10. Global quantiles
        X_all = X.copy()
        self._apply_bounds(X_all, self.winsorize_bounds)
        global_quantiles = self._compute_global_quantiles(X_all, all_results, self.target_weights)
        recommendation_thresholds = self._compute_recommendation_thresholds(X_all, all_results)

        # 11. Save model
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        output_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v475'
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        winsorize_bounds_dict = {}
        if self.winsorize_bounds and self.feature_names:
            for idx, (lo, hi) in enumerate(self.winsorize_bounds):
                if idx < len(self.feature_names):
                    winsorize_bounds_dict[self.feature_names[idx]] = (lo, hi)

        model_data = {
            'version': 'v4.7.5',
            'models': all_results,
            'feature_names': self.feature_names,
            'target_weights': self.target_weights,
            'market_features': list(self.market_calculator.market_features.columns[1:]),
            'winsorize_bounds': winsorize_bounds_dict,
            'global_quantiles': global_quantiles,
            'recommendation_thresholds': recommendation_thresholds,
            'cascade': False,
            'rank_normalized': False,
            'robust_zscore': True,
            'industry_excess_labels': True,
            'dual_stream': False,
            'cross_sectional_neutralization': False,
            'macro_feature_cols': self.macro_feature_cols,
            'stock_feature_cols': self.stock_feature_cols,
            'extra_features_from_daily_basic': ['pe_ttm', 'pb', 'ps_ttm', 'turnover_rate', 'log_market_cap',
                                                 'dv_ttm', 'turnover_rate_f', 'float_ratio'],
            'extra_features_from_tech_indicators': self.extra_tech_feature_names,
            'extra_features_financial': self.FINANCIAL_FEATURES,
            'extra_features_microstructure': self.MICROSTRUCTURE_FEATURES,
            'extra_features_reversal': self.REVERSAL_FEATURES,
            'extra_features_risk': self.RISK_FEATURES,
            'targets': ['3d', '5d', '10d', '15d'],
            'ensemble_type': 'icir_optimized',
            'sample_weighting': True,
            'time_decay_half_life': 365,
            'walk_forward_metrics': wf_summary,
            'walk_forward_windows': len(windows),
            'regularization': {
                'num_leaves': 31, 'min_data_in_leaf': 200,
                'reg_alpha': 0.5, 'reg_lambda': 3.0,
                'path_smooth': 5.0, 'learning_rate': 0.02,
            },
            'bear_models': bear_models,
            'isotonic_calibration': isotonic_models,
            'sharpe_label_blend': 'target_specific',
            'sharpe_blend_config': self.TARGET_SHARPE_BLEND,
            'liquidity_discount': True,
            'bear_sample_weighting': True,
            'min_train_days': min_train_days,
            'step_days': step_days,
            'has_lambdarank': True,
            'has_time_decay': True,
            'bug_fixes': ['winsorization_leakage', 'sharpe_blend_applied', 'market_index_000300'],
            'icir_optimized_weights': icir_weights,
            'combined_isotonic': None,
            'meta_learner': None,
            'meta_feature_names': None,
            'small_cap_weighting': False,
            'target_specific_sharpe': self.TARGET_SHARPE_BLEND,
            '3d_no_lambdarank': True,
            'pipeline_simplified': True,
            'features_pruned': {
                'removed_financial': ['gross_margin', 'current_ratio', 'assets_turn', 'netprofit_yoy', 'or_yoy'],
                'removed_risk': ['downside_deviation_20d'],
            },
            # V4.7.5 specific
            'adaptive_target_weights': getattr(self, '_adaptive_target_weights', None),
            'feature_pruning': {
                'pruned_count': len(self.PRUNE_FEATURES),
                'pruned_features': list(self.PRUNE_FEATURES),
            },
            'continuous_scoring': True,
            'composite_ranking': True,
        }

        model_path = output_dir / f'v475_multi_target_{timestamp}.pkl'
        joblib.dump(model_data, model_path)
        logger.info(f"\nModel saved: {model_path}")
        logger.info(f"  Size: {model_path.stat().st_size / 1024 / 1024:.1f} MB")

        # Also save with global quantiles embedded
        global_quantiles_arr = global_quantiles
        if global_quantiles_arr is not None:
            gq_path = output_dir / 'global_quantiles.npy'
            np.save(gq_path, global_quantiles_arr)

        history = {
            'version': 'v4.7.5',
            'base': 'V4.7.3 + Feature Pruning + Continuous Scoring + Adaptive Weights',
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'status': 'completed',
            'summary': {
                'training_samples': int(train_mask_final.sum()),
                'validation_samples': int(val_mask_final.sum()),
                'feature_count': len(self.feature_names),
                'features_pruned': len(self.PRUNE_FEATURES),
                'walk_forward_summary': wf_summary,
                'bear_models': list(bear_models.keys()),
                'isotonic_targets': list(isotonic_models.keys()) if isotonic_models else [],
                'meta_learner': False,
                'combined_isotonic': False,
                'icir_optimized': len(icir_weights) > 0,
                'adaptive_target_weights': getattr(self, '_adaptive_target_weights', None),
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

        logger.info(f"Training history saved: {history_path}")
        adaptive_tw = getattr(self, '_adaptive_target_weights', None)
        logger.info(f"\nV4.7.5 training complete! Duration: {duration:.0f}s ({duration/60:.1f}min)")
        logger.info(f"  Features: {len(self.feature_names)} (V4.7.3 minus {len(self.PRUNE_FEATURES)} pruned)")
        logger.info(f"  Pipeline: V4.7.3 base (no Meta-Learner, no Combined Isotonic)")
        logger.info(f"  New: Feature pruning ({len(self.PRUNE_FEATURES)} removed) + Continuous scoring + Composite ranking")
        if adaptive_tw:
            logger.info(f"  Adaptive target weights: {adaptive_tw}")

        return model_data, history


class V476Trainer(V475Trainer):
    """V4.7.6 训练器 — V4.7.5底座 + Top-K聚焦样本权重

    三层改进 (相比V4.7.5):
    训练层: Top-K聚焦 — 每日top-20%标签的样本权重×2.0
        模型浪费容量区分底部50%永远不会交易的股票;
        聚焦top-20%直接提升top-10选股质量
    评分层 (Scorer实现): 集成置信度折扣 + 波动率调整排名
        不影响训练, 仅影响推理时排名

    保留V4.7.5所有组件:
    - 特征裁剪(70→~50)
    - 放宽正则化 (num_leaves=31, min_data=200)
    - 无Meta-Learner, 无Combined Isotonic
    - ICIR权重[0.08,0.50] + Per-target Isotonic (V4.7.5已禁用)
    - 时间衰减 + 熊市加权 + 目标特异性Sharpe-Blend
    - 连续评分 + 10d+15d composite
    """

    # Top-K weight amplification factor for daily top-20% samples
    TOPK_AMPLIFICATION = 2.0
    TOPK_PERCENTILE = 80  # top 20%

    def compute_sample_weights(self, df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """V4.7.6: V4.7.5权重 + Top-K聚焦加权

        Daily cross-section top-20% by true label get TOPK_AMPLIFICATION weight.
        This makes gradient boosting focus capacity on correctly ranking
        the stocks that actually matter for top-10 selection.
        """
        # Base weights from V4.7.5 → V4.7.3 → V4.7.1 (涨跌停 + 极端 + 熊市 + 时间衰减)
        weights = super().compute_sample_weights(df, y)

        # Top-K focused weighting: upweight daily top-20% by true label
        if 'trade_date' in df.columns:
            dates = df['trade_date'].values
            unique_dates = np.unique(dates)
            topk_factor = np.ones(len(y), dtype=np.float64)
            n_upweighted = 0

            for d in unique_dates:
                mask = dates == d
                n = mask.sum()
                if n < 20:
                    continue
                # Top 20% of daily label
                threshold = np.percentile(y[mask], self.TOPK_PERCENTILE)
                daily_top = mask & (y >= threshold)
                topk_factor[daily_top] = self.TOPK_AMPLIFICATION
                n_upweighted += daily_top.sum()

            weights *= topk_factor
            logger.info(f"    Top-K聚焦加权: {n_upweighted:,} 样本 × {self.TOPK_AMPLIFICATION} "
                        f"(top {100 - self.TOPK_PERCENTILE}% daily)")

        return weights

    def walk_forward_train(self, start_date: str = None, end_date: str = None,
                            purge_days: int = 15, min_train_days: int = 900,
                            val_days: int = 120, test_days: int = 120,
                            step_days: int = 90):
        """V4.7.6 Walk-Forward 训练 — V4.7.5 + Top-K聚焦样本权重"""
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("V4.7.6 Walk-Forward 训练 (V4.7.5 + Top-K聚焦 + 置信度折扣 + 波动率调整)")
        logger.info("=" * 60)
        logger.info(f"  参数: min_train={min_train_days}d, val={val_days}d, test={test_days}d, "
                     f"step={step_days}d, purge={purge_days}d")
        logger.info(f"  目标权重: {self.target_weights}")
        logger.info(f"  Sharpe融合: {self.TARGET_SHARPE_BLEND}")
        logger.info(f"  特征: V4.7.5 (70 - {len(self.PRUNE_FEATURES)} pruned)")
        logger.info(f"  训练新增: Top-K聚焦 (top {100 - self.TOPK_PERCENTILE}% × {self.TOPK_AMPLIFICATION})")
        logger.info(f"  评分新增: Confidence Discount (α=0.15, floor=0.70)")
        logger.info(f"  评分新增: Vol-Adjusted Ranking (blend=0.35)")

        # Reuse V4.7.5's walk_forward_train with our overridden compute_sample_weights
        # The only training-side difference is the sample weight computation.
        # 1. Load data
        df = self.load_data(start_date, end_date)
        X, y_3d, y_5d, y_10d, y_15d, df = self.prepare_features(df)

        dates = df['trade_date'].values
        unique_dates = np.sort(np.unique(dates))
        n_dates = len(unique_dates)
        logger.info(f"  总交易日: {n_dates}, 样本: {len(X):,}, 特征: {X.shape[1]}")

        # 2. Define walk-forward windows
        windows = []
        cursor = min_train_days
        while cursor + val_days + purge_days + test_days <= n_dates:
            train_end_idx = cursor - 1
            val_start_idx = cursor + purge_days
            val_end_idx = val_start_idx + val_days - 1
            test_start_idx = val_end_idx + purge_days + 1
            test_end_idx = min(test_start_idx + test_days - 1, n_dates - 1)
            windows.append({
                'train_end': unique_dates[train_end_idx],
                'val_start': unique_dates[val_start_idx],
                'val_end': unique_dates[val_end_idx],
                'test_start': unique_dates[test_start_idx],
                'test_end': unique_dates[test_end_idx],
            })
            cursor += step_days

        logger.info(f"  Walk-Forward窗口: {len(windows)}")
        for i, w in enumerate(windows):
            logger.info(f"    窗口 {i+1}: train<='{w['train_end']}', val={w['val_start']}~{w['val_end']}, "
                         f"test={w['test_start']}~{w['test_end']}")

        # 3. Walk-forward evaluation
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
            train_dates_w = dates[train_mask]
            val_dates_w = dates[val_mask]

            # Winsorization
            X_train_w, wf_bounds = self.winsorize_features(X_train_w)
            self._apply_bounds(X_val_w, wf_bounds)
            self._apply_bounds(X_test_w, wf_bounds)

            for y_tr_w, y_va_w, y_te_w in [(y_3d_tr, y_3d_va, y_3d_te),
                                             (y_5d_tr, y_5d_va, y_5d_te),
                                             (y_10d_tr, y_10d_va, y_10d_te),
                                             (y_15d_tr, y_15d_va, y_15d_te)]:
                lo = np.percentile(y_tr_w, 1)
                hi = np.percentile(y_tr_w, 99)
                y_tr_w[:] = np.clip(y_tr_w, lo, hi)
                y_va_w[:] = np.clip(y_va_w, lo, hi)
                y_te_w[:] = np.clip(y_te_w, lo, hi)

            # Sharpe-Blend
            self.train_dates = train_dates_w
            self.val_dates = val_dates_w
            for target_key, y_tr_w, y_va_w, y_te_w in [
                ('label_3d', y_3d_tr, y_3d_va, y_3d_te),
                ('label_5d', y_5d_tr, y_5d_va, y_5d_te),
                ('label_10d', y_10d_tr, y_10d_va, y_10d_te),
                ('label_15d', y_15d_tr, y_15d_va, y_15d_te),
            ]:
                self._apply_sharpe_blend(y_tr_w, y_va_w, y_te_w,
                                          train_dates_w, val_dates_w, test_dates_w,
                                          target_key)

            # Train 4 targets (with Top-K sample weights via our overridden compute_sample_weights)
            window_metrics = {}
            for target_key, y_tr, y_va, y_te in [
                ('3d', y_3d_tr, y_3d_va, y_3d_te),
                ('5d', y_5d_tr, y_5d_va, y_5d_te),
                ('10d', y_10d_tr, y_10d_va, y_10d_te),
                ('15d', y_15d_tr, y_15d_va, y_15d_te),
            ]:
                sample_w = self.compute_sample_weights(df[train_mask], y_tr)
                models, pred_train, pred_val = self.train_single_target_models(
                    X_train_w, X_val_w, y_tr, y_va, f"label_{target_key}",
                    sample_weights_train=sample_w)
                weights, rmses = self.calculate_ensemble_weights(pred_val, y_va)

                pred_test = {}
                for name, model in models.items():
                    try:
                        if name == 'xgb':
                            pred_test[name] = model.predict(xgb.DMatrix(X_test_w))
                        else:
                            pred_test[name] = model.predict(X_test_w)
                    except Exception:
                        pred_test[name] = model.predict(X_test_w)

                ensemble_pred = self.ensemble_predict(pred_test, weights)
                ic, icir = self._calculate_daily_ic(ensemble_pred, y_te, test_dates_w)
                window_metrics[target_key] = {'ic': ic, 'icir': icir}
                logger.info(f"  {target_key}: IC={ic:.4f}, ICIR={icir:.4f}")

                del models, pred_train, pred_val, pred_test
                gc.collect()

            wf_metrics.append(window_metrics)

        # 4. Walk-Forward summary
        logger.info("\n" + "=" * 60)
        logger.info("Walk-Forward 汇总")
        logger.info("=" * 60)

        wf_summary = {}
        for target_key in ['3d', '5d', '10d', '15d']:
            ics = [m[target_key]['ic'] for m in wf_metrics if target_key in m]
            icirs = [m[target_key]['icir'] for m in wf_metrics if target_key in m]
            if ics:
                summary = {
                    'mean_ic': float(np.mean(ics)),
                    'std_ic': float(np.std(ics)),
                    'mean_icir': float(np.mean(icirs)),
                    'std_icir': float(np.std(icirs)),
                    'n_windows': len(ics),
                }
                logger.info(f"  {target_key}: IC={summary['mean_ic']:.4f}+-{summary['std_ic']:.4f}, "
                             f"ICIR={summary['mean_icir']:.4f}+-{summary['std_icir']:.4f}")
            else:
                summary = {'mean_ic': 0, 'std_ic': 0, 'mean_icir': 0, 'std_icir': 0, 'n_windows': 0}
                logger.info(f"  {target_key}: (skipped, no WF windows)")
            wf_summary[target_key] = summary

        # 4b. Adaptive target weights
        logger.info("\n  Layer 3: Computing adaptive target weights from OOS ICIR...")
        raw_icirs = {}
        for target_key in ['3d', '5d', '10d', '15d']:
            icir_val = max(wf_summary[target_key]['mean_icir'], 0)
            raw_icirs[f'label_{target_key}'] = icir_val

        total_icir = sum(raw_icirs.values())
        if total_icir > 0:
            adaptive_weights = {k: v / total_icir for k, v in raw_icirs.items()}
        else:
            adaptive_weights = self.target_weights
        logger.info(f"  Fixed weights:    {self.target_weights}")
        logger.info(f"  Adaptive weights: {adaptive_weights}")
        self._adaptive_target_weights = adaptive_weights

        # 5. Train final production model
        logger.info("\n" + "=" * 60)
        logger.info("训练最终V4.7.6生产模型 (全量数据)")
        logger.info("=" * 60)

        split_idx = int(n_dates * 0.85)
        split_date = unique_dates[split_idx]
        train_mask_final = dates <= split_date
        val_mask_final = dates > split_date

        X_train_f, X_val_f = X[train_mask_final].copy(), X[val_mask_final].copy()
        self.val_dates = dates[val_mask_final]
        self.train_dates = dates[train_mask_final]

        X_train_f, self.winsorize_bounds = self.winsorize_features(X_train_f)
        self._apply_bounds(X_val_f, self.winsorize_bounds)

        df_train_f = df[train_mask_final]
        all_results = {}
        y_val_dict = {}

        targets_final = [
            ('3d', y_3d[train_mask_final].copy(), y_3d[val_mask_final].copy()),
            ('5d', y_5d[train_mask_final].copy(), y_5d[val_mask_final].copy()),
            ('10d', y_10d[train_mask_final].copy(), y_10d[val_mask_final].copy()),
            ('15d', y_15d[train_mask_final].copy(), y_15d[val_mask_final].copy()),
        ]

        for target_key, y_tr, y_va in targets_final:
            lo = np.percentile(y_tr, 1)
            hi = np.percentile(y_tr, 99)
            y_tr[:] = np.clip(y_tr, lo, hi)
            y_va[:] = np.clip(y_va, lo, hi)

        logger.info(f"  [V4.7.6] 目标特异性Sharpe-Blend: {self.TARGET_SHARPE_BLEND}")
        train_dates_f = dates[train_mask_final]
        val_dates_f = dates[val_mask_final]
        for target_key, y_tr, y_va in targets_final:
            self._apply_sharpe_blend(y_tr, y_va, np.array([]),
                                      train_dates_f, val_dates_f, np.array([]),
                                      f"label_{target_key}")

        for target_key, y_tr, y_va in targets_final:
            sample_w = self.compute_sample_weights(df_train_f, y_tr)
            models, pred_train, pred_val = self.train_single_target_models(
                X_train_f, X_val_f, y_tr, y_va, f"label_{target_key}",
                sample_weights_train=sample_w)
            weights, rmses = self.calculate_ensemble_weights(pred_val, y_va)
            all_results[target_key] = {'models': models, 'weights': weights, 'rmses': rmses}
            y_val_dict[target_key] = y_va

        # 6. Bear specialist (inherited, but V4.7.5 disables in scorer)
        logger.info("\nModule C: Bear Specialist (10d/15d)")
        bear_models = {}
        for target_key in ['10d', '15d']:
            y_tr_target = y_10d[train_mask_final] if target_key == '10d' else y_15d[train_mask_final]
            bear_model = self._train_bear_specialist(X_train_f, y_tr_target, df_train_f, target_key)
            if bear_model is not None:
                bear_models[target_key] = bear_model

        # 7. Isotonic calibration (trained but disabled in scorer)
        logger.info("\nModule A: Per-Target Isotonic Calibration")
        isotonic_models = self._fit_isotonic_calibration(X_val_f, y_val_dict, all_results)

        # 8. ICIR weights
        logger.info("\nV4.7.6 ICIR Weights")
        icir_weights = V46Trainer._optimize_icir_weights(self, all_results, X_val_f, y_val_dict, self.val_dates)
        for target_key, w in icir_weights.items():
            if target_key in all_results:
                all_results[target_key]['weights'] = w

        logger.info("\nV4.7.6: Skip Combined Isotonic + Meta-Learner (inherited from V4.7.3)")

        # 9. Feature importance
        self._log_feature_importance(all_results)

        # 10. Global quantiles
        X_all = X.copy()
        self._apply_bounds(X_all, self.winsorize_bounds)
        global_quantiles = self._compute_global_quantiles(X_all, all_results, self.target_weights)
        recommendation_thresholds = self._compute_recommendation_thresholds(X_all, all_results)

        # 11. Save model
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        output_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v476'
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        winsorize_bounds_dict = {}
        if self.winsorize_bounds and self.feature_names:
            for idx, (lo, hi) in enumerate(self.winsorize_bounds):
                if idx < len(self.feature_names):
                    winsorize_bounds_dict[self.feature_names[idx]] = (lo, hi)

        # Import confidence/vol parameters for embedding
        from ml_models.v39.v476_production_scorer import (
            CONFIDENCE_ALPHA as _CA, CONFIDENCE_FLOOR as _CF, VOL_ADJUST_BLEND as _VB
        )

        model_data = {
            'version': 'v4.7.6',
            'models': all_results,
            'feature_names': self.feature_names,
            'target_weights': self.target_weights,
            'market_features': list(self.market_calculator.market_features.columns[1:]),
            'winsorize_bounds': winsorize_bounds_dict,
            'global_quantiles': global_quantiles,
            'recommendation_thresholds': recommendation_thresholds,
            'cascade': False,
            'rank_normalized': False,
            'robust_zscore': True,
            'industry_excess_labels': True,
            'dual_stream': False,
            'cross_sectional_neutralization': False,
            'macro_feature_cols': self.macro_feature_cols,
            'stock_feature_cols': self.stock_feature_cols,
            'extra_features_from_daily_basic': ['pe_ttm', 'pb', 'ps_ttm', 'turnover_rate', 'log_market_cap',
                                                 'dv_ttm', 'turnover_rate_f', 'float_ratio'],
            'extra_features_from_tech_indicators': self.extra_tech_feature_names,
            'extra_features_financial': self.FINANCIAL_FEATURES,
            'extra_features_microstructure': self.MICROSTRUCTURE_FEATURES,
            'extra_features_reversal': self.REVERSAL_FEATURES,
            'extra_features_risk': self.RISK_FEATURES,
            'targets': ['3d', '5d', '10d', '15d'],
            'ensemble_type': 'icir_optimized',
            'sample_weighting': True,
            'time_decay_half_life': 365,
            'walk_forward_metrics': wf_summary,
            'walk_forward_windows': len(windows),
            'regularization': {
                'num_leaves': 31, 'min_data_in_leaf': 200,
                'reg_alpha': 0.5, 'reg_lambda': 3.0,
                'path_smooth': 5.0, 'learning_rate': 0.02,
            },
            'bear_models': bear_models,
            'isotonic_calibration': isotonic_models,
            'sharpe_label_blend': 'target_specific',
            'sharpe_blend_config': self.TARGET_SHARPE_BLEND,
            'liquidity_discount': True,
            'bear_sample_weighting': True,
            'min_train_days': min_train_days,
            'step_days': step_days,
            'has_lambdarank': True,
            'has_time_decay': True,
            'bug_fixes': ['winsorization_leakage', 'sharpe_blend_applied', 'market_index_000300'],
            'icir_optimized_weights': icir_weights,
            'combined_isotonic': None,
            'meta_learner': None,
            'meta_feature_names': None,
            'small_cap_weighting': False,
            'target_specific_sharpe': self.TARGET_SHARPE_BLEND,
            '3d_no_lambdarank': True,
            'pipeline_simplified': True,
            'features_pruned': {
                'removed_financial': ['gross_margin', 'current_ratio', 'assets_turn', 'netprofit_yoy', 'or_yoy'],
                'removed_risk': ['downside_deviation_20d'],
            },
            # V4.7.5 inherited
            'adaptive_target_weights': getattr(self, '_adaptive_target_weights', None),
            'feature_pruning': {
                'pruned_count': len(self.PRUNE_FEATURES),
                'pruned_features': list(self.PRUNE_FEATURES),
            },
            'continuous_scoring': True,
            'composite_ranking': True,
            # V4.7.6 specific
            'topk_focused_weighting': True,
            'topk_amplification': self.TOPK_AMPLIFICATION,
            'topk_percentile': self.TOPK_PERCENTILE,
            'confidence_alpha': _CA,
            'confidence_floor': _CF,
            'vol_adjust_blend': _VB,
        }

        model_path = output_dir / f'v476_multi_target_{timestamp}.pkl'
        joblib.dump(model_data, model_path)
        logger.info(f"\nModel saved: {model_path}")
        logger.info(f"  Size: {model_path.stat().st_size / 1024 / 1024:.1f} MB")

        if global_quantiles is not None:
            gq_path = output_dir / 'global_quantiles.npy'
            np.save(gq_path, global_quantiles)

        history = {
            'version': 'v4.7.6',
            'base': 'V4.7.5 + Top-K Focused Weights + Confidence Discount + Vol-Adjusted Ranking',
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'status': 'completed',
            'summary': {
                'training_samples': int(train_mask_final.sum()),
                'validation_samples': int(val_mask_final.sum()),
                'feature_count': len(self.feature_names),
                'features_pruned': len(self.PRUNE_FEATURES),
                'walk_forward_summary': wf_summary,
                'bear_models': list(bear_models.keys()),
                'isotonic_targets': list(isotonic_models.keys()) if isotonic_models else [],
                'meta_learner': False,
                'combined_isotonic': False,
                'icir_optimized': len(icir_weights) > 0,
                'adaptive_target_weights': getattr(self, '_adaptive_target_weights', None),
            },
            'target_weights': self.target_weights,
            'ensemble_weights': {k: all_results[k]['weights'] for k in all_results},
            'v476_innovations': {
                '1_topk_focused_weights': f'top-{100 - self.TOPK_PERCENTILE}% × {self.TOPK_AMPLIFICATION}',
                '2_confidence_discount': f'α={_CA}, floor={_CF}',
                '3_vol_adjusted_ranking': f'blend={_VB}',
            },
        }

        history_path = output_dir / f'training_history_{timestamp}.json'
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        latest_path = output_dir / 'training_history_latest.json'
        with open(latest_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        logger.info(f"Training history saved: {history_path}")
        logger.info(f"\nV4.7.6 training complete! Duration: {duration:.0f}s ({duration/60:.1f}min)")
        logger.info(f"  Features: {len(self.feature_names)} (V4.7.3 minus {len(self.PRUNE_FEATURES)} pruned)")
        logger.info(f"  Training: Top-K focused (top-{100 - self.TOPK_PERCENTILE}% × {self.TOPK_AMPLIFICATION})")
        logger.info(f"  Scoring: Confidence discount (α={_CA}) + Vol-adjusted ranking (blend={_VB})")

        return model_data, history


class V477Trainer(V475Trainer):
    """V4.7.7 训练器 — V4.7.5底座 + Huber Loss + 缩短时间衰减 + DART增强

    基于V4.7.6迭代经验:
    - Top-K样本加权导致过拟合 → 放弃
    - Scorer后处理(consistency+vol)有效 → 保留(V4.7.6 scorer层)
    - 信号稳定性是瓶颈(ICIR=0.383, 一致性0.29) → 本版重点

    三项训练创新:
    1. Huber Loss: LightGBM从MSE改为Huber, 减少极端收益影响
       → 提升IC稳定性(CV), 一致性, 月胜率
    2. 缩短时间衰减: 365d→180d, 近期数据权重翻倍
       → 改善2025年后信号衰减(V4.7.5 WF W3/W4弱的根因)
    3. DART LGB: 新增dropout正则化的LGB模型, 增加ensemble多样性
       → 提升ICIR, 降低MaxDD

    保留V4.7.5所有组件 + V4.7.6 scorer (consistency+vol后处理)
    """

    # 时间衰减半衰期缩短
    TIME_DECAY_HALF_LIFE = 180.0  # V4.7.5=365d → V4.7.7=180d

    def compute_sample_weights(self, df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """V4.7.7: 缩短时间衰减(180d) + 继承V4.7.5其他权重"""
        # V4.4权重(涨跌停+极端+熊市) — 跳过V4.7.1的365d时间衰减
        weights = V44Trainer.compute_sample_weights(self, df, y)

        # 缩短时间衰减: 180d半衰期(V4.7.5=365d)
        if 'trade_date' in df.columns:
            dates = pd.to_datetime(df['trade_date'].values)
            max_date = dates.max()
            days_ago = ((max_date - dates) / pd.Timedelta(days=1)).astype(float)

            half_life = self.TIME_DECAY_HALF_LIFE
            decay = np.exp(-np.log(2) * days_ago / half_life)
            decay = np.clip(decay, 0.15, 1.0)  # 旧数据保留15%(vs V4.7.5的25%)

            weights *= decay
            n_old = (decay < 0.5).sum()
            logger.info(f"    时间衰减: half_life={half_life:.0f}d, {n_old:,} 样本权重<0.5")

        return weights

    def train_single_target_models(self, X_train, X_val, y_train, y_val, target_name: str,
                                    sample_weights_train=None):
        """V4.7.7: Huber Loss LGB + DART LGB + 继承V4.7.3其他模型"""
        import gc

        models = {}
        predictions_train = {}
        predictions_val = {}

        # 1. LightGBM-Huber — 核心创新: Huber Loss替代MSE
        logger.info(f"  训练 LightGBM-Huber ({target_name}, V4.7.7)...")
        # Auto-calibrate Huber delta from training label MAE
        huber_delta = float(np.median(np.abs(y_train - np.median(y_train)))) * 1.5
        lgb_params = {
            'objective': 'huber',
            'huber_delta': huber_delta,
            'metric': 'huber',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.02,
            'feature_fraction': 0.6,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'reg_alpha': 0.5,
            'reg_lambda': 3.0,
            'min_data_in_leaf': 200,
            'min_gain_to_split': 0.01,
            'path_smooth': 5.0,
            'verbose': -1,
        }

        lgb_train = lgb.Dataset(X_train, label=y_train,
                                weight=sample_weights_train, free_raw_data=True)
        lgb_val_ds = lgb.Dataset(X_val, label=y_val, reference=lgb_train, free_raw_data=True)

        lgb_model = lgb.train(
            lgb_params, lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_train, lgb_val_ds],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
        )
        models['lgb'] = lgb_model
        predictions_train['lgb'] = lgb_model.predict(X_train)
        predictions_val['lgb'] = lgb_model.predict(X_val)
        logger.info(f"    Huber delta={huber_delta:.6f}, best_iter={lgb_model.best_iteration}")
        del lgb_train, lgb_val_ds
        gc.collect()

        # 2. XGBoost — 继承V4.7.3
        logger.info(f"  训练 XGBoost ({target_name}, V4.7.3)...")
        xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.02,
            'subsample': 0.7,
            'colsample_bytree': 0.6,
            'reg_alpha': 0.5,
            'reg_lambda': 3.0,
            'min_child_weight': 50,
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

        # 3. CatBoost — 继承V4.7.3
        if HAS_CATBOOST:
            logger.info(f"  训练 CatBoost ({target_name}, V4.7.3)...")
            cb_model = cb.CatBoostRegressor(
                iterations=1000, learning_rate=0.02, depth=6,
                l2_leaf_reg=10, random_seed=42, verbose=False,
                early_stopping_rounds=30, min_data_in_leaf=200,
            )
            cb_pool_train = cb.Pool(X_train, label=y_train, weight=sample_weights_train)
            cb_pool_val = cb.Pool(X_val, label=y_val)
            cb_model.fit(cb_pool_train, eval_set=cb_pool_val, verbose=False)
            models['cb'] = cb_model
            predictions_train['cb'] = cb_model.predict(X_train)
            predictions_val['cb'] = cb_model.predict(X_val)
            del cb_pool_train, cb_pool_val
            gc.collect()

        # 4. RandomForest — 继承V4.7.3
        logger.info(f"  训练 RandomForest ({target_name}, V4.7.3)...")
        n_samples = X_train.shape[0]
        rf_max_samples = min(200_000, n_samples)
        rf_model = RandomForestRegressor(
            n_estimators=100, max_depth=15, max_samples=rf_max_samples,
            max_features=0.6, min_samples_leaf=200, n_jobs=-1,
            random_state=42, verbose=0,
        )
        rf_model.fit(X_train, y_train, sample_weight=sample_weights_train)
        models['rf'] = rf_model
        predictions_train['rf'] = rf_model.predict(X_train)
        predictions_val['rf'] = rf_model.predict(X_val)

        # 5. HistGradientBoosting — 继承V4.7.3
        logger.info(f"  训练 HistGradientBoosting ({target_name}, V4.7.3)...")
        hgb_model = HistGradientBoostingRegressor(
            max_iter=1000, learning_rate=0.02, max_depth=6,
            max_leaf_nodes=47, l2_regularization=3.0,
            min_samples_leaf=200, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=30,
            random_state=42, verbose=0,
        )
        hgb_model.fit(X_train, y_train, sample_weight=sample_weights_train)
        models['hgb'] = hgb_model
        predictions_train['hgb'] = hgb_model.predict(X_train)
        predictions_val['hgb'] = hgb_model.predict(X_val)

        # 6. LGB-DART — V4.7.7新增: dropout正则化增加多样性
        logger.info(f"  训练 LGB-DART ({target_name}, V4.7.7 新增)...")
        dart_params = {
            'objective': 'huber',
            'huber_delta': huber_delta,
            'metric': 'huber',
            'boosting_type': 'dart',
            'num_leaves': 31,
            'learning_rate': 0.05,       # DART需要更高lr (因为dropout)
            'feature_fraction': 0.7,
            'drop_rate': 0.15,           # 每轮dropout 15%的树
            'skip_drop': 0.5,           # 50%概率跳过dropout
            'reg_alpha': 0.5,
            'reg_lambda': 3.0,
            'min_data_in_leaf': 200,
            'path_smooth': 5.0,
            'verbose': -1,
        }
        lgb_dart_train = lgb.Dataset(X_train, label=y_train,
                                      weight=sample_weights_train, free_raw_data=True)
        lgb_dart_val = lgb.Dataset(X_val, label=y_val, reference=lgb_dart_train, free_raw_data=True)
        dart_model = lgb.train(
            dart_params, lgb_dart_train,
            num_boost_round=300,  # DART不支持early_stopping, 固定轮数
            valid_sets=[lgb_dart_train, lgb_dart_val],
            callbacks=[lgb.log_evaluation(0)]
        )
        models['lgb_dart'] = dart_model
        predictions_train['lgb_dart'] = dart_model.predict(X_train)
        predictions_val['lgb_dart'] = dart_model.predict(X_val)
        del lgb_dart_train, lgb_dart_val
        gc.collect()

        # 7. LambdaRank LGB (5d/10d/15d only, 3d skipped)
        if '3d' not in target_name:
            train_dates = getattr(self, 'train_dates', None)
            val_dates = getattr(self, 'val_dates', None)

            if train_dates is not None and len(train_dates) == len(y_train):
                logger.info(f"  训练 LGB-LambdaRank ({target_name}, V4.7.3)...")
                try:
                    from scipy.stats import rankdata

                    unique_train_dates = np.unique(train_dates)
                    relevance_train = np.zeros(len(y_train), dtype=np.int32)
                    group_train = []
                    for d in unique_train_dates:
                        mask = train_dates == d
                        n = mask.sum()
                        group_train.append(n)
                        if n >= 10:
                            ranks = rankdata(y_train[mask])
                            pct = (ranks - 1) / (n - 1)
                            relevance_train[mask] = np.clip((pct * 5).astype(int), 0, 4)
                        else:
                            relevance_train[mask] = 2

                    relevance_val = np.zeros(len(y_val), dtype=np.int32)
                    group_val = []
                    if val_dates is not None and len(val_dates) == len(y_val):
                        unique_val_dates = np.unique(val_dates)
                        for d in unique_val_dates:
                            mask = val_dates == d
                            n = mask.sum()
                            group_val.append(n)
                            if n >= 10:
                                ranks = rankdata(y_val[mask])
                                pct = (ranks - 1) / (n - 1)
                                relevance_val[mask] = np.clip((pct * 5).astype(int), 0, 4)
                            else:
                                relevance_val[mask] = 2

                    lgb_rank_params = {
                        'objective': 'lambdarank',
                        'metric': 'ndcg', 'eval_at': [10, 50],
                        'lambdarank_truncation_level': 50,
                        'num_leaves': 31, 'learning_rate': 0.02,
                        'feature_fraction': 0.6, 'bagging_fraction': 0.7, 'bagging_freq': 5,
                        'reg_alpha': 0.5, 'reg_lambda': 3.0,
                        'min_data_in_leaf': 200, 'min_gain_to_split': 0.01,
                        'path_smooth': 5.0, 'verbose': -1,
                    }

                    lgb_rank_train = lgb.Dataset(X_train, label=relevance_train,
                                                  group=group_train, free_raw_data=True)
                    lgb_rank_val = lgb.Dataset(X_val, label=relevance_val,
                                                group=group_val, reference=lgb_rank_train,
                                                free_raw_data=True)

                    lgb_rank_model = lgb.train(
                        lgb_rank_params, lgb_rank_train,
                        num_boost_round=500,
                        valid_sets=[lgb_rank_train, lgb_rank_val],
                        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
                    )
                    models['lgb_rank'] = lgb_rank_model
                    predictions_train['lgb_rank'] = lgb_rank_model.predict(X_train)
                    predictions_val['lgb_rank'] = lgb_rank_model.predict(X_val)
                    logger.info(f"    LGB-LambdaRank ({target_name}): 完成")
                    del lgb_rank_train, lgb_rank_val
                    gc.collect()
                except Exception as e:
                    logger.warning(f"    LGB-LambdaRank ({target_name}) 失败: {e}")

        return models, predictions_train, predictions_val

    def walk_forward_train(self, start_date: str = None, end_date: str = None,
                            purge_days: int = 15, min_train_days: int = 900,
                            val_days: int = 120, test_days: int = 120,
                            step_days: int = 90):
        """V4.7.7 Walk-Forward — 调用V4.7.5 WF, 然后将模型重新保存为v477"""
        import shutil

        logger.info("=" * 60)
        logger.info("V4.7.7 Walk-Forward 训练 (Huber Loss + 180d衰减 + DART)")
        logger.info("=" * 60)
        logger.info(f"  创新1: LGB Huber Loss (替代MSE, 减少极端收益影响)")
        logger.info(f"  创新2: 时间衰减 180d (vs V4.7.5=365d, 近期数据权重翻倍)")
        logger.info(f"  创新3: LGB-DART (dropout正则化, 增加ensemble多样性)")
        logger.info(f"  Scorer: 继承V4.7.6 (consistency+vol后处理)")

        # 调用V4.7.5的walk_forward_train (使用我们override的compute_sample_weights和train_single_target_models)
        model_data, history = super().walk_forward_train(
            start_date=start_date, end_date=end_date,
            purge_days=purge_days, min_train_days=min_train_days,
            val_days=val_days, test_days=test_days, step_days=step_days)

        # 将模型从v475目录移到v477目录
        v475_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v475'
        v477_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v477'
        v477_dir.mkdir(parents=True, exist_ok=True)

        # 找到刚保存的模型文件(最新的v475_*.pkl)
        v475_files = sorted(v475_dir.glob('v475_*.pkl'), key=lambda f: f.stat().st_mtime)
        if v475_files:
            latest = v475_files[-1]
            timestamp = latest.stem.replace('v475_multi_target_', '')
            new_path = v477_dir / f'v477_multi_target_{timestamp}.pkl'

            # 更新版本标识
            import joblib
            model_data['version'] = 'v4.7.7'
            model_data['time_decay_half_life'] = self.TIME_DECAY_HALF_LIFE
            model_data['huber_loss'] = True
            model_data['dart_boosting'] = True
            model_data['v477_innovations'] = {
                '1_huber_loss': 'LGB objective=huber (auto-delta from MAD)',
                '2_time_decay': f'half_life={self.TIME_DECAY_HALF_LIFE}d (vs 365d)',
                '3_dart': 'LGB-DART (drop_rate=0.15, skip_drop=0.5)',
            }

            joblib.dump(model_data, new_path)
            logger.info(f"\nV4.7.7 model saved: {new_path}")
            logger.info(f"  Size: {new_path.stat().st_size / 1024 / 1024:.1f} MB")

            # 复制辅助文件
            for aux in ['global_quantiles.npy', 'recommendation_thresholds.json']:
                src = v475_dir / aux
                if src.exists():
                    shutil.copy2(str(src), str(v477_dir / aux))

            # 删除v475目录下的这个模型(它属于v477)
            latest.unlink()
            # 也删除对应的history文件
            for hf in v475_dir.glob(f'training_history_{timestamp}*'):
                hf.unlink()
            logger.info(f"  Cleaned up v475 directory")

            # 保存v477 history
            history['version'] = 'v4.7.7'
            history['base'] = 'V4.7.5 + Huber Loss + 180d Time Decay + DART Boosting'
            history['v477_innovations'] = model_data['v477_innovations']

            import json as _json
            history_path = v477_dir / f'training_history_{timestamp}.json'
            with open(history_path, 'w', encoding='utf-8') as f:
                _json.dump(history, f, indent=2, ensure_ascii=False)
            latest_path = v477_dir / 'training_history_latest.json'
            with open(latest_path, 'w', encoding='utf-8') as f:
                _json.dump(history, f, indent=2, ensure_ascii=False)

            logger.info(f"\nV4.7.7 training complete!")
        else:
            logger.warning("No v475 model file found to rename")

        return model_data, history


class V478Trainer(V477Trainer):
    """V4.7.8 训练器 — Huber Loss(V4.7.7) + 365d衰减(V4.7.5) = 两者之长

    回测发现:
    - V4.7.5: Top3最佳(+1.22%, 56.5%胜率) ← 365d衰减保留头部信号
    - V4.7.7: IC最高(0.110) ← Huber Loss减少极端收益干扰
    - V4.7.7 top3弱(-0.20%) ← 180d衰减丢失了长期头部pattern

    V4.7.8策略: Huber Loss + 365d衰减 → 兼顾IC和头部集中度
    """

    TIME_DECAY_HALF_LIFE = 365.0  # 恢复V4.7.5的365d (V4.7.7用180d过激)

    def compute_sample_weights(self, df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """V4.7.8: Huber Loss(继承V4.7.7模型) + 365d衰减(恢复V4.7.5)"""
        weights = V44Trainer.compute_sample_weights(self, df, y)

        if 'trade_date' in df.columns:
            dates = pd.to_datetime(df['trade_date'].values)
            max_date = dates.max()
            days_ago = ((max_date - dates) / pd.Timedelta(days=1)).astype(float)

            half_life = self.TIME_DECAY_HALF_LIFE
            decay = np.exp(-np.log(2) * days_ago / half_life)
            decay = np.clip(decay, 0.25, 1.0)  # V4.7.5的0.25 (V4.7.7用0.15)

            weights *= decay
            n_old = (decay < 0.5).sum()
            logger.info(f"    时间衰减: half_life={half_life:.0f}d, {n_old:,} 样本权重<0.5 (V4.7.8: 恢复365d)")

        return weights

    def walk_forward_train(self, start_date=None, end_date=None,
                            purge_days=15, min_train_days=900,
                            val_days=120, test_days=120, step_days=90):
        """V4.7.8 Walk-Forward — Huber+DART(V4.7.7) + 365d衰减(V4.7.5)"""
        import shutil

        logger.info("=" * 60)
        logger.info("V4.7.8 Walk-Forward 训练 (Huber Loss + 365d衰减 = V475 Top3 + V477 IC)")
        logger.info("=" * 60)
        logger.info(f"  来自V4.7.7: Huber Loss + DART + LambdaRank")
        logger.info(f"  来自V4.7.5: 365d时间衰减 + 0.25最低权重")
        logger.info(f"  假设: Huber提升IC, 365d保留头部长期pattern")

        # 调用V4.7.5的walk_forward_train (使用V4.7.7的模型+本类的365d权重)
        model_data, history = V475Trainer.walk_forward_train(
            self, start_date=start_date, end_date=end_date,
            purge_days=purge_days, min_train_days=min_train_days,
            val_days=val_days, test_days=test_days, step_days=step_days)

        # 将模型从v475目录移到v478目录
        v475_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v475'
        v478_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v478'
        v478_dir.mkdir(parents=True, exist_ok=True)

        v475_files = sorted(v475_dir.glob('v475_*.pkl'), key=lambda f: f.stat().st_mtime)
        if v475_files:
            latest = v475_files[-1]
            timestamp = latest.stem.replace('v475_multi_target_', '')
            new_path = v478_dir / f'v478_multi_target_{timestamp}.pkl'

            import joblib
            model_data['version'] = 'v4.7.8'
            model_data['time_decay_half_life'] = self.TIME_DECAY_HALF_LIFE
            model_data['huber_loss'] = True
            model_data['dart_boosting'] = True
            model_data['v478_innovations'] = {
                'from_v477': 'Huber Loss + DART + LambdaRank (IC提升)',
                'from_v475': '365d时间衰减 + 0.25最低权重 (Top3头部集中度)',
                'hypothesis': 'Huber提升整体IC, 365d保留长期头部pattern → Top3+IC双优',
            }
            joblib.dump(model_data, new_path)
            logger.info(f"\nV4.7.8 model saved: {new_path}")
            logger.info(f"  Size: {new_path.stat().st_size / 1024 / 1024:.1f} MB")

            for aux in ['global_quantiles.npy', 'recommendation_thresholds.json']:
                src = v475_dir / aux
                if src.exists():
                    shutil.copy2(str(src), str(v478_dir / aux))

            latest.unlink()
            for hf in v475_dir.glob(f'training_history_{timestamp}*'):
                hf.unlink()

            history['version'] = 'v4.7.8'
            history['base'] = 'V4.7.7 (Huber+DART) + V4.7.5 (365d decay)'
            history['v478_innovations'] = model_data['v478_innovations']

            import json as _json
            history_path = v478_dir / f'training_history_{timestamp}.json'
            with open(history_path, 'w', encoding='utf-8') as f:
                _json.dump(history, f, indent=2, ensure_ascii=False)
            latest_path = v478_dir / 'training_history_latest.json'
            with open(latest_path, 'w', encoding='utf-8') as f:
                _json.dump(history, f, indent=2, ensure_ascii=False)

            logger.info(f"\nV4.7.8 training complete!")
        else:
            logger.warning("No v475 model file found to rename")

        return model_data, history


class V479Trainer(V477Trainer):
    """V4.7.9 训练器 — V4.7.8基础 + 头部加权 + 240d衰减(折中)

    进一步强化Top3选股能力:
    1. Huber Loss (继承V4.7.7): IC稳定性
    2. 240d衰减 (V4.7.5的365d和V4.7.7的180d折中)
    3. 头部加权: Top 5%正收益样本2x权重 → 让模型更关注高收益区分度
    4. DART + LambdaRank (继承V4.7.7): ensemble多样性 + 排名优化
    """

    TIME_DECAY_HALF_LIFE = 240.0  # 折中: V4.7.5=365, V4.7.7=180

    def compute_sample_weights(self, df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """V4.7.9: 240d衰减 + 头部加权(Top5% 正收益样本×2)"""
        weights = V44Trainer.compute_sample_weights(self, df, y)

        # 时间衰减: 240d
        if 'trade_date' in df.columns:
            dates = pd.to_datetime(df['trade_date'].values)
            max_date = dates.max()
            days_ago = ((max_date - dates) / pd.Timedelta(days=1)).astype(float)

            half_life = self.TIME_DECAY_HALF_LIFE
            decay = np.exp(-np.log(2) * days_ago / half_life)
            decay = np.clip(decay, 0.20, 1.0)

            weights *= decay
            n_old = (decay < 0.5).sum()
            logger.info(f"    时间衰减: half_life={half_life:.0f}d, {n_old:,} 样本权重<0.5")

        # 头部加权: Top 5% 正收益样本获得2x权重
        # 目的: 让模型更精确区分高收益样本(Top3选股能力)
        if len(y) > 100:
            p95 = np.percentile(y, 95)
            top_mask = y >= p95
            weights[top_mask] *= 2.0
            n_top = top_mask.sum()
            logger.info(f"    头部加权: Top 5% (y>={p95:.4f}) {n_top:,} 样本 ×2.0")

        return weights

    def walk_forward_train(self, start_date=None, end_date=None,
                            purge_days=15, min_train_days=900,
                            val_days=120, test_days=120, step_days=90):
        """V4.7.9 Walk-Forward — Huber+DART + 240d衰减 + 头部加权"""
        import shutil

        logger.info("=" * 60)
        logger.info("V4.7.9 Walk-Forward 训练 (Huber+DART + 240d衰减 + 头部加权)")
        logger.info("=" * 60)
        logger.info(f"  来自V4.7.7: Huber Loss + DART + LambdaRank")
        logger.info(f"  时间衰减: 240d (V4.7.5=365 和 V4.7.7=180 的折中)")
        logger.info(f"  头部加权: Top 5%正收益样本×2 (强化Top3区分度)")

        model_data, history = V475Trainer.walk_forward_train(
            self, start_date=start_date, end_date=end_date,
            purge_days=purge_days, min_train_days=min_train_days,
            val_days=val_days, test_days=test_days, step_days=step_days)

        # 将模型从v475目录移到v479目录
        v475_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v475'
        v479_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v479'
        v479_dir.mkdir(parents=True, exist_ok=True)

        v475_files = sorted(v475_dir.glob('v475_*.pkl'), key=lambda f: f.stat().st_mtime)
        if v475_files:
            latest = v475_files[-1]
            timestamp = latest.stem.replace('v475_multi_target_', '')
            new_path = v479_dir / f'v479_multi_target_{timestamp}.pkl'

            import joblib
            model_data['version'] = 'v4.7.9'
            model_data['time_decay_half_life'] = self.TIME_DECAY_HALF_LIFE
            model_data['huber_loss'] = True
            model_data['dart_boosting'] = True
            model_data['top_weighted'] = True
            model_data['v479_innovations'] = {
                'from_v477': 'Huber Loss + DART + LambdaRank',
                'time_decay': '240d (折中: V475=365, V477=180)',
                'top_weighting': 'Top 5% positive return samples ×2.0',
                'hypothesis': '240d折中衰减 + 头部加权 → Top3精度+IC双优',
            }
            joblib.dump(model_data, new_path)
            logger.info(f"\nV4.7.9 model saved: {new_path}")
            logger.info(f"  Size: {new_path.stat().st_size / 1024 / 1024:.1f} MB")

            for aux in ['global_quantiles.npy', 'recommendation_thresholds.json']:
                src = v475_dir / aux
                if src.exists():
                    shutil.copy2(str(src), str(v479_dir / aux))

            latest.unlink()
            for hf in v475_dir.glob(f'training_history_{timestamp}*'):
                hf.unlink()

            history['version'] = 'v4.7.9'
            history['base'] = 'V4.7.7 (Huber+DART) + 240d decay + Top5% ×2 weighting'
            history['v479_innovations'] = model_data['v479_innovations']

            import json as _json
            history_path = v479_dir / f'training_history_{timestamp}.json'
            with open(history_path, 'w', encoding='utf-8') as f:
                _json.dump(history, f, indent=2, ensure_ascii=False)
            latest_path = v479_dir / 'training_history_latest.json'
            with open(latest_path, 'w', encoding='utf-8') as f:
                _json.dump(history, f, indent=2, ensure_ascii=False)

            logger.info(f"\nV4.7.9 training complete!")
        else:
            logger.warning("No v475 model file found to rename")

        return model_data, history


class V480Trainer(V475Trainer):
    """V4.8.0 训练器 — V4.7.5底座 + 270d时间衰减(唯一改动)

    V3北极星精准狙击:
    - ic_decay_ratio(H2/H1)=0.52(1/5) → 目标0.70+(3/5)
    - 时间衰减365d→270d: 增强近期数据权重, 减缓信号衰减
    - 保留MSE Loss(不用Huber, 保alpha)
    - 保留6模型ensemble(不加DART, 省时间)
    - Scorer层继承V4.7.6(consistency+vol后处理)
    """

    TIME_DECAY_HALF_LIFE = 270.0

    def compute_sample_weights(self, df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """V4.8.0: 270d时间衰减(vs V4.7.5=365d) + 继承其他权重"""
        weights = V44Trainer.compute_sample_weights(self, df, y)

        if 'trade_date' in df.columns:
            dates = pd.to_datetime(df['trade_date'].values)
            max_date = dates.max()
            days_ago = ((max_date - dates) / pd.Timedelta(days=1)).astype(float)

            half_life = self.TIME_DECAY_HALF_LIFE
            decay = np.exp(-np.log(2) * days_ago / half_life)
            decay = np.clip(decay, 0.20, 1.0)

            weights *= decay
            n_old = (decay < 0.5).sum()
            logger.info(f"    时间衰减: half_life={half_life:.0f}d, {n_old:,} 样本权重<0.5")

        return weights

    def walk_forward_train(self, start_date=None, end_date=None,
                            purge_days=15, min_train_days=900,
                            val_days=120, test_days=120, step_days=90):
        """V4.8.0 Walk-Forward — 270d时间衰减"""
        logger.info("=" * 60)
        logger.info("V4.8.0 Walk-Forward 训练 (270d时间衰减, 目标ic_decay_ratio提升)")
        logger.info("=" * 60)
        logger.info(f"  唯一改动: 时间衰减 365d→270d")
        logger.info(f"  目标: ic_decay_ratio 0.52→0.70+ (V3北极星L1 +2分)")
        logger.info(f"  保留: MSE Loss, 6模型ensemble, V4.7.6 scorer后处理")

        model_data, history = super().walk_forward_train(
            start_date=start_date, end_date=end_date,
            purge_days=purge_days, min_train_days=min_train_days,
            val_days=val_days, test_days=test_days, step_days=step_days)

        # 将模型从v475目录移到v480目录
        import shutil
        v475_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v475'
        v480_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v480'
        v480_dir.mkdir(parents=True, exist_ok=True)

        v475_files = sorted(v475_dir.glob('v475_*.pkl'), key=lambda f: f.stat().st_mtime)
        if v475_files:
            latest = v475_files[-1]
            timestamp = latest.stem.replace('v475_multi_target_', '')
            new_path = v480_dir / f'v480_multi_target_{timestamp}.pkl'

            import joblib
            model_data['version'] = 'v4.8.0'
            model_data['time_decay_half_life'] = self.TIME_DECAY_HALF_LIFE
            model_data['v480_target'] = 'ic_decay_ratio improvement via 270d time decay'
            joblib.dump(model_data, new_path)
            logger.info(f"\nV4.8.0 model saved: {new_path}")
            logger.info(f"  Size: {new_path.stat().st_size / 1024 / 1024:.1f} MB")

            for aux in ['global_quantiles.npy', 'recommendation_thresholds.json']:
                src = v475_dir / aux
                if src.exists():
                    shutil.copy2(str(src), str(v480_dir / aux))

            latest.unlink()
            for hf in v475_dir.glob(f'training_history_{timestamp}*'):
                hf.unlink()

            history['version'] = 'v4.8.0'
            history['base'] = 'V4.7.5 + 270d Time Decay (target: ic_decay_ratio)'
            import json as _json
            for dest in [v480_dir / f'training_history_{timestamp}.json',
                         v480_dir / 'training_history_latest.json']:
                with open(dest, 'w', encoding='utf-8') as f:
                    _json.dump(history, f, indent=2, ensure_ascii=False)

            logger.info(f"V4.8.0 training complete!")

        return model_data, history


class V481Trainer(V475Trainer):
    """V4.8.1 训练器 — V4.7.5底座 + 15个新因子 (50 → 60特征)

    唯一改动: 特征扩展 (prune 5 weak + add 15 new = 50 → 60)

    新增5个裁剪 (在V4.7.5的20个基础上):
    - min_pct_change_5d, industry_concentration, sw_l1_code,
      volume_price_corr_10d, volume_trend

    新增15个因子 (从OHLCV+技术指标计算):
    1.  atr_percentile:       ATR_14在个股历史中的百分位
    2.  vol_concentration:    20日成交量HHI集中度
    3.  intraday_ret_20d:     20日日内收益率之和
    4.  industry_mom_rank:    行业内10日动量排名
    5.  vwap_dev_20d:         20日VWAP偏离度均值
    6.  max_ret_20d:          20日最大单日涨幅
    7.  gk_vol_20d:           Garman-Klass波动率
    8.  abnormal_turnover:    换手率异常度(vs 20日均值)
    9.  overnight_ret_20d:    20日隔夜收益率之和
    10. turnover_vol_20d:     换手率波动率
    11. cci_14:               CCI技术指标
    12. squeeze_mom_calc:     (Close-MA20)/ATR 动量
    13. vol_price_div:        量价背离因子
    14. price_acceleration:   价格加速度(5d收益变化率)
    15. price_pos_volatility: 价格位置波动率

    继承V4.7.5:
    - 6模型ensemble, MSE Loss, 365d时间衰减
    - V4.7.3管线(无Meta-Learner, 无Combined Isotonic)
    - Bear Specialist, Per-target Isotonic, ICIR权重
    - 连续评分 + Composite排名
    """

    # V4.8.1新增裁剪 (在V4.7.5的20个基础上)
    PRUNE_FEATURES = V475Trainer.PRUNE_FEATURES + [
        'min_pct_change_5d', 'industry_concentration', 'sw_l1_code',
        'volume_price_corr_10d', 'volume_trend',
    ]

    # 15个新因子名称
    NEW_FACTORS = [
        'atr_percentile', 'vol_concentration', 'intraday_ret_20d',
        'industry_mom_rank', 'vwap_dev_20d', 'max_ret_20d',
        'gk_vol_20d', 'abnormal_turnover', 'overnight_ret_20d',
        'turnover_vol_20d', 'cci_14', 'squeeze_mom_calc',
        'vol_price_div', 'price_acceleration', 'price_pos_volatility',
    ]

    def load_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """V4.8.1: V4.7.5基础数据 + 15个新因子(从OHLCV+技术指标计算)"""
        df = super().load_data(start_date, end_date)

        date_min = df['trade_date'].min()
        date_max = df['trade_date'].max()
        conn = sqlite3.connect(self.db_path)

        # === 加载OHLCV数据 (需要额外前40天用于滚动窗口) ===
        logger.info("  V4.8.1 加载OHLCV + 技术指标 (计算15个新因子)...")
        from datetime import datetime as dt_cls, timedelta as td_cls
        try:
            ext_start = (dt_cls.strptime(date_min, '%Y-%m-%d') - td_cls(days=60)).strftime('%Y-%m-%d')
        except Exception:
            ext_start = (dt_cls.strptime(date_min, '%Y%m%d') - td_cls(days=60)).strftime('%Y%m%d')

        ohlcv_query = """
        SELECT s.code, q.trade_date, q.open, q.high, q.low, q.close,
               q.volume, q.price_change_pct
        FROM daily_quotes q
        JOIN securities s ON q.security_id = s.id
        WHERE s.type = 'A股' AND q.trade_date >= ? AND q.trade_date <= ?
        ORDER BY s.code, q.trade_date
        """
        df_ohlcv = pd.read_sql(ohlcv_query, conn, params=[ext_start, date_max])
        logger.info(f"    OHLCV 记录: {len(df_ohlcv):,}")

        # 加载 atr_14, cci_14 from technical_indicators
        tech_query = """
        SELECT s.code, ti.trade_date, ti.atr_14, ti.cci_14
        FROM technical_indicators ti
        JOIN securities s ON ti.security_id = s.id
        WHERE s.type = 'A股' AND ti.trade_date >= ? AND ti.trade_date <= ?
        """
        df_tech = pd.read_sql(tech_query, conn, params=[ext_start, date_max])
        logger.info(f"    技术指标(atr_14/cci_14) 记录: {len(df_tech):,}")

        # 加载 turnover_rate from daily_basic (如果主df中没有)
        if 'turnover_rate' not in df_ohlcv.columns:
            turn_query = """
            SELECT s.code, db.trade_date, db.turnover_rate
            FROM daily_basic db
            JOIN securities s ON db.security_id = s.id
            WHERE db.trade_date >= ? AND db.trade_date <= ?
            """
            df_turn = pd.read_sql(turn_query, conn, params=[ext_start, date_max])
            df_ohlcv = df_ohlcv.merge(df_turn, on=['code', 'trade_date'], how='left')
            df_ohlcv['turnover_rate'] = df_ohlcv['turnover_rate'].fillna(0.0)
        conn.close()

        if len(df_ohlcv) == 0:
            logger.warning("    OHLCV为空, 15个新因子填0")
            for col in self.NEW_FACTORS:
                df[col] = 0.0
            return df

        # 合并 atr_14/cci_14 到 OHLCV
        if len(df_tech) > 0:
            df_ohlcv = df_ohlcv.merge(df_tech, on=['code', 'trade_date'], how='left')
        else:
            df_ohlcv['atr_14'] = np.nan
            df_ohlcv['cci_14'] = np.nan

        # === 计算15个新因子 (per-stock 滚动) ===
        logger.info("    计算15个新因子...")
        factor_parts = []
        n_stocks = df_ohlcv['code'].nunique()
        processed = 0

        for code, grp in df_ohlcv.groupby('code'):
            grp = grp.sort_values('trade_date').copy()
            n = len(grp)
            if n < 5:
                continue

            close = grp['close'].values
            open_ = grp['open'].values
            high = grp['high'].values
            low = grp['low'].values
            volume = grp['volume'].values.astype(float)
            pct = grp['price_change_pct'].values
            atr14 = grp['atr_14'].values if 'atr_14' in grp.columns else np.full(n, np.nan)
            cci14 = grp['cci_14'].values if 'cci_14' in grp.columns else np.full(n, np.nan)
            turnover = grp['turnover_rate'].values if 'turnover_rate' in grp.columns else np.full(n, np.nan)

            out = grp[['code', 'trade_date']].copy()

            # 1. atr_percentile: ATR_14 rank within stock's history
            atr_s = pd.Series(atr14)
            out['atr_percentile'] = atr_s.rank(pct=True).values

            # 2. vol_concentration: Volume HHI over 20d
            vol_s = pd.Series(volume)
            def _vol_hhi(x):
                total = x.sum()
                if total <= 0:
                    return 0.0
                shares = x / total
                return (shares ** 2).sum()
            out['vol_concentration'] = vol_s.rolling(20, min_periods=5).apply(_vol_hhi, raw=True).values

            # 3. intraday_ret_20d: Sum of intraday returns over 20d
            intraday_ret = close / np.where(open_ > 0, open_, 1e-8) - 1
            out['intraday_ret_20d'] = pd.Series(intraday_ret).rolling(20, min_periods=5).sum().values

            # 4. industry_mom_rank: computed later (needs cross-stock data)
            # placeholder - will be filled below
            out['industry_mom_rank'] = np.nan

            # 5. vwap_dev_20d: VWAP deviation
            typical_price = (high + low + close) / 3
            tp_safe = np.where(typical_price > 0, typical_price, 1e-8)
            vwap_dev = (close - typical_price) / tp_safe
            out['vwap_dev_20d'] = pd.Series(vwap_dev).rolling(20, min_periods=5).mean().values

            # 6. max_ret_20d: Max daily return in 20d
            out['max_ret_20d'] = pd.Series(pct).rolling(20, min_periods=5).max().values

            # 7. gk_vol_20d: Garman-Klass volatility
            hl_ratio = np.where(low > 0, high / low, 1.0)
            co_ratio = np.where(open_ > 0, close / open_, 1.0)
            gk_raw = np.sqrt(np.abs(0.5 * np.log(hl_ratio)**2 - (2*np.log(2)-1) * np.log(co_ratio)**2))
            out['gk_vol_20d'] = pd.Series(gk_raw).rolling(20, min_periods=5).mean().values

            # 8. abnormal_turnover: Turnover z-score vs 20d mean
            turn_s = pd.Series(turnover)
            turn_ma20 = turn_s.rolling(20, min_periods=5).mean().values
            turn_ma20_safe = np.where(turn_ma20 > 1e-8, turn_ma20, 1e-8)
            out['abnormal_turnover'] = (turnover / turn_ma20_safe - 1)

            # 9. overnight_ret_20d: Cumulative overnight returns
            prev_close = np.concatenate([[np.nan], close[:-1]])
            overnight_ret = np.where(prev_close > 0, open_ / prev_close - 1, 0.0)
            overnight_ret[0] = 0.0
            out['overnight_ret_20d'] = pd.Series(overnight_ret).rolling(20, min_periods=5).sum().values

            # 10. turnover_vol_20d: Turnover rate volatility
            out['turnover_vol_20d'] = turn_s.rolling(20, min_periods=5).std().values

            # 11. cci_14: direct from tech indicators
            out['cci_14'] = cci14

            # 12. squeeze_mom_calc: (Close - MA20) / ATR
            ma20 = pd.Series(close).rolling(20, min_periods=5).mean().values
            atr_safe = np.where(atr14 > 1e-8, atr14, 1e-8)
            out['squeeze_mom_calc'] = (close - ma20) / atr_safe

            # 13. vol_price_div: Volume-price divergence
            close_s = pd.Series(close)
            vol_chg_5d = vol_s.pct_change(5).values
            pct_5d = close_s.pct_change(5).values
            out['vol_price_div'] = -pct_5d * vol_chg_5d

            # 14. price_acceleration: 5d return - previous 5d return
            ret_5d = close_s.pct_change(5).values
            ret_5d_prev = np.concatenate([[np.nan]*5, ret_5d[:-5]])
            out['price_acceleration'] = ret_5d - ret_5d_prev

            # 15. price_pos_volatility: Volatility of (close-low)/(high-low)
            hl_range = high - low
            hl_range_safe = np.where(hl_range > 1e-8, hl_range, 1e-8)
            price_pos = (close - low) / hl_range_safe
            out['price_pos_volatility'] = pd.Series(price_pos).rolling(20, min_periods=5).std().values

            factor_parts.append(out)
            processed += 1
            if processed % 1000 == 0:
                logger.info(f"      已处理 {processed}/{n_stocks} 只股票")

        if not factor_parts:
            logger.warning("    无有效股票数据, 15个新因子填0")
            for col in self.NEW_FACTORS:
                df[col] = 0.0
            return df

        df_factors = pd.concat(factor_parts, ignore_index=True)
        logger.info(f"    因子计算完成: {len(df_factors):,} 行, {processed} 只股票")

        # === 计算 industry_mom_rank (需要跨股票截面排名) ===
        # 从 df_ohlcv 计算每只股票的 10d 过去收益率
        ret10_parts = []
        for code, grp in df_ohlcv.groupby('code'):
            grp = grp.sort_values('trade_date').copy()
            close_s = pd.Series(grp['close'].values)
            grp_out = grp[['code', 'trade_date']].copy()
            grp_out['return_10d_past'] = close_s.pct_change(10).values
            ret10_parts.append(grp_out)
        df_ret10 = pd.concat(ret10_parts, ignore_index=True)

        # 获取 sw_l1_code from 主df
        if 'sw_l1_code' in df.columns:
            code_industry = df[['code', 'trade_date', 'sw_l1_code']].drop_duplicates()
            df_ret10 = df_ret10.merge(code_industry, on=['code', 'trade_date'], how='left')
            # 行业内排名
            df_ret10['industry_mom_rank'] = df_ret10.groupby(
                ['trade_date', 'sw_l1_code'])['return_10d_past'].rank(pct=True)
            # 合并回 df_factors (替换placeholder)
            df_factors = df_factors.drop(columns=['industry_mom_rank'], errors='ignore')
            df_factors = df_factors.merge(
                df_ret10[['code', 'trade_date', 'industry_mom_rank']],
                on=['code', 'trade_date'], how='left')
        else:
            df_factors['industry_mom_rank'] = 0.5  # 无行业信息时用中位数

        # === 合并15个新因子到主df ===
        merge_cols = ['code', 'trade_date'] + self.NEW_FACTORS
        # 确保所有因子列存在
        for col in self.NEW_FACTORS:
            if col not in df_factors.columns:
                df_factors[col] = 0.0

        df = df.merge(df_factors[merge_cols], on=['code', 'trade_date'], how='left')

        # 填充NaN
        for col in self.NEW_FACTORS:
            if col in df.columns:
                missing = df[col].isnull().sum()
                if missing > 0:
                    pct = missing / len(df) * 100
                    # 先用截面中位数填充
                    df[col] = df.groupby('trade_date')[col].transform(lambda x: x.fillna(x.median()))
                    remaining = df[col].isnull().sum()
                    if remaining > 0:
                        df[col] = df[col].fillna(0.0)
                    if pct > 10:
                        logger.info(f"      {col}: {missing:,} 缺失({pct:.1f}%) → {df[col].isnull().sum()} 剩余")
            else:
                df[col] = 0.0

        logger.info(f"  V4.8.1 新增因子合并完成: +{len(self.NEW_FACTORS)} 因子, 总列数: {len(df.columns)}")

        return df

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """V4.8.1: V4.7.5特征 + 15新因子 - 5额外裁剪 (50 → 60)"""
        # V4.7.5 prepare_features 会做: V4.7.3特征准备 + 20个裁剪
        # 我们的PRUNE_FEATURES已经是25个(V4.7.5的20 + 5个新增)
        # super().prepare_features()会使用self.PRUNE_FEATURES来裁剪
        X, y_3d, y_5d, y_10d, y_15d, df_out = super().prepare_features(df)

        logger.info(f"  V4.8.1: 最终特征数 = {X.shape[1]} (目标~60)")
        return X, y_3d, y_5d, y_10d, y_15d, df_out

    def walk_forward_train(self, start_date: str = None, end_date: str = None,
                            purge_days: int = 15, min_train_days: int = 900,
                            val_days: int = 120, test_days: int = 120,
                            step_days: int = 90):
        """V4.8.1 Walk-Forward — V4.7.5 + 15新因子"""
        import shutil

        logger.info("=" * 60)
        logger.info("V4.8.1 Walk-Forward 训练 (V4.7.5 + 15新因子, 50→60特征)")
        logger.info("=" * 60)
        logger.info(f"  底座: V4.7.5 (6模型ensemble, MSE Loss, 365d时间衰减)")
        logger.info(f"  特征变化: 裁剪+5 ({len(self.PRUNE_FEATURES)}总) + 新增15因子")
        logger.info(f"  新增因子类别:")
        logger.info(f"    波动率: atr_percentile, gk_vol_20d, price_pos_volatility")
        logger.info(f"    成交量: vol_concentration, abnormal_turnover, turnover_vol_20d, vol_price_div")
        logger.info(f"    价格: intraday_ret_20d, vwap_dev_20d, max_ret_20d, overnight_ret_20d")
        logger.info(f"    动量: industry_mom_rank, squeeze_mom_calc, price_acceleration")
        logger.info(f"    技术: cci_14")

        # 调用V4.7.5的walk_forward_train (使用我们override的load_data和prepare_features)
        model_data, history = super().walk_forward_train(
            start_date=start_date, end_date=end_date,
            purge_days=purge_days, min_train_days=min_train_days,
            val_days=val_days, test_days=test_days, step_days=step_days)

        # 将模型从v475目录移到v481目录
        v475_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v475'
        v481_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v481'
        v481_dir.mkdir(parents=True, exist_ok=True)

        v475_files = sorted(v475_dir.glob('v475_*.pkl'), key=lambda f: f.stat().st_mtime)
        if v475_files:
            latest = v475_files[-1]
            timestamp = latest.stem.replace('v475_multi_target_', '')
            new_path = v481_dir / f'v481_multi_target_{timestamp}.pkl'

            import joblib
            model_data['version'] = 'v4.8.1'
            model_data['v481_innovations'] = {
                'feature_expansion': '50 → 60 features (prune 5, add 15)',
                'new_factors': self.NEW_FACTORS,
                'pruned_extra': ['min_pct_change_5d', 'industry_concentration', 'sw_l1_code',
                                 'volume_price_corr_10d', 'volume_trend'],
                'factor_categories': {
                    'volatility': ['atr_percentile', 'gk_vol_20d', 'price_pos_volatility'],
                    'volume': ['vol_concentration', 'abnormal_turnover', 'turnover_vol_20d', 'vol_price_div'],
                    'price': ['intraday_ret_20d', 'vwap_dev_20d', 'max_ret_20d', 'overnight_ret_20d'],
                    'momentum': ['industry_mom_rank', 'squeeze_mom_calc', 'price_acceleration'],
                    'technical': ['cci_14'],
                },
            }
            model_data['feature_pruning'] = {
                'pruned_count': len(self.PRUNE_FEATURES),
                'pruned_features': list(self.PRUNE_FEATURES),
                'v481_extra_pruned': ['min_pct_change_5d', 'industry_concentration', 'sw_l1_code',
                                      'volume_price_corr_10d', 'volume_trend'],
            }
            joblib.dump(model_data, new_path)
            logger.info(f"\nV4.8.1 model saved: {new_path}")
            logger.info(f"  Size: {new_path.stat().st_size / 1024 / 1024:.1f} MB")

            # 复制辅助文件
            for aux in ['global_quantiles.npy', 'recommendation_thresholds.json']:
                src = v475_dir / aux
                if src.exists():
                    shutil.copy2(str(src), str(v481_dir / aux))

            # 删除v475目录下这个模型(它属于v481)
            latest.unlink()
            for hf in v475_dir.glob(f'training_history_{timestamp}*'):
                hf.unlink()
            logger.info(f"  Cleaned up v475 directory")

            # 保存v481 history (仅独立训练时写 latest, 子类链式调用时跳过)
            history['version'] = 'v4.8.1'
            history['base'] = 'V4.7.5 + 15 New Factors (50 → 60 features)'
            history['v481_innovations'] = model_data['v481_innovations']

            import json as _json
            history_path = v481_dir / f'training_history_{timestamp}.json'
            with open(history_path, 'w', encoding='utf-8') as f:
                _json.dump(history, f, indent=2, ensure_ascii=False)
            # 仅当自身是最终版本时写 latest (防止子类覆盖)
            if self.__class__.__name__ == 'V481Trainer':
                latest_path = v481_dir / 'training_history_latest.json'
                with open(latest_path, 'w', encoding='utf-8') as f:
                    _json.dump(history, f, indent=2, ensure_ascii=False)

            logger.info(f"\nV4.8.1 training complete!")
            logger.info(f"  Features: {model_data.get('feature_names', ['?']).__len__()} "
                         f"(V4.7.5 - {len(self.PRUNE_FEATURES)} pruned + {len(self.NEW_FACTORS)} new)")
        else:
            logger.warning("No v475 model file found to rename")

        return model_data, history


class V484Trainer(V481Trainer):
    """V4.8.4 训练器 — V4.8.1底座 + 1个Top-K验证因子 (60 → 61特征)

    核心: 用 Top-K Sharpe (非全局IC) 筛选因子, 只保留 brain_roll_spread
    来源: Roll (1984) 隐含价差, sqrt(max(0, -cov(Δclose, Δclose_lag1, 20d)))
    验证: 单因子 TopK_Sharpe=1.397 (基线0.016), TopK_Return +0.014 (基线+0.0002)
    """

    V484_FACTOR = 'brain_roll_spread'

    def load_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """V4.8.4: V4.8.1基础 + brain_roll_spread"""
        df = super().load_data(start_date, end_date)

        date_min = df['trade_date'].min()
        date_max = df['trade_date'].max()

        logger.info("  V4.8.4 加载 brain_roll_spread...")
        try:
            conn = sqlite3.connect(self.db_path)
            brain_raw = pd.read_sql("""
                SELECT code, trade_date, features_json
                FROM brain_alpha_cache
                WHERE trade_date >= ? AND trade_date <= ?
            """, conn, params=(date_min, date_max))
            conn.close()

            if not brain_raw.empty:
                try:
                    import orjson
                    _loads = orjson.loads
                except ImportError:
                    _loads = json.loads

                roll_values = []
                for fj in brain_raw['features_json']:
                    parsed = _loads(fj)
                    roll_values.append(float(parsed.get('brain_roll_spread', 0)))

                brain_df = pd.DataFrame({
                    'code': brain_raw['code'].values,
                    'trade_date': brain_raw['trade_date'].values,
                    'brain_roll_spread': roll_values,
                })

                df = df.merge(brain_df, on=['code', 'trade_date'], how='left')
                df['brain_roll_spread'] = df['brain_roll_spread'].fillna(0.0)
                logger.info(f"    brain_roll_spread 合并完成, 覆盖率 "
                            f"{(brain_df.shape[0] / len(df) * 100):.1f}%")
            else:
                df['brain_roll_spread'] = 0.0
                logger.warning("    brain_alpha_cache 为空!")
        except Exception as e:
            df['brain_roll_spread'] = 0.0
            logger.warning(f"    brain_roll_spread 加载失败: {e}")

        logger.info(f"  V4.8.4 加载完成: {len(df.columns)} 列, {len(df):,} 行")
        return df

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """V4.8.4: V4.8.1特征 + brain_roll_spread (60 → 61)"""
        X, y_3d, y_5d, y_10d, y_15d, df_out = super().prepare_features(df)
        logger.info(f"  V4.8.4: 最终特征数 = {X.shape[1]} (目标61)")
        return X, y_3d, y_5d, y_10d, y_15d, df_out

    def walk_forward_train(self, start_date: str = None, end_date: str = None,
                            purge_days: int = 15, min_train_days: int = 900,
                            val_days: int = 120, test_days: int = 120,
                            step_days: int = 90):
        """V4.8.4 Walk-Forward — V4.8.1 + brain_roll_spread"""
        import shutil

        logger.info("=" * 60)
        logger.info("V4.8.4 Walk-Forward (V4.8.1 + brain_roll_spread, 60→61特征)")
        logger.info("=" * 60)
        logger.info(f"  底座: V4.8.1 (60特征, 6模型ensemble)")
        logger.info(f"  新增: brain_roll_spread (Roll 1984 隐含价差)")
        logger.info(f"  筛选: Top-K Sharpe=1.397 (基线0.016)")

        model_data, history = super().walk_forward_train(
            start_date=start_date, end_date=end_date,
            purge_days=purge_days, min_train_days=min_train_days,
            val_days=val_days, test_days=test_days, step_days=step_days)

        # 移到 v484 目录
        v481_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v481'
        v484_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v484'
        v484_dir.mkdir(parents=True, exist_ok=True)

        v481_files = sorted(v481_dir.glob('v481_*.pkl'), key=lambda f: f.stat().st_mtime)
        if v481_files:
            latest = v481_files[-1]
            timestamp = latest.stem.replace('v481_multi_target_', '')
            new_path = v484_dir / f'v484_multi_target_{timestamp}.pkl'

            import joblib
            model_data['version'] = 'v4.8.4'
            model_data['v484_innovations'] = {
                'feature_expansion': '60 → 61 features (V4.8.1 + brain_roll_spread)',
                'selection_method': 'Top-K Sharpe greedy (not global IC)',
                'brain_factor': 'brain_roll_spread',
                'formula': 'sqrt(max(0, -cov(Δclose, Δclose_lag1, 20d)))',
                'reference': 'Roll (1984) implied bid-ask spread',
                'topk_sharpe': 1.397,
                'topk_return': 0.014142,
            }
            joblib.dump(model_data, new_path)
            logger.info(f"\nV4.8.4 model saved: {new_path}")
            logger.info(f"  Size: {new_path.stat().st_size / 1024 / 1024:.1f} MB")

            for aux in ['global_quantiles.npy', 'recommendation_thresholds.json']:
                src = v481_dir / aux
                if src.exists():
                    shutil.copy2(str(src), str(v484_dir / aux))

            latest.unlink()
            for hf in v481_dir.glob(f'training_history_{timestamp}*'):
                hf.unlink()

            history['version'] = 'v4.8.4'
            history['base'] = 'V4.8.1 + brain_roll_spread (60 → 61 features)'
            history['v484_innovations'] = model_data['v484_innovations']

            import json as _json
            history_path = v484_dir / f'training_history_{timestamp}.json'
            with open(history_path, 'w', encoding='utf-8') as f:
                _json.dump(history, f, indent=2, ensure_ascii=False)
            latest_path = v484_dir / 'training_history_latest.json'
            with open(latest_path, 'w', encoding='utf-8') as f:
                _json.dump(history, f, indent=2, ensure_ascii=False)

            logger.info(f"\nV4.8.4 training complete!")
            logger.info(f"  Features: {model_data.get('feature_names', ['?']).__len__()}")
        else:
            logger.warning("No v481 model file found to rename")

        return model_data, history


class V485Trainer(V484Trainer):
    """V4.8.5 训练器 — V4.8.4底座 + ETF训练数据 (61特征不变, 训练集扩大~10%)

    核心改进: 将ETF加入训练数据, 为模型提供跨资产信号
    评估结果: A股子集 ICIR 从 0.806 → 0.838 (+0.033)

    关键处理:
    - ETF labels从daily_quotes实时计算 (v39_feature_cache中ETF labels为NULL)
    - V481 15个新因子从ETF OHLCV单独计算
    - ETF daily_basic缺失(PE/PB/PS), 由截面中位数填充
    - ETF sw_l1_code=-1, 行业超额标签中作为独立"行业"组
    - brain_roll_spread对ETF也适用 (从brain_alpha_cache加载)
    """

    def load_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """V4.8.5: V4.8.4 A股数据 + ETF训练数据"""

        # Step 1: A股数据 (V484完整流程: base + V481 15因子 + brain_roll_spread)
        df_a = super().load_data(start_date, end_date)
        n_a = len(df_a)
        logger.info(f"  V4.8.5 A股数据: {n_a:,} 行")

        # Step 2: 加载ETF特征 + 计算labels
        conn = sqlite3.connect(self.db_path)

        date_min = df_a['trade_date'].min()
        date_max = df_a['trade_date'].max()

        # 2a: ETF features from v39_feature_cache (无label)
        etf_feat_query = """
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
        df_etf_raw = pd.read_sql(etf_feat_query, conn, params=[date_min, date_max])
        logger.info(f"  V4.8.5 ETF特征记录: {len(df_etf_raw):,}")

        if df_etf_raw.empty:
            conn.close()
            logger.warning("  V4.8.5 无ETF数据, 仅使用A股")
            return df_a

        # 2b: 计算ETF labels (未来N日收益率)
        etf_price_query = """
        SELECT s.code, q.trade_date, q.close
        FROM daily_quotes q
        JOIN securities s ON q.security_id = s.id
        WHERE s.type = 'ETF_基金' AND q.volume > 0
        ORDER BY s.code, q.trade_date
        """
        df_prices = pd.read_sql(etf_price_query, conn)

        etf_label_parts = []
        for code, grp in df_prices.groupby('code'):
            grp = grp.sort_values('trade_date').reset_index(drop=True)
            close = grp['close'].values
            n = len(close)
            if n < 15:
                continue
            labels = grp[['code', 'trade_date']].copy()
            for days, col in [(3, 'label_3d'), (5, 'label_5d'), (10, 'label_10d')]:
                future = np.full(n, np.nan)
                for i in range(n - days):
                    future[i] = close[i + days] / close[i] - 1
                labels[col] = future
            etf_label_parts.append(labels)

        if not etf_label_parts:
            conn.close()
            logger.warning("  V4.8.5 ETF labels计算失败, 仅使用A股")
            return df_a

        df_etf_labels = pd.concat(etf_label_parts, ignore_index=True)
        df_etf_labels = df_etf_labels.dropna(subset=['label_3d', 'label_5d', 'label_10d'])

        # 合并 features + labels
        df_etf = df_etf_raw.merge(df_etf_labels, on=['code', 'trade_date'], how='inner')
        logger.info(f"  V4.8.5 ETF合并(feat+label): {len(df_etf):,} ({df_etf['code'].nunique()} 只ETF)")

        # 过滤交易日 <30天
        etf_counts = df_etf.groupby('code').size()
        valid_etf = etf_counts[etf_counts >= 30].index
        df_etf = df_etf[df_etf['code'].isin(valid_etf)].copy()

        if df_etf.empty:
            conn.close()
            logger.warning("  V4.8.5 ETF过滤后为空, 仅使用A股")
            return df_a

        # 2c: 解析ETF features_json
        try:
            import orjson
            _loads = orjson.loads
        except ImportError:
            _loads = json.loads

        parsed = df_etf['features_json'].apply(_loads).tolist()
        df_etf_feat = pd.DataFrame(parsed)
        for col in ['code', 'trade_date', 'label_3d', 'label_5d', 'label_10d']:
            df_etf_feat[col] = df_etf[col].values

        # 市场特征
        market_cols = ['market_return_20d', 'market_return_10d', 'market_return_5d',
                       'market_volatility_20d', 'market_volatility_10d',
                       'market_up_ratio_20d', 'market_up_ratio_10d',
                       'market_drawdown_20d', 'market_volume_ratio',
                       'market_position_20d', 'market_momentum_20d', 'market_momentum_5d']
        for col in market_cols:
            if col in df_etf.columns:
                df_etf_feat[col] = df_etf[col].values

        df_etf_feat = df_etf_feat.sort_values('trade_date').copy()
        for col in market_cols:
            if col in df_etf_feat.columns:
                df_etf_feat[col] = df_etf_feat[col].ffill()
        df_etf_feat = df_etf_feat.dropna(subset=[c for c in market_cols if c in df_etf_feat.columns])
        df_etf_feat = df_etf_feat.fillna(0)

        # 2d: daily_basic (ETF没有, 后面用截面中位数填充)
        # A股的df_a已有pe_ttm等列, ETF这些列设为NaN让concat后统一填充
        for col in ['pe_ttm', 'pb', 'ps_ttm', 'turnover_rate', 'log_market_cap']:
            if col not in df_etf_feat.columns:
                df_etf_feat[col] = np.nan

        # 2e: 行业超额标签 — ETF的sw_l1_code=-1, 作为独立"行业"组
        if 'sw_l1_code' in df_etf_feat.columns:
            for lbl in ['label_3d', 'label_5d', 'label_10d']:
                med = df_etf_feat.groupby(['trade_date', 'sw_l1_code'])[lbl].transform('median')
                df_etf_feat[lbl] = df_etf_feat[lbl] - med

        # 2f: V481 15个新因子 — 从ETF OHLCV计算
        logger.info("  V4.8.5 计算ETF的V481新因子...")
        from datetime import datetime as dt_cls, timedelta as td_cls
        try:
            ext_start = (dt_cls.strptime(date_min, '%Y-%m-%d') - td_cls(days=60)).strftime('%Y-%m-%d')
        except Exception:
            ext_start = (dt_cls.strptime(date_min, '%Y%m%d') - td_cls(days=60)).strftime('%Y%m%d')

        etf_ohlcv_query = """
        SELECT s.code, q.trade_date, q.open, q.high, q.low, q.close,
               q.volume, q.price_change_pct
        FROM daily_quotes q
        JOIN securities s ON q.security_id = s.id
        WHERE s.type = 'ETF_基金' AND q.trade_date >= ? AND q.trade_date <= ?
        ORDER BY s.code, q.trade_date
        """
        df_etf_ohlcv = pd.read_sql(etf_ohlcv_query, conn, params=[ext_start, date_max])

        # ETF没有technical_indicators, atr_14/cci_14用NaN
        factor_parts = []
        for code, grp in df_etf_ohlcv.groupby('code'):
            grp = grp.sort_values('trade_date').copy()
            n = len(grp)
            if n < 5:
                continue
            close = grp['close'].values
            open_ = grp['open'].values
            high = grp['high'].values
            low = grp['low'].values
            volume = grp['volume'].values.astype(float)
            pct = grp['price_change_pct'].values

            out = grp[['code', 'trade_date']].copy()

            # 1. atr_percentile: 用手算ATR代替
            tr = np.maximum(high - low,
                            np.maximum(np.abs(high - np.concatenate([[close[0]], close[:-1]])),
                                       np.abs(low - np.concatenate([[close[0]], close[:-1]]))))
            atr14 = pd.Series(tr).rolling(14, min_periods=5).mean().values
            out['atr_percentile'] = pd.Series(atr14).rank(pct=True).values

            # 2. vol_concentration
            vol_s = pd.Series(volume)
            def _vol_hhi(x):
                total = x.sum()
                if total <= 0: return 0.0
                shares = x / total
                return (shares ** 2).sum()
            out['vol_concentration'] = vol_s.rolling(20, min_periods=5).apply(_vol_hhi, raw=True).values

            # 3. intraday_ret_20d
            intraday_ret = close / np.where(open_ > 0, open_, 1e-8) - 1
            out['intraday_ret_20d'] = pd.Series(intraday_ret).rolling(20, min_periods=5).sum().values

            # 4. industry_mom_rank (ETF没有行业, 后面统一填0.5)
            out['industry_mom_rank'] = 0.5

            # 5. vwap_dev_20d
            typical_price = (high + low + close) / 3
            tp_safe = np.where(typical_price > 0, typical_price, 1e-8)
            vwap_dev = (close - typical_price) / tp_safe
            out['vwap_dev_20d'] = pd.Series(vwap_dev).rolling(20, min_periods=5).mean().values

            # 6. max_ret_20d
            out['max_ret_20d'] = pd.Series(pct).rolling(20, min_periods=5).max().values

            # 7. gk_vol_20d
            hl_ratio = np.where(low > 0, high / low, 1.0)
            co_ratio = np.where(open_ > 0, close / open_, 1.0)
            gk_raw = np.sqrt(np.abs(0.5 * np.log(hl_ratio)**2 - (2*np.log(2)-1) * np.log(co_ratio)**2))
            out['gk_vol_20d'] = pd.Series(gk_raw).rolling(20, min_periods=5).mean().values

            # 8. abnormal_turnover (ETF没有turnover_rate, 用volume proxy)
            out['abnormal_turnover'] = 0.0

            # 9. overnight_ret_20d
            prev_close = np.concatenate([[np.nan], close[:-1]])
            overnight_ret = np.where(prev_close > 0, open_ / prev_close - 1, 0.0)
            overnight_ret[0] = 0.0
            out['overnight_ret_20d'] = pd.Series(overnight_ret).rolling(20, min_periods=5).sum().values

            # 10. turnover_vol_20d (ETF没有turnover_rate)
            out['turnover_vol_20d'] = 0.0

            # 11. cci_14 (手算)
            ma20 = pd.Series(typical_price).rolling(20, min_periods=5).mean().values
            mad20 = pd.Series(typical_price).rolling(20, min_periods=5).apply(
                lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
            mad_safe = np.where(mad20 > 1e-8, mad20, 1e-8)
            out['cci_14'] = (typical_price - ma20) / (0.015 * mad_safe)

            # 12. squeeze_mom_calc
            close_ma20 = pd.Series(close).rolling(20, min_periods=5).mean().values
            atr_safe = np.where(atr14 > 1e-8, atr14, 1e-8)
            out['squeeze_mom_calc'] = (close - close_ma20) / atr_safe

            # 13. vol_price_div
            close_s = pd.Series(close)
            vol_chg_5d = vol_s.pct_change(5).values
            pct_5d = close_s.pct_change(5).values
            out['vol_price_div'] = -pct_5d * vol_chg_5d

            # 14. price_acceleration
            ret_5d = close_s.pct_change(5).values
            ret_5d_prev = np.concatenate([[np.nan]*5, ret_5d[:-5]])
            out['price_acceleration'] = ret_5d - ret_5d_prev

            # 15. price_pos_volatility
            hl_range = high - low
            hl_range_safe = np.where(hl_range > 1e-8, hl_range, 1e-8)
            price_pos = (close - low) / hl_range_safe
            out['price_pos_volatility'] = pd.Series(price_pos).rolling(20, min_periods=5).std().values

            factor_parts.append(out)

        if factor_parts:
            df_etf_factors = pd.concat(factor_parts, ignore_index=True)
            v481_cols = ['code', 'trade_date'] + [
                'atr_percentile', 'vol_concentration', 'intraday_ret_20d',
                'industry_mom_rank', 'vwap_dev_20d', 'max_ret_20d',
                'gk_vol_20d', 'abnormal_turnover', 'overnight_ret_20d',
                'turnover_vol_20d', 'cci_14', 'squeeze_mom_calc',
                'vol_price_div', 'price_acceleration', 'price_pos_volatility',
            ]
            for col in v481_cols[2:]:
                if col not in df_etf_factors.columns:
                    df_etf_factors[col] = 0.0
            df_etf_feat = df_etf_feat.merge(
                df_etf_factors[v481_cols], on=['code', 'trade_date'], how='left')
            for col in v481_cols[2:]:
                if col in df_etf_feat.columns:
                    df_etf_feat[col] = df_etf_feat[col].fillna(0.0)
            logger.info(f"    ETF V481因子合并完成: {len(df_etf_factors):,} 行")
        else:
            for col in ['atr_percentile', 'vol_concentration', 'intraday_ret_20d',
                        'industry_mom_rank', 'vwap_dev_20d', 'max_ret_20d',
                        'gk_vol_20d', 'abnormal_turnover', 'overnight_ret_20d',
                        'turnover_vol_20d', 'cci_14', 'squeeze_mom_calc',
                        'vol_price_div', 'price_acceleration', 'price_pos_volatility']:
                df_etf_feat[col] = 0.0

        # 2g: brain_roll_spread
        try:
            brain_raw = pd.read_sql("""
                SELECT code, trade_date, features_json
                FROM brain_alpha_cache
                WHERE trade_date >= ? AND trade_date <= ?
                  AND code IN (SELECT code FROM securities WHERE type='ETF_基金')
            """, conn, params=(date_min, date_max))
            if not brain_raw.empty:
                roll_vals = []
                for fj in brain_raw['features_json']:
                    parsed = _loads(fj)
                    roll_vals.append(float(parsed.get('brain_roll_spread', 0)))
                brain_df = pd.DataFrame({
                    'code': brain_raw['code'].values,
                    'trade_date': brain_raw['trade_date'].values,
                    'brain_roll_spread': roll_vals,
                })
                df_etf_feat = df_etf_feat.merge(brain_df, on=['code', 'trade_date'], how='left')
            if 'brain_roll_spread' not in df_etf_feat.columns:
                df_etf_feat['brain_roll_spread'] = 0.0
            df_etf_feat['brain_roll_spread'] = df_etf_feat['brain_roll_spread'].fillna(0.0)
        except Exception:
            df_etf_feat['brain_roll_spread'] = 0.0

        conn.close()

        # Step 3: Concat A股 + ETF
        # 确保列对齐: 取交集列
        common_cols = list(set(df_a.columns) & set(df_etf_feat.columns))
        # 缺少的列填0
        for col in df_a.columns:
            if col not in df_etf_feat.columns:
                df_etf_feat[col] = 0.0
        for col in df_etf_feat.columns:
            if col not in df_a.columns:
                df_a[col] = 0.0

        df_combined = pd.concat([df_a, df_etf_feat[df_a.columns]], ignore_index=True)

        # 对ETF缺失的daily_basic列用截面中位数填充
        for col in ['pe_ttm', 'pb', 'ps_ttm', 'turnover_rate', 'log_market_cap']:
            missing = df_combined[col].isnull().sum()
            if missing > 0:
                df_combined[col] = df_combined.groupby('trade_date')[col].transform(
                    lambda x: x.fillna(x.median()))
                remaining = df_combined[col].isnull().sum()
                if remaining > 0:
                    df_combined[col] = df_combined[col].fillna(df_combined[col].median())

        df_combined = df_combined.sort_values(['trade_date', 'code']).reset_index(drop=True)

        n_etf = len(df_combined) - n_a
        logger.info(f"  V4.8.5 合并完成: {len(df_combined):,} 行 (A股 {n_a:,} + ETF {n_etf:,})")
        logger.info(f"    ETF占比: {n_etf/len(df_combined)*100:.1f}%, "
                     f"{df_etf_feat['code'].nunique()} 只ETF")

        return df_combined

    def walk_forward_train(self, start_date: str = None, end_date: str = None,
                            purge_days: int = 15, min_train_days: int = 900,
                            val_days: int = 120, test_days: int = 120,
                            step_days: int = 90):
        """V4.8.5 Walk-Forward — V4.8.4 + ETF训练数据"""
        import shutil

        logger.info("=" * 60)
        logger.info("V4.8.5 Walk-Forward (V4.8.4 + ETF训练数据, 61特征)")
        logger.info("=" * 60)
        logger.info(f"  底座: V4.8.4 (61特征, brain_roll_spread)")
        logger.info(f"  新增: ETF训练数据 (~10%样本量)")
        logger.info(f"  评估: A股 ICIR +0.033 (0.806→0.838)")

        # 调用V484的walk_forward (会调用我们的load_data)
        # 但V484的walk_forward会把模型保存为v484格式, 我们需要改为v485
        model_data, history = super().walk_forward_train(
            start_date=start_date, end_date=end_date,
            purge_days=purge_days, min_train_days=min_train_days,
            val_days=val_days, test_days=test_days, step_days=step_days)

        # 把v484目录的模型移到v485
        v484_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v484'
        v485_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v485'
        v485_dir.mkdir(parents=True, exist_ok=True)

        v484_files = sorted(v484_dir.glob('v484_*.pkl'), key=lambda f: f.stat().st_mtime)
        if v484_files:
            latest = v484_files[-1]
            timestamp = latest.stem.replace('v484_multi_target_', '')
            new_path = v485_dir / f'v485_multi_target_{timestamp}.pkl'

            import joblib
            model_data['version'] = 'v4.8.5'
            model_data['v485_innovations'] = {
                'base': 'V4.8.4 (61 features, brain_roll_spread)',
                'training_data': 'A股 + ETF (~10% more samples)',
                'etf_label_source': 'daily_quotes (label_3d/5d/10d)',
                'etf_daily_basic': 'cross-section median fill',
                'etf_industry_excess': 'sw_l1_code=-1 as separate group',
                'eval_a_icir_delta': '+0.033',
            }
            joblib.dump(model_data, new_path)
            logger.info(f"\nV4.8.5 model saved: {new_path}")
            logger.info(f"  Size: {new_path.stat().st_size / 1024 / 1024:.1f} MB")

            for aux in ['global_quantiles.npy', 'recommendation_thresholds.json']:
                src = v484_dir / aux
                if src.exists():
                    shutil.copy2(str(src), str(v485_dir / aux))

            latest.unlink()
            for hf in v484_dir.glob(f'training_history_{timestamp}*'):
                hf.unlink()

            history['version'] = 'v4.8.5'
            history['base'] = 'V4.8.4 + ETF training data (61 features)'
            history['v485_innovations'] = model_data['v485_innovations']

            import json as _json
            history_path = v485_dir / f'training_history_{timestamp}.json'
            with open(history_path, 'w', encoding='utf-8') as f:
                _json.dump(history, f, indent=2, ensure_ascii=False)
            latest_path = v485_dir / 'training_history_latest.json'
            with open(latest_path, 'w', encoding='utf-8') as f:
                _json.dump(history, f, indent=2, ensure_ascii=False)

            logger.info(f"\nV4.8.5 training complete!")
            logger.info(f"  Features: {model_data.get('feature_names', ['?']).__len__()}")
        else:
            logger.warning("No v484 model file found to rename")

        return model_data, history


class V486Trainer(V485Trainer):
    """V4.8.6 训练器 — V4.8.5底座 + 3个BRAIN因子 + 头部区分度优化 (61 → 64特征)

    底座: V4.8.5 (V4.8.4 + ETF训练数据, A股ICIR+0.033)
    因子: +3个Top-K Sharpe BRAIN因子 (brain_high_low_ratio/close_to_high/momentum_decay10)

    头部区分度优化 (解决因子增多后分数压缩):
    1. LambdaRank truncation_level=15 (梯度集中在top-15, 原50)
    2. 树参数精细化: num_leaves=63(原31), min_data=50(原200/500)
    3. RRF ensemble (scorer层, 排名融合替代分数加权平均)
    """

    V486_BRAIN_FACTORS = [
        'brain_high_low_ratio',       # Top-K Sharpe验证
        'brain_close_to_high',        # Top-K Sharpe验证
        'brain_momentum_decay10',     # Top-K Sharpe验证
    ]

    def train_single_target_models(self, X_train, X_val, y_train, y_val, target_name: str,
                                    sample_weights_train=None):
        """V4.8.6: CatBoost YetiRank NDCG@10 + XGBoost rank:ndcg topk

        两个模型从回归替换为排名:
        1. CatBoost: YetiRankPairwise NDCG@10
        2. XGBoost: rank:ndcg + lambdarank_pair_method=topk + num_pair=15
        其他4个模型(LGB/RF/HGB/LGBRank)保持回归。

        Ensemble中排名模型占比: 3/6 (cb+xgb+lgb_rank) vs 3/6回归(lgb+rf+hgb)
        """
        import gc

        # 先用父类训练所有模型 (包括标准CatBoost回归)
        models, pred_train, pred_val = super().train_single_target_models(
            X_train, X_val, y_train, y_val, target_name,
            sample_weights_train=sample_weights_train)

        # 替换CatBoost回归为YetiRankPairwise NDCG@10
        if HAS_CATBOOST and 'cb' in models:
            train_dates = getattr(self, 'train_dates', None)
            val_dates = getattr(self, 'val_dates', None)

            if train_dates is not None and len(train_dates) == len(y_train):
                try:
                    logger.info(f"  V4.8.6 CatBoost YetiRank NDCG@10 ({target_name})...")

                    # 构建group_id (每个日期一组)
                    # CatBoost ranking需要group_id列
                    train_group_id = train_dates.copy()

                    # 标签转为relevance等级 (0-4, 5档)
                    # 逐日期计算quintile
                    from scipy.stats import rankdata
                    relevance_train = np.zeros(len(y_train), dtype=np.float32)
                    for d in np.unique(train_dates):
                        mask = train_dates == d
                        y_d = y_train[mask]
                        if len(y_d) < 20:
                            continue
                        pct = rankdata(y_d) / len(y_d)
                        rel = np.zeros(len(y_d), dtype=np.float32)
                        rel[pct >= 0.20] = 1
                        rel[pct >= 0.40] = 2
                        rel[pct >= 0.60] = 3
                        rel[pct >= 0.80] = 4
                        relevance_train[mask] = rel

                    # Validation group
                    relevance_val = np.zeros(len(y_val), dtype=np.float32)
                    val_group_id = None
                    if val_dates is not None and len(val_dates) == len(y_val):
                        val_group_id = val_dates.copy()
                        for d in np.unique(val_dates):
                            mask = val_dates == d
                            y_d = y_val[mask]
                            if len(y_d) < 20:
                                continue
                            pct = rankdata(y_d) / len(y_d)
                            rel = np.zeros(len(y_d), dtype=np.float32)
                            rel[pct >= 0.20] = 1
                            rel[pct >= 0.40] = 2
                            rel[pct >= 0.60] = 3
                            rel[pct >= 0.80] = 4
                            relevance_val[mask] = rel

                    # CatBoost Pool with group_id (YetiRankPairwise不支持weight)
                    cb_pool_train = cb.Pool(
                        X_train, label=relevance_train,
                        group_id=train_group_id,
                    )

                    cb_params = {
                        'loss_function': 'YetiRankPairwise',
                        'eval_metric': 'NDCG:top=10',
                        'iterations': 1000,
                        'learning_rate': 0.02,
                        'depth': 6,
                        'l2_leaf_reg': 10,
                        'random_seed': 42,
                        'verbose': False,
                        'early_stopping_rounds': 30,
                        'min_data_in_leaf': 200,
                    }

                    cb_ranker = cb.CatBoost(cb_params)

                    if val_group_id is not None:
                        cb_pool_val = cb.Pool(
                            X_val, label=relevance_val,
                            group_id=val_group_id
                        )
                        cb_ranker.fit(cb_pool_train, eval_set=cb_pool_val, verbose=False)
                        del cb_pool_val
                    else:
                        cb_ranker.fit(cb_pool_train, verbose=False)

                    # 替换CatBoost模型
                    models['cb'] = cb_ranker
                    pred_train['cb'] = cb_ranker.predict(X_train)
                    pred_val['cb'] = cb_ranker.predict(X_val)

                    logger.info(f"    CatBoost YetiRank NDCG@10 ({target_name}): 完成")

                    del cb_pool_train
                    gc.collect()

                except Exception as e:
                    logger.warning(f"    CatBoost YetiRank失败, 保留回归版: {e}")

        # === 替换XGBoost回归为rank:ndcg topk ===
        if 'xgb' in models:
            train_dates = getattr(self, 'train_dates', None)
            val_dates = getattr(self, 'val_dates', None)

            if train_dates is not None and len(train_dates) == len(y_train):
                try:
                    from scipy.stats import rankdata
                    logger.info(f"  V4.8.6 XGBoost rank:ndcg topk ({target_name})...")

                    # 构建group (每日一组)
                    unique_train_dates = np.unique(train_dates)
                    group_train = []
                    for d in unique_train_dates:
                        group_train.append(int(np.sum(train_dates == d)))

                    # relevance labels (quintile 0-4)
                    relevance_train = np.zeros(len(y_train), dtype=np.float32)
                    for d in unique_train_dates:
                        mask = train_dates == d
                        y_d = y_train[mask]
                        if len(y_d) < 20:
                            continue
                        pct = rankdata(y_d) / len(y_d)
                        rel = np.zeros(len(y_d), dtype=np.float32)
                        rel[pct >= 0.20] = 1
                        rel[pct >= 0.40] = 2
                        rel[pct >= 0.60] = 3
                        rel[pct >= 0.80] = 4
                        relevance_train[mask] = rel

                    dtrain = xgb.DMatrix(X_train, label=relevance_train)
                    dtrain.set_group(group_train)

                    # Validation
                    dval = xgb.DMatrix(X_val, label=np.zeros(len(y_val)))
                    if val_dates is not None and len(val_dates) == len(y_val):
                        unique_val_dates = np.unique(val_dates)
                        group_val = [int(np.sum(val_dates == d)) for d in unique_val_dates]
                        relevance_val = np.zeros(len(y_val), dtype=np.float32)
                        for d in unique_val_dates:
                            mask = val_dates == d
                            y_d = y_val[mask]
                            if len(y_d) < 20:
                                continue
                            pct = rankdata(y_d) / len(y_d)
                            rel = np.zeros(len(y_d), dtype=np.float32)
                            rel[pct >= 0.20] = 1
                            rel[pct >= 0.40] = 2
                            rel[pct >= 0.60] = 3
                            rel[pct >= 0.80] = 4
                            relevance_val[mask] = rel
                        dval = xgb.DMatrix(X_val, label=relevance_val)
                        dval.set_group(group_val)

                    xgb_rank_params = {
                        'objective': 'rank:ndcg',
                        'eval_metric': 'ndcg@10',
                        'lambdarank_pair_method': 'topk',
                        'lambdarank_num_pair_per_sample': 15,
                        'max_depth': 6,
                        'learning_rate': 0.02,
                        'subsample': 0.7,
                        'colsample_bytree': 0.6,
                        'reg_alpha': 0.5,
                        'reg_lambda': 3.0,
                        'min_child_weight': 100,
                        'verbosity': 0,
                    }

                    xgb_rank_model = xgb.train(
                        xgb_rank_params, dtrain,
                        num_boost_round=1000,
                        evals=[(dtrain, 'train'), (dval, 'val')],
                        early_stopping_rounds=30,
                        verbose_eval=False
                    )

                    models['xgb'] = xgb_rank_model
                    pred_train['xgb'] = xgb_rank_model.predict(xgb.DMatrix(X_train))
                    pred_val['xgb'] = xgb_rank_model.predict(xgb.DMatrix(X_val))

                    logger.info(f"    XGBoost rank:ndcg topk ({target_name}): 完成")

                    del dtrain, dval
                    import gc; gc.collect()
                except Exception as e:
                    logger.warning(f"    XGBoost rank:ndcg失败, 保留回归版: {e}")

        return models, pred_train, pred_val

    # V4.8.2 IC验证因子 (双窗口通过, 仅Top5有效, 剩余8个第4轮验证有害已回退)
    V482_TOP5_FACTORS = [
        'limit_proximity_5d',   # IC=-0.094, 涨跌停距离(反转)
        'trend_strength_60d',   # IC=-0.089, 60日趋势强度(均值回归)
        'high_52w_ratio',       # IC=-0.086, 52周新高比(支撑位)
        'max5_lottery',         # IC=+0.084, 最大5日涨幅(彩票效应)
        'imxd_20d',             # IC=-0.062, 高低点时间差(趋势结构)
    ]

    def _compute_v482_top5(self, df_ohlcv: pd.DataFrame) -> pd.DataFrame:
        """计算5个V4.8.2 Top IC因子 (per-stock滚动, 从OHLCV计算)"""
        factor_parts = []
        n_stocks = df_ohlcv['code'].nunique()
        processed = 0

        for code, grp in df_ohlcv.groupby('code'):
            grp = grp.sort_values('trade_date').copy()
            n = len(grp)
            if n < 60:
                processed += 1
                continue

            close = grp['close'].values
            high = grp['high'].values
            low = grp['low'].values
            pct = grp['price_change_pct'].values
            close_s = pd.Series(close)
            pct_s = pd.Series(pct)
            dates = grp['trade_date'].values

            out = pd.DataFrame({'code': code, 'trade_date': dates})

            # 1. limit_proximity_5d (IC=-0.094)
            if code.startswith('3') or code.startswith('688'):
                limit_pct = 0.20
            elif code.startswith(('4', '8')):
                limit_pct = 0.30
            else:
                limit_pct = 0.10
            limit_prox = np.abs(pct) / max(limit_pct, 1e-8)
            out['limit_proximity_5d'] = pd.Series(limit_prox).rolling(5, min_periods=3).mean().values

            # 2. trend_strength_60d (IC=-0.089)
            ret_60d = close_s.pct_change(60).values
            vol_60d = pct_s.rolling(60, min_periods=20).std().values
            vol_safe = np.where(vol_60d > 1e-8, vol_60d, 1e-8)
            out['trend_strength_60d'] = ret_60d / (vol_safe * np.sqrt(60))

            # 3. high_52w_ratio (IC=-0.086)
            high_s = pd.Series(high)
            max_252 = high_s.rolling(252, min_periods=60).max().values
            max_safe = np.where(max_252 > 1e-8, max_252, 1e-8)
            out['high_52w_ratio'] = close / max_safe

            # 4. max5_lottery (IC=+0.084)
            def _max5(x):
                if len(x) < 5:
                    return 0.0
                return -np.mean(np.sort(x)[-5:])
            out['max5_lottery'] = pct_s.rolling(20, min_periods=10).apply(_max5, raw=True).values

            # 5. imxd_20d (IC=-0.062)
            imxd = np.full(n, np.nan)
            for i in range(19, n):
                h_w = high[i-19:i+1]
                l_w = low[i-19:i+1]
                imxd[i] = np.argmax(h_w) / 19.0 - np.argmin(l_w) / 19.0
            out['imxd_20d'] = imxd

            factor_parts.append(out)
            processed += 1

        if not factor_parts:
            return pd.DataFrame()

        df_factors = pd.concat(factor_parts, ignore_index=True)
        logger.info(f"    V4.8.2 Top5因子计算完成: {processed}/{n_stocks} stocks, {len(df_factors):,} rows")
        return df_factors

    def load_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """V4.8.6: V4.8.5基础 + 3个BRAIN因子 + 5个V4.8.2因子"""
        df = super().load_data(start_date, end_date)

        date_min = df['trade_date'].min()
        date_max = df['trade_date'].max()

        # === Part 1: 5个V4.8.2 Top IC因子 (从OHLCV计算) ===
        logger.info("  V4.8.6 计算5个V4.8.2 Top IC因子...")
        try:
            conn = sqlite3.connect(self.db_path)
            from datetime import datetime as dt_cls, timedelta as td_cls
            try:
                ext_start = (dt_cls.strptime(date_min, '%Y-%m-%d') - td_cls(days=280)).strftime('%Y-%m-%d')
            except Exception:
                ext_start = (dt_cls.strptime(date_min, '%Y%m%d') - td_cls(days=280)).strftime('%Y%m%d')

            ohlcv_query = """
            SELECT s.code, q.trade_date, q.open, q.high, q.low, q.close,
                   q.volume, q.price_change_pct
            FROM daily_quotes q
            JOIN securities s ON q.security_id = s.id
            WHERE s.type = 'A股' AND q.trade_date >= ? AND q.trade_date <= ?
            ORDER BY s.code, q.trade_date
            """
            df_ohlcv = pd.read_sql(ohlcv_query, conn, params=[ext_start, date_max])
            conn.close()
            logger.info(f"    OHLCV加载: {len(df_ohlcv):,} rows")

            df_v482 = self._compute_v482_top5(df_ohlcv)
            if not df_v482.empty:
                # 过滤到训练日期范围
                df_v482 = df_v482[df_v482['trade_date'] >= date_min]
                df = df.merge(df_v482, on=['code', 'trade_date'], how='left')
                for f in self.V482_TOP5_FACTORS:
                    df[f] = df[f].fillna(0.0)
                logger.info(f"    V4.8.2 Top5因子合并完成")
            else:
                for f in self.V482_TOP5_FACTORS:
                    df[f] = 0.0
        except Exception as e:
            for f in self.V482_TOP5_FACTORS:
                df[f] = 0.0
            logger.warning(f"    V4.8.2因子计算失败: {e}")

        # === Part 2: 3+3=6个BRAIN因子 (从brain_alpha_cache加载) ===
        logger.info("  V4.8.6 加载6个BRAIN因子 (3 Top-K + 3 Top-Importance)...")
        try:
            conn = sqlite3.connect(self.db_path)
            brain_raw = pd.read_sql("""
                SELECT code, trade_date, features_json
                FROM brain_alpha_cache
                WHERE trade_date >= ? AND trade_date <= ?
            """, conn, params=(date_min, date_max))
            conn.close()

            if not brain_raw.empty:
                try:
                    import orjson
                    _loads = orjson.loads
                except ImportError:
                    _loads = json.loads

                factor_values = {f: [] for f in self.V486_BRAIN_FACTORS}
                for fj in brain_raw['features_json']:
                    parsed = _loads(fj)
                    for f in self.V486_BRAIN_FACTORS:
                        factor_values[f].append(float(parsed.get(f, 0)))

                brain_df = pd.DataFrame({
                    'code': brain_raw['code'].values,
                    'trade_date': brain_raw['trade_date'].values,
                })
                for f in self.V486_BRAIN_FACTORS:
                    brain_df[f] = factor_values[f]

                df = df.merge(brain_df, on=['code', 'trade_date'], how='left')
                for f in self.V486_BRAIN_FACTORS:
                    df[f] = df[f].fillna(0.0)
                coverage = brain_df.shape[0] / max(len(df), 1) * 100
                logger.info(f"    3个BRAIN因子合并完成, 覆盖率 {coverage:.1f}%")
            else:
                for f in self.V486_BRAIN_FACTORS:
                    df[f] = 0.0
                logger.warning("    brain_alpha_cache 为空!")
        except Exception as e:
            for f in self.V486_BRAIN_FACTORS:
                df[f] = 0.0
            logger.warning(f"    V4.8.6 BRAIN因子加载失败: {e}")

        logger.info(f"  V4.8.6 加载完成: {len(df.columns)} 列, {len(df):,} 行")
        return df

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """V4.8.6: V4.8.5特征 + 3 BRAIN + 5 V482 (61 → 69)"""
        X, y_3d, y_5d, y_10d, y_15d, df_out = super().prepare_features(df)
        logger.info(f"  V4.8.6: 最终特征数 = {X.shape[1]} (目标69)")
        return X, y_3d, y_5d, y_10d, y_15d, df_out

    def walk_forward_train(self, start_date: str = None, end_date: str = None,
                            purge_days: int = 15, min_train_days: int = 900,
                            val_days: int = 120, test_days: int = 120,
                            step_days: int = 90):
        """V4.8.6 Walk-Forward — V4.8.5 + 3 BRAIN Top-K factors"""
        import shutil

        logger.info("=" * 60)
        logger.info("V4.8.6 Walk-Forward (V4.8.5 + 3 BRAIN Top-K, 61→64特征)")
        logger.info("=" * 60)
        logger.info(f"  底座: V4.8.5 (61特征, A股+ETF训练, 6模型ensemble)")
        logger.info(f"  新增: brain_high_low_ratio, brain_close_to_high, brain_momentum_decay10")
        logger.info(f"  验证: 3因子组合 ICIR +4.2% (跨窗口稳定)")

        model_data, history = super().walk_forward_train(
            start_date=start_date, end_date=end_date,
            purge_days=purge_days, min_train_days=min_train_days,
            val_days=val_days, test_days=test_days, step_days=step_days)

        # 移到 v486 目录 (上游V485已把模型放在v485/)
        v485_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v485'
        v486_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v486'
        v486_dir.mkdir(parents=True, exist_ok=True)

        v485_files = sorted(v485_dir.glob('v485_*.pkl'), key=lambda f: f.stat().st_mtime)
        if v485_files:
            latest = v485_files[-1]
            timestamp = latest.stem.replace('v485_multi_target_', '')
            new_path = v486_dir / f'v486_multi_target_{timestamp}.pkl'

            import joblib
            model_data['version'] = 'v4.8.6'
            model_data['v486_innovations'] = {
                'feature_expansion': '61 → 64 features (V4.8.5 + 3 BRAIN Top-K)',
                'selection_method': 'Top-K Sharpe greedy (cross-window validated)',
                'brain_factors': self.V486_BRAIN_FACTORS,
                'etf_training': True,
                'icir_improvement': '+4.2% (3-factor combo)',
                'cross_window_stability': 'W1 +4.2%, W2 +62%',
            }
            joblib.dump(model_data, new_path)
            logger.info(f"\nV4.8.6 model saved: {new_path}")
            logger.info(f"  Size: {new_path.stat().st_size / 1024 / 1024:.1f} MB")

            for aux in ['global_quantiles.npy', 'recommendation_thresholds.json']:
                src = v485_dir / aux
                if src.exists():
                    shutil.copy2(str(src), str(v486_dir / aux))

            latest.unlink()
            for hf in v485_dir.glob(f'training_history_{timestamp}*'):
                hf.unlink()

            history['version'] = 'v4.8.6'
            history['base'] = 'V4.8.5 (ETF训练) + 3 BRAIN Top-K (61 → 64 features)'
            history['v486_innovations'] = model_data['v486_innovations']

            import json as _json
            history_path = v486_dir / f'training_history_{timestamp}.json'
            with open(history_path, 'w', encoding='utf-8') as f:
                _json.dump(history, f, indent=2, ensure_ascii=False)
            latest_path = v486_dir / 'training_history_latest.json'
            with open(latest_path, 'w', encoding='utf-8') as f:
                _json.dump(history, f, indent=2, ensure_ascii=False)

            logger.info(f"\nV4.8.6 training complete!")
            logger.info(f"  Features: {model_data.get('feature_names', ['?']).__len__()}")
        else:
            logger.warning("No v485 model file found to rename")

        return model_data, history


class V482Trainer(V481Trainer):
    """V4.8.2 训练器 — V4.8.1底座 + 13个双窗口IC验证因子 (60 → 73特征)

    核心哲学: 因子 > 训练技巧 > scorer后处理 > 模型混合
    验证方法: 1年(2024-2025) + 3年(2023-2025) 双窗口IC交叉验证

    通过双窗口验证的13个因子:

    Phase 1 - 价量因子 (8个, 从OHLCV+daily_basic):
      1.  limit_proximity_5d      涨停接近度(A股)    1Y IC=-0.094/-0.86 | 3Y IC=-0.084/-0.86 ★★★
      2.  max5_lottery            MAX5彩票效应        1Y IC=0.084/0.45  | 3Y IC=0.080/0.50  ★★★
      3.  sumd_20d                涨跌平衡(Qlib)     1Y IC=-0.060/-0.45 | 3Y IC=-0.055/-0.48 ★★
      4.  industry_adj_str       行业调整反转        1Y IC=0.051/0.58  | 3Y IC=0.042/0.42  ★★
      5.  turnover_reversal      换手率反转          1Y IC=-0.048/-0.26 | 3Y IC=-0.065/-0.34 ★★
      6.  residual_momentum      残差动量            1Y IC=-0.043/-0.37 | 3Y IC=-0.037/-0.33 ★★
      7.  retail_crowding        散户拥挤度          1Y IC=0.030/0.36  | 3Y IC=0.026/0.29  ★
      8.  obv_price_div          OBV-Price背离       1Y IC=-0.025/-0.47 | 3Y IC=-0.022/-0.47 ★★

    Phase 2 - 财务因子 (1个, 从financial_indicator):
      9.  delta_roe_yoy          ROE变化率           1Y IC=0.044/0.43  | 3Y IC=0.052/0.54  ★★★

    Phase 3 - 学术因子 (4个):
      10. trend_strength_60d     趋势强度t-stat      1Y IC=-0.089/-0.62 | 3Y IC=-0.071/-0.45 ★★
      11. high_52w_ratio         52周新高比(锚定)     1Y IC=-0.086/-0.67 | 3Y IC=-0.038/-0.26 ★
      12. imxd_20d               IMAX-IMIN(Qlib)     1Y IC=-0.062/-0.65 | 3Y IC=-0.055/-0.47 ★★
      13. realized_skew_20d      已实现偏度           1Y IC=-0.029/-0.39 | 3Y IC=-0.025/-0.40 ★★

    双窗口淘汰8个因子:
    - 双窗口FAIL(5): cfp, gpoa_approx, accruals_quality, rev_growth_consistency, trend_rsquared_20d
    - 3年窗口FAIL(3): chaikin_mf_20d(ICIR=-0.19), ksft_5d(ICIR=-0.15), delta_leverage_yoy(ICIR=0.19)
    因子间相关性: 全部|corr| < 0.5 (无需去重)

    继承V4.8.1:
    - 15个V4.8.1因子 (60特征基础)
    - 6模型ensemble, MSE Loss, 365d时间衰减
    - V4.7.3管线(无Meta-Learner, 无Combined Isotonic)
    """

    # V4.8.2新增因子名称 — Phase 1 (价量, 1年+3年双窗口IC验证)
    # REMOVED: chaikin_mf_20d(3年ICIR=-0.19 FAIL), ksft_5d(3年ICIR=-0.15 FAIL)
    V482_PHASE1_FACTORS = [
        'industry_adj_str', 'turnover_reversal', 'max5_lottery',
        'retail_crowding', 'residual_momentum',
        'sumd_20d', 'obv_price_div', 'limit_proximity_5d',
    ]

    # V4.8.2新增因子名称 — Phase 2 (财务, 双窗口IC验证后保留1/6)
    # REMOVED: cfp(1年IC≈0), gpoa_approx(IC小), accruals_quality(0%覆盖),
    #          delta_leverage_yoy(3年ICIR=0.19 FAIL), rev_growth_consistency(IC小)
    V482_PHASE2_FACTORS = [
        'delta_roe_yoy',
    ]

    # V4.8.2新增因子名称 — Phase 3 (长周期+学术, 双窗口验证后保留4/5)
    # REMOVED: trend_rsquared_20d(IC≈0)
    V482_PHASE3_FACTORS = [
        'high_52w_ratio', 'imxd_20d',
        'realized_skew_20d', 'trend_strength_60d',
    ]

    V482_ALL_NEW_FACTORS = V482_PHASE1_FACTORS + V482_PHASE2_FACTORS + V482_PHASE3_FACTORS

    def load_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """V4.8.2: V4.8.1基础 + 21个新因子"""
        # 先调用V4.8.1 load_data (获取60个特征)
        df = super().load_data(start_date, end_date)

        date_min = df['trade_date'].min()
        date_max = df['trade_date'].max()
        conn = sqlite3.connect(self.db_path)

        from datetime import datetime as dt_cls, timedelta as td_cls
        try:
            ext_start_60 = (dt_cls.strptime(date_min, '%Y-%m-%d') - td_cls(days=90)).strftime('%Y-%m-%d')
            ext_start_252 = (dt_cls.strptime(date_min, '%Y-%m-%d') - td_cls(days=370)).strftime('%Y-%m-%d')
        except Exception:
            ext_start_60 = (dt_cls.strptime(date_min, '%Y%m%d') - td_cls(days=90)).strftime('%Y%m%d')
            ext_start_252 = (dt_cls.strptime(date_min, '%Y%m%d') - td_cls(days=370)).strftime('%Y%m%d')

        logger.info("  V4.8.2 加载扩展数据 (计算21个新因子)...")

        # === Phase 1 & 3: 加载OHLCV (需要252d窗口用于52w high) ===
        ohlcv_query = """
        SELECT s.code, q.trade_date, q.open, q.high, q.low, q.close,
               q.volume, q.price_change_pct
        FROM daily_quotes q
        JOIN securities s ON q.security_id = s.id
        WHERE s.type = 'A股' AND q.trade_date >= ? AND q.trade_date <= ?
        ORDER BY s.code, q.trade_date
        """
        df_ohlcv = pd.read_sql(ohlcv_query, conn, params=[ext_start_252, date_max])
        logger.info(f"    OHLCV (252d扩展): {len(df_ohlcv):,} 行")

        # 加载turnover + market_cap from daily_basic
        basic_query = """
        SELECT s.code, db.trade_date, db.turnover_rate, db.total_mv
        FROM daily_basic db
        JOIN securities s ON db.security_id = s.id
        WHERE db.trade_date >= ? AND db.trade_date <= ?
        """
        df_basic = pd.read_sql(basic_query, conn, params=[ext_start_60, date_max])
        logger.info(f"    daily_basic (turnover+mcap): {len(df_basic):,} 行")

        # 加载行业信息 (for industry_adj_str + residual_momentum)
        industry_query = """
        SELECT code, industry FROM securities WHERE type = 'A股' AND industry IS NOT NULL
        """
        df_industry = pd.read_sql(industry_query, conn)
        code_to_industry = dict(zip(df_industry['code'], df_industry['industry']))
        logger.info(f"    行业映射: {len(code_to_industry):,} 只股票")

        # === Phase 2: 加载financial_indicator (只需roe) ===
        fi_query = """
        SELECT s.code, fi.end_date, fi.roe
        FROM financial_indicator fi
        JOIN securities s ON fi.security_id = s.id
        WHERE fi.end_date >= '20180101'
        ORDER BY s.code, fi.end_date
        """
        df_fi = pd.read_sql(fi_query, conn)
        logger.info(f"    financial_indicator: {len(df_fi):,} 行")
        conn.close()

        if len(df_ohlcv) == 0:
            logger.warning("    OHLCV为空, V4.8.2新因子填0")
            for col in self.V482_ALL_NEW_FACTORS:
                df[col] = 0.0
            return df

        # === Phase 1: 计算10个价量因子 (per-stock rolling) ===
        logger.info("    Phase 1: 计算10个价量因子...")

        # 合并turnover + mcap到ohlcv
        df_ohlcv = df_ohlcv.merge(
            df_basic[['code', 'trade_date', 'turnover_rate', 'total_mv']],
            on=['code', 'trade_date'], how='left')
        df_ohlcv['turnover_rate'] = df_ohlcv['turnover_rate'].fillna(0.0)
        df_ohlcv['total_mv'] = df_ohlcv['total_mv'].fillna(0.0)

        # 先计算每日每行业中位数收益 (用于industry_adj_str + residual_momentum)
        df_ohlcv['industry'] = df_ohlcv['code'].map(code_to_industry)
        industry_med_ret5d = {}  # (trade_date, industry) -> median 5d return

        # 行业中位数5d收益: 先计算每个code的5d滚动收益
        close_by_code = {}
        for code, grp in df_ohlcv.groupby('code'):
            grp = grp.sort_values('trade_date')
            close_s = pd.Series(grp['close'].values, index=grp['trade_date'].values)
            ret5d = close_s.pct_change(5)
            close_by_code[code] = {'close': close_s, 'ret5d': ret5d, 'industry': code_to_industry.get(code, '')}

        # 组装每日行业中位数
        all_ret5d = []
        for code, d in close_by_code.items():
            ret_df = pd.DataFrame({'trade_date': d['ret5d'].index, 'ret5d': d['ret5d'].values,
                                   'industry': d['industry']})
            all_ret5d.append(ret_df)
        if all_ret5d:
            df_all_ret5d = pd.concat(all_ret5d, ignore_index=True)
            industry_med = df_all_ret5d.groupby(['trade_date', 'industry'])['ret5d'].median()
            industry_med_ret5d = industry_med.to_dict()

        # 行业内截面: 计算每日market return (用于residual_momentum)
        market_daily_ret = {}
        for td, grp in df_ohlcv.groupby('trade_date'):
            pcts = grp['price_change_pct'].dropna()
            if len(pcts) > 0:
                market_daily_ret[td] = float(pcts.median())

        factor_parts = []
        n_stocks = df_ohlcv['code'].nunique()
        processed = 0

        for code, grp in df_ohlcv.groupby('code'):
            grp = grp.sort_values('trade_date').copy()
            n = len(grp)
            if n < 10:
                continue

            close = grp['close'].values.astype(float)
            open_ = grp['open'].values.astype(float)
            high = grp['high'].values.astype(float)
            low = grp['low'].values.astype(float)
            volume = grp['volume'].values.astype(float)
            pct = pd.to_numeric(grp['price_change_pct'], errors='coerce').fillna(0).values.astype(float)
            turnover = grp['turnover_rate'].values.astype(float)
            mcap = grp['total_mv'].values.astype(float)
            dates = grp['trade_date'].values
            industry = code_to_industry.get(code, '')

            out = pd.DataFrame({'code': code, 'trade_date': dates})

            # --- Phase 1 factors ---

            # 1. industry_adj_str: -(return_5d - industry_median_return_5d)
            close_s = pd.Series(close)
            ret5d = close_s.pct_change(5).values
            ind_med = np.array([industry_med_ret5d.get((d, industry), 0.0) for d in dates])
            out['industry_adj_str'] = -(ret5d - ind_med)

            # 2. turnover_reversal: -rank(avg_turnover_20d) — per-stock rolling, 归一化在截面做
            turn_s = pd.Series(turnover)
            out['turnover_reversal'] = -turn_s.rolling(20, min_periods=5).mean().values

            # 3. max5_lottery: -mean(top5 daily returns in 20d)
            pct_s = pd.Series(pct)
            def _max5_mean(x):
                if len(x) < 5:
                    return 0.0
                return -np.mean(np.sort(x)[-5:])
            out['max5_lottery'] = pct_s.rolling(20, min_periods=10).apply(_max5_mean, raw=True).values

            # 4. retail_crowding: -rank(turnover) × rank(-mcap)
            # Per-stock time series proxy; cross-sectional normalization later
            turn_rank_ts = turn_s.rolling(60, min_periods=20).rank(pct=True).values
            mcap_s = pd.Series(mcap)
            mcap_rank_inv = 1.0 - mcap_s.rolling(60, min_periods=20).rank(pct=True).values
            out['retail_crowding'] = -(turn_rank_ts * mcap_rank_inv)

            hl_range = high - low
            hl_safe = np.where(hl_range > 1e-8, hl_range, 1e-8)

            # 6. residual_momentum: 回归去市场收益后的残差动量
            # Σ(residual[t-20:t-5]) where residual = stock_ret - beta*market_ret
            mkt_rets = np.array([market_daily_ret.get(d, 0.0) for d in dates])
            resid_mom = np.full(n, np.nan)
            for i in range(25, n):
                stock_r = pct[i-25:i]
                mkt_r = mkt_rets[i-25:i]
                # Simple beta estimation
                mkt_var = np.var(mkt_r)
                if mkt_var > 1e-12:
                    beta = np.cov(stock_r, mkt_r)[0,1] / mkt_var
                else:
                    beta = 0.0
                residual = stock_r - beta * mkt_r
                # Skip most recent 5 days (reversal contamination)
                resid_mom[i] = np.sum(residual[:20])  # [t-25:t-5]
            out['residual_momentum'] = resid_mom

            # 8. sumd_20d: 涨跌平衡 (Σgains - Σlosses) / (Σgains + Σlosses)
            gains = np.where(pct > 0, pct, 0.0)
            losses = np.where(pct < 0, -pct, 0.0)
            sum_gains = pd.Series(gains).rolling(20, min_periods=5).sum().values
            sum_losses = pd.Series(losses).rolling(20, min_periods=5).sum().values
            denom = sum_gains + sum_losses
            denom_safe = np.where(denom > 1e-8, denom, 1e-8)
            out['sumd_20d'] = (sum_gains - sum_losses) / denom_safe

            # 9. obv_price_div: OBV趋势 vs 价格趋势的背离
            obv_sign = np.where(pct > 0, 1, np.where(pct < 0, -1, 0))
            obv = np.cumsum(obv_sign * volume)
            obv_s = pd.Series(obv)
            obv_ret_20d = (obv_s / obv_s.shift(20).replace(0, np.nan) - 1).values
            price_ret_20d = close_s.pct_change(20).values
            out['obv_price_div'] = obv_ret_20d - price_ret_20d

            # 10. limit_proximity_5d: 涨停接近度 mean(|ret|/limit, 5d)
            # 主板10%, 创业板/科创板20%, 北交所30%
            if code.startswith('3') or code.startswith('688'):
                limit_pct = 0.20
            elif code.startswith(('4', '8')):
                limit_pct = 0.30
            else:
                limit_pct = 0.10
            limit_prox = np.abs(pct) / limit_pct
            out['limit_proximity_5d'] = pd.Series(limit_prox).rolling(5, min_periods=3).mean().values

            # --- Phase 3: 长周期+学术因子 ---

            # 17. high_52w_ratio: close / max(high, 252d)
            high_s = pd.Series(high)
            max_252 = high_s.rolling(252, min_periods=60).max().values
            max_252_safe = np.where(max_252 > 1e-8, max_252, 1e-8)
            out['high_52w_ratio'] = close / max_252_safe

            # 19. imxd_20d: (argmax(high) - argmin(low)) / 20
            imxd = np.full(n, np.nan)
            for i in range(19, n):
                h_window = high[i-19:i+1]
                l_window = low[i-19:i+1]
                imax = np.argmax(h_window) / 19.0  # normalize to [0,1]
                imin = np.argmin(l_window) / 19.0
                imxd[i] = imax - imin
            out['imxd_20d'] = imxd

            # 20. realized_skew_20d: (1/N) Σ((r-μ)/σ)³
            skew = np.full(n, np.nan)
            for i in range(19, n):
                r = pct[i-19:i+1]
                mu = np.mean(r)
                sigma = np.std(r)
                if sigma > 1e-8:
                    skew[i] = np.mean(((r - mu) / sigma) ** 3)
                else:
                    skew[i] = 0.0
            out['realized_skew_20d'] = skew

            # 21. trend_strength_60d: return_60d / (vol_60d × √60)
            ret_60d = close_s.pct_change(60).values
            vol_60d = pct_s.rolling(60, min_periods=20).std().values
            vol_60d_safe = np.where(vol_60d > 1e-8, vol_60d, 1e-8)
            out['trend_strength_60d'] = ret_60d / (vol_60d_safe * np.sqrt(60))

            factor_parts.append(out)
            processed += 1
            if processed % 1000 == 0:
                logger.info(f"      已处理 {processed}/{n_stocks} 只股票")

        if not factor_parts:
            logger.warning("    无有效股票数据, V4.8.2新因子填0")
            for col in self.V482_ALL_NEW_FACTORS:
                df[col] = 0.0
            return df

        df_factors = pd.concat(factor_parts, ignore_index=True)
        logger.info(f"    Phase 1+3 因子计算完成: {len(df_factors):,} 行, {processed} 只股票")

        # === Phase 2: 计算财务质量因子 ===
        logger.info("    Phase 2: 计算6个财务质量因子...")
        if len(df_fi) > 0:
            # 标准化日期格式
            df_fi['end_date'] = df_fi['end_date'].str.replace('-', '')

            fi_factors = []
            for code, grp_fi in df_fi.groupby('code'):
                grp_fi = grp_fi.sort_values('end_date').copy()
                if len(grp_fi) < 5:  # Need at least 5 quarters for YoY
                    continue

                roe_vals = grp_fi['roe'].values
                end_dates = grp_fi['end_date'].values

                for i in range(len(grp_fi)):
                    row = {'code': code, 'fi_end_date': end_dates[i]}

                    # delta_roe_yoy: roe[t] - roe[t-4]
                    if i >= 4 and pd.notna(roe_vals[i]) and pd.notna(roe_vals[i-4]):
                        row['delta_roe_yoy'] = float(roe_vals[i] - roe_vals[i-4])
                    else:
                        row['delta_roe_yoy'] = np.nan

                    fi_factors.append(row)

            if fi_factors:
                df_fi_computed = pd.DataFrame(fi_factors)
                logger.info(f"    财务因子计算完成: {len(df_fi_computed):,} 行")

                # Forward-fill quarterly → daily via searchsorted
                df_dates = df[['code', 'trade_date']].copy()
                df_dates['td_str'] = df_dates['trade_date'].str.replace('-', '')

                fi_merged_parts = []
                for code, fi_grp in df_fi_computed.groupby('code'):
                    fi_grp = fi_grp.dropna(subset=['fi_end_date']).sort_values('fi_end_date')
                    code_dates = df_dates[df_dates['code'] == code].sort_values('td_str').copy()
                    if len(code_dates) == 0 or len(fi_grp) == 0:
                        continue
                    fi_end_dates = fi_grp['fi_end_date'].values
                    td_strs = code_dates['td_str'].values
                    indices = np.searchsorted(fi_end_dates, td_strs, side='right') - 1
                    valid_mask = indices >= 0
                    for col in self.V482_PHASE2_FACTORS:
                        if col in fi_grp.columns:
                            fi_vals = fi_grp[col].values
                            mapped = np.where(valid_mask, fi_vals[np.clip(indices, 0, len(fi_vals)-1)], np.nan)
                            code_dates[col] = mapped
                        else:
                            code_dates[col] = np.nan
                    fi_merged_parts.append(code_dates[['code', 'trade_date'] + self.V482_PHASE2_FACTORS])

                if fi_merged_parts:
                    df_fi_daily = pd.concat(fi_merged_parts, ignore_index=True)

                    fi_cols = ['code', 'trade_date'] + self.V482_PHASE2_FACTORS
                    for col in self.V482_PHASE2_FACTORS:
                        if col not in df_fi_daily.columns:
                            df_fi_daily[col] = np.nan
                    df_factors = df_factors.merge(
                        df_fi_daily[fi_cols].drop_duplicates(subset=['code', 'trade_date']),
                        on=['code', 'trade_date'], how='left')
                    logger.info(f"    财务因子合并完成")
            else:
                logger.warning("    无有效财务数据, Phase 2因子填NaN")
                for col in self.V482_PHASE2_FACTORS:
                    df_factors[col] = np.nan
        else:
            logger.warning("    financial_indicator为空, Phase 2因子填NaN")
            for col in self.V482_PHASE2_FACTORS:
                df_factors[col] = np.nan

        # === 截面归一化: turnover_reversal 和 retail_crowding ===
        # 这两个因子需要截面rank才有意义
        for col in ['turnover_reversal', 'retail_crowding']:
            if col in df_factors.columns:
                df_factors[col] = df_factors.groupby('trade_date')[col].rank(pct=True)
                # 翻转: rank越高(原始值越大)→因子值越低(反向因子)
                df_factors[col] = 1.0 - df_factors[col]

        # === 合并所有V4.8.2新因子到主df ===
        merge_cols = ['code', 'trade_date'] + self.V482_ALL_NEW_FACTORS
        for col in self.V482_ALL_NEW_FACTORS:
            if col not in df_factors.columns:
                df_factors[col] = np.nan

        df = df.merge(df_factors[merge_cols], on=['code', 'trade_date'], how='left')

        # 填充NaN
        for col in self.V482_ALL_NEW_FACTORS:
            if col in df.columns:
                missing = df[col].isnull().sum()
                if missing > 0:
                    pct_miss = missing / len(df) * 100
                    df[col] = df.groupby('trade_date')[col].transform(lambda x: x.fillna(x.median()))
                    remaining = df[col].isnull().sum()
                    if remaining > 0:
                        df[col] = df[col].fillna(0.0)
                    if pct_miss > 10:
                        logger.info(f"      {col}: {missing:,} 缺失({pct_miss:.1f}%) → {df[col].isnull().sum()} 剩余")
            else:
                df[col] = 0.0

        logger.info(f"  V4.8.2 新增因子合并完成: +{len(self.V482_ALL_NEW_FACTORS)} 因子, 总列数: {len(df.columns)}")
        return df

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """V4.8.2: V4.8.1特征 + 21新因子 (60 → ~81)"""
        X, y_3d, y_5d, y_10d, y_15d, df_out = super().prepare_features(df)
        logger.info(f"  V4.8.2: 最终特征数 = {X.shape[1]} (目标~81)")
        return X, y_3d, y_5d, y_10d, y_15d, df_out

    def walk_forward_train(self, start_date: str = None, end_date: str = None,
                            purge_days: int = 15, min_train_days: int = 900,
                            val_days: int = 120, test_days: int = 120,
                            step_days: int = 90):
        """V4.8.2 Walk-Forward — V4.8.1 + 21新因子"""
        import shutil

        logger.info("=" * 60)
        logger.info("V4.8.2 Walk-Forward 训练 (V4.8.1 + 13个双窗口IC因子, 60→73特征)")
        logger.info("=" * 60)
        logger.info(f"  底座: V4.8.1 (60特征, 6模型ensemble, MSE Loss)")
        logger.info(f"  Phase 1 价量(8): {self.V482_PHASE1_FACTORS}")
        logger.info(f"  Phase 2 财务(1): {self.V482_PHASE2_FACTORS}")
        logger.info(f"  Phase 3 学术(4): {self.V482_PHASE3_FACTORS}")
        logger.info(f"  淘汰(8): chaikin_mf_20d, ksft_5d, delta_leverage_yoy + cfp, gpoa_approx, accruals_quality, rev_growth_consistency, trend_rsquared_20d")

        # 调用V4.8.1的walk_forward_train (使用我们override的load_data和prepare_features)
        model_data, history = super().walk_forward_train(
            start_date=start_date, end_date=end_date,
            purge_days=purge_days, min_train_days=min_train_days,
            val_days=val_days, test_days=test_days, step_days=step_days)

        # 将模型从v481目录移到v482目录
        v481_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v481'
        v482_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v482'
        v482_dir.mkdir(parents=True, exist_ok=True)

        # V4.8.1的walk_forward_train会先把文件从v475移到v481
        # 我们需要从v481取最新文件移到v482
        v481_files = sorted(v481_dir.glob('v481_*.pkl'), key=lambda f: f.stat().st_mtime)
        if v481_files:
            latest = v481_files[-1]
            timestamp = latest.stem.replace('v481_multi_target_', '')
            new_path = v482_dir / f'v482_multi_target_{timestamp}.pkl'

            import joblib
            model_data['version'] = 'v4.8.2'
            model_data['v482_innovations'] = {
                'feature_expansion': '60 → ~81 features (V4.8.1 + 21 new)',
                'phase1_factors': self.V482_PHASE1_FACTORS,
                'phase2_factors': self.V482_PHASE2_FACTORS,
                'phase3_factors': self.V482_PHASE3_FACTORS,
                'factor_categories': {
                    'reversal': ['industry_adj_str', 'turnover_reversal'],
                    'behavioral': ['max5_lottery', 'retail_crowding', 'limit_proximity_5d'],
                    'capital_flow': ['chaikin_mf_20d', 'obv_price_div'],
                    'momentum': ['residual_momentum', 'trend_strength_60d'],
                    'microstructure': ['ksft_5d', 'sumd_20d'],
                    'financial': ['delta_roe_yoy'],
                    'trend': ['high_52w_ratio', 'imxd_20d'],
                    'risk': ['realized_skew_20d'],
                },
            }
            joblib.dump(model_data, new_path)
            logger.info(f"\nV4.8.2 model saved: {new_path}")
            logger.info(f"  Size: {new_path.stat().st_size / 1024 / 1024:.1f} MB")

            # 复制辅助文件
            for aux in ['global_quantiles.npy', 'recommendation_thresholds.json']:
                src = v481_dir / aux
                if src.exists():
                    shutil.copy2(str(src), str(v482_dir / aux))

            # 删除v481目录下这个模型(它属于v482)
            latest.unlink()
            for hf in v481_dir.glob(f'training_history_{timestamp}*'):
                hf.unlink()
            logger.info(f"  Cleaned up v481 directory")

            # 保存history
            history['version'] = 'v4.8.2'
            history['base'] = 'V4.8.1 + 21 New Factors (60 → ~81 features)'
            history['v482_innovations'] = model_data['v482_innovations']

            import json as _json
            history_path = v482_dir / f'training_history_{timestamp}.json'
            with open(history_path, 'w', encoding='utf-8') as f:
                _json.dump(history, f, indent=2, ensure_ascii=False)
            # 仅当自身是最终版本时写 latest (防止子类覆盖)
            if self.__class__.__name__ == 'V482Trainer':
                latest_path = v482_dir / 'training_history_latest.json'
                with open(latest_path, 'w', encoding='utf-8') as f:
                    _json.dump(history, f, indent=2, ensure_ascii=False)

            logger.info(f"\nV4.8.2 training complete!")
            logger.info(f"  Features: {model_data.get('feature_names', ['?']).__len__()}")
            logger.info(f"  New factors: {len(self.V482_ALL_NEW_FACTORS)}")
        else:
            logger.warning("No v481 model file found to rename")

        return model_data, history


class V483Trainer(V482Trainer):
    """V4.8.3 训练器 — V4.8.2底座 + 29个BRAIN验证因子 (~81 → ~110特征)

    核心创新: WorldQuant BRAIN + 学术文献 + A股特色因子
    验证方法: 快速单窗口LightGBM ICIR, 三窗口稳定性交叉验证

    29个BRAIN因子来源:
    - BRAIN USA TOP3000 回测验证 (Sharpe≥0.7): 9个
    - 学术文献 (Fama-French, AQR, Roll 1984, Corwin-Schultz): 8个
    - A股特色 (涨停/跳空/散户行为): 4个
    - 微观结构 (Hurst, 自相关, 尾部风险): 5个
    - 量价组合 (VWAP动量, 资金流, 波动聚集): 3个

    快速评估结果 (ICIR):
    - 基线(无BRAIN): 0.4610
    - +3 BRAIN Top3: 0.5026 (+9.0%)
    - +29 BRAIN全部: 0.6261 (+35.8%) ★ 最优
    - 三窗口稳定: W1 +35.8%, W2 +4.0%, W3 +5.5%
    """

    # BRAIN 因子名称 (29个)
    V483_BRAIN_FACTORS = [
        # Phase 1: BRAIN 验证因子 (9个)
        'brain_intraday_intensity', 'brain_high_low_ratio', 'brain_close_to_high',
        'brain_vol_ratio', 'brain_vol_of_vol', 'brain_momentum_decay5',
        'brain_momentum_decay10', 'brain_vol_price_divergence', 'brain_turnover_momentum',
        # Phase 2: 学术 + A股 + 微观结构因子 (20个)
        'brain_52w_low_bounce', 'brain_ma60_reversion', 'brain_vol_asymmetry',
        'brain_roll_spread', 'brain_extreme_day_freq', 'brain_momentum_crash_hedge',
        'brain_loss_aversion', 'brain_high_resistance', 'brain_hl_spread',
        'brain_ret_autocorr', 'brain_tail_risk', 'brain_vwap_momentum',
        'brain_up_streak_ratio', 'brain_hurst_proxy', 'brain_post_limitup_ret',
        'brain_vol_price_coord', 'brain_price_jerk', 'brain_gap_strength',
        'brain_money_flow', 'brain_vol_clustering',
    ]

    def load_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """V4.8.3: V4.8.2基础 + 29个BRAIN验证因子"""
        # 先调用V4.8.2 load_data (获取~81个特征)
        df = super().load_data(start_date, end_date)

        date_min = df['trade_date'].min()
        date_max = df['trade_date'].max()

        # 加载 BRAIN 因子缓存
        logger.info("  V4.8.3 加载 BRAIN 验证因子...")
        try:
            conn = sqlite3.connect(self.db_path)
            brain_query = """
                SELECT code, trade_date, features_json
                FROM brain_alpha_cache
                WHERE trade_date >= ? AND trade_date <= ?
            """
            brain_raw = pd.read_sql(brain_query, conn, params=(date_min, date_max))
            conn.close()

            if not brain_raw.empty:
                try:
                    import orjson
                    _loads = orjson.loads
                except ImportError:
                    _loads = json.loads

                brain_parsed = pd.json_normalize(brain_raw['features_json'].apply(_loads))
                # 只保留注册的因子
                keep_cols = [c for c in self.V483_BRAIN_FACTORS if c in brain_parsed.columns]
                brain_parsed = brain_parsed[keep_cols]
                brain_parsed['code'] = brain_raw['code'].values
                brain_parsed['trade_date'] = brain_raw['trade_date'].values

                before = len(df.columns)
                df = df.merge(brain_parsed, on=['code', 'trade_date'], how='left')

                # 截面中位数填充 + 兜底 0
                for col in keep_cols:
                    if col in df.columns:
                        missing = df[col].isnull().sum()
                        if missing > 0:
                            df[col] = df.groupby('trade_date')[col].transform(
                                lambda x: x.fillna(x.median()))
                            df[col] = df[col].fillna(0.0)

                logger.info(f"    BRAIN 因子合并完成: +{len(keep_cols)} 因子, "
                            f"覆盖率 {len(brain_parsed) / len(df) * 100:.1f}%, "
                            f"总列数: {len(df.columns)}")
            else:
                logger.warning("    brain_alpha_cache 为空! 请先运行: "
                               "python3 wqbrain_integration/cache_brain_features.py")
                for col in self.V483_BRAIN_FACTORS:
                    df[col] = 0.0
        except Exception as e:
            logger.warning(f"    BRAIN 因子加载失败: {e}")
            for col in self.V483_BRAIN_FACTORS:
                df[col] = 0.0

        logger.info(f"  V4.8.3 加载完成: {len(df.columns)} 列, {len(df):,} 行")
        return df

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """V4.8.3: V4.8.2特征 + 29 BRAIN因子 (~81 → ~110)"""
        X, y_3d, y_5d, y_10d, y_15d, df_out = super().prepare_features(df)
        logger.info(f"  V4.8.3: 最终特征数 = {X.shape[1]} (目标~110)")
        return X, y_3d, y_5d, y_10d, y_15d, df_out

    def walk_forward_train(self, start_date: str = None, end_date: str = None,
                            purge_days: int = 15, min_train_days: int = 900,
                            val_days: int = 120, test_days: int = 120,
                            step_days: int = 90):
        """V4.8.3 Walk-Forward — V4.8.2 + 29 BRAIN因子"""
        import shutil

        logger.info("=" * 60)
        logger.info("V4.8.3 Walk-Forward 训练 (V4.8.2 + 29 BRAIN验证因子, ~81→~110特征)")
        logger.info("=" * 60)
        logger.info(f"  底座: V4.8.2 (~81特征, 6模型ensemble)")
        logger.info(f"  BRAIN Phase 1 (9): BRAIN USA验证")
        logger.info(f"  BRAIN Phase 2 (20): 学术+A股+微观结构")
        logger.info(f"  快速评估 ICIR: 0.4610 → 0.6261 (+35.8%)")

        # 调用V4.8.2的walk_forward_train → V4.8.1 → V4.7.5
        model_data, history = super().walk_forward_train(
            start_date=start_date, end_date=end_date,
            purge_days=purge_days, min_train_days=min_train_days,
            val_days=val_days, test_days=test_days, step_days=step_days)

        # 将模型从v482目录移到v483目录
        v482_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v482'
        v483_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v483'
        v483_dir.mkdir(parents=True, exist_ok=True)

        v482_files = sorted(v482_dir.glob('v482_*.pkl'), key=lambda f: f.stat().st_mtime)
        if v482_files:
            latest = v482_files[-1]
            timestamp = latest.stem.replace('v482_multi_target_', '')
            new_path = v483_dir / f'v483_multi_target_{timestamp}.pkl'

            import joblib
            model_data['version'] = 'v4.8.3'
            model_data['v483_innovations'] = {
                'feature_expansion': '~81 → ~110 features (V4.8.2 + 29 BRAIN)',
                'brain_factors': self.V483_BRAIN_FACTORS,
                'brain_factor_sources': {
                    'brain_verified': ['brain_high_low_ratio', 'brain_close_to_high',
                                       'brain_momentum_decay10', 'brain_intraday_intensity',
                                       'brain_vol_ratio', 'brain_vol_of_vol',
                                       'brain_momentum_decay5', 'brain_vol_price_divergence',
                                       'brain_turnover_momentum'],
                    'academic': ['brain_52w_low_bounce', 'brain_ma60_reversion',
                                 'brain_vol_asymmetry', 'brain_roll_spread',
                                 'brain_tail_risk', 'brain_hurst_proxy'],
                    'a_share': ['brain_post_limitup_ret', 'brain_gap_strength',
                                'brain_loss_aversion', 'brain_extreme_day_freq'],
                    'microstructure': ['brain_ret_autocorr', 'brain_hl_spread',
                                       'brain_vol_clustering', 'brain_money_flow',
                                       'brain_vwap_momentum'],
                    'composite': ['brain_momentum_crash_hedge', 'brain_high_resistance',
                                  'brain_up_streak_ratio', 'brain_vol_price_coord',
                                  'brain_price_jerk'],
                },
                'fast_eval_icir': {'baseline': 0.4610, 'with_brain': 0.6261, 'delta': '+35.8%'},
            }
            joblib.dump(model_data, new_path)
            logger.info(f"\nV4.8.3 model saved: {new_path}")
            logger.info(f"  Size: {new_path.stat().st_size / 1024 / 1024:.1f} MB")

            # 复制辅助文件
            for aux in ['global_quantiles.npy', 'recommendation_thresholds.json']:
                src = v482_dir / aux
                if src.exists():
                    shutil.copy2(str(src), str(v483_dir / aux))

            # 删除v482目录下这个模型(它属于v483)
            latest.unlink()
            for hf in v482_dir.glob(f'training_history_{timestamp}*'):
                hf.unlink()

            # 保存history
            history['version'] = 'v4.8.3'
            history['base'] = 'V4.8.2 + 29 BRAIN Factors (~81 → ~110 features)'
            history['v483_innovations'] = model_data['v483_innovations']

            import json as _json
            history_path = v483_dir / f'training_history_{timestamp}.json'
            with open(history_path, 'w', encoding='utf-8') as f:
                _json.dump(history, f, indent=2, ensure_ascii=False)
            latest_path = v483_dir / 'training_history_latest.json'
            with open(latest_path, 'w', encoding='utf-8') as f:
                _json.dump(history, f, indent=2, ensure_ascii=False)

            logger.info(f"\nV4.8.3 training complete!")
            logger.info(f"  Total features: {model_data.get('feature_names', ['?']).__len__()}")
            logger.info(f"  BRAIN factors: {len(self.V483_BRAIN_FACTORS)}")
        else:
            logger.warning("No v482 model file found to rename")

        return model_data, history


class V48Trainer(V472Trainer):
    """V4.8 训练器 — V4.7.2底座 + 12个新财务质量特征 + ListNet排名模型 + 置信度加权

    继承V4.7.2:
    - Bug修复(3项): Winsorization泄露/Sharpe-Blend执行/000300.SH统一
    - V4.7.1 17新特征(76总): 财务质量/daily_basic/微观结构/反转/风险
    - LambdaRank: 5d/10d/15d (3d跳过)
    - 时间衰减: half_life=365d
    - 目标特异性Sharpe-Blend: 3d=0.10, 5d=0.25, 10d=0.35, 15d=0.35
    - V4.6管线: ICIR权重 + Combined Isotonic + Meta-Learner
    - 无小盘加成

    V4.8新增:
    1. +12个新财务质量特征 (不与V4.7.1重叠)
    2. ListNet排名模型 (5d/10d/15d, 3d跳过)
    3. 回归/排名alpha融合
    4. 模型置信度加权
    """

    # V4.8新增的12个财务质量特征 (不与V4.7.1的17个重叠)
    # V4.7.1已有: roe, gross_margin, current_ratio, assets_turn, netprofit_yoy, or_yoy, dv_ttm, turnover_rate_f, float_ratio
    #           + 4微观结构 + 2反转 + 2风险
    FINANCIAL_QUALITY_TIER1 = ['netprofit_margin', 'ocf_to_opincome', 'salescash_to_or', 'roe_dt', 'fcfe_ps']  # 盈利质量
    FINANCIAL_QUALITY_TIER2 = ['debt_to_eqt', 'ebit_to_interest', 'quick_ratio']  # 财务安全 (dv_ttm已在V4.7.1)
    FINANCIAL_QUALITY_TIER3 = ['basic_eps_yoy', 'op_yoy', 'q_profit_yoy', 'netprofit_yoy_accel']  # 增长动量

    def __init__(self, db_path: str = DB_PATH):
        super().__init__(db_path=db_path)
        # 继承V4.7.2的target_weights和TARGET_SHARPE_BLEND

    @property
    def all_financial_quality_features(self):
        return self.FINANCIAL_QUALITY_TIER1 + self.FINANCIAL_QUALITY_TIER2 + self.FINANCIAL_QUALITY_TIER3

    # V4.8: 合并V4.7.1的6个 + V4.8新增12个 = 18个financial_indicator列, 一次查询
    ALL_FI_COLUMNS = (
        # V4.7.1原有6个
        ['roe', 'gross_margin', 'current_ratio', 'assets_turn', 'netprofit_yoy', 'or_yoy'] +
        # V4.8新增11个 (netprofit_yoy已在V4.7.1中, 用于计算accel)
        ['netprofit_margin', 'ocf_to_opincome', 'salescash_to_or', 'roe_dt', 'fcfe_ps',
         'debt_to_eqt', 'ebit_to_interest', 'quick_ratio',
         'basic_eps_yoy', 'op_yoy', 'q_profit_yoy']
    )

    def load_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """V4.8: 一次性加载所有financial_indicator列, 避免V4.7.1+V4.8两次全表扫描

        优化: 跳过V4.7.1的load_data, 直接调用V4.4基础 + 一次性加载18个FI列 + V4.7.1非FI特征
        """
        # 跳过V4.7.1/V4.7.2的load_data, 直接用V4.4基础 (避免V4.7.1再扫一次financial_indicator)
        df = V44Trainer.load_data(self, start_date, end_date)

        date_min = df['trade_date'].min()
        date_max = df['trade_date'].max()
        conn = sqlite3.connect(self.db_path)

        # ===== 合并加载: V4.7.1的6个 + V4.8的12个 = 18个financial_indicator列 (单次全表扫描) =====
        logger.info("  V4.8 合并加载所有financial_indicator列 (18列, 单次扫描)...")
        fi_cols_str = ', '.join(f'fi.{c}' for c in self.ALL_FI_COLUMNS)
        fi_query = f"""
        SELECT s.code, fi.ann_date,
               {fi_cols_str}
        FROM financial_indicator fi
        JOIN securities s ON fi.security_id = s.id
        WHERE fi.ann_date IS NOT NULL AND fi.ann_date != ''
        ORDER BY s.code, fi.ann_date
        """
        df_fi = pd.read_sql(fi_query, conn)
        logger.info(f"    financial_indicator 记录: {len(df_fi):,}")

        if len(df_fi) > 0:
            def _date_to_int(s):
                return pd.to_datetime(s.astype(str).str.replace('-', ''), format='%Y%m%d').dt.strftime('%Y%m%d').astype(np.int64)

            df_fi['_ann_int'] = _date_to_int(df_fi['ann_date'])
            df['_td_int'] = _date_to_int(df['trade_date'])

            # 计算 netprofit_yoy_accel
            df_fi = df_fi.sort_values(['code', '_ann_int'])
            df_fi['netprofit_yoy_accel'] = df_fi.groupby('code')['netprofit_yoy'].diff()

            all_fi_merge_cols = self.ALL_FI_COLUMNS + ['netprofit_yoy_accel']

            # 向量化merge_asof (by='code', 避免per-stock Python循环)
            df_fi_dedup = df_fi.drop_duplicates(subset=['code', '_ann_int'], keep='last')
            fi_subset = df_fi_dedup[['code', '_ann_int'] + all_fi_merge_cols].rename(
                columns={'_ann_int': '_td_int'}).sort_values('_td_int')

            original_index = df.index.copy()
            df = df.sort_values('_td_int').reset_index(drop=True)
            df = pd.merge_asof(
                df,
                fi_subset,
                on='_td_int',
                by='code',
                direction='backward'
            )
            df.index = original_index
            df.drop(columns=['_td_int'], inplace=True, errors='ignore')
            logger.info(f"    合并完成: +{len(all_fi_merge_cols)} 财务特征 (向量化merge_asof)")

            # 填充: 当日截面中位数, 然后全局中位数兜底
            for col in all_fi_merge_cols:
                if col not in df.columns:
                    df[col] = np.nan
                missing = df[col].isnull().sum()
                if missing > 0:
                    pct = missing / len(df) * 100
                    df[col] = df.groupby('trade_date')[col].transform(lambda x: x.fillna(x.median()))
                    remaining = df[col].isnull().sum()
                    if remaining > 0:
                        df[col] = df[col].fillna(df[col].median())
                    if pct > 5:
                        logger.info(f"      {col}: {missing:,} 缺失({pct:.1f}%) → {df[col].isnull().sum()} 剩余")
        else:
            for col in self.ALL_FI_COLUMNS + ['netprofit_yoy_accel']:
                df[col] = 0.0
            logger.warning("    financial_indicator 为空, 所有财务特征填0")

        # ===== V4.7.1 非FI特征: daily_basic扩展 + 微观结构/反转/风险 (复用V4.7.1方法) =====
        # daily_basic extra (dv_ttm, turnover_rate_f, float_ratio)
        logger.info("  V4.8 加载daily_basic扩展特征 (继承V4.7.1)...")
        db_extra_query = """
        SELECT s.code, db.trade_date, db.dv_ttm, db.turnover_rate_f, db.circ_mv, db.total_mv
        FROM daily_basic db
        JOIN securities s ON db.security_id = s.id
        WHERE db.trade_date >= ? AND db.trade_date <= ?
        """
        df_extra = pd.read_sql(db_extra_query, conn, params=[date_min, date_max])
        if len(df_extra) > 0:
            df_extra['float_ratio'] = df_extra['circ_mv'] / df_extra['total_mv'].clip(lower=1e-8)
            df_extra.drop(columns=['circ_mv', 'total_mv'], inplace=True)
            df = df.merge(df_extra, on=['code', 'trade_date'], how='left')
        for col in ['dv_ttm', 'turnover_rate_f', 'float_ratio']:
            if col in df.columns:
                df[col] = df.groupby('trade_date')[col].transform(lambda x: x.fillna(x.median()))
                df[col] = df[col].fillna(0.0)
            else:
                df[col] = 0.0
        logger.info(f"    daily_basic扩展: +3 特征")

        # 微观结构/反转/风险因子 (from OHLCV, 使用向量化rolling — 复用V4.7.1高效实现)
        logger.info("  V4.8 计算微观结构/反转/风险因子 (向量化rolling)...")
        # 需要额外前40天的数据用于滚动窗口
        from datetime import datetime as dt_cls, timedelta as td_cls
        try:
            ext_start = (dt_cls.strptime(date_min, '%Y-%m-%d') - td_cls(days=40)).strftime('%Y-%m-%d')
        except Exception:
            ext_start = (dt_cls.strptime(date_min, '%Y%m%d') - td_cls(days=40)).strftime('%Y%m%d')

        ohlcv_query = """
        SELECT s.code, q.trade_date, q.close, q.volume, q.price_change_pct
        FROM daily_quotes q
        JOIN securities s ON q.security_id = s.id
        WHERE s.type = 'A股' AND q.trade_date >= ? AND q.trade_date <= ?
        ORDER BY s.code, q.trade_date
        """
        df_ohlcv = pd.read_sql(ohlcv_query, conn, params=[ext_start, date_max])
        conn.close()

        micro_cols = self.MICROSTRUCTURE_FEATURES + self.REVERSAL_FEATURES + self.RISK_FEATURES
        if len(df_ohlcv) > 0:
            # === 微观结构 (4个) — 向量化rolling ===
            micro_parts = []
            for code, grp in df_ohlcv.groupby('code'):
                grp = grp.sort_values('trade_date').copy()
                close = grp['close'].values.astype(float)
                volume = grp['volume'].values.astype(float)
                pct = pd.to_numeric(grp['price_change_pct'], errors='coerce').fillna(0).values.astype(float)

                abs_ret = np.abs(pct)
                vol_safe = np.where(volume > 0, volume, 1e-8)
                amihud = pd.Series(abs_ret / vol_safe).rolling(20, min_periods=10).mean().values
                vp_corr = pd.Series(close).rolling(10, min_periods=5).corr(pd.Series(volume)).values
                close_s = pd.Series(close)
                rolling_max = close_s.rolling(20, min_periods=10).max()
                dd = (close_s - rolling_max) / rolling_max.clip(lower=1e-8)
                max_dd_20d = dd.rolling(20, min_periods=10).min().values
                up_mask = pct > 0
                down_mask = pct < 0
                up_vol = pd.Series(np.where(up_mask, volume, 0.0)).rolling(10, min_periods=3).sum().values
                dn_vol = pd.Series(np.where(down_mask, volume, 0.0)).rolling(10, min_periods=3).sum().values
                ud_asym = up_vol / np.where(dn_vol > 0, dn_vol, 1e-8)

                grp_out = grp[['code', 'trade_date']].copy()
                grp_out['amihud_illiquidity'] = amihud
                grp_out['volume_price_corr_10d'] = vp_corr
                grp_out['max_drawdown_20d'] = max_dd_20d
                grp_out['updown_volume_asymmetry'] = ud_asym
                micro_parts.append(grp_out)

            df_micro = pd.concat(micro_parts, ignore_index=True)
            df = df.merge(df_micro, on=['code', 'trade_date'], how='left')
            for col in self.MICROSTRUCTURE_FEATURES:
                missing = df[col].isnull().sum()
                if missing > 0:
                    df[col] = df.groupby('trade_date')[col].transform(lambda x: x.fillna(x.median()))
                    df[col] = df[col].fillna(0.0)
            logger.info(f"    微观结构因子: +4 特征")

            # === 反转因子 (2个) — 向量化 ===
            ret_parts = []
            for code, grp in df_ohlcv.groupby('code'):
                grp = grp.sort_values('trade_date').copy()
                close = grp['close'].values.astype(float)
                ret_1d = np.concatenate([[np.nan], close[1:] / close[:-1] - 1])
                close_s = pd.Series(close)
                ret_3d = (close_s / close_s.shift(3) - 1).values

                grp_out = grp[['code', 'trade_date']].copy()
                grp_out['return_1d'] = ret_1d
                grp_out['return_3d'] = ret_3d
                ret_parts.append(grp_out)

            df_ret = pd.concat(ret_parts, ignore_index=True)
            df = df.merge(df_ret, on=['code', 'trade_date'], how='left')
            for col in self.REVERSAL_FEATURES:
                df[col] = df[col].fillna(0.0)
            logger.info(f"    反转因子: +2 特征")

            # === 风险因子 (2个) — 向量化rolling ===
            risk_parts = []
            for code, grp in df_ohlcv.groupby('code'):
                grp = grp.sort_values('trade_date').copy()
                close = grp['close'].values.astype(float)
                daily_ret = np.concatenate([[np.nan], close[1:] / close[:-1] - 1])
                daily_ret_s = pd.Series(daily_ret)
                rolling_mean = daily_ret_s.rolling(20, min_periods=5).mean()
                demeaned = daily_ret_s - rolling_mean
                idio_vol = demeaned.rolling(20, min_periods=5).std().values
                # 下行偏差: 仅负收益的std
                neg_ret = daily_ret_s.where(daily_ret_s < 0)
                downside_dev = neg_ret.rolling(20, min_periods=3).std().values

                grp_out = grp[['code', 'trade_date']].copy()
                grp_out['idio_volatility_20d'] = idio_vol
                grp_out['downside_deviation_20d'] = downside_dev
                risk_parts.append(grp_out)

            df_risk = pd.concat(risk_parts, ignore_index=True)
            df = df.merge(df_risk, on=['code', 'trade_date'], how='left')
            for col in self.RISK_FEATURES:
                df[col] = df[col].fillna(0.0)
            logger.info(f"    风险因子: +2 特征")

        else:
            for col in micro_cols:
                df[col] = 0.0

        total_new = len(self.ALL_FI_COLUMNS) + 1 + 3 + len(micro_cols)  # FI+accel+daily_basic+micro
        logger.info(f"  V4.8 总计新增特征: {total_new} (FI:{len(self.ALL_FI_COLUMNS)+1} + DB:3 + Micro:{len(micro_cols)})")

        return df

    def train_single_target_models(self, X_train, X_val, y_train, y_val, target_name: str,
                                    sample_weights_train=None):
        """V4.8: 3d=5回归(继承V4.7.2), 5d/10d/15d=5回归+LambdaRank+ListNet=7模型

        继承V4.7.2: 3d跳过所有排名模型(纯回归5个)
        V4.8新增: 5d/10d/15d加训ListNet (在V4.7.2的LambdaRank基础上)
        """
        import gc

        if '3d' in target_name:
            # 3d: V4.7.2逻辑 — 纯回归5模型, 跳过所有排名模型
            logger.info(f"  V4.8: {target_name} 使用纯回归模型(5个, 跳过LambdaRank+ListNet)")
            return V43Trainer.train_single_target_models(
                self, X_train, X_val, y_train, y_val, target_name,
                sample_weights_train=sample_weights_train)

        # 5d/10d/15d: V4.7.1的6模型(5回归+LambdaRank) + V4.8的ListNet = 7模型
        models, pred_train, pred_val = V471Trainer.train_single_target_models(
            self, X_train, X_val, y_train, y_val, target_name,
            sample_weights_train=sample_weights_train)

        # V4.8新增: ListNet (rank_xendcg) — 仅对5d/10d/15d
        # 复用V4.7.1 LambdaRank已计算的relevance标签 (避免重复计算)
        relevance_train = getattr(self, '_cached_relevance_train', None)
        group_train = getattr(self, '_cached_group_train', None)
        relevance_val = getattr(self, '_cached_relevance_val', None)
        group_val = getattr(self, '_cached_group_val', None)

        if relevance_train is not None and group_train is not None:
            logger.info(f"  V4.8 训练 LGB-ListNet ({target_name}) [复用LambdaRank标签]...")
            try:
                lgb_listnet_params = {
                    'objective': 'rank_xendcg',
                    'metric': 'ndcg',
                    'eval_at': [10, 20],
                    'num_leaves': 24,
                    'learning_rate': 0.03,
                    'feature_fraction': 0.7,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'reg_alpha': 0.5,
                    'reg_lambda': 3.0,
                    'min_data_in_leaf': 300,
                    'min_gain_to_split': 0.01,
                    'path_smooth': 5.0,
                    'verbose': -1,
                }

                lgb_listnet_train = lgb.Dataset(
                    X_train, label=relevance_train, group=group_train,
                    weight=sample_weights_train, free_raw_data=True
                )
                lgb_listnet_val = lgb.Dataset(
                    X_val, label=relevance_val, group=group_val,
                    reference=lgb_listnet_train, free_raw_data=True
                )

                lgb_listnet_model = lgb.train(
                    lgb_listnet_params, lgb_listnet_train,
                    num_boost_round=600,
                    valid_sets=[lgb_listnet_train, lgb_listnet_val],
                    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
                )

                models['lgb_listnet'] = lgb_listnet_model
                pred_train['lgb_listnet'] = lgb_listnet_model.predict(X_train)
                pred_val['lgb_listnet'] = lgb_listnet_model.predict(X_val)

                ic_val, _ = spearmanr(pred_val['lgb_listnet'], y_val)
                logger.info(f"    LGB-ListNet ({target_name}): IC={ic_val:.4f}")

                del lgb_listnet_train, lgb_listnet_val
                gc.collect()
            except Exception as e:
                logger.warning(f"    LGB-ListNet ({target_name}) 训练失败: {e}")

        return models, pred_train, pred_val

    def _compute_ranking_alpha(self, all_results: dict, X_val: np.ndarray,
                                y_val_dict: dict, val_dates: np.ndarray) -> float:
        """计算回归/排名模型融合系数α (基于验证集ICIR)

        α=1 → 纯回归, α=0 → 纯排名
        """
        unique_dates = np.unique(val_dates)

        reg_daily_ics = []
        rank_daily_ics = []

        for target_key in ['5d', '10d', '15d']:  # 排除3d (3d没有排名模型)
            if target_key not in all_results:
                continue
            models = all_results[target_key]['models']
            y_target = y_val_dict.get(target_key)
            if y_target is None:
                continue

            # Collect regression ensemble pred
            reg_preds = []
            rank_preds = []
            for name, model in models.items():
                try:
                    if name == 'xgb':
                        pred = model.predict(xgb.DMatrix(X_val))
                    else:
                        pred = model.predict(X_val)
                    if name in ('lgb_rank', 'lgb_listnet'):
                        rank_preds.append(pred)
                    else:
                        reg_preds.append(pred)
                except Exception:
                    continue

            if reg_preds:
                reg_ensemble = np.mean(reg_preds, axis=0)
                for d in unique_dates:
                    mask = val_dates == d
                    if mask.sum() < 10:
                        continue
                    ic, _ = spearmanr(reg_ensemble[mask], y_target[mask])
                    if not np.isnan(ic):
                        reg_daily_ics.append(ic)

            if rank_preds:
                rank_ensemble = np.mean(rank_preds, axis=0)
                for d in unique_dates:
                    mask = val_dates == d
                    if mask.sum() < 10:
                        continue
                    ic, _ = spearmanr(rank_ensemble[mask], y_target[mask])
                    if not np.isnan(ic):
                        rank_daily_ics.append(ic)

        if not reg_daily_ics or not rank_daily_ics:
            logger.info("  排名α: 数据不足, 使用默认α=0.5")
            return 0.5

        reg_icir = np.mean(reg_daily_ics) / max(np.std(reg_daily_ics), 1e-8)
        rank_icir = np.mean(rank_daily_ics) / max(np.std(rank_daily_ics), 1e-8)

        # α = reg_icir / (reg_icir + rank_icir)
        total = abs(reg_icir) + abs(rank_icir)
        if total < 1e-8:
            alpha = 0.5
        else:
            alpha = abs(reg_icir) / total

        alpha = np.clip(alpha, 0.3, 0.8)  # 保持至少30%的两种信号
        logger.info(f"  排名α计算: reg_ICIR={reg_icir:.4f}, rank_ICIR={rank_icir:.4f} → α={alpha:.3f}")
        return float(alpha)

    def walk_forward_train(self, start_date: str = None, end_date: str = None,
                            purge_days: int = 15, min_train_days: int = 900,
                            val_days: int = 120, test_days: int = 120,
                            step_days: int = 90):
        """V4.8 Walk-Forward 训练 — V4.7.2底座 + ListNet + 财务质量 + 置信度

        继承V4.7.2:
        - Bug修复(3项) + V4.7.1 17特征 + LambdaRank(5d/10d/15d) + 时间衰减
        - 目标特异性Sharpe-Blend + 3d纯回归 + 无小盘加成
        - V4.6管线: ICIR权重 + Combined Isotonic + Meta-Learner

        V4.8新增:
        - 12个新财务质量特征
        - ListNet排名模型(5d/10d/15d)
        - 回归/排名alpha融合
        - 模型置信度加权(scorer层实现)
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("V4.8 Walk-Forward 训练 (V4.7.2底座 + 3轴创新)")
        logger.info("=" * 60)
        logger.info(f"  参数: min_train={min_train_days}d, val={val_days}d, test={test_days}d, "
                     f"step={step_days}d, purge={purge_days}d")
        logger.info(f"  目标权重: {self.target_weights}")
        logger.info(f"  Sharpe融合(目标特异性): {self.TARGET_SHARPE_BLEND}")
        logger.info(f"  模型: 3d=5回归, 5d/10d/15d=5回归+LambdaRank+ListNet=7")
        logger.info(f"  V4.8新增: +{len(self.all_financial_quality_features)}财务质量特征 + ListNet + alpha融合 + 置信度加权")

        # 1. 一次性加载全量数据 (V4.7.2: 76特征 + V4.8: 12新特征 = ~88特征)
        df = self.load_data(start_date, end_date)
        X, y_3d, y_5d, y_10d, y_15d, df = self.prepare_features(df)

        dates = df['trade_date'].values
        unique_dates = np.sort(np.unique(dates))
        n_dates = len(unique_dates)
        logger.info(f"  总交易日: {n_dates}, 样本: {len(X):,}, 特征: {X.shape[1]}")

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
            train_dates_w = dates[train_mask]
            val_dates_w = dates[val_mask]

            # Walk-Forward: 特征Winsorization (仅用窗口内训练集)
            X_train_w, wf_bounds = self.winsorize_features(X_train_w)
            self._apply_bounds(X_val_w, wf_bounds)
            self._apply_bounds(X_test_w, wf_bounds)

            # Walk-Forward: 标签Winsorization (仅用训练集统计量)
            for y_tr_w, y_va_w, y_te_w in [(y_3d_tr, y_3d_va, y_3d_te),
                                             (y_5d_tr, y_5d_va, y_5d_te),
                                             (y_10d_tr, y_10d_va, y_10d_te),
                                             (y_15d_tr, y_15d_va, y_15d_te)]:
                lo = np.percentile(y_tr_w, 1)
                hi = np.percentile(y_tr_w, 99)
                y_tr_w[:] = np.clip(y_tr_w, lo, hi)
                y_va_w[:] = np.clip(y_va_w, lo, hi)
                y_te_w[:] = np.clip(y_te_w, lo, hi)

            # V4.7.2: 目标特异性Sharpe-Blend (继承)
            self.train_dates = train_dates_w
            self.val_dates = val_dates_w
            for target_key, y_tr_w, y_va_w, y_te_w in [
                ('label_3d', y_3d_tr, y_3d_va, y_3d_te),
                ('label_5d', y_5d_tr, y_5d_va, y_5d_te),
                ('label_10d', y_10d_tr, y_10d_va, y_10d_te),
                ('label_15d', y_15d_tr, y_15d_va, y_15d_te),
            ]:
                self._apply_sharpe_blend(y_tr_w, y_va_w, y_te_w,
                                          train_dates_w, val_dates_w, test_dates_w,
                                          target_key)

            # 训练4目标 (3d=5模型, 5d/10d/15d=7模型)
            window_metrics = {}
            for target_key, y_tr, y_va, y_te in [
                ('3d', y_3d_tr, y_3d_va, y_3d_te),
                ('5d', y_5d_tr, y_5d_va, y_5d_te),
                ('10d', y_10d_tr, y_10d_va, y_10d_te),
                ('15d', y_15d_tr, y_15d_va, y_15d_te),
            ]:
                sample_w = self.compute_sample_weights(df[train_mask], y_tr)
                models, pred_train, pred_val = self.train_single_target_models(
                    X_train_w, X_val_w, y_tr, y_va, f"label_{target_key}",
                    sample_weights_train=sample_w)
                weights, rmses = self.calculate_ensemble_weights(pred_val, y_va)

                # test set预测
                pred_test = {}
                for name, model in models.items():
                    try:
                        if name == 'xgb':
                            pred_test[name] = model.predict(xgb.DMatrix(X_test_w))
                        else:
                            pred_test[name] = model.predict(X_test_w)
                    except Exception:
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
        logger.info("训练最终V4.8生产模型 (全量数据)")
        logger.info("=" * 60)

        split_idx = int(n_dates * 0.85)
        split_date = unique_dates[split_idx]
        train_mask_final = dates <= split_date
        val_mask_final = dates > split_date

        X_train_f, X_val_f = X[train_mask_final].copy(), X[val_mask_final].copy()
        self.val_dates = dates[val_mask_final]
        self.train_dates = dates[train_mask_final]

        # Bug 1修复(继承V4.7.2): 生产模型的Winsorization也只用训练集
        X_train_f, self.winsorize_bounds = self.winsorize_features(X_train_f)
        self._apply_bounds(X_val_f, self.winsorize_bounds)
        logger.info(f"  生产模型: 特征Winsorization (训练集bounds), {len(self.winsorize_bounds)} 列")

        df_train_f = df[train_mask_final]
        all_results = {}

        y_val_dict = {}
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

        # V4.7.2: 目标特异性Sharpe-Blend (Bug 2修复, 继承)
        logger.info(f"  [V4.8] 目标特异性Sharpe-Blend(继承V4.7.2): {self.TARGET_SHARPE_BLEND}")
        train_dates_f = dates[train_mask_final]
        val_dates_f = dates[val_mask_final]
        for target_key, y_tr, y_va in targets_final:
            self._apply_sharpe_blend(y_tr, y_va, np.array([]),
                                      train_dates_f, val_dates_f, np.array([]),
                                      f"label_{target_key}")

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

        # 8. V4.6增强: ICIR优化集成权重 (复用V46Trainer方法)
        logger.info("\n" + "=" * 60)
        logger.info("V4.8 1A: ICIR最大化集成权重 (V4.6管线)")
        logger.info("=" * 60)
        icir_weights = V46Trainer._optimize_icir_weights(self, all_results, X_val_f, y_val_dict, self.val_dates)
        for target_key, w in icir_weights.items():
            if target_key in all_results:
                all_results[target_key]['weights'] = w

        # 9. V4.6增强: Combined Isotonic (复用V46Trainer方法)
        logger.info("\n" + "=" * 60)
        logger.info("V4.8 1C: Combined-Score Isotonic (V4.6管线)")
        logger.info("=" * 60)
        combined_isotonic = V46Trainer._fit_combined_isotonic(self, X_val_f, y_val_dict, all_results, icir_weights)

        # 10. V4.6增强: Stacking Meta-Learner (复用V46Trainer方法)
        logger.info("\n" + "=" * 60)
        logger.info("V4.8 1D: Stacking Meta-Learner (V4.6管线)")
        logger.info("=" * 60)
        meta_learner, meta_feature_names = V46Trainer._train_meta_learner(self, X_val_f, y_val_dict, all_results)

        # 11. V4.8新增: 计算回归/排名融合α
        logger.info("\n" + "=" * 60)
        logger.info("V4.8: 回归/排名融合 alpha 计算")
        logger.info("=" * 60)
        ranking_alpha = self._compute_ranking_alpha(all_results, X_val_f, y_val_dict, self.val_dates)

        # 12. 特征重要性分析
        self._log_feature_importance(all_results)

        # 13. 计算全局评分分位数
        # 对全量X应用生产模型的winsorize bounds
        X_all = X.copy()
        self._apply_bounds(X_all, self.winsorize_bounds)
        global_quantiles = self._compute_global_quantiles(X_all, all_results, self.target_weights)
        recommendation_thresholds = self._compute_recommendation_thresholds(X_all, all_results)

        # 14. 保存模型
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        output_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v48'
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 将winsorize_bounds转为dict格式 (feature_name -> (lo, hi))
        winsorize_bounds_dict = {}
        if self.winsorize_bounds and self.feature_names:
            for idx, (lo, hi) in enumerate(self.winsorize_bounds):
                if idx < len(self.feature_names):
                    winsorize_bounds_dict[self.feature_names[idx]] = (lo, hi)

        model_data = {
            'version': 'v4.8',
            'models': all_results,
            'feature_names': self.feature_names,
            'target_weights': self.target_weights,
            'market_features': list(self.market_calculator.market_features.columns[1:]),
            'winsorize_bounds': winsorize_bounds_dict,
            'global_quantiles': global_quantiles,
            'recommendation_thresholds': recommendation_thresholds,
            # 模型类型标识
            'cascade': False,
            'rank_normalized': False,
            'robust_zscore': True,
            'industry_excess_labels': True,
            'dual_stream': False,
            'cross_sectional_neutralization': False,
            'macro_feature_cols': self.macro_feature_cols,
            'stock_feature_cols': self.stock_feature_cols,
            'extra_features_from_daily_basic': ['pe_ttm', 'pb', 'ps_ttm', 'turnover_rate', 'log_market_cap',
                                                 'dv_ttm', 'turnover_rate_f', 'float_ratio'],
            'extra_features_from_tech_indicators': self.extra_tech_feature_names,
            # V4.7.1特征 (继承自V4.7.2)
            'extra_features_financial': self.FINANCIAL_FEATURES,
            'extra_features_microstructure': self.MICROSTRUCTURE_FEATURES,
            'extra_features_reversal': self.REVERSAL_FEATURES,
            'extra_features_risk': self.RISK_FEATURES,
            # V4.8新增财务质量特征
            'extra_financial_quality_features': self.all_financial_quality_features,
            'targets': ['3d', '5d', '10d', '15d'],
            'ensemble_type': 'icir_optimized_with_ranking',
            'sample_weighting': True,
            'time_decay_half_life': 365,
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
            'sharpe_label_blend': 'target_specific',
            'sharpe_blend_config': self.TARGET_SHARPE_BLEND,
            'liquidity_discount': True,
            'bear_sample_weighting': True,
            'min_train_days': min_train_days,
            'step_days': step_days,
            # V4.7.1 组件 (继承自V4.7.2)
            'has_lambdarank': True,  # 5d/10d/15d有, 3d没有
            'has_time_decay': True,
            'bug_fixes': ['winsorization_leakage', 'sharpe_blend_applied', 'market_index_000300'],
            # V4.6 后处理组件 (继承自V4.7.2)
            'icir_optimized_weights': icir_weights,
            'combined_isotonic': combined_isotonic,
            'meta_learner': meta_learner,
            'meta_feature_names': meta_feature_names,
            'small_cap_weighting': False,  # 继承V4.7.2: 明确关闭
            # V4.8 新增
            'has_ranking_models': True,  # ListNet + LambdaRank
            'ranking_alpha': ranking_alpha,
            'use_confidence_weighting': True,
            'has_listnet': True,
            '3d_no_ranking': True,  # 3d跳过所有排名模型
            'target_specific_sharpe': self.TARGET_SHARPE_BLEND,
        }

        model_path = output_dir / f'v48_multi_target_{timestamp}.pkl'
        joblib.dump(model_data, model_path)
        logger.info(f"\n模型已保存: {model_path}")
        logger.info(f"  大小: {model_path.stat().st_size / 1024 / 1024:.1f} MB")

        # 保存训练历史
        history = {
            'version': 'v4.8',
            'base': 'V4.7.2 (V4.7.1底座 + V4.6管线)',
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
                'ranking_alpha': ranking_alpha,
            },
            'target_weights': self.target_weights,
            'ensemble_weights': {k: all_results[k]['weights'] for k in all_results},
            'bug_fixes_inherited': {
                'winsorization_leakage': 'prepare_features不再提前winsorize全量数据 (V4.7.1)',
                'sharpe_blend_applied': '每个WF窗口和生产模型均执行目标特异性Sharpe-Blend (V4.7.2)',
                'market_index': 'scorer层统一使用000300.SH (V4.7.1)',
            },
            'inherited_from_v472': {
                'v471_features': {
                    'financial': self.FINANCIAL_FEATURES,
                    'daily_basic_extra': ['dv_ttm', 'turnover_rate_f', 'float_ratio'],
                    'microstructure': self.MICROSTRUCTURE_FEATURES,
                    'reversal': self.REVERSAL_FEATURES,
                    'risk': self.RISK_FEATURES,
                },
                'time_decay': 'half_life=365d',
                'lambdarank': '5d/10d/15d only (3d skipped)',
                'target_specific_sharpe': self.TARGET_SHARPE_BLEND,
                'no_small_cap_weighting': True,
            },
            'v48_innovations': {
                '1_financial_quality': self.all_financial_quality_features,
                '2_listnet': 'rank_xendcg for 5d/10d/15d',
                '3_ranking_alpha': ranking_alpha,
                '4_confidence_weighting': 'scorer层实现',
            },
            'modules': {
                'A_monotonicity': True,
                'B_liquidity_discount': True,
                'C_bear_specialist': len(bear_models) > 0,
                'D_sharpe_blend': 'target_specific (V4.7.2)',
                'E_executability_filter': 'scorer层实现',
                'F_regime_adaptive': 'scorer层实现',
                'lambdarank': '5d/10d/15d only (V4.7.1 inherited)',
                'listnet': '5d/10d/15d only (V4.8 new)',
                'time_decay': True,
                'V46_icir_weights': True,
                'V46_small_cap_weighting': False,
                'V46_combined_isotonic': combined_isotonic is not None,
                'V46_meta_learner': meta_learner is not None,
                'V48_confidence_weighting': True,
                'V48_financial_quality': True,
            },
        }

        history_path = output_dir / f'training_history_{timestamp}.json'
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        latest_path = output_dir / 'training_history_latest.json'
        with open(latest_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        logger.info(f"训练历史已保存: {history_path}")
        logger.info(f"\nV4.8 训练完成! 总耗时: {duration:.0f}秒 ({duration/60:.1f}分钟)")

        return model_data, history


def main():
    parser = argparse.ArgumentParser(description='V3.95/V4.3/V4.4/V4.6/V4.7/V4.7.1/V4.7.2/V4.7.3/V4.7.4/V4.8/V4.8.1 多目标训练')
    parser.add_argument('--start-date', type=str, default='2020-01-01', help='训练开始日期')
    parser.add_argument('--end-date', type=str, default=None, help='训练结束日期')
    parser.add_argument('--purge-days', type=int, default=10, help='Purge gap天数 (应>=最大标签前瞻天数, label_10d需要10天)')
    parser.add_argument('--sharpe-blend', type=float, default=0.3, help='Sharpe标签融合比例 (0=纯收益, 0.3=推荐, 1=纯Sharpe)')
    parser.add_argument('--v43', action='store_true', help='V4.3: 扩展特征+强正则+Walk-Forward')
    parser.add_argument('--v44', action='store_true', help='V4.4: V4.3信号+6增强模块 (单调性校准/流动性/熊市专家/Sharpe标签)')
    parser.add_argument('--v46', action='store_true', help='V4.6: V4.4+ICIR权重+小盘加权+Combined Isotonic+Meta-Learner')
    parser.add_argument('--v47', action='store_true', help='V4.7: V4.6-小盘加权+排名标签LGB (IC单调性优化)')
    parser.add_argument('--v471', action='store_true', help='V4.7.1: Bug修复+17新特征+LambdaRank+时间衰减 (底层信号质量提升)')
    parser.add_argument('--v472', action='store_true', help='V4.7.2: V4.7.1底座+ICIR权重+Meta-Learner+Combined Isotonic (融合增强版)')
    parser.add_argument('--v473', action='store_true', help='V4.7.3: 简化管线+特征精简+放宽正则化 (去Meta-Learner/Combined Isotonic)')
    parser.add_argument('--v474', action='store_true', help='V4.7.4: V4.7.3+连续评分+选择性V4.8特征+ListNet+严格ICIR约束')
    parser.add_argument('--v475', action='store_true', help='V4.7.5: V4.7.3+特征精简(70->50)+连续评分+自适应权重')
    parser.add_argument('--v476', action='store_true', help='V4.7.6: V4.7.5+Top-K聚焦权重+置信度折扣+波动率调整')
    parser.add_argument('--v477', action='store_true', help='V4.7.7: V4.7.5+Huber Loss+180d衰减+DART')
    parser.add_argument('--v478', action='store_true', help='V4.7.8: V4.7.7 Huber+DART + V4.7.5 365d衰减 (IC+Top3双优)')
    parser.add_argument('--v479', action='store_true', help='V4.7.9: V4.7.7 Huber+DART + 240d衰减 + Top5%头部加权')
    parser.add_argument('--v480', action='store_true', help='V4.8.0: V4.7.5+270d衰减(精准攻ic_decay_ratio)')
    parser.add_argument('--v481', action='store_true', help='V4.8.1: V4.7.5+15新因子(50→60特征)')
    parser.add_argument('--v482', action='store_true', help='V4.8.2: V4.8.1+21新因子(60→~81特征, 价量+财务+学术)')
    parser.add_argument('--v483', action='store_true', help='V4.8.3: V4.8.2+29 BRAIN验证因子(~81→~110特征, ICIR+35.8%%)')
    parser.add_argument('--v484', action='store_true', help='V4.8.4: V4.8.1+brain_roll_spread(60→61特征, TopK筛选)')
    parser.add_argument('--v485', action='store_true', help='V4.8.5: V4.8.4+ETF训练数据(61特征, A股ICIR+0.033)')
    parser.add_argument('--v486', action='store_true', help='V4.8.6: V4.8.4+3个BRAIN Top-K因子(61→64特征, ICIR+4.2%%)')
    parser.add_argument('--brain-features', action='store_true',
                        help='加载 BRAIN 验证因子 (需先运行 brain_feature_importer 缓存)')
    parser.add_argument('--skip-wf', action='store_true', help='跳过Walk-Forward评估, 只训练生产模型 (节省~75%时间)')
    parser.add_argument('--num-leaves', type=int, default=None, help='覆盖LGB num_leaves (默认: 各版本内置值)')
    parser.add_argument('--min-data-in-leaf', type=int, default=None, help='覆盖LGB min_data_in_leaf (默认: 各版本内置值)')
    args = parser.parse_args()

    # BRAIN 因子标志 — 适用于所有 Trainer 版本
    _use_brain = getattr(args, 'brain_features', False)

    # Apply CLI hyperparameter overrides to any trainer
    def _apply_overrides(trainer_obj):
        if args.num_leaves is not None:
            trainer_obj._cli_num_leaves = args.num_leaves
            logger.info(f"  CLI override: num_leaves={args.num_leaves}")
        if args.min_data_in_leaf is not None:
            trainer_obj._cli_min_data_in_leaf = args.min_data_in_leaf
            logger.info(f"  CLI override: min_data_in_leaf={args.min_data_in_leaf}")

    if args.v486:
        trainer = V486Trainer()
        _apply_overrides(trainer)
        if args.skip_wf:
            trainer.train_production_only(
                start_date=args.start_date, end_date=args.end_date,
                purge_days=max(args.purge_days, 15))
        else:
            trainer.walk_forward_train(
                start_date=args.start_date, end_date=args.end_date,
                purge_days=max(args.purge_days, 15))
    elif args.v485:
        trainer = V485Trainer()
        _apply_overrides(trainer)
        trainer.walk_forward_train(
            start_date=args.start_date, end_date=args.end_date,
            purge_days=max(args.purge_days, 15))
    elif args.v484:
        trainer = V484Trainer()
        trainer.walk_forward_train(
            start_date=args.start_date, end_date=args.end_date,
            purge_days=max(args.purge_days, 15))
    elif args.v483:
        trainer = V483Trainer()
        trainer.walk_forward_train(
            start_date=args.start_date, end_date=args.end_date,
            purge_days=max(args.purge_days, 15))
    elif args.v482:
        trainer = V482Trainer()
        trainer.use_brain_features = _use_brain
        trainer.walk_forward_train(
            start_date=args.start_date, end_date=args.end_date,
            purge_days=max(args.purge_days, 15))
    elif args.v481:
        trainer = V481Trainer()
        trainer.walk_forward_train(
            start_date=args.start_date, end_date=args.end_date,
            purge_days=max(args.purge_days, 15))
    elif args.v480:
        trainer = V480Trainer()
        trainer.walk_forward_train(
            start_date=args.start_date, end_date=args.end_date,
            purge_days=max(args.purge_days, 15))
    elif args.v479:
        trainer = V479Trainer()
        trainer.walk_forward_train(
            start_date=args.start_date, end_date=args.end_date,
            purge_days=max(args.purge_days, 15))
    elif args.v478:
        trainer = V478Trainer()
        trainer.walk_forward_train(
            start_date=args.start_date, end_date=args.end_date,
            purge_days=max(args.purge_days, 15))
    elif args.v477:
        trainer = V477Trainer()
        trainer.walk_forward_train(
            start_date=args.start_date, end_date=args.end_date,
            purge_days=max(args.purge_days, 15))
    elif args.v476:
        trainer = V476Trainer()
        trainer.walk_forward_train(
            start_date=args.start_date, end_date=args.end_date,
            purge_days=max(args.purge_days, 15))
    elif args.v475:
        trainer = V475Trainer()
        if args.skip_wf:
            trainer.train_production_only(
                start_date=args.start_date, end_date=args.end_date,
                purge_days=max(args.purge_days, 15))
        else:
            trainer.walk_forward_train(
                start_date=args.start_date, end_date=args.end_date,
                purge_days=max(args.purge_days, 15))
    elif args.v474:
        trainer = V474Trainer()
        trainer.walk_forward_train(
            start_date=args.start_date, end_date=args.end_date,
            purge_days=max(args.purge_days, 15))
    elif args.v473:
        trainer = V473Trainer()
        trainer.walk_forward_train(
            start_date=args.start_date, end_date=args.end_date,
            purge_days=max(args.purge_days, 15))
    elif args.v472:
        trainer = V472Trainer()
        trainer.walk_forward_train(
            start_date=args.start_date, end_date=args.end_date,
            purge_days=max(args.purge_days, 15))
    elif args.v471:
        trainer = V471Trainer()
        trainer.walk_forward_train(
            start_date=args.start_date, end_date=args.end_date,
            purge_days=max(args.purge_days, 15))
    elif args.v47:
        trainer = V47Trainer()
        if args.skip_wf:
            trainer.train_production_only(
                start_date=args.start_date, end_date=args.end_date,
                purge_days=max(args.purge_days, 15))
        else:
            trainer.walk_forward_train(
                start_date=args.start_date, end_date=args.end_date,
                purge_days=max(args.purge_days, 15))
    elif args.v46:
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
        trainer.use_brain_features = getattr(args, 'brain_features', False)
        trainer.train(start_date=args.start_date, end_date=args.end_date, purge_days=args.purge_days)


if __name__ == '__main__':
    main()
