#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V4.0 Cross-Sectional Alpha Model 训练脚本

核心改进:
1. 训练标签: 超额收益 (个股收益 - 沪深300收益)
2. 特征: ~55个 cross-sectional 排名特征 + 技术指标
3. 多目标训练: 3d/5d/10d 超额收益，加权融合
4. Cross-Sectional 评估指标: Daily IC, IC_IR, Top-10% Precision/Excess Return
5. 更强正则化: max_depth=5, min_child_samples=50, reg_alpha=0.3

沿用 V3.90 的:
- Temporal split + 10天 purge gap
- 4模型 Ensemble: LightGBM + XGBoost + CatBoost + RandomForest
- GradientBoosting Meta-Model (Stacking)

沿用 V3.95 的:
- 多目标训练
- Per-feature winsorization (1st-99th percentile)
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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

    def __init__(self, db_path=None, meta_model_type='ridge',
                 market_scale=0.3, industry_scale=0.5, skip_rf=False):
        self.db_path = db_path or str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')
        self.models = {}
        self.meta_model = None
        self.meta_model_type = meta_model_type  # 'ridge', 'gbm', 'avg'
        self.feature_names = None
        self.winsorize_bounds = None
        self.market_scale = market_scale
        self.industry_scale = industry_scale
        self.skip_rf = skip_rf

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

        # 计算加权融合标签
        y = (self.target_weights['label_3d_excess'] * y_3d +
             self.target_weights['label_5d_excess'] * y_5d +
             self.target_weights['label_10d_excess'] * y_10d)

        # 处理NaN
        valid_mask = ~(np.isnan(y) | np.isnan(y_3d) | np.isnan(y_5d))
        X = X.loc[valid_mask].reset_index(drop=True)
        y = y[valid_mask]
        y_5d = y_5d[valid_mask]
        dates = dates[valid_mask]
        codes = codes[valid_mask]

        logger.info(f"✅ 特征矩阵: {X.shape}")
        logger.info(f"✅ 有效样本: {len(y):,}")
        logger.info(f"✅ 日期范围: {dates[0]} ~ {dates[-1]}")

        # 缺失值填0
        if X.isnull().any().any():
            logger.warning("⚠️ 检测到缺失值，使用0填充")
            X = X.fillna(0)

        self.feature_names = X.columns.tolist()

        return X, y, y_5d, dates, codes

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

        scaled_market = 0
        scaled_industry = 0
        for col in X_train.columns:
            if col in self.MARKET_FEATURES:
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

    def temporal_split(self, X, y, y_5d, dates, codes,
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
            result[f'dates_{name}'] = dates[mask]
            result[f'codes_{name}'] = codes[mask]

        logger.info(f"  时序划分 (purge_gap={purge_days}天):")
        logger.info(f"  训练集: {len(result['X_train']):,} 样本, 截至 {train_date_end}")
        logger.info(f"  验证集: {len(result['X_val']):,} 样本, {val_date_start} ~ {val_date_end}")
        logger.info(f"  测试集: {len(result['X_test']):,} 样本, {test_date_start} 起")

        return result

    def train_base_models(self, X_train, y_train, X_val, y_val):
        """训练4个基础模型 (更强正则化)"""
        logger.info("\n" + "=" * 80)
        logger.info("🔧 训练基础模型 (更强正则化)...")
        logger.info("=" * 80)

        models_config = {
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=5,
                num_leaves=25,
                min_child_samples=50,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=0.3,
                reg_lambda=0.3,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=5,
                min_child_weight=50,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=0.3,
                reg_lambda=0.3,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            ),
        }

        if not self.skip_rf:
            models_config['random_forest'] = RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features=0.6,
                random_state=42,
                n_jobs=-1
            )
        else:
            logger.info("  ⏭️ 跳过 Random Forest (--skip-rf)")

        if HAS_CATBOOST:
            models_config['catboost'] = CatBoostRegressor(
                iterations=300,
                learning_rate=0.03,
                depth=5,
                l2_leaf_reg=5,
                min_data_in_leaf=50,
                random_state=42,
                verbose=False
            )

        for name, model in models_config.items():
            logger.info(f"\n🔹 训练 {name}...")
            start_time = datetime.now()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            ic, _ = spearmanr(y_pred, y_val)

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"  ✅ {name}: MSE={mse:.6f}, MAE={mae:.6f}, IC={ic:.4f}, 耗时={elapsed:.1f}秒")

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

        market_features = {'market_regime', 'market_vol_regime', 'market_breadth_5d',
                           'northbound_flow_zscore', 'market_volume_regime', 'market_trend_strength'}
        industry_features = {'sw_l1_code', 'industry_breadth', 'industry_volume_change',
                             'industry_kdj_avg', 'industry_macd_bullish_pct',
                             'industry_concentration', 'industry_momentum_rank',
                             'industry_rotation_signal'}

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
            'version': 'v4.0.1',
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
            'version': 'v4.0.1',
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
        # 1. 加载数据
        X, y, y_5d, dates, codes = self.load_cached_features()

        # 2. 时序划分
        logger.info("\n" + "=" * 80)
        logger.info("📊 时序划分 (purge_gap=10天)...")
        logger.info("=" * 80)
        split = self.temporal_split(X, y, y_5d, dates, codes,
                                     val_ratio=val_ratio, test_ratio=test_ratio,
                                     purge_days=purge_days)

        # 3. Winsorization
        split['X_train'], split['X_val'], split['X_test'] = self.winsorize_features(
            split['X_train'], split['X_val'], split['X_test'])

        # 3.5 市场/行业特征缩放
        split['X_train'], split['X_val'], split['X_test'] = self.scale_market_industry_features(
            split['X_train'], split['X_val'], split['X_test'])

        # 4. 训练基础模型
        self.train_base_models(split['X_train'], split['y_train'],
                               split['X_val'], split['y_val'])

        # 5. 训练元模型
        self.train_meta_model(split['X_val'], split['y_val'],
                              split['X_test'], split['y_test'])

        # 6. Cross-Sectional评估
        metrics = self.evaluate_cross_sectional(
            split['X_test'], split['y_test'], split['y5d_test'],
            split['dates_test'], split['codes_test'])

        # 7. 特征重要性
        self._log_feature_importance()

        # 8. 保存模型
        model_path = self.save_model()

        # 9. 保存评估报告
        self._save_evaluation_report(metrics)

        logger.info("\n" + "=" * 80)
        logger.info("🎉 V4.0 Cross-Sectional Alpha Model 训练完成!")
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
    parser = argparse.ArgumentParser(description='V4.0 Cross-Sectional Alpha Model 训练')
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

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("🚀 V4.0.1 Cross-Sectional Alpha Model 训练")
    logger.info("   目标: 学习个股相对强势信号，而非大盘方向")
    logger.info("   标签: 超额收益 (个股 - 沪深300)")
    logger.info("   特征: ~55个 cross-sectional 排名特征")
    logger.info("=" * 80)
    logger.info(f"数据库: {args.db_path}")
    logger.info(f"元模型: {args.meta_model}")
    logger.info(f"市场特征缩放: {args.market_scale}")
    logger.info(f"行业特征缩放: {args.industry_scale}")
    logger.info(f"验证集比例: {args.val_ratio}")
    logger.info(f"测试集比例: {args.test_ratio}")
    logger.info(f"Purge gap: {args.purge_days} 天")

    trainer = V400CrossSectionalTrainer(
        db_path=args.db_path,
        meta_model_type=args.meta_model,
        market_scale=args.market_scale,
        industry_scale=args.industry_scale,
        skip_rf=args.skip_rf,
    )
    model_path = trainer.train(
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        purge_days=args.purge_days
    )

    logger.info(f"\n✅ 模型已保存至: {model_path}")


if __name__ == "__main__":
    main()
