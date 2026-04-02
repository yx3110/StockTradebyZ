#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.9模型训练脚本 - 基于预计算特征缓存
优势：直接从数据库读取预计算特征，无需重复计算，训练速度快10-100倍
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class V390CachedTrainer:
    """基于预计算特征的V3.9训练器"""

    def __init__(self, db_path=None):
        self.db_path = db_path or str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')
        self.models = {}
        self.meta_model = None

    def load_cached_features(self, min_samples=1000):
        """
        从数据库加载预计算的特征

        Args:
            min_samples: 最小样本数阈值

        Returns:
            (X, y, dates): 特征矩阵、标签向量、日期数组
        """
        logger.info("="*80)
        logger.info("📥 从数据库加载预计算特征...")
        logger.info("="*80)

        conn = sqlite3.connect(self.db_path)

        # 查询有效样本：
        # 1. label_5d 非空
        # 2. 排除停牌日 (base_date volume=0)
        # 3. 排除交易日 <30 天的低历史股票
        query = """
            SELECT v.code, v.trade_date, v.features_json, v.label_5d
            FROM v39_feature_cache v
            JOIN securities s ON v.code = s.code
            JOIN daily_quotes q ON q.security_id = s.id AND q.trade_date = v.trade_date
            WHERE v.label_5d IS NOT NULL
              AND q.volume > 0
              AND v.code IN (
                  SELECT s2.code FROM daily_quotes q2
                  JOIN securities s2 ON q2.security_id = s2.id
                  WHERE s2.type = 'A股'
                  GROUP BY s2.code
                  HAVING COUNT(*) >= 30
              )
            ORDER BY v.trade_date, v.code
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        logger.info(f"✅ 加载 {len(df):,} 个样本 (已过滤停牌日+低历史股票)")

        if len(df) < min_samples:
            raise ValueError(f"样本数不足！需要至少{min_samples}个，实际{len(df)}个")

        # 解析JSON特征
        logger.info("📊 解析JSON特征...")
        features_list = []
        labels = []
        dates = []

        for idx, row in df.iterrows():
            try:
                # 解析JSON
                features_dict = json.loads(row['features_json'])
                features_list.append(features_dict)
                labels.append(row['label_5d'])
                dates.append(row['trade_date'])

            except Exception as e:
                logger.warning(f"跳过无效样本 {row['code']} {row['trade_date']}: {e}")
                continue

            # 进度报告
            if (idx + 1) % 10000 == 0:
                logger.info(f"  已处理: {idx+1:,}/{len(df):,} ({(idx+1)/len(df)*100:.1f}%)")

        # 转换为DataFrame
        X = pd.DataFrame(features_list)
        y = np.array(labels)
        dates = np.array(dates)

        logger.info(f"✅ 特征矩阵: {X.shape}")
        logger.info(f"✅ 标签向量: {y.shape}")
        logger.info(f"✅ 特征列数: {X.shape[1]}")
        logger.info(f"✅ 日期范围: {dates[0]} ~ {dates[-1]}")

        # 检测并处理不一致的feature keys (不同时期cache可能有不同字段)
        # 只保留所有样本都有的特征,避免部分时期用0代替缺失特征
        if X.isnull().any().any():
            # 计算每列的缺失率
            missing_pct = X.isnull().mean()
            # 丢弃缺失率>5%的列 (这些列在某些时期不可用)
            bad_cols = missing_pct[missing_pct > 0.05].index.tolist()
            if bad_cols:
                logger.warning(f"⚠️  丢弃{len(bad_cols)}个不一致特征 (部分时期缺失>5%): {bad_cols}")
                X = X.drop(columns=bad_cols)

        # 处理残留缺失值 (丢弃不一致列后应极少)
        if X.isnull().any().any():
            residual_null = X.isnull().sum().sum()
            logger.warning(f"⚠️  残留{residual_null}个缺失值，使用0填充")
            X = X.fillna(0)

        return X, y, dates

    def train_base_models(self, X_train, y_train, X_val, y_val):
        """
        训练基础模型

        Returns:
            dict: 基础模型字典
        """
        logger.info("\n" + "="*80)
        logger.info("🔧 训练基础模型...")
        logger.info("="*80)

        # 模型配置
        models_config = {
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            ),
            'catboost': CatBoostRegressor(
                iterations=200,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3,
                random_state=42,
                verbose=False
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        }

        # 训练每个模型
        for name, model in models_config.items():
            logger.info(f"\n🔹 训练 {name}...")
            start_time = datetime.now()

            model.fit(X_train, y_train)

            # 验证集预测
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)

            elapsed = (datetime.now() - start_time).total_seconds()

            logger.info(f"  ✅ {name}: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.4f}, 耗时={elapsed:.1f}秒")

            self.models[name] = model

        return self.models

    def train_meta_model(self, X_val, y_val, X_test=None, y_test=None):
        """
        训练元模型（Stacking）

        修复: 使用验证集的基础模型预测作为元特征（天然OOF，因为基础模型未见过验证集）
        旧版错误: 使用训练集预测→基础模型过拟合→元特征虚高→元模型学到虚假信号
        """
        logger.info("\n" + "="*80)
        logger.info("🔧 训练元模型 (Stacking - OOF on validation set)...")
        logger.info("="*80)

        # 用基础模型对验证集预测（天然out-of-fold，无泄漏）
        meta_features_val = np.column_stack([
            model.predict(X_val) for model in self.models.values()
        ])

        # 元模型配置
        self.meta_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )

        # 在验证集元特征上训练元模型
        logger.info("🔹 训练Gradient Boosting元模型 (基于验证集OOF预测)...")
        self.meta_model.fit(meta_features_val, y_val)

        # 在测试集上评估（如果提供）
        if X_test is not None and y_test is not None:
            meta_features_test = np.column_stack([
                model.predict(X_test) for model in self.models.values()
            ])
            y_pred = self.meta_model.predict(meta_features_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            ic_global, _ = spearmanr(y_pred, y_test)
            direction_acc = np.mean((y_pred > 0) == (y_test > 0))

            logger.info(f"✅ 元模型 (测试集): MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.4f}")
            logger.info(f"  全局IC={ic_global:.4f}  ⚠️ single-shot, 仅供参考")
            logger.info(f"  方向准确率={direction_acc:.4f}")

            # 北极星: Daily IC / ICIR / IC>0%
            test_dates = getattr(self, '_test_dates', None)
            if test_dates is not None:
                daily_ics = []
                for date in np.unique(test_dates):
                    mask = test_dates == date
                    n = mask.sum()
                    if n < 20:
                        continue
                    day_ic, _ = spearmanr(y_pred[mask], y_test[mask])
                    if not np.isnan(day_ic):
                        daily_ics.append(day_ic)

                if daily_ics:
                    daily_ic_mean = np.mean(daily_ics)
                    daily_ic_std = np.std(daily_ics)
                    daily_icir = daily_ic_mean / daily_ic_std if daily_ic_std > 1e-8 else 0
                    daily_ic_pos = np.mean(np.array(daily_ics) > 0) * 100

                    logger.info(f"  ── 北极星 Daily IC ──")
                    logger.info(f"  Daily IC均值: {daily_ic_mean:.4f} ± {daily_ic_std:.4f}  ({len(daily_ics)}天)")
                    logger.info(f"  ICIR:         {daily_icir:.4f}")
                    logger.info(f"  IC>0占比:     {daily_ic_pos:.1f}%")
                    logger.info(f"  ────────────────────")

                    # 北极星达标评估
                    status_ic = "✅" if daily_ic_mean >= 0.03 else "❌"
                    status_icir = "✅" if daily_icir >= 0.30 else "❌"
                    logger.info(f"  达标: DailyIC={daily_ic_mean:.4f} {status_ic}(≥0.03) | ICIR={daily_icir:.4f} {status_icir}(≥0.30)")

                    self._north_star_metrics = {
                        'daily_ic_mean': daily_ic_mean,
                        'daily_ic_std': daily_ic_std,
                        'daily_icir': daily_icir,
                        'daily_ic_positive_pct': daily_ic_pos,
                        'ic_global': ic_global,
                    }
            else:
                logger.info(f"  ⚠️ 无test_dates, 无法计算Daily IC (需temporal_split)")

        return self.meta_model

    def save_model(self, output_path='ml_models/trained_models/v39'):
        """保存模型"""
        logger.info("\n" + "="*80)
        logger.info("💾 保存模型...")
        logger.info("="*80)

        Path(output_path).mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存基础模型
        for name, model in self.models.items():
            model_path = f"{output_path}/v390_{name}_{timestamp}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"  ✅ {name}: {model_path}")

        # 保存元模型
        meta_path = f"{output_path}/v390_meta_{timestamp}.pkl"
        with open(meta_path, 'wb') as f:
            pickle.dump(self.meta_model, f)
        logger.info(f"  ✅ meta_model: {meta_path}")

        # 保存完整系统
        full_model = {
            'base_models': self.models,
            'meta_model': self.meta_model,
            'timestamp': timestamp,
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names),
            'version': 'v3.9.0'
        }
        full_path = f"{output_path}/v390_full_system_{timestamp}.pkl"
        with open(full_path, 'wb') as f:
            pickle.dump(full_model, f)
        logger.info(f"  ✅ full_system: {full_path}")

        return full_path

    def temporal_split(self, X, y, dates, val_ratio=0.15, test_ratio=0.15, purge_days=5):
        """
        按时间顺序划分数据集，并在边界处添加 purge gap 避免标签窗口重叠

        Args:
            X: 特征矩阵
            y: 标签
            dates: 日期数组（与X行对齐）
            val_ratio: 验证集占比
            test_ratio: 测试集占比
            purge_days: 边界 purge 天数（应 >= label 前瞻天数，label_5d 需要至少5天）

        Returns:
            X_train, y_train, X_val, y_val, X_test, y_test
        """
        unique_dates = np.sort(np.unique(dates))
        n_dates = len(unique_dates)

        # 计算各段的日期边界
        train_date_end_idx = int(n_dates * (1 - val_ratio - test_ratio)) - 1
        val_date_end_idx = int(n_dates * (1 - test_ratio)) - 1

        train_date_end = unique_dates[train_date_end_idx]
        val_date_start = unique_dates[min(train_date_end_idx + 1 + purge_days, n_dates - 1)]
        val_date_end = unique_dates[val_date_end_idx]
        test_date_start = unique_dates[min(val_date_end_idx + 1 + purge_days, n_dates - 1)]

        # 按日期筛选样本
        train_mask = dates <= train_date_end
        val_mask = (dates >= val_date_start) & (dates <= val_date_end)
        test_mask = dates >= test_date_start

        X_train = X.loc[train_mask].reset_index(drop=True)
        y_train = y[train_mask]
        X_val = X.loc[val_mask].reset_index(drop=True)
        y_val = y[val_mask]
        X_test = X.loc[test_mask].reset_index(drop=True)
        y_test = y[test_mask]

        # 保存test_dates用于Daily IC计算
        self._test_dates = dates[test_mask]

        logger.info(f"  时序划分 (purge_gap={purge_days}天):")
        logger.info(f"  训练集: {len(X_train):,} 样本, {train_date_end} 及之前")
        logger.info(f"  验证集: {len(X_val):,} 样本, {val_date_start} ~ {val_date_end}")
        logger.info(f"  测试集: {len(X_test):,} 样本, {test_date_start} 及之后")
        logger.info(f"  Purge gap: 训练/验证之间丢弃 {purge_days} 个交易日, 验证/测试之间丢弃 {purge_days} 个交易日")

        return X_train, y_train, X_val, y_val, X_test, y_test

    def train(self, val_ratio=0.15, test_ratio=0.15, purge_days=5):
        """
        完整训练流程

        修复:
        1. 使用时序划分替代随机 shuffle（避免未来数据泄漏）
        2. 添加 purge gap（避免标签窗口重叠）
        3. 元模型使用验证集 OOF 预测训练（避免 stacking 泄漏）
        4. 在独立测试集上评估真实泛化能力
        """
        # 1. 加载数据（含日期）
        X, y, dates = self.load_cached_features()

        # 2. 时序划分: train / purge / val / purge / test
        logger.info("\n" + "="*80)
        logger.info("📊 时序划分数据集 (带 Purge Gap)...")
        logger.info("="*80)

        X_train, y_train, X_val, y_val, X_test, y_test = self.temporal_split(
            X, y, dates, val_ratio=val_ratio, test_ratio=test_ratio, purge_days=purge_days
        )

        # 3. 训练基础模型（在训练集上训练，验证集上评估）
        self.train_base_models(X_train, y_train, X_val, y_val)

        # 4. 训练元模型（在验证集OOF预测上训练，测试集上评估）
        self.train_meta_model(X_val, y_val, X_test, y_test)

        # 5. 输出特征重要性
        self.feature_names = X_train.columns.tolist()
        self._log_feature_importance(self.feature_names)

        # 6. 保存模型
        model_path = self.save_model()

        logger.info("\n" + "="*80)
        logger.info("🎉 训练完成!")
        logger.info("="*80)

        return model_path

    def _log_feature_importance(self, feature_names: list, top_n: int = 20):
        """
        提取并打印各模型的特征重要性 Top N，保存到 JSON 文件

        Args:
            feature_names: 特征名称列表
            top_n: 打印的前 N 个特征
        """
        logger.info("\n" + "="*80)
        logger.info("📊 特征重要性分析")
        logger.info("="*80)

        all_importances = {}

        for name, model in self.models.items():
            importance = None
            try:
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                elif hasattr(model, 'feature_importance'):
                    importance = model.feature_importance()
            except Exception as e:
                logger.debug(f"  {name} 无法提取特征重要性: {e}")
                continue

            if importance is None:
                continue

            # 构建 (feature, importance) 对并排序
            feat_imp = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
            all_importances[name] = {f: float(v) for f, v in feat_imp}

            logger.info(f"\n🔹 {name} Top {top_n} 特征:")
            for rank, (feat, imp) in enumerate(feat_imp[:top_n], 1):
                logger.info(f"  {rank:2d}. {feat:<35s} {imp:.4f}")

        # 计算平均重要性
        if all_importances:
            avg_importance = {}
            for feat in feature_names:
                values = [imp.get(feat, 0) for imp in all_importances.values()]
                avg_importance[feat] = float(np.mean(values))
            avg_sorted = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)

            logger.info(f"\n🔹 平均特征重要性 Top {top_n}:")
            for rank, (feat, imp) in enumerate(avg_sorted[:top_n], 1):
                logger.info(f"  {rank:2d}. {feat:<35s} {imp:.4f}")

            all_importances['average'] = dict(avg_sorted)

        # 保存到 JSON
        output_dir = Path(PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v39')
        output_dir.mkdir(parents=True, exist_ok=True)
        importance_path = output_dir / f"v390_feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(importance_path, 'w', encoding='utf-8') as f:
            json.dump(all_importances, f, indent=2, ensure_ascii=False)
        logger.info(f"\n💾 特征重要性已保存: {importance_path}")


def main():
    parser = argparse.ArgumentParser(description='V3.9模型训练（基于预计算特征）')
    parser.add_argument('--db-path', type=str, default=str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db'), help='数据库路径')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='验证集比例')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='测试集比例')
    parser.add_argument('--purge-days', type=int, default=5, help='Purge gap天数 (应>=标签前瞻天数)')
    parser.add_argument('--output-dir', type=str, default=str(PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v39'), help='模型输出目录')

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("🚀 V3.9模型训练 - 基于预计算特征缓存 (时序划分, 无数据泄漏)")
    logger.info("="*80)
    logger.info(f"数据库: {args.db_path}")
    logger.info(f"验证集比例: {args.val_ratio}")
    logger.info(f"测试集比例: {args.test_ratio}")
    logger.info(f"Purge gap: {args.purge_days} 天")
    logger.info(f"输出目录: {args.output_dir}")

    # 训练
    trainer = V390CachedTrainer(db_path=args.db_path)
    model_path = trainer.train(
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        purge_days=args.purge_days
    )

    logger.info(f"\n✅ 模型已保存至: {model_path}")


if __name__ == "__main__":
    main()
