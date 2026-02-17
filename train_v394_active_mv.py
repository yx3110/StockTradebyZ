#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.9.4 活跃市值增强模型训练脚本

新增特征 (6个):
- 市场层面: market_active_mv_ratio, market_active_mv_zscore, market_active_mv_trend
- 个股层面: stock_active_mv_rank, stock_relative_liquidity, market_cap_quality_score

数据范围: 2024-01-01 ~ 2025-10-31 (使用最新数据，确保市值数据可用)
测试范围: 2025-11-01 ~ 最新

作者: Claude Code
创建时间: 2025-11-27
"""

import os
import sys
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import json
import pickle
import warnings
import logging
from pathlib import Path
import joblib
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

warnings.filterwarnings('ignore')

# ML模型
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import RobustScaler

# 项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_models.v39.v390_enhanced_feature_ml_system import V390EnhancedFeatureMLSystem

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class V394ActiveMVTrainer:
    """V3.9.4活跃市值增强模型训练器"""

    def __init__(
        self,
        train_start: str = '2024-01-01',
        train_end: str = '2025-10-31',
        test_start: str = '2025-11-01',
        lookahead_days: int = 5,
        sample_ratio: float = 0.3,  # 采样比例（加速训练）
        n_workers: int = 4
    ):
        """
        初始化训练器

        Args:
            train_start: 训练开始日期
            train_end: 训练结束日期
            test_start: 测试开始日期
            lookahead_days: 预测天数
            sample_ratio: 每日采样比例
            n_workers: 并行工作进程数
        """
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.lookahead_days = lookahead_days
        self.sample_ratio = sample_ratio
        self.n_workers = n_workers

        # 初始化ML系统（启用活跃市值特征）
        config = {
            'use_enhanced_features': False,  # 禁用Phase 1失败特征
            'use_phase2_features': False,    # 禁用Phase 2失败特征
            'use_active_mv_features': True   # 启用活跃市值特征
        }
        self.ml_system = V390EnhancedFeatureMLSystem(
            lookahead_days=lookahead_days,
            config=config
        )

        # 数据库路径
        self.db_path = 'data_adapter/stock_data.db'

        # 模型保存路径
        self.model_dir = Path('models/v394')
        self.model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"✅ V3.9.4训练器初始化完成")
        logger.info(f"   训练范围: {train_start} ~ {train_end}")
        logger.info(f"   测试范围: {test_start} ~ 最新")
        logger.info(f"   采样比例: {sample_ratio*100:.0f}%")

    def get_trading_dates(self, start: str, end: str) -> List[str]:
        """获取交易日列表"""
        conn = sqlite3.connect(self.db_path)
        query = """
        SELECT DISTINCT trade_date
        FROM daily_quotes
        WHERE trade_date BETWEEN ? AND ?
        ORDER BY trade_date
        """
        df = pd.read_sql(query, conn, params=(start, end))
        conn.close()
        return df['trade_date'].tolist()

    def get_stock_codes(self, trade_date: str, sample_ratio: float = 1.0) -> List[str]:
        """获取指定日期的股票代码（可采样）"""
        conn = sqlite3.connect(self.db_path)
        query = """
        SELECT s.code
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE dq.trade_date = ?
        AND s.type = 'A股'
        AND s.code NOT LIKE 'ST%'
        AND s.code NOT LIKE '*ST%'
        """
        df = pd.read_sql(query, conn, params=(trade_date,))
        conn.close()

        codes = df['code'].tolist()
        if sample_ratio < 1.0:
            n_sample = max(int(len(codes) * sample_ratio), 100)
            np.random.seed(hash(trade_date) % 2**32)
            codes = np.random.choice(codes, n_sample, replace=False).tolist()

        return codes

    def calculate_label(self, code: str, date: str) -> Optional[float]:
        """计算标签（未来N日收益率）"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
            SELECT dq.close
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE s.code = ?
            AND dq.trade_date >= ?
            ORDER BY dq.trade_date
            LIMIT ?
            """
            df = pd.read_sql(query, conn, params=(code, date, self.lookahead_days + 1))
            conn.close()

            if len(df) < self.lookahead_days + 1:
                return None

            start_price = df.iloc[0]['close']
            end_price = df.iloc[self.lookahead_days]['close']

            if start_price <= 0:
                return None

            return (end_price - start_price) / start_price

        except Exception as e:
            return None

    def extract_sample(self, args: Tuple[str, str]) -> Optional[Dict]:
        """提取单个样本的特征和标签"""
        code, date = args
        try:
            # 提取特征
            features_df = self.ml_system.extract_features(code, date)
            if features_df is None or features_df.empty:
                return None

            # 计算标签
            label = self.calculate_label(code, date)
            if label is None:
                return None

            features = features_df.iloc[0].to_dict()
            features['label'] = label
            features['code'] = code
            features['date'] = date

            return features

        except Exception as e:
            return None

    def prepare_data(self, start_date: str, end_date: str, desc: str = "准备数据") -> pd.DataFrame:
        """准备训练/测试数据"""
        logger.info(f"开始{desc}: {start_date} ~ {end_date}")

        # 获取交易日
        trading_dates = self.get_trading_dates(start_date, end_date)
        logger.info(f"共 {len(trading_dates)} 个交易日")

        # 收集样本
        all_samples = []

        for date in tqdm(trading_dates, desc=desc):
            # 获取当日股票（采样）
            codes = self.get_stock_codes(date, self.sample_ratio)

            # 使用多进程提取特征
            args_list = [(code, date) for code in codes]

            # 使用线程池（避免多进程的pickle问题）
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                results = list(executor.map(self.extract_sample, args_list))

            # 过滤有效结果
            valid_results = [r for r in results if r is not None]
            all_samples.extend(valid_results)

            # 清除缓存（避免内存泄漏）
            if hasattr(self.ml_system, 'active_mv_extractor'):
                self.ml_system.active_mv_extractor.clear_cache()

        logger.info(f"{desc}完成: 共 {len(all_samples)} 个样本")

        return pd.DataFrame(all_samples)

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> lgb.Booster:
        """训练LightGBM模型"""
        logger.info("开始训练模型...")

        # 时序交叉验证
        tscv = TimeSeriesSplit(n_splits=3)

        # LightGBM参数
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_jobs': -1,
            'seed': 42
        }

        # 训练
        train_data = lgb.Dataset(X_train, label=y_train)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )

        logger.info(f"✅ 模型训练完成, 最佳迭代: {model.best_iteration}")

        return model

    def evaluate_model(
        self,
        model: lgb.Booster,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        test_info: pd.DataFrame
    ) -> Dict:
        """评估模型性能"""
        logger.info("开始评估模型...")

        # 预测
        y_pred = model.predict(X_test)

        # 基础指标
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # 方向准确率
        direction_correct = ((y_test > 0) == (y_pred > 0)).mean()

        # IC信息系数
        ic = np.corrcoef(y_test, y_pred)[0, 1]

        # Top 20分析
        test_df = test_info.copy()
        test_df['pred'] = y_pred
        test_df['actual'] = y_test

        # 按日期分组，取每日预测Top 20
        top_returns = []
        for date, group in test_df.groupby('date'):
            top_20 = group.nlargest(20, 'pred')
            top_returns.extend(top_20['actual'].tolist())

        top_20_mean = np.mean(top_returns) * 100
        top_20_win_rate = (np.array(top_returns) > 0).mean() * 100

        # 特征重要性
        importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

        # 活跃市值特征重要性
        active_mv_features = [
            'market_active_mv_ratio', 'market_active_mv_zscore', 'market_active_mv_trend',
            'stock_active_mv_rank', 'stock_relative_liquidity', 'market_cap_quality_score'
        ]
        active_mv_importance = importance[importance['feature'].isin(active_mv_features)]

        results = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'ic': ic,
            'direction_accuracy': direction_correct,
            'top_20_return': top_20_mean,
            'top_20_win_rate': top_20_win_rate,
            'feature_importance': importance,
            'active_mv_importance': active_mv_importance,
            'n_test_samples': len(y_test)
        }

        return results

    def run(self):
        """运行完整的训练流程"""
        logger.info("=" * 60)
        logger.info("V3.9.4 活跃市值增强模型训练")
        logger.info("=" * 60)

        # 1. 准备训练数据
        train_df = self.prepare_data(self.train_start, self.train_end, "训练数据")

        if len(train_df) < 1000:
            logger.error(f"训练数据不足: {len(train_df)} < 1000")
            return None

        # 2. 准备测试数据
        test_df = self.prepare_data(self.test_start, datetime.now().strftime('%Y-%m-%d'), "测试数据")

        if len(test_df) < 100:
            logger.warning(f"测试数据较少: {len(test_df)}")

        # 3. 分离特征和标签
        feature_cols = [c for c in train_df.columns if c not in ['label', 'code', 'date']]
        logger.info(f"特征数量: {len(feature_cols)}")

        X_train = train_df[feature_cols].astype(float)
        y_train = train_df['label'].astype(float)

        X_test = test_df[feature_cols].astype(float)
        y_test = test_df['label'].astype(float)
        test_info = test_df[['code', 'date']]

        # 处理NaN
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)

        logger.info(f"训练集: {len(X_train)} 样本, {len(feature_cols)} 特征")
        logger.info(f"测试集: {len(X_test)} 样本")

        # 4. 训练模型
        model = self.train_model(X_train, y_train)

        # 5. 评估模型
        results = self.evaluate_model(model, X_test, y_test, test_info)

        # 6. 打印结果
        logger.info("\n" + "=" * 60)
        logger.info("模型评估结果")
        logger.info("=" * 60)
        logger.info(f"RMSE: {results['rmse']:.4f}")
        logger.info(f"MAE: {results['mae']:.4f}")
        logger.info(f"R²: {results['r2']:.4f}")
        logger.info(f"IC: {results['ic']:.4f}")
        logger.info(f"方向准确率: {results['direction_accuracy']*100:.2f}%")
        logger.info(f"Top 20 平均收益: {results['top_20_return']:.2f}%")
        logger.info(f"Top 20 胜率: {results['top_20_win_rate']:.2f}%")

        logger.info("\n活跃市值特征重要性:")
        for _, row in results['active_mv_importance'].iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.1f}")

        logger.info("\nTop 20 特征重要性:")
        for i, (_, row) in enumerate(results['feature_importance'].head(20).iterrows()):
            logger.info(f"  {i+1}. {row['feature']}: {row['importance']:.1f}")

        # 7. 保存模型
        model_path = self.model_dir / 'v394_active_mv_model.pkl'
        joblib.dump({
            'model': model,
            'feature_cols': feature_cols,
            'results': results,
            'config': {
                'train_start': self.train_start,
                'train_end': self.train_end,
                'test_start': self.test_start,
                'lookahead_days': self.lookahead_days
            }
        }, model_path)
        logger.info(f"\n✅ 模型已保存: {model_path}")

        return results


def main():
    """主函数"""
    trainer = V394ActiveMVTrainer(
        train_start='2024-01-01',
        train_end='2025-10-31',
        test_start='2025-11-01',
        lookahead_days=5,
        sample_ratio=0.2,  # 20%采样，加速训练
        n_workers=4
    )

    results = trainer.run()

    if results:
        print("\n" + "=" * 60)
        print("V3.9.4 训练完成!")
        print("=" * 60)


if __name__ == '__main__':
    main()
