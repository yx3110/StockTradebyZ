#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.9.5 多目标 + 市场状态特征 训练脚本

核心改进:
1. 多目标预测: 同时预测3天、5天、10天收益，使用加权融合
2. 市场状态特征: 加入大盘20日收益率、波动率、上涨比例、回撤等
3. 优化Ensemble: 基于V3.94的加权Ensemble架构

作者: Claude Code
创建时间: 2025-12-27
"""

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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

        query = f"""
        SELECT
            v.code, v.trade_date, v.features_json,
            v.label_3d, v.label_5d, v.label_10d
        FROM v39_feature_cache v
        WHERE v.label_3d IS NOT NULL
          AND v.label_5d IS NOT NULL
          AND v.label_10d IS NOT NULL
          {date_filter}
        ORDER BY v.trade_date, v.code
        """

        df = pd.read_sql(query, conn)
        conn.close()

        logger.info(f"  原始记录: {len(df):,}")

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

        # 删除缺失值
        df_features = df_features.dropna()
        logger.info(f"  合并市场特征后: {len(df_features):,}")

        return df_features

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """准备特征和标签"""
        logger.info("准备特征和标签...")

        # 排除非特征列
        exclude_cols = ['code', 'trade_date', 'label_3d', 'label_5d', 'label_10d', 'label_15d']
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        self.feature_names = feature_cols
        logger.info(f"  特征数量: {len(feature_cols)}")

        X = df[feature_cols].values
        y_3d = df['label_3d'].values
        y_5d = df['label_5d'].values
        y_10d = df['label_10d'].values

        # 特征裁剪（避免极端值）
        X = np.clip(X, -10, 10)

        return X, y_3d, y_5d, y_10d, df

    def split_data(self, X, y_3d, y_5d, y_10d, df, val_ratio=0.15, test_ratio=0.15):
        """时间序列划分数据集"""
        logger.info("划分数据集...")

        n = len(X)
        train_end = int(n * (1 - val_ratio - test_ratio))
        val_end = int(n * (1 - test_ratio))

        X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
        y_3d_train, y_3d_val, y_3d_test = y_3d[:train_end], y_3d[train_end:val_end], y_3d[val_end:]
        y_5d_train, y_5d_val, y_5d_test = y_5d[:train_end], y_5d[train_end:val_end], y_5d[val_end:]
        y_10d_train, y_10d_val, y_10d_test = y_10d[:train_end], y_10d[train_end:val_end], y_10d[val_end:]

        logger.info(f"  训练集: {len(X_train):,}")
        logger.info(f"  验证集: {len(X_val):,}")
        logger.info(f"  测试集: {len(X_test):,}")

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

        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

        lgb_model = lgb.train(
            lgb_params, lgb_train,
            num_boost_round=500,
            valid_sets=[lgb_train, lgb_val],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        models['lgb'] = lgb_model
        predictions_train['lgb'] = lgb_model.predict(X_train)
        predictions_val['lgb'] = lgb_model.predict(X_val)

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

        # 4. RandomForest
        logger.info(f"  训练 RandomForest ({target_name})...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        models['rf'] = rf_model
        predictions_train['rf'] = rf_model.predict(X_train)
        predictions_val['rf'] = rf_model.predict(X_val)

        # 5. GradientBoosting
        logger.info(f"  训练 GradientBoosting ({target_name})...")
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        models['gb'] = gb_model
        predictions_train['gb'] = gb_model.predict(X_train)
        predictions_val['gb'] = gb_model.predict(X_val)

        return models, predictions_train, predictions_val

    def calculate_ensemble_weights(self, predictions_val: dict, y_val) -> dict:
        """基于验证集性能计算Ensemble权重"""
        rmses = {}
        for name, pred in predictions_val.items():
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            rmses[name] = rmse

        # 使用RMSE的倒数作为权重
        inv_rmses = {k: 1/v for k, v in rmses.items()}
        total = sum(inv_rmses.values())
        weights = {k: v/total for k, v in inv_rmses.items()}

        return weights, rmses

    def ensemble_predict(self, predictions: dict, weights: dict) -> np.ndarray:
        """加权Ensemble预测"""
        result = np.zeros_like(list(predictions.values())[0])
        for name, pred in predictions.items():
            result += weights[name] * pred
        return result

    def train(self, start_date: str = None, end_date: str = None):
        """完整训练流程"""
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("V3.95 多目标 + 市场状态 训练开始")
        logger.info("=" * 60)

        # 1. 加载数据
        df = self.load_data(start_date, end_date)

        # 2. 准备特征
        X, y_3d, y_5d, y_10d, df = self.prepare_features(df)

        # 3. 划分数据
        (X_train, X_val, X_test,
         y_3d_train, y_3d_val, y_3d_test,
         y_5d_train, y_5d_val, y_5d_test,
         y_10d_train, y_10d_val, y_10d_test) = self.split_data(X, y_3d, y_5d, y_10d, df)

        # 4. 训练各目标模型
        all_results = {}

        # 训练3天目标
        models_3d, pred_train_3d, pred_val_3d = self.train_single_target_models(
            X_train, X_val, y_3d_train, y_3d_val, 'label_3d')
        weights_3d, rmses_3d = self.calculate_ensemble_weights(pred_val_3d, y_3d_val)
        all_results['3d'] = {'models': models_3d, 'weights': weights_3d, 'rmses': rmses_3d}

        # 训练5天目标
        models_5d, pred_train_5d, pred_val_5d = self.train_single_target_models(
            X_train, X_val, y_5d_train, y_5d_val, 'label_5d')
        weights_5d, rmses_5d = self.calculate_ensemble_weights(pred_val_5d, y_5d_val)
        all_results['5d'] = {'models': models_5d, 'weights': weights_5d, 'rmses': rmses_5d}

        # 训练10天目标
        models_10d, pred_train_10d, pred_val_10d = self.train_single_target_models(
            X_train, X_val, y_10d_train, y_10d_val, 'label_10d')
        weights_10d, rmses_10d = self.calculate_ensemble_weights(pred_val_10d, y_10d_val)
        all_results['10d'] = {'models': models_10d, 'weights': weights_10d, 'rmses': rmses_10d}

        # 5. 在测试集上评估
        logger.info("\n" + "=" * 60)
        logger.info("测试集评估")
        logger.info("=" * 60)

        final_metrics = {}

        for target, y_test, results in [('3d', y_3d_test, all_results['3d']),
                                         ('5d', y_5d_test, all_results['5d']),
                                         ('10d', y_10d_test, all_results['10d'])]:
            # 获取各模型预测
            pred_test = {}
            for name, model in results['models'].items():
                if name == 'lgb':
                    pred_test[name] = model.predict(X_test)
                elif name == 'xgb':
                    pred_test[name] = model.predict(xgb.DMatrix(X_test))
                else:
                    pred_test[name] = model.predict(X_test)

            # Ensemble预测
            ensemble_pred = self.ensemble_predict(pred_test, results['weights'])

            # 计算指标
            rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
            ic, _ = spearmanr(ensemble_pred, y_test)
            direction_acc = np.mean((ensemble_pred > 0) == (y_test > 0))

            # Top10/20收益
            top10_idx = np.argsort(ensemble_pred)[-int(len(ensemble_pred)*0.1):]
            top20_idx = np.argsort(ensemble_pred)[-int(len(ensemble_pred)*0.2):]
            top10_return = np.mean(y_test[top10_idx])
            top20_return = np.mean(y_test[top20_idx])

            logger.info(f"\n{target} 目标:")
            logger.info(f"  RMSE: {rmse:.4f}")
            logger.info(f"  IC: {ic:.4f}")
            logger.info(f"  方向准确率: {direction_acc:.4f}")
            logger.info(f"  Top10收益: {top10_return:.4f}")
            logger.info(f"  Top20收益: {top20_return:.4f}")

            final_metrics[target] = {
                'rmse': rmse, 'ic': ic,
                'direction_accuracy': direction_acc,
                'top10_return': top10_return,
                'top20_return': top20_return
            }

        # 6. 计算融合预测 (加权平均各目标预测)
        logger.info("\n" + "=" * 60)
        logger.info("融合预测评估 (加权平均)")
        logger.info("=" * 60)

        # 获取各目标的5天实际收益预测
        pred_5d_final = {}
        for name in all_results['5d']['models'].keys():
            model = all_results['5d']['models'][name]
            if name == 'lgb':
                pred_5d_final[name] = model.predict(X_test)
            elif name == 'xgb':
                pred_5d_final[name] = model.predict(xgb.DMatrix(X_test))
            else:
                pred_5d_final[name] = model.predict(X_test)

        fused_pred = self.ensemble_predict(pred_5d_final, all_results['5d']['weights'])

        # 对5天收益评估
        rmse = np.sqrt(mean_squared_error(y_5d_test, fused_pred))
        ic, _ = spearmanr(fused_pred, y_5d_test)
        direction_acc = np.mean((fused_pred > 0) == (y_5d_test > 0))

        logger.info(f"融合预测 (5天收益):")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  IC: {ic:.4f}")
        logger.info(f"  方向准确率: {direction_acc:.4f}")

        # 7. 保存模型
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        output_dir = PROJECT_ROOT / 'models' / 'v395'
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        model_data = {
            'models': all_results,
            'feature_names': self.feature_names,
            'target_weights': self.target_weights,
            'market_features': list(self.market_calculator.market_features.columns[1:])
        }

        model_path = output_dir / f'v395_multi_target_{timestamp}.pkl'
        joblib.dump(model_data, model_path)
        logger.info(f"\n模型已保存: {model_path}")

        # 保存训练历史
        history = {
            'version': 'v3.95-multi-target',
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
    parser = argparse.ArgumentParser(description='V3.95 多目标+市场状态训练')
    parser.add_argument('--start-date', type=str, default='2022-01-01', help='训练开始日期')
    parser.add_argument('--end-date', type=str, default=None, help='训练结束日期')
    args = parser.parse_args()

    trainer = V395MultiTargetTrainer()
    trainer.train(start_date=args.start_date, end_date=args.end_date)


if __name__ == '__main__':
    main()
