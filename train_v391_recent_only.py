#!/usr/bin/env python3
"""
V3.91 最近数据版本 - 只用最近3个月数据

核心思路：
1. 只使用最近3个月数据（约60个交易日）
2. 使用滚动窗口验证
3. 简单线性模型避免过拟合
4. 目标：捕捉最新市场模式

"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import RobustScaler
from scipy.stats import spearmanr, rankdata
import warnings
warnings.filterwarnings('ignore')

# 设置日志
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/v391_recent_only_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RecentOnlyV391Trainer:
    """只用最近数据的V3.91训练器"""

    PERIOD_WEIGHTS = {'5d': 0.40, '10d': 0.35, '15d': 0.25}

    # 使用之前表现最好的ultra_simple特征集
    CORE_FEATURES = [
        'supertrend_signal',      # 趋势信号
        'aroon_down_25',          # 趋势强度
        'williams_r_14',          # 超买超卖
        'vwap_deviation',         # 价格偏离
        'cmf_20',                 # 资金流向
        'ichimoku_base',          # 支撑阻力
        'ad_line_change_5',       # 成交量趋势
        'bb_width_20',            # 波动率
        'market_attention_score', # 市场关注度
        'advance_decline_ratio',  # 市场广度
    ]

    def __init__(self, db_path: str = 'data_adapter/stock_data.db'):
        self.db_path = db_path

    def load_recent_data(self, months: int = 3) -> Dict:
        """只加载最近N个月的数据"""
        import sqlite3

        # 计算起始日期
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months * 30)
        start_date_str = start_date.strftime('%Y-%m-%d')

        logger.info("=" * 80)
        logger.info(f"📥 加载最近 {months} 个月数据 (从 {start_date_str} 开始)")
        logger.info("=" * 80)

        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT code, trade_date, features_json as features,
                   label_5d as future_return_5d,
                   label_10d as future_return_10d,
                   label_15d as future_return_15d
            FROM v39_feature_cache
            WHERE trade_date >= ?
            AND features_json IS NOT NULL
            AND label_5d IS NOT NULL
            AND label_10d IS NOT NULL
            AND label_15d IS NOT NULL
        """

        df = pd.read_sql_query(query, conn, params=[start_date_str])
        conn.close()

        logger.info(f"✅ 加载 {len(df):,} 个样本")

        # 解析特征
        features_list = []
        valid_indices = []

        for idx, row in df.iterrows():
            try:
                features = json.loads(row['features'])
                if isinstance(features, dict):
                    features_list.append(features)
                    valid_indices.append(idx)
            except:
                continue

        df_valid = df.loc[valid_indices].copy()
        feature_df = pd.DataFrame(features_list)

        # 只选择核心特征
        available_core = [f for f in self.CORE_FEATURES if f in feature_df.columns]
        logger.info(f"📊 使用 {len(available_core)} 个核心特征")

        for col in available_core:
            df_valid[col] = feature_df[col].values

        # 按时间排序
        df_valid = df_valid.sort_values('trade_date').reset_index(drop=True)

        unique_dates = df_valid['trade_date'].unique()
        logger.info(f"📅 时间范围: {unique_dates[0]} ~ {unique_dates[-1]}")
        logger.info(f"📅 共 {len(unique_dates)} 个交易日")

        return {
            'df': df_valid,
            'feature_names': available_core,
            'unique_dates': unique_dates
        }

    def rolling_window_validation(self, data: Dict, period: str,
                                   train_days: int = 40,
                                   val_days: int = 10) -> Dict:
        """滚动窗口验证"""

        logger.info(f"\n{'='*60}")
        logger.info(f"🔹 {period} 周期 - 滚动窗口验证")
        logger.info(f"    训练窗口: {train_days}天, 验证窗口: {val_days}天")
        logger.info(f"{'='*60}")

        df = data['df']
        feature_names = data['feature_names']
        unique_dates = data['unique_dates']
        label_col = f'future_return_{period}'

        # 需要足够的数据进行滚动验证
        min_dates_required = train_days + val_days
        if len(unique_dates) < min_dates_required:
            logger.warning(f"⚠️ 数据不足: 需要 {min_dates_required} 天，实际 {len(unique_dates)} 天")
            return None

        all_val_predictions = []
        all_val_actuals = []
        all_train_ics = []

        # 滚动窗口
        n_windows = (len(unique_dates) - train_days) // val_days
        logger.info(f"📊 将进行 {n_windows} 次滚动验证")

        for i in range(n_windows):
            # 定义训练和验证日期范围
            train_start_idx = i * val_days
            train_end_idx = train_start_idx + train_days
            val_start_idx = train_end_idx
            val_end_idx = min(val_start_idx + val_days, len(unique_dates))

            if val_end_idx > len(unique_dates):
                break

            train_dates = unique_dates[train_start_idx:train_end_idx]
            val_dates = unique_dates[val_start_idx:val_end_idx]

            # 分割数据
            train_mask = df['trade_date'].isin(train_dates)
            val_mask = df['trade_date'].isin(val_dates)

            X_train = df.loc[train_mask, feature_names].values
            y_train = df.loc[train_mask, label_col].values
            X_val = df.loc[val_mask, feature_names].values
            y_val = df.loc[val_mask, label_col].values

            # 处理缺失值
            X_train = np.nan_to_num(X_train, nan=0.0)
            X_val = np.nan_to_num(X_val, nan=0.0)

            # 标准化
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # 标签排名转换
            y_train_rank = (rankdata(y_train) - 1) / (len(y_train) - 1) - 0.5

            # 训练模型 (强正则化Ridge)
            model = Ridge(alpha=10.0, random_state=42)
            model.fit(X_train_scaled, y_train_rank)

            # 预测
            train_pred = model.predict(X_train_scaled)
            val_pred = model.predict(X_val_scaled)

            # 计算IC
            train_ic = spearmanr(train_pred, y_train)[0]
            val_ic = spearmanr(val_pred, y_val)[0]

            all_train_ics.append(train_ic)
            all_val_predictions.extend(val_pred)
            all_val_actuals.extend(y_val)

            logger.info(f"  窗口 {i+1}: 训练IC={train_ic:.4f}, 验证IC={val_ic:.4f}")

        # 计算总体指标
        overall_val_ic = spearmanr(all_val_predictions, all_val_actuals)[0]
        avg_train_ic = np.mean(all_train_ics)
        ic_gap = (avg_train_ic - overall_val_ic) / avg_train_ic * 100 if avg_train_ic > 0 else 0

        logger.info(f"\n📊 {period} 周期汇总:")
        logger.info(f"  平均训练IC: {avg_train_ic:.4f}")
        logger.info(f"  总体验证IC: {overall_val_ic:.4f}")
        logger.info(f"  IC差距: {ic_gap:.1f}%")

        # 计算Top收益
        val_pred_arr = np.array(all_val_predictions)
        val_actual_arr = np.array(all_val_actuals)
        top_n = min(100, len(val_pred_arr) // 10)
        if top_n > 0:
            top_indices = np.argsort(val_pred_arr)[-top_n:]
            top_return = val_actual_arr[top_indices].mean() * 100
            logger.info(f"  Top{top_n}收益: {top_return:.2f}%")
        else:
            top_return = 0

        return {
            'avg_train_ic': avg_train_ic,
            'overall_val_ic': overall_val_ic,
            'ic_gap': ic_gap,
            'top_return': top_return,
            'n_windows': n_windows
        }

    def train_final_model(self, data: Dict, period: str) -> Dict:
        """训练最终模型（用全部数据）"""

        df = data['df']
        feature_names = data['feature_names']
        label_col = f'future_return_{period}'

        X = df[feature_names].values
        y = df[label_col].values

        # 处理缺失值
        X = np.nan_to_num(X, nan=0.0)

        # 标准化
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        # 标签排名转换
        y_rank = (rankdata(y) - 1) / (len(y) - 1) - 0.5

        # 训练最终模型
        model = Ridge(alpha=10.0, random_state=42)
        model.fit(X_scaled, y_rank)

        return {
            'model': model,
            'scaler': scaler
        }

    def train_all_periods(self, months: int = 3,
                          train_days: int = 40,
                          val_days: int = 10) -> Dict:
        """训练所有周期"""

        logger.info("=" * 80)
        logger.info("🚀 V3.91 最近数据版本训练")
        logger.info("=" * 80)
        logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"数据范围: 最近 {months} 个月")
        logger.info(f"滚动窗口: 训练{train_days}天 + 验证{val_days}天")
        logger.info("=" * 80)

        # 加载数据
        data = self.load_recent_data(months)

        # 滚动窗口验证
        period_results = {}
        for period in ['5d', '10d', '15d']:
            val_result = self.rolling_window_validation(
                data, period, train_days, val_days
            )
            if val_result:
                # 训练最终模型
                final_model = self.train_final_model(data, period)
                period_results[period] = {
                    **val_result,
                    'model': final_model['model'],
                    'scaler': final_model['scaler']
                }

        if not period_results:
            logger.error("❌ 没有足够数据进行训练")
            return None

        # 综合评估
        composite_train_ic = sum(
            period_results[p]['avg_train_ic'] * self.PERIOD_WEIGHTS[p]
            for p in period_results.keys()
        )
        composite_val_ic = sum(
            period_results[p]['overall_val_ic'] * self.PERIOD_WEIGHTS[p]
            for p in period_results.keys()
        )

        if composite_train_ic > 0:
            composite_gap = (composite_train_ic - composite_val_ic) / composite_train_ic * 100
        else:
            composite_gap = 0

        logger.info("\n" + "=" * 80)
        logger.info("📊 综合评估 (滚动窗口验证)")
        logger.info("=" * 80)
        logger.info(f"综合训练IC: {composite_train_ic:.4f}")
        logger.info(f"综合验证IC: {composite_val_ic:.4f}")
        logger.info(f"综合IC差距: {composite_gap:.1f}%")

        if composite_gap < 20:
            logger.info("✅ 过拟合已完全控制! (差距 < 20%)")
        elif composite_gap < 30:
            logger.info("✅ 过拟合控制良好! (差距 < 30%)")
        elif composite_gap < 50:
            logger.info("⚠️ 轻度过拟合 (差距 30-50%)")
        else:
            logger.info("❌ 过拟合较严重 (差距 > 50%)")

        # 计算评分
        score = self.calculate_score(composite_val_ic, composite_gap, period_results)
        logger.info(f"\n🎯 模型评分: {score:.1f}/100")

        return {
            'period_results': period_results,
            'composite_train_ic': composite_train_ic,
            'composite_val_ic': composite_val_ic,
            'composite_gap': composite_gap,
            'feature_names': data['feature_names'],
            'months': months,
            'train_days': train_days,
            'val_days': val_days,
            'score': score
        }

    def calculate_score(self, val_ic: float, gap: float, period_results: Dict) -> float:
        """计算综合评分"""

        # IC评分 (0-40分)
        if val_ic >= 0.15:
            ic_score = 40
        elif val_ic >= 0.10:
            ic_score = 30 + (val_ic - 0.10) / 0.05 * 10
        elif val_ic >= 0.05:
            ic_score = 20 + (val_ic - 0.05) / 0.05 * 10
        elif val_ic >= 0:
            ic_score = val_ic / 0.05 * 20
        else:
            ic_score = 0

        # 过拟合控制评分 (0-30分)
        if gap <= 20:
            gap_score = 30
        elif gap <= 30:
            gap_score = 25
        elif gap <= 50:
            gap_score = 15
        else:
            gap_score = max(0, 15 - (gap - 50) / 10)

        # Top收益评分 (0-30分)
        avg_top_return = np.mean([r.get('top_return', 0) for r in period_results.values()])
        if avg_top_return >= 3:
            return_score = 30
        elif avg_top_return >= 2:
            return_score = 25
        elif avg_top_return >= 1:
            return_score = 20
        elif avg_top_return >= 0:
            return_score = 10
        else:
            return_score = 0

        return ic_score + gap_score + return_score

    def save_model(self, results: Dict, version: str = 'recent_only'):
        """保存模型"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_dir = 'models/v391'
        os.makedirs(model_dir, exist_ok=True)

        filename = f'v391_{version}_{timestamp}.pkl'
        filepath = os.path.join(model_dir, filename)

        save_data = {
            'version': f'v3.91-{version}',
            'timestamp': timestamp,
            'period_models': {},
            'period_weights': self.PERIOD_WEIGHTS,
            'feature_names': results['feature_names'],
            'months': results['months'],
            'metrics': {
                'composite_train_ic': results['composite_train_ic'],
                'composite_val_ic': results['composite_val_ic'],
                'composite_gap': results['composite_gap'],
                'score': results['score']
            },
            'validation_method': 'rolling_window',
            'train_days': results['train_days'],
            'val_days': results['val_days']
        }

        for period, period_result in results['period_results'].items():
            save_data['period_models'][period] = {
                'model': period_result['model'],
                'scaler': period_result['scaler'],
                'avg_train_ic': period_result['avg_train_ic'],
                'overall_val_ic': period_result['overall_val_ic'],
                'top_return': period_result['top_return']
            }

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        logger.info(f"\n✅ 模型已保存: {filepath}")

        # 更新版本历史
        version_file = os.path.join(model_dir, 'VERSION_HISTORY.json')
        if os.path.exists(version_file):
            with open(version_file, 'r') as f:
                version_history = json.load(f)
        else:
            version_history = {'versions': []}

        version_history['updated'] = datetime.now().isoformat()
        version_history['versions'].insert(0, {
            'filename': filename,
            'version': f'v3.91-{version}',
            'timestamp': timestamp,
            'months': results['months'],
            'features': len(results['feature_names']),
            'composite_train_ic': round(results['composite_train_ic'], 4),
            'composite_val_ic': round(results['composite_val_ic'], 4),
            'composite_gap': f"{results['composite_gap']:.1f}%",
            'score': round(results['score'], 1),
            'validation': 'rolling_window',
            'description': f"最近{results['months']}个月数据，滚动窗口验证"
        })

        with open(version_file, 'w') as f:
            json.dump(version_history, f, indent=2)

        return filepath


def main():
    """主函数"""
    trainer = RecentOnlyV391Trainer()

    # 使用最近6个月数据，更小的滚动窗口
    results = trainer.train_all_periods(
        months=6,
        train_days=20,
        val_days=5
    )

    if results:
        trainer.save_model(results, version='recent_3m')

        logger.info("\n" + "=" * 80)
        logger.info("🎉 训练完成!")
        logger.info("=" * 80)
    else:
        logger.error("训练失败")


if __name__ == '__main__':
    main()
