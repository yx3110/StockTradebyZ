#!/usr/bin/env python3
"""
V3.9 / V3.94 / V3.95 模型对比回测（快速版）
使用缓存特征，避免实时计算
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import sqlite3
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class CachedScorer:
    """基于缓存特征的评分器"""

    def __init__(self, version: str):
        self.version = version
        self.db_path = Path(__file__).parent / 'data_adapter' / 'stock_data.db'
        self.model_dir = Path(__file__).parent.parent / 'ml_models' / 'trained_models'

        # 加载模型
        self.models = {}
        self.scaler = None
        self.weights = {}
        self._load_model()

    def _load_model(self):
        """加载模型"""
        if self.version == 'v39':
            self._load_v39_model()
        elif self.version == 'v394':
            self._load_v394_model()
        elif self.version == 'v395':
            self._load_v395_model()

    def _load_v39_model(self):
        """加载V3.9模型"""
        model_path = self.model_dir / 'v390_full_from_cache.pkl'
        if model_path.exists():
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                # V3.9模型直接存储在'model'键中
                self.models = {'main': data.get('model')}
                self.feature_cols = data.get('feature_names', [])
                print(f"✅ V3.9 模型加载成功 ({data.get('n_features', 0)}特征)")
        else:
            print(f"❌ V3.9 模型文件不存在: {model_path}")

    def _load_v394_model(self):
        """加载V3.94模型"""
        # 使用strict_split模型（格式正确）
        model_path = self.model_dir / 'v394' / 'v394_strict_split_20251214_161552.pkl'
        if model_path.exists():
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                # V3.94模型直接存储在'model'键中
                self.models = {'main': data.get('model')}
                self.feature_cols = data.get('feature_cols', [])
                self.v39_features = data.get('v39_features', [])
                self.active_mv_features = data.get('active_mv_features', [])
                print(f"✅ V3.94 模型加载成功 ({len(self.feature_cols)}特征)")
        else:
            print(f"❌ V3.94 模型文件不存在: {model_path}")

    def _load_v395_model(self):
        """加载V3.95滚动模型"""
        weights_path = self.model_dir / 'v395' / 'v395_rolling_weights.json'
        if weights_path.exists():
            with open(weights_path, 'r') as f:
                config = json.load(f)
                self.weights = config.get('ensemble_weights', {})
                self.feature_cols = config.get('feature_cols', [])
                self.market_feature_cols = config.get('market_feature_cols', [])

        scaler_path = self.model_dir / 'v395' / 'v395_rolling_scaler.pkl'
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

        model_names = ['lgb', 'xgb', 'cb', 'rf', 'gb']
        for target in ['3d', '5d', '10d']:
            self.models[target] = {}
            for name in model_names:
                model_path = self.model_dir / 'v395' / f'v395_rolling_{target}_{name}.pkl'
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        self.models[target][name] = pickle.load(f)

        print(f"✅ V3.95 模型加载成功")

    def score(self, date: str) -> Dict[str, float]:
        """评分所有可用股票"""
        if self.version == 'v395':
            return self._score_v395(date)
        else:
            return self._score_v39_v394(date)

    def _score_v39_v394(self, date: str) -> Dict[str, float]:
        """V3.9/V3.94评分（从缓存读取特征）"""
        conn = sqlite3.connect(self.db_path)

        query = """
        SELECT code, features_json, label_5d
        FROM v39_feature_cache
        WHERE trade_date = ?
          AND features_json IS NOT NULL
        """

        df = pd.read_sql_query(query, conn, params=[date])
        conn.close()

        if len(df) == 0:
            return {}

        # 解析特征
        features_list = []
        codes = []
        for _, row in df.iterrows():
            try:
                features = json.loads(row['features_json'])
                features_list.append(features)
                codes.append(row['code'])
            except Exception:
                continue

        if not features_list:
            return {}

        features_df = pd.DataFrame(features_list)

        # 确保特征列顺序与训练时一致
        if hasattr(self, 'feature_cols') and self.feature_cols:
            # 只使用模型训练时的特征
            available_cols = [c for c in self.feature_cols if c in features_df.columns]
            if available_cols:
                features_df = features_df[available_cols]

        X = features_df.fillna(0).values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # 预测
        predictions = np.zeros(len(X))

        if 'main' in self.models and self.models['main'] is not None:
            try:
                predictions = self.models['main'].predict(X)
            except Exception as e:
                print(f"{self.version}预测失败: {e}")
                # 尝试不使用特征子集
                try:
                    X_full = pd.DataFrame(features_list).fillna(0).values
                    X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)
                    predictions = self.models['main'].predict(X_full)
                except Exception as e2:
                    print(f"  重试也失败: {e2}")

        # 转换为百分制
        if len(predictions) > 1:
            ranks = stats.rankdata(predictions)
            percentiles = (ranks - 1) / (len(ranks) - 1) * 100
            scores = 30 + percentiles * 0.6
        else:
            scores = np.array([50.0])

        return dict(zip(codes, scores))

    def _score_v395(self, date: str) -> Dict[str, float]:
        """V3.95评分"""
        conn = sqlite3.connect(self.db_path)

        query = """
        SELECT code, features_json,
               market_return_20d, market_return_10d, market_return_5d,
               market_volatility_20d, market_volatility_10d,
               market_up_ratio_20d, market_up_ratio_10d,
               market_drawdown_20d, market_volume_ratio,
               market_position_20d, market_momentum_20d, market_momentum_5d
        FROM v39_feature_cache
        WHERE trade_date = ?
          AND features_json IS NOT NULL
          AND market_return_20d IS NOT NULL
        """

        df = pd.read_sql_query(query, conn, params=[date])
        conn.close()

        if len(df) == 0:
            return {}

        # 解析特征
        features_list = []
        codes = []
        market_data = []

        for _, row in df.iterrows():
            try:
                features = json.loads(row['features_json'])
                features_list.append(features)
                codes.append(row['code'])
                market_data.append({
                    'market_return_20d': row['market_return_20d'],
                    'market_return_10d': row['market_return_10d'],
                    'market_return_5d': row['market_return_5d'],
                    'market_volatility_20d': row['market_volatility_20d'],
                    'market_volatility_10d': row['market_volatility_10d'],
                    'market_up_ratio_20d': row['market_up_ratio_20d'],
                    'market_up_ratio_10d': row['market_up_ratio_10d'],
                    'market_drawdown_20d': row['market_drawdown_20d'],
                    'market_volume_ratio': row['market_volume_ratio'],
                    'market_position_20d': row['market_position_20d'],
                    'market_momentum_20d': row['market_momentum_20d'],
                    'market_momentum_5d': row['market_momentum_5d']
                })
            except Exception:
                continue

        if not features_list:
            return {}

        # 合并特征
        features_df = pd.DataFrame(features_list)
        market_df = pd.DataFrame(market_data)
        all_features = pd.concat([features_df, market_df], axis=1)

        X = all_features.fillna(0).values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # 标准化
        if self.scaler and 'main' in self.scaler:
            try:
                X = self.scaler['main'].transform(X)
            except Exception:
                pass

        # 多目标预测
        target_weights = {'3d': 0.4, '5d': 0.35, '10d': 0.25}
        combined_pred = np.zeros(len(X))

        for target, t_weight in target_weights.items():
            if target not in self.models:
                continue

            target_pred = np.zeros(len(X))
            total_weight = 0

            for name, model in self.models[target].items():
                try:
                    pred = model.predict(X)
                    weight = self.weights.get(f'label_{target}', {}).get(name, 0.2)
                    target_pred += weight * pred
                    total_weight += weight
                except Exception:
                    continue

            if total_weight > 0:
                target_pred /= total_weight

            combined_pred += t_weight * target_pred

        # 转换为百分制
        if len(combined_pred) > 1:
            ranks = stats.rankdata(combined_pred)
            percentiles = (ranks - 1) / (len(ranks) - 1) * 100
            scores = 30 + percentiles * 0.6
        else:
            scores = np.array([50.0])

        return dict(zip(codes, scores))


class FastBacktester:
    """快速回测器"""

    def __init__(self):
        self.db_path = Path(__file__).parent / 'data_adapter' / 'stock_data.db'

    def get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取交易日列表"""
        conn = sqlite3.connect(self.db_path)
        query = """
        SELECT DISTINCT trade_date
        FROM v39_feature_cache
        WHERE trade_date >= ?
          AND trade_date <= ?
        ORDER BY trade_date
        """
        df = pd.read_sql_query(query, conn, params=[start_date, end_date])
        conn.close()
        return df['trade_date'].tolist()

    def get_returns(self, date: str, holding_days: int = 5) -> Dict[str, float]:
        """获取该日期所有股票的未来收益"""
        conn = sqlite3.connect(self.db_path)

        if holding_days == 3:
            label_col = 'label_3d'
        elif holding_days == 5:
            label_col = 'label_5d'
        elif holding_days == 10:
            label_col = 'label_10d'
        else:
            label_col = 'label_5d'

        query = f"""
        SELECT code, {label_col} as future_return
        FROM v39_feature_cache
        WHERE trade_date = ?
          AND {label_col} IS NOT NULL
        """

        df = pd.read_sql_query(query, conn, params=[date])
        conn.close()

        return dict(zip(df['code'], df['future_return']))

    def backtest(self, scorer: CachedScorer, start_date: str, end_date: str,
                 top_n: int = 10, holding_days: int = 5) -> Dict:
        """执行回测"""
        print(f"\n{'='*60}")
        print(f"回测: {scorer.version.upper()}")
        print(f"期间: {start_date} ~ {end_date}")
        print(f"策略: Top{top_n}, 持仓{holding_days}天")
        print('='*60)

        trading_dates = self.get_trading_dates(start_date, end_date)
        print(f"交易日数: {len(trading_dates)}")

        results = []
        total_return = 0
        win_count = 0
        trade_count = 0

        for i in range(0, len(trading_dates) - holding_days, holding_days):
            date = trading_dates[i]

            # 获取评分
            scores = scorer.score(date)
            if not scores:
                print(f"  {date}: 无评分数据")
                continue

            # 获取未来收益
            returns = self.get_returns(date, holding_days)
            if not returns:
                continue

            # 选择top N
            sorted_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_stocks = [code for code, score in sorted_stocks[:top_n] if code in returns]

            if not top_stocks:
                continue

            # 计算组合收益
            period_returns = [returns[c] for c in top_stocks]
            period_return = np.mean(period_returns)

            total_return += period_return
            trade_count += 1
            if period_return > 0:
                win_count += 1

            results.append({
                'date': date,
                'return': period_return,
                'top5': top_stocks[:5],
                'top5_scores': [scores[c] for c in top_stocks[:5]]
            })

            print(f"  {date}: 收益={period_return:+.2%}, "
                  f"Top5: {top_stocks[:3]}")

        # 汇总
        if trade_count > 0:
            win_rate = win_count / trade_count
            avg_return = total_return / trade_count
            # 复利计算
            cumulative = 1
            for r in results:
                cumulative *= (1 + r['return'])
            cumulative -= 1
        else:
            win_rate = avg_return = cumulative = 0

        summary = {
            'model': scorer.version.upper(),
            'start_date': start_date,
            'end_date': end_date,
            'trade_count': trade_count,
            'win_count': win_count,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'cumulative_return': cumulative,
            'details': results
        }

        print(f"\n{scorer.version.upper()} 结果:")
        print(f"  交易次数: {trade_count}")
        print(f"  胜率: {win_rate:.1%}")
        print(f"  平均收益: {avg_return:.2%}")
        print(f"  累计收益: {cumulative:.2%}")

        return summary


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='V3.9/V3.94/V3.95快速回测')
    parser.add_argument('--start-date', type=str, default='2025-10-01')
    parser.add_argument('--end-date', type=str, default='2025-12-01')
    parser.add_argument('--top-n', type=int, default=10)
    parser.add_argument('--holding-days', type=int, default=5)
    parser.add_argument('--models', type=str, default='v39,v394,v395')

    args = parser.parse_args()

    backtester = FastBacktester()
    models_to_test = args.models.split(',')
    all_results = []

    for version in models_to_test:
        try:
            scorer = CachedScorer(version)
            result = backtester.backtest(
                scorer=scorer,
                start_date=args.start_date,
                end_date=args.end_date,
                top_n=args.top_n,
                holding_days=args.holding_days
            )
            all_results.append(result)
        except Exception as e:
            print(f"❌ {version} 回测失败: {e}")
            import traceback
            traceback.print_exc()

    # 对比结果
    print("\n" + "=" * 80)
    print("模型对比结果")
    print("=" * 80)
    print(f"{'模型':<10} {'交易次数':<10} {'胜率':<10} {'平均收益':<12} {'累计收益':<12}")
    print("-" * 80)

    for result in all_results:
        print(f"{result['model']:<10} "
              f"{result['trade_count']:<10} "
              f"{result['win_rate']:.1%}{'':>5} "
              f"{result['avg_return']:+.2%}{'':>6} "
              f"{result['cumulative_return']:+.2%}")

    # 保存结果
    output_dir = Path(__file__).parent / 'reports' / 'backtest'
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f'v39_v394_v395_fast_{timestamp}.json'

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n结果已保存: {output_path}")


if __name__ == '__main__':
    main()
