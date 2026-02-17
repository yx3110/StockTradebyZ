#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.90 vs V3.91 模型回测对比
"""

import sys
sys.path.insert(0, '/Users/yangxu/StockTradebyZ')

import pickle
import sqlite3
import json
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("V3.90 vs V3.91 模型回测对比")
print("=" * 80)

# 回测参数
BACKTEST_START = '2025-09-01'
BACKTEST_END = '2025-11-20'
TOP_N = 10
HOLDING_DAYS = 5
DB_PATH = 'data_adapter/stock_data.db'

# 模型配置
MODELS = {
    'v390_full_system': {
        'path': 'models/v39/v390_full_system_20251116_161259.pkl',
        'type': 'ensemble',  # base_models + meta_model
    },
    'v391_improved': {
        'path': 'models/v391/v391_improved_20251125_232954.pkl',
        'type': 'multiperiod',  # base_models + meta_models (5d/10d/15d)
    }
}

def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"  加载失败: {e}")
        return None

def get_trade_dates(start_date, end_date):
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT DISTINCT trade_date
        FROM v39_feature_cache
        WHERE trade_date BETWEEN ? AND ?
        ORDER BY trade_date
    """
    dates = pd.read_sql_query(query, conn, params=[start_date, end_date])
    conn.close()
    return dates['trade_date'].tolist()

def get_features_for_date(date):
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT code, features_json
        FROM v39_feature_cache
        WHERE trade_date = ?
    """
    df = pd.read_sql_query(query, conn, params=[date])
    conn.close()

    if df.empty:
        return None, None

    features_dict = {}
    for _, row in df.iterrows():
        try:
            features = json.loads(row['features_json'])
            features_dict[row['code']] = features
        except:
            pass

    if not features_dict:
        return None, None

    features_df = pd.DataFrame(features_dict).T.fillna(0)
    return features_df, list(features_dict.keys())

def predict_v390(model_data, features_df, feature_names):
    """V3.90 ensemble model prediction"""
    if model_data is None or features_df.empty:
        return None

    try:
        # 对齐特征
        available_features = [col for col in feature_names if col in features_df.columns]
        if len(available_features) < len(feature_names) * 0.5:
            return None

        for col in feature_names:
            if col not in features_df.columns:
                features_df[col] = 0

        X = features_df[feature_names].fillna(0)

        # 获取基础模型预测
        base_models = model_data.get('base_models', {})
        meta_model = model_data.get('meta_model')

        base_preds = []
        for name, model in base_models.items():
            try:
                pred = model.predict(X)
                base_preds.append(pred)
            except:
                pass

        if base_preds and meta_model:
            try:
                meta_features = np.column_stack(base_preds)
                return meta_model.predict(meta_features)
            except:
                return np.mean(base_preds, axis=0)
        elif base_preds:
            return np.mean(base_preds, axis=0)

    except Exception as e:
        print(f"    V3.90预测错误: {e}")
    return None

def predict_v391(model_data, features_df, feature_columns):
    """V3.91 multiperiod model prediction"""
    if model_data is None or features_df.empty:
        return None

    try:
        available_features = [col for col in feature_columns if col in features_df.columns]
        if len(available_features) < len(feature_columns) * 0.5:
            return None

        for col in feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0

        X = features_df[feature_columns].fillna(0)

        predictions = []
        period_weights = model_data.get('period_weights', {'5d': 0.5, '10d': 0.3, '15d': 0.2})

        for period in ['5d', '10d', '15d']:
            if period in model_data.get('base_models', {}):
                base_models = model_data['base_models'][period]
                meta_model = model_data['meta_models'].get(period)

                base_preds = []
                for name, model in base_models.items():
                    try:
                        pred = model.predict(X)
                        base_preds.append(pred)
                    except:
                        pass

                if base_preds and meta_model:
                    try:
                        meta_features = np.column_stack(base_preds)
                        period_pred = meta_model.predict(meta_features)
                        predictions.append((period_pred, period_weights.get(period, 0.33)))
                    except:
                        predictions.append((np.mean(base_preds, axis=0), period_weights.get(period, 0.33)))
                elif base_preds:
                    predictions.append((np.mean(base_preds, axis=0), period_weights.get(period, 0.33)))

        if predictions:
            total_weight = sum(w for _, w in predictions)
            if total_weight > 0:
                final_pred = np.zeros(len(features_df))
                for pred, weight in predictions:
                    final_pred += (weight / total_weight) * pred
                return final_pred

    except Exception as e:
        print(f"    V3.91预测错误: {e}")
    return None

def get_future_returns(code, entry_date, holding_days):
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT trade_date, close
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.code = ?
        AND trade_date >= ?
        ORDER BY trade_date
        LIMIT ?
    """
    prices = pd.read_sql_query(query, conn, params=[code, entry_date, holding_days + 1])
    conn.close()

    if len(prices) >= 2:
        entry_price = prices.iloc[0]['close']
        exit_price = prices.iloc[min(holding_days, len(prices)-1)]['close']
        returns = (exit_price - entry_price) / entry_price
        return returns, entry_price, exit_price
    return None, None, None

def run_backtest(model_name, model_config, trade_dates):
    print(f"\n{'='*60}")
    print(f"回测: {model_name}")
    print(f"{'='*60}")

    model_data = load_model(model_config['path'])
    if model_data is None:
        return None

    # 获取特征列
    if model_config['type'] == 'ensemble':
        feature_cols = model_data.get('feature_names', [])
    else:
        feature_cols = model_data.get('feature_columns', [])

    print(f"  特征数: {len(feature_cols)}")

    all_trades = []
    pred_count = 0
    fallback_count = 0

    for i, date in enumerate(trade_dates):
        if i % 10 == 0:
            print(f"  处理: {date} ({i+1}/{len(trade_dates)})")

        features_df, stock_codes = get_features_for_date(date)
        if features_df is None:
            continue

        # 预测
        if model_config['type'] == 'ensemble':
            predictions = predict_v390(model_data, features_df, feature_cols)
        else:
            predictions = predict_v391(model_data, features_df, feature_cols)

        if predictions is not None:
            pred_count += 1
        else:
            fallback_count += 1
            if 'return_5d' in features_df.columns:
                predictions = features_df['return_5d'].values
            else:
                continue

        # 选择top N
        pred_df = pd.DataFrame({'code': stock_codes, 'score': predictions})
        pred_df = pred_df.sort_values('score', ascending=False).head(TOP_N)

        for _, row in pred_df.iterrows():
            returns, entry_price, exit_price = get_future_returns(row['code'], date, HOLDING_DAYS)
            if returns is not None:
                all_trades.append({
                    'entry_date': date,
                    'code': row['code'],
                    'score': row['score'],
                    'returns': returns
                })

    print(f"\n  预测成功: {pred_count}, 回退: {fallback_count}")

    if not all_trades:
        return None

    results_df = pd.DataFrame(all_trades)

    stats = {
        'model': model_name,
        'total_trades': len(results_df),
        'prediction_rate': pred_count / (pred_count + fallback_count) * 100,
        'avg_return': results_df['returns'].mean() * 100,
        'win_rate': (results_df['returns'] > 0).mean() * 100,
        'sharpe': results_df['returns'].mean() / results_df['returns'].std() * np.sqrt(252/HOLDING_DAYS) if results_df['returns'].std() > 0 else 0,
        'max_return': results_df['returns'].max() * 100,
        'min_return': results_df['returns'].min() * 100,
        'avg_win': results_df[results_df['returns'] > 0]['returns'].mean() * 100 if (results_df['returns'] > 0).any() else 0,
        'avg_loss': results_df[results_df['returns'] < 0]['returns'].mean() * 100 if (results_df['returns'] < 0).any() else 0,
    }

    return stats

# 主程序
print(f"\n回测配置: {BACKTEST_START} → {BACKTEST_END}")
print(f"Top {TOP_N}, 持有{HOLDING_DAYS}天")

trade_dates = get_trade_dates(BACKTEST_START, BACKTEST_END)
print(f"交易日: {len(trade_dates)}天")

if not trade_dates:
    print("无交易日数据!")
    sys.exit(1)

# 运行回测
all_results = []
for name, config in MODELS.items():
    stats = run_backtest(name, config, trade_dates)
    if stats:
        all_results.append(stats)

# 输出对比
print("\n" + "=" * 90)
print("V3.90 vs V3.91 回测对比结果")
print("=" * 90)

if all_results:
    print(f"\n{'模型':<20} {'预测率%':>8} {'交易数':>8} {'平均收益%':>10} {'胜率%':>8} {'夏普比':>8} {'盈亏比':>8}")
    print("-" * 90)

    for r in all_results:
        profit_loss_ratio = abs(r['avg_win'] / r['avg_loss']) if r['avg_loss'] != 0 else 0
        print(f"{r['model']:<20} {r['prediction_rate']:>8.1f} {r['total_trades']:>8} {r['avg_return']:>10.2f} {r['win_rate']:>8.1f} {r['sharpe']:>8.2f} {profit_loss_ratio:>8.2f}")

    # 对比分析
    if len(all_results) >= 2:
        v390 = next((r for r in all_results if 'v390' in r['model']), None)
        v391 = next((r for r in all_results if 'v391' in r['model']), None)

        if v390 and v391:
            print(f"\n{'='*60}")
            print("对比分析:")
            print(f"  收益差异: V3.91 vs V3.90 = {v391['avg_return'] - v390['avg_return']:+.2f}%")
            print(f"  胜率差异: V3.91 vs V3.90 = {v391['win_rate'] - v390['win_rate']:+.1f}%")
            print(f"  夏普差异: V3.91 vs V3.90 = {v391['sharpe'] - v390['sharpe']:+.2f}")

            winner = "V3.91" if v391['avg_return'] > v390['avg_return'] else "V3.90"
            print(f"\n  胜出: {winner}")
else:
    print("回测失败!")

print(f"\n回测完成!")
