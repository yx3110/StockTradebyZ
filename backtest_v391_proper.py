#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.91 模型回测 - 使用正确的42特征

使用v39_feature_cache获取正确的特征数据进行回测
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
print("V3.91 模型回测 (使用正确特征)")
print("=" * 80)

# 回测参数
BACKTEST_START = '2025-09-01'
BACKTEST_END = '2025-11-20'
TOP_N = 10  # 每天选择top N只股票
HOLDING_DAYS = 5  # 持有天数

# 数据库路径
DB_PATH = 'data_adapter/stock_data.db'

def load_model(model_path):
    """加载模型"""
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"  加载失败: {e}")
        return None

def get_trade_dates(start_date, end_date):
    """获取交易日列表"""
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
    """从v39_feature_cache获取指定日期的所有股票特征"""
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

    # 解析JSON特征
    features_dict = {}
    for _, row in df.iterrows():
        code = row['code']
        try:
            features = json.loads(row['features_json'])
            features_dict[code] = features
        except:
            pass

    if not features_dict:
        return None, None

    # 转换为DataFrame
    features_df = pd.DataFrame(features_dict).T
    features_df = features_df.fillna(0)

    return features_df, list(features_dict.keys())

def predict_with_model(model_data, features_df, feature_columns):
    """使用模型进行预测"""
    if model_data is None or features_df.empty:
        return None

    try:
        # 对齐特征列
        available_features = [col for col in feature_columns if col in features_df.columns]
        if len(available_features) < len(feature_columns) * 0.5:
            print(f"    特征匹配不足: {len(available_features)}/{len(feature_columns)}")
            return None

        # 填充缺失特征
        for col in feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0

        X = features_df[feature_columns].fillna(0)

        # 检查是否是v391_improved结构
        if 'base_models' in model_data and 'meta_models' in model_data:
            predictions = []
            period_weights = model_data.get('period_weights', {'5d': 0.5, '10d': 0.3, '15d': 0.2})

            for period in ['5d', '10d', '15d']:
                if period in model_data['base_models']:
                    base_models = model_data['base_models'][period]
                    meta_model = model_data['meta_models'].get(period)

                    # 获取基础模型预测
                    base_preds = []
                    for name, base_model in base_models.items():
                        try:
                            pred = base_model.predict(X)
                            base_preds.append(pred)
                        except Exception as e:
                            pass

                    if base_preds and meta_model is not None:
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

        # MoE模型结构
        elif 'expert_models' in model_data:
            expert_preds = []
            for expert in model_data['expert_models']:
                if 'model' in expert:
                    try:
                        pred = expert['model'].predict(X)
                        expert_preds.append(pred)
                    except:
                        pass
            if expert_preds:
                return np.mean(expert_preds, axis=0)

    except Exception as e:
        print(f"    预测错误: {e}")

    return None

def get_future_returns(code, entry_date, holding_days):
    """获取未来收益"""
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

def run_backtest(model_name, model_path, trade_dates):
    """运行回测"""
    print(f"\n{'='*60}")
    print(f"回测模型: {model_name}")
    print(f"{'='*60}")

    # 加载模型
    model_data = load_model(model_path)
    if model_data is None:
        return None

    # 获取特征列
    feature_columns = model_data.get('feature_columns', [])
    if not feature_columns:
        print("  无法获取特征列信息!")
        return None

    print(f"  特征数: {len(feature_columns)}")
    print(f"  周期权重: {model_data.get('period_weights', {})}")

    # 统计
    all_trades = []
    prediction_count = 0
    fallback_count = 0

    for i, date in enumerate(trade_dates):
        if i % 10 == 0:
            print(f"  处理: {date} ({i+1}/{len(trade_dates)})")

        # 获取特征
        features_df, stock_codes = get_features_for_date(date)
        if features_df is None:
            continue

        # 预测
        predictions = predict_with_model(model_data, features_df, feature_columns)

        if predictions is not None:
            prediction_count += 1
        else:
            # 回退到简单动量
            fallback_count += 1
            if 'return_5d' in features_df.columns:
                predictions = features_df['return_5d'].values
            else:
                continue

        # 选择top N
        pred_df = pd.DataFrame({
            'code': stock_codes,
            'score': predictions
        })
        pred_df = pred_df.sort_values('score', ascending=False).head(TOP_N)

        # 记录交易
        for _, row in pred_df.iterrows():
            returns, entry_price, exit_price = get_future_returns(
                row['code'], date, HOLDING_DAYS
            )
            if returns is not None:
                all_trades.append({
                    'entry_date': date,
                    'code': row['code'],
                    'score': row['score'],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'returns': returns
                })

    print(f"\n  模型预测次数: {prediction_count}, 回退次数: {fallback_count}")

    if not all_trades:
        return None

    results_df = pd.DataFrame(all_trades)

    # 计算统计指标
    stats = {
        'model': model_name,
        'total_trades': len(results_df),
        'prediction_rate': prediction_count / (prediction_count + fallback_count) * 100 if (prediction_count + fallback_count) > 0 else 0,
        'avg_return': results_df['returns'].mean() * 100,
        'total_return': (1 + results_df['returns']).prod() - 1,
        'win_rate': (results_df['returns'] > 0).mean() * 100,
        'sharpe_ratio': results_df['returns'].mean() / results_df['returns'].std() * np.sqrt(252 / HOLDING_DAYS) if results_df['returns'].std() > 0 else 0,
        'max_return': results_df['returns'].max() * 100,
        'min_return': results_df['returns'].min() * 100,
        'avg_win': results_df[results_df['returns'] > 0]['returns'].mean() * 100 if (results_df['returns'] > 0).any() else 0,
        'avg_loss': results_df[results_df['returns'] < 0]['returns'].mean() * 100 if (results_df['returns'] < 0).any() else 0,
    }

    return stats, results_df

# 主程序
print(f"\n回测配置:")
print(f"  周期: {BACKTEST_START} → {BACKTEST_END}")
print(f"  选股数: Top {TOP_N}")
print(f"  持有天数: {HOLDING_DAYS}")

# 获取交易日
print(f"\n获取交易日...")
trade_dates = get_trade_dates(BACKTEST_START, BACKTEST_END)
print(f"  共 {len(trade_dates)} 个交易日")

if not trade_dates:
    print("没有交易日数据!")
    sys.exit(1)

# 只测试v391_improved模型 (其他模型有序列化问题)
MODEL_PATH = 'models/v391/v391_improved_20251125_232954.pkl'

stats, trades_df = run_backtest('v391_improved', MODEL_PATH, trade_dates)

if stats:
    print("\n" + "=" * 80)
    print("回测结果")
    print("=" * 80)

    print(f"\n模型: {stats['model']}")
    print(f"  模型预测率: {stats['prediction_rate']:.1f}%")
    print(f"  总交易次数: {stats['total_trades']}")
    print(f"  平均收益:   {stats['avg_return']:.2f}%")
    print(f"  胜率:       {stats['win_rate']:.1f}%")
    print(f"  夏普比率:   {stats['sharpe_ratio']:.2f}")
    print(f"  最大单笔收益: {stats['max_return']:.2f}%")
    print(f"  最大单笔亏损: {stats['min_return']:.2f}%")
    print(f"  平均盈利:   {stats['avg_win']:.2f}%")
    print(f"  平均亏损:   {stats['avg_loss']:.2f}%")

    # 按日期汇总
    print("\n日收益分布:")
    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
    daily_returns = trades_df.groupby('entry_date')['returns'].mean()
    print(f"  日均收益: {daily_returns.mean()*100:.2f}%")
    print(f"  日收益标准差: {daily_returns.std()*100:.2f}%")
    print(f"  正收益天数: {(daily_returns > 0).sum()}/{len(daily_returns)}")

    # 保存结果
    trades_df.to_csv('reports/v391_backtest_trades.csv', index=False)
    print(f"\n交易明细已保存到: reports/v391_backtest_trades.csv")
else:
    print("回测失败!")

print(f"\n回测完成!")
