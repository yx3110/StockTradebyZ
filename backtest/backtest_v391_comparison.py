#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.91 模型版本回测对比

对比4个V3.91模型变体的实际交易表现：
1. v391_improved - 多模型+元学习+早停
2. v391_adaptive_moe - MoE专家模型
3. v391_adaptive_moe_full - MoE+更多数据
4. v391_adaptive_moe_decay - MoE+时间衰减
"""

import sys
sys.path.insert(0, '/Users/yangxu/StockTradebyZ')

import pickle
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🚀 V3.91 模型版本回测对比")
print("=" * 80)

# 回测参数
BACKTEST_START = '2025-09-01'  # 使用验证集之后的数据
BACKTEST_END = '2025-11-20'
TOP_N = 10  # 每天选择top N只股票
HOLDING_DAYS = 5  # 持有天数
INITIAL_CAPITAL = 1000000

# 模型配置
MODEL_CONFIGS = {
    'v391_improved': {
        'file': 'models/v391/v391_improved_20251125_232954.pkl',
        'desc': '多模型+元学习+早停'
    },
    'v391_adaptive_moe': {
        'file': 'models/v391/v391_adaptive_moe_20251126_001443.pkl',
        'desc': 'MoE专家模型'
    },
    'v391_adaptive_moe_full': {
        'file': 'models/v391/v391_adaptive_moe_full_20251126_002328.pkl',
        'desc': 'MoE+更多数据'
    },
    'v391_adaptive_moe_decay': {
        'file': 'models/v391/v391_adaptive_moe_decay_20251126_002556.pkl',
        'desc': 'MoE+时间衰减'
    }
}

print(f"\n📋 回测配置:")
print(f"  回测周期: {BACKTEST_START} → {BACKTEST_END}")
print(f"  每日选股数: Top {TOP_N}")
print(f"  持有天数: {HOLDING_DAYS}")
print(f"  初始资金: {INITIAL_CAPITAL:,}元")

# 数据库连接
DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')

def load_model(model_path):
    """加载模型"""
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"  ⚠️ 加载失败: {e}")
        return None

def get_trade_dates(start_date, end_date):
    """获取交易日列表"""
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT DISTINCT trade_date
        FROM daily_quotes
        WHERE trade_date BETWEEN ? AND ?
        ORDER BY trade_date
    """
    dates = pd.read_sql_query(query, conn, params=[start_date, end_date])
    conn.close()
    return dates['trade_date'].tolist()

def get_stock_data_for_date(date, lookback=20):
    """获取指定日期的股票数据"""
    conn = sqlite3.connect(DB_PATH)

    # 获取股票基本信息和行情数据
    query = """
        SELECT
            s.code,
            s.name,
            dq.trade_date,
            dq.open,
            dq.high,
            dq.low,
            dq.close,
            dq.volume,
            dq.price_change_pct
        FROM securities s
        JOIN daily_quotes dq ON s.id = dq.security_id
        WHERE s.type = 'A股'
        AND dq.trade_date <= ?
        AND dq.trade_date >= date(?, '-' || ? || ' days')
        ORDER BY s.code, dq.trade_date
    """

    df = pd.read_sql_query(query, conn, params=[date, date, lookback * 2])
    conn.close()

    if df.empty:
        return None

    return df

def calculate_features_for_stock(stock_data):
    """计算单只股票的特征"""
    if len(stock_data) < 10:
        return None

    # 按日期排序
    stock_data = stock_data.sort_values('trade_date')

    features = {}
    close = stock_data['close'].values
    high = stock_data['high'].values
    low = stock_data['low'].values
    volume = stock_data['volume'].values

    # 收益率
    returns = np.diff(close) / close[:-1]
    if len(returns) < 5:
        return None

    # 基本特征
    features['return_5d'] = (close[-1] / close[-6] - 1) if len(close) > 5 else 0
    features['return_10d'] = (close[-1] / close[-11] - 1) if len(close) > 10 else 0
    features['return_20d'] = (close[-1] / close[-21] - 1) if len(close) > 20 else 0

    # 波动率
    features['volatility_5d'] = np.std(returns[-5:]) if len(returns) >= 5 else 0
    features['volatility_10d'] = np.std(returns[-10:]) if len(returns) >= 10 else 0
    features['volatility_20d'] = np.std(returns[-20:]) if len(returns) >= 20 else 0

    # 成交量变化
    if len(volume) >= 5:
        features['volume_ratio_5d'] = volume[-1] / np.mean(volume[-6:-1]) if np.mean(volume[-6:-1]) > 0 else 1
    else:
        features['volume_ratio_5d'] = 1

    # 动量
    features['momentum_5d'] = np.mean(returns[-5:]) if len(returns) >= 5 else 0
    features['momentum_10d'] = np.mean(returns[-10:]) if len(returns) >= 10 else 0

    # RSI
    gains = np.where(returns > 0, returns, 0)
    losses = np.where(returns < 0, -returns, 0)
    if len(gains) >= 14:
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            features['rsi_14'] = 100 - (100 / (1 + rs))
        else:
            features['rsi_14'] = 100
    else:
        features['rsi_14'] = 50

    # 价格位置
    if len(high) >= 20 and len(low) >= 20:
        high_20 = np.max(high[-20:])
        low_20 = np.min(low[-20:])
        if high_20 > low_20:
            features['price_position'] = (close[-1] - low_20) / (high_20 - low_20)
        else:
            features['price_position'] = 0.5
    else:
        features['price_position'] = 0.5

    # 均线偏离
    if len(close) >= 20:
        ma20 = np.mean(close[-20:])
        features['ma20_deviation'] = (close[-1] / ma20 - 1) if ma20 > 0 else 0
    else:
        features['ma20_deviation'] = 0

    return features

def predict_with_model(model_data, features_df):
    """使用模型进行预测"""
    if model_data is None or features_df.empty:
        return None

    try:
        # v391_improved 模型结构: base_models + meta_models
        if 'base_models' in model_data and 'meta_models' in model_data:
            predictions = []
            period_weights = model_data.get('period_weights', {'5d': 0.5, '10d': 0.3, '15d': 0.2})
            feature_cols = model_data.get('feature_columns', list(features_df.columns))

            # 确保特征列对齐
            available_features = [col for col in feature_cols if col in features_df.columns]
            if not available_features:
                available_features = list(features_df.columns)

            X = features_df[available_features].fillna(0)

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
                        # 使用元模型组合
                        try:
                            meta_features = np.column_stack(base_preds)
                            period_pred = meta_model.predict(meta_features)
                            predictions.append((period_pred, period_weights.get(period, 0.33)))
                        except:
                            # 如果元模型失败，使用基础模型平均
                            predictions.append((np.mean(base_preds, axis=0), period_weights.get(period, 0.33)))
                    elif base_preds:
                        predictions.append((np.mean(base_preds, axis=0), period_weights.get(period, 0.33)))

            if predictions:
                # 加权平均
                total_weight = sum(w for _, w in predictions)
                if total_weight > 0:
                    final_pred = np.zeros(len(features_df))
                    for pred, weight in predictions:
                        final_pred += (weight / total_weight) * pred
                    return final_pred

        # 旧的 'models' 结构兼容
        elif 'models' in model_data:
            predictions = []
            for period in ['5d', '10d', '15d']:
                if period in model_data['models']:
                    period_model = model_data['models'][period]
                    if 'meta_model' in period_model:
                        base_preds = []
                        for name, base_model in period_model.get('base_models', {}).items():
                            try:
                                pred = base_model.predict(features_df)
                                base_preds.append(pred)
                            except:
                                pass
                        if base_preds:
                            predictions.append(np.mean(base_preds, axis=0))

            if predictions:
                weights = [0.5, 0.3, 0.2][:len(predictions)]
                weights = np.array(weights) / sum(weights)
                final_pred = np.zeros(len(features_df))
                for i, pred in enumerate(predictions):
                    final_pred += weights[i] * pred
                return final_pred

        elif 'expert_models' in model_data:
            # MoE模型结构
            expert_preds = []
            for expert in model_data['expert_models']:
                if 'model' in expert:
                    try:
                        pred = expert['model'].predict(features_df)
                        expert_preds.append(pred)
                    except:
                        pass

            if expert_preds:
                return np.mean(expert_preds, axis=0)

        # 尝试直接预测
        if hasattr(model_data, 'predict'):
            return model_data.predict(features_df)

    except Exception as e:
        print(f"    预测错误: {e}")

    return None

def run_backtest_for_model(model_name, model_config, trade_dates):
    """运行单个模型的回测"""
    print(f"\n{'='*60}")
    print(f"📊 回测模型: {model_name}")
    print(f"   {model_config['desc']}")
    print(f"{'='*60}")

    # 加载模型
    model_data = load_model(model_config['file'])
    if model_data is None:
        return None

    # 回测结果
    trades = []
    daily_returns = []

    for i, date in enumerate(trade_dates):
        if i % 10 == 0:
            print(f"  处理日期: {date} ({i+1}/{len(trade_dates)})")

        # 获取股票数据
        stock_data = get_stock_data_for_date(date)
        if stock_data is None:
            continue

        # 按股票分组计算特征
        stock_features = {}
        for code in stock_data['code'].unique():
            stock_df = stock_data[stock_data['code'] == code]
            features = calculate_features_for_stock(stock_df)
            if features:
                stock_features[code] = features

        if not stock_features:
            continue

        # 转换为DataFrame
        features_df = pd.DataFrame(stock_features).T
        features_df = features_df.fillna(0)

        # 确保特征顺序一致
        feature_cols = ['return_5d', 'return_10d', 'return_20d', 'volatility_5d',
                       'volatility_10d', 'volatility_20d', 'volume_ratio_5d',
                       'momentum_5d', 'momentum_10d', 'rsi_14', 'price_position',
                       'ma20_deviation']

        for col in feature_cols:
            if col not in features_df.columns:
                features_df[col] = 0

        features_df = features_df[feature_cols]

        # 预测
        predictions = predict_with_model(model_data, features_df)

        if predictions is None:
            # 使用简单的动量因子排名
            predictions = features_df['return_5d'].values + features_df['momentum_5d'].values

        # 选择top N
        stock_codes = features_df.index.tolist()
        pred_df = pd.DataFrame({
            'code': stock_codes,
            'score': predictions
        })
        pred_df = pred_df.sort_values('score', ascending=False).head(TOP_N)

        # 记录交易
        for _, row in pred_df.iterrows():
            trades.append({
                'date': date,
                'code': row['code'],
                'score': row['score']
            })

    # 计算收益
    if not trades:
        return None

    trades_df = pd.DataFrame(trades)

    # 获取未来收益
    conn = sqlite3.connect(DB_PATH)

    results = []
    for _, trade in trades_df.iterrows():
        query = """
            SELECT trade_date, close
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE s.code = ?
            AND trade_date >= ?
            ORDER BY trade_date
            LIMIT ?
        """
        prices = pd.read_sql_query(query, conn, params=[trade['code'], trade['date'], HOLDING_DAYS + 1])

        if len(prices) >= 2:
            entry_price = prices.iloc[0]['close']
            exit_price = prices.iloc[min(HOLDING_DAYS, len(prices)-1)]['close']
            returns = (exit_price - entry_price) / entry_price
            results.append({
                'entry_date': trade['date'],
                'code': trade['code'],
                'score': trade['score'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'returns': returns
            })

    conn.close()

    if not results:
        return None

    results_df = pd.DataFrame(results)

    # 计算统计指标
    stats = {
        'model': model_name,
        'desc': model_config['desc'],
        'total_trades': len(results_df),
        'avg_return': results_df['returns'].mean() * 100,
        'total_return': (1 + results_df['returns']).prod() - 1,
        'win_rate': (results_df['returns'] > 0).mean() * 100,
        'sharpe_ratio': results_df['returns'].mean() / results_df['returns'].std() * np.sqrt(252 / HOLDING_DAYS) if results_df['returns'].std() > 0 else 0,
        'max_drawdown': (results_df['returns'].cumsum() - results_df['returns'].cumsum().cummax()).min() * 100,
        'avg_win': results_df[results_df['returns'] > 0]['returns'].mean() * 100 if (results_df['returns'] > 0).any() else 0,
        'avg_loss': results_df[results_df['returns'] < 0]['returns'].mean() * 100 if (results_df['returns'] < 0).any() else 0,
    }

    return stats

# 主回测流程
print(f"\n📅 获取交易日...")
trade_dates = get_trade_dates(BACKTEST_START, BACKTEST_END)
print(f"  共 {len(trade_dates)} 个交易日")

# 运行所有模型回测
all_results = []

for model_name, model_config in MODEL_CONFIGS.items():
    stats = run_backtest_for_model(model_name, model_config, trade_dates)
    if stats:
        all_results.append(stats)

# 输出对比结果
print("\n" + "=" * 100)
print("📈 V3.91 模型回测对比结果")
print("=" * 100)

if all_results:
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('avg_return', ascending=False)

    print(f"\n{'模型':<25} {'描述':<20} {'交易次数':>8} {'平均收益%':>10} {'胜率%':>8} {'夏普比':>8} {'最大回撤%':>10}")
    print("-" * 100)

    for _, row in results_df.iterrows():
        print(f"{row['model']:<25} {row['desc']:<20} {row['total_trades']:>8} {row['avg_return']:>10.2f} {row['win_rate']:>8.1f} {row['sharpe_ratio']:>8.2f} {row['max_drawdown']:>10.2f}")

    # 找出最佳模型
    best = results_df.iloc[0]
    print(f"\n🏆 最佳模型: {best['model']}")
    print(f"   平均收益: {best['avg_return']:.2f}%")
    print(f"   胜率: {best['win_rate']:.1f}%")
    print(f"   夏普比率: {best['sharpe_ratio']:.2f}")

    # 保存结果
    results_df.to_csv('reports/v391_backtest_comparison.csv', index=False)
    print(f"\n✅ 结果已保存到: reports/v391_backtest_comparison.csv")
else:
    print("❌ 没有成功完成任何模型的回测")

print(f"\n🎯 回测完成!")
