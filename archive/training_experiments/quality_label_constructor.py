#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 V3.8质量标签构造器
基于历史预测结果和实际收益率构造质量评分标签

核心思路:
1. 收集历史V3.8预测结果
2. 计算对应的未来实际收益率
3. 基于预测准确性和收益表现构造质量标签
4. 为Level 4 Quality Meta-learner提供训练目标
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QualityLabelConstructor:
    """质量标签构造器"""

    def __init__(self, db_path: str = "data_adapter/stock_data.db"):
        self.db_path = db_path
        self.conn = None

    def connect_db(self):
        """连接数据库"""
        self.conn = sqlite3.connect(self.db_path)

    def disconnect_db(self):
        """断开数据库连接"""
        if self.conn:
            self.conn.close()

    def calculate_future_returns(self, stock_code: str, prediction_date: str,
                               periods: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """
        计算指定股票在预测日期后的未来收益率

        Args:
            stock_code: 股票代码
            prediction_date: 预测日期
            periods: 计算周期列表 [1, 3, 5, 10]日

        Returns:
            Dict: {f'return_{period}d': 收益率}
        """
        try:
            # 获取股票ID
            security_query = "SELECT id FROM securities WHERE code = ?"
            security_result = pd.read_sql_query(security_query, self.conn, params=[stock_code])

            if security_result.empty:
                logger.warning(f"未找到股票代码: {stock_code}")
                return {}

            security_id = security_result['id'].iloc[0]

            # 获取预测日期当天或之前的价格
            price_query = f'''
            SELECT trade_date, close
            FROM daily_quotes
            WHERE security_id = {security_id}
            AND trade_date <= '{prediction_date}'
            ORDER BY trade_date DESC
            LIMIT 1
            '''

            prices = pd.read_sql_query(price_query, self.conn)

            if prices.empty:
                return {}

            prediction_price = prices.iloc[0]['close']

            # 获取未来价格数据
            future_query = f'''
            SELECT trade_date, close
            FROM daily_quotes
            WHERE security_id = {security_id}
            AND trade_date > '{prediction_date}'
            ORDER BY trade_date ASC
            LIMIT 15
            '''

            future_prices = pd.read_sql_query(future_query, self.conn)

            # 计算各期间收益率
            returns = {}
            for period in periods:
                if period <= len(future_prices):
                    future_price = future_prices.iloc[period-1]['close']  # period-1因为索引从0开始
                    return_rate = (future_price - prediction_price) / prediction_price
                    returns[f'return_{period}d'] = return_rate
                else:
                    returns[f'return_{period}d'] = np.nan

            return returns

        except Exception as e:
            logger.error(f"计算收益率失败 {stock_code} {prediction_date}: {e}")
            return {}

    def construct_quality_labels(self, prediction_score: float,
                               actual_returns: Dict[str, float],
                               confidence_score: float = 0.5) -> Dict[str, float]:
        """
        基于预测评分和实际收益率构造质量标签

        Args:
            prediction_score: V3.8预测评分 (0-100)
            actual_returns: 实际收益率字典
            confidence_score: 预测置信度

        Returns:
            Dict: 质量标签字典
        """
        try:
            quality_labels = {}

            # 方法1: 基于预测准确性的质量评分
            for period in [1, 3, 5, 10]:
                return_key = f'return_{period}d'
                if return_key not in actual_returns or np.isnan(actual_returns[return_key]):
                    quality_labels[f'quality_{period}d'] = 0.5  # 默认中等质量
                    continue

                actual_return = actual_returns[return_key]

                # 预测方向判断 (假设评分>50表示看涨)
                predicted_direction = 1 if prediction_score > 50 else -1
                actual_direction = 1 if actual_return > 0 else -1

                # 计算方向正确性
                direction_correct = (predicted_direction == actual_direction)

                # 计算收益率绝对值 (最大截断在50%)
                abs_return = min(abs(actual_return), 0.5)

                # 质量评分计算
                if direction_correct:
                    # 方向正确：基础分0.6 + 收益率奖励
                    quality_score = 0.6 + abs_return * 0.8  # 最高1.0
                else:
                    # 方向错误：基础分0.4 - 收益率惩罚
                    quality_score = 0.4 - abs_return * 0.6  # 最低0.1

                # 考虑置信度影响
                confidence_weight = 0.1
                quality_score += (confidence_score - 0.5) * confidence_weight

                # 确保在合理范围内
                quality_score = np.clip(quality_score, 0.1, 0.95)
                quality_labels[f'quality_{period}d'] = round(quality_score, 3)

            # 方法2: 综合质量评分 (多期间加权)
            period_weights = {1: 0.4, 3: 0.3, 5: 0.2, 10: 0.1}
            weighted_quality = 0
            total_weight = 0

            for period, weight in period_weights.items():
                quality_key = f'quality_{period}d'
                if quality_key in quality_labels:
                    weighted_quality += quality_labels[quality_key] * weight
                    total_weight += weight

            if total_weight > 0:
                quality_labels['quality_overall'] = round(weighted_quality / total_weight, 3)
            else:
                quality_labels['quality_overall'] = 0.5

            return quality_labels

        except Exception as e:
            logger.error(f"构造质量标签失败: {e}")
            return {'quality_overall': 0.5}

    def process_historical_predictions(self, reports_dir: str = "reports/daily_selection_v3.8") -> pd.DataFrame:
        """
        处理历史V3.8预测数据，构造质量标签

        Args:
            reports_dir: V3.8报告目录

        Returns:
            DataFrame: 包含预测和质量标签的数据集
        """
        try:
            self.connect_db()

            reports_path = Path(reports_dir)
            json_files = list(reports_path.glob("analysis_data_*.json"))

            logger.info(f"找到 {len(json_files)} 个V3.8预测文件")

            all_data = []

            for json_file in json_files:
                # 从文件名提取日期
                date_str = json_file.stem.split('_')[-1]  # analysis_data_20250922.json -> 20250922
                prediction_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

                logger.info(f"处理 {prediction_date} 的预测数据")

                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    detailed_stocks = data.get('detailed_stocks', [])

                    for stock in detailed_stocks:
                        stock_code = stock.get('code')
                        final_score = stock.get('final_score', 50)
                        confidence_score = stock.get('confidence_score', 0.5)

                        if not stock_code:
                            continue

                        # 计算未来收益率
                        future_returns = self.calculate_future_returns(
                            stock_code, prediction_date
                        )

                        if not future_returns:
                            continue

                        # 构造质量标签
                        quality_labels = self.construct_quality_labels(
                            final_score, future_returns, confidence_score
                        )

                        # 组合数据
                        record = {
                            'prediction_date': prediction_date,
                            'stock_code': stock_code,
                            'stock_name': stock.get('name', ''),
                            'final_score': final_score,
                            'confidence_score': confidence_score,
                            'short_term_score': stock.get('short_term_score', 50),
                            'medium_term_score': stock.get('medium_term_score', 50),
                            'long_term_score': stock.get('long_term_score', 50),
                            'risk_level': stock.get('risk_level', 'medium'),
                            'strategy': stock.get('strategy', 'V3.8'),
                            **future_returns,
                            **quality_labels
                        }

                        all_data.append(record)

                except Exception as e:
                    logger.error(f"处理文件失败 {json_file}: {e}")
                    continue

            self.disconnect_db()

            if all_data:
                df = pd.DataFrame(all_data)
                logger.info(f"成功构造 {len(df)} 条质量标签数据")
                return df
            else:
                logger.warning("未能构造任何质量标签数据")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"处理历史预测数据失败: {e}")
            self.disconnect_db()
            return pd.DataFrame()

    def save_quality_dataset(self, df: pd.DataFrame, output_path: str = "quality_training_dataset.csv"):
        """保存质量标签数据集"""
        try:
            df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"质量数据集已保存到: {output_path}")

            # 打印数据集统计信息
            print(f"\n📊 质量数据集统计:")
            print(f"总记录数: {len(df)}")
            print(f"股票数量: {df['stock_code'].nunique()}")
            print(f"时间范围: {df['prediction_date'].min()} - {df['prediction_date'].max()}")

            # 质量分布
            for period in [1, 3, 5, 10]:
                col = f'quality_{period}d'
                if col in df.columns:
                    print(f"质量_{period}d: 均值={df[col].mean():.3f}, 标准差={df[col].std():.3f}")

            if 'quality_overall' in df.columns:
                print(f"综合质量: 均值={df['quality_overall'].mean():.3f}, 标准差={df['quality_overall'].std():.3f}")

        except Exception as e:
            logger.error(f"保存数据集失败: {e}")

# 使用示例
if __name__ == "__main__":
    constructor = QualityLabelConstructor()

    # 处理历史预测数据
    df = constructor.process_historical_predictions()

    if not df.empty:
        # 保存数据集
        constructor.save_quality_dataset(df)

        # 显示样本数据
        print(f"\n📋 样本数据:")
        print(df[['prediction_date', 'stock_code', 'final_score', 'return_1d', 'quality_1d', 'quality_overall']].head(10))
    else:
        print("❌ 未能构造质量数据集")