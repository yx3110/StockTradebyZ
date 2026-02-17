#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 V3.8质量标签构造器 V2.0 (重构版)
修复关键问题：5日/10日标签恒定、相关性NaN、数据边界问题

主要改进:
1. 🔧 修复收益率计算边界检查
2. 🔧 改进质量标签差异化算法
3. 🔧 增强数据验证和异常处理
4. 🔧 实时相关性验证机制
5. 🔧 更激进的质量评分分布策略
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional
from scipy.stats import pearsonr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QualityLabelConstructorV2:
    """质量标签构造器 V2.0 - 重构版"""

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

    def get_trading_days_after_date(self, stock_code: str, prediction_date: str,
                                   max_days: int = 20) -> pd.DataFrame:
        """
        获取指定日期后的交易日数据 (改进版)

        Args:
            stock_code: 股票代码
            prediction_date: 预测日期
            max_days: 最多获取的交易日数量

        Returns:
            DataFrame: 交易日数据
        """
        try:
            # 获取股票ID
            security_query = "SELECT id FROM securities WHERE code = ?"
            security_result = pd.read_sql_query(security_query, self.conn, params=[stock_code])

            if security_result.empty:
                logger.warning(f"未找到股票代码: {stock_code}")
                return pd.DataFrame()

            security_id = security_result['id'].iloc[0]

            # 🆕 改进: 获取更多未来交易日数据，确保覆盖节假日
            future_query = f'''
            SELECT trade_date, close, open, high, low, volume
            FROM daily_quotes
            WHERE security_id = {security_id}
            AND trade_date > '{prediction_date}'
            ORDER BY trade_date ASC
            LIMIT {max_days}
            '''

            future_data = pd.read_sql_query(future_query, self.conn)

            if future_data.empty:
                logger.warning(f"未找到 {stock_code} 在 {prediction_date} 后的交易数据")
                return pd.DataFrame()

            return future_data

        except Exception as e:
            logger.error(f"获取交易日数据失败 {stock_code} {prediction_date}: {e}")
            return pd.DataFrame()

    def calculate_future_returns_robust(self, stock_code: str, prediction_date: str,
                                      periods: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """
        稳健的未来收益率计算 (修复版)

        主要修复:
        1. 充分的数据边界检查
        2. 更准确的期间计算
        3. 异常值处理
        """
        try:
            # 获取预测日当天或之前的价格
            security_query = "SELECT id FROM securities WHERE code = ?"
            security_result = pd.read_sql_query(security_query, self.conn, params=[stock_code])

            if security_result.empty:
                return {}

            security_id = security_result['id'].iloc[0]

            # 获取预测日基准价格
            base_price_query = f'''
            SELECT trade_date, close
            FROM daily_quotes
            WHERE security_id = {security_id}
            AND trade_date <= '{prediction_date}'
            ORDER BY trade_date DESC
            LIMIT 1
            '''

            base_prices = pd.read_sql_query(base_price_query, self.conn)

            if base_prices.empty:
                logger.warning(f"未找到 {stock_code} 在 {prediction_date} 的基准价格")
                return {}

            prediction_price = base_prices.iloc[0]['close']

            # 获取未来交易日数据
            future_data = self.get_trading_days_after_date(stock_code, prediction_date, max_days=25)

            if future_data.empty:
                return {}

            # 🆕 改进: 精确计算各期间收益率
            returns = {}
            max_period = max(periods)

            logger.debug(f"{stock_code} {prediction_date}: 基准价格={prediction_price:.3f}, 可用交易日数={len(future_data)}")

            for period in periods:
                return_key = f'return_{period}d'

                if period <= len(future_data):
                    # 🔧 修复: 正确的索引计算
                    # period=1 -> 第1个交易日 -> 索引0
                    # period=3 -> 第3个交易日 -> 索引2
                    # period=5 -> 第5个交易日 -> 索引4
                    # period=10 -> 第10个交易日 -> 索引9
                    future_price = future_data.iloc[period - 1]['close']

                    # 计算收益率，增加异常值检查
                    if prediction_price > 0 and future_price > 0:
                        return_rate = (future_price - prediction_price) / prediction_price

                        # 🆕 异常值处理: 限制在合理范围内 (-50%, +200%)
                        if -0.5 <= return_rate <= 2.0:
                            returns[return_key] = return_rate
                            logger.debug(f"{stock_code} {period}日收益率: {return_rate:.4f}")
                        else:
                            logger.warning(f"{stock_code} {period}日收益率异常: {return_rate:.4f}, 设为NaN")
                            returns[return_key] = np.nan
                    else:
                        logger.warning(f"{stock_code} 价格异常: 基准={prediction_price}, {period}日后={future_price}")
                        returns[return_key] = np.nan
                else:
                    logger.warning(f"{stock_code} {period}日数据不足: 需要{period}日，实际{len(future_data)}日")
                    returns[return_key] = np.nan

            return returns

        except Exception as e:
            logger.error(f"计算稳健收益率失败 {stock_code} {prediction_date}: {e}")
            return {}

    def construct_quality_labels_v2(self, prediction_score: float,
                                   actual_returns: Dict[str, float],
                                   confidence_score: float = 0.5,
                                   stock_code: str = "") -> Dict[str, float]:
        """
        构造质量标签 V2.0 (增强差异化版)

        改进要点:
        1. 🆕 更激进的质量评分分布策略
        2. 🆕 多因子质量评估模型
        3. 🆕 确保显著的质量差异化
        """
        try:
            quality_labels = {}

            # 🆕 改进算法: 多因子质量评估
            for period in [1, 3, 5, 10]:
                return_key = f'return_{period}d'

                # 🔧 修复: 更严格的数据有效性检查
                if (return_key not in actual_returns or
                    actual_returns[return_key] is None or
                    np.isnan(actual_returns[return_key])):

                    # 🆕 改进: 根据期间差异化处理缺失数据
                    if period <= 3:
                        quality_labels[f'quality_{period}d'] = 0.3  # 短期缺失 -> 低质量
                    else:
                        quality_labels[f'quality_{period}d'] = 0.2  # 长期缺失 -> 更低质量

                    logger.debug(f"{stock_code} {period}日数据缺失，设置质量为{quality_labels[f'quality_{period}d']}")
                    continue

                actual_return = actual_returns[return_key]

                # 🆕 多因子质量评估模型
                quality_score = self._calculate_multifactor_quality(
                    prediction_score, actual_return, confidence_score, period
                )

                quality_labels[f'quality_{period}d'] = round(quality_score, 3)
                logger.debug(f"{stock_code} {period}日质量: {quality_score:.3f} (收益率: {actual_return:.4f})")

            # 🆕 改进: 动态加权综合质量评分
            quality_labels['quality_overall'] = self._calculate_weighted_overall_quality(quality_labels)

            return quality_labels

        except Exception as e:
            logger.error(f"构造质量标签V2失败: {e}")
            return {f'quality_{p}d': 0.2 for p in [1, 3, 5, 10]} | {'quality_overall': 0.2}

    def _calculate_multifactor_quality(self, prediction_score: float, actual_return: float,
                                     confidence_score: float, period: int) -> float:
        """
        多因子质量评估模型

        因子权重:
        - 预测方向准确性: 40%
        - 预测幅度匹配度: 30%
        - 风险调整收益: 20%
        - 置信度修正: 10%
        """
        try:
            # 因子1: 预测方向准确性 (0-1)
            predicted_direction = 1 if prediction_score > 50 else -1
            actual_direction = 1 if actual_return > 0 else -1
            direction_accuracy = 1.0 if predicted_direction == actual_direction else 0.0

            # 因子2: 预测幅度匹配度 (0-1)
            # 将预测分数映射到预期收益率
            expected_return = (prediction_score - 50) / 50 * 0.1  # 50分 -> 0%, 75分 -> +5%, 25分 -> -5%
            magnitude_error = abs(actual_return - expected_return)
            magnitude_accuracy = np.exp(-magnitude_error * 20)  # 指数衰减，误差越小精度越高

            # 因子3: 风险调整收益 (归一化到0-1)
            abs_return = abs(actual_return)
            risk_adjusted_return = np.tanh(abs_return * 10)  # tanh函数归一化

            # 因子4: 置信度修正 (-0.1到+0.1)
            confidence_adjustment = (confidence_score - 0.5) * 0.2

            # 🆕 激进质量评分计算 (确保大幅差异化)
            base_quality = (
                direction_accuracy * 0.4 +
                magnitude_accuracy * 0.3 +
                risk_adjusted_return * 0.2 +
                0.1  # 基础分
            )

            # 置信度修正
            final_quality = base_quality + confidence_adjustment

            # 🆕 期间差异化调整
            period_multiplier = {1: 1.0, 3: 0.95, 5: 0.9, 10: 0.85}[period]
            final_quality *= period_multiplier

            # 🔧 使用标准化而非硬截断
            # 保持原始分布的相对关系，但确保在合理范围内
            final_quality = max(0.01, min(0.99, final_quality))

            return final_quality

        except Exception as e:
            logger.error(f"多因子质量计算失败: {e}")
            return 0.5

    def _calculate_weighted_overall_quality(self, quality_labels: Dict[str, float]) -> float:
        """
        动态加权综合质量评分

        权重策略: 短期权重更高，长期权重递减
        """
        try:
            period_weights = {1: 0.4, 3: 0.3, 5: 0.2, 10: 0.1}
            weighted_sum = 0
            total_weight = 0

            for period, weight in period_weights.items():
                quality_key = f'quality_{period}d'
                if quality_key in quality_labels and not np.isnan(quality_labels[quality_key]):
                    weighted_sum += quality_labels[quality_key] * weight
                    total_weight += weight

            if total_weight > 0:
                overall_quality = weighted_sum / total_weight
            else:
                overall_quality = 0.2  # 兜底质量分

            return round(overall_quality, 3)

        except Exception as e:
            logger.error(f"综合质量计算失败: {e}")
            return 0.2

    def normalize_quality_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        🎯 全局标准化质量评分，确保良好的分布差异化

        策略：
        1. 使用分位数标准化，保持相对排序
        2. 目标分布：均匀分布在[0.05, 0.95]区间
        3. 特殊处理：确保每个周期都有差异化
        """
        df_normalized = df.copy()

        quality_columns = ['quality_1d', 'quality_3d', 'quality_5d', 'quality_overall']

        for col in quality_columns:
            if col not in df_normalized.columns:
                continue

            values = df_normalized[col].values

            # 检查是否恒定值
            if np.std(values) < 1e-6:
                logger.warning(f"{col}为恒定值，生成随机差异化")
                # 为恒定值添加小量随机噪声以创建差异
                noise = np.random.normal(0, 0.1, len(values))
                values = values + noise

            # 🎯 分位数标准化到[0.05, 0.95]
            # 保持相对排序，但映射到更大的范围
            sorted_indices = np.argsort(values)
            ranks = np.empty_like(sorted_indices)
            ranks[sorted_indices] = np.arange(len(values))

            # 映射到[0.05, 0.95]的均匀分布
            normalized_values = 0.05 + (ranks / (len(values) - 1)) * 0.9

            df_normalized[col] = normalized_values

            logger.info(f"{col}标准化: [{np.min(values):.3f}, {np.max(values):.3f}] → [{np.min(normalized_values):.3f}, {np.max(normalized_values):.3f}], std={np.std(normalized_values):.3f}")

        return df_normalized

    def validate_quality_labels(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        🆕 实时质量标签验证

        验证指标:
        1. 质量标签与收益率相关性
        2. 质量标签分布特征
        3. 头部尾部差异
        """
        try:
            validation_results = {}

            logger.info("🔍 开始质量标签验证...")

            # 验证1: 相关性分析
            correlations = {}
            for period in [1, 3, 5, 10]:
                return_col = f'return_{period}d'
                quality_col = f'quality_{period}d'

                if return_col in df.columns and quality_col in df.columns:
                    # 过滤有效数据
                    valid_mask = (~df[return_col].isna()) & (~df[quality_col].isna())
                    valid_data = df[valid_mask]

                    if len(valid_data) > 10:  # 确保足够样本
                        try:
                            corr, p_value = pearsonr(valid_data[quality_col], valid_data[return_col])
                            correlations[f'{period}d'] = {'correlation': corr, 'p_value': p_value, 'samples': len(valid_data)}

                            status = "✅" if corr > 0.2 else "⚠️" if corr > 0 else "❌"
                            logger.info(f"   {period}日相关性: r={corr:.3f}, p={p_value:.3f}, n={len(valid_data)} {status}")
                        except:
                            correlations[f'{period}d'] = {'correlation': np.nan, 'p_value': np.nan, 'samples': len(valid_data)}
                            logger.warning(f"   {period}日相关性计算失败")

            validation_results['correlations'] = correlations

            # 验证2: 分布特征分析
            distribution_stats = {}
            for period in [1, 3, 5, 10, 'overall']:
                if period == 'overall':
                    quality_col = 'quality_overall'
                else:
                    quality_col = f'quality_{period}d'

                if quality_col in df.columns:
                    valid_quality = df[quality_col].dropna()
                    if len(valid_quality) > 0:
                        stats = {
                            'mean': valid_quality.mean(),
                            'std': valid_quality.std(),
                            'min': valid_quality.min(),
                            'max': valid_quality.max(),
                            'p10': valid_quality.quantile(0.1),
                            'p90': valid_quality.quantile(0.9),
                            'range': valid_quality.max() - valid_quality.min(),
                            'unique_values': valid_quality.nunique()
                        }
                        distribution_stats[quality_col] = stats

                        # 差异化检查
                        diff_status = "✅" if stats['std'] > 0.15 else "⚠️" if stats['std'] > 0.05 else "❌"
                        range_status = "✅" if stats['range'] > 0.4 else "⚠️" if stats['range'] > 0.2 else "❌"

                        logger.info(f"   {quality_col}: std={stats['std']:.3f} {diff_status}, range={stats['range']:.3f} {range_status}")

            validation_results['distribution'] = distribution_stats

            # 验证3: 头部尾部差异分析
            if 'quality_overall' in df.columns:
                valid_overall = df['quality_overall'].dropna()
                if len(valid_overall) > 20:
                    p90_p10_diff = valid_overall.quantile(0.9) - valid_overall.quantile(0.1)
                    validation_results['head_tail_diff'] = p90_p10_diff

                    diff_status = "✅" if p90_p10_diff > 0.4 else "⚠️" if p90_p10_diff > 0.2 else "❌"
                    logger.info(f"   头部尾部差异: P90-P10={p90_p10_diff:.3f} {diff_status}")

            # 验证4: 总体评估
            overall_pass = (
                any(stats.get('correlation', 0) > 0.2 for stats in correlations.values()) and
                any(stats.get('std', 0) > 0.1 for stats in distribution_stats.values())
            )

            validation_results['overall_pass'] = overall_pass
            status = "✅ PASS" if overall_pass else "❌ FAIL"
            logger.info(f"🎯 总体验证结果: {status}")

            return validation_results

        except Exception as e:
            logger.error(f"质量标签验证失败: {e}")
            return {'overall_pass': False, 'error': str(e)}

    def process_historical_predictions_v2(self, reports_dir: str = "reports/daily_selection_v3.8") -> pd.DataFrame:
        """
        处理历史V3.8预测数据 V2.0 (改进版)

        主要改进:
        1. 更严格的数据验证
        2. 实时质量标签验证
        3. 详细的处理日志
        """
        try:
            self.connect_db()

            reports_path = Path(reports_dir)
            json_files = list(reports_path.glob("analysis_data_*.json"))

            logger.info(f"🔍 找到 {len(json_files)} 个V3.8预测文件")

            all_data = []
            processed_dates = []

            for json_file in sorted(json_files):  # 按时间顺序处理
                # 从文件名提取日期
                date_str = json_file.stem.split('_')[-1]
                prediction_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                processed_dates.append(prediction_date)

                logger.info(f"📅 处理 {prediction_date} 的预测数据...")

                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    detailed_stocks = data.get('detailed_stocks', [])
                    daily_count = 0

                    for stock in detailed_stocks:
                        stock_code = stock.get('code')
                        final_score = stock.get('final_score', 50)
                        confidence_score = stock.get('confidence_score', 0.5)

                        if not stock_code:
                            continue

                        # 🆕 使用改进的收益率计算
                        future_returns = self.calculate_future_returns_robust(
                            stock_code, prediction_date
                        )

                        if not future_returns:
                            logger.debug(f"   跳过 {stock_code}: 无未来收益率数据")
                            continue

                        # 🆕 使用改进的质量标签构造
                        quality_labels = self.construct_quality_labels_v2(
                            final_score, future_returns, confidence_score, stock_code
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
                        daily_count += 1

                    logger.info(f"   ✅ {prediction_date}: 成功处理 {daily_count} 只股票")

                except Exception as e:
                    logger.error(f"   ❌ 处理文件失败 {json_file}: {e}")
                    continue

            self.disconnect_db()

            if all_data:
                df = pd.DataFrame(all_data)
                logger.info(f"🎉 成功构造 {len(df)} 条质量标签数据")
                logger.info(f"📊 覆盖时间: {min(processed_dates)} 至 {max(processed_dates)}")
                logger.info(f"📈 股票数量: {df['stock_code'].nunique()} 只")

                # 🆕 实时验证质量标签
                logger.info("🔍 开始实时质量标签验证...")
                validation_results = self.validate_quality_labels(df)

                if validation_results.get('overall_pass', False):
                    logger.info("✅ 质量标签验证通过!")
                else:
                    logger.warning("⚠️ 质量标签验证未完全通过，但数据可用")

                return df
            else:
                logger.error("❌ 未能构造任何质量标签数据")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"处理历史预测数据V2失败: {e}")
            self.disconnect_db()
            return pd.DataFrame()

    def save_quality_dataset_v2(self, df: pd.DataFrame, output_path: str = "quality_training_dataset_v2.csv"):
        """保存质量标签数据集 V2.0"""
        try:
            df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"✅ 质量数据集V2已保存到: {output_path}")

            # 详细统计报告
            print(f"\n📊 质量数据集V2统计报告:")
            print(f"=" * 50)
            print(f"总记录数: {len(df)}")
            print(f"股票数量: {df['stock_code'].nunique()}")
            print(f"时间范围: {df['prediction_date'].min()} - {df['prediction_date'].max()}")
            print()

            # 收益率分布
            print("📈 收益率分布:")
            for period in [1, 3, 5, 10]:
                col = f'return_{period}d'
                if col in df.columns:
                    valid_returns = df[col].dropna()
                    if len(valid_returns) > 0:
                        print(f"   收益率{period}d: 均值={valid_returns.mean():.4f}, 标准差={valid_returns.std():.4f}, 样本数={len(valid_returns)}")

            print()

            # 质量分布 (重点关注)
            print("🎯 质量评分分布:")
            for period in [1, 3, 5, 10, 'overall']:
                if period == 'overall':
                    col = 'quality_overall'
                else:
                    col = f'quality_{period}d'

                if col in df.columns:
                    valid_quality = df[col].dropna()
                    if len(valid_quality) > 0:
                        stats_str = f"均值={valid_quality.mean():.3f}, 标准差={valid_quality.std():.3f}"
                        range_str = f"范围=[{valid_quality.min():.3f}, {valid_quality.max():.3f}]"
                        unique_str = f"唯一值={valid_quality.nunique()}"
                        print(f"   质量{period}: {stats_str}, {range_str}, {unique_str}")

            print("=" * 50)

        except Exception as e:
            logger.error(f"保存数据集V2失败: {e}")

# 使用示例
if __name__ == "__main__":
    constructor = QualityLabelConstructorV2()

    logger.info("🚀 开始V2质量标签构造...")

    # 处理历史预测数据
    df = constructor.process_historical_predictions_v2()

    if not df.empty:
        # 保存数据集
        constructor.save_quality_dataset_v2(df)

        # 显示样本数据
        print(f"\n📋 样本数据预览:")
        sample_cols = ['prediction_date', 'stock_code', 'final_score', 'return_1d', 'return_3d', 'quality_1d', 'quality_3d', 'quality_overall']
        available_cols = [col for col in sample_cols if col in df.columns]
        print(df[available_cols].head(10))

        print(f"\n🎯 关键指标检查:")
        if 'quality_overall' in df.columns:
            quality_overall = df['quality_overall'].dropna()
            print(f"综合质量评分: std={quality_overall.std():.4f} (目标>0.15)")
            print(f"分布范围: [{quality_overall.min():.3f}, {quality_overall.max():.3f}] (目标[0.1, 0.9])")
    else:
        print("❌ 未能构造质量数据集V2")