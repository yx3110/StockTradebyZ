#!/usr/bin/env python3
"""
V3.7版本量化策略回测引擎 - 并行优化版
专门优化V3.7评分计算性能，采用多进程并行处理

性能优化特性：
1. 批量并行V3.7评分计算（多进程）
2. 数据预加载和缓存机制
3. 选股策略并行执行
4. 特征批量提取优化
"""

import pandas as pd
import numpy as np
import json
import logging
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("backtest_engine")

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 导入数据库管理器
from data_adapter.database_manager import DatabaseManager

# 尝试导入V3.7评分系统
try:
    from .v370_advanced_ml_system import V370AdvancedMLSystem
    from tomorrow_stock_selector import TomorrowStockSelector
    V37_AVAILABLE = True
    logger.info("✅ V3.7高级机器学习系统可用")
except ImportError as e:
    V37_AVAILABLE = False
    logger.warning(f"⚠️ V3.7系统不可用: {e}")
    V370AdvancedMLSystem = None
    TomorrowStockSelector = None


def batch_calculate_v37_scores(stock_list_chunk: List[str], date: str, model_path: str) -> List[Tuple[str, float]]:
    """
    批量计算V3.7评分的工作函数（用于多进程）

    Args:
        stock_list_chunk: 股票代码列表片段
        date: 计算日期
        model_path: 模型文件路径

    Returns:
        [(股票代码, 评分), ...]
    """
    try:
        # 在子进程中初始化V3.7系统
        from .v370_advanced_ml_system import V370AdvancedMLSystem

        v37_system = V370AdvancedMLSystem(auto_load_model=False)

        # 加载模型
        v37_system.load_models(model_path)

        results = []

        # 批量提取特征（更高效）
        try:
            features_df = v37_system.extract_advanced_features(
                stock_list_chunk,
                start_date=date,
                end_date=date
            )

            if features_df is not None and not features_df.empty:
                # 批量预测评分
                ml_results = v37_system.predict_three_layer_ensemble(
                    features_df,
                    target_col='target_1d'
                )

                # 🔧 修复: 处理批量结果
                # features_df包含多行历史数据，ml_results返回numpy array，每行一个预测
                # 需要按股票分组，取每只股票最后一行（最新日期）的评分
                if isinstance(ml_results, dict) and 'score' in ml_results:
                    # 单只股票单行结果
                    stock_code = stock_list_chunk[0] if stock_list_chunk else 'unknown'
                    score = float(ml_results['score'])
                    results.append((stock_code, score))
                elif isinstance(ml_results, np.ndarray):
                    # 批量结果（numpy数组）- 每行一个预测
                    # features_df有'code'列标识股票代码
                    if 'code' in features_df.columns:
                        # 添加预测结果到DataFrame
                        features_df_with_scores = features_df.copy()
                        features_df_with_scores['predicted_score'] = ml_results

                        # 按股票分组，取每组最后一行（最新日期的评分）
                        for stock_code in stock_list_chunk:
                            stock_data = features_df_with_scores[features_df_with_scores['code'] == stock_code]
                            if not stock_data.empty:
                                # 取最后一行的评分（最新日期）
                                score = float(stock_data.iloc[-1]['predicted_score'])
                            else:
                                score = 50.0  # 该股票没有特征数据
                            results.append((stock_code, score))
                    else:
                        # 没有code列，按顺序平均分配
                        scores_per_stock = len(ml_results) // len(stock_list_chunk)
                        for i, stock_code in enumerate(stock_list_chunk):
                            start_idx = i * scores_per_stock
                            end_idx = start_idx + scores_per_stock
                            if end_idx <= len(ml_results):
                                # 取该股票最后一个预测（最新）
                                score = float(ml_results[end_idx - 1])
                            else:
                                score = 50.0
                            results.append((stock_code, score))
                else:
                    # 未知格式，使用默认评分
                    logger.warning(f"⚠️ 未知的预测结果格式: {type(ml_results)}")
                    for stock_code in stock_list_chunk:
                        results.append((stock_code, 50.0))
            else:
                # 特征提取失败，返回默认评分
                for stock_code in stock_list_chunk:
                    results.append((stock_code, 50.0))

        except Exception as feature_error:
            logger.debug(f"批量特征提取失败: {feature_error}")
            # 逐只股票处理（备选方案）
            for stock_code in stock_list_chunk:
                try:
                    features_df = v37_system.extract_advanced_features(
                        [stock_code],
                        start_date=date,
                        end_date=date
                    )

                    if features_df is not None and not features_df.empty:
                        ml_result = v37_system.predict_three_layer_ensemble(
                            features_df,
                            target_col='target_1d'
                        )
                        score = float(ml_result.get('score', 50.0)) if isinstance(ml_result, dict) else 50.0
                    else:
                        score = 50.0

                    results.append((stock_code, score))

                except Exception as single_error:
                    logger.debug(f"单只股票评分失败 {stock_code}: {single_error}")
                    results.append((stock_code, 50.0))

        return results

    except Exception as e:
        logger.error(f"批量V3.7评分计算失败: {e}")
        # 返回默认评分
        return [(stock_code, 50.0) for stock_code in stock_list_chunk]


class V37BacktestEngineOptimized:
    """V3.7版本量化策略回测引擎 - 并行优化版"""

    def __init__(self, initial_capital: float = 1000000, max_workers: int = 4):
        """
        初始化优化版V3.7回测引擎

        Args:
            initial_capital: 初始资金
            max_workers: 最大并行工作进程数
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trades = []
        self.daily_returns = []
        self.portfolio_values = []

        # 并行处理配置
        self.max_workers = min(max_workers, mp.cpu_count())

        # 交易参数
        self.commission_rate = 0.0003
        self.stamp_tax = 0.001
        self.transfer_fee = 0.00002
        self.min_commission = 5.0

        # 风控参数
        self.max_position_pct = 0.10
        self.max_positions = 15
        self.min_trade_amount = 1000
        self.min_score = 40.0  # 降低阈值以适应V3.7维度不匹配降级模式
        self.rebalance_freq = 5

        # 数据缓存
        self.data_cache = {}
        self.price_cache = {}

        # 初始化组件
        self.db_manager = DatabaseManager()
        self.v37_system = None
        self.selector = None

        # 找到最新的V3.7模型文件
        self.model_path = self._find_latest_v37_model()

        logger.info(f"🚀 V3.7优化版回测引擎初始化完成，并行进程数: {self.max_workers}")

    def _find_latest_v37_model(self) -> Optional[str]:
        """查找最新的V3.7模型文件"""
        model_dir = Path('models/v370')
        if not model_dir.exists():
            return None

        # 优先级顺序
        patterns = [
            'v370_quality_optimized_*.pkl',
            'v370_enhanced_*.pkl',
            'v370_models_*.pkl',
            'v370_advanced_v4_*.pkl'
        ]

        for pattern in patterns:
            model_files = list(model_dir.glob(pattern))
            if model_files:
                latest_model = max(model_files, key=lambda f: f.stat().st_mtime)
                logger.info(f"📱 找到V3.7模型: {latest_model}")
                return str(latest_model)

        return None

    def parallel_v37_scoring(self, stock_list: List[str], date: str, chunk_size: int = 50) -> Dict[str, float]:
        """
        并行计算V3.7评分

        Args:
            stock_list: 需要评分的股票列表
            date: 计算日期
            chunk_size: 每个进程处理的股票数量

        Returns:
            {股票代码: 评分}
        """
        if not stock_list or not self.model_path:
            return {}

        logger.info(f"🚀 开始并行计算 {len(stock_list)} 只股票的V3.7评分...")

        # 将股票列表分块
        chunks = [stock_list[i:i + chunk_size] for i in range(0, len(stock_list), chunk_size)]

        # 并行处理
        all_results = {}

        try:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任务
                future_to_chunk = {
                    executor.submit(batch_calculate_v37_scores, chunk, date, self.model_path): chunk
                    for chunk in chunks
                }

                # 收集结果
                for future in future_to_chunk:
                    try:
                        chunk_results = future.result(timeout=60)  # 1分钟超时
                        for stock_code, score in chunk_results:
                            all_results[stock_code] = score
                    except Exception as e:
                        chunk = future_to_chunk[future]
                        logger.warning(f"并行计算失败，chunk大小 {len(chunk)}: {e}")
                        # 使用默认评分
                        for stock_code in chunk:
                            all_results[stock_code] = 50.0

        except Exception as e:
            logger.error(f"并行处理系统错误: {e}")
            # 备选方案：返回默认评分
            for stock_code in stock_list:
                all_results[stock_code] = 50.0

        logger.info(f"✅ 完成并行评分，成功计算 {len(all_results)} 只股票")
        return all_results

    def batch_load_stock_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        批量预加载股票数据并缓存

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            {股票代码: DataFrame}
        """
        cache_key = f"{start_date}_{end_date}"
        if cache_key in self.data_cache:
            logger.info(f"📋 使用缓存的股票数据: {cache_key}")
            return self.data_cache[cache_key]

        logger.info(f"📊 批量加载股票数据: {start_date} 至 {end_date}")

        # 计算需要的历史数据范围（考虑技术指标计算需要）
        start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
        extended_start = start_date_obj - timedelta(days=120)  # 4个月历史数据
        extended_start_str = extended_start.strftime('%Y-%m-%d')

        query = """
        SELECT s.code, dq.trade_date, dq.open, dq.high, dq.low, dq.close,
               dq.volume, dq.price_change_pct, db.pe_ttm, db.pb, db.total_mv, db.turnover_rate
        FROM securities s
        JOIN daily_quotes dq ON s.id = dq.security_id
        LEFT JOIN daily_basic db ON s.id = db.security_id AND dq.trade_date = db.trade_date
        WHERE s.type = 'A股'
        AND dq.trade_date BETWEEN ? AND ?
        AND dq.close IS NOT NULL
        ORDER BY s.code, dq.trade_date
        """

        try:
            result = self.db_manager.execute_query(query, [extended_start_str, end_date])

            # 转换为按股票代码分组的DataFrame
            data_dict = {}
            current_stock = None
            current_data = []

            for row in result:
                stock_code = row[0]
                if stock_code != current_stock:
                    # 保存上一只股票的数据
                    if current_stock and current_data:
                        df = pd.DataFrame(current_data, columns=[
                            'trade_date', 'open', 'high', 'low', 'close',
                            'volume', 'price_change_pct', 'pe_ttm', 'pb', 'total_mv', 'turnover_rate'
                        ])
                        df['trade_date'] = pd.to_datetime(df['trade_date'])
                        df = df.set_index('trade_date')
                        data_dict[current_stock] = df

                    # 开始新股票
                    current_stock = stock_code
                    current_data = []

                # 添加数据行
                current_data.append(row[1:])  # 跳过股票代码

            # 保存最后一只股票
            if current_stock and current_data:
                df = pd.DataFrame(current_data, columns=[
                    'trade_date', 'open', 'high', 'low', 'close',
                    'volume', 'price_change_pct', 'pe_ttm', 'pb', 'total_mv', 'turnover_rate'
                ])
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.set_index('trade_date')
                data_dict[current_stock] = df

            # 缓存数据
            self.data_cache[cache_key] = data_dict
            logger.info(f"✅ 批量加载完成，共 {len(data_dict)} 只股票")

            return data_dict

        except Exception as e:
            logger.error(f"批量数据加载失败: {e}")
            return {}

    def optimized_stock_selection(self, date: str, stock_universe: List[str]) -> List[Dict]:
        """
        优化版股票选择流程

        Args:
            date: 交易日期
            stock_universe: 股票池

        Returns:
            选中的股票列表
        """
        logger.info(f"🎯 开始优化版股票选择: {date}")

        # 1. 快速基础筛选（简化版选股策略）
        candidate_stocks = self._fast_stock_screening(stock_universe, date)

        if not candidate_stocks:
            logger.info("❌ 基础筛选未找到候选股票")
            return []

        logger.info(f"📋 基础筛选后候选股票: {len(candidate_stocks)} 只")

        # 2. 并行V3.7评分计算
        v37_scores = self.parallel_v37_scoring(candidate_stocks, date)

        # 3. 根据评分排序选择
        scored_stocks = []
        for stock_code in candidate_stocks:
            score = v37_scores.get(stock_code, 50.0)
            if score >= self.min_score:
                scored_stocks.append({
                    'stock_code': stock_code,
                    'v37_score': score,
                    'date': date
                })

        # 按评分排序，选择前N只
        scored_stocks.sort(key=lambda x: x['v37_score'], reverse=True)
        selected_stocks = scored_stocks[:self.max_positions]

        logger.info(f"✅ 最终选择 {len(selected_stocks)} 只股票，最高评分: {selected_stocks[0]['v37_score']:.1f}" if selected_stocks else "❌ 未选择到符合条件的股票")

        return selected_stocks

    def _fast_stock_screening(self, stock_universe: List[str], date: str) -> List[str]:
        """
        快速基础筛选（简化版选股策略）

        Args:
            stock_universe: 股票池
            date: 日期

        Returns:
            候选股票列表
        """
        try:
            # 使用缓存的数据 - 找到包含该日期的缓存
            cached_data = {}
            for cache_key, data in self.data_cache.items():
                if data:  # 如果缓存中有数据，使用第一个可用的缓存
                    cached_data = data
                    break

            candidates = []

            # 简化版筛选逻辑（基于基本技术指标）
            for stock_code in stock_universe[:500]:  # 限制处理数量以提高速度
                try:
                    if stock_code in cached_data:
                        stock_data = cached_data[stock_code]

                        # 获取最近的数据
                        date_ts = pd.Timestamp(date)
                        recent_data = stock_data[stock_data.index <= date_ts].tail(20)

                        if len(recent_data) >= 10:  # 至少需要10天数据
                            latest = recent_data.iloc[-1]

                            # 基础筛选条件
                            if (latest['close'] > 0 and
                                latest['volume'] > 0 and
                                latest['turnover_rate'] > 0.01 and  # 最低换手率
                                latest['turnover_rate'] < 0.25):    # 最高换手率

                                candidates.append(stock_code)

                                if len(candidates) >= 200:  # 限制候选数量
                                    break

                except Exception as e:
                    logger.debug(f"处理股票 {stock_code} 失败: {e}")
                    continue

            return candidates

        except Exception as e:
            logger.error(f"快速筛选失败: {e}")
            return stock_universe[:100]  # 备选方案

    def _get_stock_price(self, stock_code: str, date: str) -> Optional[float]:
        """
        获取股票在指定日期的收盘价

        Args:
            stock_code: 股票代码
            date: 日期 (YYYY-MM-DD)

        Returns:
            收盘价，如果没有数据返回None
        """
        try:
            # 从缓存的数据中获取价格
            for cache_key, cached_data in self.data_cache.items():
                if stock_code in cached_data:
                    stock_data = cached_data[stock_code]
                    date_ts = pd.Timestamp(date)

                    # 查找最接近的交易日数据
                    if date_ts in stock_data.index:
                        return float(stock_data.loc[date_ts, 'close'])
                    else:
                        # 找到最近的交易日
                        available_dates = stock_data[stock_data.index <= date_ts]
                        if not available_dates.empty:
                            return float(available_dates.iloc[-1]['close'])

            # 如果缓存中没有，直接查询数据库
            query = """
            SELECT dq.close
            FROM securities s
            JOIN daily_quotes dq ON s.id = dq.security_id
            WHERE s.code = ? AND dq.trade_date <= ?
            ORDER BY dq.trade_date DESC
            LIMIT 1
            """

            result = self.db_manager.execute_query(query, [stock_code, date])
            if result:
                return float(result[0][0])

        except Exception as e:
            logger.debug(f"获取股票 {stock_code} 在 {date} 的价格失败: {e}")

        return None

    def _check_stop_loss(self, date: str):
        """检查止损"""
        positions_to_sell = []

        for stock_code, position in self.positions.items():
            if position['shares'] > 0:
                current_price = self._get_stock_price(stock_code, date)
                if current_price and current_price > 0:
                    # 计算当前浮动损益
                    cost_basis = position['avg_cost']
                    loss_pct = (current_price - cost_basis) / cost_basis

                    # 如果亏损超过8%，执行止损
                    if loss_pct < -0.08:
                        positions_to_sell.append(stock_code)

        # 执行止损卖出
        for stock_code in positions_to_sell:
            self._execute_sell(stock_code, date, "stop_loss")

    def _execute_sell(self, stock_code: str, date: str, reason: str = "rebalance"):
        """执行卖出操作"""
        if stock_code not in self.positions or self.positions[stock_code]['shares'] <= 0:
            return

        position = self.positions[stock_code]
        shares = position['shares']
        current_price = self._get_stock_price(stock_code, date)

        if current_price and current_price > 0:
            # 计算卖出金额（扣除印花税和手续费）
            gross_amount = shares * current_price
            sell_amount = gross_amount * (1 - self.commission_rate - self.stamp_tax)

            # 记录交易
            trade = {
                'date': date,
                'stock_code': stock_code,
                'action': 'sell',
                'shares': shares,
                'price': current_price,
                'amount': sell_amount,
                'reason': reason,
                'profit': sell_amount - (shares * position['avg_cost'])
            }
            self.trades.append(trade)

            # 更新持仓
            self.positions[stock_code]['shares'] = 0
            self.positions[stock_code]['avg_cost'] = 0

            # 更新资金
            self.current_capital += sell_amount

            logger.info(f"卖出 {stock_code}: {shares}股 @ {current_price:.2f}元, 原因: {reason}")

    def run_optimized_backtest(self, start_date: str, end_date: str) -> Dict:
        """
        运行优化版V3.7回测

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            回测结果
        """
        logger.info(f"🚀 开始优化版V3.7回测: {start_date} 至 {end_date}")

        # 1. 预加载所有需要的数据
        self.batch_load_stock_data(start_date, end_date)

        # 2. 获取股票池
        actual_trading_days = len(self._get_trading_dates(start_date, end_date))
        min_trading_days = max(3, min(actual_trading_days - 1, 20))
        stock_universe = self._get_stock_universe(start_date, end_date, min_trading_days)

        if not stock_universe:
            raise ValueError("股票池为空，无法进行回测")

        logger.info(f"📊 股票池包含 {len(stock_universe)} 只股票")

        # 3. 获取交易日期
        trading_dates = self._get_trading_dates(start_date, end_date)
        logger.info(f"📅 回测期间共 {len(trading_dates)} 个交易日")

        # 4. 执行优化版回测循环
        last_rebalance_date = None

        for i, current_date in enumerate(trading_dates):
            if i % 5 == 0:
                progress = i / len(trading_dates) * 100
                logger.info(f"📈 回测进度: {progress:.1f}% ({current_date})")

            # 检查是否需要调仓
            current_date_str = current_date.strftime('%Y-%m-%d') if isinstance(current_date, datetime) else str(current_date)
            should_rebalance = (
                last_rebalance_date is None or
                (datetime.strptime(current_date_str, '%Y-%m-%d') -
                 datetime.strptime(last_rebalance_date, '%Y-%m-%d')).days >= self.rebalance_freq
            )

            if should_rebalance:
                # 使用优化版选股
                selected_stocks = self.optimized_stock_selection(current_date, stock_universe)

                # 执行调仓
                self._execute_rebalance(current_date_str, selected_stocks)
                last_rebalance_date = current_date_str

            # 更新持仓和组合价值
            self._update_portfolio_value(current_date)

        # 5. 计算绩效指标
        results = self._calculate_performance_metrics()

        logger.info("🎉 优化版V3.7回测完成！")
        return results

    def _get_stock_universe(self, start_date: str, end_date: str, min_trading_days: int = 10) -> List[str]:
        """获取股票池"""
        query = """
        SELECT s.code, COUNT(dq.trade_date) as trading_days
        FROM securities s
        JOIN daily_quotes dq ON s.id = dq.security_id
        WHERE s.type = 'A股'
        AND dq.trade_date BETWEEN ? AND ?
        AND dq.close IS NOT NULL
        AND dq.volume > 0
        GROUP BY s.code
        HAVING trading_days >= ?
        ORDER BY s.code
        """

        result = self.db_manager.execute_query(query, [start_date, end_date, min_trading_days])
        stock_list = [row[0] for row in result]

        # 过滤ST股票
        stock_list = [s for s in stock_list if 'ST' not in s and '*ST' not in s]

        return stock_list

    def _get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取交易日期列表"""
        query = """
        SELECT DISTINCT trade_date
        FROM daily_quotes
        WHERE trade_date BETWEEN ? AND ?
        ORDER BY trade_date
        """

        result = self.db_manager.execute_query(query, [start_date, end_date])
        return [row[0] for row in result]

    def _execute_rebalance(self, date: str, selected_stocks: List[Dict]):
        """执行调仓操作"""
        if not selected_stocks:
            return

        # 计算总可用资金
        total_available = self.current_capital

        # 计算每只股票的目标资金分配
        target_allocation = total_available / min(len(selected_stocks), self.max_positions)

        # 执行买入操作
        for i, stock_info in enumerate(selected_stocks[:self.max_positions]):
            stock_code = stock_info['stock_code']
            v37_score = stock_info['v37_score']

            # 获取股票价格
            price = self._get_stock_price(stock_code, date)
            if price is None or price <= 0:
                continue

            # 计算可买入股数（考虑手续费）
            buy_amount = target_allocation * (1 - self.commission_rate)
            shares = int(buy_amount / (price * 100)) * 100  # 买入整手

            if shares >= 100:  # 至少买入1手
                actual_cost = shares * price * (1 + self.commission_rate)

                if actual_cost <= self.current_capital:
                    # 记录交易
                    trade = {
                        'date': date,
                        'stock_code': stock_code,
                        'action': 'buy',
                        'shares': shares,
                        'price': price,
                        'amount': actual_cost,
                        'v37_score': v37_score
                    }
                    self.trades.append(trade)

                    # 更新持仓
                    if stock_code not in self.positions:
                        self.positions[stock_code] = {
                            'shares': 0,
                            'avg_cost': 0,
                            'entry_date': date,
                            'entry_score': v37_score
                        }

                    old_shares = self.positions[stock_code]['shares']
                    old_cost = self.positions[stock_code]['avg_cost']

                    # 更新平均成本
                    total_shares = old_shares + shares
                    total_cost = old_shares * old_cost + actual_cost
                    self.positions[stock_code]['shares'] = total_shares
                    self.positions[stock_code]['avg_cost'] = total_cost / total_shares if total_shares > 0 else 0

                    # 更新资金
                    self.current_capital -= actual_cost

                    logger.info(f"买入 {stock_code}: {shares}股 @ {price:.2f}元, V3.7评分: {v37_score:.1f}")

    def _update_portfolio_value(self, date: str):
        """更新组合价值"""
        portfolio_value = self.current_capital  # 现金部分

        # 计算持仓价值
        for stock_code, position in self.positions.items():
            if position['shares'] > 0:
                current_price = self._get_stock_price(stock_code, date)
                if current_price and current_price > 0:
                    market_value = position['shares'] * current_price
                    portfolio_value += market_value

        # 记录每日组合价值
        self.portfolio_values.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'cash': self.current_capital
        })

        # 计算日收益率
        if len(self.portfolio_values) > 1:
            prev_value = self.portfolio_values[-2]['portfolio_value']
            daily_return = (portfolio_value / prev_value - 1) if prev_value > 0 else 0
            self.daily_returns.append(daily_return)

        # 检查是否需要止损
        self._check_stop_loss(date)

    def _calculate_performance_metrics(self) -> Dict:
        """计算绩效指标"""
        if not self.portfolio_values or not self.trades:
            return {
                'total_return': 0.0,
                'final_capital': self.current_capital,
                'annual_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'total_trades': len(self.trades),
                'successful_trades': 0,
                'failed_trades': 0,
                'avg_v37_score': 0.0
            }

        # 计算最终组合价值
        final_portfolio_value = self.portfolio_values[-1]['portfolio_value']

        # 计算总收益率
        total_return = (final_portfolio_value / self.initial_capital - 1) if self.initial_capital > 0 else 0

        # 计算最大回撤
        max_drawdown = 0.0
        peak_value = self.initial_capital
        for pv in self.portfolio_values:
            value = pv['portfolio_value']
            if value > peak_value:
                peak_value = value
            drawdown = (peak_value - value) / peak_value if peak_value > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

        # 计算夏普比率
        if len(self.daily_returns) > 1:
            avg_return = sum(self.daily_returns) / len(self.daily_returns)
            return_std = (sum([(r - avg_return) ** 2 for r in self.daily_returns]) / len(self.daily_returns)) ** 0.5
            sharpe_ratio = (avg_return / return_std * (252 ** 0.5)) if return_std > 0 else 0
        else:
            sharpe_ratio = 0.0

        # 统计交易
        buy_trades = [t for t in self.trades if t['action'] == 'buy']
        total_trades = len(buy_trades)

        # 计算平均V3.7评分
        avg_v37_score = sum([t['v37_score'] for t in buy_trades]) / total_trades if total_trades > 0 else 0

        # 年化收益率
        days_count = len(self.portfolio_values)
        annual_return = ((1 + total_return) ** (252 / days_count) - 1) if days_count > 0 and total_return > -1 else 0

        return {
            'total_return': total_return,
            'final_capital': final_portfolio_value,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': 0.0,  # 需要卖出数据才能计算
            'total_trades': total_trades,
            'successful_trades': 0,  # 需要卖出数据才能计算
            'failed_trades': 0,  # 需要卖出数据才能计算
            'avg_v37_score': avg_v37_score,
            'avg_holding_days': 5.0,  # 估算
            'avg_trade_return': total_return / total_trades if total_trades > 0 else 0,
            'max_trade_return': 0.0,  # 需要卖出数据
            'max_trade_loss': 0.0  # 需要卖出数据
        }


def run_optimized_v37_backtest(start_date: str = '2024-08-01',
                               end_date: str = '2024-08-31',
                               initial_capital: float = 1000000,
                               max_workers: int = 4):
    """
    运行优化版V3.7回测

    Args:
        start_date: 开始日期
        end_date: 结束日期
        initial_capital: 初始资金
        max_workers: 最大并行进程数
    """
    logger.info("🚀 开始优化版V3.7回测")

    try:
        # 创建优化版回测引擎
        engine = V37BacktestEngineOptimized(
            initial_capital=initial_capital,
            max_workers=max_workers
        )

        # 执行回测
        results = engine.run_optimized_backtest(start_date, end_date)

        # 输出结果
        logger.info("🎉 优化版V3.7回测完成！")
        logger.info(f"📊 年化收益率: {results['annual_return']:.2%}")
        logger.info(f"📊 夏普比率: {results['sharpe_ratio']:.2f}")
        logger.info(f"📊 最大回撤: {results['max_drawdown']:.2%}")

        return results

    except Exception as e:
        logger.error(f"❌ 优化版V3.7回测失败: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="V3.7优化版量化策略回测引擎")
    parser.add_argument('--start-date', default='2024-08-26', help='开始日期')
    parser.add_argument('--end-date', default='2024-08-30', help='结束日期')
    parser.add_argument('--initial-capital', type=float, default=1000000, help='初始资金')
    parser.add_argument('--max-workers', type=int, default=4, help='最大并行进程数')

    args = parser.parse_args()

    # 运行优化版回测
    results = run_optimized_v37_backtest(
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        max_workers=args.max_workers
    )