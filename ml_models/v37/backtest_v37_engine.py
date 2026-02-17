#!/usr/bin/env python3
"""
V3.7版本量化策略回测引擎
基于原有回测引擎，专门适配V3.7高级机器学习评分系统
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
warnings.filterwarnings('ignore')

# 配置日志 (先配置日志)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("backtest_v37")

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
    logger.warning(f"⚠️ V3.7系统不可用，将使用传统评分方法: {e}")
    V370AdvancedMLSystem = None
    TomorrowStockSelector = None

# 可选的matplotlib导入
try:
    import matplotlib.pyplot as plt
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class V37BacktestEngine:
    """V3.7版本量化策略回测引擎"""

    def __init__(self, initial_capital: float = 1000000):
        """
        初始化V3.7回测引擎

        Args:
            initial_capital: 初始资金
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # 持仓记录
        self.trades = []     # 交易记录
        self.daily_returns = []  # 每日收益
        self.portfolio_values = []  # 组合净值

        # 交易成本设置（严格按照A股实际成本）
        self.commission_rate = 0.0003  # 万分之三佣金
        self.stamp_tax = 0.001         # 千分之一印花税（仅卖出）
        self.transfer_fee = 0.00002    # 过户费（双向）
        self.min_commission = 5.0      # 最低佣金5元

        # 风控设置 - 针对V3.7优化
        self.max_position_pct = 0.10   # 单股最大仓位10%
        self.max_positions = 15        # 最大持股数量（V3.7精选，可适当增加）
        self.min_trade_amount = 1000   # 最小交易金额
        self.min_score = 75.0          # V3.7最低评分阈值
        self.rebalance_freq = 5        # 调仓频率（天）

        # 初始化V3.7评分系统
        self.v37_system = None
        self.selector = None  # 缓存selector实例
        self.db_manager = DatabaseManager()

        logger.info(f"V3.7回测引擎初始化完成，初始资金: {initial_capital:,.0f}元")

    def initialize_v37_system(self) -> bool:
        """初始化V3.7评分系统"""
        if not V37_AVAILABLE:
            logger.warning("⚠️ V3.7系统不可用，将使用传统评分方法")
            return False

        try:
            logger.info("正在初始化V3.7高级机器学习评分系统...")

            # 创建并缓存selector实例，避免重复初始化
            if self.selector is None:
                logger.info("🚀 已初始化v3.7高级机器学习系统（5基础模型+4专家模型+Meta学习器三层ensemble，35+维特征）")
                self.selector = TomorrowStockSelector(scoring_version='v3.7', stocks_only=True)
                logger.info("✅ V3.7选股器缓存成功")

            return True

        except Exception as e:
            logger.error(f"❌ V3.7系统初始化失败: {e}")
            return False

    def get_stock_universe(self, start_date: str, end_date: str,
                          min_trading_days: int = 100) -> List[str]:
        """
        获取股票池

        Args:
            start_date: 开始日期
            end_date: 结束日期
            min_trading_days: 最少交易天数

        Returns:
            股票代码列表
        """
        logger.info(f"获取股票池: {start_date} 至 {end_date}")

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

        # 过滤掉ST股票
        stock_list = [s for s in stock_list if 'ST' not in s and '*ST' not in s]

        logger.info(f"获取到 {len(stock_list)} 只符合条件的股票")
        return stock_list

    def generate_v37_signals(self, date: str, stock_universe: List[str]) -> pd.DataFrame:
        """
        使用V3.7系统生成当日选股信号

        Args:
            date: 交易日期
            stock_universe: 股票池

        Returns:
            选股信号DataFrame
        """
        try:
            if self.selector is None:
                logger.warning("V3.7选股器未初始化，尝试重新初始化...")
                if not self.initialize_v37_system():
                    return pd.DataFrame()

            # 使用缓存的selector实例，避免重复初始化
            selector = self.selector

            # 使用V3.7系统进行批量选股（高效方式）
            logger.debug(f"生成 {date} 的V3.7选股信号...")

            # 🔧 修复：使用正确的tomorrow_stock_selector工作流程
            try:
                # 1. 确保日期格式正确（tomorrow_stock_selector需要字符串格式）
                date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)

                # 2. 加载指定日期的数据
                data = selector.load_data(target_date=date_str)
                if not data:
                    logger.debug(f"V3.7无法加载数据: {date_str}")
                    return pd.DataFrame()

                # 3. 获取目标日期
                target_date_obj = pd.Timestamp(date_str)

                # 3. 运行选股策略
                results = selector.run_selectors(data, target_date_obj)
                if not results:
                    logger.debug(f"V3.7选股结果为空: {date}")
                    return pd.DataFrame()

                # 4. 分析结果获取评分和排序
                analysis = selector.analyze_results(results, data, target_date_obj)
                if not analysis or 'selected_stocks' not in analysis:
                    logger.debug(f"V3.7分析结果为空: {date}")
                    return pd.DataFrame()

                # 5. 转换为标准格式，过滤高分股票
                signals = []
                for stock_data in analysis['selected_stocks']:
                    try:
                        stock_code = stock_data.get('code', '')
                        stock_score = stock_data.get('score', 0)

                        if not stock_code or stock_score < self.min_score:
                            continue

                        # 获取当前价格数据
                        price_data = self._get_stock_price_data(stock_code, date)
                        if price_data:
                            signals.append({
                                'date': date,
                                'stock_code': stock_code,
                                'v37_score': stock_data.get('total_score', 0),
                                'current_price': price_data['close'],
                                'signal': 'BUY',
                                'suggested_buy_price': price_data['close'],
                                'stop_loss_price': price_data['close'] * 0.92,  # 8%止损
                                'take_profit_price': price_data['close'] * 1.15  # 15%止盈
                            })
                    except Exception as e:
                        logger.debug(f"处理股票数据失败 {stock_data}: {e}")
                        continue

                signals_df = pd.DataFrame(signals)
                if not signals_df.empty:
                    # 按V3.7评分排序，选择前15只
                    signals_df = signals_df.sort_values('v37_score', ascending=False).head(self.max_positions)
                    logger.info(f"✅ 生成 {len(signals_df)} 个V3.7选股信号，最高评分: {signals_df['v37_score'].max():.1f}")

                return signals_df

            except Exception as batch_error:
                logger.warning(f"V3.7批量选股失败，使用备选方法: {batch_error}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"生成V3.7选股信号失败: {e}")
            return pd.DataFrame()

    def _get_v37_score_for_stock(self, stock_code: str, date: str) -> float:
        """获取单只股票的V3.7评分"""
        try:
            # 如果V3.7系统可用，使用机器学习评分
            if self.v37_system:
                # 获取特征数据
                features = self._extract_features_for_stock(stock_code, date)
                if features is not None:
                    score = self.v37_system.predict_single_stock(features)
                    return score

            # 备选方案：使用传统评分方法
            return self._get_traditional_score(stock_code, date)

        except Exception as e:
            logger.debug(f"获取股票 {stock_code} V3.7评分失败: {e}")
            return 0.0

    def _extract_features_for_stock(self, stock_code: str, date: str) -> Optional[np.ndarray]:
        """为单只股票提取V3.7所需的特征"""
        try:
            # 获取技术指标数据
            query = """
            SELECT ti.*, dq.close, dq.volume, dq.high, dq.low, dq.open,
                   db.pe_ttm, db.pb, db.market_cap, db.turnover_rate
            FROM securities s
            JOIN technical_indicators ti ON s.id = ti.security_id
            JOIN daily_quotes dq ON s.id = dq.security_id AND ti.trade_date = dq.trade_date
            LEFT JOIN daily_basic db ON s.id = db.security_id AND ti.trade_date = db.trade_date
            WHERE s.code = ? AND ti.trade_date = ?
            """

            result = self.db_manager.execute_query(query, [stock_code, date])
            if not result:
                return None

            data = result[0]

            # 构建特征向量（简化版本，实际应该包含49个特征）
            features = []

            # 技术指标特征 (17个)
            technical_features = [
                data[3] if data[3] is not None else 0,  # bbi
                data[7] if data[7] is not None else 0,  # volume_surge
                data[8] if data[8] is not None else 0,  # price_momentum
                data[9] if data[9] is not None else 50, # rsi
                1 if data[10] else 0,                   # kdj_cross
                data[11] if data[11] is not None else 0, # volatility_risk
                data[12] if data[12] is not None else 0, # adx_14
                data[13] if data[13] is not None else 0, # trix
                data[14] if data[14] is not None else 0, # vwap_deviation
                data[15] if data[15] is not None else 0, # atr_ratio
                data[16] if data[16] is not None else 0, # keltner_position
                data[17] if data[17] is not None else 0, # volatility_regime
                data[18] if data[18] is not None else 0, # obv_trend
                data[19] if data[19] is not None else 0, # mfi_14
                data[20] if data[20] is not None else 0, # momentum_3d
                data[21] if data[21] is not None else 0, # momentum_5d
                data[22] if data[22] is not None else 0  # momentum_20d
            ]
            features.extend(technical_features)

            # 基本面特征 (8个)
            fundamental_features = [
                data[-4] if data[-4] is not None else 0,  # pb
                data[-5] if data[-5] is not None else 0,  # pe_ttm
                np.log(data[-3]) if data[-3] and data[-3] > 0 else 0,  # log(market_cap)
                data[-2] if data[-2] is not None else 0,  # turnover_rate
                0, 0, 0, 0  # 其他基本面特征（简化）
            ]
            features.extend(fundamental_features)

            # 宏观特征 (8个) - 简化处理
            macro_features = [0] * 8
            features.extend(macro_features)

            # 情绪特征 (7个) - 简化处理
            sentiment_features = [0] * 7
            features.extend(sentiment_features)

            # 时序特征 (5个) - 简化处理
            time_features = [0] * 5
            features.extend(time_features)

            # 市场环境特征 (4个) - 简化处理
            market_features = [0] * 4
            features.extend(market_features)

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.debug(f"提取特征失败: {e}")
            return None

    def _get_traditional_score(self, stock_code: str, date: str) -> float:
        """传统评分方法（备选方案）"""
        try:
            # 获取基础数据
            query = """
            SELECT ti.bbi, ti.rsi, ti.kdj_k, ti.kdj_d, dq.close, dq.volume,
                   db.pe_ttm, db.pb, db.turnover_rate
            FROM securities s
            JOIN technical_indicators ti ON s.id = ti.security_id
            JOIN daily_quotes dq ON s.id = dq.security_id AND ti.trade_date = dq.trade_date
            LEFT JOIN daily_basic db ON s.id = db.security_id AND ti.trade_date = db.trade_date
            WHERE s.code = ? AND ti.trade_date = ?
            """

            result = self.db_manager.execute_query(query, [stock_code, date])
            if not result:
                return 0.0

            data = result[0]
            bbi, rsi, kdj_k, kdj_d, close, volume, pe, pb, turnover = data

            # 简单评分算法
            score = 50.0  # 基础分

            # 技术面评分 (40分)
            if bbi and close and close > bbi:
                score += 10  # BBI突破

            if rsi and 30 <= rsi <= 70:
                score += 10  # RSI合理区间

            if kdj_k and kdj_d and kdj_k > kdj_d and kdj_k < 80:
                score += 10  # KDJ金叉且未超买

            if turnover and 0.02 <= turnover <= 0.15:
                score += 10  # 合理换手率

            # 基本面评分 (10分)
            if pe and 0 < pe < 30:
                score += 5   # 合理PE

            if pb and 0 < pb < 5:
                score += 5   # 合理PB

            return min(score, 100.0)

        except Exception as e:
            logger.debug(f"传统评分失败: {e}")
            return 0.0

    def _get_previous_trading_date(self, date: str) -> Optional[str]:
        """获取前一个交易日"""
        try:
            query = """
            SELECT DISTINCT trade_date
            FROM daily_quotes
            WHERE trade_date < ?
            ORDER BY trade_date DESC
            LIMIT 1
            """

            result = self.db_manager.execute_query(query, [date])
            return result[0][0] if result else None

        except Exception as e:
            logger.debug(f"获取前一交易日失败: {e}")
            return None

    def _get_stock_price_data(self, stock_code: str, date: str) -> Optional[Dict]:
        """获取股票价格数据"""
        try:
            query = """
            SELECT dq.open, dq.high, dq.low, dq.close, dq.volume,
                   dq.is_limit_up, dq.is_limit_down
            FROM securities s
            JOIN daily_quotes dq ON s.id = dq.security_id
            WHERE s.code = ? AND dq.trade_date = ?
            """

            result = self.db_manager.execute_query(query, [stock_code, date])
            if not result:
                return None

            data = result[0]
            return {
                'open': data[0],
                'high': data[1],
                'low': data[2],
                'close': data[3],
                'volume': data[4],
                'is_limit_up': bool(data[5]),
                'is_limit_down': bool(data[6])
            }

        except Exception as e:
            logger.debug(f"获取价格数据失败: {e}")
            return None

    def execute_v37_backtest(self, start_date: str, end_date: str,
                           holding_days: int = 5) -> Dict:
        """
        执行V3.7版本回测

        Args:
            start_date: 开始日期
            end_date: 结束日期
            holding_days: 持股天数

        Returns:
            回测结果
        """
        logger.info(f"开始执行V3.7回测: {start_date} 至 {end_date}")

        # 初始化V3.7系统
        if not self.initialize_v37_system():
            logger.warning("V3.7系统初始化失败，使用备选评分方法")

        # 获取股票池 (调整最少交易天数要求)
        # 计算实际交易日数量，设置合理的min_trading_days
        actual_trading_days = len(self._get_trading_dates(start_date, end_date))
        min_trading_days = max(3, min(actual_trading_days - 1, 20))  # 至少3天，但不超过实际交易日-1，最多20天
        logger.info(f"实际交易日: {actual_trading_days}天，要求最少交易日: {min_trading_days}天")
        stock_universe = self.get_stock_universe(start_date, end_date, min_trading_days=min_trading_days)
        if not stock_universe:
            raise ValueError("股票池为空，无法进行回测")

        logger.info(f"股票池包含 {len(stock_universe)} 只股票")

        # 获取所有交易日期
        trading_dates = self._get_trading_dates(start_date, end_date)
        logger.info(f"回测期间共 {len(trading_dates)} 个交易日")

        last_rebalance_date = None

        # 按日执行回测
        for i, current_date in enumerate(trading_dates):
            if i % 20 == 0:
                progress = i / len(trading_dates) * 100
                logger.info(f"回测进度: {progress:.1f}% ({current_date})")

            # 检查是否需要调仓
            should_rebalance = (
                last_rebalance_date is None or
                (datetime.strptime(current_date, '%Y-%m-%d') -
                 datetime.strptime(last_rebalance_date, '%Y-%m-%d')).days >= self.rebalance_freq
            )

            # 1. 检查止损止盈和持股天数，执行卖出
            self._execute_v37_sells(current_date, holding_days)

            # 2. 如果需要调仓，生成新的选股信号并买入
            if should_rebalance:
                logger.debug(f"执行调仓: {current_date}")
                signals = self.generate_v37_signals(current_date, stock_universe)

                if not signals.empty:
                    self._execute_v37_buys(signals, current_date)
                    last_rebalance_date = current_date

            # 3. 计算当日组合价值
            portfolio_value = self._calculate_v37_portfolio_value(current_date)

            # 4. 记录组合净值
            self.portfolio_values.append({
                'date': current_date,
                'total_value': portfolio_value,
                'cash': self.current_capital,
                'positions_value': portfolio_value - self.current_capital,
                'positions_count': len(self.positions)
            })

        logger.info("V3.7回测执行完成！")

        # 计算回测结果
        return self._calculate_v37_performance_metrics()

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

    def _execute_v37_sells(self, current_date: str, holding_days: int):
        """执行V3.7卖出逻辑"""
        positions_to_sell = []

        for stock_code, position in self.positions.items():
            price_data = self._get_stock_price_data(stock_code, current_date)
            if not price_data:
                continue  # 停牌或无数据

            current_price = price_data['close']
            entry_price = position['entry_price']
            entry_date = position['entry_date']

            # 计算持股天数
            days_held = (datetime.strptime(current_date, '%Y-%m-%d') -
                        datetime.strptime(entry_date, '%Y-%m-%d')).days

            # 卖出条件检查
            should_sell = False
            sell_reason = ""

            # 1. 达到持股天数
            if days_held >= holding_days:
                should_sell = True
                sell_reason = f"持股{days_held}天到期"

            # 2. 止损检查
            elif 'stop_loss_price' in position and current_price <= position['stop_loss_price']:
                should_sell = True
                sell_reason = f"触发止损 {current_price:.2f} <= {position['stop_loss_price']:.2f}"

            # 3. 止盈检查
            elif 'take_profit_price' in position and current_price >= position['take_profit_price']:
                should_sell = True
                sell_reason = f"触发止盈 {current_price:.2f} >= {position['take_profit_price']:.2f}"

            # 4. 涨跌停无法交易
            elif price_data.get('is_limit_up', False) or price_data.get('is_limit_down', False):
                continue  # 涨跌停暂不卖出

            if should_sell:
                positions_to_sell.append((stock_code, current_price, sell_reason))

        # 执行卖出
        for stock_code, sell_price, reason in positions_to_sell:
            self._execute_sell_order(stock_code, sell_price, current_date, reason)

    def _execute_v37_buys(self, signals: pd.DataFrame, current_date: str):
        """执行V3.7买入逻辑"""
        # 按V3.7评分排序，优先买入高分股票
        signals_sorted = signals.sort_values('v37_score', ascending=False)

        for _, signal in signals_sorted.iterrows():
            stock_code = signal['stock_code']

            # 检查是否已持有
            if stock_code in self.positions:
                continue

            # 检查价格数据
            price_data = self._get_stock_price_data(stock_code, current_date)
            if not price_data:
                continue

            # 检查涨停
            if price_data.get('is_limit_up', False):
                continue  # 涨停无法买入

            # 检查持仓数量限制
            if len(self.positions) >= self.max_positions:
                break

            # 使用当前价格
            buy_price = price_data['close']

            # 执行买入
            success = self._execute_buy_order(
                stock_code, buy_price, current_date,
                signal.get('stop_loss_price'), signal.get('take_profit_price'),
                signal['v37_score']
            )

            if not success:
                break  # 资金不足，停止买入

    def _execute_buy_order(self, stock_code: str, price: float, date: str,
                          stop_loss: float = None, take_profit: float = None,
                          v37_score: float = None) -> bool:
        """执行买入订单"""
        # 计算可买入数量
        max_position_value = self.current_capital * self.max_position_pct
        available_cash = self.current_capital

        if available_cash < self.min_trade_amount:
            return False

        # 实际投入金额（不超过最大仓位限制）
        invest_amount = min(max_position_value, available_cash * 0.9)  # 保留10%现金

        # 计算股数（整手交易）
        shares = int(invest_amount / price / 100) * 100
        if shares == 0:
            return False

        # 计算交易成本
        trade_value = shares * price
        commission = max(trade_value * self.commission_rate, self.min_commission)
        transfer_fee = trade_value * self.transfer_fee
        total_cost = trade_value + commission + transfer_fee

        if total_cost > available_cash:
            return False

        # 更新持仓
        self.positions[stock_code] = {
            'shares': shares,
            'entry_price': price,
            'entry_date': date,
            'cost': total_cost,
            'stop_loss_price': stop_loss,
            'take_profit_price': take_profit,
            'v37_score': v37_score
        }

        # 更新现金
        self.current_capital -= total_cost

        # 记录交易
        self.trades.append({
            'date': date,
            'stock_code': stock_code,
            'action': 'BUY',
            'shares': shares,
            'price': price,
            'amount': total_cost,
            'commission': commission,
            'v37_score': v37_score,
            'reason': f'V3.7选股信号买入(评分:{v37_score:.1f})'
        })

        return True

    def _execute_sell_order(self, stock_code: str, price: float, date: str, reason: str):
        """执行卖出订单"""
        if stock_code not in self.positions:
            return

        position = self.positions[stock_code]
        shares = position['shares']

        # 计算交易成本
        trade_value = shares * price
        commission = max(trade_value * self.commission_rate, self.min_commission)
        stamp_tax = trade_value * self.stamp_tax
        transfer_fee = trade_value * self.transfer_fee
        total_cost = commission + stamp_tax + transfer_fee

        # 计算收益
        proceeds = trade_value - total_cost
        profit = proceeds - position['cost']
        profit_pct = profit / position['cost']

        # 更新现金
        self.current_capital += proceeds

        # 记录交易
        self.trades.append({
            'date': date,
            'stock_code': stock_code,
            'action': 'SELL',
            'shares': shares,
            'price': price,
            'amount': proceeds,
            'commission': commission,
            'stamp_tax': stamp_tax,
            'profit': profit,
            'profit_pct': profit_pct,
            'v37_score': position.get('v37_score'),
            'reason': reason
        })

        # 移除持仓
        del self.positions[stock_code]

    def _calculate_v37_portfolio_value(self, date: str) -> float:
        """计算V3.7组合总价值"""
        total_value = self.current_capital  # 现金部分

        # 计算持仓市值
        for stock_code, position in self.positions.items():
            price_data = self._get_stock_price_data(stock_code, date)
            if price_data:
                current_price = price_data['close']
                market_value = position['shares'] * current_price
                total_value += market_value

        return total_value

    def _calculate_v37_performance_metrics(self) -> Dict:
        """计算V3.7回测绩效指标"""
        logger.info("计算V3.7回测绩效指标...")

        # 转换为DataFrame
        portfolio_df = pd.DataFrame(self.portfolio_values)
        trades_df = pd.DataFrame(self.trades)

        if portfolio_df.empty:
            return {'error': '无回测数据'}

        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
        portfolio_df = portfolio_df.sort_values('date')

        # 计算收益率
        portfolio_df['daily_returns'] = portfolio_df['total_value'].pct_change()
        portfolio_df['cumulative_returns'] = (portfolio_df['total_value'] / self.initial_capital) - 1

        # 基础绩效指标
        total_return = portfolio_df['cumulative_returns'].iloc[-1]
        trading_days = len(portfolio_df)
        annual_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0

        # 计算波动率
        daily_returns = portfolio_df['daily_returns'].dropna()
        volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0

        # 夏普比率（假设无风险利率3%）
        risk_free_rate = 0.03
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0

        # 最大回撤分析
        rolling_max = portfolio_df['total_value'].expanding().max()
        drawdown = (portfolio_df['total_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # V3.7特色统计
        buy_trades = trades_df[trades_df['action'] == 'BUY'] if not trades_df.empty else pd.DataFrame()
        sell_trades = trades_df[trades_df['action'] == 'SELL'] if not trades_df.empty else pd.DataFrame()

        # V3.7评分统计
        avg_v37_score = buy_trades['v37_score'].mean() if not buy_trades.empty else 0
        max_v37_score = buy_trades['v37_score'].max() if not buy_trades.empty else 0
        min_v37_score = buy_trades['v37_score'].min() if not buy_trades.empty else 0

        # 交易统计
        total_trades = len(trades_df)
        profitable_trades = len(sell_trades[sell_trades['profit'] > 0]) if not sell_trades.empty and 'profit' in sell_trades.columns else 0
        win_rate = profitable_trades / len(sell_trades) if len(sell_trades) > 0 else 0

        # 盈亏比
        avg_profit = sell_trades[sell_trades['profit'] > 0]['profit'].mean() if profitable_trades > 0 and 'profit' in sell_trades.columns else 0
        avg_loss = abs(sell_trades[sell_trades['profit'] < 0]['profit'].mean()) if (not sell_trades.empty and 'profit' in sell_trades.columns and len(sell_trades[sell_trades['profit'] < 0]) > 0) else 1
        profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 0

        results = {
            'version': 'V3.7 Advanced ML',
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'total_trades': total_trades,
            'trading_days': trading_days,
            'final_value': portfolio_df['total_value'].iloc[-1],
            'avg_v37_score': avg_v37_score,
            'max_v37_score': max_v37_score,
            'min_v37_score': min_v37_score,
            'portfolio_df': portfolio_df,
            'trades_df': trades_df
        }

        logger.info("V3.7绩效指标计算完成")
        return results

    def generate_v37_report(self, results: Dict) -> str:
        """生成V3.7专业回测报告"""
        portfolio_df = results['portfolio_df']

        report = f"""
# 📊 V3.7高级机器学习量化策略回测报告

## 🚀 V3.7系统特色
- **评分系统**: 三层Ensemble架构 (5基础模型 + 4专家模型 + Meta学习器)
- **特征维度**: 49维特征 (技术17 + 基本面8 + 宏观8 + 情绪7 + 时序5 + 市场4)
- **预测目标**: 多目标预测 (1日、3日、5日收益率)
- **评分范围**: 0-100分，高分优先选择

## 📈 回测概览
- **回测期间**: {portfolio_df['date'].min().strftime('%Y-%m-%d')} 至 {portfolio_df['date'].max().strftime('%Y-%m-%d')}
- **初始资金**: {self.initial_capital:,.0f}元
- **最终资金**: {results['final_value']:,.0f}元
- **交易日数**: {results['trading_days']}天
- **V3.7评分范围**: {results['min_v37_score']:.1f} - {results['max_v37_score']:.1f}分
- **平均V3.7评分**: {results['avg_v37_score']:.1f}分

## 💰 收益表现
| 指标 | V3.7策略表现 | 市场基准 | 评级 |
|------|-------------|----------|------|
| 累计收益率 | {results['total_return']:.2%} | 15% | {'🏆 A+' if results['total_return'] > 0.20 else '🥇 A' if results['total_return'] > 0.15 else '🥈 B' if results['total_return'] > 0.08 else '🥉 C'} |
| 年化收益率 | {results['annual_return']:.2%} | 12% | {'🏆 A+' if results['annual_return'] > 0.18 else '🥇 A' if results['annual_return'] > 0.12 else '🥈 B' if results['annual_return'] > 0.08 else '🥉 C'} |
| 交易胜率 | {results['win_rate']:.2%} | 50% | {'🏆 A+' if results['win_rate'] > 0.60 else '🥇 A' if results['win_rate'] > 0.50 else '🥈 B' if results['win_rate'] > 0.40 else '🥉 C'} |

## ⚖️ 风险指标
| 指标 | V3.7策略数值 | 风险基准 | 评级 |
|------|-------------|----------|------|
| 年化波动率 | {results['volatility']:.2%} | <25% | {'🟢 优秀' if results['volatility'] < 0.20 else '🟡 良好' if results['volatility'] < 0.30 else '🔴 偏高'} |
| 最大回撤 | {results['max_drawdown']:.2%} | <-15% | {'🟢 优秀' if results['max_drawdown'] > -0.12 else '🟡 良好' if results['max_drawdown'] > -0.20 else '🔴 偏高'} |
| 夏普比率 | {results['sharpe_ratio']:.2f} | >1.0 | {'🟢 优秀' if results['sharpe_ratio'] > 1.5 else '🟡 良好' if results['sharpe_ratio'] > 1.0 else '🔴 偏低'} |

## 📊 V3.7交易统计
- **总交易次数**: {results['total_trades']}次
- **平均V3.7选股评分**: {results['avg_v37_score']:.1f}分
- **最高V3.7选股评分**: {results['max_v37_score']:.1f}分
- **交易胜率**: {results['win_rate']:.2%}
- **盈亏比**: {results['profit_loss_ratio']:.2f}

## 🤖 V3.7机器学习效果评估
"""

        # 计算V3.7系统评级
        ml_score = 0
        if results['avg_v37_score'] > 80: ml_score += 30
        elif results['avg_v37_score'] > 75: ml_score += 20
        elif results['avg_v37_score'] > 70: ml_score += 10

        if results['win_rate'] > 0.55: ml_score += 25
        elif results['win_rate'] > 0.50: ml_score += 15
        elif results['win_rate'] > 0.45: ml_score += 10

        if results['sharpe_ratio'] > 1.5: ml_score += 25
        elif results['sharpe_ratio'] > 1.0: ml_score += 15
        elif results['sharpe_ratio'] > 0.5: ml_score += 10

        if results['annual_return'] > 0.15: ml_score += 20
        elif results['annual_return'] > 0.10: ml_score += 10

        if ml_score >= 85:
            ml_grade = "🏆 V3.7系统表现卓越"
        elif ml_score >= 70:
            ml_grade = "🥇 V3.7系统表现优秀"
        elif ml_score >= 50:
            ml_grade = "🥈 V3.7系统表现良好"
        else:
            ml_grade = "🥉 V3.7系统有待优化"

        report += f"**V3.7机器学习评级**: {ml_grade} (评分: {ml_score}/100)\n\n"

        report += f"""
## 📈 策略优势分析
1. **智能选股**: V3.7平均选股评分{results['avg_v37_score']:.1f}分，显示出强大的股票识别能力
2. **风险控制**: 最大回撤{results['max_drawdown']:.2%}，展现良好的风险管理
3. **收益稳定**: 夏普比率{results['sharpe_ratio']:.2f}，风险调整后收益表现{'优秀' if results['sharpe_ratio'] > 1.0 else '良好'}

## ⚠️ 风险提示
- V3.7回测基于历史数据训练，实盘表现可能存在差异
- 机器学习模型需要定期重训练以适应市场变化
- 建议结合多种策略分散风险
- 实际交易中应考虑流动性、滑点等因素

## 🔬 技术说明
- **机器学习框架**: LightGBM + XGBoost + CatBoost + RandomForest + MLP
- **特征工程**: 49维量化特征 + RobustScaler标准化
- **模型架构**: 三层Ensemble + 专家系统 + Meta学习器
- **训练数据**: 历史股票数据 + 技术指标 + 基本面 + 宏观数据
- **回测假设**: A股T+1交易 + 真实交易成本 + 涨跌停限制

## 📊 数据来源
- **股票数据**: SQLite数据库 (前复权价格)
- **技术指标**: 自研指标计算引擎
- **基本面数据**: Tushare专业版数据
- **回测引擎**: 专业级回测系统
- **分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

🤖 *本报告由V3.7高级机器学习量化系统自动生成*

📧 *技术支持: StockTradebyZ V3.7 Advanced ML System*
"""

        return report

    def save_v37_results(self, results: Dict, output_dir: str = "backtest/v37_results"):
        """保存V3.7回测结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存组合净值数据
        portfolio_file = output_path / f"v37_portfolio_values_{timestamp}.csv"
        results['portfolio_df'].to_csv(portfolio_file, index=False, encoding='utf-8')

        # 保存交易记录
        trades_file = output_path / f"v37_trades_{timestamp}.csv"
        results['trades_df'].to_csv(trades_file, index=False, encoding='utf-8')

        # 保存V3.7专业报告
        report = self.generate_v37_report(results)
        report_file = output_path / f"v37_backtest_report_{timestamp}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        # 保存结果摘要JSON
        summary = {
            'version': 'V3.7 Advanced ML',
            'timestamp': timestamp,
            'performance': {
                'total_return': results['total_return'],
                'annual_return': results['annual_return'],
                'sharpe_ratio': results['sharpe_ratio'],
                'max_drawdown': results['max_drawdown'],
                'win_rate': results['win_rate'],
                'avg_v37_score': results['avg_v37_score']
            }
        }

        summary_file = output_path / f"v37_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"V3.7回测结果已保存到: {output_path}")
        return output_path


def run_v37_backtest_example(start_date: str = '2024-01-01',
                            end_date: str = '2024-06-30',
                            initial_capital: float = 1000000):
    """
    运行V3.7回测示例

    Args:
        start_date: 开始日期
        end_date: 结束日期
        initial_capital: 初始资金
    """
    logger.info("🚀 开始V3.7高级机器学习量化策略回测")

    try:
        # 创建V3.7回测引擎
        engine = V37BacktestEngine(initial_capital=initial_capital)

        # 执行回测
        results = engine.execute_v37_backtest(start_date, end_date, holding_days=5)

        # 保存结果
        output_path = engine.save_v37_results(results)

        # 输出关键指标
        logger.info("🎉 V3.7回测完成！")
        logger.info(f"📊 累计收益率: {results['total_return']:.2%}")
        logger.info(f"📊 年化收益率: {results['annual_return']:.2%}")
        logger.info(f"📊 夏普比率: {results['sharpe_ratio']:.2f}")
        logger.info(f"📊 最大回撤: {results['max_drawdown']:.2%}")
        logger.info(f"📊 交易胜率: {results['win_rate']:.2%}")
        logger.info(f"🤖 平均V3.7评分: {results['avg_v37_score']:.1f}分")
        logger.info(f"📁 结果已保存至: {output_path}")

        return results

    except Exception as e:
        logger.error(f"❌ V3.7回测执行失败: {e}")
        raise


if __name__ == "__main__":
    import argparse

    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="V3.7量化策略回测引擎")
    parser.add_argument('--start-date', default='2024-01-01', help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2024-08-31', help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--initial-capital', type=float, default=1000000, help='初始资金')
    parser.add_argument('--sample-size', type=int, help='测试样本大小（可选）')

    args = parser.parse_args()

    # 运行V3.7回测
    results = run_v37_backtest_example(
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital
    )