#!/usr/bin/env python3
"""
股票量化策略回测引擎
基于专业的回测专家指导，实现完整的回测系统
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

# 添加项目路径以支持V3.7导入
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 尝试导入V3.7选股系统
try:
    from tomorrow_stock_selector import TomorrowStockSelector
    from data_adapter.database_manager import DatabaseManager
    V37_AVAILABLE = True
except ImportError as e:
    V37_AVAILABLE = False
    TomorrowStockSelector = None
    DatabaseManager = None

# 可选的matplotlib导入
try:
    import matplotlib.pyplot as plt
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("backtest/logs/backtest.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("backtest_engine")

class StockBacktester:
    """股票量化策略回测器 - 专业版"""
    
    def __init__(self, initial_capital: float = 1000000, scoring_version: str = 'traditional',
                 sample_size: int = 200, parallel_workers: int = 4):
        """
        初始化回测器

        Args:
            initial_capital: 初始资金
            scoring_version: 评分版本 ('traditional', 'v3.7', 'v3.0', 'v3.4', 'v3.5')
            sample_size: 每日采样股票数量（默认200，优化性能）
            parallel_workers: 并行工作进程数（默认4）
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # 持仓记录: {stock_code: {shares, cost, entry_date, entry_price}}
        self.trades = []     # 交易记录
        self.daily_returns = []  # 每日收益
        self.portfolio_values = []  # 组合净值
        
        # 交易成本设置（严格按照A股实际成本）
        self.commission_rate = 0.0003  # 万分之三佣金
        self.stamp_tax = 0.001         # 千分之一印花税（仅卖出）
        self.transfer_fee = 0.00002    # 过户费（双向）
        self.min_commission = 5.0      # 最低佣金5元
        
        # 风控设置
        self.max_position_pct = 0.10   # 单股最大仓位10%
        self.max_positions = 20        # 最大持股数量
        self.min_trade_amount = 1000   # 最小交易金额

        # 评分版本设置
        self.scoring_version = scoring_version
        self.selector = None
        self.db_manager = None

        # 性能优化设置
        self.sample_size = sample_size
        self.parallel_workers = parallel_workers

        # 根据评分版本初始化相应的选股系统
        if scoring_version in ['v3.7', 'v3.0', 'v3.4', 'v3.5'] and V37_AVAILABLE:
            try:
                logger.info(f"初始化{scoring_version}评分系统...")
                self.selector = TomorrowStockSelector(scoring_version=scoring_version, stocks_only=True)
                self.db_manager = DatabaseManager()
                logger.info(f"✅ {scoring_version}评分系统初始化成功")
            except Exception as e:
                logger.warning(f"⚠️ {scoring_version}评分系统初始化失败，将使用传统方法: {e}")
                self.scoring_version = 'traditional'
        elif scoring_version != 'traditional':
            logger.warning(f"⚠️ 不支持的评分版本{scoring_version}，将使用传统方法")
            self.scoring_version = 'traditional'

        logger.info(f"回测器初始化完成，初始资金: {initial_capital:,.0f}元，评分版本: {self.scoring_version}")
    
    def load_stock_data_from_database(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从本地SQLite数据库加载真实股票数据

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            清洗后的股票数据
        """
        logger.info(f"从数据库加载股票数据: {start_date} 至 {end_date}")

        if not self.db_manager:
            self.db_manager = DatabaseManager()

        # 从数据库查询股票数据
        query = """
        SELECT
            s.code as stock_code,
            s.name as stock_name,
            dq.trade_date as date,
            dq.open,
            dq.high,
            dq.low,
            dq.close,
            dq.volume,
            dq.amount,
            dq.price_change_pct,
            dq.is_limit_up,
            dq.is_limit_down
        FROM securities s
        JOIN daily_quotes dq ON s.id = dq.security_id
        WHERE s.type = 'A股'
        AND dq.trade_date BETWEEN ? AND ?
        AND dq.close IS NOT NULL
        AND dq.volume > 0
        ORDER BY dq.trade_date, s.code
        """

        logger.info("执行数据库查询...")
        data_result = self.db_manager.execute_query(query, (start_date, end_date))

        # 将查询结果转换为DataFrame
        if isinstance(data_result, list):
            if not data_result:
                logger.error("未从数据库获取到股票数据")
                return pd.DataFrame()

            # 获取列名
            column_names = ['stock_code', 'stock_name', 'date', 'open', 'high', 'low',
                          'close', 'volume', 'amount', 'price_change_pct', 'is_limit_up', 'is_limit_down']
            data = pd.DataFrame(data_result, columns=column_names)
        else:
            data = data_result

        if data.empty:
            logger.error("未从数据库获取到股票数据")
            return pd.DataFrame()

        # 转换数据格式
        data['date'] = pd.to_datetime(data['date'])

        # 数据清洗：移除异常数据
        original_len = len(data)
        data = data[(data['open'] > 0) & (data['high'] > 0) &
                   (data['low'] > 0) & (data['close'] > 0) &
                   (data['volume'] > 0)]

        logger.info(f"数据清洗完成：{original_len} -> {len(data)} 条记录")
        logger.info(f"数据覆盖期间：{data['date'].min()} 至 {data['date'].max()}")
        logger.info(f"包含股票数量：{data['stock_code'].nunique()}")

        return data

    def load_stock_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从本地SQLite数据库加载真实股票数据

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            清洗后的股票数据
        """
        return self.load_stock_data_from_database(start_date, end_date)

    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据清洗和质量检查"""
        logger.info("开始数据清洗...")
        
        original_len = len(data)
        
        # 移除缺失值
        data = data.dropna(subset=['close', 'volume', 'high', 'low', 'open'])
        
        # 处理价格异常值
        data = data[data['close'] > 0]
        data = data[data['high'] >= data['low']]
        data = data[data['high'] >= data['close']]
        data = data[data['low'] <= data['close']]
        data = data[data['volume'] >= 0]
        
        # 检测并标记涨跌停
        data = data.sort_values(['stock_code', 'date'])
        data['prev_close'] = data.groupby('stock_code')['close'].shift(1)
        
        # 计算涨跌幅
        data['price_change_pct'] = (data['close'] - data['prev_close']) / data['prev_close']
        
        # 标记涨跌停（考虑10%和20%的情况）
        data['is_limit_up'] = data['price_change_pct'] >= 0.095  # 9.5%以上认为涨停
        data['is_limit_down'] = data['price_change_pct'] <= -0.095  # -9.5%以下认为跌停
        
        # 标记ST股票（简单规则：连续下跌或价格过低）
        data['is_st'] = data['close'] < 3.0  # 价格低于3元的可能是ST
        
        cleaned_len = len(data)
        logger.info(f"数据清洗完成，剩余数据: {cleaned_len} 条 (清洗掉 {original_len - cleaned_len} 条)")
        
        return data
    
    def _adjust_prices(self, data: pd.DataFrame) -> pd.DataFrame:
        """前复权价格调整"""
        logger.info("开始前复权价格调整...")
        
        # 简化的前复权处理（如果没有复权因子，使用原始价格）
        if 'adj_close' not in data.columns:
            # 如果没有复权数据，创建复权因子为1
            data['adj_factor'] = 1.0
            for col in ['open', 'high', 'low', 'close']:
                data[f'adj_{col}'] = data[col]
        else:
            # 计算复权因子
            data['adj_factor'] = data['adj_close'] / data['close']
            
            # 应用复权调整
            for col in ['open', 'high', 'low', 'close']:
                if f'adj_{col}' not in data.columns:
                    data[f'adj_{col}'] = data[col] * data['adj_factor']
        
        logger.info("前复权价格调整完成")
        return data
    
    def load_selection_reports(self, reports_dir: str) -> pd.DataFrame:
        """
        加载历史选股报告，解析推荐股票和建议价格
        
        Args:
            reports_dir: 选股报告目录
            
        Returns:
            解析后的选股信号数据
        """
        logger.info(f"加载选股报告: {reports_dir}")
        
        reports_path = Path(reports_dir)
        all_signals = []
        
        # 加载所有markdown报告文件
        report_files = list(reports_path.glob("选股分析报告_*.md"))
        logger.info(f"找到 {len(report_files)} 个选股报告文件")
        
        for report_file in report_files:
            try:
                # 从文件名提取日期
                date_str = report_file.stem.split('_')[1]  # 选股分析报告_20250101.md
                report_date = datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
                
                # 解析报告内容
                signals = self._parse_report_content(report_file, report_date)
                all_signals.extend(signals)
                
            except Exception as e:
                logger.warning(f"解析报告 {report_file} 失败: {e}")
                continue
        
        if not all_signals:
            raise ValueError("未能解析出任何选股信号")
        
        signals_df = pd.DataFrame(all_signals)
        signals_df['date'] = pd.to_datetime(signals_df['date'])
        signals_df = signals_df.sort_values('date')
        
        logger.info(f"成功解析 {len(signals_df)} 个选股信号")
        return signals_df
    
    def _parse_report_content(self, report_file: Path, report_date: str) -> List[Dict]:
        """解析单个报告文件内容，提取股票推荐信息"""
        signals = []
        
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查是否为非交易日
            if "不是交易日" in content:
                return signals
            
            # 使用正则表达式解析股票信息
            import re
            
            # 匹配股票信息的正则模式
            pattern = r'### \d+\. (\d+) - (.+?)\n.*?综合评分.*?(\d+\.?\d*)分.*?建议买入价.*?(\d+\.?\d*)元.*?建议止损价.*?(\d+\.?\d*)元.*?建议止盈价.*?(\d+\.?\d*)元.*?风险收益比.*?1:(\d+\.?\d*)'
            
            matches = re.findall(pattern, content, re.DOTALL)
            
            for match in matches:
                stock_code, stock_name, score, buy_price, stop_loss, take_profit, risk_reward = match
                
                signal = {
                    'date': report_date,
                    'stock_code': stock_code,
                    'stock_name': stock_name,
                    'signal': 'BUY',
                    'comprehensive_score': float(score),
                    'suggested_buy_price': float(buy_price),
                    'stop_loss_price': float(stop_loss),
                    'take_profit_price': float(take_profit),
                    'risk_reward_ratio': float(risk_reward)
                }
                
                signals.append(signal)
                
        except Exception as e:
            logger.warning(f"解析报告内容失败: {e}")
        
        return signals

    def generate_signals_by_date_range(self, start_date: str, end_date: str,
                                     max_stocks_per_day: int = 20) -> pd.DataFrame:
        """
        根据评分版本动态生成选股信号

        Args:
            start_date: 开始日期
            end_date: 结束日期
            max_stocks_per_day: 每日最大选股数量

        Returns:
            选股信号DataFrame
        """
        logger.info(f"使用{self.scoring_version}评分版本生成选股信号: {start_date} 至 {end_date}")

        if self.scoring_version == 'traditional':
            # 传统方法需要使用已有的选股报告
            logger.warning("传统评分版本需要使用load_selection_reports方法加载历史报告")
            return pd.DataFrame()

        if not self.selector:
            logger.error("选股系统未初始化")
            return pd.DataFrame()

        all_signals = []

        # 预先初始化V3.7系统（避免重复初始化）
        v37_scorer = None
        if self.scoring_version == 'v3.7':
            from ml_models.v37 import V370AdvancedMLSystem
            logger.info("🔄 一次性初始化V3.7评分系统...")
            v37_scorer = V370AdvancedMLSystem()
            logger.info("✅ V3.7评分系统初始化完成")


        # 获取交易日历
        from datetime import datetime, timedelta
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        current_date = start
        while current_date <= end:
            date_str = current_date.strftime('%Y-%m-%d')

            try:
                # 使用选股器生成当日选股
                logger.info(f"📅 处理交易日: {date_str}")

                # V3.7独立评分回测方式：采样+评分+买入决策
                # 1. 从数据库采样活跃股票（按成交量排序）
                sample_query = """
                SELECT DISTINCT s.code, s.name, dq.close, dq.volume
                FROM securities s
                JOIN daily_quotes dq ON s.id = dq.security_id
                WHERE s.type = 'A股'
                AND dq.trade_date = ?
                AND dq.volume > 1000000  -- 成交额大于100万
                AND dq.close > 3.0       -- 价格大于3元
                AND dq.close < 100.0     -- 价格小于100元
                ORDER BY dq.volume DESC
                LIMIT ?
                """

                logger.info(f"🔍 查询{date_str}的活跃股票...")
                sample_result = self.db_manager.execute_query(sample_query, (date_str, self.sample_size))  # 使用可配置采样数量

                if not sample_result:
                    logger.info(f"⚠️ {date_str}: 没有找到符合条件的股票，跳过")
                    continue

                logger.info(f"✅ {date_str}: 采样到 {len(sample_result)} 只活跃股票")

                # 2. 对采样股票进行V3.7评分（使用预初始化的系统）
                if not v37_scorer:
                    logger.error("V3.7评分系统未初始化")
                    continue


                scored_stocks = []
                logger.info(f"🤖 开始V3.7评分，处理{len(sample_result)}只股票...")

                for i, (stock_code, stock_name, close_price, volume) in enumerate(sample_result):
                    try:
                        if i % 10 == 0:  # 每10只股票显示一次进度
                            logger.info(f"⚡ 评分进度: {i+1}/{len(sample_result)}")

                        # 获取V3.7评分 (V3.7系统内部从数据库加载数据，只需传递股票代码)
                        try:
                            # V3.7特征提取：传递股票代码列表而非DataFrame
                            features = v37_scorer.extract_advanced_features(
                                codes=[stock_code],  # 股票代码列表
                                start_date=date_str,
                                end_date=date_str
                            )

                            if features is not None and not features.empty:
                                predictions = v37_scorer.predict_three_layer_ensemble(features)
                                v37_score = predictions['predictions_1d'][0] * 100  # 转换为0-100分数

                                scored_stocks.append({
                                    'stock_code': stock_code,
                                    'stock_name': stock_name,
                                    'close_price': close_price,
                                    'v37_score': v37_score,
                                    'volume': volume
                                })
                            else:
                                logger.warning(f"⚠️ {stock_code}特征提取失败，跳过评分")
                        except Exception as feature_error:
                            logger.warning(f"⚠️ {stock_code}V3.7特征提取异常: {feature_error}")
                            continue
                    except Exception as e:
                        logger.warning(f"⚠️ {stock_code}评分失败: {e}")
                        continue

                logger.info(f"✅ {date_str}评分完成，获得{len(scored_stocks)}只有效股票")

                # 3. 根据评分决定买入（选择高分股票）
                scored_stocks.sort(key=lambda x: x['v37_score'], reverse=True)
                top_scored = scored_stocks[:max_stocks_per_day]

                for stock in top_scored:
                    if stock['v37_score'] >= 75.0:  # 只买入高分股票
                        signal = {
                            'date': date_str,
                            'stock_code': stock['stock_code'],
                            'stock_name': stock['stock_name'],
                            'signal': 'BUY',
                            'comprehensive_score': stock['v37_score'],
                            'suggested_buy_price': stock['close_price'],
                            'stop_loss_price': stock['close_price'] * 0.92,
                            'take_profit_price': stock['close_price'] * 1.15,
                            'risk_reward_ratio': 1.88
                        }
                        all_signals.append(signal)

                logger.info(f"📊 {date_str}: 选出 {len([s for s in scored_stocks if s['v37_score'] >= 75.0])} 只高分股票")

            except Exception as e:
                logger.error(f"❌ 生成{date_str}选股信号失败: {e}")

            finally:
                # 确保每次循环都递增日期，避免无限循环
                current_date += timedelta(days=1)

        signals_df = pd.DataFrame(all_signals)
        if not signals_df.empty:
            signals_df['date'] = pd.to_datetime(signals_df['date'])
            logger.info(f"✅ 使用{self.scoring_version}评分成功生成 {len(signals_df)} 个选股信号")
        else:
            logger.warning(f"⚠️ 使用{self.scoring_version}评分未生成任何选股信号")

        return signals_df

    def load_v37_json_reports(self, reports_dir: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从V3.7 JSON报告中加载选股信号

        Args:
            reports_dir: V3.7报告目录
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            选股信号DataFrame
        """
        logger.info(f"加载V3.7 JSON报告: {reports_dir}")

        import json
        from datetime import datetime

        reports_path = Path(reports_dir)
        all_signals = []

        # 查找JSON数据文件
        json_files = list(reports_path.glob("analysis_data_*.json"))
        logger.info(f"找到 {len(json_files)} 个V3.7 JSON报告文件")

        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        for json_file in json_files:
            try:
                # 从文件名提取日期
                date_str = json_file.stem.split('_')[2]  # analysis_data_20250919.json
                report_date = datetime.strptime(date_str, '%Y%m%d')

                # 检查日期范围
                if not (start_dt <= report_date <= end_dt):
                    continue

                # 读取JSON数据
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if 'top_recommendations' in data:
                    recommendations = data['top_recommendations']

                    for rec in recommendations:
                        # 只选择高分股票
                        if rec.get('score', 0) >= 75.0:
                            signal = {
                                'date': report_date.strftime('%Y-%m-%d'),
                                'stock_code': rec.get('stock_code', ''),
                                'stock_name': rec.get('stock_name', ''),
                                'signal': 'BUY',
                                'comprehensive_score': rec.get('score', 0),
                                'suggested_buy_price': rec.get('suggested_buy_price', rec.get('close_price', 10.0)),
                                'stop_loss_price': rec.get('stop_loss_price', rec.get('close_price', 10.0) * 0.92),
                                'take_profit_price': rec.get('take_profit_price', rec.get('close_price', 10.0) * 1.15),
                                'risk_reward_ratio': rec.get('risk_reward_ratio', 1.88)
                            }
                            all_signals.append(signal)

            except Exception as e:
                logger.warning(f"解析V3.7报告 {json_file} 失败: {e}")
                continue

        signals_df = pd.DataFrame(all_signals)
        if not signals_df.empty:
            signals_df['date'] = pd.to_datetime(signals_df['date'])
            logger.info(f"✅ 成功加载 {len(signals_df)} 个V3.7真实选股信号")
        else:
            logger.warning(f"⚠️ 未找到指定期间的V3.7选股数据")

        return signals_df

    def execute_backtest(self, stock_data: pd.DataFrame, signals_data: pd.DataFrame, 
                        holding_days: int = 5) -> Dict:
        """
        执行完整回测
        
        Args:
            stock_data: 股票价格数据
            signals_data: 选股信号数据
            holding_days: 持股天数
            
        Returns:
            回测结果
        """
        logger.info("开始执行回测...")
        
        # 按日期排序
        signals_data = signals_data.sort_values('date').copy()
        stock_data = stock_data.sort_values(['date', 'stock_code']).copy()
        
        # 获取所有交易日期
        all_dates = sorted(stock_data['date'].unique())
        signal_dates = set(signals_data['date'].dt.strftime('%Y-%m-%d'))
        
        logger.info(f"回测期间: {all_dates[0]} 至 {all_dates[-1]}")
        logger.info(f"交易日数: {len(all_dates)}, 选股信号日数: {len(signal_dates)}")
        
        # 按日执行回测
        for i, current_date in enumerate(all_dates):
            if i % 50 == 0:
                progress = i / len(all_dates) * 100
                logger.info(f"回测进度: {progress:.1f}% ({current_date})")
            
            current_date_str = current_date.strftime('%Y-%m-%d')
            
            # 获取当日股票数据
            daily_prices = stock_data[stock_data['date'] == current_date]
            daily_prices_dict = daily_prices.set_index('stock_code').to_dict('index')
            
            # 1. 检查止损止盈和持股天数，执行卖出
            self._execute_sells(daily_prices_dict, current_date, holding_days)
            
            # 2. 如果有选股信号，执行买入
            if current_date_str in signal_dates:
                daily_signals = signals_data[signals_data['date'].dt.strftime('%Y-%m-%d') == current_date_str]
                self._execute_buys(daily_signals, daily_prices_dict, current_date)
            
            # 3. 计算当日组合价值
            portfolio_value = self._calculate_portfolio_value(daily_prices_dict, current_date)
            
            # 4. 记录组合净值
            self.portfolio_values.append({
                'date': current_date,
                'total_value': portfolio_value,
                'cash': self.current_capital,
                'positions_value': portfolio_value - self.current_capital,
                'positions_count': len(self.positions)
            })
        
        logger.info("回测执行完成！")
        
        # 计算回测结果
        return self._calculate_performance_metrics()
    
    def _execute_sells(self, daily_prices: Dict, current_date: datetime, holding_days: int):
        """执行卖出操作"""
        positions_to_sell = []
        
        for stock_code, position in self.positions.items():
            if stock_code not in daily_prices:
                continue  # 停牌或无数据
            
            current_price = daily_prices[stock_code]['adj_close']
            entry_price = position['entry_price']
            entry_date = position['entry_date']
            
            # 计算持股天数
            days_held = (current_date - entry_date).days
            
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
            elif daily_prices[stock_code].get('is_limit_up', False) or daily_prices[stock_code].get('is_limit_down', False):
                continue  # 涨跌停暂不卖出
            
            if should_sell:
                positions_to_sell.append((stock_code, current_price, sell_reason))
        
        # 执行卖出
        for stock_code, sell_price, reason in positions_to_sell:
            self._execute_sell_order(stock_code, sell_price, current_date, reason)
    
    def _execute_buys(self, signals: pd.DataFrame, daily_prices: Dict, current_date: datetime):
        """执行买入操作"""
        # 按综合评分排序，优先买入高分股票
        signals_sorted = signals.sort_values('comprehensive_score', ascending=False)
        
        for _, signal in signals_sorted.iterrows():
            stock_code = signal['stock_code']
            
            # 检查是否已持有
            if stock_code in self.positions:
                continue
            
            # 检查是否有价格数据
            if stock_code not in daily_prices:
                continue
            
            # 检查涨跌停
            if daily_prices[stock_code].get('is_limit_up', False):
                continue  # 涨停无法买入
            
            # 检查持仓数量限制
            if len(self.positions) >= self.max_positions:
                break
            
            # 使用建议买入价或当前价格
            suggested_price = signal.get('suggested_buy_price', 0)
            current_price = daily_prices[stock_code]['adj_close']
            
            # 如果建议价格合理，使用建议价格，否则使用当前价格
            if suggested_price > 0 and abs(suggested_price - current_price) / current_price < 0.05:
                buy_price = suggested_price
            else:
                buy_price = current_price
            
            # 执行买入
            success = self._execute_buy_order(
                stock_code, buy_price, current_date, 
                signal.get('stop_loss_price'), signal.get('take_profit_price')
            )
            
            if not success:
                break  # 资金不足，停止买入
    
    def _execute_buy_order(self, stock_code: str, price: float, date: datetime, 
                          stop_loss: float = None, take_profit: float = None) -> bool:
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
            'take_profit_price': take_profit
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
            'reason': '选股信号买入'
        })
        
        return True
    
    def _execute_sell_order(self, stock_code: str, price: float, date: datetime, reason: str):
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
            'reason': reason
        })
        
        # 移除持仓
        del self.positions[stock_code]
    
    def _calculate_portfolio_value(self, daily_prices: Dict, date: datetime) -> float:
        """计算组合总价值"""
        total_value = self.current_capital  # 现金部分
        
        # 计算持仓市值
        for stock_code, position in self.positions.items():
            if stock_code in daily_prices:
                current_price = daily_prices[stock_code]['adj_close']
                market_value = position['shares'] * current_price
                total_value += market_value
        
        return total_value
    
    def _calculate_performance_metrics(self) -> Dict:
        """计算回测绩效指标"""
        logger.info("计算回测绩效指标...")
        
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
        
        # 交易统計
        buy_trades = trades_df[trades_df['action'] == 'BUY'] if not trades_df.empty else pd.DataFrame()
        sell_trades = trades_df[trades_df['action'] == 'SELL'] if not trades_df.empty else pd.DataFrame()
        
        total_trades = len(trades_df)
        profitable_trades = len(sell_trades[sell_trades['profit'] > 0]) if not sell_trades.empty else 0
        win_rate = profitable_trades / len(sell_trades) if len(sell_trades) > 0 else 0
        
        # 盈亏比
        avg_profit = sell_trades[sell_trades['profit'] > 0]['profit'].mean() if profitable_trades > 0 else 0
        avg_loss = abs(sell_trades[sell_trades['profit'] < 0]['profit'].mean()) if len(sell_trades[sell_trades['profit'] < 0]) > 0 else 1
        profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 0
        
        # 年化换手率
        total_trade_value = buy_trades['amount'].sum() if not buy_trades.empty else 0
        avg_portfolio_value = portfolio_df['total_value'].mean()
        annual_turnover = (total_trade_value / avg_portfolio_value) * (252 / trading_days) if avg_portfolio_value > 0 and trading_days > 0 else 0
        
        results = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'total_trades': total_trades,
            'annual_turnover': annual_turnover,
            'trading_days': trading_days,
            'final_value': portfolio_df['total_value'].iloc[-1],
            'portfolio_df': portfolio_df,
            'trades_df': trades_df
        }
        
        logger.info("绩效指标计算完成")
        return results
    
    def generate_report(self, results: Dict) -> str:
        """生成专业回测报告"""
        portfolio_df = results['portfolio_df']
        
        report = f"""
# 📊 股票量化策略回测性能报告

## 回测概览
- **回测期间**: {portfolio_df['date'].min().strftime('%Y-%m-%d')} 至 {portfolio_df['date'].max().strftime('%Y-%m-%d')}
- **初始资金**: {self.initial_capital:,.0f}元
- **最终资金**: {results['final_value']:,.0f}元
- **交易日数**: {results['trading_days']}天

## 收益表现
| 指标 | 策略表现 | 行业标准 | 评级 |
|------|----------|----------|------|
| 累计收益率 | {results['total_return']:.2%} | >15% | {'A' if results['total_return'] > 0.15 else 'B' if results['total_return'] > 0.08 else 'C'} |
| 年化收益率 | {results['annual_return']:.2%} | >12% | {'A' if results['annual_return'] > 0.12 else 'B' if results['annual_return'] > 0.08 else 'C'} |
| 交易胜率 | {results['win_rate']:.2%} | >50% | {'A' if results['win_rate'] > 0.5 else 'B' if results['win_rate'] > 0.4 else 'C'} |

## 风险指标
| 指标 | 策略数值 | 行业标准 | 评级 |
|------|----------|----------|------|
| 年化波动率 | {results['volatility']:.2%} | <25% | {'A' if results['volatility'] < 0.25 else 'B' if results['volatility'] < 0.35 else 'C'} |
| 最大回撤 | {results['max_drawdown']:.2%} | <15% | {'A' if results['max_drawdown'] > -0.15 else 'B' if results['max_drawdown'] > -0.25 else 'C'} |
| 夏普比率 | {results['sharpe_ratio']:.2f} | >1.0 | {'A' if results['sharpe_ratio'] > 1.0 else 'B' if results['sharpe_ratio'] > 0.5 else 'C'} |

## 交易统计
- **总交易次数**: {results['total_trades']}次
- **交易胜率**: {results['win_rate']:.2%}
- **盈亏比**: {results['profit_loss_ratio']:.2f}
- **年化换手率**: {results['annual_turnover']:.1%}

## 📈 策略评级
"""
        
        # 计算综合评级
        score = 0
        if results['total_return'] > 0.15: score += 25
        elif results['total_return'] > 0.08: score += 15
        
        if results['max_drawdown'] > -0.15: score += 25
        elif results['max_drawdown'] > -0.25: score += 15
        
        if results['sharpe_ratio'] > 1.0: score += 25
        elif results['sharpe_ratio'] > 0.5: score += 15
        
        if results['win_rate'] > 0.5: score += 25
        elif results['win_rate'] > 0.4: score += 15
        
        if score >= 80:
            grade = "A+ (优秀)"
        elif score >= 60:
            grade = "A (良好)"
        elif score >= 40:
            grade = "B (一般)"
        else:
            grade = "C (需改进)"
        
        report += f"**综合评级**: {grade} (评分: {score}/100)\n\n"
        
        report += """
## ⚠️ 风险提示
- 本回测结果基于历史数据，不代表未来表现
- 实盘交易存在滑点、冲击成本等额外风险
- 建议从小资金开始验证策略有效性
- 市场环境变化可能影响策略表现

## 📊 数据来源
- **价格数据**: 历史股票日线数据（前复权）
- **选股信号**: 基于量化选股报告
- **交易成本**: A股实际成本模型（佣金万三+印花税千一）
- **分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return report
    
    def save_results(self, results: Dict, output_dir: str = "backtest/results"):
        """保存回测结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存组合净值数据
        portfolio_file = output_path / f"portfolio_values_{timestamp}.csv"
        results['portfolio_df'].to_csv(portfolio_file, index=False, encoding='utf-8')
        
        # 保存交易记录
        trades_file = output_path / f"trades_{timestamp}.csv"
        results['trades_df'].to_csv(trades_file, index=False, encoding='utf-8')
        
        # 保存绩效报告
        report = self.generate_report(results)
        report_file = output_path / f"backtest_report_{timestamp}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"回测结果已保存到: {output_path}")
        return output_path

def run_v37_backtest(start_date: str = '2024-01-01',
                     end_date: str = '2024-08-31',
                     initial_capital: float = 1000000,
                     sample_size: int = 100,
                     parallel_workers: int = 4):
    """
    运行V3.7版本的量化策略回测

    Args:
        start_date: 回测开始日期
        end_date: 回测结束日期
        initial_capital: 初始资金
        sample_size: 每日采样股票数量
        parallel_workers: 并行工作进程数
    """
    logger.info("🚀 开始V3.7量化策略回测")
    logger.info(f"回测期间: {start_date} 至 {end_date}")
    logger.info(f"初始资金: {initial_capital:,.0f}元")

    try:
        # 初始化V3.7回测器
        backtester = StockBacktester(initial_capital=initial_capital, scoring_version='v3.7',
                                     sample_size=sample_size, parallel_workers=parallel_workers)

        # 从数据库加载真实股票数据
        logger.info("📊 加载股票数据...")
        stock_data = backtester.load_stock_data(start_date, end_date)

        # 独立的V3.7评分回测：直接采样+评分+买入决策
        logger.info("🎯 执行V3.7独立评分回测...")
        signals_data = backtester.generate_signals_by_date_range(
            start_date, end_date, max_stocks_per_day=20
        )

        if signals_data.empty:
            logger.error("❌ 未生成任何选股信号，无法执行回测")
            return None

        # 执行回测
        logger.info("⚡ 执行回测计算...")
        results = backtester.execute_backtest(
            stock_data=stock_data,
            signals_data=signals_data,
            holding_days=5  # 持股5天
        )

        # 保存结果
        logger.info("💾 保存回测结果...")
        backtester.save_results(results, output_dir="reports/backtest")

        # 输出关键指标
        logger.info("✅ V3.7回测完成！")
        logger.info(f"📈 总收益率: {results.get('total_return', 0):.2%}")
        logger.info(f"📊 夏普比率: {results.get('sharpe_ratio', 0):.2f}")
        logger.info(f"📉 最大回撤: {results.get('max_drawdown', 0):.2%}")
        logger.info(f"🎯 胜率: {results.get('win_rate', 0):.2%}")

        return results

    except Exception as e:
        logger.error(f"❌ V3.7回测执行失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='V3.7量化策略回测')
    parser.add_argument('--start-date', default='2024-01-01', help='开始日期')
    parser.add_argument('--end-date', default='2024-08-31', help='结束日期')
    parser.add_argument('--capital', type=float, default=1000000, help='初始资金')
    parser.add_argument('--sample-size', type=int, default=100, help='每日采样股票数量（默认100）')
    parser.add_argument('--workers', type=int, default=4, help='并行工作进程数（默认4）')

    args = parser.parse_args()

    # 运行V3.7回测
    results = run_v37_backtest(
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.capital,
        sample_size=args.sample_size,
        parallel_workers=args.workers
    )