from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from paper_trading_db import (
    create_or_update_position,
    create_session,
    create_trade,
    get_positions,
    get_session,
    get_session_by_name,
    get_trades,
    init_db,
    list_sessions,
    update_position_price,
    update_session_capital,
)
from performance_metrics import calculate_all_metrics

# ---------- 日志 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("paper_trading.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("paper_trading")


# ---------- 常量 ----------
DEFAULT_COMMISSION_RATE = 0.0003  # 0.03% 手续费


class PaperTradingEngine:
    """纸上交易引擎，用于模拟股票买卖"""

    def __init__(
        self,
        session_name: str,
        initial_capital: Optional[float] = None,
        commission_rate: float = DEFAULT_COMMISSION_RATE,
        db_path: Optional[Path] = None,
    ):
        """
        初始化纸上交易引擎

        Parameters
        ----------
        session_name : str
            会话名称
        initial_capital : Optional[float]
            初始资金（仅在创建新会话时需要）
        commission_rate : float
            手续费率，默认 0.03%
        db_path : Optional[Path]
            数据库路径，默认使用 paper_trading.db
        """
        self.session_name = session_name
        self.commission_rate = commission_rate
        self.db_path = db_path

        # 初始化数据库
        init_db(self.db_path) if self.db_path else init_db()

        # 加载或创建会话
        session = get_session_by_name(session_name, self.db_path) if self.db_path else get_session_by_name(session_name)

        if session:
            self.session_id = session.id
            logger.info("加载已存在的会话: %s (ID: %d)", session_name, self.session_id)
        elif initial_capital is not None:
            self.session_id = (
                create_session(session_name, initial_capital, self.db_path)
                if self.db_path
                else create_session(session_name, initial_capital)
            )
            logger.info(
                "创建新会话: %s (ID: %d, 初始资金: %.2f)",
                session_name,
                self.session_id,
                initial_capital,
            )
        else:
            raise ValueError(
                f"会话 '{session_name}' 不存在，且未提供 initial_capital 参数"
            )

    def _get_current_capital(self) -> float:
        """获取当前可用资金"""
        from paper_trading_db import get_session

        session = (
            get_session(self.session_id, self.db_path)
            if self.db_path
            else get_session(self.session_id)
        )
        if not session:
            raise ValueError(f"会话 ID {self.session_id} 不存在")
        return session.current_capital

    def _calculate_commission(self, price: float, quantity: float) -> float:
        """计算手续费"""
        return price * quantity * self.commission_rate

    def buy(self, symbol: str, quantity: float, price: float) -> None:
        """
        买入股票

        Parameters
        ----------
        symbol : str
            股票代码
        quantity : float
            买入数量
        price : float
            买入价格
        """
        if quantity <= 0:
            raise ValueError("买入数量必须大于 0")
        if price <= 0:
            raise ValueError("买入价格必须大于 0")

        # 计算总成本（包含手续费）
        commission = self._calculate_commission(price, quantity)
        total_cost = price * quantity + commission

        # 检查资金是否充足
        current_capital = self._get_current_capital()
        if total_cost > current_capital:
            raise ValueError(
                f"资金不足：需要 {total_cost:.2f}，当前可用 {current_capital:.2f}"
            )

        # 记录交易
        if self.db_path:
            create_trade(
                self.session_id,
                symbol,
                "buy",
                quantity,
                price,
                commission,
                self.db_path,
            )
        else:
            create_trade(
                self.session_id, symbol, "buy", quantity, price, commission
            )

        # 更新持仓
        if self.db_path:
            create_or_update_position(
                self.session_id, symbol, quantity, price, price, self.db_path
            )
        else:
            create_or_update_position(self.session_id, symbol, quantity, price, price)

        # 更新资金
        new_capital = current_capital - total_cost
        if self.db_path:
            update_session_capital(self.session_id, new_capital, self.db_path)
        else:
            update_session_capital(self.session_id, new_capital)

        logger.info(
            "买入 %s: 数量=%d, 价格=%.2f, 手续费=%.2f, 总成本=%.2f",
            symbol,
            quantity,
            price,
            commission,
            total_cost,
        )

    def sell(self, symbol: str, quantity: float, price: float) -> None:
        """
        卖出股票

        Parameters
        ----------
        symbol : str
            股票代码
        quantity : float
            卖出数量
        price : float
            卖出价格
        """
        if quantity <= 0:
            raise ValueError("卖出数量必须大于 0")
        if price <= 0:
            raise ValueError("卖出价格必须大于 0")

        # 检查持仓是否充足
        positions = (
            get_positions(self.session_id, self.db_path)
            if self.db_path
            else get_positions(self.session_id)
        )
        position = next((p for p in positions if p.symbol == symbol), None)

        if not position:
            raise ValueError(f"未持有股票 {symbol}")
        if position.quantity < quantity:
            raise ValueError(
                f"持仓不足：需要卖出 {quantity}，当前持有 {position.quantity}"
            )

        # 计算收入（扣除手续费）
        commission = self._calculate_commission(price, quantity)
        total_revenue = price * quantity - commission

        # 记录交易
        if self.db_path:
            create_trade(
                self.session_id,
                symbol,
                "sell",
                quantity,
                price,
                commission,
                self.db_path,
            )
        else:
            create_trade(
                self.session_id, symbol, "sell", quantity, price, commission
            )

        # 更新持仓（减少数量）
        if self.db_path:
            create_or_update_position(
                self.session_id, symbol, -quantity, price, price, self.db_path
            )
        else:
            create_or_update_position(self.session_id, symbol, -quantity, price, price)

        # 更新资金
        current_capital = self._get_current_capital()
        new_capital = current_capital + total_revenue
        if self.db_path:
            update_session_capital(self.session_id, new_capital, self.db_path)
        else:
            update_session_capital(self.session_id, new_capital)

        logger.info(
            "卖出 %s: 数量=%d, 价格=%.2f, 手续费=%.2f, 总收入=%.2f",
            symbol,
            quantity,
            price,
            commission,
            total_revenue,
        )

    def update_positions_from_market(self, data_dir: str) -> None:
        """
        从市场数据更新持仓的当前价格

        Parameters
        ----------
        data_dir : str
            市场数据目录，包含各股票的 CSV 文件
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            logger.error("数据目录 %s 不存在", data_dir)
            raise ValueError(f"数据目录 {data_dir} 不存在")

        # 获取当前持仓
        positions = (
            get_positions(self.session_id, self.db_path)
            if self.db_path
            else get_positions(self.session_id)
        )

        if not positions:
            logger.info("当前无持仓，无需更新")
            return

        updated_count = 0
        for position in positions:
            symbol = position.symbol
            csv_file = data_path / f"{symbol}.csv"

            if not csv_file.exists():
                logger.warning("%s 的市场数据不存在，跳过", symbol)
                continue

            try:
                # 读取市场数据
                df = pd.read_csv(csv_file, parse_dates=["date"]).sort_values("date")
                if df.empty:
                    logger.warning("%s 市场数据为空，跳过", symbol)
                    continue

                # 获取最新收盘价
                latest_close = df.iloc[-1]["close"]
                if pd.isna(latest_close):
                    logger.warning("%s 最新收盘价缺失，跳过", symbol)
                    continue

                # 更新持仓价格
                if self.db_path:
                    update_position_price(position.id, latest_close, self.db_path)
                else:
                    update_position_price(position.id, latest_close)

                updated_count += 1
                logger.info(
                    "更新 %s 价格: %.2f -> %.2f",
                    symbol,
                    position.current_price,
                    latest_close,
                )

            except Exception as e:
                logger.error("更新 %s 价格失败: %s", symbol, e)
                continue

        logger.info("成功更新 %d 个持仓的价格", updated_count)

    def get_performance_metrics(self, risk_free_rate: float = 0.0) -> dict:
        """
        获取策略性能指标

        Parameters
        ----------
        risk_free_rate : float, default 0.0
            无风险利率（年化）

        Returns
        -------
        dict
            包含所有性能指标的字典：
            - total_return: 总收益率
            - sharpe_ratio: 年化夏普比率
            - max_drawdown: 最大回撤
            - win_rate: 胜率
            - avg_gain: 平均盈利
            - avg_loss: 平均亏损
            - profit_factor: 盈亏比
        """
        # 获取会话信息
        session = (
            get_session(self.session_id, self.db_path)
            if self.db_path
            else get_session(self.session_id)
        )
        if not session:
            raise ValueError(f"会话 ID {self.session_id} 不存在")

        # 获取所有交易记录
        trades = (
            get_trades(self.session_id, self.db_path)
            if self.db_path
            else get_trades(self.session_id)
        )

        # 如果没有交易记录，返回零值指标
        if not trades:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'avg_gain': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
            }

        # 构建权益曲线（账户净值序列）
        equity_data = []
        current_capital = session.initial_capital
        equity_data.append({
            'timestamp': session.created_at,
            'equity': session.initial_capital
        })

        # 按时间顺序计算每笔交易后的账户净值
        for trade in trades:
            if trade.action == 'buy':
                # 买入：扣除成本和手续费
                current_capital -= (trade.price * trade.quantity + trade.commission)
            else:  # sell
                # 卖出：增加收入（已扣除手续费）
                current_capital += (trade.price * trade.quantity - trade.commission)

            equity_data.append({
                'timestamp': trade.timestamp,
                'equity': current_capital
            })

        # 转换为 pandas Series
        equity_df = pd.DataFrame(equity_data)
        equity_curve = pd.Series(
            equity_df['equity'].values,
            index=pd.to_datetime(equity_df['timestamp'])
        )

        # 计算交易收益率（配对买卖计算盈亏）
        trade_returns_list = []
        positions_tracker = {}  # {symbol: [(buy_price, quantity, commission), ...]}

        for trade in trades:
            symbol = trade.symbol

            if trade.action == 'buy':
                # 记录买入
                if symbol not in positions_tracker:
                    positions_tracker[symbol] = []
                positions_tracker[symbol].append({
                    'price': trade.price,
                    'quantity': trade.quantity,
                    'commission': trade.commission
                })
            else:  # sell
                # 计算卖出收益
                if symbol not in positions_tracker or not positions_tracker[symbol]:
                    continue

                sell_quantity = trade.quantity
                sell_revenue = trade.price * trade.quantity - trade.commission
                total_cost = 0.0

                # FIFO（先进先出）匹配买入订单
                while sell_quantity > 0 and positions_tracker[symbol]:
                    buy_order = positions_tracker[symbol][0]
                    match_quantity = min(sell_quantity, buy_order['quantity'])

                    # 计算这部分的成本
                    cost = buy_order['price'] * match_quantity
                    commission_portion = buy_order['commission'] * (match_quantity / buy_order['quantity'])
                    total_cost += (cost + commission_portion)

                    # 更新买入订单数量
                    buy_order['quantity'] -= match_quantity
                    buy_order['commission'] -= commission_portion

                    if buy_order['quantity'] <= 0:
                        positions_tracker[symbol].pop(0)

                    sell_quantity -= match_quantity

                # 计算收益率（基于成本）
                if total_cost > 0:
                    trade_return = (sell_revenue - total_cost) / total_cost
                    trade_returns_list.append(trade_return)

        # 转换为 pandas Series
        trade_returns = pd.Series(trade_returns_list) if trade_returns_list else None

        # 使用 performance_metrics 模块计算所有指标
        metrics = calculate_all_metrics(
            equity_curve=equity_curve,
            trade_returns=trade_returns,
            risk_free_rate=risk_free_rate
        )

        return metrics


# ---------- CLI 主函数 ----------

def main():
    """命令行接口主函数"""
    parser = argparse.ArgumentParser(description="Paper Trading CLI - 纸上交易模拟系统")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # create 命令 - 创建新会话
    create_parser = subparsers.add_parser("create", help="创建新的交易会话")
    create_parser.add_argument("--name", required=True, help="会话名称")
    create_parser.add_argument("--capital", type=float, required=True, help="初始资金")
    create_parser.add_argument("--db", type=Path, default=None, help="数据库路径（可选）")

    # list 命令 - 列出所有会话
    list_parser = subparsers.add_parser("list", help="列出所有交易会话")
    list_parser.add_argument("--db", type=Path, default=None, help="数据库路径（可选）")

    # buy 命令 - 买入股票
    buy_parser = subparsers.add_parser("buy", help="买入股票")
    buy_parser.add_argument("--session", required=True, help="会话名称")
    buy_parser.add_argument("--symbol", required=True, help="股票代码")
    buy_parser.add_argument("--quantity", type=float, required=True, help="买入数量")
    buy_parser.add_argument("--price", type=float, required=True, help="买入价格")
    buy_parser.add_argument("--db", type=Path, default=None, help="数据库路径（可选）")

    # sell 命令 - 卖出股票
    sell_parser = subparsers.add_parser("sell", help="卖出股票")
    sell_parser.add_argument("--session", required=True, help="会话名称")
    sell_parser.add_argument("--symbol", required=True, help="股票代码")
    sell_parser.add_argument("--quantity", type=float, required=True, help="卖出数量")
    sell_parser.add_argument("--price", type=float, required=True, help="卖出价格")
    sell_parser.add_argument("--db", type=Path, default=None, help="数据库路径（可选）")

    # positions 命令 - 查看持仓
    positions_parser = subparsers.add_parser("positions", help="查看持仓")
    positions_parser.add_argument("--session", required=True, help="会话名称")
    positions_parser.add_argument("--db", type=Path, default=None, help="数据库路径（可选）")

    # 解析参数
    args = parser.parse_args()

    # 如果没有提供命令，显示帮助信息
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # 初始化数据库
    db_path = args.db if hasattr(args, 'db') and args.db else None
    if db_path:
        init_db(db_path)
    else:
        init_db()

    # 执行对应命令
    if args.command == "create":
        # 创建新会话
        try:
            if db_path:
                session_id = create_session(args.name, args.capital, db_path)
            else:
                session_id = create_session(args.name, args.capital)
            logger.info("创建会话成功: %s (ID: %d, 初始资金: %.2f)", args.name, session_id, args.capital)
            print(args.name)
        except Exception as e:
            logger.error("创建会话失败: %s", e)
            sys.exit(1)

    elif args.command == "list":
        # 列出所有会话
        try:
            if db_path:
                sessions = list_sessions(db_path)
            else:
                sessions = list_sessions()

            if not sessions:
                logger.info("当前无交易会话")
            else:
                logger.info("")
                logger.info("============== 交易会话列表 ==============")
                for session in sessions:
                    print(session.name)
                    logger.info(
                        "会话: %s | 初始资金: %.2f | 当前资金: %.2f | 状态: %s | 创建时间: %s",
                        session.name,
                        session.initial_capital,
                        session.current_capital,
                        session.status,
                        session.created_at
                    )
        except Exception as e:
            logger.error("列出会话失败: %s", e)
            sys.exit(1)

    elif args.command == "buy":
        # 买入股票
        try:
            # 创建或加载引擎（如果会话不存在则需要先创建）
            engine = PaperTradingEngine(
                session_name=args.session,
                initial_capital=100000.0,  # 默认初始资金（如果会话已存在则忽略）
                db_path=db_path
            )
            engine.buy(args.symbol, args.quantity, args.price)
            logger.info("买入成功")
        except Exception as e:
            logger.error("买入失败: %s", e)
            sys.exit(1)

    elif args.command == "sell":
        # 卖出股票
        try:
            engine = PaperTradingEngine(
                session_name=args.session,
                db_path=db_path
            )
            engine.sell(args.symbol, args.quantity, args.price)
            logger.info("卖出成功")
        except Exception as e:
            logger.error("卖出失败: %s", e)
            sys.exit(1)

    elif args.command == "positions":
        # 查看持仓
        try:
            # 获取会话
            if db_path:
                session = get_session_by_name(args.session, db_path)
            else:
                session = get_session_by_name(args.session)

            if not session:
                logger.error("会话 '%s' 不存在", args.session)
                sys.exit(1)

            # 获取持仓
            if db_path:
                positions = get_positions(session.id, db_path)
            else:
                positions = get_positions(session.id)

            if not positions:
                logger.info("当前无持仓")
            else:
                logger.info("")
                logger.info("============== 持仓列表 [%s] ==============", args.session)
                for position in positions:
                    print(position.symbol)
                    logger.info(
                        "股票: %s | 数量: %.0f | 成本价: %.2f | 当前价: %.2f | 盈亏: %.2f (%.2f%%)",
                        position.symbol,
                        position.quantity,
                        position.entry_price,
                        position.current_price,
                        (position.current_price - position.entry_price) * position.quantity,
                        ((position.current_price - position.entry_price) / position.entry_price) * 100
                    )
        except Exception as e:
            logger.error("查看持仓失败: %s", e)
            sys.exit(1)


if __name__ == "__main__":
    main()
