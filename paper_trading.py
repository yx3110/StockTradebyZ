from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from paper_trading_db import (
    create_or_update_position,
    create_session,
    create_trade,
    get_positions,
    get_session_by_name,
    get_trades,
    init_db,
    update_session_capital,
)

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
