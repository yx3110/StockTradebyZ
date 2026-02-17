from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# 数据库路径
DB_PATH = Path("paper_trading.db")


class PaperTradingSession:
    """纸上交易会话模型"""

    def __init__(
        self,
        id: Optional[int] = None,
        name: str = "",
        initial_capital: float = 0.0,
        current_capital: float = 0.0,
        created_at: Optional[str] = None,
        status: str = "active"
    ):
        self.id = id
        self.name = name
        self.initial_capital = initial_capital
        self.current_capital = current_capital
        self.created_at = created_at or datetime.now().isoformat()
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "created_at": self.created_at,
            "status": self.status
        }


class Position:
    """持仓模型"""

    def __init__(
        self,
        id: Optional[int] = None,
        session_id: Optional[int] = None,
        symbol: str = "",
        quantity: float = 0.0,
        entry_price: float = 0.0,
        current_price: float = 0.0,
        pnl: float = 0.0,
        entry_date: Optional[str] = None
    ):
        self.id = id
        self.session_id = session_id
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price
        self.current_price = current_price
        self.pnl = pnl
        self.entry_date = entry_date or datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "symbol": self.symbol,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "pnl": self.pnl,
            "entry_date": self.entry_date
        }


class Trade:
    """交易记录模型"""

    def __init__(
        self,
        id: Optional[int] = None,
        session_id: Optional[int] = None,
        symbol: str = "",
        action: str = "",  # 'buy' or 'sell'
        quantity: float = 0.0,
        price: float = 0.0,
        timestamp: Optional[str] = None,
        commission: float = 0.0
    ):
        self.id = id
        self.session_id = session_id
        self.symbol = symbol
        self.action = action
        self.quantity = quantity
        self.price = price
        self.timestamp = timestamp or datetime.now().isoformat()
        self.commission = commission

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "symbol": self.symbol,
            "action": self.action,
            "quantity": self.quantity,
            "price": self.price,
            "timestamp": self.timestamp,
            "commission": self.commission
        }


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """获取数据库连接"""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    """初始化数据库，创建所有必要的表"""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # 创建会话表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            initial_capital REAL NOT NULL,
            current_capital REAL NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active'
        )
    """)

    # 创建持仓表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            quantity REAL NOT NULL,
            entry_price REAL NOT NULL,
            current_price REAL NOT NULL DEFAULT 0.0,
            pnl REAL NOT NULL DEFAULT 0.0,
            entry_date TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions (id),
            UNIQUE (session_id, symbol)
        )
    """)

    # 创建交易表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            quantity REAL NOT NULL,
            price REAL NOT NULL,
            timestamp TEXT NOT NULL,
            commission REAL NOT NULL DEFAULT 0.0,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        )
    """)

    conn.commit()
    conn.close()


def create_session(name: str, initial_capital: float, db_path: Path = DB_PATH) -> int:
    """创建新的交易会话"""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    now = datetime.now().isoformat()
    cursor.execute(
        """
        INSERT INTO sessions (name, initial_capital, current_capital, created_at, status)
        VALUES (?, ?, ?, ?, ?)
        """,
        (name, initial_capital, initial_capital, now, "active")
    )

    session_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return session_id


def get_session(session_id: int, db_path: Path = DB_PATH) -> Optional[PaperTradingSession]:
    """获取指定ID的会话"""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return PaperTradingSession(
            id=row["id"],
            name=row["name"],
            initial_capital=row["initial_capital"],
            current_capital=row["current_capital"],
            created_at=row["created_at"],
            status=row["status"]
        )
    return None


def get_session_by_name(name: str, db_path: Path = DB_PATH) -> Optional[PaperTradingSession]:
    """通过名称获取会话"""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM sessions WHERE name = ?", (name,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return PaperTradingSession(
            id=row["id"],
            name=row["name"],
            initial_capital=row["initial_capital"],
            current_capital=row["current_capital"],
            created_at=row["created_at"],
            status=row["status"]
        )
    return None


def list_sessions(db_path: Path = DB_PATH) -> List[PaperTradingSession]:
    """列出所有会话"""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM sessions ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()

    return [
        PaperTradingSession(
            id=row["id"],
            name=row["name"],
            initial_capital=row["initial_capital"],
            current_capital=row["current_capital"],
            created_at=row["created_at"],
            status=row["status"]
        )
        for row in rows
    ]


def update_session_capital(session_id: int, new_capital: float, db_path: Path = DB_PATH) -> None:
    """更新会话的当前资金"""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE sessions SET current_capital = ? WHERE id = ?",
        (new_capital, session_id)
    )

    conn.commit()
    conn.close()


def create_trade(
    session_id: int,
    symbol: str,
    action: str,
    quantity: float,
    price: float,
    commission: float = 0.0,
    db_path: Path = DB_PATH
) -> int:
    """创建新交易记录"""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    now = datetime.now().isoformat()
    cursor.execute(
        """
        INSERT INTO trades (session_id, symbol, action, quantity, price, timestamp, commission)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (session_id, symbol, action, quantity, price, now, commission)
    )

    trade_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return trade_id


def get_trades(session_id: int, db_path: Path = DB_PATH) -> List[Trade]:
    """获取指定会话的所有交易记录"""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM trades WHERE session_id = ? ORDER BY timestamp",
        (session_id,)
    )
    rows = cursor.fetchall()
    conn.close()

    return [
        Trade(
            id=row["id"],
            session_id=row["session_id"],
            symbol=row["symbol"],
            action=row["action"],
            quantity=row["quantity"],
            price=row["price"],
            timestamp=row["timestamp"],
            commission=row["commission"]
        )
        for row in rows
    ]


def create_or_update_position(
    session_id: int,
    symbol: str,
    quantity: float,
    entry_price: float,
    current_price: float = 0.0,
    db_path: Path = DB_PATH
) -> int:
    """创建或更新持仓"""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # 检查是否存在持仓
    cursor.execute(
        "SELECT id, quantity, entry_price FROM positions WHERE session_id = ? AND symbol = ?",
        (session_id, symbol)
    )
    row = cursor.fetchone()

    if row:
        # 更新现有持仓
        old_quantity = row["quantity"]
        old_entry_price = row["entry_price"]

        # 计算新的平均入场价
        new_quantity = old_quantity + quantity
        if new_quantity > 0:
            new_entry_price = (old_quantity * old_entry_price + quantity * entry_price) / new_quantity
        else:
            new_entry_price = 0.0

        # 计算盈亏
        pnl = (current_price - new_entry_price) * new_quantity if current_price > 0 else 0.0

        cursor.execute(
            """
            UPDATE positions
            SET quantity = ?, entry_price = ?, current_price = ?, pnl = ?
            WHERE id = ?
            """,
            (new_quantity, new_entry_price, current_price, pnl, row["id"])
        )

        position_id = row["id"]
    else:
        # 创建新持仓
        now = datetime.now().isoformat()
        pnl = (current_price - entry_price) * quantity if current_price > 0 else 0.0

        cursor.execute(
            """
            INSERT INTO positions (session_id, symbol, quantity, entry_price, current_price, pnl, entry_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, symbol, quantity, entry_price, current_price, pnl, now)
        )

        position_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return position_id


def get_positions(session_id: int, db_path: Path = DB_PATH) -> List[Position]:
    """获取指定会话的所有持仓"""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM positions WHERE session_id = ? AND quantity > 0 ORDER BY entry_date",
        (session_id,)
    )
    rows = cursor.fetchall()
    conn.close()

    return [
        Position(
            id=row["id"],
            session_id=row["session_id"],
            symbol=row["symbol"],
            quantity=row["quantity"],
            entry_price=row["entry_price"],
            current_price=row["current_price"],
            pnl=row["pnl"],
            entry_date=row["entry_date"]
        )
        for row in rows
    ]


def update_position_price(position_id: int, current_price: float, db_path: Path = DB_PATH) -> None:
    """更新持仓的当前价格和盈亏"""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # 获取持仓信息
    cursor.execute(
        "SELECT quantity, entry_price FROM positions WHERE id = ?",
        (position_id,)
    )
    row = cursor.fetchone()

    if row:
        quantity = row["quantity"]
        entry_price = row["entry_price"]
        pnl = (current_price - entry_price) * quantity

        cursor.execute(
            "UPDATE positions SET current_price = ?, pnl = ? WHERE id = ?",
            (current_price, pnl, position_id)
        )

    conn.commit()
    conn.close()


def delete_session(session_id: int, db_path: Path = DB_PATH) -> None:
    """删除会话及其所有相关数据"""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # 删除持仓
    cursor.execute("DELETE FROM positions WHERE session_id = ?", (session_id,))

    # 删除交易记录
    cursor.execute("DELETE FROM trades WHERE session_id = ?", (session_id,))

    # 删除会话
    cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    conn.commit()
    conn.close()
