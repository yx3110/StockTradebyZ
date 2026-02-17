"""
Unit tests for paper_trading_db module.

Tests database operations including session management, trade creation,
position tracking, and CRUD operations.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from paper_trading_db import (
    PaperTradingSession,
    Position,
    Trade,
    create_or_update_position,
    create_session,
    create_trade,
    delete_session,
    get_connection,
    get_positions,
    get_session,
    get_session_by_name,
    get_trades,
    init_db,
    list_sessions,
    update_position_price,
    update_session_capital,
)


@pytest.fixture
def test_db():
    """Create a temporary test database."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db_path = Path(path)
    init_db(db_path)
    yield db_path
    # Cleanup
    if db_path.exists():
        os.unlink(db_path)


def test_init_db(test_db):
    """Test database initialization creates all tables."""
    conn = get_connection(test_db)
    cursor = conn.cursor()

    # Check sessions table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'")
    assert cursor.fetchone() is not None

    # Check positions table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='positions'")
    assert cursor.fetchone() is not None

    # Check trades table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
    assert cursor.fetchone() is not None

    conn.close()


def test_create_session(test_db):
    """Test creating a new trading session."""
    session_id = create_session("test_session", 100000.0, test_db)

    assert session_id > 0

    # Verify session was created
    session = get_session(session_id, test_db)
    assert session is not None
    assert session.name == "test_session"
    assert session.initial_capital == 100000.0
    assert session.current_capital == 100000.0
    assert session.status == "active"


def test_create_duplicate_session(test_db):
    """Test that creating duplicate session name raises error."""
    create_session("duplicate_test", 50000.0, test_db)

    # Should raise sqlite3.IntegrityError due to UNIQUE constraint
    with pytest.raises(sqlite3.IntegrityError):
        create_session("duplicate_test", 75000.0, test_db)


def test_get_session(test_db):
    """Test retrieving session by ID."""
    session_id = create_session("get_test", 200000.0, test_db)

    session = get_session(session_id, test_db)

    assert session is not None
    assert session.id == session_id
    assert session.name == "get_test"
    assert session.initial_capital == 200000.0


def test_get_session_nonexistent(test_db):
    """Test getting non-existent session returns None."""
    session = get_session(99999, test_db)
    assert session is None


def test_get_session_by_name(test_db):
    """Test retrieving session by name."""
    create_session("name_test", 150000.0, test_db)

    session = get_session_by_name("name_test", test_db)

    assert session is not None
    assert session.name == "name_test"
    assert session.initial_capital == 150000.0


def test_get_session_by_name_nonexistent(test_db):
    """Test getting session by non-existent name returns None."""
    session = get_session_by_name("does_not_exist", test_db)
    assert session is None


def test_list_sessions(test_db):
    """Test listing all sessions."""
    # Create multiple sessions
    create_session("session1", 100000.0, test_db)
    create_session("session2", 200000.0, test_db)
    create_session("session3", 300000.0, test_db)

    sessions = list_sessions(test_db)

    assert len(sessions) == 3
    session_names = [s.name for s in sessions]
    assert "session1" in session_names
    assert "session2" in session_names
    assert "session3" in session_names


def test_list_sessions_empty(test_db):
    """Test listing sessions when none exist."""
    sessions = list_sessions(test_db)
    assert len(sessions) == 0


def test_update_session_capital(test_db):
    """Test updating session capital."""
    session_id = create_session("capital_test", 100000.0, test_db)

    update_session_capital(session_id, 95000.0, test_db)

    session = get_session(session_id, test_db)
    assert session.current_capital == 95000.0
    assert session.initial_capital == 100000.0  # Should not change


def test_create_trade(test_db):
    """Test creating a trade record."""
    session_id = create_session("trade_session", 100000.0, test_db)

    trade_id = create_trade(
        session_id=session_id,
        symbol="600000",
        action="buy",
        quantity=100,
        price=10.50,
        commission=3.15,
        db_path=test_db
    )

    assert trade_id > 0

    # Verify trade was created
    trades = get_trades(session_id, test_db)
    assert len(trades) == 1
    assert trades[0].symbol == "600000"
    assert trades[0].action == "buy"
    assert trades[0].quantity == 100
    assert trades[0].price == 10.50
    assert trades[0].commission == 3.15


def test_get_trades(test_db):
    """Test retrieving all trades for a session."""
    session_id = create_session("trades_test", 100000.0, test_db)

    # Create multiple trades
    create_trade(session_id, "600000", "buy", 100, 10.0, 0.3, test_db)
    create_trade(session_id, "000001", "buy", 200, 15.0, 0.9, test_db)
    create_trade(session_id, "600000", "sell", 50, 11.0, 0.165, test_db)

    trades = get_trades(session_id, test_db)

    assert len(trades) == 3
    assert trades[0].symbol == "600000"
    assert trades[1].symbol == "000001"
    assert trades[2].action == "sell"


def test_get_trades_empty(test_db):
    """Test getting trades when none exist."""
    session_id = create_session("no_trades", 100000.0, test_db)

    trades = get_trades(session_id, test_db)
    assert len(trades) == 0


def test_create_position(test_db):
    """Test creating a new position."""
    session_id = create_session("position_test", 100000.0, test_db)

    position_id = create_or_update_position(
        session_id=session_id,
        symbol="600000",
        quantity=100,
        entry_price=10.50,
        current_price=10.50,
        db_path=test_db
    )

    assert position_id > 0

    # Verify position was created
    positions = get_positions(session_id, test_db)
    assert len(positions) == 1
    assert positions[0].symbol == "600000"
    assert positions[0].quantity == 100
    assert positions[0].entry_price == 10.50


def test_update_position_add_quantity(test_db):
    """Test updating position by adding more shares."""
    session_id = create_session("update_test", 100000.0, test_db)

    # Create initial position
    create_or_update_position(session_id, "600000", 100, 10.0, 10.0, test_db)

    # Add more shares at different price
    create_or_update_position(session_id, "600000", 100, 12.0, 11.0, test_db)

    positions = get_positions(session_id, test_db)
    assert len(positions) == 1
    assert positions[0].quantity == 200
    # Average entry price: (100*10 + 100*12) / 200 = 11.0
    assert positions[0].entry_price == 11.0


def test_update_position_reduce_quantity(test_db):
    """Test updating position by reducing shares."""
    session_id = create_session("reduce_test", 100000.0, test_db)

    # Create initial position
    create_or_update_position(session_id, "600000", 100, 10.0, 10.0, test_db)

    # Reduce shares
    create_or_update_position(session_id, "600000", -50, 11.0, 11.0, test_db)

    positions = get_positions(session_id, test_db)
    assert len(positions) == 1
    assert positions[0].quantity == 50


def test_get_positions(test_db):
    """Test retrieving all positions for a session."""
    session_id = create_session("positions_test", 100000.0, test_db)

    # Create multiple positions
    create_or_update_position(session_id, "600000", 100, 10.0, 10.5, test_db)
    create_or_update_position(session_id, "000001", 200, 15.0, 15.5, test_db)
    create_or_update_position(session_id, "600519", 50, 100.0, 105.0, test_db)

    positions = get_positions(session_id, test_db)

    assert len(positions) == 3
    symbols = [p.symbol for p in positions]
    assert "600000" in symbols
    assert "000001" in symbols
    assert "600519" in symbols


def test_get_positions_filters_zero_quantity(test_db):
    """Test that positions with zero quantity are filtered out."""
    session_id = create_session("filter_test", 100000.0, test_db)

    # Create position then reduce to zero
    create_or_update_position(session_id, "600000", 100, 10.0, 10.0, test_db)
    create_or_update_position(session_id, "600000", -100, 11.0, 11.0, test_db)

    positions = get_positions(session_id, test_db)
    assert len(positions) == 0  # Should filter out zero quantity


def test_update_position_price(test_db):
    """Test updating position price and P&L."""
    session_id = create_session("price_update_test", 100000.0, test_db)

    position_id = create_or_update_position(
        session_id, "600000", 100, 10.0, 10.0, test_db
    )

    # Update price
    update_position_price(position_id, 12.0, test_db)

    positions = get_positions(session_id, test_db)
    assert len(positions) == 1
    assert positions[0].current_price == 12.0
    # P&L = (12.0 - 10.0) * 100 = 200.0
    assert positions[0].pnl == 200.0


def test_update_position_price_negative_pnl(test_db):
    """Test updating position price with loss."""
    session_id = create_session("loss_test", 100000.0, test_db)

    position_id = create_or_update_position(
        session_id, "600000", 100, 15.0, 15.0, test_db
    )

    # Price drops
    update_position_price(position_id, 12.0, test_db)

    positions = get_positions(session_id, test_db)
    assert positions[0].current_price == 12.0
    # P&L = (12.0 - 15.0) * 100 = -300.0
    assert positions[0].pnl == -300.0


def test_delete_session(test_db):
    """Test deleting a session and all related data."""
    session_id = create_session("delete_test", 100000.0, test_db)

    # Create related data
    create_trade(session_id, "600000", "buy", 100, 10.0, 0.3, test_db)
    create_or_update_position(session_id, "600000", 100, 10.0, 10.0, test_db)

    # Delete session
    delete_session(session_id, test_db)

    # Verify session and related data are deleted
    session = get_session(session_id, test_db)
    assert session is None

    trades = get_trades(session_id, test_db)
    assert len(trades) == 0

    positions = get_positions(session_id, test_db)
    assert len(positions) == 0


def test_paper_trading_session_model():
    """Test PaperTradingSession model."""
    session = PaperTradingSession(
        id=1,
        name="test",
        initial_capital=100000.0,
        current_capital=95000.0,
        status="active"
    )

    assert session.id == 1
    assert session.name == "test"
    assert session.initial_capital == 100000.0
    assert session.current_capital == 95000.0
    assert session.status == "active"

    # Test to_dict
    session_dict = session.to_dict()
    assert session_dict["id"] == 1
    assert session_dict["name"] == "test"
    assert session_dict["initial_capital"] == 100000.0


def test_position_model():
    """Test Position model."""
    position = Position(
        id=1,
        session_id=1,
        symbol="600000",
        quantity=100,
        entry_price=10.0,
        current_price=11.0,
        pnl=100.0
    )

    assert position.id == 1
    assert position.symbol == "600000"
    assert position.quantity == 100
    assert position.pnl == 100.0

    # Test to_dict
    pos_dict = position.to_dict()
    assert pos_dict["symbol"] == "600000"
    assert pos_dict["quantity"] == 100


def test_trade_model():
    """Test Trade model."""
    trade = Trade(
        id=1,
        session_id=1,
        symbol="600000",
        action="buy",
        quantity=100,
        price=10.50,
        commission=3.15
    )

    assert trade.id == 1
    assert trade.symbol == "600000"
    assert trade.action == "buy"
    assert trade.quantity == 100
    assert trade.price == 10.50
    assert trade.commission == 3.15

    # Test to_dict
    trade_dict = trade.to_dict()
    assert trade_dict["symbol"] == "600000"
    assert trade_dict["action"] == "buy"
    assert trade_dict["commission"] == 3.15
