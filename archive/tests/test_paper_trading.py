"""
Unit tests for paper_trading module.

Tests the PaperTradingEngine class including buy/sell operations,
commission calculations, capital validation, and performance metrics.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from paper_trading import PaperTradingEngine
from paper_trading_db import get_positions, get_session, get_trades, init_db


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


@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary data directory with sample CSV files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create sample CSV for 600000
    csv_600000 = data_dir / "600000.csv"
    df_600000 = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10),
        'open': [10.0, 10.5, 10.2, 10.8, 11.0, 11.2, 11.5, 11.3, 11.8, 12.0],
        'close': [10.5, 10.2, 10.8, 11.0, 11.2, 11.5, 11.3, 11.8, 12.0, 12.5],
        'high': [10.6, 10.7, 11.0, 11.2, 11.4, 11.7, 11.6, 12.0, 12.2, 12.8],
        'low': [9.9, 10.0, 10.1, 10.5, 10.8, 11.0, 11.1, 11.2, 11.5, 11.8],
        'volume': [1000000] * 10
    })
    df_600000.to_csv(csv_600000, index=False)

    # Create sample CSV for 000001
    csv_000001 = data_dir / "000001.csv"
    df_000001 = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10),
        'open': [15.0, 15.2, 15.5, 15.8, 16.0, 16.2, 16.5, 16.3, 16.8, 17.0],
        'close': [15.2, 15.5, 15.8, 16.0, 16.2, 16.5, 16.3, 16.8, 17.0, 17.5],
        'high': [15.5, 15.8, 16.0, 16.3, 16.5, 16.8, 16.7, 17.0, 17.3, 17.8],
        'low': [14.8, 15.0, 15.2, 15.5, 15.8, 16.0, 16.2, 16.1, 16.5, 16.8],
        'volume': [1200000] * 10
    })
    df_000001.to_csv(csv_000001, index=False)

    return data_dir


def test_engine_initialization_new_session(test_db):
    """Test engine initialization creates new session."""
    engine = PaperTradingEngine(
        session_name="test_new",
        initial_capital=100000.0,
        db_path=test_db
    )

    assert engine.session_id > 0
    assert engine.session_name == "test_new"

    # Verify session in database
    session = get_session(engine.session_id, test_db)
    assert session is not None
    assert session.initial_capital == 100000.0
    assert session.current_capital == 100000.0


def test_engine_initialization_existing_session(test_db):
    """Test engine initialization loads existing session."""
    # Create session first
    engine1 = PaperTradingEngine(
        session_name="existing",
        initial_capital=200000.0,
        db_path=test_db
    )
    session_id = engine1.session_id

    # Load existing session
    engine2 = PaperTradingEngine(
        session_name="existing",
        db_path=test_db
    )

    assert engine2.session_id == session_id


def test_engine_initialization_missing_session_no_capital(test_db):
    """Test initialization fails when session doesn't exist and no capital provided."""
    with pytest.raises(ValueError, match="不存在，且未提供 initial_capital"):
        PaperTradingEngine(
            session_name="nonexistent",
            db_path=test_db
        )


def test_buy_sufficient_capital(test_db):
    """Test buying stock with sufficient capital."""
    engine = PaperTradingEngine(
        session_name="buy_test",
        initial_capital=100000.0,
        db_path=test_db
    )

    engine.buy("600000", 100, 10.50)

    # Verify trade was created
    trades = get_trades(engine.session_id, test_db)
    assert len(trades) == 1
    assert trades[0].symbol == "600000"
    assert trades[0].action == "buy"
    assert trades[0].quantity == 100
    assert trades[0].price == 10.50

    # Verify commission was calculated (0.03% of 100 * 10.50 = 3.15)
    expected_commission = 100 * 10.50 * 0.0003
    assert abs(trades[0].commission - expected_commission) < 0.01

    # Verify position was created
    positions = get_positions(engine.session_id, test_db)
    assert len(positions) == 1
    assert positions[0].symbol == "600000"
    assert positions[0].quantity == 100
    assert positions[0].entry_price == 10.50

    # Verify capital was deducted
    session = get_session(engine.session_id, test_db)
    total_cost = 100 * 10.50 + expected_commission
    assert abs(session.current_capital - (100000.0 - total_cost)) < 0.01


def test_buy_insufficient_capital(test_db):
    """Test buying stock with insufficient capital raises ValueError."""
    engine = PaperTradingEngine(
        session_name="insufficient_test",
        initial_capital=100.0,
        db_path=test_db
    )

    with pytest.raises(ValueError, match="资金不足"):
        engine.buy("600000", 100, 100.0)  # Needs 10,000+


def test_buy_zero_quantity(test_db):
    """Test buying zero quantity raises ValueError."""
    engine = PaperTradingEngine(
        session_name="zero_qty_test",
        initial_capital=100000.0,
        db_path=test_db
    )

    with pytest.raises(ValueError, match="买入数量必须大于 0"):
        engine.buy("600000", 0, 10.0)


def test_buy_negative_quantity(test_db):
    """Test buying negative quantity raises ValueError."""
    engine = PaperTradingEngine(
        session_name="neg_qty_test",
        initial_capital=100000.0,
        db_path=test_db
    )

    with pytest.raises(ValueError, match="买入数量必须大于 0"):
        engine.buy("600000", -100, 10.0)


def test_buy_zero_price(test_db):
    """Test buying at zero price raises ValueError."""
    engine = PaperTradingEngine(
        session_name="zero_price_test",
        initial_capital=100000.0,
        db_path=test_db
    )

    with pytest.raises(ValueError, match="买入价格必须大于 0"):
        engine.buy("600000", 100, 0)


def test_buy_negative_price(test_db):
    """Test buying at negative price raises ValueError."""
    engine = PaperTradingEngine(
        session_name="neg_price_test",
        initial_capital=100000.0,
        db_path=test_db
    )

    with pytest.raises(ValueError, match="买入价格必须大于 0"):
        engine.buy("600000", 100, -10.0)


def test_sell_sufficient_quantity(test_db):
    """Test selling stock with sufficient quantity."""
    engine = PaperTradingEngine(
        session_name="sell_test",
        initial_capital=100000.0,
        db_path=test_db
    )

    # Buy first
    engine.buy("600000", 100, 10.0)

    # Then sell
    engine.sell("600000", 50, 11.0)

    # Verify sell trade was created
    trades = get_trades(engine.session_id, test_db)
    assert len(trades) == 2
    assert trades[1].action == "sell"
    assert trades[1].quantity == 50
    assert trades[1].price == 11.0

    # Verify position quantity was reduced
    positions = get_positions(engine.session_id, test_db)
    assert len(positions) == 1
    assert positions[0].quantity == 50

    # Verify capital increased
    session = get_session(engine.session_id, test_db)
    # Capital should be: initial - buy_cost + sell_revenue
    buy_commission = 100 * 10.0 * 0.0003
    buy_cost = 100 * 10.0 + buy_commission
    sell_commission = 50 * 11.0 * 0.0003
    sell_revenue = 50 * 11.0 - sell_commission
    expected_capital = 100000.0 - buy_cost + sell_revenue
    assert abs(session.current_capital - expected_capital) < 0.01


def test_sell_insufficient_quantity(test_db):
    """Test selling more than held quantity raises ValueError."""
    engine = PaperTradingEngine(
        session_name="insufficient_sell_test",
        initial_capital=100000.0,
        db_path=test_db
    )

    engine.buy("600000", 100, 10.0)

    with pytest.raises(ValueError, match="持仓不足"):
        engine.sell("600000", 150, 11.0)


def test_sell_no_position(test_db):
    """Test selling stock with no position raises ValueError."""
    engine = PaperTradingEngine(
        session_name="no_position_test",
        initial_capital=100000.0,
        db_path=test_db
    )

    with pytest.raises(ValueError, match="未持有股票"):
        engine.sell("600000", 100, 11.0)


def test_commission_calculation_accuracy(test_db):
    """Test commission calculation is accurate."""
    engine = PaperTradingEngine(
        session_name="commission_test",
        initial_capital=100000.0,
        commission_rate=0.0005,  # 0.05%
        db_path=test_db
    )

    engine.buy("600000", 200, 15.0)

    trades = get_trades(engine.session_id, test_db)
    # Commission = 200 * 15.0 * 0.0005 = 1.5
    expected_commission = 200 * 15.0 * 0.0005
    assert abs(trades[0].commission - expected_commission) < 0.01


def test_capital_deduction_accuracy(test_db):
    """Test capital deduction is accurate."""
    engine = PaperTradingEngine(
        session_name="capital_deduction_test",
        initial_capital=100000.0,
        db_path=test_db
    )

    # Buy multiple stocks
    engine.buy("600000", 100, 10.0)
    engine.buy("000001", 200, 15.0)

    session = get_session(engine.session_id, test_db)

    # Calculate expected capital
    commission1 = 100 * 10.0 * 0.0003
    commission2 = 200 * 15.0 * 0.0003
    total_cost = (100 * 10.0 + commission1) + (200 * 15.0 + commission2)
    expected_capital = 100000.0 - total_cost

    assert abs(session.current_capital - expected_capital) < 0.01


def test_update_positions_from_market(test_db, test_data_dir):
    """Test updating positions with valid market data."""
    engine = PaperTradingEngine(
        session_name="update_test",
        initial_capital=100000.0,
        db_path=test_db
    )

    # Create positions
    engine.buy("600000", 100, 10.0)
    engine.buy("000001", 200, 15.0)

    # Update positions from market data
    engine.update_positions_from_market(str(test_data_dir))

    # Verify positions were updated with latest prices
    positions = get_positions(engine.session_id, test_db)
    positions_dict = {p.symbol: p for p in positions}

    # Check 600000 was updated to latest close (12.5)
    assert abs(positions_dict["600000"].current_price - 12.5) < 0.01

    # Check 000001 was updated to latest close (17.5)
    assert abs(positions_dict["000001"].current_price - 17.5) < 0.01


def test_update_positions_missing_data(test_db, test_data_dir):
    """Test updating positions when market data is missing."""
    engine = PaperTradingEngine(
        session_name="missing_data_test",
        initial_capital=100000.0,
        db_path=test_db
    )

    # Create position for stock without data file
    engine.buy("600519", 50, 100.0)

    # Should not raise error, just skip
    engine.update_positions_from_market(str(test_data_dir))

    # Position should remain unchanged
    positions = get_positions(engine.session_id, test_db)
    assert positions[0].current_price == 100.0  # Still at entry price


def test_update_positions_invalid_directory(test_db):
    """Test update positions with invalid directory raises error."""
    engine = PaperTradingEngine(
        session_name="invalid_dir_test",
        initial_capital=100000.0,
        db_path=test_db
    )

    engine.buy("600000", 100, 10.0)

    with pytest.raises(ValueError, match="数据目录.*不存在"):
        engine.update_positions_from_market("/nonexistent/path")


def test_get_performance_metrics_no_trades(test_db):
    """Test performance metrics with no trades."""
    engine = PaperTradingEngine(
        session_name="no_trades_metrics",
        initial_capital=100000.0,
        db_path=test_db
    )

    metrics = engine.get_performance_metrics()

    assert metrics['total_return'] == 0.0
    assert metrics['sharpe_ratio'] == 0.0
    assert metrics['max_drawdown'] == 0.0
    assert metrics['win_rate'] == 0.0
    assert metrics['avg_gain'] == 0.0
    assert metrics['avg_loss'] == 0.0
    assert metrics['profit_factor'] == 0.0


def test_get_performance_metrics_one_trade(test_db):
    """Test performance metrics with one trade."""
    engine = PaperTradingEngine(
        session_name="one_trade_metrics",
        initial_capital=100000.0,
        db_path=test_db
    )

    # Single buy trade
    engine.buy("600000", 100, 10.0)

    metrics = engine.get_performance_metrics()

    # Should have equity curve data but no completed trades
    assert metrics['total_return'] < 0  # Negative due to commission
    assert metrics['win_rate'] == 0.0  # No completed buy-sell pairs


def test_get_performance_metrics_multiple_trades(test_db):
    """Test performance metrics with multiple trades."""
    engine = PaperTradingEngine(
        session_name="multiple_trades_metrics",
        initial_capital=100000.0,
        db_path=test_db
    )

    # Buy and sell for profit
    engine.buy("600000", 100, 10.0)
    engine.sell("600000", 100, 12.0)

    # Buy and sell for loss
    engine.buy("000001", 200, 20.0)
    engine.sell("000001", 200, 18.0)

    metrics = engine.get_performance_metrics()

    # Win rate should be 50% (1 win, 1 loss)
    assert 0.4 < metrics['win_rate'] < 0.6

    # Should have avg_gain and avg_loss
    assert metrics['avg_gain'] > 0
    assert metrics['avg_loss'] < 0

    # Profit factor should be calculable
    assert metrics['profit_factor'] > 0


def test_multiple_buys_average_entry_price(test_db):
    """Test multiple buys calculate correct average entry price."""
    engine = PaperTradingEngine(
        session_name="avg_price_test",
        initial_capital=100000.0,
        db_path=test_db
    )

    # Buy at different prices
    engine.buy("600000", 100, 10.0)
    engine.buy("600000", 100, 12.0)

    positions = get_positions(engine.session_id, test_db)
    assert len(positions) == 1
    assert positions[0].quantity == 200

    # Average price: (100*10 + 100*12) / 200 = 11.0
    assert abs(positions[0].entry_price - 11.0) < 0.01


def test_sell_all_position(test_db):
    """Test selling entire position removes it from active positions."""
    engine = PaperTradingEngine(
        session_name="sell_all_test",
        initial_capital=100000.0,
        db_path=test_db
    )

    engine.buy("600000", 100, 10.0)
    engine.sell("600000", 100, 11.0)

    # Position should be removed (quantity = 0)
    positions = get_positions(engine.session_id, test_db)
    assert len(positions) == 0


def test_custom_commission_rate(test_db):
    """Test engine with custom commission rate."""
    engine = PaperTradingEngine(
        session_name="custom_rate_test",
        initial_capital=100000.0,
        commission_rate=0.001,  # 0.1%
        db_path=test_db
    )

    engine.buy("600000", 100, 10.0)

    trades = get_trades(engine.session_id, test_db)
    # Commission = 100 * 10.0 * 0.001 = 1.0
    expected_commission = 100 * 10.0 * 0.001
    assert abs(trades[0].commission - expected_commission) < 0.01
