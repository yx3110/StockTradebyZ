from paper_trading import PaperTradingEngine

# Create a test session with initial capital
engine = PaperTradingEngine('test_session', initial_capital=100000)

# Update positions from market data
engine.update_positions_from_market('./data')

print('Positions updated')
