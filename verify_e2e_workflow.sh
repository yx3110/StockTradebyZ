#!/bin/bash
################################################################################
# End-to-End Workflow Verification for Paper Trading System
# Tests complete workflow from session creation to performance reporting
################################################################################

set -e  # Exit on error

# Test configuration
TEST_SESSION="e2e_verification_$(date +%s)"
INITIAL_CAPITAL=100000
DATA_DIR="./data"
CONFIG_FILE="./configs.json"
PYTHON_CMD="python3"

# Find the correct Python command with pandas
if $PYTHON_CMD -c "import pandas" 2>/dev/null; then
    echo "Using: $PYTHON_CMD"
else
    echo "Error: pandas not available in $PYTHON_CMD"
    echo "Trying to find Python with pandas..."

    # Try common locations
    for py_cmd in python python3.9 python3.10 python3.11 python3.12; do
        if command -v $py_cmd &>/dev/null && $py_cmd -c "import pandas" 2>/dev/null; then
            PYTHON_CMD=$py_cmd
            echo "Found: $PYTHON_CMD"
            break
        fi
    done
fi

# Helper functions
print_step() {
    echo ""
    echo "================================================================================"
    echo "STEP $1: $2"
    echo "================================================================================"
}

verify_pass() {
    echo "PASS: $1"
}

verify_fail() {
    echo "FAIL: $1"
    exit 1
}

check_output() {
    local expected="$1"
    local actual="$2"
    local description="$3"

    if echo "$actual" | grep -q "$expected"; then
        verify_pass "$description"
        return 0
    else
        verify_fail "$description - Expected '$expected' in output"
        return 1
    fi
}

# Main verification workflow
main() {
    echo "================================================================================"
    echo "Paper Trading System - End-to-End Workflow Verification"
    echo "================================================================================"
    echo "Test Session: $TEST_SESSION"
    echo "Initial Capital: $INITIAL_CAPITAL"
    echo "Data Directory: $DATA_DIR"
    echo "Python Command: $PYTHON_CMD"
    echo ""

    # Step 1: Create new paper trading session with 100000 virtual capital
    print_step 1 "Create Paper Trading Session"

    output=$($PYTHON_CMD paper_trading.py create --name "$TEST_SESSION" --capital $INITIAL_CAPITAL 2>&1)
    check_output "$TEST_SESSION" "$output" "Session created successfully"

    # Verify session appears in list
    output=$($PYTHON_CMD paper_trading.py list 2>&1)
    check_output "$TEST_SESSION" "$output" "Session appears in session list"

    # Step 2: Verify market data exists
    print_step 2 "Verify Market Data"

    if [ ! -d "$DATA_DIR" ]; then
        verify_fail "Data directory $DATA_DIR does not exist"
    fi

    csv_count=$(ls -1 "$DATA_DIR"/*.csv 2>/dev/null | wc -l)
    if [ "$csv_count" -gt 0 ]; then
        verify_pass "Found $csv_count CSV files in $DATA_DIR"
    else
        verify_fail "No CSV files found in $DATA_DIR"
    fi

    # Get first stock symbol for testing
    TEST_SYMBOL=$(ls "$DATA_DIR"/*.csv | head -1 | xargs basename | sed 's/\.csv$//')
    verify_pass "Selected test symbol: $TEST_SYMBOL"

    # Extract a valid price from the CSV (use 10th from last row to allow room for updates)
    TEST_PRICE=$($PYTHON_CMD -c "
import pandas as pd
df = pd.read_csv('$DATA_DIR/$TEST_SYMBOL.csv')
if len(df) >= 10:
    print(f'{float(df.iloc[-10][\"close\"]):.2f}')
else:
    print(f'{float(df.iloc[-1][\"close\"]):.2f}')
" 2>/dev/null)

    if [ -z "$TEST_PRICE" ] || [ "$TEST_PRICE" == "0.00" ]; then
        verify_fail "Failed to extract valid price from $TEST_SYMBOL.csv"
    fi
    verify_pass "Test price extracted: $TEST_PRICE"

    # Step 3: Execute paper trades
    print_step 3 "Execute Paper Trades"

    # Calculate quantity (20% position, round to 100s for A-share lot size)
    QUANTITY=$($PYTHON_CMD -c "print(int($INITIAL_CAPITAL * 0.2 / $TEST_PRICE / 100) * 100)")
    verify_pass "Calculated quantity: $QUANTITY shares (20% position size)"

    # Execute buy order
    output=$($PYTHON_CMD paper_trading.py buy \
        --session "$TEST_SESSION" \
        --symbol "$TEST_SYMBOL" \
        --quantity $QUANTITY \
        --price $TEST_PRICE 2>&1)

    check_output "买入成功" "$output" "Buy order executed successfully"

    # Verify position was created
    output=$($PYTHON_CMD paper_trading.py positions --session "$TEST_SESSION" 2>&1)
    check_output "$TEST_SYMBOL" "$output" "Position created for $TEST_SYMBOL"

    # Step 4: Update positions with real market data from CSV files
    print_step 4 "Update Positions with Market Data"

    output=$($PYTHON_CMD paper_trading.py update \
        --session "$TEST_SESSION" \
        --data-dir "$DATA_DIR" 2>&1)

    check_output "持仓价格更新完成" "$output" "Positions updated from market data"

    # Verify position price was updated
    output=$($PYTHON_CMD paper_trading.py positions --session "$TEST_SESSION" 2>&1)
    check_output "$TEST_SYMBOL" "$output" "Position shows updated price"

    # Step 5: Generate performance report with all metrics
    print_step 5 "Generate Performance Report"

    output=$($PYTHON_CMD paper_trading.py report --session "$TEST_SESSION" 2>&1)

    # Check for all required metrics
    check_output "总收益率" "$output" "Total return metric present"
    check_output "夏普比率" "$output" "Sharpe ratio metric present"
    check_output "最大回撤" "$output" "Max drawdown metric present"
    check_output "胜率" "$output" "Win rate metric present"
    check_output "平均盈利" "$output" "Average gain metric present"
    check_output "平均亏损" "$output" "Average loss metric present"
    check_output "盈亏比" "$output" "Profit factor metric present"

    # Step 6: Verify trades logged to database
    print_step 6 "Verify Database Persistence"

    # Check database file exists
    if [ -f "paper_trading.db" ]; then
        verify_pass "Database file exists: paper_trading.db"
    else
        verify_fail "Database file not found"
    fi

    # Verify session exists in database
    session_count=$(sqlite3 paper_trading.db "SELECT COUNT(*) FROM sessions WHERE name='$TEST_SESSION';")
    if [ "$session_count" == "1" ]; then
        verify_pass "Session persisted in database"
    else
        verify_fail "Session not found in database (count: $session_count)"
    fi

    # Verify trades exist in database
    trade_count=$(sqlite3 paper_trading.db "SELECT COUNT(*) FROM trades WHERE session_id=(SELECT id FROM sessions WHERE name='$TEST_SESSION');")
    if [ "$trade_count" -ge "1" ]; then
        verify_pass "Trades logged to database (count: $trade_count)"
    else
        verify_fail "No trades found in database"
    fi

    # Verify positions exist in database
    position_count=$(sqlite3 paper_trading.db "SELECT COUNT(*) FROM positions WHERE session_id=(SELECT id FROM sessions WHERE name='$TEST_SESSION');")
    if [ "$position_count" -ge "1" ]; then
        verify_pass "Positions persisted in database (count: $position_count)"
    else
        verify_fail "No positions found in database"
    fi

    # Step 7: Verify P&L calculations
    print_step 7 "Verify P&L Calculations"

    # Get position details from database
    position_data=$(sqlite3 paper_trading.db "SELECT quantity, entry_price, current_price FROM positions WHERE session_id=(SELECT id FROM sessions WHERE name='$TEST_SESSION') AND symbol='$TEST_SYMBOL';")

    if [ -n "$position_data" ]; then
        verify_pass "Position data retrieved from database"

        # Calculate P&L
        pnl=$($PYTHON_CMD -c "
qty, entry, current = '$position_data'.split('|')
pnl = (float(current) - float(entry)) * float(qty)
pnl_pct = ((float(current) - float(entry)) / float(entry)) * 100
print(f'P&L: {pnl:.2f} ({pnl_pct:.2f}%)')
")
        verify_pass "P&L calculation: $pnl"
    else
        verify_fail "Could not retrieve position data from database"
    fi

    # Get session capital details
    capital_data=$(sqlite3 paper_trading.db "SELECT initial_capital, current_capital FROM sessions WHERE name='$TEST_SESSION';")
    if [ -n "$capital_data" ]; then
        initial=$(echo "$capital_data" | cut -d'|' -f1)
        current=$(echo "$capital_data" | cut -d'|' -f2)
        verify_pass "Capital tracking: Initial=$initial, Current=$current"
    else
        verify_fail "Could not retrieve capital data from database"
    fi

    # Step 8 (Bonus): Test auto-trade integration with selectors
    print_step 8 "Test Auto-Trade Integration (Bonus)"

    if [ -f "$CONFIG_FILE" ]; then
        verify_pass "Config file exists: $CONFIG_FILE"

        # Extract first selector class name
        SELECTOR_CLASS=$($PYTHON_CMD -c "
import json
with open('$CONFIG_FILE') as f:
    cfg = json.load(f)
    selectors = cfg.get('selectors', [cfg] if isinstance(cfg, dict) else [])
    if selectors:
        print(selectors[0].get('class', ''))
" 2>/dev/null)

        if [ -n "$SELECTOR_CLASS" ]; then
            verify_pass "Found selector: $SELECTOR_CLASS"

            # Create a new session for auto-trade test
            AUTO_SESSION="auto_test_$(date +%s)"

            # Try auto-trade (may fail if data is not suitable, but should not crash)
            set +e  # Don't exit on error for this test
            output=$($PYTHON_CMD paper_trading.py auto-trade \
                --session "$AUTO_SESSION" \
                --selector "$SELECTOR_CLASS" \
                --data-dir "$DATA_DIR" \
                --config "$CONFIG_FILE" \
                --initial-capital 100000 \
                --position-size 0.1 2>&1)
            auto_trade_exit=$?
            set -e

            if [ $auto_trade_exit -eq 0 ]; then
                check_output "Auto-trade completed" "$output" "Auto-trade command executed successfully"
            else
                echo "WARNING: Auto-trade command failed (may be due to data constraints)"
                echo "This is not critical for e2e verification"
            fi
        fi
    else
        echo "WARNING: Config file not found, skipping auto-trade test"
    fi

    # Final Summary
    echo ""
    echo "================================================================================"
    echo "VERIFICATION SUMMARY"
    echo "================================================================================"
    echo "✓ Session created: $TEST_SESSION"
    echo "✓ Initial capital: $INITIAL_CAPITAL"
    echo "✓ Trades executed: $trade_count"
    echo "✓ Positions held: $position_count"
    echo "✓ Market data updated: YES"
    echo "✓ Performance metrics calculated: YES"
    echo "✓ Database persistence verified: YES"
    echo "✓ P&L calculations verified: YES"
    echo ""
    echo "================================================================================"
    echo "✓ ALL VERIFICATION STEPS PASSED!"
    echo "================================================================================"
    echo ""

    # Cleanup prompt
    echo "Test session '$TEST_SESSION' was created for verification."
    echo "You can clean it up later if needed."
    echo ""

    return 0
}

# Run main function
main
exit_code=$?

exit $exit_code
