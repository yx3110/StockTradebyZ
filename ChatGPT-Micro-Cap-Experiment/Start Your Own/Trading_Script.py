"""Utilities for maintaining the ChatGPT micro cap portfolio.

The script processes portfolio positions, logs trades, and prints daily
results. It is intentionally lightweight and avoids changing existing
logic or behaviour.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from typing import cast

# Shared file locations
DATA_DIR = Path(__file__).resolve().parent
PORTFOLIO_CSV = DATA_DIR / "chatgpt_portfolio_update.csv"
TRADE_LOG_CSV = DATA_DIR / "chatgpt_trade_log.csv"

# Today's date reused across logs
today = datetime.today().strftime("%Y-%m-%d")


def process_portfolio(portfolio: pd.DataFrame, starting_cash: float) -> tuple[pd.DataFrame, float]:
    """Update daily price information, log stop-loss sells, and prompt for trades.

    The function iterates through each position, retrieves the latest close
    price and appends a summary row. Before processing, the user may record
    one or more manual buys or sells which are then applied to the portfolio.
    Results are appended to ``PORTFOLIO_CSV``.
    """
    results: list[dict[str, object]] = []
    total_value = 0.0
    total_pnl = 0.0
    cash = starting_cash

    while True:
        action = input(
            "Would you like to log a manual trade? Enter 'b' for buy, 's' for sell, or press Enter to continue: "
        ).strip().lower()
        if action == "b":
            try:
                ticker = input("Enter ticker symbol: ").strip().upper()
                shares = float(input("Enter number of shares: "))
                buy_price = float(input("Enter buy price: "))
                stop_loss = float(input("Enter stop loss: "))
                if shares <= 0 or buy_price <= 0 or stop_loss <= 0:
                    raise ValueError
            except ValueError:
                print("Invalid input. Manual buy cancelled.")
            else:
                cash, portfolio = log_manual_buy(
                    buy_price, shares, ticker, stop_loss, cash, portfolio
                )
            continue
        if action == "s":
            try:
                ticker = input("Enter ticker symbol: ").strip().upper()
                shares = float(input("Enter number of shares to sell: "))
                sell_price = float(input("Enter sell price: "))
                if shares <= 0 or sell_price <= 0:
                    raise ValueError
            except ValueError:
                print("Invalid input. Manual sell cancelled.")
            else:
                cash, portfolio = log_manual_sell(
                    sell_price, shares, ticker, cash, portfolio
                )
            continue
        break

    for _, stock in portfolio.iterrows():
        ticker = stock["ticker"]
        shares = int(stock["shares"])
        cost = stock["buy_price"]
        stop = stock["stop_loss"]
        data = yf.Ticker(ticker).history(period="1d")

        if data.empty:
            print(f"No data for {ticker}")
            row = {
                "Date": today,
                "Ticker": ticker,
                "Shares": shares,
                "Cost Basis": cost,
                "Stop Loss": stop,
                "Current Price": "",
                "Total Value": "",
                "PnL": "",
                "Action": "NO DATA",
                "Cash Balance": "",
                "Total Equity": "",
            }
        else:
            price = round(data["Close"].iloc[-1], 2)
            value = round(price * shares, 2)
            pnl = round((price - cost) * shares, 2)

            if price <= stop:
                action = "SELL - Stop Loss Triggered"
                cash += value
                portfolio = log_sell(ticker, shares, price, cost, pnl, portfolio)
            else:
                action = "HOLD"
                total_value += value
                total_pnl += pnl

            row = {
                "Date": today,
                "Ticker": ticker,
                "Shares": shares,
                "Cost Basis": cost,
                "Stop Loss": stop,
                "Current Price": price,
                "Total Value": value,
                "PnL": pnl,
                "Action": action,
                "Cash Balance": "",
                "Total Equity": "",
            }

        results.append(row)

    # Append TOTAL summary row
    total_row = {
        "Date": today,
        "Ticker": "TOTAL",
        "Shares": "",
        "Cost Basis": "",
        "Stop Loss": "",
        "Current Price": "",
        "Total Value": round(total_value, 2),
        "PnL": round(total_pnl, 2),
        "Action": "",
        "Cash Balance": round(cash, 2),
        "Total Equity": round(total_value + cash, 2),
    }
    results.append(total_row)

    df = pd.DataFrame(results)
    if PORTFOLIO_CSV.exists():
        existing = pd.read_csv(PORTFOLIO_CSV)
        existing = existing[existing["Date"] != today]
        print("rows for today already logged, not saving results to CSV...")
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(PORTFOLIO_CSV, index=False)
    return portfolio, cash


def log_sell(
    ticker: str,
    shares: float,
    price: float,
    cost: float,
    pnl: float,
    portfolio: pd.DataFrame,
) -> pd.DataFrame:
    """Record a stop-loss sale in ``TRADE_LOG_CSV`` and remove the ticker."""
    log = {
        "Date": today,
        "Ticker": ticker,
        "Shares Sold": shares,
        "Sell Price": price,
        "Cost Basis": cost,
        "PnL": pnl,
        "Reason": "AUTOMATED SELL - STOPLOSS TRIGGERED",
    }

    portfolio = portfolio[portfolio["ticker"] != ticker]

    if TRADE_LOG_CSV.exists():
        df = pd.read_csv(TRADE_LOG_CSV)
        df = pd.concat([df, pd.DataFrame([log])], ignore_index=True)
    else:
        df = pd.DataFrame([log])
    df.to_csv(TRADE_LOG_CSV, index=False)
    return portfolio


def log_manual_buy(
    buy_price: float,
    shares: float,
    ticker: str,
    stoploss: float,
    cash: float,
    chatgpt_portfolio: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    """Log a manual purchase and append to the portfolio."""
    check = input(
        f"You are currently trying to buy {ticker}."
        " If this a mistake enter 1."
    )
    if check == "1":
        raise SystemExit("Please remove this function call.")

    data = yf.download(ticker, period="1d")
    data = cast(pd.DataFrame, data)
    if data.empty:
        SystemExit(f"error, could not find ticker {ticker}")
    if buy_price * shares > cash:
        SystemExit(
            f"error, you have {cash} but are trying to spend {buy_price * shares}. Are you sure you can do this?"
        )
    pnl = 0.0

    log = {
        "Date": today,
        "Ticker": ticker,
        "Shares Bought": shares,
        "Buy Price": buy_price,
        "Cost Basis": buy_price * shares,
        "PnL": pnl,
        "Reason": "MANUAL BUY - New position",
    }

    if TRADE_LOG_CSV.exists():
        df = pd.read_csv(TRADE_LOG_CSV)
        df = pd.concat([df, pd.DataFrame([log])], ignore_index=True)
    else:
        df = pd.DataFrame([log])
    df.to_csv(TRADE_LOG_CSV, index=False)

    new_trade = {
        "ticker": ticker,
        "shares": shares,
        "stop_loss": stoploss,
        "buy_price": buy_price,
        "cost_basis": buy_price * shares,
    }
    chatgpt_portfolio = pd.concat(
        [chatgpt_portfolio, pd.DataFrame([new_trade])], ignore_index=True
    )
    cash = cash - shares * buy_price
    print(f"Manual buy for {ticker} complete!")
    return cash, chatgpt_portfolio


def log_manual_sell(
    sell_price: float,
    shares_sold: float,
    ticker: str,
    cash: float,
    chatgpt_portfolio: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    """Log a manual sale and update the portfolio."""
    reason = input(
        f"You are currently trying to sell {ticker}.\nIf this is a mistake, enter 1. "
    )

    if reason == "1":
        raise SystemExit("Delete this function call from the program.")

    if isinstance(chatgpt_portfolio, list):
        chatgpt_portfolio = pd.DataFrame(chatgpt_portfolio)
    if ticker not in chatgpt_portfolio["ticker"].values:
        raise KeyError(f"error, could not find {ticker} in portfolio")
    ticker_row = chatgpt_portfolio[chatgpt_portfolio["ticker"] == ticker]

    total_shares = int(ticker_row["shares"].item())
    print(total_shares)
    if shares_sold > total_shares:
        raise ValueError(
            f"You are trying to sell {shares_sold} but only own {total_shares}."
        )
    buy_price = float(ticker_row["buy_price"].item())

    cost_basis = buy_price * shares_sold
    pnl = sell_price * shares_sold - cost_basis
    log = {
        "Date": today,
        "Ticker": ticker,
        "Shares Bought": "",
        "Buy Price": "",
        "Cost Basis": cost_basis,
        "PnL": pnl,
        "Reason": f"MANUAL SELL - {reason}",
        "Shares Sold": shares_sold,
        "Sell Price": sell_price,
    }
    if TRADE_LOG_CSV.exists():
        df = pd.read_csv(TRADE_LOG_CSV)
        df = pd.concat([df, pd.DataFrame([log])], ignore_index=True)
    else:
        df = pd.DataFrame([log])
    df.to_csv(TRADE_LOG_CSV, index=False)

    if total_shares == shares_sold:
        chatgpt_portfolio = chatgpt_portfolio[chatgpt_portfolio["ticker"] != ticker]
    else:
        ticker_row["shares"] = total_shares - shares_sold
        ticker_row["cost_basis"] = ticker_row["shares"] * ticker_row["buy_price"]

    cash = cash + shares_sold * sell_price
    print(f"manual sell for {ticker} complete!")
    return cash, chatgpt_portfolio


def daily_results(chatgpt_portfolio: pd.DataFrame, cash: float) -> None:
    """Print daily price updates and performance metrics."""
    if isinstance(chatgpt_portfolio, pd.DataFrame):
        portfolio_dict = chatgpt_portfolio.to_dict(orient="records")
    print(f"prices and updates for {today}")
    for stock in portfolio_dict + [{"ticker": "^RUT"}] + [{"ticker": "IWO"}] + [{"ticker": "XBI"}]:
        ticker = stock["ticker"]
        try:
            data = yf.download(ticker, period="2d", progress=False)
            data = cast(pd.DataFrame, data)
            if data.empty or len(data) < 2:
                print(f"Data for {ticker} was empty or incomplete.")
                continue
            price = float(data["Close"].iloc[-1].item())
            last_price = float(data["Close"].iloc[-2].item())

            percent_change = ((price - last_price) / last_price) * 100
            volume = float(data["Volume"].iloc[-1].item())
        except Exception as e:
            raise Exception(f"Download for {ticker} failed. {e} Try checking internet connection.")
        print(f"{ticker} closing price: {price:.2f}")
        print(f"{ticker} volume for today: ${volume:,}")
        print(f"percent change from the day before: {percent_change:.2f}%")
    chatgpt_df = pd.read_csv(PORTFOLIO_CSV)

    # Filter TOTAL rows and get latest equity
    chatgpt_totals = chatgpt_df[chatgpt_df["Ticker"] == "TOTAL"].copy()
    chatgpt_totals["Date"] = pd.to_datetime(chatgpt_totals["Date"])
    final_date = chatgpt_totals["Date"].max()
    final_value = chatgpt_totals[chatgpt_totals["Date"] == final_date]
    final_equity = float(final_value["Total Equity"].values[0])
    equity_series = chatgpt_totals["Total Equity"].astype(float).reset_index(drop=True)

    # Daily returns
    daily_pct = equity_series.pct_change().dropna()

    total_return = (equity_series.iloc[-1] - equity_series.iloc[0]) / equity_series.iloc[0]

    # Number of total trading days
    n_days = len(chatgpt_totals)

    # Risk-free return over total trading period (assuming 4.5% risk-free rate)
    rf_annual = 0.045
    rf_period = (1 + rf_annual) ** (n_days / 252) - 1

    # Standard deviation of daily returns
    std_daily = daily_pct.std()
    negative_pct = daily_pct[daily_pct < 0]
    negative_std = negative_pct.std()
    # Sharpe Ratio
    sharpe_total = (total_return - rf_period) / (std_daily * np.sqrt(n_days))
    # Sortino Ratio
    sortino_total = (total_return - rf_period) / (negative_std * np.sqrt(n_days))

    # Output
    print(f"Total Sharpe Ratio over {n_days} days: {sharpe_total:.4f}")
    print(f"Total Sortino Ratio over {n_days} days: {sortino_total:.4f}")
    print(f"Latest ChatGPT Equity: ${final_equity:.2f}")
    # Get S&P 500 data
    spx = yf.download("^SPX", start="2025-06-27", end=final_date + pd.Timedelta(days=1), progress=False)
    spx = cast(pd.DataFrame, spx)
    spx = spx.reset_index()

    # Normalize to $100
    initial_price = spx["Close"].iloc[0].item()
    price_now = spx["Close"].iloc[-1].item()
    scaling_factor = 100 / initial_price
    spx_value = price_now * scaling_factor
    print(f"$100 Invested in the S&P 500: ${spx_value:.2f}")
    print(f"today's portfolio: {chatgpt_portfolio}")
    print(f"cash balance: {cash}")

    print(
        "Here are is your update for today. You can make any changes you see fit (if necessary),\n"
        "but you may not use deep research. You do have to ask premissons for any changes, as you have full control.\n"
        "You can however use the Internet and check current prices for potenial buys."
    )


def main() -> None:
    """Example execution using the default portfolio.
        Be sure to fill in starting capital.
        Edit rows with your portfolio
        Note: Cost Basis = Shares X Buying Price"""
    
    starting_capital = 1_000
    cash = starting_capital
    chatgpt_portfolio = [
        {"ticker": "ABEO", "shares": 6, "stop_loss": 4.9, "buy_price": 5.77, "cost_basis": 34.62},
        {"ticker": "IINN", "shares": 14, "stop_loss": 1.1, "buy_price": 1.5, "cost_basis": 21.0},
        {"ticker": "ACTU", "shares": 6, "stop_loss": 4.89, "buy_price": 5.75, "cost_basis": 34.5},
    ]
    chatgpt_portfolio = pd.DataFrame(chatgpt_portfolio)


    process_portfolio(chatgpt_portfolio, cash)
    daily_results(chatgpt_portfolio, cash)


if __name__ == "__main__":
    main()

