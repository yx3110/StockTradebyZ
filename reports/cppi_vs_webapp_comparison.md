# CPPI vs Webapp Advanced Risk Management Comparison

**Date**: 2026-04-05 23:52
**Reports**: ng1.0.2 | Top-10 | Hold 10d | Cost 0.15%/leg
**Period**: 20180402 to 20260403 | Rebalances: 195

## Results

| Metric                         |        Simple CPPI |    Webapp Advanced |
|--------------------------------|--------------------|--------------------|
| Annual Return (%)              |              -0.88 |              -0.65 |
| Total Return (%)               |              -6.86 |              -5.05 |
| Max Drawdown (%)               |               6.82 |               4.87 |
| Sharpe Ratio                   |             -0.659 |             -0.759 |
| Win Rate (%)                   |               0.00 |               0.52 |
| CVaR 5% (%)                    |              -0.07 |              -0.08 |
| Avg Exposure (%)               |               0.51 |               0.82 |
| Active Rebalances              |                  1 |                  5 |
| ATR Stop Triggers              |                  0 |                 26 |
| Annual Turnover                |                2.5 |               12.5 |
| Number of Trades               |                 20 |                100 |
| Longest DD Period (days)       |               1939 |               1938 |
| Final NAV                      |            931,387 |            949,480 |

## Winner Summary

- **Best Annual Return**: Webapp Advanced
- **Best Sharpe Ratio**: Simple CPPI
- **Lowest Max Drawdown**: Webapp Advanced

## Strategy Descriptions

### Strategy A: Simple CPPI (EastMoneyTrader)
- Floor=5%, Multiplier=20
- Equal weight across Top-10
- No stops, no regime adaptation

### Strategy B: Webapp Advanced Risk Management
- **Market Regime** (CSI300 20d return): Bull >3% = 85% cap, Neutral = 70%, Bear <-3% = 40%
- **Circuit Breaker**: DD >15% -> 0.4x exposure, DD >10% -> 0.8x
- **ATR Stop-Loss**: Exit if price < entry - 2.5 * ATR_14
- **Risk-Parity Weighting**: Inverse 20d volatility (lower vol = higher weight)
- **Sector Limit**: Max 25% in any single industry
- **CPPI Base**: Floor=5%, Multiplier=20, capped by regime exposure

## Diagnostic Info

- **Regime distribution**: Bull=45, Neutral=104, Bear=46
- **Last active date**: A=2018-04-03, B=2018-06-04
- **ATR stop triggers**: 26

### CPPI Death Spiral Note

With floor=5% and multiplier=20, the CPPI formula gives 100% exposure when NAV=peak,
but 0% exposure once NAV drops below 95% of peak. Since peak_nav only increases,
a single bad period (>5% loss) permanently locks exposure at 0%.
Strategy A hit this after the first 10-day period; Strategy B survived longer
because the regime cap limited initial exposure, but eventually also locked out.
