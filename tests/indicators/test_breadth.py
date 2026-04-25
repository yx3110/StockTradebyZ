"""B1 breadth signal unit tests."""
import numpy as np
import pandas as pd
import pytest

from indicators.breadth import compute_breadth_signal


def _fake_close_panel(n_dates=80, n_stocks=100, seed=42):
    """Generate synthetic close panel: rows=date, cols=stock_code."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2024-01-01', periods=n_dates, freq='B')
    codes = [f'00000{i}.SZ' for i in range(n_stocks)]
    rets = rng.normal(0, 0.02, size=(n_dates, n_stocks))
    prices = 10.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=codes)


def test_breadth_returns_dataframe_with_required_cols():
    panel = _fake_close_panel()
    out = compute_breadth_signal(panel, ma_short=20, ma_long=60, streak_days=3)
    assert {'pct_above_ma20', 'pct_above_ma60', 'b1_score', 'b1_bull', 'b1_streak'} <= set(out.columns)
    assert len(out) == len(panel)


def test_breadth_bull_when_majority_above_ma():
    """All stocks trending up → pct_above_ma should rise and b1_bull=1 after streak."""
    n = 80
    dates = pd.date_range('2024-01-01', periods=n, freq='B')
    codes = ['A.SZ', 'B.SZ', 'C.SZ']
    prices = np.linspace(10, 20, n).reshape(-1, 1).repeat(3, axis=1)
    panel = pd.DataFrame(prices, index=dates, columns=codes)
    out = compute_breadth_signal(panel, ma_short=20, ma_long=60, streak_days=3)
    assert out['b1_bull'].iloc[-1] == 1


def test_breadth_bear_when_majority_below_ma():
    n = 80
    dates = pd.date_range('2024-01-01', periods=n, freq='B')
    codes = ['A.SZ', 'B.SZ', 'C.SZ']
    prices = np.linspace(20, 10, n).reshape(-1, 1).repeat(3, axis=1)
    panel = pd.DataFrame(prices, index=dates, columns=codes)
    out = compute_breadth_signal(panel, ma_short=20, ma_long=60, streak_days=3)
    assert out['b1_bull'].iloc[-1] == 0


def test_breadth_hysteresis_keeps_prev_in_neutral_zone():
    """When score is in 0.45-0.55 hysteresis band, b1_bull stays at previous value."""
    n = 30
    dates = pd.date_range('2024-01-01', periods=n, freq='B')
    codes = ['A.SZ']
    panel = pd.DataFrame(np.full((n, 1), 10.0), index=dates, columns=codes)
    out = compute_breadth_signal(panel, ma_short=20, ma_long=60, streak_days=3)
    assert out['b1_bull'].iloc[-5:].nunique() == 1


def test_breadth_streak_required_for_flip():
    """Single-day above-band shouldn't flip if previous regime was bear."""
    from indicators.breadth import apply_threshold_with_hysteresis_and_streak
    score = pd.Series([0.30, 0.30, 0.30, 0.60, 0.30, 0.30],
                     index=pd.date_range('2024-01-01', periods=6, freq='B'))
    bull, streak = apply_threshold_with_hysteresis_and_streak(
        score, lo=0.45, hi=0.55, streak_days=3
    )
    # 0.60 spike: 1 day raw_bull=1, streak=1, but persist_n requires 3 → stay bear
    assert bull.iloc[3] == 0


def test_breadth_warmup_window_returns_nan_for_pct():
    """First (ma_long - 1) rows should have NaN in pct/score columns."""
    panel = _fake_close_panel(n_dates=80, n_stocks=10)
    out = compute_breadth_signal(panel, ma_short=20, ma_long=60, streak_days=3)
    # First 19 rows: no MA20 yet
    assert out['pct_above_ma20'].iloc[:19].isna().all()
    # First 59 rows: no MA60 yet
    assert out['pct_above_ma60'].iloc[:59].isna().all()
    # b1_score depends on both, so first 59 rows NaN
    assert out['b1_score'].iloc[:59].isna().all()
    # b1_bull is integer-typed (Int8), so it shouldn't be NaN — should be the initial value (0)
    assert (out['b1_bull'].iloc[:59] == 0).all()
