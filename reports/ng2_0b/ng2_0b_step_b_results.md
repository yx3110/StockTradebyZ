# ng2.0b Step B+ Results

**Setup:**
- Sub-models: ng2.0b-bull (regime-weight bull, ng1.0.1 schema, seed=42) + ng2.0b-bear (regime-weight bear, ng1.0.1 schema, seed=42)
- Regime: v2 multi-beta vote (V11+B1+B2 hard vote 2-of-3, baseline calibration), `market_regime_signals` table
- Pickle: `ng2_0b_bull_seed42_20260425_231859.pkl` (67MB), `ng2_0b_bear_seed42_20260426_002115.pkl` (69MB)
- Reports: 1886 days each (2018-04-02 → 2026-04-24)
- Merged: regime-switch dispatch via `regime_switch_backtest.py --regime-version v2 --regime-table market_regime_signals`

## WF-OOS 2020-2026

| Metric | Production ng106v2 | ng2.0a (v2+ng101 baseline) | **ng2.0b (v2+ng2.0b-bull/bear)** | Δ vs ng2.0a |
|---|---:|---:|---:|---:|
| V5.2 | 80.4% S | 79.3% A+ | **76.1% A+** | -3.2pp |
| Sharpe (10d) | 2.39 | 2.751 | 2.927 | +0.18 |
| MaxDD (10d) | -23.7% | -17.6% | **-31.6%** | **-14.0pp** ❌ |
| Annual gross | 104.3% | 96.2% | 145.1% | +48.9pp |
| Annual net | — | 89.3% | 138.1% | +48.8pp |
| ICIR | 0.89 | 0.7189 | 0.6179 | -0.10 |

## Pre-2020 2018-2019

| Metric | ng2.0a (v2+ng101 baseline) | **ng2.0b** | Δ vs ng2.0a |
|---|---:|---:|---:|
| V5.2 (×0.85) | 37.2% C | **45.9% B** | **+8.7pp** ✓ |
| Sharpe (10d) | -0.436 | -0.23 | +0.21 |
| MaxDD | -33.9% | -31.6% | +2.3pp |
| Annual gross | -10.3% | -5.9% | +4.4pp |
| Annual net | -17.3% | -13.0% | +4.3pp |
| ICIR | 0.0266 | 0.2006 | +0.17 |

## Acceptance gate (per Phase B spec)

| Gate | Target | ng2.0b actual | PASS? |
|---|---|---|---|
| WF-OOS V5.2 ≥ 81% | 81% | 76.1% | ❌ FAIL (-4.9pp) |
| WF-OOS MaxDD ≤ -18% | -18% | -31.6% | ❌ FAIL (-13.6pp) |
| WF-OOS Sharpe ≥ 2.5 | 2.5 | 2.927 | ✅ PASS |
| Pre-2020 net annual ≥ -5% | -5% | -13.0% | ❌ FAIL (-8pp) |

## Verdict: **ABORT** (3 of 4 gates failed)

### Diagnosis

Sample weighting via regime_v2 ×2 amplified raw alpha (WF-OOS annual gross +49pp vs ng2.0a) but **destroyed the MaxDD profile** (-17.6% → -31.6%, the headline benefit user wanted from ng2.0a). The shape is high-vol high-return — opposite of what the user prioritizes ("回撤控制上的提升是很好的").

**Pre-2020 IS modestly improved** (+8.7pp V5.2, +0.17 ICIR — the only place sample weighting helped meaningfully) but absolute level still fails gate (-13.0% annual net vs -5% required).

### Likely root cause

Bull-mode ×2 weights bull-regime samples (regime_v2=+1) which are 40.6% of WF-OOS data (620/1526 days). Combined with parent class's existing `bear×2.0` (market_return_20d<-5%) the resulting weight distribution is unstable — model overweights the small set of high-vol bull-regime days, learning a high-vol high-return strategy that breaks under regime transitions (hence the -31.6% MaxDD blow-out).

### Recommendation

- **Stay with ng2.0a baseline (Phase A) as the production candidate** — the -17.6% MaxDD is the genuine improvement.
- Phase B (ng2.0b) is a **shelved finding**, not a regression for production.
- If pursued further: try a milder weight (×1.5 instead of ×2.0) and disable parent class's hardcoded `bear×2.0` to avoid double-counting. Out of current Phase B scope.
- The ng2.0b sub-model pickles + reports remain on disk in case future iteration wants to revisit.
