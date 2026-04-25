# ng2.0a Step B Option C: ng1.0.1 bull sub-model variant (2026-04-25)

**Question:** Does v2 regime beat v1 regime when paired with ng1.0.1 bull (ng106 v1 architecture)?

**Context:** Original Step B used ng1.0.7 as bull sub-model. Here we swap in ng1.0.1 (stronger picker per MEMORY).
If v2 still beats v1 with ng1.0.1, regime signal is sound. If v2 still underperforms, regime itself is the issue.

## WF-OOS 2020-2026

| Metric | v1+ng101 baseline | v2+ng101 ng2.0a | Δ |
|---|---:|---:|---:|
| V5.2 score (10d) | 76.5% (A+级) | 79.3% (A+级) | **+2.8 pp** |
| Sharpe (10d) | 2.580 | 2.751 | **+0.171** |
| MaxDD (10d) | -26.8% | -17.6% | **+9.2 pp** |
| Annual return gross (10d) | 113.9% | 96.2% | -17.7 pp |
| Annual return net (10d) | 107.1% | 89.3% | -17.8 pp |
| Excess annual (10d) | 117.1% | 100.8% | -16.3 pp |
| ICIR (10d) | 0.742 | 0.719 | -0.023 |
| Bull/bear day split | 511/1002 | 620/893 | +109 bull |

**Key observation:** v2 improves V5.2 (+2.8 pp), Sharpe (+0.17), and MaxDD (+9.2 pp) at the cost of lower raw annual return
(-17.7 pp gross). This tradeoff occurs because v2 adds +109 bull days (switching from ng104 bear to ng101 bull on those
days), and ng101 is a weaker bear-market picker than ng104-3s. V5.2 north-star rewards the risk-adjusted profile.

## Pre-2020 2018-2019

| Metric | v1+ng101 baseline | v2+ng101 ng2.0a | Δ |
|---|---:|---:|---:|
| V5.2 score (×0.85) | 48.2% × 0.85 = 40.7% (C级) | 44.1% × 0.85 = 37.2% (C级) | -3.5 pp |
| Sharpe (10d) | 0.034 | -0.436 | -0.470 |
| MaxDD (10d) | -27.9% | -33.9% | -6.0 pp |
| Annual return gross (10d) | +3.0% | -10.3% | -13.3 pp |
| Annual return net (10d) | -4.0% | -17.3% | -13.3 pp |
| Excess annual (10d) | +22.1% | +7.8% | -14.3 pp |
| Bull/bear day split | 71/289 | 104/256 | +33 bull |

**Key observation:** v2 is decisively worse on Pre-2020 with ng1.0.1 bull. Adding +33 bull days (switching bear-market
2018-2019 days to ng1.0.1) hurts: ng1.0.1 misfires in true bear conditions. v1 ekes out a near-flat gross return (+3.0%)
while v2 loses -10.3%.

## Verdict

**v2 regime improves WF-OOS risk-adjusted metrics (V5.2 +2.8pp, Sharpe +0.17, MaxDD -9.2pp) but hurts Pre-2020.**

This is the same failure pattern as original Step B (ng1.0.7 bull): v2 consistently over-calls bull in 2018-2019
(a bear-dominant period), leading to Pre-2020 degradation regardless of which bull sub-model is used.

**Conclusion:** The regime signal itself is the issue for Pre-2020, not the sub-model choice. v2 has a structural
tendency to call more bull days (+33 in pre2020, +109 in WF-OOS) because its multi-beta vote is more sensitive to
positive breadth/momentum signals that occur even during bear years. This helps in WF-OOS (where both bull and bear
years appear and the additional bull days land in actual bull segments) but hurts in a pure bear window (2018-2019).

**Decision gate:** v2 regime is NOT ready for production — Pre-2020 is a clear regression (C级 vs C级 but -3.5 pp, and
v1 is already barely passing at C级 40.7%). The Pre-2020 degradation is regime-structural, not sub-model-fixable.

## Comparison vs original Step B (ng1.0.7 bull)

| Metric | v1+ng107 | v2+ng107 | v1+ng101 (optC) | v2+ng101 (optC) |
|---|---:|---:|---:|---:|
| WF-OOS V5.2 | 80.4% S | 80.2% S | 76.5% A+ | 79.3% A+ |
| WF-OOS Sharpe | 2.386 | 2.587 | 2.580 | 2.751 |
| WF-OOS MaxDD | -23.7% | -22.3% | -26.8% | -17.6% |
| WF-OOS ICIR | 0.891 | 0.795 | 0.742 | 0.719 |
| Pre-2020 V5.2 | 33.4% C | 32.5% C | 40.7% C | 37.2% C |
| Pre-2020 Sharpe | -0.498 | -0.999 | +0.034 | -0.436 |
| Pre-2020 gross | -10.8% | -21.7% | +3.0% | -10.3% |

**Key finding:** ng1.0.1 as bull sub-model is BETTER than ng1.0.7 for Pre-2020 (v1: +3.0% vs -10.8%; v2: -10.3% vs
-21.7%). ng1.0.7 is the weaker picker confirmed. However, even with the stronger ng1.0.1 bull, v2 regime still
underperforms v1 on Pre-2020 — confirming the issue is the regime classifier over-calling bull, not the sub-model.

The v2+ng101 WF-OOS MaxDD improvement (-17.6% vs v1+ng101 -26.8%) is notable and likely reflects v2's additional
bull days landing in correctly-identified bull periods within 2020-2026.

## Summary

- WF-OOS: v2 regime is marginally better on risk-adjusted metrics (V5.2 +2.8pp A+, Sharpe +0.17), but lower raw returns
- Pre-2020: v2 regime structurally worse regardless of bull sub-model (same pattern as ng1.0.7 bull)
- Sub-model swap (ng1.0.7 → ng1.0.1) improves absolute Pre-2020 performance (+13pp gross in v1, +11pp in v2)
- **Root cause confirmed:** v2 regime over-calls bull in true bear conditions (2018-2019), this is regime-structural
- **Recommendation:** Do not deploy v2 regime in current form. Either (a) add a Pre-2020 fine-tuning penalty to the
  regime votes, (b) require higher vote threshold for bull call, or (c) keep v1 regime with ng1.0.1 bull as sub-model
  (v1+ng101 WF-OOS V5.2=76.5% A+, Sharpe=2.580, Pre-2020 gross +3.0% — best Pre-2020 of any regime switch config)
