# NG v2.0a Multi-Beta Regime Classifier

**Status:** 灰度对比生产 ng106v2 (2026-04-26 上线候选)

## 核心架构

3 个独立信号 hard vote (2-of-3 多数, 系统级 3d streak) → 路由 ng1.0.1 (bull) 或 ng1.0.4 (bear).

| 信号 | 计算 | hysteresis | streak |
|---|---|---|---|
| V11 | 0AMV 位置 + MACD 水上/上升 + 3 日平滑 (existing `regime_classifier.py:_v11_loose_smooth3`) | — | 3d |
| B1 | % stocks closing above MA20 / MA60 (panel) | (0.45, 0.55) | 3d |
| B2 | 沪深300 60d realized vol → 252d percentile (反向: 低vol=bull) | (0.30, 0.70) | 3d |

vote_threshold=2 (多数), system_streak=3 (default).

Phase C calibration sweep 试了 5 variants (baseline / strict_b1 / strict_b2 / streak5 / unanimous), 用户选 **baseline**, `unanimous` 备选 (Pre-2020 改善但 WF-OOS MaxDD 退步).

## 性能 vs 生产 ng106v2

(基于 sub-model = ng1.0.1 bull + ng1.0.4-3s bear offline ensemble 评估)

| 指标 | ng106v2 (生产) | **ng2.0a (baseline)** | Δ |
|---|---:|---:|---:|
| WF-OOS V5.2 | 80.4% S | 79.3% A+ | -1.1pp |
| WF-OOS Sharpe | 2.39 | 2.751 | +0.36 |
| **WF-OOS MaxDD** | **-23.7%** | **-17.6%** | **+6.1pp** ✓ |
| WF-OOS 年化(净) | — | 89.3% | — |
| ICIR | — | 0.7189 | — |
| Pre-2020 V5.2 | 33.4% C | 37.2% C | +3.8pp |
| Pre-2020 净年化 | -17.8% | -17.3% | +0.5pp |

注: 生产 ng106v2 用的是 ng1.0.4 single-model bear; ng2.0a Phase C 评估用 ng1.0.4-3s offline ensemble; 生产 wiring 也用 ng1.0.4 single-model 保持一致.

## 落地

- selector: `--scoring-version ng2.0a`
- 报告: `reports/daily_selection_ng2_0a_fullmarket/` (`--full-market` 默认开启)
- regime 数据: `market_regime_signals` table (主表, baseline 校准); 备选 `market_regime_signals_unanimous`
- 模块:
  - `indicators/breadth.py` (B1)
  - `indicators/realized_vol.py` (B2)
  - `indicators/regime_classifier.py:compute_regime_v2` (vote)
  - `tomorrow_stock_selector.py:5811-5847` (ng200a_mode 分支)
- backfill: `scripts/build_regime_v2_history.py` (主表) / `scripts/build_regime_v2_history_variant.py` (变体)

## 关键决策日志

- 2026-04-25 Step A PASS (84% agreement vs V11 baseline, 0.87x flips, v2 抓 2019Q1 反弹早 14 天)
- 2026-04-25 Step B 原版 (ng1.0.7 bull) 实质平局 (V5.2 -0.2pp)
- 2026-04-25 Step B Option C (ng1.0.1 bull) v2 +2.8pp V5.2 + 9pp MaxDD 改善 → 决策切换 sub-model
- 2026-04-26 Phase C calibration sweep, 5 variants 全跑, 无一通过 Pre-2020 annual_net ≥ -8% 硬 gate, 用户选 baseline 进 Phase A
- 2026-04-26 Phase A 接生产, 灰度起跑 (单日 smoke 2026-04-24, 与 ng1.0.62 100% Top-10 overlap, 都判 bear → ng1.0.4)

## 已知限制

- B1/B2 在长期熊市末段 (2018Q4 → 2019Q1) 倾向 over-call bull (Pre-2020 +33 多余 bull 天 vs V11). Phase C `unanimous` variant 部分缓解 (Pre-2020 净年化 -11.0%, +6.3pp 改善) 但 WF-OOS MaxDD 退到 -23.2%, 用户优先保留 -17.6% MaxDD.
- regime 数据需要 daily 更新: `python3 scripts/build_regime_v2_history.py --start <yesterday> --end <today>`. 已加入 daily_update workflow (Phase A7).
- Production bear scorer 是 ng1.0.4 single-model (与 ng106v1/v2 wiring 一致); Phase C 评估用的 ng1.0.4-3s 是 offline 3-seed ensemble 报告. 灰度评估时若想完全复现 Phase C 数字, 需要单独跑 ng1.0.4-3s scorer (未上 selector).

## 关联

- spec: `docs/superpowers/specs/2026-04-25-ng200-regime-refined-architecture-design.md`
- plan 主线: `docs/superpowers/plans/2026-04-25-ng200a-multi-beta-regime.md`
- plan 后续: `docs/superpowers/plans/2026-04-26-ng2_0a-followup-CAB.md` (C→A→B 闭环)
- 上游: `regime_classifier_v1` (V11 baseline, `_v11_loose_smooth3`)
- 下游: `ng2_0b` (sub-model regime-weighted retrain, 待 Phase B)
