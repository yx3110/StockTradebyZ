# ng1.3.0 Stage 3.5 Gate REJECTED (2026-04-19)

## 结果
- **V5.2 (composite, top10, 10d, 2025 全年)**: **60.6% raw** / 39.9% C 级 (242天 × 0.66 折扣)
- **Gate 阈值**: V5.2 ≥ 65% — **FAIL by 4.4pp**

## 关键指标 vs ng1.0.1 baseline
| 指标 | ng1.0.1 | ng1.3.0 (current) | 差距 |
|---|---|---|---|
| V5.2 (WF-OOS) | 72.1% A+ | 60.6% C | -11.5pp |
| Sharpe | 2.753 | 0.571 | -2.18 |
| MaxDD | -11.7% | -16.4% | -4.7pp |
| 超额年化 | +91% | -13.6% | 反向 |
| 年化(净) | 165.7% | 15.4% | -150pp |

## 根本原因 (3 层)
1. **特征集不完整** (实际训练): 81 features = 56 stock (NG107 base) + 18 market + 7 cond_ix
   - Tier A 4 个 downside 特征 (current_drawdown 等) **未进训练**
   - Tier B 3 个 mf 特征 (elg_z 等) **未进训练**
   - 原因: `prepare_features` 走 NG107 fallback path, 覆盖 init 时 NGTrainer 设的 NG130 配置
2. **β 未调优**: composite 用默认 β=0.3, 未跑 Phase 5 的 WF 网格搜索
3. **AMV regime 信号有限**: AMV 进了训练 (我 Round 2 fix 后), 但 dual-head 架构没有 Tier A downside 配套, 无法形成完整的 risk-aware 选股

## 已修复的 4 个 critical bugs (永久价值)
1. `_compute_cache_key` 加 head_suffix → excess vs downside 不再 bit-identical
2. `_load_moneyflow_data` 用 code_6 列 → mf 因子真正 join 成功 (ng1.2.x 也受益)
3. `compute_ng130_mf_factors` log-sign fallback → elg_z 不再永远 NaN
4. `load_data` AMV 从 features_json 读取 → AMV regime 特征不再 NaN

## 已通过的部分
- Dual-head 架构 work: corr(excess, downside) = -0.04 (真不同)
- Seed diversity: 0.82-0.97 (合规 [0.85, 0.95] 范围)
- Composite scorer + predict_scores API 完整可用
- 223 tests pass

## 下一步选择 (用户决策)

**A. ABORT — 保持 ng1.0.1 生产 (推荐 if 时间紧)**
- PRODUCTION_VERSION 已是 'ng1.0.1', 不变
- ng1.3.0 artifacts 永久保留作 postmortem
- 总投入: ~16h trace+ ~21h training + ~3h evaluate = ~40h "证伪"
- 价值: 4 个 critical bug fixes 长期受益

**B. 修 prepare_features → ng1.3.1 retrain (~7h)**
- 修 prepare_features 确保 NG130 完整 81 features 进训练 (Tier A + B + AMV)
- 重训 6 runs (~7h)
- 再跑 Stage 3.5
- Cost: 7h 训练 + 2h Stage 3.5 = 9h
- 风险: 可能 still fail (Tier A/B IC 弱过 EMT gate threshold), 但至少 properly 测试

**C. 跳过 Tier A/B, 直接 β 网格搜索**
- 现有模型 base 已是 NG107, 试 β ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5} 看哪个最好
- β=0 退化到纯 excess head (ng1.0.1-like)
- Cost: 2-4h grid search
- 价值: 知道 β 对 V5.2 的影响曲线

