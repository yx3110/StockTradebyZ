# Plan: 急跌切熊信号 (var1 单日 ≤ -2.3%) 引入与回测

_Created: 2026-04-25 · Branch: main · Author: Claude session before context clear_

## 0. 上下文

- 用户提议: **var1 单日跌幅 ≥ 2.3%** 作为牛→熊触发信号
- 现状: V3 strict 已含此信号, 但作为 AND 组合 (`pct≤-2.3% AND var1<ma60 AND macd<0`), 三重确认偏保守, 实际很少触发
- 改进方向: 让急跌信号更**独立/快速**触发熊市切换
- 当前生产: V11 (位置 + (水上 OR 上升) + 3日平滑), 无急跌触发, 切熊全靠正常条件失效
- 测试框架: `scripts/regime_variant_compare.py` 已就绪
- 相关文件: `indicators/regime_classifier.py`, `indicators/market_amv.py`, `data_adapter/stock_data.db:market_amv`

## 1. 新原子信号 (在 `regime_classifier.py` 加 1 个)

```python
def sig_panic_drop(var1, drop_thresh=-0.023, **_):
    """急跌信号: var1 单日跌幅 ≤ -2.3%"""
    arr = np.zeros(len(var1), dtype=bool)
    arr[1:] = (var1[1:] - var1[:-1]) / (var1[:-1] + 1e-15) <= drop_thresh
    return arr
```

## 2. 新 variant 设计 (5 个, 都基于 V11 加急跌 overlay)

| ID | 配方 | 设计意图 |
|---|---|---|
| **V16** panic_immediate | V11 + 单日 -2.3% **OR-触发**强制熊 (覆盖 V11 牛市判定) | 最 aggressive: 见急跌就跑 |
| **V17** panic_cooldown_3d | V11 + 单日 -2.3% 触发**未来 3 日强制熊** | 给市场 3 日缓冲消化恐慌, 不会因后续 1 日反弹立刻切回 |
| **V18** panic_cash_3d | V11 + 单日 -2.3% 触发**未来 3 日 cash** (3-state) | 极度防御: 急跌后既不信牛也不信熊, 完全 cash 等市场出方向 |
| **V19** panic_AND_position | V11 + 单日 -2.3% **AND** var1<ma60 → 强制熊 | 次于 V16: 急跌且已经在均线下方才切 (避免高位震荡误切) |
| **V20** panic_streak_2d | V11 + 连续 2 日累计跌 ≥ -3.5% → 强制熊 | 不看单日, 看 2 日累计 — 更平稳但稍慢 |

**控制组**:
- V11 (当前生产) — baseline
- V3 strict (旧生产) — 已含 -2.3% 但 AND 严格

### V11 现状参考 (写死, 用于通过门槛)
- 2024-2026: 净年化 108.9%, Sharpe 1.989, MaxDD -14.0%
- 2020-2026: 净年化 97.5%, Sharpe 2.386, MaxDD -23.7%
- pre-2020: 净年化 -17.8%, Sharpe -0.498, MaxDD -31.7%

## 3. 测试矩阵

3 窗口 × 7 variants (5 新 + 2 控制):

| Window | 报告范围 | bull_dir | bear_dir |
|---|---|---|---|
| 2024-2026 | 556 天 | reports/daily_selection_ng1.0.7_fast | reports/daily_selection_ng104_ensemble_3seed |
| 2020-2026 | 1526 天 | 同上 | 同上 |
| pre-2020 | 360 天 | reports/daily_selection_ng107_pre2020 | reports/daily_selection_ng104_pre2020 |

预计耗时 (并行): ~14 min total (2024≈5, 2020≈14, pre2020≈3)

## 4. 评估指标 (按重要性)

1. **Robust 优先** (不受 sparse-trading 影响): MaxDD, cumulative return over fixed span, monthly win rate
2. **方向**: 净年化 — 但**cash 比例 >20% 时打折看** (sparse-trading 膨胀, 见 wiki/lessons)
3. **稳健性**: Sharpe (**>4 直接 red flag**), Sortino, 跨窗口排名一致性
4. **副作用监控**: cash 比例, 切换次数 (>30 次/年偏频繁), 急跌触发次数 (-2.3% 在每窗口被触发几次)

## 5. 通过门槛 (写死, 不许事后调)

新 variant 替换 V11 的条件 (**全部满足**):

| 维度 | 门槛 |
|---|---|
| 2024-2026 净年化 | ≥ 108.9% (V11 baseline) |
| 2024-2026 Sharpe | ≥ 1.989 **且** ≤ 4.0 (sanity) |
| 2020-2026 净年化 | ≥ 92.5% (V11 - 5pp 容差) |
| 2020-2026 MaxDD | ≤ -23.7% (V11) 容差 +2pp 即 ≤ -25.7% |
| pre-2020 净年化 | ≥ -17.8% (V11; 不要求达正, 只看是否减损) |
| pre-2020 MaxDD | ≤ -31.7% (V11) |
| cash 比例 | ≤ 30% (用户体验门槛) |
| 急跌触发次数 | ≥ 5 次/窗口 (信号必须真有 bite) |

## 6. 执行步骤

```bash
# Step 1: 加 sig_panic_drop + 5 个新 preset 到 indicators/regime_classifier.py
#   - 在 "原子信号" 段加 sig_panic_drop
#   - 在 "预设" 段加 _v16 ~ _v20 函数
#   - PRESETS dict 加 5 entries
#   - DEFAULT_PRESET 不动 (仍是 v11_loose_smooth3)

# Step 2: 同步加到 scripts/regime_variant_compare.py
#   - regime_v16/17/18/19/20 函数 (与 classifier 等价但独立, 因 script 用自己的 sig_*)
#   - 注意 script 里 sig_* 已存在, 只需加 panic_drop 函数 + 5 variant 函数
#   - VARIANTS 列表加 5 entries
#   - 注意 V18 是 3-state (含 cash), merge_reports 已支持

# Step 3: Sanity check (~10 秒, 必跑)
python3 -c "
import sqlite3, pandas as pd
from indicators.regime_classifier import RegimeClassifier
with sqlite3.connect('data_adapter/stock_data.db') as conn:
    df = pd.read_sql('SELECT * FROM market_amv ORDER BY trade_date', conn)
for p in ['v16_panic_immediate','v17_panic_cooldown_3d','v18_panic_cash_3d','v19_panic_AND_position','v20_panic_streak_2d']:
    r = RegimeClassifier(p).fit_predict(df)
    n_panic = sum(1 for i in range(1,len(df)) if (df['var1'].iloc[i]-df['var1'].iloc[i-1])/df['var1'].iloc[i-1] <= -0.023)
    print(f'{p}: 总={len(r)}, 牛={(r==1).sum()}, 熊={(r==-1).sum()}, cash={(r==0).sum()}, 急跌触发={n_panic}')
"

# Step 4: 三窗口并行回测 (background)
python3 scripts/regime_variant_compare.py --window 2024 --variants V11,V3,V16,V17,V18,V19,V20 --label-suffix panic &
python3 scripts/regime_variant_compare.py --window 2020 --variants V11,V3,V16,V17,V18,V19,V20 --label-suffix panic &
python3 scripts/regime_variant_compare.py --window pre2020 --variants V11,V3,V16,V17,V18,V19,V20 --label-suffix panic &
wait

# Step 5: 比对汇总, 用第 5 节门槛逐项筛选
#   - 写跨窗口对比表 (markdown), 标注每项是否 PASS
#   - 列出 cash 比例 / 急跌触发次数 / Sharpe sanity flag
#   - 如有 variant 全部 PASS:
#       1. 改 indicators/regime_classifier.py:DEFAULT_PRESET 为新 preset
#       2. 重算 market_amv: python3 indicators/market_amv.py
#       3. 重生成 ng106v2 历史: python3 scripts/merge_ng1062_historical_reports.py --start-date 2024-01-01 --end-date $(date +%Y-%m-%d) --overwrite
#       4. commit + 更新 wiki + log + memory

# Step 6: 失败兜底 (5 个 variant 都 FAIL):
#   - 写 memory (feedback 类型): "急跌切熊在 V11 base 上无增益, 因为 ng104 bear 模型已能处理急跌"
#   - 写 wiki/lessons 一条
#   - 不改生产
#   - 不浪费时间继续 grid search 阈值 (-2.3% → -1.8% / -3% 等), 4-22 memory 已证 0AMV 信号细化多失败
```

## 7. 关键风险

| 风险 | 应对 |
|---|---|
| 与 memory 4-22 "crisis overlay -6pp 全败" 矛盾 | 任何 variant 看起来强先看 cash 比例 + Sharpe sanity (>4 警惕) |
| 过拟合 pre-2020 (急跌信号天然适合 18-19 长熊) | 必须**所有窗口都过门槛**, 不允许"2024 损失换 pre-2020 收益" |
| 稀疏 trading 数字膨胀 | V18 (3-state cash) 必有此风险, Sharpe sanity 必检 |
| 急跌信号在 2024 牛市误伤 | V16 (immediate) 高风险, V17/V20 (cooldown/streak) 中风险, V19 (AND position) 低风险 |

## 8. 文件清单 (执行后产出)

| 路径 | 状态 | 说明 |
|---|---|---|
| `indicators/regime_classifier.py` | 修改 | +1 信号 (sig_panic_drop) +5 preset |
| `scripts/regime_variant_compare.py` | 修改 | +1 信号 +5 variant |
| `reports/regime_variant_compare_2024_panic.md` | 新 | 2024 窗口 7 variants 结果 |
| `reports/regime_variant_compare_2020_panic.md` | 新 | 2020 窗口 |
| `reports/regime_variant_compare_pre2020_panic.md` | 新 | pre-2020 窗口 |
| `docs/wiki/architecture/regime-classifier-v1.md` | 修改 | 追加 panic-drop 实验段 |
| `docs/wiki/log.md` | 修改 | +1 entry (PASS or FAIL) |
| `~/.claude/.../memory/regime_panic_drop_2026_04_25.md` | 新 | project memory (PASS) 或 feedback memory (FAIL) |

## 9. Decision Tree

```
执行回测
   │
   ├─ 有 variant 全部门槛 PASS?
   │     │
   │     ├─ YES → 选最强 → 切 DEFAULT_PRESET → 重算 market_amv → 重生成 ng106v2 → commit
   │     │
   │     └─ NO → 写 memory + wiki/lessons → 不改生产 → 结束
```

## 10. 备注 (执行者读)

- 当前 `regime_classifier.py` 已含 `_v3_strict` 函数可参考 (它有 -2.3% AND 三重逻辑作模板)
- `merge_reports` 已支持 3-state (cash=0), V18 直接能跑
- V18 看到 Sharpe>4 / 净年化>200% 立即怀疑数字膨胀
- 灰度 V11 1 周后 (5-1 后) 若无问题, 此计划可执行; 之前不要叠 panic 改动到 V11 之上
- 不要触动 `bull=ng101` 留底实验 (周末用户决策, 与本计划独立)
