# Handoff Prompt — ng2.1v2 真 WF-OOS 重训 + 全模型对比

**Session 起点**: 2026-04-28 21:16

把下面整段贴给下一个 session, 它能接着跑。

---

# 任务

接管 ng101/ng104/ng107 真 WF-OOS 重训 (with `--wf-report-dir` + purge=16 fix), 然后拼装 ng106 v1 / ng1062 v2 系统并跑 V5.2 真 OOS eval, 输出 5 模型 apples-to-apples 对比表。

# 背景 (重要, 不读过会重蹈覆辙)

## 关键发现 (本 session 验证)

1. **历史所有 `batch_generate_v395_reports.py` + `run_north_star_eval.py` 测出的 V5.2 数字都被 IS-inflated ~30pp**, 因为 batch_generate 用 production model (训于全 2020-2026 数据) 在训练期评分。Memory 里 ng2.0a "V5.2=79.3% A+" 真 OOS 实际只有 46.2% B。所有过去 V5.2 决策都基于这些虚高数字。

2. **修复方法**: 用 trainer `--wf-report-dir` 让 trainer 在每个 WF fold 写出 test 期 OOS predictions (raw 模型 in-fold ensemble), 然后 stitch 这些 fold preds 成完整 OOS 时间序列, 再跑 V5.2 eval。

3. **次要 bug**: `purge_days=15` 与 `label_15d` 单日重叠 (label_15d at T 用 close[T+16], test_start = train_end+1+purge=T+16 with train_end=T). 修复: **`--purge-days 16`**。

4. **真 OOS 5 模型对比 (已部分完成)**:

| 模型 | 真 OOS V5.2 | MaxDD | 超额年化 | 备注 |
|---|---|---|---|---|
| ng1.0.1 single | **54.6% B** ⭐ | -27.5% | **+29.5%** | 当前已知最强 |
| ng2.1v2 raw (regime-tailored) | 48.8% B | -25.7% | -4.6% | ng21v2 实验 |
| ng1.0.6 v1 (生产) | 46.4% B | -39.6% | -5.3% | AMV+101+104 |
| ng2.0a (V11+101+104) | 46.2% B | -70.9% | -20.9% | |
| ng2.1v2 + L1L2L4 overlay | 35.5% C | -24.0% | -3.3% | overlay net negative |

待补: ng1.0.4 单模, ng1.0.7 单模, ng1.0.62 v2 系统 (=ng107+ng104+AMV).

## 当前在跑的训练 (PIDs)

```bash
# 检查
ps -p $(cat /tmp/retrain_ng101.pid) -o pid,etime,rss
ps -p $(cat /tmp/retrain_ng104.pid) -o pid,etime,rss
```

启动时间 21:14, 单 seed, --purge-days 16, --target-parallel 4。

输出目录:
- `reports/ng101_baseline_wf_oos/seed42/` (期望 ~360 fold preds 真 OOS)
- `reports/ng104_baseline_wf_oos/seed42/` (期望 ~360 fold preds 真 OOS)

预计完成 ~22:45-23:15.

## 已有资产 (可重用)

- `reports/ng21v2_bull_wf_oos/seed42/` - 720 fold preds (ng2.1v2-bull)
- `reports/ng21v2_bear_wf_oos/seed42/` - 720 fold preds (ng2.1v2-bear)
- `ml_models/trained_models/ng/ng21v2-bull_seed42_multi_target_20260428_010035.pkl`
- `ml_models/trained_models/ng/ng21v2-bear_seed42_multi_target_20260427_232258.pkl`
- `scripts/ng21v2_phase4_system_eval.py` - stitch + V5.2 eval helper
- `scripts/ng21_apply_overlay_to_reports.py` - L1-L5 overlay (但真 OOS 上 net negative, 不要用)

## ⚠️ 内存约束 (硬约束, 非常重要)

47GB 总内存。每个 ng_trainer / ng21v2_trainer proc 峰值 RSS ~14-21GB。

- **2 procs 并行** = 安全 (每个 ~6-9GB RSS during normal, 14-21GB peak in production model phase)
- **3 procs 并行** = 触发 swap, 性能崩
- **4 procs 并行** = 25GB+ swap, OOM 风险, **绝对禁止** (本 session 已踩坑)

启动新训练前必查:
```bash
ps aux | grep "python3 -m ml_models" | grep -v grep | wc -l
top -l 1 -n 0 -s 0 | grep PhysMem
sysctl vm.swapusage
```

如果 swap used > 5GB, kill 一个 proc 再启动新的。

# 待办清单

## 步骤 1: 等 ng101 + ng104 完成 (~22:45-23:15)

监控:
```bash
# 检查活跃
ps aux | grep "python3 -m ml_models.ng.ng_trainer" | grep -v grep

# 完成标志: pkl 落盘
ls -la ml_models/trained_models/ng/ng101_seed42_multi_target_20260428_2*.pkl
ls -la ml_models/trained_models/ng/ng104_seed42_multi_target_20260428_2*.pkl

# fold preds count (期望 ~360)
ls reports/ng101_baseline_wf_oos/seed42/ | wc -l
ls reports/ng104_baseline_wf_oos/seed42/ | wc -l
```

## 步骤 2: 起 ng107 (单 proc, 等 ng101/104 完成后)

```bash
cd /Users/yangxu/StockTradebyZ
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p reports/ng107_baseline_wf_oos

nohup caffeinate -i python3 -m ml_models.ng.ng_trainer \
  --version ng1.0.7 \
  --wf-report-dir reports/ng107_baseline_wf_oos \
  --start-date 2020-01-01 \
  --purge-days 16 \
  --target-parallel 4 \
  --seed 42 > logs/ng21/retrain_ng107_${TS}.log 2>&1 &
echo $! > /tmp/retrain_ng107.pid
disown
```

预计 ~1.5h. 期望输出: `reports/ng107_baseline_wf_oos/seed42/` 360 fold preds.

⚠️ 注意 ng1.0.7 用 `ng107_feature_cache` 不同 schema, 训练前确认表存在:
```bash
python3 -c "import sqlite3; c=sqlite3.connect('data_adapter/stock_data.db'); print(c.execute('SELECT COUNT(*) FROM ng107_feature_cache').fetchone())"
```

## 步骤 3: 拼装 + V5.2 eval (~30min)

```bash
# 3a. ng106 v1 (生产, AMV→ng101+ng104)
python3 << 'PYEOF'
import sqlite3, shutil
from pathlib import Path
conn = sqlite3.connect('data_adapter/stock_data.db', timeout=30)
amv = {str(d): int(r) for d, r in conn.execute(
    "SELECT trade_date, amv_regime FROM market_amv WHERE amv_regime IS NOT NULL"
).fetchall()}
conn.close()

bull, bear = Path('reports/ng101_baseline_wf_oos/seed42'), Path('reports/ng104_baseline_wf_oos/seed42')
out = Path('reports/ng106v1_amv_system_oos'); out.mkdir(parents=True, exist_ok=True)
all_files = sorted({f.name for f in bull.glob('*.json')} | {f.name for f in bear.glob('*.json')})
n_b = n_d = 0
for fn in all_files:
    iso = f"{fn[14:18]}-{fn[18:20]}-{fn[20:22]}"
    r = amv.get(iso)
    if r == 1 and (bull/fn).exists():
        shutil.copy2(bull/fn, out/fn); n_b += 1
    elif r == -1 and (bear/fn).exists():
        shutil.copy2(bear/fn, out/fn); n_d += 1
print(f'ng106v1 stitched: bull={n_b}, bear={n_d}, total={len(list(out.glob("*.json")))}')
PYEOF

# 3b. ng1062 v2 (AMV→ng107+ng104)
python3 << 'PYEOF'
import sqlite3, shutil
from pathlib import Path
conn = sqlite3.connect('data_adapter/stock_data.db', timeout=30)
amv = {str(d): int(r) for d, r in conn.execute(
    "SELECT trade_date, amv_regime FROM market_amv WHERE amv_regime IS NOT NULL"
).fetchall()}
conn.close()

bull, bear = Path('reports/ng107_baseline_wf_oos/seed42'), Path('reports/ng104_baseline_wf_oos/seed42')
out = Path('reports/ng1062v2_amv_system_oos'); out.mkdir(parents=True, exist_ok=True)
all_files = sorted({f.name for f in bull.glob('*.json')} | {f.name for f in bear.glob('*.json')})
n_b = n_d = 0
for fn in all_files:
    iso = f"{fn[14:18]}-{fn[18:20]}-{fn[20:22]}"
    r = amv.get(iso)
    if r == 1 and (bull/fn).exists():
        shutil.copy2(bull/fn, out/fn); n_b += 1
    elif r == -1 and (bear/fn).exists():
        shutil.copy2(bear/fn, out/fn); n_d += 1
print(f'ng1062v2 stitched: bull={n_b}, bear={n_d}, total={len(list(out.glob("*.json")))}')
PYEOF

# 3c. V5.2 eval all 5 (find common date range first)
python3 -c "
from pathlib import Path
for d in ['ng101_baseline_wf_oos/seed42', 'ng104_baseline_wf_oos/seed42',
          'ng107_baseline_wf_oos/seed42', 'ng106v1_amv_system_oos', 'ng1062v2_amv_system_oos',
          'ng21v2_bull_wf_oos/seed42', 'ng21v2_bear_wf_oos/seed42']:
    p = Path('reports')/d
    files = sorted(p.glob('analysis_data_*.json'))
    if files:
        print(f'{d}: {files[0].stem[-8:]} ~ {files[-1].stem[-8:]} ({len(files)} files)')
"
```

找出最窄共同日期 (very likely 2024-05-08 ~ 2025-10-30 for ng_trainer-based with min_train=900). 然后跑 5 个 V5.2:

```bash
COMMON_START=2024-05-08  # 调整为实际共同最早日
COMMON_END=2025-10-30    # 调整为实际共同最晚日

for cfg in \
  "reports/ng101_baseline_wf_oos/seed42:ng1.0.1-single" \
  "reports/ng104_baseline_wf_oos/seed42:ng1.0.4-single" \
  "reports/ng107_baseline_wf_oos/seed42:ng1.0.7-single" \
  "reports/ng106v1_amv_system_oos:ng1.0.6v1-system" \
  "reports/ng1062v2_amv_system_oos:ng1.0.62v2-system"; do
  DIR="${cfg%:*}"; LABEL="${cfg#*:}"
  caffeinate -i python3 backtest/run_north_star_eval.py --backtest \
    --report-dir "$DIR" --label "$LABEL" \
    --top-n 10 --focus-days 10 --rank-field composite \
    --start-date $COMMON_START --end-date $COMMON_END \
    > logs/ng21/eval_${LABEL//./_}_$(date +%Y%m%d_%H%M%S).log 2>&1
  echo "Done: $LABEL"
done

# 提取 V5.2 score (10d holding)
for f in logs/ng21/eval_*$(date +%Y%m%d).log; do
  echo "=== $f ==="
  grep -E "V5.2:|加权评分.*等级|10日持仓|年化收益.*净|Sharpe:|最大回撤|超额年化|月度胜率" "$f" | head -15
done
```

## 步骤 4: 输出对比表

把 5 个 system + ng21v2 raw (V5.2=48.8%, 已知) 整合成最终 8 行表:

| Model | V5.2 | MaxDD | 月度胜率 | 超额年化 | 切生产建议 |
|---|---|---|---|---|---|
| ng1.0.1 single | 54.6% B | -27.5% | 61.1% | +29.5% | ⭐ 候选 |
| ng2.1v2 raw | 48.8% B | -25.7% | 66.7% | -4.6% | 备选 |
| ng1.0.6 v1 (重测) | TBD | TBD | TBD | TBD | (当前生产) |
| ng2.0a | 46.2% B | -70.9% | 22.2% | -20.9% | × |
| ng1.0.4 single (新) | TBD | TBD | TBD | TBD | (单模 alpha 弱) |
| ng1.0.7 single (新) | TBD | TBD | TBD | TBD | |
| ng1.0.62 v2 (新) | TBD | TBD | TBD | TBD | (生产 v2) |
| overlay 系列 | <40 | n/a | n/a | n/a | × |

## 步骤 5: 写报告 + 更新 memory

```
1. 写 docs/superpowers/plans/2026-04-28-ng-true-oos-baseline-rebaseline.md (5 模型对比 + 真 OOS V5.2 校准)
2. 更新 ~/.claude/projects/-Users-yangxu-StockTradebyZ/memory/MEMORY.md 顶部加 entry:
   "## 🚨 真 WF-OOS 校准 (2026-04-28) → 详见 ..."
3. 标记历史 V5.2 IS-inflated memory 加 caveat (不要删除, 加 ⚠️ IS-inflated 注释)
4. 给用户最终建议: 切 ng1.0.1 single 还是保 ng1.0.6 v1?
```

# 重要约束

- **不要并行 3+ 训练任务** (47GB 内存硬上限, 25GB swap 已踩过)
- **不要再用 batch_generate 测训练期 V5.2** (IS-inflated)
- **purge_days >= 16** for label_15d (标准 ng_trainer 默认 15 是 bug)
- **保留所有 pkl** (`.gitignore` 排除, 不要 git add)
- **L1-L5 overlay 真 OOS 上 net negative**, 不要用作生产 (memory 之前的"风控才是真价值"是基于 IS 错觉)

# 关键文件

- 设计 plan: `docs/superpowers/plans/2026-04-26-ng21-bull-bear-specialist.md`
- ng2.1v2 trainer: `ml_models/ng21/ng21v2_trainer.py`
- Phase 4 helper: `scripts/ng21v2_phase4_system_eval.py`
- Bug 审计 + 此 handoff: `docs/HANDOFF_NG21V2_RETRAIN.md`
- 上一 session 详细记录: 见对话历史

# 完成标志

- ✅ ng101/ng104/ng107 各 ~360 fold preds 落盘
- ✅ ng106 v1 + ng1062 v2 stitched (各 ~360)
- ✅ 5 systems V5.2 真 OOS 数字出
- ✅ 对比表更新到 plan
- ✅ memory 加新 entry + 历史 V5.2 加 IS caveat
- ✅ 给用户切生产候选清晰建议 (附 paper trade 验证流程)

---

**当前 session 进度**: ng101 + ng104 在跑 (~22:30 完成), 详见 `/tmp/retrain_ng101.pid`, `/tmp/retrain_ng104.pid`. 接管时直接进步骤 1 监控等待。
