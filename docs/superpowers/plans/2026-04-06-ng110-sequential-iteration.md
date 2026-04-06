# NG v1.1.0 顺序迭代实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复ng1.0.2的2018-2020 OOS失效问题(年化超额从-7.6%变为>0%), 通过残差标签去除动量因子暴露(β_UMD=3.029)

**Architecture:** ng1.1.0训练器从ng110_feature_cache读取已预计算的风格残差标签(label_Xd), 训练数据限定2020-01-01起, 2018-2020数据仅用于OOS评估。按step1(残差标签)→step2(+资金流)→step3(+WF8)逐步验证, 每步需通过2018-2020 OOS硬门槛。

**Tech Stack:** Python, LightGBM/XGBoost/CatBoost/RF/HGB ensemble, SQLite (ng110_feature_cache), 北极星评估框架

**关键约束:** 2018-2020数据绝对不进入训练集, 仅用于OOS回测评估。`--start-date 2020-01-01` 参数保证此约束。

---

### Task 1: 修复ng_trainer.py缺失的特征名常量

**Files:**
- Modify: `ml_models/ng/ng_trainer.py:83-92`

- [ ] **Step 1: 添加缺失的特征名**

在 `ml_models/ng/ng_trainer.py` 中, MONEYFLOW_FEATURE_NAMES 缺少 `northbound_stock_5d`, INTERACTION_FEATURE_NAMES 缺少 `ix_north_cap`:

```python
# Line 83-92, replace:
MONEYFLOW_FEATURE_NAMES: List[str] = [
    'net_mf_ratio_5d', 'big_order_ratio', 'big_order_trend_5d',
    'small_vs_big_divergence', 'mf_concentration', 'mf_momentum_10d',
    'mf_volume_divergence',
]

INTERACTION_FEATURE_NAMES: List[str] = [
    'ix_vol_pullback', 'ix_big_trend', 'ix_rsi_mf', 'ix_ind_big',
    'ix_mf_efficiency', 'ix_vol_surge_pullback', 'ix_alpha_conc',
]

# With:
MONEYFLOW_FEATURE_NAMES: List[str] = [
    'net_mf_ratio_5d', 'big_order_ratio', 'big_order_trend_5d',
    'small_vs_big_divergence', 'mf_concentration', 'mf_momentum_10d',
    'mf_volume_divergence', 'northbound_stock_5d',
]

INTERACTION_FEATURE_NAMES: List[str] = [
    'ix_vol_pullback', 'ix_big_trend', 'ix_rsi_mf', 'ix_ind_big',
    'ix_mf_efficiency', 'ix_vol_surge_pullback', 'ix_alpha_conc',
    'ix_north_cap',
]
```

- [ ] **Step 2: 验证import不报错**

Run: `python3 -c "from ml_models.ng.ng_trainer import MONEYFLOW_FEATURE_NAMES, INTERACTION_FEATURE_NAMES; print(f'MF={len(MONEYFLOW_FEATURE_NAMES)}, IX={len(INTERACTION_FEATURE_NAMES)}')"`
Expected: `MF=8, IX=8`

- [ ] **Step 3: Commit**

```bash
git add ml_models/ng/ng_trainer.py
git commit -m "fix(ng): 补全MONEYFLOW/INTERACTION特征名常量 (各+1)"
```

---

### Task 2: 回填2018-2020 ng110特征缓存 (仅用于OOS评估)

**Files:**
- No code changes — data operation only

- [ ] **Step 1: 确认moneyflow_daily覆盖2018**

Run: `python3 -c "import sqlite3; c=sqlite3.connect('data_adapter/stock_data.db'); r=c.execute('SELECT MIN(trade_date), COUNT(DISTINCT trade_date) FROM moneyflow_daily WHERE trade_date < \"2020-01-01\"').fetchone(); print(f'moneyflow pre-2020: start={r[0]}, days={r[1]}'); c.close()"`
Expected: start=2018-01-02, days≈480+

- [ ] **Step 2: 回填ng110缓存2018-2020**

Run: `python3 ml_models/ng/ng_cache_updater.py --start-date 2018-01-01 --end-date 2019-12-31 --version ng1.1.0 2>&1 | tail -5`
Expected: 完成~500天回填, 无ERROR。耗时约15-30分钟。

- [ ] **Step 3: 验证回填结果**

Run: `python3 -c "import sqlite3; c=sqlite3.connect('data_adapter/stock_data.db'); r=c.execute('SELECT MIN(trade_date), MAX(trade_date), COUNT(DISTINCT trade_date) FROM ng110_feature_cache WHERE trade_date < \"2020-01-01\"').fetchone(); print(f'ng110 pre-2020: {r[0]} ~ {r[1]}, {r[2]} days'); c.close()"`
Expected: 2018-04-02 ~ 2019-12-31, ~400+ days

---

### Task 3: Step 1 — 残差标签fast-check

**Files:**
- No code changes — training run

**关键**: `--start-date 2020-01-01` 确保2018-2020不进入训练。ng110_feature_cache的label_Xd已经是残差标签。

- [ ] **Step 1: 运行fast-check**

Run:
```bash
python3 ml_models/ng/ng_trainer.py \
  --start-date 2020-01-01 \
  --fast-check \
  --purge-days 15 \
  --lambda-risk 0.5 \
  2>&1 | tee logs/ng110_step1_fastcheck.log
```
Expected: 完成~2分钟, 输出10d IC/ICIR。
通过条件: 10d ICIR > 0.5

- [ ] **Step 2: 记录fast-check结果**

Run: `grep -E "10d.*IC|ICIR|summary" logs/ng110_step1_fastcheck.log | tail -10`
记录10d ICIR数值, 与ng1.0.2基线对比。

---

### Task 4: Step 1 — 残差标签完整训练

**Files:**
- No code changes — training run
- Output: `ml_models/trained_models/ng/ng_multi_target_*.pkl`

**关键**: 训练数据从2020-01-01开始, 绝不包含2018-2020。

- [ ] **Step 1: 完整训练 (仅在fast-check ICIR>0.5时执行)**

Run:
```bash
python3 ml_models/ng/ng_trainer.py \
  --start-date 2020-01-01 \
  --purge-days 15 \
  --lambda-risk 0.5 \
  2>&1 | tee logs/ng110_step1_full.log
```
Expected: ~1-2小时, 生成模型文件。

- [ ] **Step 2: 确认模型保存**

Run: `ls -lt ml_models/trained_models/ng/ng_multi_target_*.pkl | head -1`
Expected: 新生成的.pkl文件, 大小60-80MB

- [ ] **Step 3: 记录WF OOS IC/ICIR**

Run: `grep -E "Window.*IC|OOS|ICIR|summary|10d" logs/ng110_step1_full.log | tail -20`

---

### Task 5: Step 1 — 2018-2020 OOS评估 (硬门槛)

**Files:**
- No code changes — evaluation run

- [ ] **Step 1: 生成2018-2020报告**

Run:
```bash
python3 backtest/batch_generate_v395_reports.py \
  --version ng1.1.0 \
  --start-date 2018-04-02 --end-date 2020-12-31 \
  --output-dir reports/daily_selection_ng110_pre2020 \
  --force \
  2>&1 | tail -10
```
Expected: ~400-600天报告生成成功

- [ ] **Step 2: 2018-2020顺序回测**

Run:
```python
python3 << 'PYEOF'
import json, sqlite3, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict

DB_PATH = 'data_adapter/stock_data.db'
REPORT_DIR = Path('reports/daily_selection_ng110_pre2020')

conn = sqlite3.connect(DB_PATH)
price_df = pd.read_sql("""
    SELECT s.code, dq.trade_date, dq.close
    FROM daily_quotes dq JOIN securities s ON s.id = dq.security_id
    WHERE dq.trade_date >= '2018-01-01' AND dq.trade_date <= '2021-06-30'
    ORDER BY s.code, dq.trade_date
""", conn)
idx_df = pd.read_sql("""
    SELECT dq.trade_date, dq.close as idx_close
    FROM daily_quotes dq JOIN securities s ON s.id = dq.security_id
    WHERE s.code = '000300.SH' AND dq.trade_date >= '2018-01-01' AND dq.trade_date <= '2021-06-30'
    ORDER BY dq.trade_date
""", conn)
conn.close()

price_dict = defaultdict(dict)
for _, row in price_df.iterrows():
    price_dict[row['code']][row['trade_date']] = row['close']
all_trade_dates = sorted(price_df['trade_date'].unique())
date_index = {d: i for i, d in enumerate(all_trade_dates)}
idx_dict = {row['trade_date']: row['idx_close'] for _, row in idx_df.iterrows()}

files = sorted(REPORT_DIR.glob('analysis_data_*.json'))
files_map = {}
for f in files:
    ds = f.stem.split('_')[-1]
    files_map[f"{ds[:4]}-{ds[4:6]}-{ds[6:]}"] = f
report_dates = sorted(files_map.keys())

COST = 0.003
HOLD = 10

for top_n in [5, 10]:
    nav = 1.0
    idx_nav = 1.0
    prev = set()
    rets = []
    i = 0
    while i < len(report_dates):
        date = report_dates[i]
        with open(files_map[date], 'r') as fh:
            data = json.load(fh)
        stocks = data.get('all_stocks_with_scores', [])
        a = [(s['stock_code'], float(s.get('composite', s.get('rank_score', 0)) or 0))
             for s in stocks if s.get('industry', '') and len(s.get('stock_code', '')) <= 6]
        a.sort(key=lambda x: x[1], reverse=True)
        hold = [c for c, _ in a[:top_n]]
        new_set = set(hold)
        turn = len(new_set - prev) / max(len(new_set), 1)
        prev = new_set
        exit_i = min(i + HOLD, len(report_dates) - 1)
        exit_date = report_dates[exit_i]
        pr = 0; v = 0
        for c in hold:
            p0 = price_dict.get(c, {}).get(date)
            p1 = price_dict.get(c, {}).get(exit_date)
            if p0 and p1 and p0 > 0:
                pr += (p1 - p0) / p0; v += 1
        if v > 0:
            pr /= v
            nr = pr - COST * turn
            nav *= (1 + nr)
            rets.append(nr)
        ix0 = idx_dict.get(date); ix1 = idx_dict.get(exit_date)
        if ix0 and ix1 and ix0 > 0:
            idx_nav *= (1 + (ix1 - ix0) / ix0)
        if exit_i >= len(report_dates) - 1: break
        i = exit_i + 1

    days = (pd.to_datetime(report_dates[-1]) - pd.to_datetime(report_dates[0])).days
    yrs = days / 365.25
    ann = (nav ** (1/yrs) - 1) * 100
    idx_ann = (idx_nav ** (1/yrs) - 1) * 100
    ra = np.array(rets)
    navs = [1.0]
    for r in rets: navs.append(navs[-1]*(1+r))
    navs = np.array(navs)
    pk = np.maximum.accumulate(navs)
    mdd = np.min((navs-pk)/pk)*100
    sh = np.mean(ra)/np.std(ra)*np.sqrt(25) if np.std(ra)>0 else 0
    wr = np.mean(ra>0)*100
    print(f"Top-{top_n}: 年化{ann:+.1f}%, 沪深300{idx_ann:+.1f}%, "
          f"超额{ann-idx_ann:+.1f}%, Sharpe={sh:.2f}, MaxDD={mdd:.1f}%, 胜率={wr:.1f}%")

print(f"\n硬门槛: 年化超额 > 0% ? (ng1.0.2基线: -7.6%)")
PYEOF
```

Expected output: 年化超额数值。**硬门槛: 超额 > 0%**。

- [ ] **Step 3: 判定PASS/FAIL**

如果Top-5年化超额 > 0% → PASS, 继续Task 6 (Step 2)
如果Top-5年化超额 ≤ 0% → FAIL, 停止并诊断

---

### Task 6: Step 2 — 残差标签 + 资金流 fast-check

**Files:**
- No code changes — training run
- **前置条件**: Task 5 PASS

- [ ] **Step 1: fast-check with moneyflow**

Run:
```bash
python3 ml_models/ng/ng_trainer.py \
  --start-date 2020-01-01 \
  --fast-check \
  --purge-days 15 \
  --lambda-risk 0.5 \
  --enable-moneyflow \
  2>&1 | tee logs/ng110_step2_fastcheck.log
```
Expected: ~2-3分钟, 输出76特征的IC/ICIR

- [ ] **Step 2: 对比Step 1的ICIR**

Run: `grep -E "10d.*IC|ICIR" logs/ng110_step2_fastcheck.log | tail -5`
判定: 10d ICIR > Step 1的ICIR → 有增量, 继续完整训练
       10d ICIR ≤ Step 1的ICIR → 无增量, 跳过Step 2, 用Step 1的模型

---

### Task 7: Step 2 — 完整训练 + OOS (仅在有增量时执行)

**Files:**
- No code changes
- **前置条件**: Task 6 显示正增量

- [ ] **Step 1: 完整训练**

Run:
```bash
python3 ml_models/ng/ng_trainer.py \
  --start-date 2020-01-01 \
  --purge-days 15 \
  --lambda-risk 0.5 \
  --enable-moneyflow \
  2>&1 | tee logs/ng110_step2_full.log
```

- [ ] **Step 2: 生成2018-2020报告 + OOS回测**

重复Task 5的Step 1-2, 使用新生成的模型。
报告目录: `reports/daily_selection_ng110_step2_pre2020`

硬门槛: 年化超额 > Step 1的结果 (增量为正)

---

### Task 8: Step 3 — WF8 + regime完整训练 (最终模型)

**Files:**
- No code changes
- **前置条件**: 选择Step 1或Step 2中OOS更好的配置作为基础

- [ ] **Step 1: 完整训练 WF8**

Run (如果Step 2有增量):
```bash
python3 ml_models/ng/ng_trainer.py \
  --start-date 2020-01-01 \
  --purge-days 15 \
  --lambda-risk 0.5 \
  --enable-moneyflow \
  --wf-windows 8 \
  --regime-weight \
  2>&1 | tee logs/ng110_step3_full.log
```

Run (如果Step 2无增量, 只用残差标签):
```bash
python3 ml_models/ng/ng_trainer.py \
  --start-date 2020-01-01 \
  --purge-days 15 \
  --lambda-risk 0.5 \
  --wf-windows 8 \
  --regime-weight \
  2>&1 | tee logs/ng110_step3_full.log
```

Expected: ~3-5小时

- [ ] **Step 2: 双向评估**

生成2018-2020报告 + 顺序回测 (同Task 5流程)
生成2024-2026报告 + 顺序回测 (同之前session的回测脚本)

最终通过条件:
- 2018-2020 OOS 年化超额 > 0%
- 2024+ in-sample不低于ng1.0.2的70% (~82%年化)
- MaxDD < -30%

- [ ] **Step 3: 更新production_config.json (如果通过)**

将最佳模型路径写入production_config.json, 更新version为ng1.1.0。

---

### Task 9: Commit结果并更新记忆

- [ ] **Step 1: Commit所有变更**

```bash
git add ml_models/ng/ng_trainer.py production_config.json
git commit -m "feat(ng): ng1.1.0训练完成 — 残差标签+OOS验证"
```

- [ ] **Step 2: 输出最终对比表**

格式:
```
| 模型 | 2018-2020 OOS超额 | 2024+ IS超额 | Sharpe | MaxDD |
|------|-------------------|-------------|--------|-------|
| ng1.0.2 (基线) | -7.6% | +107% | 2.62 | -7.7% |
| ng1.1.0 Step1 | ?% | ?% | ? | ?% |
| ng1.1.0 Step2 | ?% | ?% | ? | ?% |
| ng1.1.0 Final | ?% | ?% | ? | ?% |
```
