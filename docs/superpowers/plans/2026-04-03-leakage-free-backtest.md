# 无泄露双向回测系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立完整的无泄露评估体系——WF OOS 向前评估 + 2018-2020 向后评估，两个proxy综合描述生产模型真实表现。

**Architecture:** 
1. 回填2018-2020特征缓存（确认数据完整性→补缺→回填）
2. 改造北极星评估流程：训练后自动执行双向评估
3. 重训V4901验证完整流程

**Tech Stack:** Python3, SQLite, Tushare API, V39 Feature Cache Updater, V4901 Scorer, North Star V4/V5

---

## Task 1: 确认2018-2020数据完整性

**Files:**
- Read: `data_adapter/stock_data.db`
- Create: `scripts/check_2018_data_completeness.py`

- [ ] **Step 1: 编写数据完整性检查脚本**

```python
#!/usr/bin/env python3
"""检查2018-2020回测所需的所有数据完整性"""
import sqlite3
import sys

DB_PATH = 'data_adapter/stock_data.db'

def check():
    conn = sqlite3.connect(DB_PATH)
    issues = []

    # 1. daily_quotes: 2018-01-01 ~ 2019-12-31
    r = conn.execute("""
        SELECT COUNT(*), COUNT(DISTINCT trade_date), COUNT(DISTINCT security_id)
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.type = 'A股' AND dq.trade_date >= '2018-01-01' AND dq.trade_date < '2020-01-01'
    """).fetchone()
    print(f"daily_quotes (A股): {r[0]:,} 条, {r[1]} 交易日, {r[2]} 只股票")
    if r[1] < 480:
        issues.append(f"daily_quotes 交易日不足: {r[1]}/~487")

    # 2. daily_basic: PE/PB/PS 覆盖率
    r2 = conn.execute("""
        SELECT COUNT(*), COUNT(DISTINCT trade_date)
        FROM daily_basic db
        JOIN securities s ON db.security_id = s.id
        WHERE s.type = 'A股' AND db.trade_date >= '2018-01-01' AND db.trade_date < '2020-01-01'
    """).fetchone()
    print(f"daily_basic (A股): {r2[0]:,} 条, {r2[1]} 交易日")
    # daily_basic覆盖率低于50%需要补数据
    coverage = r2[0] / max(r[0], 1) * 100
    print(f"  daily_basic覆盖率: {coverage:.1f}%")
    if coverage < 40:
        issues.append(f"daily_basic覆盖率过低: {coverage:.1f}%")

    # 3. 市场指数(沪深300)是否有数据 — v39特征的market_*列需要
    r3 = conn.execute("""
        SELECT COUNT(*) FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.code = '000300.SH'
          AND dq.trade_date >= '2018-01-01' AND dq.trade_date < '2020-01-01'
    """).fetchone()
    print(f"沪深300指数: {r3[0]} 交易日")
    if r3[0] < 480:
        issues.append(f"沪深300指数数据不足: {r3[0]}/~487")

    # 4. sw_industry (行业映射) — 静态表，检查是否存在
    r4 = conn.execute("SELECT COUNT(*) FROM sw_industry WHERE level = 1").fetchone()
    print(f"申万一级行业: {r4[0]} 条")
    if r4[0] == 0:
        issues.append("sw_industry表为空")

    # 5. label计算依赖: 2020年初的close/open (用于计算2019-12-xx的label_15d)
    r5 = conn.execute("""
        SELECT MIN(trade_date), MAX(trade_date) FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.code = '000001.SZ'
    """).fetchone()
    print(f"000001.SZ 行情范围: {r5[0]} → {r5[1]}")

    conn.close()

    if issues:
        print(f"\n❌ 发现 {len(issues)} 个问题:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        return 1
    else:
        print("\n✅ 数据完整性检查通过")
        return 0

if __name__ == '__main__':
    sys.exit(check())
```

- [ ] **Step 2: 运行检查脚本**

Run: `python3 scripts/check_2018_data_completeness.py`

Expected: 输出各项数据量。如果 daily_basic 覆盖率不足或沪深300数据缺失，进入 Task 2 补数据。如果全部通过，跳到 Task 3。

---

## Task 2: 补齐缺失数据（如需要）

**Files:**
- Modify: `fetch_data/quick_daily_update.py` (参考其Tushare调用方式)
- Run: Tushare API 补数据

- [ ] **Step 1: 补 daily_basic（如覆盖率 <40%）**

```bash
# 使用已有的回填脚本
python3 fetch_data/backfill_historical_data.py --mode daily_basic \
  --start-date 2018-01-01 --end-date 2019-12-31
```

如果该脚本不支持指定日期范围，用 tushare 直接补：

```python
#!/usr/bin/env python3
"""补齐2018-2019 daily_basic数据"""
import tushare as ts
import sqlite3
import json
import time

with open('config.json') as f:
    token = json.load(f)['tushare']['token']
pro = ts.pro_api(token)

conn = sqlite3.connect('data_adapter/stock_data.db')

# 按月拉取
dates = []
for y in [2018, 2019]:
    for m in range(1, 13):
        dates.append(f"{y}{m:02d}01")

for start in dates:
    try:
        df = pro.daily_basic(trade_date=start[:6], 
                              fields='ts_code,trade_date,pe_ttm,pb,ps_ttm,turnover_rate,circ_mv,total_mv')
        if df is not None and not df.empty:
            print(f"  {start[:6]}: {len(df)} 条")
            # 写入数据库...（参考quick_daily_update.py的写入逻辑）
        time.sleep(0.3)  # 尊重API限频
    except Exception as e:
        print(f"  {start[:6]}: 错误 {e}")

conn.close()
```

- [ ] **Step 2: 补沪深300指数数据（如不足）**

```bash
python3 fetch_data/quick_daily_update.py --date 20180102
# 或使用batch_fetch_historical_data.py补充指数数据
```

- [ ] **Step 3: 重新运行完整性检查确认**

Run: `python3 scripts/check_2018_data_completeness.py`
Expected: ✅ 全部通过

---

## Task 3: 回填2018-2020特征缓存

**Files:**
- Run: `fetch_data/v39_feature_cache_updater.py`

- [ ] **Step 1: 回填v39_feature_cache（2018-01-01 → 2019-12-31）**

```bash
python3 fetch_data/v39_feature_cache_updater.py \
  --start-date 2018-01-01 \
  --end-date 2019-12-31
```

v39_feature_cache_updater 直接从 `daily_quotes` 计算特征，不依赖 `technical_indicators`。
预计耗时：487天 × ~1.8s/天 ≈ 15分钟。

- [ ] **Step 2: 验证回填结果**

```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('data_adapter/stock_data.db')
r = conn.execute('''
    SELECT MIN(trade_date), MAX(trade_date), COUNT(*), COUNT(DISTINCT trade_date)
    FROM v39_feature_cache
    WHERE trade_date >= \"2018-01-01\" AND trade_date < \"2020-01-01\"
''').fetchone()
print(f'2018-2019 feature cache: {r[0]} → {r[1]}, {r[2]:,} 条, {r[3]} 交易日')

# 检查label覆盖
r2 = conn.execute('''
    SELECT COUNT(*) FROM v39_feature_cache
    WHERE trade_date >= \"2018-01-01\" AND trade_date < \"2020-01-01\"
      AND label_3d IS NOT NULL AND label_5d IS NOT NULL AND label_10d IS NOT NULL
''').fetchone()
print(f'有完整label的记录: {r2[0]:,}')
conn.close()
"
```

Expected: ~487交易日, 每天~3500只A股 ≈ 170万条记录, label覆盖率>95%

---

## Task 4: 改造训练脚本——自动生成2018-2020回测报告

**Files:**
- Modify: `ml_models/training/train_v395_multi_target.py` (末尾CLI区域)

当前训练完成后只做了WF OOS评估。新增：用生产模型对2018-2020生成报告并评估。

- [ ] **Step 1: 在训练脚本末尾新增2018-2020回测**

在 `train_v395_multi_target.py` 的 `main()` 函数末尾，现有WF OOS自动评估代码之后，添加：

```python
    # ── 训练后自动评估2: 2018-2020 向后泛化 (生产模型在训练前数据上的表现) ──
    if _actual_wf_dir and not args.skip_wf and not args.fast_check:
        # 检查2018-2020特征缓存是否存在
        import sqlite3
        _db = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')
        _conn = sqlite3.connect(_db)
        _n_pre = _conn.execute(
            "SELECT COUNT(DISTINCT trade_date) FROM v39_feature_cache "
            "WHERE trade_date >= '2018-01-01' AND trade_date < '2020-01-01'"
        ).fetchone()[0]
        _conn.close()

        if _n_pre >= 200:  # 至少200个交易日才有统计意义
            logger.info("\n" + "=" * 60)
            logger.info(f"自动向后泛化评估: 2018-2020 ({_n_pre} 交易日)")
            logger.info("=" * 60)

            # 用生产模型生成2018-2020报告
            _pre_report_dir = str(Path(_actual_wf_dir).parent / 
                                   f'{Path(_actual_wf_dir).name.replace("_wf_oos", "")}_pre2020')
            _gen_cmd = [
                sys.executable, str(PROJECT_ROOT / 'backtest' / 'batch_generate_v395_reports.py'),
                '--version', _ver,
                '--start-date', '2018-01-01',
                '--end-date', '2019-12-31',
                '--output-dir', _pre_report_dir,
                '--force',
            ]
            logger.info(f"  生成报告: {' '.join(_gen_cmd)}")
            subprocess.run(_gen_cmd, cwd=str(PROJECT_ROOT))

            # 北极星评估
            _eval_cmd = [
                sys.executable, str(PROJECT_ROOT / 'backtest' / 'run_north_star_eval.py'),
                '--backtest',
                '--report-dir', _pre_report_dir,
                '--label', 'PRE-2020',
                '--top-n', '10',
                '--focus-days', '10',
                '--rank-field', 'composite',
            ]
            logger.info(f"  北极星评估: {' '.join(_eval_cmd)}")
            subprocess.run(_eval_cmd, cwd=str(PROJECT_ROOT))
        else:
            logger.info(f"\n  ⚠️ 2018-2020特征缓存不足({_n_pre}天), 跳过向后泛化评估")
            logger.info(f"  回填命令: python3 fetch_data/v39_feature_cache_updater.py "
                         f"--start-date 2018-01-01 --end-date 2019-12-31")
```

- [ ] **Step 2: 验证语法**

Run: `python3 -c "import ast; ast.parse(open('ml_models/training/train_v395_multi_target.py').read()); print('OK')"`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add ml_models/training/train_v395_multi_target.py scripts/check_2018_data_completeness.py
git commit -m "feat: 训练后自动双向无泄露评估 — WF OOS(向前) + 2018-2020(向后)"
```

---

## Task 5: 停止后台训练 + 重训V4901验证完整流程

**Files:**
- Run: `ml_models/training/train_v395_multi_target.py`

- [ ] **Step 1: 停止当前后台训练（如仍在运行）**

```bash
# 查找并终止
ps aux | grep train_v395 | grep -v grep
kill <PID>  # 如果仍在运行
```

- [ ] **Step 2: 启动完整训练（含双向评估）**

```bash
python3 ml_models/training/train_v395_multi_target.py --v4901 --purge-days 15 \
  2>&1 | tee logs/v4901_dual_eval_$(date +%Y%m%d_%H%M%S).log
```

训练完成后会自动依次执行：
1. WF OOS 北极星评估 (向前泛化)
2. 2018-2020 北极星评估 (向后泛化)

- [ ] **Step 3: 对比双向评估结果**

训练日志末尾会输出两份北极星评分卡：
- `WF-OOS`: 向前泛化（多个WF窗口test period）
- `PRE-2020`: 向后泛化（生产模型在训练前数据上）

关注：
- 两个proxy的IC/ICIR是否都为正
- 年化收益是否都>0
- 北极星等级对比

---

## 执行顺序总结

```
Task 1: 检查数据完整性 (~1min)
  ↓ 如有缺失
Task 2: 补齐缺失数据 (~10min, 可能不需要)
  ↓
Task 3: 回填2018-2020特征缓存 (~15min)
  ↓
Task 4: 改造训练脚本 (~10min)
  ↓
Task 5: 重训V4901 + 双向评估 (~3-5h)
```

总计：除训练外约30分钟准备工作。
