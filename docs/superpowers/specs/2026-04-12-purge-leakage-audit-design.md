# Purge Leakage Audit — NG 训练集泄漏量化审计设计

**日期**: 2026-04-12
**状态**: Design Approved, pending implementation
**仓库**: StockTradebyZ（本仓库）
**对应外部 todo**: EastMoneyTrader `docs/ideas_todo.md` #2「模型数据泄漏验证」

---

## 目标

对 NG 训练器当前默认的 `purge_days=15` 做一次性量化审计：把 purge 从 15 天增加到 30 天，重新训练，看 OOS IC 下降多少。若下降显著，说明 15 天 purge 不够，训练集对测试集有泄漏，当前生产模型的 OOS 指标虚高。

核心原理：
- NG 标签最长 forward 15 日（`label_15d`），当前 purge 恰好等于最长标签 forward，**无安全裕量**
- Lopez de Prado Purged K-Fold 的标准要求是 `gap >= label_forward`。15 = 15 处于临界，理论上任何"浮点舍入"或"dataframe 边界计算"的瑕疵都会让标签未来信息漏进测试集
- 如果 15 天和 30 天 purge 的 OOS IC 差距 < 10%，结论是"15 天足够"；差距 > 30%，结论是"15 天严重不够"

## 关键决策（已对齐）

| # | 决策点 | 选择 |
|---|---|---|
| 1 | 工具归属仓库 | StockTradebyZ（本仓库，不在 EMT 侧） |
| 2 | 实验规模 | 2 purge × 3 版本 × 4 标签 = 12 比较点 / 6 次训练 |
| 3 | Purge 对照值 | `15`（基线，当前默认）vs `30`（控制，double gap） |
| 4 | 版本集合 | `ng1.0.1`（当前 prod）、`ng106`（EMT rebalancer 默认读）、`ng1.1.0`（路线图下一代） |
| 5 | 标签集合 | `label_3d / label_5d / label_10d / label_15d` |
| 6 | WF 窗口数 | 默认 3 窗口（`ng_trainer.py` 当前默认）；不用 `--fast-check`，不用 8 窗口 |
| 7 | 工具形态 | Runner + Analyzer 两个 CLI 分离，用 run.json 通信 |
| 8 | 报告格式 | Markdown 决策表 + 判决汇总（一页） |
| 9 | 判决阈值 | Δ% < 10 → 🟢 GREEN；10-30 → 🟡 YELLOW；>30 → 🔴 RED；baseline IC < 0.005 → ⚪ N/A |
| 10 | 存储 | `reports/purge_audit_YYYYMMDD/{run_id}/run.json` + skip-if-exists（`--force` 重跑） |

---

## 架构

```
StockTradebyZ/ (本仓库)
├ scripts/
│   ├ run_purge_experiment.py       (runner: subprocess 驱动 ng_trainer + run.json 汇总)
│   └ analyze_purge_leakage.py      (analyzer: 读 run.json × 生成 REPORT.md)
│
├ reports/purge_audit_20260412/
│   ├ ng1.0.1_purge15/
│   │   ├ run.json                   (metadata + per-label mean OOS IC + 耗时)
│   │   └ trainer.log                (ng_trainer stdout+stderr 捕获)
│   ├ ng1.0.1_purge30/
│   ├ ng106_purge15/
│   ├ ng106_purge30/
│   ├ ng1.1.0_purge15/
│   ├ ng1.1.0_purge30/
│   └ REPORT.md                      (最终产物)
│
└ ml_models/ng/ng_trainer.py         (只读消费 CLI，不改)
```

**边界**：
- runner 只通过 `subprocess.run([sys.executable, "ml_models/ng/ng_trainer.py", "--version", V, "--purge-days", P])` 驱动 ng_trainer，不 import 其内部
- runner 扫描 `ml_models/trained_models/ng/` 目录定位训练产出的 `wf_summary.json`（通过 mtime 差 + "新出现"的原则）
- analyzer 只读 runner 产出的 6 份 `run.json`，离线可反复重跑不重训
- ng_trainer.py 本身不改任何代码（`--purge-days` 和 `--version` 已经是它的 CLI 参数）

---

## 数据流

### Runner (`run_purge_experiment.py`)

```python
for version in ['ng1.0.1', 'ng106', 'ng1.1.0']:
    for purge in [15, 30]:
        run_id = f"{version}_purge{purge}"
        run_dir = reports_root / f"purge_audit_{today}" / run_id

        # Skip if already done
        if (run_dir / "run.json").exists() and not args.force:
            log("skip %s", run_id); continue
        run_dir.mkdir(parents=True, exist_ok=True)

        # Snapshot existing wf_summary files (to detect new one after training)
        pre_snapshot = _snapshot_wf_summaries(trained_models_dir)

        # Run training (blocking, up to 4h)
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, "ml_models/ng/ng_trainer.py",
             "--version", version, "--purge-days", str(purge),
             "--start-date", "2020-01-01"],
            capture_output=True, text=True, timeout=14400,
            cwd=str(STZ_ROOT))
        elapsed = time.time() - t0
        (run_dir / "trainer.log").write_text(
            proc.stdout + "\n---STDERR---\n" + proc.stderr)

        # Locate new wf_summary
        post_snapshot = _snapshot_wf_summaries(trained_models_dir)
        new_wf_path = _find_new_wf_summary(pre_snapshot, post_snapshot)
        oos_ics = (_extract_per_label_mean_oos_ic(json.load(new_wf_path.open()))
                   if new_wf_path else {})

        (run_dir / "run.json").write_text(json.dumps({
            "run_id": run_id,
            "version": version,
            "purge_days": purge,
            "started_at": start_iso,
            "elapsed_seconds": elapsed,
            "returncode": proc.returncode,
            "wf_summary_path": str(new_wf_path) if new_wf_path else None,
            "per_label_mean_oos_ic": oos_ics,
            "per_label_n_windows": _extract_n_windows(new_wf_path),
        }, indent=2, ensure_ascii=False))
```

**容错**：
- `returncode != 0`：记录非 0 code，trainer.log 保留 stderr，不 raise；analyzer 跳过这条。继续下一个 run。
- `TimeoutExpired`（>4h）：raise 给用户，让他知道超时；未完成的 run.json 不写。
- 找不到新 wf_summary：`wf_summary_path: null`，analyzer 视为无数据跳过。

**CLI**：
```
python3 scripts/run_purge_experiment.py [--date YYYYMMDD] [--force] [-v]
```

### Analyzer (`analyze_purge_leakage.py`)

```python
audit_dir = (_find_latest_audit_dir() if args.date is None
             else reports_root / f"purge_audit_{args.date}")

# Load 6 run.json files
runs = {}  # (version, purge) → dict
for run_json_path in audit_dir.glob("*/run.json"):
    data = json.loads(run_json_path.read_text())
    runs[(data['version'], data['purge_days'])] = data

# Derive versions present in the audit dir (in case some runs didn't complete)
versions_found = sorted({v for (v, _) in runs.keys()})

# Compute delta per (version, label)
rows = []
for version in versions_found:
    baseline = runs.get((version, 15), {})
    control  = runs.get((version, 30), {})
    for label in ['3d', '5d', '10d', '15d']:
        b = baseline.get('per_label_mean_oos_ic', {}).get(f'label_{label}')
        c = control.get('per_label_mean_oos_ic', {}).get(f'label_{label}')
        delta_pct = (b - c) / b if b and b > 0 else None
        rows.append({
            'version': version, 'label': label,
            'baseline_ic': b, 'control_ic': c,
            'delta_abs': (b - c) if b is not None and c is not None else None,
            'delta_pct': delta_pct,
            'verdict': classify_verdict(b, delta_pct),
        })

# Render REPORT.md
(audit_dir / "REPORT.md").write_text(render_report(rows, runs))
```

**判决函数**（纯函数，易测）：
```python
def classify_verdict(baseline_ic: float | None, delta_pct: float | None) -> str:
    """Classify leakage severity.

    Rules:
      - baseline_ic < 0.005 (or None) → N/A (signal too weak to judge)
      - delta_pct None → N/A (missing control data)
      - 0 ≤ delta_pct < 0.10 → GREEN (no significant leakage)
      - 0.10 ≤ delta_pct < 0.30 → YELLOW (mild leakage)
      - delta_pct ≥ 0.30 → RED (severe leakage)
      - delta_pct < 0 → GREEN (control actually better; noise, not leakage)
    """
    if baseline_ic is None or baseline_ic < 0.005:
        return "⚪ N/A (baseline IC 过低)"
    if delta_pct is None:
        return "⚪ N/A"
    if delta_pct < 0.10:
        return "🟢 GREEN"
    if delta_pct < 0.30:
        return "🟡 YELLOW"
    return "🔴 RED"
```

**CLI**：
```
python3 scripts/analyze_purge_leakage.py [--date YYYYMMDD]
```

---

## REPORT.md 格式

```markdown
# Purge Leakage Audit — 2026-04-12

实验配置: 3 versions × 2 purge × 4 labels = 12 比较点
- Baseline: `purge_days=15` (当前 ng_trainer 默认)
- Control:  `purge_days=30` (double gap)
- WF: 3 windows (default), `--start-date 2020-01-01`

## 全量判决表

| 版本 | 标签 | baseline IC (p=15) | control IC (p=30) | Δ% | verdict |
|---|---|---|---|---|---|
| ng1.0.1 | 3d  | 0.0612 | 0.0585 | +4.4%  | 🟢 GREEN |
| ng1.0.1 | 5d  | 0.0583 | 0.0501 | +14.1% | 🟡 YELLOW |
| ng1.0.1 | 10d | 0.0548 | 0.0439 | +19.9% | 🟡 YELLOW |
| ng1.0.1 | 15d | 0.0498 | 0.0180 | +63.9% | 🔴 RED |
| ng106   | 3d  | ... | ... | ... | ... |
| ...     | ... | ... | ... | ... | ... |

## 汇总

- 🟢 GREEN: N / 12
- 🟡 YELLOW: N / 12
- 🔴 RED: N / 12
- ⚪ N/A: N / 12

## 发现 / 建议（auto-rendered）

{per-row narrative: 如果某 version 某 label 是 RED，给出建议 "purge_days 默认应改为 ≥30 / 需要进一步跑 purge=60 验证饱和点"}

## 运行元数据

| run_id | 耗时 | returncode | WF windows (label_5d) |
|---|---|---|---|
| ng1.0.1_purge15 | 42 min | 0 | 3 |
| ... |
```

---

## 模块分解

| 文件 | 操作 | 职责 | 估算 LOC |
|---|---|---|---|
| `scripts/run_purge_experiment.py` | Create | Subprocess 驱动 + wf_summary 定位 + run.json 持久化 | ~180 |
| `scripts/analyze_purge_leakage.py` | Create | 读 runs × 算 Δ × 生成 Markdown | ~150 |
| `tests/test_purge_leakage_analyzer.py` | Create | Analyzer 纯函数 + 聚合单测 | ~120 |

### Analyzer 纯函数（可单测）

```python
def classify_verdict(baseline_ic: float | None, delta_pct: float | None) -> str
def compute_delta_rows(runs: dict, versions: list[str], labels: list[str]) -> list[dict]
def load_runs(audit_dir: Path) -> dict[tuple[str, int], dict]
def render_report(rows: list[dict], runs: dict, audit_date: str) -> str
```

### Runner 辅助函数

```python
def _snapshot_wf_summaries(model_dir: Path) -> dict[Path, float]   # path → mtime
def _find_new_wf_summary(pre: dict, post: dict) -> Path | None
    # Returns path that either:
    #   - exists in post but not in pre (new file), OR
    #   - exists in both but post[path] > pre[path] (updated mtime)
    # If multiple candidates, returns the one with latest post mtime.
def _extract_per_label_mean_oos_ic(wf_summary: dict) -> dict[str, float]
def _extract_n_windows(wf_summary_path: Path | None) -> dict[str, int]
```

---

## 测试策略

### Analyzer（纯函数，高测试 ROI）
- `classify_verdict` 边界：
  - `baseline=0.004` → N/A
  - `baseline=0.06, delta=0.09` → GREEN
  - `baseline=0.06, delta=0.10` → YELLOW
  - `baseline=0.06, delta=0.30` → RED
  - `delta=None` → N/A
  - `delta=-0.05`（control 更高）→ GREEN
- `compute_delta_rows` 用合成 runs 字典验证输出结构
- `load_runs` 用 `tmp_path` 造 6 个假 run.json 验证组织
- `render_report` 快速 snapshot test（verify key 字符串存在，不做精确格式比对）

### Runner（subprocess 驱动，低测试 ROI）
- 只测 `_extract_per_label_mean_oos_ic`（合成 dict 喂入）
- `_snapshot_wf_summaries` / `_find_new_wf_summary` 用 tmp_path + 手动 touch
- 不 mock subprocess（runner 主要价值就是真跑，mock 测没意义）

### 无端到端测试
实验本身就是 end-to-end。fast-check 跑一次（Task 4）算管道验证。

---

## 实施顺序

| 阶段 | 内容 | 代码时间 | 计算时间 |
|---|---|---|---|
| 1 | Analyzer 纯函数 + TDD 单测（`classify_verdict`、`compute_delta_rows`、`render_report`） | ~2h | 0 |
| 2 | Runner：subprocess driver + wf_summary 定位 + run.json 落盘 | ~3h | 0 |
| 3 | Analyzer：读 run.json × 组织 × 生成 Markdown | ~2h | 0 |
| 4 | 管道验证：用 `--fast-check` 跑 1 个版本 1 个 purge 值（smoke test） | ~0.5h | ~30 min |
| 5 | 正式实验：runner 跑 6 次 × 默认 3 窗口（后台） | ~0.5h | **3-9 h** |
| 6 | Analyzer 跑 + 人眼 review REPORT.md | ~0.5h | 0 |
| 7 | 在 EMT 侧的 `docs/ideas_todo.md` 标记 #2 完成（跨仓编辑） | ~0.2h | 0 |

**代码小时 ~9h + 计算时间 ~5h**。

---

## YAGNI 边界

- ❌ 不跑 `purge=60/120` — 推迟到后续（如果 30 结果有大量 RED 再说）
- ❌ 不改 `ng_trainer.py` 的默认 `purge_days=15` — 审计结论后由人类决策
- ❌ 不生成 PNG / 图表 — Markdown 表够决策用
- ❌ 不自动触发后续补救 — 发现泄漏后人工决定如何改
- ❌ 不并行训练 runs —— ng_trainer 内部已有 `--parallel`，外层叠加并行会抢 CPU
- ❌ 不监控训练进度 — runner 是 blocking 调用，用户自己看 stdout
- ❌ 不集成进 EMT 的 daily_pipeline 或 IC Monitor — 这是一次性审计，不是监控
- ❌ 不支持 8 窗口 — 初期 3 窗口若不够敏感再说

---

## 风险与未决

1. **`wf_summary.json` 定位不稳**：ng_trainer 每次训练会在 `ml_models/trained_models/<version>/` 下写 `wf_summary.json`，但多次训练会覆盖。runner 必须 snapshot 训练前 mtime，训完后找"新出现或更新"的 wf_summary。Task 2 必须显式处理此逻辑。
2. **`ng_trainer.py --version ng106` 格式接受性**：已见到 `--version ng1.0.3/ng1.0.4` 示例，但 `ng106`（无点号）是新格式。Task 1 第一步必须运行 `python3 ml_models/ng/ng_trainer.py --help` 验证，并 `--version ng106 --fast-check` 做快速空跑确认能接受。若不接受，查 `ng_schema.py` 的 `version_ge` 逻辑找正确形式。
3. **训练时间估算**：ng1.0.1 (~69 特征) vs ng1.1.0 (~77) 特征数不同，训练时间可能差 1.5-2x。Task 4 的 fast-check 跑完校准真实时间估算，再决定 Task 5 是否分阶段跑。
4. **3 WF 窗口统计功效**：对 `label_15d` 的 IC 方差可能淹没 10-15% 的真实差异。本次先用 3 窗口看趋势；若结论边缘不清（`verdict` 大量在 10-15% 临界），二期补 8 窗口重跑。
5. **跨仓 todo 同步**：EMT `docs/ideas_todo.md` #2 的标记"已完成"需要在 EMT 仓库做（`git add && git commit` in /Users/yangxu/EastMoneyTrader）。Task 7 明确处理。

---

## 参考

- EMT `docs/ideas_todo.md` #2（原始 todo 条目）
- EMT `docs/quant_theory_level2_3.md` Level 2 §1.4「训练样本构造 / Purged K-Fold」
- StockTradebyZ `ml_models/training/train_v395_multi_target.py:916-964`（split_data + purge 实现）
- StockTradebyZ `ml_models/training/train_v395_multi_target.py:2356-2500`（walk_forward_train 父类）
- StockTradebyZ `ml_models/ng/ng_trainer.py:1036-1080`（NG WF 训练入口）
- StockTradebyZ `ml_models/ng/ng_trainer.py:1248-1295`（CLI argparse）
- Lopez de Prado 2018, "Advances in Financial Machine Learning" — Purged K-Fold (Ch. 7)

---

*文档最后更新：2026-04-12*
