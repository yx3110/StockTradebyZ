# Signal Trust 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现选股信号可信度系统：为每日 Top-50 选股贴 🟢🟡🔴⚪ 标签，周度输出市值/行业/流动性分组的模型失效诊断。

**Architecture:** 新建 `signal_trust/` Python 包 + 两张 SQLite 缓存表（`signal_trust_samples` / `signal_trust_scores`）。首次扫全历史报告建库，每日增量 <5 秒。日报生成后 hook 进 `report_appender`。详细设计见 `docs/superpowers/specs/2026-04-12-signal-trust-design.md`。

**Tech Stack:** Python 3 + sqlite3（stdlib）+ pytest 9 + 项目现有 `data_adapter/stock_data.db`

**关键设计参数**（来自 spec，不要修改除非用户确认）：
- 样本池过滤：`pred_10d > 0.01`
- 颜色标签阈值：🟢 `hit≥0.55 AND bias≥-0.02 AND realize≥0.40`，🔴 `hit<0.45 OR bias<-0.03 OR realize<0.20`
- min_samples = 10
- 版本优先级（高→低）：`['ng106', 'ng1.0.6', 'ng101', 'ng1.0.1', 'ng100', 'ng1.0.0', 'v4901', 'v4.9.0.1', 'v490', 'v475', 'v473', 'v3.9', 'v39']`
- 市值分档：微盘<30亿、小盘30-100亿、中盘100-500亿、大盘>500亿（基于 `circ_mv`，单位：万元，所以对应 30_0000 / 100_0000 / 500_0000）
- 流动性分档：按 30 日日均成交额 `amount` 的 p25/p50/p75 动态分桶

---

## 项目关键事实（请先读）

1. **数据库表**：`data_adapter/stock_data.db`
   - `securities(id, code, name, industry, ...)` — `code` 带后缀如 `000001.SZ`；`industry` 字段在这里，不在 stock_basic_info
   - `daily_quotes(security_id, trade_date, close, amount, ...)` — 外键 `security_id → securities.id`
   - `daily_basic(security_id, trade_date, circ_mv, ...)` — `circ_mv` 单位为**万元**
   - 日期字段统一 `'YYYY-MM-DD'` 文本
2. **报告 JSON 格式**：
   - 路径：`reports/daily_selection_{version}/analysis_data_{YYYYMMDD}.json`
   - 顶层键：`{analysis_date, scoring_version, all_stocks_with_scores: [...]}`
   - 股票字段：`stock_code, stock_name, pred_3d, pred_5d, pred_10d, pred_15d, rank_score, ...`
3. **版本目录命名混乱**：同时存在 `daily_selection_ng106` 和 `daily_selection_ng1.0.6`。用目录名尾部字符串做版本 key。
4. **SQLite 并发规范**：所有连接必须 `cursor.execute('PRAGMA busy_timeout=30000')`（项目铁律）。
5. **pytest 已装**：项目用 pytest 9.0.2。

---

## 文件结构

**创建的文件**：
```
signal_trust/
├── __init__.py
├── constants.py                # 全部常量(阈值/版本优先级/分档)
├── db.py                       # DB 连接 + schema migration
├── sample_builder.py           # 报告扫描+去重+actual+分组+入库
├── scorer.py                   # 三指标聚合+泄露防护+标签
├── report_appender.py          # 日报 JSON 追加 trust_* 字段
├── global_stats.py             # 周度分组聚合+Markdown
└── tests/
    ├── __init__.py
    ├── conftest.py             # 临时 DB fixture + mock 报告工厂
    ├── test_db.py
    ├── test_sample_builder.py
    ├── test_scorer.py
    ├── test_report_appender.py
    ├── test_global_stats.py
    └── test_integration.py

scripts/
├── rebuild_signal_trust.py
├── update_signal_trust_daily.py
├── weekly_signal_trust_stats.py
└── validate_signal_trust.py

docs/wiki/architecture/signal-trust.md   # wiki 新条目
```

**修改的文件**：
- `run_daily_update.sh` — 选股报告生成后追加两行（贴标签 + 增量更新）
- `tomorrow_stock_selector.py:6103` 附近 — 可选 hook（在 JSON 写完后调用 `report_appender`）
- `docs/wiki/index.md` — 加入新条目
- `CLAUDE.md` — 在"Core Components"节增加 `signal_trust/` 说明

---

## Task 1: 包骨架 + 常量 + DB 迁移

**Files:**
- Create: `signal_trust/__init__.py`
- Create: `signal_trust/constants.py`
- Create: `signal_trust/db.py`
- Create: `signal_trust/tests/__init__.py`
- Create: `signal_trust/tests/conftest.py`
- Create: `signal_trust/tests/test_db.py`

- [ ] **Step 1: 创建常量文件**

Create `signal_trust/constants.py`:
```python
"""Signal Trust 全部常量。改阈值从这里改。"""

# 样本池过滤
PRED_THRESHOLD = 0.01  # pred_10d > 0.01 才纳入样本

# 可信度最小样本数
MIN_SAMPLES = 10

# 颜色标签阈值
GREEN_HIT_MIN = 0.55
GREEN_BIAS_MIN = -0.02
GREEN_REALIZE_MIN = 0.40
RED_HIT_MAX = 0.45
RED_BIAS_MAX = -0.03
RED_REALIZE_MAX = 0.20

# 标签字符串
TAG_GREEN = "🟢可信"
TAG_YELLOW = "🟡存疑"
TAG_RED = "🔴高风险"
TAG_NO_DATA = "⚪数据不足"

# ④ 高预测兑现率中"实际兑现"门槛
REALIZE_ACTUAL_THRESHOLD = 0.02  # actual_10d > 0.02 算兑现

# 持有期：10 个交易日
HOLD_DAYS = 10

# 市值分档（单位：万元，与 daily_basic.circ_mv 一致）
MARKET_CAP_BUCKETS = [
    (0, 30_0000, "微盘"),
    (30_0000, 100_0000, "小盘"),
    (100_0000, 500_0000, "中盘"),
    (500_0000, float("inf"), "大盘"),
]

# 版本优先级：同 (code, trade_date) 在多个目录出现时取最高优先级
# 新版本上线时加到列表顶部
VERSION_PRIORITY = [
    "ng106", "ng1.0.6",
    "ng101", "ng1.0.1",
    "ng100", "ng1.0.0",
    "v4901", "v4.9.0.1", "v490",
    "v475", "v473",
    "v3.9", "v39",
]

# 数据库路径（可被测试覆盖）
DEFAULT_DB_PATH = "data_adapter/stock_data.db"
```

- [ ] **Step 2: 创建 DB 迁移模块**

Create `signal_trust/db.py`:
```python
"""数据库连接与 schema 迁移。"""
import sqlite3
from pathlib import Path
from .constants import DEFAULT_DB_PATH


SAMPLES_SCHEMA = """
CREATE TABLE IF NOT EXISTS signal_trust_samples (
    code TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    sample_end_date TEXT NOT NULL,
    pred_10d REAL NOT NULL,
    actual_10d REAL,
    version TEXT NOT NULL,
    market_cap_bucket TEXT,
    industry TEXT,
    liquidity_bucket TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (code, trade_date)
);
"""

SAMPLES_INDICES = [
    "CREATE INDEX IF NOT EXISTS idx_sts_code ON signal_trust_samples(code);",
    "CREATE INDEX IF NOT EXISTS idx_sts_trade_date ON signal_trust_samples(trade_date);",
    "CREATE INDEX IF NOT EXISTS idx_sts_end_date ON signal_trust_samples(sample_end_date);",
    "CREATE INDEX IF NOT EXISTS idx_sts_mc ON signal_trust_samples(market_cap_bucket);",
    "CREATE INDEX IF NOT EXISTS idx_sts_ind ON signal_trust_samples(industry);",
    "CREATE INDEX IF NOT EXISTS idx_sts_liq ON signal_trust_samples(liquidity_bucket);",
]

SCORES_SCHEMA = """
CREATE TABLE IF NOT EXISTS signal_trust_scores (
    code TEXT PRIMARY KEY,
    as_of_date TEXT NOT NULL,
    n_samples INTEGER NOT NULL,
    direction_hit_rate REAL,
    systematic_bias REAL,
    high_pred_realize_rate REAL,
    trust_tag TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def connect(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """标准连接：busy_timeout + row_factory."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def migrate(db_path: str = DEFAULT_DB_PATH) -> None:
    """幂等创建两张表和所有索引。"""
    conn = connect(db_path)
    try:
        conn.execute(SAMPLES_SCHEMA)
        for sql in SAMPLES_INDICES:
            conn.execute(sql)
        conn.execute(SCORES_SCHEMA)
        conn.commit()
    finally:
        conn.close()
```

- [ ] **Step 3: 创建 `__init__.py`**

Create `signal_trust/__init__.py`:
```python
"""Signal Trust — 选股信号可信度验证系统。"""
from .constants import (
    PRED_THRESHOLD, MIN_SAMPLES, HOLD_DAYS,
    TAG_GREEN, TAG_YELLOW, TAG_RED, TAG_NO_DATA,
)
from .db import connect, migrate
__all__ = [
    "PRED_THRESHOLD", "MIN_SAMPLES", "HOLD_DAYS",
    "TAG_GREEN", "TAG_YELLOW", "TAG_RED", "TAG_NO_DATA",
    "connect", "migrate",
]
```

Create `signal_trust/tests/__init__.py` (empty).

- [ ] **Step 4: 创建测试 fixture**

Create `signal_trust/tests/conftest.py`:
```python
"""共享 fixtures：临时 DB + mock 报告工厂。"""
import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from signal_trust.db import connect, migrate


@pytest.fixture
def tmp_db(tmp_path):
    """临时 DB 文件，含 securities/daily_quotes/daily_basic 三张基础表 + signal_trust 两张表。"""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""CREATE TABLE securities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        code TEXT UNIQUE NOT NULL,
        name TEXT,
        industry TEXT
    )""")
    conn.execute("""CREATE TABLE daily_quotes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        security_id INTEGER NOT NULL,
        trade_date TEXT NOT NULL,
        close REAL,
        amount REAL,
        UNIQUE(security_id, trade_date)
    )""")
    conn.execute("""CREATE TABLE daily_basic (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        security_id INTEGER NOT NULL,
        trade_date TEXT NOT NULL,
        circ_mv REAL,
        UNIQUE(security_id, trade_date)
    )""")
    conn.commit()
    conn.close()
    migrate(str(db_path))
    return str(db_path)


def _seed_stock(db_path: str, code: str, industry: str,
                quotes: list[tuple[str, float, float]],
                circ_mv_by_date: dict[str, float] | None = None) -> int:
    """写入一只股票和它的日线。quotes: [(date, close, amount), ...]"""
    conn = connect(db_path)
    cur = conn.execute("INSERT INTO securities(code, industry) VALUES (?, ?)", (code, industry))
    sid = cur.lastrowid
    for d, close, amount in quotes:
        conn.execute(
            "INSERT INTO daily_quotes(security_id, trade_date, close, amount) VALUES (?, ?, ?, ?)",
            (sid, d, close, amount),
        )
    if circ_mv_by_date:
        for d, mv in circ_mv_by_date.items():
            conn.execute(
                "INSERT INTO daily_basic(security_id, trade_date, circ_mv) VALUES (?, ?, ?)",
                (sid, d, mv),
            )
    conn.commit()
    conn.close()
    return sid


@pytest.fixture
def seed_stock(tmp_db):
    def _factory(code: str, industry: str = "计算机",
                 quotes: list[tuple[str, float, float]] = None,
                 circ_mv_by_date: dict[str, float] | None = None) -> int:
        return _seed_stock(tmp_db, code, industry, quotes or [], circ_mv_by_date)
    return _factory


def _write_report(report_dir: Path, date: str, stocks: list[dict], version: str = "ng101"):
    """写一份 mock analysis_data JSON。stocks: [{stock_code, pred_10d, ...}, ...]"""
    d = report_dir / f"daily_selection_{version}"
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis_date": date,
        "scoring_version": version,
        "all_stocks_with_scores": stocks,
    }
    (d / f"analysis_data_{date.replace('-', '')}.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


@pytest.fixture
def write_report(tmp_path):
    reports_root = tmp_path / "reports"
    reports_root.mkdir(exist_ok=True)
    def _factory(date: str, stocks: list[dict], version: str = "ng101"):
        _write_report(reports_root, date, stocks, version)
        return reports_root / f"daily_selection_{version}"
    return _factory
```

- [ ] **Step 5: 写 DB migration 的失败测试**

Create `signal_trust/tests/test_db.py`:
```python
import sqlite3
from signal_trust.db import migrate, connect


def test_migrate_creates_both_tables(tmp_path):
    db_path = str(tmp_path / "new.db")
    migrate(db_path)
    conn = sqlite3.connect(db_path)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert "signal_trust_samples" in tables
    assert "signal_trust_scores" in tables


def test_migrate_idempotent(tmp_path):
    db_path = str(tmp_path / "new.db")
    migrate(db_path)
    migrate(db_path)  # 再跑一次不应报错
    conn = connect(db_path)
    # 能正常查询
    conn.execute("SELECT COUNT(*) FROM signal_trust_samples").fetchone()


def test_samples_primary_key(tmp_path):
    db_path = str(tmp_path / "new.db")
    migrate(db_path)
    conn = connect(db_path)
    conn.execute(
        "INSERT INTO signal_trust_samples(code, trade_date, sample_end_date, pred_10d, version) "
        "VALUES (?, ?, ?, ?, ?)",
        ("000001.SZ", "2026-01-01", "2026-01-15", 0.02, "ng101"),
    )
    # 同 (code, trade_date) 第二次应失败
    import pytest
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO signal_trust_samples(code, trade_date, sample_end_date, pred_10d, version) "
            "VALUES (?, ?, ?, ?, ?)",
            ("000001.SZ", "2026-01-01", "2026-01-15", 0.03, "ng106"),
        )
```

- [ ] **Step 6: 运行测试**

Run: `python3 -m pytest signal_trust/tests/test_db.py -v`
Expected: 3 passed

- [ ] **Step 7: Commit**

```bash
git add signal_trust/ docs/superpowers/plans/2026-04-12-signal-trust.md
git commit -m "feat(signal-trust): 包骨架+常量+DB迁移+测试 fixture"
```

---

## Task 2: sample_builder — 报告扫描 + pred 过滤 + 跨版本去重

**Files:**
- Create: `signal_trust/sample_builder.py` (只写 `scan_reports` 和 `dedupe` 两个纯函数)
- Create: `signal_trust/tests/test_sample_builder.py`

- [ ] **Step 1: 写失败测试**

Create `signal_trust/tests/test_sample_builder.py`:
```python
from pathlib import Path
from signal_trust.sample_builder import scan_reports, dedupe_by_version
from signal_trust.constants import PRED_THRESHOLD


def test_scan_reports_filters_threshold(write_report):
    report_dir = write_report(
        "2026-01-10",
        [
            {"stock_code": "000001.SZ", "pred_10d": 0.015},   # 入选
            {"stock_code": "000002.SZ", "pred_10d": 0.005},   # 低于阈值
            {"stock_code": "000003.SZ", "pred_10d": 0.0},     # 0 被过滤
            {"stock_code": "000004.SZ", "pred_10d": None},    # None 被过滤
            {"stock_code": "000005.SZ"},                       # 缺字段
        ],
    )
    records = list(scan_reports([str(report_dir.parent)]))
    codes = {r["code"] for r in records}
    assert codes == {"000001.SZ"}


def test_scan_reports_extracts_version_from_dir(write_report):
    d106 = write_report("2026-01-10", [{"stock_code": "A.SZ", "pred_10d": 0.02}], version="ng106")
    d101 = write_report("2026-01-10", [{"stock_code": "A.SZ", "pred_10d": 0.02}], version="ng101")
    records = list(scan_reports([str(d106.parent)]))
    versions = {r["version"] for r in records}
    assert versions == {"ng106", "ng101"}


def test_dedupe_keeps_highest_priority():
    records = [
        {"code": "A.SZ", "trade_date": "2026-01-10", "pred_10d": 0.02, "version": "ng101"},
        {"code": "A.SZ", "trade_date": "2026-01-10", "pred_10d": 0.03, "version": "ng106"},
        {"code": "A.SZ", "trade_date": "2026-01-10", "pred_10d": 0.025, "version": "v39"},
    ]
    out = dedupe_by_version(records)
    assert len(out) == 1
    assert out[0]["version"] == "ng106"
    assert out[0]["pred_10d"] == 0.03


def test_dedupe_unknown_version_lowest_priority():
    records = [
        {"code": "A.SZ", "trade_date": "2026-01-10", "pred_10d": 0.02, "version": "未知xyz"},
        {"code": "A.SZ", "trade_date": "2026-01-10", "pred_10d": 0.03, "version": "ng101"},
    ]
    out = dedupe_by_version(records)
    assert out[0]["version"] == "ng101"


def test_dedupe_preserves_different_dates():
    records = [
        {"code": "A.SZ", "trade_date": "2026-01-10", "pred_10d": 0.02, "version": "ng101"},
        {"code": "A.SZ", "trade_date": "2026-01-11", "pred_10d": 0.03, "version": "ng101"},
    ]
    out = dedupe_by_version(records)
    assert len(out) == 2
```

- [ ] **Step 2: 运行测试（应失败）**

Run: `python3 -m pytest signal_trust/tests/test_sample_builder.py -v`
Expected: ImportError 或 ModuleNotFoundError for `signal_trust.sample_builder`

- [ ] **Step 3: 实现 scan_reports + dedupe_by_version**

Create `signal_trust/sample_builder.py`:
```python
"""样本池构建器。"""
import json
import logging
import re
from pathlib import Path
from typing import Iterator
from .constants import PRED_THRESHOLD, VERSION_PRIORITY

logger = logging.getLogger(__name__)

# 优先级映射：越小越高
_PRIORITY_MAP = {v: i for i, v in enumerate(VERSION_PRIORITY)}
_UNKNOWN_PRIORITY = len(VERSION_PRIORITY) + 1

_DIR_VERSION_RE = re.compile(r"^daily_selection_(.+?)$")


def _extract_version(dir_name: str) -> str | None:
    m = _DIR_VERSION_RE.match(dir_name)
    if not m:
        return None
    version = m.group(1)
    # 剥离后缀如 _fullmarket / _pre2020 / _wf_oos / _fast / _ensemble_3seed
    for suffix in ("_fullmarket", "_pre2020", "_wf_oos", "_fast",
                   "_ensemble_3seed", "_ensemble_5seed", "_ensemble", "_90d", "_fixed"):
        if version.endswith(suffix):
            version = version[: -len(suffix)]
    return version


def scan_reports(report_parent_dirs: list[str]) -> Iterator[dict]:
    """
    扫描 `reports/` 下所有 `daily_selection_*` 子目录, 产出满足 pred_10d > 阈值的记录.
    
    report_parent_dirs: 形如 ['reports'] 的父目录列表(测试时用 tmp_path 对应目录).
    Yields: {code, trade_date, pred_10d, version}
    """
    for parent in report_parent_dirs:
        p = Path(parent)
        if not p.exists():
            logger.warning(f"目录不存在: {p}")
            continue
        for sub in sorted(p.iterdir()):
            if not sub.is_dir():
                continue
            version = _extract_version(sub.name)
            if not version:
                continue
            # 跳过带有分析后缀的报告目录(它们是离线实验, 不应污染生产可信度)
            if any(x in sub.name for x in ("_pre2020", "_wf_oos", "_fast")):
                continue
            for json_file in sorted(sub.glob("analysis_data_*.json")):
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(f"跳过坏文件 {json_file}: {e}")
                    continue
                trade_date = data.get("analysis_date")
                if not trade_date:
                    # 从文件名推断: analysis_data_20260412.json → 2026-04-12
                    m = re.search(r"(\d{4})(\d{2})(\d{2})", json_file.stem)
                    if m:
                        trade_date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                    else:
                        continue
                for stock in data.get("all_stocks_with_scores", []):
                    pred = stock.get("pred_10d")
                    try:
                        pred_val = float(pred) if pred is not None else 0.0
                    except (TypeError, ValueError):
                        continue
                    if pred_val <= PRED_THRESHOLD:
                        continue
                    code = stock.get("stock_code")
                    if not code:
                        continue
                    yield {
                        "code": code,
                        "trade_date": trade_date,
                        "pred_10d": pred_val,
                        "version": version,
                    }


def dedupe_by_version(records: list[dict]) -> list[dict]:
    """按 (code, trade_date) 去重, 保留版本优先级最高的记录."""
    best: dict[tuple[str, str], dict] = {}
    for r in records:
        key = (r["code"], r["trade_date"])
        cur = best.get(key)
        new_p = _PRIORITY_MAP.get(r["version"], _UNKNOWN_PRIORITY)
        if cur is None or new_p < _PRIORITY_MAP.get(cur["version"], _UNKNOWN_PRIORITY):
            best[key] = r
    return list(best.values())
```

- [ ] **Step 4: 运行测试**

Run: `python3 -m pytest signal_trust/tests/test_sample_builder.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add signal_trust/sample_builder.py signal_trust/tests/test_sample_builder.py
git commit -m "feat(signal-trust): 报告扫描+阈值过滤+跨版本去重"
```

---

## Task 3: sample_builder — actual_10d 计算

**Files:**
- Modify: `signal_trust/sample_builder.py` (新增 `compute_actual_10d`)
- Modify: `signal_trust/tests/test_sample_builder.py`

- [ ] **Step 1: 写失败测试（追加）**

追加到 `signal_trust/tests/test_sample_builder.py`：
```python
from signal_trust.sample_builder import compute_actual_10d
from signal_trust.db import connect


def test_actual_10d_normal_case(seed_stock, tmp_db):
    # 10 个交易日后, close 上涨 5%
    quotes = []
    start_close = 100.0
    for i in range(15):
        d = f"2026-01-{i+1:02d}"
        quotes.append((d, start_close * (1 + 0.005 * i), 1e8))
    seed_stock("A.SZ", "计算机", quotes)
    # T = 2026-01-01 (idx 0), T+10 trading days = 2026-01-11 (idx 10)
    actual = compute_actual_10d(tmp_db, "A.SZ", "2026-01-01")
    expected = (quotes[10][1] - quotes[0][1]) / quotes[0][1]
    assert abs(actual - expected) < 1e-9


def test_actual_10d_missing_future_returns_none(seed_stock, tmp_db):
    # 只有 5 个交易日数据, T+10 不存在
    quotes = [(f"2026-01-{i+1:02d}", 100.0 + i, 1e8) for i in range(5)]
    seed_stock("A.SZ", "计算机", quotes)
    assert compute_actual_10d(tmp_db, "A.SZ", "2026-01-01") is None


def test_actual_10d_suspended_days_still_count(seed_stock, tmp_db):
    # 数据库里只存在交易日(停牌日无记录). T+10 指的是"数据库里第 10 个后续交易日"
    quotes = [(f"2026-01-{i+1:02d}", 100.0 + i * 2, 1e8) for i in range(12)]
    seed_stock("A.SZ", "计算机", quotes)
    actual = compute_actual_10d(tmp_db, "A.SZ", "2026-01-01")
    expected = (quotes[10][1] - quotes[0][1]) / quotes[0][1]
    assert abs(actual - expected) < 1e-9


def test_actual_10d_stock_not_found(tmp_db):
    assert compute_actual_10d(tmp_db, "NONEXIST.SZ", "2026-01-01") is None
```

- [ ] **Step 2: 运行测试（应失败）**

Run: `python3 -m pytest signal_trust/tests/test_sample_builder.py -v -k actual`
Expected: 4 failures (ImportError for `compute_actual_10d`)

- [ ] **Step 3: 实现 compute_actual_10d**

Append to `signal_trust/sample_builder.py`:
```python
from .db import connect
from .constants import HOLD_DAYS


def compute_actual_10d(db_path: str, code: str, trade_date: str) -> float | None:
    """
    用 daily_quotes 查 T 日 close 和 T+HOLD_DAYS(默认10)个交易日后的 close.
    返回实际收益率, 查不到返回 None.
    """
    conn = connect(db_path)
    try:
        row = conn.execute(
            "SELECT s.id FROM securities s WHERE s.code = ?", (code,)
        ).fetchone()
        if row is None:
            return None
        sid = row["id"]
        quotes = conn.execute(
            "SELECT trade_date, close FROM daily_quotes "
            "WHERE security_id = ? AND trade_date >= ? "
            "ORDER BY trade_date ASC LIMIT ?",
            (sid, trade_date, HOLD_DAYS + 1),
        ).fetchall()
        if len(quotes) < HOLD_DAYS + 1:
            return None
        if quotes[0]["trade_date"] != trade_date:
            # T 日本身也需是交易日
            return None
        p0 = quotes[0]["close"]
        pN = quotes[HOLD_DAYS]["close"]
        if p0 is None or pN is None or p0 == 0:
            return None
        return (pN - p0) / p0
    finally:
        conn.close()


def compute_sample_end_date(db_path: str, trade_date: str) -> str | None:
    """
    给定 trade_date, 返回 sample_end_date = 市场第 HOLD_DAYS 个交易日后的日期.
    用任意一只活跃股票的 daily_quotes 推(这些大盘行情日期一致).
    若无法确定(如未来日期), 返回 None.
    """
    conn = connect(db_path)
    try:
        row = conn.execute(
            "SELECT DISTINCT trade_date FROM daily_quotes "
            "WHERE trade_date >= ? ORDER BY trade_date ASC LIMIT ?",
            (trade_date, HOLD_DAYS + 1),
        ).fetchall()
        if len(row) < HOLD_DAYS + 1:
            return None
        return row[HOLD_DAYS]["trade_date"]
    finally:
        conn.close()
```

- [ ] **Step 4: 运行测试**

Run: `python3 -m pytest signal_trust/tests/test_sample_builder.py -v`
Expected: 9 passed (5 old + 4 new)

- [ ] **Step 5: Commit**

```bash
git add signal_trust/sample_builder.py signal_trust/tests/test_sample_builder.py
git commit -m "feat(signal-trust): actual_10d 计算含停牌/未到期处理"
```

---

## Task 4: sample_builder — 分组标签冻结 + 入库

**Files:**
- Modify: `signal_trust/sample_builder.py`
- Modify: `signal_trust/tests/test_sample_builder.py`

- [ ] **Step 1: 写失败测试（追加）**

Append to test file:
```python
from signal_trust.sample_builder import (
    compute_market_cap_bucket, compute_liquidity_bucket, upsert_samples,
)


def test_market_cap_bucket_thresholds(seed_stock, tmp_db):
    # daily_basic.circ_mv 单位是万元, 30亿 = 30_0000 万元
    seed_stock("A.SZ", "计算机",
               quotes=[("2026-01-01", 100, 1e8)],
               circ_mv_by_date={"2026-01-01": 20_0000})  # 20亿 → 微盘
    seed_stock("B.SZ", "计算机",
               quotes=[("2026-01-01", 100, 1e8)],
               circ_mv_by_date={"2026-01-01": 50_0000})  # 50亿 → 小盘
    seed_stock("C.SZ", "计算机",
               quotes=[("2026-01-01", 100, 1e8)],
               circ_mv_by_date={"2026-01-01": 200_0000})  # 200亿 → 中盘
    seed_stock("D.SZ", "计算机",
               quotes=[("2026-01-01", 100, 1e8)],
               circ_mv_by_date={"2026-01-01": 800_0000})  # 800亿 → 大盘
    assert compute_market_cap_bucket(tmp_db, "A.SZ", "2026-01-01") == "微盘"
    assert compute_market_cap_bucket(tmp_db, "B.SZ", "2026-01-01") == "小盘"
    assert compute_market_cap_bucket(tmp_db, "C.SZ", "2026-01-01") == "中盘"
    assert compute_market_cap_bucket(tmp_db, "D.SZ", "2026-01-01") == "大盘"


def test_market_cap_bucket_missing_data(seed_stock, tmp_db):
    seed_stock("X.SZ", "计算机", quotes=[("2026-01-01", 100, 1e8)])
    assert compute_market_cap_bucket(tmp_db, "X.SZ", "2026-01-01") == "未知"


def test_liquidity_bucket_uses_30day_mean(seed_stock, tmp_db):
    # 股票 X 的 30 日均成交额 5e7, 股票 Y 的 30 日均成交额 5e9
    qx = [(f"2026-01-{i+1:02d}", 100, 5e7) for i in range(30)]
    qy = [(f"2026-01-{i+1:02d}", 100, 5e9) for i in range(30)]
    seed_stock("X.SZ", "计算机", qx)
    seed_stock("Y.SZ", "计算机", qy)
    thresholds = (1e8, 3e8, 1e9)  # p25/p50/p75 假设阈值
    bx = compute_liquidity_bucket(tmp_db, "X.SZ", "2026-01-30", thresholds)
    by = compute_liquidity_bucket(tmp_db, "Y.SZ", "2026-01-30", thresholds)
    assert bx == "低"
    assert by == "高"


def test_upsert_samples_idempotent(tmp_db):
    rows = [{
        "code": "A.SZ", "trade_date": "2026-01-10", "sample_end_date": "2026-01-24",
        "pred_10d": 0.015, "actual_10d": 0.02, "version": "ng106",
        "market_cap_bucket": "小盘", "industry": "计算机", "liquidity_bucket": "中高",
    }]
    upsert_samples(tmp_db, rows)
    upsert_samples(tmp_db, rows)  # 重跑
    conn = connect(tmp_db)
    (n,) = conn.execute("SELECT COUNT(*) FROM signal_trust_samples").fetchone()
    assert n == 1


def test_upsert_backfills_actual_10d(tmp_db):
    # 首次入库 actual_10d=None, 二次更新应覆盖
    rows1 = [{
        "code": "A.SZ", "trade_date": "2026-01-10", "sample_end_date": "2026-01-24",
        "pred_10d": 0.015, "actual_10d": None, "version": "ng106",
        "market_cap_bucket": "小盘", "industry": "计算机", "liquidity_bucket": "中高",
    }]
    upsert_samples(tmp_db, rows1)
    rows2 = [{**rows1[0], "actual_10d": 0.025}]
    upsert_samples(tmp_db, rows2, update_actual=True)
    conn = connect(tmp_db)
    row = conn.execute(
        "SELECT actual_10d FROM signal_trust_samples WHERE code='A.SZ'"
    ).fetchone()
    assert abs(row["actual_10d"] - 0.025) < 1e-9
```

- [ ] **Step 2: 运行测试（应失败）**

Run: `python3 -m pytest signal_trust/tests/test_sample_builder.py -v -k "bucket or upsert"`
Expected: 5 failures

- [ ] **Step 3: 实现分组 + upsert**

Append to `signal_trust/sample_builder.py`:
```python
from .constants import MARKET_CAP_BUCKETS


def compute_market_cap_bucket(db_path: str, code: str, trade_date: str) -> str:
    conn = connect(db_path)
    try:
        row = conn.execute(
            "SELECT db.circ_mv FROM daily_basic db "
            "JOIN securities s ON s.id = db.security_id "
            "WHERE s.code = ? AND db.trade_date = ?",
            (code, trade_date),
        ).fetchone()
        if row is None or row["circ_mv"] is None:
            return "未知"
        mv = row["circ_mv"]
        for lo, hi, label in MARKET_CAP_BUCKETS:
            if lo <= mv < hi:
                return label
        return "未知"
    finally:
        conn.close()


def compute_liquidity_bucket(
    db_path: str, code: str, trade_date: str, thresholds: tuple[float, float, float]
) -> str:
    """thresholds = (p25, p50, p75); 基于该股 30 日日均成交额."""
    conn = connect(db_path)
    try:
        row = conn.execute(
            "SELECT AVG(amount) AS m FROM daily_quotes dq "
            "JOIN securities s ON s.id = dq.security_id "
            "WHERE s.code = ? AND dq.trade_date <= ? "
            "ORDER BY dq.trade_date DESC LIMIT 30",
            (code, trade_date),
        ).fetchone()
        if row is None or row["m"] is None:
            return "未知"
        m = row["m"]
        p25, p50, p75 = thresholds
        if m < p25:
            return "低"
        elif m < p50:
            return "中低"
        elif m < p75:
            return "中高"
        else:
            return "高"
    finally:
        conn.close()


def compute_liquidity_thresholds(db_path: str, as_of_date: str) -> tuple[float, float, float]:
    """基于所有股票的 30 日均成交额 p25/p50/p75."""
    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT security_id, AVG(amount) AS m FROM daily_quotes "
            "WHERE trade_date <= ? AND trade_date >= date(?, '-45 days') "
            "GROUP BY security_id HAVING COUNT(*) >= 15",
            (as_of_date, as_of_date),
        ).fetchall()
        vals = sorted(float(r["m"]) for r in rows if r["m"] is not None)
        if len(vals) < 4:
            return (1e8, 3e8, 1e9)  # 回退默认
        n = len(vals)
        return (vals[n // 4], vals[n // 2], vals[3 * n // 4])
    finally:
        conn.close()


def _industry_lookup(db_path: str, codes: list[str]) -> dict[str, str]:
    if not codes:
        return {}
    conn = connect(db_path)
    try:
        placeholders = ",".join("?" * len(codes))
        rows = conn.execute(
            f"SELECT code, industry FROM securities WHERE code IN ({placeholders})",
            codes,
        ).fetchall()
        return {r["code"]: (r["industry"] or "未分类") for r in rows}
    finally:
        conn.close()


def upsert_samples(db_path: str, rows: list[dict], update_actual: bool = False) -> int:
    """
    写入样本表. 幂等.
    update_actual=True 时允许覆盖已有的 actual_10d(用于回填).
    """
    if not rows:
        return 0
    conn = connect(db_path)
    try:
        n = 0
        for r in rows:
            if update_actual:
                conn.execute(
                    "INSERT INTO signal_trust_samples "
                    "(code, trade_date, sample_end_date, pred_10d, actual_10d, version, "
                    " market_cap_bucket, industry, liquidity_bucket) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(code, trade_date) DO UPDATE SET "
                    "  actual_10d = excluded.actual_10d",
                    (r["code"], r["trade_date"], r["sample_end_date"],
                     r["pred_10d"], r.get("actual_10d"), r["version"],
                     r.get("market_cap_bucket"), r.get("industry"), r.get("liquidity_bucket")),
                )
            else:
                conn.execute(
                    "INSERT OR IGNORE INTO signal_trust_samples "
                    "(code, trade_date, sample_end_date, pred_10d, actual_10d, version, "
                    " market_cap_bucket, industry, liquidity_bucket) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (r["code"], r["trade_date"], r["sample_end_date"],
                     r["pred_10d"], r.get("actual_10d"), r["version"],
                     r.get("market_cap_bucket"), r.get("industry"), r.get("liquidity_bucket")),
                )
            n += 1
        conn.commit()
        return n
    finally:
        conn.close()
```

- [ ] **Step 4: 运行测试**

Run: `python3 -m pytest signal_trust/tests/test_sample_builder.py -v`
Expected: 14 passed (9 old + 5 new)

- [ ] **Step 5: Commit**

```bash
git add signal_trust/sample_builder.py signal_trust/tests/test_sample_builder.py
git commit -m "feat(signal-trust): 分组标签(市值/行业/流动性)+幂等入库"
```

---

## Task 5: scorer — 三指标 + 泄露防护 + 标签

**Files:**
- Create: `signal_trust/scorer.py`
- Create: `signal_trust/tests/test_scorer.py`

- [ ] **Step 1: 写失败测试**

Create `signal_trust/tests/test_scorer.py`:
```python
from signal_trust.scorer import compute_scores, trust_tag
from signal_trust.constants import (
    TAG_GREEN, TAG_YELLOW, TAG_RED, TAG_NO_DATA,
)
from signal_trust.db import connect


def _insert_samples(db_path: str, rows: list[dict]):
    conn = connect(db_path)
    for r in rows:
        conn.execute(
            "INSERT INTO signal_trust_samples "
            "(code, trade_date, sample_end_date, pred_10d, actual_10d, version) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (r["code"], r["trade_date"], r["sample_end_date"],
             r["pred_10d"], r.get("actual_10d"), r.get("version", "ng106")),
        )
    conn.commit()
    conn.close()


def test_min_samples_returns_no_data_tag(tmp_db):
    # 只有 5 个样本 < 10
    rows = [{
        "code": "A.SZ", "trade_date": f"2026-01-{i+1:02d}",
        "sample_end_date": f"2026-01-{i+15:02d}",
        "pred_10d": 0.02, "actual_10d": 0.03,
    } for i in range(5)]
    _insert_samples(tmp_db, rows)
    scores = compute_scores(tmp_db, as_of_date="2026-03-01")
    assert scores["A.SZ"]["trust_tag"] == TAG_NO_DATA
    assert scores["A.SZ"]["n_samples"] == 5


def test_three_metrics_correct(tmp_db):
    # 10 个样本: 方向命中 7/10 = 0.7, bias = -0.01, realize rate actual>0.02 4/10=0.4
    rows = []
    data = [
        (0.02,  0.03),  # dir=1, hit, bias=+0.01, realize=yes(0.03>0.02)
        (0.02,  0.03),
        (0.02,  0.03),
        (0.02,  0.025),
        (0.02,  0.01),  # dir=1, hit, bias=-0.01, realize=no
        (0.02,  0.005),
        (0.02,  0.005),
        (0.02, -0.01),  # dir=1, miss(actual negative), bias=-0.03
        (0.02, -0.02),
        (0.02, -0.03),
    ]
    for i, (p, a) in enumerate(data):
        rows.append({
            "code": "B.SZ", "trade_date": f"2026-01-{i+1:02d}",
            "sample_end_date": f"2026-01-{i+15:02d}",
            "pred_10d": p, "actual_10d": a,
        })
    _insert_samples(tmp_db, rows)
    scores = compute_scores(tmp_db, as_of_date="2026-03-01")
    s = scores["B.SZ"]
    assert s["n_samples"] == 10
    assert abs(s["direction_hit_rate"] - 0.7) < 1e-9
    assert abs(s["systematic_bias"] - (sum(a - p for p, a in data) / 10)) < 1e-9
    assert abs(s["high_pred_realize_rate"] - 0.4) < 1e-9


def test_leakage_prevention_excludes_unripe_samples(tmp_db):
    """🚨 核心测试: 样本 sample_end_date >= as_of_date 的不参与计算."""
    base = [{
        "code": "C.SZ", "trade_date": f"2025-{m:02d}-10",
        "sample_end_date": f"2025-{m:02d}-24",
        "pred_10d": 0.02, "actual_10d": 0.03,
    } for m in range(1, 12)]  # 11 个已过期样本
    future = [{
        "code": "C.SZ", "trade_date": "2026-04-10",
        "sample_end_date": "2026-04-26",  # 在 as_of_date=2026-04-12 之后
        "pred_10d": 0.02, "actual_10d": -0.05,  # 极端负值: 若泄露会显著拉低指标
    }]
    _insert_samples(tmp_db, base + future)
    scores = compute_scores(tmp_db, as_of_date="2026-04-12")
    s = scores["C.SZ"]
    assert s["n_samples"] == 11  # future 被排除
    assert s["direction_hit_rate"] == 1.0  # 所有历史样本命中


def test_excludes_null_actual(tmp_db):
    rows = [{"code": "D.SZ", "trade_date": f"2026-01-{i+1:02d}",
             "sample_end_date": f"2026-01-{i+15:02d}",
             "pred_10d": 0.02, "actual_10d": 0.03 if i < 10 else None}
            for i in range(12)]
    _insert_samples(tmp_db, rows)
    scores = compute_scores(tmp_db, as_of_date="2026-03-01")
    assert scores["D.SZ"]["n_samples"] == 10  # 2 条 NULL 被排除


def test_trust_tag_green():
    assert trust_tag(hit=0.60, bias=-0.01, realize=0.50, n=20) == TAG_GREEN


def test_trust_tag_red_any_condition():
    assert trust_tag(hit=0.40, bias=0.0, realize=0.5, n=20) == TAG_RED  # hit 触发
    assert trust_tag(hit=0.60, bias=-0.05, realize=0.5, n=20) == TAG_RED  # bias
    assert trust_tag(hit=0.60, bias=0.0, realize=0.10, n=20) == TAG_RED  # realize


def test_trust_tag_yellow_middle():
    assert trust_tag(hit=0.50, bias=-0.025, realize=0.30, n=20) == TAG_YELLOW


def test_trust_tag_no_data_below_min_samples():
    assert trust_tag(hit=0.99, bias=0.0, realize=0.99, n=8) == TAG_NO_DATA
```

- [ ] **Step 2: 运行测试（应失败）**

Run: `python3 -m pytest signal_trust/tests/test_scorer.py -v`
Expected: ImportError

- [ ] **Step 3: 实现 scorer**

Create `signal_trust/scorer.py`:
```python
"""可信度聚合器."""
import logging
from .db import connect
from .constants import (
    MIN_SAMPLES, REALIZE_ACTUAL_THRESHOLD,
    GREEN_HIT_MIN, GREEN_BIAS_MIN, GREEN_REALIZE_MIN,
    RED_HIT_MAX, RED_BIAS_MAX, RED_REALIZE_MAX,
    TAG_GREEN, TAG_YELLOW, TAG_RED, TAG_NO_DATA,
)

logger = logging.getLogger(__name__)


def trust_tag(hit: float | None, bias: float | None, realize: float | None, n: int) -> str:
    if n < MIN_SAMPLES:
        return TAG_NO_DATA
    if hit is None or bias is None or realize is None:
        return TAG_NO_DATA
    if hit < RED_HIT_MAX or bias < RED_BIAS_MAX or realize < RED_REALIZE_MAX:
        return TAG_RED
    if hit >= GREEN_HIT_MIN and bias >= GREEN_BIAS_MIN and realize >= GREEN_REALIZE_MIN:
        return TAG_GREEN
    return TAG_YELLOW


def compute_scores(db_path: str, as_of_date: str, codes: list[str] | None = None) -> dict[str, dict]:
    """
    对 codes 列表(None=全库)每只股票聚合三指标 + 标签. 返回 {code: {...}}.
    泄露防护: WHERE sample_end_date < as_of_date AND actual_10d IS NOT NULL.
    """
    conn = connect(db_path)
    try:
        params: list = [as_of_date]
        where = "sample_end_date < ? AND actual_10d IS NOT NULL"
        if codes is not None:
            if not codes:
                return {}
            placeholders = ",".join("?" * len(codes))
            where += f" AND code IN ({placeholders})"
            params.extend(codes)
        # 一次查所有样本, Python 端聚合(样本量百万级也 OK)
        rows = conn.execute(
            f"SELECT code, pred_10d, actual_10d FROM signal_trust_samples WHERE {where}",
            params,
        ).fetchall()
    finally:
        conn.close()

    by_code: dict[str, list] = {}
    for r in rows:
        by_code.setdefault(r["code"], []).append((r["pred_10d"], r["actual_10d"]))

    result: dict[str, dict] = {}
    for code, pairs in by_code.items():
        n = len(pairs)
        if n == 0:
            continue
        hits = sum(1 for p, a in pairs if (p > 0) == (a > 0))
        biases = [a - p for p, a in pairs]
        realizes = sum(1 for _, a in pairs if a > REALIZE_ACTUAL_THRESHOLD)
        hit = hits / n
        bias = sum(biases) / n
        realize = realizes / n
        result[code] = {
            "code": code,
            "as_of_date": as_of_date,
            "n_samples": n,
            "direction_hit_rate": hit,
            "systematic_bias": bias,
            "high_pred_realize_rate": realize,
            "trust_tag": trust_tag(hit, bias, realize, n),
        }
    # 对 codes 中有指定但无样本的, 写一个 no-data 记录
    if codes is not None:
        for c in codes:
            if c not in result:
                result[c] = {
                    "code": c, "as_of_date": as_of_date, "n_samples": 0,
                    "direction_hit_rate": None, "systematic_bias": None,
                    "high_pred_realize_rate": None, "trust_tag": TAG_NO_DATA,
                }
    return result


def upsert_scores(db_path: str, scores: dict[str, dict]) -> int:
    if not scores:
        return 0
    conn = connect(db_path)
    try:
        for s in scores.values():
            conn.execute(
                "INSERT OR REPLACE INTO signal_trust_scores "
                "(code, as_of_date, n_samples, direction_hit_rate, systematic_bias, "
                " high_pred_realize_rate, trust_tag, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))",
                (s["code"], s["as_of_date"], s["n_samples"],
                 s["direction_hit_rate"], s["systematic_bias"],
                 s["high_pred_realize_rate"], s["trust_tag"]),
            )
        conn.commit()
        return len(scores)
    finally:
        conn.close()
```

- [ ] **Step 4: 运行测试**

Run: `python3 -m pytest signal_trust/tests/test_scorer.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add signal_trust/scorer.py signal_trust/tests/test_scorer.py
git commit -m "feat(signal-trust): 三指标+泄露防护+颜色标签"
```

---

## Task 6: report_appender — 日报贴标签 + 原子写

**Files:**
- Create: `signal_trust/report_appender.py`
- Create: `signal_trust/tests/test_report_appender.py`

- [ ] **Step 1: 写失败测试**

Create `signal_trust/tests/test_report_appender.py`:
```python
import json
from pathlib import Path

from signal_trust.report_appender import append_trust_tags
from signal_trust.db import connect
from signal_trust.constants import TAG_GREEN, TAG_NO_DATA


def _write_report_file(path: Path, stocks: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "analysis_date": "2026-04-12",
        "all_stocks_with_scores": stocks,
    }, ensure_ascii=False), encoding="utf-8")


def _seed_score(db_path: str, code: str, tag: str, n: int):
    conn = connect(db_path)
    conn.execute(
        "INSERT INTO signal_trust_scores "
        "(code, as_of_date, n_samples, direction_hit_rate, systematic_bias, "
        " high_pred_realize_rate, trust_tag) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (code, "2026-04-12", n, 0.6, -0.01, 0.5, tag),
    )
    conn.commit()
    conn.close()


def test_appends_trust_fields(tmp_db, tmp_path):
    report = tmp_path / "report.json"
    _write_report_file(report, [
        {"stock_code": "A.SZ", "rank_score": 95, "pred_10d": 0.015},
        {"stock_code": "B.SZ", "rank_score": 90, "pred_10d": 0.012},
    ])
    _seed_score(tmp_db, "A.SZ", TAG_GREEN, 42)
    n = append_trust_tags(str(report), db_path=tmp_db, top_n=50)
    assert n == 2
    data = json.loads(report.read_text(encoding="utf-8"))
    a = next(s for s in data["all_stocks_with_scores"] if s["stock_code"] == "A.SZ")
    b = next(s for s in data["all_stocks_with_scores"] if s["stock_code"] == "B.SZ")
    assert a["trust_tag"] == TAG_GREEN
    assert a["trust_samples"] == 42
    assert a["trust_details"]["direction_hit_rate"] == 0.6
    assert b["trust_tag"] == TAG_NO_DATA
    assert b["trust_samples"] == 0


def test_only_top_n_tagged(tmp_db, tmp_path):
    report = tmp_path / "r.json"
    stocks = [{"stock_code": f"S{i}.SZ", "rank_score": 100 - i, "pred_10d": 0.015}
              for i in range(100)]
    _write_report_file(report, stocks)
    _seed_score(tmp_db, "S0.SZ", TAG_GREEN, 20)
    _seed_score(tmp_db, "S99.SZ", TAG_GREEN, 20)
    n = append_trust_tags(str(report), db_path=tmp_db, top_n=10)
    assert n == 10
    data = json.loads(report.read_text(encoding="utf-8"))
    # 前 10 应有 trust_tag, 后 90 没有
    tagged = [s for s in data["all_stocks_with_scores"] if "trust_tag" in s]
    assert len(tagged) == 10


def test_missing_scores_table_graceful(tmp_path):
    """scores 表不存在时不应抛. 为日报流程容错保命."""
    # 不经 tmp_db fixture, 用一个空目录里的空 db
    import sqlite3
    db_path = tmp_path / "empty.db"
    sqlite3.connect(str(db_path)).close()
    report = tmp_path / "r.json"
    _write_report_file(report, [{"stock_code": "A.SZ", "rank_score": 100, "pred_10d": 0.015}])
    n = append_trust_tags(str(report), db_path=str(db_path), top_n=50)
    # 返回 0 或不崩溃即可, JSON 保留原样
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["all_stocks_with_scores"][0]["stock_code"] == "A.SZ"


def test_atomic_write_on_crash(tmp_db, tmp_path, monkeypatch):
    """写中途失败原文件保留."""
    report = tmp_path / "r.json"
    original_content = json.dumps({
        "analysis_date": "2026-04-12",
        "all_stocks_with_scores": [{"stock_code": "A.SZ", "rank_score": 100, "pred_10d": 0.015}],
    }, ensure_ascii=False)
    report.write_text(original_content, encoding="utf-8")
    _seed_score(tmp_db, "A.SZ", TAG_GREEN, 20)

    # patch Path.replace 抛异常
    import signal_trust.report_appender as mod
    def boom(self, target):
        raise OSError("disk full")
    monkeypatch.setattr(Path, "replace", boom)
    import pytest
    with pytest.raises(OSError):
        append_trust_tags(str(report), db_path=tmp_db, top_n=50)
    # 原文件应未变
    assert report.read_text(encoding="utf-8") == original_content
```

- [ ] **Step 2: 运行测试（应失败）**

Run: `python3 -m pytest signal_trust/tests/test_report_appender.py -v`
Expected: ImportError

- [ ] **Step 3: 实现 report_appender**

Create `signal_trust/report_appender.py`:
```python
"""为日报 JSON 追加 trust_* 字段."""
import json
import logging
import sqlite3
from pathlib import Path

from .db import connect
from .constants import TAG_NO_DATA

logger = logging.getLogger(__name__)


def _query_scores(db_path: str, codes: list[str]) -> dict[str, dict]:
    if not codes:
        return {}
    try:
        conn = connect(db_path)
    except sqlite3.OperationalError as e:
        logger.warning(f"连接 DB 失败: {e}")
        return {}
    try:
        placeholders = ",".join("?" * len(codes))
        try:
            rows = conn.execute(
                f"SELECT * FROM signal_trust_scores WHERE code IN ({placeholders})",
                codes,
            ).fetchall()
        except sqlite3.OperationalError as e:
            # scores 表不存在 → 优雅降级
            logger.warning(f"signal_trust_scores 表不可用: {e}")
            return {}
        return {r["code"]: dict(r) for r in rows}
    finally:
        conn.close()


def append_trust_tags(report_json_path: str, db_path: str, top_n: int = 50) -> int:
    """
    为日报 JSON 的 Top-N 股票追加 trust_tag/trust_samples/trust_details.
    原子写 (临时文件 + rename).
    返回被追加的股票数.
    """
    p = Path(report_json_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    stocks = data.get("all_stocks_with_scores", [])
    # 按 rank_score 倒序取 Top-N
    ranked = sorted(
        stocks,
        key=lambda s: float(s.get("rank_score", 0) or 0),
        reverse=True,
    )[:top_n]
    codes = [s["stock_code"] for s in ranked if "stock_code" in s]
    scores = _query_scores(db_path, codes)

    for s in ranked:
        code = s.get("stock_code")
        if not code:
            continue
        sc = scores.get(code)
        if sc is None:
            s["trust_tag"] = TAG_NO_DATA
            s["trust_samples"] = 0
            s["trust_details"] = None
        else:
            s["trust_tag"] = sc["trust_tag"]
            s["trust_samples"] = sc["n_samples"]
            s["trust_details"] = {
                "direction_hit_rate": sc["direction_hit_rate"],
                "systematic_bias": sc["systematic_bias"],
                "high_pred_realize_rate": sc["high_pred_realize_rate"],
                "as_of_date": sc["as_of_date"],
            }

    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(p)
    return len(ranked)
```

- [ ] **Step 4: 运行测试**

Run: `python3 -m pytest signal_trust/tests/test_report_appender.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add signal_trust/report_appender.py signal_trust/tests/test_report_appender.py
git commit -m "feat(signal-trust): 日报 JSON 贴标签+原子写"
```

---

## Task 7: global_stats — 分组聚合 + Markdown

**Files:**
- Create: `signal_trust/global_stats.py`
- Create: `signal_trust/tests/test_global_stats.py`

- [ ] **Step 1: 写失败测试**

Create `signal_trust/tests/test_global_stats.py`:
```python
from signal_trust.global_stats import (
    aggregate_by_bucket, format_markdown_report,
)
from signal_trust.db import connect


def _seed(db_path: str, rows: list[dict]):
    conn = connect(db_path)
    for r in rows:
        conn.execute(
            "INSERT INTO signal_trust_samples "
            "(code, trade_date, sample_end_date, pred_10d, actual_10d, version, "
            " market_cap_bucket, industry, liquidity_bucket) "
            "VALUES (?, ?, ?, ?, ?, 'ng106', ?, ?, ?)",
            (r["code"], r["trade_date"], r["sample_end_date"],
             r["pred_10d"], r["actual_10d"],
             r.get("market_cap_bucket"), r.get("industry"), r.get("liquidity_bucket")),
        )
    conn.commit()
    conn.close()


def test_aggregate_by_market_cap(tmp_db):
    rows = []
    # 微盘 5 条 方向命中率 0.4 (2/5)
    for i in range(5):
        rows.append({
            "code": f"M{i}.SZ", "trade_date": f"2025-01-{i+1:02d}",
            "sample_end_date": f"2025-01-{i+15:02d}",
            "pred_10d": 0.02, "actual_10d": 0.02 if i < 2 else -0.01,
            "market_cap_bucket": "微盘", "industry": "计算机", "liquidity_bucket": "低",
        })
    # 大盘 10 条 方向命中率 0.9 (9/10)
    for i in range(10):
        rows.append({
            "code": f"L{i}.SZ", "trade_date": f"2025-02-{i+1:02d}",
            "sample_end_date": f"2025-02-{i+15:02d}",
            "pred_10d": 0.02, "actual_10d": 0.03 if i < 9 else -0.01,
            "market_cap_bucket": "大盘", "industry": "银行", "liquidity_bucket": "高",
        })
    _seed(tmp_db, rows)
    agg = aggregate_by_bucket(tmp_db, "market_cap_bucket", as_of_date="2026-04-12")
    by_b = {r["bucket"]: r for r in agg}
    assert by_b["微盘"]["n_samples"] == 5
    assert abs(by_b["微盘"]["direction_hit_rate"] - 0.4) < 1e-9
    assert by_b["大盘"]["n_samples"] == 10
    assert abs(by_b["大盘"]["direction_hit_rate"] - 0.9) < 1e-9


def test_format_markdown_contains_all_sections(tmp_db):
    rows = [{
        "code": "A.SZ", "trade_date": "2025-01-01",
        "sample_end_date": "2025-01-15",
        "pred_10d": 0.02, "actual_10d": 0.03,
        "market_cap_bucket": "小盘", "industry": "计算机", "liquidity_bucket": "中高",
    }]
    _seed(tmp_db, rows)
    md = format_markdown_report(tmp_db, as_of_date="2026-04-12")
    assert "按市值分组" in md
    assert "按行业" in md
    assert "按流动性分组" in md
    assert "小盘" in md
```

- [ ] **Step 2: 运行测试（应失败）**

Run: `python3 -m pytest signal_trust/tests/test_global_stats.py -v`
Expected: ImportError

- [ ] **Step 3: 实现 global_stats**

Create `signal_trust/global_stats.py`:
```python
"""周度全局失效统计."""
import logging
from .db import connect
from .constants import (
    REALIZE_ACTUAL_THRESHOLD,
    GREEN_HIT_MIN, GREEN_BIAS_MIN, GREEN_REALIZE_MIN,
    RED_HIT_MAX, RED_BIAS_MAX, RED_REALIZE_MAX,
)

logger = logging.getLogger(__name__)


def aggregate_by_bucket(db_path: str, bucket_column: str, as_of_date: str) -> list[dict]:
    """
    按 bucket_column (market_cap_bucket / industry / liquidity_bucket) GROUP BY, 
    返回每桶的样本数/三指标.
    """
    allowed = {"market_cap_bucket", "industry", "liquidity_bucket"}
    if bucket_column not in allowed:
        raise ValueError(f"bucket_column must be one of {allowed}")
    conn = connect(db_path)
    try:
        sql = (
            f"SELECT {bucket_column} AS bucket, "
            f"  COUNT(*) AS n, "
            f"  AVG(CASE WHEN (pred_10d > 0) = (actual_10d > 0) THEN 1.0 ELSE 0.0 END) AS hit, "
            f"  AVG(actual_10d - pred_10d) AS bias, "
            f"  AVG(CASE WHEN actual_10d > ? THEN 1.0 ELSE 0.0 END) AS realize "
            f"FROM signal_trust_samples "
            f"WHERE sample_end_date < ? AND actual_10d IS NOT NULL "
            f"  AND {bucket_column} IS NOT NULL "
            f"GROUP BY {bucket_column} "
            f"ORDER BY n DESC"
        )
        rows = conn.execute(sql, (REALIZE_ACTUAL_THRESHOLD, as_of_date)).fetchall()
        return [{
            "bucket": r["bucket"],
            "n_samples": r["n"],
            "direction_hit_rate": r["hit"],
            "systematic_bias": r["bias"],
            "high_pred_realize_rate": r["realize"],
        } for r in rows]
    finally:
        conn.close()


def _tag_symbol(hit, bias, realize) -> str:
    if hit is None:
        return "⚪"
    if hit < RED_HIT_MAX or bias < RED_BIAS_MAX or realize < RED_REALIZE_MAX:
        return "⚠️"
    if hit >= GREEN_HIT_MIN and bias >= GREEN_BIAS_MIN and realize >= GREEN_REALIZE_MIN:
        return "✓"
    return "🟡"


def _section(title: str, agg: list[dict], order: list[str] | None = None) -> str:
    lines = [f"### {title}", "",
             "| 组 | 样本数 | 方向命中 | 系统偏差 | 兑现率 | 标识 |",
             "|----|--------|---------|---------|-------|------|"]
    if order is not None:
        agg = sorted(agg, key=lambda r: order.index(r["bucket"]) if r["bucket"] in order else 999)
    for r in agg:
        sym = _tag_symbol(r["direction_hit_rate"], r["systematic_bias"], r["high_pred_realize_rate"])
        lines.append(
            f"| {r['bucket']} | {r['n_samples']:,} | "
            f"{r['direction_hit_rate']:.1%} | {r['systematic_bias']:+.2%} | "
            f"{r['high_pred_realize_rate']:.1%} | {sym} |"
        )
    return "\n".join(lines)


def format_markdown_report(db_path: str, as_of_date: str) -> str:
    market_cap = aggregate_by_bucket(db_path, "market_cap_bucket", as_of_date)
    industry = aggregate_by_bucket(db_path, "industry", as_of_date)
    liquidity = aggregate_by_bucket(db_path, "liquidity_bucket", as_of_date)

    # 行业挑 Top5/Bottom5 (按方向命中率)
    ind_sorted = sorted(
        [r for r in industry if r["n_samples"] >= 50],
        key=lambda r: r["direction_hit_rate"] or 0,
    )
    ind_bottom = ind_sorted[:5]
    ind_top = ind_sorted[-5:][::-1]

    parts = [
        f"# 信号可信度 · 全局失效诊断",
        f"\n**截止日**: {as_of_date}\n",
        _section("按市值分组", market_cap, order=["微盘", "小盘", "中盘", "大盘", "未知"]),
        "",
        _section("按流动性分组", liquidity, order=["低", "中低", "中高", "高", "未知"]),
        "",
        "### 按行业分组 (样本≥50 的前 5 / 后 5)",
        "",
        "**⚠️ 失效最严重**:",
        _section("失效 Top5", ind_bottom).split("\n", 2)[2],
        "",
        "**✓ 最可靠**:",
        _section("可靠 Top5", ind_top).split("\n", 2)[2],
    ]
    return "\n".join(parts)
```

- [ ] **Step 4: 运行测试**

Run: `python3 -m pytest signal_trust/tests/test_global_stats.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add signal_trust/global_stats.py signal_trust/tests/test_global_stats.py
git commit -m "feat(signal-trust): 全局分组聚合+Markdown报告"
```

---

## Task 8: rebuild 脚本 — 首次建库

**Files:**
- Create: `scripts/rebuild_signal_trust.py`
- Create: `signal_trust/tests/test_integration.py` (小数据端到端)

- [ ] **Step 1: 写集成测试**

Create `signal_trust/tests/test_integration.py`:
```python
"""端到端集成测试: mock 报告 → 建库 → 分数 → 贴标签."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from signal_trust.db import connect
from signal_trust.constants import TAG_GREEN, TAG_RED


def test_full_cycle(tmp_db, tmp_path, seed_stock):
    """3 股票 × 2 月, 验证建库和分数正确."""
    # 种子数据: 20 天的 daily_quotes
    quotes_days = [(f"2025-01-{d:02d}", 100.0 + d * 0.5, 1e8) for d in range(1, 21)]
    # A: 预测准(方向命中+正偏差)
    # B: 预测假(方向经常错)
    for code, industry in [("A.SZ", "银行"), ("B.SZ", "传媒"), ("C.SZ", "计算机")]:
        seed_stock(code, industry, quotes_days,
                   circ_mv_by_date={d: 500_0000 for d, *_ in quotes_days})

    # 写报告 JSON (15 天, 每只股票 pred_10d=0.015)
    reports_root = tmp_path / "reports"
    for i, (d, *_) in enumerate(quotes_days[:10]):  # 只有前 10 天的 sample 会有 T+10
        stocks = [
            {"stock_code": "A.SZ", "pred_10d": 0.015, "rank_score": 95},
            {"stock_code": "B.SZ", "pred_10d": 0.015, "rank_score": 90},
        ]
        version_dir = reports_root / "daily_selection_ng106"
        version_dir.mkdir(parents=True, exist_ok=True)
        (version_dir / f"analysis_data_{d.replace('-', '')}.json").write_text(
            json.dumps({"analysis_date": d, "all_stocks_with_scores": stocks}, ensure_ascii=False)
        )

    # 调用 rebuild 主函数
    from scripts.rebuild_signal_trust import main as rebuild_main
    rebuild_main(db_path=tmp_db, reports_root=str(reports_root), as_of_date="2026-04-12")

    # 验证: samples 表至少 20 行 (10 天 × 2 股票)
    conn = connect(tmp_db)
    (n,) = conn.execute("SELECT COUNT(*) FROM signal_trust_samples").fetchone()
    assert n >= 15  # 允许部分 T+10 不可得
    # scores 表也有对应记录
    (n2,) = conn.execute("SELECT COUNT(*) FROM signal_trust_scores").fetchone()
    assert n2 >= 1
```

- [ ] **Step 2: 运行测试（应失败）**

Run: `python3 -m pytest signal_trust/tests/test_integration.py -v`
Expected: ImportError for `scripts.rebuild_signal_trust`

- [ ] **Step 3: 实现 rebuild 脚本**

Create `scripts/rebuild_signal_trust.py`:
```python
#!/usr/bin/env python3
"""首次建库: 扫所有历史报告 → samples 表 → scores 表."""
import argparse
import logging
import sys
from pathlib import Path

# 允许从项目根目录运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_trust.db import migrate
from signal_trust.sample_builder import (
    scan_reports, dedupe_by_version, compute_actual_10d,
    compute_market_cap_bucket, compute_liquidity_bucket,
    compute_liquidity_thresholds, compute_sample_end_date,
    _industry_lookup, upsert_samples,
)
from signal_trust.scorer import compute_scores, upsert_scores
from signal_trust.constants import DEFAULT_DB_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(db_path: str = DEFAULT_DB_PATH, reports_root: str = "reports",
         as_of_date: str | None = None):
    logger.info("迁移 DB schema...")
    migrate(db_path)

    logger.info(f"扫描报告目录: {reports_root}")
    raw = list(scan_reports([reports_root]))
    logger.info(f"  原始记录: {len(raw):,}")
    deduped = dedupe_by_version(raw)
    logger.info(f"  去重后: {len(deduped):,}")

    # 计算 sample_end_date + actual_10d + 分组标签
    logger.info("计算 actual_10d + 分组...")
    codes = list({r["code"] for r in deduped})
    industry_map = _industry_lookup(db_path, codes)
    # 流动性阈值按当前截止日动态算
    if as_of_date is not None:
        as_of = as_of_date
    elif deduped:
        as_of = max(r["trade_date"] for r in deduped)
    else:
        logger.warning("无样本, 退出")
        return
    liq_thresholds = compute_liquidity_thresholds(db_path, as_of)

    # T+10 未到的样本用哨兵日期占位, 后续 daily update 的 _backfill_actuals 会补齐.
    END_DATE_SENTINEL = "9999-12-31"
    enriched = []
    for i, r in enumerate(deduped):
        if i % 10000 == 0 and i > 0:
            logger.info(f"  进度: {i:,}/{len(deduped):,}")
        end_date = compute_sample_end_date(db_path, r["trade_date"])
        actual = compute_actual_10d(db_path, r["code"], r["trade_date"])
        mc = compute_market_cap_bucket(db_path, r["code"], r["trade_date"])
        liq = compute_liquidity_bucket(db_path, r["code"], r["trade_date"], liq_thresholds)
        enriched.append({
            "code": r["code"],
            "trade_date": r["trade_date"],
            "sample_end_date": end_date if end_date else END_DATE_SENTINEL,
            "pred_10d": r["pred_10d"],
            "actual_10d": actual,
            "version": r["version"],
            "market_cap_bucket": mc,
            "industry": industry_map.get(r["code"], "未分类"),
            "liquidity_bucket": liq,
        })

    logger.info(f"入库: {len(enriched):,} 条")
    upsert_samples(db_path, enriched, update_actual=True)

    logger.info(f"计算全市场分数 as_of_date={as_of}...")
    scores = compute_scores(db_path, as_of_date=as_of)
    upsert_scores(db_path, scores)
    logger.info(f"  {len(scores):,} 只股票已打分")

    # 标签分布
    from collections import Counter
    dist = Counter(s["trust_tag"] for s in scores.values())
    logger.info(f"标签分布: {dict(dist)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", default=DEFAULT_DB_PATH)
    ap.add_argument("--reports-root", default="reports")
    ap.add_argument("--as-of-date", default=None)
    args = ap.parse_args()
    main(args.db_path, args.reports_root, args.as_of_date)
```

- [ ] **Step 4: 运行测试**

Run: `python3 -m pytest signal_trust/tests/test_integration.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/rebuild_signal_trust.py signal_trust/tests/test_integration.py
git commit -m "feat(signal-trust): rebuild 首次建库脚本+集成测试"
```

---

## Task 9: daily update 脚本 + 集成到 run_daily_update.sh

**Files:**
- Create: `scripts/update_signal_trust_daily.py`
- Modify: `run_daily_update.sh`

- [ ] **Step 1: 实现 daily update 脚本**

Create `scripts/update_signal_trust_daily.py`:
```python
#!/usr/bin/env python3
"""每日增量: (A) 新报告入库, (B) 回填 T-10 的 actual_10d, (C) 刷新分数."""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_trust.db import connect, migrate
from signal_trust.sample_builder import (
    scan_reports, dedupe_by_version, compute_actual_10d,
    compute_market_cap_bucket, compute_liquidity_bucket,
    compute_liquidity_thresholds, compute_sample_end_date,
    _industry_lookup, upsert_samples,
)
from signal_trust.scorer import compute_scores, upsert_scores
from signal_trust.constants import DEFAULT_DB_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _new_samples_for_date(db_path: str, reports_root: str, trade_date: str) -> int:
    """(A) 扫当日新报告, 入库(actual_10d 可能 NULL)."""
    # 只扫指定日期的 JSON
    all_records = []
    for r in scan_reports([reports_root]):
        if r["trade_date"] == trade_date:
            all_records.append(r)
    deduped = dedupe_by_version(all_records)
    if not deduped:
        logger.info(f"  (A) 无新样本 @ {trade_date}")
        return 0

    codes = [r["code"] for r in deduped]
    industry_map = _industry_lookup(db_path, codes)
    end_date = compute_sample_end_date(db_path, trade_date)  # 可能为 None
    liq_thr = compute_liquidity_thresholds(db_path, trade_date)

    # 若 end_date 未到, 用未来哨兵值让 scorer 过滤 (sample_end_date < today 为 false)
    # 同时让 validate 泄露自检识别它们 (actual_10d IS NULL 时自检不会误报)
    END_DATE_SENTINEL = "9999-12-31"
    enriched = []
    for r in deduped:
        actual = compute_actual_10d(db_path, r["code"], r["trade_date"])  # 一般为 None
        mc = compute_market_cap_bucket(db_path, r["code"], r["trade_date"])
        liq = compute_liquidity_bucket(db_path, r["code"], r["trade_date"], liq_thr)
        enriched.append({
            "code": r["code"], "trade_date": r["trade_date"],
            "sample_end_date": end_date if end_date else END_DATE_SENTINEL,
            "pred_10d": r["pred_10d"], "actual_10d": actual,
            "version": r["version"],
            "market_cap_bucket": mc, "industry": industry_map.get(r["code"], "未分类"),
            "liquidity_bucket": liq,
        })
    n = upsert_samples(db_path, enriched, update_actual=False)
    logger.info(f"  (A) 入库 {n} 条 @ {trade_date}")
    return n


def _backfill_actuals(db_path: str) -> int:
    """(B) 对所有 actual_10d IS NULL 样本, 尝试计算实际收益 + 补 sample_end_date."""
    conn = connect(db_path)
    try:
        pending = conn.execute(
            "SELECT code, trade_date FROM signal_trust_samples "
            "WHERE actual_10d IS NULL"
        ).fetchall()
    finally:
        conn.close()
    if not pending:
        logger.info("  (B) 无回填任务")
        return 0
    logger.info(f"  (B) 尝试回填 {len(pending)} 条")
    updated = 0
    conn = connect(db_path)
    try:
        for row in pending:
            actual = compute_actual_10d(db_path, row["code"], row["trade_date"])
            if actual is None:
                continue
            end_date = compute_sample_end_date(db_path, row["trade_date"])
            if end_date is None:
                continue
            conn.execute(
                "UPDATE signal_trust_samples SET actual_10d = ?, sample_end_date = ? "
                "WHERE code = ? AND trade_date = ? AND actual_10d IS NULL",
                (actual, end_date, row["code"], row["trade_date"]),
            )
            updated += 1
        conn.commit()
    finally:
        conn.close()
    logger.info(f"  (B) 实际回填 {updated} 条")
    return updated


def main(db_path: str = DEFAULT_DB_PATH, reports_root: str = "reports",
         trade_date: str | None = None):
    migrate(db_path)
    today = trade_date or datetime.today().strftime("%Y-%m-%d")
    logger.info(f"=== Signal Trust 增量 @ {today} ===")

    _new_samples_for_date(db_path, reports_root, today)
    _backfill_actuals(db_path)

    logger.info(f"(C) 刷新全部分数 as_of_date={today}")
    scores = compute_scores(db_path, as_of_date=today)
    n = upsert_scores(db_path, scores)
    logger.info(f"  刷新 {n} 条 scores")

    from collections import Counter
    dist = Counter(s["trust_tag"] for s in scores.values())
    logger.info(f"标签分布: {dict(dist)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", default=DEFAULT_DB_PATH)
    ap.add_argument("--reports-root", default="reports")
    ap.add_argument("--date", default=None, help="trade_date; 默认今日")
    args = ap.parse_args()
    main(args.db_path, args.reports_root, args.date)
```

- [ ] **Step 2: 修改 `run_daily_update.sh`**

Find the line `$PYTHON_CMD $SCRIPT_DIR/tomorrow_stock_selector.py` (around line 145) and after the next `fi` block, add:

```bash
            # 步骤 2.5: Signal Trust 增量 + 贴标签
            print_info "步骤2.5: 更新信号可信度"
            $PYTHON_CMD $SCRIPT_DIR/scripts/update_signal_trust_daily.py
            if [ $? -ne 0 ]; then
                print_warning "Signal Trust 更新失败(非阻塞)"
            fi
            # 查找今日最新的选股 JSON 并贴标签
            LATEST_JSON=$(ls -t $SCRIPT_DIR/reports/daily_selection_*/analysis_data_$(date +%Y%m%d).json 2>/dev/null | head -1)
            if [ -n "$LATEST_JSON" ]; then
                $PYTHON_CMD -c "from signal_trust.report_appender import append_trust_tags; n = append_trust_tags('$LATEST_JSON', 'data_adapter/stock_data.db'); print(f'已为 {n} 只股票贴可信度标签')"
            fi
```

**精确插入位置**：在 `run_daily_update.sh` 中搜索 `print_info "步骤2: 生成量化选股报告"` 所在的 `fi` 闭合之后。如果找不到该注释(脚本已演化)，在 `$PYTHON_CMD $SCRIPT_DIR/tomorrow_stock_selector.py` 行**之后、下一个函数定义之前**的位置插入。

- [ ] **Step 3: 语法检查**

Run: `bash -n run_daily_update.sh`
Expected: 无输出(语法正确)

- [ ] **Step 4: Commit**

```bash
git add scripts/update_signal_trust_daily.py run_daily_update.sh
git commit -m "feat(signal-trust): 日增量脚本+集成 run_daily_update.sh"
```

---

## Task 10: weekly stats + validate 脚本

**Files:**
- Create: `scripts/weekly_signal_trust_stats.py`
- Create: `scripts/validate_signal_trust.py`

- [ ] **Step 1: 实现 weekly stats**

Create `scripts/weekly_signal_trust_stats.py`:
```python
#!/usr/bin/env python3
"""周度全局失效统计, 输出 Markdown 报告."""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_trust.global_stats import format_markdown_report
from signal_trust.constants import DEFAULT_DB_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(db_path: str = DEFAULT_DB_PATH, out_dir: str = "reports/signal_trust",
         as_of_date: str | None = None):
    d = as_of_date or datetime.today().strftime("%Y-%m-%d")
    md = format_markdown_report(db_path, as_of_date=d)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"global_stats_{d.replace('-', '')}.md"
    f.write_text(md, encoding="utf-8")
    logger.info(f"已写入: {f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", default=DEFAULT_DB_PATH)
    ap.add_argument("--out-dir", default="reports/signal_trust")
    ap.add_argument("--as-of-date", default=None)
    args = ap.parse_args()
    main(args.db_path, args.out_dir, args.as_of_date)
```

- [ ] **Step 2: 实现 validate**

Create `scripts/validate_signal_trust.py`:
```python
#!/usr/bin/env python3
"""数据健康检查 + 泄露自检."""
import argparse
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_trust.db import connect
from signal_trust.constants import DEFAULT_DB_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(db_path: str = DEFAULT_DB_PATH):
    conn = connect(db_path)
    cur = conn.cursor()

    # 样本统计
    (total,) = cur.execute("SELECT COUNT(*) FROM signal_trust_samples").fetchone()
    (with_actual,) = cur.execute(
        "SELECT COUNT(*) FROM signal_trust_samples WHERE actual_10d IS NOT NULL"
    ).fetchone()
    (date_min, date_max) = cur.execute(
        "SELECT MIN(trade_date), MAX(trade_date) FROM signal_trust_samples"
    ).fetchone()
    logger.info(f"样本总数: {total:,} (含 actual: {with_actual:,}, NULL: {total-with_actual:,})")
    logger.info(f"覆盖期: {date_min} ~ {date_max}")

    # 版本贡献
    logger.info("各版本样本数:")
    for row in cur.execute(
        "SELECT version, COUNT(*) AS n FROM signal_trust_samples "
        "GROUP BY version ORDER BY n DESC"
    ).fetchall():
        logger.info(f"  {row['version']}: {row['n']:,}")

    # 标签分布
    tags = cur.execute(
        "SELECT trust_tag, COUNT(*) AS n FROM signal_trust_scores GROUP BY trust_tag"
    ).fetchall()
    logger.info("标签分布: " + ", ".join(f"{r['trust_tag']}={r['n']}" for r in tags))

    # 🚨 泄露自检
    today = datetime.today().strftime("%Y-%m-%d")
    (leak,) = cur.execute(
        "SELECT COUNT(*) FROM signal_trust_samples "
        "WHERE sample_end_date >= ? AND actual_10d IS NOT NULL",
        (today,),
    ).fetchone()
    if leak > 0:
        logger.warning(f"⚠️ 疑似泄露: {leak} 条未来样本已有 actual_10d (应为 NULL)")
    else:
        logger.info("✓ 泄露自检通过")

    conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", default=DEFAULT_DB_PATH)
    args = ap.parse_args()
    main(args.db_path)
```

- [ ] **Step 3: 语法/导入检查**

Run: `python3 -c "import scripts.weekly_signal_trust_stats; import scripts.validate_signal_trust"`
Expected: 无错误

或 `python3 scripts/weekly_signal_trust_stats.py --help` / `python3 scripts/validate_signal_trust.py --help`
Expected: 各自打印 argparse help

- [ ] **Step 4: Commit**

```bash
git add scripts/weekly_signal_trust_stats.py scripts/validate_signal_trust.py
git commit -m "feat(signal-trust): 周度统计+数据健康检查脚本"
```

---

## Task 11: Wiki 更新 + CLAUDE.md + Cron

**Files:**
- Create: `docs/wiki/architecture/signal-trust.md`
- Modify: `docs/wiki/index.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: 创建 wiki 条目**

Create `docs/wiki/architecture/signal-trust.md`:
```markdown
# Signal Trust — 选股信号可信度系统

**建立日期**: 2026-04-12
**状态**: 生产中

## 目的

基于每只股票的历史 "预测 vs 实际" 统计, 给每日选股贴可信度标签, 识别庄家释放的假信号; 周度输出市值/行业/流动性分组的模型失效诊断.

## 标签语义

- 🟢 可信: 方向命中≥55% 且 系统偏差≥-2% 且 兑现率≥40%
- 🔴 高风险: 任一严重指标(方向<45% 或 偏差<-3% 或 兑现率<20%)
- 🟡 存疑: 介于两者间
- ⚪ 数据不足: 全历史样本 <10 次

## 数据流

1. **样本池**: `pred_10d > 0.01` (温和看多以上) 的历史记录
2. **跨版本去重**: 同 (code, date) 在多个版本报告出现时, 按 `VERSION_PRIORITY` 取最新
3. **实际收益**: T+10 交易日的 close / T 日 close - 1
4. **可信度分数**: 每日 `update_signal_trust_daily.py` 刷新, SQL 过滤 `sample_end_date < today` 防泄露

## 已知边界

- 停牌 ≥10 日的样本 `actual_10d=NULL` 永不回填, 对应股票样本偏少
- 全历史累计, 不做时间衰减 (早期市场环境不同会有噪音)
- 今日新入库样本要到 T+10 日才参与可信度计算 (正确行为, 非 bug)

## 命令速查

```bash
# 首次建库(一次性)
python3 scripts/rebuild_signal_trust.py

# 每日增量(集成在 run_daily_update.sh)
python3 scripts/update_signal_trust_daily.py

# 周度全局统计(建议周日晚跑)
python3 scripts/weekly_signal_trust_stats.py

# 健康检查
python3 scripts/validate_signal_trust.py
```

## 关键文件

- `signal_trust/` — Python 包
- `signal_trust_samples` / `signal_trust_scores` 两张 SQLite 表
- Design spec: `docs/superpowers/specs/2026-04-12-signal-trust-design.md`
- Implementation plan: `docs/superpowers/plans/2026-04-12-signal-trust.md`
```

- [ ] **Step 2: 在 `docs/wiki/index.md` 加入引用**

查看 `docs/wiki/index.md` 当前结构, 在 architecture 区段追加一行:
```markdown
- [Signal Trust 信号可信度系统](architecture/signal-trust.md) — 给选股贴 🟢🟡🔴 可信度标签, 识别假信号
```

- [ ] **Step 3: 更新 `CLAUDE.md`**

在 `## 🏗️ System Architecture Overview` 节的 "Core Data Flow" 图后追加一小节:

```markdown
### 🆕 Signal Trust 信号可信度
- 独立模块 `signal_trust/`, 基于历史 "预测 vs 实际" 统计给选股贴可信度标签
- 详见 `docs/wiki/architecture/signal-trust.md`
- 日报 JSON 的 Top-50 会追加 `trust_tag`/`trust_samples`/`trust_details` 字段
- 周度全局统计: `python3 scripts/weekly_signal_trust_stats.py` → `reports/signal_trust/`
```

- [ ] **Step 4: Commit**

```bash
git add docs/wiki/architecture/signal-trust.md docs/wiki/index.md CLAUDE.md
git commit -m "docs(signal-trust): wiki条目+CLAUDE.md更新"
```

---

## Task 12: 真实数据建库 + 验收

**Files:** 无新文件, 只执行命令

- [ ] **Step 1: 在生产数据库跑首次建库**

Run: `python3 scripts/rebuild_signal_trust.py 2>&1 | tee logs/signal_trust_rebuild_$(date +%Y%m%d).log`
Expected: 
- 完成时间 < 15 分钟
- 打印"样本总数 > 50万"
- 打印标签分布(🟢🟡🔴⚪ 各占一部分, 不出现某档 100%)

- [ ] **Step 2: 健康检查**

Run: `python3 scripts/validate_signal_trust.py`
Expected: 
- 泄露自检通过
- 各版本都有样本贡献
- 标签分布合理

- [ ] **Step 3: 抽样人工对照**

Run: 
```bash
python3 -c "
from signal_trust.db import connect
conn = connect('data_adapter/stock_data.db')
# 抽 5 只 🔴 高风险股票手动看
rows = conn.execute('SELECT code, n_samples, direction_hit_rate, systematic_bias, high_pred_realize_rate FROM signal_trust_scores WHERE trust_tag = \"🔴高风险\" LIMIT 5').fetchall()
for r in rows:
    print(dict(r))
"
```
**人工检查**: 挑其中一只股票, 到 `reports/daily_selection_ng106/` 找它某天 pred_10d 高的记录, 去数据库查对应 T+10 close, 手动算 actual_10d, 确认与样本表一致.

- [ ] **Step 4: 周度统计首次生成**

Run: `python3 scripts/weekly_signal_trust_stats.py`
Check: `reports/signal_trust/global_stats_*.md` 存在, 内容三节齐全.

- [ ] **Step 5: 跑一次日报贴标签**

```bash
LATEST=$(ls -t reports/daily_selection_ng106/analysis_data_*.json | head -1)
python3 -c "from signal_trust.report_appender import append_trust_tags; print(append_trust_tags('$LATEST', 'data_adapter/stock_data.db'))"
python3 -c "
import json
data = json.load(open('$LATEST'))
for s in sorted(data['all_stocks_with_scores'], key=lambda x: x.get('rank_score', 0), reverse=True)[:10]:
    print(s.get('stock_code'), s.get('trust_tag'), s.get('trust_samples'), s.get('trust_details', {}).get('direction_hit_rate'))
"
```
Expected: 前 10 行每行带 trust_tag + n_samples + hit_rate

- [ ] **Step 6: Commit日志(如有)**

```bash
git add logs/signal_trust_rebuild_*.log 2>/dev/null || true
git commit -m "chore(signal-trust): 首次建库验收通过" --allow-empty
```

---

## 验收 Checklist

- [ ] 所有单元测试通过 (`pytest signal_trust/tests/`)
- [ ] 真实库建库用时 <15 分钟
- [ ] 泄露自检通过 0 条未来样本
- [ ] 标签分布合理(没有某档 100%)
- [ ] 一份最新 analysis_data JSON 能正确追加 trust_* 字段
- [ ] 周度统计 Markdown 报告生成成功
- [ ] Wiki + CLAUDE.md 更新
- [ ] 所有 commit 已提交
