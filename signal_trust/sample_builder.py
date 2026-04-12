"""样本池构建器。"""
import json
import logging
import math
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
                    if math.isnan(pred_val) or pred_val <= PRED_THRESHOLD:
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
