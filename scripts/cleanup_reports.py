"""P3.2: reports/ 清理 + naming convention.

把已 REJECTED 版本的 daily_selection_*/grid_* 目录移到 reports/archive/, 减少日常 ls 噪声.

REJECTED 版本来自 memory + wiki/log.md:
- ng1.2.3 / ng1.2.4 (REJECTED, ng12x_iteration / ng124_plan)
- ng1.3.0 (REJECTED, ng130_stage35_v2_rejected)
- ng1.4.0 / 1.4.1 / 1.4.2 (Tier A 全败, ng14x_tier_a_closed)
- ng1.5.0 (REJECTED, ng150_rejected)
- ng1.6.x (memory 未明确, 但 stage4a/35 测试性命名暗示 rejected)
- ng120 / ng121 / ng122 (ng1.2.x 三个变体的 alias, 全败)
- ng111 (ABANDONED, ng111_abandoned)
- _ng150_sanity_* (sanity 跑丢的 artifact)
- *_grid_*  (grid search 临时输出)

Usage:
    python3 scripts/cleanup_reports.py --dry-run   # 默认, 只打印要移的
    python3 scripts/cleanup_reports.py --execute   # 真正移动到 archive
    python3 scripts/cleanup_reports.py --list-only # 只列已识别为 REJECTED 的目录
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
ARCHIVE = REPORTS / "archive"

# 已 REJECTED 版本的前缀关键字 (case-sensitive)
REJECTED_PREFIXES = [
    # ng 已 REJECTED
    "daily_selection_ng1.2.3",
    "daily_selection_ng1.2.4",
    "daily_selection_ng1.3.0",
    "daily_selection_ng1.4.0",
    "daily_selection_ng1.4.1",
    "daily_selection_ng1.4.2",
    "daily_selection_ng1.5.0",
    "daily_selection_ng1.6.1",
    "daily_selection_ng111",
    "daily_selection_ng120",
    "daily_selection_ng121",
    "daily_selection_ng122",
    "daily_selection_ng123",
    "daily_selection_ng124",
    "daily_selection_ng130",
    "daily_selection_ng14",          # 1.4.x catch-all
    "daily_selection_ng150",
    # sanity / grid 临时 artifact
    "_ng150_sanity_",
    "ng106_3state_grid_",
    "ng106_crisis_grid_",
    "ng106_grid_",
]


def list_targets() -> list[Path]:
    if not REPORTS.exists():
        return []
    targets = []
    for entry in sorted(REPORTS.iterdir()):
        if entry == ARCHIVE:
            continue
        if any(entry.name.startswith(p) for p in REJECTED_PREFIXES):
            targets.append(entry)
    return targets


def move_to_archive(entries: list[Path], dry_run: bool = True) -> int:
    if not entries:
        print("[cleanup] 无 rejected 目录可移")
        return 0
    if not dry_run:
        ARCHIVE.mkdir(parents=True, exist_ok=True)
    moved = 0
    for src in entries:
        dst = ARCHIVE / src.name
        if dst.exists():
            print(f"[skip]  目标已存在: {dst}")
            continue
        if dry_run:
            print(f"[DRY]   {src} → {dst}")
        else:
            shutil.move(str(src), str(dst))
            print(f"[moved] {src.name}")
            moved += 1
    return moved


def write_archive_readme():
    """Write reports/archive/README.md documenting naming convention + which are archived."""
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    targets = list_targets()
    lines = [
        "# reports/archive/",
        "",
        "已 REJECTED 模型版本的报告目录归档区. 不参与日常 eval, 仅作历史参考.",
        "",
        "## reports/ 命名规范 (2026-04-27 起)",
        "",
        "活跃模型的命名: `daily_selection_{version}_{purpose}`",
        "- version: ng106 / ng106v2 / ng2_0a / ng2_1 (生产灰度)",
        "- purpose: ",
        "  - `_fullmarket` — 实时生产报告",
        "  - `_wf_oos` — Walk-Forward OOS 测试段",
        "  - `_pre2020` — 训练区前真零泄漏 OOS",
        "  - `_stage35` / `_stage4a` / `_fast` — 训练阶段评估 (eval 完成后归档)",
        "",
        "## 当前 archive 内容",
        "",
        "REJECTED 版本根据 memory + wiki/log.md (2026-04-27 cleanup):",
        "",
    ]
    for t in targets:
        lines.append(f"- `{t.name}`")
    lines.extend([
        "",
        "如要回溯原因, 查 `docs/wiki/log.md` 时间线.",
    ])
    (ARCHIVE / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", default=True,
                   help="只打印要移的目录 (默认)")
    g.add_argument("--execute", action="store_true",
                   help="真正移动到 archive/")
    g.add_argument("--list-only", action="store_true",
                   help="只列出 REJECTED 目录数量")
    args = ap.parse_args()

    targets = list_targets()
    print(f"[cleanup] 识别到 {len(targets)} 个 REJECTED 目录")

    if args.list_only:
        for t in targets:
            print(f"  {t.name}")
        return

    if args.execute:
        moved = move_to_archive(targets, dry_run=False)
        write_archive_readme()
        print(f"\n[done] 移动 {moved}/{len(targets)} 个目录到 {ARCHIVE}")
        print(f"[done] README written: {ARCHIVE / 'README.md'}")
    else:
        move_to_archive(targets, dry_run=True)
        print(f"\n[hint] 用 --execute 真正移动")


if __name__ == "__main__":
    main()
