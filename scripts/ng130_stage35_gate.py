"""ng1.3.0 Stage 3.5 Gate — 2025 spot-check V5.2 raw ≥ 65%.

Single-config gate (composite top-10, 10d focus) that:
  1. Generates reports for a gate window (default 2025 full year) via
     batch_generate_v395_reports.py
  2. Runs run_north_star_eval.py in backtest mode with the spec config
  3. Parses V5.2 raw score (not discounted) and emits PASS / FAIL to stdout
  4. Writes postmortem to reports/ng130/stage35_<pass|rejected>_v2.md

Usage:
  python3 scripts/ng130_stage35_gate.py                        # 2025 full year
  python3 scripts/ng130_stage35_gate.py --start 2024-01-01 --end 2026-04-15
  python3 scripts/ng130_stage35_gate.py --skip-generate        # reuse cached reports
"""
import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REPORT_BASE = REPO / 'reports'
GATE_THRESHOLD = 65.0  # V5.2 raw percentage threshold
DEFAULT_VERSION = 'ng1.3.0'
DEFAULT_START = '2025-01-01'
DEFAULT_END = '2025-12-31'


def run(cmd: list[str], cwd: Path = REPO) -> str:
    print(f"$ {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        raise SystemExit(f"Command failed (rc={res.returncode})")
    return res.stdout


def generate_reports(version: str, start: str, end: str, output_dir: Path) -> None:
    cmd = [
        sys.executable, 'backtest/batch_generate_v395_reports.py',
        '--version', version,
        '--start-date', start,
        '--end-date', end,
        '--output-dir', str(output_dir),
    ]
    run(cmd)


def run_north_star(report_dir: Path, label: str) -> str:
    cmd = [
        sys.executable, 'backtest/run_north_star_eval.py',
        '--backtest',
        '--report-dir', str(report_dir),
        '--label', label,
        '--top-n', '10',
        '--focus-days', '10',
        '--rank-field', 'composite',
        '--score-version', 'v52',
    ]
    return run(cmd)


# V5.2 scorecard emits: "原始总分: 63.8/105 (未加权61%)".
# V5.2 is printed last (after V2/V3/V4/V5/V5.1), so we split on the V5.2 header
# and parse the 原始总分 line inside that block to avoid capturing earlier cards.
V52_HEADER_RE = re.compile(r'北极星评分卡\s*V5\.2')
RAW_SCORE_RE = re.compile(r'原始总分:\s*([\d.]+)\s*/\s*105\s*\(未加权([\d.]+)%\)')


def parse_v52_raw(north_star_output: str) -> tuple[float, float] | None:
    """Return (score_out_of_105, pct) of V5.2 scorecard in north_star stdout."""
    header = V52_HEADER_RE.search(north_star_output)
    block = north_star_output[header.start():] if header else north_star_output
    m = RAW_SCORE_RE.search(block)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def write_postmortem(passed: bool, v52_pct: float, label: str, start: str, end: str,
                     log_excerpt: str) -> Path:
    status = 'pass' if passed else 'rejected_v2'
    out = REPORT_BASE / 'ng130' / f'stage35_{status}.md'
    out.parent.mkdir(parents=True, exist_ok=True)
    verdict = 'PASS' if passed else 'REJECTED'
    with open(out, 'w') as f:
        f.write(f"# ng1.3.0 Stage 3.5 Gate — {verdict}\n\n")
        f.write(f"- Date: {datetime.now():%Y-%m-%d %H:%M}\n")
        f.write(f"- Window: {start} → {end}\n")
        f.write(f"- Config: composite top10, 10d focus (spec gate config)\n")
        f.write(f"- Threshold: V5.2 raw ≥ {GATE_THRESHOLD}%\n")
        f.write(f"- Observed: V5.2 raw = {v52_pct:.1f}%\n")
        f.write(f"- Label: {label}\n\n")
        f.write("## North Star excerpt\n\n```\n")
        f.write(log_excerpt)
        f.write("\n```\n")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', default=DEFAULT_VERSION)
    ap.add_argument('--start', default=DEFAULT_START)
    ap.add_argument('--end', default=DEFAULT_END)
    ap.add_argument('--skip-generate', action='store_true')
    ap.add_argument('--output-dir', default=None)
    ap.add_argument('--threshold', type=float, default=GATE_THRESHOLD)
    args = ap.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else \
        REPORT_BASE / f'daily_selection_{args.version}_stage35'

    if not args.skip_generate:
        print(f"\n=== Step 1/3: generating {args.version} reports for {args.start} → {args.end} ===")
        generate_reports(args.version, args.start, args.end, output_dir)
    else:
        print(f"\n=== Step 1/3: skipping report generation (--skip-generate) ===")
        if not output_dir.exists():
            raise SystemExit(f"Report dir missing: {output_dir}")

    print(f"\n=== Step 2/3: running north_star_eval backtest ===")
    label = f'NG130-STAGE35-{args.start}-{args.end}'
    ns_output = run_north_star(output_dir, label)
    print(ns_output[-4000:])

    print(f"\n=== Step 3/3: parse V5.2 raw score ===")
    parsed = parse_v52_raw(ns_output)
    if parsed is None:
        print("⚠️ Could not parse V5.2 from north_star output — gate FAIL (parse error)")
        write_postmortem(False, 0.0, label, args.start, args.end,
                         ns_output[-2000:])
        return 2

    v52_score, v52_pct = parsed
    passed = v52_pct >= args.threshold
    verdict = "✅ PASS" if passed else "❌ REJECTED"
    print(f"\n{verdict}: V5.2 raw = {v52_pct:.1f}% ({v52_score:.1f}/105), "
          f"threshold = {args.threshold}%")

    pm_path = write_postmortem(passed, v52_pct, label, args.start, args.end,
                               ns_output[-4000:])
    print(f"Postmortem: {pm_path}")
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
