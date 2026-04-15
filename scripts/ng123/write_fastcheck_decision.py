#!/usr/bin/env python3
"""ng1.2.3 Fast-Check Decision Aggregator.

Reads stage1_recalibrated.json + stage2_status.json + stage3_status.json and
writes a combined `decision.md` per spec §6.4 (Go/No-Go gate).

Usage:
    python3 scripts/ng123/write_fastcheck_decision.py

Output: reports/ng123/fastcheck/decision.md
"""
import datetime
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT = PROJECT_ROOT / 'reports' / 'ng123' / 'fastcheck'


def _load(name: str) -> dict:
    p = OUT / name
    if not p.exists():
        return {'missing': True, 'path': str(p)}
    with open(p) as f:
        return json.load(f)


def main():
    s1 = _load('stage1_recalibrated.json')
    s2 = _load('stage2_status.json')
    s3 = _load('stage3_status.json')

    s1_pass = s1.get('overall_pass', False) if not s1.get('missing') else None
    s2_pass = s2.get('overall_pass', False) if not s2.get('missing') else None
    s3_pass = s3.get('overall_pass', False) if not s3.get('missing') else None

    # Spec §6.4 Go conditions
    if s1_pass and s2_pass:
        condition = "A"
        decision = "GO — full training with 6 mf + 6 mined + selected λ"
    elif s1_pass and not s2_pass:
        condition = "B"
        decision = "GO — full training with 6 mf only + selected λ (mined factors abandoned)"
    elif not s1_pass and s3_pass:
        condition = "D (unplanned)"
        decision = "PARTIAL — no moneyflow, proceed if user decides"
    else:
        condition = "C"
        decision = "NO-GO — project terminated; write postmortem"

    selected_lambda = s3.get('selected_lambda', 'N/A') if s3_pass else 'N/A'

    def _icon(pass_val):
        if pass_val is None:
            return "⏳ NOT RUN"
        return "✅ PASS" if pass_val else "❌ FAIL"

    lines = [
        f"# ng1.2.3 Fast-Check Decision",
        "",
        f"**Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Stage Results",
        "",
        f"### Stage 1: Moneyflow IC Validation {_icon(s1_pass)}",
    ]
    if not s1.get('missing'):
        lines.append("")
        verdict = s1.get('verdict', 'unknown')
        lines.append(f"- Verdict: {verdict}")
        if 'accepted_factors' in s1:
            lines.append(f"- Accepted factors: {len(s1['accepted_factors'])}")
            for f in s1['accepted_factors']:
                lines.append(f"  - `{f['name']}` (|IC|={f['abs_ic']}, |ICIR|={f['abs_icir']}, ng101 {f['ng101_percentile']})")
    else:
        lines.append("- _stage1_recalibrated.json missing_")

    lines.extend([
        "",
        f"### Stage 2: Mined Factor Validation {_icon(s2_pass)}",
    ])
    if not s2.get('missing'):
        lines.append("")
        lines.append(f"- Regime-stable factors: {s2.get('n_regime_stable', s2.get('n_stable', 'n/a'))}")
        for f in s2.get('top_6', [])[:6]:
            name = f.get('name', 'unknown')
            ic = f.get('original_ic', 'n/a')
            icir = f.get('original_icir', 'n/a')
            lines.append(f"  - `{name}` (IC={ic}, ICIR={icir})")
    else:
        lines.append("- _stage2_status.json missing — mining/validation not run_")

    lines.extend([
        "",
        f"### Stage 3: Lambda Ablation {_icon(s3_pass)}",
    ])
    if not s3.get('missing'):
        lines.append("")
        lines.append(f"- Verdict: {s3.get('verdict', 'unknown')}")
        lines.append(f"- Selected λ: {selected_lambda}")
    else:
        lines.append("- _stage3_status.json missing — ablation not run_")

    lines.extend([
        "",
        "## Decision",
        "",
        f"**Spec §6.4 Condition {condition}**: {decision}",
        "",
    ])

    if condition == "A":
        lines.append("### Go Path")
        lines.append("- Proceed to Phase 2 (full backfill 2018-2026, Task 15)")
        lines.append("- Then Phase 3 (train 3 seeds, Task 16)")
        lines.append(f"- Use selected λ = {selected_lambda} for training")
        lines.append("- Expected features: 52 base + 6 mf + 6 mined + 10 market = 74 total")
    elif condition == "B":
        lines.append("### Go Path (moneyflow-only, no mined)")
        lines.append("- Proceed to Phase 2 + 3 with 52 base + 6 mf = 58 stock features")
        lines.append("- Remove `compute_all_mined_factors_for_stock` call from cache_updater")
        lines.append(f"- Use selected λ = {selected_lambda}")
    elif condition == "D (unplanned)":
        lines.append("### Partial Path (user decision required)")
        lines.append("- Stage 1 failed but Stage 3 passed — label change alone has value")
        lines.append("- Options: skip moneyflow entirely + use downside label only")
    else:
        lines.append("### Postmortem Path")
        lines.append("- Write `docs/wiki/architecture/ng123_postmortem.md` (Task 21)")
        lines.append("- Preserve ng123 cache + scripts for future research")
        lines.append("- Keep `PRODUCTION_VERSION = 'ng1.0.1'`")

    content = "\n".join(lines) + "\n"
    target = OUT / 'decision.md'
    target.write_text(content, encoding='utf-8')
    print(f"Decision written: {target}")
    print(f"Condition: {condition} — {decision}")

    return 0 if (s1_pass and s3_pass) else 1


if __name__ == '__main__':
    sys.exit(main())
