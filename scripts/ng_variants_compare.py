"""Aggregate fast-check WF summaries across ng1.2.x variants into one table.

Usage:
    python3 scripts/ng_variants_compare.py

Scans `logs/` for ng12x fast-check runs and prints a horizon × variant matrix
of (IC, ICIR) averaged across walk-forward windows.
"""
from __future__ import annotations

import glob
import os
import re
from collections import OrderedDict

LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')

# Summary lines look like:
#   [INFO]   3d: IC=0.0704+-0.0087, ICIR=0.5977+-0.0860
SUMMARY_RE = re.compile(
    r'(\d+d):\s*IC=([-\d.]+)\+-([\d.]+),\s*ICIR=([-\d.]+)\+-([\d.]+)'
)


def parse_log(path: str) -> dict | None:
    """Return {horizon: (ic, ic_std, icir, icir_std)} or None if not finished."""
    with open(path, encoding='utf-8', errors='replace') as f:
        text = f.read()
    # Only count summary block (after `Walk-Forward 汇总`)
    marker = text.find('Walk-Forward 汇总')
    if marker < 0:
        return None
    summary = text[marker:]
    out = {}
    for m in SUMMARY_RE.finditer(summary):
        h, ic, ic_s, icir, icir_s = m.groups()
        out[h] = (float(ic), float(ic_s), float(icir), float(icir_s))
    return out or None


def variant_label(fname: str) -> str | None:
    base = os.path.basename(fname)
    # Accept both `fastcheck` (direct run) and `fast-check` (grid-script run).
    m = re.search(r'ng120_fast-?check_m(\d{3})_', base)
    if m:
        cents = int(m.group(1))
        return f'ng120_m=0.{cents:02d}'
    if 'ng121_fast' in base:
        return 'ng121_rank'
    if 'ng122_fast' in base:
        return 'ng122_quint'
    return None


def collect() -> OrderedDict:
    """Return variant → summary dict; prefers the newest log per variant."""
    files = sorted(
        glob.glob(os.path.join(LOG_DIR, 'ng12*_fastcheck_*.log'))
        + glob.glob(os.path.join(LOG_DIR, 'ng12*_fast-check_*.log'))
    )
    by_variant: dict[str, tuple[str, dict]] = {}
    for f in files:
        label = variant_label(f)
        if not label:
            continue
        summary = parse_log(f)
        if not summary:
            continue
        by_variant[label] = (f, summary)  # later file overwrites → newest wins

    order = ['ng120_m=0.03', 'ng120_m=0.05', 'ng120_m=0.08', 'ng120_m=0.10',
             'ng121_rank', 'ng122_quint']
    return OrderedDict((k, by_variant[k]) for k in order if k in by_variant)


def render(table: OrderedDict):
    horizons = ['3d', '5d', '10d', '15d']
    print()
    print('ng1.2.x Fast-Check WF Comparison')
    print('=' * 90)
    header = f'{"variant":<16}' + ''.join(f'{h+" IC":>12}{h+" ICIR":>14}' for h in horizons)
    print(header)
    print('-' * 90)
    for variant, (path, summary) in table.items():
        row = f'{variant:<16}'
        for h in horizons:
            if h in summary:
                ic, _, icir, _ = summary[h]
                row += f'{ic:>12.4f}{icir:>14.4f}'
            else:
                row += f'{"-":>12}{"-":>14}'
        print(row)
    print('=' * 90)
    print(f'{len(table)} variants; log source paths:')
    for variant, (path, _) in table.items():
        print(f'  {variant}: {os.path.basename(path)}')


def main():
    table = collect()
    if not table:
        print('No finished ng1.2.x fast-check logs found in logs/')
        return
    render(table)


if __name__ == '__main__':
    main()
