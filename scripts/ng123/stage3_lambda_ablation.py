#!/usr/bin/env python3
"""ng1.2.3 Stage 3: Lambda ablation for label downside penalty.

Per spec §6.3 — 5 λ × 2 WF windows = 10 mini-trainings.

Pass criterion: largest λ>0 with 10d ICIR >= 0.95 × ng1.0.1 baseline.
- If all λ>0 degrade >5%, fall back to λ=0 (no label change).
- If λ=0 wins, label change has no value (still acceptable per spec).

Output: reports/ng123/fastcheck/stage3_lambda_ablation.csv
        reports/ng123/fastcheck/stage3_status.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / 'reports' / 'ng123' / 'fastcheck'
LOG_DIR = OUTPUT_DIR / 'stage3_logs'

LAMBDAS = [0.0, 0.15, 0.30, 0.45, 0.60]
NG101_BASELINE_10D_ICIR = 0.93  # from MEMORY.md ng_production_switch + existing NG training
PASS_THRESHOLD = 0.95            # >=95% of baseline


def run_one_lambda(lam: float, start: str, end: str) -> dict:
    """Run fast-check training for one lambda. Returns metrics dict."""
    log_file = LOG_DIR / f'lambda_{lam:.2f}.log'
    cmd = [
        sys.executable, 'ml_models/ng/ng_trainer.py',
        '--version', 'ng1.2.3',
        '--start-date', start, '--end-date', end,
        '--fast-check',
        '--lambda-downside', str(lam),
        '--purge-days', '15',
    ]
    print(f"\n  lambda={lam:.2f}: running fast-check...", flush=True)
    with open(log_file, 'w') as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                cwd=str(PROJECT_ROOT))

    # Parse log for 10d ICIR
    # Trainer typically logs lines like: "10d ICIR: 0.923" or "horizon=10 ICIR=0.845"
    icir_10d = None
    log_content = log_file.read_text()
    # Try multiple regex patterns (trainer output format may vary)
    patterns = [
        r'10d\s+ICIR[:=\s]+([-+]?\d+\.\d+)',
        r'horizon[_\s]*=?\s*10[dD]?\s+.*?ICIR[:=\s]+([-+]?\d+\.\d+)',
        r'ICIR\s+10d[:=\s]+([-+]?\d+\.\d+)',
        r'10d.*?ICIR.*?([-+]?\d+\.\d+)',
    ]
    for pat in patterns:
        matches = re.findall(pat, log_content, re.IGNORECASE)
        if matches:
            # Take the LAST match (final OOS ICIR, not intermediate)
            try:
                icir_10d = float(matches[-1])
                break
            except ValueError:
                continue
    if icir_10d is None:
        print(f"    WARNING: Could not parse 10d ICIR from {log_file} -- check log format",
              flush=True)

    return {
        'lambda': lam,
        'icir_10d': icir_10d if icir_10d is not None else float('nan'),
        'log_file': str(log_file.relative_to(PROJECT_ROOT)),
        'trainer_exit_code': result.returncode,
    }


def main():
    p = argparse.ArgumentParser(
        description='ng1.2.3 Stage 3: Lambda ablation for downside penalty label.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Lambda candidates: {LAMBDAS}
Baseline 10d ICIR (ng1.0.1): {NG101_BASELINE_10D_ICIR}
Pass threshold: {PASS_THRESHOLD} x baseline = {NG101_BASELINE_10D_ICIR * PASS_THRESHOLD:.3f}

Requires: ng123_feature_cache backfilled for the full train period (Task 13).
Output: {OUTPUT_DIR}
        """,
    )
    p.add_argument('--start-date', default='2022-01-01',
                   help='WF start date (default 2022-01-01, requires mini-backfill)')
    p.add_argument('--end-date', default='2022-12-31',
                   help='WF end date')
    p.add_argument('--baseline-icir', type=float, default=NG101_BASELINE_10D_ICIR,
                   help='ng1.0.1 baseline 10d ICIR for comparison')
    p.add_argument('--pass-threshold', type=float, default=PASS_THRESHOLD)
    args = p.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Stage 3: Lambda Ablation ===")
    print(f"lambda candidates: {LAMBDAS}")
    print(f"Baseline 10d ICIR: {args.baseline_icir}")
    print(f"Pass threshold: {args.pass_threshold * 100}% x baseline = "
          f"{args.baseline_icir * args.pass_threshold:.3f}")
    print(f"Train period: {args.start_date} -> {args.end_date}")

    results = []
    for lam in LAMBDAS:
        r = run_one_lambda(lam, args.start_date, args.end_date)
        results.append(r)
        icir = r['icir_10d']
        if not pd.isna(icir):
            print(f"  lambda={lam:.2f} -> 10d ICIR = {icir:.3f}")
        else:
            print(f"  lambda={lam:.2f} -> PARSE FAIL")

    df = pd.DataFrame(results)
    df['ratio_to_baseline'] = df['icir_10d'] / args.baseline_icir
    df['passes_threshold'] = df['ratio_to_baseline'] >= args.pass_threshold
    df = df.sort_values('lambda')

    print("\n=== Results ===")
    print(df.to_string())

    # Selection: largest lambda > 0 with passes_threshold=True
    passing_nonzero = df[df['passes_threshold'] & (df['lambda'] > 0)]
    if len(passing_nonzero) > 0:
        best_lam = float(passing_nonzero['lambda'].max())
        overall_pass = True
        verdict = f"PASS (lambda={best_lam} selected)"
    elif df[df['lambda'] == 0.0]['passes_threshold'].any():
        best_lam = 0.0
        overall_pass = True  # Technically pass but with lambda=0 (no label change)
        verdict = "PASS (lambda=0, label change has no value -> use ng1.0.1 style label)"
    else:
        best_lam = None
        overall_pass = False
        verdict = "FAIL (all lambda fail threshold -- label transform does not work)"

    print(f"\nSelected lambda: {best_lam}")
    print(f"STAGE 3 OVERALL: {verdict}")

    csv_path = OUTPUT_DIR / 'stage3_lambda_ablation.csv'
    df.to_csv(csv_path, index=False)
    status = {
        'stage': 3,
        'overall_pass': bool(overall_pass),
        'verdict': verdict,
        'selected_lambda': best_lam,
        'baseline_icir': args.baseline_icir,
        'pass_threshold': args.pass_threshold,
        'results': df.to_dict('records'),
    }
    json_path = OUTPUT_DIR / 'stage3_status.json'
    with open(json_path, 'w') as f:
        json.dump(status, f, indent=2, default=str)
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == '__main__':
    main()
