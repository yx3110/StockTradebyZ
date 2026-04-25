"""Phase C calibration sweep: run regime-switch backtest + north-star eval for each B1/B2 variant.

Outputs a markdown comparison table to reports/ng2_0a_calibration_sweep_results.md.

Sub-models used (per user decision 2026-04-25):
  bull = ng1.0.1 (reports/daily_selection_ng101 for WF-OOS, _pre2020 for Pre-2020)
  bear = ng1.0.4-3s (reports/daily_selection_ng104_ensemble_3seed for WF-OOS, _pre2020 for Pre-2020)
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

VARIANTS = [
    'baseline',     # V0: B1=(0.45,0.55), B2=(0.30,0.70), streak=3, vote=2
    'strict_b1',    # V1: B1=(0.40,0.65)
    'strict_b2',    # V2: B2=(0.25,0.75)
    'streak5',      # V3: system_streak=5
    'unanimous',    # V4: vote_threshold=3
]

REPO = Path(__file__).resolve().parents[1]


def run(cmd: list, log_path: Path) -> str:
    """Run shell command, tee to log, return stdout."""
    print(f'  > {" ".join(cmd)}')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    full = res.stdout + res.stderr
    log_path.write_text(full)
    if res.returncode != 0:
        print(f'  ! exited {res.returncode}, see {log_path}')
    return full


def extract_metrics(log_text: str) -> dict:
    """Extract V5.2 final score + 10d Sharpe/MaxDD/annual from north-star eval log.

    The eval log structure has these blocks per holding period:
      📊 10日持仓 (NNN天):       — IC stats (Top10 returns, ICIR)
      🎯 10日持仓 北极星指标 (..): — full risk metrics (Sharpe, MaxDD, annual)
      北极星评分卡 VN: ... (10日持仓) — scorecard

    We target the 🎯 block for risk metrics, the 📊 block for ICIR.
    """
    m = {}
    # Last 加权评分 line (V5.2 final card, the gate metric)
    matches = re.findall(
        r'加权评分:\s*([\d.]+)%\s*(?:×\s*[\d.]+\s*=\s*([\d.]+)%)?\s*→\s*等级\s*(\S+)',
        log_text,
    )
    if matches:
        last = matches[-1]
        m['v52_score'] = float(last[1]) if last[1] else float(last[0])
        m['v52_grade'] = last[2]

    # 10d 🎯 北极星指标 block — risk metrics (Sharpe/MaxDD/annual)
    risk_sec = re.search(
        r'🎯\s*10日持仓\s*北极星指标[^\n]*\n(.+?)(?=\n\s*🎯|\n\s*📊|\n\s*北极星评分卡|\Z)',
        log_text, re.DOTALL,
    )
    if risk_sec:
        block = risk_sec.group(1)
        for key, label in [
            ('sharpe', r'Sharpe:\s*([-\d.]+)'),
            ('maxdd', r'最大回撤:\s*([-\d.]+)%'),
            ('annual_gross', r'年化收益\(毛\):\s*([-\d.]+)%'),
            ('annual_net', r'年化收益\(净\):\s*([-\d.]+)%'),
        ]:
            mm = re.search(label, block)
            if mm:
                m[key] = float(mm.group(1))

    # 10d 📊 IC block — ICIR
    ic_sec = re.search(
        r'📊\s*10日持仓\s*\([^\n]+\):\s*\n(.+?)(?=\n\s*📊|\n\s*🎯|\Z)',
        log_text, re.DOTALL,
    )
    if ic_sec:
        mm = re.search(r'^\s*ICIR:\s*([-\d.]+)', ic_sec.group(1), re.MULTILINE)
        if mm:
            m['icir'] = float(mm.group(1))
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variants', nargs='+', default=VARIANTS,
                   help='Subset of variants to test')
    p.add_argument('--skip-merge', action='store_true',
                   help='Skip regime_switch merge step (assume merged dirs already exist)')
    args = p.parse_args()

    bull_wfo = 'reports/daily_selection_ng101'
    bear_wfo = 'reports/daily_selection_ng104_ensemble_3seed'
    bull_pre = 'reports/daily_selection_ng101_pre2020'
    bear_pre = 'reports/daily_selection_ng104_pre2020'

    rows = []  # (variant, window, metrics_dict)

    for variant in args.variants:
        table = f'market_regime_signals_{variant}'
        for window, start, end, bd, brd in [
            ('WF-OOS', '2020-01-01', '2026-04-25', bull_wfo, bear_wfo),
            ('Pre-2020', '2018-01-01', '2019-12-31', bull_pre, bear_pre),
        ]:
            tag = f'{variant}_{window.lower().replace("-", "_")}'
            out_dir = f'reports/daily_selection_regime_switch_{tag}'

            if not args.skip_merge:
                print(f'\n[Merge] variant={variant} window={window}')
                run([
                    'python3', 'backtest/regime_switch_backtest.py',
                    '--regime-version', 'v2',
                    '--regime-table', table,
                    '--bull-dir', bd,
                    '--bear-dir', brd,
                    '--out-dir', out_dir,
                ], Path(f'logs/ng2_0a_phase_c_merge_{tag}.log'))

            print(f'\n[Eval] variant={variant} window={window}')
            log = Path(f'logs/ng2_0a_phase_c_eval_{tag}.log')
            run([
                'python3', 'backtest/run_north_star_eval.py', '--backtest',
                '--report-dir', out_dir,
                '--label', f'ng2.0a-v2-{variant}-{window}',
                '--top-n', '10', '--focus-days', '10', '--rank-field', 'composite',
                '--start-date', start, '--end-date', end,
            ], log)
            metrics = extract_metrics(log.read_text())
            rows.append((variant, window, metrics))
            print(f'  -> V5.2={metrics.get("v52_score", "N/A")}%, '
                  f'Sharpe={metrics.get("sharpe", "N/A")}, '
                  f'MaxDD={metrics.get("maxdd", "N/A")}%')

    # Write markdown report
    out = Path('reports/ng2_0a_calibration_sweep_results.md')
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ['# ng2.0a Calibration Sweep Results\n']
    lines.append('Sub-models: bull=ng1.0.1, bear=ng1.0.4-3s (per Step B Option C decision)\n')
    lines.append('## WF-OOS 2020-2026\n')
    lines.append('| Variant | V5.2 | Grade | Sharpe | MaxDD | Annual gross | Annual net | ICIR |')
    lines.append('|---|---:|---|---:|---:|---:|---:|---:|')
    for v, w, m in rows:
        if w != 'WF-OOS': continue
        lines.append(f'| {v} | {m.get("v52_score", "?")}% | {m.get("v52_grade", "?")} | '
                     f'{m.get("sharpe", "?")} | {m.get("maxdd", "?")}% | '
                     f'{m.get("annual_gross", "?")}% | {m.get("annual_net", "?")}% | '
                     f'{m.get("icir", "?")} |')
    lines.append('\n## Pre-2020 2018-2019\n')
    lines.append('| Variant | V5.2 (×0.85) | Grade | Sharpe | MaxDD | Annual gross |')
    lines.append('|---|---:|---|---:|---:|---:|')
    for v, w, m in rows:
        if w != 'Pre-2020': continue
        lines.append(f'| {v} | {m.get("v52_score", "?")}% | {m.get("v52_grade", "?")} | '
                     f'{m.get("sharpe", "?")} | {m.get("maxdd", "?")}% | '
                     f'{m.get("annual_gross", "?")}% |')

    # Acceptance gate evaluation
    lines.append('\n## Gate evaluation (Phase C acceptance)\n')
    lines.append('Acceptance: WF-OOS V5.2 ≥ 78% AND Pre-2020 V5.2 ≥ 38% AND Pre-2020 net annual ≥ -8%\n')
    lines.append('| Variant | WF-OOS V5.2 ≥78 | Pre-2020 V5.2 ≥38 | Pre-2020 annual_net ≥-8 | Verdict |')
    lines.append('|---|---|---|---|---|')
    for v in args.variants:
        wfo = next((m for vv, w, m in rows if vv == v and w == 'WF-OOS'), {})
        pre = next((m for vv, w, m in rows if vv == v and w == 'Pre-2020'), {})
        g1 = wfo.get('v52_score', 0) >= 78
        g2 = pre.get('v52_score', 0) >= 38
        g3 = pre.get('annual_net', -100) >= -8
        verdict = 'PASS' if (g1 and g2 and g3) else 'FAIL'
        lines.append(f'| {v} | {"OK" if g1 else "X"} {wfo.get("v52_score", "?")}% | '
                     f'{"OK" if g2 else "X"} {pre.get("v52_score", "?")}% | '
                     f'{"OK" if g3 else "X"} {pre.get("annual_net", "?")}% | **{verdict}** |')

    out.write_text('\n'.join(lines) + '\n')
    print(f'\nReport: {out}')


if __name__ == '__main__':
    main()
