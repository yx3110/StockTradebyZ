#!/usr/bin/env python3
"""IC Stability Analyzer — 6-regime IC analysis for feature screening.

Loads feature data from an NG feature cache table, computes Spearman IC per
feature across 5 market regimes, and classifies each feature as STABLE, FLIP,
or UNSTABLE.  Outputs a markdown report.

Usage:
    python3 scripts/ic_stability_analyzer.py \\
      --cache-table ng104_feature_cache \\
      --label ra_label_10d \\
      --output reports/ic_stability_ng104.md
"""

import sys
import os
import json
import sqlite3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
DB_PATH = os.path.join(PROJECT_ROOT, "data_adapter", "stock_data.db")

try:
    import orjson
    _json_loads = orjson.loads
except ImportError:
    _json_loads = json.loads

# ── Regime thresholds ────────────────────────────────────────────────────────
BULL_THRESHOLD = 0.05       # market_return_20d > 5 %  → bull
BEAR_THRESHOLD = -0.05      # market_return_20d < -5 % → bear
IC_SIGN_MIN = 0.01          # |IC| must exceed this to count as a sign signal
MIN_SAMPLES = 500           # minimum rows per regime for valid IC
CV_UNSTABLE_THRESHOLD = 2.0 # coefficient of variation above this → UNSTABLE


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(cache_table: str, label_col: str) -> pd.DataFrame:
    """Load feature + label rows from the cache table.

    Returns a DataFrame with one column per feature plus the label column and
    the two market-condition columns needed for regime definition.
    Missing market columns fall back to NaN (handled in define_regimes).
    """
    required_market_cols = ["market_return_20d", "market_volatility_20d"]

    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        # Discover which columns actually exist in this table
        cur = conn.execute(f"PRAGMA table_info({cache_table})")
        all_cols = {row[1] for row in cur.fetchall()}

    if label_col not in all_cols:
        raise ValueError(
            f"Label column '{label_col}' not found in {cache_table}. "
            f"Available label-like columns: "
            f"{[c for c in all_cols if 'label' in c or 'ra_' in c]}"
        )

    select_cols = [label_col] + [c for c in required_market_cols if c in all_cols]
    select_sql = ", ".join(select_cols)

    print(f"Loading data from {cache_table} …", flush=True)
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        rows = conn.execute(
            f"SELECT features_json, {select_sql} FROM {cache_table} "
            f"WHERE {label_col} IS NOT NULL"
        ).fetchall()

    if not rows:
        raise ValueError(f"No rows with non-null {label_col} in {cache_table}")

    print(f"  Parsing {len(rows):,} rows …", flush=True)

    # Parse JSON features
    feat_dicts = []
    labels = []
    market_return = []
    market_vol = []

    market_return_idx = 1 + ([c for c in required_market_cols if c in all_cols].index("market_return_20d")
                              if "market_return_20d" in all_cols else -1)
    market_vol_idx = 1 + ([c for c in required_market_cols if c in all_cols].index("market_volatility_20d")
                           if "market_volatility_20d" in all_cols else -1)

    for row in rows:
        feat_dicts.append(_json_loads(row[0]))
        labels.append(row[1])
        market_return.append(row[market_return_idx] if "market_return_20d" in all_cols else np.nan)
        market_vol.append(row[market_vol_idx] if "market_volatility_20d" in all_cols else np.nan)

    df = pd.DataFrame(feat_dicts)
    df[label_col] = labels
    df["market_return_20d"] = market_return
    df["market_volatility_20d"] = market_vol

    # Drop rows where label is NaN (belt-and-suspenders after SQL filter)
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows, {len(df.columns) - 3} features", flush=True)
    return df


# ── Regime definitions ───────────────────────────────────────────────────────

def define_regimes(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Return dict of regime_name → boolean mask Series.

    Five regimes:
      bull      : market_return_20d  > 5 %
      bear      : market_return_20d  < -5 %
      sideways  : -5 % ≤ market_return_20d ≤ 5 %
      high_vol  : market_volatility_20d > 75th-percentile
      low_vol   : market_volatility_20d ≤ 75th-percentile

    Rows where the relevant column is NaN are excluded from that regime mask.
    """
    ret = df["market_return_20d"]
    vol = df["market_volatility_20d"]
    vol_p75 = vol.quantile(0.75)

    ret_valid = ret.notna()
    vol_valid = vol.notna()

    regimes = {
        "bull":     ret_valid & (ret > BULL_THRESHOLD),
        "bear":     ret_valid & (ret < BEAR_THRESHOLD),
        "sideways": ret_valid & (ret >= BEAR_THRESHOLD) & (ret <= BULL_THRESHOLD),
        "high_vol": vol_valid & (vol > vol_p75),
        "low_vol":  vol_valid & (vol <= vol_p75),
    }

    for name, mask in regimes.items():
        print(f"  Regime '{name}': {mask.sum():,} samples", flush=True)

    return regimes


# ── Per-feature IC computation ────────────────────────────────────────────────

def compute_regime_ic(
    df: pd.DataFrame,
    feature_col: str,
    label_col: str,
    regimes: dict[str, pd.Series],
) -> dict[str, float | None]:
    """Compute Spearman IC for feature_col vs label_col in each regime.

    Returns dict of regime_name → IC (float) or None if insufficient samples.
    """
    result: dict[str, float | None] = {}
    feat_series = df[feature_col]

    for regime_name, mask in regimes.items():
        sub = df[mask & feat_series.notna() & df[label_col].notna()]
        if len(sub) < MIN_SAMPLES:
            result[regime_name] = None
            continue
        ic, _ = spearmanr(sub[feature_col], sub[label_col])
        result[regime_name] = float(ic) if not np.isnan(ic) else None

    return result


# ── Stability classification ──────────────────────────────────────────────────

def classify_stability(
    regime_ics: dict[str, float | None],
) -> tuple[str, float | None]:
    """Classify feature stability from per-regime IC values.

    Returns (flag, ic_cv) where flag is one of:
      STABLE       — IC signs consistent, low variance
      FLIP         — IC sign flips across regimes (with |IC| > threshold)
      UNSTABLE     — IC coefficient of variation > CV_UNSTABLE_THRESHOLD
      INSUFFICIENT — fewer than 2 regimes have valid IC

    ic_cv is the coefficient of variation across valid IC values, or None.
    """
    valid_ics = [v for v in regime_ics.values() if v is not None]

    if len(valid_ics) < 2:
        return "INSUFFICIENT", None

    ic_array = np.array(valid_ics)
    ic_mean = np.mean(ic_array)
    ic_std = np.std(ic_array)
    ic_cv = ic_std / (abs(ic_mean) + 1e-8)

    # FLIP: signs disagree among regimes where |IC| > threshold
    sign_ics = [v for v in valid_ics if abs(v) > IC_SIGN_MIN]
    if len(sign_ics) >= 2:
        signs = set(np.sign(v) for v in sign_ics)
        if len(signs) > 1:
            return "FLIP", ic_cv

    # UNSTABLE: high variance even if no sign flip
    if ic_cv > CV_UNSTABLE_THRESHOLD:
        return "UNSTABLE", ic_cv

    return "STABLE", ic_cv


# ── Report generation ─────────────────────────────────────────────────────────

def generate_report(
    results: list[dict],
    output_path: str,
    cache_table: str,
    label_col: str,
    total_rows: int,
) -> None:
    """Write markdown report with per-feature IC stability table."""
    regime_names = ["bull", "bear", "sideways", "high_vol", "low_vol"]

    # Sort: FLIP first, then UNSTABLE, then INSUFFICIENT, then STABLE
    order = {"FLIP": 0, "UNSTABLE": 1, "INSUFFICIENT": 2, "STABLE": 3}
    results_sorted = sorted(results, key=lambda r: (order.get(r["flag"], 9), -(r["ic_cv"] or 0)))

    # Summary counts
    counts = {k: sum(1 for r in results if r["flag"] == k) for k in order}

    lines = [
        f"# IC Stability Analysis Report",
        f"",
        f"**Cache table:** `{cache_table}`  ",
        f"**Label column:** `{label_col}`  ",
        f"**Total rows loaded:** {total_rows:,}  ",
        f"**Min samples per regime:** {MIN_SAMPLES:,}  ",
        f"**IC sign threshold:** {IC_SIGN_MIN}  ",
        f"**CV unstable threshold:** {CV_UNSTABLE_THRESHOLD}  ",
        f"",
        f"## Summary",
        f"",
        f"| Flag | Count | Description |",
        f"|------|------:|-------------|",
        f"| FLIP | {counts.get('FLIP', 0)} | IC sign reverses across regimes — exclude or handle with care |",
        f"| UNSTABLE | {counts.get('UNSTABLE', 0)} | IC variance high (CV > {CV_UNSTABLE_THRESHOLD}) — noisy signal |",
        f"| INSUFFICIENT | {counts.get('INSUFFICIENT', 0)} | < 2 regimes with ≥ {MIN_SAMPLES} samples |",
        f"| STABLE | {counts.get('STABLE', 0)} | Consistent IC direction — keep |",
        f"",
        f"## Per-feature IC Table",
        f"",
        f"Columns: `bull` / `bear` / `sideways` — market-return regimes; "
        f"`high_vol` / `low_vol` — volatility regimes.",
        f"`—` = insufficient samples (< {MIN_SAMPLES}).  "
        f"Values are Spearman IC × 100.",
        f"",
    ]

    # Table header
    header = "| Feature | Flag | IC_CV |"
    separator = "|---------|------|------:|"
    for rn in regime_names:
        header += f" {rn} |"
        separator += "------:|"
    lines += [header, separator]

    for r in results_sorted:
        ic_cv_str = f"{r['ic_cv']:.2f}" if r["ic_cv"] is not None else "—"
        row = f"| `{r['feature']}` | **{r['flag']}** | {ic_cv_str} |"
        for rn in regime_names:
            v = r["regime_ics"].get(rn)
            row += f" {v*100:+.2f} |" if v is not None else " — |"
        lines.append(row)

    # Flip features section
    flip_features = [r["feature"] for r in results_sorted if r["flag"] == "FLIP"]
    if flip_features:
        lines += [
            f"",
            f"## FLIP Features (exclude from ng1.0.4)",
            f"",
            f"These features show IC sign reversal across market regimes and should be",
            f"excluded or replaced with regime-conditioned variants.",
            f"",
        ]
        for f in flip_features:
            lines.append(f"- `{f}`")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport saved → {output_path}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    global MIN_SAMPLES, CV_UNSTABLE_THRESHOLD  # noqa: PLW0603
    parser = argparse.ArgumentParser(
        description="Analyze IC stability across market regimes for NG feature cache tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--cache-table",
        default="ng103_feature_cache",
        help="SQLite table name (default: ng103_feature_cache)",
    )
    parser.add_argument(
        "--label",
        default="label_10d",
        help="Label column to use for IC computation (default: label_10d)",
    )
    parser.add_argument(
        "--output",
        default="reports/ic_stability.md",
        help="Output markdown report path (default: reports/ic_stability.md)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=MIN_SAMPLES,
        help=f"Min samples per regime (default: {MIN_SAMPLES})",
    )
    parser.add_argument(
        "--cv-threshold",
        type=float,
        default=CV_UNSTABLE_THRESHOLD,
        help=f"CV threshold for UNSTABLE (default: {CV_UNSTABLE_THRESHOLD})",
    )
    args = parser.parse_args()

    # Apply CLI overrides to module-level thresholds
    MIN_SAMPLES = args.min_samples
    CV_UNSTABLE_THRESHOLD = args.cv_threshold

    # Resolve output path relative to project root
    output_path = (
        args.output
        if os.path.isabs(args.output)
        else os.path.join(PROJECT_ROOT, args.output)
    )

    print(f"=== IC Stability Analyzer ===")
    print(f"Table : {args.cache_table}")
    print(f"Label : {args.label}")
    print(f"Output: {output_path}")
    print()

    # 1. Load data
    df = load_data(args.cache_table, args.label)

    # 2. Define regimes
    print("\nDefining market regimes …")
    regimes = define_regimes(df)

    # 3. Identify feature columns (exclude label + market columns)
    non_feature_cols = {
        args.label,
        "market_return_20d",
        "market_volatility_20d",
        # also exclude other label / market columns that may bleed in
        "label_3d", "label_5d", "label_10d", "label_15d",
        "label_raw_3d", "label_raw_5d", "label_raw_10d", "label_raw_15d",
        "downside_10d",
        "maxdd_3d", "maxdd_5d", "maxdd_10d", "maxdd_15d",
        "ra_label_3d", "ra_label_5d", "ra_label_10d", "ra_label_15d",
        "market_return_5d", "market_breadth", "market_new_high_ratio",
        "northbound_flow_5d", "market_volume_ratio", "market_drawdown",
        "vix_proxy", "market_momentum_diff",
    }
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    print(f"\nAnalyzing {len(feature_cols)} features across {len(regimes)} regimes …")

    # 4. Compute IC and classify each feature
    results = []
    for i, feat in enumerate(feature_cols):
        if (i + 1) % 20 == 0 or i == len(feature_cols) - 1:
            print(f"  [{i+1:3d}/{len(feature_cols)}] {feat}", flush=True)

        regime_ics = compute_regime_ic(df, feat, args.label, regimes)
        flag, ic_cv = classify_stability(regime_ics)
        results.append({
            "feature": feat,
            "flag": flag,
            "ic_cv": ic_cv,
            "regime_ics": regime_ics,
        })

    # 5. Print summary to console
    from collections import Counter
    flag_counts = Counter(r["flag"] for r in results)
    print(f"\n{'='*50}")
    print(f"Results for table={args.cache_table}, label={args.label}")
    for flag in ("FLIP", "UNSTABLE", "INSUFFICIENT", "STABLE"):
        print(f"  {flag:14s}: {flag_counts.get(flag, 0)}")

    flip_feats = [r["feature"] for r in results if r["flag"] == "FLIP"]
    if flip_feats:
        print(f"\nFLIP features ({len(flip_feats)}):")
        for f in flip_feats:
            print(f"  - {f}")

    # 6. Generate report
    generate_report(
        results,
        output_path,
        cache_table=args.cache_table,
        label_col=args.label,
        total_rows=len(df),
    )


if __name__ == "__main__":
    main()
