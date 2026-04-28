"""P2.7c: ng1.0.6+soft batch report generator.

Loads bull (ng1.0.1) + bear (ng1.0.4) scorers in same process, computes
EMA-smoothed P_bull from market_regime_signals, scores each stock with
both, blends via P × bull_score + (1-P) × bear_score, writes JSON in
ng1.0.6 report format.

Comparison vs production hard-switch (`ng1.0.6` v1):
  - Hard: load only one scorer per day, 100% commitment
  - Soft: load both, blend continuously, smoother regime transitions

Usage:
  python3 scripts/batch_generate_ng_soft.py \\
      --start-date 2026-04-01 --end-date 2026-04-24 \\
      --output-dir reports/daily_selection_ng106_soft

Note: ~2× memory + ~2× scoring cost vs hard mode (both pickles loaded). For
dev/eval; production wiring would happen in selector after positive A/B.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from ml_models.ng.ng_production_scorer import NGProductionScorer
from indicators.regime_classifier import compute_bull_proba, smooth_proba_ema


DB_PATH = ROOT / "data_adapter" / "stock_data.db"


def load_regime_proba(start: str, end: str) -> dict[str, float]:
    """Compute EMA-smoothed P_bull series, return {date: P} mapping."""
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        df = pd.read_sql_query(
            """
            SELECT trade_date, v11_var1, v11_ma60, v11_macd,
                   v11_bull, v11_streak
              FROM market_regime_signals
             WHERE trade_date BETWEEN date(?, '-30 days') AND ?
             ORDER BY trade_date
            """,
            conn, params=[start, end],
        )
    finally:
        conn.close()
    if df.empty:
        return {}
    streak_signed = np.where(df["v11_bull"] == 1, df["v11_streak"], -df["v11_streak"])
    p_raw = compute_bull_proba(
        df["v11_var1"].values, df["v11_ma60"].values,
        df["v11_macd"].values, streak_signed,
    )
    p_smooth = smooth_proba_ema(p_raw, span=5)
    return {d: float(p) for d, p in zip(df["trade_date"], p_smooth)}


def stock_universe(date: str, conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT s.code FROM securities s
          JOIN daily_quotes dq ON s.id = dq.security_id
         WHERE s.type = 'A股' AND dq.trade_date = ?
        """, (date,),
    ).fetchall()
    return [r[0] for r in rows]


def get_industries(codes: list[str]) -> dict[str, str]:
    if not codes:
        return {}
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    try:
        ph = ",".join("?" * len(codes))
        rows = conn.execute(
            f"SELECT code, name, industry FROM securities WHERE code IN ({ph})",
            codes,
        ).fetchall()
    finally:
        conn.close()
    return {r[0]: {"name": r[1], "industry": r[2] or ""} for r in rows}


def blend_score_dicts(bull_scores: dict, bear_scores: dict, p_bull: float) -> dict:
    """Per-stock blend bull + bear scorer outputs."""
    blended = {}
    all_codes = set(bull_scores.keys()) | set(bear_scores.keys())
    for c in all_codes:
        b = bull_scores.get(c, {})
        r = bear_scores.get(c, {})
        out = {}
        for k in ("score", "pred_3d", "pred_5d", "pred_10d", "pred_15d",
                 "rank_score", "composite"):
            bv = float(b.get(k, 0) or 0)
            rv = float(r.get(k, 0) or 0)
            out[k] = p_bull * bv + (1 - p_bull) * rv
        out["recommendation"] = b.get("recommendation") or r.get("recommendation") or "观望"
        out["_p_bull_used"] = p_bull
        blended[c] = out
    return blended


def generate_one_day(date: str, p_bull: float,
                     bull_scorer: NGProductionScorer,
                     bear_scorer: NGProductionScorer,
                     output_dir: Path) -> dict:
    print(f"\n=== {date} (P_bull={p_bull:.3f}) ===")
    t0 = time.time()
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    try:
        codes = stock_universe(date, conn)
    finally:
        conn.close()
    if not codes:
        print(f"  no stocks; skipping")
        return {"date": date, "n_stocks": 0}
    print(f"  universe: {len(codes)} stocks")

    bull_scores = bull_scorer.predict_scores(codes, date)
    bear_scores = bear_scorer.predict_scores(codes, date)
    blended = blend_score_dicts(bull_scores, bear_scores, p_bull)

    # Build report rows
    info = get_industries(codes)
    all_stocks = []
    for code, vals in blended.items():
        if vals.get("rank_score", 0) == 0 and vals.get("score", 0) == 0:
            continue
        meta = info.get(code, {})
        all_stocks.append({
            "stock_code": code,
            "stock_name": meta.get("name", ""),
            "industry": meta.get("industry", ""),
            "score": vals.get("score", 0),
            "pred_3d": vals.get("pred_3d", 0),
            "pred_5d": vals.get("pred_5d", 0),
            "pred_10d": vals.get("pred_10d", 0),
            "pred_15d": vals.get("pred_15d", 0),
            "rank_score": vals.get("rank_score", 0),
            "composite": vals.get("composite", 0),
            "recommendation": vals.get("recommendation", "观望"),
            "_p_bull_used": p_bull,
            "analysis_date": date,
        })
    all_stocks.sort(key=lambda s: -float(s.get("rank_score", 0) or 0))

    report = {
        "analysis_date": date,
        "scoring_version": "ng1.0.6+soft",
        "p_bull": p_bull,
        "all_stocks_with_scores": all_stocks,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"analysis_data_{date.replace('-', '')}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    elapsed = time.time() - t0
    print(f"  wrote {out_path} ({len(all_stocks)} stocks, {elapsed:.1f}s)")
    return {"date": date, "n_stocks": len(all_stocks), "elapsed_s": elapsed,
            "p_bull": p_bull}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--output-dir",
                    default="reports/daily_selection_ng106_soft")
    ap.add_argument("--bull-version", default="ng1.0.1")
    ap.add_argument("--bear-version", default="ng1.0.4")
    args = ap.parse_args()

    print("[setup] computing EMA-smoothed P_bull series...")
    proba_map = load_regime_proba(args.start_date, args.end_date)
    if not proba_map:
        print("ERROR: empty regime proba", file=sys.stderr)
        return 2
    print(f"  P_bull computed for {len(proba_map)} dates")

    print(f"[setup] loading {args.bull_version} scorer (bull expert)...")
    bull = NGProductionScorer(version=args.bull_version)
    print(f"[setup] loading {args.bear_version} scorer (bear expert)...")
    bear = NGProductionScorer(version=args.bear_version)

    # Trading dates from regime panel (it's already filtered to trading days)
    dates = [d for d in sorted(proba_map.keys())
             if args.start_date <= d <= args.end_date]
    print(f"[run] processing {len(dates)} trading days...")

    results = []
    for d in dates:
        try:
            r = generate_one_day(d, proba_map[d], bull, bear,
                                 Path(args.output_dir))
            results.append(r)
        except Exception as e:
            print(f"  FAILED on {d}: {e}", file=sys.stderr)
            results.append({"date": d, "error": str(e)})

    print("\n[summary]")
    succ = [r for r in results if "error" not in r]
    print(f"  succeeded: {len(succ)}/{len(results)}")
    if succ:
        avg_n = np.mean([r["n_stocks"] for r in succ])
        avg_p = np.mean([r["p_bull"] for r in succ])
        avg_t = np.mean([r["elapsed_s"] for r in succ])
        print(f"  avg stocks/day: {avg_n:.0f}")
        print(f"  avg P_bull: {avg_p:.3f}")
        print(f"  avg elapsed: {avg_t:.1f}s")
    return 0 if all("error" not in r for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
