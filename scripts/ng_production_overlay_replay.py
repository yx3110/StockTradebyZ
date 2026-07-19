#!/usr/bin/env python3
"""生产 L1/L2/L4 列表级风控 overlay 重放 (评估-生产同构化, T02-a 2026-07-19).

把生产 ng1.0.6-v3 的列表级风控链重放到历史报告目录, 使北极星评估能同时输出
"裸信号" 与 "生产 overlay 重放" 两行数字 (审计遗留 must-do: 生产真实 MaxDD)。

三个重放脚本的分工 (本脚本为列表级重放权威):
  - 本脚本: 列表级 L1/L2/L4 重放 → 改写报告目录喂北极星 eval (T02-a)。
    regime 解析与 overlay 规则全部直调生产代码, 零第二副本:
      regime  = scoring_router._resolve_regime + _read_amv_regime
                (V11 0AMV, 含 staleness fail-defensive, 与生产 bit-exact)
      overlay = ng21_risk_overlay.build_risk_decision + apply_overlay_to_picks
                (含 2026-07-11 L4 加固三件套 hysteresis/cum5d/staleness)
  - scripts/production_overlay_replay.py (2026-07-11): 脚本内自算 NAV 的
    L3 VT sweep 专用; 其 regime map 为裸 SELECT 无 fail-defensive, 且 overlay
    只喂 top-60 (L1 percentile floor 口径与生产的全市场列表不同) —
    T02-b 的 eval 端 --vol-target/--stop-loss 网格落地后建议归档。
  - scripts/ng21_apply_overlay_to_reports.py (2026-04-28, ng2.1 历史):
    维护 L1/L2/L4 规则第二副本 + regime_v2 来源, 已归档留底。

已知语义缺口 (无法历史重放, 记录在案):
  - Signal Trust 红标剔除 / exec_warning (涨停停牌不占席位): 历史报告无该字段
  - post_filters industry_cap=3 与 overlay L2 (bull3/bear2) 在排序保留语义下
    等效收敛, 不单独重放
  - L3 VT sizing / L5 SL 是 NAV 级, 由 eval 侧 --vol-target/--stop-loss 承接 (T02-b)
  - 重放目录每日仅保留幸存者 (≤top_n 行), L1 IC / L6 归因等宇宙级指标不可与
    裸信号线混比; 重放线只看 NAV 链指标 (Sharpe/MaxDD/CVaR/换手/累积收益)
  - bear 日逐日调 build_risk_decision (每日 1 连接 2 小查询) 是刻意保真:
    保证 L4 看到的输入窗口与生产逐日运行完全一致, 开销 <3% 不值得批量化

Usage:
  python3 scripts/ng_production_overlay_replay.py \
      --src reports/daily_selection_ng101_3seed \
      --dst reports/daily_selection_ng101_3seed_prodoverlay
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from core.config import get_db_path  # noqa: E402
from stock_selctor.ng21_risk_overlay import (  # noqa: E402
    apply_overlay_to_picks,
    build_risk_decision,
    rank_key,
)
from stock_selctor.scoring_router import (  # noqa: E402
    _read_amv_regime,
    _resolve_regime,
)

try:
    import orjson

    def _load_json(p: Path) -> dict:
        return orjson.loads(p.read_bytes())

    def _dump_json(p: Path, d: dict) -> None:
        p.write_bytes(orjson.dumps(d))
except ImportError:
    def _load_json(p: Path) -> dict:
        with open(p, 'r') as fh:
            return json.load(fh)

    def _dump_json(p: Path, d: dict) -> None:
        with open(p, 'w') as fh:
            json.dump(d, fh, separators=(',', ':'))

DROPPED_AUDIT_HEAD = 30  # 每日留存最前排被剔股票的审计条数


def process_one(src_file: Path, dst_dir: Path, date_iso: str, db: str) -> dict:
    d = _load_json(src_file)
    stocks = d.get('all_stocks_with_scores', [])

    # 生产同源 regime: V11 0AMV + staleness fail-defensive (bit-exact)
    regime_int, _row_date, _label, degraded = _resolve_regime(
        lambda: _read_amv_regime(db, date_iso), 'prod-replay', date_iso, db)
    regime_label = 'bull' if regime_int == 1 else 'bear'

    decision = build_risk_decision(
        regime=regime_label,
        target_date=date_iso,
        db_path=db,
        base_top_n=10,
        regime_table='market_regime_signals',
    )
    # 生产同源: pick_pipeline 预排序 (rank_key desc) 后进 overlay
    kept, dropped = apply_overlay_to_picks(
        sorted(stocks, key=rank_key, reverse=True), decision)

    # 幸存者结构化保留 (含 overlay 附加的 _ng21_* sizing/SL 元数据);
    # 淘汰行直接移出列表 — 不依赖 eval parse 的 score>0 隐式契约
    d['all_stocks_with_scores'] = kept
    d['_prod_overlay_replay'] = {
        'regime': decision.regime,
        'regime_degraded': degraded,
        'crisis_active': decision.crisis_active,
        'top_n': decision.top_n,
        'industry_cap': decision.industry_cap,
        'vol_target_annual': decision.vol_target_annual,
        'stop_loss_pct': decision.stop_loss,
        'trailing_stop_pct': decision.trailing_stop,
        'cash_floor': decision.cash_floor,
        'n_kept': len(kept),
        'n_dropped': len(dropped),
        'dropped_head': [
            {'code': s.get('stock_code') or s.get('code'),
             'reason': s.get('_drop_reason')}
            for s in dropped[:DROPPED_AUDIT_HEAD]
        ],
    }
    _dump_json(dst_dir / src_file.name, d)
    return {'regime': decision.regime, 'crisis': decision.crisis_active,
            'n_kept': len(kept), 'degraded': bool(degraded)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--limit', type=int, default=0, help='仅处理前 N 天 (smoke test)')
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not src.exists():
        print(f'ERR: {src} missing', file=sys.stderr)
        sys.exit(2)
    dst.mkdir(parents=True, exist_ok=True)
    db = str(get_db_path())

    files = sorted(src.glob('analysis_data_*.json'))
    if args.limit:
        files = files[:args.limit]

    counts = Counter()
    for i, f in enumerate(files, 1):
        ymd = f.stem.replace('analysis_data_', '')
        date_iso = f'{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}'
        r = process_one(f, dst, date_iso, db)
        counts.update({
            r['regime']: 1,
            'crisis': int(r['crisis']),
            'defensive': int(r['degraded']),
            'kept_below_target': int(r['n_kept'] < (5 if r['crisis'] else 10)),
        })
        if i == 1:
            # loud-failure 锚定: 首文件回读, 确认输出列表 == 幸存者
            chk = _load_json(dst / f.name)
            n_out = len(chk['all_stocks_with_scores'])
            assert n_out == r['n_kept'], f'重放输出异常: {n_out} != {r["n_kept"]}'
        if i % 200 == 0:
            print(f'  {i}/{len(files)} ({date_iso})', flush=True)

    print(f'Done: {len(files)} days → bull={counts["bull"]}, '
          f'bear={counts["bear"]}, crisis_days={counts["crisis"]}, '
          f'fail_defensive={counts["defensive"]}, '
          f'kept_below_target={counts["kept_below_target"]}')


if __name__ == '__main__':
    main()
