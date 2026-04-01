#!/usr/bin/env python3
"""
回测不同模型加权方案对综合选股的影响。

从 analysis_data JSON 加载历史选股数据，用不同加权方式合成 Top10，
计算实际 10 日收益，对比各方案表现。
"""

import json
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')

# ── 从缓存自动加载模型V4分数 ──
def _load_models_from_cache(top_n=4):
    """从V4缓存和discover自动构建MODELS字典。"""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ensemble_daily_recommend import discover_available_models, _load_v4_cache

    candidates = discover_available_models()
    cache = _load_v4_cache()

    models = {}
    # 按V4分数排序
    scored = []
    for version, report_dir in candidates.items():
        cached = cache.get(version)
        if cached and cached.get('final_pct', 0) > 0:
            scored.append((version, report_dir, cached['final_pct']))

    scored.sort(key=lambda x: -x[2])

    for version, report_dir, v4_pct in scored[:top_n]:
        models[version] = {'dir': report_dir, 'v4_pct': v4_pct}

    return models


MODELS = _load_models_from_cache(top_n=4)


def load_model_rankings(model_dir: str, date_str: str, top_n: int = 30) -> dict:
    """加载单个模型某天的 Top N 股票排名。

    Returns: {code: {'rank': int, 'pred_10d': float, 'name': str}}
    """
    fpath = PROJECT_ROOT / 'reports' / model_dir / f'analysis_data_{date_str}.json'
    if not fpath.exists():
        return {}

    with open(fpath, 'r') as f:
        data = json.load(f)

    stocks = data.get('all_stocks_with_scores', [])
    result = {}
    for i, s in enumerate(stocks[:top_n]):
        code = s.get('stock_code', '')
        result[code] = {
            'rank': i + 1,
            'pred_10d': s.get('pred_10d', 0),
            'name': s.get('stock_name', ''),
        }
    return result


def get_actual_returns(codes: list, buy_date: str, hold_days: int = 10) -> dict:
    """获取一组股票从 buy_date 开始持有 hold_days 的实际收益。

    buy_date: 买入日 (报告日的下一个交易日)
    Returns: {code: actual_return_pct}
    """
    conn = sqlite3.connect(DB_PATH)
    placeholders = ','.join(['?'] * len(codes))

    # 获取 buy_date 之后的交易日
    df = pd.read_sql_query(f"""
        SELECT dq.trade_date, s.code, dq.close
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.code IN ({placeholders})
          AND dq.trade_date >= ?
        ORDER BY dq.trade_date
    """, conn, params=codes + [buy_date])
    conn.close()

    if df.empty:
        return {}

    results = {}
    for code in codes:
        sub = df[df['code'] == code].sort_values('trade_date')
        if len(sub) < 2:
            continue
        # 买入价 = buy_date 的收盘价
        buy_price = sub.iloc[0]['close']
        # 卖出价 = hold_days 个交易日后的收盘价
        sell_idx = min(hold_days, len(sub) - 1)
        sell_price = sub.iloc[sell_idx]['close']
        results[code] = (sell_price - buy_price) / buy_price

    return results


def get_next_trading_date(date_str: str) -> str:
    """获取 date_str 之后的下一个交易日。"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT DISTINCT trade_date FROM daily_quotes
        WHERE trade_date > ? ORDER BY trade_date LIMIT 1
    """, (date_str,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


def ensemble_score_equal(all_rankings: dict, top_n: int = 20) -> list:
    """等权方案: count↓ + avg_rank↑ (当前方法)"""
    stocks = defaultdict(lambda: {'ranks': [], 'name': ''})
    for model, rankings in all_rankings.items():
        for code, info in rankings.items():
            stocks[code]['ranks'].append(info['rank'])
            stocks[code]['name'] = info['name']

    scored = []
    for code, info in stocks.items():
        count = len(info['ranks'])
        avg_rank = np.mean(info['ranks'])
        # score: 更多命中+更小排名 = 更好
        score = count * 1000 - avg_rank
        scored.append((code, info['name'], score, count, avg_rank))

    scored.sort(key=lambda x: -x[2])
    return [(c, n) for c, n, *_ in scored[:top_n]]


def ensemble_score_v4_weighted(all_rankings: dict, model_weights: dict,
                                top_n: int = 20, rank_pool: int = 30) -> list:
    """V4加权方案: 每个模型按V4分数加权，排名越高得分越多。

    score = Σ model_weight × (rank_pool + 1 - rank) / rank_pool
    """
    stocks = defaultdict(lambda: {'score': 0.0, 'name': ''})
    for model, rankings in all_rankings.items():
        w = model_weights.get(model, 1.0)
        for code, info in rankings.items():
            rank = info['rank']
            rank_score = (rank_pool + 1 - rank) / rank_pool
            stocks[code]['score'] += w * rank_score
            stocks[code]['name'] = info['name']

    scored = [(code, info['name'], info['score'])
              for code, info in stocks.items()]
    scored.sort(key=lambda x: -x[2])
    return [(c, n) for c, n, _ in scored[:top_n]]


def ensemble_score_v4_exp(all_rankings: dict, model_weights: dict,
                           top_n: int = 20, rank_pool: int = 30,
                           alpha: float = 2.0) -> list:
    """指数加权: 高排名股票得分指数放大。

    score = Σ model_weight × exp(-alpha × (rank-1) / rank_pool)
    """
    stocks = defaultdict(lambda: {'score': 0.0, 'name': ''})
    for model, rankings in all_rankings.items():
        w = model_weights.get(model, 1.0)
        for code, info in rankings.items():
            rank = info['rank']
            rank_score = np.exp(-alpha * (rank - 1) / rank_pool)
            stocks[code]['score'] += w * rank_score
            stocks[code]['name'] = info['name']

    scored = [(code, info['name'], info['score'])
              for code, info in stocks.items()]
    scored.sort(key=lambda x: -x[2])
    return [(c, n) for c, n, _ in scored[:top_n]]


def ensemble_single_best(all_rankings: dict, best_model: str,
                          top_n: int = 10) -> list:
    """单模型基线: 只用V4最佳模型。"""
    rankings = all_rankings.get(best_model, {})
    sorted_stocks = sorted(rankings.items(), key=lambda x: x[1]['rank'])
    return [(code, info['name']) for code, info in sorted_stocks[:top_n]]


def ensemble_best_with_consensus(all_rankings: dict, best_model: str,
                                  top_n: int = 10, min_hits: int = 2,
                                  pool: int = 30) -> list:
    """最佳模型 + 共识过滤: 只从被≥min_hits个模型都看好的股票中，按最佳模型排名选。"""
    # 统计每只股票被几个模型选入 top pool
    hit_count = defaultdict(int)
    for model, rankings in all_rankings.items():
        for code in rankings:
            hit_count[code] += 1

    # 只保留被足够多模型看好的
    consensus_codes = {c for c, h in hit_count.items() if h >= min_hits}

    # 按最佳模型排名筛选
    rankings = all_rankings.get(best_model, {})
    filtered = [(code, info) for code, info in rankings.items()
                if code in consensus_codes]
    filtered.sort(key=lambda x: x[1]['rank'])
    return [(code, info['name']) for code, info in filtered[:top_n]]


def ensemble_weighted_top_heavy(all_rankings: dict, model_weights: dict,
                                 top_n: int = 10, rank_pool: int = 30) -> list:
    """重头加权: 第1名得分远大于第10名 (1/rank 加权)。"""
    stocks = defaultdict(lambda: {'score': 0.0, 'name': ''})
    for model, rankings in all_rankings.items():
        w = model_weights.get(model, 1.0)
        for code, info in rankings.items():
            rank = info['rank']
            rank_score = 1.0 / rank  # 第1名=1.0, 第10名=0.1
            stocks[code]['score'] += w * rank_score
            stocks[code]['name'] = info['name']

    scored = [(code, info['name'], info['score'])
              for code, info in stocks.items()]
    scored.sort(key=lambda x: -x[2])
    return [(c, n) for c, n, _ in scored[:top_n]]


def run_backtest():
    """主回测流程。"""

    # 找出所有模型共同有数据的交易日
    all_dates = None
    for model, info in MODELS.items():
        d = PROJECT_ROOT / 'reports' / info['dir']
        dates = {f.stem.replace('analysis_data_', '')
                 for f in d.glob('analysis_data_*.json')}
        all_dates = dates if all_dates is None else all_dates & dates

    all_dates = sorted(all_dates)
    # 留出最后20天做收益计算
    all_dates = all_dates[:-20]

    print(f"回测窗口: {all_dates[0]} → {all_dates[-1]} ({len(all_dates)} 天)")
    print(f"模型: {', '.join(MODELS.keys())}")

    # V4 权重方案
    v4_scores = {m: info['v4_pct'] for m, info in MODELS.items()}
    total_v4 = sum(v4_scores.values())

    # 归一化权重
    w_equal = {m: 1.0 for m in MODELS}
    w_v4_linear = {m: s / total_v4 * len(MODELS) for m, s in v4_scores.items()}
    w_v4_squared = {m: (s / total_v4)**2 * len(MODELS)**2 for m, s in v4_scores.items()}

    # 排名权重: 第1名=4, 第2名=3, ...
    sorted_models = sorted(v4_scores.items(), key=lambda x: -x[1])
    w_rank = {}
    for i, (m, _) in enumerate(sorted_models):
        w_rank[m] = len(MODELS) - i

    # V4差异放大: 相对于最低分的差值作为权重
    min_v4 = min(v4_scores.values())
    w_v4_delta = {m: 1.0 + (s - min_v4) / 10.0 for m, s in v4_scores.items()}

    best_model = sorted_models[0][0]
    second_model = sorted_models[1][0]

    strategies = {
        # ── 基线 ──
        'single_best':    (f'单模型({best_model})',     lambda r: ensemble_single_best(r, best_model, 10)),
        'single_2nd':     (f'单模型({second_model})',    lambda r: ensemble_single_best(r, second_model, 10)),
        # ── 等权 ──
        'equal':          ('4模型等权',                  lambda r: ensemble_score_equal(r, 10)),
        # ── V4加权系列 ──
        'v4_linear':      ('V4线性加权',                 lambda r: ensemble_score_v4_weighted(r, w_v4_linear, 10)),
        'v4_squared':     ('V4平方加权',                 lambda r: ensemble_score_v4_weighted(r, w_v4_squared, 10)),
        'rank_4321':      ('排名加权(4321)',              lambda r: ensemble_score_v4_weighted(r, w_rank, 10)),
        'v4_delta':       ('V4差异加权',                 lambda r: ensemble_score_v4_weighted(r, w_v4_delta, 10)),
        # ── 排名衰减系列 ──
        'inv_rank_v4':    ('V4+1/rank',                 lambda r: ensemble_weighted_top_heavy(r, w_v4_linear, 10)),
        'inv_rank_equal': ('等权+1/rank',                lambda r: ensemble_weighted_top_heavy(r, w_equal, 10)),
        'v4_exp3':        ('V4+exp衰减α3',              lambda r: ensemble_score_v4_exp(r, w_v4_linear, 10, alpha=3.0)),
        'v4_exp5':        ('V4+exp衰减α5',              lambda r: ensemble_score_v4_exp(r, w_v4_linear, 10, alpha=5.0)),
        # ── 最佳+共识过滤 ──
        'best_cons2':     (f'{best_model}+共识≥2',      lambda r: ensemble_best_with_consensus(r, best_model, 10, min_hits=2)),
        'best_cons3':     (f'{best_model}+共识≥3',      lambda r: ensemble_best_with_consensus(r, best_model, 10, min_hits=3)),
    }

    print(f"\n权重设置:")
    for m in MODELS:
        print(f"  {m}: V4={v4_scores[m]:.1f}%  "
              f"equal={w_equal[m]:.2f}  linear={w_v4_linear[m]:.2f}  "
              f"squared={w_v4_squared[m]:.2f}  rank={w_rank[m]}  "
              f"delta={w_v4_delta[m]:.2f}")

    # 回测
    results = {name: [] for name in strategies}
    processed = 0

    for date_str in all_dates:
        # 加载所有模型的排名
        all_rankings = {}
        for model, info in MODELS.items():
            rankings = load_model_rankings(info['dir'], date_str, top_n=30)
            if rankings:
                all_rankings[model] = rankings

        if len(all_rankings) < 3:  # 至少3个模型有数据
            continue

        # 获取下一个交易日 (买入日)
        dt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        buy_date = get_next_trading_date(dt)
        if not buy_date:
            continue

        # 收集所有候选股票
        all_codes = set()
        for rankings in all_rankings.values():
            all_codes.update(rankings.keys())

        # 获取实际10日收益
        actual_returns = get_actual_returns(list(all_codes), buy_date, hold_days=10)
        if not actual_returns:
            continue

        # 对每个策略计算选股和收益
        for strat_name, (_, select_fn) in strategies.items():
            selected = select_fn(all_rankings)
            codes = [c for c, _ in selected]

            # 等权组合收益
            rets = [actual_returns.get(c, 0) for c in codes if c in actual_returns]
            if rets:
                avg_ret = np.mean(rets)
                results[strat_name].append({
                    'date': date_str,
                    'return': avg_ret,
                    'n_stocks': len(rets),
                    'positive': sum(1 for r in rets if r > 0),
                })

        processed += 1
        if processed % 50 == 0:
            print(f"  进度: {processed}/{len(all_dates)}")

    print(f"\n  完成: {processed} 个交易日")

    # ── 汇总结果 ──
    print(f"\n{'=' * 100}")
    print(f"🏆 加权方案回测对比 ({all_dates[0]} → {all_dates[-1]}, {processed}天, 10日持仓)")
    print(f"{'=' * 100}")
    print(f"{'策略':<22}{'日均收益':<10}{'年化收益':<10}{'胜率':<8}"
          f"{'Sharpe':<10}{'最大回撤':<10}{'波动率':<10}{'盈亏比':<10}")
    print("-" * 100)

    summary_rows = []
    for strat_name, (display_name, _) in strategies.items():
        data = results[strat_name]
        if not data:
            continue

        rets = [d['return'] for d in data]
        rets_arr = np.array(rets)

        daily_mean = np.mean(rets_arr)
        annual_ret = daily_mean * 252 / 10  # 10日持仓
        win_rate = np.mean(rets_arr > 0) * 100
        vol = np.std(rets_arr) * np.sqrt(252 / 10)
        sharpe = annual_ret / vol if vol > 0 else 0

        # 最大回撤 (简化)
        cum = np.cumprod(1 + rets_arr)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        max_dd = np.min(dd)

        # 盈亏比
        wins = rets_arr[rets_arr > 0]
        losses = rets_arr[rets_arr < 0]
        pnl_ratio = (np.mean(wins) / abs(np.mean(losses))) if len(losses) > 0 and np.mean(losses) != 0 else float('inf')

        print(f"{display_name:<22}{daily_mean:>+8.3%}  {annual_ret:>+8.1%}  "
              f"{win_rate:>5.1f}%  {sharpe:>7.3f}   {max_dd:>+8.1%}  "
              f"{vol:>8.1%}  {pnl_ratio:>7.2f}")

        summary_rows.append({
            'strategy': display_name,
            'daily_mean': daily_mean,
            'annual_return': annual_ret,
            'win_rate': win_rate,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'volatility': vol,
            'pnl_ratio': pnl_ratio,
        })

    # 找最优
    if summary_rows:
        best = max(summary_rows, key=lambda x: x['sharpe'])
        print(f"\n🥇 最优方案 (Sharpe): {best['strategy']}"
              f" — Sharpe={best['sharpe']:.3f}, 年化={best['annual_return']:+.1%}")

        best_ret = max(summary_rows, key=lambda x: x['annual_return'])
        print(f"🥇 最优方案 (收益): {best_ret['strategy']}"
              f" — 年化={best_ret['annual_return']:+.1%}, Sharpe={best_ret['sharpe']:.3f}")

    # 保存结果
    output_file = PROJECT_ROOT / 'reports' / 'ensemble_recommend' / 'weighting_backtest.json'
    with open(output_file, 'w') as f:
        json.dump({
            'window': f"{all_dates[0]} → {all_dates[-1]}",
            'n_days': processed,
            'models': {m: info['v4_pct'] for m, info in MODELS.items()},
            'strategies': summary_rows,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n💾 结果保存: {output_file}")


if __name__ == '__main__':
    run_backtest()
