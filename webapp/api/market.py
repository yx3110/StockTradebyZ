"""
市场行情 API — 板块排行 / 资金流向 / 全A行情(市场情绪)

数据来源: stock_data.db 预计算表 (scripts/build_market_pulse.py 产出)
  - sector_daily_stats     板块日频统计 (taxonomy: sw_l1/sw_l2/concept)
  - market_sentiment_daily 市场情绪日频
"""
from flask import Blueprint, jsonify, request

from api._helpers import get_db_manager, api_error_handler

market_bp = Blueprint('market', __name__)

VALID_TAXONOMIES = {'sw_l1', 'sw_l2', 'concept'}


def _recent_dates(conn, table: str, days: int, where: str = '', params: tuple = ()):
    """表内最近 N 个交易日 (升序)"""
    rows = conn.execute(
        f"SELECT DISTINCT trade_date FROM {table} {where} "
        f"ORDER BY trade_date DESC LIMIT ?", (*params, days)).fetchall()
    return sorted(r[0] for r in rows)


def _parse_args():
    taxonomy = request.args.get('taxonomy', 'sw_l1')
    if taxonomy not in VALID_TAXONOMIES:
        taxonomy = 'sw_l1'
    days = min(max(int(request.args.get('days', 40)), 1), 250)
    return taxonomy, days


def _group_by_date(rows):
    by_date = {}
    for row in rows:
        by_date.setdefault(row[0], []).append(row[1:])
    return by_date


@market_bp.route('/rotation')
@api_error_handler
def rotation():
    """轮动日历: 每日涨幅 TOP 板块 + 每日20日新高个股数 TOP 板块"""
    taxonomy, days = _parse_args()
    top = min(int(request.args.get('top', 25)), 50)

    db = get_db_manager()
    with db.get_stock_db_connection() as conn:
        dates = _recent_dates(conn, 'sector_daily_stats', days,
                              'WHERE taxonomy=?', (taxonomy,))
        if not dates:
            return jsonify({'success': True, 'taxonomy': taxonomy, 'dates': [],
                            'calendar': {}, 'newhighs': {}})
        rows = conn.execute(
            """SELECT trade_date, sector_name, pct_change, newhigh_cnt
               FROM sector_daily_stats
               WHERE taxonomy=? AND trade_date BETWEEN ? AND ?""",
            (taxonomy, dates[0], dates[-1])).fetchall()

    calendar, newhighs = {}, {}
    for d, items in _group_by_date(rows).items():
        ranked = sorted((it for it in items if it[1] is not None),
                        key=lambda x: x[1], reverse=True)[:top]
        calendar[d] = [{'name': n, 'pct': round(p, 2)} for n, p, _ in ranked]
        nh_ranked = sorted((it for it in items if it[2] > 0),
                           key=lambda x: x[2], reverse=True)[:top]
        newhighs[d] = [{'name': n, 'cnt': c} for n, _, c in nh_ranked]

    return jsonify({'success': True, 'taxonomy': taxonomy, 'dates': dates,
                    'calendar': calendar, 'newhighs': newhighs})


@market_bp.route('/fundflow')
@api_error_handler
def fundflow():
    """资金流日历: 每日主力净流入 / 净流出 TOP 板块 (单位: 亿元)"""
    taxonomy, days = _parse_args()
    top = min(int(request.args.get('top', 20)), 50)

    db = get_db_manager()
    with db.get_stock_db_connection() as conn:
        dates = _recent_dates(conn, 'sector_daily_stats', days,
                              'WHERE taxonomy=?', (taxonomy,))
        if not dates:
            return jsonify({'success': True, 'taxonomy': taxonomy, 'dates': [],
                            'inflow': {}, 'outflow': {}})
        rows = conn.execute(
            """SELECT trade_date, sector_name, main_net_inflow
               FROM sector_daily_stats
               WHERE taxonomy=? AND trade_date BETWEEN ? AND ?
                 AND main_net_inflow IS NOT NULL""",
            (taxonomy, dates[0], dates[-1])).fetchall()

    inflow, outflow = {}, {}
    for d, items in _group_by_date(rows).items():
        ranked = sorted(items, key=lambda x: x[1], reverse=True)
        inflow[d] = [{'name': n, 'amount': round(a, 1)} for n, a in ranked[:top] if a > 0]
        outflow[d] = [{'name': n, 'amount': round(a, 1)} for n, a in ranked[::-1][:top] if a < 0]

    return jsonify({'success': True, 'taxonomy': taxonomy, 'dates': dates,
                    'inflow': inflow, 'outflow': outflow})


@market_bp.route('/sentiment')
@api_error_handler
def sentiment():
    """市场情绪时序: 涨停/跌停/炸板数, 涨跌家数, 沪深300/中证2000 20日新高数"""
    days = min(max(int(request.args.get('days', 250)), 1), 750)

    db = get_db_manager()
    with db.get_stock_db_connection() as conn:
        dates = _recent_dates(conn, 'market_sentiment_daily', days)
        if not dates:
            return jsonify({'success': True, 'dates': [], 'series': {}})
        rows = conn.execute(
            """SELECT trade_date, limit_up_cnt, limit_down_cnt, broken_cnt,
                      up_cnt, down_cnt, hs300_newhigh20, zz2000_newhigh20
               FROM market_sentiment_daily
               WHERE trade_date BETWEEN ? AND ? ORDER BY trade_date""",
            (dates[0], dates[-1])).fetchall()

    keys = ['limit_up', 'limit_down', 'broken', 'up_cnt', 'down_cnt',
            'hs300_newhigh20', 'zz2000_newhigh20']
    series = {k: [] for k in keys}
    out_dates = []
    for r in rows:
        out_dates.append(r[0])
        for i, k in enumerate(keys):
            series[k].append(r[i + 1])

    return jsonify({'success': True, 'dates': out_dates, 'series': series})


@market_bp.route('/status')
@api_error_handler
def status():
    """数据新鲜度: 各表最新日期 (页面头部展示 + 排查用)"""
    db = get_db_manager()
    info = {}
    with db.get_stock_db_connection() as conn:
        for key, sql in [
            ('sector_stats', "SELECT MAX(trade_date) FROM sector_daily_stats"),
            ('sentiment', "SELECT MAX(trade_date) FROM market_sentiment_daily"),
            ('quotes', "SELECT MAX(trade_date) FROM daily_quotes"),
        ]:
            try:
                info[key] = conn.execute(sql).fetchone()[0]
            except Exception:
                info[key] = None
    return jsonify({'success': True, **info})
