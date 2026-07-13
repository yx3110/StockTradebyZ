"""
个股数据API - 搜索、K线、技术指标、基本面
"""
import logging
from flask import Blueprint, jsonify, request

from api._helpers import api_error_handler, get_db_manager, parse_int_arg

logger = logging.getLogger(__name__)

stock_bp = Blueprint('stock', __name__)


@stock_bp.route('/search', methods=['GET'])
@api_error_handler
def search():
    """
    搜索股票（代码/名称模糊匹配）

    Query Parameters:
        q: 搜索关键词
        limit: 返回数量限制 (默认20)
    """
    query = request.args.get('q', '').strip()
    limit = parse_int_arg('limit', 20)
    if not query:
        return jsonify({'success': True, 'results': []})
    results = get_db_manager().search_stocks(query, limit=limit)
    return jsonify({'success': True, 'results': results})


@stock_bp.route('/<code>/kline', methods=['GET'])
@api_error_handler
def kline(code: str):
    """获取K线数据（OHLCV + MA）"""
    days = parse_int_arg('days', 120)
    data = get_db_manager().get_stock_kline(code, days=days)
    return jsonify({'success': True, 'data': data})


@stock_bp.route('/<code>/technical', methods=['GET'])
@api_error_handler
def technical(code: str):
    """获取技术指标（KDJ/MACD/RSI/BOLL）"""
    days = parse_int_arg('days', 120)
    data = get_db_manager().get_stock_technical(code, days=days)
    return jsonify({'success': True, 'data': data})


@stock_bp.route('/<code>/fundamental', methods=['GET'])
@api_error_handler
def fundamental(code: str):
    """获取基本面数据（PE/PB/市值/换手率）"""
    data = get_db_manager().get_stock_fundamental(code)
    return jsonify({'success': True, 'data': data})


@stock_bp.route('/<code>/info', methods=['GET'])
@api_error_handler
def info(code: str):
    """获取公司信息"""
    data = get_db_manager().get_stock_info(code)
    return jsonify({'success': True, 'data': data})
