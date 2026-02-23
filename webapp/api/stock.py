"""
个股数据API - 搜索、K线、技术指标、基本面
"""
import logging
from flask import Blueprint, jsonify, request, current_app

from core.database import DatabaseManager

logger = logging.getLogger(__name__)

stock_bp = Blueprint('stock', __name__)


def get_db_manager():
    """获取数据库管理器实例"""
    return DatabaseManager(
        current_app.config['STOCK_DB_PATH'],
        current_app.config['WEBAPP_DB_PATH']
    )


@stock_bp.route('/search', methods=['GET'])
def search():
    """
    搜索股票（代码/名称模糊匹配）

    Query Parameters:
        q: 搜索关键词
        limit: 返回数量限制 (默认20)
    """
    try:
        query = request.args.get('q', '').strip()
        limit = int(request.args.get('limit', 20))

        if not query or len(query) < 1:
            return jsonify({'success': True, 'results': []})

        db_manager = get_db_manager()
        results = db_manager.search_stocks(query, limit=limit)

        return jsonify({
            'success': True,
            'results': results
        })
    except Exception as e:
        logger.error(f'搜索股票失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@stock_bp.route('/<code>/kline', methods=['GET'])
def kline(code: str):
    """
    获取K线数据（OHLCV + MA）

    Query Parameters:
        days: 天数 (默认120)
    """
    try:
        days = int(request.args.get('days', 120))
        db_manager = get_db_manager()
        data = db_manager.get_stock_kline(code, days=days)

        return jsonify({
            'success': True,
            'data': data
        })
    except Exception as e:
        logger.error(f'获取K线数据失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@stock_bp.route('/<code>/technical', methods=['GET'])
def technical(code: str):
    """
    获取技术指标（KDJ/MACD/RSI/BOLL）

    Query Parameters:
        days: 天数 (默认120)
    """
    try:
        days = int(request.args.get('days', 120))
        db_manager = get_db_manager()
        data = db_manager.get_stock_technical(code, days=days)

        return jsonify({
            'success': True,
            'data': data
        })
    except Exception as e:
        logger.error(f'获取技术指标失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@stock_bp.route('/<code>/fundamental', methods=['GET'])
def fundamental(code: str):
    """获取基本面数据（PE/PB/市值/换手率）"""
    try:
        db_manager = get_db_manager()
        data = db_manager.get_stock_fundamental(code)

        return jsonify({
            'success': True,
            'data': data
        })
    except Exception as e:
        logger.error(f'获取基本面数据失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@stock_bp.route('/<code>/info', methods=['GET'])
def info(code: str):
    """获取公司信息"""
    try:
        db_manager = get_db_manager()
        data = db_manager.get_stock_info(code)

        return jsonify({
            'success': True,
            'data': data
        })
    except Exception as e:
        logger.error(f'获取公司信息失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500
