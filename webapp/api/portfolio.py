"""
持仓管理API - 持仓、交易记录、操作建议、评估
"""
import logging
from datetime import datetime, timedelta
from flask import Blueprint, jsonify, request, current_app
import json
import sys
from pathlib import Path

from core.database import DatabaseManager
from core.position_analyzer import PositionAnalyzer
from core.portfolio_importer import parse_csv, parse_web_paste, parse_html_table, merge_positions, parse_trade_html_table, merge_trades
from core.portfolio_scorer import PortfolioScorer
from core.portfolio_manager import PortfolioManager

logger = logging.getLogger(__name__)

# 全局PositionAnalyzer实例
_position_analyzer = None
_portfolio_manager = None

def get_position_analyzer():
    """获取PositionAnalyzer单例"""
    global _position_analyzer
    if _position_analyzer is None:
        from flask import current_app
        stock_db_path = current_app.config['STOCK_DB_PATH']
        _position_analyzer = PositionAnalyzer(stock_db_path)
    return _position_analyzer

def get_portfolio_manager():
    """获取PortfolioManager单例"""
    global _portfolio_manager
    if _portfolio_manager is None:
        from flask import current_app
        _portfolio_manager = PortfolioManager(
            current_app.config['STOCK_DB_PATH'],
            current_app.config['WEBAPP_DB_PATH']
        )
    return _portfolio_manager

portfolio_bp = Blueprint('portfolio', __name__)


def get_db_manager():
    """获取数据库管理器实例"""
    return DatabaseManager(
        current_app.config['STOCK_DB_PATH'],
        current_app.config['WEBAPP_DB_PATH']
    )


# ==================== 持仓管理 API ====================

@portfolio_bp.route('/positions', methods=['GET'])
def get_positions():
    """
    获取所有当前持仓

    Query Parameters:
        status: holding|closed (默认 holding)
        refresh_prices: true|false (是否刷新价格)
        group_id: 分组ID (可选)

    Returns:
        {
            "success": True,
            "positions": [...],
            "summary": {
                "total_market_value": ...,
                "total_profit_loss": ...,
                "total_profit_loss_pct": ...
            }
        }
    """
    try:
        db = get_db_manager()
        status = request.args.get('status', 'holding')
        refresh = request.args.get('refresh_prices', 'false').lower() == 'true'
        group_id = request.args.get('group_id', type=int)

        # 是否刷新价格
        if refresh:
            db.update_position_prices()

        positions = db.get_all_positions(status, group_id=group_id)

        # 计算汇总
        total_mv = sum(p['market_value'] or 0 for p in positions)
        total_cost = sum((p['quantity'] * p['avg_cost']) for p in positions)
        total_pl = total_mv - total_cost
        total_pl_pct = (total_pl / total_cost * 100) if total_cost > 0 else 0

        return jsonify({
            'success': True,
            'positions': positions,
            'summary': {
                'total_market_value': round(total_mv, 2),
                'total_cost': round(total_cost, 2),
                'total_profit_loss': round(total_pl, 2),
                'total_profit_loss_pct': round(total_pl_pct, 2),
                'position_count': len(positions)
            }
        })

    except Exception as e:
        logger.error(f'获取持仓失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@portfolio_bp.route('/positions', methods=['POST'])
def add_position():
    """
    添加新持仓

    Request Body:
        {
            "code": "000001",
            "quantity": 1000,
            "avg_cost": 10.5,
            "notes": "测试买入"
        }

    Returns:
        {
            "success": True,
            "position_id": 1
        }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': '缺少请求数据'}), 400

        required_fields = ['code', 'quantity', 'avg_cost']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'缺少必填字段: {field}'}), 400

        db = get_db_manager()

        # 检查是否已有该持仓
        existing = db.get_position_by_code(data['code'])
        if existing:
            return jsonify({
                'success': False,
                'error': f'股票 {data["code"]} 已有持仓，请使用加仓功能'
            }), 400

        # 获取股票名称和当前价格
        name = db.get_stock_name(data['code'])
        current_price = db.get_stock_latest_price(data['code'])

        data['name'] = name or data.get('name', '')
        data['current_price'] = current_price or data['avg_cost']
        data['first_buy_date'] = data.get('first_buy_date', datetime.now().strftime('%Y-%m-%d'))

        # 质量门控检查 (非force模式)
        if not data.get('force'):
            try:
                manager = get_portfolio_manager()
                positions = db.get_all_positions()
                total_capital = float(db.get_portfolio_setting('total_capital') or 0)
                validation = manager.validate_new_position(
                    data['code'], data['quantity'], data['avg_cost'],
                    total_capital, positions)

                if validation.get('blocks'):
                    return jsonify({
                        'success': False,
                        'error': '质量门控未通过',
                        'validation': validation,
                    }), 400
            except Exception as e:
                logger.warning(f'质量门控检查异常(不阻止建仓): {e}')
                validation = {}
        else:
            validation = {}

        position_id = db.add_position(data)

        # 自动计算并设置SL/TP
        try:
            manager = get_portfolio_manager()
            risk = manager.compute_initial_risk_levels(
                data['code'], data['avg_cost'], data['current_price'])
            ml_score = validation.get('ml_score')
            db.update_position_risk(
                position_id,
                stop_loss_price=risk['stop_loss'],
                take_profit_price=risk['take_profit'],
                trailing_stop_price=risk['trailing_stop'],
                ml_score_at_entry=ml_score,
                last_risk_update=datetime.now().isoformat(),
            )
        except Exception as e:
            logger.warning(f'自动SL/TP设置异常: {e}')

        # 记录交易
        trade_data = {
            'trade_date': datetime.now().strftime('%Y-%m-%d'),
            'code': data['code'],
            'name': data['name'],
            'action': 'buy',
            'quantity': data['quantity'],
            'price': data['avg_cost'],
            'reason': data.get('notes', '建仓买入'),
            'signal_source': 'manual'
        }
        db.add_trade(trade_data)

        msg = f'成功添加持仓: {data["code"]} {name}'
        warnings = validation.get('warnings', [])
        if warnings:
            msg += f' (警告: {"; ".join(warnings)})'

        return jsonify({
            'success': True,
            'position_id': position_id,
            'message': msg,
            'validation': validation,
        })

    except Exception as e:
        logger.error(f'添加持仓失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@portfolio_bp.route('/positions/<int:position_id>', methods=['PUT'])
def update_position(position_id: int):
    """
    更新持仓

    Path Parameters:
        position_id: 持仓ID

    Request Body:
        {
            "quantity": 2000,
            "avg_cost": 10.8,
            "notes": "更新备注"
        }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': '缺少请求数据'}), 400

        db = get_db_manager()
        success = db.update_position(position_id, data)

        if success:
            return jsonify({'success': True, 'message': '持仓已更新'})
        else:
            return jsonify({'success': False, 'error': '更新失败，持仓不存在或无变更'}), 404

    except Exception as e:
        logger.error(f'更新持仓失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@portfolio_bp.route('/positions/<int:position_id>', methods=['DELETE'])
def delete_position(position_id: int):
    """删除持仓"""
    try:
        db = get_db_manager()
        success = db.delete_position(position_id)

        if success:
            return jsonify({'success': True, 'message': '持仓已删除'})
        else:
            return jsonify({'success': False, 'error': '删除失败，持仓不存在'}), 404

    except Exception as e:
        logger.error(f'删除持仓失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@portfolio_bp.route('/positions/<int:position_id>/add', methods=['POST'])
def add_to_position(position_id: int):
    """
    加仓

    Request Body:
        {
            "quantity": 500,
            "price": 11.0,
            "reason": "技术信号加仓"
        }
    """
    try:
        data = request.get_json()
        if not data or 'quantity' not in data or 'price' not in data:
            return jsonify({'success': False, 'error': '缺少quantity或price'}), 400

        db = get_db_manager()
        positions = db.get_all_positions()
        position = next((p for p in positions if p['id'] == position_id), None)

        if not position:
            return jsonify({'success': False, 'error': '持仓不存在'}), 404

        # 计算新的平均成本
        old_quantity = position['quantity']
        old_cost = position['avg_cost']
        add_quantity = data['quantity']
        add_price = data['price']

        new_quantity = old_quantity + add_quantity
        new_avg_cost = (old_quantity * old_cost + add_quantity * add_price) / new_quantity

        # 仓位比例检查 (加仓)
        warnings = []
        total_capital = float(db.get_portfolio_setting('total_capital') or 0)
        if total_capital > 0:
            new_value = new_quantity * add_price
            weight = new_value / total_capital
            if weight > 0.10 and not data.get('force'):
                return jsonify({
                    'success': False,
                    'error': f'加仓后仓位占比 {weight:.1%} 超过10%上限'
                }), 400
            elif weight > 0.08:
                warnings.append(f'加仓后仓位占比 {weight:.1%} 接近10%上限')

        # 更新持仓
        db.update_position(position_id, {
            'quantity': new_quantity,
            'avg_cost': new_avg_cost
        })

        # 重新计算SL/TP
        try:
            manager = get_portfolio_manager()
            current_price = db.get_stock_latest_price(
                position['code'].split('.')[0] if '.' in position['code'] else position['code']
            ) or add_price
            risk = manager.compute_initial_risk_levels(
                position['code'], new_avg_cost, current_price)
            db.update_position_risk(
                position_id,
                stop_loss_price=risk['stop_loss'],
                take_profit_price=risk['take_profit'],
                trailing_stop_price=risk['trailing_stop'],
                last_risk_update=datetime.now().isoformat(),
            )
        except Exception as e:
            logger.warning(f'加仓后SL/TP更新异常: {e}')

        # 记录交易
        trade_data = {
            'trade_date': datetime.now().strftime('%Y-%m-%d'),
            'code': position['code'],
            'name': position['name'],
            'action': 'add',
            'quantity': add_quantity,
            'price': add_price,
            'reason': data.get('reason', '加仓'),
            'signal_source': data.get('signal_source', 'manual')
        }
        db.add_trade(trade_data)

        msg = f'成功加仓 {add_quantity} 股，新均价 {new_avg_cost:.2f}'
        if warnings:
            msg += f' (警告: {"; ".join(warnings)})'

        return jsonify({
            'success': True,
            'message': msg,
            'warnings': warnings,
        })

    except Exception as e:
        logger.error(f'加仓失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@portfolio_bp.route('/positions/<int:position_id>/reduce', methods=['POST'])
def reduce_position(position_id: int):
    """
    减仓

    Request Body:
        {
            "quantity": 500,
            "price": 12.0,
            "reason": "止盈减仓"
        }
    """
    try:
        data = request.get_json()
        if not data or 'quantity' not in data or 'price' not in data:
            return jsonify({'success': False, 'error': '缺少quantity或price'}), 400

        db = get_db_manager()
        positions = db.get_all_positions()
        position = next((p for p in positions if p['id'] == position_id), None)

        if not position:
            return jsonify({'success': False, 'error': '持仓不存在'}), 404

        reduce_quantity = data['quantity']
        if reduce_quantity > position['quantity']:
            return jsonify({'success': False, 'error': '减仓数量超过持仓数量'}), 400

        new_quantity = position['quantity'] - reduce_quantity

        # 记录交易
        trade_data = {
            'trade_date': datetime.now().strftime('%Y-%m-%d'),
            'code': position['code'],
            'name': position['name'],
            'action': 'reduce' if new_quantity > 0 else 'sell',
            'quantity': reduce_quantity,
            'price': data['price'],
            'reason': data.get('reason', '减仓'),
            'signal_source': data.get('signal_source', 'manual')
        }
        db.add_trade(trade_data)

        if new_quantity == 0:
            # 清仓 - 更新状态
            db.update_position(position_id, {'status': 'closed', 'quantity': 0})
            return jsonify({
                'success': True,
                'message': '已清仓',
                'cleared': True
            })
        else:
            # 减仓
            db.update_position(position_id, {'quantity': new_quantity})
            return jsonify({
                'success': True,
                'message': f'成功减仓 {reduce_quantity} 股，剩余 {new_quantity} 股',
                'cleared': False
            })

    except Exception as e:
        logger.error(f'减仓失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@portfolio_bp.route('/summary', methods=['GET'])
def get_summary():
    """获取持仓汇总"""
    try:
        db = get_db_manager()
        positions = db.get_all_positions()

        total_mv = sum(p['market_value'] or 0 for p in positions)
        total_cost = sum((p['quantity'] * p['avg_cost']) for p in positions)
        total_pl = total_mv - total_cost
        total_pl_pct = (total_pl / total_cost * 100) if total_cost > 0 else 0

        # 盈亏分布
        winning = [p for p in positions if (p['profit_loss_pct'] or 0) > 0]
        losing = [p for p in positions if (p['profit_loss_pct'] or 0) < 0]

        return jsonify({
            'success': True,
            'summary': {
                'total_market_value': round(total_mv, 2),
                'total_cost': round(total_cost, 2),
                'total_profit_loss': round(total_pl, 2),
                'total_profit_loss_pct': round(total_pl_pct, 2),
                'position_count': len(positions),
                'winning_count': len(winning),
                'losing_count': len(losing),
                'win_rate': round(len(winning) / len(positions) * 100, 1) if positions else 0
            }
        })

    except Exception as e:
        logger.error(f'获取汇总失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== 交易记录 API ====================

@portfolio_bp.route('/trades', methods=['GET'])
def get_trades():
    """
    获取交易记录

    Query Parameters:
        code: 股票代码 (可选)
        limit: 限制数量 (默认100)

    Returns:
        {
            "success": True,
            "trades": [...]
        }
    """
    try:
        db = get_db_manager()
        code = request.args.get('code')
        limit = int(request.args.get('limit', 100))

        trades = db.get_all_trades(code=code, limit=limit)

        return jsonify({
            'success': True,
            'trades': trades,
            'total': len(trades)
        })

    except Exception as e:
        logger.error(f'获取交易记录失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@portfolio_bp.route('/trades', methods=['POST'])
def add_trade():
    """
    添加交易记录

    Request Body:
        {
            "trade_date": "2025-01-15",
            "code": "000001",
            "action": "buy|sell|add|reduce",
            "quantity": 1000,
            "price": 10.5,
            "reason": "买入原因"
        }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': '缺少请求数据'}), 400

        required_fields = ['trade_date', 'code', 'action', 'quantity', 'price']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'缺少必填字段: {field}'}), 400

        db = get_db_manager()

        # 获取股票名称
        if not data.get('name'):
            data['name'] = db.get_stock_name(data['code'])

        trade_id = db.add_trade(data)

        return jsonify({
            'success': True,
            'trade_id': trade_id,
            'message': '交易记录已添加'
        })

    except Exception as e:
        logger.error(f'添加交易记录失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@portfolio_bp.route('/trades/<int:trade_id>', methods=['PUT'])
def update_trade(trade_id: int):
    """更新交易记录"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': '缺少请求数据'}), 400

        db = get_db_manager()
        success = db.update_trade(trade_id, data)

        if success:
            return jsonify({'success': True, 'message': '交易记录已更新'})
        else:
            return jsonify({'success': False, 'error': '更新失败'}), 404

    except Exception as e:
        logger.error(f'更新交易记录失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@portfolio_bp.route('/trades/<int:trade_id>', methods=['DELETE'])
def delete_trade(trade_id: int):
    """删除交易记录"""
    try:
        db = get_db_manager()
        success = db.delete_trade(trade_id)

        if success:
            return jsonify({'success': True, 'message': '交易记录已删除'})
        else:
            return jsonify({'success': False, 'error': '删除失败'}), 404

    except Exception as e:
        logger.error(f'删除交易记录失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== 操作建议 API ====================

@portfolio_bp.route('/recommendations', methods=['GET'])
def get_recommendations():
    """
    获取操作建议

    Query Parameters:
        date: 日期 (可选，默认最新)

    Returns:
        {
            "success": True,
            "recommendations": [...],
            "date": "2025-01-15"
        }
    """
    try:
        db = get_db_manager()
        date = request.args.get('date')

        recommendations = db.get_recommendations(date)

        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'date': date or (recommendations[0]['date'] if recommendations else None)
        })

    except Exception as e:
        logger.error(f'获取建议失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@portfolio_bp.route('/recommendations/generate', methods=['POST'])
def generate_recommendations():
    """
    生成操作建议

    基于ML评分系统和多维度分析生成专业操作建议
    优先使用V3.9.4 (IC=0.1363)，回退到V3.9.0 (IC=0.0489)

    分析维度:
    1. ML评分 (V3.9.4: 48特征, V3.9.0: 42特征)
    2. 技术面分析 (趋势、RSI、MACD、KDJ)
    3. 风险指标 (ATR止损、波动率、Kelly仓位)
    4. 基本面数据 (PE、PB、换手率)
    """
    try:
        db = get_db_manager()
        positions = db.get_all_positions()

        if not positions:
            return jsonify({
                'success': True,
                'recommendations': [],
                'message': '当前无持仓'
            })

        # 使用专业分析器
        analyzer = get_position_analyzer()
        today = datetime.now().strftime('%Y-%m-%d')

        # 批量分析所有持仓 (含组合风控)
        portfolio_analysis = analyzer.analyze_portfolio(positions)
        analysis_results = portfolio_analysis['positions']
        portfolio_risk = portfolio_analysis['portfolio_risk']

        recommendations = []
        for result in analysis_results:
            # 获取最新价格并更新持仓
            code = result['code']
            current_price = db.get_stock_latest_price(code.split('.')[0] if '.' in code else code)
            if current_price:
                pos_id = result.get('position_id')
                if pos_id:
                    db.update_position(pos_id, {'current_price': current_price})

            # 构建建议数据
            rec_data = {
                'date': today,
                'code': result['code'],
                'name': result.get('name', ''),
                'action': result.get('action', 'hold'),
                'action_cn': result.get('action_cn', '持有'),
                'action_reason': result.get('action_reason', ''),
                'action_color': result.get('action_color', 'secondary'),
                'reduce_pct': result.get('reduce_pct'),
                'add_pct': result.get('add_pct'),
                'urgency': result.get('urgency', 'normal'),
                'reason': result.get('reason', ''),
                'confidence': result.get('confidence', 0.5),
                'stop_loss_price': result.get('stop_loss_price'),
                'take_profit_price': result.get('take_profit_price'),
                # 绝对ML评分
                'ml_score': result.get('ml_score'),
                'score_source': result.get('score_source', 'ml'),
                'ml_recommendation': result.get('ml_recommendation'),
                'predicted_return_5d': result.get('predicted_return_5d'),
                # V4.4.1 多目标预测
                'pred_3d': result.get('pred_3d'),
                'pred_5d': result.get('pred_5d'),
                'pred_10d': result.get('pred_10d'),
                'pred_15d': result.get('pred_15d'),
                'pred_targets': result.get('pred_targets', {}),
                'exec_filter': result.get('exec_filter'),
                'regime_info': result.get('regime_info'),
                # 风险评分 (新)
                'risk_score': result.get('risk_score'),
                'risk_level': result.get('risk_level'),
                'risk_level_text': result.get('risk_level_text'),
                'risk_breakdown': result.get('risk_breakdown', {}),
                # 技术面
                'trend': result.get('trend'),
                'trend_strength': result.get('trend_strength'),
                'rsi': result.get('rsi'),
                'macd_signal': result.get('macd_signal'),
                'kdj_signal': result.get('kdj_signal'),
                # 风险指标
                'atr': result.get('atr'),
                'atr_pct': result.get('atr_pct'),
                'volatility_20d': result.get('volatility_20d'),
                'dynamic_stop_loss': result.get('dynamic_stop_loss'),
                'hard_stop_loss': result.get('hard_stop_loss'),
                'kelly_position': result.get('kelly_position'),
                # 其他
                'total_score': result.get('total_score'),
                'component_scores': result.get('component_scores'),
                'target_price': result.get('target_price'),
                'support': result.get('support'),
                'resistance': result.get('resistance'),
                # 盈亏
                'profit_loss_pct': result.get('profit_loss_pct'),
                'market_value': result.get('market_value'),
                'quantity': result.get('quantity'),
                'avg_cost': result.get('avg_cost'),
                'current_price': result.get('current_price'),
            }

            # 保存到数据库 (包含关键字段)
            db_rec = {
                'date': today,
                'code': result['code'],
                'name': result.get('name', ''),
                'action': result.get('action', 'hold'),
                'urgency': result.get('urgency', 'normal'),
                'reason': result.get('action_reason', ''),
                'ml_score': result.get('ml_score'),
                'stop_loss_price': result.get('stop_loss_price'),
                'take_profit_price': result.get('take_profit_price'),
                'confidence': result.get('confidence', 0.5),
                'kelly_position': result.get('kelly_position'),
                'predicted_return_5d': result.get('predicted_return_5d')
            }
            db.add_recommendation(db_rec)

            recommendations.append(rec_data)

        # 按风险评分排序 (高风险优先)
        recommendations.sort(key=lambda x: -(x.get('risk_score') or 0))

        # 顺带更新追踪止损 (保持SL/TP同步最新)
        try:
            manager = get_portfolio_manager()
            manager.update_trailing_stops(positions)
        except Exception as e:
            logger.warning(f'追踪止损更新失败: {e}')

        # 获取ML版本信息
        ml_version = analyzer.ml_version or 'N/A'
        ml_info = {
            'v4.4.1': 'S级, 59特征, 6增强模块',
            'v3.9.4': 'IC=0.1363, 48特征',
            'v3.9.0': 'IC=0.0489, 42特征'
        }.get(ml_version, '')

        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'portfolio_risk': portfolio_risk,
            'date': today,
            'message': f'已使用{ml_version.upper()} ML系统为{len(recommendations)}只持仓生成专业建议',
            'ml_version': ml_version,
            'ml_info': ml_info,
            'analysis_summary': {
                'total_positions': len(recommendations),
                'critical_actions': sum(1 for r in recommendations if r.get('action') in ('sell', 'stop_loss')),
                'high_actions': sum(1 for r in recommendations if r.get('action') == 'reduce'),
                'ml_scored': sum(1 for r in recommendations if r.get('ml_score') is not None),
                'high_risk_count': portfolio_risk.get('high_risk_count', 0)
            }
        })

    except Exception as e:
        logger.error(f'生成建议失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@portfolio_bp.route('/recommendations/<int:rec_id>/execute', methods=['POST'])
def execute_recommendation(rec_id: int):
    """标记建议已执行"""
    try:
        db = get_db_manager()
        success = db.mark_recommendation_executed(rec_id)

        if success:
            return jsonify({'success': True, 'message': '已标记为已执行'})
        else:
            return jsonify({'success': False, 'error': '建议不存在'}), 404

    except Exception as e:
        logger.error(f'标记执行失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== 交易评估 API ====================

@portfolio_bp.route('/evaluations', methods=['GET'])
def get_evaluations():
    """
    获取交易评估

    Query Parameters:
        trade_id: 交易ID (可选)

    Returns:
        {
            "success": True,
            "evaluations": [...],
            "stats": {...}
        }
    """
    try:
        db = get_db_manager()
        trade_id = request.args.get('trade_id', type=int)

        evaluations = db.get_trade_evaluations(trade_id)
        stats = db.get_evaluation_stats()

        return jsonify({
            'success': True,
            'evaluations': evaluations,
            'stats': stats
        })

    except Exception as e:
        logger.error(f'获取评估失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@portfolio_bp.route('/evaluations/run', methods=['POST'])
def run_evaluations():
    """
    执行交易评估

    对所有未评估的交易进行评估，使用交易日计数（非日历日）

    Request Body:
        {
            "days_after": 5  # 评估交易后N个交易日的表现
        }
    """
    try:
        data = request.get_json() or {}
        days_after = data.get('days_after', 5)

        db = get_db_manager()
        trades = db.get_all_trades(limit=500)

        evaluated_count = 0
        too_recent_count = 0
        already_evaluated_count = 0
        no_price_data_count = 0
        earliest_evaluable_date = None
        today = datetime.now()
        today_str = today.strftime('%Y-%m-%d')

        for trade in trades:
            # 检查是否已评估
            existing = db.get_trade_evaluations(trade['id'])
            if any(e['days_after'] == days_after for e in existing):
                already_evaluated_count += 1
                continue

            # 获取交易日之后的价格数据（使用trade_date的下一天作为起点，确保只取交易后的数据）
            trade_date = datetime.strptime(trade['trade_date'], '%Y-%m-%d')
            next_day = trade_date + timedelta(days=1)
            end_date = trade_date + timedelta(days=days_after * 2 + 10)
            price_history = db.get_stock_price_history(
                trade['code'],
                next_day.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )

            if not price_history:
                # 区分：完全无价格数据 vs 交易太新
                # 检查该股票是否存在于行情库中
                any_price = db.get_stock_latest_price(trade['code'])
                if any_price is None:
                    no_price_data_count += 1
                else:
                    too_recent_count += 1
                    # 至少需要1个交易日后的数据
                    est_date = today + timedelta(days=2)
                    est_str = est_date.strftime('%Y-%m-%d')
                    if earliest_evaluable_date is None or est_str < earliest_evaluable_date:
                        earliest_evaluable_date = est_str
                continue

            # price_history 现在只包含交易日之后的数据
            # price_history[0] = 第1个交易日后, price_history[N-1] = 第N个交易日后
            if len(price_history) < days_after:
                too_recent_count += 1
                # 估算最早可评估日期：还差多少个交易日
                remaining = days_after - len(price_history)
                est_date = today + timedelta(days=int(remaining * 1.5) + 1)  # 粗略估算（含周末）
                est_str = est_date.strftime('%Y-%m-%d')
                if earliest_evaluable_date is None or est_str < earliest_evaluable_date:
                    earliest_evaluable_date = est_str
                continue

            # 计算评估：第N个交易日后的收盘价
            trade_price = trade['price']
            price_after = price_history[days_after - 1]['close']

            if trade['action'] in ['buy', 'add']:
                return_pct = (price_after - trade_price) / trade_price * 100
            else:  # sell, reduce
                return_pct = (trade_price - price_after) / trade_price * 100

            # 计算最大盈亏（从第1个交易日到第N个交易日）
            prices = [p['close'] for p in price_history[:days_after]]
            if trade['action'] in ['buy', 'add']:
                max_profit = (max(prices) - trade_price) / trade_price * 100
                max_loss = (min(prices) - trade_price) / trade_price * 100
            else:
                max_profit = (trade_price - min(prices)) / trade_price * 100
                max_loss = (trade_price - max(prices)) / trade_price * 100

            # 评分和等级
            if return_pct >= 10:
                grade, score = 'A', 95
                comments = f'优秀操作，{days_after}个交易日后收益超过10%'
            elif return_pct >= 5:
                grade, score = 'B', 80
                comments = f'良好操作，{days_after}个交易日后收益5-10%'
            elif return_pct >= 0:
                grade, score = 'C', 65
                comments = f'一般操作，{days_after}个交易日后小幅盈利'
            elif return_pct >= -5:
                grade, score = 'D', 45
                comments = f'较差操作，{days_after}个交易日后小幅亏损'
            else:
                grade, score = 'F', 25
                comments = f'失败操作，{days_after}个交易日后亏损超过5%'

            # 保存评估
            eval_data = {
                'trade_id': trade['id'],
                'eval_date': today_str,
                'days_after': days_after,
                'price_after': price_after,
                'return_pct': round(return_pct, 2),
                'score': score,
                'grade': grade,
                'comments': comments,
                'max_profit_pct': round(max_profit, 2),
                'max_loss_pct': round(max_loss, 2)
            }
            db.add_trade_evaluation(eval_data)
            evaluated_count += 1

        stats = db.get_evaluation_stats()

        return jsonify({
            'success': True,
            'evaluated_count': evaluated_count,
            'stats': stats,
            'message': f'已评估{evaluated_count}笔交易',
            'diagnostics': {
                'total_trades': len(trades),
                'evaluated': evaluated_count,
                'too_recent': too_recent_count,
                'already_evaluated': already_evaluated_count,
                'no_price_data': no_price_data_count,
                'earliest_evaluable_date': earliest_evaluable_date,
                'days_after': days_after
            }
        })

    except Exception as e:
        logger.error(f'执行评估失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@portfolio_bp.route('/evaluations/stats', methods=['GET'])
def get_evaluation_stats():
    """获取评估统计"""
    try:
        db = get_db_manager()
        stats = db.get_evaluation_stats()

        return jsonify({
            'success': True,
            'stats': stats
        })

    except Exception as e:
        logger.error(f'获取统计失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== 持仓快照 API ====================

@portfolio_bp.route('/snapshots', methods=['GET'])
def get_snapshots():
    """获取持仓快照历史"""
    try:
        db = get_db_manager()
        days = int(request.args.get('days', 30))
        snapshots = db.get_position_snapshots(days)

        return jsonify({
            'success': True,
            'snapshots': snapshots
        })

    except Exception as e:
        logger.error(f'获取快照失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@portfolio_bp.route('/snapshots', methods=['POST'])
def save_snapshot():
    """保存当日持仓快照"""
    try:
        db = get_db_manager()
        snapshot_id = db.save_position_snapshot()

        if snapshot_id:
            return jsonify({
                'success': True,
                'snapshot_id': snapshot_id,
                'message': '快照已保存'
            })
        else:
            return jsonify({
                'success': True,
                'message': '无持仓，未保存快照'
            })

    except Exception as e:
        logger.error(f'保存快照失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== 股票搜索 API ====================

@portfolio_bp.route('/stocks/search', methods=['GET'])
def search_stocks():
    """
    搜索股票

    Query Parameters:
        q: 搜索关键词（代码或名称）
        limit: 限制数量 (默认10)
    """
    try:
        query = request.args.get('q', '')
        limit = int(request.args.get('limit', 10))

        if len(query) < 1:
            return jsonify({'success': True, 'stocks': []})

        db = get_db_manager()

        with db.get_stock_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT code, name, type
                FROM securities
                WHERE (code LIKE ? OR name LIKE ?)
                AND type = 'A股'
                LIMIT ?
            ''', (f'%{query}%', f'%{query}%', limit))

            stocks = [dict(row) for row in cursor.fetchall()]

        return jsonify({
            'success': True,
            'stocks': stocks
        })

    except Exception as e:
        logger.error(f'搜索股票失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@portfolio_bp.route('/stocks/<code>/price', methods=['GET'])
def get_stock_price(code: str):
    """获取股票当前价格"""
    try:
        db = get_db_manager()
        price = db.get_stock_latest_price(code)
        name = db.get_stock_name(code)

        if price:
            return jsonify({
                'success': True,
                'code': code,
                'name': name,
                'price': price
            })
        else:
            return jsonify({
                'success': False,
                'error': f'未找到股票 {code} 的价格数据'
            }), 404

    except Exception as e:
        logger.error(f'获取价格失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== CSV导入 API ====================

@portfolio_bp.route('/import/parse', methods=['POST'])
def parse_import_csv():
    """
    解析上传的CSV文件,返回预览数据

    Request: multipart/form-data with 'file' field
    Returns: {success, positions: [...], warnings: [...]}
    """
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '请选择CSV文件'}), 400

        file = request.files['file']
        if not file.filename:
            return jsonify({'success': False, 'error': '请选择CSV文件'}), 400

        if not file.filename.lower().endswith(('.csv', '.txt', '.xls')):
            return jsonify({'success': False, 'error': '仅支持CSV/TXT格式文件'}), 400

        content = file.read()
        if len(content) > 5 * 1024 * 1024:  # 5MB limit
            return jsonify({'success': False, 'error': '文件过大(最大5MB)'}), 400

        positions, warnings = parse_csv(content)

        return jsonify({
            'success': True,
            'positions': positions,
            'warnings': warnings,
            'count': len(positions)
        })

    except Exception as e:
        logger.error(f'解析CSV失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@portfolio_bp.route('/import/paste', methods=['POST'])
def parse_import_paste():
    """
    解析粘贴的持仓文本 (从券商网页直接复制)

    Request Body: {text: "粘贴的文本内容"}
    Returns: {success, positions: [...], warnings: [...]}
    """
    try:
        data = request.get_json()
        if not data or not data.get('text'):
            return jsonify({'success': False, 'error': '请粘贴持仓文本'}), 400

        text = data['text']
        if len(text) > 100000:
            return jsonify({'success': False, 'error': '文本过长(最大100KB)'}), 400

        positions, warnings = parse_web_paste(text)

        return jsonify({
            'success': True,
            'positions': positions,
            'warnings': warnings,
            'count': len(positions)
        })

    except Exception as e:
        logger.error(f'解析粘贴文本失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@portfolio_bp.route('/import/confirm', methods=['POST'])
def confirm_import():
    """
    确认导入解析好的持仓

    Request Body: {positions: [{code, name, quantity, avg_cost, current_price}, ...]}
    Returns: {success, added, updated, skipped, details: [...]}
    """
    try:
        data = request.get_json()
        if not data or not data.get('positions'):
            return jsonify({'success': False, 'error': '无导入数据'}), 400

        db = get_db_manager()
        existing = db.get_all_positions()

        result = merge_positions(data['positions'], existing, db)

        return jsonify({
            'success': True,
            **result,
            'message': f'导入完成: 新增{result["added"]}只, 更新{result["updated"]}只, 跳过{result["skipped"]}只'
        })

    except Exception as e:
        logger.error(f'导入持仓失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== 成交记录导入 API ====================

@portfolio_bp.route('/trades/import/parse', methods=['POST'])
def parse_trade_import():
    """
    解析粘贴的当日成交HTML文本

    Request Body: {text: "HTML内容"}
    Returns: {success, trades: [...], warnings: [...], count: N}
    """
    try:
        data = request.get_json()
        if not data or not data.get('text'):
            return jsonify({'success': False, 'error': '请粘贴成交记录HTML'}), 400

        text = data['text']
        if len(text) > 200000:
            return jsonify({'success': False, 'error': '文本过长(最大200KB)'}), 400

        trades, warnings = parse_trade_html_table(text)

        return jsonify({
            'success': True,
            'trades': trades,
            'warnings': warnings,
            'count': len(trades)
        })

    except Exception as e:
        logger.error(f'解析成交记录失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@portfolio_bp.route('/trades/import/confirm', methods=['POST'])
def confirm_trade_import():
    """
    确认导入成交记录

    Request Body: {trade_date: "2026-02-26", trades: [{code, name, action, quantity, price, ...}, ...]}
    Returns: {success, added, skipped, details: [...]}
    """
    try:
        data = request.get_json()
        if not data or not data.get('trades'):
            return jsonify({'success': False, 'error': '无导入数据'}), 400
        if not data.get('trade_date'):
            return jsonify({'success': False, 'error': '请选择交易日期'}), 400

        db = get_db_manager()
        result = merge_trades(data['trades'], data['trade_date'], db)

        return jsonify({
            'success': True,
            **result,
            'message': f'导入完成: 新增{result["added"]}笔, 跳过{result["skipped"]}笔(重复)'
        })

    except Exception as e:
        logger.error(f'导入成交记录失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== 持仓分组 API ====================

@portfolio_bp.route('/groups', methods=['GET'])
def get_groups():
    """获取所有分组"""
    try:
        db = get_db_manager()
        groups = db.get_all_groups()

        # 为每个分组计算持仓数量和市值
        for group in groups:
            positions = db.get_all_positions(group_id=group['id'])
            group['position_count'] = len(positions)
            group['total_market_value'] = sum(p['market_value'] or 0 for p in positions)

        return jsonify({
            'success': True,
            'groups': groups
        })

    except Exception as e:
        logger.error(f'获取分组失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@portfolio_bp.route('/groups', methods=['POST'])
def add_group():
    """添加分组"""
    try:
        data = request.get_json()
        if not data or not data.get('name'):
            return jsonify({'success': False, 'error': '请输入分组名称'}), 400

        db = get_db_manager()
        group_id = db.add_group(data)

        return jsonify({
            'success': True,
            'group_id': group_id,
            'message': f'分组 "{data["name"]}" 已创建'
        })

    except Exception as e:
        if 'UNIQUE constraint' in str(e):
            return jsonify({'success': False, 'error': '分组名称已存在'}), 400
        logger.error(f'添加分组失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@portfolio_bp.route('/groups/<int:group_id>', methods=['PUT'])
def update_group(group_id: int):
    """更新分组"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': '缺少请求数据'}), 400

        db = get_db_manager()
        success = db.update_group(group_id, data)

        if success:
            return jsonify({'success': True, 'message': '分组已更新'})
        else:
            return jsonify({'success': False, 'error': '分组不存在或无变更'}), 404

    except Exception as e:
        logger.error(f'更新分组失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@portfolio_bp.route('/groups/<int:group_id>', methods=['DELETE'])
def delete_group(group_id: int):
    """删除分组"""
    try:
        db = get_db_manager()
        success = db.delete_group(group_id)

        if success:
            return jsonify({'success': True, 'message': '分组已删除'})
        else:
            return jsonify({'success': False, 'error': '分组不存在'}), 404

    except Exception as e:
        logger.error(f'删除分组失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@portfolio_bp.route('/positions/<int:position_id>/group', methods=['PUT'])
def set_position_group(position_id: int):
    """设置持仓分组"""
    try:
        data = request.get_json()
        group_id = data.get('group_id')  # None 表示移到未分组

        db = get_db_manager()
        success = db.set_position_group(position_id, group_id)

        if success:
            return jsonify({'success': True, 'message': '分组已更新'})
        else:
            return jsonify({'success': False, 'error': '持仓不存在'}), 404

    except Exception as e:
        logger.error(f'设置分组失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== 开仓建议 API (选股报告) ====================

@portfolio_bp.route('/selections/today', methods=['GET'])
def get_today_selections():
    """
    获取最新选股报告的开仓建议

    Query Parameters:
        top_n: 返回前N只 (默认15)
        min_score: 最低评分 (默认60)

    Returns:
        {success, selections: [...], report_date, held_codes: [...]}
    """
    try:
        import glob as glob_mod
        top_n = int(request.args.get('top_n', 15))
        min_score = float(request.args.get('min_score', 60))
        version = request.args.get('version')

        # 未指定版本时，从用户设置读取
        if not version:
            db = get_db_manager()
            version = db.get_portfolio_setting('selection_version', 'v3.9')

        # 用DAILY_SELECTION_DIRS查找目录，fallback到v3.9
        reports_dir = current_app.config.get('REPORTS_DIR',
                        Path(current_app.config['STOCK_DB_PATH']).parent.parent / 'reports')
        selection_dirs = current_app.config.get('DAILY_SELECTION_DIRS', {})
        if version in selection_dirs:
            report_dirs = [selection_dirs[version]]
        else:
            version = 'v3.9'
            report_dirs = [reports_dir / 'daily_selection_v3.9']

        latest_file = None
        latest_date = ''
        for rd in report_dirs:
            for f in sorted(rd.glob('analysis_data_*.json'), reverse=True):
                date_str = f.stem.replace('analysis_data_', '')
                if date_str > latest_date:
                    latest_date = date_str
                    latest_file = f

        if not latest_file:
            return jsonify({'success': True, 'selections': [],
                           'message': '未找到选股报告', 'version': version})

        with open(latest_file) as f:
            report_data = json.load(f)

        all_stocks = report_data.get('all_stocks_with_scores', [])

        # 获取当前持仓代码，排除已持有
        db = get_db_manager()
        positions = db.get_all_positions()
        held_codes = set(p['code'] for p in positions)

        # 过滤: 排除已持仓 + 最低评分 + 推荐为买入
        selections = []
        for s in all_stocks:
            code = s.get('stock_code', '')
            if code in held_codes:
                continue
            score = s.get('score', 0) or 0
            if score < min_score:
                continue
            rec = s.get('recommendation', '')
            if '卖出' in rec or '观望' in rec:
                continue

            selections.append({
                'code': code,
                'name': s.get('stock_name', ''),
                'score': round(score, 1),
                'strategies': s.get('strategies', []),
                'strategy_count': s.get('selected_by_strategies', 1),
                'recommendation': rec,
                'predicted_return_5d': s.get('predicted_return_5d'),
                'confidence': s.get('confidence_score'),
                'risk_level': s.get('risk_level', 'medium'),
                'close_price': s.get('close_price'),
                'buy_price': s.get('suggested_buy_price'),
                'stop_loss': s.get('stop_loss_price'),
                'take_profit': s.get('take_profit_price'),
                'industry': s.get('industry', ''),
            })

        # 按评分排序，取top_n
        selections.sort(key=lambda x: -x['score'])
        selections = selections[:top_n]

        return jsonify({
            'success': True,
            'selections': selections,
            'report_date': latest_date,
            'total_in_report': len(all_stocks),
            'held_codes': list(held_codes),
            'version': version,
        })

    except Exception as e:
        logger.error(f'获取选股建议失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@portfolio_bp.route('/settings/selection-version', methods=['PUT'])
def set_selection_version():
    """持久化用户选择的选股报告版本"""
    try:
        data = request.get_json()
        version = data.get('version', 'v3.9')

        selection_dirs = current_app.config.get('DAILY_SELECTION_DIRS', {})
        if version not in selection_dirs:
            return jsonify({'success': False, 'error': f'未知版本: {version}'}), 400

        db = get_db_manager()
        db.set_portfolio_setting('selection_version', version)
        return jsonify({'success': True, 'version': version})

    except Exception as e:
        logger.error(f'设置选股版本失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== 组合评分 API (Portfolio Pilot Score) ====================

@portfolio_bp.route('/score', methods=['POST'])
def calculate_portfolio_score():
    """
    计算 Portfolio Pilot Score (仓位领航评分)

    4层/20指标/100分评分体系:
    - L1 持仓质量 (5指标/25分)
    - L2 风险控制 (5指标/25分)
    - L3 组合效率 (5指标/25分)
    - L4 执行纪律 (5指标/25分)

    Request Body (optional):
        {
            "total_capital": 500000,  // 总资金(含现金)
            "cash_amount": 50000,     // 现金金额
            "save": true              // 是否保存到历史
        }

    Returns:
        完整评分报告
    """
    try:
        data = request.get_json() or {}
        total_capital = data.get('total_capital', 0)
        cash_amount = data.get('cash_amount', 0)
        save_score = data.get('save', True)

        db = get_db_manager()

        # Auto-load persisted capital settings if not provided in request
        if total_capital <= 0:
            saved_capital = db.get_portfolio_setting('total_capital')
            if saved_capital:
                total_capital = float(saved_capital)
        if cash_amount <= 0:
            saved_cash = db.get_portfolio_setting('cash_amount')
            if saved_cash:
                cash_amount = float(saved_cash)

        positions = db.get_all_positions()

        if not positions:
            return jsonify({
                'success': True,
                'score': None,
                'message': '当前无持仓，无法评分'
            })

        # 获取ML分析结果
        analyzer = get_position_analyzer()
        portfolio_analysis = analyzer.analyze_portfolio(positions)

        # 获取交易记录和建议
        trades = db.get_all_trades(limit=200)
        recommendations = db.get_recommendations()
        snapshots = db.get_position_snapshots(days=60)

        # 计算评分
        scorer = PortfolioScorer(
            current_app.config['STOCK_DB_PATH'],
            current_app.config['WEBAPP_DB_PATH']
        )

        # Auto-match trades + positions to recommendations before scoring
        recommendations = scorer.auto_match_recommendations(trades, recommendations, positions)

        score_result = scorer.calculate_score(
            positions=positions,
            trades=trades,
            recommendations=recommendations,
            snapshots=snapshots,
            portfolio_analysis=portfolio_analysis,
            total_capital=total_capital,
            cash_amount=cash_amount,
        )

        # 保存到历史
        if save_score:
            db.save_portfolio_score(score_result)

        # 序列化层级数据 (将int keys转为str for JSON)
        layers_serialized = {}
        for layer_num, layer_data in score_result['layers'].items():
            layers_serialized[str(layer_num)] = {
                'name': layer_data['name'],
                'score': layer_data['score'],
                'max_score': layer_data['max_score'],
                'pct': layer_data['pct'],
                'metrics': {k: {
                    'value': round(v['value'], 4) if isinstance(v['value'], float) else v['value'],
                    'score': v['score'],
                    'max_score': v['max_score'],
                    'name': v['name'],
                    'unit': v['unit'],
                    'description': v['description'],
                    'direction': v['direction'],
                } for k, v in layer_data['metrics'].items()}
            }

        return jsonify({
            'success': True,
            'score': {
                'total_score': score_result['total_score'],
                'total_max': score_result['total_max'],
                'total_pct': score_result['total_pct'],
                'grade': score_result['grade'],
                'grade_label': score_result['grade_label'],
                'layers': layers_serialized,
                'improvements': score_result['improvements'],
                'position_count': score_result['position_count'],
                'total_market_value': score_result['total_market_value'],
                'timestamp': score_result['timestamp'],
            }
        })

    except Exception as e:
        logger.error(f'计算组合评分失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@portfolio_bp.route('/score/history', methods=['GET'])
def get_score_history():
    """
    获取组合评分历史

    Query Parameters:
        days: 获取最近N天 (默认30)

    Returns:
        评分历史列表
    """
    try:
        days = int(request.args.get('days', 30))
        db = get_db_manager()
        scores = db.get_portfolio_scores(days)

        return jsonify({
            'success': True,
            'scores': scores,
            'count': len(scores)
        })

    except Exception as e:
        logger.error(f'获取评分历史失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@portfolio_bp.route('/score/<int:score_id>', methods=['GET'])
def get_score_detail(score_id: int):
    """获取评分详情"""
    try:
        db = get_db_manager()
        detail = db.get_portfolio_score_detail(score_id)

        if detail:
            return jsonify({'success': True, 'detail': detail})
        else:
            return jsonify({'success': False, 'error': '评分记录不存在'}), 404

    except Exception as e:
        logger.error(f'获取评分详情失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== 资金设置 API ====================

@portfolio_bp.route('/settings/capital', methods=['GET'])
def get_capital_settings():
    """获取资金设置 (总资金/现金)"""
    try:
        db = get_db_manager()
        total_capital = db.get_portfolio_setting('total_capital')
        cash_amount = db.get_portfolio_setting('cash_amount')
        return jsonify({
            'success': True,
            'total_capital': float(total_capital) if total_capital else None,
            'cash_amount': float(cash_amount) if cash_amount else None,
        })
    except Exception as e:
        logger.error(f'获取资金设置失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@portfolio_bp.route('/settings/capital', methods=['POST'])
def set_capital_settings():
    """
    设置资金参数 (持久化)

    Request Body:
        {
            "total_capital": 500000,
            "cash_amount": 50000
        }
    """
    try:
        data = request.get_json() or {}
        db = get_db_manager()
        if 'total_capital' in data:
            db.set_portfolio_setting('total_capital', str(data['total_capital']))
        if 'cash_amount' in data:
            db.set_portfolio_setting('cash_amount', str(data['cash_amount']))
        return jsonify({'success': True, 'message': '资金设置已保存'})
    except Exception as e:
        logger.error(f'设置资金失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== 风控管理 API ====================

@portfolio_bp.route('/risk/update', methods=['POST'])
def risk_update():
    """一键运行风控更新 (trailing stops + risk parity + regime + rebalance)"""
    try:
        db = get_db_manager()
        manager = get_portfolio_manager()
        positions = db.get_all_positions()

        if not positions:
            return jsonify({'success': True, 'message': '当前无持仓', 'result': {}})

        total_capital = float(db.get_portfolio_setting('total_capital') or 0)
        cash_amount = float(db.get_portfolio_setting('cash_amount') or 0)
        trades = db.get_all_trades(limit=200)
        snapshots = db.get_position_snapshots(days=60)

        result = manager.run_daily_risk_update(
            db, positions, total_capital, cash_amount, trades, snapshots)

        return jsonify({
            'success': True,
            'message': f'风控更新完成: {result["regime"]["regime"]}市况, '
                       f'{len(result["triggered_stops"])}只触发止损, '
                       f'{len(result["suggestions"])}条再平衡建议',
            'result': {
                'regime': result['regime'],
                'triggered_stops': result['triggered_stops'],
                'suggestions_count': len(result['suggestions']),
                'timestamp': result['timestamp'],
            }
        })
    except Exception as e:
        logger.error(f'风控更新失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@portfolio_bp.route('/risk/state', methods=['GET'])
def risk_state():
    """获取当前风控状态"""
    try:
        db = get_db_manager()
        state = db.get_portfolio_risk_state()
        return jsonify({'success': True, 'state': state})
    except Exception as e:
        logger.error(f'获取风控状态失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@portfolio_bp.route('/risk/validate', methods=['POST'])
def risk_validate():
    """建仓前预校验"""
    try:
        data = request.get_json() or {}
        code = data.get('code', '')
        quantity = int(data.get('quantity', 0))
        avg_cost = float(data.get('avg_cost', 0))

        if not code or quantity <= 0 or avg_cost <= 0:
            return jsonify({'success': False, 'error': '缺少code/quantity/avg_cost'}), 400

        db = get_db_manager()
        manager = get_portfolio_manager()
        positions = db.get_all_positions()
        total_capital = float(db.get_portfolio_setting('total_capital') or 0)

        validation = manager.validate_new_position(
            code, quantity, avg_cost, total_capital, positions)

        return jsonify({'success': True, 'validation': validation})
    except Exception as e:
        logger.error(f'建仓预校验失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@portfolio_bp.route('/rebalance/suggestions', methods=['GET'])
def get_rebalance_suggestions():
    """获取再平衡建议"""
    try:
        db = get_db_manager()
        status = request.args.get('status')  # None = all
        suggestions = db.get_rebalance_suggestions(status=status)
        # Enrich with stock names
        for s in suggestions:
            if s.get('code'):
                name = db.get_stock_name(s['code'])
                s['stock_name'] = name or ''
        return jsonify({'success': True, 'suggestions': suggestions})
    except Exception as e:
        logger.error(f'获取再平衡建议失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@portfolio_bp.route('/rebalance/suggestions/<int:suggestion_id>', methods=['PUT'])
def update_rebalance_suggestion(suggestion_id: int):
    """标记再平衡建议状态 (executed/dismissed)"""
    try:
        data = request.get_json() or {}
        status = data.get('status', 'dismissed')
        if status not in ('executed', 'dismissed', 'pending'):
            return jsonify({'success': False, 'error': '无效状态'}), 400

        db = get_db_manager()
        success = db.update_rebalance_suggestion(suggestion_id, status)
        if success:
            return jsonify({'success': True, 'message': f'建议已标记为{status}'})
        else:
            return jsonify({'success': False, 'error': '建议不存在'}), 404
    except Exception as e:
        logger.error(f'更新再平衡建议失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500
