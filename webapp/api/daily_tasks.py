"""
日常任务API - 数据更新和选股
"""
import os
import subprocess
import logging
import json
import io
import csv
from datetime import datetime
from flask import Blueprint, jsonify, request, Response, current_app

from core.task_manager import task_manager, TaskType, TaskStatus
from core.report_parser import ReportParser
from api._helpers import api_error_handler, task_progress_sse, get_db_manager


logger = logging.getLogger(__name__)

daily_tasks_bp = Blueprint('daily_tasks', __name__)


def _get_reports_dir(version: str):
    """从 config 拿版本目录, 不存在返回 None。"""
    rd = current_app.config['DAILY_SELECTION_DIRS'].get(version)
    return rd if rd and rd.exists() else None


@daily_tasks_bp.route('/status', methods=['GET'])
@api_error_handler
def get_status():
    """获取数据更新状态 (含 update_status 实时计算)"""
    stats = get_db_manager().get_database_stats()
    running = any(
        t['status'] in (TaskStatus.RUNNING.value, TaskStatus.PENDING.value)
        for t in task_manager.get_tasks_by_type(TaskType.DATA_UPDATE, limit=20)
    )
    return jsonify({
        'success': True,
        'database_stats': stats,
        'last_update': stats.get('latest_date'),
        'update_status': 'running' if running else 'idle',
    })


@daily_tasks_bp.route('/update', methods=['POST'])
@api_error_handler
def trigger_update():
    """触发数据更新"""
    data = request.get_json() or {}
    update_date = data.get('date', datetime.now().strftime('%Y%m%d'))

    task_id = task_manager.submit_task(
        task_type=TaskType.DATA_UPDATE,
        func=_run_data_update,
        metadata={
            'date': update_date,
            '_update_script': str(current_app.config['QUICK_DAILY_UPDATE_SCRIPT']),
            '_python_exec': current_app.config['PYTHON_EXECUTABLE'],
            '_base_dir': str(current_app.config['BASE_DIR']),
            '_task_timeout': current_app.config['TASK_TIMEOUT'],
        }
    )
    return jsonify({'success': True, 'task_id': task_id, 'message': '数据更新任务已启动'})


@daily_tasks_bp.route('/update/stream', methods=['GET'])
def update_stream():
    """SSE: 数据更新进度流"""
    return task_progress_sse(request.args.get('task_id'))


@daily_tasks_bp.route('/selections', methods=['GET'])
@api_error_handler
def get_selections():
    """获取选股日期列表"""
    version = request.args.get('version', 'v3.0')
    limit = int(request.args.get('limit', 50))
    reports_dir = _get_reports_dir(version)
    if reports_dir is None:
        return jsonify({'success': False, 'error': f'版本 {version} 的报告目录不存在'}), 404

    dates = ReportParser.get_selection_dates(reports_dir, limit=limit)
    return jsonify({'success': True, 'version': version, 'dates': dates, 'total': len(dates)})


@daily_tasks_bp.route('/selection/<date>', methods=['GET'])
@api_error_handler
def get_selection_detail(date: str):
    """获取指定日期的选股详情"""
    version = request.args.get('version', 'v3.0')
    reports_dir = _get_reports_dir(version)
    if reports_dir is None:
        return jsonify({'success': False, 'error': f'版本 {version} 的报告目录不存在'}), 404

    selection_data = ReportParser.get_selection_by_date(reports_dir, date)
    if not selection_data:
        return jsonify({'success': False, 'error': f'未找到日期 {date} 的选股数据'}), 404
    return jsonify({'success': True, 'version': version, 'selection': selection_data})


@daily_tasks_bp.route('/selection/<date>/export', methods=['GET'])
@api_error_handler
def export_selection(date: str):
    """导出选股结果为CSV"""
    version = request.args.get('version', 'v3.0')
    reports_dir = _get_reports_dir(version)
    if reports_dir is None:
        return jsonify({'success': False, 'error': f'版本 {version} 的报告目录不存在'}), 404

    selection_data = ReportParser.get_selection_by_date(reports_dir, date)
    if not selection_data:
        return jsonify({'success': False, 'error': f'未找到日期 {date} 的选股数据'}), 404

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        '代码', '名称', '行业', '收盘价', '涨跌幅%', '评分',
        '置信度', '建议', '策略', '策略数', '风险回报比',
        '止损价', '止盈价', '技术评级', '风险评级'
    ])
    for stock in selection_data.get('top_recommendations', []):
        writer.writerow([
            stock.get('code', ''),
            stock.get('name', ''),
            stock.get('industry', ''),
            stock.get('close_price', ''),
            round(stock.get('price_change_pct', 0), 2),
            round(stock.get('score', 0), 2),
            stock.get('confidence', ''),
            stock.get('recommendation', ''),
            ','.join(stock.get('strategies', [])),
            stock.get('strategy_count', 0),
            round(stock.get('risk_reward_ratio', 0), 2),
            stock.get('stop_loss_price', ''),
            stock.get('take_profit_price', ''),
            stock.get('technical_rating', ''),
            stock.get('risk_rating', ''),
        ])
    output.seek(0)

    file_date = date.replace('-', '') if '-' in date else date
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={
            'Content-Disposition': f'attachment; filename=selection_{file_date}.csv',
            'Content-Type': 'text/csv; charset=utf-8',
        }
    )


@daily_tasks_bp.route('/select', methods=['POST'])
@api_error_handler
def trigger_selection():
    """触发选股任务"""
    data = request.get_json() or {}
    select_date = data.get('date', datetime.now().strftime('%Y-%m-%d'))
    version = data.get('version', 'v3.9')

    task_id = task_manager.submit_task(
        task_type=TaskType.STOCK_SELECTION,
        func=_run_stock_selection,
        metadata={
            'date': select_date,
            'version': version,
            '_selector_script': str(current_app.config['STOCK_SELECTOR_SCRIPT']),
            '_python_exec': current_app.config['PYTHON_EXECUTABLE'],
            '_base_dir': str(current_app.config['BASE_DIR']),
            '_task_timeout': current_app.config['TASK_TIMEOUT'],
        }
    )
    return jsonify({'success': True, 'task_id': task_id, 'message': '选股任务已启动'})


@daily_tasks_bp.route('/select/stream', methods=['GET'])
def select_stream():
    """SSE: 选股进度流"""
    return task_progress_sse(request.args.get('task_id'))


@daily_tasks_bp.route('/reports', methods=['GET'])
@api_error_handler
def get_reports():
    """获取选股报告列表"""
    version = request.args.get('version', 'v3.9')
    limit = int(request.args.get('limit', 50))
    reports_dir = _get_reports_dir(version)
    if reports_dir is None:
        return jsonify({'success': True, 'reports': [], 'total': 0})

    reports = []
    json_files = sorted(reports_dir.glob('analysis_data_*.json'),
                        key=lambda f: f.stat().st_mtime, reverse=True)
    for jf in json_files[:limit]:
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            raw_date = jf.stem.replace('analysis_data_', '')
            date_str = (f'{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}'
                        if len(raw_date) == 8 else raw_date)
            reports.append({
                'date': date_str,
                'file': jf.name,
                'stock_count': len(data.get('top_recommendations', [])),
                'strategies_used': list(data.get('strategy_results', {}).keys()),
                'modified_time': datetime.fromtimestamp(jf.stat().st_mtime).strftime('%Y-%m-%d %H:%M'),
            })
        except Exception as e:
            logger.warning("解析报告失败 %s: %s", jf, e)

    return jsonify({'success': True, 'reports': reports, 'total': len(reports), 'version': version})


@daily_tasks_bp.route('/market-indices', methods=['GET'])
@api_error_handler
def get_market_indices():
    """获取大盘指数最新数据 (含 30d 趋势)"""
    db_manager = get_db_manager()
    indices = db_manager.get_market_indices()
    for idx in indices:
        history = db_manager.get_index_history(idx['code'], days=30)
        idx['trend'] = [h['close'] for h in history]
    return jsonify({'success': True, 'indices': indices})


@daily_tasks_bp.route('/ai-analysis/<date>', methods=['GET'])
@api_error_handler
def get_ai_analysis(date: str):
    """获取AI分析报告"""
    file_date = date if len(date) == 8 else date.replace('-', '')
    ai_dir = current_app.config['AI_ENHANCED_DIR']
    if not ai_dir.exists():
        return jsonify({'success': False, 'error': 'AI分析报告目录不存在'}), 404

    report_file = ai_dir / f'AI分析报告_{file_date}.md'
    if not report_file.exists():
        return jsonify({'success': False, 'error': f'未找到日期 {date} 的AI分析报告'}), 404

    return jsonify({'success': True, 'analysis': ReportParser.parse_ai_analysis_report(report_file)})


# ==================== 内部任务执行函数 ====================

def _run_subprocess_task(progress_callback, cmd, base_dir, task_timeout, label):
    """执行外部脚本的统一封装。"""
    progress_callback(10, f'执行{label}脚本...')
    env = os.environ.copy()
    env['PYTHONPATH'] = base_dir
    try:
        result = subprocess.run(cmd, cwd=base_dir, capture_output=True,
                                text=True, timeout=task_timeout, env=env)
    except subprocess.TimeoutExpired:
        raise Exception(f'{label}超时')
    if result.returncode != 0:
        error_msg = result.stderr[:500] if result.stderr else '未知错误'
        raise Exception(f'{label}失败: {error_msg}')
    return result.stdout


def _run_data_update(progress_callback, **params):
    """执行数据更新任务"""
    try:
        progress_callback(0, '开始数据更新...')
        cmd = [params.get('_python_exec', 'python3'), params['_update_script']]
        if params.get('date'):
            cmd.extend(['--date', params['date']])
        out = _run_subprocess_task(progress_callback, cmd, params['_base_dir'],
                                   params.get('_task_timeout', 3600), '数据更新')
        progress_callback(100, '数据更新完成')
        return {'success': True, 'output': out}
    except Exception as e:
        logger.error("数据更新任务失败: %s", e)
        raise


def _run_stock_selection(progress_callback, **params):
    """执行选股任务"""
    try:
        progress_callback(0, '开始选股...')
        version = params.get('version', 'v3.9')
        cmd = [params.get('_python_exec', 'python3'),
               params['_selector_script'], params['date'],
               '--scoring-version', version]
        out = _run_subprocess_task(progress_callback, cmd, params['_base_dir'],
                                   params.get('_task_timeout', 3600),
                                   f'选股(版本 {version})')
        progress_callback(100, '选股完成')
        return {'success': True, 'output': out}
    except Exception as e:
        logger.error("选股任务失败: %s", e)
        raise
