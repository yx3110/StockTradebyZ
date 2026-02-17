"""
日常任务API - 数据更新和选股
"""
import os
import subprocess
import logging
from datetime import datetime, timedelta
from pathlib import Path
from flask import Blueprint, jsonify, request, Response, current_app
import json

from core.task_manager import task_manager, TaskType
from core.database import DatabaseManager
from core.report_parser import ReportParser


logger = logging.getLogger(__name__)

daily_tasks_bp = Blueprint('daily_tasks', __name__)


def get_db_manager():
    """获取数据库管理器实例"""
    return DatabaseManager(
        current_app.config['STOCK_DB_PATH'],
        current_app.config['WEBAPP_DB_PATH']
    )


@daily_tasks_bp.route('/status', methods=['GET'])
def get_status():
    """
    获取数据更新状态

    Returns:
        {
            "database_stats": {...},
            "last_update": "2025-11-24",
            "update_status": "idle|running|failed"
        }
    """
    try:
        db_manager = get_db_manager()
        stats = db_manager.get_database_stats()

        return jsonify({
            'success': True,
            'database_stats': stats,
            'last_update': stats.get('latest_date'),
            'update_status': 'idle'  # TODO: 实际检查更新状态
        })

    except Exception as e:
        logger.error(f'获取状态失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@daily_tasks_bp.route('/update', methods=['POST'])
def trigger_update():
    """
    触发数据更新

    Request Body:
        {
            "date": "2025-11-24"  # 可选，默认当天
        }

    Returns:
        {
            "success": True,
            "task_id": "uuid"
        }
    """
    try:
        data = request.get_json() or {}
        update_date = data.get('date', datetime.now().strftime('%Y%m%d'))

        # 在主线程中获取配置（线程池中无法访问current_app）
        update_script = str(current_app.config['QUICK_DAILY_UPDATE_SCRIPT'])
        python_exec = current_app.config['PYTHON_EXECUTABLE']
        base_dir = str(current_app.config['BASE_DIR'])
        task_timeout = current_app.config['TASK_TIMEOUT']

        # 提交更新任务
        task_id = task_manager.submit_task(
            task_type=TaskType.DATA_UPDATE,
            func=_run_data_update,
            metadata={
                'date': update_date,
                '_update_script': update_script,
                '_python_exec': python_exec,
                '_base_dir': base_dir,
                '_task_timeout': task_timeout
            }
        )

        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': '数据更新任务已启动'
        })

    except Exception as e:
        logger.error(f'触发数据更新失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@daily_tasks_bp.route('/update/stream', methods=['GET'])
def update_stream():
    """
    SSE端点：实时推送数据更新进度

    Query Parameters:
        task_id: 任务ID

    Returns:
        Server-Sent Events stream
    """
    task_id = request.args.get('task_id')

    if not task_id:
        return jsonify({'error': 'Missing task_id'}), 400

    def generate():
        """生成SSE数据流"""
        for progress_data in task_manager.get_task_progress_stream(task_id):
            yield f"data: {json.dumps(progress_data, ensure_ascii=False)}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@daily_tasks_bp.route('/selections', methods=['GET'])
def get_selections():
    """
    获取选股日期列表

    Query Parameters:
        version: v3.0|v3.7|v3.8|v3.81|v3.9 (默认v3.0)
        limit: 返回数量限制 (默认50)

    Returns:
        {
            "success": True,
            "dates": ["2025-09-30", "2025-09-29", ...],
            "version": "v3.0"
        }
    """
    try:
        version = request.args.get('version', 'v3.0')
        limit = int(request.args.get('limit', 50))

        # 获取报告目录
        reports_dir = current_app.config['DAILY_SELECTION_DIRS'].get(version)
        if not reports_dir or not reports_dir.exists():
            return jsonify({
                'success': False,
                'error': f'版本 {version} 的报告目录不存在'
            }), 404

        # 获取可用日期列表
        dates = ReportParser.get_selection_dates(reports_dir, limit=limit)

        return jsonify({
            'success': True,
            'version': version,
            'dates': dates,
            'total': len(dates)
        })

    except Exception as e:
        logger.error(f'获取选股列表失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@daily_tasks_bp.route('/selection/<date>', methods=['GET'])
def get_selection_detail(date: str):
    """
    获取指定日期的选股详情

    Path Parameters:
        date: 日期（YYYY-MM-DD或YYYYMMDD）

    Query Parameters:
        version: v3.0|v3.7|v3.8|v3.81|v3.9 (默认v3.0)

    Returns:
        {
            "success": True,
            "selection": {
                "date": "2025-09-30",
                "strategy_results": {...},
                "top_recommendations": [...],
                ...
            }
        }
    """
    try:
        version = request.args.get('version', 'v3.0')

        # 获取报告目录
        reports_dir = current_app.config['DAILY_SELECTION_DIRS'].get(version)
        if not reports_dir or not reports_dir.exists():
            return jsonify({
                'success': False,
                'error': f'版本 {version} 的报告目录不存在'
            }), 404

        # 获取选股数据
        selection_data = ReportParser.get_selection_by_date(reports_dir, date)

        if not selection_data:
            return jsonify({
                'success': False,
                'error': f'未找到日期 {date} 的选股数据'
            }), 404

        return jsonify({
            'success': True,
            'version': version,
            'selection': selection_data
        })

    except Exception as e:
        logger.error(f'获取选股详情失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@daily_tasks_bp.route('/selection/<date>/export', methods=['GET'])
def export_selection(date: str):
    """
    导出选股结果为CSV

    Path Parameters:
        date: 日期（YYYY-MM-DD或YYYYMMDD）

    Query Parameters:
        version: v3.0|v3.7|v3.8|v3.81|v3.9 (默认v3.0)

    Returns:
        CSV文件下载
    """
    try:
        version = request.args.get('version', 'v3.0')

        # 获取报告目录
        reports_dir = current_app.config['DAILY_SELECTION_DIRS'].get(version)
        if not reports_dir or not reports_dir.exists():
            return jsonify({'success': False, 'error': f'版本 {version} 的报告目录不存在'}), 404

        # 获取选股数据
        selection_data = ReportParser.get_selection_by_date(reports_dir, date)
        if not selection_data:
            return jsonify({'success': False, 'error': f'未找到日期 {date} 的选股数据'}), 404

        # 生成CSV
        import io
        import csv

        output = io.StringIO()
        writer = csv.writer(output)

        # 写入表头
        writer.writerow([
            '代码', '名称', '行业', '收盘价', '涨跌幅%', '评分',
            '置信度', '建议', '策略', '策略数', '风险回报比',
            '止损价', '止盈价', '技术评级', '风险评级'
        ])

        # 写入数据
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

        # 格式化日期
        file_date = date.replace('-', '') if '-' in date else date

        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename=selection_{file_date}.csv',
                'Content-Type': 'text/csv; charset=utf-8'
            }
        )

    except Exception as e:
        logger.error(f'导出选股结果失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@daily_tasks_bp.route('/select', methods=['POST'])
def trigger_selection():
    """
    触发选股任务

    Request Body:
        {
            "date": "2025-11-24",
            "version": "v3.9"  # 可选
        }

    Returns:
        {
            "success": True,
            "task_id": "uuid"
        }
    """
    try:
        data = request.get_json() or {}
        select_date = data.get('date', datetime.now().strftime('%Y-%m-%d'))
        version = data.get('version', 'v3.9')

        # 在主线程中获取配置（线程池中无法访问current_app）
        selector_script = str(current_app.config['STOCK_SELECTOR_SCRIPT'])
        python_exec = current_app.config['PYTHON_EXECUTABLE']
        base_dir = str(current_app.config['BASE_DIR'])
        task_timeout = current_app.config['TASK_TIMEOUT']

        # 提交选股任务
        task_id = task_manager.submit_task(
            task_type=TaskType.STOCK_SELECTION,
            func=_run_stock_selection,
            metadata={
                'date': select_date,
                'version': version,
                '_selector_script': selector_script,
                '_python_exec': python_exec,
                '_base_dir': base_dir,
                '_task_timeout': task_timeout
            }
        )

        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': '选股任务已启动'
        })

    except Exception as e:
        logger.error(f'触发选股失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@daily_tasks_bp.route('/select/stream', methods=['GET'])
def select_stream():
    """
    SSE端点：实时推送选股进度

    Query Parameters:
        task_id: 任务ID

    Returns:
        Server-Sent Events stream
    """
    task_id = request.args.get('task_id')

    if not task_id:
        return jsonify({'error': 'Missing task_id'}), 400

    def generate():
        """生成SSE数据流"""
        for progress_data in task_manager.get_task_progress_stream(task_id):
            yield f"data: {json.dumps(progress_data, ensure_ascii=False)}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@daily_tasks_bp.route('/reports', methods=['GET'])
def get_reports():
    """
    获取选股报告列表

    Query Parameters:
        version: v3.0|v3.7|v3.8|v3.81|v3.9 (默认v3.9)
        limit: 返回数量限制 (默认50)

    Returns:
        {
            "success": True,
            "reports": [...],
            "total": 50
        }
    """
    try:
        version = request.args.get('version', 'v3.9')
        limit = int(request.args.get('limit', 50))

        # 获取报告目录
        reports_dir = current_app.config['DAILY_SELECTION_DIRS'].get(version)
        if not reports_dir or not reports_dir.exists():
            return jsonify({
                'success': True,
                'reports': [],
                'total': 0
            })

        # 获取报告列表
        reports = []
        json_files = list(reports_dir.glob('selection_*.json'))
        json_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        for jf in json_files[:limit]:
            try:
                with open(jf, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 从文件名提取日期
                date_str = jf.stem.replace('selection_', '')

                reports.append({
                    'date': date_str,
                    'file': jf.name,
                    'stock_count': len(data.get('top_recommendations', [])),
                    'strategies_used': list(data.get('strategy_results', {}).keys()),
                    'modified_time': datetime.fromtimestamp(jf.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
                })
            except Exception as e:
                logger.warning(f'解析报告失败: {jf}, {e}')

        return jsonify({
            'success': True,
            'reports': reports,
            'total': len(reports),
            'version': version
        })

    except Exception as e:
        logger.error(f'获取报告列表失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@daily_tasks_bp.route('/ai-analysis/<date>', methods=['GET'])
def get_ai_analysis(date: str):
    """
    获取AI分析报告

    Path Parameters:
        date: 日期（YYYY-MM-DD或YYYYMMDD）

    Returns:
        {
            "success": True,
            "analysis": {...}
        }
    """
    try:
        # 格式化日期
        if len(date) == 8:  # YYYYMMDD
            file_date = date
        else:  # YYYY-MM-DD
            file_date = date.replace('-', '')

        # 获取AI分析报告目录
        ai_dir = current_app.config['AI_ENHANCED_DIR']
        if not ai_dir.exists():
            return jsonify({
                'success': False,
                'error': 'AI分析报告目录不存在'
            }), 404

        # 查找报告文件
        report_file = ai_dir / f'AI分析报告_{file_date}.md'

        if not report_file.exists():
            return jsonify({
                'success': False,
                'error': f'未找到日期 {date} 的AI分析报告'
            }), 404

        # 解析报告
        analysis_data = ReportParser.parse_ai_analysis_report(report_file)

        return jsonify({
            'success': True,
            'analysis': analysis_data
        })

    except Exception as e:
        logger.error(f'获取AI分析失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== 内部任务执行函数 ====================

def _run_data_update(progress_callback, **params):
    """
    执行数据更新任务

    Args:
        progress_callback: 进度回调函数
        **params: 参数（包含配置信息）
    """
    try:
        progress_callback(0, '开始数据更新...')

        # 从参数获取配置（在主线程中已提前获取）
        script_path = params.get('_update_script')
        python_exec = params.get('_python_exec', 'python3')
        base_dir = params.get('_base_dir')
        task_timeout = params.get('_task_timeout', 3600)
        date = params.get('date')

        cmd = [python_exec, script_path]
        if date:
            cmd.extend(['--date', date])

        progress_callback(10, '执行数据更新脚本...')

        # 设置环境变量
        env = os.environ.copy()
        env['PYTHONPATH'] = base_dir

        # 执行脚本
        result = subprocess.run(
            cmd,
            cwd=base_dir,
            capture_output=True,
            text=True,
            timeout=task_timeout,
            env=env
        )

        if result.returncode == 0:
            progress_callback(100, '数据更新完成')
            return {'success': True, 'output': result.stdout}
        else:
            error_msg = result.stderr[:500] if result.stderr else '未知错误'
            raise Exception(f'数据更新失败: {error_msg}')

    except subprocess.TimeoutExpired:
        raise Exception('数据更新超时（超过1小时）')
    except Exception as e:
        logger.error(f'数据更新任务失败: {e}')
        raise


def _run_stock_selection(progress_callback, **params):
    """
    执行选股任务

    Args:
        progress_callback: 进度回调函数
        **params: 参数（包含配置信息）
    """
    try:
        progress_callback(0, '开始选股...')

        # 从参数获取配置（在主线程中已提前获取）
        script_path = params.get('_selector_script')
        python_exec = params.get('_python_exec', 'python3')
        base_dir = params.get('_base_dir')
        task_timeout = params.get('_task_timeout', 3600)
        date = params.get('date')
        version = params.get('version', 'v3.9')

        cmd = [
            python_exec,
            script_path,
            date,
            '--scoring-version',
            version
        ]

        progress_callback(10, f'执行选股脚本（版本: {version}）...')

        # 设置环境变量
        env = os.environ.copy()
        env['PYTHONPATH'] = base_dir

        # 执行脚本
        result = subprocess.run(
            cmd,
            cwd=base_dir,
            capture_output=True,
            text=True,
            timeout=task_timeout,
            env=env
        )

        if result.returncode == 0:
            progress_callback(100, '选股完成')
            return {'success': True, 'output': result.stdout}
        else:
            error_msg = result.stderr[:500] if result.stderr else '未知错误'
            raise Exception(f'选股失败: {error_msg}')

    except subprocess.TimeoutExpired:
        raise Exception('选股超时（超过1小时）')
    except Exception as e:
        logger.error(f'选股任务失败: {e}')
        raise
