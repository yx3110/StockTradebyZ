"""
任务管理API - 查看和管理所有后台任务
"""
import logging
from flask import Blueprint, jsonify, request

from core.task_manager import task_manager
from api._helpers import api_error_handler

logger = logging.getLogger(__name__)

tasks_bp = Blueprint('tasks', __name__)


@tasks_bp.route('/', methods=['GET'])
@api_error_handler
def get_all_tasks():
    """获取所有任务列表 (支持 type / status 过滤)"""
    limit = int(request.args.get('limit', 50))
    task_type = request.args.get('type')
    status_filter = request.args.get('status')

    # 先取足够大的集合, 过滤后再截断 —— 否则 type/status 过滤发生在 limit 截断之后,
    # 会把命中项挡在窗口外 (例如新任务刷屏时按 type 过滤可能返回空)。
    fetch_limit = max(limit, 1000) if (task_type or status_filter) else limit
    all_tasks = task_manager.get_all_tasks(limit=fetch_limit)
    if task_type:
        all_tasks = [t for t in all_tasks if t.get('type') == task_type]
    if status_filter:
        all_tasks = [t for t in all_tasks if t.get('status') == status_filter]
    all_tasks = all_tasks[:limit]

    return jsonify({'success': True, 'tasks': all_tasks, 'total': len(all_tasks)})


@tasks_bp.route('/running', methods=['GET'])
@api_error_handler
def get_running_tasks():
    """获取所有正在运行/排队的任务"""
    running = [t for t in task_manager.get_all_tasks(limit=100)
               if t.get('status') in ('running', 'pending')]
    return jsonify({'success': True, 'tasks': running, 'count': len(running)})


@tasks_bp.route('/summary', methods=['GET'])
@api_error_handler
def get_tasks_summary():
    """获取任务汇总统计 (按状态 + 按类型)"""
    all_tasks = task_manager.get_all_tasks(limit=1000)

    status_counts = {'running': 0, 'pending': 0, 'completed': 0, 'failed': 0, 'cancelled': 0}
    for task in all_tasks:
        status = task.get('status', 'unknown')
        if status in status_counts:
            status_counts[status] += 1

    type_counts = {}
    for task in all_tasks:
        task_type = task.get('type', 'unknown')
        slot = type_counts.setdefault(task_type, {'total': 0, 'running': 0, 'completed': 0, 'failed': 0})
        slot['total'] += 1
        status = task.get('status')
        if status in ('running', 'pending'):
            slot['running'] += 1
        elif status == 'completed':
            slot['completed'] += 1
        elif status == 'failed':
            slot['failed'] += 1

    return jsonify({
        'success': True,
        'summary': {'total': len(all_tasks), **status_counts, 'by_type': type_counts},
    })


@tasks_bp.route('/<task_id>', methods=['GET'])
@api_error_handler
def get_task_detail(task_id: str):
    """获取单个任务详情"""
    task_info = task_manager.get_task_status(task_id)
    if not task_info:
        return jsonify({'success': False, 'error': f'任务不存在: {task_id}'}), 404
    return jsonify({'success': True, 'task': task_info.to_dict()})


@tasks_bp.route('/<task_id>/cancel', methods=['POST'])
@api_error_handler
def cancel_task(task_id: str):
    """取消任务"""
    if task_manager.cancel_task(task_id):
        return jsonify({'success': True, 'message': '任务已取消'})
    return jsonify({'success': False, 'error': '无法取消任务（可能已完成或正在运行）'}), 400


@tasks_bp.route('/cleanup', methods=['POST'])
@api_error_handler
def cleanup_tasks():
    """清理旧任务 (默认 24h)"""
    max_age_hours = (request.get_json(silent=True) or {}).get('max_age_hours', 24)
    task_manager.cleanup_old_tasks(max_age_hours=max_age_hours)
    return jsonify({'success': True, 'message': f'已清理 {max_age_hours} 小时前的任务'})
