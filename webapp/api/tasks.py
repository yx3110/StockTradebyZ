"""
任务管理API - 查看和管理所有后台任务
"""
import logging
from flask import Blueprint, jsonify, request

from core.task_manager import task_manager, TaskType, TaskStatus

logger = logging.getLogger(__name__)

tasks_bp = Blueprint('tasks', __name__)


@tasks_bp.route('/', methods=['GET'])
def get_all_tasks():
    """
    获取所有任务列表

    Query Parameters:
        limit: 返回数量限制（默认50）
        type: 任务类型过滤（可选）
        status: 任务状态过滤（可选）

    Returns:
        {
            "success": True,
            "tasks": [...],
            "total": 10
        }
    """
    try:
        limit = int(request.args.get('limit', 50))
        task_type = request.args.get('type')
        status_filter = request.args.get('status')

        # 获取所有任务
        all_tasks = task_manager.get_all_tasks(limit=limit)

        # 按类型过滤
        if task_type:
            all_tasks = [t for t in all_tasks if t.get('type') == task_type]

        # 按状态过滤
        if status_filter:
            all_tasks = [t for t in all_tasks if t.get('status') == status_filter]

        return jsonify({
            'success': True,
            'tasks': all_tasks,
            'total': len(all_tasks)
        })

    except Exception as e:
        logger.error(f'获取任务列表失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@tasks_bp.route('/running', methods=['GET'])
def get_running_tasks():
    """
    获取所有正在运行的任务

    Returns:
        {
            "success": True,
            "tasks": [...],
            "count": 2
        }
    """
    try:
        all_tasks = task_manager.get_all_tasks(limit=100)
        running_tasks = [
            t for t in all_tasks
            if t.get('status') in ['running', 'pending']
        ]

        return jsonify({
            'success': True,
            'tasks': running_tasks,
            'count': len(running_tasks)
        })

    except Exception as e:
        logger.error(f'获取运行中任务失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@tasks_bp.route('/summary', methods=['GET'])
def get_tasks_summary():
    """
    获取任务汇总统计

    Returns:
        {
            "success": True,
            "summary": {
                "total": 10,
                "running": 2,
                "pending": 1,
                "completed": 6,
                "failed": 1,
                "by_type": {...}
            }
        }
    """
    try:
        all_tasks = task_manager.get_all_tasks(limit=1000)

        # 按状态统计
        status_counts = {
            'running': 0,
            'pending': 0,
            'completed': 0,
            'failed': 0,
            'cancelled': 0
        }
        for task in all_tasks:
            status = task.get('status', 'unknown')
            if status in status_counts:
                status_counts[status] += 1

        # 按类型统计
        type_counts = {}
        for task in all_tasks:
            task_type = task.get('type', 'unknown')
            if task_type not in type_counts:
                type_counts[task_type] = {
                    'total': 0,
                    'running': 0,
                    'completed': 0,
                    'failed': 0
                }
            type_counts[task_type]['total'] += 1
            status = task.get('status')
            if status in ['running', 'pending']:
                type_counts[task_type]['running'] += 1
            elif status == 'completed':
                type_counts[task_type]['completed'] += 1
            elif status == 'failed':
                type_counts[task_type]['failed'] += 1

        summary = {
            'total': len(all_tasks),
            **status_counts,
            'by_type': type_counts
        }

        return jsonify({
            'success': True,
            'summary': summary
        })

    except Exception as e:
        logger.error(f'获取任务汇总失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@tasks_bp.route('/<task_id>', methods=['GET'])
def get_task_detail(task_id: str):
    """
    获取单个任务详情

    Path Parameters:
        task_id: 任务ID

    Returns:
        {
            "success": True,
            "task": {...}
        }
    """
    try:
        task_info = task_manager.get_task_status(task_id)

        if not task_info:
            return jsonify({
                'success': False,
                'error': f'任务不存在: {task_id}'
            }), 404

        return jsonify({
            'success': True,
            'task': task_info.to_dict()
        })

    except Exception as e:
        logger.error(f'获取任务详情失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@tasks_bp.route('/<task_id>/cancel', methods=['POST'])
def cancel_task(task_id: str):
    """
    取消任务

    Path Parameters:
        task_id: 任务ID

    Returns:
        {
            "success": True,
            "message": "任务已取消"
        }
    """
    try:
        success = task_manager.cancel_task(task_id)

        if success:
            return jsonify({
                'success': True,
                'message': '任务已取消'
            })
        else:
            return jsonify({
                'success': False,
                'error': '无法取消任务（可能已完成或正在运行）'
            }), 400

    except Exception as e:
        logger.error(f'取消任务失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@tasks_bp.route('/cleanup', methods=['POST'])
def cleanup_tasks():
    """
    清理旧任务

    Request Body:
        {
            "max_age_hours": 24  # 可选，默认24小时
        }

    Returns:
        {
            "success": True,
            "message": "清理完成"
        }
    """
    try:
        data = request.get_json() or {}
        max_age_hours = data.get('max_age_hours', 24)

        task_manager.cleanup_old_tasks(max_age_hours=max_age_hours)

        return jsonify({
            'success': True,
            'message': f'已清理 {max_age_hours} 小时前的任务'
        })

    except Exception as e:
        logger.error(f'清理任务失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500
