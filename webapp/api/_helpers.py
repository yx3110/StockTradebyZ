"""
api blueprint shared helpers
"""
import json
import logging
from functools import wraps
from flask import jsonify, Response, current_app

from core.task_manager import task_manager
from core.database import DatabaseManager


def get_db_manager() -> DatabaseManager:
    """获取 DatabaseManager 实例 (使用当前 app config 的 DB 路径)。"""
    return DatabaseManager(
        current_app.config['STOCK_DB_PATH'],
        current_app.config['WEBAPP_DB_PATH'],
    )


def api_error_handler(view_func):
    """Wrap a Flask view in try/except → JSON error response.

    Logs with the view's own module logger so the log line still attributes
    to api/portfolio.py etc. instead of api/_helpers.py.

    Use after the route decorator:

        @bp.route('/foo')
        @api_error_handler
        def foo():
            ...
            return jsonify({'success': True, ...})
    """
    logger = logging.getLogger(view_func.__module__)

    @wraps(view_func)
    def wrapper(*args, **kwargs):
        try:
            return view_func(*args, **kwargs)
        except Exception as e:
            logger.error("%s failed: %s", view_func.__name__, e, exc_info=True)
            return jsonify({'success': False, 'error': str(e)}), 500

    return wrapper


def task_progress_sse(task_id):
    """Return an SSE Response streaming progress events for `task_id`.

    Mirrors the boilerplate previously copy-pasted across daily_tasks /
    backtest / data_management / model_training stream endpoints.
    Returns 400 JSON if task_id is missing.
    """
    if not task_id:
        return jsonify({'error': 'Missing task_id'}), 400

    def generate():
        for event in task_manager.get_task_progress_stream(task_id):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )
