"""
StockTradebyZ Web可视化系统 - Flask主应用
"""
import os
import logging
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

from config import config


def create_app(config_name='default'):
    """Flask应用工厂"""

    app = Flask(__name__)
    app.config.from_object(config[config_name])

    # 启用CORS
    CORS(app)

    # 配置日志
    setup_logging(app)

    # 确保必要目录存在
    ensure_directories(app)

    # 注册蓝图
    register_blueprints(app)

    # 注册错误处理器
    register_error_handlers(app)

    # 注册上下文处理器
    register_context_processors(app)

    # 主页路由
    @app.route('/')
    def index():
        """首页/仪表板"""
        return render_template('index.html')

    @app.route('/daily-tasks')
    def daily_tasks():
        """日常任务页面"""
        return render_template('daily_tasks.html')

    @app.route('/model-training')
    def model_training():
        """模型训练页面"""
        return render_template('model_training.html')

    @app.route('/backtest')
    def backtest():
        """回测页面"""
        return render_template('backtest.html')

    @app.route('/portfolio')
    def portfolio():
        """持仓管理页面"""
        return render_template('portfolio.html')

    @app.route('/data-management')
    def data_management():
        """数据管理页面"""
        return render_template('data_management.html')

    @app.route('/stock/<code>')
    def stock_detail(code):
        """个股详情页"""
        return render_template('stock_detail.html', stock_code=code)

    # 健康检查端点
    @app.route('/api/health')
    def health_check():
        """健康检查"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        })

    return app


def setup_logging(app):
    """配置日志系统"""
    if not app.debug:
        # 生产环境日志配置
        log_dir = app.config['WEBAPP_DIR'] / 'logs'
        log_dir.mkdir(exist_ok=True)

        log_file = log_dir / 'webapp.log'

        # 文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(
            '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
        ))

        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
        app.logger.info('StockTradebyZ Web应用启动')
    else:
        # 开发环境使用控制台日志
        app.logger.setLevel(logging.DEBUG)


def ensure_directories(app):
    """确保必要的目录存在"""
    dirs_to_create = [
        app.config['WEBAPP_DIR'] / 'data',
        app.config['WEBAPP_DIR'] / 'logs',
        app.config['WEBAPP_DIR'] / 'configs',
        app.config['WEBAPP_DIR'] / 'data' / 'cache',
    ]

    for directory in dirs_to_create:
        directory.mkdir(parents=True, exist_ok=True)


def register_blueprints(app):
    """注册蓝图"""
    from api.daily_tasks import daily_tasks_bp
    from api.model_training import model_training_bp
    from api.backtest import backtest_bp
    from api.tasks import tasks_bp
    from api.portfolio import portfolio_bp
    from api.data_management import data_management_bp
    from api.stock import stock_bp
    from api.data_explorer import data_explorer_bp   # NEW

    app.register_blueprint(daily_tasks_bp, url_prefix='/api/daily')
    app.register_blueprint(model_training_bp, url_prefix='/api/models')
    app.register_blueprint(backtest_bp, url_prefix='/api/backtest')
    app.register_blueprint(tasks_bp, url_prefix='/api/tasks')
    app.register_blueprint(portfolio_bp, url_prefix='/api/portfolio')
    app.register_blueprint(data_management_bp, url_prefix='/api/data')
    app.register_blueprint(stock_bp, url_prefix='/api/stock')
    app.register_blueprint(data_explorer_bp, url_prefix='/api/explorer')  # NEW


def register_error_handlers(app):
    """注册错误处理器"""

    @app.errorhandler(404)
    def not_found_error(error):
        if request.path.startswith('/api/'):
            return jsonify({'error': 'Not Found', 'message': str(error)}), 404
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f'Internal Server Error: {error}')
        if request.path.startswith('/api/'):
            return jsonify({'error': 'Internal Server Error', 'message': str(error)}), 500
        return render_template('500.html'), 500

    @app.errorhandler(Exception)
    def handle_exception(error):
        app.logger.error(f'Unhandled Exception: {error}', exc_info=True)
        if request.path.startswith('/api/'):
            return jsonify({'error': 'Internal Server Error', 'message': str(error)}), 500
        return render_template('500.html'), 500


def register_context_processors(app):
    """注册上下文处理器"""

    @app.context_processor
    def inject_globals():
        """注入全局变量到模板"""
        # 动态获取所有报告版本
        selection_dirs = app.config.get('DAILY_SELECTION_DIRS', {})
        versions = sorted(selection_dirs.keys())
        return {
            'now': datetime.now(),
            'app_name': 'StockTradebyZ Web',
            'app_version': '1.0.0',
            'selection_versions': versions,
        }


if __name__ == '__main__':
    app = create_app(os.environ.get('FLASK_ENV', 'development'))
    port = int(os.environ.get('PORT', 8000))
    # 禁用auto-reloader以防止SSE连接中断
    # 当项目中任何Python文件变化时，reloader会重启Flask，导致所有SSE连接断开
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True, use_reloader=False)
