"""
Flask应用配置文件
"""
import os
import re
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent
WEBAPP_DIR = Path(__file__).resolve().parent


def scan_model_directories(models_dir: Path) -> dict:
    """
    自动扫描模型目录，发现所有可用的模型版本

    目录命名规则:
    - v360 -> v3.6
    - v370 -> v3.7
    - v380 -> v3.8
    - v39 -> v3.9
    - v391 -> v3.91
    - v392 -> v3.92

    Returns:
        dict: {版本号: 目录路径} 例如 {'v3.9': Path(...), 'v3.91': Path(...)}
    """
    model_dirs = {}

    if not models_dir.exists():
        return model_dirs

    # 扫描所有以v开头的目录
    for item in models_dir.iterdir():
        if item.is_dir() and item.name.startswith('v'):
            dir_name = item.name

            # 解析目录名称
            # v360 -> 3.6.0 -> v3.6
            # v370 -> 3.7.0 -> v3.7
            # v39 -> 3.9 -> v3.9
            # v391 -> 3.9.1 -> v3.91

            # 移除v前缀
            version_str = dir_name[1:]

            # 尝试解析版本号
            if len(version_str) == 2:
                # v39 -> 3.9
                major = version_str[0]
                minor = version_str[1]
                version = f"v{major}.{minor}"
            elif len(version_str) == 3:
                if version_str.endswith('0'):
                    # v360, v370, v380 -> 3.6, 3.7, 3.8
                    major = version_str[0]
                    minor = version_str[1]
                    version = f"v{major}.{minor}"
                else:
                    # v391, v392 -> 3.91, 3.92
                    major = version_str[0]
                    minor = version_str[1:]
                    version = f"v{major}.{minor}"
            else:
                # 跳过无法识别的目录
                continue

            model_dirs[version] = item

    # v3.81 使用 v380 目录，需要特殊处理
    if 'v3.8' in model_dirs:
        # 检查是否存在 level4 模型文件
        level4_file = models_dir / 'level4_quality_meta_learner.pkl'
        if level4_file.exists():
            model_dirs['v3.81'] = model_dirs['v3.8']

    return model_dirs


class Config:
    """基础配置"""

    # Flask配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = True

    # 目录配置
    BASE_DIR = BASE_DIR
    WEBAPP_DIR = WEBAPP_DIR

    # 数据库配置
    STOCK_DB_PATH = BASE_DIR / 'data_adapter' / 'stock_data.db'
    WEBAPP_DB_PATH = WEBAPP_DIR / 'data' / 'webapp.db'

    # 报告目录配置
    REPORTS_DIR = BASE_DIR / 'reports'
    DAILY_SELECTION_DIRS = {
        'v3.0': REPORTS_DIR / 'daily_selection_v3',
        'v3.7': REPORTS_DIR / 'daily_selection_v3.7',
        'v3.8': REPORTS_DIR / 'daily_selection_v3.8',
        'v3.81': REPORTS_DIR / 'daily_selection_v3.81',
        'v3.9': REPORTS_DIR / 'daily_selection_v3.9',
        'v3.94': REPORTS_DIR / 'daily_selection_v3.94',
        'v3.95': REPORTS_DIR / 'daily_selection_v3.95',
    }
    AI_ENHANCED_DIR = REPORTS_DIR / 'ai_enhanced'
    BACKTEST_DIR = REPORTS_DIR / 'backtest'

    # 模型目录配置
    MODELS_DIR = BASE_DIR / 'models'
    # 自动扫描模型目录
    MODEL_DIRS = scan_model_directories(MODELS_DIR)

    # 日志配置
    LOGS_DIR = BASE_DIR / 'logs'
    WEBAPP_LOG_PATH = WEBAPP_DIR / 'logs' / 'webapp.log'

    # 脚本路径配置
    SCRIPTS_DIR = BASE_DIR
    QUICK_DAILY_UPDATE_SCRIPT = SCRIPTS_DIR / 'fetch_data' / 'quick_daily_update.py'
    STOCK_SELECTOR_SCRIPT = SCRIPTS_DIR / 'tomorrow_stock_selector.py'
    AI_ENHANCED_SCRIPT = SCRIPTS_DIR / 'ai_enhanced_daily_report.py'
    TRAIN_SCRIPT = SCRIPTS_DIR / 'train_v380_parameterized.py'
    BACKTEST_SCRIPT = SCRIPTS_DIR / 'extensible_backtest_engine.py'

    # 任务配置
    MAX_WORKERS = 4  # 最大并发任务数
    TASK_TIMEOUT = 3600  # 任务超时时间（秒）

    # 缓存配置
    CACHE_ENABLED = True
    CACHE_TTL = 300  # 缓存时间（秒）

    # 分页配置
    ITEMS_PER_PAGE = 50

    # Python解释器路径
    PYTHON_EXECUTABLE = '/Users/yangxu/miniconda3/bin/python3'


class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True


class ProductionConfig(Config):
    """生产环境配置"""
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY')


# 配置字典
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
