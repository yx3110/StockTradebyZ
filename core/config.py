"""
集中式配置管理

所有项目路径、数据库路径、API密钥获取统一在此管理。
各模块应通过 `from core.config import PROJECT_ROOT, DB_PATH, get_tushare_token` 来使用。
"""

import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================
# 路径常量
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# ============================================================
# .env 加载 (密钥已迁移至 .env, 2026-07-11)
# 模块加载时执行, 使 get_tushare_token / get_anthropic_api_key
# 的环境变量优先级路径直接受益; 不覆盖已存在的环境变量。
# ============================================================
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / '.env')
except ImportError:  # python-dotenv 未安装时静默降级 (仍可用真实环境变量/config.json)
    logger.debug("python-dotenv 未安装, 跳过 .env 加载")

DB_PATH = PROJECT_ROOT / 'data_adapter' / 'stock_data.db'

# 模型目录
TRAINED_MODELS_DIR = PROJECT_ROOT / 'ml_models' / 'trained_models'
V39_MODEL_DIR = TRAINED_MODELS_DIR / 'v39'
V43_MODEL_DIR = TRAINED_MODELS_DIR / 'v43'
V44_MODEL_DIR = TRAINED_MODELS_DIR / 'v44'
V395_MODEL_DIR = TRAINED_MODELS_DIR / 'v395'

# 模型目录 (更多版本)
V400_MODEL_DIR = TRAINED_MODELS_DIR / 'v400'
V500_MODEL_DIR = TRAINED_MODELS_DIR / 'v500'

# 报告目录
REPORTS_DIR = PROJECT_ROOT / 'reports'
BACKTEST_REPORTS_DIR = REPORTS_DIR / 'backtest'

# 日志目录
LOGS_DIR = PROJECT_ROOT / 'logs'

# 配置文件路径
CONFIG_JSON_PATH = PROJECT_ROOT / 'config.json'
STRATEGY_CONFIGS_PATH = PROJECT_ROOT / 'strategy_configs.json'

# ============================================================
# 市场常量
# ============================================================
MARKET_INDICES = {
    '000001.SH': '上证指数',
    '399001.SZ': '深证成指',
    '399006.SZ': '创业板指',
    '000688.SH': '科创50',
    '000016.SH': '上证50',
    '000300.SH': '沪深300',
    '000905.SH': '中证500',
    '000852.SH': '中证1000',
    '932000.CSI': '中证2000',
    '000985.SH': '中证全指',
}

# ============================================================
# 配置文件加载
# ============================================================
_config_cache = None


def load_config(config_path: str = None) -> dict:
    """加载项目配置文件 (带缓存)

    优先级:
    1. 指定路径
    2. PROJECT_ROOT/config.json
    """
    global _config_cache
    if _config_cache is not None and config_path is None:
        return _config_cache

    if config_path is None:
        config_path = PROJECT_ROOT / 'config.json'
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(f"配置文件不存在: {config_path}")
        return {}

    with open(config_path, 'r') as f:
        config = json.load(f)

    if config_path == PROJECT_ROOT / 'config.json':
        _config_cache = config

    return config


# ============================================================
# API 密钥获取 (环境变量优先)
# ============================================================
def get_tushare_token() -> str:
    """获取 Tushare API Token

    优先级:
    1. 环境变量 TUSHARE_TOKEN
    2. config.json 中的 tushare.token
    """
    token = os.environ.get('TUSHARE_TOKEN')
    if token:
        return token

    config = load_config()
    token = config.get('tushare', {}).get('token')
    if token:
        return token

    raise ValueError(
        "未找到 Tushare Token。请设置环境变量 TUSHARE_TOKEN 或在 config.json 中配置 tushare.token"
    )


def get_anthropic_api_key() -> str:
    """获取 Anthropic API Key

    优先级:
    1. 环境变量 ANTHROPIC_API_KEY
    2. config.json 中的 anthropic.api_key
    """
    key = os.environ.get('ANTHROPIC_API_KEY')
    if key:
        return key

    config = load_config()
    key = config.get('anthropic', {}).get('api_key')
    if key:
        return key

    raise ValueError(
        "未找到 Anthropic API Key。请设置环境变量 ANTHROPIC_API_KEY 或在 config.json 中配置 anthropic.api_key"
    )


def get_db_path() -> Path:
    """获取数据库路径，支持环境变量覆盖"""
    custom = os.environ.get('STOCK_DB_PATH')
    if custom:
        return Path(custom)
    return DB_PATH


def get_report_dir(version: str) -> Path:
    """获取指定版本的报告目录

    Args:
        version: 版本标识，如 'v3.9', 'v3.95', 'v4.3' 等

    Returns:
        报告目录 Path，如 reports/daily_selection_v3.9/
    """
    return REPORTS_DIR / f'daily_selection_{version}'
