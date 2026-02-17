"""
API模块初始化
"""
from .daily_tasks import daily_tasks_bp
from .model_training import model_training_bp
from .backtest import backtest_bp

__all__ = [
    'daily_tasks_bp',
    'model_training_bp',
    'backtest_bp',
]
