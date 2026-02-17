#!/usr/bin/env python3
"""
评分系统配置模块
Scoring System Configuration

基于实际选股表现优化的配置参数
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import json
import os

@dataclass
class ScoringConfig:
    """评分系统配置类"""
    
    # 因子权重配置 (基于3949样本分析优化)
    momentum_factor_weight: float = 0.40      # 动量因子40% (提升识别强势股)
    mean_reversion_weight: float = 0.25       # 均值回归25% (价值修复机会)
    volume_breakout_weight: float = 0.20      # 量价突破20% (突破确认)
    relative_performance_weight: float = 0.10 # 相对强度10% (相对表现)
    stability_factor_weight: float = 0.05     # 稳定性5% (风险控制)
    
    # 评分阈值配置 (基于70-75分区间最佳表现优化)
    buy_threshold: float = 75.0               # 买入门槛75分 (从80分降低)
    cautious_buy_threshold: float = 70.0      # 谨慎买入70分 (对应最佳区间)
    watch_threshold: float = 60.0             # 观望60分
    avoid_threshold: float = 0.0              # 回避区间
    
    # 目标评分分布 (优化股票分布到高效区间)
    target_score_ranges: Dict[str, Tuple[float, float]] = None
    
    def __post_init__(self):
        if self.target_score_ranges is None:
            self.target_score_ranges = {
                'high_confidence': (75, 85),      # 减少80+区间
                'moderate_confidence': (70, 75),  # 扩大最佳表现区间
                'low_confidence': (60, 70),       # 观望区间
                'avoid': (0, 60)                  # 回避区间
            }
    
    def validate(self) -> bool:
        """验证配置参数有效性"""
        # 检查权重之和
        total_weight = (
            self.momentum_factor_weight +
            self.mean_reversion_weight +
            self.volume_breakout_weight +
            self.relative_performance_weight +
            self.stability_factor_weight
        )
        
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"权重之和应为1.0，实际为{total_weight:.3f}")
        
        # 检查阈值顺序
        thresholds = [
            self.buy_threshold,
            self.cautious_buy_threshold,
            self.watch_threshold,
            self.avoid_threshold
        ]
        
        if thresholds != sorted(thresholds, reverse=True):
            raise ValueError("评分阈值应按降序排列")
        
        return True
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'momentum_factor_weight': self.momentum_factor_weight,
            'mean_reversion_weight': self.mean_reversion_weight,
            'volume_breakout_weight': self.volume_breakout_weight,
            'relative_performance_weight': self.relative_performance_weight,
            'stability_factor_weight': self.stability_factor_weight,
            'buy_threshold': self.buy_threshold,
            'cautious_buy_threshold': self.cautious_buy_threshold,
            'watch_threshold': self.watch_threshold,
            'avoid_threshold': self.avoid_threshold,
            'target_score_ranges': self.target_score_ranges
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'ScoringConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'ScoringConfig':
        """从JSON文件加载配置"""
        if not os.path.exists(config_path):
            return cls()  # 使用默认配置
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def save_to_file(self, config_path: str):
        """保存配置到JSON文件"""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

# 默认配置实例
DEFAULT_CONFIG = ScoringConfig()

# 市场环境适配配置
class MarketAdaptiveConfig:
    """市场环境自适应配置"""
    
    @staticmethod
    def get_bull_market_config() -> ScoringConfig:
        """牛市配置 - 加强动量和技术因子"""
        config = ScoringConfig()
        config.momentum_factor_weight = 0.45      # 动量加强
        config.volume_breakout_weight = 0.25      # 突破加强
        config.mean_reversion_weight = 0.20       # 均值回归降低
        config.relative_performance_weight = 0.08
        config.stability_factor_weight = 0.02     # 稳定性降低
        
        # 牛市中提高买入门槛
        config.buy_threshold = 78.0
        config.cautious_buy_threshold = 73.0
        
        return config
    
    @staticmethod
    def get_bear_market_config() -> ScoringConfig:
        """熊市配置 - 加强价值和稳定性因子"""
        config = ScoringConfig()
        config.momentum_factor_weight = 0.30      # 动量降低
        config.mean_reversion_weight = 0.35       # 价值加强
        config.volume_breakout_weight = 0.15      # 突破降低
        config.relative_performance_weight = 0.12
        config.stability_factor_weight = 0.08     # 稳定性加强
        
        # 熊市中降低买入门槛，寻找价值
        config.buy_threshold = 70.0
        config.cautious_buy_threshold = 65.0
        
        return config
    
    @staticmethod
    def get_sideways_market_config() -> ScoringConfig:
        """震荡市配置 - 均衡配置"""
        return ScoringConfig()  # 使用默认均衡配置

def get_optimized_config(market_state: str = "normal") -> ScoringConfig:
    """
    根据市场状态获取优化配置
    
    Args:
        market_state: 市场状态 ('bull', 'bear', 'sideways', 'normal')
    """
    if market_state == "bull":
        return MarketAdaptiveConfig.get_bull_market_config()
    elif market_state == "bear":
        return MarketAdaptiveConfig.get_bear_market_config()
    elif market_state == "sideways":
        return MarketAdaptiveConfig.get_sideways_market_config()
    else:
        return DEFAULT_CONFIG