
# V3.7快速修复配置
# 当检测到维度不匹配时的处理策略

DIMENSION_MISMATCH_CONFIG = {
    "expected_features": 53,
    "current_features": 48,
    "missing_features": 5,
    "degradation_mode": True,
    "adjusted_threshold": 40,  # 从默认80降低到40
    "confidence_penalty": 0.8  # 置信度惩罚系数
}

def get_adjusted_threshold():
    """获取调整后的选择阈值"""
    return DIMENSION_MISMATCH_CONFIG["adjusted_threshold"]

def is_degradation_mode():
    """检查是否处于维度不匹配降级模式"""
    return DIMENSION_MISMATCH_CONFIG["degradation_mode"]
