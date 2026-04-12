"""Signal Trust 全部常量。改阈值从这里改。"""

# 样本池过滤
PRED_THRESHOLD = 0.01  # pred_10d > 0.01 才纳入样本

# 可信度最小样本数
MIN_SAMPLES = 10

# 颜色标签阈值
GREEN_HIT_MIN = 0.55
GREEN_BIAS_MIN = -0.02
GREEN_REALIZE_MIN = 0.40
RED_HIT_MAX = 0.45
RED_BIAS_MAX = -0.03
RED_REALIZE_MAX = 0.20

# 标签字符串
TAG_GREEN = "🟢可信"
TAG_YELLOW = "🟡存疑"
TAG_RED = "🔴高风险"
TAG_NO_DATA = "⚪数据不足"

# ④ 高预测兑现率中"实际兑现"门槛
REALIZE_ACTUAL_THRESHOLD = 0.02  # actual_10d > 0.02 算兑现

# 持有期：10 个交易日
HOLD_DAYS = 10

# 市值分档（单位：万元，与 daily_basic.circ_mv 一致）
MARKET_CAP_BUCKETS = [
    (0, 30_0000, "微盘"),
    (30_0000, 100_0000, "小盘"),
    (100_0000, 500_0000, "中盘"),
    (500_0000, float("inf"), "大盘"),
]

# 版本优先级：同 (code, trade_date) 在多个目录出现时取最高优先级
# 新版本上线时加到列表顶部
VERSION_PRIORITY = [
    "ng106", "ng1.0.6",
    "ng101", "ng1.0.1",
    "ng100", "ng1.0.0",
    "v4901", "v4.9.0.1", "v490",
    "v475", "v473",
    "v3.9", "v39",
]

# 数据库路径（可被测试覆盖）
DEFAULT_DB_PATH = "data_adapter/stock_data.db"
