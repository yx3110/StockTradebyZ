"""
v3.9 活跃市值特征提取器

新增6个活跃市值相关特征：
- 市场层面: market_active_mv_ratio, market_active_mv_zscore, market_active_mv_trend
- 个股层面: stock_active_mv_rank, stock_relative_liquidity, market_cap_quality_score

活跃市值 = 流通市值 × 换手率
代表每日实际参与交易的市值金额，是衡量市场流动性和资金活跃度的关键指标。

设计目标：
1. 大盘状态影响选股：市场整体活跃度应该影响是否购买某个股票
2. 降低小市值权重：小市值股票被操纵的可能性较高，需要惩罚
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging
import sqlite3
from datetime import datetime, timedelta
from functools import lru_cache

logger = logging.getLogger(__name__)


class ActiveMarketCapFeaturesV39:
    """v3.9活跃市值特征提取器"""

    def __init__(self, db_path: str = "data_adapter/stock_data.db"):
        self.db_path = db_path
        self.feature_names = [
            # 市场层面 (3个)
            'market_active_mv_ratio',      # 市场活跃度比率
            'market_active_mv_zscore',     # 活跃市值Z-score
            'market_active_mv_trend',      # 活跃市值趋势

            # 个股层面 (3个)
            'stock_active_mv_rank',        # 活跃市值排名
            'stock_relative_liquidity',    # 相对流动性
            'market_cap_quality_score',    # 市值质量分 (小市值惩罚)
        ]

        # 缓存市场层面数据（同一交易日只需计算一次）
        self._market_cache = {}
        self._stock_cache = {}

        logger.info(f"✅ v3.9活跃市值特征提取器初始化，共{len(self.feature_names)}个特征")

    def extract_features(self, stock_code: str, trade_date: str) -> Dict[str, float]:
        """
        提取活跃市值特征

        Args:
            stock_code: 股票代码
            trade_date: 交易日期 (YYYYMMDD 或 YYYY-MM-DD)

        Returns:
            特征字典
        """
        try:
            # 统一日期格式
            trade_date = self._normalize_date(trade_date)

            features = {}

            # 1. 市场层面特征（缓存计算）
            market_features = self._get_market_features(trade_date)
            features.update(market_features)

            # 2. 个股层面特征
            stock_features = self._get_stock_features(stock_code, trade_date)
            features.update(stock_features)

            return features

        except Exception as e:
            logger.error(f"{stock_code}: 活跃市值特征提取失败 - {e}")
            return {name: np.nan for name in self.feature_names}

    def _normalize_date(self, date_str: str) -> str:
        """统一日期格式为 YYYY-MM-DD"""
        if len(date_str) == 8:
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        return date_str

    def _get_market_features(self, trade_date: str) -> Dict[str, float]:
        """获取市场层面特征（带缓存）"""
        if trade_date in self._market_cache:
            return self._market_cache[trade_date]

        features = self._calculate_market_features(trade_date)
        self._market_cache[trade_date] = features
        return features

    def _calculate_market_features(self, trade_date: str) -> Dict[str, float]:
        """计算市场层面活跃市值特征"""
        features = {
            'market_active_mv_ratio': 0.0,
            'market_active_mv_zscore': 0.0,
            'market_active_mv_trend': 0.0,
        }

        try:
            conn = sqlite3.connect(self.db_path)

            # 获取最近60个交易日的市场活跃市值数据
            query = """
            WITH market_daily AS (
                SELECT
                    db.trade_date,
                    SUM(db.circ_mv) as total_circ_mv,
                    SUM(db.circ_mv * db.turnover_rate / 100) as total_active_mv
                FROM daily_basic db
                JOIN securities s ON db.security_id = s.id
                WHERE s.type = 'A股'
                    AND db.circ_mv IS NOT NULL
                    AND db.turnover_rate IS NOT NULL
                    AND db.trade_date <= ?
                GROUP BY db.trade_date
                ORDER BY db.trade_date DESC
                LIMIT 60
            )
            SELECT
                trade_date,
                total_circ_mv,
                total_active_mv,
                total_active_mv / total_circ_mv as active_ratio
            FROM market_daily
            ORDER BY trade_date DESC
            """
            df = pd.read_sql(query, conn, params=(trade_date,))

            if df.empty:
                conn.close()
                return features

            # 1. market_active_mv_ratio: 当日活跃度比率
            current_ratio = df.iloc[0]['active_ratio']
            features['market_active_mv_ratio'] = current_ratio

            # 2. market_active_mv_zscore: Z-score标准化
            if len(df) >= 20:
                mean_active_mv = df['total_active_mv'].mean()
                std_active_mv = df['total_active_mv'].std()
                current_active_mv = df.iloc[0]['total_active_mv']

                if std_active_mv > 0:
                    zscore = (current_active_mv - mean_active_mv) / std_active_mv
                    # 限制在 [-3, 3] 范围内
                    features['market_active_mv_zscore'] = np.clip(zscore, -3, 3)

            # 3. market_active_mv_trend: MA5/MA20 - 1
            if len(df) >= 20:
                ma5 = df['total_active_mv'].head(5).mean()
                ma20 = df['total_active_mv'].head(20).mean()
                if ma20 > 0:
                    trend = (ma5 / ma20) - 1
                    # 限制在 [-0.5, 0.5] 范围内
                    features['market_active_mv_trend'] = np.clip(trend, -0.5, 0.5)

            conn.close()

        except Exception as e:
            logger.error(f"市场层面活跃市值特征计算失败: {e}")

        return features

    def _get_stock_features(self, stock_code: str, trade_date: str) -> Dict[str, float]:
        """获取个股层面特征"""
        cache_key = f"{stock_code}_{trade_date}"
        if cache_key in self._stock_cache:
            return self._stock_cache[cache_key]

        features = self._calculate_stock_features(stock_code, trade_date)
        self._stock_cache[cache_key] = features
        return features

    def _calculate_stock_features(self, stock_code: str, trade_date: str) -> Dict[str, float]:
        """计算个股层面活跃市值特征"""
        features = {
            'stock_active_mv_rank': 0.5,       # 中性值
            'stock_relative_liquidity': 1.0,   # 中性值
            'market_cap_quality_score': 0.5,   # 中性值
        }

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 获取股票的 security_id
            cursor.execute("SELECT id FROM securities WHERE code = ?", (stock_code,))
            result = cursor.fetchone()
            if result is None:
                conn.close()
                return features

            security_id = result[0]

            # 获取当日该股票的数据
            stock_query = """
            SELECT
                circ_mv / 10000 as circ_mv_yi,
                turnover_rate,
                circ_mv * turnover_rate / 100 / 10000 as active_mv_yi
            FROM daily_basic
            WHERE security_id = ? AND trade_date = ?
            """
            cursor.execute(stock_query, (security_id, trade_date))
            stock_result = cursor.fetchone()

            if stock_result is None or stock_result[0] is None:
                conn.close()
                return features

            circ_mv_yi, turnover_rate, stock_active_mv = stock_result

            # 1. market_cap_quality_score: 市值质量分 (小市值惩罚)
            features['market_cap_quality_score'] = self._calculate_market_cap_quality(circ_mv_yi)

            # 获取全市场当日活跃市值数据用于排名
            market_query = """
            SELECT
                circ_mv * turnover_rate / 100 / 10000 as active_mv_yi
            FROM daily_basic db
            JOIN securities s ON db.security_id = s.id
            WHERE db.trade_date = ?
                AND s.type = 'A股'
                AND db.circ_mv IS NOT NULL
                AND db.turnover_rate IS NOT NULL
            ORDER BY active_mv_yi DESC
            """
            df_market = pd.read_sql(market_query, conn, params=(trade_date,))

            if not df_market.empty:
                # 2. stock_active_mv_rank: 活跃市值排名百分位
                rank = (df_market['active_mv_yi'] <= stock_active_mv).sum()
                features['stock_active_mv_rank'] = rank / len(df_market)

                # 3. stock_relative_liquidity: 相对流动性
                mean_active_mv = df_market['active_mv_yi'].mean()
                if mean_active_mv > 0:
                    relative = stock_active_mv / mean_active_mv
                    # 使用 tanh 归一化到 0-1
                    features['stock_relative_liquidity'] = np.tanh(relative / 2)

            conn.close()

        except Exception as e:
            logger.error(f"{stock_code}: 个股活跃市值特征计算失败 - {e}")

        return features

    def _calculate_market_cap_quality(self, circ_mv_yi: float) -> float:
        """
        计算市值质量分数（小市值惩罚）

        使用 sigmoid 函数进行平滑惩罚:
        - <20亿: 严重惩罚 (分数 < 0.3)
        - 20-50亿: 中度惩罚 (分数 0.3-0.5)
        - 50-200亿: 中性 (分数 0.5-0.7)
        - 200-500亿: 轻度奖励 (分数 0.7-0.85)
        - >500亿: 高度奖励 (分数 0.85-1.0)

        Args:
            circ_mv_yi: 流通市值(亿元)

        Returns:
            质量分数 0-1
        """
        if circ_mv_yi <= 0:
            return 0.0

        # 参数设计
        center = np.log(50)  # 50亿为中点 (sigmoid中心)
        scale = 0.8          # 控制曲线陡峭程度

        # sigmoid 函数
        score = 1 / (1 + np.exp(-(np.log(circ_mv_yi) - center) / scale))

        return score

    def clear_cache(self):
        """清除缓存"""
        self._market_cache.clear()
        self._stock_cache.clear()
        logger.info("活跃市值特征缓存已清除")

    def get_market_status(self, trade_date: str) -> Dict[str, any]:
        """
        获取市场活跃度状态报告

        Returns:
            包含市场状态解读的字典
        """
        features = self._get_market_features(self._normalize_date(trade_date))

        status = {
            'active_ratio': features['market_active_mv_ratio'],
            'zscore': features['market_active_mv_zscore'],
            'trend': features['market_active_mv_trend'],
        }

        # 解读市场状态
        zscore = features['market_active_mv_zscore']
        if zscore > 1.5:
            status['signal'] = '极度活跃'
            status['recommendation'] = '市场热度高，可适度参与追涨'
        elif zscore > 0.5:
            status['signal'] = '活跃'
            status['recommendation'] = '市场活跃，正常操作'
        elif zscore > -0.5:
            status['signal'] = '平稳'
            status['recommendation'] = '市场正常，按计划操作'
        elif zscore > -1.5:
            status['signal'] = '低迷'
            status['recommendation'] = '市场冷淡，谨慎操作'
        else:
            status['signal'] = '极度低迷'
            status['recommendation'] = '市场低迷，建议观望'

        return status


def batch_extract_features(
    stock_codes: list,
    trade_date: str,
    db_path: str = "data_adapter/stock_data.db"
) -> pd.DataFrame:
    """
    批量提取活跃市值特征

    Args:
        stock_codes: 股票代码列表
        trade_date: 交易日期
        db_path: 数据库路径

    Returns:
        特征DataFrame
    """
    extractor = ActiveMarketCapFeaturesV39(db_path)

    results = []
    for code in stock_codes:
        features = extractor.extract_features(code, trade_date)
        features['code'] = code
        results.append(features)

    df = pd.DataFrame(results)
    df = df.set_index('code')

    return df


if __name__ == "__main__":
    # 测试
    logging.basicConfig(level=logging.INFO)

    extractor = ActiveMarketCapFeaturesV39()

    # 测试提取特征
    print("\n" + "=" * 60)
    print("活跃市值特征测试")
    print("=" * 60)

    test_stocks = ['000001.SZ', '600519.SH', '300750.SZ']
    trade_date = '2025-11-03'

    for code in test_stocks:
        print(f"\n{code}:")
        features = extractor.extract_features(code, trade_date)
        for name, value in features.items():
            if not np.isnan(value):
                print(f"  {name}: {value:.4f}")
            else:
                print(f"  {name}: NaN")

    # 测试市场状态
    print("\n" + "=" * 60)
    print("市场活跃度状态")
    print("=" * 60)
    status = extractor.get_market_status(trade_date)
    print(f"活跃比率: {status['active_ratio']:.4f}")
    print(f"Z-score: {status['zscore']:.2f}")
    print(f"趋势: {status['trend']:.4f}")
    print(f"信号: {status['signal']}")
    print(f"建议: {status['recommendation']}")

    # 测试市值质量分数
    print("\n" + "=" * 60)
    print("市值质量分数测试 (小市值惩罚)")
    print("=" * 60)
    test_caps = [10, 20, 30, 50, 100, 200, 500, 1000]
    for cap in test_caps:
        score = extractor._calculate_market_cap_quality(cap)
        print(f"  {cap:4d}亿 -> 质量分: {score:.3f}")
