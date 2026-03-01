"""
v3.9 市场特征提取器

新增8个市场特征：
- 市场情绪: 上涨家数/下跌家数比率, 涨停板数量, 北向资金净流入, 融资余额变化
- 板块效应: 所属板块强度排名, 行业资金流入排名, 概念热度指数, 市场关注度
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging
import sqlite3
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MarketFeaturesV39:
    """v3.9市场特征提取器"""

    def __init__(self, db_path: str = "data_adapter/stock_data.db"):
        self.db_path = db_path
        self.feature_names = [
            # 市场情绪
            'advance_decline_ratio',
            'limit_up_count',
            'northbound_net_inflow',
            'margin_balance_change',

            # 板块效应
            'sector_strength_rank',
            'industry_fund_flow_rank',
            'concept_heat_index',
            'market_attention_score'
        ]
        logger.info(f"✅ v3.9市场特征提取器初始化，共{len(self.feature_names)}个特征")

    def extract_features(self, stock_code: str, trade_date: str) -> Dict[str, float]:
        """
        提取市场特征

        Args:
            stock_code: 股票代码
            trade_date: 交易日期 (YYYYMMDD)

        Returns:
            特征字典
        """
        try:
            features = {}

            # 市场情绪特征
            features.update(self._calculate_market_sentiment(trade_date))

            # 板块效应特征
            features.update(self._calculate_sector_effects(stock_code, trade_date))

            return features

        except Exception as e:
            logger.error(f"{stock_code}: 市场特征提取失败 - {e}")
            return {name: np.nan for name in self.feature_names}

    def _calculate_market_sentiment(self, trade_date: str) -> Dict[str, float]:
        """计算市场情绪特征"""
        features = {}

        try:
            conn = sqlite3.connect(self.db_path)

            # 1. 上涨家数/下跌家数比率
            advance_decline_query = """
            SELECT
                SUM(CASE WHEN price_change_pct > 0 THEN 1 ELSE 0 END) as advance,
                SUM(CASE WHEN price_change_pct < 0 THEN 1 ELSE 0 END) as decline
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE dq.trade_date = ? AND s.type = 'A股'
            """
            cursor = conn.cursor()
            cursor.execute(advance_decline_query, (trade_date,))
            ad_result = cursor.fetchone()

            if ad_result and ad_result[1] > 0:
                features['advance_decline_ratio'] = ad_result[0] / ad_result[1]
            else:
                features['advance_decline_ratio'] = 1.0  # 中性值

            # 2. 涨停板数量（归一化到0-1）
            limit_up_query = """
            SELECT COUNT(*) as limit_up_count
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE dq.trade_date = ? AND dq.is_limit_up = 1 AND s.type = 'A股'
            """
            cursor.execute(limit_up_query, (trade_date,))
            limit_up_result = cursor.fetchone()

            if limit_up_result:
                # 归一化：涨停板数量 / 100（假设最多100个涨停板）
                features['limit_up_count'] = min(limit_up_result[0] / 100.0, 1.0)
            else:
                features['limit_up_count'] = 0.0

            # 3. 北向资金净流入（简化版：使用沪深300ETF的资金流向估算）
            # 注意：这里简化处理，实际应该使用北向资金数据
            northbound_query = """
            SELECT
                SUM(dq.volume * dq.close * dq.price_change_pct) as estimated_flow
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE dq.trade_date = ? AND s.code IN ('510300.SH', '159919.SZ')
            """
            cursor.execute(northbound_query, (trade_date,))
            northbound_result = cursor.fetchone()

            if northbound_result and northbound_result[0] is not None:
                # 归一化到-1到1之间
                features['northbound_net_inflow'] = np.tanh(northbound_result[0] / 1e10)
            else:
                features['northbound_net_inflow'] = 0.0

            # 4. 融资余额变化（简化版：使用市场整体成交量变化估算）
            margin_query = """
            SELECT
                AVG(dq1.volume) as volume_today,
                AVG(dq2.volume) as volume_yesterday
            FROM daily_quotes dq1
            JOIN securities s ON dq1.security_id = s.id
            LEFT JOIN daily_quotes dq2 ON dq1.security_id = dq2.security_id
                AND dq2.trade_date = (
                    SELECT MAX(trade_date)
                    FROM daily_quotes
                    WHERE trade_date < ?
                )
            WHERE dq1.trade_date = ? AND s.type = 'A股'
            """
            cursor.execute(margin_query, (trade_date, trade_date))
            margin_result = cursor.fetchone()

            if margin_result and margin_result[0] and margin_result[1]:
                change_pct = (margin_result[0] - margin_result[1]) / margin_result[1]
                features['margin_balance_change'] = change_pct
            else:
                features['margin_balance_change'] = 0.0

            conn.close()

        except Exception as e:
            logger.error(f"市场情绪特征计算失败: {e}")
            for key in ['advance_decline_ratio', 'limit_up_count', 'northbound_net_inflow', 'margin_balance_change']:
                features[key] = np.nan

        return features

    def _calculate_sector_effects(self, stock_code: str, trade_date: str) -> Dict[str, float]:
        """计算板块效应特征（简化版，不依赖行业分类）"""
        features = {}

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 获取股票security_id
            cursor.execute("SELECT id FROM securities WHERE code = ?", (stock_code,))
            security_id_result = cursor.fetchone()

            if security_id_result is None:
                for key in ['sector_strength_rank', 'industry_fund_flow_rank',
                           'concept_heat_index', 'market_attention_score']:
                    features[key] = 0.5  # 中性值
                conn.close()
                return features

            security_id = security_id_result[0]

            # 5. 所属板块强度排名（简化：使用股票相对市场的表现）
            # 计算股票涨跌幅在全市场的排名
            sector_strength_query = """
            SELECT
                COUNT(*) as total_stocks,
                SUM(CASE WHEN dq2.price_change_pct > dq1.price_change_pct THEN 1 ELSE 0 END) as worse_count
            FROM daily_quotes dq1
            CROSS JOIN daily_quotes dq2
            JOIN securities s ON dq2.security_id = s.id
            WHERE dq1.security_id = ? AND dq1.trade_date = ?
                AND dq2.trade_date = ? AND s.type = 'A股'
            """
            cursor.execute(sector_strength_query, (security_id, trade_date, trade_date))
            strength_result = cursor.fetchone()

            if strength_result and strength_result[0] > 0:
                features['sector_strength_rank'] = 1 - (strength_result[1] / strength_result[0])
            else:
                features['sector_strength_rank'] = 0.5

            # 6. 行业资金流入排名（简化：使用量比指标）
            # 量比 = 当日成交量 / 近5日平均成交量
            volume_ratio_query = """
            SELECT
                dq1.volume * 1.0 / AVG(dq2.volume) as volume_ratio
            FROM daily_quotes dq1
            JOIN daily_quotes dq2 ON dq1.security_id = dq2.security_id
            WHERE dq1.security_id = ? AND dq1.trade_date = ?
                AND dq2.trade_date < ? AND dq2.trade_date >= date(?, '-5 days')
            """
            cursor.execute(volume_ratio_query, (security_id, trade_date, trade_date, trade_date))
            volume_ratio_result = cursor.fetchone()

            if volume_ratio_result and volume_ratio_result[0]:
                # 归一化到0-1，假设量比>2为满分
                features['industry_fund_flow_rank'] = min(volume_ratio_result[0] / 2.0, 1.0)
            else:
                features['industry_fund_flow_rank'] = 0.5

            # 7. 概念热度指数（简化：全市场涨停板占比）
            concept_heat_query = """
            SELECT
                COUNT(CASE WHEN is_limit_up = 1 THEN 1 END) * 1.0 / COUNT(*) as heat_index
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE dq.trade_date = ? AND s.type = 'A股'
            """
            cursor.execute(concept_heat_query, (trade_date,))
            heat_result = cursor.fetchone()

            if heat_result and heat_result[0] is not None:
                features['concept_heat_index'] = heat_result[0]
            else:
                features['concept_heat_index'] = 0.0

            # 8. 市场关注度评分（使用换手率相对市场平均）
            attention_query = """
            SELECT
                db1.turnover_rate,
                AVG(db2.turnover_rate) as market_avg_turnover
            FROM daily_basic db1
            CROSS JOIN daily_basic db2
            WHERE db1.security_id = ? AND db1.trade_date = ?
                AND db2.trade_date = ?
            GROUP BY db1.turnover_rate
            """
            cursor.execute(attention_query, (security_id, trade_date, trade_date))
            attention_result = cursor.fetchone()

            if attention_result and attention_result[0] and attention_result[1]:
                features['market_attention_score'] = attention_result[0] / (attention_result[1] + 1e-6)
            else:
                features['market_attention_score'] = 1.0

            conn.close()

        except Exception as e:
            logger.error(f"板块效应特征计算失败: {e}")
            for key in ['sector_strength_rank', 'industry_fund_flow_rank',
                       'concept_heat_index', 'market_attention_score']:
                features[key] = 0.5  # 失败时返回中性值

        return features


if __name__ == "__main__":
    # 简单测试
    logging.basicConfig(level=logging.INFO)

    extractor = MarketFeaturesV39()

    # 测试提取特征（需要数据库中有数据）
    features = extractor.extract_features('000001.SZ', '20251031')

    print("\n=== v3.9 市场特征测试 ===")
    for name, value in features.items():
        if not np.isnan(value):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: NaN")

    print(f"\n总计: {len(features)}个特征")
    print(f"有效特征: {sum(1 for v in features.values() if not np.isnan(v))}个")
