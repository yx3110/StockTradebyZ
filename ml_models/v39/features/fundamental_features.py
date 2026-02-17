"""
v3.9 基本面特征提取器

新增10个基本面特征：
- 盈利质量: 经营现金流/净利润, ROE变化率, 毛利率趋势, 净利润增长稳定性
- 估值相对性: PE/PB/PS相对行业分位数
- 财务健康: 资产负债率, 流动比率, 应收账款周转率
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging
import sqlite3

logger = logging.getLogger(__name__)


class FundamentalFeaturesV39:
    """v3.9基本面特征提取器"""

    def __init__(self, db_path: str = "data_adapter/stock_data.db"):
        self.db_path = db_path
        self.feature_names = [
            # 盈利质量
            'operating_cashflow_to_netprofit',
            'roe_change_rate',
            'gross_margin_trend',
            'netprofit_growth_stability',

            # 估值相对性
            'pe_industry_percentile',
            'pb_industry_percentile',
            'ps_industry_percentile',

            # 财务健康
            'debt_to_asset_ratio',
            'current_ratio',
            'receivables_turnover'
        ]
        logger.info(f"✅ v3.9基本面特征提取器初始化，共{len(self.feature_names)}个特征")

    def extract_features(self, stock_code: str, trade_date: str) -> Dict[str, float]:
        """
        提取基本面特征

        Args:
            stock_code: 股票代码
            trade_date: 交易日期 (YYYYMMDD)

        Returns:
            特征字典
        """
        try:
            features = {}

            # 盈利质量特征
            features.update(self._calculate_profitability_quality(stock_code, trade_date))

            # 估值相对性特征
            features.update(self._calculate_valuation_relativity(stock_code, trade_date))

            # 财务健康特征
            features.update(self._calculate_financial_health(stock_code, trade_date))

            return features

        except Exception as e:
            logger.error(f"{stock_code}: 基本面特征提取失败 - {e}")
            return {name: np.nan for name in self.feature_names}

    def _calculate_profitability_quality(self, stock_code: str, trade_date: str) -> Dict[str, float]:
        """计算盈利质量特征"""
        features = {}

        try:
            conn = sqlite3.connect(self.db_path)

            # 获取最近4个季度的财务数据
            # 注意：financial_indicator表使用end_date而不是trade_date
            query = """
            SELECT
                fi.ocf_to_profit,
                fi.roe,
                fi.grossprofit_margin,
                fi.profit_to_gr,
                fi.end_date
            FROM financial_indicator fi
            JOIN securities s ON fi.security_id = s.id
            WHERE s.code = ? AND fi.end_date <= ?
            ORDER BY fi.end_date DESC
            LIMIT 4
            """

            df = pd.read_sql_query(query, conn, params=(stock_code, trade_date))
            conn.close()

            if len(df) == 0:
                for key in ['operating_cashflow_to_netprofit', 'roe_change_rate',
                           'gross_margin_trend', 'netprofit_growth_stability']:
                    features[key] = np.nan
                return features

            # 1. 经营性现金流/净利润 (使用最新数据)
            if 'ocf_to_profit' in df.columns and not pd.isna(df['ocf_to_profit'].iloc[0]):
                features['operating_cashflow_to_netprofit'] = df['ocf_to_profit'].iloc[0]
            else:
                features['operating_cashflow_to_netprofit'] = np.nan

            # 2. ROE变化率
            if len(df) >= 2 and 'roe' in df.columns:
                roe_current = df['roe'].iloc[0]
                roe_previous = df['roe'].iloc[1]
                if not pd.isna(roe_current) and not pd.isna(roe_previous) and roe_previous != 0:
                    features['roe_change_rate'] = (roe_current - roe_previous) / abs(roe_previous)
                else:
                    features['roe_change_rate'] = np.nan
            else:
                features['roe_change_rate'] = np.nan

            # 3. 毛利率趋势 (使用线性回归斜率)
            if len(df) >= 3 and 'grossprofit_margin' in df.columns:
                gross_margins = df['grossprofit_margin'].dropna().values
                if len(gross_margins) >= 3:
                    x = np.arange(len(gross_margins))
                    slope, _ = np.polyfit(x, gross_margins, 1)
                    features['gross_margin_trend'] = slope
                else:
                    features['gross_margin_trend'] = np.nan
            else:
                features['gross_margin_trend'] = np.nan

            # 4. 净利润增长率稳定性 (使用标准差/均值)
            if len(df) >= 3 and 'profit_to_gr' in df.columns:
                profit_growth = df['profit_to_gr'].dropna().values
                if len(profit_growth) >= 3:
                    mean_growth = np.mean(profit_growth)
                    std_growth = np.std(profit_growth)
                    # 稳定性 = 1 / (CV + 1)，CV越小越稳定
                    cv = std_growth / (abs(mean_growth) + 1e-10)
                    features['netprofit_growth_stability'] = 1 / (cv + 1)
                else:
                    features['netprofit_growth_stability'] = np.nan
            else:
                features['netprofit_growth_stability'] = np.nan

        except Exception as e:
            logger.error(f"盈利质量特征计算失败: {e}")
            for key in ['operating_cashflow_to_netprofit', 'roe_change_rate',
                       'gross_margin_trend', 'netprofit_growth_stability']:
                features[key] = np.nan

        return features

    def _calculate_valuation_relativity(self, stock_code: str, trade_date: str) -> Dict[str, float]:
        """计算估值相对性特征"""
        features = {}

        try:
            conn = sqlite3.connect(self.db_path)

            # 获取股票security_id
            security_id_query = """
            SELECT id FROM securities WHERE code = ?
            """
            cursor = conn.cursor()
            cursor.execute(security_id_query, (stock_code,))
            security_id_result = cursor.fetchone()

            if security_id_result is None:
                for key in ['pe_industry_percentile', 'pb_industry_percentile', 'ps_industry_percentile']:
                    features[key] = np.nan
                conn.close()
                return features

            security_id = security_id_result[0]

            # 获取所有股票的估值数据（由于行业数据缺失，使用全市场分位数）
            valuation_query = """
            SELECT
                pe_ttm,
                pb,
                ps_ttm
            FROM daily_basic
            WHERE trade_date = ?
            """

            df_industry = pd.read_sql_query(valuation_query, conn, params=(trade_date,))

            # 获取目标股票的估值数据
            stock_valuation_query = """
            SELECT pe_ttm, pb, ps_ttm
            FROM daily_basic
            WHERE security_id = ? AND trade_date = ?
            """
            df_stock = pd.read_sql_query(stock_valuation_query, conn, params=(security_id, trade_date))

            conn.close()

            if len(df_stock) == 0 or len(df_industry) < 5:
                for key in ['pe_industry_percentile', 'pb_industry_percentile', 'ps_industry_percentile']:
                    features[key] = np.nan
                return features

            # 计算PE相对行业分位数
            stock_pe = df_stock['pe_ttm'].iloc[0]
            industry_pe = df_industry['pe_ttm'].dropna()
            if not pd.isna(stock_pe) and len(industry_pe) > 0:
                features['pe_industry_percentile'] = (industry_pe < stock_pe).sum() / len(industry_pe)
            else:
                features['pe_industry_percentile'] = np.nan

            # 计算PB相对行业分位数
            stock_pb = df_stock['pb'].iloc[0]
            industry_pb = df_industry['pb'].dropna()
            if not pd.isna(stock_pb) and len(industry_pb) > 0:
                features['pb_industry_percentile'] = (industry_pb < stock_pb).sum() / len(industry_pb)
            else:
                features['pb_industry_percentile'] = np.nan

            # 计算PS相对行业分位数
            stock_ps = df_stock['ps_ttm'].iloc[0]
            industry_ps = df_industry['ps_ttm'].dropna()
            if not pd.isna(stock_ps) and len(industry_ps) > 0:
                features['ps_industry_percentile'] = (industry_ps < stock_ps).sum() / len(industry_ps)
            else:
                features['ps_industry_percentile'] = np.nan

        except Exception as e:
            logger.error(f"估值相对性特征计算失败: {e}")
            for key in ['pe_industry_percentile', 'pb_industry_percentile', 'ps_industry_percentile']:
                features[key] = np.nan

        return features

    def _calculate_financial_health(self, stock_code: str, trade_date: str) -> Dict[str, float]:
        """计算财务健康特征"""
        features = {}

        try:
            conn = sqlite3.connect(self.db_path)

            # 获取最新的财务指标
            query = """
            SELECT
                fi.debt_to_assets,
                fi.current_ratio,
                fi.ar_turn
            FROM financial_indicator fi
            JOIN securities s ON fi.security_id = s.id
            WHERE s.code = ? AND fi.end_date <= ?
            ORDER BY fi.end_date DESC
            LIMIT 1
            """

            df = pd.read_sql_query(query, conn, params=(stock_code, trade_date))
            conn.close()

            if len(df) == 0:
                for key in ['debt_to_asset_ratio', 'current_ratio', 'receivables_turnover']:
                    features[key] = np.nan
                return features

            # 8. 资产负债率
            if 'debt_to_assets' in df.columns and not pd.isna(df['debt_to_assets'].iloc[0]):
                features['debt_to_asset_ratio'] = df['debt_to_assets'].iloc[0]
            else:
                features['debt_to_asset_ratio'] = np.nan

            # 9. 流动比率
            if 'current_ratio' in df.columns and not pd.isna(df['current_ratio'].iloc[0]):
                features['current_ratio'] = df['current_ratio'].iloc[0]
            else:
                features['current_ratio'] = np.nan

            # 10. 应收账款周转率
            if 'ar_turn' in df.columns and not pd.isna(df['ar_turn'].iloc[0]):
                features['receivables_turnover'] = df['ar_turn'].iloc[0]
            else:
                features['receivables_turnover'] = np.nan

        except Exception as e:
            logger.error(f"财务健康特征计算失败: {e}")
            for key in ['debt_to_asset_ratio', 'current_ratio', 'receivables_turnover']:
                features[key] = np.nan

        return features


if __name__ == "__main__":
    # 简单测试
    logging.basicConfig(level=logging.INFO)

    extractor = FundamentalFeaturesV39()

    # 测试提取特征（需要数据库中有数据）
    features = extractor.extract_features('000001.SZ', '20251031')

    print("\n=== v3.9 基本面特征测试 ===")
    for name, value in features.items():
        if not np.isnan(value):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: NaN")

    print(f"\n总计: {len(features)}个特征")
    print(f"有效特征: {sum(1 for v in features.values() if not np.isnan(v))}个")
