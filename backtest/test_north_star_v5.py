"""北极星V5单元测试"""
import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFactorReturns:
    """因子收益构建测试"""

    def test_build_factor_returns_columns(self):
        """构建结果应包含4因子列"""
        from backtest.factor_returns import build_factor_returns
        df = build_factor_returns('2025-01-01', '2025-01-31')
        assert 'MKT' in df.columns
        assert 'SMB' in df.columns
        assert 'HML' in df.columns
        assert 'UMD' in df.columns
        assert len(df) > 0

    def test_factor_returns_no_extreme_values(self):
        """因子日收益应在合理范围 (-15%, +15%)"""
        from backtest.factor_returns import build_factor_returns
        df = build_factor_returns('2025-01-01', '2025-03-31')
        for col in ['MKT', 'SMB', 'HML', 'UMD']:
            assert df[col].abs().max() < 0.15, f"{col} has extreme value"

    def test_factor_returns_low_correlation(self):
        """因子间相关性应 < 0.5"""
        from backtest.factor_returns import build_factor_returns
        df = build_factor_returns('2024-01-01', '2025-01-01')
        corr = df[['SMB', 'HML', 'UMD']].corr()
        for i in range(3):
            for j in range(i+1, 3):
                assert abs(corr.iloc[i, j]) < 0.5, \
                    f"High correlation: {corr.columns[i]} vs {corr.columns[j]} = {corr.iloc[i,j]:.3f}"
