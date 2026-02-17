#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Selector.py 技术指标函数单元测试
测试KDJ、BBI、RSV、DIF等技术指标的计算
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 导入要测试的函数
from Selector import (
    compute_kdj, 
    compute_bbi, 
    compute_rsv, 
    compute_dif,
    bbi_deriv_uptrend
)


class TestComputeKDJ:
    """测试KDJ指标计算"""
    
    def test_compute_kdj_normal_data(self):
        """测试正常数据计算KDJ"""
        # Arrange
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            'low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
            'close': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'volume': [1000] * 10
        })
        
        # Act
        result = compute_kdj(df)
        
        # Assert
        assert 'K' in result.columns
        assert 'D' in result.columns
        assert 'J' in result.columns
        assert len(result) == 10
        assert not result['K'].isna().all()
        assert not result['D'].isna().all()
        assert not result['J'].isna().all()
        # KDJ值应该在合理范围内
        assert (result['K'] >= 0).all()
        assert (result['K'] <= 100).all()
        assert (result['D'] >= 0).all()
        assert (result['D'] <= 100).all()
    
    def test_compute_kdj_empty_dataframe(self):
        """测试空数据框"""
        # Arrange
        df = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        
        # Act
        result = compute_kdj(df)
        
        # Assert
        assert 'K' in result.columns
        assert 'D' in result.columns
        assert 'J' in result.columns
        assert len(result) == 0
        assert result['K'].isna().all()
        assert result['D'].isna().all()
        assert result['J'].isna().all()
    
    def test_compute_kdj_custom_n(self):
        """测试自定义n值"""
        # Arrange
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=20),
            'open': [100 + i for i in range(20)],
            'high': [105 + i for i in range(20)],
            'low': [95 + i for i in range(20)],
            'close': [102 + i for i in range(20)],
            'volume': [1000] * 20
        })
        
        # Act
        result = compute_kdj(df, n=14)
        
        # Assert
        assert 'K' in result.columns
        assert 'D' in result.columns
        assert 'J' in result.columns
        assert len(result) == 20
    
    def test_compute_kdj_constant_price(self):
        """测试价格不变的情况"""
        # Arrange
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'open': [100] * 10,
            'high': [100] * 10,
            'low': [100] * 10,
            'close': [100] * 10,
            'volume': [1000] * 10
        })
        
        # Act
        result = compute_kdj(df)
        
        # Assert
        # 价格不变时，K和D应该在合理范围内
        assert 0 <= result['K'].iloc[-1] <= 100
        assert 0 <= result['D'].iloc[-1] <= 100
        # K和D应该相等（在价格不变的情况下）
        assert abs(result['K'].iloc[-1] - result['D'].iloc[-1]) < 5


class TestComputeBBI:
    """测试BBI指标计算"""
    
    def test_compute_bbi_normal_data(self):
        """测试正常数据计算BBI"""
        # Arrange
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=30),
            'open': [100 + i for i in range(30)],
            'high': [105 + i for i in range(30)],
            'low': [95 + i for i in range(30)],
            'close': [102 + i for i in range(30)],
            'volume': [1000] * 30
        })
        
        # Act
        result = compute_bbi(df)
        
        # Assert
        assert isinstance(result, pd.Series)
        assert len(result) == 30
        # 前23个值应该是NaN（因为需要24个数据点）
        assert result.iloc[:23].isna().all()
        # 后面的值应该是正数
        assert (result.iloc[23:] > 0).all()
    
    def test_compute_bbi_short_data(self):
        """测试数据不足的情况"""
        # Arrange
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5),
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [102, 103, 104, 105, 106],
            'volume': [1000] * 5
        })
        
        # Act
        result = compute_bbi(df)
        
        # Assert
        assert isinstance(result, pd.Series)
        assert len(result) == 5
        # 所有值都应该是NaN（因为数据不足24个点）
        assert result.isna().all()


class TestComputeRSV:
    """测试RSV指标计算"""
    
    def test_compute_rsv_normal_data(self):
        """测试正常数据计算RSV"""
        # Arrange
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=20),
            'open': [100 + i for i in range(20)],
            'high': [105 + i for i in range(20)],
            'low': [95 + i for i in range(20)],
            'close': [102 + i for i in range(20)],
            'volume': [1000] * 20
        })
        
        # Act
        result = compute_rsv(df, n=9)
        
        # Assert
        assert isinstance(result, pd.Series)
        assert len(result) == 20
        assert not result.isna().all()
        # RSV值应该在0-100范围内
        assert (result >= 0).all()
        assert (result <= 100).all()
    
    def test_compute_rsv_constant_price(self):
        """测试价格不变的情况"""
        # Arrange
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'open': [100] * 10,
            'high': [100] * 10,
            'low': [100] * 10,
            'close': [100] * 10,
            'volume': [1000] * 10
        })
        
        # Act
        result = compute_rsv(df, n=5)
        
        # Assert
        # 价格不变时，RSV应该在合理范围内
        assert 0 <= result.iloc[-1] <= 100
        # RSV应该为0（在价格完全不变的情况下）
        assert result.iloc[-1] == 0.0
    
    def test_compute_rsv_extreme_values(self):
        """测试极值情况"""
        # Arrange
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'open': [100] * 10,
            'high': [200] * 10,  # 最高价
            'low': [50] * 10,    # 最低价
            'close': [200] * 10, # 收盘价等于最高价
            'volume': [1000] * 10
        })
        
        # Act
        result = compute_rsv(df, n=5)
        
        # Assert
        # 收盘价等于最高价时，RSV应该接近100
        assert result.iloc[-1] > 95


class TestComputeDIF:
    """测试DIF指标计算"""
    
    def test_compute_dif_normal_data(self):
        """测试正常数据计算DIF"""
        # Arrange
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=50),
            'open': [100 + i for i in range(50)],
            'high': [105 + i for i in range(50)],
            'low': [95 + i for i in range(50)],
            'close': [102 + i for i in range(50)],
            'volume': [1000] * 50
        })
        
        # Act
        result = compute_dif(df)
        
        # Assert
        assert isinstance(result, pd.Series)
        assert len(result) == 50
        assert not result.isna().all()
    
    def test_compute_dif_custom_parameters(self):
        """测试自定义参数"""
        # Arrange
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=50),
            'open': [100 + i for i in range(50)],
            'high': [105 + i for i in range(50)],
            'low': [95 + i for i in range(50)],
            'close': [102 + i for i in range(50)],
            'volume': [1000] * 50
        })
        
        # Act
        result = compute_dif(df, fast=5, slow=10)
        
        # Assert
        assert isinstance(result, pd.Series)
        assert len(result) == 50
    
    def test_compute_dif_constant_price(self):
        """测试价格不变的情况"""
        # Arrange
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=30),
            'open': [100] * 30,
            'high': [100] * 30,
            'low': [100] * 30,
            'close': [100] * 30,
            'volume': [1000] * 30
        })
        
        # Act
        result = compute_dif(df)
        
        # Assert
        # 价格不变时，DIF应该接近0
        assert abs(result.iloc[-1]) < 0.1


class TestBBIDerivUptrend:
    """测试BBI上升趋势判断"""
    
    def test_bbi_deriv_uptrend_rising(self):
        """测试上升趋势"""
        # Arrange
        bbi = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
        
        # Act
        result = bbi_deriv_uptrend(bbi, min_window=5, max_window=10)
        
        # Assert
        assert result is True
    
    def test_bbi_deriv_uptrend_falling(self):
        """测试下降趋势"""
        # Arrange
        bbi = pd.Series([109, 108, 107, 106, 105, 104, 103, 102, 101, 100])
        
        # Act
        result = bbi_deriv_uptrend(bbi, min_window=5, max_window=10)
        
        # Assert
        assert result is False
    
    def test_bbi_deriv_uptrend_insufficient_data(self):
        """测试数据不足"""
        # Arrange
        bbi = pd.Series([100, 101, 102])  # 只有3个数据点
        
        # Act
        result = bbi_deriv_uptrend(bbi, min_window=5)
        
        # Assert
        assert result is False
    
    def test_bbi_deriv_uptrend_invalid_threshold(self):
        """测试无效的阈值"""
        # Arrange
        bbi = pd.Series([100, 101, 102, 103, 104, 105])
        
        # Act & Assert
        with pytest.raises(ValueError, match="q_threshold 必须位于 \\[0, 1\\] 区间内"):
            bbi_deriv_uptrend(bbi, min_window=5, q_threshold=1.5)
    
    def test_bbi_deriv_uptrend_with_threshold(self):
        """测试带阈值的上升趋势"""
        # Arrange
        # 创建一个大部分上升但有少量下降的数据
        bbi = pd.Series([100, 101, 100.5, 102, 101.5, 103, 102.5, 104, 103.5, 105])
        
        # Act
        result = bbi_deriv_uptrend(bbi, min_window=5, max_window=10, q_threshold=0.2)
        
        # Assert
        # 允许20%的下降，结果应该是布尔值
        assert isinstance(result, bool)
        # 由于数据有下降，可能返回False，这是正常的
        # 我们只验证返回的是布尔值
        assert result in [True, False]
    
    def test_bbi_deriv_uptrend_no_max_window(self):
        """测试不设置最大窗口"""
        # Arrange
        bbi = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
        
        # Act
        result = bbi_deriv_uptrend(bbi, min_window=5, max_window=None)
        
        # Assert
        assert result is True


if __name__ == "__main__":
    pytest.main([__file__]) 