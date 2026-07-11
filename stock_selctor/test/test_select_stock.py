#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_stock.py 单元测试
测试股票选择器的主要工具函数
"""

import json
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import sys
import os

# 添加父目录到路径，以便导入模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from select_stock import load_data, load_config, instantiate_selector


class TestLoadData:
    """测试数据加载函数"""
    
    def test_load_data_success(self, tmp_path):
        """测试成功加载数据"""
        # Arrange
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # 创建测试CSV文件
        test_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5),
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [102, 103, 104, 105, 106],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        test_data.to_csv(data_dir / "000001.csv", index=False)
        test_data.to_csv(data_dir / "000002.csv", index=False)
        
        codes = ["000001", "000002"]
        
        # Act
        result = load_data(data_dir, codes)
        
        # Assert
        assert len(result) == 2
        assert "000001" in result
        assert "000002" in result
        assert isinstance(result["000001"], pd.DataFrame)
        assert isinstance(result["000002"], pd.DataFrame)
        assert len(result["000001"]) == 5
        assert "date" in result["000001"].columns
    
    def test_load_data_missing_file(self, tmp_path, caplog):
        """测试加载不存在的文件"""
        # Arrange
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # 只创建一个文件
        test_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=3),
            'close': [100, 101, 102]
        })
        test_data.to_csv(data_dir / "000001.csv", index=False)
        
        codes = ["000001", "000002"]  # 000002不存在
        
        # Act
        result = load_data(data_dir, codes)
        
        # Assert
        assert len(result) == 1
        assert "000001" in result
        assert "000002" not in result
        assert "000002.csv 不存在，跳过" in caplog.text
    
    def test_load_data_empty_codes(self, tmp_path):
        """测试空代码列表"""
        # Arrange
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Act
        result = load_data(data_dir, [])
        
        # Assert
        assert result == {}


class TestLoadConfig:
    """测试配置加载函数"""
    
    def test_load_config_single_object(self, tmp_path):
        """测试加载单个对象配置"""
        # Arrange
        config_data = {
            "class": "BBIKDJSelector",
            "alias": "测试选择器",
            "params": {"j_threshold": 1}
        }
        
        config_file = tmp_path / "strategy_configs.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f)
        
        # Act
        result = load_config(config_file)
        
        # Assert
        assert len(result) == 1
        assert result[0]["class"] == "BBIKDJSelector"
        assert result[0]["alias"] == "测试选择器"
    
    def test_load_config_array(self, tmp_path):
        """测试加载数组配置"""
        # Arrange
        config_data = [
            {
                "class": "BBIKDJSelector",
                "alias": "选择器1",
                "params": {"j_threshold": 1}
            },
            {
                "class": "PeakKDJSelector", 
                "alias": "选择器2",
                "params": {"j_threshold": 2}
            }
        ]
        
        config_file = tmp_path / "strategy_configs.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f)
        
        # Act
        result = load_config(config_file)
        
        # Assert
        assert len(result) == 2
        assert result[0]["class"] == "BBIKDJSelector"
        assert result[1]["class"] == "PeakKDJSelector"
    
    def test_load_config_with_selectors_key(self, tmp_path):
        """测试加载带selectors键的配置"""
        # Arrange
        config_data = {
            "selectors": [
                {
                    "class": "BBIKDJSelector",
                    "alias": "选择器1",
                    "params": {"j_threshold": 1}
                }
            ]
        }
        
        config_file = tmp_path / "strategy_configs.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f)
        
        # Act
        result = load_config(config_file)
        
        # Assert
        assert len(result) == 1
        assert result[0]["class"] == "BBIKDJSelector"
    
    def test_load_config_file_not_exists(self, tmp_path):
        """测试配置文件不存在"""
        # Arrange
        config_file = tmp_path / "nonexistent.json"
        
        # Act & Assert
        with pytest.raises(SystemExit):
            load_config(config_file)
    
    def test_load_config_empty_selectors(self, tmp_path):
        """测试空的selectors配置"""
        # Arrange
        config_data = {"selectors": []}
        
        config_file = tmp_path / "strategy_configs.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f)
        
        # Act & Assert
        with pytest.raises(SystemExit):
            load_config(config_file)


class TestInstantiateSelector:
    """测试选择器实例化函数"""
    
    @patch('select_stock.importlib.import_module')
    def test_instantiate_selector_success(self, mock_import_module):
        """测试成功实例化选择器"""
        # Arrange
        mock_module = Mock()
        mock_class = Mock()
        mock_module.BBIKDJSelector = mock_class
        mock_import_module.return_value = mock_module
        
        config = {
            "class": "BBIKDJSelector",
            "alias": "测试选择器",
            "params": {"j_threshold": 1}
        }
        
        # Act
        alias, instance = instantiate_selector(config)
        
        # Assert
        assert alias == "测试选择器"
        mock_class.assert_called_once_with(j_threshold=1)
    
    def test_instantiate_selector_missing_class(self):
        """测试缺少class字段"""
        # Arrange
        config = {
            "alias": "测试选择器",
            "params": {"j_threshold": 1}
        }
        
        # Act & Assert
        with pytest.raises(ValueError, match="缺少 class 字段"):
            instantiate_selector(config)
    
    @patch('select_stock.importlib.import_module')
    def test_instantiate_selector_module_not_found(self, mock_import_module):
        """测试模块不存在"""
        # Arrange
        mock_import_module.side_effect = ModuleNotFoundError("No module named 'Selector'")
        
        config = {
            "class": "BBIKDJSelector",
            "params": {"j_threshold": 1}
        }
        
        # Act & Assert
        with pytest.raises(ImportError, match="无法加载 Selector.BBIKDJSelector"):
            instantiate_selector(config)
    
    @patch('select_stock.importlib.import_module')
    def test_instantiate_selector_class_not_found(self, mock_import_module):
        """测试类不存在"""
        # Arrange
        mock_module = Mock()
        # 模拟AttributeError
        mock_import_module.return_value = mock_module
        # 删除该属性，使 getattr 抛出 AttributeError
        # (注: 给 Mock 属性赋 None 不会让 getattr 抛错, 只会返回 None)
        del mock_module.BBIKDJSelector
        
        config = {
            "class": "BBIKDJSelector",
            "params": {"j_threshold": 1}
        }
        
        # Act & Assert
        with pytest.raises(ImportError, match="无法加载 Selector.BBIKDJSelector"):
            instantiate_selector(config)
    
    @patch('select_stock.importlib.import_module')
    def test_instantiate_selector_no_alias(self, mock_import_module):
        """测试没有alias字段时使用class名"""
        # Arrange
        mock_module = Mock()
        mock_class = Mock()
        mock_module.BBIKDJSelector = mock_class
        mock_import_module.return_value = mock_module
        
        config = {
            "class": "BBIKDJSelector",
            "params": {"j_threshold": 1}
        }
        
        # Act
        alias, instance = instantiate_selector(config)
        
        # Assert
        assert alias == "BBIKDJSelector"
        mock_class.assert_called_once_with(j_threshold=1)
    
    @patch('select_stock.importlib.import_module')
    def test_instantiate_selector_no_params(self, mock_import_module):
        """测试没有params字段"""
        # Arrange
        mock_module = Mock()
        mock_class = Mock()
        mock_module.BBIKDJSelector = mock_class
        mock_import_module.return_value = mock_module
        
        config = {
            "class": "BBIKDJSelector",
            "alias": "测试选择器"
        }
        
        # Act
        alias, instance = instantiate_selector(config)
        
        # Assert
        assert alias == "测试选择器"
        mock_class.assert_called_once_with()


if __name__ == "__main__":
    pytest.main([__file__]) 