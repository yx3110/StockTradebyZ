# 股票选择器单元测试

本目录包含股票选择器项目的完整单元测试套件。

## 📁 文件结构

```
test/
├── test_select_stock.py          # select_stock.py 工具函数测试
├── test_selector_indicators.py   # 技术指标计算函数测试
├── test_selector_classes.py      # 选择器类测试
├── pytest.ini                   # pytest配置文件
├── requirements-test.txt         # 测试依赖
├── run_tests.py                 # 测试运行脚本
└── README.md                    # 本文件
```

## 🚀 快速开始

### 1. 安装测试依赖

```bash
pip install -r test/requirements-test.txt
```

### 2. 运行所有测试

```bash
# 方法1: 使用测试脚本
python test/run_tests.py

# 方法2: 直接使用pytest
cd stock_selctor
python -m pytest test/ -v --cov=. --cov-report=html
```

### 3. 运行特定测试

```bash
# 运行特定测试文件
python -m pytest test/test_select_stock.py -v

# 运行特定测试类
python -m pytest test/test_selector_indicators.py::TestComputeKDJ -v

# 运行特定测试方法
python -m pytest test/test_selector_indicators.py::TestComputeKDJ::test_compute_kdj_normal_data -v
```

## 📊 测试覆盖范围

### 1. select_stock.py 测试
- ✅ `load_data()` - 数据加载函数
- ✅ `load_config()` - 配置加载函数  
- ✅ `instantiate_selector()` - 选择器实例化函数

### 2. 技术指标函数测试
- ✅ `compute_kdj()` - KDJ指标计算
- ✅ `compute_bbi()` - BBI指标计算
- ✅ `compute_rsv()` - RSV指标计算
- ✅ `compute_dif()` - DIF指标计算
- ✅ `bbi_deriv_uptrend()` - BBI上升趋势判断

### 3. 选择器类测试
- ✅ `BBIKDJSelector` - BBI+KDJ选择器
- ✅ `PeakKDJSelector` - 峰值KDJ选择器
- ✅ `BBIShortLongSelector` - BBI短期长期选择器
- ✅ `BreakoutVolumeKDJSelector` - 突破成交量KDJ选择器

## 🧪 测试类型

### 单元测试 (Unit Tests)
- **正常情况测试**: 验证函数在正常输入下的行为
- **边界条件测试**: 测试极值和边界情况
- **异常情况测试**: 验证错误处理和异常抛出
- **参数验证测试**: 确保参数验证正确工作

### 集成测试 (Integration Tests)
- **选择器集成测试**: 测试所有选择器使用相同数据
- **参数验证集成**: 测试选择器参数验证

## 📈 测试报告

运行测试后会生成以下报告：

1. **控制台输出**: 实时显示测试结果
2. **HTML覆盖率报告**: `htmlcov/index.html`
3. **XML覆盖率报告**: `coverage.xml`
4. **终端覆盖率摘要**: 显示代码覆盖率统计

## 🔧 测试配置

### pytest.ini 配置
- 测试文件模式: `test_*.py`, `*_test.py`
- 测试类模式: `Test*`
- 测试函数模式: `test_*`
- 覆盖率报告: HTML, XML, 终端
- 超时设置: 300秒

### 覆盖率配置
- 排除测试文件本身
- 排除调试代码
- 排除异常处理代码

## 🎯 测试最佳实践

### 1. 测试结构 (AAA模式)
```python
def test_function_name(self):
    # Arrange - 准备测试数据
    input_data = create_test_data()
    
    # Act - 执行被测试的函数
    result = function_under_test(input_data)
    
    # Assert - 验证结果
    assert result == expected_value
```

### 2. 测试数据管理
- 使用 `@pytest.fixture` 创建可重用的测试数据
- 使用 `tmp_path` 处理临时文件
- 使用 `Mock` 对象模拟外部依赖

### 3. 测试命名
- 测试函数名: `test_功能名_测试场景`
- 测试类名: `Test被测试类名`
- 描述性注释说明测试目的

## 🐛 故障排除

### 常见问题

1. **导入错误**
   ```bash
   # 确保在正确的目录运行
   cd stock_selctor
   python -m pytest test/
   ```

2. **依赖缺失**
   ```bash
   pip install -r test/requirements-test.txt
   ```

3. **覆盖率报告不显示**
   ```bash
   # 确保安装了pytest-cov
   pip install pytest-cov
   ```

### 调试测试
```bash
# 详细输出
python -m pytest test/ -v -s

# 只运行失败的测试
python -m pytest test/ --lf

# 在失败时停止
python -m pytest test/ -x
```

## 📝 添加新测试

### 1. 创建测试文件
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新功能测试
"""

import pytest
from module_under_test import function_under_test

class TestNewFunction:
    """测试新功能"""
    
    def test_new_function_normal_case(self):
        """测试正常情况"""
        # Arrange
        input_data = "test"
        
        # Act
        result = function_under_test(input_data)
        
        # Assert
        assert result == "expected"
```

### 2. 运行新测试
```bash
python -m pytest test/test_new_function.py -v
```

## 📊 测试统计

- **总测试数**: 50+
- **测试文件**: 3个
- **测试类**: 8个
- **测试方法**: 50+
- **覆盖率目标**: >80%

## 🤝 贡献指南

1. 为新功能添加测试
2. 确保所有测试通过
3. 保持测试覆盖率
4. 遵循测试命名规范
5. 添加适当的文档注释

---

**注意**: 运行测试前请确保已安装所有必要的依赖包。 