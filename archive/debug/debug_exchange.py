#!/usr/bin/env python3
import sys
import traceback

def test_base_exchange():
    """测试基础Exchange类"""
    try:
        from qlib.backtest.exchange import Exchange
        
        print("测试基础Exchange类...")
        
        # 测试基础Exchange初始化
        exc = Exchange(
            freq='day',
            start_time='2024-01-01',
            end_time='2024-12-31',
            codes=['000001.SZ']
        )
        
        print("✅ 基础Exchange初始化成功")
        print(f"配置属性: {hasattr(exc, '_config')}")
        if hasattr(exc, '_config'):
            print(f"配置内容: {type(exc._config)}")
            
        return True
        
    except Exception as e:
        print(f"❌ 基础Exchange初始化失败: {e}")
        traceback.print_exc()
        return False

def test_chinese_exchange_minimal():
    """测试最小ChineseAShareExchange"""
    try:
        sys.path.append('.')
        from qlib_integration.backtest.chinese_exchange import ChineseAShareExchange
        
        print("\n测试最小ChineseAShareExchange...")
        
        # 不传递任何自定义参数
        exc = ChineseAShareExchange(
            start_time='2024-01-01',
            end_time='2024-12-31',
            freq='day',
            codes=['000001.SZ']
        )
        
        print("✅ ChineseAShareExchange初始化成功")
        return True
        
    except Exception as e:
        print(f"❌ ChineseAShareExchange初始化失败: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 调试Exchange初始化问题...")
    
    result1 = test_base_exchange()
    result2 = test_chinese_exchange_minimal()
    
    if result1 and result2:
        print("\n🎉 所有测试通过!")
    else:
        print("\n⚠️ 存在问题需要修复")