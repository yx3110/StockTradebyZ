#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试运行脚本
运行所有单元测试并生成报告
"""

import subprocess
import sys
import os
from pathlib import Path

def run_tests():
    """运行所有测试"""
    print("🚀 开始运行股票选择器单元测试...")
    
    # 获取当前脚本所在目录
    test_dir = Path(__file__).parent
    project_dir = test_dir.parent
    
    # 切换到项目目录
    os.chdir(project_dir)
    
    # 运行测试命令
    cmd = [
        sys.executable, "-m", "pytest",
        "test/",
        "-v",
        "--tb=short",
        "--cov=.",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--cov-report=xml"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("📊 测试结果:")
        print(result.stdout)
        
        if result.stderr:
            print("⚠️  警告信息:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ 所有测试通过!")
        else:
            print("❌ 部分测试失败!")
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ 运行测试时出错: {e}")
        return False

def install_test_dependencies():
    """安装测试依赖"""
    print("📦 安装测试依赖...")
    
    test_dir = Path(__file__).parent
    requirements_file = test_dir / "requirements-test.txt"
    
    if requirements_file.exists():
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ 测试依赖安装成功!")
                return True
            else:
                print("❌ 测试依赖安装失败!")
                print(result.stderr)
                return False
        except Exception as e:
            print(f"❌ 安装依赖时出错: {e}")
            return False
    else:
        print("⚠️  未找到测试依赖文件")
        return True

def main():
    """主函数"""
    print("🧪 股票选择器测试套件")
    print("=" * 50)
    
    # 安装依赖
    if not install_test_dependencies():
        print("❌ 依赖安装失败，退出测试")
        sys.exit(1)
    
    # 运行测试
    success = run_tests()
    
    if success:
        print("\n🎉 测试完成! 请查看 htmlcov/ 目录下的覆盖率报告")
        sys.exit(0)
    else:
        print("\n💥 测试失败!")
        sys.exit(1)

if __name__ == "__main__":
    main() 