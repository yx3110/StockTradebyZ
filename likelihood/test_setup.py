#!/usr/bin/env python3
"""
测试基础设施是否正常工作
Test if the infrastructure is working properly
"""

import sys
from pathlib import Path
import json
import yaml

# 添加父目录到系统路径
sys.path.append(str(Path(__file__).parent.parent))

from data_adapter.database_manager import DatabaseManager


def test_project_structure():
    """测试项目结构是否完整"""
    print("=" * 60)
    print("测试项目结构")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    required_dirs = [
        'algorithms',
        'backtest', 
        'data_preprocessing',
        'reports',
        'tests',
        'configs',
        'utils'
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        exists = dir_path.exists()
        status = "✅" if exists else "❌"
        print(f"{status} {dir_name}: {dir_path}")
        if not exists:
            all_exist = False
    
    return all_exist


def test_config_file():
    """测试配置文件"""
    print("\n" + "=" * 60)
    print("测试配置文件")
    print("=" * 60)
    
    config_path = Path(__file__).parent / 'configs' / 'default_config.yaml'
    
    if config_path.exists():
        print(f"✅ 配置文件存在: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            print(f"✅ 配置文件格式正确")
            print(f"   - 包含 {len(config)} 个顶级配置项")
            print(f"   - 相似度算法: {config['similarity']['algorithms'].keys()}")
            print(f"   - 回测时间窗口: {config['backtest']['evaluation_horizons']}")
            return True
            
        except Exception as e:
            print(f"❌ 读取配置文件失败: {str(e)}")
            return False
    else:
        print(f"❌ 配置文件不存在: {config_path}")
        return False


def test_database_connection():
    """测试数据库连接"""
    print("\n" + "=" * 60)
    print("测试数据库连接")
    print("=" * 60)
    
    # 尝试多个可能的数据库路径
    possible_paths = [
        Path(__file__).parent.parent / 'data_adapter' / 'stock_data.db',
        Path(__file__).parent.parent / 'stock_data.db',
        Path(__file__).parent / 'stock_data.db'
    ]
    
    db_found = False
    for db_path in possible_paths:
        if db_path.exists():
            print(f"✅ 找到数据库文件: {db_path}")
            
            try:
                db = DatabaseManager(str(db_path))
                stats = db.get_database_stats()
                
                print(f"✅ 数据库连接成功")
                print(f"   - 证券总数: {stats.get('total_securities', 0)}")
                print(f"   - 股票数量: {stats.get('stocks', 0)}")
                print(f"   - ETF数量: {stats.get('etfs', 0)}")
                print(f"   - 数据表数: {stats.get('total_tables', 0)}")
                
                # 测试查询
                import sqlite3
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                
                # 查询最新数据日期
                cursor.execute("""
                    SELECT MAX(trade_date) as latest_date
                    FROM daily_quotes
                """)
                latest_date = cursor.fetchone()[0]
                print(f"   - 最新数据日期: {latest_date}")
                
                # 查询示例股票
                cursor.execute("""
                    SELECT code, name 
                    FROM securities 
                    WHERE code IN ('000001.SZ', '600000.SH')
                    LIMIT 2
                """)
                samples = cursor.fetchall()
                if samples:
                    print(f"   - 示例股票:")
                    for code, name in samples:
                        print(f"     • {code}: {name}")
                
                conn.close()
                db_found = True
                break
                
            except Exception as e:
                print(f"❌ 数据库连接失败: {str(e)}")
    
    if not db_found:
        print(f"❌ 未找到数据库文件")
        print(f"   检查过的路径: {[str(p) for p in possible_paths]}")
    
    return db_found


def test_data_loader():
    """测试数据加载器"""
    print("\n" + "=" * 60)
    print("测试数据加载器")
    print("=" * 60)
    
    try:
        from data_preprocessing.data_loader import DataLoader
        print(f"✅ 成功导入DataLoader")
        
        # 创建加载器实例
        loader = DataLoader()
        print(f"✅ 成功创建DataLoader实例")
        
        # 测试加载数据
        try:
            df = loader.load_stock_data('000001', '2025-08-01', '2025-08-08')
            print(f"✅ 成功加载股票数据")
            print(f"   - 数据形状: {df.shape}")
            print(f"   - 数据列数: {len(df.columns)}")
            print(f"   - 日期范围: {df.index[0]} 到 {df.index[-1]}")
            
            # 测试数据标准化
            df_norm = loader.normalize_prices(df, method='log_return')
            print(f"✅ 成功执行价格标准化")
            
            return True
            
        except Exception as e:
            print(f"⚠️  加载数据失败（可能是日期范围问题）: {str(e)}")
            return False
            
    except ImportError as e:
        print(f"❌ 导入DataLoader失败: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ 测试DataLoader失败: {str(e)}")
        return False


def test_main_program():
    """测试主程序"""
    print("\n" + "=" * 60)
    print("测试主程序")
    print("=" * 60)
    
    main_path = Path(__file__).parent / 'main.py'
    
    if main_path.exists():
        print(f"✅ 主程序文件存在: {main_path}")
        
        # 测试帮助信息
        import subprocess
        result = subprocess.run(
            [sys.executable, str(main_path), '--help'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ 主程序可以正常运行")
            print(f"   - 支持的参数:")
            for line in result.stdout.split('\n'):
                if line.strip().startswith('--'):
                    print(f"     • {line.strip()[:50]}")
            return True
        else:
            print(f"❌ 主程序运行失败")
            return False
    else:
        print(f"❌ 主程序文件不存在")
        return False


def main():
    """主测试函数"""
    print("\n" + "🚀 股票相似度回测系统 - 基础设施测试")
    print("=" * 60)
    
    tests = [
        ("项目结构", test_project_structure),
        ("配置文件", test_config_file),
        ("数据库连接", test_database_connection),
        ("数据加载器", test_data_loader),
        ("主程序", test_main_program)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ 测试 {test_name} 出现异常: {str(e)}")
            results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    all_pass = True
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{status}: {test_name}")
        if not success:
            all_pass = False
    
    print("\n" + "=" * 60)
    if all_pass:
        print("🎉 所有测试通过！系统基础设施正常。")
        print("\n下一步:")
        print("1. 运行单只股票分析: python main.py --stock 000001.SZ --date 2025-08-08")
        print("2. 检查数据库连接: python main.py --check-db")
        print("3. 查看帮助信息: python main.py --help")
    else:
        print("⚠️  部分测试失败，请检查相关配置和依赖。")
        print("\n可能的解决方案:")
        print("1. 确保数据库文件 stock_data.db 存在")
        print("2. 安装必要的依赖: pip install -r requirements.txt")
        print("3. 检查配置文件 configs/default_config.yaml")
    
    print("=" * 60)
    
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())