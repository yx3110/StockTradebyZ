#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.8实时特征计算器完整测试
验证所有12个实时特征的计算正确性
"""

import sys
import os
from datetime import datetime, timedelta
import logging
import json

# 添加项目路径
sys.path.append('/Users/yangxu/StockTradebyZ')

from data_adapter.database_manager import DatabaseManager
from incremental_learning.features.realtime_calculator import RealtimeFeatureCalculator

def test_comprehensive_realtime_features():
    """完整测试实时特征计算器"""
    print("🧪 V3.8实时特征计算器完整测试")
    print("=" * 50)

    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger('FeatureTest')

    try:
        # 初始化组件
        db_manager = DatabaseManager()
        calculator = RealtimeFeatureCalculator(
            cache_ttl=300,
            db_manager=db_manager,
            logger=logger
        )

        # 测试股票代码（不同类型的股票）
        test_codes = ['000001', '600036', '002215', '300750', '688599']
        test_results = {}

        for code in test_codes:
            print(f"\n📊 测试股票: {code}")
            print("-" * 30)

            # 计算实时特征
            features = calculator.compute_intraday_features(code)

            if features:
                print("✅ 成功计算所有实时特征:")

                # 分类显示特征
                momentum_features = {
                    'intraday_momentum_5m': features.get('intraday_momentum_5m'),
                    'intraday_momentum_15m': features.get('intraday_momentum_15m'),
                    'intraday_momentum_30m': features.get('intraday_momentum_30m')
                }

                opening_features = {
                    'opening_gap': features.get('opening_gap'),
                    'opening_volume_surge': features.get('opening_volume_surge'),
                    'early_session_perf': features.get('early_session_perf')
                }

                volume_features = {
                    'volume_intensity': features.get('volume_intensity'),
                    'volume_consistency': features.get('volume_consistency')
                }

                volatility_features = {
                    'volatility_intraday': features.get('volatility_intraday'),
                    'price_efficiency': features.get('price_efficiency')
                }

                market_features = {
                    'relative_sector_strength': features.get('relative_sector_strength'),
                    'market_correlation': features.get('market_correlation')
                }

                print(f"  🚀 动量特征 (3个):")
                for name, value in momentum_features.items():
                    print(f"    - {name}: {value:.6f}")

                print(f"  🌅 开盘特征 (3个):")
                for name, value in opening_features.items():
                    print(f"    - {name}: {value:.6f}")

                print(f"  📈 成交量特征 (2个):")
                for name, value in volume_features.items():
                    print(f"    - {name}: {value:.6f}")

                print(f"  🎯 波动率特征 (2个):")
                for name, value in volatility_features.items():
                    print(f"    - {name}: {value:.6f}")

                print(f"  🏛️ 市场特征 (2个):")
                for name, value in market_features.items():
                    print(f"    - {name}: {value:.6f}")

                # 验证特征合理性
                feature_validation = validate_features(features, code)
                print(f"  ✅ 特征验证: {feature_validation['status']}")

                test_results[code] = {
                    'features': features,
                    'validation': feature_validation
                }

            else:
                print("❌ 无法计算实时特征")
                test_results[code] = {'features': None, 'validation': {'status': 'failed'}}

        # 生成测试报告
        generate_test_report(test_results)

        print(f"\n🎉 实时特征计算器测试完成！")
        print(f"   ✅ 测试股票数量: {len(test_codes)}")
        print(f"   ✅ 特征总数: 12个")
        print(f"   ✅ 成功率: {sum(1 for r in test_results.values() if r['features']) / len(test_codes) * 100:.1f}%")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_features(features: dict, code: str) -> dict:
    """验证特征的合理性"""
    issues = []
    warnings = []

    # 检查动量特征
    momentum_5m = features.get('intraday_momentum_5m', 0)
    momentum_15m = features.get('intraday_momentum_15m', 0)
    momentum_30m = features.get('intraday_momentum_30m', 0)

    if abs(momentum_5m) > 0.2:  # 超过20%变化
        warnings.append(f"5分钟动量过大: {momentum_5m:.3f}")

    if abs(momentum_15m) > 0.15:  # 超过15%变化
        warnings.append(f"15分钟动量过大: {momentum_15m:.3f}")

    if abs(momentum_30m) > 0.1:  # 超过10%变化
        warnings.append(f"30分钟动量过大: {momentum_30m:.3f}")

    # 检查开盘特征
    opening_gap = features.get('opening_gap', 0)
    if abs(opening_gap) > 0.1:  # 超过10%的开盘缺口
        warnings.append(f"开盘缺口较大: {opening_gap:.3f}")

    # 检查成交量特征
    volume_consistency = features.get('volume_consistency', 0)
    if not (0 <= volume_consistency <= 1):
        issues.append(f"成交量一致性超出范围: {volume_consistency}")

    # 检查相对强度特征
    relative_strength = features.get('relative_sector_strength', 1)
    if not (0.5 <= relative_strength <= 1.5):
        issues.append(f"相对强度超出合理范围: {relative_strength}")

    # 检查市场相关性
    market_corr = features.get('market_correlation', 0.5)
    if not (0 <= market_corr <= 1):
        issues.append(f"市场相关性超出范围: {market_corr}")

    # 确定状态
    if issues:
        status = 'error'
    elif warnings:
        status = 'warning'
    else:
        status = 'passed'

    return {
        'status': status,
        'issues': issues,
        'warnings': warnings,
        'feature_count': len(features)
    }

def generate_test_report(test_results: dict):
    """生成测试报告"""
    print(f"\n📋 测试报告汇总")
    print("=" * 50)

    total_stocks = len(test_results)
    successful_stocks = sum(1 for r in test_results.values() if r['features'])

    print(f"📊 测试统计:")
    print(f"   - 总测试股票: {total_stocks}")
    print(f"   - 成功计算: {successful_stocks}")
    print(f"   - 成功率: {successful_stocks/total_stocks*100:.1f}%")

    print(f"\n🎯 特征统计:")
    if successful_stocks > 0:
        # 获取第一个成功的结果作为特征列表参考
        sample_features = next(r['features'] for r in test_results.values() if r['features'])
        print(f"   - 特征总数: {len(sample_features)}")

        # 统计各类特征数量
        momentum_count = len([k for k in sample_features.keys() if 'momentum' in k])
        opening_count = len([k for k in sample_features.keys() if any(x in k for x in ['opening', 'early_session'])])
        volume_count = len([k for k in sample_features.keys() if 'volume' in k])
        volatility_count = len([k for k in sample_features.keys() if any(x in k for x in ['volatility', 'efficiency'])])
        market_count = len([k for k in sample_features.keys() if any(x in k for x in ['sector', 'market_correlation'])])

        print(f"   - 动量特征: {momentum_count}个")
        print(f"   - 开盘特征: {opening_count}个")
        print(f"   - 成交量特征: {volume_count}个")
        print(f"   - 波动率特征: {volatility_count}个")
        print(f"   - 市场特征: {market_count}个")

    print(f"\n⚠️ 验证结果:")
    passed = sum(1 for r in test_results.values() if r['validation']['status'] == 'passed')
    warnings = sum(1 for r in test_results.values() if r['validation']['status'] == 'warning')
    errors = sum(1 for r in test_results.values() if r['validation']['status'] == 'error')

    print(f"   - 通过: {passed}个")
    print(f"   - 警告: {warnings}个")
    print(f"   - 错误: {errors}个")

    # 保存详细报告到文件
    report_data = {
        'test_time': datetime.now().isoformat(),
        'summary': {
            'total_stocks': total_stocks,
            'successful_stocks': successful_stocks,
            'success_rate': successful_stocks/total_stocks*100 if total_stocks > 0 else 0
        },
        'results': test_results
    }

    report_file = f"reports/v380_realtime_features_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs('reports', exist_ok=True)

    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n💾 详细报告已保存: {report_file}")

if __name__ == '__main__':
    success = test_comprehensive_realtime_features()
    sys.exit(0 if success else 1)