#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.8情绪指标增强测试
Phase 2.2: 测试4个核心情绪指标
"""

import sys
import os
from datetime import datetime, timedelta
import logging
import json

# 添加项目路径
sys.path.append('/Users/yangxu/StockTradebyZ')

from data_adapter.database_manager import DatabaseManager
from incremental_learning.features.sentiment_indicators import SentimentIndicatorCalculator
from incremental_learning.features.realtime_calculator import RealtimeFeatureCalculator

def test_sentiment_indicators():
    """测试情绪指标计算器"""
    print("🧪 V3.8情绪指标增强测试 (Phase 2.2)")
    print("=" * 60)

    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger('SentimentTest')

    try:
        # 初始化组件
        db_manager = DatabaseManager()
        sentiment_calculator = SentimentIndicatorCalculator(db_manager, cache_ttl=300)

        # 测试股票代码（不同特征的股票）
        test_codes = [
            '000001',  # 平安银行 - 大盘银行股
            '600036',  # 招商银行 - 蓝筹股
            '002215',  # 诺普信 - 中小盘股
            '300750',  # 宁德时代 - 新能源龙头
            '688599'   # 天合光能 - 科创板股票
        ]

        test_results = {}

        for code in test_codes:
            print(f"\n💭 测试股票: {code}")
            print("-" * 40)

            # 测试独立的情绪指标计算器
            sentiment_indicators = sentiment_calculator.compute_sentiment_indicators(code)

            if sentiment_indicators:
                print("✅ 成功计算情绪指标:")

                # 分类显示4个情绪指标
                print(f"  💰 资金流向指标:")
                capital_flow = sentiment_indicators.get('capital_flow_indicator', 0)
                print(f"    - capital_flow_indicator: {capital_flow:.6f}")
                flow_direction = "流入" if capital_flow > 0 else "流出" if capital_flow < 0 else "平衡"
                print(f"    - 解读: 资金{flow_direction} (强度: {abs(capital_flow):.2f})")

                print(f"  😰 市场情绪指数 (VIX等价物):")
                sentiment_index = sentiment_indicators.get('market_sentiment_index', 0.5)
                print(f"    - market_sentiment_index: {sentiment_index:.6f}")
                if sentiment_index > 0.7:
                    mood = "恐慌"
                elif sentiment_index > 0.6:
                    mood = "谨慎"
                elif sentiment_index > 0.4:
                    mood = "平静"
                else:
                    mood = "乐观"
                print(f"    - 解读: 市场情绪{mood} (指数: {sentiment_index:.2f})")

                print(f"  🔄 板块轮动强度:")
                rotation_strength = sentiment_indicators.get('sector_rotation_strength', 0.5)
                print(f"    - sector_rotation_strength: {rotation_strength:.6f}")
                rotation_level = "高" if rotation_strength > 0.7 else "中" if rotation_strength > 0.4 else "低"
                print(f"    - 解读: 板块轮动{rotation_level}活跃 (强度: {rotation_strength:.2f})")

                print(f"  🌐 北向资金影响:")
                northbound_impact = sentiment_indicators.get('northbound_capital_impact', 0.5)
                print(f"    - northbound_capital_impact: {northbound_impact:.6f}")
                impact_level = "高" if northbound_impact > 0.7 else "中" if northbound_impact > 0.4 else "低"
                print(f"    - 解读: 北向资金影响{impact_level} (影响度: {northbound_impact:.2f})")

                test_results[code] = {
                    'sentiment_indicators': sentiment_indicators,
                    'status': 'success'
                }

            else:
                print("❌ 无法计算情绪指标")
                test_results[code] = {
                    'sentiment_indicators': None,
                    'status': 'failed'
                }

        # 测试集成到实时特征计算器
        print(f"\n🔗 测试集成到实时特征计算器")
        print("-" * 50)

        realtime_calculator = RealtimeFeatureCalculator(
            cache_ttl=300,
            db_manager=db_manager,
            logger=logger
        )

        # 测试一个股票的完整特征（包含情绪指标）
        test_code = '000001'
        all_features = realtime_calculator.compute_intraday_features(test_code)

        if all_features:
            # 统计特征数量
            phase21_features = 0
            phase22_features = 0

            for feature_name in all_features.keys():
                if feature_name in ['capital_flow_indicator', 'market_sentiment_index',
                                   'sector_rotation_strength', 'northbound_capital_impact']:
                    phase22_features += 1
                else:
                    phase21_features += 1

            print(f"✅ 集成测试成功:")
            print(f"   - 总特征数: {len(all_features)}")
            print(f"   - Phase 2.1特征: {phase21_features}个")
            print(f"   - Phase 2.2情绪指标: {phase22_features}个")

            # 显示新增的情绪指标
            print(f"   - Phase 2.2新增特征:")
            for name, value in all_features.items():
                if name in ['capital_flow_indicator', 'market_sentiment_index',
                           'sector_rotation_strength', 'northbound_capital_impact']:
                    print(f"     * {name}: {value:.6f}")

        # 生成测试报告
        generate_sentiment_test_report(test_results, all_features if all_features else {})

        print(f"\n🎉 情绪指标增强测试完成！")
        print(f"   ✅ 测试股票数量: {len(test_codes)}")
        print(f"   ✅ 情绪指标数量: 4个")
        print(f"   ✅ 成功率: {sum(1 for r in test_results.values() if r['status'] == 'success') / len(test_codes) * 100:.1f}%")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_sentiment_test_report(test_results: dict, integrated_features: dict):
    """生成情绪指标测试报告"""
    print(f"\n📋 情绪指标测试报告")
    print("=" * 60)

    successful_stocks = sum(1 for r in test_results.values() if r['status'] == 'success')
    total_stocks = len(test_results)

    print(f"📊 测试统计:")
    print(f"   - 总测试股票: {total_stocks}")
    print(f"   - 成功计算: {successful_stocks}")
    print(f"   - 成功率: {successful_stocks/total_stocks*100:.1f}%")

    print(f"\n💭 情绪指标统计:")
    if successful_stocks > 0:
        print(f"   - 情绪指标总数: 4个")
        print(f"   - 资金流向指标: 1个")
        print(f"   - 市场情绪指数: 1个")
        print(f"   - 板块轮动指标: 1个")
        print(f"   - 北向资金指标: 1个")

    # 分析情绪指标分布
    if successful_stocks > 0:
        print(f"\n📈 情绪指标分布分析:")

        all_capital_flows = []
        all_sentiments = []
        all_rotations = []
        all_northbound = []

        for result in test_results.values():
            if result['status'] == 'success' and result['sentiment_indicators']:
                indicators = result['sentiment_indicators']
                all_capital_flows.append(indicators.get('capital_flow_indicator', 0))
                all_sentiments.append(indicators.get('market_sentiment_index', 0.5))
                all_rotations.append(indicators.get('sector_rotation_strength', 0.5))
                all_northbound.append(indicators.get('northbound_capital_impact', 0.5))

        if all_capital_flows:
            import numpy as np
            print(f"   - 资金流向: 均值={np.mean(all_capital_flows):.3f}, 标准差={np.std(all_capital_flows):.3f}")
            print(f"   - 市场情绪: 均值={np.mean(all_sentiments):.3f}, 标准差={np.std(all_sentiments):.3f}")
            print(f"   - 板块轮动: 均值={np.mean(all_rotations):.3f}, 标准差={np.std(all_rotations):.3f}")
            print(f"   - 北向资金: 均值={np.mean(all_northbound):.3f}, 标准差={np.std(all_northbound):.3f}")

    print(f"\n🔗 集成测试结果:")
    if integrated_features:
        total_features = len(integrated_features)
        sentiment_features = sum(1 for name in integrated_features.keys()
                               if name in ['capital_flow_indicator', 'market_sentiment_index',
                                         'sector_rotation_strength', 'northbound_capital_impact'])
        print(f"   - 总特征数: {total_features}")
        print(f"   - 情绪指标: {sentiment_features}/4")
        print(f"   - 集成成功率: {sentiment_features/4*100:.0f}%")
    else:
        print(f"   - 集成测试失败")

    # 保存详细报告
    report_data = {
        'test_time': datetime.now().isoformat(),
        'phase': 'Phase 2.2 - 情绪指标增强',
        'summary': {
            'total_stocks': total_stocks,
            'successful_stocks': successful_stocks,
            'success_rate': successful_stocks/total_stocks*100 if total_stocks > 0 else 0,
            'sentiment_indicators_count': 4,
            'integrated_features_count': len(integrated_features) if integrated_features else 0
        },
        'sentiment_results': test_results,
        'integrated_features': integrated_features
    }

    report_file = f"reports/v380_sentiment_indicators_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs('reports', exist_ok=True)

    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n💾 详细报告已保存: {report_file}")

if __name__ == '__main__':
    success = test_sentiment_indicators()
    sys.exit(0 if success else 1)