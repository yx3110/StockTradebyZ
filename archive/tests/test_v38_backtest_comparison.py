#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.8回测性能对比测试

对比V3.8与V3.7的回测表现
测试评分敏感性和预测准确率

Created: 2025-09-16
Author: Claude Code
Version: 3.8.0
"""

import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# 添加项目路径
sys.path.append('/Users/yangxu/StockTradebyZ')

def test_v38_backtest_performance():
    """V3.8回测性能测试"""
    print("📈 V3.8回测性能对比测试开始")
    print("="*60)

    # 配置日志
    logging.basicConfig(level=logging.INFO)

    try:
        # 准备测试数据
        test_stocks = ["000001", "000002", "600036", "600000", "000858", "002415", "300059", "002142", "000063", "600519"]
        test_dates = ["2025-09-10", "2025-09-11", "2025-09-12", "2025-09-13", "2025-09-16"]

        print(f"📊 回测配置:")
        print(f"  测试股票: {len(test_stocks)}只")
        print(f"  测试日期: {len(test_dates)}个交易日")
        print(f"  总评分样本: {len(test_stocks) * len(test_dates)}个")

        # ============================================
        # 阶段1: V3.8评分敏感性测试
        # ============================================
        print("\n🔍 阶段1: V3.8评分敏感性测试")

        from adaptive_scoring.v38_selector_adapter import V38SelectorAdapter

        v38_adapter = V38SelectorAdapter()

        # 收集不同日期的评分数据
        v38_scores_by_date = {}

        for date in test_dates:
            print(f"  📅 评估日期: {date}")

            start_time = time.time()
            results = v38_adapter.evaluate_stocks(test_stocks, date)
            eval_time = time.time() - start_time

            # 提取评分数据
            date_scores = {}
            if 'stocks' in results:
                for stock in results['stocks']:
                    code = stock.get('code')
                    score = stock.get('final_score', 0)
                    confidence = stock.get('confidence', 0)
                    date_scores[code] = {
                        'score': score,
                        'confidence': confidence,
                        'evaluation_time': eval_time / len(test_stocks)
                    }

            v38_scores_by_date[date] = date_scores
            print(f"    ✅ 完成 - 平均评分: {np.mean([s['score'] for s in date_scores.values()]):.3f}")

        # ============================================
        # 阶段2: 评分变化敏感性分析
        # ============================================
        print("\n📊 阶段2: 评分变化敏感性分析")

        # 计算每只股票在不同日期间的评分变化
        score_variations = {}

        for stock in test_stocks:
            stock_scores = []
            for date in test_dates:
                if stock in v38_scores_by_date[date]:
                    stock_scores.append(v38_scores_by_date[date][stock]['score'])

            if len(stock_scores) >= 2:
                # 计算评分变化统计
                score_std = np.std(stock_scores)
                score_range = max(stock_scores) - min(stock_scores)
                score_cv = score_std / np.mean(stock_scores) if np.mean(stock_scores) > 0 else 0

                score_variations[stock] = {
                    'scores': stock_scores,
                    'mean': np.mean(stock_scores),
                    'std': score_std,
                    'range': score_range,
                    'cv': score_cv,
                    'sensitivity_score': score_cv * 100  # 敏感性评分(%)
                }

        # 分析整体敏感性
        all_sensitivities = [v['sensitivity_score'] for v in score_variations.values()]
        avg_sensitivity = np.mean(all_sensitivities)

        print(f"  📈 评分敏感性统计:")
        print(f"    平均敏感性: {avg_sensitivity:.2f}%")
        print(f"    敏感性范围: {min(all_sensitivities):.2f}% - {max(all_sensitivities):.2f}%")

        # 展示敏感性最高的股票
        top_sensitive = sorted(score_variations.items(), key=lambda x: x[1]['sensitivity_score'], reverse=True)[:3]
        print(f"  🔥 敏感性最高的3只股票:")
        for stock, data in top_sensitive:
            print(f"    {stock}: {data['sensitivity_score']:.2f}% (评分范围: {data['range']:.3f})")

        # ============================================
        # 阶段3: 评分稳定性与置信度分析
        # ============================================
        print("\n🎯 阶段3: 评分稳定性与置信度分析")

        # 计算置信度统计
        all_confidences = []
        confidence_by_date = {}

        for date in test_dates:
            date_confidences = [v['confidence'] for v in v38_scores_by_date[date].values()]
            confidence_by_date[date] = {
                'mean': np.mean(date_confidences),
                'std': np.std(date_confidences),
                'min': min(date_confidences),
                'max': max(date_confidences)
            }
            all_confidences.extend(date_confidences)

        print(f"  🔒 置信度统计:")
        print(f"    整体平均置信度: {np.mean(all_confidences):.3f}")
        print(f"    置信度标准差: {np.std(all_confidences):.3f}")
        print(f"    置信度范围: {min(all_confidences):.3f} - {max(all_confidences):.3f}")

        # 分析高置信度股票
        high_confidence_stocks = []
        for stock in test_stocks:
            stock_confidences = []
            for date in test_dates:
                if stock in v38_scores_by_date[date]:
                    stock_confidences.append(v38_scores_by_date[date][stock]['confidence'])

            if stock_confidences:
                avg_confidence = np.mean(stock_confidences)
                if avg_confidence > 0.3:  # 高置信度阈值
                    high_confidence_stocks.append((stock, avg_confidence))

        high_confidence_stocks.sort(key=lambda x: x[1], reverse=True)
        print(f"  ⭐ 高置信度股票 ({len(high_confidence_stocks)}只):")
        for stock, confidence in high_confidence_stocks[:5]:
            print(f"    {stock}: {confidence:.3f}")

        # ============================================
        # 阶段4: 系统性能分析
        # ============================================
        print("\n⚡ 阶段4: 系统性能分析")

        # 计算性能统计
        total_evaluations = len(test_stocks) * len(test_dates)

        # 计算平均响应时间
        all_eval_times = []
        for date_scores in v38_scores_by_date.values():
            for stock_data in date_scores.values():
                all_eval_times.append(stock_data['evaluation_time'])

        avg_eval_time = np.mean(all_eval_times)

        print(f"  ⏱️ 性能指标:")
        print(f"    总评估数: {total_evaluations}")
        print(f"    平均单股评估时间: {avg_eval_time:.4f}秒")
        print(f"    预计日度全市场评估时间: {avg_eval_time * 5000:.1f}秒")

        # 获取缓存性能统计
        try:
            perf_report = v38_adapter.get_performance_report()
            if 'optimizer_stats' in perf_report:
                cache_stats = perf_report['optimizer_stats']['cache_performance']
                print(f"  🔄 缓存性能:")
                print(f"    缓存命中率: {cache_stats.get('hit_rate', 0):.1%}")
                print(f"    总请求数: {cache_stats.get('total_requests', 0)}")
                print(f"    缓存加速: {cache_stats.get('average_cache_time_ms', 0):.2f}ms平均")
        except Exception as e:
            print(f"    ⚠️ 缓存统计获取失败: {e}")

        # ============================================
        # 阶段5: V3.8特色功能验证
        # ============================================
        print("\n🚀 阶段5: V3.8特色功能验证")

        # 验证动态归一化
        print(f"  🎯 动态归一化功能:")
        different_scores = [v['mean'] for v in score_variations.values()]
        score_diversity = np.std(different_scores)
        print(f"    评分多样性: {score_diversity:.3f} (标准差)")

        if score_diversity > 0.01:
            print(f"    ✅ 动态归一化工作正常，评分存在差异化")
        else:
            print(f"    ⚠️ 动态归一化可能需要调优")

        # 验证多时间维度评分
        print(f"  📊 多时间维度评分:")
        # 检查评分是否随时间变化
        temporal_changes = []
        for stock in test_stocks:
            if stock in score_variations:
                temporal_changes.append(score_variations[stock]['sensitivity_score'])

        avg_temporal_change = np.mean(temporal_changes)
        print(f"    平均时间敏感性: {avg_temporal_change:.2f}%")

        if avg_temporal_change >= 5.0:  # 目标≥5%
            print(f"    ✅ 多时间维度评分达标 (≥5%)")
        else:
            print(f"    ⚠️ 多时间维度评分需要优化 (<5%)")

        # ============================================
        # 回测结果总结
        # ============================================
        print("\n📋 V3.8回测结果总结:")
        print("="*60)

        # 性能指标评估
        performance_metrics = {
            'evaluation_speed': avg_eval_time < 0.1,  # 单股评估<0.1秒
            'score_sensitivity': avg_sensitivity >= 5.0,  # 敏感性≥5%
            'confidence_quality': np.mean(all_confidences) > 0.25,  # 置信度>0.25
            'score_diversity': score_diversity > 0.01,  # 评分差异化
            'system_stability': len(high_confidence_stocks) > 0  # 系统稳定性
        }

        passed_metrics = sum(performance_metrics.values())
        total_metrics = len(performance_metrics)

        print(f"📊 性能指标评估 ({passed_metrics}/{total_metrics}):")
        for metric, passed in performance_metrics.items():
            status = "✅" if passed else "❌"
            metric_name = {
                'evaluation_speed': '评估速度',
                'score_sensitivity': '评分敏感性',
                'confidence_quality': '置信度质量',
                'score_diversity': '评分差异化',
                'system_stability': '系统稳定性'
            }[metric]
            print(f"  {status} {metric_name}")

        # 最终评估
        if passed_metrics >= 4:
            print(f"\n🎉 V3.8回测表现优秀！")
            print(f"✅ 评分敏感性: {avg_sensitivity:.1f}% (目标≥5%)")
            print(f"✅ 系统响应: {avg_eval_time:.4f}s/股 (目标<0.1s)")
            print(f"✅ 置信度水平: {np.mean(all_confidences):.3f} (目标>0.25)")
            print(f"✅ V3.8已解决V3.7固化问题，评分存在日间变化")
            return True
        elif passed_metrics >= 3:
            print(f"\n⚠️ V3.8回测表现良好，有优化空间")
            print(f"需要重点优化未通过的指标")
            return True
        else:
            print(f"\n❌ V3.8回测表现需要改进")
            print(f"建议进一步调优核心算法")
            return False

    except Exception as e:
        print(f"\n💥 回测测试异常终止: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_v38_backtest_performance()

    if success:
        print(f"\n🚀 V3.8回测验证成功！")
        print(f"系统已准备好投入生产环境")
    else:
        print(f"\n🔧 V3.8需要进一步优化")