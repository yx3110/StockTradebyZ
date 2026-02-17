#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.8 Phase 2.3: 特征有效性验证
实现新特征与收益率相关性分析、稳定性测试、共线性检验和重要性排序
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
import logging
from typing import Dict, List, Optional, Tuple, Any
import json
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append('/Users/yangxu/StockTradebyZ')

from data_adapter.database_manager import DatabaseManager
from incremental_learning.features.realtime_calculator import RealtimeFeatureCalculator

class V38FeatureValidator:
    """
    V3.8特征有效性验证器

    Phase 2.3核心任务:
    1. 新特征与收益率相关性分析
    2. 特征稳定性测试
    3. 多重共线性检验
    4. 特征重要性排序
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = logging.getLogger('FeatureValidator')

        # 初始化实时特征计算器
        self.feature_calculator = RealtimeFeatureCalculator(
            cache_ttl=300,
            db_manager=db_manager,
            logger=self.logger
        )

        # 验证结果存储
        self.validation_results = {}

    def run_comprehensive_validation(self, test_stocks: List[str] = None,
                                   test_days: int = 10) -> Dict[str, Any]:
        """
        运行完整的特征有效性验证

        Args:
            test_stocks: 测试股票列表，默认选择代表性股票
            test_days: 测试天数，默认10天

        Returns:
            Dict: 完整的验证结果
        """
        if test_stocks is None:
            test_stocks = ['000001', '600036', '002215', '300750', '688599']

        print(f"🔍 V3.8 Phase 2.3: 特征有效性验证")
        print(f"=" * 60)
        print(f"📊 测试配置:")
        print(f"   - 测试股票: {len(test_stocks)}只")
        print(f"   - 测试周期: 最近{test_days}个交易日")
        print(f"   - 特征数量: 16个 (12实时 + 4情绪)")

        try:
            # 1. 收集特征和收益率数据
            print(f"\n📈 步骤1: 收集特征和收益率数据")
            feature_data, return_data = self._collect_feature_return_data(test_stocks, test_days)

            if feature_data.empty:
                print("❌ 无法收集到足够的特征数据")
                return {'status': 'failed', 'reason': 'insufficient_data'}

            print(f"   ✅ 成功收集数据: {len(feature_data)}条记录")

            # 2. 相关性分析
            print(f"\n🔗 步骤2: 特征与收益率相关性分析")
            correlation_results = self._analyze_feature_correlations(feature_data, return_data)

            # 3. 特征稳定性测试
            print(f"\n⚖️ 步骤3: 特征稳定性测试")
            stability_results = self._test_feature_stability(feature_data)

            # 4. 多重共线性检验
            print(f"\n🔀 步骤4: 多重共线性检验")
            collinearity_results = self._test_multicollinearity(feature_data)

            # 5. 特征重要性排序
            print(f"\n🏆 步骤5: 特征重要性排序")
            importance_results = self._rank_feature_importance(
                feature_data, return_data, correlation_results, stability_results
            )

            # 6. 汇总验证结果
            validation_summary = self._generate_validation_summary(
                correlation_results, stability_results,
                collinearity_results, importance_results
            )

            # 7. 生成详细报告
            self._generate_validation_report(
                validation_summary, test_stocks, test_days
            )

            print(f"\n🎉 Phase 2.3特征有效性验证完成!")
            return validation_summary

        except Exception as e:
            print(f"❌ 验证过程失败: {e}")
            import traceback
            traceback.print_exc()
            return {'status': 'error', 'error': str(e)}

    def _collect_feature_return_data(self, test_stocks: List[str],
                                   test_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """收集特征数据和对应的收益率数据"""
        feature_records = []
        return_records = []

        # 获取最近的交易日期
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT trade_date
                FROM daily_quotes
                ORDER BY trade_date DESC
                LIMIT ?
            """, (test_days + 5,))  # 多取几天以计算收益率
            trade_dates = [row[0] for row in cursor.fetchall()]

        if len(trade_dates) < test_days + 1:
            print(f"⚠️ 可用交易日不足: {len(trade_dates)}天")
            return pd.DataFrame(), pd.DataFrame()

        # 为每只股票的每个交易日计算特征
        for stock_code in test_stocks:
            print(f"   📊 处理 {stock_code}...")

            for i, trade_date in enumerate(trade_dates[:test_days]):
                try:
                    # 模拟该交易日的市场开盘时间
                    if isinstance(trade_date, str):
                        market_time = datetime.strptime(trade_date, '%Y-%m-%d')
                        market_time = market_time.replace(hour=10, minute=0)
                    else:
                        # trade_date可能是date对象，转换为datetime
                        market_time = datetime.combine(trade_date, datetime.min.time())
                        market_time = market_time.replace(hour=10, minute=0)

                    # 计算实时特征
                    features = self.feature_calculator.compute_intraday_features(
                        stock_code, market_time
                    )

                    if features:
                        # 添加股票和日期信息
                        feature_record = {
                            'code': stock_code,
                            'trade_date': trade_date,
                            **features
                        }
                        feature_records.append(feature_record)

                        # 计算未来收益率（如果有下一个交易日）
                        if i < len(trade_dates) - 1:
                            current_price, next_price = self._get_price_data(
                                stock_code, trade_date, trade_dates[i + 1]
                            )

                            if current_price and next_price:
                                future_return = (next_price - current_price) / current_price
                                return_record = {
                                    'code': stock_code,
                                    'trade_date': trade_date,
                                    'future_return_1d': future_return
                                }
                                return_records.append(return_record)

                except Exception as e:
                    self.logger.warning(f"处理{stock_code} {trade_date}失败: {e}")
                    continue

        feature_df = pd.DataFrame(feature_records)
        return_df = pd.DataFrame(return_records)

        print(f"   ✅ 收集完成: {len(feature_df)}条特征记录, {len(return_df)}条收益率记录")
        return feature_df, return_df

    def _get_price_data(self, code: str, date1, date2) -> Tuple[Optional[float], Optional[float]]:
        """获取两个日期的收盘价数据"""
        try:
            query = """
            SELECT dq.close, dq.trade_date
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE s.code = ? AND dq.trade_date IN (?, ?)
            ORDER BY dq.trade_date
            """

            with self.db_manager.get_connection() as conn:
                results = conn.execute(query, (code, date1, date2)).fetchall()

                if len(results) == 2:
                    return float(results[0][0]), float(results[1][0])
                else:
                    return None, None

        except Exception as e:
            self.logger.warning(f"获取{code}价格数据失败: {e}")
            return None, None

    def _analyze_feature_correlations(self, feature_df: pd.DataFrame,
                                    return_df: pd.DataFrame) -> Dict[str, Any]:
        """分析特征与收益率的相关性"""
        print("   🔍 计算特征与未来收益率相关性...")

        # 合并特征和收益率数据
        merged_df = feature_df.merge(
            return_df, on=['code', 'trade_date'], how='inner'
        )

        if merged_df.empty:
            print("   ❌ 无法匹配特征和收益率数据")
            return {'status': 'failed'}

        # 获取特征列
        feature_columns = [col for col in feature_df.columns
                          if col not in ['code', 'trade_date']]

        correlations = {}
        significant_features = []

        for feature in feature_columns:
            try:
                # 清理数据
                valid_data = merged_df[[feature, 'future_return_1d']].dropna()

                if len(valid_data) < 10:  # 至少需要10个数据点
                    continue

                # 计算皮尔逊相关系数
                pearson_corr, pearson_p = pearsonr(
                    valid_data[feature], valid_data['future_return_1d']
                )

                # 计算斯皮尔曼相关系数（非参数）
                spearman_corr, spearman_p = spearmanr(
                    valid_data[feature], valid_data['future_return_1d']
                )

                correlations[feature] = {
                    'pearson_corr': pearson_corr,
                    'pearson_pvalue': pearson_p,
                    'spearman_corr': spearman_corr,
                    'spearman_pvalue': spearman_p,
                    'sample_size': len(valid_data),
                    'significant': abs(pearson_corr) >= 0.1 and pearson_p < 0.1
                }

                if correlations[feature]['significant']:
                    significant_features.append(feature)

                print(f"     - {feature}: r={pearson_corr:.4f} (p={pearson_p:.4f})")

            except Exception as e:
                self.logger.warning(f"计算{feature}相关性失败: {e}")
                continue

        result = {
            'status': 'success',
            'correlations': correlations,
            'significant_features': significant_features,
            'total_features': len(feature_columns),
            'significant_count': len(significant_features),
            'significant_rate': len(significant_features) / len(feature_columns) if feature_columns else 0
        }

        print(f"   ✅ 相关性分析完成: {len(significant_features)}/{len(feature_columns)}个特征显著")
        return result

    def _test_feature_stability(self, feature_df: pd.DataFrame) -> Dict[str, Any]:
        """测试特征稳定性（变异系数）"""
        print("   ⚖️ 测试特征稳定性...")

        feature_columns = [col for col in feature_df.columns
                          if col not in ['code', 'trade_date']]

        stability_results = {}
        stable_features = []

        for feature in feature_columns:
            try:
                feature_values = feature_df[feature].dropna()

                if len(feature_values) < 5:
                    continue

                # 计算变异系数 (标准差 / 均值)
                mean_val = feature_values.mean()
                std_val = feature_values.std()

                if abs(mean_val) < 1e-10:  # 避免除零
                    cv = float('inf')
                else:
                    cv = abs(std_val / mean_val)

                # 稳定性判断：变异系数 < 2.0
                is_stable = cv < 2.0

                stability_results[feature] = {
                    'mean': mean_val,
                    'std': std_val,
                    'cv': cv,
                    'is_stable': is_stable,
                    'sample_size': len(feature_values)
                }

                if is_stable:
                    stable_features.append(feature)

                print(f"     - {feature}: CV={cv:.4f} ({'稳定' if is_stable else '不稳定'})")

            except Exception as e:
                self.logger.warning(f"计算{feature}稳定性失败: {e}")
                continue

        result = {
            'status': 'success',
            'stability_results': stability_results,
            'stable_features': stable_features,
            'total_features': len(feature_columns),
            'stable_count': len(stable_features),
            'stable_rate': len(stable_features) / len(feature_columns) if feature_columns else 0
        }

        print(f"   ✅ 稳定性测试完成: {len(stable_features)}/{len(feature_columns)}个特征稳定")
        return result

    def _test_multicollinearity(self, feature_df: pd.DataFrame) -> Dict[str, Any]:
        """测试多重共线性"""
        print("   🔀 检验特征间多重共线性...")

        feature_columns = [col for col in feature_df.columns
                          if col not in ['code', 'trade_date']]

        # 计算特征相关性矩阵
        feature_data = feature_df[feature_columns].fillna(0)
        correlation_matrix = feature_data.corr()

        # 找出高相关性的特征对
        high_corr_pairs = []
        independent_features = list(feature_columns)

        for i, feature1 in enumerate(feature_columns):
            for j, feature2 in enumerate(feature_columns[i+1:], i+1):
                corr_value = abs(correlation_matrix.iloc[i, j])

                if corr_value >= 0.8:  # 高相关性阈值
                    high_corr_pairs.append({
                        'feature1': feature1,
                        'feature2': feature2,
                        'correlation': corr_value
                    })

                    # 从独立特征列表中移除一个（保留更重要的）
                    if feature2 in independent_features:
                        independent_features.remove(feature2)

                print(f"     - {feature1} <-> {feature2}: r={corr_value:.4f}")

        result = {
            'status': 'success',
            'correlation_matrix': correlation_matrix.to_dict(),
            'high_corr_pairs': high_corr_pairs,
            'independent_features': independent_features,
            'total_features': len(feature_columns),
            'independent_count': len(independent_features),
            'multicollinearity_rate': len(high_corr_pairs) / (len(feature_columns) * (len(feature_columns) - 1) / 2) if len(feature_columns) > 1 else 0
        }

        print(f"   ✅ 共线性检验完成: {len(high_corr_pairs)}对高相关，{len(independent_features)}个独立特征")
        return result

    def _rank_feature_importance(self, feature_df: pd.DataFrame, return_df: pd.DataFrame,
                                correlation_results: Dict, stability_results: Dict) -> Dict[str, Any]:
        """特征重要性排序"""
        print("   🏆 计算特征重要性排序...")

        feature_columns = [col for col in feature_df.columns
                          if col not in ['code', 'trade_date']]

        importance_scores = {}

        for feature in feature_columns:
            # 综合评分考虑：相关性 + 稳定性 + 独立性
            score = 0

            # 相关性得分 (40%)
            if feature in correlation_results.get('correlations', {}):
                corr_info = correlation_results['correlations'][feature]
                corr_score = abs(corr_info.get('pearson_corr', 0)) * 0.4
                score += corr_score

            # 稳定性得分 (30%)
            if feature in stability_results.get('stability_results', {}):
                stability_info = stability_results['stability_results'][feature]
                if stability_info['is_stable']:
                    # 稳定特征得分更高，变异系数越小越好
                    stability_score = min(1.0, 1.0 / (1 + stability_info['cv'])) * 0.3
                    score += stability_score

            # 预测意义得分 (30%) - 基于相关性显著性
            if feature in correlation_results.get('significant_features', []):
                score += 0.3

            importance_scores[feature] = score

        # 排序
        sorted_features = sorted(
            importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        print("   🏆 特征重要性排名:")
        for i, (feature, score) in enumerate(sorted_features[:10], 1):
            print(f"     {i:2d}. {feature}: {score:.4f}")

        result = {
            'status': 'success',
            'importance_scores': importance_scores,
            'feature_ranking': sorted_features,
            'top_10_features': [f[0] for f in sorted_features[:10]],
            'top_5_features': [f[0] for f in sorted_features[:5]]
        }

        print(f"   ✅ 重要性排序完成")
        return result

    def _generate_validation_summary(self, correlation_results: Dict,
                                   stability_results: Dict,
                                   collinearity_results: Dict,
                                   importance_results: Dict) -> Dict[str, Any]:
        """生成验证总结"""
        return {
            'validation_time': datetime.now().isoformat(),
            'phase': 'Phase 2.3 - 特征有效性验证',
            'correlation_analysis': correlation_results,
            'stability_analysis': stability_results,
            'multicollinearity_analysis': collinearity_results,
            'importance_ranking': importance_results,
            'overall_assessment': {
                'total_features': 16,
                'significant_features': correlation_results.get('significant_count', 0),
                'stable_features': stability_results.get('stable_count', 0),
                'independent_features': collinearity_results.get('independent_count', 0),
                'validation_status': 'completed'
            }
        }

    def _generate_validation_report(self, validation_summary: Dict,
                                  test_stocks: List[str], test_days: int):
        """生成详细验证报告"""
        report_file = f"reports/v380_feature_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        report_data = {
            **validation_summary,
            'test_configuration': {
                'test_stocks': test_stocks,
                'test_days': test_days,
                'total_test_stocks': len(test_stocks)
            }
        }

        os.makedirs('reports', exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)

        print(f"\n💾 详细验证报告已保存: {report_file}")

def main():
    """主函数"""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    try:
        # 初始化验证器
        db_manager = DatabaseManager()
        validator = V38FeatureValidator(db_manager)

        # 运行完整验证
        results = validator.run_comprehensive_validation(
            test_stocks=['000001', '600036', '002215', '300750', '688599'],
            test_days=10
        )

        if results.get('status') == 'error':
            print(f"❌ 验证失败: {results.get('error')}")
            return False

        # 打印验证结果摘要
        print(f"\n📋 Phase 2.3验证结果摘要")
        print("=" * 60)

        overall = results.get('overall_assessment', {})
        print(f"📊 整体评估:")
        print(f"   - 特征总数: {overall.get('total_features', 0)}")
        print(f"   - 显著特征: {overall.get('significant_features', 0)}")
        print(f"   - 稳定特征: {overall.get('stable_features', 0)}")
        print(f"   - 独立特征: {overall.get('independent_features', 0)}")

        # 成功条件检查
        success_criteria = {
            'significant_rate': overall.get('significant_features', 0) / 16 >= 0.3,  # ≥30%特征显著
            'stable_rate': overall.get('stable_features', 0) / 16 >= 0.8,           # ≥80%特征稳定
            'independent_rate': overall.get('independent_features', 0) / 16 >= 0.6  # ≥60%特征独立
        }

        all_success = all(success_criteria.values())

        print(f"\n✅ Phase 2.3目标达成情况:")
        print(f"   - 相关性目标 (≥30%显著): {'✅' if success_criteria['significant_rate'] else '❌'}")
        print(f"   - 稳定性目标 (≥80%稳定): {'✅' if success_criteria['stable_rate'] else '❌'}")
        print(f"   - 独立性目标 (≥60%独立): {'✅' if success_criteria['independent_rate'] else '❌'}")
        print(f"   - 总体评估: {'✅ 通过' if all_success else '⚠️ 部分通过'}")

        return all_success

    except Exception as e:
        print(f"❌ Phase 2.3验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)