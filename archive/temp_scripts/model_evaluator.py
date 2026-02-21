#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.9模型完整评估器
实现所有金融和机器学习评估指标
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, shapiro, entropy
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class ModelEvaluator:
    """完整的模型评估器"""

    def __init__(self, y_true, y_pred):
        """
        Args:
            y_true: 实际收益率
            y_pred: 预测收益率
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)

    def direction_accuracy(self):
        """方向准确率 - 最重要！"""
        return np.mean((self.y_pred > 0) == (self.y_true > 0))

    def information_coefficient(self):
        """IC - 信息系数"""
        ic, p_value = spearmanr(self.y_pred, self.y_true)
        return ic

    def r_squared(self):
        """R² - 决定系数"""
        return r2_score(self.y_true, self.y_pred)

    def mean_errors(self):
        """MAE, MSE, RMSE"""
        mse = mean_squared_error(self.y_true, self.y_pred)
        mae = mean_absolute_error(self.y_true, self.y_pred)
        rmse = np.sqrt(mse)
        return {'mse': mse, 'mae': mae, 'rmse': rmse}

    def top_n_performance(self, n=20):
        """Top N股票的平均收益"""
        if len(self.y_pred) < n:
            n = len(self.y_pred)

        top_n_idx = np.argsort(self.y_pred)[-n:]
        return {
            f'top_{n}_mean_return': self.y_true[top_n_idx].mean(),
            f'top_{n}_median_return': np.median(self.y_true[top_n_idx]),
            f'top_{n}_positive_rate': (self.y_true[top_n_idx] > 0).mean()
        }

    def quantile_analysis(self, n_quantiles=5):
        """分位数性能分析"""
        try:
            quantiles = pd.qcut(self.y_pred, q=n_quantiles, labels=False, duplicates='drop')
        except:
            # 如果值太少无法分位，返回空结果
            return {
                'quantile_returns': [],
                'is_monotonic': False
            }

        results = []
        for q in range(n_quantiles):
            mask = (quantiles == q)
            if mask.sum() > 0:
                results.append({
                    'quantile': q + 1,
                    'mean_return': self.y_true[mask].mean(),
                    'median_return': np.median(self.y_true[mask]),
                    'win_rate': (self.y_true[mask] > 0).mean(),
                    'count': mask.sum()
                })

        # 检查单调性
        returns = [r['mean_return'] for r in results]
        is_monotonic = all(returns[i] <= returns[i+1] for i in range(len(returns)-1))

        return {
            'quantile_returns': results,
            'is_monotonic': is_monotonic
        }

    def win_rate_and_profit_factor(self, percentile=80):
        """
        胜率和盈亏比

        Args:
            percentile: 买入阈值（只买预测排名前X%）
        """
        threshold = np.percentile(self.y_pred, percentile)
        buy_mask = (self.y_pred >= threshold)

        if buy_mask.sum() == 0:
            return {
                'win_rate': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0
            }

        # 胜率
        win_rate = (self.y_true[buy_mask] > 0).mean()

        # 盈亏比
        wins = self.y_true[buy_mask & (self.y_true > 0)]
        losses = self.y_true[buy_mask & (self.y_true < 0)]

        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0

        total_profit = wins.sum() if len(wins) > 0 else 0
        total_loss = -losses.sum() if len(losses) > 0 else 0

        profit_factor = total_profit / total_loss if total_loss > 0 else np.inf

        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_wins': len(wins),
            'total_losses': len(losses)
        }

    def distribution_match(self, bins=20):
        """预测分布 vs 实际分布"""
        hist_pred, _ = np.histogram(self.y_pred, bins=bins, density=True)
        hist_true, _ = np.histogram(self.y_true, bins=bins, density=True)

        # 避免0
        hist_pred = hist_pred + 1e-10
        hist_true = hist_true + 1e-10

        kl_div = entropy(hist_true, hist_pred)

        return {
            'kl_divergence': kl_div,
            'match_quality': 'good' if kl_div < 0.5 else 'poor'
        }

    def residual_analysis(self):
        """残差分析"""
        residuals = self.y_true - self.y_pred

        # 正态性检验（最多5000样本）
        sample_size = min(5000, len(residuals))
        stat, p_value = shapiro(residuals[:sample_size])

        return {
            'mean_residual': residuals.mean(),
            'std_residual': residuals.std(),
            'skewness': pd.Series(residuals).skew(),
            'kurtosis': pd.Series(residuals).kurtosis(),
            'normality_p_value': p_value,
            'is_normal': p_value > 0.05
        }

    def comprehensive_score(self):
        """
        综合评分（0-100）

        权重分配：
        - 方向准确率：30分
        - IC：25分
        - R²：15分
        - Top 20收益：20分
        - 分位数单调性：10分
        """
        dir_acc = self.direction_accuracy()
        ic = self.information_coefficient()
        r2 = self.r_squared()
        top_20 = self.top_n_performance(20)['top_20_mean_return']
        quantile = self.quantile_analysis()

        score = (
            dir_acc * 30 +  # 方向准确率（30分）
            min(ic / 0.10, 1.0) * 25 +  # IC（25分）
            min(max(r2, 0) / 0.40, 1.0) * 15 +  # R²（15分）
            min(max(top_20, 0) / 0.05, 1.0) * 20 +  # Top20收益（20分）
            (10 if quantile['is_monotonic'] else 0)  # 分位数单调性（10分）
        )

        grade = (
            'A' if score >= 80 else
            'B' if score >= 70 else
            'C' if score >= 60 else
            'D' if score >= 50 else
            'F'
        )

        return {
            'score': score,
            'grade': grade,
            'components': {
                'direction_accuracy_score': dir_acc * 30,
                'ic_score': min(ic / 0.10, 1.0) * 25,
                'r2_score': min(max(r2, 0) / 0.40, 1.0) * 15,
                'top20_score': min(max(top_20, 0) / 0.05, 1.0) * 20,
                'monotonic_score': 10 if quantile['is_monotonic'] else 0
            }
        }

    def full_evaluation(self):
        """完整评估 - 返回所有指标"""
        results = {}

        # 1. 基础准确性
        results['direction_accuracy'] = self.direction_accuracy()
        results['ic'] = self.information_coefficient()
        results['r2'] = self.r_squared()
        results.update(self.mean_errors())

        # 2. 金融指标
        results.update(self.top_n_performance(20))
        results.update(self.top_n_performance(50))
        results['quantile_analysis'] = self.quantile_analysis()
        results.update(self.win_rate_and_profit_factor(80))

        # 3. 分布分析
        results['distribution'] = self.distribution_match()
        results['residuals'] = self.residual_analysis()

        # 4. 综合评分
        results['comprehensive'] = self.comprehensive_score()

        return results

    def print_report(self):
        """打印评估报告"""
        eval_results = self.full_evaluation()

        print("=" * 80)
        print("📊 V3.9模型综合评估报告")
        print("=" * 80)
        print()

        print("【基础准确性指标】")
        print(f"  方向准确率:    {eval_results['direction_accuracy']*100:.2f}%  " +
              ("✅" if eval_results['direction_accuracy'] > 0.70 else
               "🟡" if eval_results['direction_accuracy'] > 0.60 else "❌"))
        print(f"  IC (信息系数):  {eval_results['ic']:.4f}         " +
              ("✅" if eval_results['ic'] > 0.05 else
               "🟡" if eval_results['ic'] > 0.03 else "❌"))
        print(f"  R²:            {eval_results['r2']:.4f}         " +
              ("✅" if eval_results['r2'] > 0.20 else
               "🟡" if eval_results['r2'] > 0.10 else "❌"))
        print(f"  MAE:           {eval_results['mae']:.4f} ({eval_results['mae']*100:.2f}%)  " +
              ("✅" if eval_results['mae'] < 0.03 else
               "🟡" if eval_results['mae'] < 0.05 else "❌"))
        print()

        print("【金融实战指标】")
        print(f"  Top 20平均收益: {eval_results['top_20_mean_return']*100:.2f}%    " +
              ("✅" if eval_results['top_20_mean_return'] > 0.02 else
               "🟡" if eval_results['top_20_mean_return'] > 0 else "❌"))
        print(f"  Top 20胜率:     {eval_results['top_20_positive_rate']*100:.2f}%    " +
              ("✅" if eval_results['top_20_positive_rate'] > 0.60 else
               "🟡" if eval_results['top_20_positive_rate'] > 0.50 else "❌"))
        print(f"  分位数单调性:   {'是' if eval_results['quantile_analysis']['is_monotonic'] else '否'}           " +
              ("✅" if eval_results['quantile_analysis']['is_monotonic'] else "❌"))
        print(f"  胜率 (Top20%):  {eval_results['win_rate']*100:.2f}%      " +
              ("✅" if eval_results['win_rate'] > 0.60 else
               "🟡" if eval_results['win_rate'] > 0.50 else "❌"))
        print(f"  盈亏比:         {eval_results['profit_factor']:.2f}         " +
              ("✅" if eval_results['profit_factor'] > 1.5 else
               "🟡" if eval_results['profit_factor'] > 1.0 else "❌"))
        print()

        print("【分位数详情】")
        for q in eval_results['quantile_analysis']['quantile_returns']:
            print(f"  Q{q['quantile']} (n={q['count']:4d}): " +
                  f"收益={q['mean_return']*100:+.2f}%, 胜率={q['win_rate']*100:.1f}%")
        print()

        print("=" * 80)
        print(f"📈 综合评分: {eval_results['comprehensive']['score']:.1f}/100  " +
              f"等级: {eval_results['comprehensive']['grade']}")
        print("=" * 80)
        print()

        print("【评级标准】")
        print("  A (80-100): 卓越，可以实盘")
        print("  B (70-79):  优秀，谨慎实盘")
        print("  C (60-69):  良好，模拟盘测试")
        print("  D (50-59):  及格，继续改进")
        print("  F (<50):    不及格，重新训练")
        print("=" * 80)

        # 实盘建议
        grade = eval_results['comprehensive']['grade']
        if grade in ['A', 'B']:
            print()
            print("✅ 建议：模型通过验收，可以考虑实盘（建议从小仓位开始）")
        elif grade == 'C':
            print()
            print("🟡 建议：模型勉强可用，建议先模拟盘测试1-2个月")
        else:
            print()
            print("❌ 建议：模型不合格，需要继续优化")

        return eval_results


# 使用示例
if __name__ == "__main__":
    # 模拟数据
    np.random.seed(42)
    y_true = np.random.randn(1000) * 0.05
    y_pred = y_true + np.random.randn(1000) * 0.03

    evaluator = ModelEvaluator(y_true, y_pred)
    results = evaluator.print_report()
