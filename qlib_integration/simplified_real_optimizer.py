#!/usr/bin/env python3
"""
简化版真实数据权重优化器
使用真实数据但简化查询，确保能正常运行
"""

import os
import sys
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
import json

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from hyperopt import hp, fmin, tpe, STATUS_OK, STATUS_FAIL, Trials

class SimplifiedRealDataOptimizer:
    """简化版真实数据权重优化器"""
    
    def __init__(self, database_path: str = "./data_adapter/stock_data.db"):
        self.database_path = os.path.join(project_root, database_path)
        self.logger = self._setup_logging()
        
        # 缓存
        self.historical_data_cache = {}
        self.future_returns_cache = {}
        self.trials = Trials()
        self.best_weights = None
        self.v30_baseline_correlation = 0.065
        
        self.logger.info("🚀 简化版真实数据优化器已初始化")
        
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("SimplifiedRealOptimizer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def prepare_data(self, max_stocks: int = 200) -> Tuple[Dict, Dict]:
        """准备优化数据"""
        self.logger.info("📊 准备真实数据...")
        
        # 获取最近的交易日
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.execute("""
                SELECT MAX(trade_date) 
                FROM daily_quotes 
                WHERE trade_date <= date('now', '-1 day')
            """)
            end_date = cursor.fetchone()[0]
        
        if not end_date:
            raise ValueError("未找到有效的交易数据")
        
        self.logger.info(f"📅 使用数据截止日期: {end_date}")
        
        # 获取活跃股票（使用简化查询）
        with sqlite3.connect(self.database_path) as conn:
            query = """
                SELECT DISTINCT s.code, s.name
                FROM securities s
                JOIN daily_quotes dq ON s.id = dq.security_id
                WHERE s.type = 'A股'
                  AND dq.trade_date = ?
                  AND dq.volume > 1000000
                  AND dq.close > 1.0
                  AND s.code NOT LIKE '%ST%'
                  AND s.delist_date IS NULL
                ORDER BY dq.volume DESC
                LIMIT ?
            """
            df_stocks = pd.read_sql_query(query, conn, params=(end_date, max_stocks))
        
        if len(df_stocks) == 0:
            raise ValueError("未找到符合条件的股票")
        
        self.logger.info(f"📈 获取到 {len(df_stocks)} 只活跃A股")
        
        historical_features = {}
        future_returns = {}
        
        processed = 0
        for _, row in df_stocks.iterrows():
            stock_code = row['code']
            try:
                # 获取基础价格数据
                features = self._get_basic_features(stock_code, end_date)
                if features is None:
                    continue
                
                # 计算未来收益
                returns = self._get_future_returns(stock_code, end_date)
                if returns is None:
                    continue
                
                historical_features[stock_code] = features
                future_returns[stock_code] = returns
                
                processed += 1
                if processed % 50 == 0:
                    self.logger.info(f"✅ 已处理 {processed} 只股票")
                    
            except Exception as e:
                self.logger.warning(f"❌ 处理 {stock_code} 出错: {str(e)}")
                continue
        
        self.logger.info(f"🎯 数据准备完成，共 {len(historical_features)} 只股票")
        
        self.historical_data_cache = historical_features
        self.future_returns_cache = future_returns
        
        return historical_features, future_returns
    
    def _get_basic_features(self, stock_code: str, end_date: str) -> Optional[Dict]:
        """获取基础特征数据"""
        try:
            # 获取最近90天的数据用于计算指标
            start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=90)).strftime('%Y-%m-%d')
            
            with sqlite3.connect(self.database_path) as conn:
                # 基础价格数据
                query = """
                    SELECT dq.trade_date, dq.open, dq.high, dq.low, dq.close, dq.volume
                    FROM daily_quotes dq
                    JOIN securities s ON dq.security_id = s.id
                    WHERE s.code = ?
                      AND dq.trade_date BETWEEN ? AND ?
                    ORDER BY dq.trade_date ASC
                """
                df_price = pd.read_sql_query(query, conn, params=(stock_code, start_date, end_date))
                
                if len(df_price) < 20:
                    return None
                
                # 基本面数据（最新一天）
                query_basic = """
                    SELECT db.pe_ttm, db.pb, db.total_mv, db.turnover_rate
                    FROM daily_basic db
                    JOIN securities s ON db.security_id = s.id
                    WHERE s.code = ? AND db.trade_date = ?
                """
                df_basic = pd.read_sql_query(query_basic, conn, params=(stock_code, end_date))
                
                # 技术指标（最新一天）
                query_tech = """
                    SELECT ti.kdj_k, ti.kdj_d, ti.kdj_j, ti.rsi24, ti.bbi
                    FROM technical_indicators ti
                    JOIN securities s ON ti.security_id = s.id
                    WHERE s.code = ? AND ti.trade_date = ?
                """
                df_tech = pd.read_sql_query(query_tech, conn, params=(stock_code, end_date))
            
            # 计算特征
            features = {}
            
            # 价格相关
            features['close'] = float(df_price['close'].iloc[-1])
            
            # 技术指标
            if not df_tech.empty:
                features['kdj_k'] = float(df_tech['kdj_k'].iloc[0]) if pd.notna(df_tech['kdj_k'].iloc[0]) else 50.0
                features['kdj_d'] = float(df_tech['kdj_d'].iloc[0]) if pd.notna(df_tech['kdj_d'].iloc[0]) else 50.0
                features['kdj_j'] = float(df_tech['kdj_j'].iloc[0]) if pd.notna(df_tech['kdj_j'].iloc[0]) else 50.0
                features['rsi'] = float(df_tech['rsi24'].iloc[0]) if pd.notna(df_tech['rsi24'].iloc[0]) else 50.0
                features['bbi'] = float(df_tech['bbi'].iloc[0]) if pd.notna(df_tech['bbi'].iloc[0]) else features['close']
            else:
                features.update({'kdj_k': 50.0, 'kdj_d': 50.0, 'kdj_j': 50.0, 'rsi': 50.0, 'bbi': features['close']})
            
            # 知行指标（简化计算）
            if len(df_price) >= 12:
                ema12 = df_price['close'].ewm(span=12).mean().iloc[-1]
                features['zhixing_trend'] = float(ema12)
            else:
                features['zhixing_trend'] = features['close']
                
            if len(df_price) >= 20:
                ma_values = []
                for period in [5, 10, 20]:
                    if len(df_price) >= period:
                        ma_values.append(df_price['close'].iloc[-period:].mean())
                features['zhixing_multiavg'] = float(np.mean(ma_values)) if ma_values else features['close']
            else:
                features['zhixing_multiavg'] = features['close']
            
            # 成交量指标
            if len(df_price) >= 10:
                current_vol = df_price['volume'].iloc[-1]
                avg_vol = df_price['volume'].iloc[-10:-1].mean()
                features['volume_surge'] = float(current_vol / (avg_vol + 1e-9)) if avg_vol > 0 else 1.0
            else:
                features['volume_surge'] = 1.0
            
            # 价格动量
            if len(df_price) >= 10:
                features['price_momentum'] = float((df_price['close'].iloc[-1] / df_price['close'].iloc[-10] - 1))
            else:
                features['price_momentum'] = 0.0
            
            # 波动率
            if len(df_price) >= 20:
                returns = df_price['close'].pct_change().dropna()
                features['volatility'] = float(returns.iloc[-20:].std()) if len(returns) >= 20 else 0.02
            else:
                features['volatility'] = 0.02
            
            # 基本面
            if not df_basic.empty:
                features['pe_ttm'] = float(df_basic['pe_ttm'].iloc[0]) if pd.notna(df_basic['pe_ttm'].iloc[0]) and df_basic['pe_ttm'].iloc[0] > 0 else 30.0
                features['pb'] = float(df_basic['pb'].iloc[0]) if pd.notna(df_basic['pb'].iloc[0]) and df_basic['pb'].iloc[0] > 0 else 2.0
                features['market_cap'] = float(df_basic['total_mv'].iloc[0]) if pd.notna(df_basic['total_mv'].iloc[0]) else 1000000.0
                features['turnover_rate'] = float(df_basic['turnover_rate'].iloc[0]) if pd.notna(df_basic['turnover_rate'].iloc[0]) else 1.0
            else:
                features.update({'pe_ttm': 30.0, 'pb': 2.0, 'market_cap': 1000000.0, 'turnover_rate': 1.0})
            
            return features
            
        except Exception as e:
            self.logger.warning(f"计算特征失败 {stock_code}: {str(e)}")
            return None
    
    def _get_future_returns(self, stock_code: str, end_date: str) -> Optional[Dict]:
        """计算未来收益"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                # 获取未来20天数据
                query = """
                    SELECT dq.trade_date, dq.close
                    FROM daily_quotes dq
                    JOIN securities s ON dq.security_id = s.id
                    WHERE s.code = ?
                      AND dq.trade_date > ?
                      AND dq.trade_date <= date(?, '+20 days')
                    ORDER BY dq.trade_date ASC
                    LIMIT 20
                """
                df_future = pd.read_sql_query(query, conn, params=(stock_code, end_date, end_date))
                
                # 获取当前价格
                query_current = """
                    SELECT dq.close
                    FROM daily_quotes dq
                    JOIN securities s ON dq.security_id = s.id
                    WHERE s.code = ? AND dq.trade_date = ?
                """
                df_current = pd.read_sql_query(query_current, conn, params=(stock_code, end_date))
            
            if df_current.empty or df_future.empty:
                return None
            
            current_price = df_current['close'].iloc[0]
            returns = {}
            
            # 计算不同期限收益
            for days in [1, 3, 5, 10]:
                if len(df_future) >= days:
                    future_price = df_future['close'].iloc[days-1]
                    returns[f'return_{days}d'] = float(future_price / current_price - 1)
                else:
                    returns[f'return_{days}d'] = 0.0
            
            return returns
            
        except Exception as e:
            self.logger.warning(f"计算未来收益失败 {stock_code}: {str(e)}")
            return None
    
    def setup_optimization_space(self) -> Dict:
        """设置优化空间"""
        return {
            'kdj_strength': hp.uniform('kdj_strength', 0.08, 0.18),
            'rsi_momentum': hp.uniform('rsi_momentum', 0.06, 0.16),
            'bbi_trend': hp.uniform('bbi_trend', 0.04, 0.14),
            'volume_surge': hp.uniform('volume_surge', 0.06, 0.16),
            'zhixing_trend': hp.uniform('zhixing_trend', 0.06, 0.20),
            'zhixing_multiavg': hp.uniform('zhixing_multiavg', 0.04, 0.14),
            'pe_valuation': hp.uniform('pe_valuation', 0.01, 0.04),
            'pb_valuation': hp.uniform('pb_valuation', 0.01, 0.04),
            'price_momentum': hp.uniform('price_momentum', 0.04, 0.14),
            'volatility_risk': hp.uniform('volatility_risk', 0.01, 0.05),
            'risk_penalty': hp.uniform('risk_penalty', 0.001, 0.1),
        }
    
    def objective_function(self, params: Dict) -> Dict:
        """目标函数"""
        try:
            # 归一化权重
            weight_params = {k: v for k, v in params.items() if k != 'risk_penalty'}
            total_weight = sum(weight_params.values())
            normalized_weights = {k: v / total_weight for k, v in weight_params.items()}
            
            # 计算评分
            scores = {}
            for stock_code, features in self.historical_data_cache.items():
                score = self._calculate_score(features, normalized_weights)
                scores[stock_code] = score
            
            if len(scores) == 0:
                return {'loss': float('inf'), 'status': STATUS_FAIL}
            
            # 计算相关性
            correlations = {}
            for period in ['1d', '3d', '5d', '10d']:
                future_returns = []
                stock_scores = []
                
                for stock_code in scores.keys():
                    if stock_code in self.future_returns_cache:
                        returns = self.future_returns_cache[stock_code]
                        if f'return_{period}' in returns:
                            future_returns.append(returns[f'return_{period}'])
                            stock_scores.append(scores[stock_code])
                
                if len(future_returns) > 10:
                    corr = np.corrcoef(stock_scores, future_returns)[0, 1]
                    correlations[period] = corr if not np.isnan(corr) else 0.0
                else:
                    correlations[period] = 0.0
            
            # 综合相关性
            correlation_score = (
                0.4 * correlations.get('1d', 0) +
                0.3 * correlations.get('3d', 0) +
                0.2 * correlations.get('5d', 0) +
                0.1 * correlations.get('10d', 0)
            )
            
            # 评分分布质量
            score_values = list(scores.values())
            score_std = np.std(score_values)
            distribution_score = min(score_std / 15.0, 1.0) * 0.1
            
            # 风险惩罚
            risk_penalty = params.get('risk_penalty', 0.01) * score_std * 0.05
            
            # 目标函数（负号因为hyperopt求最小值）
            objective_value = -(0.8 * correlation_score + 0.2 * distribution_score - risk_penalty)
            
            self.logger.info(f"📊 相关性: {correlation_score:.4f}, 目标函数: {-objective_value:.4f}")
            
            return {
                'loss': objective_value,
                'status': STATUS_OK,
                'detailed_info': {
                    'correlations': correlations,
                    'correlation_score': correlation_score,
                    'score_stats': {
                        'mean': np.mean(score_values),
                        'std': score_std,
                        'range': np.ptp(score_values)
                    },
                    'weights': normalized_weights,
                    'total_objective': -objective_value
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ 目标函数计算出错: {str(e)}")
            return {'loss': float('inf'), 'status': STATUS_FAIL}
    
    def _calculate_score(self, features: Dict, weights: Dict) -> float:
        """计算加权评分"""
        score = 0.0
        
        # 简化的评分函数
        # KDJ
        kdj_k = features.get('kdj_k', 50)
        kdj_score = 0.9 if kdj_k < 30 else (0.7 if kdj_k < 50 else (0.2 if kdj_k > 80 else 0.5))
        score += weights.get('kdj_strength', 0) * kdj_score
        
        # RSI
        rsi = features.get('rsi', 50)
        rsi_score = 0.9 if rsi < 30 else (0.7 if rsi < 50 else (0.2 if rsi > 70 else 0.5))
        score += weights.get('rsi_momentum', 0) * rsi_score
        
        # BBI趋势
        close = features.get('close', 0)
        bbi = features.get('bbi', close)
        bbi_score = 0.8 if close > bbi * 1.02 else (0.6 if close > bbi else 0.3)
        score += weights.get('bbi_trend', 0) * bbi_score
        
        # 成交量
        volume_ratio = features.get('volume_surge', 1)
        volume_score = 0.9 if volume_ratio > 3 else (0.7 if volume_ratio > 2 else (0.6 if volume_ratio > 1.5 else 0.4))
        score += weights.get('volume_surge', 0) * volume_score
        
        # 知行指标
        zhixing_trend = features.get('zhixing_trend', close)
        trend_score = 0.8 if close > zhixing_trend * 1.01 else (0.6 if close > zhixing_trend else 0.3)
        score += weights.get('zhixing_trend', 0) * trend_score
        
        zhixing_multiavg = features.get('zhixing_multiavg', close)
        multiavg_score = 0.8 if close > zhixing_multiavg * 1.01 else (0.6 if close > zhixing_multiavg else 0.3)
        score += weights.get('zhixing_multiavg', 0) * multiavg_score
        
        # 基本面
        pe = features.get('pe_ttm', 30)
        pe_score = 0.8 if pe < 15 else (0.6 if pe < 25 else (0.4 if pe < 40 else 0.2))
        score += weights.get('pe_valuation', 0) * pe_score
        
        pb = features.get('pb', 2)
        pb_score = 0.8 if pb < 1 else (0.6 if pb < 2 else (0.4 if pb < 3 else 0.2))
        score += weights.get('pb_valuation', 0) * pb_score
        
        # 价格动量
        momentum = features.get('price_momentum', 0)
        momentum_score = 0.8 if momentum > 0.1 else (0.6 if momentum > 0.05 else (0.5 if momentum > 0 else 0.3))
        score += weights.get('price_momentum', 0) * momentum_score
        
        # 波动率风险
        volatility = features.get('volatility', 0.02)
        vol_score = 0.8 if volatility < 0.02 else (0.6 if volatility < 0.03 else (0.4 if volatility < 0.05 else 0.2))
        score += weights.get('volatility_risk', 0) * vol_score
        
        return max(0, min(100, score * 100))
    
    def run_optimization(self, max_evals: int = 30) -> Dict:
        """运行优化"""
        self.logger.info(f"🚀 开始简化版真实数据优化，评估次数: {max_evals}")
        
        # 准备数据
        if not self.historical_data_cache:
            self.prepare_data()
        
        if len(self.historical_data_cache) < 10:
            raise ValueError("有效数据太少")
        
        # 运行优化
        space = self.setup_optimization_space()
        best = fmin(
            fn=self.objective_function,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=self.trials,
            verbose=True
        )
        
        # 归一化最佳权重
        weight_params = {k: v for k, v in best.items() if k != 'risk_penalty'}
        total_weight = sum(weight_params.values())
        self.best_weights = {k: v / total_weight for k, v in weight_params.items()}
        
        # 分析结果
        best_trial = min(self.trials.trials, key=lambda x: x['result']['loss'])
        best_detailed_info = best_trial['result'].get('detailed_info', {})
        
        current_correlation = best_detailed_info.get('correlation_score', 0)
        improvement = current_correlation - self.v30_baseline_correlation
        
        results = {
            'optimization_summary': {
                'total_trials': len(self.trials.trials),
                'best_correlation': current_correlation,
                'v30_baseline_correlation': self.v30_baseline_correlation,
                'improvement_vs_v30': improvement,
                'stocks_used': len(self.historical_data_cache),
            },
            'best_weights': self.best_weights,
            'detailed_analysis': best_detailed_info,
        }
        
        # 保存结果
        self._save_results(results)
        
        self.logger.info("✅ 优化完成！")
        return results
    
    def _save_results(self, results: Dict):
        """保存结果"""
        reports_dir = Path("reports/qlib_optimization")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON结果
        results_file = reports_dir / f"simplified_real_optimization_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # Markdown报告
        report_file = reports_dir / f"简化版真实数据优化报告_{timestamp}.md"
        self._generate_report(results, report_file)
        
        self.logger.info(f"📄 结果已保存: {results_file}")
        self.logger.info(f"📊 报告已生成: {report_file}")
    
    def _generate_report(self, results: Dict, report_file: Path):
        """生成报告"""
        summary = results['optimization_summary']
        weights = results['best_weights']
        
        content = f"""# 简化版真实数据Qlib权重优化报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 优化结果

### 核心指标
- **使用股票数**: {summary['stocks_used']} 只真实A股
- **最佳相关性**: {summary['best_correlation']:.4f}
- **V3.0基线**: {summary['v30_baseline_correlation']:.4f}
- **相对改进**: {summary['improvement_vs_v30']:+.4f}
- **改进幅度**: {(summary['improvement_vs_v30']/summary['v30_baseline_correlation']*100):+.1f}%

## 🎯 优化权重配置

### 技术指标权重
- **KDJ强度**: {weights.get('kdj_strength', 0):.3f}
- **RSI动量**: {weights.get('rsi_momentum', 0):.3f}
- **BBI趋势**: {weights.get('bbi_trend', 0):.3f}
- **成交量激增**: {weights.get('volume_surge', 0):.3f}
- **知行趋势线**: {weights.get('zhixing_trend', 0):.3f}
- **知行多空线**: {weights.get('zhixing_multiavg', 0):.3f}

### 基本面权重
- **PE估值**: {weights.get('pe_valuation', 0):.3f}
- **PB估值**: {weights.get('pb_valuation', 0):.3f}

### 市场表现权重
- **价格动量**: {weights.get('price_momentum', 0):.3f}
- **波动风险**: {weights.get('volatility_risk', 0):.3f}

## ✅ 结论

基于{summary['stocks_used']}只真实A股数据的优化结果显示：
"""
        
        if summary['improvement_vs_v30'] > 0:
            content += f"**✅ 优化成功**: 相关性提升{summary['improvement_vs_v30']:.4f}，改进幅度{(summary['improvement_vs_v30']/summary['v30_baseline_correlation']*100):+.1f}%\n\n"
            content += "**建议**: 可以将这些权重应用到V3.5系统中"
        else:
            content += f"**⚠️ 需要调整**: 相关性下降{abs(summary['improvement_vs_v30']):.4f}\n\n"
            content += "**建议**: 需要进一步优化参数或增加数据量"
        
        content += "\n---\n\n🤖 *Generated by Simplified Real Data Qlib Optimizer*"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(content)


def main():
    optimizer = SimplifiedRealDataOptimizer()
    
    try:
        results = optimizer.run_optimization(max_evals=30)
        
        print("🎉 优化完成！")
        print(f"📊 最佳相关性: {results['optimization_summary']['best_correlation']:.4f}")
        print(f"📈 相对V3.0改进: {results['optimization_summary']['improvement_vs_v30']:+.4f}")
        print(f"📊 使用股票数: {results['optimization_summary']['stocks_used']}")
        
        print("\n🎯 最优权重配置:")
        for param, weight in sorted(results['best_weights'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {param}: {weight:.4f}")
        
    except Exception as e:
        print(f"❌ 优化失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()