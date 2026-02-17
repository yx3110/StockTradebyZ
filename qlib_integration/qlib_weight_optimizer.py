#!/usr/bin/env python3
"""
Qlib权重优化器 - 针对V3.5评分系统的智能权重优化

基于Qlib的Hyperopt框架，对V3.5评分系统的权重参数进行自动优化，
目标是修复V3.5相关性逆转问题，恢复到V3.0的正相关水平。

核心问题：
- V3.5相关性从V3.0的正相关(+0.05~+0.09)变为负相关(-0.02~-0.04)  
- 评分分布异常：标准差从12.94降至9.73，区分度不足
- 高分股票样本偏少：90+分股票样本为0

优化策略：
1. 多目标优化：最大化相关性 + 优化评分分布 + 控制风险
2. 约束优化：确保权重和为1，各权重非负
3. A/B测试：与V3.0基线对比验证
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # 上一级目录是项目根目录
sys.path.append(project_root)

# 导入hyperopt
try:
    from hyperopt import hp, fmin, tpe, STATUS_OK, STATUS_FAIL, Trials
except ImportError:
    print("请安装hyperopt: pip3 install hyperopt")
    sys.exit(1)

# 导入项目模块
from data_adapter.database_manager import DatabaseManager
# 暂时不导入，在需要时动态导入


class QlibWeightOptimizer:
    """Qlib权重优化器 - 专门针对V3.5评分系统"""
    
    def __init__(self, optimization_period_days: int = 60):
        """
        初始化优化器
        
        Args:
            optimization_period_days: 优化使用的历史数据天数
        """
        self.db_manager = DatabaseManager()
        self.optimization_period_days = optimization_period_days
        self.logger = self._setup_logging()
        
        # 缓存历史数据
        self.historical_data_cache = {}
        self.future_returns_cache = {}
        
        # 优化结果存储
        self.trials = Trials()
        self.best_weights = None
        self.optimization_results = {}
        
        # V3.0基线数据用于对比
        self.v30_baseline_correlation = 0.065  # V3.0平均相关性基线
        
        self.logger.info("🚀 Qlib权重优化器已初始化")
        
    def _setup_logging(self) -> logging.Logger:
        """设置日志系统"""
        logger = logging.getLogger("QlibWeightOptimizer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # 文件日志
            os.makedirs("logs", exist_ok=True)
            file_handler = logging.FileHandler(
                f"logs/qlib_weight_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                encoding='utf-8'
            )
            
            # 控制台日志
            console_handler = logging.StreamHandler()
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
        
    def prepare_optimization_data(self, end_date: str = None) -> Tuple[Dict, Dict]:
        """
        准备优化用的历史数据
        
        Args:
            end_date: 结束日期，默认为最新交易日
            
        Returns:
            Tuple[历史特征数据, 未来收益数据]
        """
        if end_date is None:
            # 获取最新交易日
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT MAX(trade_date) 
                    FROM daily_quotes 
                    WHERE trade_date <= date('now')
                """)
                end_date = cursor.fetchone()[0]
        
        self.logger.info(f"📊 准备优化数据，结束日期: {end_date}")
        
        # 计算开始日期
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        start_dt = end_dt - timedelta(days=self.optimization_period_days + 30)  # 多留30天用于计算指标
        start_date = start_dt.strftime('%Y-%m-%d')
        
        # 获取活跃股票列表（排除停牌、退市等）
        active_stocks = self._get_active_stocks(end_date)
        self.logger.info(f"📈 获取到{len(active_stocks)}只活跃股票")
        
        # 批量获取历史数据
        historical_features = {}
        future_returns = {}
        
        processed_count = 0
        for stock_code in active_stocks[:1000]:  # 限制1000只股票避免内存问题
            try:
                # 获取历史价格数据
                stock_data = self._get_stock_data(stock_code, start_date, end_date)
                if len(stock_data) < 30:  # 至少需要30天数据
                    continue
                
                # 计算各种技术指标特征
                features = self._calculate_technical_features(stock_data)
                if features is None:
                    continue
                
                # 计算未来收益（1日、3日、5日、10日）
                returns = self._calculate_future_returns(stock_data, stock_code, end_date)
                if returns is None:
                    continue
                
                historical_features[stock_code] = features
                future_returns[stock_code] = returns
                
                processed_count += 1
                if processed_count % 100 == 0:
                    self.logger.info(f"✅ 已处理 {processed_count} 只股票")
                    
            except Exception as e:
                self.logger.warning(f"❌ 处理股票 {stock_code} 时出错: {str(e)}")
                continue
        
        self.logger.info(f"🎯 数据准备完成，共处理 {len(historical_features)} 只股票")
        
        # 缓存数据
        self.historical_data_cache = historical_features
        self.future_returns_cache = future_returns
        
        return historical_features, future_returns
    
    def _get_active_stocks(self, end_date: str) -> List[str]:
        """获取活跃股票列表"""
        with self.db_manager.get_connection() as conn:
            query = """
                SELECT DISTINCT s.code 
                FROM securities s
                JOIN daily_quotes dq ON s.id = dq.security_id
                WHERE s.type = 'A股'
                  AND dq.trade_date = ?
                  AND dq.volume > 0
                  AND dq.close > 0
                  AND s.code NOT LIKE '%ST%'
                  AND s.code NOT LIKE '%st%'
                ORDER BY dq.volume DESC
            """
            cursor = conn.execute(query, (end_date,))
            return [row[0] for row in cursor.fetchall()]
    
    def _get_stock_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取股票历史数据"""
        with self.db_manager.get_connection() as conn:
            query = """
                SELECT dq.trade_date, dq.open, dq.high, dq.low, dq.close, 
                       dq.volume, dq.price_change_pct,
                       db.pe_ttm, db.pb, db.total_mv as market_cap, db.turnover_rate
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                LEFT JOIN daily_basic db ON s.id = db.security_id AND dq.trade_date = db.trade_date
                WHERE s.code = ?
                  AND dq.trade_date BETWEEN ? AND ?
                ORDER BY dq.trade_date ASC
            """
            df = pd.read_sql_query(query, conn, params=(stock_code, start_date, end_date))
            return df
    
    def _calculate_technical_features(self, stock_data: pd.DataFrame) -> Optional[Dict]:
        """计算技术指标特征"""
        try:
            if len(stock_data) < 20:
                return None
            
            close_prices = stock_data['close'].values
            volume = stock_data['volume'].values
            high_prices = stock_data['high'].values
            low_prices = stock_data['low'].values
            
            # 计算各种技术指标
            features = {}
            
            # KDJ指标
            kdj_data = self._calculate_kdj(high_prices, low_prices, close_prices)
            features['kdj_k'] = kdj_data['K'][-1] if len(kdj_data['K']) > 0 else 50
            features['kdj_d'] = kdj_data['D'][-1] if len(kdj_data['D']) > 0 else 50
            features['kdj_j'] = kdj_data['J'][-1] if len(kdj_data['J']) > 0 else 50
            
            # RSI指标
            features['rsi'] = self._calculate_rsi(close_prices)
            
            # BBI指标
            features['bbi'] = self._calculate_bbi(close_prices)
            
            # 知行指标
            features['zhixing_trend'] = self._calculate_zhixing_trend(close_prices)
            features['zhixing_multiavg'] = self._calculate_zhixing_multiavg(close_prices)
            
            # 成交量指标
            features['volume_surge'] = self._calculate_volume_surge(volume)
            
            # 价格动量
            features['price_momentum'] = (close_prices[-1] / close_prices[-10] - 1) if len(close_prices) >= 10 else 0
            
            # 波动率
            returns = np.diff(close_prices) / close_prices[:-1]
            features['volatility'] = np.std(returns[-20:]) if len(returns) >= 20 else 0
            
            # 当前价格（用于趋势判断）
            features['close'] = close_prices[-1]
            
            # 基本面数据
            if len(stock_data) > 0 and 'pe_ttm' in stock_data.columns and not pd.isna(stock_data['pe_ttm'].iloc[-1]):
                features['pe_ttm'] = stock_data['pe_ttm'].iloc[-1]
            else:
                features['pe_ttm'] = 30.0  # 默认值
            
            if len(stock_data) > 0 and 'pb' in stock_data.columns and not pd.isna(stock_data['pb'].iloc[-1]):
                features['pb'] = stock_data['pb'].iloc[-1]  
            else:
                features['pb'] = 2.0  # 默认值
                
            return features
            
        except Exception as e:
            return None
    
    def _calculate_kdj(self, high: np.array, low: np.array, close: np.array, n: int = 9) -> Dict:
        """计算KDJ指标"""
        rsv = np.zeros_like(close)
        K = np.zeros_like(close)
        D = np.zeros_like(close)
        
        for i in range(len(close)):
            if i == 0:
                K[i] = D[i] = 50.0
            else:
                # 计算RSV
                if i >= n-1:
                    high_n = np.max(high[i-n+1:i+1])
                    low_n = np.min(low[i-n+1:i+1])
                    rsv[i] = (close[i] - low_n) / (high_n - low_n + 1e-9) * 100
                else:
                    rsv[i] = 50.0
                
                # 计算K, D
                K[i] = 2/3 * K[i-1] + 1/3 * rsv[i]
                D[i] = 2/3 * D[i-1] + 1/3 * K[i]
        
        J = 3 * K - 2 * D
        return {'K': K, 'D': D, 'J': J}
    
    def _calculate_rsi(self, close: np.array, n: int = 14) -> float:
        """计算RSI指标"""
        if len(close) < n + 1:
            return 50.0
            
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.mean(gain[-n:])
        avg_loss = np.mean(loss[-n:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bbi(self, close: np.array) -> float:
        """计算BBI指标"""
        if len(close) < 24:
            return close[-1]
            
        ma3 = np.mean(close[-3:])
        ma6 = np.mean(close[-6:])  
        ma12 = np.mean(close[-12:])
        ma24 = np.mean(close[-24:])
        
        return (ma3 + ma6 + ma12 + ma24) / 4
    
    def _calculate_zhixing_trend(self, close: np.array) -> float:
        """计算知行趋势线: EMA(EMA(C,10),10)"""
        if len(close) < 20:
            return close[-1]
            
        # 第一层EMA(C, 10)
        close_series = pd.Series(close)
        ema1 = close_series.ewm(span=10).mean().values
        
        # 第二层EMA(EMA1, 10)
        ema1_series = pd.Series(ema1)
        ema2 = ema1_series.ewm(span=10).mean().values
        
        return ema2[-1]
    
    def _calculate_zhixing_multiavg(self, close: np.array) -> float:
        """计算知行多空线: (MA(C,M1)+MA(C,M2)+MA(C,M3)+MA(C,M4))/4"""
        periods = [5, 10, 20, 60]
        mas = []
        
        for period in periods:
            if len(close) >= period:
                mas.append(np.mean(close[-period:]))
            else:
                mas.append(close[-1])
        
        return np.mean(mas)
    
    def _calculate_volume_surge(self, volume: np.array) -> float:
        """计算成交量激增指标"""
        if len(volume) < 10:
            return 1.0
            
        current_vol = volume[-1]
        avg_vol = np.mean(volume[-10:-1])
        
        return current_vol / (avg_vol + 1e-9)
    
    def _calculate_future_returns(self, stock_data: pd.DataFrame, stock_code: str, end_date: str) -> Optional[Dict]:
        """计算未来收益"""
        try:
            # 获取未来收益数据
            with self.db_manager.get_connection() as conn:
                query = """
                    SELECT dq.trade_date, dq.close
                    FROM daily_quotes dq
                    JOIN securities s ON dq.security_id = s.id
                    WHERE s.code = ?
                      AND dq.trade_date > ?
                      AND dq.trade_date <= date(?, '+30 days')
                    ORDER BY dq.trade_date ASC
                    LIMIT 30
                """
                future_data = pd.read_sql_query(query, conn, params=(stock_code, end_date, end_date))
            
            if len(future_data) == 0:
                return None
            
            current_price = stock_data['close'].iloc[-1]
            returns = {}
            
            # 计算1日、3日、5日、10日收益
            for days in [1, 3, 5, 10]:
                if len(future_data) >= days:
                    future_price = future_data.iloc[days-1]['close']
                    returns[f'return_{days}d'] = (future_price / current_price - 1)
                else:
                    returns[f'return_{days}d'] = 0.0
            
            return returns
            
        except Exception as e:
            return None
    
    def setup_optimization_space(self) -> Dict:
        """设置优化空间 - 定义需要优化的权重参数"""
        
        # V3.5权重优化空间 - 基于原始配置进行调整
        space = {
            # 技术指标权重 (总和约0.6)
            'kdj_strength': hp.uniform('kdj_strength', 0.08, 0.18),           # 原0.12, 范围±50%
            'rsi_momentum': hp.uniform('rsi_momentum', 0.06, 0.16),           # 原0.10, 范围±60%
            'bbi_trend': hp.uniform('bbi_trend', 0.04, 0.14),                 # 原0.08, 范围±75%  
            'volume_surge': hp.uniform('volume_surge', 0.06, 0.16),           # 原0.10, 范围±60%
            'zhixing_trend': hp.uniform('zhixing_trend', 0.06, 0.20),         # 原0.12, 范围±67%
            'zhixing_multiavg': hp.uniform('zhixing_multiavg', 0.04, 0.14),   # 原0.08, 范围±75%
            
            # 基本面权重 (总和约0.14)
            'pe_valuation': hp.uniform('pe_valuation', 0.01, 0.04),           # 原0.025, 范围±60%
            'pb_valuation': hp.uniform('pb_valuation', 0.01, 0.04),           # 原0.025, 范围±60%
            'roe_profitability': hp.uniform('roe_profitability', 0.01, 0.04), # 原0.025, 范围±60%
            'market_cap': hp.uniform('market_cap', 0.01, 0.04),               # 原0.025, 范围±60%
            
            # 市场表现权重 (总和约0.13)  
            'price_momentum': hp.uniform('price_momentum', 0.04, 0.14),       # 原0.08, 范围±75%
            'volatility_risk': hp.uniform('volatility_risk', 0.01, 0.05),     # 原0.02, 范围±150%
            
            # 风险惩罚系数
            'risk_penalty': hp.uniform('risk_penalty', 0.001, 0.1),           # 新增风险控制参数
            
            # 评分分布优化参数
            'score_spread_bonus': hp.uniform('score_spread_bonus', 0.0, 0.2), # 奖励评分分散度
        }
        
        return space
    
    def objective_function(self, params: Dict) -> Dict:
        """
        优化目标函数
        
        多目标优化：
        1. 最大化与未来收益的相关性 (主要目标)
        2. 优化评分分布质量 (标准差、范围)
        3. 控制过拟合风险
        
        Args:
            params: 权重参数字典
            
        Returns:
            Dict: hyperopt格式的结果
        """
        try:
            self.logger.info("🎯 开始评估权重参数组合")
            
            # 1. 归一化权重 (确保和为1)
            normalized_weights = self._normalize_weights(params)
            
            # 2. 计算所有股票的评分
            scores = {}
            for stock_code, features in self.historical_data_cache.items():
                score = self._calculate_weighted_score(features, normalized_weights)
                scores[stock_code] = score
            
            if len(scores) == 0:
                return {'loss': float('inf'), 'status': STATUS_FAIL}
            
            # 3. 计算相关性 (多个时间期限)
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
            
            # 4. 计算评分分布质量
            score_values = list(scores.values())
            score_std = np.std(score_values)
            score_range = np.ptp(score_values)  # peak-to-peak range
            score_mean = np.mean(score_values)
            
            # 5. 计算高分股票比例 (解决90+分样本为0的问题)
            high_score_ratio = sum(1 for s in score_values if s >= 85) / len(score_values)
            
            # 6. 多目标优化函数
            # 主要目标：相关性 (权重70%)
            correlation_score = (
                0.4 * correlations.get('1d', 0) +
                0.3 * correlations.get('3d', 0) +
                0.2 * correlations.get('5d', 0) +
                0.1 * correlations.get('10d', 0)
            )
            
            # 分布质量 (权重20%)
            distribution_score = (
                0.1 * min(score_std / 15.0, 1.0) +         # 标准差目标15
                0.05 * min(score_range / 80.0, 1.0) +      # 范围目标80
                0.05 * min(high_score_ratio / 0.1, 1.0)    # 高分比例目标10%
            )
            
            # 风险控制 (权重10%)
            risk_penalty = params.get('risk_penalty', 0.01)
            volatility_penalty = np.std(score_values) * risk_penalty
            
            # 评分分散奖励
            spread_bonus = params.get('score_spread_bonus', 0.0) * (score_std / 20.0)
            
            # 综合目标函数 (注意：hyperopt寻找最小值，所以用负号)
            objective_value = -(
                0.7 * correlation_score +      # 相关性 70%
                0.2 * distribution_score +     # 分布质量 20%
                0.05 * spread_bonus -          # 分散奖励 5%
                0.05 * volatility_penalty      # 风险惩罚 5%
            )
            
            # 记录详细信息
            detailed_info = {
                'correlations': correlations,
                'correlation_score': correlation_score,
                'distribution_score': distribution_score,
                'score_stats': {
                    'mean': score_mean,
                    'std': score_std,
                    'range': score_range,
                    'high_score_ratio': high_score_ratio
                },
                'objective_components': {
                    'correlation_part': 0.7 * correlation_score,
                    'distribution_part': 0.2 * distribution_score,
                    'spread_bonus': 0.05 * spread_bonus,
                    'risk_penalty': 0.05 * volatility_penalty
                },
                'weights': normalized_weights,
                'total_objective': -objective_value
            }
            
            self.logger.info(f"📊 相关性: {correlation_score:.4f}, 分布质量: {distribution_score:.4f}, 目标函数: {-objective_value:.4f}")
            
            return {
                'loss': objective_value, 
                'status': STATUS_OK,
                'eval_time': datetime.now(),
                'detailed_info': detailed_info
            }
            
        except Exception as e:
            self.logger.error(f"❌ 目标函数计算出错: {str(e)}")
            return {'loss': float('inf'), 'status': STATUS_FAIL}
    
    def _normalize_weights(self, params: Dict) -> Dict:
        """归一化权重确保和为1"""
        # 提取所有权重参数 (排除非权重参数)
        weight_params = {k: v for k, v in params.items() 
                        if k not in ['risk_penalty', 'score_spread_bonus']}
        
        total_weight = sum(weight_params.values())
        
        # 归一化
        if total_weight > 0:
            normalized = {k: v / total_weight for k, v in weight_params.items()}
        else:
            # 如果总和为0，使用均匀分布
            n = len(weight_params)
            normalized = {k: 1.0 / n for k in weight_params.keys()}
        
        return normalized
    
    def _calculate_weighted_score(self, features: Dict, weights: Dict) -> float:
        """计算加权评分"""
        score = 0.0
        
        # 技术指标评分
        kdj_score = self._score_kdj(features.get('kdj_k', 50), features.get('kdj_d', 50), features.get('kdj_j', 50))
        score += weights.get('kdj_strength', 0) * kdj_score
        
        rsi_score = self._score_rsi(features.get('rsi', 50))
        score += weights.get('rsi_momentum', 0) * rsi_score
        
        bbi_score = self._score_bbi_trend(features)
        score += weights.get('bbi_trend', 0) * bbi_score
        
        volume_score = self._score_volume_surge(features.get('volume_surge', 1.0))
        score += weights.get('volume_surge', 0) * volume_score
        
        # 知行指标评分
        zhixing_trend_score = self._score_zhixing_trend(features)
        score += weights.get('zhixing_trend', 0) * zhixing_trend_score
        
        zhixing_multiavg_score = self._score_zhixing_multiavg(features)
        score += weights.get('zhixing_multiavg', 0) * zhixing_multiavg_score
        
        # 基本面评分
        pe_score = self._score_pe_valuation(features.get('pe_ttm', 30))
        score += weights.get('pe_valuation', 0) * pe_score
        
        pb_score = self._score_pb_valuation(features.get('pb', 2.0))
        score += weights.get('pb_valuation', 0) * pb_score
        
        # 市场表现评分
        momentum_score = self._score_price_momentum(features.get('price_momentum', 0))
        score += weights.get('price_momentum', 0) * momentum_score
        
        volatility_score = self._score_volatility_risk(features.get('volatility', 0))
        score += weights.get('volatility_risk', 0) * volatility_score
        
        # 将评分标准化到0-100范围
        return max(0, min(100, score * 100))
    
    def _score_kdj(self, k: float, d: float, j: float) -> float:
        """KDJ指标评分"""
        # 低位金叉给高分
        if k < 30 and d < 30 and k > d:
            return 0.9
        elif k < 50 and k > d:
            return 0.7
        elif k > 80:
            return 0.2  # 超买区域给低分
        else:
            return 0.5
    
    def _score_rsi(self, rsi: float) -> float:
        """RSI指标评分"""
        if rsi < 30:
            return 0.9  # 超卖
        elif rsi < 50:
            return 0.7
        elif rsi > 70:
            return 0.2  # 超买
        else:
            return 0.5
    
    def _score_bbi_trend(self, features: Dict) -> float:
        """BBI趋势评分 - 简化版本"""
        bbi = features.get('bbi', features.get('close', 0))
        close = features.get('close', bbi)
        
        if close > bbi * 1.02:
            return 0.8  # 价格在BBI上方
        elif close > bbi:
            return 0.6
        else:
            return 0.3
    
    def _score_volume_surge(self, volume_ratio: float) -> float:
        """成交量激增评分"""
        if volume_ratio > 3.0:
            return 0.9
        elif volume_ratio > 2.0:
            return 0.7
        elif volume_ratio > 1.5:
            return 0.6
        else:
            return 0.4
    
    def _score_zhixing_trend(self, features: Dict) -> float:
        """知行趋势线评分"""
        trend = features.get('zhixing_trend', features.get('close', 0))
        close = features.get('close', trend)
        
        if close > trend * 1.01:
            return 0.8
        elif close > trend:
            return 0.6
        else:
            return 0.3
    
    def _score_zhixing_multiavg(self, features: Dict) -> float:
        """知行多空线评分"""
        multiavg = features.get('zhixing_multiavg', features.get('close', 0))
        close = features.get('close', multiavg)
        
        if close > multiavg * 1.01:
            return 0.8
        elif close > multiavg:
            return 0.6
        else:
            return 0.3
    
    def _score_pe_valuation(self, pe: float) -> float:
        """PE估值评分"""
        if pe < 15:
            return 0.8
        elif pe < 25:
            return 0.6
        elif pe < 40:
            return 0.4
        else:
            return 0.2
    
    def _score_pb_valuation(self, pb: float) -> float:
        """PB估值评分"""
        if pb < 1.0:
            return 0.8
        elif pb < 2.0:
            return 0.6
        elif pb < 3.0:
            return 0.4
        else:
            return 0.2
    
    def _score_price_momentum(self, momentum: float) -> float:
        """价格动量评分"""
        if momentum > 0.1:
            return 0.8
        elif momentum > 0.05:
            return 0.6
        elif momentum > 0:
            return 0.5
        else:
            return 0.3
    
    def _score_volatility_risk(self, volatility: float) -> float:
        """波动率风险评分 (低波动率给高分)"""
        if volatility < 0.02:
            return 0.8
        elif volatility < 0.03:
            return 0.6
        elif volatility < 0.05:
            return 0.4
        else:
            return 0.2
    
    def run_optimization(self, max_evals: int = 100) -> Dict:
        """
        运行权重优化
        
        Args:
            max_evals: 最大评估次数
            
        Returns:
            优化结果字典
        """
        self.logger.info(f"🚀 开始Qlib权重优化，最大评估次数: {max_evals}")
        
        # 确保数据已准备
        if not self.historical_data_cache:
            self.logger.info("📊 准备优化数据...")
            self.prepare_optimization_data()
        
        # 设置优化空间
        space = self.setup_optimization_space()
        
        # 运行优化
        self.logger.info("🎯 开始超参数优化...")
        best = fmin(
            fn=self.objective_function,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=self.trials,
            verbose=True
        )
        
        self.best_weights = self._normalize_weights(best)
        
        # 分析结果
        self.logger.info("📊 分析优化结果...")
        results = self._analyze_optimization_results()
        
        # 保存结果
        self._save_optimization_results(results)
        
        self.logger.info("✅ 权重优化完成！")
        
        return results
    
    def _analyze_optimization_results(self) -> Dict:
        """分析优化结果"""
        
        if not self.best_weights:
            return {}
        
        # 获取最佳试验的详细信息
        best_trial = min(self.trials.trials, key=lambda x: x['result']['loss'])
        best_detailed_info = best_trial['result'].get('detailed_info', {})
        
        # 计算V3.0基线对比
        current_correlation = best_detailed_info.get('correlation_score', 0)
        improvement = current_correlation - self.v30_baseline_correlation
        
        results = {
            'optimization_summary': {
                'total_trials': len(self.trials.trials),
                'best_loss': best_trial['result']['loss'],
                'best_correlation': current_correlation,
                'v30_baseline_correlation': self.v30_baseline_correlation,
                'improvement_vs_v30': improvement,
                'optimization_time': datetime.now().isoformat()
            },
            'best_weights': self.best_weights,
            'detailed_analysis': best_detailed_info,
            'convergence_history': [
                {
                    'trial': i,
                    'loss': trial['result']['loss'],
                    'correlation': trial['result'].get('detailed_info', {}).get('correlation_score', 0)
                }
                for i, trial in enumerate(self.trials.trials)
            ]
        }
        
        return results
    
    def _save_optimization_results(self, results: Dict):
        """保存优化结果"""
        
        # 创建报告目录
        reports_dir = Path("reports/qlib_optimization")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果
        results_file = reports_dir / f"qlib_weight_optimization_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # 生成报告
        report_file = reports_dir / f"权重优化报告_{timestamp}.md"
        self._generate_optimization_report(results, report_file)
        
        self.logger.info(f"📄 优化结果已保存至: {results_file}")
        self.logger.info(f"📊 优化报告已生成: {report_file}")
    
    def _generate_optimization_report(self, results: Dict, report_file: Path):
        """生成优化报告"""
        
        summary = results['optimization_summary']
        weights = results['best_weights']
        analysis = results['detailed_analysis']
        
        report_content = f"""# Qlib权重优化报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 优化概览

### 核心指标
- **总优化轮次**: {summary['total_trials']}
- **最佳相关性**: {summary['best_correlation']:.4f}
- **V3.0基线**: {summary['v30_baseline_correlation']:.4f} 
- **相对改进**: {summary['improvement_vs_v30']:+.4f}
- **改进幅度**: {(summary['improvement_vs_v30']/summary['v30_baseline_correlation']*100):+.1f}%

### 相关性分析
"""

        if 'correlations' in analysis:
            corrs = analysis['correlations']
            report_content += f"""
- **1日收益相关性**: {corrs.get('1d', 0):.4f}
- **3日收益相关性**: {corrs.get('3d', 0):.4f}  
- **5日收益相关性**: {corrs.get('5d', 0):.4f}
- **10日收益相关性**: {corrs.get('10d', 0):.4f}
"""

        if 'score_stats' in analysis:
            stats = analysis['score_stats']
            report_content += f"""
### 评分分布质量
- **评分均值**: {stats.get('mean', 0):.2f}
- **评分标准差**: {stats.get('std', 0):.2f}
- **评分范围**: {stats.get('range', 0):.2f}
- **高分股票比例**: {stats.get('high_score_ratio', 0)*100:.1f}%
"""

        report_content += f"""
## 🎯 最优权重配置

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
- **ROE盈利**: {weights.get('roe_profitability', 0):.3f}
- **市值因子**: {weights.get('market_cap', 0):.3f}

### 市场表现权重
- **价格动量**: {weights.get('price_momentum', 0):.3f}
- **波动风险**: {weights.get('volatility_risk', 0):.3f}

## 📈 优化收敛过程

前10轮优化结果:
"""

        convergence = results.get('convergence_history', [])
        for i, trial in enumerate(convergence[:10]):
            report_content += f"- 第{trial['trial']+1}轮: 相关性={trial['correlation']:.4f}, 损失={trial['loss']:.4f}\n"

        if 'objective_components' in analysis:
            components = analysis['objective_components']
            report_content += f"""
## 🔍 目标函数组成分析

- **相关性部分**: {components.get('correlation_part', 0):.4f} (70%权重)
- **分布质量部分**: {components.get('distribution_part', 0):.4f} (20%权重)  
- **分散度奖励**: {components.get('spread_bonus', 0):.4f} (5%权重)
- **风险惩罚**: {components.get('risk_penalty', 0):.4f} (5%权重)
- **最终目标值**: {components.get('total_objective', 0):.4f}
"""

        report_content += f"""
## ✅ 结论与建议

### 优化效果评估
"""
        
        if summary['improvement_vs_v30'] > 0:
            report_content += f"✅ **显著改进**: 相关性相对V3.0基线提升{summary['improvement_vs_v30']:.4f}，改进幅度{(summary['improvement_vs_v30']/summary['v30_baseline_correlation']*100):+.1f}%"
        else:
            report_content += f"⚠️ **需要调整**: 相关性相对V3.0基线下降{abs(summary['improvement_vs_v30']):.4f}，需要进一步优化"

        report_content += f"""

### 部署建议
1. **渐进式部署**: 建议先进行A/B测试验证
2. **监控指标**: 重点关注相关性和评分分布  
3. **回滚方案**: 保持V3.0权重作为备份

---

🤖 *Generated by Qlib Weight Optimizer*
"""

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)


def main():
    """主函数 - 演示用法"""
    
    print("🚀 启动Qlib权重优化器")
    
    # 创建优化器
    optimizer = QlibWeightOptimizer()
    
    # 准备数据
    print("📊 准备优化数据...")
    optimizer.prepare_optimization_data()
    
    # 运行优化
    print("🎯 开始权重优化...")
    results = optimizer.run_optimization(max_evals=50)  # 演示用途，使用较少轮次
    
    print("✅ 优化完成！")
    print(f"📊 最佳相关性: {results['optimization_summary']['best_correlation']:.4f}")
    print(f"📈 相对V3.0改进: {results['optimization_summary']['improvement_vs_v30']:+.4f}")


if __name__ == "__main__":
    main()