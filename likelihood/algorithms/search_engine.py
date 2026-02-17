"""
相似度搜索引擎
集成多种算法的统一搜索接口
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional, Union
import logging
import sys
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from .matrix_profile import MatrixProfileSimilarity
from .dtw_similarity import DTWSimilarity  
from .mass_similarity import MASSimilarity
from data_preprocessing.data_loader import DataLoader

logger = logging.getLogger(__name__)


class SimilaritySearchEngine:
    """
    相似度搜索引擎
    
    集成多种相似度算法，提供统一的搜索接口，
    支持多算法融合、并行计算和结果排序。
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化搜索引擎
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.data_loader = None
        
        # 初始化算法
        self.algorithms = {}
        self._init_algorithms()
        
        # 设置权重
        self.algorithm_weights = self._get_algorithm_weights()
        
        # 并行设置
        self.parallel_workers = self.config.get('search', {}).get('parallel_workers', 4)
        self.use_multiprocessing = self.config.get('performance', {}).get('n_jobs', -1) != 1
        
        logger.info(f"搜索引擎初始化完成，启用算法: {list(self.algorithms.keys())}")
    
    def _init_algorithms(self):
        """初始化相似度算法"""
        similarity_config = self.config.get('similarity', {})
        algorithms_config = similarity_config.get('algorithms', {})
        
        # Matrix Profile
        if algorithms_config.get('matrix_profile', {}).get('enabled', True):
            mp_config = algorithms_config.get('matrix_profile', {})
            mp_config.update({
                'window_length': similarity_config.get('default_window', 30),
                'normalize': True
            })
            self.algorithms['matrix_profile'] = MatrixProfileSimilarity(mp_config)
        
        # DTW
        if algorithms_config.get('dtw', {}).get('enabled', True):
            dtw_config = algorithms_config.get('dtw', {})
            dtw_config.update({'normalize': True})
            self.algorithms['dtw'] = DTWSimilarity(dtw_config)
        
        # MASS
        if algorithms_config.get('mass', {}).get('enabled', True):
            mass_config = algorithms_config.get('mass', {})
            mass_config.update({'normalize': True})
            self.algorithms['mass'] = MASSimilarity(mass_config)
        
        if not self.algorithms:
            logger.warning("没有启用任何算法，使用默认Matrix Profile")
            self.algorithms['matrix_profile'] = MatrixProfileSimilarity()
    
    def _get_algorithm_weights(self) -> Dict[str, float]:
        """获取算法权重"""
        algorithms_config = self.config.get('similarity', {}).get('algorithms', {})
        weights = {}
        
        for alg_name in self.algorithms.keys():
            weight = algorithms_config.get(alg_name, {}).get('weight', 1.0)
            weights[alg_name] = weight
        
        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            for alg_name in weights:
                weights[alg_name] /= total_weight
        
        return weights
    
    def search_similar_patterns(self,
                              stock_code: str,
                              query_date: str,
                              window_length: int,
                              **kwargs) -> Dict[str, Any]:
        """
        搜索相似的股票走势模式
        
        Args:
            stock_code: 股票代码
            query_date: 查询日期
            window_length: 窗口长度
            **kwargs: 额外参数
            
        Returns:
            搜索结果字典
        """
        start_time = time.time()
        
        try:
            # 1. 加载数据
            if self.data_loader is None:
                self.data_loader = DataLoader(config=self.config)
            
            # 计算查询时间范围
            query_end_date = pd.to_datetime(query_date)
            query_start_date = query_end_date - pd.Timedelta(days=window_length + 10)  # 额外缓冲
            
            # 加载查询股票数据
            query_data = self.data_loader.load_stock_data(
                stock_code,
                query_start_date.strftime('%Y-%m-%d'),
                query_end_date.strftime('%Y-%m-%d')
            )
            
            if len(query_data) < window_length:
                return {
                    'status': 'error',
                    'message': f'数据不足，需要至少 {window_length} 天数据，实际只有 {len(query_data)} 天',
                    'query_stock': stock_code,
                    'query_date': query_date,
                    'window_length': window_length
                }
            
            # 2. 提取查询序列
            query_series = self._extract_query_series(query_data, window_length)
            
            # 3. 获取候选股票池
            candidate_stocks = self._get_candidate_stocks(stock_code, kwargs)
            
            # 4. 搜索相似模式
            similar_patterns = self._search_in_database(
                query_series, candidate_stocks, query_start_date, query_end_date, kwargs
            )
            
            # 5. 后处理和排序
            similar_patterns = self._post_process_results(similar_patterns, kwargs)
            
            # 6. 构建返回结果
            search_time = time.time() - start_time
            
            result = {
                'status': 'success',
                'query_stock': stock_code,
                'query_date': query_date,
                'window_length': window_length,
                'search_time': search_time,
                'similar_patterns': similar_patterns,
                'metadata': {
                    'algorithms_used': list(self.algorithms.keys()),
                    'algorithm_weights': self.algorithm_weights,
                    'candidate_stocks_count': len(candidate_stocks),
                    'total_matches_found': len(similar_patterns)
                }
            }
            
            logger.info(f"搜索完成: {stock_code}, 耗时 {search_time:.2f}s, 找到 {len(similar_patterns)} 个匹配")
            
            return result
            
        except Exception as e:
            logger.error(f"搜索失败: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e),
                'query_stock': stock_code,
                'query_date': query_date,
                'window_length': window_length
            }
    
    def _extract_query_series(self, query_data: pd.DataFrame, window_length: int) -> Dict[str, np.ndarray]:
        """提取查询序列"""
        # 获取最近的window_length天数据
        recent_data = query_data.tail(window_length)
        
        # 提取多维特征
        series = {}
        
        # 价格序列（对数收益率）
        if 'close' in recent_data.columns:
            close_prices = recent_data['close'].values
            if len(close_prices) > 1:
                log_returns = np.diff(np.log(close_prices))
                series['price'] = log_returns
        
        # 成交量序列
        if 'volume' in recent_data.columns and 'turnover_rate' in recent_data.columns:
            volume_data = recent_data['turnover_rate'].fillna(recent_data['volume']).values
            if len(volume_data) > 1:
                volume_change = np.diff(np.log(volume_data + 1e-8))  # 避免log(0)
                series['volume'] = volume_change
        
        # 技术指标序列
        tech_indicators = ['rsi6', 'kdj_k', 'macd_dif']
        for indicator in tech_indicators:
            if indicator in recent_data.columns:
                indicator_data = recent_data[indicator].ffill().values
                # 确保数据类型为float
                try:
                    indicator_data = indicator_data.astype(float)
                    if len(indicator_data) > 1 and not np.all(np.isnan(indicator_data)):
                        series[indicator] = indicator_data[1:]  # 与价格收益率对齐
                except (TypeError, ValueError) as e:
                    logger.debug(f"技术指标 {indicator} 数据类型转换失败: {str(e)}")
                    continue
        
        # 确保至少有价格数据
        if 'price' not in series or len(series['price']) == 0:
            raise ValueError("无法提取有效的价格序列")
        
        return series
    
    def _get_candidate_stocks(self, query_stock: str, kwargs: Dict[str, Any]) -> List[str]:
        """获取候选股票池"""
        filters_config = self.config.get('filters', {})
        
        # 使用数据加载器的筛选功能
        candidate_stocks = self.data_loader.filter_stocks_by_criteria(
            min_volume=filters_config.get('min_daily_volume'),
            min_market_cap=filters_config.get('min_market_cap'),
            industries=filters_config.get('include_industries'),
            date=kwargs.get('end_date')
        )
        
        # 排除自身
        if filters_config.get('exclude_self', True) and query_stock in candidate_stocks:
            candidate_stocks.remove(query_stock)
        
        # 限制候选数量（性能考虑）
        max_candidates = kwargs.get('max_candidates', 1000)
        if len(candidate_stocks) > max_candidates:
            # 随机采样
            np.random.seed(42)
            candidate_stocks = list(np.random.choice(candidate_stocks, max_candidates, replace=False))
        
        logger.info(f"候选股票池大小: {len(candidate_stocks)}")
        return candidate_stocks
    
    def _search_in_database(self,
                           query_series: Dict[str, np.ndarray],
                           candidate_stocks: List[str],
                           start_date: pd.Timestamp,
                           end_date: pd.Timestamp,
                           kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """在数据库中搜索相似模式"""
        search_config = self.config.get('search', {})
        top_k = search_config.get('top_k', 10)
        
        all_matches = []
        
        # 分批处理候选股票
        batch_size = kwargs.get('batch_size', 100)
        
        for i in range(0, len(candidate_stocks), batch_size):
            batch_stocks = candidate_stocks[i:i + batch_size]
            
            if self.use_multiprocessing and len(batch_stocks) > 10:
                # 并行处理
                batch_matches = self._parallel_search_batch(
                    query_series, batch_stocks, start_date, end_date, kwargs
                )
            else:
                # 串行处理
                batch_matches = self._sequential_search_batch(
                    query_series, batch_stocks, start_date, end_date, kwargs
                )
            
            all_matches.extend(batch_matches)
            
            logger.debug(f"已处理 {i + len(batch_stocks)}/{len(candidate_stocks)} 只股票")
        
        # 按综合相似度排序
        all_matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return all_matches[:top_k]
    
    def _sequential_search_batch(self,
                               query_series: Dict[str, np.ndarray],
                               batch_stocks: List[str],
                               start_date: pd.Timestamp,
                               end_date: pd.Timestamp,
                               kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """串行搜索批次"""
        matches = []
        
        for stock_code in batch_stocks:
            try:
                stock_matches = self._search_single_stock(
                    query_series, stock_code, start_date, end_date, kwargs
                )
                matches.extend(stock_matches)
            except Exception as e:
                logger.debug(f"搜索股票 {stock_code} 失败: {str(e)}")
                continue
        
        return matches
    
    def _parallel_search_batch(self,
                             query_series: Dict[str, np.ndarray],
                             batch_stocks: List[str],
                             start_date: pd.Timestamp,
                             end_date: pd.Timestamp,
                             kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """并行搜索批次"""
        matches = []
        
        # 使用进程池进行并行计算
        with ProcessPoolExecutor(max_workers=self.parallel_workers) as executor:
            futures = []
            
            for stock_code in batch_stocks:
                future = executor.submit(
                    self._search_single_stock_wrapper,
                    query_series, stock_code, start_date, end_date, kwargs, self.config
                )
                futures.append((stock_code, future))
            
            for stock_code, future in futures:
                try:
                    stock_matches = future.result(timeout=30)  # 30秒超时
                    matches.extend(stock_matches)
                except Exception as e:
                    logger.debug(f"并行搜索股票 {stock_code} 失败: {str(e)}")
                    continue
        
        return matches
    
    @staticmethod
    def _search_single_stock_wrapper(query_series: Dict[str, np.ndarray],
                                   stock_code: str,
                                   start_date: pd.Timestamp,
                                   end_date: pd.Timestamp,
                                   kwargs: Dict[str, Any],
                                   config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """单股票搜索包装器（用于多进程）"""
        # 在子进程中重新创建必要对象
        search_engine = SimilaritySearchEngine(config)
        return search_engine._search_single_stock(
            query_series, stock_code, start_date, end_date, kwargs
        )
    
    def _search_single_stock(self,
                           query_series: Dict[str, np.ndarray],
                           stock_code: str,
                           start_date: pd.Timestamp,
                           end_date: pd.Timestamp,
                           kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """搜索单只股票"""
        try:
            # 加载候选股票的历史数据（更长的时间范围用于搜索）
            search_start = start_date - pd.Timedelta(days=365)  # 搜索过去一年
            
            candidate_data = self.data_loader.load_stock_data(
                stock_code,
                search_start.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if len(candidate_data) < len(query_series['price']) + 10:
                return []
            
            # 提取候选时间序列
            candidate_series = self._extract_candidate_series(candidate_data, query_series)
            
            # 使用多种算法计算相似度
            matches = self._compute_similarities(
                query_series, candidate_series, stock_code, candidate_data
            )
            
            return matches
            
        except Exception as e:
            logger.debug(f"搜索股票 {stock_code} 失败: {str(e)}")
            return []
    
    def _extract_candidate_series(self, 
                                 candidate_data: pd.DataFrame,
                                 query_series: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """从候选数据中提取时间序列"""
        series = {}
        
        # 价格序列
        if 'close' in candidate_data.columns:
            close_prices = candidate_data['close'].values
            if len(close_prices) > 1:
                log_returns = np.diff(np.log(close_prices))
                series['price'] = log_returns
        
        # 成交量序列
        if 'volume' in candidate_data.columns and 'turnover_rate' in candidate_data.columns:
            volume_data = candidate_data['turnover_rate'].fillna(candidate_data['volume']).values
            if len(volume_data) > 1:
                volume_change = np.diff(np.log(volume_data + 1e-8))
                series['volume'] = volume_change
        
        # 技术指标序列  
        tech_indicators = ['rsi6', 'kdj_k', 'macd_dif']
        for indicator in tech_indicators:
            if indicator in candidate_data.columns and indicator in query_series:
                indicator_data = candidate_data[indicator].ffill().values
                try:
                    indicator_data = indicator_data.astype(float)
                    if len(indicator_data) > 1 and not np.all(np.isnan(indicator_data)):
                        series[indicator] = indicator_data[1:]
                except (TypeError, ValueError) as e:
                    logger.debug(f"候选技术指标 {indicator} 数据类型转换失败: {str(e)}")
                    continue
        
        return series
    
    def _compute_similarities(self,
                            query_series: Dict[str, np.ndarray],
                            candidate_series: Dict[str, np.ndarray],
                            stock_code: str,
                            candidate_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """计算相似度"""
        matches = []
        
        if 'price' not in candidate_series or len(candidate_series['price']) < len(query_series['price']):
            return matches
        
        query_length = len(query_series['price'])
        candidate_length = len(candidate_series['price'])
        
        # 滑动窗口搜索
        for i in range(candidate_length - query_length + 1):
            try:
                # 提取候选窗口
                candidate_window = {}
                for feature_name in query_series.keys():
                    if feature_name in candidate_series:
                        candidate_window[feature_name] = candidate_series[feature_name][i:i + query_length]
                
                # 计算综合相似度
                similarity_scores = {}
                
                for alg_name, algorithm in self.algorithms.items():
                    try:
                        if len(candidate_window.get('price', [])) == len(query_series['price']):
                            # 主要基于价格计算相似度
                            similarity = algorithm.compute_similarity(
                                query_series['price'], candidate_window['price']
                            )
                            similarity_scores[alg_name] = similarity
                    except Exception as e:
                        logger.debug(f"算法 {alg_name} 计算失败: {str(e)}")
                        continue
                
                if not similarity_scores:
                    continue
                
                # 加权平均
                weighted_similarity = sum(
                    self.algorithm_weights.get(alg_name, 0) * score
                    for alg_name, score in similarity_scores.items()
                )
                
                # 构建匹配结果
                match_start_date = candidate_data.index[i]
                match_end_date = candidate_data.index[i + query_length - 1]
                
                match = {
                    'stock': stock_code,
                    'period_start': match_start_date.strftime('%Y-%m-%d'),
                    'period_end': match_end_date.strftime('%Y-%m-%d'),
                    'similarity_score': weighted_similarity,
                    'algorithm_scores': similarity_scores,
                    'position': i
                }
                
                matches.append(match)
                
            except Exception as e:
                logger.debug(f"计算位置 {i} 相似度失败: {str(e)}")
                continue
        
        # 只保留该股票的最佳匹配（前3个）
        matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        return matches[:3]
    
    def _post_process_results(self, 
                            similar_patterns: List[Dict[str, Any]],
                            kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """后处理搜索结果"""
        min_similarity = self.config.get('search', {}).get('min_similarity', 0.0)
        
        # 过滤低相似度结果
        filtered_patterns = [
            pattern for pattern in similar_patterns
            if pattern['similarity_score'] >= min_similarity
        ]
        
        # 添加排名
        for i, pattern in enumerate(filtered_patterns):
            pattern['rank'] = i + 1
        
        return filtered_patterns


if __name__ == '__main__':
    # 测试代码
    import yaml
    
    # 加载配置
    config_path = Path(__file__).parent.parent / 'configs' / 'default_config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建搜索引擎
    search_engine = SimilaritySearchEngine(config)
    
    # 测试搜索
    result = search_engine.search_similar_patterns(
        stock_code='000001',
        query_date='2025-08-08',
        window_length=20
    )
    
    print(f"搜索状态: {result['status']}")
    if result['status'] == 'success':
        print(f"搜索耗时: {result['search_time']:.2f}s")
        print(f"找到 {len(result['similar_patterns'])} 个相似模式")
        
        for pattern in result['similar_patterns'][:3]:
            print(f"\n第 {pattern['rank']} 名:")
            print(f"  股票: {pattern['stock']}")
            print(f"  时期: {pattern['period_start']} 至 {pattern['period_end']}")
            print(f"  相似度: {pattern['similarity_score']:.4f}")
            print(f"  算法分数: {pattern['algorithm_scores']}")
    else:
        print(f"搜索失败: {result['message']}")