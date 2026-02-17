"""
MASS (Mueen's Algorithm for Similarity Search) 相似度算法实现
快速的时间序列子序列搜索算法
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging
from .base_similarity import BaseSimilarityAlgorithm
from scipy import signal
import warnings

logger = logging.getLogger(__name__)


class MASSimilarity(BaseSimilarityAlgorithm):
    """
    MASS (Mueen's Algorithm for Similarity Search)
    
    MASS是一种高效的时间序列子序列搜索算法，
    使用FFT实现O(n log n)时间复杂度的相似度搜索。
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化MASS算法
        
        Args:
            config: 配置参数
                - normalize: 是否标准化
                - threshold: 相似度阈值
        """
        super().__init__(config)
        self.normalize = self.config.get('normalize', True)
        self.threshold = self.config.get('threshold', 0.5)
    
    def compute_similarity(self,
                          query_series: np.ndarray,
                          candidate_series: np.ndarray,
                          **kwargs) -> float:
        """
        计算两个时间序列的MASS相似度
        
        Args:
            query_series: 查询序列
            candidate_series: 候选序列
            
        Returns:
            相似度分数 (0-1，越高越相似)
        """
        if not self.validate_input(query_series, candidate_series):
            return 0.0
        
        try:
            # 使用查询序列在候选序列中搜索
            distances = self.mass_search(query_series, candidate_series)
            
            if len(distances) == 0:
                return 0.0
            
            # 找到最小距离
            min_distance = np.min(distances[np.isfinite(distances)])
            
            # 转换为相似度
            similarity = 1.0 / (1.0 + min_distance)
            
            return similarity
            
        except Exception as e:
            logger.error(f"MASS计算失败: {str(e)}")
            return self._fallback_similarity(query_series, candidate_series)
    
    def search_similar(self,
                      query_series: np.ndarray,
                      database_series: np.ndarray,
                      top_k: int = 10,
                      **kwargs) -> List[Tuple[int, float]]:
        """
        在数据库中搜索最相似的序列
        
        Args:
            query_series: 查询序列
            database_series: 数据库序列 (2D array)
            top_k: 返回最相似的K个结果
            
        Returns:
            List of (index, similarity_score)
        """
        if not self.validate_input(query_series):
            return []
        
        results = []
        
        # 遍历数据库中的每个序列
        for i, candidate in enumerate(database_series):
            if len(candidate) < len(query_series):
                continue
                
            similarity = self.compute_similarity(query_series, candidate)
            results.append((i, similarity))
        
        # 按相似度降序排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def mass_search(self, query: np.ndarray, timeseries: np.ndarray) -> np.ndarray:
        """
        MASS算法核心实现
        
        Args:
            query: 查询序列
            timeseries: 时间序列
            
        Returns:
            距离数组
        """
        n = len(timeseries)
        m = len(query)
        
        if m > n:
            logger.warning("查询序列长度超过时间序列长度")
            return np.array([])
        
        # 标准化查询序列
        if self.normalize:
            query = self.normalize_series(query, 'zscore')
        
        # 使用FFT计算互相关
        try:
            # 填充查询序列到时间序列长度
            query_padded = np.zeros(n)
            query_padded[:m] = query[::-1]  # 反转查询序列
            
            # FFT互相关
            timeseries_fft = np.fft.fft(timeseries)
            query_fft = np.fft.fft(query_padded)
            
            # 计算互相关
            correlation = np.fft.ifft(timeseries_fft * np.conj(query_fft)).real
            
            # 提取有效部分
            correlation = correlation[m-1:n]
            
            # 计算滑动均值和方差（用于z标准化）
            if self.normalize:
                distances = self._compute_normalized_distances(
                    correlation, timeseries, query, m
                )
            else:
                # 转换为欧几里德距离
                distances = np.sqrt(2 * m * (1 - correlation / m))
            
            return distances
            
        except Exception as e:
            logger.error(f"FFT互相关计算失败: {str(e)}")
            return self._sliding_window_search(query, timeseries)
    
    def _compute_normalized_distances(self, 
                                    correlation: np.ndarray,
                                    timeseries: np.ndarray,
                                    query: np.ndarray,
                                    m: int) -> np.ndarray:
        """计算标准化距离"""
        n = len(timeseries)
        
        # 计算滑动均值
        sliding_mean = self._sliding_window_mean(timeseries, m)
        
        # 计算滑动方差
        sliding_var = self._sliding_window_var(timeseries, m, sliding_mean)
        
        # 避免除零
        sliding_std = np.sqrt(np.maximum(sliding_var, 1e-8))
        
        # 计算z标准化距离
        query_std = np.std(query) if np.std(query) > 0 else 1.0
        
        distances = []
        for i in range(len(correlation)):
            if i + m > n:
                break
                
            # z标准化距离公式
            mean_i = sliding_mean[i]
            std_i = sliding_std[i]
            
            # 标准化互相关
            normalized_corr = (correlation[i] - m * np.mean(query) * mean_i) / (m * query_std * std_i)
            
            # 转换为欧几里德距离
            distance = np.sqrt(2 * m * (1 - normalized_corr))
            distances.append(distance)
        
        return np.array(distances)
    
    def _sliding_window_mean(self, timeseries: np.ndarray, window_size: int) -> np.ndarray:
        """计算滑动窗口均值"""
        n = len(timeseries)
        if window_size > n:
            return np.array([np.mean(timeseries)])
        
        # 使用卷积计算滑动均值
        kernel = np.ones(window_size) / window_size
        padded = np.pad(timeseries, (window_size//2, window_size//2), mode='edge')
        means = np.convolve(padded, kernel, mode='valid')
        
        return means[:n-window_size+1]
    
    def _sliding_window_var(self, 
                           timeseries: np.ndarray, 
                           window_size: int, 
                           means: np.ndarray) -> np.ndarray:
        """计算滑动窗口方差"""
        n = len(timeseries)
        variances = []
        
        for i in range(n - window_size + 1):
            window = timeseries[i:i + window_size]
            var = np.sum((window - means[i]) ** 2) / window_size
            variances.append(var)
        
        return np.array(variances)
    
    def _sliding_window_search(self, query: np.ndarray, timeseries: np.ndarray) -> np.ndarray:
        """备用的滑动窗口搜索"""
        n = len(timeseries)
        m = len(query)
        distances = []
        
        if self.normalize:
            query_normalized = self.normalize_series(query, 'zscore')
        else:
            query_normalized = query
        
        for i in range(n - m + 1):
            window = timeseries[i:i + m]
            
            if self.normalize:
                window_normalized = self.normalize_series(window, 'zscore')
            else:
                window_normalized = window
            
            # 计算欧几里德距离
            distance = np.sqrt(np.sum((query_normalized - window_normalized) ** 2))
            distances.append(distance)
        
        return np.array(distances)
    
    def _fallback_similarity(self, query_series: np.ndarray, candidate_series: np.ndarray) -> float:
        """备用相似度计算"""
        try:
            # 使用滑动窗口搜索作为备用
            distances = self._sliding_window_search(query_series, candidate_series)
            
            if len(distances) == 0:
                return 0.0
            
            min_distance = np.min(distances)
            similarity = 1.0 / (1.0 + min_distance)
            
            return similarity
            
        except Exception as e:
            logger.error(f"备用相似度计算失败: {str(e)}")
            return 0.0
    
    def find_top_matches(self, 
                        query: np.ndarray, 
                        timeseries: np.ndarray, 
                        k: int = 5) -> List[Tuple[int, float]]:
        """
        找到时间序列中最匹配的k个子序列
        
        Args:
            query: 查询序列
            timeseries: 时间序列
            k: 返回top-k结果
            
        Returns:
            List of (position, distance)
        """
        distances = self.mass_search(query, timeseries)
        
        if len(distances) == 0:
            return []
        
        # 找到最小的k个距离及其位置
        top_k_indices = np.argpartition(distances, min(k-1, len(distances)-1))[:k]
        
        results = []
        for idx in top_k_indices:
            if idx < len(distances):
                distance = distances[idx]
                similarity = 1.0 / (1.0 + distance)
                results.append((idx, similarity))
        
        # 按相似度降序排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def batch_search(self, 
                    queries: List[np.ndarray],
                    timeseries: np.ndarray) -> List[List[Tuple[int, float]]]:
        """
        批量搜索多个查询序列
        
        Args:
            queries: 查询序列列表
            timeseries: 时间序列
            
        Returns:
            每个查询的搜索结果列表
        """
        results = []
        
        for query in queries:
            if len(query) == 0:
                results.append([])
                continue
            
            try:
                matches = self.find_top_matches(query, timeseries, k=5)
                results.append(matches)
            except Exception as e:
                logger.error(f"批量搜索失败: {str(e)}")
                results.append([])
        
        return results
    
    def approximate_search(self, 
                          query: np.ndarray,
                          timeseries: np.ndarray,
                          approximation_factor: int = 4) -> np.ndarray:
        """
        近似搜索（降采样加速）
        
        Args:
            query: 查询序列
            timeseries: 时间序列
            approximation_factor: 近似因子（降采样倍数）
            
        Returns:
            近似距离数组
        """
        # 降采样
        query_downsampled = query[::approximation_factor]
        timeseries_downsampled = timeseries[::approximation_factor]
        
        # 在降采样数据上搜索
        distances_downsampled = self.mass_search(query_downsampled, timeseries_downsampled)
        
        # 上采样结果
        if len(distances_downsampled) > 0:
            # 简单的最近邻上采样
            upsampled_length = len(timeseries) - len(query) + 1
            distances_upsampled = np.interp(
                np.arange(upsampled_length),
                np.arange(0, upsampled_length, approximation_factor),
                distances_downsampled
            )
            return distances_upsampled
        
        return np.array([])


if __name__ == '__main__':
    # 测试代码
    np.random.seed(42)
    
    # 创建测试数据
    t = np.linspace(0, 4*np.pi, 200)
    timeseries = np.sin(t) + 0.5 * np.sin(3*t) + 0.1 * np.random.randn(200)
    
    # 创建查询序列（从时间序列中提取一段）
    query = timeseries[50:80] + 0.05 * np.random.randn(30)  # 添加噪声
    
    # 测试MASS算法
    mass = MASSimilarity({'normalize': True})
    
    # 搜索匹配
    distances = mass.mass_search(query, timeseries)
    print(f"MASS搜索完成，距离数组长度: {len(distances)}")
    
    # 找到最佳匹配位置
    if len(distances) > 0:
        best_match_idx = np.argmin(distances)
        best_distance = distances[best_match_idx]
        print(f"最佳匹配位置: {best_match_idx}, 距离: {best_distance:.4f}")
        print(f"真实位置: 50, 误差: {abs(best_match_idx - 50)}")
    
    # 测试top-k搜索
    top_matches = mass.find_top_matches(query, timeseries, k=5)
    print(f"\nTop-5 匹配:")
    for i, (pos, sim) in enumerate(top_matches):
        print(f"  {i+1}. 位置: {pos}, 相似度: {sim:.4f}")
    
    # 测试不同查询
    query2 = np.cos(np.linspace(0, 2*np.pi, 30))  # 不同模式
    similarity1 = mass.compute_similarity(query, timeseries)
    similarity2 = mass.compute_similarity(query2, timeseries)
    
    print(f"\n相似度比较:")
    print(f"原始模式查询: {similarity1:.4f}")
    print(f"不同模式查询: {similarity2:.4f}")
    
    # 测试批量搜索
    queries = [query, query2]
    batch_results = mass.batch_search(queries, timeseries)
    print(f"\n批量搜索结果:")
    for i, results in enumerate(batch_results):
        print(f"  查询 {i+1}: {len(results)} 个匹配")
    
    # 测试近似搜索
    approx_distances = mass.approximate_search(query, timeseries, approximation_factor=2)
    if len(approx_distances) > 0:
        approx_best_idx = np.argmin(approx_distances)
        print(f"\n近似搜索最佳位置: {approx_best_idx}")
        print(f"精确搜索最佳位置: {best_match_idx}")
        print(f"近似搜索误差: {abs(approx_best_idx - best_match_idx)}")
    
    # 性能比较测试
    import time
    
    # 测试大数据集
    large_timeseries = np.sin(np.linspace(0, 20*np.pi, 2000)) + 0.1 * np.random.randn(2000)
    large_query = large_timeseries[500:550]
    
    start_time = time.time()
    large_distances = mass.mass_search(large_query, large_timeseries)
    fft_time = time.time() - start_time
    
    start_time = time.time() 
    sliding_distances = mass._sliding_window_search(large_query, large_timeseries)
    sliding_time = time.time() - start_time
    
    print(f"\n性能比较 (数据长度: {len(large_timeseries)}):")
    print(f"FFT方法: {fft_time:.4f}s")
    print(f"滑动窗口方法: {sliding_time:.4f}s")
    print(f"加速比: {sliding_time/fft_time:.2f}x")