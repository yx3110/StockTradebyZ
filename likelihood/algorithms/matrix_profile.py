"""
Matrix Profile相似度算法实现
基于STUMPY库实现高效的时间序列相似度搜索
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
import logging
from .base_similarity import BaseSimilarityAlgorithm, MultivariateSimilarityAlgorithm

logger = logging.getLogger(__name__)

# 可选依赖处理
try:
    import stumpy
    STUMPY_AVAILABLE = True
except ImportError:
    STUMPY_AVAILABLE = False
    logger.warning("STUMPY库未安装，Matrix Profile功能将受限")


class MatrixProfileSimilarity(MultivariateSimilarityAlgorithm):
    """
    Matrix Profile相似度算法
    
    Matrix Profile是一种用于时间序列分析的强大工具，可以高效地找到
    时间序列中的相似子序列。它的优势在于O(n log n)的时间复杂度。
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化Matrix Profile算法
        
        Args:
            config: 配置参数
                - window_length: 滑动窗口长度
                - exclusion_zone: 排除区域大小
                - normalize: 是否标准化
        """
        super().__init__(config)
        self.window_length = self.config.get('window_length', 30)
        self.exclusion_zone = self.config.get('exclusion_zone', 
                                            self.window_length // 4)
        self.normalize = self.config.get('normalize', True)
        
        if not STUMPY_AVAILABLE:
            logger.warning("STUMPY库未安装，将使用简化实现")
    
    def compute_similarity(self,
                          query_series: np.ndarray,
                          candidate_series: np.ndarray,
                          **kwargs) -> float:
        """
        计算两个时间序列的Matrix Profile相似度
        
        Args:
            query_series: 查询序列
            candidate_series: 候选序列
            
        Returns:
            相似度分数 (0-1，越高越相似)
        """
        if not self.validate_input(query_series, candidate_series):
            return 0.0
        
        # 确保序列长度足够
        if len(query_series) < self.window_length or len(candidate_series) < self.window_length:
            logger.warning(f"序列长度不足，需要至少 {self.window_length} 个数据点")
            return self._fallback_similarity(query_series, candidate_series)
        
        # 标准化
        if self.normalize:
            query_series = self.normalize_series(query_series, 'zscore')
            candidate_series = self.normalize_series(candidate_series, 'zscore')
        
        try:
            if STUMPY_AVAILABLE:
                return self._stumpy_similarity(query_series, candidate_series)
            else:
                return self._manual_similarity(query_series, candidate_series)
        except Exception as e:
            logger.error(f"Matrix Profile计算失败: {str(e)}")
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
            if len(candidate) < self.window_length:
                continue
                
            similarity = self.compute_similarity(query_series, candidate)
            results.append((i, similarity))
        
        # 按相似度降序排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def _stumpy_similarity(self, query_series: np.ndarray, candidate_series: np.ndarray) -> float:
        """使用STUMPY库计算相似度"""
        # 使用STUMP计算Matrix Profile
        mp = stumpy.stump(candidate_series, self.window_length)
        
        # 计算查询序列与候选序列的距离
        distances = stumpy.mass(query_series[-self.window_length:], candidate_series)
        
        # 找到最小距离并转换为相似度
        min_distance = np.min(distances[np.isfinite(distances)])
        
        # 转换为相似度 (距离越小，相似度越高)
        similarity = 1.0 / (1.0 + min_distance)
        
        return similarity
    
    def _manual_similarity(self, query_series: np.ndarray, candidate_series: np.ndarray) -> float:
        """手动实现的相似度计算"""
        # 简化的Matrix Profile实现
        query_window = query_series[-self.window_length:]
        
        min_distance = float('inf')
        
        # 滑动窗口计算距离
        for i in range(len(candidate_series) - self.window_length + 1):
            candidate_window = candidate_series[i:i + self.window_length]
            
            # 计算欧几里德距离
            distance = np.sqrt(np.sum((query_window - candidate_window) ** 2))
            min_distance = min(min_distance, distance)
        
        # 转换为相似度
        similarity = 1.0 / (1.0 + min_distance)
        
        return similarity
    
    def _fallback_similarity(self, query_series: np.ndarray, candidate_series: np.ndarray) -> float:
        """备用相似度计算方法"""
        # 使用皮尔逊相关系数作为备用
        try:
            # 截取相同长度
            min_length = min(len(query_series), len(candidate_series))
            query_truncated = query_series[-min_length:]
            candidate_truncated = candidate_series[-min_length:]
            
            correlation = np.corrcoef(query_truncated, candidate_truncated)[0, 1]
            
            # 处理NaN值
            if np.isnan(correlation):
                return 0.0
            
            # 转换为0-1范围的相似度
            similarity = (correlation + 1.0) / 2.0
            
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"备用相似度计算失败: {str(e)}")
            return 0.0
    
    def compute_matrix_profile(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算完整的Matrix Profile
        
        Args:
            series: 时间序列
            
        Returns:
            (matrix_profile, matrix_profile_index)
        """
        if not STUMPY_AVAILABLE:
            return self._manual_matrix_profile(series)
        
        try:
            mp = stumpy.stump(series, self.window_length)
            return mp[:, 0], mp[:, 1].astype(int)
        except Exception as e:
            logger.error(f"Matrix Profile计算失败: {str(e)}")
            return self._manual_matrix_profile(series)
    
    def _manual_matrix_profile(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """手动实现Matrix Profile"""
        n = len(series)
        m = self.window_length
        
        if n < m:
            return np.array([]), np.array([])
        
        matrix_profile = np.full(n - m + 1, np.inf)
        matrix_profile_index = np.zeros(n - m + 1, dtype=int)
        
        # 对每个位置计算最近邻距离
        for i in range(n - m + 1):
            query = series[i:i + m]
            query_normalized = self.normalize_series(query, 'zscore')
            
            min_dist = np.inf
            min_idx = -1
            
            # 搜索最相似的子序列
            for j in range(n - m + 1):
                # 排除区域
                if abs(i - j) < self.exclusion_zone:
                    continue
                
                candidate = series[j:j + m]
                candidate_normalized = self.normalize_series(candidate, 'zscore')
                
                # 计算欧几里德距离
                dist = np.sqrt(np.sum((query_normalized - candidate_normalized) ** 2))
                
                if dist < min_dist:
                    min_dist = dist
                    min_idx = j
            
            matrix_profile[i] = min_dist
            matrix_profile_index[i] = min_idx
        
        return matrix_profile, matrix_profile_index
    
    def find_motifs(self, 
                   series: np.ndarray, 
                   k: int = 3) -> List[Tuple[int, int, float]]:
        """
        发现时间序列中的重复模式(motifs)
        
        Args:
            series: 时间序列
            k: 返回前k个motifs
            
        Returns:
            List of (index1, index2, distance) - motif pairs
        """
        mp, mpi = self.compute_matrix_profile(series)
        
        if len(mp) == 0:
            return []
        
        motifs = []
        
        # 找到前k个最小距离的motif pairs
        sorted_indices = np.argsort(mp)
        
        for i in range(min(k, len(sorted_indices))):
            idx = sorted_indices[i]
            motif_distance = mp[idx]
            nearest_neighbor_idx = mpi[idx]
            
            motifs.append((idx, nearest_neighbor_idx, motif_distance))
        
        return motifs
    
    def find_discords(self, 
                     series: np.ndarray, 
                     k: int = 3) -> List[Tuple[int, float]]:
        """
        发现时间序列中的异常模式(discords)
        
        Args:
            series: 时间序列
            k: 返回前k个discords
            
        Returns:
            List of (index, distance) - discord positions
        """
        mp, _ = self.compute_matrix_profile(series)
        
        if len(mp) == 0:
            return []
        
        # 找到距离最大的k个位置（异常）
        sorted_indices = np.argsort(mp)[::-1]  # 降序排列
        
        discords = []
        for i in range(min(k, len(sorted_indices))):
            idx = sorted_indices[i]
            discord_distance = mp[idx]
            discords.append((idx, discord_distance))
        
        return discords


if __name__ == '__main__':
    # 测试代码
    np.random.seed(42)
    
    # 创建测试数据
    t = np.linspace(0, 4*np.pi, 100)
    series1 = np.sin(t) + 0.1 * np.random.randn(100)
    series2 = np.sin(t + 0.5) + 0.1 * np.random.randn(100)  # 稍有相位差
    series3 = np.cos(t) + 0.1 * np.random.randn(100)       # 不同模式
    
    # 测试Matrix Profile算法
    mp = MatrixProfileSimilarity({'window_length': 20})
    
    # 计算相似度
    sim12 = mp.compute_similarity(series1, series2)
    sim13 = mp.compute_similarity(series1, series3)
    
    print(f"Series1 vs Series2 相似度: {sim12:.4f}")
    print(f"Series1 vs Series3 相似度: {sim13:.4f}")
    
    # 测试搜索功能
    database = np.array([series1, series2, series3])
    query = series1
    
    results = mp.search_similar(query, database, top_k=3)
    print(f"\n搜索结果:")
    for idx, similarity in results:
        print(f"  序列 {idx}: 相似度 {similarity:.4f}")
    
    # 测试Matrix Profile计算
    mp_values, mp_indices = mp.compute_matrix_profile(series1)
    print(f"\nMatrix Profile计算完成，长度: {len(mp_values)}")
    
    # 测试motifs发现
    motifs = mp.find_motifs(series1, k=3)
    print(f"\n发现的Motifs:")
    for i, (idx1, idx2, dist) in enumerate(motifs):
        print(f"  Motif {i+1}: 位置 ({idx1}, {idx2}), 距离: {dist:.4f}")
        
    # 测试discords发现
    discords = mp.find_discords(series1, k=3)
    print(f"\n发现的Discords:")
    for i, (idx, dist) in enumerate(discords):
        print(f"  Discord {i+1}: 位置 {idx}, 距离: {dist:.4f}")