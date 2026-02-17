"""
Dynamic Time Warping (DTW) 相似度算法实现
处理时间序列在时间轴上的伸缩变形
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging
from .base_similarity import BaseSimilarityAlgorithm, MultivariateSimilarityAlgorithm

logger = logging.getLogger(__name__)

# 可选依赖处理
try:
    from dtaidistance import dtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    logger.warning("dtaidistance库未安装，DTW功能将使用简化实现")


class DTWSimilarity(MultivariateSimilarityAlgorithm):
    """
    Dynamic Time Warping相似度算法
    
    DTW通过动态规划找到两个序列之间的最佳对齐方式，
    能够处理时间序列在时间轴上的非线性变形。
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化DTW算法
        
        Args:
            config: 配置参数
                - sakoe_chiba_radius: Sakoe-Chiba带宽约束
                - itakura_max_slope: Itakura平行四边形约束
                - window_type: 窗口类型 ('sakoe_chiba', 'itakura', 'none')
                - normalize: 是否标准化
        """
        super().__init__(config)
        self.sakoe_chiba_radius = self.config.get('sakoe_chiba_radius', None)
        self.itakura_max_slope = self.config.get('itakura_max_slope', 2.0)
        self.window_type = self.config.get('window_type', 'sakoe_chiba')
        self.normalize = self.config.get('normalize', True)
        
        # 设置默认窗口
        if self.window_type == 'sakoe_chiba' and self.sakoe_chiba_radius is None:
            self.sakoe_chiba_radius = 5
    
    def compute_similarity(self,
                          query_series: np.ndarray,
                          candidate_series: np.ndarray,
                          **kwargs) -> float:
        """
        计算两个时间序列的DTW相似度
        
        Args:
            query_series: 查询序列
            candidate_series: 候选序列
            
        Returns:
            相似度分数 (0-1，越高越相似)
        """
        if not self.validate_input(query_series, candidate_series):
            return 0.0
        
        # 标准化
        if self.normalize:
            query_series = self.normalize_series(query_series, 'zscore')
            candidate_series = self.normalize_series(candidate_series, 'zscore')
        
        try:
            if DTW_AVAILABLE:
                distance = self._dtw_library_distance(query_series, candidate_series)
            else:
                distance = self._manual_dtw_distance(query_series, candidate_series)
            
            # 转换为相似度
            similarity = 1.0 / (1.0 + distance)
            return similarity
            
        except Exception as e:
            logger.error(f"DTW计算失败: {str(e)}")
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
            if len(candidate) == 0:
                continue
                
            similarity = self.compute_similarity(query_series, candidate)
            results.append((i, similarity))
        
        # 按相似度降序排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def _dtw_library_distance(self, s1: np.ndarray, s2: np.ndarray) -> float:
        """使用dtaidistance库计算DTW距离"""
        if self.window_type == 'sakoe_chiba':
            distance = dtw.distance(s1, s2, window=self.sakoe_chiba_radius)
        elif self.window_type == 'itakura':
            # 注意：dtaidistance可能不直接支持Itakura约束
            distance = dtw.distance(s1, s2)
        else:
            distance = dtw.distance(s1, s2)
        
        return distance
    
    def _manual_dtw_distance(self, s1: np.ndarray, s2: np.ndarray) -> float:
        """手动实现DTW距离计算"""
        n, m = len(s1), len(s2)
        
        # 初始化DTW矩阵
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        # 应用窗口约束
        if self.window_type == 'sakoe_chiba':
            return self._constrained_dtw(s1, s2, dtw_matrix, self._sakoe_chiba_constraint)
        elif self.window_type == 'itakura':
            return self._constrained_dtw(s1, s2, dtw_matrix, self._itakura_constraint)
        else:
            return self._unconstrained_dtw(s1, s2, dtw_matrix)
    
    def _unconstrained_dtw(self, s1: np.ndarray, s2: np.ndarray, dtw_matrix: np.ndarray) -> float:
        """无约束DTW计算"""
        n, m = len(s1), len(s2)
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = (s1[i-1] - s2[j-1]) ** 2
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],      # 插入
                    dtw_matrix[i, j-1],      # 删除
                    dtw_matrix[i-1, j-1]     # 匹配
                )
        
        return np.sqrt(dtw_matrix[n, m])
    
    def _constrained_dtw(self, 
                        s1: np.ndarray, 
                        s2: np.ndarray, 
                        dtw_matrix: np.ndarray,
                        constraint_func) -> float:
        """带约束的DTW计算"""
        n, m = len(s1), len(s2)
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if not constraint_func(i, j, n, m):
                    continue
                
                cost = (s1[i-1] - s2[j-1]) ** 2
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],      # 插入
                    dtw_matrix[i, j-1],      # 删除
                    dtw_matrix[i-1, j-1]     # 匹配
                )
        
        return np.sqrt(dtw_matrix[n, m])
    
    def _sakoe_chiba_constraint(self, i: int, j: int, n: int, m: int) -> bool:
        """Sakoe-Chiba带宽约束"""
        if self.sakoe_chiba_radius is None:
            return True
        
        # 计算对角线上的期望位置
        expected_j = int(j * n / m)
        return abs(i - expected_j) <= self.sakoe_chiba_radius
    
    def _itakura_constraint(self, i: int, j: int, n: int, m: int) -> bool:
        """Itakura平行四边形约束"""
        # 简化的Itakura约束实现
        slope_min = 1.0 / self.itakura_max_slope
        slope_max = self.itakura_max_slope
        
        # 检查是否在允许的斜率范围内
        if i == 1 and j == 1:
            return True
        
        if i == 1:
            return j <= slope_max
        if j == 1:
            return i <= slope_max
        
        current_slope = i / j
        return slope_min <= current_slope <= slope_max
    
    def _fallback_similarity(self, query_series: np.ndarray, candidate_series: np.ndarray) -> float:
        """备用相似度计算"""
        try:
            # 使用简单的欧几里德距离作为备用
            min_len = min(len(query_series), len(candidate_series))
            q_truncated = query_series[-min_len:]
            c_truncated = candidate_series[-min_len:]
            
            distance = np.sqrt(np.sum((q_truncated - c_truncated) ** 2))
            similarity = 1.0 / (1.0 + distance)
            
            return similarity
            
        except Exception as e:
            logger.error(f"备用相似度计算失败: {str(e)}")
            return 0.0
    
    def compute_dtw_path(self, s1: np.ndarray, s2: np.ndarray) -> List[Tuple[int, int]]:
        """
        计算DTW最优路径
        
        Args:
            s1: 序列1
            s2: 序列2
            
        Returns:
            最优路径 [(i1, j1), (i2, j2), ...]
        """
        n, m = len(s1), len(s2)
        
        # 计算DTW矩阵
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if (self.window_type == 'sakoe_chiba' and 
                    not self._sakoe_chiba_constraint(i, j, n, m)):
                    continue
                
                cost = (s1[i-1] - s2[j-1]) ** 2
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],
                    dtw_matrix[i, j-1],
                    dtw_matrix[i-1, j-1]
                )
        
        # 回溯路径
        path = []
        i, j = n, m
        
        while i > 0 and j > 0:
            path.append((i-1, j-1))
            
            # 选择最小成本的前一步
            if i == 1:
                j -= 1
            elif j == 1:
                i -= 1
            else:
                prev_costs = [
                    dtw_matrix[i-1, j-1],  # 对角
                    dtw_matrix[i-1, j],    # 垂直
                    dtw_matrix[i, j-1]     # 水平
                ]
                min_idx = np.argmin(prev_costs)
                if min_idx == 0:
                    i -= 1
                    j -= 1
                elif min_idx == 1:
                    i -= 1
                else:
                    j -= 1
        
        return path[::-1]  # 反转路径
    
    def compute_multivariate_dtw(self,
                               query_series: np.ndarray,
                               candidate_series: np.ndarray,
                               dimension_weights: np.ndarray = None) -> float:
        """
        计算多变量DTW相似度
        
        Args:
            query_series: 查询序列 (n_timesteps x n_dimensions)
            candidate_series: 候选序列 (n_timesteps x n_dimensions)
            dimension_weights: 维度权重
            
        Returns:
            相似度分数
        """
        if query_series.ndim == 1:
            query_series = query_series.reshape(-1, 1)
        if candidate_series.ndim == 1:
            candidate_series = candidate_series.reshape(-1, 1)
        
        n_dimensions = query_series.shape[1]
        
        if dimension_weights is None:
            dimension_weights = self.dimension_weights
        if dimension_weights is None:
            dimension_weights = np.ones(n_dimensions) / n_dimensions
        
        # 计算每个维度的DTW距离
        total_distance = 0.0
        
        for i in range(n_dimensions):
            query_dim = query_series[:, i]
            candidate_dim = candidate_series[:, i]
            
            try:
                if DTW_AVAILABLE:
                    distance = self._dtw_library_distance(query_dim, candidate_dim)
                else:
                    distance = self._manual_dtw_distance(query_dim, candidate_dim)
                
                total_distance += dimension_weights[i] * distance
                
            except Exception as e:
                logger.error(f"维度 {i} DTW计算失败: {str(e)}")
                # 使用欧几里德距离作为备用
                min_len = min(len(query_dim), len(candidate_dim))
                euclidean_dist = np.sqrt(np.sum((query_dim[-min_len:] - candidate_dim[-min_len:]) ** 2))
                total_distance += dimension_weights[i] * euclidean_dist
        
        # 转换为相似度
        similarity = 1.0 / (1.0 + total_distance)
        
        return similarity


if __name__ == '__main__':
    # 测试代码
    np.random.seed(42)
    
    # 创建测试数据
    t1 = np.linspace(0, 2*np.pi, 50)
    t2 = np.linspace(0, 2*np.pi, 60)  # 不同长度
    
    series1 = np.sin(t1) + 0.1 * np.random.randn(50)
    series2 = np.sin(t2) + 0.1 * np.random.randn(60)  # 时间伸缩
    series3 = np.cos(t1) + 0.1 * np.random.randn(50)  # 不同模式
    
    # 测试DTW算法
    dtw_alg = DTWSimilarity({
        'window_type': 'sakoe_chiba',
        'sakoe_chiba_radius': 5
    })
    
    # 计算相似度
    sim12 = dtw_alg.compute_similarity(series1, series2)
    sim13 = dtw_alg.compute_similarity(series1, series3)
    
    print(f"DTW相似度测试:")
    print(f"Series1 vs Series2 (时间伸缩): {sim12:.4f}")
    print(f"Series1 vs Series3 (不同模式): {sim13:.4f}")
    
    # 测试不同约束
    dtw_unconstrained = DTWSimilarity({'window_type': 'none'})
    sim12_unconstrained = dtw_unconstrained.compute_similarity(series1, series2)
    
    print(f"\n无约束DTW: {sim12_unconstrained:.4f}")
    print(f"带约束DTW: {sim12:.4f}")
    
    # 测试路径计算
    path = dtw_alg.compute_dtw_path(series1[:20], series2[:20])
    print(f"\nDTW路径长度: {len(path)}")
    print(f"路径示例: {path[:5]}")
    
    # 测试多变量DTW
    multi_query = np.column_stack([series1, series1 * 0.5])  # 2维
    multi_candidate = np.column_stack([series2, series2 * 0.5])  # 2维
    
    multi_sim = dtw_alg.compute_multivariate_dtw(multi_query, multi_candidate)
    print(f"\n多变量DTW相似度: {multi_sim:.4f}")
    
    # 测试搜索功能
    database = np.array([series1, series2, series3])
    results = dtw_alg.search_similar(series1, database, top_k=3)
    
    print(f"\nDTW搜索结果:")
    for idx, similarity in results:
        print(f"  序列 {idx}: 相似度 {similarity:.4f}")