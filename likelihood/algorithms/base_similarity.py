"""
相似度算法基类
Base class for similarity algorithms
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseSimilarityAlgorithm(ABC):
    """相似度算法基类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化基础相似度算法
        
        Args:
            config: 算法配置参数
        """
        self.config = config or {}
        self.algorithm_name = self.__class__.__name__
        
    @abstractmethod
    def compute_similarity(self, 
                          query_series: np.ndarray,
                          candidate_series: np.ndarray,
                          **kwargs) -> float:
        """
        计算两个时间序列的相似度
        
        Args:
            query_series: 查询序列
            candidate_series: 候选序列
            **kwargs: 额外参数
            
        Returns:
            相似度分数 (越高越相似)
        """
        pass
    
    @abstractmethod
    def search_similar(self,
                      query_series: np.ndarray,
                      database_series: np.ndarray,
                      top_k: int = 10,
                      **kwargs) -> List[Tuple[int, float]]:
        """
        在数据库中搜索最相似的序列
        
        Args:
            query_series: 查询序列
            database_series: 数据库序列 (2D array: n_samples x n_features)
            top_k: 返回最相似的K个结果
            **kwargs: 额外参数
            
        Returns:
            List of (index, similarity_score) 按相似度降序排列
        """
        pass
    
    def validate_input(self, 
                      query_series: np.ndarray,
                      candidate_series: np.ndarray = None) -> bool:
        """
        验证输入数据
        
        Args:
            query_series: 查询序列
            candidate_series: 候选序列
            
        Returns:
            是否有效
        """
        if query_series is None or len(query_series) == 0:
            logger.error("查询序列为空")
            return False
        
        if np.any(np.isnan(query_series)):
            logger.warning("查询序列包含NaN值")
            return False
        
        if candidate_series is not None:
            if len(candidate_series) == 0:
                logger.error("候选序列为空")
                return False
            
            if np.any(np.isnan(candidate_series)):
                logger.warning("候选序列包含NaN值") 
                return False
                
        return True
    
    def normalize_series(self, series: np.ndarray, method: str = 'zscore') -> np.ndarray:
        """
        标准化时间序列
        
        Args:
            series: 输入序列
            method: 标准化方法 ('zscore', 'minmax', 'none')
            
        Returns:
            标准化后的序列
        """
        if method == 'zscore':
            mean = np.mean(series)
            std = np.std(series)
            if std == 0:
                return np.zeros_like(series)
            return (series - mean) / std
        
        elif method == 'minmax':
            min_val = np.min(series)
            max_val = np.max(series)
            if max_val == min_val:
                return np.zeros_like(series)
            return (series - min_val) / (max_val - min_val)
        
        elif method == 'none':
            return series
        
        else:
            raise ValueError(f"未知的标准化方法: {method}")
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """
        获取算法信息
        
        Returns:
            算法信息字典
        """
        return {
            'name': self.algorithm_name,
            'config': self.config,
            'description': self.__doc__ or "无描述"
        }


class MultivariateSimilarityAlgorithm(BaseSimilarityAlgorithm):
    """多变量相似度算法基类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.dimension_weights = self.config.get('dimension_weights', None)
    
    def compute_multivariate_similarity(self,
                                      query_series: np.ndarray,
                                      candidate_series: np.ndarray,
                                      dimension_weights: np.ndarray = None,
                                      **kwargs) -> float:
        """
        计算多变量时间序列相似度
        
        Args:
            query_series: 查询序列 (n_timesteps x n_dimensions)
            candidate_series: 候选序列 (n_timesteps x n_dimensions)
            dimension_weights: 维度权重
            **kwargs: 额外参数
            
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
        
        # 计算每个维度的相似度并加权平均
        dimension_similarities = []
        for i in range(n_dimensions):
            query_dim = query_series[:, i]
            candidate_dim = candidate_series[:, i]
            
            # 调用子类实现的单变量相似度计算
            sim = self.compute_similarity(query_dim, candidate_dim, **kwargs)
            dimension_similarities.append(sim)
        
        # 加权平均
        weighted_similarity = np.average(dimension_similarities, weights=dimension_weights)
        
        return weighted_similarity
    
    def set_dimension_weights(self, weights: np.ndarray):
        """设置维度权重"""
        self.dimension_weights = weights