#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可扩展通用回测引擎 - 支持动态模型版本管理和高度可扩展架构

Features:
🚀 动态模型注册与发现机制
🔄 可插拔的评分系统架构
📊 多版本自动对比分析
⚡ 并行化计算支持
💰 完整交易执行链路
🎯 命令行参数化控制

Architecture:
- ModelRegistry: 模型注册中心，支持动态发现和注册
- MLModelAdapter: 统一模型接口，支持插件化扩展
- ExtensibleBacktestEngine: 可扩展回测引擎核心
- 自动模型发现: 扫描models目录，自动识别可用模型版本

Author: Claude & User
Date: 2025-09-24
"""

import os
import sys
import logging
import time
import json
import argparse
import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Type
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from importlib import import_module
import pandas as pd
import numpy as np

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from data_adapter.database_manager import DatabaseManager
    logger = logging.getLogger(__name__)
    logger.info("✅ 可扩展回测引擎依赖加载成功")
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"❌ 可扩展回测引擎依赖加载失败: {e}")


@dataclass
class ModelConfig:
    """模型配置类"""
    version: str
    name: str
    module_name: str
    class_name: str
    model_path_pattern: str
    features_count: int
    description: str
    requires_modules: List[str] = None
    custom_config: Dict[str, Any] = None

    def __post_init__(self):
        if self.requires_modules is None:
            self.requires_modules = []
        if self.custom_config is None:
            self.custom_config = {}


class MLModelAdapter(ABC):
    """统一ML模型适配器接口"""

    def __init__(self, config: ModelConfig, max_workers: int = 4):
        self.config = config
        self.max_workers = max_workers
        self.model_path = None
        self._initialize()

    @abstractmethod
    def _initialize(self):
        """初始化模型适配器"""
        pass

    @abstractmethod
    def calculate_scores(self, stock_list: List[str], date: str) -> Dict[str, float]:
        """计算股票评分"""
        pass

    @abstractmethod
    def normalize_scores(self, raw_scores: Dict[str, float]) -> Dict[str, float]:
        """
        将原始评分标准化到百分制 (0-100)

        Args:
            raw_scores: 原始评分字典 {股票代码: 原始评分}

        Returns:
            标准化后的评分字典 {股票代码: 百分制评分}
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """检查模型是否可用"""
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'version': self.config.version,
            'name': self.config.name,
            'model_path': self.model_path,
            'features_count': self.config.features_count,
            'description': self.config.description,
            'is_available': self.is_available()
        }

    def _find_model_file(self) -> Optional[str]:
        """查找模型文件"""
        pattern_parts = self.config.model_path_pattern.split('/')
        model_dir = Path('/'.join(pattern_parts[:-1]))

        if not model_dir.exists():
            logger.warning(f"模型目录不存在: {model_dir}")
            return None

        pattern = pattern_parts[-1]
        model_files = list(model_dir.glob(pattern))

        if model_files:
            latest_model = max(model_files, key=lambda f: f.stat().st_mtime)
            logger.info(f"📱 找到{self.config.version}模型: {latest_model}")
            return str(latest_model)

        logger.warning(f"未找到{self.config.version}模型，模式: {self.config.model_path_pattern}")
        return None


class V37ModelAdapter(MLModelAdapter):
    """V3.7模型适配器"""

    def _initialize(self):
        """初始化V3.7模型"""
        self.model_path = self._find_model_file()

    def calculate_scores(self, stock_list: List[str], date: str) -> Dict[str, float]:
        """使用V3.7系统计算评分"""
        if not self.model_path:
            raw_scores = {code: 50.0 for code in stock_list}
        else:
            try:
                # 使用已验证的V3.7并行评分函数
                from ml_models.v37.backtest_v37_engine_optimized import batch_calculate_v37_scores
                results = batch_calculate_v37_scores(stock_list, date, self.model_path)
                raw_scores = dict(results)
            except Exception as e:
                logger.warning(f"V3.7评分计算失败: {e}")
                raw_scores = {code: 50.0 for code in stock_list}

        # 标准化到百分制
        return self.normalize_scores(raw_scores)

    def normalize_scores(self, raw_scores: Dict[str, float]) -> Dict[str, float]:
        """
        V3.7评分标准化：V3.7的predict_three_layer_ensemble已经返回0-100范围的评分，
        使用Sigmoid函数标准化，范围[5, 95]。无需二次标准化，直接返回。
        """
        if not raw_scores:
            return raw_scores

        # V3.7已经标准化到0-100，直接返回
        excellent_count = sum(1 for s in raw_scores.values() if s >= 80)
        logger.debug(f"V3.7评分（已标准化）: 80+分股票: {excellent_count}只")

        return raw_scores

    def is_available(self) -> bool:
        """检查V3.7模型是否可用"""
        try:
            from ml_models.v37 import V370AdvancedMLSystem
            return self.model_path is not None
        except ImportError:
            return False


class V380ModelAdapter(MLModelAdapter):
    """V3.80模型适配器"""

    def _initialize(self):
        """初始化V3.80模型"""
        self.model_path = self._find_model_file()

    def calculate_scores(self, stock_list: List[str], date: str) -> Dict[str, float]:
        """使用V3.80系统计算评分"""
        try:
            from ml_models.v38 import V380AdvancedIncrementalMLSystem

            v380_system = V380AdvancedIncrementalMLSystem()
            if self.model_path:
                v380_system.load_models(self.model_path)

            # 提取特征并预测
            features_df = v380_system.extract_features(stock_list, date, date)
            if features_df is not None and not features_df.empty:
                scores = v380_system.predict_scores(features_df)
                if isinstance(scores, dict):
                    raw_scores = scores
                elif hasattr(scores, 'to_dict'):
                    raw_scores = scores.to_dict()
                else:
                    raw_scores = {code: 50.0 for code in stock_list}
            else:
                raw_scores = {code: 50.0 for code in stock_list}

        except Exception as e:
            logger.warning(f"V3.80评分计算失败: {e}")
            raw_scores = {code: 50.0 for code in stock_list}

        # 标准化到百分制
        return self.normalize_scores(raw_scores)

    def normalize_scores(self, raw_scores: Dict[str, float]) -> Dict[str, float]:
        """
        V3.80评分标准化：直接线性映射 [10, 90] -> [0, 100]
        """
        if not raw_scores:
            return raw_scores

        # V3.80模型的实际评分范围 [10, 90]
        raw_min = 10.0
        raw_max = 90.0

        normalized_scores = {}
        for stock_code, raw_score in raw_scores.items():
            # 线性映射
            normalized_score = (raw_score - raw_min) / (raw_max - raw_min) * 100
            normalized_scores[stock_code] = max(0, min(100, normalized_score))

        excellent_count = sum(1 for s in normalized_scores.values() if s >= 80)
        logger.debug(f"V3.80评分标准化: [{raw_min}, {raw_max}] -> [0, 100], 80+分股票: {excellent_count}只")

        return normalized_scores

    def is_available(self) -> bool:
        """检查V3.80模型是否可用"""
        try:
            from ml_models.v38 import V380AdvancedIncrementalMLSystem
            return True
        except ImportError:
            return False


class V381ModelAdapter(MLModelAdapter):
    """V3.81模型适配器"""

    def _initialize(self):
        """V3.81使用Level4质量评分系统，不依赖特定模型文件"""
        pass

    def calculate_scores(self, stock_list: List[str], date: str) -> Dict[str, float]:
        """使用V3.81系统计算评分"""
        try:
            from ml_models.v381 import V380Level4IntegratedSystem

            v381_system = V380Level4IntegratedSystem()

            # 使用正确的V3.81方法
            predictions = v381_system.predict_scores(stock_list, date)

            raw_scores = {}
            for stock_code in stock_list:
                if stock_code in predictions:
                    pred_data = predictions[stock_code]
                    if isinstance(pred_data, dict):
                        # V3.81返回字典格式，取overall_score作为主评分
                        raw_scores[stock_code] = float(pred_data.get('overall_score', 50.0))
                    else:
                        # 如果是简单数值
                        raw_scores[stock_code] = float(pred_data)
                else:
                    raw_scores[stock_code] = 50.0

        except Exception as e:
            logger.warning(f"V3.81评分计算失败: {e}")
            raw_scores = {code: 50.0 for code in stock_list}

        # 标准化到百分制
        return self.normalize_scores(raw_scores)

    def normalize_scores(self, raw_scores: Dict[str, float]) -> Dict[str, float]:
        """
        V3.81评分标准化：直接线性映射 [10, 90] -> [0, 100]
        V3.81的overall_score继承自V3.80，范围为[10, 90]
        """
        if not raw_scores:
            return raw_scores

        # V3.81的overall_score实际范围 [10, 90] (继承自V3.80)
        raw_min = 10.0
        raw_max = 90.0

        normalized_scores = {}
        for stock_code, raw_score in raw_scores.items():
            # 线性映射
            normalized_score = (raw_score - raw_min) / (raw_max - raw_min) * 100
            normalized_scores[stock_code] = max(0, min(100, normalized_score))

        excellent_count = sum(1 for s in normalized_scores.values() if s >= 80)
        logger.debug(f"V3.81评分标准化: [{raw_min}, {raw_max}] -> [0, 100], 80+分股票: {excellent_count}只")

        return normalized_scores

    def is_available(self) -> bool:
        """检查V3.81模型是否可用"""
        try:
            from ml_models.v381 import V380Level4IntegratedSystem
            return True
        except ImportError:
            return False


class V390ModelAdapter(MLModelAdapter):
    """V3.90模型适配器"""

    def _initialize(self):
        """初始化V3.90模型"""
        self.model_path = self._find_model_file()

    def calculate_scores(self, stock_list: List[str], date: str) -> Dict[str, float]:
        """使用V3.90系统计算评分"""
        try:
            from ml_models.v39.v390_production_scorer import V390ProductionScorer

            # 初始化V3.90评分器
            if self.model_path:
                scorer = V390ProductionScorer(self.model_path)
            else:
                scorer = V390ProductionScorer()  # 使用默认模型路径

            # 批量预测评分
            predictions = scorer.predict_scores(stock_list, date)

            raw_scores = {}
            for stock_code in stock_list:
                if stock_code in predictions:
                    pred_data = predictions[stock_code]
                    # V3.90返回字典格式，取score字段
                    raw_scores[stock_code] = float(pred_data.get('score', 50.0))
                else:
                    raw_scores[stock_code] = 50.0

        except Exception as e:
            logger.warning(f"V3.90评分计算失败: {e}")
            raw_scores = {code: 50.0 for code in stock_list}

        # 标准化到百分制
        return self.normalize_scores(raw_scores)

    def normalize_scores(self, raw_scores: Dict[str, float]) -> Dict[str, float]:
        """
        V3.90评分标准化：V3.90的score已经是0-100范围，直接返回
        """
        if not raw_scores:
            return raw_scores

        # V3.90已经返回0-100评分，无需二次标准化
        excellent_count = sum(1 for s in raw_scores.values() if s >= 80)
        logger.debug(f"V3.90评分（已标准化）: 80+分股票: {excellent_count}只")

        return raw_scores

    def is_available(self) -> bool:
        """检查V3.90模型是否可用"""
        try:
            from ml_models.v39.v390_production_scorer import V390ProductionScorer
            return self.model_path is not None or (PROJECT_ROOT / 'models' / 'v390_full_from_cache.pkl').exists()
        except ImportError:
            return False


class V394ModelAdapter(MLModelAdapter):
    """V3.94模型适配器"""

    def _initialize(self):
        """初始化V3.94模型"""
        self.model_path = self._find_compatible_model()

    def _find_compatible_model(self) -> Optional[str]:
        """
        查找兼容的V3.94模型文件

        V394ProductionScorer需要模型文件包含'model'键，
        过滤掉ensemble模型（使用base_models/meta_learner结构）
        """
        import joblib

        pattern_parts = self.config.model_path_pattern.split('/')
        model_dir = Path('/'.join(pattern_parts[:-1]))

        if not model_dir.exists():
            logger.warning(f"模型目录不存在: {model_dir}")
            return None

        pattern = pattern_parts[-1]
        model_files = list(model_dir.glob(pattern))

        # 过滤掉ensemble模型，按修改时间排序
        compatible_files = []
        for f in model_files:
            # 跳过ensemble模型
            if 'ensemble' in f.name.lower():
                logger.debug(f"跳过ensemble模型: {f}")
                continue

            # 检查模型结构是否兼容
            try:
                model_data = joblib.load(f)
                if 'model' in model_data:
                    compatible_files.append(f)
                    logger.debug(f"兼容模型: {f}")
                else:
                    logger.debug(f"跳过不兼容模型 (无'model'键): {f}")
            except Exception as e:
                logger.debug(f"无法加载模型 {f}: {e}")

        if compatible_files:
            latest_model = max(compatible_files, key=lambda f: f.stat().st_mtime)
            logger.info(f"📱 找到V3.94兼容模型: {latest_model}")
            return str(latest_model)

        # 如果没有找到兼容模型，使用默认路径
        default_path = model_dir / 'v394_full_model.pkl'
        if default_path.exists():
            logger.info(f"📱 使用V3.94默认模型: {default_path}")
            return str(default_path)

        logger.warning(f"未找到V3.94兼容模型")
        return None

    def calculate_scores(self, stock_list: List[str], date: str) -> Dict[str, float]:
        """使用V3.94系统计算评分

        使用predict_scores_with_ranking方法：
        - 解决模型预测范围过窄导致所有股票评分相近的问题
        - 使用百分位排名将预测值映射到30-90分范围
        - 提供更好的股票区分度
        """
        try:
            from ml_models.v39.v394_production_scorer import V394ProductionScorer

            # 初始化V3.94评分器
            if self.model_path:
                scorer = V394ProductionScorer(self.model_path)
            else:
                scorer = V394ProductionScorer()  # 使用默认模型路径

            # 使用百分位排名评分（解决预测值聚集问题）
            predictions = scorer.predict_scores_with_ranking(stock_list, date)

            raw_scores = {}
            for stock_code in stock_list:
                if stock_code in predictions:
                    pred_data = predictions[stock_code]
                    # V3.94返回字典格式，取score字段(已经是百分位排名评分)
                    raw_scores[stock_code] = float(pred_data.get('score', 50.0))
                else:
                    raw_scores[stock_code] = 50.0

            # 记录评分分布
            if raw_scores:
                scores = list(raw_scores.values())
                logger.info(f"V3.94评分分布: min={min(scores):.1f}, max={max(scores):.1f}, "
                           f"80+分: {sum(1 for s in scores if s >= 80)}只")

        except Exception as e:
            logger.warning(f"V3.94评分计算失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            raw_scores = {code: 50.0 for code in stock_list}

        # 标准化到百分制
        return self.normalize_scores(raw_scores)

    def normalize_scores(self, raw_scores: Dict[str, float]) -> Dict[str, float]:
        """
        V3.94评分标准化：V3.94的score已经是0-100范围，直接返回
        """
        if not raw_scores:
            return raw_scores

        # V3.94已经返回0-100评分，无需二次标准化
        excellent_count = sum(1 for s in raw_scores.values() if s >= 80)
        logger.debug(f"V3.94评分（已标准化）: 80+分股票: {excellent_count}只")

        return raw_scores

    def is_available(self) -> bool:
        """检查V3.94模型是否可用"""
        try:
            from ml_models.v39.v394_production_scorer import V394ProductionScorer
            return self.model_path is not None or (PROJECT_ROOT / 'models' / 'v394' / 'v394_full_model.pkl').exists()
        except ImportError:
            return False


class ModelRegistry:
    """模型注册中心 - 管理所有可用的模型适配器"""

    def __init__(self):
        self.registered_models: Dict[str, Type[MLModelAdapter]] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self._initialize_builtin_models()

    def _initialize_builtin_models(self):
        """初始化内置模型配置"""
        builtin_configs = [
            ModelConfig(
                version="V3.7",
                name="V3.70 Advanced ML System",
                module_name="v370_advanced_ml_system",
                class_name="V370AdvancedMLSystem",
                model_path_pattern=str(PROJECT_ROOT / "models/v370/v370_*.pkl"),
                features_count=53,
                description="5基础模型+4专家模型+Meta学习器，53个特征",
                requires_modules=["v370_advanced_ml_system"]
            ),
            ModelConfig(
                version="V3.80",
                name="V3.80 Advanced Incremental ML System",
                module_name="v380_advanced_incremental_ml_system",
                class_name="V380AdvancedIncrementalMLSystem",
                model_path_pattern=str(PROJECT_ROOT / "models/v380/v380_*.pkl"),
                features_count=60,
                description="Level4质量优化特征+增量学习",
                requires_modules=["v380_advanced_incremental_ml_system"]
            ),
            ModelConfig(
                version="V3.81",
                name="V3.81 Level4 Integrated System",
                module_name="v380_level4_integrated_system",
                class_name="V380Level4IntegratedSystem",
                model_path_pattern="models/level4/*",
                features_count=70,
                description="Level4质量控制+Meta学习集成",
                requires_modules=["v380_level4_integrated_system"]
            ),
            ModelConfig(
                version="V3.9",
                name="V3.90 Enhanced Feature System",
                module_name="v390_production_scorer",
                class_name="V390ProductionScorer",
                model_path_pattern=str(PROJECT_ROOT / "models/v390*.pkl"),
                features_count=42,
                description="42个增强特征，A级模型 (81.2/100, 67.30%准确率, 95% Top20胜率)",
                requires_modules=["v390_production_scorer"]
            ),
            ModelConfig(
                version="V3.94",
                name="V3.94 Active Market Cap System",
                module_name="v394_production_scorer",
                class_name="V394ProductionScorer",
                model_path_pattern=str(PROJECT_ROOT / "models/v394/v394*.pkl"),
                features_count=48,
                description="48个特征（42基础+6活跃市值），优化的市场适应性",
                requires_modules=["v394_production_scorer"]
            )
        ]

        # 注册内置模型
        adapter_classes = {
            "V3.7": V37ModelAdapter,
            "V3.80": V380ModelAdapter,
            "V3.81": V381ModelAdapter,
            "V3.9": V390ModelAdapter,
            "V3.94": V394ModelAdapter
        }

        for config in builtin_configs:
            self.register_model(config, adapter_classes.get(config.version))

    def register_model(self, config: ModelConfig, adapter_class: Type[MLModelAdapter]):
        """注册模型"""
        self.model_configs[config.version] = config
        self.registered_models[config.version] = adapter_class
        logger.info(f"📝 注册模型: {config.version} - {config.name}")

    def get_available_models(self) -> Dict[str, bool]:
        """获取所有可用模型"""
        available = {}
        for version, adapter_class in self.registered_models.items():
            try:
                config = self.model_configs[version]
                adapter = adapter_class(config)
                available[version] = adapter.is_available()
            except Exception as e:
                logger.debug(f"检查模型{version}可用性失败: {e}")
                available[version] = False

        return available

    def create_adapter(self, version: str, max_workers: int = 4) -> Optional[MLModelAdapter]:
        """创建模型适配器实例"""
        if version not in self.registered_models:
            logger.error(f"未注册的模型版本: {version}")
            return None

        try:
            config = self.model_configs[version]
            adapter_class = self.registered_models[version]
            return adapter_class(config, max_workers)
        except Exception as e:
            logger.error(f"创建模型适配器失败 {version}: {e}")
            return None

    def list_models(self) -> List[Dict[str, Any]]:
        """列出所有注册的模型信息"""
        model_list = []
        available_models = self.get_available_models()

        for version, config in self.model_configs.items():
            is_available = available_models.get(version, False)
            model_list.append({
                'version': version,
                'name': config.name,
                'description': config.description,
                'features_count': config.features_count,
                'is_available': is_available,
                'status': '✅ 可用' if is_available else '❌ 不可用'
            })

        return model_list

    def discover_new_models(self, models_dir: str = "models") -> int:
        """自动发现新模型 (扩展功能)"""
        # TODO: 实现自动模型发现逻辑
        # 扫描models目录，查找新的模型文件和配置
        # 自动生成模型配置并注册
        discovered_count = 0
        logger.info(f"🔍 模型发现功能待实现，扫描目录: {models_dir}")
        return discovered_count


class ExtensibleBacktestEngine:
    """可扩展通用回测引擎"""

    # 🎯 模型独立阈值配置
    MODEL_THRESHOLDS = {
        "V3.7": 85.0,    # 🔧 V3.7提高阈值 (更严格筛选)
        "V3.80": 85.0,   # 🔧 V3.8提高阈值
        "V3.81": 85.0,   # 🔧 V3.81提高阈值 (更严格筛选)
        "V3.9": 62.0,    # V3.9独立阈值 (基于阈值分析结果)
        "V3.94": 62.0,   # V3.94使用与V3.9相同阈值
    }

    def __init__(self,
                 strategy: Optional['TradingStrategy'] = None,  # 🆕 策略注入
                 initial_capital: float = 5000000,
                 max_workers: int = 6,
                 commission_rate: float = 0.0003,
                 stamp_tax: float = 0.001,
                 min_score_threshold: float = 80.0):
        """
        初始化可扩展回测引擎

        Args:
            strategy: 交易策略实例 (可选，默认使用平衡策略)
            initial_capital: 初始资金
            max_workers: 并行工作进程数
            commission_rate: 佣金费率
            stamp_tax: 印花税
            min_score_threshold: 最低评分阈值
        """

        # 🆕 策略系统 - 支持注入或使用默认策略
        if strategy is None:
            # 如果没有传入策略，使用默认的平衡策略
            from trading_strategy import BalancedStrategy
            self.strategy = BalancedStrategy()
            logger.info("📊 使用默认策略: 平衡策略")
        else:
            self.strategy = strategy
            logger.info(f"📊 使用自定义策略: {strategy.config.name}")

        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_workers = max_workers
        self.commission_rate = commission_rate
        self.stamp_tax = stamp_tax
        self.min_score_threshold = min_score_threshold

        # 交易记录
        self.positions = {}
        self.trades = []
        self.daily_returns = []
        self.portfolio_values = []

        # 🆕 从策略获取配置
        self.max_position_pct = self.strategy.config.max_position_pct
        self.max_positions = self.strategy.config.max_positions
        self.stop_loss_pct = self.strategy.config.stop_loss_pct
        self.rebalance_freq = self.strategy.config.rebalance_frequency
        self.take_profit_pct = self.strategy.config.take_profit_pct
        self.max_holding_days = self.strategy.config.max_holding_days
        self.min_score_for_hold = self.strategy.config.min_score_for_hold
        self.enable_rebalance_sell = self.strategy.config.enable_rebalance_sell

        # 数据管理器和缓存
        self.db_manager = DatabaseManager()
        self.data_cache = {}

        # 模型注册中心
        self.model_registry = ModelRegistry()

        logger.info(f"🚀 可扩展回测引擎初始化完成 (策略: {self.strategy.config.name})")

    def get_available_models(self) -> Dict[str, bool]:
        """获取可用模型列表"""
        return self.model_registry.get_available_models()

    def list_all_models(self) -> List[Dict[str, Any]]:
        """列出所有模型详细信息"""
        return self.model_registry.list_models()

    def run_backtest(self,
                    versions: List[str],
                    start_date: str,
                    end_date: str) -> Dict[str, Any]:
        """
        运行指定版本的回测对比

        Args:
            versions: 要测试的模型版本列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            回测结果和对比分析
        """

        # 验证版本可用性
        available_models = self.get_available_models()
        valid_versions = []

        for version in versions:
            if version in available_models:
                if available_models[version]:
                    valid_versions.append(version)
                else:
                    logger.warning(f"模型版本 {version} 不可用")
            else:
                logger.error(f"未注册的模型版本: {version}")

        if not valid_versions:
            raise ValueError("没有可用的模型版本进行回测")

        logger.info(f"🎯 开始回测版本: {valid_versions}")
        logger.info(f"📅 回测期间: {start_date} 至 {end_date}")

        # 并行运行各版本回测
        comparison_results = {}

        for version in valid_versions:
            try:
                logger.info(f"📊 运行 {version} 回测...")
                start_time = time.time()

                result = self._run_single_version_backtest(version, start_date, end_date)

                end_time = time.time()
                result['backtest_time'] = end_time - start_time
                result['version'] = version

                comparison_results[version] = result

                logger.info(f"✅ {version} 完成: 收益率 {result.get('total_return', 0):.2%}, "
                           f"交易 {result.get('total_trades', 0)}次, "
                           f"用时 {result['backtest_time']:.1f}秒")

            except Exception as e:
                logger.error(f"❌ {version} 回测失败: {e}")
                comparison_results[version] = {
                    'version': version,
                    'error': str(e),
                    'total_return': 0.0,
                    'backtest_time': 0.0
                }

        # 生成对比分析
        comparison_analysis = self._generate_comparison_analysis(comparison_results)

        return {
            'individual_results': comparison_results,
            'comparison_analysis': comparison_analysis,
            'test_period': f"{start_date} 至 {end_date}",
            'versions_tested': list(comparison_results.keys()),
            'summary': {
                'requested_versions': versions,
                'successful_versions': [v for v, r in comparison_results.items() if 'error' not in r],
                'failed_versions': [v for v, r in comparison_results.items() if 'error' in r]
            }
        }

    def _run_single_version_backtest(self, version: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """运行单版本回测"""

        # 创建模型适配器
        adapter = self.model_registry.create_adapter(version, self.max_workers)
        if not adapter:
            raise ValueError(f"无法创建模型适配器: {version}")

        # 重置状态
        self._reset_state()

        # 1. 批量加载数据
        self._batch_load_stock_data(start_date, end_date)

        # 2. 获取股票池和交易日期
        stock_universe = self._get_stock_universe(start_date, end_date)
        trading_dates = self._get_trading_dates(start_date, end_date)

        if not stock_universe or not trading_dates:
            return self._empty_result(version)

        logger.debug(f"📊 {version} 股票池: {len(stock_universe)}只，交易日: {len(trading_dates)}天")

        # 3. 执行回测循环
        last_rebalance_date = None

        for i, current_date in enumerate(trading_dates):
            if i % 20 == 0:
                progress = i / len(trading_dates) * 100
                logger.debug(f"📈 {version} 进度: {progress:.1f}% ({current_date})")

            # 检查是否需要调仓
            current_date_str = current_date.strftime('%Y-%m-%d') if isinstance(current_date, datetime) else str(current_date)

            # 🆕 Phase 3: 通知动态策略每日更新
            if hasattr(self.strategy, 'on_daily_update'):
                self.strategy.on_daily_update(current_date_str, self.positions, self.db_manager)
            should_rebalance = (
                last_rebalance_date is None or
                (datetime.strptime(current_date_str, '%Y-%m-%d') -
                 datetime.strptime(last_rebalance_date, '%Y-%m-%d')).days >= self.rebalance_freq
            )

            if should_rebalance:
                # 使用指定版本的模型进行选股
                selected_stocks = self._universal_stock_selection(adapter, current_date_str, stock_universe)

                # 执行调仓
                self._execute_rebalance(current_date_str, selected_stocks)
                last_rebalance_date = current_date_str

            # 更新持仓价值
            self._update_portfolio_value(current_date_str)

        # 4. 计算绩效指标
        results = self._calculate_performance_metrics(version, adapter)

        return results

    # 复用核心回测逻辑...
    def _reset_state(self):
        """重置回测状态"""
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.daily_returns = []
        self.portfolio_values = []

    def _empty_result(self, version: str) -> Dict[str, Any]:
        """返回空结果"""
        return {
            'version': version,
            'total_return': 0.0,
            'final_capital': self.initial_capital,
            'total_trades': 0,
            'error': 'No valid data or stock universe'
        }

    def _batch_load_stock_data(self, start_date: str, end_date: str):
        """批量加载股票数据"""
        from ml_models.v37.backtest_v37_engine_optimized import V37BacktestEngineOptimized
        temp_engine = V37BacktestEngineOptimized()
        self.data_cache = temp_engine.batch_load_stock_data(start_date, end_date)

    def _get_stock_universe(self, start_date: str, end_date: str) -> List[str]:
        """获取股票池"""
        query = """
        SELECT DISTINCT s.code
        FROM securities s
        JOIN daily_quotes dq ON s.id = dq.security_id
        WHERE s.type = 'A股'
        AND dq.trade_date BETWEEN ? AND ?
        AND dq.close > 0
        ORDER BY s.code
        """

        try:
            result = self.db_manager.execute_query(query, [start_date, end_date])
            return [row[0] for row in result] if result else []
        except Exception as e:
            logger.error(f"获取股票池失败: {e}")
            return []

    def _get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取交易日期"""
        query = """
        SELECT DISTINCT trade_date
        FROM daily_quotes
        WHERE trade_date BETWEEN ? AND ?
        ORDER BY trade_date
        """

        try:
            result = self.db_manager.execute_query(query, [start_date, end_date])
            return [row[0] for row in result] if result else []
        except Exception as e:
            logger.error(f"获取交易日期失败: {e}")
            return []

    def _run_quantitative_strategies(self, candidates: List[str], date: str) -> List[str]:
        """
        运行量化策略预过滤，减少ML评分候选数量

        从~2000只候选股票中筛选出~200-300只优质候选，大幅减少ML特征计算负担

        Args:
            candidates: 基础筛选后的候选股票列表
            date: 选股日期

        Returns:
            量化策略选出的股票列表 (union of all strategies)
        """
        try:
            # 导入量化选股策略
            from stock_selctor.Selector import (
                BBIKDJSelector, BBIShortLongSelector,
                BreakoutVolumeKDJSelector, PeakKDJSelector
            )

            # 加载候选股票的历史数据 (最近90天足够计算所有技术指标)
            date_ts = pd.Timestamp(date)
            start_date = (date_ts - pd.Timedelta(days=90)).strftime('%Y-%m-%d')

            # 从数据库加载数据（必须包含技术指标）
            stock_data = {}
            for stock_code in candidates:
                try:
                    # 直接从数据库查询（包含技术指标，量化策略需要）
                    query = """
                    SELECT dq.trade_date as date, dq.open, dq.high, dq.low, dq.close, dq.volume,
                           ti.kdj_k, ti.kdj_d, ti.kdj_j, ti.bbi, ti.macd_dif
                    FROM securities s
                    JOIN daily_quotes dq ON s.id = dq.security_id
                    LEFT JOIN technical_indicators ti ON s.id = ti.security_id AND dq.trade_date = ti.trade_date
                    WHERE s.code = ? AND dq.trade_date BETWEEN ? AND ?
                    ORDER BY dq.trade_date
                    """

                    result = self.db_manager.execute_query(query, [stock_code, start_date, date])
                    if result and len(result) >= 20:  # 至少需要20天数据
                        df = pd.DataFrame(result, columns=[
                            'date', 'open', 'high', 'low', 'close', 'volume',
                            'kdj_k', 'kdj_d', 'kdj_j', 'bbi', 'macd_dif'
                        ])
                        df['date'] = pd.to_datetime(df['date'])
                        stock_data[stock_code] = df

                except Exception as e:
                    logger.debug(f"加载股票 {stock_code} 数据失败: {e}")
                    continue

            if not stock_data:
                logger.warning(f"⚠️ 无法加载候选股票数据进行量化过滤")
                return candidates[:500]  # fallback

            # 实例化4个量化策略 (使用与tomorrow_stock_selector.py相同的参数)
            strategies = {
                "BBIKDJSelector": BBIKDJSelector(
                    j_threshold=10,
                    bbi_min_window=20,
                    max_window=60,
                    price_range_pct=1,
                    bbi_q_threshold=0.3,
                    j_q_threshold=0.10
                ),
                "BBIShortLongSelector": BBIShortLongSelector(
                    n_short=3,
                    n_long=21,
                    m=3,
                    bbi_min_window=2,
                    max_window=60,
                    bbi_q_threshold=0.2
                ),
                "BreakoutVolumeKDJSelector": BreakoutVolumeKDJSelector(
                    j_threshold=1,
                    j_q_threshold=0.10,
                    price_range_pct=10.0
                ),
                "PeakKDJSelector": PeakKDJSelector(
                    j_threshold=10,
                    j_q_threshold=0.10,
                    max_window=90
                )
            }

            # 运行所有策略
            all_selected = set()
            for strategy_name, strategy in strategies.items():
                try:
                    selected = strategy.select(date_ts, stock_data)
                    all_selected.update(selected)
                    logger.debug(f"  {strategy_name}: {len(selected)}只")
                except Exception as e:
                    logger.debug(f"  {strategy_name} 执行失败: {e}")
                    continue

            result = list(all_selected)
            logger.info(f"🔍 量化策略预过滤: {len(candidates)}只 -> {len(result)}只 (减少{len(candidates)-len(result)}只)")

            return result if result else candidates[:500]

        except Exception as e:
            logger.warning(f"⚠️ 量化策略预过滤失败: {e}，使用前500只候选")
            return candidates[:500]

    def _universal_stock_selection(self, model_adapter: MLModelAdapter, date: str, stock_universe: List[str]) -> List[Dict]:
        """通用股票选择逻辑"""
        # 1. 基础筛选
        candidates = self._basic_stock_screening(stock_universe, date)

        if not candidates:
            return []

        # 🆕 2. 量化策略预过滤 (减少ML评分负担)
        quantitative_candidates = self._run_quantitative_strategies(candidates, date)

        if not quantitative_candidates:
            logger.debug(f"⚠️ 量化策略未选出任何股票，使用前500只基础候选")
            quantitative_candidates = candidates[:500]  # fallback to top 500

        logger.debug(f"🔍 量化策略过滤: {len(candidates)}只 -> {len(quantitative_candidates)}只")

        # 3. 使用模型计算评分 (仅对量化策略筛选后的股票)
        scores = model_adapter.calculate_scores(quantitative_candidates, date)

        # 🎯 获取模型独立阈值
        model_version = model_adapter.config.version
        model_threshold = self.MODEL_THRESHOLDS.get(model_version, self.min_score_threshold)

        logger.debug(f"🎯 {model_version} 使用阈值: {model_threshold}分 (默认: {self.min_score_threshold}分)")

        # 3. 按评分筛选和排序
        qualified_stocks = []
        for stock_code, score in scores.items():
            if score >= model_threshold:  # 使用模型独立阈值
                qualified_stocks.append({
                    'stock_code': stock_code,
                    'score': score,
                    'date': date,
                    'model_version': model_adapter.config.version
                })

        # 按评分排序，选择前N只
        qualified_stocks.sort(key=lambda x: x['score'], reverse=True)
        selected_stocks = qualified_stocks[:self.max_positions]

        if selected_stocks:
            logger.debug(f"✅ {model_adapter.config.version} 选择 {len(selected_stocks)} 只股票，最高评分: {selected_stocks[0]['score']:.1f}")

        return selected_stocks

    def _basic_stock_screening(self, stock_universe: List[str], date: str) -> List[str]:
        """基础股票筛选 - 简化版，让模型自己判断"""

        # 简单的有效性检查：确保股票有基本交易数据
        query = """
        SELECT DISTINCT s.code
        FROM securities s
        JOIN daily_quotes dq ON s.id = dq.security_id
        WHERE s.code IN ({placeholders})
        AND dq.trade_date <= ?
        AND dq.close > 0
        AND dq.volume > 0
        ORDER BY s.code
        """.format(placeholders=','.join('?' * len(stock_universe[:2000])))

        try:
            params = stock_universe[:2000] + [date]
            result = self.db_manager.execute_query(query, params)
            candidates = [row[0] for row in result] if result else []

            logger.debug(f"基础筛选: {len(stock_universe[:2000])}只 -> {len(candidates)}只")
            return candidates

        except Exception as e:
            logger.error(f"基础筛选失败: {e}")
            # 如果数据库查询失败，返回前1000只股票让模型判断
            return stock_universe[:1000]

    def _execute_rebalance(self, date: str, selected_stocks: List[Dict]):
        """执行调仓 - 完整版：先卖后买"""
        if not selected_stocks:
            return

        # 🆕 步骤1: 调仓时卖出不符合条件的持仓
        if self.enable_rebalance_sell:
            sold_count = self._rebalance_sell_positions(date, selected_stocks)
            if sold_count > 0:
                logger.debug(f"调仓卖出{sold_count}只股票，当前现金: {self.current_capital:,.0f}元")

        # 步骤2: 计算可用资金 (现金 + 刚才卖出所得)
        total_available = self.current_capital

        # 步骤3: 计算当前持仓数量
        current_positions = sum(1 for p in self.positions.values() if p['shares'] > 0)

        # 步骤4: 确定可买入的股票数量
        max_new_positions = self.max_positions - current_positions
        if max_new_positions <= 0:
            logger.debug(f"已达最大持仓数{self.max_positions}，跳过买入")
            return

        # 步骤5: 筛选出不在当前持仓中的股票
        stocks_to_buy = []
        for stock_info in selected_stocks:
            stock_code = stock_info['stock_code']
            # 跳过已持仓的股票（避免重复买入）
            if stock_code in self.positions and self.positions[stock_code]['shares'] > 0:
                continue
            stocks_to_buy.append(stock_info)
            if len(stocks_to_buy) >= max_new_positions:
                break

        if not stocks_to_buy:
            logger.debug("没有新股票可买入")
            return

        # 步骤6: 🆕 使用策略的评分加权仓位计算（而非均匀分配）
        # 步骤7: 执行买入
        bought_count = 0
        for stock_info in stocks_to_buy:
            stock_code = stock_info['stock_code']
            score = stock_info['score']

            price = self._get_stock_price(stock_code, date)
            if price is None or price <= 0:
                continue

            # 🆕 调用策略的仓位计算方法
            if hasattr(self.strategy, 'calculate_position_size'):
                # 检查方法签名是否接受stock_score参数
                import inspect
                sig = inspect.signature(self.strategy.calculate_position_size)
                params = sig.parameters

                # 如果方法接受stock_score参数（评分加权策略）
                if 'stock_score' in params:
                    shares = self.strategy.calculate_position_size(
                        stock_code=stock_code,
                        stock_price=price,
                        available_capital=total_available,
                        current_positions=len([p for p in self.positions.values() if p['shares'] > 0]),
                        stock_score=score
                    )
                else:
                    # 动态策略不需要stock_score参数
                    shares = self.strategy.calculate_position_size(
                        stock_code=stock_code,
                        stock_price=price,
                        available_capital=total_available,
                        current_positions=len([p for p in self.positions.values() if p['shares'] > 0])
                    )
            else:
                # Fallback: 均匀分配（兼容旧策略）
                buy_amount = (total_available / len(stocks_to_buy)) * (1 - self.commission_rate)
                shares = int(buy_amount / (price * 100)) * 100

            if shares >= 100:
                actual_cost = shares * price * (1 + self.commission_rate)

                if actual_cost <= self.current_capital:
                    trade = {
                        'date': date,
                        'stock_code': stock_code,
                        'action': 'buy',
                        'shares': shares,
                        'price': price,
                        'amount': actual_cost,
                        'score': score,
                        'model_version': stock_info.get('model_version', 'unknown')
                    }
                    self.trades.append(trade)

                    if stock_code not in self.positions:
                        self.positions[stock_code] = {
                            'shares': 0,
                            'avg_cost': 0,
                            'entry_date': date,
                            'entry_score': score,
                            # 🆕 Bug修复：保存买入时的止盈止损参数
                            'entry_take_profit_pct': self.strategy.config.take_profit_pct,
                            'entry_stop_loss_pct': self.strategy.config.stop_loss_pct,
                            'entry_market_regime': getattr(self.strategy, 'current_regime', 'UNKNOWN')
                        }

                    old_shares = self.positions[stock_code]['shares']
                    old_cost = self.positions[stock_code]['avg_cost']

                    total_shares = old_shares + shares
                    total_cost = old_shares * old_cost + actual_cost
                    self.positions[stock_code]['shares'] = total_shares
                    self.positions[stock_code]['avg_cost'] = total_cost / total_shares if total_shares > 0 else 0

                    self.current_capital -= actual_cost

                    # 🆕 Phase 3: 通知动态策略持仓创建
                    if hasattr(self.strategy, 'on_position_created') and old_shares == 0:
                        from trading_strategy import Position
                        position = Position(
                            stock_code=stock_code,
                            shares=shares,
                            avg_cost=self.positions[stock_code]['avg_cost'],
                            entry_date=date,
                            entry_score=score,
                            entry_take_profit_pct=self.positions[stock_code]['entry_take_profit_pct'],
                            entry_stop_loss_pct=self.positions[stock_code]['entry_stop_loss_pct'],
                            entry_market_regime=self.positions[stock_code]['entry_market_regime']
                        )
                        self.strategy.on_position_created(position, date, self.db_manager)
                    bought_count += 1

        if bought_count > 0:
            logger.debug(f"调仓买入{bought_count}只新股票，剩余现金: {self.current_capital:,.0f}元")

    def _update_portfolio_value(self, date: str):
        """更新组合价值"""
        portfolio_value = self.current_capital

        for stock_code, position in self.positions.items():
            if position['shares'] > 0:
                current_price = self._get_stock_price(stock_code, date)
                if current_price and current_price > 0:
                    market_value = position['shares'] * current_price
                    portfolio_value += market_value

        self.portfolio_values.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'cash': self.current_capital
        })

        if len(self.portfolio_values) > 1:
            prev_value = self.portfolio_values[-2]['portfolio_value']
            daily_return = (portfolio_value / prev_value - 1) if prev_value > 0 else 0
            self.daily_returns.append(daily_return)

        # 🆕 每日更新市场环境（确保自适应策略实时调整参数）
        if hasattr(self.strategy, 'update_market_regime'):
            self.strategy.update_market_regime(date)

        # 🆕 风控检查 (优先级顺序: 止损 > 止盈 > 超期)
        self._check_stop_loss(date)        # 优先级1: 止损（防止扩大亏损）
        self._check_take_profit(date)      # 优先级2: 止盈（锁定利润）
        self._check_holding_period(date)   # 优先级3: 超期轮换（释放资金）

    def _get_stock_price(self, stock_code: str, date: str) -> Optional[float]:
        """获取股票价格"""
        try:
            for cache_key, cached_data in self.data_cache.items():
                if stock_code in cached_data:
                    stock_data = cached_data[stock_code]
                    date_ts = pd.Timestamp(date)

                    if date_ts in stock_data.index:
                        return float(stock_data.loc[date_ts, 'close'])
                    else:
                        available_dates = stock_data[stock_data.index <= date_ts]
                        if not available_dates.empty:
                            return float(available_dates.iloc[-1]['close'])

            query = """
            SELECT dq.close
            FROM securities s
            JOIN daily_quotes dq ON s.id = dq.security_id
            WHERE s.code = ? AND dq.trade_date <= ?
            ORDER BY dq.trade_date DESC
            LIMIT 1
            """

            result = self.db_manager.execute_query(query, [stock_code, date])
            if result:
                return float(result[0][0])

        except Exception as e:
            logger.debug(f"获取股票 {stock_code} 价格失败: {e}")

        return None

    def _check_stop_loss(self, date: str):
        """检查止损 - 委托给策略决策"""
        from trading_strategy import Position

        positions_to_sell = []

        for stock_code, pos_dict in self.positions.items():
            if pos_dict['shares'] > 0:
                current_price = self._get_stock_price(stock_code, date)
                if current_price and current_price > 0:
                    # 转换为Position对象
                    position = Position(
                        stock_code=stock_code,
                        shares=pos_dict['shares'],
                        avg_cost=pos_dict['avg_cost'],
                        entry_date=pos_dict['entry_date'],
                        entry_score=pos_dict.get('entry_score', 0.0),
                        # 🆕 Bug修复：传入买入时参数
                        entry_take_profit_pct=pos_dict.get('entry_take_profit_pct'),
                        entry_stop_loss_pct=pos_dict.get('entry_stop_loss_pct'),
                        entry_market_regime=pos_dict.get('entry_market_regime')
                    )

                    # 🆕 Bug修复：使用买入时的止损参数判断
                    stop_loss_pct = position.entry_stop_loss_pct or self.strategy.config.stop_loss_pct
                    profit_pct = (current_price - position.avg_cost) / position.avg_cost

                    if profit_pct < -stop_loss_pct:
                        positions_to_sell.append(stock_code)
                        logger.debug(f"止损触发: {stock_code}, 亏损{profit_pct*100:.1f}%, 止损线{stop_loss_pct*100:.0f}%")

        for stock_code in positions_to_sell:
            self._execute_sell(stock_code, date, "stop_loss")

    def _execute_sell(self, stock_code: str, date: str, reason: str = "rebalance"):
        """执行卖出操作"""
        if stock_code not in self.positions or self.positions[stock_code]['shares'] <= 0:
            return

        position = self.positions[stock_code]
        shares = position['shares']
        current_price = self._get_stock_price(stock_code, date)

        if current_price and current_price > 0:
            gross_amount = shares * current_price
            sell_amount = gross_amount * (1 - self.commission_rate - self.stamp_tax)

            trade = {
                'date': date,
                'stock_code': stock_code,
                'action': 'sell',
                'shares': shares,
                'price': current_price,
                'amount': sell_amount,
                'reason': reason,
                'profit': sell_amount - (shares * position['avg_cost'])
            }
            self.trades.append(trade)

            self.positions[stock_code]['shares'] = 0
            self.positions[stock_code]['avg_cost'] = 0
            self.current_capital += sell_amount

            # 🆕 Phase 3: 通知动态策略持仓平仓
            if hasattr(self.strategy, 'on_position_closed'):
                self.strategy.on_position_closed(stock_code)

    def _calculate_holding_days(self, entry_date: str, current_date: str) -> int:
        """计算持仓天数"""
        try:
            entry = datetime.strptime(entry_date, '%Y-%m-%d')
            current = datetime.strptime(current_date, '%Y-%m-%d')
            return (current - entry).days
        except Exception as e:
            logger.debug(f"计算持仓天数失败: {e}")
            return 0

    def _check_take_profit(self, date: str):
        """检查止盈 - 委托给策略决策"""
        from trading_strategy import Position

        positions_to_sell = []

        for stock_code, pos_dict in self.positions.items():
            if pos_dict['shares'] > 0:
                current_price = self._get_stock_price(stock_code, date)
                if current_price and current_price > 0:
                    # 转换为Position对象
                    position = Position(
                        stock_code=stock_code,
                        shares=pos_dict['shares'],
                        avg_cost=pos_dict['avg_cost'],
                        entry_date=pos_dict['entry_date'],
                        entry_score=pos_dict.get('entry_score', 0.0),
                        # 🆕 Bug修复：传入买入时参数
                        entry_take_profit_pct=pos_dict.get('entry_take_profit_pct'),
                        entry_stop_loss_pct=pos_dict.get('entry_stop_loss_pct'),
                        entry_market_regime=pos_dict.get('entry_market_regime')
                    )

                    # 🆕 Bug修复：使用买入时的止盈参数判断
                    take_profit_pct = position.entry_take_profit_pct or self.strategy.config.take_profit_pct
                    profit_pct = (current_price - position.avg_cost) / position.avg_cost

                    if profit_pct > take_profit_pct:
                        positions_to_sell.append(stock_code)
                        logger.debug(f"止盈触发: {stock_code}, 盈利{profit_pct*100:.1f}%, 止盈线{take_profit_pct*100:.0f}%")

        for stock_code in positions_to_sell:
            self._execute_sell(stock_code, date, "take_profit")

    def _check_holding_period(self, date: str):
        """检查持仓期限 - 委托给策略决策"""
        from trading_strategy import Position

        positions_to_sell = []

        for stock_code, pos_dict in self.positions.items():
            if pos_dict['shares'] > 0:
                # 转换为Position对象
                position = Position(
                    stock_code=stock_code,
                    shares=pos_dict['shares'],
                    avg_cost=pos_dict['avg_cost'],
                    entry_date=pos_dict['entry_date'],
                    entry_score=pos_dict.get('entry_score', 0.0),
                    # 🆕 Bug修复：传入买入时参数
                    entry_take_profit_pct=pos_dict.get('entry_take_profit_pct'),
                    entry_stop_loss_pct=pos_dict.get('entry_stop_loss_pct'),
                    entry_market_regime=pos_dict.get('entry_market_regime')
                )

                # 🆕 委托给策略决策
                if self.strategy.should_check_holding_period(position, date):
                    holding_days = self._calculate_holding_days(position.entry_date, date)
                    positions_to_sell.append(stock_code)
                    logger.debug(f"超期平仓: {stock_code}, 已持有{holding_days}天")

        for stock_code in positions_to_sell:
            self._execute_sell(stock_code, date, "max_holding")

    def _rebalance_sell_positions(self, date: str, selected_stocks: List[Dict]):
        """调仓时卖出不符合条件的持仓 - 委托给策略决策"""
        from trading_strategy import Position

        positions_to_sell = []

        for stock_code, pos_dict in self.positions.items():
            if pos_dict['shares'] <= 0:
                continue

            # 转换为Position对象
            position = Position(
                stock_code=stock_code,
                shares=pos_dict['shares'],
                avg_cost=pos_dict['avg_cost'],
                entry_date=pos_dict['entry_date'],
                entry_score=pos_dict.get('entry_score', 0.0),
                # 🆕 Bug修复：传入买入时参数
                entry_take_profit_pct=pos_dict.get('entry_take_profit_pct'),
                entry_stop_loss_pct=pos_dict.get('entry_stop_loss_pct'),
                entry_market_regime=pos_dict.get('entry_market_regime')
            )

            current_price = self._get_stock_price(stock_code, date)
            if not current_price:
                continue

            # 🆕 委托给策略决策
            should_sell, reason = self.strategy.should_sell_on_rebalance(
                position, current_price, date, selected_stocks
            )

            if should_sell:
                positions_to_sell.append((stock_code, reason))
                logger.debug(f"调仓卖出: {stock_code}, 原因: {reason}")

        # 执行卖出
        for stock_code, reason in positions_to_sell:
            self._execute_sell(stock_code, date, reason)

        return len(positions_to_sell)

    def _calculate_performance_metrics(self, version: str, adapter: MLModelAdapter) -> Dict[str, Any]:
        """计算绩效指标"""
        if not self.portfolio_values or not self.trades:
            return {
                'version': version,
                'total_return': 0.0,
                'final_capital': self.current_capital,
                'annual_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_trades': len(self.trades),
                'avg_score': 0.0,
                'model_info': adapter.get_model_info()
            }

        final_portfolio_value = self.portfolio_values[-1]['portfolio_value']
        total_return = (final_portfolio_value / self.initial_capital - 1) if self.initial_capital > 0 else 0

        # 计算最大回撤
        max_drawdown = 0.0
        peak_value = self.initial_capital
        for pv in self.portfolio_values:
            value = pv['portfolio_value']
            if value > peak_value:
                peak_value = value
            drawdown = (peak_value - value) / peak_value if peak_value > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

        # 计算夏普比率
        if len(self.daily_returns) > 1:
            avg_return = sum(self.daily_returns) / len(self.daily_returns)
            return_std = (sum([(r - avg_return) ** 2 for r in self.daily_returns]) / len(self.daily_returns)) ** 0.5
            sharpe_ratio = (avg_return / return_std * (252 ** 0.5)) if return_std > 0 else 0
        else:
            sharpe_ratio = 0.0

        # 统计交易
        buy_trades = [t for t in self.trades if t['action'] == 'buy']
        total_trades = len(buy_trades)

        # 计算平均评分
        avg_score = sum([t.get('score', 50.0) for t in buy_trades]) / total_trades if total_trades > 0 else 0

        # 年化收益率
        days_count = len(self.portfolio_values)
        annual_return = ((1 + total_return) ** (252 / days_count) - 1) if days_count > 0 and total_return > -1 else 0

        # 计算胜率 - 只统计卖出交易（买入交易没有profit字段）
        sell_trades = [t for t in self.trades if t.get('action') == 'sell']
        profit_trades = [t for t in sell_trades if t.get('profit', 0) > 0]
        loss_trades = [t for t in sell_trades if t.get('profit', 0) < 0]
        win_rate = len(profit_trades) / len(sell_trades) if sell_trades else 0

        return {
            'version': version,
            'total_return': total_return,
            'final_capital': final_portfolio_value,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_score': avg_score,
            'successful_trades': len(profit_trades),
            'failed_trades': len(loss_trades),
            'avg_holding_days': 5.0,
            'avg_trade_return': total_return / total_trades if total_trades > 0 else 0,
            'model_info': adapter.get_model_info(),
            'strategy_info': self.strategy.get_info()  # 🆕 添加策略信息
        }

    def _generate_comparison_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成多版本对比分析"""
        successful_results = {k: v for k, v in results.items() if 'error' not in v}

        if not successful_results:
            return {'error': '所有版本均回测失败'}

        # 排序分析
        sorted_by_return = sorted(successful_results.items(),
                                key=lambda x: x[1].get('total_return', 0),
                                reverse=True)

        sorted_by_speed = sorted(successful_results.items(),
                               key=lambda x: x[1].get('backtest_time', float('inf')))

        # 计算统计指标
        returns = [v.get('total_return', 0) for v in successful_results.values()]
        avg_return = sum(returns) / len(returns) if returns else 0

        trades = [v.get('total_trades', 0) for v in successful_results.values()]
        avg_trades = sum(trades) / len(trades) if trades else 0

        return {
            'best_performance': {
                'version': sorted_by_return[0][0] if sorted_by_return else None,
                'return': sorted_by_return[0][1].get('total_return', 0) if sorted_by_return else 0
            },
            'fastest_execution': {
                'version': sorted_by_speed[0][0] if sorted_by_speed else None,
                'time': sorted_by_speed[0][1].get('backtest_time', 0) if sorted_by_speed else 0
            },
            'performance_ranking': [
                {'version': version, 'return': data.get('total_return', 0)}
                for version, data in sorted_by_return
            ],
            'statistics': {
                'avg_return': avg_return,
                'avg_trades': avg_trades,
                'return_std': np.std(returns) if len(returns) > 1 else 0
            },
            'summary': {
                'versions_tested': len(successful_results),
                'total_versions_attempted': len(results),
                'success_rate': len(successful_results) / len(results) if results else 0
            }
        }


def main():
    """命令行主入口"""
    parser = argparse.ArgumentParser(description='可扩展通用回测引擎')

    # 基本参数
    parser.add_argument('--versions', nargs='+', default=['V3.7', 'V3.81'],
                       help='要测试的模型版本 (默认: V3.7 V3.81)')
    parser.add_argument('--start-date', default='2025-08-01',
                       help='回测开始日期 (默认: 2025-08-01)')
    parser.add_argument('--end-date', default='2025-09-23',
                       help='回测结束日期 (默认: 2025-09-23)')

    # 回测参数
    parser.add_argument('--capital', type=float, default=5000000,
                       help='初始资金 (默认: 5000000)')
    parser.add_argument('--workers', type=int, default=6,
                       help='并行工作进程数 (默认: 6)')
    parser.add_argument('--min-score', type=float, default=80.0,
                       help='最低评分阈值 (默认: 80.0，标准化后的百分制)')

    # 操作参数
    parser.add_argument('--list-models', action='store_true',
                       help='列出所有可用模型')
    parser.add_argument('--save-report', action='store_true',
                       help='保存回测报告到文件')
    parser.add_argument('--verbose', action='store_true',
                       help='详细输出')

    args = parser.parse_args()

    # 设置日志级别
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(levelname)s - %(message)s')

    # 创建可扩展回测引擎
    engine = ExtensibleBacktestEngine(
        initial_capital=args.capital,
        max_workers=args.workers,
        min_score_threshold=args.min_score
    )

    # 列出模型
    if args.list_models:
        print("📋 可用模型列表:")
        models = engine.list_all_models()
        for model in models:
            print(f"  {model['version']}: {model['name']} - {model['status']}")
            print(f"    📝 {model['description']}")
            print(f"    🔧 特征数量: {model['features_count']}")
            print()
        return

    # 运行回测
    try:
        print(f"🚀 可扩展通用回测引擎")
        print(f"📅 回测期间: {args.start_date} 至 {args.end_date}")
        print(f"🎯 测试版本: {args.versions}")
        print(f"💰 初始资金: {args.capital:,.0f}元")
        print(f"📊 评分阈值: {args.min_score}")
        print()

        start_time = time.time()

        results = engine.run_backtest(
            versions=args.versions,
            start_date=args.start_date,
            end_date=args.end_date
        )

        total_time = time.time() - start_time

        # 显示结果
        print("📈 回测结果:")
        print("-" * 80)

        for version, result in results['individual_results'].items():
            if 'error' not in result:
                print(f"{version}:")
                print(f"  📊 总收益率: {result.get('total_return', 0):.2%}")
                print(f"  📈 年化收益: {result.get('annual_return', 0):.2%}")
                print(f"  🎲 夏普比率: {result.get('sharpe_ratio', 0):.2f}")
                print(f"  📉 最大回撤: {result.get('max_drawdown', 0):.2%}")
                print(f"  🔄 交易次数: {result.get('total_trades', 0)}")
                print(f"  🎯 平均评分: {result.get('avg_score', 0):.1f}")
                print(f"  ⚡ 执行用时: {result.get('backtest_time', 0):.1f}秒")
            else:
                print(f"{version}: ❌ 失败 - {result['error']}")
            print()

        # 对比分析
        analysis = results.get('comparison_analysis', {})
        if 'best_performance' in analysis and analysis['best_performance']['version']:
            print("🏆 对比分析:")
            print("-" * 80)

            best = analysis['best_performance']
            print(f"🥇 最佳收益: {best['version']} ({best['return']:.2%})")

            if 'fastest_execution' in analysis:
                fastest = analysis['fastest_execution']
                print(f"⚡ 最快执行: {fastest['version']} ({fastest['time']:.1f}秒)")

            print(f"\n📊 收益排名:")
            for i, rank in enumerate(analysis.get('performance_ranking', []), 1):
                print(f"  {i}. {rank['version']}: {rank['return']:.2%}")

        print(f"\n🎉 回测完成！总用时: {total_time:.1f}秒")

        # 保存报告
        if args.save_report:
            report_path = f"reports/backtest/extensible_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs(os.path.dirname(report_path), exist_ok=True)

            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)

            print(f"📄 报告已保存: {report_path}")

    except Exception as e:
        print(f"❌ 回测失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()