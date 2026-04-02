#!/usr/bin/env python3
"""
明日股票选股分析器
基于已下载的7055只证券数据，使用量化选股策略推荐明天适合买入的股票
"""

import sys
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import importlib.util
from data_adapter.stock_data_loader import StockDataLoader

# 导入优化后的评分系统 (v2, 已deprecated但保留兼容)
try:
    from scoring.scoring_engine import ScoringEngine, get_daily_recommendations
    from scoring.config import ScoringConfig, DEFAULT_CONFIG
except ImportError:
    ScoringEngine = None
    get_daily_recommendations = None
    ScoringConfig = None
    DEFAULT_CONFIG = None

# 确保logs目录存在
import os
os.makedirs("logs", exist_ok=True)

# 设置日志 - 只输出到文件，不输出到stdout避免污染报告
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/tomorrow_selection_results.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("tomorrow_selector")

# === 版本管理 ===
DEPRECATED_VERSIONS = {'v2', 'v3', 'v3.1', 'v3.2', 'v3.3', 'v3.4', 'v3.41',
                       'v3.5', 'v3.51', 'v3.52', 'v3.53', 'v3.6', 'v3.7',
                       'v3.8', 'v3.81', 'v3.94', 'v4'}
ACTIVE_VERSIONS = {'v3.9', 'v3.95', 'v3.96', 'v4.0', 'v4.2', 'v4.3', 'v4.4', 'v4.4.2', 'v4.5', 'v4.6', 'v4.7.1', 'v4.7.2', 'v4.7.3', 'v4.7.4', 'v4.7.5', 'v4.7.6', 'v4.7.7', 'v4.7.8', 'v4.7.9', 'v4.8.0', 'v4.8.1', 'v4.8.2', 'v4.8.4', 'v4.8.5', 'v4.8.6', 'v4.8.7', 'v4.8.8', 'v4.9.0', 'v4.9.0.1', 'v4.9.0.2', 'v4.9.1', 'v5.0'}

class TomorrowStockSelector:
    """明日股票选择器"""

    def __init__(self, scoring_version: str = "v3.9", stocks_only: bool = False, skip_strategies: bool = False, **kwargs):
        self.use_database = True  # 强制使用数据库模式
        self.selectors = {}
        self.data_cache = {}
        self.securities_info = {}  # 证券基本信息缓存
        self.scoring_version = scoring_version
        self.stocks_only = stocks_only  # 是否只考虑股票，不包括ETF基金
        self.skip_strategies = skip_strategies  # 跳过策略筛选，全市场ML评分

        # V2 optimizer
        self.optimizer_version = kwargs.get('optimizer_version', 'v1')
        if self.optimizer_version == 'v2':
            from portfolio_optimizer import PortfolioOptimizer
            self.portfolio_optimizer = PortfolioOptimizer(params_path=kwargs.get('optimizer_params_path'))

        # Deprecation warning for old versions
        if scoring_version in DEPRECATED_VERSIONS:
            import warnings
            warnings.warn(
                f"评分版本 {scoring_version} 已弃用，建议使用 v3.9 或 v3.95。"
                f"旧版本将在未来版本中移除。",
                DeprecationWarning, stacklevel=2
            )
            print(f"\n{'='*60}")
            print(f"  WARNING: 评分版本 {scoring_version} 已弃用!")
            print(f"  推荐使用: v3.9 (生产A级) 或 v3.95 (多目标预测)")
            print(f"{'='*60}\n")
        self.v381_batch_cache = {}  # V3.81批处理结果缓存
        self.v39_batch_cache = {}   # V3.9批处理结果缓存
        self.v394_batch_cache = {}  # V3.94批处理结果缓存（用于百分位排名）
        self.v395_batch_cache = {}  # V3.95批处理结果缓存（多目标预测）
        self.v40_batch_cache = {}   # V4.0批处理结果缓存（cross-sectional alpha）
        self.v43_batch_cache = {}   # V4.3批处理结果缓存（扩展特征+强正则）
        self.v44_batch_cache = {}   # V4.4批处理结果缓存（V4.3+6增强模块）

        # 初始化数据加载器
        self.data_loader = StockDataLoader()

        # 根据版本初始化评分引擎
        if scoring_version == "v5.0":
            # V5.0 Unified Feature Fusion (v39+v40+neural)
            from ml_models.v39.v500_production_scorer import V500ProductionScorer
            self.scoring_engine_v500 = V500ProductionScorer(model_type='small_data')
            self.v500_batch_cache = {}
            from ml_models.v39.strategy_based_return_predictor import StrategyBasedReturnPredictor
            self.strategy_return_predictor = StrategyBasedReturnPredictor()
            logger.info("🔬 已初始化V5.0 Unified Feature Fusion评分系统 (v39+v40+neural)")
        elif scoring_version == "v4.8.8":
            from ml_models.v39.v488_production_scorer import V488ProductionScorer
            self.scoring_engine_v44 = V488ProductionScorer(model_type='small_data')
            self.v44_batch_cache = {}
            logger.info("🔬 已初始化V4.9.0评分系统 (69特征+基准超额+熊市加权+单调性集成)")
        elif scoring_version == "v4.9.0":
            from ml_models.v39.v490_production_scorer import V490ProductionScorer
            self.scoring_engine_v44 = V490ProductionScorer(model_type='small_data')
            self.v44_batch_cache = {}
            logger.info("🔬 已初始化V4.9.0评分系统 (Q95+头部加权+truncation=10)")
        elif scoring_version == "v4.9.0.1":
            from ml_models.v39.v4901_production_scorer import V4901ProductionScorer
            self.scoring_engine_v44 = V4901ProductionScorer(model_type='small_data')
            self.v44_batch_cache = {}
            logger.info("🔬 已初始化V4.9.0.1评分系统 (去头尾加权+composite排序)")
        elif scoring_version == "v4.9.0.2":
            from ml_models.v39.v4902_production_scorer import V4902ProductionScorer
            self.scoring_engine_v44 = V4902ProductionScorer(model_type='small_data')
            self.v44_batch_cache = {}
            logger.info("🔬 已初始化V4.9.0.2评分系统 (风控增强+CVaR止损+换手优化)")
        elif scoring_version == "v4.8.7":
            from ml_models.v39.v487_production_scorer import V487ProductionScorer
            self.scoring_engine_v44 = V487ProductionScorer(model_type='small_data')
            self.v44_batch_cache = {}
            logger.info("🔬 已初始化V4.8.7评分系统 (69特征+YetiRank+RRF)")
        elif scoring_version == "v4.8.6":
            from ml_models.v39.v486_production_scorer import V486ProductionScorer
            self.scoring_engine_v44 = V486ProductionScorer(model_type='small_data')
            self.v44_batch_cache = {}
            from ml_models.v39.strategy_based_return_predictor import StrategyBasedReturnPredictor
            self.strategy_return_predictor = StrategyBasedReturnPredictor()
            logger.info("V4.8.6 (64 features + RRF ensemble, 头部区分度优化)")
        elif scoring_version == "v4.9.1":
            from ml_models.v39.v491_production_scorer import V491ProductionScorer
            self.scoring_engine_v44 = V491ProductionScorer(model_type='small_data')
            self.v44_batch_cache = {}
            from ml_models.v39.strategy_based_return_predictor import StrategyBasedReturnPredictor
            self.strategy_return_predictor = StrategyBasedReturnPredictor()
            logger.info("🛡️ V4.9.1 (超额标签+市场门控+排名平滑, 基于V4.8.5)")
        elif scoring_version == "v4.8.5":
            from ml_models.v39.v485_production_scorer import V485ProductionScorer
            self.scoring_engine_v44 = V485ProductionScorer(model_type='small_data')
            self.v44_batch_cache = {}
            from ml_models.v39.strategy_based_return_predictor import StrategyBasedReturnPredictor
            self.strategy_return_predictor = StrategyBasedReturnPredictor()
            logger.info("V4.8.5 (61 features, trained on A股+ETF)")
        elif scoring_version == "v4.8.4":
            from ml_models.v39.v484_production_scorer import V484ProductionScorer
            self.scoring_engine_v44 = V484ProductionScorer(model_type='small_data')
            self.v44_batch_cache = {}
            from ml_models.v39.strategy_based_return_predictor import StrategyBasedReturnPredictor
            self.strategy_return_predictor = StrategyBasedReturnPredictor()
            logger.info("V4.8.4 (61 features = V4.8.1 + brain_roll_spread)")
        elif scoring_version == "v4.8.2":
            from ml_models.v39.v482_production_scorer import V482ProductionScorer
            self.scoring_engine_v44 = V482ProductionScorer(model_type='small_data')
            self.v44_batch_cache = {}
            from ml_models.v39.strategy_based_return_predictor import StrategyBasedReturnPredictor
            self.strategy_return_predictor = StrategyBasedReturnPredictor()
            logger.info("V4.8.2 loaded")
        elif scoring_version == "v4.8.1":
            from ml_models.v39.v481_production_scorer import V481ProductionScorer
            self.scoring_engine_v44 = V481ProductionScorer(model_type='small_data')
            self.v44_batch_cache = {}
            from ml_models.v39.strategy_based_return_predictor import StrategyBasedReturnPredictor
            self.strategy_return_predictor = StrategyBasedReturnPredictor()
            logger.info("V4.8.1 (60 features = V4.7.5 - 5 pruned + 15 new factors + V4.7.6 post-processing)")
        elif scoring_version == "v4.8.0":
            from ml_models.v39.v480_production_scorer import V480ProductionScorer
            self.scoring_engine_v44 = V480ProductionScorer(model_type='small_data')
            self.v44_batch_cache = {}
            from ml_models.v39.strategy_based_return_predictor import StrategyBasedReturnPredictor
            self.strategy_return_predictor = StrategyBasedReturnPredictor()
            logger.info("V4.8.0 (270d衰减 + V4.7.6 scorer后处理, 目标ic_decay_ratio)")
        elif scoring_version == "v4.7.9":
            from ml_models.v39.v479_production_scorer import V479ProductionScorer
            self.scoring_engine_v44 = V479ProductionScorer(model_type='small_data')
            self.v44_batch_cache = {}
            from ml_models.v39.strategy_based_return_predictor import StrategyBasedReturnPredictor
            self.strategy_return_predictor = StrategyBasedReturnPredictor()
            logger.info("V4.7.9 (Huber+DART+240d衰减+Top5%头部加权)")
        elif scoring_version == "v4.7.8":
            from ml_models.v39.v478_production_scorer import V478ProductionScorer
            self.scoring_engine_v44 = V478ProductionScorer(model_type='small_data')
            self.v44_batch_cache = {}
            from ml_models.v39.strategy_based_return_predictor import StrategyBasedReturnPredictor
            self.strategy_return_predictor = StrategyBasedReturnPredictor()
            logger.info("V4.7.8 (Huber+DART+365d衰减 = V475 Top3 + V477 IC)")
        elif scoring_version == "v4.7.7":
            from ml_models.v39.v477_production_scorer import V477ProductionScorer
            self.scoring_engine_v44 = V477ProductionScorer(model_type='small_data')
            self.v44_batch_cache = {}
            from ml_models.v39.strategy_based_return_predictor import StrategyBasedReturnPredictor
            self.strategy_return_predictor = StrategyBasedReturnPredictor()
            logger.info("V4.7.7 (Huber+DART+180d衰减 + V4.7.6 scorer后处理)")
        elif scoring_version == "v4.7.6":
            from ml_models.v39.v476_production_scorer import V476ProductionScorer
            self.scoring_engine_v44 = V476ProductionScorer(model_type='small_data')
            self.v44_batch_cache = {}
            from ml_models.v39.strategy_based_return_predictor import StrategyBasedReturnPredictor
            self.strategy_return_predictor = StrategyBasedReturnPredictor()
            logger.info("V4.7.6 (V4.7.5 + Top-K聚焦 + 置信度折扣 + 波动率调整)")
        elif scoring_version == "v4.7.5":
            from ml_models.v39.v475_production_scorer import V475ProductionScorer
            self.scoring_engine_v44 = V475ProductionScorer(model_type='small_data')
            self.v44_batch_cache = {}
            from ml_models.v39.strategy_based_return_predictor import StrategyBasedReturnPredictor
            self.strategy_return_predictor = StrategyBasedReturnPredictor()
            logger.info("V4.7.5 (V4.7.3 + Top-Quantile Asymmetric Weighting)")
        elif scoring_version == "v4.7.4":
            from ml_models.v39.v474_production_scorer import V474ProductionScorer
            self.scoring_engine_v44 = V474ProductionScorer(model_type='small_data')
            self.v44_batch_cache = {}
            from ml_models.v39.strategy_based_return_predictor import StrategyBasedReturnPredictor
            self.strategy_return_predictor = StrategyBasedReturnPredictor()
            logger.info("V4.7.4 (连续评分+选择性V4.8+ListNet+严格ICIR约束)")
        elif scoring_version == "v4.7.3":
            # V4.7.3: 简化管线+特征精简+放宽正则化 (无Meta-Learner/Combined Isotonic)
            from ml_models.v39.v473_production_scorer import V473ProductionScorer
            self.scoring_engine_v44 = V473ProductionScorer(model_type='small_data')
            self.v44_batch_cache = {}
            from ml_models.v39.strategy_based_return_predictor import StrategyBasedReturnPredictor
            self.strategy_return_predictor = StrategyBasedReturnPredictor()
            logger.info("🚀 已初始化V4.7.3评分系统 (简化管线+精简特征+ICIR权重, 无压缩)")
        elif scoring_version == "v4.7.2":
            # V4.7.2: V4.7.1底座 + V4.6管线 (ICIR权重+Meta-Learner+Combined Isotonic)
            from ml_models.v39.v472_production_scorer import V472ProductionScorer
            self.scoring_engine_v44 = V472ProductionScorer(model_type='small_data')
            self.v44_batch_cache = {}
            from ml_models.v39.strategy_based_return_predictor import StrategyBasedReturnPredictor
            self.strategy_return_predictor = StrategyBasedReturnPredictor()
            logger.info("🚀 已初始化V4.7.2评分系统 (V4.7.1底座+V4.6管线: ICIR+MetaLearner+CombinedIsotonic)")
        elif scoring_version == "v4.7.1":
            # V4.7.1: V4.4底座 + Bug修复 + 17新特征 + LambdaRank + 时间衰减
            from ml_models.v39.v471_production_scorer import V471ProductionScorer
            self.scoring_engine_v44 = V471ProductionScorer(model_type='small_data')
            self.v44_batch_cache = {}
            from ml_models.v39.strategy_based_return_predictor import StrategyBasedReturnPredictor
            self.strategy_return_predictor = StrategyBasedReturnPredictor()
            logger.info("🚀 已初始化V4.7.1评分系统 (V4.4底座+Bug修复+17新特征+LambdaRank+时间衰减)")
        elif scoring_version == "v4.6":
            # V4.6: V4.4底座 + ICIR权重 + Combined Isotonic + Meta-Learner + 增强流动性 + 小盘加成
            from ml_models.v39.v46_production_scorer import V46ProductionScorer
            self.scoring_engine_v44 = V46ProductionScorer(model_type='small_data')
            self.v44_batch_cache = {}
            from ml_models.v39.strategy_based_return_predictor import StrategyBasedReturnPredictor
            self.strategy_return_predictor = StrategyBasedReturnPredictor()
            logger.info("🚀 已初始化V4.6评分系统 (V4.4+ICIR权重+CombinedIsotonic+MetaLearner+增强流动性+小盘加成)")
        elif scoring_version == "v4.5":
            # V4.5: V4.4.1 scorer + CPPI exposure overlay (cppi_floor=0.10, m=10)
            from ml_models.v39.v44_production_scorer import V44ProductionScorer
            self.scoring_engine_v44 = V44ProductionScorer(model_type='small_data')
            self.v44_batch_cache = {}
            from ml_models.v39.strategy_based_return_predictor import StrategyBasedReturnPredictor
            self.strategy_return_predictor = StrategyBasedReturnPredictor()
            self.cppi_floor = 0.10
            self.cppi_multiplier = 10
            self.cppi_decay = 0.995  # peak decay per day, half-life ~139d
            logger.info("🚀 已初始化V4.5评分系统 (V4.4.1+CPPI Trailing Floor, floor=10%, m=10)")
        elif scoring_version == "v4.4.2":
            # V4.4.2: V4.4.1 + 三层组合风控 (Module G/H/I)
            from ml_models.v39.v44_production_scorer import V442ProductionScorer
            self.scoring_engine_v44 = V442ProductionScorer(model_type='small_data')
            self.v44_batch_cache = {}
            from ml_models.v39.strategy_based_return_predictor import StrategyBasedReturnPredictor
            self.strategy_return_predictor = StrategyBasedReturnPredictor()
            logger.info("🛡️ 已初始化V4.4.2评分系统 (V4.4.1+市况压缩+置信度门槛+行业集中度)")
        elif scoring_version == "v4.4":
            # V4.4: V4.3信号底座 + 6增强模块
            from ml_models.v39.v44_production_scorer import V44ProductionScorer
            self.scoring_engine_v44 = V44ProductionScorer(model_type='small_data')
            self.v44_batch_cache = {}
            from ml_models.v39.strategy_based_return_predictor import StrategyBasedReturnPredictor
            self.strategy_return_predictor = StrategyBasedReturnPredictor()
            logger.info("🚀 已初始化V4.4评分系统 (V4.3信号+单调性校准+流动性+熊市专家+可执行性过滤)")
        elif scoring_version == "v4.3":
            # 初始化v4.3 扩展特征+强正则+等权+4目标 评分系统
            from ml_models.v39.v43_production_scorer import V43ProductionScorer
            self.scoring_engine_v43 = V43ProductionScorer(model_type='small_data')
            self.v43_batch_cache = {}
            from ml_models.v39.strategy_based_return_predictor import StrategyBasedReturnPredictor
            self.strategy_return_predictor = StrategyBasedReturnPredictor()
            logger.info("🚀 已初始化V4.3评分系统 (59特征, 强正则, Walk-Forward, 4目标, 等权集成)")
        elif scoring_version == "v4.2":
            # 初始化v4.2 Hybrid Alpha评分系统 (同一个scorer class, 自动检测v42模型)
            from ml_models.v40.v400_production_scorer import V400ProductionScorer
            self.scoring_engine_v40 = V400ProductionScorer()
            logger.info("🔬 已初始化V4.2 Hybrid Alpha评分系统（行业超额+RobustZScore+V39市场+5模型）")
        elif scoring_version == "v4.0":
            # 初始化v4.0 Cross-Sectional Alpha评分系统
            from ml_models.v40.v400_production_scorer import V400ProductionScorer
            self.scoring_engine_v40 = V400ProductionScorer()
            logger.info("🔬 已初始化V4.0 Cross-Sectional Alpha评分系统（超额收益预测，~55个cross-sectional特征）")
        elif scoring_version == "v3.96":
            # 初始化v3.96 Robust Z-Score + Industry-Excess 评分系统
            from ml_models.v39.v396_production_scorer import V396ProductionScorer
            self.scoring_engine_v396 = V396ProductionScorer(model_type='small_data')
            self.v396_batch_cache = {}
            from ml_models.v39.strategy_based_return_predictor import StrategyBasedReturnPredictor
            self.strategy_return_predictor = StrategyBasedReturnPredictor()
            logger.info("🚀 已初始化V3.96 Robust Z-Score评分系统 (49特征, ICIR全周期>0.2)")
        elif scoring_version == "v3.95":
            # 初始化v3.9.5生产版评分系统 - 🚀 MULTI-TARGET PREDICTION MODEL
            from ml_models.v39.v395_production_scorer import V395ProductionScorer
            self.scoring_engine_v395 = V395ProductionScorer(model_type='small_data')
            # 🎯 初始化策略驱动预测器（基于12,655历史样本统计）
            from ml_models.v39.strategy_based_return_predictor import StrategyBasedReturnPredictor
            self.strategy_return_predictor = StrategyBasedReturnPredictor()
            logger.info("🚀 已初始化V3.9.5多目标预测评分系统 + 策略驱动收益预测器")
        elif scoring_version == "v3.94":
            # 初始化v3.9.4生产版评分系统 - 🏆 PRODUCTION A+ GRADE MODEL (带活跃市值特征)
            from ml_models.v39.v394_production_scorer import V394ProductionScorer
            self.scoring_engine_v394 = V394ProductionScorer()
            logger.info("🏆 已初始化V3.9.4生产版评分系统（48特征=42基础+6活跃市值，IC+166%，Top20胜率56.43%）")
        elif scoring_version == "v3.9":
            # 初始化v3.9.0生产版评分系统 - 🏆 PRODUCTION A-GRADE MODEL
            from ml_models.v39.v390_production_scorer import V390ProductionScorer
            self.scoring_engine_v39 = V390ProductionScorer()
            logger.info("🏆 已初始化V3.9.0生产版评分系统（81.2/100 A级，67.30%方向准确率，95%Top20胜率，42基础特征）")
        elif scoring_version == "v3.81":
            raise ValueError("v3.81已弃用并删除，请使用v3.9或v3.95")
        elif scoring_version == "v3.8":
            # 初始化v3.80高级机器学习评分引擎 - 🚀 ADVANCED ML SYSTEM
            from ml_models.v38 import V380AdvancedIncrementalMLSystem
            self.scoring_engine_v38 = V380AdvancedIncrementalMLSystem()
            logger.info("🚀 已初始化V3.80高级机器学习系统（三层Ensemble+增量学习+自适应评分）")
        elif scoring_version == "v3.7":
            raise ValueError("v3.7已弃用并删除，请使用v3.9或v3.95")
        elif scoring_version == "v3.6":
            # 初始化v3.6机器学习评分引擎 - 🆕 MACHINE LEARNING
            from v360_ml_scoring_system import V360MLScoringSystem
            self.scoring_engine_v36 = V360MLScoringSystem()
            # 强制加载模型
            model_loaded = self.scoring_engine_v36.load_models('ml_models/trained_models/v360')
            if not model_loaded:
                logger.warning("⚠️  V3.6模型未找到，将使用实时训练模式")
            else:
                logger.info("✅ V3.6模型加载成功")
            logger.info("🤖 已初始化v3.6机器学习评分系统（LightGBM+XGBoost双模型ensemble，非线性建模）")
        elif scoring_version == "v4":
            # 初始化v4评分引擎
            sys.path.append(os.path.join(os.path.dirname(__file__), 'scoring_improvements'))
            from quantitative_scorer_v4 import QuantitativeScorerV4
            self.scoring_engine_v4 = QuantitativeScorerV4()
            logger.info("🚀 已初始化挤压动量增强评分系统 v4.0")
        elif scoring_version == "v3.53":
            # 初始化v3.53 多时间周期IC优化评分引擎 - 🆕 MULTI-PERIOD
            sys.path.append(os.path.join(os.path.dirname(__file__), 'scoring', 'v3.5'))
            from quantitative_scorer_v3_53 import QuantitativeScorerV353MultiPeriod
            self.scoring_engine_v353_multiperiod = QuantitativeScorerV353MultiPeriod()
            logger.info("🚀 已初始化v3.53 多时间周期IC优化评分系统（分层权重架构，1日IC=6.5%，3日IC=5.7%）")
        elif scoring_version == "v3.52":
            # 初始化v3.5 全面优化评分引擎 - 🆕 COMPREHENSIVE
            sys.path.append(os.path.join(os.path.dirname(__file__), 'scoring', 'v3.5'))
            from quantitative_scorer_v3_52 import QuantitativeScorerV35Comprehensive
            self.scoring_engine_v35_comprehensive = QuantitativeScorerV35Comprehensive()
            logger.info("🚀 已初始化v3.5 全面优化评分系统（38个参数全面优化，基于21,744条样本数据）")
        elif scoring_version == "v3.51":
            # 初始化v3.5 Qlib优化评分引擎 - 🆕 OPTIMIZED
            sys.path.append(os.path.join(os.path.dirname(__file__), 'scoring', 'v3.5'))
            from quantitative_scorer_v3_51 import QuantitativeScorerV35Optimized
            self.scoring_engine_v35_optimized = QuantitativeScorerV35Optimized()
            logger.info("🚀 已初始化v3.5 Qlib优化评分系统（Phase 2权重+知行参数，+3.12% IC，知行指标权重降至7.4%）")
        elif scoring_version == "v3.5":
            # 初始化v3.5知行指标集成评分引擎 - 🆕
            sys.path.append(os.path.join(os.path.dirname(__file__), 'scoring', 'v3.5'))
            from quantitative_scorer_v3_5 import QuantitativeScorerV35
            self.scoring_engine_v35 = QuantitativeScorerV35()
            logger.info("🚀 已初始化v3.5知行指标集成评分系统（知行趋势线+多空线，权重20%）")
        elif scoring_version == "v3.41":
            # 初始化v3.41反向工程重构评分引擎 - 🆕
            sys.path.append(os.path.join(os.path.dirname(__file__), 'scoring', 'v3.4'))
            from quantitative_scorer_v3_41 import QuantitativeScorerV341
            self.scoring_engine_v341 = QuantitativeScorerV341()
            logger.info("🔄 已初始化v3.41反向工程重构评分系统（基于负相关发现的革命性改进）")
        elif scoring_version == "v3.4":
            # 初始化v3.4基于v3.0优化的评分引擎 - 🆕  
            sys.path.append(os.path.join(os.path.dirname(__file__), 'scoring', 'v3.4'))
            from quantitative_scorer_v3_4 import QuantitativeScorerV34
            self.scoring_engine_v34 = QuantitativeScorerV34()
            logger.info("🚀 已初始化v3.4增强评分系统（基于v3.0成功经验优化，新增ROE和营收增长）")
        elif scoring_version == "v3.3":
            # 初始化v3.3相关性优化评分引擎 - 🆕
            from scoring.v3.quantitative_scorer_v3_3 import QuantitativeScorerV33
            self.scoring_engine_v33 = QuantitativeScorerV33()
            logger.info("🚀 已初始化v3.3相关性优化评分系统（基于相关性分析深度优化）")
        elif scoring_version == "v3.2":
            # 初始化v3.2挤压动量评分引擎 - 🆕
            from scoring.v3.quantitative_scorer_v3_2 import QuantitativeScorerV32
            self.scoring_engine_v32 = QuantitativeScorerV32()
            logger.info("🚀 已初始化v3.2挤压动量增强评分系统（集成v4.0挤压动量因子）")
        elif scoring_version == "v3.1":
            # 初始化v3.1优化评分引擎
            from scoring.v3.quantitative_scorer_v3_1 import QuantitativeScorerV31
            self.scoring_engine_v31 = QuantitativeScorerV31()
            logger.info("🚀 已初始化v3.1优化评分系统（基于214万条数据优化权重）")
        elif scoring_version == "v3":
            # 初始化v3评分引擎
            from scoring.v3.quantitative_scorer_v3 import QuantitativeScorerV3
            self.scoring_engine_v3 = QuantitativeScorerV3()
            logger.info("🚀 已初始化智能动态权重评分系统 v3.0")
        else:
            # 初始化优化后的评分引擎v2
            if ScoringEngine is None:
                raise ValueError(f"评分版本 {scoring_version} 需要 scoring 模块，但该模块已移除。请使用 v3.9 或 v3.95。")
            self.scoring_engine = ScoringEngine()
            logger.info("🚀 已初始化基于实际数据优化的评分系统 v2.0")
        
        # 导入选股器
        self._import_selectors()
        # 加载证券基本信息
        self._load_securities_info()
        
    @classmethod
    def create_for_batch(cls, scoring_version: str, stocks_only: bool = True,
                         scorer=None, data_loader=None, securities_info=None):
        """
        批量模式构造方法 — 接受预初始化的资源，跳过模型加载和数据库查询

        Args:
            scoring_version: 'v3.9' 或 'v3.95'
            stocks_only: 是否只考虑A股
            scorer: 预初始化的评分器实例 (V390ProductionScorer 或 V395ProductionScorer)
            data_loader: 预初始化的 StockDataLoader 实例
            securities_info: 预加载的证券信息字典

        Returns:
            TomorrowStockSelector 实例
        """
        instance = cls.__new__(cls)
        instance.use_database = True
        instance.selectors = {}
        instance.data_cache = {}
        instance.scoring_version = scoring_version
        instance.stocks_only = stocks_only
        instance.v381_batch_cache = {}
        instance.v39_batch_cache = {}
        instance.v394_batch_cache = {}
        instance.v395_batch_cache = {}
        instance.v396_batch_cache = {}
        instance.v40_batch_cache = {}
        instance.v43_batch_cache = {}
        instance.v44_batch_cache = {}
        instance.v500_batch_cache = {}

        # 使用传入的资源而不是重新创建
        instance.data_loader = data_loader or StockDataLoader()
        instance.securities_info = securities_info or {}

        # 注入评分器
        if scoring_version == "v3.9" and scorer is not None:
            instance.scoring_engine_v39 = scorer
        elif scoring_version == "v3.95" and scorer is not None:
            instance.scoring_engine_v395 = scorer
            # 初始化策略驱动预测器
            from ml_models.v39.strategy_based_return_predictor import StrategyBasedReturnPredictor
            instance.strategy_return_predictor = StrategyBasedReturnPredictor()

        # 仍然需要导入选股器类
        instance._import_selectors()

        return instance

    def _import_selectors(self):
        """动态导入选股器类"""
        try:
            # 导入Selector模块
            selector_path = Path("stock_selctor/Selector.py")
            if not selector_path.exists():
                logger.error(f"找不到选股器模块: {selector_path}")
                return
                
            spec = importlib.util.spec_from_file_location("Selector", selector_path)
            selector_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(selector_module)
            
            # 获取可用的选股器类
            self.selector_classes = {
                'BBIKDJSelector': selector_module.BBIKDJSelector,
                'BBIShortLongSelector': selector_module.BBIShortLongSelector,
                'BreakoutVolumeKDJSelector': selector_module.BreakoutVolumeKDJSelector,
                'PeakKDJSelector': selector_module.PeakKDJSelector,
                'SuperB1Selector': selector_module.SuperB1Selector,
                'ZhiXingSelector': selector_module.ZhiXingSelector,
                'MA60CrossVolumeWaveSelector': selector_module.MA60CrossVolumeWaveSelector,
                'BigBullishVolumeSelector': selector_module.BigBullishVolumeSelector
            }
            # 保存预计算函数引用
            self._precompute_indicators = getattr(selector_module, 'precompute_indicators', None)
            logger.info(f"成功导入 {len(self.selector_classes)} 个选股器类")
            
        except Exception as e:
            logger.error(f"导入选股器失败: {e}")
    
    def _load_securities_info(self):
        """加载证券基本信息（股票名称、板块、行业等）"""
        try:
            # 从数据库加载证券信息
            self.securities_info = self.data_loader.load_securities_info()
            logger.info(f"从数据库加载了 {len(self.securities_info)} 只证券的基本信息")
        except Exception as e:
            logger.error(f"加载证券信息失败: {e}")
    
    def update_securities_list_with_details(self):
        """更新securities_list.csv，添加详细的基本面信息"""
        try:
            import tushare as ts
            
            # 获取配置中的token
            try:
                with open('config.json', 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    token = config.get('tushare', {}).get('token')
                    if not token or token == "YOUR_TUSHARE_TOKEN_HERE":
                        logger.warning("未配置Tushare Token，无法更新详细基本面信息")
                        return False
                        
                    ts.set_token(token)
                    pro = ts.pro_api()
            except Exception as e:
                logger.error(f"配置Tushare失败: {e}")
                return False
            
            logger.info("开始更新证券列表详细信息...")
            
            # 获取A股基本信息
            logger.info("获取A股基本信息...")
            a_stocks = pro.stock_basic(
                exchange='',
                list_status='L',
                fields='ts_code,symbol,name,area,industry,market,list_date'
            )
            
            # 添加类型标识
            a_stocks['type'] = 'A股'
            a_stocks['code'] = a_stocks['symbol']
            
            # 获取基金ETF信息
            logger.info("获取基金ETF信息...")
            try:
                funds = pro.fund_basic(
                    market='E',
                    fields='ts_code,name,fund_type,list_date'
                )
                
                # 为基金添加缺失字段
                funds['type'] = 'ETF_基金' 
                funds['market'] = funds['ts_code'].apply(lambda x: 'SH' if x.endswith('.SH') else 'SZ')
                funds['area'] = '未知'
                funds['industry'] = funds.get('fund_type', 'ETF基金')
                funds['code'] = funds['ts_code'].str.split('.').str[0]
                
                # 统一列名
                funds = funds.rename(columns={'fund_type': 'industry'})
                
                # 选择需要的列
                funds = funds[['ts_code', 'code', 'name', 'area', 'industry', 'market', 'list_date', 'type']]
                
                # 合并数据
                all_securities = pd.concat([a_stocks, funds], ignore_index=True)
                
            except Exception as e:
                logger.warning(f"获取基金信息失败，仅使用A股数据: {e}")
                all_securities = a_stocks
            
            # 重新排列列顺序
            all_securities = all_securities[['ts_code', 'code', 'name', 'type', 'market', 'industry', 'area', 'list_date']]
            
            # 保存到文件
            securities_file = Path("securities_list.csv")
            all_securities.to_csv(securities_file, index=False, encoding='utf-8')
            
            logger.info(f"成功更新证券列表，共{len(all_securities)}只证券")
            logger.info(f"A股: {len(a_stocks)}只")
            if 'funds' in locals():
                logger.info(f"ETF/基金: {len(funds)}只")
            
            return True
            
        except Exception as e:
            logger.error(f"更新证券列表失败: {e}")
            return False
            
    def load_data(self, limit: int = None, target_date: str = None) -> Dict[str, pd.DataFrame]:
        """加载股票数据"""
        # 从数据库加载数据
        logger.info("从数据库加载股票数据...")
        
        # 根据stocks_only参数决定加载的证券类型
        if self.stocks_only:
            security_types = ['A股']  # 只加载A股
            logger.info("⚙️  仅加载A股数据（排除ETF/基金）")
        else:
            security_types = ['A股', 'ETF_基金']  # 默认加载A股和ETF基金
            logger.info("⚙️  加载A股和ETF/基金数据")
            
        data = self.data_loader.load_all_stock_data(days=200, security_types=security_types, target_date=target_date)
        
        if limit:
            # 限制加载的股票数量
            limited_data = {}
            for i, (code, df) in enumerate(data.items()):
                if i >= limit:
                    break
                limited_data[code] = df
            return limited_data
        
        return data

    def analyze_v38_results(self, evaluation_result: Dict, data: Dict[str, pd.DataFrame], target_date: pd.Timestamp = None) -> Dict[str, Any]:
        """分析V3.8自适应评分结果"""
        try:
            stocks = evaluation_result.get('stocks', [])
            if not stocks:
                return {
                    "total_strategies": 1,
                    "strategy_results": {"V3.8自适应评分": {"count": 0, "stocks": []}},
                    "strategy_details": {"V3.8自适应评分": []},
                    "multi_strategy_stocks": [],
                    "single_strategy_stocks": [],
                    "all_selected_stocks": [],
                    "total_unique_stocks": 0,
                    "detailed_analysis_count": 0,
                    "detailed_stocks": []
                }

            # 按评分排序
            sorted_stocks = sorted(stocks, key=lambda x: x['final_score'], reverse=True)

            # 选择top股票进行详细分析
            top_count = min(50, len(sorted_stocks))
            top_stocks = sorted_stocks[:top_count]

            # 格式化股票信息
            detailed_stocks = []
            for stock in top_stocks:
                # 从证券信息中获取正确的股票名称
                stock_code = stock['code']
                stock_name = self.securities_info.get(stock_code, {}).get('name', f'股票{stock_code}')

                stock_info = {
                    'code': stock_code,
                    'name': stock_name,
                    'final_score': stock['final_score'] * 100,  # 转换为0-100分制
                    'confidence_score': stock.get('confidence_score', 0.0),
                    'confidence_level': stock.get('confidence_level', 'unknown'),
                    'short_term_score': stock.get('short_term_score', 0.5) * 100,
                    'medium_term_score': stock.get('medium_term_score', 0.5) * 100,
                    'long_term_score': stock.get('long_term_score', 0.5) * 100,
                    'risk_level': stock.get('risk_level', 'medium'),
                    'overall_quality': stock.get('overall_quality', 0.5),
                    'strategy': 'V3.8自适应评分'
                }
                detailed_stocks.append(stock_info)

            analysis = {
                "total_strategies": 1,
                "strategy_results": {
                    "V3.8自适应评分": {
                        "count": len(top_stocks),
                        "stocks": [s['code'] for s in top_stocks]
                    }
                },
                "strategy_details": {"V3.8自适应评分": [s['code'] for s in top_stocks]},
                "multi_strategy_stocks": [],  # V3.8是单一策略
                "single_strategy_stocks": [s['code'] for s in top_stocks],
                "all_selected_stocks": [s['code'] for s in top_stocks],
                "total_unique_stocks": len(top_stocks),
                "detailed_analysis_count": len(detailed_stocks),
                "detailed_stocks": detailed_stocks,
                "v38_summary": evaluation_result.get('summary', {}),
                "evaluation_metadata": evaluation_result.get('metadata', {})
            }

            return analysis

        except Exception as e:
            logger.error(f"V3.8结果分析失败: {e}")
            return {
                "total_strategies": 0,
                "strategy_results": {},
                "strategy_details": {},
                "multi_strategy_stocks": [],
                "single_strategy_stocks": [],
                "all_selected_stocks": [],
                "total_unique_stocks": 0,
                "detailed_analysis_count": 0,
                "detailed_stocks": [],
                "error": str(e)
            }

    def analyze_v38_mixed_results(self, traditional_results: Dict, evaluation_result: Dict, data: Dict[str, pd.DataFrame], target_date: pd.Timestamp = None) -> Dict[str, Any]:
        """分析V3.8混合模式结果（传统策略+V3.8评分）"""
        try:
            v38_stocks = evaluation_result.get('stocks', [])
            if not v38_stocks:
                # 如果没有V3.8评分结果，回退到传统分析
                return self.analyze_results(traditional_results, data, target_date)

            # 按V3.8评分排序
            sorted_stocks = sorted(v38_stocks, key=lambda x: x['final_score'], reverse=True)

            # 创建V3.8评分的详细股票信息
            detailed_stocks = []
            for stock in sorted_stocks:
                stock_code = stock['code']
                stock_name = self.securities_info.get(stock_code, {}).get('name', f'股票{stock_code}')

                # 查找这只股票被哪些传统策略选中
                selected_by_strategies = []
                for strategy, stocks in traditional_results.items():
                    if stock_code in stocks:
                        selected_by_strategies.append(strategy)

                stock_info = {
                    'code': stock_code,
                    'name': stock_name,
                    'final_score': stock['final_score'] * 100,  # 转换为0-100分制
                    'confidence_score': stock.get('confidence_score', 0.0),
                    'confidence_level': stock.get('confidence_level', 'unknown'),
                    'short_term_score': stock.get('short_term_score', 0.5) * 100,
                    'medium_term_score': stock.get('medium_term_score', 0.5) * 100,
                    'long_term_score': stock.get('long_term_score', 0.5) * 100,
                    'risk_level': stock.get('risk_level', 'medium'),
                    'overall_quality': stock.get('overall_quality', 0.5),
                    'strategy': 'V3.8自适应评分',
                    'traditional_strategies': selected_by_strategies,  # 新增：被哪些传统策略选中
                    'strategy_count': len(selected_by_strategies)  # 新增：被多少个策略选中
                }
                detailed_stocks.append(stock_info)

            # 计算传统策略的交集信息
            traditional_analysis = self.analyze_results(traditional_results, data, target_date)

            # 构建混合分析结果
            analysis = {
                "total_strategies": len(traditional_results) + 1,  # 传统策略数量 + V3.8
                "traditional_strategies": len(traditional_results),
                "strategy_results": {},
                "strategy_details": traditional_results,  # 保存传统策略详细结果
                "multi_strategy_stocks": traditional_analysis.get("multi_strategy_stocks", {}),
                "single_strategy_stocks": traditional_analysis.get("single_strategy_stocks", []),
                "all_selected_stocks": [s['code'] for s in sorted_stocks],
                "total_unique_stocks": len(sorted_stocks),
                "detailed_analysis_count": len(detailed_stocks),
                "detailed_stocks": detailed_stocks,
                "v38_summary": evaluation_result.get('summary', {}),
                "evaluation_metadata": evaluation_result.get('metadata', {}),
                "traditional_analysis": traditional_analysis,  # 保存传统分析结果
                "v38_mixed_mode": True  # 标记为混合模式
            }

            # 统计每个策略的结果（包括V3.8）
            for strategy, stocks in traditional_results.items():
                analysis["strategy_results"][strategy] = len(stocks)
            analysis["strategy_results"]["V3.8自适应评分"] = len(sorted_stocks)

            logger.info(f"V3.8混合模式分析完成 - 传统策略: {len(traditional_results)}个, V3.8评分股票: {len(sorted_stocks)}只")

            return analysis

        except Exception as e:
            logger.error(f"V3.8混合结果分析失败: {e}")
            # 出错时回退到传统分析
            return self.analyze_results(traditional_results, data, target_date)

    def analyze_v381_mixed_results(self, traditional_results: Dict, evaluation_result: Dict, data: Dict[str, pd.DataFrame], target_date: pd.Timestamp = None) -> Dict[str, Any]:
        """分析V3.81混合模式结果（传统策略+V3.81 Level 4质量评分）"""
        try:
            v381_stocks = evaluation_result.get('stocks', [])
            if not v381_stocks:
                # 如果没有V3.81评分结果，回退到传统分析
                return self.analyze_results(traditional_results, data, target_date)

            # 🎯 按综合评分排序 (质量评分作为辅助信息，不影响主排序)
            sorted_stocks = sorted(v381_stocks, key=lambda x: x['final_score'], reverse=True)

            # 创建V3.81评分的详细股票信息
            detailed_stocks = []
            for stock in sorted_stocks:
                stock_code = stock['code']
                stock_name = self.securities_info.get(stock_code, {}).get('name', f'股票{stock_code}')

                # 查找这只股票被哪些传统策略选中
                selected_by_strategies = []
                for strategy, stocks in traditional_results.items():
                    if stock_code in stocks:
                        selected_by_strategies.append(strategy)

                stock_info = {
                    'code': stock_code,
                    'name': stock_name,
                    'final_score': stock['final_score'] * 100,  # 转换为0-100分制
                    'confidence_score': stock.get('confidence_score', 0.0),
                    'confidence_level': stock.get('confidence_level', 'unknown'),
                    'short_term_score': stock.get('short_term_score', 0.5) * 100,
                    'medium_term_score': stock.get('medium_term_score', 0.5) * 100,
                    'long_term_score': stock.get('long_term_score', 0.5) * 100,
                    'risk_level': stock.get('risk_level', 'medium'),
                    # 🎯 Level 4质量评分作为核心指标
                    'overall_quality': stock.get('quality_score', 0.5),
                    'quality_score': stock.get('quality_score', 0.5),
                    'strategy': 'V3.81 Level 4质量评分',
                    'traditional_strategies': selected_by_strategies,
                    'strategy_count': len(selected_by_strategies),
                    # 🔧 保留V3.81原始投资建议
                    'recommendation': stock.get('recommendation', '观望'),
                    # 新增V3.81特有字段
                    'level4_features': {
                        'quality_differentiation': True,
                        'meta_learning': True,
                        'end_to_end_ml': True
                    }
                }
                detailed_stocks.append(stock_info)

            # 计算传统策略的交集信息
            traditional_analysis = self.analyze_results(traditional_results, data, target_date)

            # 构建V3.81混合分析结果
            analysis = {
                "total_strategies": len(traditional_results) + 1,  # 传统策略数量 + V3.81
                "traditional_strategies": len(traditional_results),
                "strategy_results": {},
                "strategy_details": traditional_results,
                "multi_strategy_stocks": traditional_analysis.get("multi_strategy_stocks", {}),
                "single_strategy_stocks": traditional_analysis.get("single_strategy_stocks", []),
                "all_selected_stocks": [s['code'] for s in sorted_stocks],
                "total_unique_stocks": len(sorted_stocks),
                "detailed_analysis_count": len(detailed_stocks),
                "detailed_stocks": detailed_stocks,
                "v381_summary": evaluation_result.get('summary', {}),
                "evaluation_metadata": evaluation_result.get('metadata', {}),
                "traditional_analysis": traditional_analysis,
                "v381_mixed_mode": True,  # 标记为V3.81混合模式
                # 🎯 V3.81特有的质量评分统计
                "quality_score_stats": {
                    'mean': sum(s.get('quality_score', 0) for s in v381_stocks) / len(v381_stocks),
                    'std': np.std([s.get('quality_score', 0) for s in v381_stocks]),
                    'min': min(s.get('quality_score', 0) for s in v381_stocks),
                    'max': max(s.get('quality_score', 0) for s in v381_stocks),
                    'high_quality_count': sum(1 for s in v381_stocks if s.get('quality_score', 0) >= 0.7),
                    'low_quality_count': sum(1 for s in v381_stocks if s.get('quality_score', 0) < 0.3)
                }
            }

            # 统计每个策略的结果（包括V3.81）
            for strategy, stocks in traditional_results.items():
                analysis["strategy_results"][strategy] = len(stocks)
            analysis["strategy_results"]["V3.81 Level 4质量评分"] = len(sorted_stocks)

            logger.info(f"V3.81混合模式分析完成 - 传统策略: {len(traditional_results)}个, V3.81 Level 4评分股票: {len(sorted_stocks)}只")
            logger.info(f"质量评分分布: 均值={analysis['quality_score_stats']['mean']:.3f}, std={analysis['quality_score_stats']['std']:.3f}")

            return analysis

        except Exception as e:
            logger.error(f"V3.81混合结果分析失败: {e}")
            # 出错时回退到传统分析
            return self.analyze_results(traditional_results, data, target_date)

    def get_latest_trading_date(self, data: Dict[str, pd.DataFrame]) -> pd.Timestamp:
        """获取最新交易日"""
        # 从数据库获取最新交易日
        latest_date = self.data_loader.get_latest_trading_date()
        if latest_date:
            return latest_date
        
        # 如果数据库没有返回日期，从数据中获取最新交易日
        latest_dates = []
        for df in data.values():
            if not df.empty:
                latest_dates.append(df['date'].max())
                
        if not latest_dates:
            return pd.Timestamp.now()
            
        return max(latest_dates)
        
    def run_selectors(self, data: Dict[str, pd.DataFrame], target_date: pd.Timestamp) -> Dict[str, List[str]]:
        """运行所有选股策略（预截断数据 + 预计算指标 + 并行执行）"""
        import time as _time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = {}

        # ---- 预截断数据：一次性完成 date filter，避免 8 个策略重复过滤 ----
        t_pre = _time.time()
        truncated_data = {}
        for code, df in data.items():
            hist = df[df["date"] <= target_date]
            if len(hist) >= 20:
                truncated_data[code] = hist
        logger.info(f"数据预截断完成: {len(truncated_data)}只股票, 耗时 {_time.time()-t_pre:.2f}秒")

        # ---- 预计算公共技术指标（BBI/KDJ/DIF/ZX/MA60）----
        if hasattr(self, '_precompute_indicators') and self._precompute_indicators:
            t0 = _time.time()
            self._precompute_indicators(truncated_data, target_date)
            logger.info(f"指标预计算完成, 耗时 {_time.time()-t0:.2f}秒")

        # 配置八个选股策略
        strategies = {
            "少负战法": {
                "class": "BBIKDJSelector",
                "params": {
                    "j_threshold": -5,           # 从10→-5: J<-5表示严重超卖
                    "bbi_min_window": 20,
                    "max_window": 60,
                    "price_range_pct": 0.4,      # 从1→0.4: 限制40%价格波动
                    "bbi_q_threshold": 0.10,     # 从0.3→0.10: 只允许10%下跌天数
                    "j_q_threshold": 0.05        # 从0.10→0.05: J值需在5%分位以下
                }
            },
            "SuperB1战法": {
                "class": "SuperB1Selector",
                "params": {
                    "lookback_n": 15,
                    "close_vol_pct": 0.02,
                    "price_drop_pct": 0.02,
                    "j_threshold": 10,
                    "j_q_threshold": 0.10,
                    "B1_params": {
                        "j_threshold": 10,
                        "bbi_min_window": 20,
                        "max_window": 60,
                        "price_range_pct": 2.0,
                        "bbi_q_threshold": 0.3,
                        "j_q_threshold": 0.10
                    }
                }
            },
            "补票战法": {
                "class": "BBIShortLongSelector",
                "params": {
                    "n_short": 3,
                    "n_long": 21,
                    "m": 3,
                    "bbi_min_window": 2,
                    "max_window": 60,
                    "bbi_q_threshold": 0.2
                }
            },
            "TePu战法": {
                "class": "BreakoutVolumeKDJSelector",
                "params": {
                    "j_threshold": 1,
                    "j_q_threshold": 0.10,
                    "up_threshold": 3.0,
                    "volume_threshold": 0.6667,
                    "offset": 15,
                    "max_window": 60,
                    "price_range_pct": 1
                }
            },
            "填坑战法": {
                "class": "PeakKDJSelector",
                "params": {
                    "j_threshold": 10,
                    "max_window": 100,
                    "fluc_threshold": 0.03,
                    "j_q_threshold": 0.10,
                    "gap_threshold": 0.2
                }
            },
            "知行战法": {
                "class": "ZhiXingSelector",
                "params": {
                    "j_threshold": 5.0,          # J<5表示深度超卖
                    "min_change_pct": -1.0,      # 涨幅>-1%
                    "max_change_pct": 1.0,       # 涨幅<1%更精准
                    "max_amplitude_pct": 4.0,    # 振幅<4%过滤大波动
                    "close_threshold_pct": 100.0, # 收盘必须在多空线之上
                    "max_window": 120
                }
            },
            "上穿60放量战法": {
                "class": "MA60CrossVolumeWaveSelector",
                "params": {
                    "lookback_n": 20,            # 30→20: 缩短回看窗口
                    "vol_multiple": 2.2,         # 1.8→2.2: 放量要求更高
                    "j_threshold": 5,            # 15→5: J<5表示超卖
                    "j_q_threshold": 0.05,       # 0.10→0.05: 更严格分位
                    "ma60_slope_days": 5,
                    "max_window": 120
                }
            },
            "暴力K战法": {
                "class": "BigBullishVolumeSelector",
                "params": {
                    "up_pct_threshold": 0.04,
                    "upper_wick_pct_max": 0.5,
                    "vol_lookback_n": 20,
                    "vol_multiple": 1.5,
                    "require_bullish_close": True,
                    "ignore_zero_volume": True,
                    "close_lt_zxdq_mult": 1.0
                }
            }
        }

        # ---- 并行执行 8 个策略（线程安全：每个 selector 内部对 hist 做 copy）----
        def _run_one(strategy_name, config):
            selector_class = self.selector_classes[config["class"]]
            selector = selector_class(**config["params"])
            picks = selector.select(target_date, truncated_data)
            return strategy_name, picks

        t0 = _time.time()
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(_run_one, name, cfg): name
                for name, cfg in strategies.items()
            }
            for future in as_completed(futures):
                strategy_name = futures[future]
                try:
                    name, picks = future.result()
                    results[name] = picks
                    logger.info(f"{name} 选出 {len(picks)} 只股票")
                except Exception as e:
                    logger.error(f"运行 {strategy_name} 失败: {e}")
                    results[strategy_name] = []

        logger.info(f"8个策略并行执行完成, 耗时 {_time.time()-t0:.2f}秒")
        return results
        
    def get_stock_info(self, stock_code: str, data: Dict[str, pd.DataFrame], target_date: pd.Timestamp = None) -> Dict[str, Any]:
        """获取股票基本信息和技术指标"""
        if stock_code not in data:
            return {}
            
        df = data[stock_code]
        if df.empty:
            return {}
            
        try:
            # 如果指定了目标日期，使用该日期的数据；否则使用最新数据
            if target_date is not None:
                target_data = df[df['date'] <= target_date]
                if target_data.empty:
                    return {}
                latest = target_data.iloc[-1]
                df_for_indicators = target_data
            else:
                latest = df.iloc[-1]
                df_for_indicators = df
            
            # 计算技术指标
            df_with_indicators = self._calculate_indicators(df_for_indicators)
            latest_indicators = df_with_indicators.iloc[-1]
            
            # 计算价格变化
            if len(df_for_indicators) > 1:
                prev_close = df_for_indicators.iloc[-2]['close']
            else:
                prev_close = latest['close']
            price_change = latest['close'] - prev_close
            price_change_pct = (price_change / prev_close) * 100 if prev_close > 0 else 0
            
            # 计算近期波动率
            recent_prices = df_for_indicators['close'].tail(20)
            volatility = recent_prices.std() / recent_prices.mean() * 100 if len(recent_prices) > 1 else 0
            
            # 获取股票基本信息（名称、板块等）
            security_info = self.securities_info.get(stock_code, {})
            
            # 短线量化交易大师级定价策略
            close_price = latest['close']
            high_price = latest['high']
            low_price = latest['low']
            volume = latest['volume']
            
            # 判断交易制度：T+0(ETF/基金) vs T+1(股票)
            is_t0_instrument = self._is_t0_instrument(stock_code, security_info)
            
            # 计算科学定价
            buy_price, stop_loss, take_profit = self._calculate_smart_prices(
                df_for_indicators, latest, latest_indicators, volatility, is_t0_instrument
            )
            
            # 计算盈亏百分比和风险收益比
            risk_amount = buy_price - stop_loss
            reward_amount = take_profit - buy_price
            risk_pct = (risk_amount / buy_price) * 100 if buy_price > 0 else 0
            reward_pct = (reward_amount / buy_price) * 100 if buy_price > 0 else 0
            risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
            
            return {
                "stock_code": stock_code,
                "stock_name": security_info.get('name', '未知'),
                "market": security_info.get('market', '未知'),
                "ts_code": security_info.get('ts_code', ''),
                # 基本面信息
                "industry": security_info.get('industry', '未知'),
                "area": security_info.get('area', '未知'),
                "list_date": security_info.get('list_date', '未知'),
                "stock_type": security_info.get('type', '未知'),
                "analysis_date": latest['date'].strftime('%Y-%m-%d'),
                "close_price": round(close_price, 2),
                "trading_type": "T+0" if is_t0_instrument else "T+1",
                "suggested_buy_price": buy_price,
                "stop_loss_price": stop_loss,
                "take_profit_price": take_profit,
                "risk_pct": round(risk_pct, 2),
                "reward_pct": round(reward_pct, 2),
                "risk_reward_ratio": round(risk_reward_ratio, 2),
                "price_change": round(price_change, 2),
                "price_change_pct": round(price_change_pct, 2),
                "volume": int(latest['volume']) if not np.isnan(latest['volume']) else 0,
                "high": round(latest['high'], 2),
                "low": round(latest['low'], 2),
                "volatility": round(volatility, 2),
                "kdj_k": round(latest_indicators.get('K', 0), 2),
                "kdj_d": round(latest_indicators.get('D', 0), 2), 
                "kdj_j": round(latest_indicators.get('J', 0), 2),
                "bbi": round(latest_indicators.get('BBI', 0), 2),
                "dif": round(latest_indicators.get('DIF', 0), 4)
            }
            
        except Exception as e:
            logger.warning(f"获取 {stock_code} 股票信息失败: {e}")
            return {}
    
    def _is_t0_instrument(self, stock_code: str, security_info: dict) -> bool:
        """判断是否为T+0交易品种"""
        ts_code = security_info.get('ts_code', '')
        
        # ETF和基金可以T+0交易
        if (ts_code.endswith('.SH') and (stock_code.startswith('51') or stock_code.startswith('50'))) or \
           (ts_code.endswith('.SZ') and (stock_code.startswith('15') or stock_code.startswith('16'))):
            return True
        
        # 其他股票都是T+1
        return False
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """计算平均真实波幅(ATR)"""
        if len(df) < period + 1:
            return 0
        
        high = df['high']
        low = df['low'] 
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean().iloc[-1]
        
        return atr if not np.isnan(atr) else 0
    
    def _calculate_zhixing_trend(self, close_prices: np.array) -> Optional[float]:
        """
        计算知行短期趋势线: EMA(EMA(C,10),10)
        通达信公式: 知行短期趋势线:EMA(EMA(C,10),10),COLORFFFFFF,LINETHICK1;
        """
        try:
            if len(close_prices) < 20:  # 至少需要20天数据
                return None
                
            # 第一层EMA(C,10)
            alpha1 = 2.0 / (10 + 1)
            ema1 = np.zeros_like(close_prices, dtype=float)
            ema1[0] = close_prices[0]
            
            for i in range(1, len(close_prices)):
                ema1[i] = alpha1 * close_prices[i] + (1 - alpha1) * ema1[i-1]
            
            # 第二层EMA(EMA(C,10),10)
            alpha2 = 2.0 / (10 + 1)
            ema2 = np.zeros_like(ema1, dtype=float)
            ema2[0] = ema1[0]
            
            for i in range(1, len(ema1)):
                ema2[i] = alpha2 * ema1[i] + (1 - alpha2) * ema2[i-1]
            
            return float(ema2[-1])
            
        except Exception as e:
            logger.warning(f"计算知行短期趋势线失败: {e}")
            return None
    
    def _calculate_zhixing_multiavg(self, close_prices: np.array, periods: List[int] = None) -> Optional[float]:
        """
        计算知行多空线: (MA(CLOSE,M1)+MA(CLOSE,M2)+MA(CLOSE,M3)+MA(CLOSE,M4))/4
        通达信公式: 知行多空线:(MA(CLOSE,M1)+MA(CLOSE,M2)+MA(CLOSE,M3)+MA(CLOSE,M4))/4;

        默认使用周期 [14, 28, 57, 114] 对应 M1, M2, M3, M4 (与 Selector.py compute_zx_lines 一致)
        """
        try:
            if periods is None:
                periods = [14, 28, 57, 114]
                
            if len(close_prices) < max(periods):
                return None
                
            ma_values = []
            for period in periods:
                if len(close_prices) >= period:
                    ma = np.mean(close_prices[-period:])
                    ma_values.append(ma)
                else:
                    return None
            
            # 返回四个移动平均的平均值
            return float(np.mean(ma_values))
            
        except Exception as e:
            logger.warning(f"计算知行多空线失败: {e}")
            return None

    def _get_stock_data_for_scoring(self, stock_code: str, trade_date: str) -> Dict:
        """获取股票数据用于优化版评分系统"""
        try:
            # 直接从数据库获取数据，类似于 enhanced_data_manager 的方式
            with self.data_loader.db_manager.get_connection() as conn:
                # 获取股票ID
                security_query = "SELECT id FROM securities WHERE code = ?"
                security_result = conn.execute(security_query, (stock_code,)).fetchone()
                if not security_result:
                    return {}
                security_id = security_result[0]
                
                # 获取最新的技术指标数据 - 需要更多数据计算知行指标
                tech_query = """
                SELECT dq.trade_date, dq.close, dq.high, dq.low, dq.volume, dq.price_change_pct,
                       ti.rsi6, ti.kdj_k, ti.kdj_d, ti.bbi
                FROM daily_quotes dq
                LEFT JOIN technical_indicators ti ON dq.security_id = ti.security_id AND dq.trade_date = ti.trade_date
                WHERE dq.security_id = ? AND dq.trade_date <= ?
                ORDER BY dq.trade_date DESC
                LIMIT 80
                """
                tech_df = pd.read_sql_query(tech_query, conn, params=(security_id, trade_date))
                
                if tech_df.empty:
                    return {}
                
                latest_data = tech_df.iloc[0]
                
                # 获取基本面数据（PE, PB, 市值）
                basic_query = """
                SELECT pe_ttm, pb, total_mv 
                FROM daily_basic
                WHERE security_id = ? AND trade_date <= ?
                ORDER BY trade_date DESC
                LIMIT 1
                """
                basic_result = conn.execute(basic_query, (security_id, trade_date)).fetchone()
                
                # 计算周期收益率
                def calc_period_return(df, periods):
                    if len(df) < periods + 1:
                        return 0.0
                    current_price = df.iloc[0]['close']
                    past_price = df.iloc[periods]['close']
                    return ((current_price - past_price) / past_price * 100) if past_price > 0 else 0.0
                
                # 计算平均成交量
                def calc_avg_volume(df, periods):
                    if len(df) < periods:
                        return df['volume'].mean() if not df.empty else 0
                    return df.head(periods)['volume'].mean()
                
                # 计算知行指标 - 使用历史收盘价
                close_prices = tech_df['close'].dropna().values
                zhixing_trend = self._calculate_zhixing_trend(close_prices) if len(close_prices) >= 20 else None
                zhixing_multiavg = self._calculate_zhixing_multiavg(close_prices) if len(close_prices) >= 60 else None
                
                # 构建结果字典
                result = {
                    'close': latest_data['close'] or 0,
                    'high': latest_data['high'] or latest_data['close'] or 0,
                    'low': latest_data['low'] or latest_data['close'] or 0,
                    'volume': latest_data['volume'] or 0,
                    'pct_chg': latest_data['price_change_pct'] or 0,
                    
                    # 计算周期收益率
                    'pct_chg_5d': calc_period_return(tech_df, 5),
                    'pct_chg_10d': calc_period_return(tech_df, 10),
                    'pct_chg_20d': calc_period_return(tech_df, 20),
                    
                    # 技术指标 - 使用默认值处理空值
                    'rsi6': latest_data['rsi6'] or 50,
                    'kdj_k': latest_data['kdj_k'] or 50,
                    'kdj_d': latest_data['kdj_d'] or 50,
                    'bbi': latest_data['bbi'] or latest_data['close'] or 0,
                    
                    # 知行指标 - 实际计算值
                    'zhixing_trend': zhixing_trend if zhixing_trend is not None else 50,
                    'zhixing_multiavg': zhixing_multiavg if zhixing_multiavg is not None else 50,
                    
                    # 成交量指标
                    'avg_volume_5': calc_avg_volume(tech_df, 5),
                    'avg_volume_20': calc_avg_volume(tech_df, 20),
                    
                    # 基本面数据
                    'pe_ttm': basic_result[0] if basic_result and basic_result[0] else 0,
                    'pb': basic_result[1] if basic_result and basic_result[1] else 0,
                    'market_cap': basic_result[2] if basic_result and basic_result[2] else 0,  # 万元
                }
                
                return result
                
        except Exception as e:
            logger.error(f"获取股票评分数据失败 {stock_code}: {str(e)}")
            return {}
            
    def _calculate_period_return(self, stock_data: pd.DataFrame, periods: int) -> float:
        """计算指定周期的收益率"""
        try:
            if len(stock_data) < periods + 1:
                return 0.0
            current_price = stock_data.iloc[-1]['close']
            past_price = stock_data.iloc[-(periods+1)]['close']
            return ((current_price - past_price) / past_price * 100) if past_price > 0 else 0.0
        except Exception:
            return 0.0

    def _calculate_avg_volume(self, stock_data: pd.DataFrame, periods: int) -> float:
        """计算指定周期的平均成交量"""
        try:
            if len(stock_data) < periods:
                return stock_data['volume'].mean() if not stock_data.empty else 0
            return stock_data.tail(periods)['volume'].mean()
        except Exception:
            return 0.0
    
    def _calculate_signal_strength(self, indicators: dict, volatility: float) -> float:
        """计算技术信号强度 0-1"""
        strength = 0.5  # 基础强度
        
        # KDJ信号强度
        kdj_k = indicators.get('K', 50)
        kdj_d = indicators.get('D', 50)
        kdj_j = indicators.get('J', 50)
        
        # 低位金叉信号强度高
        if kdj_k > kdj_d and kdj_k < 30:
            strength += 0.3
        elif kdj_k > kdj_d and kdj_k < 50:
            strength += 0.2
        
        # MACD信号强度
        dif = indicators.get('DIF', 0)
        if dif > 0:
            strength += 0.1
        
        # 波动率调整：适度波动有利于短线
        if 2 <= volatility <= 6:
            strength += 0.1
        elif volatility > 10:
            strength -= 0.2
        
        return min(max(strength, 0), 1)
    
    def _find_support_resistance(self, df: pd.DataFrame) -> tuple:
        """寻找近期支撑阻力位"""
        if len(df) < 20:
            close = df['close'].iloc[-1]
            return close * 0.95, close * 1.05
        
        recent_data = df.tail(20)
        
        # 支撑位：近20日最低点附近
        support = recent_data['low'].min()
        
        # 阻力位：近20日最高点附近  
        resistance = recent_data['high'].max()
        
        return support, resistance
    
    def _calculate_smart_prices(self, df: pd.DataFrame, latest: pd.Series, 
                              indicators: dict, volatility: float, is_t0: bool) -> tuple:
        """量化交易专家级智能定价系统 - 基于专业风险管理体系"""
        close_price = latest['close']
        high_price = latest['high']
        low_price = latest['low']
        
        # 计算技术指标和市场环境
        atr = self._calculate_atr(df)
        signal_strength = self._calculate_signal_strength(indicators, volatility)
        support, resistance = self._find_support_resistance(df)
        
        # 市场状态评估
        market_regime = self._assess_market_regime(df, volatility)
        trend_strength = self._calculate_trend_strength(df)
        
        # === 量化交易专家级买入价策略 ===
        buy_price = self._calculate_optimal_entry_price(
            close_price, signal_strength, volatility, is_t0, market_regime
        )
        
        # === 分层风险控制止损策略 ===
        stop_loss_price = self._calculate_multi_layer_stop_loss(
            buy_price, close_price, atr, support, signal_strength, 
            volatility, is_t0, market_regime
        )
        
        # === 动态止盈目标策略 ===
        take_profit_price = self._calculate_dynamic_take_profit(
            buy_price, stop_loss_price, resistance, signal_strength,
            volatility, is_t0, trend_strength, market_regime
        )
        
        # === 最终风险收益比验证 ===
        return self._validate_risk_reward_ratio(buy_price, stop_loss_price, take_profit_price)
    
    def _assess_market_regime(self, df: pd.DataFrame, volatility: float) -> str:
        """评估市场状态：趋势/震荡/高波动"""
        if len(df) < 20:
            return "NORMAL"
        
        recent_data = df.tail(20)
        price_range = (recent_data['high'].max() - recent_data['low'].min()) / recent_data['close'].iloc[-1]
        
        if volatility > 8:
            return "HIGH_VOLATILITY"
        elif price_range > 0.15:
            return "TRENDING"
        else:
            return "CONSOLIDATION"
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """计算趋势强度 0-1"""
        if len(df) < 10:
            return 0.5
        
        recent_closes = df['close'].tail(10)
        if len(recent_closes) < 2:
            return 0.5
            
        # 计算价格动量
        momentum = (recent_closes.iloc[-1] - recent_closes.iloc[0]) / recent_closes.iloc[0]
        
        # 计算趋势一致性
        up_days = sum(recent_closes.diff() > 0)
        consistency = up_days / 9 if momentum > 0 else (9 - up_days) / 9
        
        trend_strength = min(abs(momentum) * 10 * consistency, 1.0)
        return max(trend_strength, 0.1)
    
    def _calculate_optimal_entry_price(self, close_price: float, signal_strength: float, 
                                     volatility: float, is_t0: bool, market_regime: str) -> float:
        """计算最优买入价 - 考虑市场状态和信号质量"""
        base_premium = 0.001  # 基础溢价
        
        # 根据信号强度调整
        if signal_strength > 0.8:
            signal_premium = 0.003  # 强信号可以适度追高
        elif signal_strength > 0.6:
            signal_premium = 0.002
        else:
            signal_premium = 0.001
        
        # 根据市场状态调整
        if market_regime == "HIGH_VOLATILITY":
            regime_premium = -0.001  # 高波动时更保守
        elif market_regime == "TRENDING":
            regime_premium = 0.002   # 趋势市可以追高
        else:
            regime_premium = 0.001   # 震荡市中性
        
        # T+0 vs T+1调整
        trading_premium = 0.001 if is_t0 else 0.0005
        
        # 波动率调整
        volatility_premium = min(volatility / 500, 0.003)
        
        total_premium = base_premium + signal_premium + regime_premium + trading_premium + volatility_premium
        total_premium = max(min(total_premium, 0.012), 0.0005)  # 限制在0.05%-1.2%之间
        
        return round(close_price * (1 + total_premium), 2)
    
    def _calculate_multi_layer_stop_loss(self, buy_price: float, close_price: float, 
                                       atr: float, support: float, signal_strength: float,
                                       volatility: float, is_t0: bool, market_regime: str) -> float:
        """多层次止损体系"""
        # 第一层：技术止损（基于ATR）
        if atr > 0:
            atr_multiplier = self._get_atr_multiplier(signal_strength, market_regime, is_t0)
            technical_stop = buy_price - (atr * atr_multiplier)
        else:
            technical_stop = buy_price * (1 - min(volatility / 100 * 0.5, 0.03))
        
        # 第二层：支撑位止损
        support_stop = support * 0.995  # 支撑位下方0.5%
        
        # 第三层：百分比止损（最后防线）
        # A股T+1日内波动通常2-4%, 止损需留足空间避免被噪声触发
        if market_regime == "HIGH_VOLATILITY":
            max_loss_pct = 0.04 if is_t0 else 0.06   # 高波动时更宽
        elif signal_strength > 0.8:
            max_loss_pct = 0.03 if is_t0 else 0.045  # 强信号适中
        else:
            max_loss_pct = 0.035 if is_t0 else 0.055  # 正常情况
        
        percentage_stop = buy_price * (1 - max_loss_pct)
        
        # 选择最优止损价：取中位数，平衡紧/宽
        candidate_stops = [technical_stop, support_stop, percentage_stop]
        candidate_stops = [s for s in candidate_stops if s < buy_price * 0.995]  # 必须低于买价

        if len(candidate_stops) >= 2:
            # 取中位数：既不过紧也不过松
            candidate_stops.sort()
            optimal_stop = candidate_stops[len(candidate_stops) // 2]
        elif candidate_stops:
            optimal_stop = candidate_stops[0]
        else:
            # 备用方案
            optimal_stop = buy_price * 0.95
        
        return round(optimal_stop, 2)
    
    def _get_atr_multiplier(self, signal_strength: float, market_regime: str, is_t0: bool) -> float:
        """根据市场条件动态调整ATR倍数"""
        base_multiplier = 1.5 if is_t0 else 1.2
        
        # 信号强度调整
        if signal_strength > 0.8:
            base_multiplier *= 0.8  # 强信号收紧止损
        elif signal_strength < 0.5:
            base_multiplier *= 1.2  # 弱信号放宽止损
        
        # 市场状态调整
        if market_regime == "HIGH_VOLATILITY":
            base_multiplier *= 1.5  # 高波动时放宽
        elif market_regime == "CONSOLIDATION":
            base_multiplier *= 0.8  # 震荡市收紧
        
        return max(min(base_multiplier, 2.5), 0.8)
    
    def _calculate_dynamic_take_profit(self, buy_price: float, stop_loss_price: float,
                                     resistance: float, signal_strength: float,
                                     volatility: float, is_t0: bool, 
                                     trend_strength: float, market_regime: str) -> float:
        """动态止盈目标计算"""
        risk_amount = buy_price - stop_loss_price
        
        # 基础风险收益比：根据信号质量动态调整 (10天持仓期，目标需可达)
        if signal_strength > 0.8 and trend_strength > 0.7:
            base_ratio = 2.5  # 强信号强趋势
        elif signal_strength > 0.6:
            base_ratio = 2.0  # 中等信号
        else:
            base_ratio = 1.8  # 弱信号：保守但可达目标
        
        # 市场状态调整
        if market_regime == "TRENDING":
            regime_adjustment = 1.2  # 趋势市提高目标
        elif market_regime == "HIGH_VOLATILITY":
            regime_adjustment = 0.8  # 高波动降低目标
        else:
            regime_adjustment = 1.0  # 震荡市保持中性
        
        # T+0 vs T+1调整
        trading_adjustment = 0.8 if is_t0 else 1.2  # T+0快进快出，T+1可以拿久一点
        
        # 计算动态收益比
        dynamic_ratio = base_ratio * regime_adjustment * trading_adjustment
        dynamic_ratio = max(min(dynamic_ratio, 3.5), 1.5)  # 限制在1.5-3.5倍之间
        
        # 计算目标价
        target_price = buy_price + (risk_amount * dynamic_ratio)
        
        # 阻力位检查：如果阻力位合理且低于目标价，考虑调整
        if resistance > buy_price * 1.01:  # 阻力位必须合理
            resistance_target = resistance * 0.98  # 阻力位下方2%
            if resistance_target < target_price and resistance_target >= buy_price + (risk_amount * 2.0):
                target_price = resistance_target  # 使用阻力位目标，但确保最小2:1收益比
        
        return round(target_price, 2)
    
    def _validate_risk_reward_ratio(self, buy_price: float, stop_loss_price: float, 
                                  take_profit_price: float) -> tuple:
        """最终风险收益比验证和调整"""
        risk = buy_price - stop_loss_price
        reward = take_profit_price - buy_price
        
        if risk <= 0:
            # 异常情况：强制设置合理的风险收益
            stop_loss_price = round(buy_price * 0.975, 2)  # 2.5%止损
            take_profit_price = round(buy_price * 1.05, 2)   # 5%止盈，确保2:1比例
            return buy_price, stop_loss_price, take_profit_price
        
        current_ratio = reward / risk if risk > 0 else 0
        
        if current_ratio < 1.5:
            # 收益比不足，调整止盈价 (最低1.5:1)
            take_profit_price = round(buy_price + (risk * 1.5), 2)
        elif current_ratio > 4.0:
            # 收益比过高，可能不现实，适度调整
            take_profit_price = round(buy_price + (risk * 3.5), 2)
        
        return buy_price, stop_loss_price, take_profit_price

    def _enhance_prices_with_ml(self, stock_info: dict) -> dict:
        """融合ML预测增强止盈止损目标价

        Autoresearch优化参数 (2026-03-23, V4.8.1, 536天回测):
        - 买入价: pred>=0按收盘价, pred<0折价0.5~2.5% (回测最优)
        - 止损: 主板-10% / 创科-15% (宽止损给足弹性空间)
        - 目标: 主板+8% / 创科+12% (高目标持有更大波动)
        - 仓位: 强烈买入15%, 买入8%, 谨慎3%, 观望1%

        回测实证 (V4.8.1, 2024-01-01~2026-03-20):
        - 超额年化: +20383% vs 中证2000
        - Sharpe: 7.246, 胜率: 56.4%, 交易笔数: 6563
        - vs 旧参数: composite 436→20420 (+4580%)
        """
        # V2 optimizer: 使用portfolio_optimizer模块
        if getattr(self, 'optimizer_version', 'v1') == 'v2' and hasattr(self, 'portfolio_optimizer'):
            return self._enhance_prices_with_optimizer_v2(stock_info)

        close = stock_info.get('close_price', 0)
        if close <= 0:
            return stock_info

        # === ML预测数据 ===
        pred_10d = stock_info.get('pred_10d', 0) or 0
        pred_5d = stock_info.get('pred_5d', stock_info.get('predicted_return_5d', 0)) or 0
        pred_3d = stock_info.get('pred_3d', 0) or 0
        pred_15d = stock_info.get('pred_15d', 0) or 0

        # === 已有技术面价格 ===
        tech_buy = stock_info.get('suggested_buy_price', close)
        tech_stop = stock_info.get('stop_loss_price', close * 0.95)
        tech_target = stock_info.get('take_profit_price', close * 1.05)

        # === 选择主力预测horizon ===
        primary_pred = pred_10d if pred_10d != 0 else pred_5d

        # === A股板块判断 ===
        stock_code = stock_info.get('stock_code', '')
        is_wide_limit = stock_code.startswith('30') or stock_code.startswith('688')
        daily_limit = 0.20 if is_wide_limit else 0.10

        # ========== 买入价: 回测最优折扣 (V4901, 532天, 290万样本) ==========
        # pred>=0: 0%折扣期望收益最高 (成交率92%, 折价反而掉成交率损失更大)
        # pred -0.5~0%: 0.5%折扣最优 (成交率78%, 期望+0.71%)
        # pred<-0.5%: 期望为负, 深折等极端低吸
        if primary_pred >= 0:
            entry_discount = 0.0
        elif primary_pred >= -0.005:
            entry_discount = 0.005
        else:
            entry_discount = 0.025
        buy_price = round(close * (1 - entry_discount), 2)
        stock_info['suggested_buy_price'] = buy_price

        # ========== 止损价 ==========
        # 主板-10%, 创/科-15% (宽止损, autoresearch验证更宽=更优)
        base_stop_pct = 0.15 if is_wide_limit else 0.10
        enhanced_stop = close * (1 - base_stop_pct)

        # 约束
        min_stop = close * (1 - daily_limit)
        enhanced_stop = max(enhanced_stop, min_stop)
        enhanced_stop = max(enhanced_stop, buy_price * 0.85)
        enhanced_stop = min(enhanced_stop, buy_price * 0.96)

        # ========== 目标价 ==========
        # 主板+8%, 创/科+12% (高目标, autoresearch验证高目标=高收益)
        max_target_pct = 0.12 if is_wide_limit else 0.10
        min_target_pct = 0.12 if is_wide_limit else 0.08
        ml_target_pct = max(min(primary_pred * 0.8, max_target_pct), min_target_pct)

        # 技术目标(限制)
        tech_target_pct = (tech_target - close) / close if close > 0 else min_target_pct
        tech_target_pct = max(min(tech_target_pct, max_target_pct), min_target_pct)

        # 加权混合
        risk_level = stock_info.get('risk_level', 'medium')
        confidence = {'low': 0.85, 'medium': 0.6, 'high': 0.3}.get(risk_level, 0.5)
        if confidence >= 0.7 and primary_pred > 0.01:
            ml_w, tech_w = 0.65, 0.35
        elif confidence >= 0.4:
            ml_w, tech_w = 0.45, 0.55
        else:
            ml_w, tech_w = 0.25, 0.75

        blended_pct = ml_target_pct * ml_w + tech_target_pct * tech_w
        blended_target = close * (1 + blended_pct)

        # R:R上限约束
        final_risk = buy_price - enhanced_stop
        final_reward = blended_target - buy_price
        if final_risk > 0 and final_reward / final_risk > 3.0:
            blended_target = buy_price + final_risk * 2.5

        # ========== 回写stock_info ==========
        stock_info['stop_loss_price'] = round(enhanced_stop, 2)
        stock_info['take_profit_price'] = round(blended_target, 2)

        final_risk = buy_price - enhanced_stop
        final_reward = blended_target - buy_price
        stock_info['risk_pct'] = round((final_risk / buy_price) * 100, 2) if buy_price > 0 else 0
        stock_info['reward_pct'] = round((final_reward / buy_price) * 100, 2) if buy_price > 0 else 0
        stock_info['risk_reward_ratio'] = round(final_reward / final_risk, 2) if final_risk > 0 else 0

        # 仓位建议
        stock_info['position_pct'] = self._suggest_position_size(stock_info)

        return stock_info

    def _enhance_prices_with_optimizer_v2(self, stock_info: dict) -> dict:
        """V2: 使用portfolio_optimizer计算自适应价格"""
        close = stock_info.get('close_price', 0)
        if close <= 0:
            return stock_info

        code = stock_info.get('stock_code', '')
        env_score = getattr(self, '_cached_env_score', 50.0)
        analysis_date = stock_info.get('analysis_date', '')

        # 获取分析日期前80日K线 (避免look-ahead bias)
        try:
            import sqlite3
            import numpy as np
            db_path = Path(__file__).parent / 'data_adapter' / 'stock_data.db'
            conn = sqlite3.connect(str(db_path))
            try:
                if analysis_date:
                    df = pd.read_sql_query("""
                        SELECT dq.trade_date, dq.high, dq.low, dq.close
                        FROM daily_quotes dq
                        JOIN securities s ON dq.security_id = s.id
                        WHERE s.code = ? AND dq.trade_date <= ?
                        ORDER BY dq.trade_date DESC LIMIT 80
                    """, conn, params=[code, analysis_date])
                else:
                    df = pd.read_sql_query("""
                        SELECT dq.trade_date, dq.high, dq.low, dq.close
                        FROM daily_quotes dq
                        JOIN securities s ON dq.security_id = s.id
                        WHERE s.code = ? ORDER BY dq.trade_date DESC LIMIT 80
                    """, conn, params=[code])
            finally:
                conn.close()
            if df is None or len(df) < 20:
                return stock_info
            df = df.sort_values('trade_date')
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
        except Exception as e:
            logger.debug(f"V2 optimizer获取K线失败 {code}: {e}")
            return stock_info

        stock_info = self.portfolio_optimizer.compute_prices(
            stock_info, highs, lows, closes, env_score)
        stock_info['position_pct'] = stock_info.get('position_pct', 5)
        return stock_info

    def _suggest_position_size(self, stock_info: dict) -> int:
        """基于风险等级和投资建议计算仓位百分比

        Autoresearch优化 (2026-03-23):
        - 强信号集中, 弱信号降权 → 提升仓位加权收益
        - 强烈买入15%, 买入8%, 谨慎3%, 观望1%
        """
        rec = stock_info.get('recommendation', '观望')
        risk = stock_info.get('risk_level', 'medium')

        # 回避类直接0
        if rec in ('回避', '卖出', '谨慎卖出', '强烈卖出'):
            return 0

        # 基础仓位: 强信号集中
        base = {'强烈买入': 15, '买入': 8, '谨慎买入': 3, '观望': 1}.get(rec, 1)

        # 风险调整
        risk_adj = {'low': 0, 'medium': -2, 'high': -4}.get(risk, -2)

        return max(base + risk_adj, 0)

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = df.copy()
        
        try:
            # 导入计算函数
            selector_path = Path("stock_selctor/Selector.py")
            spec = importlib.util.spec_from_file_location("Selector", selector_path)
            selector_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(selector_module)
            
            # 计算KDJ
            df = selector_module.compute_kdj(df)
            
            # 计算BBI
            df['BBI'] = selector_module.compute_bbi(df)
            
            # 计算DIF
            df['DIF'] = selector_module.compute_dif(df)
            
        except Exception as e:
            logger.warning(f"计算技术指标失败: {e}")
            
        return df
    
    def _get_recommendation(self, pred_3d: float, pred_5d: float,
                             pred_10d: float, pred_15d: float = 0.0) -> str:
        """基于composite score的历史百分位阈值生成投资建议

        优先使用scorer中嵌入的recommendation_thresholds (基于全市场历史校准)。
        """
        # 尝试从各版本scorer获取方法
        for engine_attr in ('scoring_engine_v44', 'scoring_engine_v43', 'scoring_engine_v396', 'scoring_engine_v395'):
            engine = getattr(self, engine_attr, None)
            if engine and hasattr(engine, '_recommendation_from_composite'):
                return engine._recommendation_from_composite(pred_3d, pred_5d, pred_10d, pred_15d)
        # fallback: 基于composite绝对值
        composite = pred_3d * 0.1 + pred_5d * 0.2 + pred_10d * 0.4 + pred_15d * 0.3
        if composite >= 0.015:
            return '强烈买入'
        elif composite >= 0.008:
            return '买入'
        elif composite >= 0.003:
            return '谨慎买入'
        elif composite >= -0.002:
            return '观望'
        return '回避'

    def _get_risk_level(self, pred_3d: float, pred_5d: float,
                        pred_10d: float, pred_15d: float = 0.0) -> str:
        """基于composite score的历史百分位阈值生成风险等级"""
        for engine_attr in ('scoring_engine_v44', 'scoring_engine_v43', 'scoring_engine_v396', 'scoring_engine_v395'):
            engine = getattr(self, engine_attr, None)
            if engine and hasattr(engine, '_risk_level_from_composite'):
                return engine._risk_level_from_composite(pred_3d, pred_5d, pred_10d, pred_15d)
        # fallback
        composite = pred_3d * 0.1 + pred_5d * 0.2 + pred_10d * 0.4 + pred_15d * 0.3
        if composite >= 0.008:
            return 'low'
        elif composite >= -0.002:
            return 'medium'
        return 'high'

    def generate_investment_recommendation(self, stock_info: Dict[str, Any]) -> Dict[str, str]:
        """生成投资建议 - 基于优化后评分系统生成买入/持有/卖出建议"""
        try:
            # 🔧 V4.9.0/V4.9.0.1/V4.9.1: 直接使用scorer的动态投资建议
            if hasattr(self, 'scoring_version') and self.scoring_version in ("v4.9.0", "v4.9.0.1", "v4.9.0.2", "v4.9.1"):
                stock_code = stock_info.get('stock_code', '')
                cached = self.v44_batch_cache.get(stock_code, {})
                if cached:
                    rec = cached.get('recommendation', '观望')
                    fs = cached.get('score', 50.0)
                    return {
                        'recommendation': rec,
                        'confidence': 'high' if fs >= 90 else 'medium' if fs >= 71 else 'low',
                        'technical_rating': f"Q95={cached.get('q95_pred_10d',0):.3f}",
                        'risk_rating': '低' if fs >= 90 else '中等' if fs >= 71 else '偏高',
                        'score': fs,
                        'final_score': fs,
                        'pred_3d': cached.get('pred_3d', 0),
                        'pred_5d': cached.get('pred_5d', 0),
                        'pred_10d': cached.get('pred_10d', 0),
                        'pred_15d': cached.get('pred_15d', 0),
                        'rank_score': cached.get('rank_score', 0),
                        'predicted_return_5d': cached.get('pred_5d', 0),
                        'overall_quality': 0.85,
                        'quality_score': 0.85,
                        'confidence_score': 0.85,
                        'risk_level': '低' if fs >= 90 else '中等',
                    }

            # 🔧 V3.81版本直接使用已计算的投资建议，不重新计算
            if hasattr(self, 'scoring_version') and self.scoring_version == "v3.81":
                # V3.81版本已经在批处理阶段计算了投资建议
                existing_recommendation = stock_info.get('recommendation', None)
                existing_confidence = stock_info.get('confidence_level', None)
                if existing_recommendation and existing_confidence:
                    return {
                        'recommendation': existing_recommendation,
                        'confidence': existing_confidence,
                        'technical_rating': '优秀',
                        'risk_rating': stock_info.get('risk_level', '中等'),
                        'score': stock_info.get('final_score', 50.0)
                    }

            # 使用优化后的综合评分系统
            score, detailed_info = self.calculate_comprehensive_score(stock_info)

            # 获取新评分系统的推荐
            new_recommendation = detailed_info.get('recommendation', '观望')
            confidence = detailed_info.get('confidence', '低')
            
            # 策略交集强度评估
            strategy_count = stock_info.get('selected_by_strategies', 1)
            strategies = stock_info.get('strategies', [])
            
            # 技术指标强度评估
            kdj_k = stock_info.get('kdj_k', 50)
            kdj_j = stock_info.get('kdj_j', 50) 
            dif = stock_info.get('dif', 0)
            bbi_signal = stock_info.get('close', 0) > stock_info.get('bbi', 0)
            
            # 风险评估
            volatility = stock_info.get('volatility', 0) 
            risk_reward_ratio = stock_info.get('risk_reward_ratio', 1.0)
            
            # 综合决策逻辑
            base_recommendation = new_recommendation
            base_confidence = confidence

            # V3.9+ 生产版本: 完全信任composite阈值系统的推荐，不做任何策略加成/惩罚
            # 回测证明composite阈值(强烈买入=Top5%, 买入=Top20%)比策略数+score混合逻辑更可靠
            if hasattr(self, 'scoring_version') and self.scoring_version in [
                "v3.9", "v3.94", "v3.95", "v3.96", "v4.0", "v4.2", "v4.3",
                "v4.4", "v4.4.2", "v4.5", "v4.6", "v4.7", "v4.7.1", "v4.7.2",
                "v4.7.3", "v4.7.4", "v4.7.5", "v4.7.6", "v4.7.7", "v4.7.8", "v4.7.9", "v4.8.0", "v4.8.1", "v4.8.2", "v4.8.4", "v4.8.5", "v4.8.6", "v4.8.7", "v4.8.8", "v5.0"]:
                recommendation = base_recommendation
                confidence = base_confidence

            # 旧版本: 保留策略交集加成逻辑
            elif strategy_count >= 3 or 'SuperB1战法' in strategies:
                if base_recommendation == "观望" and score >= 65:
                    recommendation = "谨慎买入"
                    confidence = "中"
                elif base_recommendation == "谨慎买入" and score >= 75:
                    recommendation = "买入"
                    confidence = "高"
                elif base_recommendation == "买入" and score >= 80 and risk_reward_ratio >= 3.0:
                    recommendation = "强烈买入"
                    confidence = "优秀"
                else:
                    recommendation = base_recommendation
                    confidence = base_confidence
            elif strategy_count == 2:
                if base_recommendation == "观望" and score >= 68:
                    recommendation = "谨慎买入"
                    confidence = "中"
                elif base_recommendation == "谨慎买入" and score >= 78:
                    recommendation = "买入"
                    confidence = "高"
                else:
                    recommendation = base_recommendation
                    confidence = base_confidence
            else:
                if hasattr(self, 'scoring_version') and self.scoring_version == "v3.41":
                    if base_recommendation == "买入" and score < 60:
                        recommendation = "谨慎买入"
                        confidence = "中"
                    elif base_recommendation == "谨慎买入" and score < 50:
                        recommendation = "观望"
                        confidence = "低"
                    else:
                        recommendation = base_recommendation
                        confidence = base_confidence
                elif hasattr(self, 'scoring_version') and self.scoring_version == "v3.81":
                    if 'recommendation' in detailed_info:
                        recommendation = detailed_info.get('recommendation', base_recommendation)
                        confidence = detailed_info.get('confidence', base_confidence)
                        return {
                            'recommendation': recommendation,
                            'confidence': confidence,
                            'score': score,
                            'detailed_info': detailed_info
                        }
                    else:
                        if score >= 75:
                            recommendation = "买入"
                            confidence = "高"
                        elif score >= 60:
                            recommendation = "谨慎买入"
                            confidence = "中"
                        else:
                            recommendation = base_recommendation
                            confidence = base_confidence
                else:
                    if base_recommendation == "买入" and score < 75:
                        recommendation = "谨慎买入"
                        confidence = "中"
                    elif base_recommendation == "谨慎买入" and score < 65:
                        recommendation = "观望"
                        confidence = "低"
                    else:
                        recommendation = base_recommendation
                        confidence = base_confidence
                    
            # 技术面评价
            if kdj_k < 30 and kdj_j < 20 and dif > 0:
                technical_rating = "优秀"
            elif kdj_k < 50 and bbi_signal and dif > -0.1:
                technical_rating = "良好"
            elif kdj_k < 70:
                technical_rating = "中性"
            else:
                technical_rating = "谨慎"
                
            # 风险评价
            if volatility < 3.0 and risk_reward_ratio >= 3.0:
                risk_rating = "优秀"
            elif volatility < 5.0 and risk_reward_ratio >= 2.0:
                risk_rating = "良好"
            elif volatility < 8.0:
                risk_rating = "中性"
            else:
                risk_rating = "谨慎"
                
            return {
                'recommendation': recommendation,
                'confidence': confidence,
                'technical_rating': technical_rating,
                'risk_rating': risk_rating,
                'score': score,  # 保留数字评分用于内部排序
                'detailed_scoring': detailed_info,  # 新增：详细评分信息
                'factor_scores': detailed_info.get('factor_scores', {}),  # 因子分数
                # 🏆 V3.9+专用字段 - 提升到根层级供报告使用
                'predicted_return_5d': detailed_info.get('predicted_return_5d', 0.0),
                'pred_3d': detailed_info.get('pred_3d', 0.0),
                'pred_5d': detailed_info.get('pred_5d', 0.0),
                'pred_10d': detailed_info.get('pred_10d', 0.0),
                'pred_15d': detailed_info.get('pred_15d', 0.0),
                # 原始预测值(校准前), 用于报告展示; 校准后的pred_Xd用于评分/推荐
                'raw_pred_3d': detailed_info.get('raw_pred_3d', detailed_info.get('pred_3d', 0.0)),
                'raw_pred_5d': detailed_info.get('raw_pred_5d', detailed_info.get('pred_5d', 0.0)),
                'raw_pred_10d': detailed_info.get('raw_pred_10d', detailed_info.get('pred_10d', 0.0)),
                'raw_pred_15d': detailed_info.get('raw_pred_15d', detailed_info.get('pred_15d', 0.0)),
                'confidence_score': detailed_info.get('confidence_score', 0.0),
                'risk_level': detailed_info.get('risk_level', 'medium'),
                'rank_score': detailed_info.get('rank_score'),
                'scoring_system': 'v2.0 - 基于3949只股票实际表现优化'
            }
            
        except Exception as e:
            logger.warning(f"生成投资建议失败: {e}")
            return {
                'recommendation': '观望',
                'confidence': '中性', 
                'technical_rating': '中性',
                'risk_rating': '中性',
                'score': 50.0
            }

    def _calculate_quality_score(self, final_score, confidence_score, prediction_data):
        """🔧 新增：基于多因素计算质量评分"""
        try:
            # 因素1：置信度权重 (40%)
            confidence_component = confidence_score * 0.4

            # 因素2：评分高度权重 (30%) - 高分股票质量更高
            score_component = (final_score / 100.0) * 0.3

            # 因素3：预测一致性权重 (30%) - 基于原始预测的方差
            if isinstance(prediction_data, dict) and 'raw_predictions' in prediction_data:
                raw_preds = list(prediction_data['raw_predictions'].values())
                if len(raw_preds) > 1:
                    # 低方差=高一致性=高质量
                    variance = np.var(raw_preds)
                    consistency_component = (1.0 / (1.0 + variance)) * 0.3
                else:
                    consistency_component = 0.15  # 默认中等
            else:
                consistency_component = 0.15  # 默认中等

            # 综合质量评分
            quality_score = confidence_component + score_component + consistency_component
            return round(np.clip(quality_score, 0.1, 0.95), 3)

        except Exception as e:
            logger.warning(f"质量评分计算失败: {e}")
            return confidence_score  # 回退到置信度

    def _calculate_risk_level(self, final_score, confidence_score, prediction_data):
        """🔧 新增：基于多因素计算风险等级"""
        try:
            # 因素1：评分风险 - 低分=高风险
            score_risk = 1.0 - (final_score / 100.0)  # 0-1，1表示高风险

            # 因素2：置信度风险 - 低置信度=高风险
            confidence_risk = 1.0 - confidence_score  # 0-1，1表示高风险

            # 因素3：预测波动风险 - 高方差=高风险
            if isinstance(prediction_data, dict) and 'raw_predictions' in prediction_data:
                raw_preds = list(prediction_data['raw_predictions'].values())
                if len(raw_preds) > 1:
                    variance = np.var(raw_preds)
                    volatility_risk = min(variance / 10.0, 1.0)  # 标准化到0-1
                else:
                    volatility_risk = 0.5  # 默认中等风险
            else:
                volatility_risk = 0.5  # 默认中等风险

            # 综合风险评分：评分风险40% + 置信度风险35% + 波动风险25%
            risk_score = score_risk * 0.4 + confidence_risk * 0.35 + volatility_risk * 0.25

            # 转换为风险等级
            if risk_score <= 0.3:
                return 'low'
            elif risk_score <= 0.6:
                return 'medium'
            else:
                return 'high'

        except Exception as e:
            logger.warning(f"风险等级计算失败: {e}")
            # 回退到简单规则
            return 'low' if final_score > 70 else 'medium' if final_score > 50 else 'high'

    def _calculate_risk_level_v381(self, final_score, confidence_score, quality_score):
        """🎯 V3.81专用：基于Level 4质量评分的风险等级计算"""
        try:
            # 因素1：评分风险 - 低分=高风险
            score_risk = 1.0 - (final_score / 100.0)  # 0-1，1表示高风险

            # 因素2：置信度风险 - 低置信度=高风险
            confidence_risk = 1.0 - confidence_score  # 0-1，1表示高风险

            # 因素3：🎯 Level 4质量风险 - 低质量=高风险
            quality_risk = 1.0 - quality_score  # 0-1，1表示高风险

            # V3.81特殊权重：更重视Level 4质量评分
            # 质量风险45% + 评分风险30% + 置信度风险25%
            risk_score = quality_risk * 0.45 + score_risk * 0.30 + confidence_risk * 0.25

            # 基于Level 4质量评分的精细化风险等级
            if risk_score <= 0.25 and quality_score >= 0.7:
                return 'very_low'  # 新增：极低风险
            elif risk_score <= 0.35:
                return 'low'
            elif risk_score <= 0.55:
                return 'medium'
            elif risk_score <= 0.75:
                return 'high'
            else:
                return 'very_high'  # 新增：极高风险

        except Exception as e:
            logger.warning(f"V3.81风险等级计算失败: {e}")
            # 回退到基于质量评分的简单规则
            if quality_score >= 0.7:
                return 'low'
            elif quality_score >= 0.4:
                return 'medium'
            else:
                return 'high'

    def _calculate_risk_level_v39(self, final_score, confidence_score):
        """🏆 V3.9.0专用：基于A级模型的风险等级计算"""
        try:
            # 因素1：评分风险 - 低分=高风险
            score_risk = 1.0 - (final_score / 100.0)  # 0-1，1表示高风险

            # 因素2：置信度风险 - 低置信度=高风险
            confidence_risk = 1.0 - confidence_score  # 0-1，1表示高风险

            # V3.9.0权重：评分60% + 置信度40% (简化，更依赖模型预测)
            risk_score = score_risk * 0.6 + confidence_risk * 0.4

            # 基于A级模型的风险等级划分
            if risk_score <= 0.20:
                return 'very_low'
            elif risk_score <= 0.35:
                return 'low'
            elif risk_score <= 0.55:
                return 'medium'
            elif risk_score <= 0.75:
                return 'high'
            else:
                return 'very_high'

        except Exception as e:
            logger.warning(f"V3.9.0风险等级计算失败: {e}")
            return 'medium'

    def calculate_comprehensive_score(self, stock_info: Dict[str, Any], trade_date: str = None) -> Tuple[float, Dict]:
        """
        使用优化后的评分系统计算股票综合评分
        
        v2版本 - 基于3949只股票实际表现优化的多因子评分系统:
        - 动量因子40% (识别强势股)
        - 均值回归25% (价值修复)  
        - 量价突破20% (突破确认)
        - 相对强度10% (相对表现)
        - 稳定性5% (风险控制)
        
        v3版本 - 智能动态权重评分系统:
        - 技术指标动态权重
        - 基本面自适应权重
        - 市场环境智能识别
        - 多时间窗口综合
        """
        try:
            # 处理 stock_info 参数（可能是字符串或字典）
            if isinstance(stock_info, str):
                stock_code = stock_info
            else:
                stock_code = stock_info.get('stock_code', stock_info.get('code', ''))

            if not stock_code:
                return 50.0, {'error': '股票代码缺失'}
            
            if trade_date is None:
                trade_date = datetime.now().strftime('%Y-%m-%d')

            if self.scoring_version in ("v4.4", "v4.4.2", "v4.5", "v4.6", "v4.7.1", "v4.7.2", "v4.7.3", "v4.7.4", "v4.7.5", "v4.7.6", "v4.7.7", "v4.7.8", "v4.7.9", "v4.8.0", "v4.8.1", "v4.8.2", "v4.8.4", "v4.8.5", "v4.8.6", "v4.8.7", "v4.8.8", "v4.9.0", "v4.9.0.1", "v4.9.0.2", "v4.9.1", "v5.0"):
                # V4.4+: V4.3信号+增强模块
                try:
                    if stock_code in self.v44_batch_cache:
                        result = self.v44_batch_cache[stock_code]
                    else:
                        results = self.scoring_engine_v44.predict_scores([stock_code], trade_date)
                        result = results.get(stock_code, {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0, 'pred_15d': 0})

                    if not result:
                        return 45, {'error': '无法获取评分', 'scoring_method': 'V4.4_Enhanced'}

                    final_score = result.get('score', 50.0)
                    pred_3d = result.get('pred_3d', 0.0)
                    pred_5d = result.get('pred_5d', 0.0)
                    pred_10d = result.get('pred_10d', 0.0)
                    pred_15d = result.get('pred_15d', 0.0)

                    detailed_info = {
                        'final_score': final_score,
                        'confidence_score': 0.85,
                        'confidence_level': 'high' if final_score >= 70 else 'medium' if final_score >= 55 else 'low',
                        'short_term_score': final_score,
                        'medium_term_score': final_score,
                        'long_term_score': final_score,
                        'predicted_return_5d': pred_5d,
                        'pred_3d': pred_3d,
                        'pred_5d': pred_5d,
                        'pred_10d': pred_10d,
                        'pred_15d': pred_15d,
                        'overall_quality': 0.85,
                        'quality_score': 0.85,
                        # 推荐阈值已校准到与pred_Xd相同尺度(含isotonic)
                        'risk_level': self._get_risk_level(pred_3d, pred_5d, pred_10d, pred_15d),
                        'recommendation': self._get_recommendation(pred_3d, pred_5d, pred_10d, pred_15d),
                        'confidence': 'high' if final_score >= 70 else 'medium' if final_score >= 55 else 'low',
                        'scoring_method': 'V4.4_Enhanced_6Modules',
                        'exec_filter': result.get('exec_filter', ''),
                        'regime_info': result.get('regime_info', {}),
                        'rank_score': result.get('rank_score'),
                    }
                    return final_score, detailed_info

                except Exception as e:
                    logger.error(f"v4.4评分系统错误 {stock_code}: {str(e)}")
                    return 45, {'error': f'系统错误: {str(e)}', 'scoring_method': 'V4.4_Enhanced'}

            elif self.scoring_version == "v4.3":
                # 使用v4.3 扩展特征+强正则+4目标 评分系统
                try:
                    if stock_code in self.v43_batch_cache:
                        result = self.v43_batch_cache[stock_code]
                    else:
                        results = self.scoring_engine_v43.predict_scores([stock_code], trade_date)
                        result = results.get(stock_code, {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0, 'pred_15d': 0})

                    if not result:
                        return 45, {'error': '无法获取评分', 'scoring_method': 'V4.3_Enhanced'}

                    final_score = result.get('score', 50.0)
                    pred_3d = result.get('pred_3d', 0.0)
                    pred_5d = result.get('pred_5d', 0.0)
                    pred_10d = result.get('pred_10d', 0.0)
                    pred_15d = result.get('pred_15d', 0.0)

                    detailed_info = {
                        'final_score': final_score,
                        'confidence_score': 0.85,
                        'confidence_level': 'high' if final_score >= 70 else 'medium' if final_score >= 55 else 'low',
                        'short_term_score': final_score,
                        'medium_term_score': final_score,
                        'long_term_score': final_score,
                        'predicted_return_5d': pred_5d,
                        'pred_3d': pred_3d,
                        'pred_5d': pred_5d,
                        'pred_10d': pred_10d,
                        'pred_15d': pred_15d,
                        'overall_quality': 0.85,
                        'quality_score': 0.85,
                        'risk_level': self._get_risk_level(pred_3d, pred_5d, pred_10d, pred_15d),
                        'recommendation': self._get_recommendation(pred_3d, pred_5d, pred_10d, pred_15d),
                        'confidence': 'high' if final_score >= 70 else 'medium' if final_score >= 55 else 'low',
                        'scoring_method': 'V4.3_Enhanced_WalkForward',
                    }
                    return final_score, detailed_info

                except Exception as e:
                    logger.error(f"v4.3评分系统错误 {stock_code}: {str(e)}")
                    return 45, {'error': f'系统错误: {str(e)}', 'scoring_method': 'V4.3_Enhanced'}

            elif self.scoring_version in ("v4.0", "v4.2"):
                # 使用v4.0/v4.2 Cross-Sectional Alpha评分系统
                try:
                    if stock_code in self.v40_batch_cache:
                        result = self.v40_batch_cache[stock_code]
                    else:
                        result = self.scoring_engine_v40.predict_score(stock_code, trade_date)

                    if not result:
                        return 45, {'error': '无法获取评分', 'scoring_method': 'V4.0_CrossSectional'}

                    final_score = result.get('score', 50.0)
                    predicted_excess = result.get('predicted_excess_return_5d', 0.0)
                    confidence_score = result.get('confidence', 0.7)
                    recommendation = result.get('recommendation', '观望')

                    detailed_info = {
                        'final_score': final_score,
                        'confidence_score': confidence_score,
                        'confidence_level': 'high' if confidence_score > 0.7 else 'medium' if confidence_score > 0.4 else 'low',
                        'short_term_score': final_score,
                        'medium_term_score': final_score,
                        'long_term_score': final_score,
                        'predicted_excess_return_5d': predicted_excess,
                        'predicted_return_5d': predicted_excess,
                        'overall_quality': confidence_score,
                        'quality_score': confidence_score,
                        'risk_level': 'medium',
                        'recommendation': recommendation,
                        'confidence': 'high' if confidence_score > 0.7 else 'medium',
                        'scoring_method': result.get('scoring_method', 'V4.0_CrossSectional_Alpha'),
                        'model_grade': result.get('model_grade', 'TBD'),
                    }

                    return final_score, detailed_info

                except Exception as e:
                    logger.error(f"v4.0/v4.2评分系统错误 {stock_code}: {str(e)}")
                    method = 'V4.2_HybridAlpha' if self.scoring_version == 'v4.2' else 'V4.0_CrossSectional'
                    return 45, {'error': f'系统错误: {str(e)}', 'scoring_method': method}

            elif self.scoring_version == "v5.0":
                # V5.0 Unified Feature Fusion (v39+v40+neural)
                try:
                    if stock_code in self.v500_batch_cache:
                        result = self.v500_batch_cache[stock_code]
                    else:
                        results = self.scoring_engine_v500.predict_scores([stock_code], trade_date)
                        result = results.get(stock_code, {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0})

                    if not result:
                        return 45, {'error': '无法获取评分', 'scoring_method': 'V5.0_UnifiedFusion'}

                    final_score = result.get('score', 50.0)
                    pred_3d = result.get('pred_3d', 0.0)
                    pred_5d = result.get('pred_5d', 0.0)
                    pred_10d = result.get('pred_10d', 0.0)

                    detailed_info = {
                        'final_score': final_score,
                        'confidence_score': 0.85,
                        'confidence_level': 'high' if final_score >= 75 else 'medium' if final_score >= 55 else 'low',
                        'short_term_score': final_score,
                        'medium_term_score': final_score,
                        'long_term_score': final_score,
                        'predicted_return_5d': pred_5d,
                        'pred_3d': pred_3d,
                        'pred_5d': pred_5d,
                        'pred_10d': pred_10d,
                        'overall_quality': 0.85,
                        'quality_score': 0.85,
                        'risk_level': 'medium',
                        'recommendation': '买入' if final_score >= 75 else '观望' if final_score >= 55 else '回避',
                        'confidence': 'high' if final_score >= 75 else 'medium',
                        'scoring_method': 'V5.0_UnifiedFusion',
                    }
                    return final_score, detailed_info

                except Exception as e:
                    logger.error(f"v5.0评分系统错误 {stock_code}: {str(e)}")
                    return 45, {'error': f'系统错误: {str(e)}', 'scoring_method': 'V5.0_UnifiedFusion'}

            elif self.scoring_version == "v3.96":
                # 使用v3.96 Robust Z-Score评分系统
                try:
                    if stock_code in self.v396_batch_cache:
                        result = self.v396_batch_cache[stock_code]
                    else:
                        results = self.scoring_engine_v396.predict_scores([stock_code], trade_date)
                        result = results.get(stock_code, {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0})

                    if not result:
                        return 45, {'error': '无法获取评分', 'scoring_method': 'V3.96_RobustZScore'}

                    final_score = result.get('score', 50.0)
                    pred_3d = result.get('pred_3d', 0.0)
                    pred_5d = result.get('pred_5d', 0.0)
                    pred_10d = result.get('pred_10d', 0.0)

                    detailed_info = {
                        'final_score': final_score,
                        'confidence_score': 0.85,
                        'confidence_level': 'high' if final_score >= 75 else 'medium' if final_score >= 55 else 'low',
                        'short_term_score': final_score,
                        'medium_term_score': final_score,
                        'long_term_score': final_score,
                        'predicted_return_5d': pred_5d,
                        'pred_3d': pred_3d,
                        'pred_5d': pred_5d,
                        'pred_10d': pred_10d,
                        'overall_quality': 0.85,
                        'quality_score': 0.85,
                        'risk_level': self._get_risk_level(pred_3d, pred_5d, pred_10d, 0.0),
                        'recommendation': self._get_recommendation(pred_3d, pred_5d, pred_10d, 0.0),
                        'confidence': 'high' if final_score >= 75 else 'medium',
                        'scoring_method': 'V3.96_RobustZScore_IndustryExcess',
                    }
                    return final_score, detailed_info

                except Exception as e:
                    logger.error(f"v3.96评分系统错误 {stock_code}: {str(e)}")
                    return 45, {'error': f'系统错误: {str(e)}', 'scoring_method': 'V3.96_RobustZScore'}

            elif self.scoring_version == "v3.95":
                # 使用v3.9.5多目标预测评分系统 - 🚀 MULTI-TARGET PREDICTION MODEL
                try:
                    # 🔥 优先使用缓存的V3.95批量结果
                    if stock_code in self.v395_batch_cache:
                        result = self.v395_batch_cache[stock_code]
                        logger.debug(f"使用V3.95批量缓存结果 {stock_code}")
                    else:
                        # 回退到单只评分
                        results = self.scoring_engine_v395.predict_scores([stock_code], trade_date)
                        result = results.get(stock_code, {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0})
                        logger.debug(f"V3.95使用单只评分（无缓存）{stock_code}")

                    if not result:
                        logger.warning(f"v3.9.5无法获取股票评分 {stock_code}")
                        return 45, {'error': '无法获取评分', 'scoring_method': 'V3.9.5_MultiTarget'}

                    # V3.9.5返回的结果格式
                    final_score = result.get('score', 50.0)
                    pred_3d = result.get('pred_3d', 0.0)
                    pred_5d = result.get('pred_5d', 0.0)
                    pred_10d = result.get('pred_10d', 0.0)

                    # 计算综合预测收益（加权平均）
                    predicted_return = 0.4 * pred_3d + 0.35 * pred_5d + 0.25 * pred_10d
                    confidence_score = min(0.9, 0.6 + abs(predicted_return) * 5)  # 基于预测强度的置信度

                    # 投资建议 - 使用composite历史百分位阈值 (v3.96无15d，传0)
                    recommendation = self._get_recommendation(pred_3d, pred_5d, pred_10d, 0.0)

                    # 计算分期评分
                    short_term_score = 50 + pred_3d * 500  # 3日收益映射
                    medium_term_score = 50 + pred_5d * 400  # 5日收益映射
                    long_term_score = 50 + pred_10d * 300  # 10日收益映射

                    logger.debug(f"V3.9.5 {stock_code}: 综合评分={final_score:.1f}, 3d={pred_3d:.2%}, 5d={pred_5d:.2%}, 10d={pred_10d:.2%}")

                    detailed_info = {
                        'final_score': final_score,
                        'pred_3d': pred_3d,
                        'pred_5d': pred_5d,
                        'pred_10d': pred_10d,
                        'predicted_return': predicted_return,
                        'confidence_score': confidence_score,
                        'confidence_level': 'high' if confidence_score > 0.7 else 'medium' if confidence_score > 0.4 else 'low',
                        'short_term_score': short_term_score,
                        'medium_term_score': medium_term_score,
                        'long_term_score': long_term_score,
                        'predicted_return_5d': pred_5d,
                        'overall_quality': confidence_score,
                        'quality_score': confidence_score,
                        'risk_level': self._get_risk_level(pred_3d, pred_5d, pred_10d, 0.0),
                        'recommendation': recommendation,
                        'confidence': 'high' if confidence_score > 0.7 else 'medium' if confidence_score > 0.4 else 'low',
                        'scoring_method': 'V3.9.5_MultiTarget_Rolling',
                        'model_type': 'multi_target_prediction',
                        'targets': '3d/5d/10d',
                        'temporal_weights': {
                            'short_term': 0.4,
                            'medium_term': 0.35,
                            'long_term': 0.25
                        }
                    }

                    return final_score, detailed_info

                except Exception as e:
                    logger.error(f"v3.9.5评分系统错误 {stock_code}: {str(e)}")
                    return 45, {'error': f'系统错误: {str(e)}', 'scoring_method': 'V3.9.5_MultiTarget'}

            elif self.scoring_version == "v3.94":
                # 使用v3.9.4生产版评分系统 - 🏆 PRODUCTION A+ GRADE MODEL (带活跃市值特征)
                try:
                    # 🔥 优先使用缓存的V3.94批量百分位排名结果
                    if stock_code in self.v394_batch_cache:
                        result = self.v394_batch_cache[stock_code]
                        logger.debug(f"使用V3.94批量缓存结果 {stock_code}")
                    else:
                        # 回退到单只评分（不推荐，会导致评分集中）
                        result = self.scoring_engine_v394.predict_score(stock_code, trade_date)
                        logger.debug(f"V3.94使用单只评分（无缓存）{stock_code}")

                    if not result:
                        logger.warning(f"v3.9.4无法获取股票评分 {stock_code}")
                        return 45, {'error': '无法获取评分', 'scoring_method': 'V3.9.4_Production'}

                    # V3.9.4返回的结果格式（批量模式与单只模式兼容）
                    final_score = result.get('score', 50.0)
                    predicted_return = result.get('predicted_return_5d', result.get('predicted_return', 0.0))
                    confidence_score = result.get('confidence', 0.8)
                    recommendation = result.get('recommendation', '观望')
                    percentile_rank = result.get('percentile_rank', 50.0)  # 百分位排名

                    # 计算分期评分 (基于5日收益预测)
                    short_term_score = final_score * 1.1  # 短期略高
                    medium_term_score = final_score
                    long_term_score = final_score * 0.9  # 长期略低

                    logger.debug(f"V3.9.4 {stock_code}: 综合评分={final_score:.1f}, 百分位排名={percentile_rank:.1f}%, 预测5日收益={predicted_return:.2%}, 投资建议={recommendation}")

                    detailed_info = {
                        'final_score': final_score,
                        'percentile_rank': percentile_rank,  # 🔥 百分位排名（核心区分度指标）
                        'confidence_score': confidence_score,
                        'confidence_level': 'high' if confidence_score > 0.7 else 'medium' if confidence_score > 0.4 else 'low',
                        'short_term_score': short_term_score,
                        'medium_term_score': medium_term_score,
                        'long_term_score': long_term_score,
                        'predicted_return_5d': predicted_return,
                        'overall_quality': confidence_score,
                        'quality_score': confidence_score,
                        'risk_level': self._calculate_risk_level_v39(final_score, confidence_score),
                        'recommendation': recommendation,
                        'confidence': 'high' if confidence_score > 0.7 else 'medium' if confidence_score > 0.4 else 'low',
                        'scoring_method': 'V3.9.4_Production_A+_Grade_Percentile',
                        'model_grade': 'A+',
                        'model_ic': 0.1363,
                        'top20_winrate': 0.5643,
                        'features': '48 (42基础+6活跃市值)',
                        'scoring_mode': 'percentile_ranking',  # 🔥 标识使用百分位排名模式
                        'temporal_weights': {
                            'short_term': 0.3,
                            'medium_term': 0.4,
                            'long_term': 0.3
                        }
                    }

                    return final_score, detailed_info

                except Exception as e:
                    logger.error(f"v3.9.4评分系统错误 {stock_code}: {str(e)}")
                    return 45, {'error': f'系统错误: {str(e)}', 'scoring_method': 'V3.9.4_Production'}

            elif self.scoring_version == "v3.9":
                # 使用v3.9.0生产版评分系统 - 🏆 PRODUCTION A-GRADE MODEL
                try:
                    # 优先使用批量预计算缓存
                    if stock_code in self.v39_batch_cache:
                        result = self.v39_batch_cache[stock_code]
                        logger.debug(f"使用V3.9缓存结果 {stock_code}")
                    else:
                        # 使用V3.9.0的预测接口评估单只股票
                        result = self.scoring_engine_v39.predict_score(stock_code, trade_date)

                    if not result:
                        logger.warning(f"v3.9.0无法获取股票评分 {stock_code}")
                        return 45, {'error': '无法获取评分', 'scoring_method': 'V3.9.0_Production'}

                    # V3.9.0返回的结果格式
                    final_score = result.get('score', 50.0)
                    predicted_return = result.get('predicted_return_5d', 0.0)
                    confidence_score = result.get('confidence', 0.8)
                    recommendation = result.get('recommendation', '观望')

                    # 计算分期评分 (基于5日收益预测)
                    short_term_score = final_score * 1.1  # 短期略高
                    medium_term_score = final_score
                    long_term_score = final_score * 0.9  # 长期略低

                    logger.debug(f"V3.9.0 {stock_code}: 综合评分={final_score:.1f}, 预测5日收益={predicted_return:.2%}, 投资建议={recommendation}")

                    detailed_info = {
                        'final_score': final_score,
                        'confidence_score': confidence_score,
                        'confidence_level': 'high' if confidence_score > 0.7 else 'medium' if confidence_score > 0.4 else 'low',
                        'short_term_score': short_term_score,
                        'medium_term_score': medium_term_score,
                        'long_term_score': long_term_score,
                        'predicted_return_5d': predicted_return,
                        'overall_quality': confidence_score,
                        'quality_score': confidence_score,
                        'risk_level': self._calculate_risk_level_v39(final_score, confidence_score),
                        'recommendation': recommendation,
                        'confidence': 'high' if confidence_score > 0.7 else 'medium' if confidence_score > 0.4 else 'low',
                        'scoring_method': 'V3.9.0_Production_A_Grade',
                        'model_grade': 'A',
                        'model_accuracy': 0.6730,
                        'model_ic': 0.4892,
                        'top20_winrate': 0.95,
                        'temporal_weights': {
                            'short_term': 0.3,
                            'medium_term': 0.4,
                            'long_term': 0.3
                        }
                    }

                    return final_score, detailed_info

                except Exception as e:
                    logger.error(f"v3.9.0评分系统错误 {stock_code}: {str(e)}")
                    return 45, {'error': f'系统错误: {str(e)}', 'scoring_method': 'V3.9.0_Production'}

            elif self.scoring_version == "v3.81":
                # 使用v3.81 Level 4质量评分集成系统 - 🎯 LEVEL 4 QUALITY META-LEARNER
                try:
                    # 🔧 优先使用缓存的V3.81批处理结果，避免single prediction的不一致问题
                    if stock_code in self.v381_batch_cache:
                        predictions = {stock_code: self.v381_batch_cache[stock_code]}
                        logger.debug(f"使用V3.81缓存结果 {stock_code}")
                    else:
                        # 使用V3.81的预测接口评估单只股票 (包含Level 4质量评分)
                        predictions = self.scoring_engine_v381.predict_scores_with_quality([stock_code], trade_date)
                        logger.debug(f"V3.81实时计算 {stock_code}")

                    if not predictions or stock_code not in predictions:
                        logger.warning(f"v3.81无法获取股票评分 {stock_code}")
                        return 45, {'error': '无法获取评分', 'scoring_method': 'V3.81_Level4'}

                    # 处理V3.81的完整预测结果（包含Level 4质量评分）
                    prediction_data = predictions[stock_code]

                    if isinstance(prediction_data, dict):
                        # 新格式：使用V380预测 + Level 4质量评分
                        final_score = prediction_data.get('overall_score', 50.0)
                        short_term_score = prediction_data.get('short_term_score', 50.0)
                        medium_term_score = prediction_data.get('medium_term_score', 50.0)
                        long_term_score = prediction_data.get('long_term_score', 50.0)
                        confidence_score = prediction_data.get('confidence_score', 0.8)
                        # 🎯 Level 4质量评分！
                        quality_score = prediction_data.get('quality_score', 0.5)
                        # 🔧 关键修复：使用V3.81已计算的推荐，不重新计算
                        recommendation = prediction_data.get('recommendation', '观望')
                        confidence_level = prediction_data.get('confidence_level', 'medium')
                    else:
                        # 兼容旧格式
                        final_score = prediction_data if isinstance(prediction_data, (int, float)) else 50.0
                        short_term_score = final_score * 1.1
                        medium_term_score = final_score
                        long_term_score = final_score * 0.9
                        confidence_score = 0.8
                        quality_score = 0.5
                        recommendation = "观望"
                        confidence_level = "medium"

                    # 🔧 不再重新计算投资建议，直接使用V3.81的结果
                    logger.debug(f"V3.81 {stock_code}: 综合评分={final_score}, 投资建议={recommendation}")

                    detailed_info = {
                        'final_score': final_score,
                        'confidence_score': confidence_score,
                        'confidence_level': 'high' if confidence_score > 0.7 else 'medium' if confidence_score > 0.4 else 'low',
                        'short_term_score': short_term_score,
                        'medium_term_score': medium_term_score,
                        'long_term_score': long_term_score,
                        # 🎯 Level 4质量评分作为核心质量指标
                        'overall_quality': quality_score,
                        'quality_score': quality_score,  # 直接使用Level 4评分
                        'risk_level': self._calculate_risk_level_v381(final_score, confidence_score, quality_score),
                        'recommendation': recommendation,
                        'confidence': confidence_level,
                        'scoring_method': 'V3.81_Level4_Quality',
                        'temporal_weights': {
                            'short_term': 0.3,
                            'medium_term': 0.4,
                            'long_term': 0.3
                        },
                        'level4_features': {
                            'quality_differentiation': True,
                            'meta_learning': True,
                            'end_to_end_ml': True
                        }
                    }

                    return final_score, detailed_info

                except Exception as e:
                    logger.error(f"v3.81 Level 4评分系统错误 {stock_code}: {str(e)}")
                    return 45, {'error': f'系统错误: {str(e)}', 'scoring_method': 'V3.81_Level4'}

            elif self.scoring_version == "v3.8":
                # 使用v3.80高级机器学习评分引擎 - 🚀 ADVANCED ML SYSTEM
                try:
                    # 使用V3.80的预测接口评估单只股票
                    predictions = self.scoring_engine_v38.predict_scores([stock_code], trade_date)

                    if not predictions or stock_code not in predictions:
                        logger.warning(f"v3.80无法获取股票评分 {stock_code}")
                        return 45, {'error': '无法获取评分', 'scoring_method': 'V3.80_ML'}

                    # 🔧 修复：处理V3.8的字典格式预测结果
                    prediction_data = predictions[stock_code]

                    if isinstance(prediction_data, dict):
                        # 新格式：使用真实的分期评分
                        final_score = prediction_data.get('overall_score', 50.0)
                        short_term_score = prediction_data.get('short_term_score', 50.0)
                        medium_term_score = prediction_data.get('medium_term_score', 50.0)
                        long_term_score = prediction_data.get('long_term_score', 50.0)
                        confidence_score = prediction_data.get('confidence_score', 0.8)
                    else:
                        # 兼容旧格式
                        final_score = prediction_data if isinstance(prediction_data, (int, float)) else 50.0
                        short_term_score = final_score * 1.1
                        medium_term_score = final_score
                        long_term_score = final_score * 0.9
                        confidence_score = 0.8

                    # 🔧 修复：调整投资建议阈值，更符合实际得分分布
                    if final_score >= 70 and confidence_score > 0.6:
                        recommendation = "强烈买入"
                        confidence = "高"
                    elif final_score >= 60 and confidence_score > 0.4:
                        recommendation = "买入"
                        confidence = "中高"
                    elif final_score >= 50 and confidence_score > 0.3:
                        recommendation = "谨慎买入"
                        confidence = "中"
                    elif final_score >= 40:
                        recommendation = "观望"
                        confidence = "低"
                    elif final_score >= 30:
                        recommendation = "谨慎卖出"
                        confidence = "中"
                    else:
                        recommendation = "卖出"
                        confidence = "高"

                    detailed_info = {
                        'final_score': final_score,
                        'confidence_score': confidence_score,
                        'confidence_level': 'high' if confidence_score > 0.7 else 'medium' if confidence_score > 0.4 else 'low',
                        'short_term_score': short_term_score,    # 🔧 使用真实短期评分
                        'medium_term_score': medium_term_score,  # 🔧 使用真实中期评分
                        'long_term_score': long_term_score,      # 🔧 使用真实长期评分
                        # 🔧 修复：基于多因素计算质量评分和风险等级
                        'overall_quality': self._calculate_quality_score(final_score, confidence_score, prediction_data),
                        'risk_level': self._calculate_risk_level(final_score, confidence_score, prediction_data),
                        'recommendation': recommendation,
                        'confidence': confidence,
                        'scoring_method': 'V3.8_Adaptive',
                        'temporal_weights': {
                            'short_term': 0.3,
                            'medium_term': 0.4,
                            'long_term': 0.3
                        }
                    }

                    return final_score, detailed_info

                except Exception as e:
                    logger.error(f"v3.8自适应评分系统错误 {stock_code}: {str(e)}")
                    return 45, {'error': f'系统错误: {str(e)}', 'scoring_method': 'V3.8_Adaptive'}

            elif self.scoring_version == "v3.7":
                # 使用v3.7高级机器学习评分引擎 - 🚀 ADVANCED ML ENSEMBLE
                try:
                    # 获取股票数据用于机器学习预测
                    stock_data = self._get_stock_data_for_scoring(stock_code, trade_date)
                    if not stock_data:
                        logger.warning(f"v3.7无法获取股票数据 {stock_code}")
                        return 45, {'error': '无法获取股票数据', 'scoring_method': 'V3.7_Advanced_ML'}
                    
                    # 准备特征数据
                    predict_date_obj = datetime.strptime(trade_date, '%Y-%m-%d')
                    feature_date = (predict_date_obj - timedelta(days=1)).strftime('%Y-%m-%d')
                    
                    # 使用V3.7的高级特征提取 (35+维特征)
                    features_df = self.scoring_engine_v37.extract_advanced_features(
                        [stock_code], 
                        feature_date, 
                        feature_date,
                        target_only=True
                    )
                    
                    if features_df is None or len(features_df) == 0:
                        logger.warning(f"v3.7无法提取特征数据 {stock_code}")
                        return 45, {'error': '特征提取失败', 'scoring_method': 'V3.7_Advanced_ML'}
                    
                    # 检查模型是否已训练
                    if 'target_1d' not in self.scoring_engine_v37.base_models or not self.scoring_engine_v37.base_models['target_1d']:
                        logger.info(f"v3.7模型未训练，开始实时训练...")
                        
                        # 获取更多历史数据进行训练
                        training_start_date = (predict_date_obj - timedelta(days=365)).strftime('%Y-%m-%d')
                        
                        # 获取活跃股票列表进行训练
                        with self.scoring_engine_v37.db_manager.get_connection() as conn:
                            active_stocks_query = """
                            SELECT DISTINCT s.code 
                            FROM securities s
                            JOIN daily_quotes dq ON s.id = dq.security_id
                            WHERE s.industry IS NOT NULL 
                            AND dq.trade_date >= ?
                            ORDER BY RANDOM()
                            LIMIT 500
                            """
                            active_stocks = pd.read_sql_query(active_stocks_query, conn, params=(training_start_date,))
                            training_codes = active_stocks['code'].tolist()
                        
                        if len(training_codes) < 50:
                            logger.warning(f"训练数据不足: {len(training_codes)}只股票")
                            return 45, {'error': '训练数据不足', 'scoring_method': 'V3.7_Advanced_ML'}
                        
                        # 提取训练特征
                        logger.info(f"提取训练特征: {len(training_codes)}只股票")
                        training_features = self.scoring_engine_v37.extract_advanced_features(
                            training_codes,
                            training_start_date,
                            feature_date,
                            target_only=False
                        )
                        
                        if training_features is None or len(training_features) < 100:
                            logger.warning(f"训练特征不足: {len(training_features) if training_features is not None else 0}条")
                            return 45, {'error': '训练特征不足', 'scoring_method': 'V3.7_Advanced_ML'}
                        
                        # 构建模型架构
                        self.scoring_engine_v37.build_three_layer_architecture('target_1d')
                        
                        # 准备训练数据
                        training_data, feature_groups = self.scoring_engine_v37.prepare_training_data(
                            training_features, 
                            target_days=[1]
                        )
                        
                        if len(training_data) < 200:
                            logger.warning(f"训练样本不足: {len(training_data)}条")
                            return 45, {'error': '训练样本不足', 'scoring_method': 'V3.7_Advanced_ML'}
                        
                        # 训练三层ensemble模型
                        logger.info(f"开始训练V3.7三层ensemble模型: {len(training_data)}条样本")
                        training_success = self.scoring_engine_v37.train_three_layer_ensemble(
                            training_data, 
                            feature_groups,
                            'target_1d'
                        )
                        
                        if not training_success:
                            logger.error("V3.7模型训练失败")
                            return 45, {'error': '模型训练失败', 'scoring_method': 'V3.7_Advanced_ML'}
                        
                        # 保存训练好的模型
                        model_file = self.scoring_engine_v37.save_models("_realtime_trained")
                        logger.info(f"V3.7模型训练完成并保存: {model_file}")
                    
                    # 使用三层ensemble模型进行评分预测
                    ml_result = self.scoring_engine_v37.predict_three_layer_ensemble(
                        features_df,
                        target_col='target_1d'
                    )

                    if ml_result is None:
                        logger.warning(f"v3.7评分预测失败 {stock_code}")
                        return 45, {'error': '评分预测失败', 'scoring_method': 'V3.7_Advanced_ML'}

                    # 解析结果
                    if isinstance(ml_result, dict):
                        base_score = float(ml_result['score'])
                        factor_scores_v37 = ml_result['factor_scores']
                    else:
                        # 兼容旧格式
                        base_score = float(ml_result[0]) if isinstance(ml_result, (list, tuple)) else float(ml_result)
                        factor_scores_v37 = {'technical': 50.0, 'fundamental': 50.0, 'macro': 50.0, 'sentiment': 50.0, 'temporal': 50.0}
                    
                    # 获取特征重要性用于breakdown
                    feature_importance = self.scoring_engine_v37.feature_importance_history.get('target_1d', {})
                    
                    # 使用V3.7的因子评分
                    factor_scores = factor_scores_v37
                    
                    # 生成投资建议 (V3.7更严格的标准)
                    if base_score >= 85:
                        recommendation = "强烈买入"
                        confidence = "极高"
                    elif base_score >= 75:
                        recommendation = "买入"
                        confidence = "高"
                    elif base_score >= 65:
                        recommendation = "谨慎买入"
                        confidence = "中"
                    elif base_score >= 55:
                        recommendation = "观望"
                        confidence = "中"
                    else:
                        recommendation = "回避"
                        confidence = "低"
                    
                    # 格式化V3.7评分结果
                    scoring_result = {
                        'base_score': base_score,
                        'factor_scores': factor_scores,
                        'recommendation': recommendation,
                        'confidence': confidence,
                        'scoring_method': 'V3.7_Advanced_ML_Ensemble',
                        'model_type': '三层Ensemble(5基础+4专家+Meta)',
                        'features_count': len(features_df.columns) - 2,  # 减去code和trade_date
                        'ensemble_layers': 3
                    }
                    
                    return base_score, scoring_result
                    
                except Exception as e:
                    logger.error(f"v3.7评分系统错误 {stock_code}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return 45, {'error': f'系统错误: {str(e)}', 'scoring_method': 'V3.7_Advanced_ML'}
            
            elif self.scoring_version == "v3.6":
                # 使用v3.6机器学习评分引擎 - 🆕 MACHINE LEARNING
                try:
                    # 获取股票数据用于机器学习预测
                    stock_data = self._get_stock_data_for_scoring(stock_code, trade_date)
                    if not stock_data:
                        logger.warning(f"v3.6无法获取股票数据 {stock_code}")
                        return 45, {'error': '无法获取股票数据', 'scoring_method': 'V3.6_ML'}
                    
                    # 准备特征数据 (使用最近一个交易日的数据进行预测)
                    # trade_date是预测日期（如2025-09-10），我们需要用前一天的数据（2025-09-09）
                    predict_date_obj = datetime.strptime(trade_date, '%Y-%m-%d')
                    feature_date = (predict_date_obj - timedelta(days=1)).strftime('%Y-%m-%d')
                    
                    features_df = self.scoring_engine_v36.extract_features([stock_code], feature_date, feature_date)
                    
                    if features_df is None or len(features_df) == 0:
                        logger.warning(f"v3.6无法提取特征数据 {stock_code}")
                        return 45, {'error': '特征提取失败', 'scoring_method': 'V3.6_ML'}
                    
                    # 使用机器学习模型进行评分预测
                    ml_scores = self.scoring_engine_v36.predict_scores(features_df, target_col='target_1d')
                    
                    if ml_scores is None or len(ml_scores) == 0:
                        logger.warning(f"v3.6评分预测失败 {stock_code}")
                        return 45, {'error': '评分预测失败', 'scoring_method': 'V3.6_ML'}
                    
                    # 取最新的评分
                    base_score = float(ml_scores[-1])
                    
                    # 获取特征重要性用于breakdown
                    feature_importance = self.scoring_engine_v36.feature_importance.get('target_1d')
                    
                    if feature_importance is not None:
                        # 构建详细的因子评分breakdown
                        feature_values = features_df.iloc[-1]
                        factor_scores = {}
                        
                        for _, row in feature_importance.iterrows():
                            feature_name = row['feature']
                            # 计算平均重要性（如果avg_importance列存在则使用，否则计算lgb和xgb的平均值）
                            if 'avg_importance' in row:
                                importance = row['avg_importance']
                            else:
                                importance = (row['lgb_importance'] + row['xgb_importance']) / 2
                            if feature_name in feature_values:
                                raw_value = feature_values[feature_name]
                                # 直接使用原始特征值，不进行人工映射
                                factor_scores[feature_name] = round(raw_value, 4)
                    else:
                        factor_scores = {'ml_prediction': base_score}
                    
                    # 生成投资建议
                    if base_score >= 80:
                        recommendation = "买入"
                        confidence = "高"
                    elif base_score >= 70:
                        recommendation = "谨慎买入"
                        confidence = "中"
                    elif base_score >= 60:
                        recommendation = "观望"
                        confidence = "中"
                    else:
                        recommendation = "回避"
                        confidence = "低"
                    
                    # 格式化V3.6评分结果
                    scoring_result = {
                        'base_score': base_score,
                        'factor_scores': factor_scores,
                        'recommendation': recommendation,
                        'confidence': confidence,
                        'scoring_method': 'V3.6_MachineLearning',
                        'model_type': 'LightGBM+XGBoost_Ensemble',
                        'features_used': list(factor_scores.keys()),
                        'prediction_target': '1日收益率预测'
                    }
                    
                    return base_score, scoring_result
                    
                except Exception as e:
                    logger.error(f"v3.6机器学习评分失败 {stock_code}: {str(e)}")
                    return 45, {'error': f'ML评分失败: {str(e)}', 'scoring_method': 'V3.6_ML'}
            elif self.scoring_version == "v4":
                # 使用v4评分引擎
                scoring_result = self.scoring_engine_v4.calculate_comprehensive_score(stock_code, trade_date)
            elif self.scoring_version == "v3.53":
                # 使用v3.53 多时间周期IC优化评分引擎 - 🆕 MULTI-PERIOD
                try:
                    # 获取股票数据用于评分
                    stock_data = self._get_stock_data_for_scoring(stock_code, trade_date)
                    if not stock_data:
                        return 0, {"quantitative_score": 0, "factor_scores": {}, "recommendation": "数据不足", "scoring_method": "V3.53_MultiPeriod"}
                    
                    # 计算多时间周期评分
                    composite_score, detailed_breakdown = self.scoring_engine_v353_multiperiod.calculate_multi_period_score(stock_data, trade_date, 'composite')
                    
                    # 生成投资建议（基于0-100分制）
                    score = composite_score * 100
                    if score >= 80:
                        recommendation = "买入"
                    elif score >= 70:
                        recommendation = "谨慎买入"
                    elif score >= 60:
                        recommendation = "观望"
                    else:
                        recommendation = "回避"
                    
                    # 格式化结果
                    detailed_info = {
                        "base_score": score,
                        "factor_scores": detailed_breakdown.get('period_details', {}),
                        "recommendation": recommendation,
                        "confidence": "高" if score >= 80 else "中" if score >= 70 else "低",
                        "scoring_method": "V3.53_MultiPeriod_Composite",
                        "period_scores": detailed_breakdown.get('period_scores', {}),
                        "period_weights": detailed_breakdown.get('period_weights', {}),
                        "detailed_scoring": detailed_breakdown
                    }
                    
                    return score, detailed_info
                    
                except Exception as e:
                    logger.error(f"v3.53 多时间周期评分失败 {stock_code}: {e}")
                    return 0, {"quantitative_score": 0, "factor_scores": {}, "recommendation": "评分失败", "scoring_method": "V3.53_MultiPeriod"}
            elif self.scoring_version == "v3.52":
                # 使用v3.5 全面优化评分引擎 - 🆕 COMPREHENSIVE
                try:
                    # 获取股票数据用于评分
                    stock_data = self._get_stock_data_for_scoring(stock_code, trade_date)
                    if not stock_data:
                        logger.warning(f"v3.52无法获取股票数据 {stock_code}")
                        return 45, {'error': '无法获取股票数据'}
                    
                    # 使用全面优化后的评分系统
                    final_score, breakdown = self.scoring_engine_v35_comprehensive.calculate_comprehensive_score(stock_data, trade_date)
                    
                    base_score = final_score
                    factor_scores = breakdown.get('factor_scores', {})
                    
                    # 格式化输出 - 12因子评分
                    scoring_result = {
                        'base_score': base_score,
                        'breakdown': breakdown,
                        'factor_scores': factor_scores,  # 添加顶层factor_scores用于报告生成
                        'report_columns': ['波动', '市值', '动量', 'PB', 'PE', 'RSI6', 'KDJ_K', 'BBI', 'KDJ_D', '知行趋势', '成交量', '知行多均'],
                        'report_values': [
                            round(factor_scores.get('volatility_risk', 0), 1),
                            round(factor_scores.get('market_cap', 0), 1),
                            round(factor_scores.get('price_momentum', 0), 1),
                            round(factor_scores.get('pb', 0), 1),
                            round(factor_scores.get('pe_ttm', 0), 1),
                            round(factor_scores.get('rsi6', 0), 1),
                            round(factor_scores.get('kdj_k', 0), 1),
                            round(factor_scores.get('bbi', 0), 1),
                            round(factor_scores.get('kdj_d', 0), 1),
                            round(factor_scores.get('zhixing_trend', 0), 1),
                            round(factor_scores.get('volume_surge', 0), 1),
                            round(factor_scores.get('zhixing_multiavg', 0), 1)
                        ]
                    }
                    
                    return base_score, scoring_result
                    
                except Exception as e:
                    logger.error(f"v3.52评分失败 {stock_code}: {str(e)}")
                    return 45, {'error': f'评分失败: {str(e)}'}
            elif self.scoring_version == "v3.51":
                # 使用v3.5 Qlib优化评分引擎 - 🆕 OPTIMIZED
                try:
                    # 获取股票数据用于评分
                    stock_data = self._get_stock_data_for_scoring(stock_code, trade_date)
                    if not stock_data:
                        logger.warning(f"v3.51无法获取股票数据 {stock_code}")
                        return 45, {'error': '无法获取股票数据'}
                    
                    # 使用优化后的评分系统
                    final_score, breakdown = self.scoring_engine_v35_optimized.calculate_comprehensive_score(stock_data, trade_date)
                    
                    base_score = final_score
                    factor_scores = breakdown.get('factor_scores', {})
                    
                    # 构建置信度和推荐
                    if base_score >= 75:
                        confidence = "高"
                        recommendation = "强烈推荐"
                    elif base_score >= 65:
                        confidence = "中"
                        recommendation = "推荐"
                    elif base_score >= 55:
                        confidence = "低" 
                        recommendation = "关注"
                    else:
                        confidence = "很低"
                        recommendation = "观望"
                    
                    return base_score, {
                        'factor_scores': factor_scores,
                        'detailed_scores': breakdown,
                        'confidence': confidence,
                        'recommendation': recommendation,
                        '优化评分系统': 'v3.51 - Qlib Phase 2 权重 (+2.88% IC)'
                    }
                    
                except Exception as e:
                    logger.warning(f"v3.51评分计算失败 {stock_code}: {str(e)}")
                    return 45, {'error': str(e)}
                    
            elif self.scoring_version == "v3.5":
                # 使用v3.5知行指标集成评分引擎 - 🆕
                scoring_result = self.scoring_engine_v35.calculate_quantitative_score(stock_code, trade_date)
                
                # v3.5返回的是0-100分的量化评分
                if 'error' in scoring_result:
                    logger.warning(f"v3.5评分计算失败 {stock_code}: {scoring_result['error']}")
                    return 45, {'error': scoring_result['error']}
                
                base_score = scoring_result['quantitative_score']
                factor_scores = {
                    'technical': scoring_result.get('technical_score', 0),
                    'fundamental': scoring_result.get('fundamental_score', 0),
                    'performance': scoring_result.get('performance_score', 0),
                    'market_regime': scoring_result.get('market_regime_score', 0),
                    'zhixing': scoring_result.get('zhixing_score', 0)
                }
                
                # 从评分推导置信度
                if base_score >= 75:
                    confidence = "高"
                elif base_score >= 65:
                    confidence = "中"
                else:
                    confidence = "低"
                
                # 推导投资建议
                if base_score >= 80:
                    recommendation = "强烈推荐"
                elif base_score >= 70:
                    recommendation = "推荐"
                elif base_score >= 60:
                    recommendation = "关注"
                else:
                    recommendation = "观望"
                    
                detailed_scores = {
                    'zhixing_signals': scoring_result.get('zhixing_signals', {}),
                    'zhixing_trend': scoring_result.get('zhixing_trend'),
                    'zhixing_multiavg': scoring_result.get('zhixing_multiavg'),
                    'market_multiplier': scoring_result.get('market_multiplier', 1.0)
                }
                
                # 添加调试信息
                logger.info(f"🔍 v3.5 {stock_code} 知行信号调试:")
                logger.info(f"   原始评分结果zhixing_signals: {scoring_result.get('zhixing_signals', '未找到')}")  
                logger.info(f"   detailed_scores zhixing_signals: {detailed_scores.get('zhixing_signals', '未找到')}")
                signal_in_detailed = detailed_scores.get('zhixing_signals', {}).get('signal_strength', '无')
                logger.info(f"   最终信号强度: {signal_in_detailed}")
                
                return base_score, {
                    'factor_scores': factor_scores,
                    'detailed_scores': detailed_scores,
                    'raw_scoring_data': scoring_result,  # 添加完整的评分结果供后续使用
                    'confidence': confidence,
                    'recommendation': recommendation
                }
            elif self.scoring_version == "v3.3":
                # 使用v3.3相关性优化评分引擎 - 🆕
                scoring_result = self.scoring_engine_v33.calculate_comprehensive_score(stock_code, trade_date)
                
                # v3.3返回的是0-100分的综合评分
                if 'error' in scoring_result:
                    logger.warning(f"v3.3评分计算失败 {stock_code}: {scoring_result['error']}")
                    return 45, {'error': scoring_result['error']}
                
                base_score = scoring_result['comprehensive_score']
                factor_scores = scoring_result['dimension_scores'] 
                detailed_scores = scoring_result.get('detailed_scores', {})
                recommendation = scoring_result['recommendation']
                
                # 从评分推导置信度
                if base_score >= 75:
                    confidence = "高"
                elif base_score >= 65:
                    confidence = "中"
                elif base_score >= 55:
                    confidence = "低"
                else:
                    confidence = "很低"
                
                # 构建详细信息
                detailed_info = {
                    'base_score': base_score,
                    'strategy_bonus': 0,  # v3.3不使用策略加成
                    'final_score': base_score,
                    'recommendation': recommendation,
                    'confidence': confidence,
                    'factor_scores': factor_scores,
                    'factor_weights': {
                        'technical': 0.35,
                        'volume_momentum': 0.25, 
                        'fundamental': 0.20,
                        'sentiment_capital': 0.08,
                        'risk_control': 0.07,
                        'market_environment': 0.05
                    },
                    'detailed_scores': detailed_scores,
                    'industry': scoring_result.get('industry', 'Unknown'),
                    'stock_name': scoring_result.get('stock_name', 'Unknown'),
                    '优化评分系统': 'v3.3 - 相关性深度优化版 (成交量核心+基本面强化)'
                }
                
                return base_score, detailed_info
            elif self.scoring_version == "v3.1":
                # 使用v3.1评分引擎
                scoring_result = self.scoring_engine_v31.calculate_stock_score(stock_code, trade_date)
                
                # v3.1返回的是0-1的分数，但已经包含total_score_100
                base_score = scoring_result.get('total_score_100', scoring_result.get('total_score', 0.5) * 100)
                
                # 从v3.1结果中获取推荐等级
                if base_score >= 80:
                    recommendation = "买入"
                    confidence = "高"
                elif base_score >= 70:
                    recommendation = "谨慎买入"
                    confidence = "中"
                elif base_score >= 60:
                    recommendation = "观望"
                    confidence = "低"
                else:
                    recommendation = "回避"
                    confidence = "低"
                
                # 构建详细信息
                detailed_info = {
                    'base_score': base_score,
                    'strategy_bonus': 0,  # v3.1不使用策略加成
                    'final_score': base_score,
                    'recommendation': recommendation,
                    'confidence': confidence,
                    'factor_scores': scoring_result.get('scores', {}),
                    'factor_weights': self.scoring_engine_v31.get_weight_summary(),
                    'industry': scoring_result.get('details', {}).get('industry', 'Unknown'),
                    'stock_name': scoring_result.get('details', {}).get('stock_name', 'Unknown'),
                    '优化评分系统': 'v3.1 - 相关性分析优化版 (情绪指标+风险控制)'
                }
                
                return base_score, detailed_info
            elif self.scoring_version == "v3.2":
                # 使用v3.2挤压动量增强评分引擎 - 🆕
                scoring_result = self.scoring_engine_v32.calculate_comprehensive_score(stock_code, trade_date)
                
                # v3.2返回的是0-100分的综合评分
                if 'error' in scoring_result:
                    return 50, {'error': scoring_result['error']}
                
                base_score = scoring_result.get('comprehensive_score', 50.0)
                
                # 从v3.2结果中获取推荐等级
                recommendation = scoring_result.get('recommendation', '观望')
                
                if base_score >= 80:
                    confidence = "高"
                elif base_score >= 65:
                    confidence = "中"
                else:
                    confidence = "低"
                
                # 构建详细信息，展示挤压动量因子
                detailed_info = {
                    'base_score': base_score,
                    'strategy_bonus': 0,  # v3.2不使用策略加成
                    'final_score': base_score,
                    'recommendation': recommendation,
                    'confidence': confidence,
                    'dimension_scores': scoring_result.get('dimension_scores', {}),
                    'squeeze_momentum_scores': scoring_result.get('detailed_scores', {}).get('squeeze_momentum', {}),  # 🆕 挤压动量详情
                    'factor_scores': scoring_result.get('dimension_scores', {}),  # 🔧 修复：使用维度评分而不是详细子项评分
                    'stock_name': stock_code,
                    '优化评分系统': 'v3.2 - 挤压动量增强版 (集成v4.0挤压动量因子)'
                }
                
                return base_score, detailed_info
            elif self.scoring_version == "v3.41":
                # 使用v3.41反向工程重构的评分引擎 - 🆕
                scoring_result = self.scoring_engine_v341.calculate_quantitative_score(stock_code, trade_date)
                
                # v3.41返回的是0-100分的量化评分（已反转）
                if 'error' in scoring_result:
                    logger.warning(f"v3.41评分计算失败 {stock_code}: {scoring_result['error']}")
                    return 45, {'error': scoring_result['error']}
                
                base_score = scoring_result['quantitative_score']
                
                # v3.41的评分已经反转：分数越高，投资价值越高！
                # 基于实际分数分布（最高约65分）调整阈值
                if base_score >= 55:
                    recommendation = "买入"
                    confidence = "高"
                elif base_score >= 45:
                    recommendation = "谨慎买入"
                    confidence = "中"
                elif base_score >= 30:
                    recommendation = "观望"
                    confidence = "低"
                else:
                    recommendation = "回避"
                    confidence = "低"
                
                detailed_info = {
                    'base_score': base_score,
                    'recommendation': recommendation,
                    'confidence': confidence,
                    'original_v34_score': scoring_result.get('original_v34_score', 0),
                    'reversed_score': scoring_result.get('reversed_score', 0),
                    'risk_signals': scoring_result.get('risk_signals', {}),
                    'technical_score': scoring_result.get('technical_score', 0),
                    'fundamental_score': scoring_result.get('fundamental_score', 0),
                    'performance_score': scoring_result.get('performance_score', 0),
                    'market_regime_score': scoring_result.get('market_regime_score', 0),
                    'market_regime': scoring_result.get('market_regime', 'neutral'),
                    'market_multiplier': scoring_result.get('market_multiplier', 1.0),
                    'version': scoring_result.get('version', 'v3.41'),
                    'stock_code': stock_code,
                    'stock_name': stock_code,
                    '优化评分系统': 'v3.41 - 反向工程重构版（基于负相关发现的革命性改进）',
                    'factor_scores': {
                        'technical_score': scoring_result.get('technical_score', 0),
                        'fundamental_score': scoring_result.get('fundamental_score', 0),
                        'performance_score': scoring_result.get('performance_score', 0),
                        'market_regime_score': scoring_result.get('market_regime_score', 0)
                    },
                    'reverse_engineering': True
                }
                
                return base_score, detailed_info
            elif self.scoring_version == "v3.4":
                # 使用v3.4基于v3.0优化的评分引擎 - 🆕
                scoring_result = self.scoring_engine_v34.calculate_quantitative_score(stock_code, trade_date)
                
                # v3.4返回的是0-100分的量化评分
                if 'error' in scoring_result:
                    logger.warning(f"v3.4评分计算失败 {stock_code}: {scoring_result['error']}")
                    return 45, {'error': scoring_result['error']}
                
                base_score = scoring_result['quantitative_score']
                
                # 从v3.4结果中获取推荐等级
                if base_score >= 85:
                    recommendation = "买入"
                    confidence = "高"
                elif base_score >= 75:
                    recommendation = "谨慎买入"
                    confidence = "中"
                elif base_score >= 65:
                    recommendation = "观望"
                    confidence = "低"
                else:
                    recommendation = "回避"
                    confidence = "低"
                
                detailed_info = {
                    'base_score': base_score,
                    'recommendation': recommendation,
                    'confidence': confidence,
                    'technical_score': scoring_result.get('technical_score', 0),
                    'fundamental_score': scoring_result.get('fundamental_score', 0),
                    'performance_score': scoring_result.get('performance_score', 0),
                    'market_regime_score': scoring_result.get('market_regime_score', 0),
                    'market_regime': scoring_result.get('market_regime', 'neutral'),
                    'market_multiplier': scoring_result.get('market_multiplier', 1.0),
                    'version': scoring_result.get('version', 'v3.4'),
                    'stock_code': stock_code,
                    'stock_name': stock_code,
                    '优化评分系统': 'v3.4 - 基于v3.0成功经验优化增强版 (新增ROE和营收增长)',
                    'factor_scores': {
                        'technical_score': scoring_result.get('technical_score', 0),
                        'fundamental_score': scoring_result.get('fundamental_score', 0),
                        'performance_score': scoring_result.get('performance_score', 0),
                        'market_regime_score': scoring_result.get('market_regime_score', 0)
                    }
                }
                
                return base_score, detailed_info
            elif self.scoring_version == "v3":
                # 使用v3评分引擎
                scoring_result = self.scoring_engine_v3.calculate_stock_score(stock_code, trade_date)
                
                # v3返回的是0-1的分数，转换为0-100
                base_score = scoring_result.get('total_score', 0.5) * 100
                
                # 从v3结果中获取推荐等级
                if base_score >= 80:
                    recommendation = "买入"
                    confidence = "高"
                elif base_score >= 70:
                    recommendation = "谨慎买入"
                    confidence = "中"
                elif base_score >= 60:
                    recommendation = "观望"
                    confidence = "低"
                else:
                    recommendation = "回避"
                    confidence = "低"
                
                # 构建详细信息
                detailed_info = {
                    'base_score': base_score,
                    'strategy_bonus': 0,  # v3不使用策略加成
                    'final_score': base_score,
                    'recommendation': recommendation,
                    'confidence': confidence,
                    'factor_scores': scoring_result.get('scores', {}),
                    'factor_weights': self.scoring_engine_v3.get_weight_summary(),
                    'industry': scoring_result.get('details', {}).get('industry', 'Unknown'),
                    'stock_name': scoring_result.get('details', {}).get('stock_name', 'Unknown'),
                    '优化评分系统': 'v3.0 - 智能动态权重评分系统'
                }
                
                return base_score, detailed_info
                
            else:
                # 使用v2评分引擎
                scoring_result = self.scoring_engine.score_single_stock(stock_code, trade_date)
                
                # 合并策略信息到评分结果中
                strategy_bonus = self._calculate_strategy_bonus(stock_info)
                base_score = scoring_result.get('composite_score', 50.0)
                
                # 策略加成：多策略选中给予额外加分
                final_score = min(base_score + strategy_bonus, 100.0)
                
                # 返回详细评分信息
                detailed_info = {
                    'base_score': base_score,
                    'strategy_bonus': strategy_bonus,
                    'final_score': final_score,
                    'recommendation': scoring_result.get('recommendation', '观望'),
                    'confidence': scoring_result.get('confidence', '低'),
                    'factor_scores': scoring_result.get('factor_scores', {}),
                    'factor_weights': scoring_result.get('factor_weights', {}),
                    'industry': scoring_result.get('industry', 'Unknown'),
                    'stock_name': scoring_result.get('stock_name', 'Unknown'),
                    '优化评分系统': 'v2.0 - 基于3949只股票实际表现优化'
                }
                
                return final_score, detailed_info
            
        except Exception as e:
            # 处理 stock_info 参数类型
            if isinstance(stock_info, str):
                error_code = stock_info
            else:
                error_code = stock_info.get('stock_code', 'Unknown')
            logger.warning(f"使用优化评分系统失败 {error_code}: {e}")
            # 回退到简化评分逻辑
            return self._fallback_scoring(stock_info), {'error': str(e)}
    
    def _calculate_strategy_bonus(self, stock_info: Dict[str, Any]) -> float:
        """计算策略交集加成分数"""
        strategy_count = stock_info.get('selected_by_strategies', 1)
        strategies = stock_info.get('strategies', [])
        
        # 多策略加成
        if strategy_count >= 4:
            bonus = 15.0  # 四策略强力加成
        elif strategy_count == 3:
            bonus = 10.0  # 三策略优秀加成
        elif strategy_count == 2:
            bonus = 6.0   # 二策略良好加成
        else:
            bonus = 0.0   # 单策略无加成
        
        # SuperB1战法特殊加成
        if 'SuperB1战法' in strategies:
            bonus += 5.0  # SuperB1额外加成
        
        return min(bonus, 20.0)  # 限制最高20分加成
    
    def _fallback_scoring(self, stock_info: Dict[str, Any]) -> float:
        """回退评分逻辑 - 当新评分系统失败时使用"""
        try:
            base_score = 50.0
            
            # 简化的策略评分
            strategy_count = stock_info.get('selected_by_strategies', 1)
            if strategy_count >= 3:
                base_score += 15.0
            elif strategy_count == 2:
                base_score += 8.0
            
            # KDJ简单评分
            kdj_k = stock_info.get('kdj_k', 50)
            kdj_d = stock_info.get('kdj_d', 50)
            if kdj_k > kdj_d and kdj_k < 70:
                base_score += 10.0
            
            # 风险收益比评分
            rr_ratio = stock_info.get('risk_reward_ratio', 2.0)
            if rr_ratio >= 3.0:
                base_score += 10.0
            elif rr_ratio >= 2.5:
                base_score += 5.0
            
            return min(max(base_score, 0), 100.0)
            
        except Exception:
            return 50.0
        
    def analyze_results(self, results: Dict[str, List[str]], data: Dict[str, pd.DataFrame], target_date: pd.Timestamp = None) -> Dict[str, Any]:
        """分析选股结果"""
        # 区分真实策略和虚拟的"全市场ML评分"
        VIRTUAL_STRATEGY = "全市场ML评分"
        real_strategies = {k: v for k, v in results.items() if k != VIRTUAL_STRATEGY}
        has_full_market = VIRTUAL_STRATEGY in results

        analysis = {
            "total_strategies": len(real_strategies),
            "strategy_results": {},
            "strategy_details": real_strategies,  # 只保存真实策略的详细选股结果
            "multi_strategy_stocks": {},
            "top_recommendations": [],
            "all_stocks_with_scores": [],  # 新增：记录所有股票及其评分
            "full_market_mode": has_full_market,
        }

        # 统计每个真实策略的结果
        all_picks = set()
        for strategy, picks in real_strategies.items():
            analysis["strategy_results"][strategy] = len(picks)
            all_picks.update(picks)

        # 全市场模式：加入非策略股票
        if has_full_market:
            full_market_stocks = results[VIRTUAL_STRATEGY]
            all_picks.update(full_market_stocks)
            analysis["strategy_stock_count"] = len(all_picks) - len(full_market_stocks)
            analysis["full_market_extra_count"] = len(full_market_stocks)

        # 找出被多个真实策略选中的股票（不计虚拟策略）
        stock_counts = {}
        for strategy, picks in real_strategies.items():
            for stock in picks:
                stock_counts[stock] = stock_counts.get(stock, 0) + 1
        # 全市场非策略股票计数为0
        if has_full_market:
            for stock in full_market_stocks:
                if stock not in stock_counts:
                    stock_counts[stock] = 0

        # 按被选中次数排序（策略股在前，非策略股在后）
        sorted_stocks = sorted(stock_counts.items(), key=lambda x: x[1], reverse=True)

        # 生成多策略股票统计（只统计真实策略）
        for count in range(len(real_strategies), 0, -1):
            stocks_with_count = [stock for stock, c in sorted_stocks if c == count]
            if stocks_with_count:
                analysis["multi_strategy_stocks"][f"{count}个策略"] = stocks_with_count

        # 生成推荐股票 - 对所有候选股票进行综合评分排序
        all_stocks = [stock for stock, _ in sorted_stocks]  # 获取所有候选股票
        stock_with_scores = []

        # 🚀 V4.4/V4.4.2批量评分预计算
        if hasattr(self, 'scoring_version') and self.scoring_version in ("v4.4", "v4.4.2", "v4.5", "v4.6", "v4.7.1", "v4.7.2", "v4.7.3", "v4.7.4", "v4.7.5", "v4.7.6", "v4.7.7", "v4.7.8", "v4.7.9", "v4.8.0", "v4.8.1", "v4.8.2", "v4.8.4", "v4.8.5", "v4.8.6", "v4.8.7", "v4.8.8", "v4.9.0", "v4.9.0.1", "v4.9.0.2", "v4.9.1", "v5.0") and all_stocks:
            if self.v44_batch_cache:
                logger.info(f"✅ V4.4使用预填充缓存：{len(self.v44_batch_cache)}只股票")
            else:
                try:
                    logger.info(f"🚀 V4.4批量评分预计算：{len(all_stocks)}只股票...")
                    trade_date_str = target_date.strftime('%Y-%m-%d') if hasattr(target_date, 'strftime') else str(target_date)
                    batch_results = self.scoring_engine_v44.predict_scores(all_stocks, trade_date_str)
                    self.v44_batch_cache = batch_results
                    if batch_results:
                        scores = [r.get('score', 0) for r in batch_results.values()]
                        if scores:
                            import numpy as np
                            logger.info(f"✅ V4.4批量预计算完成：{len(batch_results)}只股票，评分范围 {min(scores):.1f}-{max(scores):.1f}")
                            rec_t = getattr(self.scoring_engine_v44, 'recommendation_thresholds', None)
                            if rec_t:
                                logger.info(f"📊 composite推荐阈值(历史校准)：强烈买入≥{rec_t['strong_buy']:.6f}, 买入≥{rec_t['buy']:.6f}")
                            else:
                                logger.info(f"📊 使用composite绝对值fallback阈值（未校准）")
                except Exception as e:
                    logger.warning(f"⚠️ V4.4批量预计算失败，将使用单只评分: {e}")
                    self.v44_batch_cache = {}

        # 🚀 V4.3批量评分预计算
        if hasattr(self, 'scoring_version') and self.scoring_version == "v4.3" and all_stocks:
            if self.v43_batch_cache:
                logger.info(f"✅ V4.3使用预填充缓存：{len(self.v43_batch_cache)}只股票")
            else:
                try:
                    logger.info(f"🚀 V4.3批量评分预计算：{len(all_stocks)}只股票...")
                    trade_date_str = target_date.strftime('%Y-%m-%d') if hasattr(target_date, 'strftime') else str(target_date)
                    batch_results = self.scoring_engine_v43.predict_scores(all_stocks, trade_date_str)
                    self.v43_batch_cache = batch_results
                    if batch_results:
                        scores = [r.get('score', 0) for r in batch_results.values()]
                        if scores:
                            import numpy as np
                            logger.info(f"✅ V4.3批量预计算完成：{len(batch_results)}只股票，评分范围 {min(scores):.1f}-{max(scores):.1f}")
                            rec_t = getattr(self.scoring_engine_v43, 'recommendation_thresholds', None)
                            if rec_t:
                                logger.info(f"📊 composite推荐阈值(历史校准)：强烈买入≥{rec_t['strong_buy']:.6f}, 买入≥{rec_t['buy']:.6f}")
                            else:
                                logger.info(f"📊 使用composite绝对值fallback阈值（未校准）")
                except Exception as e:
                    logger.warning(f"⚠️ V4.3批量预计算失败，将使用单只评分: {e}")
                    self.v43_batch_cache = {}

        # 🔬 V4.0批量cross-sectional评分预计算
        if hasattr(self, 'scoring_version') and self.scoring_version in ("v4.0", "v4.2") and all_stocks:
            if self.v40_batch_cache:
                logger.info(f"✅ V4.0使用预填充缓存：{len(self.v40_batch_cache)}只股票")
            else:
                try:
                    logger.info(f"🔬 V4.0批量cross-sectional评分：{len(all_stocks)}只股票...")
                    trade_date_str = target_date.strftime('%Y-%m-%d') if hasattr(target_date, 'strftime') else str(target_date)
                    batch_results = self.scoring_engine_v40.predict_scores(all_stocks, trade_date_str)
                    self.v40_batch_cache = batch_results
                    if batch_results:
                        scores = [r.get('score', 0) for r in batch_results.values()]
                        if scores:
                            logger.info(f"✅ V4.0批量预计算完成：{len(batch_results)}只股票，评分范围 {min(scores):.1f}-{max(scores):.1f}")
                except Exception as e:
                    logger.warning(f"⚠️ V4.0批量预计算失败，将使用单只评分: {e}")
                    self.v40_batch_cache = {}

        # 🔬 V5.0批量评分预计算
        if hasattr(self, 'scoring_version') and self.scoring_version == "v5.0" and all_stocks:
            if self.v500_batch_cache:
                logger.info(f"✅ V5.0使用预填充缓存：{len(self.v500_batch_cache)}只股票")
            else:
                try:
                    logger.info(f"🔬 V5.0批量评分预计算：{len(all_stocks)}只股票...")
                    trade_date_str = target_date.strftime('%Y-%m-%d') if hasattr(target_date, 'strftime') else str(target_date)
                    batch_results = self.scoring_engine_v500.predict_scores(all_stocks, trade_date_str)
                    self.v500_batch_cache = batch_results
                    if batch_results:
                        scores = [r.get('score', 0) for r in batch_results.values()]
                        if scores:
                            logger.info(f"✅ V5.0批量预计算完成：{len(batch_results)}只股票，评分范围 {min(scores):.1f}-{max(scores):.1f}")
                except Exception as e:
                    logger.warning(f"⚠️ V5.0批量预计算失败，将使用单只评分: {e}")
                    self.v500_batch_cache = {}

        # 🔥 V3.96批量评分预计算
        if hasattr(self, 'scoring_version') and self.scoring_version == "v3.96" and all_stocks:
            if self.v396_batch_cache:
                logger.info(f"✅ V3.96使用预填充缓存：{len(self.v396_batch_cache)}只股票")
            else:
                try:
                    logger.info(f"🚀 V3.96批量评分预计算：{len(all_stocks)}只股票...")
                    trade_date_str = target_date.strftime('%Y-%m-%d') if hasattr(target_date, 'strftime') else str(target_date)
                    batch_results = self.scoring_engine_v396.predict_scores(all_stocks, trade_date_str)
                    self.v396_batch_cache = batch_results
                    if batch_results:
                        scores = [r.get('score', 0) for r in batch_results.values()]
                        if scores:
                            logger.info(f"✅ V3.96批量预计算完成：{len(batch_results)}只股票，评分范围 {min(scores):.1f}-{max(scores):.1f}")
                except Exception as e:
                    logger.warning(f"⚠️ V3.96批量预计算失败，将使用单只评分: {e}")
                    self.v396_batch_cache = {}

        # 🔥 V3.95批量多目标预测预计算
        if hasattr(self, 'scoring_version') and self.scoring_version == "v3.95" and all_stocks:
            if self.v395_batch_cache:
                # 批量模式：缓存已由外部预填充，跳过DB查询
                logger.info(f"✅ V3.95使用预填充缓存：{len(self.v395_batch_cache)}只股票")
            else:
                try:
                    logger.info(f"🚀 V3.95批量多目标预测预计算：{len(all_stocks)}只股票...")
                    trade_date_str = target_date.strftime('%Y-%m-%d') if hasattr(target_date, 'strftime') else str(target_date)

                    # 调用批量预测方法
                    batch_results = self.scoring_engine_v395.predict_scores(all_stocks, trade_date_str)

                    # 缓存结果
                    self.v395_batch_cache = batch_results

                    # 统计信息
                    if batch_results:
                        scores = [r.get('score', 0) for r in batch_results.values()]
                        if scores:
                            logger.info(f"✅ V3.95批量预计算完成：{len(batch_results)}只股票，评分范围 {min(scores):.1f}-{max(scores):.1f}")
                except Exception as e:
                    logger.warning(f"⚠️ V3.95批量预计算失败，将使用单只评分: {e}")
                    self.v395_batch_cache = {}

        # 🔥 V3.9批量评分预计算（批量SQL + 批量predict）
        if hasattr(self, 'scoring_version') and self.scoring_version == "v3.9" and all_stocks:
            if self.v39_batch_cache:
                # 批量模式：缓存已由外部预填充，跳过DB查询
                logger.info(f"✅ V3.9使用预填充缓存：{len(self.v39_batch_cache)}只股票")
            else:
                try:
                    logger.info(f"🔥 V3.9批量评分预计算：{len(all_stocks)}只股票...")
                    trade_date_str = target_date.strftime('%Y-%m-%d') if hasattr(target_date, 'strftime') else str(target_date)

                    batch_results = self.scoring_engine_v39.predict_scores(all_stocks, trade_date_str)

                    self.v39_batch_cache = batch_results

                    if batch_results:
                        scores = [r.get('score', 0) for r in batch_results.values()]
                        if scores:
                            logger.info(f"✅ V3.9批量预计算完成：{len(batch_results)}只股票，评分范围 {min(scores):.1f}-{max(scores):.1f}")
                except Exception as e:
                    logger.warning(f"⚠️ V3.9批量预计算失败，将使用单只评分: {e}")
                    self.v39_batch_cache = {}

        # 🔥 V3.94批量百分位排名预计算（解决评分集中问题）
        if hasattr(self, 'scoring_version') and self.scoring_version == "v3.94" and all_stocks:
            try:
                logger.info(f"🔥 V3.94批量百分位排名预计算：{len(all_stocks)}只股票...")
                trade_date_str = target_date.strftime('%Y-%m-%d') if hasattr(target_date, 'strftime') else str(target_date)

                # 调用批量百分位排名方法
                ranked_results = self.scoring_engine_v394.predict_scores_with_ranking(all_stocks, trade_date_str)

                # 缓存结果
                self.v394_batch_cache = ranked_results

                # 统计信息
                if ranked_results:
                    scores = [r.get('score', 0) for r in ranked_results.values()]
                    if scores:
                        logger.info(f"✅ V3.94批量预计算完成：{len(ranked_results)}只股票，评分范围 {min(scores):.1f}-{max(scores):.1f}")
            except Exception as e:
                logger.warning(f"⚠️ V3.94批量预计算失败，将使用单只评分: {e}")
                self.v394_batch_cache = {}

        logger.info(f"开始计算 {len(all_stocks)} 只股票的综合评分...")
        
        for i, stock in enumerate(all_stocks, 1):
            stock_info = self.get_stock_info(stock, data, target_date)
            if stock_info:
                stock_info["selected_by_strategies"] = stock_counts[stock]
                stock_info["strategies"] = [s for s, picks in real_strategies.items() if stock in picks]

                # 生成投资建议
                investment_rec = self.generate_investment_recommendation(stock_info)
                stock_info.update(investment_rec)

                # 计算composite用于统一排序 (推荐阈值已校准到pred_Xd同一尺度)
                # V4.7.6+: 优先使用scorer返回的rank_score (含consistency_bonus + vol_discount)
                _rank_score = stock_info.get('rank_score')
                if _rank_score is not None:
                    stock_info['composite'] = _rank_score
                else:
                    _p3 = stock_info.get('pred_3d', 0) or 0
                    _p5 = stock_info.get('pred_5d', 0) or 0
                    _p10 = stock_info.get('pred_10d', 0) or 0
                    _p15 = stock_info.get('pred_15d', 0) or 0
                    # V4.6/V4.7.x scorer 使用 0.6*10d + 0.4*15d composite
                    if hasattr(self, 'scoring_version') and self.scoring_version in ('v4.9.0', 'v4.9.1'):
                        # V4.9.0/V4.9.1: Q95 head_rank排序 (score=100→head_rank=1, 71→head_rank=30)
                        hr_score = stock_info.get('final_score', 0)
                        stock_info['composite'] = hr_score if hr_score >= 71 else _p10 * 0.6 + _p15 * 0.4
                    elif hasattr(self, 'scoring_version') and self.scoring_version in (
                        'v4.6', 'v4.7', 'v4.7.1', 'v4.7.2', 'v4.7.3', 'v4.7.4', 'v4.7.5', 'v4.7.6', 'v4.7.7', 'v4.7.8', 'v4.7.9', 'v4.8.0', 'v4.8.1', 'v4.8.2'):
                        stock_info['composite'] = _p10 * 0.6 + _p15 * 0.4
                    else:
                        stock_info['composite'] = _p3 * 0.1 + _p5 * 0.2 + _p10 * 0.4 + _p15 * 0.3

                # 🎯 V4.9.0/V4.9.1: 基于Q95绝对值动态投资建议 (弱市自动减少强买数量)
                if hasattr(self, 'scoring_version') and self.scoring_version in ('v4.9.0', 'v4.9.1'):
                    stock_code = stock_info.get('stock_code', '')
                    if stock_code in self.v44_batch_cache:
                        cached = self.v44_batch_cache[stock_code]
                        stock_info['recommendation'] = cached.get('recommendation', '观望')
                        stock_info['head_rank'] = cached.get('head_rank', 9999)
                        stock_info['in_head_pool'] = cached.get('in_head_pool', False)
                        stock_info['q95_pred_10d'] = cached.get('q95_pred_10d', 0)
                        stock_info['gate_confidence'] = cached.get('gate_confidence')
                        stock_info['gate_regime'] = cached.get('gate_regime', 'normal')
                        # score用head_rank覆盖值(排序用), composite保留真实预测值(显示用)
                        stock_info['score'] = cached.get('score', 0)
                        p10 = cached.get('pred_10d', 0)
                        p15 = cached.get('pred_15d', 0)
                        stock_info['composite'] = 0.6 * p10 + 0.4 * p15

                # 🎯 V4.9.0.1: composite排序, 从scorer获取recommendation和Q95字段
                if hasattr(self, 'scoring_version') and self.scoring_version in ('v4.9.0.1', 'v4.9.0.2'):
                    stock_code = stock_info.get('stock_code', '')
                    if stock_code in self.v44_batch_cache:
                        cached = self.v44_batch_cache[stock_code]
                        stock_info['recommendation'] = cached.get('recommendation', '观望')
                        stock_info['composite'] = cached.get('composite', 0)
                        stock_info['q95_pred_10d'] = cached.get('q95_pred_10d', 0)
                        stock_info['head_rank'] = cached.get('head_rank', 9999)
                        stock_info['in_head_pool'] = cached.get('in_head_pool', False)
                        stock_info['gate_confidence'] = cached.get('gate_confidence')
                        stock_info['gate_regime'] = cached.get('gate_regime', 'normal')

                # 🎯 V3.95: 使用策略驱动预测器重新计算预测收益（基于12,655历史样本统计）
                # 注意：必须在 generate_investment_recommendation 之后执行，因为它会覆盖 predicted_return_5d
                if hasattr(self, 'strategy_return_predictor') and self.scoring_version in ("v3.95", "v3.96", "v4.3"):
                    try:
                        strategies_str = ', '.join(stock_info["strategies"])
                        trade_date_str = target_date.strftime('%Y-%m-%d') if hasattr(target_date, 'strftime') else str(target_date)

                        # 从数据库获取真实的技术特征
                        features = self.strategy_return_predictor._get_features_from_db(stock, trade_date_str)

                        # 调用策略驱动预测器
                        prediction = self.strategy_return_predictor.predict_return(
                            score=stock_info.get('score', 50),
                            strategies=strategies_str,
                            features=features,
                            period='5d'
                        )

                        # 更新预测收益和相关字段（覆盖原来的错误值）
                        stock_info['predicted_return_5d'] = prediction['predicted_return']
                        stock_info['pred_5d'] = prediction['predicted_return']
                        stock_info['confidence_score'] = prediction['confidence']
                        stock_info['win_rate'] = prediction['win_rate']
                        # 不覆盖recommendation: composite历史百分位阈值优先于strategy_return_predictor

                        # 更新 factor_scores 中的相关字段
                        if 'factor_scores' in stock_info:
                            stock_info['factor_scores']['predicted_return_5d'] = prediction['predicted_return']
                            stock_info['factor_scores']['confidence_score'] = prediction['confidence']
                    except Exception as e:
                        logger.debug(f"策略驱动预测器更新失败 {stock}: {e}")

                # ML增强止盈止损目标价 (仅ML版本，需要pred_Xd数据)
                if self.scoring_version not in ('v2', 'v3', 'v3.1', 'v3.2', 'v3.3', 'v3.4', 'v3.41',
                                                  'v3.5', 'v3.51', 'v3.52', 'v3.53', 'v3.6', 'v3.7'):
                    stock_info = self._enhance_prices_with_ml(stock_info)

                stock_with_scores.append(stock_info)
            
            # 进度显示（全市场模式每500只，策略模式每10只）
            progress_interval = 500 if len(all_stocks) > 500 else 10
            if i % progress_interval == 0:
                logger.info(f"已计算 {i}/{len(all_stocks)} 只股票的评分...")

        # V2: 批量信号筛选+仓位分配
        if getattr(self, 'optimizer_version', 'v1') == 'v2' and hasattr(self, 'portfolio_optimizer'):
            env_score = getattr(self, '_cached_env_score', 50.0)
            stock_with_scores = self.portfolio_optimizer.filter_and_allocate(
                stock_with_scores, env_score)

        # 过滤 *ST退市风险股/涨停板/停牌股 (T+1不可买入)
        # 注意: 普通ST保留(有些假ST股质量不错)，只剔除*ST(退市风险警示)
        if stock_with_scores and target_date is not None:
            try:
                trade_date_str = target_date.strftime('%Y-%m-%d') if hasattr(target_date, 'strftime') else str(target_date)
                from data_adapter.database_manager import DatabaseManager
                db = DatabaseManager()

                # 不用IN子句(全市场7000+股票会超SQLite参数限制)，直接查当天全部
                import sqlite3
                conn = sqlite3.connect(str(db.db_path))
                filter_df = pd.read_sql_query("""
                    SELECT s.code, s.name, dq.is_limit_up, dq.is_suspend
                    FROM daily_quotes dq
                    JOIN securities s ON dq.security_id = s.id
                    WHERE dq.trade_date = ?
                """, conn, params=[trade_date_str])
                conn.close()

                if len(filter_df) > 0:
                    # *ST退市风险股 (用securities.name判断，比is_st字段更可靠)
                    star_st_codes = set(filter_df[filter_df['name'].str.contains(r'\*ST', na=False)]['code'])
                    # 涨停/停牌股
                    limit_up_codes = set(filter_df[filter_df['is_limit_up'] == 1]['code'])
                    suspend_codes = set(filter_df[filter_df['is_suspend'] == 1]['code'])

                    def _get_stock_code(s):
                        return s.get('stock_code') or s.get('code') or s.get('stock') or ''

                    before_count = len(stock_with_scores)
                    # *ST股直接从列表中移除(不值得展示)
                    stock_with_scores = [
                        s for s in stock_with_scores
                        if _get_stock_code(s) not in star_st_codes
                    ]
                    # 涨停/停牌股标记为观望
                    for s in stock_with_scores:
                        code = _get_stock_code(s)
                        reasons = []
                        if code in limit_up_codes:
                            reasons.append('涨停')
                        if code in suspend_codes:
                            reasons.append('停牌')
                        if reasons:
                            s['exec_warning'] = '|'.join(reasons)
                            s['recommendation'] = '观望'

                    removed_st = before_count - len(stock_with_scores)
                    marked_count = len([s for s in stock_with_scores if s.get('exec_warning')])
                    if removed_st > 0 or marked_count > 0:
                        logger.info(f"⚠️ 剔除{removed_st}只*ST退市风险股, 标记{marked_count}只涨停/停牌股")
            except Exception as e:
                logger.warning(f"*ST/涨停/停牌过滤失败: {e}")

        # 定义推荐等级权重用于排序
        def get_recommendation_weight(stock):
            rec = stock.get('recommendation', '观望')
            if rec == '强烈买入':
                return 5
            elif rec == '买入':
                return 4
            elif rec == '谨慎买入':
                return 3
            elif rec == '观望':
                return 2
            else:
                return 1

        # 按composite排序 (与recommendation阈值一致，回测验证最可靠)
        # V4.9.0: 用final_score排序 (已被Q95 head_rank覆盖: score=100→head_rank=1)
        def composite_sort_key(x):
            has_error = 'error' in x.get('factor_scores', {}) or x.get('confidence_score', 1) == 0
            if has_error:
                return (0, 0)
            if hasattr(self, 'scoring_version') and self.scoring_version in ('v4.9.0', 'v4.9.1'):
                return (1, x.get('score', 0))
            return (1, x.get('composite', 0))
        stock_with_scores.sort(key=composite_sort_key, reverse=True)
        
        logger.info(f"投资建议生成完成，强烈买入: {len([s for s in stock_with_scores if s.get('recommendation') == '强烈买入'])}, 买入: {len([s for s in stock_with_scores if s.get('recommendation') == '买入'])}")
        
        # 分离多策略股票、单策略股票和全市场非策略股票
        multi_strategy_stocks = [stock for stock in stock_with_scores if stock["selected_by_strategies"] > 1]
        single_strategy_stocks = [stock for stock in stock_with_scores if stock["selected_by_strategies"] == 1]
        no_strategy_stocks = [stock for stock in stock_with_scores if stock["selected_by_strategies"] == 0]

        # 均按composite排序
        multi_strategy_stocks.sort(key=composite_sort_key, reverse=True)
        single_strategy_stocks.sort(key=composite_sort_key, reverse=True)
        no_strategy_stocks.sort(key=composite_sort_key, reverse=True)

        # 标记需要详细分析的股票
        for stock in multi_strategy_stocks:
            stock["needs_detailed_analysis"] = True
            stock["analysis_reason"] = "多策略选中"

        # 只对前20只单策略股票做详细分析
        TOP_SINGLE_STRATEGY_STOCKS = 20
        for i, stock in enumerate(single_strategy_stocks):
            if i < TOP_SINGLE_STRATEGY_STOCKS:
                stock["needs_detailed_analysis"] = True
                stock["analysis_reason"] = f"单策略TOP{i+1}"
            else:
                stock["needs_detailed_analysis"] = False
                stock["analysis_reason"] = "单策略排名靠后"

        # 全市场非策略股票：前10只做详细分析
        TOP_NO_STRATEGY_STOCKS = 10
        for i, stock in enumerate(no_strategy_stocks):
            if i < TOP_NO_STRATEGY_STOCKS:
                stock["needs_detailed_analysis"] = True
                stock["analysis_reason"] = f"全市场ML TOP{i+1}"
            else:
                stock["needs_detailed_analysis"] = False
                stock["analysis_reason"] = "全市场排名靠后"

        # 组合最终推荐：多策略 + 单策略前20 + 全市场前10（用于详细分析）
        detailed_analysis_stocks = (multi_strategy_stocks
                                    + single_strategy_stocks[:TOP_SINGLE_STRATEGY_STOCKS]
                                    + no_strategy_stocks[:TOP_NO_STRATEGY_STOCKS])

        analysis["multi_strategy_recommendations"] = multi_strategy_stocks
        analysis["single_strategy_recommendations"] = single_strategy_stocks
        analysis["no_strategy_recommendations"] = no_strategy_stocks
        analysis["top_recommendations"] = detailed_analysis_stocks  # 只包含需要详细分析的股票
        analysis["total_unique_stocks"] = len(stock_with_scores)
        analysis["all_stock_details"] = stock_with_scores  # 保存所有股票的详细信息
        analysis["all_stocks_with_scores"] = stock_with_scores  # 新增：保存所有股票及其评分
        analysis["detailed_analysis_count"] = len(detailed_analysis_stocks)  # 新增：需要详细分析的股票数量
                
        return analysis
        
    def _generate_stock_detail(self, stock: Dict[str, Any]) -> str:
        """生成单个股票的详细信息"""
        detail = f"""
**基本信息**
- **股票名称**: {stock.get('stock_name', '未知')}
- **股票代码**: {stock['stock_code']}
- **交易所板块**: {stock.get('market', '未知')}
- **股票类型**: {stock.get('stock_type', '未知')}
- **所属行业**: {stock.get('industry', '未知')}
- **注册地**: {stock.get('area', '未知')}
- **上市日期**: {stock.get('list_date', '未知')}

**市场表现**
- **分析日期**: {stock.get('analysis_date', '未知')}
- **收盘价**: {stock['close_price']}元
- **涨跌幅**: {stock['price_change']:+.2f}元 ({stock['price_change_pct']:+.2f}%)
- **成交量**: {stock['volume']:,}手
- **波动率**: {stock['volatility']:.2f}%

**技术指标**
- **KDJ**: K={stock['kdj_k']}, D={stock['kdj_d']}, J={stock['kdj_j']}
- **BBI**: {stock['bbi']}
- **MACD DIF**: {stock['dif']}

**选股策略**
- **通过策略数**: {stock['selected_by_strategies']}个
- **策略名称**: {', '.join(stock['strategies']) if stock['strategies'] else '无（仅ML评分）'}

**分析评价**
- **技术面评价**: {stock.get('technical_rating', '中性')}
- **风险评价**: {stock.get('risk_rating', '中性')}
- **投资建议**: {stock.get('recommendation', '观望')}
- **建议置信度**: {stock.get('confidence', '中性')}

**操作建议**
- **交易制度**: {stock.get('trading_type', 'T+1')}
- **建议操作**: {stock.get('recommendation', '观望')}
- **建议买入价**: {stock.get('suggested_buy_price', 0)}元 
- **建议止损价**: {stock.get('stop_loss_price', 0)}元
- **建议止盈价**: {stock.get('take_profit_price', 0)}元
- **最大风险**: -{stock.get('risk_pct', 0)}% 
- **目标收益**: +{stock.get('reward_pct', 0)}%
- **风险收益比**: 1:{stock.get('risk_reward_ratio', 0)}
- **风险等级**: 中等
- **持仓建议**: 单只股票仓位不超过10%
"""
        return detail

    def _compute_cppi_exposure(self, target_date: pd.Timestamp) -> Dict[str, Any]:
        """计算CPPI Trailing Floor动态仓位建议 (V4.5专用)

        基于沪深300最近60天走势，模拟CPPI overlay:
        - 从60天前NAV=1.0开始，逐日累积收益
        - peak_nav按decay衰减 (0.995/day, half-life ~139d)
        - floor = peak_nav * (1 - cppi_floor)
        - exposure = min(1.0, max(0.05, m * cushion / nav))

        Returns:
            dict with keys: exposure, label, nav, peak_nav, drawdown, details
        """
        try:
            # 加载沪深300最近60天数据
            trade_date_str = target_date.strftime('%Y-%m-%d')
            hs300_df = self.data_loader.load_stock_data_by_code('000300.SH', days=90, target_date=trade_date_str)
            if hs300_df is None or len(hs300_df) < 20:
                logger.warning("CPPI: 无法加载沪深300数据，使用默认exposure=1.0")
                return {'exposure': 1.0, 'label': '正常', 'nav': 1.0, 'peak_nav': 1.0,
                        'drawdown': 0.0, 'details': '数据不足，使用默认仓位'}

            # 取最近60个交易日
            hs300_df = hs300_df.tail(60).copy()
            if len(hs300_df) < 20:
                return {'exposure': 1.0, 'label': '正常', 'nav': 1.0, 'peak_nav': 1.0,
                        'drawdown': 0.0, 'details': '数据不足，使用默认仓位'}

            # 计算日收益率
            closes = hs300_df['close'].values
            daily_returns = np.diff(closes) / closes[:-1]

            # 模拟NAV和peak
            nav = 1.0
            peak_nav = 1.0
            cppi_floor = getattr(self, 'cppi_floor', 0.10)
            cppi_multiplier = getattr(self, 'cppi_multiplier', 10)
            cppi_decay = getattr(self, 'cppi_decay', 0.995)

            for ret in daily_returns:
                nav *= (1 + ret)
                peak_nav = max(nav, peak_nav * cppi_decay)

            # 计算最终exposure
            floor = peak_nav * (1 - cppi_floor)
            cushion = nav - floor
            if cushion <= 0:
                exposure = 0.05
            else:
                exposure = cppi_multiplier * cushion / nav
            exposure = max(0.05, min(1.0, exposure))

            # 当前回撤
            drawdown = (nav / peak_nav - 1) if peak_nav > 0 else 0

            # 仓位标签
            if exposure >= 0.9:
                label = '激进'
            elif exposure >= 0.7:
                label = '正常'
            elif exposure >= 0.4:
                label = '保守'
            else:
                label = '防御'

            details = (f"沪深300近{len(hs300_df)}日NAV={nav:.4f}, "
                       f"峰值={peak_nav:.4f}, 回撤={drawdown:.1%}, "
                       f"CPPI(floor={cppi_floor:.0%},m={cppi_multiplier})")

            return {
                'exposure': round(exposure, 3),
                'label': label,
                'nav': round(nav, 4),
                'peak_nav': round(peak_nav, 4),
                'drawdown': round(drawdown, 4),
                'details': details,
            }
        except Exception as e:
            logger.error(f"CPPI exposure计算失败: {e}")
            return {'exposure': 1.0, 'label': '正常', 'nav': 1.0, 'peak_nav': 1.0,
                    'drawdown': 0.0, 'details': f'计算失败: {e}'}

    def analyze_trading_environment(self, target_date: pd.Timestamp,
                                     all_stocks_with_scores: List[Dict] = None) -> Dict[str, Any]:
        """交易环境监测: 综合大盘趋势/动量/成交量/市场宽度/波动风险/模型信号6个维度"""
        from data_adapter.database_manager import DatabaseManager
        import sqlite3

        db = DatabaseManager()
        date_str = target_date.strftime('%Y-%m-%d') if hasattr(target_date, 'strftime') else str(target_date)

        # 取近250个交易日的指数数据 (MA60 + CPPI需要更长历史)
        start_date = (target_date - timedelta(days=400)).strftime('%Y-%m-%d')

        # --- 1. 获取沪深300数据 ---
        index_codes = ['000300.SH', '000001.SH', '399001.SZ', '399006.SZ', '932000.CSI', '000985.SH']
        index_data = {}
        for code in index_codes:
            df = db.get_security_data(code, start_date, date_str)
            if len(df) > 0:
                index_data[code] = df

        hs300 = index_data.get('000300.SH', pd.DataFrame())
        sh_index = index_data.get('000001.SH', pd.DataFrame())

        results = {
            'ma_position': {'score': 50, 'label': '中性', 'signals': []},
            'breadth': {'score': 50, 'label': '中性', 'signals': []},
            'volume': {'score': 50, 'label': '中性', 'signals': []},
            'growth_value': {'score': 50, 'label': '中性', 'signals': []},
            'model_signal': {'score': 50, 'label': '中性', 'signals': []},
        }

        # ======== MA位置维度 (25%) ========
        # 价格vs均线: 直接反映市场当前涨跌状态 (方向准确率92%)
        if len(hs300) >= 20:
            closes_ma = hs300['close'].values.astype(float)
            latest = closes_ma[-1]
            ma5 = np.mean(closes_ma[-5:])
            ma20 = np.mean(closes_ma[-20:])
            ma60 = np.mean(closes_ma[-60:]) if len(closes_ma) >= 60 else ma20

            ma_score = 50
            ma_signals = []

            # 价格偏离MA5 (连续, ±2%→±12分)
            dev5 = (latest - ma5) / ma5
            ma_score += np.clip(dev5 * 600, -12, 12)

            # 价格偏离MA20 (连续, ±4%→±15分)
            dev20 = (latest - ma20) / ma20
            ma_score += np.clip(dev20 * 375, -15, 15)

            # 价格偏离MA60 (连续, ±8%→±8分)
            dev60 = (latest - ma60) / ma60
            ma_score += np.clip(dev60 * 100, -8, 8)

            # MA斜率 (MA20的5日变化, 连续)
            if len(closes_ma) >= 25:
                ma20_5ago = np.mean(closes_ma[-25:-5])
                slope = (ma20 - ma20_5ago) / ma20_5ago
                ma_score += np.clip(slope * 500, -8, 8)

            # 距前高回撤 (连续)
            peak60 = np.max(closes_ma[-60:]) if len(closes_ma) >= 60 else np.max(closes_ma)
            dd = latest / peak60 - 1
            ma_score += np.clip(dd * 80, -10, 0)

            # 信号描述
            if ma5 > ma20 > ma60:
                ma_signals.append('多头排列')
            elif ma5 < ma20 < ma60:
                ma_signals.append('空头排列')
            else:
                ma_signals.append('均线交织')

            ret5 = closes_ma[-1] / closes_ma[-5] - 1 if len(closes_ma) >= 5 else 0
            ma_signals.append(f'偏离MA20 {dev20:+.1%}')
            ma_signals.append(f'5日{ret5:+.1%}')

            results['ma_position'] = {
                'score': max(0, min(100, ma_score)),
                'signals': ma_signals
            }

        # ======== 模型分化度→信号质量 (不参与评分, 仅显示) ========
        # 全市场ML评分离散度: std高→模型区分度好→Top10选股更可靠 (diff=+2.55%)
        if all_stocks_with_scores and len(all_stocks_with_scores) > 100:
            all_rs = [s.get('composite', s.get('rank_score', 0)) or 0
                      for s in all_stocks_with_scores]
            all_rs = [r for r in all_rs if r != 0]
            if len(all_rs) > 100:
                std_rs = np.std(all_rs)
                disp_score = 50 + np.clip((std_rs - 0.004) * 8000, -30, 30)
                disp_signals = [f'评分离散度σ={std_rs:.4f}']
                if std_rs > 0.006:
                    disp_signals.append('模型分化充分')
                elif std_rs < 0.003:
                    disp_signals.append('模型分化不足')
                results['model_dispersion'] = {
                    'score': max(0, min(100, disp_score)),
                    'signals': disp_signals
                }

        # ======== 模型预测极差维度 (17%) ========
        # P95-P5 spread: 极差大→模型confident→Top10 alpha高 (diff=+2.48%)
        if all_stocks_with_scores and len(all_stocks_with_scores) > 100:
            all_rs2 = [s.get('composite', s.get('rank_score', 0)) or 0
                       for s in all_stocks_with_scores]
            all_rs2 = [r for r in all_rs2 if r != 0]
            if len(all_rs2) > 100:
                p95 = np.percentile(all_rs2, 95)
                p5 = np.percentile(all_rs2, 5)
                spread = p95 - p5
                range_score = 50 + np.clip((spread - 0.015) * 3000, -30, 30)
                range_signals = [f'预测极差P95-P5={spread:.4f}']
                if spread > 0.020:
                    range_signals.append('信号强度高')
                elif spread < 0.010:
                    range_signals.append('信号强度低')
                results['model_range'] = {
                    'score': max(0, min(100, range_score)),
                    'signals': range_signals
                }

        # ======== 成长价值维度 (15%) ========
        # 创业板vs沪深300 10日相对动量: 6年验证 diff=+3.62% (Top10), dir=61.5%
        gem_df = index_data.get('399006.SZ', pd.DataFrame())
        if len(hs300) >= 10 and len(gem_df) >= 10:
            c300_gv = hs300['close'].values.astype(float)
            cgem_gv = gem_df['close'].values.astype(float)
            ret300_10d = c300_gv[-1] / c300_gv[-10] - 1
            retgem_10d = cgem_gv[-1] / cgem_gv[-10] - 1
            gv_spread = retgem_10d - ret300_10d  # 正=创业板领先
            gv_score = 50 + np.clip(gv_spread * 400, -35, 35)
            gv_signals = []
            if gv_spread > 0.03:
                gv_signals.append(f'创业板10日领先沪深300 {gv_spread:+.1%}(风险偏好上升)')
            elif gv_spread < -0.03:
                gv_signals.append(f'沪深300 10日领先创业板 {-gv_spread:+.1%}(避险)')
            else:
                gv_signals.append(f'成长价值10日均衡{gv_spread:+.1%}')
            results['growth_value'] = {
                'score': max(0, min(100, gv_score)),
                'signals': gv_signals
            }

        # ======== 成交量维度 (18%) ========
        # 聚合全A股成交量
        try:
            vol_query = """
                SELECT dq.trade_date, SUM(dq.volume) as total_volume, COUNT(*) as stock_count
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                WHERE s.type = 'A股' AND dq.trade_date >= ? AND dq.trade_date <= ?
                  AND dq.volume > 0
                GROUP BY dq.trade_date
                ORDER BY dq.trade_date
            """
            conn = sqlite3.connect(str(db.db_path))
            vol_df = pd.read_sql_query(vol_query, conn, params=[start_date, date_str])
            conn.close()

            if len(vol_df) >= 5:
                volumes = vol_df['total_volume'].values
                vol_today = volumes[-1]
                vol_ma5 = np.mean(volumes[-5:])
                vol_ma20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else vol_ma5

                vol_ratio_5 = vol_today / vol_ma5 if vol_ma5 > 0 else 1.0
                vol_ratio_20 = vol_today / vol_ma20 if vol_ma20 > 0 else 1.0

                volume_score = 50
                signals = []

                # 量比连续映射 (vs 5日均量, 线性插值提升区分度)
                # vr5分布: P10=0.78, P25=0.88, P50=0.98, P75=1.10, P90=1.28
                vr5_adj = np.clip((vol_ratio_5 - 1.0) * 60, -25, 30)
                volume_score += vr5_adj
                if vol_ratio_5 > 1.3:
                    signals.append(f'量比5日{vol_ratio_5:.2f}(显著放量)')
                elif vol_ratio_5 > 1.1:
                    signals.append(f'量比5日{vol_ratio_5:.2f}(温和放量)')
                elif vol_ratio_5 < 0.75:
                    signals.append(f'量比5日{vol_ratio_5:.2f}(明显缩量)')
                elif vol_ratio_5 < 0.9:
                    signals.append(f'量比5日{vol_ratio_5:.2f}(略缩量)')
                else:
                    signals.append(f'量比5日{vol_ratio_5:.2f}(正常)')

                # 量比vs20日 (中期趋势)
                vr20_adj = np.clip((vol_ratio_20 - 1.0) * 30, -15, 20)
                volume_score += vr20_adj

                # 量能趋势 (连续放量/缩量, 用线性分数)
                if len(volumes) >= 5:
                    vol_trend = 0
                    for i in range(-4, 0):
                        if volumes[i] > volumes[i-1]:
                            vol_trend += 1
                        else:
                            vol_trend -= 1
                    volume_score += vol_trend * 2.5  # -10 ~ +10
                    if vol_trend >= 3:
                        signals.append('连续放量')
                    elif vol_trend <= -3:
                        signals.append('连续缩量')

                # 量价配合 (沪深300涨+放量 vs 跌+放量)
                if len(hs300) >= 1:
                    today_pct = hs300['price_change_pct'].values[-1]
                    if today_pct > 0.005 and vol_ratio_5 > 1.1:
                        volume_score += 8
                        signals.append('量价齐升')
                    elif today_pct < -0.005 and vol_ratio_5 > 1.2:
                        volume_score -= 8
                        signals.append('放量下跌')
                    elif today_pct > 0.005 and vol_ratio_5 < 0.85:
                        volume_score -= 5
                        signals.append('涨但缩量')

                # 总成交额 (亿元, volume单位=手, 粗略估算)
                vol_billion = vol_today / 1e8  # 粗略换算
                signals.append(f'全A成交{vol_billion:.0f}亿手')

                results['volume'] = {
                    'score': max(0, min(100, volume_score)),
                    'signals': signals
                }
        except Exception as e:
            logger.warning(f"成交量维度计算失败: {e}")

        # ======== 市场宽度维度 (15%) ========
        # 查询最近5个交易日宽度数据 (用于当日评分 + 多日清仓预警检测)
        breadth_multi_day = []  # [(date, up_ratio, limit_down), ...]
        try:
            # 涨停从price_change_pct直接计算 (is_limit_up字段数据有问题)
            # 主板≥9.5%, 创/科≥19.5%, 北交≥29.5% (pct已是小数: 0.095=9.5%)
            breadth_start = (target_date - timedelta(days=12)).strftime('%Y-%m-%d')
            breadth_query = """
                SELECT
                    dq.trade_date,
                    COUNT(*) as total,
                    SUM(CASE WHEN dq.price_change_pct > 0 THEN 1 ELSE 0 END) as up_count,
                    SUM(CASE WHEN dq.price_change_pct < 0 THEN 1 ELSE 0 END) as down_count,
                    SUM(CASE WHEN dq.price_change_pct = 0 THEN 1 ELSE 0 END) as flat_count,
                    SUM(CASE WHEN (s.code LIKE '6%' OR s.code LIKE '0%') AND dq.price_change_pct >= 0.095 THEN 1
                             WHEN (s.code LIKE '3%' OR s.code LIKE '688%') AND dq.price_change_pct >= 0.195 THEN 1
                             WHEN (s.code LIKE '8%' OR s.code LIKE '920%') AND dq.price_change_pct >= 0.295 THEN 1
                             ELSE 0 END) as limit_up_count,
                    SUM(CASE WHEN (s.code LIKE '6%' OR s.code LIKE '0%') AND dq.price_change_pct <= -0.095 THEN 1
                             WHEN (s.code LIKE '3%' OR s.code LIKE '688%') AND dq.price_change_pct <= -0.195 THEN 1
                             WHEN (s.code LIKE '8%' OR s.code LIKE '920%') AND dq.price_change_pct <= -0.295 THEN 1
                             ELSE 0 END) as limit_down_count,
                    SUM(CASE WHEN dq.price_change_pct > 0.05 THEN 1 ELSE 0 END) as strong_up,
                    SUM(CASE WHEN dq.price_change_pct < -0.05 THEN 1 ELSE 0 END) as strong_down
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                WHERE s.type = 'A股' AND dq.trade_date >= ? AND dq.trade_date <= ? AND dq.volume > 0
                GROUP BY dq.trade_date
                HAVING COUNT(*) > 1000
                ORDER BY dq.trade_date
            """
            conn = sqlite3.connect(str(db.db_path))
            breadth_df = pd.read_sql_query(breadth_query, conn, params=[breadth_start, date_str])
            conn.close()

            # 保存多日宽度数据 (用于清仓预警)
            for _, row in breadth_df.iterrows():
                r_total = row['total']
                r_up = row['up_count']
                r_ld = row['limit_down_count']
                r_ratio = r_up / r_total if r_total > 0 else 0.5
                breadth_multi_day.append((row['trade_date'], r_ratio, int(r_ld)))

            # 用最后一天(当日)数据做现有评分
            if len(breadth_df) > 0:
                today_row = breadth_df.iloc[-1]
                total = today_row['total']
                up = today_row['up_count']
                down = today_row['down_count']
                limit_up = today_row['limit_up_count']
                limit_down = today_row['limit_down_count']
                strong_up = today_row['strong_up']
                strong_down = today_row['strong_down']

                up_ratio = up / total if total > 0 else 0.5

                breadth_score = 50
                signals = []

                # 涨跌比
                if up_ratio > 0.75:
                    breadth_score += 25
                    signals.append(f'涨跌比{up}:{down}(普涨)')
                elif up_ratio > 0.6:
                    breadth_score += 15
                    signals.append(f'涨跌比{up}:{down}(偏多)')
                elif up_ratio > 0.45:
                    breadth_score += 0
                    signals.append(f'涨跌比{up}:{down}(均衡)')
                elif up_ratio > 0.3:
                    breadth_score -= 15
                    signals.append(f'涨跌比{up}:{down}(偏空)')
                else:
                    breadth_score -= 25
                    signals.append(f'涨跌比{up}:{down}(普跌)')

                # 涨停/跌停
                if limit_up > 50:
                    breadth_score += 10
                    signals.append(f'涨停{limit_up}只(活跃)')
                elif limit_up > 20:
                    breadth_score += 5
                    signals.append(f'涨停{limit_up}只')

                if limit_down > 30:
                    breadth_score -= 15
                    signals.append(f'跌停{limit_down}只(恐慌)')
                elif limit_down > 10:
                    breadth_score -= 5
                    signals.append(f'跌停{limit_down}只')

                # 强势股占比 (>5%)
                strong_ratio = strong_up / total if total > 0 else 0
                if strong_ratio > 0.15:
                    breadth_score += 10
                    signals.append(f'强势股{strong_up}只({strong_ratio:.1%})')
                elif strong_ratio < 0.02:
                    breadth_score -= 5

                results['breadth'] = {
                    'score': max(0, min(100, breadth_score)),
                    'signals': signals
                }
        except Exception as e:
            logger.warning(f"市场宽度维度计算失败: {e}")

        # ======== 清仓预警检测 (三级) ========
        # 基于多日市场宽度崩溃 + 指数回撤的前瞻性风控
        # Tier 0 清仓: 连续2日涨家比<15% + 20d峰值回撤>2% → 15个月0误报
        # Tier 1 严重预警: 当日涨家比<15% + 20d峰值回撤>3%
        # Tier 2 高危预警: 当日涨家比<20% + 20d峰值回撤>3% + 5d跌幅>2%
        crash_warning = {'level': None, 'reasons': [], 'multi_day_breadth': []}
        try:
            if len(hs300) >= 20 and len(breadth_multi_day) >= 2:
                closes_cw = hs300['close'].values.astype(float)

                # 计算关键指标
                peak_20d = np.max(closes_cw[-20:])
                dd_from_peak = closes_cw[-1] / peak_20d - 1  # 20日峰值回撤
                ret_5d = closes_cw[-1] / closes_cw[-5] - 1 if len(closes_cw) >= 5 else 0

                # 最近几天涨家比
                recent_ratios = breadth_multi_day[-5:]  # 最多5天
                today_up_ratio = recent_ratios[-1][1]
                yesterday_up_ratio = recent_ratios[-2][1] if len(recent_ratios) >= 2 else 1.0

                # 保存多日宽度明细 (用于报告展示)
                crash_warning['multi_day_breadth'] = recent_ratios[-5:]
                crash_warning['dd_from_peak'] = dd_from_peak
                crash_warning['ret_5d'] = ret_5d

                # 近5日低涨家比天数 (用于Tier 2)
                days_below_30 = sum(1 for _, r, _ in recent_ratios if r < 0.30)

                # Tier 0: 极端风险 — 连续2日涨家比<15% + 20d峰值回撤>2%
                # 6年回测: 11次触发, ~50%后续反弹(国家队/V型), ~50%继续下跌
                # 定位: 极端风险警告(非清仓指令), 历史上50%概率继续跌、50%急反弹
                if (today_up_ratio < 0.15 and yesterday_up_ratio < 0.15
                        and dd_from_peak < -0.02):
                    crash_warning['level'] = 0
                    crash_warning['reasons'] = [
                        f'连续2日涨家比极低: {yesterday_up_ratio:.1%} → {today_up_ratio:.1%} (阈值<15%)',
                        f'20日峰值回撤: {dd_from_peak:.1%} (阈值<-2%)',
                        f'5日跌幅: {ret_5d:.1%}',
                        f'注意: 6年回测此信号约50%后续反弹，需结合消息面判断',
                    ]
                # Tier 1: 严重预警 — 连续2日宽度恶化(今<20%+昨<25%) + 20d回撤>2%
                # 6年回测: 24次触发, 次日基本持平(+0.05%), 标志极端波动区间
                elif (today_up_ratio < 0.20 and yesterday_up_ratio < 0.25
                        and dd_from_peak < -0.02):
                    crash_warning['level'] = 1
                    crash_warning['reasons'] = [
                        f'连续2日宽度恶化: 昨{yesterday_up_ratio:.1%}(<25%) → 今{today_up_ratio:.1%}(<20%)',
                        f'20日峰值回撤: {dd_from_peak:.1%} (阈值<-2%)',
                    ]
                # Tier 2: 高危预警 — 5日内3天涨家比<30% + 20d回撤>3% + 当日<40%
                # 6年回测: ~70次触发, 次日均-0.11%, 弱预警信号
                elif (days_below_30 >= 3 and dd_from_peak < -0.03
                        and today_up_ratio < 0.40):
                    crash_warning['level'] = 2
                    crash_warning['reasons'] = [
                        f'5日内{days_below_30}天涨家比<30% (阈值>=3天)',
                        f'当日涨家比: {today_up_ratio:.1%} (确认非反弹)',
                        f'20日峰值回撤: {dd_from_peak:.1%} (阈值<-3%)',
                    ]

                if crash_warning['level'] is not None:
                    logger.warning(f"清仓预警触发! Level={crash_warning['level']}, "
                                   f"涨家比={today_up_ratio:.1%}, DD={dd_from_peak:.1%}")
        except Exception as e:
            logger.warning(f"清仓预警检测失败: {e}")

        # ======== 波动/风险维度 (15%) ========
        # 多指数综合: 沪深300(大盘) + 中证2000(小盘) + 中证全指(全市场)
        # 选股偏小盘, 中证2000权重最高
        # 阈值基于2020-2026历史分位校准:
        #   沪深300 P25=13% P50=15.5% P75=20%  中证2000 P25=16% P50=21% P75=27%
        vol_indices = {}
        for idx_code in ['000300.SH', '932000.CSI', '000985.SH']:
            idx_df = index_data.get(idx_code, pd.DataFrame())
            if len(idx_df) >= 20:
                vol_indices[idx_code] = idx_df

        if vol_indices:
            vol_score = 50
            signals = []

            # 加权年化波动率: 沪深300×0.3 + 中证2000×0.4 + 中证全指×0.3
            idx_weights = {'000300.SH': 0.3, '932000.CSI': 0.4, '000985.SH': 0.3}
            weighted_vol = 0
            total_w = 0
            idx_vols = {}
            for idx_code, idx_df in vol_indices.items():
                pcts_idx = idx_df['price_change_pct'].values
                v = np.std(pcts_idx[-20:]) * np.sqrt(250)
                idx_vols[idx_code] = v
                w = idx_weights.get(idx_code, 0.33)
                weighted_vol += w * v
                total_w += w
            if total_w > 0:
                weighted_vol /= total_w

            # 波动率评分 (2019-2026加权分位校准, P10=10% P25=13% P50=15% P75=20% P90=26%)
            if weighted_vol < 0.10:
                vol_score += 22
                signals.append(f'极低波动{weighted_vol:.1%}')
            elif weighted_vol < 0.13:
                vol_score += 12
                signals.append(f'低波动{weighted_vol:.1%}')
            elif weighted_vol < 0.155:
                vol_score += 2
                signals.append(f'正常偏低{weighted_vol:.1%}')
            elif weighted_vol < 0.20:
                vol_score -= 5
                signals.append(f'正常偏高{weighted_vol:.1%}')
            elif weighted_vol < 0.265:
                vol_score -= 15
                signals.append(f'高波动{weighted_vol:.1%}')
            else:
                vol_score -= 25
                signals.append(f'极高波动{weighted_vol:.1%}')

            # 各指数波动差异 (大小盘分化加罚)
            if '000300.SH' in idx_vols and '932000.CSI' in idx_vols:
                vol_spread = idx_vols['932000.CSI'] - idx_vols['000300.SH']
                if vol_spread > 0.15:
                    vol_score -= 5
                    signals.append(f'大小盘波动分化{vol_spread:.1%}')

            # 极端日 (|pct|>2%, 中证全指为主, 比纯小盘更均衡)
            ext_idx = vol_indices.get('000985.SH', vol_indices.get('932000.CSI', vol_indices.get('000300.SH')))
            if ext_idx is not None:
                ext_pcts = ext_idx['price_change_pct'].values
                extreme_days = sum(1 for p in ext_pcts[-10:] if abs(p) > 0.02)
                if extreme_days >= 5:
                    vol_score -= 12
                    signals.append(f'近10日{extreme_days}个极端日(>2%)')
                elif extreme_days >= 3:
                    vol_score -= 5
                    signals.append(f'近10日{extreme_days}个极端日')
                elif extreme_days == 0:
                    vol_score += 8
                    signals.append('近期无极端波动')

            # 波动趋势: 20日vol vs 60日vol (P50=0.93, P75=1.09, P90=1.26)
            ext_idx_for_trend = vol_indices.get('000985.SH', vol_indices.get('000300.SH'))
            if ext_idx_for_trend is not None and len(ext_idx_for_trend) >= 60:
                pcts_trend = ext_idx_for_trend['price_change_pct'].values
                vol_20 = np.std(pcts_trend[-20:]) * np.sqrt(250)
                vol_60 = np.std(pcts_trend[-60:]) * np.sqrt(250)
                if vol_60 > 0:
                    vol_ratio = vol_20 / vol_60
                    if vol_ratio > 1.3:
                        vol_score -= 8
                        signals.append(f'波动率急升(×{vol_ratio:.2f})')
                    elif vol_ratio < 0.75:
                        vol_score += 5
                        signals.append(f'波动率收敛(×{vol_ratio:.2f})')

            # 连续下跌 (用沪深300判断)
            if len(hs300) >= 5:
                main_pcts = hs300['price_change_pct'].values
                consec_down = 0
                for p in reversed(main_pcts):
                    if p < 0:
                        consec_down += 1
                    else:
                        break
                if consec_down >= 5:
                    vol_score -= 12
                    signals.append(f'连跌{consec_down}日')
                elif consec_down >= 3:
                    vol_score -= 5
                    signals.append(f'连跌{consec_down}日')

            results['volatility'] = {
                'score': max(0, min(100, vol_score)),
                'signals': signals
            }

        # ======== 宽度动量维度 (12%) ========
        # 涨家数MA5变化: 市场宽度趋势改善=利好 (回测diff=+1.37%, 满分100)
        try:
            ur_query = """
                SELECT dq.trade_date,
                    CAST(SUM(CASE WHEN dq.price_change_pct > 0 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) as up_ratio
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                WHERE s.type = 'A股' AND dq.trade_date >= ? AND dq.trade_date <= ?
                  AND dq.volume > 0
                GROUP BY dq.trade_date ORDER BY dq.trade_date
            """
            conn_ur = sqlite3.connect(str(db.db_path))
            ur_start = (target_date - timedelta(days=25)).strftime('%Y-%m-%d')
            ur_df = pd.read_sql_query(ur_query, conn_ur, params=[ur_start, date_str])
            conn_ur.close()

            if len(ur_df) >= 8:
                ur_vals = ur_df['up_ratio'].values.astype(float)
                ma5_now = np.mean(ur_vals[-5:]) if len(ur_vals) >= 5 else ur_vals[-1]
                ma5_prev = np.mean(ur_vals[-10:-5]) if len(ur_vals) >= 10 else np.mean(ur_vals[:max(1, len(ur_vals)-5)])
                delta = ma5_now - ma5_prev

                bm_score = 50 + np.clip(delta * 350, -35, 35)
                bm_signals = []
                if delta > 0.05:
                    bm_signals.append(f'涨家数趋势改善(Δ{delta:+.1%})')
                elif delta < -0.05:
                    bm_signals.append(f'涨家数趋势恶化(Δ{delta:+.1%})')
                else:
                    bm_signals.append(f'涨家数趋势平稳(Δ{delta:+.1%})')

                results['breadth_momentum'] = {
                    'score': max(0, min(100, bm_score)),
                    'signals': bm_signals
                }
        except Exception as e:
            logger.warning(f"宽度动量维度计算失败: {e}")

        # ======== 风格动量维度 (10%) ========
        # 大盘领先小盘=利好 (回测验证: diff=+0.90%, 强信号)
        csi2000 = index_data.get('932000.CSI', pd.DataFrame())
        if len(hs300) >= 5 and len(csi2000) >= 5:
            c300 = hs300['close'].values.astype(float)
            c2000 = csi2000['close'].values.astype(float)
            ret300_5d = c300[-1] / c300[-5] - 1
            ret2000_5d = c2000[-1] / c2000[-5] - 1
            spread = ret300_5d - ret2000_5d  # 正=大盘领先

            style_score = 50 + np.clip(spread * 500, -30, 30)
            style_signals = []
            if spread > 0.02:
                style_signals.append(f'大盘领先小盘{spread:+.1%}(利好)')
            elif spread < -0.02:
                style_signals.append(f'小盘领先大盘{-spread:+.1%}(风格切换)')
            else:
                style_signals.append(f'大小盘均衡{spread:+.1%}')

            results['style_momentum'] = {
                'score': max(0, min(100, style_score)),
                'signals': style_signals
            }

        # ======== 模型信号维度 (25%) ========
        if all_stocks_with_scores and len(all_stocks_with_scores) > 0:
            composites = []
            scores_list = []
            q95_values = []
            head_pool_stocks = []
            gate_conf = None
            recs = {'强烈买入': 0, '买入': 0, '谨慎买入': 0, '观望': 0}

            for s in all_stocks_with_scores:
                comp = s.get('composite', s.get('rank_score', None))
                if comp is not None and comp != 0:
                    composites.append(comp)
                sc = s.get('score', 0)
                if sc > 0:
                    scores_list.append(sc)
                # V490/V491 head_rank fields
                q95 = s.get('q95_pred_10d')
                if q95 is not None and q95 != 0:
                    q95_values.append(q95)
                if s.get('in_head_pool'):
                    head_pool_stocks.append(s)
                if gate_conf is None and s.get('gate_confidence') is not None:
                    gate_conf = s.get('gate_confidence')
                rec = s.get('recommendation', '观望')
                if rec in recs:
                    recs[rec] += 1

            signals = []

            # ========================================================
            # 模型信号 = 选股信心×40% + 模型择时×60%
            # 选股信心: Top10 Q95强度 + Head Pool厚度 (截面排名质量)
            # 模型择时: GateV2 + 全市场pred均值 + 强买比例 (方向性判断)
            # ========================================================

            # --- 子维度1: 选股信心 (0-100) ---
            pick_score = 50
            pick_signals = []

            use_headrank = len(q95_values) >= 10
            use_composite = not use_headrank and len(composites) > 100

            if use_headrank:
                arr_q95 = np.array(q95_values)
                top10_q95 = np.mean(sorted(arr_q95)[-10:])
                head_count = len(head_pool_stocks)

                # Top10 Q95强度
                if top10_q95 > 0.17:
                    pick_score += 25
                    pick_signals.append(f'Q95极强({top10_q95:.4f})')
                elif top10_q95 > 0.15:
                    pick_score += 18
                    pick_signals.append(f'Q95强({top10_q95:.4f})')
                elif top10_q95 > 0.13:
                    pick_score += 8
                    pick_signals.append(f'Q95正常({top10_q95:.4f})')
                elif top10_q95 > 0.10:
                    pick_signals.append(f'Q95偏弱({top10_q95:.4f})')
                else:
                    pick_score -= 15
                    pick_signals.append(f'Q95弱({top10_q95:.4f})')

                # Head Pool厚度
                if head_count >= 25:
                    pick_score += 10
                    pick_signals.append(f'头部池{head_count}只')
                elif head_count >= 15:
                    pick_score += 5
                elif head_count > 0:
                    pick_score -= 5
                else:
                    pick_score -= 15

                # Head Pool内Q95均值
                if head_pool_stocks:
                    head_q95s = [v for v in (s.get('q95_pred_10d', 0) for s in head_pool_stocks) if v > 0]
                    if head_q95s and np.mean(head_q95s) > 0.15:
                        pick_score += 5

            elif use_composite:
                arr_sig = np.array(composites)
                top10_avg = np.mean(sorted(arr_sig)[-10:])
                if top10_avg > 0.016:
                    pick_score += 25
                    pick_signals.append(f'Top10极强({top10_avg:.4f})')
                elif top10_avg > 0.012:
                    pick_score += 15
                    pick_signals.append(f'Top10强({top10_avg:.4f})')
                elif top10_avg > 0.009:
                    pick_score += 5
                elif top10_avg > 0.006:
                    pick_score -= 5
                else:
                    pick_score -= 15
                    pick_signals.append(f'Top10弱({top10_avg:.4f})')

            elif scores_list:
                arr_sig = np.array(scores_list)
                top10_avg = np.mean(sorted(arr_sig)[-10:])
                if top10_avg > 90:
                    pick_score += 20
                elif top10_avg > 80:
                    pick_score += 10
                elif top10_avg < 65:
                    pick_score -= 10

            pick_score = max(0, min(100, pick_score))

            # --- 子维度2: 模型择时 (0-100) ---
            timing_score = 50
            timing_signals = []

            # 信号源1: GateV2 confidence
            if gate_conf is not None:
                if gate_conf >= 0.65:
                    timing_score += 20
                    timing_signals.append(f'门控{gate_conf:.0%}(看多)')
                elif gate_conf >= 0.45:
                    timing_signals.append(f'门控{gate_conf:.0%}(中性)')
                else:
                    timing_score -= 20
                    timing_signals.append(f'门控{gate_conf:.0%}(看空)')

            # 信号源2: 全市场pred_10d均值 (负值=模型整体看空)
            if composites:
                market_pred_mean = np.mean(composites)
            elif use_headrank and q95_values:
                market_pred_mean = np.median(q95_values)
            else:
                market_pred_mean = 0
            if market_pred_mean > 0.001:
                timing_score += 15
                timing_signals.append(f'全市场预测偏多({market_pred_mean:.4f})')
            elif market_pred_mean > -0.002:
                timing_signals.append(f'全市场预测中性({market_pred_mean:.4f})')
            else:
                timing_score -= 15
                timing_signals.append(f'全市场预测偏空({market_pred_mean:.4f})')

            # 信号源3: 强买/买入数量 vs 历史均值 (强买~60/天, 买入~140/天)
            strong_buy_n = recs['强烈买入']
            buy_n = recs['买入']
            rec_total = strong_buy_n + buy_n
            # 历史均值: 强买60+买入140=200只/天
            HIST_REC_AVG = 200
            if rec_total > HIST_REC_AVG * 1.2:
                timing_score += 15
                timing_signals.append(f'推荐{rec_total}只(高于均值)')
            elif rec_total > HIST_REC_AVG * 0.8:
                timing_signals.append(f'推荐{rec_total}只(正常)')
            elif rec_total > HIST_REC_AVG * 0.5:
                timing_score -= 10
                timing_signals.append(f'推荐{rec_total}只(偏少)')
            else:
                timing_score -= 15
                timing_signals.append(f'推荐仅{rec_total}只(显著偏少)')

            timing_score = max(0, min(100, timing_score))

            # --- 合并: 选股信心×40% + 模型择时×60% ---
            model_score = int(pick_score * 0.4 + timing_score * 0.6)
            signals = [f'选股{pick_score}'] + pick_signals + [f'择时{timing_score}'] + timing_signals

            results['model_signal'] = {
                'score': max(0, min(100, model_score)),
                'signals': signals
            }

        # ======== 综合评分 ========
        # 5维度: MA位置(方向) + 市场宽度(方向) + 成交量(预测) + 成长价值(预测) + 模型信号(预测)
        # MA位置: 92%方向准确率, 核心方向指标
        # model_dispersion/model_range: 仅展示, 不参与加权评分
        # 权重 (6年全样本前瞻10天验证):
        #   MA位置: 方向描述(dir=92%), 无前瞻预测力 → 25%
        #   市场宽度: 方向描述(dir=60%) → 10%
        #   成交量: Top10预测 diff=+2.39% → 15%
        #   成长价值(10d): Top10预测 diff=+3.62% (最强!) → 15%
        #   模型信号: Top10预测 diff=+1.61% → 35%
        weights = {
            'ma_position': 0.25,
            'breadth': 0.10,
            'volume': 0.15,
            'growth_value': 0.15,
            'model_signal': 0.35,
        }
        total_score = sum(results[k]['score'] * weights[k] for k in weights)
        total_score = round(total_score, 1)
        raw_score = total_score  # 保存原始评分

        # ======== CPPI风控叠加 + 市况熔断 ========
        # 参考: _compute_cppi_exposure() + portfolio_manager._detect_market_regime()
        cppi_info = {}
        if len(hs300) >= 20:
            closes = hs300['close'].values

            # CPPI trailing floor (参数来自回测优化: 小floor + 大multiplier)
            nav = 1.0
            peak_nav = 1.0
            cppi_floor_pct = 0.08   # 8% floor
            cppi_mult = 15          # multiplier
            cppi_decay = 0.997      # ~231天半衰期

            for i in range(1, len(closes)):
                daily_ret = closes[i] / closes[i-1] - 1
                nav *= (1 + daily_ret)
                peak_nav = max(peak_nav * cppi_decay, nav)

            floor_val = peak_nav * (1 - cppi_floor_pct)
            cushion = max(0, nav - floor_val) / nav if nav > 0 else 0
            cppi_exposure = min(1.0, max(0.05, cppi_mult * cushion))
            cppi_dd = nav / peak_nav - 1

            cppi_info = {
                'exposure': cppi_exposure,
                'drawdown': cppi_dd,
                'nav': round(nav, 4),
                'peak_nav': round(peak_nav, 4),
            }

            # 风险预警评分上限 (优先级最高, 在市况熔断之前)
            # 6年回测: 极端宽度崩溃约50%反弹/50%继续跌, 不宜强制清仓
            # 定位: 降低仓位+提示风险, 而非自动清仓
            crash_level = crash_warning.get('level', None)
            if crash_level == 0:
                total_score = min(total_score, 15)   # 极端风险: 建议15%仓位
            elif crash_level == 1:
                total_score = min(total_score, 25)   # 严重预警: 建议20%仓位
            elif crash_level == 2:
                total_score = min(total_score, 35)   # 高危预警: 建议25%仓位

            # 市况熔断 — 仅极端情况硬限 (portfolio_manager circuit breaker)
            ret_20d = closes[-1] / closes[-20] - 1 if len(closes) >= 20 else 0
            ma60_val = np.mean(closes[-60:]) if len(closes) >= 60 else closes[-1]

            if ret_20d < -0.10:
                total_score = min(total_score, 25)  # 严重熊市硬限
            elif ret_20d < -0.05 and closes[-1] < ma60_val:
                total_score = min(total_score, 40)  # 确认熊市硬限

        total_score = round(total_score, 1)

        # 建议仓位: 评分驱动基础仓位 × CPPI动态调节
        # CPPI不改分数, 只调仓位 — 防止熊市反弹中误加仓
        crash_level = crash_warning.get('level', None)
        if crash_level == 0:
            base_pos = 0.15; env_label = '极端风险'; env_emoji = '🚨🚨'
        elif total_score >= 80:
            base_pos = 0.90; env_label = '强势'; env_emoji = '🟢🟢'
        elif total_score >= 60:
            base_pos = 0.65; env_label = '偏多'; env_emoji = '🟢'
        elif total_score >= 40:
            base_pos = 0.40; env_label = '中性'; env_emoji = '🟡'
        elif total_score >= 20:
            base_pos = 0.20; env_label = '偏空'; env_emoji = '🟠'
        else:
            base_pos = 0.05; env_label = '恶劣'; env_emoji = '🔴'

        # 仓位建议 (回测验证: 评分分档直接映射最优, CPPI乘数无增量价值)
        if crash_level == 0:
            position_advice = '5%-15% (极端风险)'
        else:
            position_advice = f'{max(0, base_pos - 0.10):.0%}-{min(1, base_pos + 0.10):.0%}'

        # 为每个维度添加label
        for key in results:
            s = results[key]['score']
            if s >= 70:
                results[key]['label'] = '偏多' if key != 'volatility' else '低风险'
                results[key]['emoji'] = '🟢'
            elif s >= 50:
                results[key]['label'] = '中性' if key != 'volatility' else '正常'
                results[key]['emoji'] = '🟡'
            elif s >= 30:
                results[key]['label'] = '偏空' if key != 'volatility' else '偏高'
                results[key]['emoji'] = '🟠'
            else:
                results[key]['label'] = '弱势' if key != 'volatility' else '高风险'
                results[key]['emoji'] = '🔴'

        return {
            'dimensions': results,
            'total_score': total_score,
            'raw_score': raw_score,
            'position_advice': position_advice,
            'env_label': env_label,
            'env_emoji': env_emoji,
            'cppi': cppi_info,
            'crash_warning': crash_warning,
        }

    def _format_trading_environment(self, env: Dict[str, Any]) -> str:
        """将交易环境监测结果格式化为报告文本"""
        dim_names = {
            'ma_position': 'MA位置',
            'breadth': '市场宽度',
            'volume': '成交量',
            'growth_value': '成长价值',
            'model_signal': '模型信号',
        }
        dim_weights = {
            'ma_position': '25%',
            'breadth': '10%',
            'volume': '15%',
            'growth_value': '15%',
            'model_signal': '35%',
        }

        # 清仓预警横幅 (在所有内容之前, 最醒目位置)
        crash_warning = env.get('crash_warning', {})
        crash_level = crash_warning.get('level', None)

        section = "\n## 🌡️ 交易环境监测\n\n"

        if crash_level is not None:
            if crash_level == 0:
                section += "> ## 🚨🚨🚨 极端风险预警 🚨🚨🚨\n"
                section += "> **连续2日涨家比<15%，市场处于极端状态！建议仓位降至15%以下。**\n"
                section += "> **注意: 此信号历史上约50%后续继续下跌、50%急速反弹(国家队/V型)，需结合消息面判断。**\n>\n"
            elif crash_level == 1:
                section += "> ## ⚠️⚠️ 严重预警 ⚠️⚠️\n"
                section += "> **连续2日市场宽度恶化，建议仓位降至20%以下！**\n>\n"
            elif crash_level == 2:
                section += "> ## ⚠️ 高危预警 ⚠️\n"
                section += "> **近5日市场宽度持续低迷，建议谨慎控制仓位。**\n>\n"

            for reason in crash_warning.get('reasons', []):
                section += f"> - {reason}\n"

            # 近日宽度明细表
            breadth_history = crash_warning.get('multi_day_breadth', [])
            if breadth_history:
                section += ">\n> | 日期 | 涨家比 | 跌停数 |\n"
                section += "> |------|--------|--------|\n"
                for bdate, bratio, bld in breadth_history[-5:]:
                    marker = ' !!!' if bratio < 0.15 else ''
                    section += f"> | {bdate} | {bratio:.1%}{marker} | {bld} |\n"
            section += "\n"

        section += "| 维度 | 权重 | 评分 | 状态 | 关键信号 |\n"
        section += "|------|------|------|------|----------|\n"

        for key in ['ma_position', 'breadth', 'volume', 'growth_value', 'model_signal']:
            d = env['dimensions'].get(key, {'score': 50, 'signals': [], 'emoji': '🟡', 'label': '中性'})
            sig_text = ', '.join(d['signals'][:3]) if d['signals'] else '-'
            section += f"| {dim_names[key]} | {dim_weights[key]} | {round(d['score'])}/100 | {d['emoji']} {d['label']} | {sig_text} |\n"

        section += f"| **综合** | **100%** | **{env['total_score']}/100** | **{env['env_emoji']} {env['env_label']}** | **建议仓位: {env['position_advice']}** |\n"

        # CPPI风控状态
        cppi = env.get('cppi', {})
        raw_score = env.get('raw_score', env['total_score'])
        cppi_triggered = raw_score > env['total_score']

        if cppi_triggered and cppi:
            section += f"\n> 🛡️ **CPPI风控触发**: 原始评分 {raw_score}→{env['total_score']} | "
            section += f"CPPI exposure={cppi.get('exposure', 1):.0%}, 距峰值回撤{cppi.get('drawdown', 0):.1%}\n"

        # 环境解读
        section += f"> 📌 **环境评级: {env['env_emoji']} {env['env_label']}** — 综合评分 {env['total_score']}/100，建议总仓位 {env['position_advice']}\n"

        # 生成操作建议 (清仓预警覆盖常规建议)
        score = env['total_score']
        if crash_level == 0:
            advice = "极端风险！连续2日涨家比<15%，市场处于极端波动区间。建议仓位降至15%以下，严格止损。注意: 历史上此信号约50%后续反弹，需结合政策面/消息面综合判断"
        elif crash_level == 1:
            advice = "严重预警！连续2日市场宽度恶化，建议仓位降至20%以下，仅保留最强确定性品种"
        elif crash_level == 2:
            advice = "高危预警！近5日市场宽度持续低迷，建议控制仓位在25%以下，严格止损"
        elif score >= 80:
            advice = "市场强势运行，可积极参与，关注量能持续性和板块轮动节奏"
        elif score >= 60:
            advice = "环境偏暖，可正常操作，侧重模型高分+策略确认的品种"
        elif score >= 40:
            advice = "环境中性，控制仓位，精选个股，避免追高"
        elif score >= 20:
            advice = "环境偏冷，大幅降低仓位，仅参与强确定性机会"
        else:
            advice = "环境恶劣，建议空仓观望，等待企稳信号"
        section += f"> 💡 **操作建议**: {advice}\n\n"

        return section

    def generate_report(self, analysis: Dict[str, Any], target_date: pd.Timestamp) -> str:
        """生成分析报告"""
        # 跳过周末/节假日，使用下一个交易日
        tomorrow = target_date + timedelta(days=1)
        while tomorrow.weekday() >= 5:  # 5=周六, 6=周日
            tomorrow = tomorrow + timedelta(days=1)
        
        # 根据评分版本调整标题
        version_titles = {
            "v5.0": "V5.0 Unified Fusion版 (v39+v40+neural, 90特征)",
            "v4.7.1": "V4.7.1 信号增强版 (V4.4底座+Bug修复+76特征+LambdaRank+时间衰减)",
            "v4.6": "V4.6 增强版 (ICIR权重+CombinedIsotonic+MetaLearner+增强流动性+小盘加成)",
            "v4.5": "V4.5 CPPI版 (V4.4.1模型+CPPI Trailing Floor动态仓位管理)",
            "v4.3": "V4.3 增强版 (59特征+强正则+Walk-Forward+4目标+等权集成)",
            "v4.2": "V4.2 Hybrid Alpha版 (行业超额+RobustZScore+V39市场特征+5模型Ensemble)",
            "v4.0": "V4.0 Cross-Sectional Alpha版 (超额收益预测+55个截面特征)",
            "v4": "v4.0 挤压动量增强版",
            "v3.96": "V3.96 Robust Z-Score版 (49特征+行业超额标签+全周期ICIR>0.2)",
            "v3.95": "V3.95 多目标预测版 (3d/5d/10d多目标+滚动训练窗口)",
            "v3.94": "V3.94 活跃市值增强版 (48特征=42基础+6活跃市值)",
            "v3.9": "V3.9 生产Ensemble版 (42特征+17财务指标+LGB/XGB/CB/RF四模型)",
            "v3.8": "V3.8 自适应评分版 (动态归一化+多时间维度+置信度评估)",
            "v3.81": "V3.81 Level 4质量评分版 (V380+Level 4 Quality Meta-learner)",
            "v3.7": "V3.7 高级机器学习版 (49特征三层Ensemble)",
            "v3.6": "V3.6 机器学习版 (LightGBM+XGBoost双模型)",
            "v3.53": "v3.53 多时间周期IC优化版",
            "v3.52": "v3.52 全面优化版 (38参数优化)",
            "v3.51": "v3.51 Qlib优化版 (+2.88% IC)",
            "v3.5": "v3.5 知行指标集成版",
            "v3.41": "v3.41 反向工程重构版",
            "v3.4": "v3.4 基于v3.0优化增强版",
            "v3.3": "v3.3 相关性深度优化版",
            "v3.2": "v3.2 挤压动量集成版",
            "v3.1": "v3.1 相关性优化增强版",
            "v3": "v3.0 智能动态权重版",
            "v2": "v2.0 优化版"
        }
        version_title = version_titles.get(self.scoring_version, f"{self.scoring_version} 评分版")
        
        # 为不同版本添加特殊说明
        if hasattr(self, 'scoring_version') and self.scoring_version == "v3.41":
            scoring_explanation = f"""

## 🔄 **v3.41反向评分系统说明**

**✅ 重要说明：v3.41已完成反向工程修正！**

- 🎯 **评分理解**：分数**越高**代表机会**越大**（已修正）
- 📊 **投资指引**：
  - **55分以上** → ✅ **买入推荐**（优质机会）
  - **45-54分** → 🟡 **谨慎买入**（一般机会） 
  - **30-44分** → ⚠️ **观望为主**（机会有限）
  - **30分以下** → 🚫 **建议回避**（风险较高）

- 💡 **核心原理**：基于相关性分析发现，原评分系统存在负相关问题，v3.41通过反向工程完全解决了这个问题
- 📈 **验证结果**：现在高分股票平均收益显著优于低分股票，成功实现正相关

**📋 说明：下表按评分从高到低排序，分数越高投资价值越大**
"""
        elif hasattr(self, 'scoring_version') and self.scoring_version == "v3.7":
            scoring_explanation = f"""

## 🚀 **V3.7高级机器学习评分系统说明**

**🌟 最强版本：三层Ensemble架构 + 49维特征！**

- 🏗️ **三层架构**：
  - **Level 1**: 5个基础模型 (LightGBM, XGBoost, CatBoost, RandomForest, MLP)
  - **Level 2**: 4个专家模型 (技术分析、基本面、宏观、情绪)
  - **Level 3**: Meta学习器 (神经网络)
- 📊 **49维特征**：技术(17) + 基本面(8) + 宏观(8) + 情绪(7) + 时序(5) + 市场环境(4)
- 🎯 **多目标预测**：同时预测1日、3日、5日收益率
- ✅ **性能表现**：
  - **1日预测 R²**: 92.34%
  - **3日预测 R²**: 94.34%
  - **5日预测 R²**: 96.22%
- 💡 **Sigmoid映射**：将原始预测值转换为0-100评分，确保结果可解释性

**📋 说明：采用最先进的机器学习技术，评分越高代表投资价值越大**
"""
        elif hasattr(self, 'scoring_version') and self.scoring_version == "v3.6":
            scoring_explanation = f"""

## 🤖 **V3.6机器学习评分系统说明**

**🚀 革命性升级：采用机器学习非线性建模！**

- 🧠 **核心技术**：LightGBM + XGBoost 双模型ensemble (60%/40%权重)
- 📊 **训练数据**：基于qlib优化结果的12个核心特征，1000+样本训练
- 🎯 **预测目标**：1日未来收益率，告别传统线性权重
- ✅ **模型优势**：
  - **非线性建模**：捕捉特征间复杂交互关系
  - **集成学习**：多模型投票提升预测稳定性
  - **时序交叉验证**：防止数据泄露，确保泛化性能
  - **特征重要性**：自动发现最有价值的预测因子

- 📈 **评分指引**：
  - **80分以上** → 🔥 **强烈买入**（机器学习高置信度）
  - **70-79分** → ✅ **谨慎买入**（模型看好）
  - **60-69分** → 🟡 **观望**（中性偏好）
  - **60分以下** → ⚠️ **回避**（模型不看好）

**🎯 核心理念：让机器学习识别市场规律，实现智能量化选股**
"""
        elif hasattr(self, 'scoring_version') and self.scoring_version in ["v4.3", "v4.4", "v4.4.2", "v4.5", "v4.6", "v4.7.1"]:
            scoring_explanation = """

## 📊 **全局百分位评分说明**

**评分含义**: 分数代表该股票在2020年以来所有历史预测中的全局排名百分位。

- 📈 **评分指引**：
  - **85分以上** → **强烈买入**（历史top 15%信号，极强机会）
  - **70-84分** → **买入**（历史top 30%信号，优质机会）
  - **55-69分** → **谨慎买入**（高于历史中位数，可关注）
  - **40-54分** → **观望**（接近历史中位数，信号一般）
  - **40分以下** → **回避**（低于历史中位数，不建议操作）

- 💡 **与旧版的区别**: 旧版每天都有90分的股票（截面排名），新版只有真正优秀的信号才能获得高分（全局排名）
- 📉 **弱市表现**: 在市场低迷时，可能没有任何股票超过70分 — 这是正确的，说明当天没有历史级别的好机会
"""
        else:
            scoring_explanation = ""

        # V4.5: 计算CPPI仓位建议
        cppi_section = ""
        if self.scoring_version == "v4.5":
            cppi = self._compute_cppi_exposure(target_date)
            exposure_pct = cppi['exposure'] * 100
            top_n = len(analysis.get('top_recommendations', []))
            adjusted_n = max(1, int(top_n * cppi['exposure']))
            cppi_section = f"""
## 🛡️ CPPI动态仓位建议

| 指标 | 数值 |
|------|------|
| **建议仓位** | **{exposure_pct:.0f}%** ({cppi['label']}) |
| 沪深300 NAV | {cppi['nav']:.4f} |
| 峰值 NAV | {cppi['peak_nav']:.4f} |
| 当前回撤 | {cppi['drawdown']:.1%} |
| CPPI参数 | floor={getattr(self, 'cppi_floor', 0.10):.0%}, multiplier={getattr(self, 'cppi_multiplier', 10)} |
| 推荐持仓数 | {adjusted_n}只 (原{top_n}只 x {exposure_pct:.0f}%) |

> **说明**: V4.5使用V4.4.1模型评分 + CPPI Trailing Floor动态仓位管理。
> 当市场接近回撤极限时自动减仓，远离极限时恢复满仓。
> {cppi['details']}
"""

        is_full_market = analysis.get('full_market_mode', False)
        if is_full_market:
            mode_label = "全市场+策略标注"
            strategy_stock_count = analysis.get('strategy_stock_count', 0)
            full_market_extra = analysis.get('full_market_extra_count', 0)
            overview_extra = f"""- **策略选中股票**: {strategy_stock_count}只
- **全市场ML补充**: {full_market_extra}只
- **全市场总股票**: {analysis.get('total_unique_stocks', 0)}只"""
        else:
            mode_label = "策略筛选"
            overview_extra = f"""- **独特股票**: {analysis.get('total_unique_stocks', 0)}只"""

        # 交易环境监测
        env_section = ""
        try:
            stocks_data_for_env = analysis.get('all_stocks_with_scores', [])
            env_result = self.analyze_trading_environment(target_date, stocks_data_for_env)
            env_section = self._format_trading_environment(env_result)
            analysis['trading_environment'] = env_result  # 保存供后续使用
            self._cached_env_score = env_result.get('total_score', 50.0)
        except Exception as e:
            logger.warning(f"交易环境监测失败: {e}")

        report = f"""# 📈 量化选股分析报告 ({version_title})
{scoring_explanation}{cppi_section}{env_section}
## 📊 分析概览
- **分析日期**: {target_date.strftime('%Y-%m-%d')}
- **推荐买入日期**: {tomorrow.strftime('%Y-%m-%d')}
- **运行模式**: {mode_label}
- **分析策略**: {analysis['total_strategies']}个
- **总股票池**: {sum(v.get('count', 0) if isinstance(v, dict) else v for v in analysis['strategy_results'].values())}只(含重复)
{overview_extra}
- **多策略选中**: {len(analysis.get('multi_strategy_recommendations', []))}只
- **单策略选中**: {len(analysis.get('single_strategy_recommendations', []))}只
- **详细分析股票数**: {analysis.get('detailed_analysis_count', 0)}只（多策略全部 + 单策略前20{' + 全市场ML前10' if is_full_market else ''}）
- **推荐股票总数**: {len(analysis.get('top_recommendations', []))}只

## 🎯 各策略筛选结果
"""
        
        for strategy, count in analysis["strategy_results"].items():
            report += f"- **{strategy}**: {count}只股票\n"
        
        report += "\n## 🔄 策略选股交集分析\n\n"
        
        # 分开显示不同数量策略的股票
        multi_strategy_stocks = analysis["multi_strategy_stocks"]
        if isinstance(multi_strategy_stocks, dict):
            # 传统格式：字典格式
            for combo, stocks in multi_strategy_stocks.items():
                report += f"### {combo}选中的股票 ({len(stocks)}只)\n"
                if len(stocks) <= 10:
                    report += f"{', '.join(stocks)}\n\n"
                else:
                    report += f"{', '.join(stocks[:10])}... (共{len(stocks)}只)\n\n"
        else:
            # V3.8格式：列表格式
            if multi_strategy_stocks:
                report += f"### 自适应评分选中的股票 ({len(multi_strategy_stocks)}只)\n"
                if len(multi_strategy_stocks) <= 10:
                    report += f"{', '.join(multi_strategy_stocks)}\n\n"
                else:
                    report += f"{', '.join(multi_strategy_stocks[:10])}... (共{len(multi_strategy_stocks)}只)\n\n"
            else:
                report += "### 无多策略交集股票\n\n"
                
        # 所有股票评分列表（包含策略信息）
        if is_full_market:
            report += "## 📊 全市场股票评分排名\n\n"
        else:
            report += "## 📊 所有选中股票评分排名\n\n"
        # 选择正确的股票数据源
        if hasattr(self, 'scoring_version') and self.scoring_version in ["v3.8", "v3.81"]:
            stocks_data = analysis.get('detailed_stocks', [])
        else:
            stocks_data = analysis.get('all_stocks_with_scores', [])
        if is_full_market:
            strategy_count = len([s for s in stocks_data if s.get('selected_by_strategies', 0) > 0])
            report += f"*全市场 {len(stocks_data)} 只股票ML评分排名，其中 {strategy_count} 只被策略选中（策略列标注具体策略名称，\"-\" 表示仅ML评分）：*\n\n"
        else:
            report += f"*共有 {len(stocks_data)} 只股票被选中，以下显示所有股票的量化评分和策略信息：*\n\n"
        # 根据评分系统版本设置表头
        if hasattr(self, 'scoring_version') and self.scoring_version == "v3.51":
            report += "| 排名 | 股票代码 | 股票名称 | 选中策略 | 量化评分 | 投资建议 | 波动 | 市值 | 动量 | PB | PE | RSI6 | KDJ_K | BBI | KDJ_D | 知行趋势 | 成交量 | 知行多均 |\n"
            report += "|------|----------|----------|----------|----------|----------|------|------|------|------|------|------|------|------|------|------|------|------|\n"
        elif hasattr(self, 'scoring_version') and self.scoring_version == "v3.53":
            report += "| 排名 | 股票代码 | 股票名称 | 选中策略 | 复合评分 | 投资建议 | 1日评分 | 3日评分 | 5日评分 | 10日评分 | 15日评分 | 主要因子 |\n"
            report += "|------|----------|----------|----------|----------|----------|---------|---------|---------|----------|----------|----------|\n"
        elif hasattr(self, 'scoring_version') and self.scoring_version == "v3.52":
            report += "| 排名 | 股票代码 | 股票名称 | 选中策略 | 量化评分 | 投资建议 | 波动 | 市值 | 动量 | PB | PE | RSI6 | KDJ_K | BBI | KDJ_D | 知行趋势 | 成交量 | 知行多均 |\n"
            report += "|------|----------|----------|----------|----------|----------|------|------|------|------|------|------|------|------|------|------|------|------|\n"
        elif hasattr(self, 'scoring_version') and self.scoring_version == "v3.5":
            report += "| 排名 | 股票代码 | 股票名称 | 选中策略 | 量化评分 | 投资建议 | 技术 | 基本 | 表现 | 市场 | 知行 | 知行信号 |\n"
            report += "|------|----------|----------|----------|----------|----------|------|------|------|------|------|----------|\n"
        elif hasattr(self, 'scoring_version') and self.scoring_version in ("v4.9.0", "v4.9.1"):
            # V4.9.0/V4.9.1: Q95 Widen-then-Concentrate
            report += "| 排名 | 股票代码 | 股票名称 | 选中策略 | Q95预测 | 投资建议 | 预测10d | 收盘价 | 买入价 | 止损价 | 目标价 | 仓位 |\n"
            report += "|------|----------|----------|----------|---------|----------|---------|--------|--------|--------|--------|------|\n"
        elif hasattr(self, 'scoring_version') and self.scoring_version in ["v4.4", "v4.4.2", "v4.5", "v4.6", "v4.7.1", "v4.7.2", "v4.7.3", "v4.7.4", "v4.7.5", "v4.7.6", "v4.7.7", "v4.7.8", "v4.7.9", "v4.8.0", "v4.8.1", "v4.8.2", "v4.8.4", "v4.8.5", "v4.8.6", "v4.8.7", "v4.8.8", "v4.9.0.1", "v4.9.0.2", "v5.0"]:
            # V4.4+ 多目标预测 - 按composite排序
            if getattr(self, 'optimizer_version', 'v1') == 'v2':
                report += "| 排名 | 股票代码 | 股票名称 | 选中策略 | Composite | 投资建议 | 预测10d | 收盘价 | 买入价 | 止损价 | 目标价 | 仓位 | 止损% | R:R | ATR% |\n"
                report += "|------|----------|----------|----------|-----------|----------|---------|--------|--------|--------|--------|------|-------|-----|------|\n"
            else:
                report += "| 排名 | 股票代码 | 股票名称 | 选中策略 | Composite | 投资建议 | 预测10d | 收盘价 | 买入价 | 止损价 | 目标价 | 仓位 |\n"
                report += "|------|----------|----------|----------|-----------|----------|---------|--------|--------|--------|--------|------|\n"
        elif hasattr(self, 'scoring_version') and self.scoring_version in ["v3.9", "v3.94", "v3.95", "v3.96", "v4.0", "v4.2", "v4.3"]:
            # 🏆 V3.9.x Production Model - 简化表头
            report += "| 排名 | 股票代码 | 股票名称 | 选中策略 | 综合评分 | 投资建议 | 预测5d | 收盘价 | 买入价 | 止损价 | 目标价 | 仓位 |\n"
            report += "|------|----------|----------|----------|----------|----------|--------|--------|--------|--------|--------|------|\n"
        elif hasattr(self, 'scoring_version') and self.scoring_version in ["v3.8", "v3.81"]:
            # 检查是否为混合模式
            if analysis.get("v38_mixed_mode", False):
                report += "| 排名 | 股票代码 | 股票名称 | 传统策略选中 | 策略数 | 综合评分 | 投资建议 | 置信度 | 短期评分 | 中期评分 | 长期评分 | 风险等级 | 质量评分 |\n"
                report += "|------|----------|----------|--------------|--------|----------|----------|--------|----------|----------|----------|----------|----------|\n"
            else:
                report += "| 排名 | 股票代码 | 股票名称 | 选中策略 | 综合评分 | 投资建议 | 置信度 | 短期评分 | 中期评分 | 长期评分 | 风险等级 | 质量评分 |\n"
                report += "|------|----------|----------|----------|----------|----------|--------|----------|----------|----------|----------|----------|\n"
        elif hasattr(self, 'scoring_version') and self.scoring_version == "v3.7":
            report += "| 排名 | 股票代码 | 股票名称 | 选中策略 | 量化评分 | 投资建议 | 技术 | 基本面 | 宏观 | 情绪 | 时序 |\n"
            report += "|------|----------|----------|----------|----------|----------|------|------|------|------|------|\n"
        elif hasattr(self, 'scoring_version') and self.scoring_version == "v3.6":
            report += "| 排名 | 股票代码 | 股票名称 | 选中策略 | 量化评分 | 投资建议 | BBI | 成交量 | 价格动量 | 知行多均 | RSI | 市值 | KDJ交叉 | PB | 换手率 | 波动风险 | 相对强度 | PE |\n"
            report += "|------|----------|----------|----------|----------|----------|------|------|------|------|------|------|------|------|------|------|------|------|\n"
        elif hasattr(self, 'scoring_version') and self.scoring_version in ["v3.4", "v3.41"]:
            report += "| 排名 | 股票代码 | 股票名称 | 选中策略 | 量化评分 | 投资建议 | 技术 | 基本 | 表现 | 市场 |\n"
            report += "|------|----------|----------|----------|----------|----------|------|------|------|------|\n"
        elif hasattr(self, 'scoring_version') and self.scoring_version == "v3.3":
            report += "| 排名 | 股票代码 | 股票名称 | 选中策略 | 量化评分 | 投资建议 | 技术 | 成交量 | 基本 | 情绪 | 风控 | 市场 |\n"
            report += "|------|----------|----------|----------|----------|----------|------|--------|------|------|------|------|\n"
        elif hasattr(self, 'scoring_version') and self.scoring_version == "v3.2":
            report += "| 排名 | 股票代码 | 股票名称 | 选中策略 | 量化评分 | 投资建议 | 技术 | 挤压 | 基本 | 表现 | 情绪 | 风控 | 市场 |\n"
            report += "|------|----------|----------|----------|----------|----------|------|------|------|------|------|------|------|\n"
        elif hasattr(self, 'scoring_version') and self.scoring_version == "v3.1":
            report += "| 排名 | 股票代码 | 股票名称 | 选中策略 | 量化评分 | 投资建议 | 技术 | 基本 | 表现 | 情绪 | 风控 | 市场 |\n"
            report += "|------|----------|----------|----------|----------|----------|------|------|------|------|------|------|\n"
        else:
            report += "| 排名 | 股票代码 | 股票名称 | 选中策略 | 量化评分 | 投资建议 | 动量 | 回归 | 突破 | 相对 | 稳定 |\n"
            report += "|------|----------|----------|----------|----------|----------|------|------|------|------|------|\n"
        
        # 全市场模式：显示前200 + 所有策略股（即使排名靠后也保留）
        if is_full_market:
            TABLE_TOP_N = 200
            # 构建需要显示的行集合：前200 + 所有策略股
            show_indices = set(range(min(TABLE_TOP_N, len(stocks_data))))
            strategy_indices = []
            for idx, s in enumerate(stocks_data):
                if s.get('selected_by_strategies', 0) > 0 and idx >= TABLE_TOP_N:
                    strategy_indices.append(idx)
                    show_indices.add(idx)
            total_show = len(show_indices)
            if len(stocks_data) > TABLE_TOP_N:
                extra_strategy = len(strategy_indices)
                report += f"*（显示ML评分前 {TABLE_TOP_N} 只 + {extra_strategy} 只策略股，共 {total_show} 行。完整数据见 analysis_data JSON 文件）*\n\n"
        else:
            show_indices = set(range(len(stocks_data)))

        # V4.4+: 按排名覆盖投资建议 (回测验证: top8 alpha最强)
        # V4.9.0: 跳过硬编码覆盖，保留Q95动态推荐（弱市0强买，强市多强买）
        if hasattr(self, 'scoring_version') and self.scoring_version in ["v4.4", "v4.4.2", "v4.5", "v4.6", "v4.7.1", "v4.7.2", "v4.7.3", "v4.7.4", "v4.7.5", "v4.7.6", "v4.7.7", "v4.7.8", "v4.7.9", "v4.8.0", "v4.8.1", "v4.8.2", "v4.8.4", "v4.8.5", "v4.8.6", "v4.8.7", "v4.8.8", "v5.0"]:
            for rank_i, s in enumerate(stocks_data):
                if rank_i < 8:
                    s['recommendation'] = '强烈买入'
                elif rank_i < 20:
                    s['recommendation'] = '买入'
                elif rank_i < 50:
                    s['recommendation'] = '谨慎买入'
                elif rank_i < 200:
                    s['recommendation'] = '观望'
                else:
                    s['recommendation'] = '回避'

        prev_was_gap = False
        for i, stock in enumerate(stocks_data):
            if i not in show_indices:
                if not prev_was_gap and is_full_market:
                    report += f"| ... | ... | *（省略排名 {i+1}-后的非策略股票）* | ... | ... | ... | ... | ... | ... | ... | ... | ... |\n"
                    prev_was_gap = True
                continue
            prev_was_gap = False
            # 处理不同评分版本的字段名差异
            if hasattr(self, 'scoring_version') and self.scoring_version in ["v3.8", "v3.81"]:
                stock_code = stock.get('code', stock.get('stock_code', ''))
                stock_name = stock.get('name', stock.get('stock_name', '未知'))

                # 检查是否为混合模式
                if analysis.get("v38_mixed_mode", False) or analysis.get("v381_mixed_mode", False):
                    # 混合模式：显示传统策略和V3.8/V3.81策略
                    traditional_strategies = stock.get('traditional_strategies', [])
                    strategies = traditional_strategies if traditional_strategies else ['无']
                    strategy_count = stock.get('strategy_count', 0)
                    # 🔧 V3.81混合模式：显示传统策略名称
                    if analysis.get("v381_mixed_mode", False):
                        # V3.81模式下也显示传统策略，不需要特殊处理
                        pass
                else:
                    # 纯V3.8模式
                    strategies = [stock.get('strategy', 'V3.8自适应评分')]
            else:
                stock_code = stock['stock_code']
                stock_name = stock.get('stock_name', '未知')
                strategies = stock.get('strategies', [])
            # 简化策略名称显示
            strategy_names = []
            for s in strategies:
                if '少妇' in s:
                    strategy_names.append('少妇')
                elif 'SuperB1' in s:
                    strategy_names.append('SuperB1')
                elif '补票' in s:
                    strategy_names.append('补票')
                elif 'TePu' in s:
                    strategy_names.append('TePu')
                elif '填坑' in s:
                    strategy_names.append('填坑')
                elif '知行' in s:
                    strategy_names.append('知行')
                elif 'V3.81 Level 4质量评分' in s:
                    strategy_names.append('V3.81')  # 如果出现V3.81评分策略，显示V3.81
                else:
                    strategy_names.append(s[:4])  # 取前4个字符
            strategies_str = ', '.join(strategy_names) if strategy_names else ('-' if is_full_market else '未知')
            # 处理不同评分版本的评分字段差异
            if hasattr(self, 'scoring_version') and self.scoring_version in ["v3.8", "v3.81"]:
                score = stock.get('final_score', 0)  # V3.8使用final_score
                recommendation = stock.get('recommendation', '观望')
            else:
                score = stock.get('score', 0)
                recommendation = stock.get('recommendation', '观望')
            
            # 获取因子评分信息
            factor_scores = stock.get('factor_scores', {})
            
            # 根据评分系统版本处理不同的因子评分
            if hasattr(self, 'scoring_version') and self.scoring_version in ["v4.4", "v4.4.2", "v4.5", "v4.6", "v4.7.1", "v4.7.2", "v4.7.3", "v4.7.4", "v4.7.5", "v4.7.6", "v4.7.7", "v4.7.8", "v4.7.9", "v4.8.0", "v4.8.1", "v4.8.2", "v4.8.4", "v4.8.5", "v4.8.6", "v4.8.7", "v4.8.8", "v4.9.0", "v4.9.0.1", "v4.9.0.2", "v4.9.1", "v5.0"]:
                # V4.4+ 多目标预测字段 (推荐阈值已校准到post-isotonic尺度)
                pred_3d = stock.get('pred_3d', 0.0)
                predicted_return = stock.get('predicted_return_5d', stock.get('pred_5d', 0.0))
                pred_10d = stock.get('pred_10d', 0.0)
                risk_level = stock.get('risk_level', 'medium')
            elif hasattr(self, 'scoring_version') and self.scoring_version in ["v3.9", "v3.94", "v3.95", "v3.96", "v4.0", "v4.2", "v4.3"]:
                # 🏆 V3.9.x Production Model的专用字段
                predicted_return = stock.get('predicted_return_5d', stock.get('pred_5d', 0.0))  # 预测5日收益率
                confidence_score = stock.get('confidence_score', 0.0)   # 置信度
                risk_level = stock.get('risk_level', 'medium')          # 风险等级
            elif hasattr(self, 'scoring_version') and self.scoring_version in ["v3.8", "v3.81"]:
                # v3.8自适应评分系统的专用字段
                confidence_score = stock.get('confidence_score', 0.0)
                confidence_level = stock.get('confidence_level', 'unknown')
                short_term_score = stock.get('short_term_score', 50)
                medium_term_score = stock.get('medium_term_score', 50)
                long_term_score = stock.get('long_term_score', 50)
                risk_level = stock.get('risk_level', 'medium')
                overall_quality = stock.get('overall_quality', 0.5)
            elif hasattr(self, 'scoring_version') and self.scoring_version == "v3.5":
                # v3.5评分系统的5个维度因子（集成知行指标）
                technical_score = factor_scores.get('technical', 0)  # 技术指标
                fundamental_score = factor_scores.get('fundamental', 0)  # 基本面
                performance_score = factor_scores.get('performance', 0)  # 市场表现
                market_regime_score = factor_scores.get('market_regime', 0)  # 市场环境
                zhixing_score = factor_scores.get('zhixing', 0)  # 知行指标
                
                # 获取知行信号信息 - 修复数据传递问题
                detailed_scoring = stock.get('detailed_scoring', {})
                detailed_scores = detailed_scoring.get('detailed_scores', {})
                zhixing_signals = detailed_scores.get('zhixing_signals', {})
                
                # 如果还是没有找到，尝试从raw_scoring_data中获取
                if not zhixing_signals:
                    raw_data = stock.get('raw_scoring_data', {})
                    zhixing_signals = raw_data.get('zhixing_signals', {})
                
                signal_strength = zhixing_signals.get('signal_strength', '无信号')
                
                # v3.5知行信号显示修复完成
            elif hasattr(self, 'scoring_version') and self.scoring_version in ["v3.4", "v3.41"]:
                # v3.4评分系统的4个维度因子（基于v3.0优化）
                technical_score = factor_scores.get('technical_score', 0)  # 技术指标
                fundamental_score = factor_scores.get('fundamental_score', 0)  # 基本面（增强版含ROE和营收增长）
                performance_score = factor_scores.get('performance_score', 0)  # 市场表现
                market_regime_score = factor_scores.get('market_regime_score', 0)  # 市场环境
            elif hasattr(self, 'scoring_version') and self.scoring_version == "v3.3":
                # v3.3评分系统的6个维度因子
                technical_score = factor_scores.get('technical', 0)  # 技术指标
                volume_momentum_score = factor_scores.get('volume_momentum', 0)  # 成交量动量
                fundamental_score = factor_scores.get('fundamental', 0)  # 基本面
                sentiment_capital_score = factor_scores.get('sentiment_capital', 0)  # 情绪资金
                risk_control_score = factor_scores.get('risk_control', 0)  # 风险控制
                market_environment_score = factor_scores.get('market_environment', 0)  # 市场环境
            elif hasattr(self, 'scoring_version') and self.scoring_version == "v3.2":
                # v3.2评分系统的7个维度因子
                technical_score = factor_scores.get('technical', 0)  # 技术指标
                squeeze_score = factor_scores.get('squeeze_momentum', 0)  # 挤压动量
                fundamental_score = factor_scores.get('fundamental', 0)  # 基本面
                performance_score = factor_scores.get('performance', 0)  # 市场表现
                sentiment_score = factor_scores.get('sentiment', 0)  # 情绪指标
                risk_control_score = factor_scores.get('risk_control', 0)  # 风险控制
                market_regime_score = factor_scores.get('market_regime', 0)  # 市场环境
            elif hasattr(self, 'scoring_version') and self.scoring_version == "v3.51":
                # v3.51评分系统的12个因子 - Qlib优化权重版本
                volatility_risk_score = factor_scores.get('volatility_risk', 50)       # 15.32% - 波动风险
                market_cap_score = factor_scores.get('market_cap', 50)                 # 14.20% - 市值
                price_momentum_score = factor_scores.get('price_momentum', 50)         # 13.10% - 价格动量
                pb_score = factor_scores.get('pb', 50)                                 # 12.13% - PB估值
                pe_ttm_score = factor_scores.get('pe_ttm', 50)                         # 8.75% - PE估值
                rsi6_score = factor_scores.get('rsi6', 50)                             # 8.39% - RSI6
                kdj_k_score = factor_scores.get('kdj_k', 50)                           # 6.61% - KDJ_K
                bbi_score = factor_scores.get('bbi', 50)                               # 5.66% - BBI
                kdj_d_score = factor_scores.get('kdj_d', 50)                           # 5.29% - KDJ_D
                zhixing_trend_score = factor_scores.get('zhixing_trend', 50)           # 4.78% - 知行趋势 (降权)
                volume_surge_score = factor_scores.get('volume_surge', 50)             # 3.15% - 成交量突破
                zhixing_multiavg_score = factor_scores.get('zhixing_multiavg', 50)     # 2.62% - 知行多均线 (降权)
            elif hasattr(self, 'scoring_version') and self.scoring_version == "v3.53":
                # v3.53多时间周期评分系统 - 🆕 MULTI-PERIOD
                # 从stock数据中获取详细评分信息
                detailed_scoring = stock.get('detailed_scoring', {})
                period_scores = detailed_scoring.get('period_scores', {})
                score_1d = period_scores.get('1d', 0) * 100
                score_3d = period_scores.get('3d', 0) * 100  
                score_5d = period_scores.get('5d', 0) * 100
                score_10d = period_scores.get('10d', 0) * 100
                score_15d = period_scores.get('15d', 0) * 100
                
                # 找出主要贡献因子
                period_details = detailed_scoring.get('factor_scores', {})
                main_factors = []
                if period_details:
                    # 从各周期中找出权重最高的因子
                    for period, details in period_details.items():
                        if isinstance(details, dict) and 'factor_contributions' in details:
                            contributions = details['factor_contributions']
                            if contributions:
                                top_factor = max(contributions.items(), key=lambda x: abs(x[1]))
                                main_factors.append(f"{period}:{top_factor[0]}")
                
                main_factors_str = ", ".join(main_factors[:3]) if main_factors else "复合因子"
            elif hasattr(self, 'scoring_version') and self.scoring_version == "v3.52":
                # v3.52评分系统的12个因子 - 全面优化版本
                volatility_risk_score = factor_scores.get('volatility_risk', 50)       # 波动风险
                market_cap_score = factor_scores.get('market_cap', 50)                 # 市值
                price_momentum_score = factor_scores.get('price_momentum', 50)         # 价格动量
                pb_score = factor_scores.get('pb', 50)                                 # PB估值
                pe_ttm_score = factor_scores.get('pe_ttm', 50)                         # PE估值
                rsi6_score = factor_scores.get('rsi6', 50)                             # RSI6
                kdj_k_score = factor_scores.get('kdj_k', 50)                           # KDJ_K
                bbi_score = factor_scores.get('bbi', 50)                               # BBI
                kdj_d_score = factor_scores.get('kdj_d', 50)                           # KDJ_D
                zhixing_trend_score = factor_scores.get('zhixing_trend', 50)           # 知行趋势
                volume_surge_score = factor_scores.get('volume_surge', 50)             # 成交量突破
                zhixing_multiavg_score = factor_scores.get('zhixing_multiavg', 50)     # 知行多均线
            elif hasattr(self, 'scoring_version') and self.scoring_version == "v3.7":
                # v3.7高级机器学习评分系统的5个维度因子（从49维特征聚合）
                technical_score = factor_scores.get('technical', 50)  # 技术分析因子(17维)
                fundamental_score = factor_scores.get('fundamental', 50)  # 基本面因子(8维)
                macro_score = factor_scores.get('macro', 50)  # 宏观因子(8维)
                sentiment_score = factor_scores.get('sentiment', 50)  # 情绪因子(7维)
                temporal_score = factor_scores.get('temporal', 50)  # 时序因子(5维)
            elif hasattr(self, 'scoring_version') and self.scoring_version == "v3.1":
                # v3.1评分系统的因子
                momentum = factor_scores.get('technical', 50) * 100  # 技术指标
                mean_reversion = factor_scores.get('fundamental', 50) * 100  # 基本面
                volume_breakout = factor_scores.get('performance', 50) * 100  # 市场表现
                relative_performance = factor_scores.get('sentiment', 50) * 100  # 情绪指标
                stability = factor_scores.get('risk_control', 50) * 100  # 风险控制
                market_regime_score = factor_scores.get('market_regime', 50) * 100  # 市场环境
            elif hasattr(self, 'scoring_version') and self.scoring_version == "v3.6":
                # v3.6机器学习评分系统的因子映射
                # 将机器学习特征映射到传统的5个维度
                momentum = factor_scores.get('price_momentum', 50)  # 价格动量
                mean_reversion = (factor_scores.get('bbi', 50) + factor_scores.get('pb', 50) + factor_scores.get('pe_ttm', 50)) / 3  # 均值回归
                volume_breakout = (factor_scores.get('volume_surge', 50) + factor_scores.get('kdj_cross', 50)) / 2  # 成交量突破
                relative_performance = (factor_scores.get('rsi', 50) + factor_scores.get('relative_strength', 50)) / 2  # 相对表现
                stability = (factor_scores.get('volatility_risk', 50) + factor_scores.get('market_cap', 50) + factor_scores.get('turnover_rate', 50)) / 3  # 稳定性
                market_regime_score = 0  # v3.6没有市场环境分
            else:
                # v2/v3评分系统的因子
                momentum = factor_scores.get('momentum', 50)
                mean_reversion = factor_scores.get('mean_reversion', 50)
                volume_breakout = factor_scores.get('volume_breakout', 50)
                relative_performance = factor_scores.get('relative_performance', 50)
                stability = factor_scores.get('stability', 50)
                market_regime_score = 0  # v2/v3没有市场环境分
            
            # 根据评分系统版本输出不同格式
            if hasattr(self, 'scoring_version') and self.scoring_version in ("v4.9.0", "v4.9.1"):
                # V4.9.0/V4.9.1: Q95 Widen-then-Concentrate — 显示Q95值和head_rank
                q95_val = stock.get('q95_pred_10d', 0)
                hr = stock.get('head_rank', '-')
                close_price = stock.get('close_price', 0)
                buy_price = stock.get('suggested_buy_price', 0)
                stop_loss = stock.get('stop_loss_price', 0)
                target = stock.get('take_profit_price', 0)
                pos_pct = stock.get('position_pct', 0)
                pos_str = f"{pos_pct}%" if pos_pct > 0 else "—"
                report += f"| {i+1} | {stock_code} | {stock_name} | {strategies_str} | {q95_val:.4f} | {recommendation} | {pred_10d*100:+.2f}% | {close_price:.2f} | {buy_price:.2f} | {stop_loss:.2f} | {target:.2f} | {pos_str} |\n"
            elif hasattr(self, 'scoring_version') and self.scoring_version in ["v4.4", "v4.4.2", "v4.5", "v4.6", "v4.7.1", "v4.7.2", "v4.7.3", "v4.7.4", "v4.7.5", "v4.7.6", "v4.7.7", "v4.7.8", "v4.7.9", "v4.8.0", "v4.8.1", "v4.8.2", "v4.8.4", "v4.8.5", "v4.8.6", "v4.8.7", "v4.8.8", "v4.9.0.1", "v4.9.0.2", "v5.0"]:
                # V4.4+ 多目标预测 - composite排序
                composite_val = stock.get('composite', 0)
                close_price = stock.get('close_price', 0)
                buy_price = stock.get('suggested_buy_price', 0)
                stop_loss = stock.get('stop_loss_price', 0)
                target = stock.get('take_profit_price', 0)
                pos_pct = stock.get('position_pct', 0)
                pos_str = f"{pos_pct}%" if pos_pct > 0 else "—"
                if getattr(self, 'optimizer_version', 'v1') == 'v2':
                    stop_pct_val = stock.get('risk_pct', 0)
                    rr_val = stock.get('risk_reward_ratio', 0)
                    atr_pct_val = stock.get('atr_pct', 0) * 100
                    report += f"| {i+1} | {stock_code} | {stock_name} | {strategies_str} | {composite_val:.6f} | {recommendation} | {pred_10d*100:+.2f}% | {close_price:.2f} | {buy_price:.2f} | {stop_loss:.2f} | {target:.2f} | {pos_str} | {stop_pct_val:.1f}% | {rr_val:.1f} | {atr_pct_val:.1f}% |\n"
                else:
                    report += f"| {i+1} | {stock_code} | {stock_name} | {strategies_str} | {composite_val:.6f} | {recommendation} | {pred_10d*100:+.2f}% | {close_price:.2f} | {buy_price:.2f} | {stop_loss:.2f} | {target:.2f} | {pos_str} |\n"
            elif hasattr(self, 'scoring_version') and self.scoring_version in ["v3.9", "v3.94", "v3.95", "v3.96", "v4.0", "v4.2", "v4.3"]:
                # 🏆 V3.9.x Production Model
                predicted_return_pct = predicted_return * 100  # 转换为百分比
                confidence_pct = confidence_score * 100  # 转换为百分比
                close_price = stock.get('close_price', 0)
                buy_price = stock.get('suggested_buy_price', 0)
                stop_loss = stock.get('stop_loss_price', 0)
                target = stock.get('take_profit_price', 0)
                pos_pct = stock.get('position_pct', 0)
                pos_str = f"{pos_pct}%" if pos_pct > 0 else "—"
                report += f"| {i+1} | {stock_code} | {stock_name} | {strategies_str} | {score:.1f} | {recommendation} | {predicted_return_pct:+.2f}% | {close_price:.2f} | {buy_price:.2f} | {stop_loss:.2f} | {target:.2f} | {pos_str} |\n"
            elif hasattr(self, 'scoring_version') and self.scoring_version in ["v3.8", "v3.81"]:
                # 检查是否为混合模式
                if analysis.get("v38_mixed_mode", False):
                    # 混合模式：包含传统策略和策略数量
                    report += f"| {i+1} | {stock_code} | {stock_name} | {strategies_str} | {strategy_count} | {score:.1f} | {recommendation} | {confidence_score:.3f} | {short_term_score:.1f} | {medium_term_score:.1f} | {long_term_score:.1f} | {risk_level} | {overall_quality:.2f} |\n"
                else:
                    # 纯V3.8模式
                    report += f"| {i+1} | {stock_code} | {stock_name} | {strategies_str} | {score:.1f} | {recommendation} | {confidence_score:.3f} | {short_term_score:.1f} | {medium_term_score:.1f} | {long_term_score:.1f} | {risk_level} | {overall_quality:.2f} |\n"
            elif hasattr(self, 'scoring_version') and self.scoring_version == "v3.7":
                report += f"| {i+1} | {stock_code} | {stock_name} | {strategies_str} | {score:.1f} | {recommendation} | {technical_score:.1f} | {fundamental_score:.1f} | {macro_score:.1f} | {sentiment_score:.1f} | {temporal_score:.1f} |\n"
            elif hasattr(self, 'scoring_version') and self.scoring_version == "v3.5":
                report += f"| {i+1} | {stock_code} | {stock_name} | {strategies_str} | {score:.1f} | {recommendation} | {technical_score:.1f} | {fundamental_score:.1f} | {performance_score:.1f} | {market_regime_score:.1f} | {zhixing_score:.1f} | {signal_strength} |\n"
            elif hasattr(self, 'scoring_version') and self.scoring_version in ["v3.4", "v3.41"]:
                report += f"| {i+1} | {stock_code} | {stock_name} | {strategies_str} | {score:.1f} | {recommendation} | {technical_score:.1f} | {fundamental_score:.1f} | {performance_score:.1f} | {market_regime_score:.1f} |\n"
            elif hasattr(self, 'scoring_version') and self.scoring_version == "v3.3":
                report += f"| {i+1} | {stock_code} | {stock_name} | {strategies_str} | {score:.1f} | {recommendation} | {technical_score:.1f} | {volume_momentum_score:.1f} | {fundamental_score:.1f} | {sentiment_capital_score:.1f} | {risk_control_score:.1f} | {market_environment_score:.1f} |\n"
            elif hasattr(self, 'scoring_version') and self.scoring_version == "v3.2":
                report += f"| {i+1} | {stock_code} | {stock_name} | {strategies_str} | {score:.1f} | {recommendation} | {technical_score:.1f} | {squeeze_score:.1f} | {fundamental_score:.1f} | {performance_score:.1f} | {sentiment_score:.1f} | {risk_control_score:.1f} | {market_regime_score:.1f} |\n"
            elif hasattr(self, 'scoring_version') and self.scoring_version == "v3.51":
                report += f"| {i+1} | {stock_code} | {stock_name} | {strategies_str} | {score:.1f} | {recommendation} | {volatility_risk_score:.1f} | {market_cap_score:.1f} | {price_momentum_score:.1f} | {pb_score:.1f} | {pe_ttm_score:.1f} | {rsi6_score:.1f} | {kdj_k_score:.1f} | {bbi_score:.1f} | {kdj_d_score:.1f} | {zhixing_trend_score:.1f} | {volume_surge_score:.1f} | {zhixing_multiavg_score:.1f} |\n"
            elif hasattr(self, 'scoring_version') and self.scoring_version == "v3.53":
                report += f"| {i+1} | {stock_code} | {stock_name} | {strategies_str} | {score:.1f} | {recommendation} | {score_1d:.1f} | {score_3d:.1f} | {score_5d:.1f} | {score_10d:.1f} | {score_15d:.1f} | {main_factors_str} |\n"
            elif hasattr(self, 'scoring_version') and self.scoring_version == "v3.52":
                report += f"| {i+1} | {stock_code} | {stock_name} | {strategies_str} | {score:.1f} | {recommendation} | {volatility_risk_score:.1f} | {market_cap_score:.1f} | {price_momentum_score:.1f} | {pb_score:.1f} | {pe_ttm_score:.1f} | {rsi6_score:.1f} | {kdj_k_score:.1f} | {bbi_score:.1f} | {kdj_d_score:.1f} | {zhixing_trend_score:.1f} | {volume_surge_score:.1f} | {zhixing_multiavg_score:.1f} |\n"
            elif hasattr(self, 'scoring_version') and self.scoring_version == "v3.6":
                # v3.6机器学习特征显示 - 使用原始值和变换值的混合
                bbi_score = factor_scores.get('bbi', 40)
                volume_surge_score = factor_scores.get('volume_surge', 40) 
                price_momentum_score = factor_scores.get('price_momentum', 40)
                zhixing_multiavg_score = factor_scores.get('zhixing_multiavg', 40)
                rsi_score = factor_scores.get('rsi', 40)
                kdj_cross_score = factor_scores.get('kdj_cross', 40)
                turnover_rate_score = factor_scores.get('turnover_rate', 40)
                volatility_risk_score = factor_scores.get('volatility_risk', 40)
                relative_strength_score = factor_scores.get('relative_strength', 40)
                
                # 转换显示值：将变换值转换回用户友好的原始值
                pb_raw_val = factor_scores.get('pb', 40)
                pe_raw_val = factor_scores.get('pe_ttm', 40)
                market_cap_raw_val = factor_scores.get('market_cap', 40)
                
                # 如果是倒数形式，转换回原值
                if pb_raw_val > 0 and pb_raw_val <= 1:  # 倒数形式
                    pb_display = 1.0 / pb_raw_val if pb_raw_val > 0.001 else 999
                else:
                    pb_display = pb_raw_val
                    
                if pe_raw_val > 0 and pe_raw_val <= 1:  # 倒数形式
                    pe_display = 1.0 / pe_raw_val if pe_raw_val > 0.001 else 999
                else:
                    pe_display = pe_raw_val
                    
                # 市值：如果是对数形式，转换回原值
                if market_cap_raw_val < 20:  # 对数形式
                    market_cap_display = f"{np.exp(market_cap_raw_val)/10000:.1f}万"
                else:
                    market_cap_display = f"{market_cap_raw_val/10000:.1f}万"
                
                # RSI显示（0.0-100.0范围内都是有效值）
                rsi_display = f"{rsi_score:.1f}"
                
                report += f"| {i+1} | {stock_code} | {stock_name} | {strategies_str} | {score:.1f} | {recommendation} | {bbi_score:.1f} | {volume_surge_score:.1f} | {price_momentum_score:.1f} | {zhixing_multiavg_score:.1f} | {rsi_display} | {market_cap_display} | {kdj_cross_score:.1f} | {pb_display:.1f} | {turnover_rate_score:.1f} | {volatility_risk_score:.1f} | {relative_strength_score:.1f} | {pe_display:.1f} |\n"
            elif hasattr(self, 'scoring_version') and self.scoring_version == "v3.1":
                report += f"| {i+1} | {stock_code} | {stock_name} | {strategies_str} | {score:.1f} | {recommendation} | {momentum:.0f} | {mean_reversion:.0f} | {volume_breakout:.0f} | {relative_performance:.0f} | {stability:.0f} | {market_regime_score:.0f} |\n"
            else:
                report += f"| {i+1} | {stock_code} | {stock_name} | {strategies_str} | {score:.1f} | {recommendation} | {momentum:.0f} | {mean_reversion:.0f} | {volume_breakout:.0f} | {relative_performance:.0f} | {stability:.0f} |\n"
        
        report += "\n"
        
        report += "## 🏆 明日买入推荐股票详细分析\n\n"
        
        # 分别显示多策略和单策略股票
        multi_stocks = analysis.get("multi_strategy_recommendations", [])
        single_detailed = [s for s in analysis.get("single_strategy_recommendations", []) if s.get('needs_detailed_analysis', False)]
        
        # 多策略股票详细分析
        if multi_stocks:
            report += "### 🌟 多策略推荐股票\n\n"
            report += f"*共有 {len(multi_stocks)} 只股票被多个策略选中*\n\n"
            
            for i, stock in enumerate(multi_stocks, 1):
                report += f"#### {i}. {stock['stock_code']} - {stock.get('stock_name', '未知')}\n\n"
                report += self._generate_stock_detail(stock)
                report += "\n---\n\n"
        
        # 单策略股票详细分析（TOP 20）
        if single_detailed:
            report += "### 📊 单策略推荐股票（TOP 20）\n\n"
            report += f"*单策略选中股票共 {len(analysis.get('single_strategy_recommendations', []))} 只，以下显示前20只的详细分析*\n\n"

            for i, stock in enumerate(single_detailed, 1):
                report += f"#### {i}. {stock['stock_code']} - {stock.get('stock_name', '未知')}\n\n"
                report += self._generate_stock_detail(stock)
                report += "\n---\n\n"

        # 全市场ML推荐股票详细分析（TOP 10）
        no_strategy_detailed = [s for s in analysis.get("no_strategy_recommendations", []) if s.get('needs_detailed_analysis', False)]
        if no_strategy_detailed:
            report += "### 🌐 全市场ML推荐股票（TOP 10）\n\n"
            report += f"*以下股票未被任何策略选中，但ML评分排名靠前（共 {len(analysis.get('no_strategy_recommendations', []))} 只非策略股票）*\n\n"

            for i, stock in enumerate(no_strategy_detailed, 1):
                report += f"#### {i}. {stock['stock_code']} - {stock.get('stock_name', '未知')}\n\n"
                report += self._generate_stock_detail(stock)
                report += "\n---\n\n"

        report += """## ⚠️ 风险提示

### 市场风险
- 股市有风险，投资需谨慎
- 历史表现不代表未来收益
- 市场环境变化可能影响策略效果

### 策略风险  
- 技术指标存在滞后性
- 量化策略可能在特殊市场环境下失效
- 建议结合基本面分析和市场情绪

### 操作建议
- 建议分批买入，避免集中投资
- 设置合理的止损位(建议8-10%)
- 单只股票仓位控制在总资金的5-10%
- 密切关注市场变化，及时调整策略

### 免责声明
本分析报告仅供参考，不构成投资建议。投资者应根据自身风险承受能力做出投资决策。

## 📊 数据来源
- **数据源**: Tushare Pro API
- **股票范围**: A股主板、科创板、创业板、北交所 + ETF/基金
- **数据频率**: 日线数据
- **分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🤖 Generated with Claude Code
"""
        
        return report

# is_trading_day 已提取到 core.trading_calendar 模块
from core.trading_calendar import is_trading_day

def main(target_date: str = None, scoring_version: str = "v3", stocks_only: bool = False, skip_strategies: bool = False, full_market: bool = False, optimizer_version: str = 'v1', optimizer_params_path: str = None):
    """主函数

    Args:
        target_date: 分析日期，格式YYYY-MM-DD
        scoring_version: 评分版本，'v2'、'v3'或'v4'，默认'v3'
        stocks_only: 是否只考虑股票，不包括ETF基金，默认False
        skip_strategies: 跳过策略筛选，全市场ML评分，默认False
        full_market: 全市场ML评分 + 策略标注模式，默认False
    """
    # v4.8 alias → v4.8.0
    if scoring_version == "v4.8":
        scoring_version = "v4.8.0"
    # v3.6、v3.7、v3.8、v3.9、v3.94、v3.95版本应该只评价股票，因为ETF等因子无法与股票直接对比
    if scoring_version in ["v3.6", "v3.7", "v3.8", "v3.81", "v3.9", "v3.94", "v3.95", "v3.96", "v4.0", "v4.2", "v4.3", "v4.4", "v4.4.2", "v4.5", "v4.6", "v4.7.1", "v4.7.2", "v4.7.3", "v4.7.5", "v4.7.6", "v4.7.7", "v4.7.8", "v4.7.9", "v4.8.0", "v4.8.1", "v4.8.2", "v4.8.4", "v4.9.0.2", "v5.0"] and not stocks_only:
        stocks_only = True
        logger.info(f"🔍 {scoring_version}机器学习版本自动开启仅股票模式（ETF预测信心不足）")
    
    if target_date:
        # 检查是否为交易日
        if not is_trading_day(target_date):
            logger.info(f"{target_date} 不是交易日，跳过选股分析")
            print(f"# {target_date} 不是交易日")
            print(f"跳过选股分析，请选择交易日进行分析。")
            return False
            
        logger.info(f"=== {target_date} 股票选股分析开始 (评分版本: {scoring_version}) ===")
    else:
        logger.info(f"=== 最新日期股票选股分析开始 (评分版本: {scoring_version}) ===")
    
    # 创建选股器，传入评分版本和股票筛选选项
    selector = TomorrowStockSelector(scoring_version=scoring_version, stocks_only=stocks_only, skip_strategies=skip_strategies,
                                     optimizer_version=optimizer_version, optimizer_params_path=optimizer_params_path)
    
    # 获取分析日期
    if target_date:
        latest_date = pd.Timestamp(target_date)
        logger.info(f"指定分析日期: {latest_date.strftime('%Y-%m-%d')}")
        target_date_str = target_date
    else:
        target_date_str = None
        
    # 加载数据（现在传入目标日期）
    logger.info("加载股票数据...")
    data = selector.load_data(target_date=target_date_str)
    
    if not data:
        logger.error("未能加载任何数据，退出")
        return
    
    # 如果没有指定日期，从加载的数据中获取最新交易日
    if not target_date:
        latest_date = selector.get_latest_trading_date(data)
        logger.info(f"最新交易日: {latest_date.strftime('%Y-%m-%d')}")
    
    # 根据评分版本选择不同的工作流程
    if scoring_version == "v3.81":
        # V3.81混合工作流程：先运行传统策略，再对选中股票进行Level 4质量评分
        logger.info("V3.81 Level 4质量评分模式：先运行传统选股策略...")

        # 第一步：运行传统选股策略
        traditional_results = selector.run_selectors(data, latest_date)
        logger.info(f"传统策略完成，共 {len(traditional_results)} 个策略")

        # 收集所有被传统策略选中的股票
        selected_stocks = set()
        for strategy, stocks in traditional_results.items():
            selected_stocks.update(stocks)
            logger.info(f"  {strategy}: {len(stocks)}只股票")

        selected_stock_list = list(selected_stocks)
        logger.info(f"传统策略共选中 {len(selected_stock_list)} 只独特股票")

        # 第二步：对被选中的股票进行V3.81 Level 4质量评分
        logger.info("对传统策略选中的股票进行V3.81 Level 4质量评分...")

        # 初始化V3.81评分器
        v381_selector = TomorrowStockSelector(scoring_version="v3.81", stocks_only=stocks_only)

        # 使用V3.81批量预测接口
        predictions = v381_selector.scoring_engine_v381.predict_scores_with_quality(
            selected_stock_list,
            latest_date.strftime('%Y-%m-%d')
        )

        # 🔧 缓存V3.81批处理结果到主实例，避免individual processing时重复计算不一致问题
        selector.v381_batch_cache = predictions.copy()

        # 转换为兼容格式
        evaluation_result = {
            'error': False,
            'stocks': []
        }

        for code in selected_stock_list:
            prediction_data = predictions.get(code, {})

            if isinstance(prediction_data, dict):
                # V3.81格式：包含Level 4质量评分
                overall_score = prediction_data.get('overall_score', 50.0)
                short_term_score = prediction_data.get('short_term_score', 50.0)
                medium_term_score = prediction_data.get('medium_term_score', 50.0)
                long_term_score = prediction_data.get('long_term_score', 50.0)
                confidence_score = prediction_data.get('confidence_score', 0.8)
                # 🎯 Level 4质量评分
                quality_score = prediction_data.get('quality_score', 0.5)
            else:
                # 兼容格式
                overall_score = prediction_data if isinstance(prediction_data, (int, float)) else 50.0
                short_term_score = overall_score * 1.1
                medium_term_score = overall_score
                long_term_score = overall_score * 0.9
                confidence_score = 0.8
                quality_score = 0.5
                # 兼容格式的默认投资建议
                prediction_data = {'recommendation': '观望'}

            stock_result = {
                'code': code,
                'final_score': overall_score / 100.0,  # 转为0-1分制
                'confidence_score': confidence_score,
                'short_term_score': short_term_score / 100.0,
                'medium_term_score': medium_term_score / 100.0,
                'long_term_score': long_term_score / 100.0,
                # 🎯 使用Level 4质量评分作为质量指标
                'overall_quality': quality_score,
                'quality_score': quality_score,
                'risk_level': v381_selector._calculate_risk_level_v381(overall_score, confidence_score, quality_score),
                'confidence_level': 'high' if confidence_score > 0.7 else 'medium' if confidence_score > 0.4 else 'low',
                # 🔧 保留V3.81原始投资建议，避免在后续处理中丢失
                'recommendation': prediction_data.get('recommendation', '观望')
            }
            evaluation_result['stocks'].append(stock_result)

        if evaluation_result.get('error'):
            logger.error(f"V3.81评分失败: {evaluation_result['error']}")
            return False

        # 分析V3.81混合结果
        analysis = selector.analyze_v381_mixed_results(traditional_results, evaluation_result, data)
    elif scoring_version == "v3.8":
        # V3.8混合工作流程：先运行传统策略，再对选中股票进行自适应评分
        logger.info("V3.8混合评分模式：先运行传统选股策略...")

        # 第一步：运行传统选股策略
        traditional_results = selector.run_selectors(data, latest_date)
        logger.info(f"传统策略完成，共 {len(traditional_results)} 个策略")

        # 收集所有被传统策略选中的股票
        selected_stocks = set()
        for strategy, stocks in traditional_results.items():
            selected_stocks.update(stocks)
            logger.info(f"  {strategy}: {len(stocks)}只股票")

        selected_stock_list = list(selected_stocks)
        logger.info(f"传统策略共选中 {len(selected_stock_list)} 只独特股票")

        # 第二步：对被选中的股票进行V3.8自适应评分
        logger.info("对传统策略选中的股票进行V3.8自适应评分...")

        # 使用V3.80批量预测接口
        predictions = selector.scoring_engine_v38.predict_scores(
            selected_stock_list,
            latest_date.strftime('%Y-%m-%d')
        )

        # 转换为兼容格式
        evaluation_result = {
            'error': False,
            'stocks': []
        }

        for code in selected_stock_list:
            prediction_data = predictions.get(code, {})

            # 🔧 修复：处理新的预测格式，使用真实的分期评分
            if isinstance(prediction_data, dict):
                # 新格式：包含详细分期信息
                overall_score = prediction_data.get('overall_score', 50.0)
                short_term_score = prediction_data.get('short_term_score', 50.0)
                medium_term_score = prediction_data.get('medium_term_score', 50.0)
                long_term_score = prediction_data.get('long_term_score', 50.0)
                confidence_score = prediction_data.get('confidence_score', 0.8)
            else:
                # 旧格式：单一评分
                overall_score = prediction_data if isinstance(prediction_data, (int, float)) else 50.0
                short_term_score = overall_score * 1.1  # 简单计算
                medium_term_score = overall_score
                long_term_score = overall_score * 0.9
                confidence_score = 0.8

            stock_result = {
                'code': code,
                'final_score': overall_score / 100.0,  # 转为0-1分制
                'confidence_score': confidence_score,
                'short_term_score': short_term_score / 100.0,   # 🔧 使用真实短期评分
                'medium_term_score': medium_term_score / 100.0, # 🔧 使用真实中期评分
                'long_term_score': long_term_score / 100.0,     # 🔧 使用真实长期评分
                'overall_quality': confidence_score,
                'risk_level': 'low' if overall_score > 70 else 'medium' if overall_score > 50 else 'high',
                'confidence_level': 'high' if overall_score > 70 else 'medium'
            }
            evaluation_result['stocks'].append(stock_result)

        if evaluation_result.get('error'):
            logger.error(f"V3.8评分失败: {evaluation_result['error']}")
            return False

        # 第三步：使用V3.8评分重新排序和分析
        v38_stocks = evaluation_result.get('stocks', [])

        # 按V3.8评分排序所有被传统策略选中的股票
        v38_sorted_stocks = sorted(v38_stocks, key=lambda x: x['final_score'], reverse=True)

        # 构建结果格式，包含传统策略信息和V3.8评分
        results = traditional_results.copy()  # 保留传统策略结果
        results["V3.8自适应评分"] = [stock['code'] for stock in v38_sorted_stocks]

        logger.info(f"V3.8评分完成，对 {len(v38_sorted_stocks)} 只传统策略选中的股票进行了评分")

        # 使用混合分析逻辑
        analysis = selector.analyze_v38_mixed_results(traditional_results, evaluation_result, data, latest_date)

    else:
        if full_market:
            # 全市场+策略标注模式：先跑策略获取标注，再对全市场所有股票ML评分
            logger.info("🌐 全市场+策略标注模式：运行策略筛选 + 全市场ML评分...")
            strategy_results = selector.run_selectors(data, latest_date)
            strategy_stocks = set()
            for picks in strategy_results.values():
                strategy_stocks.update(picks)
            all_codes = list(data.keys())
            non_strategy = [c for c in all_codes if c not in strategy_stocks]
            logger.info(f"  策略选中: {len(strategy_stocks)} 只, 全市场补充: {len(non_strategy)} 只, 总计: {len(all_codes)} 只")
            # 添加全市场非策略股票到虚拟策略组
            strategy_results["全市场ML评分"] = non_strategy
            results = strategy_results
        elif selector.skip_strategies:
            # 全市场ML评分模式：跳过策略筛选，所有股票直接进入ML评分
            logger.info("🌐 全市场ML评分模式：跳过8大策略筛选...")
            all_codes = list(data.keys())
            logger.info(f"  候选股票: {len(all_codes)} 只 (全市场)")
            # 构造一个虚拟的 results，让 analyze_results 对全部股票评分
            results = {"全市场ML评分": all_codes}
        else:
            # 传统选股策略
            logger.info("运行选股策略...")
            results = selector.run_selectors(data, latest_date)

        # 分析结果
        logger.info("分析选股结果...")
        analysis = selector.analyze_results(results, data, latest_date)
    
    # 生成报告
    logger.info("生成分析报告...")
    report = selector.generate_report(analysis, latest_date)
    
    # 根据评分版本选择不同的报告目录
    if scoring_version == "v5.0":
        report_dir = Path("reports/daily_selection_v5.0")
    elif scoring_version == "v4.7.1":
        report_dir = Path("reports/daily_selection_v4.7.1")
    elif scoring_version == "v4.8.8":
        report_dir = Path("reports/daily_selection_v4.8.8")
    elif scoring_version == "v4.9.1":
        report_dir = Path("reports/daily_selection_v4.9.1")
    elif scoring_version == "v4.9.0.2":
        report_dir = Path("reports/daily_selection_v4.9.0.2")
    elif scoring_version == "v4.9.0.1":
        report_dir = Path("reports/daily_selection_v4.9.0.1")
    elif scoring_version == "v4.9.0":
        report_dir = Path("reports/daily_selection_v4.9.0")
    elif scoring_version == "v4.8.7":
        report_dir = Path("reports/daily_selection_v4.8.7")
    elif scoring_version == "v4.8.6":
        report_dir = Path("reports/daily_selection_v4.8.6")
    elif scoring_version == "v4.8.5":
        report_dir = Path("reports/daily_selection_v4.8.5")
    elif scoring_version == "v4.8.4":
        report_dir = Path("reports/daily_selection_v4.8.4")
    elif scoring_version == "v4.8.2":
        report_dir = Path("reports/daily_selection_v4.8.2")
    elif scoring_version == "v4.8.1":
        report_dir = Path("reports/daily_selection_v4.8.1")
    elif scoring_version == "v4.8.0":
        report_dir = Path("reports/daily_selection_v4.8.0")
    elif scoring_version == "v4.7.9":
        report_dir = Path("reports/daily_selection_v4.7.9")
    elif scoring_version == "v4.7.8":
        report_dir = Path("reports/daily_selection_v4.7.8")
    elif scoring_version == "v4.7.7":
        report_dir = Path("reports/daily_selection_v4.7.7")
    elif scoring_version == "v4.7.6":
        report_dir = Path("reports/daily_selection_v4.7.6")
    elif scoring_version == "v4.7.5":
        report_dir = Path("reports/daily_selection_v4.7.5")
    elif scoring_version == "v4.7.3":
        report_dir = Path("reports/daily_selection_v4.7.3")
    elif scoring_version == "v4.7.2":
        report_dir = Path("reports/daily_selection_v4.7.2")
    elif scoring_version == "v4.6":
        report_dir = Path("reports/daily_selection_v4.6")
    elif scoring_version == "v4.5":
        report_dir = Path("reports/daily_selection_v4.5")
    elif scoring_version == "v4.4.2":
        report_dir = Path("reports/daily_selection_v4.4.2")
    elif scoring_version == "v4.4":
        report_dir = Path("reports/daily_selection_v4.4")
    elif scoring_version == "v4.3":
        report_dir = Path("reports/daily_selection_v4.3")
    elif scoring_version == "v4.2":
        report_dir = Path("reports/daily_selection_v4.2")
    elif scoring_version == "v4.0":
        report_dir = Path("reports/daily_selection_v4.0")
    elif scoring_version == "v4":
        report_dir = Path("reports/daily_selection_v4")
    elif scoring_version == "v3.96":
        report_dir = Path("reports/daily_selection_v3.96")
    elif scoring_version == "v3.95":
        report_dir = Path("reports/daily_selection_v3.95")
    elif scoring_version == "v3.94":
        report_dir = Path("reports/daily_selection_v3.94")
    elif scoring_version == "v3.9":
        report_dir = Path("reports/daily_selection_v3.9")
    elif scoring_version == "v3.81":
        report_dir = Path("reports/daily_selection_v3.81")
    elif scoring_version == "v3.8":
        report_dir = Path("reports/daily_selection_v3.8")
    elif scoring_version == "v3.7":
        report_dir = Path("reports/daily_selection_v3.7")
    elif scoring_version == "v3.6":
        report_dir = Path("reports/daily_selection_v3.6")
    elif scoring_version == "v3.53":
        report_dir = Path("reports/daily_selection_v3.53")
    elif scoring_version == "v3.52":
        report_dir = Path("reports/daily_selection_v3.52")
    elif scoring_version == "v3.51":
        report_dir = Path("reports/daily_selection_v3.51")
    elif scoring_version == "v3.5":
        report_dir = Path("reports/daily_selection_v3.5")
    elif scoring_version == "v3.41":
        report_dir = Path("reports/daily_selection_v3.41")
    elif scoring_version == "v3.4":
        report_dir = Path("reports/daily_selection_v3.4")
    elif scoring_version == "v3.3":
        report_dir = Path("reports/daily_selection_v3.3")
    elif scoring_version == "v3.2":
        report_dir = Path("reports/daily_selection_v3.2")
    elif scoring_version == "v3.1":
        report_dir = Path("reports/daily_selection_v3.1")
    elif scoring_version == "v3":
        report_dir = Path("reports/daily_selection_v3")
    else:
        report_dir = Path("reports/daily_selection")

    # 全市场模式：报告目录加 _fullmarket 后缀
    if skip_strategies or full_market:
        report_dir = Path(str(report_dir) + "_fullmarket")

    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_filename = f"选股分析报告_{latest_date.strftime('%Y%m%d')}.md"
    report_file = report_dir / report_filename
    
    # 保存分析数据到JSON文件（供AI增强报告使用）
    json_filename = f"analysis_data_{latest_date.strftime('%Y%m%d')}.json"
    json_file = report_dir / json_filename
    
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"分析数据已保存到: {json_file}")
    except Exception as e:
        logger.error(f"保存JSON数据失败: {e}")
    
    # 输出完整报告到stdout
    print(report)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
        
    logger.info(f"分析报告已保存到: {report_file}")
    logger.info(f"共分析 {analysis.get('total_unique_stocks', 0)} 只股票")
    logger.info(f"详细分析 {analysis.get('detailed_analysis_count', 0)} 只股票")
    logger.info("=== 选股分析完成 ===")

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='量化选股分析器')
    parser.add_argument('date', nargs='?', help='分析日期 (YYYY-MM-DD格式)')
    parser.add_argument('--scoring-version', '-v',
                       choices=['v2', 'v3', 'v3.1', 'v3.2', 'v3.3', 'v3.4', 'v3.41',
                                'v3.5', 'v3.51', 'v3.52', 'v3.53', 'v3.6', 'v3.7',
                                'v3.8', 'v3.81', 'v3.9', 'v3.94', 'v3.95', 'v3.96',
                                'v4', 'v4.0', 'v4.2', 'v4.3', 'v4.4', 'v4.4.2', 'v4.5', 'v4.6', 'v4.7.1', 'v4.7.2', 'v4.7.3', 'v4.7.5', 'v4.7.6', 'v4.7.7', 'v4.7.8', 'v4.7.9', 'v4.8', 'v4.8.0', 'v4.8.1', 'v4.8.2', 'v4.8.4', 'v4.8.5', 'v4.8.6', 'v4.8.7', 'v4.8.8', 'v4.9.0', 'v4.9.0.1', 'v4.9.0.2', 'v4.9.1', 'v5.0'],
                       default='v4.9.0.1',
                       help='评分版本 (默认v4.9.0.1, 生产推荐, 配合focus_days=15+EMA0.7+CPPI(8,20)+SF30 → V4=92.8%% S级)。'
                            '活跃版本: v3.9(生产A级), v3.96(Robust Z-Score,ICIR>0.2), '
                            'v4.3(Walk-Forward+强正则), v4.4(V4.3+6增强模块), '
                            'v4.4.2(V4.4+三层组合风控), '
                            'v4.5(V4.4.1+CPPI动态仓位,S级84/105), '
                            'v4.6(V4.4+ICIR权重+MetaLearner+增强流动性+小盘加成), '
                            'v4.7.1(V4.4+Bug修复+17新特征+LambdaRank), '
                            'v5.0(Unified Fusion,v39+v40+neural)。'
                            '已弃用: v2-v3.81, v3.94, v4 (仍可使用但不推荐)')
    parser.add_argument('--stocks-only', '-s', action='store_true',
                       help='只考虑A股股票，不包括ETF基金等')
    parser.add_argument('--skip-strategies', action='store_true',
                       help='跳过8大量化策略筛选，直接对全市场所有股票进行ML评分')
    parser.add_argument('--full-market', action='store_true', default=True,
                       help='全市场ML评分+策略标注模式（默认开启）')
    parser.add_argument('--no-full-market', action='store_true',
                       help='关闭全市场模式，仅对策略选中的股票评分')
    parser.add_argument('--optimizer', choices=['v1', 'v2'], default='v2',
                       help='价格/仓位优化器版本: v1=旧逻辑, v2=自适应价格+风险预算(默认)')
    parser.add_argument('--optimizer-params', default=None,
                       help='v2优化器参数文件路径 (默认optimizer_params.json)')

    args = parser.parse_args()

    full_market = args.full_market and not args.no_full_market
    main(target_date=args.date, scoring_version=args.scoring_version,
         stocks_only=args.stocks_only, skip_strategies=args.skip_strategies,
         full_market=full_market, optimizer_version=getattr(args, 'optimizer', 'v1'),
         optimizer_params_path=getattr(args, 'optimizer_params', None))