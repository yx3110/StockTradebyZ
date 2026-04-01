#!/usr/bin/env python3
"""
V4.9.3 production scorer — V4.9.0.1底座 + 特征精选 + BRAIN代理因子 + 权重收缩

与V4901的区别:
  - V4901: 标准61特征 composite排序
  - V4.9.3: 裁剪13弱特征 + 3 BRAIN代理因子 (~51特征)
  - V4.9.3: LGB feature_fraction=0.5 + 集成权重clip+shrinkage

继承V4901的composite排序 (pred_10d*0.6 + pred_15d*0.4) 和市场门控。
"""

import json
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List

from .v4901_production_scorer import V4901ProductionScorer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class V493ProductionScorer(V4901ProductionScorer):
    """V4.9.3 scorer — 特征精选 + BRAIN代理因子, composite排序"""

    def __init__(self, model_type: str = 'small_data'):
        self._v493_model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v493'
        super().__init__(model_type=model_type)

    def _load_models(self):
        """加载v493模型, fallback到v4901→v490→v485"""
        v493_files = list(self._v493_model_dir.glob('v493_*.pkl'))
        if v493_files:
            self.model_dir = self._v493_model_dir
            latest = max(v493_files, key=lambda f: f.stat().st_mtime)
            self._load_model_from_file(latest, label='V4.9.3')
            has_q95 = 'lgb_q95' in self.models.get('10d', {})
            logger.info(f"  Q95 in ensemble: {has_q95}")
            # 加载V493的推荐阈值
            th_path = self._v493_model_dir / 'recommendation_thresholds.json'
            if th_path.exists():
                with open(th_path) as f:
                    self._comp_thresholds = json.load(f)
                logger.info(f"  V493 composite阈值: strong_buy>={self._comp_thresholds.get('strong_buy', 'N/A')}")
            return
        logger.warning("  V493模型未找到, fallback到V4901")
        super()._load_models()
