"""超参数试验 profile 注册表 (2026-07-13 hp sweep).

背景: NG 系列生产训练 (NGTrainer → V473.train_single_target_models) 的 6 成员
ensemble 超参数自 V4.7.3 起从未系统性调过。本注册表配合 ng_trainer CLI
`--hp-profile` 使用: 训练器把选中 profile 挂到 `self._hp_overrides`,
V473.train_single_target_models 在构建各成员参数后 update 进去。

基线 (V473 现行值, 即 profile=None 时):
  lgb/lgb_rank: num_leaves=31, lr=0.02, feature_fraction=0.6, bagging_fraction=0.7,
                reg_alpha=0.5, reg_lambda=3.0, min_data_in_leaf=200,
                min_gain_to_split=0.01, path_smooth=5.0, 1000轮 es30
  xgb: max_depth=6, lr=0.02, subsample=0.7, colsample=0.6, L1=0.5, L2=3.0,
       min_child_weight=50, gamma=0.1, 1000轮 es30
  cb:  iterations=1000, lr=0.02, depth=6, l2_leaf_reg=10, min_data_in_leaf=200, es30
  rf:  n_estimators=100, max_depth=15, max_features=0.6, min_samples_leaf=200
  hgb: max_iter=1000, lr=0.02, max_depth=6, max_leaf_nodes=47, l2=3.0,
       min_samples_leaf=200

profile 结构: {成员名: {参数覆盖}, 'num_boost_round': int(可选)}
成员名: lgb / lgb_rank / xgb / cb / rf / hgb
'num_boost_round' 统一控制 lgb/xgb/lgb_rank 轮数上限与 cb iterations / hgb max_iter,
无需 (也不要) 在成员 dict 里重复写轮数。

注意: NG 的 downside/risk 辅助头 (ng_trainer._train_downside_model) 有独立超参数,
**有意**不参与本 sweep — 只扫收益 ensemble 6 成员, 保持辅助头恒定以免混淆归因。
"""

_TIGHT_LGB = {
    'num_leaves': 20, 'min_data_in_leaf': 500,
    'reg_alpha': 1.0, 'reg_lambda': 5.0, 'path_smooth': 10.0,
}

_LOOSE_LGB = {
    'num_leaves': 63, 'min_data_in_leaf': 100,
    'reg_alpha': 0.1, 'reg_lambda': 1.0, 'path_smooth': 2.0,
}

_FF085_LGB = {'feature_fraction': 0.85, 'bagging_fraction': 0.8}

PROFILES = {
    # 复合: 回到 V4.4 时代强正则 (V4.7.3 放宽前的方向)
    'tight': {
        'lgb': _TIGHT_LGB,
        'lgb_rank': _TIGHT_LGB,
        'xgb': {'max_depth': 5, 'min_child_weight': 100,
                'reg_alpha': 1.0, 'reg_lambda': 5.0},
        'cb': {'depth': 5, 'l2_leaf_reg': 20, 'min_data_in_leaf': 500},
        'rf': {'max_depth': 10, 'min_samples_leaf': 500},
        'hgb': {'max_depth': 5, 'max_leaf_nodes': 20,
                'l2_regularization': 5.0, 'min_samples_leaf': 500},
    },
    # 复合: 高容量低正则
    'loose': {
        'lgb': _LOOSE_LGB,
        'lgb_rank': _LOOSE_LGB,
        'xgb': {'max_depth': 8, 'min_child_weight': 20,
                'reg_alpha': 0.1, 'reg_lambda': 1.0},
        'cb': {'depth': 8, 'l2_leaf_reg': 5, 'min_data_in_leaf': 100},
        'rf': {'max_depth': 20, 'min_samples_leaf': 100},
        'hgb': {'max_depth': 8, 'max_leaf_nodes': 95,
                'l2_regularization': 1.0, 'min_samples_leaf': 100},
    },
    # 单因子: 学习率 0.02→0.05 (轮数上限不变, early stop 决定实际轮数)
    'lr005': {
        'lgb': {'learning_rate': 0.05},
        'lgb_rank': {'learning_rate': 0.05},
        'xgb': {'learning_rate': 0.05},
        'cb': {'learning_rate': 0.05},
        'hgb': {'learning_rate': 0.05},
    },
    # 单因子: 学习率 0.02→0.01, 轮数上限 1000→2000 补偿
    'lr001': {
        'num_boost_round': 2000,
        'lgb': {'learning_rate': 0.01},
        'lgb_rank': {'learning_rate': 0.01},
        'xgb': {'learning_rate': 0.01},
        'cb': {'learning_rate': 0.01},
        'hgb': {'learning_rate': 0.01},
    },
    # 单因子: 特征采样 0.6→0.85
    'ff085': {
        'lgb': _FF085_LGB,
        'lgb_rank': _FF085_LGB,
        'xgb': {'colsample_bytree': 0.85, 'subsample': 0.8},
        'cb': {'rsm': 0.85},
        'rf': {'max_features': 0.85},
    },
    # 单因子: 叶子最小样本 200→500
    'mdil500': {
        'lgb': {'min_data_in_leaf': 500},
        'lgb_rank': {'min_data_in_leaf': 500},
        'xgb': {'min_child_weight': 100},
        'cb': {'min_data_in_leaf': 500},
        'rf': {'min_samples_leaf': 500},
        'hgb': {'min_samples_leaf': 500},
    },
    # 单因子: 树容量 31→63 叶 (xgb/cb 深度 6→7)
    'leaves63': {
        'lgb': {'num_leaves': 63},
        'lgb_rank': {'num_leaves': 63},
        'xgb': {'max_depth': 7},
        'cb': {'depth': 7},
        'hgb': {'max_leaf_nodes': 63, 'max_depth': 7},
    },
}
