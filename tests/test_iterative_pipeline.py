#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the iterative pipeline — Task 2/3: L1 Fast Trainer

Tests:
- test_load_data_returns_dataframe: L1 trainer loads data with expected columns
- test_split_data_no_leakage: Train/val/test have no overlapping dates with purge gap
- test_train_returns_valid_result: Full L1 training returns structured result
"""

import sys
import os
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helper: build a minimal params dict for fast test runs
# ---------------------------------------------------------------------------
def _fast_params(start_date='2025-06-01', num_boost_round=30):
    """Return minimal params for a fast L1 test run."""
    return {
        'variant_name': 'l1_test',
        'training': {
            'l1_start_date': start_date,
            'purge_days': 5,
            'l1_num_boost_round': num_boost_round,
            'num_leaves': 16,
            'min_data_in_leaf': 50,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
        },
        'features': {
            'remove': [],
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestL1FastTrainer:

    def test_load_data_returns_dataframe(self):
        """L1 trainer loads data with expected columns."""
        from scripts.l1_fast_trainer import L1FastTrainer

        params = _fast_params(start_date='2025-06-01')
        trainer = L1FastTrainer(params)

        df = trainer._load_data()

        # Basic type check
        assert isinstance(df, pd.DataFrame), "Should return a DataFrame"

        # Must have rows
        assert len(df) > 0, "DataFrame should not be empty"

        # Required columns must be present
        required = ['code', 'trade_date', 'label_5d', 'label_10d']
        for col in required:
            assert col in df.columns, f"Missing required column: {col}"

        # feature_cols must be set and non-empty
        assert len(trainer.feature_cols) > 0, "feature_cols should be set after _load_data"

        # All feature_cols should be in df columns
        for fc in trainer.feature_cols:
            assert fc in df.columns, f"feature_col '{fc}' missing from DataFrame"

        # No all-NaN feature columns
        feat_in_df = [c for c in trainer.feature_cols if c in df.columns]
        all_nan_cols = [c for c in feat_in_df if df[c].isna().all()]
        assert len(all_nan_cols) == 0, f"Feature columns are all-NaN: {all_nan_cols}"

        # Check market features included
        market_cols = [c for c in df.columns if c.startswith('market_')]
        assert len(market_cols) > 0, "Should include market_* columns"

    def test_split_data_no_leakage(self):
        """Train/val/test have no overlapping dates, with purge gap."""
        from scripts.l1_fast_trainer import L1FastTrainer

        params = _fast_params(start_date='2025-06-01')
        trainer = L1FastTrainer(params)

        # Load first to set feature_cols
        df = trainer._load_data()
        train_df, val_df, test_df = trainer._split_data(df)

        # All three splits should be non-empty
        assert len(train_df) > 0, "train_df should not be empty"
        assert len(val_df) > 0, "val_df should not be empty"
        assert len(test_df) > 0, "test_df should not be empty"

        train_dates = set(train_df['trade_date'].unique())
        val_dates = set(val_df['trade_date'].unique())
        test_dates = set(test_df['trade_date'].unique())

        # No date overlap between any pair
        assert train_dates.isdisjoint(val_dates), \
            f"Train/val overlap: {train_dates & val_dates}"
        assert train_dates.isdisjoint(test_dates), \
            f"Train/test overlap: {train_dates & test_dates}"
        assert val_dates.isdisjoint(test_dates), \
            f"Val/test overlap: {val_dates & test_dates}"

        # Temporal order: all train dates < all val dates < all test dates
        assert max(train_dates) < min(val_dates), \
            "Train dates should all come before val dates"
        assert max(val_dates) < min(test_dates), \
            "Val dates should all come before test dates"

        # Purge gap: the gap between train end and val start should be >= purge_days
        # We check that val doesn't immediately follow train (at least purge_days gap)
        purge_days = params['training']['purge_days']
        all_sorted = sorted(df['trade_date'].unique())

        train_end_idx = all_sorted.index(max(train_dates))
        val_start_idx = all_sorted.index(min(val_dates))
        actual_gap = val_start_idx - train_end_idx - 1  # trading days skipped
        assert actual_gap >= purge_days - 1, \
            f"Purge gap too small: {actual_gap} < {purge_days - 1} (allowing 1 day tolerance)"

        # Approximate size checks: train ~70%, val ~15%, test ~15%
        n_dates = len(all_sorted)
        assert len(train_dates) > len(val_dates), "Train should have more dates than val"
        assert len(train_dates) > len(test_dates), "Train should have more dates than test"

    def test_train_returns_valid_result(self):
        """Full L1 training returns structured result dict."""
        from scripts.l1_fast_trainer import L1FastTrainer

        # Use small window and few rounds for speed
        params = _fast_params(start_date='2025-06-01', num_boost_round=30)
        trainer = L1FastTrainer(params)

        result = trainer.train()

        # Check result structure
        assert isinstance(result, dict), "train() should return a dict"

        required_keys = [
            'variant_name', 'level', 'duration_sec', 'metrics',
            'gate_pass', 'model_path', 'feature_cols',
        ]
        for key in required_keys:
            assert key in result, f"Missing key in result: {key}"

        # Type checks
        assert result['level'] == 'L1', "level should be 'L1'"
        assert result['variant_name'] == 'l1_test'
        assert isinstance(result['duration_sec'], float), "duration_sec should be float"
        assert result['duration_sec'] > 0, "duration_sec should be positive"
        assert isinstance(result['gate_pass'], bool), "gate_pass should be bool"
        assert isinstance(result['feature_cols'], list), "feature_cols should be list"
        assert len(result['feature_cols']) > 0, "feature_cols should be non-empty"

        # Metrics checks
        metrics = result['metrics']
        assert isinstance(metrics, dict), "metrics should be dict"

        metric_keys = [
            'test_ic_5d', 'test_ic_10d',
            'test_icir_5d', 'test_icir_10d',
            'val_ic_10d', 'train_val_gap',
            'n_features', 'top10_feature_importance',
        ]
        for key in metric_keys:
            assert key in metrics, f"Missing metric: {key}"

        # IC values should be numeric and in reasonable range
        for ic_key in ['test_ic_5d', 'test_ic_10d', 'val_ic_10d']:
            val = metrics[ic_key]
            assert isinstance(val, float), f"{ic_key} should be float"
            assert -1.0 <= val <= 1.0, f"{ic_key}={val} out of range [-1, 1]"

        # ICIR can be larger but should be finite
        for icir_key in ['test_icir_5d', 'test_icir_10d']:
            val = metrics[icir_key]
            assert np.isfinite(val), f"{icir_key}={val} should be finite"

        # train_val_gap should be numeric and finite
        assert np.isfinite(metrics['train_val_gap']), "train_val_gap should be finite"

        # n_features matches feature_cols length
        assert metrics['n_features'] == len(result['feature_cols']), \
            "n_features should match len(feature_cols)"

        # top10_feature_importance should be a list of up to 10 tuples
        top10 = metrics['top10_feature_importance']
        assert isinstance(top10, list), "top10_feature_importance should be list"
        assert len(top10) <= 10, "top10 should have at most 10 entries"
        if top10:
            feat_name, feat_score = top10[0]
            assert isinstance(feat_name, str), "Feature name should be str"
            assert isinstance(feat_score, (int, float, np.integer, np.floating)), \
                "Feature score should be numeric"

        # Model path should exist as a file
        import os
        assert os.path.exists(result['model_path']), \
            f"Model file should exist: {result['model_path']}"

        # models dict should have both targets trained
        assert 'label_5d' in trainer.models, "Should have label_5d model"
        assert 'label_10d' in trainer.models, "Should have label_10d model"

        print(f"\n  test_ic_10d  = {metrics['test_ic_10d']:.4f}")
        print(f"  test_icir_10d= {metrics['test_icir_10d']:.4f}")
        print(f"  gate_pass    = {result['gate_pass']}")
        print(f"  duration     = {result['duration_sec']:.1f}s")
