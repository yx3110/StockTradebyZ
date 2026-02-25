#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alpha158 特征计算模块 (Microsoft Qlib 标准)

158个纯OHLCV特征:
- KBAR (9): 蜡烛图形态特征
- Price (4): 当日价格比率
- Rolling Price (115): 23个算子 × 5个窗口[5,10,20,30,60]
- Rolling Volume (30): 6个算子 × 5个窗口[5,10,20,30,60]

参考: https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

WINDOWS = [5, 10, 20, 30, 60]


class Alpha158FeatureCalculator:
    """Alpha158 特征计算器

    输入: 单只股票的 DataFrame (≥61行, 含 open/high/low/close/volume 列)
    输出: 158个特征值的 dict (最后一行对应的特征)
    """

    # 特征名称列表 (按计算顺序)
    FEATURE_NAMES: List[str] = []

    @classmethod
    def get_feature_names(cls) -> List[str]:
        """返回158个特征名称列表"""
        if cls.FEATURE_NAMES:
            return cls.FEATURE_NAMES

        names = []
        # KBAR (9)
        names.extend(['KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2'])
        # Price (4)
        names.extend(['OPEN0', 'HIGH0', 'LOW0', 'VWAP0'])
        # Rolling Price (115 = 23 operators × 5 windows)
        for w in WINDOWS:
            names.extend([
                f'ROC_{w}', f'MA_{w}', f'STD_{w}', f'BETA_{w}', f'RSQR_{w}', f'RESI_{w}',
                f'MAX_{w}', f'MIN_{w}', f'QTLU_{w}', f'QTLD_{w}',
                f'RANK_{w}', f'RSV_{w}', f'IMAX_{w}', f'IMIN_{w}', f'IMXD_{w}',
                f'CORR_{w}', f'CORD_{w}', f'CNTP_{w}', f'CNTN_{w}', f'CNTD_{w}',
                f'SUMP_{w}', f'SUMN_{w}', f'SUMD_{w}',
            ])
        # Rolling Volume (30 = 6 operators × 5 windows)
        for w in WINDOWS:
            names.extend([
                f'VMA_{w}', f'VSTD_{w}', f'WVMA_{w}',
                f'VSUMP_{w}', f'VSUMN_{w}', f'VSUMD_{w}',
            ])
        cls.FEATURE_NAMES = names
        assert len(names) == 158, f"Expected 158 features, got {len(names)}"
        return names

    @staticmethod
    def compute_features(df: pd.DataFrame) -> Optional[Dict[str, float]]:
        """计算单只股票最后一行的158个Alpha158特征

        Args:
            df: DataFrame with columns [open, high, low, close, volume],
                sorted by date ascending, ≥61行

        Returns:
            dict of {feature_name: value} or None if insufficient data
        """
        if len(df) < 61:
            return None

        o = df['open'].values.astype(np.float64)
        h = df['high'].values.astype(np.float64)
        lo = df['low'].values.astype(np.float64)
        c = df['close'].values.astype(np.float64)
        v = df['volume'].values.astype(np.float64)

        # Current day values
        c_cur = c[-1]
        if c_cur == 0 or np.isnan(c_cur):
            return None

        vwap = (o + h + lo + c) / 4.0  # VWAP近似

        result = {}

        # ====================================================================
        # KBAR (9 features)
        # ====================================================================
        hl = h[-1] - lo[-1]
        hl_safe = max(hl, 1e-12)
        result['KMID'] = (c[-1] - o[-1]) / c_cur
        result['KLEN'] = hl / c_cur
        result['KMID2'] = (c[-1] - o[-1]) / hl_safe
        result['KUP'] = (h[-1] - max(o[-1], c[-1])) / c_cur
        result['KUP2'] = (h[-1] - max(o[-1], c[-1])) / hl_safe
        result['KLOW'] = (min(o[-1], c[-1]) - lo[-1]) / c_cur
        result['KLOW2'] = (min(o[-1], c[-1]) - lo[-1]) / hl_safe
        result['KSFT'] = (2 * c[-1] - h[-1] - lo[-1]) / c_cur
        result['KSFT2'] = (2 * c[-1] - h[-1] - lo[-1]) / hl_safe

        # ====================================================================
        # Price (4 features)
        # ====================================================================
        result['OPEN0'] = o[-1] / c_cur
        result['HIGH0'] = h[-1] / c_cur
        result['LOW0'] = lo[-1] / c_cur
        result['VWAP0'] = vwap[-1] / c_cur

        # ====================================================================
        # Rolling Price (115 features = 23 operators × 5 windows)
        # ====================================================================
        # Precompute log returns for BETA/RSQR/RESI
        log_ret = np.diff(np.log(np.maximum(c, 1e-12)))  # len = N-1

        for w in WINDOWS:
            suffix = f'_{w}'
            window_c = c[-w:]
            window_v = v[-w:]
            window_h = h[-w:]
            window_lo = lo[-w:]
            window_vwap = vwap[-w:]
            window_log_ret = log_ret[-w:]

            # ROC: Rate of Change
            result['ROC' + suffix] = (c[-1] / c[-w - 1] - 1) if c[-w - 1] != 0 else 0.0

            # MA: Moving Average ratio
            ma = np.mean(window_c)
            result['MA' + suffix] = (ma / c_cur - 1) if c_cur != 0 else 0.0

            # STD: Standard deviation of close/close[-1]
            result['STD' + suffix] = np.std(window_c / c_cur) if c_cur != 0 else 0.0

            # BETA, RSQR, RESI: Linear regression of log returns on time
            if len(window_log_ret) >= 2:
                x = np.arange(len(window_log_ret), dtype=np.float64)
                x_mean = np.mean(x)
                y_mean = np.mean(window_log_ret)
                ss_xx = np.sum((x - x_mean) ** 2)
                ss_xy = np.sum((x - x_mean) * (window_log_ret - y_mean))
                beta = ss_xy / ss_xx if ss_xx > 0 else 0.0
                y_hat = beta * (x - x_mean) + y_mean
                ss_res = np.sum((window_log_ret - y_hat) ** 2)
                ss_tot = np.sum((window_log_ret - y_mean) ** 2)
                rsqr = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
                resi = window_log_ret[-1] - y_hat[-1]
            else:
                beta, rsqr, resi = 0.0, 0.0, 0.0
            result['BETA' + suffix] = beta
            result['RSQR' + suffix] = rsqr
            result['RESI' + suffix] = resi

            # MAX, MIN: Max/Min of close in window
            max_c = np.max(window_c)
            min_c = np.min(window_c)
            result['MAX' + suffix] = (max_c / c_cur - 1)
            result['MIN' + suffix] = (min_c / c_cur - 1)

            # QTLU, QTLD: Upper/Lower quantile
            result['QTLU' + suffix] = (np.quantile(window_c, 0.8) / c_cur - 1)
            result['QTLD' + suffix] = (np.quantile(window_c, 0.2) / c_cur - 1)

            # RANK: Percentile rank of current close in window
            result['RANK' + suffix] = np.mean(window_c <= c[-1])

            # RSV: (close - min) / (max - min)
            range_c = max_c - min_c
            result['RSV' + suffix] = (c[-1] - min_c) / range_c if range_c > 1e-12 else 0.5

            # IMAX, IMIN: Position of max/min in window (normalized 0-1)
            imax = np.argmax(window_c) / (w - 1) if w > 1 else 0.5
            imin = np.argmin(window_c) / (w - 1) if w > 1 else 0.5
            result['IMAX' + suffix] = imax
            result['IMIN' + suffix] = imin
            result['IMXD' + suffix] = imax - imin

            # CORR: Correlation of close and log(volume+1)
            log_v = np.log1p(window_v)
            if np.std(window_c) > 1e-12 and np.std(log_v) > 1e-12:
                result['CORR' + suffix] = np.corrcoef(window_c, log_v)[0, 1]
            else:
                result['CORR' + suffix] = 0.0

            # CORD: Correlation of close_return and volume_return
            c_ret = np.diff(window_c)
            v_ret = np.diff(window_v)
            if len(c_ret) >= 2 and np.std(c_ret) > 1e-12 and np.std(v_ret) > 1e-12:
                result['CORD' + suffix] = np.corrcoef(c_ret, v_ret)[0, 1]
            else:
                result['CORD' + suffix] = 0.0

            # CNTP, CNTN, CNTD: Proportion of positive/negative returns
            daily_ret = np.diff(window_c)
            n_days = len(daily_ret)
            if n_days > 0:
                cntp = np.mean(daily_ret > 0)
                cntn = np.mean(daily_ret < 0)
            else:
                cntp, cntn = 0.5, 0.5
            result['CNTP' + suffix] = cntp
            result['CNTN' + suffix] = cntn
            result['CNTD' + suffix] = cntp - cntn

            # SUMP, SUMN, SUMD: Sum of positive/negative returns
            pos_ret = np.sum(daily_ret[daily_ret > 0]) / c_cur if c_cur != 0 else 0.0
            neg_ret = np.sum(daily_ret[daily_ret < 0]) / c_cur if c_cur != 0 else 0.0
            result['SUMP' + suffix] = pos_ret
            result['SUMN' + suffix] = neg_ret
            result['SUMD' + suffix] = pos_ret - neg_ret

        # ====================================================================
        # Rolling Volume (30 features = 6 operators × 5 windows)
        # ====================================================================
        v_cur = max(v[-1], 1e-12)
        for w in WINDOWS:
            suffix = f'_{w}'
            window_v = v[-w:]

            # VMA: Volume moving average ratio
            vma = np.mean(window_v)
            result['VMA' + suffix] = (vma / v_cur - 1) if v_cur > 1e-12 else 0.0

            # VSTD: Volume std ratio
            result['VSTD' + suffix] = np.std(window_v) / v_cur if v_cur > 1e-12 else 0.0

            # WVMA: Weighted volume (by abs return) std / mean
            window_c_for_v = c[-w:]
            abs_ret = np.abs(np.diff(window_c_for_v))
            if len(abs_ret) > 0:
                wv = window_v[1:] * abs_ret  # weighted volume
                wv_mean = np.mean(wv) if np.mean(wv) > 1e-12 else 1e-12
                result['WVMA' + suffix] = np.std(wv) / wv_mean
            else:
                result['WVMA' + suffix] = 0.0

            # VSUMP, VSUMN, VSUMD: Sum of volume on up/down days
            daily_ret = np.diff(window_c_for_v)
            v_daily = window_v[1:]
            if len(daily_ret) > 0:
                up_vol = np.sum(v_daily[daily_ret > 0]) / (np.sum(v_daily) + 1e-12)
                dn_vol = np.sum(v_daily[daily_ret < 0]) / (np.sum(v_daily) + 1e-12)
            else:
                up_vol, dn_vol = 0.5, 0.5
            result['VSUMP' + suffix] = up_vol
            result['VSUMN' + suffix] = dn_vol
            result['VSUMD' + suffix] = up_vol - dn_vol

        # Replace NaN/Inf
        for k in result:
            val = result[k]
            if np.isnan(val) or np.isinf(val):
                result[k] = 0.0

        return result

    @staticmethod
    def compute_features_batch(df: pd.DataFrame) -> pd.DataFrame:
        """批量计算一只股票所有日期的特征 (pandas rolling 向量化版本)

        使用 pandas rolling 做C级别优化, 比纯Python循环快50-100倍。

        Args:
            df: DataFrame with [trade_date, open, high, low, close, volume],
                sorted by date ascending

        Returns:
            DataFrame with trade_date + 158 feature columns (仅包含有足够历史数据的日期)
        """
        if len(df) < 61:
            return pd.DataFrame()

        n = len(df)
        start_idx = 60

        # Series for rolling operations
        cs = pd.Series(df['close'].values, dtype=np.float64)
        os_ = pd.Series(df['open'].values, dtype=np.float64)
        hs = pd.Series(df['high'].values, dtype=np.float64)
        ls = pd.Series(df['low'].values, dtype=np.float64)
        vs = pd.Series(df['volume'].values, dtype=np.float64)
        dates = df['trade_date'].values

        c = cs.values
        o = os_.values
        h = hs.values
        lo = ls.values
        v = vs.values

        vwap_s = (os_ + hs + ls + cs) / 4.0
        log_c = np.log(np.maximum(c, 1e-12))
        # daily log returns (shifted so index aligns: ret[i] = log(c[i]/c[i-1]))
        log_ret_s = cs.pct_change().apply(lambda x: np.log1p(x) if not np.isnan(x) else 0.0)
        log_ret_s.iloc[0] = 0.0
        # daily close diff
        close_diff = cs.diff()
        close_diff.iloc[0] = 0.0
        vol_diff = vs.diff()
        vol_diff.iloc[0] = 0.0

        features = {}

        # ================================================================
        # KBAR (9) - element-wise, fully vectorized
        # ================================================================
        c_safe = cs.replace(0, np.nan)
        hl = hs - ls
        hl_safe = hl.replace(0, 1e-12).where(hl > 0, 1e-12)
        oc_max = pd.concat([os_, cs], axis=1).max(axis=1)
        oc_min = pd.concat([os_, cs], axis=1).min(axis=1)

        features['KMID'] = (cs - os_) / c_safe
        features['KLEN'] = hl / c_safe
        features['KMID2'] = (cs - os_) / hl_safe
        features['KUP'] = (hs - oc_max) / c_safe
        features['KUP2'] = (hs - oc_max) / hl_safe
        features['KLOW'] = (oc_min - ls) / c_safe
        features['KLOW2'] = (oc_min - ls) / hl_safe
        features['KSFT'] = (2 * cs - hs - ls) / c_safe
        features['KSFT2'] = (2 * cs - hs - ls) / hl_safe

        # ================================================================
        # Price (4) - element-wise
        # ================================================================
        features['OPEN0'] = os_ / c_safe
        features['HIGH0'] = hs / c_safe
        features['LOW0'] = ls / c_safe
        features['VWAP0'] = vwap_s / c_safe

        # ================================================================
        # Rolling Price (115) + Rolling Volume (30)
        # ================================================================
        for w in WINDOWS:
            r = cs.rolling(w, min_periods=w)
            vr = vs.rolling(w, min_periods=w)

            # --- ROC: c[t] / c[t-w] - 1 ---
            features[f'ROC_{w}'] = cs / cs.shift(w) - 1

            # --- MA: mean(window) / close - 1 ---
            features[f'MA_{w}'] = r.mean() / c_safe - 1

            # --- STD: std of (window / close) ---
            # Approximate: std(close_window) / close
            features[f'STD_{w}'] = r.std(ddof=0) / c_safe

            # --- BETA, RSQR, RESI: linear regression of log returns on time ---
            lr_r = log_ret_s.rolling(w, min_periods=w)
            # Using closed-form: beta = Cov(x,y)/Var(x), x=0..w-1
            # For fixed x, Var(x) = (w^2-1)/12, Cov(x,y) = mean(x*y) - mean(x)*mean(y)
            # mean(x) = (w-1)/2
            x_mean = (w - 1) / 2.0
            var_x = (w ** 2 - 1) / 12.0
            if var_x > 0:
                # Need sum(i * y_i) for i=0..w-1 in rolling window
                # Create weighted series: multiply by position
                # We compute beta via: sum(y * (x - x_mean)) / (w * var_x)
                # = (sum(x*y) - x_mean * sum(y)) / (w * var_x)
                # sum(x*y) for x=0..w-1 in window ending at t:
                # y values are log_ret[t-w+1], ..., log_ret[t], paired with x=0,...,w-1
                weights = np.arange(w, dtype=np.float64)

                def _rolling_beta_rsqr_resi(vals):
                    if len(vals) < w or np.any(np.isnan(vals)):
                        return (0.0, 0.0, 0.0)
                    y = vals
                    y_mean = np.mean(y)
                    ss_xy = np.sum((weights - x_mean) * (y - y_mean))
                    beta = ss_xy / (w * var_x)
                    y_hat = beta * (weights - x_mean) + y_mean
                    ss_res = np.sum((y - y_hat) ** 2)
                    ss_tot = np.sum((y - y_mean) ** 2)
                    rsqr = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
                    resi = y[-1] - y_hat[-1]
                    return (beta, rsqr, resi)

                beta_arr = np.full(n, np.nan)
                rsqr_arr = np.full(n, np.nan)
                resi_arr = np.full(n, np.nan)
                lr_vals = log_ret_s.values
                for i in range(w - 1, n):
                    window = lr_vals[i - w + 1: i + 1]
                    b, r2, res = _rolling_beta_rsqr_resi(window)
                    beta_arr[i] = b
                    rsqr_arr[i] = r2
                    resi_arr[i] = res
            else:
                beta_arr = np.zeros(n)
                rsqr_arr = np.zeros(n)
                resi_arr = np.zeros(n)
            features[f'BETA_{w}'] = pd.Series(beta_arr)
            features[f'RSQR_{w}'] = pd.Series(rsqr_arr)
            features[f'RESI_{w}'] = pd.Series(resi_arr)

            # --- MAX, MIN ---
            roll_max = r.max()
            roll_min = r.min()
            features[f'MAX_{w}'] = roll_max / c_safe - 1
            features[f'MIN_{w}'] = roll_min / c_safe - 1

            # --- QTLU, QTLD ---
            features[f'QTLU_{w}'] = r.quantile(0.8) / c_safe - 1
            features[f'QTLD_{w}'] = r.quantile(0.2) / c_safe - 1

            # --- RANK: fraction of window values <= current close ---
            def _rank_pct(vals):
                if len(vals) < w:
                    return np.nan
                return np.mean(vals <= vals[-1])
            features[f'RANK_{w}'] = cs.rolling(w, min_periods=w).apply(_rank_pct, raw=True)

            # --- RSV: (close - min) / (max - min) ---
            range_c = roll_max - roll_min
            range_c_safe = range_c.replace(0, 1e-12).where(range_c > 0, 1e-12)
            features[f'RSV_{w}'] = (cs - roll_min) / range_c_safe

            # --- IMAX, IMIN, IMXD ---
            def _imax(vals):
                if len(vals) < w:
                    return np.nan
                return np.argmax(vals) / (w - 1) if w > 1 else 0.5
            def _imin(vals):
                if len(vals) < w:
                    return np.nan
                return np.argmin(vals) / (w - 1) if w > 1 else 0.5
            imax_s = cs.rolling(w, min_periods=w).apply(_imax, raw=True)
            imin_s = cs.rolling(w, min_periods=w).apply(_imin, raw=True)
            features[f'IMAX_{w}'] = imax_s
            features[f'IMIN_{w}'] = imin_s
            features[f'IMXD_{w}'] = imax_s - imin_s

            # --- CORR: corr(close, log(volume+1)) ---
            log_v = np.log1p(vs)
            features[f'CORR_{w}'] = cs.rolling(w, min_periods=w).corr(log_v)

            # --- CORD: corr(close_diff, volume_diff) ---
            # np.diff(wc) has w-1 elements, so use rolling(w-1) on close_diff
            wd = w - 1  # diff-based window size
            features[f'CORD_{w}'] = close_diff.rolling(wd, min_periods=wd).corr(vol_diff)

            # --- CNTP, CNTN, CNTD ---
            pos_flag = (close_diff > 0).astype(float)
            neg_flag = (close_diff < 0).astype(float)
            cntp = pos_flag.rolling(wd, min_periods=wd).mean()
            cntn = neg_flag.rolling(wd, min_periods=wd).mean()
            features[f'CNTP_{w}'] = cntp
            features[f'CNTN_{w}'] = cntn
            features[f'CNTD_{w}'] = cntp - cntn

            # --- SUMP, SUMN, SUMD ---
            pos_ret = close_diff.clip(lower=0)
            neg_ret = close_diff.clip(upper=0)
            sump = pos_ret.rolling(wd, min_periods=wd).sum() / c_safe
            sumn = neg_ret.rolling(wd, min_periods=wd).sum() / c_safe
            features[f'SUMP_{w}'] = sump
            features[f'SUMN_{w}'] = sumn
            features[f'SUMD_{w}'] = sump - sumn

            # ============================================================
            # Rolling Volume (6 per window)
            # ============================================================
            v_safe = vs.replace(0, 1e-12).where(vs > 0, 1e-12)

            # VMA
            features[f'VMA_{w}'] = vr.mean() / v_safe - 1

            # VSTD
            features[f'VSTD_{w}'] = vr.std(ddof=0) / v_safe

            # WVMA: std(volume * |close_diff|) / mean(volume * |close_diff|)
            # Single version uses np.diff(wc) (w-1 elems) paired with volume[1:]
            wv_product = vs * close_diff.abs()
            wvp_r = wv_product.rolling(wd, min_periods=wd)
            wvp_mean = wvp_r.mean().replace(0, 1e-12)
            wvp_mean = wvp_mean.where(wvp_mean.abs() > 1e-12, 1e-12)
            features[f'WVMA_{w}'] = wvp_r.std(ddof=0) / wvp_mean

            # VSUMP, VSUMN, VSUMD: volume on up/down days
            vol_up = (vs * pos_flag)
            vol_dn = (vs * neg_flag)
            # total volume over w-1 days (matching single version's sum(v_d))
            vol_total_wd = vs.rolling(wd, min_periods=wd).sum().replace(0, 1e-12)
            features[f'VSUMP_{w}'] = vol_up.rolling(wd, min_periods=wd).sum() / vol_total_wd
            features[f'VSUMN_{w}'] = vol_dn.rolling(wd, min_periods=wd).sum() / vol_total_wd
            features[f'VSUMD_{w}'] = features[f'VSUMP_{w}'] - features[f'VSUMN_{w}']

        # ================================================================
        # Assemble output
        # ================================================================
        feature_names = Alpha158FeatureCalculator.get_feature_names()
        out_df = pd.DataFrame({name: features[name].values for name in feature_names})
        out_df['trade_date'] = dates

        # Trim to valid range (start_idx onward)
        out_df = out_df.iloc[start_idx:].reset_index(drop=True)

        # Replace NaN/Inf
        out_df[feature_names] = out_df[feature_names].fillna(0.0)
        for col in feature_names:
            arr = out_df[col].values
            arr[np.isinf(arr)] = 0.0
            out_df[col] = arr

        return out_df
