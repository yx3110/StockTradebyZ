from typing import Dict, List, Optional, Any, Union

from scipy.signal import find_peaks
import numpy as np
import pandas as pd


# --------------------------- 通用指标 --------------------------- #

def compute_kdj(df: pd.DataFrame, n: int = 9) -> pd.DataFrame:
    """
    计算KDJ指标 - 知行战法专用版本
    使用SMA方式计算K和D值，完全按照知行战法公式：
    RSV:=(CLOSE-LLV(LOW,9))/(HHV(HIGH,9)-LLV(LOW,9))*100;
    K:=SMA(RSV,3,1);
    D:=SMA(K,3,1);
    J:=3*K-2*D;
    """
    if df.empty:
        return df.assign(K=np.nan, D=np.nan, J=np.nan)

    # 计算RSV
    low_n = df["low"].rolling(window=n, min_periods=1).min()  # LLV(LOW,9)
    high_n = df["high"].rolling(window=n, min_periods=1).max() # HHV(HIGH,9)
    rsv = (df["close"] - low_n) / (high_n - low_n + 1e-9) * 100

    # 使用SMA方式计算K和D (SMA(X,N,1)相当于简单移动平均)
    # SMA(RSV,3,1) 表示RSV的3周期简单移动平均
    K = rsv.rolling(window=3, min_periods=1).mean()
    D = K.rolling(window=3, min_periods=1).mean()
    
    # 计算J值：J = 3*K - 2*D
    J = 3 * K - 2 * D
    
    return df.assign(K=K, D=D, J=J)


def compute_bbi(df: pd.DataFrame) -> pd.Series:
    ma3 = df["close"].rolling(3).mean()
    ma6 = df["close"].rolling(6).mean()
    ma12 = df["close"].rolling(12).mean()
    ma24 = df["close"].rolling(24).mean()
    return (ma3 + ma6 + ma12 + ma24) / 4


def compute_rsv(
    df: pd.DataFrame,
    n: int,
) -> pd.Series:
    """
    按公式：RSV(N) = 100 × (C - LLV(L,N)) ÷ (HHV(C,N) - LLV(L,N))
    - C 用收盘价最高值 (HHV of close)
    - L 用最低价最低值 (LLV of low)
    """
    low_n = df["low"].rolling(window=n, min_periods=1).min()
    high_close_n = df["close"].rolling(window=n, min_periods=1).max()
    rsv = (df["close"] - low_n) / (high_close_n - low_n + 1e-9) * 100.0
    return rsv


def compute_dif(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
    """计算 MACD 指标中的 DIF (EMA fast - EMA slow)。"""
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


def compute_zx_lines(
    df: pd.DataFrame,
    m1: int = 14, m2: int = 28, m3: int = 57, m4: int = 114
) -> tuple:
    """返回 (ZXDQ, ZXDKX)
    ZXDQ = EMA(EMA(C,10),10) - 知行短期线
    ZXDKX = (MA(C,14)+MA(C,28)+MA(C,57)+MA(C,114))/4 - 知行长期线（多空线）
    """
    close = df["close"].astype(float)
    zxdq = close.ewm(span=10, adjust=False).mean().ewm(span=10, adjust=False).mean()

    ma1 = close.rolling(window=m1, min_periods=m1).mean()
    ma2 = close.rolling(window=m2, min_periods=m2).mean()
    ma3 = close.rolling(window=m3, min_periods=m3).mean()
    ma4 = close.rolling(window=m4, min_periods=m4).mean()
    zxdkx = (ma1 + ma2 + ma3 + ma4) / 4.0
    return zxdq, zxdkx


def bbi_deriv_uptrend(
    bbi: pd.Series,
    *,
    min_window: int,
    max_window: Union[int, None] = None,
    q_threshold: float = 0.0,
) -> bool:
    """
    判断 BBI 是否“整体上升”。

    令最新交易日为 T，在区间 [T-w+1, T]（w 自适应，w ≥ min_window 且 ≤ max_window）
    内，先将 BBI 归一化：BBI_norm(t) = BBI(t) / BBI(T-w+1)。

    再计算一阶差分 Δ(t) = BBI_norm(t) - BBI_norm(t-1)。  
    若 Δ(t) 的前 q_threshold 分位数 ≥ 0，则认为该窗口通过；只要存在
    **最长** 满足条件的窗口即可返回 True。q_threshold=0 时退化为
    “全程单调不降”（旧版行为）。

    Parameters
    ----------
    bbi : pd.Series
        BBI 序列（最新值在最后一位）。
    min_window : int
        检测窗口的最小长度。
    max_window : int | None
        检测窗口的最大长度；None 表示不设上限。
    q_threshold : float, default 0.0
        允许一阶差分为负的比例（0 ≤ q_threshold ≤ 1）。
    """
    if not 0.0 <= q_threshold <= 1.0:
        raise ValueError("q_threshold 必须位于 [0, 1] 区间内")

    bbi = bbi.dropna()
    if len(bbi) < min_window:
        return False

    longest = min(len(bbi), max_window or len(bbi))

    # 自最长窗口向下搜索，找到任一满足条件的区间即通过
    for w in range(longest, min_window - 1, -1):
        seg = bbi.iloc[-w:]                # 区间 [T-w+1, T]
        norm = seg / seg.iloc[0]           # 归一化
        diffs = np.diff(norm.values)       # 一阶差分
        if np.quantile(diffs, q_threshold) >= 0:
            return True
    return False


def _find_peaks(
    df: pd.DataFrame,
    *,
    column: str = "high",
    distance: Optional[int] = None,
    prominence: Optional[float] = None,
    height: Optional[float] = None,
    width: Optional[float] = None,
    rel_height: float = 0.5,
    **kwargs: Any,
) -> pd.DataFrame:
    
    if column not in df.columns:
        raise KeyError(f"'{column}' not found in DataFrame columns: {list(df.columns)}")

    y = df[column].to_numpy()

    indices, props = find_peaks(
        y,
        distance=distance,
        prominence=prominence,
        height=height,
        width=width,
        rel_height=rel_height,
        **kwargs,
    )

    peaks_df = df.iloc[indices].copy()
    peaks_df["is_peak"] = True

    # Flatten SciPy arrays into columns (only those with same length as indices)
    for key, arr in props.items():
        if isinstance(arr, (list, np.ndarray)) and len(arr) == len(indices):
            peaks_df[f"peak_{key}"] = arr

    return peaks_df


# --------------------------- Selector 类 --------------------------- #
class BBIKDJSelector:
    """
    自适应 *BBI(导数)* + *KDJ* 选股器
        • BBI: 允许 bbi_q_threshold 比例的回撤
        • KDJ: J < threshold ；或位于历史 J 的 j_q_threshold 分位及以下
        • MACD: DIF > 0
        • 收盘价波动幅度 ≤ price_range_pct
    """

    def __init__(
        self,
        j_threshold: float = -5,
        bbi_min_window: int = 90,
        max_window: int = 90,
        price_range_pct: float = 100.0,
        bbi_q_threshold: float = 0.05,
        j_q_threshold: float = 0.10,
    ) -> None:
        self.j_threshold = j_threshold
        self.bbi_min_window = bbi_min_window
        self.max_window = max_window
        self.price_range_pct = price_range_pct
        self.bbi_q_threshold = bbi_q_threshold  # ← 原 q_threshold
        self.j_q_threshold = j_q_threshold      # ← 新增

    # ---------- 单支股票过滤 ---------- #
    def _passes_filters(self, hist: pd.DataFrame) -> bool:
        hist = hist.copy()
        if "BBI" not in hist.columns:
            hist["BBI"] = compute_bbi(hist)

        # 0. 收盘价波动幅度约束（最近 max_window 根 K 线）
        win = hist.tail(self.max_window)
        high, low = win["close"].max(), win["close"].min()
        if low <= 0 or (high / low - 1) > self.price_range_pct:
            return False

        # 1. BBI 上升（允许部分回撤）
        if not bbi_deriv_uptrend(
            hist["BBI"],
            min_window=self.bbi_min_window,
            max_window=self.max_window,
            q_threshold=self.bbi_q_threshold,
        ):
            return False

        # 2. KDJ 过滤 —— 双重条件
        if "J" not in hist.columns:
            hist = compute_kdj(hist)
        j_today = float(hist.iloc[-1]["J"])

        # 最近 max_window 根 K 线的 J 分位
        j_window = hist["J"].tail(self.max_window).dropna()
        if j_window.empty:
            return False
        j_quantile = float(j_window.quantile(self.j_q_threshold))

        if not (j_today < self.j_threshold or j_today <= j_quantile):
            return False

        # 3. MACD：DIF > 0
        if "DIF" not in hist.columns:
            hist["DIF"] = compute_dif(hist)
        return hist["DIF"].iloc[-1] > 0

    # ---------- 多股票批量 ---------- #
    def select(
        self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]
    ) -> List[str]:
        picks: List[str] = []
        for code, df in data.items():
            hist = df[df["date"] <= date]
            if hist.empty:
                continue
            # 额外预留 20 根 K 线缓冲
            hist = hist.tail(self.max_window + 20)
            if self._passes_filters(hist):
                picks.append(code)
        return picks


class SuperB1Selector:
    """SuperB1 选股器

    过滤逻辑概览
    ----------------
    1. **历史匹配 (t_m)** — 在 *lookback_n* 个交易日窗口内，至少存在一日
       满足 :class:`BBIKDJSelector`。

    2. **盘整区间** — 区间 ``[t_m, date-1]`` 收盘价波动率不超过 ``close_vol_pct``。

    3. **当日下跌** — ``(close_{date-1} - close_date) / close_{date-1}``
       ≥ ``price_drop_pct``。

    4. **J 值极低** — ``J < j_threshold`` *或* 位于历史 ``j_q_threshold`` 分位。
    """

    # ---------------------------------------------------------------------
    # 构造函数
    # ---------------------------------------------------------------------
    def __init__(
        self,
        *,
        lookback_n: int = 60,
        close_vol_pct: float = 0.05,
        price_drop_pct: float = 0.03,
        j_threshold: float = -5,
        j_q_threshold: float = 0.10,
        # ↓↓↓ 新增：嵌套 BBIKDJSelector 配置
        B1_params: Optional[Dict[str, Any]] = None        
    ) -> None:        
        # ---------- 参数合法性检查 ----------
        if lookback_n < 2:
            raise ValueError("lookback_n 应 ≥ 2")
        if not (0 < close_vol_pct < 1):
            raise ValueError("close_vol_pct 应位于 (0, 1) 区间")
        if not (0 < price_drop_pct < 1):
            raise ValueError("price_drop_pct 应位于 (0, 1) 区间")
        if not (0 <= j_q_threshold <= 1):
            raise ValueError("j_q_threshold 应位于 [0, 1] 区间")
        if B1_params is None:
            raise ValueError("bbi_params没有给出")

        # ---------- 基本参数 ----------
        self.lookback_n = lookback_n
        self.close_vol_pct = close_vol_pct
        self.price_drop_pct = price_drop_pct
        self.j_threshold = j_threshold
        self.j_q_threshold = j_q_threshold

        # ---------- 内部 BBIKDJSelector ----------
        self.bbi_selector = BBIKDJSelector(**(B1_params or {}))

        # 为保证给 BBIKDJSelector 提供足够历史，预留额外缓冲
        self._extra_for_bbi = self.bbi_selector.max_window + 20

    # 单支股票过滤核心
    def _passes_filters(self, hist: pd.DataFrame) -> bool:
        """*hist* 必须按日期升序，且最后一行为目标 *date*。"""
        if len(hist) < 2:
            return False

        # ---------- Step-0: 数据量判断 ----------
        if len(hist) < self.lookback_n + self._extra_for_bbi:
            return False

        # ---------- Step-1: 搜索满足 BBIKDJ 的 t_m ----------
        lb_hist = hist.tail(self.lookback_n + 1)  # +1 以排除自身
        tm_idx: int | None = None
        # 遍历回溯窗口
        for idx in lb_hist.index[:-1]:            
            if self.bbi_selector._passes_filters(hist.loc[:idx]):
                tm_idx = idx
                stable_seg = hist.loc[tm_idx : hist.index[-2], "close"]
                if len(stable_seg) < 3:
                    tm_idx = None
                    break
                high, low = stable_seg.max(), stable_seg.min()
                if low <= 0 or (high / low - 1) > self.close_vol_pct:                                      
                    tm_idx = None
                    continue
                else:
                    break
        if tm_idx is None:            
            return False        
        

        # ---------- Step-3: 当日相对前一日跌幅 ----------
        close_today, close_prev = hist["close"].iloc[-1], hist["close"].iloc[-2]
        if close_prev <= 0 or (close_prev - close_today) / close_prev < self.price_drop_pct:            
            return False

        # ---------- Step-4: J 值极低 ----------
        if "J" not in hist.columns:
            hist = compute_kdj(hist)
        j_today = float(hist["J"].iloc[-1])
        j_window = hist["J"].iloc[-self.lookback_n:].dropna()
        j_q_val = float(j_window.quantile(self.j_q_threshold)) if not j_window.empty else np.nan
        if not (j_today < self.j_threshold or j_today <= j_q_val):
            return False

        return True

    # 批量选股接口
    def select(self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> List[str]:        
        picks: List[str] = []
        min_len = self.lookback_n + self._extra_for_bbi

        for code, df in data.items():
            hist = df[df["date"] <= date].tail(min_len)
            if len(hist) < min_len:
                continue
            if self._passes_filters(hist):
                picks.append(code)

        return picks


class PeakKDJSelector:
    """
    Peaks + KDJ 选股器    
    """

    def __init__(
        self,
        j_threshold: float = -5,
        max_window: int = 90,
        fluc_threshold: float = 0.03,
        gap_threshold: float = 0.02,
        j_q_threshold: float = 0.10,
    ) -> None:
        self.j_threshold = j_threshold
        self.max_window = max_window
        self.fluc_threshold = fluc_threshold  # 当日↔peak_(t-n) 波动率上限
        self.gap_threshold = gap_threshold    # oc_prev 必须高于区间最低收盘价的比例
        self.j_q_threshold = j_q_threshold

    # ---------- 单支股票过滤 ---------- #
        # ---------- 单支股票过滤 ---------- #
    def _passes_filters(self, hist: pd.DataFrame) -> bool:
        if hist.empty:
            return False

        hist = hist.copy().sort_values("date")
        hist["oc_max"] = hist[["open", "close"]].max(axis=1)

        # 1. 提取 peaks
        peaks_df = _find_peaks(
            hist,
            column="oc_max",
            distance=6,
            prominence=0.5,
        )
        
        # 至少两个峰      
        date_today = hist.iloc[-1]["date"]
        peaks_df = peaks_df[peaks_df["date"] < date_today]
        if len(peaks_df) < 2:               
            return False

        peak_t = peaks_df.iloc[-1]          # 最新一个峰
        peaks_list = peaks_df.reset_index(drop=True)
        oc_t = peak_t.oc_max
        total_peaks = len(peaks_list)

        # 2. 回溯寻找 peak_(t-n)
        target_peak = None        
        for idx in range(total_peaks - 2, -1, -1):
            peak_prev = peaks_list.loc[idx]
            oc_prev = peak_prev.oc_max
            if oc_t <= oc_prev:             # 要求 peak_t > peak_(t-n)
                continue

            # 只有当“总峰数 ≥ 3”时才检查区间内其他峰 oc_max
            if total_peaks >= 3 and idx < total_peaks - 2:
                inter_oc = peaks_list.loc[idx + 1 : total_peaks - 2, "oc_max"]
                if not (inter_oc < oc_prev).all():
                    continue

            # 新增： oc_prev 高于区间最低收盘价 gap_threshold
            date_prev = peak_prev.date
            mask = (hist["date"] > date_prev) & (hist["date"] < peak_t.date)
            min_close = hist.loc[mask, "close"].min()
            if pd.isna(min_close):
                continue                    # 区间无数据
            if oc_prev <= min_close * (1 + self.gap_threshold):
                continue

            target_peak = peak_prev
            
            break

        if target_peak is None:
            return False

        # 3. 当日收盘价波动率
        close_today = hist.iloc[-1]["close"]
        fluc_pct = abs(close_today - target_peak.close) / target_peak.close
        if fluc_pct > self.fluc_threshold:
            return False

        # 4. KDJ 过滤
        if "J" not in hist.columns:
            hist = compute_kdj(hist)
        j_today = float(hist.iloc[-1]["J"])
        j_window = hist["J"].tail(self.max_window).dropna()
        if j_window.empty:
            return False
        j_quantile = float(j_window.quantile(self.j_q_threshold))
        if not (j_today < self.j_threshold or j_today <= j_quantile):
            return False

        return True

    # ---------- 多股票批量 ---------- #
    def select(
        self,
        date: pd.Timestamp,
        data: Dict[str, pd.DataFrame],
    ) -> List[str]:
        picks: List[str] = []
        for code, df in data.items():
            hist = df[df["date"] <= date]
            if hist.empty:
                continue
            hist = hist.tail(self.max_window + 20)  # 额外缓冲
            if self._passes_filters(hist):
                picks.append(code)
        return picks
    

class BBIShortLongSelector:
    """
    BBI 上升 + 短/长期 RSV 条件 + DIF > 0 选股器
    """
    def __init__(
        self,
        n_short: int = 3,
        n_long: int = 21,
        m: int = 3,
        bbi_min_window: int = 90,
        max_window: int = 150,
        bbi_q_threshold: float = 0.05,
    ) -> None:
        if m < 2:
            raise ValueError("m 必须 ≥ 2")
        self.n_short = n_short
        self.n_long = n_long
        self.m = m
        self.bbi_min_window = bbi_min_window
        self.max_window = max_window
        self.bbi_q_threshold = bbi_q_threshold   # 新增参数

    # ---------- 单支股票过滤 ---------- #
    def _passes_filters(self, hist: pd.DataFrame) -> bool:
        hist = hist.copy()
        if "BBI" not in hist.columns:
            hist["BBI"] = compute_bbi(hist)

        # 1. BBI 上升（允许部分回撤）
        if not bbi_deriv_uptrend(
            hist["BBI"],
            min_window=self.bbi_min_window,
            max_window=self.max_window,
            q_threshold=self.bbi_q_threshold,
        ):
            return False

        # 2. 计算短/长期 RSV -----------------
        # RSV参数因策略而异，不做预计算guard
        hist["RSV_short"] = compute_rsv(hist, self.n_short)
        hist["RSV_long"] = compute_rsv(hist, self.n_long)

        if len(hist) < self.m:
            return False                        # 数据不足

        win = hist.iloc[-self.m :]              # 最近 m 天
        long_ok = (win["RSV_long"] >= 80).all() # 长期 RSV 全 ≥ 80

        short_series = win["RSV_short"]
        short_start_end_ok = (
            short_series.iloc[0] >= 80 and short_series.iloc[-1] >= 80
        )
        short_has_below_20 = (short_series < 20).any()

        if not (long_ok and short_start_end_ok and short_has_below_20):
            return False

        # 3. MACD：DIF > 0 -------------------
        if "DIF" not in hist.columns:
            hist["DIF"] = compute_dif(hist)
        return hist["DIF"].iloc[-1] > 0

    # ---------- 多股票批量 ---------- #
    def select(
        self,
        date: pd.Timestamp,
        data: Dict[str, pd.DataFrame],
    ) -> List[str]:
        picks: List[str] = []
        for code, df in data.items():
            hist = df[df["date"] <= date]
            if hist.empty:
                continue
            # 预留足够长度：RSV 计算窗口 + BBI 检测窗口 + m
            need_len = (
                max(self.n_short, self.n_long)
                + self.bbi_min_window
                + self.m
            )
            hist = hist.tail(max(need_len, self.max_window))
            if self._passes_filters(hist):
                picks.append(code)
        return picks


class BreakoutVolumeKDJSelector:
    """
    放量突破 + KDJ + DIF>0 + 收盘价波动幅度 选股器   
    """

    def __init__(
        self,
        j_threshold: float = 0.0,
        up_threshold: float = 3.0,
        volume_threshold: float = 2.0 / 3,
        offset: int = 15,
        max_window: int = 120,
        price_range_pct: float = 10.0,
        j_q_threshold: float = 0.10,        # ← 新增
    ) -> None:
        self.j_threshold = j_threshold
        self.up_threshold = up_threshold
        self.volume_threshold = volume_threshold
        self.offset = offset
        self.max_window = max_window
        self.price_range_pct = price_range_pct
        self.j_q_threshold = j_q_threshold  # ← 新增

    # ---------- 单支股票过滤 ---------- #
    def _passes_filters(self, hist: pd.DataFrame) -> bool:
        if len(hist) < self.offset + 2:
            return False

        hist = hist.tail(self.max_window).copy()

        # ---- 收盘价波动幅度约束 ----
        high, low = hist["close"].max(), hist["close"].min()
        if low <= 0 or (high / low - 1) > self.price_range_pct:
            return False

        # ---- 技术指标 ----
        if "J" not in hist.columns:
            hist = compute_kdj(hist)
        hist["pct_chg"] = hist["close"].pct_change() * 100
        if "DIF" not in hist.columns:
            hist["DIF"] = compute_dif(hist)

        # 0) 指定日约束：J < j_threshold 或位于历史分位；且 DIF > 0
        j_today = float(hist["J"].iloc[-1])

        j_window = hist["J"].tail(self.max_window).dropna()
        if j_window.empty:
            return False
        j_quantile = float(j_window.quantile(self.j_q_threshold))

        # 若不满足任一 J 条件，则淘汰
        if not (j_today < self.j_threshold or j_today <= j_quantile):
            return False
        if hist["DIF"].iloc[-1] <= 0:
            return False

        # ---- 放量突破条件 ----
        n = len(hist)
        wnd_start = max(0, n - self.offset - 1)
        last_idx = n - 1

        for t_idx in range(wnd_start, last_idx):  # 探索突破日 T
            row = hist.iloc[t_idx]

            # 1) 单日涨幅
            if row["pct_chg"] < self.up_threshold:
                continue

            # 2) 相对放量
            vol_T = row["volume"]
            if vol_T <= 0:
                continue
            vols_except_T = hist["volume"].drop(index=hist.index[t_idx])
            if not (vols_except_T <= self.volume_threshold * vol_T).all():
                continue

            # 3) 创新高
            if row["close"] <= hist["close"].iloc[:t_idx].max():
                continue

            # 4) T 之后 J 值维持高位
            if not (hist["J"].iloc[t_idx:last_idx] > hist["J"].iloc[-1] - 10).all():
                continue

            return True  # 满足所有条件

        return False

    # ---------- 多股票批量 ---------- #
    def select(
        self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]
    ) -> List[str]:
        picks: List[str] = []
        for code, df in data.items():
            hist = df[df["date"] <= date]
            if hist.empty:
                continue
            if self._passes_filters(hist):
                picks.append(code)
        return picks


class ZhiXingSelector:
    """
    知行选股策略
    基于KDJ指标和知行趋势线的选股器

    选股逻辑：
    1. KDJ指标中的J值小于5
    2. 涨幅在-1%到1%之间
    3. 振幅小于4%
    4. 短期趋势线(EMA(EMA(CLOSE,10),10)) > 多空线(MA14+MA28+MA57+MA114的平均)
    5. 收盘价 > 多空线 * 100%
    """

    def __init__(
        self,
        j_threshold: float = 5.0,
        min_change_pct: float = -1.0,
        max_change_pct: float = 1.0,
        max_amplitude_pct: float = 4.0,
        close_threshold_pct: float = 100.0,
        max_window: int = 120,
    ) -> None:
        self.j_threshold = j_threshold
        self.min_change_pct = min_change_pct / 100  # 转换为小数
        self.max_change_pct = max_change_pct / 100  # 转换为小数
        self.max_amplitude_pct = max_amplitude_pct / 100  # 转换为小数
        self.close_threshold_pct = close_threshold_pct / 100  # 转换为小数
        self.max_window = max_window

    def _compute_zhixing_indicators(self, hist: pd.DataFrame) -> pd.DataFrame:
        """计算知行指标"""
        hist = hist.copy()

        # 计算知行短期趋势线：EMA(EMA(CLOSE, 10), 10)
        ema10 = hist["close"].ewm(span=10, adjust=False).mean()
        hist["zhixing_short_trend"] = ema10.ewm(span=10, adjust=False).mean()

        # 计算知行多空线：(MA14 + MA28 + MA57 + MA114) / 4
        ma14 = hist["close"].rolling(window=14, min_periods=1).mean()
        ma28 = hist["close"].rolling(window=28, min_periods=1).mean()
        ma57 = hist["close"].rolling(window=57, min_periods=1).mean()
        ma114 = hist["close"].rolling(window=114, min_periods=1).mean()
        hist["zhixing_multi_kong"] = (ma14 + ma28 + ma57 + ma114) / 4

        return hist

    # ---------- 单支股票过滤 ---------- #
    def _passes_filters(self, hist: pd.DataFrame) -> bool:
        if len(hist) < 120:  # 需要足够的历史数据计算MA114
            return False

        hist = hist.tail(min(self.max_window, len(hist))).copy()

        if len(hist) < 2:  # 至少需要2天数据计算涨幅
            return False

        # 计算KDJ指标
        if "J" not in hist.columns:
            hist = compute_kdj(hist)

        # 计算知行指标（复用预计算的ZXDQ/ZXDKX）
        if "ZXDQ" in hist.columns and "ZXDKX" in hist.columns:
            hist["zhixing_short_trend"] = hist["ZXDQ"]
            hist["zhixing_multi_kong"] = hist["ZXDKX"]
        else:
            hist = self._compute_zhixing_indicators(hist)

        # 获取当日和前日数据
        today_data = hist.iloc[-1]
        prev_data = hist.iloc[-2] if len(hist) > 1 else today_data

        # 条件1：J值小于5
        j_today = float(today_data["J"])
        if j_today >= self.j_threshold:
            return False

        # 条件2：涨幅 > -1% 且 < 1% (严格不等式)
        price_change_ratio = today_data["close"] / prev_data["close"]
        if not (price_change_ratio > (1 + self.min_change_pct) and price_change_ratio < (1 + self.max_change_pct)):
            return False

        # 条件3：振幅小于4%
        amplitude = (today_data["high"] - today_data["low"]) / prev_data["close"]
        if amplitude >= self.max_amplitude_pct:
            return False

        # 条件4：短期趋势线 > 多空线
        short_trend = today_data["zhixing_short_trend"]
        multi_kong = today_data["zhixing_multi_kong"]
        if pd.isna(short_trend) or pd.isna(multi_kong) or short_trend <= multi_kong:
            return False

        # 条件5：收盘价 > 多空线 * 100%
        close_price = today_data["close"]
        threshold_price = multi_kong * self.close_threshold_pct
        if close_price <= threshold_price:
            return False

        return True

    # ---------- 多股票批量 ---------- #
    def select(
        self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]
    ) -> List[str]:
        picks: List[str] = []
        for code, df in data.items():
            hist = df[df["date"] <= date]
            if hist.empty:
                continue
            if self._passes_filters(hist):
                picks.append(code)
        return picks


class MA60CrossVolumeWaveSelector:
    """
    上穿60放量战法

    核心逻辑：
    1. 当日 J < j_threshold 或 ≤ j_q_threshold 分位
    2. 最近 lookback_n 内存在有效上穿 MA60
    3. 上穿日 T 到当日区间内 High 最大日作为 Tmax，定义上涨波段 [T, Tmax]，
       其平均成交量 ≥ vol_multiple × 上穿前等长或截断窗口的平均量
    4. MA60 的最近 ma60_slope_days 日回归斜率 > 0
    5. 知行当日约束：收盘 > 长期线 且 短期线 > 长期线
    """

    def __init__(
        self,
        lookback_n: int = 25,
        vol_multiple: float = 1.8,
        j_threshold: float = 15,
        j_q_threshold: float = 0.10,
        ma60_slope_days: int = 5,
        max_window: int = 120,
    ) -> None:
        self.lookback_n = lookback_n
        self.vol_multiple = vol_multiple
        self.j_threshold = j_threshold
        self.j_q_threshold = j_q_threshold
        self.ma60_slope_days = ma60_slope_days
        self.max_window = max_window

    # ---------- 单支股票过滤 ---------- #
    def _passes_filters(self, hist: pd.DataFrame) -> bool:
        # 需要足够的历史数据
        min_required = max(self.lookback_n + 60, 120, self.ma60_slope_days + 60)
        if len(hist) < min_required:
            return False

        hist = hist.copy()

        # 计算MA60
        if "MA60" not in hist.columns:
            hist["MA60"] = hist["close"].rolling(window=60, min_periods=60).mean()

        # 1. 当日J值约束
        if "J" not in hist.columns:
            hist = compute_kdj(hist)
        j_today = float(hist["J"].iloc[-1])

        # 计算J值分位数（最近max_window根K线）
        j_window = kdj["J"].tail(self.max_window).dropna()
        if j_window.empty:
            return False
        j_quantile = float(j_window.quantile(self.j_q_threshold))

        if not (j_today < self.j_threshold or j_today <= j_quantile):
            return False

        # 2. 检测上穿MA60（最近lookback_n天内）
        # 获取最近lookback_n + 1天的数据（+1是为了检测上穿需要前一天数据）
        lookback_data = hist.tail(self.lookback_n + 1)

        cross_idx = None
        cross_global_idx = None
        for i in range(len(lookback_data) - 1):
            prev_close = lookback_data["close"].iloc[i]
            prev_ma60 = lookback_data["MA60"].iloc[i]
            curr_close = lookback_data["close"].iloc[i + 1]
            curr_ma60 = lookback_data["MA60"].iloc[i + 1]

            # 检测上穿：前一天收盘 <= MA60，当天收盘 > MA60
            if pd.notna(prev_ma60) and pd.notna(curr_ma60):
                if prev_close <= prev_ma60 and curr_close > curr_ma60:
                    cross_idx = i + 1
                    # 记录在全局hist中的索引位置
                    cross_global_idx = lookback_data.index[i + 1]
                    break

        if cross_idx is None or cross_global_idx is None:
            return False

        # 3. 上涨波段放量验证
        # 找到上穿日到当日区间内High最大的日子Tmax
        wave_segment = hist.loc[cross_global_idx:]
        if wave_segment.empty:
            return False

        tmax_idx = wave_segment["high"].idxmax()

        # 定义上涨波段 [T, Tmax]
        wave_data = hist.loc[cross_global_idx:tmax_idx]
        if wave_data.empty or wave_data["volume"].isna().all():
            return False

        wave_avg_vol = wave_data["volume"].mean()

        # 计算上穿前等长或截断窗口的平均量
        wave_len = len(wave_data)
        cross_pos_in_hist = hist.index.get_loc(cross_global_idx)

        # 向前取wave_len个交易日（如果不够就截断）
        pre_cross_start_pos = max(0, cross_pos_in_hist - wave_len)
        pre_cross_end_pos = cross_pos_in_hist - 1

        if pre_cross_end_pos < 0:
            return False

        pre_wave_data = hist.iloc[pre_cross_start_pos:pre_cross_end_pos + 1]
        if pre_wave_data.empty or pre_wave_data["volume"].isna().all():
            return False

        pre_wave_avg_vol = pre_wave_data["volume"].mean()

        if pre_wave_avg_vol <= 0 or wave_avg_vol < self.vol_multiple * pre_wave_avg_vol:
            return False

        # 4. MA60趋势约束（最近ma60_slope_days日回归斜率 > 0）
        ma60_recent = hist["MA60"].tail(self.ma60_slope_days).dropna()
        if len(ma60_recent) < self.ma60_slope_days:
            return False

        # 使用numpy进行线性回归计算斜率
        x = np.arange(len(ma60_recent))
        y = ma60_recent.values
        slope = np.polyfit(x, y, 1)[0]

        if slope <= 0:
            return False

        # 5. 知行当日约束：收盘 > 长期线 且 短期线 > 长期线
        if "ZXDQ" in hist.columns and "ZXDKX" in hist.columns:
            zhixing_short_trend = hist["ZXDQ"]
            zhixing_multi_kong = hist["ZXDKX"]
        else:
            # 短期趋势线：EMA(EMA(CLOSE, 10), 10)
            ema10 = hist["close"].ewm(span=10, adjust=False).mean()
            zhixing_short_trend = ema10.ewm(span=10, adjust=False).mean()

            # 多空线：(MA14 + MA28 + MA57 + MA114) / 4
            ma14 = hist["close"].rolling(window=14, min_periods=1).mean()
            ma28 = hist["close"].rolling(window=28, min_periods=1).mean()
            ma57 = hist["close"].rolling(window=57, min_periods=1).mean()
            ma114 = hist["close"].rolling(window=114, min_periods=1).mean()
            zhixing_multi_kong = (ma14 + ma28 + ma57 + ma114) / 4

        short_trend_today = zhixing_short_trend.iloc[-1]
        multi_kong_today = zhixing_multi_kong.iloc[-1]
        close_today = hist["close"].iloc[-1]

        if pd.isna(short_trend_today) or pd.isna(multi_kong_today):
            return False

        if not (close_today > multi_kong_today and short_trend_today > multi_kong_today):
            return False

        return True

    # ---------- 多股票批量 ---------- #
    def select(
        self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]
    ) -> List[str]:
        picks: List[str] = []
        for code, df in data.items():
            hist = df[df["date"] <= date]
            if hist.empty:
                continue
            # 预留足够长度：max_window + 60 (MA60计算) + 20 (额外缓冲)
            hist = hist.tail(self.max_window + 80)
            if self._passes_filters(hist):
                picks.append(code)
        return picks


class BigBullishVolumeSelector:
    """
    暴力K战法选股器

    核心逻辑：捕捉近期出现强势大阳线放量但仍贴近短期均线的股票

    四大筛选条件：
    1. 长阳K线：当日涨幅 > up_pct_threshold (默认4%)
    2. 上影线控制：上影线比例 < upper_wick_pct_max (默认50%)，确保K线"干净"
    3. 放量验证：当日成交量 > 前n日均量 * vol_multiple (默认1.5倍)
    4. 短期线偏离约束：收盘价 < 知行短期线 * close_lt_zxdq_mult，避免追高
    """

    def __init__(
        self,
        *,
        up_pct_threshold: float = 0.04,       # 长阳阈值：例如 0.04 表示涨幅>4%
        upper_wick_pct_max: float = 0.5,      # 上影线比例上限
        vol_lookback_n: int = 20,             # 放量比较的历史天数
        vol_multiple: float = 1.5,            # 放量倍数阈值
        min_history: Optional[int] = None,    # 最少历史长度（默认自动 = vol_lookback_n + 2）
        require_bullish_close: bool = True,   # 可选：要求当日收阳（close >= open）
        ignore_zero_volume: bool = True,      # 计算均量时是否忽略 volume=0
        close_lt_zxdq_mult: float = 1.0       # 例如 1.0 表示 close < zxdq；1.02 表示 close < 1.02*zxdq
    ) -> None:
        if up_pct_threshold <= 0:
            raise ValueError("up_pct_threshold 应 > 0")
        if upper_wick_pct_max < 0:
            raise ValueError("upper_wick_pct_max 应 >= 0")
        if vol_lookback_n < 1:
            raise ValueError("vol_lookback_n 应 >= 1")
        if vol_multiple <= 0:
            raise ValueError("vol_multiple 应 > 0")
        if close_lt_zxdq_mult <= 0:
            raise ValueError("close_lt_zxdq_mult 应 > 0")

        self.up_pct_threshold = float(up_pct_threshold)
        self.upper_wick_pct_max = float(upper_wick_pct_max)
        self.vol_lookback_n = int(vol_lookback_n)
        self.vol_multiple = float(vol_multiple)
        self.require_bullish_close = bool(require_bullish_close)
        self.ignore_zero_volume = bool(ignore_zero_volume)
        self.close_lt_zxdq_mult = float(close_lt_zxdq_mult)
        self.eps = float(1e-12)
        self.min_history = int(min_history) if min_history is not None else (self.vol_lookback_n + 2)

    @staticmethod
    def _to_float(x) -> float:
        try:
            return float(x)
        except Exception:
            return float("nan")

    def _upper_wick_pct(self, o: float, h: float, c: float) -> float:
        """计算上影线比例"""
        return (h - max(o, c)) / max(o, c)

    def _passes_filters(self, hist: pd.DataFrame) -> bool:
        if hist is None or hist.empty:
            return False

        hist = hist.sort_values("date").copy()

        if len(hist) < self.min_history:
            return False
        if len(hist) < (self.vol_lookback_n + 2):
            return False  # 至少需要：T、T-1、以及 T-1 往前 n 天

        today = hist.iloc[-1]
        prev = hist.iloc[-2]

        oT = self._to_float(today.get("open"))
        hT = self._to_float(today.get("high"))
        lT = self._to_float(today.get("low"))
        cT = self._to_float(today.get("close"))
        vT = self._to_float(today.get("volume"))

        cP = self._to_float(prev.get("close"))

        # 基础合法性检查
        if not (np.isfinite(oT) and np.isfinite(hT) and np.isfinite(lT) and np.isfinite(cT) and np.isfinite(vT) and np.isfinite(cP)):
            return False
        if cP <= 0 or cT <= 0:
            return False
        if hT < max(oT, cT) or lT > min(oT, cT):
            # K线数据异常
            return False

        # (可选) 要求当日收阳
        if self.require_bullish_close and not (cT >= oT):
            return False

        # 1) 长阳：涨幅 > 阈值
        pct_chg = cT / cP - 1.0
        if pct_chg <= self.up_pct_threshold:
            return False

        # 2) 上影线百分比 < 阈值
        wick_pct = self._upper_wick_pct(oT, hT, cT)
        if not np.isfinite(wick_pct):
            return False
        if wick_pct >= self.upper_wick_pct_max:
            return False

        # 3) 放量：当日成交量 > 前 n 日均量 * 倍数
        vol_hist = hist["volume"].iloc[-(self.vol_lookback_n + 1):-1].astype(float)  # T-n ... T-1
        if self.ignore_zero_volume:
            vol_hist = vol_hist.replace(0, np.nan).dropna()

        if len(vol_hist) < max(3, int(self.vol_lookback_n * 0.6)):
            # 有效样本过少就不做判断
            return False

        avg_vol = float(vol_hist.mean())
        if not (np.isfinite(avg_vol) and avg_vol > 0):
            return False

        if vT < self.vol_multiple * avg_vol:
            return False

        # 4) 偏离短线小于阈值
        # 注意：BigBullishVolumeSelector使用短窗口(~22行)，
        # 必须在短窗口上重新计算ZXDQ以保持一致性
        try:
            zxdq, _ = compute_zx_lines(hist)
            zxdq_T = float(zxdq.iloc[-1])
        except Exception:
            zxdq_T = float("nan")

        if not np.isfinite(zxdq_T):
            return False
        else:
            if not (cT < zxdq_T * self.close_lt_zxdq_mult):
                return False

        return True

    def select(self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> List[str]:
        picks: List[str] = []
        need_len = max(self.min_history, self.vol_lookback_n + 2)

        for code, df in data.items():
            if df is None or df.empty:
                continue
            hist = df[df["date"] <= date].tail(need_len)
            if len(hist) < need_len:
                continue
            if self._passes_filters(hist):
                picks.append(code)

        return picks


# ─────────── 批量指标预计算 ─────────── #

def precompute_indicators(data: Dict[str, pd.DataFrame], target_date: pd.Timestamp) -> None:
    """
    一次性为所有股票预计算常用技术指标，避免 8 个策略重复计算。

    在每只股票的 DataFrame 上就地添加以下列（如果尚不存在）：
      BBI, K, D, J, DIF, ZXDQ, ZXDKX, MA60

    调用此函数后，各 Selector._passes_filters() 中的 guard 语句会
    检测到列已存在而跳过重复计算。

    Args:
        data: {code: DataFrame} 字典，每个 DF 含 date/open/high/low/close/volume
        target_date: 目标日期（用于截取有效数据范围）
    """
    for code, df in data.items():
        hist = df[df["date"] <= target_date]
        if len(hist) < 20:
            continue

        # BBI
        if "BBI" not in hist.columns:
            df.loc[hist.index, "BBI"] = compute_bbi(hist).values

        # KDJ
        if "K" not in hist.columns or "D" not in hist.columns or "J" not in hist.columns:
            kdj = compute_kdj(hist)
            df.loc[hist.index, "K"] = kdj["K"].values
            df.loc[hist.index, "D"] = kdj["D"].values
            df.loc[hist.index, "J"] = kdj["J"].values

        # DIF (MACD)
        if "DIF" not in hist.columns:
            df.loc[hist.index, "DIF"] = compute_dif(hist).values

        # 知行线
        if "ZXDQ" not in hist.columns or "ZXDKX" not in hist.columns:
            zxdq, zxdkx = compute_zx_lines(hist)
            df.loc[hist.index, "ZXDQ"] = zxdq.values
            df.loc[hist.index, "ZXDKX"] = zxdkx.values

        # MA60
        if "MA60" not in hist.columns:
            df.loc[hist.index, "MA60"] = hist["close"].rolling(window=60, min_periods=60).mean().values
