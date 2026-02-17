"""
Stock similarity analyzer using pre-allocated NumPy arrays for efficient
sliding-window batch computation.

Key optimization: batch_similarity_computation() pre-allocates a 2D windows
array once before the loop, then fills rows with views/slices, eliminating
~800K+ temporary Python list allocations across 4285 stocks × ~200 windows.
"""
from typing import Optional, Tuple

import numpy as np


class VectorizedSimilarityComputer:
    """
    Computes sliding-window similarity between a target price-return window
    and a candidate return series using vectorized Pearson correlation.

    Parameters
    ----------
    window_length : int
        Number of bars in each comparison window.
    step_size : int, optional
        Stride between successive windows (default 15).
    similarity_threshold : float, optional
        Correlation threshold used for early-exit hint (default 0.15).
        Windows whose normalized correlation exceeds this value are
        considered a meaningful match.
    """

    def __init__(
        self,
        window_length: int,
        step_size: int = 15,
        similarity_threshold: float = 0.15,
    ) -> None:
        if window_length < 1:
            raise ValueError("window_length must be >= 1")
        self.window_length = window_length
        self.step_size = step_size
        self.similarity_threshold = similarity_threshold

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_returns(arr: np.ndarray) -> np.ndarray:
        """
        Z-score normalize a 1-D return series.

        Returns a zero vector when the standard deviation is negligible
        (flat / constant series) to avoid division-by-zero.
        """
        std = arr.std()
        if std < 1e-9:
            return np.zeros_like(arr, dtype=np.float64)
        return (arr - arr.mean()) / std

    @staticmethod
    def vectorized_correlation(
        windows: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Pearson correlation between each row of *windows* and *target*.

        Parameters
        ----------
        windows : np.ndarray, shape (n_windows, window_length)
            Pre-allocated matrix of sliding windows.
        target : np.ndarray, shape (window_length,)
            The target window to compare against.

        Returns
        -------
        np.ndarray, shape (n_windows,)
            Correlation coefficient for each window (clipped to [-1, 1]).
        """
        n_windows = len(windows)
        if n_windows == 0:
            return np.empty(0, dtype=np.float64)

        # Normalize target once
        target_std = target.std()
        if target_std < 1e-9:
            return np.zeros(n_windows, dtype=np.float64)
        target_norm = (target - target.mean()) / target_std

        # Normalize every window row in a single vectorized pass
        windows_mean = windows.mean(axis=1, keepdims=True)
        windows_std = windows.std(axis=1, keepdims=True)

        # Prevent division by zero for flat windows
        safe_std = np.where(windows_std < 1e-9, 1.0, windows_std)
        windows_norm = (windows - windows_mean) / safe_std

        # Pearson correlation = mean of element-wise product of z-scores
        correlations = (windows_norm * target_norm).mean(axis=1)

        # Zero out correlations from flat windows (their std was replaced by 1.0)
        flat_mask = (windows_std.squeeze(axis=1) < 1e-9)
        correlations[flat_mask] = 0.0

        return correlations

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def batch_similarity_computation(
        self,
        target_window: np.ndarray,
        candidate_returns: np.ndarray,
        step_size: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> Tuple[float, Optional[int], int]:
        """
        Find the sliding window in *candidate_returns* most similar to
        *target_window* using pre-allocated numpy arrays (no list append).

        Parameters
        ----------
        target_window : np.ndarray, shape (window_length,)
            Reference return series to search for.
        candidate_returns : np.ndarray, shape (n,)
            The longer series to slide over.
        step_size : int, optional
            Override the instance default step_size.
        similarity_threshold : float, optional
            Override the instance default similarity_threshold.
            When the best correlation found exceeds this value the result
            is returned immediately (early-exit optimisation).

        Returns
        -------
        (best_score, best_position, best_index) : Tuple[float, Optional[int], int]
            best_score    – Pearson correlation of the best-matching window.
            best_position – Starting bar index in candidate_returns of the
                            best window, or None if no valid window exists.
            best_index    – 0-based index of the best window among all
                            generated windows (0 when no valid window exists).

        Notes
        -----
        *candidate_returns* shorter than *window_length* returns (0, None, 0)
        immediately.
        """
        _step = step_size if step_size is not None else self.step_size
        _threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.similarity_threshold
        )

        wl = self.window_length
        n = len(candidate_returns)

        # Edge case: candidate too short for even one window
        if n < wl:
            return 0.0, None, 0

        # Number of windows that fit with the given step_size
        n_windows = (n - wl) // _step + 1

        # ---------------------------------------------------------------
        # Pre-allocate the 2D windows matrix ONCE – no list appends.
        # Each row is filled via a direct slice assignment, which copies
        # the data in one contiguous C-level operation (no Python objects).
        # ---------------------------------------------------------------
        windows = np.empty((n_windows, wl), dtype=candidate_returns.dtype)
        for k in range(n_windows):
            i = k * _step
            windows[k] = candidate_returns[i : i + wl]

        # Batch correlations (fully vectorized, single numpy call)
        correlations = self.vectorized_correlation(windows, target_window)

        # Early-exit: return as soon as a window exceeds the threshold
        above = np.where(correlations >= _threshold)[0]
        if above.size > 0:
            best_idx = int(above[np.argmax(correlations[above])])
        else:
            best_idx = int(np.argmax(correlations))

        best_score = float(correlations[best_idx])
        best_position = best_idx * _step

        return best_score, best_position, best_idx
