"""Normalization functions for time series data.

Multiple normalization strategies for different model requirements.
All functions handle NaN values gracefully.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def z_normalize(
    values: NDArray[np.float64],
    mean: float | None = None,
    std: float | None = None,
) -> tuple[NDArray[np.float64], float, float]:
    """Z-score normalization (zero mean, unit variance).

    Args:
        values: Input array.
        mean: Pre-computed mean (for applying saved parameters).
        std: Pre-computed standard deviation.

    Returns:
        Tuple of (normalized_values, mean, std).
    """
    if mean is None:
        mean = float(np.nanmean(values))
    if std is None:
        std = float(np.nanstd(values))

    if std == 0:
        return np.zeros_like(values), mean, std

    normalized = (values - mean) / std
    return normalized, mean, std


def minmax_normalize(
    values: NDArray[np.float64],
    vmin: float | None = None,
    vmax: float | None = None,
    feature_range: tuple[float, float] = (0.0, 1.0),
) -> tuple[NDArray[np.float64], float, float]:
    """Min-max normalization to a target range.

    Args:
        values: Input array.
        vmin: Pre-computed minimum.
        vmax: Pre-computed maximum.
        feature_range: Target (min, max) range.

    Returns:
        Tuple of (normalized_values, data_min, data_max).
    """
    if vmin is None:
        vmin = float(np.nanmin(values))
    if vmax is None:
        vmax = float(np.nanmax(values))

    data_range = vmax - vmin
    if data_range == 0:
        return np.full_like(values, feature_range[0]), vmin, vmax

    scale = feature_range[1] - feature_range[0]
    normalized = (values - vmin) / data_range * scale + feature_range[0]
    return normalized, vmin, vmax


def robust_normalize(
    values: NDArray[np.float64],
    median: float | None = None,
    iqr: float | None = None,
) -> tuple[NDArray[np.float64], float, float]:
    """Robust normalization using median and IQR.

    Less sensitive to outliers than Z-score normalization.

    Args:
        values: Input array.
        median: Pre-computed median.
        iqr: Pre-computed interquartile range.

    Returns:
        Tuple of (normalized_values, median, iqr).
    """
    clean = values[~np.isnan(values)]

    if median is None:
        median = float(np.median(clean)) if len(clean) > 0 else 0.0
    if iqr is None:
        if len(clean) >= 4:
            q75, q25 = np.percentile(clean, [75, 25])
            iqr = float(q75 - q25)
        else:
            iqr = 1.0

    if iqr == 0:
        iqr = 1.0

    normalized = (values - median) / iqr
    return normalized, median, iqr


def log_normalize(
    values: NDArray[np.float64],
    offset: float = 1.0,
) -> NDArray[np.float64]:
    """Log normalization with offset for zero/negative values.

    Useful for highly skewed metrics like request latency.

    Args:
        values: Input array.
        offset: Added before log to handle zeros.

    Returns:
        Log-normalized array.
    """
    return np.log1p(np.maximum(values, 0) + offset - 1)
