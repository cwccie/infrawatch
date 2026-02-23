"""Outlier handling for time series data.

Provides methods to detect, clip, and replace outliers in infrastructure
metrics. Outlier handling happens before anomaly detection to prevent
measurement artifacts from triggering false alarms.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def detect_outliers_iqr(
    values: NDArray[np.float64],
    factor: float = 1.5,
) -> NDArray[np.bool_]:
    """Detect outliers using the Interquartile Range method.

    Args:
        values: Input array.
        factor: IQR multiplier for fence calculation.

    Returns:
        Boolean mask where True indicates an outlier.
    """
    clean = values[~np.isnan(values)]
    if len(clean) < 4:
        return np.zeros(len(values), dtype=bool)

    q1 = np.percentile(clean, 25)
    q3 = np.percentile(clean, 75)
    iqr = q3 - q1

    lower = q1 - factor * iqr
    upper = q3 + factor * iqr

    return (values < lower) | (values > upper)


def detect_outliers_zscore(
    values: NDArray[np.float64],
    threshold: float = 3.0,
) -> NDArray[np.bool_]:
    """Detect outliers using Z-score method.

    Args:
        values: Input array.
        threshold: Z-score threshold for outlier classification.

    Returns:
        Boolean mask where True indicates an outlier.
    """
    clean = values[~np.isnan(values)]
    if len(clean) < 2:
        return np.zeros(len(values), dtype=bool)

    mean = np.nanmean(values)
    std = np.nanstd(values)

    if std == 0:
        return np.zeros(len(values), dtype=bool)

    z_scores = np.abs((values - mean) / std)
    return z_scores > threshold


def detect_outliers_modified_zscore(
    values: NDArray[np.float64],
    threshold: float = 3.5,
) -> NDArray[np.bool_]:
    """Detect outliers using Modified Z-score (MAD-based).

    More robust than standard Z-score for non-Gaussian distributions,
    which is common in infrastructure metrics.

    Args:
        values: Input array.
        threshold: Modified Z-score threshold.

    Returns:
        Boolean mask where True indicates an outlier.
    """
    clean = values[~np.isnan(values)]
    if len(clean) < 2:
        return np.zeros(len(values), dtype=bool)

    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))

    if mad == 0:
        return np.zeros(len(values), dtype=bool)

    modified_z = 0.6745 * (values - median) / mad
    return np.abs(modified_z) > threshold


def clip_outliers(
    values: NDArray[np.float64],
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> NDArray[np.float64]:
    """Clip values to the specified percentile range.

    Args:
        values: Input array.
        lower_percentile: Lower bound percentile.
        upper_percentile: Upper bound percentile.

    Returns:
        Array with outliers clipped to the percentile bounds.
    """
    clean = values[~np.isnan(values)]
    if len(clean) < 2:
        return values.copy()

    lower = np.percentile(clean, lower_percentile)
    upper = np.percentile(clean, upper_percentile)

    result = values.copy()
    result = np.clip(result, lower, upper)
    return result


def replace_outliers(
    values: NDArray[np.float64],
    method: str = "iqr",
    replacement: str = "nan",
    **kwargs,
) -> NDArray[np.float64]:
    """Detect and replace outliers.

    Args:
        values: Input array.
        method: Detection method — 'iqr', 'zscore', or 'modified_zscore'.
        replacement: Replacement strategy — 'nan', 'median', 'mean', or 'clip'.
        **kwargs: Additional arguments passed to the detection method.

    Returns:
        Array with outliers replaced.
    """
    detectors = {
        "iqr": detect_outliers_iqr,
        "zscore": detect_outliers_zscore,
        "modified_zscore": detect_outliers_modified_zscore,
    }

    if method not in detectors:
        raise ValueError(f"Unknown outlier method: {method}. Use: {list(detectors)}")

    outlier_mask = detectors[method](values, **kwargs)
    result = values.copy()

    if not np.any(outlier_mask):
        return result

    if replacement == "nan":
        result[outlier_mask] = np.nan
    elif replacement == "median":
        result[outlier_mask] = np.nanmedian(values[~outlier_mask])
    elif replacement == "mean":
        result[outlier_mask] = np.nanmean(values[~outlier_mask])
    elif replacement == "clip":
        clean = values[~outlier_mask & ~np.isnan(values)]
        if len(clean) > 0:
            result[outlier_mask & (values > np.max(clean))] = np.max(clean)
            result[outlier_mask & (values < np.min(clean))] = np.min(clean)
    else:
        raise ValueError(f"Unknown replacement strategy: {replacement}")

    return result
