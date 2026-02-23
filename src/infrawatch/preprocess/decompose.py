"""Seasonal decomposition for infrastructure metrics.

Decomposes time series into trend, seasonal, and residual components.
Uses a moving-average-based STL-like approach implemented in pure NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class DecompositionResult:
    """Result of seasonal decomposition.

    Attributes:
        observed: Original time series.
        trend: Smoothed trend component.
        seasonal: Repeating seasonal component.
        residual: Remainder after removing trend and seasonal.
        period: Detected or specified seasonal period.
    """

    observed: NDArray[np.float64]
    trend: NDArray[np.float64]
    seasonal: NDArray[np.float64]
    residual: NDArray[np.float64]
    period: int


def _moving_average(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Centered moving average with NaN handling."""
    if window <= 1:
        return values.copy()

    result = np.full_like(values, np.nan)
    half = window // 2

    cumsum = np.nancumsum(values)
    cumsum = np.insert(cumsum, 0, 0)

    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        count = hi - lo
        window_sum = cumsum[hi] - cumsum[lo]
        # Count non-NaN values
        n_valid = count - np.sum(np.isnan(values[lo:hi]))
        if n_valid > 0:
            result[i] = window_sum / n_valid

    return result


def estimate_period(
    values: NDArray[np.float64],
    min_period: int = 2,
    max_period: int | None = None,
) -> int:
    """Estimate the dominant seasonal period using autocorrelation.

    Args:
        values: Time series values.
        min_period: Minimum period to consider.
        max_period: Maximum period to consider. Defaults to len/3.

    Returns:
        Estimated period in samples.
    """
    n = len(values)
    if max_period is None:
        max_period = max(min_period + 1, n // 3)
    max_period = min(max_period, n // 2)

    if n < 2 * min_period:
        return min_period

    # Remove mean for autocorrelation
    clean = values.copy()
    clean[np.isnan(clean)] = np.nanmean(clean)
    centered = clean - np.mean(clean)

    var = np.sum(centered**2)
    if var == 0:
        return min_period

    # Compute autocorrelation for candidate lags
    best_lag = min_period
    best_acf = -1.0

    for lag in range(min_period, max_period + 1):
        acf = np.sum(centered[:n - lag] * centered[lag:]) / var
        if acf > best_acf:
            best_acf = acf
            best_lag = lag

    return best_lag


def seasonal_decompose(
    values: NDArray[np.float64],
    period: int | None = None,
    model: str = "additive",
) -> DecompositionResult:
    """Decompose a time series into trend, seasonal, and residual components.

    Uses a classical decomposition approach:
    1. Estimate trend via centered moving average of length `period`.
    2. Detrend the series.
    3. Average the detrended values at each seasonal position to get the
       seasonal component.
    4. Residual = observed - trend - seasonal (additive) or
       observed / (trend * seasonal) (multiplicative).

    Args:
        values: Time series values.
        period: Seasonal period in samples. If None, estimated automatically.
        model: 'additive' or 'multiplicative'.

    Returns:
        DecompositionResult with all components.
    """
    n = len(values)

    if period is None:
        period = estimate_period(values)
    period = max(2, period)

    # Step 1: Trend via moving average
    trend = _moving_average(values, period)

    # Step 2: Detrend
    if model == "multiplicative":
        # Avoid division by zero
        safe_trend = np.where(trend == 0, np.nan, trend)
        detrended = values / safe_trend
    else:
        detrended = values - trend

    # Step 3: Seasonal component — average detrended at each period position
    seasonal = np.zeros(n)
    for pos in range(period):
        indices = np.arange(pos, n, period)
        vals = detrended[indices]
        valid = vals[~np.isnan(vals)]
        if len(valid) > 0:
            seasonal[indices] = np.mean(valid)

    # Normalize seasonal component to sum to zero (additive) or mean to 1 (mult)
    for pos in range(period):
        indices = np.arange(pos, n, period)
        # Already averaged, just ensure consistency

    if model == "additive":
        period_means = [
            np.nanmean(seasonal[np.arange(p, n, period)])
            for p in range(period)
        ]
        overall_mean = np.nanmean(period_means)
        seasonal -= overall_mean
    else:
        period_means = [
            np.nanmean(seasonal[np.arange(p, n, period)])
            for p in range(period)
        ]
        overall_mean = np.nanmean(period_means)
        if overall_mean != 0:
            seasonal /= overall_mean

    # Step 4: Residual
    if model == "multiplicative":
        safe_product = trend * seasonal
        safe_product = np.where(safe_product == 0, np.nan, safe_product)
        residual = values / safe_product
    else:
        residual = values - trend - seasonal

    return DecompositionResult(
        observed=values,
        trend=trend,
        seasonal=seasonal,
        residual=residual,
        period=period,
    )
