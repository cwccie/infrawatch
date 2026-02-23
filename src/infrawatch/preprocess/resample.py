"""Time series resampling.

Infrastructure metrics arrive at irregular intervals. This module
resamples them onto a uniform time grid for consistent analysis.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def resample_uniform(
    timestamps: NDArray[np.float64],
    values: NDArray[np.float64],
    interval: float | None = None,
    method: str = "mean",
    start: float | None = None,
    end: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Resample irregular time series onto a uniform grid.

    Args:
        timestamps: Original timestamps (sorted ascending).
        values: Original values.
        interval: Target interval in same units as timestamps.
                  If None, uses the median observed interval.
        method: Aggregation method for bins — 'mean', 'max', 'min',
                'sum', 'last', or 'first'.
        start: Start time for the uniform grid. Defaults to first timestamp.
        end: End time for the uniform grid. Defaults to last timestamp.

    Returns:
        Tuple of (uniform_timestamps, resampled_values).
        Bins with no data are set to NaN.
    """
    if len(timestamps) == 0:
        return np.array([]), np.array([])

    if interval is None:
        if len(timestamps) < 2:
            interval = 1.0
        else:
            interval = float(np.median(np.diff(timestamps)))

    t_start = start if start is not None else timestamps[0]
    t_end = end if end is not None else timestamps[-1]

    n_bins = max(1, int(np.ceil((t_end - t_start) / interval)))
    uniform_ts = np.linspace(t_start, t_start + n_bins * interval, n_bins, endpoint=False)
    uniform_ts += interval / 2  # Bin centers

    result = np.full(n_bins, np.nan)

    # Assign each point to a bin
    bin_indices = np.floor((timestamps - t_start) / interval).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    agg_funcs = {
        "mean": np.nanmean,
        "max": np.nanmax,
        "min": np.nanmin,
        "sum": np.nansum,
        "last": lambda x: x[-1],
        "first": lambda x: x[0],
    }

    if method not in agg_funcs:
        raise ValueError(f"Unknown resampling method: {method}. Use: {list(agg_funcs)}")

    agg = agg_funcs[method]

    for b in range(n_bins):
        mask = bin_indices == b
        if np.any(mask):
            bin_vals = values[mask]
            valid = bin_vals[~np.isnan(bin_vals)]
            if len(valid) > 0:
                result[b] = agg(valid)

    return uniform_ts, result


def downsample(
    values: NDArray[np.float64],
    factor: int,
    method: str = "mean",
) -> NDArray[np.float64]:
    """Downsample by combining every `factor` consecutive points.

    Args:
        values: Input array.
        factor: Number of points to combine.
        method: Aggregation — 'mean', 'max', 'min'.

    Returns:
        Downsampled array of length ceil(len(values) / factor).
    """
    if factor <= 1:
        return values.copy()

    n = len(values)
    # Pad to exact multiple
    pad_len = (factor - n % factor) % factor
    padded = np.append(values, np.full(pad_len, np.nan))
    reshaped = padded.reshape(-1, factor)

    agg_map = {"mean": np.nanmean, "max": np.nanmax, "min": np.nanmin}
    agg = agg_map.get(method, np.nanmean)

    return np.array([agg(row) for row in reshaped])
