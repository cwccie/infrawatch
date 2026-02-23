"""Gap filling for time series data.

Infrastructure metrics often have gaps due to network outages, collector
restarts, or sampling jitter. This module provides several strategies
for filling gaps to produce continuous time series.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def fill_gaps(
    timestamps: NDArray[np.float64],
    values: NDArray[np.float64],
    method: str = "linear",
    max_gap: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fill NaN gaps in a time series using the specified method.

    Args:
        timestamps: Array of timestamps (must be sorted ascending).
        values: Array of values (NaN where data is missing).
        method: Interpolation method — 'linear', 'ffill' (forward fill),
                'bfill' (backward fill), or 'zero'.
        max_gap: Maximum gap duration (in timestamp units) to fill.
                 Gaps larger than this remain NaN.

    Returns:
        Tuple of (timestamps, filled_values). Timestamps are unchanged;
        values array has NaNs replaced where possible.
    """
    if len(values) == 0:
        return timestamps, values

    result = values.copy().astype(np.float64)
    nan_mask = np.isnan(result)

    if not np.any(nan_mask):
        return timestamps, result

    if method == "zero":
        result[nan_mask] = 0.0
        return timestamps, result

    if method == "ffill":
        for i in range(1, len(result)):
            if np.isnan(result[i]) and not np.isnan(result[i - 1]):
                if max_gap is None or (timestamps[i] - timestamps[i - 1]) <= max_gap:
                    result[i] = result[i - 1]
        return timestamps, result

    if method == "bfill":
        for i in range(len(result) - 2, -1, -1):
            if np.isnan(result[i]) and not np.isnan(result[i + 1]):
                if max_gap is None or (timestamps[i + 1] - timestamps[i]) <= max_gap:
                    result[i] = result[i + 1]
        return timestamps, result

    # Linear interpolation (default)
    valid = ~nan_mask
    if np.sum(valid) < 2:
        return timestamps, result

    valid_idx = np.where(valid)[0]
    nan_idx = np.where(nan_mask)[0]

    interpolated = np.interp(
        timestamps[nan_idx],
        timestamps[valid_idx],
        result[valid_idx],
    )

    if max_gap is not None:
        # Only fill gaps within the max_gap threshold
        for j, idx in enumerate(nan_idx):
            # Find nearest valid neighbors
            left = valid_idx[valid_idx < idx]
            right = valid_idx[valid_idx > idx]
            if len(left) > 0 and len(right) > 0:
                gap = timestamps[right[0]] - timestamps[left[-1]]
                if gap <= max_gap:
                    result[idx] = interpolated[j]
            elif len(left) > 0:
                gap = timestamps[idx] - timestamps[left[-1]]
                if gap <= max_gap:
                    result[idx] = interpolated[j]
    else:
        result[nan_idx] = interpolated

    return timestamps, result


def detect_gaps(
    timestamps: NDArray[np.float64],
    expected_interval: float | None = None,
    tolerance: float = 1.5,
) -> list[tuple[int, int, float]]:
    """Detect gaps in a timestamp array.

    Args:
        timestamps: Sorted array of timestamps.
        expected_interval: Expected sampling interval. If None, uses the
                          median of observed intervals.
        tolerance: Multiple of expected_interval above which a gap is detected.

    Returns:
        List of (start_index, end_index, gap_duration) tuples.
    """
    if len(timestamps) < 2:
        return []

    intervals = np.diff(timestamps)

    if expected_interval is None:
        expected_interval = float(np.median(intervals))

    threshold = expected_interval * tolerance
    gaps = []

    for i, interval in enumerate(intervals):
        if interval > threshold:
            gaps.append((i, i + 1, float(interval)))

    return gaps
