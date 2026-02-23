"""Counter unwrapping for SNMP and Prometheus counter metrics.

Infrastructure counters (e.g., ifInOctets, network_receive_bytes_total)
are monotonically increasing values that wrap at 2^32 or 2^64. This module
detects and corrects wrap-arounds to produce accurate delta values.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

COUNTER32_MAX = 2**32
COUNTER64_MAX = 2**64


def unwrap_counters(
    values: NDArray[np.float64],
    bits: int = 64,
    max_rate: float | None = None,
) -> NDArray[np.float64]:
    """Convert monotonic counter values to per-interval deltas with wrap handling.

    Args:
        values: Array of raw counter readings (monotonically increasing with wraps).
        bits: Counter bit width (32 or 64). Determines the wrap-around threshold.
        max_rate: Maximum expected per-interval delta. If a delta exceeds this
                  after unwrapping, it's treated as a counter reset (set to 0).

    Returns:
        Array of per-interval deltas (length = len(values) - 1).
        First element is always 0 (no previous value to compute delta from),
        so the returned array has the same length as input.
    """
    if len(values) < 2:
        return np.zeros_like(values)

    max_val = COUNTER64_MAX if bits == 64 else COUNTER32_MAX
    result = np.zeros(len(values), dtype=np.float64)

    for i in range(1, len(values)):
        curr = values[i]
        prev = values[i - 1]

        if curr >= prev:
            delta = curr - prev
        else:
            # Counter wrapped or reset
            delta = (max_val - prev) + curr

        # Check for counter reset (unreasonably large delta)
        if max_rate is not None and delta > max_rate:
            delta = 0.0

        result[i] = delta

    return result


def detect_counter_wraps(
    values: NDArray[np.float64],
    bits: int = 64,
) -> NDArray[np.bool_]:
    """Identify indices where counter wraps occurred.

    Args:
        values: Array of raw counter readings.
        bits: Counter bit width.

    Returns:
        Boolean array where True indicates a wrap at that index.
    """
    if len(values) < 2:
        return np.zeros(len(values), dtype=bool)

    diffs = np.diff(values)
    wraps = np.zeros(len(values), dtype=bool)
    wraps[1:] = diffs < 0
    return wraps


def is_counter_metric(name: str) -> bool:
    """Heuristic to determine if a metric name represents a counter.

    Prometheus convention: counters end in _total or _count.
    SNMP convention: names containing 'Octets', 'Packets', 'Errors'.

    Args:
        name: Metric name string.

    Returns:
        True if the metric is likely a counter.
    """
    counter_suffixes = ("_total", "_count", "_sum")
    counter_keywords = ("octets", "packets", "errors", "bytes", "frames", "drops")

    lower = name.lower()
    return (
        any(lower.endswith(s) for s in counter_suffixes)
        or any(k in lower for k in counter_keywords)
    )
