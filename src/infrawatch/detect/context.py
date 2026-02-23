"""Context-aware anomaly analysis.

Adjusts anomaly thresholds and severity based on temporal context
(time of day, day of week, holidays) and historical patterns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class TemporalContext:
    """Temporal context for a detection event.

    Attributes:
        hour: Hour of day (0-23).
        day_of_week: Day of week (0=Monday, 6=Sunday).
        is_business_hours: Whether within business hours.
        is_weekend: Whether it's a weekend.
    """

    hour: int = 0
    day_of_week: int = 0
    is_business_hours: bool = False
    is_weekend: bool = False

    @classmethod
    def from_timestamp(
        cls,
        timestamp: float,
        business_start: int = 8,
        business_end: int = 18,
    ) -> TemporalContext:
        """Create context from a Unix timestamp."""
        dt = datetime.fromtimestamp(timestamp)
        is_weekend = dt.weekday() >= 5
        is_bh = business_start <= dt.hour < business_end and not is_weekend

        return cls(
            hour=dt.hour,
            day_of_week=dt.weekday(),
            is_business_hours=is_bh,
            is_weekend=is_weekend,
        )


class ContextAnalyzer:
    """Analyzes temporal and correlational context for anomalies.

    Adjusts detection sensitivity based on time-of-day patterns and
    cross-metric correlation.

    Args:
        business_hours: Tuple of (start_hour, end_hour).
        sensitivity_schedule: Mapping of context to threshold multiplier.
            Higher multiplier = less sensitive (fewer alerts).
    """

    def __init__(
        self,
        business_hours: tuple[int, int] = (8, 18),
        sensitivity_schedule: Optional[dict[str, float]] = None,
    ):
        self.business_hours = business_hours
        self.sensitivity_schedule = sensitivity_schedule or {
            "business_hours": 1.0,
            "after_hours": 1.5,
            "weekend": 2.0,
            "night": 2.0,
        }

    def get_threshold_multiplier(self, timestamp: float) -> float:
        """Get the threshold multiplier for the given timestamp.

        During low-traffic periods, increase thresholds to reduce noise.

        Args:
            timestamp: Unix epoch timestamp.

        Returns:
            Multiplier to apply to detection thresholds.
        """
        ctx = TemporalContext.from_timestamp(
            timestamp,
            business_start=self.business_hours[0],
            business_end=self.business_hours[1],
        )

        if ctx.is_weekend:
            return self.sensitivity_schedule.get("weekend", 2.0)
        if ctx.is_business_hours:
            return self.sensitivity_schedule.get("business_hours", 1.0)
        if 0 <= ctx.hour < 6:
            return self.sensitivity_schedule.get("night", 2.0)
        return self.sensitivity_schedule.get("after_hours", 1.5)

    def correlate_metrics(
        self,
        metric_anomalies: dict[str, list[float]],
        time_window: float = 300.0,
    ) -> list[set[str]]:
        """Find groups of metrics with correlated anomalies.

        Two metrics are correlated if they have anomalies within
        `time_window` seconds of each other.

        Args:
            metric_anomalies: Mapping of metric name to list of anomaly timestamps.
            time_window: Maximum time difference to consider correlated.

        Returns:
            List of sets of correlated metric names.
        """
        metrics = list(metric_anomalies.keys())
        n = len(metrics)
        if n < 2:
            return [set(metrics)] if metrics else []

        # Build adjacency
        adj: dict[str, set[str]] = {m: set() for m in metrics}

        for i in range(n):
            for j in range(i + 1, n):
                m1, m2 = metrics[i], metrics[j]
                ts1 = metric_anomalies[m1]
                ts2 = metric_anomalies[m2]

                # Check if any anomaly timestamps are within the window
                for t1 in ts1:
                    for t2 in ts2:
                        if abs(t1 - t2) <= time_window:
                            adj[m1].add(m2)
                            adj[m2].add(m1)
                            break
                    else:
                        continue
                    break

        # Find connected components
        visited: set[str] = set()
        groups: list[set[str]] = []

        for m in metrics:
            if m in visited:
                continue
            group: set[str] = set()
            stack = [m]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                group.add(node)
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        stack.append(neighbor)
            if len(group) > 1:
                groups.append(group)

        return groups
