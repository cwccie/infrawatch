"""Unified metric format for InfraWatch.

All collectors normalize their output into Metric objects, providing a consistent
interface for the preprocessing and detection pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True, slots=True)
class Metric:
    """A single timestamped metric observation.

    Attributes:
        name: Metric name (e.g., 'cpu_usage_percent', 'if_octets_in').
        value: Numeric observation value.
        timestamp: Unix epoch seconds. Defaults to current time.
        labels: Key-value metadata (host, interface, region, etc.).
        source: Collector origin identifier.
        unit: Optional unit string (percent, bytes, ms, etc.).
    """

    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)
    source: str = "unknown"
    unit: str = ""

    def with_labels(self, **extra_labels: str) -> Metric:
        """Return a new Metric with additional labels merged in."""
        merged = {**self.labels, **extra_labels}
        return Metric(
            name=self.name,
            value=self.value,
            timestamp=self.timestamp,
            labels=merged,
            source=self.source,
            unit=self.unit,
        )

    def label_fingerprint(self) -> str:
        """Deterministic string key from sorted labels for grouping."""
        parts = [f"{k}={v}" for k, v in sorted(self.labels.items())]
        return f"{self.name}{{{','.join(parts)}}}"

    def to_dict(self) -> dict:
        """Serialize to plain dict."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp,
            "labels": dict(self.labels),
            "source": self.source,
            "unit": self.unit,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Metric:
        """Deserialize from plain dict."""
        return cls(
            name=d["name"],
            value=d["value"],
            timestamp=d.get("timestamp", time.time()),
            labels=d.get("labels", {}),
            source=d.get("source", "unknown"),
            unit=d.get("unit", ""),
        )


@dataclass
class MetricBatch:
    """A batch of metrics collected in one scrape/poll cycle.

    Attributes:
        metrics: List of Metric objects.
        collected_at: Timestamp when the batch was assembled.
        collector_id: Identifier for the collector instance.
    """

    metrics: list[Metric] = field(default_factory=list)
    collected_at: float = field(default_factory=time.time)
    collector_id: str = ""

    def __len__(self) -> int:
        return len(self.metrics)

    def __iter__(self):
        return iter(self.metrics)

    def add(self, metric: Metric) -> None:
        """Append a metric to the batch."""
        self.metrics.append(metric)

    def filter_by_name(self, name: str) -> list[Metric]:
        """Return all metrics matching the given name."""
        return [m for m in self.metrics if m.name == name]

    def filter_by_label(self, key: str, value: str) -> list[Metric]:
        """Return metrics where labels[key] == value."""
        return [m for m in self.metrics if m.labels.get(key) == value]

    def group_by_name(self) -> dict[str, list[Metric]]:
        """Group metrics by name."""
        groups: dict[str, list[Metric]] = {}
        for m in self.metrics:
            groups.setdefault(m.name, []).append(m)
        return groups

    def unique_names(self) -> set[str]:
        """Return the set of distinct metric names."""
        return {m.name for m in self.metrics}
