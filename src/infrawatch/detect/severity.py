"""Severity classification for detected anomalies.

Maps anomaly scores and context into actionable severity levels.
"""

from __future__ import annotations

from enum import IntEnum


class Severity(IntEnum):
    """Anomaly severity levels."""

    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    @property
    def label(self) -> str:
        return self.name.lower()

    @property
    def emoji(self) -> str:
        return {0: "ℹ️", 1: "🟡", 2: "🟠", 3: "🔴", 4: "🚨"}.get(self.value, "❓")


def classify_severity(
    anomaly_score: float,
    duration_minutes: float = 0.0,
    metric_type: str = "",
    is_correlated: bool = False,
) -> Severity:
    """Classify anomaly severity based on score, duration, and context.

    Args:
        anomaly_score: Normalized anomaly score [0, 1].
        duration_minutes: How long the anomaly has persisted.
        metric_type: Type of metric (e.g., 'cpu', 'memory', 'error_rate').
        is_correlated: Whether the anomaly correlates with other metrics.

    Returns:
        Severity classification.
    """
    # Base severity from score
    if anomaly_score >= 0.9:
        base = Severity.CRITICAL
    elif anomaly_score >= 0.7:
        base = Severity.HIGH
    elif anomaly_score >= 0.5:
        base = Severity.MEDIUM
    elif anomaly_score >= 0.3:
        base = Severity.LOW
    else:
        base = Severity.INFO

    # Escalate for sustained anomalies
    if duration_minutes > 30:
        base = min(Severity.CRITICAL, Severity(base.value + 1))
    elif duration_minutes > 10:
        if base < Severity.HIGH:
            base = Severity(base.value + 1)

    # Escalate for critical metric types
    critical_types = {"error_rate", "packet_loss", "disk_full", "oom"}
    if metric_type.lower() in critical_types and base < Severity.HIGH:
        base = Severity(base.value + 1)

    # Escalate for correlated anomalies (multi-metric failure)
    if is_correlated and base < Severity.CRITICAL:
        base = Severity(base.value + 1)

    return base
