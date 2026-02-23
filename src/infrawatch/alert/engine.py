"""Alert engine with deduplication, grouping, and escalation.

Manages the lifecycle of alerts from detection through notification
and resolution. Prevents alert storms via deduplication and grouping.
"""

from __future__ import annotations

import hashlib
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol

from infrawatch.detect.severity import Severity

logger = logging.getLogger(__name__)


class AlertState(Enum):
    """Alert lifecycle state."""
    FIRING = "firing"
    ACKNOWLEDGED = "acknowledged"
    SILENCED = "silenced"
    RESOLVED = "resolved"


class Notifier(Protocol):
    """Protocol for alert notification backends."""

    def send(self, alert: Alert) -> bool:
        """Send a notification for the given alert. Returns True on success."""
        ...


@dataclass
class Alert:
    """A deduplicated, grouped alert event.

    Attributes:
        id: Unique alert ID (fingerprint-based).
        fingerprint: Deduplication key.
        metric_name: Source metric.
        severity: Alert severity.
        message: Human-readable alert message.
        state: Current lifecycle state.
        first_seen: When the alert first fired.
        last_seen: When the alert was last updated.
        count: Number of times this alert has been deduplicated.
        labels: Metric labels.
        annotations: Extra information (runbook URL, description, etc.).
        value: Current metric value.
        score: Anomaly score.
    """

    id: str = ""
    fingerprint: str = ""
    metric_name: str = ""
    severity: Severity = Severity.INFO
    message: str = ""
    state: AlertState = AlertState.FIRING
    first_seen: float = 0.0
    last_seen: float = 0.0
    count: int = 1
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)
    value: float = 0.0
    score: float = 0.0

    @property
    def duration_minutes(self) -> float:
        return (self.last_seen - self.first_seen) / 60.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "fingerprint": self.fingerprint,
            "metric_name": self.metric_name,
            "severity": self.severity.label,
            "message": self.message,
            "state": self.state.value,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "count": self.count,
            "duration_minutes": self.duration_minutes,
            "labels": self.labels,
            "value": self.value,
            "score": self.score,
        }


class AlertEngine:
    """Manages alert lifecycle, deduplication, and notification routing.

    Args:
        dedup_window: Seconds within which identical alerts are deduplicated.
        group_wait: Seconds to wait before sending a group of new alerts.
        escalation_minutes: Minutes before an unacked alert escalates.
        notifiers: List of notification backends.
    """

    def __init__(
        self,
        dedup_window: float = 300.0,
        group_wait: float = 30.0,
        escalation_minutes: float = 30.0,
        notifiers: list[Notifier] | None = None,
    ):
        self.dedup_window = dedup_window
        self.group_wait = group_wait
        self.escalation_minutes = escalation_minutes
        self.notifiers = notifiers or []
        self._active_alerts: dict[str, Alert] = {}
        self._silences: dict[str, float] = {}  # fingerprint -> silence_until
        self._history: list[Alert] = []

    def _compute_fingerprint(
        self,
        metric_name: str,
        labels: dict[str, str],
    ) -> str:
        """Compute deduplication fingerprint from metric identity."""
        parts = [metric_name] + [f"{k}={v}" for k, v in sorted(labels.items())]
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def fire(
        self,
        metric_name: str,
        severity: Severity,
        value: float = 0.0,
        score: float = 0.0,
        labels: dict[str, str] | None = None,
        message: str = "",
    ) -> Alert:
        """Fire or update an alert.

        If an alert with the same fingerprint already exists and is within
        the dedup window, increments the count instead of creating a new alert.

        Args:
            metric_name: Source metric name.
            severity: Alert severity.
            value: Current metric value.
            score: Anomaly score.
            labels: Metric labels.
            message: Alert message.

        Returns:
            The created or updated Alert.
        """
        labels = labels or {}
        fp = self._compute_fingerprint(metric_name, labels)
        now = time.time()

        # Check silence
        if fp in self._silences:
            if now < self._silences[fp]:
                logger.debug("Alert silenced: %s", fp)
                existing = self._active_alerts.get(fp)
                if existing:
                    existing.state = AlertState.SILENCED
                    return existing
                return Alert(fingerprint=fp, state=AlertState.SILENCED)
            else:
                del self._silences[fp]

        # Deduplication
        if fp in self._active_alerts:
            existing = self._active_alerts[fp]
            if (now - existing.last_seen) < self.dedup_window:
                existing.last_seen = now
                existing.count += 1
                existing.value = value
                existing.score = score
                if severity > existing.severity:
                    existing.severity = severity
                return existing

        if not message:
            message = f"Anomaly detected on {metric_name}: value={value:.2f}, score={score:.2f}"

        alert = Alert(
            id=f"alert-{fp}-{int(now)}",
            fingerprint=fp,
            metric_name=metric_name,
            severity=severity,
            message=message,
            state=AlertState.FIRING,
            first_seen=now,
            last_seen=now,
            labels=labels,
            value=value,
            score=score,
        )

        self._active_alerts[fp] = alert

        # Notify
        for notifier in self.notifiers:
            try:
                notifier.send(alert)
            except Exception as exc:
                logger.error("Notification failed: %s", exc)

        return alert

    def resolve(self, fingerprint: str) -> Optional[Alert]:
        """Resolve an active alert."""
        if fingerprint in self._active_alerts:
            alert = self._active_alerts[fingerprint]
            alert.state = AlertState.RESOLVED
            alert.last_seen = time.time()
            self._history.append(alert)
            del self._active_alerts[fingerprint]
            return alert
        return None

    def acknowledge(self, fingerprint: str) -> Optional[Alert]:
        """Acknowledge an active alert."""
        if fingerprint in self._active_alerts:
            alert = self._active_alerts[fingerprint]
            alert.state = AlertState.ACKNOWLEDGED
            return alert
        return None

    def silence(self, fingerprint: str, duration_minutes: float = 60.0) -> None:
        """Silence alerts matching a fingerprint.

        Args:
            fingerprint: Alert fingerprint to silence.
            duration_minutes: How long to silence (in minutes).
        """
        self._silences[fingerprint] = time.time() + (duration_minutes * 60)

    def silence_metric(
        self,
        metric_name: str,
        labels: dict[str, str] | None = None,
        duration_minutes: float = 60.0,
    ) -> str:
        """Silence alerts for a specific metric."""
        fp = self._compute_fingerprint(metric_name, labels or {})
        self.silence(fp, duration_minutes)
        return fp

    def active_alerts(self) -> list[Alert]:
        """Return all currently active (firing or acked) alerts."""
        return [
            a for a in self._active_alerts.values()
            if a.state in (AlertState.FIRING, AlertState.ACKNOWLEDGED)
        ]

    def check_escalations(self) -> list[Alert]:
        """Check for alerts that need escalation.

        Returns:
            List of alerts whose severity was escalated.
        """
        now = time.time()
        escalated = []

        for alert in self._active_alerts.values():
            if alert.state != AlertState.FIRING:
                continue
            duration_min = (now - alert.first_seen) / 60.0
            if duration_min > self.escalation_minutes:
                if alert.severity < Severity.CRITICAL:
                    alert.severity = Severity(alert.severity.value + 1)
                    escalated.append(alert)

        return escalated

    @property
    def alert_count(self) -> int:
        return len(self._active_alerts)
