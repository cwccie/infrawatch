"""Maintenance window manager.

Tracks scheduled maintenance windows, suppresses alerts during windows,
and recalibrates baselines after maintenance completes.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MaintenanceWindow:
    """A scheduled maintenance window.

    Attributes:
        id: Unique identifier.
        name: Human-readable description.
        start_time: Window start (Unix epoch).
        end_time: Window end (Unix epoch).
        targets: List of metric name patterns or host labels affected.
        recalibrate: Whether to recalibrate baselines after the window.
        created_by: Who created the window.
    """

    id: str
    name: str
    start_time: float
    end_time: float
    targets: list[str] = field(default_factory=list)
    recalibrate: bool = True
    created_by: str = ""

    @property
    def duration_minutes(self) -> float:
        return (self.end_time - self.start_time) / 60.0

    @property
    def is_active(self) -> bool:
        now = time.time()
        return self.start_time <= now <= self.end_time

    @property
    def is_past(self) -> bool:
        return time.time() > self.end_time

    @property
    def is_future(self) -> bool:
        return time.time() < self.start_time

    def matches_metric(self, metric_name: str, labels: dict[str, str] | None = None) -> bool:
        """Check if this window applies to the given metric.

        Args:
            metric_name: Metric name to check.
            labels: Metric labels to check against targets.

        Returns:
            True if the metric falls within this maintenance window's scope.
        """
        if not self.targets:
            return True  # Global window — matches everything

        for target in self.targets:
            if target == "*":
                return True
            if target in metric_name:
                return True
            if labels:
                for v in labels.values():
                    if target in v:
                        return True
        return False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "targets": self.targets,
            "recalibrate": self.recalibrate,
            "is_active": self.is_active,
            "duration_minutes": self.duration_minutes,
        }


class MaintenanceManager:
    """Manages maintenance windows and alert suppression.

    Args:
        recalibration_buffer_minutes: Minutes after a window ends before
            recalibrating baselines (allows metrics to stabilize).
    """

    def __init__(self, recalibration_buffer_minutes: float = 15.0):
        self._windows: dict[str, MaintenanceWindow] = {}
        self.recalibration_buffer = recalibration_buffer_minutes * 60.0
        self._pending_recalibrations: dict[str, float] = {}

    def add_window(self, window: MaintenanceWindow) -> None:
        """Register a maintenance window."""
        self._windows[window.id] = window
        logger.info(
            "Maintenance window '%s' scheduled: %s (%.0f min)",
            window.name, window.id, window.duration_minutes,
        )

    def remove_window(self, window_id: str) -> bool:
        """Remove a maintenance window."""
        if window_id in self._windows:
            del self._windows[window_id]
            return True
        return False

    def get_window(self, window_id: str) -> Optional[MaintenanceWindow]:
        """Get a window by ID."""
        return self._windows.get(window_id)

    def list_windows(self, include_past: bool = False) -> list[MaintenanceWindow]:
        """List all maintenance windows.

        Args:
            include_past: Whether to include expired windows.

        Returns:
            List of maintenance windows, sorted by start time.
        """
        windows = list(self._windows.values())
        if not include_past:
            windows = [w for w in windows if not w.is_past]
        return sorted(windows, key=lambda w: w.start_time)

    def is_suppressed(
        self,
        metric_name: str,
        labels: dict[str, str] | None = None,
        timestamp: float | None = None,
    ) -> bool:
        """Check if alerts for a metric should be suppressed.

        Args:
            metric_name: Metric name.
            labels: Metric labels.
            timestamp: Time to check (defaults to now).

        Returns:
            True if the metric is currently under maintenance suppression.
        """
        ts = timestamp or time.time()
        for window in self._windows.values():
            if window.start_time <= ts <= window.end_time:
                if window.matches_metric(metric_name, labels):
                    return True
        return False

    def check_recalibrations(self) -> list[MaintenanceWindow]:
        """Check for windows that have ended and need recalibration.

        Returns:
            List of windows ready for recalibration.
        """
        now = time.time()
        ready = []

        for window in self._windows.values():
            if not window.recalibrate:
                continue
            if window.is_past and window.id not in self._pending_recalibrations:
                if now >= window.end_time + self.recalibration_buffer:
                    ready.append(window)
                    self._pending_recalibrations[window.id] = now

        return ready

    def active_windows(self) -> list[MaintenanceWindow]:
        """Return all currently active maintenance windows."""
        return [w for w in self._windows.values() if w.is_active]

    def cleanup_expired(self, max_age_hours: float = 24.0) -> int:
        """Remove expired windows older than max_age_hours.

        Returns:
            Number of windows removed.
        """
        cutoff = time.time() - (max_age_hours * 3600)
        to_remove = [
            wid for wid, w in self._windows.items()
            if w.is_past and w.end_time < cutoff
        ]
        for wid in to_remove:
            del self._windows[wid]
            self._pending_recalibrations.pop(wid, None)
        return len(to_remove)
