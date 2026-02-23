"""Prometheus metric scraper.

Scrapes /metrics endpoints exposing Prometheus exposition format and converts
each sample into InfraWatch's unified Metric objects.
"""

from __future__ import annotations

import re
import time
import logging
from typing import Optional

from infrawatch.collect.metric import Metric, MetricBatch

logger = logging.getLogger(__name__)

# Matches lines like:  metric_name{label="val"} 1.23 1234567890
_METRIC_LINE_RE = re.compile(
    r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)'
    r'(?:\{(?P<labels>[^}]*)\})?\s+'
    r'(?P<value>[^\s]+)'
    r'(?:\s+(?P<timestamp>\d+))?$'
)

_LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


def parse_prometheus_text(text: str, source: str = "prometheus") -> MetricBatch:
    """Parse Prometheus exposition format text into a MetricBatch.

    Handles HELP/TYPE comment lines (skips them) and parses metric lines
    with optional labels and timestamps.

    Args:
        text: Raw Prometheus exposition format text.
        source: Source identifier to attach to each Metric.

    Returns:
        MetricBatch containing all parsed metrics.
    """
    batch = MetricBatch(collector_id=source)
    now = time.time()

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        match = _METRIC_LINE_RE.match(line)
        if not match:
            continue

        name = match.group("name")

        # Parse value — handle special Prometheus values
        raw_value = match.group("value")
        if raw_value in ("+Inf", "Inf"):
            value = float("inf")
        elif raw_value == "-Inf":
            value = float("-inf")
        elif raw_value == "NaN":
            value = float("nan")
        else:
            try:
                value = float(raw_value)
            except ValueError:
                logger.warning("Skipping unparseable value: %s", raw_value)
                continue

        # Parse labels
        labels: dict[str, str] = {}
        raw_labels = match.group("labels")
        if raw_labels:
            for lm in _LABEL_RE.finditer(raw_labels):
                labels[lm.group(1)] = lm.group(2)

        # Parse timestamp (Prometheus uses milliseconds)
        raw_ts = match.group("timestamp")
        if raw_ts:
            ts = int(raw_ts) / 1000.0
        else:
            ts = now

        batch.add(Metric(
            name=name,
            value=value,
            timestamp=ts,
            labels=labels,
            source=source,
        ))

    return batch


class PrometheusCollector:
    """Scrapes Prometheus /metrics endpoints.

    Args:
        targets: List of base URLs to scrape (e.g., ['http://host:9090']).
        metrics_path: Path appended to each target URL.
        timeout: HTTP request timeout in seconds.
        extra_labels: Labels added to every scraped metric.
    """

    def __init__(
        self,
        targets: Optional[list[str]] = None,
        metrics_path: str = "/metrics",
        timeout: float = 10.0,
        extra_labels: Optional[dict[str, str]] = None,
    ):
        self.targets = targets or []
        self.metrics_path = metrics_path
        self.timeout = timeout
        self.extra_labels = extra_labels or {}

    def scrape(self, target: Optional[str] = None) -> MetricBatch:
        """Scrape a single target and return parsed metrics.

        Args:
            target: URL to scrape. If None, scrapes the first configured target.

        Returns:
            MetricBatch with all parsed metrics.
        """
        import requests

        url = target or (self.targets[0] if self.targets else "")
        if not url:
            raise ValueError("No target URL provided or configured")

        full_url = url.rstrip("/") + self.metrics_path
        logger.info("Scraping %s", full_url)

        resp = requests.get(full_url, timeout=self.timeout)
        resp.raise_for_status()

        batch = parse_prometheus_text(resp.text, source=full_url)

        # Apply extra labels
        if self.extra_labels:
            enriched = MetricBatch(collector_id=batch.collector_id)
            for m in batch:
                enriched.add(m.with_labels(**self.extra_labels))
            return enriched

        return batch

    def scrape_all(self) -> MetricBatch:
        """Scrape all configured targets and return a combined batch."""
        combined = MetricBatch(collector_id="prometheus-multi")
        for target in self.targets:
            try:
                batch = self.scrape(target)
                for m in batch:
                    combined.add(m)
            except Exception as exc:
                logger.error("Failed to scrape %s: %s", target, exc)
        return combined

    def scrape_text(self, text: str) -> MetricBatch:
        """Parse raw Prometheus text directly (useful for testing).

        Args:
            text: Prometheus exposition format text.

        Returns:
            MetricBatch with all parsed metrics.
        """
        batch = parse_prometheus_text(text, source="text-input")
        if self.extra_labels:
            enriched = MetricBatch(collector_id=batch.collector_id)
            for m in batch:
                enriched.add(m.with_labels(**self.extra_labels))
            return enriched
        return batch
