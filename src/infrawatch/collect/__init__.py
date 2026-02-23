"""Metric collection subsystem.

Provides unified metric ingestion from Prometheus, SNMP, StatsD, and file sources.
All collectors emit a common Metric dataclass for downstream processing.
"""

from infrawatch.collect.metric import Metric, MetricBatch
from infrawatch.collect.prometheus import PrometheusCollector
from infrawatch.collect.snmp import SNMPPoller
from infrawatch.collect.statsd import StatsDReceiver
from infrawatch.collect.file_ingest import FileIngestor

__all__ = [
    "Metric",
    "MetricBatch",
    "PrometheusCollector",
    "SNMPPoller",
    "StatsDReceiver",
    "FileIngestor",
]
