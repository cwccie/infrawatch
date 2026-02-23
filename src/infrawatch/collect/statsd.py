"""StatsD metric receiver.

Parses StatsD-format metric strings (counters, gauges, timers, sets)
and converts them into InfraWatch's unified Metric format.
"""

from __future__ import annotations

import time
import logging
import socket
import threading
from typing import Optional

from infrawatch.collect.metric import Metric, MetricBatch

logger = logging.getLogger(__name__)


def parse_statsd_line(line: str, timestamp: Optional[float] = None) -> Optional[Metric]:
    """Parse a single StatsD protocol line into a Metric.

    Supports formats:
        metric_name:value|type
        metric_name:value|type|@sample_rate
        metric_name:value|type|#tag1:val1,tag2:val2

    Args:
        line: Raw StatsD line.
        timestamp: Override timestamp (defaults to now).

    Returns:
        Metric if parseable, None otherwise.
    """
    line = line.strip()
    if not line:
        return None

    ts = timestamp or time.time()

    # Split name:value|type[|extras]
    try:
        name_part, rest = line.split(":", 1)
    except ValueError:
        logger.warning("Malformed StatsD line (no colon): %s", line)
        return None

    parts = rest.split("|")
    if len(parts) < 2:
        logger.warning("Malformed StatsD line (missing type): %s", line)
        return None

    try:
        value = float(parts[0])
    except ValueError:
        logger.warning("Non-numeric StatsD value: %s", parts[0])
        return None

    metric_type = parts[1].strip()
    type_map = {"c": "counter", "g": "gauge", "ms": "timer", "s": "set", "h": "histogram"}
    friendly_type = type_map.get(metric_type, metric_type)

    # Parse optional tags (DogStatsD format: #tag1:val1,tag2:val2)
    labels: dict[str, str] = {"type": friendly_type}
    sample_rate = 1.0

    for extra in parts[2:]:
        extra = extra.strip()
        if extra.startswith("@"):
            try:
                sample_rate = float(extra[1:])
            except ValueError:
                pass
        elif extra.startswith("#"):
            tag_str = extra[1:]
            for tag in tag_str.split(","):
                if ":" in tag:
                    k, v = tag.split(":", 1)
                    labels[k.strip()] = v.strip()
                else:
                    labels[tag.strip()] = "true"

    # Adjust counter value by sample rate
    if friendly_type == "counter" and sample_rate > 0 and sample_rate != 1.0:
        value = value / sample_rate

    return Metric(
        name=name_part.strip(),
        value=value,
        timestamp=ts,
        labels=labels,
        source="statsd",
    )


def parse_statsd_batch(data: str, timestamp: Optional[float] = None) -> MetricBatch:
    """Parse multiple StatsD lines (newline-separated) into a MetricBatch.

    Args:
        data: Raw StatsD data (may contain multiple newline-separated metrics).
        timestamp: Override timestamp for all metrics.

    Returns:
        MetricBatch with all parsed metrics.
    """
    batch = MetricBatch(collector_id="statsd")
    for line in data.splitlines():
        metric = parse_statsd_line(line, timestamp)
        if metric is not None:
            batch.add(metric)
    return batch


class StatsDReceiver:
    """UDP receiver for StatsD protocol metrics.

    Listens on a UDP socket and buffers received metrics.

    Args:
        host: Bind address.
        port: UDP port to listen on.
        buffer_size: Maximum UDP packet size.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8125, buffer_size: int = 8192):
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self._buffer: list[Metric] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _listen(self) -> None:
        """Internal listener loop."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.settimeout(1.0)

        logger.info("StatsD receiver listening on %s:%d", self.host, self.port)

        while self._running:
            try:
                data, _ = sock.recvfrom(self.buffer_size)
                batch = parse_statsd_batch(data.decode("utf-8", errors="replace"))
                with self._lock:
                    self._buffer.extend(batch.metrics)
            except socket.timeout:
                continue
            except Exception as exc:
                logger.error("StatsD receive error: %s", exc)

        sock.close()

    def start(self) -> None:
        """Start the background listener thread."""
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background listener."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def flush(self) -> MetricBatch:
        """Return all buffered metrics and clear the buffer.

        Returns:
            MetricBatch with all metrics received since last flush.
        """
        with self._lock:
            metrics = list(self._buffer)
            self._buffer.clear()

        batch = MetricBatch(collector_id="statsd")
        batch.metrics = metrics
        return batch

    def receive_text(self, data: str) -> MetricBatch:
        """Parse StatsD text directly without the UDP socket (for testing).

        Args:
            data: StatsD-format text.

        Returns:
            Parsed MetricBatch.
        """
        return parse_statsd_batch(data)
