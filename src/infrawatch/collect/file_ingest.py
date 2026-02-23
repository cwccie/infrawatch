"""File-based metric ingestion.

Reads metrics from CSV and JSON files, converting them into InfraWatch's
unified Metric format. Useful for batch processing, historical data import,
and testing.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Optional, Union

from infrawatch.collect.metric import Metric, MetricBatch

logger = logging.getLogger(__name__)


class FileIngestor:
    """Ingests metrics from CSV and JSON files.

    CSV format expected columns: name, value, timestamp (optional), plus any
    additional columns treated as labels.

    JSON format: list of objects with 'name', 'value', 'timestamp', 'labels'.

    Args:
        default_labels: Labels added to every ingested metric.
    """

    def __init__(self, default_labels: Optional[dict[str, str]] = None):
        self.default_labels = default_labels or {}

    def ingest_csv(
        self,
        path: Union[str, Path],
        name_col: str = "name",
        value_col: str = "value",
        timestamp_col: str = "timestamp",
        delimiter: str = ",",
    ) -> MetricBatch:
        """Read metrics from a CSV file.

        Args:
            path: Path to CSV file.
            name_col: Column name for metric name.
            value_col: Column name for metric value.
            timestamp_col: Column name for timestamp (optional in data).
            delimiter: CSV delimiter character.

        Returns:
            MetricBatch with all ingested metrics.
        """
        path = Path(path)
        batch = MetricBatch(collector_id=f"csv-{path.name}")

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                if name_col not in row or value_col not in row:
                    logger.warning("Skipping row missing required columns: %s", row)
                    continue

                try:
                    value = float(row[value_col])
                except (ValueError, TypeError):
                    logger.warning("Skipping non-numeric value: %s", row[value_col])
                    continue

                ts = time.time()
                if timestamp_col in row and row[timestamp_col]:
                    try:
                        ts = float(row[timestamp_col])
                    except ValueError:
                        pass

                # All other columns become labels
                labels = dict(self.default_labels)
                for k, v in row.items():
                    if k not in (name_col, value_col, timestamp_col) and v:
                        labels[k] = v

                batch.add(Metric(
                    name=row[name_col],
                    value=value,
                    timestamp=ts,
                    labels=labels,
                    source=f"file://{path}",
                ))

        logger.info("Ingested %d metrics from CSV %s", len(batch), path)
        return batch

    def ingest_json(self, path: Union[str, Path]) -> MetricBatch:
        """Read metrics from a JSON file.

        Expects either a JSON array of metric objects or an object with a
        'metrics' key containing the array.

        Args:
            path: Path to JSON file.

        Returns:
            MetricBatch with all ingested metrics.
        """
        path = Path(path)
        batch = MetricBatch(collector_id=f"json-{path.name}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = data.get("metrics", [])
        if not isinstance(data, list):
            raise ValueError(f"Expected list of metrics in {path}, got {type(data).__name__}")

        for entry in data:
            if not isinstance(entry, dict):
                continue
            if "name" not in entry or "value" not in entry:
                logger.warning("Skipping entry missing name/value: %s", entry)
                continue

            try:
                value = float(entry["value"])
            except (ValueError, TypeError):
                continue

            labels = {**self.default_labels, **entry.get("labels", {})}

            batch.add(Metric(
                name=entry["name"],
                value=value,
                timestamp=entry.get("timestamp", time.time()),
                labels=labels,
                source=f"file://{path}",
                unit=entry.get("unit", ""),
            ))

        logger.info("Ingested %d metrics from JSON %s", len(batch), path)
        return batch

    def ingest(self, path: Union[str, Path]) -> MetricBatch:
        """Auto-detect file type and ingest.

        Args:
            path: Path to CSV or JSON file.

        Returns:
            MetricBatch with all ingested metrics.
        """
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".csv":
            return self.ingest_csv(path)
        elif suffix == ".json":
            return self.ingest_json(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix} (expected .csv or .json)")
