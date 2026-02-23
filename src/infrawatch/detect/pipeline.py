"""Detection pipeline — orchestrates preprocessing, detection, and classification.

The main entry point for running anomaly detection on metric data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from infrawatch.models.base import AnomalyDetector, AnomalyScore
from infrawatch.models.statistical import ZScoreDetector, IQRDetector
from infrawatch.models.ensemble import EnsembleDetector, EnsembleConfig
from infrawatch.preprocess.pipeline import PreprocessPipeline, PreprocessConfig
from infrawatch.detect.severity import Severity, classify_severity
from infrawatch.detect.context import ContextAnalyzer


@dataclass
class Anomaly:
    """A detected anomaly event.

    Attributes:
        metric_name: Name of the metric.
        timestamp: When the anomaly occurred.
        value: Observed value.
        score: Anomaly score.
        severity: Classified severity level.
        detectors: List of detectors that flagged this point.
        labels: Metric labels.
    """

    metric_name: str
    timestamp: float
    value: float
    score: float
    severity: Severity
    detectors: list[str] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "timestamp": self.timestamp,
            "value": self.value,
            "score": self.score,
            "severity": self.severity.label,
            "detectors": self.detectors,
            "labels": self.labels,
        }


@dataclass
class DetectionResult:
    """Result of running the detection pipeline.

    Attributes:
        anomalies: Detected anomaly events.
        scores: Raw ensemble anomaly scores.
        total_points: Total number of data points analyzed.
        detection_time_ms: Time taken for detection in milliseconds.
    """

    anomalies: list[Anomaly] = field(default_factory=list)
    scores: AnomalyScore | None = None
    total_points: int = 0
    detection_time_ms: float = 0.0

    @property
    def anomaly_count(self) -> int:
        return len(self.anomalies)

    @property
    def max_severity(self) -> Severity:
        if not self.anomalies:
            return Severity.INFO
        return max(a.severity for a in self.anomalies)


class DetectionPipeline:
    """End-to-end anomaly detection pipeline.

    Chains preprocessing, model inference, and severity classification.

    Args:
        detectors: List of anomaly detectors. If None, uses default set.
        preprocess_config: Preprocessing configuration.
        ensemble_config: Ensemble voting configuration.
        context_analyzer: Optional context-aware threshold adjustment.
    """

    def __init__(
        self,
        detectors: list[AnomalyDetector] | None = None,
        preprocess_config: PreprocessConfig | None = None,
        ensemble_config: EnsembleConfig | None = None,
        context_analyzer: ContextAnalyzer | None = None,
    ):
        if detectors is None:
            detectors = [
                ZScoreDetector(threshold=3.0),
                IQRDetector(factor=1.5),
            ]

        self.ensemble = EnsembleDetector(
            detectors=detectors,
            config=ensemble_config or EnsembleConfig(strategy="majority"),
        )
        self.preprocessor = PreprocessPipeline(preprocess_config)
        self.context_analyzer = context_analyzer

    def run(
        self,
        timestamps: NDArray[np.float64],
        values: NDArray[np.float64],
        metric_name: str = "",
        labels: dict[str, str] | None = None,
    ) -> DetectionResult:
        """Run the full detection pipeline.

        Args:
            timestamps: Time series timestamps.
            values: Time series values.
            metric_name: Name of the metric being analyzed.
            labels: Metric labels for context.

        Returns:
            DetectionResult with all detected anomalies.
        """
        start_time = time.time()
        labels = labels or {}

        # Preprocess
        pp = self.preprocessor.process(timestamps, values, metric_name=metric_name)

        # Fit and detect
        self.ensemble.fit(pp.values)
        ensemble_result = self.ensemble.detect(pp.values)

        # Classify anomalies
        anomalies: list[Anomaly] = []
        for idx in ensemble_result.anomaly_indices:
            ts = pp.timestamps[idx] if idx < len(pp.timestamps) else 0.0
            val = pp.values[idx] if idx < len(pp.values) else 0.0
            score = ensemble_result.scores[idx]

            # Context-aware threshold adjustment
            if self.context_analyzer:
                multiplier = self.context_analyzer.get_threshold_multiplier(ts)
                if score < (0.5 * multiplier):
                    continue  # Suppress low-confidence alerts outside business hours

            severity = classify_severity(
                anomaly_score=float(score),
                metric_type=metric_name,
            )

            anomalies.append(Anomaly(
                metric_name=metric_name,
                timestamp=float(ts),
                value=float(val),
                score=float(score),
                severity=severity,
                labels=labels,
            ))

        elapsed_ms = (time.time() - start_time) * 1000

        return DetectionResult(
            anomalies=anomalies,
            scores=ensemble_result,
            total_points=len(values),
            detection_time_ms=elapsed_ms,
        )
