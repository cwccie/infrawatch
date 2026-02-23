"""Base classes for anomaly detection models.

All detection models implement the AnomalyDetector interface, producing
AnomalyScore objects that can be aggregated by the ensemble.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class AnomalyScore:
    """Anomaly detection result for a time series.

    Attributes:
        scores: Per-point anomaly scores (higher = more anomalous).
                Range varies by detector, normalized to [0, 1] for ensemble.
        is_anomaly: Per-point boolean anomaly flags.
        threshold: Threshold used for anomaly classification.
        detector_name: Name of the detector that produced this result.
        metadata: Additional detector-specific information.
    """

    scores: NDArray[np.float64]
    is_anomaly: NDArray[np.bool_]
    threshold: float = 0.0
    detector_name: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def anomaly_indices(self) -> NDArray[np.int64]:
        """Indices of detected anomalies."""
        return np.where(self.is_anomaly)[0]

    @property
    def anomaly_ratio(self) -> float:
        """Fraction of points flagged as anomalous."""
        if len(self.is_anomaly) == 0:
            return 0.0
        return float(np.mean(self.is_anomaly))

    @property
    def max_score(self) -> float:
        """Maximum anomaly score."""
        if len(self.scores) == 0:
            return 0.0
        return float(np.nanmax(self.scores))


class AnomalyDetector(ABC):
    """Abstract base class for anomaly detectors.

    All detectors must implement `fit` and `detect`. The `fit_detect`
    convenience method runs both steps.
    """

    name: str = "base"

    @abstractmethod
    def fit(self, values: NDArray[np.float64]) -> None:
        """Learn normal patterns from training data.

        Args:
            values: Historical time series values representing normal behavior.
        """

    @abstractmethod
    def detect(self, values: NDArray[np.float64]) -> AnomalyScore:
        """Detect anomalies in the given time series.

        Args:
            values: Time series values to check for anomalies.

        Returns:
            AnomalyScore with per-point scores and flags.
        """

    def fit_detect(self, values: NDArray[np.float64]) -> AnomalyScore:
        """Fit on and detect anomalies in the same data.

        Args:
            values: Time series values.

        Returns:
            AnomalyScore with per-point scores and flags.
        """
        self.fit(values)
        return self.detect(values)
