"""Ensemble anomaly detection with consensus voting.

Combines results from multiple anomaly detectors using configurable
voting strategies. Reduces false positives by requiring agreement
across independent detection methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from infrawatch.models.base import AnomalyDetector, AnomalyScore


@dataclass
class EnsembleConfig:
    """Configuration for ensemble voting.

    Attributes:
        strategy: Voting strategy — 'majority', 'unanimous', 'any',
                  'weighted', or 'threshold'.
        threshold: For 'threshold' strategy, minimum fraction of detectors
                   that must agree (e.g., 0.6 = 60%).
        weights: For 'weighted' strategy, mapping of detector name to weight.
                 Detectors not in the map get weight 1.0.
    """

    strategy: str = "majority"
    threshold: float = 0.5
    weights: dict[str, float] = field(default_factory=dict)


class EnsembleDetector(AnomalyDetector):
    """Combines multiple anomaly detectors via consensus voting.

    Args:
        detectors: List of anomaly detectors to combine.
        config: Ensemble configuration.
    """

    name = "ensemble"

    def __init__(
        self,
        detectors: list[AnomalyDetector] | None = None,
        config: EnsembleConfig | None = None,
    ):
        self.detectors = detectors or []
        self.config = config or EnsembleConfig()

    def add_detector(self, detector: AnomalyDetector) -> None:
        """Add a detector to the ensemble."""
        self.detectors.append(detector)

    def fit(self, values: NDArray[np.float64]) -> None:
        """Fit all detectors on the training data."""
        for detector in self.detectors:
            detector.fit(values)

    def detect(self, values: NDArray[np.float64]) -> AnomalyScore:
        """Run all detectors and combine results via voting.

        Returns:
            AnomalyScore with combined scores and consensus-based flags.
        """
        if not self.detectors:
            return AnomalyScore(
                scores=np.zeros(len(values)),
                is_anomaly=np.zeros(len(values), dtype=bool),
                detector_name=self.name,
            )

        # Collect results from all detectors
        results: list[AnomalyScore] = []
        for detector in self.detectors:
            try:
                result = detector.detect(values)
                results.append(result)
            except Exception:
                pass

        if not results:
            return AnomalyScore(
                scores=np.zeros(len(values)),
                is_anomaly=np.zeros(len(values), dtype=bool),
                detector_name=self.name,
            )

        n = len(values)
        n_detectors = len(results)

        # Combine scores (weighted average)
        combined_scores = np.zeros(n)
        total_weight = 0.0

        for result in results:
            w = self.config.weights.get(result.detector_name, 1.0)
            # Normalize scores to [0, 1] range
            s = result.scores.copy()
            s_max = np.nanmax(s)
            if s_max > 0:
                s = s / s_max
            combined_scores += w * s
            total_weight += w

        if total_weight > 0:
            combined_scores /= total_weight

        # Vote on anomaly flags
        votes = np.zeros(n, dtype=int)
        weighted_votes = np.zeros(n, dtype=float)

        for result in results:
            w = self.config.weights.get(result.detector_name, 1.0)
            votes += result.is_anomaly.astype(int)
            weighted_votes += result.is_anomaly.astype(float) * w

        # Apply voting strategy
        strategy = self.config.strategy

        if strategy == "majority":
            is_anomaly = votes > (n_detectors / 2)
        elif strategy == "unanimous":
            is_anomaly = votes == n_detectors
        elif strategy == "any":
            is_anomaly = votes > 0
        elif strategy == "weighted":
            total_possible = sum(
                self.config.weights.get(r.detector_name, 1.0) for r in results
            )
            is_anomaly = weighted_votes > (total_possible / 2)
        elif strategy == "threshold":
            is_anomaly = (votes / n_detectors) >= self.config.threshold
        else:
            raise ValueError(f"Unknown voting strategy: {strategy}")

        return AnomalyScore(
            scores=combined_scores,
            is_anomaly=is_anomaly,
            threshold=0.5,
            detector_name=self.name,
            metadata={
                "n_detectors": n_detectors,
                "strategy": strategy,
                "votes": votes.tolist(),
                "individual_results": [
                    {
                        "detector": r.detector_name,
                        "anomaly_count": int(np.sum(r.is_anomaly)),
                        "max_score": float(r.max_score),
                    }
                    for r in results
                ],
            },
        )

    def fit_detect(self, values: NDArray[np.float64]) -> AnomalyScore:
        """Fit all detectors and run ensemble detection."""
        self.fit(values)
        return self.detect(values)
