"""Foundation model adapters for time series anomaly detection.

Provides a unified interface for time series foundation models like
Chronos-Bolt and TimesFM. Includes a mock implementation for testing
without requiring GPU or model weights.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from infrawatch.models.base import AnomalyDetector, AnomalyScore


@dataclass
class ForecastResult:
    """Result from a foundation model forecast.

    Attributes:
        mean: Point forecast (mean prediction).
        lower: Lower prediction interval bound.
        upper: Upper prediction interval bound.
        confidence: Confidence level for the interval (e.g., 0.95).
    """

    mean: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    confidence: float = 0.95


class FoundationModelInterface(ABC):
    """Interface for time series foundation models."""

    @abstractmethod
    def forecast(
        self,
        context: NDArray[np.float64],
        horizon: int,
        confidence: float = 0.95,
    ) -> ForecastResult:
        """Generate a probabilistic forecast.

        Args:
            context: Historical time series values.
            horizon: Number of future steps to predict.
            confidence: Prediction interval confidence level.

        Returns:
            ForecastResult with mean and interval predictions.
        """


class MockFoundationModel(FoundationModelInterface):
    """Mock foundation model for testing.

    Generates forecasts by extrapolating recent trends with added noise.
    Prediction intervals widen with horizon.

    Args:
        noise_scale: Scale of random noise added to predictions.
        trend_window: Number of recent points to estimate trend from.
    """

    def __init__(self, noise_scale: float = 0.1, trend_window: int = 20):
        self.noise_scale = noise_scale
        self.trend_window = trend_window
        self._rng = np.random.RandomState(42)

    def forecast(
        self,
        context: NDArray[np.float64],
        horizon: int,
        confidence: float = 0.95,
    ) -> ForecastResult:
        clean = context[~np.isnan(context)]
        if len(clean) == 0:
            return ForecastResult(
                mean=np.zeros(horizon),
                lower=np.zeros(horizon),
                upper=np.zeros(horizon),
                confidence=confidence,
            )

        # Estimate trend from recent data
        window = min(self.trend_window, len(clean))
        recent = clean[-window:]
        if len(recent) >= 2:
            trend = np.polyfit(np.arange(len(recent)), recent, 1)[0]
        else:
            trend = 0.0

        last_val = clean[-1]
        mean_vals = np.array([
            last_val + trend * (i + 1)
            for i in range(horizon)
        ])

        # Add small noise
        noise = self._rng.normal(0, self.noise_scale * np.std(recent), horizon)
        mean_vals += noise

        # Prediction intervals widen with horizon
        from scipy.stats import norm
        z = norm.ppf((1 + confidence) / 2)
        base_std = np.std(recent) if len(recent) > 1 else 1.0
        widths = np.array([
            z * base_std * np.sqrt(1 + i / horizon)
            for i in range(horizon)
        ])

        return ForecastResult(
            mean=mean_vals,
            lower=mean_vals - widths,
            upper=mean_vals + widths,
            confidence=confidence,
        )


class FoundationModelAdapter(AnomalyDetector):
    """Adapts a foundation model for anomaly detection.

    Uses the foundation model to generate forecasts, then flags points
    where actual values fall outside prediction intervals as anomalous.

    Args:
        model: Foundation model instance. Uses MockFoundationModel if None.
        context_length: Number of historical points to provide as context.
        confidence: Prediction interval confidence level.
    """

    name = "foundation_model"

    def __init__(
        self,
        model: FoundationModelInterface | None = None,
        context_length: int = 100,
        confidence: float = 0.95,
    ):
        self.model = model or MockFoundationModel()
        self.context_length = context_length
        self.confidence = confidence
        self._baseline: NDArray[np.float64] | None = None

    def fit(self, values: NDArray[np.float64]) -> None:
        """Store baseline data for context window."""
        self._baseline = values.copy()

    def detect(self, values: NDArray[np.float64]) -> AnomalyScore:
        """Detect anomalies by comparing actuals against forecast intervals.

        For each point, uses preceding data as context to generate a
        one-step forecast, then checks if the actual falls within bounds.
        """
        n = len(values)
        scores = np.zeros(n)
        is_anomaly = np.zeros(n, dtype=bool)

        # Use baseline + current for full context
        if self._baseline is not None:
            full_series = np.concatenate([self._baseline, values])
            offset = len(self._baseline)
        else:
            full_series = values
            offset = 0

        # Step through and forecast 1-step ahead
        min_context = min(self.context_length, 20)

        for i in range(n):
            idx = offset + i
            if idx < min_context:
                continue

            context_start = max(0, idx - self.context_length)
            context = full_series[context_start:idx]

            result = self.model.forecast(context, horizon=1, confidence=self.confidence)

            actual = values[i]
            if np.isnan(actual):
                continue

            predicted = result.mean[0]
            lower = result.lower[0]
            upper = result.upper[0]

            # Score = normalized distance from predicted
            interval_width = max(upper - lower, 1e-10)
            deviation = abs(actual - predicted)
            scores[i] = deviation / (interval_width / 2)

            if actual < lower or actual > upper:
                is_anomaly[i] = True

        return AnomalyScore(
            scores=scores,
            is_anomaly=is_anomaly,
            threshold=1.0,
            detector_name=self.name,
        )


def load_chronos_model(model_name: str = "amazon/chronos-bolt-small"):
    """Load a Chronos-Bolt model (requires torch and chronos-forecasting).

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        A FoundationModelInterface wrapping the Chronos model.
    """
    try:
        import torch
        from chronos import ChronosPipeline
    except ImportError:
        raise ImportError(
            "Chronos requires torch and chronos-forecasting. "
            "Install with: pip install infrawatch[foundation]"
        )

    class ChronosAdapter(FoundationModelInterface):
        def __init__(self, pipeline):
            self._pipeline = pipeline

        def forecast(self, context, horizon, confidence=0.95):
            context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
            forecast = self._pipeline.predict(context_tensor, horizon)
            mean = forecast.numpy().mean(axis=1).squeeze()
            lower_q = (1 - confidence) / 2
            upper_q = 1 - lower_q
            lower = np.quantile(forecast.numpy().squeeze(0), lower_q, axis=0)
            upper = np.quantile(forecast.numpy().squeeze(0), upper_q, axis=0)
            return ForecastResult(mean=mean, lower=lower, upper=upper, confidence=confidence)

    pipeline = ChronosPipeline.from_pretrained(model_name)
    return ChronosAdapter(pipeline)
