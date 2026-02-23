"""Capacity forecasting engine.

Projects trends and seasonal patterns to predict future capacity needs,
SLA violation risk, and resource exhaustion dates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

from infrawatch.preprocess.decompose import seasonal_decompose


@dataclass
class CapacityForecast:
    """Forecast result with capacity projections.

    Attributes:
        timestamps: Future timestamp array.
        predicted: Point predictions.
        lower_bound: Lower prediction interval.
        upper_bound: Upper prediction interval.
        confidence: Confidence level.
        trend_slope: Estimated trend slope per unit time.
        exhaustion_timestamp: Predicted time when capacity is exhausted (None if N/A).
        sla_risk_score: Risk score for SLA violation (0-1).
        days_to_exhaustion: Estimated days until capacity exhaustion.
    """

    timestamps: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    predicted: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    lower_bound: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    upper_bound: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    confidence: float = 0.95
    trend_slope: float = 0.0
    exhaustion_timestamp: float | None = None
    sla_risk_score: float = 0.0
    days_to_exhaustion: float | None = None

    def to_dict(self) -> dict:
        result = {
            "confidence": self.confidence,
            "trend_slope": self.trend_slope,
            "sla_risk_score": self.sla_risk_score,
            "n_points": len(self.predicted),
        }
        if self.exhaustion_timestamp:
            result["exhaustion_date"] = datetime.fromtimestamp(
                self.exhaustion_timestamp
            ).isoformat()
        if self.days_to_exhaustion is not None:
            result["days_to_exhaustion"] = self.days_to_exhaustion
        return result


class ForecastEngine:
    """Capacity forecasting engine.

    Combines trend analysis, seasonal decomposition, and statistical
    projection for infrastructure capacity planning.

    Args:
        confidence: Prediction interval confidence level.
        capacity_threshold: Value at which capacity is considered exhausted
                           (e.g., 100 for percentage metrics, None for auto).
        sla_threshold: Value above which SLA is violated.
    """

    def __init__(
        self,
        confidence: float = 0.95,
        capacity_threshold: float | None = None,
        sla_threshold: float | None = None,
    ):
        self.confidence = confidence
        self.capacity_threshold = capacity_threshold
        self.sla_threshold = sla_threshold

    def forecast(
        self,
        timestamps: NDArray[np.float64],
        values: NDArray[np.float64],
        horizon: int = 168,
        interval: float | None = None,
    ) -> CapacityForecast:
        """Generate a capacity forecast.

        Args:
            timestamps: Historical timestamps.
            values: Historical values.
            horizon: Number of future points to forecast.
            interval: Time interval between forecast points.
                     Defaults to median observed interval.

        Returns:
            CapacityForecast with projections and risk analysis.
        """
        if len(timestamps) < 3:
            return CapacityForecast()

        # Determine interval
        if interval is None:
            diffs = np.diff(timestamps)
            interval = float(np.median(diffs))

        # Clean NaN values
        valid = ~np.isnan(values)
        clean_ts = timestamps[valid]
        clean_vals = values[valid]

        if len(clean_vals) < 3:
            return CapacityForecast()

        # Fit linear trend
        x = np.arange(len(clean_vals), dtype=np.float64)
        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, clean_vals)

        # Try seasonal decomposition
        seasonal_component = np.zeros(horizon)
        try:
            if len(clean_vals) >= 14:
                decomp = seasonal_decompose(clean_vals)
                period = decomp.period
                # Project seasonal pattern
                for i in range(horizon):
                    pos = (len(clean_vals) + i) % period
                    if pos < len(decomp.seasonal):
                        seasonal_component[i] = decomp.seasonal[pos]
        except Exception:
            pass

        # Generate future timestamps
        last_ts = timestamps[-1]
        future_ts = np.array([last_ts + (i + 1) * interval for i in range(horizon)])

        # Project trend
        future_x = np.arange(len(clean_vals), len(clean_vals) + horizon, dtype=np.float64)
        trend_projection = slope * future_x + intercept

        # Combine trend + seasonal
        predicted = trend_projection + seasonal_component

        # Prediction intervals
        n = len(clean_vals)
        residuals = clean_vals - (slope * x + intercept)
        residual_std = float(np.std(residuals))

        z = sp_stats.norm.ppf((1 + self.confidence) / 2)
        # Interval widens with forecast horizon
        widths = np.array([
            z * residual_std * np.sqrt(1 + 1/n + (fi - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
            for fi in future_x
        ])

        lower = predicted - widths
        upper = predicted + widths

        # Capacity exhaustion calculation
        exhaustion_ts = None
        days_to_exhaustion = None

        if self.capacity_threshold is not None and slope > 0:
            # When does the trend line cross the threshold?
            current_level = slope * len(clean_vals) + intercept
            if current_level < self.capacity_threshold:
                steps_to_exhaust = (self.capacity_threshold - current_level) / slope
                exhaustion_ts = last_ts + steps_to_exhaust * interval
                days_to_exhaustion = (steps_to_exhaust * interval) / 86400.0

        # SLA risk score
        sla_risk = 0.0
        if self.sla_threshold is not None:
            # Probability of exceeding SLA threshold within forecast horizon
            for i in range(min(horizon, 24)):  # Check first 24 points
                mean_val = predicted[i]
                std_val = widths[i] / z if z > 0 else residual_std
                if std_val > 0:
                    prob_exceed = 1 - sp_stats.norm.cdf(self.sla_threshold, mean_val, std_val)
                    sla_risk = max(sla_risk, prob_exceed)

        # Convert slope to per-unit-time
        trend_per_time = slope / interval if interval > 0 else slope

        return CapacityForecast(
            timestamps=future_ts,
            predicted=predicted,
            lower_bound=lower,
            upper_bound=upper,
            confidence=self.confidence,
            trend_slope=trend_per_time,
            exhaustion_timestamp=exhaustion_ts,
            sla_risk_score=sla_risk,
            days_to_exhaustion=days_to_exhaustion,
        )

    def trend_summary(
        self,
        timestamps: NDArray[np.float64],
        values: NDArray[np.float64],
    ) -> dict:
        """Generate a summary of trend characteristics.

        Args:
            timestamps: Historical timestamps.
            values: Historical values.

        Returns:
            Dict with trend analysis results.
        """
        valid = ~np.isnan(values)
        clean_vals = values[valid]

        if len(clean_vals) < 3:
            return {"status": "insufficient_data"}

        x = np.arange(len(clean_vals), dtype=np.float64)
        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, clean_vals)

        return {
            "slope": float(slope),
            "r_squared": float(r_value ** 2),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
            "current_value": float(clean_vals[-1]),
            "mean": float(np.mean(clean_vals)),
            "std": float(np.std(clean_vals)),
        }
