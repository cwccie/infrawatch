"""Capacity forecasting.

Trend projection, seasonality-aware prediction, SLA violation risk,
and capacity exhaustion date estimation.
"""

from infrawatch.forecast.engine import ForecastEngine, CapacityForecast

__all__ = ["ForecastEngine", "CapacityForecast"]
