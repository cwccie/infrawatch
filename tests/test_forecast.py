"""Tests for the forecasting engine."""

import numpy as np
from infrawatch.forecast.engine import ForecastEngine


class TestForecastEngine:
    def test_basic_forecast(self, normal_series):
        timestamps, values = normal_series
        engine = ForecastEngine()
        result = engine.forecast(timestamps, values, horizon=48)
        assert len(result.predicted) == 48
        assert len(result.lower_bound) == 48
        assert len(result.upper_bound) == 48
        # Lower should be <= predicted <= upper
        assert np.all(result.lower_bound <= result.predicted + 1e-10)
        assert np.all(result.predicted <= result.upper_bound + 1e-10)

    def test_trend_detection(self):
        n = 200
        timestamps = np.linspace(0, n * 60, n)
        values = np.linspace(10, 80, n) + np.random.RandomState(42).normal(0, 2, n)
        engine = ForecastEngine()
        summary = engine.trend_summary(timestamps, values)
        assert summary["direction"] == "increasing"
        assert summary["significant"]
        assert summary["slope"] > 0

    def test_capacity_exhaustion(self):
        n = 200
        timestamps = np.linspace(0, n * 60, n)
        values = np.linspace(50, 80, n) + np.random.RandomState(42).normal(0, 1, n)
        engine = ForecastEngine(capacity_threshold=100.0)
        result = engine.forecast(timestamps, values, horizon=200)
        assert result.days_to_exhaustion is not None
        assert result.days_to_exhaustion > 0

    def test_sla_risk(self):
        n = 200
        timestamps = np.linspace(0, n * 60, n)
        values = np.linspace(80, 95, n) + np.random.RandomState(42).normal(0, 2, n)
        engine = ForecastEngine(sla_threshold=98.0)
        result = engine.forecast(timestamps, values, horizon=48)
        assert result.sla_risk_score > 0

    def test_insufficient_data(self):
        timestamps = np.array([0, 60])
        values = np.array([50, 51])
        engine = ForecastEngine()
        result = engine.forecast(timestamps, values)
        assert len(result.predicted) == 0
