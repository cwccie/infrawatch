"""Tests for anomaly detection models."""

import numpy as np
import pytest

from infrawatch.models.statistical import ZScoreDetector, IQRDetector, GESDDetector, STLDetector
from infrawatch.models.ml import IsolationForestDetector, LOFDetector, AutoencoderDetector
from infrawatch.models.foundation import MockFoundationModel, FoundationModelAdapter
from infrawatch.models.ensemble import EnsembleDetector, EnsembleConfig


class TestZScoreDetector:
    def test_detects_spikes(self, anomalous_series):
        _, values, known_indices = anomalous_series
        detector = ZScoreDetector(threshold=3.0)
        result = detector.fit_detect(values)
        detected = set(result.anomaly_indices)
        for idx in known_indices:
            assert idx in detected or any(abs(idx - d) <= 2 for d in detected)

    def test_no_false_positives_normal(self, normal_series):
        _, values = normal_series
        detector = ZScoreDetector(threshold=4.0)
        result = detector.fit_detect(values)
        # Should be very few false positives with threshold=4
        assert result.anomaly_ratio < 0.05

    def test_sliding_window(self, anomalous_series):
        _, values, _ = anomalous_series
        detector = ZScoreDetector(threshold=3.0, window=50)
        result = detector.fit_detect(values)
        assert result.anomaly_ratio > 0


class TestIQRDetector:
    def test_detects_outliers(self, anomalous_series):
        _, values, known_indices = anomalous_series
        detector = IQRDetector(factor=1.5)
        result = detector.fit_detect(values)
        assert result.anomaly_ratio > 0
        assert "lower_fence" in result.metadata

    def test_no_anomalies_in_tight_data(self):
        values = np.array([50, 51, 49, 50, 52, 48, 50], dtype=np.float64)
        detector = IQRDetector(factor=3.0)
        result = detector.fit_detect(values)
        assert result.anomaly_ratio == 0


class TestGESDDetector:
    def test_finds_outliers(self, anomalous_series):
        _, values, known_indices = anomalous_series
        detector = GESDDetector(max_anomalies=10)
        result = detector.fit_detect(values)
        assert result.anomaly_ratio > 0


class TestSTLDetector:
    def test_detects_residual_anomalies(self):
        n = 200
        period = 24
        values = 10 * np.sin(2 * np.pi * np.arange(n) / period) + 50
        values[100] += 40  # Big anomaly
        detector = STLDetector(period=period, residual_threshold=3.0)
        result = detector.fit_detect(values)
        assert 100 in result.anomaly_indices


class TestIsolationForest:
    def test_detects_anomalies(self, anomalous_series):
        _, values, _ = anomalous_series
        detector = IsolationForestDetector(contamination=0.05, window=5)
        result = detector.fit_detect(values)
        assert result.anomaly_ratio > 0
        assert result.anomaly_ratio < 0.2  # Not too many


class TestLOF:
    def test_detects_anomalies(self, anomalous_series):
        _, values, _ = anomalous_series
        detector = LOFDetector(n_neighbors=10, contamination=0.05, window=5)
        result = detector.fit_detect(values)
        assert result.anomaly_ratio > 0


class TestAutoencoder:
    def test_detects_anomalies(self, anomalous_series):
        _, values, _ = anomalous_series
        detector = AutoencoderDetector(
            hidden_dim=3, window=5, epochs=20, threshold_percentile=95.0,
        )
        result = detector.fit_detect(values)
        assert result.anomaly_ratio > 0

    def test_trains_without_error(self, normal_series):
        _, values = normal_series
        detector = AutoencoderDetector(hidden_dim=3, window=5, epochs=10)
        detector.fit(values)
        result = detector.detect(values)
        assert len(result.scores) == len(values)


class TestFoundationModel:
    def test_mock_forecast(self, normal_series):
        _, values = normal_series
        model = MockFoundationModel()
        result = model.forecast(values, horizon=24)
        assert len(result.mean) == 24
        assert len(result.lower) == 24
        assert len(result.upper) == 24
        assert all(result.lower <= result.mean)
        assert all(result.mean <= result.upper)

    def test_adapter_detects_anomalies(self, anomalous_series):
        _, values, _ = anomalous_series
        adapter = FoundationModelAdapter(context_length=50)
        result = adapter.fit_detect(values)
        assert len(result.scores) == len(values)


class TestEnsemble:
    def test_majority_voting(self, anomalous_series):
        _, values, _ = anomalous_series
        ensemble = EnsembleDetector(
            detectors=[
                ZScoreDetector(threshold=3.0),
                IQRDetector(factor=1.5),
            ],
            config=EnsembleConfig(strategy="majority"),
        )
        result = ensemble.fit_detect(values)
        assert result.anomaly_ratio > 0
        assert "n_detectors" in result.metadata
        assert result.metadata["n_detectors"] == 2

    def test_unanimous_voting(self, anomalous_series):
        _, values, _ = anomalous_series
        majority = EnsembleDetector(
            detectors=[ZScoreDetector(threshold=3.0), IQRDetector(factor=1.5)],
            config=EnsembleConfig(strategy="majority"),
        )
        unanimous = EnsembleDetector(
            detectors=[ZScoreDetector(threshold=3.0), IQRDetector(factor=1.5)],
            config=EnsembleConfig(strategy="unanimous"),
        )
        r_majority = majority.fit_detect(values)
        r_unanimous = unanimous.fit_detect(values)
        # Unanimous should flag fewer or equal anomalies
        assert np.sum(r_unanimous.is_anomaly) <= np.sum(r_majority.is_anomaly)

    def test_any_voting(self, anomalous_series):
        _, values, _ = anomalous_series
        ensemble = EnsembleDetector(
            detectors=[ZScoreDetector(threshold=3.0), IQRDetector(factor=1.5)],
            config=EnsembleConfig(strategy="any"),
        )
        result = ensemble.fit_detect(values)
        assert result.anomaly_ratio > 0

    def test_threshold_voting(self, anomalous_series):
        _, values, _ = anomalous_series
        ensemble = EnsembleDetector(
            detectors=[ZScoreDetector(threshold=3.0), IQRDetector(factor=1.5)],
            config=EnsembleConfig(strategy="threshold", threshold=0.6),
        )
        result = ensemble.fit_detect(values)
        assert len(result.scores) == len(values)

    def test_empty_ensemble(self, normal_series):
        _, values = normal_series
        ensemble = EnsembleDetector(detectors=[])
        result = ensemble.detect(values)
        assert result.anomaly_ratio == 0
