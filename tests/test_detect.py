"""Tests for the detection pipeline."""

import numpy as np
import time

from infrawatch.detect.pipeline import DetectionPipeline, DetectionResult
from infrawatch.detect.severity import Severity, classify_severity
from infrawatch.detect.context import ContextAnalyzer, TemporalContext


class TestSeverity:
    def test_severity_ordering(self):
        assert Severity.INFO < Severity.LOW < Severity.MEDIUM < Severity.HIGH < Severity.CRITICAL

    def test_classify_low_score(self):
        assert classify_severity(0.1) == Severity.INFO

    def test_classify_high_score(self):
        assert classify_severity(0.9) == Severity.CRITICAL

    def test_duration_escalation(self):
        base = classify_severity(0.5, duration_minutes=0)
        escalated = classify_severity(0.5, duration_minutes=35)
        assert escalated > base

    def test_critical_metric_escalation(self):
        normal = classify_severity(0.4, metric_type="cpu")
        critical = classify_severity(0.4, metric_type="error_rate")
        assert critical > normal

    def test_correlation_escalation(self):
        solo = classify_severity(0.5, is_correlated=False)
        corr = classify_severity(0.5, is_correlated=True)
        assert corr >= solo


class TestContextAnalyzer:
    def test_business_hours_multiplier(self):
        from datetime import datetime
        analyzer = ContextAnalyzer()
        # Construct a timestamp that is Tuesday 10 AM local time
        dt = datetime(2023, 11, 21, 10, 0, 0)  # Tuesday 10 AM local
        ts = dt.timestamp()
        mult = analyzer.get_threshold_multiplier(ts)
        assert mult == 1.0

    def test_correlate_metrics(self):
        analyzer = ContextAnalyzer()
        anomalies = {
            "cpu": [1000.0, 1005.0],
            "memory": [1002.0, 1010.0],
            "disk": [5000.0],
        }
        groups = analyzer.correlate_metrics(anomalies, time_window=60)
        # CPU and memory should be correlated
        found_cpu_mem = any(
            "cpu" in g and "memory" in g for g in groups
        )
        assert found_cpu_mem

    def test_no_correlation_distant_metrics(self):
        analyzer = ContextAnalyzer()
        anomalies = {
            "cpu": [1000.0],
            "memory": [9000.0],
        }
        groups = analyzer.correlate_metrics(anomalies, time_window=60)
        # Should not be correlated (8000s apart)
        assert len(groups) == 0


class TestDetectionPipeline:
    def test_pipeline_runs(self, anomalous_series):
        timestamps, values, _ = anomalous_series
        pipeline = DetectionPipeline()
        result = pipeline.run(timestamps, values, metric_name="test_metric")
        assert isinstance(result, DetectionResult)
        assert result.total_points == len(values)
        assert result.detection_time_ms > 0

    def test_pipeline_detects_anomalies(self):
        import numpy as np
        from infrawatch.models.statistical import ZScoreDetector
        from infrawatch.preprocess.pipeline import PreprocessConfig
        rng = np.random.RandomState(42)
        n = 500
        timestamps = np.linspace(0, n * 60, n)
        values = 50.0 + rng.normal(0, 3, n)
        # Inject very clear anomalies
        values[100] += 50.0
        values[200] += 50.0
        values[300] += 50.0
        # Disable outlier handling so anomalies survive to the detector
        config = PreprocessConfig(handle_outliers=False)
        pipeline = DetectionPipeline(
            detectors=[ZScoreDetector(threshold=3.0)],
            preprocess_config=config,
        )
        result = pipeline.run(timestamps, values, metric_name="test_metric")
        assert result.anomaly_count > 0

    def test_pipeline_normal_data(self, normal_series):
        timestamps, values = normal_series
        pipeline = DetectionPipeline()
        result = pipeline.run(timestamps, values, metric_name="test_metric")
        # Should have few or no anomalies on normal data
        assert result.anomaly_count < len(values) * 0.1
