"""Tests for preprocessing modules."""

import numpy as np
import pytest

from infrawatch.preprocess.counters import unwrap_counters, detect_counter_wraps, is_counter_metric
from infrawatch.preprocess.gaps import fill_gaps, detect_gaps
from infrawatch.preprocess.outliers import (
    detect_outliers_iqr, detect_outliers_zscore, clip_outliers, replace_outliers,
)
from infrawatch.preprocess.resample import resample_uniform, downsample
from infrawatch.preprocess.decompose import seasonal_decompose, estimate_period
from infrawatch.preprocess.normalize import z_normalize, minmax_normalize, robust_normalize
from infrawatch.preprocess.pipeline import PreprocessPipeline, PreprocessConfig


class TestCounters:
    def test_unwrap_no_wraps(self):
        values = np.array([0, 100, 200, 300, 400], dtype=np.float64)
        deltas = unwrap_counters(values)
        assert deltas[0] == 0
        assert deltas[1] == 100
        assert deltas[4] == 100

    def test_unwrap_32bit_wrap(self):
        values = np.array([2**32 - 100, 50], dtype=np.float64)
        deltas = unwrap_counters(values, bits=32)
        assert deltas[1] == 150

    def test_unwrap_with_max_rate(self):
        values = np.array([100, 200, 50, 150], dtype=np.float64)
        deltas = unwrap_counters(values, max_rate=200)
        # The wrap from 200->50 creates a huge delta; max_rate should zero it
        assert deltas[2] == 0

    def test_detect_wraps(self):
        values = np.array([100, 200, 50, 100], dtype=np.float64)
        wraps = detect_counter_wraps(values)
        assert not wraps[0]
        assert not wraps[1]
        assert wraps[2]  # Wrap happened here

    def test_is_counter_metric(self):
        assert is_counter_metric("http_requests_total")
        assert is_counter_metric("ifInOctets")
        assert is_counter_metric("network_bytes_total")
        assert not is_counter_metric("cpu_usage_percent")
        assert not is_counter_metric("temperature")


class TestGapFilling:
    def test_linear_fill(self, gappy_series):
        timestamps, values = gappy_series
        _, filled = fill_gaps(timestamps, values, method="linear")
        assert not np.any(np.isnan(filled))

    def test_ffill(self):
        ts = np.array([0, 1, 2, 3, 4], dtype=np.float64)
        vals = np.array([1.0, np.nan, np.nan, 4.0, 5.0])
        _, filled = fill_gaps(ts, vals, method="ffill")
        assert filled[1] == 1.0
        assert filled[2] == 1.0

    def test_bfill(self):
        ts = np.array([0, 1, 2, 3, 4], dtype=np.float64)
        vals = np.array([1.0, np.nan, np.nan, 4.0, 5.0])
        _, filled = fill_gaps(ts, vals, method="bfill")
        assert filled[1] == 4.0
        assert filled[2] == 4.0

    def test_zero_fill(self):
        ts = np.array([0, 1, 2], dtype=np.float64)
        vals = np.array([1.0, np.nan, 3.0])
        _, filled = fill_gaps(ts, vals, method="zero")
        assert filled[1] == 0.0

    def test_detect_gaps(self):
        ts = np.array([0, 60, 120, 500, 560], dtype=np.float64)
        gaps = detect_gaps(ts)
        assert len(gaps) == 1
        assert gaps[0][0] == 2  # Gap after index 2
        assert gaps[0][2] == 380.0  # Gap duration


class TestOutliers:
    def test_detect_iqr(self):
        rng = np.random.RandomState(42)
        values = rng.normal(50, 5, 100)
        values[50] = 150  # Obvious outlier
        mask = detect_outliers_iqr(values)
        assert mask[50]

    def test_detect_zscore(self):
        values = np.array([50, 51, 49, 52, 48, 150, 50, 51], dtype=np.float64)
        mask = detect_outliers_zscore(values, threshold=2.0)
        assert mask[5]  # 150 is an outlier

    def test_clip_outliers(self):
        values = np.array([1, 2, 3, 100, 2, 3, -50], dtype=np.float64)
        clipped = clip_outliers(values, lower_percentile=5, upper_percentile=95)
        assert np.max(clipped) < 100
        assert np.min(clipped) > -50

    def test_replace_outliers_with_median(self):
        values = np.array([50, 51, 49, 52, 48, 150, 50, 51], dtype=np.float64)
        result = replace_outliers(values, method="zscore", replacement="median", threshold=2.0)
        assert result[5] != 150


class TestResampling:
    def test_resample_uniform(self):
        # Irregular timestamps
        ts = np.array([0, 57, 123, 178, 245, 301, 362], dtype=np.float64)
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        new_ts, new_vals = resample_uniform(ts, vals, interval=60.0)
        assert len(new_ts) == len(new_vals)
        # Should be approximately 6 bins for 362s range at 60s intervals
        assert len(new_ts) >= 5

    def test_downsample(self):
        values = np.arange(100, dtype=np.float64)
        result = downsample(values, factor=10)
        assert len(result) == 10
        # Mean of first 10 values should be 4.5
        assert abs(result[0] - 4.5) < 0.01


class TestDecomposition:
    def test_seasonal_decompose(self):
        n = 200
        period = 24
        seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / period)
        trend = np.linspace(0, 10, n)
        noise = np.random.RandomState(42).normal(0, 0.5, n)
        values = trend + seasonal + noise

        result = seasonal_decompose(values, period=period)
        assert result.period == period
        assert len(result.trend) == n
        assert len(result.seasonal) == n
        assert len(result.residual) == n

    def test_estimate_period(self):
        n = 500
        true_period = 50
        # Add noise to make it more realistic (pure sine autocorrelation is pathological)
        rng = np.random.RandomState(42)
        values = 10 * np.sin(2 * np.pi * np.arange(n) / true_period) + rng.normal(0, 1, n)
        estimated = estimate_period(values, min_period=10, max_period=100)
        assert abs(estimated - true_period) <= 5


class TestNormalization:
    def test_z_normalize(self):
        values = np.array([10, 20, 30, 40, 50], dtype=np.float64)
        normalized, mean, std = z_normalize(values)
        assert abs(np.mean(normalized)) < 1e-10
        assert abs(np.std(normalized) - 1.0) < 1e-10

    def test_minmax_normalize(self):
        values = np.array([10, 20, 30, 40, 50], dtype=np.float64)
        normalized, vmin, vmax = minmax_normalize(values)
        assert abs(normalized[0]) < 1e-10
        assert abs(normalized[-1] - 1.0) < 1e-10

    def test_robust_normalize(self):
        values = np.array([10, 20, 30, 40, 50], dtype=np.float64)
        normalized, median, iqr = robust_normalize(values)
        assert abs(normalized[2]) < 0.01  # Median should be ~0


class TestPreprocessPipeline:
    def test_default_pipeline(self, normal_series):
        timestamps, values = normal_series
        pipeline = PreprocessPipeline()
        result = pipeline.process(timestamps, values)
        assert len(result.values) == len(values)
        assert "fill_gaps" in result.steps_applied

    def test_pipeline_with_normalization(self, normal_series):
        timestamps, values = normal_series
        config = PreprocessConfig(normalize=True)
        pipeline = PreprocessPipeline(config)
        result = pipeline.process(timestamps, values)
        assert "normalize" in result.steps_applied
        assert "normalize_mean" in result.metadata
