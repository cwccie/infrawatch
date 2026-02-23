"""Shared test fixtures for InfraWatch test suite."""

import numpy as np
import pytest
import time


@pytest.fixture
def normal_series():
    """Generate a normal time series (no anomalies)."""
    rng = np.random.RandomState(42)
    n = 500
    timestamps = np.linspace(0, n * 60, n)
    values = 50.0 + 10.0 * np.sin(2 * np.pi * np.arange(n) / 100) + rng.normal(0, 2, n)
    return timestamps, values


@pytest.fixture
def anomalous_series():
    """Generate a time series with known anomalies."""
    rng = np.random.RandomState(42)
    n = 500
    timestamps = np.linspace(0, n * 60, n)
    values = 50.0 + rng.normal(0, 3, n)
    # Inject anomalies at known positions
    anomaly_indices = [100, 200, 300, 400]
    for idx in anomaly_indices:
        values[idx] += 30.0  # Clear spike
    return timestamps, values, anomaly_indices


@pytest.fixture
def counter_series():
    """Generate a monotonically increasing counter series with a wrap."""
    n = 100
    timestamps = np.linspace(0, n * 60, n)
    values = np.cumsum(np.random.RandomState(42).uniform(100, 200, n))
    # Inject a wrap at position 50
    values[50:] -= values[50] - 100  # Reset to near-zero
    values[50:] += np.cumsum(np.random.RandomState(43).uniform(100, 200, 50))
    return timestamps, values


@pytest.fixture
def gappy_series():
    """Generate a time series with NaN gaps."""
    n = 200
    timestamps = np.linspace(0, n * 60, n)
    values = 50.0 + np.sin(np.linspace(0, 4 * np.pi, n)) * 10
    # Insert gaps
    values[30:35] = np.nan
    values[80:90] = np.nan
    values[150] = np.nan
    return timestamps, values


@pytest.fixture
def sample_csv_path(tmp_path):
    """Create a temporary CSV file with sample metrics."""
    csv_path = tmp_path / "test_metrics.csv"
    csv_path.write_text(
        "name,value,timestamp,host\n"
        "cpu,50.0,1000,web-01\n"
        "cpu,52.0,1060,web-01\n"
        "cpu,48.0,1120,web-01\n"
        "cpu,51.0,1180,web-01\n"
        "cpu,55.0,1240,web-01\n"
        "memory,60.0,1000,web-01\n"
        "memory,61.0,1060,web-01\n"
        "memory,62.0,1120,web-01\n"
    )
    return csv_path


@pytest.fixture
def sample_json_path(tmp_path):
    """Create a temporary JSON file with sample metrics."""
    import json
    json_path = tmp_path / "test_metrics.json"
    data = {
        "metrics": [
            {"name": "cpu", "value": 50.0, "timestamp": 1000, "labels": {"host": "web-01"}},
            {"name": "cpu", "value": 52.0, "timestamp": 1060, "labels": {"host": "web-01"}},
            {"name": "cpu", "value": 48.0, "timestamp": 1120, "labels": {"host": "web-01"}},
            {"name": "memory", "value": 60.0, "timestamp": 1000, "labels": {"host": "web-01"}},
        ]
    }
    json_path.write_text(json.dumps(data))
    return json_path
