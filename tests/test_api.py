"""Tests for the REST API."""

import json
import time
import pytest
from infrawatch.api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestMetricEndpoints:
    def test_ingest_metrics(self, client):
        metrics = [
            {"name": "cpu", "value": 50.0, "timestamp": 1000},
            {"name": "cpu", "value": 52.0, "timestamp": 1060},
        ]
        resp = client.post("/api/v1/metrics", json=metrics)
        assert resp.status_code == 201
        assert resp.get_json()["ingested"] == 2

    def test_list_metrics(self, client):
        client.post("/api/v1/metrics", json=[
            {"name": "cpu", "value": 50.0},
        ])
        resp = client.get("/api/v1/metrics")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["metrics"]) >= 1

    def test_get_metric(self, client):
        client.post("/api/v1/metrics", json=[
            {"name": "test_cpu", "value": 50.0, "timestamp": 1000},
            {"name": "test_cpu", "value": 52.0, "timestamp": 1060},
        ])
        resp = client.get("/api/v1/metrics/test_cpu")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["name"] == "test_cpu"
        assert len(data["points"]) == 2

    def test_get_nonexistent_metric(self, client):
        resp = client.get("/api/v1/metrics/nonexistent")
        assert resp.status_code == 404


class TestDetectionEndpoints:
    def test_detect_insufficient_data(self, client):
        client.post("/api/v1/metrics", json=[
            {"name": "short", "value": 50.0},
        ])
        resp = client.post("/api/v1/detect/short")
        assert resp.status_code == 400

    def test_detect_anomalies(self, client):
        import numpy as np
        rng = np.random.RandomState(42)
        metrics = []
        for i in range(100):
            val = 50.0 + rng.normal(0, 3)
            if i == 50:
                val += 30  # Anomaly
            metrics.append({"name": "det_cpu", "value": float(val), "timestamp": float(1000 + i * 60)})
        client.post("/api/v1/metrics", json=metrics)

        resp = client.post("/api/v1/detect/det_cpu")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total_points"] == 100


class TestMaintenanceEndpoints:
    def test_create_maintenance(self, client):
        now = time.time()
        resp = client.post("/api/v1/maintenance", json={
            "name": "Test maintenance",
            "start_time": now,
            "end_time": now + 3600,
            "targets": ["cpu"],
        })
        assert resp.status_code == 201

    def test_list_maintenance(self, client):
        resp = client.get("/api/v1/maintenance")
        assert resp.status_code == 200


class TestAlertEndpoints:
    def test_list_alerts(self, client):
        resp = client.get("/api/v1/alerts")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "count" in data

    def test_silence_alert(self, client):
        resp = client.post("/api/v1/silence", json={
            "metric_name": "cpu",
            "duration_minutes": 30,
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert "fingerprint" in data
