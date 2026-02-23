"""Flask REST API for InfraWatch.

Provides endpoints for:
  - Metric query and ingestion
  - Anomaly detection status
  - Forecasting
  - Alert management
  - Maintenance window management
  - Health checks
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from flask import Flask, jsonify, request

from infrawatch.detect.pipeline import DetectionPipeline
from infrawatch.detect.severity import Severity
from infrawatch.alert.engine import AlertEngine
from infrawatch.forecast.engine import ForecastEngine
from infrawatch.maintenance.manager import MaintenanceManager, MaintenanceWindow


def create_app(
    detection_pipeline: Optional[DetectionPipeline] = None,
    alert_engine: Optional[AlertEngine] = None,
    forecast_engine: Optional[ForecastEngine] = None,
    maintenance_manager: Optional[MaintenanceManager] = None,
) -> Flask:
    """Create and configure the Flask application.

    Args:
        detection_pipeline: Pre-configured detection pipeline.
        alert_engine: Pre-configured alert engine.
        forecast_engine: Pre-configured forecast engine.
        maintenance_manager: Pre-configured maintenance manager.

    Returns:
        Configured Flask application.
    """
    app = Flask(__name__)

    pipeline = detection_pipeline or DetectionPipeline()
    alerts = alert_engine or AlertEngine()
    forecaster = forecast_engine or ForecastEngine()
    maintenance = maintenance_manager or MaintenanceManager()

    # In-memory metric store for demo purposes
    _metric_store: dict[str, list[dict]] = {}

    # ── Health ──────────────────────────────────────────────────────────

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "healthy",
            "timestamp": time.time(),
            "version": "0.1.0",
            "active_alerts": alerts.alert_count,
            "active_maintenance_windows": len(maintenance.active_windows()),
        })

    # ── Metrics ─────────────────────────────────────────────────────────

    @app.route("/api/v1/metrics", methods=["POST"])
    def ingest_metrics():
        """Ingest metrics via JSON body."""
        data = request.get_json(force=True)
        if not isinstance(data, list):
            data = [data]

        count = 0
        for entry in data:
            name = entry.get("name", "")
            value = entry.get("value")
            if name and value is not None:
                ts = entry.get("timestamp", time.time())
                _metric_store.setdefault(name, []).append({
                    "value": float(value),
                    "timestamp": float(ts),
                    "labels": entry.get("labels", {}),
                })
                count += 1

        return jsonify({"ingested": count}), 201

    @app.route("/api/v1/metrics", methods=["GET"])
    def list_metrics():
        """List available metrics."""
        return jsonify({
            "metrics": [
                {"name": name, "points": len(points)}
                for name, points in _metric_store.items()
            ]
        })

    @app.route("/api/v1/metrics/<name>", methods=["GET"])
    def get_metric(name: str):
        """Get metric data points."""
        if name not in _metric_store:
            return jsonify({"error": "metric not found"}), 404

        limit = request.args.get("limit", 1000, type=int)
        points = _metric_store[name][-limit:]
        return jsonify({"name": name, "points": points})

    # ── Detection ───────────────────────────────────────────────────────

    @app.route("/api/v1/detect/<name>", methods=["POST"])
    def detect_anomalies(name: str):
        """Run anomaly detection on stored metric data."""
        if name not in _metric_store:
            return jsonify({"error": "metric not found"}), 404

        points = _metric_store[name]
        if len(points) < 10:
            return jsonify({"error": "insufficient data (need >= 10 points)"}), 400

        timestamps = np.array([p["timestamp"] for p in points])
        values = np.array([p["value"] for p in points])

        result = pipeline.run(timestamps, values, metric_name=name)

        # Fire alerts for detected anomalies
        for anomaly in result.anomalies:
            if not maintenance.is_suppressed(name):
                alerts.fire(
                    metric_name=name,
                    severity=anomaly.severity,
                    value=anomaly.value,
                    score=anomaly.score,
                )

        return jsonify({
            "metric": name,
            "total_points": result.total_points,
            "anomalies": [a.to_dict() for a in result.anomalies],
            "detection_time_ms": result.detection_time_ms,
            "max_severity": result.max_severity.label,
        })

    @app.route("/api/v1/anomalies", methods=["GET"])
    def list_anomalies():
        """List all active anomaly alerts."""
        active = alerts.active_alerts()
        return jsonify({
            "count": len(active),
            "alerts": [a.to_dict() for a in active],
        })

    # ── Forecast ────────────────────────────────────────────────────────

    @app.route("/api/v1/forecast/<name>", methods=["POST"])
    def forecast_metric(name: str):
        """Generate capacity forecast for a metric."""
        if name not in _metric_store:
            return jsonify({"error": "metric not found"}), 404

        points = _metric_store[name]
        if len(points) < 10:
            return jsonify({"error": "insufficient data"}), 400

        horizon = request.args.get("horizon", 168, type=int)

        timestamps = np.array([p["timestamp"] for p in points])
        values = np.array([p["value"] for p in points])

        result = forecaster.forecast(timestamps, values, horizon=horizon)
        return jsonify({
            "metric": name,
            "forecast": result.to_dict(),
            "predicted": result.predicted.tolist(),
        })

    # ── Alerts ──────────────────────────────────────────────────────────

    @app.route("/api/v1/alerts", methods=["GET"])
    def get_alerts():
        active = alerts.active_alerts()
        return jsonify({
            "count": len(active),
            "alerts": [a.to_dict() for a in active],
        })

    @app.route("/api/v1/alerts/<fingerprint>/acknowledge", methods=["POST"])
    def ack_alert(fingerprint: str):
        alert = alerts.acknowledge(fingerprint)
        if alert:
            return jsonify(alert.to_dict())
        return jsonify({"error": "alert not found"}), 404

    @app.route("/api/v1/alerts/<fingerprint>/resolve", methods=["POST"])
    def resolve_alert(fingerprint: str):
        alert = alerts.resolve(fingerprint)
        if alert:
            return jsonify(alert.to_dict())
        return jsonify({"error": "alert not found"}), 404

    @app.route("/api/v1/silence", methods=["POST"])
    def silence_alert():
        data = request.get_json(force=True)
        metric_name = data.get("metric_name", "")
        duration = data.get("duration_minutes", 60)
        fp = alerts.silence_metric(metric_name, duration_minutes=duration)
        return jsonify({"fingerprint": fp, "silenced_minutes": duration})

    # ── Maintenance ─────────────────────────────────────────────────────

    @app.route("/api/v1/maintenance", methods=["GET"])
    def list_maintenance():
        windows = maintenance.list_windows(include_past=False)
        return jsonify({
            "windows": [w.to_dict() for w in windows],
        })

    @app.route("/api/v1/maintenance", methods=["POST"])
    def create_maintenance():
        data = request.get_json(force=True)
        window = MaintenanceWindow(
            id=data.get("id", f"maint-{int(time.time())}"),
            name=data.get("name", "Maintenance"),
            start_time=data.get("start_time", time.time()),
            end_time=data.get("end_time", time.time() + 3600),
            targets=data.get("targets", []),
            recalibrate=data.get("recalibrate", True),
        )
        maintenance.add_window(window)
        return jsonify(window.to_dict()), 201

    @app.route("/api/v1/maintenance/<window_id>", methods=["DELETE"])
    def delete_maintenance(window_id: str):
        if maintenance.remove_window(window_id):
            return jsonify({"deleted": window_id})
        return jsonify({"error": "window not found"}), 404

    return app
