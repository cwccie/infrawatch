"""Tests for the alerting engine."""

import time
from infrawatch.alert.engine import AlertEngine, Alert, AlertState
from infrawatch.detect.severity import Severity


class TestAlertEngine:
    def test_fire_alert(self):
        engine = AlertEngine()
        alert = engine.fire(
            metric_name="cpu",
            severity=Severity.HIGH,
            value=95.0,
            score=0.8,
        )
        assert alert.state == AlertState.FIRING
        assert alert.metric_name == "cpu"
        assert engine.alert_count == 1

    def test_deduplication(self):
        engine = AlertEngine(dedup_window=300)
        engine.fire(metric_name="cpu", severity=Severity.HIGH, value=95.0, score=0.8)
        engine.fire(metric_name="cpu", severity=Severity.HIGH, value=96.0, score=0.85)
        assert engine.alert_count == 1
        alerts = engine.active_alerts()
        assert alerts[0].count == 2
        assert alerts[0].value == 96.0

    def test_different_metrics_not_deduped(self):
        engine = AlertEngine()
        engine.fire(metric_name="cpu", severity=Severity.HIGH, value=95.0, score=0.8)
        engine.fire(metric_name="memory", severity=Severity.MEDIUM, value=85.0, score=0.6)
        assert engine.alert_count == 2

    def test_acknowledge(self):
        engine = AlertEngine()
        alert = engine.fire(metric_name="cpu", severity=Severity.HIGH, value=95.0, score=0.8)
        acked = engine.acknowledge(alert.fingerprint)
        assert acked is not None
        assert acked.state == AlertState.ACKNOWLEDGED

    def test_resolve(self):
        engine = AlertEngine()
        alert = engine.fire(metric_name="cpu", severity=Severity.HIGH, value=95.0, score=0.8)
        resolved = engine.resolve(alert.fingerprint)
        assert resolved is not None
        assert resolved.state == AlertState.RESOLVED
        assert engine.alert_count == 0

    def test_silence(self):
        engine = AlertEngine()
        fp = engine.silence_metric("cpu", duration_minutes=60)
        alert = engine.fire(metric_name="cpu", severity=Severity.HIGH, value=95.0, score=0.8)
        assert alert.state == AlertState.SILENCED

    def test_severity_escalation_on_dedup(self):
        engine = AlertEngine()
        engine.fire(metric_name="cpu", severity=Severity.MEDIUM, value=85.0, score=0.5)
        engine.fire(metric_name="cpu", severity=Severity.HIGH, value=95.0, score=0.8)
        alerts = engine.active_alerts()
        assert alerts[0].severity == Severity.HIGH

    def test_alert_serialization(self):
        engine = AlertEngine()
        alert = engine.fire(
            metric_name="cpu",
            severity=Severity.HIGH,
            value=95.0,
            score=0.8,
            labels={"host": "web-01"},
        )
        d = alert.to_dict()
        assert d["metric_name"] == "cpu"
        assert d["severity"] == "high"
        assert d["labels"]["host"] == "web-01"
