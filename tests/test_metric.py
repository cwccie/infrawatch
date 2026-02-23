"""Tests for the unified metric format."""

import time
from infrawatch.collect.metric import Metric, MetricBatch


class TestMetric:
    def test_create_metric(self):
        m = Metric(name="cpu", value=50.0, timestamp=1000.0)
        assert m.name == "cpu"
        assert m.value == 50.0
        assert m.timestamp == 1000.0

    def test_metric_default_timestamp(self):
        before = time.time()
        m = Metric(name="cpu", value=50.0)
        after = time.time()
        assert before <= m.timestamp <= after

    def test_metric_with_labels(self):
        m = Metric(name="cpu", value=50.0, labels={"host": "web-01"})
        m2 = m.with_labels(region="us-east-1")
        assert m2.labels["host"] == "web-01"
        assert m2.labels["region"] == "us-east-1"
        # Original unchanged
        assert "region" not in m.labels

    def test_label_fingerprint(self):
        m1 = Metric(name="cpu", value=50.0, labels={"host": "a", "region": "b"})
        m2 = Metric(name="cpu", value=60.0, labels={"region": "b", "host": "a"})
        assert m1.label_fingerprint() == m2.label_fingerprint()

    def test_metric_serialization(self):
        m = Metric(name="cpu", value=50.0, timestamp=1000.0, labels={"host": "web-01"})
        d = m.to_dict()
        m2 = Metric.from_dict(d)
        assert m2.name == m.name
        assert m2.value == m.value
        assert m2.timestamp == m.timestamp
        assert m2.labels == m.labels


class TestMetricBatch:
    def test_batch_operations(self):
        batch = MetricBatch()
        batch.add(Metric(name="cpu", value=50.0))
        batch.add(Metric(name="cpu", value=52.0))
        batch.add(Metric(name="memory", value=60.0))
        assert len(batch) == 3

    def test_batch_filter_by_name(self):
        batch = MetricBatch()
        batch.add(Metric(name="cpu", value=50.0))
        batch.add(Metric(name="memory", value=60.0))
        cpu_metrics = batch.filter_by_name("cpu")
        assert len(cpu_metrics) == 1
        assert cpu_metrics[0].name == "cpu"

    def test_batch_filter_by_label(self):
        batch = MetricBatch()
        batch.add(Metric(name="cpu", value=50.0, labels={"host": "web-01"}))
        batch.add(Metric(name="cpu", value=52.0, labels={"host": "web-02"}))
        result = batch.filter_by_label("host", "web-01")
        assert len(result) == 1

    def test_batch_group_by_name(self):
        batch = MetricBatch()
        batch.add(Metric(name="cpu", value=50.0))
        batch.add(Metric(name="cpu", value=52.0))
        batch.add(Metric(name="memory", value=60.0))
        groups = batch.group_by_name()
        assert len(groups) == 2
        assert len(groups["cpu"]) == 2

    def test_batch_unique_names(self):
        batch = MetricBatch()
        batch.add(Metric(name="cpu", value=50.0))
        batch.add(Metric(name="cpu", value=52.0))
        batch.add(Metric(name="memory", value=60.0))
        assert batch.unique_names() == {"cpu", "memory"}
