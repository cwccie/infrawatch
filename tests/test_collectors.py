"""Tests for metric collectors."""

from infrawatch.collect.prometheus import PrometheusCollector, parse_prometheus_text
from infrawatch.collect.snmp import SNMPPoller, SNMPTarget, unwrap_counter
from infrawatch.collect.statsd import StatsDReceiver, parse_statsd_line, parse_statsd_batch
from infrawatch.collect.file_ingest import FileIngestor


class TestPrometheusCollector:
    def test_parse_simple_metric(self):
        text = 'cpu_usage{host="web-01"} 42.5'
        batch = parse_prometheus_text(text)
        assert len(batch) == 1
        assert batch.metrics[0].name == "cpu_usage"
        assert batch.metrics[0].value == 42.5
        assert batch.metrics[0].labels["host"] == "web-01"

    def test_parse_multiple_metrics(self):
        text = (
            "# HELP cpu_usage CPU usage\n"
            "# TYPE cpu_usage gauge\n"
            'cpu_usage{host="web-01"} 42.5\n'
            'cpu_usage{host="web-02"} 55.0\n'
            "memory_bytes 1073741824\n"
        )
        batch = parse_prometheus_text(text)
        assert len(batch) == 3

    def test_parse_with_timestamp(self):
        text = "metric_name 1.5 1700000000000"
        batch = parse_prometheus_text(text)
        assert len(batch) == 1
        assert batch.metrics[0].timestamp == 1700000000.0

    def test_parse_special_values(self):
        text = "metric_inf +Inf\nmetric_nan NaN\n"
        batch = parse_prometheus_text(text)
        assert len(batch) == 2
        import math
        assert math.isinf(batch.metrics[0].value)
        assert math.isnan(batch.metrics[1].value)

    def test_collector_scrape_text(self):
        collector = PrometheusCollector(extra_labels={"env": "test"})
        text = 'cpu_usage{host="web-01"} 42.5'
        batch = collector.scrape_text(text)
        assert len(batch) == 1
        assert batch.metrics[0].labels["env"] == "test"
        assert batch.metrics[0].labels["host"] == "web-01"


class TestSNMPPoller:
    def test_unwrap_counter_no_wrap(self):
        delta = unwrap_counter(1000, 500, bits=64)
        assert delta == 500

    def test_unwrap_counter_32bit_wrap(self):
        delta = unwrap_counter(100, 2**32 - 100, bits=32)
        assert delta == 200

    def test_unwrap_counter_64bit_wrap(self):
        delta = unwrap_counter(100, 2**64 - 100, bits=64)
        assert delta == 200

    def test_poll_simulated(self):
        target = SNMPTarget(host="192.168.1.1", oids=[], labels={"site": "dc1"})
        poller = SNMPPoller()
        batch = poller.poll_simulated(target, {
            "1.3.6.1.4.1.2021.11.9.0": 25.0,
            "1.3.6.1.4.1.2021.4.6.0": 8388608.0,
        })
        assert len(batch) >= 2

    def test_poll_simulated_counter_delta(self):
        target = SNMPTarget(host="10.0.0.1")
        poller = SNMPPoller()
        # First poll establishes baseline
        poller.poll_simulated(target, {"1.3.6.1.2.1.2.2.1.10.1": 1000.0})
        # Second poll computes delta
        batch = poller.poll_simulated(target, {"1.3.6.1.2.1.2.2.1.10.1": 2000.0})
        delta_metrics = [m for m in batch.metrics if "delta" in m.name]
        assert len(delta_metrics) == 1
        assert delta_metrics[0].value == 1000.0


class TestStatsDReceiver:
    def test_parse_counter(self):
        m = parse_statsd_line("page.views:1|c")
        assert m is not None
        assert m.name == "page.views"
        assert m.value == 1.0
        assert m.labels["type"] == "counter"

    def test_parse_gauge(self):
        m = parse_statsd_line("cpu.usage:42.5|g")
        assert m is not None
        assert m.value == 42.5
        assert m.labels["type"] == "gauge"

    def test_parse_timer(self):
        m = parse_statsd_line("response.time:320|ms")
        assert m is not None
        assert m.value == 320.0

    def test_parse_with_tags(self):
        m = parse_statsd_line("request.count:1|c|#host:web-01,region:us")
        assert m is not None
        assert m.labels["host"] == "web-01"
        assert m.labels["region"] == "us"

    def test_parse_with_sample_rate(self):
        m = parse_statsd_line("request.count:1|c|@0.5")
        assert m is not None
        assert m.value == 2.0  # 1 / 0.5

    def test_parse_batch(self):
        data = "cpu:42|g\nmemory:60|g\nrequests:100|c"
        batch = parse_statsd_batch(data)
        assert len(batch) == 3

    def test_receiver_text(self):
        receiver = StatsDReceiver()
        batch = receiver.receive_text("cpu:42|g\nmemory:60|g")
        assert len(batch) == 2


class TestFileIngestor:
    def test_ingest_csv(self, sample_csv_path):
        ingestor = FileIngestor()
        batch = ingestor.ingest_csv(sample_csv_path)
        assert len(batch) == 8
        assert "cpu" in batch.unique_names()
        assert "memory" in batch.unique_names()

    def test_ingest_json(self, sample_json_path):
        ingestor = FileIngestor()
        batch = ingestor.ingest_json(sample_json_path)
        assert len(batch) == 4

    def test_ingest_auto_detect_csv(self, sample_csv_path):
        ingestor = FileIngestor()
        batch = ingestor.ingest(sample_csv_path)
        assert len(batch) == 8

    def test_ingest_auto_detect_json(self, sample_json_path):
        ingestor = FileIngestor()
        batch = ingestor.ingest(sample_json_path)
        assert len(batch) == 4

    def test_ingest_with_default_labels(self, sample_csv_path):
        ingestor = FileIngestor(default_labels={"env": "test"})
        batch = ingestor.ingest_csv(sample_csv_path)
        assert all(m.labels.get("env") == "test" for m in batch.metrics)
