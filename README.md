<div align="center">

# InfraWatch

**Production anomaly detection for infrastructure metrics using time series foundation models**

[![CI](https://github.com/cwccie/infrawatch/actions/workflows/ci.yml/badge.svg)](https://github.com/cwccie/infrawatch/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[Quickstart](#quickstart) · [Architecture](#architecture) · [API Reference](#api-reference) · [Contributing](CONTRIBUTING.md)

</div>

---

## The Problem

Infrastructure monitoring generates thousands of metrics per second — CPU, memory, bandwidth, latency, error rates — across hundreds of hosts. Traditional threshold-based alerting fails because:

- **Static thresholds can't handle seasonality.** CPU at 85% is normal during business hours but alarming at 3 AM.
- **Every metric needs manual tuning.** Different hosts, different baselines, different thresholds. It doesn't scale.
- **Maintenance creates alert storms.** Deploying a new build saturates CPU for 10 minutes. Your on-call engineer's phone explodes with false alarms.
- **Anomalies span multiple metrics.** A memory leak causes CPU spikes, which increases latency, which increases error rates. You get 4 separate alerts instead of one root cause.

InfraWatch solves all of this.

## How It Works

InfraWatch combines **classical statistical methods**, **machine learning**, and **time series foundation models** (TSFMs) into an ensemble detection pipeline that learns what "normal" looks like for your infrastructure — automatically.

### What Are Time Series Foundation Models?

TSFMs like [Chronos-Bolt](https://github.com/amazon-science/chronos-forecasting) and [TimesFM](https://github.com/google-research/timesfm) are transformer-based models pre-trained on billions of time series data points. They understand temporal patterns — seasonality, trends, level shifts — without being trained on *your* data. Think of them as GPT for time series: zero-shot forecasting that works out of the box.

InfraWatch uses TSFMs as one signal in an ensemble, combined with battle-tested statistical methods:

| Layer | Methods | Purpose |
|-------|---------|---------|
| **Statistical** | Z-score, IQR, GESD, STL decomposition | Fast, interpretable, low-latency |
| **Machine Learning** | Isolation Forest, LOF, Autoencoder | Pattern-based, handles multivariate |
| **Foundation Model** | Chronos-Bolt / TimesFM adapter | Zero-shot, seasonality-aware |
| **Ensemble** | Consensus voting (majority/unanimous/weighted) | Reduces false positives |

The ensemble requires **majority agreement** across independent methods before firing an alert. This dramatically reduces false positives while catching real anomalies that any single method would miss.

## Zero-Config Philosophy

```bash
pip install infrawatch
infrawatch demo
```

That's it. No YAML files to write, no thresholds to tune, no training data to prepare. InfraWatch ships with sensible defaults that work for common infrastructure metrics:

- **Automatic seasonality detection** — discovers daily/weekly patterns without configuration
- **Counter unwrapping** — handles 32-bit and 64-bit SNMP counter wraps transparently
- **Gap filling** — interpolates missing data from collector outages
- **Context-aware thresholds** — automatically relaxes sensitivity during nights and weekends
- **Maintenance suppression** — silences alerts during scheduled windows, recalibrates after

Advanced users can tune everything. But you shouldn't have to.

## Quickstart

### Install

```bash
pip install infrawatch                    # Core (NumPy/SciPy only)
pip install infrawatch[ml]                # + scikit-learn models
pip install infrawatch[foundation]        # + Chronos-Bolt (requires PyTorch)
pip install infrawatch[full]              # Everything
```

### Run the Demo

```bash
infrawatch demo
```

Generates 7 days of synthetic CPU data with injected anomalies, runs the full detection pipeline, and shows results:

```
============================================================
  InfraWatch Demo — Anomaly Detection Pipeline
============================================================

Generated 500 points of synthetic CPU data (7 days)
Injected 5 anomalies

--- Detection Results ---
Total points analyzed: 500
Anomalies detected:   5
Max severity:         high
Detection time:       12.3 ms

Top anomalies:
  Day 2.1 | Value:   91.42 | Score: 0.847 | Severity: high
  Day 4.3 | Value:   14.23 | Score: 0.792 | Severity: high
  ...
```

### Start the Dashboard

```bash
infrawatch dashboard
```

Opens a real-time web dashboard at `http://localhost:8080` with:
- Live metric graphs with anomaly overlay
- Active alert list with severity indicators
- Maintenance window calendar
- System health overview

### Docker

```bash
docker compose up -d
# Dashboard at http://localhost:8080
```

### Use as a Library

```python
import numpy as np
from infrawatch.detect.pipeline import DetectionPipeline
from infrawatch.models.statistical import ZScoreDetector, IQRDetector
from infrawatch.models.ensemble import EnsembleDetector, EnsembleConfig

# Your metric data
timestamps = np.array([...])  # Unix epochs
values = np.array([...])       # Metric values

# Run detection
pipeline = DetectionPipeline()
result = pipeline.run(timestamps, values, metric_name="cpu_usage_percent")

for anomaly in result.anomalies:
    print(f"{anomaly.severity.label}: {anomaly.value:.1f} (score={anomaly.score:.3f})")
```

### Collect from Prometheus

```python
from infrawatch.collect.prometheus import PrometheusCollector

collector = PrometheusCollector(targets=["http://prometheus:9090"])
batch = collector.scrape_all()

for metric in batch:
    print(f"{metric.name}: {metric.value} {metric.labels}")
```

### Capacity Forecasting

```python
from infrawatch.forecast.engine import ForecastEngine

engine = ForecastEngine(capacity_threshold=95.0)
forecast = engine.forecast(timestamps, values, horizon=168)  # 1 week ahead

if forecast.days_to_exhaustion:
    print(f"Capacity exhaustion in {forecast.days_to_exhaustion:.0f} days")
print(f"SLA violation risk: {forecast.sla_risk_score:.1%}")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        InfraWatch Pipeline                       │
│                                                                   │
│  ┌──────────┐   ┌──────────────┐   ┌──────────┐   ┌──────────┐ │
│  │ Collect   │──▶│ Preprocess   │──▶│ Detect   │──▶│ Alert    │ │
│  │          │   │              │   │          │   │          │ │
│  │Prometheus│   │Counter unwrap│   │Z-score   │   │Dedup     │ │
│  │SNMP      │   │Gap fill     │   │IQR       │   │Group     │ │
│  │StatsD    │   │Outlier clip │   │GESD      │   │Escalate  │ │
│  │CSV/JSON  │   │Resample     │   │STL       │   │          │ │
│  │          │   │Decompose    │   │IsoForest │   │Webhook   │ │
│  │          │   │Normalize    │   │LOF       │   │Email     │ │
│  │          │   │              │   │Autoencod.│   │Slack     │ │
│  │          │   │              │   │TSFM      │   │PagerDuty │ │
│  │          │   │              │   │          │   │          │ │
│  │          │   │              │   │Ensemble  │   │          │ │
│  └──────────┘   └──────────────┘   └──────────┘   └──────────┘ │
│                                          │                       │
│                                    ┌─────┴─────┐                │
│                                    │ Context    │                │
│                                    │ Analyzer   │                │
│                                    │            │                │
│                                    │Time-of-day │                │
│                                    │Correlation │                │
│                                    │Maintenance │                │
│                                    └────────────┘                │
│                                                                   │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────────────┐   │
│  │ Forecast │   │ Maintenance  │   │ REST API + Dashboard   │   │
│  │          │   │ Manager      │   │                        │   │
│  │Trend     │   │Calendar      │   │/api/v1/metrics         │   │
│  │Seasonal  │   │Suppression   │   │/api/v1/detect          │   │
│  │Capacity  │   │Recalibrate   │   │/api/v1/forecast        │   │
│  │SLA risk  │   │              │   │/api/v1/alerts          │   │
│  └──────────┘   └──────────────┘   │/api/v1/maintenance     │   │
│                                     └────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health check |
| `POST` | `/api/v1/metrics` | Ingest metric data points |
| `GET` | `/api/v1/metrics` | List available metrics |
| `GET` | `/api/v1/metrics/<name>` | Get metric time series |
| `POST` | `/api/v1/detect/<name>` | Run anomaly detection |
| `GET` | `/api/v1/anomalies` | List active anomalies |
| `POST` | `/api/v1/forecast/<name>` | Generate capacity forecast |
| `GET` | `/api/v1/alerts` | List active alerts |
| `POST` | `/api/v1/alerts/<fp>/acknowledge` | Acknowledge an alert |
| `POST` | `/api/v1/alerts/<fp>/resolve` | Resolve an alert |
| `POST` | `/api/v1/silence` | Silence alerts for a metric |
| `GET` | `/api/v1/maintenance` | List maintenance windows |
| `POST` | `/api/v1/maintenance` | Create a maintenance window |
| `DELETE` | `/api/v1/maintenance/<id>` | Delete a maintenance window |

### CLI Commands

```bash
infrawatch collect <file>       # Ingest metrics from CSV/JSON
infrawatch detect <file>        # Run anomaly detection
infrawatch forecast <file>      # Generate capacity forecast
infrawatch alert                # Show alert status
infrawatch dashboard            # Start web dashboard
infrawatch demo                 # Run interactive demo
```

## Sample Data

The `sample_data/` directory contains 7 days of realistic infrastructure metrics:

| File | Metric | Pattern |
|------|--------|---------|
| `cpu.csv` | CPU usage (%) | Daily seasonality, slight upward trend |
| `memory.csv` | Memory usage (%) | Gradual increase (leak pattern) with GC drops |
| `bandwidth.csv` | Network bandwidth (Mbps) | Daily pattern with random bursts |
| `latency.csv` | Request latency (ms) | Log-normal with peak-hour amplification |
| `errors.csv` | Error rate (errors/min) | Low baseline with incident spikes |

## Requirements

- **Python 3.10+**
- **Core**: NumPy, SciPy, Flask, Click (installed automatically)
- **ML models** (optional): scikit-learn
- **Foundation models** (optional): PyTorch, chronos-forecasting

## License

[MIT](LICENSE) — Corey A. Wade

## Author

**Corey A. Wade** — [GitHub](https://github.com/cwccie)

Infrastructure security researcher. PhD candidate (AI + Security). CISSP. Retired CCIE.
Building tools that make infrastructure monitoring intelligent.
