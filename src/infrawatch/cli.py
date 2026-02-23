"""InfraWatch CLI.

Commands: collect, detect, forecast, alert, dashboard, demo
"""

from __future__ import annotations

import json
import sys
import time

import click
import numpy as np


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """InfraWatch — Production anomaly detection for infrastructure metrics."""
    pass


@cli.command()
@click.argument("source", type=click.Path(exists=True))
@click.option("--format", "fmt", type=click.Choice(["csv", "json", "auto"]), default="auto")
def collect(source: str, fmt: str):
    """Collect metrics from a file source."""
    from infrawatch.collect.file_ingest import FileIngestor

    ingestor = FileIngestor()
    if fmt == "auto":
        batch = ingestor.ingest(source)
    elif fmt == "csv":
        batch = ingestor.ingest_csv(source)
    else:
        batch = ingestor.ingest_json(source)

    click.echo(f"Collected {len(batch)} metrics from {source}")
    for name in sorted(batch.unique_names()):
        points = batch.filter_by_name(name)
        click.echo(f"  {name}: {len(points)} points")


@cli.command()
@click.argument("source", type=click.Path(exists=True))
@click.option("--threshold", "-t", default=3.0, help="Z-score threshold")
@click.option("--method", "-m", default="ensemble",
              type=click.Choice(["zscore", "iqr", "gesd", "stl", "ensemble"]))
@click.option("--output", "-o", type=click.Path(), help="Output JSON file")
def detect(source: str, threshold: float, method: str, output: str):
    """Run anomaly detection on metric data."""
    from infrawatch.collect.file_ingest import FileIngestor
    from infrawatch.detect.pipeline import DetectionPipeline
    from infrawatch.models.statistical import ZScoreDetector, IQRDetector, GESDDetector, STLDetector
    from infrawatch.models.ensemble import EnsembleDetector

    ingestor = FileIngestor()
    batch = ingestor.ingest(source)

    # Build detector
    if method == "zscore":
        detectors = [ZScoreDetector(threshold=threshold)]
    elif method == "iqr":
        detectors = [IQRDetector(factor=1.5)]
    elif method == "gesd":
        detectors = [GESDDetector()]
    elif method == "stl":
        detectors = [STLDetector()]
    else:
        detectors = [ZScoreDetector(threshold=threshold), IQRDetector(factor=1.5)]

    pipeline = DetectionPipeline(detectors=detectors)

    all_anomalies = []
    for name in sorted(batch.unique_names()):
        points = batch.filter_by_name(name)
        timestamps = np.array([m.timestamp for m in points])
        values = np.array([m.value for m in points])

        result = pipeline.run(timestamps, values, metric_name=name)

        if result.anomalies:
            click.echo(
                f"  {name}: {result.anomaly_count} anomalies "
                f"(max severity: {result.max_severity.label})"
            )
            all_anomalies.extend(result.anomalies)
        else:
            click.echo(f"  {name}: no anomalies detected")

    click.echo(f"\nTotal: {len(all_anomalies)} anomalies detected")

    if output:
        with open(output, "w") as f:
            json.dump([a.to_dict() for a in all_anomalies], f, indent=2)
        click.echo(f"Results written to {output}")


@cli.command()
@click.argument("source", type=click.Path(exists=True))
@click.option("--horizon", "-h", default=168, help="Forecast horizon (number of points)")
@click.option("--capacity", "-c", type=float, help="Capacity threshold")
def forecast(source: str, horizon: int, capacity: float):
    """Generate capacity forecasts for metric data."""
    from infrawatch.collect.file_ingest import FileIngestor
    from infrawatch.forecast.engine import ForecastEngine

    ingestor = FileIngestor()
    batch = ingestor.ingest(source)
    engine = ForecastEngine(capacity_threshold=capacity)

    for name in sorted(batch.unique_names()):
        points = batch.filter_by_name(name)
        timestamps = np.array([m.timestamp for m in points])
        values = np.array([m.value for m in points])

        result = engine.forecast(timestamps, values, horizon=horizon)
        summary = engine.trend_summary(timestamps, values)

        click.echo(f"\n{name}:")
        click.echo(f"  Trend: {summary.get('direction', 'unknown')} "
                    f"(slope={summary.get('slope', 0):.4f}, R²={summary.get('r_squared', 0):.3f})")
        click.echo(f"  Current: {summary.get('current_value', 0):.2f}")

        if result.days_to_exhaustion is not None:
            click.echo(f"  Days to exhaustion: {result.days_to_exhaustion:.1f}")
        if result.sla_risk_score > 0:
            click.echo(f"  SLA risk: {result.sla_risk_score:.2%}")


@cli.command()
def alert():
    """Show current alert status."""
    click.echo("Alert engine status:")
    click.echo("  No active alerts (start the API server to track alerts)")
    click.echo("  Use 'infrawatch dashboard' to see alerts in the web UI")


@cli.command()
@click.option("--host", default="0.0.0.0", help="Bind address")
@click.option("--port", "-p", default=8080, help="Port")
@click.option("--debug/--no-debug", default=False)
def dashboard(host: str, port: int, debug: bool):
    """Start the web dashboard and API server."""
    from infrawatch.dashboard.server import create_dashboard_app

    app = create_dashboard_app()
    click.echo(f"InfraWatch Dashboard: http://{host}:{port}")
    click.echo("Press Ctrl+C to stop")
    app.run(host=host, port=port, debug=debug)


@cli.command()
@click.option("--points", "-n", default=500, help="Number of data points")
@click.option("--anomalies", "-a", default=5, help="Number of injected anomalies")
def demo(points: int, anomalies: int):
    """Run a demo with synthetic data showing anomaly detection."""
    from infrawatch.detect.pipeline import DetectionPipeline
    from infrawatch.models.statistical import ZScoreDetector, IQRDetector
    from infrawatch.forecast.engine import ForecastEngine

    click.echo("=" * 60)
    click.echo("  InfraWatch Demo — Anomaly Detection Pipeline")
    click.echo("=" * 60)

    # Generate synthetic CPU usage with daily seasonality
    rng = np.random.RandomState(42)
    t = np.linspace(0, 7 * 86400, points)  # 7 days
    seasonal = 20 * np.sin(2 * np.pi * t / 86400) + 50  # Daily pattern
    trend = 0.001 * np.arange(points)  # Slight upward trend
    noise = rng.normal(0, 3, points)
    values = seasonal + trend + noise

    # Inject anomalies
    anomaly_positions = rng.choice(range(50, max(51, points - 50)), size=min(anomalies, max(1, points - 100)), replace=False)
    for pos in anomaly_positions:
        values[pos] += rng.choice([-1, 1]) * rng.uniform(40, 60)

    click.echo(f"\nGenerated {points} points of synthetic CPU data (7 days)")
    click.echo(f"Injected {anomalies} anomalies")

    # Run detection (disable outlier handling to preserve injected anomalies for demo)
    from infrawatch.preprocess.pipeline import PreprocessConfig
    from infrawatch.models.ensemble import EnsembleConfig
    preprocess_config = PreprocessConfig(handle_outliers=False)
    pipeline = DetectionPipeline(
        detectors=[ZScoreDetector(threshold=3.0), IQRDetector(factor=1.5)],
        preprocess_config=preprocess_config,
        ensemble_config=EnsembleConfig(strategy="any"),
    )
    result = pipeline.run(t, values, metric_name="cpu_usage_percent")

    click.echo(f"\n--- Detection Results ---")
    click.echo(f"Total points analyzed: {result.total_points}")
    click.echo(f"Anomalies detected:   {result.anomaly_count}")
    click.echo(f"Max severity:         {result.max_severity.label}")
    click.echo(f"Detection time:       {result.detection_time_ms:.1f} ms")

    if result.anomalies:
        click.echo(f"\nTop anomalies:")
        sorted_anomalies = sorted(result.anomalies, key=lambda a: a.score, reverse=True)[:10]
        for a in sorted_anomalies:
            day = a.timestamp / 86400
            click.echo(
                f"  Day {day:.1f} | Value: {a.value:7.2f} | "
                f"Score: {a.score:.3f} | Severity: {a.severity.label}"
            )

    # Run forecast
    click.echo(f"\n--- Capacity Forecast ---")
    engine = ForecastEngine(capacity_threshold=95.0)
    fc = engine.forecast(t, values, horizon=168)
    summary = engine.trend_summary(t, values)

    click.echo(f"Trend: {summary.get('direction', '?')} "
               f"(slope={summary.get('slope', 0):.6f}/s)")
    click.echo(f"Current level: {values[-1]:.1f}%")
    if fc.days_to_exhaustion is not None:
        click.echo(f"Estimated days to 95% capacity: {fc.days_to_exhaustion:.1f}")
    else:
        click.echo("No capacity exhaustion projected")

    click.echo(f"\n{'=' * 60}")
    click.echo("  Demo complete. Run 'infrawatch dashboard' for the web UI.")
    click.echo(f"{'=' * 60}")


if __name__ == "__main__":
    cli()
